"""Trainer: `python -m harness model train --out data/models/<name>`. Resumable from <out>/last.pt.

Loss: weighted L1 in the log domain plus L1 on finite differences (edges), never a metric. Pixels the input
clips (any channel at the clip level) weigh CLIP_WEIGHT: what was clipped is not in the input. The
validation set is a fixed draw from held-out clean items; the restore subset is scored with the twelve tile
metrics on input and output, so the log shows which directions the model moves and by how much."""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .. import loader_context
from ..data.tilemetrics import tile_metrics
from ..metrics import STILL_METRICS
from .data import Normaliser, PairStream, TileSource, cond_vector, decode, encode, fixed_set, real_fixed_set
from .losses import best_shift, overshoot_loss, pyramid_loss, ssim_loss, transfer_loss
from .net import CondUNet

DEFAULT_SOURCES = ("hf-kadid10k-images", "hf-tid2013", "polyhaven-hdris", "wikimedia-featured-pictures")
"""The clean sources a run trains on unless --sources says otherwise. raw-pixls-us is never among them: it is the
held-out real-capture evaluation (harness.model.heldout)."""
CLIP_WEIGHT = 0.25
GRAD_WEIGHT = 0.5
# Local-range term (ninth run): a restoration must not create luminance extrema beyond the target's own local
# range. The seventh and eighth runs rang on edges (a 1 px bright-side halo of 5 to 18 % of the step on the demo's
# slanted edge, up to 55 % at 0.002 to 0.05 of the clip, and on already sharp edges, where no deconvolution is
# due): the L1 and edge L1 pay for a halo only in proportion to its area, so a thin ring is cheap. Here every
# output value above the target's RANGE_K x RANGE_K maximum or below its minimum is charged once more, in the
# encoded domain; a perfect restoration has zero excess, so the optimum is unchanged.
RANGE_WEIGHT = 1.0
RANGE_K = 3
# Flat-texture term (a twelfth-run lever, off until --flat-weight names a weight): the tenth run put 2.4 to 3 % of
# relative texture into the demo's flat bright patches where the camera has 0.2 to 0.4 %, and the clean targets'
# halo proxy (ISP outputs at 2 to 7 times the linear references' local excursion) says where it learnt that. Where
# the target is locally flat (its FLAT_K x FLAT_K standard deviation below FLAT_TAU, encoded domain) the output's
# standard deviation beyond the target's is charged; where the target has texture nothing is, so real detail is
# untouched and a perfect restoration has zero excess.
FLAT_WEIGHT = 0.0
FLAT_K = 7
FLAT_TAU = 0.01
# Texture-deficit term (a thirteenth-run lever, off until --texture-weight names a weight): the twelfth run showed
# what one-sided penalties buy. The range and flat terms charge excess and never deficit, so where the model cannot
# tell the phase of fine detail from a noisy input the cheapest answer is to smooth it: at step 8000 its halo was at
# the camera's and its texture below it. Where the target has texture (its FLAT_K x FLAT_K standard deviation at or
# above FLAT_TAU) the output's standard deviation short of the target's is charged; where the target is flat nothing
# is. The two std terms split the plane between them, excess on flat targets and deficit on textured ones, each where
# it means what it says, and a perfect restoration has zero of both.
TEXTURE_WEIGHT = 0.0
# MS-SSIM term (off until --ssim-weight names a weight; losses.py says why it stays off): the thirteenth run smoothed
# what its real inputs carry (RawNIND MTF50 0.292 cycles per pixel against the capture's 0.387), and SSIM's
# contrast-structure term was the first lever tried. On the display rendering the fine bands' contrast sits under
# its C2, and no weight moved v13's best fine-band gain past 0.9.
SSIM_WEIGHT = 0.0
# Coherent-transfer term (off until --transfer-weight names a weight; losses.transfer_loss): per radial band of the
# clipped linear luminance, 1 - Re<P Y*> / <|Y|^2>, the quantity mtf50_cpp and texture_hf_fidelity read, held to 0
# from both sides.
TRANSFER_WEIGHT = 0.0
# Edge-overshoot and residual-pyramid terms (sixteenth run; off until --overshoot-weight / --pyramid-weight name a
# weight; losses.py says why): edge_overshoot as a share of the step over the windows that hold an edge, and the L1
# of the residual's block means at 2 to 16 px, where banding, shading and blotches keep their size.
OVERSHOOT_WEIGHT = 0.0
PYRAMID_WEIGHT = 0.0
WARMUP = 500
LR_FLOOR = 0.05      # the cosine's end as a share of the peak rate
EMA_DECAY = 0.999
CLIP_LOG = float(encode(np.array([0.995]))[0])


class StreamDataset(IterableDataset):
    def __init__(self, stream: PairStream):
        self.stream = stream

    def __iter__(self):
        return iter(self.stream)


def range_excess(pred: torch.Tensor, y: torch.Tensor, k: int = RANGE_K) -> torch.Tensor:
    """How far each output value lies outside the target's k x k local range (zero inside it)."""
    hi = torch.nn.functional.max_pool2d(y, k, stride=1, padding=k // 2)
    lo = -torch.nn.functional.max_pool2d(-y, k, stride=1, padding=k // 2)
    return torch.relu(pred - hi) + torch.relu(lo - pred)


def local_std(t: torch.Tensor, k: int) -> torch.Tensor:
    """Standard deviation over k x k windows, same size as `t` (border windows use the pixels they have). Computed
    in float64: E[t^2] - E[t]^2 cancels to noise of the order of the camera's own texture in float32 at the encoded
    domain's scale."""
    def pool(a):
        return torch.nn.functional.avg_pool2d(a, k, stride=1, padding=k // 2, count_include_pad=False)
    t64 = t.double()
    m = pool(t64)
    return (pool(t64 * t64) - m * m).clamp_min(0.0).add(1e-18).sqrt().to(t.dtype)


def std_mismatch(pred: torch.Tensor, y: torch.Tensor, k: int = FLAT_K,
                 tau: float = FLAT_TAU) -> tuple[torch.Tensor, torch.Tensor]:
    """(flat excess, texture deficit): the k x k local standard deviation of the output beyond the target's on pixels
    whose target standard deviation is below `tau`, and short of the target's on the other pixels; zero elsewhere. No
    pixel is in both."""
    sy, sp = local_std(y, k), local_std(pred, k)
    flat = (sy < tau).to(pred.dtype)
    return torch.relu(sp - sy) * flat, torch.relu(sy - sp) * (1.0 - flat)


def flat_excess(pred: torch.Tensor, y: torch.Tensor, k: int = FLAT_K, tau: float = FLAT_TAU) -> torch.Tensor:
    """Output texture beyond the target's where the target is flat (std_mismatch's first half)."""
    return std_mismatch(pred, y, k, tau)[0]


def texture_deficit(pred: torch.Tensor, y: torch.Tensor, k: int = FLAT_K, tau: float = FLAT_TAU) -> torch.Tensor:
    """Output texture short of the target's where the target has texture (std_mismatch's second half)."""
    return std_mismatch(pred, y, k, tau)[1]


def pair_loss(pred: torch.Tensor, y: torch.Tensor, x: torch.Tensor, range_weight: float = RANGE_WEIGHT,
              range_k: int = RANGE_K, flat_weight: float = FLAT_WEIGHT, flat_k: int = FLAT_K,
              texture_weight: float = TEXTURE_WEIGHT, ssim_weight: float = SSIM_WEIGHT,
              transfer_weight: float = TRANSFER_WEIGHT, overshoot_weight: float = OVERSHOOT_WEIGHT,
              pyramid_weight: float = PYRAMID_WEIGHT) -> torch.Tensor:
    """The weights and windows of the range, flat-texture, texture-deficit, MS-SSIM, coherent-transfer, overshoot
    and pyramid terms are run arguments (--range-weight, --range-k, --flat-weight, --flat-k, --texture-weight,
    --ssim-weight, --transfer-weight, --overshoot-weight, --pyramid-weight; the two std terms share the window),
    recorded in the checkpoint's `train` record, so a run
    that varies them is identifiable from its checkpoint."""
    w = torch.where((x >= CLIP_LOG).any(dim=1, keepdim=True), CLIP_WEIGHT, 1.0)
    l1 = (w * (pred - y).abs()).mean()
    gx = (pred[..., :, 1:] - pred[..., :, :-1]) - (y[..., :, 1:] - y[..., :, :-1])
    gy = (pred[..., 1:, :] - pred[..., :-1, :]) - (y[..., 1:, :] - y[..., :-1, :])
    rng = (w * range_excess(pred, y, range_k)).mean() if range_weight else pred.new_zeros(())
    flat = tex = pred.new_zeros(())
    if flat_weight or texture_weight:
        excess, deficit = std_mismatch(pred, y, flat_k)
        flat, tex = (w * excess).mean(), (w * deficit).mean()
    ssim = ssim_loss(pred, y) if ssim_weight else pred.new_zeros(())
    tr = transfer_loss(pred, y) if transfer_weight else pred.new_zeros(())
    ov = overshoot_loss(pred, y) if overshoot_weight else pred.new_zeros(())
    pyr = pyramid_loss(pred, y, w) if pyramid_weight else pred.new_zeros(())
    return (l1 + GRAD_WEIGHT * 0.5 * (gx.abs().mean() + gy.abs().mean()) + range_weight * rng + flat_weight * flat
            + texture_weight * tex + ssim_weight * ssim + transfer_weight * tr + overshoot_weight * ov
            + pyramid_weight * pyr)


def lr_at(step: int, steps: int, lr: float, spent: float = 0.0, warmup: int = WARMUP,
          floor: float = LR_FLOOR) -> float:
    """Linear warmup over `warmup` steps, then a cosine to `floor` x `lr` over whichever budget runs out first: the
    steps, or `spent`, the share of the wall-clock budget used (0 without one). A run that its --minutes stop
    therefore still anneals, whatever its size's speed."""
    if step < warmup:
        return lr * (step + 1) / warmup
    t = min(1.0, max((step - warmup) / max(1, steps - warmup), spent))
    return lr * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t)))


def to_batch(samples: list[dict], norm: Normaliser, device) -> tuple:
    x = torch.from_numpy(np.stack([encode(s["x"]) for s in samples])).permute(0, 3, 1, 2).to(device)
    y = torch.from_numpy(np.stack([encode(s["y"]) for s in samples])).permute(0, 3, 1, 2).to(device)
    c = torch.from_numpy(np.stack([cond_vector(norm, s) for s in samples])).to(device)
    return x, y, c


@torch.no_grad()
def validate(model: torch.nn.Module, val: list[dict], norm: Normaliser, device, batch: int = 8,
             range_weight: float = RANGE_WEIGHT, range_k: int = RANGE_K, flat_weight: float = FLAT_WEIGHT,
             flat_k: int = FLAT_K, texture_weight: float = TEXTURE_WEIGHT, ssim_weight: float = SSIM_WEIGHT,
             transfer_weight: float = TRANSFER_WEIGHT, overshoot_weight: float = OVERSHOOT_WEIGHT,
             pyramid_weight: float = PYRAMID_WEIGHT, shift: bool = False) -> dict:
    """The set's mean loss and its restore subset's tile-metric medians, input and output, noise read as the
    evaluation reads it (tilemetrics' "residual"). `shift` scores every sample as a real pair under train's
    --shift-real (losses.best_shift)."""
    model.eval()
    losses, rows = [], {"input": [], "output": []}
    for i in range(0, len(val), batch):
        part = val[i:i + batch]
        x, y, c = to_batch(part, norm, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(x, c)
        pred = pred.float()
        lp, ly, lx = (best_shift(pred, y, x, torch.ones(len(part), dtype=torch.bool, device=device)) if shift
                      else (pred, y, x))
        losses.append(float(pair_loss(lp, ly, lx, range_weight, range_k, flat_weight, flat_k, texture_weight,
                                      ssim_weight, transfer_weight, overshoot_weight, pyramid_weight)))
        out = decode(pred.permute(0, 2, 3, 1).cpu().numpy())
        for s, o in zip(part, out):
            if s["restore"]:
                rows["input"].append(tile_metrics(s["x"], s["y"], "residual"))
                rows["output"].append(tile_metrics(o, s["y"], "residual"))
    model.train()
    def _med(v, m):
        vals = [r[m] for r in v if np.isfinite(r[m])]
        return float(np.median(vals)) if vals else None
    med = {k: {m: _med(v, m) for m in STILL_METRICS} for k, v in rows.items()}
    return {"val_loss": float(np.mean(losses)), "n_restore": len(rows["input"]),
            "restore": {m: {"input": med["input"][m], "output": med["output"][m]} for m in STILL_METRICS}}


def save(path: Path, model, ema, opt, step: int, norm: Normaliser, cond_restore, args: dict,
         elapsed_min: float = 0.0) -> None:
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                "elapsed_min": elapsed_min,
                "config": model.config, "normaliser": norm.state(), "cond_restore": list(map(float, cond_restore)),
                "train": args}, path)


def train(out: str, sources=DEFAULT_SOURCES, steps: int = 20000, batch: int = 16, tile: int = 256, workers: int = 12,
          lr: float = 2e-4, minutes: float | None = None, seed: int = 0, base: int = 48, val_n: int = 48,
          calib_n: int = 128, log_every: int = 50, ckpt_every: int = 500, device: str | None = None,
          real=(), p_real: float = 0.0, weights: dict | None = None, range_weight: float | None = None,
          range_k: int | None = None, flat_weight: float | None = None, flat_k: int | None = None,
          texture_weight: float | None = None, p_dead_leaves: float = 0.0, decoded_store: bool = True,
          mults: tuple = (1, 2, 4, 4), blocks: int | tuple = 2, mid_blocks: int | None = None,
          grad_ckpt: bool = False, ssim_weight: float | None = None, transfer_weight: float | None = None,
          shift_real: bool = False, subpix_real: bool = False,
          noise_map: bool = False, compile_model: bool = False, warmup: int | None = None,
          lr_floor: float | None = None, target_audit: bool = False, overshoot_weight: float | None = None,
          pyramid_weight: float | None = None, resample: str = "learned", val_seed: int | None = None,
          log=print) -> dict:
    """`real` names stored real-pair sources (data.realstore) drawn with probability `p_real` per sample; `weights`
    maps a clean source to its draw weight (1 unnamed); `range_weight`, `range_k`, `flat_weight`, `flat_k` and
    `texture_weight` set the range, flat-texture and texture-deficit terms; `p_dead_leaves` is the share of rendered
    pairs drawn on procedural colour dead-leaves tiles (deadleaves.py). All are saved in the checkpoint's `train`
    record. `decoded_store` lets the loader workers read and fill the decoded clean-image store
    (data.decodedstore): the same stream at a fraction of the decode cost. `base`, `mults`, `blocks` and `mid_blocks`
    shape the network (net.CondUNet; the shape is the checkpoint's `config`); `grad_ckpt` trades a third more
    compute for a fraction of the activation memory and changes nothing else. The fifteenth run's levers, each off by
    default and recorded: `ssim_weight` (the MS-SSIM term), `transfer_weight` (the coherent-transfer term), `shift_real` (each real pair scored at its best integer
    shift within a pixel, losses.best_shift), `subpix_real` (each real pair's reference read at its measured
    sub-pixel residual on top of the integer shift, data.subpix), `noise_map` (the network reads the capture's noise
    law per pixel,
    net.noise_map), `compile_model` (torch.compile of the training forward), `warmup` and `lr_floor` (the schedule's
    warmup steps and its end as a share of `lr`), `target_audit` (the clean items the target audit keeps,
    data.targetaudit). The sixteenth run's, as off by default and as recorded: `overshoot_weight` and
    `pyramid_weight` (losses.overshoot_loss, losses.pyramid_loss), `resample` (how the network's levels change
    scale, net.CondUNet; in its `config`). `seed` draws the initial weights and the training stream;
    `val_seed` (default `seed`) the calibration and validation draws, so the runs of a seed sweep fit one
    normaliser and choose best.pt on one validation set."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    real, weights = tuple(real), dict(weights or {})
    range_weight = RANGE_WEIGHT if range_weight is None else float(range_weight)
    range_k = RANGE_K if range_k is None else int(range_k)
    if range_weight < 0 or range_k < 1 or range_k % 2 == 0:
        raise ValueError(f"range term wants weight >= 0 and an odd window >= 1, got {range_weight}, {range_k}")
    flat_weight = FLAT_WEIGHT if flat_weight is None else float(flat_weight)
    flat_k = FLAT_K if flat_k is None else int(flat_k)
    if flat_weight < 0 or flat_k < 3 or flat_k % 2 == 0:
        raise ValueError(f"flat-texture term wants weight >= 0 and an odd window >= 3, got {flat_weight}, {flat_k}")
    texture_weight = TEXTURE_WEIGHT if texture_weight is None else float(texture_weight)
    if texture_weight < 0:
        raise ValueError(f"texture-deficit term wants weight >= 0, got {texture_weight}")
    p_dead_leaves = float(p_dead_leaves)
    if not 0.0 <= p_dead_leaves <= 1.0:
        raise ValueError(f"p_dead_leaves is a probability, got {p_dead_leaves}")
    ssim_weight = SSIM_WEIGHT if ssim_weight is None else float(ssim_weight)
    if ssim_weight < 0:
        raise ValueError(f"MS-SSIM term wants weight >= 0, got {ssim_weight}")
    transfer_weight = TRANSFER_WEIGHT if transfer_weight is None else float(transfer_weight)
    if transfer_weight < 0:
        raise ValueError(f"transfer term wants weight >= 0, got {transfer_weight}")
    overshoot_weight = OVERSHOOT_WEIGHT if overshoot_weight is None else float(overshoot_weight)
    pyramid_weight = PYRAMID_WEIGHT if pyramid_weight is None else float(pyramid_weight)
    if overshoot_weight < 0 or pyramid_weight < 0:
        raise ValueError(f"overshoot and pyramid terms want weights >= 0, got {overshoot_weight}, {pyramid_weight}")
    warmup = WARMUP if warmup is None else int(warmup)
    lr_floor = LR_FLOOR if lr_floor is None else float(lr_floor)
    if warmup < 0 or not 0.0 <= lr_floor <= 1.0:
        raise ValueError(f"schedule wants warmup >= 0 and a floor in [0, 1], got {warmup}, {lr_floor}")
    loss_kw = dict(range_weight=range_weight, range_k=range_k, flat_weight=flat_weight, flat_k=flat_k,
                   texture_weight=texture_weight, ssim_weight=ssim_weight, transfer_weight=transfer_weight,
                   overshoot_weight=overshoot_weight, pyramid_weight=pyramid_weight)
    mults, blocks = tuple(int(m) for m in mults), (int(blocks) if isinstance(blocks, int) else tuple(map(int, blocks)))
    if tile % 2 ** (len(mults) - 1):
        raise ValueError(f"tile {tile} must be a multiple of {2 ** (len(mults) - 1)} for {len(mults)} levels")
    vs = seed if val_seed is None else int(val_seed)
    args = dict(sources=list(sources), steps=steps, batch=batch, tile=tile, lr=lr, seed=seed, val_seed=vs,
                base=base,
                real=list(real), p_real=p_real, weights=weights, p_dead_leaves=p_dead_leaves, mults=list(mults),
                blocks=blocks if isinstance(blocks, int) else list(blocks), mid_blocks=mid_blocks,
                shift_real=bool(shift_real), subpix_real=bool(subpix_real), noise_map=bool(noise_map), compile=bool(compile_model), warmup=warmup,
                lr_floor=lr_floor, target_audit=bool(target_audit), resample=resample, **loss_kw)
    last = out / "last.pt"
    ck = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else None
    quiet = (lambda *a, **k: None)
    if ck:
        norm, cond_restore = Normaliser.from_state(ck["normaliser"]), np.array(ck["cond_restore"], dtype=np.float32)
        log(f"resuming from step {ck['step']}")
    else:
        log(f"calibration draw: {calib_n} samples")
        calib = fixed_set(list(sources), "train", calib_n, min(tile, 128), vs + 1, log=quiet, audit=target_audit)
        norm = Normaliser.fit(np.array([[s["metrics"][m] for m in STILL_METRICS] for s in calib]))
        cond_restore = np.mean([norm(s["metrics"]) for s in calib if s["restore"]], axis=0).astype(np.float32)
    log(f"validation draw: {val_n} samples at {tile} px")
    val = fixed_set(list(sources), "val", val_n, tile, vs + 2, log=quiet, audit=target_audit)
    if target_audit:
        rep = TileSource(list(sources), "train", tile, audit=True).audit_report
        log("target audit: " + ("; ".join(f"{k} max_halo {r['max_halo']}: {r['kept']} kept, {r['over']} over, "
                                          f"{r['unaudited']} not audited" for k, r in sorted(rep.items()))
                                or "no source sets clean.max_halo"))
    if weights:
        shares = TileSource(list(sources), "train", tile, weights=weights, audit=target_audit).shares()
        log("draw shares by source: " + ", ".join(f"{k} {v:.1%}" for k, v in sorted(shares.items(), key=lambda kv: -kv[1])))
    if p_dead_leaves > 0:
        log(f"dead leaves: {p_dead_leaves:.1%} of the rendered pairs on procedural tiles")
    val_real = None
    if real and p_real > 0:
        from ..data.realstore import RealStore
        log(f"real pairs drawn with probability {p_real:.2f}: {RealStore(list(real), 'train', tile=tile, subpix=subpix_real).summary()}")
        try:
            val_real = real_fixed_set(list(real), "val", val_n, tile, vs + 3, subpix=subpix_real)
            log(f"real validation draw: {len(val_real)} tiles of the stored val scenes")
        except (ValueError, FileNotFoundError) as ex:
            log(f"no real validation set: {ex}")
    torch.manual_seed(seed)   # the initial weights (a resume loads its own)
    model = CondUNet(base=base, mults=mults, blocks=blocks, mid_blocks=mid_blocks, noise_map=noise_map,
                     resample=resample).to(dev)
    ema = copy.deepcopy(model).eval()
    model.set_grad_ckpt(grad_ckpt)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.01, fused=dev.type == "cuda")
    ema_params, model_params = list(ema.parameters()), list(model.parameters())
    step, best = 0, float("inf")
    if ck:
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    log(f"model {model.n_params() / 1e6:.2f} M params on {dev} (base {base}, mults {mults}, blocks {blocks}, mid "
        f"{mid_blocks or 'as the last level'}{', gradient checkpointing' if grad_ckpt else ''}"
        f"{', noise map' if noise_map else ''}{', compiled' if compile_model else ''}); {steps} steps, "
        f"batch {batch}, tile {tile}, {workers} workers")
    if decoded_store:
        log("decoded store: loader workers read clean images from derived/decoded and store the ones they decode")
    stream = PairStream(list(sources), tile, norm, seed + step, real=real, p_real=p_real, weights=weights or None,
                        p_dead_leaves=p_dead_leaves, decoded_store=decoded_store, audit=target_audit,
                        subpix_real=subpix_real)
    loader = DataLoader(StreamDataset(stream), batch_size=batch,
                        num_workers=workers, pin_memory=dev.type == "cuda", persistent_workers=workers > 0,
                        prefetch_factor=4 if workers > 0 else None, multiprocessing_context=loader_context(workers))
    logf = open(out / "train_log.jsonl", "a", encoding="utf-8")

    def real_val(at: int) -> None:
        """The real validation set's loss and restore medians, logged beside the synthetic ones (information: best.pt
        stays on the synthetic val loss, since the stored val scenes are the ones eval-raw --real reports on)."""
        if not val_real:
            return
        vr = validate(ema, val_real, norm, dev, shift=shift_real, **loss_kw)
        logf.write(json.dumps({"step": at, "val_real_loss": vr["val_loss"], "val_real_n": vr["n_restore"],
                               "val_real_restore": vr["restore"]}) + "\n"); logf.flush()
        log(f"  real val loss {vr['val_loss']:.4f}; real medians in -> out: " + ", ".join(
            f"{m} {r['input']:.3g}->{r['output']:.3g}" for m, r in vr["restore"].items() if r["input"] is not None))

    t0, t_log, n_log, run_loss = time.perf_counter(), time.perf_counter(), 0, 0.0
    t_prev = float(ck.get("elapsed_min", 0.0)) if ck else 0.0     # --minutes is the budget across resumes

    def spent() -> float:
        return t_prev + (time.perf_counter() - t0) / 60.0

    def frac() -> float:
        return spent() / minutes if minutes else 0.0

    fwd = torch.compile(model) if compile_model else model   # the same parameters; save() and the EMA see `model`
    model.train()
    for x, y, c, _, is_real in loader:
        if step >= steps or frac() >= 1.0:
            break
        x, y, c = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), c.to(dev, non_blocking=True)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr, frac(), warmup, lr_floor)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pred = fwd(x, c)
        pred, yt, xt = pred.float(), y, x
        if shift_real:
            pred, yt, xt = best_shift(pred, y, x, is_real.to(dev))
        loss = pair_loss(pred, yt, xt, **loss_kw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():     # one fused update over every tensor: a per-tensor loop costs ms at a large shape
            torch._foreach_lerp_(ema_params, model_params, 1.0 - EMA_DECAY)
        step += 1
        run_loss += loss.item(); n_log += 1
        if step % log_every == 0:
            now = time.perf_counter()
            rec = {"step": step, "loss": run_loss / n_log, "lr": lr_at(step, steps, lr, frac(), warmup, lr_floor),
                   "imgs_s": n_log * batch / (now - t_log), "elapsed_s": now - t0,
                   "gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            log(f"step {step} loss {rec['loss']:.4f} lr {rec['lr']:.2e} {rec['imgs_s']:.0f} img/s")
            t_log, n_log, run_loss = now, 0, 0.0
        if step % ckpt_every == 0:
            v = validate(ema, val, norm, dev, **loss_kw)
            v["step"] = step
            logf.write(json.dumps(v) + "\n"); logf.flush()
            real_val(step)
            save(last, model, ema, opt, step, norm, cond_restore, args, spent())
            if v["val_loss"] < best:
                best = v["val_loss"]; save(out / "best.pt", model, ema, opt, step, norm, cond_restore, args, spent())
            log(f"  val loss {v['val_loss']:.4f} (best {best:.4f}); restore medians in -> out: " + ", ".join(
                f"{m} {r['input']:.3g}->{r['output']:.3g}" for m, r in v["restore"].items() if r["input"] is not None))
    v = validate(ema, val, norm, dev, **loss_kw)
    v["step"] = step
    logf.write(json.dumps(v) + "\n")
    real_val(step)
    logf.close()
    save(last, model, ema, opt, step, norm, cond_restore, args, spent())
    if v["val_loss"] < best:
        save(out / "best.pt", model, ema, opt, step, norm, cond_restore, args, spent())
    log(f"done at step {step}: val loss {v['val_loss']:.4f}, {(time.perf_counter() - t0) / 60:.1f} min")
    return v
