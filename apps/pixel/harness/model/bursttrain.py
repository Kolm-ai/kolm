"""Training and evaluating the burst layer (burst.py): python -m harness.model.bursttrain {train | eval}.

A burst run starts from a single-frame checkpoint (train.save), whose EMA weights become BurstNet's single network,
whose normaliser and restore conditioning it keeps, and whose loss terms, sources, real pairs and flags it trains
with unless told otherwise; only the burst front end is new, and it starts at zero, so step 0 is the checkpoint.
The front end trains at the full rate and the single network at a share of it (`single_lr`): the frames are new
information, the single network's skill is not.

The stream is BurstStream's: stored real bursts (data.burststore, split train) against their HDR+ merge, synthetic
bursts rendered from the clean sources, and single-frame samples with no frames. Each step warps the frames onto the
reference on the GPU (burst.warp_frames) and encodes them.

Validation, every checkpoint: the synthetic burst set (burst.synthetic_fixed_set, the clean sources' val split) with
all frames and with none, beside the starting checkpoint on the same tiles (logged once), and, where the val burst
store holds bursts, a fixed set of its tiles the same three ways. best.pt follows the synthetic set's loss with all
frames: the stored val scenes are what `eval` reports on.

`eval` scores a burst checkpoint on the val burst store's tiles under heldout.score (the capture gates, the
residual noise reading and the oracle linear frontier), with all frames, with none, and the starting checkpoint
alone: what the frames add, and what the burst run did to the single-frame skill, on the same real tiles."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import loader_context
from ..data import data_root
from ..data.tilemetrics import tile_metrics
from ..metrics import STILL_METRICS
from .burst import BurstNet, BurstStream, encode_t, load_burst, load_single, synthetic_fixed_set, warp_frames
from .data import Normaliser, _h, cond_vector, decode, encode
from .losses import best_shift
from .net import CondUNet
from .train import EMA_DECAY, LR_FLOOR, StreamDataset, lr_at, pair_loss, save

LOSS_KEYS = ("range_weight", "range_k", "flat_weight", "flat_k", "texture_weight", "ssim_weight", "transfer_weight",
             "overshoot_weight", "pyramid_weight")
SINGLE_LR = 0.3      # the single network's rate as a share of the front end's
WARMUP = 300


def predict(model, x: torch.Tensor, c: torch.Tensor, frames: torch.Tensor | None = None,
            coords: torch.Tensor | None = None, mask: torch.Tensor | None = None) -> torch.Tensor:
    """The encoded restoration of encoded x by a CondUNet (the frames ignored) or a BurstNet, the frames linear
    windows warped at `coords` and encoded here."""
    if not isinstance(model, BurstNet):
        return model(x, c)
    if frames is None or frames.shape[1] == 0 or mask is None or not bool(mask.any()):
        return model(x, c)
    with torch.no_grad():
        w = encode_t(warp_frames(frames, coords))
    return model(x, c, w.to(x.dtype), mask)


def to_batch(samples: list[dict], norm: Normaliser, device, frames: bool = True) -> tuple:
    """(x, y, cond, frames, coords, mask) of a list of burst samples; `frames` False masks every frame."""
    x = torch.from_numpy(np.stack([encode(s["x"]) for s in samples])).permute(0, 3, 1, 2).to(device)
    y = torch.from_numpy(np.stack([encode(s["y"]) for s in samples])).permute(0, 3, 1, 2).to(device)
    c = torch.from_numpy(np.stack([cond_vector(norm, s) for s in samples])).to(device)
    fr = torch.from_numpy(np.stack([s["frames"] for s in samples]).astype(np.float32)).permute(0, 1, 4, 2, 3).to(device)
    co = torch.from_numpy(np.stack([s["coords"] for s in samples]).astype(np.float32)).to(device)
    m = torch.from_numpy(np.stack([s["mask"] for s in samples]).astype(bool)).to(device)
    return x, y, c, fr, co, (m if frames else torch.zeros_like(m))


@torch.no_grad()
def validate(model, val: list[dict], norm: Normaliser, device, frames: bool = True, batch: int = 8,
             shift: bool = False, **loss_kw) -> dict:
    """train.validate for burst samples: the mean loss and the restore medians in and out under the residual noise
    reading, with the frames (`frames`) or without."""
    was = model.training
    model.eval()
    losses, rows = [], {"input": [], "output": []}
    for i in range(0, len(val), batch):
        part = val[i:i + batch]
        x, y, c, fr, co, m = to_batch(part, norm, device, frames)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = predict(model, x, c, fr, co, m)
        pred = pred.float()
        lp, ly, lx = (best_shift(pred, y, x, torch.ones(len(part), dtype=torch.bool, device=device)) if shift
                      else (pred, y, x))
        losses.append(float(pair_loss(lp, ly, lx, **loss_kw)))
        out = decode(pred.permute(0, 2, 3, 1).cpu().numpy())
        for s, o in zip(part, out):
            rows["input"].append(tile_metrics(s["x"], s["y"], "residual"))
            rows["output"].append(tile_metrics(o, s["y"], "residual"))
    model.train(was)

    def _med(v, k):
        vals = [r[k] for r in v if np.isfinite(r[k])]
        return float(np.median(vals)) if vals else None
    return {"val_loss": float(np.mean(losses)), "n": len(val),
            "restore": {k: {"input": _med(rows["input"], k), "output": _med(rows["output"], k)} for k in STILL_METRICS}}


def real_burst_set(n: int, tile: int, frames: int, seed: int, root: Path | None = None, split: str = "val",
                   subpix: bool = False) -> list[dict]:
    """n reproducible tiles of the stored bursts of a split, every frame the tile holds kept, no rotation: the real
    burst validation and evaluation set."""
    from ..data.burststore import BurstStore
    store = BurstStore(split, root, tile, frames, subpix=subpix)
    rng = np.random.default_rng(_h("fixed-real-burst", split, n, tile, frames, seed))
    out = []
    for _ in range(n):
        s = store.random_tile(rng, n=frames)
        out.append({"x": s["x"].astype(np.float64), "y": s["y"].astype(np.float64), "capture": s["capture"],
                    "metrics": tile_metrics(s["y"], s["y"]), "restore": True, "real": True, "frames": s["frames"],
                    "coords": s["coords"], "mask": s["mask"], "burst": s["burst"], "pair": s["pair"],
                    "y0": s["y0"], "x0": s["x0"]})
    return out


def _line(v: dict) -> str:
    return ", ".join(f"{k} {r['input']:.3g}->{r['output']:.3g}" for k, r in v["restore"].items()
                     if r["input"] is not None and r["output"] is not None)


def burst_train(out: str, init: str, steps: int = 40000, batch: int = 16, tile: int = 128, frames: int = 4,
                workers: int = 12, lr: float = 1e-4, single_lr: float = SINGLE_LR, minutes: float | None = None,
                seed: int = 0, feat: int = 64, p_real_burst: float = 0.5, p_single: float = 0.1,
                sources: list[str] | None = None, val_n: int = 48, log_every: int = 50, ckpt_every: int = 500,
                device: str | None = None, grad_ckpt: bool = False, compile_model: bool = False,
                warmup: int = WARMUP, lr_floor: float = LR_FLOOR, shift_real: bool | None = None,
                log=print) -> dict:
    """A burst run in `out` from the single-frame checkpoint `init`, resumed from out/last.pt when there is one."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    single, ck0 = load_single(init)
    targs = dict(ck0.get("train") or {})
    norm = Normaliser.from_state(ck0["normaliser"])
    cond_restore = np.array(ck0["cond_restore"], dtype=np.float32)
    loss_kw = {k: targs[k] for k in LOSS_KEYS if targs.get(k) is not None}
    sources = list(sources or targs.get("sources") or [])
    if not sources:
        raise ValueError(f"{init} records no sources: pass --sources")
    real, p_real = tuple(targs.get("real") or ()), float(targs.get("p_real") or 0.0)
    subpix = bool(targs.get("subpix_real", False))
    shift = bool(targs.get("shift_real", False)) if shift_real is None else bool(shift_real)
    audit = bool(targs.get("target_audit", False))
    weights = targs.get("weights") or None
    if tile % 2 ** (single.n_levels - 1):
        raise ValueError(f"tile {tile} must be a multiple of {2 ** (single.n_levels - 1)}")
    args = dict(init=str(init), init_step=int(ck0.get("step", 0)), steps=steps, batch=batch, tile=tile, frames=frames,
                lr=lr, single_lr=single_lr, seed=seed, feat=feat, p_real_burst=p_real_burst, p_single=p_single,
                sources=sources, real=list(real), p_real=p_real, subpix_real=subpix, shift_real=shift,
                target_audit=audit, weights=weights, warmup=warmup, lr_floor=lr_floor, compile=bool(compile_model),
                **loss_kw)
    last = out / "last.pt"
    ck = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else None
    model = BurstNet(single, feat, frames)
    if ck:
        model.load_state_dict(ck["model"])
        log(f"resuming from step {ck['step']}")
    model.to(dev)
    ema = copy.deepcopy(model).eval()
    if ck:
        ema.load_state_dict(ck["ema"])
    for p in ema.parameters():
        p.requires_grad_(False)
    model.set_grad_ckpt(grad_ckpt)
    front = [p for n_, p in model.named_parameters() if not n_.startswith("single.")]
    body = [p for n_, p in model.named_parameters() if n_.startswith("single.")]
    opt = torch.optim.AdamW([{"params": front, "scale": 1.0}, {"params": body, "scale": float(single_lr)}], lr=lr,
                            betas=(0.9, 0.99), weight_decay=0.01, fused=dev.type == "cuda")
    step, best = 0, float("inf")
    if ck:
        opt.load_state_dict(ck["opt"])
        step = int(ck["step"])
    log(f"burst model {model.n_params() / 1e6:.2f} M params ({sum(p.numel() for p in front) / 1e6:.2f} M new) on "
        f"{dev}, from {init} at step {args['init_step']}; {frames} frames, tile {tile}, batch {batch}, "
        f"{workers} workers; stream: {p_real_burst:.0%} stored bursts, {p_single:.0%} single frames, the rest "
        f"synthetic bursts of {len(sources)} sources")
    log(f"validation draw: {val_n} synthetic bursts at {tile} px")
    val = synthetic_fixed_set(sources, "val", val_n, tile, frames, seed + 2, audit=audit)
    try:
        val_real = real_burst_set(val_n, tile, frames, seed + 3, subpix=subpix)
        log(f"real burst validation draw: {len(val_real)} tiles of the stored val bursts")
    except (ValueError, FileNotFoundError) as ex:
        val_real = None
        log(f"no real burst validation set: {ex}")
    logf = open(out / "train_log.jsonl", "a", encoding="utf-8")
    ref = single.to(dev).eval()
    if not ck:
        for name, vs, sh in (("val_single", val, False), ("val_real_single", val_real, shift)):
            if vs:
                v = validate(ref, vs, norm, dev, shift=sh, **loss_kw)
                logf.write(json.dumps({"step": 0, name: v}) + "\n"); logf.flush()
                log(f"  {name}: loss {v['val_loss']:.4f}; {_line(v)}")
    ref.cpu()
    del ref

    def val_all(at: int) -> dict:
        rec = {"step": at}
        for name, vs, sh in (("val", val, False), ("val_real", val_real, shift)):
            if not vs:
                continue
            for tag, fr in (("frames", True), ("none", False)):
                v = validate(ema, vs, norm, dev, fr, shift=sh, **loss_kw)
                rec[f"{name}_{tag}"] = v
                log(f"  {name} {tag}: loss {v['val_loss']:.4f}; {_line(v)}")
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        return rec

    stream = BurstStream(sources, tile, norm, seed + step, frames=frames, p_real_burst=p_real_burst,
                         p_single=p_single, real=real, p_real=p_real, weights=weights, audit=audit,
                         subpix_real=subpix)
    loader = DataLoader(StreamDataset(stream), batch_size=batch, num_workers=workers, pin_memory=dev.type == "cuda",
                        persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None,
                        multiprocessing_context=loader_context(workers))
    ema_params, model_params = list(ema.parameters()), list(model.parameters())
    t0, t_log, n_log, run_loss = time.perf_counter(), time.perf_counter(), 0, 0.0
    t_prev = float(ck.get("elapsed_min", 0.0)) if ck else 0.0

    def spent() -> float:
        return t_prev + (time.perf_counter() - t0) / 60.0

    def frac() -> float:
        return spent() / minutes if minutes else 0.0

    fwd = torch.compile(model) if compile_model else model
    model.train()
    for x, y, c, fr, co, m, is_real in loader:
        if step >= steps or frac() >= 1.0:
            break
        x, y, c = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), c.to(dev, non_blocking=True)
        fr, co, m = fr.to(dev, non_blocking=True), co.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
        rate = lr_at(step, steps, lr, frac(), warmup, lr_floor)
        for g in opt.param_groups:
            g["lr"] = rate * g["scale"]
        with torch.no_grad():
            w = encode_t(warp_frames(fr, co))
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pred = fwd(x, c, w, m)
        pred, yt, xt = pred.float(), y, x
        if shift:
            pred, yt, xt = best_shift(pred, y, x, is_real.to(dev))
        loss = pair_loss(pred, yt, xt, **loss_kw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, model_params, 1.0 - EMA_DECAY)
        step += 1
        run_loss += loss.item(); n_log += 1
        if step % log_every == 0:
            now = time.perf_counter()
            rec = {"step": step, "loss": run_loss / n_log, "lr": rate, "imgs_s": n_log * batch / (now - t_log),
                   "frames_per_sample": float(m.float().sum(dim=1).mean()), "real_share": float(is_real.float().mean()),
                   "elapsed_s": now - t0,
                   "gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            log(f"step {step} loss {rec['loss']:.4f} lr {rate:.2e} {rec['imgs_s']:.0f} img/s")
            t_log, n_log, run_loss = now, 0, 0.0
        if step % ckpt_every == 0:
            v = val_all(step)
            save(last, model, ema, opt, step, norm, cond_restore, args, spent())
            if v["val_frames"]["val_loss"] < best:
                best = v["val_frames"]["val_loss"]
                save(out / "best.pt", model, ema, opt, step, norm, cond_restore, args, spent())
    v = val_all(step)
    logf.close()
    save(last, model, ema, opt, step, norm, cond_restore, args, spent())
    if v["val_frames"]["val_loss"] < best:
        save(out / "best.pt", model, ema, opt, step, norm, cond_restore, args, spent())
    log(f"done at step {step}: synthetic val loss with frames {v['val_frames']['val_loss']:.4f}, "
        f"{(time.perf_counter() - t0) / 60:.1f} min")
    return v


def _restorer(model, norm: Normaliser, cond_restore: np.ndarray, dev, frames: bool, bf16: bool = True):
    """heldout.score's restore(sample) for a CondUNet or a BurstNet, the capture descriptor known."""
    @torch.no_grad()
    def restore(s: dict) -> np.ndarray:
        x = torch.from_numpy(encode(s["x"])).permute(2, 0, 1)[None].to(dev)
        c = torch.from_numpy(np.concatenate([cond_restore, np.asarray(s["capture"], dtype=np.float32)]))[None].to(dev)
        fr = torch.from_numpy(np.asarray(s["frames"], dtype=np.float32)).permute(0, 3, 1, 2)[None].to(dev)
        co = torch.from_numpy(np.asarray(s["coords"], dtype=np.float32))[None].to(dev)
        m = torch.from_numpy(np.asarray(s["mask"], dtype=bool))[None].to(dev)
        if not frames:
            m = torch.zeros_like(m)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16 and dev.type == "cuda"):
            y = predict(model, x, c, fr, co, m)
        return np.maximum(decode(y.float()[0].permute(1, 2, 0).cpu().numpy()), 0.0).astype(np.float32)
    return restore


def burst_evaluate(ckpt, n: int = 64, tile: int = 256, seed: int = 0, frames: int | None = None,
                   device: str | None = None, out: Path | None = None, frontier: bool = True,
                   bf16: bool = True, log=print) -> dict:
    """The burst checkpoint on the val burst store's tiles (real_burst_set) with its frames, with none, and its
    starting checkpoint alone, each a heldout.score report; with `out` the three reports and their restored tiles
    are written there."""
    from .frontier import cached_ladder
    from .heldout import OUTPUTS_FILE, score
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ck = load_burst(ckpt)
    targs = ck.get("train") or {}
    frames = int(frames or model.frames)
    norm, cond_restore = Normaliser.from_state(ck["normaliser"]), np.array(ck["cond_restore"], dtype=np.float32)
    samples = real_burst_set(n, tile, frames, seed, subpix=bool(targs.get("subpix_real", False)))
    log(f"{len(samples)} tiles of the val bursts at {tile} px, {np.mean([s['mask'].sum() for s in samples]):.2f} "
        f"frames a tile")
    table = cached_ladder(samples, data_root(), "residual", log=log) if frontier else None
    model.to(dev).eval()
    reps = {}
    for tag, fr in (("frames", True), ("none", False)):
        t0 = time.perf_counter()
        reps[tag] = score(samples, _restorer(model, norm, cond_restore, dev, fr, bf16), "residual", table)
        reps[tag]["seconds"] = round(time.perf_counter() - t0, 1)
    model.cpu()
    init = targs.get("init")
    if init and Path(init).exists():
        single, ck0 = load_single(init)
        single.to(dev).eval()
        n0 = Normaliser.from_state(ck0["normaliser"])
        c0 = np.array(ck0["cond_restore"], dtype=np.float32)
        reps["single"] = score(samples, _restorer(single, n0, c0, dev, False, bf16), "residual", table)
    meta = dict(ckpt=str(ckpt), step=int(ck["step"]), init=init, n=len(samples), tile=tile, seed=seed, frames=frames,
                bursts=sorted({int(s["burst"]) for s in samples}))
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for tag, r in reps.items():
            outputs = r.pop("outputs")
            np.savez(out / f"{tag}_{OUTPUTS_FILE}", outputs=np.stack(outputs).astype(np.float32))
            (out / f"burst_{tag}.json").write_text(json.dumps({**meta, "mode": tag, **r}, indent=1), encoding="utf-8")
    else:
        for r in reps.values():
            r.pop("outputs")
    return {**meta, "reports": reps}


def summary_table(rep: dict) -> str:
    """The medians of each mode beside the input and the truth, and each mode's gate verdicts."""
    modes = list(rep["reports"])
    first = rep["reports"][modes[0]]["metrics"]
    lines = ["| metric | input | " + " | ".join(modes) + " | truth |", "|---" * (len(modes) + 3) + "|"]
    for m in STILL_METRICS:
        row = [first[m]["input"]] + [rep["reports"][k]["metrics"][m]["output"] for k in modes] + [first[m]["truth"]]
        lines.append(f"| {m} | " + " | ".join("-" if v is None else f"{v:.4g}" for v in row) + " |")
    lines.append("")
    for k in modes:
        g = rep["reports"][k]["gates"]
        lines.append(f"- {k}: " + ", ".join(f"{name} {gg['overall']}" for name, gg in g.items()))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness.model.bursttrain")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train the burst layer from a single-frame checkpoint")
    t.add_argument("--out", required=True)
    t.add_argument("--init", required=True, help="the single-frame checkpoint (its EMA weights)")
    t.add_argument("--steps", type=int, default=40000)
    t.add_argument("--minutes", type=float)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--tile", type=int, default=128)
    t.add_argument("--frames", type=int, default=4)
    t.add_argument("--workers", type=int, default=12)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--single-lr", type=float, default=SINGLE_LR, help="the single network's rate as a share of --lr")
    t.add_argument("--feat", type=int, default=64)
    t.add_argument("--p-real-burst", type=float, default=0.5)
    t.add_argument("--p-single", type=float, default=0.1)
    t.add_argument("--sources", nargs="+", help="clean sources (default: the starting checkpoint's)")
    t.add_argument("--val-n", type=int, default=48)
    t.add_argument("--ckpt-every", type=int, default=500)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--warmup", type=int, default=WARMUP)
    t.add_argument("--grad-ckpt", action="store_true")
    t.add_argument("--compile", action="store_true")
    t.add_argument("--device")
    e = sub.add_parser("eval", help="score a burst checkpoint on the val burst store")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", help="report directory (default: burst_eval beside the checkpoint)")
    e.add_argument("--n", type=int, default=64)
    e.add_argument("--tile", type=int, default=256)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--frames", type=int)
    e.add_argument("--no-frontier", action="store_true")
    e.add_argument("--device")
    a = ap.parse_args(argv)
    if a.cmd == "train":
        burst_train(a.out, a.init, steps=a.steps, batch=a.batch, tile=a.tile, frames=a.frames, workers=a.workers,
                    lr=a.lr, single_lr=a.single_lr, minutes=a.minutes, seed=a.seed, feat=a.feat,
                    p_real_burst=a.p_real_burst, p_single=a.p_single, sources=a.sources, val_n=a.val_n,
                    ckpt_every=a.ckpt_every, device=a.device, grad_ckpt=a.grad_ckpt, compile_model=a.compile,
                    warmup=a.warmup)
        return 0
    out = Path(a.out) if a.out else Path(a.ckpt).resolve().parent / "burst_eval"
    rep = burst_evaluate(a.ckpt, n=a.n, tile=a.tile, seed=a.seed, frames=a.frames, device=a.device, out=out,
                         frontier=not a.no_frontier)
    md = summary_table(rep)
    (out / "burst_eval.md").write_text(md, encoding="utf-8")
    print(md, end="")
    print(f"  report: {out / 'burst_eval.md'}")
    return 0


if __name__ == "__main__":
    import harness  # noqa: F401  (workers import harness before torch: harness.loader_context)
    raise SystemExit(main())
