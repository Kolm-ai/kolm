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

from ..data.tilemetrics import tile_metrics
from ..metrics import STILL_METRICS
from .data import Normaliser, PairStream, cond_vector, decode, encode, fixed_set
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
WARMUP = 500
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


def pair_loss(pred: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    w = torch.where((x >= CLIP_LOG).any(dim=1, keepdim=True), CLIP_WEIGHT, 1.0)
    l1 = (w * (pred - y).abs()).mean()
    gx = (pred[..., :, 1:] - pred[..., :, :-1]) - (y[..., :, 1:] - y[..., :, :-1])
    gy = (pred[..., 1:, :] - pred[..., :-1, :]) - (y[..., 1:, :] - y[..., :-1, :])
    rng = (w * range_excess(pred, y)).mean()
    return l1 + GRAD_WEIGHT * 0.5 * (gx.abs().mean() + gy.abs().mean()) + RANGE_WEIGHT * rng


def lr_at(step: int, steps: int, lr: float) -> float:
    if step < WARMUP:
        return lr * (step + 1) / WARMUP
    t = min(1.0, (step - WARMUP) / max(1, steps - WARMUP))
    return lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * t)))


def to_batch(samples: list[dict], norm: Normaliser, device) -> tuple:
    x = torch.from_numpy(np.stack([encode(s["x"]) for s in samples])).permute(0, 3, 1, 2).to(device)
    y = torch.from_numpy(np.stack([encode(s["y"]) for s in samples])).permute(0, 3, 1, 2).to(device)
    c = torch.from_numpy(np.stack([cond_vector(norm, s) for s in samples])).to(device)
    return x, y, c


@torch.no_grad()
def validate(model: torch.nn.Module, val: list[dict], norm: Normaliser, device, batch: int = 8) -> dict:
    model.eval()
    losses, rows = [], {"input": [], "output": []}
    for i in range(0, len(val), batch):
        part = val[i:i + batch]
        x, y, c = to_batch(part, norm, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(x, c)
        pred = pred.float()
        losses.append(float(pair_loss(pred, y, x)))
        out = decode(pred.permute(0, 2, 3, 1).cpu().numpy())
        for s, o in zip(part, out):
            if s["restore"]:
                rows["input"].append(tile_metrics(s["x"], s["y"]))
                rows["output"].append(tile_metrics(o, s["y"]))
    model.train()
    def _med(v, m):
        vals = [r[m] for r in v if np.isfinite(r[m])]
        return float(np.median(vals)) if vals else None
    med = {k: {m: _med(v, m) for m in STILL_METRICS} for k, v in rows.items()}
    return {"val_loss": float(np.mean(losses)), "n_restore": len(rows["input"]),
            "restore": {m: {"input": med["input"][m], "output": med["output"][m]} for m in STILL_METRICS}}


def save(path: Path, model, ema, opt, step: int, norm: Normaliser, cond_restore, args: dict) -> None:
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                "config": model.config, "normaliser": norm.state(), "cond_restore": list(map(float, cond_restore)),
                "train": args}, path)


def train(out: str, sources=DEFAULT_SOURCES, steps: int = 20000, batch: int = 16, tile: int = 256, workers: int = 12,
          lr: float = 2e-4, minutes: float | None = None, seed: int = 0, base: int = 48, val_n: int = 48,
          calib_n: int = 128, log_every: int = 50, ckpt_every: int = 500, device: str | None = None, log=print) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args = dict(sources=list(sources), steps=steps, batch=batch, tile=tile, lr=lr, seed=seed, base=base)
    last = out / "last.pt"
    ck = torch.load(last, map_location="cpu", weights_only=False) if last.exists() else None
    quiet = (lambda *a, **k: None)
    if ck:
        norm, cond_restore = Normaliser.from_state(ck["normaliser"]), np.array(ck["cond_restore"], dtype=np.float32)
        log(f"resuming from step {ck['step']}")
    else:
        log(f"calibration draw: {calib_n} samples")
        calib = fixed_set(list(sources), "train", calib_n, min(tile, 128), seed + 1, log=quiet)
        norm = Normaliser.fit(np.array([[s["metrics"][m] for m in STILL_METRICS] for s in calib]))
        cond_restore = np.mean([norm(s["metrics"]) for s in calib if s["restore"]], axis=0).astype(np.float32)
    log(f"validation draw: {val_n} samples at {tile} px")
    val = fixed_set(list(sources), "val", val_n, tile, seed + 2, log=quiet)
    model = CondUNet(base=base).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.01)
    step, best = 0, float("inf")
    if ck:
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
    log(f"model {model.n_params() / 1e6:.2f} M params on {dev}; {steps} steps, batch {batch}, tile {tile}, {workers} workers")
    loader = DataLoader(StreamDataset(PairStream(list(sources), tile, norm, seed + step)), batch_size=batch,
                        num_workers=workers, pin_memory=dev.type == "cuda", persistent_workers=workers > 0,
                        prefetch_factor=4 if workers > 0 else None)
    logf = open(out / "train_log.jsonl", "a", encoding="utf-8")
    t0, t_log, n_log, run_loss = time.perf_counter(), time.perf_counter(), 0, 0.0
    model.train()
    for x, y, c, _ in loader:
        if step >= steps or (minutes and (time.perf_counter() - t0) / 60.0 >= minutes):
            break
        x, y, c = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), c.to(dev, non_blocking=True)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pred = model(x, c)
        loss = pair_loss(pred.float(), y, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1.0 - EMA_DECAY)
        step += 1
        run_loss += loss.item(); n_log += 1
        if step % log_every == 0:
            now = time.perf_counter()
            rec = {"step": step, "loss": run_loss / n_log, "lr": lr_at(step, steps, lr),
                   "imgs_s": n_log * batch / (now - t_log), "elapsed_s": now - t0,
                   "gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            log(f"step {step} loss {rec['loss']:.4f} lr {rec['lr']:.2e} {rec['imgs_s']:.0f} img/s")
            t_log, n_log, run_loss = now, 0, 0.0
        if step % ckpt_every == 0:
            v = validate(ema, val, norm, dev)
            v["step"] = step
            logf.write(json.dumps(v) + "\n"); logf.flush()
            save(last, model, ema, opt, step, norm, cond_restore, args)
            if v["val_loss"] < best:
                best = v["val_loss"]; save(out / "best.pt", model, ema, opt, step, norm, cond_restore, args)
            log(f"  val loss {v['val_loss']:.4f} (best {best:.4f}); restore medians in -> out: " + ", ".join(
                f"{m} {r['input']:.3g}->{r['output']:.3g}" for m, r in v["restore"].items() if r["input"] is not None))
    v = validate(ema, val, norm, dev)
    v["step"] = step
    logf.write(json.dumps(v) + "\n"); logf.close()
    save(last, model, ema, opt, step, norm, cond_restore, args)
    if v["val_loss"] < best:
        save(out / "best.pt", model, ema, opt, step, norm, cond_restore, args)
    log(f"done at step {step}: val loss {v['val_loss']:.4f}, {(time.perf_counter() - t0) / 60:.1f} min")
    return v
