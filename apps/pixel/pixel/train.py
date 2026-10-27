"""Conditional restoration / edit model. Pairs are regenerated, never stored.

Each sample is a clean crop ``C`` and its degraded twin ``D = apply(C, recipe(seed))``
with direction vector ``v``. Half the time the model sees ``(D, -v) -> C``
(restore), half the time ``(C, +v) -> D`` (degrade on purpose). So one network
learns to move an image along every quality axis in both directions, which is
what "edit in different directions" means operationally. An auxiliary head
predicts the direction of the *input* (0 for clean), so the same network is
also a no-reference quality estimator.

torch is imported lazily: the rest of the package never needs it.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .degrade import DIRECTION_DIM, label_pair
from .label import prepare
from .ops import decode_image, load_image


def _torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
        import torch.nn.functional as F  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("training needs torch: pip install 'kolm-pixel[train]'") from e
    return torch, nn, F


@dataclass
class TrainConfig:
    sources: str = ""  # directory of images, or a directory of shards (*.tar)
    out_dir: str = "runs/pixel"
    size: int = 128
    batch: int = 8
    steps: int = 1000
    lr: float = 2e-4
    base_channels: int = 32
    depth: int = 3
    recipes_per_source: int = 16
    max_ops: int = 3
    direction_weight: float = 0.1
    val_every: int = 200
    val_samples: int = 32
    ckpt_every: int = 500
    workers: int = 0
    seed: int = 0
    device: str = "auto"
    smoke: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# -- data ----------------------------------------------------------------------
def list_sources(path: str | Path) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    imgs = sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"})
    if imgs:
        return imgs
    return sorted(str(f) for f in p.glob("*.tar"))


class _SourceCache:
    """Decoded sources, LRU-bounded; shards are expanded lazily by key."""

    def __init__(self, paths: list[str], size: int, capacity: int = 256):
        self.paths = paths
        self.size = size
        self.capacity = capacity
        self._cache: dict[int, np.ndarray] = {}
        self._order: list[int] = []
        self._shard_index: list[tuple[str, str]] | None = None
        if paths and paths[0].endswith(".tar"):
            from .data.shards import read_shard

            self._shard_index = []
            for shard in paths:
                for key, _ext, _data, _meta in read_shard(shard):
                    self._shard_index.append((shard, key))

    def __len__(self) -> int:
        return len(self._shard_index) if self._shard_index is not None else len(self.paths)

    def source_id(self, i: int) -> str:
        return self._shard_index[i][1] if self._shard_index is not None else Path(self.paths[i]).name

    def get(self, i: int) -> np.ndarray:
        if i in self._cache:
            return self._cache[i]
        if self._shard_index is not None:
            from .data.shards import read_shard

            shard, key = self._shard_index[i]
            img = None
            for k, _ext, data, _meta in read_shard(shard):
                if k == key:
                    img = decode_image(data)
                    break
            if img is None:
                raise KeyError(key)
        else:
            img = load_image(self.paths[i])
        H, W = img.shape[:2]
        if min(H, W) > 2 * self.size:
            img = prepare(img, 2 * self.size)
        self._cache[i] = img
        self._order.append(i)
        if len(self._order) > self.capacity:
            del self._cache[self._order.pop(0)]
        return img


def make_dataset(cfg: TrainConfig, paths: list[str], epoch: int = 0):
    torch, _nn, _F = _torch()

    class PairDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.cache = _SourceCache(paths, cfg.size)
            self.R = cfg.recipes_per_source

        def __len__(self):
            return len(self.cache) * self.R

        def __getitem__(self, i):
            si, r = i // self.R, i % self.R
            img = self.cache.get(si)
            rng = np.random.default_rng((epoch * 1_000_003 + i) & 0x7FFFFFFF)
            H, W = img.shape[:2]
            y0 = int(rng.integers(0, max(1, H - cfg.size + 1)))
            x0 = int(rng.integers(0, max(1, W - cfg.size + 1)))
            clean = img[y0 : y0 + cfg.size, x0 : x0 + cfg.size]
            if clean.shape[0] != cfg.size or clean.shape[1] != cfg.size:
                clean = prepare(clean, cfg.size)
            degraded, _recipe, v, _seed = label_pair(clean, self.cache.source_id(si), r + epoch * self.R, max_ops=cfg.max_ops)
            if rng.random() < 0.5:
                x, cond, y, vin = degraded, -v, clean, v
            else:
                x, cond, y, vin = clean, v, degraded, np.zeros_like(v)
            to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))  # noqa: E731
            return to_t(x), torch.from_numpy(cond.astype(np.float32)), to_t(y), torch.from_numpy(vin.astype(np.float32))

    return PairDataset()


# -- model ---------------------------------------------------------------------
def build_model(base_channels: int = 32, depth: int = 3, direction_dim: int = DIRECTION_DIM):
    torch, nn, F = _torch()

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.SiLU(),
        )

    class PixelNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.direction_dim = direction_dim
            chans = [base_channels * (2**i) for i in range(depth)]
            self.inp = block(3 + direction_dim, chans[0])
            self.down = nn.ModuleList([block(chans[i], chans[i + 1]) for i in range(depth - 1)])
            self.up = nn.ModuleList([block(chans[i + 1] + chans[i], chans[i]) for i in reversed(range(depth - 1))])
            self.out = nn.Conv2d(chans[0], 3, 3, padding=1)
            self.head = nn.Sequential(nn.Linear(chans[-1], chans[-1]), nn.SiLU(), nn.Linear(chans[-1], direction_dim))

        def forward(self, x, cond):
            B, _, H, W = x.shape
            c = cond.view(B, self.direction_dim, 1, 1).expand(B, self.direction_dim, H, W)
            h = self.inp(torch.cat([x, c], dim=1))
            skips = [h]
            for d in self.down:
                h = d(F.avg_pool2d(h, 2))
                skips.append(h)
            pred_dir = self.head(h.mean(dim=(2, 3)))
            skips.pop()
            for u in self.up:
                h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
                h = u(torch.cat([h, skips.pop()], dim=1))
            return (x + self.out(h)).clamp(0.0, 1.0), pred_dir

    return PixelNet()


def _device(name: str):
    torch, _nn, _F = _torch()
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- loop ----------------------------------------------------------------------
def train(cfg: TrainConfig) -> dict:
    torch, nn, F = _torch()
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if cfg.smoke:
        cfg = _smoke_sources(cfg, out)
    paths = list_sources(cfg.sources)
    if not paths:
        raise FileNotFoundError(f"no sources under {cfg.sources}")
    dev = _device(cfg.device)
    model = build_model(cfg.base_channels, cfg.depth).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, cfg.steps) / cfg.steps)))
    log_path = out / "log.jsonl"
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    step, epoch, losses, t0 = 0, 0, [], time.time()
    history = []
    while step < cfg.steps:
        ds = make_dataset(cfg, paths, epoch)
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.workers, drop_last=True)
        for x, cond, y, vin in loader:
            x, cond, y, vin = x.to(dev), cond.to(dev), y.to(dev), vin.to(dev)
            model.train()
            pred, pdir = model(x, cond)
            loss_img = F.l1_loss(pred, y)
            loss_dir = F.mse_loss(pdir, vin)
            loss = loss_img + cfg.direction_weight * loss_dir
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            loss_v, l1_v, dir_v = loss.item(), loss_img.item(), loss_dir.item()
            losses.append(loss_v)
            row = {"step": step, "loss": loss_v, "l1": l1_v, "dir_mse": dir_v, "lr": sched.get_last_lr()[0], "t": time.time() - t0}
            if step % cfg.val_every == 0 or step == cfg.steps:
                row["val"] = validate(model, cfg, paths, dev)
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step}, out / "last.pt")
            if step % cfg.ckpt_every == 0:
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step}, out / f"step{step:07d}.pt")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            history.append(row)
            if step >= cfg.steps:
                break
        epoch += 1
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step}, out / "last.pt")
    k = max(1, min(10, len(losses) // 4))
    return {
        "steps": step,
        "loss_first": float(np.mean(losses[:k])),
        "loss_last": float(np.mean(losses[-k:])),
        "val": history[-1].get("val") if history else None,
        "checkpoint": str(out / "last.pt"),
        "log": str(log_path),
    }


def validate(model, cfg: TrainConfig, paths: list[str], dev) -> dict:
    torch, _nn, F = _torch()
    model.eval()
    ds = make_dataset(cfg, paths, epoch=10_000)  # a seed range training never sees
    n = min(cfg.val_samples, len(ds))
    psnr_in, psnr_out, dir_mae = [], [], []
    with torch.no_grad():
        for i in range(n):
            x, cond, y, vin = ds[i]
            x, cond, y, vin = x[None].to(dev), cond[None].to(dev), y[None].to(dev), vin[None].to(dev)
            pred, pdir = model(x, cond)
            psnr_in.append(float(-10 * torch.log10(F.mse_loss(x, y).clamp_min(1e-10))))
            psnr_out.append(float(-10 * torch.log10(F.mse_loss(pred, y).clamp_min(1e-10))))
            dir_mae.append(float((pdir - vin).abs().mean()))
    return {"n": n, "psnr_input": float(np.mean(psnr_in)), "psnr_output": float(np.mean(psnr_out)), "direction_mae": float(np.mean(dir_mae))}


def _smoke_sources(cfg: TrainConfig, out: Path) -> TrainConfig:
    """Procedural scenes as sources so the loop can be exercised with no data at all."""
    from .colorsci import display_encode
    from .ops import save_image
    from .scene import Scene, SceneSpec

    src = out / "smoke_sources"
    src.mkdir(parents=True, exist_ok=True)
    for seed in range(12):
        p = src / f"scene{seed:03d}.png"
        if not p.exists():
            save_image(p, display_encode(Scene(SceneSpec(seed=100 + seed, width=192, height=192)).render(0.0)))
    cfg.sources = str(src)
    cfg.size = min(cfg.size, 64)
    cfg.base_channels = min(cfg.base_channels, 16)
    cfg.depth = min(cfg.depth, 2)
    cfg.recipes_per_source = min(cfg.recipes_per_source, 4)
    cfg.val_samples = min(cfg.val_samples, 8)
    return cfg


# -- inference -------------------------------------------------------------------
def load_model(ckpt: str | Path, device: str = "auto"):
    torch, _nn, _F = _torch()
    state = torch.load(ckpt, map_location="cpu")
    cfg = TrainConfig.from_dict(state["config"])
    model = build_model(cfg.base_channels, cfg.depth)
    model.load_state_dict(state["model"])
    return model.to(_device(device)).eval(), cfg


def edit(model, img: np.ndarray, direction: np.ndarray, device: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """Move ``img`` along ``direction``; returns (edited image, predicted direction of the input)."""
    torch, _nn, F = _torch()
    dev = _device(device)
    H, W = img.shape[:2]
    m = 2 ** (len(list(model.up)) + 1)
    ph, pw = (-H) % m, (-W) % m
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].to(dev)
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    cond = torch.from_numpy(np.asarray(direction, dtype=np.float32))[None].to(dev)
    with torch.no_grad():
        pred, pdir = model(x, cond)
    out = pred[0, :, :H, :W].cpu().numpy().transpose(1, 2, 0)
    return out.astype(np.float32), pdir[0].cpu().numpy()


def parse_direction(text: str) -> np.ndarray:
    """'blur=-1,noise=-0.5' -> direction vector; unnamed families are 0."""
    from .degrade import FAMILIES

    v = np.zeros(DIRECTION_DIM, dtype=np.float32)
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, _, val = part.partition("=")
        if name not in FAMILIES:
            raise KeyError(f"unknown family {name!r}; one of {FAMILIES}")
        v[FAMILIES.index(name)] = float(val or 1.0)
    return v
