"""The burst layer (MODEL_PLAN section 9): a front end that reads up to F more frames of the reference frame's
burst, each warped onto the reference's pixels, and adds what they carry to the single-frame network's stem.

A frame is warped where the burst store (data.burststore) or the synthetic renderer below says each reference pixel
falls in it, bicubic in linear light. Each frame, and the reference itself, goes through one small dilated encoder
that sees the frame, its difference to the reference and the reference's noise map (net.noise_map), and puts out
features and a weight logit. A softmax over the reference and the frames present gives per-pixel weights, which pool
the features (with their maximum beside them) and merge the frames in linear light. The pooled features, the
merge's difference to the reference, the share of frames present and the merge's effective frame count go through a
zero-initialised convolution into the stem's output: at initialisation the burst network is the single network it
was warm-started from, whatever frames it is given.

A synthetic burst renders one clean crop as up to F + 1 frames of one camera. Each frame sees the scene through its
own sub-pixel translation, small rotation and scale, sometimes with a rectangle of the scene moved by a few pixels
(motion the registration does not know about), and draws its own noise. The coordinates handed to the warp are the
frames' true transforms plus a registration error, so the network learns to weight a frame by how well it agrees
with the reference, not to trust the registration."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

from ..data.burststore import WARP_MARGIN
from ..data.degrade import DegradeParams, Distributions, degrade, prepare_tile
from ..data.tilemetrics import tile_metrics
from .data import (LOG_GAIN, P_CAPTURE_KNOWN, TILES_PER_IMAGE, Normaliser, TileSource, _h, cond_vector, draw_state,
                   encode, input_shading, make_sample, real_sample, restore_target)
from .net import CondUNet, noise_map

SYN_PAD = 16            # px of clean crop beyond a synthetic frame's window: room for the transforms and the motion
MAX_SHIFT_PX = 0.5      # a frame's translation against the reference, uniform per axis: a hand-held burst's sub-pixel phase
MAX_ROT_DEG = 0.3       # its rotation, uniform
MAX_SCALE = 0.002       # its scale change, uniform
P_MOVER = 0.3           # a frame with a rectangle of the scene moved that the registration does not know about
MOVER_PX = (16, 64)     # the rectangle's sides
MOVER_SHIFT_PX = (1.0, 6.0)   # how far it moved
REG_SIGMA_PX = 0.1      # the registration's error per frame, a whole-window offset
GAIN_JITTER = 0.01      # a frame's exposure against the reference's after normalisation
P_BURST_CAPTURE = 0.8   # synthetic bursts drawn with the sampler's raw "capture" profile, else "iqa"


def encode_t(x: torch.Tensor) -> torch.Tensor:
    """data.encode on a tensor."""
    return torch.log1p(LOG_GAIN * x.clamp_min(0.0))


def warp_frames(frames: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """(N, F, 3, H, W) linear frame windows read at (N, F, T, T, 2) (row, column) window coordinates: (N, F, 3, T, T),
    bicubic, clamped at zero (the cubic's undershoot at an edge is not light)."""
    n, f, c, h, w = frames.shape
    th, tw = coords.shape[2], coords.shape[3]
    g = coords.reshape(n * f, th, tw, 2).float()
    grid = torch.stack([(2.0 * g[..., 1] + 1.0) / w - 1.0, (2.0 * g[..., 0] + 1.0) / h - 1.0], dim=-1)
    out = F.grid_sample(frames.reshape(n * f, c, h, w).float(), grid, mode="bicubic", padding_mode="border",
                        align_corners=False)
    return out.clamp_min(0.0).reshape(n, f, c, th, tw)


class FrameEncoder(nn.Module):
    """Features and a weight logit per pixel of one warped frame (or the reference), from the frame, its difference to
    the reference and the reference's noise map, over a 17 px receptive field (dilations 1, 2, 4). The logit starts at
    zero, so the first merge is the plain mean of the frames present."""
    def __init__(self, feat: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(7, feat, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(feat, feat, 3, padding=2, dilation=2), nn.SiLU(),
                                 nn.Conv2d(feat, feat, 3, padding=4, dilation=4), nn.SiLU(),
                                 nn.Conv2d(feat, feat + 1, 3, padding=1))
        with torch.no_grad():
            self.net[-1].weight[-1].zero_()
            self.net[-1].bias[-1].zero_()

    def forward(self, w: torch.Tensor, x: torch.Tensor, nm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(torch.cat([w, w - x, nm], dim=1))
        return out[:, :-1], out[:, -1:]


def _body(u: CondUNet, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """CondUNet.forward between the stem and the head."""
    skips = []
    for i in range(u.n_levels):
        h = u.down[i](h, e)
        if i + 1 < u.n_levels:
            skips.append(h)
            h = u.pool[i](h)
    h = u.mid(h, e)
    for j in range(u.n_levels - 1):
        h = u.up[j](h)
        h = u.merge[j](torch.cat([h, skips.pop()], dim=1))
        h = u.down[u.n_levels + j](h, e)
    return h


class BurstNet(nn.Module):
    """A single-frame CondUNet with the burst front end added to its stem's output. forward(x, cond, frames, mask):
    x (N, 3, T, T) and frames (N, F, 3, T, T) encoded (frames already warped onto x's pixels), mask (N, F) the frames
    present. With no frames, or none present, the front end sees the reference alone."""
    def __init__(self, single: CondUNet | dict, feat: int = 64, frames: int = 4):
        super().__init__()
        self.single = single if isinstance(single, CondUNet) else CondUNet(**single)
        self.feat, self.frames = int(feat), int(frames)
        self.config = {"single": dict(self.single.config), "feat": self.feat, "frames": self.frames}
        self.encoder = FrameEncoder(self.feat)
        c0 = self.single.stem.out_channels
        self.mix = nn.Sequential(nn.Conv2d(2 * self.feat + 5, 2 * self.feat, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(2 * self.feat, c0, 1))
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def set_grad_ckpt(self, on: bool = True) -> "BurstNet":
        self.single.set_grad_ckpt(on)
        return self

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def fuse(self, x: torch.Tensor, nm: torch.Tensor, frames: torch.Tensor | None = None,
             mask: torch.Tensor | None = None) -> torch.Tensor:
        n, _, hh, ww = x.shape
        if frames is None or frames.shape[1] == 0:
            frames = x.new_zeros((n, 0, 3, hh, ww))
            mask = torch.zeros((n, 0), dtype=torch.bool, device=x.device)
        k = frames.shape[1] + 1
        allw = torch.cat([x[:, None], frames.to(x.dtype)], dim=1)                       # (N, F+1, 3, H, W)
        valid = torch.cat([torch.ones((n, 1), dtype=torch.bool, device=x.device), mask.bool()], dim=1)
        feats, logit = self.encoder(allw.reshape(n * k, 3, hh, ww),
                                    x[:, None].expand(n, k, 3, hh, ww).reshape(n * k, 3, hh, ww),
                                    nm[:, None].expand(n, k, 1, hh, ww).reshape(n * k, 1, hh, ww))
        feats = feats.reshape(n, k, self.feat, hh, ww)
        vm = valid[:, :, None, None, None]
        a = torch.softmax(logit.reshape(n, k, 1, hh, ww).float().masked_fill(~vm, float("-inf")), dim=1)
        pooled = (a.to(feats.dtype) * feats).sum(dim=1)
        top = feats.masked_fill(~vm, float("-inf")).amax(dim=1)
        lin = torch.expm1(allw.float().clamp_min(0.0)) / LOG_GAIN
        merged = torch.log1p(LOG_GAIN * (a * lin).sum(dim=1)) - x.float()             # (N, 3, H, W)
        neff = 1.0 / (a * a).sum(dim=1) / (self.frames + 1)                            # (N, 1, H, W)
        share = ((valid.sum(dim=1).float() - 1.0) / max(self.frames, 1)).view(n, 1, 1, 1).expand(n, 1, hh, ww)
        z = torch.cat([pooled, top, merged.to(pooled.dtype), share.to(pooled.dtype), neff.to(pooled.dtype)], dim=1)
        return self.mix(z)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, frames: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        u = self.single
        e = u.embed(cond)
        nm = noise_map(x, cond)
        h = u.stem(torch.cat([x, nm], dim=1) if u.noise_map else x)
        h = h + self.fuse(x, nm, frames, mask)
        return x + u.head(_body(u, h, e))


def load_single(path, use_ema: bool = True) -> tuple[CondUNet, dict]:
    """A single-frame checkpoint (train.save) as its CondUNet, the EMA weights by default, and the checkpoint."""
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "single" in ck["config"]:
        raise ValueError(f"{path} is a burst checkpoint: load_burst")
    m = CondUNet(**ck["config"])
    m.load_state_dict(ck["ema" if use_ema else "model"])
    return m, ck


def load_burst(path, use_ema: bool = True) -> tuple[BurstNet, dict]:
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    c = ck["config"]
    m = BurstNet(c["single"], c["feat"], c["frames"])
    m.load_state_dict(ck["ema" if use_ema else "model"])
    return m, ck


# ---- synthetic bursts ---------------------------------------------------------------------------------------------
def crop_side(tile: int) -> int:
    """The clean crop a synthetic burst of `tile` px is rendered from."""
    return tile + 2 * (WARP_MARGIN + SYN_PAD)


def _linear_map(rng: np.random.Generator) -> np.ndarray:
    """A frame's linear map less the identity: a small rotation and scale."""
    th = math.radians(rng.uniform(-MAX_ROT_DEG, MAX_ROT_DEG))
    sc = 1.0 + rng.uniform(-MAX_SCALE, MAX_SCALE)
    return sc * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) - np.eye(2)


def render_burst(big: np.ndarray, a: DegradeParams, rng: np.random.Generator, tile: int, n: int, k: int,
                 is_hdr: bool = False, motion: bool = True) -> dict:
    """A synthetic burst of a clean crop `big` (at least crop_side(tile) px): the reference frame x and its restore
    target y (tile px, as data.render_pair makes them for a restore of state `a`), and k of n frame slots filled:
    `frames` (n, S, S, 3) float32 linear windows (S = tile + 2 WARP_MARGIN, the burst store's), `coords` (n, tile,
    tile, 2) (row, column) where each tile pixel falls in its frame's window, as the registration reports it, and
    `mask` (n,). The crop is prepared once (data.degrade.prepare_tile reads its statistics), then every frame is
    degraded by `a` with the preparation off. `motion` False renders frames with no transform, no moving rectangle,
    no gain jitter and no registration error (the tests' identity)."""
    W, L, S = WARP_MARGIN, crop_side(tile), tile + 2 * WARP_MARGIN
    if big.shape[0] < L or big.shape[1] < L:
        raise ValueError(f"a burst of {tile} px renders from a {L} px crop, got {big.shape[:2]}")
    prep = prepare_tile(np.asarray(big[:L, :L], dtype=np.float64), a, is_hdr)
    a0 = DegradeParams(**{**a.to_dict(), "expose": 0, "exposure_stops": 0.0, "scene_peak_over_median": 0.0})
    o = SYN_PAD
    win = prep[o:o + S, o:o + S]
    seed = int(rng.integers(1 << 62))
    x = degrade(win, a0, seed)[W:W + tile, W:W + tile]
    t = restore_target(a0)
    shade = input_shading(a0, S, S)
    y = degrade(win, t, seed, scene_gain=shade)[W:W + tile, W:W + tile]
    ref = y if shade is None else degrade(win, t, seed)[W:W + tile, W:W + tile]
    frames = np.zeros((n, S, S, 3), dtype=np.float32)
    coords = np.zeros((n, tile, tile, 2), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    uu = np.stack(np.meshgrid(np.arange(S, dtype=np.float64), np.arange(S, dtype=np.float64), indexing="ij"))
    pp = np.stack(np.meshgrid(np.arange(tile, dtype=np.float64), np.arange(tile, dtype=np.float64), indexing="ij"))
    cw = np.full((2, 1, 1), (S - 1) / 2.0)
    for j in range(min(k, n)):
        if motion:
            sh = rng.uniform(-MAX_SHIFT_PX, MAX_SHIFT_PX, 2).reshape(2, 1, 1)
            A = _linear_map(rng)
        else:
            sh, A = np.zeros((2, 1, 1)), np.zeros((2, 2))
        src = o + uu + sh + np.einsum("ij,jhw->ihw", A, uu - cw)        # frame j's window pixel u sees scene src(u)
        if motion and rng.random() < P_MOVER:
            hh, ww = (int(v) for v in rng.integers(MOVER_PX[0], MOVER_PX[1] + 1, 2))
            r0, c0 = int(rng.integers(0, S - hh + 1)), int(rng.integers(0, S - ww + 1))
            ang, mag = rng.uniform(0.0, 2.0 * math.pi), rng.uniform(*MOVER_SHIFT_PX)
            src[0, r0:r0 + hh, c0:c0 + ww] += mag * math.sin(ang)
            src[1, r0:r0 + hh, c0:c0 + ww] += mag * math.cos(ang)
        img = np.stack([map_coordinates(prep[..., c], src, order=3, mode="mirror") for c in range(3)], axis=-1)
        img = np.maximum(img, 0.0) * ((1.0 + rng.normal(0.0, GAIN_JITTER)) if motion else 1.0)
        frames[j] = degrade(img, a0, int(rng.integers(1 << 62)))
        # the tile pixel p is scene point o + W + p; frame j shows it at u with o + u + sh + A (u - cw) = o + W + p
        rhs = W + pp - sh + np.einsum("ij,jhw->ihw", A, np.broadcast_to(cw, pp.shape))
        u = np.einsum("ij,jhw->ihw", np.linalg.inv(np.eye(2) + A), rhs)
        if motion:
            u = u + rng.normal(0.0, REG_SIGMA_PX, 2).reshape(2, 1, 1)
        coords[j] = np.moveaxis(u, 0, -1)
        mask[j] = True
    return {"x": x, "y": y, "metrics": tile_metrics(y, ref), "restore": True, "a": a0, "frames": frames,
            "coords": coords, "mask": mask, "synthetic": True}


def synthetic_burst(src: TileSource, dists: Distributions, rng: np.random.Generator, tile: int, n: int) -> dict:
    """A synthetic burst on a clean tile of `src` (whose tile is crop_side(tile)), with a uniform count of 0 to n
    frames: the network sees every count, none included."""
    big, it, y0, x0 = src.random_tile(rng)
    a = draw_state(rng, dists, "capture" if rng.random() < P_BURST_CAPTURE else "iqa")
    s = render_burst(big, a, rng, tile, n, int(rng.integers(n + 1)), it.is_hdr)
    s.update(item=it, y0=y0, x0=x0)
    return s


def real_burst_sample(store, rng: np.random.Generator, n: int) -> dict:
    """A stored burst's tile (data.burststore.BurstStore) as a restore sample, as data.real_sample makes one of a real
    pair, with a uniform count of 0 to n of its frames kept, and the rotation and flip applied to the coordinates as
    an image (each tile pixel keeps where it falls in its frame's window; the windows are not turned)."""
    s = store.random_tile(rng, n=n)
    x, y, co = s["x"], s["y"], s["coords"]
    mask = s["mask"].copy()
    mask[int(rng.integers(n + 1)):] = False
    r = int(rng.integers(4))
    if r:
        x, y, co = np.rot90(x, r), np.rot90(y, r), np.rot90(co, r, axes=(1, 2))
    if rng.random() < 0.5:
        x, y, co = x[:, ::-1], y[:, ::-1], co[:, :, ::-1]
    x, y, co = np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(co)
    return {"x": x, "y": y, "metrics": tile_metrics(y, y), "restore": True, "capture": s["capture"], "real": True,
            "frames": s["frames"], "coords": co, "mask": mask, "burst": s["burst"], "pair": s["pair"],
            "y0": s["y0"], "x0": s["x0"], "clipped": s["clipped"]}


def no_frames(s: dict, tile: int, n: int) -> dict:
    """A single-frame sample with n empty frame slots."""
    side = tile + 2 * WARP_MARGIN
    s.update(frames=np.zeros((n, side, side, 3), dtype=np.float32),
             coords=np.zeros((n, tile, tile, 2), dtype=np.float32), mask=np.zeros(n, dtype=bool))
    return s


def to_tensors(s: dict, norm: Normaliser, rng: np.random.Generator) -> tuple:
    """(x, y encoded (3, T, T), cond, frames linear (n, 3, S, S), coords (n, T, T, 2), mask (n,), real)."""
    x = torch.from_numpy(encode(s["x"])).permute(2, 0, 1).contiguous()
    y = torch.from_numpy(encode(s["y"])).permute(2, 0, 1).contiguous()
    c = torch.from_numpy(cond_vector(norm, s, rng.random() < P_CAPTURE_KNOWN))
    fr = torch.from_numpy(np.ascontiguousarray(s["frames"], dtype=np.float32)).permute(0, 3, 1, 2).contiguous()
    return (x, y, c, fr, torch.from_numpy(np.ascontiguousarray(s["coords"], dtype=np.float32)),
            torch.from_numpy(np.asarray(s["mask"], dtype=bool)), bool(s.get("real", False)))


class BurstStream:
    """Infinite iterator of to_tensors tuples for one loader worker: with probability `p_real_burst` a stored burst's
    tile (data.burststore, split train), with `p_single` a single-frame sample of the ordinary stream (a rendered pair
    of `sources`, or with `p_real` a stored real pair of `real`) with no frames, else a synthetic burst on a clean
    tile of `sources`. The single-frame share keeps what the network does on its own sources and on edits."""
    def __init__(self, sources: list[str], tile: int, normaliser: Normaliser, seed: int, root: Path | None = None,
                 frames: int = 4, p_real_burst: float = 0.5, p_single: float = 0.1, real: tuple[str, ...] = (),
                 p_real: float = 0.0, weights: dict[str, float] | None = None, decoded_store: bool = True,
                 audit: bool = False, subpix_real: bool = False):
        self.sources, self.tile, self.norm, self.seed, self.root = list(sources), int(tile), normaliser, seed, root
        self.frames, self.p_real_burst, self.p_single = int(frames), float(p_real_burst), float(p_single)
        self.real, self.p_real, self.weights = tuple(real), float(p_real), weights
        self.decoded_store, self.audit, self.subpix_real = bool(decoded_store), bool(audit), bool(subpix_real)
        if self.p_real_burst + self.p_single > 1.0:
            raise ValueError(f"p_real_burst + p_single is at most 1, got {self.p_real_burst} + {self.p_single}")

    def __iter__(self):
        from torch.utils.data import get_worker_info
        wi = get_worker_info()
        wid = wi.id if wi else 0
        src = single = rstore = bstore = None
        if self.p_real_burst + self.p_single < 1.0:
            src = TileSource(self.sources, "train", crop_side(self.tile), self.root, tiles_per_image=TILES_PER_IMAGE,
                             weights=self.weights, store=self.decoded_store, audit=self.audit)
        if self.p_single > 0:
            single = TileSource(self.sources, "train", self.tile, self.root, tiles_per_image=TILES_PER_IMAGE,
                                weights=self.weights, store=self.decoded_store, audit=self.audit)
            if self.real and self.p_real > 0:
                from ..data.realstore import RealStore
                rstore = RealStore(list(self.real), "train", self.root, self.tile, subpix=self.subpix_real)
        if self.p_real_burst > 0:
            from ..data.burststore import BurstStore
            bstore = BurstStore("train", self.root, self.tile, self.frames, subpix=self.subpix_real)
        dists = Distributions.load(self.root)
        rng = np.random.default_rng(_h("burst-stream", self.seed, wid))
        while True:
            u = rng.random()
            if bstore is not None and u < self.p_real_burst:
                s = real_burst_sample(bstore, rng, self.frames)
            elif single is not None and u < self.p_real_burst + self.p_single:
                if rstore is not None and rng.random() < self.p_real:
                    s = real_sample(rstore, rng)
                else:
                    s = make_sample(single, dists, rng)
                s = no_frames(s, self.tile, self.frames)
            else:
                s = synthetic_burst(src, dists, rng, self.tile, self.frames)
            yield to_tensors(s, self.norm, rng)


def synthetic_fixed_set(sources: list[str], split: str, count: int, tile: int, frames: int, seed: int,
                        root: Path | None = None, audit: bool = False) -> list[dict]:
    """`count` reproducible synthetic bursts of a split's clean items, every one with all `frames` frames: the burst
    validation set. Scored with all frames, with none, and by the single network, it says what the frames add."""
    src, dists = TileSource(sources, split, crop_side(tile), root, audit=audit), Distributions.load(root)
    rng = np.random.default_rng(_h("fixed-burst", split, count, tile, frames, seed))
    out = []
    for _ in range(count):
        big, it, y0, x0 = src.random_tile(rng)
        a = draw_state(rng, dists, "capture" if rng.random() < P_BURST_CAPTURE else "iqa")
        s = render_burst(big, a, rng, tile, frames, frames, it.is_hdr)
        s.update(item=it, y0=y0, x0=x0)
        out.append(s)
    return out
