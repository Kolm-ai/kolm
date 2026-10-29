"""Inference: load a checkpoint (EMA weights) and run frames through the recovered layer.

Default conditioning is the mean restore target seen at calibration; `target_metrics` conditions on any
other state (an edit). Frames up to MAX_SIDE run whole (the network is fully convolutional); larger ones
are tiled with an overlap and a raised-cosine blend. Inference runs in fp32 unless `bf16` is asked for: the
network is residual, and a bf16 residual steps smooth gradients (banding)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import Normaliser, UNKNOWN_CAPTURE, capture_descriptor, decode, encode
from .net import CondUNet, pad_to

MAX_SIDE = 1024
TILE = 512
OVERLAP = 64


@dataclass
class Bundle:
    model: CondUNet
    normaliser: Normaliser
    cond_restore: np.ndarray
    step: int
    device: torch.device
    bf16: bool = False


def load_model(ckpt, device: str | None = None, bf16: bool = False) -> Bundle:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(Path(ckpt), map_location="cpu", weights_only=False)
    model = CondUNet(**ck["config"])
    model.load_state_dict(ck["ema"])
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return Bundle(model, Normaliser.from_state(ck["normaliser"]), np.array(ck["cond_restore"], dtype=np.float32),
                  int(ck["step"]), dev, bf16)


def _window(n: int, edge: int) -> np.ndarray:
    w = np.ones(n)
    if edge > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(edge) + 0.5) / edge)
        w[:edge], w[-edge:] = ramp, ramp[::-1]
    return w


@torch.no_grad()
def _run(b: Bundle, x: np.ndarray, cond: np.ndarray) -> np.ndarray:
    """One frame, (H, W, 3) log domain -> (H, W, 3) log domain."""
    t = torch.from_numpy(x).permute(2, 0, 1)[None].to(b.device)
    t, (ph, pw) = pad_to(t, 2 ** (b.model.n_levels - 1))     # each level halves the side
    c = torch.from_numpy(cond)[None].to(b.device)
    with torch.autocast(b.device.type, dtype=torch.bfloat16, enabled=b.bf16 and b.device.type == "cuda"):
        y = b.model(t, c)
    y = y.float()[0].permute(1, 2, 0).cpu().numpy()
    return y[:y.shape[0] - ph or None, :y.shape[1] - pw or None]


def full_cond(b: Bundle, cond: np.ndarray | None, capture: np.ndarray | None) -> np.ndarray:
    """The metric conditioning plus the capture descriptor the checkpoint expects (older checkpoints take none;
    an unknown capture is UNKNOWN_CAPTURE)."""
    cond = b.cond_restore if cond is None else np.asarray(cond, dtype=np.float32)
    want = int(b.model.config["cond_dim"])
    if cond.shape[0] < want:
        cond = np.concatenate([cond, UNKNOWN_CAPTURE if capture is None else np.asarray(capture, dtype=np.float32)])
    assert cond.shape[0] == want, (cond.shape, want)
    return cond


def capture_of_constraints(c) -> np.ndarray:
    """The descriptor of a camera-sim frame scaled to its own clip (camera_white_level): the clip is the full well."""
    return capture_descriptor(c.airy_first_zero_um(550.0) / c.pixel_pitch_um, c.residual_blur_um / c.pixel_pitch_um,
                              c.full_well_e, c.read_noise_e, c.bit_depth, c.bayer)


def restore_frame(b: Bundle, frame: np.ndarray, cond: np.ndarray | None = None,
                  capture: np.ndarray | None = None) -> np.ndarray:
    cond = full_cond(b, cond, capture)
    x = encode(frame)
    h, w = x.shape[:2]
    if max(h, w) <= MAX_SIDE:
        return np.maximum(decode(_run(b, x, cond)), 0.0)
    out, wsum = np.zeros((h, w, 3)), np.zeros((h, w, 1))
    step = TILE - OVERLAP
    ys = sorted(set(list(range(0, max(1, h - TILE), step)) + [max(0, h - TILE)]))
    xs = sorted(set(list(range(0, max(1, w - TILE), step)) + [max(0, w - TILE)]))
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(h, y0 + TILE), min(w, x0 + TILE)
            part = _run(b, np.ascontiguousarray(x[y0:y1, x0:x1]), cond)
            win = _window(y1 - y0, OVERLAP // 2)[:, None] * _window(x1 - x0, OVERLAP // 2)[None, :]
            out[y0:y1, x0:x1] += part * win[..., None]
            wsum[y0:y1, x0:x1] += win[..., None]
    return np.maximum(decode(out / wsum), 0.0)


def restore_frames(b: Bundle, frames: np.ndarray, target_metrics: dict | None = None,
                   white_level: float = 1.0, capture: np.ndarray | None = None) -> np.ndarray:
    """(T, H, W, 3) linear -> (T, H, W, 3) linear float64. `white_level` is the input's clip level in its own
    units (the sampler clips at 1.0; a camera-sim frame clips at white_radiance / white_fill): the model sees
    1.0 as the clip, so frames are scaled into that domain and back."""
    cond = b.cond_restore if target_metrics is None else b.normaliser(target_metrics)
    frames = np.asarray(frames, dtype=np.float64) / white_level
    return np.stack([restore_frame(b, f, cond, capture) for f in frames]) * white_level


def camera_white_level(world) -> float:
    """The clip level of a World's camera-sim frames in radiance units."""
    c = world.camera_constraints
    if c.scene_white_cd_m2 > 0:
        return c.full_well_e / c.electrons_per_unit_radiance()
    white = c.white_radiance if c.white_radiance is not None else world.scene.peak_radiance()
    return white / c.white_fill
