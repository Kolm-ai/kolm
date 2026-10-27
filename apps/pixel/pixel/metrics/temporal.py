"""Video-only metrics: flicker, temporal noise, motion consistency."""

from __future__ import annotations

import numpy as np

from ..colorsci import luminance
from ..ops import gaussian_blur, shift_subpixel


def flicker_score(cand_frames: list[np.ndarray], ref_frames: list[np.ndarray]) -> float:
    """Std over time of the candidate's mean-luminance error (0 = no flicker)."""
    d = [float(luminance(c).mean() - luminance(r).mean()) for c, r in zip(cand_frames, ref_frames)]
    return float(np.std(d)) if len(d) > 1 else 0.0


def temporal_noise(cand_frames: list[np.ndarray], ref_frames: list[np.ndarray]) -> float:
    """Mean per-pixel std over time of the candidate error (frame-to-frame boiling)."""
    if len(cand_frames) < 2:
        return 0.0
    err = np.stack([luminance(c) - luminance(r) for c, r in zip(cand_frames, ref_frames)], axis=0)
    return float(err.std(axis=0).mean())


def _integer_shift(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    H, W = a.shape
    w = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    A = np.fft.fft2((a - a.mean()) * w)
    B = np.fft.fft2((b - b.mean()) * w)
    cross = B * np.conj(A)
    cps = cross / np.maximum(np.abs(cross), 1e-9)
    corr = np.fft.ifft2(cps).real
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    return int((px + W // 2) % W - W // 2), int((py + H // 2) % H - H // 2)


def estimate_shift(a: np.ndarray, b: np.ndarray, iters: int = 6, prefilter_sigma: float = 1.5) -> tuple[float, float]:
    """Global translation from frame ``a`` to ``b`` (content moved by +d).

    Integer part by phase correlation, sub-pixel part by symmetric Lucas-Kanade
    on low-passed frames: both frames are warped half way so interpolation blur
    cancels, and the low-pass keeps near-Nyquist content from biasing the
    gradients. Unbiased for a pure shift.
    """
    a = gaussian_blur(np.asarray(a, dtype=np.float32), prefilter_sigma)
    b = gaussian_blur(np.asarray(b, dtype=np.float32), prefilter_sigma)
    H, W = a.shape
    ix, iy = _integer_shift(a, b)
    dx, dy = float(ix), float(iy)
    m = int(np.ceil(max(abs(dx), abs(dy)))) + 8
    if H <= 2 * m + 8 or W <= 2 * m + 8:
        return dx, dy
    for _ in range(iters):
        aw = shift_subpixel(a, dx / 2.0, dy / 2.0)[m:-m, m:-m]
        bw = shift_subpixel(b, -dx / 2.0, -dy / 2.0)[m:-m, m:-m]
        gy_a, gx_a = np.gradient(aw)
        gy_b, gx_b = np.gradient(bw)
        gx, gy = 0.5 * (gx_a + gx_b), 0.5 * (gy_a + gy_b)
        e = bw - aw
        M = np.array([[(gx * gx).sum(), (gx * gy).sum()], [(gx * gy).sum(), (gy * gy).sum()]])
        v = np.array([(gx * e).sum(), (gy * e).sum()])
        try:
            delta = np.linalg.solve(M, v)
        except np.linalg.LinAlgError:
            break
        dx -= float(delta[0])
        dy -= float(delta[1])
        if np.abs(delta).max() < 1e-4:
            break
    return dx, dy


def motion_error_px(cand_frames: list[np.ndarray], flows: list[tuple[float, float]]) -> float:
    """Mean |estimated - analytic| frame-to-frame translation, in pixels."""
    if len(cand_frames) < 2:
        return 0.0
    errs = []
    for i in range(len(cand_frames) - 1):
        ex, ey = estimate_shift(luminance(cand_frames[i]), luminance(cand_frames[i + 1]))
        fx, fy = flows[i]
        errs.append(float(np.hypot(ex - fx, ey - fy)))
    return float(np.mean(errs))
