"""Mask-free metrics for arbitrary (degraded, clean) pairs. This is what the labeller
computes at scale: it needs no scene, only a clean source and its degraded copy.
"""

from __future__ import annotations

import numpy as np

from ..colorsci import luminance
from ..ops import gaussian_blur
from .artifacts import blocking_score, halo_score
from .color import delta_e_p95
from .spectral import mtf50, tiling_score


def psnr(cand: np.ndarray, ref: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(cand, dtype=np.float32) - np.asarray(ref, dtype=np.float32)) ** 2))
    return 99.0 if mse < 1e-12 else float(10.0 * np.log10(1.0 / mse))


def noise_sigma(cand: np.ndarray, ref: np.ndarray, sigma: float = 1.5) -> float:
    """Std of the high-pass error: isolates noise from blur and tone shifts."""
    d = luminance(cand) - luminance(ref)
    hp = d - gaussian_blur(d, sigma)
    return float(hp.std())


def clip_fraction(cand: np.ndarray, ref: np.ndarray) -> float:
    """Pixels the candidate drove to the rails that the reference kept inside them."""
    c = np.asarray(cand)
    r = np.asarray(ref)
    newly_white = (c >= 0.995).any(axis=-1) & (r < 0.98).all(axis=-1)
    newly_black = (c <= 0.005).any(axis=-1) & (r > 0.02).all(axis=-1)
    return float((newly_white | newly_black).mean())


def smooth_banding(cand: np.ndarray, ref: np.ndarray) -> float:
    """Banding in the reference's smoothest third (lowest gradient magnitude)."""
    lr = luminance(ref)
    gy, gx = np.gradient(lr)
    g = np.hypot(gx, gy)
    thresh = np.percentile(g, 33)
    mask = g <= thresh
    if mask.sum() < 64:
        return 0.0
    qc = np.unique(np.rint(luminance(cand)[mask] * 255.0)).size
    qr = np.unique(np.rint(lr[mask] * 255.0)).size
    return float(np.clip(1.0 - qc / max(qr, 1), 0.0, 1.0))


GENERIC_METRICS = (
    "psnr",
    "mtf50",
    "noise_sigma",
    "delta_e_p95",
    "blocking_score",
    "halo_score",
    "tiling_score",
    "clip_fraction",
    "banding_score",
)


def generic_metrics(cand: np.ndarray, ref: np.ndarray) -> dict:
    lc, lr = luminance(cand), luminance(ref)
    return {
        "psnr": psnr(cand, ref),
        "mtf50": mtf50(lc, lr),
        "noise_sigma": noise_sigma(cand, ref),
        "delta_e_p95": delta_e_p95(cand, ref),
        "blocking_score": blocking_score(cand, ref),
        "halo_score": halo_score(cand, ref),
        "tiling_score": tiling_score(lc),
        "clip_fraction": clip_fraction(cand, ref),
        "banding_score": smooth_banding(cand, ref),
    }
