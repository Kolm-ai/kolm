"""Shadows, highlights and smooth gradients."""

from __future__ import annotations

import numpy as np

from ..colorsci import luminance
from ..ops import box_blur


def shadow_noise_sigma(cand: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    """Std of the candidate error inside the shadow region, display units (1/255 = one code value)."""
    d = (np.asarray(cand, dtype=np.float32) - np.asarray(ref, dtype=np.float32))[mask]
    return float(d.std()) if d.size else 0.0


def shadow_detail_corr(cand: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    """Correlation of candidate and reference luminance inside the shadow region (1 = detail intact)."""
    c = luminance(cand)[mask]
    r = luminance(ref)[mask]
    if c.size < 8 or r.std() < 1e-9:
        return 1.0
    if c.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(c, r)[0, 1])


def _local_std(l: np.ndarray, size: int) -> np.ndarray:
    m = box_blur(l, size)
    v = box_blur(l * l, size) - m * m
    return np.sqrt(np.maximum(v, 0.0))


def highlight_loss_fraction(cand: np.ndarray, ref: np.ndarray, mask: np.ndarray, flat: float = 0.3 / 255.0, textured: float = 1.0 / 255.0, size: int = 9) -> float:
    """Fraction of highlight pixels the candidate rendered flat (local std below a third of a code value)
    where the reference carries at least one code value of local texture. Measures clipping, not blur."""
    sc = _local_std(luminance(cand), size)
    sr = _local_std(luminance(ref), size)
    lost = (sc < flat) & (sr > textured)
    return float(lost[mask].mean()) if mask.any() else 0.0


def banding_score(cand: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    """1 - (distinct 8-bit levels in candidate / in reference) within a smooth region, clipped to [0, 1]."""
    qc = np.unique(np.rint(np.asarray(cand)[mask] * 255.0).astype(np.int32), axis=0).shape[0]
    qr = np.unique(np.rint(np.asarray(ref)[mask] * 255.0).astype(np.int32), axis=0).shape[0]
    if qr == 0:
        return 0.0
    return float(np.clip(1.0 - qc / qr, 0.0, 1.0))
