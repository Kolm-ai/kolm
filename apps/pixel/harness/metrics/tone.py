"""Shadows, highlights and dynamic range on the 14-step wedge."""
from __future__ import annotations

import numpy as np

from ..colorspace import luminance
from ..scene import Scene, crop

PATCH_MARGIN = 2
SHADOW_MAX = 0.02
SNR_CAP_DB = 100.0
HIGHLIGHT_TOL = 0.95
SEPARATION_SIGMAS = 2.0


def _patch_stats(frame: np.ndarray, scene: Scene) -> list[tuple[float, float, float]]:
    """(truth value, mean, std) of luminance per wedge patch, in wedge order."""
    out = []
    for region, v in scene.wedge:
        p = luminance(crop(frame, region, PATCH_MARGIN))
        out.append((v, float(p.mean()), float(p.std())))
    return out


def shadow_snr_db(frame: np.ndarray, scene: Scene) -> float:
    snrs = []
    for v, m, s in _patch_stats(frame, scene):
        if v < SHADOW_MAX:
            snrs.append(SNR_CAP_DB if s < 1e-12 or m <= 0 else min(SNR_CAP_DB, 20 * np.log10(m / s)))
    return float(np.median(snrs))


def highlight_loss_fraction(frame: np.ndarray, scene: Scene) -> float:
    lost = []
    for region, v in scene.wedge:
        if v > 1.0:
            lost.append(luminance(crop(frame, region, PATCH_MARGIN)) < HIGHLIGHT_TOL * v)
    return float(np.concatenate([x.ravel() for x in lost]).mean())


def dr_steps(frame: np.ndarray, scene: Scene) -> float:
    """Adjacent wedge patches that remain ordered and separated by more than 2 sigma of pixel noise."""
    st = _patch_stats(frame, scene)
    n = 0
    for (_, m0, s0), (_, m1, s1) in zip(st[:-1], st[1:]):
        if m1 - m0 > max(SEPARATION_SIGMAS * max(s0, s1), 1e-9):
            n += 1
    return float(n)
