"""Colour error in CIEDE2000."""

from __future__ import annotations

import numpy as np

from ..colorsci import delta_e2000, display_to_lab, linear_to_lab, srgb_decode, tonemap


def patch_delta_e(cand: np.ndarray, patch_masks: list[np.ndarray], patch_colors_linear: np.ndarray) -> float:
    """Mean dE2000 of each patch's mean colour against its reference colour (after the shared tonemap)."""
    errs = []
    ref_lab = linear_to_lab(tonemap(np.asarray(patch_colors_linear, dtype=np.float32)))
    for mask, lab_ref in zip(patch_masks, ref_lab):
        if not mask.any():
            continue
        mean_rgb = np.asarray(cand, dtype=np.float32)[mask].mean(axis=0)
        lab = linear_to_lab(srgb_decode(mean_rgb))
        errs.append(delta_e2000(lab, lab_ref)[0])
    return float(np.mean(errs)) if errs else 0.0


def delta_e_p95(cand: np.ndarray, ref: np.ndarray, stride: int = 2) -> float:
    """95th percentile per-pixel dE2000 over the frame (subsampled for speed)."""
    c = display_to_lab(np.asarray(cand)[::stride, ::stride])
    r = display_to_lab(np.asarray(ref)[::stride, ::stride])
    return float(np.percentile(delta_e2000(c, r), 95))
