"""Landmark localisation."""

from __future__ import annotations

import numpy as np

from ..colorsci import luminance


def landmark_error_px(cand: np.ndarray, positions: np.ndarray, radius: int = 5) -> tuple[float, float]:
    """Mean and max distance between expected landmark centres and candidate centroids."""
    lum = luminance(cand)
    H, W = lum.shape
    errs = []
    for x, y in np.asarray(positions, dtype=np.float32):
        cx, cy = int(round(float(x))), int(round(float(y)))
        y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)
        win = lum[y0:y1, x0:x1]
        if win.size == 0:
            errs.append(float(radius))
            continue
        w = np.clip(win - np.median(win), 0.0, None)
        if w.sum() <= 1e-9:
            errs.append(float(radius))
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mx = float((w * xx).sum() / w.sum())
        my = float((w * yy).sum() / w.sum())
        errs.append(float(np.hypot(mx - x, my - y)))
    return (float(np.mean(errs)), float(np.max(errs))) if errs else (0.0, 0.0)
