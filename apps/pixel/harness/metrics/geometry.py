"""Geometry and lighting on the stage: landmark centroids vs known centres, shading vs Lambertian truth."""
from __future__ import annotations

import math

import numpy as np

from ..colorspace import luminance
from ..scene import DISK_THRESHOLD, Scene

WINDOW_PAD_PX = 2.0  # the bounce boxes keep every disk >= 2 px inside its own column
SHADING_SHRINK_PX = 1.5  # ignore the anti-aliased rim when comparing shading


def detect_landmark(lum: np.ndarray, cx: float, cy: float, r: float,
                    threshold: float = DISK_THRESHOLD) -> tuple[float, float] | None:
    """Centroid (pixel-centre convention) of above-threshold pixels within r + pad of the expected centre."""
    H, W = lum.shape
    pad = r + WINDOW_PAD_PX
    y0, y1 = max(0, math.floor(cy - pad)), min(H, math.ceil(cy + pad) + 1)
    x0, x1 = max(0, math.floor(cx - pad)), min(W, math.ceil(cx + pad) + 1)
    m = lum[y0:y1, x0:x1] > threshold
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return float(xs.mean() + x0 + 0.5), float(ys.mean() + y0 + 0.5)


def landmark_error_px(frame: np.ndarray, scene: Scene, k: int) -> float:
    """Mean distance between detected and true disk centres at frame k. A missing disk counts as 2r."""
    lum = luminance(frame)
    errs = []
    for lm, (cx, cy) in zip(scene.landmarks, scene.landmark_positions(scene.spec.frame_time(k))):
        p = detect_landmark(lum, cx, cy, lm.radius)
        errs.append(2 * lm.radius if p is None else math.hypot(p[0] - cx, p[1] - cy))
    return float(np.mean(errs))


def shading_rmse(frame: np.ndarray, truth: np.ndarray, scene: Scene, k: int) -> float:
    """RMSE over RGB inside the disks (rim excluded) against the known Lambertian shading."""
    m = scene.moving_mask(scene.spec.frame_time(k), grow=-SHADING_SHRINK_PX)
    d = frame[m] - truth[m]
    return float(np.sqrt(np.mean(d * d)))
