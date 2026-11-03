"""Colour accuracy: CIEDE2000 between each colour patch's mean and its known Lab."""
from __future__ import annotations

import numpy as np

from ..colorspace import delta_e_2000, linear_rgb_to_lab
from ..scene import Scene, crop

PATCH_MARGIN = 2


def delta_e_stats(frame: np.ndarray, scene: Scene) -> tuple[float, float]:
    """(mean, max) CIEDE2000 over the colour patches. Patch means, so noise averages out and bias stays."""
    des = []
    for region, _, lab_true in scene.colors:
        mean_rgb = crop(frame, region, PATCH_MARGIN).reshape(-1, 3).mean(axis=0)
        des.append(float(delta_e_2000(linear_rgb_to_lab(mean_rgb), lab_true)))
    return float(np.mean(des)), float(np.max(des))
