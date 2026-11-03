"""Temporal metrics driven by the scene's analytic motion. Flow is known, never estimated."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from ..colorspace import luminance
from ..scene import Scene
from .geometry import landmark_error_px

MOTION_GROW_PX = 3.0  # dilate the moving masks so motion blur and halos around movers count


def warp_back(frame_next: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Sample frame k+1 at x + flow(x): what frame k should look like if the motion is honoured."""
    H, W = flow.shape[:2]
    Y, X = np.mgrid[0:H, 0:W].astype(np.float64)
    coords = [Y + flow[..., 1], X + flow[..., 0]]
    out = np.empty_like(frame_next)
    for ch in range(frame_next.shape[-1]):
        out[..., ch] = map_coordinates(frame_next[..., ch], coords, order=1, mode="nearest")
    return out


def flow_warped_rmse(frames: np.ndarray, scene: Scene) -> float:
    errs = []
    for k in range(frames.shape[0] - 1):
        valid = scene.flow_valid(k)
        d = (warp_back(frames[k + 1], scene.flow(k)) - frames[k])[valid]
        errs.append(np.mean(d * d))
    return float(np.sqrt(np.mean(errs)))


def flow_warped_excess(frames: np.ndarray, truth: np.ndarray, scene: Scene) -> float:
    """Candidate self-inconsistency under the true flow, beyond the truth's own (anti-aliased rims)."""
    return flow_warped_rmse(frames, scene) - flow_warped_rmse(truth, scene)


def flicker(frames: np.ndarray, scene: Scene) -> float:
    """Mean temporal std of luminance over pixels no disk ever touches, relative to their mean."""
    lum = luminance(frames)[:, scene.static_mask_all()]
    sd = lum.std(axis=0)
    sd[sd < 1e-12] = 0.0  # identical frames: kill the float residual of mean-subtraction
    return float(sd.mean() / max(lum.mean(), 1e-12))


def motion_region_error(frames: np.ndarray, truth: np.ndarray, scene: Scene) -> float:
    errs = []
    for k in range(frames.shape[0]):
        m = scene.moving_mask(scene.spec.frame_time(k), grow=MOTION_GROW_PX)
        d = (frames[k] - truth[k])[m]
        errs.append(np.mean(d * d))
    return float(np.sqrt(np.mean(errs)))


def trajectory_error_px(frames: np.ndarray, scene: Scene) -> float:
    """Worst-frame landmark error: one jump or drift anywhere in the clip shows up here."""
    return float(max(landmark_error_px(frames[k], scene, k) for k in range(frames.shape[0])))
