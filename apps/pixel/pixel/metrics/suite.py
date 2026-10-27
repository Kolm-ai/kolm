"""Scene-aware suite: the metrics a claim is judged on, with their better-direction and gate group."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ..colorsci import luminance
from ..scene import PATCH_COLORS, Scene
from .artifacts import blocking_score, halo_score
from .color import delta_e_p95, patch_delta_e
from .geometry import landmark_error_px
from .spectral import mtf50, tiling_score
from .temporal import flicker_score, motion_error_px, temporal_noise
from .tone import banding_score, highlight_loss_fraction, shadow_detail_corr, shadow_noise_sigma

HIGHER, LOWER = "higher", "lower"

# name -> (better direction, gate group, tie tolerance)
METRICS = OrderedDict(
    [
        ("landmark_error_px", (LOWER, "geometry", 0.05)),
        ("highlight_loss_fraction", (LOWER, "tone_color", 0.01)),
        ("shadow_noise_sigma", (LOWER, "tone_color", 0.0005)),
        ("shadow_detail_corr", (HIGHER, "tone_color", 0.02)),
        ("banding_score", (LOWER, "tone_color", 0.02)),
        ("patch_delta_e", (LOWER, "tone_color", 0.2)),
        ("delta_e_p95", (LOWER, "tone_color", 0.2)),
        ("halo_score", (LOWER, "artifacts", 0.003)),
        ("blocking_score", (LOWER, "artifacts", 0.1)),
        ("tiling_score", (LOWER, "artifacts", 0.1)),
        ("mtf50", (HIGHER, "resolution", 0.01)),
        ("flicker_score", (LOWER, "temporal", 0.0005)),
        ("temporal_noise", (LOWER, "temporal", 0.0005)),
        ("motion_error_px", (LOWER, "temporal", 0.03)),
    ]
)

GROUPS = OrderedDict()
for _name, (_dir, _group, _tol) in METRICS.items():
    GROUPS.setdefault(_group, []).append(_name)


def evaluate_still(cand: np.ndarray, ref: np.ndarray, scene: Scene, t: float = 0.0) -> dict:
    reg = scene.regions(t)
    m = reg.masks
    lc, lr = luminance(cand), luminance(ref)
    tex = m["texture"]
    ys, xs = np.where(tex)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    mean_err, max_err = landmark_error_px(cand, reg.landmarks)
    return {
        "landmark_error_px": mean_err,
        "landmark_error_max_px": max_err,
        "highlight_loss_fraction": highlight_loss_fraction(cand, ref, m["highlight"]),
        "shadow_noise_sigma": shadow_noise_sigma(cand, ref, m["shadow"]),
        "shadow_detail_corr": shadow_detail_corr(cand, ref, m["shadow"]),
        "banding_score": banding_score(cand, ref, m["gradient"]),
        "patch_delta_e": patch_delta_e(cand, reg.patch_masks, PATCH_COLORS),
        "delta_e_p95": delta_e_p95(cand, ref),
        "halo_score": halo_score(cand, ref),
        "blocking_score": blocking_score(cand, ref),
        "tiling_score": tiling_score(lc),
        "mtf50": mtf50(lc[y0:y1, x0:x1], lr[y0:y1, x0:x1]),
    }


def evaluate_sequence(cand_frames: list[np.ndarray], ref_frames: list[np.ndarray], scene: Scene) -> dict:
    """Per-frame metrics averaged over frames, plus the temporal metrics when there is more than one frame."""
    per_frame = [evaluate_still(c, r, scene, float(i)) for i, (c, r) in enumerate(zip(cand_frames, ref_frames))]
    out = {k: float(np.mean([pf[k] for pf in per_frame])) for k in per_frame[0]}
    out["landmark_error_max_px"] = float(np.max([pf["landmark_error_max_px"] for pf in per_frame]))
    if len(cand_frames) > 1:
        flows = [scene.flow(float(i)) for i in range(len(cand_frames) - 1)]
        out["flicker_score"] = flicker_score(cand_frames, ref_frames)
        out["temporal_noise"] = temporal_noise(cand_frames, ref_frames)
        out["motion_error_px"] = motion_error_px(cand_frames, flows)
    out["frames"] = len(cand_frames)
    return out
