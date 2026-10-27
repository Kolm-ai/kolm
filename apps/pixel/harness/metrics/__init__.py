"""Metric registry and the one entry point: measure_all(frames, truth, scene, display)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..display import DisplaySpec, ppd
from ..hallucination import fraction, invented_maps, undeclared
from ..scene import Scene
from . import artifacts, color, geometry, resolution, temporal, tone


@dataclass(frozen=True)
class MetricInfo:
    higher_is_better: bool
    tie_margin: float
    scope: str  # still | video | invention | provenance | display | human


REGISTRY: dict[str, MetricInfo] = {
    "mtf50_cpp": MetricInfo(True, 0.02, "still"),
    "texture_hf_fidelity": MetricInfo(True, 0.03, "still"),
    "shadow_snr_db": MetricInfo(True, 1.0, "still"),
    "highlight_loss_fraction": MetricInfo(False, 0.01, "still"),
    "dr_steps": MetricInfo(True, 0.5, "still"),
    "delta_e_mean": MetricInfo(False, 0.5, "still"),
    "delta_e_max": MetricInfo(False, 1.0, "still"),
    "edge_overshoot": MetricInfo(False, 0.01, "still"),
    "banding_score": MetricInfo(False, 0.002, "still"),
    "periodicity": MetricInfo(False, 0.05, "still"),
    "landmark_error_px": MetricInfo(False, 0.1, "still"),
    "shading_rmse": MetricInfo(False, 0.005, "still"),
    "flicker": MetricInfo(False, 0.001, "video"),
    "flow_warped_excess": MetricInfo(False, 0.002, "video"),
    "motion_region_error": MetricInfo(False, 0.005, "video"),
    "trajectory_error_px": MetricInfo(False, 0.1, "video"),
    "invented_detail_fraction": MetricInfo(False, 0.002, "invention"),
    "rendered_fraction": MetricInfo(False, 0.0, "provenance"),
    "undeclared_invention_fraction": MetricInfo(False, 0.002, "provenance"),
    "display_ppd": MetricInfo(True, 0.0, "display"),
    "pref_beautiful_p_beats_camera_low": MetricInfo(True, 0.0, "human"),
    "pref_convincing_p_beats_camera_low": MetricInfo(True, 0.0, "human"),
}

STILL_METRICS = [m for m, i in REGISTRY.items() if i.scope == "still"]
VIDEO_METRICS = [m for m, i in REGISTRY.items() if i.scope == "video"]


def verdict(name: str, candidate: float, camera: float) -> tuple[str, float]:
    """'beats' | 'ties' | 'loses' and the signed margin (positive = candidate better)."""
    info = REGISTRY[name]
    diff = candidate - camera if info.higher_is_better else camera - candidate
    if abs(diff) <= info.tie_margin:
        return "ties", float(diff)
    return ("beats" if diff > 0 else "loses"), float(diff)


def still_metrics(frame: np.ndarray, truth: np.ndarray, scene: Scene, k: int) -> dict[str, float]:
    de_mean, de_max = color.delta_e_stats(frame, scene)
    return {
        "mtf50_cpp": resolution.mtf50_edge(frame, scene),
        "texture_hf_fidelity": resolution.texture_hf_fidelity(frame, truth, scene),
        "shadow_snr_db": tone.shadow_snr_db(frame, scene),
        "highlight_loss_fraction": tone.highlight_loss_fraction(frame, scene),
        "dr_steps": tone.dr_steps(frame, scene),
        "delta_e_mean": de_mean,
        "delta_e_max": de_max,
        "edge_overshoot": artifacts.edge_overshoot(frame, scene),
        "banding_score": artifacts.banding_score(frame, scene),
        "periodicity": artifacts.periodicity(frame, scene),
        "landmark_error_px": geometry.landmark_error_px(frame, scene, k),
        "shading_rmse": geometry.shading_rmse(frame, truth, scene, k),
    }


def video_metrics(frames: np.ndarray, truth: np.ndarray, scene: Scene) -> dict[str, float]:
    return {
        "flicker": temporal.flicker(frames, scene),
        "flow_warped_excess": temporal.flow_warped_excess(frames, truth, scene),
        "motion_region_error": temporal.motion_region_error(frames, truth, scene),
        "trajectory_error_px": temporal.trajectory_error_px(frames, scene),
    }


def measure_all(frames: np.ndarray, truth: np.ndarray, scene: Scene,
                display: DisplaySpec | None = None, invented: np.ndarray | None = None,
                declared: np.ndarray | None = None) -> dict[str, float | None]:
    """`invented`: precomputed bool (T,H,W) invented-detail maps (else measured here). `declared`: the
    candidate's rendered-pixel map, (H,W) or (T,H,W); None declares every pixel recovered."""
    if frames.shape != truth.shape:
        raise ValueError(f"candidate shape {frames.shape} != truth shape {truth.shape}; "
                         "the harness never resamples - render the candidate at the claim's size")
    per = [still_metrics(frames[k], truth[k], scene, k) for k in range(frames.shape[0])]
    out: dict[str, float | None] = {m: float(np.nanmean([p[m] for p in per])) for m in STILL_METRICS}
    out.update({m: None for m in VIDEO_METRICS})
    if frames.shape[0] >= 2:
        out.update(video_metrics(frames, truth, scene))
    if invented is None:
        invented = invented_maps(frames, truth, scene)
    if declared is not None:
        declared = np.broadcast_to(np.asarray(declared).astype(bool), invented.shape)
    out["invented_detail_fraction"] = fraction(invented)
    out["rendered_fraction"] = 0.0 if declared is None else fraction(declared)
    out["undeclared_invention_fraction"] = fraction(undeclared(invented, declared))
    out["display_ppd"] = ppd(display) if display is not None else None
    return out
