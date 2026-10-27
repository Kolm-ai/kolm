"""The claim tuple: load, validate, default, and build the scene / camera / display it names."""
from __future__ import annotations

import json
from pathlib import Path

from .camera import CaptureConstraints, from_dict
from .display import DisplaySpec
from .display import from_dict as display_from_dict
from .gate import OPS
from .metrics import REGISTRY
from .scene import Scene, SceneSpec

DEFAULT_QUALITIES = {
    "resolution": ["mtf50_cpp", "texture_hf_fidelity"],
    "shadows": ["shadow_snr_db"],
    "highlights": ["highlight_loss_fraction", "dr_steps"],
    "color": ["delta_e_mean", "delta_e_max"],
    "artifacts": ["edge_overshoot", "banding_score", "periodicity"],
    "geometry_lighting": ["landmark_error_px", "shading_rmse"],
    "temporal": ["flicker", "flow_warped_excess", "motion_region_error", "trajectory_error_px"],
    "provenance": ["invented_detail_fraction", "undeclared_invention_fraction", "rendered_fraction"],
}

RENDERED_MAP_POLICIES = ("optional", "required")  # required: no rendered-pixel map, no provenance verdict

DEFAULT_GATES = [
    {"name": "composition", "checks": [
        {"metric": "pref_convincing_p_beats_camera_low", "op": ">", "value": 0.5}]},
    {"name": "provenance", "checks": [
        {"metric": "undeclared_invention_fraction", "op": "<=", "value": 0.01}]},
    {"name": "lighting_materials", "checks": [
        {"metric": "shading_rmse", "op": "<=", "value": 0.02},
        {"metric": "landmark_error_px", "op": "<=", "value": 0.5}]},
    {"name": "tone_color", "checks": [
        {"metric": "delta_e_mean", "op": "<=", "value": 2.0},
        {"metric": "highlight_loss_fraction", "op": "<=", "value": 0.0},
        {"metric": "dr_steps", "op": "ties_or_beats_camera"}]},
    {"name": "artifacts", "checks": [
        {"metric": "edge_overshoot", "op": "<=", "value": 0.02},
        {"metric": "banding_score", "op": "<=", "value": 0.01},
        {"metric": "periodicity", "op": "<=", "value": 0.3}]},
    {"name": "resolution", "checks": [
        {"metric": "display_ppd", "op": ">=", "value": 94.0},
        {"metric": "mtf50_cpp", "op": "beats_camera"}]},
    {"name": "micro_detail", "checks": [
        {"metric": "texture_hf_fidelity", "op": "beats_camera"}]},
    {"name": "temporal", "checks": [
        {"metric": "flicker", "op": "beats_camera"},
        {"metric": "flow_warped_excess", "op": "<=", "value": 0.005},
        {"metric": "motion_region_error", "op": "beats_camera"}]},
]


def default_claim(video: bool = True) -> dict:
    scene = ({"width": 384, "height": 216, "supersample": 2, "seed": 7, "n_frames": 24, "fps": 24.0,
              "exposure_s": 0.0} if video else
             {"width": 768, "height": 432, "supersample": 4, "seed": 7, "n_frames": 1, "fps": 24.0,
              "exposure_s": 0.0})
    return {
        "qualities": {k: list(v) for k, v in DEFAULT_QUALITIES.items()},
        "scene": scene,
        "camera": {"preset": "best_case_earth", "exposure_s": 1 / 48, "seed": 1},
        "display": {"diagonal_in": 31.5, "width_px": 7680, "height_px": 4320, "distance_m": 0.6,
                    "delivery": {"format": "jpeg", "quality": 92}},
        "rendered_map": {"policy": "optional"},
        "gates": json.loads(json.dumps(DEFAULT_GATES)),
    }


def load_claim(path) -> dict:
    claim = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_claim(claim)
    return claim


def validate_claim(claim: dict) -> None:
    for key in ("qualities", "scene", "camera", "display", "gates"):
        if key not in claim:
            raise ValueError(f"claim is missing {key!r}; no tuple, no claim")
    for q, ms in claim["qualities"].items():
        for m in ms:
            if m not in REGISTRY:
                raise ValueError(f"quality {q!r} names unknown metric {m!r}")
    for g in claim["gates"]:
        for c in g["checks"]:
            if c["metric"] not in REGISTRY:
                raise ValueError(f"gate {g['name']!r} names unknown metric {c['metric']!r}")
            if c["op"] not in OPS:
                raise ValueError(f"gate {g['name']!r}: unknown op {c['op']!r}")
    policy = claim.get("rendered_map", {}).get("policy", "optional")
    if policy not in RENDERED_MAP_POLICIES:
        raise ValueError(f"rendered_map.policy must be one of {RENDERED_MAP_POLICIES}, not {policy!r}")
    from_dict(claim["camera"])  # raises on bad camera block
    display_from_dict(claim["display"])


def build_scene(claim: dict) -> Scene:
    return Scene(SceneSpec(**claim["scene"]))


def build_camera(claim: dict) -> tuple[CaptureConstraints, int]:
    return from_dict(claim["camera"])


def build_display(claim: dict) -> DisplaySpec:
    return display_from_dict(claim["display"])
