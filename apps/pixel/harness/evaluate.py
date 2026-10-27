"""One evaluation: build the world a claim names, then candidate vs camera-sim vs truth → gates → report."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import CaptureConstraints, simulate_sequence
from .claim import build_camera, build_display, build_scene, validate_claim
from .display import DisplaySpec, encode_roundtrip_frames, ppd, required_master_px
from .gate import run_gates
from .hallucination import invented_maps, region_breakdown
from .io import load_mask, save_frames, save_mask
from .metrics import measure_all
from .report import write_report
from .scene import Scene


@dataclass
class World:
    claim: dict
    scene: Scene
    camera_constraints: CaptureConstraints
    display: DisplaySpec
    truth: np.ndarray
    camera: np.ndarray
    truth_metrics: dict
    camera_metrics: dict


def build_world(claim: dict) -> World:
    validate_claim(claim)
    scene = build_scene(claim)
    c, seed = build_camera(claim)
    display = build_display(claim)
    truth = scene.truth_frames()
    camera = simulate_sequence(scene, c, seed)
    return World(claim, scene, c, display, truth, camera,
                 measure_all(truth, truth, scene, display), measure_all(camera, truth, scene, display))


def evaluate(world: World, candidate: np.ndarray, out_dir, candidate_name: str = "candidate",
             preference: dict | None = None, rendered_map=None) -> dict:
    """`rendered_map`: path or array of the candidate's rendered-pixel declaration (>0.5 = rendered), (H,W)
    or (T,H,W). None declares every pixel recovered, so all measured invention counts as undeclared; under
    claim rendered_map.policy = "required" it leaves the provenance checks without data instead."""
    policy = world.claim.get("rendered_map", {}).get("policy", "optional")
    declared = None if rendered_map is None else load_mask(rendered_map, candidate.shape[:3])
    frame_infos: list[dict] = []
    invented = invented_maps(candidate, world.truth, world.scene, frame_infos)
    cand = measure_all(candidate, world.truth, world.scene, world.display, invented=invented, declared=declared)
    if policy == "required" and declared is None:
        cand["rendered_fraction"] = cand["undeclared_invention_fraction"] = None
    if preference:
        cand.update(preference.get("metrics", {}))
    gates = run_gates(world.claim["gates"], cand, world.camera_metrics)
    delivery = None
    spec = world.display.delivery
    if spec:
        delivered, info = encode_roundtrip_frames(candidate, spec.get("format", "jpeg"),
                                                  spec.get("quality", 92), world.scene.spec.fps)
        dm = measure_all(delivered, world.truth, world.scene, world.display, declared=declared)
        delivery = {**info, "metrics": {
            m: {"delivered": dm[m],
                "delta": None if dm[m] is None or cand.get(m) is None else dm[m] - cand[m]}
            for m in dm}}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "invented_detail.npy", invented)
    save_mask(out / "invented_detail_0000.png", invented[0])
    provenance = {"policy": policy, "map_supplied": declared is not None,
                  "regions": region_breakdown(invented, declared, world.scene),
                  "per_frame": frame_infos,
                  "files": {"invented_map": "invented_detail.npy", "invented_png": "invented_detail_0000.png"}}
    return write_report(out_dir, world.claim, candidate_name, cand, world.camera_metrics,
                        world.truth_metrics, gates, delivery, preference, provenance)


def export_world(world: World, out_dir) -> None:
    """Truth and camera-sim frames (.npy master + PNG) and world.json: everything a generator may be told."""
    out = Path(out_dir)
    save_frames(out, world.truth, "truth")
    save_frames(out, world.camera, "camera")
    scene, spec = world.scene, world.scene.spec
    world_json = {
        "claim": world.claim,
        "display_ppd": ppd(world.display),
        "required_master_px": required_master_px(world.display),
        "frame_shape": list(world.truth.shape),
        "regions": {k: list(v) for k, v in scene.regions.items()},
        "wedge": [{"region": list(r), "value": v} for r, v in scene.wedge],
        "colors": [{"region": list(r), "linear_rgb": rgb.tolist(), "lab": lab.tolist()}
                   for r, rgb, lab in scene.colors],
        "landmarks": [{"radius": lm.radius, "color": list(lm.color), "vx_px_s": lm.vx, "vy_px_s": lm.vy,
                       "positions": [list(lm.position(spec.frame_time(k))) for k in range(spec.n_frames)]}
                      for lm in scene.landmarks],
        "camera": {"f_number": world.camera_constraints.f_number,
                   "pixel_pitch_um": world.camera_constraints.pixel_pitch_um,
                   "exposure_s": world.camera_constraints.exposure_s},
    }
    (out / "world.json").write_text(json.dumps(world_json, indent=2), encoding="utf-8")
    (out / "claim.json").write_text(json.dumps(world.claim, indent=2), encoding="utf-8")
