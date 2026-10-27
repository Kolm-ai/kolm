"""Run a claim end to end: scene -> reference -> camera floor -> candidate -> metrics -> gates -> delivery."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .camera import BestCaseCamera, motion_px_for_exposure
from .claim import Claim
from .colorsci import display_encode
from .display import Display
from .gate import FAIL, PASS, compare, run_gates, verdict
from .metrics.suite import METRICS, evaluate_sequence
from .ops import decode_image, encode_image, load_image
from .scene import Scene, SceneSpec

ENVELOPE_EVS = (-3.0, -2.0, -1.0, 0.0)


def _load_candidate_frames(spec: dict, n_frames: int, size: tuple[int, int]) -> list[np.ndarray]:
    path = Path(spec["path"])
    if path.is_dir():
        pattern = spec.get("pattern", "frame_{i:04d}.png")
        files = [path / pattern.format(i=i) for i in range(n_frames)]
    else:
        files = [path]
    frames = []
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"candidate frame missing: {f}")
        img = load_image(f)
        if img.shape[:2] != size:
            raise ValueError(f"candidate frame {f} is {img.shape[1]}x{img.shape[0]}, claim scene is {size[1]}x{size[0]}")
        frames.append(img)
    if len(frames) != n_frames:
        raise ValueError(f"claim has {n_frames} frames, candidate supplied {len(frames)}")
    return frames


def _deliver(frames: list[np.ndarray], delivery: dict) -> list[np.ndarray]:
    fmt = str(delivery.get("format", "png")).lower()
    q = int(delivery.get("quality", 92))
    return [decode_image(encode_image(f, fmt, q)) for f in frames]


def camera_frames(scene: Scene, cam: BestCaseCamera, truth: list[np.ndarray], exposure_ev: float, motion_px, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [display_encode(cam.capture_linear(t, exposure_ev=exposure_ev, motion_px=motion_px, rng=rng)) for t in truth]


def evaluate_claim(claim: Claim, out_dir: str | Path | None = None, save_frames: bool = False, envelope_evs=ENVELOPE_EVS) -> dict:
    spec = SceneSpec.from_dict(claim.scene)
    scene = Scene(spec)
    display = Display(**{k: claim.display[k] for k in ("diagonal_in", "width_px", "height_px", "distance_m")})
    cam = BestCaseCamera.from_dict(claim.capture.get("camera"))
    exposure_s = float(claim.capture["exposure_s"])
    exposure_ev = float(claim.capture.get("exposure_ev", 0.0))
    motion_px = motion_px_for_exposure(tuple(claim.capture.get("motion_px_per_s", (0.0, 0.0))), exposure_s)

    truth = [scene.render(float(i)) for i in range(spec.frames)]
    ref = [display_encode(t) for t in truth]

    result = {
        "claim": claim.to_dict(),
        "provable": claim.provable,
        "display": display.summary(),
        "camera": {**cam.summary(), "exposure_ev": exposure_ev, "exposure_s": exposure_s, "motion_px": list(motion_px)},
        "scene": asdict(spec),
    }
    if not claim.provable:
        result.update(
            supported=False,
            verdict="NOT PROVABLE: regime 'record_of_reality' cannot be certified by measurement against a synthetic truth",
            gates=[],
            metrics={},
        )
        return result

    cam_at_claim = camera_frames(scene, cam, truth, exposure_ev, motion_px, seed=1)
    m_camera = evaluate_sequence(cam_at_claim, ref, scene)

    # Envelope: the best the camera reaches on each metric over an exposure sweep.
    sweep = {}
    for ev in envelope_evs:
        frames = cam_at_claim if ev == exposure_ev else camera_frames(scene, cam, truth, ev, motion_px, seed=1)
        sweep[ev] = evaluate_sequence(frames, ref, scene)
    envelope = {}
    for name, (direction, _g, _t) in METRICS.items():
        vals = [m[name] for m in sweep.values() if name in m]
        if vals:
            envelope[name] = float(max(vals) if direction == "higher" else min(vals))

    kind = claim.candidate["kind"]
    if kind == "truth":
        cand = ref
    elif kind == "camera":
        cand = cam_at_claim
    else:
        cand = _load_candidate_frames(claim.candidate, spec.frames, (spec.height, spec.width))
    m_cand = evaluate_sequence(cand, ref, scene)
    m_truth = evaluate_sequence(ref, ref, scene)

    gates = run_gates(m_cand, envelope, claim.qualities, claim.is_video)
    ok, why = verdict(gates)

    delivered = _deliver(cand, claim.delivery)
    m_deliv = evaluate_sequence(delivered, ref, scene)
    gates_deliv = run_gates(m_deliv, envelope, claim.qualities, claim.is_video)
    ok_deliv, why_deliv = verdict(gates_deliv)

    comparison = {}
    for name in METRICS:
        if name in m_cand:
            comparison[name] = {
                "candidate": m_cand[name],
                "delivered": m_deliv[name],
                "camera_at_claim": m_camera[name],
                "camera_envelope": envelope.get(name),
                "truth": m_truth[name],
                "vs_camera": compare(name, m_cand[name], m_camera[name]),
                "vs_envelope": compare(name, m_cand[name], envelope[name]) if name in envelope else "n/a",
                "delivered_vs_envelope": compare(name, m_deliv[name], envelope[name]) if name in envelope else "n/a",
            }

    result.update(
        metrics=m_cand,
        metrics_delivered=m_deliv,
        metrics_camera=m_camera,
        metrics_truth=m_truth,
        camera_sweep={str(ev): m for ev, m in sweep.items()},
        envelope=envelope,
        comparison=comparison,
        gates=[asdict(g) for g in gates],
        gates_delivered=[asdict(g) for g in gates_deliv],
        supported=bool(ok and ok_deliv),
        supported_master=bool(ok),
        supported_delivered=bool(ok_deliv),
        verdict=(
            f"SUPPORTED on {claim.qualities} (master and delivered as {claim.delivery})"
            if ok and ok_deliv
            else f"NOT SUPPORTED: master {why}" if not ok else f"SUPPORTED on the master, NOT after delivery as {claim.delivery}: {why_deliv}"
        ),
        wins=[n for n, c in comparison.items() if c["vs_envelope"] == "beats"],
        wins_delivered=[n for n, c in comparison.items() if c["delivered_vs_envelope"] == "beats"],
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        if save_frames:
            for i, (r, c, k) in enumerate(zip(ref, cand, cam_at_claim)):
                for tag, img in (("reference", r), ("candidate", c), ("camera", k)):
                    (out / f"{tag}_{i:04d}.png").write_bytes(encode_image(img, "png"))
    return result
