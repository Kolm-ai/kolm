import json

from pixel.claim import Claim
from pixel.evaluate import evaluate_claim
from pixel.ops import save_image
from pixel.report import render_report

STILL = {
    "claim_id": "pipe-still",
    "regime": "synthetic_world",
    "qualities": ["geometry", "tone_color", "artifacts", "resolution"],
    "scene": {"seed": 7, "width": 192, "height": 192, "frames": 1},
    "capture": {"exposure_s": 0.01, "exposure_ev": 0.0, "motion_px_per_s": [120.0, 0.0]},
    "display": {"diagonal_in": 31.5, "width_px": 7680, "height_px": 4320, "distance_m": 0.6},
    "candidate": {"kind": "truth"},
    "delivery": {"format": "jpeg", "quality": 92},
}


def test_truth_supported_camera_fails_tone(tmp_path):
    truth = evaluate_claim(Claim.from_dict({**STILL, "delivery": {"format": "png"}}), out_dir=tmp_path / "truth", envelope_evs=(-3.0, 0.0))
    assert truth["supported"], truth["verdict"]
    assert {"mtf50", "shadow_noise_sigma", "shadow_detail_corr"} <= set(truth["wins"])  # vs the camera at its best exposure per metric
    assert truth["comparison"]["highlight_loss_fraction"]["vs_camera"] == "beats"  # vs the camera at the claim exposure
    assert truth["comparison"]["highlight_loss_fraction"]["vs_envelope"] == "ties"  # the camera can protect highlights at -3 EV, at a shadow cost
    jpeg = evaluate_claim(Claim.from_dict(STILL), envelope_evs=(-3.0, 0.0))
    assert jpeg["supported_master"] and set(jpeg["wins_delivered"]) <= set(jpeg["wins"])
    assert "shadow_noise_sigma" in set(jpeg["wins"]) - set(jpeg["wins_delivered"])  # q92 costs the shadow win
    assert (tmp_path / "truth" / "result.json").exists()
    cam = evaluate_claim(Claim.from_dict({**STILL, "candidate": {"kind": "camera"}}), envelope_evs=(-3.0, 0.0))
    assert not cam["supported"]
    failed = [g["name"] for g in cam["gates"] if g["status"] == "FAIL"]
    assert failed == ["tone_color"]
    assert all(c["vs_camera"] == "ties" for c in cam["comparison"].values())
    md = render_report(cam)
    assert "NOT SUPPORTED" in md and "| mtf50 |" in md


def test_delivery_can_cost_a_win(tmp_path):
    harsh = evaluate_claim(Claim.from_dict({**STILL, "delivery": {"format": "jpeg", "quality": 20}}), envelope_evs=(0.0,))
    assert set(harsh["wins"]) - set(harsh["wins_delivered"])  # at least one win did not survive q20
    assert harsh["supported_master"] and not harsh["supported"] and "after delivery" in harsh["verdict"]


def test_video_claim_temporal_gate(tmp_path):
    video = {**STILL, "claim_id": "pipe-video", "qualities": STILL["qualities"] + ["temporal"],
             "scene": {"seed": 7, "width": 160, "height": 160, "frames": 3, "pan_px_per_frame": [1.37, -0.61]},
             "capture": {"exposure_s": 0.0208, "exposure_ev": 0.0, "motion_px_per_s": [32.9, -14.6]},
             "delivery": {"format": "png"}}
    res = evaluate_claim(Claim.from_dict(video), envelope_evs=(0.0,))
    assert res["supported"], res["verdict"]
    assert res["gates"][4]["name"] == "temporal" and res["gates"][4]["status"] == "PASS"
    assert res["metrics"]["frames"] == 3


def test_files_candidate_roundtrip(tmp_path):
    from pixel.colorsci import display_encode
    from pixel.scene import Scene, SceneSpec

    img = display_encode(Scene(SceneSpec(seed=7, width=192, height=192)).render(0.0))
    d = tmp_path / "cand"
    save_image(d / "frame_0000.png", img)
    claim = Claim.from_dict({**STILL, "regime": "capture_exceed", "candidate": {"kind": "files", "path": str(d)}, "delivery": {"format": "png"}})
    res = evaluate_claim(claim, envelope_evs=(0.0,))
    assert res["supported"]
    assert res["metrics"]["delta_e_p95"] < 1.0  # 8-bit PNG roundtrip: deep-shadow quantisation alone costs ~0.7 dE at p95


def test_not_provable_regime_measures_nothing():
    res = evaluate_claim(Claim.from_dict({**STILL, "regime": "record_of_reality"}))
    assert not res["supported"] and "NOT PROVABLE" in res["verdict"] and res["metrics"] == {}
    assert "Nothing was measured" in render_report(res)
