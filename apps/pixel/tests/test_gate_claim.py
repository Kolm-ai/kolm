import json

import pytest

from pixel.claim import Claim, ClaimError
from pixel.gate import FAIL, PASS, SKIP, compare, run_gates, verdict

BASE = {
    "claim_id": "t",
    "regime": "synthetic_world",
    "qualities": ["geometry", "tone_color", "artifacts", "resolution"],
    "scene": {"seed": 1, "width": 128, "height": 128, "frames": 1},
    "capture": {"exposure_s": 0.01},
    "display": {"diagonal_in": 31.5, "width_px": 7680, "height_px": 4320, "distance_m": 0.6},
    "candidate": {"kind": "truth"},
}


def test_claim_requires_the_tuple():
    for key in ("regime", "qualities", "scene", "capture", "display", "candidate"):
        d = dict(BASE)
        del d[key]
        with pytest.raises(ClaimError):
            Claim.from_dict(d)
    with pytest.raises(ClaimError):
        Claim.from_dict({**BASE, "qualities": []})
    with pytest.raises(ClaimError):
        Claim.from_dict({**BASE, "qualities": ["temporal"]})  # one frame
    with pytest.raises(ClaimError):
        Claim.from_dict({**BASE, "candidate": {"kind": "files"}})  # no path


def test_record_of_reality_is_not_provable():
    c = Claim.from_dict({**BASE, "regime": "record_of_reality"})
    assert not c.provable
    assert Claim.from_dict(BASE).provable


def test_gate_halts_at_first_failure_and_skips_unclaimed():
    good = {"landmark_error_px": 0.0, "highlight_loss_fraction": 0.0, "shadow_noise_sigma": 0.0, "shadow_detail_corr": 1.0,
            "banding_score": 0.0, "patch_delta_e": 0.0, "delta_e_p95": 0.0, "halo_score": 0.0, "blocking_score": 0.0,
            "tiling_score": 0.1, "mtf50": 0.5}
    env = {"mtf50": 0.26}
    res = run_gates(good, env, ["geometry", "tone_color", "artifacts", "resolution"], is_video=False)
    assert [r.status for r in res] == [PASS, PASS, PASS, PASS, SKIP]
    assert verdict(res)[0]
    bad = {**good, "highlight_loss_fraction": 0.7}
    res = run_gates(bad, env, ["geometry", "tone_color", "artifacts", "resolution"], is_video=False)
    assert [r.status for r in res] == [PASS, FAIL, SKIP, SKIP, SKIP]
    assert "highlight_loss_fraction" in verdict(res)[1]
    res = run_gates(good, env, ["resolution"], is_video=False)
    assert [r.status for r in res] == [SKIP, SKIP, SKIP, PASS, SKIP]
    res = run_gates({**good, "mtf50": 0.2}, env, ["resolution"], is_video=False)
    assert res[3].status == FAIL and "loses floor" in res[3].reason


def test_compare_uses_tolerance_and_direction():
    assert compare("mtf50", 0.30, 0.26) == "beats"
    assert compare("mtf50", 0.265, 0.26) == "ties"
    assert compare("shadow_noise_sigma", 0.0, 0.002) == "beats"
    assert compare("shadow_noise_sigma", 0.003, 0.002) == "loses"


def test_claim_roundtrip(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(BASE))
    c = Claim.load(p)
    assert c.to_dict()["claim_id"] == "t" and c.frames == 1 and not c.is_video
