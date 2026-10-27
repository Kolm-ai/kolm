import csv
import json

import numpy as np
import pytest

from harness.claim import default_claim, load_claim, validate_claim
from harness.evaluate import build_world, evaluate
from harness.gate import run_gates
from harness.preference import fit_from_files, make_manifest, preference_fit


def test_gate_halts_on_first_failure_and_skips_missing():
    gates = [
        {"name": "a", "checks": [{"metric": "pref_convincing_p_beats_camera_low", "op": ">", "value": 0.5}]},
        {"name": "b", "checks": [{"metric": "delta_e_mean", "op": "<=", "value": 2.0},
                                 {"metric": "mtf50_cpp", "op": "beats_camera"}]},
        {"name": "c", "checks": [{"metric": "dr_steps", "op": "beats_camera"}]},
    ]
    cand = {"delta_e_mean": 0.3, "mtf50_cpp": 0.4, "dr_steps": 13.0}
    cam = {"delta_e_mean": 0.2, "mtf50_cpp": 0.4, "dr_steps": 11.0}
    res = run_gates(gates, cand, cam)
    assert [g.status for g in res] == ["skip", "fail"]
    assert res[1].checks[0].status == "pass" and res[1].checks[1].status == "fail"
    assert "ties" in res[1].checks[1].detail
    cand["mtf50_cpp"] = 0.5
    assert [g.status for g in run_gates(gates, cand, cam)] == ["skip", "pass", "pass"]


def test_claim_validation(tmp_path):
    claim = default_claim()
    validate_claim(claim)
    bad = json.loads(json.dumps(claim))
    bad["gates"][0]["checks"][0]["metric"] = "sharpness_vibes"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_claim(p)
    with pytest.raises(ValueError):
        validate_claim({"qualities": {}})


def test_bradley_terry_recovers_order():
    rng = np.random.default_rng(0)
    strength = {"candidate": 4.0, "camera": 2.0, "truth": 1.0}
    items = list(strength)
    pairs = []
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = items[i], items[j]
            for _ in range(60):
                pa = strength[a] / (strength[a] + strength[b])
                pairs.append((a, b) if rng.random() < pa else (b, a))
    fit = preference_fit(pairs, items, n_boot=100, seed=0)
    s = fit["strength"]
    assert s["candidate"] > s["camera"] > s["truth"]
    pb = fit["p_beats"]["candidate"]["camera"]
    assert pb["low"] > 0.5 and abs(pb["p"] - 2 / 3) < 0.12


def test_manifest_roundtrip(tmp_path):
    items = {"candidate": "c.png", "camera": "k.png"}
    manifest, key = make_manifest(items, trials_per_pair=6, seed=1)
    assert len(manifest["trials"]) == 6 and set(key.values()) == set(items)
    assert {t["left"] for t in manifest["trials"]} == set(key)  # both sides used
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    (tmp_path / "k.json").write_text(json.dumps(key))
    inv = {v: k for k, v in key.items()}
    with open(tmp_path / "r.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "question", "choice"])
        for t in manifest["trials"]:
            for q in manifest["questions"]:
                w.writerow([t["trial"], q, "left" if t["left"] == inv["candidate"] else "right"])
    fit = fit_from_files(tmp_path / "m.json", tmp_path / "k.json", tmp_path / "r.csv", n_boot=50)
    assert fit["metrics"]["pref_convincing_p_beats_camera_low"] > 0.5


@pytest.fixture(scope="module")
def small_world():
    claim = default_claim(video=True)
    claim["scene"].update(width=256, height=144, supersample=2, n_frames=4)
    return build_world(claim)


def test_truth_as_candidate_passes(small_world, tmp_path):
    rep = evaluate(small_world, small_world.truth, tmp_path / "t", "truth")
    assert rep["overall"] == "INCOMPLETE (skipped: composition)"  # no human study: not a PASS
    statuses = {g["name"]: g["status"] for g in rep["gates"]}
    assert statuses["composition"] == "skip"
    assert all(s == "pass" for n, s in statuses.items() if n != "composition")
    q = {k: v["verdict"] for k, v in rep["qualities"].items()}
    assert q["resolution"] == q["shadows"] == q["temporal"] == "beats"
    assert q["color"] == q["artifacts"] == q["highlights"] == "ties"  # the best-case camera exposes for the peak
    assert q["geometry_lighting"] in ("ties", "beats")  # blur+noise cost the camera ~0.3 px / 0.006 rmse
    assert rep["metrics"]["highlight_loss_fraction"]["camera"] == 0.0  # it no longer clips the wedge
    assert rep["metrics"]["dr_steps"]["camera"] == rep["metrics"]["dr_steps"]["truth"]  # the 11-stop wedge is inside its DR
    assert rep["delivery"]["metrics"]["mtf50_cpp"]["delivered"] is not None


def test_camera_as_candidate_ties_itself(small_world, tmp_path):
    rep = evaluate(small_world, small_world.camera, tmp_path / "c", "camera")
    assert rep["overall"].startswith("FAIL at ")  # it cannot beat itself where a gate demands beating the camera
    failed = [c for g in rep["gates"] for c in g["checks"] if c["status"] == "fail"]
    assert failed and all(c["op"] == "beats_camera" for c in failed)
    for m, r in rep["metrics"].items():
        if r["verdict"] != "n/a":
            assert r["verdict"] == "ties", m
