"""The scene-paired run verdict (harness.model.verdict) on synthetic held-out reports: no files, no GPU."""
from __future__ import annotations

import numpy as np
import pytest

from harness.metrics import REGISTRY
from harness.model.verdict import ALPHA, FEW_SCENES, TONE, compare, margin, reports, scenes


def report(scene_vals: dict[str, np.ndarray], truth: dict[str, float] | None = None, tiles_per: int = 4,
           noise: float = 0.0, seed: int = 0) -> dict:
    """a raw.pixls.us-style report: `tiles_per` tiles per scene; scene_vals[m][k] is metric m's value on scene k, each
    tile that plus Gaussian noise of sd `noise`"""
    rng = np.random.default_rng(seed)
    n_sc = len(next(iter(scene_vals.values())))
    records, tiles = [], []
    for k in range(n_sc):
        for j in range(tiles_per):
            records.append({"file": f"scene{k}.dng", "y0": j, "x0": 0})
            out = {m: float(v[k] + noise * rng.normal()) for m, v in scene_vals.items()}
            tiles.append({"input": {}, "output": out, "truth": dict(truth or {}), "tone": {"output": {}}})
    return {"real": False, "records": records, "tiles": tiles}


def by_metric(res: dict) -> dict:
    return {(o["set"], o["metric"]): o for o in res["cells"]}


def base_vals(n_sc: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.02, 0.005, n_sc)


def test_identical_runs_read_non_inferior_with_no_change():
    v = {"shading_rmse": base_vals(12), "banding_score": base_vals(12, 2)}
    rep = report(v, noise=0.001)
    res = compare([{"s": rep}], [{"s": rep}], scenes, boot=200)
    for o in res["cells"]:
        assert o["margin"] == 0 and o["outcome"] == "non-inferior" and not o["lower"]
        assert o["p"] == 1.0
    assert res["pooled"]["iqm"] == 0 and res["pooled"]["improve"] == 0.5
    assert res["pooled"]["reading"] == "no clear change"


def test_a_shift_on_many_scenes_is_resolved_both_ways():
    n = 30
    d = REGISTRY["shading_rmse"].tie_margin
    b = base_vals(n)
    good = report({"shading_rmse": b - 3 * d}, noise=0.001, seed=1)
    bad = report({"shading_rmse": b + 3 * d}, noise=0.001, seed=2)
    base = report({"shading_rmse": b}, noise=0.001, seed=3)
    up = by_metric(compare([{"s": good}], [{"s": base}], scenes, boot=300))[("s", "shading_rmse")]
    assert up["outcome"] == "better" and up["p"] < ALPHA and up["lo"] > 0
    assert up["margin"] == pytest.approx(3 * d, rel=0.2) and not up["few"]
    down = by_metric(compare([{"s": bad}], [{"s": base}], scenes, boot=300))[("s", "shading_rmse")]
    assert down["outcome"] == "worse" and down["lower"] and down["hi"] < -d


def test_a_resolved_loss_inside_the_margin_is_non_inferior_and_marked_lower():
    n, d = 30, REGISTRY["shading_rmse"].tie_margin
    b = base_vals(n)
    res = compare([{"s": report({"shading_rmse": b + 0.3 * d}, noise=1e-4, seed=1)}],
                  [{"s": report({"shading_rmse": b}, noise=1e-4, seed=2)}], scenes, boot=300)
    o = by_metric(res)[("s", "shading_rmse")]
    assert o["outcome"] == "non-inferior" and o["lower"] and -d < o["lo"] <= o["hi"] < 0


def test_a_to_truth_metric_is_judged_by_distance_to_the_truth():
    t = 0.30
    assert margin("mtf50_cpp", 0.36, 0.31, t) == pytest.approx(0.01 - 0.06)    # sharper, but past the truth
    assert margin("mtf50_cpp", 0.29, 0.25, t) == pytest.approx(0.05 - 0.01)
    assert margin("shading_rmse", 0.01, 0.02) == pytest.approx(0.01)          # lower is better
    assert margin("shadow_snr_db", 30.0, 28.0) == pytest.approx(2.0)          # higher is better
    assert margin(TONE + "0.2-0.5", 1.01, 0.97) == pytest.approx(abs(np.log(0.97)) - abs(np.log(1.01)))
    n = 25
    b = np.full(n, 0.31) + base_vals(n) * 0.01
    res = compare([{"s": report({"mtf50_cpp": b + 0.05}, {"mtf50_cpp": t}, noise=1e-3, seed=1)}],
                  [{"s": report({"mtf50_cpp": b}, {"mtf50_cpp": t}, noise=1e-3, seed=2)}], scenes, boot=300)
    o = by_metric(res)[("s", "mtf50_cpp")]
    assert o["margin"] < -REGISTRY["mtf50_cpp"].tie_margin and o["outcome"] == "worse"


def test_three_scenes_can_never_read_better():
    """2^3 relabellings: the smallest p is 1/8, above ALPHA, whatever the shift"""
    b = base_vals(3)
    res = compare([{"s": report({"shading_rmse": b - 0.05}, seed=1)}], [{"s": report({"shading_rmse": b}, seed=2)}],
                  scenes, boot=200)
    o = by_metric(res)[("s", "shading_rmse")]
    assert o["p"] == pytest.approx(1 / 8) and o["outcome"] != "better" and o["few"] and 3 < FEW_SCENES


def test_seeds_and_sets():
    n = 24
    b = base_vals(n)
    cand = [{"a": report({"shading_rmse": b - 0.02}, noise=1e-3, seed=s),
             "b": report({"banding_score": b}, noise=1e-3, seed=10 + s)} for s in range(3)]
    base = [{"a": report({"shading_rmse": b}, noise=1e-3, seed=20 + s),
             "b": report({"banding_score": b}, noise=1e-3, seed=30 + s)} for s in range(2)]
    other = report({"shading_rmse": b}, tiles_per=3)                      # different tiles: skipped
    base[0]["c"] = base[1]["c"] = other
    for c in cand:
        c["c"] = report({"shading_rmse": b})
    res = compare(cand, base, scenes, boot=200)
    got = by_metric(res)
    assert got[("a", "shading_rmse")]["outcome"] == "better"
    assert got[("b", "banding_score")]["outcome"] == "non-inferior"
    assert res["skipped"] == ["c"] and res["pooled"]["cells"] == 2


def test_reports_reads_a_run_or_else_a_baseline_and_leaves_pre_release_sets_out(tmp_path):
    import json
    for d, rep in (("recovered_v1/heldout_blind", {"n": 1}), ("recovered_v1/heldout_pre_blind", {"n": 2}),
                   ("baseline_nafnet/heldout_blind", {"n": 3}), ("baseline_v1/heldout_blind", {"n": 4})):
        (tmp_path / "models" / d).mkdir(parents=True)
        (tmp_path / "models" / d / "heldout.json").write_text(json.dumps(rep), encoding="utf-8")
    assert reports("v1", tmp_path) == {"blind": {"n": 1}}            # a run's own directory wins
    assert reports("nafnet", tmp_path) == {"blind": {"n": 3}}
    assert reports("missing", tmp_path) == {}
