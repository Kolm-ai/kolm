"""compare.py: every run's held-out reports side by side."""
from __future__ import annotations

import json

from harness.model.compare import GATES, METRICS, cells, collect, to_markdown


def _rep(step, edge, frontier_edge=None):
    rep = {"step": step, "metrics": {m: {"output": 0.5} for m in METRICS},
           "gates": {"capture": {"overall": "FAIL at artifacts"}, "camera": {"overall": "PASS"}}}
    rep["metrics"]["edge_overshoot"]["output"] = edge
    if frontier_edge is not None:
        rep["gates"]["frontier"] = {"overall": "PASS", "metrics": {
            "edge_overshoot": {"frontier": frontier_edge}, "shadow_snr_db": {"frontier": 20.0}}}
    return rep


def _write(root, run, name, rep):
    d = root / f"recovered_{run}" / name
    d.mkdir(parents=True)
    (d / "heldout.json").write_text(json.dumps(rep), encoding="utf-8")


def test_collect_orders_runs_by_version_and_skips_kept_reports(tmp_path):
    for run in ("v14b", "v9", "v14a", "v10"):
        _write(tmp_path, run, "heldout_known", _rep(1, 0.02))
    _write(tmp_path, "v14b", "heldout_known_pre_v16", _rep(1, 0.03))
    _write(tmp_path, "v14b", "heldout_real_rawnind_blind", _rep(2, 0.01, 0.0))
    t = collect(tmp_path)
    assert list(t["heldout_known"]) == ["v9", "v10", "v14a", "v14b"]
    assert "heldout_known_pre_v16" not in t and list(t["heldout_real_rawnind_blind"]) == ["v14b"]
    assert list(collect(tmp_path, ["v10", "v14b"])["heldout_known"]) == ["v10", "v14b"]


def test_cells_pair_the_output_with_the_frontier_only_where_the_report_has_one():
    old, new = cells(_rep(5, 0.0207)), cells(_rep(6, 0.0207, 0.0))
    e = 1 + len(GATES) + METRICS.index("edge_overshoot")
    assert old[:4] == ["5", "fail: artifacts", "PASS", "-"] and old[e] == "0.0207"
    assert new[3] == "PASS" and new[e] == "0.0207 / 0"
    assert new[1 + len(GATES) + METRICS.index("shadow_snr_db")] == "0.5"   # matched, so the output's alone
    md = to_markdown({"heldout_known": {"v14b": _rep(6, 0.0207, 0.0)}})
    assert "## heldout_known" in md and "| v14b | 6 | fail: artifacts | PASS | PASS |" in md
