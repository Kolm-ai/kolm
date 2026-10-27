"""Preference model with ties and rater scales: recovers a planted truth, scores held-out votes,
reads every public vote-set schema from tiny synthetic parquet files, and the collector's
column-projected hf:// specs round-trip through the manifest."""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.stats import spearmanr

from harness import preference as P
from harness.data import prefdata as D
from harness.data.manifest import FileSpec, Source


def _simulate(n_votes=6000, theta=0.6, seed=0, betas=(0.4, 1.0, 2.5)):
    rng = np.random.default_rng(seed)
    s_true = np.array([1.0, 0.5, 0.0, -0.3, -1.2])
    items = [f"m{i}" for i in range(len(s_true))]
    a = rng.integers(0, 5, n_votes)
    b = (a + rng.integers(1, 5, n_votes)) % 5
    r = rng.integers(0, len(betas), n_votes)
    pa_, pb_, pt_ = P.rk_probabilities(s_true[a], s_true[b], theta, np.asarray(betas)[r])
    u = rng.random(n_votes)
    y = np.where(u < pa_, 1, np.where(u < pa_ + pb_, -1, 0)).astype(np.int8)
    return P.Votes(items, a, b, y, r, [f"r{i}" for i in range(len(betas))]), s_true, theta, np.asarray(betas)


def test_rao_kupper_recovers_strengths_tie_band_and_rater_scales():
    v, s_true, theta, betas = _simulate()
    fit = P.fit_rao_kupper(v)
    assert fit.converged
    assert spearmanr(fit.s, s_true).statistic == pytest.approx(1.0)
    assert abs(fit.theta - theta) < 0.15
    assert np.all(np.diff(np.exp(fit.log_beta)) > 0)                       # 0.4 < 1 < 2.5 recovered in order
    assert abs(np.exp(fit.log_beta[1]) - 1.0) < 0.35                         # the unit rater reads near 1


def test_no_ties_and_no_raters_reduces_to_bradley_terry():
    v, s_true, _, _ = _simulate(theta=0.0, betas=(1.0,))
    v = v.subset(v.y != 0)
    v.r[:] = -1
    fit = P.fit_rao_kupper(v)
    assert fit.theta == 0.0 and fit.log_beta.size == 0
    pairs = [(v.items[a], v.items[b]) if y > 0 else (v.items[b], v.items[a]) for a, b, y in zip(v.a, v.b, v.y)]
    mm = np.log(P.fit_bradley_terry(pairs, v.items))
    assert np.abs((mm - mm.mean()) - fit.s).max() < 0.05                     # same answer as the MM reference


def test_holdout_scores_ties_and_reports_calibration():
    v, *_ = _simulate(n_votes=4000)
    rep = P.holdout_report(v, seed=1)
    m = rep["models"]
    assert set(m) == {"bradley_terry", "rao_kupper", "rao_kupper_rater"}
    assert m["rao_kupper"]["mean_loglik_all"] > -1.1                          # three outcomes: log(1/3) = -1.10 is chance
    assert m["rao_kupper_rater"]["mean_loglik_all"] >= m["rao_kupper"]["mean_loglik_all"] - 1e-3
    assert m["bradley_terry"]["decisive_accuracy"] > 0.6
    assert rep["calibration"]["expected_calibration_error"] < 0.1
    fc = P.fold_consistency(v, n_folds=4, n_boot=20)
    assert 0.3 < fc["median_boot_sd_over_fold_sd"] < 3.0


def test_preference_fit_votes_shares_the_p_beats_shape():
    v, *_ = _simulate(n_votes=1500)
    out = P.preference_fit_votes(v, n_boot=10)
    assert out["model"] == "rao_kupper_logistic_rater" and out["n_raters"] == 3
    cell = out["p_beats"]["m0"]["m4"]
    assert cell["low"] <= cell["p"] <= cell["high"] and cell["p"] > 0.5


def test_votes_by_question_maps_tie_words_and_raters(tmp_path):
    manifest, key = P.make_manifest({"candidate": "x.png", "camera": "y.png"}, trials_per_pair=3)
    rows = ["trial,question,choice,rater"]
    for t in manifest["trials"]:
        rows.append(f"{t['trial']},beautiful,{'left' if key[t['left']] == 'candidate' else 'right'},alice")
        rows.append(f"{t['trial']},convincing,tie,bob")
    p = tmp_path / "r.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    votes = P.votes_by_question(manifest, key, P.read_responses(p))
    assert votes["beautiful"].tie_share == 0 and votes["convincing"].tie_share == 1.0
    assert votes["beautiful"].raters == ["alice"]
    fit = P.fit_from_files(_dump(tmp_path, "m.json", manifest), _dump(tmp_path, "k.json", key), p, n_boot=5)
    assert fit["metrics"]["pref_beautiful_p_beats_camera_low"] > 0.5


def _dump(d, name, obj):
    p = d / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _write(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_readers_on_minimal_parquet(tmp_path):
    root = tmp_path / "rapidata"
    detail = json.dumps({"votes": [{"votedFor": "flux/1.jpg", "userDetails": {"userScore": 0.9}},
                                   {"votedFor": "dalle/1.jpg", "userDetails": {"userScore": 0.4}},
                                   {"votedFor": "flux/1.jpg", "userDetails": {}}]})
    _write(root / "default/train_0001/0000.parquet", pa.table({
        "votes_image1": [1], "votes_image2": [2], "model1": ["dalle"], "model2": ["flux"], "detailed_results": [detail],
        "image1_path": ["dalle/1.jpg"], "image2_path": ["flux/1.jpg"]}))
    vs = D.read_rapidata_700k(root)
    assert vs.votes.y.tolist() == [-1, 1, -1] and vs.checks["share_rows_matching"] == 1.0
    assert np.isnan(vs.score[2]) and vs.score[0] == 0.9

    root = tmp_path / "oip"
    _write(root / "default/train/0000.parquet", pa.table({
        "model_1": ["dev"], "model_2": ["sd"], "preference.responses": [["both_good", "image_1", "image_2", "skipped"]],
        "preference.responses.users": [["u1", "u2", "u3", "u4"]],
        "preference.responses.status": [["submitted", "submitted", "draft", "submitted"]]}))
    vs = D.read_open_image_preferences(root)
    assert vs.votes.y.tolist() == [0, 1] and vs.votes.raters == ["u1", "u2"] and vs.checks["dropped"] == 2

    root = tmp_path / "arena"
    _write(root / "battle/test/0000.parquet", pa.table({
        "left_model": ["A", "B"], "right_model": ["B", "C"], "vote_type": ["leftvote", "bothbad_vote"],
        "judge": ["j1", None], "left_image": [b"x", b"y"]}))
    vs = D.read_arena_human_eval(root)
    assert vs.votes.y.tolist() == [1, 0] and vs.votes.r.tolist() == [0, -1]

    root = tmp_path / "hpd"
    _write(root / "default/test/0000.parquet", pa.table({
        "image_path": [["a.jpg", "b.jpg", "c.jpg"]],
        "raw_annotations": [[{"annotation": [2, 0, 1], "user_hash": 7}, {"annotation": [0, 1], "user_hash": 8}]]}))
    vs = D.read_hpdv2(root)
    assert len(vs.votes) == 3 and vs.checks["malformed_annotations"] == 1
    # ranks a=2, b=0, c=1: b beats a, c beats a, b beats c
    assert [(vs.votes.items[x], vs.votes.items[y], int(s)) for x, y, s in zip(vs.votes.a, vs.votes.b, vs.votes.y)] == [
        ("a.jpg", "b.jpg", -1), ("a.jpg", "c.jpg", -1), ("b.jpg", "c.jpg", 1)]
    assert D.read_genai_bench(tmp_path / "nothing") is None


def test_analyse_and_cross_set_on_synthetic_votes():
    v, *_ = _simulate(n_votes=1500)
    vs = D.VoteSet("x", "src", "model", v, score=np.random.default_rng(0).random(len(v)))
    r = D.analyse(vs, n_boot=5)
    assert r["counts"]["n_votes"] == 1500 and r["rater_check"]["raters_with_20_votes"] == 3
    assert r["score_check"]["by_quartile"][0]["n"] > 0 and r["fit"]["ranking"][0] == "m0"
    other = {"fit": {"log_strength": {"M0": 0.0, "m1": -1.0, "m4": -2.0}}, "item_kind": "model"}
    overlap = D.cross_set_overlap({"x": r, "y": other})
    assert overlap[0]["shared"] == ["m0", "m1", "m4"] and overlap[0]["sign_agreement"] == 1.0
    assert len(D.verdict_lines({"x": r}, overlap)) >= 3


def test_filespec_columns_round_trip_and_projected_specs():
    spec = FileSpec(url="hf://datasets/d@~parquet/c/s/0000.parquet", path="c/s/0000.parquet", columns=["p", "q"])
    d = spec.to_dict()
    assert d["columns"] == ["p", "q"] and FileSpec(**d) == spec
    src = Source.from_dict({"id": "t", "name": "t", "kind": "k", "access": "api", "url": "u", "license": "l",
                            "files": [d]})
    assert src.files[0].columns == ["p", "q"]

    from harness.data import apis

    class FakeCollector:
        def get_json(self, url, params=None, headers=None):
            return {"parquet_files": [{"config": "c", "split": "s", "filename": "0000.parquet", "url": "https://x/0000.parquet", "size": 5},
                                      {"config": "c", "split": "other", "filename": "0000.parquet", "url": "https://x/1", "size": 5}]}

    s = Source.from_dict({"id": "t", "name": "t", "kind": "k", "access": "api", "url": "u", "license": "l",
                          "expander": "hf_parquet", "params": {"dataset": "d", "config": "c", "split": "s", "columns": "p,q"}})
    specs = apis.hf_parquet(FakeCollector(), s)
    assert [x.path for x in specs] == ["parquet.json", "c/s/0000.parquet"]
    assert specs[1].url == "hf://datasets/d@~parquet/c/s/0000.parquet" and specs[1].columns == ["p", "q"] and specs[1].size is None
    s.params.pop("columns")
    assert apis.hf_parquet(FakeCollector(), s)[1].url == "https://x/0000.parquet"
