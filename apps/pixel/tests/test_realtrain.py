"""Real pairs and source weights in the training stream (harness.model.data, train, cli): a TileSource draws by its
sources' weights and keeps them through a drop; cond_vector carries a measured capture or withholds it; PairStream at
p_real 1 yields stored real pairs in the log domain with the capture known or withheld; the real validation set is
seeded; a two-step run on the fake store logs the real validation and saves the real-pair and dead-leaves arguments;
and the train CLI parses --real, --p-real, --weights, the loss weights and --p-dead-leaves."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import harness.model.data as D
from harness.data import data_root
from harness.data.realstore import RealStore, build_store
from harness.metrics import STILL_METRICS
from harness.model.cli import parse_weights
from harness.model.data import (COND_DIM, METRIC_COND_DIM, UNKNOWN_CAPTURE, Normaliser, PairStream, TileSource,
                                cond_vector, decode, real_fixed_set, real_sample)

from test_realpairs import _fake_pairs

DERIVED_TABLES = ("camera_presets.json", "noise_check.json", "capture_stats.json", "hdri_stats.json",
                  "lens_traces.json", "gates_fitted.json", "isp_stats.json")
needs_data = pytest.mark.skipif(not all((data_root() / "derived" / f).exists() for f in DERIVED_TABLES),
                                reason="derived stats not built")


def _tiny_sources(tmp_path, names, n=3, size=40, seed=0):
    rng = np.random.default_rng(seed)
    for name in names:
        d = tmp_path / "raw" / name
        d.mkdir(parents=True, exist_ok=True)
        for k in range(n):
            Image.fromarray((rng.random((size, size, 3)) * 255).astype(np.uint8)).save(d / f"{name}_{k}.png")


def _fake_store(tmp_path):
    imgs, pairs = _fake_pairs(np.random.default_rng(5))
    build_store("fake", ("train", "val"), scale=1, root=tmp_path, workers=1, log=None,
                decode=lambda path, scale: (imgs[path.name], 14), pairs=pairs)
    return imgs, pairs


def test_tile_source_draws_by_source_weight_and_keeps_the_weights_through_a_drop(tmp_path, monkeypatch):
    _tiny_sources(tmp_path, ["wa", "wb"])
    monkeypatch.setattr(D, "split_of", lambda it: "train")
    rng = np.random.default_rng(0)
    plain = TileSource(["wa", "wb"], "train", 16, root=tmp_path)
    assert plain.shares() == {"wa": 0.5, "wb": 0.5}
    src = TileSource(["wa", "wb"], "train", 16, root=tmp_path, weights={"wa": 9.0})     # wb unnamed: weight 1
    assert src.shares() == pytest.approx({"wa": 0.9, "wb": 0.1})
    drawn = [src.random_tile(rng)[1].source for _ in range(400)]
    assert 0.84 < drawn.count("wa") / 400 < 0.96
    it = next(it for it in src.items if it.source == "wa")
    with pytest.warns(RuntimeWarning):
        src.drop(it, ValueError("gone"))
    assert len(src._w) == len(src.items) == 5 and src.shares() == pytest.approx({"wa": 18 / 21, "wb": 3 / 21})
    assert {src.pick(rng).source for _ in range(60)} == {"wa", "wb"}
    with pytest.raises(ValueError, match="non-negative"):
        TileSource(["wa", "wb"], "train", 16, root=tmp_path, weights={"wa": -1.0})
    with pytest.raises(ValueError, match="positive sum"):
        TileSource(["wa", "wb"], "train", 16, root=tmp_path, weights={"wa": 0.0, "wb": 0.0})
    only = TileSource(["wa", "wb"], "train", 16, root=tmp_path, weights={"wb": 0.0})
    assert {only.random_tile(rng)[1].source for _ in range(30)} == {"wa"}


def test_cond_vector_carries_a_measured_capture_or_withholds_it():
    norm = Normaliser(np.zeros(len(STILL_METRICS)), np.ones(len(STILL_METRICS)))
    cap = np.arange(7, dtype=np.float32) / 10
    s = {"metrics": {m: 0.5 for m in STILL_METRICS}, "capture": cap}
    c = cond_vector(norm, s)
    assert c.shape == (COND_DIM,) and c.dtype == np.float32 and np.isfinite(c).all()
    assert np.array_equal(c[METRIC_COND_DIM:], cap)
    assert np.array_equal(cond_vector(norm, s, known=False)[METRIC_COND_DIM:], UNKNOWN_CAPTURE)


def test_real_sample_and_pair_stream_yield_stored_pairs_with_their_capture(tmp_path, monkeypatch):
    _fake_store(tmp_path)
    _tiny_sources(tmp_path, ["tiny"], size=70)
    monkeypatch.setattr(D, "split_of", lambda it: "train")
    monkeypatch.setattr(D.Distributions, "load", classmethod(lambda cls, root=None: None))   # never drawn at p_real 1
    store = RealStore(["fake"], "train", tmp_path, tile=64)
    rng = np.random.default_rng(0)
    s = real_sample(store, rng)
    assert s["restore"] and s["real"] and s["x"].shape == s["y"].shape == (64, 64, 3)
    assert s["x"].flags.c_contiguous and s["y"].flags.c_contiguous and 0.0 <= s["clipped"] <= 0.01
    assert set(s["metrics"]) == set(STILL_METRICS) and s["capture"].shape == (7,) and s["capture"][-1] == 1.0
    fixed = real_fixed_set(["fake"], "val", 4, 64, 0, tmp_path)
    again = real_fixed_set(["fake"], "val", 4, 64, 0, tmp_path)
    assert len(fixed) == 4 and all(np.array_equal(a["x"], b["x"]) for a, b in zip(fixed, again))
    other = real_fixed_set(["fake"], "val", 4, 64, 1, tmp_path)
    assert [(a["y0"], a["x0"]) for a in fixed] != [(a["y0"], a["x0"]) for a in other]
    norm = Normaliser(np.zeros(len(STILL_METRICS)), np.ones(len(STILL_METRICS)))
    it = iter(PairStream(["tiny"], 64, norm, seed=3, root=tmp_path, real=("fake",), p_real=1.0))
    tails = []
    for _ in range(12):
        x, y, c, restore, real = next(it)
        assert tuple(x.shape) == tuple(y.shape) == (3, 64, 64) and x.dtype == y.dtype == torch.float32
        assert restore is True and real is True and tuple(c.shape) == (COND_DIM,)
        xl, yl = decode(x.permute(1, 2, 0).numpy()), decode(y.permute(1, 2, 0).numpy())
        assert xl.min() >= -1e-4 and xl.max() <= 1.0 + 1e-4 and yl.min() >= -1e-4 and yl.max() <= 1.0 + 1e-4
        tails.append(c[METRIC_COND_DIM:].numpy())
    cap = store.pairs[0].capture
    assert any(np.allclose(t, cap) for t in tails) and any(np.array_equal(t, UNKNOWN_CAPTURE) for t in tails)
    with pytest.raises(ValueError, match="p_real > 0 needs real sources"):
        PairStream(["tiny"], 64, norm, seed=0, root=tmp_path, p_real=0.5)


@needs_data
def test_train_two_steps_on_real_pairs_logs_the_real_validation_and_saves_the_arguments(tmp_path, monkeypatch):
    from harness.model.train import train
    src_root = data_root()
    (tmp_path / "derived").mkdir()
    for f in DERIVED_TABLES:
        shutil.copy(src_root / "derived" / f, tmp_path / "derived" / f)
    monkeypatch.setenv("HARNESS_DATA_ROOT", str(tmp_path))
    _fake_store(tmp_path)
    _tiny_sources(tmp_path, ["tiny"], n=4, size=48)
    monkeypatch.setattr(D, "split_of", lambda it: "val" if it.file.endswith("_0.png") else "train")
    out = tmp_path / "m"
    log = []
    v = train(str(out), ["tiny"], steps=2, batch=2, tile=32, workers=0, base=8, val_n=4, calib_n=8, ckpt_every=1,
              device="cpu", real=("fake",), p_real=0.5, weights={"tiny": 2.0}, p_dead_leaves=0.5, log=log.append)
    assert np.isfinite(v["val_loss"]) and v["step"] == 2
    assert any(s.startswith("draw shares by source: tiny 100.0%") for s in log)
    assert any("real pairs drawn with probability 0.50: real store, split train: 1 pairs in 1 scenes (fake 1 drawn 100%)" in s for s in log)
    assert any("real validation draw: 4 tiles" in s for s in log) and any("real val loss" in s for s in log)
    assert any(s == "dead leaves: 50.0% of the rendered pairs on procedural tiles" for s in log)
    rows = [json.loads(line) for line in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    real_rows = [r for r in rows if "val_real_loss" in r]
    assert len(real_rows) == 3 and all(np.isfinite(r["val_real_loss"]) and r["val_real_n"] == 4 for r in real_rows)
    assert all(set(r["val_real_restore"]) == set(STILL_METRICS) for r in real_rows)
    assert all(set(v) == {"input", "output"} for r in real_rows for v in r["val_real_restore"].values())
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert ck["train"]["real"] == ["fake"] and ck["train"]["p_real"] == 0.5 and ck["train"]["weights"] == {"tiny": 2.0}
    assert ck["train"]["sources"] == ["tiny"] and ck["step"] in (1, 2)
    assert ck["train"]["range_weight"] == 1.0 and ck["train"]["range_k"] == 3
    assert ck["train"]["flat_weight"] == 0.0 and ck["train"]["flat_k"] == 7 and ck["train"]["texture_weight"] == 0.0
    assert ck["train"]["p_dead_leaves"] == 0.5


def test_train_cli_parses_real_pairs_and_weights():
    from harness.model.cli import add_parser, cmd_train
    ap = argparse.ArgumentParser()
    add_parser(ap.add_subparsers())
    a = ap.parse_args(["model", "train", "--out", "x", "--real", "rawnind,hdr-plus-burst", "--p-real", "0.5",
                       "--weights", "rawnind=4,hdr-plus-burst=4"])
    assert a.func is cmd_train and a.real == "rawnind,hdr-plus-burst" and a.p_real == 0.5
    assert a.range_weight is None and a.range_k is None and a.flat_weight is None and a.flat_k is None
    assert a.texture_weight is None
    c = ap.parse_args(["model", "train", "--out", "x", "--range-weight", "3", "--range-k", "5",
                       "--flat-weight", "0.5", "--flat-k", "9", "--texture-weight", "0.7", "--p-dead-leaves", "0.25"])
    assert c.range_weight == 3.0 and c.range_k == 5 and c.flat_weight == 0.5 and c.flat_k == 9
    assert c.texture_weight == 0.7 and c.p_dead_leaves == 0.25 and a.p_dead_leaves == 0.0
    assert parse_weights(a.weights) == {"rawnind": 4.0, "hdr-plus-burst": 4.0}
    assert a.overshoot_weight is None and a.pyramid_weight is None and a.resample == "learned"
    d = ap.parse_args(["model", "train", "--out", "x", "--overshoot-weight", "1.5", "--pyramid-weight", "0.5",
                       "--resample", "smooth"])
    assert d.overshoot_weight == 1.5 and d.pyramid_weight == 0.5 and d.resample == "smooth"
    with pytest.raises(SystemExit):
        ap.parse_args(["model", "train", "--out", "x", "--resample", "nearest"])
    b = ap.parse_args(["model", "train", "--out", "x"])
    assert b.real is None and b.p_real == 0.0 and b.weights is None and parse_weights(b.weights) == {}
    assert parse_weights(" a = 2 , b=0.5,") == {"a": 2.0, "b": 0.5}
    with pytest.raises(ValueError, match="source=number"):
        parse_weights("rawnind")
    with pytest.raises(ValueError, match="non-negative"):
        parse_weights("rawnind=-1")
    assert isinstance(Path("x"), Path)
