"""The real-pair training store (harness.data.realstore): a build over the fake pairs that stores the aligned
scenes once, records the rejected pair and resumes without decoding; the store that serves unclipped tiles whose
input, target, shift, gains and capture descriptor are realpairs.match_pair's; a reference the source's
clean.exclude names left out at load; HDR+'s synthetic chart bursts excluded; and the CLI wiring."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from harness.data.realpairs import CLIP, match_pair
from harness.data.realstore import RealStore, build_store, key_of, pair_json, scene_dir, store_dir

from test_realpairs import GAIN, _fake_pairs


def _decoder(imgs, calls):
    def decode(path, scale):
        calls.append((path.name, scale))
        return imgs[path.name], 14
    return decode


def test_build_store_stores_each_scene_once_records_rejections_and_resumes(tmp_path):
    imgs, pairs = _fake_pairs(np.random.default_rng(5))
    calls, logged = [], []
    tally = build_store("fake", ("train", "val"), scale=1, root=tmp_path, workers=1, log=logged.append,
                        decode=_decoder(imgs, calls), pairs=pairs)
    assert tally["pairs"] == 3 and tally["scenes"] == 3 and tally["stored"] == 2 and tally["rejected"] == 1
    assert tally["already"] == 0 and tally["bytes"] > 0 and tally["scale"] == 1
    assert sorted(calls) == sorted([("ref0.raw", 1), ("in0.raw", 1), ("ref2.raw", 1), ("in2.raw", 1), ("ref3.raw", 1),
                                    ("in3.raw", 1)])                      # the test scene s1 is not stored
    d = store_dir("fake", tmp_path)
    assert json.loads((d / "store.json").read_text())["scale"] == 1
    for k in (0, 2):
        sd = scene_dir("fake", f"s{k}", tmp_path)
        assert (sd / "ref.npy").exists() and (sd / f"{key_of(f'in{k}.raw')}.npy").exists()
        meta = json.loads(pair_json("fake", pairs[k], tmp_path).read_text())
        assert "rejected" not in meta and meta["shape"] == [192, 256] and meta["bit_depth"] == 14
        assert np.allclose(meta["gain"], GAIN, rtol=0.02) and meta["corr"] > 0.95 and len(meta["capture"]) == 7
        assert 1600 < meta["noise"]["electrons_at_clip"] < 2400 and meta["licence"] == "CC0"
    bad = json.loads(pair_json("fake", pairs[3], tmp_path).read_text())
    assert bad["rejected"].startswith("ValueError: not aligned") and "shape" not in bad
    assert not (scene_dir("fake", "s3", tmp_path) / f"{key_of('in3.raw')}.npy").exists()
    assert np.load(scene_dir("fake", "s0", tmp_path) / "ref.npy").dtype == np.float16
    assert any("1 rejected" in line for line in logged) and any("2 pair(s) stored" in line for line in logged)
    calls.clear()
    again = build_store("fake", ("train", "val"), scale=1, root=tmp_path, workers=1, log=None,
                        decode=_decoder(imgs, calls), pairs=pairs)
    assert calls == [] and again["already"] == 3 and again["stored"] == 0 and again["rejected"] == 0
    with pytest.raises(ValueError, match="one store per source"):
        build_store("fake", ("train",), scale=2, root=tmp_path, workers=1, log=None, decode=_decoder(imgs, calls),
                    pairs=pairs)
    with pytest.raises(ValueError, match="no pairs in splits"):
        build_store("fake", ("nope",), scale=1, root=tmp_path, workers=1, log=None, decode=_decoder(imgs, calls),
                    pairs=pairs)


def test_build_store_stops_at_the_drive_floor_and_resumes_when_there_is_room(tmp_path, monkeypatch):
    imgs, pairs = _fake_pairs(np.random.default_rng(5))
    calls, logged = [], []
    monkeypatch.setenv("HARNESS_REALSTORE_RESERVE_GB", "1e9")          # a floor no drive is above
    tally = build_store("fake", ("train", "val"), scale=1, root=tmp_path, workers=1, log=logged.append,
                        decode=_decoder(imgs, calls), pairs=pairs)
    assert tally["full"].startswith("writing 0 MB") and "HARNESS_REALSTORE_RESERVE_GB" in tally["full"]
    assert calls == [] and tally["stored"] == 0 and tally["rejected"] == 0     # stopped before a decode, none rejected
    assert not list(store_dir("fake", tmp_path).rglob("*.npy")) and any("stopped" in line for line in logged)
    import harness.data.realstore as rs
    room = iter([True, True])                                          # the scene's check and its reference, then full
    monkeypatch.setattr(rs, "keeps_reserve", lambda path, need, reserve: next(room, False))
    mid = build_store("fake", ("train", "val"), scale=1, root=tmp_path, workers=1, log=None,
                      decode=_decoder(imgs, calls), pairs=pairs)
    assert mid["full"] and mid["stored"] == 0 and mid["rejected"] == 0
    assert (scene_dir("fake", "s0", tmp_path) / "ref.npy").exists() and not pair_json("fake", pairs[0], tmp_path).exists()
    monkeypatch.undo()
    monkeypatch.setenv("HARNESS_REALSTORE_RESERVE_GB", "0")
    again = build_store("fake", ("train", "val"), scale=1, root=tmp_path, workers=1, log=None,
                        decode=_decoder(imgs, calls), pairs=pairs)
    assert again["full"] == "" and again["stored"] == 2 and again["rejected"] == 1 and again["already"] == 0


def test_real_store_serves_match_pairs_tiles_with_their_capture(tmp_path):
    imgs, pairs = _fake_pairs(np.random.default_rng(5))
    build_store("fake", ("train", "val"), scale=1, root=tmp_path, workers=1, log=None, decode=_decoder(imgs, []),
                pairs=pairs)
    train = RealStore(["fake"], "train", tmp_path, tile=64)
    val = RealStore(["fake"], "val", tmp_path, tile=64)
    assert len(train) == 1 and len(val) == 1 and val.pairs[0].meta["scene"] == "s0"      # s3 rejected, s1 test
    with pytest.raises(ValueError, match="no stored pairs"):
        RealStore(["fake"], "test", tmp_path)
    with pytest.raises(FileNotFoundError, match="realstore"):
        RealStore(["other"], "train", tmp_path)
    m = match_pair(imgs["in2.raw"], imgs["ref2.raw"])
    p = train.pairs[0]
    dy, dx = p.shift
    assert (dy, dx) == tuple(m["align"]["shift"]) and np.allclose(p.gain, m["gain"])
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(8):
        s = train.random_tile(rng)
        x, y = s["x"], s["y"]
        assert x.shape == y.shape == (64, 64, 3) and x.dtype == y.dtype == np.float64
        assert 0.0 <= x.min() and x.max() <= 1.0 and 0.0 <= y.min() and y.max() <= 1.0 and s["clipped"] <= 0.01
        assert s["capture"].shape == (7,) and s["capture"][2] > 0 and s["capture"][-1] == 1.0 and s["pair"] == 0
        y0, x0 = s["y0"], s["x0"]
        oy, ox = y0 - max(0, -dy), x0 - max(0, -dx)                 # the tile inside match_pair's overlap crop
        assert np.allclose(x, m["x"][oy:oy + 64, ox:ox + 64], atol=2e-3)
        assert np.allclose(y, m["t"][oy:oy + 64, ox:ox + 64], atol=2e-3)
        assert np.mean(m["valid"][oy:oy + 64, ox:ox + 64]) >= 0.99
        seen.add((y0, x0))
    assert len(seen) > 1
    ylo, yhi, xlo, xhi = train.bounds(0)
    assert ylo == max(0, -dy) and xlo == max(0, -dx) and yhi == min(192, 192 - dy) - 64 and xhi == min(256, 256 - dx) - 64
    x, t, clipped = train.tile_at(0, ylo, xlo)
    assert x.shape == (64, 64, 3) and 0.0 <= clipped <= 1.0 and t.max() <= 1.0
    text = train.summary()
    assert "split train: 1 pairs in 1 scenes (fake 1 drawn 100%)" in text and "electrons at clip" in text
    assert CLIP > 0.9


def test_real_store_leaves_out_a_reference_the_clean_exclude_names(tmp_path, monkeypatch):
    import harness.data.label as label
    imgs, pairs = _fake_pairs(np.random.default_rng(5))
    build_store("fake", ("train", "val"), scale=1, root=tmp_path, workers=1, log=None, decode=_decoder(imgs, []),
                pairs=pairs)
    real = label.clean_settings
    monkeypatch.setattr(label, "clean_settings",
                        lambda s: {"exclude": ["REF2.*"]} if s == "fake" else real(s))   # fnmatch, either case
    with pytest.raises(ValueError, match="no stored pairs"):
        RealStore(["fake"], "train", tmp_path)                    # s2 was its only pair; no rebuild needed
    assert [p.meta["scene"] for p in RealStore(["fake"], "val", tmp_path).pairs] == ["s0"]


def test_hdrplus_synthetic_chart_bursts_are_not_clean_targets_or_real_pairs():
    from harness.data.label import file_filter
    keep = file_filter("hdr-plus-burst")
    assert keep("results_20171023/0006_20160721_163256_525/merged.dng")
    assert not keep("results_20171023/synthetic_lines_day/merged.dng")
    assert not keep("results_20171023/synthetic_ramp_night_motion_blur/merged.dng")
    assert not keep("bursts/0006_20160721_163256_525/payload_N000.dng")   # only merges are clean targets


def test_realstore_is_wired_into_the_data_cli():
    from harness.data.cli import add_parser, cmd_realstore
    ap = argparse.ArgumentParser()
    add_parser(ap.add_subparsers())
    a = ap.parse_args(["data", "realstore", "--sources", "rawnind", "hdr-plus-burst", "--splits", "train", "val",
                       "--workers", "2"])
    assert a.func is cmd_realstore and a.sources == ["rawnind", "hdr-plus-burst"] and a.splits == ["train", "val"]
    assert a.workers == 2 and a.scale is None and a.tile == 256
