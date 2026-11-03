"""Sub-pixel registration in the real store and the real-pair sets (realstore and realpairs with data.subpix): the
residual measured as a pair is written and by the backfill of a store written before, the store's targets and a
set's targets and camera photos read at it sit where the input does, a set's tiles are the plain set's, and the CLI
wiring."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from harness.data import subpix as S
from harness.data.realpairs import Pair, build_set, load_set, overlap, set_path
from harness.data.realstore import RealStore, build_store, measure_store, pair_json, store_dir

from test_realpairs import GAIN
from test_subpix import _scene

D = (3, -2)                                          # the integer part of every pair's shift
SHIFTS = [(0.3, -0.25), (-0.2, 0.35), (0.25, 0.2)]   # and its sub-pixel part, pair by pair
SPLITS = ["train", "train", "val"]
H, W = 321, 385


def _smooth_pairs(noise: float = 0.003):
    """Three pairs of a fake source on band-limited periodic scenes f: the input f with noise, the reference f
    read at p - D - s at GAIN, so x[i, j] ~ ref[i + D0 + s0, j + D1 + s1] / GAIN. The decoder's images, the pairs
    and the scenes."""
    rng = np.random.default_rng(11)
    imgs, pairs, clean = {}, [], {}
    for k, (s, split) in enumerate(zip(SHIFTS, SPLITS)):
        f = 0.6 * _scene(H, W, seed=20 + k, cut=0.4)
        ref = np.asarray(S._spectral_shift(f, (-(D[0] + s[0]), -(D[1] + s[1]))), dtype=np.float64) * GAIN
        imgs[f"in{k}.raw"] = np.clip(f + rng.normal(0.0, noise, f.shape), 0.0, 1.0)
        imgs[f"ref{k}.raw"] = np.clip(ref, 0.0, 1.0)
        clean[f"s{k}"] = f
        pairs.append(Pair("fake", f"s{k}", f"ref{k}.raw", f"in{k}.raw", split, iso=100.0, licence="CC0"))
    return imgs, pairs, clean


def _decoder(imgs):
    return lambda path, scale: (imgs[path.name], 14)


def _err(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def test_the_store_measures_each_pair_and_reads_its_targets_at_the_residual(tmp_path):
    imgs, pairs, clean = _smooth_pairs()
    build_store("fake", ("train", "val"), scale=1, root=tmp_path, workers=1, log=None, decode=_decoder(imgs),
                pairs=pairs)
    for k, s in enumerate(SHIFTS):
        meta = json.loads(pair_json("fake", pairs[k], tmp_path).read_text())
        assert meta["shift"] == list(D) and meta["subpix"]["trusted"] and meta["subpix"]["spread"] < 0.1
        assert meta["subpix"]["shift"] == pytest.approx(list(s), abs=0.03)
    plain = RealStore(["fake"], "train", tmp_path, tile=128)
    sp = RealStore(["fake"], "train", tmp_path, tile=128, subpix=True)
    assert len(sp.pairs) == 2 and "sub-pixel residual applied on 2 of 2 pairs (median 0.3" in sp.summary()
    assert "sub-pixel" not in plain.summary()
    for k, p in enumerate(sp.pairs):
        f = clean[p.meta["scene"]]
        for y0, x0 in ((60, 80), (0, 2), (H - 3 - 128, W - 128)):
            x, t, c = sp.tile_at(k, y0, x0)
            x0_, t0, c0 = plain.tile_at(k, y0, x0)
            want = f[y0:y0 + 128, x0:x0 + 128]
            assert c == c0 == 0.0 and np.array_equal(x, x0_)
            assert _err(t, want) < 0.25 * _err(t0, want) and _err(t, want) < 0.004
        x, t, c = sp.tile_at(k, 60, 80, max_clipped=-1.0)       # a tile the caller drops is not moved
        assert np.array_equal(t, plain.tile_at(k, 60, 80)[1])
    r = sp.random_tile(np.random.default_rng(0))
    assert r["y"].shape == (128, 128, 3) and _err(r["y"], clean[sp.pairs[r["pair"]].meta["scene"]][r["y0"]:r["y0"] + 128, r["x0"]:r["x0"] + 128]) < 0.004


def test_the_backfill_measures_a_store_written_before_and_skips_what_it_has(tmp_path):
    imgs, pairs, clean = _smooth_pairs()
    build_store("fake", ("train", "val"), scale=1, root=tmp_path, workers=1, log=None, decode=_decoder(imgs),
                pairs=pairs)
    first = {}
    for p in pairs:
        j = pair_json("fake", p, tmp_path)
        meta = json.loads(j.read_text())
        first[p.in_file] = meta.pop("subpix")
        j.write_text(json.dumps(meta))
    (j.parent / "x.tmp.json").write_text("{not json")       # an older build's part is no pair
    old = RealStore(["fake"], "train", tmp_path, tile=128, subpix=True)
    assert "applied on 0 of 2 pairs, 2 not measured" in old.summary() and all(p.residual is None for p in old.pairs)
    logged = []
    tally = measure_store("fake", tmp_path, workers=1, log=logged.append)
    assert tally["pairs"] == 3 and tally["measured"] == 3 and tally["trusted"] == 3 and tally["failed"] == 0
    assert tally["already"] == 0 and 0.25 < tally["median_px"] < 0.45
    for p in pairs:
        got = json.loads(pair_json("fake", p, tmp_path).read_text())["subpix"]
        assert got["shift"] == pytest.approx(first[p.in_file]["shift"], abs=0.01)    # float16 arrays, the same shift
    assert not list(store_dir("fake", tmp_path).rglob("*.part")) and any("3 measured" in line for line in logged)
    again = measure_store("fake", tmp_path, workers=1, log=None)
    assert again["already"] == 3 and again["measured"] == 0
    assert measure_store("fake", tmp_path, workers=1, log=None, again=True)["measured"] == 3
    assert "applied on 2 of 2 pairs" in RealStore(["fake"], "train", tmp_path, subpix=True).summary()
    with pytest.raises(FileNotFoundError):
        measure_store("nope", tmp_path, log=None)


def test_a_subpix_set_moves_its_targets_and_camera_on_the_plain_sets_tiles(tmp_path, monkeypatch):
    import harness.data.realpairs as rp
    imgs, pairs, clean = _smooth_pairs()
    c = (0.2, -0.3)                                             # the camera's photo against the reference
    cams = {f"ref{k}.raw": np.clip(np.asarray(S._spectral_shift(imgs[f"ref{k}.raw"], c), dtype=np.float64), 0, 1)
            for k in range(3)}
    monkeypatch.setattr(rp, "camera_file", lambda p: p.ref_file + ".jpg")
    monkeypatch.setattr(rp, "decode_camera", lambda path, scale: cams[path.name[:-4]])
    (tmp_path / "raw" / "fake").mkdir(parents=True)
    for k in range(3):
        (tmp_path / "raw" / "fake" / f"ref{k}.raw.jpg").write_bytes(b"")
    kw = dict(n=6, tile=64, scale=1, seed=0, splits=("train", "val"), root=tmp_path, log=None,
              decode=_decoder(imgs), pairs=pairs)
    plain = build_set("fake", **kw)
    logged = []
    sp = build_set("fake", **dict(kw, log=logged.append), subpix=True)
    path = set_path("fake", 0, 6, 64, 1, tmp_path, subpix=True)
    assert path.name.endswith("_k1_sp.npz") and path.exists() and set_path("fake", 0, 6, 64, 1, tmp_path).exists()
    assert [(r.pair, r.y0, r.x0) for r in sp.records] == [(r.pair, r.y0, r.x0) for r in plain.records]
    assert np.array_equal(sp.tiles, plain.tiles) and plain.subpix is None
    assert any("residual (" in line and "applied" in line for line in logged)
    for i, d in sp.subpix.items():
        k = int(sp.pairs[int(i)].scene[1:])
        assert d["target"]["trusted"] and d["target"]["shift"] == pytest.approx(list(SHIFTS[k]), abs=0.03)
        assert d["camera"]["trusted"]
        assert d["camera"]["shift"] == pytest.approx([SHIFTS[k][0] - c[0], SHIFTS[k][1] - c[1]], abs=0.05)
    assert sp.cameras is not None and sp.has_camera.all() and plain.has_camera.all()
    for n, r in enumerate(sp.records):
        f = clean[sp.pairs[r.pair].scene]
        fo, _ = overlap(f, f, *D)
        want = fo[r.y0:r.y0 + 64, r.x0:r.x0 + 64]
        assert _err(sp.targets[n], want) < 0.3 * _err(plain.targets[n], want)
        assert _err(sp.cameras[n], want) < 0.75 * _err(plain.cameras[n], want)   # the rendering fit is its own error
    again = load_set(path)
    assert again.subpix == sp.subpix and np.array_equal(again.targets, sp.targets)
    assert build_set("fake", **dict(kw, decode=None, pairs=None), subpix=True).subpix == sp.subpix


def test_subpix_is_wired_into_the_clis():
    from harness.data.cli import add_parser as add_data
    from harness.model.cli import add_parser as add_model
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    add_data(sub)
    add_model(sub)
    a = p.parse_args(["data", "subpix", "--sources", "rawnind", "si-hdr", "--workers", "8", "--background"])
    assert a.func.__name__ == "cmd_subpix" and a.sources == ["rawnind", "si-hdr"] and a.workers == 8
    assert a.background and not a.again
    d = p.parse_args(["data", "subpix"])
    assert d.sources == ["rawnind", "hdr-plus-burst"] and d.workers == 4
    assert p.parse_args(["data", "realpairs", "--subpix"]).subpix and not p.parse_args(["data", "realpairs"]).subpix
    t = p.parse_args(["model", "train", "--out", "o", "--subpix-real"])
    assert t.subpix_real and not p.parse_args(["model", "train", "--out", "o"]).subpix_real
    e = p.parse_args(["model", "eval-raw", "--ckpt", "x.pt", "--real", "--subpix"])
    assert e.real and e.subpix
