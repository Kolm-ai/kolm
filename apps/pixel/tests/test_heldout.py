"""The held-out real-capture evaluation (harness.model.heldout): the demo camera's parameters, the exposure policy,
the set and its cache, the scoring and the report. No torch: evaluate() on a checkpoint is in test_model.py."""
from __future__ import annotations

import argparse
import shutil

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from harness.data.label import Item, load_items
from harness.metrics import STILL_METRICS
from harness.model.data import capture_of_params, restore_target
from harness.model.heldout import (build_set, demo_camera_params, exposure_policy, load_set, render_set, score,
                                   set_path, to_markdown)

SOURCE = "test-raws"


@pytest.fixture()
def raw_root(tmp_path):
    from PIL import Image
    src = tmp_path / "raw" / SOURCE
    src.mkdir(parents=True)
    rng = np.random.default_rng(7)
    for i in range(4):
        img = (np.clip(gaussian_filter(rng.random((200, 240, 3)), 2) * 2.5, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(src / f"r{i}.png")
    Image.fromarray(np.full((40, 40, 3), 128, np.uint8)).save(src / "small.png")  # smaller than the tile
    (src / "broken.png").write_bytes(b"")                                            # the decoder rejects it
    return tmp_path


def test_demo_camera_params_are_the_demo_claims_camera():
    from harness.camera import PRESETS
    c = PRESETS["best_case_earth"]
    a = demo_camera_params(exposure_stops=-1.0)
    assert a.camera_on == 1 and a.expose == 1 and a.exposure_stops == -1.0 and a.preset == "best_case_earth"
    assert a.f_number == pytest.approx(1.4) and a.pixel_pitch_um == c.pixel_pitch_um and a.focal_mm == c.focal_mm
    assert a.full_well_e == c.full_well_e and a.read_noise_e == c.read_noise_e and a.bit_depth == c.bit_depth
    assert a.bayer == int(c.bayer) and a.residual_blur_um == c.residual_blur_um and a.white_fill == c.white_fill
    assert a.shading_corner_stops == 0 and a.ca_shift_px == 0                                 # the floor has none
    assert a.unsharp_gain == 0 and a.jpeg_quality == 0 and a.contrast_gain == 1.0 and a.saturation == 1.0  # no ISP
    cap = capture_of_params(a)
    assert cap.shape == (7,) and np.isfinite(cap).all() and cap[0] > 0


def test_exposure_policy_is_mid_grey_or_for_the_peak():
    tile = np.ones((16, 16, 3))
    tile[0, :4] = 40.0  # a bright strip: the 99.9th percentile sits far above the median
    stops = {exposure_policy(tile, np.random.default_rng(s)) for s in range(32)}
    assert 0.0 in stops and any(v < 0 for v in stops) and all(-4.0 <= v <= 1.0 for v in stops)
    flat = {exposure_policy(np.ones((16, 16, 3)), np.random.default_rng(s)) for s in range(32)}
    assert all(v >= 0 for v in flat) and any(v > 0 for v in flat)  # a flat tile's peak is its median: exposed up
    assert exposure_policy(np.zeros((8, 8, 3)), np.random.default_rng(0)) == 0.0


def test_set_is_seeded_cached_and_records_the_skipped_files(raw_root):
    hs = build_set(SOURCE, n=4, tile=64, seed=4, root=raw_root, log=None)  # seed 4 visits both bad files before the fourth good one
    assert hs.tiles.shape == (4, 64, 64, 3) and hs.tiles.dtype == np.float32 and hs.scale == 1
    assert sorted(r.file for r in hs.records) == [f"r{i}.png" for i in range(4)]
    assert set(hs.skipped) == {"small.png", "broken.png"}
    assert "smaller than the tile" in hs.skipped["small.png"] and "UnidentifiedImageError" in hs.skipped["broken.png"]
    assert len({(r.seed_a, r.seed_b) for r in hs.records}) == 4 and all(0 <= r.y0 <= 136 and 0 <= r.x0 <= 176 for r in hs.records)
    for crop, r in zip(hs.tiles, hs.records):  # the records reproduce the tiles from the raws
        it = Item(SOURCE, r.file)
        img = load_items([it], raw_root)[it]
        assert np.array_equal(crop, img[r.y0:r.y0 + 64, r.x0:r.x0 + 64].astype(np.float32))
    other = build_set(SOURCE, n=4, tile=64, seed=6, root=raw_root, log=None)
    assert [(r.y0, r.x0) for r in other.records] != [(r.y0, r.x0) for r in hs.records]
    with pytest.raises(ValueError, match="only 4 of 6 tiles .6 items, 2 skipped."):  # the short set visits everything
        build_set(SOURCE, n=6, tile=64, seed=0, root=raw_root, log=None)
    p = set_path(SOURCE, 4, 4, 64, raw_root)
    assert p.exists()
    shutil.rmtree(raw_root / "raw")  # the cache is the set: a second checkpoint needs no raws
    again = build_set(SOURCE, n=4, tile=64, seed=4, root=raw_root, log=None)
    assert again.records == hs.records and again.skipped == hs.skipped and np.array_equal(again.tiles, hs.tiles)
    assert load_set(p).records == hs.records


def test_score_ties_the_identity_and_never_loses_to_the_target(raw_root):
    hs = build_set(SOURCE, n=3, tile=64, seed=1, root=raw_root, log=None)
    samples = render_set(hs)
    assert len(samples) == 3 and all(s["x"].shape == (64, 64, 3) and s["y"].shape == (64, 64, 3) for s in samples)
    assert all(s["a"].camera_on == 1 and s["b"] == restore_target(s["a"]) and s["restore"] for s in samples)
    assert all(set(s["metrics"]) == set(STILL_METRICS) for s in samples)
    same = score(samples, lambda s: s["x"])
    assert same["n"] == 3 and set(same["metrics"]) == set(STILL_METRICS) and len(same["outputs"]) == 3
    for m, s in same["metrics"].items():
        assert s["beats"] == 0 and s["loses"] == 0 and s["median_verdict"] in ("ties", None), (m, s)
        assert s["input"] == s["output"]
    perfect = score(samples, lambda s: s["y"])
    for m, s in perfect["metrics"].items():  # smooth 64 px tiles have no edges: mtf50 and the chart metrics are unmeasurable
        assert s["loses"] == 0 and s["median_verdict"] in ("beats", "ties", None), (m, s)
        assert (s["median_verdict"] is None) == (s["input"] is None and s["output"] is None and s["ties"] + s["beats"] == 0)
    assert perfect["metrics"]["periodicity"]["median_verdict"] == "beats"    # the Bayer mosaic's periodicity is gone
    assert perfect["metrics"]["delta_e_mean"]["output"] == pytest.approx(0.0, abs=1e-6)
    rep = {**perfect, "ckpt": "x.pt", "step": 1, "blind": True, "source": SOURCE, "seed": 1, "tile": 64, "scale": 1,
           "preset": "best_case_earth", "ss": 1, "seconds": 0.0, "records": [],
           "skipped": {"broken.png": "UnidentifiedImageError: cannot identify image file"}}
    md = to_markdown(rep)
    assert "| mtf50_cpp |" in md and "withheld" in md and "broken.png" in md and "beats the simulated capture" in md


def test_eval_raw_is_wired_into_the_cli():
    from harness.model.cli import add_parser
    p = argparse.ArgumentParser()
    add_parser(p.add_subparsers(dest="cmd"))
    a = p.parse_args(["model", "eval-raw", "--ckpt", "x.pt", "--blind", "--n", "4"])
    assert a.func.__name__ == "cmd_eval_raw" and a.blind and a.n == 4 and a.out is None
    assert a.source == "raw-pixls-us" and a.tile == 256 and a.seed == 0 and a.save_tiles == 8
