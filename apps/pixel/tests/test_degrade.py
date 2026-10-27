"""Synthetic degradation: sampler inside the measured distributions, identity reproduces the tile,
labeller resumable and deterministic, stored metrics match a recomputation."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from harness.colorspace import luminance
from harness.data.degrade import (EXPOSE_FOR_PEAK_STOPS, MID_GREY_FILL, PEAK_FILL, PEAK_PERCENTILE, DegradeParams,
                                  Distributions, PARAM_NAMES, degrade, expose_for_peak_stops, prepare_tile,
                                  sample_params)
from harness.data.label import Item, crop_boxes, label_tile, load_items, read_labels, read_ledger, run_label
from harness.data.tilemetrics import tile_metrics
from harness.metrics import STILL_METRICS

DERIVED = Path("data/derived")
pytestmark = pytest.mark.skipif(not (DERIVED / "camera_presets.json").exists(), reason="derived stats not built")


@pytest.fixture(scope="module")
def dists():
    return Distributions.load()


def _tile(rng, n=128):
    """Smooth coloured field plus edges and fine texture, in [0.05, 0.9]."""
    y, x = np.mgrid[0:n, 0:n] / n
    base = np.stack([0.3 + 0.4 * x, 0.3 + 0.3 * y, 0.5 - 0.2 * x * y], -1)
    base[n // 4:n // 2, n // 4:n // 2] *= 0.3
    base += 0.05 * gaussian_filter(rng.standard_normal((n, n, 3)), 0.7)
    return np.clip(base, 0.05, 0.9)


def test_expose_for_peak_places_the_scene_peak_at_the_drawn_fill(dists):
    """With the policy on, prepare_tile lands the tile's p99.9 luminance at peak_fill of the clip level, so the
    restore target (the same prepare) is a legitimately dark frame, not a mean-shift defect."""
    rng = np.random.default_rng(3)
    tile = _tile(rng)
    for peak_over_median, fill in ((6.0, 0.5), (12.0, 0.8), (20.0, 1.0)):
        stops = expose_for_peak_stops(peak_over_median, fill)
        assert EXPOSE_FOR_PEAK_STOPS[0] <= stops <= EXPOSE_FOR_PEAK_STOPS[1]
        p = DegradeParams(expose=1, exposure_stops=stops, scene_peak_over_median=peak_over_median)
        t = prepare_tile(tile, p)
        lum = luminance(t)
        med, peak = float(np.median(lum)), float(np.percentile(lum, PEAK_PERCENTILE))
        assert math.isclose(med, MID_GREY_FILL * 2.0 ** stops, rel_tol=1e-6)
        if math.isclose(peak / med, peak_over_median, rel_tol=0.05):       # the stretch reached the drawn range
            assert math.isclose(peak, fill, rel_tol=0.05), (peak_over_median, fill, peak)
        else:                                                                # capped stretch: the peak lands below
            assert peak < fill + 1e-9
    assert expose_for_peak_stops(1e6, 0.5) == EXPOSE_FOR_PEAK_STOPS[0]
    assert expose_for_peak_stops(1.0, 1.0) == EXPOSE_FOR_PEAK_STOPS[1]
    # the mixture reaches dark captures: over many draws the median exposure is below the measured offsets
    rng = np.random.default_rng(5)
    stops_drawn = np.array([sample_params(rng, dists).exposure_stops for _ in range(400)])
    assert (stops_drawn < -0.5).mean() > 0.2, stops_drawn.mean()


def test_sampler_draws_lie_inside_measured_distributions(dists):
    rng = np.random.default_rng(1)
    lens_names = {name for name, _, _ in dists.lenses}
    isp = {k: dists.isp_values(k) for k in ("jpeg", "mean_shift", "contrast", "saturation", "chromatic_aberration",
                                            "wb_r_over_g", "wb_b_over_g")}
    wb_r = isp["wb_r_over_g"]
    for _ in range(500):
        p = sample_params(rng, dists)
        if p.camera_on:
            assert p.preset in dists.presets
            assert any(math.isclose(p.iso, v) for v in dists.iso_values)
            measured = any(math.isclose(p.exposure_stops, v) for v in dists.exposure_offset_stops)
            fill = p.scene_peak_over_median * MID_GREY_FILL * 2.0 ** p.exposure_stops   # where the policy put the peak
            for_peak = (PEAK_FILL[0] - 1e-9 <= fill <= PEAK_FILL[1] + 1e-9
                        or any(math.isclose(p.exposure_stops, b) for b in EXPOSE_FOR_PEAK_STOPS))
            assert measured or for_peak, p.exposure_stops
            assert any(math.isclose(p.shading_corner_stops, v) for v in dists.shading_stops)
            assert p.lens in lens_names and p.residual_blur_um >= 0 and 0 <= p.field_fraction <= 1
            assert p.read_noise_e > 0 and p.full_well_e > 0
        assert any(math.isclose(p.scene_peak_over_median, v) for v in dists.peak_over_median)
        assert 0 <= p.ca_shift_px <= isp["chromatic_aberration"].max() + 1e-9
        for name, key in (("contrast_gain", "contrast"), ("mean_shift", "mean_shift"), ("saturation", "saturation")):
            v = getattr(p, name)
            assert v == 1.0 or v == 0.0 or any(math.isclose(v, u) for u in isp[key]), (name, v)
        assert p.jpeg_quality == 0 or any(p.jpeg_quality == int(q) for q in isp["jpeg"])
        assert wb_r.min() / wb_r.max() - 1e-9 <= p.cast_r <= wb_r.max() / wb_r.min() + 1e-9
        assert p.posterise_bits == 0 or 3 <= p.posterise_bits <= 8
        assert set(p.to_dict()) == set(PARAM_NAMES)


def test_zero_strength_parameters_reproduce_the_clean_tile():
    t = _tile(np.random.default_rng(2))
    p = DegradeParams()
    assert p.is_identity()
    assert np.array_equal(degrade(t, p, seed=7), t)
    m = tile_metrics(t, t)
    assert set(m) == set(STILL_METRICS)
    assert m["mtf50_cpp"] == pytest.approx(0.5) and m["texture_hf_fidelity"] == pytest.approx(1.0)
    for k in ("delta_e_mean", "delta_e_max", "edge_overshoot", "banding_score", "periodicity",
              "landmark_error_px", "shading_rmse"):
        assert m[k] == pytest.approx(0.0, abs=1e-9), k


def test_tile_metrics_read_blur_and_shift():
    t = _tile(np.random.default_rng(3), 256)
    blurred = np.stack([gaussian_filter(t[..., c], 1.0) for c in range(3)], -1)
    assert tile_metrics(blurred, t)["mtf50_cpp"] == pytest.approx(0.187, abs=0.02)
    shifted = np.roll(t, 1, axis=1)
    m = tile_metrics(shifted, t)
    assert m["landmark_error_px"] == pytest.approx(1.0, abs=0.05) and m["mtf50_cpp"] == pytest.approx(0.5, abs=0.02)


@pytest.fixture()
def tiny_root(tmp_path):
    from PIL import Image
    src = tmp_path / "raw" / "test-tiles"
    src.mkdir(parents=True)
    rng = np.random.default_rng(4)
    for i in range(4):
        img = (np.clip(gaussian_filter(rng.random((300, 300, 3)), 2) * 2.5, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(src / f"t{i}.png")
    return tmp_path


def _run(root, out, jobs=1, log=lambda *_: None):
    return run_label(["test-tiles"], 16, out=out, per_tile=2, crops=2, tile=64, jobs=jobs, chunk=1,
                     root=root, derived_root=DERIVED.resolve().parent, log=log)


def _key(r):
    return (r["file"], r["row"], r["y0"], r["x0"], r["draw"])


def _same(rows_a, rows_b):
    a, b = {_key(r): r for r in rows_a}, {_key(r): r for r in rows_b}
    assert a.keys() == b.keys()
    for k in a:
        for c in a[k]:
            va, vb = a[k][c], b[k][c]
            if c == "ms":
                continue
            if isinstance(va, float):
                assert (math.isnan(va) and math.isnan(vb)) or va == pytest.approx(vb, rel=1e-9, abs=1e-12), (k, c)
            else:
                assert va == vb, (k, c)


def test_labeller_is_resumable_and_deterministic(tiny_root):
    out = tiny_root / "labels"
    s1 = _run(tiny_root, out)
    assert s1["chunks"] == 4 and s1["rows_new"] == 16 and s1["chunks_skipped"] == 0
    rows1 = read_labels(out)
    assert len(rows1) == 16 and len(read_ledger(out)) == 4
    s2 = _run(tiny_root, out)
    assert s2["chunks"] == 0 and s2["chunks_skipped"] == 4 and s2["rows"] == 16
    victim = sorted(out.glob("labels-*.parquet"))[1]
    victim.unlink()
    s3 = _run(tiny_root, out)
    assert s3["chunks"] == 1 and s3["chunks_skipped"] == 3
    _same(rows1, read_labels(out))
    out2 = tiny_root / "labels2"
    _run(tiny_root, out2, jobs=2)
    _same(rows1, read_labels(out2))


def test_stored_metric_vector_matches_recomputation(tiny_root):
    out = tiny_root / "labels"
    _run(tiny_root, out)
    rows = read_labels(out)
    for r in rows[:6]:
        item = Item(r["source"], r["file"], r["row"], "", r["is_hdr"])
        img = load_items([item], tiny_root)[item]
        assert (r["y0"], r["x0"]) in crop_boxes(item, img.shape[0], img.shape[1], r["size"], 2)
        t = np.ascontiguousarray(img[r["y0"]:r["y0"] + r["size"], r["x0"]:r["x0"] + r["size"]])
        p = DegradeParams.from_dict({k[2:]: v for k, v in r.items() if k.startswith("p_")})
        m = label_tile(t, p, r["seed"], r["ss"], r["is_hdr"])
        for name in STILL_METRICS:
            s, v = r[f"m_{name}"], m[name]
            assert (math.isnan(s) and math.isnan(v)) or s == pytest.approx(v, rel=1e-9, abs=1e-12), name


def test_capture_profile_is_a_raw_pipeline_without_iqa_ladder_stages(dists):
    """The capture profile keeps the camera stage on and the ISP stages that were measured on real pipelines
    (tone, saturation, cast, JPEG) as coins; the unsharp and posterise stages, whose levels come from IQA
    ladders no measured ISP justifies, are off. The default profile is the unchanged seed-to-vector map."""
    from harness.data.degrade import PROFILES, sample_params as sp
    assert PROFILES == ("iqa", "capture")
    isp = {k: dists.isp_values(k) for k in ("jpeg", "mean_shift", "contrast", "saturation")}
    rng = np.random.default_rng(11)
    draws = [sp(rng, dists, "capture") for _ in range(300)]
    assert all(p.camera_on and p.unsharp_gain == 0.0 and p.posterise_bits == 0 for p in draws)
    assert all(p.preset in dists.presets and p.read_noise_e > 0 for p in draws)
    for p in draws:
        for name, key in (("contrast_gain", "contrast"), ("mean_shift", "mean_shift"), ("saturation", "saturation")):
            v = getattr(p, name)
            assert v == 1.0 or v == 0.0 or any(math.isclose(v, u) for u in isp[key]), (name, v)
        assert p.jpeg_quality == 0 or any(p.jpeg_quality == int(q) for q in isp["jpeg"])
    on = sum(p.jpeg_quality > 0 for p in draws) / len(draws)
    assert 0.35 < on < 0.65, on
    a = [sp(np.random.default_rng(s), dists) for s in range(20)]
    b = [sp(np.random.default_rng(s), dists, "iqa") for s in range(20)]
    assert a == b
    with pytest.raises(AssertionError):
        sp(rng, dists, "raw")
