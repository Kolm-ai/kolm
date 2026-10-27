"""Noise model validation (DATA_PLAN section 4 step 5): the mean-variance fit recovers gain, read noise and
full well from a synthetic Poisson + Gaussian burst, flags row noise, scales gain with ISO, and refuses
to fit from too few bins."""
import csv
import math

import numpy as np
import pytest

from harness.data import noise as NZ


def _burst(n_frames=24, shape=(96, 128), gain=0.25, read_e=2.0, black=63.0, white=1023.0, row_sigma_e=0.0, seed=0,
           flicker=0.0, edges=False, shift_every=0, dark_first=False):
    """Frames in DN from a smooth exposure ramp: electrons ~ Poisson(e * (1 + flicker_f)) + N(0, read_e)
    (+ per-row offsets); flicker is the per-frame relative illumination jitter; edges adds a sharp stripe
    pattern and shift_every moves the scene by one pixel on every k-th frame; dark_first makes frame 0 7 % dark."""
    rng = np.random.default_rng(seed)
    fw_e = (white - black) * gain
    ramp = np.linspace(0.01, 0.95, shape[1])[None, :] * np.ones((shape[0], 1))
    e = ramp * fw_e
    if edges:
        stripes = (np.arange(shape[0])[:, None] // 12 % 2 == 0) & (np.arange(shape[1])[None, :] // 16 % 2 == 0)
        e = np.where(stripes, e * 0.4 + 0.02 * fw_e, e)
    frames = []
    for i in range(n_frames):
        ei = e * (1.0 + rng.normal(0.0, flicker))
        if shift_every and i % shift_every == 0:
            ei = np.pad(ei, ((1, 0), (0, 0)), mode="edge")[:-1, :]      # the scene moves one row (mean unchanged, no wrap)
        if dark_first and i == 0:
            ei = ei * 0.93
        x = rng.poisson(ei) + rng.normal(0.0, read_e, size=shape)
        if row_sigma_e:
            x = x + rng.normal(0.0, row_sigma_e, size=(shape[0], 1))
        dn = np.clip(np.round(x / gain + black), 0, white)
        frames.append(dn.astype(np.float64))
    return np.stack(frames, axis=0), black, white


def test_synthetic_burst_recovers_gain_read_noise_and_full_well():
    frames, black, white = _burst()
    res = NZ.analyse_stack(frames, black, white)
    fit = res["fit"]
    assert fit["ok"] and fit["n_bins_fit"] >= 10
    assert fit["gain_e_per_dn"] == pytest.approx(0.25, rel=0.03)
    assert fit["read_noise_e"] == pytest.approx(2.0, rel=0.15)
    assert fit["full_well_e"] == pytest.approx((white - black) * 0.25, rel=0.03)
    assert fit["r2"] > 0.99 and abs(fit["quadratic_share_mid"]) < 0.05
    assert abs(res["shape"]["excess_kurtosis"]) < 0.2 and res["shape"]["tail_4sigma"] < 5e-4
    assert 0.8 < res["row_noise_ratio"] < 1.3
    assert abs(res["drift_sigma"]) < 4.0
    assert res["model_vs_measured"]["median_abs_rel_error"] < 0.05
    assert res["n_frames"] == 24 and res["clipped_fraction"] == 0.0


def test_plane_stats_streams_the_same_answer_as_the_stack():
    frames, black, white = _burst(n_frames=12, shape=(64, 96))
    ps = NZ.PlaneStats(black, white, keep=4)
    for f in frames:
        ps.add(f)
    a, b = ps.finish(), NZ.analyse_stack(frames, black, white, keep_for_shape=4, correct_drift=False)
    assert a["fit"]["gain_e_per_dn"] == pytest.approx(b["fit"]["gain_e_per_dn"])
    assert a["row_noise_ratio"] == pytest.approx(b["row_noise_ratio"]) and a["n_frames"] == 12


def test_frame_scaling_removes_illumination_flicker():
    frames, black, white = _burst(n_frames=30, shape=(120, 160), flicker=0.08, seed=2)
    raw = NZ.analyse_stack(frames, black, white, correct_drift=False)
    fixed = NZ.analyse_stack(frames, black, white)
    assert raw["fit"]["quadratic_share_mid"] > 0.1                    # flicker adds variance ~ mean squared
    assert raw["fit"]["gain_e_per_dn"] < 0.9 * fixed["fit"]["gain_e_per_dn"]
    assert raw["drift_corrected"] is False and fixed["drift_corrected"] is True
    assert fixed["fit"]["gain_e_per_dn"] == pytest.approx(0.25, rel=0.03)
    assert fixed["fit"]["read_noise_e"] == pytest.approx(2.0, rel=0.2)
    assert abs(fixed["fit"]["quadratic_share_mid"]) < 0.05 and abs(fixed["shape"]["excess_kurtosis"]) < 0.3
    lo, hi = fixed["frame_scale_min_max"]
    assert lo < 0.99 < 1.01 < hi
    assert NZ.frame_scales([2.0, 4.0]).tolist() == [1.5, 0.75] and NZ.frame_scales([0.0, 1.0]).tolist() == [1.0, 1.0]


def test_outlier_first_frame_is_dropped_and_edges_are_masked():
    frames, black, white = _burst(n_frames=30, shape=(120, 160), dark_first=True, seed=4)
    res = NZ.analyse_stack(frames, black, white)
    assert res["n_frames_dropped"] == 1 and res["n_frames"] == 29
    assert res["fit"]["gain_e_per_dn"] == pytest.approx(0.25, rel=0.03)
    keep = NZ.outlier_frames([10.0, 10.05, 9.95, 10.0, 9.3])
    assert keep.tolist() == [True, True, True, True, False] and NZ.outlier_frames([1.0, 2.0]).all()
    frames, black, white = _burst(n_frames=30, shape=(120, 160), edges=True, shift_every=5, seed=5)
    res = NZ.analyse_stack(frames, black, white)
    assert 0.3 < res["flat_fraction"] < 0.95 and res["n_frames_dropped"] == 0
    assert res["edge_variance_ratio"] > 2.0                          # the shifted frames live at the stripe edges
    assert res["fit"]["gain_e_per_dn"] == pytest.approx(0.25, rel=0.04)
    assert res["fit"]["read_noise_e"] < res["fit_all_pixels"]["read_noise_e"]
    assert abs(res["shape"]["excess_kurtosis"]) < 0.5


def test_row_noise_is_flagged():
    frames, black, white = _burst(row_sigma_e=3.0, seed=1)
    res = NZ.analyse_stack(frames, black, white)
    assert res["row_noise_ratio"] > 1.5 and 0.7 < res["column_noise_ratio"] < 1.4


def test_cfa_split_and_fit_refuses_too_few_bins():
    img = np.arange(16, dtype=float).reshape(4, 4)
    planes = NZ.cfa_split(img)
    assert sorted(planes) == ["00", "01", "10", "11"] and planes["11"].tolist() == [[5.0, 7.0], [13.0, 15.0]]
    bins = {"mean_dn": [10.0, 20.0, 30.0], "var_median": [3.0, 5.5, 8.0], "var_mean": [3.0, 5.5, 8.0], "count": [100] * 3}
    out = NZ.fit_nlf(bins, 0.0, 100.0)
    assert out["ok"] is False and "usable bins" in out["reason"]


def _row(inst, iso, gain, read_e, fw):
    return {"instance": inst, "ok": True, "iso": iso, "camera": "GP", "camera_full": "Google Pixel",
            "gain_e_per_dn": gain, "read_noise_e": read_e, "full_well_e": fw}


def test_iso_scaling_slope_and_verdict():
    rows = [_row("a", 100, 4.0, 2.0, 4000.0), _row("b", 400, 1.0, 2.1, 3990.0),
            _row("c", 1600, 0.25, 1.9, 4010.0), {"instance": "d", "ok": False}]
    sc = NZ.iso_scaling(rows)
    s = sc["Google Pixel"]
    assert s["n_instances"] == 3 and s["log_gain_vs_log_iso_slope"] == pytest.approx(-1.0, abs=1e-9)
    assert s["read_noise_e_min_max"] == [1.9, 2.1] and s["full_well_e_median"] == 4000.0
    assert s["unity_gain_iso"] == pytest.approx(400.0, rel=1e-6)
    for r in rows[:3]:
        r.update(r2=0.999, quadratic_share_mid=0.01, excess_kurtosis=0.05, row_noise_ratio=1.1,
                 column_noise_ratio=1.0, drift_sigma=12.0, model_median_abs_rel_error=0.01,
                 flat_fraction=0.8, edge_variance_ratio=2.5, n_frames_dropped=1)
    v = NZ.verdict(rows, sc)
    assert v["linear_variance"] is True and v["shot_noise_scales_with_gain"] is True
    assert v["row_noise_present"] is False and v["column_noise_present"] is False and v["heavy_tails"] is False
    assert "slope -1.00" in v["text"] and "up to 12 sigma" in v["text"] and "3 outlier frames" in v["text"]
    assert NZ.verdict([rows[3]], {})["linear_variance"] is None


def test_instance_jobs_and_metadata(tmp_path):
    assert NZ.instance_jobs(None) == [] and NZ.instance_jobs(tmp_path / "nowhere") == []
    (tmp_path / "0063_DNG.zip").write_bytes(b"")
    (tmp_path / "notes.zip").write_bytes(b"")
    assert [p.name for p in NZ.instance_jobs(tmp_path)] == ["0063_DNG.zip"]
    csv_path = tmp_path / "meta.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance", "scene_name", "scene_number", "camera", "camera_full", "iso", "shutter",
                    "illuminant_temp", "brightness", "num_images", "zip_file", "zip_bytes"])
        w.writerow(["0063", "0063_001_GP_00100_00060_3200_N", "001", "GP", "Google Pixel", "100", "60", "3200",
                    "N", "150", "data/train/0063_DNG.zip", "1"])
    meta = NZ.sidd_metadata(csv_path)
    assert meta["0063"]["iso"] == 100 and meta["0063"]["camera_full"] == "Google Pixel"
    assert math.isfinite(meta["0063"]["illuminant_k"])
