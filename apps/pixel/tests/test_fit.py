"""Gate fitting: the natural-image overshoot estimator, the KADID row addressing and the DCR fit."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from harness.data import fit
from harness.metrics import artifacts
from harness.metrics.natural import edge_overshoot_pair, unsharp
from harness.scene import Scene, SceneSpec

SMALL = SceneSpec(width=256, height=144, supersample=2, seed=3)


@pytest.fixture(scope="module")
def scene_truth():
    scene = Scene(SMALL)
    return scene, scene.render(0.0)


def test_pair_estimator_matches_scene_metric(scene_truth):
    scene, truth = scene_truth
    assert edge_overshoot_pair(truth, truth) == 0.0
    y0, y1, x0, x1 = scene.regions["slanted_edge"]
    for gain, sigma in ((0.1, 1.0), (0.5, 1.0), (0.25, 1.5)):
        sharp = unsharp(truth, gain, sigma)
        want = artifacts.edge_overshoot(sharp, scene)
        got_edge = edge_overshoot_pair(sharp[y0:y1, x0:x1], truth[y0:y1, x0:x1])
        got_whole = edge_overshoot_pair(sharp, truth)
        assert want > 0.02
        assert abs(got_edge - want) / want < 0.15, (gain, sigma, want, got_edge)
        assert 0.5 * want < got_whole < 2.0 * want, (gain, sigma, want, got_whole)


def test_pair_estimator_is_monotone_in_gain(scene_truth):
    _, truth = scene_truth
    vals = [edge_overshoot_pair(unsharp(truth, g, 1.0), truth) for g in (0.05, 0.1, 0.2, 0.4)]
    assert vals == sorted(vals) and vals[0] > 0


def test_kadid_index_block_order():
    assert fit.kadid_index(1) == 0
    assert fit.kadid_index(1, 1, 1) == 1
    assert fit.kadid_index(1, fit.TYPE_SHARPEN, 1) == 116
    assert fit.kadid_index(2) == fit.BLOCK
    assert fit.kadid_index(81, 25, 5) == 81 * fit.BLOCK - 1
    with pytest.raises(ValueError):
        fit.kadid_index(82)
    with pytest.raises(ValueError):
        fit.kadid_index(1, 26, 1)


def test_read_kadid_dmos_from_label_table(tmp_path):
    rows = ["dis_img_path,dis_type,ref_img_path,score",
            "KADID-10k/images/I01_24_03.png,High_Sharpen,KADID-10k/images/I01.png,2.14",
            "TID2013/distorted_images/i01_01_1.bmp,noise,TID2013/reference_images/I01.BMP,4.0",
            "KADID-10k/images/I81_01_05.png,GBLUR,KADID-10k/images/I81.png,1.5"]
    p = tmp_path / "0000.parquet"
    pq.write_table(pa.table({"text": rows}), p)
    assert fit.read_kadid_dmos([p]) == {(1, 24, 3): 2.14, (81, 1, 5): 1.5}


def test_iter_parquet_rows_spans_shards_and_row_groups(tmp_path):
    for k in range(2):
        tbl = pa.table({"image": [{"bytes": bytes([k, i]), "path": "z"} for i in range(3)]})
        pq.write_table(tbl, tmp_path / f"000{k}.parquet", row_group_size=2)
    got = {i: v["bytes"] for i, v in fit.iter_parquet_rows(sorted(tmp_path.glob("*.parquet")), {0, 2, 5})}
    assert got == {0: bytes([0, 0]), 2: bytes([0, 2]), 5: bytes([1, 2])}


def test_fit_dcr_curve_recovers_threshold_with_ci():
    rng = np.random.default_rng(1)
    refs = np.repeat(np.arange(40), 5)
    ladder = np.tile([1.0, 2.0, 3.0, 6.0, 12.0], 40) / 3.0
    m = ladder * np.exp(rng.normal(np.log(0.3), 0.3, refs.size))
    true_x50, true_k = np.log(0.4), 1.6
    y = fit.logistic_dcr(np.log(m), true_x50, true_k) + rng.normal(0.0, 0.15, m.size)
    res = fit.fit_dcr_curve(m, y, refs, n_boot=150, seed=0)
    want = fit._crossing_logistic(true_x50, true_k, 4.5)
    got = res["at"]["4.5"]
    assert abs(got["value"] - want) / want < 0.2
    assert got["ci95"][0] <= want <= got["ci95"][1]
    assert res["spearman_dmos_vs_log_metric"] < -0.9
    assert res["n"] == 200 and res["n_units"] == 40
    assert res["loglinear"]["at"]["4.0"]["value"] > 0


def test_derive_fit_gates_without_data_says_what_to_collect(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_DATA_ROOT", str(tmp_path))
    from harness.data.derive import derive_fit_gates
    assert "collect" in derive_fit_gates()


# ---- banding gate ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wide_scene_truth():
    # the banding estimator needs the gradient to be smooth at < 0.006 per px, which the 256-px scene is not
    scene = Scene(SceneSpec(width=512, height=288, supersample=2, seed=3))
    return scene, scene.render(0.0)


def test_banding_pair_matches_scene_metric_on_posterised_truth(wide_scene_truth):
    from harness.metrics.natural import banding_pair, quantize_srgb
    scene, truth = wide_scene_truth
    assert banding_pair(truth, truth) == 0.0
    y0, y1, x0, x1 = scene.regions["gradient"]
    prev = float("inf")
    for bits in (5, 6, 7):
        q = quantize_srgb(truth, bits)
        want = artifacts.banding_score(q, scene)
        got = banding_pair(q[y0:y1, x0:x1], truth[y0:y1, x0:x1])
        assert want > 0.02 and want < prev        # coarser quantisation, bigger steps
        assert 0.6 * want < got < 1.4 * want, (bits, want, got)
        prev = want


def test_banding_pair_ignores_smooth_tone_and_colour_changes(wide_scene_truth):
    from harness.metrics.natural import banding_pair
    _, truth = wide_scene_truth
    lum = truth @ np.array([0.2126, 0.7152, 0.0722])
    for name, dist in (("gain", truth * 1.2), ("gamma", truth ** 1.1),
                       ("saturation", np.clip(lum[..., None] + 1.3 * (truth - lum[..., None]), 0, None))):
        assert banding_pair(dist, truth) < 0.01, name


def test_step_contrast_is_weber_and_one_sided():
    ramp = np.linspace(0.02, 0.30, 90)
    assert artifacts.step_contrast(ramp) < 1e-9
    assert artifacts.step_contrast(ramp ** 1.3) < 0.005         # smooth tone curve of the steep scene ramp
    stair = np.repeat(np.linspace(0.1, 0.3, 9), 10)             # 8 band edges of 0.025
    assert abs(artifacts.step_contrast(stair) - 0.025 / (0.1 + artifacts.BANDING_BLACK)) < 1e-9
    # direction-free: the same bands read from the bright end; the first edge then sits one band higher
    assert abs(artifacts.step_contrast(stair[::-1]) - 0.025 / (0.125 + artifacts.BANDING_BLACK)) < 1e-9


def test_kadid_ladder_generic_and_quantization_type():
    assert fit.TYPE_QUANTIZATION == 22 and fit.KADID_TYPE_NAMES[22] == "Quantization"
    assert fit.kadid_index(1, fit.TYPE_QUANTIZATION, 1) == 106
    assert fit.kadid_sharpen_ladder.__name__ == "kadid_sharpen_ladder"   # wrapper kept for the overshoot fit


def _bmp_bytes(arr8):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr8, "RGB").save(buf, format="BMP")
    return buf.getvalue()


def test_tid2013_ladder_reads_rows_of_one_type(tmp_path):
    rng = np.random.default_rng(0)
    ref = (rng.uniform(0.2, 0.8, (24, 32, 3)) * 255).astype(np.uint8)
    rows = {"reference": [], "distorted": [], "mos": [], "reference_id": [], "distortion_id": [], "distortion_intensity": []}
    for t, lv, mos in ((7, 1, 5.0), (1, 1, 6.0), (7, 2, 4.0)):
        dist = (ref // (8 * lv)) * (8 * lv) if t == 7 else ref
        rows["reference"].append({"bytes": _bmp_bytes(ref), "path": "I01.BMP"})
        rows["distorted"].append({"bytes": _bmp_bytes(dist), "path": f"i01_{t:02d}_{lv}.bmp"})
        rows["mos"].append(mos); rows["reference_id"].append(1); rows["distortion_id"].append(t); rows["distortion_intensity"].append(lv)
    pq.write_table(pa.table(rows), tmp_path / "0000.parquet")
    got = fit.tid2013_ladder([tmp_path / "0000.parquet"], 7, lambda d, r: float(np.abs(d - r).mean()))
    assert [(g["level"], g["mos"]) for g in got] == [(1, 5.0), (2, 4.0)]
    assert got[1]["metric"] > got[0]["metric"] > 0
    res = fit.crosscheck_tid2013([tmp_path / "0000.parquet"], 7, lambda d, r: 1.0, "x")
    assert "too few" in res["note"]


def test_fit_support_rejects_weak_or_wide_fits():
    strong = {"at": {"4.5": {"ci95": [0.19, 0.26]}}, "spearman_dmos_vs_log_metric": -0.8, "n": 405, "n_zero_metric_dropped": 0}
    weak = {"at": {"4.5": {"ci95": [1e-9, 1e-3]}}, "spearman_dmos_vs_log_metric": -0.31, "n": 333, "n_zero_metric_dropped": 72}
    assert fit.fit_support(strong)["supported"] is True
    res = fit.fit_support(weak)
    assert res["supported"] is False
    assert set(res["failed"]) == {"spearman", "ci_ratio"}


def test_rated_upper_bound_takes_mildest_level_below_criterion():
    ladder = [{"level": 1, "metric_median": 0.14, "dmos_mean": 4.2},
              {"level": 2, "metric_median": 0.17, "dmos_mean": 3.9},
              {"level": 3, "metric_median": 0.21, "dmos_mean": 3.7}]
    ub = fit.rated_upper_bound(ladder)
    assert ub["level"] == 1 and ub["metric_median"] == 0.14
    assert fit.rated_upper_bound([{"level": 1, "metric_median": 0.01, "dmos_mean": 4.8}]) is None
