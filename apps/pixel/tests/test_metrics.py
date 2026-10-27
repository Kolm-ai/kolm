import numpy as np

from pixel import degrade
from pixel.colorsci import display_encode, luminance
from pixel.metrics import evaluate_sequence, evaluate_still, generic_metrics
from pixel.gate import THRESHOLDS


def _still(small_scene, cand, ref):
    return evaluate_still(cand, ref, small_scene, 0.0)


def test_truth_scores_perfectly(small_scene, small_ref):
    m = _still(small_scene, small_ref, small_ref)
    assert m["mtf50"] == 0.5
    for k in ("highlight_loss_fraction", "shadow_noise_sigma", "banding_score", "halo_score", "blocking_score", "delta_e_p95"):
        assert m[k] == 0.0, k
    assert m["shadow_detail_corr"] == 1.0
    assert m["patch_delta_e"] < 0.05
    assert m["landmark_error_px"] < 0.02
    assert m["tiling_score"] < 0.6
    for name, (op, limit) in THRESHOLDS.items():
        if name in m:
            assert (m[name] <= limit) if op == "<=" else (m[name] >= limit), name


def test_each_degradation_trips_its_metric(small_scene, small_ref):
    base = _still(small_scene, small_ref, small_ref)
    cases = {
        "blur": ({"sigma": 1.5}, "mtf50", "lower"),
        "motion": ({"length": 6.0, "angle": 0.3}, "mtf50", "lower"),
        "downsample": ({"factor": 2.0}, "mtf50", "lower"),
        "noise": ({"photons": 200.0}, "shadow_noise_sigma", "higher"),
        "jpeg": ({"quality": 30}, "blocking_score", "higher"),
        "clip": ({"ev": 1.0}, "highlight_loss_fraction", "higher"),
        "crush": ({"black": 0.1}, "shadow_detail_corr", "lower"),
        "posterize": ({"bits": 5}, "banding_score", "higher"),
        "sharpen": ({"amount": 1.5, "radius": 2.0}, "halo_score", "higher"),
        "tile": ({"period": 16}, "tiling_score", "higher"),
        "color": ({"da": 5.0, "db": 0.0}, "patch_delta_e", "higher"),
    }
    for op, (params, metric, direction) in cases.items():
        cand = degrade.apply(small_ref, [(op, params)], seed=3)
        m = _still(small_scene, cand, small_ref)
        if direction == "higher":
            assert m[metric] > base[metric] + 0.01, (op, metric, m[metric])
        else:
            assert m[metric] < base[metric] - 0.01, (op, metric, m[metric])


def test_gate_thresholds_separate_artifacts(small_scene, small_ref):
    sharp = _still(small_scene, degrade.apply(small_ref, [("sharpen", {"amount": 1.5, "radius": 2.0})]), small_ref)
    assert sharp["halo_score"] > THRESHOLDS["halo_score"][1]
    blocky = _still(small_scene, degrade.apply(small_ref, [("jpeg", {"quality": 30})]), small_ref)
    assert blocky["blocking_score"] > THRESHOLDS["blocking_score"][1]
    tiled = _still(small_scene, degrade.apply(small_ref, [("tile", {"period": 16})]), small_ref)
    assert tiled["tiling_score"] > THRESHOLDS["tiling_score"][1]
    banded = _still(small_scene, degrade.apply(small_ref, [("posterize", {"bits": 5})]), small_ref)
    assert banded["banding_score"] > THRESHOLDS["banding_score"][1]
    clipped = _still(small_scene, degrade.apply(small_ref, [("clip", {"ev": 1.0})]), small_ref)
    assert clipped["highlight_loss_fraction"] > THRESHOLDS["highlight_loss_fraction"][1]
    blurred = _still(small_scene, degrade.apply(small_ref, [("blur", {"sigma": 1.5})]), small_ref)
    assert blurred["highlight_loss_fraction"] <= THRESHOLDS["highlight_loss_fraction"][1]  # blur is not clipping
    assert blurred["halo_score"] <= THRESHOLDS["halo_score"][1]


def test_temporal_metrics(small_scene):
    refs = [display_encode(small_scene.render(float(i))) for i in range(3)]
    truth = evaluate_sequence(refs, refs, small_scene)
    assert truth["flicker_score"] == 0.0 and truth["temporal_noise"] == 0.0
    assert truth["motion_error_px"] < 0.1
    frozen = evaluate_sequence([refs[0]] * 3, refs, small_scene)
    assert frozen["motion_error_px"] > 1.0  # analytic flow is 1.5 px/frame, candidate moved 0
    assert frozen["landmark_error_px"] > 0.5
    flicker = evaluate_sequence([np.clip(r + (0.01 if i % 2 else 0.0), 0, 1) for i, r in enumerate(refs)], refs, small_scene)
    assert flicker["flicker_score"] > THRESHOLDS["flicker_score"][1]


def test_generic_metrics_need_no_scene(small_ref):
    clean = generic_metrics(small_ref, small_ref)
    assert clean["psnr"] == 99.0 and clean["mtf50"] == 0.5 and clean["noise_sigma"] == 0.0
    noisy = generic_metrics(degrade.apply(small_ref, [("noise", {"photons": 300.0})]), small_ref)
    assert noisy["noise_sigma"] > 0.01 and noisy["psnr"] < 40
    clipped = generic_metrics(degrade.apply(small_ref, [("clip", {"ev": 1.5})]), small_ref)
    assert clipped["clip_fraction"] > 0.05
