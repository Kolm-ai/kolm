import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from harness.camera import PRESETS, simulate_sequence
from harness.display import DisplaySpec, ppd
from harness.metrics import artifacts, color, geometry, measure_all, resolution, temporal, tone
from harness.scene import Scene, SceneSpec, crop

SMALL = SceneSpec(width=256, height=144, supersample=2, seed=3)
CLIP = SceneSpec(width=256, height=144, supersample=2, seed=3, n_frames=6)


@pytest.fixture(scope="module")
def scene():
    return Scene(SMALL)


@pytest.fixture(scope="module")
def truth(scene):
    return scene.render(0.0)


@pytest.fixture(scope="module")
def clip_scene():
    return Scene(CLIP)


@pytest.fixture(scope="module")
def clip_truth(clip_scene):
    return clip_scene.truth_frames()


def test_ppd_31in_8k_at_60cm():
    assert abs(ppd(DisplaySpec()) - 115) < 2


def test_mtf50_drops_with_aperture(scene):
    base = PRESETS["diffraction_only"].replace(exposure_s=1 / 125)
    wide = simulate_sequence(scene, base.replace(aperture_mm=50 / 1.4))[0]
    narrow = simulate_sequence(scene, base.replace(aperture_mm=50 / 16))[0]
    m_wide, m_narrow = resolution.mtf50_edge(wide, scene), resolution.mtf50_edge(narrow, scene)
    assert m_wide > 1.5 * m_narrow


def test_truth_mtf50_is_pixel_limited(scene, truth):
    assert resolution.mtf50_edge(truth, scene) > 0.3


def test_snr_scales_with_photons(scene):
    c = PRESETS["best_case_earth"].replace(exposure_s=1 / 125, bayer=False, read_noise_e=0.0)
    a = simulate_sequence(scene, c, seed=5)[0]
    b = simulate_sequence(scene, c.replace(full_well_e=c.full_well_e * 4), seed=5)[0]
    gain_db = tone.shadow_snr_db(b, scene) - tone.shadow_snr_db(a, scene)
    assert abs(gain_db - 6.02) < 1.5


def test_delta_e_identical_is_zero(scene, truth):
    mean, mx = color.delta_e_stats(truth, scene)
    assert mean < 1e-4 and mx < 1e-4  # float32 render layer; real colour errors are >= 0.1


def test_truth_wins_tone(scene, truth):
    assert tone.dr_steps(truth, scene) == 13
    assert tone.highlight_loss_fraction(truth, scene) == 0.0
    assert tone.shadow_snr_db(truth, scene) == tone.SNR_CAP_DB


def test_edge_overshoot_catches_sharpening(scene, truth):
    assert artifacts.edge_overshoot(truth, scene) < 0.005
    sharp = truth + 1.5 * (truth - gaussian_filter(truth, (1.5, 1.5, 0)))
    assert artifacts.edge_overshoot(sharp, scene) > 0.02


def test_periodicity_catches_tiling(scene, truth):
    assert artifacts.periodicity(truth, scene) < 0.3
    tiled = truth.copy()
    y0, y1, x0, x1 = scene.regions["texture"]
    tile = truth[y0:y0 + 16, x0:x0 + 16]
    reps = ((y1 - y0) // 16 + 1, (x1 - x0) // 16 + 1, 1)
    tiled[y0:y1, x0:x1] = np.tile(tile, reps)[:y1 - y0, :x1 - x0]
    assert artifacts.periodicity(tiled, scene) > 0.5  # circular wrap of a non-multiple crop caps it ~0.75


def test_banding_catches_posterisation(scene, truth):
    assert artifacts.banding_score(truth, scene) < 1e-6
    poster = np.round(truth * 16) / 16
    assert artifacts.banding_score(poster, scene) > 0.01


def test_landmarks_and_shading(scene, truth):
    assert geometry.landmark_error_px(truth, scene, 0) < 0.1
    assert geometry.shading_rmse(truth, truth, scene, 0) == 0.0
    shifted = np.roll(truth, 2, axis=1)
    assert geometry.landmark_error_px(shifted, scene, 0) > 1.5


def test_texture_fidelity(scene, truth):
    assert abs(resolution.texture_hf_fidelity(truth, truth, scene) - 1.0) < 1e-9
    blurred = gaussian_filter(truth, (1.0, 1.0, 0))
    assert resolution.texture_hf_fidelity(blurred, truth, scene) < 0.6
    rng = np.random.default_rng(0)
    invented = truth.copy()
    r = scene.regions["texture"]
    invented[r[0]:r[1], r[2]:r[3]] = 0.4 + 0.05 * rng.standard_normal(crop(truth, r).shape)
    assert abs(resolution.texture_hf_fidelity(invented, truth, scene)) < 0.1


def test_temporal_truth_is_self_consistent(clip_scene, clip_truth):
    assert temporal.flow_warped_excess(clip_truth, clip_truth, clip_scene) == 0.0
    assert temporal.flicker(clip_truth, clip_scene) == 0.0
    assert temporal.motion_region_error(clip_truth, clip_truth, clip_scene) == 0.0
    assert temporal.trajectory_error_px(clip_truth, clip_scene) < 0.1


def test_flow_is_integer_and_exact(clip_scene, clip_truth):
    for k in range(CLIP.n_frames - 1):
        f = clip_scene.flow(k)
        assert np.all(f == np.round(f))
    rmse = temporal.flow_warped_rmse(clip_truth, clip_scene)
    assert rmse < 0.02  # only anti-aliased rims differ


def test_temporal_catches_flicker_and_motion_errors(clip_scene, clip_truth):
    rng = np.random.default_rng(1)
    noisy = clip_truth + 0.01 * rng.standard_normal(clip_truth.shape)
    assert temporal.flicker(noisy, clip_scene) > 0.01
    assert temporal.flow_warped_excess(noisy, clip_truth, clip_scene) > 0.005
    frozen = np.repeat(clip_truth[:1], CLIP.n_frames, axis=0)
    assert temporal.motion_region_error(frozen, clip_truth, clip_scene) > 0.05
    assert temporal.trajectory_error_px(frozen, clip_scene) > 1.0


def test_measure_all_never_resamples(scene, truth):
    with pytest.raises(ValueError):
        measure_all(truth[None, :100, :100], truth[None], scene)
