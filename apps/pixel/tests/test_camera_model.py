"""Exposure decision, absolute radiometry, black pedestal and row noise in camera.py."""
import numpy as np
import pytest

from harness.camera import CaptureConstraints, PRESETS, simulate_frame, simulate_sequence
from harness.scene import Scene, SceneSpec, WEDGE_VALUES, crop


def _flat(v, h=64, w=64):
    return np.full((h, w, 3), v, dtype=np.float32)


def _cam(**kw):
    base = dict(bayer=False, residual_blur_um=0.0, aperture_mm=1e6, read_noise_e=0.0, full_well_e=20_000.0,
                bit_depth=16)
    base.update(kw)
    return CaptureConstraints(**base)


def test_best_case_exposes_for_the_scene_peak():
    scene = Scene(SceneSpec(width=192, height=108, supersample=2, n_frames=1))
    assert scene.peak_radiance() == pytest.approx(max(WEDGE_VALUES))
    c = PRESETS["best_case_earth"].replace(bayer=False, read_noise_e=0.0)
    out = simulate_sequence(scene, c)[0]
    top = scene.wedge[-1][0]
    assert crop(out, top, 2).mean() == pytest.approx(max(WEDGE_VALUES), rel=0.02)  # the 2.0 step is not clipped
    pinned = simulate_sequence(scene, c.replace(white_radiance=1.0))[0]
    assert crop(pinned, top, 2).mean() < 1.3  # the old exposure decision (radiance 1.0 at 0.8 FW) clips it


def test_relative_mode_ignores_exposure_time_for_noise():
    a = simulate_frame(_flat(0.5), _cam(exposure_s=1 / 100), 1, np.random.default_rng(0))
    b = simulate_frame(_flat(0.5), _cam(exposure_s=1 / 25), 1, np.random.default_rng(0))
    assert a.std() == pytest.approx(b.std(), rel=0.05)


def test_absolute_mode_electrons_follow_exposure_f_number_pitch_and_qe():
    c = _cam(scene_white_cd_m2=100.0, aperture_mm=25.0, focal_mm=50.0, exposure_s=0.01, pixel_pitch_um=4.0,
             quantum_efficiency=0.5, lens_transmission=1.0, full_well_e=1e6, bit_depth=20)
    g = c.electrons_per_unit_radiance()
    assert g == pytest.approx(100 * np.pi / 16 * 4.09e15 * 0.01 * 16e-12 * 0.5, rel=1e-9)
    assert c.replace(exposure_s=0.02).electrons_per_unit_radiance() == pytest.approx(2 * g)
    assert c.replace(aperture_mm=50.0).electrons_per_unit_radiance() == pytest.approx(4 * g)  # f/1 vs f/2
    assert c.replace(pixel_pitch_um=8.0).electrons_per_unit_radiance() == pytest.approx(4 * g)
    assert c.replace(quantum_efficiency=1.0).electrons_per_unit_radiance() == pytest.approx(2 * g)
    # shot noise follows: SNR of a flat patch grows with sqrt(exposure)
    s1 = simulate_frame(_flat(0.5), c, 1, np.random.default_rng(1))
    s4 = simulate_frame(_flat(0.5), c.replace(exposure_s=0.04), 1, np.random.default_rng(1))
    assert (s1.mean() / s1.std()) * 2 == pytest.approx(s4.mean() / s4.std(), rel=0.1)


def test_absolute_mode_clips_when_overexposed_and_exposure_for_peak_fixes_it():
    c = _cam(scene_white_cd_m2=634.0, exposure_s=0.1, full_well_e=5000.0, pixel_pitch_um=1.55, aperture_mm=2.0,
             focal_mm=4.0)
    assert c.electrons_per_unit_radiance(2.0) * 2.0 > c.full_well_e  # the 2.0 step would clip
    t = c.exposure_for_peak(2.0)
    fixed = c.replace(exposure_s=t)
    assert fixed.electrons_per_unit_radiance() * 2.0 == pytest.approx(0.8 * c.full_well_e, rel=1e-6)
    out = simulate_frame(_flat(2.0), fixed, 1, np.random.default_rng(0))
    assert out.mean() == pytest.approx(2.0, rel=0.02)


def test_saturation_iso_matches_the_pixel_iso_100_clip():
    pixel = CaptureConstraints(pixel_pitch_um=1.55, full_well_e=4662.0, quantum_efficiency=0.6)
    assert pixel.saturation_iso == pytest.approx(100.0, rel=0.03)


def test_black_pedestal_keeps_the_shadow_mean_unbiased():
    # radiance 0: with read noise the old clip-at-zero pipeline returned a positive mean (half a Gaussian)
    c = _cam(read_noise_e=20.0, full_well_e=20_000.0)
    out = simulate_frame(_flat(0.0, 128, 128), c, 1, np.random.default_rng(0))
    e_mean = out.mean() * c.electrons_per_unit_radiance()
    assert abs(e_mean) < 1.0  # electrons; the pedestal (160 e-) keeps the floor unclipped
    clipped = simulate_frame(_flat(0.0, 128, 128), c.replace(black_pedestal_e=0.0), 1, np.random.default_rng(0))
    assert clipped.mean() * c.electrons_per_unit_radiance() > 5.0  # ~ sigma * 0.4 = 8 e- without a pedestal


def test_row_noise_is_shared_along_rows():
    c = _cam(row_noise_e=50.0, read_noise_e=0.0)
    out = simulate_frame(_flat(0.5, 64, 256), c, 1, np.random.default_rng(0))[..., 0] * c.electrons_per_unit_radiance()
    row_means = out.mean(axis=1) - out.mean()
    col_means = out.mean(axis=0) - out.mean()
    assert row_means.std() > 5 * col_means.std()
    assert row_means.std() == pytest.approx(50.0, rel=0.3)
