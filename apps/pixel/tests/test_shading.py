"""Lens shading in camera.py: corners get less light, the ISP's correction restores brightness and multiplies the
corner noise with it (DATA_PLAN.md section 4 step 11)."""
import numpy as np

from harness.camera import CaptureConstraints, shading_map, simulate_frame


def _flat(h=96, w=128, v=0.5):
    return np.full((h, w, 3), v, dtype=np.float64)


def _cam(**kw):
    return CaptureConstraints(bayer=False, residual_blur_um=0.0, read_noise_e=0.0, full_well_e=20_000.0,
                              bit_depth=16, **kw)


def test_shading_map_geometry():
    m = shading_map(9, 17, 1.0)
    assert m.shape == (9, 17) and abs(m[4, 8] - 1.0) < 1e-12
    for corner in (m[0, 0], m[0, -1], m[-1, 0], m[-1, -1]):
        assert abs(corner - 0.5) < 1e-12  # one stop down at every corner
    assert shading_map(9, 17, 0.0) is None


def test_no_shading_is_the_old_pipeline():
    a = simulate_frame(_flat(), _cam(), 1, np.random.default_rng(3))
    b = simulate_frame(_flat(), _cam(shading_corner_stops=0.0), 1, np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_corrected_shading_keeps_brightness_and_raises_corner_variance():
    out = simulate_frame(_flat(256, 256), _cam(shading_corner_stops=1.0), 1, np.random.default_rng(0))
    centre = out[112:144, 112:144, 1]
    corner = out[:16, :16, 1]
    assert abs(centre.mean() - 0.5) < 0.01 and abs(corner.mean() - 0.5) < 0.02  # brightness restored everywhere
    # the 16x16 corner block sits at mean r2 ~0.9, so it collects 2**-0.9 of the centre's photons; after the
    # correction its variance is about 2**0.9 ~ 1.9 times the centre's
    ratio = corner.var() / centre.var()
    assert 1.5 < ratio < 2.4


def test_uncorrected_shading_darkens_corners():
    out = simulate_frame(_flat(256, 256), _cam(shading_corner_stops=1.0, shading_corrected=False), 1,
                         np.random.default_rng(0))
    centre = out[112:144, 112:144, 1].mean()
    assert abs(centre - 0.5) < 0.01
    assert abs(out[:8, :8, 1].mean() / centre - 2 ** -0.94) < 0.05  # the 8x8 corner block, mean r2 ~0.94
