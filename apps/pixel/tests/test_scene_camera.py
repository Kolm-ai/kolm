import numpy as np

from pixel.camera import BestCaseCamera
from pixel.colorsci import display_encode, luminance
from pixel.metrics.temporal import estimate_shift
from pixel.ops import shift_subpixel
from pixel.scene import PATCH_COLORS, Scene, SceneSpec


def test_scene_is_deterministic_and_hdr(small_scene, small_truth):
    again = Scene(SceneSpec(seed=7, width=256, height=256, frames=3)).render(0.0)
    assert np.array_equal(small_truth, again)
    reg = small_scene.regions(0.0)
    hl = luminance(small_truth[reg.masks["highlight"]])
    assert 0.6 < (hl > 1.0).mean() < 0.85  # three quarters of the wedge is above full well
    assert luminance(small_truth[reg.masks["shadow"]]).mean() < 0.004
    for mask, col in zip(reg.patch_masks, PATCH_COLORS):
        assert np.allclose(small_truth[mask].mean(axis=0), col, atol=1e-3)


def test_scene_frames_are_exact_translations(small_scene):
    f0 = luminance(display_encode(small_scene.render(0.0)))
    f1 = luminance(display_encode(small_scene.render(1.0)))
    dx, dy = estimate_shift(f0, f1)
    vx, vy = small_scene.flow(0.0)
    assert abs(dx - vx) < 0.08 and abs(dy - vy) < 0.08
    lm0, lm1 = small_scene.landmark_positions(0.0), small_scene.landmark_positions(1.0)
    assert np.allclose(lm1 - lm0, [vx, vy])


def test_shift_estimator_on_synthetic_shift(small_ref):
    lum = luminance(small_ref)
    dx, dy = estimate_shift(lum, shift_subpixel(lum, 0.3, -0.7))
    assert abs(dx - 0.3) < 0.05 and abs(dy + 0.7) < 0.05


def test_camera_tradeoff_single_exposure(small_scene, small_truth):
    cam = BestCaseCamera()
    reg = small_scene.regions(0.0)
    ref = display_encode(small_truth)
    bright = cam.capture_linear(small_truth, exposure_ev=0.0, rng=np.random.default_rng(1))
    dark = cam.capture_linear(small_truth, exposure_ev=-3.0, rng=np.random.default_rng(1))
    hl = reg.masks["highlight"]
    assert bright.max() <= 1.0 + 1e-6  # clipped at full well
    assert (bright[hl].max(axis=-1) >= 0.999).mean() > 0.5
    assert dark[hl].max() > 4.0  # protected highlights
    sh = reg.masks["shadow"]
    noise_bright = (display_encode(bright)[sh] - ref[sh]).std()
    noise_dark = (display_encode(dark)[sh] - ref[sh]).std()
    assert noise_dark > 2.0 * noise_bright  # and paid for it in the shadows


def test_camera_motion_blur_and_bayer_reduce_resolution(small_truth, small_ref):
    from pixel.metrics.spectral import mtf50

    cam = BestCaseCamera()
    lum_ref = luminance(small_ref)
    still = cam.capture_display(small_truth, exposure_ev=0.0, motion_px=(0.0, 0.0), rng=np.random.default_rng(2))
    moving = cam.capture_display(small_truth, exposure_ev=0.0, motion_px=(4.0, 0.0), rng=np.random.default_rng(2))
    assert mtf50(lum_ref, lum_ref) == 0.5
    assert mtf50(luminance(still), lum_ref) < 0.5
    assert mtf50(luminance(moving), lum_ref) < mtf50(luminance(still), lum_ref)
