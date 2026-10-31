"""Sub-pixel registration (harness/data/subpix.py): the residual's sign and size, the shift's fidelity."""
import numpy as np
import pytest

from harness.data import subpix as S


def _scene(h=641, w=769, seed=0, cut=0.45):
    """A periodic RGB scene with structure at every scale up to `cut` cycles a pixel, in (0.02, 0.9), on an odd
    grid: its shift by a phase ramp is exact (an even grid's Nyquist term has no one shift)."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    base = sum(gaussian_filter(rng.standard_normal((h, w)), s, mode="wrap") * s for s in (0.8, 2.0, 6.0, 20.0))
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
    base = np.real(np.fft.ifft2(np.where(np.hypot(fy, fx) <= cut, np.fft.fft2(base), 0.0)))
    base = (base - base.min()) / (base.max() - base.min())
    tint = np.array([0.9, 1.0, 0.8])
    return 0.02 + 0.88 * base[..., None] * tint


@pytest.mark.parametrize("s", [(0.3, -0.2), (0.5, 0.45), (-0.12, 0.07), (0.0, -0.35)])
def test_measure_reads_the_shift_and_its_sign(s):
    f = _scene()
    t = S._spectral_shift(f, (-s[0], -s[1]))        # t(p) = f(p - s), so t(p + s) = f(p)
    rng = np.random.default_rng(1)
    x = f + rng.normal(0.0, 0.004, f.shape)
    crop = (slice(40, 600), slice(40, 728))
    m = S.measure(x[crop], t[crop])
    assert m["shift"] == pytest.approx(list(s), abs=0.02)
    assert m["spread"] < 0.05 and m["trusted"]
    moved = S.fourier_shift(t[crop], tuple(m["shift"]))
    inner = (slice(32, -32), slice(32, -32))
    assert np.abs(moved[inner] - f[crop][inner]).max() < 0.01


def test_a_scene_that_is_not_one_shift_is_not_trusted():
    f = _scene()
    t = f.copy()
    h, w = f.shape[:2]
    t[:, w // 2:] = S._spectral_shift(f, (0.0, 0.8))[:, w // 2:]    # the right half moved, the left not
    m = S.measure(f, t)
    assert m["spread"] > S.TRUST_SPREAD and not m["trusted"]


def test_the_shift_is_an_all_pass_and_matches_the_periodic_one_inside():
    f = _scene(seed=3)
    d = (0.37, -0.41)
    exact = S._spectral_shift(f, d)                  # the periodic scene moved exactly
    crop = (slice(100, 500), slice(120, 620))
    moved = S.fourier_shift(f[crop], d)
    inner = (slice(S.MARGIN, -S.MARGIN), slice(S.MARGIN, -S.MARGIN))
    assert np.abs(moved[inner] - exact[crop][inner]).max() < 1.5e-3
    # the move keeps the power of every frequency (an all-pass): no blur
    p0 = np.abs(np.fft.fft2(f[..., 1])) ** 2
    p1 = np.abs(np.fft.fft2(exact[..., 1])) ** 2
    assert np.allclose(p0, p1, rtol=1e-6, atol=1e-6 * p0.max())


@pytest.mark.parametrize("y0, x0", [(200, 260), (0, 0), (641 - 128, 769 - 128), (5, 700 - 128)])
def test_a_tile_moved_on_its_window_matches_the_frame_moved(y0, x0):
    f = _scene(seed=4)
    d = (-0.28, 0.33)
    tile = 128
    exact = S._spectral_shift(f, d)
    got = S.shifted_window(f, y0, x0, tile, d)
    assert got.shape == (tile, tile, 3)
    want = exact[y0:y0 + tile, x0:x0 + tile]
    h, w = f.shape[:2]
    # away from the frame's edge (reflected there, so not the periodic scene) the two agree
    ys = slice(max(0, S.MARGIN - y0), tile - max(0, y0 + tile + S.MARGIN - h))
    xs = slice(max(0, S.MARGIN - x0), tile - max(0, x0 + tile + S.MARGIN - w))
    assert np.abs(got[ys, xs] - want[ys, xs]).max() < 2e-3
    assert np.abs(got - want).max() < 0.1          # at the frame's own edge what lies beyond is unknown


def test_no_shift_is_the_frame_itself_and_holes_stay():
    f = _scene(seed=5)[:200, :240]
    big = _scene(h=901, seed=6)[:700, :600]
    blocks = S.fourier_shift(big, (0.2, 0.1))
    whole = S.shifted_window(big, 0, 0, (700, 600), (0.2, 0.1))
    inner = (slice(S.MARGIN, -S.MARGIN), slice(S.MARGIN, -S.MARGIN))
    assert np.abs(blocks[inner] - whole[inner]).max() < 2e-3      # the block seams do not show
    assert S.fourier_shift(f, (0.0, 0.0)) is f
    g = f.copy()
    g[50:60, 70:90] = np.nan
    out = S.fourier_shift(g, (0.2, 0.1))
    assert np.isnan(out[50:60, 70:90]).all() and not np.isnan(out[:40]).any()
    win = S.shifted_window(g, 30, 40, 64, (0.2, 0.1))
    assert np.isnan(win[20:30, 30:50]).all() and np.isfinite(win[:15]).all()


def test_residual_of_applies_only_a_trusted_one():
    assert S.residual_of({}) is None
    assert S.residual_of({"subpix": {"shift": [0.2, -0.1], "trusted": False}}) is None
    assert S.residual_of({"subpix": {"shift": [0.2, -0.1], "trusted": True}}) == (0.2, -0.1)


def test_a_hole_does_not_ring_into_its_neighbours_and_windows_are_fast():
    f = _scene(seed=7)
    d = (0.31, -0.27)
    exact = S._spectral_shift(f, d)
    g = f.copy()
    g[300:340, 400:450] = np.nan
    win = S.shifted_window(g, 250, 350, 160, d)
    want = exact[250:410, 350:510]
    near = np.zeros((160, 160), bool)
    near[47:93, 47:103] = True                    # within 3 px of the hole
    near[50:90, 50:100] = False
    assert np.isnan(win[50:90, 50:100]).all()
    assert np.abs(win[near] - want[near]).max() < 0.05     # a hole read as 0 put a step of the scene's level here
    for n in (128, 256, 512):
        a, b = S._span(1000, n, 3012)
        assert b - a >= n + 2 * S.MARGIN and S._smooth(2 * (b - a) - 1)
    assert S._span(0, 256, 3012)[0] == 0 and S._span(2900, 112, 3012)[1] == 3012
