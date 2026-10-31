"""noise="residual", the evaluation reading of shadow SNR and DR steps: what of the degraded tile no linear filter
of the clean tile explains. hp stays the default (labels and conditioning), the truth reaches the cap, and blur,
sharpening, a tone curve and a sub-pixel shift are not read as noise, while added noise reads as the oracle does."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from harness.colorspace import luminance
from harness.data import tilemetrics as T
from harness.metrics import STILL_METRICS

N, n, O = 512, 256, 128
SIG = 0.0004


def _scene():
    rng = np.random.default_rng(0)
    _, x = np.mgrid[0:N, 0:N] / N
    base = np.exp(np.log(0.002) * (1 - x) + np.log(0.03) * x)                       # shadows across the tile
    tex = gaussian_filter(rng.standard_normal((N, N)), 0.8, mode="wrap")
    tex /= tex.std()
    return (base * np.clip(1 + 0.4 * tex, 0.05, None))[..., None] * np.array([1.0, 0.9, 1.1]), \
        rng.normal(0, SIG, (n, n))


S, NZ = _scene()


def _crop(a):
    return a[O:O + n, O:O + n]


def _move(img, s):
    """the whole scene moved by s periodically, so a crop's interior is an exact sub-pixel shift"""
    f = np.exp(-2j * np.pi * (np.fft.fftfreq(N)[:, None] * s[0] + np.fft.fftfreq(N)[None, :] * s[1]))
    return np.real(np.fft.ifft2(np.fft.fft2(img, axes=(0, 1)) * f[..., None], axes=(0, 1)))


C = _crop(S)
B = (slice(T.RESID_BORDER_PX, n - T.RESID_BORDER_PX),) * 2


def _resid(d, c=C):
    return T.noise_metrics(d, c, "residual")["shadow_snr_db"]


def _oracle():
    return T.shadow_snr_db(luminance(C)[B], luminance(C)[B], (T._noise(NZ)[B], None))


def _same(a, b):
    return list(a) == list(b) and all(a[k] == b[k] or (np.isnan(a[k]) and np.isnan(b[k])) for k in a)


def test_hp_is_the_default_and_a_bad_mode_raises():
    d = C + NZ[..., None]
    assert _same(T.tile_metrics(d, C), T.tile_metrics(d, C, "hp"))
    assert _same(T.self_metrics(C), T.self_metrics(C, "hp"))
    assert set(T.tile_metrics(d, C, "residual")) == set(STILL_METRICS)
    for f in (lambda: T.tile_metrics(d, C, "nope"), lambda: T.self_metrics(C, "nope"),
              lambda: T.noise_metrics(d, C, "nope")):
        with pytest.raises(ValueError):
            f()


def test_only_the_noise_metrics_change():
    d = C + NZ[..., None]
    a, b = T.tile_metrics(d, C, "hp"), T.tile_metrics(d, C, "residual")
    drop = ("shadow_snr_db", "dr_steps")
    assert _same({k: a[k] for k in a if k not in drop}, {k: b[k] for k in b if k not in drop})


def test_truth_reaches_the_cap_on_both_paths():
    s, t = T.self_metrics(C, "residual"), T.tile_metrics(C, C.copy(), "residual")
    assert s["shadow_snr_db"] == t["shadow_snr_db"] == T.SNR_CAP_DB
    assert s["dr_steps"] == t["dr_steps"]
    assert T.self_metrics(C, "hp")["shadow_snr_db"] < 10                            # hp reads texture as noise


def test_white_noise_reads_as_the_oracle():
    d = C + NZ[..., None]
    assert _resid(d) == pytest.approx(_oracle(), abs=0.3)
    assert T.noise_metrics(d, C, "hp")["shadow_snr_db"] < _oracle() - 5


@pytest.mark.parametrize("s", [(0.1, 0.0), (0.5, 0.0), (1.0, 0.0), (0.4, -0.3), (1.7, 0.6)])
def test_a_shift_is_not_noise(s):
    d = _crop(_move(S, s))
    assert _resid(d) > 30
    assert _resid(d + NZ[..., None]) == pytest.approx(_oracle(), abs=0.5)


def test_blur_sharpening_and_tone_are_not_noise():
    bl = _crop(gaussian_filter(S, (1.2, 1.2, 0), mode="wrap"))
    sharp = C + 0.8 * (C - _crop(gaussian_filter(S, (1.0, 1.0, 0), mode="wrap")))
    for d in (bl, sharp, 0.9 * np.clip(C, 0, None) ** 1.1):
        assert _resid(d) > 28
    d = bl + NZ[..., None]
    dm, cb, _ = T.residual_noise(luminance(d), luminance(C), T.phase_shift(luminance(np.clip(d, 0, 1)),
                                                                            luminance(C)))
    assert _resid(d) == pytest.approx(T.shadow_snr_db(dm, cb, (T._noise(NZ)[B], None)), abs=0.3)


def test_a_uniform_patch_reads_as_hp():
    u = np.full((n, n, 3), 0.01)
    d = u + NZ[..., None]
    assert T.noise_metrics(d, u, "residual")["shadow_snr_db"] == \
        pytest.approx(T.noise_metrics(d, u, "hp")["shadow_snr_db"], abs=0.3)


def test_the_truth_has_the_most_dr_steps():
    best = T.self_metrics(C, "residual")["dr_steps"]
    for d in (C + NZ[..., None], C + 4 * NZ[..., None]):
        assert T.noise_metrics(d, C, "residual")["dr_steps"] <= best
