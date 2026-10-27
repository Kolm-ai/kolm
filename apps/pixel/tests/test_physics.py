import numpy as np

from pixel import physics
from pixel.colorsci import delta_e2000, display_encode, luminance, srgb_decode, srgb_encode
from pixel.display import Display


def test_bessel_j1_known_values():
    assert abs(physics.bessel_j1(1.0) - 0.4400506) < 1e-6
    assert abs(physics.bessel_j1(5.0) + 0.3275791) < 1e-6
    assert abs(physics.bessel_j1(-1.0) + 0.4400506) < 1e-6


def test_airy_psf_grows_with_f_number():
    fast = physics.airy_psf(2.8, 550e-9, 3.76e-6)
    slow = physics.airy_psf(16.0, 550e-9, 3.76e-6)
    assert abs(fast.sum() - 1.0) < 1e-5 and abs(slow.sum() - 1.0) < 1e-5
    assert fast.max() > slow.max()
    assert slow.shape[0] > fast.shape[0]


def test_ppd_reference_display():
    assert abs(physics.pixels_per_degree(31.5, 7680, 4320, 0.6) - 115.3) < 0.2
    assert abs(physics.pixels_per_degree(27.0, 3840, 2160, 0.6) - 67.3) < 0.2
    d = Display()
    assert d.summary()["ppd"] == 115.3
    assert d.master_size() == (15360, 8640)


def test_photon_snr_doubles_with_four_times_photons():
    a = physics.snr_db(physics.photon_snr(1000.0))
    b = physics.snr_db(physics.photon_snr(4000.0))
    assert abs((b - a) - 6.02) < 0.05


def test_srgb_roundtrip_and_delta_e2000():
    x = np.random.default_rng(1).random((8, 8, 3)).astype(np.float32)
    assert np.allclose(srgb_decode(srgb_encode(x)), x, atol=1e-4)
    # Sharma et al. 2005 test pair 1
    assert abs(delta_e2000([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485])[0] - 2.0425) < 1e-3
    assert delta_e2000([50.0, 2.0, 3.0], [50.0, 2.0, 3.0])[0] == 0.0


def test_display_encode_keeps_highlights_unclipped():
    hdr = np.array([[[0.18, 0.18, 0.18], [8.0, 8.0, 8.0]]], dtype=np.float32)
    out = display_encode(hdr)
    assert 0.0 < luminance(out)[0, 0] < luminance(out)[0, 1] < 1.0
