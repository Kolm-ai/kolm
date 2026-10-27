"""Radiance reader round trip (flat and RLE), weighted percentiles, and the dynamic-range statistics on a
synthetic environment with known stops."""
import numpy as np

from harness.data import hdri as HD


def _env(h=64, w=128, floor=1.0, sky=16.0, sun=2.0 ** 18, seed=0):
    rng = np.random.default_rng(seed)
    y = np.full((h, w), floor)
    y[: h // 2] = sky                                   # upper hemisphere 4 stops above the ground
    y *= np.exp(rng.normal(0, 0.05, size=(h, w)))        # mild texture
    y[h // 4, w // 2] = sun                             # one sun pixel
    y[-1, :4] = 0.0                                     # a few clipped blacks
    return np.stack([y * 0.9, y, y * 1.1], axis=-1).astype(np.float32)


def test_rgbe_round_trip_flat_and_rle(tmp_path):
    img = _env()
    for rle in (False, True):
        p = tmp_path / f"e_{int(rle)}.hdr"
        HD.write_rgbe(p, img, rle=rle)
        back = HD.read_rgbe(p)
        assert back.shape == img.shape
        nz = img.max(axis=-1) > 0
        rel = np.abs(back[nz] - img[nz]) / img[nz].max(axis=-1, keepdims=True)
        assert rel.max() < 1 / 128                        # 8-bit mantissa with a shared exponent
        assert (back[~nz] == 0).all()


def test_weighted_percentile_and_latitude_weights():
    w = HD.latitude_weights(90, 4)
    assert abs(w.sum() - 1) < 1e-12 and w[45, 0] > w[0, 0] * 10      # equator rows weigh far more than the poles
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert HD.weighted_percentile(v, np.ones(4), 50.0)[0] == 2.0
    assert HD.weighted_percentile(v, np.array([0.0, 0.0, 0.0, 1.0]), 50.0)[0] == 4.0


def test_hdri_stats_recovers_known_stops():
    s = HD.hdri_stats(_env())
    assert s["ok"]
    assert abs(s["sky_over_ground"] - 4.0) < 0.3
    assert 3.5 < s["dr_stops"] < 5.0                     # p0.1 .. p99.9 spans ground to sky, not the sun
    assert 17.0 < s["dr_stops_full"] < 19.5              # the sun pixel sets the full range
    assert 13.0 < s["peak_over_median"] < 19.0
    assert 0 < s["zero_fraction"] < 0.01 and s["bright_fraction"] < 0.01


def test_run_joins_index_and_summarises(tmp_path):
    for i, sky in enumerate((8.0, 32.0, 32.0, 64.0, 64.0)):
        HD.write_rgbe(tmp_path / f"m{i}.hdr", _env(h=16, w=32, sky=sky, seed=i), rle=bool(i % 2))
    index = {f"m{i}": {"categories": ["outdoor"], "attributes": {"environment": "outdoor"}, "evs_cap": 12} for i in range(5)}
    out = HD.run(tmp_path, index, workers=1)
    assert out["n_ok"] == 5 and out["n_failed"] == 0
    assert out["by_category"]["outdoor"]["n"] == 5 and out["by_attribute"]["environment=outdoor"]["n"] == 5
    assert out["maps"][0]["evs_cap"] == 12 and out["all"]["dr_stops"]["median"] > 4.5
