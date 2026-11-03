"""self_metrics, the label of a tile against itself, agrees with the full tile_metrics path on every kind of tile
the training stream labels, and tile_metrics takes it only when one array is passed as both."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from harness.data import tilemetrics as T
from harness.metrics import STILL_METRICS
from harness.model.deadleaves import dead_leaves, leaf_colours


def _tiles():
    rng = np.random.default_rng(2)
    n = 256
    y, x = np.mgrid[0:n, 0:n] / n
    field = np.stack([0.3 + 0.4 * x, 0.3 + 0.3 * y, 0.5 - 0.2 * x * y], -1)
    field += 0.05 * gaussian_filter(rng.standard_normal((n, n, 3)), 0.7)
    hdr = np.exp(np.log(1e-4) * (1 - x) + np.log(8.0) * x)[..., None] * np.ones(3)   # deep shadows to over-range
    hdr *= 1 + 0.02 * rng.standard_normal((n, n, 3))
    black = field.copy()
    black[: n // 2] = 0.0                                                                 # exact zeros
    flat = np.full((n, n, 3), 0.18)
    c = leaf_colours(np.random.default_rng(3), 2, iso_lum=0.05)
    iso = np.where((x < 0.5)[..., None], c[0], c[1])                                       # chroma edge only
    assert np.ptp(T.luminance(iso)) < 1e-9 and np.abs(c[0] - c[1]).max() > 0.01
    return {"field": np.clip(field, 0.05, 0.9), "hdr": hdr, "black": np.clip(black, 0.0, 0.9), "flat": flat,
            "dead leaves": dead_leaves(np.random.default_rng(4), n), "equal luminance": iso}


@pytest.mark.parametrize("name", list(_tiles()))
def test_self_metrics_agree_with_the_full_path(name):
    t = _tiles()[name]
    fast, full = T.self_metrics(t), T.tile_metrics(t, t.copy())
    assert set(fast) == set(full) == set(STILL_METRICS)
    for k in STILL_METRICS:
        a, b = fast[k], full[k]
        assert np.isnan(a) == np.isnan(b), (name, k, a, b)
        if not np.isnan(a):
            assert a == pytest.approx(b, abs=1e-12), (name, k, a, b)


def test_tile_metrics_takes_the_identity_only_for_one_array():
    t = _tiles()["hdr"]
    calls = []
    real = T.self_metrics
    try:
        T.self_metrics = lambda r, *a: calls.append(r) or real(r, *a)
        T.tile_metrics(t, t)
        T.tile_metrics(t, t.copy())
    finally:
        T.self_metrics = real
    assert len(calls) == 1 and calls[0] is t
    m = T.tile_metrics(t, t)
    assert m["mtf50_cpp"] == 0.5 and m["highlight_loss_fraction"] == 0.0 and not np.isnan(m["shadow_snr_db"])
