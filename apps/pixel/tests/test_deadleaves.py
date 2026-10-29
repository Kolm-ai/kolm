"""Procedural colour dead leaves (harness.model.deadleaves) and their place in the training stream: radii follow the
r^-3 law, the z-buffer renderer matches a painter's reference disc for disc on both of its paths, a tile is
reproducible, in range, flat inside its leaves and holds both saturated and neutral leaves, equal-luminance leaves
share one luminance, a dead-leaves sample is a rendered pair whose item reproduces its tile, and PairStream draws
dead leaves at their share and refuses a share outside [0, 1]."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import maximum_filter, minimum_filter

import harness.model.data as D
from harness.colorspace import luminance
from harness.metrics import STILL_METRICS
from harness.model.data import Normaliser, PairStream, dead_leaves_sample
from harness.model.deadleaves import DEAD_LEAVES, dead_leaves, leaf_colours, leaf_radii, render_discs

from test_realtrain import _tiny_sources, needs_data


def test_leaf_radii_follow_the_power_law():
    r = leaf_radii(np.random.default_rng(0), 200_000, 1.0, 64.0)
    assert r.min() >= 1.0 and r.max() <= 64.0
    norm = 1.0 - 64.0 ** -2
    assert abs(np.mean(r < 2.0) - (1.0 - 0.25) / norm) < 0.005          # P(r < 2) for density r^-3 on [1, 64]
    assert abs(np.mean((r >= 2.0) & (r < 4.0)) - (0.25 - 0.0625) / norm) < 0.005


def test_render_discs_matches_a_painter_reference_on_both_paths():
    rng = np.random.default_rng(1)
    size, ss = 40, 4
    radius = rng.permutation(np.concatenate([rng.uniform(0.2, 5.5, 200), rng.uniform(6.0, 15.0, 20)]))
    centre = rng.uniform(-5.0, 45.0, (len(radius), 2))
    z = render_discs(size, radius, centre, ss)
    S = size * ss
    ref = np.full((S, S), -1, dtype=np.int64)
    j = np.arange(S)
    for i in range(len(radius)):                                          # later discs drawn over earlier ones
        cy, cx, rs = centre[i, 0] * ss - 0.5, centre[i, 1] * ss - 0.5, radius[i] * ss
        ref[(j[:, None] - cy) ** 2 + (j[None, :] - cx) ** 2 <= rs ** 2] = i
    assert z.shape == (S, S) and np.array_equal(z, ref)
    vis = np.unique(z[z >= 0])
    assert (radius[vis] < 5.5).sum() > 20 and (radius[vis] > 6.0).sum() > 5     # both paths drawn and visible


def test_one_disc_is_sharp_to_a_pixel_and_covers_its_area():
    size, ss, r = 64, 4, 10.0
    z = render_discs(size, np.array([r]), np.array([[32.0, 32.0]]), ss)
    cover = (z == 0).reshape(size, ss, size, ss).mean(axis=(1, 3))
    assert abs(cover.sum() - np.pi * r * r) / (np.pi * r * r) < 0.01
    d = np.hypot(np.arange(size)[:, None] + 0.5 - 32.0, np.arange(size)[None, :] + 0.5 - 32.0)
    assert np.all(cover[d < r - 0.75] == 1.0) and np.all(cover[d > r + 0.75] == 0.0)


def test_a_tile_is_reproducible_in_range_flat_inside_its_leaves_with_saturated_and_neutral_leaves():
    a, b = dead_leaves(np.random.default_rng(3), 96), dead_leaves(np.random.default_rng(3), 96)
    assert a.shape == (96, 96, 3) and a.dtype == np.float64 and np.array_equal(a, b)
    assert not np.array_equal(a, dead_leaves(np.random.default_rng(4), 96))
    flat, sat, neutral = [], [], []
    for s in range(6):
        t = dead_leaves(np.random.default_rng(s), 96)
        assert t.min() >= 0.0 and t.max() <= 1.0 and t.flags.c_contiguous
        spread = (maximum_filter(t, size=(3, 3, 1)) - minimum_filter(t, size=(3, 3, 1))).max(axis=-1)
        ratio = t.max(axis=-1) / np.maximum(t.min(axis=-1), 1e-12)
        flat.append((spread == 0).mean()); sat.append((ratio > 20).mean()); neutral.append((ratio < 1.1).mean())
    assert np.mean(flat) > 0.2 and np.mean(sat) > 0.05 and np.mean(neutral) > 0.02


def test_equal_luminance_leaves_share_one_luminance_and_vary_in_colour():
    c = leaf_colours(np.random.default_rng(5), 2000, iso_lum=0.05)
    assert np.allclose(luminance(c), 0.05, rtol=1e-9, atol=0) and c.min() >= 0.0 and c.max() <= 1.0
    chroma = c / c.sum(axis=1, keepdims=True)
    assert np.ptp(chroma[:, 0]) > 0.5 and np.ptp(chroma[:, 2]) > 0.5
    hi = leaf_colours(np.random.default_rng(5), 2000, iso_lum=0.5)       # past pure blue's reach: clipped darker
    assert hi.max() <= 1.0 + 1e-12 and luminance(hi).max() <= 0.5 + 1e-12 and np.mean(np.isclose(luminance(hi), 0.5)) > 0.3
    free = leaf_colours(np.random.default_rng(5), 2000)
    assert np.allclose(free.max(axis=1), np.exp(np.log(free.max(axis=1)))) and free.max() <= 1.0
    assert free.max(axis=1).min() >= 2.0 ** -8 - 1e-12


@needs_data
def test_a_dead_leaves_sample_is_a_rendered_pair_whose_item_reproduces_its_tile():
    from harness.data.degrade import Distributions
    dists = Distributions.load()
    s = dead_leaves_sample(64, dists, np.random.default_rng(6))
    assert s["item"].source == DEAD_LEAVES and not s["item"].is_hdr and (s["y0"], s["x0"]) == (0, 0)
    assert s["x"].shape == s["y"].shape == (64, 64, 3) and set(s["metrics"]) == set(STILL_METRICS)
    assert np.isfinite(s["x"]).all() and np.isfinite(s["y"]).all()
    seed = int(s["item"].file)
    assert np.array_equal(dead_leaves(np.random.default_rng(seed), 64), dead_leaves(np.random.default_rng(seed), 64))
    again = dead_leaves_sample(64, dists, np.random.default_rng(6))
    assert np.array_equal(s["x"], again["x"]) and np.array_equal(s["y"], again["y"])


def test_pair_stream_draws_dead_leaves_at_their_share(tmp_path, monkeypatch):
    _tiny_sources(tmp_path, ["tiny"], size=70)
    monkeypatch.setattr(D, "split_of", lambda it: "train")
    monkeypatch.setattr(D.Distributions, "load", classmethod(lambda cls, root=None: None))

    def fake(level):
        return lambda *a, **k: {"x": np.full((64, 64, 3), level), "y": np.full((64, 64, 3), level), "restore": True,
                                "metrics": {m: 0.0 for m in STILL_METRICS}, "capture": np.ones(7)}
    monkeypatch.setattr(D, "dead_leaves_sample", fake(0.25))
    monkeypatch.setattr(D, "make_sample", fake(0.5))
    norm = Normaliser(np.zeros(len(STILL_METRICS)), np.ones(len(STILL_METRICS)))
    it = iter(PairStream(["tiny"], 64, norm, seed=0, root=tmp_path, p_dead_leaves=0.3))
    levels = [float(D.decode(next(it)[0].permute(1, 2, 0).numpy()).mean()) for _ in range(400)]
    n = sum(abs(v - 0.25) < 1e-3 for v in levels)
    assert 80 < n < 160 and n + sum(abs(v - 0.5) < 1e-3 for v in levels) == 400
    off = iter(PairStream(["tiny"], 64, norm, seed=0, root=tmp_path))
    assert next(off)[4] is False                 # a rendered pair: the shift search leaves it at zero
    assert all(abs(float(D.decode(next(off)[0].permute(1, 2, 0).numpy()).mean()) - 0.5) < 1e-3 for _ in range(20))
    with pytest.raises(ValueError, match="p_dead_leaves is a probability"):
        PairStream(["tiny"], 64, norm, seed=0, root=tmp_path, p_dead_leaves=1.5)
