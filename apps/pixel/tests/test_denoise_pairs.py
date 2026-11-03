"""The denoise-only pairs (MODEL_PLAN 20.12): with p_denoise a camera restore's target is the input's capture without
its noise (same optics, shading and mosaic), conditioned as a real pair is; with it off the stream is unchanged."""
from __future__ import annotations

import numpy as np
import pytest

import harness.model.data as D
from harness.data.degrade import DegradeParams, degrade
from harness.data.tilemetrics import tile_metrics
from harness.metrics import STILL_METRICS
from harness.model.data import (LOCAL_ISP_FIELDS, Normaliser, PairStream, capture_of_params, dead_leaves_sample,
                                denoise_target, input_shading, restore_target)
from harness.model.deadleaves import dead_leaves

from test_realtrain import _tiny_sources, needs_data


def _camera_state(**kw) -> DegradeParams:
    base = dict(camera_on=1, preset="p", pixel_pitch_um=1.4, half_diag_mm=4.0, focal_mm=5.0, f_number=1.8, iso=800.0,
                full_well_e=4000.0, white_fill=0.9, read_noise_e=3.0, row_noise_e=0.5, bit_depth=10, bayer=1,
                lens="l", field_fraction=0.6, residual_blur_um=1.1, ca_shift_px=0.3, shading_corner_stops=1.5,
                shading_corrected=0, contrast_gain=1.1, mean_shift=0.02, saturation=0.9, cast_r=1.05, cast_b=0.97,
                unsharp_gain=0.8, unsharp_sigma_px=1.2, posterise_bits=6, jpeg_quality=85, expose=1,
                exposure_stops=-0.5)
    base.update(kw)
    return DegradeParams(**base)


def test_the_denoise_target_keeps_the_camera_and_the_rendering_intent_and_drops_the_local_isp():
    a = _camera_state()
    b, off = denoise_target(a), DegradeParams()
    assert all(getattr(b, k) == getattr(off, k) for k in LOCAL_ISP_FIELDS)
    assert b.shading_corner_stops == 0.0 and b.shading_corrected == 1   # the shading travels as scene (input_shading)
    kept = [k for k in a.to_dict() if k not in LOCAL_ISP_FIELDS + ("shading_corner_stops", "shading_corrected")]
    assert all(getattr(b, k) == getattr(a, k) for k in kept)


def test_a_noiseless_render_is_the_camera_chain_without_the_noise():
    tile = dead_leaves(np.random.default_rng(2), 64)
    a = _camera_state(unsharp_gain=0.0, unsharp_sigma_px=0.0, posterise_bits=0, jpeg_quality=0, shading_corrected=1)
    quiet = [degrade(tile, a, s, noiseless=True) for s in (1, 2)]
    assert np.array_equal(quiet[0], quiet[1])   # the sensor's noise is the only draw
    # a sensor whose noise is negligible renders the same chain: chromatic shift, shading, optics, mosaic, ISP
    faint = DegradeParams(**{**a.to_dict(), "full_well_e": 1e9, "read_noise_e": 0.0, "row_noise_e": 0.0,
                             "bit_depth": 16})
    assert np.abs(degrade(tile, faint, 1) - degrade(tile, faint, 1, noiseless=True)).max() < 1e-3
    assert np.abs(degrade(tile, a, 1) - quiet[0]).mean() > 1e-3   # the noisy render is not it
    assert np.abs(quiet[0] - degrade(tile, restore_target(a), 1)).mean() > 1e-3   # nor is the clean tile


@needs_data
def test_the_stream_gives_camera_restores_a_denoise_target_conditioned_as_a_real_pair():
    from harness.data.degrade import Distributions
    dists = Distributions.load()
    rng = np.random.default_rng(5)
    n = 0
    for _ in range(24):
        s = dead_leaves_sample(64, dists, rng, p_denoise=1.0)
        a = s["a"]
        if not (s["restore"] and a.camera_on):
            assert "denoise" not in s and "capture" not in s
            continue
        n += 1
        assert s["denoise"] and s["b"] == denoise_target(a)
        clean = dead_leaves(np.random.default_rng(int(s["item"].file)), 64)
        y = degrade(clean, s["b"], 0, scene_gain=input_shading(a, 64, 64), noiseless=True)
        assert np.array_equal(s["y"], y) and np.array_equal(s["x"], degrade(clean, a, s["seed_a"]))
        cap, ref = s["capture"], capture_of_params(a)
        assert cap[0] == cap[1] == 0.0 and np.array_equal(cap[2:], ref[2:])   # no blur stated, the noise law kept
        m = tile_metrics(s["y"], s["y"])   # the target's own, as real_sample conditions
        assert set(s["metrics"]) == set(m) and all(np.isclose(s["metrics"][k], m[k], equal_nan=True) for k in m)
    assert n >= 6


@needs_data
def test_a_zero_denoise_share_draws_nothing_more():
    from harness.data.degrade import Distributions
    dists = Distributions.load()
    r0, r1 = np.random.default_rng(9), np.random.default_rng(9)
    for _ in range(8):
        s0, s1 = dead_leaves_sample(64, dists, r0), dead_leaves_sample(64, dists, r1, p_denoise=0.0)
        assert "denoise" not in s1 and np.array_equal(s0["x"], s1["x"]) and np.array_equal(s0["y"], s1["y"])
    assert r0.bit_generator.state == r1.bit_generator.state


def test_pair_stream_passes_its_denoise_share_and_checks_it(tmp_path, monkeypatch):
    _tiny_sources(tmp_path, ["tiny"], size=70)
    monkeypatch.setattr(D, "split_of", lambda it: "train")
    monkeypatch.setattr(D.Distributions, "load", classmethod(lambda cls, root=None: None))
    seen = []

    def fake(*a, **k):
        seen.append(a[-1])
        return {"x": np.full((64, 64, 3), 0.5), "y": np.full((64, 64, 3), 0.5), "restore": True,
                "metrics": {m: 0.0 for m in STILL_METRICS}, "capture": np.ones(7)}
    monkeypatch.setattr(D, "make_sample", fake)
    norm = Normaliser(np.zeros(len(STILL_METRICS)), np.ones(len(STILL_METRICS)))
    next(iter(PairStream(["tiny"], 64, norm, seed=0, root=tmp_path, p_denoise=0.4)))
    assert seen == [0.4]
    with pytest.raises(ValueError, match="p_denoise is a probability"):
        PairStream(["tiny"], 64, norm, seed=0, root=tmp_path, p_denoise=-0.1)
