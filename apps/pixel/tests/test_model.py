"""The recovered layer: identity at init, conditioning masks NaN, the restore target is the prepared tile,
the stream yields the right shapes, a two-step CPU training run checkpoints and resumes, and inference
round-trips a frame (whole and tiled) at the input's white level."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from harness.data.degrade import DegradeParams, Distributions, degrade, prepare_tile
from harness.metrics import STILL_METRICS
from harness.model.data import (COND_DIM, METRIC_COND_DIM, Normaliser, decode, draw_pair, encode, render_pair, restore_target,
                                split_of)
from harness.model.net import CondUNet, pad_to

from harness.data import data_root

DERIVED = data_root() / "derived"
needs_data = pytest.mark.skipif(not (DERIVED / "camera_presets.json").exists(), reason="derived stats not built")
SOURCES = ["hf-kadid10k-images", "hf-tid2013"]
needs_clean = pytest.mark.skipif(not all((Path("data/raw") / s).exists() for s in SOURCES), reason="clean sources absent")


def _tile(rng, n=64):
    y, x = np.mgrid[0:n, 0:n] / n
    base = np.stack([0.3 + 0.4 * x, 0.3 + 0.3 * y, 0.5 - 0.2 * x * y], -1)
    base[n // 4:n // 2, n // 4:n // 2] *= 0.3
    return np.clip(base + 0.02 * rng.standard_normal(base.shape), 0.02, 0.95)


def test_encode_decode_round_trip():
    x = np.linspace(0, 3, 50).reshape(5, 10, 1).repeat(3, -1)
    assert np.allclose(decode(encode(x)), x, atol=1e-5)
    assert encode(np.array([-1.0])).item() == 0.0
    assert encode(np.array([1.0])).item() == pytest.approx(np.log1p(4096.0))  # the clip encodes to 8.3
    e = encode(np.array([0.001, 0.002, 0.99, 1.0]))
    assert e[1] - e[0] > 10 * (e[3] - e[2])                                 # a 2x shadow error outweighs 1 % at the clip
    assert encode(np.array([1 / 4096])).item() == pytest.approx(np.log(2.0))  # the floor is one 12-bit LSB


def test_net_identity_at_init_and_padding():
    torch.manual_seed(0)
    net = CondUNet(base=8, mults=(1, 2), blocks=1)
    x = torch.rand(2, 3, 40, 56)
    c = torch.randn(2, COND_DIM)
    xp, (ph, pw) = pad_to(x, 8)
    assert xp.shape[-2:] == (40, 56) and (ph, pw) == (0, 0)
    xp, (ph, pw) = pad_to(torch.rand(1, 3, 37, 50), 8)
    assert xp.shape[-2:] == (40, 56) and (ph, pw) == (3, 6)
    with torch.no_grad():
        y = net(x, c)
    assert y.shape == x.shape
    assert torch.allclose(y, x)


def test_normaliser_masks_nan_and_clips():
    vecs = np.array([[1.0, 2.0, 3.0, np.nan] + [0.0] * 8, [2.0, 2.0, 1.0, 5.0] + [0.0] * 8,
                     [3.0, 2.0, 2.0, np.nan] + [0.0] * 8])
    n = Normaliser.fit(vecs)
    out = n({m: v for m, v in zip(STILL_METRICS, [1e9, np.nan, 2.0, np.nan] + [0.0] * 8)})
    assert out.shape == (METRIC_COND_DIM,) and out.dtype == np.float32
    assert out[0] == 6.0 and out[1] == 0.0 and out[3] == 0.0
    assert list(out[12:16]) == [1.0, 0.0, 1.0, 0.0]
    assert np.isfinite(out).all()
    assert Normaliser.from_state(n.state()).state() == n.state()


@needs_data
def test_restore_target_keeps_the_input_lens_shading():
    """Shading is scene, not defect: an uncorrected input's target carries the same shading map, a corrected
    input's target is flat, and an edit target never carries B's own shading draw."""
    from harness.data.degrade import tile_shading
    dists = Distributions.load()
    rng = np.random.default_rng(6)
    tile = _tile(rng)
    cam = dict(camera_on=1, preset="x", pixel_pitch_um=1.0, half_diag_mm=4.0, focal_mm=4.0, f_number=2.0, iso=100.0,
               analog_gain=1, full_well_e=5000.0, white_fill=1.0, read_noise_e=1.0, bit_depth=12, bayer=0,
               field_fraction=0.8, residual_blur_um=0.0, shading_corner_stops=1.0)
    a = DegradeParams(expose=1, exposure_stops=0.0, scene_peak_over_median=5.0, shading_corrected=0, **cam)
    b = restore_target(a)
    xa, xb, m = render_pair(tile, a, b, 1, 2, False)
    shade = tile_shading(tile.shape[0], tile.shape[1], a)
    assert shade.min() < 0.8 and np.allclose(xb, prepare_tile(tile, b, False) * shade[..., None])
    assert m["shading_rmse"] > 1e-3          # the label reports the kept shading (the metric is a normalised low-frequency RMS)
    a_c = DegradeParams(**{**a.to_dict(), "shading_corrected": 1})
    _, xb_c, m_c = render_pair(tile, a_c, restore_target(a_c), 1, 2, False)
    assert np.allclose(xb_c, prepare_tile(tile, restore_target(a_c), False)) and m_c["shading_rmse"] < 1e-9
    b_edit = DegradeParams(**{**a.to_dict(), "shading_corrected": 0, "iso": 400.0})
    _, xb_e, _ = render_pair(tile, a_c, b_edit, 1, 2, False)
    b_edit_corrected = DegradeParams(**{**b_edit.to_dict(), "shading_corrected": 1})
    assert np.allclose(xb_e, degrade(tile, b_edit_corrected, 2, 1, False))


@needs_data
def test_shading_target_sits_ahead_of_the_tone_affine():
    """The input had its lens shading before the ISP, so the target gets it there too: with the tone affine on,
    shade * isp(tile) is not isp(shade * tile). Runs four to seven multiplied the shade in after the ISP and a
    third of their restore targets sat up to 2.5x darker than the input at mid-tones."""
    from harness.colorspace import luminance
    from harness.data.degrade import _isp, tile_shading
    rng = np.random.default_rng(7)
    tile = _tile(rng)
    cam = dict(camera_on=1, preset="x", pixel_pitch_um=1.0, half_diag_mm=4.0, focal_mm=4.0, f_number=2.0, iso=100.0,
               analog_gain=1, full_well_e=5000.0, white_fill=1.0, read_noise_e=1.0, bit_depth=12, bayer=0,
               field_fraction=0.8, residual_blur_um=0.0, shading_corner_stops=1.0)
    a = DegradeParams(expose=1, exposure_stops=0.0, scene_peak_over_median=5.0, shading_corrected=0,
                      contrast_gain=0.5, mean_shift=0.1, **cam)
    b = restore_target(a)
    xa, xb, _ = render_pair(tile, a, b, 1, 2, False)
    shade = tile_shading(tile.shape[0], tile.shape[1], a)
    prepared = prepare_tile(tile, b, False)
    assert np.allclose(xb, _isp(prepared * shade[..., None], b))
    old = _isp(prepared, b) * shade[..., None]
    lum_in, lum_new, lum_old = luminance(xa), luminance(xb), luminance(old)
    mid = (lum_in > 0.1) & (lum_in < 0.8)
    assert mid.sum() > 1000
    assert abs(np.median(lum_new[mid] / lum_in[mid]) - 1.0) < 0.05
    assert np.median(lum_old[mid] / lum_in[mid]) < 0.9


@needs_data
def test_identity_pairs_render_the_clean_tile_as_both_input_and_target():
    """A clean input must come out unchanged: a share of the pairs is (clean, clean) with the restore conditioning."""
    from harness.model.data import P_IDENTITY
    dists = Distributions.load()
    rng = np.random.default_rng(4)
    tile = _tile(rng)
    pairs = [draw_pair(rng, dists) for _ in range(400)]
    ident = [(a, b) for a, b, restore in pairs if restore and a == b]
    assert 0.5 * P_IDENTITY < len(ident) / len(pairs) < 2.0 * P_IDENTITY, len(ident)
    a, b = ident[0]
    assert a == restore_target(a) and not a.camera_on
    xa, xb, m = render_pair(tile, a, b, 1, 2, False)
    assert np.allclose(xa, xb) and np.allclose(xb, degrade(tile, restore_target(a), 2, 1, False))
    if a.contrast_gain == 1.0 and a.mean_shift == 0.0 and a.saturation == 1.0 and a.cast_r == 1.0 and a.cast_b == 1.0:
        assert np.allclose(xb, prepare_tile(tile, b, False))
    assert m["delta_e_mean"] < 1e-6 and m["mtf50_cpp"] > 0


@needs_data
def test_restore_pair_target_is_the_prepared_tile():
    dists = Distributions.load()
    rng = np.random.default_rng(3)
    tile = _tile(rng)
    for _ in range(60):
        a, b, restore = draw_pair(rng, dists)
        if restore and (not a.camera_on or a.shading_corrected or a.shading_corner_stops <= 0):
            break
    assert restore and b == restore_target(a)
    xa, xb, m = render_pair(tile, a, b, 1, 2, False)
    assert xa.shape == xb.shape == tile.shape
    assert np.allclose(xb, degrade(tile, restore_target(a), 2, 1, False))
    assert set(m) == set(STILL_METRICS)
    if a.contrast_gain == 1.0 and a.mean_shift == 0.0 and a.saturation == 1.0 and a.cast_r == 1.0 and a.cast_b == 1.0:
        assert np.allclose(xb, prepare_tile(tile, b, False))


@needs_data
def test_targets_keep_the_input_global_tone_and_colour():
    """Exposure, the sRGB tone affine, saturation and the colour cast are not identifiable from a tile and are
    not in the metric vector, so every target keeps the input's values and a restore label reads as clean."""
    from harness.model.data import GLOBAL_FIELDS
    dists = Distributions.load()
    rng = np.random.default_rng(6)
    tile = _tile(rng)
    pairs = [draw_pair(rng, dists) for _ in range(400)]
    for a, b, restore in pairs:
        for k in GLOBAL_FIELDS:
            assert getattr(b, k) == getattr(a, k), (k, restore)
    cast = next(a for a, b, r in pairs if r and a != b and (a.cast_r != 1.0 or a.contrast_gain != 1.0))
    a = DegradeParams(**{**cast.to_dict(), "camera_on": 0, "shading_corner_stops": 0.0})
    b = restore_target(a)
    xa, xb, m = render_pair(tile, a, b, 1, 2, False)
    assert m["delta_e_mean"] < 1e-6 and m["shading_rmse"] < 1e-9
    assert not np.allclose(xb, prepare_tile(tile, b, False))


def test_split_is_deterministic_and_mostly_train():
    from harness.data.label import Item
    items = [Item(source="s", file=f"f{i}", row=0) for i in range(400)]
    splits = [split_of(it) for it in items]
    assert splits == [split_of(it) for it in items]
    val = splits.count("val")
    assert 8 <= val <= 50


@needs_data
@needs_clean
def test_train_two_steps_resume_and_apply(tmp_path):
    from harness.model.apply import load_model, restore_frames
    from harness.model.train import train
    out = tmp_path / "m"
    log = []
    v = train(str(out), SOURCES, steps=2, batch=2, tile=32, workers=0, base=8, val_n=4, calib_n=8, ckpt_every=1,
              device="cpu", log=log.append)
    assert (out / "last.pt").exists() and (out / "best.pt").exists()
    assert np.isfinite(v["val_loss"]) and v["step"] == 2
    v2 = train(str(out), SOURCES, steps=3, batch=2, tile=32, workers=0, base=8, val_n=4, calib_n=8, ckpt_every=1,
               device="cpu", log=log.append)
    assert v2["step"] == 3 and any("resuming from step 2" in s for s in log)
    b = load_model(out / "last.pt", "cpu")
    assert b.step == 3 and b.cond_restore.shape == (METRIC_COND_DIM,)
    frames = np.random.default_rng(0).random((2, 20, 28, 3)) * 2.5
    y = restore_frames(b, frames, white_level=2.5)
    assert y.shape == frames.shape and np.isfinite(y).all() and (y >= 0).all()
    assert np.abs(y - frames).max() < 0.5  # two steps from identity stay close to identity


@needs_clean
def test_stream_source_stays_on_one_image_for_several_tiles():
    from harness.model.data import TileSource
    src = TileSource(SOURCES, "train", 32, tiles_per_image=3)
    rng = np.random.default_rng(0)
    items = [src.random_tile(rng)[1] for _ in range(6)]
    assert items[0] == items[1] == items[2] and items[3] == items[4] == items[5]
    one = TileSource(SOURCES, "train", 32)
    rng = np.random.default_rng(0)
    assert len({one.random_tile(rng)[1] for _ in range(12)}) > 1


def test_tiled_inference_matches_whole_frame_on_identity(monkeypatch):
    from harness.model import apply as A
    from harness.model.apply import Bundle, restore_frames
    torch.manual_seed(0)
    net = CondUNet(base=8, mults=(1, 2), blocks=1).eval()
    norm = Normaliser(np.zeros(12), np.ones(12))
    b = Bundle(net, norm, np.zeros(METRIC_COND_DIM, dtype=np.float32), 0, torch.device("cpu"))
    frames = np.random.default_rng(1).random((1, 70, 90, 3))
    whole = restore_frames(b, frames)
    monkeypatch.setattr(A, "MAX_SIDE", 48)
    monkeypatch.setattr(A, "TILE", 48)
    monkeypatch.setattr(A, "OVERLAP", 16)
    tiled = restore_frames(b, frames)
    assert np.allclose(whole, frames, atol=1e-5)
    assert np.allclose(tiled, frames, atol=1e-5)


@needs_data
def test_edit_pairs_keep_the_input_exposure_and_scene_range():
    """The target's brightness is not in the metric vector, so no pair may change it: every edit target
    carries the input's expose flag, exposure offset and scene range; only the other fields differ."""
    dists = Distributions.load()
    rng = np.random.default_rng(5)
    pairs = [(a, b) for a, b, restore in (draw_pair(rng, dists) for _ in range(400)) if not restore]
    assert len(pairs) > 40, len(pairs)
    for a, b in pairs:
        assert b.expose == a.expose and b.exposure_stops == a.exposure_stops
        assert b.scene_peak_over_median == a.scene_peak_over_median
    assert any(b.iso != a.iso or b.residual_blur_um != a.residual_blur_um or b.jpeg_quality != a.jpeg_quality for a, b in pairs)


def test_capture_descriptor_is_physical_and_optional():
    """The input's known capture (Airy radius, residual blur, shot and read noise at the clip, one LSB) rides
    with the metric conditioning; a sampler camera state and the demo camera give the same descriptor for the
    same physics; an unknown capture is all zero with known = 0."""
    from harness.camera import PRESETS
    from harness.model.apply import capture_of_constraints
    from harness.model.data import CAPTURE_DIM, UNKNOWN_CAPTURE, capture_descriptor, capture_of_params
    c = PRESETS["best_case_earth"]
    d = capture_of_constraints(c)
    assert d.shape == (CAPTURE_DIM,) and d[-1] == 1.0 and d[5] == 1.0
    assert np.isclose(d[0] * 2.0, 1.22 * 0.55 * c.f_number / c.pixel_pitch_um)
    assert np.isclose(d[2] - 3.0, np.log10(1.0 / np.sqrt(c.full_well_e)))
    p = DegradeParams(camera_on=1, pixel_pitch_um=c.pixel_pitch_um, focal_mm=c.focal_mm, f_number=c.f_number,
                      full_well_e=c.full_well_e, white_fill=1.0, read_noise_e=c.read_noise_e, bit_depth=c.bit_depth,
                      bayer=1, residual_blur_um=c.residual_blur_um)
    assert np.allclose(capture_of_params(p), d)
    half = DegradeParams(**{**p.to_dict(), "white_fill": 0.5})
    assert capture_of_params(half)[2] > d[2]  # half the electrons at the clip: more shot noise
    clean = capture_of_params(DegradeParams())
    assert clean[-1] == 1.0 and clean[0] == 0.0 and clean[2] == -1.0 and clean[3] == -1.0 and clean[4] == -2.0
    assert UNKNOWN_CAPTURE.shape == (CAPTURE_DIM,) and not UNKNOWN_CAPTURE.any()
    assert np.isfinite(capture_descriptor(0.3, 0.0, 1e9, 0.0, 16, False)).all()


@needs_data
def test_restore_inputs_mostly_come_from_the_capture_profile():
    """Restore pairs draw the sampler's raw capture profile with P_RESTORE_CAPTURE: camera on, no unsharp halo,
    no posterise. The rest keep the IQA profile, so halos and banding are still restored when present."""
    from harness.model.data import P_RESTORE_CAPTURE
    dists = Distributions.load()
    rng = np.random.default_rng(8)
    pairs = [draw_pair(rng, dists) for _ in range(600)]
    restore = [a for a, b, r in pairs if r and a != b]
    assert len(restore) > 200, len(restore)
    raw = [a for a in restore if a.camera_on and a.unsharp_gain == 0.0 and a.posterise_bits == 0]
    frac = len(raw) / len(restore)
    assert P_RESTORE_CAPTURE - 0.1 < frac < P_RESTORE_CAPTURE + 0.2, frac   # 0.7 + 0.3 x (0.8 x 0.25) expected
    assert any(a.unsharp_gain > 0 for a in restore)                         # the IQA share is still there
    edits = [(a, b) for a, b, r in pairs if not r]
    assert any(a.unsharp_gain > 0 or a.posterise_bits > 0 for a, b in edits)
