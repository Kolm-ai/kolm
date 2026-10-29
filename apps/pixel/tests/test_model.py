"""The recovered layer: identity at init, conditioning masks NaN, the restore target is the prepared tile,
the stream yields the right shapes, a two-step CPU training run checkpoints and resumes, and inference
round-trips a frame (whole and tiled) at the input's white level."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # the model extra is optional; CI installs only [data,dev]

from harness.data.degrade import DegradeParams, Distributions, degrade, prepare_tile
from harness.metrics import STILL_METRICS
from harness.model.data import (COND_DIM, METRIC_COND_DIM, Normaliser, decode, draw_pair, encode, render_pair, restore_target,
                                split_of)
from harness.model.net import BlurDown, CondUNet, SmoothUp, pad_to

from harness.data import data_root

DERIVED = data_root() / "derived"
needs_data = pytest.mark.skipif(not (DERIVED / "camera_presets.json").exists(), reason="derived stats not built")
SOURCES = ["hf-kadid10k-images", "hf-tid2013"]
needs_clean = pytest.mark.skipif(not all((data_root() / "raw" / s).exists() for s in SOURCES), reason="clean sources absent")


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


def test_stream_source_drops_an_item_its_decoder_rejects(tmp_path, monkeypatch):
    """A 188-megapixel featured picture past Pillow's pixel cap ended the tenth run at step 4200 from a loader
    worker: a file the decoder rejects leaves the sampler's list with its skip reason and the draw moves on."""
    from PIL import Image

    import harness.model.data as D
    from harness.model.data import TileSource

    d = tmp_path / "raw" / "tiny"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    Image.fromarray((rng.random((40, 40, 3)) * 255).astype(np.uint8)).save(d / "small.png")
    Image.fromarray((rng.random((120, 120, 3)) * 255).astype(np.uint8)).save(d / "huge.png")
    (d / "torn.png").write_bytes(b"not a picture")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 5000)  # 120 x 120 is past twice the cap: DecompressionBombError
    monkeypatch.setattr(D, "split_of", lambda it: "train")
    src = TileSource(["tiny"], "train", 16, root=tmp_path, tiles_per_image=4)
    assert sorted(it.file for it in src.items) == ["huge.png", "small.png", "torn.png"]
    with pytest.warns(RuntimeWarning, match="dropped from the sampler"):
        drawn = {src.random_tile(rng)[1].file for _ in range(40)}
    assert drawn == {"small.png"} and [it.file for it in src.items] == ["small.png"]
    reasons = {it.file: r for it, r in src.dropped.items()}
    assert reasons["huge.png"].startswith("decode_error:DecompressionBombError: Image size (14400 pixels)")
    assert reasons["torn.png"].startswith("decode_error:") and src._cur is not None
    with pytest.warns(RuntimeWarning):
        for it in list(src.items):
            src.drop(it, ValueError("gone"))
    with pytest.raises(ValueError, match="every item failed to decode"):
        src.random_tile(rng)


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


def test_net_scales_by_level_depth_and_bottleneck_and_rebuilds_from_its_config():
    """Per-level block counts and a bottleneck count shape the net; its config rebuilds it, and a config written
    before the counts existed (one count, no mid_blocks) builds the net it always did."""
    flat = CondUNet(base=8, mults=(1, 2, 4), blocks=1)
    old = {k: v for k, v in flat.config.items() if k != "mid_blocks"}
    assert CondUNet(**old).n_params() == flat.n_params()
    deep = CondUNet(base=8, mults=(1, 2, 4), blocks=(1, 1, 2), mid_blocks=3)
    assert [len(lv.blocks) for lv in deep.down] == [1, 1, 2, 1, 1] and len(deep.mid.blocks) == 3
    assert CondUNet(**deep.config).n_params() == deep.n_params() > flat.n_params()
    with pytest.raises(ValueError, match="one per level"):
        CondUNet(base=8, mults=(1, 2, 4), blocks=(1, 2))


def test_smooth_resampling_leaves_no_grid_and_rebuilds_from_its_config():
    """SmoothUp takes a constant to a constant where a stride-2 transposed conv paints its four 2 px phases, BlurDown
    keeps a constant's interior flat, the smooth net rebuilds and reloads from its config, and a config written
    before `resample` existed builds the learned net it always did."""
    torch.manual_seed(0)
    c = torch.ones(1, 4, 8, 8)
    up = SmoothUp(4, 4)(c)
    assert up.shape == (1, 4, 16, 16) and float(up.std(dim=(2, 3)).max()) < 1e-6
    assert float(torch.nn.ConvTranspose2d(4, 4, 2, stride=2)(c).std(dim=(2, 3)).max()) > 1e-3
    down = BlurDown(4, 6)(c)
    assert down.shape == (1, 6, 4, 4) and float(down[..., 1:-1, 1:-1].std(dim=(2, 3)).max()) < 1e-6
    smooth = CondUNet(base=8, mults=(1, 2, 4), blocks=1, resample="smooth")
    again = CondUNet(**smooth.config)
    again.load_state_dict(smooth.state_dict())          # the blur kernel is a buffer outside the checkpoint
    assert isinstance(again.up[0], SmoothUp) and isinstance(again.pool[0], BlurDown)
    x, cond = torch.rand(2, 3, 32, 32), torch.zeros(2, COND_DIM)
    assert smooth(x, cond).shape == x.shape
    old = {k: v for k, v in smooth.config.items() if k != "resample"}
    assert isinstance(CondUNet(**old).up[0], torch.nn.ConvTranspose2d)
    with pytest.raises(ValueError, match="resample"):
        CondUNet(base=8, mults=(1, 2), resample="pixel-shuffle")


def test_gradient_checkpointing_changes_no_output_or_gradient():
    torch.manual_seed(0)
    net = CondUNet(base=8, mults=(1, 2), blocks=2)
    with torch.no_grad():
        for p in net.parameters():          # off the zero init, so every block carries a gradient
            p.add_(0.05 * torch.randn_like(p))
    x, c = torch.rand(2, 3, 16, 16), torch.randn(2, COND_DIM)
    runs = []
    for on in (False, True):
        net.set_grad_ckpt(on).zero_grad()
        y = net(x, c)
        y.square().mean().backward()
        runs.append((y.detach(), [p.grad.clone() for p in net.parameters()]))
    assert torch.allclose(runs[0][0], runs[1][0])
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(runs[0][1], runs[1][1]))


def test_inference_pads_to_the_net_level_multiple():
    """A five-level net halves the side four times: apply pads to 16, not 8, and strips the padding again."""
    from harness.model.apply import Bundle, restore_frames
    net = CondUNet(base=8, mults=(1, 2, 2, 2, 2), blocks=1).eval()
    b = Bundle(net, Normaliser(np.zeros(12), np.ones(12)), np.zeros(METRIC_COND_DIM, dtype=np.float32), 0,
               torch.device("cpu"))
    frames = np.random.default_rng(2).random((1, 40, 56, 3))     # multiples of 8, not of 16
    assert np.allclose(restore_frames(b, frames), frames, atol=1e-5)


def test_train_rejects_a_tile_the_levels_cannot_halve(tmp_path):
    from harness.model.train import train
    with pytest.raises(ValueError, match="multiple of 16"):
        train(str(tmp_path / "m"), SOURCES, steps=1, tile=24, workers=0, mults=(1, 2, 4, 8, 8), device="cpu")


@needs_data
@needs_clean
def test_train_records_the_shape_and_apply_rebuilds_it(tmp_path):
    from harness.model.apply import load_model
    from harness.model.train import train
    out = tmp_path / "m"
    train(str(out), SOURCES, steps=1, batch=2, tile=32, workers=0, base=8, mults=(1, 2, 2, 2, 2), blocks=(1, 1, 1, 2, 2),
          mid_blocks=3, grad_ckpt=True, val_n=4, calib_n=8, ckpt_every=1, device="cpu", log=lambda *a: None)
    ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["blocks"] == (1, 1, 1, 2, 2) and ck["config"]["mid_blocks"] == 3
    assert ck["train"]["mults"] == [1, 2, 2, 2, 2] and ck["train"]["blocks"] == [1, 1, 1, 2, 2]
    b = load_model(out / "last.pt", "cpu")
    assert b.model.n_levels == 5 and len(b.model.mid.blocks) == 3


@needs_data
@needs_clean
def test_train_records_the_sixteenth_run_terms_and_apply_rebuilds_a_smooth_net(tmp_path):
    from harness.model.apply import load_model
    from harness.model.train import train
    out = tmp_path / "m"
    train(str(out), SOURCES, steps=1, batch=2, tile=32, workers=0, base=8, val_n=4, calib_n=8, ckpt_every=1,
          overshoot_weight=1.5, pyramid_weight=0.5, resample="smooth", device="cpu", log=lambda *a: None)
    ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["resample"] == "smooth" and ck["train"]["resample"] == "smooth"
    assert ck["train"]["overshoot_weight"] == 1.5 and ck["train"]["pyramid_weight"] == 0.5
    assert isinstance(load_model(out / "last.pt", "cpu").model.pool[0], BlurDown)
    with pytest.raises(ValueError, match="overshoot and pyramid"):
        train(str(tmp_path / "n"), SOURCES, steps=1, workers=0, pyramid_weight=-1.0, device="cpu")


@needs_data
@needs_clean
def test_train_records_the_val_seed_and_defaults_it_to_the_seed(tmp_path):
    from harness.model.train import train
    kw = dict(steps=1, batch=2, tile=32, workers=0, base=8, val_n=4, calib_n=8, ckpt_every=1, device="cpu",
              log=lambda *a: None)
    train(str(tmp_path / "a"), SOURCES, seed=2, val_seed=0, **kw)
    train(str(tmp_path / "b"), SOURCES, seed=2, **kw)
    a = torch.load(tmp_path / "a" / "last.pt", map_location="cpu", weights_only=False)["train"]
    b = torch.load(tmp_path / "b" / "last.pt", map_location="cpu", weights_only=False)["train"]
    assert (a["seed"], a["val_seed"]) == (2, 0) and (b["seed"], b["val_seed"]) == (2, 2)


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


def test_range_loss_is_zero_at_the_target_and_charges_a_halo():
    import pytest
    import torch
    from harness.model.train import pair_loss, range_excess
    y = torch.zeros(1, 3, 8, 16)
    y[..., 8:] = 1.0                      # a vertical step
    x = y.clone()
    assert float(range_excess(y, y).sum()) == 0.0
    out = y.clone()
    out[..., 8] = 1.2                     # a one-pixel bright halo on the bright side
    assert float(range_excess(out, y).max()) == pytest.approx(0.2)
    assert float(range_excess(out, y)[..., :, 7].max()) == 0.0   # the dark side is inside its range
    assert float(pair_loss(out, y, x)) > float(pair_loss(y, y, x)) == 0.0
    base = float(pair_loss(out, y, x, range_weight=0.0))
    one, two = float(pair_loss(out, y, x, range_weight=1.0)), float(pair_loss(out, y, x, range_weight=2.0))
    assert base < one < two and two - one == pytest.approx(one - base)     # linear in the weight, zero turns it off
    assert float(range_excess(out, y, 5).sum()) >= float(range_excess(out, y, 3).sum()) > 0.0


def test_flat_loss_charges_texture_only_where_the_target_is_flat():
    import pytest
    import torch
    from harness.model.train import flat_excess, local_std, pair_loss
    g = torch.Generator().manual_seed(0)
    d = 0.001 / 15
    y = torch.full((1, 3, 16, 32), 6.0)                       # at the encoded domain's scale ...
    y[..., :16] += d * torch.arange(16.0)                     # ... a gentle ramp (std 2 d over 7 px) ...
    y[..., 16:] += torch.rand(1, 3, 16, 16, generator=g)       # ... beside a textured half
    x = y.clone()
    s = local_std(y, 7)[..., :, 3:13]
    assert torch.allclose(s, torch.full_like(s, 2 * d), rtol=1e-3, atol=1e-7)   # no float32 cancellation noise
    assert float(flat_excess(y, y).sum()) == 0.0              # the target itself is never charged
    out = y + 0.02 * torch.randn(y.shape, generator=g)        # texture everywhere
    e = flat_excess(out, y)
    assert float(e[..., :, :8].min()) > 0.0                   # charged on the flat half ...
    assert float(e[..., :, 24:].max()) == 0.0                 # ... not on the textured one
    smooth = y.clone()
    smooth[..., 16:] = y[..., 16:].mean()                     # smoother than the target is not excess
    assert float(flat_excess(smooth, y).sum()) == 0.0
    assert float(flat_excess(out, y, tau=0.0).sum()) == 0.0   # tau 0 calls nothing flat
    base = float(pair_loss(out, y, x))                        # off by default
    assert base == float(pair_loss(out, y, x, flat_weight=0.0))
    one, two = float(pair_loss(out, y, x, flat_weight=1.0)), float(pair_loss(out, y, x, flat_weight=2.0))
    assert base < one < two and two - one == pytest.approx(one - base, rel=1e-4)   # linear in the weight


def test_texture_loss_charges_smoothing_only_where_the_target_has_texture():
    import pytest
    import torch
    from harness.model.train import flat_excess, pair_loss, std_mismatch, texture_deficit
    g = torch.Generator().manual_seed(1)
    y = torch.full((1, 3, 16, 32), 6.0)
    y[..., 16:] += torch.rand(1, 3, 16, 16, generator=g)       # a flat half beside a textured half
    x = y.clone()
    assert float(texture_deficit(y, y).sum()) == 0.0           # the target itself is never charged
    smooth = y.clone()
    smooth[..., 16:] = y[..., 16:].mean()                      # the textured half smoothed away
    d = texture_deficit(smooth, y)
    assert float(d[..., :, 24:].min()) > 0.1                   # charged where the target has texture ...
    assert float(d[..., :, :8].max()) == 0.0                   # ... not on the flat half
    assert float(flat_excess(smooth, y).sum()) == 0.0          # smoothing is never flat excess
    boost = y.clone()
    boost[..., 16:] = 6.0 + 2.0 * (y[..., 16:] - 6.0)          # more texture than the target is not a deficit
    assert float(texture_deficit(boost, y).sum()) == 0.0
    noisy = y + 0.02 * torch.randn(y.shape, generator=g)       # texture everywhere
    e, t = std_mismatch(noisy, y)
    assert torch.equal(e, flat_excess(noisy, y)) and torch.equal(t, texture_deficit(noisy, y))
    assert float(e[..., :, :8].min()) > 0.0 and float((e * t).sum()) == 0.0   # the halves never share a pixel
    base = float(pair_loss(smooth, y, x))                      # off by default
    assert base == float(pair_loss(smooth, y, x, texture_weight=0.0))
    one, two = float(pair_loss(smooth, y, x, texture_weight=1.0)), float(pair_loss(smooth, y, x, texture_weight=2.0))
    assert base < one < two and two - one == pytest.approx(one - base, rel=1e-4)   # linear in the weight


def test_eval_raw_scores_a_checkpoint_on_held_out_tiles(tmp_path):
    """heldout.evaluate: a tiny checkpoint on a two-file held-out source, known and blind, from one cached set."""
    import json
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from harness.model.heldout import evaluate, to_markdown
    src = tmp_path / "raw" / "test-raws"
    src.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for i in range(2):
        img = (np.clip(gaussian_filter(rng.random((80, 96, 3)), 2) * 2.5, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(src / f"r{i}.png")
    torch.manual_seed(0)
    net = CondUNet(base=8, mults=(1, 2), blocks=1)
    norm = Normaliser(np.zeros(len(STILL_METRICS)), np.ones(len(STILL_METRICS)))
    ck = tmp_path / "tiny.pt"
    torch.save({"ema": net.state_dict(), "config": net.config, "normaliser": norm.state(),
                "cond_restore": [0.0] * METRIC_COND_DIM, "step": 7}, ck)
    out = tmp_path / "rep"
    rep = evaluate(ck, blind=False, source="test-raws", n=2, tile=32, seed=0, root=tmp_path, device="cpu", out=out,
                   save_tiles=2, log=None)
    assert rep["n"] == 2 and rep["step"] == 7 and rep["scale"] == 1 and not rep["blind"] and rep["skipped"] == {}
    assert set(rep["metrics"]) == set(STILL_METRICS) and len(rep["tiles"]) == 2 and len(rep["records"]) == 2
    assert np.load(out / "tiles" / "model.npy").shape == (2, 32, 32, 3)
    assert (out / "tiles" / "input.npy").exists() and (out / "tiles" / "target.npy").exists()
    json.dumps(rep)
    assert "step 7" in to_markdown(rep) and "known" in to_markdown(rep)
    for m, s in rep["metrics"].items():  # the net is the identity at init: the output is the input, nothing moves
        assert s["beats"] == 0 and s["loses"] == 0, (m, s)
    blind = evaluate(ck, blind=True, source="test-raws", n=2, tile=32, seed=0, root=tmp_path, device="cpu", log=None)
    assert blind["blind"] and blind["records"] == rep["records"] and "withheld" in to_markdown(blind)


def test_the_cosine_anneals_over_whichever_budget_runs_out_first():
    from harness.model.train import WARMUP, lr_at
    lr, steps = 2e-4, 30000
    assert lr_at(0, steps, lr, spent=0.9) == pytest.approx(lr / WARMUP)          # the warmup stays step-based
    assert lr_at(WARMUP, steps, lr) == pytest.approx(lr)
    assert lr_at(steps, steps, lr) == pytest.approx(0.05 * lr)
    mid = lr_at(WARMUP + 100, steps, lr, spent=0.5)                               # the clock ahead of the steps
    assert mid == pytest.approx(lr * (0.05 + 0.95 * 0.5))
    assert lr_at(WARMUP + 100, steps, lr, spent=1.0) == pytest.approx(0.05 * lr)  # a run its --minutes stop ends annealed
    assert lr_at(steps - 10, steps, lr, spent=0.1) < lr_at(steps // 2, steps, lr, spent=0.1)   # the steps ahead


def test_the_trainer_holds_its_allocator_to_the_card(monkeypatch):
    # Past the card, the Windows driver spills into shared system memory instead of raising (the burst run on v17,
    # 10-30, ran at 1 img/s). The trainers cap the allocator at what is free at start, their own share counted back
    # in, less a margin, so a run that does not fit fails at once and its retry comes next.
    import harness.model.train as T
    gb = 2**30
    set_to = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (20 * gb, 32 * gb))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda dev=None: 2 * gb)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f, dev=None: set_to.append(f))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(T, "card_used_bytes", lambda idx: None)
    frac = T.hold_to_card(torch.device("cuda"), log=lambda *a: None)
    assert frac == pytest.approx((20 + 2 - T.SPILL_MARGIN_GB) / 32) and set_to == [frac]
    # under WDDM mem_get_info sees this process alone; the card's own count (nvidia-smi) wins where it is lower
    monkeypatch.setattr(T, "card_used_bytes", lambda idx: 15 * gb)
    frac = T.hold_to_card(torch.device("cuda"), log=lambda *a: None)
    assert frac == pytest.approx((32 - 15 + 2 - T.SPILL_MARGIN_GB) / 32) and set_to[-1] == frac
    assert T.hold_to_card(torch.device("cpu"), log=lambda *a: None) is None and len(set_to) == 2


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="needs a GPU")
def test_the_card_count_reads_on_a_gpu():
    import harness.model.train as T
    idx = torch.cuda.current_device()
    used = T.card_used_bytes(idx)
    assert used is not None and 0 < used < torch.cuda.mem_get_info(idx)[1]


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="needs a GPU")
def test_an_allocation_past_the_cap_raises_instead_of_spilling():
    # what hold_to_card relies on, at a 0.5 GB cap so a failure here costs 1 GB beside a running job
    dev = torch.cuda.current_device()
    total = torch.cuda.mem_get_info(dev)[1]
    torch.cuda.set_per_process_memory_fraction(0.5 * 2**30 / total, dev)
    try:
        with pytest.raises(torch.OutOfMemoryError):
            torch.empty(2**30, dtype=torch.uint8, device=dev)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, dev)
