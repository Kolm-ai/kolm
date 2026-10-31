"""The product path: the TIFF writer round-trips 16-bit and float frames (and Pillow reads the 16-bit one), the blind
noise law finds a Poisson-Gaussian law's electrons from the frame alone, and the photo command takes a raw (a stub
decode) and an sRGB image through a checkpoint to a TIFF, a JPEG preview and a record of what it did."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harness.data.blindnoise import blind_law
from harness.tiff import read_tiff, write_tiff


def _grey_ramp(h=512, w=512):
    g = 0.02 + 0.8 * (np.mgrid[0:h, 0:w][1] / w)
    return np.stack([g, g, g], -1)


def _noisy(clean, electrons, read_e, seed=0):
    rng = np.random.default_rng(seed)
    return rng.poisson(clean * electrons) / electrons + rng.normal(0.0, read_e / electrons, clean.shape)


def test_tiff_round_trips_16_bit_and_float_and_pillow_reads_it(tmp_path):
    from PIL import Image
    a = (np.random.default_rng(0).random((130, 77, 3)) * 65535).astype(np.uint16)   # 130 rows: three strips
    write_tiff(tmp_path / "a.tif", a, "test")
    assert (read_tiff(tmp_path / "a.tif") == a).all()
    assert (np.asarray(Image.open(tmp_path / "a.tif")) == (a >> 8)).all()
    f = (np.random.default_rng(1).random((65, 40, 3)) * 3).astype(np.float32)
    write_tiff(tmp_path / "f.tif", f)
    assert (read_tiff(tmp_path / "f.tif") == f).all()
    with pytest.raises(ValueError):
        write_tiff(tmp_path / "g.tif", a[..., :2])


# unscaled electrons over the true ones on iid noise: the "mad" residual keeps ~94 % of the noise's variance; the
# "blocks" reading (sigma 1, a low quantile of 8 x 8 cells) reads ~64 % of it. Either is a scale, the same at every
# exposure, which VAR_SCALE (calibrated on real demosaiced pairs, not on iid noise) replaces.
IID_RATIO = {"mad": (0.95, 1.2), "blocks": (1.45, 1.7)}


@pytest.mark.parametrize("method", ["blocks", "mad"])
@pytest.mark.parametrize("electrons", [500.0, 4000.0])
def test_blind_law_reads_the_electrons_of_a_poisson_gaussian_frame(electrons, method):
    law = blind_law(_noisy(_grey_ramp(), electrons, 3.0), var_scale=1.0, method=method)
    assert law["bins"] >= 3 and law["method"] == method
    lo, hi = IID_RATIO[method]
    assert lo < law["electrons_at_clip"] / electrons < hi
    half = blind_law(_noisy(_grey_ramp(), electrons, 3.0), var_scale=2.0, method=method)
    assert half["electrons_at_clip"] == pytest.approx(law["electrons_at_clip"] / 2.0)
    assert half["blind_a"] == pytest.approx(law["blind_a"])


def test_blind_law_has_no_law_without_flat_unclipped_pixels():
    law = blind_law(np.ones((64, 64, 3)), var_scale=1.0)
    assert law["electrons_at_clip"] is None and law["a"] is None


def test_exposure_gain_auto_never_darkens():
    from harness.photo import exposure_gain
    x = _grey_ramp(64, 64) * 0.25
    assert exposure_gain(x, None) == 1.0 and exposure_gain(x, "1") == 2.0 and exposure_gain(x, "-1") == 0.5
    assert exposure_gain(x, "auto") > 1.0 and exposure_gain(x * 8, "auto") == 1.0


torch = pytest.importorskip("torch")


def _checkpoint(path: Path) -> Path:
    from harness.metrics import STILL_METRICS
    from harness.model.data import METRIC_COND_DIM, Normaliser
    from harness.model.net import CondUNet
    net = CondUNet(base=8, mults=(1, 2), blocks=1)
    norm = Normaliser(np.zeros(len(STILL_METRICS)), np.ones(len(STILL_METRICS)))
    torch.save({"config": net.config, "ema": net.state_dict(), "normaliser": norm.state(),
                "cond_restore": np.zeros(METRIC_COND_DIM, dtype=np.float32).tolist(), "step": 7}, path)
    return path


def test_photo_restores_a_raw_with_its_measured_law_and_an_image_blind(tmp_path):
    from harness.colorspace import srgb_to_linear
    from harness.model.apply import load_model
    from harness.photo import restore_photo
    b = load_model(_checkpoint(tmp_path / "m.pt"), "cpu")
    x = _noisy(_grey_ramp(256, 320), 2000.0, 3.0)
    raw = tmp_path / "shot.dng"
    raw.write_bytes(b"not a dng")                 # the stub decode stands in for LibRaw; the NoiseProfile read fails soft
    rec = restore_photo(b, raw, tmp_path / "out", linear=True, original=True,
                        decode=lambda p, s: (np.clip(x, 0.0, 1.0), 12))
    assert rec["capture"]["mode"] == "estimated" and rec["bit_depth"] == 12 and rec["dng_noise_profile"] is None
    assert len(rec["descriptor"]) == 7 and rec["descriptor"][-1] == 1.0 and rec["checkpoint_step"] == 7
    out = tmp_path / "out"
    assert json.loads((out / "shot.json").read_text(encoding="utf-8"))["files"] == rec["files"]
    t = read_tiff(out / "shot.tif")
    assert t.shape == (256, 320, 3) and t.dtype == np.uint16
    lin = read_tiff(out / "shot_linear.tif")
    assert lin.dtype == np.float32 and np.abs(lin - np.clip(x, 0, 1)).max() < 0.05   # a fresh net is near identity
    assert np.abs(srgb_to_linear(t / 65535.0) - np.clip(lin, 0, 1)).max() < 1e-3
    assert (out / "shot.jpg").exists() and (out / "shot_original.tif").exists()

    from PIL import Image
    Image.fromarray((np.clip(_grey_ramp(64, 96), 0, 1) * 255).astype(np.uint8)).save(tmp_path / "web.png")
    rec = restore_photo(b, tmp_path / "web.png", out)
    assert rec["capture"]["mode"] == "unknown" and rec["descriptor"] is None and not rec["raw"]


def test_photo_command_takes_a_folder(tmp_path):
    from PIL import Image
    from harness.cli import main
    ck = _checkpoint(tmp_path / "m.pt")
    d = tmp_path / "in"
    d.mkdir()
    for k in range(2):
        Image.fromarray((_grey_ramp(48, 64) * 255).astype(np.uint8)).save(d / f"p{k}.png")
    (d / "notes.txt").write_text("skipped", encoding="utf-8")
    assert main(["photo", str(d), "--ckpt", str(ck), "--out", str(tmp_path / "o"), "--device", "cpu", "--ev", "auto"]) == 0
    assert sorted(p.name for p in (tmp_path / "o").glob("*.tif")) == ["p0.tif", "p1.tif"]


def test_export_writes_a_release_that_restores_as_the_checkpoint_does(tmp_path):
    from harness.model.apply import load_model, restore_frames
    from harness.model.export import CONFIG, export
    ck = _checkpoint(tmp_path / "m.pt")
    cfg = export(ck, tmp_path / "rel", name="tiny")
    assert cfg["step"] == 7 and cfg["parameters"] > 0 and cfg["blind_noise"]["method"]
    x = np.clip(_noisy(_grey_ramp(64, 64), 800.0, 3.0), 0, 1)[None]
    a = restore_frames(load_model(ck, "cpu"), x)
    for p in (tmp_path / "rel", tmp_path / "rel" / CONFIG):
        assert np.array_equal(restore_frames(load_model(p, "cpu"), x), a)
    w = tmp_path / "rel" / "model.safetensors"
    w.write_bytes(w.read_bytes()[:-1] + b"\x00")
    with pytest.raises(ValueError):
        load_model(tmp_path / "rel", "cpu")


def test_capture_eval_scores_the_three_modes(tmp_path, monkeypatch):
    import harness.data.realpairs as rp
    from harness.model.captureeval import evaluate, to_markdown
    clean = _grey_ramp(160, 192)

    def match_of(p, base, scale, decode=None):
        x = np.clip(_noisy(clean, 1500.0, 3.0, seed=len(p.in_file)), 0, 1)
        m = {"x": x, "t": clean, "valid": np.ones(clean.shape[:2], dtype=bool),
             "noise": {"electrons_at_clip": 1500.0, "read_noise_e": 3.0}}
        return m, x, 12, clean

    pairs = [rp.Pair("fake", f"s{k}", f"s{k}/ref.dng", f"s{k}/in{'x' * k}.dng", "val") for k in range(3)]
    monkeypatch.setattr(rp, "pairs_of", lambda source, root=None, log=None: pairs + [
        rp.Pair("fake", "t", "t/ref.dng", "t/in.dng", "train")])
    monkeypatch.setattr(rp, "match_of", match_of)
    rep = evaluate(_checkpoint(tmp_path / "m.pt"), ["fake"], n_pairs=2, tiles_per_pair=2, tile=96, scale=2,
                   device="cpu", root=tmp_path, log=lambda *a: None)
    s = rep["sources"]["fake"]
    assert len(s["pairs"]) == 2 and s["tiles"] == 4 and set(s["modes"]) == {"measured", "estimated", "unknown"}
    assert all(p["split"] == "val" and p["estimated_mode"] == "estimated" for p in s["pairs"])
    assert abs(s["log10_electrons_error"]["median"]) < 1.0
    md = to_markdown(rep)
    assert "| metric | input | measured | estimated | unknown | truth |" in md


def test_card_states_the_release_its_reports_and_the_licences(tmp_path):
    from harness.model.card import render
    from harness.model.export import export
    run = tmp_path / "recovered_x"
    run.mkdir()
    ck = _checkpoint(run / "best.pt")
    c = torch.load(ck, weights_only=False)
    c["train"] = {"sources": ["hf-tid2013", "polyhaven-hdris"], "real": [], "weights": {"polyhaven-hdris": 2}}
    torch.save(c, ck)
    gate = {"name": "artifacts", "status": "fail", "checks": [
        {"metric": "edge_overshoot", "status": "fail", "detail": "loses capture, margin -0.02"}]}
    for name, overall in (("heldout_real_fake_known", "PASS"), ("heldout_real_fake_blind", "FAIL at artifacts")):
        d = run / name
        d.mkdir()
        g = {"overall": overall, "gates": [] if overall == "PASS" else [gate]}
        (d / "heldout.json").write_text(json.dumps({"n": 8, "noise": "residual", "gates": {"capture": g}}),
                                        encoding="utf-8")
        (d / "heldout.md").write_text(f"# Real-pair evaluation: {name}\n", encoding="utf-8")
    export(ck, tmp_path / "rel", name="tiny")
    md = render(tmp_path / "rel")
    assert md.startswith("# tiny") and "| step | 7 |" in md
    assert "| `heldout_real_fake_known` | Real-pair evaluation: heldout_real_fake_known | 8 | **PASS** |" in md
    assert "capture: artifacts: edge_overshoot loses capture, margin -0.02" in md
    assert "Every gate set passes on 1 of 2 set(s): `heldout_real_fake_known`." in md
    lic = md.split("## Licence", 1)[1]
    assert "`hf-tid2013`" in lic and "polyhaven" not in lic
    cal = tmp_path / "calib_s1.json"
    cal.write_text(json.dumps({"scale": 1, "method": "blocks", "crop": 1024, "pairs": 4, "factor": 2.5,
                               "spread": {"n": 4, "median": 0.398, "p10": 0.2, "p90": 0.5},
                               "per_source": {"fake": {"n": 4, "median": 0.398, "p10": 0.2, "p90": 0.5}}}),
                   encoding="utf-8")
    md = render(tmp_path / "rel", calibrations=[cal])
    assert "The factor at decode scale 1 is the median" in md and "on 4 train pair(s)" in md and "2.500" in md


def test_calibrate_decoded_finds_the_factor_that_reads_a_scale_as_measured(tmp_path, monkeypatch):
    import harness.data.blindnoise as bn
    import harness.data.realpairs as rp
    clean = _grey_ramp(256, 256)

    def match_of(p, base, scale, decode=None):
        x = np.clip(_noisy(clean, 2500.0, 3.0, seed=len(p.in_file)), 0, 1)
        return {"x": x, "noise": {"electrons_at_clip": 2500.0}}, x, 12, clean

    pairs = [rp.Pair("fake", f"s{k}", f"s{k}/ref.dng", f"s{k}/in{'x' * k}.dng", "train") for k in range(3)]
    monkeypatch.setattr(rp, "pairs_of", lambda source, root=None, log=None: pairs + [
        rp.Pair("fake", "v", "v/ref.dng", "v/in.dng", "val")])
    monkeypatch.setattr(rp, "match_of", match_of)
    rep = bn.calibrate_decoded(["fake"], scale=1, n=5, root=tmp_path, log=lambda *a: None)
    assert rep["pairs"] == 3 and all(r["split"] == "train" for r in rep["rows"])
    monkeypatch.setitem(bn.SCALE_FACTOR, (bn.METHOD, 1), rep["factor"])
    x = np.clip(_noisy(clean, 2500.0, 3.0, seed=9), 0, 1)
    assert bn.var_scale_for(bn.METHOD, 1) == pytest.approx(bn.VAR_SCALE[bn.METHOD] * rep["factor"])
    assert 0.85 < bn.blind_law(x, scale=1)["electrons_at_clip"] / 2500.0 < 1.15
    assert bn.blind_law(x, scale=2)["electrons_at_clip"] == pytest.approx(bn.blind_law(x)["electrons_at_clip"])
