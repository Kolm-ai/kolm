# The real-pair gates: mtf50 and texture judged by distance to the truth's own value, a beats_camera check passed
# on a tie when the camera already sits at the truth, and the camera's own finished photo (HDR+ final.jpg)
# matched to the target as the second baseline.
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from harness.colorspace import linear_to_srgb
from harness.data.realpairs import (HDRPLUS, RAWNIND, Pair, _isotonic, build_set, camera_file, camera_overlap,
                                    load_set, match_camera, match_pair, save_set, set_path)
from harness.gate import overall, run_gates
from harness.metrics import STILL_METRICS, at_truth, verdict
from harness.model.heldout import CAMERA_GATES, CAPTURE_GATES, score, to_markdown

MIX = np.array([[0.85, 0.10, 0.05], [0.08, 0.80, 0.12], [0.04, 0.16, 0.80]])


def _gate(metric: str, op: str = "beats_camera") -> list[dict]:
    return [{"name": "g", "checks": [{"metric": metric, "op": op}]}]


def _scene(rng: np.random.Generator, h: int, w: int, block: int = 8) -> np.ndarray:
    # a colour chart of constant blocks, each channel its own shuffled levels, with a soft texture on top
    nb = (h // block) * (w // block)
    levels = np.stack([rng.permutation(np.geomspace(0.01, 0.8, nb)) for _ in range(3)], axis=-1)
    img = np.kron(levels.reshape(h // block, w // block, 3), np.ones((block, block, 1)))
    return np.clip(img * (1 + 0.1 * gaussian_filter(rng.normal(0, 1, (h, w, 1)), 1.5)), 0.0, 0.95)


def _render(t: np.ndarray) -> np.ndarray:
    # the camera's rendering of t: a low-frequency gain (its local tone mapping), a colour mix, a tone curve
    h, w = t.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    g = (0.7 + 0.5 * np.exp(-((yy - 0.4) ** 2 + (xx - 0.6) ** 2) / 0.1))[..., None]
    return np.clip((t * g) @ MIX.T, 0.0, 1.0) ** 0.75


# ---- the verdict and the gate --------------------------------------------------------------------------------------

def test_mtf50_and_texture_are_judged_by_distance_to_the_truth():
    assert verdict("mtf50_cpp", 0.9, 0.3, 0.6)[0] == "ties"            # as far above the truth as the camera is below
    assert verdict("mtf50_cpp", 0.55, 0.3, 0.6) == ("beats", pytest.approx(0.25))
    assert verdict("mtf50_cpp", 0.9, 0.3)[0] == "beats"                 # with no truth, higher is better
    assert verdict("mtf50_cpp", 0.4, 0.5, float("nan"))[0] == "loses"   # an unknown truth is no truth
    assert verdict("texture_hf_fidelity", 1.03, 1.22, 1.0)[0] == "beats"  # the capture's noise read as texture
    assert verdict("texture_hf_fidelity", 1.03, 1.22)[0] == "loses"
    assert verdict("shadow_snr_db", 20.0, 14.0, 17.0)[0] == "beats"     # not judged to the truth: a denoiser may pass it


def test_a_beats_check_passes_on_a_tie_only_when_the_camera_is_at_the_truth():
    g = run_gates(_gate("mtf50_cpp"), {"mtf50_cpp": 0.49}, {"mtf50_cpp": 0.5}, {"mtf50_cpp": 0.5})
    assert g[0].status == "pass" and overall(g) == "PASS"
    assert "by distance to the truth 0.5" in g[0].checks[0].detail
    assert "a tie is the most any candidate can do" in g[0].checks[0].detail
    assert run_gates(_gate("mtf50_cpp"), {"mtf50_cpp": 0.49}, {"mtf50_cpp": 0.5})[0].status == "fail"
    assert run_gates(_gate("mtf50_cpp"), {"mtf50_cpp": 0.3}, {"mtf50_cpp": 0.5}, {"mtf50_cpp": 0.5})[0].status == "fail"
    hl = {"highlight_loss_fraction": 0.0}
    assert run_gates(_gate("highlight_loss_fraction"), hl, hl, hl)[0].status == "pass"
    snr = {"shadow_snr_db": 14.0}
    s = run_gates(_gate("shadow_snr_db"), snr, snr, snr)[0]
    assert s.status == "fail" and "the most any" not in s.checks[0].detail
    assert run_gates(_gate("shadow_snr_db", "ties_or_beats_camera"), snr, snr, snr)[0].status == "pass"
    assert at_truth("mtf50_cpp", 0.5, 0.51) and not at_truth("mtf50_cpp", 0.5, 0.6)
    assert not at_truth("mtf50_cpp", 0.5, None) and not at_truth("mtf50_cpp", float("nan"), 0.5)
    assert not at_truth("dr_steps", 10.0, 10.0) and not at_truth("shadow_snr_db", 14.0, 14.0)


def test_both_gate_sets_check_every_still_metric_once():
    for spec in (CAPTURE_GATES, CAMERA_GATES):
        # every still metric once but highlight loss, which a target clipped at 1 cannot measure
        ms = sorted(c["metric"] for g in spec for c in g["checks"])
        assert ms == sorted(m for m in STILL_METRICS if m != "highlight_loss_fraction")
    beats = {c["metric"] for g in CAPTURE_GATES for c in g["checks"] if c["op"] == "beats_camera"}
    assert beats == {"shadow_snr_db", "banding_score"}
    ties = {c["metric"] for g in CAMERA_GATES for c in g["checks"] if c["op"] == "ties_or_beats_camera"}
    assert ties == {"dr_steps"}


# ---- the held-out score ----------------------------------------------------------------------------------------------

def test_score_judges_the_camera_on_the_tiles_that_have_one():
    rng = np.random.default_rng(3)
    shade = np.where(np.arange(64) < 24, 0.02, 1.0)[None, :, None]     # a band of shadows for shadow SNR
    ys = [_scene(rng, 64, 64) * shade for _ in range(3)]
    samples = [{"x": np.clip(y + rng.normal(0, 0.03, y.shape), 0.0, 1.0), "y": y,
                "camera": (y.copy() if k < 2 else None)} for k, y in enumerate(ys)]
    rep = score(samples, lambda s: s["y"])
    assert rep["n"] == 3 and len(rep["outputs"]) == 3 and "camera" not in rep["tiles"][2]
    assert all(set(r["truth"]) == set(STILL_METRICS) for r in rep["tiles"])
    for m, s in rep["metrics"].items():
        assert {"input", "output", "truth", "median_verdict", "camera"} <= set(s), m
        c = s["camera"]
        assert c["beats"] + c["ties"] + c["loses"] <= 2 and c["loses"] == 0, (m, c)
    assert rep["metrics"]["mtf50_cpp"]["truth"] == pytest.approx(0.5)
    g = rep["gates"]
    assert g["camera"]["tiles"] == 2
    # model and camera both at the truth: every truth-bound tie passes, and shadow SNR cannot be beaten
    assert g["camera"]["overall"] == "FAIL at denoise"
    assert [x["name"] for x in g["camera"]["gates"]] == ["lighting_materials", "tone_color", "artifacts", "denoise"]
    cap = {c["metric"]: c for x in g["capture"]["gates"] for c in x["checks"]}
    assert cap["delta_e_mean"]["status"] == "pass" and cap["banding_score"]["status"] == "pass"
    assert "the truth" + chr(39) + "s value" in cap["banding_score"]["detail"]   # the input already has no banding
    rep.pop("outputs")
    rep.update(ckpt="x.pt", step=1, blind=False, source=HDRPLUS, seed=0, tile=64, scale=2, preset="real", ss=1,
               real=True, seconds=0.0, records=[], skipped={})
    md = to_markdown(rep)
    assert "Against the camera's own finished photo (2 tiles" in md and "| truth |" in md
    assert "Gates against the camera's photo: **FAIL at denoise**" in md and "Gates against the real capture: **" in md
    assert "camera at the truth's value" in md
    plain = score(samples[2:], lambda s: s["y"])
    assert "camera" not in plain["gates"] and all("camera" not in s for s in plain["metrics"].values())


# ---- matching the camera's photo ---------------------------------------------------------------------------------------

def test_isotonic_pools_adjacent_violators():
    assert _isotonic(np.array([1.0, 3.0, 2.0, 4.0]), np.ones(4)).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert _isotonic(np.array([1.0, 3.0, 2.0]), np.array([1.0, 1.0, 3.0])).tolist() == [1.0, 2.25, 2.25]
    assert _isotonic(np.array([3.0, 2.0, 1.0]), np.ones(3)).tolist() == [2.0, 2.0, 2.0]
    rng = np.random.default_rng(0)
    y, w = rng.normal(size=300).cumsum(), rng.uniform(0.1, 2.0, 300)
    f = _isotonic(y, w)
    assert f.shape == y.shape and np.all(np.diff(f) >= -1e-12)
    assert np.isclose((f * w).sum(), (y * w).sum())     # each pool keeps its weighted mean
    assert np.allclose(_isotonic(np.sort(y), w), np.sort(y))


def test_match_camera_undoes_the_cameras_rendering():
    rng = np.random.default_rng(2)
    t = _scene(rng, 128, 160)
    c = _render(t)
    c[:6] = np.nan                                        # rows the camera's photo does not cover
    valid = np.ones(t.shape[:2], dtype=bool)
    out, info = match_camera(c, t, valid)
    assert np.isnan(out[:6]).all() and np.isfinite(out[6:]).all() and out[6:].min() >= 0.0 and out[6:].max() <= 1.0
    before, after = np.median(np.abs(c[6:] - t[6:])), np.median(np.abs(out[6:] - t[6:]))
    assert after < 0.25 * before and after < 0.02, (before, after)
    assert info["residual_rms"] < 0.02 and info["fit_pixels"] > 1000 and np.array(info["matrix"]).shape == (3, 3)
    lo, hi = info["gain_p5_p95"]
    assert 0.25 <= lo <= hi <= 4.0
    with pytest.raises(ValueError, match="pixels to fit"):
        match_camera(c, t, np.zeros_like(valid))


def test_camera_overlap_aligns_the_photo_to_the_pair_and_cuts_its_overlap():
    rng = np.random.default_rng(4)
    h, w, pad = 160, 192, 16
    big = _scene(rng, h + 2 * pad, w + 2 * pad)
    y = big[pad:pad + h, pad:pad + w]
    x = np.clip(big[pad + 3:pad + 3 + h, pad - 2:pad - 2 + w] * 0.8 + rng.normal(0, 0.01, (h, w, 3)), 0.0, 1.0)
    cam = _render(big[pad + 4:pad + 4 + h, pad - 5:pad - 5 + w])   # y[i, j] ~ cam[i - 4, j + 5]
    m = match_pair(x, y)
    assert tuple(m["align"]["shift"]) == (3, -2)
    out, info = camera_overlap(m, y, cam)
    assert info["shift"] == [-4, 5] and info["corr"] > 0.9 and out.shape == m["t"].shape
    have = np.isfinite(out).all(axis=-1)
    assert have[1:, :-3].all() and not have[0].any() and not have[:, -3:].any()
    assert np.median(np.abs(out[have] - m["t"][have])) < 0.02
    with pytest.raises(ValueError, match="camera photo 191x160"):
        camera_overlap(m, y, cam[:, :-1])
    with pytest.raises(ValueError, match="not aligned"):
        camera_overlap(m, y, _render(_scene(np.random.default_rng(9), h, w)))


def test_camera_file_is_hdrplus_final_jpg_beside_the_merge():
    p = Pair(HDRPLUS, "b0", "results_20171023/b0/merged.dng", "bursts/b0/payload_N000.dng", "val")
    assert camera_file(p) == "results_20171023/b0/final.jpg"
    assert camera_file(Pair(RAWNIND, "s", "a_GT_ISO100.dng", "a_ISO3200.dng", "val")) is None


# ---- the set ---------------------------------------------------------------------------------------------------------

def _hdrplus(tmp_path, rng):
    # three HDR+-shaped pairs, the raws from a stub decoder, final.jpg a real 8-bit sRGB JPEG of the camera's
    # rendering, shifted against the merge; the third burst has no final.jpg
    from PIL import Image
    base = tmp_path / "raw" / HDRPLUS
    imgs, pairs = {}, []
    h, w, pad = 128, 160, 16
    for k, split in enumerate(["val", "test", "val"]):
        big = _scene(rng, h + 2 * pad, w + 2 * pad)
        d = f"results_20171023/b{k}"
        imgs[f"b{k}/merged.dng"] = big[pad:pad + h, pad:pad + w]
        imgs[f"b{k}/payload_N000.dng"] = np.clip(big[pad + 2:pad + 2 + h, pad + 1:pad + 1 + w] * 0.8
                                                 + rng.normal(0, 0.01, (h, w, 3)), 0.0, 1.0)
        if k < 2:
            (base / d).mkdir(parents=True)
            cam = _render(big[pad - 1:pad - 1 + h, pad + 3:pad + 3 + w])
            Image.fromarray(np.round(np.asarray(linear_to_srgb(cam)) * 255).astype(np.uint8)).save(
                base / d / "final.jpg", quality=95)
        pairs.append(Pair(HDRPLUS, f"b{k}", f"{d}/merged.dng", f"bursts/b{k}/payload_N000.dng", split, licence="CC-BY"))
    return imgs, pairs


def test_the_hdrplus_set_keeps_the_camera_photo_and_attaches_it_to_an_old_cache(tmp_path):
    imgs, pairs = _hdrplus(tmp_path, np.random.default_rng(6))

    def decode(path, scale):
        return imgs[f"{path.parent.name}/{path.name}"], 12

    rs = build_set(HDRPLUS, n=9, tile=32, scale=1, seed=0, root=tmp_path, log=None, decode=decode, pairs=pairs)
    assert rs.cameras.shape == rs.targets.shape and rs.cameras.dtype == np.float32 and rs.has_camera.dtype == bool
    assert set(rs.camera_fits) == {"0", "1"} and all(f["shift"] == [1, -3] for f in rs.camera_fits.values())
    assert all(f["residual_rms"] < 0.03 for f in rs.camera_fits.values())
    pair_of = np.array([r.pair for r in rs.records])
    assert not rs.has_camera[pair_of == 2].any() and rs.has_camera[pair_of < 2].any()
    has = rs.has_camera
    assert np.median(np.abs(rs.cameras[has] - rs.targets[has])) < 0.03
    assert (rs.cameras[~has] == 0).all()

    p = set_path(HDRPLUS, 0, 9, 32, 1, tmp_path)
    again = load_set(p)
    assert np.array_equal(again.cameras, rs.cameras) and np.array_equal(again.has_camera, rs.has_camera)
    assert again.camera_fits == rs.camera_fits

    old = load_set(p)                                      # a set cached before the camera photo was kept
    old.cameras, old.has_camera, old.camera_fits = None, None, {}
    save_set(old, p)
    assert load_set(p).cameras is None
    logged = []
    cached = build_set(HDRPLUS, n=9, tile=32, scale=1, seed=0, root=tmp_path, log=logged.append, decode=decode,
                       pairs=None)
    assert np.allclose(cached.cameras, rs.cameras, atol=1e-6) and np.array_equal(cached.has_camera, rs.has_camera)
    assert np.array_equal(cached.tiles, rs.tiles) and np.array_equal(cached.targets, rs.targets)
    assert cached.camera_fits["2"] == {"error": "no camera photo on disk"} and "residual_rms" in cached.camera_fits["0"]
    assert any("camera photo on" in line for line in logged)
    saved = load_set(p)
    assert saved.cameras is not None and saved.camera_fits == cached.camera_fits
    # attached once, the cache is the set again: no decoder needed
    assert build_set(HDRPLUS, n=9, tile=32, scale=1, seed=0, root=tmp_path, log=None, decode=None,
                     pairs=None).camera_fits == cached.camera_fits


def test_evaluate_real_scores_against_the_camera_where_the_set_has_it(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from harness.model import apply
    from harness.model.heldout import evaluate
    imgs, pairs = _hdrplus(tmp_path, np.random.default_rng(6))
    rs = build_set(HDRPLUS, n=9, tile=32, scale=1, seed=0, root=tmp_path, log=None,
                   decode=lambda p, s: (imgs[f"{p.parent.name}/{p.name}"], 12), pairs=pairs)
    monkeypatch.setattr(apply, "load_model", lambda ckpt, device=None, bf16=False: SimpleNamespace(step=3))
    monkeypatch.setattr(apply, "restore_frames", lambda b, frames, white_level=1.0, capture=None: frames)
    rep = evaluate("x.pt", source=HDRPLUS, n=9, tile=32, seed=0, root=tmp_path, log=None, real=True, scale=1)
    assert rep["gates"]["camera"]["tiles"] == int(rs.has_camera.sum())
    assert sum("camera" in r for r in rep["tiles"]) == int(rs.has_camera.sum())
    assert rep["noise"] == "residual" and rep["gates"]["capture"]["overall"] != "PASS"  # 32 px: too few shadows inside the border
    assert rep["gates"]["capture_hp"]["overall"].startswith("FAIL at")    # the identity does not denoise
    assert rep["gates"]["frontier"]["match"]["tiles"] == 9 and rep["gates"]["frontier"]["overall"] != "PASS"
    assert "Gates against the camera's photo" in to_markdown(rep)
