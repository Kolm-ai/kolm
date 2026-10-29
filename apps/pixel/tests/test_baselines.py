"""baselines.py: public restoration baselines through the held-out gates and on SIDD validation. No released
weights: a stand-in network takes their place, and the vendored architectures are built only where present."""
from __future__ import annotations

import argparse
import io
import json

import numpy as np
import pytest
from scipy.ndimage import correlate, gaussian_filter
from scipy.signal import correlate2d

from harness.colorspace import srgb_to_linear
from harness.model import baselines as bl
from harness.model.compare import collect, collect_sidd
from harness.model.compare import to_markdown as compare_markdown
from harness.model.heldout import build_set, report_name
from test_heldout import SOURCE, raw_root  # noqa: F401  (the raw_root fixture)


def _cpu(module):
    import torch
    return module, torch.device("cpu")


def _sidd_blocks(n: int = 3, side: int = 64, seed: int = 3):
    rng = np.random.default_rng(seed)
    smooth = gaussian_filter(rng.random((n, side, side, 3)), (0, 2, 2, 0))
    clean = (np.clip(smooth * 2.5 - 0.75, 0, 1) * 255).round().astype(np.uint8)
    noisy = np.clip(clean + rng.normal(0, 10, clean.shape), 0, 255).round().astype(np.uint8)
    return noisy, clean


def test_report_name_is_the_rule_the_runs_directories_follow():
    assert report_name("raw-pixls-us") == "heldout_known"
    assert report_name("raw-pixls-us", blind=True) == "heldout_blind"
    assert report_name("raw-pixls-us", subpix=True, blind=True) == "heldout_blind"   # sp_ only on real pairs
    assert report_name("rawnind", real=True, blind=True) == "heldout_real_rawnind_blind"
    assert report_name("hdr-plus-burst", True, True) == "heldout_real_hdr-plus-burst_sp_known"


def test_every_baseline_is_a_ledgered_evaluation_only_source():
    from harness.data.manifest import by_id, load_manifest
    src = by_id(load_manifest())
    for b in bl.BASELINES.values():
        s = src[b.source]
        assert s.kind == "model" and s.access == "direct" and "evaluation" in s.harness_use
        assert [f.path for f in s.files] == [b.file] and s.files[0].size
    assert bl.SIDD_SOURCE in src


def test_load_net_names_the_collect_command_when_the_weights_are_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="data collect --source baseline-scunet-real"):
        bl.load_net("scunet", device="cpu", root=tmp_path)


def test_the_srgb_curve_there_and_back_is_the_identity_on_the_unit_range():
    x = np.random.default_rng(0).random((16, 16, 3))
    np.testing.assert_allclose(bl.through_srgb(lambda s: s)(x), x, atol=1e-9)
    np.testing.assert_allclose(bl.through_linear(lambda s: s)(x), x, atol=1e-9)
    np.testing.assert_allclose(bl.through_srgb(lambda s: s)(np.full((2, 2, 3), 3.0)), 1.0)   # clipped at 1
    grey = bl.through_srgb(lambda s: np.full_like(s, 0.5))(x)                                 # the net sees sRGB
    np.testing.assert_allclose(grey, srgb_to_linear(np.full_like(x, 0.5)))


def test_srgb_restorer_pads_to_the_multiple_crops_back_and_clamps():
    torch = pytest.importorskip("torch")
    seen = []

    class Probe(torch.nn.Module):
        def forward(self, t):
            seen.append(tuple(t.shape))
            return t * 2.0
    img = np.random.default_rng(1).random((20, 30, 3))
    out = bl.srgb_restorer(Probe(), 16, torch.device("cpu"))(img)
    assert seen == [(1, 3, 32, 32)] and out.shape == (20, 30, 3) and out.dtype == np.float64
    np.testing.assert_allclose(out, np.clip(img * 2.0, 0, 1), atol=1e-6)


def test_psnr_is_the_mean_squared_error_against_the_peak():
    a, b = np.zeros((4, 4, 3)), np.full((4, 4, 3), 10.0)
    assert bl.psnr(a, b) == pytest.approx(10 * np.log10(255 ** 2 / 100))
    assert bl.psnr(a / 255, b / 255, 1.0) == pytest.approx(bl.psnr(a, b))
    assert bl.psnr(a, a) == float("inf")


def _ref_map(x, y, mean, peak=255.0):
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    mx, my = mean(x), mean(y)
    sx, sy, sxy = mean(x * x) - mx ** 2, mean(y * y) - my ** 2, mean(x * y) - mx * my
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sx + sy + c2))


def test_both_ssims_are_their_references_and_one_on_the_truth():
    noisy, clean = _sidd_blocks(1, 32)
    x, y = noisy[0].astype(np.float64), clean[0].astype(np.float64)
    g = bl._gauss()
    assert g.size == 11 and g.sum() == pytest.approx(1.0)
    w2 = np.outer(g, g)   # BasicSR: cv2.filter2D with the outer product, the valid region [5:-5, 5:-5]
    ref2 = np.mean([_ref_map(x[..., c], y[..., c], lambda z: correlate2d(z, w2, mode="valid")).mean()
                    for c in range(3)])
    w3 = np.einsum("i,j,k->ijk", g, g, g)   # NAFNet: Conv3d over (H, W, C), replicate padding, the whole map
    ref3 = _ref_map(x, y, lambda z: correlate(z, w3, mode="nearest")).mean()
    assert bl.ssim(x, y) == pytest.approx(ref2, rel=1e-9) and bl.ssim3d(x, y) == pytest.approx(ref3, rel=1e-9)
    assert bl.ssim(y, y) == pytest.approx(1.0) and bl.ssim3d(y, y) == pytest.approx(1.0)
    assert bl.ssim(x, y) < 1.0 and bl.ssim3d(x, y) < 1.0


def test_sidd_val_reads_the_packs_in_block_order(tmp_path, monkeypatch):
    lmdb = pytest.importorskip("lmdb")
    from PIL import Image
    order = (2, 10, 1)   # meta_info.txt's order; the blocks come back 1, 2, 10
    for rel, base in zip(bl.SIDD_VAL, (0, 100)):
        d = tmp_path / "raw" / bl.SIDD_SOURCE / rel
        d.mkdir(parents=True)
        env = lmdb.open(str(d), map_size=1 << 22)
        with env.begin(write=True) as txn:
            for i in order:
                buf = io.BytesIO()
                Image.fromarray(np.full((8, 8, 3), base + i, np.uint8)).save(buf, format="PNG")
                txn.put(f"ValidationBlocksSrgb_{i}".encode("ascii"), buf.getvalue())
        env.close()
        (d / "meta_info.txt").write_text("".join(f"ValidationBlocksSrgb_{i}.png (8,8,3) 1\n" for i in order),
                                         encoding="utf-8")
    monkeypatch.setattr(bl, "SIDD_BLOCKS", 3)
    noisy, clean = bl.sidd_val(tmp_path)
    assert noisy.shape == clean.shape == (3, 8, 8, 3) and noisy.dtype == np.uint8
    assert list(noisy[:, 0, 0, 0]) == [1, 2, 10] and list(clean[:, 0, 0, 0]) == [101, 102, 110]
    monkeypatch.setattr(bl, "SIDD_BLOCKS", 1280)
    with pytest.raises(ValueError, match="want 1280 pairs"):
        bl.sidd_val(tmp_path)


def test_sidd_score_of_the_identity_is_the_input_and_its_report():
    noisy, clean = _sidd_blocks()
    rep = bl.sidd_score(lambda img: img, noisy, clean, log=None)
    assert rep["n"] == 3 and len(rep["blocks"]) == 3
    for k in bl.SIDD_SCORES + bl.SIDD_METRICS:
        assert rep["output"][k] == pytest.approx(rep["input"][k]), k
    assert rep["input"]["psnr_float"] == pytest.approx(rep["input"]["psnr"])
    blur = bl.sidd_score(lambda img: gaussian_filter(img, (1, 1, 0)), noisy, clean, log=None)
    assert blur["output"]["psnr"] != pytest.approx(blur["input"]["psnr"])
    md = bl.sidd_markdown({"baseline": "nafnet", "licence": "MIT", "trained_on": "SIDD Medium sRGB",
                           "reported": "40.3045 dB", **rep})
    assert md.startswith("# SIDD validation: 3 sRGB blocks") and "| input |" in md and "| output |" in md
    assert "Public baseline `nafnet`" in md and "report 40.3045 dB" in md


def test_evaluate_sidd_writes_both_reports_where_compare_reads_them(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from harness.model import apply
    noisy, clean = _sidd_blocks()
    monkeypatch.setattr(bl, "sidd_val", lambda root=None: (noisy, clean))
    rep = bl.evaluate_sidd("nafnet", root=tmp_path, log=None, net=_cpu(torch.nn.Identity()))
    d = tmp_path / "models" / "baseline_nafnet" / "sidd_val"
    assert json.loads((d / "sidd.json").read_text(encoding="utf-8"))["output"] == rep["output"]
    assert "its authors report 40.3045 dB" in (d / "sidd.md").read_text(encoding="utf-8")
    monkeypatch.setattr(apply, "load_model", lambda ckpt, device=None, bf16=False: SimpleNamespace(step=7))
    monkeypatch.setattr(apply, "restore_frames",
                        lambda b, frames, white_level=1.0, capture=None: gaussian_filter(frames, (0, 0.8, 0.8, 0)))
    ckpt = tmp_path / "models" / "recovered_v99" / "best.pt"
    ours = bl.evaluate_sidd(ckpt=str(ckpt), root=tmp_path, log=None)
    assert ours["step"] == 7 and "Checkpoint" in (ckpt.parent / "sidd_val" / "sidd.md").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="one of them"):
        bl.evaluate_sidd(root=tmp_path, log=None)
    sidd = collect_sidd(tmp_path / "models")
    assert list(sidd) == ["v99", "nafnet"]
    md = compare_markdown(collect(tmp_path / "models"), sidd)
    assert "## SIDD validation" in md and "| input | - |" in md
    assert "| v99 | 7 |" in md and "| nafnet (public) | - |" in md


def test_evaluate_heldout_writes_blind_reports_beside_the_runs(raw_root, monkeypatch):  # noqa: F811
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(bl, "HELDOUT_SETS", ((SOURCE, False, False), ("rawnind", True, False)))
    build_set(SOURCE, n=3, tile=64, seed=1, root=raw_root, log=None)   # a baseline never builds a set
    logs = []
    reps = bl.evaluate_heldout("restormer", root=raw_root, n=3, tile=64, seed=1, save_tiles=2, log=logs.append,
                               net=_cpu(torch.nn.Identity()))
    assert list(reps) == ["heldout_blind"] and any("heldout_real_rawnind_blind skipped" in s for s in logs)
    d = raw_root / "models" / "baseline_restormer" / "heldout_blind"
    rep = json.loads((d / "heldout.json").read_text(encoding="utf-8"))
    assert rep["baseline"] == "restormer" and rep["blind"] and "step" not in rep and rep["n"] == 3
    assert {"capture", "frontier", "tone"} <= set(rep["gates"])
    assert np.load(d / "outputs.npz")["outputs"].shape == (3, 64, 64, 3) and (d / "tiles").is_dir()
    md = (d / "heldout.md").read_text(encoding="utf-8")
    assert md.startswith("# Held-out raw evaluation") and "Public baseline `restormer`" in md
    t = collect(raw_root / "models")
    assert list(t["heldout_blind"]) == ["restormer"]
    assert "| restormer (public) | - |" in compare_markdown(t)


def test_the_model_cli_registers_the_baseline_commands():
    from harness.model.cli import add_parser
    p = argparse.ArgumentParser()
    add_parser(p.add_subparsers(dest="cmd"))
    h = p.parse_args(["model", "baseline", "heldout", "--name", "scunet", "--n", "8"])
    assert h.func is bl.cmd_baseline_heldout and h.n == 8 and h.noise == "residual" and not h.no_frontier
    s = p.parse_args(["model", "baseline", "sidd", "--ckpt", "best.pt"])
    assert s.func is bl.cmd_baseline_sidd and s.name is None and s.ckpt == "best.pt"
    with pytest.raises(SystemExit):
        p.parse_args(["model", "baseline", "sidd", "--name", "nafnet", "--ckpt", "x.pt"])


@pytest.mark.parametrize("name", sorted(bl.BASELINES))
def test_a_vendored_architecture_builds_and_keeps_the_shape(name):
    torch = pytest.importorskip("torch")
    b = bl.BASELINES[name]
    mod, cls = b.arch.split(":")
    m = pytest.importorskip(f"harness.model.baseline_archs.{mod}")
    net = getattr(m, cls)(**b.kwargs).eval()
    side = max(64, b.multiple)
    with torch.no_grad():
        y = net(torch.rand(1, 3, side, side))
    assert tuple(y.shape) == (1, 3, side, side)
