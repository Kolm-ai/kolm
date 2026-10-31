"""Real capture/reference pairs (harness.data.realpairs): the shift convention, the gain, the noise law, the pair
listings from a fake RawNIND index and fake HDR+ directories, the seeded cached set with an injected decoder,
its label rows, the real-against-synthetic comparison, the clean include/scene/holdout manifest keys they rely
on, and the CLI wiring. No LibRaw: the decoder is injected. The real-pair evaluation needs torch and is skipped
without it."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import rotate

from harness.data import label as L
from harness.data.label import Item, list_items, scene_of
from harness.data.realpairs import (HDRPLUS, RAWNIND, Pair, align,
                                    build_set, capture_of, compare, compare_markdown, gain, hdrplus_pairs,
                                    input_fits, load_set, match_of, match_pair, noise_law, overlap, phase_shift, rawnind_iso, rawnind_pairs,
                                    rows_path, set_path, summarise, tile_rows)
from harness.metrics import STILL_METRICS
from harness.model.data import UNKNOWN_CAPTURE, capture_descriptor, split_of

RAWNIND_SCENE = "^(?:Bayer|X-Trans)_(?:TEST_)?(.+?)_(?:GT_)?ISO"
HDRPLUS_SCENE = "^(?:bursts|results_[0-9]+)/([^/]+)/"
ELECTRONS, READ = 2000.0, 4.0       # the noise law the fake captures follow: var = t / ELECTRONS + (READ / ELECTRONS)^2
GAIN = np.array([1.3, 1.5, 1.1])    # the reference's exposure and white balance against the input's


def _blocks(rng: np.random.Generator, h: int = 224, w: int = 288, block: int = 16) -> np.ndarray:
    """A grey chart of constant blocks at levels across the noise law's bins, shuffled so nothing repeats."""
    nb = (h // block) * (w // block)
    levels = rng.permutation(np.geomspace(0.006, 0.85, nb)).reshape(h // block, w // block)
    img = np.kron(levels, np.ones((block, block)))
    return np.repeat(img[..., None], 3, axis=-1)


def _capture(clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    a, b = 1.0 / ELECTRONS, (READ / ELECTRONS) ** 2
    return np.clip(clean + rng.normal(0, 1, clean.shape) * np.sqrt(a * clean + b), 0.0, 1.0)


def _pair(rng: np.random.Generator, shift=(5, -3), h: int = 192, w: int = 256):
    """(input, reference, clean input) with input[i, j] ~ reference[i + sy, j + sx]: the reference is the clean
    scene at GAIN times the input's exposure, clipped at 1; the input carries the noise law."""
    pad = 16
    big = _blocks(rng, h + 2 * pad, w + 2 * pad)
    sy, sx = shift
    clean = big[pad:pad + h, pad:pad + w]
    ref = np.clip(big[pad - sy:pad - sy + h, pad - sx:pad - sx + w] * GAIN, 0.0, 1.0)
    return _capture(clean, rng), ref, clean


def test_phase_shift_and_overlap_follow_the_stated_convention():
    rng = np.random.default_rng(1)
    x, y, clean = _pair(rng, (5, -3))
    assert phase_shift(x, y) == (5, -3)
    assert phase_shift(y, x) == (-5, 3)
    xs, ys = overlap(clean, y, 5, -3)
    assert xs.shape == ys.shape == (187, 253, 3)
    valid = ys.max(axis=-1) < 0.98
    assert np.allclose(ys[valid], (xs * GAIN)[valid])


def test_align_refines_and_rejects_rotation_or_another_scene():
    rng = np.random.default_rng(2)
    x, y, _ = _pair(rng, (-7, 4))
    al = align(x, y)
    assert al["shift"] == (-7, 4) and al["aligned"] and al["corr"] > 0.95 and al["quadrant_spread"] <= 1.0
    assert len(al["quadrants"]) == 4 and al["quadrants_held"] == 4
    rotated = np.clip(rotate(y, 3.0, reshape=False, order=1), 0, 1)
    assert not align(x, rotated)["aligned"]
    other, _, _ = _pair(np.random.default_rng(3), (0, 0))
    bad = align(x, other)
    assert not bad["aligned"] and bad["corr"] < 0.5


def test_a_black_or_clipped_quadrant_holds_no_position():
    """A quadrant black in both frames (noise only) or clipped in both has a phase peak that is noise; the
    quadrants that hold a position decide, and too few of them is no alignment."""
    rng = np.random.default_rng(5)
    x, y, _ = _pair(rng, (3, 2))
    x[:96, :128] = _capture(np.zeros((96, 128, 3)), rng)
    y[3:99, 2:130] = _capture(np.zeros((96, 128, 3)), rng)
    x[96:, 128:], y[99:, 130:] = 1.0, 1.0
    al = align(x, y)
    assert al["shift"] == (3, 2) and al["quadrants_held"] == 2 and al["quadrant_spread"] <= 1.0 and al["aligned"]
    x[:96, 128:], y[3:99, 130:] = 1.0, 1.0
    few = align(x, y)
    assert few["shift"] == (3, 2) and few["quadrants_held"] == 1 and not few["aligned"]


def test_an_input_cut_or_turned_against_its_reference_is_brought_to_its_frame(tmp_path):
    """A reference a few rows shorter than its input (a merge that dropped edge rows) or turned a quarter (an
    orientation tag read differently) is still the input's pair: the input is centre-cut, or turned and cut, to
    the reference's frame, the cut's leftover offset is the alignment's, and the reference is never touched."""
    rng = np.random.default_rng(6)
    x, y, _ = _pair(rng, (2, 1), h=244)
    cut = y[4:]
    (xf, fit), = input_fits(x, cut)
    assert fit == {"turn": 0, "crop": [2, 0]} and xf.shape == cut.shape
    assert align(xf, cut)["shift"] == (0, 1) and align(xf, cut)["aligned"]
    turned = np.rot90(cut, 1)
    assert [f["turn"] for _, f in input_fits(x, turned)] == [1, 3]
    imgs = {"in.dng": x, "ref.dng": turned}
    p = Pair("fake", "s", "ref.dng", "in.dng", "train")
    m, xm, _, ym = match_of(p, tmp_path, 1, lambda path, scale: (imgs[path.name], 14))
    assert m["fit"] == {"turn": 1, "crop": [0, 2], "shape_in": [244, 256]} and ym is turned
    assert xm.shape == turned.shape and m["align"]["aligned"] and m["align"]["corr"] > 0.9
    imgs["ref.dng"] = y
    assert "fit" not in match_of(p, tmp_path, 1, lambda path, scale: (imgs[path.name], 14))[0]
    imgs["ref.dng"] = y[20:]
    with pytest.raises(ValueError, match="shapes differ"):
        match_of(p, tmp_path, 1, lambda path, scale: (imgs[path.name], 14))


def test_gain_and_noise_law_recover_what_the_capture_was_made_with():
    rng = np.random.default_rng(4)
    x, y, clean = _pair(rng, (2, 1))
    m = match_pair(x, y)
    assert m["align"]["shift"] == (2, 1) and m["align"]["aligned"]
    assert np.allclose(m["gain"], GAIN, rtol=0.02) and m["gain_pixels"] > 1000
    assert 0.0 < m["clipped"] < 0.2 and m["valid"].shape == m["t"].shape[:2]
    xs, _ = overlap(clean, y, 2, 1)
    assert np.allclose(m["t"][m["valid"]], xs[m["valid"]], atol=0.02)
    nl = m["noise"]
    assert nl["bins"] >= 3 and nl["a"] > 0 and 0.05 < nl["flat_fraction"] < 0.95
    assert 1600 < nl["electrons_at_clip"] < 2400 and 1.0 < nl["read_noise_e"] < 8.0
    assert len(nl["levels"]) == len(nl["variances"]) == nl["bins"]
    g, n = gain(xs, overlap(clean, y, 2, 1)[1])
    assert np.allclose(g, GAIN, rtol=1e-6) and n > 1000     # noiseless: exact
    flat = noise_law(np.full((40, 40, 3), 0.5), np.full((40, 40, 3), 0.5), np.ones((40, 40), bool))
    assert flat["bins"] == 1 and flat["a"] is None and flat["electrons_at_clip"] is None  # one level fits nothing


def test_capture_of_is_the_measured_law_or_unknown():
    cap = capture_of({"electrons_at_clip": 1500.0, "read_noise_e": 3.0}, 14)
    assert np.array_equal(cap, capture_descriptor(0.0, 0.0, 1500.0, 3.0, 14, True)) and cap[-1] == 1.0
    assert np.array_equal(capture_of({"electrons_at_clip": None, "read_noise_e": None, "bins": 0}, 14), UNKNOWN_CAPTURE)
    assert np.array_equal(capture_of({}, 12), UNKNOWN_CAPTURE)


def test_rawnind_iso_reads_the_file_name():
    assert rawnind_iso("Bayer_x_ISO3200_sha1=ab.arw") == 3200.0
    assert rawnind_iso("X-Trans_y_GT_ISO65535_sha1=cd.raf") == 65535.0
    assert rawnind_iso("nothing.arw") is None


def test_clean_include_scene_and_holdout_keys(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(L, "_manifest_clean", lambda: {
        "src": {"include": ["*_GT_ISO*"], "scene": RAWNIND_SCENE, "holdout": ["held*", "UNK_TEST_heldC"]}})
    d = tmp_path / "raw" / "src"
    d.mkdir(parents=True)
    names = ["Bayer_sceneA_GT_ISO100_sha1=a.png", "Bayer_sceneA_ISO3200_sha1=b.png",
             "X-Trans_TEST_heldB_GT_ISO200_sha1=c.png", "other/plain_4k.png", "Bayer_UNK_TEST_heldC_GT_ISO100_sha1=d.png"]
    for n in names:
        (d / n).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(d / n)
    items = list_items("src", tmp_path)
    assert sorted(i.file for i in items) == sorted([names[0], names[2], names[4]])  # include keeps the GT raws
    assert scene_of(Item("src", names[0])) == scene_of(Item("src", names[1])) == "sceneA"
    assert scene_of(Item("src", names[2])) == "heldB"                        # the TEST_ prefix is not the scene
    assert scene_of(Item("src", names[3])) == "other/plain.png"              # no match: the path rule
    assert scene_of(Item("src", names[4])) == "UNK_TEST_heldC"               # an unknown sensor's prefix stays
    assert split_of(Item("src", names[4])) == "test"                         # and the holdout names that key
    assert split_of(Item("src", names[2])) == "test" and split_of(Item("src", names[0])) in ("train", "val")
    assert split_of(Item("src", names[0])) == split_of(Item("src", names[1]))  # the GT and its sweep split together


def test_rawnind_pairs_from_the_index_and_the_reserve(tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    monkeypatch.setattr(L, "_manifest_clean", lambda: {RAWNIND: {"scene": RAWNIND_SCENE, "holdout": ["sceneT"]}})
    base = tmp_path / "raw" / RAWNIND
    base.mkdir(parents=True)

    def entry(scene, cleans, noisy, **flags):
        return {"unknown_sensor": False, "test_reserve": False, **flags,
                "clean_images": [{"filename": f, "sha1": "x"} for f in cleans],
                "noisy_images": [{"filename": f, "sha1": "x"} for f in noisy]}

    a_gt = [f"Bayer_sceneA_GT_ISO100_sha1={k}.arw" for k in ("bb", "aa")]
    a_noisy = ["Bayer_sceneA_ISO6400_sha1=cc.arw", "Bayer_sceneA_ISO800_sha1=dd.arw", "Bayer_sceneA_ISO200_sha1=gone.arw"]
    a_other = ["Bayer_sceneA_GT_ISO100_sha1=00.xmp", "Bayer_sceneA_ISO6400_sha1=01.jpg"]   # listed, on disk, not raws
    t_gt, t_noisy = ["Bayer_TEST_sceneT_GT_ISO100_sha1=ee.cr2"], ["Bayer_TEST_sceneT_ISO400_sha1=ff.cr2"]
    r_gt, r_noisy = ["X-Trans_sceneR_GT_ISO200_sha1=gg.raf"], ["X-Trans_sceneR_ISO3200_sha1=hh.raf"]
    index = {"Bayer": {"sceneA": entry("sceneA", a_gt + a_other[:1], a_noisy + a_other[1:]), "sceneT": entry("sceneT", t_gt, t_noisy, test_reserve=True),
                       "sceneM": entry("sceneM", ["Bayer_sceneM_GT_ISO100_sha1=zz.arw"], ["Bayer_sceneM_ISO800_sha1=yy.arw"])},
             "X-Trans": {"sceneR": entry("sceneR", r_gt, r_noisy, unknown_sensor=True)}}
    (base / "dataset.yaml").write_text(yaml.safe_dump(index), encoding="utf-8")
    (base / "test_reserve.yaml").write_text(yaml.safe_dump({"test_reserve": ["sceneR"]}), encoding="utf-8")
    (base / "permissions.txt").write_text("sceneA: CC0, by A. Author\nsceneT: CC BY 4.0, by T\n", encoding="utf-8")
    for f in a_gt + a_noisy[:2] + a_other + t_gt + t_noisy + r_gt + r_noisy:
        (base / f).write_bytes(b"")
    with pytest.raises(ValueError, match="sceneR") as e:   # the reserve the manifest misses: no pairs at all
        rawnind_pairs(tmp_path)
    assert "sceneT" not in str(e.value)
    monkeypatch.setattr(L, "_manifest_clean", lambda: {RAWNIND: {"scene": RAWNIND_SCENE, "holdout": ["sceneT", "sceneR"]}})
    pairs = rawnind_pairs(tmp_path)
    assert [(p.scene, p.ref_file, p.in_file) for p in pairs] == [
        ("sceneA", a_gt[1], a_gt[0]), ("sceneA", a_gt[1], a_noisy[0]), ("sceneA", a_gt[1], a_noisy[1]),
        ("sceneT", t_gt[0], t_noisy[0]), ("sceneR", r_gt[0], r_noisy[0])]       # sceneM has no GT on disk
    a = pairs[1]
    assert a.source == RAWNIND and a.split in ("train", "val") and a.iso == 6400.0 and a.licence == "CC0, by A. Author"
    assert a.meta == {"cfa": "Bayer", "unknown_sensor": False, "ref_iso": 100.0} and pairs[0].iso == 100.0
    assert pairs[3].split == "test" and pairs[3].licence == "CC BY 4.0, by T"
    assert pairs[4].split == "test" and pairs[4].meta["unknown_sensor"] and pairs[4].licence == ""


def test_hdrplus_pairs_take_the_newest_merge_and_the_reference_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "_manifest_clean", lambda: {HDRPLUS: {"scene": HDRPLUS_SCENE}})
    base = tmp_path / "raw" / HDRPLUS
    for b, frames in {"b1": [0], "b2": [0, 2], "b3": [0], "b4": [0]}.items():
        (base / "bursts" / b).mkdir(parents=True)
        for k in frames:
            (base / "bursts" / b / f"payload_N{k:03d}.dng").write_bytes(b"")
    for r, b, ref in [("results_20161014", "b1", None), ("results_20171023", "b1", "0"), ("results_20171023", "b2", "2"),
                      ("results_20171023", "b3", "1")]:
        (base / r / b).mkdir(parents=True)
        (base / r / b / "merged.dng").write_bytes(b"")
        if ref is not None:
            (base / r / b / "reference_frame.txt").write_text(ref + "\n")
    pairs = hdrplus_pairs(tmp_path)
    assert [(p.scene, p.ref_file, p.in_file) for p in pairs] == [
        ("b1", "results_20171023/b1/merged.dng", "bursts/b1/payload_N000.dng"),
        ("b2", "results_20171023/b2/merged.dng", "bursts/b2/payload_N002.dng"),
        ("b3", "results_20171023/b3/merged.dng", "bursts/b3/payload_N000.dng")]      # b4 has no merge
    assert [p.meta["reference_frame"] for p in pairs] == [0, 2, 1]
    assert [p.meta["input_is_reference_frame"] for p in pairs] == [True, True, False]
    assert all(p.meta["results"] == "results_20171023" and "tags_error" in p.meta and p.licence for p in pairs)
    assert all(p.split == split_of(Item(HDRPLUS, p.ref_file)) and p.split in ("train", "val") for p in pairs)
    assert hdrplus_pairs(tmp_path / "nowhere") == []


def _fake_pairs(rng: np.random.Generator):
    """Four pairs of a fake source: three real ones and one whose input is another scene."""
    imgs, pairs = {}, []
    for k, split in enumerate(["val", "test", "train", "val"]):
        x, y, _ = _pair(rng, ((-1) ** k * (k + 2), 3 - k))
        if k == 3:
            x = _pair(np.random.default_rng(99), (0, 0))[0]
        imgs[f"in{k}.raw"], imgs[f"ref{k}.raw"] = x, y
        pairs.append(Pair("fake", f"s{k}", f"ref{k}.raw", f"in{k}.raw", split, iso=100.0 * (k + 1), licence="CC0"))
    return imgs, pairs


def test_build_set_is_seeded_cached_and_records_the_rejected_pairs(tmp_path):
    imgs, pairs = _fake_pairs(np.random.default_rng(5))
    scales = []

    def decode(path, scale):
        scales.append(scale)
        return imgs[path.name], 14

    logged = []
    rs = build_set("fake", n=6, tile=64, scale=1, seed=0, root=tmp_path, log=logged.append, decode=decode, pairs=pairs)
    assert rs.tiles.shape == rs.targets.shape == (6, 64, 64, 3) and rs.tiles.dtype == rs.targets.dtype == np.float32
    assert rs.captures.shape == (6, 7) and rs.captures.dtype == np.float32 and np.isfinite(rs.captures).all()
    assert rs.scale == 1 and set(scales) == {1} and rs.splits == ("val", "test") and len(rs.pairs) == 3
    assert [p.scene for p in rs.pairs] == ["s0", "s1", "s3"]
    assert {r.pair for r in rs.records} == {0, 1} and len({(r.pair, r.y0, r.x0) for r in rs.records}) == 6
    assert list(rs.rejected) == ["in3.raw"] and rs.rejected["in3.raw"].startswith("ValueError: not aligned")
    for r in rs.records:
        assert r.shift in ((2, 3), (-3, 2)) and np.allclose(r.gain, GAIN, rtol=0.02) and r.corr > 0.95
        assert r.bit_depth == 14 and 0.0 <= r.clipped <= 0.01 and 1600 < r.noise["electrons_at_clip"] < 2400
    assert all(c[2] > 0 and c[-1] == 1.0 for c in rs.captures)
    assert all(rs.tiles[k].max() <= 1.0 and rs.targets[k].max() <= 1.0 for k in range(6))
    p = set_path("fake", 0, 6, 64, 1, tmp_path)
    assert p.exists() and rows_path(p).exists()
    import pyarrow.parquet as pq
    rows = pq.read_table(str(rows_path(p))).to_pylist()
    assert len(rows) == 6 and all(f"m_{m}" in r for r in rows for m in STILL_METRICS)
    assert {r["scene"] for r in rows} == {"s0", "s1"} and all(r["licence"] == "CC0" and r["size"] == 64 for r in rows)
    assert all(r["noise_bins"] >= 3 and r["electrons_at_clip"] > 0 and r["split"] in ("val", "test") for r in rows)
    again = load_set(p)
    assert again.records == rs.records and again.rejected == rs.rejected and again.pairs == rs.pairs
    assert np.array_equal(again.tiles, rs.tiles) and np.array_equal(again.captures, rs.captures)
    cached = build_set("fake", n=6, tile=64, scale=1, seed=0, root=tmp_path, log=None, decode=None, pairs=None)
    assert cached.records == rs.records                   # the cache is the set: no decoder, no listing
    other = build_set("fake", n=6, tile=64, scale=1, seed=1, root=tmp_path, log=None, decode=decode, pairs=pairs)
    assert [(r.pair, r.y0, r.x0) for r in other.records] != [(r.pair, r.y0, r.x0) for r in rs.records]
    train = build_set("fake", n=4, tile=64, scale=1, seed=0, splits=("train",), root=tmp_path, log=logged.append,
                      decode=decode, pairs=pairs)
    assert [p.scene for p in train.pairs] == ["s2"] and len(train.records) == 4 and train.rejected == {}
    text = summarise(rs)
    assert "fake: 6 tiles of 64 px at scale 1 from 2 pairs in 2 scenes" in text and "electrons at clip" in text
    assert "ISO: 100 to 200" in text and "rejected in3.raw" in text
    with pytest.raises(ValueError, match="no pairs in splits"):
        build_set("fake", n=2, tile=64, scale=1, seed=0, splits=("nope",), root=tmp_path, log=None, decode=decode, pairs=pairs)
    with pytest.raises(ValueError, match="0 tile.s. of 2 from 0 usable pair.s. of 1; 1 rejected"):
        build_set("fake", n=2, tile=64, scale=1, seed=0, splits=("val",), root=tmp_path, log=None, decode=decode,
                  pairs=[pairs[3]])


def test_compare_puts_real_rows_against_the_synthetic_ones(tmp_path):
    imgs, pairs = _fake_pairs(np.random.default_rng(6))
    rs = build_set("fake", n=4, tile=64, scale=1, seed=0, root=tmp_path, log=None,
                   decode=lambda p, s: (imgs[p.name], 14), pairs=pairs)
    real = tile_rows(rs)
    rng = np.random.default_rng(0)
    synthetic = [{"p_camera_on": k % 2, **{f"m_{m}": float(rng.random()) for m in STILL_METRICS}} for k in range(40)]
    rep = compare(real, synthetic)
    assert rep["n_real"] == 4 and rep["n_synthetic"] == 40 and rep["n_camera_on"] == 20
    assert set(rep["metrics"]) == set(STILL_METRICS)
    for s in rep["metrics"].values():
        assert len(s["synthetic"]) == 5 and len(s["camera_on"]) == 5
        assert s["real"] is None or len(s["real"]) == 5
        assert s["coverage"] is None or 0.0 <= s["coverage"] <= 1.0
    assert rep["metrics"]["shadow_snr_db"]["real"] is not None
    md = compare_markdown(rep, "fake")
    assert md.startswith("# Real against synthetic: fake") and "| shadow_snr_db |" in md and " %" in md
    empty = compare([], synthetic)
    assert empty["n_real"] == 0 and all(s["real"] is None and s["coverage"] is None for s in empty["metrics"].values())


def test_evaluate_real_scores_the_cached_set_with_the_measured_capture(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from harness.model import apply
    from harness.model.heldout import evaluate, to_markdown
    imgs, pairs = _fake_pairs(np.random.default_rng(7))
    rs = build_set("fake", n=4, tile=64, scale=1, seed=0, root=tmp_path, log=None,
                   decode=lambda p, s: (imgs[p.name], 14), pairs=pairs)
    seen = []

    def fake_restore(b, frames, white_level=1.0, capture=None):
        seen.append(capture)
        return frames

    monkeypatch.setattr(apply, "load_model", lambda ckpt, device=None, bf16=False: SimpleNamespace(step=3))
    monkeypatch.setattr(apply, "restore_frames", fake_restore)
    rep = evaluate("x.pt", source="fake", n=4, tile=64, seed=0, root=tmp_path, log=None, real=True, scale=1)
    assert rep["real"] and rep["preset"] == "real" and rep["scale"] == 1 and rep["n"] == 4 and rep["step"] == 3
    assert len(seen) == 4 and all(np.array_equal(c, rs.captures[k]) for k, c in enumerate(seen))
    assert [r["pair"] for r in rep["records"]] == [r.pair for r in rs.records] and rep["skipped"] == rs.rejected
    for m, s in rep["metrics"].items():   # the identity ties the capture
        assert s["beats"] == 0 and s["loses"] == 0, (m, s)
    md = to_markdown(rep)
    assert md.startswith("# Real-pair evaluation: fake at clean scale 1, 4 tiles of 64 px") and "measured noise law" in md
    assert "beats the real capture on 0 of" in md and "1 pair(s) rejected" in md and "in3.raw" in md
    blind = evaluate("x.pt", blind=True, source="fake", n=4, tile=64, seed=0, root=tmp_path, log=None, real=True, scale=1)
    assert all(c is None for c in seen[4:]) and blind["blind"]


def test_realpairs_and_eval_raw_real_are_wired_into_the_clis():
    from harness.data.cli import add_parser as add_data
    from harness.model.cli import add_parser as add_model
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    add_data(sub)
    add_model(sub)
    a = p.parse_args(["data", "realpairs", "--sources", "rawnind", "--n", "8", "--compare", "--splits", "test"])
    assert a.func.__name__ == "cmd_realpairs" and a.sources == ["rawnind"] and a.n == 8 and a.compare
    assert a.splits == ["test"] and a.tile == 256 and a.scale is None and a.seed == 0 and a.labels is None
    d = p.parse_args(["data", "realpairs"])
    assert d.sources == ["rawnind", "hdr-plus-burst"] and d.splits == ["val", "test"] and not d.compare
    e = p.parse_args(["model", "eval-raw", "--ckpt", "x.pt", "--real", "--source", "rawnind", "--scale", "2"])
    assert e.func.__name__ == "cmd_eval_raw" and e.real and e.source == "rawnind" and e.scale == 2
    f = p.parse_args(["model", "eval-raw", "--ckpt", "x.pt"])
    assert not f.real and f.scale is None
