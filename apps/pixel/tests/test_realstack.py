"""Exposure stacks as real pairs (harness.data.realpairs and realstore, SI-HDR): fake tripod stacks whose frames step
by RATIO in exposure and settle by a pixel or two, each frame carrying the noise law ELECTRONS and READ, decoded by
an injected decoder. The links' shifts, gains and offsets; the stack's noise law, and the link a moving scene spoils
left out of it; the merged reference's noise, its independence from the input, its ghosts and its holes; the pairs
a stack refuses (a link that does not hold, no law, a reference that is mostly hole); the stack listing from a fake
raw.zip; the pair scale; and the store's stack pairs, a <key>.ref.npy per input, whose holes a tile fills with the
input and counts as clipped. No LibRaw."""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from harness.data import dng
from harness.data.label import Item, clean_settings
from harness.data.manifest import by_id, load_manifest
from harness.data.realpairs import (CLIP, MAX_TILE_CLIPPED, RAWNIND, SIHDR, SIHDR_STACKS, STACK_LAW_TOL,
                                    STACK_MAX_HOLE, Pair, capture_of, decode_stack, match_of, pair_scale,
                                    sihdr_pairs, stack_reference, stack_run)
from harness.data.realstore import RealStore, build_store, key_of, pair_json, scene_dir
from harness.model.data import split_of

from test_realstore import _decoder

ELECTRONS, READ = 30000.0, 8.0      # a full-frame sensor at base ISO: SI-HDR's stacks measure 25k to 50k at the clip
RATIO = 8.0                         # 3 EV between frames, as SI-HDR's
OFFSETS = [(0, 0), (1, 0), (1, -2), (2, -1)]   # frame j at p + OFFSETS[j] ~ frame 0 at p: a tripod's settle
PAD = 8


def _scene(rng: np.random.Generator, lo: float = 0.0005, hi: float = 0.9, h: int = 384, w: int = 512,
           block: int = 32) -> np.ndarray:
    """A grey chart of constant blocks at frame 0's exposure, shuffled: from near black in the shortest frame to
    clipped in all but it."""
    nb = (h // block) * (w // block)
    levels = rng.permutation(np.geomspace(lo, hi, nb)).reshape(h // block, w // block)
    return np.repeat(np.kron(levels, np.ones((block, block)))[..., None], 3, axis=-1)


def _framed(big: np.ndarray, offset, j: int, pad: int = PAD) -> np.ndarray:
    """The chart at frame j's exposure in the framing of a frame at `offset` (frame 0 at p ~ that frame at
    p + offset)."""
    h, w = big.shape[0] - 2 * pad, big.shape[1] - 2 * pad
    oy, ox = offset
    return big[pad - oy:pad - oy + h, pad - ox:pad - ox + w] * RATIO ** j


def _frames(rng: np.random.Generator, big: np.ndarray, n: int = 4, offsets=OFFSETS, edit=None, pad: int = PAD):
    """(frames, clean frames): frame j the chart at RATIO^j frame 0's exposure, framed at its offset, edited by
    edit(j, clean) when given, with the noise law, clipped at 0 and 1."""
    a, b = 1.0 / ELECTRONS, (READ / ELECTRONS) ** 2
    frames, cleans = [], []
    for j in range(n):
        c = _framed(big, offsets[j], j, pad)
        if edit:
            c = edit(j, c)
        cleans.append(c)
        frames.append(np.clip(c + rng.normal(0, 1, c.shape) * np.sqrt(a * c + b), 0.0, 1.0).astype(np.float32))
    return frames, cleans


def _stack(name: str, frames: list[np.ndarray], split: str = "train"):
    """(images by file name, the stack's pairs as sihdr_pairs lists them: each frame but the longest an input)."""
    ref = f"{SIHDR_STACKS}::sihdr/raw/{name}"
    files = [f"{ref}/{name}_{j}.cr2" for j in range(len(frames))]
    meta = {"frames": files, "exposures_s": [RATIO ** j / 1000.0 for j in range(len(frames))]}
    imgs = {f.rsplit("/", 1)[1]: x for f, x in zip(files, frames)}
    pairs = [Pair(SIHDR, name, ref, files[k], split, iso=100.0, licence="CC BY 4.0", meta=dict(meta, frame=k))
             for k in range(len(frames) - 1)]
    return imgs, pairs


def _decoded(tmp_path, name, frames, calls=None):
    imgs, pairs = _stack(name, frames)
    decode = _decoder(imgs, [] if calls is None else calls)
    st = decode_stack(tmp_path, pairs[0].ref_file, pairs[0].meta["frames"], 1, decode, pairs[0].meta["exposures_s"])
    return st, pairs, decode


def test_decode_stack_measures_the_links_and_the_stacks_noise_law(tmp_path):
    frames, _ = _frames(np.random.default_rng(1), _scene(np.random.default_rng(0)))
    calls = []
    st, pairs, decode = _decoded(tmp_path, "901", frames, calls)
    assert [ln["shift"] for ln in st["links"]] == [[1, 0], [0, -2], [1, 1]]
    assert st["offset"] == OFFSETS and all(ln["ok"] and ln["exposure_ratio"] == pytest.approx(RATIO) for ln in st["links"])
    assert all(np.allclose(ln["gain"], RATIO, rtol=0.01) for ln in st["links"])
    assert np.allclose(st["cum"][-1], RATIO ** 3, rtol=0.02) and st["bits"] == 14
    law = st["noise"]
    assert 0.95 * ELECTRONS < law["electrons_at_clip"] < 1.05 * ELECTRONS and 0.8 * READ < law["read_noise_e"] < 1.25 * READ
    assert law["links"] == 3 and law["misfit"] < 0.1 and max(law["levels"]) > 4 * min(law["levels"])
    for ln in st["links"]:
        own = ln["noise"]
        assert own["used"] and own["misfit"] < 0.1 and 0.9 * ELECTRONS < own["electrons_at_clip"] < 1.1 * ELECTRONS
    assert [stack_run(st, k) for k in range(4)] == [[1, 2, 3], [2, 3], [3], []]
    assert len(calls) == 4
    again = decode_stack(tmp_path, pairs[0].ref_file, pairs[0].meta["frames"], 1, decode, pairs[0].meta["exposures_s"])
    assert again is st and len(calls) == 4             # a scene's inputs decode the stack once


def test_a_link_the_scene_moved_on_is_left_out_of_the_law(tmp_path):
    big = _scene(np.random.default_rng(0))
    h, w = big.shape[0] - 2 * PAD, big.shape[1] - 2 * PAD
    field = gaussian_filter(np.random.default_rng(7).normal(size=(h, w)), 1.0)
    field /= field.std()

    def leaves(j, c):                                  # frame 2's left half moved: its texture is in no other frame
        if j == 2:
            c = c.copy()
            c[:, :w // 2] *= np.maximum(1.0 + 0.3 * field[:, :w // 2, None], 0.1)
        return c

    frames, _ = _frames(np.random.default_rng(2), big, edit=leaves)
    st, _, _ = _decoded(tmp_path, "902", frames)
    assert all(ln["ok"] for ln in st["links"])         # it still aligns: the motion is not the framing's
    moved = st["links"][2]["noise"]
    assert not moved["used"] and moved["misfit"] > STACK_LAW_TOL
    assert st["links"][0]["noise"]["used"] and st["links"][1]["noise"]["used"]
    law = st["noise"]
    assert law["links"] == 2 and 0.95 * ELECTRONS < law["electrons_at_clip"] < 1.05 * ELECTRONS


def test_stack_reference_merges_the_longer_frames_and_never_the_input(tmp_path):
    big = _scene(np.random.default_rng(0))
    frames, cleans = _frames(np.random.default_rng(3), big)
    st, _, _ = _decoded(tmp_path, "903", frames)
    inner = np.zeros(frames[0].shape[:2], dtype=bool)
    inner[2 * PAD:-2 * PAD, 2 * PAD:-2 * PAD] = True   # a shifted frame and the ghost test's box leave the border
    a, b = 1.0 / ELECTRONS, (READ / ELECTRONS) ** 2
    for k, run in [(0, [1, 2, 3]), (1, [2, 3]), (2, [3])]:
        y, info = stack_reference(st, k)
        assert y.dtype == np.float32 and info["run"] == run and k not in info["run"]
        assert info["offsets"] == [[OFFSETS[j][0] - OFFSETS[k][0], OFFSETS[j][1] - OFFSETS[k][1]] for j in run]
        hole = np.isnan(y).any(axis=-1)
        assert info["hole"] == pytest.approx(hole.mean()) and max(info["ghost"]) < 0.01
        c, held = cleans[k], ~hole
        var_in = a * c + b
        err = ((y - c) ** 2)[held].mean() / var_in[held].mean()
        assert err < 0.25 and abs((y - c)[held].mean()) < 1e-3 * c[held].mean()
        assert np.median((((y - c) ** 2).mean(-1) / var_in.mean(-1))[held]) < 1.2 / RATIO ** len(run) + 0.03
        nx, ny = (frames[k] - c)[held].ravel(), (y - c)[held].ravel()
        assert abs(np.corrcoef(nx, ny)[0, 1]) < 0.02    # the target's noise is not the input's
        nxt = _framed(big, OFFSETS[k], k + 1).max(axis=-1)          # frame k + 1, clean, in frame k's framing
        assert hole[inner & (nxt >= CLIP + 0.03)].mean() > 0.99      # where every longer frame clips there is none
        assert hole[inner & (nxt < 0.8)].mean() < 0.01               # where the next frame holds there is one


def test_a_ghost_is_left_out_of_the_merge(tmp_path):
    big = _scene(np.random.default_rng(0))
    big[PAD + 80:PAD + 176, PAD + 176:PAD + 272] = 0.001          # a dark patch every frame holds but the shortest

    def walker(j, c):                                  # someone in the longest frame only
        if j == 3:
            c = c.copy()
            c[96:160, 192:256] = 0.2
        return c

    frames, cleans = _frames(np.random.default_rng(4), big, edit=walker)
    st, _, _ = _decoded(tmp_path, "904", frames)
    inside = (slice(100, 156), slice(196, 252))
    for k in (0, 1):
        y, info = stack_reference(st, k)
        assert info["ghost"][-1] >= 64 * 64 / (frames[0].shape[0] * frames[0].shape[1])
        assert not np.isnan(y[inside]).any()
        assert np.median(y[inside]) == pytest.approx(0.001 * RATIO ** k, rel=0.05)   # not 0.2 / RATIO^(3 - k)


def test_match_of_makes_a_stack_frame_a_pair_or_says_why_not(tmp_path):
    frames, _ = _frames(np.random.default_rng(5), _scene(np.random.default_rng(0)))
    imgs, pairs = _stack("905", frames)
    m, x, bx, y = match_of(pairs[0], tmp_path, 1, _decoder(imgs, []))
    assert tuple(m["align"]["shift"]) == (0, 0) and m["align"]["aligned"] and np.allclose(m["gain"], 1.0, atol=0.01)
    assert x is frames[0] and bx == 14 and y.shape == x.shape and m["hole"] == pytest.approx(np.isnan(y).any(-1).mean())
    assert 0.95 * ELECTRONS < m["noise"]["electrons_at_clip"] < 1.05 * ELECTRONS       # the stack's law, the input's
    assert 0.8 * ELECTRONS < m["noise_pair"]["electrons_at_clip"] < m["noise"]["electrons_at_clip"]  # + the merge's
    assert m["stack"]["run"] == [1, 2, 3] and len(m["stack"]["links"]) == 3
    assert m["clipped"] >= m["hole"] and not m["valid"][np.isnan(y).any(-1)].any()

    off = [(0, 0), (1, 0), (11, -2), (12, -1)]          # frame 2 is 10 px from frame 1: that link does not hold
    frames, _ = _frames(np.random.default_rng(6), _scene(np.random.default_rng(0), h=416, w=544), offsets=off,
                        pad=16)
    imgs, pairs = _stack("906", frames)
    decode = _decoder(imgs, [])
    with pytest.raises(ValueError, match="frame 1 not chained to a longer one"):
        match_of(pairs[1], tmp_path, 1, decode)
    assert match_of(pairs[0], tmp_path, 1, decode)[0]["stack"]["run"] == [1]
    assert match_of(pairs[2], tmp_path, 1, decode)[0]["stack"]["run"] == [3]

    frames, _ = _frames(np.random.default_rng(7), _scene(np.random.default_rng(0), lo=0.012), n=3)
    imgs, pairs = _stack("907", frames)                # frame 2 clips wherever frame 1 is above 0.12
    with pytest.raises(ValueError, match=r"frame 1's reference holds [0-9.]+% of the frame, under 10%"):
        match_of(pairs[1], tmp_path, 1, _decoder(imgs, []))
    assert STACK_MAX_HOLE == 0.9

    frames, _ = _frames(np.random.default_rng(8), _scene(np.random.default_rng(0), lo=0.01, hi=0.02), n=2)
    imgs, pairs = _stack("908", frames)                # one link over a twofold span of levels: no law to merge with
    with pytest.raises(ValueError, match="stack has no noise law"):
        match_of(pairs[0], tmp_path, 1, _decoder(imgs, []))


def test_sihdr_pairs_order_each_stack_by_its_exposures(tmp_path, monkeypatch):
    monkeypatch.setattr(dng, "read_tags", lambda head: json.loads(head.decode()))
    monkeypatch.setattr(dng, "exposure_summary", lambda tags: tags)
    base = tmp_path / "raw" / SIHDR
    base.mkdir(parents=True)
    tags = lambda t, iso=100: json.dumps({"exposure_s": t, "iso": iso, "f_number": 8.0, "model": "Canon"}).encode()
    with zipfile.ZipFile(base / SIHDR_STACKS, "w") as z:
        for name, t in [("IMG_3.CR2", 0.5), ("IMG_1.CR2", 0.001), ("IMG_2.CR2", 0.008)]:
            z.writestr(f"sihdr/raw/002/{name}", tags(t))
        for name, t in [("IMG_1.CR2", 0.001), ("IMG_2.CR2", 0.008)]:
            z.writestr(f"sihdr/raw/001/{name}", tags(t))
        for name in ("IMG_1.CR2", "IMG_2.CR2"):
            z.writestr(f"sihdr/raw/003/{name}", tags(0.008))           # one exposure twice: no order
        z.writestr("sihdr/raw/004/IMG_1.CR2", b"not tags")
        z.writestr("sihdr/raw/004/IMG_2.CR2", tags(0.5))
        z.writestr("sihdr/raw/005/IMG_1.CR2", tags(0.5))                # one frame: no pair
        z.writestr("sihdr/readme.md", b"")
    logged = []
    pairs = sihdr_pairs(tmp_path, log=logged.append)
    assert [(p.scene, p.in_file, p.meta["frame"]) for p in pairs] == [
        ("001", "raw.zip::sihdr/raw/001/IMG_1.CR2", 0),
        ("002", "raw.zip::sihdr/raw/002/IMG_1.CR2", 0), ("002", "raw.zip::sihdr/raw/002/IMG_2.CR2", 1)]
    p = pairs[2]
    assert p.ref_file == "raw.zip::sihdr/raw/002" and p.source == SIHDR and p.iso == 100.0
    assert p.meta["frames"] == [f"raw.zip::sihdr/raw/002/IMG_{k}.CR2" for k in (1, 2, 3)]
    assert p.meta["exposures_s"] == [0.001, 0.008, 0.5] and p.meta["f_number"] == 8.0 and p.meta["model"] == "Canon"
    assert p.licence == by_id(load_manifest())[SIHDR].license and p.split == split_of(Item(SIHDR, p.ref_file))
    assert pairs[0].split == "test"                    # 001 is in the manifest's holdout
    assert len(logged) == 1 and logged[0].startswith(f"{SIHDR}: 3 stack(s) skipped")
    assert all(f"sihdr/raw/00{k}" in logged[0] for k in (3, 4, 5))
    assert sihdr_pairs(tmp_path / "nowhere") == []


def test_pair_scale_is_the_manifests_pair_scale_or_its_clean_scale():
    assert pair_scale(SIHDR) == 2 and clean_settings(SIHDR)["scale"] == 1
    assert pair_scale(RAWNIND) == int(clean_settings(RAWNIND)["scale"])
    assert pair_scale("no-such-source") == 1


def test_store_keeps_a_reference_per_stack_input_and_fills_its_holes(tmp_path):
    frames, _ = _frames(np.random.default_rng(9), _scene(np.random.default_rng(0)))
    imgs, pairs = _stack("909", frames)
    calls, logged = [], []
    tally = build_store(SIHDR, ("train",), scale=1, root=tmp_path, workers=1, log=logged.append,
                        decode=_decoder(imgs, calls), pairs=pairs)
    assert tally["stored"] == 3 and tally["rejected"] == 0 and len(calls) == 4     # the stack decoded once
    d = scene_dir(SIHDR, "909", tmp_path)
    assert not (d / "ref.npy").exists()
    for k, p in enumerate(pairs):
        meta = json.loads(pair_json(SIHDR, p, tmp_path).read_text())
        assert "rejected" not in meta and meta["ref_npy"] == f"{key_of(p.in_file)}.ref.npy"
        assert (d / meta["ref_npy"]).exists() and (d / f"{key_of(p.in_file)}.npy").exists()
        assert meta["shift"] == [0, 0] and np.allclose(meta["gain"], 1.0, atol=0.01) and meta["ref_bit_depth"] == 14
        assert meta["stack"]["run"] == list(range(k + 1, 4)) and 0.0 < meta["hole"] < STACK_MAX_HOLE
        assert 0.95 * ELECTRONS < meta["noise"]["electrons_at_clip"] < 1.05 * ELECTRONS
        assert meta["capture"] == [float(v) for v in capture_of(meta["noise"], 14)]
        assert meta["noise_pair"]["electrons_at_clip"] < meta["noise"]["electrons_at_clip"]
    store = RealStore([SIHDR], "train", root=tmp_path, tile=64)
    assert len(store) == 3 and store.scenes == 1
    for k in range(3):
        ref = np.load(d / store.pairs[k].meta["ref_npy"]).astype(np.float64)
        inp = np.load(store.pairs[k].npy).astype(np.float64)
        for y0, x0 in [(0, 0), (96, 160), (300, 400)]:
            x, t, clipped = store.tile_at(k, y0, x0)
            ys = ref[y0:y0 + 64, x0:x0 + 64]
            hole = np.isnan(ys).any(-1)
            assert clipped == pytest.approx(np.mean(hole | (np.nan_to_num(ys).max(-1) >= CLIP)))
            assert np.array_equal(t[hole], np.clip(inp[y0:y0 + 64, x0:x0 + 64][hole], 0.0, 1.0))
            assert np.isfinite(t).all() and np.allclose(t[~hole], np.clip(ys[~hole] / store.pairs[k].gain, 0, 1))
    rng = np.random.default_rng(0)
    for _ in range(8):
        tile = store.random_tile(rng)
        assert tile["clipped"] <= MAX_TILE_CLIPPED and np.isfinite(tile["y"]).all()
    calls.clear()
    again = build_store(SIHDR, ("train",), scale=1, root=tmp_path, workers=1, log=None, decode=_decoder(imgs, calls),
                        pairs=pairs)
    assert calls == [] and again["already"] == 3
