"""The decoded clean-image store (harness.data.decodedstore) and the sampler reading it: a TileSource with the store
draws the very tiles one without it draws, storing on the first pass and memmapping on the next; the key misses when
the file, the manifest's clean settings or DECODE_VERSION change and leaves parquet rows and missing files out; an
unreadable array is a miss; a failed write turns writing off with one warning and the stream goes on; and the fill
stores every item once, records the files the decoder rejects, and resumes."""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

import harness.data.decodedstore as S
import harness.model.data as D
from harness.data.label import Item, list_items
from harness.model.data import TileSource


def _tiny(tmp_path, n=4, size=48, seed=0, torn=False):
    rng = np.random.default_rng(seed)
    d = tmp_path / "raw" / "tiny"
    d.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        Image.fromarray((rng.random((size + 8 * k, size, 3)) * 255).astype(np.uint8)).save(d / f"t{k}.png")
    if torn:
        (d / "torn.png").write_bytes(b"not a picture")
    return d


def _draws(src, seed, n=24):
    rng = np.random.default_rng(seed)
    return [src.random_tile(rng) for _ in range(n)]


def test_the_stored_stream_is_the_decoded_stream(tmp_path, monkeypatch):
    _tiny(tmp_path)
    monkeypatch.setattr(D, "split_of", lambda it: "train")
    plain = _draws(TileSource(["tiny"], "train", 16, root=tmp_path, tiles_per_image=3), 1)
    first = TileSource(["tiny"], "train", 16, root=tmp_path, tiles_per_image=3, store=True)
    filling = _draws(first, 1)
    assert first.store_hits == 0 and first.store_misses == 4
    assert len(list((tmp_path / "derived" / S.DECODED_DIR / "tiny").glob("*.npy"))) == 4
    again = TileSource(["tiny"], "train", 16, root=tmp_path, tiles_per_image=3, store=True)
    stored = _draws(again, 1)
    assert again.store_misses == 0 and again.store_hits == 4
    assert isinstance(again.image(again.items[0]), np.memmap)
    for a, b, c in zip(plain, filling, stored):
        assert a[1:] == b[1:] == c[1:]                       # item and crop
        assert a[0].dtype == c[0].dtype == np.float64
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[0], c[0])
    it = again.items[0]
    np.testing.assert_array_equal(np.asarray(again.image(it)), S.decode_clean(it, tmp_path))
    assert again.image(it).dtype == np.float16                    # an 8-bit picture: float16 holds it


def test_the_key_follows_the_file_the_settings_and_the_version(tmp_path, monkeypatch):
    d = _tiny(tmp_path, n=1)
    it = list_items("tiny", tmp_path)[0]
    k0 = S.decoded_key(it, tmp_path)
    assert k0 and S.decoded_key(it, tmp_path) == k0
    assert S.decoded_path(it, tmp_path) == tmp_path / "derived" / "decoded" / "tiny" / f"{k0}.npy"
    monkeypatch.setattr(S, "clean_settings", lambda source: {"scale": 2})
    assert S.decoded_key(it, tmp_path) != k0
    monkeypatch.setattr(S, "clean_settings", lambda source: {"max_halo": 0.1, "holdout": ["t*"]})
    assert S.decoded_key(it, tmp_path) == k0                       # what chooses items is not how one decodes
    monkeypatch.undo()
    monkeypatch.setattr(S, "DECODE_VERSION", S.DECODE_VERSION + 1)
    assert S.decoded_key(it, tmp_path) != k0
    monkeypatch.undo()
    st = (d / "t0.png").stat()
    os.utime(d / "t0.png", ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    assert S.decoded_key(it, tmp_path) != k0
    assert S.decoded_key(Item("tiny", "t0.parquet", row=3, column="image"), tmp_path) is None
    assert S.decoded_key(Item("tiny", "gone.png"), tmp_path) is None
    assert S.read_decoded(None) is None and S.read_decoded(tmp_path / "absent.npy") is None


def test_an_unreadable_array_is_a_miss_and_is_stored_again(tmp_path, monkeypatch):
    _tiny(tmp_path, n=1)
    monkeypatch.setattr(D, "split_of", lambda it: "train")
    src = TileSource(["tiny"], "train", 16, root=tmp_path, store=True)
    p = S.decoded_path(src.items[0], tmp_path)
    p.parent.mkdir(parents=True)
    p.write_bytes(b"half an array")
    assert S.read_decoded(p) is None
    np.save(p.with_name("flat.npy"), np.zeros((4, 4), dtype=np.float32))
    assert S.read_decoded(p.with_name("flat.npy")) is None           # not an (H, W, 3) float16 or float32 image
    img = src.image(src.items[0])
    assert src.store_misses == 1 and S.read_decoded(p) is not None
    np.testing.assert_array_equal(S.read_decoded(p), img)
    assert not list(p.parent.glob("*.tmp"))


def test_a_failed_write_turns_writing_off_and_the_stream_goes_on(tmp_path, monkeypatch):
    _tiny(tmp_path, n=3)
    monkeypatch.setattr(D, "split_of", lambda it: "train")
    calls = []
    monkeypatch.setattr(D, "write_decoded", lambda p, img: calls.append(p) or False)
    src = TileSource(["tiny"], "train", 16, root=tmp_path, store=True)
    rng = np.random.default_rng(0)
    with pytest.warns(RuntimeWarning, match="decodes without storing"):
        src.random_tile(rng)
    for it in src.items:
        src.image(it)
    assert len(calls) == 1 and src.store_misses == 3 and not src._store_writes
    assert S.write_decoded(tmp_path / "raw" / "tiny" / "t0.png" / "under_a_file.npy", np.zeros((2, 2, 3))) is False


@pytest.mark.parametrize("workers", [1, 2])
def test_the_fill_stores_each_item_once_records_rejects_and_resumes(tmp_path, workers):
    _tiny(tmp_path, n=3, torn=True)
    items = list_items("tiny", tmp_path) + [Item("tiny", "t0.parquet", row=0, column="image")]
    tally = S.fill_store(items, tmp_path, workers=workers, log=None)
    assert tally["stored"] == 3 and tally["already"] == 0 and tally["left_out"] == 1
    assert list(tally["rejected"]) == ["tiny/torn.png"] and tally["rejected"]["tiny/torn.png"].startswith("decode_error:")
    assert tally["bytes"] == sum(S.decode_clean(it, tmp_path).nbytes // 2 for it in items[:4] if it.file != "torn.png")
    for it in items:
        if it.row < 0 and it.file != "torn.png":
            np.testing.assert_array_equal(S.read_decoded(S.decoded_path(it, tmp_path)), S.decode_clean(it, tmp_path))
    again = S.fill_store(items, tmp_path, workers=workers, log=None)
    assert again["stored"] == 0 and again["already"] == 3 and list(again["rejected"]) == ["tiny/torn.png"]


def test_the_store_and_its_fill_stop_at_the_free_space_floor(tmp_path, monkeypatch):
    p = tmp_path / "derived" / "decoded" / "s" / "k.npy"
    monkeypatch.setenv(S.RESERVE_ENV, "1e9")                      # a floor no drive clears
    assert S.write_decoded(p, np.zeros((2, 2, 3))) is False and not p.exists()
    _tiny(tmp_path, n=2)
    tally = S.fill_store(list_items("tiny", tmp_path), tmp_path, workers=1, log=None)
    assert tally["stored"] == 0 and set(tally["rejected"].values()) == {"store_full"}
    monkeypatch.setenv(S.RESERVE_ENV, "0")
    assert S.write_decoded(p, np.zeros((2, 2, 3))) is True and p.exists()
    assert S.fill_store(list_items("tiny", tmp_path), tmp_path, workers=1, log=None)["stored"] == 2


def test_float16_holds_a_display_referred_image_and_float32_a_bright_one(tmp_path):
    p = tmp_path / "derived" / "decoded" / "s" / "k.npy"
    small = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    assert S.write_decoded(p, small)
    got = S.read_decoded(p)
    assert got.dtype == np.float16 and p.stat().st_size < small.nbytes
    np.testing.assert_array_equal(got, small.astype(np.float16))
    assert np.abs(got - small).max() <= small.max() * 2.0 ** -11      # float16's step
    bright = small * 1e5                                              # an HDRI's sun
    q = p.with_name("b.npy")
    assert S.write_decoded(q, bright) and S.read_decoded(q).dtype == np.float32
    np.testing.assert_array_equal(S.read_decoded(q), bright)
    assert not S.fits_half(np.array([[[np.nan, 0, 0]]], dtype=np.float32)) and S.fits_half(np.zeros((0, 0, 3)))
    np.save(p.with_name("f64.npy"), small.astype(np.float64))
    assert S.read_decoded(p.with_name("f64.npy")) is None
    _tiny(tmp_path, n=1)
    d = S.decode_clean(list_items("tiny", tmp_path)[0], tmp_path)
    assert d.dtype == np.float32 and np.array_equal(d, d.astype(np.float16).astype(np.float32))


@pytest.mark.parametrize("workers", [1, 2])
def test_the_compaction_rewrites_a_float32_store_and_resumes(tmp_path, workers):
    _tiny(tmp_path, n=3)
    d = tmp_path / "derived" / S.DECODED_DIR / "tiny"
    d.mkdir(parents=True)
    rng = np.random.default_rng(1)
    olds = {}
    for it in list_items("tiny", tmp_path):                          # a store written before 2026-10-29: float32
        p = S.decoded_path(it, tmp_path)
        olds[p] = rng.random((20, 24, 3)).astype(np.float32)
        np.save(p, olds[p])
    np.save(d / "bright.npy", np.full((4, 4, 3), 1e6, dtype=np.float32))
    np.save(d / "flat.npy", np.zeros((4, 4), dtype=np.float32))
    (d / "half.npy").write_bytes(b"half an array")
    (d / "x.123.tmp").write_bytes(b"a writer's part")
    logged = []
    t = S.compact_store(tmp_path, ["tiny"], workers=workers, log=logged.append)
    assert t["files"] == 6 and t["compacted"] == 3 and t["kept"] == 1 and t["other"] == 2 and t["busy"] == 0
    assert t["freed"] == sum(a.nbytes // 2 for a in olds.values()) and any("3 compacted" in s for s in logged)
    for p, a in olds.items():
        got = S.read_decoded(p)
        assert got.dtype == np.float16
        np.testing.assert_array_equal(got, a.astype(np.float16))
    assert S.read_decoded(d / "bright.npy").dtype == np.float32
    assert sorted(f.name for f in d.glob("*.tmp")) == ["x.123.tmp"]
    again = S.compact_store(tmp_path, None, workers=workers, log=None)
    assert again["already"] == 3 and again["compacted"] == 0 and again["kept"] == 1


@pytest.mark.skipif(os.name != "nt", reason="only Windows refuses to replace a mapped file")
def test_a_mapped_array_is_left_for_a_later_pass(tmp_path):
    p = tmp_path / "derived" / S.DECODED_DIR / "s" / "k.npy"
    p.parent.mkdir(parents=True)
    np.save(p, np.full((8, 8, 3), 0.25, dtype=np.float32))
    m = S.read_decoded(p)
    t = S.compact_store(tmp_path, ["s"], workers=1, log=None)
    assert t["busy"] == 1 and float(m[0, 0, 0]) == 0.25 and not list(p.parent.glob("*.tmp"))
    del m
    assert S.compact_store(tmp_path, ["s"], workers=1, log=None)["compacted"] == 1


def test_compact_is_wired_into_the_cli():
    import argparse
    from harness.data.cli import add_parser
    ap = argparse.ArgumentParser()
    add_parser(ap.add_subparsers(dest="cmd"))
    a = ap.parse_args(["data", "decoded", "--sources", "a", "b", "--compact", "--workers", "3"])
    assert a.compact and a.sources == ["a", "b"] and a.workers == 3
    assert not ap.parse_args(["data", "decoded", "--sources", "a"]).compact


def test_the_prune_removes_what_no_current_item_keys(tmp_path):
    _tiny(tmp_path, n=3)
    items = list_items("tiny", tmp_path)
    assert S.fill_store(items, tmp_path, workers=1, log=None)["stored"] == 3
    d = tmp_path / "derived" / S.DECODED_DIR / "tiny"
    stale = d / ("0" * 32 + ".npy")
    np.save(stale, np.zeros((4, 4, 3), np.float16))
    (d / "k.123.tmp").write_bytes(b"in flight")
    other = tmp_path / "derived" / S.DECODED_DIR / "gone"
    other.mkdir()
    np.save(other / "x.npy", np.zeros((4, 4, 3), np.float16))
    keep = {"tiny": {S.decoded_key(it, tmp_path) for it in items}, "gone": set()}
    t = S.prune_store(keep, tmp_path, log=None)
    assert t["pruned"] == 1 and t["kept"] == 3 and t["busy"] == 0 and t["freed"] > 0 and t["skipped"] == ["gone"]
    assert not stale.exists() and (d / "k.123.tmp").exists() and (other / "x.npy").exists()
    assert all(S.read_decoded(S.decoded_path(it, tmp_path)) is not None for it in items)
    assert S.prune_store(keep, tmp_path, log=None)["pruned"] == 0


def test_prune_is_wired_into_the_cli(tmp_path, monkeypatch):
    import argparse
    from harness.data.cli import add_parser
    _tiny(tmp_path, n=2)
    monkeypatch.setenv("HARNESS_DATA_ROOT", str(tmp_path))
    items = list_items("tiny", tmp_path)
    S.fill_store(items, tmp_path, workers=1, log=None)
    stale = tmp_path / "derived" / S.DECODED_DIR / "tiny" / ("f" * 32 + ".npy")
    np.save(stale, np.zeros((4, 4, 3), np.float16))
    ap = argparse.ArgumentParser()
    add_parser(ap.add_subparsers(dest="cmd"))
    a = ap.parse_args(["data", "decoded", "--sources", "tiny", "--prune"])
    assert a.prune and not a.compact and not ap.parse_args(["data", "decoded", "--sources", "a"]).prune
    assert a.func(a) == 0 and not stale.exists()
    assert all(S.decoded_path(it, tmp_path).exists() for it in items)
