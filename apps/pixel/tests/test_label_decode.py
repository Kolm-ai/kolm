"""Labeller inputs beyond 8-bit images: EXR members of a zip, camera raws through LibRaw, 16-bit images, the
skip-with-reason path when a decoder is absent or a file is unreadable, and per-item resume across zip members."""
from __future__ import annotations

import io
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from harness.colorspace import srgb_to_linear
from harness.data import data_root
from harness.data import label as L
from harness.data.label import (CONFIG_KEYS, Item, clean_scale, clean_settings, decode_item, item_key, jpeg_quality,
                                labelled_items, ledger_items, list_items, load_items, read_labels, read_ledger,
                                run_label, scene_of, skip_class, split_member)

DERIVED = data_root() / "derived"
needs_dists = pytest.mark.skipif(not (DERIVED / "camera_presets.json").exists(), reason="derived stats not built")


# ---- fixtures: an EXR in memory, a minimal DNG in memory ------------------------------------------------------

def _exr_bytes(rgb: np.ndarray, grouped: bool = False, half: bool = False) -> bytes:
    """OpenEXR bytes of a float (H, W, 3) image: separate R, G, B channels, or one grouped RGB array."""
    OpenEXR = pytest.importorskip("OpenEXR")
    rgb = rgb.astype(np.float16 if half else np.float32)
    channels = {"RGB": rgb} if grouped else {c: np.ascontiguousarray(rgb[..., i]) for i, c in enumerate("RGB")}
    bio = io.BytesIO()
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}, channels).write(bio)
    return bio.getvalue()


def _dng_bytes(cfa: np.ndarray, neutral=(0.5, 1.0, 0.7)) -> bytes:
    """A minimal valid DNG: one 16-bit uncompressed RGGB strip, identity colour matrix, AsShotNeutral `neutral`
    (so the camera white balance is (1/n_r, 1, 1/n_b)), black 0, white 65535."""
    h, w = cfa.shape
    bo = "<"
    fmt = {1: "B", 3: "H", 4: "I", 5: "I", 10: "i"}

    def pack(typ, val):
        if typ == 2:
            return val.encode("latin-1") + b"\0"
        if typ == 1:
            return bytes(val)
        if typ in (5, 10):
            flat = []
            for v in val:
                flat += [int(round(v * 10000)), 10000]
            return struct.pack(bo + fmt[typ] * len(flat), *flat)
        return struct.pack(bo + fmt[typ] * len(val), *val)

    strip = np.ascontiguousarray(cfa, dtype="<u2").tobytes()
    entries = [(254, 4, [0]), (256, 4, [w]), (257, 4, [h]), (258, 3, [16]), (259, 3, [1]), (262, 3, [32803]),
               (271, 2, "harness"), (272, 2, "synthetic"), (273, 4, [0]), (274, 3, [1]), (277, 3, [1]),
               (278, 4, [h]), (279, 4, [len(strip)]), (284, 3, [1]), (33421, 3, [2, 2]), (33422, 1, [0, 1, 1, 2]),
               (50706, 1, [1, 4, 0, 0]), (50708, 2, "harness synthetic"), (50714, 4, [0]), (50717, 4, [65535]),
               (50721, 10, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]), (50728, 5, list(neutral)), (50778, 3, [21])]
    data_off = 8 + 2 + 12 * len(entries) + 4

    def build(strip_off):
        blobs, out = b"", b""
        for tag, typ, val in sorted(entries):
            payload = pack(typ, [strip_off] if tag == 273 else val)
            count = len(payload) if typ in (1, 2) else len(val)
            if len(payload) <= 4:
                inline = payload + b"\0" * (4 - len(payload))
            else:
                inline = struct.pack(bo + "I", data_off + len(blobs))
                blobs += payload + (b"\0" if len(payload) % 2 else b"")
            out += struct.pack(bo + "HHI", tag, typ, count) + inline
        return struct.pack(bo + "H", len(entries)) + out + struct.pack(bo + "I", 0), blobs

    _, blobs = build(0)
    ifd, blobs = build(data_off + len(blobs))
    return b"II" + struct.pack(bo + "HI", 42, 8) + ifd + blobs + strip


def _source(tmp_path: Path, name: str) -> Path:
    d = tmp_path / "raw" / name
    d.mkdir(parents=True)
    return d


def _run(tmp_path, sources, n, out, **kw):
    kw = {"per_tile": 2, "crops": 2, "tile": 64, "jobs": 1, "chunk": 1, "root": tmp_path, "derived_root": data_root(),
          "log": lambda *_: None, **kw}
    return run_label(sources, n, out=out, **kw)


# ---- listing ---------------------------------------------------------------------------------------------------

def test_list_items_lists_decodable_zip_members_and_keeps_previews_out(tmp_path):
    d = _source(tmp_path, "src")
    with zipfile.ZipFile(d / "scene.zip", "w") as z:
        for name in ("Scene_1K_HDR.exr", "Scene_1K_TONEMAPPED.jpg", "Scene.png", "Scene.blend", "burst/frame.dng",
                     "old.hdr"):
            z.writestr(name, b"x")
    (d / "bad.zip").write_bytes(b"not a zip at all")
    (d / "plain.png").write_bytes(b"x")
    (d / "shot.NEF").write_bytes(b"x")
    (d / "notes.txt").write_bytes(b"x")
    items = list_items("src", tmp_path)
    assert [(it.file, it.is_hdr) for it in items] == [
        ("bad.zip", False), ("plain.png", False), ("scene.zip::Scene_1K_HDR.exr", True),
        ("scene.zip::burst/frame.dng", True), ("scene.zip::old.hdr", True), ("shot.NEF", True)]
    assert split_member("scene.zip::burst/frame.dng") == ("scene.zip", "burst/frame.dng")
    assert split_member("a::b.png") == ("a::b.png", None)      # only an archive path carries a member
    assert len({item_key(it) for it in items}) == len(items)  # every member is its own resume key


def test_a_tar_lists_its_images_only_when_the_source_says_so_and_reads_members_in_place(tmp_path, monkeypatch):
    import tarfile
    from PIL import Image
    d = _source(tmp_path, "uhd")
    big = np.random.default_rng(5).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    blobs = {}
    for fmt, kw in (("PNG", {}), ("JPEG", {"quality": 50}), ("JPEG2000", {"irreversible": False})):
        buf = io.BytesIO()
        Image.fromarray(big).save(buf, fmt, **kw)
        blobs[fmt] = buf.getvalue()
    with tarfile.open(d / "test.tar", "w") as t:
        for name, data in (("test/b.png", blobs["PNG"]), ("test/a.jpg", blobs["JPEG"]), ("test/c.jp2", blobs["JPEG2000"]),
                           ("../escape.png", blobs["PNG"]), ("test/notes.txt", b"x")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    (d / "bad.tar").write_bytes(b"not a tar " * 100)
    assert [it.file for it in list_items("uhd", tmp_path)] == ["bad.tar"]   # an archive's images are its previews
    monkeypatch.setattr(L, "_manifest_clean", lambda: {"uhd": {"archive_images": True, "scale": 2,
                                                               "low_quality_scale": 4}})
    items = list_items("uhd", tmp_path)
    assert [(it.file, it.is_hdr) for it in items] == [
        ("bad.tar", False), ("test.tar::../escape.png", False), ("test.tar::test/a.jpg", False),
        ("test.tar::test/b.png", False), ("test.tar::test/c.jp2", False)]
    assert split_member("test.tar::test/a.jpg") == ("test.tar", "test/a.jpg")
    bad, escape, jpg, png, jp2 = items
    dec = decode_item(png, tmp_path)
    assert dec.domain == "srgb8" and dec.scale == 2
    assert np.allclose(dec.img, L.box_downsample(srgb_to_linear(big / 255.0), 2))
    assert decode_item(jpg, tmp_path).scale == 4               # the member's own quantisation tables set its scale
    dec = decode_item(jp2, tmp_path)                            # lossless JPEG 2000 (UHD-IQA's originals)
    assert dec.domain == "srgb8" and dec.scale == 2
    assert np.allclose(dec.img, L.box_downsample(srgb_to_linear(big / 255.0), 2))
    assert decode_item(escape, tmp_path).img.shape == (20, 30, 3)
    assert not (tmp_path / "raw" / "escape.png").exists()      # members are read, never extracted
    errors = {}
    load_items([bad], tmp_path, errors)
    assert errors[bad].startswith("decode_error:ReadError")


# ---- decoders --------------------------------------------------------------------------------------------------

def test_exr_zip_members_decode_to_clean_linear_radiance(tmp_path):
    pytest.importorskip("OpenEXR")
    rng = np.random.default_rng(1)
    a = np.exp(rng.normal(0.0, 1.5, (24, 40, 3))).astype(np.float32)     # relative radiance, well over 1.0
    a[0, 0, 0], a[0, 1, 1] = np.nan, -3.0                                 # a bad sample and a negative one
    b = np.exp(rng.normal(0.0, 1.0, (16, 20, 3))).astype(np.float32)
    d = _source(tmp_path, "src")
    with zipfile.ZipFile(d / "env.zip", "w") as z:
        z.writestr("Env_HDR.exr", _exr_bytes(a))
        z.writestr("Env_grouped_half.exr", _exr_bytes(b, grouped=True, half=True))
        z.writestr("Env_TONEMAPPED.jpg", b"x")
    items = list_items("src", tmp_path)
    assert [it.file for it in items] == ["env.zip::Env_HDR.exr", "env.zip::Env_grouped_half.exr"]
    dec = decode_item(items[0], tmp_path)
    assert dec.domain == "linear_float_hdr" and dec.is_hdr and dec.img.dtype == np.float64
    assert dec.img.shape == (24, 40, 3) and dec.img[0, 0, 0] == 0.0 and dec.img[0, 1, 1] == 0.0
    good = np.isfinite(a) & (a >= 0)
    assert np.allclose(dec.img[good], a[good]) and dec.img.max() > 1.0
    dec = decode_item(items[1], tmp_path)
    assert dec.img.shape == (16, 20, 3) and np.allclose(dec.img, b, rtol=2e-3)
    assert load_items(items, tmp_path)[items[1]].shape == (16, 20, 3)        # the dict API still serves members


def test_raw_decodes_linear_with_camera_white_balance_and_no_brightening(tmp_path):
    pytest.importorskip("rawpy")
    d = _source(tmp_path, "src")
    # a neutral patch as the camera saw it: each CFA site holds the as-shot neutral times 12000 counts, so only
    # the camera white balance (the reciprocal of that neutral) turns it grey again
    neutral = (0.5, 1.0, 0.7)
    cfa = np.full((64, 80), 12000.0)
    cfa[0::2, 0::2] *= neutral[0]
    cfa[1::2, 1::2] *= neutral[2]
    cfa = cfa.astype(np.uint16)
    (d / "flat.dng").write_bytes(_dng_bytes(cfa, neutral))
    (d / "twice.dng").write_bytes(_dng_bytes(cfa * 2, neutral))
    (d / "unbalanced.dng").write_bytes(_dng_bytes(cfa, (1.0, 1.0, 1.0)))
    with zipfile.ZipFile(d / "burst.zip", "w") as z:
        z.writestr("frames/flat.dng", _dng_bytes(cfa, neutral))
    one, two, other, zipped = (decode_item(Item("src", f, is_hdr=True), tmp_path)
                               for f in ("flat.dng", "twice.dng", "unbalanced.dng", "burst.zip::frames/flat.dng"))
    assert one.domain == "linear16" and one.is_hdr and one.img.shape == (64, 80, 3) and one.img.dtype == np.float64
    centre = one.img[8:-8, 8:-8]
    r, g, b = centre.reshape(-1, 3).mean(axis=0)
    assert abs(g - 12000 / 65535) < 0.01                      # no auto-brightening: 1.0 is the sensor white level
    assert abs(r / g - 1.0) < 0.03 and abs(b / g - 1.0) < 0.03  # grey: the camera white balance was applied
    r2, g2, b2 = other.img[8:-8, 8:-8].reshape(-1, 3).mean(axis=0)
    assert not (abs(r2 / g2 - 1.0) < 0.2 and abs(b2 / g2 - 1.0) < 0.2)   # another neutral: not grey any more
    assert np.allclose(two.img[8:-8, 8:-8], 2.0 * centre, rtol=0.02)  # linear: double the counts, double the value
    assert np.array_equal(zipped.img, one.img)


def test_a_pixel_shift_merge_renders_as_its_camera_renders_the_single_shot(tmp_path, monkeypatch):
    pytest.importorskip("rawpy")
    sio = pytest.importorskip("scipy.io")
    monkeypatch.setattr(L, "_manifest_clean", lambda: {
        "ps": {"include": ["*_rggb.mat"], "companion": "ARW/{name}_1.dng", "scene": "shot_([0-9]+)_"}})
    d = _source(tmp_path, "ps")
    (d / "ARW").mkdir()
    (d / "merged").mkdir()
    # the raw test's neutral patch twice: as the companion's CFA, and as the merge's R, Gr, Gb, B at every site on
    # its 14-bit scale, so only the companion's as-shot white balance turns the merge grey
    neutral = (0.5, 1.0, 0.7)
    cfa = np.full((64, 80), 12000.0)
    cfa[0::2, 0::2] *= neutral[0]
    cfa[1::2, 1::2] *= neutral[2]
    (d / "ARW" / "shot_7_1.dng").write_bytes(_dng_bytes(cfa.astype(np.uint16), neutral))
    site = 12000.0 * L.RGGB_WHITE / 65535 * np.array([neutral[0], 1.0, 1.0, neutral[2]])
    sio.savemat(str(d / "merged" / "shot_7_rggb.mat"), {"rggb": np.tile(site, (60, 80, 1)).round().astype(np.int16)})
    sio.savemat(str(d / "merged" / "bad_rggb.mat"), {"rggb": np.zeros((8, 8, 3), np.int16)})
    assert [it.file for it in list_items("ps", tmp_path)] == ["merged/bad_rggb.mat", "merged/shot_7_rggb.mat"]
    merge = decode_item(Item("ps", "merged/shot_7_rggb.mat", is_hdr=True), tmp_path)
    shot = decode_item(Item("ps", "ARW/shot_7_1.dng", is_hdr=True), tmp_path)
    assert merge.domain == "linear16" and merge.is_hdr and merge.img.dtype == np.float64
    assert merge.img.shape == (60, 78, 3)             # the columns past the raw's visible area (80 - 2) are cut
    assert scene_of(Item("ps", "merged/shot_7_rggb.mat")) == "7"
    r, g, b = merge.img.reshape(-1, 3).mean(axis=0)
    assert abs(g - 12000 / 65535) < 0.003 and abs(r / g - 1.0) < 0.01 and abs(b / g - 1.0) < 0.01
    assert np.allclose(merge.img.reshape(-1, 3).mean(axis=0), shot.img[8:-8, 8:-8].reshape(-1, 3).mean(axis=0),
                       rtol=0.02)
    with pytest.raises(ValueError, match="pixel-shift"):
        decode_item(Item("ps", "merged/bad_rggb.mat", is_hdr=True), tmp_path)


def test_sixteen_bit_png_is_srgb16_and_tiff_falls_back_to_libraw_only_when_small(tmp_path):
    from PIL import Image
    d = _source(tmp_path, "src")
    grey = np.zeros((40, 50), np.uint16)
    grey[:, 25:] = 32768
    grey[0, 0] = 65535
    Image.fromarray(grey).save(d / "deep.png")
    Image.fromarray(np.full((20, 30, 3), 128, np.uint8)).save(d / "small.tif")
    dec = decode_item(Item("src", "deep.png"), tmp_path)
    assert dec.domain == "srgb16" and not dec.is_hdr and dec.img.shape == (40, 50, 3)
    assert dec.img[0, 0, 0] == 1.0 and dec.img[1, 0, 0] == 0.0
    assert abs(dec.img[1, 30, 1] - srgb_to_linear(np.float64(32768 / 65535))) < 1e-12
    dec = decode_item(Item("src", "small.tif"), tmp_path, tile=64)   # LibRaw rejects an RGB TIFF: Pillow's stands
    assert dec.domain == "srgb8" and dec.img.shape == (20, 30, 3)


# ---- clean sources: the manifest's scale and exclusions, the scene key, errors without raising -------------------

def test_clean_scale_exclusions_and_scene_key_follow_the_manifest(tmp_path, monkeypatch):
    from PIL import Image, UnidentifiedImageError
    monkeypatch.setattr(L, "_manifest_clean", lambda: {
        "src": {"scale": 2, "low_quality_scale": 4, "exclude": ["*.tif", "*lens_shading_map_*"]}})
    assert clean_settings("src")["scale"] == 2 and clean_settings("other") == {}
    assert scene_of(Item("src", "1k/a.hdr")) == "a.hdr" and scene_of(Item("src", "b_4k.exr")) == "b.exr"
    assert scene_of(Item("src", "x/y/c.jpg")) == "x/y/c.jpg" and scene_of(Item("src", "k1/d.png")) == "k1/d.png"
    d = _source(tmp_path, "src")
    rng = np.random.default_rng(3)
    big = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
    Image.fromarray(big).save(d / "photo.png")
    Image.fromarray(big).save(d / "fine.jpg", quality=95)
    Image.fromarray(big).save(d / "coarse.jpg", quality=50)
    Image.fromarray(big).save(d / "listed.tif")                    # excluded by pattern
    (d / "broken.png").write_bytes(b"junk")
    with zipfile.ZipFile(d / "pack.zip", "w") as z:
        z.writestr("frames/a.dng", b"x")
        z.writestr("frames/lens_shading_map_N000.dng", b"x")       # excluded as a member too
    items = list_items("src", tmp_path)
    assert [it.file for it in items] == ["broken.png", "coarse.jpg", "fine.jpg", "pack.zip::frames/a.dng",
                                         "photo.png"]
    broken, coarse, fine, member, photo = items
    assert abs(jpeg_quality(d / "fine.jpg") - 95) <= 1 and abs(jpeg_quality(d / "coarse.jpg") - 50) <= 1
    assert jpeg_quality(d / "photo.png") is None
    assert clean_scale(fine, d / "fine.jpg") == 2 and clean_scale(coarse, d / "coarse.jpg") == 4
    assert clean_scale(photo, d / "photo.png") == 2 and clean_scale(Item("other", "x.jpg"), d / "coarse.jpg") == 1
    dec = decode_item(photo, tmp_path)
    assert dec.scale == 2 and dec.domain == "srgb8" and dec.img.shape == (20, 30, 3)
    assert np.allclose(dec.img, L.box_downsample(srgb_to_linear(big / 255.0), 2))
    assert decode_item(coarse, tmp_path).scale == 4 and decode_item(coarse, tmp_path).img.shape == (10, 15, 3)
    errors, scales = {}, {}
    imgs = load_items(items, tmp_path, errors, scales)
    assert set(imgs) == {coarse, fine, photo} and scales == {coarse: 4, fine: 2, photo: 2}
    assert imgs[coarse].shape == (10, 15, 3) and np.array_equal(imgs[photo], dec.img)
    assert errors[broken].startswith("decode_error:UnidentifiedImageError")
    assert errors[member].startswith(("no_decoder:rawpy", "decode_error:"))
    with pytest.raises(UnidentifiedImageError):                    # without `errors` the exception still propagates
        load_items([broken], tmp_path)


@needs_dists
def test_rows_record_the_clean_scale_they_were_measured_at(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(L, "_manifest_clean", lambda: {"src": {"scale": 2}})
    rng = np.random.default_rng(5)
    y, x = np.mgrid[0:192, 0:256] / 128.0
    scene = 128 + 100 * np.sin(5 * x) * np.cos(3 * y) + rng.normal(0, 4, (192, 256))
    img = np.repeat(np.clip(scene, 0, 255).astype(np.uint8)[..., None], 3, axis=-1)
    d = _source(tmp_path, "src")
    Image.fromarray(img).save(d / "photo.png")
    out = tmp_path / "labels"
    s = _run(tmp_path, ["src"], 4, out)
    assert s["rows"] == 4 and s["skipped"] == {}
    rows = read_labels(out)
    assert all(r["scale"] == 2 and r["domain"] == "srgb8" and not r["is_hdr"] for r in rows)
    assert all(r["y0"] + r["size"] <= 96 and r["x0"] + r["size"] <= 128 for r in rows)   # boxes in the halved image


# ---- skips: missing decoder, unreadable file, too small; the ledger and the planner --------------------------------

def test_undecodable_items_are_recorded_with_a_reason_and_only_missing_decoders_are_retried(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setitem(sys.modules, "rawpy", None)       # import rawpy -> ModuleNotFoundError, name "rawpy"
    monkeypatch.setitem(sys.modules, "OpenEXR", None)
    d = _source(tmp_path, "src")
    (d / "shot.dng").write_bytes(_dng_bytes(np.full((64, 80), 9000, np.uint16)))
    (d / "env.exr").write_bytes(b"junk")
    (d / "bad.zip").write_bytes(b"junk")
    (d / "broken.png").write_bytes(b"junk")
    Image.fromarray(np.full((10, 12, 3), 90, np.uint8)).save(d / "tiny.png")
    out = tmp_path / "labels"
    s = _run(tmp_path, ["src"], 100, out, chunk=8, derived_root=None)
    assert s["rows"] == 0 and s["chunks"] == 1 and s["files"] == []
    assert s["skipped"] == {"decode_error:BadZipFile": 1, "decode_error:UnidentifiedImageError": 1,
                            "no_decoder:OpenEXR": 1, "no_decoder:rawpy": 1, "too_small": 1}
    reasons = {e["file"]: e["reason"] for e in s["items_skipped"]}
    assert reasons["tiny.png"] == "too_small:10x12" and reasons["shot.dng"] == "no_decoder:rawpy"
    assert all(e["source"] == "src" for e in s["items_skipped"])
    ledger = read_ledger(out)
    assert list(ledger) == ["src-00000"]
    e = ledger["src-00000"]
    assert e["n_rows"] == 0 and e["n_items"] == 5 and len(e["skipped"]) == 5 and e["file"] == ""  # no shard
    config = dict(zip(CONFIG_KEYS, (64, 2, 2, 1, "parquet")))
    have, skipped = ledger_items(out, config)
    assert have == set() and labelled_items(out, config) == set()
    assert {k[1]: skip_class(r) for k, r in skipped.items()} == {
        "bad.zip": "decode_error:BadZipFile", "broken.png": "decode_error:UnidentifiedImageError",
        "env.exr": "no_decoder:OpenEXR", "shot.dng": "no_decoder:rawpy", "tiny.png": "too_small"}
    assert ledger_items(out, {**config, "tile": 32})[1] == {}   # another configuration owes nothing to these skips
    # the next run leaves the permanent skips alone and tries the two items that only lacked a decoder
    s = _run(tmp_path, ["src"], 100, out, chunk=8, derived_root=None)
    assert s["chunks"] == 1 and s["items_skipped_before"] == 3
    assert s["skipped"] == {"no_decoder:OpenEXR": 1, "no_decoder:rawpy": 1}
    assert list(read_ledger(out)) == ["src-00000", "src-00001"]
    # --retry-skipped puts every skipped item back on the plan
    s = _run(tmp_path, ["src"], 100, out, chunk=8, derived_root=None, retry_skipped=True)
    assert s["items_skipped_before"] == 0 and sum(s["skipped"].values()) == 5
    assert read_labels(out) == []


# ---- real labelling: rows, domains, resume member by member (needs the measured distributions) -----------------

@needs_dists
def test_zip_members_label_and_resume_one_member_at_a_time(tmp_path):
    pytest.importorskip("OpenEXR")

    def env(seed):
        r = np.random.default_rng(seed)
        y, x = np.mgrid[0:96, 0:128] / 96.0
        base = np.stack([0.2 + 0.6 * x, 0.3 + 0.4 * y, 0.5 - 0.2 * x * y], -1)
        base[20:50, 30:70] *= 6.0                                        # a bright patch: HDR range
        return base * np.exp(r.normal(0.0, 0.05, base.shape))

    d = _source(tmp_path, "z")
    with zipfile.ZipFile(d / "env.zip", "w") as z:
        z.writestr("A_HDR.exr", _exr_bytes(env(1)))
        z.writestr("B_HDR.exr", _exr_bytes(env(2), grouped=True, half=True))
        z.writestr("A_TONEMAPPED.jpg", b"x")
    out = tmp_path / "labels"
    config = dict(zip(CONFIG_KEYS, (64, 2, 2, 1, "parquet")))
    s = _run(tmp_path, ["z"], 4, out)                                  # one member: 2 crops x 2 draws
    assert s["rows"] == 4 and s["chunks"] == 1 and s["skipped"] == {}
    assert labelled_items(out, config) == {("z", "env.zip::A_HDR.exr", -1)}
    rows = read_labels(out)
    assert {r["file"] for r in rows} == {"env.zip::A_HDR.exr"}
    assert all(r["domain"] == "linear_float_hdr" and r["is_hdr"] for r in rows)
    s = _run(tmp_path, ["z"], 8, out)                                  # resume: only the second member runs
    assert s["rows"] == 8 and s["rows_new"] == 4 and s["chunks"] == 1 and s["chunks_skipped"] == 1
    assert labelled_items(out, config) == {("z", "env.zip::A_HDR.exr", -1), ("z", "env.zip::B_HDR.exr", -1)}
    assert _run(tmp_path, ["z"], 8, out)["chunks"] == 0
    rows = read_labels(out)
    assert sorted(r["file"] for r in rows) == ["env.zip::A_HDR.exr"] * 4 + ["env.zip::B_HDR.exr"] * 4
    assert all(np.isfinite(r["m_delta_e_mean"]) and r["size"] == 64 for r in rows)
    # the metric vector does not depend on how the members were chunked
    fresh = read_labels(_run(tmp_path, ["z"], 8, tmp_path / "labels2", chunk=2)["out"])
    key = lambda r: (r["file"], r["y0"], r["x0"], r["draw"])  # noqa: E731
    for a, b in zip(sorted(rows, key=key), sorted(fresh, key=key)):
        assert key(a) == key(b)
        for c in a:
            if c.startswith("m_") or c.startswith("p_"):
                assert a[c] == b[c] or (isinstance(a[c], float) and np.isnan(a[c]) and np.isnan(b[c])), (c, a[c], b[c])


@needs_dists
def test_raw_rows_carry_the_linear16_domain(tmp_path):
    pytest.importorskip("rawpy")
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:128, 0:160] / 128.0
    scene = 4000 + 30000 * (0.5 + 0.5 * np.sin(6 * x) * np.cos(4 * y)) + rng.normal(0, 300, (128, 160))
    d = _source(tmp_path, "r")
    (d / "shot.dng").write_bytes(_dng_bytes(np.clip(scene, 0, 65535).astype(np.uint16)))
    out = tmp_path / "labels"
    s = _run(tmp_path, ["r"], 4, out)
    assert s["rows"] == 4 and s["skipped"] == {}
    rows = read_labels(out)
    assert all(r["domain"] == "linear16" and r["is_hdr"] and r["file"] == "shot.dng" for r in rows)
    assert all(np.isfinite(r["m_delta_e_mean"]) and np.isfinite(r["m_mtf50_cpp"]) for r in rows)


# ---- process pool ------------------------------------------------------------------------------------------------

def test_broken_pool_resubmits_only_the_unfinished_chunks(tmp_path, monkeypatch):
    """A worker that dies takes the pool down; the chunks it had not finished go to a fresh pool and nothing
    is labelled twice. The pool and the worker are faked, so this runs without decoders or derived stats."""
    from concurrent.futures import Future
    from concurrent.futures.process import BrokenProcessPool
    from PIL import Image

    from harness.data import label as L

    for name in ("src", "src2"):
        d = _source(tmp_path, name)
        for i in range(4):
            Image.new("RGB", (8, 8), (40 * i, 80, 120)).save(d / f"img{i}.png")
    submitted: list[tuple[int, str]] = []

    def fake_run_chunk(job):
        rows = [{"source": job["source"], "file": it["file"], "row": it["row"], "crop": c, "ms": 1.0}
                for it in job["items"] for c in range(2)]
        return {"chunk": job["chunk"], "rows": rows, "skipped": [], "seconds": 0.01}

    class FlakyPool:
        """First pool: the first chunk completes, every later one is lost with the worker. Later pools work."""
        made = 0

        def __init__(self, max_workers):
            FlakyPool.made += 1
            self.n = FlakyPool.made

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def broken(self):
            return self.n == 1 and len(submitted) > 1

        def submit(self, fn, job):
            submitted.append((self.n, job["chunk"]))
            fut = Future()
            if self.broken():
                fut.set_exception(BrokenProcessPool("a worker died"))
            else:
                fut.set_result(fn(job))
            return fut

    class DeadPool(FlakyPool):
        def broken(self):
            return True

    monkeypatch.setattr(L, "ProcessPoolExecutor", FlakyPool)
    monkeypatch.setattr(L, "run_chunk", fake_run_chunk)
    out = tmp_path / "labels"
    s = _run(tmp_path, ["src"], 100, out, jobs=4)
    chunks = [f"src-{i:05d}" for i in range(4)]
    assert (s["rows_new"], s["chunks"]) == (8, 4)
    assert FlakyPool.made == 2
    assert [c for n, c in submitted if n == 1] == chunks
    assert [c for n, c in submitted if n == 2] == chunks[1:]
    led = read_ledger(out)
    assert sorted(led) == chunks and all(e["n_rows"] == 2 for e in led.values())
    assert len(read_labels(out)) == 8

    # a pool that never recovers gives up after POOL_ATTEMPTS fresh pools, and the lost chunks stay unrecorded
    FlakyPool.made, submitted[:] = 0, []
    monkeypatch.setattr(L, "ProcessPoolExecutor", DeadPool)
    with pytest.raises(BrokenProcessPool, match="still unfinished"):
        _run(tmp_path, ["src2"], 100, tmp_path / "labels2", jobs=4)
    assert FlakyPool.made == L.POOL_ATTEMPTS
    assert not (tmp_path / "labels2" / "ledger.jsonl").exists()
