"""The clean-target audit (harness.data.targetaudit): a picture with sharpening halos reads above one with clean
steps, a flat one has no reading; the audit appends one row per item and resumes, reading again only an item whose
file changed; the filter keeps a source's items at or below its max_halo and the ones too smooth to read, leaves
out the ones over it and the ones not audited (with a warning), and passes a source without max_halo whole; a
TileSource with audit draws the filtered items; and max_halo, a selection setting, does not move the decoded key."""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

import harness.data.decodedstore as S
import harness.data.targetaudit as A
import harness.model.data as D
from harness.data.label import list_items
from harness.model.data import TileSource


def _stripes(halo: bool, w: int = 96, h: int = 64) -> np.ndarray:
    """Vertical stripes 16 px wide in sRGB codes 100 and 200; with `halo` each edge overshoots both ways by 2 px
    (codes 70 and 235), as an unsharp mask leaves it."""
    row = np.where((np.arange(w) // 16) % 2 == 0, 100, 200).astype(np.uint8)
    if halo:
        for e in range(16, w, 16):
            dark_left = row[e - 1] == 100
            row[e - 2:e] = 70 if dark_left else 235
            row[e:e + 2] = 235 if dark_left else 70
    return np.repeat(np.repeat(row[None, :, None], h, 0), 3, 2)


def _source(tmp_path, name="tiny"):
    d = tmp_path / "raw" / name
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_stripes(False)).save(d / "a_clean.png")
    Image.fromarray(_stripes(True)).save(d / "b_halo.png")
    Image.fromarray(np.full((64, 96, 3), 150, np.uint8)).save(d / "c_flat.png")
    return d


def _names(items):
    return sorted(it.file for it in items)


def test_the_reading_separates_a_sharpened_picture_from_a_clean_one():
    lin = lambda u: np.where(u / 255 <= 0.04045, u / 255 / 12.92, ((u / 255 + 0.055) / 1.055) ** 2.4)
    clean, halo = A.halo_reading(lin(_stripes(False))), A.halo_reading(lin(_stripes(True)))
    assert clean == pytest.approx(0.0, abs=1e-6) and halo > 0.05
    assert np.isnan(A.halo_reading(lin(np.full((64, 96, 3), 150.0))))


def test_the_audit_reads_every_item_once_and_resumes(tmp_path):
    d = _source(tmp_path)
    items = list_items("tiny", tmp_path)
    t = A.audit_items(items, root=tmp_path, log=None)
    assert t["items"] == 3 and t["read"] == 3 and t["already"] == 0 and not t["rejected"]
    rows = A.read_audit("tiny", tmp_path)
    assert set(rows) == {(f, -1) for f in _names(items)}
    assert rows[("a_clean.png", -1)]["halo"] == pytest.approx(0.0, abs=1e-6)
    assert rows[("b_halo.png", -1)]["halo"] > 0.05 and rows[("c_flat.png", -1)]["halo"] is None
    again = A.audit_items(items, root=tmp_path, log=None)
    assert again["read"] == 0 and again["already"] == 3
    Image.fromarray(_stripes(True, w=112)).save(d / "a_clean.png")       # the file changed: its key moved
    os.utime(d / "a_clean.png", ns=(1, 1))
    third = A.audit_items(list_items("tiny", tmp_path), root=tmp_path, log=None)
    assert third["read"] == 1 and third["already"] == 2
    assert A.read_audit("tiny", tmp_path)[("a_clean.png", -1)]["halo"] > 0.05   # the last row wins
    assert len(A.audit_path("tiny", tmp_path).read_text(encoding="utf-8").splitlines()) == 4


def test_a_torn_file_is_rejected_with_its_reason_and_not_recorded(tmp_path):
    d = _source(tmp_path)
    (d / "d_torn.png").write_bytes(b"not a picture")
    t = A.audit_items(list_items("tiny", tmp_path), root=tmp_path, log=None)
    assert t["read"] == 3 and list(t["rejected"]) == ["tiny/d_torn.png"] and t["rejected"]["tiny/d_torn.png"]
    assert ("d_torn.png", -1) not in A.read_audit("tiny", tmp_path)


def test_the_filter_keeps_what_its_threshold_allows(tmp_path, monkeypatch):
    d = _source(tmp_path)
    A.audit_items(list_items("tiny", tmp_path), root=tmp_path, log=None)
    Image.fromarray(_stripes(False)).save(d / "e_new.png")               # pulled after the audit
    items = list_items("tiny", tmp_path)
    monkeypatch.setattr(A, "clean_settings", lambda source: {})
    kept, report = A.filter_targets(items, tmp_path)
    assert kept == items and report == {}                               # no max_halo: the source passes whole
    monkeypatch.setattr(A, "clean_settings", lambda source: {"max_halo": 0.04})
    with pytest.warns(RuntimeWarning, match="1 items of tiny have no current reading"):
        kept, report = A.filter_targets(items, tmp_path)
    assert _names(kept) == ["a_clean.png", "c_flat.png"]
    assert report == {"tiny": {"max_halo": 0.04, "kept": 2, "over": 1, "no_reading": 1, "unaudited": 1}}
    monkeypatch.setattr(A, "clean_settings", lambda source: {"max_halo": 1.0})
    A.audit_items(items, root=tmp_path, log=None)
    kept, report = A.filter_targets(items, tmp_path)
    assert kept == items and report["tiny"]["unaudited"] == 0 and report["tiny"]["over"] == 0


def test_a_tile_source_with_the_audit_draws_only_the_kept_items(tmp_path, monkeypatch):
    _source(tmp_path)
    A.audit_items(list_items("tiny", tmp_path), root=tmp_path, log=None)
    monkeypatch.setattr(D, "split_of", lambda it: "train")
    monkeypatch.setattr(A, "clean_settings", lambda source: {"max_halo": 0.04})
    plain = TileSource(["tiny"], "train", 16, root=tmp_path)
    audited = TileSource(["tiny"], "train", 16, root=tmp_path, audit=True)
    assert len(plain.items) == 3 and plain.audit_report == {}
    assert _names(audited.items) == ["a_clean.png", "c_flat.png"]
    assert audited.audit_report["tiny"]["over"] == 1
    rng = np.random.default_rng(0)
    assert {audited.random_tile(rng)[1].file for _ in range(40)} <= {"a_clean.png", "c_flat.png"}


def test_max_halo_does_not_move_the_decoded_key(tmp_path, monkeypatch):
    _source(tmp_path)
    it = list_items("tiny", tmp_path)[0]
    monkeypatch.setattr(S, "clean_settings", lambda source: {"scale": 2})
    k = S.decoded_key(it, tmp_path)
    monkeypatch.setattr(S, "clean_settings", lambda source: {"scale": 2, "max_halo": 0.04})
    assert S.decoded_key(it, tmp_path) == k
    monkeypatch.setattr(S, "clean_settings", lambda source: {"scale": 3, "max_halo": 0.04})
    assert S.decoded_key(it, tmp_path) != k
