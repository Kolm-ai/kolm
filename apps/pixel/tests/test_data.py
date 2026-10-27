import hashlib
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from pixel.colorsci import display_encode
from pixel.data import budget
from pixel.data.collect import Store, collect
from pixel.data.dedup import dhash, hamming, keep_set, near_duplicates
from pixel.data.shards import list_shards, read_shard, write_shards
from pixel.data.sources import Item, expand, license_ok
from pixel.label import label_shard, read_labels
from pixel.ops import encode_image, save_image
from pixel.scene import Scene, SceneSpec


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    root = tmp_path_factory.mktemp("www")
    files = {}
    for i in range(4):
        img = display_encode(Scene(SceneSpec(seed=50 + i, width=96, height=96)).render(0.0))
        data = encode_image(img, "png")
        (root / f"img{i}.png").write_bytes(data)
        files[f"img{i}.png"] = hashlib.sha256(data).hexdigest()
    (root / "bad.txt").write_bytes(b"not an image")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(root)))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}", files
    srv.shutdown()


def _registry(tmp_path, base, entries):
    p = tmp_path / "registry.json"
    p.write_text(json.dumps({"sources": entries}))
    return p


def test_license_allowlist():
    assert license_ok("CC0") and license_ok("CC BY-SA 4.0") and license_ok("Public domain") and license_ok("cc-by 2.0")
    assert not license_ok("All rights reserved") and not license_ok("") and not license_ok(None) and not license_ok("CC BY-NC 4.0")


def test_collect_is_verified_resumable_and_license_gated(tmp_path, served):
    base, files = served
    reg = _registry(tmp_path, base, [
        {"id": "ok", "kind": "http", "url_template": base + "/img{i}.png", "range": [0, 2], "license": "CC0"},
        {"id": "nope", "kind": "http", "url": base + "/img3.png", "license": "All rights reserved"},
        {"id": "wrong-hash", "kind": "http", "url": base + "/bad.txt", "license": "CC0", "sha256": "00" * 32},
    ])
    store = tmp_path / "store"
    stats = collect(reg, store, workers=2)
    assert stats.downloaded == 3 and stats.skipped_license == 1 and stats.failed == 1
    assert any("sha256 mismatch" in e for e in stats.errors)
    st = Store(store)
    rows = list(st.rows())
    assert {r["sha256"] for r in rows} == {files[f"img{i}.png"] for i in range(3)}
    assert all(st.path_for(r["sha256"], r["ext"]).exists() for r in rows)
    assert not list((store / "tmp").glob("*.part"))
    again = collect(reg, store, workers=2)
    assert again.downloaded == 0 and again.skipped_existing == 3
    v = st.verify()
    assert v["ok"] == 3 and not v["missing"] and not v["corrupt"]
    dry = collect(reg, tmp_path / "store2", dry_run=True)
    assert dry.downloaded == 4 and dry.skipped_license == 1  # dry runs list, they cannot verify hashes
    assert not any(p.is_dir() and len(p.name) == 2 for p in (tmp_path / "store2").iterdir())
    capped = collect(reg, tmp_path / "store3", max_items=1)
    assert capped.downloaded == 1


def test_shards_dedup_and_label(tmp_path, served):
    base, files = served
    reg = _registry(tmp_path, base, [{"id": "ok", "kind": "http", "url_template": base + "/img{i}.png", "range": [0, 3], "license": "CC0"}])
    store = tmp_path / "store"
    collect(reg, store, workers=2)
    paths = write_shards(Store(store), tmp_path / "shards", shard_bytes=40_000)
    assert len(paths) >= 1 and list_shards(tmp_path / "shards") == paths
    samples = [s for p in paths for s in read_shard(p)]
    assert len(samples) == 4 and all(s[3]["license"] == "CC0" for s in samples)

    hashes = {k: dhash(d) for k, _e, d, _m in samples}
    assert dhash(samples[0][2]) == hashes[samples[0][0]]  # same bytes, same hash
    rng = np.random.default_rng(0)
    synth = {f"k{i}": int(rng.integers(0, 2**63)) for i in range(200)}
    synth["copy"] = synth["k7"] ^ 0b101  # two bits away from k7
    dupes = near_duplicates(synth, max_distance=3)
    assert ("copy", "k7", 2) in dupes and hamming(synth["copy"], synth["k7"]) == 2
    assert all(d <= 3 for _a, _b, d in dupes)
    assert len(keep_set(list(synth), dupes)) == len(synth) - len(dupes)

    out = tmp_path / "labels" / "s0.jsonl"
    n = label_shard(paths[0], out, recipes_per_source=2, size=64)
    rows = list(read_labels(out))
    assert n == len(rows) and n > 0 and n % 2 == 0
    r = rows[0]
    assert len(r["direction"]) == 12 and set(r["metrics"]) >= {"psnr", "mtf50", "noise_sigma", "blocking_score"}
    assert r["seed"] == r["seed"] and r["recipe"]
    # deterministic: relabelling gives byte-identical rows
    out2 = tmp_path / "labels" / "s0b.jsonl"
    label_shard(paths[0], out2, recipes_per_source=2, size=64)
    assert [json.dumps(x) for x in read_labels(out)] == [json.dumps(x) for x in read_labels(out2)]


def test_expanders_parse_published_hashes(monkeypatch):
    class FakeSession:
        def get(self, url, **kw):
            class R:
                def __init__(self, payload):
                    self.payload = payload

                def raise_for_status(self):
                    pass

                def json(self):
                    return self.payload

            if "huggingface.co/api" in url:
                return R([{"type": "file", "path": "data/a.parquet", "size": 10, "lfs": {"oid": "ab" * 32}}, {"type": "directory", "path": "x"}])
            if "zenodo.org" in url:
                return R({"doi": "10.5281/zenodo.1", "metadata": {"license": {"id": "cc-by-4.0"}},
                          "files": [{"key": "f.tif", "size": 5, "checksum": "md5:" + "cd" * 16, "links": {"self": "https://z/f.tif"}}]})
            if "polyhaven.com/assets" in url:
                return R({"sky": {}})
            if "polyhaven.com/files" in url:
                return R({"hdri": {"4k": {"hdr": {"url": "https://p/sky_4k.hdr", "size": 7, "md5": "ef" * 16}}}})
            raise AssertionError(url)

    s = FakeSession()
    hf = list(expand({"id": "h", "kind": "hf_tree", "repo": "o/d", "path": "data", "license": "CC BY 4.0"}, s))
    assert len(hf) == 1 and hf[0].sha256 == "ab" * 32 and hf[0].url.endswith("/resolve/main/data/a.parquet")
    z = list(expand({"id": "z", "kind": "zenodo", "record": "1"}, s))
    assert z[0].md5 == "cd" * 16 and z[0].license == "cc-by-4.0" and z[0].provenance["doi"] == "10.5281/zenodo.1"
    ph = list(expand({"id": "p", "kind": "polyhaven_hdris"}, s))
    assert ph[0].md5 == "ef" * 16 and ph[0].license == "CC0"
    with pytest.raises(KeyError):
        list(expand({"id": "x", "kind": "nope"}, s))


def test_budget_never_stores_degraded_pairs():
    p = budget.plan(budget.BudgetInputs(disk_tb=4.0))
    assert p["degraded_pairs_stored"] == 0
    assert 1_000_000 < p["max_sources"] < 3_000_000
    assert p["label_bytes"] < 0.05 * p["usable_bytes"]
    assert "regenerated" in p["note"]
    assert "max LDR sources" in budget.render_plan(p)
