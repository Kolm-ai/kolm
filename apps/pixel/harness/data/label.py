"""Offline labels for synthetic degradation: python -m harness data label --sources <ids> --n <count> --out data/labels

Per clean tile (a random crop of a source image) draw parameter vectors with harness.data.degrade, render
each, score it against the prepared clean tile with harness.data.tilemetrics and write one row per draw:
source, file, crop box, seed, the parameter vector (p_*) and the metric vector (m_*). No image is stored:
(source, file, row, crop box, seed) reproduce the pair bit for bit. Rows go to parquet or JSONL shards of
`chunk` items each; the ledger (<out>/ledger.jsonl) records finished chunks so a re-run with the same
configuration skips them. Chunks run in a process pool; every draw is deterministic in its seed, so the
output does not depend on the pool size or on the order chunks finish. An item its decoder rejects (a float
TIFF, a truncated file) is recorded on its chunk's ledger entry under `unreadable` and skipped; a chunk that
fails outright is logged, counted in the summary and left for the next run. Neither stops the pass.
A source's manifest `clean` block says how it is read as a clean source: `scale` box-downsamples each image
in linear light after decoding (a 50 MP JPEG at 2, a demosaiced raw at 4) and every row records the factor
used; `exclude` lists files that are not pictures (lens-shading maps). Splits key on the scene (scene_of), so
two resolution tiers of one HDRI never straddle train and validation."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import data_root
from .degrade import PARAM_NAMES, DegradeParams, Distributions, degrade, prepare_tile, sample_params
from .manifest import load_manifest
from .tilemetrics import tile_metrics
from ..colorspace import srgb_to_linear
from ..metrics import STILL_METRICS
from ..scene import box_downsample

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
RAW_EXT = {".dng", ".nef", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}
HDR_EXT = {".hdr"}
PARQUET_SOURCES = {"hf-kadid10k-images": ("image", "kadid"), "hf-tid2013": ("reference", "tid")}
SEED_MASK = (1 << 63) - 1
LEDGER = "ledger.jsonl"
CONFIG_KEYS = ("tile", "per_tile", "crops", "ss", "fmt")
FAST_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}  # 8-bit LDR formats decoded through the sRGB table
LOW_QUALITY_JPEG = 90    # a JPEG whose quantisation tables put it below this IJG quality takes low_quality_scale
_LUMA_TABLE_SUM = 3688   # sum of the ITU T.81 K.1 luminance table, what the IJG quality scale factor multiplies
_RES_DIR = re.compile("^[0-9]+k$", re.IGNORECASE)      # Poly Haven's resolution tiers: 1k/<name>.hdr, 2k/<name>.hdr
_RES_SUFFIX = re.compile("_[0-9]+k$", re.IGNORECASE)
_SRGB_LUT = srgb_to_linear(np.arange(256) / 255.0)


@dataclass(frozen=True)
class Item:
    """One clean image: a file under <root>/raw/<source>, or one row of a parquet shard there."""
    source: str
    file: str            # path relative to <root>/raw/<source>, posix
    row: int = -1        # parquet row (global over the sorted shards), -1 for a plain file
    column: str = ""
    is_hdr: bool = False


def _seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & SEED_MASK


def scene_of(item: Item) -> str:
    """The scene an item depicts, the key of the train/validation split: its path with resolution directories
    (`1k/`, `2k/`) and resolution stem suffixes (`_4k`) removed, so every resolution tier of one Poly Haven HDRI is
    one scene. Any other path is its own scene."""
    parts = item.file.split("/")
    stem, ext = os.path.splitext(parts[-1])
    kept = [d for d in parts[:-1] if not _RES_DIR.match(d)]
    return "/".join(kept + [_RES_SUFFIX.sub("", stem) + ext])


@lru_cache(maxsize=None)
def _manifest_clean() -> dict[str, dict]:
    return {s.id: dict(s.clean) for s in load_manifest() if s.clean}


def clean_settings(source: str) -> dict:
    """The manifest's `clean` block for a source, empty for one it does not know (a test fixture): `scale`, the
    box-downsampling factor applied in linear light after decoding; `low_quality_scale`, the factor for a JPEG
    whose quantisation tables put it below LOW_QUALITY_JPEG; `exclude`, fnmatch patterns (case-insensitive, on
    the posix path relative to the source) of files that are not pictures."""
    return _manifest_clean().get(source, {})


def list_items(source: str, root: Path | None = None) -> list[Item]:
    """Every clean image of a source on disk, in a fixed order."""
    root = Path(root) if root else data_root()
    base = root / "raw" / source
    if not base.is_dir():
        raise FileNotFoundError(f"{source}: nothing under {base}")
    if source in PARQUET_SOURCES:
        column, kind = PARQUET_SOURCES[source]
        files = sorted(base.rglob("*.parquet"))
        if kind == "kadid":
            from .fit import N_REFS, kadid_index
            rows = [kadid_index(r, 0, 0) for r in range(1, N_REFS + 1)]
        else:
            import pyarrow.parquet as pq
            seen, rows, g = set(), [], 0
            for f in files:
                for rid in pq.read_table(str(f), columns=["reference_id"]).column("reference_id").to_pylist():
                    if rid not in seen:
                        seen.add(rid)
                        rows.append(g)
                    g += 1
        rel = "|".join(f.relative_to(base).as_posix() for f in files)
        return [Item(source, rel, r, column) for r in rows]
    items = []
    excluded = [pat.lower() for pat in clean_settings(source).get("exclude", [])]
    for f in sorted(base.rglob("*")):
        ext = f.suffix.lower()
        if any(fnmatch.fnmatchcase(f.relative_to(base).as_posix().lower(), pat) for pat in excluded):
            continue
        if ext in HDR_EXT:
            items.append(Item(source, f.relative_to(base).as_posix(), is_hdr=True))
        elif ext in IMAGE_EXT:
            items.append(Item(source, f.relative_to(base).as_posix()))
        elif ext in RAW_EXT:
            items.append(Item(source, f.relative_to(base).as_posix(), is_hdr=True))
    return items


def _load_file(it: Item, p: Path) -> np.ndarray:
    if it.file.lower().endswith(tuple(HDR_EXT)):
        from .hdri import read_rgbe
        return read_rgbe(p).astype(np.float64)
    if it.file.lower().endswith(tuple(RAW_EXT)):
        import rawpy
        with rawpy.imread(str(p)) as raw:
            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True)
        return np.asarray(rgb, dtype=np.float64) / 65535.0
    from PIL import Image
    with Image.open(p) as im:
        return srgb_to_linear(np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0)


def _record(errors: dict[Item, str] | None, it: Item, e: Exception) -> None:
    """Keep a decoder's rejection of one file as text, or re-raise when the caller wants exceptions. A missing
    decoder module or exhausted memory is never a property of the file and always propagates."""
    if errors is None or isinstance(e, (ImportError, MemoryError)):
        raise e
    errors[it] = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}".rstrip(": ")


def _load_8bit(it: Item, p: Path) -> np.ndarray | None:
    """8-bit LDR files through a 256-entry sRGB table: _load_file's values to the bit (the same float64 transfer
    curve of the same 256 inputs) in a quarter of its time on a 50 MP JPEG, where that curve over 160 M values was
    the cost. None for any other file, a 16-bit PNG included, which keeps the general decoder."""
    if it.is_hdr or not it.file.lower().endswith(tuple(FAST_EXT)):
        return None
    from PIL import Image
    with Image.open(p) as im:
        if im.mode not in ("1", "L", "LA", "P", "PA", "RGB", "RGBA", "RGBX", "CMYK", "YCbCr"):
            return None
        a = np.asarray(im.convert("RGB"))
    return _SRGB_LUT[a]


def jpeg_quality(p: Path) -> int | None:
    """IJG quality estimated from a JPEG's luminance quantisation table, which the encoder scaled from the standard
    table by (200 - 2q) % at q >= 50 and 5000 / q % below; None for a file that is not a JPEG."""
    from PIL import Image
    with Image.open(p) as im:
        tables = getattr(im, "quantization", None)
        if im.format != "JPEG" or not tables:
            return None
        table = tables.get(0) or next(iter(tables.values()))
    s = 100.0 * sum(table) / _LUMA_TABLE_SUM
    return int(round((200.0 - s) / 2.0 if s <= 100.0 else 5000.0 / s))


def clean_scale(it: Item, p: Path) -> int:
    """The downsampling factor the manifest asks of this item: its source's `scale`, or the `low_quality_scale` for
    a JPEG below LOW_QUALITY_JPEG (at 4 the 8 x 8 blocking of a quality-85 web JPEG is below one pixel)."""
    c = clean_settings(it.source)
    scale = int(c.get("scale", 1) or 1)
    low = int(c.get("low_quality_scale", 0) or 0)
    if low > scale and it.file.lower().endswith((".jpg", ".jpeg")):
        q = jpeg_quality(p)
        if q is not None and q < LOW_QUALITY_JPEG:
            scale = low
    return max(scale, 1)


def load_items(items: list[Item], root: Path | None = None, errors: dict[Item, str] | None = None,
               scales: dict[Item, int] | None = None) -> dict[Item, np.ndarray]:
    """Linear RGB float64 (H, W, 3) per item, box-downsampled by its clean scale (clean_scale; the factor used
    goes to `scales` when given); parquet rows of one shard set are read in a single pass. With `errors` given,
    an item its decoder rejects is left out of the result and recorded there as "ExceptionName: message"
    instead of raising."""
    root = Path(root) if root else data_root()
    out: dict[Item, np.ndarray] = {}
    by_parquet: dict[tuple[str, str, str], list[Item]] = {}
    for it in items:
        if it.row >= 0:
            by_parquet.setdefault((it.source, it.file, it.column), []).append(it)
            continue
        try:
            p = root / "raw" / it.source / it.file
            img = _load_8bit(it, p)
            if img is None:
                img = _load_file(it, p)
            k = clean_scale(it, p)
            out[it] = box_downsample(img, k) if k > 1 else img
            if scales is not None:
                scales[it] = k
        except Exception as e:
            _record(errors, it, e)
    for (source, files, column), its in by_parquet.items():
        from .fit import decode_image, iter_parquet_rows
        paths = [root / "raw" / source / f for f in files.split("|")]
        want = {it.row: it for it in its}
        for g, blob in iter_parquet_rows(paths, set(want), column):
            try:
                out[want[g]] = decode_image(blob["bytes"] if isinstance(blob, dict) else blob)
                if scales is not None:
                    scales[want[g]] = 1
            except Exception as e:
                _record(errors, want[g], e)
    return out


def crop_boxes(item: Item, h: int, w: int, tile: int, crops: int) -> list[tuple[int, int]]:
    """(y0, x0) of `crops` random tile positions, fixed by the item's identity; none if the image is small."""
    if h < tile or w < tile:
        return []
    rng = np.random.default_rng(_seed("crop", item.source, item.file, item.row, tile, crops))
    return [(int(rng.integers(0, h - tile + 1)), int(rng.integers(0, w - tile + 1))) for _ in range(crops)]


def row_seed(item: Item, y0: int, x0: int, tile: int, draw: int) -> int:
    return _seed("draw", item.source, item.file, item.row, y0, x0, tile, draw)


def label_tile(tile_img: np.ndarray, params: DegradeParams, seed: int, ss: int, is_hdr: bool) -> dict[str, float]:
    """Metric vector of one draw: the degraded tile against the prepared clean tile."""
    clean = prepare_tile(tile_img, params, is_hdr)
    return tile_metrics(degrade(tile_img, params, seed, ss, is_hdr), clean)


def label_item(item: Item, img: np.ndarray, dists: Distributions, tile: int, crops: int, per_tile: int,
               ss: int, scale: int = 1) -> list[dict]:
    """Rows of one item; `scale` is the clean scale its image was read at (rows without the column are scale 1)."""
    rows = []
    for y0, x0 in crop_boxes(item, img.shape[0], img.shape[1], tile, crops):
        t = np.ascontiguousarray(img[y0:y0 + tile, x0:x0 + tile])
        for k in range(per_tile):
            seed = row_seed(item, y0, x0, tile, k)
            params = sample_params(np.random.default_rng(seed), dists)
            t0 = time.perf_counter()
            m = label_tile(t, params, seed, ss, item.is_hdr)
            row = {"source": item.source, "file": item.file, "row": item.row, "y0": y0, "x0": x0, "size": tile,
                   "draw": k, "seed": seed, "is_hdr": item.is_hdr, "ss": ss, "scale": scale}
            row.update({f"p_{n}": v for n, v in params.to_dict().items()})
            row.update({f"m_{n}": float(m[n]) for n in STILL_METRICS})
            row["ms"] = (time.perf_counter() - t0) * 1000.0
            rows.append(row)
    return rows


# ---- chunks, ledger, pool -------------------------------------------------------------------------------

def item_key(it: Item) -> tuple[str, str, int]:
    return (it.source, it.file, it.row)


def plan_items(sources: list[str], n: int, per_tile: int, crops: int, root: Path | None,
               done: set[tuple[str, str, int]] | None = None) -> dict[str, list[Item]]:
    """Unlabelled items per source for about n rows in total, the items already labelled (`done`, see
    labelled_items) counting towards n: spread over sources by availability, small sources first. Re-running the
    same command after the corpus grew, with a larger n, therefore labels only the new files."""
    done = done or set()
    listed = {s: list_items(s, root) for s in sources}
    n_done = sum(item_key(it) in done for s in sources for it in listed[s])
    avail = {s: [it for it in listed[s] if item_key(it) not in done] for s in sources}
    need = max(math.ceil(n / max(per_tile * crops, 1)) - n_done, 0)
    chosen: dict[str, list[Item]] = {}
    order = sorted(sources, key=lambda s: len(avail[s]))
    for i, s in enumerate(order):
        take = min(len(avail[s]), math.ceil(need / (len(order) - i)))
        chosen[s] = avail[s][:take]
        need -= take
    return {s: chosen[s] for s in sources if chosen[s]}


def _chunks(items: list[Item], size: int) -> list[list[Item]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


_DISTS: dict[str, Distributions] = {}


def _dists(root: str | None) -> Distributions:
    if root not in _DISTS:
        _DISTS[root] = Distributions.load(root)
    return _DISTS[root]


def run_chunk(job: dict) -> dict:
    """Worker: label every item of one chunk. Pure in its inputs; the parent writes the shard."""
    t0 = time.perf_counter()
    items = [Item(**d) for d in job["items"]]
    dists = _dists(job["derived_root"])
    errors: dict[Item, str] = {}
    scales: dict[Item, int] = {}
    imgs = load_items(items, job["root"], errors, scales)
    rows = []
    for it in items:
        if it in imgs:
            rows.extend(label_item(it, imgs[it], dists, job["tile"], job["crops"], job["per_tile"], job["ss"],
                                   scales.get(it, 1)))
        else:
            errors.setdefault(it, "not in its shard")
    unreadable = [{"file": it.file, "row": it.row, "error": errors[it]} for it in items if it in errors]
    return {"chunk": job["chunk"], "rows": rows, "seconds": time.perf_counter() - t0, "unreadable": unreadable}


def _write_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(rows), str(path))
    else:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: (None if isinstance(v, float) and math.isnan(v) else v)
                                    for k, v in r.items()}) + "\n")


def read_ledger(out: Path) -> dict[str, dict]:
    p = out / LEDGER
    done = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                done[e["chunk"]] = e
    return done


def labelled_items(out: Path, config: dict, ledger: dict[str, dict] | None = None) -> set[tuple[str, str, int]]:
    """(source, file, row) of every item whose rows are on disk under this config: the ledger entries whose config
    matches and whose shard exists, read back by their source/file/row columns only (a deleted shard frees its
    items; a shard labelled under another tile size or draw count does not count). Items a matching entry lists
    as `unreadable` count too: they are not retried until that entry is removed from the ledger."""
    out = Path(out)
    ledger = read_ledger(out) if ledger is None else ledger
    keys: set[tuple[str, str, int]] = set()
    for e in ledger.values():
        if e.get("config") != config:
            continue
        keys.update((e["source"], u["file"], u["row"]) for u in e.get("unreadable", ()))
        p = out / e["file"] if e.get("file") else None
        if p is None or not p.exists():
            continue
        if p.suffix == ".parquet":
            import pyarrow.parquet as pq
            t = pq.read_table(str(p), columns=["source", "file", "row"])
            keys.update(zip(t.column("source").to_pylist(), t.column("file").to_pylist(), t.column("row").to_pylist()))
        else:
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    keys.add((r["source"], r["file"], r["row"]))
    return keys


def read_labels(out: Path) -> list[dict]:
    """Every row of every shard under out, in shard then row order (NaN metrics read back as NaN)."""
    out = Path(out)
    rows = []
    for p in sorted(out.glob("labels-*.parquet")):
        import pyarrow.parquet as pq
        rows.extend(pq.read_table(str(p)).to_pylist())
    for p in sorted(out.glob("labels-*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append({k: (math.nan if v is None and k.startswith("m_") else v) for k, v in r.items()})
    return rows


def run_label(sources: list[str], n: int, out: Path | None = None, per_tile: int = 4, crops: int = 4,
              tile: int = 256, jobs: int | None = None, fmt: str = "parquet", chunk: int = 8, ss: int = 1,
              root: Path | None = None, derived_root: Path | None = None, log=print) -> dict:
    """Label about n rows from the given sources into shards under out; resumable per item (a re-run after the
    corpus grew labels only the new files, into new chunks); returns a summary."""
    root = Path(root) if root else data_root()
    out = Path(out) if out else root / "labels"
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs or os.cpu_count() or 1
    config = dict(zip(CONFIG_KEYS, (tile, per_tile, crops, ss, fmt)))
    ext = ".parquet" if fmt == "parquet" else ".jsonl"
    done = read_ledger(out)
    have = labelled_items(out, config, done)
    kept = [e for e in done.values() if e.get("config") == config
            and (not e.get("file") or (out / e["file"]).exists())]
    skipped, rows_old = len(kept), sum(e["n_rows"] for e in kept)
    next_index: dict[str, int] = {}
    for cid, e in done.items():  # chunk ids never repeat, whether or not their shard survived
        next_index[e["source"]] = max(next_index.get(e["source"], 0), int(cid.rsplit("-", 1)[1]) + 1)
    todo = []
    for source, items in plan_items(sources, n, per_tile, crops, root, done=have).items():
        for i, part in enumerate(_chunks(items, chunk), start=next_index.get(source, 0)):
            cid = f"{source}-{i:05d}"
            todo.append({"chunk": cid, "source": source, "items": [asdict(it) for it in part], "tile": tile,
                         "crops": crops, "per_tile": per_tile, "ss": ss, "root": str(root),
                         "derived_root": str(derived_root) if derived_root else None})
    log(f"label: {len(todo)} chunks to run, {skipped} shards already done ({rows_old} rows, {len(have)} items), "
        f"jobs={jobs}")
    t0 = time.perf_counter()
    rows_new, ms_total, files, unreadable, failed = 0, 0, [], 0, []

    def finish(res: dict, job: dict) -> None:
        nonlocal rows_new, ms_total, unreadable
        path = out / f"labels-{res['chunk']}{ext}"
        if res["rows"]:
            _write_shard(path, res["rows"])
            files.append(path.name)
        bad = res.get("unreadable", [])
        entry = {"chunk": res["chunk"], "source": job["source"], "file": path.name if res["rows"] else "",
                 "n_rows": len(res["rows"]), "n_items": len(job["items"]), "seconds": round(res["seconds"], 3),
                 "done_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config": config,
                 "unreadable": bad}
        with open(out / LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        rows_new += len(res["rows"])
        ms_total += sum(r["ms"] for r in res["rows"])
        unreadable += len(bad)
        note = f", {len(bad)} unreadable ({bad[0]['file']}: {bad[0]['error']})" if bad else ""
        log(f"  {res['chunk']}: {len(res['rows'])} rows in {res['seconds']:.1f} s{note}")

    def fail(job: dict, e: BaseException) -> None:
        failed.append({"chunk": job["chunk"], "source": job["source"], "n_items": len(job["items"]),
                       "error": f"{type(e).__name__}: {e}"})
        log(f"  {job['chunk']}: FAILED {failed[-1]['error']}")

    if jobs == 1 or len(todo) <= 1:
        for job in todo:
            try:
                finish(run_chunk(job), job)
            except Exception as e:
                fail(job, e)
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(todo))) as ex:
            pending = {ex.submit(run_chunk, job): job for job in todo}
            for fut in as_completed(pending):
                try:
                    finish(fut.result(), pending[fut])
                except BrokenProcessPool as e:
                    fail(pending[fut], e)
                    log("label: worker pool broken, the remaining chunks wait for the next run")
                    break
                except Exception as e:
                    fail(pending[fut], e)
    seconds = time.perf_counter() - t0
    summary = {"rows": rows_old + rows_new, "rows_new": rows_new, "chunks": len(todo), "chunks_skipped": skipped,
               "chunks_failed": len(failed), "failed": failed, "items_unreadable": unreadable,
               "items_done_before": len(have),
               "seconds": round(seconds, 2), "rows_per_s_per_proc": (rows_new / (ms_total / 1000.0)) if ms_total else 0.0,
               "out": str(out), "files": files}
    log(f"label: {rows_new} new rows in {seconds:.1f} s, {summary['rows_per_s_per_proc']:.2f} rows/s per process, "
        f"{unreadable} unreadable items, {len(failed)} failed chunks")
    return summary
