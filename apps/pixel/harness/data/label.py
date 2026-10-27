"""Offline labels for synthetic degradation: python -m harness data label --sources <ids> --n <count> --out data/labels

Per clean tile (a random crop of a source image) draw parameter vectors with harness.data.degrade, render
each, score it against the prepared clean tile with harness.data.tilemetrics and write one row per draw:
source, file, crop box, seed, the parameter vector (p_*) and the metric vector (m_*). No image is stored:
(source, file, row, crop box, seed) reproduce the pair bit for bit. Rows go to parquet or JSONL shards of
`chunk` items each; the ledger (<out>/ledger.jsonl) records finished chunks so a re-run with the same
configuration skips them. Chunks run in a process pool; every draw is deterministic in its seed, so the
output does not depend on the pool size or on the order chunks finish."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import data_root
from .degrade import PARAM_NAMES, DegradeParams, Distributions, degrade, prepare_tile, sample_params
from .tilemetrics import tile_metrics
from ..colorspace import srgb_to_linear
from ..metrics import STILL_METRICS

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
RAW_EXT = {".dng", ".nef", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}
HDR_EXT = {".hdr"}
PARQUET_SOURCES = {"hf-kadid10k-images": ("image", "kadid"), "hf-tid2013": ("reference", "tid")}
SEED_MASK = (1 << 63) - 1
LEDGER = "ledger.jsonl"
CONFIG_KEYS = ("tile", "per_tile", "crops", "ss", "fmt")


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
    for f in sorted(base.rglob("*")):
        ext = f.suffix.lower()
        if ext in HDR_EXT:
            items.append(Item(source, f.relative_to(base).as_posix(), is_hdr=True))
        elif ext in IMAGE_EXT:
            items.append(Item(source, f.relative_to(base).as_posix()))
        elif ext in RAW_EXT:
            items.append(Item(source, f.relative_to(base).as_posix(), is_hdr=True))
    return items


def load_items(items: list[Item], root: Path | None = None) -> dict[Item, np.ndarray]:
    """Linear RGB float64 (H, W, 3) per item; parquet rows of one shard set are read in a single pass."""
    root = Path(root) if root else data_root()
    out: dict[Item, np.ndarray] = {}
    by_parquet: dict[tuple[str, str, str], list[Item]] = {}
    for it in items:
        p = root / "raw" / it.source / it.file
        if it.row >= 0:
            by_parquet.setdefault((it.source, it.file, it.column), []).append(it)
        elif it.file.lower().endswith(tuple(HDR_EXT)):
            from .hdri import read_rgbe
            out[it] = read_rgbe(p).astype(np.float64)
        elif it.file.lower().endswith(tuple(RAW_EXT)):
            import rawpy
            with rawpy.imread(str(p)) as raw:
                rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True)
            out[it] = np.asarray(rgb, dtype=np.float64) / 65535.0
        else:
            from PIL import Image
            with Image.open(p) as im:
                out[it] = srgb_to_linear(np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0)
    for (source, files, column), its in by_parquet.items():
        from .fit import decode_image, iter_parquet_rows
        paths = [root / "raw" / source / f for f in files.split("|")]
        want = {it.row: it for it in its}
        for g, blob in iter_parquet_rows(paths, set(want), column):
            out[want[g]] = decode_image(blob["bytes"] if isinstance(blob, dict) else blob)
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
               ss: int) -> list[dict]:
    rows = []
    for y0, x0 in crop_boxes(item, img.shape[0], img.shape[1], tile, crops):
        t = np.ascontiguousarray(img[y0:y0 + tile, x0:x0 + tile])
        for k in range(per_tile):
            seed = row_seed(item, y0, x0, tile, k)
            params = sample_params(np.random.default_rng(seed), dists)
            t0 = time.perf_counter()
            m = label_tile(t, params, seed, ss, item.is_hdr)
            row = {"source": item.source, "file": item.file, "row": item.row, "y0": y0, "x0": x0, "size": tile,
                   "draw": k, "seed": seed, "is_hdr": item.is_hdr, "ss": ss}
            row.update({f"p_{n}": v for n, v in params.to_dict().items()})
            row.update({f"m_{n}": float(m[n]) for n in STILL_METRICS})
            row["ms"] = (time.perf_counter() - t0) * 1000.0
            rows.append(row)
    return rows


# ---- chunks, ledger, pool -------------------------------------------------------------------------------

def plan_items(sources: list[str], n: int, per_tile: int, crops: int, root: Path | None) -> dict[str, list[Item]]:
    """Items per source for about n rows: spread over sources by availability, small sources first."""
    avail = {s: list_items(s, root) for s in sources}
    need = math.ceil(n / max(per_tile * crops, 1))
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
    imgs = load_items(items, job["root"])
    rows = []
    for it in items:
        rows.extend(label_item(it, imgs[it], dists, job["tile"], job["crops"], job["per_tile"], job["ss"]))
    return {"chunk": job["chunk"], "rows": rows, "seconds": time.perf_counter() - t0}


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
    """Label about n draws from the given sources into shards under out; resumable; returns a summary."""
    root = Path(root) if root else data_root()
    out = Path(out) if out else root / "labels"
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs or os.cpu_count() or 1
    config = dict(zip(CONFIG_KEYS, (tile, per_tile, crops, ss, fmt)))
    ext = ".parquet" if fmt == "parquet" else ".jsonl"
    done = read_ledger(out)
    todo, skipped, rows_old = [], 0, 0
    for source, items in plan_items(sources, n, per_tile, crops, root).items():
        for i, part in enumerate(_chunks(items, chunk)):
            cid = f"{source}-{i:05d}"
            path = out / f"labels-{cid}{ext}"
            e = done.get(cid)
            if e and e.get("config") == config and path.exists():
                skipped += 1
                rows_old += e["n_rows"]
                continue
            todo.append({"chunk": cid, "source": source, "items": [asdict(it) for it in part], "tile": tile,
                         "crops": crops, "per_tile": per_tile, "ss": ss, "root": str(root),
                         "derived_root": str(derived_root) if derived_root else None})
    log(f"label: {len(todo)} chunks to run, {skipped} already done ({rows_old} rows), jobs={jobs}")
    t0 = time.perf_counter()
    rows_new, ms_total, files = 0, 0.0, []

    def finish(res: dict, job: dict) -> None:
        nonlocal rows_new, ms_total
        path = out / f"labels-{res['chunk']}{ext}"
        _write_shard(path, res["rows"])
        entry = {"chunk": res["chunk"], "source": job["source"], "file": path.name, "n_rows": len(res["rows"]),
                 "n_items": len(job["items"]), "seconds": round(res["seconds"], 3),
                 "done_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "config": config}
        with open(out / LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        rows_new += len(res["rows"])
        ms_total += sum(r["ms"] for r in res["rows"])
        files.append(path.name)
        log(f"  {res['chunk']}: {len(res['rows'])} rows in {res['seconds']:.1f} s")

    if jobs == 1 or len(todo) <= 1:
        for job in todo:
            finish(run_chunk(job), job)
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(todo))) as ex:
            for job, res in zip(todo, ex.map(run_chunk, todo)):
                finish(res, job)
    seconds = time.perf_counter() - t0
    summary = {"rows": rows_old + rows_new, "rows_new": rows_new, "chunks": len(todo), "chunks_skipped": skipped,
               "seconds": round(seconds, 2), "rows_per_s_per_proc": (rows_new / (ms_total / 1000.0)) if ms_total else 0.0,
               "out": str(out), "files": files}
    log(f"label: {rows_new} new rows in {seconds:.1f} s, {summary['rows_per_s_per_proc']:.2f} rows/s per process")
    return summary
