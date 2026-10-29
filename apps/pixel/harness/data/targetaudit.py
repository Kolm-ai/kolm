"""The clean-target audit: one reading per clean image of the halo it already carries, and the filter that keeps
sharpened pictures out of the training targets. python -m harness data audit fills it.

The thirteenth run's restorations overshoot edges more than the captures they restore (HDR+ real pairs: 0.0216
known, 0.0498 blind, the capture 0.0063). Part of its clean targets are display-referred pictures, finished with the
sharpening their makers applied, and a restorer trained toward them learns to add it. edge_overshoot_nr
(metrics.natural) read on 120 decoded clean images per source (2026-10-28): medians 0.046 for Commons featured
pictures (75th percentile 0.072), 0.061 and 0.054 for the KADID and TID references (20 each), against 0.014 for
the HDR+ merges, 0.022 for the Poly Haven HDRIs and 0.002 for the RawNIND ground truths (75th percentiles 0.044,
0.045, 0.012). One more 2x box downsample moved the Commons median only to 0.038: a coarser clean scale does not
remove it, so the filter is per item. The reading is not zero on a pristine picture (texture and thin lines
exceed their surroundings too), so a source's threshold, `clean.max_halo` in its manifest block, is set from the
linear references' own readings.

Each item is read once, on its clean image as the sampler draws it (the decoded store's array, stored on a miss,
or a parquet row's PNG export), as the median over an even grid of AUDIT_CROP crops (at most AUDIT_MAX_CROPS, so
a 100-megapixel panorama costs what a 16-megapixel picture does). Rows go to <root>/derived/target_audit/
<source>.jsonl, keyed by the item, its decode key and the reading's settings, so a changed file, decoder or
setting reads again. filter_targets() keeps the items of a source with a `max_halo` at or below it, and leaves
out its items not yet audited (counted: the sampler's draw never changes mid-run as the audit fills)."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import data_root
from ..metrics.natural import HALF_WINDOW, MIN_STEP, NR_FAR, edge_overshoot_nr
from .collect import _file_lock
from .decodedstore import _lower_priority, decode_clean, decoded_key, decoded_path, read_decoded, write_decoded
from .label import Item, clean_settings, skip_reason

AUDIT_DIR = "target_audit"   # <root>/derived/target_audit/<source>.jsonl
AUDIT_VERSION = 1            # bump when the reading changes: every item is read again
AUDIT_CROP = 1024            # px: the side of the crops the reading's median runs over
AUDIT_MAX_CROPS = 16


def audit_path(source: str, root: Path) -> Path:
    return Path(root) / "derived" / AUDIT_DIR / f"{source}.jsonl"


def audit_key(it: Item, root: Path) -> str:
    """The item's reading key: the item, its decode key (a plain file's size, time and clean settings) and the
    reading's own settings."""
    dk = decoded_key(it, root) if it.row < 0 else ""
    blob = json.dumps([AUDIT_VERSION, AUDIT_CROP, AUDIT_MAX_CROPS, HALF_WINDOW, NR_FAR, MIN_STEP, it.source, it.file,
                       it.row, it.column, dk], default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _starts(n: int, k: int) -> list[int]:
    return [0] if n <= AUDIT_CROP else [int(round(v)) for v in np.linspace(0, n - AUDIT_CROP, k)]


def halo_reading(img: np.ndarray) -> float:
    """edge_overshoot_nr's median over an even grid of AUDIT_CROP crops of a linear clean image; nan when no crop
    holds enough edges for a reading."""
    h, w = img.shape[:2]
    ny, nx = max(1, round(h / AUDIT_CROP)), max(1, round(w / AUDIT_CROP))
    while ny * nx > AUDIT_MAX_CROPS:
        if ny >= nx:
            ny -= 1
        else:
            nx -= 1
    vals = [edge_overshoot_nr(img[y:y + AUDIT_CROP, x:x + AUDIT_CROP]) for y in _starts(h, ny) for x in _starts(w, nx)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def clean_image(it: Item, root: Path) -> np.ndarray:
    """The item's clean image as the sampler draws it: a parquet row's PNG export, or a plain file's stored array,
    decoded and stored when it is not there."""
    if it.row >= 0:
        from ..model.data import row_image
        return row_image(it, root)
    p = decoded_path(it, root)
    img = read_decoded(p)
    if img is None:
        img = decode_clean(it, root)
        if p is not None:
            write_decoded(p, img)
    return np.asarray(img, dtype=np.float32)        # a float16 array read whole, as the decode gives it


def _audit_task(task: tuple[dict, str]) -> tuple[dict, str, float | None, float, str]:
    d, root = task
    it = Item(**d)
    t0 = time.perf_counter()
    key = audit_key(it, Path(root))
    try:
        v = halo_reading(clean_image(it, Path(root)))
    except Exception as e:  # noqa: BLE001 - recorded with its reason; the sampler drops the same item
        return d, key, None, time.perf_counter() - t0, skip_reason(e)
    return d, key, (v if np.isfinite(v) else None), time.perf_counter() - t0, ""


def read_audit(source: str, root: Path | None = None) -> dict[tuple[str, int], dict]:
    """A source's readings by (file, row), the last row of an item winning; a torn line is skipped."""
    p = audit_path(source, Path(root or data_root()))
    out: dict[tuple[str, int], dict] = {}
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                out[(r["file"], int(r["row"]))] = r
            except (ValueError, KeyError, TypeError):
                continue
    return out


def audit_items(items: list[Item], root: Path | None = None, workers: int = 1, background: bool = False,
                log=print) -> dict:
    """Read every item of `items` without a current reading, `workers` processes at once (below normal priority
    with `background`), appending each reading as it lands. Returns the tally: items, read, already read,
    rejected with their reasons, seconds."""
    root = Path(root or data_root())
    log = log or (lambda *a: None)
    have = {s: read_audit(s, root) for s in {it.source for it in items}}

    def current(it: Item) -> bool:
        r = have[it.source].get((it.file, it.row))
        return r is not None and r.get("key") == audit_key(it, root)

    todo = [it for it in items if not current(it)]
    tally = {"items": len(items), "read": 0, "already": len(items) - len(todo), "rejected": {}, "seconds": 0.0}
    log(f"target audit: {len(items)} items, {tally['already']} read, {len(todo)} to read with {workers} worker(s)")
    t0, done = time.perf_counter(), 0

    def record(d: dict, key: str, v: float | None, secs: float, reason: str) -> None:
        nonlocal done
        done += 1
        if reason:
            tally["rejected"][f"{d['source']}/{d['file']}"] = reason
        else:
            p = audit_path(d["source"], root)
            p.parent.mkdir(parents=True, exist_ok=True)
            row = {"source": d["source"], "file": d["file"], "row": d["row"], "column": d["column"], "key": key,
                   "halo": v, "seconds": round(secs, 2)}
            with _file_lock(p.with_suffix(".lock")), open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            tally["read"] += 1
        if done % 100 == 0 or done == len(todo):
            log(f"  {done}/{len(todo)} read, {len(tally['rejected'])} rejected, {time.perf_counter() - t0:.0f} s")

    tasks = [(asdict(it), str(root)) for it in todo]
    if workers <= 1:
        if background:
            _lower_priority()
        for t in tasks:
            record(*_audit_task(t))
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_lower_priority if background else None) as pool:
            for r in pool.imap_unordered(_audit_task, tasks):
                record(*r)
            pool.close()
            pool.join()
    tally["seconds"] = time.perf_counter() - t0
    return tally


def filter_targets(items: list[Item], root: Path | None = None) -> tuple[list[Item], dict[str, dict]]:
    """The items the sampler may draw: every item of a source without `clean.max_halo`, and of one with it the
    items whose current reading is at or below it (an image too smooth for a reading is kept: it carries no
    halo). Returns them with a report per filtered source: the threshold, kept, over, no reading (kept) and not
    audited (left out, with a warning naming the command)."""
    root = Path(root or data_root())
    readings: dict[str, dict] = {}
    report: dict[str, dict] = {}
    kept = []
    for it in items:
        mx = clean_settings(it.source).get("max_halo")
        if mx is None:
            kept.append(it)
            continue
        if it.source not in readings:
            readings[it.source] = read_audit(it.source, root)
            report[it.source] = {"max_halo": float(mx), "kept": 0, "over": 0, "no_reading": 0, "unaudited": 0}
        r, rep = readings[it.source].get((it.file, it.row)), report[it.source]
        if r is None or r.get("key") != audit_key(it, root):
            rep["unaudited"] += 1
        elif r["halo"] is None or r["halo"] <= mx:
            rep["kept"] += 1
            rep["no_reading"] += r["halo"] is None
            kept.append(it)
        else:
            rep["over"] += 1
    for s, rep in report.items():
        if rep["unaudited"]:
            warnings.warn(f"target audit: {rep['unaudited']} items of {s} have no current reading and are left out; "
                          f"python -m harness data audit --sources {s}", RuntimeWarning, stacklevel=2)
    return kept, report
