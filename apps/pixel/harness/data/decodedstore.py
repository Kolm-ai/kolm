"""The decoded clean-image store: the training sampler's read-through cache, and python -m harness data decoded.

The training stream (model.data.TileSource) draws its tiles from clean images decoded at their clean scale, and a
50-megapixel featured picture takes seconds to decode and box-downsample: with twelve loader workers on the GPU
box's 16 threads the CPU sat at 100 % while the GPU averaged 45 % (the thirteenth run, 2026-10-28). The store keeps
each plain-file item decoded once, as exactly the array the sampler computes
(<root>/derived/decoded/<source>/<key>.npy), and a worker memmaps it instead of decoding: the same values, so the
stream is the same stream, only cheaper. The key hashes DECODE_VERSION, the item, its source's clean settings and the
raw file's size and modification time, so a changed decode setting, decoder or file misses and is decoded again (the
stale array stays on disk until prune_store, data decoded --prune, removes it); the settings that only choose items
(SELECTION_KEYS: a halo threshold, the held-out scenes) are not keyed, so editing them decodes nothing again. Parquet
rows are left out: their PNG export decodes in milliseconds.
The array is float16 when float16 holds it (every value finite and within HALF_MAX) and float32 otherwise (an
HDRI's sun, past float16's 65504), and the decode itself rounds to float16's values when they hold the image
(decode_clean), so the stream with the store and the stream without it are still one stream. Since 2026-10-29, when
the float32 store of the featured pictures alone took 448 GB of the 1.9 TB drive: float16's step is 2^-11 of the
value (4.9e-4 in the log domain the losses work in), about twenty times finer than an 8-bit source's own
quantisation and below a merged raw reference's remaining noise. compact_store (data decoded --compact) rewrites a
store written before as float16, which is what decoding it again would store, so its keys stay valid.
A worker that misses decodes the item and stores it (a temporary name, then os.replace: a reader never sees half a
file, and two writers of one item end with one), so the store fills while a run trains; the command fills it ahead,
in parallel processes, and resumes, since an item whose array is on disk is not decoded again."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import data_root, keeps_reserve, reserve_bytes
from .label import Item, clean_settings, load_items, skip_reason, split_member

DECODED_DIR = "decoded"   # <root>/derived/decoded/<source>/<key>.npy
DECODE_VERSION = 1        # bump when a decoder's output changes: every key then misses and is decoded again
SELECTION_KEYS = ("max_halo", "holdout")   # clean settings that choose which items train, not how one decodes:
                                           # not keyed
HALF_MAX = 65000.0        # an image whose values are finite and within this is stored as float16 (largest 65504)
RESERVE_ENV, RESERVE_GB = "HARNESS_DECODED_RESERVE_GB", 300   # the cache stops here, above the collector's 150 GB,
                                                              # so raw downloads keep the room the cache gives up


def store_has_room(p: Path, nbytes: int) -> bool:
    """True when storing nbytes at p keeps the drive above the store's free-space floor."""
    return keeps_reserve(p.parent, nbytes, reserve_bytes(RESERVE_ENV, RESERVE_GB))


def decoded_key(it: Item, root: Path) -> str | None:
    """The item's key in the store, or None for a parquet row or a file that is not on disk."""
    if it.row >= 0:
        return None
    try:
        st = (Path(root) / "raw" / it.source / split_member(it.file)[0]).stat()
    except OSError:
        return None
    settings = {k: v for k, v in clean_settings(it.source).items() if k not in SELECTION_KEYS}
    blob = json.dumps([DECODE_VERSION, it.source, it.file, it.column, bool(it.is_hdr), settings,
                       st.st_size, st.st_mtime_ns], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def decoded_path(it: Item, root: Path) -> Path | None:
    k = decoded_key(it, root)
    return None if k is None else Path(root) / "derived" / DECODED_DIR / it.source / f"{k}.npy"


def fits_half(img: np.ndarray) -> bool:
    """True when float16 holds img: every value finite and within HALF_MAX."""
    return bool(np.isfinite(img).all()) and float(np.abs(img).max(initial=0.0)) <= HALF_MAX


def stored_dtype(img: np.ndarray) -> type:
    """The dtype the store keeps img in: float16 when it holds it, float32 otherwise."""
    return np.float16 if fits_half(img) else np.float32


def stored_nbytes(img: np.ndarray) -> int:
    """The bytes img's array takes in the store."""
    return int(img.size) * np.dtype(stored_dtype(img)).itemsize


def decode_clean(it: Item, root: Path) -> np.ndarray:
    """The sampler's clean image of a plain-file item: load_items at the clean scale, as float32, rounded to
    float16's values when they hold it (fits_half), as the store keeps it: with the store or without, one array."""
    img = load_items([it], Path(root))[it].astype(np.float32)
    if fits_half(img):
        np.copyto(img, img.astype(np.float16))
    return img


def read_decoded(p: Path | None) -> np.ndarray | None:
    """The stored array at p (float16 or float32), memmapped read-only; None when it is absent or unreadable (a
    miss). The sampler's crops are float64 whichever it is."""
    if p is None or not p.exists():
        return None
    try:
        a = np.load(p, mmap_mode="r")
    except (OSError, ValueError):
        return None
    return a if a.dtype in (np.float16, np.float32) and a.ndim == 3 and a.shape[2] == 3 else None


def write_decoded(p: Path, img: np.ndarray) -> bool:
    """Store img at p (as stored_dtype: float16 when it holds img) under a temporary name, then rename. True when
    p holds an array afterwards (another
    writer's counts, Windows refuses to replace a file a reader has mapped); never raises. A write that would take
    the drive below the store's floor (HARNESS_DECODED_RESERVE_GB, default 300) is not made: the store is a cache."""
    tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp")
    try:
        dt = stored_dtype(img)
        if not store_has_room(p, int(img.size) * np.dtype(dt).itemsize):
            return p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            np.save(f, np.ascontiguousarray(img, dtype=dt))
        os.replace(tmp, p)
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return p.exists()


def _lower_priority() -> None:
    """A fill that runs beside a training run yields the CPU, the disk and memory to it: on Windows the below-normal
    class and then background mode (which also lowers I/O and memory priority), on POSIX nice 10. The handle and
    its arguments are typed: ctypes' default int would truncate GetCurrentProcess's pseudo-handle (-1) to 32 bits,
    and the call would fail without a word, as the target audit's first run beside v14a found (its workers stayed
    at normal priority and cut the run's sampler from 57 to 17 images a second). Background mode reports the
    normal class (GetPriorityClass 0x20) while it lowers every thread's priority, I/O and memory with it."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            k.GetCurrentProcess.restype = wintypes.HANDLE
            k.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            k.SetPriorityClass.restype = wintypes.BOOL
            me = k.GetCurrentProcess()
            k.SetPriorityClass(me, 0x00004000)   # BELOW_NORMAL_PRIORITY_CLASS
            k.SetPriorityClass(me, 0x00100000)   # PROCESS_MODE_BACKGROUND_BEGIN
        else:
            os.nice(10)
    except Exception:  # noqa: BLE001 - priority is a courtesy, never a reason to fail
        pass


def _fill_task(task: tuple[dict, str]) -> tuple[dict, int, float, str]:
    d, root = task
    it = Item(**d)
    t0 = time.perf_counter()
    p = decoded_path(it, Path(root))
    if p is not None and not store_has_room(p, 0):
        return d, 0, 0.0, "store_full"
    try:
        img = decode_clean(it, Path(root))
    except Exception as e:  # noqa: BLE001 - recorded with its reason; the sampler drops the same item
        return d, 0, time.perf_counter() - t0, skip_reason(e)
    ok = p is not None and write_decoded(p, img)
    return d, stored_nbytes(img) if ok else 0, time.perf_counter() - t0, "" if ok else "write_failed"


def fill_store(items: list[Item], root: Path | None = None, workers: int = 1, background: bool = False,
               log=print) -> dict:
    """Decode and store every plain-file item of `items` not yet stored, `workers` processes at once (below
    normal priority with `background`). Returns the tally: items, stored, already there, parquet rows left out,
    rejected with their reasons, bytes written and seconds."""
    root = Path(root or data_root())
    log = log or (lambda *a: None)
    paths = {it: decoded_path(it, root) for it in items}
    todo = [it for it, p in paths.items() if p is not None and not p.exists()]
    tally = {"items": len(items), "stored": 0, "already": sum(p is not None and p.exists() for p in paths.values()),
             "left_out": sum(p is None for p in paths.values()), "rejected": {}, "bytes": 0, "seconds": 0.0}
    log(f"decoded store: {len(items)} items, {tally['already']} stored, {tally['left_out']} left out (parquet rows "
        f"or missing files), {len(todo)} to decode with {workers} worker(s)")
    t0, done = time.perf_counter(), 0

    def record(d: dict, nbytes: int, secs: float, reason: str) -> None:
        nonlocal done
        done += 1
        if reason:
            tally["rejected"][f"{d['source']}/{d['file']}"] = reason
        else:
            tally["stored"] += 1
            tally["bytes"] += nbytes
        if done % 100 == 0 or done == len(todo):
            log(f"  {done}/{len(todo)} decoded, {tally['bytes'] / 1e9:.1f} GB, {len(tally['rejected'])} rejected, "
                f"{time.perf_counter() - t0:.0f} s")

    tasks = [(asdict(it), str(root)) for it in todo]
    if workers <= 1:
        if background:
            _lower_priority()
        for t in tasks:
            record(*_fill_task(t))
    else:
        # a Pool and not a ProcessPoolExecutor: on Windows, with the CPU busy, an executor's worker that is still
        # starting when the others finish dies on an invalid semaphore handle and can break the pool
        with mp.get_context("spawn").Pool(workers, initializer=_lower_priority if background else None) as pool:
            for r in pool.imap_unordered(_fill_task, tasks):
                record(*r)
            pool.close()
            pool.join()
    tally["seconds"] = time.perf_counter() - t0
    return tally


def _npy_header(p: Path) -> tuple[tuple, np.dtype]:
    with open(p, "rb") as f:
        v = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = read(f)
    return shape, np.dtype(dtype)


def _compact_task(path: str) -> tuple[str, int]:
    """One stored array rewritten as float16 when it is a float32 image float16 holds: (what happened, bytes freed).
    The array is read whole (no map stays open on it) and replaced through a temporary name; Windows refuses to
    replace a file another process has mapped, and that one is left as it is ("busy", for a later pass)."""
    p = Path(path)
    try:
        shape, dtype = _npy_header(p)
    except (OSError, ValueError):
        return "other", 0
    if len(shape) != 3 or shape[2] != 3 or dtype not in (np.float16, np.float32):
        return "other", 0
    if dtype == np.float16:
        return "already", 0
    try:
        img = np.load(p)
    except (OSError, ValueError):
        return "other", 0
    if not fits_half(img):
        return "kept", 0
    before = p.stat().st_size
    tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            np.save(f, img.astype(np.float16))
        os.replace(tmp, p)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return "busy", 0
    return "compacted", before - p.stat().st_size


def compact_store(root: Path | None = None, sources: list[str] | None = None, workers: int = 1,
                  background: bool = False, log=print) -> dict:
    """Rewrite the store's float32 arrays that float16 holds as float16 (a store written before 2026-10-29),
    `workers` processes at once (below normal priority with `background`): what decoding them again would store, so
    their keys stay. Returns the tally: files, compacted, already float16, kept as float32 (past float16), busy
    (mapped by another process), other (not a stored image), bytes freed and seconds. A run again resumes."""
    root = Path(root or data_root())
    log = log or (lambda *a: None)
    base = root / "derived" / DECODED_DIR
    dirs = [base / s for s in sources] if sources else sorted(d for d in base.iterdir() if d.is_dir())
    files = [str(f) for d in dirs if d.is_dir() for f in sorted(d.glob("*.npy"))]
    tally = {"files": len(files), "compacted": 0, "already": 0, "kept": 0, "busy": 0, "other": 0, "freed": 0,
             "seconds": 0.0}
    log(f"decoded store compaction: {len(files)} arrays in {len(dirs)} source(s), {workers} worker(s)")
    t0, done = time.perf_counter(), 0

    def record(r: tuple[str, int]) -> None:
        nonlocal done
        done += 1
        tally[r[0]] += 1
        tally["freed"] += r[1]
        if done % 500 == 0 or done == len(files):
            log(f"  {done}/{len(files)} read, {tally['compacted']} compacted, {tally['freed'] / 1e9:.1f} GB freed, "
                f"{time.perf_counter() - t0:.0f} s")

    if workers <= 1:
        if background:
            _lower_priority()
        for f in files:
            record(_compact_task(f))
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_lower_priority if background else None) as pool:
            for r in pool.imap_unordered(_compact_task, files, chunksize=4):
                record(r)
            pool.close()
            pool.join()
    tally["seconds"] = time.perf_counter() - t0
    return tally


def prune_store(keep: dict[str, set[str]], root: Path | None = None, log=print) -> dict:
    """Remove the arrays no current item keys: for each source in `keep` (the current keys of its items), every
    stored array whose name is not one of them, the arrays a changed decode setting, decoder or raw file left
    behind. A source with no current key is left alone (its raws are not on disk or nothing lists it: no reason to
    call its arrays stale), a file another process has mapped stays ("busy", for a later pass), and a temporary
    name belongs to a write in flight and is not touched. Returns the tally: files, kept, pruned, busy, bytes freed
    and the sources left alone."""
    root = Path(root or data_root())
    log = log or (lambda *a: None)
    base = root / "derived" / DECODED_DIR
    tally = {"files": 0, "kept": 0, "pruned": 0, "busy": 0, "freed": 0, "skipped": []}
    for source, keys in sorted(keep.items()):
        d = base / source
        if not d.is_dir():
            continue
        if not keys:
            tally["skipped"].append(source)
            log(f"  {source}: no item has a current key, nothing removed")
            continue
        n0, b0 = tally["pruned"], tally["freed"]
        for f in sorted(d.glob("*.npy")):
            tally["files"] += 1
            if f.stem in keys:
                tally["kept"] += 1
                continue
            try:
                st = f.stat()
                size = st.st_size if st.st_nlink <= 1 else 0   # another name keeps a linked file's bytes
                f.unlink()
            except OSError:
                tally["busy"] += 1
                continue
            tally["pruned"] += 1
            tally["freed"] += size
        log(f"  {source}: {tally['pruned'] - n0} stale array(s) removed, {(tally['freed'] - b0) / 1e9:.1f} GB")
    return tally
