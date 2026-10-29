"""A store of the real capture/reference pairs for training: python -m harness data realstore.

realpairs.build_set draws a few dozen tiles of the val and test scenes for an evaluation. Training wants every
pair of the train scenes, drawn from at random for hours by a dozen loader workers, so the store keeps each pair
decoded once at the clean scale: a scene's reference as float16 (ref.npy), each input as float16 beside it
(<key>.npy) with a JSON of its match (shift, gains, correlation, noise law, bit depth, capture descriptor), and a
worker takes a tile from the memmaps without a decode: the input at (y0, x0), the reference at (y0 + dy, x0 + dx)
over the gains and clipped at 1, exactly realpairs.match_pair's target, tiles with clipped reference pixels
rejected as the evaluation set rejects them. A pair the aligner rejects is recorded with its reason and never
drawn. The build goes scene by scene (the reference decoded once), in parallel processes, and resumes: a pair
whose JSON is on disk is not decoded again, so a killed build loses one scene. float16 holds 11 bits of relative
precision, finer than any noise law measured on disk, at half the bytes of float32: about 36 MB per 6-megapixel
frame, 95 GB for RawNIND's and HDR+'s train and val scenes at scale 2. An exposure stack's pairs (SI-HDR) have no
scene reference: each input's reference is the merge of the longer frames chained to it (realpairs.stack_reference),
stored beside it as <key>.ref.npy with NaN where the merge holds none, about 270 MB per 5-frame scene, 44 GB for
SI-HDR's train and val scenes at scale 2. A write that would take the drive below the store's free-space floor
(HARNESS_REALSTORE_RESERVE_GB, default 150) stops the build where it is, and a later run resumes there.

Each pair's JSON also records the residual its integer shift leaves (`subpix`, subpix.measure on the stored arrays'
overlap: about 0.14 px median on RawNIND's train pairs on 10-29), measured as the pair is written or, for a store
written before, by `measure_store` (python -m harness data subpix). A RealStore with `subpix` reads each tile's
reference at its pair's trusted residual on top of the integer shift (subpix.shifted_window): the target sits where
the input does, so training does not learn the blur of an uncertain position."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import re
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import data_root, keeps_reserve, reserve_bytes
from .decodedstore import _lower_priority
from .label import file_filter
from .realpairs import (CLIP, MAX_TILE_CLIPPED, STACK_SOURCES, TILE_DRAWS, Pair, capture_of, decode_raw, match_of,
                        pair_scale, pairs_of, subpix_residual)
from .subpix import residual_of, shifted_window

STORE_DIR = "realstore"   # <data_root>/derived/realstore/<source>/<scene key>/{ref.npy, <pair key>.npy, <pair key>.json}
#                           (a stack source: <pair key>.ref.npy for each pair, no ref.npy)
DTYPE = np.float16
OPEN_PAIRS = 64           # memmaps a RealStore keeps open per process (two per pair)
PAIR_TRIES = 32           # pairs a draw tries before giving up on an unclipped tile
NOISE_KEYS = ("a", "b", "electrons_at_clip", "read_noise_e", "bins", "flat_fraction")
REPLACE_TRIES = 25        # a JSON's move onto its name, retried 0.2 s apart while a reader holds the old one open
RESERVE_ENV, RESERVE_GB = "HARNESS_REALSTORE_RESERVE_GB", 150   # the collector's floor: training data, not a cache,
#                                                                so it writes on past the decoded cache's 300 GB


class StoreFull(RuntimeError):
    """A write would take the drive below the store's floor: the build stops, and the next run resumes there."""


def _check_room(path: Path, nbytes: int) -> None:
    if not keeps_reserve(path, nbytes, reserve_bytes(RESERVE_ENV, RESERVE_GB)):
        raise StoreFull(f"writing {nbytes / 1e6:.0f} MB under {path} would take the drive below its free-space "
                        f"floor ({RESERVE_ENV}, default {RESERVE_GB} GB)")


def key_of(name: str) -> str:
    """A file-system name for a scene or a raw: the stem's safe characters and a short hash so two never collide."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem)[:48].strip("._") or "x"
    return f"{safe}_{hashlib.sha1(name.encode('utf-8')).hexdigest()[:8]}"


def store_dir(source: str, root: Path | None = None) -> Path:
    return Path(root or data_root()) / "derived" / STORE_DIR / source


def scene_dir(source: str, scene: str, root: Path | None = None) -> Path:
    return store_dir(source, root) / key_of(scene)


def pair_json(source: str, p: Pair, root: Path | None = None) -> Path:
    return scene_dir(source, p.scene, root) / f"{key_of(p.in_file)}.json"


def _save_atomic(path: Path, arr: np.ndarray) -> None:
    _check_room(path.parent, arr.nbytes)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def _write_json(path: Path, meta: dict) -> None:
    """The JSON written beside its name and moved onto it, so a reader sees the old or the new and never a part (the
    part, <name>.part, matches no reader's glob); the move is retried while a reader on Windows holds the old open."""
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    for i in range(REPLACE_TRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == REPLACE_TRIES - 1:
                raise
            time.sleep(0.2)


def write_scene(source: str, ref_file: str, pairs: list[Pair], scale: int, root: Path, decode=decode_raw,
                log=None) -> list[dict]:
    """One scene: the reference decoded once and written when absent, then each input without a JSON decoded,
    matched (realpairs.match_of) and written, the input's array first and its JSON last as the commit mark. A stack
    source's scene has no reference of its own: each input's is the merge of its stack's longer frames
    (realpairs.stack_reference), written as <key>.ref.npy before the JSON, which names it (ref_npy) and records the
    merge (stack: frames used, ghost and hole fractions, the links) and the pair's own residual law (noise_pair) beside
    the stack's (noise), and the residual the integer shift leaves (subpix). Returns the metas written, rejected
    ones included; raises StoreFull, before the scene's
    decode or before an array's write, when the drive is at the store's floor (the pairs not written have no JSON)."""
    base = Path(root) / "raw" / source
    d = scene_dir(source, pairs[0].scene, root)
    stack = source in STACK_SOURCES
    todo = [p for p in pairs if not pair_json(source, p, root).exists()]
    ref_p = d / "ref.npy"
    if not todo and (stack or ref_p.exists()):
        return []
    _check_room(d, 0)
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    y, by = (None, None) if stack else decode(base / ref_file, scale)
    if not stack and not ref_p.exists():
        _save_atomic(ref_p, np.asarray(y, dtype=DTYPE))
    out = []
    for p in todo:
        meta = {"source": source, "scene": p.scene, "ref_file": ref_file, "in_file": p.in_file, "split": p.split,
                "iso": p.iso, "licence": p.licence, "scale": int(scale), "ref_bit_depth": None if stack else int(by)}
        try:
            m, x, bx, yp = match_of(p, base, scale, decode, ref=y)
            al = m["align"]
            _save_atomic(d / f"{key_of(p.in_file)}.npy", np.asarray(x, dtype=DTYPE))
            if stack:
                _save_atomic(d / f"{key_of(p.in_file)}.ref.npy", np.asarray(yp, dtype=DTYPE))
                meta.update(ref_npy=f"{key_of(p.in_file)}.ref.npy", ref_bit_depth=int(bx),
                            noise_pair={k: m["noise_pair"].get(k) for k in NOISE_KEYS}, hole=float(m["hole"]),
                            stack=m["stack"])
            try:
                sp = subpix_residual(x, yp, al["shift"], m["gain"])
            except Exception as ex:  # noqa: BLE001 - the pair stays at its integer shift
                sp = {"error": f"{type(ex).__name__}: {ex}"[:300]}
            del yp
            meta.update(subpix=sp, shape=[int(x.shape[0]), int(x.shape[1])], shift=[int(v) for v in al["shift"]],
                        gain=[float(g) for g in m["gain"]], corr=float(al["corr"]),
                        quadrant_spread=float(al["quadrant_spread"]), quadrants_held=int(al["quadrants_held"]),
                        gain_pixels=int(m["gain_pixels"]),
                        noise={k: m["noise"].get(k) for k in NOISE_KEYS}, bit_depth=int(bx), clipped=float(m["clipped"]),
                        capture=[float(v) for v in capture_of(m["noise"], bx)])
            if "fit" in m:
                meta["fit"] = m["fit"]
        except StoreFull:
            raise
        except Exception as ex:  # noqa: BLE001 - the reason is the record; the pair is never drawn
            meta["rejected"] = f"{type(ex).__name__}: {ex}"[:300]
        _write_json(pair_json(source, p, root), meta)
        out.append(meta)
    if log:
        ok = sum("rejected" not in m for m in out)
        log(f"{source}: {pairs[0].scene}: {ok} pair(s) stored, {len(out) - ok} rejected, "
            f"{time.perf_counter() - t0:.0f} s")
    return out


def _scene_task(args: tuple) -> list[dict]:
    source, ref_file, pairs, scale, root = args
    return write_scene(source, ref_file, [Pair(**p) for p in pairs], scale, Path(root))


def _scene_task_safe(args: tuple) -> tuple[str, list[dict], str]:
    """`_scene_task` for a pool: the scene's reference file, its records, and why it failed ("" when it did not), so
    one scene that raises takes nothing else down."""
    try:
        return args[1], _scene_task(args), ""
    except Exception as e:  # noqa: BLE001 - a scene that failed; the rest go on
        return args[1], [], f"{type(e).__name__}: {e}"


def _measure_task(path: str) -> tuple[str, dict | None, str]:
    """One stored pair's residual (realpairs.subpix_residual on its arrays) written into its JSON: the path, the
    measurement, and why it failed ("" when it did not)."""
    j = Path(path)
    try:
        meta = json.loads(j.read_text(encoding="utf-8"))
        x = np.load(j.with_suffix(".npy"), mmap_mode="r")
        ref = np.load(j.parent / meta.get("ref_npy", "ref.npy"), mmap_mode="r")
        sp = subpix_residual(x, ref, meta["shift"], meta["gain"])
        del x, ref
        meta["subpix"] = sp
        _write_json(j, meta)
        return path, sp, ""
    except Exception as e:  # noqa: BLE001 - a pair that failed; the rest go on
        return path, None, f"{type(e).__name__}: {e}"[:300]


def measure_store(source: str, root: Path | None = None, workers: int = 1, log=print, background: bool = False,
                  again: bool = False) -> dict:
    """The residual the integer shift leaves for every stored pair of `source` whose JSON has none (every stored
    pair with `again`), `workers` processes at once (below normal priority with `background`), each JSON rewritten
    with it as write_scene writes it. Safe beside a build (which never rewrites a JSON) and beside readers (the
    rewrite is a move). Returns the tally: pairs stored, measured, trusted, failed, measured before, the trusted
    residuals' median length (px), seconds."""
    root = Path(root or data_root())
    log = log or (lambda *a: None)
    d = store_dir(source, root)
    if not d.is_dir():
        raise FileNotFoundError(f"no real-pair store for {source} under {d}")
    todo, have, stored = [], 0, 0
    for j in sorted(d.glob("*/*.json")):
        if j.name.endswith(".tmp.json"):
            continue
        meta = json.loads(j.read_text(encoding="utf-8"))
        if meta.get("rejected") or "shape" not in meta:
            continue
        stored += 1
        if "subpix" in meta and not again:
            have += 1
        else:
            todo.append(str(j))
    log(f"{source}: {stored} stored pair(s), {have} measured before, {len(todo)} to measure with {workers} worker(s)")
    t0 = time.perf_counter()
    got: list[dict] = []
    failed = 0

    def take(done: int, path: str, sp: dict | None, err: str) -> None:
        nonlocal failed
        if err:
            failed += 1
            log(f"{source}: {Path(path).parent.name}/{Path(path).name} failed: {err}")
        else:
            got.append(sp)
        if done % 50 == 0 or done == len(todo):
            log(f"{source}: [{done}/{len(todo)}] {sum(bool(g.get('trusted')) for g in got)} trusted of {len(got)} "
                f"measured, {failed} failed, {time.perf_counter() - t0:.0f} s elapsed")

    if workers <= 1 or len(todo) <= 1:
        if background:
            _lower_priority()
        for done, path in enumerate(todo, 1):
            take(done, *_measure_task(path))
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_lower_priority if background else None) as pool:
            for done, r in enumerate(pool.imap_unordered(_measure_task, todo), 1):
                take(done, *r)
            pool.close()
            pool.join()
    trusted = [g for g in got if g.get("trusted")]
    med = round(float(np.median([np.hypot(*g["shift"]) for g in trusted])), 4) if trusted else None
    tally = {"source": source, "pairs": stored, "measured": len(got), "trusted": len(trusted), "failed": failed,
             "already": have, "median_px": med, "seconds": round(time.perf_counter() - t0, 1)}
    log(f"{source}: {len(got)} measured ({len(trusted)} trusted, median {med if med is not None else '-'} px), "
        f"{failed} failed, {have} measured before")
    return tally


def build_store(source: str, splits: tuple[str, ...] = ("train",), scale: int | None = None,
                root: Path | None = None, workers: int = 1, log=print, decode=decode_raw,
                pairs: list[Pair] | None = None, background: bool = False) -> dict:
    """Every pair of `splits` into the store at the pair scale (realpairs.pair_scale, or `scale`), scene by scene,
    `workers` processes at once (one when `decode` is not the raw decoder: a test's decoder does not pickle;
    below normal priority with `background`, decodedstore._lower_priority). Returns the tally: pairs listed,
    scenes, stored, rejected, scenes that failed whole, already there, and why it stopped short (`full`, "" when it
    did not: the drive reached the store's floor, StoreFull)."""
    root = Path(root or data_root())
    log = log or (lambda *a: None)
    scale = int(scale or pair_scale(source))
    d = store_dir(source, root)
    info_p = d / "store.json"
    if info_p.exists():
        info = json.loads(info_p.read_text(encoding="utf-8"))
        if int(info["scale"]) != scale:
            raise ValueError(f"{d} holds scale {info['scale']}, asked for {scale}: one store per source")
    pairs = pairs if pairs is not None else pairs_of(source, root, log)
    chosen = [p for p in pairs if p.split in splits]
    if not chosen:
        raise ValueError(f"{source}: no pairs in splits {splits} ({len(pairs)} pairs on disk)")
    scenes: dict[tuple[str, str], list[Pair]] = {}
    for p in chosen:
        scenes.setdefault((p.scene, p.ref_file), []).append(p)
    have = sum(pair_json(source, p, root).exists() for p in chosen)
    todo = [(scene, ref, ps) for (scene, ref), ps in scenes.items()
            if not all(pair_json(source, p, root).exists() for p in ps)]
    log(f"{source}: {len(chosen)} pairs in {len(scenes)} scenes of splits {', '.join(splits)} at scale {scale}; "
        f"{have} already stored, {len(todo)} scene(s) to build with {workers} worker(s)")
    d.mkdir(parents=True, exist_ok=True)
    _write_json(info_p, {"source": source, "scale": scale, "splits": sorted(set(splits) | set(
        (info.get("splits", []) if info_p.exists() else [])))})
    metas: list[dict] = []
    failed, full = 0, ""
    t0 = time.perf_counter()
    if workers <= 1 or decode is not decode_raw:
        if background:
            _lower_priority()
        for scene, ref, ps in todo:
            try:
                metas += write_scene(source, ref, ps, scale, root, decode, log)
            except StoreFull as e:
                full = str(e)
                log(f"{source}: stopped: {full}")
                break
    else:
        tasks = [(source, ref, [asdict(p) for p in ps], scale, str(root)) for scene, ref, ps in todo]
        # a Pool and not a ProcessPoolExecutor, as in decodedstore: on this Windows Python an executor's worker can die
        # on an invalid handle in the call queue (WinError 6), which breaks the pool and fails every scene left
        with mp.get_context("spawn").Pool(workers, initializer=_lower_priority if background else None) as pool:
            for done, (ref, part, err) in enumerate(pool.imap_unordered(_scene_task_safe, tasks), 1):
                if err.startswith(StoreFull.__name__):
                    full = err.split(": ", 1)[1]
                    log(f"{source}: stopped: {full}")
                    pool.terminate()
                    break
                if err:
                    failed += 1
                    log(f"{source}: scene of {ref} failed: {err}")
                    continue
                metas += part
                ok = sum("rejected" not in m for m in part)
                log(f"{source}: [{done}/{len(tasks)}] {part[0]['scene'] if part else ref}: {ok} stored, "
                    f"{len(part) - ok} rejected, {time.perf_counter() - t0:.0f} s elapsed")
            pool.close()
            pool.join()
    rejected = [m for m in metas if "rejected" in m]
    size = sum(f.stat().st_size for f in d.rglob("*.npy"))
    tally = {"source": source, "scale": scale, "splits": list(splits), "pairs": len(chosen), "scenes": len(scenes),
             "stored": len(metas) - len(rejected), "rejected": len(rejected), "failed": failed, "already": have,
             "bytes": size, "full": full,
             "seconds": round(time.perf_counter() - t0, 1)}
    log(f"{source}: {tally['stored']} pair(s) stored, {len(rejected)} rejected, {failed} scene(s) failed, "
        f"{have} were there; "
        f"{size / 1e9:.1f} GB under {d}")
    for m in rejected[:8]:
        log(f"  rejected {m['in_file']}: {m['rejected'][:100]}")
    return tally


@dataclass
class StoredPair:
    meta: dict
    npy: Path
    ref: Path

    @property
    def shift(self) -> tuple[int, int]:
        return int(self.meta["shift"][0]), int(self.meta["shift"][1])

    @property
    def gain(self) -> np.ndarray:
        return np.asarray(self.meta["gain"], dtype=np.float64)

    @property
    def capture(self) -> np.ndarray:
        return np.asarray(self.meta["capture"], dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.meta["shape"][0]), int(self.meta["shape"][1])

    @property
    def residual(self) -> tuple[float, float] | None:
        return residual_of(self.meta)


class RealStore:
    """The stored pairs of one split across sources, memmapped on demand (OPEN_PAIRS kept open); random tiles
    whose reference is unclipped, as realpairs._draw_tiles accepts them. A pair whose reference the source's
    `clean.exclude` now names is left out, so a store built before the pattern needs no rebuild. With `subpix` a
    tile's reference is read at its pair's trusted residual on top of the integer shift (tile_at); a pair without
    one (not measured, or not trusted) is read at its integer shift."""
    def __init__(self, sources: list[str], split: str = "train", root: Path | None = None, tile: int = 256,
                 subpix: bool = False):
        self.root, self.split, self.tile, self.subpix = Path(root or data_root()), split, tile, bool(subpix)
        self.pairs: list[StoredPair] = []
        for s in sources:
            keep = file_filter(s)
            d = store_dir(s, self.root)
            if not d.is_dir():
                raise FileNotFoundError(f"no real-pair store for {s} under {d}: python -m harness data realstore "
                                        f"--sources {s} --splits {split}")
            for j in sorted(d.glob("*/*.json")):
                if j.name.endswith(".tmp.json"):   # an older build's part, not a pair
                    continue
                meta = json.loads(j.read_text(encoding="utf-8"))
                if (meta.get("rejected") or meta.get("split") != split or "shape" not in meta
                        or not keep(meta["ref_file"])):
                    continue
                npy, ref = j.with_suffix(".npy"), j.parent / meta.get("ref_npy", "ref.npy")
                if npy.exists() and ref.exists():
                    self.pairs.append(StoredPair(meta, npy, ref))
        if not self.pairs:
            raise ValueError(f"no stored pairs of split {split!r} in {sources}")
        by: dict[tuple, list[int]] = {}
        for k, p in enumerate(self.pairs):
            by.setdefault((p.meta["source"], p.meta["scene"]), []).append(k)
        self._scene_pairs = list(by.values())
        self.scenes = len(self._scene_pairs)
        self._open: OrderedDict[Path, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.pairs)

    def _mm(self, path: Path) -> np.ndarray:
        if path in self._open:
            self._open.move_to_end(path)
            return self._open[path]
        arr = np.load(path, mmap_mode="r")
        self._open[path] = arr
        while len(self._open) > 2 * OPEN_PAIRS:
            self._open.popitem(last=False)
        return arr

    def bounds(self, k: int, tile: int | None = None) -> tuple[int, int, int, int]:
        """(y_lo, y_hi, x_lo, x_hi), inclusive, of the tile origins in the input whose reference tile lies within
        the frame under the pair's shift: x[i, j] ~ ref[i + dy, j + dx]."""
        tile = tile or self.tile
        p = self.pairs[k]
        h, w = p.shape
        dy, dx = p.shift
        return max(0, -dy), min(h, h - dy) - tile, max(0, -dx), min(w, w - dx) - tile

    def tile_at(self, k: int, y0: int, x0: int, tile: int | None = None,
                max_clipped: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
        """(input, target, clipped fraction): the target the reference over the gains, clipped at 1; the clipped
        fraction that of reference pixels at or above CLIP in any channel, or without a reference (a stack merge's
        NaN, where the target is the input, as realpairs.match_stack fills it). With the store's `subpix` and a
        trusted residual on the pair, the reference is read at the residual on top of the integer shift
        (subpix.shifted_window), unless the tile's clipped fraction is above `max_clipped` (the caller drops it);
        the clipped fraction and the holes are the window's at the integer shift either way."""
        tile = tile or self.tile
        p = self.pairs[k]
        dy, dx = p.shift
        x = np.asarray(self._mm(p.npy)[y0:y0 + tile, x0:x0 + tile], dtype=np.float64)
        ref = self._mm(p.ref)
        ys = np.asarray(ref[y0 + dy:y0 + dy + tile, x0 + dx:x0 + dx + tile], dtype=np.float64)
        hole = np.isnan(ys).any(axis=-1)
        clipped = float(np.mean(hole | (np.nan_to_num(ys, nan=0.0).max(axis=-1) >= CLIP)))
        d = p.residual if self.subpix else None
        if d is not None and (max_clipped is None or clipped <= max_clipped):
            ys = shifted_window(ref, y0 + dy, x0 + dx, tile, d)
        t = np.clip(ys / p.gain, 0.0, 1.0)
        if hole.any():
            t = np.where(hole[..., None], np.clip(x, 0.0, 1.0), t)
        return x, t, clipped

    def random_tile(self, rng: np.random.Generator, tile: int | None = None) -> dict:
        """A tile of a random pair of a random scene (every scene has the same draw mass, its captures share it),
        TILE_DRAWS positions tried per pair and PAIR_TRIES pairs, the first whose reference is clipped on at most
        MAX_TILE_CLIPPED of its pixels."""
        tile = tile or self.tile
        for _ in range(PAIR_TRIES):
            ks = self._scene_pairs[int(rng.integers(len(self._scene_pairs)))]
            k = ks[int(rng.integers(len(ks)))]
            ylo, yhi, xlo, xhi = self.bounds(k, tile)
            if yhi < ylo or xhi < xlo:
                continue
            for _ in range(TILE_DRAWS):
                y0, x0 = int(rng.integers(ylo, yhi + 1)), int(rng.integers(xlo, xhi + 1))
                x, t, clipped = self.tile_at(k, y0, x0, tile, MAX_TILE_CLIPPED)
                if clipped <= MAX_TILE_CLIPPED:
                    return {"x": x, "y": t, "capture": self.pairs[k].capture, "pair": k, "y0": y0, "x0": x0,
                            "clipped": clipped}
        raise ValueError(f"no tile with an unclipped reference in {PAIR_TRIES} pairs of {len(self.pairs)}")

    def shares(self) -> dict[str, float]:
        """The draw mass per source: its share of the scenes."""
        by: dict[str, int] = {}
        for ks in self._scene_pairs:
            s = self.pairs[ks[0]].meta["source"]
            by[s] = by.get(s, 0) + 1
        return {s: n / len(self._scene_pairs) for s, n in sorted(by.items())}

    def summary(self) -> str:
        by = {}
        for p in self.pairs:
            by[p.meta["source"]] = by.get(p.meta["source"], 0) + 1
        es = [p.meta["noise"].get("electrons_at_clip") for p in self.pairs if p.meta["noise"].get("electrons_at_clip")]
        sh = self.shares()
        parts = ", ".join(f"{s} {n} drawn {100 * sh[s]:.0f}%" for s, n in sorted(by.items()))
        e = (f"; electrons at clip {min(es):.4g} to {max(es):.4g}, median {float(np.median(es)):.4g}, "
             f"{len(self.pairs) - len(es)} pair(s) without a noise fit") if es else ""
        sp = ""
        if self.subpix:
            on = [float(np.hypot(*p.residual)) for p in self.pairs if p.residual is not None]
            un = sum("subpix" not in p.meta for p in self.pairs)
            sp = (f"; sub-pixel residual applied on {len(on)} of {len(self.pairs)} pairs"
                  + (f" (median {float(np.median(on)):.3f} px)" if on else "") + (f", {un} not measured" if un else ""))
        return f"real store, split {self.split}: {len(self.pairs)} pairs in {self.scenes} scenes ({parts}){e}{sp}"
