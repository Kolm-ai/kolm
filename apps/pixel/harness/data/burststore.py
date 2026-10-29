"""The burst store: each HDR+ burst's other frames, registered to its merge, for the burst layer (MODEL_PLAN section
9): python -m harness data burststore.

The real store (realstore) holds a burst's merge (ref.npy) and its reference frame, the frame the merge is aligned
to, as the pair's input. A burst network reads the reference frame and some of the burst's other frames; this store
holds those, each decoded at the real store's scale, brought to the merge's frame as the reference frame was
(realpairs.input_fits, the reference pair's quarter turn), and registered to the merge in two steps. First the whole
frame's integer shift (realpairs.align; its quadrant test is recorded, not required, since a hand-held burst turns a
little between frames, and its overlap must correlate by MIN_CORR). Then a grid of local shifts on BLOCK px blocks
STRIDE px apart: the phase-correlation peak of the frame's block against the merge's within LOCAL_SHIFT px of the
whole frame's, then its sub-pixel part (subpix's band-passed peak fit). A block holds when its overlap correlates by
MIN_CORR at that shift, the sub-pixel part is within subpix.MAX_RESIDUAL, and it agrees within OUTLIER_PX with the
median of the held blocks around it. A block that does not hold (flat, dark, clipped, moving against its
neighbours) takes the shift of the nearest one that does, and a frame where fewer than HELD_MIN of the blocks hold
takes the whole frame's shift everywhere.

The flow f maps the merge's frame to the burst frame's: frame(p + f(p)) ~ merge(p), with p and p + f(p) in pixels
of the merge and of the fitted frame. A reader takes it at a tile's pixels by bilinear interpolation between the
block centres, and the network's warp (model.burst.warp_frames) reads the frame there. The warp is left to the GPU
so a loader worker reads only a window of the frame and the coordinates.

Files: <data_root>/derived/burststore/<source>/<scene key>/<frame key>.npy (float16, the fitted frame or, with a
`window`, the part of it a window of the merge reaches) and <frame key>.json (the fit, the shifts, the grid, the
gain, the frame's index in its burst), the array first and the JSON last as the commit mark, as the real store
writes. Scene and frame keys are the real store's (realstore.key_of), so a reader joins a burst's frames to its
merge and its reference frame there. A full frame is about 18 MB at scale 2; with `window` W a frame keeps about
(W + 2 * STORE_MARGIN) ** 2 pixels, about 6.6 MB at W = 1024. The store writes until the drive's free space would
fall below the real store's floor (HARNESS_REALSTORE_RESERVE_GB), and resumes: a frame whose JSON is on disk is not
decoded again.

HDR+ bursts are taken at one exposure (Hasinoff et al. 2016), so a frame's gain against the merge matches its
reference frame's to within the estimate; the reader brings each frame to the reference frame's exposure anyway
(frame * gain / reference gain), so the target, the merge over the reference's gain, fits every frame. The merge
holds each of its frames' noise at about 1 / N of its variance (N up to 10), so a network shown four of them is
taught to keep that share: the target's bias, on every pair the real store draws already."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..colorspace import luminance
from . import data_root
from .decodedstore import _lower_priority
from .label import file_filter
from .realpairs import (HDRPLUS, MAX_TILE_CLIPPED, MIN_CORR, TILE_DRAWS, _corr, align, decode_raw, gain,
                        input_fits, overlap, phase_shift)
from .realstore import DTYPE, RealStore, StoreFull, _check_room, _save_atomic, _write_json, key_of
from .realstore import store_dir as real_store_dir
from .subpix import BAND, MAX_RESIDUAL, _band, _measure_band

STORE_DIR = "burststore"   # <data_root>/derived/burststore/<source>/<scene key>/{<frame key>.npy, <frame key>.json}
BLOCK = 128                # px at the store's scale: the blocks the local shifts are read on
STRIDE = 64                # px between block origins
LOCAL_SHIFT = 16           # px a block's integer shift may differ from the whole frame's
OUTLIER_PX = 1.5           # px a held block may differ from the median of the held blocks around it
HELD_MIN = 0.1             # share of the blocks that must hold, else the whole frame's shift stands everywhere
STORE_MARGIN = 24          # px of the frame kept around what a stored window's flow reaches
WARP_MARGIN = 8            # px a reader's window holds around a tile's flow at its centre
WARP_SUPPORT = 2           # px the bicubic warp reads beyond a coordinate
FRAME_RE = re.compile(r"payload_N(\d+)[.]dng$", re.IGNORECASE)


def store_dir(source: str, root: Path | None = None) -> Path:
    return Path(root or data_root()) / "derived" / STORE_DIR / source


def frame_json(source: str, scene: str, in_file: str, root: Path | None = None) -> Path:
    return store_dir(source, root) / key_of(scene) / f"{key_of(in_file)}.json"


def frame_index(in_file: str) -> int | None:
    m = FRAME_RE.search(in_file)
    return int(m.group(1)) if m else None


# ---- registration ----------------------------------------------------------------------------------------------

def local_grid(x: np.ndarray, y: np.ndarray, flow: tuple[int, int]) -> dict:
    """The local shifts of frame x against merge y (both (H, W, 3) linear, the frame fitted to the merge's frame)
    around the whole frame's flow (frame(p + flow) ~ merge(p)): per BLOCK px block of the merge, STRIDE px apart, the
    flow (fy, fx) in px and whether the block holds. Returns {"origin": the first block centre, "stride", "shape":
    [rows, columns], "fy", "fx": the flow per block with the blocks that do not hold filled from the nearest that
    does, "held": 0/1 per block, "held_share"}."""
    from scipy.ndimage import distance_transform_edt
    hy, wy = y.shape[:2]
    hx, wx = x.shape[:2]
    lx, ly = luminance(x), luminance(y)
    bx, by = _band(lx, BAND), _band(ly, BAND)   # once per frame: a block's band is the frame's away from its edges
    ys = list(range(0, hy - BLOCK + 1, STRIDE))
    xs = list(range(0, wy - BLOCK + 1, STRIDE))
    fy = np.full((len(ys), len(xs)), np.nan)
    fx = np.full((len(ys), len(xs)), np.nan)
    gy, gx = int(flow[0]), int(flow[1])
    for i, y0 in enumerate(ys):
        for j, x0 in enumerate(xs):
            a0, b0 = y0 + gy, x0 + gx
            if a0 < 0 or b0 < 0 or a0 + BLOCK > hx or b0 + BLOCK > wx:
                continue
            lm = ly[y0:y0 + BLOCK, x0:x0 + BLOCK]
            if float(lm.std()) <= 1e-6:
                continue
            d = phase_shift(lx[a0:a0 + BLOCK, b0:b0 + BLOCK], lm, max_shift=LOCAL_SHIFT)
            iy, ix = gy - int(d[0]), gx - int(d[1])
            a1, b1 = y0 + iy, x0 + ix
            if a1 < 0 or b1 < 0 or a1 + BLOCK > hx or b1 + BLOCK > wx:
                continue
            lf = lx[a1:a1 + BLOCK, b1:b1 + BLOCK]
            if _corr(lf, lm) < MIN_CORR:
                continue
            r = _measure_band(by[y0:y0 + BLOCK, x0:x0 + BLOCK], bx[a1:a1 + BLOCK, b1:b1 + BLOCK])
            if max(abs(r[0]), abs(r[1])) > MAX_RESIDUAL:
                continue
            fy[i, j], fx[i, j] = iy + r[0], ix + r[1]
    held = ~np.isnan(fy)
    if held.any():   # an isolated outlier is a block the peak got wrong, not a moving one
        med_y, med_x = _neighbour_median(fy), _neighbour_median(fx)
        bad = held & (np.hypot(fy - med_y, fx - med_x) > OUTLIER_PX)
        fy[bad], fx[bad] = np.nan, np.nan
        held = ~np.isnan(fy)
    share = float(held.mean()) if held.size else 0.0
    if share < HELD_MIN:
        fy_f, fx_f = np.full(fy.shape, float(gy)), np.full(fx.shape, float(gx))
    else:
        iy_, ix_ = distance_transform_edt(~held, return_distances=False, return_indices=True)
        fy_f, fx_f = fy[iy_, ix_], fx[iy_, ix_]
    return {"origin": [BLOCK / 2, BLOCK / 2], "stride": STRIDE, "shape": [len(ys), len(xs)],
            "fy": np.round(fy_f, 3).tolist(), "fx": np.round(fx_f, 3).tolist(),
            "held": held.astype(int).tolist(), "held_share": round(share, 4)}


def _neighbour_median(a: np.ndarray) -> np.ndarray:
    """The median of each cell's 3 x 3 neighbourhood over the cells that are not NaN (NaN where none is)."""
    h, w = a.shape
    p = np.pad(a, 1, constant_values=np.nan)
    stack = np.stack([p[i:i + h, j:j + w] for i in range(3) for j in range(3)])
    out = np.full(a.shape, np.nan)
    ok = (~np.isnan(stack)).any(axis=0)
    out[ok] = np.nanmedian(stack[:, ok], axis=0)
    return out


def flow_at(grid: dict, py: np.ndarray, px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The grid's flow (fy, fx) at merge coordinates (py, px), bilinear between the block centres and held at the
    outermost ones beyond them."""
    fy, fx = np.asarray(grid["fy"], dtype=np.float64), np.asarray(grid["fx"], dtype=np.float64)
    oy, ox = grid["origin"]
    s = float(grid["stride"])
    gy = np.clip((np.asarray(py, dtype=np.float64) - oy) / s, 0.0, fy.shape[0] - 1)
    gx = np.clip((np.asarray(px, dtype=np.float64) - ox) / s, 0.0, fy.shape[1] - 1)
    i0 = np.minimum(np.floor(gy).astype(int), max(fy.shape[0] - 2, 0))
    j0 = np.minimum(np.floor(gx).astype(int), max(fy.shape[1] - 2, 0))
    i1, j1 = np.minimum(i0 + 1, fy.shape[0] - 1), np.minimum(j0 + 1, fy.shape[1] - 1)
    ty, tx = gy - i0, gx - j0

    def lerp(f):
        top = f[i0, j0] * (1 - tx) + f[i0, j1] * tx
        bot = f[i1, j0] * (1 - tx) + f[i1, j1] * tx
        return top * (1 - ty) + bot * ty
    return lerp(fy), lerp(fx)


def window_of(scene: str, shape: tuple[int, int], side: int) -> list[int]:
    """[y0, x0, rows, columns] of a `side` px window of a merge of `shape`, at a position the scene's name fixes."""
    h, w = shape
    sy, sx = min(side, h), min(side, w)
    d = hashlib.sha256(("burst-window|" + scene).encode("utf-8")).digest()
    y0 = int.from_bytes(d[:4], "little") % (h - sy + 1)
    x0 = int.from_bytes(d[4:8], "little") % (w - sx + 1)
    return [y0, x0, sy, sx]


# ---- writing ---------------------------------------------------------------------------------------------------

def burst_frames(source: str, ref_in_file: str, root: Path) -> list[str]:
    """The burst's frames on disk other than its reference frame (posix, relative to <root>/raw/<source>), the
    nearest to the reference first (ties: the later frame)."""
    base = Path(root) / "raw" / source
    d = (base / ref_in_file).parent
    k_ref = frame_index(ref_in_file) or 0
    out = []
    for f in d.glob("payload_N*.dng"):
        rel = f.relative_to(base).as_posix()
        k = frame_index(rel)
        if rel != ref_in_file and k is not None:
            out.append((abs(k - k_ref), -k, rel))
    return [r for _, _, r in sorted(out)]


def write_burst(source: str, pair_meta: dict, root: Path, window: int | None = None, decode=decode_raw,
                log=None) -> list[dict]:
    """One burst: each of its frames on disk without a JSON decoded, fitted to the merge as the reference frame
    was, registered (align, local_grid), measured for its gain and written, the array first and the JSON last. The
    merge is the real store's ref.npy. With `window` only the part of the frame a `window` px window of the merge
    (window_of) reaches, and STORE_MARGIN px around it, is written. Returns the metas written, rejected ones too;
    raises StoreFull before an array's write when the drive is at the real store's floor."""
    root = Path(root)
    scene, ref_in = pair_meta["scene"], pair_meta["in_file"]
    d = store_dir(source, root) / key_of(scene)
    todo = [f for f in burst_frames(source, ref_in, root) if not frame_json(source, scene, f, root).exists()]
    if not todo:
        return []
    _check_room(d, 0)
    d.mkdir(parents=True, exist_ok=True)
    y = np.asarray(np.load(real_store_dir(source, root) / key_of(scene) / pair_meta.get("ref_npy", "ref.npy"),
                           mmap_mode="r"), dtype=np.float32)
    turn = int((pair_meta.get("fit") or {}).get("turn", 0))
    win = window_of(scene, y.shape[:2], int(window)) if window else None
    base = root / "raw" / source
    scale = int(pair_meta["scale"])
    k_ref = frame_index(ref_in)
    out = []
    for f in todo:
        t0 = time.perf_counter()
        meta = {"source": source, "scene": scene, "in_file": f, "ref_in_file": ref_in, "split": pair_meta["split"],
                "frame": frame_index(f), "ref_frame": k_ref, "scale": scale, "licence": pair_meta.get("licence", "")}
        try:
            x, bits = decode(base / f, scale)
            fits = [(xf, fit) for xf, fit in input_fits(np.asarray(x, dtype=np.float32), y) if fit["turn"] == turn]
            if not fits:
                raise ValueError(f"no fit of the frame {tuple(x.shape)} to the merge {tuple(y.shape)} at the "
                                 f"reference's turn {turn}")
            x, fit = fits[0]
            al = align(x, y)
            if al["corr"] < MIN_CORR:
                raise ValueError(f"not the merge's framing: corr {al['corr']:.2f} at shift {al['shift']}")
            sy, sx = (int(v) for v in al["shift"])     # x[i, j] ~ y[i + sy, j + sx]: frame(p - s) ~ merge(p)
            xs, ys = overlap(x, y, sy, sx)
            g, n_gain = gain(xs, ys)
            grid = local_grid(x, y, (-sy, -sx))
            origin, arr = [0, 0], x
            if win is not None:
                wy0, wx0, wh, ww = win
                py, px = np.meshgrid(np.arange(wy0, wy0 + wh, STRIDE // 2), np.arange(wx0, wx0 + ww, STRIDE // 2),
                                     indexing="ij")
                fy, fx = flow_at(grid, py, px)
                a0 = max(0, int(np.floor(wy0 + fy.min())) - STORE_MARGIN)
                b0 = max(0, int(np.floor(wx0 + fx.min())) - STORE_MARGIN)
                a1 = min(x.shape[0], int(np.ceil(wy0 + wh + fy.max())) + STORE_MARGIN)
                b1 = min(x.shape[1], int(np.ceil(wx0 + ww + fx.max())) + STORE_MARGIN)
                origin, arr = [a0, b0], x[a0:a1, b0:b1]
            _save_atomic(d / f"{key_of(f)}.npy", np.asarray(arr, dtype=DTYPE))
            meta.update(shape=[int(x.shape[0]), int(x.shape[1])], stored=[int(arr.shape[0]), int(arr.shape[1])],
                        origin=origin, window=win, fit=fit, shift=[sy, sx], corr=float(al["corr"]),
                        aligned=bool(al["aligned"]), quadrant_spread=float(al["quadrant_spread"]),
                        quadrants_held=int(al["quadrants_held"]), gain=[float(v) for v in g],
                        gain_pixels=int(n_gain), ref_gain=[float(v) for v in pair_meta["gain"]], bit_depth=int(bits),
                        grid=grid, seconds=round(time.perf_counter() - t0, 1))
        except StoreFull:
            raise
        except Exception as ex:  # noqa: BLE001 - the reason is the record; the frame is never drawn
            meta["rejected"] = f"{type(ex).__name__}: {ex}"[:300]
        _write_json(frame_json(source, scene, f, root), meta)
        out.append(meta)
    if log:
        ok = [m for m in out if "rejected" not in m]
        log(f"{source}: {scene}: {len(ok)} frame(s) stored, {len(out) - len(ok)} rejected"
            + (f", blocks held {np.median([m['grid']['held_share'] for m in ok]):.0%}" if ok else ""))
    return out


def _burst_task(args: tuple) -> tuple[str, list[dict], str]:
    source, meta, root, window = args
    try:
        return meta["scene"], write_burst(source, meta, Path(root), window), ""
    except StoreFull as ex:
        return meta["scene"], [], f"{StoreFull.__name__}: {ex}"
    except Exception as ex:  # noqa: BLE001 - one burst's failure is logged, the rest go on
        return meta["scene"], [], f"{type(ex).__name__}: {ex}"[:300]


def reference_pairs(source: str, splits: tuple[str, ...], root: Path) -> list[dict]:
    """The real store's stored pairs of `splits` whose input is a burst frame: one per burst, its reference frame."""
    out, seen = [], set()
    keep = file_filter(source)
    for j in sorted(real_store_dir(source, root).glob("*/*.json")):
        if j.name.endswith(".tmp.json"):
            continue
        m = json.loads(j.read_text(encoding="utf-8"))
        if (m.get("rejected") or "shape" not in m or m.get("split") not in splits or frame_index(m["in_file"]) is None
                or not keep(m["ref_file"]) or m["scene"] in seen):
            continue
        seen.add(m["scene"])
        out.append(m)
    return out


def build_store(source: str = HDRPLUS, splits: tuple[str, ...] = ("val",), root: Path | None = None,
                workers: int = 1, window: int | None = None, log=print, background: bool = False,
                decode=decode_raw) -> dict:
    """Every burst of `splits` with a reference pair in the real store and frames on disk to store, `workers`
    processes at once (one when `decode` is not the raw decoder). Returns the tally: bursts, frames stored and
    rejected, bursts that failed whole, why it stopped short (`full`), bytes, seconds."""
    root = Path(root or data_root())
    log = log or (lambda *a: None)
    refs = reference_pairs(source, splits, root)
    todo = [m for m in refs if any(not frame_json(source, m["scene"], f, root).exists()
                                   for f in burst_frames(source, m["in_file"], root))]
    log(f"{source}: {len(refs)} burst(s) of splits {', '.join(splits)} in the real store, {len(todo)} with frames to "
        f"store, window {window or 'none'}, {workers} worker(s)")
    metas, failed, full = [], 0, ""
    t0 = time.perf_counter()
    tasks = [(source, m, str(root), window) for m in todo]
    if workers <= 1 or decode is not decode_raw:
        if background:
            _lower_priority()
        results = ((m["scene"], *_run_one(source, m, root, window, decode, log)) for m in todo)
    else:
        pool = mp.get_context("spawn").Pool(workers, initializer=_lower_priority if background else None)
        results = pool.imap_unordered(_burst_task, tasks)
    try:
        for done, (scene, part, err) in enumerate(results, 1):
            if err.startswith(StoreFull.__name__):
                full = err.split(": ", 1)[1]
                log(f"{source}: stopped: {full}")
                break
            if err:
                failed += 1
                log(f"{source}: burst {scene} failed: {err}")
                continue
            metas += part
            if done % 10 == 0 or done == len(tasks):
                ok = sum("rejected" not in m for m in metas)
                log(f"{source}: [{done}/{len(tasks)}] {ok} frame(s) stored, {len(metas) - ok} rejected, "
                    f"{time.perf_counter() - t0:.0f} s elapsed")
    finally:
        if workers > 1 and decode is decode_raw:
            pool.terminate() if full else pool.close()
            pool.join()
    rejected = [m for m in metas if "rejected" in m]
    d = store_dir(source, root)
    size = sum(f.stat().st_size for f in d.rglob("*.npy")) if d.is_dir() else 0
    tally = {"source": source, "splits": list(splits), "bursts": len(refs), "todo": len(todo),
             "stored": len(metas) - len(rejected), "rejected": len(rejected), "failed": failed, "full": full,
             "bytes": size, "seconds": round(time.perf_counter() - t0, 1)}
    log(f"{source}: {tally['stored']} frame(s) stored, {len(rejected)} rejected, {failed} burst(s) failed; "
        f"{size / 1e9:.1f} GB under {d}")
    for m in rejected[:8]:
        log(f"  rejected {m['in_file']}: {m['rejected'][:120]}")
    return tally


def _run_one(source: str, meta: dict, root: Path, window: int | None, decode, log) -> tuple[list[dict], str]:
    try:
        return write_burst(source, meta, root, window, decode, log), ""
    except StoreFull as ex:
        return [], f"{StoreFull.__name__}: {ex}"
    except Exception as ex:  # noqa: BLE001
        return [], f"{type(ex).__name__}: {ex}"[:300]


# ---- reading ---------------------------------------------------------------------------------------------------

@dataclass
class StoredFrame:
    meta: dict
    npy: Path

    @property
    def scale_to_ref(self) -> np.ndarray:
        """Per channel, what brings the frame to its reference frame's exposure: its gain over the reference's."""
        return np.asarray(self.meta["gain"], dtype=np.float64) / np.asarray(self.meta["ref_gain"], dtype=np.float64)


class BurstStore:
    """The bursts of one split: the real store's pairs whose input is a burst's reference frame, each with its
    stored frames. A tile is the real store's (input, target, capture) and up to `frames` of the burst's frames,
    each a window of the frame around where the tile's pixels fall in it, with the coordinates of those pixels in
    the window (the warp reads them), at the reference frame's exposure. A frame whose window leaves its stored
    array is not in the tile (its mask is False), so a tile can hold fewer frames than asked, none at the edge."""

    def __init__(self, split: str = "train", root: Path | None = None, tile: int = 256, frames: int = 3,
                 source: str = HDRPLUS, subpix: bool = False):
        self.root, self.split, self.tile, self.frames, self.source = Path(root or data_root()), split, tile, frames, source
        self.real = RealStore([source], split, self.root, tile, subpix=subpix)
        by_scene: dict[str, list[StoredFrame]] = {}
        d = store_dir(source, self.root)
        if not d.is_dir():
            raise FileNotFoundError(f"no burst store for {source} under {d}: python -m harness data burststore "
                                    f"--splits {split}")
        for j in sorted(d.glob("*/*.json")):
            m = json.loads(j.read_text(encoding="utf-8"))
            if m.get("rejected") or m.get("split") != split or "grid" not in m:
                continue
            npy = j.with_suffix(".npy")
            if npy.exists():
                by_scene.setdefault(m["scene"], []).append(StoredFrame(m, npy))
        self.bursts: list[tuple[int, list[StoredFrame]]] = []
        for k, p in enumerate(self.real.pairs):
            fr = by_scene.get(p.meta["scene"])
            if fr and frame_index(p.meta["in_file"]) is not None and fr[0].meta["ref_in_file"] == p.meta["in_file"]:
                ref_k = frame_index(p.meta["in_file"])
                fr.sort(key=lambda s: (abs(s.meta["frame"] - ref_k), -s.meta["frame"]))
                self.bursts.append((k, fr))
        if not self.bursts:
            raise ValueError(f"no bursts with stored frames in split {split!r} of {source}")
        self._open: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.bursts)

    def _mm(self, path: Path) -> np.ndarray:
        arr = self._open.get(path)
        if arr is None:
            if len(self._open) >= 256:
                self._open.clear()
            arr = self._open[path] = np.load(path, mmap_mode="r")
        return arr

    def merge_box(self, b: int) -> list[int] | None:
        """[y0, x0, rows, columns] of the merge a windowed burst's frames reach, or None (whole frames)."""
        return self.bursts[b][1][0].meta.get("window")

    def tile_at(self, b: int, y0: int, x0: int, tile: int | None = None, n: int | None = None,
                choose: list[int] | None = None) -> dict:
        """Burst b's tile at input (y0, x0): the real store's input, target, clipped fraction and capture, and
        `frames` (n, tile + 2 WARP_MARGIN, same, 3) float32 at the reference frame's exposure, `coords` (n, tile,
        tile, 2) float32 (row, column) of each tile pixel in its frame's window, `mask` (n,) bool, `ids` the
        frames' indices in the burst (-1 where none). `choose` names the stored frames to read (default the n
        nearest the reference)."""
        tile = tile or self.tile
        n = self.frames if n is None else n
        k, fr = self.bursts[b]
        p = self.real.pairs[k]
        x, t, clipped = self.real.tile_at(k, y0, x0, tile, MAX_TILE_CLIPPED)
        dy, dx = p.shift
        r = p.residual if self.real.subpix else None
        ry, rx = r if r is not None else (0.0, 0.0)
        ii, jj = np.meshgrid(np.arange(tile, dtype=np.float64), np.arange(tile, dtype=np.float64), indexing="ij")
        py, px = y0 + dy + ry + ii, x0 + dx + rx + jj        # the merge's point under each input pixel
        side = tile + 2 * WARP_MARGIN
        frames = np.zeros((n, side, side, 3), dtype=np.float32)
        coords = np.zeros((n, tile, tile, 2), dtype=np.float32)
        mask = np.zeros(n, dtype=bool)
        ids = np.full(n, -1, dtype=np.int64)
        pick = list(choose) if choose is not None else list(range(min(n, len(fr))))
        for slot, s in enumerate(pick[:n]):
            f = fr[s]
            fy, fx = flow_at(f.meta["grid"], py, px)
            qy, qx = py + fy - f.meta["origin"][0], px + fx - f.meta["origin"][1]
            cy, cx = int(np.floor(qy[tile // 2, tile // 2])), int(np.floor(qx[tile // 2, tile // 2]))
            a0, b0 = cy - side // 2, cx - side // 2
            arr = self._mm(f.npy)
            if a0 < 0 or b0 < 0 or a0 + side > arr.shape[0] or b0 + side > arr.shape[1]:
                continue
            cy_, cx_ = qy - a0, qx - b0
            if (cy_.min() < WARP_SUPPORT or cx_.min() < WARP_SUPPORT or cy_.max() > side - 1 - WARP_SUPPORT
                    or cx_.max() > side - 1 - WARP_SUPPORT):
                continue
            frames[slot] = np.asarray(arr[a0:a0 + side, b0:b0 + side], dtype=np.float32) * f.scale_to_ref
            coords[slot, ..., 0], coords[slot, ..., 1] = cy_, cx_
            mask[slot], ids[slot] = True, f.meta["frame"]
        return {"x": x, "y": t, "capture": p.capture, "pair": k, "burst": b, "y0": y0, "x0": x0, "clipped": clipped,
                "frames": frames, "coords": coords, "mask": mask, "ids": ids}

    def bounds(self, b: int, tile: int | None = None) -> tuple[int, int, int, int]:
        """The real store's bounds of the tile origins, cut to a windowed burst's merge window (less the pair's
        shift), so its frames reach every tile."""
        tile = tile or self.tile
        k, _ = self.bursts[b]
        ylo, yhi, xlo, xhi = self.real.bounds(k, tile)
        box = self.merge_box(b)
        if box:
            dy, dx = self.real.pairs[k].shift
            wy0, wx0, wh, ww = box
            ylo, yhi = max(ylo, wy0 - dy), min(yhi, wy0 + wh - dy - tile)
            xlo, xhi = max(xlo, wx0 - dx), min(xhi, wx0 + ww - dx - tile)
        return ylo, yhi, xlo, xhi

    def random_tile(self, rng: np.random.Generator, tile: int | None = None, n: int | None = None,
                    tries: int = 32) -> dict:
        """A tile of a random burst, TILE_DRAWS positions tried per burst, the first whose target is clipped on at
        most MAX_TILE_CLIPPED of its pixels; its frames a random subset of the burst's stored ones, in random
        order, so the network sees every count up to `n` and no frame order."""
        tile = tile or self.tile
        n = self.frames if n is None else n
        for _ in range(tries):
            b = int(rng.integers(len(self.bursts)))
            ylo, yhi, xlo, xhi = self.bounds(b, tile)
            if yhi < ylo or xhi < xlo:
                continue
            have = len(self.bursts[b][1])
            for _ in range(TILE_DRAWS):
                y0, x0 = int(rng.integers(ylo, yhi + 1)), int(rng.integers(xlo, xhi + 1))
                out = self.tile_at(b, y0, x0, tile, n, [int(v) for v in rng.permutation(have)[:n]])
                if out["clipped"] <= MAX_TILE_CLIPPED:
                    return out
        raise ValueError(f"no burst tile with an unclipped target in {tries} bursts of {len(self.bursts)}")

    def summary(self) -> str:
        nf = [len(fr) for _, fr in self.bursts]
        held = [f.meta["grid"]["held_share"] for _, fr in self.bursts for f in fr]
        win = sum(1 for b in range(len(self.bursts)) if self.merge_box(b))
        return (f"burst store, split {self.split}: {len(self.bursts)} bursts, {sum(nf)} frames (per burst "
                f"{min(nf)} to {max(nf)}), blocks held median {float(np.median(held)):.0%}, "
                f"{win} windowed")
