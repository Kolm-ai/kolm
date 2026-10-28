"""Real noisy/clean pairs from captures: DEGRADATION_PLAN step 4. python -m harness data realpairs.

The sampler's pairs are synthetic: a clean tile degraded by the camera model. A real pair is two captures of
one scene by one camera, an exposure whose noise the model should remove (the input) and a reference whose
own noise sits well below it (a long low-ISO exposure, or a merge of a burst). Two sources on disk give them:

- RawNIND (rawnind): per scene one or two low-ISO "GT" raws and an ISO sweep of noisy raws, on a tripod, one
  framing. The first GT on disk is the reference; the other GT and every noisy raw are inputs.
- HDR+ (hdr-plus-burst): per burst the frames and Google's merge of them (results_20171023, else the older
  results_20161014). The merge is the reference; the burst's reference frame (reference_frame.txt), or frame 0
  when that frame is not on disk, is the input.

Both frames go through LibRaw as the labeller decodes raws (camera white balance, linear, no brightening) and
the source's clean scale (box_downsample). Then: the integer shift between them from phase correlation of their
luminance, refined over +-REFINE px on the normalised MSE, with the shift of each quadrant checked against the
whole (a disagreement past QUADRANT_TOL means rotation or parallax: the pair is recorded as not aligned and not
used); a per-channel gain from the means of the pixels the reference holds unclipped, which brings the reference
to the input's exposure and white balance (t = y / g, clipped at 1); and the input's noise law against that
target on flat unclipped pixels, var = a t + b, read as electrons at the clip 1/a and read noise sqrt(b)/a.
These are effective numbers on demosaiced, white-balanced RGB at the working scale, what the model sees, not
the sensor's. They fill capture_descriptor() for the "known" evaluation; the Airy radius is 0 (unknown: no
source states its pixel pitch), which the report says.

A set is a function of (source, seed, n, tile, scale, splits): tiles drawn at random from the aligned, gain-
matched pairs of the chosen splits (val and test by default, scenes no run trains on; "test" is what the
manifest's clean.holdout names, RawNIND's test reserve), rejecting tiles with clipped reference pixels. It is
cached under <data_root>/derived/realpairs/<source>/ with a parquet of label rows beside it: the twelve tile
metrics of input against target plus scene, ISO, shift, gains and noise law, the real counterpart of the
labeller's synthetic rows. compare() puts the quantiles of those rows against the synthetic rows', so the
sampler's coverage of real degradations is a number."""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..colorspace import luminance
from ..metrics import STILL_METRICS
from ..scene import box_downsample
from . import data_root
from .label import Item, _seed, _write_shard, clean_settings, scene_of
from .tilemetrics import tile_metrics

RAWNIND = "rawnind"
HDRPLUS = "hdr-plus-burst"
SOURCES = (RAWNIND, HDRPLUS)
REALPAIR_DIR = "realpairs"       # <data_root>/derived/realpairs/<source>/s<seed>_n<n>_t<tile>_k<scale>.npz
DEFAULT_SPLITS = ("val", "test")
MAX_SHIFT = 64                   # px at the working scale a phase-correlation peak may claim
REFINE = 2                       # px of MSE refinement around the peak
QUADRANT_TOL = 1.0               # px the four quadrants' shifts may differ from the whole's
MIN_CORR = 0.5                   # luminance correlation of the aligned overlap below which a pair is not a pair
CLIP = 0.98                      # a pixel at or above this in any channel is clipped
GAIN_RANGE = (0.02, 0.8)         # reference pixels the gain is estimated on
FLAT_GRAD = 0.01                 # luminance step to a neighbour below which a pixel counts as flat
MIN_BIN = 500                    # flat pixels a noise-law bin needs
MAX_TILE_CLIPPED = 0.01          # fraction of clipped reference pixels a tile may hold
TILE_DRAWS = 24                  # positions tried per tile
_RAWNIND_ISO = re.compile(r"_ISO([0-9]+)_sha1=", re.IGNORECASE)
ISO_UNKNOWN = 65535              # RawNIND's ISO label for a raw whose metadata gave none


@dataclass
class Pair:
    source: str
    scene: str
    ref_file: str          # the reference (clean) raw, posix path relative to <root>/raw/<source>
    in_file: str           # the input (noisy) raw
    split: str             # train | val | test, of the reference's scene
    iso: float | None = None
    licence: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class RealTile:
    pair: int              # index into RealPairSet.pairs
    y0: int                # in the aligned overlap, input coordinates at the working scale
    x0: int
    shift: tuple[int, int]
    gain: tuple[float, float, float]
    corr: float
    noise: dict            # a, b, electrons_at_clip, read_noise_e, bins
    bit_depth: int
    clipped: float         # fraction of clipped reference pixels in the tile


@dataclass
class RealPairSet:
    source: str
    seed: int
    n: int
    tile: int
    scale: int
    splits: tuple[str, ...]
    pairs: list[Pair]               # every pair of the chosen splits, used or not
    tiles: np.ndarray               # (n, tile, tile, 3) float32 inputs, linear, clip at 1.0
    targets: np.ndarray             # (n, tile, tile, 3) float32 references at the input's exposure
    captures: np.ndarray            # (n, CAPTURE_DIM) float32 capture descriptors
    records: list[RealTile]
    rejected: dict[str, str]        # in_file -> why the pair gave no tile


# ---- listing ---------------------------------------------------------------------------------------------------

def _split_of(item: Item) -> str:
    from ..model.data import split_of
    return split_of(item)


def _licences(base: Path) -> dict[str, str]:
    """RawNIND's permissions.txt: "<scene>: CC0, by <author>" per line."""
    p = base / "permissions.txt"
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def rawnind_iso(file: str) -> float | None:
    m = _RAWNIND_ISO.search(file)
    return float(m.group(1)) if m else None


def rawnind_pairs(root: Path | None = None, log=None) -> list[Pair]:
    """Every (reference GT, input) pair of the RawNIND scenes with a GT on disk, from dataset.yaml. The split
    is "test" for the test reserve (the yaml's flag or test_reserve.yaml), else the reference's hashed split."""
    import yaml
    base = Path(root or data_root()) / "raw" / RAWNIND
    index = yaml.safe_load((base / "dataset.yaml").read_text(encoding="utf-8"))
    reserve = set()
    tr = base / "test_reserve.yaml"
    if tr.exists():
        reserve |= set((yaml.safe_load(tr.read_text(encoding="utf-8")) or {}).get("test_reserve", []))
    licences = _licences(base)
    on_disk = {p.name for p in base.iterdir() if p.is_file()}
    pairs, unmarked = [], []
    for cfa, scenes in sorted(index.items()):
        for scene, d in sorted((scenes or {}).items()):
            cleans = sorted(c["filename"] for c in d.get("clean_images", []) if c["filename"] in on_disk)
            noisy = sorted(c["filename"] for c in d.get("noisy_images", []) if c["filename"] in on_disk)
            if not cleans:
                continue
            ref = cleans[0]
            hashed = _split_of(Item(RAWNIND, ref))
            test = bool(d.get("test_reserve")) or scene in reserve
            if test and hashed != "test":
                unmarked.append(scene)
            split = "test" if test else hashed
            meta = {"cfa": cfa, "unknown_sensor": bool(d.get("unknown_sensor")), "ref_iso": rawnind_iso(ref)}
            for f in cleans[1:] + noisy:
                pairs.append(Pair(RAWNIND, scene, ref, f, split, iso=rawnind_iso(f), licence=licences.get(scene, ""),
                                  meta=dict(meta)))
    if unmarked and log:
        log(f"{RAWNIND}: {len(unmarked)} test-reserve scene(s) the manifest's clean.holdout does not name "
            f"(the sampler may train on them): {', '.join(unmarked[:8])}")
    return pairs


def hdrplus_pairs(root: Path | None = None, log=None) -> list[Pair]:
    """Every (merge, reference frame) pair of the HDR+ bursts with a merge on disk, the newest results first."""
    from .dng import exposure_summary, read_tags, sensor_summary
    from .manifest import by_id, load_manifest
    base = Path(root or data_root()) / "raw" / HDRPLUS
    bursts = base / "bursts"
    if not bursts.is_dir():
        return []
    results = sorted((d for d in base.iterdir() if d.is_dir() and d.name.startswith("results_")), reverse=True)
    licence = by_id(load_manifest()).get(HDRPLUS)
    licence = licence.license if licence else ""
    pairs = []
    for b in sorted(p for p in bursts.iterdir() if p.is_dir()):
        ref = next((r / b.name / "merged.dng" for r in results if (r / b.name / "merged.dng").exists()), None)
        if ref is None:
            continue
        k = 0
        rf = ref.parent / "reference_frame.txt"
        if rf.exists():
            try:
                k = int(rf.read_text().strip())
            except ValueError:
                k = 0
        frame = b / f"payload_N{k:03d}.dng"
        is_ref_frame = frame.exists()
        if not is_ref_frame:
            frame = b / "payload_N000.dng"
            if not frame.exists():
                continue
        meta = {"results": ref.parent.parent.name, "reference_frame": k, "input_is_reference_frame": is_ref_frame}
        try:
            tags = read_tags(frame)
            meta.update(_plain(exposure_summary(tags)))
            meta["sensor"] = _plain(sensor_summary(tags))
        except Exception as ex:  # a frame whose tags do not parse still pairs; the noise law is measured anyway
            meta["tags_error"] = f"{type(ex).__name__}: {ex}"[:200]
        ref_rel = ref.relative_to(base).as_posix()
        pairs.append(Pair(HDRPLUS, b.name, ref_rel, frame.relative_to(base).as_posix(),
                          _split_of(Item(HDRPLUS, ref_rel)), iso=meta.get("iso"), licence=licence, meta=meta))
    return pairs


def _plain(v):
    """JSON-ready: numpy scalars and arrays to Python, anything else left alone."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def pairs_of(source: str, root: Path | None = None, log=None) -> list[Pair]:
    if source == RAWNIND:
        return rawnind_pairs(root, log)
    if source == HDRPLUS:
        return hdrplus_pairs(root, log)
    raise ValueError(f"no real pairs for {source!r}: one of {SOURCES}")


# ---- decoding, alignment, gain, noise law ------------------------------------------------------------------------

def decode_raw(path: Path, scale: int) -> tuple[np.ndarray, int]:
    """A raw through LibRaw as the labeller decodes it (camera white balance, linear, no brightening, 1.0 at
    the white level), box-downsampled by `scale`; with the sensor's bit depth from its white level."""
    import rawpy
    with rawpy.imread(str(path)) as raw:
        white = int(raw.white_level)
        rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True)
    img = np.asarray(rgb, dtype=np.float64) / 65535.0
    bits = int(math.ceil(math.log2(white + 1))) if white > 0 else 0
    return (box_downsample(img, scale) if scale > 1 else img), bits


def _norm_lum(a: np.ndarray) -> np.ndarray:
    lum = luminance(a)
    lum = lum - lum.mean()
    s = lum.std()
    return lum / s if s > 0 else lum


def _hann(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w))


def phase_shift(x: np.ndarray, y: np.ndarray, max_shift: int = MAX_SHIFT) -> tuple[int, int]:
    """(dy, dx) with x[i, j] ~ y[i + dy, j + dx], from the phase-correlation peak of the two luminances."""
    h, w = x.shape[:2]
    win = _hann(h, w)
    fx = np.fft.fft2(_norm_lum(x) * win)
    fy = np.fft.fft2(_norm_lum(y) * win)
    cross = fx * np.conj(fy)
    mag = np.abs(cross)
    cross /= np.where(mag > 1e-12, mag, 1.0)
    r = np.real(np.fft.ifft2(cross))
    # x(r) = y(r - p) at the peak p, so y[i + dy] with dy = -p; keep the peak within +-max_shift
    allowed = np.zeros_like(r, dtype=bool)
    m = min(max_shift, h // 2 - 1, w // 2 - 1)
    allowed[:m + 1, :m + 1] = allowed[:m + 1, w - m:] = allowed[h - m:, :m + 1] = allowed[h - m:, w - m:] = True
    r[~allowed] = -np.inf
    py, px = np.unravel_index(int(np.argmax(r)), r.shape)
    py = py - h if py > h // 2 else py
    px = px - w if px > w // 2 else px
    return -int(py), -int(px)


def overlap(x: np.ndarray, y: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    """The crops of x and y that coincide under x[i, j] ~ y[i + dy, j + dx]."""
    h, w = x.shape[:2]
    xs = x[max(0, -dy):min(h, h - dy), max(0, -dx):min(w, w - dx)]
    ys = y[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)]
    return xs, ys


def _nmse(x: np.ndarray, y: np.ndarray, dy: int, dx: int) -> float:
    xs, ys = overlap(x, y, dy, dx)
    if xs.size == 0:
        return math.inf
    a, b = luminance(xs), luminance(ys)
    a = a / max(a.mean(), 1e-9)
    b = b / max(b.mean(), 1e-9)
    return float(np.mean((a - b) ** 2))


def align(x: np.ndarray, y: np.ndarray) -> dict:
    """The shift of y against x: the phase-correlation peak, refined on the normalised MSE over +-REFINE px, with
    the quadrants' own peaks and the luminance correlation of the aligned overlap. `aligned` is False when a
    quadrant disagrees by more than QUADRANT_TOL px (rotation, parallax) or the overlap correlates below
    MIN_CORR (not the same framing)."""
    dy, dx = phase_shift(x, y)
    best = (dy, dx)
    best_err = _nmse(x, y, dy, dx)
    for ddy in range(-REFINE, REFINE + 1):
        for ddx in range(-REFINE, REFINE + 1):
            e = _nmse(x, y, dy + ddy, dx + ddx)
            if e < best_err:
                best, best_err = (dy + ddy, dx + ddx), e
    dy, dx = best
    h, w = x.shape[:2]
    quads = []
    for sy in (slice(0, h // 2), slice(h // 2, h)):
        for sx in (slice(0, w // 2), slice(w // 2, w)):
            quads.append(phase_shift(x[sy, sx], y[sy, sx]))
    spread = max(max(abs(q[0] - dy), abs(q[1] - dx)) for q in quads) if quads else 0.0
    xs, ys = overlap(x, y, dy, dx)
    a, b = luminance(xs).ravel(), luminance(ys).ravel()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 and a.std() > 0 and b.std() > 0 else 0.0
    return {"shift": (int(dy), int(dx)), "nmse": best_err, "quadrants": [list(q) for q in quads],
            "quadrant_spread": float(spread), "corr": corr,
            "aligned": bool(spread <= QUADRANT_TOL and corr >= MIN_CORR)}


def gain(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, int]:
    """Per-channel g = mean(y) / mean(x) over the pixels the reference holds in GAIN_RANGE and the input below
    the clip: the reference at the input's exposure and white balance is y / g."""
    lo, hi = GAIN_RANGE
    g = np.ones(3)
    n_min = xs.shape[0] * xs.shape[1]
    for c in range(3):
        m = (ys[..., c] > lo) & (ys[..., c] < hi) & (xs[..., c] < CLIP)
        n_min = min(n_min, int(m.sum()))
        if m.sum() >= 100:
            mx = float(xs[..., c][m].mean())
            if mx > 1e-6:
                g[c] = float(ys[..., c][m].mean()) / mx
    return g, n_min


def noise_law(xs: np.ndarray, t: np.ndarray, valid: np.ndarray) -> dict:
    """var(x - t) on flat, unclipped pixels binned by the target's luminance, fitted as a t + b (weighted least
    squares, weights sqrt of the bin counts): electrons at the clip 1/a, read noise sqrt(b)/a, both in the
    working scale's demosaiced RGB. The reference's own noise and demosaic differences sit in the residual too,
    so the electrons are a floor on the sensor's and the read noise a ceiling."""
    tl = luminance(t)
    flat = np.ones(tl.shape, dtype=bool)
    gy = np.abs(np.diff(tl, axis=0)) < FLAT_GRAD
    gx = np.abs(np.diff(tl, axis=1)) < FLAT_GRAD
    flat[1:] &= gy
    flat[:-1] &= gy
    flat[:, 1:] &= gx
    flat[:, :-1] &= gx
    m = flat & valid & (xs.max(axis=-1) < CLIP)
    edges = np.geomspace(0.005, 0.9, 17)
    idx = np.digitize(tl, edges)
    r = xs - t
    lv, vv, nn = [], [], []
    for b in range(1, len(edges)):
        sel = m & (idx == b)
        n = int(sel.sum())
        if n < MIN_BIN:
            continue
        lv.append(float(tl[sel].mean()))
        vv.append(float(np.mean(r[sel].var(axis=0))))
        nn.append(n)
    out = {"a": None, "b": None, "electrons_at_clip": None, "read_noise_e": None, "bins": len(lv),
           "levels": lv, "variances": vv, "flat_fraction": float(m.mean())}
    if len(lv) < 3:
        return out
    w = np.sqrt(np.asarray(nn, dtype=np.float64))
    design = np.stack([np.asarray(lv), np.ones(len(lv))], axis=1) * w[:, None]
    (a, b), *_ = np.linalg.lstsq(design, np.asarray(vv) * w, rcond=None)
    out.update(a=float(a), b=float(b))
    if a > 0:
        out["electrons_at_clip"] = float(1.0 / a)
        out["read_noise_e"] = float(math.sqrt(max(b, 0.0)) / a)
    return out


def match_pair(x: np.ndarray, y: np.ndarray) -> dict:
    """Input x and reference y (same scale) -> the aligned input crop, the gain-matched target, the valid mask
    (reference unclipped) and the measurements: alignment, gain, noise law."""
    al = align(x, y)
    dy, dx = al["shift"]
    xs, ys = overlap(x, y, dy, dx)
    g, n_gain = gain(xs, ys)
    t = np.clip(ys / g, 0.0, 1.0)
    valid = ys.max(axis=-1) < CLIP
    nl = noise_law(xs, t, valid) if al["aligned"] else {"a": None, "b": None, "electrons_at_clip": None,
                                                        "read_noise_e": None, "bins": 0}
    return {"x": xs, "t": t, "valid": valid, "align": al, "gain": g, "gain_pixels": n_gain, "noise": nl,
            "clipped": float(1.0 - valid.mean())}


def capture_of(noise: dict, bit_depth: int) -> np.ndarray:
    """The capture descriptor of a measured pair: Airy radius 0 (no source states its pixel pitch), no residual
    blur, the measured noise law, the sensor's bit depth, Bayer. Unknown when the noise law did not fit."""
    from ..model.data import UNKNOWN_CAPTURE, capture_descriptor
    e, read = noise.get("electrons_at_clip"), noise.get("read_noise_e")
    if not e:
        return UNKNOWN_CAPTURE.copy()
    return capture_descriptor(0.0, 0.0, float(e), float(read or 0.0), int(bit_depth), True)


# ---- the set -----------------------------------------------------------------------------------------------------

def set_path(source: str, seed: int, n: int, tile: int, scale: int, root: Path | None = None) -> Path:
    return Path(root or data_root()) / "derived" / REALPAIR_DIR / source / f"s{seed}_n{n}_t{tile}_k{scale}.npz"


def save_set(rs: RealPairSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"source": rs.source, "seed": rs.seed, "n": rs.n, "tile": rs.tile, "scale": rs.scale, "splits": list(rs.splits),
            "pairs": [asdict(p) for p in rs.pairs], "records": [asdict(r) for r in rs.records], "rejected": rs.rejected}
    np.savez(path, tiles=rs.tiles.astype(np.float32), targets=rs.targets.astype(np.float32),
             captures=rs.captures.astype(np.float32), meta=np.array(json.dumps(meta)))


def load_set(path: Path) -> RealPairSet:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        tiles, targets, captures = z["tiles"], z["targets"], z["captures"]
    records = [RealTile(**{**r, "shift": tuple(r["shift"]), "gain": tuple(r["gain"])}) for r in meta["records"]]
    return RealPairSet(meta["source"], meta["seed"], meta["n"], meta["tile"], meta["scale"], tuple(meta["splits"]),
                       [Pair(**p) for p in meta["pairs"]], tiles, targets, captures, records, meta["rejected"])


def rows_path(path: Path) -> Path:
    return path.with_name(path.stem + "_rows.parquet")


def tile_rows(rs: RealPairSet) -> list[dict]:
    """One label row per tile: the twelve metrics of the input against the target (m_*, as the labeller's rows)
    with the pair's identity and measurements."""
    rows = []
    for k, r in enumerate(rs.records):
        p = rs.pairs[r.pair]
        m = tile_metrics(rs.tiles[k].astype(np.float64), rs.targets[k].astype(np.float64))
        rows.append({"source": p.source, "scene": p.scene, "ref_file": p.ref_file, "in_file": p.in_file,
                     "split": p.split, "iso": p.iso, "licence": p.licence, "y0": r.y0, "x0": r.x0, "size": rs.tile,
                     "scale": rs.scale, "shift_y": r.shift[0], "shift_x": r.shift[1], "gain_r": r.gain[0],
                     "gain_g": r.gain[1], "gain_b": r.gain[2], "corr": r.corr, "clipped": r.clipped,
                     "bit_depth": r.bit_depth, "electrons_at_clip": r.noise.get("electrons_at_clip"),
                     "read_noise_e": r.noise.get("read_noise_e"), "noise_bins": r.noise.get("bins", 0),
                     **{f"m_{k2}": float(v) for k2, v in m.items()}})
    return rows


def _draw_tiles(m: dict, tile: int, want: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Up to `want` tile origins in the aligned overlap whose reference is unclipped (MAX_TILE_CLIPPED), each
    position tried TILE_DRAWS times; tiles do not repeat an origin."""
    h, w = m["x"].shape[:2]
    if h < tile or w < tile:
        return []
    valid = m["valid"]
    out, seen = [], set()
    for _ in range(want):
        for _ in range(TILE_DRAWS):
            y0 = int(rng.integers(0, h - tile + 1))
            x0 = int(rng.integers(0, w - tile + 1))
            if (y0, x0) in seen:
                continue
            if 1.0 - valid[y0:y0 + tile, x0:x0 + tile].mean() <= MAX_TILE_CLIPPED:
                out.append((y0, x0))
                seen.add((y0, x0))
                break
    return out


def build_set(source: str, n: int = 64, tile: int = 256, scale: int | None = None, seed: int = 0,
              splits: tuple[str, ...] = DEFAULT_SPLITS, root: Path | None = None, log=print,
              decode=decode_raw, pairs: list[Pair] | None = None) -> RealPairSet:
    """The cached set, or a new one: the pairs of `splits` in a seeded order, each decoded once at the clean
    scale (the manifest's, or `scale`), matched, and asked for ceil(n / pairs) tiles, with more passes over the
    pairs while the set is short. `decode(path, scale) -> (image, bit_depth)` is the raw decoder."""
    root = Path(root or data_root())
    log = log or (lambda *a: None)
    scale = int(scale or clean_settings(source).get("scale", 1))
    path = set_path(source, seed, n, tile, scale, root)
    if path.exists():
        rs = load_set(path)
        if tuple(rs.splits) == tuple(splits):
            return rs
        log(f"{path.name}: cached with splits {rs.splits}, rebuilding for {splits}")
    pairs = pairs if pairs is not None else pairs_of(source, root, log)
    chosen = [p for p in pairs if p.split in splits]
    if not chosen:
        raise ValueError(f"{source}: no pairs in splits {splits} ({len(pairs)} pairs on disk)")
    rng = np.random.default_rng(_seed("realpairs", source, seed, n, tile, scale))
    order = [int(i) for i in rng.permutation(len(chosen))]
    base = root / "raw" / source
    tiles, targets, captures, records, rejected = [], [], [], [], {}
    matched: dict[int, dict | None] = {}
    bits: dict[int, int] = {}
    passes = 0
    while len(tiles) < n and passes < 4:
        passes += 1
        want = max(1, math.ceil((n - len(tiles)) / max(1, len(order))))
        for i in order:
            if len(tiles) >= n:
                break
            p = chosen[i]
            if i not in matched:
                try:
                    x, bx = decode(base / p.in_file, scale)
                    y, _ = decode(base / p.ref_file, scale)
                    if x.shape != y.shape:
                        raise ValueError(f"shapes differ: input {x.shape}, reference {y.shape}")
                    m = match_pair(x, y)
                    if not m["align"]["aligned"]:
                        raise ValueError(f"not aligned: shift {m['align']['shift']}, quadrant spread "
                                         f"{m['align']['quadrant_spread']:.1f} px, corr {m['align']['corr']:.2f}")
                    matched[i], bits[i] = m, bx
                    log(f"{source}: {p.in_file} vs {p.ref_file}: shift {m['align']['shift']}, gain "
                        f"{np.round(m['gain'], 3).tolist()}, corr {m['align']['corr']:.3f}, electrons at clip "
                        f"{_fmt(m['noise'].get('electrons_at_clip'))}, read {_fmt(m['noise'].get('read_noise_e'))} e")
                except Exception as ex:
                    matched[i] = None
                    rejected[p.in_file] = f"{type(ex).__name__}: {ex}"[:300]
                    log(f"{source}: {p.in_file}: {rejected[p.in_file]}")
            m = matched[i]
            if m is None:
                continue
            for y0, x0 in _draw_tiles(m, tile, want, rng):
                tiles.append(m["x"][y0:y0 + tile, x0:x0 + tile].astype(np.float32))
                targets.append(m["t"][y0:y0 + tile, x0:x0 + tile].astype(np.float32))
                captures.append(capture_of(m["noise"], bits[i]))
                records.append(RealTile(i, y0, x0, tuple(m["align"]["shift"]), tuple(float(g) for g in m["gain"]),
                                        float(m["align"]["corr"]), m["noise"], int(bits[i]),
                                        float(1.0 - m["valid"][y0:y0 + tile, x0:x0 + tile].mean())))
                if len(tiles) >= n:
                    break
        if all(v is None for v in matched.values()):
            break
    if len(tiles) < n:
        raise ValueError(f"{source}: {len(tiles)} tile(s) of {n} from {sum(v is not None for v in matched.values())} "
                         f"usable pair(s) of {len(chosen)}; {len(rejected)} rejected: "
                         + "; ".join(f"{k}: {v[:60]}" for k, v in list(rejected.items())[:5]))
    rs = RealPairSet(source, seed, n, tile, scale, tuple(splits), chosen, np.stack(tiles), np.stack(targets),
                     np.stack(captures), records, rejected)
    save_set(rs, path)
    _write_shard(rows_path(path), tile_rows(rs))
    log(f"{source}: {n} tiles from {len({r.pair for r in records})} pairs of {len(chosen)} ({len(rejected)} rejected) "
        f"-> {path}")
    return rs


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.4g}"


def summarise(rs: RealPairSet) -> str:
    """A few lines: pairs, tiles, scenes, the measured noise range."""
    used = sorted({r.pair for r in rs.records})
    scenes = sorted({rs.pairs[i].scene for i in used})
    es = [r.noise["electrons_at_clip"] for r in rs.records if r.noise.get("electrons_at_clip")]
    isos = [rs.pairs[r.pair].iso for r in rs.records if rs.pairs[r.pair].iso]
    lines = [f"{rs.source}: {rs.n} tiles of {rs.tile} px at scale {rs.scale} from {len(used)} pairs in {len(scenes)} "
             f"scenes (splits {', '.join(rs.splits)}; {len(rs.pairs)} pairs listed, {len(rs.rejected)} rejected)"]
    if es:
        lines.append(f"  electrons at clip: {_fmt(float(np.min(es)))} to {_fmt(float(np.max(es)))} "
                     f"(median {_fmt(float(np.median(es)))}); {len(rs.records) - len(es)} tile(s) without a noise fit")
    if isos:
        lines.append(f"  ISO: {_fmt(float(np.min(isos)))} to {_fmt(float(np.max(isos)))}")
    for k, why in list(rs.rejected.items())[:6]:
        lines.append(f"  rejected {k}: {why[:100]}")
    return "\n".join(lines)


# ---- real against synthetic ------------------------------------------------------------------------------------

QUANTILES = (5, 25, 50, 75, 95)


def _quantiles(vals: list[float]) -> list[float] | None:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=np.float64)
    return [float(q) for q in np.percentile(v, QUANTILES)] if v.size else None


def compare(real_rows: list[dict], synthetic_rows: list[dict]) -> dict:
    """Per metric the quantiles of the real rows and of the synthetic rows (all, and the camera-on rows: the
    sampler's capture stage, the part a real pair exercises), and the fraction of real values inside the
    synthetic camera-on 1st to 99th percentile: the sampler's coverage of what the cameras did."""
    cam = [r for r in synthetic_rows if r.get("p_camera_on")]
    out = {"n_real": len(real_rows), "n_synthetic": len(synthetic_rows), "n_camera_on": len(cam), "metrics": {}}
    for m in STILL_METRICS:
        key = f"m_{m}"
        real = [r.get(key) for r in real_rows]
        syn = [r.get(key) for r in synthetic_rows]
        camv = np.asarray([r.get(key) for r in cam if r.get(key) is not None and np.isfinite(r.get(key))])
        cover = None
        rv = np.asarray([x for x in real if x is not None and np.isfinite(x)])
        if camv.size and rv.size:
            lo, hi = np.percentile(camv, [1, 99])
            cover = float(np.mean((rv >= lo) & (rv <= hi)))
        out["metrics"][m] = {"real": _quantiles(real), "synthetic": _quantiles(syn), "camera_on": _quantiles(camv.tolist()),
                             "coverage": cover}
    return out


def _pct(v) -> str:
    return "-" if v is None else f"{100 * v:.0f} %"


def compare_markdown(rep: dict, title: str = "") -> str:
    lines = [f"# Real against synthetic{': ' + title if title else ''}", "",
             f"{rep['n_real']} real tiles against {rep['n_synthetic']} synthetic label rows, {rep['n_camera_on']} of "
             f"them with the camera stage on. Quantiles {', '.join(str(q) for q in QUANTILES)} %; coverage is the "
             "fraction of real values inside the camera-on rows' 1st to 99th percentile.", "",
             "| metric | real | synthetic (camera on) | synthetic (all) | coverage |", "|---|---|---|---|---:|"]

    def q(v):
        return "-" if v is None else " / ".join(_fmt(x) for x in v)

    for m, s in rep["metrics"].items():
        lines.append(f"| {m} | {q(s['real'])} | {q(s['camera_on'])} | {q(s['synthetic'])} | "
                     f"{_pct(s['coverage'])} |")
    return "\n".join(lines) + "\n"
