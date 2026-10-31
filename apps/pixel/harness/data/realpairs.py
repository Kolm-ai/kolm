"""Real noisy/clean pairs from captures: DEGRADATION_PLAN step 4. python -m harness data realpairs.

The sampler's pairs are synthetic: a clean tile degraded by the camera model. A real pair is two captures of
one scene by one camera, an exposure whose noise the model should remove (the input) and a reference whose
own noise sits well below it (a long low-ISO exposure, or a merge of a burst). Three sources on disk give them:

- RawNIND (rawnind): per scene one or two low-ISO "GT" raws and an ISO sweep of noisy raws, on a tripod, one
  framing. The first GT on disk is the reference; the other GT and every noisy raw are inputs.
- HDR+ (hdr-plus-burst): per burst the frames and Google's merge of them (results_20171023, else the older
  results_20161014). The merge is the reference; the burst's reference frame (reference_frame.txt), or frame 0
  when that frame is not on disk, is the input.
- SI-HDR (si-hdr): per scene a tripod stack of five Canon 5D Mark III raws at one ISO, 3 EV apart (raw.zip). Each
  frame but the longest is an input; its reference is the merge of the stack's longer frames brought to its
  exposure (stack_reference): weighted by their noise, each left out where it is clipped or disagrees with the
  input beyond its noise (a ghost), the input itself never in it. The stack's noise law is measured on the
  frames against each other, so the input's is its own and not the pair's (decode_stack).

Both frames go through LibRaw as the labeller decodes raws (camera white balance, linear, no brightening) and the
source's clean scale (box_downsample). Then: the integer shift between them from phase correlation of their
luminance, refined over +-REFINE px on the normalised MSE, with the shift of each quadrant that holds a position
(its own peak correlates by MIN_CORR; a black, clipped or featureless quadrant's peak is noise) checked against the
whole (a disagreement past QUADRANT_TOL means rotation or parallax, and fewer than QUADRANT_MIN such quadrants too
little to tell: the pair is recorded as not aligned and not used); a per-channel gain from the means of the pixels
the reference holds unclipped, which brings the reference to the input's exposure and white balance (t = y / g,
clipped at 1); and the input's noise law against that target on flat unclipped pixels, var = a t + b, read as
electrons at the clip 1/a and read noise sqrt(b)/a.
These are effective numbers on demosaiced, white-balanced RGB at the working scale, what the model sees, not
the sensor's. They fill capture_descriptor() for the "known" evaluation; the Airy radius is 0 (unknown: no
source states its pixel pitch), which the report says.

A set is a function of (source, seed, n, tile, scale, splits): tiles drawn at random from the aligned, gain-
matched pairs of the chosen splits (val and test by default, scenes no run trains on; "test" is what the
manifest's clean.holdout names, RawNIND's test reserve), rejecting tiles with clipped reference pixels. It is
cached under <data_root>/derived/realpairs/<source>/ with a parquet of label rows beside it: the twelve tile
metrics of input against target plus scene, ISO, shift, gains and noise law, the real counterpart of the
labeller's synthetic rows. compare() puts the quantiles of those rows against the synthetic rows', so the
sampler's coverage of real degradations is a number.

HDR+ also holds, beside each merge, the phone's own finished photo of the burst (final.jpg, camera_file): the
best-case camera, which a restoration has to beat. decode_camera linearises it (sRGB) at the pair's scale;
camera_overlap aligns it to the merge by the same phase correlation and quadrant check and cuts it to the
pair's overlap; match_camera fits its rendering to the target on blurred pixels (CAMERA_FIT_BLUR_PX), a
per-channel isotonic tone curve then a 3x3 colour matrix by least squares, and takes out its local tone
mapping with a luminance gain at CAMERA_GAIN_SIGMA_PX clamped to CAMERA_GAIN_RANGE. What is left to compare is
the camera's detail, noise and artifacts, not its look. A set keeps the matched tile per sample (cameras,
has_camera) and the fit per scene (camera_fits: the shift, or why there is no photo); attach_cameras fills
them into a set cached before the photo was kept. heldout.score judges a model against it with
CAMERA_GATES."""
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
from .label import RAW_EXT, Item, _seed, _write_shard, clean_settings, file_filter, scene_of, split_member
from .subpix import MAX_SIDE, _centre, fourier_shift
from .subpix import measure as measure_residual
from .tilemetrics import tile_metrics

RAWNIND = "rawnind"
HDRPLUS = "hdr-plus-burst"
SIHDR = "si-hdr"
SOURCES = (RAWNIND, HDRPLUS, SIHDR)
STACK_SOURCES = (SIHDR,)         # a pair's reference is a merge of the input's stack (decode_stack), not a file
REALPAIR_DIR = "realpairs"       # <data_root>/derived/realpairs/<source>/s<seed>_n<n>_t<tile>_k<scale>.npz
DEFAULT_SPLITS = ("val", "test")
MAX_SHIFT = 64                   # px at the working scale a phase-correlation peak may claim
REFINE = 2                       # px of MSE refinement around the peak
QUADRANT_TOL = 1.0               # px the four quadrants' shifts may differ from the whole's
QUADRANT_MIN = 2                 # quadrants that hold a position (correlate by MIN_CORR at their own peak) a pair needs
MIN_CORR = 0.5                   # luminance correlation of the aligned overlap below which a pair is not a pair
SHAPE_CROP = 0.02                # share of a side by which a reference's frame may be cut from its input's
CLIP = 0.98                      # a pixel at or above this in any channel is clipped
GAIN_RANGE = (0.02, 0.8)         # reference pixels the gain is estimated on
FLAT_GRAD = 0.01                 # luminance step to a neighbour below which a pixel counts as flat
MIN_BIN = 500                    # flat pixels a noise-law bin needs
MAX_TILE_CLIPPED = 0.01          # fraction of clipped reference pixels a tile may hold
TILE_DRAWS = 24                  # positions tried per tile
CAMERA_FILE = "final.jpg"        # HDR+: the phone's own finished photo of the burst, beside merged.dng
CAMERA_FIT_BLUR_PX = 2.0         # px: the tone and colour fit reads both images blurred, so detail does not set it
CAMERA_GAIN_SIGMA_PX = 8.0       # px: the local gain that takes out the camera's local tone mapping
CAMERA_CURVE_BINS = 64           # quantile bins of the per-channel tone curve
CAMERA_GAIN_RANGE = (0.25, 4.0)  # the local gain's clamp
_RAWNIND_ISO = re.compile(r"_ISO([0-9]+)_sha1=", re.IGNORECASE)
ISO_UNKNOWN = 65535              # RawNIND's ISO label for a raw whose metadata gave none
SIHDR_STACKS = "raw.zip"         # SI-HDR: sihdr/raw/<scene>/<frame>.CR2, the frames of one stack per directory
STACK_TAG_BYTES = 1 << 16        # bytes of a zipped raw's head its exposure tags are read from
STACK_MIN_GAIN_PIXELS = 1000     # pixels a link's gain is measured on, at least, in every channel
STACK_GAIN_TOL = 1.5             # factor a link's measured gain may differ from its exposure times' ratio by
STACK_MAX_SHIFT = 4              # px at the working scale two frames of a stack may be offset by (a tripod's settle)
STACK_TAPER = 0.9                # a stack frame's weight falls linearly from here to 0 at CLIP (its brightest channel)
STACK_MAX_VAR = 0.25             # merged-reference noise variance, as a share of the input's, above which no reference
GHOST_BLUR = 5                   # px box whose luminance means the ghost test compares
GHOST_SIGMA = 4.0                # noise sigmas of those means a frame may differ from the input by...
GHOST_REL = 0.03                 # ...plus this share of the brighter mean, before it is a ghost there
GHOST_DILATE = 2                 # px a ghost grows by
STACK_CACHE = 2                  # decoded stacks a process keeps (one scene's pairs come one after another)
STACK_NOISE_BOX = 15             # px box whose mean a link's residual is high-passed by before its noise is read
STACK_FLAT_REL = 0.003           # gradient of the target's 5 px box-mean luminance, as a share of that mean (floored
#                                  at 0.005), below which a pixel is flat enough for a stack's noise law
STACK_NOISE_EDGES = np.geomspace(0.002, 0.9, 25)   # luminance bins of a stack's noise law
STACK_LAW_TOL = 0.15             # median relative misfit of a link's own a t + b law above which the link has moved
STACK_LAW_SPAN = 4.0             # ratio of the brightest bin a stack's law rests on to the darkest, at least
STACK_MAX_HOLE = 0.9             # share of a stack input's frame without a reference above which it is not a pair


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
    cameras: np.ndarray | None = None     # (n, tile, tile, 3) float32 the camera's own photo matched to the target
    has_camera: np.ndarray | None = None  # (n,) bool: the tile has a camera photo (HDR+ final.jpg) in bounds
    camera_fits: dict = field(default_factory=dict)  # pair index -> the camera match's measurements or its error
    subpix: dict | None = None      # a sub-pixel set's pair index -> the target's and the camera's residual (subpix.measure)


# ---- listing ---------------------------------------------------------------------------------------------------

def pair_scale(source: str) -> int:
    """The scale real pairs of `source` decode at: the manifest's clean.pair_scale, else its clean.scale (the
    labeller's). SI-HDR's merged EXR references are a third of its raws' size, so its labels and its pairs differ."""
    c = clean_settings(source)
    return int(c.get("pair_scale", c.get("scale", 1)) or 1)


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
    """Every (reference GT, input) pair of the RawNIND scenes with a GT on disk, from dataset.yaml; the raws only
    (the yaml lists a sidecar .xmp and a .jpg among the images, which LibRaw rejects). The split is "test" for the
    test reserve (the yaml's flag or test_reserve.yaml), else the reference's hashed split. A reserve scene the
    manifest's clean.holdout does not name (its GT raws would train) is a ValueError naming its scene key."""
    import yaml
    base = Path(root or data_root()) / "raw" / RAWNIND
    index = yaml.safe_load((base / "dataset.yaml").read_text(encoding="utf-8"))
    reserve = set()
    tr = base / "test_reserve.yaml"
    if tr.exists():
        reserve |= set((yaml.safe_load(tr.read_text(encoding="utf-8")) or {}).get("test_reserve", []))
    licences = _licences(base)
    on_disk = {p.name for p in base.iterdir() if p.is_file() and p.suffix.lower() in RAW_EXT}
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
                unmarked.append(f"{scene} (key {scene_of(Item(RAWNIND, ref))})")
            split = "test" if test else hashed
            meta = {"cfa": cfa, "unknown_sensor": bool(d.get("unknown_sensor")), "ref_iso": rawnind_iso(ref)}
            for f in cleans[1:] + noisy:
                pairs.append(Pair(RAWNIND, scene, ref, f, split, iso=rawnind_iso(f), licence=licences.get(scene, ""),
                                  meta=dict(meta)))
    if unmarked:   # the clean sampler would train on them: no pairs until clean.holdout names their keys
        raise ValueError(f"{RAWNIND}: {len(unmarked)} test-reserve scene(s) the manifest's clean.holdout does not "
                         f"name, so the clean sampler would train on them: {', '.join(unmarked[:8])}")
    return pairs


def hdrplus_pairs(root: Path | None = None, log=None) -> list[Pair]:
    """Every (merge, reference frame) pair of the HDR+ bursts with a merge on disk, the newest results first; a merge
    the source's `clean.exclude` names (the synthetic chart bursts) pairs nothing."""
    from .dng import exposure_summary, read_tags, sensor_summary
    from .manifest import by_id, load_manifest
    base = Path(root or data_root()) / "raw" / HDRPLUS
    bursts = base / "bursts"
    if not bursts.is_dir():
        return []
    results = sorted((d for d in base.iterdir() if d.is_dir() and d.name.startswith("results_")), reverse=True)
    licence = by_id(load_manifest()).get(HDRPLUS)
    licence = licence.license if licence else ""
    keep = file_filter(HDRPLUS)
    pairs = []
    for b in sorted(p for p in bursts.iterdir() if p.is_dir()):
        ref = next((r / b.name / "merged.dng" for r in results if (r / b.name / "merged.dng").exists()), None)
        if ref is None or not keep(ref.relative_to(base).as_posix()):
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


def sihdr_pairs(root: Path | None = None, log=None) -> list[Pair]:
    """Every (merge of the longer frames, frame) pair of the SI-HDR stacks in raw.zip: per scene directory the
    CR2 members ordered by exposure time (read from each member's head), each frame but the longest an input. The
    reference names the stack ("raw.zip::sihdr/raw/<scene>"), its frames in order in meta["frames"], the input's
    index in meta["frame"]; decode_stack and stack_reference make it."""
    import zipfile
    from .dng import exposure_summary, read_tags
    from .manifest import by_id, load_manifest
    base = Path(root or data_root()) / "raw" / SIHDR
    zp = base / SIHDR_STACKS
    if not zp.exists():
        return []
    entry = by_id(load_manifest()).get(SIHDR)
    licence = entry.license if entry else ""
    keep = file_filter(SIHDR)
    stacks: dict[str, list[str]] = {}
    pairs, skipped = [], {}
    with zipfile.ZipFile(zp) as z:
        for i in z.infolist():
            if not i.is_dir() and i.filename.lower().endswith(".cr2"):
                stacks.setdefault(i.filename.rsplit("/", 1)[0], []).append(i.filename)
        for d, members in sorted(stacks.items()):
            ref = f"{SIHDR_STACKS}::{d}"
            if not keep(ref):
                continue
            frames = []
            try:
                for m in members:
                    with z.open(m) as f:
                        e = exposure_summary(read_tags(f.read(STACK_TAG_BYTES)))
                    frames.append((float(e["exposure_s"]), m, e))
            except Exception as ex:  # a stack whose tags do not parse cannot be ordered: none of it pairs
                skipped[d] = f"{type(ex).__name__}: {ex}"[:200]
                continue
            frames.sort(key=lambda f: f[0])
            if len(frames) < 2 or len({f[0] for f in frames}) < len(frames):
                skipped[d] = f"{len(frames)} frame(s), exposures {[f[0] for f in frames]}"
                continue
            split = _split_of(Item(SIHDR, ref))
            files = [f"{SIHDR_STACKS}::{f[1]}" for f in frames]
            meta = {"frames": files, "exposures_s": [f[0] for f in frames],
                    "f_number": _plain(frames[0][2].get("f_number")), "model": _plain(frames[0][2].get("model"))}
            scene = scene_of(Item(SIHDR, ref))
            for k in range(len(frames) - 1):
                iso = frames[k][2].get("iso")
                pairs.append(Pair(SIHDR, scene, ref, files[k], split, iso=float(iso) if iso else None,
                                  licence=licence, meta=dict(meta, frame=k)))
    if skipped and log:
        log(f"{SIHDR}: {len(skipped)} stack(s) skipped: " + "; ".join(f"{k}: {v}" for k, v in list(skipped.items())[:5]))
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
    if source == SIHDR:
        return sihdr_pairs(root, log)
    raise ValueError(f"no real pairs for {source!r}: one of {SOURCES}")


# ---- decoding, alignment, gain, noise law ------------------------------------------------------------------------

def decode_raw(path: Path, scale: int) -> tuple[np.ndarray, int]:
    """A raw through LibRaw as the labeller decodes it (camera white balance, linear, no brightening, 1.0 at
    the white level), box-downsampled by `scale`; with the sensor's bit depth from its white level. A path
    "<zip>::<member>" is that member of the archive, read into memory."""
    import io
    import zipfile
    import rawpy
    disk, member = split_member(Path(path).as_posix())
    if member is not None:
        with zipfile.ZipFile(disk) as z:
            src = io.BytesIO(z.read(member))
    else:
        src = str(path)
    with rawpy.imread(src) as raw:
        white = int(raw.white_level)
        rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True)
    img = np.asarray(rgb, dtype=np.float64) / 65535.0
    bits = int(math.ceil(math.log2(white + 1))) if white > 0 else 0
    return (box_downsample(img, scale) if scale > 1 else img), bits


def _norm_lum(a: np.ndarray) -> np.ndarray:
    """An RGB image's luminance, or a luminance as given, less its mean over its std."""
    lum = luminance(a) if a.ndim == 3 else np.asarray(a, dtype=np.float64)
    lum = lum - lum.mean()
    s = lum.std()
    return lum / s if s > 0 else lum


def _hann(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w))


def phase_shift(x: np.ndarray, y: np.ndarray, max_shift: int = MAX_SHIFT) -> tuple[int, int]:
    """(dy, dx) with x[i, j] ~ y[i + dy, j + dx], from the phase-correlation peak of the two luminances (x and y
    RGB, or luminances already)."""
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


def _nmse(lx: np.ndarray, ly: np.ndarray, dy: int, dx: int) -> float:
    """The MSE of the two luminances' overlap at (dy, dx), each over its own mean."""
    a, b = overlap(lx, ly, dy, dx)
    if a.size == 0:
        return math.inf
    a = a / max(a.mean(), 1e-9)
    b = b / max(b.mean(), 1e-9)
    return float(np.mean((a - b) ** 2))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.size > 1 and a.std() > 0 and b.std() > 0 else 0.0


def align(x: np.ndarray, y: np.ndarray) -> dict:
    """The shift of y against x: the phase-correlation peak, refined on the normalised MSE over +-REFINE px, with
    the quadrants' own peaks and the luminance correlation of the aligned overlap. A quadrant holds a position when
    its overlap at its own peak correlates by MIN_CORR: a black, clipped or featureless one's peak is noise and
    says nothing (a stack's shortest frame is black over most of the frame, its longest clipped). `aligned` is
    False when a quadrant that holds one disagrees by more than QUADRANT_TOL px (rotation, parallax), fewer than
    QUADRANT_MIN hold one, or the overlap correlates below MIN_CORR (not the same framing); the spread is over
    the quadrants that hold one. The luminances are taken once and read by every step."""
    lx, ly = luminance(x), luminance(y)
    dy, dx = phase_shift(lx, ly)
    best = (dy, dx)
    best_err = _nmse(lx, ly, dy, dx)
    for ddy in range(-REFINE, REFINE + 1):
        for ddx in range(-REFINE, REFINE + 1):
            e = _nmse(lx, ly, dy + ddy, dx + ddx)
            if e < best_err:
                best, best_err = (dy + ddy, dx + ddx), e
    dy, dx = best
    h, w = x.shape[:2]
    quads, held = [], []
    for sy in (slice(0, h // 2), slice(h // 2, h)):
        for sx in (slice(0, w // 2), slice(w // 2, w)):
            q = phase_shift(lx[sy, sx], ly[sy, sx])
            quads.append(q)
            held.append(_corr(*overlap(lx[sy, sx], ly[sy, sx], *q)) >= MIN_CORR)
    spread = max((max(abs(q[0] - dy), abs(q[1] - dx)) for q, k in zip(quads, held) if k), default=0.0)
    corr = _corr(*overlap(lx, ly, dy, dx))
    return {"shift": (int(dy), int(dx)), "nmse": best_err, "quadrants": [list(q) for q in quads],
            "quadrants_held": int(sum(held)), "quadrant_spread": float(spread), "corr": corr,
            "aligned": bool(sum(held) >= QUADRANT_MIN and spread <= QUADRANT_TOL and corr >= MIN_CORR)}


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


def noise_law(xs: np.ndarray, t: np.ndarray, valid: np.ndarray, robust: bool = False) -> dict:
    """var(x - t) on flat, unclipped pixels binned by the target's luminance, fitted as a t + b (weighted least
    squares, weights sqrt of the bin counts): electrons at the clip 1/a, read noise sqrt(b)/a, both in the
    working scale's demosaiced RGB. The reference's own noise and demosaic differences sit in the residual too,
    so the electrons are a floor on the sensor's and the read noise a ceiling. `robust` reads each bin's variance
    as (1.4826 MAD)^2, the Gaussian's, so a minority of pixels that moved between the two (a stack's ghosts) do
    not set it."""
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
        if robust:
            rs = r[sel]
            mad = np.median(np.abs(rs - np.median(rs, axis=0)), axis=0)
            vv.append(float(np.mean((1.4826 * mad) ** 2)))
        else:
            vv.append(float(np.mean(r[sel].var(axis=0))))
        nn.append(n)
    out = {"a": None, "b": None, "electrons_at_clip": None, "read_noise_e": None, "bins": len(lv),
           "levels": lv, "variances": vv, "counts": nn, "flat_fraction": float(m.mean())}
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


# ---- exposure stacks --------------------------------------------------------------------------------------------

_STACKS: dict = {}


def _no_law() -> dict:
    return {"a": None, "b": None, "electrons_at_clip": None, "read_noise_e": None, "bins": 0}


def link_bins(xs: np.ndarray, t: np.ndarray, valid: np.ndarray) -> tuple[list, list, list, float]:
    """One stack link's noise bins: (levels, variances, counts, flat fraction). The residual x - t high-passed by
    its STACK_NOISE_BOX px box mean over the valid pixels (times sqrt(n / (n - 1)), n the valid pixels in the box,
    as white noise needs; a clipped pixel's residual is not noise and would shift its neighbours'), on pixels flat
    relative to their level (STACK_FLAT_REL), valid, and with at least half their box valid, binned by the target's
    luminance (STACK_NOISE_EDGES, MIN_BIN pixels a bin), each bin's variance (1.4826 MAD)^2 per channel, the
    channels' mean. Two frames of a stack differ by more than noise: their gain drifts smoothly over the frame and
    with level (the mean residual by up to 2%), and a sub-pixel misregistration leaves the target's gradient in the
    residual; a plain variance reads both as noise growing with t^2. The box takes out the first, the relative flat
    test the second; the demosaic correlates neighbours a pixel or two apart, which a box this wide leaves alone (a
    3 px box read the variance 1.4-1.7x low, this one within 3% of the plain variance where both are noise)."""
    from scipy.ndimage import uniform_filter
    tl = luminance(t).astype(np.float64)
    sm = uniform_filter(tl, size=5, mode="nearest")
    gy, gx = np.gradient(sm)
    v = valid.astype(np.float64)
    n = uniform_filter(v, size=STACK_NOISE_BOX, mode="nearest") * STACK_NOISE_BOX ** 2
    m = valid & (np.hypot(gy, gx) < STACK_FLAT_REL * np.maximum(sm, 0.005)) & (n >= STACK_NOISE_BOX ** 2 / 2)
    r = (np.asarray(xs, dtype=np.float64) - t) * v[..., None]
    n = np.maximum(n, 2.0)
    mean = np.stack([uniform_filter(r[..., c], size=STACK_NOISE_BOX, mode="nearest") for c in range(r.shape[-1])],
                    axis=-1) * (STACK_NOISE_BOX ** 2 / n)[..., None]
    hp = (r - mean) * np.sqrt(n / (n - 1.0))[..., None]
    idx = np.digitize(tl, STACK_NOISE_EDGES)
    lv, vv, nn = [], [], []
    for b in range(1, len(STACK_NOISE_EDGES)):
        sel = m & (idx == b)
        n = int(sel.sum())
        if n < MIN_BIN:
            continue
        rs = hp[sel]
        mad = np.median(np.abs(rs - np.median(rs, axis=0)), axis=0)
        lv.append(float(tl[sel].mean()))
        vv.append(float(np.mean((1.4826 * mad) ** 2)))
        nn.append(n)
    return lv, vv, nn, float(m.mean())


def _fit_link_law(lv, vv, nn, gg) -> tuple[float, float, float]:
    """(a, b, median relative misfit) of var = a t (1 + 1 / G) + b (1 + 1 / G^2) over the bins, weighted least
    squares with weights sqrt of the counts: the shorter frame's noise a t + b plus the longer one's over its gain."""
    lv, vv, nn, gg = (np.asarray(c, dtype=np.float64) for c in (lv, vv, nn, gg))
    design = np.stack([lv * (1.0 + 1.0 / gg), 1.0 + 1.0 / gg ** 2], axis=1)
    w = np.sqrt(nn)
    (a, b), *_ = np.linalg.lstsq(design * w[:, None], vv * w, rcond=None)
    misfit = float(np.median(np.abs(design @ np.array([a, b]) - vv) / np.maximum(vv, 1e-20)))
    return float(a), float(b), misfit


def stack_noise(frames: list[np.ndarray], links: list[dict]) -> dict:
    """The stack's noise law, one for every frame (one ISO, so one gain and read noise in the decoder's units; the
    camera's white balance differs by scene, so the law in its white-balanced RGB is the stack's, not the camera's):
    on each link that holds, frame j against the next over the link's gain G at the link's shift (link_bins), whose
    own noise adds a t / G + b / G^2 to a t + b. A link whose bins its own law misses by more than STACK_LAW_TOL (a
    scene that moved between the two: the residual is no longer noise) is left out; the bins of the rest fitted at
    once (_fit_link_law), and no law unless they span STACK_LAW_SPAN in level with a > 0. Each link's own law and
    whether it was used are recorded on it (ln["noise"]). Electrons at the clip 1/a, read noise sqrt(b)/a, as
    noise_law."""
    rows, flats = [], []
    for j, ln in enumerate(links):
        if not ln["ok"]:
            continue
        g = np.asarray(ln["gain"], dtype=np.float64)
        xa, ya = overlap(frames[j], frames[j + 1], *ln["shift"])
        valid = (ya.max(axis=-1) < CLIP) & (xa.max(axis=-1) < CLIP)
        lv, vv, nn, flat = link_bins(xa, np.clip(ya / g, 0.0, 1.0), valid)
        gm = float(g.mean())
        own = {"bins": len(lv), "electrons_at_clip": None, "misfit": None, "used": False}
        if len(lv) >= 3:
            a, b, misfit = _fit_link_law(lv, vv, nn, [gm] * len(lv))
            own.update(electrons_at_clip=float(1.0 / a) if a > 0 else None, misfit=misfit,
                       used=bool(a > 0 and misfit <= STACK_LAW_TOL))
        ln["noise"] = own
        if own["used"]:
            flats.append(flat)
            rows += [(lv_, vv_, n_, gm) for lv_, vv_, n_ in zip(lv, vv, nn)]
    out = dict(_no_law(), bins=len(rows), levels=[r[0] for r in rows], variances=[r[1] for r in rows],
               counts=[r[2] for r in rows], flat_fraction=float(np.mean(flats)) if flats else 0.0,
               links=len(flats))
    if len(rows) < 3 or max(r[0] for r in rows) < STACK_LAW_SPAN * min(r[0] for r in rows):
        return out
    a, b, misfit = _fit_link_law(*zip(*rows))
    out.update(a=a, b=b, misfit=misfit)
    if a > 0:
        out["electrons_at_clip"] = float(1.0 / a)
        out["read_noise_e"] = float(math.sqrt(max(b, 0.0)) / a)
    return out


def decode_stack(base: Path, ref_file: str, frames: list[str], scale: int, decode=decode_raw,
                 exposures: list[float] | None = None) -> dict:
    """The frames of one exposure stack (`frames` in exposure order, relative to `base`) decoded at `scale` as
    float32, and measured: each frame against the next, the shift (align, both frames brought to the longer one's
    exposure and clipped where it clips, so its saturated areas read alike in both) and per-channel gain (gain, on
    their overlap). A link holds when the two align within STACK_MAX_SHIFT px (a tripod stack settles by a pixel or
    two; a merge shifts frames by whole pixels and is never resampled), its gain rests on STACK_MIN_GAIN_PIXELS in
    every channel, and, given the `exposures`, its mean is within STACK_GAIN_TOL of their ratio. cum[j] is frame j's
    gain to frame 0, the product of the links' (frame j ~ cum[j] x frame 0), offset[j] its shift (frame 0 at p ~
    frame j at p + offset[j]); the noise law is stack_noise's. The last STACK_CACHE stacks are kept, so a scene's
    inputs decode it once."""
    key = (str(base / ref_file), int(scale), decode)
    if key in _STACKS:
        st = _STACKS.pop(key)
        _STACKS[key] = st
        return st
    xs, bits = [], 0
    for f in frames:
        x, b = decode(base / f, scale)
        xs.append(np.asarray(x, dtype=np.float32))
        bits = max(bits, int(b))
    if len({x.shape for x in xs}) != 1:
        raise ValueError(f"stack frames differ in shape: {sorted({x.shape for x in xs})}")
    links, cum, offset = [], [np.ones(3)], [(0, 0)]
    for j in range(len(xs) - 1):
        ratio = float(exposures[j + 1] / exposures[j]) if exposures else None
        r = ratio if ratio else float(gain(xs[j], xs[j + 1])[0].mean())
        al = align(np.minimum(xs[j] * r, CLIP), np.minimum(xs[j + 1], CLIP))
        dy, dx = al["shift"]
        g, n_gain = gain(*overlap(xs[j], xs[j + 1], dy, dx))
        ok = bool(al["aligned"] and max(abs(dy), abs(dx)) <= STACK_MAX_SHIFT and n_gain >= STACK_MIN_GAIN_PIXELS
                  and (ratio is None or 1.0 / STACK_GAIN_TOL <= float(g.mean()) / ratio <= STACK_GAIN_TOL))
        links.append({"shift": [int(dy), int(dx)], "corr": float(al["corr"]),
                      "quadrant_spread": float(al["quadrant_spread"]), "quadrants_held": al["quadrants_held"],
                      "gain": [float(v) for v in g],
                      "gain_pixels": int(n_gain), "exposure_ratio": ratio, "ok": ok})
        cum.append(cum[-1] * g)
        offset.append((offset[-1][0] + int(dy), offset[-1][1] + int(dx)))
    st = {"frames": xs, "bits": bits, "links": links, "cum": cum, "offset": offset,
          "noise": stack_noise(xs, links)}
    _STACKS[key] = st
    while len(_STACKS) > STACK_CACHE:
        _STACKS.pop(next(iter(_STACKS)))
    return st


def stack_run(st: dict, k: int) -> list[int]:
    """The frames longer than k chained to it by links that hold."""
    out = []
    for j in range(k, len(st["links"])):
        if not st["links"][j]["ok"]:
            break
        out.append(j + 1)
    return out


def _box(a: np.ndarray, size: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    return uniform_filter(np.asarray(a, dtype=np.float64), size=size, mode="nearest")


def _shifted(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """a in another frame's framing: out[i, j] = a[i + dy, j + dx], NaN where that falls outside a."""
    out = np.full(a.shape, np.nan, dtype=np.float32)
    o, src = overlap(out, a, dy, dx)
    o[...] = src
    return out


def stack_reference(st: dict, k: int) -> tuple[np.ndarray, dict]:
    """The reference of frame k of a decoded stack, at k's exposure and in its framing: every longer frame j chained
    to k (stack_run), moved by its offset to k (whole pixels) and brought to k's exposure (z = x_j / G, G = cum[j] /
    cum[k] per channel), averaged with the inverse of its noise variance, G^2 / (a L_j + b) (L_j the GHOST_BLUR px
    box mean of its luminance, a and b the stack's law), tapered to 0 from STACK_TAPER to CLIP in its brightest
    channel. A frame is left out where it is a ghost: its box mean differs from the input's by more than GHOST_SIGMA
    sigmas of the two means' noise plus GHOST_REL of the brighter, grown by GHOST_DILATE px (not tested where either
    frame nears the clip, where the means do not compare), and where its offset leaves the frame. Frame k is never
    in its own reference, so the target's noise is independent of the input's; nor are the shorter frames (8x darker
    at 3 EV, they would add a 64th of the weight where a longer frame holds, and no reference where none does). A
    pixel no frame holds, or whose merged noise variance exceeds STACK_MAX_VAR of the input's, is NaN: there is no
    reference there. Returns the reference (H, W, 3) float32 and what was used."""
    from scipy.ndimage import maximum_filter
    law = st["noise"]
    a, b = law["a"], law["b"]
    if a is None or a <= 0:
        raise ValueError(f"stack has no noise law: {law['bins']} bin(s) over {law.get('links', 0)} link(s)")
    b = max(float(b), 0.0)
    xk = st["frames"][k]
    lk = _box(luminance(xk), GHOST_BLUR)
    var_k = a * np.maximum(lk, 0.0) + b
    near_k = maximum_filter(xk.max(axis=-1), size=GHOST_BLUR) >= STACK_TAPER
    run = stack_run(st, k)
    num = np.zeros(xk.shape, dtype=np.float64)
    den = np.zeros(xk.shape[:2], dtype=np.float64)
    var_num = np.zeros(xk.shape[:2], dtype=np.float64)
    info = {"frame": k, "run": run, "offsets": [], "ghost": [], "gains": []}
    ok, ok_dy = st["offset"][k]
    for j in run:
        dy, dx = st["offset"][j][0] - ok, st["offset"][j][1] - ok_dy
        xj = _shifted(st["frames"][j], dy, dx)
        inside = ~np.isnan(xj).any(axis=-1)
        xj = np.nan_to_num(xj, nan=0.0)
        whole = _box(inside, GHOST_BLUR) > 1.0 - 1e-6      # the ghost test's box lies inside the shifted frame
        g = np.asarray(st["cum"][j], dtype=np.float64) / np.asarray(st["cum"][k], dtype=np.float64)
        gm = float(g.mean())
        lj = _box(luminance(xj), GHOST_BLUR)
        var_z = (a * np.maximum(lj, 0.0) + b) / gm ** 2
        lz = lj / gm
        near = near_k | (maximum_filter(xj.max(axis=-1), size=GHOST_BLUR) >= STACK_TAPER)
        tol = GHOST_SIGMA * np.sqrt((var_z + var_k) / GHOST_BLUR ** 2) + GHOST_REL * np.maximum(lz, lk)
        ghost = (np.abs(lz - lk) > tol) & ~near
        if GHOST_DILATE:
            ghost = maximum_filter(ghost, size=2 * GHOST_DILATE + 1)
        taper = np.clip((CLIP - xj.max(axis=-1)) / (CLIP - STACK_TAPER), 0.0, 1.0)
        w = np.where(ghost | ~whole, 0.0, taper / np.maximum(var_z, 1e-12))
        num += w[..., None] * (xj / g)
        den += w
        var_num += w ** 2 * var_z
        info["offsets"].append([int(dy), int(dx)])
        info["ghost"].append(float((ghost & whole).mean()))
        info["gains"].append([float(v) for v in g])
    held = den > 0
    safe = np.where(held, den, 1.0)
    var_ref = np.where(held, var_num / safe ** 2, np.inf)
    hole = ~held | (var_ref > STACK_MAX_VAR * var_k)
    ref = num / safe[..., None]
    ref[hole] = np.nan
    info["hole"] = float(hole.mean())
    info["var_ratio_median"] = float(np.median((var_ref / var_k)[~hole])) if (~hole).any() else None
    return ref.astype(np.float32), info


def match_stack(x: np.ndarray, ref: np.ndarray) -> dict:
    """match_pair against a stack reference (stack_reference: NaN where it holds none), with the input in its holes,
    which are then invalid as clipped pixels are. The merge is in the input's framing, so any shift but zero is a
    failure (not aligned)."""
    hole = np.isnan(ref).any(axis=-1)
    m = match_pair(x, np.where(hole[..., None], x, ref))
    if tuple(m["align"]["shift"]) != (0, 0):
        m["align"]["aligned"] = False
    else:
        m["valid"] = m["valid"] & ~hole
        m["clipped"] = float(1.0 - m["valid"].mean())
        m["noise"] = noise_law(m["x"], m["t"], m["valid"])
    m["hole"] = float(hole.mean())
    return m


def match_of(p: Pair, base: Path, scale: int, decode=decode_raw,
             ref: np.ndarray | None = None) -> tuple[dict, np.ndarray, int, np.ndarray]:
    """One pair decoded and matched: (match_pair's dict, the input as decoded, its bit depth, the reference as
    decoded). A stack source's reference is stack_reference of its stack (decode_stack), not a pair where it holds
    less than 1 - STACK_MAX_HOLE of the frame (the tiles drawn from it would be few), matched by match_stack,
    with the stack's noise law as the pair's (m["noise"]; the pair's own residual, which holds the reference's
    noise too, in m["noise_pair"]) and what the merge used in m["stack"]; any other source's is the reference raw,
    decoded here unless `ref` gives it, with the input brought to its frame (input_fits; of several ways, the one
    that aligns with the higher correlation; how, in m["fit"], when it was cut or turned, and the input returned
    so). Raises when the pair is not one (no fit of the shapes, not aligned)."""
    if p.source in STACK_SOURCES:
        k = int(p.meta["frame"])
        st = decode_stack(base, p.ref_file, p.meta["frames"], scale, decode, p.meta.get("exposures_s"))
        if not stack_run(st, k):
            ln = st["links"][k]
            raise ValueError(f"frame {k} not chained to a longer one: shift {ln['shift']}, quadrant spread "
                             f"{ln['quadrant_spread']:.1f} px over {ln['quadrants_held']}, corr {ln['corr']:.2f}, gain "
                             f"{np.round(ln['gain'], 2).tolist()} on {ln['gain_pixels']} px, exposure ratio "
                             f"{ln['exposure_ratio']}")
        y, info = stack_reference(st, k)
        if info["hole"] > STACK_MAX_HOLE:
            raise ValueError(f"frame {k}'s reference holds {1.0 - info['hole']:.1%} of the frame, under "
                             f"{1.0 - STACK_MAX_HOLE:.0%}")
        x, bx = st["frames"][k], st["bits"]
        m = match_stack(x, y)
        m["noise_pair"], m["noise"] = m["noise"], st["noise"]
        m["stack"] = dict(info, links=st["links"])
    else:
        x, bx = decode(base / p.in_file, scale)
        y = ref if ref is not None else decode(base / p.ref_file, scale)[0]
        fits = input_fits(x, y)
        if not fits:
            raise ValueError(f"shapes differ: input {x.shape}, reference {y.shape}")
        shape_in, best = [int(v) for v in x.shape[:2]], None
        for xf, fit in fits:
            mf = match_pair(xf, y)
            if best is None or (mf["align"]["aligned"], mf["align"]["corr"]) > (best[0]["align"]["aligned"],
                                                                                best[0]["align"]["corr"]):
                best = (mf, xf, fit)
        m, x, fit = best
        if fit["turn"] or any(fit["crop"]):
            m["fit"] = dict(fit, shape_in=shape_in)
    al = m["align"]
    if not al["aligned"]:
        raise ValueError(f"not aligned: shift {al['shift']}, quadrant spread {al['quadrant_spread']:.1f} px "
                         f"over {al['quadrants_held']}, corr {al['corr']:.2f}")
    return m, x, int(bx), y


def input_fits(x: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, dict]]:
    """The ways the input x can be brought to the reference y's frame, each (x so brought, {"turn": quarter turns
    np.rot90 gave it, "crop": [rows, columns] cut from its top and left}): as it is when the shapes agree; else
    centre-cut to y's shape where y is the smaller by at most SHAPE_CROP of each side (HDR+'s merges drop a few
    edge rows or columns of their bursts' frames: 4 of 1522 rows, 24 of 2040 columns at scale 2; the offset the cut
    leaves is the alignment's), and failing that turned a quarter either way and cut so (the two files' orientation
    tags read differently). The reference is never cut or turned, so a scene keeps one reference for all of its
    inputs. Empty when nothing fits."""
    (hy, wy) = y.shape[:2]
    out = []
    for k in (0, 1, 3):
        xt = np.rot90(x, k) if k else x
        h, w = xt.shape[:2]
        if hy <= h <= hy * (1 + SHAPE_CROP) and wy <= w <= wy * (1 + SHAPE_CROP):
            r0, c0 = (h - hy) // 2, (w - wy) // 2
            xt = xt if (h, w) == (hy, wy) else xt[r0:r0 + hy, c0:c0 + wy]
            out.append((np.ascontiguousarray(xt), {"turn": k, "crop": [r0, c0]}))
        if k == 0 and out:
            break
    return out


def capture_of(noise: dict, bit_depth: int) -> np.ndarray:
    """The capture descriptor of a measured pair: Airy radius 0 (no source states its pixel pitch), no residual
    blur, the measured noise law, the sensor's bit depth, Bayer. Unknown when the noise law did not fit."""
    from ..model.data import UNKNOWN_CAPTURE, capture_descriptor
    e, read = noise.get("electrons_at_clip"), noise.get("read_noise_e")
    if not e:
        return UNKNOWN_CAPTURE.copy()
    return capture_descriptor(0.0, 0.0, float(e), float(read or 0.0), int(bit_depth), True)


# ---- the camera's own photo -------------------------------------------------------------------------------------

def camera_file(p: Pair) -> str | None:
    """The camera's finished photo of the pair's scene, relative to the source's root: HDR+'s final.jpg beside the
    merge. RawNIND has none (its inputs are raws, and no camera rendering of them was published)."""
    if p.source != HDRPLUS:
        return None
    return (Path(p.ref_file).parent / CAMERA_FILE).as_posix()


def decode_camera(path: Path, scale: int) -> np.ndarray:
    """An 8-bit sRGB photo as linear RGB in [0, 1] (the sRGB curve undone), box-downsampled by `scale`."""
    from PIL import Image
    from ..colorspace import srgb_to_linear
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    lin = np.asarray(srgb_to_linear(rgb), dtype=np.float64)
    return box_downsample(lin, scale) if scale > 1 else lin


def _isotonic(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool adjacent violators: the non-decreasing sequence nearest y in weighted least squares."""
    vals, wts, cnt = [], [], []
    for v, ww in zip(y.tolist(), w.tolist()):
        vals.append(v)
        wts.append(ww)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            wsum = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / max(wsum, 1e-12)
            wts[-2], cnt[-2] = wsum, cnt[-2] + cnt[-1]
            vals.pop()
            wts.pop()
            cnt.pop()
    return np.repeat(np.asarray(vals, dtype=np.float64), cnt)


def _blur_masked(a: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a over the pixels mask holds, normalised by the blurred mask (no bleed from holes)."""
    from scipy.ndimage import gaussian_filter
    m = gaussian_filter(mask.astype(np.float64), sigma)
    if a.ndim == 3:
        a = np.where(mask[..., None], a, 0.0)
        b = np.stack([gaussian_filter(a[..., c], sigma) for c in range(a.shape[-1])], axis=-1)
        return b / np.maximum(m, 1e-9)[..., None]
    return gaussian_filter(np.where(mask, a, 0.0), sigma) / np.maximum(m, 1e-9)


def match_camera(c: np.ndarray, t: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """The camera's photo c (linear, aligned to the target t, NaN where it has no pixel) brought to t's rendering
    intent with no more than undoes the camera's own rendering, so what is left is its detail, noise and
    artifacts: a monotone tone curve per channel and a 3x3 colour matrix, both fitted on the two images blurred
    by CAMERA_FIT_BLUR_PX over pixels both hold unclipped, then a local luminance gain at CAMERA_GAIN_SIGMA_PX
    for the local tone mapping. The fit reads the truth, so the camera's tone, colour and shading metrics sit at
    or near the truth's values by construction; its fine structure is its own."""
    have = np.isfinite(c).all(axis=-1)
    c0 = np.where(have[..., None], c, 0.0)
    cb, tb = _blur_masked(c0, have, CAMERA_FIT_BLUR_PX), _blur_masked(t, have, CAMERA_FIT_BLUR_PX)
    sel = have & valid & (c0.max(axis=-1) < CLIP) & (t.max(axis=-1) < CLIP)
    if sel.sum() < 1000:
        raise ValueError(f"{int(sel.sum())} pixels to fit the camera's rendering on")
    curved = np.empty_like(c0)
    curves = []
    for ch in range(3):
        xv, yv = cb[..., ch][sel], tb[..., ch][sel]
        edges = np.unique(np.quantile(xv, np.linspace(0.0, 1.0, CAMERA_CURVE_BINS + 1)))
        if len(edges) < 3:
            raise ValueError(f"channel {ch} of the camera photo is flat")
        idx = np.clip(np.searchsorted(edges, xv, side="right") - 1, 0, len(edges) - 2)
        n = np.bincount(idx, minlength=len(edges) - 1).astype(np.float64)
        keep = n > 0
        xm = np.bincount(idx, xv, minlength=len(edges) - 1)[keep] / n[keep]
        ym = _isotonic(np.bincount(idx, yv, minlength=len(edges) - 1)[keep] / n[keep], n[keep])
        curved[..., ch] = np.interp(c0[..., ch], xm, ym)
        curves.append((xm, ym))
    cbc = np.stack([np.interp(cb[..., ch], *curves[ch]) for ch in range(3)], axis=-1)
    pix = np.flatnonzero(sel.ravel())
    if pix.size > 200_000:
        pix = np.random.default_rng(0).choice(pix, 200_000, replace=False)
    M, *_ = np.linalg.lstsq(cbc.reshape(-1, 3)[pix], tb.reshape(-1, 3)[pix], rcond=None)
    cm = curved @ M
    lt, lc = luminance(t), luminance(np.clip(cm, 0.0, None))
    g = _blur_masked(lt, have, CAMERA_GAIN_SIGMA_PX) / np.maximum(_blur_masked(lc, have, CAMERA_GAIN_SIGMA_PX), 1e-9)
    g = np.clip(g, *CAMERA_GAIN_RANGE)
    out = np.clip(cm * g[..., None], 0.0, 1.0)
    fit = luminance(_blur_masked(out, have, CAMERA_FIT_BLUR_PX))
    resid = float(np.sqrt(np.mean((fit[sel] - luminance(tb)[sel]) ** 2)))
    out[~have] = np.nan
    return out, {"matrix": M.tolist(), "gain_p5_p95": [float(np.percentile(g[have], 5)),
                                                      float(np.percentile(g[have], 95))],
                 "fit_pixels": int(sel.sum()), "residual_rms": resid}


def camera_overlap(m: dict, y: np.ndarray, cam: np.ndarray) -> tuple[np.ndarray, dict]:
    """The camera photo aligned to the reference y (the pair's phase correlation and checks), cut to the pair's
    overlap and matched to its target: an array the shape of m["t"], NaN where the camera has no pixel. Raises
    when the photo is another size or does not align with the reference."""
    if cam.shape != y.shape:
        raise ValueError(f"camera photo {cam.shape[1]}x{cam.shape[0]}, reference {y.shape[1]}x{y.shape[0]}")
    al = align(y, cam)
    if not al["aligned"]:
        raise ValueError(f"camera photo not aligned: shift {al['shift']}, quadrant spread "
                         f"{al['quadrant_spread']:.1f} px over {al['quadrants_held']}, corr {al['corr']:.2f}")
    cdy, cdx = al["shift"]
    h, w = y.shape[:2]
    on_y = np.full(y.shape, np.nan)
    on_y[max(0, -cdy):min(h, h - cdy), max(0, -cdx):min(w, w - cdx)] = \
        cam[max(0, cdy):min(h, h + cdy), max(0, cdx):min(w, w + cdx)]
    dy, dx = m["align"]["shift"]
    c_ov = on_y[max(0, dy):min(h, h + dy), max(0, dx):min(w, w + dx)]
    out, info = match_camera(c_ov, m["t"], m["valid"])
    info.update(shift=[int(cdy), int(cdx)], corr=al["corr"], quadrant_spread=al["quadrant_spread"])
    return out, info


def _camera_tile(c_ov: np.ndarray | None, y0: int, x0: int, tile: int) -> np.ndarray | None:
    if c_ov is None:
        return None
    t = c_ov[y0:y0 + tile, x0:x0 + tile]
    return t.astype(np.float32) if t.shape[:2] == (tile, tile) and np.isfinite(t).all() else None


def _pack_cameras(cams: list, tile: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    has = np.array([c is not None for c in cams], dtype=bool)
    if not has.any():
        return None, None
    z = np.zeros((tile, tile, 3), dtype=np.float32)
    return np.stack([c if c is not None else z for c in cams]), has


def attach_cameras(rs: RealPairSet, root: Path | None = None, decode=decode_raw, log=print) -> RealPairSet:
    """The camera photo for every tile of a set cached before cameras were kept: each used pair decoded and matched
    again (deterministic: the same shift and target), its camera photo aligned and matched, and the tiles cut at
    the recorded origins. The set's tiles and targets are unchanged."""
    root = Path(root or data_root())
    base = root / "raw" / rs.source
    log = log or (lambda *a: None)
    per_pair: dict[int, np.ndarray | None] = {}
    fits: dict = {}
    for i in sorted({r.pair for r in rs.records}):
        p = rs.pairs[i]
        cf = camera_file(p)
        per_pair[i] = None
        if cf is None or not (base / cf).exists():
            fits[str(i)] = {"error": "no camera photo on disk"}
            continue
        try:
            x, _ = decode(base / p.in_file, rs.scale)
            y, _ = decode(base / p.ref_file, rs.scale)
            per_pair[i], fits[str(i)] = camera_overlap(match_pair(x, y), y, decode_camera(base / cf, rs.scale))
        except Exception as ex:
            fits[str(i)] = {"error": f"{type(ex).__name__}: {ex}"[:300]}
            log(f"{rs.source}: camera for {p.in_file}: {fits[str(i)]['error']}")
    cams = [_camera_tile(per_pair[r.pair], r.y0, r.x0, rs.tile) for r in rs.records]
    rs.cameras, rs.has_camera = _pack_cameras(cams, rs.tile)
    rs.camera_fits = fits
    log(f"{rs.source}: camera photo on {sum(c is not None for c in cams)} of {rs.n} tiles")
    return rs


# ---- the set -----------------------------------------------------------------------------------------------------

def set_path(source: str, seed: int, n: int, tile: int, scale: int, root: Path | None = None,
             subpix: bool = False) -> Path:
    sp = "_sp" if subpix else ""
    return Path(root or data_root()) / "derived" / REALPAIR_DIR / source / f"s{seed}_n{n}_t{tile}_k{scale}{sp}.npz"


def save_set(rs: RealPairSet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"source": rs.source, "seed": rs.seed, "n": rs.n, "tile": rs.tile, "scale": rs.scale, "splits": list(rs.splits),
            "pairs": [asdict(p) for p in rs.pairs], "records": [asdict(r) for r in rs.records], "rejected": rs.rejected,
            "camera_fits": rs.camera_fits, "subpix": rs.subpix}
    extra = {} if rs.cameras is None else {"cameras": rs.cameras.astype(np.float32), "has_camera": rs.has_camera}
    np.savez(path, tiles=rs.tiles.astype(np.float32), targets=rs.targets.astype(np.float32),
             captures=rs.captures.astype(np.float32), meta=np.array(json.dumps(meta)), **extra)


def load_set(path: Path) -> RealPairSet:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        tiles, targets, captures = z["tiles"], z["targets"], z["captures"]
        cameras = z["cameras"] if "cameras" in z.files else None
        has_camera = z["has_camera"].astype(bool) if "has_camera" in z.files else None
    records = [RealTile(**{**r, "shift": tuple(r["shift"]), "gain": tuple(r["gain"])}) for r in meta["records"]]
    return RealPairSet(meta["source"], meta["seed"], meta["n"], meta["tile"], meta["scale"], tuple(meta["splits"]),
                       [Pair(**p) for p in meta["pairs"]], tiles, targets, captures, records, meta["rejected"],
                       cameras, has_camera, meta.get("camera_fits", {}), meta.get("subpix"))


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


def subpix_residual(x: np.ndarray, ref: np.ndarray, shift, gain) -> dict:
    """The residual the integer shift leaves (subpix.measure) of an input against its reference as decoded, on their
    overlap at `shift` (x[i, j] ~ ref[i + dy, j + dx]) with the reference over `gain` and clipped as match_pair's
    target is; NaN in the reference (a stack merge's holes) stays NaN, so it reads as the input there."""
    xs, ys = overlap(x, ref, int(shift[0]), int(shift[1]))
    xs, ys = _centre(xs, MAX_SIDE), _centre(ys, MAX_SIDE)     # what measure reads, cut before it is copied
    t = np.clip(np.asarray(ys, dtype=np.float32) / np.asarray(gain, dtype=np.float32), 0.0, 1.0)
    return measure_residual(xs, t)


def _subpix_pair(m: dict, x: np.ndarray, y: np.ndarray, cam: np.ndarray | None) -> tuple[dict, np.ndarray | None, dict]:
    """A matched pair (match_of: x the input as fitted, y the reference as decoded) with its target moved by the
    residual the integer shift leaves, when it is trusted: the reference's overlap moved by its spectrum
    (subpix.fourier_shift), then over the gains and clipped as match_pair's target, its holes and its validity as
    they were. The camera's photo, matched to the target as it was, is registered to the moved target the same way,
    so the camera too is scored at its best registration. Returns the pair, the camera and the two measurements."""
    dy, dx = (int(v) for v in m["align"]["shift"])
    sp = subpix_residual(x, y, (dy, dx), m["gain"])
    if sp["trusted"]:
        _, ys = overlap(x, y, dy, dx)
        t = np.clip(fourier_shift(np.asarray(ys, dtype=np.float32), tuple(sp["shift"]))
                    / np.asarray(m["gain"], dtype=np.float64), 0.0, 1.0)
        m["t"] = np.where(np.isnan(t), m["t"], t).astype(np.float32)
    out = {"target": sp}
    if cam is not None:
        try:
            spc = measure_residual(m["t"], cam)
            if spc["trusted"]:
                cam = fourier_shift(cam, tuple(spc["shift"]))
            out["camera"] = spc
        except Exception as ex:  # noqa: BLE001 - the camera stays as matched
            out["camera"] = {"error": f"{type(ex).__name__}: {ex}"[:300]}
    return m, cam, out


def _sp_note(d: dict | None) -> str:
    if not d:
        return ""
    if "error" in d:
        return f", sub-pixel residual failed ({d['error'][:80]})"
    parts = []
    for k, v in (("residual", d["target"]), ("camera", d.get("camera"))):
        if v and "shift" in v:
            parts.append(f"{k} ({v['shift'][0]:+.3f}, {v['shift'][1]:+.3f}) px "
                         f"{'applied' if v['trusted'] else 'not trusted'}")
        elif v:
            parts.append(f"{k} failed ({v['error'][:60]})")
    return "".join(", " + p for p in parts)


def build_set(source: str, n: int = 64, tile: int = 256, scale: int | None = None, seed: int = 0,
              splits: tuple[str, ...] = DEFAULT_SPLITS, root: Path | None = None, log=print,
              decode=decode_raw, pairs: list[Pair] | None = None, subpix: bool = False) -> RealPairSet:
    """The cached set, or a new one: the pairs of `splits` in a seeded order, each decoded once at the pair
    scale (pair_scale, or `scale`), matched, and asked for ceil(n / pairs) tiles, with more passes over the
    pairs while the set is short. `decode(path, scale) -> (image, bit_depth)` is the raw decoder. With `subpix`
    each pair's target, and the camera's photo, is moved by the residual the integer shift leaves (_subpix_pair):
    a set of its own (set_path's _sp) whose tiles are the plain set's, so the two score the same tiles."""
    root = Path(root or data_root())
    log = log or (lambda *a: None)
    scale = int(scale or pair_scale(source))
    path = set_path(source, seed, n, tile, scale, root, subpix=subpix)
    if path.exists():
        rs = load_set(path)
        if tuple(rs.splits) == tuple(splits):
            if source == HDRPLUS and rs.cameras is None and not rs.camera_fits:
                save_set(attach_cameras(rs, root, decode, log), path)
            return rs
        log(f"{path.name}: cached with splits {rs.splits}, rebuilding for {splits}")
    pairs = pairs if pairs is not None else pairs_of(source, root, log)
    chosen = [p for p in pairs if p.split in splits]
    if not chosen:
        raise ValueError(f"{source}: no pairs in splits {splits} ({len(pairs)} pairs on disk)")
    rng = np.random.default_rng(_seed("realpairs", source, seed, n, tile, scale))
    order = [int(i) for i in rng.permutation(len(chosen))]
    if source in STACK_SOURCES:  # a scene's pairs one after another, so its stack decodes once (decode_stack's cache)
        first: dict[str, int] = {}
        for i in order:
            first.setdefault(chosen[i].scene, len(first))
        order.sort(key=lambda i: first[chosen[i].scene])
    base = root / "raw" / source
    tiles, targets, captures, records, rejected, cams = [], [], [], [], {}, []
    # one matched pair's arrays are held at a time (a pair at the working scale is hundreds of MB as float64); a pair
    # a later pass draws from again is decoded again
    cur: tuple[int, dict, np.ndarray | None] | None = None
    fits: dict = {}
    sps: dict = {}
    bits: dict[int, int] = {}
    passes = 0
    while len(tiles) < n and passes < 4:
        passes += 1
        want = max(1, math.ceil((n - len(tiles)) / max(1, len(order))))
        for i in order:
            if len(tiles) >= n:
                break
            p = chosen[i]
            if p.in_file in rejected:
                continue
            if cur is None or cur[0] != i:
                cur = None
                try:
                    m, xin, bx, y = match_of(p, base, scale, decode)
                except Exception as ex:
                    rejected[p.in_file] = f"{type(ex).__name__}: {ex}"[:300]
                    log(f"{source}: {p.in_file}: {rejected[p.in_file]}")
                    continue
                cam = None
                cf = camera_file(p)
                if cf is not None and (base / cf).exists():
                    try:
                        cam, fit = camera_overlap(m, y, decode_camera(base / cf, scale))
                        fits.setdefault(str(i), fit)
                    except Exception as ex:  # the pair stays; only its camera column is missing
                        fits.setdefault(str(i), {"error": f"{type(ex).__name__}: {ex}"[:300]})
                        log(f"{source}: camera for {p.in_file}: {fits[str(i)]['error']}")
                if subpix:
                    try:
                        m, cam, sps[str(i)] = _subpix_pair(m, xin, y, cam)
                    except Exception as ex:  # noqa: BLE001 - the pair stays at its integer shift
                        sps[str(i)] = {"error": f"{type(ex).__name__}: {ex}"[:300]}
                del y, xin
                m["x"], m["t"] = np.asarray(m["x"], dtype=np.float32), np.asarray(m["t"], dtype=np.float32)
                cur = (i, m, cam)
                if i not in bits:
                    bits[i] = bx
                    log(f"{source}: {p.in_file} vs {p.ref_file}: shift {m['align']['shift']}, gain "
                        f"{np.round(m['gain'], 3).tolist()}, corr {m['align']['corr']:.3f}, electrons at clip "
                        f"{_fmt(m['noise'].get('electrons_at_clip'))}, read {_fmt(m['noise'].get('read_noise_e'))} e"
                        + (f", {100 * m['hole']:.1f}% without a reference" if "hole" in m else "")
                        + _sp_note(sps.get(str(i))))
            _, m, cam = cur
            for y0, x0 in _draw_tiles(m, tile, want, rng):
                tiles.append(m["x"][y0:y0 + tile, x0:x0 + tile].astype(np.float32))
                targets.append(m["t"][y0:y0 + tile, x0:x0 + tile].astype(np.float32))
                captures.append(capture_of(m["noise"], bits[i]))
                cams.append(_camera_tile(cam, y0, x0, tile))
                records.append(RealTile(i, y0, x0, tuple(m["align"]["shift"]), tuple(float(g) for g in m["gain"]),
                                        float(m["align"]["corr"]), m["noise"], int(bits[i]),
                                        float(1.0 - m["valid"][y0:y0 + tile, x0:x0 + tile].mean())))
                if len(tiles) >= n:
                    break
        if not bits:
            break
    if len(tiles) < n:
        raise ValueError(f"{source}: {len(tiles)} tile(s) of {n} from {len(bits)} "
                         f"usable pair(s) of {len(chosen)}; {len(rejected)} rejected: "
                         + "; ".join(f"{k}: {v[:60]}" for k, v in list(rejected.items())[:5]))
    cameras, has_camera = _pack_cameras(cams, tile)
    rs = RealPairSet(source, seed, n, tile, scale, tuple(splits), chosen, np.stack(tiles), np.stack(targets),
                     np.stack(captures), records, rejected, cameras, has_camera, fits, sps if subpix else None)
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
