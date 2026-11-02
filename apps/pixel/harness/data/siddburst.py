"""SIDD's source bursts as real training pairs: python -m harness data siddburst.

SIDD (Abdelhamed, Lin and Brown, CVPR 2018) shot every scene instance as a burst of 150 raw frames of a static scene
from a tripod. The Hugging Face mirror MhdMohammadi/SIDD-CC-DNG (MIT, the 115 instances of SIDD Medium's training
list) holds each burst as one zip of DNGs (data/train/<instance>_DNG.zip, 2 to 4 GB). A burst gives what no single
exposure pair does: many inputs of one framing with independent noise, and a reference averaged from frames that
are none of them. `reduce_instance` turns a zip into that, so the zip need not stay on disk:

- pass 1 reads every frame's raw mosaic: its mean per CFA plane, and its motion against the burst's middle frame.
  The bursts are not still (read 11-02 on 0001, 0012 and 0017): the frames drift by a few tenths of a pixel over the
  burst, a slow zoom as much as a shift, and now and then the tripod is knocked several pixels with a turn; an
  affine map fits the motion to 0.01 to 0.02 quads at five places across the frame. So every frame is registered
  on its 2x2 quads' luminance: the integer shift of the whole frame (frame_shift), then per patch of a grid an
  integer offset by phase correlation on the luminance smoothed and a sub-quad one by Gauss-Newton on it as it is,
  and an affine map fitted to the patches robustly (register). A frame whose plane mean is off the burst's
  (noise.outlier_frames: an unsettled first exposure) is dropped, and so is one that does not register while most
  do; where most do not (too little structure to match), the burst is taken as it lies and the pair test below
  judges it;
- the inputs are up to INPUTS frames, spread over the burst, of the largest set that lies within INPUT_PX of one
  frame at every corner (choose_inputs). The mean is made in the inputs' average geometry, so no input is further
  than about INPUT_PX from it, and of every other frame that holds;
- pass 2 merges those frames onto each CFA plane's own lattice by kernel regression of order 1 (KernelMerge): per
  pixel, a plane fitted to every frame's nearest raw sample to it, each weighted by its distance from the pixel,
  each frame scaled per plane to the mean's frames' mean above black (noise.frame_scales: lamp drift and flicker
  out). Nothing is interpolated. A cubic warp of each frame, averaged, was what this did first, and it is biased
  wherever the scene has detail near the plane lattice's Nyquist (every edge: a Bayer plane is sampled at half the
  sensor's pitch, so it aliases): on a synthetic aliased scene its edges were 10 to 20 times further off than the
  kernel's, and on real bursts its mean sat 2 to 6 noise variances off the still inputs' own mean at edges (0017)
  where the kernel's sat 0.2 to 0.4 (0.06 to 0.14 on 0008; 11-02). The same pass gives the noise law, from the
  frames within INPUT_PX of the anchor as they lie (noise.PlaneStats, the noise check's own statistics);
- pass 3 judges each input against the mean (pair_excess): its gain fitted out, its mismatch with the mean beyond
  the noise both carry, on the mean's edges and its flat pixels, in the input's noise variances. An input is kept
  when the median over the planes of the worse of the two is at most MAX_PAIR_EXCESS, and copied out as it is
  (stacks/<stack>/frames/<member>.DNG). The excess holds the input's own offset from the mean (up to INPUT_PX,
  which training's sub-pixel read takes out) as its square: on six bursts (11-02) the still inputs read 0.83 at
  most, 11 to 33 noise variances per px squared at ISO 100, and one of 0017's, 0.249 px off, read 2.16 and was
  dropped for that alone (1 of 117 inputs); a 1 px move reads about 27. The gate errs toward dropping. The burst
  is a pair source when MIN_INPUTS are kept and at least MIN_MEAN frames are in the mean: stacks/<stack>/mean.npy,
  float32 raw DN over the frame's whole raw_image (the visible area the mean, the margins the template frame's),
  black not taken off. Its noise is about 1/sqrt(n_eff) of a frame's (n_eff about 40 at MERGE_SIGMA 0.15 of 100 or
  so frames: the weights trade averaging for bias) and independent of every input's;
- stacks/<stack>/mean.json, written last, records the instance (SIDD's metadata: camera, ISO, shutter,
  illuminant, brightness, scene number), the levels, the frames used and dropped and why, every frame's affine map
  and the inputs' and the mean frames' distances from the mean's geometry, the plane statistics and n_eff, each
  input's pair excess, and whether the burst pairs (`ok`, else `why`). A stack whose mean.json is on disk is done.

The stack is named <instance>_s<scene number> so the source's clean.scene regex keys the split on SIDD's scene (ten
scenes shot under several cameras, ISOs and lights), not the instance: two instances of one scene never fall on
both sides of it. The pairs' scene in the store is the stack, one reference each.

decode_sidd decodes the inputs and the mean alike through LibRaw (camera white balance, linear, no brightening, the
labeller's settings) at explicit levels: black and saturation from the DNG's own tags, no adjust_maximum. LibRaw's
default puts 1.0 below these DNGs' WhiteLevel (about 0.94 of it on the Pixel, 0.87 on the iPhone 7, read 11-02),
and adjust_maximum_thr at a frame's own maximum, so a frame and a mean would clip at different levels. The mean
goes through a template, the stack's first input with its raw_image replaced by the mean times K = 65535 // white
and black and saturation times K: on a Pixel and an iPhone frame that is the plain explicit decode of the same data
to the last of 65535 (max difference 0, 11-02). realpairs.decoder_of gives it to this source's pairs."""
from __future__ import annotations

import io
import json
import math
import multiprocessing as mp
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np

from ..scene import box_downsample
from . import data_root, keeps_reserve, reserve_bytes
from .collect import consumed_marker
from .decodedstore import _lower_priority
from .noise import FIT_HI, PlaneStats, cfa_split, frame_scales, outlier_frames, sidd_metadata

SIDD = "sidd-dng"
ZIP_DIR = "data/train"            # <root>/raw/sidd-dng/data/train/<instance>_DNG.zip
METADATA = "metadata/train_metadata.csv"
STACK_DIR = "stacks"              # <root>/raw/sidd-dng/stacks/<instance>_s<scene>/{mean.npy, mean.json, frames/}
INPUTS = 24                       # frames of a burst kept as inputs, at most
MIN_INPUTS = 8                    # inputs a burst needs, at least
MIN_MEAN = 64                     # frames the mean needs, at least
MERGE_SIGMA = 0.15                # plane px: the Gaussian weight on a sample's distance from the pixel it stands for
MERGE_RIDGE = 1e-3                # plane px^2, times the weights' sum, on the merge's local fit's slope terms
MAX_PAIR_EXCESS = 1.5             # an input's mismatch with the mean beyond their noise, in its noise variances
EDGE_PERCENTILE = 98.0            # the mean's gradient percentile above which a pixel is an edge for that mismatch
PAIR_BLOCK = 64                   # plane px: side of the blocks the mismatch's local maximum is read on
PAIR_BORDER = 4                   # plane px along the frame's edge the mismatch leaves out
CLIP_SHARE = 0.95                 # share of white - black above which a pixel of the mean or an input is clipped
SHIFT_CROP = 1024                 # side of the centre crop of 2x2 quads a frame's integer shift is read on
SHIFT_MAX = 16                    # quads (2 px) a frame's integer shift may claim
MOVE_MARGIN = 0.02                # share by which a shift's normalised MSE must beat no shift's to be taken
REG_PATCH = 128                   # side, in quads, of the patches a frame's motion is measured on
REG_GRID = (6, 8)                 # patches along the frame's shorter and its longer side
REG_MARGIN = 48                   # quads between the patch grid and the frame's edge
REG_SIGMA = 1.0                   # Gaussian, in quads, on the luminance a patch's integer offset and gradient are taken on
REG_SEARCH = 6                    # quads a patch's integer offset may differ from its frame's
REG_PAD = 6                       # quads of frame around a patch its sub-quad search resamples
REG_ITERS = 12                    # Gauss-Newton steps per patch, at most
REG_MIN_NCC = 0.3                 # a matched patch's correlation with the reference's, at least (else no structure)
REG_MIN_PATCHES = 12              # patches an affine fit keeps, at least
REG_FLOOR = 0.05                  # quads: the least residual the robust fit drops a patch for
REG_MAX_RMS = 0.15                # quads: the kept patches' rms off the fit, at most (else they matched noise)
INPUT_PX = 0.25                   # mosaic px: the inputs lie within this of one frame (the anchor) at every corner
MAX_WARP_PX = 16.0                # mosaic px from the mean's geometry, at some corner, past which a frame is dropped
KEEP_FOR_SHAPE = 4                # frames PlaneStats keeps for the residual-shape and line-noise checks
FIT_QUANTILE = 99.5               # percentile of a plane's mean the noise-law fit window's top sits at (fit_span)
RESERVE_ENV, RESERVE_GB = "HARNESS_SIDD_RESERVE_GB", 100.0
ZIP_RE = re.compile(r"^([0-9]{4})_DNG[.]zip$")
LEVEL_SETTINGS = dict(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True, adjust_maximum_thr=0.0)


# ---- decoding ----------------------------------------------------------------------------------------------------

def levels(raw, k: int = 1) -> dict:
    """LibRaw's black and saturation set to the raw's own tags times `k` (per-channel blacks through user_cblack
    when they differ), and no adjust_maximum: the explicit levels both inputs and means decode at."""
    black = [int(b) for b in raw.black_level_per_channel]
    lo = min(black)
    kw = dict(LEVEL_SETTINGS, user_black=lo * k, user_sat=int(raw.white_level) * k)
    if any(b != lo for b in black):
        kw["user_cblack"] = [(b - lo) * k for b in black]
    return kw


def _bits(white: int) -> int:
    return int(math.ceil(math.log2(white + 1))) if white > 0 else 0


def decode_sidd(path: Path, scale: int) -> tuple[np.ndarray, int]:
    """A SIDD frame (a .DNG) or a stack's mean (mean.npy, its template named in the mean.json beside it) through
    LibRaw at the explicit levels, 1.0 at the white level, box-downsampled by `scale`; with the sensor's bit depth
    from its white level. The realpairs decoder's contract (decode_raw)."""
    import rawpy
    path = Path(path)
    if path.suffix.lower() == ".npy":
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        mean = np.load(path)
        with rawpy.imread(str(path.parent / meta["template"])) as raw:
            white = int(raw.white_level)
            k = 65535 // white
            if raw.raw_image.shape != mean.shape:
                raise ValueError(f"{path}: mean {mean.shape}, template raw_image {raw.raw_image.shape}")
            raw.raw_image[:] = np.clip(np.rint(mean.astype(np.float64) * k), 0, 65535).astype(np.uint16)
            rgb = raw.postprocess(**levels(raw, k))
    else:
        with rawpy.imread(str(path)) as raw:
            white = int(raw.white_level)
            rgb = raw.postprocess(**levels(raw))
    img = np.asarray(rgb, dtype=np.float64) / 65535.0
    return (box_downsample(img, scale) if scale > 1 else img), _bits(white)


# ---- reduction ---------------------------------------------------------------------------------------------------

def stack_name(instance: str, scene: str) -> str:
    return f"{instance}_s{scene}"


def stack_root(root: Path | None = None) -> Path:
    return Path(root or data_root()) / "raw" / SIDD / STACK_DIR


def zips(root: Path | None = None) -> list[Path]:
    """The complete instance zips on disk (a download in progress is a .part)."""
    d = Path(root or data_root()) / "raw" / SIDD / ZIP_DIR
    return sorted(p for p in d.glob("*_DNG.zip") if ZIP_RE.match(p.name)) if d.is_dir() else []


def spread(n: int, k: int) -> list[int]:
    """`k` indices of `n`, evenly spread from the first to the last (all of them when k >= n)."""
    if k >= n:
        return list(range(n))
    if k <= 0:
        return []
    return sorted({int(round(v)) for v in np.linspace(0, n - 1, k)})


def quad_luminance(vis: np.ndarray, crop: int = SHIFT_CROP) -> np.ndarray:
    """The sum of each 2x2 quad of a mosaic (one value per Bayer quad, every colour in it; quad (i, j) is centred on
    the mosaic's (2i + 0.5, 2j + 0.5)): its centre crop of side `crop`, all of it with crop 0."""
    h, w = (vis.shape[0] // 2) * 2, (vis.shape[1] // 2) * 2
    q = vis[:h, :w].astype(np.float32)
    q = q[0::2, 0::2] + q[0::2, 1::2] + q[1::2, 0::2] + q[1::2, 1::2]
    if not crop:
        return q
    cy, cx = q.shape[0] // 2, q.shape[1] // 2
    r = crop // 2
    return q[max(0, cy - r):cy + r, max(0, cx - r):cx + r]


def frame_shift(x: np.ndarray, ref: np.ndarray, max_shift: int = SHIFT_MAX, margin: float = MOVE_MARGIN) -> tuple[int, int]:
    """(dy, dx) with x[i, j] ~ ref[i + dy, j + dx], in quads: of the phase-correlation peak and the eight shifts of
    one, the one with the least normalised MSE over the overlap (realpairs._nmse), where it beats no shift's by
    `margin`; else (0, 0). Phase correlation alone puts its peak on noise between two noisy frames of a static scene
    now and then (the whitened spectrum is mostly noise where the scene has little texture); a frame that did not
    move fits best where it is."""
    from .realpairs import _nmse, phase_shift
    cands = {phase_shift(x, ref, max_shift)} | {(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
    scores = {c: _nmse(x, ref, *c) for c in cands}
    best = min(scores, key=scores.get)
    return best if best != (0, 0) and scores[best] < (1.0 - margin) * scores[(0, 0)] else (0, 0)


def fit_span(ps: PlaneStats) -> float:
    """DN above black the plane's bins and noise-law fit window span: its mean's FIT_QUANTILE percentile over
    noise.FIT_HI, so the window's top sits at it, capped at white - black. SIDD's 'low' bursts (48 of the 115)
    are dim: 0018's planes average 10 to 23 DN above black of 960, nearly every pixel in the first two or three
    of 48 bins over the full span, and no law fitted there, so its edge test had nothing to test against."""
    span = ps.white - ps.black
    top = float(np.percentile(ps.w.mean, FIT_QUANTILE)) - ps.black
    return min(span, top / FIT_HI) if top > 0 else span


# ---- registration ------------------------------------------------------------------------------------------------

QUAD = np.array([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])   # quad (i, j) -> its centre on the mosaic


def match_luminance(vis: np.ndarray, quad_black: float) -> np.ndarray:
    """What frames are matched on: the quads' luminance less a quad's black (float64)."""
    return quad_luminance(vis, 0).astype(np.float64) - quad_black


class Reference:
    """The frame a burst's frames are registered to: its patches on a REG_GRID grid (the first count along the
    shorter side), each as it is and smoothed by REG_SIGMA, with the smoothed one's gradient and the sum of the
    gradient's outer product."""

    def __init__(self, lum: np.ndarray):
        from scipy import ndimage
        self.shape = lum.shape
        self.mean = float(lum.mean())
        h, w = lum.shape
        p = REG_PATCH
        ny, nx = REG_GRID if h <= w else REG_GRID[::-1]
        smooth = ndimage.gaussian_filter(lum, REG_SIGMA)
        gy, gx = np.gradient(smooth)
        self.patches = []
        for y0 in sorted({int(v) for v in np.linspace(REG_MARGIN, h - REG_MARGIN - p, ny)}):
            for x0 in sorted({int(v) for v in np.linspace(REG_MARGIN, w - REG_MARGIN - p, nx)}):
                if min(y0, x0) < 0 or y0 + p > h or x0 + p > w:
                    continue
                s = (slice(y0, y0 + p), slice(x0, x0 + p))
                ry, rx = gy[s].copy(), gx[s].copy()
                rr = np.array([[(ry * ry).sum(), (ry * rx).sum()], [(ry * rx).sum(), (rx * rx).sum()]])
                self.patches.append((y0, x0, lum[s].copy(), smooth[s].copy(), ry, rx, rr))


def _ssd(a: np.ndarray, b: np.ndarray, dy: int, dx: int) -> float:
    """The mean squared difference of a and b's overlap under a[i, j] ~ b[i + dy, j + dx], each less its mean."""
    from .realpairs import overlap
    x, y = overlap(a, b, dy, dx)
    return float(np.mean(((x - x.mean()) - (y - y.mean())) ** 2)) if x.size else math.inf


def register(lum: np.ndarray, ref: Reference, shift: tuple[int, int] = (0, 0)) -> dict | None:
    """The affine map T (3x3, on quad coordinates) with lum(T p) ~ ref(p), or None. Per reference patch, from
    `shift` (the frame's integer shift: lum(p + shift) ~ ref(p) roughly), an integer offset by phase correlation
    on the luminance smoothed by REG_SIGMA, within REG_SEARCH quads where it matches better than none, then a
    sub-quad one by Gauss-Newton on the frame resampled by a cubic spline: the residual on the luminance as it is,
    weighted by the smoothed reference's gradient. A quad's noise is independent of its neighbours', and the
    gradient at a quad (odd about it) carries none of that quad's own, so the offsets are unbiased; with the
    residual on the smoothed luminance too, both carry the reference's noise, and where the light changes across a
    patch the offsets leaned toward its brighter side (0.07 quads on a ramp, 11-02). The step's matrix is that
    gradient against the resampled frame's: their noise is independent, so noise does not inflate it and shorten
    the steps on a dim patch, as the reference's gradient against itself does. A patch that does not settle, runs
    off its window or, smoothed, correlates with the reference's under REG_MIN_NCC (noise against noise: no
    structure) is not used. T is fitted to the patches'
    offsets by least squares, each weighted by one over its standard error, dropping the patches off it by more
    than max(REG_FLOOR, 3 robust sigma) until none is; with fewer than REG_MIN_PATCHES left, or the rest off it by
    more than REG_MAX_RMS (offsets that agree on nothing), there is no map. Returns T, the patches kept and
    matched, and the fit's rms in quads."""
    from scipy import ndimage
    from .realpairs import _corr, phase_shift
    m = float(lum.mean())
    x = lum * (ref.mean / m) if m > 0 and ref.mean > 0 else lum   # flicker out
    xs = ndimage.gaussian_filter(x, REG_SIGMA)
    h, w = x.shape
    p, pad = REG_PATCH, REG_PAD
    rows = []
    for y0, x0, r, rs, ry, rx, rr in ref.patches:
        cy, cx = y0 + shift[0], x0 + shift[1]
        if cy < 0 or cx < 0 or cy + p > h or cx + p > w:
            continue
        here = xs[cy:cy + p, cx:cx + p]
        d = phase_shift(rs, here, REG_SEARCH)                        # r[i] ~ here[i + d]
        if d != (0, 0) and not _ssd(rs, here, *d) < _ssd(rs, here, 0, 0):
            d = (0, 0)
        ey, ex = shift[0] + d[0], shift[1] + d[1]                    # lum(p + e) ~ ref(p), to the quad
        wy, wx = y0 + ey - pad, x0 + ex - pad
        if wy < 0 or wx < 0 or wy + p + 2 * pad > h or wx + p + 2 * pad > w:
            continue
        coef = ndimage.spline_filter(x[wy:wy + p + 2 * pad, wx:wx + p + 2 * pad], order=3, mode="mirror")
        f, a, settled = np.zeros(2), None, False
        for _ in range(REG_ITERS):
            xw = ndimage.shift(coef, -f, order=3, mode="mirror", prefilter=False)   # xw(i) = lum(p + e + f)
            gy, gx = (g[pad:-pad, pad:-pad] for g in np.gradient(xw))
            res = r - xw[pad:-pad, pad:-pad]
            res -= res.mean()
            a = np.array([[(ry * gy).sum(), (ry * gx).sum()], [(rx * gy).sum(), (rx * gx).sum()]])
            try:
                step = np.linalg.solve(a, np.array([(ry * res).sum(), (rx * res).sum()]))
            except np.linalg.LinAlgError:
                break
            f += step
            if not np.isfinite(f).all() or np.abs(f).max() > pad - 2:
                break
            if np.abs(step).max() < 1e-3:
                settled = True
                break
        if not settled:
            continue
        xw = ndimage.shift(coef, -f, order=3, mode="mirror", prefilter=False)
        if _corr(rs, ndimage.gaussian_filter(xw, REG_SIGMA)[pad:-pad, pad:-pad]) < REG_MIN_NCC:
            continue
        res = r - xw[pad:-pad, pad:-pad]
        ainv = np.linalg.inv(a)
        var = float(res.var()) * float(np.trace(ainv @ rr @ ainv.T))
        c0, c1 = y0 + (p - 1) / 2.0, x0 + (p - 1) / 2.0
        rows.append((c0, c1, c0 + ey + f[0], c1 + ex + f[1], var))
    if len(rows) < REG_MIN_PATCHES:
        return None
    pts = np.asarray(rows)
    xs, ys = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))], pts[:, 2:4]
    sw = 1.0 / np.sqrt(np.maximum(pts[:, 4], 0.0) + 0.01 ** 2)   # one over the offset's standard error
    keep = np.ones(len(pts), dtype=bool)
    for _ in range(20):
        if keep.sum() < REG_MIN_PATCHES:
            return None
        b = np.linalg.lstsq(xs[keep] * sw[keep, None], ys[keep] * sw[keep, None], rcond=None)[0]
        off = np.hypot(*(xs @ b - ys).T)
        new = off <= max(REG_FLOOR, 3.0 * 1.4826 * float(np.median(off[keep])))
        if np.array_equal(new, keep):
            break
        keep = new
    rms = float(np.sqrt(np.mean(off[keep] ** 2))) if keep.any() else math.inf
    if keep.sum() < REG_MIN_PATCHES or not rms <= REG_MAX_RMS:
        return None
    t = np.eye(3)
    t[:2] = b.T
    return {"T": t, "patches": int(keep.sum()), "matched": len(rows), "rms_quads": rms}


def grid_corners(shape: tuple[int, int]) -> np.ndarray:
    """The corners of a quad grid of `shape`, as homogeneous columns."""
    h, w = shape
    return np.array([[0, 0, h - 1, h - 1], [0, w - 1, 0, w - 1], [1, 1, 1, 1]], dtype=float)


def displacement_px(ta: np.ndarray, tb: np.ndarray, corners: np.ndarray) -> float:
    """Mosaic px between where frame a shows a point of frame b's quad grid and where b has it (ta and tb map the
    reference into each), the most over the grid's corners: an affine map moves a corner the most."""
    d = (ta @ np.linalg.inv(tb) @ corners - corners)[:2]
    return 2.0 * float(np.hypot(d[0], d[1]).max())


def choose_inputs(maps: dict[int, np.ndarray], corners: np.ndarray, middle: int) -> tuple[int, list[int]]:
    """The anchor, the frame with the most frames within INPUT_PX of it at every corner (on a tie the nearest the
    burst's middle), and those frames, the anchor among them."""
    best = None
    for k in sorted(maps):
        near = [j for j in sorted(maps) if displacement_px(maps[j], maps[k], corners) <= INPUT_PX]
        key = (len(near), -abs(k - middle))
        if best is None or key > best[0]:
            best = (key, k, near)
    return best[1], best[2]


def plane_map(m: np.ndarray, cy: int, cx: int) -> np.ndarray:
    """A map on mosaic coordinates (3x3) as one on CFA plane (cy, cx)'s own lattice, whose pixel (u, v) is the
    mosaic's (2u + cy, 2v + cx)."""
    e = np.array([[2.0, 0.0, cy], [0.0, 2.0, cx], [0.0, 0.0, 1.0]])
    return np.linalg.inv(e) @ m @ e


def nearest_samples(plane: np.ndarray, pm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each pixel (u, v) of a CFA plane's lattice, the plane's own sample nearest the point `pm` (a plane_map)
    takes it to: its value (float64), its offset from that point in plane px (dy, dx, each within half a pixel
    where the point falls on the plane), and whether the point falls on the plane or within half a pixel of its
    edge. Nothing is interpolated: a sample is the frame's own reading, with its own noise and nobody else's."""
    h, w = plane.shape
    u, v = np.arange(h, dtype=np.float64)[:, None], np.arange(w, dtype=np.float64)[None, :]
    yy = pm[0, 0] * u + pm[0, 1] * v + pm[0, 2]
    xx = pm[1, 0] * u + pm[1, 1] * v + pm[1, 2]
    iy = np.clip(np.rint(yy), 0, h - 1).astype(np.intp)
    ix = np.clip(np.rint(xx), 0, w - 1).astype(np.intp)
    valid = (yy >= -0.5) & (yy <= h - 0.5) & (xx >= -0.5) & (xx <= w - 0.5)
    return plane[iy, ix].astype(np.float64), iy - yy, ix - xx, valid


class KernelMerge:
    """Frames merged onto one CFA plane's lattice by kernel regression of order 1 (Takeda, Farsiu and Milanfar,
    IEEE TIP 2007): at each pixel a plane fitted by weighted least squares to every frame's nearest sample
    (nearest_samples), each weighted by a Gaussian of MERGE_SIGMA on its offset from the pixel, the fit's slopes
    held toward 0 by MERGE_RIDGE times the weights' sum (where the offsets barely differ the slope is left at 0, not
    fitted to noise); the pixel is the fit's value at it. That value is linear in the samples: with M the weights'
    moments ([1, dy, dx] against itself) and u the first row of M's inverse, it is sum_i w_i (u . [1, dy_i, dx_i])
    x_i, so its noise variance over one frame's is u' Q u, Q the moments of the weights squared: 1 / n_eff, n_eff
    the frames' worth of averaging the pixel has. S and Z moments are float64 (the fit subtracts nearly equal
    terms), Q float32."""
    S = ("S0", "Sy", "Sx", "Syy", "Sxy", "Sxx", "Z0", "Zy", "Zx")
    Q = ("Q0", "Qy", "Qx", "Qyy", "Qxy", "Qxx")

    def __init__(self, shape: tuple[int, int], sigma: float = MERGE_SIGMA, ridge: float = MERGE_RIDGE):
        self.k, self.ridge = -0.5 / (sigma * sigma), ridge
        self.m = {n: np.zeros(shape) for n in self.S}
        self.m.update({n: np.zeros(shape, dtype=np.float32) for n in self.Q})
        self.cover = np.zeros(shape, dtype=np.uint16)

    def add(self, val: np.ndarray, dy: np.ndarray, dx: np.ndarray, valid: np.ndarray) -> None:
        """One frame's nearest samples: their values, offsets and cover (nearest_samples)."""
        m = self.m
        w = np.exp((dy * dy + dx * dx) * self.k)
        w *= valid
        wy, wx, wv = w * dy, w * dx, w * val
        for n, a in (("S0", w), ("Sy", wy), ("Sx", wx), ("Syy", wy * dy), ("Sxy", wy * dx), ("Sxx", wx * dx),
                     ("Z0", wv), ("Zy", wv * dy), ("Zx", wv * dx)):
            m[n] += a
        w *= w
        wy, wx = w * dy, w * dx
        for n, a in (("Q0", w), ("Qy", wy), ("Qx", wx), ("Qyy", wy * dy), ("Qxy", wy * dx), ("Qxx", wx * dx)):
            m[n] += a
        self.cover += valid

    def finish(self, rows: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """The merged plane (float64, 0 where no frame covers) and its noise variance over one frame's (float32,
        inf where no frame covers), a block of rows at a time."""
        m, r = self.m, self.ridge
        out = np.zeros(self.cover.shape)
        var = np.full(self.cover.shape, np.inf, dtype=np.float32)
        for y in range(0, out.shape[0], rows):
            s = slice(y, y + rows)
            g = {n: a[s].astype(np.float64) for n, a in m.items()}
            a, b, c = g["S0"], g["Sy"], g["Sx"]
            d, e, f = g["Syy"] + r * a, g["Sxy"], g["Sxx"] + r * a
            c0, c1, c2 = d * f - e * e, c * e - b * f, b * e - c * d     # M's cofactors along its first row
            det = a * c0 + b * c1 + c * c2
            ok = (self.cover[s] > 0) & (det > 0)
            det = np.where(ok, det, 1.0)
            u0, u1, u2 = c0 / det, c1 / det, c2 / det
            t = u0 * g["Z0"] + u1 * g["Zy"] + u2 * g["Zx"]
            v = (u0 * u0 * g["Q0"] + u1 * u1 * g["Qyy"] + u2 * u2 * g["Qxx"]
                 + 2.0 * (u0 * u1 * g["Qy"] + u0 * u2 * g["Qx"] + u1 * u2 * g["Qxy"]))
            out[s] = np.where(ok, t, 0.0)
            var[s] = np.where(ok, v, np.inf)
        return out, var


def pair_masks(t: np.ndarray, covered: np.ndarray, span: float) -> dict:
    """The pixels a plane's pairs are judged on, from the mean `t` (DN above black) and where it is `covered` (its
    variance finite): `base`, those more than PAIR_BORDER from the frame's edge and from any pixel the mean does
    not cover (its 0 there would read as an edge), whose mean is under CLIP_SHARE of `span` (white - black);
    `edge`, the base's pixels above its EDGE_PERCENTILE of the gradient of the mean smoothed by a Gaussian of 1 px
    (where a misplaced input differs most); `flat`, the base's lower half by it."""
    from scipy import ndimage
    base = ndimage.binary_erosion(covered, iterations=PAIR_BORDER, border_value=0)
    base &= t < CLIP_SHARE * span
    gr = np.hypot(*np.gradient(ndimage.gaussian_filter(t, 1.0)))
    if not base.any():
        return {"base": base, "edge": base, "flat": base, "clip": CLIP_SHARE * span}
    lo, hi = np.percentile(gr[base], [50.0, EDGE_PERCENTILE])
    return {"base": base, "edge": base & (gr > hi), "flat": base & (gr < lo), "clip": CLIP_SHARE * span}


def pair_excess(x: np.ndarray, t: np.ndarray, v: np.ndarray, fit: dict, masks: dict) -> dict:
    """An input's mismatch with the mean on one plane beyond the noise the two carry: `x` the input's plane and `t`
    the mean's, both DN above black, `v` the mean's noise variance over one frame's (KernelMerge), `fit` the plane's
    noise law (var = a level + b) and `masks` the plane's (pair_masks; the input's own clipped pixels out too). The
    input's gain against the mean, g = sum x t / sum t^2 (its exposure's flicker), is fitted out; on the edge and
    the flat pixels the excess is mean((x - g t)^2) less the noise both carry, mean(var (1 + v)), over mean(var),
    var the law at g t: 0 for a pair that differs by noise alone, the misregistration's share where the two lie
    apart. block_max, recorded only, is the largest over PAIR_BLOCK blocks (half of them judged at least) of the
    mean of (x - g t)^2 / (var (1 + v)) - 1: a local move shows there before it moves the plane's mean."""
    ok = masks["base"] & (x < masks["clip"])
    xb, tb = x[ok], t[ok]
    tt = float(np.dot(tb, tb))
    g = float(np.dot(xb, tb)) / tt if tt > 0 else 1.0
    r2 = (x - g * t) ** 2
    var = np.maximum(fit["a_dn_per_dn"] * g * t + fit["b_dn2"], 1.0 / 12.0)   # never under a DN's quantisation
    noise = var * (1.0 + v)
    out = {"gain": round(g, 5)}
    for name in ("edge", "flat"):
        sel = masks[name] & ok
        out[name] = (float((r2[sel].mean() - noise[sel].mean()) / var[sel].mean()) if sel.sum() >= 50 else None)
    z = np.where(ok, r2 / noise - 1.0, 0.0)
    bs = PAIR_BLOCK
    h, w = (z.shape[0] // bs) * bs, (z.shape[1] // bs) * bs
    zs = z[:h, :w].reshape(h // bs, bs, w // bs, bs).sum(axis=(1, 3))
    ns = ok[:h, :w].reshape(h // bs, bs, w // bs, bs).sum(axis=(1, 3))
    judged = ns >= bs * bs // 2
    out["block_max"] = float((zs[judged] / ns[judged]).max()) if judged.any() else None
    return out


# ---- reduction (continued) ---------------------------------------------------------------------------------------

def _read(z: zipfile.ZipFile, member: str) -> tuple[bytes, np.ndarray, dict]:
    """A member's bytes, its full raw_image, and its levels and geometry."""
    import rawpy
    data = z.read(member)
    with rawpy.imread(io.BytesIO(data)) as raw:
        full = raw.raw_image.copy()
        s = raw.sizes
        info = {"black": [int(b) for b in raw.black_level_per_channel], "white": int(raw.white_level),
                "pattern": raw.raw_pattern.tolist(), "raw_shape": [int(v) for v in full.shape],
                "visible": [int(s.top_margin), int(s.left_margin), int(s.height), int(s.width)]}
    return data, full, info


def _visible(full: np.ndarray, info: dict) -> np.ndarray:
    t, l, h, w = info["visible"]
    return full[t:t + h, l:l + w]


def _plane_black(info: dict) -> dict[str, float]:
    return {f"{cy}{cx}": float(info["black"][info["pattern"][cy][cx]]) for cy in range(2) for cx in range(2)}


def _affine(t: np.ndarray) -> list:
    return [[round(float(v), 6) for v in row] for row in t[:2]]


def reduce_instance(zip_path: Path, meta: dict | None, out_root: Path, log=None) -> dict:
    """One SIDD burst zip reduced to its stack under `out_root` (see the module's notes); returns the mean.json
    written. Raises on an unreadable zip, before anything is written."""
    from scipy import ndimage
    t0 = time.time()
    inst = ZIP_RE.match(zip_path.name).group(1)
    if not meta:
        raise ValueError(f"{zip_path.name}: no metadata row for instance {inst} (its scene keys the split)")
    d = out_root / stack_name(inst, meta["scene"])
    rec = {"source": SIDD, "instance": inst, "stack": d.name, "zip": zip_path.name,
           "zip_bytes": zip_path.stat().st_size, **{k: meta[k] for k in meta}, "ok": False}
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".dng"))
        if len(names) < MIN_INPUTS + MIN_MEAN:
            raise ValueError(f"{zip_path.name}: {len(names)} frames")
        middle = len(names) // 2
        _, mid, info = _read(z, names[middle])
        blacks = _plane_black(info)
        quad_black = sum(blacks.values())
        shapes = {key: plane.shape for key, plane in cfa_split(_visible(mid, info)).items()}
        ref_crop = quad_luminance(_visible(mid, info))
        ref = Reference(match_luminance(_visible(mid, info), quad_black))
        means: dict[str, list[float]] = {k: [] for k in blacks}
        maps, unmatched, dropped = {}, [], {}
        for i, n in enumerate(names):                                 # pass 1: plane means, motion
            full, fi = (mid, info) if i == middle else _read(z, n)[1:]
            if fi["raw_shape"] != info["raw_shape"] or fi["black"] != info["black"] or fi["white"] != info["white"]:
                dropped[n] = f"levels or size differ: {fi['raw_shape']}, black {fi['black']}, white {fi['white']}"
                for key in blacks:                                    # the median's value: neutral to the outlier test
                    means[key].append(float(np.median(means[key])) if means[key] else blacks[key])
                continue
            vis = _visible(full, info)
            for key, plane in cfa_split(vis).items():
                means[key].append(float(plane.mean()))
            dy, dx = frame_shift(quad_luminance(vis), ref_crop)       # x[i] ~ ref[i + d]: lum(p - d) ~ ref(p)
            reg = register(match_luminance(vis, quad_black), ref, (-dy, -dx))
            if reg is None:
                unmatched.append(i)
            else:
                maps[i] = reg
            if log and (i + 1) % 50 == 0:
                log(f"  {zip_path.name}: pass 1 {i + 1}/{len(names)}")
        del mid
        keep = np.ones(len(names), dtype=bool)
        for key, b in blacks.items():
            keep &= outlier_frames(np.asarray(means[key]) - b)
        for i in np.flatnonzero(~keep):
            dropped.setdefault(names[i], "plane mean off the burst's")
        settled = [i for i in range(len(names)) if names[i] not in dropped]
        failed = [i for i in unmatched if names[i] not in dropped]
        registered = 2 * len(failed) <= len(settled)                  # most frames register
        reg_rec = {"registered": registered, "reference": Path(names[middle]).name, "patches": len(ref.patches),
                   "unregistered": [Path(names[i]).name for i in failed]}
        if registered:
            for i in failed:
                dropped[names[i]] = "not registered: too few patches matched the reference"
        else:
            reg_rec["why"] = (f"{len(failed)} of {len(settled)} frames do not register (too little structure to "
                              "match): taken as they lie")
            maps = {i: {"T": np.eye(3), "patches": 0, "matched": 0, "rms_quads": None} for i in settled}
        held = [i for i in settled if names[i] not in dropped]
        tmaps = {i: maps[i]["T"] for i in held}
        corners = grid_corners(ref.shape)
        anchor, near = choose_inputs(tmaps, corners, middle) if held else (middle, [])
        inputs = [near[j] for j in spread(len(near), min(INPUTS, len(near), len(held) - MIN_MEAN))]
        target = np.mean([tmaps[i] for i in inputs], axis=0) if inputs else np.eye(3)
        dist = {i: displacement_px(tmaps[i], target, corners) for i in held}
        for i in held:
            if i not in inputs and dist[i] > MAX_WARP_PX:
                dropped[names[i]] = f"{dist[i]:.1f} px from the inputs' geometry, over {MAX_WARP_PX}"
        in_mean = [i for i in held if i not in inputs and names[i] not in dropped]
        reg_rec.update(anchor=Path(names[anchor]).name, near_anchor=len(near), target_affine_quads=_affine(target),
                       inputs_px={Path(names[i]).name: round(dist[i], 4) for i in inputs},
                       mean_px_max=round(max((dist[i] for i in in_mean), default=0.0), 4),
                       frames={Path(names[i]).name: {"affine": _affine(maps[i]["T"]), "patches": maps[i]["patches"],
                                                     "matched": maps[i]["matched"],
                                                     "rms_quads": None if maps[i]["rms_quads"] is None
                                                     else round(maps[i]["rms_quads"], 4)} for i in sorted(maps)})
        rec.update(levels={"black": info["black"], "white": info["white"], "pattern": info["pattern"]},
                   raw_shape=info["raw_shape"], visible=info["visible"], frames=len(names), dropped=dropped,
                   registration=reg_rec, inputs=[], mean_frames=[Path(names[i]).name for i in in_mean])
        if len(inputs) < MIN_INPUTS or len(in_mean) < MIN_MEAN:     # inputs: the frames copied out, none here
            rec["why"] = (f"{len(held)} of {len(names)} frames hold, {len(near)} of them within {INPUT_PX} px of "
                          f"one: {len(inputs)} inputs, {len(in_mean)} in the mean")
            return _finish(d, rec, t0)
        to_target = np.linalg.inv(QUAD @ target @ np.linalg.inv(QUAD))
        warps = {i: QUAD @ tmaps[i] @ np.linalg.inv(QUAD) @ to_target
                 for i in in_mean}                                    # mosaic: target point -> where frame i has it
        near_set, at = set(near), {i: t for t, i in enumerate(in_mean)}
        law_frames = [i for i in sorted(inputs + in_mean) if i in near_set]
        rec["law_frames"] = len(law_frames)
        stats = {key: PlaneStats(b, float(info["white"]), KEEP_FOR_SHAPE,
                                 scales=frame_scales(np.asarray(means[key])[law_frames] - b),
                                 raw_frame_means=[means[key][i] for i in law_frames]) for key, b in blacks.items()}
        scales = {key: frame_scales(np.asarray(means[key])[in_mean] - b) for key, b in blacks.items()}
        merges = {key: KernelMerge(s) for key, s in shapes.items()}
        for c, i in enumerate(sorted(inputs + in_mean)):              # pass 2: the noise law, the merge
            full = _read(z, names[i])[1]
            for key, plane in cfa_split(_visible(full, info)).items():
                if i in near_set:
                    stats[key].add(plane)
                if i in at:
                    val, dy, dx, valid = nearest_samples(plane, plane_map(warps[i], int(key[0]), int(key[1])))
                    val -= blacks[key]
                    val *= scales[key][at[i]]
                    merges[key].add(val, dy, dx, valid)
            if log and (c + 1) % 50 == 0:
                log(f"  {zip_path.name}: pass 2 {c + 1}/{len(inputs) + len(in_mean)}")
        planes = {key: ps.finish(fit_span(ps)) for key, ps in stats.items()}
        del stats
        merged, masks, cover = {}, {}, {}
        for key in list(merges):
            mg = merges.pop(key)
            merged[key], cover[key] = mg.finish(), mg.cover
            masks[key] = pair_masks(merged[key][0], np.isfinite(merged[key][1]), info["white"] - blacks[key])
            del mg
        laws = {key: p["fit"] for key, p in planes.items() if p["fit"].get("ok")}
        pairs, kept, template = {}, [], None
        for i in inputs if laws else []:                              # pass 3: each input against the mean
            full = _read(z, names[i])[1]
            per = {key: pair_excess(plane.astype(np.float64) - blacks[key], *merged[key], laws[key], masks[key])
                   for key, plane in cfa_split(_visible(full, info)).items() if key in laws}
            worst = [max(e for e in (p["edge"], p["flat"]) if e is not None) for p in per.values()
                     if p["edge"] is not None or p["flat"] is not None]
            ex = float(np.median(worst)) if worst else None
            pairs[Path(names[i]).name] = {"excess": ex, "planes": per}
            if ex is not None and ex <= MAX_PAIR_EXCESS:
                kept.append(i)
                if template is None:
                    template = full
        frames_dir = d / "frames"
        ok = bool(laws) and len(kept) >= MIN_INPUTS
        if ok:                                                        # the inputs that pair out, as they are
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i in kept:
                data = z.read(names[i])
                dst = frames_dir / Path(names[i]).name
                if not keeps_reserve(dst.parent, len(data), reserve_bytes(RESERVE_ENV, RESERVE_GB)):
                    raise OSError(f"writing under {dst.parent} would take the drive below {RESERVE_ENV}")
                tmp = dst.with_name(dst.name + ".part")
                tmp.write_bytes(data)
                os.replace(tmp, dst)
            rec["template"] = f"frames/{Path(names[kept[0]]).name}"
    names_kept = {Path(names[i]).name for i in kept}
    rec["inputs"] = [Path(names[i]).name for i in kept]
    rec["inputs_dropped"] = {n: (f"pair excess {p['excess']:.2f} over {MAX_PAIR_EXCESS}" if p["excess"] is not None
                                 else "no plane judged") for n, p in pairs.items() if n not in names_kept}
    rec["pairs"] = pairs
    excess = sorted(p["excess"] for p in pairs.values() if p["excess"] is not None)
    rec["pair_excess"] = [excess[0], float(np.median(excess)), excess[-1]] if excess else None
    reg_rec["cover_min"] = min(int(n.min()) for n in cover.values())
    keys = ("n_frames", "mean_dn_above_black", "clipped_fraction", "fit", "flat_fraction", "edge_variance_ratio",
            "row_noise_ratio", "column_noise_ratio", "drift_sigma", "frame_scale_min_max")
    rec["planes"] = {k: {f: v for f, v in p.items() if f in keys} for k, p in planes.items()}
    for key, (t, v) in merged.items():
        b = masks[key]["base"]
        rec["planes"][key]["n_eff_p1_p50"] = ([round(float(q), 2) for q in np.percentile(1.0 / v[b], [1.0, 50.0])]
                                              if b.any() else None)
    n50 = [p["n_eff_p1_p50"][1] for p in rec["planes"].values() if p["n_eff_p1_p50"]]
    rec["n_eff"] = float(np.median(n50)) if n50 else None
    if not laws:
        rec["why"] = "no plane gave a noise law (none fitted on its flat pixels): nothing to judge the pairs against"
    elif not ok:
        rec["why"] = (f"{len(kept)} of {len(inputs)} inputs within {MAX_PAIR_EXCESS} noise variances of the mean "
                      f"(pair excess {rec['pair_excess']}): something in the burst moved")
    else:
        rec["ok"] = True
        mean = template.astype(np.float32)
        vis = _visible(mean, info)
        for key, (t, _) in merged.items():
            n = cover[key]
            m = t + blacks[key]
            if not n.all():                                          # a corner no frame covers: its nearest pixel's
                m = m[tuple(ndimage.distance_transform_edt(n == 0, return_distances=False, return_indices=True))]
            vis[int(key[0])::2, int(key[1])::2] = m
        np.clip(mean, 0, info["white"], out=mean)
        p = d / "mean.npy"
        if not keeps_reserve(d, mean.nbytes, reserve_bytes(RESERVE_ENV, RESERVE_GB)):
            raise OSError(f"writing under {d} would take the drive below {RESERVE_ENV}")
        tmp = d / "mean.tmp.npy"
        np.save(tmp, mean)
        os.replace(tmp, p)
    if frames_dir.is_dir():                 # a burst keeps its paired inputs' frames and no others
        for f in frames_dir.glob("*"):
            if not ok or f.name not in names_kept:
                f.unlink()
        if not ok:
            frames_dir.rmdir()
    if not ok and (d / "mean.npy").exists():   # a stack reduced again that no longer pairs
        (d / "mean.npy").unlink()
    return _finish(d, rec, t0)


def _clean(v):
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _finish(d: Path, rec: dict, t0: float) -> dict:
    rec["seconds"] = round(time.time() - t0, 1)
    rec = _clean(rec)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "mean.json.part"
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, d / "mean.json")
    return rec


def _task(args: tuple) -> tuple[str, dict | None, str]:
    zp, meta, out_root, delete = args
    try:
        rec = reduce_instance(Path(zp), meta, Path(out_root))
        if delete and rec["ok"]:
            consumed_marker(zp).write_text(json.dumps({"stack": rec["stack"], "instance": rec["instance"]}) + "\n",
                                           encoding="utf-8")
            Path(zp).unlink()
            rec["zip_deleted"] = True
        return zp, rec, ""
    except Exception as e:  # noqa: BLE001 - one zip that fails; the rest go on
        return zp, None, f"{type(e).__name__}: {e}"[:300]


def done(root: Path | None = None) -> dict[str, dict]:
    """instance -> its mean.json, for every stack reduced."""
    out = {}
    for j in sorted(stack_root(root).glob("*/mean.json")):
        rec = json.loads(j.read_text(encoding="utf-8"))
        out[rec["instance"]] = rec
    return out


def reduce_all(root: Path | None = None, workers: int = 2, background: bool = True, delete_zips: bool = False,
               instances: list[str] | None = None, log=print) -> dict:
    """Every complete zip whose stack is not reduced yet, `workers` processes at once (below normal priority with
    `background`); with `delete_zips`, a zip whose burst pairs is deleted once its stack is written (one that does
    not pair is kept, for a second look). Returns the tally."""
    root = Path(root or data_root())
    base = root / "raw" / SIDD
    meta = sidd_metadata(base / METADATA)
    have = done(root)
    todo = [p for p in zips(root) if ZIP_RE.match(p.name).group(1) not in have
            and (not instances or ZIP_RE.match(p.name).group(1) in instances)]
    out_root = stack_root(root)
    log(f"{SIDD}: {len(have)} stack(s) reduced, {len(todo)} zip(s) to reduce with {workers} worker(s)")
    tasks = [(str(p), meta.get(ZIP_RE.match(p.name).group(1)), str(out_root), delete_zips) for p in todo]
    t0, ok, bad, failed = time.time(), 0, 0, 0

    def take(n: int, zp: str, rec: dict | None, err: str) -> None:
        nonlocal ok, bad, failed
        if err:
            failed += 1
            log(f"{SIDD}: [{n}/{len(tasks)}] {Path(zp).name} failed: {err}")
            return
        ok += rec["ok"]
        bad += not rec["ok"]
        log(f"{SIDD}: [{n}/{len(tasks)}] {rec['stack']}: {'pairs' if rec['ok'] else 'no pairs: ' + rec.get('why', '')}"
            f"; {len(rec.get('inputs', []))} inputs, {len(rec.get('mean_frames', []))} in the mean, n_eff "
            f"{rec.get('n_eff')}, pair excess {rec.get('pair_excess')}, {rec['seconds']:.0f} s "
            f"({time.time() - t0:.0f} s elapsed)")

    if workers <= 1 or len(tasks) <= 1:
        if background:
            _lower_priority()
        for n, t in enumerate(tasks, 1):
            take(n, *_task(t))
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_lower_priority if background else None) as pool:
            for n, r in enumerate(pool.imap_unordered(_task, tasks), 1):
                take(n, *r)
            pool.close()
            pool.join()
    tally = {"reduced": ok + bad, "pairs": ok, "no_pairs": bad, "failed": failed, "before": len(have),
             "seconds": round(time.time() - t0, 1)}
    log(f"{SIDD}: {tally}")
    return tally


# ---- pairs -------------------------------------------------------------------------------------------------------

def sidd_pairs(root: Path | None = None, log=None) -> list:
    """Every (burst mean, input frame) pair of the reduced stacks that pair (mean.json ok): the split hashed on the
    stack's SIDD scene (the source's clean.scene), the pair's scene the stack."""
    from .label import Item, file_filter
    from .manifest import by_id, load_manifest
    from .realpairs import Pair, _split_of
    root = Path(root or data_root())
    entry = by_id(load_manifest()).get(SIDD)
    licence = entry.license if entry else ""
    keep = file_filter(SIDD)
    pairs, skipped = [], {}
    for inst, rec in done(root).items():
        if not rec.get("ok"):
            skipped[rec["stack"]] = rec.get("why", "")
            continue
        ref = f"{STACK_DIR}/{rec['stack']}/mean.npy"
        if not keep(ref):
            continue
        split = _split_of(Item(SIDD, ref))
        meta = {k: rec.get(k) for k in ("instance", "scene", "camera", "iso", "shutter", "illuminant_k", "brightness",
                                         "n_eff")}
        meta["mean_frames"] = len(rec["mean_frames"])
        for f in rec["inputs"]:
            pairs.append(Pair(SIDD, rec["stack"], ref, f"{STACK_DIR}/{rec['stack']}/frames/{f}", split,
                              iso=float(rec["iso"]) if rec.get("iso") is not None else None, licence=licence,
                              meta=dict(meta, pair_excess=rec.get("pairs", {}).get(f, {}).get("excess"))))
    if skipped and log:
        log(f"{SIDD}: {len(skipped)} stack(s) do not pair: " + "; ".join(f"{k}: {v}" for k, v in list(skipped.items())[:5]))
    return pairs
