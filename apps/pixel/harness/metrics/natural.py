"""Reference-based artefact estimators for natural images (no scene geometry needed).

The scene-based metrics in `artifacts.py` know where the slanted edge is. Fitting gates to a
human-rated database (KADID-10k, TID2013) needs the same quantity measured on photographs, where
the only ground truth is the pristine reference. `edge_overshoot_pair` is calibrated against
`artifacts.edge_overshoot` on the synthetic scene (see `data/derived/gates_fitted.json`
"calibration"): on unsharp-masked frames the two agree to a few percent.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, label, maximum_filter1d, minimum_filter1d

from ..colorspace import linear_to_srgb, luminance, srgb_to_linear

HALF_WINDOW = 4      # px each side; halos further than this from an edge are not "edge overshoot"
MIN_STEP = 0.1       # linear-luminance contrast a window needs to count as an edge
CLIP_GUARD = 0.98    # windows whose reference touches the clip points are skipped

BANDING_BINS = 1024          # bins over the sRGB-encoded reference level: 4 per 8-bit code value
BANDING_MIN_PER_BIN = 64     # pixels a level bin needs before its mean residual counts
BANDING_GRAD_MAX = 0.006     # linear-luminance slope per px below which the reference is "smooth"
BANDING_L_MIN = 0.002        # pixels darker than this are display black, not picture
BANDING_BLACK = 0.005        # adapting luminance = L + display black and flare (sRGB viewing: 0.2 of 80 cd/m2, plus glare)
BANDING_SMOOTH_SIGMA = 1.5   # px, blur before taking the reference gradient
BANDING_ERODE_PX = 4         # keep smooth pixels at least this far from any edge (blur and halo residuals live there)


def _lum(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return luminance(a) if a.ndim == 3 else a


def edge_overshoot_pair(dist: np.ndarray, ref: np.ndarray, half: int = HALF_WINDOW,
                        min_step: float = MIN_STEP, percentile: float = 50.0,
                        clip_guard: float = CLIP_GUARD) -> float:
    """Added overshoot of `dist` relative to `ref`, as a fraction of the local step.

    Both inputs are linear RGB (or luminance) frames of the same shape. For every pixel and both
    axes, a (2*half+1)-px window gives the reference's local min/max (the step) and the distorted
    image's; overshoot is how far the distorted min/max exceeds the reference's, over the step.
    Only windows with a real edge (step >= min_step) whose reference stays inside the clip guard
    are kept; the given percentile over them is returned (the median tracks the slanted-edge ESF
    peak; high percentiles pick up corners and texture). Identical images score 0.
    """
    d, r = _lum(dist), _lum(ref)
    if d.shape != r.shape:
        raise ValueError(f"shape mismatch {d.shape} vs {r.shape}")
    size = 2 * half + 1
    vals = []
    for axis in (0, 1):
        rmax = maximum_filter1d(r, size, axis=axis, mode="nearest")
        rmin = minimum_filter1d(r, size, axis=axis, mode="nearest")
        dmax = maximum_filter1d(d, size, axis=axis, mode="nearest")
        dmin = minimum_filter1d(d, size, axis=axis, mode="nearest")
        step = rmax - rmin
        keep = (step >= min_step) & (rmax < clip_guard) & (rmin > 1.0 - clip_guard)
        over = np.maximum(dmax - rmax, rmin - dmin) / np.maximum(step, 1e-9)
        vals.append(np.maximum(over[keep], 0.0))
    v = np.concatenate(vals)
    return float(np.percentile(v, percentile)) if v.size else 0.0


def unsharp(frame: np.ndarray, gain: float, sigma: float) -> np.ndarray:
    """Unsharp mask on a linear RGB frame: frame + gain * (frame - gaussian(frame, sigma))."""
    from scipy.ndimage import gaussian_filter
    f = np.asarray(frame, dtype=np.float64)
    return f + gain * (f - gaussian_filter(f, (sigma, sigma, 0) if f.ndim == 3 else sigma))


def _band_jump(res: np.ndarray, lum: np.ndarray, lev: np.ndarray, bins: int, min_per_bin: int, max_gap: int,
               black: float) -> float:
    """Largest slope-corrected jump of the per-level mean residual, as Weber contrast, over one region."""
    idx = np.minimum((lev * bins).astype(np.int64), bins - 1)
    cnt = np.bincount(idx, minlength=bins)
    keep = np.flatnonzero(cnt >= min_per_bin)
    if keep.size < 3:
        return 0.0
    mean_res = np.bincount(idx, weights=res, minlength=bins)[keep] / cnt[keep]
    mean_l = np.bincount(idx, weights=lum, minlength=bins)[keep] / cnt[keep]
    k = keep.astype(np.float64)
    worst = 0.0
    for i in range(keep.size - 2):
        h1, h2 = k[i + 1] - k[i], k[i + 2] - k[i + 1]
        if h1 + h2 > max_gap:
            continue
        # change over the next interval minus what the local slope predicts: 0 for a linear trend at
        # any spacing, the band height at a staircase jump; equals |second difference| at unit spacing
        jump = (mean_res[i + 2] - mean_res[i + 1]) - (mean_res[i + 1] - mean_res[i]) * h2 / h1
        worst = max(worst, jump / (mean_l[i + 1] + black))   # positive only: bands steepen the rise, blur softens it
    return float(worst)


def banding_pair(dist: np.ndarray, ref: np.ndarray, bins: int = BANDING_BINS, min_per_bin: int = BANDING_MIN_PER_BIN,
                 grad_max: float = BANDING_GRAD_MAX, l_min: float = BANDING_L_MIN, black: float = BANDING_BLACK,
                 clip_guard: float = CLIP_GUARD, max_gap: int = 12, erode_px: int = BANDING_ERODE_PX) -> float:
    """Largest band step that `dist` adds in the reference's smooth regions, as Weber contrast: the
    quantity `artifacts.banding_score` measures on the synthetic gradient.

    Smooth pixels are those whose Gaussian-blurred reference luminance has a slope below
    `grad_max` per px, at least `erode_px` from any pixel that is not, away from black and the clip
    point. Each connected smooth region (a sky, a wall, a patch: one material, so any tone or colour
    operation is a smooth function of level inside it) is binned by the reference's sRGB-encoded
    luminance (4 bins per 8-bit code value) and the mean residual dist - ref is taken per bin. Along
    the level axis that residual is ~linear for a smooth tone change and a staircase for
    posterisation, so its slope-corrected second difference over consecutive populated bins (gaps
    up to `max_gap` bins) is ~0 for the former and +step at a band edge for the latter (a blurred
    weak edge gives a negative jump, which does not count). Returns the max over regions of
    max jump / (L + black); identical images score 0. Noise inflates the reading;
    blur and sharpening residuals sit near edges, outside the eroded mask.
    """
    d, r = _lum(dist), _lum(ref)
    if d.shape != r.shape:
        raise ValueError(f"shape mismatch {d.shape} vs {r.shape}")
    gy, gx = np.gradient(gaussian_filter(r, BANDING_SMOOTH_SIGMA))
    smooth = (np.hypot(gy, gx) < grad_max) & (r > l_min) & (r < clip_guard)
    if erode_px > 0:
        smooth = binary_erosion(smooth, iterations=erode_px, border_value=1)
    labels, n = label(smooth)
    if n == 0:
        return 0.0
    res, lev = d - r, linear_to_srgb(r)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    worst = 0.0
    for comp in np.flatnonzero(sizes >= 3 * min_per_bin):
        if comp == 0:
            continue
        m = labels == comp
        worst = max(worst, _band_jump(res[m], r[m], lev[m], bins, min_per_bin, max_gap, black))
    return float(worst)


def quantize_srgb(frame: np.ndarray, bits: int) -> np.ndarray:
    """Posterise a linear frame to 2**bits levels in sRGB encoding (what a low-bit-depth display path does)."""
    from ..colorspace import srgb_to_linear
    n = 2 ** bits - 1
    return srgb_to_linear(np.round(linear_to_srgb(np.asarray(frame, dtype=np.float64)) * n) / n)


# ---- no-reference proxies (for the shift check on generated images, which have no reference) ----
NR_FAR = 8                   # px: plateau band ends this far from the window centre
NR_MIN_EDGE_PX = 200         # an image needs at least this many edge windows for a reading
NR_BAND_MAX_GAP_CODES = 16   # adjacent occupied levels further apart than this are two patches, not a band


def edge_overshoot_nr(frame: np.ndarray, half: int = HALF_WINDOW, far: int = NR_FAR, min_step: float = MIN_STEP,
                      percentile: float = 50.0, clip_guard: float = CLIP_GUARD) -> float:
    """No-reference halo proxy. For every pixel and both axes the near window (2*half+1 px) gives
    the local extremes; the two far bands (half+1 .. far px either side) give the plateaus. A clean
    step never exceeds its plateaus inside the near window; a halo does, by (near max - far max)
    or (far min - near min), over the plateau step. Texture and thin lines also exceed their
    surroundings, so the proxy is not zero on pristine photographs: it is read against the KADID
    reference distribution, never against the fitted gate directly. Returns the given percentile
    over edge windows, or nan when the image has too few edges."""
    l = _lum(frame)
    vals = []
    for axis in (0, 1):
        a = np.moveaxis(l, axis, 0)
        pad = np.pad(a, ((far, far), (0, 0)), mode="edge")
        n = a.shape[0]
        far_max = np.full(a.shape, -np.inf)
        far_min = np.full(a.shape, np.inf)
        for off in range(half + 1, far + 1):
            for sgn in (-1, 1):
                sl = pad[far + sgn * off: far + sgn * off + n]
                far_max = np.maximum(far_max, sl)
                far_min = np.minimum(far_min, sl)
        near_max = maximum_filter1d(a, 2 * half + 1, axis=0, mode="nearest")
        near_min = minimum_filter1d(a, 2 * half + 1, axis=0, mode="nearest")
        step = far_max - far_min
        keep = (step >= min_step) & (far_max < clip_guard) & (far_min > 1.0 - clip_guard)
        over = np.maximum(near_max - far_max, far_min - near_min) / np.maximum(step, 1e-9)
        vals.append(np.maximum(over[keep], 0.0))
    v = np.concatenate(vals)
    if v.size < NR_MIN_EDGE_PX:
        return float("nan")
    return float(np.percentile(v, percentile))


NR_BAND_MIN_REGION_PX = 2000   # smallest smooth region the no-reference banding proxy reads
NR_BAND_CLOSE_PX = 2           # closing radius that re-joins a smooth region split by its own band edges
NR_BAND_PLATEAU_PX = 6         # a code run must be at least this long on both sides of a step to be a band edge
NR_BAND_EDGE_HALO_PX = 3       # pixels this close to a band edge count as banded area


def _run_lengths(eq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """eq[:, j] says whether columns j and j+1 hold the same code. Returns, for every pair index j,
    the length of the constant run that ends at column j and the length of the run that starts at
    column j+1 (both >= 1)."""
    rows, n = eq.shape
    idx = np.arange(n)[None, :]
    change = ~eq
    last = np.maximum.accumulate(np.where(change, idx, -1), axis=1)          # last change at pair <= j
    last_before = np.concatenate([np.full((rows, 1), -1), last[:, :-1]], axis=1)
    left = idx - last_before
    nxt = np.minimum.accumulate(np.where(change, idx, n)[:, ::-1], axis=1)[:, ::-1]  # first change at pair >= j
    next_after = np.concatenate([nxt[:, 1:], np.full((rows, 1), n)], axis=1)
    right = next_after - idx
    return left, right


def banding_nr(frame: np.ndarray, grad_max: float = BANDING_GRAD_MAX, l_min: float = BANDING_L_MIN,
               black: float = BANDING_BLACK, clip_guard: float = CLIP_GUARD, erode_px: int = BANDING_ERODE_PX,
               close_px: int = NR_BAND_CLOSE_PX, min_region_px: int = NR_BAND_MIN_REGION_PX,
               plateau_px: int = NR_BAND_PLATEAU_PX, halo_px: int = NR_BAND_EDGE_HALO_PX,
               percentile: float = 95.0) -> dict:
    """No-reference banding proxy: plateau-to-plateau code steps inside the frame's own smooth
    regions, in the spirit of CAMBI (Netflix, Tandon et al. 2021) but on the harness's Weber scale.
    Smooth regions are found as `banding_pair` finds them on the reference (gradient of the
    sigma-1.5 blurred luminance below `grad_max`, not black, not clipped), closed by `close_px` so
    the thin lines a staircase's own steps cut into the mask do not split the region, then eroded.
    The frame's luminance is read as its 8-bit sRGB code; along rows and columns, a change of code
    between two constant runs of at least `plateau_px` pixels is a band edge with Weber step
    (L(hi) - L(lo)) / (L(lo) + black). Noise or dither breaks the runs and reads 0; a clean 8-bit
    gradient reads its one-code step (0.015-0.03, the 8-bit floor also seen by `banding_pair`);
    posterisation reads the codes it skipped. Returns {"step": median band-edge step over all
    band-edge pixels, "step_p95": its `percentile`, "band_area": share of smooth pixels within
    `halo_px` of a band edge, "n_edges": band-edge pixels, "n_regions"}; step is nan when no smooth
    region is large enough, 0 when regions exist but hold no plateau step. A low-contrast real edge
    between two flat areas reads as a band edge too (the dark end of a step wedge, a wall meeting
    its shadow), which is why the headline is the median, not the maximum, and why the proxy is
    read against its distribution on pristine images rather than against the gate."""
    l = _lum(frame)
    gy, gx = np.gradient(gaussian_filter(l, BANDING_SMOOTH_SIGMA))
    smooth = (np.hypot(gy, gx) < grad_max) & (l > l_min) & (l < clip_guard)
    if close_px > 0:
        smooth = binary_erosion(binary_dilation(smooth, iterations=close_px), iterations=close_px, border_value=1)
    if erode_px > 0:
        smooth = binary_erosion(smooth, iterations=erode_px, border_value=1)
    labels, n = label(smooth)
    nan = {"step": float("nan"), "step_p95": float("nan"), "band_area": float("nan"), "n_edges": 0, "n_regions": 0}
    if n == 0:
        return nan
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes[1:] >= min_region_px
    region = keep[labels]
    n_regions = int(keep.sum())
    if n_regions == 0:
        return nan
    code = np.rint(np.clip(linear_to_srgb(np.clip(l, 0.0, 1.0)), 0.0, 1.0) * 255.0).astype(np.int64)
    lut = srgb_to_linear(np.arange(256) / 255.0)
    steps = []
    banded = np.zeros_like(region)
    for axis in (1, 0):
        c = code if axis == 1 else code.T
        reg = region if axis == 1 else region.T
        eq = c[:, 1:] == c[:, :-1]
        left, right = _run_lengths(eq)
        edge = (~eq) & (left >= plateau_px) & (right >= plateau_px) & reg[:, 1:] & reg[:, :-1]
        if not edge.any():
            continue
        lo = np.minimum(c[:, 1:], c[:, :-1])[edge]
        hi = np.maximum(c[:, 1:], c[:, :-1])[edge]
        steps.append((lut[hi] - lut[lo]) / (lut[lo] + black))
        marks = np.zeros_like(c, dtype=bool)
        marks[:, 1:] |= edge
        marks[:, :-1] |= edge
        if halo_px > 0:
            marks = maximum_filter1d(marks, size=2 * halo_px + 1, axis=1)
        banded |= marks if axis == 1 else marks.T
    if not steps:
        return {"step": 0.0, "step_p95": 0.0, "band_area": 0.0, "n_edges": 0, "n_regions": n_regions}
    st = np.concatenate(steps)
    return {"step": float(np.median(st)), "step_p95": float(np.percentile(st, percentile)),
            "band_area": float((banded & region).sum() / max(region.sum(), 1)),
            "n_edges": int(st.size), "n_regions": n_regions}
