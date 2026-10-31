"""Sub-pixel registration of a real pair's reference: the part of the shift the integer alignment leaves.

The real store and the real-pair evaluation sets take a pair's reference at the integer shift that fits it best
(realpairs.align), so a target can sit up to half a pixel off its input. Training on such targets teaches the blur
of that shift: the best prediction of a target whose position is uncertain by s px is the sharp image blurred by s.
On 10-29 a probe read a median of 0.17 px on RawNIND's train pairs (0.18 px at ISO below 1600, the quadrants
agreeing within 0.11 px, so a shift of the whole frame) and 0.024 px on HDR+'s.

`measure` reads the residual on the aligned overlap: the cross-correlation of the two band-passed log luminances,
its peak fitted to a Gaussian on each axis, then measured again with the target moved by the first estimate (the
fit's bias is least at a peak near zero), on the whole overlap and on its four quadrants, whose spread says
whether it is one shift. `fourier_shift` moves a target by it: a phase ramp on its spectrum, an all-pass, so the
target moves without the blur interpolation would add. A tile is moved on a window with real margin around it,
mirrored to twice its size less a sample on each axis, so its periodic extension is continuous in value (a padding
of the window alone wraps its last row onto its first, which rings into the tile) and its grid odd, with no Nyquist
term left ambiguous. On a scene band-limited to 0.45 cycles a pixel, a 256 px tile moved by (-0.28, 0.33) px on a
32 px margin is within 1e-3 of the scene moved whole (0.1 % of its range, float16's precision), where a reflected
padding of the same margin was within 2e-3. The transforms are real and single precision, on sizes whose factors
are 2, 3, 5 and 7 (the window's far margin grown to one, the measured overlap cropped to one): a prime factor
such as the 503 in RawNIND's 2012 rows made a transform two to three times slower. A stack merge's holes (NaN) read
as their nearest pixel for the move, so no step at their edge rings into the target."""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft

from ..colorspace import luminance

BAND = (1.0, 4.0)         # px: the Gaussian sigmas of the band-pass, fine detail kept and shading taken out
LOG_GAIN = 4096.0         # model.data.LOG_GAIN: the band-pass reads the log domain the network and the gates read
MAX_SIDE = 3072           # px: the central crop of a larger overlap that is read (a shift of the whole frame)
TRUST_SPREAD = 0.3        # px: the quadrants' estimates agree within this for the residual to be applied
MAX_RESIDUAL = 1.0        # px: a larger residual on either axis means the integer alignment itself is off
MARGIN = 32               # px of real data read around a tile before it is moved, so its seams are far from it
BLOCK = 512               # px: the blocks a whole frame is moved in (each on its own margin)


def _smooth(n: int) -> bool:
    for f in (2, 3, 5, 7):
        while n % f == 0:
            n //= f
    return n == 1


def _fast_crop(n: int) -> int:
    """The largest size at most n whose factors are 2, 3, 5 and 7."""
    while n > 1 and not _smooth(n):
        n -= 1
    return n


def _fast_window(n: int) -> int:
    """The smallest window of at least n px whose mirrored extension (2 n - 1) has factors 2, 3, 5 and 7 only."""
    while not _smooth(2 * n - 1):
        n += 1
    return n


def _band(lum: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    a = np.log1p(LOG_GAIN * np.maximum(np.asarray(lum, dtype=np.float32), 0.0))
    return gaussian_filter(a, band[0]) - gaussian_filter(a, band[1])


def _fit(c: np.ndarray, i: int) -> float:
    """The sub-sample offset of a peak at i: a Gaussian (a parabola in the log) through it and its neighbours."""
    n = c.size
    lo, mid, hi = (float(np.log(max(c[k % n], 1e-300))) for k in (i - 1, i, i + 1))
    d = lo - 2.0 * mid + hi
    return 0.0 if d >= 0 else 0.5 * (lo - hi) / d


def _peak(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(dy, dx) with b(p + d) ~ a(p): the Hann-windowed cross-correlation's peak, fitted on each axis."""
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    c = sfft.irfft2(np.conj(sfft.rfft2(a * win)) * sfft.rfft2(b * win), s=(h, w)).astype(np.float64)
    c = c - c.min() + 1e-12
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    dy, dx = iy + _fit(c[:, ix], iy), ix + _fit(c[iy, :], ix)
    return (dy - h if dy > h / 2 else dy), (dx - w if dx > w / 2 else dx)


def _spectral_shift(a: np.ndarray, d: tuple[float, float]) -> np.ndarray:
    """a (H, W[, C]) read at p + d on its periodic extension: its spectrum times exp(2 pi i f d), in single
    precision (a real transform: on an odd size it is the complex one's real part exactly)."""
    h, w = a.shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    ramp = np.exp(2j * np.pi * (fy * d[0] + fx * d[1])).astype(np.complex64)
    if a.ndim == 3:
        ramp = ramp[..., None]
    spec = sfft.rfftn(np.asarray(a, dtype=np.float32), axes=(0, 1))
    return sfft.irfftn(spec * ramp, s=(h, w), axes=(0, 1))


def _measure_band(la: np.ndarray, lb: np.ndarray) -> tuple[float, float]:
    la = _centre(la, (_fast_crop(la.shape[0]), _fast_crop(la.shape[1])))
    lb = _centre(lb, la.shape)
    d1 = _peak(la, lb)
    d2 = _peak(la, _spectral_shift(lb, d1))
    return d1[0] + d2[0], d1[1] + d2[1]


def _centre(a: np.ndarray, side: int | tuple[int, int]) -> np.ndarray:
    sy, sx = (side, side) if isinstance(side, int) else side
    h, w = a.shape[:2]
    y0, x0 = max(0, (h - sy) // 2), max(0, (w - sx) // 2)
    return a[y0:y0 + min(h, sy), x0:x0 + min(w, sx)]


def measure(x: np.ndarray, t: np.ndarray, band: tuple[float, float] = BAND) -> dict:
    """The residual of a target t against its input x, both (H, W, 3) linear on the aligned overlap (the target
    as realpairs.match_of gives it, the reference at the integer shift over the gains): shift (dy, dx) with
    t(p + d) ~ x(p), the four quadrants' own, their spread (the largest distance between two), and `trusted`
    when the spread is within TRUST_SPREAD and the shift within MAX_RESIDUAL on both axes. NaN in t (a stack
    merge's holes) reads as x there."""
    xa = np.asarray(x, dtype=np.float32)
    ta = np.asarray(t, dtype=np.float32)
    ta = np.where(np.isnan(ta), xa, ta)
    lx = _band(_centre(luminance(xa), MAX_SIDE), band)
    lt = _band(_centre(luminance(ta), MAX_SIDE), band)
    d = _measure_band(lx, lt)
    h, w = lx.shape
    quads = [_measure_band(lx[sy, sx], lt[sy, sx])
             for sy in (slice(0, h // 2), slice(h // 2, h)) for sx in (slice(0, w // 2), slice(w // 2, w))]
    q = np.asarray(quads)
    spread = float(max(np.hypot(*(q[i] - q[j])) for i in range(4) for j in range(i + 1, 4)))
    return {"shift": [round(float(d[0]), 4), round(float(d[1]), 4)],
            "quadrants": [[round(float(a), 4), round(float(b), 4)] for a, b in quads],
            "spread": round(spread, 4), "band": list(band),
            "trusted": bool(spread <= TRUST_SPREAD and max(abs(d[0]), abs(d[1])) <= MAX_RESIDUAL)}


def residual_of(meta: dict) -> tuple[float, float] | None:
    """A stored pair's residual to apply (meta["subpix"], from `measure`), or None when it has none or it is not
    trusted."""
    sp = meta.get("subpix")
    if not sp or not sp.get("trusted"):
        return None
    return float(sp["shift"][0]), float(sp["shift"][1])


def _mirrored(a: np.ndarray) -> np.ndarray:
    """a (H, W[, C]) and its mirror images, (2H - 1, 2W - 1[, C]): continuous in value across every seam."""
    a = np.concatenate([a, a[::-1][1:]], axis=0)
    return np.concatenate([a, a[:, ::-1][:, 1:]], axis=1)


def _span(p0: int, size: int, total: int) -> tuple[int, int]:
    """The rows (or columns) [a, b) of a window around [p0, p0 + size): MARGIN px before it where the frame has
    them, the far side grown so the window's mirrored extension is a fast size, moved back inside the frame."""
    want = min(total, _fast_window(size + 2 * MARGIN))
    a = max(0, p0 - MARGIN)
    b = min(total, a + want)
    return max(0, b - want), b


def _fill_holes(win: np.ndarray) -> np.ndarray:
    """NaN in a window (in any channel) replaced by the nearest pixel that has none, so the move sees no step."""
    from scipy.ndimage import distance_transform_edt
    hole = np.isnan(win).any(axis=-1) if win.ndim == 3 else np.isnan(win)
    if hole.all():
        return np.zeros_like(win)
    iy, ix = distance_transform_edt(hole, return_distances=False, return_indices=True)
    return win[iy, ix]


def shifted_window(ref: np.ndarray, y0: int, x0: int, tile: int | tuple[int, int],
                   d: tuple[float, float]) -> np.ndarray:
    """ref's window of `tile` px (or (th, tw)) at (y0, x0) read at p + d, as float64: the window and at least
    MARGIN px of the frame around it (as far as the frame goes), mirrored, moved by its spectrum, cropped back.
    NaN stays where it was; for the move it reads as its nearest pixel that is not NaN."""
    th, tw = (tile, tile) if isinstance(tile, int) else tile
    h, w = ref.shape[:2]
    ya, yb = _span(y0, th, h)
    xa, xb = _span(x0, tw, w)
    win = np.asarray(ref[ya:yb, xa:xb], dtype=np.float32)
    hole = np.isnan(win)
    if hole.any():
        win = _fill_holes(win)
    out = _spectral_shift(_mirrored(win), d)
    out = np.asarray(out[y0 - ya:y0 - ya + th, x0 - xa:x0 - xa + tw], dtype=np.float64)
    if hole.any():
        out = np.where(hole[y0 - ya:y0 - ya + th, x0 - xa:x0 - xa + tw], np.nan, out)
    return out


def fourier_shift(a: np.ndarray, d: tuple[float, float]) -> np.ndarray:
    """A whole frame (H, W[, C]) read at p + d, moved in BLOCK px blocks (shifted_window), so its memory stays a
    block's whatever the frame's size. NaN stays where it was."""
    if not (d[0] or d[1]):
        return a
    h, w = a.shape[:2]
    out = np.empty(a.shape, dtype=np.float64)
    for y0 in range(0, h, BLOCK):
        for x0 in range(0, w, BLOCK):
            th, tw = min(BLOCK, h - y0), min(BLOCK, w - x0)
            out[y0:y0 + th, x0:x0 + tw] = shifted_window(a, y0, x0, (th, tw), d)
    return out
