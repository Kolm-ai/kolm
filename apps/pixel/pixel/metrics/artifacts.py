"""Processing artifacts: codec blocking and sharpening halos."""

from __future__ import annotations

import numpy as np

from ..colorsci import luminance
from ..ops import box_blur, local_min_max

ONE_CODE_VALUE = 1.0 / 255.0


def _project(g: np.ndarray, freq: float) -> float:
    t = np.arange(g.size)
    return float(abs((g * np.exp(-2j * np.pi * freq * t)).sum()) ** 2)


def _period_prominence(profile: np.ndarray, period: int, neighbours=(-5, -4, -3, -2, 2, 3, 4, 5)) -> float:
    """Spectral line at 1/period relative to its neighbouring bins (0 = no line)."""
    g = profile - profile.mean()
    if float((g * g).sum()) < 1e-12 or g.size < 4 * period:
        return 0.0
    n = g.size
    peak = _project(g, 1.0 / period)
    nb = np.mean([_project(g, 1.0 / period + k / n) for k in neighbours])
    return max(0.0, peak / max(nb, 1e-12) - 1.0)


def blocking_score(cand: np.ndarray, ref: np.ndarray, block: int = 8) -> float:
    """log10(1 + prominence) of the 8 px spectral line in the error-gradient profile (0 = no grid).

    Per-column and per-row mean absolute error gradient is projected onto the
    1/8 cycle/px line and compared with its spectral neighbourhood. Codec
    blocking puts a sharp line there (q92 ~0.8, q30 ~1.7); noise, blur, edges
    and sharpening leave the neighbourhood flat (< 0.2) whatever columns they
    land on.
    """
    err = luminance(cand) - luminance(ref)
    gx = np.abs(np.diff(err, axis=1)).mean(axis=0)
    gy = np.abs(np.diff(err, axis=0)).mean(axis=1)
    return float(np.log10(1.0 + 0.5 * (_period_prominence(gx, block) + _period_prominence(gy, block))))


def _prominence(l: np.ndarray) -> np.ndarray:
    """How far a strict local extremum stands proud of its 8 neighbours' mean (0 where not an extremum)."""
    lo, hi = local_min_max(l, 1)
    nb_mean = (box_blur(l, 3) * 9.0 - l) / 8.0
    up = np.where(l >= hi, l - nb_mean, 0.0)
    down = np.where(l <= lo, nb_mean - l, 0.0)
    return np.maximum(np.maximum(up, down), 0.0).astype(np.float32)


def halo_score(cand: np.ndarray, ref: np.ndarray, margin: float = 3.0 * ONE_CODE_VALUE, edge_grad: float = 0.02, near: int = 4) -> float:
    """Fraction of near-edge pixels where the candidate has an extremum standing more than ``margin``
    prouder than any extremum the reference has within 2 px.

    An overshoot or ringing lobe is an extra, or much taller, extremum next to
    an edge. A global gain, a blur, a clip or 8-bit quantisation never adds
    more than a code value of prominence, so they do not count. The margin is
    three code values: the smallest overshoot that reads as a halo.
    """
    lc = luminance(cand)
    lr = luminance(ref)
    gy, gx = np.gradient(lr)
    _lo, near_edge = local_min_max(np.hypot(gx, gy), near)
    near_edge = near_edge > edge_grad
    if not near_edge.any():
        return 0.0
    prom_c = _prominence(lc)
    _lo, prom_r_near = local_min_max(_prominence(lr), 2)
    halo = (prom_c - prom_r_near) > margin
    return float(halo[near_edge].mean())
