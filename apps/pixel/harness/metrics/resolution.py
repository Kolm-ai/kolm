"""Spatial resolution: slanted-edge SFR (ISO 12233 style) and high-frequency texture fidelity."""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..colorspace import luminance
from ..scene import Scene, crop

EDGE_MARGIN = 4
TEXTURE_MARGIN = 2
HF_CUTOFF_CPP = 0.25


def edge_profile(roi: np.ndarray, oversample: int = 4) -> tuple[np.ndarray, float, float]:
    """roi: 2-D luminance with one near-vertical dark->bright edge.
    Returns (esf sampled every 1/oversample px, dark level, bright level)."""
    h, w = roi.shape
    sm = uniform_filter1d(roi, 3, axis=1, mode="nearest")
    tail = max(2, w // 5)
    xs_edge, ys_edge = [], []
    for y in range(h):
        row = sm[y]
        dark, bright = row[:tail].mean(), row[-tail:].mean()
        if bright - dark < 1e-6:
            continue
        mid = 0.5 * (dark + bright)
        cross = np.nonzero((row[:-1] < mid) & (row[1:] >= mid))[0]
        if cross.size == 0:
            continue
        d = np.clip(np.diff(row), 0, None)
        xc0 = ((np.arange(w - 1) + 0.5) * d).sum() / max(d.sum(), 1e-12)
        i = int(cross[np.argmin(np.abs(cross + 0.5 - xc0))])
        xs_edge.append(i + (mid - row[i]) / (row[i + 1] - row[i]))
        ys_edge.append(y)
    if len(xs_edge) < 4:
        raise ValueError("no usable edge in ROI")
    a, b = np.polyfit(ys_edge, xs_edge, 1)
    Y, X = np.mgrid[0:h, 0:w]
    dist = (X - (a * Y + b)) / math.sqrt(1.0 + a * a)
    bins = np.round(dist * oversample).astype(int)
    bins -= bins.min()
    n = int(bins.max()) + 1
    sums = np.bincount(bins.ravel(), weights=roi.ravel(), minlength=n)
    counts = np.bincount(bins.ravel(), minlength=n)
    nz = counts > 0
    esf = np.interp(np.arange(n), np.nonzero(nz)[0], sums[nz] / counts[nz])
    t = max(1, n // 10)
    return esf, float(esf[:t].mean()), float(esf[-t:].mean())


def sfr(roi: np.ndarray, oversample: int = 4) -> tuple[np.ndarray, np.ndarray]:
    esf, _, _ = edge_profile(roi, oversample)
    lsf = np.diff(esf)
    n = lsf.size
    peak = int(np.argmax(np.abs(lsf)))
    win = np.hamming(2 * n)[n - peak:2 * n - peak]  # Hamming centred on the LSF peak
    spec = np.abs(np.fft.rfft(lsf * win))
    spec = spec / max(spec[0], 1e-12)
    return np.fft.rfftfreq(n, d=1.0 / oversample), spec


def mtf50(roi: np.ndarray, oversample: int = 4) -> float:
    f, s = sfr(roi, oversample)
    below = np.nonzero(s < 0.5)[0]
    if below.size == 0:
        return float(f[-1])
    i = int(below[0])
    if i == 0:
        return 0.0
    return float(f[i - 1] + (0.5 - s[i - 1]) * (f[i] - f[i - 1]) / (s[i] - s[i - 1]))


def mtf50_edge(frame: np.ndarray, scene: Scene) -> float:
    return mtf50(luminance(crop(frame, scene.regions["slanted_edge"], EDGE_MARGIN)))


def texture_hf_fidelity(frame: np.ndarray, truth: np.ndarray, scene: Scene,
                        cutoff_cpp: float = HF_CUTOFF_CPP) -> float:
    """Fraction of the truth's high-frequency texture that is present in the candidate:
    Re<C,T> / <T,T> over |f| >= cutoff. Blur lowers it; noise and invented detail average to zero."""
    c = luminance(crop(frame, scene.regions["texture"], TEXTURE_MARGIN))
    t = luminance(crop(truth, scene.regions["texture"], TEXTURE_MARGIN))
    h, w = t.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    C = np.fft.fft2((c - c.mean()) * win)
    T = np.fft.fft2((t - t.mean()) * win)
    R = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :])
    band = R >= cutoff_cpp
    denom = np.sum(np.abs(T[band]) ** 2)
    return float(np.real(np.sum(C[band] * np.conj(T[band]))) / max(denom, 1e-18))
