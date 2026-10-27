"""Resolution and periodicity from spectra."""

from __future__ import annotations

import numpy as np

from ..ops import gaussian_blur


def _hann2d(H: int, W: int) -> np.ndarray:
    wy = np.hanning(H).astype(np.float32)
    wx = np.hanning(W).astype(np.float32)
    return np.outer(wy, wx)


def radial_mtf(cand: np.ndarray, ref: np.ndarray, nbins: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Cross-spectrum MTF estimate, radially binned, in cycles per pixel.

    MTF(f) = Re<C R*> / <|R|^2>. Noise in the candidate is uncorrelated with the
    reference, so it cancels in expectation instead of inflating the estimate
    the way a plain power ratio would.
    """
    cand = np.asarray(cand, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    H, W = ref.shape
    w = _hann2d(H, W)
    C = np.fft.fft2((cand - cand.mean()) * w)
    R = np.fft.fft2((ref - ref.mean()) * w)
    cross = (C * np.conj(R)).real
    power = (R * np.conj(R)).real
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    fr = np.sqrt(fx**2 + fy**2)
    edges = np.linspace(0.0, 0.5, nbins + 1)
    idx = np.clip(np.digitize(fr, edges) - 1, 0, nbins - 1)
    num = np.bincount(idx.ravel(), weights=cross.ravel(), minlength=nbins)
    den = np.bincount(idx.ravel(), weights=power.ravel(), minlength=nbins)
    mtf = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, mtf


def mtf50(cand: np.ndarray, ref: np.ndarray) -> float:
    """Frequency (cycles/px) where the MTF first drops below 0.5. Identical images give 0.5 (Nyquist)."""
    f, m = radial_mtf(cand, ref)
    m = np.clip(m, 0.0, 1.5)
    for i in range(1, len(m)):
        if m[i] < 0.5:
            if i == 0:
                return 0.0
            t = (m[i - 1] - 0.5) / max(m[i - 1] - m[i], 1e-9)
            return float(f[i - 1] + t * (f[i] - f[i - 1]))
    return 0.5


def _window_periodicity(hp: np.ndarray, min_lag: int) -> float:
    """Max over 2-D lags v of min(ac(v), ac(2v)): a tile repeats, so its autocorrelation peaks at v and again at 2v.

    Unbiased (zero-padded, overlap-normalised) autocorrelation; lags limited to a
    quarter of the window so 2v stays inside half of it.
    """
    hp = hp - hp.mean()
    H, W = hp.shape
    F = np.fft.fft2(hp, s=(2 * H, 2 * W))
    ac = np.fft.ifft2(F * np.conj(F)).real[:H, :W]
    if ac[0, 0] <= 1e-12:
        return 0.0
    dy = np.arange(H)[:, None]
    dx = np.arange(W)[None, :]
    ac = ac / ((H - dy) * (W - dx)) / (ac[0, 0] / (H * W))
    ly, lx = H // 4, W // 4
    if ly < min_lag or lx < min_lag:
        return 0.0
    first = ac[:ly + 1, :lx + 1]
    second = ac[: 2 * ly + 1 : 2, : 2 * lx + 1 : 2]
    both = np.minimum(first, second)
    sel = (np.arange(ly + 1)[:, None] >= min_lag) & (np.arange(lx + 1)[None, :] >= min_lag)
    return float(np.clip(both[sel].max(), 0.0, 1.0)) if sel.any() else 0.0


def tiling_score(lum: np.ndarray, window: int = 192, stride: int | None = None, min_lag: int = 6, highpass_sigma: float = 2.0, min_energy: float = 0.004) -> float:
    """Max over textured windows of the repeated-autocorrelation peak (see ``_window_periodicity``).

    Repeated texture gives a window whose autocorrelation is ~1 at the tile
    period and again at twice it; natural texture is not high at both. Lags
    with a zero component are excluded so a straight edge, which correlates
    with itself along its own direction, does not count. Flat windows are
    skipped. Detectable periods go up to a quarter of the window, which is a
    third of the frame up to 192 px. Brick walls and other truly periodic
    subjects score high by design; the claim scene has none.
    """
    lum = np.asarray(lum, dtype=np.float32)
    hp = lum - gaussian_blur(lum, highpass_sigma)
    H, W = hp.shape
    window = int(min(window, max(48, min(H, W) // 3)))
    stride = max(8, window // 2) if stride is None else int(stride)
    best = 0.0
    for y in range(0, max(1, H - window + 1), stride):
        for x in range(0, max(1, W - window + 1), stride):
            win = hp[y : y + window, x : x + window]
            if win.std() < min_energy:
                continue
            best = max(best, _window_periodicity(win, min_lag))
    return best
