"""The harness's twelve still metrics on a natural tile, with the clean tile as the truth.

The scene metrics read known regions (slanted edge, wedge patches, colour patches, disks). A natural tile
has none, so each metric here estimates the same quantity from the pair (degraded, clean): the names and
units are those of harness.metrics.REGISTRY, the estimators differ and are stated per function. NaN means
the tile holds no pixels the metric needs (no shadows below SHADOW_MAX, nothing above the clip level, no
in-range blocks). Pixel noise is read from the degraded tile's own high-pass residual, as the scene metrics
read it from a uniform patch; clean texture leaks into that residual, so noise is overestimated on busy tiles
by an amount that is constant across parameter draws on the same tile. That reading ("hp", the default) is what
every label and conditioning vector carries.

Evaluation reads noise="residual" instead (residual_noise): the part of the degraded tile no shift-invariant
linear filter of the clean explains. Under "hp" a clean tile against itself reads its own texture as noise, so a
perfect restoration loses shadow SNR and dynamic-range steps to a blurred capture (whose blur took texture, and
so "noise", away); under "residual" it reads no noise, as the scene metrics' truth does."""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

from ..colorspace import delta_e_2000, linear_rgb_to_lab, luminance
from ..metrics import STILL_METRICS
from ..metrics.artifacts import MIN_LAG_PX
from ..metrics.natural import banding_pair, edge_overshoot_pair
from ..metrics.resolution import HF_CUTOFF_CPP
from ..metrics.tone import HIGHLIGHT_TOL, SEPARATION_SIGMAS, SHADOW_MAX, SNR_CAP_DB
from ..scene import WEDGE_VALUES

COLOR_BLOCK = 8        # px: block means stand in for the colour patches (noise averages, bias stays)
SHADING_BLOCK = 16     # px: block-mean luminance ratio stands in for the Lambertian disks
MIN_PX = 64            # a luminance band needs this many pixels to count
MIN_BLOCKS = 4
WEDGE_TOL = 0.10       # a wedge "patch" is the pixels within +-10 % of the wedge value in the clean tile
SHADOW_BINS = 6
NOISE_HP_SIGMA = 1.0   # px: noise = residual of the degraded luminance against its own Gaussian smooth
HP_SIGMA_PX = 2.0      # high-pass before the periodicity autocorrelation
GAIN_SIGMA_PX = 8.0    # px: local gain map that takes tone, shading and exposure out before the halo metric
LP_CPP = 0.06          # cycles/px: phase correlation weight, so blur-killed bands do not add random phase
MTF_BIN_CPP = 1.0 / 64
NOISE_MODES = ("hp", "residual")
RESID_BIN_CPP = 1.0 / 64   # radial bands of the residual reading's linear fit
RESID_BORDER_PX = 4        # the residual reading leaves out a border its frequency-domain move reaches across

_d = np.zeros((33, 33)); _d[16, 16] = 1.0
NOISE_HP_GAIN = float(np.sqrt(np.sum((_d - gaussian_filter(_d, NOISE_HP_SIGMA)) ** 2)))  # white-noise std ratio
del _d


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Translation (dy, dx) of `a` relative to `b`, a(x) = b(x - shift): phase correlation weighted by a
    Gaussian low-pass of LP_CPP, parabolic sub-pixel peak, wrap-around to signed shifts."""
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    X = np.fft.fft2((a - a.mean()) * win) * np.conj(np.fft.fft2((b - b.mean()) * win))
    R = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :])
    r = np.real(np.fft.ifft2(X / np.maximum(np.abs(X), 1e-12) * np.exp(-(R / LP_CPP) ** 2)))
    iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
    m = r[iy, ix]
    out = []
    for idx, n, p, q in ((iy, h, r[(iy - 1) % h, ix], r[(iy + 1) % h, ix]),
                         (ix, w, r[iy, (ix - 1) % w], r[iy, (ix + 1) % w])):
        den = p - 2 * m + q
        d = idx + (0.5 * (p - q) / den if den < 0 else 0.0)
        out.append(d - n if d > n / 2 else d)
    return out[0], out[1]


def _spectra(d: np.ndarray, c: np.ndarray, shift: tuple[float, float] | None = None):
    """Windowed spectra of the mean-subtracted luminances, the degraded one compensated for its global
    translation against the clean (so a shift reads as landmark error, not as lost resolution); `shift` is
    phase_shift(d, c) when the caller has it."""
    h, w = c.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    D, C = np.fft.fft2((d - d.mean()) * win), np.fft.fft2((c - c.mean()) * win)
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
    dy, dx = phase_shift(d, c) if shift is None else shift
    D = D * np.exp(2j * np.pi * (fy * dy + fx * dx))
    return D, C, np.hypot(fy, fx)


def mtf50_cpp(D, C, R, bin_cpp: float = MTF_BIN_CPP) -> float:
    """Transfer function H(f) = Re<D C*> / <C C*> in radial bins, normalised at its lowest bin as an SFR is at
    DC, and the first crossing below 0.5 (linear between bin centres). 0.5 if it never crosses (Nyquist)."""
    nb = int(round(0.5 / bin_cpp))
    idx = np.minimum((R / bin_cpp).astype(int), nb - 1).ravel()
    num = np.bincount(idx, np.real(D * np.conj(C)).ravel(), nb)
    den = np.bincount(idx, (np.abs(C) ** 2).ravel(), nb)
    ok = den > 1e-12
    if ok.sum() < 2 or num[ok][0] <= 0:
        return float("nan")
    H = np.where(ok, num / np.maximum(den, 1e-12), np.nan)
    H = H / H[np.flatnonzero(ok)[0]]
    centres = (np.arange(nb) + 0.5) * bin_cpp
    below = np.flatnonzero(ok & (H < 0.5))
    if below.size == 0:
        return 0.5
    i = int(below[0])
    if i == 0:
        return 0.0
    j = int(np.flatnonzero(ok[:i])[-1])
    return float(centres[j] + (H[j] - 0.5) / max(H[j] - H[i], 1e-12) * (centres[i] - centres[j]))


def texture_hf_fidelity(D, C, R, cutoff: float = HF_CUTOFF_CPP) -> float:
    band = R >= cutoff
    return float(np.real(np.sum(D[band] * np.conj(C[band]))) / max(float(np.sum(np.abs(C[band]) ** 2)), 1e-18))


def _noise(dl: np.ndarray) -> np.ndarray:
    """Per-pixel noise sample: the degraded luminance minus its own Gaussian smooth, scaled so white
    noise of std s gives std s."""
    return (dl - gaussian_filter(dl, NOISE_HP_SIGMA)) / NOISE_HP_GAIN


def residual_noise(dl: np.ndarray, cl: np.ndarray, shift: tuple[float, float]):
    """noise="residual": what of the degraded luminance no shift-invariant linear filter of the clean explains.
    dl is moved into cl's frame by `shift` (phase_shift of the two) and predicted from cl by real least-squares
    gains in RESID_BIN_CPP radial bands. The gains are fitted on the tiles' own Hann-windowed spectra with the shift
    compensated, as _spectra does (on a symmetric padding the mirrored copies move the other way and would pull the
    gains down); the move and the prediction run on symmetric 2x padding so the periodic FFT sees no seam. The
    noise sample is the moved tile's high-pass (_noise) and q the prediction's, so _band_std
    reads the part of the one the other does not explain. Returns (moved dl, cl, (n, q)), each without a
    RESID_BORDER_PX border. The clean tile against itself reads no noise (the SNR cap), blur and sharpening read
    none (a linear filter explains them), and on a uniform patch, where the clean has nothing to predict with, the
    reading is the default high-pass one. A sub-pixel move leaves a floor near 34 dB on a 256 tile (the padding's
    mirrored copies ring into the interior), well above what any real pair reads."""
    h, w = cl.shape
    nb = int(round(0.5 / RESID_BIN_CPP))
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
    win = np.outer(np.hanning(h), np.hanning(w))
    D = np.fft.fft2((dl - dl.mean()) * win) * np.exp(2j * np.pi * (fy * shift[0] + fx * shift[1]))
    C = np.fft.fft2((cl - cl.mean()) * win)
    idx = np.minimum((np.hypot(fy, fx) / RESID_BIN_CPP).astype(int), nb - 1).ravel()
    num = np.bincount(idx, np.real(D * np.conj(C)).ravel(), nb)
    den = np.bincount(idx, (np.abs(C) ** 2).ravel(), nb)
    g = np.where(den > 1e-30, num / np.maximum(den, 1e-30), 0.0)
    pad = ((0, h), (0, w))
    fy, fx = np.fft.fftfreq(2 * h)[:, None], np.fft.fftfreq(2 * w)[None, :]
    Dp = np.fft.fft2(np.pad(dl, pad, mode="symmetric")) * np.exp(2j * np.pi * (fy * shift[0] + fx * shift[1]))
    Cp = np.fft.fft2(np.pad(cl, pad, mode="symmetric"))
    gp = g[np.minimum((np.hypot(fy, fx) / RESID_BIN_CPP).astype(int), nb - 1)]
    dm = np.real(np.fft.ifft2(Dp))[:h, :w]
    pr = np.real(np.fft.ifft2(Cp * gp))[:h, :w]
    b = (slice(RESID_BORDER_PX, h - RESID_BORDER_PX), slice(RESID_BORDER_PX, w - RESID_BORDER_PX))
    return dm[b], cl[b], (_noise(dm)[b], _noise(pr)[b])


def _band_std(n: np.ndarray, q: np.ndarray | None, m: np.ndarray) -> float:
    """Noise std over the band mask m: of the noise sample n, or with q (residual_noise's prediction) of n's
    least-squares residual on q, one gain per band, so a tone curve's local slope is not read as noise."""
    a = n[m]
    if q is None:
        return float(a.std())
    b = q[m]
    a, b = a - a.mean(), b - b.mean()
    bb = float(b @ b)
    return float((a - float(a @ b) / bb * b).std()) if bb > 1e-30 else float(a.std())


def shadow_snr_db(dl: np.ndarray, cl: np.ndarray, ns=None) -> float:
    """Median over log-spaced shadow bands (clean < SHADOW_MAX) of 20 log10(band mean / band noise std); the
    noise is `ns` = (n, q) of residual_noise when given, else dl's own high-pass (_noise)."""
    sel = (cl < SHADOW_MAX) & (cl > 0)
    if sel.sum() < MIN_PX:
        return float("nan")
    n, q = ns if ns is not None else (_noise(dl), None)
    lc, d, n, q = np.log(cl[sel]), dl[sel], n[sel], (None if q is None else q[sel])
    edges = np.linspace(lc.min(), lc.max() + 1e-9, SHADOW_BINS + 1)
    snrs = []
    for b in range(SHADOW_BINS):
        m = (lc >= edges[b]) & (lc < edges[b + 1])
        if m.sum() < MIN_PX:
            continue
        mean, std = float(d[m].mean()), _band_std(n, q, m)
        snrs.append(SNR_CAP_DB if std < 1e-12 or mean <= 0 else min(SNR_CAP_DB, 20 * math.log10(mean / std)))
    return float(np.median(snrs)) if snrs else float("nan")


def highlight_loss_fraction(dl: np.ndarray, cl: np.ndarray) -> float:
    sel = cl > 1.0
    if sel.sum() < MIN_PX:
        return float("nan")
    return float(np.mean(dl[sel] < HIGHLIGHT_TOL * cl[sel]))


def dr_steps(dl: np.ndarray, cl: np.ndarray, ns=None) -> float:
    """Wedge values present in the clean tile (+-WEDGE_TOL bands) whose degraded means stay ordered and
    separated by SEPARATION_SIGMAS of the band's noise std, counted as adjacent pairs; `ns` as shadow_snr_db's."""
    tol = math.log1p(WEDGE_TOL)
    lc = np.log(np.maximum(cl, 1e-12))
    n, q = ns if ns is not None else (_noise(dl), None)
    st = []
    for v in WEDGE_VALUES:
        m = np.abs(lc - math.log(v)) <= tol
        if m.sum() >= MIN_PX:
            st.append((float(dl[m].mean()), _band_std(n, q, m)))
    if len(st) < 2:
        return float("nan")
    return float(sum(1 for (m0, s0), (m1, s1) in zip(st[:-1], st[1:])
                     if m1 - m0 > max(SEPARATION_SIGMAS * max(s0, s1), 1e-9)))


def _blocks(x: np.ndarray, b: int, op=np.mean) -> np.ndarray:
    h, w = x.shape[0] // b * b, x.shape[1] // b * b
    if x.ndim == 3:
        return op(x[:h, :w].reshape(h // b, b, w // b, b, x.shape[2]), axis=(1, 3))
    return op(x[:h, :w].reshape(h // b, b, w // b, b), axis=(1, 3))


def _in_range_blocks(cl: np.ndarray, b: int) -> np.ndarray:
    """Blocks the scene's patches would occupy: clean luminance above the shadow floor, nothing clipped."""
    return (_blocks(cl, b) > SHADOW_MAX) & (_blocks(cl, b, np.max) <= 1.0)


def delta_e(d: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    """CIEDE2000 between COLOR_BLOCK-pixel block means of the in-range blocks (the colour-patch estimator)."""
    sel = _in_range_blocks(luminance(c), COLOR_BLOCK)
    if sel.sum() < MIN_BLOCKS:
        return float("nan"), float("nan")
    bd, bc = _blocks(np.maximum(d, 0.0), COLOR_BLOCK)[sel], _blocks(np.maximum(c, 0.0), COLOR_BLOCK)[sel]
    de = delta_e_2000(linear_rgb_to_lab(bd), linear_rgb_to_lab(bc))
    return float(np.mean(de)), float(np.max(de))


def _autocorr(x: np.ndarray) -> np.ndarray:
    F = np.fft.fft2(x - x.mean())
    ac = np.real(np.fft.ifft2(np.abs(F) ** 2))
    return np.fft.fftshift(ac / max(ac[0, 0], 1e-18))


def periodicity(dl: np.ndarray, cl: np.ndarray, min_lag: int = MIN_LAG_PX) -> float:
    """Autocorrelation of the high-passed degraded luminance at lags >= min_lag, in excess of the clean's."""
    hd, hc = dl - gaussian_filter(dl, HP_SIGMA_PX), cl - gaussian_filter(cl, HP_SIGMA_PX)
    h, w = cl.shape
    Y, X = np.mgrid[0:h, 0:w]
    Rl = np.hypot(Y - h // 2, X - w // 2)
    sel = (Rl >= min_lag) & (Rl <= min(h, w) / 2)
    return float(max(0.0, np.max(_autocorr(hd)[sel] - _autocorr(hc)[sel])))


def landmark_error_px(d: np.ndarray, c: np.ndarray) -> float:
    """Mean over the three channels of the whole-tile translation against the clean one (lateral colour
    reads as the R and B offsets). Under blur of several px with JPEG and clipping the phase correlation
    peak broadens and the reading is biased high by 1 to 2 px; the identity reads exactly 0."""
    return float(np.mean([math.hypot(*phase_shift(d[..., ch], c[..., ch])) for ch in range(3)]))


def shading_rmse(dl: np.ndarray, cl: np.ndarray) -> float:
    """RMS deviation of the in-range block-mean luminance ratio (degraded / clean) from its median:
    low-frequency gain errors such as uncorrected shading, with the global exposure divided out."""
    sel = _in_range_blocks(cl, SHADING_BLOCK)
    if sel.sum() < MIN_BLOCKS:
        return float("nan")
    ratio = _blocks(dl, SHADING_BLOCK)[sel] / _blocks(cl, SHADING_BLOCK)[sel]
    ratio = ratio / max(float(np.median(ratio)), 1e-12)
    return float(np.sqrt(np.mean((ratio - 1.0) ** 2)))


def gain_matched(d: np.ndarray, dl: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """The degraded tile with its GAIN_SIGMA_PX-scale luminance gain against the clean divided out, so the
    halo metric (which compares plateaus) sees sharpening, not tone curves, shading or exposure."""
    g = gaussian_filter(cl, GAIN_SIGMA_PX) / np.maximum(gaussian_filter(dl, GAIN_SIGMA_PX), 1e-9)
    return d * g[..., None]


def noise_metrics(dist: np.ndarray, ref: np.ndarray, noise: str = "hp") -> dict[str, float]:
    """shadow_snr_db and dr_steps alone, under either noise reading, as tile_metrics(dist, ref, noise) has them
    (the same array as both is the tile against itself, as there)."""
    if noise not in NOISE_MODES:
        raise ValueError(f"noise {noise!r} not in {NOISE_MODES}")
    c = np.asarray(ref, dtype=np.float64)
    cl = luminance(c)
    if dist is ref:
        if noise == "hp":
            return {"shadow_snr_db": shadow_snr_db(cl, cl), "dr_steps": dr_steps(cl, cl)}
        b = (slice(RESID_BORDER_PX, cl.shape[0] - RESID_BORDER_PX), slice(RESID_BORDER_PX, cl.shape[1] - RESID_BORDER_PX))
        cb, n = cl[b], _noise(cl)[b]
        return {"shadow_snr_db": shadow_snr_db(cb, cb, (n, n)), "dr_steps": dr_steps(cb, cb, (n, n))}
    d = np.asarray(dist, dtype=np.float64)
    dl = luminance(d)
    if noise == "hp":
        return {"shadow_snr_db": shadow_snr_db(dl, cl), "dr_steps": dr_steps(dl, cl)}
    sh = phase_shift(luminance(np.clip(d, 0.0, 1.0)), luminance(np.clip(c, 0.0, 1.0)))
    dm, cb, ns = residual_noise(dl, cl, sh)
    return {"shadow_snr_db": shadow_snr_db(dm, cb, ns), "dr_steps": dr_steps(dm, cb, ns)}


def self_metrics(ref: np.ndarray, noise: str = "hp") -> dict[str, float]:
    """tile_metrics of a tile against itself, what every restore target and real pair is labelled with: the
    comparison estimators take their identity values (mtf50 0.5 and texture 1 wherever the tile has the
    spectrum to read them, no overshoot, banding, periodicity or translation, the docstrings' "identical images
    score 0"), and only what the tile itself sets is computed (shadow SNR, dynamic-range steps, highlight loss,
    and which metrics are NaN). The full path spends a fifth of a training sample's CPU time, 120 ms of 18 FFTs
    and window filters, on this answer; the two agree to 1e-12 (tests/test_tilemetrics_identity.py). `noise` as
    tile_metrics'."""
    nm = noise_metrics(ref, ref, noise)
    c = np.asarray(ref, dtype=np.float64)
    cl = luminance(c)
    ckl = luminance(np.clip(c, 0.0, 1.0))
    h, w = ckl.shape
    C = np.fft.fft2((ckl - ckl.mean()) * np.outer(np.hanning(h), np.hanning(w)))
    R = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :])
    de_mean, de_max = delta_e(c, c)
    out = {"mtf50_cpp": mtf50_cpp(C, C, R), "texture_hf_fidelity": texture_hf_fidelity(C, C, R),
           "shadow_snr_db": nm["shadow_snr_db"], "highlight_loss_fraction": highlight_loss_fraction(cl, cl),
           "dr_steps": nm["dr_steps"], "delta_e_mean": de_mean, "delta_e_max": de_max, "edge_overshoot": 0.0,
           "banding_score": 0.0, "periodicity": 0.0, "landmark_error_px": 0.0, "shading_rmse": shading_rmse(cl, cl)}
    assert set(out) == set(STILL_METRICS)
    return out


def tile_metrics(dist: np.ndarray, ref: np.ndarray, noise: str = "hp") -> dict[str, float]:
    """All twelve still metrics of `dist` against the clean `ref` (linear RGB, same shape); the same array as
    both is a tile against itself (self_metrics). `noise` picks the shadow SNR and dynamic-range reading: "hp"
    (the default, what labels and conditioning carry) or "residual" (evaluation; residual_noise)."""
    if noise not in NOISE_MODES:
        raise ValueError(f"noise {noise!r} not in {NOISE_MODES}")
    if dist is ref:
        return self_metrics(ref, noise)
    d, c = np.asarray(dist, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    if d.shape != c.shape:
        raise ValueError(f"shape mismatch {d.shape} vs {c.shape}")
    dl, cl = luminance(d), luminance(c)
    # appearance metrics see both tiles as a display would, clipped to [0, 1]; the tone metrics read the
    # clean tile's real range (over-range pixels are what highlight loss is about)
    dk, ck = np.clip(d, 0.0, 1.0), np.clip(c, 0.0, 1.0)
    dkl, ckl = luminance(dk), luminance(ck)
    sh = phase_shift(dkl, ckl)
    D, C, R = _spectra(dkl, ckl, sh)
    de_mean, de_max = delta_e(d, c)
    if noise == "hp":
        nm = {"shadow_snr_db": shadow_snr_db(dl, cl), "dr_steps": dr_steps(dl, cl)}
    else:
        dm, cb, ns = residual_noise(dl, cl, sh)
        nm = {"shadow_snr_db": shadow_snr_db(dm, cb, ns), "dr_steps": dr_steps(dm, cb, ns)}
    out = {"mtf50_cpp": mtf50_cpp(D, C, R), "texture_hf_fidelity": texture_hf_fidelity(D, C, R),
           "shadow_snr_db": nm["shadow_snr_db"], "highlight_loss_fraction": highlight_loss_fraction(dl, cl),
           "dr_steps": nm["dr_steps"], "delta_e_mean": de_mean, "delta_e_max": de_max,
           "edge_overshoot": float(edge_overshoot_pair(gain_matched(dk, dkl, ckl), ck)),
           "banding_score": float(banding_pair(dk, ck)), "periodicity": periodicity(dkl, ckl),
           "landmark_error_px": landmark_error_px(dk, ck), "shading_rmse": shading_rmse(dl, cl)}
    assert set(out) == set(STILL_METRICS)
    return out
