"""Invented detail and pixel provenance.

A candidate pixel is *invented* when it carries structured content that no local linear filter of the truth
explains and that is not noise. The truth is first blurred by the one global Gaussian that best matches the
candidate's high-pass content, so a candidate that merely lost detail is compared against a truth that lost the
same. The luminance is then split into half-octave bands; in each band the candidate is fitted to the truth by
one local gain, and the residual is what the truth does not explain. Noise is spatially white, so its residual
energy sits in the finest band in a fixed proportion (read off the candidate itself on lit pixels where the
truth is flat); structure moves residual energy into the mid bands. Missing detail leaves no residual, so blur
is not invention; a linear sharpening is explained, so halos are not invention either (they are the artifacts
gate's business). Luminance only: a Bayer camera's demosaic puts structure into each channel that cancels in
luminance, so a per-channel test flags a third of the best-case camera while the luminance test flags 0 to
0.5 % of it (isoluminant colour edges, where its chromatic PSF and demosaic do create luminance structure the
truth lacks). Pixels under a landmark that moves during a frame's interval are never flagged: the truth is
instantaneous, a camera integrates, and the temporal gate's motion_region_error judges those pixels.

Outputs: the per-pixel map, its fraction over the frame and per scene region, and `undeclared(flag, declared)`
for a candidate that ships a rendered-pixel map. A candidate that ships no map declares every pixel recovered,
so everything the detector flags is undeclared. `inject` plants labelled hallucinations for validation.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .colorspace import luminance
from .scene import Scene

BAND_SIGMAS_PX = (0.7, 1.0, 1.4, 2.0, 2.8)  # fine = L - G(0.7); mid bands between successive sigmas
BLUR_GRID_PX = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)  # global blur match, coarse pass
BLUR_REFINE_PX = 0.25         # then the two neighbours of the best grid point
MATCH_HP_SIGMA_PX = 4.0       # the match compares high-pass content, where blur acts
FIT_SIGMA_PX = 3.0            # window of the local per-band gain fit
ENERGY_SIGMA_PX = 5.0         # window of the local residual energies
NOISE_SIGMA_PX = 10.0         # window of the fine-band noise level the mid bands are compared against
GAIN_MAX = 4.0
STRUCTURE_RATIO = 3.0         # mid residual energy over its noise expectation before a pixel is structured
EXPLAINED_TOL = 0.03          # plus this fraction of the energy the truth does explain (per-band gains are not exact:
                              # the best-case camera's own residual reaches 0.022 of it on the demo still)
MOTION_GROW_PX = 3.0          # moving objects, swept over the frame interval and grown by this, are not judged here
MOTION_SUBSTEPS = 5
MOTION_MIN_PX = 0.5           # a landmark that moves less than this over the interval counts as static
MIN_CONTRAST = 0.02           # Weber: rms high-pass residual (all bands) over local truth luminance + display black
BLACK = 0.005
FLAT_QUANTILE = 0.25          # truth pixels in the lowest quarter of truth band energy calibrate the noise shape
NOISE_QUANTILE = 0.25         # of the mid/fine ratio over those pixels: invention there pulls the median up, not this
MIN_FLAT_COVER = 0.5          # the candidate must carry residual on at least this share of them, else it is noise-free
KAPPA_CAP = 8.0               # the calibrated noise shape may exceed white noise by at most this factor
                              # (a bilinear demosaic correlates the noise: the best-case camera measures 7.5x)
MAJORITY_SIGMA_PX = 1.5       # a flag must hold over its neighbourhood, so lone noise exceedances drop out
BORDER_PX = 8


def _g(x: np.ndarray, s: float) -> np.ndarray:
    return gaussian_filter(x, s, mode="reflect")


def bands(lum: np.ndarray) -> list[np.ndarray]:
    """[fine, mid_1 .. mid_n]: differences of Gaussians at BAND_SIGMAS_PX."""
    g = [_g(lum, s) for s in BAND_SIGMAS_PX]
    return [lum - g[0]] + [g[i] - g[i + 1] for i in range(len(g) - 1)]


def _kappa_white() -> float:
    """Mid-over-fine residual energy ratio of white noise, the reference the calibration is bounded by."""
    n = np.random.default_rng(0).standard_normal((256, 256))
    b = bands(n)
    return float(np.mean(sum(x * x for x in b[1:])) / np.mean(b[0] * b[0]))


KAPPA_WHITE = _kappa_white()


def residuals(c: np.ndarray, t: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per band: candidate minus the best local gain on the truth, and the explained part."""
    res, expl = [], []
    for xc, xt in zip(bands(c), bands(t)):
        den = _g(xt * xt, FIT_SIGMA_PX)
        eps = 1e-3 * float(np.mean(xt * xt)) + 1e-12
        a = np.clip(_g(xc * xt, FIT_SIGMA_PX) / (den + eps), 0.0, GAIN_MAX)
        res.append(xc - a * xt)
        expl.append(a * xt)
    return res, expl


def match_blur(c: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, float]:
    """The truth blurred by the global Gaussian (sigma in px) whose high-pass content best matches the
    candidate's, after one global gain. Coarse grid, then the two neighbours of the best point."""
    hc = c - _g(c, MATCH_HP_SIGMA_PX)
    cache: dict[float, tuple[float, np.ndarray]] = {}

    def score(s: float) -> float:
        if s not in cache:
            tb = _g(t, s) if s > 0 else t
            hb = tb - _g(tb, MATCH_HP_SIGMA_PX)
            a = float(np.sum(hc * hb) / (np.sum(hb * hb) + 1e-12))
            cache[s] = (float(np.sum((hc - a * hb) ** 2)), tb)
        return cache[s][0]

    best = min(BLUR_GRID_PX, key=score)
    for s in (best - BLUR_REFINE_PX, best + BLUR_REFINE_PX):
        if s >= 0 and score(s) < score(best):
            best = s
    return cache[best][1], float(best)


def noise_shape(e_mid: np.ndarray, e_fine: np.ndarray, t_energy: np.ndarray, lum: np.ndarray) -> tuple[float, float]:
    """(bounded, raw) mid-over-fine residual energy ratio where the truth is flat and the candidate is lit: the
    candidate's own noise spectrum shape, bounded between white noise and KAPPA_CAP times white. Pixels the
    candidate clips to black carry no residual and would read as noise-free, so they do not vote."""
    inner = np.zeros(lum.shape, dtype=bool)
    inner[BORDER_PX:-BORDER_PX, BORDER_PX:-BORDER_PX] = True
    lit = inner & (_g(lum, NOISE_SIGMA_PX) > BLACK)
    if not lit.any():
        return KAPPA_WHITE, 0.0
    flat = lit & (t_energy <= np.quantile(t_energy[lit], FLAT_QUANTILE))
    ok = flat & (e_fine > 0)
    n_flat = int(flat.sum())
    if n_flat == 0 or ok.sum() < MIN_FLAT_COVER * n_flat:
        return KAPPA_WHITE, 0.0
    k = float(np.quantile(e_mid[ok] / e_fine[ok], NOISE_QUANTILE))
    return float(min(max(k, KAPPA_WHITE), KAPPA_CAP * KAPPA_WHITE)), k


def invented_map(frame: np.ndarray, truth: np.ndarray, info: dict | None = None) -> np.ndarray:
    """(H,W,3) linear candidate and truth -> bool (H,W): structured content the truth does not explain.
    `info`, if given, receives matched_blur_px and the noise shape used."""
    c = luminance(frame).astype(np.float64)
    t = luminance(truth).astype(np.float64)
    tb, sigma = match_blur(c, t)
    res, expl = residuals(c, tb)
    e_fine = _g(res[0] ** 2, NOISE_SIGMA_PX)
    e_hp = _g(sum(res) ** 2, ENERGY_SIGMA_PX)  # every band: the residual as a luminance contrast
    e_mid = _g(sum(r * r for r in res[1:]), ENERGY_SIGMA_PX)
    e_expl = _g(sum(x * x for x in expl[1:]), ENERGY_SIGMA_PX)
    t_energy = _g(sum(x * x for x in bands(t)), ENERGY_SIGMA_PX)
    kappa, kappa_raw = noise_shape(e_mid, e_fine, t_energy, c)
    if info is not None:
        info.update(matched_blur_px=sigma, noise_shape=kappa, noise_shape_raw=kappa_raw)
    structured = e_mid > STRUCTURE_RATIO * kappa * e_fine + EXPLAINED_TOL * e_expl
    visible = np.sqrt(e_hp) > MIN_CONTRAST * (_g(t, ENERGY_SIGMA_PX) + BLACK)
    flag = _g((structured & visible).astype(np.float64), MAJORITY_SIGMA_PX) > 0.5
    flag[:BORDER_PX] = flag[-BORDER_PX:] = False
    flag[:, :BORDER_PX] = flag[:, -BORDER_PX:] = False
    return flag


def motion_exclusion(scene: Scene, k: int) -> np.ndarray:
    """Pixels a landmark that moves during frame k's interval can cover, grown by MOTION_GROW_PX. The truth is
    instantaneous and a camera integrates its exposure, so a mover's trajectory blur is no local filter of the
    truth; the temporal gate's motion_region_error judges those pixels instead. Static landmarks stay judged."""
    spec = scene.spec
    t = spec.frame_time(k)
    dt = 1.0 / spec.fps if spec.n_frames > 1 else 0.0
    times = [t + ((j + 0.5) / MOTION_SUBSTEPS - 0.5) * dt for j in range(MOTION_SUBSTEPS)]
    pos = np.stack([scene.landmark_positions(tt) for tt in times])  # (sub, L, 2)
    out = np.zeros(scene._xc.shape, dtype=bool)
    if pos.shape[1] == 0:
        return out
    moved = np.linalg.norm(pos.max(axis=0) - pos.min(axis=0), axis=-1) >= MOTION_MIN_PX
    if not moved.any():
        return out
    for tt in times:
        out |= scene.disk_masks(tt, MOTION_GROW_PX)[moved].any(axis=0)
    return out


def invented_maps(frames: np.ndarray, truth: np.ndarray, scene: Scene | None = None,
                  infos: list[dict] | None = None) -> np.ndarray:
    """(T,H,W,3) -> bool (T,H,W); with a scene, pixels under a moving landmark are never flagged. `infos`, if
    given, receives one dict per frame (see invented_map)."""
    out = []
    for k in range(frames.shape[0]):
        info: dict = {}
        out.append(invented_map(frames[k], truth[k], info))
        if infos is not None:
            infos.append(info)
    out = np.stack(out)
    if scene is not None:
        for k in range(out.shape[0]):
            out[k] &= ~motion_exclusion(scene, k)
    return out


def fraction(flag: np.ndarray, region: tuple[int, int, int, int] | None = None) -> float:
    """Flagged fraction of the interior (border excluded), or of a (y0, y1, x0, x1) region."""
    if region is not None:
        y0, y1, x0, x1 = region
        sub = flag[..., y0:y1, x0:x1]
    else:
        sub = flag[..., BORDER_PX:-BORDER_PX, BORDER_PX:-BORDER_PX]
    return float(sub.mean()) if sub.size else 0.0


def undeclared(flag: np.ndarray, declared: np.ndarray | None) -> np.ndarray:
    """Invented and not declared rendered. No declaration means everything was declared recovered."""
    return flag if declared is None else (flag & ~declared.astype(bool))


def region_breakdown(flag: np.ndarray, declared: np.ndarray | None, scene: Scene) -> dict[str, dict]:
    """Per scene region and the whole interior: measured invented, declared rendered, undeclared fractions."""
    und = undeclared(flag, declared)
    out = {}
    for name, region in [("frame", None)] + list(scene.regions.items()):
        out[name] = {
            "invented": fraction(flag, region),
            "rendered": 0.0 if declared is None else fraction(declared.astype(bool), region),
            "undeclared": fraction(und, region),
        }
    return out


def inject(frame: np.ndarray, rng: np.random.Generator, n_patches: int = 6, size_px: int = 24,
           contrast: float = 0.2, sigma_px: float = 1.0,
           region: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Plant `n_patches` square patches of band-limited value noise (rms `contrast` times the local luminance,
    correlation length `sigma_px`) under raised-cosine windows. Returns the frame with the hallucinations and
    the label mask (window > 0.5), the controllable truth a detector is validated against."""
    out = np.array(frame, dtype=np.float64, copy=True)
    H, W = out.shape[:2]
    y0, y1, x0, x1 = region if region is not None else (BORDER_PX, H - BORDER_PX, BORDER_PX, W - BORDER_PX)
    mask = np.zeros((H, W), dtype=bool)
    lum = luminance(out)
    half = size_px // 2
    ramp = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(size_px) + 0.5) / size_px)  # full raised cosine: a bump
    win = np.outer(ramp, ramp)
    win = win / win.max()
    for _ in range(n_patches):
        cy = int(rng.integers(y0 + half, max(y0 + half + 1, y1 - half)))
        cx = int(rng.integers(x0 + half, max(x0 + half + 1, x1 - half)))
        ys, xs = slice(cy - half, cy - half + size_px), slice(cx - half, cx - half + size_px)
        sub = out[ys, xs]
        if sub.shape[0] != size_px or sub.shape[1] != size_px:
            continue
        pat = gaussian_filter(rng.standard_normal((size_px, size_px)), sigma_px, mode="reflect")
        pat = pat / (pat.std() + 1e-12)
        level = float(np.mean(lum[ys, xs])) + BLACK
        out[ys, xs] = np.clip(sub + (contrast * level * pat * win)[..., None], 0.0, None)
        mask[ys, xs] |= win > 0.5
    return out.astype(frame.dtype, copy=False), mask
