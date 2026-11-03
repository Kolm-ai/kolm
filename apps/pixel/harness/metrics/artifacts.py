"""Generator/sharpening artefacts: edge halos, banding on a smooth ramp, repetition in texture."""
from __future__ import annotations

import numpy as np

from ..colorspace import luminance
from ..scene import Scene, crop
from .resolution import EDGE_MARGIN, TEXTURE_MARGIN, edge_profile

GRADIENT_MARGIN = 6  # rows/cols dropped at the region edge; a candidate's resampling or blur bleeds in there
MIN_LAG_PX = 8


def edge_overshoot(frame: np.ndarray, scene: Scene) -> float:
    """Largest ESF excursion beyond the dark/bright plateaus, as a fraction of the step."""
    esf, dark, bright = edge_profile(luminance(crop(frame, scene.regions["slanted_edge"], EDGE_MARGIN)))
    step = max(bright - dark, 1e-9)
    return float(max(0.0, (esf.max() - bright) / step, (dark - esf.min()) / step))


BANDING_BLACK = 0.005  # adapting luminance = L + display black and flare (sRGB viewing: 0.2 of 80 cd/m2, plus glare)


def step_contrast(levels: np.ndarray, black: float = BANDING_BLACK) -> float:
    """Largest luminance step in a sequence that should vary smoothly, as Weber contrast.
    The second difference of a ramp (or any smooth tone curve of it) is ~0; a staircase gives
    +step where a band edge begins and -step where it ends. Only the positive side counts: a band
    edge makes the rise locally steeper than the reference, blur makes it shallower. Returns
    max second difference / (L + black)."""
    v = np.asarray(levels, dtype=np.float64)
    if v.size < 3:
        return 0.0
    up = 1.0 if v[-1] >= v[0] else -1.0           # direction in which the sequence rises
    d2 = np.maximum((v[2:] - 2 * v[1:-1] + v[:-2]) * up, 0.0)   # band edges steepen the rise; blur softens it
    return float(np.max(d2 / (np.maximum(v[1:-1], 0.0) + black)))


def banding_score(frame: np.ndarray, scene: Scene) -> float:
    """Largest band step on the smooth gradient as Weber contrast: the gradient's row means form a
    sequence whose second difference is ~0 for a ramp under any smooth tone curve and equals the
    step height at a band edge; that height over the adapting luminance (L + display black) is the
    contrast of a visible band. The sourced gates (Rep. ITU-R BT.2246 1/50-1/100, CAMBI 0.019) are
    Weber step contrasts, so they apply directly. Noise inflates the reading (recorded by the
    specificity probe in data/derived/gates_fitted.json)."""
    g = luminance(crop(frame, scene.regions["gradient"], GRADIENT_MARGIN))
    return step_contrast(g.mean(axis=1))



def periodicity(frame: np.ndarray, scene: Scene, min_lag: int = MIN_LAG_PX) -> float:
    """Max normalised autocorrelation of the aperiodic texture at lags >= min_lag px.
    Tiled or repeated detail scores near 1; the truth scores near 0."""
    t = luminance(crop(frame, scene.regions["texture"], TEXTURE_MARGIN))
    t = t - t.mean()
    h, w = t.shape
    F = np.fft.fft2(t)
    ac = np.real(np.fft.ifft2(np.abs(F) ** 2))
    ac = np.fft.fftshift(ac / max(ac[0, 0], 1e-18))
    Y, X = np.mgrid[0:h, 0:w]
    R = np.hypot(Y - h // 2, X - w // 2)
    sel = (R >= min_lag) & (R <= min(h, w) / 2)
    return float(ac[sel].max())
