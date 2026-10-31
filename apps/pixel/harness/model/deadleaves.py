"""Procedural clean tiles: colour dead leaves, the occlusion model of natural images (Matheron's random sets; Lee,
Mumford and Huang fit its scale-invariant form to natural-image statistics), opaque discs of power-law radii in random
colours stacked in a random order. ISO 19567-2 measures a camera's texture reproduction on this pattern, and Achddou,
Gousseau and Ladjal train restoration networks on it. In the training stream it is a clean source with no file, no
licence and no ISP of its own: every edge is sharp at every contrast and every colour pairing, and every interior is
flat, so a smoothed edge and an invented texture both cost. The natural targets rarely hold a saturated edge with
little luminance step, and the twelfth run spread the demo chart's red patch edge over ten pixels against the
camera's three, its colour following a luminance edge that is not there (DEGRADATION_PLAN.md section 7); a share
ISO_SHARE of these tiles give every leaf one luminance, so their edges are chroma alone.

Rendered as the chart is (harness.scene): exact discs on a grid SS times finer, the top disc per sample kept in a
z-buffer, then box-downsampled. Linear RGB in [0, 1], an LDR tile, so the sampler stretches its scene range as it
does a picture's."""
from __future__ import annotations

import numpy as np

from ..colorspace import luminance

DEAD_LEAVES = "dead-leaves"     # the source name a procedural item and a checkpoint's record carry
SS = 4                          # render grid per tile pixel, as the demo chart's still
ALPHA = 3.0                     # radius density r^-ALPHA: 3 is scale invariant (equal area per octave of radius)
RMIN_PX = (1.0, 4.0)            # smallest radius per tile, log-uniform: the finest texture the tile holds
RMAX_FRAC = 0.5                 # largest radius, a fraction of the tile side
COVER = 4.5                     # summed disc area in tile areas; centres spread 10 % past each side, so about 4 % of
                                # the tile is left to the background leaf
VALUE_STOPS = 8.0               # a leaf's brightest channel, log-uniform over this many stops below 1
CONC = (0.2, 200.0)             # Gamma concentration of a leaf's channel mix, log-uniform: low is one channel (the
                                # gamut boundary), high is near neutral
ISO_SHARE = 0.25                # tiles whose leaves share one luminance
ISO_LUM = (1 / 256, 0.07)       # that luminance, log-uniform; under 0.072 (pure blue at 1) every hue reaches it
SMALL_HALF = 24                 # discs whose stencil half-width (render px) is at most this are drawn vectorised


def leaf_radii(rng: np.random.Generator, n: int, rmin: float, rmax: float, alpha: float = ALPHA) -> np.ndarray:
    """n radii with density proportional to r^-alpha on [rmin, rmax] (inverse CDF; alpha != 1)."""
    a = 1.0 - alpha
    u = rng.random(n)
    return (rmin ** a - u * (rmin ** a - rmax ** a)) ** (1.0 / a)


def leaf_colours(rng: np.random.Generator, n: int, iso_lum: float | None = None) -> np.ndarray:
    """(n, 3) linear RGB: a channel mix drawn from a Gamma of random concentration, scaled so the brightest channel
    sits log-uniform over VALUE_STOPS below 1, or, with `iso_lum`, so the luminance is that value."""
    conc = np.exp(rng.uniform(np.log(CONC[0]), np.log(CONC[1]), n))
    w = rng.gamma(conc[:, None], 1.0, (n, 3))
    w = w / np.maximum(w.max(axis=1, keepdims=True), 1e-300)
    if iso_lum is None:
        return np.exp(rng.uniform(-VALUE_STOPS * np.log(2.0), 0.0, n))[:, None] * w
    rgb = w * (iso_lum / np.maximum(luminance(w), 1e-300))[:, None]
    return rgb / np.maximum(rgb.max(axis=1, keepdims=True), 1.0)


def render_discs(size: int, radius: np.ndarray, centre: np.ndarray, ss: int = SS) -> np.ndarray:
    """(size*ss, size*ss) int64 map of the top disc at each render sample, -1 where none: disc i is drawn above
    disc j < i. Radius and centre (y, x) are in tile pixels; a render sample is inside a disc when its centre is
    within the radius. Small discs go through one maximum-reduction per stencil size, large ones one by one."""
    S = size * ss
    z = np.full(S * S, -1, dtype=np.int64)
    rs = np.asarray(radius, dtype=np.float64) * ss
    cs = np.asarray(centre, dtype=np.float64) * ss - 0.5      # sample (j, k) has its centre at (j + 0.5, k + 0.5)
    half = np.ceil(rs).astype(np.int64) + 1
    small = half <= SMALL_HALF
    for h in np.unique(half[small]):
        k = np.nonzero(small & (half == h))[0]
        off = np.arange(-h, h + 1)
        yy = np.floor(cs[k, 0]).astype(np.int64)[:, None, None] + off[None, :, None]
        xx = np.floor(cs[k, 1]).astype(np.int64)[:, None, None] + off[None, None, :]
        d2 = (yy - cs[k, 0][:, None, None]) ** 2 + (xx - cs[k, 1][:, None, None]) ** 2
        inside = (d2 <= rs[k][:, None, None] ** 2) & (yy >= 0) & (yy < S) & (xx >= 0) & (xx < S)
        ids = np.broadcast_to(k[:, None, None], inside.shape)
        np.maximum.at(z, (yy * S + xx)[inside], ids[inside])
    zz = z.reshape(S, S)
    for i in np.nonzero(~small)[0]:
        y0, y1 = max(0, int(np.floor(cs[i, 0] - rs[i]))), min(S, int(np.ceil(cs[i, 0] + rs[i])) + 1)
        x0, x1 = max(0, int(np.floor(cs[i, 1] - rs[i]))), min(S, int(np.ceil(cs[i, 1] + rs[i])) + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.arange(y0, y1)[:, None], np.arange(x0, x1)[None, :]
        inside = (yy - cs[i, 0]) ** 2 + (xx - cs[i, 1]) ** 2 <= rs[i] ** 2
        sub = zz[y0:y1, x0:x1]
        np.maximum(sub, np.where(inside, i, -1), out=sub)
    return zz


def dead_leaves(rng: np.random.Generator, size: int, ss: int = SS) -> np.ndarray:
    """One (size, size, 3) float64 linear tile in [0, 1]: discs drawn until their summed area is COVER tile areas,
    the smallest radius log-uniform in RMIN_PX, the largest RMAX_FRAC of the side, centres uniform over the tile
    widened by a tenth on each side, a background leaf under them all."""
    rmin = float(np.exp(rng.uniform(np.log(RMIN_PX[0]), np.log(RMIN_PX[1]))))
    rmax = max(RMAX_FRAC * size, 2.0 * rmin)
    target, radii = COVER * size * size, []
    while True:
        r = leaf_radii(rng, 4096, rmin, rmax)
        radii.append(r)
        area = np.cumsum(np.pi * np.concatenate(radii) ** 2)
        if area[-1] >= target:
            break
    r = np.concatenate(radii)[: int(np.searchsorted(area, target)) + 1]
    centre = rng.uniform(-0.1 * size, 1.1 * size, (len(r), 2))
    iso = float(np.exp(rng.uniform(np.log(ISO_LUM[0]), np.log(ISO_LUM[1])))) if rng.random() < ISO_SHARE else None
    colours = leaf_colours(rng, len(r) + 1, iso)              # the last row is the background: z = -1 indexes it
    z = render_discs(size, r, centre, ss)
    img = colours[z].reshape(size, ss, size, ss, 3).mean(axis=(1, 3))
    return np.ascontiguousarray(np.clip(img, 0.0, 1.0))
