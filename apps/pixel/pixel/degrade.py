"""Parametric degradations: the free, scalable labeller.

A ``Recipe`` is a list of ``(op, params)``. ``recipe_from_seed`` makes one
deterministically, ``apply`` renders it, and ``direction`` turns it into a fixed
length float vector (one slot per family) that a model can regress or be
conditioned on. Nothing here needs to be stored: a training pair is a pure
function of ``(source bytes, seed)``, which is what keeps a 4 TB disk enough.

Ops act on display-domain sRGB floats in [0, 1]. Physical ops (shot noise,
clipping) convert to linear internally.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .colorsci import display_to_lab, linear_to_lab, srgb_decode, srgb_encode
from .ops import decode_image, encode_image, fft_convolve2d, gaussian_blur, motion_kernel, resize

# Family order is the direction-vector layout. Append only; never reorder.
FAMILIES = (
    "blur",
    "motion",
    "downsample",
    "noise",
    "jpeg",
    "clip",
    "crush",
    "posterize",
    "sharpen",
    "tile",
    "color",
    "saturation",
)
DIRECTION_DIM = len(FAMILIES)

# Canonical application order (optics -> tone -> noise -> colour -> processing -> codec).
_APPLY_ORDER = ("blur", "motion", "downsample", "clip", "crush", "noise", "color", "saturation", "sharpen", "posterize", "tile", "jpeg")

Recipe = list


@dataclass(frozen=True)
class Ranges:
    blur_sigma: tuple[float, float] = (0.4, 4.0)
    motion_len: tuple[float, float] = (1.0, 12.0)
    downsample: tuple[float, ...] = (1.5, 2.0, 3.0, 4.0)
    noise_photons: tuple[float, float] = (50.0, 5000.0)
    jpeg_q: tuple[int, int] = (15, 90)
    clip_ev: tuple[float, float] = (0.3, 2.5)
    crush_black: tuple[float, float] = (0.02, 0.15)
    posterize_bits: tuple[int, ...] = (3, 4, 5, 6)
    sharpen_amount: tuple[float, float] = (0.5, 3.0)
    sharpen_radius: tuple[float, float] = (1.0, 3.0)
    tile_period: tuple[int, ...] = (16, 24, 32, 48)
    color_shift: tuple[float, float] = (-8.0, 8.0)
    saturation: tuple[float, float] = (0.5, 1.6)


RANGES = Ranges()


def seed_for(source_id: str, index: int) -> int:
    """Stable 63-bit seed from a source identifier and a per-source index."""
    h = hashlib.blake2b(f"{source_id}:{index}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") & 0x7FFFFFFFFFFFFFFF


def _sample_params(op: str, rng: np.random.Generator, r: Ranges) -> dict:
    if op == "blur":
        return {"sigma": float(np.exp(rng.uniform(*np.log(r.blur_sigma))))}
    if op == "motion":
        return {"length": float(rng.uniform(*r.motion_len)), "angle": float(rng.uniform(0, np.pi))}
    if op == "downsample":
        return {"factor": float(rng.choice(r.downsample))}
    if op == "noise":
        return {"photons": float(np.exp(rng.uniform(*np.log(r.noise_photons))))}
    if op == "jpeg":
        return {"quality": int(rng.integers(r.jpeg_q[0], r.jpeg_q[1] + 1))}
    if op == "clip":
        return {"ev": float(rng.uniform(*r.clip_ev))}
    if op == "crush":
        return {"black": float(rng.uniform(*r.crush_black))}
    if op == "posterize":
        return {"bits": int(rng.choice(r.posterize_bits))}
    if op == "sharpen":
        return {"amount": float(rng.uniform(*r.sharpen_amount)), "radius": float(rng.uniform(*r.sharpen_radius))}
    if op == "tile":
        return {"period": int(rng.choice(r.tile_period))}
    if op == "color":
        return {"da": float(rng.uniform(*r.color_shift)), "db": float(rng.uniform(*r.color_shift))}
    if op == "saturation":
        return {"scale": float(np.exp(rng.uniform(*np.log(r.saturation))))}
    raise KeyError(op)


def sample_recipe(rng: np.random.Generator, max_ops: int = 3, ranges: Ranges = RANGES, families=FAMILIES) -> Recipe:
    k = int(rng.integers(1, max_ops + 1))
    ops = list(rng.choice(families, size=k, replace=False))
    ops.sort(key=_APPLY_ORDER.index)
    return [(op, _sample_params(op, rng, ranges)) for op in ops]


def recipe_from_seed(seed: int, max_ops: int = 3, ranges: Ranges = RANGES) -> Recipe:
    return sample_recipe(np.random.default_rng(seed), max_ops=max_ops, ranges=ranges)


def direction(recipe: Recipe, ranges: Ranges = RANGES) -> np.ndarray:
    """Signed magnitude per family, roughly in [-1, 1]; 0 means the family is absent."""
    v = np.zeros(DIRECTION_DIM, dtype=np.float32)
    r = ranges
    for op, p in recipe:
        i = FAMILIES.index(op)
        if op == "blur":
            v[i] = np.log(p["sigma"] / r.blur_sigma[0]) / np.log(r.blur_sigma[1] / r.blur_sigma[0])
        elif op == "motion":
            v[i] = p["length"] / r.motion_len[1]
        elif op == "downsample":
            v[i] = np.log2(p["factor"]) / np.log2(max(r.downsample))
        elif op == "noise":
            v[i] = 1.0 - np.log(p["photons"] / r.noise_photons[0]) / np.log(r.noise_photons[1] / r.noise_photons[0])
        elif op == "jpeg":
            v[i] = (100.0 - p["quality"]) / (100.0 - r.jpeg_q[0])
        elif op == "clip":
            v[i] = p["ev"] / r.clip_ev[1]
        elif op == "crush":
            v[i] = p["black"] / r.crush_black[1]
        elif op == "posterize":
            v[i] = (8.0 - p["bits"]) / (8.0 - min(r.posterize_bits))
        elif op == "sharpen":
            v[i] = p["amount"] / r.sharpen_amount[1]
        elif op == "tile":
            v[i] = 1.0
        elif op == "color":
            v[i] = np.hypot(p["da"], p["db"]) / (r.color_shift[1] * np.sqrt(2.0))
        elif op == "saturation":
            v[i] = np.log2(p["scale"])
    return v


# -- operators --------------------------------------------------------------
def op_blur(img, p, rng):
    return gaussian_blur(img, p["sigma"])


def op_motion(img, p, rng):
    return fft_convolve2d(img, motion_kernel(p["length"], p["angle"]))


def op_downsample(img, p, rng):
    H, W = img.shape[:2]
    f = p["factor"]
    small = resize(img, max(1, int(round(W / f))), max(1, int(round(H / f))), "box")
    return np.clip(resize(small, W, H, "bilinear"), 0.0, 1.0)


def op_noise(img, p, rng):
    lin = srgb_decode(img)
    photons = p["photons"]
    noisy = rng.poisson(lin * photons).astype(np.float32) / photons
    return srgb_encode(noisy)


def op_jpeg(img, p, rng):
    return decode_image(encode_image(img, "jpeg", p["quality"]))


def op_clip(img, p, rng):
    return srgb_encode(np.clip(srgb_decode(img) * (2.0 ** p["ev"]), 0.0, 1.0))


def op_crush(img, p, rng):
    b = p["black"]
    return np.clip((img - b) / (1.0 - b), 0.0, 1.0).astype(np.float32)


def op_posterize(img, p, rng):
    levels = 2 ** p["bits"] - 1
    return (np.rint(np.clip(img, 0, 1) * levels) / levels).astype(np.float32)


def op_sharpen(img, p, rng):
    blurred = gaussian_blur(img, p["radius"])
    return np.clip(img + p["amount"] * (img - blurred), 0.0, 1.0).astype(np.float32)


def op_tile(img, p, rng):
    """Replace the central third with one patch repeated: the generator's 'repeated texture' failure."""
    H, W = img.shape[:2]
    T = int(p["period"])
    y0, x0, y1, x1 = H // 3, W // 3, 2 * H // 3, 2 * W // 3
    patch = img[y0 : y0 + T, x0 : x0 + T]
    if patch.shape[0] < T or patch.shape[1] < T:
        return img.copy()
    reps = (int(np.ceil((y1 - y0) / T)), int(np.ceil((x1 - x0) / T)), 1)
    tiled = np.tile(patch, reps)[: y1 - y0, : x1 - x0]
    out = img.copy()
    out[y0:y1, x0:x1] = tiled
    return out


def op_color(img, p, rng):
    lab = display_to_lab(img)
    lab[..., 1] += p["da"]
    lab[..., 2] += p["db"]
    return _lab_to_display(lab)


def op_saturation(img, p, rng):
    lab = display_to_lab(img)
    lab[..., 1:] *= p["scale"]
    return _lab_to_display(lab)


def _lab_to_display(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0

    def finv(f):
        f3 = f**3
        return np.where(f3 > eps, f3, (116.0 * f - 16.0) / kappa)

    from .colorsci import D65, M_RGB2XYZ

    xyz = np.stack([finv(fx), finv(fy), finv(fz)], axis=-1) * D65
    lin = xyz @ np.linalg.inv(M_RGB2XYZ).T
    return srgb_encode(np.clip(lin, 0.0, 1.0).astype(np.float32))


OPS = {
    "blur": op_blur,
    "motion": op_motion,
    "downsample": op_downsample,
    "noise": op_noise,
    "jpeg": op_jpeg,
    "clip": op_clip,
    "crush": op_crush,
    "posterize": op_posterize,
    "sharpen": op_sharpen,
    "tile": op_tile,
    "color": op_color,
    "saturation": op_saturation,
}


def apply(img: np.ndarray, recipe: Recipe, seed: int = 0) -> np.ndarray:
    """Render a recipe. Deterministic for a given (img, recipe, seed)."""
    rng = np.random.default_rng(seed)
    out = np.asarray(img, dtype=np.float32)
    for op, params in sorted(recipe, key=lambda r: _APPLY_ORDER.index(r[0])):
        out = OPS[op](out, params, rng)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def recipe_to_json(recipe: Recipe) -> list:
    return [[op, {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in p.items()}] for op, p in recipe]


def recipe_from_json(data: list) -> Recipe:
    return [(op, dict(p)) for op, p in data]


def label_pair(img: np.ndarray, source_id: str, index: int, max_ops: int = 3) -> tuple[np.ndarray, Recipe, np.ndarray, int]:
    """(degraded, recipe, direction, seed) for one source and index. The whole labeller is this line."""
    seed = seed_for(source_id, index)
    recipe = recipe_from_seed(seed, max_ops=max_ops)
    return apply(img, recipe, seed), recipe, direction(recipe), seed
