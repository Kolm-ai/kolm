"""Procedural HDR ground truth with analytic regions, landmarks and motion.

Every element is an analytic function of continuous content coordinates, so a
frame at time ``t`` is the same world translated by exactly ``t * pan`` pixels.
That gives exact sub-pixel flow for the temporal metrics and an unlimited supply
of scenes (one per seed) at any resolution: nothing is stored, everything is a
function of ``(seed, width, height, t)``.

Regions (content-normalised boxes, each metric reads the one built for it):

* ``gradient``   smooth sky ramp            -> banding
* ``highlight``  log ramp 0.5 .. 8 radiance -> highlight clipping / loss
* ``patches``    six flat reference colours -> colour error
* ``texture``    full-band non-periodic texture -> MTF50, tiling
* ``edge``       slanted step edge          -> halos
* ``shadow``     radiance ~0.002 with faint texture -> shadow noise / crushed detail
* ``landmarks``  3x3 Gaussian dots          -> geometry
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Linear sRGB reference colours (roughly ColorChecker rows 1 and 2).
PATCH_COLORS = np.array(
    [
        [0.172, 0.090, 0.059],
        [0.570, 0.316, 0.228],
        [0.100, 0.182, 0.350],
        [0.088, 0.146, 0.064],
        [0.244, 0.237, 0.497],
        [0.251, 0.510, 0.410],
    ],
    dtype=np.float32,
)

MASK_INSET_PX = 4


@dataclass(frozen=True)
class SceneSpec:
    seed: int = 7
    width: int = 512
    height: int = 512
    frames: int = 1
    fps: float = 24.0
    pan_px_per_frame: tuple[float, float] = (1.37, -0.61)

    @classmethod
    def from_dict(cls, d: dict) -> "SceneSpec":
        pan = d.get("pan_px_per_frame", (1.37, -0.61))
        return cls(
            seed=int(d.get("seed", 7)),
            width=int(d.get("width", 512)),
            height=int(d.get("height", 512)),
            frames=int(d.get("frames", 1)),
            fps=float(d.get("fps", 24.0)),
            pan_px_per_frame=(float(pan[0]), float(pan[1])),
        )


@dataclass
class _Bank:
    fx: np.ndarray
    fy: np.ndarray
    amp: np.ndarray
    phase: np.ndarray

    def eval(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        out = np.zeros(np.broadcast(u, v).shape, dtype=np.float32)
        for fx, fy, a, p in zip(self.fx, self.fy, self.amp, self.phase):
            out += a * np.sin(2.0 * np.pi * (fx * u + fy * v) + p)
        return out / float(np.abs(self.amp).sum()) * 2.5


EDGE_SIGMA_PX = 0.7


def _ncdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF via Abramowitz & Stegun 7.1.26 (|err| < 1.5e-7). Keeps edges band-limited."""
    z = np.asarray(z, dtype=np.float32) / np.sqrt(2.0)
    sign = np.sign(z)
    a = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-a * a)
    return (0.5 * (1.0 + sign * erf)).astype(np.float32)


def _hash01(ix: np.ndarray, iy: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic lattice hash -> uniform [0, 1]. Integer mixing only, so it is exact everywhere."""
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + int(salt) * 1013904223) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return ((h & 0xFFFFFF) / float(0xFFFFFF)).astype(np.float32)


def value_noise(X: np.ndarray, Y: np.ndarray, spacing: float, salt: int) -> np.ndarray:
    """Smoothstep-interpolated lattice noise in [-1, 1]: non-periodic and a pure function of coordinates."""
    gx, gy = X / spacing, Y / spacing
    ix, iy = np.floor(gx), np.floor(gy)
    fx, fy = gx - ix, gy - iy
    sx, sy = fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy)
    ix, iy = ix.astype(np.int64), iy.astype(np.int64)
    n00, n10 = _hash01(ix, iy, salt), _hash01(ix + 1, iy, salt)
    n01, n11 = _hash01(ix, iy + 1, salt), _hash01(ix + 1, iy + 1, salt)
    top = n00 + (n10 - n00) * sx
    bot = n01 + (n11 - n01) * sx
    return (2.0 * (top + (bot - top) * sy) - 1.0).astype(np.float32)


def fractal_noise(X: np.ndarray, Y: np.ndarray, salt: int, spacings=(2.3, 4.7, 9.1, 17.3, 31.7), slope: float = 0.5) -> np.ndarray:
    """Octaves of value noise, each on its own rotated lattice so no lattice line aligns with the
    pixel grid or the 8x8 codec grid. Rotation is linear, so translation stays exact."""
    amps = np.array([float(s) ** slope for s in spacings], dtype=np.float32)
    out = np.zeros(np.broadcast(X, Y).shape, dtype=np.float32)
    for k, (sp, a) in enumerate(zip(spacings, amps)):
        ang = np.radians(17.0 + 23.0 * k + (salt % 7))
        c, sn = np.cos(ang), np.sin(ang)
        out += a * value_noise(c * X - sn * Y, sn * X + c * Y, float(sp), salt * 1000 + k)
    return out / float(amps.sum()) * 2.0


def _make_bank(rng: np.random.Generator, n: int, fmin: float, fmax: float, slope: float) -> _Bank:
    f = np.exp(rng.uniform(np.log(fmin), np.log(fmax), size=n))
    theta = rng.uniform(0, np.pi, size=n)
    return _Bank(
        fx=(f * np.cos(theta)).astype(np.float32),
        fy=(f * np.sin(theta)).astype(np.float32),
        amp=(f**-slope).astype(np.float32),
        phase=rng.uniform(0, 2 * np.pi, size=n).astype(np.float32),
    )


@dataclass
class Regions:
    masks: dict = field(default_factory=dict)
    patch_masks: list = field(default_factory=list)
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    edge_direction: tuple[float, float] = (1.0, 0.0)


class Scene:
    """One procedural world. ``render(t)`` is HDR linear ``(H, W, 3)``."""

    BOXES = {
        "gradient": (0.05, 0.05, 0.55, 0.28),
        "highlight": (0.62, 0.05, 0.95, 0.28),
        "texture": (0.62, 0.34, 0.95, 0.64),
        "edge": (0.34, 0.50, 0.56, 0.70),
        "shadow": (0.05, 0.72, 0.28, 0.95),
        "landmarks": (0.34, 0.74, 0.58, 0.96),
    }
    PATCH_ROW = (0.34, 0.46)
    PATCH_X0, PATCH_STEP, PATCH_W = 0.05, 0.09, 0.075

    def __init__(self, spec: SceneSpec):
        self.spec = spec
        rng = np.random.default_rng(spec.seed)
        self.salt = int(spec.seed) & 0xFFFF
        self.fine_bank = _make_bank(rng, 24, 0.06, 0.45, 0.5)
        self.edge_angle = np.radians(5.0 + rng.uniform(-2.0, 2.0))
        self.landmark_jitter = rng.uniform(-0.35, 0.35, size=(9, 2)).astype(np.float32)

    # -- coordinates -------------------------------------------------------
    def _coords(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        W, H = self.spec.width, self.spec.height
        vx, vy = self.spec.pan_px_per_frame
        x = (np.arange(W, dtype=np.float32) - t * vx)[None, :]
        y = (np.arange(H, dtype=np.float32) - t * vy)[:, None]
        return x, y

    def _box_mask(self, x: np.ndarray, y: np.ndarray, box: tuple, inset_px: float = 0.0) -> np.ndarray:
        W, H = self.spec.width, self.spec.height
        x0, y0, x1, y1 = box[0] * W + inset_px, box[1] * H + inset_px, box[2] * W - inset_px, box[3] * H - inset_px
        return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)

    def _box_cov(self, x: np.ndarray, y: np.ndarray, box: tuple) -> np.ndarray:
        """Coverage of a box with Gaussian-integrated edges (sigma 0.7 px): sub-pixel exact and band-limited,
        so a translated frame is the translated image and nothing aliases."""
        W, H = self.spec.width, self.spec.height
        x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
        cx = _ncdf((x - x0) / EDGE_SIGMA_PX) * _ncdf((x1 - x) / EDGE_SIGMA_PX)
        cy = _ncdf((y - y0) / EDGE_SIGMA_PX) * _ncdf((y1 - y) / EDGE_SIGMA_PX)
        return (cx * cy).astype(np.float32)

    @staticmethod
    def _blend(rgb: np.ndarray, cov: np.ndarray, layer: np.ndarray) -> np.ndarray:
        c = cov[..., None]
        return (c * layer + (1.0 - c) * rgb).astype(np.float32)

    def _patch_boxes(self):
        for i in range(len(PATCH_COLORS)):
            x0 = self.PATCH_X0 + i * self.PATCH_STEP
            yield (x0, self.PATCH_ROW[0], x0 + self.PATCH_W, self.PATCH_ROW[1])

    def landmark_positions(self, t: float = 0.0) -> np.ndarray:
        W, H = self.spec.width, self.spec.height
        vx, vy = self.spec.pan_px_per_frame
        x0, y0, x1, y1 = self.BOXES["landmarks"]
        xs = np.linspace(x0 * W + 10, x1 * W - 10, 3)
        ys = np.linspace(y0 * H + 10, y1 * H - 10, 3)
        pts = np.array([(x, y) for y in ys for x in xs], dtype=np.float32) + self.landmark_jitter
        return pts + np.array([t * vx, t * vy], dtype=np.float32)

    def flow(self, t: float) -> tuple[float, float]:
        return self.spec.pan_px_per_frame

    # -- rendering ---------------------------------------------------------
    def render(self, t: float = 0.0) -> np.ndarray:
        W, H = self.spec.width, self.spec.height
        x, y = self._coords(t)
        X, Y = np.broadcast_arrays(x, y)
        fine = 0.6 * fractal_noise(X, Y, self.salt + 1, spacings=(4.1, 8.3, 16.7, 33.1), slope=0.5) + 0.4 * self.fine_bank.eval(X, Y)
        medium = fractal_noise(X, Y, self.salt + 3, spacings=(5.3, 10.7, 21.1, 41.3), slope=0.7)
        coarse = fractal_noise(X, Y, self.salt + 2, spacings=(23.7, 47.9, 95.3, 191.1), slope=1.0)

        lum = 0.18 * (1.0 + 0.15 * coarse)
        rgb = np.stack([lum, lum * 0.98, lum * 0.95], axis=-1).astype(np.float32)

        # sky gradient: pure ramp, no texture (banding probe)
        cov = self._box_cov(X, Y, self.BOXES["gradient"])
        y0, y1 = self.BOXES["gradient"][1] * H, self.BOXES["gradient"][3] * H
        ramp = 0.55 - 0.35 * np.clip((Y - y0) / max(y1 - y0, 1.0), 0, 1)
        sky = np.stack([0.75 * ramp, 0.85 * ramp, 1.0 * ramp], axis=-1)
        rgb = self._blend(rgb, cov, sky)

        # highlight wedge: log ramp from -1 EV to +3 EV with 25% texture
        cov = self._box_cov(X, Y, self.BOXES["highlight"])
        x0, x1 = self.BOXES["highlight"][0] * W, self.BOXES["highlight"][2] * W
        ev = -1.0 + 4.0 * np.clip((X - x0) / max(x1 - x0, 1.0), 0, 1)
        hl = np.power(2.0, ev) * (1.0 + 0.25 * medium)
        wedge = np.stack([hl, 0.95 * hl, 0.85 * hl], axis=-1)
        rgb = self._blend(rgb, cov, wedge)

        # colour patches
        for box, col in zip(self._patch_boxes(), PATCH_COLORS):
            rgb = self._blend(rgb, self._box_cov(X, Y, box), np.broadcast_to(col[None, None, :], rgb.shape))

        # full-band texture
        cov = self._box_cov(X, Y, self.BOXES["texture"])
        tex = np.maximum(0.3 * (1.0 + 0.6 * fine), 0.01)
        rgb = self._blend(rgb, cov, np.stack([tex, 0.97 * tex, 0.9 * tex], axis=-1))

        # slanted edge, Gaussian-integrated (band-limited)
        cov = self._box_cov(X, Y, self.BOXES["edge"])
        bx0, by0, bx1, by1 = self.BOXES["edge"]
        cx, cy = 0.5 * (bx0 + bx1) * W, 0.5 * (by0 + by1) * H
        d = (X - cx) * np.cos(self.edge_angle) - (Y - cy) * np.sin(self.edge_angle)
        step = 0.1 + 0.5 * _ncdf(d / EDGE_SIGMA_PX)
        rgb = self._blend(rgb, cov, np.stack([step, step, step], axis=-1))

        # deep shadow with faint texture
        cov = self._box_cov(X, Y, self.BOXES["shadow"])
        sh = 0.002 * (1.0 + 0.4 * (0.6 * medium + 0.4 * coarse))
        rgb = self._blend(rgb, cov, np.stack([0.8 * sh, 0.85 * sh, sh], axis=-1))

        # landmarks on a dark ground
        cov = self._box_cov(X, Y, self.BOXES["landmarks"])
        ground = np.full_like(lum, 0.08)
        for px, py in self.landmark_positions(0.0):
            r2 = (X - px) ** 2 + (Y - py) ** 2
            ground = ground + 0.8 * np.exp(-r2 / (2.0 * 1.3**2))
        rgb = self._blend(rgb, cov, np.stack([ground, ground, ground], axis=-1))
        return rgb.astype(np.float32)

    def regions(self, t: float = 0.0) -> Regions:
        x, y = self._coords(t)
        X, Y = np.broadcast_arrays(x, y)
        masks = {name: self._box_mask(X, Y, box, MASK_INSET_PX) for name, box in self.BOXES.items()}
        patch_masks = [self._box_mask(X, Y, box, MASK_INSET_PX) for box in self._patch_boxes()]
        return Regions(
            masks=masks,
            patch_masks=patch_masks,
            landmarks=self.landmark_positions(t),
            edge_direction=(float(np.cos(self.edge_angle)), float(-np.sin(self.edge_angle))),
        )

    def frames(self):
        for i in range(self.spec.frames):
            yield i, self.render(float(i))
