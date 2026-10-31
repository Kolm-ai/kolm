"""Procedural ground-truth world. Every pixel, mask, landmark and flow field is known analytically.

Layout (fractions of the frame): slanted edge, aperiodic texture, vertical gradient and colour patches
across the top; a 14-step wedge in the middle; a stage with four shaded, moving disks at the bottom.
Disks move an integer number of pixels per frame, so ground-truth flow warps are exact.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .colorspace import linear_rgb_to_lab, luminance

Region = tuple[int, int, int, int]  # (y0, y1, x0, x1), exclusive ends, target-resolution pixels

WEDGE_VALUES = (0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.125, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0)
COLOR_VALUES = (  # linear sRGB, in gamut, spread over hue and lightness
    (0.17, 0.08, 0.06), (0.58, 0.32, 0.26), (0.11, 0.21, 0.42), (0.09, 0.15, 0.05),
    (0.46, 0.05, 0.06), (0.06, 0.32, 0.09), (0.82, 0.60, 0.02), (0.30, 0.09, 0.40),
)
DISK_COLORS = ((0.95, 0.85, 0.30), (0.40, 0.90, 0.50), (0.50, 0.70, 0.95), (0.90, 0.90, 0.90))
DISK_VELOCITIES = ((2, 1), (-1, 2), (3, 0), (0, -2))  # px/frame at REF_HEIGHT, scaled with height
REF_HEIGHT = 216

BG = 0.10
STAGE_BG = 0.06
EDGE_DARK, EDGE_BRIGHT, EDGE_ANGLE_DEG = 0.05, 0.90, 5.0
GRADIENT_LO, GRADIENT_HI = 0.02, 0.30
AMBIENT = 0.4
_L = np.array([-0.45, -0.55, 0.70])
LIGHT_DIR = _L / np.linalg.norm(_L)
DISK_THRESHOLD = 0.5 * (STAGE_BG + AMBIENT * min(float(luminance(np.array(c))) for c in DISK_COLORS))


@dataclass(frozen=True)
class SceneSpec:
    width: int = 768
    height: int = 432
    supersample: int = 4
    seed: int = 7
    n_frames: int = 1
    fps: float = 24.0
    exposure_s: float = 0.0  # shutter the truth is rendered with; 0 = instantaneous

    def frame_time(self, k: int) -> float:
        return k / self.fps


def _bounce(x0: float, v: float, t: float, lo: float, hi: float) -> float:
    span = hi - lo
    if span <= 0 or v == 0:
        return x0
    u = (x0 - lo + v * t) % (2 * span)
    return lo + (span - abs(u - span))


@dataclass(frozen=True)
class Landmark:
    x0: float
    y0: float
    vx: float  # px / s
    vy: float
    xlo: float
    xhi: float
    ylo: float
    yhi: float
    radius: float
    color: tuple[float, float, float]

    def position(self, t: float) -> tuple[float, float]:
        return (_bounce(self.x0, self.vx, t, self.xlo, self.xhi),
                _bounce(self.y0, self.vy, t, self.ylo, self.yhi))


def box_downsample(a: np.ndarray, ss: int) -> np.ndarray:
    if ss == 1:
        return a
    H, W = a.shape[0] // ss, a.shape[1] // ss
    return a[:H * ss, :W * ss].reshape(H, ss, W, ss, *a.shape[2:]).mean(axis=(1, 3))


def crop(frame: np.ndarray, region: Region, margin: int = 0) -> np.ndarray:
    y0, y1, x0, x1 = region
    return frame[y0 + margin:y1 - margin, x0 + margin:x1 - margin]


class Scene:
    def __init__(self, spec: SceneSpec):
        self.spec = spec
        self.regions: dict[str, Region] = {
            "slanted_edge": self._box(0.02, 0.05, 0.22, 0.36),
            "texture": self._box(0.25, 0.05, 0.50, 0.36),
            "gradient": self._box(0.53, 0.05, 0.70, 0.36),
            "colors": self._box(0.72, 0.05, 0.98, 0.36),
            "wedge": self._box(0.02, 0.40, 0.98, 0.50),
            "stage": self._box(0.02, 0.54, 0.98, 0.98),
        }
        self.wedge: list[tuple[Region, float]] = list(
            zip(self._cells(self.regions["wedge"], 1, len(WEDGE_VALUES)), WEDGE_VALUES))
        self.colors: list[tuple[Region, np.ndarray, np.ndarray]] = [
            (r, np.array(c), linear_rgb_to_lab(np.array(c)))
            for r, c in zip(self._cells(self.regions["colors"], 2, 4), COLOR_VALUES)]
        self.landmarks = self._make_landmarks()
        H, W = spec.height, spec.width
        self._yc, self._xc = np.mgrid[0:H, 0:W] + 0.5
        self._static_ss = self._build_static_ss()

    # ---- layout -------------------------------------------------------------------------------
    def _box(self, fx0, fy0, fx1, fy1) -> Region:
        W, H = self.spec.width, self.spec.height
        return (int(round(fy0 * H)), int(round(fy1 * H)), int(round(fx0 * W)), int(round(fx1 * W)))

    @staticmethod
    def _cells(area: Region, rows: int, cols: int, fill: float = 0.8) -> list[Region]:
        y0, y1, x0, x1 = area
        ch, cw = (y1 - y0) / rows, (x1 - x0) / cols
        out = []
        for r in range(rows):
            for c in range(cols):
                cy, cx = y0 + (r + 0.5) * ch, x0 + (c + 0.5) * cw
                hh, hw = fill * ch / 2, fill * cw / 2
                out.append((int(round(cy - hh)), int(round(cy + hh)), int(round(cx - hw)), int(round(cx + hw))))
        return out

    def _make_landmarks(self) -> list[Landmark]:
        spec = self.spec
        sy0, sy1, sx0, sx1 = self.regions["stage"]
        r = 0.055 * spec.height
        n = len(DISK_COLORS)
        colw = (sx1 - sx0) / n
        scale = spec.height / REF_HEIGHT

        def q(u):  # integer px/frame, scaled with resolution, sign kept
            return 0 if u == 0 else int(math.copysign(max(1, round(abs(u) * scale)), u))

        out = []
        for i, (color, (ux, uy)) in enumerate(zip(DISK_COLORS, DISK_VELOCITIES)):
            # bounce box on pixel centres (k + 0.5) so frame-to-frame displacement stays integer
            xlo = math.floor(sx0 + i * colw + r + 2) + 0.5
            xhi = math.floor(sx0 + (i + 1) * colw - r - 2) + 0.5
            ylo = math.floor(sy0 + r + 2) + 0.5
            yhi = math.floor(sy1 - r - 2) + 0.5
            x0 = xlo + math.floor((xhi - xlo) / 2)
            y0 = ylo + math.floor((yhi - ylo) / 2)
            out.append(Landmark(x0, y0, q(ux) * spec.fps, q(uy) * spec.fps, xlo, xhi, ylo, yhi, r, color))
        return out

    # ---- static layers ------------------------------------------------------------------------
    def _ss_region(self, region: Region) -> Region:
        ss = self.spec.supersample
        return tuple(v * ss for v in region)  # type: ignore[return-value]

    def _fill(self, f: np.ndarray, region: Region, value) -> None:
        y0, y1, x0, x1 = self._ss_region(region)
        f[y0:y1, x0:x1] = np.asarray(value, dtype=np.float32)

    def _texture(self, h: int, w: int) -> np.ndarray:
        """Aperiodic, high-frequency-rich value noise. Deterministic by seed."""
        ss = self.spec.supersample
        rng = np.random.default_rng(self.spec.seed)
        acc = np.zeros((h, w), dtype=np.float64)
        for k, sigma_px in enumerate((0.5, 1.0, 2.0, 4.0, 8.0)):
            n = gaussian_filter(rng.standard_normal((h, w)), sigma_px * ss, mode="reflect")
            acc += n / (n.std() + 1e-12) * (0.55 ** k)
        acc /= acc.std()
        return np.clip(0.40 + 0.12 * acc, 0.02, 0.98).astype(np.float32)

    def _build_static_ss(self) -> np.ndarray:
        spec = self.spec
        ss = spec.supersample
        Hs, Ws = spec.height * ss, spec.width * ss
        f = np.full((Hs, Ws, 3), BG, dtype=np.float32)
        self._fill(f, self.regions["stage"], STAGE_BG)

        y0, y1, x0, x1 = self._ss_region(self.regions["slanted_edge"])
        ys, xs = np.mgrid[y0:y1, x0:x1]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        th = math.radians(EDGE_ANGLE_DEG)
        d = (xs + 0.5 - cx) * math.cos(th) + (ys + 0.5 - cy) * math.sin(th)
        f[y0:y1, x0:x1] = np.where(d > 0, EDGE_BRIGHT, EDGE_DARK).astype(np.float32)[..., None]

        for region, v in self.wedge:
            self._fill(f, region, v)
        for region, rgb, _ in self.colors:
            self._fill(f, region, rgb)

        y0, y1, x0, x1 = self._ss_region(self.regions["gradient"])
        frac = (np.arange(y0, y1) + 0.5 - y0) / (y1 - y0)
        f[y0:y1, x0:x1] = (GRADIENT_LO + (GRADIENT_HI - GRADIENT_LO) * frac).astype(np.float32)[:, None, None]

        y0, y1, x0, x1 = self._ss_region(self.regions["texture"])
        f[y0:y1, x0:x1] = self._texture(y1 - y0, x1 - x0)[..., None]
        return f

    # ---- rendering ----------------------------------------------------------------------------
    def _draw_disk(self, f: np.ndarray, lm: Landmark, t: float) -> None:
        ss = self.spec.supersample
        Hs, Ws = f.shape[:2]
        cx, cy = lm.position(t)
        r = lm.radius
        I0, I1 = max(0, math.floor((cy - r) * ss) - 1), min(Hs, math.ceil((cy + r) * ss) + 2)
        J0, J1 = max(0, math.floor((cx - r) * ss) - 1), min(Ws, math.ceil((cx + r) * ss) + 2)
        ys, xs = np.mgrid[I0:I1, J0:J1]
        dx = (xs + 0.5) / ss - cx
        dy = (ys + 0.5) / ss - cy
        rho2 = (dx * dx + dy * dy) / (r * r)
        mask = rho2 <= 1.0
        nz = np.sqrt(np.clip(1.0 - rho2, 0.0, 1.0))
        ndotl = (dx / r) * LIGHT_DIR[0] + (dy / r) * LIGHT_DIR[1] + nz * LIGHT_DIR[2]
        shade = (AMBIENT + (1 - AMBIENT) * np.clip(ndotl, 0.0, None)).astype(np.float32)
        col = np.asarray(lm.color, dtype=np.float32)[None, None, :] * shade[..., None]
        sub = f[I0:I1, J0:J1]
        sub[mask] = col[mask]

    def render_ss(self, t: float) -> np.ndarray:
        """Instantaneous supersampled render at time t (seconds)."""
        f = self._static_ss.copy()
        for lm in self.landmarks:
            self._draw_disk(f, lm, t)
        return f

    def peak_radiance(self) -> float:
        """Brightest radiance anywhere in the scene (the wedge's top step, 2.0, unless a landmark is brighter): the
        best-case camera exposes for it (camera.py, relative mode)."""
        if getattr(self, "_peak", None) is None:
            self._peak = float(max(self.render_ss(0.0).max(), max(WEDGE_VALUES)))
        return self._peak

    def _n_sub(self, exposure_s: float) -> int:
        vmax = max((math.hypot(lm.vx, lm.vy) for lm in self.landmarks), default=0.0)
        return int(np.clip(math.ceil(vmax * exposure_s * self.spec.supersample * 1.5), 1, 48))

    def render_ss_integrated(self, t: float, exposure_s: float) -> np.ndarray:
        """Supersampled render averaged over a shutter window centred on t."""
        if exposure_s <= 0:
            return self.render_ss(t)
        n = self._n_sub(exposure_s)
        acc = np.zeros_like(self._static_ss)
        for j in range(n):
            acc += self.render_ss(t + ((j + 0.5) / n - 0.5) * exposure_s)
        return acc / n

    def render(self, t: float, exposure_s: float | None = None) -> np.ndarray:
        """Truth frame at target resolution: ideal pixel-aperture sampling, no optics, no noise."""
        e = self.spec.exposure_s if exposure_s is None else exposure_s
        return box_downsample(self.render_ss_integrated(t, e), self.spec.supersample).astype(np.float64)

    def truth_frames(self) -> np.ndarray:
        return np.stack([self.render(self.spec.frame_time(k)) for k in range(self.spec.n_frames)])

    # ---- analytic masks, landmarks, flow ------------------------------------------------------
    def landmark_positions(self, t: float) -> np.ndarray:
        return np.array([lm.position(t) for lm in self.landmarks])

    def disk_masks(self, t: float, grow: float = 0.0) -> np.ndarray:
        out = np.empty((len(self.landmarks),) + self._xc.shape, dtype=bool)
        for i, lm in enumerate(self.landmarks):
            cx, cy = lm.position(t)
            rr = max(lm.radius + grow, 0.0)
            out[i] = (self._xc - cx) ** 2 + (self._yc - cy) ** 2 <= rr * rr
        return out

    def moving_mask(self, t: float, grow: float = 0.0) -> np.ndarray:
        return self.disk_masks(t, grow).any(axis=0)

    def flow(self, k: int) -> np.ndarray:
        """Displacement (dx, dy) in px from frame k to k+1 for pixels inside a disk at frame k; 0 elsewhere."""
        t0, t1 = self.spec.frame_time(k), self.spec.frame_time(k + 1)
        flow = np.zeros(self._xc.shape + (2,), dtype=np.float64)
        for lm, m in zip(self.landmarks, self.disk_masks(t0)):
            p0, p1 = lm.position(t0), lm.position(t1)
            flow[m] = (p1[0] - p0[0], p1[1] - p0[1])
        return flow

    def flow_valid(self, k: int) -> np.ndarray:
        """Pixels whose flow is known: inside a disk at k, or background at both k and k+1."""
        t0, t1 = self.spec.frame_time(k), self.spec.frame_time(k + 1)
        return self.moving_mask(t0) | ~self.moving_mask(t1)

    def static_mask_all(self) -> np.ndarray:
        """Pixels never covered by a disk in any frame."""
        covered = np.zeros(self._xc.shape, dtype=bool)
        for k in range(self.spec.n_frames):
            covered |= self.moving_mask(self.spec.frame_time(k), grow=1.0)
        return ~covered
