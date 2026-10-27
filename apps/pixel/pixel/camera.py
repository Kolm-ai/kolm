"""The floor: the best single-exposure camera physics allows.

``BestCaseCamera`` defaults to a full-frame 61 MP class sensor behind a
diffraction-limited f/2.8 lens with no aberrations at all. Every loss it
suffers is a physical one:

* diffraction          Airy PSF per channel wavelength
* motion               exposure-integrated blur along the stated motion
* photon statistics    Poisson shot noise at the stated full well
* saturation           hard clip at full well (one exposure, one well)
* read noise           Gaussian, in electrons
* sampling             RGGB Bayer mosaic + bilinear demosaic, ADC quantisation

Anything a candidate beats here it beats for a physical reason, not because
the comparison camera was a bad one.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .colorsci import display_encode
from .ops import fft_convolve2d, shift_subpixel
from .physics import RGB_WAVELENGTHS_M, airy_psf

_G_KERNEL = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float32) / 4.0
_RB_KERNEL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 4.0


@dataclass(frozen=True)
class BestCaseCamera:
    pixel_pitch_um: float = 3.76
    f_number: float = 2.8
    full_well_e: float = 51000.0
    read_noise_e: float = 1.5
    bit_depth: int = 14
    bayer: bool = True
    diffraction: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "BestCaseCamera":
        d = dict(d or {})
        allowed = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**allowed)

    def summary(self) -> dict:
        return asdict(self)

    # -- stages --------------------------------------------------------------
    def optical(self, linear: np.ndarray) -> np.ndarray:
        if not self.diffraction:
            return linear.astype(np.float32)
        pitch = self.pixel_pitch_um * 1e-6
        out = np.empty_like(linear, dtype=np.float32)
        for c, lam in enumerate(RGB_WAVELENGTHS_M):
            out[..., c] = fft_convolve2d(linear[..., c], airy_psf(self.f_number, lam, pitch))
        return out

    @staticmethod
    def motion(linear: np.ndarray, motion_px: tuple[float, float]) -> np.ndarray:
        dx, dy = float(motion_px[0]), float(motion_px[1])
        length = float(np.hypot(dx, dy))
        if length < 1e-3:
            return linear.astype(np.float32)
        n = max(2, int(np.ceil(length * 2)) + 1)
        acc = np.zeros_like(linear, dtype=np.float32)
        for s in np.linspace(-0.5, 0.5, n):
            acc += shift_subpixel(linear, s * dx, s * dy)
        return acc / n

    def sensor(self, linear: np.ndarray, exposure_ev: float, rng: np.random.Generator) -> np.ndarray:
        """Photons -> electrons -> ADU -> normalised linear (per channel, pre-mosaic)."""
        gain = 2.0**exposure_ev
        mean_e = np.clip(linear * gain * self.full_well_e, 0.0, None)
        electrons = rng.poisson(mean_e).astype(np.float32)
        electrons = np.minimum(electrons, self.full_well_e)
        electrons += rng.normal(0.0, self.read_noise_e, size=electrons.shape).astype(np.float32)
        levels = 2**self.bit_depth
        adu = np.clip(np.rint(electrons / self.full_well_e * (levels - 1)), 0, levels - 1)
        return (adu / (levels - 1)).astype(np.float32)

    @staticmethod
    def mosaic_demosaic(linear: np.ndarray) -> np.ndarray:
        """Sample an RGGB mosaic from the per-channel image and rebuild it bilinearly."""
        H, W, _ = linear.shape
        yy, xx = np.mgrid[0:H, 0:W]
        r_mask = ((yy % 2) == 0) & ((xx % 2) == 0)
        b_mask = ((yy % 2) == 1) & ((xx % 2) == 1)
        g_mask = ~(r_mask | b_mask)
        out = np.empty_like(linear)
        for c, mask, k in ((0, r_mask, _RB_KERNEL), (1, g_mask, _G_KERNEL), (2, b_mask, _RB_KERNEL)):
            sparse = np.where(mask, linear[..., c], 0.0).astype(np.float32)
            out[..., c] = fft_convolve2d(sparse, k)
        return out

    # -- full pipeline ----------------------------------------------------
    def capture_linear(
        self,
        linear_hdr: np.ndarray,
        exposure_ev: float = 0.0,
        motion_px: tuple[float, float] = (0.0, 0.0),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Scene radiance -> exposure-compensated linear estimate in scene units."""
        rng = rng if rng is not None else np.random.default_rng(0)
        x = self.optical(np.asarray(linear_hdr, dtype=np.float32))
        x = self.motion(x, motion_px)
        x = self.sensor(x, exposure_ev, rng)
        if self.bayer:
            x = self.mosaic_demosaic(x)
        return (x / (2.0**exposure_ev)).astype(np.float32)

    def capture_display(self, linear_hdr: np.ndarray, **kw) -> np.ndarray:
        return display_encode(self.capture_linear(linear_hdr, **kw))


def motion_px_for_exposure(motion_px_per_s: tuple[float, float], exposure_s: float) -> tuple[float, float]:
    return (float(motion_px_per_s[0]) * float(exposure_s), float(motion_px_per_s[1]) * float(exposure_s))
