"""Optics and photon statistics. Pure functions, numpy only.

Every number a claim leans on comes from here, so the formulas are the
textbook ones and the tests pin them to known values.
"""

from __future__ import annotations

import math

import numpy as np

# Representative sensor wavelengths for the R, G, B channels (metres).
RGB_WAVELENGTHS_M = (0.610e-6, 0.550e-6, 0.460e-6)


def rayleigh_angle_rad(wavelength_m: float, aperture_m: float) -> float:
    """Rayleigh angular resolution limit for a circular aperture: 1.22 lambda / D."""
    return 1.22 * wavelength_m / aperture_m


def airy_first_zero_m(f_number: float, wavelength_m: float) -> float:
    """Radius of the first dark ring of the Airy pattern on the sensor plane."""
    return 1.22 * wavelength_m * f_number


def bessel_j1(x: np.ndarray | float) -> np.ndarray:
    """Bessel function of the first kind, order 1 (Abramowitz & Stegun 9.4.4 / 9.4.6).

    Absolute error below 1e-7 over the real line. Implemented here so the package
    stays numpy-only.
    """
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    out = np.empty_like(ax)
    small = ax < 3.0
    if np.any(small):
        y = (ax[small] / 3.0) ** 2
        out[small] = ax[small] * (
            0.5
            - 0.56249985 * y
            + 0.21093573 * y**2
            - 0.03954289 * y**3
            + 0.00443319 * y**4
            - 0.00031761 * y**5
            + 0.00001109 * y**6
        )
    big = ~small
    if np.any(big):
        xb = ax[big]
        yb = 3.0 / xb
        f1 = (
            0.79788456
            + 0.00000156 * yb
            + 0.01659667 * yb**2
            + 0.00017105 * yb**3
            - 0.00249511 * yb**4
            + 0.00113653 * yb**5
            - 0.00020033 * yb**6
        )
        th = (
            xb
            - 2.35619449
            + 0.12499612 * yb
            + 0.00005650 * yb**2
            - 0.00637879 * yb**3
            + 0.00074348 * yb**4
            + 0.00079824 * yb**5
            - 0.00029166 * yb**6
        )
        out[big] = f1 * np.cos(th) / np.sqrt(xb)
    return np.where(x < 0, -out, out)


def airy_psf(
    f_number: float,
    wavelength_m: float,
    pixel_pitch_m: float,
    supersample: int = 4,
    rings: float = 3.0,
) -> np.ndarray:
    """Diffraction point-spread function sampled on the pixel grid, normalised to sum 1.

    Intensity follows (2 J1(x) / x)^2 with x = pi r / (lambda N). The kernel
    extends ``rings`` first-zero radii, is supersampled ``supersample`` times per
    pixel and box-integrated back onto pixels, so sub-pixel PSFs (fast lenses on
    coarse pixels) are handled correctly instead of collapsing to a delta.
    """
    r0 = airy_first_zero_m(f_number, wavelength_m)
    radius_px = max(1, int(math.ceil(rings * r0 / pixel_pitch_m)))
    n = 2 * radius_px + 1
    s = max(1, int(supersample))
    coords = (np.arange(n * s) - (n * s - 1) / 2.0) / s
    xx, yy = np.meshgrid(coords, coords)
    r = np.sqrt(xx**2 + yy**2) * pixel_pitch_m
    x = np.pi * r / (wavelength_m * f_number)
    safe = np.where(x == 0, 1.0, x)
    intensity = np.where(x == 0, 1.0, (2.0 * bessel_j1(safe) / safe) ** 2)
    psf = intensity.reshape(n, s, n, s).sum(axis=(1, 3))
    return (psf / psf.sum()).astype(np.float32)


def pixels_per_degree(diagonal_in: float, width_px: int, height_px: int, distance_m: float) -> float:
    """Angular pixel density at screen centre for a flat display viewed on-axis."""
    diag_m = diagonal_in * 0.0254
    pitch_m = diag_m / math.hypot(width_px, height_px)
    deg_per_px = math.degrees(2.0 * math.atan(pitch_m / (2.0 * distance_m)))
    return 1.0 / deg_per_px


def photon_snr(signal_e: float | np.ndarray, read_noise_e: float = 0.0) -> np.ndarray:
    """Signal-to-noise ratio for a shot-noise-limited detector with Gaussian read noise."""
    s = np.asarray(signal_e, dtype=np.float64)
    return s / np.sqrt(np.maximum(s + read_noise_e**2, 1e-12))


def snr_db(snr: float | np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(snr, dtype=np.float64), 1e-12))
