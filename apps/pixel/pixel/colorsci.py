"""Colour science: sRGB transfer, Lab, CIEDE2000, and the display tonemap.

All image arrays are float32 ``(H, W, 3)``. "linear" means linear sRGB primaries,
"display" means sRGB-encoded in [0, 1]. Scene radiance is HDR linear and may
exceed 1; ``display_encode`` is the one tonemap used for reference and candidate
alike so the comparison is apples to apples.
"""

from __future__ import annotations

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

M_RGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def srgb_decode(encoded: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(encoded, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Weighted luminance of an ``(H, W, 3)`` array in whatever domain it is in."""
    return np.tensordot(np.asarray(rgb, dtype=np.float32), LUMA, axes=([-1], [0]))


def tonemap(linear_hdr: np.ndarray) -> np.ndarray:
    """Global Reinhard ``L / (1 + L)``: monotonic, never clips, keeps highlight texture.

    Mid grey 0.18 maps to 0.153, radiance 8 maps to 0.889. The same curve is used
    for ground truth and for the simulated camera (after exposure compensation),
    so the only thing that differs between them is what the sensor could record.
    """
    x = np.maximum(np.asarray(linear_hdr, dtype=np.float32), 0.0)
    return x / (1.0 + x)


def display_encode(linear_hdr: np.ndarray) -> np.ndarray:
    return srgb_encode(tonemap(linear_hdr))


def linear_to_lab(linear: np.ndarray) -> np.ndarray:
    rgb = np.asarray(linear, dtype=np.float64).reshape(-1, 3)
    xyz = rgb @ M_RGB2XYZ.T / D65
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=-1).reshape(np.shape(linear))


def display_to_lab(display: np.ndarray) -> np.ndarray:
    return linear_to_lab(srgb_decode(display))


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (Sharma, Wu & Dalal 2005), vectorised over leading dims."""
    lab1 = np.asarray(lab1, dtype=np.float64).reshape(-1, 3)
    lab2 = np.asarray(lab2, dtype=np.float64).reshape(-1, 3)
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - np.sqrt(Cbar**7 / (Cbar**7 + 25.0**7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    def hue(ap, b):
        h = np.degrees(np.arctan2(b, ap))
        h = np.where(h < 0, h + 360.0, h)
        return np.where((ap == 0) & (b == 0), 0.0, h)

    h1p = hue(a1p, b1)
    h2p = hue(a2p, b2)
    dLp = L2 - L1
    dCp = C2p - C1p
    zero = (C1p * C2p) == 0
    dh = h2p - h1p
    dhp = np.where(zero, 0.0, np.where(np.abs(dh) <= 180.0, dh, np.where(dh > 180.0, dh - 360.0, dh + 360.0)))
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)
    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hbp = np.where(
        zero,
        hsum,
        np.where(np.abs(h1p - h2p) <= 180.0, 0.5 * hsum, np.where(hsum < 360.0, 0.5 * (hsum + 360.0), 0.5 * (hsum - 360.0))),
    )
    T = (
        1.0
        - 0.17 * np.cos(np.radians(hbp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hbp))
        + 0.32 * np.cos(np.radians(3.0 * hbp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hbp - 63.0))
    )
    dtheta = 30.0 * np.exp(-(((hbp - 275.0) / 25.0) ** 2))
    RC = 2.0 * np.sqrt(Cbp**7 / (Cbp**7 + 25.0**7))
    SL = 1.0 + 0.015 * (Lbp - 50.0) ** 2 / np.sqrt(20.0 + (Lbp - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cbp
    SH = 1.0 + 0.015 * Cbp * T
    RT = -np.sin(np.radians(2.0 * dtheta)) * RC
    dE = np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH))
    return dE
