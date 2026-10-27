"""Small image operations shared by the camera, the degrader and the metrics.

numpy + Pillow only. Arrays are float32 ``(H, W)`` or ``(H, W, 3)``.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image


def _pad_reflect(img: np.ndarray, py: int, px: int) -> np.ndarray:
    pad = [(py, py), (px, px)] + [(0, 0)] * (img.ndim - 2)
    return np.pad(img, pad, mode="reflect")


def fft_convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """'same' convolution with reflect padding, applied per channel via FFT."""
    img = np.asarray(img, dtype=np.float32)
    kh, kw = kernel.shape
    py, px = kh // 2, kw // 2
    padded = _pad_reflect(img, py, px)
    H, W = padded.shape[:2]
    k = np.zeros((H, W), dtype=np.float32)
    k[:kh, :kw] = kernel
    k = np.roll(k, (-py, -px), axis=(0, 1))
    K = np.fft.rfft2(k)
    if padded.ndim == 2:
        out = np.fft.irfft2(np.fft.rfft2(padded) * K, s=(H, W))
    else:
        out = np.stack(
            [np.fft.irfft2(np.fft.rfft2(padded[..., c]) * K, s=(H, W)) for c in range(padded.shape[2])],
            axis=-1,
        )
    return out[py : py + img.shape[0], px : px + img.shape[1]].astype(np.float32)


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(img, dtype=np.float32).copy()
    k = gaussian_kernel1d(sigma)
    return fft_convolve2d(img, np.outer(k, k))


def box_blur(img: np.ndarray, size: int = 5) -> np.ndarray:
    k = np.full((size, size), 1.0 / (size * size), dtype=np.float32)
    return fft_convolve2d(img, k)


def motion_kernel(length_px: float, angle_rad: float, supersample: int = 8) -> np.ndarray:
    """Line kernel of the given length and direction, sub-pixel accurate."""
    length_px = max(float(length_px), 1e-3)
    radius = int(np.ceil(length_px / 2.0)) + 1
    n = 2 * radius + 1
    k = np.zeros((n, n), dtype=np.float64)
    steps = max(2, int(np.ceil(length_px * supersample)))
    ts = np.linspace(-0.5, 0.5, steps) * length_px
    xs = radius + ts * np.cos(angle_rad)
    ys = radius + ts * np.sin(angle_rad)
    for x, y in zip(xs, ys):
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                yy, xx = y0 + dy, x0 + dx
                if 0 <= yy < n and 0 <= xx < n:
                    k[yy, xx] += wx * wy
    return (k / k.sum()).astype(np.float32)


def shift_subpixel(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate by (dx, dy) pixels with bilinear interpolation and edge replication."""
    img = np.asarray(img, dtype=np.float32)
    ix, iy = int(np.floor(dx)), int(np.floor(dy))
    fx, fy = dx - ix, dy - iy

    def _int_shift(a: np.ndarray, sx: int, sy: int) -> np.ndarray:
        out = np.roll(a, (sy, sx), axis=(0, 1))
        if sy > 0:
            out[:sy] = out[sy : sy + 1]
        elif sy < 0:
            out[sy:] = out[sy - 1 : sy]
        if sx > 0:
            out[:, :sx] = out[:, sx : sx + 1]
        elif sx < 0:
            out[:, sx:] = out[:, sx - 1 : sx]
        return out

    a = _int_shift(img, ix, iy)
    b = _int_shift(img, ix + 1, iy)
    c = _int_shift(img, ix, iy + 1)
    d = _int_shift(img, ix + 1, iy + 1)
    return ((1 - fx) * (1 - fy) * a + fx * (1 - fy) * b + (1 - fx) * fy * c + fx * fy * d).astype(np.float32)


def local_min_max(img: np.ndarray, radius: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel min and max over a (2r+1)^2 neighbourhood, edges replicated."""
    img = np.asarray(img, dtype=np.float32)
    pad = [(radius, radius), (radius, radius)] + [(0, 0)] * (img.ndim - 2)
    p = np.pad(img, pad, mode="edge")
    H, W = img.shape[:2]
    lo = np.full_like(img, np.inf)
    hi = np.full_like(img, -np.inf)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            win = p[dy : dy + H, dx : dx + W]
            lo = np.minimum(lo, win)
            hi = np.maximum(hi, win)
    return lo, hi


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(img, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def from_uint8(arr: np.ndarray) -> np.ndarray:
    return (np.asarray(arr).astype(np.float32) / 255.0).astype(np.float32)


def resize(img: np.ndarray, width: int, height: int, resample: str = "bilinear") -> np.ndarray:
    """Resize a float image through Pillow in 8-bit-free float mode per channel."""
    img = np.asarray(img, dtype=np.float32)
    mode = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "box": Image.BOX, "lanczos": Image.LANCZOS}[resample]
    if img.ndim == 2:
        return np.asarray(Image.fromarray(img, mode="F").resize((width, height), mode), dtype=np.float32)
    return np.stack(
        [np.asarray(Image.fromarray(img[..., c], mode="F").resize((width, height), mode), dtype=np.float32) for c in range(img.shape[2])],
        axis=-1,
    )


def encode_image(img: np.ndarray, fmt: str = "png", quality: int = 92) -> bytes:
    """Encode a display-domain float image to bytes (png / jpeg / webp)."""
    im = Image.fromarray(to_uint8(img))
    buf = io.BytesIO()
    fmt = fmt.lower()
    if fmt in ("jpg", "jpeg"):
        im.save(buf, format="JPEG", quality=int(quality), subsampling=2, optimize=True)
    elif fmt == "webp":
        im.save(buf, format="WEBP", quality=int(quality))
    else:
        im.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def decode_image(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return from_uint8(np.asarray(im.convert("RGB")))


def load_image(path: str | Path) -> np.ndarray:
    return decode_image(Path(path).read_bytes())


def save_image(path: str | Path, img: np.ndarray, quality: int = 92) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_image(img, fmt=path.suffix.lstrip("."), quality=quality))
