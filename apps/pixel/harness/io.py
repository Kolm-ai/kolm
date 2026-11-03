"""Load and save candidates. Frames are (T, H, W, 3) float64 linear RGB; stills have T = 1.

Accepted inputs: 8-bit sRGB images (PNG/JPEG/...), a `.npy` float linear array (H,W,3) or (T,H,W,3),
a directory of image frames or of one `.npy`, or a video file if `imageio` + ffmpeg are installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .colorspace import linear_to_srgb, srgb_to_linear

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def load_image(path) -> np.ndarray:
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(im, dtype=np.float64) / 65535.0
        arr = np.repeat(arr[..., None], 3, axis=-1)
    else:
        arr = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
    return srgb_to_linear(arr)


def _as_frames(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 3:
        a = a[None]
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3) or (H,W,3), got {a.shape}")
    return a


def load_frames(path) -> np.ndarray:
    p = Path(path)
    if p.is_dir():
        npys = sorted(p.glob("*.npy"))
        if len(npys) == 1:
            return _as_frames(np.load(npys[0]))
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT)
        if not files:
            raise ValueError(f"{p}: no image frames or .npy stack found")
        return np.stack([load_image(f) for f in files])
    suf = p.suffix.lower()
    if suf == ".npy":
        return _as_frames(np.load(p))
    if suf in IMAGE_EXT:
        return load_image(p)[None]
    if suf in VIDEO_EXT:
        try:
            import imageio.v3 as iio  # optional dependency
        except ImportError as e:
            raise ImportError("video files need `pip install imageio imageio-ffmpeg`; "
                              "or pass a directory of PNG frames / a .npy stack") from e
        frames = np.stack([np.asarray(f, dtype=np.float64)[..., :3] / 255.0 for f in iio.imiter(p)])
        return srgb_to_linear(frames)
    raise ValueError(f"{p}: unsupported input")


def to_srgb8(frame: np.ndarray) -> np.ndarray:
    return (linear_to_srgb(frame) * 255 + 0.5).astype(np.uint8)


def save_png(path, frame: np.ndarray) -> None:
    Image.fromarray(to_srgb8(frame)).save(path)


def save_frames(directory, frames: np.ndarray, stem: str = "frame") -> None:
    """Lossless `.npy` master plus 8-bit sRGB PNGs for viewing."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{stem}.npy", frames.astype(np.float32))
    if frames.shape[0] == 1:
        save_png(d / f"{stem}.png", frames[0])
    else:
        for k, f in enumerate(frames):
            save_png(d / f"{stem}_{k:04d}.png", f)


def load_mask(src, shape: tuple[int, int, int]) -> np.ndarray:
    """A per-pixel declaration as bool (T,H,W): an array, a .npy, or an image where luminance > 0.5 means
    rendered. (H,W) broadcasts over the frames of `shape` = (T,H,W)."""
    if isinstance(src, (str, Path)):
        p = Path(src)
        if p.suffix.lower() == ".npy":
            a = np.load(p)
        elif p.suffix.lower() in IMAGE_EXT:
            a = np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255.0
        else:
            raise ValueError(f"{p}: a mask is a .npy array or an image")
    else:
        a = np.asarray(src)
    if a.dtype != bool:
        a = a.astype(np.float64)
        if a.ndim == 4:
            a = a.mean(axis=-1)
        a = a > 0.5
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3 or a.shape[1:] != tuple(shape[1:]) or a.shape[0] not in (1, shape[0]):
        raise ValueError(f"mask shape {a.shape} does not match frames {tuple(shape)}")
    return np.broadcast_to(a, tuple(shape)).copy()


def save_mask(path, mask: np.ndarray) -> None:
    Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8)).save(path)
