"""Display ceiling (pixels per degree) and delivery round-trips (encode, decode, re-measure)."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .colorspace import srgb_to_linear
from .io import to_srgb8

VISIBILITY_PPD = (94.0, 120.0)  # mean and upper resolution threshold, high-contrast foveal patterns


@dataclass(frozen=True)
class DisplaySpec:
    name: str = "display"
    diagonal_in: float = 31.5
    width_px: int = 7680
    height_px: int = 4320
    distance_m: float = 0.6
    delivery: dict | None = field(default=None)


def from_dict(d: dict) -> DisplaySpec:
    """Claim `display` block. `preset`: `derived:<id>` loads data/derived/displays.json (sourced size,
    resolution and viewing distance); explicit keys override."""
    import json
    d = dict(d)
    preset = d.pop("preset", None)
    base: dict = {}
    if preset:
        if not preset.startswith("derived:"):
            raise ValueError(f"display preset must be 'derived:<id>', got {preset!r}")
        from .data import data_root
        path = data_root() / "derived" / "displays.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `python -m harness data derive --ids displays`")
        rows = json.loads(path.read_text(encoding="utf-8"))["displays"]
        name = preset.split(":", 1)[1]
        if name not in rows:
            raise KeyError(f"derived display {name!r}; have {sorted(rows)}")
        r = rows[name]
        base = {"name": f"derived:{name}", "diagonal_in": r["diagonal_in"], "width_px": r["width_px"],
                "height_px": r["height_px"], "distance_m": r["distance_m"]}
    return DisplaySpec(**{**base, **d})


def pixel_pitch_m(d: DisplaySpec) -> float:
    return d.diagonal_in * 0.0254 / math.hypot(d.width_px, d.height_px)


def ppd(d: DisplaySpec) -> float:
    return 1.0 / math.degrees(2 * math.atan(pixel_pitch_m(d) / (2 * d.distance_m)))


def required_master_px(d: DisplaySpec, factor: float = 2.0) -> tuple[int, int]:
    return int(math.ceil(d.width_px * factor)), int(math.ceil(d.height_px * factor))


def encode_roundtrip(frame: np.ndarray, fmt: str = "jpeg", quality: int = 92) -> tuple[np.ndarray, int]:
    """8-bit sRGB encode/decode of one frame. Returns (linear frame, encoded bytes)."""
    buf = io.BytesIO()
    im = Image.fromarray(to_srgb8(frame))
    fmt = fmt.lower()
    if fmt in ("jpeg", "jpg"):
        im.save(buf, format="JPEG", quality=int(quality), subsampling=0)
    elif fmt == "png":
        im.save(buf, format="PNG")
    elif fmt == "webp":
        im.save(buf, format="WEBP", quality=int(quality))
    else:
        raise ValueError(f"unknown delivery format {fmt!r}")
    n = buf.tell()
    buf.seek(0)
    back = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float64) / 255.0
    return srgb_to_linear(back), n


def encode_roundtrip_frames(frames: np.ndarray, fmt: str = "jpeg", quality: int = 92,
                            fps: float = 24.0) -> tuple[np.ndarray, dict]:
    """Round-trip a clip. 'h264' needs imageio + ffmpeg and falls back to per-frame JPEG with a note."""
    info: dict = {"format": fmt, "quality": quality}
    if fmt.lower() == "h264":
        try:
            import imageio.v3 as iio  # optional
            buf = io.BytesIO()
            u8 = np.stack([to_srgb8(f) for f in frames])
            iio.imwrite(buf, u8, extension=".mp4", fps=fps, codec="libx264",
                        output_params=["-crf", str(int(quality))])
            info["bytes"] = buf.tell()
            buf.seek(0)
            back = np.stack([np.asarray(f, dtype=np.float64)[..., :3] / 255.0
                             for f in iio.imiter(buf, extension=".mp4")])
            return srgb_to_linear(back), info
        except Exception as e:  # noqa: BLE001 - any encoder failure degrades to per-frame JPEG
            info["note"] = f"h264 unavailable ({type(e).__name__}); used per-frame JPEG"
            fmt = "jpeg"
    out, total = [], 0
    for f in frames:
        g, n = encode_roundtrip(f, fmt, quality)
        out.append(g)
        total += n
    info["format"], info["bytes"] = fmt, total
    return np.stack(out), info
