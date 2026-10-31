"""Baseline TIFF writer (struct only): RGB, uncompressed, 16-bit unsigned or 32-bit float samples, written in
strips of STRIP_ROWS rows, little-endian. Pillow writes neither 16-bit nor float RGB. A 16-bit file carries no ICC
profile and is read as sRGB, so the photo command writes sRGB-encoded values there; a float file is read as linear
by the editors that take 32-bit TIFF (TIFF 6.0 with the SampleFormat tag, 339)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

STRIP_ROWS = 64
SHORT, LONG, RATIONAL, ASCII = 3, 4, 5, 2


def write_tiff(path, img: np.ndarray, software: str = "") -> None:
    """(H, W, 3) uint16 or float32 -> path."""
    a = np.ascontiguousarray(img)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype not in (np.uint16, np.float32):
        raise ValueError(f"expected (H, W, 3) uint16 or float32, got {a.shape} {a.dtype}")
    a = a.astype(a.dtype.newbyteorder("<"), copy=False)
    h, w = a.shape[:2]
    bits, fmt = (16, 1) if a.dtype.kind == "u" else (32, 3)
    row = w * 3 * a.dtype.itemsize
    strips = [(y, min(h, y + STRIP_ROWS)) for y in range(0, h, STRIP_ROWS)]
    if 8 + h * row + 4096 + 8 * len(strips) > 0xFFFFFFFF:
        raise ValueError("frame too large for a classic TIFF")
    sw = (software or "harness").encode("ascii", "replace") + b"\0"
    entries = []        # (tag, type, count, payload bytes)

    def add(tag, typ, values):
        if typ == ASCII:
            entries.append((tag, typ, len(values), values))
        elif typ == RATIONAL:
            entries.append((tag, typ, len(values) // 2, struct.pack(f"<{len(values)}I", *values)))
        else:
            entries.append((tag, typ, len(values), struct.pack(f"<{len(values)}{'H' if typ == SHORT else 'I'}", *values)))

    data_at = 8
    offsets = [data_at + y0 * row for y0, _ in strips]
    counts = [(y1 - y0) * row for y0, y1 in strips]
    add(256, LONG, [w])
    add(257, LONG, [h])
    add(258, SHORT, [bits] * 3)
    add(259, SHORT, [1])
    add(262, SHORT, [2])
    add(273, LONG, offsets)
    add(277, SHORT, [3])
    add(278, LONG, [STRIP_ROWS])
    add(279, LONG, counts)
    add(282, RATIONAL, [72, 1])
    add(283, RATIONAL, [72, 1])
    add(284, SHORT, [1])
    add(296, SHORT, [2])
    add(305, ASCII, sw)
    add(339, SHORT, [fmt] * 3)
    entries.sort(key=lambda e: e[0])
    ifd_at = data_at + h * row
    ifd_at += ifd_at & 1
    extra_at = ifd_at + 2 + 12 * len(entries) + 4
    ifd, extra = [struct.pack("<H", len(entries))], []
    for tag, typ, count, payload in entries:
        if len(payload) <= 4:
            ifd.append(struct.pack("<HHI", tag, typ, count) + payload.ljust(4, b"\0"))
        else:
            ifd.append(struct.pack("<HHII", tag, typ, count, extra_at + sum(len(e) for e in extra)))
            extra.append(payload + (b"\0" if len(payload) & 1 else b""))
    ifd.append(struct.pack("<I", 0))
    with open(Path(path), "wb") as f:
        f.write(b"II" + struct.pack("<HI", 42, ifd_at))
        f.write(a.tobytes())
        f.write(b"\0" * (ifd_at - data_at - h * row))
        f.write(b"".join(ifd))
        f.write(b"".join(extra))


def read_tiff(path) -> np.ndarray:
    """The inverse for what write_tiff writes (tests and round trips): (H, W, 3) uint16 or float32."""
    from .data.dng import read_tags
    buf = Path(path).read_bytes()
    i0 = read_tags(buf)["ifd0"]
    w, h = int(_one(i0["ImageWidth"])), int(_one(i0["ImageLength"]))
    bits = _list(i0["BitsPerSample"])[0]
    dt = np.dtype("<u2") if bits == 16 else np.dtype("<f4")
    raw = b"".join(buf[o:o + c] for o, c in zip(_list(i0["StripOffsets"]), _list(i0["StripByteCounts"])))
    return np.frombuffer(raw, dtype=dt).reshape(h, w, 3).astype(dt.newbyteorder("="))


def _list(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _one(v):
    return _list(v)[0]
