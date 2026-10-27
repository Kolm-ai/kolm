"""Real-environment luminance statistics from HDR environment maps (DATA_PLAN.md section 4 step 7).

The harness's scene generator chooses how many stops a scene spans; this module measures that quantity on
real environments so the scene class S in a claim can quote a sourced distribution instead of a guess, and
so the camera baseline's dynamic range (9.3 stops for the measured phone, SPEC.md) is read against what
real environments actually contain. Input: Radiance RGBE files (.hdr) as published by Poly Haven (CC0),
equirectangular, relative radiance (no absolute calibration, so only ratios and stops are reported).

Reader: `read_rgbe` decodes the Radiance format (flat and new-style run-length scanlines) with numpy only.
Statistics per map (`hdri_stats`), on solid-angle-weighted luminance Y = 0.2126 R + 0.7152 G + 0.0722 B:
  dr_stops          log2(p99.9 / p0.1): the range that holds 99.8 % of the sphere, the "scene" range
  dr_stops_full     log2(max / p0.01): the range with the brightest source (sun, lamp) and the darkest shade
  peak_over_median  log2(max / median): how far the brightest source sits above the typical pixel
  bright_fraction   solid-angle fraction above 32x the median (sources and their halos)
  zero_fraction     solid-angle fraction of exactly zero pixels (clipped blacks in the capture)
  sky_over_ground   log2(median upper hemisphere / median lower hemisphere)
Solid-angle weights: each row of an equirectangular map is weighted by cos(latitude); percentiles are
weighted percentiles. Per category (Poly Haven `categories` and `attributes`) the deriver reports the
median, p10 and p90 of each statistic over the maps in that category."""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

METHOD = ("Radiance RGBE decoded with numpy; luminance Y = 0.2126 R + 0.7152 G + 0.0722 B on linear relative "
          "radiance; equirectangular rows weighted by cos(latitude); weighted percentiles; stops are log2 ratios. "
          "dr_stops = log2(p99.9/p0.1), dr_stops_full = log2(max/p0.01), peak_over_median = log2(max/median), "
          "bright_fraction = weight above 32x median, zero_fraction = weight of exact zeros, sky_over_ground = "
          "log2(median of upper hemisphere / median of lower hemisphere). Relative radiance only: Poly Haven "
          "maps carry no absolute calibration, so no cd/m2 is reported.")

_RES_RE = re.compile(rb"^-Y\s+(\d+)\s+\+X\s+(\d+)\s*$")


def read_rgbe(path: str | os.PathLike) -> np.ndarray:
    """Decode a Radiance .hdr file into float32 linear RGB (H, W, 3). Supports the flat 4-byte-per-pixel
    layout and the new-style run-length scanlines (2, 2, hi, lo header, four planes RLE). Standard
    orientation -Y H +X W only (what Poly Haven and most tools write)."""
    data = Path(path).read_bytes()
    if not data.startswith(b"#?"):
        raise ValueError(f"{path}: not a Radiance file")
    pos = 0
    while True:
        nl = data.index(b"\n", pos)
        line = data[pos:nl]
        pos = nl + 1
        if line.strip() == b"":
            break
    nl = data.index(b"\n", pos)
    m = _RES_RE.match(data[pos:nl].strip())
    if not m:
        raise ValueError(f"{path}: unsupported resolution line {data[pos:nl]!r}")
    pos = nl + 1
    h, w = int(m.group(1)), int(m.group(2))
    buf = np.frombuffer(data, dtype=np.uint8, offset=pos)
    rgbe = _decode_scanlines(buf, h, w)
    return rgbe_to_float(rgbe)


def _decode_scanlines(buf: np.ndarray, h: int, w: int) -> np.ndarray:
    out = np.empty((h, w, 4), dtype=np.uint8)
    p = 0
    n = buf.size
    for y in range(h):
        if p + 4 <= n and buf[p] == 2 and buf[p + 1] == 2 and (int(buf[p + 2]) << 8 | int(buf[p + 3])) == w \
                and 8 <= w <= 0x7FFF:
            p += 4
            for ch in range(4):
                x = 0
                row = out[y, :, ch]
                while x < w:
                    c = int(buf[p]); p += 1
                    if c > 128:
                        c -= 128
                        row[x:x + c] = buf[p]; p += 1
                    else:
                        row[x:x + c] = buf[p:p + c]; p += c
                    x += c
                if x != w:
                    raise ValueError("corrupt RLE scanline")
        else:
            out[y] = buf[p:p + 4 * w].reshape(w, 4)
            p += 4 * w
    return out


def rgbe_to_float(rgbe: np.ndarray) -> np.ndarray:
    """RGBE -> linear float32: mantissa bytes scaled by 2^(E - 128 - 8); E == 0 means black."""
    e = rgbe[..., 3].astype(np.int32)
    scale = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0).astype(np.float32)
    return rgbe[..., :3].astype(np.float32) * scale[..., None]


def float_to_rgbe(rgb: np.ndarray) -> np.ndarray:
    """Inverse of `rgbe_to_float` (for tests and writers): shared exponent from the max channel."""
    rgb = np.asarray(rgb, dtype=np.float64)
    mx = rgb.max(axis=-1)
    mant, expo = np.frexp(mx)                       # mx = mant * 2**expo, mant in [0.5, 1)
    scale = np.where(mx > 1e-32, 256.0 / np.ldexp(1.0, expo), 0.0)
    out = np.zeros(rgb.shape[:-1] + (4,), dtype=np.uint8)
    out[..., :3] = np.clip(np.floor(rgb * scale[..., None]), 0, 255).astype(np.uint8)
    out[..., 3] = np.where(mx > 1e-32, expo + 128, 0).astype(np.uint8)
    return out


def write_rgbe(path: str | os.PathLike, rgb: np.ndarray, rle: bool = False) -> None:
    """Minimal Radiance writer (flat, or new-style RLE with literal runs only) for tests and fixtures."""
    rgbe = float_to_rgbe(rgb)
    h, w = rgbe.shape[:2]
    chunks = [b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n", f"-Y {h} +X {w}\n".encode()]
    if rle and 8 <= w <= 0x7FFF:
        for y in range(h):
            chunks.append(bytes([2, 2, w >> 8, w & 255]))
            for ch in range(4):
                row = rgbe[y, :, ch].tobytes()
                for x in range(0, w, 127):
                    piece = row[x:x + 127]
                    chunks.append(bytes([len(piece)]) + piece)
    else:
        chunks.append(rgbe.tobytes())
    Path(path).write_bytes(b"".join(chunks))


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def latitude_weights(h: int, w: int) -> np.ndarray:
    """cos(latitude) per row of an equirectangular map, broadcast to (h, w), normalised to sum 1."""
    lat = (np.arange(h) + 0.5) / h * np.pi - np.pi / 2
    wgt = np.cos(lat)
    full = np.broadcast_to(wgt[:, None], (h, w))
    return full / full.sum()


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q) -> np.ndarray:
    """Weighted percentile(s) q in [0, 100] by cumulative weight (no interpolation)."""
    v = np.asarray(values).ravel()
    wt = np.asarray(weights).ravel()
    order = np.argsort(v, kind="stable")
    v, wt = v[order], wt[order]
    cum = np.cumsum(wt)
    cum = cum / cum[-1]
    qs = np.atleast_1d(np.asarray(q, dtype=np.float64)) / 100.0
    idx = np.minimum(np.searchsorted(cum, qs, side="left"), v.size - 1)
    return v[idx]


def hdri_stats(rgb: np.ndarray) -> dict:
    """Dynamic-range statistics of one equirectangular map (see module docstring). Zero pixels are kept
    for zero_fraction and excluded from the log ratios."""
    y = luminance(rgb.astype(np.float64))
    h, w = y.shape
    wt = latitude_weights(h, w)
    zero = y <= 0
    zero_fraction = float(wt[zero].sum())
    pos = ~zero
    if pos.sum() < 16:
        return {"ok": False, "reason": "fewer than 16 positive pixels"}
    yp, wp = y[pos], wt[pos]
    p001, p01, p50, p999 = weighted_percentile(yp, wp, [0.01, 0.1, 50.0, 99.9])
    mx = float(yp.max())
    log2 = np.log2
    upper = wt[: h // 2].sum()
    sky = weighted_percentile(y[: h // 2], wt[: h // 2], 50.0)[0] if upper > 0 else np.nan
    ground = weighted_percentile(y[h // 2:], wt[h // 2:], 50.0)[0]
    return {
        "ok": True,
        "height": int(h), "width": int(w),
        "dr_stops": float(log2(p999 / p01)),
        "dr_stops_full": float(log2(mx / p001)),
        "peak_over_median": float(log2(mx / p50)),
        "bright_fraction": float(wp[yp > 32.0 * p50].sum()),
        "zero_fraction": zero_fraction,
        "sky_over_ground": float(log2(sky / ground)) if sky > 0 and ground > 0 else None,
        "median_rel": float(p50), "max_rel": mx,
    }


def analyse_file(path: str) -> dict:
    name = Path(path).stem
    try:
        rgb = read_rgbe(path)
        row = hdri_stats(rgb)
    except Exception as exc:                                   # noqa: BLE001 - one bad file must not stop the run
        row = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    row["id"] = name
    row["file"] = Path(path).name
    return row


def _summary(rows: list[dict], keys=("dr_stops", "dr_stops_full", "peak_over_median", "bright_fraction",
                                     "zero_fraction", "sky_over_ground")) -> dict:
    out = {"n": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) is not None], dtype=np.float64)
        if v.size:
            p10, p50, p90 = np.percentile(v, [10, 50, 90])
            out[k] = {"median": float(p50), "p10": float(p10), "p90": float(p90), "n": int(v.size)}
    return out


def run(raw_dir: Path, index: dict | None, workers: int | None = None, log=None, limit: int = 0) -> dict:
    """Analyse every .hdr under raw_dir (recursively) in a process pool, join Poly Haven index metadata
    (categories, attributes, evs_cap) and summarise per category and per attribute value."""
    files = sorted(str(p) for p in Path(raw_dir).rglob("*.hdr"))
    if limit:
        files = files[:limit]
    rows: list[dict] = []
    if files:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, row in enumerate(ex.map(analyse_file, files, chunksize=4), 1):
                rows.append(row)
                if log and i % 100 == 0:
                    log(f"  {i}/{len(files)} maps")
    index = index or {}
    for r in rows:
        meta = index.get(r["id"], {})
        r["categories"] = sorted(meta.get("categories", []))
        r["attributes"] = meta.get("attributes", {})
        r["evs_cap"] = meta.get("evs_cap")
    ok = [r for r in rows if r.get("ok")]
    by_cat: dict[str, list[dict]] = {}
    for r in ok:
        for c in r["categories"]:
            by_cat.setdefault(c, []).append(r)
    by_attr: dict[str, list[dict]] = {}
    for r in ok:
        for k, v in (r.get("attributes") or {}).items():
            by_attr.setdefault(f"{k}={v}", []).append(r)
    return {
        "method": METHOD,
        "source": "polyhaven-hdris (CC0), 1k equirectangular Radiance HDR",
        "n_files": len(files), "n_ok": len(ok), "n_failed": len(rows) - len(ok),
        "all": _summary(ok),
        "by_category": {c: _summary(v) for c, v in sorted(by_cat.items()) if len(v) >= 5},
        "by_attribute": {c: _summary(v) for c, v in sorted(by_attr.items()) if len(v) >= 5},
        "maps": sorted(rows, key=lambda r: r["id"]),
    }


def load_index(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
