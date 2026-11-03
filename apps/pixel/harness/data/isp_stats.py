"""Measured ISP-side degradation parameters, fitted on the TID2013 reference/distorted pairs on disk.

The camera model covers optics and sensor. The harness metrics also respond to what an ISP or an
editor does afterwards: JPEG quantisation, tone and contrast curves, mean shift, colour saturation,
lateral chromatic shift. `data/derived/` holds no distribution for those, so this module measures
one from the only human-rated pairs on disk (TID2013: 25 references x 24 types x 5 levels) by
fitting each distorted image's parameter against its pristine reference:

  JPEG (#10)            quality factor: the Pillow quality whose re-encoding of the reference is
                        closest (MSE) to the distorted image
  mean shift (#16)      median sRGB offset distorted - reference
  contrast change (#17) gain of the sRGB luminance about mid-grey, least squares
  saturation (#18)      ratio of chroma spread (sRGB a*b* std) distorted / reference
  chromatic aberr (#23) red-to-green and blue-to-green displacement (px), by phase
                        correlation of each channel against green, distorted minus reference

Output: data/derived/isp_stats.json, one list of fitted values per parameter plus the fit quality,
read by `harness.data.degrade.Distributions`. Values the sampler draws are the fitted ones
(empirical resampling), never a hand-set range.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np

from . import data_root
from ..colorspace import linear_rgb_to_lab, linear_to_srgb, luminance, srgb_to_linear

TID_JPEG, TID_MEAN_SHIFT, TID_CONTRAST, TID_SATURATION, TID_CA = 10, 16, 17, 18, 23
TID_TYPES = {TID_JPEG: "jpeg", TID_MEAN_SHIFT: "mean_shift", TID_CONTRAST: "contrast",
             TID_SATURATION: "saturation", TID_CA: "chromatic_aberration"}
JPEG_QUALITIES = tuple(range(2, 101))


def _decode(blob: bytes) -> np.ndarray:
    from PIL import Image
    im = Image.open(io.BytesIO(blob)).convert("RGB")
    return np.asarray(im, dtype=np.float64) / 255.0  # sRGB-encoded 0..1


def _jpeg_roundtrip(srgb: np.ndarray, quality: int) -> np.ndarray:
    from PIL import Image
    im = Image.fromarray(np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8), "RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality), subsampling=0 if quality >= 90 else 2)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float64) / 255.0


def fit_jpeg_quality(ref: np.ndarray, dist: np.ndarray, qualities=JPEG_QUALITIES) -> tuple[int, float]:
    """Quality whose re-encoding of the reference is nearest to the distorted image; coarse-to-fine."""
    def err(q):
        d = _jpeg_roundtrip(ref, q) - dist
        return float(np.mean(d * d))
    coarse = list(range(2, 101, 7))
    best = min(coarse, key=err)
    fine = [q for q in qualities if abs(q - best) <= 7]
    best = min(fine, key=err)
    return int(best), err(best)


def fit_mean_shift(ref: np.ndarray, dist: np.ndarray) -> float:
    return float(np.median(dist - ref))


def fit_contrast(ref: np.ndarray, dist: np.ndarray) -> tuple[float, float]:
    """dist_lum - 0.5 = gain * (ref_lum - 0.5) in sRGB luminance, least squares; returns (gain, r2)."""
    x = luminance(ref).ravel() - 0.5
    y = luminance(dist).ravel() - 0.5
    gain = float(np.dot(x, y) / max(np.dot(x, x), 1e-12))
    res = y - gain * x
    r2 = 1.0 - float(np.dot(res, res) / max(np.dot(y - y.mean(), y - y.mean()), 1e-12))
    return gain, r2


def fit_saturation(ref: np.ndarray, dist: np.ndarray) -> float:
    lr = linear_rgb_to_lab(srgb_to_linear(ref))
    ld = linear_rgb_to_lab(srgb_to_linear(dist))
    cr = np.hypot(lr[..., 1], lr[..., 2])
    cd = np.hypot(ld[..., 1], ld[..., 2])
    return float(cd.mean() / max(cr.mean(), 1e-9))


def _phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Sub-pixel (dy, dx) that maps a onto b, by phase correlation with a parabolic peak fit."""
    a = a - a.mean()
    b = b - b.mean()
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = np.fft.fft2(a * win), np.fft.fft2(b * win)
    cross = fb * np.conj(fa)
    cross /= np.maximum(np.abs(cross), 1e-12)
    pc = np.real(np.fft.ifft2(cross))
    iy, ix = np.unravel_index(int(np.argmax(pc)), pc.shape)
    out = []
    for i, n, axis in ((iy, pc.shape[0], 0), (ix, pc.shape[1], 1)):
        m = pc[iy, :] if axis == 1 else pc[:, ix]
        c0, cm, cp = m[i], m[(i - 1) % n], m[(i + 1) % n]
        den = cm - 2 * c0 + cp
        frac = 0.5 * (cm - cp) / den if abs(den) > 1e-12 else 0.0
        v = i + frac
        if v > n / 2:
            v -= n
        out.append(float(v))
    return out[0], out[1]


def fit_chromatic_shift(ref: np.ndarray, dist: np.ndarray) -> float:
    """Displacement (px) of the red and blue channels relative to green that `dist` adds over `ref`,
    by whole-frame phase correlation of each channel against green; returns the mean of |red| and
    |blue|. TID2013 #23 translates red by +s px and blue by -s px horizontally (s = 0.2, 0.9, 1.0,
    1.9, 5.0 at levels 1..5); the sampler applies the same amplitude as a radial lateral shift."""
    mags = []
    for ch in (0, 2):
        dy_d, dx_d = _phase_shift(dist[..., 1], dist[..., ch])
        dy_r, dx_r = _phase_shift(ref[..., 1], ref[..., ch])
        mags.append(float(np.hypot(dy_d - dy_r, dx_d - dx_r)))
    return float(np.mean(mags))


def iter_tid_pairs(files: list[Path], types: set[int]):
    """Yield (reference_id, distortion_id, level, ref_srgb, dist_srgb) for the wanted distortion types."""
    import pyarrow.parquet as pq
    for f in sorted(files):
        pf = pq.ParquetFile(str(f))
        for rg in range(pf.metadata.num_row_groups):
            meta = pf.read_row_group(rg, columns=["reference_id", "distortion_id", "distortion_intensity"]).to_pylist()
            hits = [i for i, m in enumerate(meta) if m["distortion_id"] in types]
            if not hits:
                continue
            tab = pf.read_row_group(rg, columns=["reference", "distorted"])
            for i in hits:
                m = meta[i]
                yield (m["reference_id"], m["distortion_id"], m["distortion_intensity"],
                       _decode(tab.column("reference")[i].as_py()["bytes"]),
                       _decode(tab.column("distorted")[i].as_py()["bytes"]))


def derive_white_balance(root: Path, log=print) -> dict[str, list]:
    """Camera-space AsShotNeutral of every HDR+ burst payload on disk (one DNG per burst): the
    illuminant the camera measured. R/G and B/G of the neutral, aligned by burst. A colour cast is
    white-balancing for one burst's illuminant when the scene had another's, so the sampler draws
    two bursts and applies the ratio of their neutrals."""
    import rawpy
    bursts = sorted((root / "raw" / "hdr-plus-burst" / "bursts").glob("*/payload_N000.dng"))
    names, rg, bg = [], [], []
    for f in bursts:
        try:
            with rawpy.imread(str(f)) as raw:
                wb = [float(x) for x in raw.camera_whitebalance]
        except Exception as e:  # noqa: BLE001 - a bad DNG is skipped, counted below
            log(f"isp_stats: white balance unreadable {f.parent.name}: {e}")
            continue
        if len(wb) < 3 or wb[1] <= 0 or wb[0] <= 0 or wb[2] <= 0:
            continue
        names.append(f.parent.name)
        rg.append(wb[0] / wb[1])  # rawpy reports multipliers (1 / neutral); the ratio is what the cast needs
        bg.append(wb[2] / wb[1])
    log(f"isp_stats: white balance read on {len(names)} of {len(bursts)} bursts")
    return {"bursts": names, "r_over_g": rg, "b_over_g": bg}


def derive_isp_stats(root: Path | None = None, log=print, limit_per_type: int | None = None) -> dict:
    """Fit every TID2013 pair of the five ISP types and write data/derived/isp_stats.json."""
    root = Path(root) if root else data_root()
    files = sorted((root / "raw" / "hf-tid2013").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{root / 'raw' / 'hf-tid2013'}: no parquet shards; run "
                                "`python -m harness data collect --ids hf-tid2013`")
    rows = {name: [] for name in TID_TYPES.values()}
    t0 = time.time()
    n = 0
    for ref_id, dtype, level, ref, dist in iter_tid_pairs(files, set(TID_TYPES)):
        name = TID_TYPES[dtype]
        if limit_per_type is not None and len(rows[name]) >= limit_per_type:
            continue
        row = {"reference": ref_id, "level": level}
        if dtype == TID_JPEG:
            q, mse = fit_jpeg_quality(ref, dist)
            row.update(quality=q, mse=mse, psnr_db=float(10 * np.log10(1.0 / max(mse, 1e-12))))
        elif dtype == TID_MEAN_SHIFT:
            row.update(offset=fit_mean_shift(ref, dist))
        elif dtype == TID_CONTRAST:
            g, r2 = fit_contrast(ref, dist)
            row.update(gain=g, r2=r2)
        elif dtype == TID_SATURATION:
            row.update(ratio=fit_saturation(ref, dist))
        elif dtype == TID_CA:
            row.update(shift_px=fit_chromatic_shift(ref, dist))
        rows[name].append(row)
        n += 1
        if n % 50 == 0:
            log(f"isp_stats: {n} pairs fitted, {time.time() - t0:.0f} s")

    def summary(vals):
        v = np.asarray(vals, dtype=np.float64)
        if v.size == 0:
            return {"n": 0}
        return {"n": int(v.size), "min": float(v.min()), "p10": float(np.percentile(v, 10)),
                "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max())}

    keys = {"jpeg": "quality", "mean_shift": "offset", "contrast": "gain", "saturation": "ratio",
            "chromatic_aberration": "shift_px"}
    out = {
        "method": __doc__.strip(),
        "source": "hf-tid2013 (TID2013 reference/distorted BMP pairs, 512x384 sRGB); distortion ids "
                  f"{dict((k, v) for k, v in TID_TYPES.items())}",
        "n_pairs": n,
        "seconds": round(time.time() - t0, 1),
        "parameters": {},
    }
    for name, key in keys.items():
        vals = [r[key] for r in rows[name]]
        by_level = {}
        for lvl in sorted({r["level"] for r in rows[name]}):
            by_level[str(lvl)] = summary([r[key] for r in rows[name] if r["level"] == lvl])
        entry = {"key": key, "values": vals, "summary": summary(vals), "by_level": by_level,
                 "rows": rows[name]}
        if name == "jpeg" and rows[name]:
            entry["psnr_db_median"] = float(np.median([r["psnr_db"] for r in rows[name]]))
        if name == "contrast" and rows[name]:
            entry["r2_median"] = float(np.median([r["r2"] for r in rows[name]]))
        out["parameters"][name] = entry
    wb = derive_white_balance(root, log)
    for key in ("r_over_g", "b_over_g"):
        out["parameters"]["wb_" + key] = {
            "key": key, "values": wb[key], "summary": summary(wb[key]), "by_level": {},
            "rows": [{"burst": b, key: v} for b, v in zip(wb["bursts"], wb[key])],
            "source": "hdr-plus-burst payload_N000.dng AsShotNeutral via rawpy camera_whitebalance "
                      "(camera-space multipliers, ratio to green)"}
    path = root / "derived" / "isp_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"isp_stats: {n} pairs -> {path} ({time.time() - t0:.0f} s)")
    return out
