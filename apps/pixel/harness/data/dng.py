"""Minimal TIFF/DNG tag reader (struct only, no image decode) and the capture statistics derived from real
raw frames: exposure time, ISO, f-number, the exposure value and the scene luminance it implies, the
manufacturer's noise profile converted to gain and read noise in electrons, and the lens shading
(vignetting) map's corner-to-centre range. Used on the HDR+ burst first frames (DATA_PLAN.md section 4 step 9).

Conventions
- EV100 = log2(N^2 / t) - log2(ISO / 100); scene luminance L = 2^EV100 * K / 100 cd/m^2 with K = 12.5
  (ISO 2720:1974 reflected-light meter calibration constant; the standard allows 10.6 to 13.4).
- DNG NoiseProfile holds (scale, offset) per CFA plane with var(x) = scale * x + offset for the signal x
  normalised to 0..1 between BlackLevel and WhiteLevel. With R = white - black the model in DN is
  var_DN = R * scale * DN + R^2 * offset, and camera.py's law var_DN = DN / g + sigma_r^2 / g^2 gives
  g = 1 / (R * scale) e-/DN and sigma_r = sqrt(offset) / scale e-.
"""
from __future__ import annotations

import math
import os
import re
import struct
import zlib
from pathlib import Path

import numpy as np

K_CDM2 = 12.5  # ISO 2720 reflected-light constant, cd/m^2 at EV100 = 0 ... times 100

TAG_NAMES = {
    254: "NewSubfileType", 256: "ImageWidth", 257: "ImageLength", 258: "BitsPerSample", 259: "Compression",
    262: "PhotometricInterpretation", 271: "Make", 272: "Model", 273: "StripOffsets", 274: "Orientation",
    277: "SamplesPerPixel", 278: "RowsPerStrip", 279: "StripByteCounts", 284: "PlanarConfiguration",
    305: "Software", 306: "DateTime", 322: "TileWidth", 323: "TileLength", 324: "TileOffsets",
    325: "TileByteCounts", 330: "SubIFDs", 339: "SampleFormat", 33421: "CFARepeatPatternDim",
    33422: "CFAPattern", 33434: "ExposureTime", 33437: "FNumber", 34665: "ExifIFD", 34855: "ISOSpeedRatings",
    36867: "DateTimeOriginal", 37377: "ShutterSpeedValue", 37378: "ApertureValue", 37379: "BrightnessValue",
    37380: "ExposureBiasValue", 37381: "MaxApertureValue", 37385: "Flash", 37386: "FocalLength",
    50706: "DNGVersion", 50707: "DNGBackwardVersion", 50708: "UniqueCameraModel", 50713: "BlackLevelRepeatDim",
    50714: "BlackLevel", 50717: "WhiteLevel", 50719: "DefaultCropOrigin", 50720: "DefaultCropSize",
    50721: "ColorMatrix1", 50722: "ColorMatrix2", 50727: "AnalogBalance", 50728: "AsShotNeutral",
    50730: "BaselineExposure", 50731: "BaselineNoise", 50732: "BaselineSharpness", 50733: "BayerGreenSplit",
    50734: "LinearResponseLimit", 50739: "ShadowScale", 50778: "CalibrationIlluminant1",
    50779: "CalibrationIlluminant2", 50780: "BestQualityScale", 50829: "ActiveArea", 50964: "ForwardMatrix1",
    50965: "ForwardMatrix2", 51041: "NoiseProfile",
}
_TYPES = {1: (1, "B"), 2: (1, None), 3: (2, "H"), 4: (4, "I"), 5: (8, "II"), 6: (1, "b"), 7: (1, None),
          8: (2, "h"), 9: (4, "i"), 10: (8, "ii"), 11: (4, "f"), 12: (8, "d"), 13: (4, "I")}
MAX_ENTRIES = 4096
EXIF_IN_IFD0 = {"ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength", "BrightnessValue", "DateTimeOriginal",
                "ShutterSpeedValue", "ApertureValue"}


class DngError(ValueError):
    pass


def _entry_value(buf: bytes, bo: str, typ: int, count: int, inline: bytes):
    size, fmt = _TYPES.get(typ, (1, None))
    total = size * count
    if total <= 4:
        data = inline[:total]
    else:
        off = struct.unpack(bo + "I", inline)[0]
        if off + total > len(buf):
            raise DngError(f"tag data out of file (offset {off}, {total} bytes)")
        data = buf[off:off + total]
    if typ == 2:
        return data.split(b"\0", 1)[0].decode("latin-1")
    if typ in (1, 6, 7):
        return data
    vals = struct.unpack(bo + fmt * count, data)
    if typ in (5, 10):
        return [vals[i] / vals[i + 1] if vals[i + 1] else float("nan") for i in range(0, len(vals), 2)]
    return list(vals)


def _read_ifd(buf: bytes, bo: str, off: int) -> tuple[dict, int]:
    if off + 2 > len(buf):
        raise DngError(f"IFD offset {off} out of file")
    n = struct.unpack(bo + "H", buf[off:off + 2])[0]
    if n > MAX_ENTRIES or off + 2 + 12 * n + 4 > len(buf):
        raise DngError(f"IFD at {off} has {n} entries")
    out = {}
    for i in range(n):
        e = buf[off + 2 + 12 * i: off + 14 + 12 * i]
        tag, typ, count = struct.unpack(bo + "HHI", e[:8])
        if typ not in _TYPES or count > 10_000_000:
            continue
        try:
            out[TAG_NAMES.get(tag, tag)] = _entry_value(buf, bo, typ, count, e[8:12])
        except (struct.error, DngError):
            continue
    nxt = struct.unpack(bo + "I", buf[off + 2 + 12 * n: off + 6 + 12 * n])[0]
    return out, nxt


def read_tags(path: str | Path | bytes) -> dict:
    """{'ifd0': {...}, 'exif': {...}, 'subifds': [{...}, ...]} with named tags (unknown tags keep their number).
    Takes a path or the file's bytes (a zip member read without extraction)."""
    buf = bytes(path) if isinstance(path, (bytes, bytearray)) else Path(path).read_bytes()
    if len(buf) < 8 or buf[:2] not in (b"II", b"MM"):
        raise DngError("not a TIFF/DNG file")
    bo = "<" if buf[:2] == b"II" else ">"
    magic, off = struct.unpack(bo + "HI", buf[2:8])
    if magic != 42:
        raise DngError(f"TIFF magic {magic}")
    ifd0, _ = _read_ifd(buf, bo, off)
    exif = {}
    if isinstance(ifd0.get("ExifIFD"), list) and ifd0["ExifIFD"]:
        exif, _ = _read_ifd(buf, bo, int(ifd0["ExifIFD"][0]))
    subs = []
    for o in (ifd0.get("SubIFDs") or [])[:8]:
        try:
            subs.append(_read_ifd(buf, bo, int(o))[0])
        except DngError:
            continue
    return {"ifd0": ifd0, "exif": exif, "subifds": subs, "byte_order": bo, "size": len(buf)}


def _first(x, default=None):
    if isinstance(x, list):
        return x[0] if x else default
    return x if x is not None else default


def exposure_summary(tags: dict) -> dict:
    """Exposure time, ISO, f-number, focal length, EV100 and the implied scene luminance (cd/m^2, K = 12.5)."""
    i0 = tags.get("ifd0", {})
    # Older Android DNGs (Nexus 5/6/6P in HDR+) keep the Exif exposure tags in IFD0 without an Exif IFD
    ex = {k: v for k, v in i0.items() if isinstance(k, str) and k in EXIF_IN_IFD0}
    ex.update(tags.get("exif", {}))
    t = _first(ex.get("ExposureTime"))
    iso = _first(ex.get("ISOSpeedRatings"))
    n = _first(ex.get("FNumber"))
    f = _first(ex.get("FocalLength"))
    bv = _first(ex.get("BrightnessValue"))
    # a zero in any of these is a tag the writer filled without a value (16 HDR+ Nexus 5 frames carry 0 s, ISO 0)
    t, iso, n, f = [None if v is not None and v <= 0 else v for v in (t, iso, n, f)]
    out = {"exposure_s": t, "iso": iso, "f_number": n, "focal_mm": f, "brightness_apex": bv,
           "make": i0.get("Make"), "model": i0.get("Model"), "unique_model": i0.get("UniqueCameraModel"),
           "datetime": ex.get("DateTimeOriginal") or i0.get("DateTime"), "software": i0.get("Software")}
    if t and iso and n:
        ev100 = math.log2(n * n / t) - math.log2(iso / 100.0)
        out["ev100"] = ev100
        out["luminance_cdm2"] = 2 ** ev100 * K_CDM2 / 100.0
    return out


def noise_profile_electrons(profile: list[float], white: float, black: float) -> dict:
    """DNG NoiseProfile (scale, offset pairs) -> gain e-/DN and read noise e- per plane and their means."""
    rng = float(white) - float(black)
    if rng <= 0 or not profile or len(profile) % 2:
        raise DngError("noise profile needs white > black and (scale, offset) pairs")
    planes = []
    for i in range(0, len(profile), 2):
        s, o = float(profile[i]), float(profile[i + 1])
        if s <= 0:
            raise DngError("noise profile scale must be > 0")
        planes.append({"scale": s, "offset": o, "gain_e_per_dn": 1.0 / (rng * s),
                       "read_noise_e": math.sqrt(max(o, 0.0)) / s,
                       "full_well_e": rng / (rng * s)})  # clip level in electrons at this gain
    return {"planes": planes, "gain_e_per_dn": float(np.mean([p["gain_e_per_dn"] for p in planes])),
            "read_noise_e": float(np.mean([p["read_noise_e"] for p in planes])),
            "full_well_e": float(np.mean([p["full_well_e"] for p in planes])), "white": float(white),
            "black": float(black)}


def sensor_summary(tags: dict) -> dict:
    i0 = tags["ifd0"]
    raw = i0
    for s in tags.get("subifds", []):  # some DNGs keep the raw plane in a SubIFD
        if s.get("PhotometricInterpretation") in (32803, 34892) and "WhiteLevel" in s:
            raw = s
            break
    white = _first(raw.get("WhiteLevel"))
    blacks = raw.get("BlackLevel")
    black = float(np.mean(blacks)) if isinstance(blacks, list) and blacks else (float(blacks) if blacks is not None else None)
    out = {"width_px": _first(raw.get("ImageWidth")), "height_px": _first(raw.get("ImageLength")),
           "bits": _first(raw.get("BitsPerSample")), "white": white, "black": black,
           "cfa": list(raw["CFAPattern"]) if isinstance(raw.get("CFAPattern"), (bytes, bytearray)) else None,
           "baseline_exposure": _first(raw.get("BaselineExposure")), "noise": None}
    prof = raw.get("NoiseProfile") or i0.get("NoiseProfile")
    if prof and white and black is not None:
        try:
            out["noise"] = noise_profile_electrons(prof, white, black)
        except DngError:
            out["noise"] = None
    return out


def read_strip_image(path: str | Path) -> np.ndarray | None:
    """Strip TIFF (uncompressed or Deflate) -> (H, W, S) float array, or None if tiled or odd."""
    buf = Path(path).read_bytes()
    tags = read_tags(path)
    i0, bo = tags["ifd0"], tags["byte_order"]
    comp = int(_first(i0.get("Compression"), 1))
    if comp not in (1, 8, 32946) or "StripOffsets" not in i0:  # uncompressed or Deflate strips
        return None
    w, h = int(_first(i0.get("ImageWidth"), 0)), int(_first(i0.get("ImageLength"), 0))
    spp = int(_first(i0.get("SamplesPerPixel"), 1))
    bits = int(_first(i0.get("BitsPerSample"), 8))
    fmt = int(_first(i0.get("SampleFormat"), 1))
    dt = {(1, 8): "u1", (1, 16): "u2", (1, 32): "u4", (2, 8): "i1", (2, 16): "i2", (2, 32): "i4",
          (3, 32): "f4", (3, 64): "f8"}.get((fmt, bits))
    if dt is None or w * h * spp == 0 or w * h * spp > 50_000_000:
        return None
    offs, counts = i0["StripOffsets"], i0.get("StripByteCounts") or [w * h * spp * bits // 8]
    strips = [buf[int(o):int(o) + int(c)] for o, c in zip(offs, counts)]
    if comp != 1:
        try:
            strips = [zlib.decompress(s) for s in strips]
        except zlib.error:
            return None
    data = b"".join(strips)
    need = w * h * spp * (bits // 8)
    if len(data) < need:
        return None
    arr = np.frombuffer(data[:need], dtype=np.dtype(bo + dt)).astype(np.float64)
    if _first(i0.get("PlanarConfiguration"), 1) == 2:
        return arr.reshape(spp, h, w).transpose(1, 2, 0)
    return arr.reshape(h, w, spp)


def lens_shading_summary(path: str | Path) -> dict | None:
    """Lens shading gain map -> per-channel min/max and the corner-to-centre range in stops."""
    arr = read_strip_image(path)
    if arr is None or arr.size == 0 or not np.all(np.isfinite(arr)) or arr.min() <= 0:
        return None
    h, w, s = arr.shape
    centre = arr[h // 2, w // 2, :]
    corners = np.stack([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]])
    per = []
    for c in range(s):
        per.append({"min": float(arr[..., c].min()), "max": float(arr[..., c].max()),
                    "corner_over_centre": float(corners[:, c].mean() / centre[c])})
    rng = float(arr.max() / arr.min())
    return {"shape": [h, w, s], "channels": per, "range_stops": math.log2(rng),
            "corner_falloff_stops": float(np.mean([math.log2(p["corner_over_centre"]) for p in per]))}


# ----------------------------------------------------------------------------------------------------------
# HDR+ first frames

HDRPLUS_FRAME = re.compile(r"payload_N000[.]dng$")
# DNG Model codenames of the HDR+ capture devices -> SIDD camera_full names (noise_check.json keys)
MODEL_TO_SIDD = {"sailfish": "Google Pixel", "marlin": "Google Pixel", "Pixel": "Google Pixel", "Pixel XL": "Google Pixel",
                 "shamu": "Motorola Nexus 6", "Nexus 6": "Motorola Nexus 6"}
MODEL_NAMES = {"sailfish": "Pixel", "marlin": "Pixel XL", "bullhead": "Nexus 5X", "angler": "Nexus 6P",
               "shamu": "Nexus 6", "hammerhead": "Nexus 5", "walleye": "Pixel 2", "taimen": "Pixel 2 XL",
               "Pixel": "Pixel", "Pixel XL": "Pixel XL", "Nexus 5": "Nexus 5", "Nexus 6": "Nexus 6", "Nexus 6P": "Nexus 6P"}


def frame_jobs(raw_dir: Path | None) -> list[Path]:
    if raw_dir is None or not raw_dir.exists():
        return []
    return sorted(p for p in raw_dir.rglob("*.dng") if HDRPLUS_FRAME.search(p.name))


def analyse_frame(path: Path) -> dict:
    row = {"burst": path.parent.name, "file": str(path.relative_to(path.parents[2])) if len(path.parents) > 2 else path.name}
    try:
        tags = read_tags(path)
        row.update(exposure_summary(tags))
        row["sensor"] = sensor_summary(tags)
        shading = path.with_name(path.name.replace("payload_", "lens_shading_map_").replace(".dng", ".tiff"))
        row["lens_shading"] = lens_shading_summary(shading) if shading.exists() else None
        row["ok"] = True  # the sensor block and shading map are read; exposure tags are a separate flag
        row["exposure_tags"] = row.get("exposure_s") is not None and row.get("iso") is not None
        if not row["exposure_tags"]:
            row["note"] = "no exposure tags"
    except (DngError, OSError, struct.error, ValueError) as e:
        row["ok"] = False
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def _pct(vals, q):
    return float(np.percentile(np.asarray(vals, dtype=float), q)) if len(vals) else None


def _dist(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "min": float(min(vals)), "p10": _pct(vals, 10), "median": _pct(vals, 50),
            "p90": _pct(vals, 90), "max": float(max(vals))}


def _loglog_slope(x: list[float], y: list[float]) -> float | None:
    pts = [(a, b) for a, b in zip(x, y) if a and b and a > 0 and b > 0]
    if len(pts) < 3:
        return None
    lx, ly = np.log2([p[0] for p in pts]), np.log2([p[1] for p in pts])
    return float(np.polyfit(lx, ly, 1)[0])


def sidd_gain_at(iso: float, scaling: dict) -> float | None:
    """Gain e-/DN that the SIDD power law (log gain vs log ISO through the measured points) predicts at this ISO."""
    isos, gains = scaling.get("iso") or [], scaling.get("gain_e_per_dn") or []
    if len(isos) < 2 or iso <= 0:
        return None
    lx, ly = np.log2(isos), np.log2(gains)
    slope, icpt = np.polyfit(lx, ly, 1)
    return float(2 ** (slope * math.log2(iso) + icpt))


def run(raw_dir: Path, log=None, noise_check: dict | None = None) -> dict:
    jobs = frame_jobs(raw_dir)
    rows = [analyse_frame(p) for p in jobs]
    if log:
        for r in rows:
            log(f"{r['burst']}: {'ok' if r['ok'] else 'FAILED ' + r.get('error', '')}")
    ok = [r for r in rows if r["ok"]]
    exp = [r["exposure_s"] for r in ok if r.get("exposure_s")]
    iso = [float(r["iso"]) for r in ok if r.get("iso")]
    ev = [r["ev100"] for r in ok if "ev100" in r]
    lum = [r["luminance_cdm2"] for r in ok if "luminance_cdm2" in r]
    gains = [(r["iso"], r["sensor"]["noise"]["gain_e_per_dn"]) for r in ok if r["sensor"].get("noise")]
    reads = [r["sensor"]["noise"]["read_noise_e"] for r in ok if r["sensor"].get("noise")]
    shading = [r["lens_shading"] for r in ok if r.get("lens_shading")]
    by_model: dict[str, dict] = {}
    for r in ok:
        key = MODEL_NAMES.get(r.get("model") or "", r.get("model") or "?")  # codename and marketing name together
        m = by_model.setdefault(key, {"n": 0, "codenames": set(), "exposure_s": [], "iso": [], "gain": [], "read": [],
                                      "shading_stops": []})
        m["n"] += 1
        m["codenames"].add(r.get("model") or "?")
        if r.get("exposure_s"):
            m["exposure_s"].append(r["exposure_s"])
        if r.get("iso"):
            m["iso"].append(float(r["iso"]))
        if r["sensor"].get("noise"):
            m["gain"].append((float(r["iso"]), r["sensor"]["noise"]["gain_e_per_dn"]))
            m["read"].append(r["sensor"]["noise"]["read_noise_e"])
        if r.get("lens_shading"):
            m["shading_stops"].append(r["lens_shading"]["corner_falloff_stops"])
    models = {}
    for name, m in by_model.items():
        g_iso = [g[0] for g in m["gain"]]
        g_val = [g[1] for g in m["gain"]]
        models[name] = {"name": name, "codenames": sorted(m["codenames"]), "n": m["n"],
                        "n_with_exposure": len(m["exposure_s"]), "exposure_s": _dist(m["exposure_s"]),
                        "iso": _dist(m["iso"]), "noise_profile_gain_e_per_dn": _dist(g_val),
                        "noise_profile_read_noise_e": _dist(m["read"]),
                        "noise_profile_log_gain_vs_log_iso_slope": _loglog_slope(g_iso, g_val),
                        "lens_shading_corner_falloff_stops": _dist(m["shading_stops"])}
    # Manufacturer noise profile against the SIDD measurement for the same camera
    cross = {}
    if noise_check:
        scal = noise_check.get("iso_scaling", {})
        for name, m in by_model.items():
            sidd = MODEL_TO_SIDD.get(name)
            if not sidd or sidd not in scal or not m["gain"]:
                continue
            ratios = []
            for i, g in m["gain"]:
                pred = sidd_gain_at(i, scal[sidd])
                if pred:
                    ratios.append(g / pred)
            if ratios:
                cross[name] = {"sidd_camera": sidd, "n": len(ratios), "profile_over_measured_gain": _dist(ratios),
                               "profile_read_noise_e": _dist(m["read"]),
                               "sidd_read_noise_e_min_max": scal[sidd].get("read_noise_e_min_max")}
    # paired per frame, so the two slopes against luminance sum to -1 when the f-number is fixed
    trip = [(r["luminance_cdm2"], r["exposure_s"], float(r["iso"])) for r in ok
            if r.get("luminance_cdm2") and r.get("exposure_s") and r.get("iso")]
    summary = {
        "n_frames": len(rows), "n_ok": len(ok), "n_with_exposure": len(exp), "n_with_iso": len(iso),
        "exposure_s": _dist(exp), "iso": _dist(iso), "ev100": _dist(ev),
        "luminance_cdm2": _dist(lum), "noise_profile_gain_e_per_dn": _dist([g for _, g in gains]),
        "noise_profile_read_noise_e": _dist(reads),
        "lens_shading_corner_falloff_stops": _dist([s["corner_falloff_stops"] for s in shading]),
        "lens_shading_range_stops": _dist([s["range_stops"] for s in shading]),
        "fraction_exposure_ge_1_60": float(np.mean([e >= 1 / 60 * (1 - 1e-4) for e in exp])) if exp else None,
        "fraction_exposure_ge_1_30": float(np.mean([e >= 1 / 30 * (1 - 1e-4) for e in exp])) if exp else None,
        "fraction_exposure_ge_1_15": float(np.mean([e >= 1 / 15 * (1 - 1e-4) for e in exp])) if exp else None,
        "fraction_iso_ge_800": float(np.mean([i >= 800 for i in iso])) if iso else None,
        "log_exposure_vs_log_luminance_slope": _loglog_slope([q[0] for q in trip], [q[1] for q in trip]),
        "log_iso_vs_log_luminance_slope": _loglog_slope([q[0] for q in trip], [q[2] for q in trip]),
        "f_numbers": sorted({r["f_number"] for r in ok if r.get("f_number")}),
    }
    return {"method": METHOD, "summary": summary, "models": models, "sidd_cross_check": cross, "frames": rows}


METHOD = ("First raw frame (payload_N000.dng) of each HDR+ burst read with a struct-only TIFF tag reader: ExposureTime, "
          "ISOSpeedRatings, FNumber, FocalLength from the Exif IFD; EV100 = log2(N^2/t) - log2(ISO/100); scene "
          "luminance 2^EV100 x K/100 cd/m^2 with K = 12.5 (ISO 2720); DNG NoiseProfile (scale, offset) per CFA plane "
          "converted to gain 1/((white-black) x scale) e-/DN and read noise sqrt(offset)/scale e-; the lens shading "
          "gain map (lens_shading_map_N000.tiff) read as an uncompressed strip TIFF, corner falloff = log2 of the mean "
          "corner gain over the centre gain. Cross-check: NoiseProfile gain at the frame's ISO over the gain the SIDD "
          "power law (noise_check.json) predicts for the same camera.")
