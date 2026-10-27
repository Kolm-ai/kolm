"""Derivers: raw downloads (data/raw/<id>) + checked-in reference tables → data/derived/*.
Each returns a one-line summary string. Registered by source id or by table name."""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np

from . import data_root
from .collect import fs
from .manifest import load_manifest

REFERENCE_DIR = Path(__file__).with_name("reference")


def _raw(source_id: str) -> Path:
    return data_root() / "raw" / source_id


def _derived() -> Path:
    d = data_root() / "derived"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_json(name: str, obj) -> Path:
    """Write data/derived/<name> through fs() so deep HARNESS_DATA_ROOT paths work on Windows."""
    path = _derived() / name
    with open(fs(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def _json_default(o):
    """numpy scalars and arrays that slip into a derived table."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def _read_cie_csv(path: Path) -> np.ndarray:
    rows = []
    with open(fs(path), encoding="utf-8-sig") as f:
        for r in csv.reader(f):
            try:
                vals = [float(x) for x in r if x != ""]
            except ValueError:
                continue  # header or footnote line
            if vals:
                rows.append(vals)
    width = max((len(r) for r in rows), default=0)
    return np.array([r for r in rows if len(r) == width], dtype=np.float64)


def derive_cie() -> str:
    """Every fetched CIE table → one npz (column 0 = wavelength nm) plus a json of licences."""
    raw = _raw("cie-colorimetry")
    tables, meta = {}, {}
    for csv_path in sorted(raw.glob("*.csv")):
        arr = _read_cie_csv(csv_path)
        if arr.size:
            tables[csv_path.stem] = arr
        m = csv_path.with_suffix(".metadata.json")
        if os.path.exists(fs(m)):
            try:
                with open(fs(m), encoding="utf-8") as f:
                    meta[csv_path.stem] = json.load(f)
            except json.JSONDecodeError:
                pass
    if not tables:
        return "no CIE tables on disk (run collect --ids cie-colorimetry)"
    np.savez_compressed(_derived() / "cie_tables.npz", **tables)

    def licence(k):
        rights = (meta.get(k) or {}).get("rightsList") or []
        # four tables (D65, A, CRI, CFI samples) ship no _metadata.json; the collection page states the licence
        return rights[0].get("rightsIdentifier") if rights else "CC BY-SA 4.0 (collection-level, cie.co.at/data-tables)"

    def doi(k):
        return ((meta.get(k) or {}).get("identifier") or {}).get("identifier")

    summary = {k: {"shape": list(v.shape), "wavelength_nm": [float(v[0, 0]), float(v[-1, 0])],
                   "licence": licence(k), "doi": doi(k)} for k, v in tables.items()}
    _write_json("cie_tables.json", summary)
    return f"{len(tables)} tables -> derived/cie_tables.npz"


def derive_polyhaven_index() -> str:
    """Merge the three asset indexes into one catalogue with counts per category and what is on disk."""
    out, n_total = {}, 0
    for sid, typ in (("polyhaven-hdris", "hdris"), ("polyhaven-textures", "textures"), ("polyhaven-models", "models")):
        idx = _raw(sid) / "index.json"
        if not idx.exists():
            continue
        assets = json.loads(idx.read_text(encoding="utf-8"))
        cats: dict[str, int] = {}
        for a in assets.values():
            for c in a.get("categories", []):
                cats[c] = cats.get(c, 0) + 1
        on_disk = sorted(p.name for p in (_raw(sid)).rglob("*") if p.is_file() and p.name != "index.json")
        out[typ] = {"count": len(assets), "categories": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
                    "on_disk": len(on_disk), "ids": sorted(assets)}
        n_total += len(assets)
    acg = {}
    for sid, typ in (("ambientcg-materials", "Material"), ("ambientcg-hdris", "HDRI")):
        idx = _raw(sid) / f"index_{typ}.json"
        if not idx.exists():
            continue
        page = json.loads(idx.read_text(encoding="utf-8"))
        on_disk = sorted(p.name for p in _raw(sid).rglob("*.zip"))
        acg[typ.lower()] = {"count": int(page.get("numberOfResults", len(page.get("foundAssets", [])))),
                            "on_disk": len(on_disk), "source": sid}
        n_total += acg[typ.lower()]["count"]
    if not out and not acg:
        return "no Poly Haven or ambientCG index on disk"
    if acg:
        out["ambientcg"] = acg
    _write_json("polyhaven_catalog.json", out)
    parts = [f"{typ} {v['count']} ({v['on_disk']} on disk)" for typ, v in out.items() if typ != "ambientcg"]
    parts += [f"ambientcg {typ} {v['count']} ({v['on_disk']} on disk)" for typ, v in acg.items()]
    return f"{n_total} CC0 assets catalogued: " + ", ".join(parts) + " -> derived/polyhaven_catalog.json"


# ---- colour references: X-Rite CGATS, camera sensitivities, measured spectra ------------------------
def _cgats_lab(text: str) -> dict[str, list[float]]:
    """SAMPLE_NAME Lab_L Lab_a Lab_b rows between BEGIN_DATA/END_DATA; X-Rite files use decimal commas."""
    out, inside = {}, False
    for line in text.splitlines():
        s = line.strip()
        if s == "BEGIN_DATA":
            inside = True
            continue
        if s == "END_DATA":
            break
        if inside and s:
            parts = s.replace(",", ".").split()
            if len(parts) >= 4:
                out[parts[0]] = [float(x) for x in parts[1:4]]
    return out


def derive_colorchecker() -> str:
    """X-Rite CGATS zips -> derived/colorchecker.json: nominal Lab per patch for the Classic 24 and SG 140
    charts, before and after the November 2014 formulation change."""
    raw = _raw("zenodo-xrite-colorchecker")
    charts = {}
    for zp in sorted(raw.glob("*.zip")):
        with zipfile.ZipFile(fs(zp)) as z:
            for n in z.namelist():
                if not n.lower().endswith(".txt"):
                    continue
                text = z.read(n).decode("latin-1")
                orig = re.search(r'ORIGINATOR "([^"]*)"', text)
                cond = re.search(r"MeasurementCondition=(\w+)", text)
                charts[zp.stem.lower()] = {
                    "lab": _cgats_lab(text),
                    "originator": orig.group(1) if orig else None,
                    "measurement_condition": cond.group(1) if cond else None,
                    "illuminant_observer": "D50 / 2 deg (X-Rite ColorData-1p_EN.pdf)",
                    "file": zp.name, "source_url": "https://zenodo.org/records/3245895",
                }
    if not charts:
        return "no CGATS zips on disk (run collect --ids zenodo-xrite-colorchecker)"
    _write_json("colorchecker.json", charts)
    sizes = ", ".join(f"{k}:{len(v['lab'])}" for k, v in charts.items())
    return f"{len(charts)} charts ({sizes}) -> derived/colorchecker.json"


def derive_camspec() -> str:
    """Jiang 2013 camera sensitivities -> derived/camspec.json (+ npz): per camera the R, G, B curves at
    400-720 nm / 10 nm and the sensitivity-weighted effective wavelength of each channel, which is what
    the Airy PSF in camera.py needs instead of a guessed (610, 550, 465)."""
    src = _raw("zenodo-camspec-jiang2013") / "camspec_database.txt"
    if not src.exists():
        return "no camspec_database.txt on disk (run collect --ids zenodo-camspec-jiang2013)"
    wl = np.arange(400.0, 721.0, 10.0)
    cams: dict[str, list[np.ndarray]] = {}
    name = None
    with open(fs(src), encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = np.array([float(x) for x in s.split()])
            except ValueError:
                name = s
                cams.setdefault(name, [])
                continue
            if name is not None and row.size == wl.size:
                cams[name].append(row)
    out, arrays, eff = {}, {}, []
    for k, rows in cams.items():
        if len(rows) != 3:
            continue
        rgb = np.stack(rows)
        e = [float((wl * r).sum() / r.sum()) for r in rgb]
        out[k] = {"rgb": rgb.round(6).tolist(), "effective_wavelength_nm": [round(x, 1) for x in e]}
        arrays[k] = rgb
        eff.append(e)
    if not out:
        return "camspec_database.txt parsed to zero cameras"
    typical = [round(float(x), 1) for x in np.median(np.array(eff), axis=0)]
    summary = {"wavelength_nm": wl.tolist(), "cameras": out, "typical_effective_wavelength_nm": typical,
               "licence": "CC BY-NC-SA 4.0", "source_url": "https://zenodo.org/records/3245883",
               "note": "effective wavelength = sum(lambda * S) / sum(S) per channel; typical = median over cameras"}
    _write_json("camspec.json", summary)
    np.savez_compressed(_derived() / "camspec.npz", wavelength_nm=wl, **{k.replace(" ", "_"): v for k, v in arrays.items()})
    return f"{len(out)} cameras -> derived/camspec.json (typical effective R,G,B = {typical} nm)"


def derive_spectra() -> str:
    """Measured reflectance spectra -> derived/spectra.npz (+ json summary): Labsphere SRS-99 white
    standard and the 1269 UEF Munsell matt chips (380-800 nm, 1 nm)."""
    arrays, summary = {}, {}
    lab = _raw("zenodo-labsphere-srs99") / "SRS-99-020.txt"
    if lab.exists():
        rows = []
        with open(fs(lab), encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        rows.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
        if rows:
            a = np.array(rows)
            arrays["labsphere_srs99"] = a
            vis = a[(a[:, 0] >= 400) & (a[:, 0] <= 700), 1]
            summary["labsphere_srs99"] = {"n": len(a), "wavelength_nm": [float(a[0, 0]), float(a[-1, 0])],
                                          "mean_reflectance_400_700": round(float(vis.mean()), 4),
                                          "licence": "CC BY 4.0", "source_url": "https://zenodo.org/records/3245875"}
    mun = _raw("zenodo-munsell-matt") / "munsell380_800_1.asc.gz"
    if mun.exists():
        with gzip.open(fs(mun), "rt") as f:
            vals = np.loadtxt(f)
        n_wl = 421
        if vals.size % n_wl == 0:
            m = vals.reshape(-1, n_wl)  # chip-major per README: first chip = values 1..421
            arrays["munsell_matt"] = m
            arrays["munsell_wavelength_nm"] = np.arange(380.0, 801.0)
            summary["munsell_matt"] = {"chips": int(m.shape[0]), "wavelength_nm": [380, 800], "step_nm": 1,
                                       "licence": "not specified on Zenodo (UEF spectral database)",
                                       "source_url": "https://zenodo.org/records/3269912"}
    if not arrays:
        return "no spectra on disk (collect --ids zenodo-labsphere-srs99 zenodo-munsell-matt)"
    np.savez_compressed(_derived() / "spectra.npz", **arrays)
    _write_json("spectra.json", summary)
    return f"{', '.join(summary)} -> derived/spectra.npz"


# ---- checked-in reference tables (harness/data/reference/*.csv) -> harness presets ----------------
REQUIRED_COLUMNS = {
    "sensors": ("id", "make", "model", "format", "pixel_pitch_um", "width_px", "height_px", "full_well_e",
                "read_noise_e", "bit_depth", "condition", "source_url", "quote"),
    "lenses": ("id", "make", "model", "format", "focal_mm", "f_number", "mtf50_lpmm_center", "condition",
               "source_url", "quote"),
    "displays": ("id", "make", "model", "class", "diagonal_in", "width_px", "height_px", "peak_nits",
                 "distance_m", "condition", "source_url", "quote"),
    "vision_thresholds": ("id", "harness_param", "value", "unit", "condition", "source", "source_url", "quote"),
}


def read_reference(name: str) -> list[dict]:
    """Rows of reference/<name>.csv. Every row must carry a source_url and a quote: no source, no row."""
    path = REFERENCE_DIR / f"{name}.csv"
    if not path.exists():
        return []
    with open(fs(path), encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    missing = set(REQUIRED_COLUMNS[name]) - set(rows[0].keys() if rows else REQUIRED_COLUMNS[name])
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    for r in rows:
        if not r["source_url"].startswith("http") or not r["quote"].strip():
            raise ValueError(f"{path.name} row {r.get('id')}: needs source_url and quote")
    return rows


def _num(x: str, default: float | None = None) -> float | None:
    x = (x or "").strip()
    if x == "":
        return default
    return float(x)


def _fmt(v, spec: str, none: str = "n/a") -> str:
    """format() that survives a None (a fit that could not be made, e.g. unity-gain ISO when gain is flat)."""
    return none if v is None else format(v, spec)


def gaussian_sigma_from_mtf50_um(mtf50_lpmm: float, f_number: float, wavelength_nm: float = 550.0) -> float:
    """Residual (non-diffraction) Gaussian blur sigma in um that, together with the Airy PSF at this
    f-number, gives the measured MTF50. Gaussian: MTF50 at f50 means sigma = sqrt(ln2 / 2pi^2) / f50;
    the Airy MTF50 is ~0.404 of the cutoff 1/(lambda N). Subtracted in quadrature."""
    f50 = mtf50_lpmm / 1000.0  # cycles per um
    k = (math.log(2) / (2 * math.pi ** 2)) ** 0.5  # 0.1874
    sigma_total = k / f50
    f50_airy = 0.404 / (wavelength_nm * 1e-3 * f_number)
    sigma_airy = k / f50_airy
    return float(max(sigma_total ** 2 - sigma_airy ** 2, 0.0) ** 0.5)


# pixel pitch of the measured phones: not in any collected dataset (SIDD and HDR+ DNGs carry no FocalPlaneResolution),
# so hand-set from the makers' and teardown figures; the saturation ISO check in camera.py (99 at the Pixel's ISO 100
# clip) depends on it
PHONE_PIXEL_PITCH_UM = {
    "Google Pixel": (1.55, "hand-set: Google Pixel (2016) tech specs, 12.3 MP rear camera with 1.55 um pixels "
                           "(Sony IMX378)"),
    "iPhone 7": (1.22, "hand-set: Chipworks/TechInsights iPhone 7 teardown, 12 MP Sony rear sensor with 1.22 um pixels"),
}


def _dr_simulated(full_well_e: float, read_noise_e: float, bit_depth: int) -> float:
    """Stops of range camera.py realises: full well over the quadrature sum of read noise and the quantisation
    step's noise, LSB / sqrt(12), with LSB = full_well / (2^bits - 1)."""
    lsb = full_well_e / (2 ** bit_depth - 1)
    return round(math.log2(full_well_e / math.sqrt(max(read_noise_e, 0.0) ** 2 + lsb ** 2 / 12.0)), 2)


def derive_camera_presets() -> str:
    """sensors.csv x lenses.csv -> derived/camera_presets.json. One preset per sensor, paired with the
    sharpest lens listed for its format; `best_case_earth_sourced` is the preset with the most *simulated*
    dynamic range, log2(full_well / sqrt(read_noise^2 + LSB^2 / 12)) with LSB = full_well / (2^bits - 1): the
    range camera.py's linear single-gain model realises at the listed bit depth. The engineering figure
    log2(full_well / read_noise) is reported beside it; a sensor whose quoted range comes from a dual-gain HDR
    readout keeps that quote in its row but is not simulated that way. Claims use "preset": "derived:<id>"."""
    sensors, lenses = read_reference("sensors"), read_reference("lenses")
    if not sensors:
        return "no reference/sensors.csv"
    presets, frontier = {}, {}
    for s in sensors:
        fmt = s["format"]
        cands = [l for l in lenses if l["format"] == fmt and _num(l["mtf50_lpmm_center"])]
        lens = max(cands, key=lambda l: _num(l["mtf50_lpmm_center"])) if cands else None
        if lens:
            focal, fnum = _num(lens["focal_mm"]), _num(lens["f_number"])
            residual = gaussian_sigma_from_mtf50_um(_num(lens["mtf50_lpmm_center"]), fnum)
        else:
            focal, fnum, residual = 50.0, 1.4, 1.0  # harness default when no lens row exists for the format
        fw, rn = _num(s["full_well_e"]), _num(s["read_noise_e"], 1.0)
        bits = int(_num(s["bit_depth"], 14))
        presets[s["id"]] = {
            "aperture_mm": focal / fnum, "focal_mm": focal, "pixel_pitch_um": _num(s["pixel_pitch_um"]),
            "full_well_e": fw, "read_noise_e": rn, "bit_depth": bits,
            "residual_blur_um": round(residual, 3), "bayer": True,
            "dr_stops_engineering": round(math.log2(fw / max(rn, 1e-6)), 2),
            "dr_stops_simulated": _dr_simulated(fw, rn, bits),
            "optics_source": (f"lenses.csv: {lens['make']} {lens['model']}" if lens else
                              "harness default 50 mm f/1.4, residual 1.0 um: no lens row for this format"),
            "sensor": {"make": s["make"], "model": s["model"], "format": fmt, "resolution_px":
                       [int(_num(s["width_px"], 0)), int(_num(s["height_px"], 0))],
                       "condition": s["condition"], "source_url": s["source_url"], "quote": s["quote"]},
            "lens": {k: lens[k] for k in ("make", "model", "focal_mm", "f_number", "mtf50_lpmm_center",
                                           "condition", "source_url", "quote")} if lens else None,
        }
    camspec = _derived() / "camspec.json"
    if camspec.exists():  # sourced per-channel wavelengths for the Airy PSF instead of the hand-set triple
        cs = json.loads(camspec.read_text(encoding="utf-8"))
        for v in presets.values():
            v["wavelengths_nm"] = cs["typical_effective_wavelength_nm"]
            v["wavelengths_source"] = cs["source_url"]
    axes = {"dr_stops_simulated": True, "dr_stops_engineering": True, "full_well_e": True, "read_noise_e": False,
            "pixel_pitch_um": False}
    for axis, higher in axes.items():
        best = max(presets, key=lambda k: presets[k][axis] * (1 if higher else -1))
        frontier[axis] = {"best": best, "value": presets[best][axis]}
    best = frontier["dr_stops_simulated"]["best"]  # the range the harness can realise, not the vendor's HDR quote
    presets["best_case_earth_sourced"] = dict(presets[best], alias_of=best)
    traced = _derived() / "lens_traces.json"
    floor = json.loads(traced.read_text(encoding="utf-8"))["summary"].get("best_full_frame") if traced.exists() else None
    if floor:  # the sharpest traced full-frame prescription, lens only, swapped into the best-DR sensor
        presets["best_case_traced"] = dict(
            presets[best], alias_of=best, focal_mm=abs(floor["focal_mm"]),
            optics_source=f"lens_traces.json: traced prescription {floor['name']} (design intent, lens only)",
            aperture_mm=abs(floor["focal_mm"]) / floor["f_number"], residual_blur_um=round(floor["residual_blur_um"], 3),
            lens={"make": "traced prescription", "model": floor["name"], "focal_mm": round(abs(floor["focal_mm"]), 2),
                  "f_number": round(floor["f_number"], 2), "mtf50_lpmm_center": round(floor["mtf50_centre_lpmm"], 1),
                  "condition": "optiland trace at the d line, on-axis best focus, lens only, no sensor, design intent",
                  "source_url": "data/derived/lens_traces.json",
                  "quote": f"strehl {floor['strehl_centre']:.3f}; aperture {floor.get('aperture')}; "
                           f"source {floor['source']}; {floor['selection']}"})
    noise_path = _derived() / "noise_check.json"
    measured_cameras: dict[str, dict] = {}
    cs_path = _derived() / "capture_stats.json"
    cs_models = json.loads(cs_path.read_text(encoding="utf-8")).get("models", {}) if cs_path.exists() else {}
    if noise_path.exists():
        from .dng import MODEL_TO_SIDD, exposure_summary, read_tags
        import zipfile

        def sidd_exif(zip_name: str | None) -> dict | None:
            # f-number and focal length from the Exif of the first frame of the instance the preset is built from
            zp = next(_raw("hf-sidd-dng-mirror").rglob(zip_name), None) if zip_name else None
            if zp is None:
                return None
            try:
                with zipfile.ZipFile(zp) as z:
                    name = next(n for n in z.namelist() if n.lower().endswith(".dng"))
                    ex = exposure_summary(read_tags(z.read(name)))
            except Exception:
                return None
            return ex if ex.get("f_number") and ex.get("focal_mm") else None
        nz = json.loads(noise_path.read_text(encoding="utf-8"))
        from ..camera import PRESETS
        phone = PRESETS["phone"]
        for cam, scal in nz.get("iso_scaling", {}).items():
            rows = sorted((r for r in nz["instances"] if r.get("ok") and r.get("camera_full") == cam),
                          key=lambda r: r["iso"])
            if not rows:
                continue
            # one ISO for the whole preset: clip level and read noise both at the lowest measured ISO, where the
            # clip is highest; read noise at the other ISO values goes into the sensor block, since camera.py's
            # sigma_r is one number and a phone's read noise in electrons falls with ISO (ADC-limited at base ISO)
            base = rows[0]
            rn = float(base["read_noise_e"])
            by_iso = {str(int(r["iso"])): round(r["read_noise_e"], 2) for r in rows}
            white = base["raw"]["white"]
            slope = scal.get("log_gain_vs_log_iso_slope")
            # heuristic label, not a sourced threshold: a pure analog-gain camera gives gain ~ ISO^-1 (slope -1;
            # the Pixel measures -1.00), a camera that scales digitally gives 0 (the iPhone 7 measures -0.05);
            # -0.7 splits the two about midway
            analog = slope is not None and slope < -0.7
            slug = "measured_" + re.sub(r"[^a-z0-9]+", "_", cam.lower()).strip("_")
            # lens shading from the HDR+ maps of the same camera family (capture_stats): median corner falloff
            fam = sorted(m for m, sidd in MODEL_TO_SIDD.items() if sidd == cam and m in cs_models)
            sh_vals = [cs_models[m]["lens_shading_corner_falloff_stops"]["median"] for m in fam
                       if cs_models[m].get("lens_shading_corner_falloff_stops", {}).get("n")]
            shading = round(float(np.median(sh_vals)), 2) if sh_vals else 0.0
            # row-correlated noise: the ratio R = var(row means) / (var / W) measured on flat patches gives
            # sigma_row = sigma * sqrt((R - 1) / W); applied to the read-noise floor it is a lower bound, since the
            # patches span every brightness and their per-pixel variance exceeds the read noise
            width = int(base["raw"]["shape"][1])
            rr = base.get("row_noise_ratio")
            row_noise = round(rn * math.sqrt(max(float(rr) - 1.0, 0.0) / width), 3) if rr is not None else 0.0
            bits_m = int(math.ceil(math.log2(white + 1)))
            ex = sidd_exif(base.get("zip"))
            if ex:
                focal_m, fnum_m = float(ex["focal_mm"]), float(ex["f_number"])
                optics_note = (f"optics from the Exif of SIDD instance {base['instance']} (FocalLength {focal_m:g} mm, "
                               f"FNumber {fnum_m:g}); residual blur from the phone preset, not measured")
            else:
                focal_m, fnum_m = phone.focal_mm, phone.focal_mm / phone.aperture_mm
                optics_note = "optics from the phone preset (hand-set), not measured"
            pitch, pitch_src = PHONE_PIXEL_PITCH_UM.get(cam, (phone.pixel_pitch_um, "phone preset default, not sourced"))
            caveat = ("" if analog else
                      f"; the gain does not fall with ISO (log slope {_fmt(slope, '+.2f')}), so this camera's DNG "
                      f"values are not raw analog-gain steps and its electron units are relative to the DNG scale, "
                      f"not to the pixel: a floor to beat at this ISO, not a sensor characterisation")
            measured_cameras[slug] = {
                "aperture_mm": round(focal_m / fnum_m, 4), "focal_mm": focal_m, "pixel_pitch_um": pitch,
                "pixel_pitch_source": pitch_src,
                "full_well_e": round(base["full_well_e"], 1), "read_noise_e": round(rn, 2),
                "bit_depth": bits_m, "residual_blur_um": phone.residual_blur_um,
                "bayer": True, "dr_stops_engineering": round(math.log2(base["full_well_e"] / max(rn, 1e-6)), 2),
                "dr_stops_simulated": _dr_simulated(float(base["full_well_e"]), rn, bits_m),
                "row_noise_e": row_noise,
                "row_noise_source": (f"noise_check: row-noise ratio {float(rr):.3f} at ISO {base['iso']} on {width} "
                                     f"columns, applied to the read noise; a lower bound" if rr is not None else
                                     "no row-noise ratio in noise_check: 0 (not measured)"),
                "optics_source": optics_note,
                "iso_scaling_is_analog_gain": analog,
                "shading_corner_stops": shading, "shading_corrected": True,
                "shading_source": (f"capture_stats: HDR+ lens shading maps of {fam}, median corner falloff"
                                   if sh_vals else "no HDR+ shading map for this camera family: 0 (not measured)"),
                "sensor": {"make": "measured", "model": f"{cam} raw bursts (SIDD)", "format": "phone",
                           "resolution_px": [base["raw"]["shape"][1], base["raw"]["shape"][0]],
                           "condition": f"noise_check: clip level (white - black) x gain and read noise at ISO "
                                        f"{base['iso']} (lowest measured ISO); read noise e- at the other ISO values: "
                                        f"{by_iso}; {optics_note}{caveat}",
                           "read_noise_e_by_iso": by_iso,
                           "source_url": "data/derived/noise_check.json",
                           "quote": f"gain {base['gain_e_per_dn']:.3f} e-/DN at ISO {base['iso']}; log gain vs log ISO "
                                    f"slope {_fmt(slope, '.2f')}; instances {scal.get('instances')}"},
                "lens": None}
        presets.update(measured_cameras)
        pixel = next((k for k in measured_cameras if "pixel" in k), None) or next(iter(measured_cameras), None)
        if pixel:  # the historical name stays as an alias of the Pixel (the first and best-characterised camera)
            presets["phone_measured"] = dict(measured_cameras[pixel], alias_of=pixel)
    measured = presets.get("phone_measured")
    out = {"presets": presets, "frontier": frontier, "traced_lens_floor": floor,
           "measured_cameras": {k: {"camera": v["sensor"]["model"], "full_well_e": v["full_well_e"],
                                    "read_noise_e": v["read_noise_e"], "bit_depth": v["bit_depth"],
                                    "dr_stops_engineering": v["dr_stops_engineering"],
                                    "dr_stops_simulated": v["dr_stops_simulated"], "row_noise_e": v["row_noise_e"],
                                    "read_noise_e_by_iso": v["sensor"]["read_noise_e_by_iso"],
                                    "iso_scaling_is_analog_gain": v["iso_scaling_is_analog_gain"],
                                    "shading_corner_stops": v["shading_corner_stops"]}
                                for k, v in measured_cameras.items()},
           "note": "best_case_earth_sourced is the sensor with the most dr_stops_simulated, "
                   "log2(FW / sqrt(read_noise^2 + LSB^2 / 12)): what camera.py's linear single-gain model realises at "
                   "the row's bit depth. dr_stops_engineering (log2 FW / read_noise) is reported beside it; a vendor's "
                   "dual-gain HDR figure (e.g. 90 dB) is not simulated. "
                   "residual_blur_um is the Gaussian sigma left after removing diffraction from the lens MTF50 "
                   "(quadrature); sensors without a lens row for their format get the harness default 50 mm f/1.4 "
                   "(optics_source says which). "
                   "lenses.csv MTF50 values are system measurements (lens + sensor), so their residual also holds "
                   "the sensor; best_case_traced uses the lens-only residual of the sharpest traced full-frame "
                   "prescription (patent design intent, not a built lens)."}
    _write_json("camera_presets.json", out)
    tail = "".join(f"; {k} FW {v['full_well_e']:.0f} e-, read {v['read_noise_e']:.2f} e-, {v['bit_depth']}-bit, "
                   f"DR_sim {v['dr_stops_simulated']:.2f} stops, row noise {v['row_noise_e']:.3f} e-"
                   + ("" if v["iso_scaling_is_analog_gain"] else " (gain flat vs ISO)")
                   + (f", shading {v['shading_corner_stops']:.2f} stops" if v["shading_corner_stops"] else "")
                   for k, v in measured_cameras.items())
    if measured:
        tail += f"; phone_measured = {measured['alias_of']}"
    return (f"{len(sensors)} sensors x {len(lenses)} lenses -> derived/camera_presets.json (best by simulated DR: "
            f"{best}, {presets[best]['dr_stops_simulated']:.2f} stops){tail}")


ITU_DESIGN_DISTANCE_H = {7680: 0.75, 3840: 1.5, 1920: 3.1}  # Rep. ITU-R BT.2246-9 Table 1 (from Rec. BT.1845), 16:9: one pixel per arc-minute, 60 ppd


def derive_displays() -> str:
    """displays.csv -> derived/displays.json: pixels per degree at the sourced viewing distance and at the
    ITU design distance (0.75 H for 7680-wide, 1.5 H for 3840, 3.1 H for 1920). Headset rows carry a
    vendor-stated ppd instead of a geometry and are listed separately."""
    from ..display import VISIBILITY_PPD, DisplaySpec, ppd
    rows = read_reference("displays")
    if not rows:
        return "no reference/displays.csv"
    out, headsets = {}, {}
    for r in rows:
        base = {"make": r["make"], "model": r["model"], "class": r["class"], "peak_nits": _num(r["peak_nits"]),
                "condition": r["condition"], "source_url": r["source_url"], "quote": r["quote"]}
        if r["class"] == "headset" or not r["width_px"].strip() or not r["distance_m"].strip():
            headsets[r["id"]] = dict(base, ppd_stated=_num(r.get("ppd_stated", "")),
                                     width_px=int(_num(r["width_px"], 0)), height_px=int(_num(r["height_px"], 0)))
            continue
        spec = DisplaySpec(name=r["id"], diagonal_in=_num(r["diagonal_in"]), width_px=int(_num(r["width_px"])),
                           height_px=int(_num(r["height_px"])), distance_m=_num(r["distance_m"]))
        d = dict(base, diagonal_in=spec.diagonal_in, width_px=spec.width_px, height_px=spec.height_px,
                 distance_m=spec.distance_m, ppd=round(ppd(spec), 1))
        ratio = ITU_DESIGN_DISTANCE_H.get(spec.width_px)
        if ratio:
            h_m = spec.diagonal_in * 0.0254 * spec.height_px / math.hypot(spec.width_px, spec.height_px)
            design = DisplaySpec(name=r["id"], diagonal_in=spec.diagonal_in, width_px=spec.width_px,
                                 height_px=spec.height_px, distance_m=round(ratio * h_m, 4))
            d["itu_design_distance_m"] = design.distance_m
            d["ppd_at_design_distance"] = round(ppd(design), 1)
        out[r["id"]] = d
    vals = sorted(v["ppd"] for v in out.values())
    stated = sorted(v["ppd_stated"] for v in headsets.values() if v["ppd_stated"])
    everything = sorted(vals + stated)
    summary = {"n": len(out), "n_headsets": len(headsets), "ppd_min": vals[0], "ppd_median": vals[len(vals) // 2],
               "ppd_max": vals[-1], "above_threshold_low": sum(v >= VISIBILITY_PPD[0] for v in everything),
               "above_threshold_high": sum(v >= VISIBILITY_PPD[1] for v in everything),
               "threshold_ppd": list(VISIBILITY_PPD)}
    _write_json("displays.json", {"displays": out, "headsets": headsets, "summary": summary})
    return (f"{len(out)} displays + {len(headsets)} headsets -> derived/displays.json "
            f"(ppd {vals[0]}..{vals[-1]}; {summary['above_threshold_low']} of {len(everything)} reach {VISIBILITY_PPD[0]} ppd)")


def derive_vision_thresholds() -> str:
    """vision_thresholds.csv -> derived/vision_thresholds.json keyed by harness_param, and
    derived/claim_sourced.json: the default claim with every gate value that has a sourced row."""
    from ..claim import default_claim
    rows = read_reference("vision_thresholds")
    if not rows:
        return "no reference/vision_thresholds.csv"
    params: dict[str, list] = {}
    for r in rows:
        params.setdefault(r["harness_param"], []).append(
            {"value": _num(r["value"]), "unit": r["unit"], "condition": r["condition"], "source": r["source"],
             "source_url": r["source_url"], "quote": r["quote"], "id": r["id"]})
    _write_json("vision_thresholds.json", params)
    claim = default_claim(video=True)
    applied = []
    for g in claim["gates"]:
        for c in g["checks"]:
            key = f"gate.{c['metric']}"
            if key in params and "value" in c:
                c["value"] = params[key][0]["value"]
                c["source"] = params[key][0]["source_url"]
                applied.append(key)
    # Gates fitted to human ratings (derive_fit_gates) override both the default and a sourced row.
    fitted_path = _derived() / "gates_fitted.json"
    fitted = json.loads(fitted_path.read_text(encoding="utf-8"))["gates"] if fitted_path.exists() else {}
    for g in claim["gates"]:
        for c in g["checks"]:
            f = fitted.get(c["metric"])
            if f and "value" in c:
                sup = f.get("support", {"supported": True, "failed": []})
                c["fit"] = {"criterion": f["criterion"], "value": round(f["value"], 6), "ci95": f["ci95"],
                            "extrapolated": f["extrapolated"], "n": f["n"], "condition": f["condition"],
                            "supported": sup["supported"], "failed_criteria": sup["failed"],
                            "upper_bound": f.get("upper_bound")}
                if sup["supported"] and not f["extrapolated"]:
                    c["value"] = round(f["value"], 4)
                    c["source"] = "data/derived/gates_fitted.json"
                    applied.append(f"gate.{c['metric']} (fitted)")
                elif sup["supported"]:
                    c["fit"]["note"] = (f"fit supported but extrapolated: the crossing {round(f['value'], 4)} lies "
                                        f"outside the rated metric range, so the value {c['value']} stays the pass "
                                        f"line and the fit is reported only")
                    applied.append(f"gate.{c['metric']} (fitted; extrapolated, reported only)")
                else:
                    c["fit"]["note"] = (f"fit not supported ({', '.join(sup['failed'])}); the sourced value "
                                        f"{c['value']} is kept, the rated data only bound the threshold from above")
                    applied.append(f"gate.{c['metric']} (sourced; fit unsupported)")
    if "display.distance_m" in params:
        claim["display"]["distance_m"] = params["display.distance_m"][0]["value"]
        applied.append("display.distance_m")
    # Sourced hardware where the derivers have produced it: the camera frontier and a sourced monitor
    # replace the invented presets; explicit keys (distance, delivery, exposure) still override.
    cam_path, disp_path = _derived() / "camera_presets.json", _derived() / "displays.json"
    if cam_path.exists():
        presets = json.loads(cam_path.read_text(encoding="utf-8"))["presets"]
        if "best_case_earth_sourced" in presets:
            claim["camera"]["preset"] = "derived:best_case_earth_sourced"
            applied.append("camera.preset")
    if disp_path.exists():
        drows = json.loads(disp_path.read_text(encoding="utf-8"))["displays"]
        monitors = {k: v for k, v in drows.items() if v["class"] == "monitor"}
        if monitors:
            best = max(monitors, key=lambda k: monitors[k]["ppd"])
            for k in ("diagonal_in", "width_px", "height_px"):
                claim["display"].pop(k, None)
            claim["display"] = {"preset": f"derived:{best}", **claim["display"]}
            applied.append("display.preset")
    claim["provenance"] = {"thresholds": "data/derived/vision_thresholds.json",
                           "gates_fitted": "data/derived/gates_fitted.json",
                           "camera_presets": "data/derived/camera_presets.json",
                           "displays": "data/derived/displays.json", "applied": applied}
    _write_json("claim_sourced.json", claim)
    return f"{len(rows)} rows, {len(params)} params -> derived/vision_thresholds.json; {len(applied)} gate values sourced"


def calibrate_pair_estimator() -> dict:
    """Agreement between the scene metric and the natural-image estimator on unsharp-masked truth:
    the record that `edge_overshoot_pair` measures the quantity the gate is written in."""
    from ..metrics import artifacts
    from ..metrics.natural import edge_overshoot_pair, unsharp
    from ..scene import Scene, SceneSpec
    scene = Scene(SceneSpec(width=512, height=288, supersample=3, seed=3))
    truth = scene.render(0.0)
    y0, y1, x0, x1 = scene.regions["slanted_edge"]
    rows = []
    for sigma in (1.0, 1.5):
        for gain in (0.1, 0.25, 0.5):
            sharp = unsharp(truth, gain, sigma)
            rows.append({"sigma_px": sigma, "gain": gain, "scene_metric": artifacts.edge_overshoot(sharp, scene),
                         "pair_whole_frame": edge_overshoot_pair(sharp, truth),
                         "pair_edge_region": edge_overshoot_pair(sharp[y0:y1, x0:x1], truth[y0:y1, x0:x1])})
    dev = lambda key: max(abs(r[key] - r["scene_metric"]) / r["scene_metric"] for r in rows)
    return {"cases": rows, "max_rel_dev_whole_frame": dev("pair_whole_frame"),
            "max_rel_dev_edge_region": dev("pair_edge_region")}


def calibrate_banding_estimator() -> dict:
    """Agreement between the scene metric and the natural-image estimator on sRGB-posterised truth
    (5-8 bits), plus what the estimator reads on things that are not banding (noise, blur,
    sharpening, tone and colour changes): the record that `banding_pair` measures the quantity the
    gate is written in, and how far each confound moves it."""
    from scipy.ndimage import gaussian_filter
    from ..metrics import artifacts
    from ..metrics.natural import banding_pair, quantize_srgb, unsharp
    from ..scene import Scene, SceneSpec
    scene = Scene(SceneSpec(width=512, height=288, supersample=3, seed=3))
    truth = scene.render(0.0)
    y0, y1, x0, x1 = scene.regions["gradient"]
    rows = []
    for bits in (5, 6, 7, 8):
        q = quantize_srgb(truth, bits)
        rows.append({"srgb_bits": bits, "scene_metric": artifacts.banding_score(q, scene),
                     "pair_whole_frame": banding_pair(q, truth),
                     "pair_gradient_region": banding_pair(q[y0:y1, x0:x1], truth[y0:y1, x0:x1])})
    dev = lambda key: max(abs(r[key] - r["scene_metric"]) / r["scene_metric"] for r in rows)
    rng = np.random.default_rng(0)
    lum = truth @ np.array([0.2126, 0.7152, 0.0722])
    confounds = {
        "identical": banding_pair(truth, truth),
        "awgn_sigma_0.002": banding_pair(np.clip(truth + rng.normal(0.0, 0.002, truth.shape), 0, None), truth),
        "awgn_sigma_0.005": banding_pair(np.clip(truth + rng.normal(0.0, 0.005, truth.shape), 0, None), truth),
        "gblur_sigma_0.7px": banding_pair(gaussian_filter(truth, (0.7, 0.7, 0)), truth),
        "gblur_sigma_1.5px": banding_pair(gaussian_filter(truth, (1.5, 1.5, 0)), truth),
        "unsharp_gain_0.25": banding_pair(unsharp(truth, 0.25, 1.0), truth),
        "gain_x1.2": banding_pair(truth * 1.2, truth),
        "gamma_0.9": banding_pair(truth ** 0.9, truth),
        "gamma_1.1": banding_pair(truth ** 1.1, truth),
        "saturation_x1.3": banding_pair(np.clip(lum[..., None] + 1.3 * (truth - lum[..., None]), 0, None), truth),
    }
    return {"cases": rows, "max_rel_dev_whole_frame": dev("pair_whole_frame"),
            "max_rel_dev_gradient_region": dev("pair_gradient_region"), "confounds_whole_frame": confounds,
            "note": "scene gradient is steep (about 1.2 sRGB codes per row at 512 px wide), so 8-bit "
                    "posterisation already reads as 2-3% steps in its shadows; blur of weak edges is the "
                    "estimator's main confound, noise the scene metric's"}


FITS = {
    "edge_overshoot": ("fit_edge_overshoot_kadid", "calibrate_pair_estimator", "TYPE_SHARPEN"),
    "banding_score": ("fit_banding_kadid", "calibrate_banding_estimator", "TYPE_QUANTIZATION"),
}


def derive_fit_gates() -> str:
    """KADID-10k ladders + DMOS -> derived/gates_fitted.json: gate.edge_overshoot (high-sharpen
    ladder) and gate.banding_score (quantization ladder) at the DCR criterion with a
    reference-bootstrap CI, the per-level ladder, the row-order checks, the specificity probe and
    the estimator calibration; banding also carries the TID2013 quantization-noise rank cross-check
    when its shards are on disk. Needs the lossless KADID shards (3.1 GB) and the label table."""
    from . import fit as F
    images = sorted((_raw("hf-kadid10k-images") / "default" / "train").glob("*.parquet"))
    labels = sorted(_raw("hf-iqa-dataset-labels").rglob("*.parquet"))
    if not images or not labels:
        return ("missing shards; run: python -m harness data collect --ids hf-kadid10k-images "
                "hf-iqa-dataset-labels --jobs 6 --max-bytes 4e9  (3.1 GB)")
    gates, lines = {}, []
    for name, (fit_fn, cal_fn, _) in FITS.items():
        res = getattr(F, fit_fn)(images, labels)
        res["calibration"] = globals()[cal_fn]()
        if name == "banding_score":
            tid = sorted((_raw("hf-tid2013") / "default" / "train").glob("*.parquet"))
            if tid:
                from ..metrics.natural import banding_pair
                res["crosscheck_tid2013"] = F.crosscheck_tid2013(tid, F.TID_TYPE_QUANTIZATION, banding_pair, name)
        gates[name] = res
        alt = res["fit"]["at"][str(F.TARGETS[1])]
        oc = res["order_checks"]
        sup = res["support"]
        ub = res.get("upper_bound")
        verdict = ("supported" if sup["supported"]
                   else f"NOT supported ({', '.join(sup['failed'])}); sourced value kept"
                        + (f"; rated data bound it below {ub['metric_median']:.3f}" if ub else ""))
        lines.append(f"gate.{name} = {res['value']:.3g} (95% CI {res['ci95'][0]:.3g}-{res['ci95'][1]:.3g}) "
                     f"at DCR {F.TARGETS[0]}{' [extrapolated beyond the rated range]' if res['extrapolated'] else ''}; "
                     f"at DCR {F.TARGETS[1]}: {alt['value']:.3g}; {res['n']} rated images, {res['n_refs']} references; "
                     f"order checks {oc['sharpen_monotone']}/{oc['refs']}; spearman {res['fit']['spearman_dmos_vs_log_metric']:.2f}; {verdict}")
    _write_json("gates_fitted.json", {
        "gates": gates,
        "note": "metric thresholds at the stated DCR criterion, fitted to human ratings; "
                "consumed by derive_vision_thresholds into claim_sourced.json (DATA_PLAN.md section 4 step 1)"})
    return "; ".join(lines)


def derive_shift_check() -> str:
    """AIGCIQA2023 images + MOS, KADID-10k references and ladders -> derived/shift_check.json: the
    no-reference halo and banding proxies on generated images, read against their distribution on
    pristine camera images and against the pair metrics the gates were fitted on (DATA_PLAN.md
    section 4 step 2). Needs hf-aigciqa2023-images (893 MB), the KADID shards and the label table."""
    from . import shift as S
    ai = _raw("hf-aigciqa2023-images")
    images = sorted((_raw("hf-kadid10k-images") / "default" / "train").glob("*.parquet"))
    labels = sorted(_raw("hf-iqa-dataset-labels").rglob("*.parquet"))
    if not (ai / "allimg.zip").exists():
        return "missing images; run: python -m harness data collect --ids hf-aigciqa2023-images --max-bytes 1e9  (893 MB)"
    if not images or not labels:
        return ("missing shards; run: python -m harness data collect --ids hf-kadid10k-images "
                "hf-iqa-dataset-labels --jobs 6 --max-bytes 4e9  (3.1 GB)")
    fitted = _derived() / "gates_fitted.json"
    gates = json.loads(fitted.read_text(encoding="utf-8")) if fitted.exists() else None
    jobs = max(1, min(8, (os.cpu_count() or 2) - 1))
    log = (lambda m: print(m, flush=True)) if os.environ.get("HARNESS_VERBOSE") else None
    res = S.shift_check(ai, images, labels, gates, jobs=jobs, log=log)
    _write_json("shift_check.json", res)
    checks = res["data"]["checks"]
    return (f"{res['summary']['ai']['n']} generated images, {len(res['kadid_pristine'])} pristine references, "
            f"{len(res['kadid_ladders'])} ladder images; alignment checks {'ok' if checks['ok'] else 'FAILED'}; "
            + "; ".join(res["verdict"]))


def derive_preference_public() -> str:
    """Public pairwise vote sets -> derived/preference_public.json: the Rao–Kupper tie-and-rater
    preference model fitted per set, scored against plain Bradley–Terry on held-out votes, with
    calibration, bootstrap honesty and cross-set ordering checks (DATA_PLAN.md section 4 step 3).
    Uses whichever of the five sets is on disk; names the rest."""
    from . import prefdata as P
    log = (lambda m: print(m, flush=True)) if os.environ.get("HARNESS_VERBOSE") else None
    only = [x for x in os.environ.get("HARNESS_PREF_SETS", "").split(",") if x] or None
    prev_path = _derived() / "preference_public.json"
    previous = json.loads(prev_path.read_text(encoding="utf-8")) if only and prev_path.exists() else None
    res = P.preference_public(_raw, only=only, log=log, previous=previous)
    _write_json("preference_public.json", res)
    head = f"{len(res['sets'])} sets fitted" + (f", missing {', '.join(res['missing_sources'])}" if res["missing_sources"] else "")
    return head + "; " + "; ".join(res["verdict"])


def derive_lens_traces() -> str:
    """Claff .zmx + DeepLens JSON prescriptions -> derived/lens_traces.json: every lens ray-traced in
    optiland, refocused on axis, with centre Strehl, tangential/sagittal MTF50 at four fields, the
    diffraction-limited MTF50 and the equivalent residual Gaussian blur (DATA_PLAN.md section 4 step 4).
    HARNESS_WORKERS sets the pool size, HARNESS_LENS_LIMIT caps the file count for a quick run."""
    from . import optics as O
    log = (lambda m: print(m, flush=True)) if os.environ.get("HARNESS_VERBOSE") else None
    claff, deep = _raw("lens-designs-claff"), _raw("deeplens-lenses")
    jobs = O.prescription_jobs(claff, deep)
    limit = int(os.environ.get("HARNESS_LENS_LIMIT", "0") or 0)
    if limit:
        jobs = jobs[:limit]
    if not jobs:
        return "no prescriptions on disk: collect lens-designs-claff and deeplens-lenses first"
    workers = int(os.environ.get("HARNESS_WORKERS", "0") or 0) or None
    results = sorted(O.trace_all(jobs, workers=workers, log=log), key=lambda r: r["name"])
    summary = O.summarise(results)
    out = {"method": O.METHOD,
           "sources": {k: sum(1 for kind, _ in jobs if kind == v) for k, v in
                       (("lens-designs-claff", "claff"), ("deeplens-lenses", "deeplens"))},
           "summary": summary, "lenses": {r["name"]: r for r in results}}
    _write_json("lens_traces.json", out)
    best = summary["best_full_frame"]
    tail = (f"; sharpest full-frame {best['name']} f {best['focal_mm']:.0f} F/{best['f_number']:.1f}, "
            f"MTF50 {best['mtf50_centre_lpmm']:.0f} lp/mm, residual {best['residual_blur_um']:.2f} um, "
            f"{summary['n_stopped_down']} stopped down") if best else ""
    return (f"{summary['n_ok']}/{summary['n_files']} prescriptions traced, {summary['n_failed']} failed, "
            f"{summary['n_fields_untraceable']}/{summary['n_fields_total']} fields untraceable" + tail)


def derive_noise_check() -> str:
    """SIDD burst zips on disk -> derived/noise_check.json: camera.py's shot + read noise law measured on real
    raw bursts (per-pixel temporal variance against the temporal mean), gain / read noise / clip level per
    instance, ISO scaling per camera, distribution and row-noise checks (DATA_PLAN.md section 4 step 5).
    HARNESS_NOISE_FRAMES frames per burst (default 60), HARNESS_WORKERS processes, HARNESS_VERBOSE=1 to log."""
    from . import noise as NZ
    raw = _raw("hf-sidd-dng-mirror")
    jobs = NZ.instance_jobs(raw)
    if not jobs:
        return ("no SIDD bursts on disk: python -m harness data collect --ids hf-sidd-dng-mirror "
                "--param pattern='data/train/(0063|0130|0064|0108)_DNG' --max-bytes 1e10 (8.5 GB, one camera, four ISOs)")
    n_frames = int(os.environ.get("HARNESS_NOISE_FRAMES", "") or NZ.DEFAULT_FRAMES)
    log = (lambda m: print(m, flush=True)) if os.environ.get("HARNESS_VERBOSE") else None
    out = NZ.run(raw, n_frames=n_frames, log=log)
    _write_json("noise_check.json", out)
    ok = [r for r in out["instances"] if r.get("ok")]
    parts = []
    for cam, s in out["iso_scaling"].items():
        rn = s.get("read_noise_e_min_max") or [float("nan"), float("nan")]
        base = int(np.argmin(s["iso"]))
        lin = out["verdict"].get("per_camera", {}).get(cam, {})
        parts.append(f"{cam}: linear variance {lin.get('linear_variance')} on {len(lin.get('instances_linear', []))}/"
                     f"{lin.get('n_instances')} instances, gain slope {_fmt(s.get('log_gain_vs_log_iso_slope'), '.2f')} over ISO "
                     f"{int(min(s['iso']))}-{int(max(s['iso']))}, read {rn[0]:.1f}-{rn[1]:.1f} e-, "
                     f"clip {s['full_well_e'][base]:.0f} e- at ISO {int(s['iso'][base])} (the ADC ceiling at base ISO, "
                     f"a lower bound on the pixel's full well), unity-gain ISO {_fmt(s.get('unity_gain_iso'), '.0f')}")
    return (f"{len(ok)}/{len(out['instances'])} bursts x {n_frames} frames; linear variance "
            f"{out['verdict']['linear_variance']}; " + "; ".join(parts))


def derive_capture_stats() -> str:
    """HDR+ first frames on disk -> derived/capture_stats.json: the capture constraints C a phone chose in the
    field (exposure time, ISO, scene luminance from the EXIF exposure with K = 12.5 cd/m2, ISO 2720), the
    manufacturer's DNG NoiseProfile in electrons against the SIDD measurement of the same camera, and the lens
    shading falloff (DATA_PLAN.md section 4 step 9). HARNESS_VERBOSE=1 logs one line per frame."""
    from . import dng as DNG
    raw = _raw("hdr-plus-burst")
    if not DNG.frame_jobs(raw):
        return ("no HDR+ first frames on disk: python -m harness data collect --ids hdr-plus-burst "
                "--param 'pattern=(payload|lens_shading_map)_N000[.](dng|tiff)$' --max-bytes 4e9 (153 bursts, 3.3 GB)")
    nz_path = _derived() / "noise_check.json"
    nz = json.loads(nz_path.read_text(encoding="utf-8")) if nz_path.exists() else None
    log = (lambda m: print(m, flush=True)) if os.environ.get("HARNESS_VERBOSE") else None
    out = DNG.run(raw, log=log, noise_check=nz)
    _write_json("capture_stats.json", out)
    s = out["summary"]
    e, i, lum, sh = s["exposure_s"], s["iso"], s["luminance_cdm2"], s["lens_shading_corner_falloff_stops"]
    cross = "; ".join(f"{k}: profile/measured gain {v['profile_over_measured_gain']['median']:.2f} (n {v['n']})"
                      for k, v in out["sidd_cross_check"].items()) or "no SIDD cross-check"
    return (f"{s['n_ok']}/{s['n_frames']} frames; exposure median {1000 * e['median']:.1f} ms "
            f"(p10 {1000 * e['p10']:.1f}, p90 {1000 * e['p90']:.1f}), {100 * s['fraction_exposure_ge_1_30']:.0f} % at "
            f"1/30 s or longer; ISO median {i['median']:.0f} (p90 {i['p90']:.0f}), {100 * s['fraction_iso_ge_800']:.0f} % at "
            f"ISO 800+; scene luminance median {lum['median']:.0f} cd/m2 (p10 {lum['p10']:.1f}, p90 {lum['p90']:.0f}); "
            f"noise profiles on {s['noise_profile_gain_e_per_dn']['n']} frames, {cross}; lens shading corner falloff "
            f"median {sh['median']:.2f} stops (n {sh['n']})")


def derive_hdri_stats() -> str:
    """Poly Haven 1k HDRIs on disk -> derived/hdri_stats.json: solid-angle-weighted luminance dynamic range
    per real environment map, summarised per category and attribute (DATA_PLAN.md section 4 step 7). Gives
    the scene class S a sourced stop count and reads the camera baseline's dynamic range against it.
    HARNESS_WORKERS processes, HARNESS_HDRI_LIMIT caps the file count, HARNESS_VERBOSE=1 to log."""
    from . import hdri as HD
    raw = _raw("polyhaven-hdris")
    if not any(raw.rglob("*.hdr")):
        return "no HDRIs on disk: python -m harness data collect --ids polyhaven-hdris --max-bytes 2e9 (997 maps, 1.6 GB)"
    workers = int(os.environ.get("HARNESS_WORKERS", "0") or 0) or None
    limit = int(os.environ.get("HARNESS_HDRI_LIMIT", "0") or 0)
    log = (lambda m: print(m, flush=True)) if os.environ.get("HARNESS_VERBOSE") else None
    out = HD.run(raw, HD.load_index(raw / "index.json"), workers=workers, log=log, limit=limit)
    _write_json("hdri_stats.json", out)
    a = out["all"]
    parts = [f"{out['n_ok']}/{out['n_files']} maps"]
    if "dr_stops" in a:
        parts.append(f"scene range (p0.1-p99.9) median {a['dr_stops']['median']:.1f} stops "
                     f"(p10 {a['dr_stops']['p10']:.1f}, p90 {a['dr_stops']['p90']:.1f}); with the brightest source "
                     f"{a['dr_stops_full']['median']:.1f} stops; peak {a['peak_over_median']['median']:.1f} stops over the median")
    for key in ("environment=outdoor", "environment=indoor", "light_type=natural", "light_type=artificial"):
        s = out["by_attribute"].get(key)
        if s and "dr_stops" in s:
            parts.append(f"{key}: {s['dr_stops']['median']:.1f} stops (n {s['n']})")
    return "; ".join(parts)


DERIVERS: dict[str, Callable[[], str]] = {
    "cie": derive_cie,
    "polyhaven_index": derive_polyhaven_index,
    "colorchecker": derive_colorchecker,
    "camspec": derive_camspec,
    "spectra": derive_spectra,
    "camera_presets": derive_camera_presets,
    "displays": derive_displays,
    "fit_gates": derive_fit_gates,
    "shift_check": derive_shift_check,
    "lens_traces": derive_lens_traces,
    "noise_check": derive_noise_check,
    "capture_stats": derive_capture_stats,
    "hdri_stats": derive_hdri_stats,
    "preference_public": derive_preference_public,
    "vision_thresholds": derive_vision_thresholds,
}


def run_derivers(ids: list[str] | None = None) -> dict[str, str]:
    wanted = set(ids or [])
    names = []
    for s in load_manifest():
        if s.deriver and (not wanted or s.id in wanted or s.deriver in wanted) and s.deriver not in names:
            names.append(s.deriver)
    for extra in DERIVERS:  # reference-table derivers not tied to a fetched source
        if extra not in names and (not wanted or extra in wanted) and extra not in {s.deriver for s in load_manifest()}:
            names.append(extra)
    report = {}
    for n in names:
        try:
            report[n] = DERIVERS[n]()
        except Exception as ex:  # noqa: BLE001
            report[n] = f"FAILED {type(ex).__name__}: {ex}"
    return report
