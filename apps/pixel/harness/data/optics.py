"""Traced optics (DATA_PLAN.md section 4 step 4): real lens prescriptions through optiland's ray tracer
to get per-field MTF50, Strehl and the residual blur that the Airy-plus-Gaussian camera model needs.

Two prescription sources: the Claff Zemax archive (`lens-designs-claff`, patent examples as .zmx) and the
DeepLens camera lenses (`deeplens-lenses`, JSON). Every lens is refocused on axis (minimum RMS wavefront
error at the primary wavelength) and then measured at that one flat image plane, which is what a sensor
sees. Reported per lens: focal length, f-number, image circle, best-focus shift, centre Strehl, tangential
and sagittal MTF50 at 0, 0.5, 0.7 and 1.0 of the maximum field, RMS OPD per field, the diffraction-limited
MTF50 0.404 / (lambda N) for comparison and the equivalent residual Gaussian sigma in micrometres from
`derive.gaussian_sigma_from_mtf50_um`. Monochromatic at the d line; the F and C lines are loaded so the
glass dispersion is defined but the MTF here is not polychromatic. Failures are recorded, never skipped.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np

WAVELENGTHS_UM = (0.4861, 0.5876, 0.6563)  # F, d (primary), C
PRIMARY_INDEX = 1
FIELD_FRACTIONS = (0.0, 0.5, 0.7, 1.0)
NUM_RAYS = 128
GRID_SIZE = 1024
OPD_RAYS = 64
COARSE_STEPS = 12          # refocus: coarse scan over +-R in 2*COARSE_STEPS steps, then a fine scan
FINE_STEPS = 10
MAX_STREHL_FOR_RESIDUAL = 1.0
RESULT_TIMEOUT_S = 600.0   # no result from any worker for this long -> remaining lenses are recorded as timed out


def scalar(x) -> float:
    """optiland returns 0-d arrays, 1-element arrays and plain floats; always give back a float."""
    return float(np.asarray(x, dtype=float).reshape(-1)[0])


def mtf50_lpmm(freq, mtf) -> float:
    """First 0.5 crossing of an MTF curve with linear interpolation; nan when it never drops below 0.5."""
    freq, mtf = np.asarray(freq, dtype=float), np.asarray(mtf, dtype=float)
    below = np.nonzero(mtf < 0.5)[0]
    if below.size == 0 or below[0] == 0:
        return float("nan")
    i = int(below[0])
    return float(freq[i - 1] + (0.5 - mtf[i - 1]) * (freq[i] - freq[i - 1]) / (mtf[i] - mtf[i - 1]))


def diffraction_mtf50_lpmm(wavelength_um: float, f_number: float) -> float:
    """MTF50 of an aberration-free circular pupil: 0.404 of the cutoff 1 / (lambda N)."""
    return 0.404 / (wavelength_um * 1e-3 * f_number)


def read_text(path: Path) -> str:
    """Zemax files are UTF-16 with a BOM or 8-bit; decide by the first two bytes."""
    b = Path(path).read_bytes()
    return b.decode("utf-16") if b[:2] in (b"\xff\xfe", b"\xfe\xff") else b.decode("latin-1")


def zemax_meta(text: str) -> dict:
    """The few header fields the trace needs: field type (FTYP), the y field list and the title."""
    m = re.search(r"^FTYP\s+(\d+)", text, re.M)
    y = re.search(r"^YFLN\s+(.*)$", text, re.M)
    t = re.search(r"^NAME\s+(.*)$", text, re.M)
    return {"ftyp": int(m.group(1)) if m else None,
            "yfln": [float(v) for v in y.group(1).split()] if y else [],
            "name": t.group(1).strip() if t else ""}


# ----------------------------------------------------------------------------------------------------------
# loaders


def _abbe(spec: str):
    """DeepLens 'nd/vd' -> optiland AbbeMaterial (the same Buchdahl model the Zemax reader uses for GLAS
    ___BLANK rows)."""
    from optiland.materials import AbbeMaterial
    nd, vd = (float(v) for v in spec.split("/"))
    return AbbeMaterial(nd, vd, model="buchdahl")


def build_deeplens(d: dict):
    """DeepLens lens JSON -> optiland Optic. Surfaces carry their own z (`d`) and the gap to the next
    (`d_next`); the last gap reaches the sensor. 'Aperture'/'Stop' rows become the stop plane; 'Aspheric'
    rows map ai = [a2, a4, ...] straight onto optiland's even-asphere coefficients (C_i r^(2i+2))."""
    from optiland.optic import Optic

    o = Optic()
    o.add_surface(index=0, radius=np.inf, thickness=np.inf)
    stop_set = False
    rows = d["surfaces"]
    for i, s in enumerate(rows, start=1):
        kind = s["type"]
        thick = float(s["d_next"])
        mat2 = s.get("mat2", "air")
        material = "air" if mat2 == "air" else _abbe(mat2)
        if kind in ("Aperture", "Stop"):
            o.add_surface(index=i, radius=np.inf, thickness=thick, is_stop=True, material="air")
            stop_set = True
        elif kind == "Aspheric":
            coeffs = [float(v) for v in s.get("ai", [])]
            o.add_surface(index=i, surface_type="even_asphere", radius=float(s["roc"]), thickness=thick,
                          conic=float(s.get("k", 0.0)), coefficients=coeffs, material=material)
        elif kind == "Spheric":
            c = float(s.get("c", 0.0))
            radius = float(s["roc"]) if c != 0.0 and "roc" in s else (1.0 / c if c != 0.0 else np.inf)
            o.add_surface(index=i, radius=radius, thickness=thick, material=material)
        else:
            raise ValueError(f"unsupported DeepLens surface type {kind!r}")
    o.add_surface(index=len(rows) + 1)
    if not stop_set:
        o.surfaces.surfaces[1].is_stop = True
    o.set_aperture("imageFNO", float(d["fnum"]))
    for k, wl in enumerate(WAVELENGTHS_UM):
        o.add_wavelength(wl, is_primary=(k == PRIMARY_INDEX))
    o.fields.set_type("angle")
    max_angle = math.degrees(math.atan(float(d["r_sensor"]) / abs(float(d["foclen"]))))
    for fr in FIELD_FRACTIONS:
        o.add_field(y=fr * max_angle)
    return o, {"field_kind": "angle", "max_field": max_angle, "image_height_mm": float(d["r_sensor"]),
               "field_source": "r_sensor / foclen", "info": d.get("info", "")}


def repair_materials(o) -> int:
    """A GLAS row with Abbe number 0 (a plane filter plate in a few patent examples) gives a nan index in
    optiland's Buchdahl model and a nan paraxial focal length. Such a material is replaced by a constant
    index at the filed nd (surface i + 1 derives its material_pre from surface i's material_post).
    Returns the number of gaps repaired."""
    from optiland.materials import IdealMaterial

    wl = float(o.primary_wavelength)
    surfaces = o.surfaces.surfaces
    count = 0
    for i, s in enumerate(surfaces[:-1]):
        m = s.material_post
        try:
            n = scalar(m.n(wl))
        except Exception:  # noqa: BLE001  (a material that cannot be evaluated at all)
            n = float("nan")
        if math.isfinite(n):
            continue
        nd = scalar(getattr(getattr(m, "model", None), "index", float("nan")))
        if not (math.isfinite(nd) and nd > 0):
            continue
        s.material_post = IdealMaterial(nd)  # surface i + 1 reads its material_pre through this one
        count += 1
    return count


def load_claff(path: Path):
    """Claff .zmx -> optiland Optic with fields normalised. Real-image-height fields (FTYP 3, the bulk of
    the archive) are turned into angles atan(h / f) so that optiland does not ray-aim every field; angle
    fields are kept; object-height fields (finite conjugates) are kept as they are."""
    from optiland.fileio import load_zemax_file

    meta = zemax_meta(read_text(path))
    o = load_zemax_file(str(path))
    repaired = repair_materials(o)
    kind = type(o.fields.field_definition).__name__
    f = abs(scalar(o.paraxial.f2()))
    info = {"zemax_name": meta["name"], "ftyp": meta["ftyp"], "field_kind_loaded": kind, "repaired_glass": repaired}
    ys = [abs(float(fl.y)) for fl in o.fields.fields] or [0.0]
    h = max(ys)
    if kind in ("RealImageHeightField", "ParaxialImageHeightField"):
        max_angle = math.degrees(math.atan(h / f)) if f > 0 else 0.0
        o.fields.set_type("angle")
        o.fields.fields.clear()
        for fr in FIELD_FRACTIONS:
            o.add_field(y=fr * max_angle)
        info.update(field_kind="angle", max_field=max_angle, image_height_mm=h, field_source="atan(YFLN max / f)")
    elif kind == "AngleField":
        o.fields.fields.clear()
        for fr in FIELD_FRACTIONS:
            o.add_field(y=fr * h)
        info.update(field_kind="angle", max_field=h, image_height_mm=f * math.tan(math.radians(h)),
                    field_source="YFLN angles")
    else:  # object height, finite conjugate
        o.fields.fields.clear()
        for fr in FIELD_FRACTIONS:
            o.add_field(y=fr * h)
        info.update(field_kind="object_height", max_field=h, image_height_mm=None, field_source="YFLN object heights")
    return o, info


# ----------------------------------------------------------------------------------------------------------
# analysis


def _rms_opd(o, field, wl) -> float:
    """RMS OPD in waves; nan when the rays for that field do not get through (the caller decides)."""
    import optiland.wavefront as wf
    try:
        v = scalar(wf.OPD(o, field, wl, num_rays=OPD_RAYS).rms())
    except Exception:  # noqa: BLE001 - vignetted or missed field, not a programming error
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _mtf_field(o, fraction: float, max_field: float, wl: float) -> dict:
    """One field: tangential/sagittal MTF50 and RMS OPD; nan values with an error note when it fails."""
    from optiland.mtf import FFTMTF

    row = {"fraction": fraction, "field": fraction * max_field, "mtf50_t_lpmm": float("nan"),
           "mtf50_s_lpmm": float("nan"), "rms_opd_waves": _rms_opd(o, (0.0, fraction), wl)}
    try:
        m = FFTMTF(o, fields=[(0.0, fraction)], wavelength="primary", num_rays=NUM_RAYS, grid_size=GRID_SIZE,
                   max_freq="cutoff")
        freq, mtf = np.asarray(m.freq, dtype=float), np.asarray(m.mtf, dtype=float)
        row["mtf50_t_lpmm"] = mtf50_lpmm(freq[0], mtf[0, 0])
        row["mtf50_s_lpmm"] = mtf50_lpmm(freq[0], mtf[0, 1])
    except Exception as ex:  # noqa: BLE001
        row["error"] = f"{type(ex).__name__}: {str(ex)[:120]}"
    return row


def refocus_on_axis(o, wl: float, focal_mm: float) -> dict:
    """Paraxial focus, then a coarse scan over +-max(0.5 mm, 2 % of f) and a fine scan, minimising the
    on-axis RMS OPD. The shift is applied to the gap before the image surface and returned."""
    last = o.surfaces.num_surfaces - 2
    o.image_solve()
    th0 = float(o.surfaces.surfaces[last].thickness)

    def rms_at(dz):
        o.set_thickness(th0 + dz, last)
        return _rms_opd(o, (0.0, 0.0), wl)

    rms0 = rms_at(0.0)
    if not math.isfinite(rms0):
        raise ValueError("on-axis wavefront cannot be traced at the paraxial focus")
    half = max(0.5, 0.02 * abs(focal_mm))
    step = half / COARSE_STEPS
    best = (rms0, 0.0)
    for dz in np.linspace(-half, half, 2 * COARSE_STEPS + 1):
        r = rms_at(float(dz))
        if r < best[0]:
            best = (r, float(dz))
    fine = step / FINE_STEPS
    for dz in np.linspace(best[1] - step, best[1] + step, 2 * FINE_STEPS + 1):
        r = rms_at(float(dz))
        if r < best[0]:
            best = (r, float(dz))
    o.set_thickness(th0 + best[1], last)
    return {"paraxial_rms_opd_waves": rms0, "best_rms_opd_waves": best[0], "shift_mm": best[1],
            "scan_half_range_mm": half, "hit_scan_edge": abs(best[1]) >= half - fine}


def _min_ts(row: dict) -> float:
    """The weaker of tangential and sagittal MTF50; nan only when both are nan."""
    v = [row["mtf50_t_lpmm"], row["mtf50_s_lpmm"]]
    v = [x for x in v if math.isfinite(x)]
    return min(v) if v else float("nan")


def analyse_optic(o, info: dict) -> dict:
    """Refocus, then Strehl on axis and tangential/sagittal MTF50 per field at the primary wavelength."""
    from optiland.psf import FFTPSF

    from .derive import gaussian_sigma_from_mtf50_um

    f = scalar(o.paraxial.f2())
    fno = scalar(o.paraxial.FNO())
    wl = float(o.primary_wavelength)
    focus = refocus_on_axis(o, wl, f)
    psf = FFTPSF(o, (0.0, 0.0), wl, num_rays=NUM_RAYS, grid_size=GRID_SIZE)
    strehl = scalar(psf.strehl_ratio())
    dl = diffraction_mtf50_lpmm(wl, fno)
    max_field = float(info.get("max_field") or 0.0)
    fields = [_mtf_field(o, fr, max_field, wl) for fr in FIELD_FRACTIONS]
    centre = _min_ts(fields[0])
    edge = _min_ts(fields[-1])
    if not math.isfinite(centre):
        raise ValueError("on-axis MTF could not be traced: " + fields[0].get("error", "MTF never drops below 0.5"))
    residual = gaussian_sigma_from_mtf50_um(centre, fno, wl * 1000.0) if math.isfinite(centre) and centre > 0 else None
    return {
        "focal_mm": f, "f_number": fno, "entrance_pupil_mm": abs(f) / fno if fno else None,
        "wavelength_um": wl, "field_kind": info.get("field_kind"), "max_field": info.get("max_field"),
        "image_height_mm": info.get("image_height_mm"), "n_surfaces": int(o.surfaces.num_surfaces),
        "focus": focus, "strehl_centre": strehl,
        "diffraction_mtf50_lpmm": dl, "mtf50_centre_lpmm": centre, "mtf50_edge_lpmm": edge,
        "mtf50_centre_over_diffraction": centre / dl if math.isfinite(centre) else None,
        "residual_blur_um": residual, "fields": fields,
    }


def trace_file(job: tuple[str, str]) -> dict:
    """Worker entry: (kind, path) -> result dict with ok / error. Never raises."""
    kind, path = job
    t0 = time.time()
    name = Path(path).name
    out = {"name": name, "source": kind, "path": str(path), "ok": False}
    try:
        warnings.simplefilter("ignore")
        if kind in ("claff", "deeplens") and not Path(path).is_file():
            raise FileNotFoundError(path)  # a missing file is a missing file, whether or not optiland is installed
        if kind == "claff":
            o, info = load_claff(Path(path))
            try:
                res = analyse_optic(o, info)
                info["aperture"] = "as filed"
            except ValueError as first:  # marginal rays miss on axis at the filed aperture: stop down
                res, info = _stopped_down_claff(Path(path), first)
        elif kind == "deeplens":
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            o, info = build_deeplens(d)
            try:
                res = analyse_optic(o, info)
                info["aperture"] = "imageFNO"
            except ValueError as first:  # pupil-edge rays of a fast lens may miss at the nominal F/#;
                o, info = build_deeplens(d)  # the nominal entrance pupil f / N is the honest fallback
                o.set_aperture("EPD", abs(float(d["foclen"])) / float(d["fnum"]))
                res = analyse_optic(o, info)
                info["aperture"] = f"EPD after imageFNO failed ({str(first)[:80]})"
        else:
            raise ValueError(f"unknown prescription kind {kind!r}")
        out.update(res)
        out.update({k: v for k, v in info.items() if k not in out})
        out["ok"] = True
    except Exception as ex:  # noqa: BLE001 - every failure is a row in the report
        out["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
    out["seconds"] = round(time.time() - t0, 2)
    return _clean(out)


STOP_DOWN_FRACTIONS = (0.9, 0.8, 0.7, 0.6, 0.5)


def _stopped_down_claff(path: Path, first: Exception):
    """Fast patent examples whose marginal rays miss at the filed aperture are re-traced at a smaller
    entrance pupil, largest first, and the result says so. A lens with no finite paraxial focal length
    or pupil (unresolved glass or surface in the reader) is not retried."""
    o, info = load_claff(path)
    f, epd = scalar(o.paraxial.f2()), scalar(o.paraxial.EPD())
    if not (math.isfinite(f) and math.isfinite(epd) and epd > 0):
        raise ValueError(f"paraxial focal length or pupil not finite (f {f}, EPD {epd}); {str(first)[:100]}")
    last = first
    for frac in STOP_DOWN_FRACTIONS:
        o, info = load_claff(path)
        o.set_aperture("EPD", epd * frac)
        try:
            res = analyse_optic(o, info)
        except ValueError as ex:
            last = ex
            continue
        info["aperture"] = f"EPD {frac:.1f}x filed (filed aperture fails on axis: {str(first)[:80]})"
        info["stopped_down"] = frac
        return res, info
    raise ValueError(f"on-axis trace fails at the filed aperture and down to {STOP_DOWN_FRACTIONS[-1]}x: "
                     f"{str(last)[:120]}")


def _clean(x):
    """JSON-safe: numpy scalars to Python, nan and inf to None, recursively."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _round(x, n: int):
    return None if x is None or not math.isfinite(x) else round(float(x), n)


# ----------------------------------------------------------------------------------------------------------
# batch


def design_jobs(claff_dir: Path | None, deeplens_dir: Path | None) -> list[tuple[str, str]]:
    jobs = []
    if claff_dir and claff_dir.exists():
        seen = set()
        for p in sorted(claff_dir.iterdir()):
            if p.suffix.lower() == ".zmx" and p.name.lower() not in seen:
                seen.add(p.name.lower())
                jobs.append(("claff", str(p)))
    if deeplens_dir and deeplens_dir.exists():
        for p in sorted(deeplens_dir.glob("*.json")):
            jobs.append(("deeplens", str(p)))
    return jobs


def trace_all(jobs: list[tuple[str, str]], workers: int | None = None, log=None) -> list[dict]:
    """Run every job in a process pool; a stretch of RESULT_TIMEOUT_S without any result ends the run and
    records the unfinished lenses as timed out."""
    import multiprocessing as mp

    if not jobs:
        return []
    workers = workers or max(1, min(16, (os.cpu_count() or 2) - 1))
    results, pending = [], {Path(p).name for _, p in jobs}
    if workers == 1 or len(jobs) == 1:
        for j in jobs:
            r = trace_file(j)
            results.append(r)
            pending.discard(r["name"])
            if log:
                log(_line(r))
        return results
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        it = pool.imap_unordered(trace_file, jobs)
        try:
            while pending:
                r = it.next(timeout=RESULT_TIMEOUT_S)
                results.append(r)
                pending.discard(r["name"])
                if log:
                    log(_line(r))
        except mp.TimeoutError:
            pool.terminate()
            for name in sorted(pending):
                results.append({"name": name, "ok": False, "error": f"timeout: no result within {RESULT_TIMEOUT_S:.0f} s"})
        except StopIteration:
            pass
    return results


def _line(r: dict) -> str:
    if not r["ok"]:
        return f"{r['name']}: FAILED {r['error']}"
    return (f"{r['name']}: f {r['focal_mm']:.1f} F/{r['f_number']:.2f} strehl {r['strehl_centre']:.3f} "
            f"mtf50 {r['mtf50_centre_lpmm']:.1f}/{r['diffraction_mtf50_lpmm']:.1f} lp/mm residual "
            f"{r['residual_blur_um'] if r['residual_blur_um'] is None else round(r['residual_blur_um'], 2)} um "
            f"{r['seconds']}s")


def _pct(values: list[float]) -> dict | None:
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "p10": float(np.percentile(v, 10)), "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max())}


def summarise(results: list[dict]) -> dict:
    """Distributions over the lenses that traced, the best by MTF50 and by residual blur (full-frame image
    circle only, so the number can stand in for a 35 mm lens), and f-number bins."""
    ok = [r for r in results if r.get("ok")]
    failed = {r["name"]: r.get("error", "") for r in results if not r.get("ok")}
    n_field_nan = sum(1 for r in ok for f in r["fields"] if f.get("mtf50_t_lpmm") is None)
    full_frame = [r for r in ok if (r.get("image_height_mm") or 0) >= 21.0]
    usable = [r for r in ok if r["residual_blur_um"] is not None and not r["focus"]["hit_scan_edge"]]
    ff_usable = [r for r in full_frame if r["residual_blur_um"] is not None and not r["focus"]["hit_scan_edge"]]

    def top(rows, key, reverse, n=10):
        rows = [r for r in rows if r.get(key) is not None and math.isfinite(r[key])]
        rows = sorted(rows, key=lambda r: r[key], reverse=reverse)[:n]
        return [{"name": r["name"], "source": r["source"], "focal_mm": _round(r["focal_mm"], 2),
                 "f_number": _round(r["f_number"], 2), "image_height_mm": r.get("image_height_mm"),
                 "strehl_centre": _round(r["strehl_centre"], 4), "mtf50_centre_lpmm": _round(r["mtf50_centre_lpmm"], 1),
                 "mtf50_edge_lpmm": _round(r.get("mtf50_edge_lpmm"), 1),
                 "residual_blur_um": _round(r.get("residual_blur_um"), 3),
                 "diffraction_mtf50_lpmm": _round(r["diffraction_mtf50_lpmm"], 1)} for r in rows]

    bins = {}
    for lo, hi in ((0, 1.6), (1.6, 2.2), (2.2, 3.2), (3.2, 4.5), (4.5, 100)):
        rows = [r for r in usable if lo <= r["f_number"] < hi]
        bins[f"F{lo}-{hi}"] = {"n": len(rows), "mtf50_centre_lpmm": _pct([r["mtf50_centre_lpmm"] for r in rows]),
                               "strehl_centre": _pct([r["strehl_centre"] for r in rows]),
                               "residual_blur_um": _pct([r["residual_blur_um"] for r in rows])}
    best_ff = max(ff_usable, key=lambda r: r["mtf50_centre_lpmm"]) if ff_usable else None
    return {
        "n_files": len(results), "n_ok": len(ok), "n_failed": len(failed), "failures": failed,
        "n_full_frame": len(full_frame), "n_hit_scan_edge": sum(1 for r in ok if r["focus"]["hit_scan_edge"]),
        "n_fields_untraceable": n_field_nan, "n_fields_total": sum(len(r["fields"]) for r in ok),
        "n_stopped_down": sum(1 for r in ok if r.get("stopped_down")),
        "strehl_centre": _pct([r["strehl_centre"] for r in usable]),
        "mtf50_centre_lpmm": _pct([r["mtf50_centre_lpmm"] for r in usable]),
        "mtf50_edge_lpmm": _pct([r["mtf50_edge_lpmm"] for r in usable]),
        "mtf50_centre_over_diffraction": _pct([r["mtf50_centre_over_diffraction"] for r in usable]),
        "residual_blur_um": _pct([r["residual_blur_um"] for r in usable]),
        "share_diffraction_limited_centre": (sum(1 for r in usable if r["strehl_centre"] >= 0.8) / len(usable)) if usable else None,
        "by_f_number": bins,
        "best_by_mtf50_centre": top(usable, "mtf50_centre_lpmm", True),
        "best_by_strehl": top(usable, "strehl_centre", True),
        "best_full_frame_by_mtf50": top(ff_usable, "mtf50_centre_lpmm", True),
        "best_full_frame_by_residual": top(ff_usable, "residual_blur_um", False),
        "best_full_frame": {"name": best_ff["name"], "source": best_ff["source"], "focal_mm": best_ff["focal_mm"],
                            "f_number": best_ff["f_number"], "residual_blur_um": best_ff["residual_blur_um"],
                            "mtf50_centre_lpmm": best_ff["mtf50_centre_lpmm"], "strehl_centre": best_ff["strehl_centre"],
                            "mtf50_edge_lpmm": best_ff.get("mtf50_edge_lpmm"), "image_height_mm": best_ff["image_height_mm"],
                            "aperture": best_ff.get("aperture"),
                            "selection": "highest centre MTF50 among full-frame lenses (image height >= 21 mm) whose "
                                         "refocus stayed inside the scan; residual alone would pick a slow lens at "
                                         "the diffraction limit"} if best_ff else None,
    }


METHOD = ("optiland 0.6.x sequential ray trace. Fields: real image heights converted to angles atan(h/f), "
          "measured at 0/0.5/0.7/1.0 of the maximum field. Focus: paraxial image solve, then coarse and fine "
          "scans of the last gap minimising on-axis RMS OPD at the d line (0.5876 um); all fields are read at "
          "that one flat plane. Strehl: FFT PSF on axis, 128x128 pupil rays, 1024 grid. MTF: FFT MTF to the "
          "diffraction cutoff, tangential and sagittal, MTF50 by first 0.5 crossing. Diffraction reference "
          "0.404/(lambda N). residual_blur_um: Gaussian sigma that, in quadrature with the Airy MTF50 at this "
          "f-number, reproduces the centre MTF50 (lens only, no sensor). Monochromatic; no vignetting from "
          "semi-apertures in the DeepLens JSON build; patent examples are design intent, not built lenses.")
