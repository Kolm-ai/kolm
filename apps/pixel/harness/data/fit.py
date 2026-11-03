"""Fit gate thresholds to human ratings instead of guessing them.

Targets so far: `gate.edge_overshoot` from the KADID-10k high-sharpen ladder (distortion #24) and
`gate.banding_score` from its quantization ladder (#22); each ladder is 5 levels on 81 references
with 30 degradation category ratings per image. Images come from the lossless parquet mirror
(`hf-kadid10k-images`); DMOS comes from the IQA-Dataset label table (`hf-iqa-dataset-labels`).
The mirror carries no filenames, so rows are addressed by the block order (reference, then 25
types x 5 levels) and that order is re-checked on every reference with two ladders that must be
monotone (brighten: mean rises; sharpen: Laplacian energy rises). TID2013's quantization-noise
ladder (#7, MOS 0-9, 25 references) is a cross-check for banding: its rows carry their own ids.

The gate is the metric value at which the fitted mean DCR crosses 4.5: halfway between
"imperceptible" (5) and "perceptible but not annoying" (4), i.e. where the median observer first
sees the artefact. The DCR scale is bounded, so the primary model is a logistic in log(metric)
pinned to [1, 5]; an unbounded log-linear fit is reported as a sensitivity check. Confidence
intervals come from a bootstrap over references (the images are the sampling unit, not the
ratings).
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from ..colorspace import luminance, srgb_to_linear
from .collect import fs

BLOCK = 126                     # one reference + 25 distortion types x 5 levels
N_REFS, N_TYPES, N_LEVELS = 81, 25, 5
TYPE_BRIGHTEN, TYPE_SHARPEN, TYPE_QUANTIZATION = 16, 24, 22
KADID_TYPE_NAMES = {1: "GBLUR", 10: "JPEG", 11: "AWGN", 15: "Denoising", 16: "Brighten",
                    22: "Quantization", 24: "High_Sharpen", 25: "Contrast Dist."}
TID_TYPE_QUANTIZATION = 7       # TID2013 "Quantization_Noise"
DCR_LABELS = {5: "imperceptible", 4: "perceptible but not annoying", 3: "slightly annoying",
              2: "annoying", 1: "very annoying"}
TARGETS = (4.5, 4.0)            # primary criterion first
PairMetric = Callable[[np.ndarray, np.ndarray], float]


def kadid_index(ref: int, dist_type: int = 0, level: int = 0) -> int:
    """Row of I<ref>_<type>_<level>.png in the mirror; type 0 is the reference itself."""
    if not (1 <= ref <= N_REFS):
        raise ValueError(f"ref {ref} out of range")
    base = BLOCK * (ref - 1)
    if dist_type == 0:
        return base
    if not (1 <= dist_type <= N_TYPES and 1 <= level <= N_LEVELS):
        raise ValueError(f"type {dist_type} level {level} out of range")
    return base + N_LEVELS * (dist_type - 1) + level


def iter_parquet_rows(files: Iterable[Path], wanted: set[int], column: str = "image") -> Iterator[tuple[int, dict]]:
    """Yield (global_row, value) for the wanted rows, reading every shard once, in order."""
    import pyarrow.parquet as pq
    g = 0
    for f in sorted(files):
        pf = pq.ParquetFile(fs(f))
        for rg in range(pf.metadata.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            hits = [i for i in range(g, g + n) if i in wanted]
            if hits:
                col = pf.read_row_group(rg, columns=[column]).column(column)
                for i in hits:
                    yield i, col[i - g].as_py()
            g += n


def decode_image(blob: bytes) -> np.ndarray:
    """PNG/BMP bytes -> linear RGB float64 in [0, 1] (the harness's working space)."""
    from PIL import Image
    im = Image.open(io.BytesIO(blob)).convert("RGB")
    return srgb_to_linear(np.asarray(im, dtype=np.float64) / 255.0)


decode_png = decode_image


def laplacian_energy(frame: np.ndarray) -> float:
    g = luminance(frame) if frame.ndim == 3 else np.asarray(frame, dtype=np.float64)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(np.abs(lap).mean())


def _monotone(values: list[float], base: float) -> bool:
    return all(b > a for a, b in zip([base] + values[:-1], values))


def read_kadid_dmos(label_files: Iterable[Path]) -> dict[tuple[int, int, int], float]:
    """(ref, type, level) -> DMOS from the IQA-Dataset text rows
    `KADID-10k/images/I01_24_03.png,High_Sharpen,KADID-10k/images/I01.png,3.21`."""
    import pyarrow.parquet as pq
    pat = re.compile(r"^KADID-10k/images/I(\d+)_(\d+)_(\d+)\.png,([^,]*),[^,]*,([0-9.]+)$")
    out: dict[tuple[int, int, int], float] = {}
    for f in sorted(label_files):
        for row in pq.read_table(fs(f), columns=["text"]).column("text").to_pylist():
            m = pat.match(row or "")
            if m:
                out[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = float(m.group(5))
    return out


# ---- curve fitting ------------------------------------------------------------------------------

def logistic_dcr(x: np.ndarray, x50: float, k: float) -> np.ndarray:
    """Mean DCR as a function of x = log(metric), pinned to the scale's bounds [1, 5]."""
    return 1.0 + 4.0 / (1.0 + np.exp(k * (np.asarray(x) - x50)))


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy.optimize import curve_fit
    p, _ = curve_fit(logistic_dcr, x, y, p0=(float(np.median(x)), 1.5),
                     bounds=([-50.0, 1e-3], [50.0, 50.0]), maxfev=20000)
    return float(p[0]), float(p[1])


def _crossing_logistic(x50: float, k: float, target: float) -> float:
    return float(np.exp(x50 + np.log(4.0 / (target - 1.0) - 1.0) / k))


def _crossing_linear(a: float, b: float, target: float) -> float:
    return float(np.exp((target - a) / b)) if b < 0 else float("nan")


def fit_dcr_curve(metric: np.ndarray, dmos: np.ndarray, group: np.ndarray, targets: tuple[float, ...] = TARGETS,
                  n_boot: int = 1000, seed: int = 0) -> dict:
    """Fit mean-DCR-vs-log(metric) and return the metric value at each target DCR with a
    reference-level bootstrap 95% interval. `group` labels the sampling unit (reference image)."""
    from scipy.stats import spearmanr
    m, y, g = (np.asarray(metric, dtype=np.float64), np.asarray(dmos, dtype=np.float64), np.asarray(group))
    ok = m > 0
    x = np.log(m[ok])
    y, g = y[ok], g[ok]
    x50, k = _fit_logistic(x, y)
    a, b = np.polyfit(x, y, 1)[::-1]
    rng = np.random.default_rng(seed)
    units = np.unique(g)
    boot = {t: [] for t in targets}
    boot_lin = {t: [] for t in targets}
    for _ in range(n_boot):
        pick = rng.choice(units, size=units.size, replace=True)
        idx = np.concatenate([np.flatnonzero(g == u) for u in pick])
        try:
            bx50, bk = _fit_logistic(x[idx], y[idx])
            ba, bb = np.polyfit(x[idx], y[idx], 1)[::-1]
        except Exception:  # a degenerate resample; skip it
            continue
        for t in targets:
            boot[t].append(_crossing_logistic(bx50, bk, t))
            boot_lin[t].append(_crossing_linear(ba, bb, t))
    pred = logistic_dcr(x, x50, k)
    lo_m, hi_m = float(m[ok].min()), float(m[ok].max())
    out = {
        "model": "mean DCR = 1 + 4 / (1 + exp(k * (log m - x50)))",
        "params": {"x50": x50, "k": k, "m50": float(np.exp(x50))},
        "n": int(ok.sum()), "n_zero_metric_dropped": int((~ok).sum()),
        "n_units": int(units.size), "n_boot": len(boot[targets[0]]),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "spearman_dmos_vs_log_metric": float(spearmanr(x, y).statistic),
        "metric_range": [lo_m, hi_m],
        "at": {},
        "loglinear": {"model": "mean DCR = a + b * log m", "params": {"a": float(a), "b": float(b)}, "at": {}},
    }
    for t in targets:
        v = _crossing_logistic(x50, k, t)
        bs = np.asarray(boot[t])
        out["at"][str(t)] = {"value": v, "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs.size else None,
                             "extrapolated": not (lo_m <= v <= hi_m)}
        vl = _crossing_linear(a, b, t)
        bl = np.asarray([z for z in boot_lin[t] if np.isfinite(z)])
        out["loglinear"]["at"][str(t)] = {"value": vl, "ci95": [float(np.percentile(bl, 2.5)), float(np.percentile(bl, 97.5))] if bl.size else None}
    return out


# ---- KADID-10k ladders ----------------------------------------------------------------------------

def kadid_ladder(image_files: Iterable[Path], metric: PairMetric, dist_type: int = TYPE_SHARPEN,
                 refs: Iterable[int] = range(1, N_REFS + 1), log: Callable[[str], None] | None = None) -> dict:
    """Measure `metric(distorted, reference)` on every level of `dist_type` for the given references,
    after checking the block order with the brighten and sharpen ladders of that reference."""
    refs = list(refs)
    rows: dict[int, bytes] = {}
    wanted: set[int] = set()
    for r in refs:
        wanted.add(kadid_index(r))
        for lv in range(1, N_LEVELS + 1):
            for t in {TYPE_BRIGHTEN, TYPE_SHARPEN, dist_type}:
                wanted.add(kadid_index(r, t, lv))
    for gidx, val in iter_parquet_rows(image_files, wanted):
        rows[gidx] = val["bytes"]
    measured: list[dict] = []
    checks = {"brighten_monotone": 0, "sharpen_monotone": 0, "refs": len(refs), "excluded": []}
    for r in refs:
        try:
            ref = decode_image(rows[kadid_index(r)])
            bright = [decode_image(rows[kadid_index(r, TYPE_BRIGHTEN, lv)]) for lv in range(1, N_LEVELS + 1)]
            sharp = [decode_image(rows[kadid_index(r, TYPE_SHARPEN, lv)]) for lv in range(1, N_LEVELS + 1)]
            target = [decode_image(rows[kadid_index(r, dist_type, lv)]) for lv in range(1, N_LEVELS + 1)]
        except KeyError:
            checks["excluded"].append({"ref": r, "why": "rows missing from the shards on disk"})
            continue
        ok_b = _monotone([float(luminance(b).mean()) for b in bright], float(luminance(ref).mean()))
        ok_s = _monotone([laplacian_energy(s) for s in sharp], laplacian_energy(ref))
        checks["brighten_monotone"] += ok_b
        checks["sharpen_monotone"] += ok_s
        if not (ok_b and ok_s):
            checks["excluded"].append({"ref": r, "why": f"order check failed (brighten {ok_b}, sharpen {ok_s})"})
            continue
        for lv, d in enumerate(target, start=1):
            measured.append({"ref": r, "type": dist_type, "level": lv, "metric": metric(d, ref)})
        if log:
            log(f"ref {r:02d}: " + " ".join(f"{d['metric']:.3f}" for d in measured[-N_LEVELS:]))
    return {"measured": measured, "order_checks": checks}


def kadid_sharpen_ladder(image_files, metric, refs=range(1, N_REFS + 1), log=None) -> dict:
    return kadid_ladder(image_files, metric, TYPE_SHARPEN, refs, log)


SPECIFICITY_TYPES = {11: "AWGN", 1: "GBLUR", 10: "JPEG", 22: "Quantization", 24: "High_Sharpen", 16: "Brighten"}


def kadid_specificity(image_files: Iterable[Path], metric: PairMetric, types: dict[int, str] = SPECIFICITY_TYPES,
                      levels: tuple[int, ...] = (1, 3, 5), refs: Iterable[int] = range(1, N_REFS + 1),
                      exclude: int | None = None) -> dict:
    """Median metric on other distortion ladders: how much of a reading is not the targeted artefact.
    Noise and posterisation move edge pixels, so a halo estimator registers them; blur should not."""
    refs = list(refs)
    types = {t: n for t, n in types.items() if t != exclude}
    wanted = {kadid_index(r) for r in refs}
    for r in refs:
        for t in types:
            for lv in levels:
                wanted.add(kadid_index(r, t, lv))
    rows = {i: v["bytes"] for i, v in iter_parquet_rows(image_files, wanted)}
    out: dict[str, dict] = {}
    for t, name in types.items():
        per: dict[int, list[float]] = {lv: [] for lv in levels}
        for r in refs:
            if kadid_index(r) not in rows:
                continue
            ref = decode_image(rows[kadid_index(r)])
            for lv in levels:
                if kadid_index(r, t, lv) in rows:
                    per[lv].append(metric(decode_image(rows[kadid_index(r, t, lv)]), ref))
        out[name] = {"type": t, "median_by_level": {str(lv): float(np.median(v)) for lv, v in per.items() if v}}
    return out


def _ladder_summary(pts: list[dict], score_key: str) -> list[dict]:
    levels = []
    for lv in range(1, N_LEVELS + 1):
        sel = [p for p in pts if p["level"] == lv]
        if not sel:
            continue
        ms = [p["metric"] for p in sel]
        levels.append({"level": lv, "n": len(sel), "metric_median": float(np.median(ms)),
                       "metric_iqr": [float(np.percentile(ms, q)) for q in (25, 75)],
                       f"{score_key}_mean": float(np.mean([p[score_key] for p in sel]))})
    return levels


SUPPORT_MIN_SPEARMAN = -0.5     # DMOS must fall with log(metric) at least this strongly
SUPPORT_MAX_CI_RATIO = 3.0      # upper/lower bootstrap bound at the primary criterion
SUPPORT_MIN_MEASURED = 0.8      # share of rated images on which the estimator reads > 0


def fit_support(fit: dict, target: float = TARGETS[0]) -> dict:
    """Decide whether a fitted crossing is usable as a gate value. Three stated criteria: the rank
    correlation between DMOS and log(metric) is at least SUPPORT_MIN_SPEARMAN in magnitude, the
    bootstrap interval at `target` spans less than SUPPORT_MAX_CI_RATIO (an interval several
    decades wide is not a threshold), and the estimator reads > 0 on at least SUPPORT_MIN_MEASURED
    of the rated images (otherwise the ratings answer a question the metric does not measure).
    Unsupported fits stay in gates_fitted.json for the record but do not override a sourced row."""
    at = fit["at"][str(target)]
    ci = at.get("ci95") or [0.0, float("inf")]
    rho = fit["spearman_dmos_vs_log_metric"]
    measured = fit["n"] / max(fit["n"] + fit.get("n_zero_metric_dropped", 0), 1)
    ratio = (ci[1] / ci[0]) if ci[0] > 0 else float("inf")
    checks = {
        "spearman": {"value": rho, "limit": SUPPORT_MIN_SPEARMAN, "ok": bool(rho <= SUPPORT_MIN_SPEARMAN)},
        "ci_ratio": {"value": ratio, "limit": SUPPORT_MAX_CI_RATIO, "ok": bool(ratio <= SUPPORT_MAX_CI_RATIO)},
        "measured_share": {"value": measured, "limit": SUPPORT_MIN_MEASURED, "ok": bool(measured >= SUPPORT_MIN_MEASURED)},
    }
    reasons = [k for k, v in checks.items() if not v["ok"]]
    return {"supported": not reasons, "checks": checks, "failed": reasons}


def rated_upper_bound(ladder: list[dict], score_key: str = "dmos_mean", target: float = TARGETS[0]) -> dict | None:
    """The mildest rated level whose mean rating already sits below the criterion: its median metric
    is an upper bound on the threshold even when the curve fit is not supported."""
    below = [l for l in ladder if l[score_key] < target]
    if not below:
        return None
    l = min(below, key=lambda z: z["metric_median"])
    return {"metric_median": l["metric_median"], "level": l["level"], score_key: l[score_key],
            "meaning": f"this level is already rated below {target}, so the threshold lies below its median metric"}


def fit_gate_kadid(image_files: Iterable[Path], label_files: Iterable[Path], *, metric_name: str, dist_type: int,
                   metric: PairMetric, estimator: dict, n_boot: int = 1000, refs: Iterable[int] = range(1, N_REFS + 1),
                   log: Callable[[str], None] | None = None, specificity: bool = True) -> dict:
    """Generic KADID-10k fit: measure `metric` on the `dist_type` ladder, join DMOS, fit the DCR curve."""
    refs = list(refs)
    dmos = read_kadid_dmos(label_files)
    if not dmos:
        raise FileNotFoundError("no KADID-10k rows in the label table")
    ladder = kadid_ladder(image_files, metric, dist_type, refs, log)
    pts = [d | {"dmos": dmos[(d["ref"], d["type"], d["level"])]} for d in ladder["measured"]
           if (d["ref"], d["type"], d["level"]) in dmos]
    if len(pts) < 2 * N_LEVELS:
        raise ValueError(f"only {len(pts)} rated, measured images; need more references on disk")
    fit = fit_dcr_curve(np.array([p["metric"] for p in pts]), np.array([p["dmos"] for p in pts]),
                        np.array([p["ref"] for p in pts]), n_boot=n_boot)
    primary = fit["at"][str(TARGETS[0])]
    out = {
        "metric": metric_name,
        "value": primary["value"], "ci95": primary["ci95"], "extrapolated": primary["extrapolated"],
        "criterion": f"fitted mean DCR = {TARGETS[0]} (midpoint of 5 '{DCR_LABELS[5]}' and 4 '{DCR_LABELS[4]}')",
        "n": fit["n"], "n_refs": fit["n_units"],
        "estimator": estimator,
        "data": {"images": "hf-kadid10k-images (myzhao1999/kadid10k parquet, lossless PNG, 512x384)",
                 "ratings": "hf-iqa-dataset-labels (IQA-Dataset-team-IVC/IQA-Dataset: KADID-10k dmos)",
                 "distortion": f"#{dist_type} {KADID_TYPE_NAMES.get(dist_type, '')}, levels 1-{N_LEVELS}"},
        "condition": "KADID-10k crowdsourced DCR study: 512x384 PNG, uncontrolled displays and distances",
        "ladder": _ladder_summary(pts, "dmos"),
        "fit": fit,
        "order_checks": ladder["order_checks"],
        "support": fit_support(fit),
        "upper_bound": rated_upper_bound(out_ladder := _ladder_summary(pts, "dmos")),
    }
    out["ladder"] = out_ladder
    if specificity:
        out["specificity"] = kadid_specificity(image_files, metric, refs=refs, exclude=dist_type)
    return out


def fit_edge_overshoot_kadid(image_files, label_files, n_boot: int = 1000, refs=range(1, N_REFS + 1), log=None) -> dict:
    from ..metrics.natural import HALF_WINDOW, MIN_STEP, edge_overshoot_pair
    return fit_gate_kadid(image_files, label_files, metric_name="edge_overshoot", dist_type=TYPE_SHARPEN,
                          metric=edge_overshoot_pair, n_boot=n_boot, refs=refs, log=log,
                          estimator={"name": "metrics.natural.edge_overshoot_pair", "statistic": "median over edge windows",
                                     "half_window_px": HALF_WINDOW, "min_step": MIN_STEP, "space": "linear luminance"})


def fit_banding_kadid(image_files, label_files, n_boot: int = 1000, refs=range(1, N_REFS + 1), log=None) -> dict:
    from ..metrics.natural import BANDING_BINS, BANDING_GRAD_MAX, BANDING_MIN_PER_BIN, banding_pair
    return fit_gate_kadid(image_files, label_files, metric_name="banding_score", dist_type=TYPE_QUANTIZATION,
                          metric=banding_pair, n_boot=n_boot, refs=refs, log=log,
                          estimator={"name": "metrics.natural.banding_pair",
                                     "statistic": "max over reference-level bins of |sawtooth residual| / L (Weber step contrast)",
                                     "bins_per_srgb_range": BANDING_BINS, "min_pixels_per_bin": BANDING_MIN_PER_BIN,
                                     "smooth_if_gradient_below_per_px": BANDING_GRAD_MAX, "space": "linear luminance"})


# ---- TID2013 cross-check ---------------------------------------------------------------------------

def tid2013_ladder(files: Iterable[Path], dist_type: int, metric: PairMetric,
                   log: Callable[[str], None] | None = None) -> list[dict]:
    """Measure `metric(distorted, reference)` on every TID2013 row of one distortion type.
    Rows carry reference_id / distortion_id / distortion_intensity / mos and both images (BMP)."""
    import pyarrow.parquet as pq
    cols = ["reference", "distorted", "mos", "reference_id", "distortion_id", "distortion_intensity"]
    out: list[dict] = []
    for f in sorted(files):
        pf = pq.ParquetFile(fs(f))
        for rg in range(pf.metadata.num_row_groups):
            ids = pf.read_row_group(rg, columns=["distortion_id"]).column("distortion_id").to_pylist()
            hits = [i for i, t in enumerate(ids) if t == dist_type]
            if not hits:
                continue
            tbl = pf.read_row_group(rg, columns=cols)
            for i in hits:
                row = {c: tbl.column(c)[i].as_py() for c in cols}
                ref = decode_image(row["reference"]["bytes"])
                dist = decode_image(row["distorted"]["bytes"])
                out.append({"ref": int(row["reference_id"]), "type": dist_type, "level": int(row["distortion_intensity"]),
                            "metric": metric(dist, ref), "mos": float(row["mos"])})
                if log:
                    log(f"tid ref {out[-1]['ref']:02d} lv {out[-1]['level']}: {out[-1]['metric']:.4f} mos {out[-1]['mos']:.2f}")
    return out


def crosscheck_tid2013(files: Iterable[Path], dist_type: int, metric: PairMetric, metric_name: str,
                       log: Callable[[str], None] | None = None) -> dict:
    """Rank agreement between the estimator and TID2013 MOS (0-9, pairwise Swiss-system study) on one
    ladder. No gate is derived: the MOS scale has no 'imperceptible' anchor to put a criterion on."""
    from scipy.stats import spearmanr
    pts = tid2013_ladder(files, dist_type, metric, log)
    if len(pts) < 2 * N_LEVELS:
        return {"metric": metric_name, "n": len(pts), "note": "too few TID2013 rows on disk for this ladder"}
    m = np.array([p["metric"] for p in pts]); y = np.array([p["mos"] for p in pts])
    ok = m > 0
    rho = float(spearmanr(np.log(m[ok]), y[ok]).statistic) if ok.sum() > 3 else float("nan")
    a, b = np.polyfit(np.log(m[ok]), y[ok], 1)[::-1] if ok.sum() > 3 else (float("nan"), float("nan"))
    return {"metric": metric_name, "n": len(pts), "n_refs": len({p["ref"] for p in pts}),
            "data": "hf-tid2013 (Jorgvt/TID2013 parquet, BMP 512x384), distortion #%d, MOS 0-9" % dist_type,
            "spearman_mos_vs_log_metric": rho, "loglinear": {"a": float(a), "b": float(b)},
            "ladder": _ladder_summary(pts, "mos")}
