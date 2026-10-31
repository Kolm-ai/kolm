"""Shift check (DATA_PLAN section 4 step 2): do the artifact gates, fitted and sourced on camera
images, carry over to generated images?

Generated images have no reference, so the pair estimators of `metrics/natural.py` cannot run on
them. The check therefore uses the two no-reference proxies of the same module, `edge_overshoot_nr`
and `banding_nr`, and reads them in three places:

1. the 81 pristine KADID-10k references: the proxies' distribution on good camera images;
2. the KADID sharpening (#24) and quantization (#22) ladders: the proxies next to the pair metrics
   and the DMOS, which says how well each proxy tracks the metric the gate was fitted on;
3. the 2400 official AIGCIQA2023 images (six generators x 100 prompts x 4 images) with their
   quality, authenticity and correspondence MOS: how the proxies are distributed on generated
   content, how far they drift from the pristine distribution, and whether they rank the raters'
   quality judgement.

Everything here is a proxy read against a distribution, never a threshold: the output says how far
generated images sit from camera images on the harness's own artifact scales and whether that
distance means anything to raters. It does not refit a gate on AI ratings.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .collect import fs
from .fit import (N_LEVELS, N_REFS, TYPE_QUANTIZATION, TYPE_SHARPEN, decode_image, iter_parquet_rows,
                  kadid_index, read_kadid_dmos)

GENERATOR_BLOCK = 400          # images per generator in AIGCIQA2023: 100 prompts x 4 images
N_AI = 2400
MOS_FILES = {"quality": "moz1.txt", "authenticity": "moz2.txt", "correspondence": "moz3.txt"}
MIN_PAIR_FOR_RATIO = 0.05      # pair-metric floor below which the NR/pair ratio is noise
MIN_N_RANK = 8


# ----------------------------------------------------------------------------- AIGCIQA2023 files

def read_pic_index(path: Path) -> list[tuple[str, str]]:
    """pic-index.xlsx without openpyxl: (generator, file name) per row, in sheet order. The sheet
    is parsed as the zip of XML it is; only columns A and B are read."""
    with zipfile.ZipFile(fs(path)) as z:
        names = z.namelist()
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            shared = re.findall(r"<t[^>]*>([^<]*)</t>", z.read("xl/sharedStrings.xml").decode("utf-8"))
        sheet = z.read("xl/worksheets/sheet1.xml").decode("utf-8")
    rows = []
    for row in re.findall(r"<row[^>]*>(.*?)</row>", sheet, flags=re.S):
        cells = {}
        for m in re.finditer(r'<c r="([A-Z]+)\d+"(?: s="\d+")?(?: t="(\w+)")?>(?:<v>([^<]*)</v>)?', row):
            col, typ, val = m.groups()
            if typ == "s" and val is not None:
                val = shared[int(val)]
            cells[col] = val
        rows.append((cells.get("A", ""), cells.get("B", "")))
    return rows


def read_mos(path: Path) -> np.ndarray:
    with open(fs(path), encoding="utf-8") as f:
        return np.array([float(line) for line in f if line.strip()], dtype=np.float64)


def load_aigciqa2023(root: Path) -> dict:
    """Image member names in index order, the three MOS vectors, the generator of every image and
    the alignment checks that justify joining them by index."""
    with zipfile.ZipFile(fs(root / "allimg.zip")) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".png")]
    members.sort(key=lambda n: int(Path(n).stem))
    mos = {k: read_mos(root / v) for k, v in MOS_FILES.items()}
    index = read_pic_index(root / "pic-index.xlsx")
    generators = [g for g, _ in index]
    checks: dict = {"n_images": len(members), "n_index_rows": len(index),
                    "n_mos": {k: int(v.size) for k, v in mos.items()}}
    # AIGIQA2023.json lists every image as "<i>.png" with moz1/2/3 on a 0-1 scale: it ties the image
    # index to the MOS line without any ordering assumption.
    with open(fs(root / "AIGIQA2023.json"), encoding="utf-8") as f:
        qa = json.load(f)
    by_img: dict[int, dict] = {}
    prompts: dict[int, str] = {}
    for item in qa:
        try:
            i = int(Path(str(item.get("img", ""))).stem)
        except ValueError:
            continue
        by_img.setdefault(i, item)
        prompts.setdefault(i, str(item.get("prompt", "")))
    diffs = [abs(float(by_img[i]["moz1"]) * 100.0 - mos["quality"][i]) for i in by_img if i < mos["quality"].size]
    checks["json_mos_alignment"] = {"n_compared": len(diffs), "max_abs_diff": float(max(diffs)) if diffs else None,
                                    "ok": bool(diffs) and bool(max(diffs) < 1e-6)}
    # The sheet lists generators in blocks of 400 (100 prompts x 4). If that is the image order, the
    # prompt of image 400*g + k must not depend on g.
    n_blocks = len(members) // GENERATOR_BLOCK
    consistent = all(prompts.get(g * GENERATOR_BLOCK + k) == prompts.get(k)
                     for g in range(n_blocks) for k in range(GENERATOR_BLOCK)
                     if (g * GENERATOR_BLOCK + k) in prompts and k in prompts)
    blocks = [generators[g * GENERATOR_BLOCK] if g * GENERATOR_BLOCK < len(generators) else "?" for g in range(n_blocks)]
    uniform = all(len({generators[i] for i in range(g * GENERATOR_BLOCK, min((g + 1) * GENERATOR_BLOCK, len(generators)))}) == 1
                  for g in range(n_blocks))
    checks["generator_blocks"] = {"order": blocks, "uniform_blocks": uniform, "prompts_repeat_per_block": consistent,
                                  "ok": uniform and consistent}
    checks["ok"] = bool(len(members) == len(index) == N_AI and all(v.size == N_AI for v in mos.values())
                        and checks["json_mos_alignment"]["ok"] and checks["generator_blocks"]["ok"])
    return {"members": members, "mos": mos, "generators": generators, "prompts": [prompts.get(i, "") for i in range(len(members))],
            "checks": checks}


# ----------------------------------------------------------------------------- measurement

def nr_metrics(frame: np.ndarray) -> dict:
    from ..metrics.natural import banding_nr, edge_overshoot_nr
    b = banding_nr(frame)
    return {"halo_nr": float(edge_overshoot_nr(frame)), "band_step": b["step"], "band_step_p95": b["step_p95"],
            "band_area": b["band_area"], "band_edges": b["n_edges"], "band_regions": b["n_regions"]}


def _measure(task: tuple) -> dict:
    """Worker: ("nr", blob) or ("pair", blob, ref_blob). Top-level so it pickles for the pool."""
    kind, blob = task[0], task[1]
    frame = decode_image(blob)
    out = nr_metrics(frame)
    if kind == "pair":
        from ..metrics.natural import banding_pair, edge_overshoot_pair
        ref = decode_image(task[2])
        out["halo_pair"] = float(edge_overshoot_pair(frame, ref))
        out["band_pair"] = float(banding_pair(frame, ref))
    return out


def measure_tasks(tasks: list[tuple], jobs: int = 1, batch: int = 240,
                  log: Callable[[str], None] | None = None) -> list[dict]:
    """Run `_measure` over tasks, `jobs` processes, in batches so blobs are not all held at once."""
    out: list[dict] = []
    if jobs <= 1:
        for i, t in enumerate(tasks):
            out.append(_measure(t))
            if log and (i + 1) % batch == 0:
                log(f"  {i + 1}/{len(tasks)}")
        return out
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for start in range(0, len(tasks), batch):
            out.extend(ex.map(_measure, tasks[start:start + batch], chunksize=4))
            if log:
                log(f"  {min(start + batch, len(tasks))}/{len(tasks)}")
    return out


def measure_ai(root: Path, data: dict, jobs: int = 1, log: Callable[[str], None] | None = None) -> list[dict]:
    rows: list[dict] = []
    with zipfile.ZipFile(fs(root / "allimg.zip")) as z:
        for start in range(0, len(data["members"]), 240):
            names = data["members"][start:start + 240]
            tasks = [("nr", z.read(n)) for n in names]
            rows.extend(measure_tasks(tasks, jobs=jobs, batch=240))
            if log:
                log(f"  ai {min(start + 240, len(names) + start)}/{len(data['members'])}")
    for i, r in enumerate(rows):
        r["index"] = i
        r["generator"] = data["generators"][i] if i < len(data["generators"]) else "?"
        for k, v in data["mos"].items():
            r[f"mos_{k}"] = float(v[i])
    return rows


def measure_kadid(image_files: Iterable[Path], label_files: Iterable[Path], refs: Iterable[int] = range(1, N_REFS + 1),
                  dist_types: tuple[int, ...] = (TYPE_SHARPEN, TYPE_QUANTIZATION), jobs: int = 1,
                  log: Callable[[str], None] | None = None) -> dict:
    """NR proxies on the pristine references; NR proxies + pair metrics + DMOS on the ladders."""
    refs = list(refs)
    wanted: set[int] = set()
    for r in refs:
        wanted.add(kadid_index(r))
        for t in dist_types:
            for lv in range(1, N_LEVELS + 1):
                wanted.add(kadid_index(r, t, lv))
    blobs: dict[int, bytes] = {}
    for gidx, val in iter_parquet_rows(image_files, wanted):
        blobs[gidx] = val["bytes"]
    if log:
        log(f"  kadid rows on disk: {len(blobs)}/{len(wanted)}")
    dmos = read_kadid_dmos(label_files)
    pristine_tasks, pristine_meta = [], []
    ladder_tasks, ladder_meta = [], []
    for r in refs:
        if kadid_index(r) not in blobs:
            continue
        pristine_tasks.append(("nr", blobs[kadid_index(r)]))
        pristine_meta.append({"ref": r})
        for t in dist_types:
            for lv in range(1, N_LEVELS + 1):
                gi = kadid_index(r, t, lv)
                if gi in blobs:
                    ladder_tasks.append(("pair", blobs[gi], blobs[kadid_index(r)]))
                    ladder_meta.append({"ref": r, "type": t, "level": lv, "dmos": dmos.get((r, t, lv))})
    pristine = measure_tasks(pristine_tasks, jobs=jobs, log=log)
    ladders = measure_tasks(ladder_tasks, jobs=jobs, log=log)
    for m, row in zip(pristine_meta, pristine):
        row.update(m)
    for m, row in zip(ladder_meta, ladders):
        row.update(m)
    return {"pristine": pristine, "ladders": ladders, "n_refs": len(pristine)}


# ----------------------------------------------------------------------------- statistics

def spearman(x: Iterable[float], y: Iterable[float]) -> dict:
    """Rank correlation over the pairs where both values are finite; nan below MIN_N_RANK pairs."""
    from scipy.stats import spearmanr
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    ok = np.isfinite(xa) & np.isfinite(ya)
    n = int(ok.sum())
    if n < MIN_N_RANK or np.ptp(xa[ok]) == 0 or np.ptp(ya[ok]) == 0:
        return {"rho": float("nan"), "n": n}
    rho = spearmanr(xa[ok], ya[ok]).statistic
    return {"rho": float(rho), "n": n}


def distribution(values: Iterable[float]) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    fin = v[np.isfinite(v)]
    if fin.size == 0:
        return {"n": int(v.size), "n_defined": 0}
    return {"n": int(v.size), "n_defined": int(fin.size), "p50": float(np.median(fin)),
            "p95": float(np.percentile(fin, 95)), "max": float(fin.max()),
            "share_positive": float((fin > 0).mean())}


def fraction_above(values: Iterable[float], bound: float | None) -> float | None:
    if bound is None or not np.isfinite(bound):
        return None
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    fin = v[np.isfinite(v)]
    return float((fin > bound).mean()) if fin.size else None


def _col(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) if r.get(key) is not None else float("nan") for r in rows]


def ratio_calibration(rows: list[dict], nr_key: str, pair_key: str, floor: float = MIN_PAIR_FOR_RATIO) -> dict:
    """How the NR proxy reads relative to the pair metric where the pair metric is clearly > 0."""
    nr = np.asarray(_col(rows, nr_key)); pair = np.asarray(_col(rows, pair_key))
    ok = np.isfinite(nr) & np.isfinite(pair) & (pair >= floor)
    if ok.sum() < MIN_N_RANK:
        return {"n": int(ok.sum()), "median_ratio": None}
    ratio = nr[ok] / pair[ok]
    return {"n": int(ok.sum()), "median_ratio": float(np.median(ratio)),
            "iqr_ratio": [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))],
            "spearman_nr_vs_pair": spearman(nr[ok], pair[ok])}


def summarise(ai_rows: list[dict], kadid: dict, gates_fitted: dict | None) -> dict:
    pristine = kadid["pristine"]
    sharpen = [r for r in kadid["ladders"] if r.get("type") == TYPE_SHARPEN]
    quant = [r for r in kadid["ladders"] if r.get("type") == TYPE_QUANTIZATION]
    proxies = ("halo_nr", "band_step", "band_area")
    out: dict = {"pristine": {k: distribution(_col(pristine, k)) for k in proxies}}
    out["ladders"] = {
        "high_sharpen_24": {
            "n": len(sharpen),
            "halo_nr_vs_pair": ratio_calibration(sharpen, "halo_nr", "halo_pair"),
            "spearman_dmos": {"halo_nr": spearman(_col(sharpen, "halo_nr"), _col(sharpen, "dmos")),
                              "halo_pair": spearman(_col(sharpen, "halo_pair"), _col(sharpen, "dmos"))},
        },
        "quantization_22": {
            "n": len(quant),
            "band_step_vs_pair": ratio_calibration(quant, "band_step", "band_pair"),
            "spearman_dmos": {"band_step": spearman(_col(quant, "band_step"), _col(quant, "dmos")),
                              "band_area": spearman(_col(quant, "band_area"), _col(quant, "dmos")),
                              "band_pair": spearman(_col(quant, "band_pair"), _col(quant, "dmos"))},
        },
    }
    # the fitted/sourced gates, carried onto the NR scale through the ladder calibration
    gate_equiv: dict = {}
    halo_gate = None
    if gates_fitted:
        g = gates_fitted.get("gates", {}).get("edge_overshoot")
        if g and g.get("support", {}).get("supported"):
            halo_gate = float(g["value"])
    cal = out["ladders"]["high_sharpen_24"]["halo_nr_vs_pair"]
    if halo_gate is not None and cal.get("median_ratio"):
        eq = halo_gate * cal["median_ratio"]
        pr95 = out["pristine"]["halo_nr"].get("p95")
        # an NR equivalent is only usable as a bound when pristine photographs sit below it
        gate_equiv["halo_nr"] = {"pair_gate": halo_gate, "nr_equivalent": eq, "via": "median NR/pair ratio on the #24 ladder",
                                 "pristine_p95": pr95, "pristine_share_above": fraction_above(_col(kadid["pristine"], "halo_nr"), eq),
                                 "usable": bool(pr95 is not None and eq > pr95)}
    gate_equiv["band_step"] = {"sourced_gate": 0.01, "note": "Weber scale shared with banding_pair; an 8-bit "
                               "one-code step reads 0.013-0.03, so every undithered 8-bit gradient sits above it"}
    out["gate_equivalents"] = gate_equiv
    # generated images
    ai: dict = {"n": len(ai_rows), "distribution": {k: distribution(_col(ai_rows, k)) for k in proxies}}
    p95 = {k: out["pristine"][k].get("p95") for k in proxies}
    ai["share_above_pristine_p95"] = {k: fraction_above(_col(ai_rows, k), p95[k]) for k in proxies}
    if "halo_nr" in gate_equiv:
        ai["share_above_halo_gate_equivalent"] = fraction_above(_col(ai_rows, "halo_nr"), gate_equiv["halo_nr"]["nr_equivalent"])
    ai["spearman_mos"] = {k: {m: spearman(_col(ai_rows, k), _col(ai_rows, f"mos_{m}")) for m in MOS_FILES}
                          for k in proxies}
    per_gen: dict = {}
    for gname in sorted({r["generator"] for r in ai_rows}, key=lambda g: min(r["index"] for r in ai_rows if r["generator"] == g)):
        rows = [r for r in ai_rows if r["generator"] == gname]
        per_gen[gname] = {"n": len(rows),
                          "p50": {k: distribution(_col(rows, k)).get("p50") for k in proxies},
                          "share_above_pristine_p95": {k: fraction_above(_col(rows, k), p95[k]) for k in proxies},
                          "mos_quality_p50": float(np.median(_col(rows, "mos_quality"))),
                          "spearman_quality": {k: spearman(_col(rows, k), _col(rows, "mos_quality")) for k in proxies}}
    ai["per_generator"] = per_gen
    out["ai"] = ai
    return out


def verdict_lines(summary: dict) -> list[str]:
    ai = summary["ai"]; pr = summary["pristine"]; lad = summary["ladders"]
    lines = []
    hs = lad["high_sharpen_24"]
    lines.append(f"halo proxy on the #24 ladder: rank correlation with the pair metric "
                 f"{hs['halo_nr_vs_pair'].get('spearman_nr_vs_pair', {}).get('rho', float('nan')):.2f} "
                 f"(n {hs['halo_nr_vs_pair']['n']}), with DMOS {hs['spearman_dmos']['halo_nr']['rho']:.2f} "
                 f"vs {hs['spearman_dmos']['halo_pair']['rho']:.2f} for the pair metric")
    q = lad["quantization_22"]
    lines.append(f"banding proxy on the #22 ladder: rank correlation with the pair metric "
                 f"{q['band_step_vs_pair'].get('spearman_nr_vs_pair', {}).get('rho', float('nan')):.2f}, "
                 f"with DMOS {q['spearman_dmos']['band_step']['rho']:.2f} (band_area {q['spearman_dmos']['band_area']['rho']:.2f}) "
                 f"vs {q['spearman_dmos']['band_pair']['rho']:.2f} for the pair metric")
    for k in ("halo_nr", "band_step", "band_area"):
        d = ai["distribution"][k]; p = pr[k]
        share = ai["share_above_pristine_p95"][k]
        rho = ai["spearman_mos"][k]["quality"]
        lines.append(f"{k}: generated p50 {d.get('p50', float('nan')):.4f} vs pristine p50 {p.get('p50', float('nan')):.4f}; "
                     f"{(share or 0) * 100:.0f}% of generated images above the pristine 95th percentile "
                     f"{p.get('p95', float('nan')):.4f}; rank correlation with quality MOS {rho['rho']:.2f} (n {rho['n']})")
    if "share_above_halo_gate_equivalent" in ai:
        ge = summary["gate_equivalents"]["halo_nr"]
        share = ai["share_above_halo_gate_equivalent"] * 100
        if ge["usable"]:
            lines.append(f"{share:.1f}% of generated images exceed the NR equivalent "
                         f"({ge['nr_equivalent']:.3f}) of the fitted halo gate {ge['pair_gate']:.3f}")
        else:
            lines.append(f"the NR equivalent of the fitted halo gate ({ge['nr_equivalent']:.3f} for {ge['pair_gate']:.3f}) lies below "
                         f"the pristine 95th percentile ({ge['pristine_p95']:.3f}; {(ge['pristine_share_above'] or 0) * 100:.0f}% of "
                         f"references exceed it), so the gate has no usable no-reference form and its drift is unmeasured; "
                         f"{share:.1f}% of generated images exceed it, which says nothing")
    return lines


def shift_check(ai_root: Path, image_files: Iterable[Path], label_files: Iterable[Path], gates_fitted: dict | None,
                jobs: int = 1, log: Callable[[str], None] | None = None) -> dict:
    data = load_aigciqa2023(ai_root)
    if log:
        log(f"aigciqa2023: {data['checks']}")
    kadid = measure_kadid(list(image_files), list(label_files), jobs=jobs, log=log)
    ai_rows = measure_ai(ai_root, data, jobs=jobs, log=log)
    summary = summarise(ai_rows, kadid, gates_fitted)
    lines = verdict_lines(summary)
    return {
        "data": {
            "generated": "hf-aigciqa2023-images (IntMeGroup/AIGCIQA2023: allimg.zip, moz1-3.txt, pic-index.xlsx, AIGIQA2023.json); "
                         "2400 images, six generators x 100 prompts x 4; MOS 0-100 for quality, authenticity, correspondence",
            "pristine": "hf-kadid10k-images, the 81 reference images",
            "ladders": "KADID-10k #24 High_Sharpen and #22 Quantization, 5 levels each, DMOS from hf-iqa-dataset-labels",
            "checks": data["checks"],
        },
        "method": "no-reference proxies metrics/natural.edge_overshoot_nr and banding_nr on every image; pair metrics "
                  "edge_overshoot_pair / banding_pair on the ladders for calibration; distributions, shares above the "
                  "pristine 95th percentile, Spearman rank correlations (DATA_PLAN.md section 4 step 2)",
        "summary": summary,
        "verdict": lines,
        "ai_rows": ai_rows,
        "kadid_pristine": kadid["pristine"],
        "kadid_ladders": kadid["ladders"],
    }
