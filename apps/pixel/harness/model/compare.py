"""Every run's held-out reports side by side (python -m harness model compare): one table per report directory name
(heldout_known, heldout_real_rawnind_blind, ...), a row per run, the three gate verdicts and the medians that decide
the frontier gates, each the output's over the oracle frontier's at the run's own shadow SNR where the report has a
frontier (heldout.score with frontier.py, every report from 10-29 on), else the output's alone. The public
baselines (baselines.py, <models>/baseline_<name>/) follow the runs, on the blind sets they are scored on, and a
last table holds SIDD validation for every run and baseline scored on it.

Why: a run is judged on a dozen held-out sets, each its own heldout.md; the question each new run answers is which
gaps it closed against the runs before it, which takes them in one view. Report directories a rescore kept aside
(*_pre_*, heldout_before_rescore) are not read."""
from __future__ import annotations

import json
import re
from pathlib import Path

RUN_DIR = "recovered_"   # <models>/recovered_<run>/heldout_*/heldout.json
BASELINE_DIR = "baseline_"   # <models>/baseline_<name>/heldout_*_blind/heldout.json (baselines.py)
SIDD_FILE = "sidd_val/sidd.json"   # beside either (baselines.evaluate_sidd)
SIDD_COLS = (("psnr", "PSNR 8-bit"), ("psnr_float", "PSNR float"), ("ssim", "SSIM 2-D"), ("ssim3d", "SSIM 3-D"),
             ("mtf50_cpp", "mtf50"), ("texture_hf_fidelity", "texture"))
METRICS = ("shadow_snr_db", "mtf50_cpp", "texture_hf_fidelity", "edge_overshoot", "banding_score", "periodicity",
           "shading_rmse", "delta_e_mean", "landmark_error_px")
GATES = ("capture", "camera", "frontier", "tone")


def _run_key(run: str) -> tuple:
    """v9 < v10 < v14a < v14b: the number, then the suffix"""
    m = re.fullmatch(r"v(\d+)(.*)", run)
    return (int(m.group(1)), m.group(2)) if m else (10 ** 9, run)


def _dirs(models: Path, runs: list[str] | None) -> list[tuple[str, Path]]:
    """(run, directory): the runs in version order, then the public baselines by name; with `runs` those named"""
    m = Path(models)
    found = sorted(((d.name[len(RUN_DIR):], d) for d in m.glob(f"{RUN_DIR}*")), key=lambda t: _run_key(t[0]))
    found += sorted((d.name[len(BASELINE_DIR):], d) for d in m.glob(f"{BASELINE_DIR}*"))
    return [(r, d) for r, d in found if d.is_dir() and (not runs or r in runs)]


def collect(models: Path, runs: list[str] | None = None) -> dict[str, dict[str, dict]]:
    """{report name: {run: heldout.json}} over <models>/recovered_<run>/heldout_*/, runs in version order, then
    <models>/baseline_<name>/heldout_*/"""
    out: dict[str, dict[str, dict]] = {}
    for run, d in _dirs(models, runs):
        for rep in sorted(d.glob("heldout_*/heldout.json")):
            name = rep.parent.name
            if "_pre_" in name or name.endswith("_pre1029"):
                continue
            out.setdefault(name, {})[run] = json.loads(rep.read_text(encoding="utf-8"))
    return out


def collect_sidd(models: Path, runs: list[str] | None = None) -> dict[str, dict]:
    """{run: sidd.json} for every run and baseline scored on SIDD validation, in collect's order"""
    return {run: json.loads((d / SIDD_FILE).read_text(encoding="utf-8"))
            for run, d in _dirs(models, runs) if (d / SIDD_FILE).exists()}


def _f(v) -> str:
    return "-" if v is None else f"{v:.3g}"


def _verdict(g: dict | None) -> str:
    if not g:
        return "-"
    o = g["overall"]
    return o.replace("FAIL at ", "fail: ") if o != "PASS" else "PASS"


def cells(rep: dict) -> list[str]:
    """one report's row: step, the gate verdicts, then per metric output / frontier (or the output alone)"""
    gates = rep.get("gates", {})
    fr = (gates.get("frontier") or {}).get("metrics", {})
    row = [str(rep.get("step", "-"))] + [_verdict(gates.get(k)) for k in GATES]
    for m in METRICS:
        out = rep["metrics"].get(m, {}).get("output")
        f = fr.get(m, {}).get("frontier") if m in fr else None
        row.append(_f(out) if f is None or m == "shadow_snr_db" else f"{_f(out)} / {_f(f)}")
    return row


def _who(run: str, rep: dict) -> str:
    return f"{run} (public)" if rep.get("baseline") else run


def to_markdown(table: dict[str, dict[str, dict]], sidd: dict[str, dict] | None = None) -> str:
    lines = ["# Held-out reports by run", "",
             "Per report set, a row per run: its gate verdicts against the capture, the camera's own photo, the "
             "oracle linear frontier and on tone (heldout.tone_gate), and its medians, as output / frontier at the run's own shadow SNR where the "
             "report has a frontier (lower is better but for mtf50 and texture, judged by distance to the truth)."]
    head = "| run | step | " + " | ".join(GATES) + " | " + " | ".join(METRICS) + " |"
    rule = "|---|---:|" + "---|" * len(GATES) + "---:|" * len(METRICS)
    for name in sorted(table):
        lines += ["", f"## {name}", "", head, rule]
        for run, rep in table[name].items():
            lines.append(f"| {_who(run, rep)} | " + " | ".join(cells(rep)) + " |")
    if sidd:
        lines += ["", "## SIDD validation", "",
                  "1,280 sRGB blocks (baselines.py): PSNR and SSIM averaged over the blocks, mtf50 and texture "
                  "fidelity medians on the linear blocks (truth 0.5 and 1). Our runs see it blind and out of domain.",
                  "", "| run | step | " + " | ".join(h for _, h in SIDD_COLS) + " |",
                  "|---|---:|" + "---:|" * len(SIDD_COLS)]
        rows = [("input", "-", next(iter(sidd.values()))["input"])]
        rows += [(_who(r, rep), rep.get("step", "-"), rep["output"]) for r, rep in sidd.items()]
        for who, step, s in rows:
            lines.append(f"| {who} | {step} | " + " | ".join(_f(s.get(k)) for k, _ in SIDD_COLS) + " |")
    return "\n".join(lines) + "\n"
