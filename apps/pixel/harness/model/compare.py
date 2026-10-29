"""Every run's held-out reports side by side (python -m harness model compare): one table per report directory name
(heldout_known, heldout_real_rawnind_blind, ...), a row per run, the three gate verdicts and the medians that decide
the frontier gates, each the output's over the oracle frontier's at the run's own shadow SNR where the report has a
frontier (heldout.score with frontier.py, every report from 10-29 on), else the output's alone.

Why: a run is judged on a dozen held-out sets, each its own heldout.md; the question each new run answers is which
gaps it closed against the runs before it, which takes them in one view. Report directories a rescore kept aside
(*_pre_*, heldout_before_rescore) are not read."""
from __future__ import annotations

import json
import re
from pathlib import Path

RUN_DIR = "recovered_"   # <models>/recovered_<run>/heldout_*/heldout.json
METRICS = ("shadow_snr_db", "mtf50_cpp", "texture_hf_fidelity", "edge_overshoot", "banding_score", "periodicity",
           "shading_rmse", "delta_e_mean", "landmark_error_px")
GATES = ("capture", "camera", "frontier", "tone")


def _run_key(run: str) -> tuple:
    """v9 < v10 < v14a < v14b: the number, then the suffix"""
    m = re.fullmatch(r"v(\d+)(.*)", run)
    return (int(m.group(1)), m.group(2)) if m else (10 ** 9, run)


def collect(models: Path, runs: list[str] | None = None) -> dict[str, dict[str, dict]]:
    """{report name: {run: heldout.json}} over <models>/recovered_<run>/heldout_*/, runs in version order"""
    out: dict[str, dict[str, dict]] = {}
    dirs = sorted(Path(models).glob(f"{RUN_DIR}*"), key=lambda d: _run_key(d.name[len(RUN_DIR):]))
    for d in dirs:
        run = d.name[len(RUN_DIR):]
        if runs and run not in runs:
            continue
        for rep in sorted(d.glob("heldout_*/heldout.json")):
            name = rep.parent.name
            if "_pre_" in name or name.endswith("_pre1029"):
                continue
            out.setdefault(name, {})[run] = json.loads(rep.read_text(encoding="utf-8"))
    return out


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


def to_markdown(table: dict[str, dict[str, dict]]) -> str:
    lines = ["# Held-out reports by run", "",
             "Per report set, a row per run: its gate verdicts against the capture, the camera's own photo, the "
             "oracle linear frontier and on tone (heldout.tone_gate), and its medians, as output / frontier at the run's own shadow SNR where the "
             "report has a frontier (lower is better but for mtf50 and texture, judged by distance to the truth)."]
    head = "| run | step | " + " | ".join(GATES) + " | " + " | ".join(METRICS) + " |"
    rule = "|---|---:|" + "---|" * len(GATES) + "---:|" * len(METRICS)
    for name in sorted(table):
        lines += ["", f"## {name}", "", head, rule]
        for run, rep in table[name].items():
            lines.append(f"| {run} | " + " | ".join(cells(rep)) + " |")
    return "\n".join(lines) + "\n"
