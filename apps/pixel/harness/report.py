"""report.json + report.md: claim, per-metric table, per-quality verdict, gate trail, delivery delta, preference."""
from __future__ import annotations

import json
from pathlib import Path

from .gate import GateResult, overall
from .metrics import REGISTRY, verdict

_RANK = {"loses": 0, "ties": 1, "beats": 2}


def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def metric_rows(cand: dict, cam: dict, truth: dict) -> dict[str, dict]:
    rows = {}
    for m in cand:
        c, k = cand.get(m), cam.get(m)
        if m in REGISTRY and c is not None and k is not None:
            vd, margin = verdict(m, c, k, truth.get(m))
        else:
            vd, margin = "n/a", None
        rows[m] = {"candidate": c, "camera": k, "truth": truth.get(m), "verdict": vd, "margin": margin}
    return rows


def quality_verdicts(qualities: dict[str, list[str]], rows: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for q, metrics in qualities.items():
        vds = [rows[m]["verdict"] for m in metrics if m in rows and rows[m]["verdict"] in _RANK]
        out[q] = {"verdict": min(vds, key=_RANK.get) if vds else "n/a", "metrics": metrics}
    return out


def write_report(out_dir, claim: dict, candidate_name: str, cand: dict, cam: dict, truth: dict,
                 gates: list[GateResult], delivery: dict | None = None,
                 preference: dict | None = None, provenance: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = metric_rows(cand, cam, truth)
    qual = quality_verdicts(claim["qualities"], rows)
    rep = {
        "candidate": candidate_name,
        "overall": overall(gates),
        "claim": claim,
        "metrics": rows,
        "qualities": qual,
        "gates": [g.to_dict() for g in gates],
        "delivery": delivery,
        "preference": preference,
        "provenance": provenance,
    }
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    (out_dir / "report.md").write_text(_markdown(rep), encoding="utf-8")
    return rep


def _markdown(rep: dict) -> str:
    claim = rep["claim"]
    from .display import from_dict as display_from_dict
    s, cam = claim["scene"], claim["camera"]
    spec = display_from_dict(claim["display"])  # resolves derived:<id> presets to a geometry
    d = {"diagonal_in": spec.diagonal_in, "width_px": spec.width_px, "height_px": spec.height_px,
         "distance_m": spec.distance_m}
    L = [f"# {rep['candidate']} — {rep['overall']}", ""]
    L.append(f"Scene {s['width']}×{s['height']} px, {s['n_frames']} frame(s) @ {s['fps']} fps, "
             f"supersample {s['supersample']}, seed {s['seed']}. "
             f"Camera {cam.get('preset', 'custom')}, exposure {cam['exposure_s']:.4g} s. "
             f"Display {d['diagonal_in']}″ {d['width_px']}×{d['height_px']} at {d['distance_m']} m → "
             f"{_fmt(rep['metrics'].get('display_ppd', {}).get('candidate'))} ppd.")
    L += ["", "## Metrics", "", "| metric | candidate | camera | truth | verdict | margin |", "|---|---|---|---|---|---|"]
    for m, r in rep["metrics"].items():
        L.append(f"| {m} | {_fmt(r['candidate'])} | {_fmt(r['camera'])} | {_fmt(r['truth'])} | "
                 f"{r['verdict']} | {_fmt(r['margin'])} |")
    L += ["", "## Qualities", "", "| quality | verdict | metrics |", "|---|---|---|"]
    for q, r in rep["qualities"].items():
        L.append(f"| {q} | **{r['verdict']}** | {', '.join(r['metrics'])} |")
    L += ["", "## Gates", ""]
    for i, g in enumerate(rep["gates"], 1):
        L.append(f"{i}. **{g['name']}** — {g['status'].upper()}")
        for c in g["checks"]:
            L.append(f"   - {c['metric']} {c['op']}{'' if c['value'] is None else ' ' + _fmt(c['value'])}: "
                     f"{c['status']} ({c['detail']})")
    n_spec = len(claim["gates"])
    if len(rep["gates"]) < n_spec:
        L.append(f"   - {n_spec - len(rep['gates'])} later gate(s) not run.")
    if rep.get("delivery"):
        dv = rep["delivery"]
        L += ["", f"## Delivery — {dv.get('format')} q{dv.get('quality')}, {dv.get('bytes', 0):,} bytes"
              + (f" ({dv['note']})" if dv.get("note") else ""), "",
              "| metric | delivered | Δ vs candidate |", "|---|---|---|"]
        for m, r in dv["metrics"].items():
            if r["delivered"] is not None:
                L.append(f"| {m} | {_fmt(r['delivered'])} | {_fmt(r['delta'])} |")
    if rep.get("provenance"):
        pv = rep["provenance"]
        L += ["", "## Provenance: rendered-pixel map " + ("supplied" if pv["map_supplied"] else
              f"not supplied (policy {pv['policy']}: " + ("every pixel declared recovered)" if pv["policy"] == "optional"
              else "no provenance verdict)")), "",
              "| region | invented (measured) | rendered (declared) | undeclared |", "|---|---|---|---|"]
        for name, r in pv["regions"].items():
            L.append(f"| {name} | {_fmt(r['invented'])} | {_fmt(r['rendered'])} | {_fmt(r['undeclared'])} |")
        pf = pv.get("per_frame") or []
        if pf:
            sig = [f["matched_blur_px"] for f in pf]
            L += ["", f"Candidate blur relative to the truth, matched Gaussian sigma in px: "
                  f"{min(sig):.2f} to {max(sig):.2f} over {len(pf)} frame(s). Pixels under moving landmarks "
                  "are not judged here (see motion_region_error)."]
        L += ["", f"Maps: `{pv['files']['invented_map']}`, `{pv['files']['invented_png']}` (frame 0)."]
    if rep.get("preference"):
        L += ["", "## Preference", ""]
        for q, fit in rep["preference"].get("questions", {}).items():
            pb = fit["p_beats"].get("candidate", {}).get("camera")
            if pb:
                L.append(f"- *{q}*: P(candidate ≻ camera) = {pb['p']:.3f} "
                         f"[{pb['low']:.3f}, {pb['high']:.3f}] from {fit['n_pairs']} trials")
    return "\n".join(L) + "\n"
