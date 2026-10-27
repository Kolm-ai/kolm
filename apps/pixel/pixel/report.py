"""Markdown report from an evaluate_claim result."""

from __future__ import annotations

from pathlib import Path

from .metrics.suite import METRICS


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def render_report(result: dict) -> str:
    claim = result["claim"]
    lines = [f"# Claim report: {claim['claim_id']}", ""]
    lines.append(f"**Verdict:** {result['verdict']}")
    lines.append("")
    lines.append(f"- regime: `{claim['regime']}` (provable: {result['provable']})")
    lines.append(f"- qualities claimed: {', '.join(claim['qualities'])}")
    lines.append(f"- candidate: `{claim['candidate']}`")
    lines.append(f"- delivery: `{claim['delivery']}`")
    lines.append("")
    if not result["provable"]:
        lines.append("Nothing was measured. A claim about a real event needs independent evidence, not a metric.")
        return "\n".join(lines) + "\n"

    d = result["display"]
    lines.append("## Ceiling (display)")
    lines.append("")
    lines.append(f"- {d['diagonal_in']}\" {d['width_px']}x{d['height_px']} at {d['distance_m']} m: **{d['ppd']} ppd** "
                 f"(acuity threshold {d['ppd_threshold_mean']} mean, {d['ppd_threshold_best']} best observers; headroom {d['resolution_headroom']}x)")
    lines.append(f"- master at 2x display: {d['master_size'][0]}x{d['master_size'][1]}")
    lines.append("")
    c = result["camera"]
    lines.append("## Floor (best-case camera)")
    lines.append("")
    lines.append(f"- pitch {c['pixel_pitch_um']} um, f/{c['f_number']}, full well {c['full_well_e']:.0f} e-, read noise {c['read_noise_e']} e-, "
                 f"{c['bit_depth']}-bit, Bayer={c['bayer']}, diffraction={c['diffraction']}")
    lines.append(f"- claim exposure {c['exposure_ev']:+.1f} EV, {c['exposure_s']} s, motion blur {c['motion_px'][0]:.2f}, {c['motion_px'][1]:.2f} px")
    lines.append(f"- envelope = best value per metric over EV sweep {sorted(result['camera_sweep'])}")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| metric | better | candidate | delivered | camera @claim | camera envelope | truth | vs envelope | delivered vs envelope |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, row in result["comparison"].items():
        direction = METRICS[name][0]
        lines.append(
            f"| {name} | {direction} | {_fmt(row['candidate'])} | {_fmt(row['delivered'])} | {_fmt(row['camera_at_claim'])} | "
            f"{_fmt(row['camera_envelope'])} | {_fmt(row['truth'])} | {row['vs_envelope']} | {row['delivered_vs_envelope']} |"
        )
    lines.append("")
    for title, key in (("Gates (master)", "gates"), (f"Gates (delivered as {claim['delivery']})", "gates_delivered")):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| gate | status | detail |")
        lines.append("|---|---|---|")
        for g in result[key]:
            detail = g["reason"] or "; ".join(f"{ch['metric']} {ch['rule']}" for ch in g["checks"])
            lines.append(f"| {g['name']} | {g['status']} | {detail} |")
        lines.append("")
    lines.append("## Wins over the camera envelope")
    lines.append("")
    lines.append(f"- master: {', '.join(result['wins']) or 'none'}")
    lines.append(f"- delivered: {', '.join(result['wins_delivered']) or 'none'}")
    lost = sorted(set(result["wins"]) - set(result["wins_delivered"]))
    if lost:
        lines.append(f"- **lost in delivery:** {', '.join(lost)}")
    lines.append("")
    if result.get("preference"):
        p = result["preference"]
        lines.append("## Blinded preference")
        lines.append("")
        lines.append(f"- {p.get('status')} (gap {p.get('gap_logit', 0):+.3f} logit, 95% CI {p.get('ci95')}, n={p.get('n_pairs')})")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_report(result: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(result), encoding="utf-8")
    return path
