"""Ordered gates with absolute thresholds and a 'beat the floor' rule.

Verification order is objective-first (cheap, deterministic) and ends at human
preference. Production effort runs the other way (subject, light, materials
first); the SPEC says both. The gate halts at the first failure so a report
never shows a resolution win on an image whose highlights are gone.

Thresholds are display-visibility thresholds, not tuned to any candidate:
one 8-bit code value is 1/255 = 0.0039, dE2000 of 2 is the usual 'just
noticeable in side-by-side' figure, and 0.5 px is half the sampling grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .metrics.suite import GROUPS, METRICS

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

ONE_CODE_VALUE = 1.0 / 255.0

THRESHOLDS = {
    "landmark_error_px": ("<=", 0.5),
    "highlight_loss_fraction": ("<=", 0.02),
    "shadow_noise_sigma": ("<=", ONE_CODE_VALUE),
    "shadow_detail_corr": (">=", 0.8),
    "banding_score": ("<=", 0.1),
    "patch_delta_e": ("<=", 2.0),
    "delta_e_p95": ("<=", 4.0),
    "halo_score": ("<=", 0.02),
    "blocking_score": ("<=", 1.0),
    "tiling_score": ("<=", 0.6),
    "flicker_score": ("<=", ONE_CODE_VALUE),
    "temporal_noise": ("<=", 1.5 * ONE_CODE_VALUE),
    "motion_error_px": ("<=", 0.25),
}

# Metrics that must additionally beat-or-tie the camera envelope to pass their gate.
FLOOR_METRICS = {"mtf50"}

GATE_ORDER = ("geometry", "tone_color", "artifacts", "resolution", "temporal")


@dataclass
class GateResult:
    name: str
    status: str
    checks: list = field(default_factory=list)
    reason: str = ""


def _ok(op: str, value: float, limit: float) -> bool:
    return value <= limit if op == "<=" else value >= limit


def compare(name: str, cand: float, floor: float) -> str:
    """'beats' / 'ties' / 'loses' for one metric against the floor, using the metric's tie tolerance."""
    direction, _group, tol = METRICS[name]
    if abs(cand - floor) <= tol:
        return "ties"
    better = cand > floor if direction == "higher" else cand < floor
    return "beats" if better else "loses"


def run_gates(
    metrics: dict,
    envelope: dict | None,
    qualities: list[str],
    is_video: bool,
    halt_on_fail: bool = True,
) -> list[GateResult]:
    """Evaluate every gate in order. Gates for qualities not claimed are SKIP; so is temporal on stills."""
    results: list[GateResult] = []
    halted = False
    for gate in GATE_ORDER:
        if halted:
            results.append(GateResult(gate, SKIP, reason="earlier gate failed"))
            continue
        if gate not in qualities:
            results.append(GateResult(gate, SKIP, reason="not claimed"))
            continue
        if gate == "temporal" and not is_video:
            results.append(GateResult(gate, SKIP, reason="still image"))
            continue
        checks = []
        status = PASS
        for name in GROUPS[gate]:
            if name not in metrics:
                continue
            value = float(metrics[name])
            if name in THRESHOLDS:
                op, limit = THRESHOLDS[name]
                ok = _ok(op, value, limit)
                checks.append({"metric": name, "value": value, "rule": f"{op} {limit}", "ok": ok})
                if not ok:
                    status = FAIL
            if name in FLOOR_METRICS and envelope and name in envelope:
                verdict = compare(name, value, float(envelope[name]))
                ok = verdict in ("beats", "ties")
                checks.append({"metric": name, "value": value, "rule": f"{verdict} floor {envelope[name]:.4g}", "ok": ok})
                if not ok:
                    status = FAIL
        reason = "; ".join(f"{c['metric']}={c['value']:.4g} not {c['rule']}" for c in checks if not c["ok"])
        results.append(GateResult(gate, status, checks, reason))
        if status == FAIL and halt_on_fail:
            halted = True
    return results


def verdict(results: list[GateResult]) -> tuple[bool, str]:
    for r in results:
        if r.status == FAIL:
            return False, f"halted at gate '{r.name}': {r.reason}"
    return True, "all claimed gates passed"
