"""Ordered gates from the claim. First failing gate stops the run; a missing metric skips its check.

With the truth's own metrics given, a to_truth metric is judged by distance to the truth's value, and a beats_camera
check passes on a tie where the camera already sits at the truth's value of a truth-bound metric: nothing can beat
it there, and a gate no candidate can pass measures nothing."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .metrics import REGISTRY, at_truth, verdict

OPS = ("<=", ">=", "<", ">", "beats_camera", "ties_or_beats_camera")


@dataclass
class CheckResult:
    metric: str
    op: str
    value: float | None
    candidate: float | None
    camera: float | None
    status: str  # pass | fail | skip
    detail: str = ""


@dataclass
class GateResult:
    name: str
    status: str  # pass | fail | skip
    checks: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _check(spec: dict, cand: dict, cam: dict, truth: dict | None = None, baseline: str = "camera") -> CheckResult:
    m, op = spec["metric"], spec["op"]
    if op not in OPS:
        raise ValueError(f"unknown op {op!r} for {m}; use one of {OPS}")
    c, k, v = cand.get(m), cam.get(m), spec.get("value")
    r = CheckResult(m, op, v, c, k, "skip")
    if _missing(c):
        r.detail = "candidate metric unavailable"
        return r
    if op.endswith("camera"):
        if _missing(k):
            r.detail = f"{baseline} metric unavailable"
            return r
        t = (truth or {}).get(m)
        t = None if _missing(t) else t
        vd, margin = verdict(m, c, k, t)
        ceiling = vd == "ties" and at_truth(m, k, t)
        ok = vd == "beats" or (vd == "ties" and (op == "ties_or_beats_camera" or ceiling))
        r.status = "pass" if ok else "fail"
        r.detail = (f"{vd} {baseline}, margin {margin:+.4g} (tie within {REGISTRY[m].tie_margin:g}"
                    f"{', by distance to the truth ' + format(t, '.4g') if REGISTRY[m].to_truth and t is not None else ''})")
        if ceiling and op == "beats_camera":
            r.detail += f"; {baseline} at the truth's value, a tie is the most any candidate can do"
        return r
    if v is None:
        raise ValueError(f"check {m} {op} needs a value")
    ok = {"<=": c <= v, ">=": c >= v, "<": c < v, ">": c > v}[op]
    r.status = "pass" if ok else "fail"
    r.detail = f"{c:.4g} {op} {v:g}"
    return r


def run_gates(gates_spec: list[dict], cand: dict, cam: dict, truth: dict | None = None,
              baseline: str = "camera") -> list[GateResult]:
    """`baseline` names what `cam` holds in the checks' details (the camera; on real pairs also the capture)."""
    out: list[GateResult] = []
    for g in gates_spec:
        gr = GateResult(g["name"], "pass", [_check(c, cand, cam, truth, baseline) for c in g["checks"]])
        statuses = {c.status for c in gr.checks}
        gr.status = "fail" if "fail" in statuses else ("skip" if statuses <= {"skip"} else "pass")
        out.append(gr)
        if gr.status == "fail":
            break
    return out


def overall(results: list[GateResult]) -> str:
    """FAIL at <gate> on the first failure; INCOMPLETE when every gate that ran passed but some check could not be
    evaluated (no human study, a metric the candidate lacks): a skipped check is not a pass. PASS only when every
    check of every gate was evaluated and passed."""
    for g in results:
        if g.status == "fail":
            return f"FAIL at {g.name}"
    skipped = [g.name for g in results if g.status == "skip" or any(c.status == "skip" for c in g.checks)]
    if skipped:
        return "INCOMPLETE (skipped: " + ", ".join(skipped) + ")"
    return "PASS"
