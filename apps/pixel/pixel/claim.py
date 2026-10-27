"""Claim schema. No tuple, no claim.

A claim names the regime, the qualities, the scene, the capture conditions and
the display. Without all five there is nothing to measure, and the harness
refuses to run rather than producing a number that means nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REGIMES = {
    # A world we specify completely: every quality below is provable.
    "synthetic_world": True,
    # Exceeding what the stated capture can record of that world: provable.
    "capture_exceed": True,
    # Accuracy about a real event: not provable by this harness. A convincing
    # image is not evidence of what happened.
    "record_of_reality": False,
}

QUALITIES = {"geometry", "tone_color", "artifacts", "resolution", "temporal"}

CANDIDATE_KINDS = {"truth", "camera", "files"}


class ClaimError(ValueError):
    pass


@dataclass
class Claim:
    claim_id: str
    regime: str
    qualities: list[str]
    scene: dict
    capture: dict
    display: dict
    candidate: dict
    delivery: dict = field(default_factory=lambda: {"format": "png"})
    preference: dict | None = None

    @property
    def provable(self) -> bool:
        return REGIMES.get(self.regime, False)

    @property
    def frames(self) -> int:
        return int(self.scene.get("frames", 1))

    @property
    def is_video(self) -> bool:
        return self.frames > 1

    @classmethod
    def from_dict(cls, d: dict) -> "Claim":
        missing = [k for k in ("claim_id", "regime", "qualities", "scene", "capture", "display", "candidate") if k not in d]
        if missing:
            raise ClaimError(f"claim is missing required fields: {', '.join(missing)}. No tuple, no claim.")
        if d["regime"] not in REGIMES:
            raise ClaimError(f"unknown regime {d['regime']!r}; one of {sorted(REGIMES)}")
        qualities = list(d["qualities"])
        if not qualities:
            raise ClaimError("qualities must name at least one of " + ", ".join(sorted(QUALITIES)))
        bad = sorted(set(qualities) - QUALITIES)
        if bad:
            raise ClaimError(f"unknown qualities {bad}; one of {sorted(QUALITIES)}")
        cand = dict(d["candidate"])
        if cand.get("kind") not in CANDIDATE_KINDS:
            raise ClaimError(f"candidate.kind must be one of {sorted(CANDIDATE_KINDS)}")
        if cand["kind"] == "files" and not cand.get("path"):
            raise ClaimError("candidate.kind=files needs candidate.path (a file or a directory of frames)")
        for key, needed in (("capture", ("exposure_s",)), ("display", ("diagonal_in", "width_px", "height_px", "distance_m"))):
            lacking = [k for k in needed if k not in d[key]]
            if lacking:
                raise ClaimError(f"{key} is missing {lacking}")
        if "temporal" in qualities and int(d["scene"].get("frames", 1)) < 2:
            raise ClaimError("a temporal claim needs scene.frames >= 2")
        return cls(
            claim_id=str(d["claim_id"]),
            regime=d["regime"],
            qualities=qualities,
            scene=dict(d["scene"]),
            capture=dict(d["capture"]),
            display=dict(d["display"]),
            candidate=cand,
            delivery=dict(d.get("delivery", {"format": "png"})),
            preference=dict(d["preference"]) if d.get("preference") else None,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Claim":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "regime": self.regime,
            "qualities": self.qualities,
            "scene": self.scene,
            "capture": self.capture,
            "display": self.display,
            "candidate": self.candidate,
            "delivery": self.delivery,
            "preference": self.preference,
        }
