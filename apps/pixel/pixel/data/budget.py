"""Storage planner for a fixed disk.

The one structural decision that makes the pipeline fit: store sources once,
never store degraded copies, never store labels you can recompute. A training
pair is ``(source, seed)``; a label row is a few hundred bytes. So the disk
holds sources + a thin label index + checkpoints, and the budget is arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

TB = 1024**4
GB = 1024**3
MB = 1024**2


@dataclass(frozen=True)
class BudgetInputs:
    disk_tb: float = 4.0
    headroom_fraction: float = 0.15
    mean_source_bytes: float = 2.0 * MB  # 2-4 MP JPEG at q95
    hdr_sources: int = 2000
    mean_hdr_bytes: float = 25.0 * MB  # 4k .hdr
    recipes_per_source: int = 32
    label_row_bytes: float = 320.0  # JSONL with recipe + direction + 9 generic metrics
    checkpoints: int = 10
    checkpoint_bytes: float = 0.5 * GB
    shard_overhead_fraction: float = 0.01

    @classmethod
    def from_dict(cls, d: dict) -> "BudgetInputs":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def plan(inp: BudgetInputs) -> dict:
    usable = inp.disk_tb * TB * (1.0 - inp.headroom_fraction)
    hdr = inp.hdr_sources * inp.mean_hdr_bytes
    ckpt = inp.checkpoints * inp.checkpoint_bytes
    per_source = inp.mean_source_bytes * (1.0 + inp.shard_overhead_fraction) + inp.recipes_per_source * inp.label_row_bytes
    remaining = usable - hdr - ckpt
    n_sources = int(max(0, remaining // per_source))
    labels = n_sources * inp.recipes_per_source
    return {
        "inputs": asdict(inp),
        "usable_bytes": int(usable),
        "hdr_bytes": int(hdr),
        "checkpoint_bytes": int(ckpt),
        "per_source_bytes": int(per_source),
        "max_sources": n_sources,
        "source_bytes": int(n_sources * inp.mean_source_bytes),
        "label_rows": labels,
        "label_bytes": int(labels * inp.label_row_bytes),
        "degraded_pairs_available": labels,
        "degraded_pairs_stored": 0,
        "note": "degraded pairs are regenerated from (source, seed) at train time; storing them would cost "
        f"{int(labels * inp.mean_source_bytes) / TB:.1f} TB and buy nothing",
    }


def render_plan(p: dict) -> str:
    rows = [
        ("usable after headroom", p["usable_bytes"]),
        ("HDR sources", p["hdr_bytes"]),
        ("checkpoints", p["checkpoint_bytes"]),
        ("LDR sources", p["source_bytes"]),
        ("label index", p["label_bytes"]),
    ]
    lines = ["| item | size |", "|---|---|"]
    for name, b in rows:
        lines.append(f"| {name} | {b / GB:,.0f} GB |")
    lines.append("")
    lines.append(f"max LDR sources: {p['max_sources']:,}")
    lines.append(f"label rows: {p['label_rows']:,} ({p['inputs']['recipes_per_source']} per source)")
    lines.append(f"degraded pairs: {p['degraded_pairs_available']:,} available, {p['degraded_pairs_stored']} stored")
    lines.append(p["note"])
    return "\n".join(lines) + "\n"
