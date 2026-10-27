"""Offline labelling at scale.

For every source image and index ``r`` in ``range(recipes_per_source)``:
seed = blake2b(source_sha, r) -> recipe -> degraded image -> direction vector +
generic metrics against the clean source. One JSONL row per (source, r). The
row holds everything needed to rebuild the pair; the degraded image is thrown
away. Shards map over a process pool, so throughput is cores x shards.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .data.shards import read_shard
from .degrade import label_pair, recipe_to_json
from .metrics.generic import generic_metrics
from .ops import decode_image, resize


def prepare(img: np.ndarray, size: int) -> np.ndarray:
    """Centre-crop to square then resize to ``size``; keeps labelling cost flat per source."""
    H, W = img.shape[:2]
    s = min(H, W)
    y0, x0 = (H - s) // 2, (W - s) // 2
    return resize(img[y0 : y0 + s, x0 : x0 + s], size, size, "box" if s > size else "bilinear")


def label_image(img: np.ndarray, source_id: str, recipes_per_source: int, max_ops: int = 3) -> list[dict]:
    rows = []
    for r in range(recipes_per_source):
        degraded, recipe, direction, seed = label_pair(img, source_id, r, max_ops=max_ops)
        rows.append(
            {
                "source": source_id,
                "r": r,
                "seed": seed,
                "recipe": recipe_to_json(recipe),
                "direction": [round(float(v), 5) for v in direction],
                "metrics": {k: round(float(v), 5) for k, v in generic_metrics(degraded, img).items()},
            }
        )
    return rows


def label_shard(shard: str | Path, out_path: str | Path, recipes_per_source: int = 8, size: int = 256, max_ops: int = 3, limit: int | None = None) -> int:
    shard, out_path = Path(shard), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".part")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for key, _ext, data, _meta in read_shard(shard):
            try:
                img = prepare(decode_image(data), size)
            except Exception:  # noqa: BLE001 - a corrupt source is skipped and counted by the caller
                continue
            for row in label_image(img, key, recipes_per_source, max_ops=max_ops):
                row["shard"] = shard.name
                f.write(json.dumps(row) + "\n")
                n += 1
            if limit is not None and n >= limit * recipes_per_source:
                break
    os.replace(tmp, out_path)
    return n


def label_shards(shards: list[Path], out_dir: str | Path, workers: int | None = None, **kw) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(s, out_dir / (s.stem + ".jsonl")) for s in shards if not (out_dir / (s.stem + ".jsonl")).exists()]
    counts = {}
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        futs = {pool.submit(label_shard, s, o, **kw): s for s, o in todo}
        for fut, s in futs.items():
            counts[s.name] = fut.result()
    return {"shards": len(shards), "labelled_now": len(todo), "rows": counts}


def read_labels(path: str | Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
