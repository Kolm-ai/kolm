"""Data layer: a manifest of every external source the harness rests on, a resumable
parallel collector, API expanders that turn one manifest entry into thousands of files,
and derivers that turn raw downloads into harness-ready tables."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def data_root() -> Path:
    """Where raw and derived data live. Large tiers belong on a real disk: set HARNESS_DATA_ROOT."""
    return Path(os.environ.get("HARNESS_DATA_ROOT", "data")).resolve()


def reserve_bytes(env: str, default_gb: float) -> float:
    """A free-space floor in bytes: the environment variable `env` in GB when set, else `default_gb`."""
    return float(os.environ.get(env, default_gb)) * 1e9


def keeps_reserve(path, need: int, reserve: float) -> bool:
    """True when writing `need` more bytes under `path` (or its nearest existing parent) leaves at least `reserve`
    bytes free on that drive. Raw downloads and the decoded cache both stop at their floor rather than fill a disk."""
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free - max(0, int(need)) >= reserve
