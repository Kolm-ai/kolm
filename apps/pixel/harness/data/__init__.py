"""Data layer: a manifest of every external source the harness rests on, a resumable
parallel collector, API expanders that turn one manifest entry into thousands of files,
and derivers that turn raw downloads into harness-ready tables."""
from __future__ import annotations

import os
from pathlib import Path


def data_root() -> Path:
    """Where raw and derived data live. Large tiers belong on a real disk: set HARNESS_DATA_ROOT."""
    return Path(os.environ.get("HARNESS_DATA_ROOT", "data")).resolve()
