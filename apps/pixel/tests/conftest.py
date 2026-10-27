import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pixel.colorsci import display_encode  # noqa: E402
from pixel.scene import Scene, SceneSpec  # noqa: E402


@pytest.fixture(scope="session")
def small_scene():
    return Scene(SceneSpec(seed=7, width=256, height=256, frames=3, pan_px_per_frame=(1.37, -0.61)))


@pytest.fixture(scope="session")
def small_ref(small_scene):
    return display_encode(small_scene.render(0.0))


@pytest.fixture(scope="session")
def small_truth(small_scene):
    return small_scene.render(0.0)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)
