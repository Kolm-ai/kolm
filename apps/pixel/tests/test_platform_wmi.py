"""harness/__init__.py turns WMI off on Windows: CPython 3.12's _wmi handle race killed loader workers at step 0."""
import platform
import sys

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import harness  # noqa: F401  (importing the package is the fix)
from harness import loader_context

win = pytest.mark.skipif(sys.platform != "win32", reason="the _wmi module is Windows only")


@win
def test_platform_answers_without_wmi():
    with pytest.raises(OSError):
        platform._wmi_query("CPU", "Architecture")
    assert platform.machine() and platform.win32_ver()[1] and platform.uname().system == "Windows"


class _Probe(Dataset):
    """what a loader worker had imported, and in which order, when it fetched an item"""

    def __len__(self):
        return 2

    def __getitem__(self, i):
        mods = list(sys.modules)
        return mods.index("harness") < mods.index("torch"), platform._wmi_query.__name__


@win
def test_loader_workers_import_harness_before_torch():
    """torch asks WMI as it imports, so the fix has to be in place before the worker's first torch import"""
    assert loader_context(0) is None
    rows = list(DataLoader(_Probe(), batch_size=None, num_workers=2, multiprocessing_context=loader_context(2)))
    assert [tuple(r) for r in rows] == [(True, "_no_wmi")] * 2
