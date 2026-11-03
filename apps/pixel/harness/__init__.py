"""Measurement harness: is a generated image or video verifiably better than a simulated best-case camera?"""
import sys

if sys.platform == "win32":
    # CPython 3.12's _wmi races when a query times out (fixed upstream by gh-130727 in May 2025, after 3.12's last
    # bugfix release): the abandoned query thread later writes its UTF-16 answer into, and closes, a handle number
    # the process has reused by then. Every spawned DataLoader worker asks WMI through platform.uname() when torch
    # imports, so under a busy CPU the loader died at step 0 with "[WinError 6] The handle is invalid" on its queue
    # semaphore and "Architecture=9" in the log. Without WMI, platform answers from sys.getwindowsversion() and the
    # environment (the same uname, win32_ver and machine on this box). A spawned child does not re-import a
    # `-m harness` main (spawn skips a *.__main__), and a DataLoader worker unpickles its target, torch's
    # _worker_loop, first, so loaders take loader_context(): its Process class lives here, the first name in the
    # child's pickle, and the child imports this package before torch.
    import platform

    def _no_wmi(*_keys):
        raise OSError("WMI queries are off in this process (harness/__init__.py)")

    platform._wmi_query = _no_wmi

import multiprocessing.context as _mpc


class _HarnessFirstProcess(_mpc.SpawnProcess):
    """a spawn Process whose child imports harness first (its class is the first global the child unpickles)"""


class HarnessSpawnContext(_mpc.SpawnContext):
    Process = _HarnessFirstProcess


def loader_context(workers: int):
    """a DataLoader's multiprocessing_context: on Windows, spawn with harness imported first in every worker
    (above); elsewhere, and without workers, the default"""
    return HarnessSpawnContext() if sys.platform == "win32" and workers > 0 else None


__version__ = "0.1.0"
