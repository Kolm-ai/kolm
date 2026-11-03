"""The verdict is the process exit code: `python -m harness` must return what cli.main returns
(0 PASS, 1 FAIL, 2 INCOMPLETE). Until 2026-10-27 the entry point dropped it and every run exited 0."""
import runpy

import pytest

import harness.cli


@pytest.mark.parametrize("code", [0, 1, 2])
def test_module_entry_point_propagates_exit_code(monkeypatch, code):
    monkeypatch.setattr(harness.cli, "main", lambda argv=None: code)
    with pytest.raises(SystemExit) as e:
        runpy.run_module("harness", run_name="__main__")
    assert e.value.code == code
