"""The burst run (harness.model.bursttrain): two steps from a tiny single-frame checkpoint on synthetic bursts and
single-frame samples log the synthetic validation with all frames and with none beside the starting checkpoint,
save a BurstNet checkpoint that loads back, and keep the starting checkpoint at step 0 of the stream; the
evaluation scores the frames, no frames and the starting checkpoint on the same tiles."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
import torch

import harness.model.data as D
from harness.data import data_root
from harness.model.burst import BurstNet, load_burst, synthetic_fixed_set
from harness.model.data import capture_of_params

from test_realtrain import DERIVED_TABLES, _tiny_sources, needs_data


def _setup(tmp_path, monkeypatch):
    src_root = data_root()
    (tmp_path / "derived").mkdir()
    for f in DERIVED_TABLES:
        shutil.copy(src_root / "derived" / f, tmp_path / "derived" / f)
    monkeypatch.setenv("HARNESS_DATA_ROOT", str(tmp_path))
    _tiny_sources(tmp_path, ["tiny"], n=4, size=112)
    monkeypatch.setattr(D, "split_of", lambda it: "val" if it.file.endswith("_0.png") else "train")
    from harness.model.train import train
    init = tmp_path / "single"
    train(str(init), ["tiny"], steps=1, batch=2, tile=32, workers=0, base=8, val_n=2, calib_n=8, ckpt_every=1,
          device="cpu", noise_map=True, log=lambda *a: None)
    return init / "last.pt"


@needs_data
def test_burst_run_and_eval_from_a_single_checkpoint(tmp_path, monkeypatch):
    from harness.model import bursttrain as BT
    init = _setup(tmp_path, monkeypatch)
    out = tmp_path / "burst"
    log = []
    v = BT.burst_train(str(out), str(init), steps=2, batch=2, tile=32, frames=2, workers=0, p_real_burst=0.0,
                       p_single=0.5, val_n=2, ckpt_every=1, device="cpu", log=log.append)
    assert np.isfinite(v["val_frames"]["val_loss"]) and np.isfinite(v["val_none"]["val_loss"])
    assert any(s.startswith("no real burst validation set") for s in log)
    rows = [json.loads(line) for line in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "val_single" in rows[0] and rows[0]["step"] == 0
    assert sum("val_frames" in r for r in rows) == 3
    model, ck = load_burst(out / "last.pt")
    assert isinstance(model, BurstNet) and ck["step"] == 2 and ck["train"]["init"] == str(init)
    assert ck["train"]["sources"] == ["tiny"] and ck["train"]["frames"] == 2

    def fake_set(n, tile, frames, seed, root=None, split="val", subpix=False):
        s = synthetic_fixed_set(["tiny"], "val", n, tile, frames, seed)
        for k, t in enumerate(s):
            t.update(capture=capture_of_params(t["a"]), burst=k, real=True)
        return s
    monkeypatch.setattr(BT, "real_burst_set", fake_set)
    rep = BT.burst_evaluate(out / "last.pt", n=2, tile=32, device="cpu", out=tmp_path / "ev", frontier=False,
                            log=lambda *a: None)
    assert set(rep["reports"]) == {"frames", "none", "single"} and rep["n"] == 2
    for tag in rep["reports"]:
        assert (tmp_path / "ev" / f"burst_{tag}.json").exists()
    md = BT.summary_table(rep)
    assert "| metric | input | frames | none | single | truth |" in md


@needs_data
def test_burst_run_moves_no_module(tmp_path, monkeypatch):
    # The step-0 reference is the burst model's own body (BurstNet(single) keeps it as model.single): moving it to
    # the CPU after its validation left the body there and the first step failed on two devices (10-30, the burst
    # run on v17). A run moves its modules once, to the device, before the reference is scored.
    from harness.model import bursttrain as BT
    init = _setup(tmp_path, monkeypatch)
    moved = []
    real_cpu = torch.nn.Module.cpu

    def cpu(self):
        moved.append(type(self).__name__)
        return real_cpu(self)
    monkeypatch.setattr(torch.nn.Module, "cpu", cpu)
    BT.burst_train(str(tmp_path / "burst"), str(init), steps=1, batch=2, tile=32, frames=2, workers=0,
                   p_real_burst=0.0, p_single=0.5, val_n=2, ckpt_every=1, device="cpu", log=lambda *a: None)
    assert moved == []


@needs_data
@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="needs a GPU")
def test_burst_run_steps_on_the_gpu(tmp_path, monkeypatch):
    from harness.model import bursttrain as BT
    init = _setup(tmp_path, monkeypatch)
    v = BT.burst_train(str(tmp_path / "burst"), str(init), steps=2, batch=2, tile=32, frames=2, workers=0,
                       p_real_burst=0.0, p_single=0.5, val_n=2, ckpt_every=1, device="cuda", log=lambda *a: None)
    assert np.isfinite(v["val_frames"]["val_loss"])
