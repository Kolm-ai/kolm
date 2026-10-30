"""The learning-curve law (harness.model.curves): its areas, its fit and its prediction across schedules, on curves
the law itself draws. No files, no GPU."""
from __future__ import annotations

import json

import numpy as np
import pytest

from harness.model import curves as C

TRUE = (0.30, 2.0, 0.6, 2e-6)


def test_areas_of_a_constant_rate_and_of_a_drop():
    s1, s2 = C.areas(np.full(1000, 1e-3), 1e-3)
    assert s1[-1] == pytest.approx(1000) and np.all(s2 == 0)
    rates = np.r_[np.full(500, 1.0), np.full(500, 0.5)]
    s1, s2 = C.areas(rates, 1.0)
    assert s1[-1] == pytest.approx(750)
    # one drop of 0.5 at step 501, its momentum decaying by LAM a step: S2 = 0.5 * sum LAM^k over the 500 steps left
    assert s2[-1] == pytest.approx(0.5 * (1 - C.LAM ** 500) / (1 - C.LAM))


def test_the_fit_on_one_schedule_predicts_another():
    peak = 2e-4
    short = C.cosine(40000, peak)
    s1, s2 = C.areas(short, peak)
    steps = np.arange(500, 40001, 500)
    y = C.law(TRUE, s1[steps - 1], s2[steps - 1]) + np.random.default_rng(0).normal(0, 1e-4, steps.size)
    res = C.assess(short, steps, y, totals=[120000])
    assert res["backtest"][0.75]["rel_err"] == pytest.approx(0, abs=0.01)
    t1, t2 = C.areas(C.cosine(120000, peak), peak)
    assert res["cosine"][120000] == pytest.approx(C.law(TRUE, t1[-1], t2[-1]), rel=0.01)


def test_annealing_lowers_the_loss_at_equal_forward_area():
    """a cooldown is worth more than its lost forward area: the law's S2 term"""
    const = np.full(20000, 1.0)
    wsd = np.r_[np.full(16000, 1.0), np.linspace(1.0, 0.0, 4000)]
    l_const = C.law(TRUE, *(a[-1] for a in C.areas(const[:18000], 1.0)))
    l_wsd = C.law(TRUE, *(a[-1] for a in C.areas(wsd, 1.0)))
    assert C.areas(wsd, 1.0)[0][-1] == pytest.approx(18000, rel=1e-3) and l_wsd < l_const


def test_a_metric_that_grows_is_fitted_as_its_negative():
    peak = 1e-3
    rates = C.cosine(30000, peak)
    s1, s2 = C.areas(rates, peak)
    steps = np.arange(500, 30001, 500)
    y = 25.0 - C.law((0.0, 40.0, 0.5, 1e-5), s1[steps - 1], s2[steps - 1])
    res = C.assess(rates, steps, y, totals=[60000], higher_is_better=True)
    assert res["cosine"][60000] > res["last"] and abs(res["backtest"][0.75]["rel_err"]) < 0.01


def test_read_log_holds_each_logged_rate(tmp_path):
    rows = [{"step": 50, "lr": 1.0}, {"step": 100, "lr": 2.0, "val_real_loss": 0.5},
            {"step": 150, "lr": 3.0, "val_real_restore": {"shading_rmse": {"output": 0.01}}}]
    p = tmp_path / "train_log.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    rates, steps, y = C.read_log(p, "val_real_loss")
    assert rates.size == 150 and rates[0] == 1.0 and rates[99] == 2.0 and rates[-1] == 3.0
    assert steps.tolist() == [100] and y.tolist() == [0.5]
    assert C.read_log(p, "real:shading_rmse")[2].tolist() == [0.01]
