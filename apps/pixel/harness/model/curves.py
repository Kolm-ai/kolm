"""How much more training buys, read off a run's own validation curve before a longer run is spent.

The law: L(s) = L0 + A S1^-alpha - C S2 (Tissue et al. 2024, "Scaling law with learning rate annealing",
arXiv 2408.11029). S1 is the forward area, the sum of the learning rate over the steps in units of the peak rate;
S2 the annealing area, the sum over steps of m, the momentum of the rate's decrease (m_i = LAM m_(i-1) + eta_(i-1) -
eta_i). One fit on one run's curve, whatever its schedule, predicts the same recipe under another: a longer cosine,
or a constant trunk with a cooldown (Hagele et al. 2024, arXiv 2405.18392). The fit's own honesty is the backtest:
fit on the first part of the curve, predict the rest. On the seventeenth run's real-pair validation loss a fit on
three quarters of the curve predicted its end within 2 % (optimistic), one on half within 7 %; extrapolations beyond
the run should be read as at least as optimistic (MODEL_PLAN section 20).

Usage: CUDA_VISIBLE_DEVICES= PYTHONPATH=. HARNESS_DATA_ROOT=D:/lens-harness-data
python -X utf8 -m harness.model.curves v17 [--key val_real_loss] [--to 150000,300000]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter

from ..data import data_root
from .train import LR_FLOOR, WARMUP, lr_at

LAM = 0.999          # the annealing momentum per step (the paper's default)
MIN_STEP = 2000      # validation points before this are warmup transients, left out of the fit
BACKTEST = (0.5, 0.75)


def areas(rates: np.ndarray, peak: float) -> tuple[np.ndarray, np.ndarray]:
    """S1 and S2 after each step, from the rate at every step (rates[i] at step i + 1)"""
    eta = np.asarray(rates, dtype=float) / peak
    s1 = np.cumsum(eta)
    drop = np.diff(eta, prepend=eta[0]) * -1.0
    return s1, np.cumsum(lfilter([1.0], [1.0, -LAM], drop))


def law(p, s1, s2):
    l0, a, alpha, c = p
    return l0 + a * np.power(np.maximum(s1, 1.0), -alpha) - c * s2


def fit(s1: np.ndarray, s2: np.ndarray, y: np.ndarray, higher_is_better: bool = False) -> np.ndarray:
    """(L0, A, alpha, C) by least squares; a metric that grows with training is fitted as its negative"""
    y = -np.asarray(y, dtype=float) if higher_is_better else np.asarray(y, dtype=float)
    p0 = [y.min() - 0.1 * abs(y.min()), max(y.max() - y.min(), 1e-6) * 10, 0.5, 1e-6]
    r = least_squares(lambda p: law(p, s1, s2) - y, p0, bounds=([-np.inf, 0, 0.01, 0], [np.inf, np.inf, 3, np.inf]),
                      max_nfev=20000)
    return r.x


def predict(p, s1, s2, higher_is_better: bool = False):
    v = law(p, s1, s2)
    return -v if higher_is_better else v


def cosine(total: int, peak: float, warmup: int = WARMUP, floor: float = LR_FLOOR) -> np.ndarray:
    """the trainer's rate at every step of a `total`-step run"""
    return np.array([lr_at(k, total, peak, 0.0, warmup, floor) for k in range(total)])


def read_log(path: Path, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """the run's rate at every step (held from each logged value to the next), and the steps and values of `key`:
    a top-level field of the log's rows (val_real_loss, val_loss) or real:<metric> of its val_real_restore"""
    lr, val = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "lr" in r:
            lr[r["step"]] = r["lr"]
        if key.startswith("real:"):
            v = r.get("val_real_restore", {}).get(key[5:], {}).get("output")
        else:
            v = r.get(key)
        if v is not None:
            val[r["step"]] = float(v)
    ks = np.array(sorted(lr))
    rates = np.array([lr[k] for k in ks])
    every = np.repeat(rates, np.diff(ks, prepend=0))           # rows log the rate of their own step
    steps = np.array(sorted(val))
    return every, steps, np.array([val[k] for k in steps])


def assess(rates: np.ndarray, steps: np.ndarray, y: np.ndarray, totals=(), higher_is_better: bool = False,
           peak: float | None = None) -> dict:
    """the fit, its backtests and its predictions under the trainer's cosine to each of `totals` steps"""
    peak = peak or float(rates.max())
    s1, s2 = areas(rates, peak)
    keep = (steps >= MIN_STEP) & (steps <= len(rates))
    st, y = steps[keep], y[keep]
    a1, a2 = s1[st - 1], s2[st - 1]
    out = {"points": int(keep.sum()), "last": float(y[-1]), "backtest": {}}
    for frac in BACKTEST:
        cut = int(len(st) * frac)
        p = fit(a1[:cut], a2[:cut], y[:cut], higher_is_better)
        pr = predict(p, a1[cut:], a2[cut:], higher_is_better)
        out["backtest"][frac] = {"end": float(pr[-1]), "rel_err": float((pr[-1] - y[-1]) / abs(y[-1])),
                                 "mean_abs_err": float(np.mean(np.abs(pr - y[cut:])))}
    p = fit(a1, a2, y, higher_is_better)
    out["params"] = p.tolist()
    out["resid_sd"] = float(np.std(predict(p, a1, a2, higher_is_better) - y))
    floor = float(rates[-1] / peak)
    out["cosine"] = {}
    for total in totals:
        t1, t2 = areas(cosine(int(total), peak, WARMUP, floor), peak)
        out["cosine"][int(total)] = float(predict(p, t1[-1:], t2[-1:], higher_is_better)[0])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--key", action="append", help="val_real_loss (default), val_loss, real:<metric>; repeatable")
    ap.add_argument("--to", default="150000,300000", help="total steps to predict the same recipe's cosine at")
    a = ap.parse_args(argv)
    from ..metrics import REGISTRY
    totals = [int(t) for t in a.to.split(",") if t]
    for run in a.runs:
        path = Path(data_root()) / "models" / f"recovered_{run}" / "train_log.jsonl"
        for key in a.key or ["val_real_loss"]:
            hib = key.startswith("real:") and REGISTRY[key[5:]].higher_is_better
            rates, steps, y = read_log(path, key)
            res = assess(rates, steps, y, [len(rates)] + totals, hib)
            bt = ", ".join(f"fit on {int(f * 100)} %: end {b['end']:.4g} ({b['rel_err']:+.1%})"
                           for f, b in res["backtest"].items())
            pr = ", ".join(f"{t // 1000}k {v:.4g}" for t, v in res["cosine"].items())
            print(f"{run} {key}: last {res['last']:.4g}; backtest {bt}; same recipe's cosine to {pr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
