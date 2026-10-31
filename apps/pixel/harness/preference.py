"""Blinded pairwise preference: manifest, response ingest, preference model with bootstrap intervals.

No UI. Any collector that shows two stimuli side by side and records left/right/tie per question
works. Stimuli are copied under opaque ids; the key that maps ids back to roles lives in a separate file.

Two estimators share one output shape (`p_beats[a][b] = {p, low, high}`):

* `fit_bradley_terry`: the classic MM fit (Hunter 2004) on decisive wins only. Kept as the reference.
* `fit_rao_kupper`: Bradley–Terry with a tie band (Rao & Kupper 1967, logistic link) and one
  reliability scale per rater. With log-strengths s, tie parameter θ ≥ 0 and rater scale β_r,
  d = β_r (s_a − s_b), P(a wins) = σ(d − θ), P(b wins) = σ(−d − θ), P(tie) = σ(d + θ) − σ(d − θ).
  θ = 0 and β = 1 is plain Bradley–Terry. Fitted by L-BFGS on the penalised log-likelihood
  (ridge on s, unit-normal prior on log β), so raters with few votes stay near β = 1 and
  items never seen win or lose stay finite. Validated at scale on public vote sets by
  `harness/data/prefdata.py` (DATA_PLAN.md section 4 step 3).

Votes are arrays: a, b item indices, y ∈ {+1 (a wins), −1 (b wins), 0 (tie)}, r rater index (−1 = unknown).
"""
from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

QUESTIONS = ("beautiful", "convincing")
PRIOR_PSEUDO_TRIALS = 0.5  # per pair, split evenly: keeps every strength finite with sparse data
RIDGE_S = 0.01             # L2 on log-strengths: identifiability and finiteness, not shrinkage of note
RIDGE_LOG_BETA = 1.0       # unit-normal prior on log rater scale
TAU_MIN = -10.0            # θ = exp(τ) floor: a set with no ties drives τ here and the model is plain BT
TIE_CHOICES = ("tie", "both", "both_good", "both_bad", "neither", "equal")


def make_manifest(items: dict[str, str], trials_per_pair: int = 10, seed: int = 0,
                  questions=QUESTIONS, stimuli_dir=None) -> tuple[dict, dict]:
    """items: role -> path. Returns (manifest, key). With stimuli_dir, copies each item under its opaque id."""
    rng = np.random.default_rng(seed)
    roles = list(items)
    ids = {r: rng.bytes(8).hex() for r in roles}
    files = {}
    for r in roles:
        src = Path(items[r])
        if stimuli_dir is not None:
            dst = Path(stimuli_dir) / (ids[r] + ("" if src.is_dir() else src.suffix))
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            files[ids[r]] = str(dst)
        else:
            files[ids[r]] = str(src)
    trials = []
    for a, b in combinations(roles, 2):
        for _ in range(trials_per_pair):
            left, right = (a, b) if rng.random() < 0.5 else (b, a)
            trials.append({"left": ids[left], "right": ids[right]})
    rng.shuffle(trials)
    for n, t in enumerate(trials):
        t["trial"] = n
    manifest = {"questions": list(questions), "files": files, "trials": trials,
                "response_csv_columns": ["trial", "question", "choice(left|right|tie)", "rater(optional)"]}
    key = {ids[r]: r for r in roles}
    return manifest, key


def read_responses(path) -> list[dict]:
    """CSV columns trial, question, choice (left|right|tie); optional rater id."""
    with open(path, newline="", encoding="utf-8") as f:
        return [{"trial": int(r["trial"]), "question": r["question"].strip(),
                 "choice": r["choice"].strip().lower(), "rater": (r.get("rater") or "").strip()}
                for r in csv.DictReader(f)]


@dataclass
class Votes:
    """Pairwise votes over `items`, optionally by `raters`. y: +1 a wins, −1 b wins, 0 tie; r: −1 unknown."""
    items: list[str]
    a: np.ndarray
    b: np.ndarray
    y: np.ndarray
    r: np.ndarray
    raters: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.a.size)

    @property
    def tie_share(self) -> float:
        return float(np.mean(self.y == 0)) if len(self) else 0.0

    def subset(self, mask) -> "Votes":
        return Votes(self.items, self.a[mask], self.b[mask], self.y[mask], self.r[mask], self.raters)

    @classmethod
    def from_records(cls, records, items=None, raters=None) -> "Votes":
        """records: iterable of (a, b, outcome, rater) with outcome in {'a','b','tie'} and rater str or None."""
        recs = list(records)
        items = list(items) if items is not None else sorted({x for rec in recs for x in rec[:2]})
        idx = {it: i for i, it in enumerate(items)}
        rater_names = list(raters) if raters is not None else sorted({rec[3] for rec in recs if rec[3]})
        ridx = {name: i for i, name in enumerate(rater_names)}
        out = {"a": [], "b": [], "y": [], "r": []}
        for a, b, outcome, rater in recs:
            out["a"].append(idx[a]); out["b"].append(idx[b])
            out["y"].append(1 if outcome == "a" else -1 if outcome == "b" else 0)
            out["r"].append(ridx[rater] if rater else -1)
        return cls(items, np.asarray(out["a"], dtype=np.int64), np.asarray(out["b"], dtype=np.int64),
                   np.asarray(out["y"], dtype=np.int8), np.asarray(out["r"], dtype=np.int64), rater_names)


def votes_by_question(manifest: dict, key: dict, responses: list[dict]) -> dict[str, Votes]:
    """question -> Votes over roles. left/right resolve through the manifest; tie words count as ties."""
    trials = {t["trial"]: t for t in manifest["trials"]}
    items = sorted(set(key.values()))
    recs: dict[str, list] = {q: [] for q in manifest["questions"]}
    for r in responses:
        t = trials[r["trial"]]
        if r["choice"] in ("left", "right"):
            outcome = "a" if r["choice"] == "left" else "b"
        elif r["choice"] in TIE_CHOICES:
            outcome = "tie"
        else:
            raise ValueError(f"trial {r['trial']}: choice must be left, right or tie, got {r['choice']!r}")
        recs.setdefault(r["question"], []).append((key[t["left"]], key[t["right"]], outcome, r["rater"] or None))
    return {q: Votes.from_records(v, items) for q, v in recs.items()}


def pairs_by_question(manifest: dict, key: dict, responses: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """question -> list of (winner role, loser role); ties dropped. Kept for the MM reference fit."""
    out = {}
    for q, v in votes_by_question(manifest, key, responses).items():
        dec = v.y != 0
        out[q] = [(v.items[a], v.items[b]) if y > 0 else (v.items[b], v.items[a])
                  for a, b, y in zip(v.a[dec], v.b[dec], v.y[dec])]
    return out


# ---- reference estimator: Bradley–Terry by MM --------------------------------------------
def fit_bradley_terry(pairs, items: list[str], n_iter: int = 1000, tol: float = 1e-10) -> np.ndarray:
    """MM algorithm (Hunter 2004). Strengths normalised to geometric mean 1."""
    idx = {it: i for i, it in enumerate(items)}
    n = len(items)
    wins = np.zeros(n)
    N = np.zeros((n, n))
    for a, b in combinations(range(n), 2):
        wins[a] += PRIOR_PSEUDO_TRIALS / 2
        wins[b] += PRIOR_PSEUDO_TRIALS / 2
        N[a, b] = N[b, a] = PRIOR_PSEUDO_TRIALS
    for w, l in pairs:
        i, j = idx[w], idx[l]
        wins[i] += 1
        N[i, j] += 1
        N[j, i] += 1
    g = np.ones(n)
    for _ in range(n_iter):
        denom = (N / (g[:, None] + g[None, :])).sum(axis=1)
        new = wins / denom
        new /= np.exp(np.log(new).mean())
        done = np.abs(new - g).max() < tol
        g = new
        if done:
            break
    return g


def preference_fit(pairs, items, n_boot: int = 500, seed: int = 0) -> dict:
    """MM Bradley–Terry on (winner, loser) pairs with a pair bootstrap. Output shape shared with the RK fit."""
    items = list(items)
    pairs = list(pairs)
    g = fit_bradley_terry(pairs, items)

    def p_matrix(s):
        return s[:, None] / (s[:, None] + s[None, :])

    P = p_matrix(g)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        samp = [pairs[i] for i in rng.integers(0, len(pairs), len(pairs))] if pairs else []
        boots.append(p_matrix(fit_bradley_terry(samp, items)))
    lo, hi = np.percentile(np.stack(boots), [2.5, 97.5], axis=0)
    return {
        "model": "bradley_terry_mm",
        "items": items,
        "n_pairs": len(pairs),
        "strength": dict(zip(items, g.tolist())),
        "p_beats": {a: {b: {"p": float(P[i, j]), "low": float(lo[i, j]), "high": float(hi[i, j])}
                        for j, b in enumerate(items) if j != i} for i, a in enumerate(items)},
    }


# ---- Rao–Kupper ties + rater reliability ---------------------------------------------------
@dataclass
class RKFit:
    s: np.ndarray          # log-strengths, mean 0
    theta: float           # tie band (0 = none)
    log_beta: np.ndarray   # per-rater log scale (empty when no raters)
    loglik: float          # unpenalised, on the fitted votes
    n_iter: int
    converged: bool


def _rk_negloglik(params, v: Votes, n_items: int, n_raters: int, ridge_s: float, ridge_b: float, weights):
    s = params[:n_items]
    tau = params[n_items]
    lb = params[n_items + 1:]
    theta = np.exp(tau)
    beta = np.ones(len(v)) if n_raters == 0 else np.where(v.r >= 0, np.exp(lb[np.maximum(v.r, 0)]), 1.0)
    x = beta * (s[v.a] - s[v.b])
    win, lose, tie = v.y > 0, v.y < 0, v.y == 0
    ll = np.empty(len(v))
    gx = np.empty(len(v))
    gt = np.empty(len(v))
    ll[win] = log_expit(x[win] - theta)
    gx[win] = expit(theta - x[win]); gt[win] = -gx[win]
    ll[lose] = log_expit(-x[lose] - theta)
    gx[lose] = -expit(x[lose] + theta); gt[lose] = gx[lose]
    if tie.any():
        xp, xm = x[tie] + theta, x[tie] - theta
        sp, sm = expit(xp), expit(xm)
        p_tie = np.maximum(sp - sm, 1e-300)
        ll[tie] = np.log(p_tie)
        dsp, dsm = sp * (1 - sp), sm * (1 - sm)
        gx[tie] = (dsp - dsm) / p_tie
        gt[tie] = (dsp + dsm) / p_tie
    w = weights if weights is not None else 1.0
    nll = -(w * ll).sum() + ridge_s * (s @ s) + ridge_b * (lb @ lb)
    gxw = w * gx
    gs = np.bincount(v.a, weights=gxw * beta, minlength=n_items) - np.bincount(v.b, weights=gxw * beta, minlength=n_items)
    grad = np.empty_like(params)
    grad[:n_items] = -gs + 2 * ridge_s * s
    grad[n_items] = -(w * gt).sum() * theta          # dθ/dτ = θ
    if n_raters:
        known = v.r >= 0
        grad[n_items + 1:] = -np.bincount(v.r[known], weights=(gxw * x)[known], minlength=n_raters) + 2 * ridge_b * lb
    return nll, grad


def fit_rao_kupper(v: Votes, ridge_s: float = RIDGE_S, ridge_b: float = RIDGE_LOG_BETA, raters: bool = True,
                   weights=None, x0: np.ndarray | None = None, maxiter: int = 500) -> RKFit:
    """L-BFGS fit of (s, τ, log β). `raters=False` or r all −1 → one global scale (absorbed into s)."""
    n_items = len(v.items)
    n_raters = len(v.raters) if raters and len(v.raters) and (v.r >= 0).any() else 0
    if x0 is None:
        x0 = np.zeros(n_items + 1 + n_raters)
        x0[n_items] = np.log(0.5) if (v.y == 0).any() else TAU_MIN
    bounds = [(None, None)] * n_items + [(TAU_MIN, 5.0)] + [(-4.0, 4.0)] * n_raters
    res = minimize(_rk_negloglik, x0, args=(v, n_items, n_raters, ridge_s, ridge_b, weights), jac=True,
                   method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8})
    s = res.x[:n_items] - res.x[:n_items].mean()
    theta = float(np.exp(res.x[n_items])) if res.x[n_items] > TAU_MIN + 1e-9 else 0.0
    lb = res.x[n_items + 1:]
    nll, _ = _rk_negloglik(res.x, v, n_items, n_raters, 0.0, 0.0, None)
    return RKFit(s, theta, lb, float(-nll), int(res.nit), bool(res.success))


def rk_probabilities(s_a, s_b, theta: float, beta: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P(a wins), P(b wins), P(tie)) for log-strength arrays."""
    x = beta * (np.asarray(s_a, dtype=float) - np.asarray(s_b, dtype=float))
    pa, pb = expit(x - theta), expit(-x - theta)
    return pa, pb, np.clip(1.0 - pa - pb, 0.0, 1.0)


def p_beats_matrix(s: np.ndarray, theta: float) -> np.ndarray:
    """P(a beats b) conditional on a decisive outcome (ties set aside), the quantity a gate compares."""
    pa, pb, _ = rk_probabilities(s[:, None], s[None, :], theta)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pa + pb > 0, pa / (pa + pb), 0.5)


def preference_fit_votes(v: Votes, n_boot: int = 200, seed: int = 0, raters: bool = True) -> dict:
    """RK fit with a vote bootstrap. Same `p_beats` shape as `preference_fit`, plus tie and rater terms."""
    fit = fit_rao_kupper(v, raters=raters)
    P = p_beats_matrix(fit.s, fit.theta)
    rng = np.random.default_rng(seed)
    boots, thetas = [], []
    n = len(v)
    for _ in range(n_boot):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float) if n else None   # weighted refit = resample
        b = fit_rao_kupper(v, raters=raters, weights=w, x0=np.concatenate([fit.s, [np.log(max(fit.theta, 1e-4))], fit.log_beta]))
        boots.append(p_beats_matrix(b.s, b.theta)); thetas.append(b.theta)
    if boots:
        lo, hi = np.percentile(np.stack(boots), [2.5, 97.5], axis=0)
    else:
        lo = hi = P
    items = v.items
    out = {
        "model": "rao_kupper_logistic" + ("_rater" if fit.log_beta.size else ""),
        "items": items,
        "n_votes": n,
        "n_ties": int((v.y == 0).sum()),
        "n_raters": int(fit.log_beta.size),
        "log_strength": dict(zip(items, fit.s.tolist())),
        "strength": dict(zip(items, np.exp(fit.s).tolist())),
        "tie_theta": fit.theta,
        "tie_theta_ci": [float(np.percentile(thetas, 2.5)), float(np.percentile(thetas, 97.5))] if thetas else None,
        "converged": fit.converged,
        "p_beats": {a: {b: {"p": float(P[i, j]), "low": float(lo[i, j]), "high": float(hi[i, j])}
                        for j, b in enumerate(items) if j != i} for i, a in enumerate(items)},
    }
    if fit.log_beta.size:
        beta = np.exp(fit.log_beta)
        counts = np.bincount(v.r[v.r >= 0], minlength=len(v.raters))
        out["rater_scale"] = {"p10": float(np.percentile(beta, 10)), "p50": float(np.percentile(beta, 50)),
                              "p90": float(np.percentile(beta, 90)), "share_below_0.5": float(np.mean(beta < 0.5)),
                              "votes_per_rater_p50": float(np.median(counts)) if counts.size else 0.0}
    return out


# ---- validation on held-out votes ----------------------------------------------------------
def predict_loglik(fit: RKFit, v: Votes, raters: bool = True) -> np.ndarray:
    """Per-vote log-likelihood under a fit; unknown raters (or raters=False) use β = 1."""
    beta = np.ones(len(v))
    if raters and fit.log_beta.size:
        known = (v.r >= 0) & (v.r < fit.log_beta.size)
        beta[known] = np.exp(fit.log_beta[v.r[known]])
    pa, pb, pt = rk_probabilities(fit.s[v.a], fit.s[v.b], fit.theta, beta)
    p = np.where(v.y > 0, pa, np.where(v.y < 0, pb, pt))
    return np.log(np.maximum(p, 1e-12))


def calibration(fit: RKFit, v: Votes, bins: int = 10) -> dict:
    """Decisive votes only: predicted P(a wins | decisive) vs the empirical share, in equal-count bins."""
    dec = v.y != 0
    if dec.sum() < bins:
        return {"n": int(dec.sum()), "bins": [], "expected_calibration_error": None}
    pa, pb, _ = rk_probabilities(fit.s[v.a[dec]], fit.s[v.b[dec]], fit.theta)
    p = pa / (pa + pb)
    won = (v.y[dec] > 0).astype(float)
    order = np.argsort(p)
    rows, ece = [], 0.0
    for chunk in np.array_split(order, bins):
        pred, obs = float(p[chunk].mean()), float(won[chunk].mean())
        rows.append({"n": int(chunk.size), "predicted": pred, "observed": obs})
        ece += chunk.size / dec.sum() * abs(pred - obs)
    return {"n": int(dec.sum()), "bins": rows, "expected_calibration_error": float(ece)}


def holdout_report(v: Votes, seed: int = 0, test_share: float = 0.2) -> dict:
    """Fit on a random 80 % of votes, score the rest: plain BT (ties as a no-tie model cannot score them,
    so decisive votes only), RK with ties, RK with ties and rater scales. Numbers per held-out vote."""
    rng = np.random.default_rng(seed)
    test = rng.random(len(v)) < test_share
    tr, te = v.subset(~test), v.subset(test)
    dec = te.y != 0
    out: dict = {"n_train": len(tr), "n_test": len(te), "tie_share_test": te.tie_share, "models": {}}
    fits = {"bradley_terry": fit_rao_kupper(tr.subset(tr.y != 0), raters=False),
            "rao_kupper": fit_rao_kupper(tr, raters=False)}
    if len(tr.raters) and (tr.r >= 0).any():
        fits["rao_kupper_rater"] = fit_rao_kupper(tr, raters=True)
    for name, f in fits.items():
        use_r = name.endswith("rater")
        ll = predict_loglik(f, te, raters=use_r)
        pa, pb, pt = rk_probabilities(f.s[te.a], f.s[te.b], f.theta)
        pred_a = pa > pb
        acc = float(np.mean(pred_a[dec] == (te.y[dec] > 0))) if dec.any() else None
        row = {"theta": f.theta, "decisive_accuracy": acc,
               "mean_loglik_decisive": float(ll[dec].mean()) if dec.any() else None,
               "predicted_tie_share": float(pt.mean())}
        if name != "bradley_terry":
            row["mean_loglik_all"] = float(ll.mean()) if len(te) else None
        out["models"][name] = row
    best = fits.get("rao_kupper_rater", fits["rao_kupper"])
    out["calibration"] = calibration(best, te)
    return out


def fold_consistency(v: Votes, n_folds: int = 5, n_boot: int = 100, seed: int = 0) -> dict:
    """Does the bootstrap interval tell the truth? Fit disjoint folds; the spread of the pairwise
    log-strength differences across folds is compared with the bootstrap spread on one fold."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, len(v))
    diffs = []
    for k in range(n_folds):
        f = fit_rao_kupper(v.subset(fold == k), raters=False)
        diffs.append(f.s[:, None] - f.s[None, :])
    diffs = np.stack(diffs)
    iu = np.triu_indices(len(v.items), 1)
    fold_sd = diffs.std(axis=0, ddof=1)[iu]
    v0 = v.subset(fold == 0)
    base = fit_rao_kupper(v0, raters=False)
    boots = []
    n = len(v0)
    for _ in range(n_boot):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        b = fit_rao_kupper(v0, raters=False, weights=w, x0=np.concatenate([base.s, [np.log(max(base.theta, 1e-4))]]))
        boots.append(b.s[:, None] - b.s[None, :])
    boot_sd = np.stack(boots).std(axis=0, ddof=1)[iu]
    ok = fold_sd > 0
    ratio = boot_sd[ok] / fold_sd[ok]
    return {"n_folds": n_folds, "n_pairs": int(iu[0].size), "median_boot_sd_over_fold_sd": float(np.median(ratio)) if ratio.size else None,
            "iqr": [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))] if ratio.size else None}


def fit_from_files(manifest_path, key_path, responses_path, n_boot: int = 500, seed: int = 0) -> dict:
    """Fit the study's responses per question. Plain MM Bradley–Terry when the CSV has no ties and no
    raters (unchanged behaviour); the Rao–Kupper fit when it has either."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    responses = read_responses(responses_path)
    out: dict = {"questions": {}, "metrics": {}}
    for q, v in votes_by_question(manifest, key, responses).items():
        if (v.y == 0).any() or len(v.raters):
            fit = preference_fit_votes(v, n_boot=min(n_boot, 200), seed=seed)
        else:
            dec = v.y != 0
            pairs = [(v.items[a], v.items[b]) if y > 0 else (v.items[b], v.items[a]) for a, b, y in zip(v.a[dec], v.b[dec], v.y[dec])]
            fit = preference_fit(pairs, v.items, n_boot, seed)
        out["questions"][q] = fit
        if "candidate" in v.items and "camera" in v.items:
            out["metrics"][f"pref_{q}_p_beats_camera_low"] = fit["p_beats"]["candidate"]["camera"]["low"]
    return out
