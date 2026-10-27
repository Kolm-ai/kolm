"""Blinded pairwise preference: manifests, Bradley-Terry fit, bootstrap intervals.

The repo's JS Bradley-Terry fitter (src/bradley-terry.js) is the same model;
this one keeps the image pipeline single-language and adds the bootstrap. A
preference result is only admissible when the manifest was generated here
(blinded, side-balanced, order-randomised) and the key stayed sealed until
after the votes were in.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np


def make_manifest(candidate_ids: list[str], rounds: int = 1, seed: int = 0) -> tuple[list[dict], dict]:
    """Every unordered pair ``rounds`` times, random side assignment, shuffled order.

    Returns (presentations, key). ``presentations`` hides identities behind
    'left'/'right'; ``key`` maps presentation id -> {left: id, right: id}.
    """
    rng = np.random.default_rng(seed)
    pairs = list(itertools.combinations(sorted(set(candidate_ids)), 2)) * max(1, rounds)
    order = rng.permutation(len(pairs))
    presentations, key = [], {}
    for n, i in enumerate(order):
        a, b = pairs[i]
        if rng.random() < 0.5:
            a, b = b, a
        pid = f"p{n:05d}"
        presentations.append({"id": pid, "questions": ["more_beautiful", "more_convincing"]})
        key[pid] = {"left": a, "right": b}
    return presentations, key


def unblind(votes: list[dict], key: dict, question: str = "more_beautiful") -> list[dict]:
    """Votes are {id, question, choice: 'left'|'right'|'tie'} -> BT pairs {a, b, pref}."""
    out = []
    for v in votes:
        if v.get("question", question) != question:
            continue
        k = key[v["id"]]
        choice = v["choice"]
        pref = "tie" if choice == "tie" else ("a" if choice == "left" else "b")
        out.append({"a": k["left"], "b": k["right"], "pref": pref})
    return out


def bradley_terry(pairs: list[dict], ridge: float = 1e-3, max_iter: int = 2000, tol: float = 1e-8) -> dict:
    """MLE of centred logit skills; ties contribute half a win each way."""
    items = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    idx = {it: i for i, it in enumerate(items)}
    n = len(items)
    if n == 0:
        return {}
    A = np.array([idx[p["a"]] for p in pairs])
    B = np.array([idx[p["b"]] for p in pairs])
    W = np.array([1.0 if p["pref"] == "a" else 0.0 if p["pref"] == "b" else 0.5 for p in pairs])
    theta = np.zeros(n)
    step = 1.0
    for _ in range(max_iter):
        d = theta[A] - theta[B]
        p = 1.0 / (1.0 + np.exp(-d))
        g = np.zeros(n)
        np.add.at(g, A, W - p)
        np.add.at(g, B, p - W)
        g -= ridge * theta
        if np.abs(g).max() < tol:
            break
        theta = theta + step * g / max(1.0, len(pairs) / n)
        theta -= theta.mean()
    return {it: float(theta[idx[it]]) for it in items}


def bootstrap(pairs: list[dict], n_boot: int = 500, seed: int = 0) -> dict:
    """Percentile 95% intervals on each skill and on each pairwise skill gap."""
    rng = np.random.default_rng(seed)
    base = bradley_terry(pairs)
    items = sorted(base)
    samples = {it: [] for it in items}
    gaps = {(a, b): [] for a, b in itertools.combinations(items, 2)}
    pairs_arr = np.array(pairs, dtype=object)
    for _ in range(n_boot):
        resampled = list(pairs_arr[rng.integers(0, len(pairs), size=len(pairs))])
        th = bradley_terry(resampled)
        for it in items:
            samples[it].append(th.get(it, 0.0))
        for a, b in gaps:
            gaps[(a, b)].append(th.get(a, 0.0) - th.get(b, 0.0))
    out = {
        "theta": base,
        "ci95": {it: [float(np.percentile(samples[it], 2.5)), float(np.percentile(samples[it], 97.5))] for it in items},
        "gaps": {},
    }
    for (a, b), g in gaps.items():
        lo, hi = float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))
        out["gaps"][f"{a} - {b}"] = {"mean": float(np.mean(g)), "ci95": [lo, hi], "significant": (lo > 0) or (hi < 0)}
    return out


def summarize(pairs: list[dict], candidate: str, reference: str, n_boot: int = 500, seed: int = 0) -> dict:
    res = bootstrap(pairs, n_boot=n_boot, seed=seed)
    key = f"{min(candidate, reference)} - {max(candidate, reference)}"
    gap = res["gaps"].get(key)
    if gap is None:
        return {"status": "no_data", "theta": res["theta"]}
    sign = 1.0 if candidate < reference else -1.0
    lo, hi = sorted([sign * gap["ci95"][0], sign * gap["ci95"][1]])
    return {
        "status": "beats" if lo > 0 else "loses" if hi < 0 else "inconclusive",
        "gap_logit": sign * gap["mean"],
        "ci95": [lo, hi],
        "n_pairs": len(pairs),
        "theta": res["theta"],
        "ci95_theta": res["ci95"],
    }


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
