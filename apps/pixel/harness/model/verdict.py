"""Run against run on the held-out sets, judged scene by scene: what a count of lost gates cannot tell.

A gate passed on the median of 64 tiles from 9 or 10 real scenes flips with the draw of scenes, and in a cascade one
failure stops its gate set, so a single flip reads as several losses (MODEL_PLAN section 20). Here every (held-out
set, metric) cell is judged on its own, on the statistic the gates use: verdict()'s signed margin between the two
runs' medians (positive when the candidate is better; for a to_truth metric, distance to the truth's median), and
for the tone gate's bands |log| of the base's median gain less the candidate's.

Inference resamples scenes, not tiles: tiles of one scene are not independent, so more tiles add little and more
scenes add much (the design effect; Field and Welsh 2007 on the cluster bootstrap). A scene is the raw.pixls.us file
or the real pair's scene. Per cell:
- a percentile bootstrap over scenes, the same draw for both runs (paired), and over seeds within each run when a
  run is a list of seeds (hierarchical: seed variance and test-set variance both enter);
- a sign-flip randomisation test: with no difference the two runs' labels are exchangeable scene by scene, so the
  statistic is recomputed with each scene's tiles swapped between the runs (every relabelling up to EXACT_MAX scenes,
  PERMS drawn above), the seeds averaged per tile. On 9 or 10 scenes the bootstrap interval is too narrow (it cannot
  see what the few scenes leave out), so a direction needs the test as well;
- the outcome against the metric's tie margin delta, a non-inferiority read of the 1 - 2 ALPHA interval (two one-
  sided tests; Walker and Nowacki 2011): "better" (interval above 0 and p < ALPHA), "non-inferior" (interval above
  -delta), "worse" (interval below 0, reaching past -delta, and p < ALPHA), else "undetermined"; `lower` marks a loss
  the sets resolve (interval below 0, p < ALPHA), inside the margin when the cell is non-inferior. A cell on fewer
  than FEW_SCENES scenes is flagged: three seeds and ten scenes resolve only differences of about three standard
  errors.
The headline pools the cells: the interquartile mean of margin / delta and the share of cells the candidate improves,
a tie counted half (Agarwal et al. 2021, "Deep RL at the edge of the statistical precipice"), each with a stratified
bootstrap (each set's scenes drawn on their own, every metric of a set from the same draw, the seeds once per draw).

Usage: CUDA_VISIBLE_DEVICES= PYTHONPATH=. HARNESS_DATA_ROOT=D:/lens-harness-data
python -X utf8 -m harness.model.verdict v18s0,v18s1,v18s2 v17 [--boot 2000] [--json out.json]; a baseline by its
name (nafnet, restormer, scunet), on the blind sets it is scored on."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import trim_mean

from ..data import data_root
from ..metrics import REGISTRY, STILL_METRICS
from .heldout import TONE_BANDS, TONE_MARGIN, _band

ALPHA = 0.05
BOOT = 2000
EXACT_MAX = 14          # scenes up to this: every one of the 2^n relabellings
PERMS = 20000           # relabellings drawn above it
FEW_SCENES = 20
TONE = "tone "          # a tone band's cell name: TONE + the band


def reports(run: str, root: Path | None = None) -> dict[str, dict]:
    """{set name: heldout.json} of a run's held-out reports, the pre-release ("_pre_") ones left out. A name with no
    run directory is read as a public baseline's (baselines.py: <models>/baseline_<name>/, the blind sets only)."""
    m = Path(root or data_root()) / "models"
    d = m / f"recovered_{run}"
    if not d.is_dir() and (m / f"baseline_{run}").is_dir():
        d = m / f"baseline_{run}"
    return {p.parent.name.removeprefix("heldout_"): json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(d.glob("heldout_*/heldout.json")) if "_pre_" not in p.parent.name}


def scenes(rep: dict, root: Path | None = None) -> list[str]:
    """each tile's scene: the raw.pixls.us file, or the real pair's scene from its stored set"""
    if not rep.get("real"):
        return [r["file"] for r in rep["records"]]
    from ..data.realpairs import set_path
    p = set_path(rep["source"], rep["seed"], rep["n"], rep["tile"], rep["scale"], root, bool(rep.get("subpix")))
    with np.load(p) as z:
        meta = json.loads(str(z["meta"]))
    if len(meta["records"]) != len(rep["records"]) or any(
            (a["pair"], a["y0"], a["x0"]) != (b["pair"], b["y0"], b["x0"]) for a, b in zip(meta["records"], rep["records"])):
        raise ValueError(f"{p} does not hold the report's tiles")
    return [meta["pairs"][r["pair"]]["scene"] for r in rep["records"]]


def _f(x) -> float:
    return float(x) if x is not None else np.nan


def cells(rep: dict) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """{cell name: (the output's value per tile, the truth's per tile or None)}; a tone band's values are the output's
    gains there (nan on a tile without the band)"""
    t = rep["tiles"]
    out = {m: (np.array([_f(r["output"].get(m)) for r in t]), np.array([_f(r["truth"].get(m)) for r in t]))
           for m in STILL_METRICS}
    for lo, hi in TONE_BANDS:
        k = _band(lo, hi)
        out[TONE + k] = (np.array([_f(r.get("tone", {}).get("output", {}).get(k)) for r in t]), None)
    return out


def delta(name: str) -> float:
    return TONE_MARGIN if name.startswith(TONE) else REGISTRY[name].tie_margin


def margin(name: str, c, b, t=None):
    """verdict()'s signed margin on medians (arrays allowed): positive when c is better than b"""
    c, b = np.asarray(c, dtype=float), np.asarray(b, dtype=float)
    if name.startswith(TONE):
        return np.abs(np.log(np.maximum(b, 1e-9))) - np.abs(np.log(np.maximum(c, 1e-9)))
    info = REGISTRY[name]
    if info.to_truth and t is not None and np.isfinite(t):
        return np.abs(b - t) - np.abs(c - t)
    return c - b if info.higher_is_better else b - c


def _med(a: np.ndarray, axis=-1):
    """nanmedian, quiet on an all-nan slice (nan)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def _improve(z: np.ndarray) -> float:
    return float(np.mean(z > 0) + 0.5 * np.mean(z == 0))


def _signflip(name: str, c: np.ndarray, b: np.ndarray, t, sid: np.ndarray, n_sc: int, rng) -> np.ndarray:
    """the statistic under scene relabellings (every one up to EXACT_MAX scenes, else PERMS drawn); c and b per tile"""
    if n_sc <= EXACT_MAX:
        flips = ((np.arange(2 ** n_sc)[:, None] >> np.arange(n_sc)) & 1).astype(bool)
    else:
        flips = rng.random((PERMS, n_sc)) < 0.5
        flips[0] = False
    sw = flips[:, sid]
    return margin(name, _med(np.where(sw, b, c)), _med(np.where(sw, c, b)), t)


def compare(cand: list[dict[str, dict]], base: list[dict[str, dict]], scene_of, boot: int = BOOT,
            seed: int = 0) -> dict:
    """cand and base: one {set: report} per seed of each run; scene_of(report) -> scene per tile. Every set all of
    them hold with the same tiles is judged."""
    rng = np.random.default_rng(seed)
    names = sorted(set.intersection(*(set(r) for r in cand + base)))
    sets, skipped = [], []
    for s in names:
        reps = [r[s] for r in cand + base]
        if any(r["records"] != reps[0]["records"] for r in reps[1:]):
            skipped.append(s)
            continue
        sc = scene_of(reps[0])
        uniq = sorted(set(sc))
        sid = np.array([uniq.index(x) for x in sc])
        by = [np.flatnonzero(sid == k) for k in range(len(uniq))]
        cv = [cells(r[s]) for r in cand]
        bv = [cells(r[s]) for r in base]
        for name in cv[0]:
            c = np.stack([v[name][0] for v in cv])
            b = np.stack([v[name][0] for v in bv])
            truth = cv[0][name][1]
            sets.append({"set": s, "metric": name, "c": c, "b": b, "truth": truth, "sid": sid, "by": by,
                         "scenes": len(uniq), "tiles": len(sc)})
    # the point estimates and the sign-flip tests
    out = []
    for k in sets:
        tm = None if k["truth"] is None else float(_med(k["truth"]))
        th = float(margin(k["metric"], _med(k["c"]).mean(), _med(k["b"]).mean(), tm))
        if not np.isfinite(th):
            continue
        perm = _signflip(k["metric"], k["c"].mean(0), k["b"].mean(0), tm, k["sid"], k["scenes"], rng)
        perm = perm[np.isfinite(perm)]
        exact = k["scenes"] <= EXACT_MAX
        hit = np.sum(perm >= th - 1e-12) if th >= 0 else np.sum(perm <= th + 1e-12)
        p = hit / perm.size if exact else (hit + 1) / (perm.size + 1)
        out.append({**{x: k[x] for x in ("set", "metric", "scenes", "tiles")}, "delta": delta(k["metric"]),
                    "margin": th, "p": float(p), "_k": k, "_tm": tm, "boot": []})
    # one stratified, hierarchical bootstrap for every cell and the pooled headline
    by_set = {}
    for o in out:
        by_set.setdefault(o["set"], o["_k"]["by"])
    nc, nb = len(cand), len(base)
    pooled = {"iqm": [], "improve": []}
    for _ in range(boot):
        ci, bi = rng.integers(nc, size=nc), rng.integers(nb, size=nb)
        idx = {s: np.concatenate([by[j] for j in rng.integers(len(by), size=len(by))]) for s, by in by_set.items()}
        z = []
        for o in out:
            k, ix = o["_k"], idx[o["set"]]
            tm = None if k["truth"] is None else float(_med(k["truth"][ix]))
            th = float(margin(k["metric"], _med(k["c"][ci][:, ix]).mean(), _med(k["b"][bi][:, ix]).mean(), tm))
            o["boot"].append(th)
            if np.isfinite(th):
                z.append(th / o["delta"])
        z = np.array(z)
        pooled["iqm"].append(trim_mean(z, 0.25))
        pooled["improve"].append(_improve(z))
    q = (100 * ALPHA, 100 * (1 - ALPHA))
    for o in out:
        bt = np.array(o.pop("boot"))
        bt = bt[np.isfinite(bt)]
        o.pop("_k"), o.pop("_tm")
        lo, hi = (float(v) for v in np.percentile(bt, q)) if bt.size else (np.nan, np.nan)
        o["lo"], o["hi"] = lo, hi
        shift = o["p"] < ALPHA and (lo > 0 or hi < 0)
        if shift and lo > 0:
            o["outcome"] = "better"
        elif lo > -o["delta"]:
            o["outcome"] = "non-inferior"
        elif shift:
            o["outcome"] = "worse"
        else:
            o["outcome"] = "undetermined"
        o["lower"] = bool(shift and hi < 0)      # a loss the sets resolve, inside the margin when non-inferior
        o["few"] = o["scenes"] < FEW_SCENES
    z = np.array([o["margin"] / o["delta"] for o in out])
    head = {"cells": len(out), "iqm": float(trim_mean(z, 0.25)), "improve": _improve(z)}
    for key in ("iqm", "improve"):
        head[key + "_lo"], head[key + "_hi"] = (float(v) for v in np.percentile(pooled[key], q))
    head["reading"] = ("better" if head["iqm_lo"] > 0 else "worse" if head["iqm_hi"] < 0 else "no clear change")
    return {"cells": out, "pooled": head, "skipped": skipped, "alpha": ALPHA, "boot": boot}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("candidate", help="run name, or seeds comma-separated (v18s0,v18s1,v18s2)")
    ap.add_argument("base", help="run name, or seeds comma-separated")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, help="also write the whole result here")
    a = ap.parse_args(argv)
    root = data_root()
    cand = [reports(r, root) for r in a.candidate.split(",")]
    base = [reports(r, root) for r in a.base.split(",")]
    res = compare(cand, base, lambda rep: scenes(rep, root), a.boot, a.seed)
    res["candidate"], res["base"] = a.candidate, a.base
    if a.json:
        a.json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    last = None
    for o in res["cells"]:
        if o["set"] != last:
            print(f"\n== {o['set']}: {o['scenes']} scenes, {o['tiles']} tiles" + ("  (few scenes)" if o["few"] else ""))
            last = o["set"]
        mark = {"better": "+", "worse": "-", "non-inferior": "~" if o["lower"] else " ", "undetermined": "?"}[o["outcome"]]
        print(f" {mark} {o['metric']:<26} {o['margin']:+.4g} [{o['lo']:+.4g}, {o['hi']:+.4g}] p {o['p']:.3f}"
              f"  delta {o['delta']:g}  {o['outcome']}" + (", lower" if o["lower"] and o["outcome"] != "worse" else ""))
    h = res["pooled"]
    print()
    for want, sel in (("worse", lambda o: o["outcome"] == "worse"), ("better", lambda o: o["outcome"] == "better"),
                      ("lower inside the margin", lambda o: o["lower"] and o["outcome"] == "non-inferior")):
        hit = [f"{o['set']}/{o['metric']}" for o in res["cells"] if sel(o)]
        print(f"{want}: " + (", ".join(hit) if hit else "none"))
    print(f"pooled over {h['cells']} cells: IQM of margin/delta {h['iqm']:+.3f} [{h['iqm_lo']:+.3f}, {h['iqm_hi']:+.3f}],"
          f" improves {h['improve']:.2f} of cells [{h['improve_lo']:.2f}, {h['improve_hi']:.2f}] -> {h['reading']}")
    if res["skipped"]:
        print("skipped (different tiles): " + ", ".join(res["skipped"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
