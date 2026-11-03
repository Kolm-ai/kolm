"""Public pairwise-preference vote sets -> the harness preference model at scale (DATA_PLAN.md section 4
step 3). Each reader turns one collected source into `Votes` over items (generator names, or images
where the set has no generator labels); `analyse` fits the Rao–Kupper tie-and-rater model, scores it
against plain Bradley–Terry on held-out votes, checks calibration and whether the bootstrap interval
is honest, and `preference_public` assembles the report the deriver writes.

The sets (all fetched by `python -m harness data collect`, see sources.json for ids and licences):

* Rapidata 700k (hf-rapidata-700k, column-projected): binary votes per model pair, no rater id but a
  per-vote `userScore` from the vendor, which is checked against the fitted model.
* open-image-preferences-v1-results (hf-open-image-preferences): two generators, rater ids, and
  explicit `both_good` / `both_bad` answers that the model treats as ties.
* GenAI-Bench image_generation (hf-genai-bench-image, projected): arena-style votes with tie and
  both-bad answers over 17 generators, no rater id.
* GenAI-Arena human eval (hf-genai-arena-human-eval): the same vote vocabulary, judge ids, 350 votes.
* HPDv2 test (hf-hpdv2-test): ten annotators each ranking every image set; the ranking is unrolled
  into pairs, so the items are images and the rater scale is identified from ~14k votes per rater.
  `annotation[i]` is read as the rank of `image_path[i]` (0 best); the inverse reading would give an
  equally consistent set, so nothing below depends on which is right.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..preference import (Votes, fit_rao_kupper, fold_consistency, holdout_report, preference_fit_votes,
                          rk_probabilities)

VOTE_WORDS = {"leftvote": "a", "rightvote": "b", "tievote": "tie", "bothbad_vote": "tie",
              "image_1": "a", "image_2": "b", "both_good": "tie", "both_bad": "tie"}
MAX_ITEMS_FOR_MATRIX = 40     # above this the p_beats matrix is left out of the report (images as items)


@dataclass
class VoteSet:
    name: str
    source_id: str
    item_kind: str                     # "model" | "image"
    votes: Votes
    notes: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    score: np.ndarray | None = None    # per-vote vendor reliability score where the set has one


def _spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return None
    from scipy.stats import spearmanr
    return float(spearmanr(x[ok], y[ok]).statistic)


# ---- readers ----------------------------------------------------------------------------
def read_rapidata_700k(root: Path) -> VoteSet | None:
    files = sorted(Path(root).glob("default/train_*/*.parquet"))
    if not files:
        return None
    recs, scores = [], []
    rows = matched_rows = unmatched_votes = 0
    for f in files:
        t = pq.read_table(f, columns=["votes_image1", "votes_image2", "model1", "model2", "detailed_results",
                                      "image1_path", "image2_path"]).to_pylist()
        for row in t:
            rows += 1
            try:
                votes = json.loads(row["detailed_results"]).get("votes", [])
            except (TypeError, ValueError):
                votes = []
            n_before = len(recs)
            for v in votes:
                who = v.get("votedFor")
                if who == row["image1_path"]:
                    recs.append((row["model1"], row["model2"], "a", None))
                elif who == row["image2_path"]:
                    recs.append((row["model1"], row["model2"], "b", None))
                else:
                    unmatched_votes += 1
                    continue
                scores.append(float((v.get("userDetails") or {}).get("userScore", np.nan)))
            if len(recs) - n_before == (row["votes_image1"] or 0) + (row["votes_image2"] or 0):
                matched_rows += 1
    v = Votes.from_records(recs)
    return VoteSet("rapidata_700k", "hf-rapidata-700k", "model", v,
                   notes=[f"{len(files)} shards, {rows} pairs"],
                   checks={"rows": rows, "rows_whose_vote_count_matches": matched_rows,
                           "share_rows_matching": matched_rows / rows if rows else None,
                           "unmatched_votes": unmatched_votes},
                   score=np.asarray(scores, float))


def read_open_image_preferences(root: Path) -> VoteSet | None:
    files = sorted(Path(root).glob("default/train/*.parquet"))
    if not files:
        return None
    recs, dropped, words = [], 0, {}
    for f in files:
        t = pq.read_table(f)
        cols = {name: t.column(name).to_pylist() for name in
                ("model_1", "model_2", "preference.responses", "preference.responses.users", "preference.responses.status")}
        for m1, m2, resp, users, status in zip(*cols.values()):
            for ans, user, st in zip(resp or [], users or [], status or []):
                words[ans] = words.get(ans, 0) + 1
                if st != "submitted" or ans not in VOTE_WORDS:
                    dropped += 1
                    continue
                recs.append((m1, m2, VOTE_WORDS[ans], user))
    v = Votes.from_records(recs)
    return VoteSet("open_image_preferences", "hf-open-image-preferences", "model", v,
                   notes=[f"{len(files)} shards"], checks={"answer_words": words, "dropped": dropped})


def _arena_rows(files, with_judge: bool) -> tuple[list, dict]:
    recs, words = [], {}
    cols = ["left_model", "right_model", "vote_type"] + (["judge"] if with_judge else [])
    for f in files:
        for row in pq.read_table(f, columns=cols).to_pylist():
            words[row["vote_type"]] = words.get(row["vote_type"], 0) + 1
            if row["vote_type"] in VOTE_WORDS:
                recs.append((row["left_model"], row["right_model"], VOTE_WORDS[row["vote_type"]],
                             str(row["judge"]) if with_judge and row.get("judge") else None))
    return recs, words


def read_genai_bench(root: Path) -> VoteSet | None:
    files = sorted(Path(root).glob("image_generation/test/*.parquet"))
    if not files:
        return None
    recs, words = _arena_rows(files, with_judge=False)
    return VoteSet("genai_bench_image", "hf-genai-bench-image", "model", Votes.from_records(recs),
                   notes=[f"{len(files)} shards"], checks={"vote_words": words})


def read_arena_human_eval(root: Path) -> VoteSet | None:
    files = sorted(Path(root).glob("battle/test/*.parquet"))
    if not files:
        return None
    recs, words = _arena_rows(files, with_judge=True)
    return VoteSet("genai_arena_human_eval", "hf-genai-arena-human-eval", "model", Votes.from_records(recs),
                   notes=[f"{len(files)} shards"], checks={"vote_words": words})


def read_hpdv2(root: Path) -> VoteSet | None:
    files = sorted(Path(root).glob("default/test/*.parquet"))
    if not files:
        return None
    recs, prompts, bad = [], 0, 0
    for f in files:
        for row in pq.read_table(f, columns=["image_path", "raw_annotations"]).to_pylist():
            prompts += 1
            paths = row["image_path"]
            for ann in row["raw_annotations"] or []:
                ranks = ann.get("annotation")
                if not ranks or len(ranks) != len(paths):
                    bad += 1
                    continue
                rater = str(ann.get("user_hash"))
                for i, j in combinations(range(len(paths)), 2):
                    if ranks[i] == ranks[j]:
                        recs.append((paths[i], paths[j], "tie", rater))
                    else:
                        recs.append((paths[i], paths[j], "a" if ranks[i] < ranks[j] else "b", rater))
    return VoteSet("hpdv2_test", "hf-hpdv2-test", "image", Votes.from_records(recs),
                   notes=[f"{prompts} prompts, rankings unrolled to pairs"], checks={"malformed_annotations": bad})


READERS = {
    "rapidata_700k": ("hf-rapidata-700k", read_rapidata_700k),
    "open_image_preferences": ("hf-open-image-preferences", read_open_image_preferences),
    "genai_bench_image": ("hf-genai-bench-image", read_genai_bench),
    "genai_arena_human_eval": ("hf-genai-arena-human-eval", read_arena_human_eval),
    "hpdv2_test": ("hf-hpdv2-test", read_hpdv2),
}


# ---- analysis ---------------------------------------------------------------------------
def _dist(x) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)), "min": float(x.min()), "max": float(x.max())}


def counts(v: Votes) -> dict:
    per_item = np.bincount(np.concatenate([v.a, v.b]), minlength=len(v.items))
    out = {"n_votes": len(v), "n_decisive": int((v.y != 0).sum()), "n_ties": int((v.y == 0).sum()),
           "tie_share": v.tie_share, "n_items": len(v.items), "votes_per_item": _dist(per_item),
           "n_raters": len(v.raters)}
    if len(v.raters):
        out["votes_per_rater"] = _dist(np.bincount(v.r[v.r >= 0], minlength=len(v.raters)))
    return out


def rater_check(v: Votes, fit_out: dict) -> dict | None:
    """Does the fitted rater scale track plain agreement with the consensus? Per rater: share of
    decisive votes on the fitted favourite, against β_r."""
    if not len(v.raters) or "rater_scale" not in fit_out:
        return None
    s = np.array([fit_out["log_strength"][it] for it in v.items])
    dec = (v.y != 0) & (v.r >= 0)
    fav = np.sign(s[v.a[dec]] - s[v.b[dec]])
    agree = (fav == v.y[dec]).astype(float)
    n = np.bincount(v.r[dec], minlength=len(v.raters))
    share = np.bincount(v.r[dec], weights=agree, minlength=len(v.raters)) / np.maximum(n, 1)
    beta = np.exp(np.array(fit_out["_log_beta"]))
    ok = n >= 20
    return {"raters_with_20_votes": int(ok.sum()), "agreement_with_favourite": _dist(share[ok]),
            "spearman_beta_vs_agreement": _spearman(beta[ok], share[ok]) if ok.sum() >= 4 else None,
            "beta": _dist(beta[ok])}


def score_check(vs: VoteSet, fit_out: dict) -> dict | None:
    """Rapidata's per-vote userScore against the fitted consensus: votes from higher-scored raters
    should land on the fitted favourite more often if the score means what it says."""
    if vs.score is None or not np.isfinite(vs.score).any():
        return None
    v = vs.votes
    s = np.array([fit_out["log_strength"][it] for it in v.items])
    dec = v.y != 0
    fav = np.sign(s[v.a] - s[v.b])
    agree = (fav == v.y).astype(float)
    sc = vs.score
    ok = dec & np.isfinite(sc) & (fav != 0)
    qs = np.percentile(sc[ok], [25, 50, 75])
    bins = np.digitize(sc[ok], qs)
    rows = [{"quartile": q + 1, "n": int((bins == q).sum()), "score_mean": float(sc[ok][bins == q].mean()),
             "agreement_with_favourite": float(agree[ok][bins == q].mean())} for q in range(4) if (bins == q).any()]
    return {"score": _dist(sc), "by_quartile": rows,
            "spearman_score_vs_agreement": _spearman(sc[ok], agree[ok])}


def analyse(vs: VoteSet, n_boot: int | None = None, seed: int = 0, log=None) -> dict:
    v = vs.votes
    say = log or (lambda m: None)
    nb = n_boot if n_boot is not None else (100 if len(v) <= 100_000 else 40)
    say(f"{vs.name}: {len(v)} votes over {len(v.items)} {vs.item_kind}s, {len(v.raters)} raters; fitting")
    fit_out = preference_fit_votes(v, n_boot=nb, seed=seed)
    fit = fit_rao_kupper(v)
    fit_out["_log_beta"] = fit.log_beta.tolist()
    out = {"source_id": vs.source_id, "item_kind": vs.item_kind, "notes": vs.notes, "checks": vs.checks,
           "counts": counts(v), "n_boot": nb}
    say(f"{vs.name}: theta {fit_out['tie_theta']:.3f}; held-out comparison")
    out["holdout"] = holdout_report(v, seed=seed)
    say(f"{vs.name}: fold consistency")
    out["fold_consistency"] = fold_consistency(v, n_boot=min(nb, 50), seed=seed)
    out["rater_check"] = rater_check(v, fit_out)
    out["score_check"] = score_check(vs, fit_out)
    trimmed = {k: val for k, val in fit_out.items() if not k.startswith("_")}
    if len(v.items) > MAX_ITEMS_FOR_MATRIX:
        for k in ("p_beats", "strength", "log_strength", "items"):
            trimmed.pop(k, None)
        trimmed["log_strength_dist"] = _dist(fit.s)
    else:
        order = np.argsort(-fit.s)
        trimmed["ranking"] = [v.items[i] for i in order]
    out["fit"] = trimmed
    return out


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def cross_set_overlap(results: dict) -> list[dict]:
    """Generator names shared between sets (normalised exact match) and whether the two fits order them
    the same way: sign agreement of log-strength differences over the shared pairs."""
    strengths = {name: {_norm(k): val for k, val in r["fit"].get("log_strength", {}).items()}
                 for name, r in results.items() if r["item_kind"] == "model" and "log_strength" in r["fit"]}
    rows = []
    for x, y in combinations(sorted(strengths), 2):
        shared = sorted(set(strengths[x]) & set(strengths[y]))
        if len(shared) < 2:
            rows.append({"sets": [x, y], "shared": shared, "pairs": 0})
            continue
        agree = total = 0
        for a, b in combinations(shared, 2):
            dx, dy = strengths[x][a] - strengths[x][b], strengths[y][a] - strengths[y][b]
            total += 1
            agree += int(np.sign(dx) == np.sign(dy))
        sx, sy = [strengths[x][k] for k in shared], [strengths[y][k] for k in shared]
        rows.append({"sets": [x, y], "shared": shared, "pairs": total, "sign_agreement": agree / total,
                     "spearman": _spearman(sx, sy) if len(shared) >= 4 else None})
    return rows


def verdict_lines(results: dict, overlap: list[dict]) -> list[str]:
    lines = []
    rater_gain, honesty = [], []
    for name, r in results.items():
        c, h = r["counts"], r["holdout"]["models"]
        bt, rk = h["bradley_terry"], h["rao_kupper"]
        line = (f"{name}: {c['n_votes']} votes over {c['n_items']} {r['item_kind']}s, tie share {c['tie_share']:.2f}, "
                f"theta {r['fit']['tie_theta']:.2f}; held-out decisive accuracy BT {bt['decisive_accuracy']:.3f} "
                f"RK {rk['decisive_accuracy']:.3f}")
        if "rao_kupper_rater" in h:
            rr = h["rao_kupper_rater"]
            d = rr["mean_loglik_all"] - rk["mean_loglik_all"]
            rater_gain.append(d)
            line += f", rater scales {'+' if d >= 0 else ''}{d:.4f} nats per vote"
        ece = r["holdout"]["calibration"].get("expected_calibration_error")
        if ece is not None:
            line += f", ECE {ece:.3f}"
        fc = r["fold_consistency"].get("median_boot_sd_over_fold_sd")
        if fc is not None:
            honesty.append(fc)
            line += f", bootstrap sd / fold sd {fc:.2f}"
        lines.append(line)
    with_ties = [n for n, r in results.items() if r["counts"]["tie_share"] > 0]
    if with_ties:
        lines.append("ties are a third outcome in " + ", ".join(with_ties) + "; plain Bradley–Terry cannot score "
                     "them at all, the Rao–Kupper band gives them a likelihood and leaves the decisive accuracy unchanged")
    if rater_gain:
        lines.append(f"per-rater scales change held-out log-likelihood by {min(rater_gain):+.4f} to {max(rater_gain):+.4f} "
                     f"nats per vote across {len(rater_gain)} sets with rater ids")
    if honesty:
        lines.append(f"bootstrap interval width over the between-fold spread: median {np.median(honesty):.2f} "
                     f"(1 = honest, below 1 = overconfident) across {len(honesty)} sets")
    shared = [o for o in overlap if o["pairs"] > 0]
    if shared:
        lines.append("; ".join(f"{o['sets'][0]} vs {o['sets'][1]} share {len(o['shared'])} generators, "
                               f"sign agreement {o['sign_agreement']:.2f}" for o in shared))
    else:
        lines.append("no two sets share enough generator names for a cross-set ordering check")
    return lines


def preference_public(raw_dir, only: list[str] | None = None, n_boot: int | None = None, log=None,
                      previous: dict | None = None) -> dict:
    """raw_dir: source id -> Path. Reads every set that is on disk, analyses each, and returns the report.
    With `only`, the other sets are carried over from `previous` (an earlier report) so a long set can be
    refitted alone; the overlap and verdict are recomputed over the merged sets."""
    results, missing = {}, []
    for name, (source_id, reader) in READERS.items():
        if only and name not in only:
            if previous and name in previous.get("sets", {}):
                results[name] = previous["sets"][name]
            elif previous and source_id in previous.get("missing_sources", []):
                missing.append(source_id)
            continue
        vs = reader(Path(raw_dir(source_id)))
        if vs is None or len(vs.votes) == 0:
            missing.append(source_id)
            continue
        results[name] = analyse(vs, n_boot=n_boot, log=log)
    overlap = cross_set_overlap(results) if results else []
    return {"sets": results, "missing_sources": missing, "cross_set_overlap": overlap,
            "verdict": verdict_lines(results, overlap) if results else ["no preference sets on disk"]}
