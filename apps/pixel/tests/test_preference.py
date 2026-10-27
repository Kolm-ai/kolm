import numpy as np

from pixel.preference import bootstrap, bradley_terry, make_manifest, summarize, unblind


def _simulate(theta: dict, rounds: int, seed: int = 0):
    pres, key = make_manifest(list(theta), rounds=rounds, seed=seed)
    rng = np.random.default_rng(seed + 1)
    votes = []
    for p in pres:
        a, b = key[p["id"]]["left"], key[p["id"]]["right"]
        pa = 1.0 / (1.0 + np.exp(-(theta[a] - theta[b])))
        votes.append({"id": p["id"], "question": "more_beautiful", "choice": "left" if rng.random() < pa else "right"})
    return pres, key, votes


def test_manifest_is_blinded_and_balanced():
    pres, key = make_manifest(["cam", "gen", "truth"], rounds=20, seed=3)
    assert len(pres) == 60 and all("left" not in p for p in pres)
    sides = [k["left"] for k in key.values()]
    assert 8 <= sides.count("gen") <= 32  # 'gen' is in 40 pairs; random side, not fixed


def test_bradley_terry_recovers_order():
    theta = {"cam": -1.0, "gen": 0.5, "truth": 1.5}
    _pres, key, votes = _simulate(theta, rounds=120)
    pairs = unblind(votes, key)
    fit = bradley_terry(pairs)
    assert fit["cam"] < fit["gen"] < fit["truth"]
    res = bootstrap(pairs, n_boot=100, seed=0)
    assert res["gaps"]["cam - truth"]["significant"]
    s = summarize(pairs, "gen", "cam", n_boot=100)
    assert s["status"] == "beats" and s["ci95"][0] > 0


def test_ties_and_inconclusive():
    pairs = [{"a": "x", "b": "y", "pref": "tie"}] * 10 + [{"a": "x", "b": "y", "pref": "a"}, {"a": "x", "b": "y", "pref": "b"}]
    s = summarize(pairs, "x", "y", n_boot=50)
    assert s["status"] == "inconclusive"
