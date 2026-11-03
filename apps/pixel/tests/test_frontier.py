"""The oracle linear frontier (harness.model.frontier): the Wiener ladder fitted on the truth, its cache, the match
at the output's own shadow SNR and FRONTIER_GATES. A rung of the ladder ties itself; over-smoothing loses detail."""
import json

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from harness.data.tilemetrics import tile_metrics
from harness.model import frontier as F

N, n = 384, 128
LAMS = (0.0, 0.1, 1.0, 8.0)


def _scene():
    rng = np.random.default_rng(0)
    _, x = np.mgrid[0:N, 0:N] / N
    base = np.exp(np.log(0.002) * (1 - x) + np.log(0.05) * x)                        # shadows across the scene
    tex = gaussian_filter(rng.standard_normal((N, N)), 0.8, mode="wrap")
    y = base * np.clip(1 + 0.4 * tex / tex.std(), 0.05, None)
    y[:, N // 2:N // 2 + 40] *= 3.0                                                     # an edge
    y = y[..., None] * np.array([1.0, 0.9, 1.1])
    out = []
    for a, b in ((0, 0), (0, 128), (128, 64), (256, 200)):
        t = y[a:a + n, b:b + n]
        xb = gaussian_filter(t, (0.7, 0.7, 0))
        out.append({"x": xb + rng.normal(0, 0.0006, t.shape), "y": t, "xb": xb})
    return out


SAMPLES = _scene()


def _mse(a, b):
    return float(np.mean((a - b) ** 2))


@pytest.fixture(scope="module")
def table():
    return F.ladder(SAMPLES, "residual")


def test_the_mmse_rung_denoises_and_the_inverse_rung_deblurs():
    s = SAMPLES[0]
    assert _mse(F.wiener(s["x"], s["y"], 1.0), s["y"]) < 0.7 * _mse(s["x"], s["y"])
    assert _mse(F.wiener(s["xb"], s["y"], 0.0), s["y"]) < 0.05 * _mse(s["xb"], s["y"])   # no noise: undoes the blur


def test_the_lowest_band_is_kept_so_the_mean_stays():
    s = SAMPLES[1]
    for lam in (0.0, 1.0, 32.0):
        g = F.gains(s["x"], s["y"], lam)
        assert g.shape == (3, F.NB) and np.all(g[:, 0] == 1.0) and np.all((g >= 0) & (g <= F.G_MAX))
        assert np.allclose(F.wiener(s["x"], s["y"], lam).mean((0, 1)), s["x"].mean((0, 1)), rtol=1e-9, atol=0)


def test_snr_rises_and_detail_falls_along_the_ladder(table):
    assert table["lams"] == list(F.LAMS) and table["noise"] == "residual" and table["version"] == F.VERSION
    assert len(table["rows"]) == len(F.LAMS) and all(len(r) == len(SAMPLES) for r in table["rows"])
    med = {m: np.array([F._median(t[m] for t in r) for r in table["rows"]])
           for m in ("shadow_snr_db", "texture_hf_fidelity", "mtf50_cpp")}
    assert np.all(np.diff(med["shadow_snr_db"]) > 0) and med["shadow_snr_db"][-1] > med["shadow_snr_db"][0] + 20
    assert np.all(np.diff(med["texture_hf_fidelity"]) < 0) and np.all(np.diff(med["mtf50_cpp"]) <= 0)


def test_a_rung_of_the_ladder_ties_itself(table):
    truths = [tile_metrics(s["y"], s["y"], "residual") for s in SAMPLES]
    k = F.LAMS.index(1.0)
    g = F.gates(table, table["rows"][k], truths)
    assert g["overall"] == "PASS" and g["match"]["rung"] == pytest.approx(k) and g["match"]["clamped"] == 0
    assert g["match"]["set_median_match"] == 0 and g["match"]["unmatched"] == 0 and g["match"]["tiles"] == len(SAMPLES)
    assert g["metrics"]["shadow_snr_db"]["verdict"] is None                           # matched, so not judged
    assert all(c["verdict"] in ("ties", None) for c in g["metrics"].values())
    assert [x["name"] for x in g["gates"]] == [x["name"] for x in F.FRONTIER_GATES]


def test_over_smoothing_loses_detail_to_the_frontier(table):
    truths = [tile_metrics(s["y"], s["y"], "residual") for s in SAMPLES]
    outs = [tile_metrics(gaussian_filter(F.wiener(s["x"], s["y"], 1.0), (1.5, 1.5, 0)), s["y"], "residual")
            for s in SAMPLES]
    g = F.gates(table, outs, truths)
    assert g["overall"].startswith("FAIL at")
    assert g["metrics"]["texture_hf_fidelity"]["verdict"] == "loses" and g["metrics"]["mtf50_cpp"]["verdict"] == "loses"


def test_a_tile_without_its_own_snr_takes_the_set_median_match(table):
    k = F.LAMS.index(1.0)
    outs = [dict(r) for r in table["rows"][k]]
    outs[2]["shadow_snr_db"] = float("nan")
    base, info = F.matched(table, outs)
    assert info["set_median_match"] == 1 and info["clamped"] == 0 and all(np.isfinite(b["mtf50_cpp"]) for b in base)
    none = [dict(r, shadow_snr_db=float("nan")) for r in table["rows"][k]]             # nothing to place them by
    base, info = F.matched(table, none)
    assert info["unmatched"] == len(SAMPLES) and info["rung"] is None and all(b["mtf50_cpp"] is None for b in base)
    assert F.gates(table, none, none)["overall"].startswith("INCOMPLETE")
    off = [dict(r, shadow_snr_db=r["shadow_snr_db"] + 100.0) for r in table["rows"][k]]   # above every rung
    assert F.matched(table, off)[1]["clamped"] == len(SAMPLES) and F.matched(table, off)[1]["rung"] == len(F.LAMS) - 1


def test_matched_wants_one_output_per_tile(table):
    with pytest.raises(ValueError, match="the ladder has 4 tiles, the outputs 3"):
        F.matched(table, table["rows"][0][:3])


def test_the_ladder_is_cached_per_set_reading_and_ladder(tmp_path, monkeypatch):
    logged = []
    t = F.cached_ladder(SAMPLES[:2], tmp_path, "residual", LAMS, log=logged.append)
    files = list((tmp_path / "derived" / F.FRONTIER_DIR).glob("*.json"))
    assert len(files) == 1 and files[0].stem == F.key(SAMPLES[:2], "residual", LAMS) and logged
    assert t["lams"] == list(LAMS) and len(t["rows"]) == len(LAMS)
    monkeypatch.setattr(F, "ladder", lambda *a, **k: pytest.fail("scored again"))
    assert json.dumps(F.cached_ladder(SAMPLES[:2], tmp_path, "residual", LAMS, log=None)) == json.dumps(t)  # NaN-safe
    keys = {F.key(SAMPLES[:2], "residual", LAMS), F.key(SAMPLES[:2], "hp", LAMS), F.key(SAMPLES[:2], "residual"),
            F.key(SAMPLES[1:3], "residual", LAMS),
            F.key([dict(SAMPLES[0], x=SAMPLES[0]["x"] + 1e-6), SAMPLES[1]], "residual", LAMS)}
    assert len(keys) == 5                                       # the reading, the ladder and every tile are in the key
