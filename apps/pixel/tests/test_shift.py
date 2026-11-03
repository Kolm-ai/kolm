"""Shift check: the no-reference proxies behave on known content, the AIGCIQA2023 sheet parser
reads a minimal xlsx, and the summary statistics do what they say."""
import io
import math
import zipfile

import numpy as np
import pytest

from harness.data import shift
from harness.metrics.natural import (banding_nr, banding_pair, edge_overshoot_nr, edge_overshoot_pair,
                                     quantize_srgb, unsharp)
from harness.scene import Scene, SceneSpec

SMALL = SceneSpec(width=256, height=144, supersample=2, seed=3)


@pytest.fixture(scope="module")
def truth():
    return Scene(SMALL).render(0.0)


@pytest.fixture(scope="module")
def ramp():
    return np.linspace(0.02, 0.5, 512)[None, :, None] * np.ones((288, 1, 3))


def test_nr_banding_reads_one_code_on_a_clean_8bit_gradient(ramp):
    r = banding_nr(quantize_srgb(ramp, 8))
    assert 0.008 < r["step"] < 0.03           # the 8-bit floor, same scale as banding_pair
    assert r["band_area"] > 0.05 and r["n_edges"] > 100 and r["n_regions"] == 1


def test_nr_banding_tracks_posterisation_without_a_reference(ramp):
    q = quantize_srgb(ramp, 5)
    nr = banding_nr(q)
    pair = banding_pair(q, ramp)
    assert nr["step"] > 5 * banding_nr(quantize_srgb(ramp, 8))["step"]
    assert abs(nr["step_p95"] - pair) / pair < 0.3


def test_nr_banding_is_silent_on_noise_flat_and_texture(ramp):
    rng = np.random.default_rng(0)
    noisy = quantize_srgb(np.clip(ramp + rng.normal(0, 0.004, ramp.shape), 0, 1), 8)
    assert banding_nr(noisy)["n_edges"] < 20          # dither breaks the plateaus
    flat = np.full((300, 300, 3), 0.2)
    assert banding_nr(flat) == {"step": 0.0, "step_p95": 0.0, "band_area": 0.0, "n_edges": 0, "n_regions": 1}
    checker = (np.indices((300, 300)).sum(0) % 2).astype(float)[..., None] * np.ones(3)
    assert math.isnan(banding_nr(checker)["step"]) and banding_nr(checker)["n_regions"] == 0


def test_nr_halo_follows_the_pair_estimator(truth):
    assert edge_overshoot_nr(truth) == 0.0
    for gain, sigma in ((0.1, 1.0), (0.5, 1.0), (0.25, 1.5)):
        u = unsharp(truth, gain, sigma)
        nr, pair = edge_overshoot_nr(u), edge_overshoot_pair(u, truth)
        assert 0.5 * pair < nr < 1.1 * pair, (gain, sigma, nr, pair)


def _xlsx(rows):
    strings = sorted({c for row in rows for c in row})
    sst = "<sst>" + "".join(f"<si><t>{s}</t></si>" for s in strings) + "</sst>"
    body = ""
    for i, (a, b) in enumerate(rows, start=1):
        body += (f'<row r="{i}"><c r="A{i}" t="s"><v>{strings.index(a)}</v></c>'
                 f'<c r="B{i}" t="s"><v>{strings.index(b)}</v></c></row>')
    sheet = f'<worksheet><sheetData>{body}</sheetData></worksheet>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/sharedStrings.xml", sst)
        z.writestr("xl/worksheets/sheet1.xml", sheet)
    return buf.getvalue()


def test_pic_index_parser_reads_a_minimal_sheet(tmp_path):
    rows = [("Controlnet", "000-1.png"), ("Controlnet", "000-2.png"), ("DALLE", "000-1.png")]
    p = tmp_path / "pic-index.xlsx"
    p.write_bytes(_xlsx(rows))
    assert shift.read_pic_index(p) == rows


def test_summary_statistics():
    assert shift.spearman([1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9])["rho"] == pytest.approx(1.0)
    assert math.isnan(shift.spearman([1, 2, 3], [3, 2, 1])["rho"])                 # below MIN_N_RANK
    assert shift.spearman([1, float("nan"), 3] * 4, [1, 2, 3] * 4)["n"] == 8          # nan pairs dropped
    d = shift.distribution([0.0, 0.1, float("nan"), 0.2])
    assert d["n"] == 4 and d["n_defined"] == 3 and d["p50"] == pytest.approx(0.1)
    assert shift.fraction_above([0.0, 0.5, 1.0, float("nan")], 0.4) == pytest.approx(2 / 3)
    assert shift.fraction_above([0.1], None) is None


def test_summarise_on_synthetic_rows():
    rng = np.random.default_rng(1)
    pristine = [{"halo_nr": float(x), "band_step": 0.01, "band_area": 0.1} for x in rng.uniform(0, 0.05, 20)]
    ladders = []
    for ref in range(1, 11):
        for lv in range(1, 6):
            pair = 0.1 * lv
            ladders.append({"type": shift.TYPE_SHARPEN, "ref": ref, "level": lv, "dmos": 5 - 0.8 * lv,
                            "halo_nr": 0.7 * pair, "halo_pair": pair, "band_step": 0.0, "band_area": 0.0})
            ladders.append({"type": shift.TYPE_QUANTIZATION, "ref": ref, "level": lv, "dmos": 5 - 0.8 * lv,
                            "halo_nr": 0.0, "band_step": 0.02 * lv, "band_pair": 0.03 * lv, "band_area": 0.1 * lv})
    ai = [{"index": i, "generator": "G%d" % (i // 5), "halo_nr": 0.1 if i % 2 else 0.0, "band_step": 0.01, "band_area": 0.2,
           "mos_quality": 50.0 - 10 * (i % 2), "mos_authenticity": 50.0, "mos_correspondence": 50.0} for i in range(20)]
    gates = {"gates": {"edge_overshoot": {"value": 0.2, "support": {"supported": True}}}}
    s = shift.summarise(ai, {"pristine": pristine, "ladders": ladders}, gates)
    cal = s["ladders"]["high_sharpen_24"]["halo_nr_vs_pair"]
    assert cal["median_ratio"] == pytest.approx(0.7)
    assert s["gate_equivalents"]["halo_nr"]["nr_equivalent"] == pytest.approx(0.14)
    assert s["gate_equivalents"]["halo_nr"]["usable"] is True          # 0.14 is above the synthetic pristine p95
    assert s["ai"]["share_above_pristine_p95"]["halo_nr"] == pytest.approx(0.5)
    assert s["ai"]["share_above_halo_gate_equivalent"] == pytest.approx(0.0)
    assert s["ai"]["spearman_mos"]["halo_nr"]["quality"]["rho"] < -0.9          # halo images rated lower
    assert set(s["ai"]["per_generator"]) == {"G0", "G1", "G2", "G3"}
    assert len(shift.verdict_lines(s)) == 6
