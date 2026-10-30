"""The loss-neutrality check (harness.model.neutrality) on a small CPU configuration: L1 alone is its own baseline,
an empty field leaves the output alone, and the one-sided terms the sixteenth and seventeenth runs paid for read as
what they did on the held-out sets."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from harness.model import neutrality as N
from harness.model import train as T

SMALL = dict(n=4, size=64, levels=(0.02,), steps=120, seed=0)


@pytest.fixture(autouse=True)
def _threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_recipe_is_trains_defaults_and_the_baseline_turns_every_term_off():
    r = N.recipe()
    assert r["range_weight"] == T.RANGE_WEIGHT and r["range_k"] == T.RANGE_K and r["flat_k"] == T.FLAT_K
    assert N.recipe(overshoot_weight=0.75)["overshoot_weight"] == 0.75
    with pytest.raises(ValueError):
        N.recipe(lr=1e-3)
    b = N.baseline(N.recipe(texture_weight=1.0, range_k=5))
    assert all(b[k] == 0.0 for k in N.WEIGHTS) and b["range_k"] == 5


def test_an_empty_field_leaves_the_output_alone():
    clean = N.tiles(2, 64, 0)
    bs = N.basis(clean)
    ps = N.params(2, 64, 64)
    assert torch.count_nonzero(N.field(ps, bs)) == 0
    pred0 = N._nchw(N.encode(clean))
    out = N.settle(N.baseline(N.recipe()), pred0, pred0, pred0, bs, steps=0)
    assert np.allclose(out, clean, atol=1e-5)


def test_l1_alone_is_neutral_by_construction():
    res = N.check(N.baseline(N.recipe()), **SMALL)
    assert res["pass"] and all(v == 0 for v in res["worst"].values())


def test_the_overshoot_term_darkens_mid_tones_on_a_noisy_output():
    """v16 (overshoot 1.5): the held-out shading and 0.2-0.5 tone losses, ahead of any training"""
    res = N.check(N.baseline(N.recipe()) | {"overshoot_weight": 1.5}, where=("output",), **SMALL)
    c = res["cases"][0]
    assert c["shading"] > 0.002 and c["tone"]["0.2-0.5"] < -0.005 and not res["pass"]


def test_the_texture_term_paints_a_noisy_targets_noise_back_as_shading():
    """the texture-deficit term matches the reference's local std, its noise included: on a real pair's noisy
    reference it pays the output to add low-frequency mottle"""
    res = N.check(N.baseline(N.recipe()) | {"texture_weight": 1.0}, where=("output", "target"), **SMALL)
    out, tgt = res["cases"]
    assert tgt["shading"] > N.limits()["shading"] > out["shading"]


def test_against_the_incumbent_is_paired_and_reads_only_the_realistic_levels():
    """v16 and v17 passed the absolute limits at s <= 0.01 and lost tone on the held-out sets; the paired change
    against v15 is what showed it"""
    base = N.baseline(N.recipe())
    small = SMALL | {"levels": (0.01, 0.02)}
    inc = N.check(base, where=("output",), **small)
    same = N.against(inc, inc)
    assert same["pass"] and all(v == 0 for v in same["worst"].values())
    assert [c["s"] for c in same["cases"]] == [0.01]
    darker = N.check(base | {"overshoot_weight": 1.5}, where=("output",), **small)
    rel = N.against(darker, inc)
    assert rel["cases"][0]["tone"]["0.2-0.5"] < -N.RELATIVE["tone"] and not rel["pass"]
    with pytest.raises(ValueError):
        N.against(darker, N.check(base, where=("output",), **(small | {"seed": 1})))
