import numpy as np

from pixel import degrade


def test_recipe_is_deterministic_and_serialisable():
    r1 = degrade.recipe_from_seed(123)
    r2 = degrade.recipe_from_seed(123)
    assert degrade.recipe_to_json(r1) == degrade.recipe_to_json(r2)
    back = degrade.recipe_from_json(degrade.recipe_to_json(r1))
    assert [op for op, _ in back] == [op for op, _ in r1]
    assert degrade.seed_for("abc", 1) == degrade.seed_for("abc", 1) != degrade.seed_for("abc", 2)


def test_apply_is_deterministic_and_bounded(small_ref):
    recipe = [("noise", {"photons": 300.0}), ("jpeg", {"quality": 40})]
    a = degrade.apply(small_ref, recipe, seed=5)
    b = degrade.apply(small_ref, recipe, seed=5)
    assert np.array_equal(a, b)
    assert a.min() >= 0.0 and a.max() <= 1.0 and a.dtype == np.float32
    assert not np.array_equal(a, degrade.apply(small_ref, recipe, seed=6))


def test_direction_vector_layout():
    v = degrade.direction([("blur", {"sigma": 4.0}), ("jpeg", {"quality": 15}), ("saturation", {"scale": 0.5})])
    assert v.shape == (degrade.DIRECTION_DIM,)
    assert abs(v[degrade.FAMILIES.index("blur")] - 1.0) < 1e-6
    assert abs(v[degrade.FAMILIES.index("jpeg")] - 1.0) < 1e-6
    assert abs(v[degrade.FAMILIES.index("saturation")] + 1.0) < 1e-6
    assert v[degrade.FAMILIES.index("noise")] == 0.0


def test_every_family_samples_and_applies(small_ref):
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(60):
        recipe = degrade.sample_recipe(rng, max_ops=3)
        out = degrade.apply(small_ref[:128, :128], recipe, seed=1)
        assert out.shape == (128, 128, 3)
        seen.update(op for op, _ in recipe)
    assert seen == set(degrade.FAMILIES)


def test_label_pair_is_pure_function_of_source_and_index(small_ref):
    d1, r1, v1, s1 = degrade.label_pair(small_ref, "sha-x", 3)
    d2, r2, v2, s2 = degrade.label_pair(small_ref, "sha-x", 3)
    assert s1 == s2 and np.array_equal(d1, d2) and np.array_equal(v1, v2)
    d3, _r3, _v3, s3 = degrade.label_pair(small_ref, "sha-x", 4)
    assert s3 != s1 and not np.array_equal(d3, d1)
