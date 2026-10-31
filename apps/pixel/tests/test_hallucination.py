"""Invented-detail map: invariances (truth, blur, sharpening, noise, the best-case camera), labelled injections,
the declared rendered-pixel map, the rendered_map policy and the mask loader."""
import json

import numpy as np
import pytest
from scipy.ndimage import binary_dilation, gaussian_filter

from harness import hallucination as H
from harness.claim import default_claim
from harness.evaluate import build_world, evaluate
from harness.io import load_mask


@pytest.fixture(scope="module")
def world():
    claim = default_claim(video=True)
    claim["scene"].update(width=256, height=144, supersample=2, n_frames=3)
    return build_world(claim)


def _blur(frames, s):
    return np.stack([gaussian_filter(f, (s, s, 0)) for f in frames])


def _far(mask):
    out = ~binary_dilation(mask, iterations=10)
    out[:H.BORDER_PX] = out[-H.BORDER_PX:] = False
    out[:, :H.BORDER_PX] = out[:, -H.BORDER_PX:] = False
    return out


def test_truth_and_its_lossy_relatives_are_not_invented(world):
    T, S = world.truth, world.scene
    assert H.fraction(H.invented_maps(T, T, S)) == 0.0
    for s in (1.0, 2.0, 3.0):
        assert H.fraction(H.invented_maps(_blur(T, s), T, S)) == 0.0, s
    usm = np.clip(T + (T - _blur(T, 1.0)), 0.0, None)
    assert H.fraction(H.invented_maps(usm, T, S)) == 0.0
    rng = np.random.default_rng(1)
    for level in (0.02, 0.05):
        noisy = np.clip(T + rng.standard_normal(T.shape) * level * (T + 0.01), 0.0, None)
        assert H.fraction(H.invented_maps(noisy, T, S)) == 0.0, level


def test_blur_match_recovers_the_blur(world):
    t = world.truth[0]
    for s in (1.0, 2.5):
        _, sigma = H.match_blur(H.luminance(gaussian_filter(t, (s, s, 0))), H.luminance(t))
        assert sigma == pytest.approx(s, abs=0.25)


def test_camera_is_far_below_the_gate(world):
    infos = []
    flag = H.invented_maps(world.camera, world.truth, world.scene, infos)
    assert H.fraction(flag) < 0.01
    assert all(0.0 < i["matched_blur_px"] <= 1.5 for i in infos)  # the camera blurs, a little
    assert all(i["noise_shape"] >= H.KAPPA_WHITE for i in infos)


def test_motion_exclusion_covers_only_movers(world):
    ex = H.motion_exclusion(world.scene, 1)
    y0 = world.scene.regions["stage"][0]
    assert 0.0 < ex.mean() < 0.2
    assert not ex[:y0].any()


@pytest.mark.parametrize("base", ["truth", "camera"])
@pytest.mark.parametrize("region", ["colors", "texture", "stage"])
def test_injected_detail_is_found_where_it_was_planted(world, base, region):
    T, S = world.truth, world.scene
    frames = T if base == "truth" else world.camera
    rng = np.random.default_rng(3)
    recall, far = [], []
    for k in range(T.shape[0]):
        f, m = H.inject(frames[k], rng, n_patches=3, size_px=32, contrast=0.2, region=S.regions[region])
        flag = H.invented_map(f, T[k])
        recall.append((flag & m).sum() / m.sum())
        outside = _far(m)
        far.append((flag & outside).sum() / outside.sum())
    assert np.mean(recall) > 0.5, recall
    assert np.mean(far) < 0.02, far


def test_replaced_texture_is_invented(world):
    claim = json.loads(json.dumps(world.claim))
    claim["scene"]["seed"] = int(claim["scene"].get("seed", 0)) + 101
    other = build_world(claim)
    y0, y1, x0, x1 = world.scene.regions["texture"]
    swapped = world.truth.copy()
    swapped[:, y0:y1, x0:x1] = other.truth[:, y0:y1, x0:x1]
    flag = H.invented_maps(swapped, world.truth, world.scene)
    assert H.fraction(flag, world.scene.regions["texture"]) > 0.9
    assert H.fraction(flag, world.scene.regions["colors"]) == 0.0


def test_declared_rendered_map_clears_undeclared(world, tmp_path):
    T, S = world.truth, world.scene
    rng = np.random.default_rng(5)
    frames, masks = [], []
    for k in range(T.shape[0]):
        f, m = H.inject(T[k], rng, n_patches=4, size_px=32, contrast=0.2, region=S.regions["colors"])
        frames.append(f)
        masks.append(binary_dilation(m, iterations=14))  # the label core is the window above half height; declare the window
    cand, declared = np.stack(frames), np.stack(masks)
    flag = H.invented_maps(cand, T, S)
    assert H.fraction(flag) > 0.0
    assert H.fraction(H.undeclared(flag, declared)) < 0.1 * H.fraction(flag)
    np.save(tmp_path / "map.npy", declared)
    rep = evaluate(world, cand, str(tmp_path / "out"), "cand", rendered_map=str(tmp_path / "map.npy"))
    m = rep["metrics"]
    assert m["rendered_fraction"]["candidate"] == pytest.approx(H.fraction(declared))
    assert m["undeclared_invention_fraction"]["candidate"] < m["invented_detail_fraction"]["candidate"]
    assert rep["provenance"]["map_supplied"] is True
    assert len(rep["provenance"]["per_frame"]) == T.shape[0]
    assert (tmp_path / "out" / "invented_detail.npy").exists()


def test_required_policy_without_map_is_incomplete(world, tmp_path):
    claim = json.loads(json.dumps(world.claim))
    claim["rendered_map"]["policy"] = "required"
    w = build_world(claim)
    rep = evaluate(w, w.truth, str(tmp_path / "out"), "truth")
    assert rep["metrics"]["undeclared_invention_fraction"]["candidate"] is None
    assert rep["overall"].startswith("INCOMPLETE") and "provenance" in rep["overall"]


def test_load_mask_broadcasts_and_checks_shape():
    m = np.zeros((16, 24), dtype=bool)
    m[4:8, 6:12] = True
    out = load_mask(m, (3, 16, 24))
    assert out.shape == (3, 16, 24) and out[2, 5, 7] and not out[0, 0, 0]
    with pytest.raises(ValueError):
        load_mask(m, (3, 16, 25))
