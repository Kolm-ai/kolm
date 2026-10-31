"""The burst layer (harness.model.burst): the warp, the zero-initialised fusion, the masks and the synthetic
renderer geometry."""
import numpy as np
import pytest
import torch

from harness.data.burststore import WARP_MARGIN
from harness.data.degrade import DegradeParams
from harness.model import burst
from harness.model.burst import BurstNet, crop_side, encode_t, render_burst, warp_frames
from harness.model.data import COND_DIM, METRIC_COND_DIM
from harness.model.net import CondUNet


def _smooth(side, seed=0):
    """A smooth positive linear image: a few low-frequency sinusoids per channel."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float64)
    out = np.empty((side, side, 3))
    for c in range(3):
        v = 0.3 + 0.0 * yy
        for _ in range(4):
            fy, fx = rng.uniform(-0.05, 0.05, 2)
            v += 0.05 * np.sin(2 * np.pi * (fy * yy + fx * xx) + rng.uniform(0, 6.28))
        out[..., c] = v
    return out


def _single(seed=0):
    torch.manual_seed(seed)
    u = CondUNet(base=16, mults=(1, 2), blocks=1, emb=32, noise_map=True)
    with torch.no_grad():
        u.head[-1].weight.normal_(0.0, 0.05)
        u.head[-1].bias.normal_(0.0, 0.05)
    return u.eval()


def _cond(n, known=1.0):
    c = torch.zeros(n, COND_DIM)
    c[:, METRIC_COND_DIM + 2] = 1.0
    c[:, METRIC_COND_DIM + 3] = 1.0
    c[:, METRIC_COND_DIM + 4] = 0.5
    c[:, METRIC_COND_DIM + 6] = known
    return c


def test_warp_identity_and_integer_shift():
    rng = np.random.default_rng(1)
    s, t = 24, 16
    fr = torch.from_numpy(rng.uniform(0, 1, (1, 2, 3, s, s)).astype(np.float32))
    pr, pc = np.mgrid[0:t, 0:t].astype(np.float32)
    co = np.zeros((1, 2, t, t, 2), dtype=np.float32)
    co[0, 0, ..., 0], co[0, 0, ..., 1] = pr + 4, pc + 4
    co[0, 1, ..., 0], co[0, 1, ..., 1] = pr + 2, pc + 7
    out = warp_frames(fr, torch.from_numpy(co))
    assert out.shape == (1, 2, 3, t, t)
    assert torch.allclose(out[0, 0], fr[0, 0, :, 4:4 + t, 4:4 + t], atol=1e-5)
    assert torch.allclose(out[0, 1], fr[0, 1, :, 2:2 + t, 7:7 + t], atol=1e-5)


def test_burstnet_at_init_is_the_single_net():
    u = _single()
    b = BurstNet(u, feat=8, frames=3).eval()
    x = encode_t(torch.rand(2, 3, 32, 32) * 0.5)
    c = _cond(2)
    fr = encode_t(torch.rand(2, 3, 3, 32, 32) * 0.5)
    m = torch.tensor([[True, True, False], [False, False, False]])
    with torch.no_grad():
        ref = u(x, c)
        assert torch.allclose(b(x, c, fr, m), ref, atol=1e-6)
        assert torch.allclose(b(x, c), ref, atol=1e-6)


def test_masked_frames_are_no_frames_and_fusion_learns():
    u = _single()
    b = BurstNet(u, feat=8, frames=3)
    with torch.no_grad():
        b.mix[-1].weight.normal_(0.0, 0.05)
        b.encoder.net[-1].weight.normal_(0.0, 0.05)
    b.eval()
    x = encode_t(torch.rand(1, 3, 32, 32) * 0.5)
    c = _cond(1)
    fr = encode_t(torch.rand(1, 3, 3, 32, 32) * 0.5)
    none = torch.zeros(1, 3, dtype=torch.bool)
    some = torch.tensor([[True, False, True]])
    with torch.no_grad():
        assert torch.allclose(b(x, c, fr, none), b(x, c), atol=1e-6)
        assert not torch.allclose(b(x, c, fr, some), b(x, c), atol=1e-4)
    b2 = BurstNet(_single(), feat=8, frames=3).train()
    loss = (b2(x, c, fr, some) - x).square().mean()
    loss.backward()
    assert float(b2.mix[-1].weight.grad.abs().sum()) > 0


def test_clean_static_burst_warps_back_to_the_reference():
    tile = 32
    big = _smooth(crop_side(tile))
    s = render_burst(big, DegradeParams(), np.random.default_rng(0), tile, 3, 2, motion=False)
    S = tile + 2 * WARP_MARGIN
    assert s["frames"].shape == (3, S, S, 3) and s["coords"].shape == (3, tile, tile, 2)
    assert list(s["mask"]) == [True, True, False]
    np.testing.assert_allclose(s["x"], s["y"], atol=1e-9)
    fr = torch.from_numpy(s["frames"]).permute(0, 3, 1, 2)[None]
    w = warp_frames(fr, torch.from_numpy(s["coords"])[None])[0, :2].permute(0, 2, 3, 1).numpy()
    for j in range(2):
        np.testing.assert_allclose(w[j], s["x"], atol=1e-5)


def test_moving_burst_geometry(monkeypatch):
    """With the motion on but no registration error, no moving rectangle and no gain jitter, every frame read at its
    coordinates lands on the reference: exactly with the renderer's own spline, and to within the bicubic kernel's
    interpolation error with the warp."""
    monkeypatch.setattr(burst, "REG_SIGMA_PX", 0.0)
    monkeypatch.setattr(burst, "P_MOVER", 0.0)
    monkeypatch.setattr(burst, "GAIN_JITTER", 0.0)
    tile = 48
    big = _smooth(crop_side(tile), seed=3)
    s = render_burst(big, DegradeParams(), np.random.default_rng(5), tile, 4, 4)
    fr = torch.from_numpy(s["frames"]).permute(0, 3, 1, 2)[None]
    w = warp_frames(fr, torch.from_numpy(s["coords"])[None])[0].permute(0, 2, 3, 1).numpy()
    off = np.abs(s["coords"] - (WARP_MARGIN + np.stack(np.mgrid[0:tile, 0:tile], -1))).max()
    assert off > 0.05
    from scipy.ndimage import map_coordinates
    for j in range(4):
        exact = np.stack([map_coordinates(s["frames"][j][..., c].astype(np.float64),
                                          np.moveaxis(s["coords"][j], -1, 0).astype(np.float64), order=3,
                                          mode="mirror") for c in range(3)], axis=-1)
        assert np.abs(exact - s["x"]).max() < 1e-4
        assert np.abs(w[j] - s["x"]).max() < 5e-3


def test_crop_side_too_small_raises():
    with pytest.raises(ValueError):
        render_burst(np.zeros((40, 40, 3)), DegradeParams(), np.random.default_rng(0), 32, 2, 1)
