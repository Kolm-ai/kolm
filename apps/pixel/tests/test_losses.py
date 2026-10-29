"""The fifteenth run's levers: MS-SSIM against its analytic values and the contrast it keeps (rho, where a
regression keeps rho^2), the shift search on real pairs, the noise-map channel, the schedule's warmup and floor,
the coherent-transfer term against the gate it mirrors (tilemetrics.texture_hf_fidelity), and a zero weight of
either term leaving pair_loss as it was."""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from harness.model.data import CAPTURE_DIM, COND_DIM, LOG_GAIN, METRIC_COND_DIM, encode
from harness.model.losses import (MS_WEIGHTS, SSIM_C1, TRANSFER_BIN_CPP, best_shift, ms_ssim,
                                  ssim_levels, overshoot_loss, pyramid_loss, ssim_loss, to_display, transfer_bands,
                                  transfer_loss)
from harness.model.net import CondUNet, noise_map
from harness.model.train import CLIP_LOG, CLIP_WEIGHT, WARMUP, lr_at, pair_loss


def test_ssim_levels_keep_a_full_window_at_the_coarsest():
    assert [ssim_levels(s) for s in (11, 21, 22, 64, 254, 256, 1024)] == [1, 1, 2, 3, 5, 5, 5]


def test_ms_ssim_of_an_image_with_itself_is_one():
    y = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(ms_ssim(y, y), torch.ones(2), atol=1e-5)
    t = torch.as_tensor(encode(y.numpy()))
    assert float(ssim_loss(t, t)) == pytest.approx(0.0, abs=1e-5)


def test_ms_ssim_of_two_flat_images_is_the_coarsest_luminance_term():
    """Flat images carry no contrast or structure, so every level's cs is 1 and MS-SSIM is the coarsest level's
    luminance term to the power of its renormalised weight (64 px: three levels)."""
    a, b = 0.3, 0.6
    got = float(ms_ssim(torch.full((1, 1, 64, 64), a), torch.full((1, 1, 64, 64), b)))
    w = MS_WEIGHTS[2] / sum(MS_WEIGHTS[:3])
    assert got == pytest.approx(((2 * a * b + SSIM_C1) / (a * a + b * b + SSIM_C1)) ** w, rel=1e-4)


def test_ms_ssim_keeps_the_contrast_a_regression_gives_up():
    """y is iid texture, x = y + noise of the same power, so rho = 0.707. Over outputs m + a (x - m) the squared
    error is least at a = rho^2 = 0.5 and MS-SSIM is greatest near a = rho: the term that keeps the input's
    detail where the regression shrinks it."""
    g = torch.Generator().manual_seed(1)
    y = 0.5 + 0.3 * torch.randn(8, 1, 64, 64, generator=g)
    x = y + 0.3 * torch.randn(8, 1, 64, 64, generator=g)
    m = x.mean(dim=(2, 3), keepdim=True)
    grid = np.linspace(0.3, 1.0, 71)
    ss = [float(ms_ssim(m + a * (x - m), y).mean()) for a in grid]
    l2 = [float(((m + a * (x - m) - y) ** 2).mean()) for a in grid]
    assert abs(grid[int(np.argmax(ss))] - math.sqrt(0.5)) < 0.06
    assert abs(grid[int(np.argmin(l2))] - 0.5) < 0.04


def test_ssim_loss_reads_the_display_rendering_and_has_a_finite_gradient():
    lin = torch.tensor([0.0, 0.002, 0.0031308, 0.18, 1.0, 3.0])
    d = to_display(torch.as_tensor(encode(lin.numpy())))
    assert d[0] == 0 and d[-1] == pytest.approx(1.0, abs=1e-4) and d[-2] == pytest.approx(1.0, abs=1e-4)
    assert d[3] == pytest.approx(1.055 * 0.18 ** (1 / 2.4) - 0.055, abs=1e-4)
    g = torch.Generator().manual_seed(2)
    y = torch.as_tensor(encode(torch.rand(2, 3, 32, 32, generator=g).numpy() * 0.5))
    pred = (y + 0.1 * torch.randn(y.shape, generator=g)).clamp_min(0).requires_grad_()
    loss = ssim_loss(pred, y)
    loss.backward()
    assert 0 < float(loss.detach()) < 1 and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


def test_best_shift_finds_a_real_sample_off_by_a_pixel_and_leaves_the_rest():
    g = torch.Generator().manual_seed(3)
    y = torch.rand(3, 3, 16, 16, generator=g)
    pred = y.clone()
    pred[0] = torch.roll(y[0], shifts=(1, -1), dims=(1, 2))    # real, a pixel off its reference each way
    pred[1] = torch.roll(y[1], shifts=(1, 0), dims=(1, 2))     # off too, but rendered: kept at zero shift
    pred.requires_grad_()
    x = torch.rand(3, 3, 16, 16, generator=g)
    real = torch.tensor([True, False, True])
    pc, yc, xc = best_shift(pred, y, x, real)
    assert pc.shape == yc.shape == xc.shape == (3, 3, 14, 14)
    assert torch.equal(yc[0], pc[0].detach()) and torch.equal(yc[2], pc[2].detach())
    assert torch.equal(yc[1], y[1, :, 1:15, 1:15]) and torch.equal(xc, x[..., 1:15, 1:15])
    (pc - yc).abs().mean().backward()
    assert pred.grad is not None and pred.grad[:, :, 0].abs().sum() == 0    # the crop takes no gradient
    none = best_shift(pred, y, x, torch.zeros(3, dtype=torch.bool))
    assert torch.equal(none[1], y[..., 1:15, 1:15])


def _cond(n, s, r, q, known=1.0):
    c = torch.zeros(n, COND_DIM)
    d = METRIC_COND_DIM
    c[:, d + 2], c[:, d + 3], c[:, d + 4] = math.log10(s) + 3, math.log10(r) + 4, math.log10(q) + 4
    c[:, d + 6] = known
    return c


def test_noise_map_follows_the_capture_noise_law():
    assert CAPTURE_DIM == 7
    v, s, r, q = 0.05, 0.004, 0.0008, 0.002
    x = torch.full((2, 3, 20, 20), float(encode(np.array([v]))[0]))
    got = noise_map(x, _cond(2, s, r, q))
    sig = math.sqrt(v * s * s + r * r + q * q / 12) * LOG_GAIN / (1 + LOG_GAIN * v)
    assert got.shape == (2, 1, 20, 20)
    assert torch.allclose(got, torch.full_like(got, math.log1p(sig)), rtol=1e-4)
    assert float(noise_map(x, _cond(2, s, r, q, known=0.0)).abs().max()) == 0.0


def test_noise_map_widens_the_stem_and_an_older_config_still_builds():
    m = CondUNet(base=8, mults=(1, 2), noise_map=True)
    assert m.stem.in_channels == 4 and m.config["noise_map"] is True
    x = torch.as_tensor(encode(np.random.default_rng(0).random((2, 3, 16, 16)) * 0.3)).float()
    assert m(x, _cond(2, 0.004, 0.0008, 0.002)).shape == x.shape
    old = {k: v for k, v in CondUNet(base=8, mults=(1, 2)).config.items() if k != "noise_map"}
    assert CondUNet(**old).stem.in_channels == 3                    # a fourteenth-run checkpoint loads as it was


def test_lr_at_defaults_unchanged_and_the_warmup_and_floor_are_kept():
    lr, steps = 1e-3, 10000
    for step in (0, 10, WARMUP - 1, WARMUP, 3000, 9999, 20000):
        t = min(1.0, max((step - WARMUP) / (steps - WARMUP), 0.0))
        old = lr * (step + 1) / WARMUP if step < WARMUP else lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))
        assert lr_at(step, steps, lr) == pytest.approx(old)
    assert lr_at(99, steps, lr, warmup=1000) == pytest.approx(lr * 0.1)
    assert lr_at(steps, steps, lr, warmup=1000, floor=0.003) == pytest.approx(lr * 0.003)
    assert lr_at(1000, steps, lr, warmup=1000, floor=0.003) == pytest.approx(lr)
    assert lr_at(5, steps, lr, spent=1.0, warmup=0, floor=0.01) == pytest.approx(lr * 0.01)


def test_pair_loss_is_unchanged_at_a_zero_ssim_weight_and_adds_it_otherwise():
    g = torch.Generator().manual_seed(4)
    y = torch.as_tensor(encode(torch.rand(2, 3, 32, 32, generator=g).numpy() * 0.6))
    x = y + 0.2 * torch.randn(y.shape, generator=g)
    pred = y + 0.1 * torch.randn(y.shape, generator=g)
    base = pair_loss(pred, y, x)
    assert torch.equal(base, pair_loss(pred, y, x, ssim_weight=0.0))
    assert float(pair_loss(pred, y, x, ssim_weight=0.5)) == pytest.approx(float(base + 0.5 * ssim_loss(pred, y)),
                                                                         rel=1e-6)


def _textured(seed=0, n=256):
    """A clean linear tile with texture in every band (smoothed noise, a quarter of it left at full bandwidth)."""
    from scipy.ndimage import gaussian_filter
    r = np.random.default_rng(seed)
    b = gaussian_filter(r.random((n, n)), 1.2) + 0.05 * r.random((n, n))
    y = 0.2 + 0.6 * (b - b.min()) / np.ptp(b)
    return np.repeat(y[..., None], 3, -1)


def _t(img):
    return torch.as_tensor(encode(img)).permute(2, 0, 1)[None].float()


def test_transfer_loss_is_zero_at_identity_and_under_a_pixel_shift():
    y = _textured()
    assert float(transfer_loss(_t(y), _t(y))) == pytest.approx(0.0, abs=1e-6)
    assert float(transfer_loss(_t(np.roll(y, (1, -1), axis=(0, 1))), _t(y))) < 2e-3   # the gate compensates it too


def test_transfer_loss_holds_both_sides_and_reads_the_texture_gate():
    """A blur and a sharpening by the same kernel sit at the same distance from a unit transfer; the bands from
    HF_CUTOFF_CPP up give tilemetrics' texture_hf_fidelity, the gate this term is held to."""
    from scipy.ndimage import gaussian_filter
    from harness.colorspace import luminance
    from harness.data.tilemetrics import _spectra, texture_hf_fidelity
    from harness.metrics.resolution import HF_CUTOFF_CPP
    y = _textured(1)
    lo = y - gaussian_filter(y, (1, 1, 0))
    blur, sharp = y - lo, np.clip(y + lo, 0, 1)
    lb, ls = float(transfer_loss(_t(blur), _t(y))), float(transfer_loss(_t(sharp), _t(y)))
    assert lb > 0.1 and ls > 0.1 and ls == pytest.approx(lb, rel=0.05)
    first = int(round(HF_CUTOFF_CPP / TRANSFER_BIN_CPP))
    for d in (blur, sharp):
        num, den = transfer_bands(_t(d), _t(y))
        gate = texture_hf_fidelity(*_spectra(luminance(np.clip(d, 0, 1)), luminance(y)))
        assert float(num[0, first:].sum() / den[0, first:].sum()) == pytest.approx(gate, rel=1e-3)


def test_transfer_loss_gradient_restores_the_fine_bands_of_a_blurred_prediction():
    from scipy.ndimage import gaussian_filter
    y = _textured(2)
    pred = _t(gaussian_filter(y, (1, 1, 0))).requires_grad_()
    Y = _t(y)
    opt = torch.optim.Adam([pred], lr=2e-3)
    before = transfer_bands(pred.detach(), Y)
    for _ in range(30):
        opt.zero_grad()
        transfer_loss(pred, Y).backward()
        opt.step()
    after = transfer_bands(pred.detach(), Y)
    hb, ha = before[0] / before[1], after[0] / after[1]
    first = int(round(0.25 / TRANSFER_BIN_CPP))
    assert float(ha[0, first:].mean()) > float(hb[0, first:].mean()) + 0.1
    assert float(transfer_loss(pred.detach(), Y)) < float(transfer_loss(_t(gaussian_filter(y, (1, 1, 0))), Y))


def test_transfer_loss_of_a_flat_tile_is_finite_and_zero():
    y = np.full((64, 64, 3), 0.3)
    pred = _t(y).requires_grad_()
    loss = transfer_loss(pred, _t(y))
    loss.backward()
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-6) and torch.isfinite(pred.grad).all()


def test_pair_loss_is_unchanged_at_a_zero_transfer_weight_and_adds_it_otherwise():
    g = torch.Generator().manual_seed(5)
    y = torch.as_tensor(encode(torch.rand(2, 3, 64, 64, generator=g).numpy() * 0.6))
    x = y + 0.2 * torch.randn(y.shape, generator=g)
    pred = y + 0.1 * torch.randn(y.shape, generator=g)
    base = pair_loss(pred, y, x)
    assert torch.equal(base, pair_loss(pred, y, x, transfer_weight=0.0))
    assert float(pair_loss(pred, y, x, transfer_weight=0.3)) == pytest.approx(
        float(base + 0.3 * transfer_loss(pred, y)), rel=1e-6)


def test_pair_loss_is_unchanged_at_zero_overshoot_and_pyramid_weights_and_adds_them_otherwise():
    g = torch.Generator().manual_seed(6)
    y = torch.as_tensor(encode(torch.rand(2, 3, 64, 64, generator=g).numpy() * 0.6))
    x = y + 0.2 * torch.randn(y.shape, generator=g)
    pred = y + 0.1 * torch.randn(y.shape, generator=g)
    base = pair_loss(pred, y, x)
    assert torch.equal(base, pair_loss(pred, y, x, overshoot_weight=0.0, pyramid_weight=0.0))
    w = torch.where((x >= CLIP_LOG).any(dim=1, keepdim=True), CLIP_WEIGHT, 1.0)
    assert float(pair_loss(pred, y, x, overshoot_weight=1.5, pyramid_weight=0.7)) == pytest.approx(
        float(base + 1.5 * overshoot_loss(pred, y) + 0.7 * pyramid_loss(pred, y, w)), rel=1e-6)


def test_transfer_loss_of_a_flat_target_under_a_noisy_output_stays_small():
    """A flat target (a sky, a wall) holds next to no power in its bins; the output's noise there is not a transfer
    of the target's texture (on 10-29 the term read 49 on such a tile, 0.4 on a textured one). The absolute floor
    keeps it near 0, a target with its own faint noise below a textured tile's, and the cap bounds any bin."""
    from harness.model.losses import TRANSFER_CAP
    rng = np.random.default_rng(3)
    for level, own, out in ((0.3, 1e-6, 3e-3), (0.002, 1e-7, 5e-4), (0.3, 1e-4, 3e-3)):
        y = np.full((128, 128, 3), level) + rng.normal(0.0, own, (128, 128, 1))
        pred = _t(y + rng.normal(0.0, out, (128, 128, 1))).requires_grad_()
        loss = transfer_loss(pred, _t(y))
        loss.backward()
        assert torch.isfinite(pred.grad).all() and float(loss.detach()) < (0.01 if own < 1e-5 else 0.4)
    y = np.clip(rng.uniform(0.05, 0.6, (128, 128, 1)) * np.ones(3), 0, 1)
    assert float(transfer_loss(_t(y * 0 + 0.3 + rng.normal(0, 0.05, y.shape)), _t(y))) <= TRANSFER_CAP


def _leaves(seed=0, n=128):
    from harness.model.deadleaves import dead_leaves
    return np.clip(dead_leaves(np.random.default_rng(seed), n), 0.0, 1.0) * 0.8


def test_overshoot_loss_tracks_edge_overshoot_and_is_zero_at_identity():
    """The term rises with a halo as edge_overshoot does. Its mean over the edge windows also charges what a blur
    carries into a window from beyond it, where the median of a mostly clean set of windows reads 0."""
    from scipy.ndimage import gaussian_filter
    from harness.colorspace import luminance
    from harness.data.tilemetrics import gain_matched
    from harness.metrics.natural import edge_overshoot_pair, unsharp
    y = _leaves()
    assert float(overshoot_loss(_t(y), _t(y))) == 0.0
    last, last_metric = 0.0, 0.0
    for gain in (0.1, 0.4, 1.0):
        d = np.clip(unsharp(y, gain, 1.0), 0.0, None)
        dg = gain_matched(np.clip(d, 0, 1), luminance(np.clip(d, 0, 1)), luminance(y))
        v, m = float(overshoot_loss(_t(d), _t(y))), edge_overshoot_pair(dg, y)
        assert v > last and m > last_metric
        last, last_metric = v, m
    d = gaussian_filter(y, (1.0, 1.0, 0))
    dg = gain_matched(d, luminance(d), luminance(y))
    assert edge_overshoot_pair(dg, y) == 0.0
    assert float(overshoot_loss(_t(d), _t(y))) > 0.01


def test_overshoot_loss_gradient_lowers_a_halo():
    from harness.metrics.natural import unsharp
    y = _leaves(1)
    pred = _t(np.clip(unsharp(y, 0.5, 1.0), 0.0, None)).requires_grad_()
    loss = overshoot_loss(pred, _t(y))
    loss.backward()
    assert torch.isfinite(pred.grad).all() and float(pred.grad.abs().sum()) > 0
    assert float(overshoot_loss(pred.detach() - 0.05 * pred.grad / pred.grad.abs().max(), _t(y))) < float(loss)


def test_overshoot_loss_gradient_moves_no_level():
    """On the target plus zero-mean noise, every pull the term makes is a bias, not a repair. Each window's gradient
    sums to zero over the encoded channels, so it carries no level at all and next to nothing at 8 px; what is left
    is a window's own contrast. v16's form (linear luminance, the gain divided out under no gradient) read a level
    of 0.73 of its gradient's weight, 0.73 at 8 px and a pull of 0.8 on the bands above 0.1."""
    from scipy.ndimage import gaussian_filter
    y = _leaves(1)
    rng = np.random.default_rng(2)
    d = np.clip(y + 0.01 * np.sqrt(y / 0.5) * rng.standard_normal(y.shape), 1e-4, 0.97)
    lum = d @ np.array([0.2126, 0.7152, 0.0722])
    band = (lum >= 0.1) & (lum < 0.97)
    pred = _t(d).requires_grad_()
    loss = overshoot_loss(pred, _t(y))
    loss.backward()
    assert float(loss) > 0
    g = pred.grad[0].numpy().astype(np.float64).sum(axis=0)
    assert abs(g.sum()) / np.abs(g).sum() < 1e-5
    assert np.abs(gaussian_filter(g, 8.0)).sum() / np.abs(g).sum() < 0.2
    assert g[band].sum() / np.abs(g[band]).sum() < 0.5


def test_pyramid_loss_charges_a_blotch_and_shading_more_than_noise_of_the_same_size():
    """At the same per-pixel L1, white noise averages out of the block means and a blotch or a shading ramp does
    not; a constant offset reads its own size at every scale, and the identity reads 0."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(2)
    y = _leaves(2, 256)
    white = y * np.exp(rng.normal(0, 0.02, y.shape[:2]))[..., None]
    b = gaussian_filter(rng.normal(0, 1, y.shape[:2]), 6)
    blotch = y * np.exp(b / b.std() * 0.02)[..., None]
    py, l1 = {}, {}
    for name, d in (("white", white), ("blotch", blotch)):
        py[name], l1[name] = float(pyramid_loss(_t(d), _t(y))), float((_t(d) - _t(y)).abs().mean())
    assert l1["blotch"] == pytest.approx(l1["white"], rel=0.05) and py["blotch"] > 3 * py["white"]
    t = _t(y)
    assert float(pyramid_loss(t, t)) == 0.0 and float(pyramid_loss(t + 0.01, t)) == pytest.approx(0.01, rel=1e-4)
    w = torch.full_like(t[:, :1], 0.25)
    assert float(pyramid_loss(t + 0.01, t, w)) == pytest.approx(0.0025, rel=1e-4)
