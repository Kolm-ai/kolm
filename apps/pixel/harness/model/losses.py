"""The fifteenth run's loss terms beside train.pair_loss's log-domain L1, each off unless a train flag names it:
MS-SSIM on the display rendering (--ssim-weight), the coherent-transfer term (--transfer-weight) and the shift
search on real pairs (--shift-real).

Why, from the thirteenth run's gates (2026-10-28). On the RawNIND real pairs its restorations' MTF50 is 0.292
cycles per pixel against the capture's 0.387, and texture fidelity 0.387 against 0.667: it smooths what the input
carries. A per-pixel regression toward a target the input fixes only up to noise answers each band with its
posterior mean, a coherent transfer of about rho^2 where rho is the band's input-target correlation (the Wiener
gain), and an L1 on finer bands or first differences shares that optimum. SSIM's contrast-structure term compares
local standard deviations and the correlation, not values: for an output a * x its optimum is a * std(x) = std(y),
a transfer of rho, so it keeps contrast where the regression gives it up. It is computed on the display rendering
(the sRGB OETF of the clipped linear image), the domain its constants C1 = 0.01^2 and C2 = 0.03^2 of a unit range
were set for.

It did not move the thirteenth run (2026-10-28, its training tiles): the best fine-band gain of its output under
the L1 plus the MS-SSIM term stayed at 0.7 to 0.9 at every weight. On the display rendering the fine bands'
local contrast sits under C2 (a std of 0.03 of the unit range), so the contrast term reads them as equal whatever
their gain. --ssim-weight stays at 0.

The coherent-transfer term holds the quantity the gates read. tilemetrics' mtf50_cpp and texture_hf_fidelity are
built on H(f) = Re<D C*> / <|C|^2> of the clipped linear luminances, the degraded tile shift-compensated, and are
judged by distance to the truth's value (1 and 0.5 cycles per pixel), so too sharp fails as too soft does.
transfer_bands computes the same sums per sample and radial bin, differentiably; transfer_loss is a Charbonnier
on 1 - H from TRANSFER_LO_CPP to Nyquist, each bin weighted by the target's share of power in it. Its bands from
HF_CUTOFF_CPP up give texture_hf_fidelity to 1e-3 (tests/test_losses.py). On the thirteenth run's RawNIND
training tiles its H over 0.17 to 0.42 cycles per pixel is 0.52 to 0.61, the captures' 0.83 to 0.94; at weight 1
the term's gradient is 2 to 7 times the L1's, and the best fine-band gain under the sum rises from 0.8 (the L1
alone) to 0.9, 1.1, 1.2 and 1.4 at weights 0.03, 0.1, 0.3 and 1.

The real pairs' references are registered to their captures at an integer shift for the whole frame
(data.realpairs.align, the quadrants within QUADRANT_TOL = 1 px of it), so a tile can sit a pixel off its
reference, and a regression's answer to a shift it cannot see is a blurred target. best_shift scores each real
sample at the integer shift within SHIFT_R px that fits it best, under no gradient, and crops SHIFT_R px from every
side of every sample; a rendered pair is exact and keeps the zero shift. Sub-pixel shifts would interpolate the
target, which blurs it."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import LOG_GAIN

SSIM_WIN, SSIM_SIGMA = 11, 1.5
SSIM_C1, SSIM_C2 = 0.01 ** 2, 0.03 ** 2
MS_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)   # Wang, Simoncelli and Bovik 2003, finest scale first
SSIM_FLOOR = 1e-4     # a level's mean below it is held there: the weighted product's gradient stays finite
SHIFT_R = 1           # px each way the shift search on a real sample reaches


def to_display(t: torch.Tensor) -> torch.Tensor:
    """An encoded (log-domain) tensor as display values: decoded, clipped to [0, 1], through the sRGB OETF. The
    power branch's input is held at the knee so its gradient below it stays finite."""
    v = (torch.expm1(t.clamp_min(0.0)) / LOG_GAIN).clamp(0.0, 1.0)
    hi = 1.055 * v.clamp_min(0.0031308) ** (1.0 / 2.4) - 0.055
    return torch.where(v <= 0.0031308, 12.92 * v, hi)


def _window(device) -> torch.Tensor:
    r = torch.arange(SSIM_WIN, device=device, dtype=torch.float32) - (SSIM_WIN - 1) / 2
    g = torch.exp(-r * r / (2 * SSIM_SIGMA ** 2))
    return g / g.sum()


def _blur(t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """The separable Gaussian over the valid region, per channel."""
    c = t.shape[1]
    t = F.conv2d(t, g.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    return F.conv2d(t, g.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def ssim_levels(side: int) -> int:
    """The MS-SSIM levels a tile of `side` px holds: the coarsest keeps at least one full window."""
    return max(1, min(len(MS_WEIGHTS), 1 + int(math.floor(math.log2(side / SSIM_WIN)))))


def ms_ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """MS-SSIM of display-domain (N, C, H, W) tensors in [0, 1], per sample (N,), the mean over channels: the
    contrast-structure means of the finer levels and the full SSIM mean of the coarsest, weighted by MS_WEIGHTS
    cut to the levels the tile holds and renormalised. In float32 whatever the inputs are."""
    a, b = a.float(), b.float()
    levels = ssim_levels(min(a.shape[-2:]))
    w = torch.tensor(MS_WEIGHTS[:levels], device=a.device, dtype=torch.float32)
    w = w / w.sum()
    g = _window(a.device)
    out = a.new_zeros(a.shape[:2])
    for i in range(levels):
        mu_a, mu_b = _blur(a, g), _blur(b, g)
        saa, sbb = _blur(a * a, g) - mu_a * mu_a, _blur(b * b, g) - mu_b * mu_b
        sab = _blur(a * b, g) - mu_a * mu_b
        cs = (2 * sab + SSIM_C2) / (saa + sbb + SSIM_C2)
        if i + 1 < levels:
            v = cs.mean(dim=(2, 3))
            a, b = F.avg_pool2d(a, 2), F.avg_pool2d(b, 2)
        else:
            v = ((2 * mu_a * mu_b + SSIM_C1) / (mu_a * mu_a + mu_b * mu_b + SSIM_C1) * cs).mean(dim=(2, 3))
        out = out + w[i] * torch.log(v.clamp_min(SSIM_FLOOR))
    return torch.exp(out).mean(dim=1)


def ssim_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """1 - MS-SSIM of the display renderings of encoded tensors, the batch mean."""
    return 1.0 - ms_ssim(to_display(pred), to_display(y)).mean()


def best_shift(pred: torch.Tensor, y: torch.Tensor, x: torch.Tensor, real: torch.Tensor,
               r: int = SHIFT_R) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """pred, y and x cropped by `r` px on every side, each real sample's y (`real`, a bool per sample) taken at the
    integer shift within `r` px whose mean absolute difference to pred is least; the others at zero shift. The
    search runs under no gradient; the crop is the same for every sample so the batch stays one tensor."""
    h, w = pred.shape[-2:]
    pc, xc = pred[..., r:h - r, r:w - r], x[..., r:h - r, r:w - r]
    y0 = y[..., r:h - r, r:w - r]
    if r < 1 or not bool(real.any()):
        return pc, y0, xc
    best = y0
    with torch.no_grad():
        p = pc.detach().float()
        best_err = (p - y0.float()).abs().mean(dim=(1, 2, 3))
        best_err = torch.where(real.to(best_err.device), best_err, torch.full_like(best_err, -1.0))
        pick = torch.zeros_like(best_err, dtype=torch.long)
        shifts = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy or dx]
        for k, (dy, dx) in enumerate(shifts, start=1):
            ys = y[..., r + dy:h - r + dy, r + dx:w - r + dx]
            e = (p - ys.float()).abs().mean(dim=(1, 2, 3))
            better = e < best_err
            pick = torch.where(better, torch.full_like(pick, k), pick)
            best_err = torch.where(better, e, best_err)
    if bool((pick > 0).any()):
        stack = torch.stack([y0] + [y[..., r + dy:h - r + dy, r + dx:w - r + dx] for dy, dx in shifts])
        best = stack[pick, torch.arange(pick.numel(), device=pick.device)]
    return pc, best, xc


TRANSFER_BIN_CPP = 1 / 32     # cycles/px: the radial bins of the transfer term (8 px at a 256 px tile)
TRANSFER_LO_CPP = 1 / 16      # bins below it are left out: the regression keeps them (v13's H there is ~1)
TRANSFER_FLOOR = 1e-3         # a bin's target power below this share of the tile's mean bin power reads as empty
TRANSFER_DELTA = 0.02         # the Charbonnier's knee on |1 - H|
TRANSFER_EMPTY_STD = 2e-4     # linear luminance: a bin holding less than white noise of this std reads as empty
TRANSFER_CAP = 3.0            # a bin's error saturates here (cap * tanh(err / cap)): no bin outweighs the tile
TRANSFER_LP_CPP = 0.06        # tilemetrics.LP_CPP: the phase correlation's low-pass
LUMA = (0.2126729, 0.7151522, 0.0721750)   # colorspace.luminance


def _luma(t: torch.Tensor) -> torch.Tensor:
    """Encoded (N, 3, H, W) as the gates read it: decoded, clipped to [0, 1], luminance (N, H, W)."""
    v = (torch.expm1(t.float().clamp_min(0.0)) / LOG_GAIN).clamp(0.0, 1.0)
    return v[:, 0] * LUMA[0] + v[:, 1] * LUMA[1] + v[:, 2] * LUMA[2]


def _phase_shift(P: torch.Tensor, Y: torch.Tensor, R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """tilemetrics.phase_shift per sample on windowed spectra: (dy, dx) of P against Y, the low-passed phase
    correlation's peak with a parabolic sub-pixel step, signed by wrap-around."""
    n, h, w = P.shape
    X = P * Y.conj()
    r = torch.fft.ifft2(X / X.abs().clamp_min(1e-12) * torch.exp(-(R / TRANSFER_LP_CPP) ** 2)).real
    k = r.flatten(1).argmax(dim=1)
    iy, ix = k // w, k % w
    b = torch.arange(n, device=P.device)
    m = r[b, iy, ix]
    out = []
    for idx, size, p, q in ((iy, h, r[b, (iy - 1) % h, ix], r[b, (iy + 1) % h, ix]),
                            (ix, w, r[b, iy, (ix - 1) % w], r[b, iy, (ix + 1) % w])):
        den = p - 2 * m + q
        d = idx.float() + torch.where(den < 0, 0.5 * (p - q) / torch.where(den < 0, den, -torch.ones_like(den)),
                                      torch.zeros_like(den))
        out.append(torch.where(d > size / 2, d - size, d))
    return out[0], out[1]


def transfer_bands(pred: torch.Tensor, y: torch.Tensor,
                   bin_cpp: float = TRANSFER_BIN_CPP) -> tuple[torch.Tensor, torch.Tensor]:
    """Per sample and radial bin, the gates' cross-spectral sums of encoded pred against encoded y: Re sum P Y* and
    sum |Y|^2 of the mean-subtracted, Hanning-windowed, clipped linear luminances, P compensated for its global
    translation against Y as tilemetrics._spectra compensates the degraded tile (the shift estimated under no
    gradient). Their ratio is the coherent transfer H(f) that mtf50_cpp and texture_hf_fidelity read. (N, bins)
    each, float32; the last bin takes the corners beyond Nyquist, as the gates' does."""
    lp, ly = _luma(pred), _luma(y)
    n, h, w = lp.shape
    dev = lp.device
    win = torch.outer(torch.hann_window(h, periodic=False, device=dev), torch.hann_window(w, periodic=False, device=dev))
    P = torch.fft.fft2((lp - lp.mean(dim=(1, 2), keepdim=True)) * win)
    Y = torch.fft.fft2((ly - ly.mean(dim=(1, 2), keepdim=True)) * win)
    fy, fx = torch.fft.fftfreq(h, device=dev)[:, None], torch.fft.fftfreq(w, device=dev)[None, :]
    R = torch.hypot(fy, fx)
    with torch.no_grad():
        dy, dx = _phase_shift(P.detach(), Y, R)
    P = P * torch.exp(2j * math.pi * (fy * dy[:, None, None] + fx * dx[:, None, None]))
    nb = int(round(0.5 / bin_cpp))
    idx = (R / bin_cpp).long().clamp_max(nb - 1).flatten()
    num = lp.new_zeros(n, nb).index_add_(1, idx, (P * Y.conj()).real.flatten(1))
    den = lp.new_zeros(n, nb).index_add_(1, idx, (Y.real ** 2 + Y.imag ** 2).flatten(1))
    return num, den


def _empty_power(h: int, w: int, bin_cpp: float, device) -> torch.Tensor:
    """Per radial bin (transfer_bands' bins), the sum of |Y|^2 a target of white noise of TRANSFER_EMPTY_STD
    gives under the Hanning window: the bin's frequencies times the window's energy times the variance."""
    fy, fx = torch.fft.fftfreq(h, device=device)[:, None], torch.fft.fftfreq(w, device=device)[None, :]
    nb = int(round(0.5 / bin_cpp))
    idx = (torch.hypot(fy, fx) / bin_cpp).long().clamp_max(nb - 1).flatten()
    count = torch.bincount(idx, minlength=nb).float()
    win = torch.outer(torch.hann_window(h, periodic=False, device=device),
                      torch.hann_window(w, periodic=False, device=device))
    return count * float((win * win).sum()) * TRANSFER_EMPTY_STD ** 2


def transfer_loss(pred: torch.Tensor, y: torch.Tensor, lo: float = TRANSFER_LO_CPP,
                  bin_cpp: float = TRANSFER_BIN_CPP) -> torch.Tensor:
    """The coherent-transfer term: per sample, the mean over the radial bins from `lo` to Nyquist of a Charbonnier
    on e = (<|Y|^2> - Re<P Y*>) / (<|Y|^2> + floor) = (1 - H) <|Y|^2> / (<|Y|^2> + floor), H = Re<P Y*> / <|Y|^2>
    the coherent transfer (transfer_bands) and floor TRANSFER_FLOOR of the tile's mean bin power plus the power
    white noise of TRANSFER_EMPTY_STD puts in the bin, so a bin the target leaves empty counts for little and the
    identity scores 0; each bin's error saturated at TRANSFER_CAP; the batch mean. Two-sided, as the gates judge
    texture and MTF50 by distance to the truth: it lifts a band the output smooths (H < 1) and lowers one it
    sharpens past the target (H > 1, the overshoot edge_overshoot reads). The absolute floor and the cap are from
    10-29: on a flat target (a sky, a wall, a clipped highlight) the relative floor alone left the bins' power near
    zero, so an output's noise there read as H in the thousands (the term at 49 on a flat grey tile with 0.3 %
    noise, against 0.4 on a textured one; a fine-tune's 50-step mean of it reached 63749)."""
    num, den = transfer_bands(pred, y, bin_cpp)
    keep = (torch.arange(num.shape[1], device=num.device) + 0.5) * bin_cpp >= lo
    num, den = num[:, keep], den[:, keep]
    empty = _empty_power(pred.shape[-2], pred.shape[-1], bin_cpp, num.device)[keep]
    floor = TRANSFER_FLOOR * den.mean(dim=1, keepdim=True) + empty[None]
    e = (den - num) / (den + floor)
    r = torch.sqrt(e * e + TRANSFER_DELTA ** 2) - TRANSFER_DELTA
    return (TRANSFER_CAP * torch.tanh(r / TRANSFER_CAP)).mean()


# The sixteenth run's artefact terms, each off unless a train flag names it (--overshoot-weight, --pyramid-weight).
# v14b on the HDR+ real pairs (2026-10-29, the residual noise reading, against the linear frontier at its own shadow
# SNR) overshoots edges by 0.021 of the step where the frontier has none, and its banding (0.0199 against 0.0164)
# and shading (0.0074 against 0.0062) read low-frequency residual. The range term charges values beyond the
# target's 3 x 3 range in the encoded domain, averaged over every pixel, so a halo of 2 % of an edge's step is
# diluted by the flat pixels around it; the overshoot term charges it as edge_overshoot does, as a share of the
# step, over the windows that hold an edge. A per-pixel L1 pays for a blotch in proportion to its per-pixel size,
# the same whatever its extent; the pyramid term pays for the residual's block means, where a blotch, a band and
# a shading error keep their size and noise averages out.
OVERSHOOT_HALF = 4        # metrics.natural.HALF_WINDOW: px each side of the window
OVERSHOOT_MIN_STEP = 0.1  # metrics.natural.MIN_STEP: linear-luminance step a window needs to hold an edge
OVERSHOOT_CLIP = 0.98     # metrics.natural.CLIP_GUARD
OVERSHOOT_GAIN_SIGMA = 8.0   # tilemetrics.GAIN_SIGMA_PX: the local gain divided out before the windows
PYRAMID_SCALES = (2, 4, 8, 16)   # block sides in px; tilemetrics.SHADING_BLOCK is 16


def _gauss(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """A separable Gaussian blur of (N, H, W), reflected at the borders."""
    r = int(math.ceil(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=t.device, dtype=t.dtype) / sigma) ** 2)
    k = k / k.sum()
    a = F.conv2d(F.pad(t[:, None], (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(a, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))[:, 0]


def overshoot_loss(pred: torch.Tensor, y: torch.Tensor, half: int = OVERSHOOT_HALF,
                   min_step: float = OVERSHOOT_MIN_STEP, clip: float = OVERSHOOT_CLIP) -> torch.Tensor:
    """metrics.natural.edge_overshoot_pair as a loss: on the gates' luminance (clipped linear), the output's local
    gain against the target's at OVERSHOOT_GAIN_SIGMA divided out (under no gradient, as tilemetrics.gain_matched),
    then per pixel and axis a (2 half + 1) px window gives the target's step and how far the output's maximum or
    minimum passes the target's, over the step; the mean over the batch's windows that hold an edge (step >=
    min_step) inside the clip guard, where the metric takes their median. Zero for the identity and for any output
    inside the target's local range, so a perfect restoration is still the optimum. The mean also charges what a
    blur carries into a window from beyond it (a 1 px Gaussian on dead leaves reads 0.025, the metric's median 0):
    it holds an edge from both sides, softening as well as ringing."""
    lp, ly = _luma(pred), _luma(y)
    with torch.no_grad():
        g = _gauss(ly, OVERSHOOT_GAIN_SIGMA) / _gauss(lp, OVERSHOOT_GAIN_SIGMA).clamp_min(1e-9)
    lp = lp * g
    size = 2 * half + 1
    total, count = lp.new_zeros(()), lp.new_zeros(())
    for ks, pad in (((1, size), (0, half)), ((size, 1), (half, 0))):
        def hi(a):
            return F.max_pool2d(a[:, None], ks, stride=1, padding=pad)[:, 0]
        rmax, rmin, pmax, pmin = hi(ly), -hi(-ly), hi(lp), -hi(-lp)
        step = rmax - rmin
        keep = ((step >= min_step) & (rmax < clip) & (rmin > 1.0 - clip)).to(lp.dtype)
        over = torch.relu(torch.maximum(pmax - rmax, rmin - pmin)) / step.clamp_min(1e-9)
        total, count = total + (over * keep).sum(), count + keep.sum()
    return total / count.clamp_min(1.0)


def pyramid_loss(pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None,
                 scales=PYRAMID_SCALES) -> torch.Tensor:
    """The mean over `scales` of the L1 of the encoded residual's s x s block means, each block weighted by its mean
    of `w` (pair_loss's clip weight; 1 without it). Encoded values are near-logarithmic, so a block mean's residual
    reads as the Weber contrast banding_score and the gain ratio shading_rmse read. Zero for the identity."""
    r = pred - y
    terms = []
    for s in scales:
        e = F.avg_pool2d(r, s).abs()
        terms.append((e if w is None else e * F.avg_pool2d(w.expand_as(r), s)).mean())
    return sum(terms) / len(terms)
