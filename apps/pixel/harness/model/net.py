"""Conditioned residual UNet: the recovered layer's network. Input and output are 3-channel tiles in the
log domain; the output is input plus a learned residual. The conditioning vector (the target state's
normalised metric vector plus its validity mask) enters every residual block as FiLM scale and shift, so
one network serves restore and edit directions. Fully convolutional: any size that is a multiple of 2 ** (levels - 1).
It scales by width (`base`), by level (`mults`), and by depth per level (`blocks`, one count or one per level, shared
by the encoder and decoder at that level) and at the bottleneck (`mid_blocks`): a block at the lowest level costs
4 ** (levels - 1) times fewer FLOPs per parameter than one at full resolution, so the depth goes there. `resample`
picks how levels change scale: "learned" (a strided conv down, a stride-2 transposed conv up; every run up to v15) or
"smooth" (BlurDown and SmoothUp, which leave no grid of the levels' strides in the output)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .data import CAPTURE_DIM, COND_DIM, LOG_GAIN, METRIC_COND_DIM

NOISE_POOL = 5   # px: the window of the local level the noise map reads the signal-dependent shot noise at
RESAMPLE = ("learned", "smooth")
BINOMIAL = (1.0, 2.0, 1.0)   # BlurDown's low-pass, per axis


def noise_map(x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """The input's expected noise standard deviation per pixel in the encoded domain, one channel (N, 1, H, W), from
    the capture descriptor in `cond` (data.capture_descriptor: shot sigma at the clip s, read noise r and one LSB q,
    all relative to the clip) and the input's local level v (its decoded NOISE_POOL x NOISE_POOL mean): the variance
    v s^2 + r^2 + q^2 / 12 as a standard deviation, times the encoding's slope LOG_GAIN / (1 + LOG_GAIN v), the
    channel mean, as log1p of it; zero where the descriptor is withheld (known = 0). The descriptor enters every block
    as one global FiLM vector; this is the same law laid out per pixel, the map a non-blind denoiser reads (FFDNet,
    Zhang et al. 2018), so the network need not learn to multiply a global noise level by the local signal."""
    d = cond[:, METRIC_COND_DIM:METRIC_COND_DIM + CAPTURE_DIM].float()
    s, r, q, known = 10.0 ** (d[:, 2] - 3.0), 10.0 ** (d[:, 3] - 4.0), 10.0 ** (d[:, 4] - 4.0), d[:, 6]
    v = F.avg_pool2d(torch.expm1(x.float().clamp_min(0.0)) / LOG_GAIN, NOISE_POOL, 1, NOISE_POOL // 2,
                     count_include_pad=False)
    var = v * (s * s).view(-1, 1, 1, 1) + (r * r + q * q / 12.0).view(-1, 1, 1, 1)
    sig = var.sqrt() * LOG_GAIN / (1.0 + LOG_GAIN * v)
    return torch.log1p(sig.mean(dim=1, keepdim=True)) * known.view(-1, 1, 1, 1)


class FiLM(nn.Module):
    def __init__(self, emb: int, ch: int):
        super().__init__()
        self.proj = nn.Linear(emb, 2 * ch)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(e)[:, :, None, None].chunk(2, dim=1)
        return h * (1.0 + scale) + shift


class ResBlock(nn.Module):
    def __init__(self, ch: int, emb: int):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, ch), nn.Conv2d(ch, ch, 3, padding=1)
        self.film = FiLM(emb, ch)
        self.n2, self.c2 = nn.GroupNorm(8, ch), nn.Conv2d(ch, ch, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        h = self.film(h, e)
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class Level(nn.Module):
    def __init__(self, ch: int, emb: int, blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList([ResBlock(ch, emb) for _ in range(blocks)])
        self.grad_ckpt = False

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = checkpoint(b, x, e, use_reentrant=False) if self.grad_ckpt and self.training else b(x, e)
        return x


class BlurDown(nn.Module):
    """Anti-aliased downsampling (Zhang 2019, "Making convolutional networks shift-invariant again"): a [1 2 1]^2 / 16
    binomial low-pass per channel on reflect padding, then the strided 3 x 3 conv. A bare strided conv aliases, so
    what the level below sees changes with the tile's phase against the stride, and the decoder hands that grid on
    to the output."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        k = torch.tensor(BINOMIAL)
        k = (k[:, None] * k[None, :]) / k.sum() ** 2
        self.register_buffer("k", k.expand(cin, 1, 3, 3).clone(), persistent=False)   # rebuilt, never in a checkpoint
        self.conv = nn.Conv2d(cin, cout, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), self.k.to(x.dtype), groups=x.shape[1]))


class SmoothUp(nn.Module):
    """Upsampling without a checkerboard (Odena et al. 2016, "Deconvolution and checkerboard artifacts"): a 1 x 1 conv
    to the finer level's width, then bilinear x2. A stride-2 transposed conv gives each of the four output phases its
    own kernel: a pattern on the level's 2 px grid, 4, 8 and 16 px once the levels below add theirs."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(self.conv(x), scale_factor=2, mode="bilinear", align_corners=False)


class CondUNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 48, mults: tuple = (1, 2, 4, 4),
                 blocks: int | tuple = 2, cond_dim: int = COND_DIM, emb: int = 256, mid_blocks: int | None = None,
                 noise_map: bool = False, resample: str = "learned"):
        super().__init__()
        per = (int(blocks),) * len(mults) if isinstance(blocks, int) else tuple(int(b) for b in blocks)
        if len(per) != len(mults) or min(per) < 1 or (mid_blocks is not None and mid_blocks < 1):
            raise ValueError(f"blocks wants one count >= 1 or one per level ({len(mults)}), got {blocks}, mid {mid_blocks}")
        if resample not in RESAMPLE:
            raise ValueError(f"resample wants one of {RESAMPLE}, got {resample!r}")
        self.config = dict(in_ch=in_ch, out_ch=out_ch, base=base, mults=tuple(mults),
                           blocks=blocks if isinstance(blocks, int) else per, cond_dim=cond_dim, emb=emb,
                           mid_blocks=mid_blocks, noise_map=bool(noise_map), resample=resample)
        self.noise_map = bool(noise_map)   # one more stem channel: noise_map(x, cond)
        self.embed = nn.Sequential(nn.Linear(cond_dim, emb), nn.SiLU(), nn.Linear(emb, emb), nn.SiLU())
        chs = [base * m for m in mults]
        self.stem = nn.Conv2d(in_ch + int(self.noise_map), chs[0], 3, padding=1)
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        for i, c in enumerate(chs):
            self.down.append(Level(c, emb, per[i]))
            if i + 1 < len(chs):
                self.pool.append(BlurDown(c, chs[i + 1]) if resample == "smooth"
                                 else nn.Conv2d(c, chs[i + 1], 3, stride=2, padding=1))
        self.mid = Level(chs[-1], emb, per[-1] if mid_blocks is None else mid_blocks)
        self.up, self.merge = nn.ModuleList(), nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            self.up.append(SmoothUp(chs[i], chs[i - 1]) if resample == "smooth"
                           else nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
            self.merge.append(nn.Conv2d(2 * chs[i - 1], chs[i - 1], 1))
            self.down.append(Level(chs[i - 1], emb, per[i - 1]))   # decoder levels appended after the encoder ones
        self.head = nn.Sequential(nn.GroupNorm(8, chs[0]), nn.SiLU(), nn.Conv2d(chs[0], out_ch, 3, padding=1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.n_levels = len(chs)

    def set_grad_ckpt(self, on: bool = True) -> "CondUNet":
        """Recompute each residual block's activations in the backward pass instead of storing them: the same
        gradients for about a third more compute and a fraction of the activation memory."""
        for m in self.modules():
            if isinstance(m, Level):
                m.grad_ckpt = bool(on)
        return self

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        e = self.embed(cond)
        h = self.stem(torch.cat([x, noise_map(x, cond)], dim=1) if self.noise_map else x)
        skips = []
        for i in range(self.n_levels):
            h = self.down[i](h, e)
            if i + 1 < self.n_levels:
                skips.append(h)
                h = self.pool[i](h)
        h = self.mid(h, e)
        for j in range(self.n_levels - 1):
            h = self.up[j](h)
            h = self.merge[j](torch.cat([h, skips.pop()], dim=1))
            h = self.down[self.n_levels + j](h, e)
        return x + self.head(h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def pad_to(x: torch.Tensor, mult: int = 8) -> tuple[torch.Tensor, tuple[int, int]]:
    """Reflect-pad H and W up to a multiple of `mult`; returns the padding to strip."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (ph, pw)
