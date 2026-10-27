"""Conditioned residual UNet: the recovered layer's network. Input and output are 3-channel tiles in the
log domain; the output is input plus a learned residual. The conditioning vector (the target state's
normalised metric vector plus its validity mask) enters every residual block as FiLM scale and shift, so
one network serves restore and edit directions. Fully convolutional: any size that is a multiple of 8."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import COND_DIM


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

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x, e)
        return x


class CondUNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 48, mults: tuple = (1, 2, 4, 4),
                 blocks: int = 2, cond_dim: int = COND_DIM, emb: int = 256):
        super().__init__()
        self.config = dict(in_ch=in_ch, out_ch=out_ch, base=base, mults=tuple(mults), blocks=blocks,
                           cond_dim=cond_dim, emb=emb)
        self.embed = nn.Sequential(nn.Linear(cond_dim, emb), nn.SiLU(), nn.Linear(emb, emb), nn.SiLU())
        chs = [base * m for m in mults]
        self.stem = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        for i, c in enumerate(chs):
            self.down.append(Level(c, emb, blocks))
            if i + 1 < len(chs):
                self.pool.append(nn.Conv2d(c, chs[i + 1], 3, stride=2, padding=1))
        self.mid = Level(chs[-1], emb, blocks)
        self.up, self.merge = nn.ModuleList(), nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
            self.merge.append(nn.Conv2d(2 * chs[i - 1], chs[i - 1], 1))
            self.down.append(Level(chs[i - 1], emb, blocks))   # decoder levels appended after the encoder ones
        self.head = nn.Sequential(nn.GroupNorm(8, chs[0]), nn.SiLU(), nn.Conv2d(chs[0], out_ch, 3, padding=1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.n_levels = len(chs)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        e = self.embed(cond)
        h = self.stem(x)
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
