"""Whether a loss recipe moves tone, shading or colour, checked before a run trains on it.

A term is neutral when, at the truth plus noise, it pulls the output nowhere a network could move it at low frequency.
L1 on the log encoding is: any monotone map keeps the median, the minimiser of L1 (Gneiting 2011, "Making and
evaluating point forecasts"; Noise2Noise's L1 case). A one-sided term on a statistic of the noisy output, such as a
local range or maximum, is not: noise lifts the statistic, so the term pays the output to shrink it. The sixteenth
and seventeenth runs' overshoot term compressed local contrast this way, and the shading gate on the raw.pixls.us
held-out set lost by about its weight (MODEL_PLAN section 20). The dead-leaves settle in the term's own comment had
predicted the darkening; nothing held the number to a limit.

The check: dead-leaves targets (no ISP, every edge sharp, eight stops of leaf values), heteroscedastic noise at a few
residual levels on the output (what a network leaves) or on the target (a real pair's reference), and a low-frequency
field the optimiser may add to the output in the log encoding: per-block offsets at BLOCKS px, bilinear so they are
smooth as a network's shading is; a local-contrast gain per Gaussian scale in CONTRAST_SIGMAS; a tone curve by octave
of the target's luminance; a gain per channel. Adam settles the field under the recipe's full pair_loss, then under
L1 and gradients alone (every optional weight 0) on the same tiles and noise. The recipe's excess over that baseline
in shading_rmse, the tone gate's band gains, the contrast slope and the channel gains is what the recipe itself would
teach. The limits are SHARE of the gates' own margins: a term that alone uses half a tie margin at the ideal leaves
the network's own error the other half.

The absolute limits are lenient: at the noise real pairs carry (up to about RELATIVE_LEVEL) the sixteenth and
seventeenth runs' recipes pass them, and both lost tone on the held-out sets. What predicted those losses is the
change against the incumbent, on the same tiles and noise (paired): the 0.2-0.5 and 0.5-0.9 bands moved 0.0015 to
0.008 darker, and the held-out bands 4 to 37 times the output side's move at 0.01. `--against` reads a recipe that way, judged on RELATIVE: past
it, a change is probed before it is trained on (MODEL_PLAN section 20).

Usage: CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -X utf8 -m harness.model.neutrality --range-weight 3 --flat-weight 1
--texture-weight 1 --transfer-weight 0.1 (train's own flags and defaults) [--json > incumbent.json]
[--against incumbent.json]; exit status 1 when a limit is exceeded (with --against, a RELATIVE one)."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from ..colorspace import luminance
from ..data.tilemetrics import SHADING_BLOCK, _blocks, _in_range_blocks, shading_rmse
from ..metrics import REGISTRY
from . import train as T
from .data import decode, encode
from .deadleaves import dead_leaves
from .heldout import TONE_MARGIN, tone_gains

WEIGHTS = ("range_weight", "flat_weight", "texture_weight", "ssim_weight", "transfer_weight", "overshoot_weight",
           "pyramid_weight")
BLOCKS = (8, 16, 32)                  # px: the offset grids, bilinear between block centres
CONTRAST_SIGMAS = (2.0, 4.0, 8.0, 16.0)   # px: local contrast is the log luminance less its blur at this scale
OCTAVES = 9                           # tone knots at 2^-8 .. 2^0 of the target's luminance
LEVELS = (0.005, 0.01, 0.02)          # noise sd at luminance 0.5, heteroscedastic (shot-noise law) below and above
NOISE_FLOOR = 1e-3                    # the variance law's read-noise floor, in units of 0.5 luminance
WHERE = ("output", "target")          # the noisy side: a network's residual, or a real pair's reference
TILES, TILE, STEPS, LR = 12, 128, 300, 0.01
SHARE = 0.5                           # of each gate's margin a recipe may use at the ideal
RELATIVE_LEVEL = 0.01                 # --against reads the levels up to this: real pairs' references measure 0.002-0.014
RELATIVE = {"shading": 0.0005, "tone": 0.001, "chroma": 0.001}   # a change against the incumbent past these is probed


def recipe(**kw) -> dict:
    """pair_loss's keyword arguments: train's defaults, overridden by `kw` (the flags' underscore names)"""
    out = {k: getattr(T, k.upper()) for k in WEIGHTS + ("range_k", "flat_k")}
    unknown = set(kw) - set(out)
    if unknown:
        raise ValueError(f"not pair_loss weights: {sorted(unknown)}")
    return {**out, **kw}


def baseline(r: dict) -> dict:
    """the same recipe with every optional term off: L1 and the gradient term alone"""
    return {**r, **{k: 0.0 for k in WEIGHTS}}


def limits() -> dict:
    return {"shading": SHARE * REGISTRY["shading_rmse"].tie_margin, "tone": SHARE * TONE_MARGIN,
            "chroma": SHARE * TONE_MARGIN}


def tiles(n: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([dead_leaves(rng, size) for _ in range(n)])


def noisy(y: np.ndarray, s: float, seed: int) -> np.ndarray:
    sd = s * np.sqrt(np.maximum(luminance(y), 0.0)[..., None] / 0.5 + NOISE_FLOOR)
    return y + np.random.default_rng(seed).normal(size=y.shape) * sd


def _nchw(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).permute(0, 3, 1, 2)


def basis(y: np.ndarray) -> dict:
    """the fixed parts of the field, from the clean tiles (n, h, w, 3)"""
    lum = luminance(y)
    le = encode(lum).astype(np.float64)
    con = np.stack([le - gaussian_filter(le, (0, s, s)) for s in CONTRAST_SIGMAS], 1)
    u = np.clip(np.log2(np.maximum(lum, 2.0 ** -(OCTAVES - 1))) + OCTAVES - 1, 0, OCTAVES - 1 - 1e-6)
    idx = np.floor(u).astype(np.int64)
    return {"con": torch.from_numpy(con.astype(np.float32)), "idx": torch.from_numpy(idx)[:, None],
            "frac": torch.from_numpy((u - idx).astype(np.float32))[:, None]}


def params(n: int, h: int, w: int) -> list[torch.Tensor]:
    ps = [torch.zeros(n, 1, h // b, w // b) for b in BLOCKS]
    ps += [torch.zeros(len(CONTRAST_SIGMAS)), torch.zeros(OCTAVES), torch.zeros(3)]
    return [p.requires_grad_() for p in ps]


def field(ps: list[torch.Tensor], bs: dict) -> torch.Tensor:
    """(n, 3, h, w) in the log encoding: block offsets, contrast gains, tone curve, channel gains"""
    *off, gain, tone, chan = ps
    h, w = bs["con"].shape[-2:]
    f = sum(F.interpolate(o, size=(h, w), mode="bilinear", align_corners=False) for o in off)
    f = f + (gain[None, :, None, None] * bs["con"]).sum(1, keepdim=True)
    f = f + tone[bs["idx"]] * (1 - bs["frac"]) + tone[bs["idx"] + 1] * bs["frac"]
    return f + chan[None, :, None, None]


def settle(r: dict, pred0: torch.Tensor, y: torch.Tensor, x: torch.Tensor, bs: dict, steps: int = STEPS,
           lr: float = LR) -> np.ndarray:
    """the output, pred0 plus the field Adam settles under pair_loss with recipe `r`; the rate falls linearly to
    zero so an L1 optimum is reached, not orbited. (n, h, w, 3) linear."""
    ps = params(pred0.shape[0], *pred0.shape[-2:])
    opt = torch.optim.Adam(ps, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 1.0 - k / max(steps, 1))
    for _ in range(steps):
        opt.zero_grad()
        T.pair_loss(pred0 + field(ps, bs), y, x, **r).backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return decode((pred0 + field(ps, bs)).permute(0, 2, 3, 1).numpy())


def readout(out: np.ndarray, clean: np.ndarray) -> dict:
    """shading_rmse and the tone gate's band gains (log) as the held-out sets read them, the medians over tiles;
    the contrast slope (log block ratio on log relative block luminance, all tiles' in-range blocks); each channel's
    log gain less the luminance's, the median over tiles"""
    shade, tone, dev, rel, chroma = [], {}, [], [], []
    for o, c in zip(out, clean):
        lo, lc = luminance(np.maximum(o, 0.0)), luminance(c)
        shade.append(shading_rmse(lo, lc))
        for band, g in tone_gains(np.maximum(o, 0.0), c).items():
            tone.setdefault(band, []).append(np.log(g))
        sel = _in_range_blocks(lc, SHADING_BLOCK)
        if sel.sum() >= 4:
            cb, db = _blocks(lc, SHADING_BLOCK)[sel], _blocks(lo, SHADING_BLOCK)[sel]
            ratio = db / cb
            dev.append(np.log(ratio / np.median(ratio)))
            rel.append(np.log(cb / np.median(cb)))
        lum_gain = np.log(lo.mean() / lc.mean())
        chroma.append([np.log(o[..., k].clip(0).mean() / c[..., k].mean()) - lum_gain for k in range(3)])
    dev, rel = np.concatenate(dev), np.concatenate(rel)
    return {"shading": float(np.nanmedian(shade)), "tone": {b: float(np.median(v)) for b, v in tone.items()},
            "slope": float(np.polyfit(rel, dev, 1)[0]), "chroma": np.median(np.array(chroma), 0).tolist()}


def excess(rec: dict, base: dict) -> dict:
    return {"shading": rec["shading"] - base["shading"],
            "tone": {b: rec["tone"][b] - base["tone"][b] for b in rec["tone"] if b in base["tone"]},
            "slope": rec["slope"] - base["slope"], "chroma": [a - b for a, b in zip(rec["chroma"], base["chroma"])]}


def check(r: dict, n: int = TILES, size: int = TILE, levels=LEVELS, where=WHERE, steps: int = STEPS, lr: float = LR,
          seed: int = 0) -> dict:
    """the recipe's excess over its baseline per (noisy side, level), the worst of each, the limits and the verdict"""
    clean = tiles(n, size, seed)
    bs, y_clean = basis(clean), _nchw(encode(clean))
    cases = []
    for k, s in enumerate(levels):
        noise = _nchw(encode(noisy(clean, s, seed + 1 + k)))
        for side in where:
            pred0, y = (noise, y_clean) if side == "output" else (y_clean, noise)
            got = {name: readout(settle(rr, pred0, y, noise, bs, steps, lr), clean)
                   for name, rr in (("recipe", r), ("baseline", baseline(r)))}
            cases.append({"where": side, "s": s, **excess(got["recipe"], got["baseline"]),
                          "baseline_shading": got["baseline"]["shading"]})
    worst = {"shading": max(c["shading"] for c in cases),
             "tone": max((abs(v) for c in cases for v in c["tone"].values()), default=0.0),
             "chroma": max(abs(v) for c in cases for v in c["chroma"]),
             "slope": min(c["slope"] for c in cases)}
    lim = limits()
    setup = {"tiles": n, "tile": size, "levels": list(levels), "steps": steps, "lr": lr, "seed": seed}
    return {"recipe": r, "setup": setup, "cases": cases, "worst": worst, "limits": lim,
            "pass": all(worst[k] <= lim[k] for k in lim)}


def against(res: dict, inc: dict, level: float = RELATIVE_LEVEL) -> dict:
    """`res` less the incumbent's `inc` (both check() results on one setup, so the tiles and noise are the same) per
    (noisy side, level) up to `level`: shading's rise, each tone band's and channel's move either way"""
    if res.get("setup") != inc.get("setup"):
        raise ValueError(f"not one setup: {res.get('setup')} against {inc.get('setup')}")
    old = {(c["where"], c["s"]): c for c in inc["cases"]}
    cases = [{"where": c["where"], "s": c["s"], "shading": c["shading"] - o["shading"],
              "tone": {b: v - o["tone"][b] for b, v in c["tone"].items() if b in o["tone"]},
              "chroma": [a - b for a, b in zip(c["chroma"], o["chroma"])]}
             for c in res["cases"] if c["s"] <= level and (o := old.get((c["where"], c["s"]))) is not None]
    if not cases:
        raise ValueError(f"no level up to {level} in both results")
    worst = {"shading": max(c["shading"] for c in cases),
             "tone": max((abs(v) for c in cases for v in c["tone"].values()), default=0.0),
             "chroma": max(abs(v) for c in cases for v in c["chroma"])}
    return {"cases": cases, "worst": worst, "limits": RELATIVE, "pass": all(worst[k] <= RELATIVE[k] for k in RELATIVE)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for k in WEIGHTS + ("range_k", "flat_k"):
        ap.add_argument("--" + k.replace("_", "-"), type=type(getattr(T, k.upper())), default=getattr(T, k.upper()))
    ap.add_argument("--tiles", type=int, default=TILES)
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="print the whole result as JSON")
    ap.add_argument("--against", help="the incumbent's result (this check's --json on its recipe, same setup): judge "
                                      "the change against it")
    a = ap.parse_args(argv)
    res = check(recipe(**{k: getattr(a, k) for k in WEIGHTS + ("range_k", "flat_k")}), a.tiles, a.tile,
                steps=a.steps, seed=a.seed)
    if a.against:
        with open(a.against, encoding="utf-8") as f:
            res["against"] = against(res, json.load(f))
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        for c in res["cases"]:
            tone = " ".join(f"{b}:{v:+.4f}" for b, v in c["tone"].items())
            print(f"{c['where']:>6} s={c['s']:<5g} shading {c['shading']:+.4f} (baseline {c['baseline_shading']:.4f})"
                  f" slope {c['slope']:+.4f} tone {tone} chroma " + " ".join(f"{v:+.4f}" for v in c["chroma"]))
        w, lim = res["worst"], res["limits"]
        print("worst: " + ", ".join(f"{k} {w[k]:+.4f}" + (f" (limit {lim[k]:g})" if k in lim else "") for k in w)
              + f" -> {'neutral' if res['pass'] else 'NOT NEUTRAL'}")
        if a.against:
            rel = res["against"]
            for c in rel["cases"]:
                tone = " ".join(f"{b}:{v:+.4f}" for b, v in c["tone"].items())
                print(f"against the incumbent, {c['where']:>6} s={c['s']:<5g} shading {c['shading']:+.4f} tone {tone}"
                      " chroma " + " ".join(f"{v:+.4f}" for v in c["chroma"]))
            w, lim = rel["worst"], rel["limits"]
            print("against the incumbent, worst: " + ", ".join(f"{k} {w[k]:+.4f} (limit {lim[k]:g})" for k in w)
                  + f" -> {'as neutral' if rel['pass'] else 'PROBE FIRST'}")
    return 0 if res.get("against", res)["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
