"""What the photo command's capture descriptor is worth on real raws.

    python -m harness capture-eval --ckpt MODEL.pt --sources rawnind hdr-plus-burst --out DIR [--scale 1]

Real pairs of the held-out splits (realpairs.DEFAULT_SPLITS) are decoded and matched as training's were
(realpairs.match_of: decode, alignment, gain, the pair's noise law), at the source's pair scale or at --scale, and
their tiles restored three ways, each scored as heldout.score scores a held-out set:

    measured   the law the pair measures against its reference (realpairs.capture_of): what training saw
    estimated  the law the photo command reads from the whole decoded input alone (photo.capture_for), or no
               descriptor where that law does not fit, as the photo command then restores
    unknown    no descriptor (UNKNOWN_CAPTURE)

with, per pair, how far the estimated electrons sit from the measured ones. A pair whose own law does not fit is
left out (it has no measured mode). At --scale 1, every pixel as the photo command restores by default, the pairs
are decoded at a scale training's real pairs never were (pair_scale is 2), so that report is the one that says
whether full resolution holds up."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

MODES = ("measured", "estimated", "unknown")


def build(source: str, n_pairs: int = 12, tiles_per_pair: int = 4, tile: int = 256, scale: int | None = None,
          seed: int = 0, root: Path | None = None, log=print) -> tuple[list[dict], list[dict]]:
    """(samples, pair records): up to n_pairs held-out pairs in a seeded order, each decoded once, its descriptors
    read, and up to tiles_per_pair unclipped tiles drawn (realpairs._draw_tiles)."""
    from ..data import data_root
    from ..data.realpairs import DEFAULT_SPLITS, _draw_tiles, capture_of, decode_raw, match_of, pair_scale, pairs_of
    from ..photo import capture_for
    root = Path(root or data_root())
    scale = int(scale or pair_scale(source))
    pairs = [p for p in pairs_of(source, root, log) if p.split in DEFAULT_SPLITS]
    if not pairs:
        raise ValueError(f"{source}: no pairs in {DEFAULT_SPLITS}")
    rng = np.random.default_rng(seed)
    samples, recs = [], []
    for i in rng.permutation(len(pairs)):
        if len(recs) >= n_pairs:
            break
        p = pairs[int(i)]
        t0 = time.perf_counter()
        try:
            m, _, bits, _ = match_of(p, root / "raw" / source, scale, decode_raw)
        except Exception as ex:  # noqa: BLE001 - not a pair at this scale; the next one is drawn
            log(f"{source}: {p.in_file}: {type(ex).__name__}: {str(ex)[:200]}")
            continue
        if not m["noise"].get("electrons_at_clip"):
            log(f"{source}: {p.in_file}: the pair's own law did not fit; left out")
            continue
        x = np.asarray(m["x"], dtype=np.float32)
        est, how = capture_for(x, {"raw": True, "bit_depth": int(bits), "scale": scale})
        e_blind = (how.get("law") or {}).get("electrons_at_clip")
        e_meas = float(m["noise"]["electrons_at_clip"])
        origins = _draw_tiles(m, tile, tiles_per_pair, rng)
        meas = capture_of(m["noise"], bits)
        for y0, x0 in origins:
            samples.append({"x": x[y0:y0 + tile, x0:x0 + tile].astype(np.float64),
                            "y": np.asarray(m["t"][y0:y0 + tile, x0:x0 + tile], dtype=np.float64),
                            "measured": meas, "estimated": est, "pair": len(recs)})
        recs.append({"in_file": p.in_file, "ref_file": p.ref_file, "split": p.split, "scale": scale,
                     "bit_depth": int(bits), "size": [int(x.shape[1]), int(x.shape[0])], "tiles": len(origins),
                     "measured_electrons": e_meas, "measured_read_e": m["noise"].get("read_noise_e"),
                     "estimated_mode": how["mode"], "estimated_electrons": e_blind,
                     "estimated_read_e": (how.get("law") or {}).get("read_noise_e"),
                     "log10_electrons_error": (math.log10(e_blind / e_meas) if e_blind else None),
                     "seconds": round(time.perf_counter() - t0, 1)})
        r = recs[-1]
        log(f"{source}: [{len(recs)}/{n_pairs}] {Path(p.in_file).name}: {r['size'][0]}x{r['size'][1]}, measured "
            f"{e_meas:.0f} e, estimated " + (f"{e_blind:.0f} e" if e_blind else "none")
            + f", {len(origins)} tile(s), {r['seconds']} s")
        del m, x
    return samples, recs


def evaluate(ckpt, sources: list[str], n_pairs: int = 12, tiles_per_pair: int = 4, tile: int = 256,
             scale: int | None = None, seed: int = 0, device: str | None = None, bf16: bool = False,
             root: Path | None = None, log=print, modes=MODES) -> dict:
    """Per source: the pair records and, per mode, heldout.score's medians and tallies (no frontier, no camera)."""
    from .apply import load_model, restore_frames
    from .heldout import score
    b = load_model(ckpt, device, bf16)
    rep = {"ckpt": str(ckpt), "step": b.step, "tile": tile, "seed": seed, "sources": {}}
    for source in sources:
        samples, recs = build(source, n_pairs, tiles_per_pair, tile, scale, seed, root, log)
        if not samples:
            rep["sources"][source] = {"pairs": recs, "tiles": 0}
            continue
        res = {}
        for mode in modes:
            t0 = time.perf_counter()

            def restore(s, mode=mode):
                cap = None if mode == "unknown" else s[mode]
                return restore_frames(b, s["x"][None], white_level=1.0, capture=cap)[0]

            r = score(samples, restore, "residual", None)
            res[mode] = {"metrics": r["metrics"], "gates": r["gates"], "seconds": round(time.perf_counter() - t0, 1)}
            log(f"{source}: {mode} scored on {len(samples)} tile(s)")
        errs = [p["log10_electrons_error"] for p in recs if p["log10_electrons_error"] is not None]
        rep["sources"][source] = {
            "scale": recs[0]["scale"] if recs else scale, "pairs": recs, "tiles": len(samples), "modes": res,
            "estimated_fitted": f"{len(errs)}/{len(recs)}",
            "log10_electrons_error": ({"median": float(np.median(errs)), "p10": float(np.percentile(errs, 10)),
                                       "p90": float(np.percentile(errs, 90))} if errs else None)}
    return rep


def _f(v) -> str:
    return "-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))


def to_markdown(rep: dict) -> str:
    from ..metrics import STILL_METRICS
    out = [f"# Capture descriptor modes: {rep['ckpt']} (step {rep['step']})", ""]
    for source, s in rep["sources"].items():
        out += [f"## {source} at scale {s.get('scale')}: {len(s['pairs'])} pair(s), {s['tiles']} tile(s)", ""]
        if not s["tiles"]:
            continue
        e = s["log10_electrons_error"]
        out += [f"estimated law fitted on {s['estimated_fitted']} pair(s); log10(estimated / measured electrons) "
                + (f"median {e['median']:+.3f}, p10 {e['p10']:+.3f}, p90 {e['p90']:+.3f}" if e else "none"), ""]
        modes = list(s["modes"])
        out += ["| metric | input | " + " | ".join(modes) + " | truth |", "|---" * (len(modes) + 3) + "|"]
        any_m = s["modes"][modes[0]]["metrics"]
        for m in STILL_METRICS:
            row = any_m[m]
            out.append(f"| {m} | {_f(row['input'])} | " + " | ".join(_f(s["modes"][k]["metrics"][m]["output"])
                                                               for k in modes) + f" | {_f(row['truth'])} |")
        out.append("")
        out += ["| gate set | " + " | ".join(modes) + " |", "|---" * (len(modes) + 1) + "|"]
        for g in s["modes"][modes[0]]["gates"]:
            cells = []
            for k in modes:
                gg = s["modes"][k]["gates"].get(g) or {}
                cells.append(_f(gg.get("overall")))
            out.append(f"| {g} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def cmd_capture_eval(args) -> int:
    if args.gpu_mem_fraction:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_fraction))
    rep = evaluate(args.ckpt, args.sources, args.pairs, args.tiles_per_pair, args.tile, args.scale, args.seed,
                   args.device, args.bf16)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "capture_eval.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    md = to_markdown(rep)
    (out / "capture_eval.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def add_parser(sub) -> None:
    p = sub.add_parser("capture-eval", help="score real held-out pairs restored with the measured, the blind-read "
                                            "and no capture descriptor")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sources", nargs="+", default=["rawnind", "hdr-plus-burst"])
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", type=int, default=12, help="pairs per source")
    p.add_argument("--tiles-per-pair", type=int, default=4)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--scale", type=int, default=None, help="decode scale (default the source's pair scale)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gpu-mem-fraction", type=float, default=None,
                   help="cap this process's share of the GPU's memory (beside a training run)")
    p.set_defaults(func=cmd_capture_eval)
