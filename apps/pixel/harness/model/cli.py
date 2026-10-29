"""python -m harness model {train | apply | eval-demo | eval-raw | rescore | compare}"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def parse_weights(text: str | None) -> dict[str, float]:
    """--weights "source=4,other=0.5" -> {source: 4.0, other: 0.5}; a weight must be a non-negative number."""
    out: dict[str, float] = {}
    for part in (text or "").split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"--weights wants source=number, got {part!r}")
        k, v = part.split("=", 1)
        w = float(v)
        if w < 0:
            raise ValueError(f"--weights {k.strip()}: a weight must be non-negative, got {v}")
        out[k.strip()] = w
    return out


def parse_ints(text: str) -> tuple:
    return tuple(int(v) for v in text.split(",") if v.strip())


def parse_blocks(text: str):
    """One count for every level, or one per level."""
    v = parse_ints(text)
    return v[0] if len(v) == 1 else v


def cmd_train(args) -> int:
    from .train import DEFAULT_SOURCES, train
    sources = args.sources.split(",") if args.sources else DEFAULT_SOURCES
    real = tuple(s for s in args.real.split(",") if s) if args.real else ()
    train(args.out, sources, steps=args.steps, batch=args.batch, tile=args.tile, workers=args.workers, lr=args.lr,
          minutes=args.minutes, seed=args.seed, base=args.base, val_n=args.val_n, ckpt_every=args.ckpt_every,
          device=args.device, real=real, p_real=args.p_real, weights=parse_weights(args.weights),
          range_weight=args.range_weight, range_k=args.range_k, flat_weight=args.flat_weight, flat_k=args.flat_k,
          texture_weight=args.texture_weight, p_dead_leaves=args.p_dead_leaves,
          decoded_store=not args.no_decoded_store, mults=parse_ints(args.mults), blocks=parse_blocks(args.blocks),
          mid_blocks=args.mid_blocks, grad_ckpt=args.grad_ckpt, ssim_weight=args.ssim_weight,
          transfer_weight=args.transfer_weight,
          shift_real=args.shift_real, subpix_real=args.subpix_real, noise_map=args.noise_map, compile_model=args.compile, warmup=args.warmup,
          lr_floor=args.lr_floor, target_audit=args.target_audit, overshoot_weight=args.overshoot_weight,
          pyramid_weight=args.pyramid_weight, resample=args.resample)
    return 0


BF16_HELP = "run the network under bfloat16 autocast as training does (default fp32: a bf16 residual bands smooth gradients)"


def cmd_apply(args) -> int:
    from ..io import load_frames, save_frames
    from .apply import load_model, restore_frames
    from .data import capture_descriptor
    b = load_model(args.ckpt, args.device, args.bf16)
    frames = load_frames(args.input)
    target = json.loads(Path(args.target).read_text(encoding="utf-8")) if args.target else None
    t0 = time.perf_counter()
    capture = capture_descriptor(**json.loads(Path(args.capture).read_text(encoding="utf-8"))) if args.capture else None
    out = restore_frames(b, frames, target, args.white_level, capture)
    save_frames(Path(args.out), out, args.stem)
    print(f"{frames.shape[0]} frame(s) {frames.shape[2]}x{frames.shape[1]} restored in {time.perf_counter() - t0:.1f} s "
          f"(checkpoint step {b.step}) -> {args.out}")
    return 0


def cmd_eval_demo(args) -> int:
    """The demo worlds; the camera-sim frames go through the model and the result is judged like any candidate."""
    from ..claim import default_claim
    from ..cli import _summary
    from ..evaluate import build_world, evaluate, export_world
    from ..io import save_frames
    from .apply import camera_white_level, capture_of_constraints, load_model, restore_frames
    b = load_model(args.ckpt, args.device, args.bf16)
    out = Path(args.out)
    still, video = default_claim(video=False), default_claim(video=True)
    video["scene"]["n_frames"] = args.frames
    if args.quick:
        still["scene"].update(width=384, height=216, supersample=2)
        video["scene"]["n_frames"] = min(args.frames, 8)
    codes = {}
    for label, claim in (("still", still), ("video", video)):
        world = build_world(claim)
        d = out / label
        export_world(world, d)
        t0 = time.perf_counter()
        white = camera_white_level(world)
        capture = None if args.blind else capture_of_constraints(world.camera_constraints)
        restored = restore_frames(b, world.camera, white_level=white, capture=capture)
        dt = time.perf_counter() - t0
        save_frames(d, restored, "model")
        print(f"[{label}] {world.camera.shape[0]} frame(s) restored in {dt:.1f} s (checkpoint step {b.step}, "
              f"camera clip level {white:.3g})")
        print(_summary(evaluate(world, world.camera, d / "candidate_camera", "camera-as-candidate")))
        rep = evaluate(world, restored, d / "candidate_model", "model")
        print(_summary(rep))
        codes[label] = rep["overall"]
        print(f"  report: {d / 'candidate_model' / 'report.md'}")
    return 0 if all(v == "PASS" for v in codes.values()) else 1


def cmd_eval_raw(args) -> int:
    """Real raws no run trains on, through the demo camera (heldout.py), or with --real the real capture/reference
    pairs of a source (realpairs.py) with the measured noise law as the capture; one report directory per mode
    and, for the real pairs, per source."""
    from .heldout import evaluate, to_markdown
    sp = "sp_" if args.real and args.subpix else ""
    name = f"heldout_{f'real_{args.source}_' if args.real else ''}{sp}{'blind' if args.blind else 'known'}"
    out = Path(args.out) if args.out else Path(args.ckpt).resolve().parent / name
    rep = evaluate(args.ckpt, blind=args.blind, source=args.source, n=args.n, tile=args.tile, seed=args.seed,
                   ss=args.ss, device=args.device, out=out, save_tiles=args.save_tiles, real=args.real,
                   scale=args.scale, bf16=args.bf16, subpix=args.subpix, noise=args.noise,
                   frontier=not args.no_frontier)
    out.mkdir(parents=True, exist_ok=True)
    (out / "heldout.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    md = to_markdown(rep)
    (out / "heldout.md").write_text(md, encoding="utf-8")
    print(md, end="")
    print(f"  report: {out / 'heldout.md'}")
    return 0


def cmd_rescore(args) -> int:
    """eval-raw reports scored again from their stored outputs (heldout.rescore), no checkpoint or GPU; the report
    before the first rescore is kept as heldout_before_rescore.json/.md"""
    import shutil
    from .heldout import rescore, to_markdown
    for d in map(Path, args.dir):
        for ext in ("json", "md"):
            keep = d / f"heldout_before_rescore.{ext}"
            if (d / f"heldout.{ext}").exists() and not keep.exists():
                shutil.copy2(d / f"heldout.{ext}", keep)
        rep = rescore(d, noise=args.noise, frontier=not args.no_frontier)
        (d / "heldout.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        (d / "heldout.md").write_text(to_markdown(rep), encoding="utf-8")
        g = rep["gates"]
        print(f"{d}: capture {g['capture']['overall']}" + "".join(
            f" | {k} {g[k]['overall']}" for k in ("capture_hp", "camera", "frontier") if k in g))
    return 0


def cmd_compare(args) -> int:
    """every run's held-out reports side by side (compare.py), printed and with --out written"""
    from ..data import data_root
    from .compare import collect, to_markdown
    md = to_markdown(collect(Path(args.models) if args.models else data_root() / "models",
                             [r.strip() for r in args.runs.split(",")] if args.runs else None))
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    print(md, end="")
    return 0


def add_parser(sub) -> None:
    m = sub.add_parser("model", help="the recovered layer: train, apply, judge on the demo worlds")
    ms = m.add_subparsers(dest="mcmd", required=True)

    t = ms.add_parser("train", help="train on pairs the degradation sampler renders on the fly")
    t.add_argument("--out", required=True, help="checkpoint directory (resumes from last.pt)")
    t.add_argument("--sources", help="comma-separated clean sources (default: kadid10k, tid2013, polyhaven HDRIs, Wikimedia featured pictures)")
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--tile", type=int, default=256)
    t.add_argument("--workers", type=int, default=12)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--minutes", type=float, help="wall-clock budget, summed across resumes; the run stops and "
                   "checkpoints at the budget, and the cosine anneals over whichever of --steps and --minutes "
                   "runs out first")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--base", type=int, default=48, help="UNet base width")
    t.add_argument("--mults", default="1,2,4,4", help="width multiple of each level, comma-separated; a level halves the side")
    t.add_argument("--blocks", default="2", help="residual blocks per level: one count, or one per level comma-separated")
    t.add_argument("--mid-blocks", type=int, help="residual blocks at the bottleneck (default: the last level's count)")
    t.add_argument("--grad-ckpt", action="store_true",
                   help="recompute block activations in the backward pass: about a third more compute, a fraction of "
                        "the activation memory, the same gradients")
    t.add_argument("--val-n", type=int, default=48)
    t.add_argument("--ckpt-every", type=int, default=500)
    t.add_argument("--device")
    t.add_argument("--real", help="comma-separated real-pair stores to draw training pairs from (python -m harness data "
                                  "realstore): rawnind, hdr-plus-burst")
    t.add_argument("--p-real", type=float, default=0.0, help="probability a sample is a real pair rather than a rendered one")
    t.add_argument("--weights", help="draw weights of the clean sources, source=number,... (1 for a source not named)")
    t.add_argument("--range-weight", type=float,
                   help="weight of the range term (output beyond the target's local range; default 1.0, 0 turns it off)")
    t.add_argument("--range-k", type=int, help="window of the range term in px, odd (default 3)")
    t.add_argument("--flat-weight", type=float,
                   help="weight of the flat-texture term (output texture beyond the target's where the target is flat; "
                        "default 0, off)")
    t.add_argument("--flat-k", type=int, help="window of the flat-texture term in px, odd (default 7)")
    t.add_argument("--texture-weight", type=float,
                   help="weight of the texture-deficit term (output texture short of the target's where the target has "
                        "texture, in the flat term's window; default 0, off)")
    t.add_argument("--no-decoded-store", action="store_true",
                   help="decode every clean image in the loader workers instead of reading and filling the decoded "
                        "store under derived/decoded (python -m harness data decoded)")
    t.add_argument("--p-dead-leaves", type=float, default=0.0,
                   help="share of the rendered pairs drawn on procedural colour dead-leaves tiles instead of a source "
                        "tile: sharp edges at every contrast and colour pairing, flat interiors (default 0, off)")
    t.add_argument("--ssim-weight", type=float,
                   help="weight of 1 - MS-SSIM of the display renderings (losses.py, which says why it stays off; "
                        "default 0)")
    t.add_argument("--transfer-weight", type=float,
                   help="weight of the coherent-transfer term: per radial band of the luminance, the distance of "
                        "Re<P Y*> / <|Y|^2> from 1, what the MTF50 and texture gates read (losses.transfer_loss; "
                        "default 0, off)")
    t.add_argument("--overshoot-weight", type=float,
                   help="weight of the edge-overshoot term: edge_overshoot as a share of the step over the windows "
                        "that hold an edge (losses.overshoot_loss; default 0, off)")
    t.add_argument("--pyramid-weight", type=float,
                   help="weight of the residual-pyramid term: the L1 of the residual's 2 to 16 px block means, where "
                        "banding, shading and blotches keep their size (losses.pyramid_loss; default 0, off)")
    t.add_argument("--resample", choices=("learned", "smooth"), default="learned",
                   help="how the network's levels change scale: strided and transposed convolutions (learned, every "
                        "run to the fifteenth) or a binomial blur before the stride and bilinear upsampling (smooth, "
                        "no checkerboard grid; net.CondUNet)")
    t.add_argument("--shift-real", action="store_true",
                   help="score each real pair at the integer shift within a pixel that fits it best, one pixel cropped "
                        "from every sample (losses.best_shift)")
    t.add_argument("--subpix-real", action="store_true",
                   help="read each real pair's reference at its measured sub-pixel residual on top of the integer "
                        "shift (data.subpix; python -m harness data subpix measures a store written before)")
    t.add_argument("--noise-map", action="store_true",
                   help="one more input channel: the capture's noise law per pixel (net.noise_map)")
    t.add_argument("--compile", action="store_true", help="torch.compile the training forward")
    t.add_argument("--warmup", type=int, help="linear warmup steps (default 500)")
    t.add_argument("--lr-floor", type=float, help="the cosine's end as a share of --lr (default 0.05)")
    t.add_argument("--target-audit", action="store_true",
                   help="draw only the clean items the target audit keeps (a source's clean.max_halo; "
                        "python -m harness data audit reads them)")
    t.set_defaults(func=cmd_train)

    a = ms.add_parser("apply", help="restore frames with a checkpoint")
    a.add_argument("--ckpt", required=True)
    a.add_argument("--input", required=True, help=".npy linear (T,H,W,3) / image / frame dir / video")
    a.add_argument("--out", required=True)
    a.add_argument("--stem", default="model")
    a.add_argument("--target", help="JSON of target tile metrics (an edit); default is the restore target")
    a.add_argument("--white-level", type=float, default=1.0, help="the input's clip level in its units")
    a.add_argument("--capture", help="JSON with airy_first_zero_px, residual_sigma_px, electrons_at_clip, "
                                     "read_noise_e, bit_depth, bayer: the input's known capture (default unknown)")
    a.add_argument("--device")
    a.add_argument("--bf16", action="store_true", help=BF16_HELP)
    a.set_defaults(func=cmd_apply)

    e = ms.add_parser("eval-demo", help="restore the demo camera-sim and judge it through the gates")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", default="demo_out_model")
    e.add_argument("--blind", action="store_true", help="withhold the demo camera's capture descriptor")
    e.add_argument("--frames", type=int, default=24)
    e.add_argument("--quick", action="store_true")
    e.add_argument("--device")
    e.add_argument("--bf16", action="store_true", help=BF16_HELP)
    e.set_defaults(func=cmd_eval_demo)

    r = ms.add_parser("eval-raw", help="restore real raws no run trains on, through the demo camera, and score them")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--out", help="report directory (default: heldout_known or heldout_blind beside the checkpoint)")
    r.add_argument("--blind", action="store_true", help="withhold the capture descriptor")
    r.add_argument("--source", default="raw-pixls-us",
                   help="a clean source no run trains on (with --real: rawnind or hdr-plus-burst)")
    r.add_argument("--real", action="store_true",
                   help="real pairs instead of simulated captures: the source's captures against their references (data.realpairs)")
    r.add_argument("--scale", type=int, help="with --real: the working scale (default: the manifest's clean scale)")
    r.add_argument("--subpix", action="store_true",
                   help="with --real: the targets and the camera's photo moved by each pair's sub-pixel residual "
                        "(data.subpix), a set of its own on the same tiles")
    r.add_argument("--n", type=int, default=64, help="tiles in the set (the set is a function of source, seed, n, tile)")
    r.add_argument("--tile", type=int, default=256)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--ss", type=int, default=1, help="optics supersampling of the capture render")
    r.add_argument("--save-tiles", type=int, default=8, help="write the first tiles as input/model/target PNG stacks")
    r.add_argument("--device")
    r.add_argument("--bf16", action="store_true", help=BF16_HELP)
    _noise_args(r)
    r.set_defaults(func=cmd_eval_raw)

    q = ms.add_parser("rescore", help="score eval-raw reports again from their stored outputs, no checkpoint or GPU")
    q.add_argument("--dir", nargs="+", required=True, help="eval-raw report directories (heldout.json + outputs.npz)")
    _noise_args(q)
    q.set_defaults(func=cmd_rescore)

    c = ms.add_parser("compare", help="every run's held-out reports side by side, a table per report set")
    c.add_argument("--models", help="the runs' directory (default: <data root>/models)")
    c.add_argument("--runs", help="comma-separated runs to show, e.g. v13,v14b,v15 (default: every run)")
    c.add_argument("--out", help="also write the markdown here")
    c.set_defaults(func=cmd_compare)


def _noise_args(p) -> None:
    from ..data.tilemetrics import NOISE_MODES
    p.add_argument("--noise", choices=NOISE_MODES, default="residual",
                   help="the shadow SNR and dynamic-range reading (tilemetrics; the high-pass one is reported too)")
    p.add_argument("--no-frontier", action="store_true", help="skip the oracle linear frontier gates (frontier.py)")
