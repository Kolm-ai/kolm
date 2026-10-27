"""python -m harness model {train | apply | eval-demo}"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def cmd_train(args) -> int:
    from .train import DEFAULT_SOURCES, train
    sources = args.sources.split(",") if args.sources else DEFAULT_SOURCES
    train(args.out, sources, steps=args.steps, batch=args.batch, tile=args.tile, workers=args.workers, lr=args.lr,
          minutes=args.minutes, seed=args.seed, base=args.base, val_n=args.val_n, ckpt_every=args.ckpt_every,
          device=args.device)
    return 0


def cmd_apply(args) -> int:
    from ..io import load_frames, save_frames
    from .apply import load_model, restore_frames
    from .data import capture_descriptor
    b = load_model(args.ckpt, args.device)
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
    b = load_model(args.ckpt, args.device)
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


def add_parser(sub) -> None:
    m = sub.add_parser("model", help="the recovered layer: train, apply, judge on the demo worlds")
    ms = m.add_subparsers(dest="mcmd", required=True)

    t = ms.add_parser("train", help="train on pairs the degradation sampler renders on the fly")
    t.add_argument("--out", required=True, help="checkpoint directory (resumes from last.pt)")
    t.add_argument("--sources", help="comma-separated clean sources (default: kadid10k, tid2013, polyhaven HDRIs)")
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--tile", type=int, default=256)
    t.add_argument("--workers", type=int, default=12)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--minutes", type=float, help="wall-clock budget; the run stops and checkpoints at the budget")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--base", type=int, default=48, help="UNet base width")
    t.add_argument("--val-n", type=int, default=48)
    t.add_argument("--ckpt-every", type=int, default=500)
    t.add_argument("--device")
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
    a.set_defaults(func=cmd_apply)

    e = ms.add_parser("eval-demo", help="restore the demo camera-sim and judge it through the gates")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", default="demo_out_model")
    e.add_argument("--blind", action="store_true", help="withhold the demo camera's capture descriptor")
    e.add_argument("--frames", type=int, default=24)
    e.add_argument("--quick", action="store_true")
    e.add_argument("--device")
    e.set_defaults(func=cmd_eval_demo)
