"""python -m harness {demo | run | render | preference | data | model}"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .claim import default_claim, load_claim
from .display import VISIBILITY_PPD, ppd
from .evaluate import build_world, evaluate, export_world
from .io import load_frames
from .preference import QUESTIONS, fit_from_files, make_manifest


def _summary(rep: dict) -> str:
    quals = "  ".join(f"{q}={r['verdict']}" for q, r in rep["qualities"].items())
    line = f"  {rep['candidate']:<22} {rep['overall']:<22} {quals}"
    for g in rep["gates"]:
        if g["status"] == "fail":
            bad = next(c for c in g["checks"] if c["status"] == "fail")
            line += f"\n{'':25}failed check: {bad['metric']} {bad['op']}: {bad['detail']}"
    return line


def cmd_demo(args) -> int:
    out = Path(args.out)
    still, video = default_claim(video=False), default_claim(video=True)
    video["scene"]["n_frames"] = args.frames
    if args.quick:
        still["scene"].update(width=384, height=216, supersample=2)
        video["scene"]["n_frames"] = min(args.frames, 8)
    for label, claim in (("still", still), ("video", video)):
        t0 = time.time()
        world = build_world(claim)
        d = out / label
        export_world(world, d)
        s = claim["scene"]
        print(f"[{label}] {s['width']}x{s['height']} ss{s['supersample']}, {s['n_frames']} frame(s); "
              f"world built in {time.time() - t0:.1f} s; display {ppd(world.display):.1f} ppd "
              f"(visibility threshold {VISIBILITY_PPD[0]:g}-{VISIBILITY_PPD[1]:g})")
        print(_summary(evaluate(world, world.truth, d / "candidate_truth", "truth-as-candidate")))
        print(_summary(evaluate(world, world.camera, d / "candidate_camera", "camera-as-candidate")))
        print(f"  reports: {d / 'candidate_truth' / 'report.md'}, {d / 'candidate_camera' / 'report.md'}")
    return 0


def cmd_run(args) -> int:
    claim = load_claim(args.claim)
    candidate = load_frames(args.candidate)
    preference = json.loads(Path(args.preference).read_text(encoding="utf-8")) if args.preference else None
    world = build_world(claim)
    rep = evaluate(world, candidate, args.out, args.name or Path(args.candidate).name, preference,
                   rendered_map=args.rendered_map)
    print(_summary(rep))
    print(f"  report: {Path(args.out) / 'report.md'}")
    return 0 if rep["overall"] == "PASS" else (2 if rep["overall"].startswith("INCOMPLETE") else 1)


def cmd_render(args) -> int:
    world = build_world(load_claim(args.claim))
    export_world(world, args.out)
    print(f"truth + camera frames and world.json written to {args.out}")
    return 0


def cmd_pref_manifest(args) -> int:
    items = dict(kv.split("=", 1) for kv in args.item)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest, key = make_manifest(items, args.trials, args.seed, QUESTIONS, out / "stimuli")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"{len(manifest['trials'])} trials over {len(items)} items -> {out / 'manifest.json'}; "
          f"keep {out / 'key.json'} away from raters")
    return 0


def cmd_pref_fit(args) -> int:
    fit = fit_from_files(args.manifest, args.key, args.responses, args.boot, args.seed)
    Path(args.out).write_text(json.dumps(fit, indent=2), encoding="utf-8")
    for q, f in fit["questions"].items():
        pb = f["p_beats"].get("candidate", {}).get("camera")
        if pb:
            print(f"{q}: P(candidate > camera) = {pb['p']:.3f} [{pb['low']:.3f}, {pb['high']:.3f}] "
                  f"({f['n_pairs']} trials)")
    print(f"fit -> {args.out}; pass it to `run --preference` to fill the composition gate")
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to a legacy code page
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(prog="python -m harness",
                                description="Verifiably-better-than-any-lens measurement harness.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="self-test: truth and camera-sim as candidates, stills and video")
    d.add_argument("--out", default="demo_out")
    d.add_argument("--frames", type=int, default=24)
    d.add_argument("--quick", action="store_true", help="smaller scene, fewer frames")
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="evaluate a candidate against a claim")
    r.add_argument("--claim", required=True)
    r.add_argument("--candidate", required=True, help=".npy linear (T,H,W,3) / image / frame dir / video")
    r.add_argument("--out", required=True)
    r.add_argument("--preference", help="fit.json from `preference fit`")
    r.add_argument("--rendered-map", help="the candidate's rendered-pixel map (image or .npy, >0.5 = rendered, "
                   "(H,W) or (T,H,W)); without it every pixel is declared recovered")
    r.add_argument("--name")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("render", help="export truth, camera-sim and world.json for a claim")
    e.add_argument("--claim", required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_render)

    pr = sub.add_parser("preference", help="blinded pairwise study tools")
    ps = pr.add_subparsers(dest="pcmd", required=True)
    pm = ps.add_parser("manifest")
    pm.add_argument("--item", action="append", required=True, metavar="ROLE=PATH",
                    help="e.g. candidate=cand.png camera=camera.png truth=truth.png")
    pm.add_argument("--out", required=True)
    pm.add_argument("--trials", type=int, default=10, help="trials per pair")
    pm.add_argument("--seed", type=int, default=0)
    pm.set_defaults(func=cmd_pref_manifest)
    pf = ps.add_parser("fit")
    pf.add_argument("--manifest", required=True)
    pf.add_argument("--key", required=True)
    pf.add_argument("--responses", required=True, help="CSV: trial,question,choice")
    pf.add_argument("--out", required=True)
    pf.add_argument("--boot", type=int, default=500)
    pf.add_argument("--seed", type=int, default=0)
    pf.set_defaults(func=cmd_pref_fit)

    from .data.cli import add_parser as add_data_parser
    add_data_parser(sub)
    from .model.cli import add_parser as add_model_parser
    add_model_parser(sub)
    from .photo import add_parser as add_photo_parser
    add_photo_parser(sub)
    from .model.captureeval import add_parser as add_capture_eval_parser
    add_capture_eval_parser(sub)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
