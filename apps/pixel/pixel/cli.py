"""``python -m pixel <command>``. Every command is one stage of the pipeline; nothing is interactive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=float))


def cmd_demo(a) -> int:
    from .claim import Claim
    from .evaluate import evaluate_claim
    from .report import write_report

    out = Path(a.out)
    claims = sorted((HERE / "claims").glob("*.json")) if not a.claim else [Path(a.claim)]
    summary = []
    for cp in claims:
        base = Claim.load(cp).to_dict()
        if base["candidate"]["kind"] == "files" and not a.claim:
            continue  # the files example is a template for your own generator
        for kind in ("truth", "camera"):
            base["candidate"] = {"kind": kind}
            claim = Claim.from_dict(base)
            res = evaluate_claim(claim, out_dir=out / f"{cp.stem}-{kind}", save_frames=a.save_frames)
            path = write_report(res, out / f"{cp.stem}-{kind}" / "report.md")
            halted = next((g["name"] for g in res["gates"] if g["status"] == "FAIL"), None)
            halted_d = next((g["name"] for g in res["gates_delivered"] if g["status"] == "FAIL"), None)
            summary.append({"claim": cp.stem, "candidate": kind, "delivery": claim.delivery, "supported_master": res["supported_master"],
                            "halted_at": halted, "supported_delivered": res["supported_delivered"], "halted_at_delivered": halted_d,
                            "wins": res["wins"], "lost_in_delivery": sorted(set(res["wins"]) - set(res["wins_delivered"])), "report": str(path)})
    _print(summary)
    return 0


def cmd_evaluate(a) -> int:
    from .claim import Claim
    from .evaluate import evaluate_claim
    from .report import render_report, write_report

    res = evaluate_claim(Claim.load(a.claim), out_dir=a.out, save_frames=a.save_frames)
    if a.out:
        write_report(res, Path(a.out) / "report.md")
    print(render_report(res) if not a.json else json.dumps(res, indent=2, default=float))
    return 0 if res["supported"] else 2


def cmd_degrade(a) -> int:
    from .degrade import apply, direction, recipe_from_json, recipe_from_seed, recipe_to_json
    from .ops import load_image, save_image

    img = load_image(a.image)
    recipe = recipe_from_json(json.loads(a.recipe)) if a.recipe else recipe_from_seed(a.seed, max_ops=a.max_ops)
    out = apply(img, recipe, a.seed)
    save_image(a.out, out)
    _print({"recipe": recipe_to_json(recipe), "direction": direction(recipe).round(4).tolist(), "out": a.out})
    return 0


def cmd_label(a) -> int:
    from .data.shards import list_shards
    from .label import label_shards

    shards = list_shards(a.shards) if Path(a.shards).is_dir() else [Path(a.shards)]
    _print(label_shards(shards, a.out, workers=a.workers, recipes_per_source=a.recipes, size=a.size, max_ops=a.max_ops, limit=a.limit))
    return 0


def cmd_collect(a) -> int:
    from .data.collect import collect

    stats = collect(a.registry, a.store, workers=a.workers, max_bytes=a.max_bytes, max_items=a.max_items, dry_run=a.dry_run, source_ids=a.source)
    _print(stats.to_dict())
    return 0 if not stats.failed else 4


def cmd_verify(a) -> int:
    from .data.collect import Store

    res = Store(a.store).verify()
    _print({"ok": res["ok"], "missing": len(res["missing"]), "corrupt": len(res["corrupt"]), "bad": (res["missing"] + res["corrupt"])[:20]})
    return 0 if not (res["missing"] or res["corrupt"]) else 2


def cmd_shard(a) -> int:
    from .data.collect import Store
    from .data.shards import write_shards

    paths = write_shards(Store(a.store), a.out, shard_bytes=a.shard_bytes)
    _print({"shards": [str(p) for p in paths]})
    return 0


def cmd_dedup(a) -> int:
    from .data.dedup import dhash, keep_set, near_duplicates
    from .data.shards import list_shards, read_shard

    hashes = {}
    for shard in list_shards(a.shards):
        for key, _ext, data, _meta in read_shard(shard):
            try:
                hashes[key] = dhash(data)
            except Exception:  # noqa: BLE001
                continue
    dupes = near_duplicates(hashes, max_distance=a.distance)
    keep = keep_set(list(hashes), dupes)
    Path(a.out).write_text("\n".join(sorted(keep)) + "\n", encoding="utf-8")
    _print({"hashed": len(hashes), "near_duplicate_pairs": len(dupes), "kept": len(keep), "keep_list": a.out})
    return 0


def cmd_plan_storage(a) -> int:
    from .data.budget import BudgetInputs, plan, render_plan

    p = plan(BudgetInputs(disk_tb=a.tb, mean_source_bytes=a.source_mb * 1024**2, recipes_per_source=a.recipes, hdr_sources=a.hdr))
    print(render_plan(p) if not a.json else json.dumps(p, indent=2))
    return 0


def cmd_preference(a) -> int:
    from .preference import bootstrap, make_manifest, read_jsonl, summarize, unblind, write_jsonl

    if a.sub == "manifest":
        pres, key = make_manifest(a.candidates, rounds=a.rounds, seed=a.seed)
        write_jsonl(a.out, pres)
        Path(a.key).write_text(json.dumps(key, indent=2), encoding="utf-8")
        _print({"presentations": len(pres), "manifest": a.out, "key": a.key})
        return 0
    votes = read_jsonl(a.votes)
    key = json.loads(Path(a.key).read_text(encoding="utf-8"))
    pairs = unblind(votes, key, question=a.question)
    if a.candidate and a.reference:
        _print(summarize(pairs, a.candidate, a.reference, n_boot=a.boot, seed=a.seed))
    else:
        _print(bootstrap(pairs, n_boot=a.boot, seed=a.seed))
    return 0


def cmd_train(a) -> int:
    from .train import TrainConfig, train

    cfg = TrainConfig(sources=a.sources or "", out_dir=a.out, size=a.size, batch=a.batch, steps=a.steps, lr=a.lr,
                      base_channels=a.channels, depth=a.depth, recipes_per_source=a.recipes, val_every=a.val_every,
                      ckpt_every=a.ckpt_every, workers=a.workers, seed=a.seed, device=a.device, smoke=a.smoke)
    _print(train(cfg))
    return 0


def cmd_edit(a) -> int:
    from .ops import load_image, save_image
    from .train import edit, load_model, parse_direction

    model, _cfg = load_model(a.ckpt, device=a.device)
    out, pdir = edit(model, load_image(a.image), parse_direction(a.direction), device=a.device)
    save_image(a.out, out)
    _print({"out": a.out, "predicted_input_direction": np.round(pdir, 3).tolist()})
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pixel", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("demo", help="evaluate the example claims with truth and camera as candidates")
    s.add_argument("--out", default=str(HERE / "demo_out"))
    s.add_argument("--claim", default=None)
    s.add_argument("--save-frames", action="store_true")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("evaluate", help="evaluate one claim JSON; exit 2 when not supported")
    s.add_argument("claim")
    s.add_argument("--out", default=None)
    s.add_argument("--json", action="store_true")
    s.add_argument("--save-frames", action="store_true")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("degrade", help="apply a seeded or explicit recipe to an image")
    s.add_argument("image")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-ops", type=int, default=3)
    s.add_argument("--recipe", default=None, help='JSON list like [["blur",{"sigma":1.5}]]')
    s.set_defaults(fn=cmd_degrade)

    s = sub.add_parser("label", help="offline-label shards: one JSONL row per (source, recipe)")
    s.add_argument("shards")
    s.add_argument("--out", required=True)
    s.add_argument("--recipes", type=int, default=8)
    s.add_argument("--size", type=int, default=256)
    s.add_argument("--max-ops", type=int, default=3)
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--limit", type=int, default=None, help="sources per shard (for smoke runs)")
    s.set_defaults(fn=cmd_label)

    s = sub.add_parser("collect", help="download registry sources into a content-addressed store")
    s.add_argument("--registry", default=str(HERE / "sources" / "registry.json"))
    s.add_argument("--store", required=True)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--max-bytes", type=int, default=None)
    s.add_argument("--max-items", type=int, default=None)
    s.add_argument("--source", action="append", default=None, help="registry id to include (repeatable)")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_collect)

    s = sub.add_parser("verify", help="re-hash every stored file against the manifest")
    s.add_argument("--store", required=True)
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("shard", help="pack stored images into tar shards")
    s.add_argument("--store", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--shard-bytes", type=int, default=1 << 30)
    s.set_defaults(fn=cmd_shard)

    s = sub.add_parser("dedup", help="near-duplicate pass over shards; writes a keep list")
    s.add_argument("shards")
    s.add_argument("--out", required=True)
    s.add_argument("--distance", type=int, default=3)
    s.set_defaults(fn=cmd_dedup)

    s = sub.add_parser("plan-storage", help="what fits on the disk, and why degraded pairs are never stored")
    s.add_argument("--tb", type=float, default=4.0)
    s.add_argument("--source-mb", type=float, default=2.0)
    s.add_argument("--recipes", type=int, default=32)
    s.add_argument("--hdr", type=int, default=2000)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_plan_storage)

    s = sub.add_parser("preference", help="blinded manifest generation and Bradley-Terry fitting")
    ss = s.add_subparsers(dest="sub", required=True)
    m = ss.add_parser("manifest")
    m.add_argument("candidates", nargs="+")
    m.add_argument("--rounds", type=int, default=3)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", required=True)
    m.add_argument("--key", required=True)
    f = ss.add_parser("fit")
    f.add_argument("--votes", required=True)
    f.add_argument("--key", required=True)
    f.add_argument("--question", default="more_beautiful")
    f.add_argument("--candidate", default=None)
    f.add_argument("--reference", default=None)
    f.add_argument("--boot", type=int, default=500)
    f.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_preference)

    s = sub.add_parser("train", help="train the conditional restoration/edit model (needs torch)")
    s.add_argument("--sources", default=None)
    s.add_argument("--out", default="runs/pixel")
    s.add_argument("--size", type=int, default=128)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--steps", type=int, default=1000)
    s.add_argument("--lr", type=float, default=2e-4)
    s.add_argument("--channels", type=int, default=32)
    s.add_argument("--depth", type=int, default=3)
    s.add_argument("--recipes", type=int, default=16)
    s.add_argument("--val-every", type=int, default=200)
    s.add_argument("--ckpt-every", type=int, default=500)
    s.add_argument("--workers", type=int, default=0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default="auto")
    s.add_argument("--smoke", action="store_true", help="procedural sources, tiny model: proves the loop, no data needed")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("edit", help="move an image along a direction with a trained model")
    s.add_argument("image")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--direction", required=True, help="e.g. 'blur=-1,noise=-0.5'")
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="auto")
    s.set_defaults(fn=cmd_edit)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
