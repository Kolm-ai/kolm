"""A release's model card (MODEL_CARD.md), written from what the release and its run recorded, so it never states
more than the reports do.

    python -m harness model card RELEASE_DIR [--run RUN_DIR] [--capture-eval DIR ...] [--calibration JSON ...]

RELEASE_DIR is model export's directory (config.json: the network, the step, the training arguments, the blind noise
reading). RUN_DIR (default the directory of the checkpoint the release was exported from) holds the held-out reports
(heldout_*/heldout.json, heldout.score's); each becomes a row of gate verdicts with the first check that failed.
--capture-eval adds capture-eval's reports (the descriptor the photo command reads, against the measured one and none),
--calibration the blind noise reading's calibrations (blindnoise.calibrate's report, and calibrate_decoded's for a
decode scale's factor). The training sources' licences
come from the source manifest (data/sources.json); a source whose terms are research or non-commercial only is named
in the licence section, since the weights then carry those terms."""
from __future__ import annotations

import json
import re
from pathlib import Path

CARD = "MODEL_CARD.md"
GATE_SETS = ("capture", "camera", "frontier", "tone")
GATE_NAMES = {"capture": "the real capture (input)", "camera": "the camera's own photo",
              "frontier": "the oracle linear frontier (the best linear filter fitted on the target)",
              "tone": "the input's tone per target luminance band (not gain-matched; reports before v17 have none)"}
# licence text that limits use of what is trained on the source; matched case-insensitively on the manifest's licence
RESTRICTED = re.compile(r"non-?commercial|\bNC\b|research|scientific|no formal licen|none stated|unknown", re.I)


def _first_fail(g: dict) -> str:
    for gate in g.get("gates", []):
        if gate.get("status") == "fail":
            for c in gate.get("checks", []):
                if c.get("status") == "fail":
                    return f"{gate['name']}: {c['metric']} {c.get('detail', '')}".strip()
            return gate["name"]
    return ""


def heldout_rows(run: Path) -> list[dict]:
    """One row per heldout_*/heldout.json (rescore-kept *_pre_* copies left out): what the set is (its report's
    title line and descriptor sentence), tiles, and per gate set the verdict and the first failing check."""
    rows = []
    for j in sorted(Path(run).glob("heldout_*/heldout.json")):
        name = j.parent.name
        if "_pre_" in name or name.endswith("_pre1029"):
            continue
        r = json.loads(j.read_text(encoding="utf-8"))
        md = j.with_suffix(".md")
        title = ""
        if md.exists():
            text = md.read_text(encoding="utf-8")
            head = text.splitlines()
            title = head[0].lstrip("# ").strip() if head else ""
            d = re.search(r"capture descriptor (known|blind|withheld|unknown)", text[:2000])
            if d:
                title += f"; descriptor {d.group(1)}"
            if "to a fraction of a pixel" in text[:2000]:
                title += "; targets registered to a fraction of a pixel"
        row = {"report": name, "title": title, "n": r.get("n"), "noise": r.get("noise"), "gates": {}}
        for k in GATE_SETS:
            g = (r.get("gates") or {}).get(k)
            if g:
                row["gates"][k] = {"overall": g.get("overall"), "first_fail": _first_fail(g)}
        rows.append(row)
    return rows


def _licences(sources: list[str]) -> dict[str, str]:
    try:
        from ..data.manifest import load_manifest
        m = {s.id: s.license for s in load_manifest()}
    except Exception:  # noqa: BLE001 - a card without the manifest says so
        m = {}
    return {s: m.get(s, "not in the manifest") for s in sources}


def restricted(licence: str) -> bool:
    """Terms that limit use of what is trained on the source; a clause on what was skipped (Wikimedia's "NC, ND
    and GFDL-only files skipped") is not a term of what was kept."""
    return bool(RESTRICTED.search(re.sub(r"[^;.]*\bskipped\b[^;.]*", "", licence or "")))


def _calibrated_variant(cal: dict, bn: dict) -> dict:
    """The calibration report's variant whose blind_law kwargs are the release's reading (method, sigma, quantile,
    block; a kwarg a variant leaves out is the release's)."""
    want = {"method": bn.get("method"), "sigma": bn.get("smooth_sigma"), "q": bn.get("quantile"),
            "block": bn.get("block")}
    for name, v in (cal.get("variants") or {}).items():
        kw = {"method": name, **(v.get("kwargs") or {})}
        if kw["method"] != want["method"]:
            continue
        if want["method"] == "mad" or all(float(kw.get(k, want[k])) == float(want[k]) for k in ("sigma", "q", "block")):
            return v
    return {}


def _cell(v) -> str:
    return "-" if v in (None, "") else str(v).replace("|", "/")


def _verdict(g: dict | None) -> str:
    if not g:
        return "-"
    return "**PASS**" if g["overall"] == "PASS" else g["overall"].replace("FAIL at ", "fail at ")


def render(release: Path, run: Path | None = None, capture_evals: list[Path] = (),
           calibrations: list[Path] = ()) -> str:
    release = Path(release)
    cfg = json.loads((release / "config.json").read_text(encoding="utf-8"))
    run = Path(run) if run else Path(cfg["source_checkpoint"]).parent
    train = cfg.get("train") or {}
    net = cfg["net"]
    out = [f"# {cfg['name']}", "",
           "The recovered layer of kolm pixel: a conditional U-Net that takes a camera raw's linear decode (LibRaw's "
           "demosaic, camera white balance, linear sRGB primaries, 1.0 at the sensor's clip) and returns the scene it "
           "recorded: noise removed, detail and tone kept, conditioned on the capture (its noise law, bit depth, mosaic) "
           "and on target image metrics.", "",
           "## Release", "",
           "| | |", "|---|---|",
           f"| step | {cfg['step']} |",
           f"| parameters | {cfg['parameters']:,} ({cfg['parameters'] / 1e6:.1f}M) |",
           f"| training time | {_cell(round(cfg['elapsed_min']) if cfg.get('elapsed_min') else None)} min |",
           f"| weights | `{cfg['weights']}` (EMA, safetensors), SHA-256 `{cfg['sha256']}` |",
           f"| format | `{cfg['format']}` |",
           f"| network | CondUNet: base {net.get('base')}, mults {net.get('mults')}, blocks {net.get('blocks')}, "
           f"{net.get('mid_blocks')} mid blocks, conditioning {net.get('cond_dim')} dims, noise map "
           f"{net.get('noise_map')} |",
           f"| exported from | `{cfg.get('source_checkpoint')}` |", ""]

    out += ["## Use", "", "```bash",
            f"python -m harness photo IMG_0001.DNG --ckpt {release.as_posix()} --out restored/ --original",
            "```", "",
            "Each input writes a 16-bit sRGB TIFF, a JPEG preview and a JSON record of the decode, the capture "
            "descriptor and the noise law behind it. A raw's noise law is read blind from the frame "
            "(`data/blindnoise.py`); any other image is restored with no descriptor. `--scale 2` restores at the "
            "scale training's real pairs were decoded at; `--capture JSON` states the descriptor instead of reading it. "
            "`apply.load_model` reads this directory, or a training checkpoint, the same way.", ""]

    srcs = list(train.get("sources") or [])
    real = set(train.get("real") or [])
    weights = train.get("weights") or {}
    lic = _licences(srcs)
    out += ["## Training data", "",
            f"{train.get('steps', '-')} steps planned (the release is step {cfg['step']}), batch "
            f"{train.get('batch', '-')}, tiles of {train.get('tile', '-')} px, learning rate {train.get('lr', '-')}; "
            f"real capture/reference pairs drawn with probability {train.get('p_real', '-')}, degraded clean images "
            "otherwise.", "",
            "| source | licence (manifest) | sampling weight | real pairs |", "|---|---|---:|---|"]
    for s in srcs:
        out.append(f"| `{s}` | {_cell(lic[s])} | {_cell(weights.get(s, 1))} | {'yes' if s in real else '-'} |")
    limited = [s for s in srcs if restricted(lic[s])]
    out += ["", "Held out of training: every real pair's val and test scenes, RawNIND's test reserve, and raw-pixls-us "
            "(the raw evaluation set) entirely.", ""]

    rows = heldout_rows(run)
    out += ["## Held-out results", "",
            "Each set's tiles are scored by `model.heldout`: per metric, the restored tile against the truth beside the "
            "input's, and the gates in order (a set fails at its first failing gate; the gates after it are not "
            "reached). The gate sets compare against " + "; ".join(f"**{k}**: {v}" for k, v in GATE_NAMES.items())
            + ". \"known\" conditions on the measured noise law (on the raw-pixls-us sets, the demo camera's), "
            "\"blind\" on no descriptor; \"-\" is a gate set the set has no reference for.", "",
            "| set | what | tiles | " + " | ".join(GATE_SETS) + " | first failing check per gate set |",
            "|---|---|---:|" + "---|" * len(GATE_SETS) + "---|"]
    for r in rows:
        fails = [f"{k}: {r['gates'][k]['first_fail']}" for k in GATE_SETS
                 if k in r["gates"] and r["gates"][k]["first_fail"]]
        out.append(f"| `{r['report']}` | {_cell(r['title'])} | {_cell(r['n'])} | "
                   + " | ".join(_verdict(r["gates"].get(k)) for k in GATE_SETS)
                   + f" | {_cell('; '.join(fails))} |")
    passed = [r["report"] for r in rows if r["gates"] and all(g["overall"] == "PASS" for g in r["gates"].values())]
    out += ["", f"Every gate set passes on {len(passed)} of {len(rows)} set(s)"
            + (f": {', '.join(f'`{p}`' for p in passed)}." if passed else "."), ""]

    reports = [Path(d) / "capture_eval.json" for d in capture_evals if (Path(d) / "capture_eval.json").exists()]
    if reports:
        from .captureeval import to_markdown
        out += ["## The capture descriptor as the photo command reads it", "",
                "Held-out real pairs restored with the law measured against the reference (what training saw), the "
                "law read blind from the input (what the photo command uses), and no descriptor, at each decode "
                "scale reported. These are capture-eval's own draws (up to 12 pairs a source, 4 tiles a pair) against "
                "the capture gates only, smaller than the held-out sets above, whose verdicts they do not replace.",
                ""]
    for j in reports:
        md = to_markdown(json.loads(j.read_text(encoding="utf-8")))
        lines = md.splitlines()[1:]
        while lines and not lines[0]:
            lines.pop(0)
        out += [("#" + ln) if ln.startswith("#") else ln for ln in lines]
        out.append("")

    bn = cfg.get("blind_noise") or {}
    out += ["## Blind noise reading", "",
            f"Method `{bn.get('method')}` (Gaussian residual at sigma {bn.get('smooth_sigma')} px"
            + (f", {bn.get('block')} px cells, quantile {bn.get('quantile')}" if bn.get("method") == "blocks" else "")
            + f"), variance scale {bn.get('var_scale')} at decode scale {bn.get('calib_scale', 2)}"
            + "".join(f", times {f:.3f} at decode scale {k}" for k, f in sorted((bn.get("scale_factor") or {}).items()))
            + ".", ""]
    cals = [json.loads(Path(c).read_text(encoding="utf-8")) for c in calibrations if Path(c).exists()]
    for cal in [c for c in cals if "factor" in c]:
        e = cal.get("spread") or {}
        if cal.get("factor") and e.get("n"):
            out += [f"The factor at decode scale {cal['scale']} is the median of measured over blind noise variance "
                    f"on {cal['pairs']} train pair(s) decoded at that scale (centre {cal.get('crop')} px crops): "
                    f"{cal['factor']:.3f}, log10 p10 {e['p10']:+.3f}, p90 {e['p90']:+.3f}; per source "
                    + ", ".join(f"`{s}` {10 ** ps['median']:.3f} (n {ps['n']})"
                                for s, ps in (cal.get("per_source") or {}).items() if ps.get("n")) + ".", ""]
    for cal in [c for c in cals if "factor" not in c]:
        v = _calibrated_variant(cal, bn)
        e = v.get("val_log10_electrons_error") or {}
        if e.get("n"):
            out += [f"Calibrated on {cal.get('pairs')} stored real pairs (centre {cal.get('crop')} px crops): on the "
                    f"val pairs, log10(blind / measured electrons) has median {e['median']:+.3f}, p10 {e['p10']:+.3f}, "
                    f"p90 {e['p90']:+.3f} (n {e['n']}).", ""]
            for s, ps in (v.get("per_source") or {}).items():
                pe = ps.get("val") or {}
                if pe.get("n"):
                    out.append(f"- `{s}`: median {pe['median']:+.3f}, p10 {pe['p10']:+.3f}, p90 {pe['p90']:+.3f} "
                               f"(n {pe['n']})")
            out.append("")

    out += ["## Limitations", ""]
    failing = [r["report"] for r in rows if r["report"] not in passed]
    if failing:
        out.append(f"- It does not yet beat every reference on every set: {len(failing)} of {len(rows)} held-out "
                   "set(s) fail a gate (above). The camera and frontier gates are the bar; a set that fails them is "
                   "one where the model is not the best available result.")
    out += ["- Training's real pairs are decoded at scale 2 (box-downsampled), and every held-out verdict above is at "
            "that scale or coarser. Full resolution (`--scale 1`, the photo command's default) is measured only by "
            "capture-eval's capture gates where a scale-1 report is included; no camera or frontier verdict covers it.",
            "- The descriptor the photo command reads is blind: where it is far from the measured law, the restore is "
            "the one the descriptor implies, not the one the capture needs. `--capture` states it when it is known.",
            "- Inputs are LibRaw's demosaic of one frame: no burst merge, no lens correction beyond what the training "
            "degradations cover, and no claim about any image the held-out sets do not resemble.",
            "- An 8-bit sRGB input has been through a camera's own processing; it is restored with no descriptor, and "
            "no held-out set above measures that case.", ""]

    out += ["## Licence", ""]
    if limited:
        out.append("Training data includes source(s) whose terms limit use to research or non-commercial purposes: "
                   + ", ".join(f"`{s}` ({lic[s]})" for s in limited)
                   + ". These weights are for research use; a release for any other use needs a run trained without "
                   "them.")
    else:
        out.append("No training source in the manifest is marked research-only or non-commercial.")
    sa = [s for s in srcs if re.search(r"\bBY-SA\b", lic[s] or "", re.I)]
    if sa:
        out.append("")
        out.append("Source(s) under CC BY-SA (" + ", ".join(f"`{s}`" for s in sa) + ") are credited here; whether "
                   "weights trained on them are adaptations under that licence is not settled, and this card does not "
                   "decide it.")
    out.append("")
    return "\n".join(out)


def cmd_card(args) -> int:
    md = render(Path(args.release), args.run and Path(args.run), [Path(d) for d in (args.capture_eval or [])],
                [Path(c) for c in (args.calibration or [])])
    out = Path(args.out) if args.out else Path(args.release) / CARD
    out.write_text(md, encoding="utf-8")
    print(f"{out}: {len(md.splitlines())} lines")
    return 0


def add_parser(ms) -> None:
    c = ms.add_parser("card", help="a release's MODEL_CARD.md from its config and its run's held-out reports")
    c.add_argument("release", help="model export's directory")
    c.add_argument("--run", help="the run directory with heldout_*/ (default: the exported checkpoint's directory)")
    c.add_argument("--capture-eval", nargs="*", help="capture-eval output directories to include")
    c.add_argument("--calibration", nargs="*", help="blindnoise.calibrate's and calibrate_decoded's reports (JSON)")
    c.add_argument("--out", help=f"default RELEASE/{CARD}")
    c.set_defaults(func=cmd_card)
