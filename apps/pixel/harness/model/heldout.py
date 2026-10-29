"""Held-out real-capture evaluation of the recovered layer: python -m harness model eval-raw.

The gates judge a checkpoint on procedural worlds, and validate() on tiles of the training sources' validation
split. Neither is scene content a camera recorded. This set is: tiles of a raw source no run trains on
(raw-pixls-us; train.DEFAULT_SOURCES never lists it), decoded linear and box-downsampled at the manifest's clean
scale, so the camera's own demosaic, noise and blur sit well below the tile's pixel. Each tile goes through the
demo claim's camera (camera.PRESETS["best_case_earth"], the floor every candidate is judged against) rendered by
the training sampler's own camera stage, is restored by the checkpoint with the capture descriptor known or
withheld, and is scored with the twelve tile metrics against the clean tile in the capture's rendering intent,
exactly as validate() scores a restore pair. The exposure policy is the sampler's (degrade.sample_params):
mid-grey at MID_GREY_FILL, or with P_EXPOSE_FOR_PEAK the tile's own peak at a drawn PEAK_FILL of the clip.

The set is a function of (source, seed, n, tile): the same tiles for every checkpoint. It is built once and
cached under <data_root>/derived/heldout/, so a later checkpoint costs no raw decodes; the report lists every
file, crop and seed, so the set can be rebuilt from the raws alone. A file the decoder rejects, or one smaller
than the tile, is skipped and recorded.

What the numbers mean: the input column is the simulated capture and the model column its restoration, both
against the same reference, so a reference that carries its own high-ISO noise penalises both alike
(raw-pixls-us is a format corpus, not a low-ISO one). The per-tile tallies use the gates' verdict() margins.

With real=True the set is data.realpairs' instead: real captures (a RawNIND noisy raw, an HDR+ burst frame)
against a real reference (the low-ISO GT, the burst merge) aligned and gain-matched, no camera simulation, and
the capture descriptor is the measured noise law. Same scorer, same report; the cache is derived/realpairs/.

Each tile is also scored against itself (tilemetrics.self_metrics, the truth's own values): mtf50 and texture are
judged by distance to those (more than the reference holds is sharpening or invention, not a win), and a gate
whose baseline already sits at the truth's value passes on a tie (gate.py). Two gate sets judge the medians:
CAPTURE_GATES against the input (denoise without losing detail or adding artifacts: shadow SNR and banding must
beat it, everything else tie or beat), and CAMERA_GATES against the camera's own finished photo where the source
has one (HDR+ final.jpg, matched to the target's rendering by realpairs.match_camera): beat it on every metric,
the truth-bound ties allowed, dynamic range tie or beat. Highlight loss is in neither: a real target is
clipped at 1, so there is nothing above it to lose.

Noise is read as tilemetrics' "residual" (what of the tile no linear filter of the target explains), not the
labels' high-pass reading, which counts the target's own texture as noise: under it the truth itself fails the
denoise gate on textured tiles. The high-pass verdict is kept as gates["capture_hp"], the one reports before v16
gave. And since denoising trades detail for noise, so that even the best linear filter fitted on the truth falls
below a noisy capture on mtf50 and texture, gates["frontier"] judges the output against that oracle at the
output's own shadow SNR (frontier.py), every metric tie or beat.

evaluate() stores the restored tiles beside the report (OUTPUTS_FILE), so rescore() can score a report again under
a new reading or frontier without the checkpoint or the GPU."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..camera import PRESETS
from ..colorspace import luminance
from ..data import data_root
from ..data.degrade import P_EXPOSE_FOR_PEAK, PEAK_FILL, PEAK_PERCENTILE, DegradeParams, expose_for_peak_stops
from ..data.label import clean_settings, list_items, load_items
from ..data.tilemetrics import noise_metrics, self_metrics, tile_metrics
from ..gate import CheckResult, GateResult, overall, run_gates
from ..metrics import STILL_METRICS, verdict
from .data import _h, capture_of_params, render_pair, restore_target
from .frontier import cached_ladder
from .frontier import gates as frontier_gates

HELDOUT_SOURCE = "raw-pixls-us"
HELDOUT_DIR = "heldout"   # <data_root>/derived/heldout/<source>/s<seed>_n<n>_t<tile>.npz
DEMO_PRESET = "best_case_earth"
OUTPUTS_FILE = "outputs.npz"   # beside heldout.json: the restored tiles, float32 (n, tile, tile, 3)
NOISE_KEYS = ("shadow_snr_db", "dr_steps")
# Tone by the target's luminance band. Every tile metric above is gain-matched, and highlight loss reads "-" on real
# pairs, so an output that darkens a band passes them all: v16 lost 5 % of tone at 0.2 to 0.5 and 20 to 25 % on
# clipped white unseen. A tile's gain in a band is the output's mean luminance over the target's pixels in the band
# over the target's own mean there, on a band with TONE_MIN_PX of them; the set's, the median over such tiles. The
# tone gate wants the output's gain no further from 1 than the input's, within TONE_MARGIN in log, on each band
# measured on TONE_MIN_TILES tiles or more: the input sits at the target's exposure, so its gain is what tone the
# capture keeps and the model may only restore it.
TONE_BANDS = ((0.05, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 0.999), (0.999, np.inf))
TONE_MIN_PX = 256
TONE_MIN_TILES = 4
TONE_MARGIN = 0.03


def _g(name: str, *checks: tuple[str, str]) -> dict:
    return {"name": name, "checks": [{"metric": m, "op": op} for m, op in checks]}


TOB, BEATS = "ties_or_beats_camera", "beats_camera"
# against the input capture: the restoration must denoise (shadow SNR, banding) and give nothing else up. Colour
# and dynamic range are tie-or-beat, not beat: the captures' delta E (0.68 HDR+, 0.74 RawNIND medians) sits
# within the 0.5 tie margin of what any candidate can reach, and their dynamic-range steps at the tiles' ceiling.
# Neither set checks highlight loss: it reads clean pixels above 1, and a real pair's target is the reference
# clipped at 1, so it is never measured here and would hold every verdict at INCOMPLETE (the simulated set's
# gates, whose targets keep their range, still check it)
CAPTURE_GATES = [
    _g("lighting_materials", ("shading_rmse", TOB), ("landmark_error_px", TOB)),
    _g("tone_color", ("delta_e_mean", TOB), ("delta_e_max", TOB), ("dr_steps", TOB)),
    _g("artifacts", ("edge_overshoot", TOB), ("periodicity", TOB)),
    _g("denoise", ("shadow_snr_db", BEATS), ("banding_score", BEATS)),
    _g("resolution", ("mtf50_cpp", TOB)),
    _g("micro_detail", ("texture_hf_fidelity", TOB)),
]
# against the camera's own finished photo: beat it everywhere, the ties a truth-bound metric allows excepted
CAMERA_GATES = [
    _g("lighting_materials", ("shading_rmse", BEATS), ("landmark_error_px", BEATS)),
    _g("tone_color", ("delta_e_mean", BEATS), ("delta_e_max", BEATS), ("dr_steps", TOB)),
    _g("artifacts", ("edge_overshoot", BEATS), ("periodicity", BEATS), ("banding_score", BEATS)),
    _g("denoise", ("shadow_snr_db", BEATS)),
    _g("resolution", ("mtf50_cpp", BEATS)),
    _g("micro_detail", ("texture_hf_fidelity", BEATS)),
]


@dataclass(frozen=True)
class HeldoutTile:
    file: str              # relative to <root>/raw/<source>, posix
    y0: int
    x0: int
    exposure_stops: float  # the sampler's exposure policy, drawn per tile
    seed_a: int            # the capture render's seed
    seed_b: int            # the target render's seed


@dataclass
class HeldoutSet:
    source: str
    seed: int
    n: int
    tile: int
    scale: int                 # the manifest's clean scale the tiles were decoded at
    tiles: np.ndarray          # (n, tile, tile, 3) float32 linear, the clean crops
    records: list[HeldoutTile]
    skipped: dict[str, str]    # file -> why it is not in the set


def demo_camera_params(preset: str = DEMO_PRESET, exposure_stops: float = 0.0) -> DegradeParams:
    """The sampler's rendering of a claim's camera preset: the camera stage on at that preset's optics and sensor,
    exposure placing mid-grey at MID_GREY_FILL x 2**stops, no ISP stage (the demo camera-sim has none)."""
    c = PRESETS[preset]
    return DegradeParams(expose=1, exposure_stops=float(exposure_stops), camera_on=1, preset=c.name,
                         pixel_pitch_um=c.pixel_pitch_um, focal_mm=c.focal_mm, f_number=c.f_number,
                         full_well_e=c.full_well_e, white_fill=c.white_fill, read_noise_e=c.read_noise_e,
                         row_noise_e=c.row_noise_e, bit_depth=c.bit_depth, bayer=int(c.bayer),
                         residual_blur_um=c.residual_blur_um, shading_corner_stops=0.0, shading_corrected=1)


def exposure_policy(tile: np.ndarray, rng: np.random.Generator) -> float:
    """Exposure offset in stops from mid-grey: 0, or with P_EXPOSE_FOR_PEAK the offset that places the tile's own
    peak (PEAK_PERCENTILE over its median) at a drawn PEAK_FILL of the clip, as sample_params draws it."""
    lum = luminance(np.asarray(tile, dtype=np.float64))
    med, peak = float(np.median(lum)), float(np.percentile(lum, PEAK_PERCENTILE))
    if rng.random() < P_EXPOSE_FOR_PEAK and med > 0:
        return expose_for_peak_stops(peak / med, float(rng.uniform(*PEAK_FILL)))
    return 0.0


def set_path(source: str, seed: int, n: int, tile: int, root: Path) -> Path:
    return Path(root) / "derived" / HELDOUT_DIR / source / f"s{seed}_n{n}_t{tile}.npz"


def save_set(hs: HeldoutSet, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = {"source": hs.source, "seed": hs.seed, "n": hs.n, "tile": hs.tile, "scale": hs.scale,
            "records": [asdict(r) for r in hs.records], "skipped": hs.skipped}
    np.savez(p, tiles=hs.tiles.astype(np.float32), meta=json.dumps(meta))


def load_set(p: Path) -> HeldoutSet:
    with np.load(p) as z:
        meta, tiles = json.loads(str(z["meta"])), np.asarray(z["tiles"], dtype=np.float32)
    return HeldoutSet(meta["source"], int(meta["seed"]), int(meta["n"]), int(meta["tile"]), int(meta["scale"]),
                      tiles, [HeldoutTile(**r) for r in meta["records"]], dict(meta["skipped"]))


def build_set(source: str = HELDOUT_SOURCE, n: int = 64, tile: int = 256, seed: int = 0, root: Path | None = None,
              log=print) -> HeldoutSet:
    """The cached set, or a fresh one: the source's items in a seeded order, each decoded through the labeller's
    loader (the manifest's clean scale applied), one seeded crop per item until n tiles are in."""
    root = Path(root) if root else data_root()
    p = set_path(source, seed, n, tile, root)
    if p.exists():
        return load_set(p)
    scale = int(clean_settings(source).get("scale", 1) or 1)
    items = list_items(source, root)
    rng = np.random.default_rng(_h("heldout", source, seed, n, tile))
    tiles, records, skipped = [], [], {}
    t0 = time.perf_counter()
    for i in rng.permutation(len(items)):
        if len(tiles) == n:
            break
        it = items[int(i)]
        errors: dict = {}
        img = load_items([it], root, errors).get(it)
        if img is None:
            skipped[it.file] = errors.get(it, "no image")
            continue
        h, w = img.shape[:2]
        if h < tile or w < tile:
            skipped[it.file] = f"{w}x{h} at scale {scale} is smaller than the tile"
            continue
        y0, x0 = int(rng.integers(0, h - tile + 1)), int(rng.integers(0, w - tile + 1))
        crop = np.ascontiguousarray(img[y0:y0 + tile, x0:x0 + tile], dtype=np.float32)
        stops = exposure_policy(crop, rng)
        sa, sb = int(rng.integers(1 << 62)), int(rng.integers(1 << 62))
        tiles.append(crop)
        records.append(HeldoutTile(it.file, y0, x0, stops, sa, sb))
        if log and len(tiles) % 16 == 0:
            log(f"  held-out set {source}: {len(tiles)}/{n} tiles, {len(skipped)} files skipped, "
                f"{time.perf_counter() - t0:.0f} s")
    if len(tiles) < n:
        raise ValueError(f"{source}: only {len(tiles)} of {n} tiles ({len(items)} items, {len(skipped)} skipped)")
    hs = HeldoutSet(source, seed, n, tile, scale, np.stack(tiles), records, skipped)
    save_set(hs, p)
    return hs


def render_set(hs: HeldoutSet, preset: str = DEMO_PRESET, ss: int = 1) -> list[dict]:
    """The set as restore pairs in make_sample's shape: x the simulated capture, y the clean tile in the capture's
    rendering intent, a the capture's parameters, metrics y's own tile metrics (the restore conditioning)."""
    out = []
    for crop, r in zip(hs.tiles, hs.records):
        a = demo_camera_params(preset, r.exposure_stops)
        b = restore_target(a)
        xa, xb, m = render_pair(np.asarray(crop, dtype=np.float64), a, b, r.seed_a, r.seed_b, True, ss)
        out.append({"x": xa, "y": xb, "metrics": m, "restore": True, "a": a, "b": b, "record": r})
    return out


def _median(rows: list[dict], side: str, m: str) -> float | None:
    vals = [r[side][m] for r in rows if side in r and np.isfinite(r[side][m])]
    return float(np.median(vals)) if vals else None


def _tally(rows: list[dict], m: str, base: str) -> dict:
    t = {"beats": 0, "ties": 0, "loses": 0}
    for r in rows:
        if base not in r:
            continue
        b, o, tr = r[base][m], r["output"][m], r["truth"][m]
        if np.isfinite(b) and np.isfinite(o):
            t[verdict(m, o, b, tr if np.isfinite(tr) else None)[0]] += 1
    return t


def _gates(spec: list[dict], cand: dict, base: dict, truth: dict, baseline: str) -> dict:
    res = run_gates(spec, cand, base, truth, baseline)
    return {"overall": overall(res), "gates": [g.to_dict() for g in res]}


def _band(lo: float, hi: float) -> str:
    return f"{lo:g}+" if np.isinf(hi) else f"{lo:g}-{hi:g}"


def tone_gains(img: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """per TONE_BANDS band with TONE_MIN_PX of the target's pixels in it, `img`'s gain there: its mean luminance over
    those pixels over the target's"""
    li, ly = luminance(img), luminance(y)
    out = {}
    for lo, hi in TONE_BANDS:
        m = (ly >= lo) & (ly < hi)
        if int(m.sum()) >= TONE_MIN_PX:
            out[_band(lo, hi)] = float(li[m].mean() / ly[m].mean())
    return out


def tone_gate(rows: list[dict]) -> dict:
    """The tone gate on score()'s rows (each with "tone": input and output tone_gains): per band the set's gains
    ("bands": input, output, tiles), and a check per band on TONE_MIN_TILES tiles that the output's |log gain|
    is at most the input's plus TONE_MARGIN; no such band is a skipped check."""
    bands, checks = {}, []
    for lo, hi in TONE_BANDS:
        k = _band(lo, hi)
        have = [r["tone"] for r in rows if k in r["tone"]["output"]]
        if not have:
            continue
        gi, go = (float(np.median([t[side][k] for t in have])) for side in ("input", "output"))
        bands[k] = {"input": gi, "output": go, "tiles": len(have)}
        if len(have) < TONE_MIN_TILES:
            continue
        di, do = (abs(float(np.log(max(g, 1e-9)))) for g in (gi, go))
        ok = do <= di + TONE_MARGIN
        checks.append(CheckResult(f"tone {k}", "<=", di + TONE_MARGIN, do, di, "pass" if ok else "fail",
                                  f"output at {go:.4g} of the target's gain, input {gi:.4g}: |log| {do:.4g} "
                                  f"{'<=' if ok else '>'} {di:.4g} + {TONE_MARGIN:g}"))
    if not checks:
        checks.append(CheckResult("tone", "<=", None, None, None, "skip",
                                  f"no band with {TONE_MIN_PX} target pixels on {TONE_MIN_TILES} tiles"))
    statuses = {c.status for c in checks}
    g = GateResult("tone", "fail" if "fail" in statuses else ("skip" if statuses == {"skip"} else "pass"), checks)
    return {"overall": overall([g]), "gates": [g.to_dict()], "bands": bands}


def score(samples: list[dict], restore, noise: str = "residual", frontier: dict | None = None) -> dict:
    """`restore(sample) -> linear tile` judged per sample against y: the input's, the output's, the truth's own
    (self_metrics(y)) and, where the sample has one, the camera photo's tile metrics; their medians over the set;
    per metric how many tiles the output beats, ties or loses to the input (and to the camera) under verdict()'s
    margins, mtf50 and texture by distance to the tile's own values. `median_verdict` is verdict() on the medians;
    `gates` CAPTURE_GATES on the medians against the input and CAMERA_GATES against the camera, the output's
    camera medians taken over the tiles that have a camera photo; `outputs` the restored tiles. `noise` is the
    shadow SNR and dynamic-range reading (tilemetrics.NOISE_MODES). Under "residual" each row also carries the
    high-pass reading of the input, output and truth ("hp"), and gates["capture_hp"] is CAPTURE_GATES with it.
    Each row's "tone" is the input's and output's tone_gains, gates["tone"] tone_gate on them (not gain-matched,
    unlike every other gate). With `frontier`, frontier.cached_ladder of these samples under the same reading, gates["frontier"] is
    frontier.gates on the output."""
    if frontier is not None and frontier["noise"] != noise:
        raise ValueError(f"the frontier was scored under noise {frontier['noise']!r}, not {noise!r}")
    rows, outputs = [], []
    for s in samples:
        out = np.asarray(restore(s), dtype=np.float64)
        r = {"input": tile_metrics(s["x"], s["y"], noise), "output": tile_metrics(out, s["y"], noise),
             "truth": self_metrics(s["y"], noise),
             "tone": {"input": tone_gains(s["x"], s["y"]), "output": tone_gains(out, s["y"])}}
        if s.get("camera") is not None:
            r["camera"] = tile_metrics(np.asarray(s["camera"], dtype=np.float64), s["y"], noise)
        if noise != "hp":
            r["hp"] = {"input": noise_metrics(s["x"], s["y"]), "output": noise_metrics(out, s["y"]),
                       "truth": noise_metrics(s["y"], s["y"])}
        rows.append(r)
        outputs.append(out)
    cam_rows = [r for r in rows if "camera" in r]

    summary = {}
    for m in STILL_METRICS:
        mi, mo, mt = _median(rows, "input", m), _median(rows, "output", m), _median(rows, "truth", m)
        summary[m] = {"input": mi, "output": mo, "truth": mt, **_tally(rows, m, "input"),
                      "median_verdict": verdict(m, mo, mi, mt)[0] if mi is not None and mo is not None else None}
        if cam_rows:
            mc, moc, mtc = (_median(cam_rows, k, m) for k in ("camera", "output", "truth"))
            summary[m]["camera"] = {"camera": mc, "output": moc, "truth": mtc, **_tally(cam_rows, m, "camera"),
                                    "median_verdict": (verdict(m, moc, mc, mtc)[0]
                                                       if mc is not None and moc is not None else None)}
    med = {k: {m: summary[m][k] for m in STILL_METRICS} for k in ("input", "output", "truth")}
    gates = {"capture": _gates(CAPTURE_GATES, med["output"], med["input"], med["truth"], "capture")}
    if noise != "hp":
        hp_rows = [r["hp"] for r in rows]
        hp = {k: {**med[k], **{m: _median(hp_rows, k, m) for m in NOISE_KEYS}} for k in ("input", "output", "truth")}
        gates["capture_hp"] = _gates(CAPTURE_GATES, hp["output"], hp["input"], hp["truth"], "capture")
    if cam_rows:
        cm = {k: {m: summary[m]["camera"][k] for m in STILL_METRICS} for k in ("camera", "output", "truth")}
        gates["camera"] = {**_gates(CAMERA_GATES, cm["output"], cm["camera"], cm["truth"], "camera"), "tiles": len(cam_rows)}
    if frontier is not None:
        gates["frontier"] = frontier_gates(frontier, [r["output"] for r in rows], [r["truth"] for r in rows])
    gates["tone"] = tone_gate(rows)
    return {"n": len(rows), "noise": noise, "metrics": summary, "gates": gates, "tiles": rows, "outputs": outputs}


def _samples(real: bool, source: str, n: int, tile: int, scale: int | None, seed: int, root: Path | None, log,
             subpix: bool, preset: str, ss: int) -> tuple:
    """the evaluation set's samples, its scale, records, skipped files and preset ("real" for real pairs)"""
    if real:
        from ..data.realpairs import build_set as build_real
        rs = build_real(source, n, tile, scale, seed, root=root, log=log, subpix=subpix)
        samples = [{"x": rs.tiles[k].astype(np.float64), "y": rs.targets[k].astype(np.float64),
                    "capture": rs.captures[k],
                    "camera": (rs.cameras[k] if rs.cameras is not None and rs.has_camera[k] else None)}
                   for k in range(rs.n)]
        return samples, rs.scale, rs.records, rs.rejected, "real"
    hs = build_set(source, n, tile, seed, root, log)
    return render_set(hs, preset, ss), hs.scale, hs.records, hs.skipped, preset


def evaluate(ckpt, blind: bool = False, source: str = HELDOUT_SOURCE, n: int = 64, tile: int = 256, seed: int = 0,
             preset: str = DEMO_PRESET, ss: int = 1, root: Path | None = None, device: str | None = None,
             out: Path | None = None, save_tiles: int = 0, log=print, real: bool = False,
             scale: int | None = None, bf16: bool = False, subpix: bool = False, noise: str = "residual",
             frontier: bool = True) -> dict:
    """A checkpoint on the held-out set, the report as a JSON-ready dict. With `out` the restored tiles are stored
    there (OUTPUTS_FILE, for rescore), and with `save_tiles` the first ones written as input / model / target PNG
    stacks (io.save_frames) for looking at. With `real` the set is the real pairs of `source`
    (data.realpairs.build_set, at `scale` or the manifest's clean scale) and the capture descriptor the measured
    noise law; with `subpix` too the set whose targets, and the camera's photo, are moved by each pair's sub-pixel
    residual (build_set's subpix). `noise` and `frontier` (the cached frontier ladder, built on first use) as
    score's."""
    from .apply import load_model, restore_frames
    b = load_model(ckpt, device, bf16)
    samples, set_scale, records, skipped, preset = _samples(real, source, n, tile, scale, seed, root, log, subpix,
                                                            preset, ss)
    table = cached_ladder(samples, root or data_root(), noise, log=log) if frontier else None
    t0 = time.perf_counter()

    def restore(s):
        cap = None if blind else (s["capture"] if real else capture_of_params(s["a"]))
        return np.asarray(restore_frames(b, s["x"][None], white_level=1.0, capture=cap)[0], dtype=np.float32)  # as stored

    rep = score(samples, restore, noise, table)
    outputs = rep.pop("outputs")
    rep.update(ckpt=str(ckpt), step=b.step, blind=blind, source=source, seed=seed, n=n, tile=tile, scale=set_scale,
               preset=preset, ss=ss, real=real, subpix=bool(real and subpix), seconds=round(time.perf_counter() - t0, 1),
               records=[asdict(r) for r in records], skipped=skipped)
    if out is not None:
        Path(out).mkdir(parents=True, exist_ok=True)
        np.savez(Path(out) / OUTPUTS_FILE, outputs=np.stack(outputs).astype(np.float32))
    if out is not None and save_tiles > 0:
        from ..io import save_frames
        k = min(save_tiles, len(samples))
        d = Path(out) / "tiles"
        save_frames(d, np.stack([s["x"] for s in samples[:k]]), "input")
        save_frames(d, np.stack(outputs[:k]), "model")
        save_frames(d, np.stack([s["y"] for s in samples[:k]]), "target")
    return rep


def rescore(report_dir: Path, noise: str = "residual", frontier: bool = True, root: Path | None = None,
            log=print) -> dict:
    """A report scored again from the tiles evaluate() stored beside it (OUTPUTS_FILE), without the checkpoint or
    the GPU: the same set rebuilt from its cache, the stored outputs as the restoration, under `noise` and with the
    frontier. The report keeps its run fields; n, noise, metrics, gates and tiles are replaced."""
    d = Path(report_dir)
    rep = json.loads((d / "heldout.json").read_text(encoding="utf-8"))
    p = d / OUTPUTS_FILE
    if not p.exists():
        raise FileNotFoundError(f"{p}: no stored outputs (a report from before v16); run eval-raw on its checkpoint")
    outs = np.load(p)["outputs"]
    real = rep.get("real", False)
    samples = _samples(real, rep["source"], rep["n"], rep["tile"], rep["scale"] if real else None, rep["seed"], root,
                       log, rep.get("subpix", False), rep["preset"], rep["ss"])[0]
    if len(outs) != len(samples):
        raise ValueError(f"{p} holds {len(outs)} tiles, the set {len(samples)}")
    table = cached_ladder(samples, root or data_root(), noise, log=log) if frontier else None
    it = iter(outs)
    new = score(samples, lambda s: next(it), noise, table)
    new.pop("outputs")
    rep.update(new)
    return rep


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.4g}"


def to_markdown(rep: dict) -> str:
    ms = rep["metrics"]
    won = sum(s["median_verdict"] == "beats" for s in ms.values())
    lost = sum(s["median_verdict"] == "loses" for s in ms.values())
    real = rep.get("real", False)
    kind = "Real-pair evaluation" if real else "Held-out raw evaluation"
    capture = ("the measured noise law" if real else f"camera preset `{rep['preset']}`, ss {rep['ss']}")
    lines = [f"# {kind}: {rep['source']} at clean scale {rep['scale']}, {rep['n']} tiles of "
             f"{rep['tile']} px, seed {rep['seed']}", "",
             f"Checkpoint `{rep['ckpt']}` (step {rep['step']}), capture descriptor "
             f"{'withheld' if rep['blind'] else 'known'} ({capture}), "
             f"{rep['seconds']} s of inference and scoring; noise read as `{rep.get('noise', 'hp')}`."
             + (" Targets, and the camera's photo, registered to the input to a fraction of a pixel (data.subpix)."
                if rep.get("subpix") else ""), "",
             "| metric | input (median) | model (median) | truth | verdict | beats | ties | loses |",
             "|---|---:|---:|---:|---|---:|---:|---:|"]
    for m, s in ms.items():
        lines.append(f"| {m} | {_fmt(s['input'])} | {_fmt(s['output'])} | {_fmt(s.get('truth'))} | "
                     f"{s['median_verdict'] or '-'} | {s['beats']} | {s['ties']} | {s['loses']} |")
    what = "real capture" if real else "simulated capture"
    against = ("the reference aligned and brought to the capture's exposure" if real
               else "the clean tile in the capture's rendering intent")
    lines += ["", f"On the medians the model beats the {what} on {won} of {len(ms)} metrics, "
                  f"ties {len(ms) - won - lost}, loses {lost}. Input and model are both scored against {against}; "
                  f"the per-tile tallies use the gates' tie margins, mtf50 and texture by distance to the "
                  f"target's own values (truth column)."]
    gates = rep.get("gates", {})
    if any("camera" in s for s in ms.values()):
        nc = gates.get("camera", {}).get("tiles", "-")
        lines += ["", f"Against the camera's own finished photo ({nc} tiles; tone, colour and local gain matched to "
                      f"the target, its detail, noise and artifacts its own):", "",
                  "| metric | camera (median) | model (median) | truth | verdict | beats | ties | loses |",
                  "|---|---:|---:|---:|---|---:|---:|---:|"]
        for m, s in ms.items():
            c = s.get("camera")
            if c:
                lines.append(f"| {m} | {_fmt(c['camera'])} | {_fmt(c['output'])} | {_fmt(c['truth'])} | "
                             f"{c['median_verdict'] or '-'} | {c['beats']} | {c['ties']} | {c['loses']} |")
    fr = gates.get("frontier")
    if fr:
        mt = fr["match"]
        rng = mt["ladder_snr"]
        lines += ["", f"Against the oracle linear frontier at the model's own shadow SNR (frontier.py: the best "
                      f"linear filter fitted on the target, matched per tile; output median {_fmt(mt['output_snr'])} "
                      f"dB, the ladder's median {_fmt(rng[0]) if rng else '-'} to {_fmt(rng[1]) if rng else '-'} dB, "
                      f"{mt['clamped']} of {mt['tiles']} tiles clamped to an end of their ladder, "
                      f"{mt['set_median_match']} matched on the set's median, {mt.get('unmatched', 0)} unmatched "
                      f"for want of a shadow SNR):", "",
                  "| metric | frontier (matched) | model (median) | truth | verdict |", "|---|---:|---:|---:|---|"]
        for m, c in fr["metrics"].items():
            lines.append(f"| {m} | {_fmt(c['frontier'])} | {_fmt(c['output'])} | {_fmt(c['truth'])} | "
                         f"{c['verdict'] or '-'} |")
    bands = (gates.get("tone") or {}).get("bands")
    if bands:
        lines += ["", f"Tone by the target's luminance band, not gain-matched as every metric above is: per tile the "
                      f"band's mean luminance over the target's on the target's pixels in the band (on tiles with "
                      f"{TONE_MIN_PX} of them), the median over the tiles; 1 is the target's tone. The tone gate wants "
                      f"the model no further from 1 than the input, within {TONE_MARGIN:g} in log, on each band of "
                      f"{TONE_MIN_TILES} tiles or more:", "",
                  "| band | input | model | tiles |", "|---|---:|---:|---:|"]
        lines += [f"| {k} | {_fmt(b['input'])} | {_fmt(b['output'])} | {b['tiles']} |" for k, b in bands.items()]
    for key, label in (("capture", f"against the {what}"),
                       ("capture_hp", f"against the {what}, noise read high-pass (legacy: the verdict reports "
                                      f"before v16 gave; it counts the target's own texture as noise)"),
                       ("camera", "against the camera's photo"),
                       ("frontier", "against the oracle linear frontier"),
                       ("tone", "on tone (the band gains above)")):
        g = gates.get(key)
        if not g:
            continue
        lines += ["", f"Gates {label}: **{g['overall']}**", ""]
        for gr in g["gates"]:
            lines.append(f"- {gr['name']}: {gr['status']}; " + "; ".join(
                f"{c['metric']} {c['status']} ({c['detail']})" for c in gr["checks"]))
    if rep["skipped"]:
        lines += ["", f"{len(rep['skipped'])} {'pair(s) rejected' if real else 'file(s) skipped'} while building the set:"]
        lines += [f"- `{f}`: {why}" for f, why in rep["skipped"].items()]
    return "\n".join(lines) + "\n"
