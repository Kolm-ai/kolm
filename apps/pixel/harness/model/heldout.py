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
the capture descriptor is the measured noise law. Same scorer, same report; the cache is derived/realpairs/."""
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
from ..data.tilemetrics import tile_metrics
from ..metrics import STILL_METRICS, verdict
from .data import _h, capture_of_params, render_pair, restore_target

HELDOUT_SOURCE = "raw-pixls-us"
HELDOUT_DIR = "heldout"   # <data_root>/derived/heldout/<source>/s<seed>_n<n>_t<tile>.npz
DEMO_PRESET = "best_case_earth"


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


def score(samples: list[dict], restore) -> dict:
    """`restore(sample) -> linear tile` judged per sample against y: the input's and the output's tile metrics,
    their medians over the set, and per metric how many tiles the output beats, ties or loses to the input under
    verdict()'s margins. `median_verdict` is verdict() on the two medians; `outputs` the restored tiles."""
    rows, outputs = [], []
    for s in samples:
        out = np.asarray(restore(s), dtype=np.float64)
        rows.append({"input": tile_metrics(s["x"], s["y"]), "output": tile_metrics(out, s["y"])})
        outputs.append(out)

    def _med(side, m):
        vals = [r[side][m] for r in rows if np.isfinite(r[side][m])]
        return float(np.median(vals)) if vals else None

    summary = {}
    for m in STILL_METRICS:
        tally = {"beats": 0, "ties": 0, "loses": 0}
        for r in rows:
            i, o = r["input"][m], r["output"][m]
            if np.isfinite(i) and np.isfinite(o):
                tally[verdict(m, o, i)[0]] += 1
        mi, mo = _med("input", m), _med("output", m)
        summary[m] = {"input": mi, "output": mo, **tally,
                      "median_verdict": verdict(m, mo, mi)[0] if mi is not None and mo is not None else None}
    return {"n": len(rows), "metrics": summary, "tiles": rows, "outputs": outputs}


def evaluate(ckpt, blind: bool = False, source: str = HELDOUT_SOURCE, n: int = 64, tile: int = 256, seed: int = 0,
             preset: str = DEMO_PRESET, ss: int = 1, root: Path | None = None, device: str | None = None,
             out: Path | None = None, save_tiles: int = 0, log=print, real: bool = False,
             scale: int | None = None) -> dict:
    """A checkpoint on the held-out set, the report as a JSON-ready dict. With `out` and `save_tiles` the first
    tiles are written there as input / model / target PNG stacks (io.save_frames) for looking at. With `real`
    the set is the real pairs of `source` (data.realpairs.build_set, at `scale` or the manifest's clean scale)
    and the capture descriptor the measured noise law."""
    from .apply import load_model, restore_frames
    b = load_model(ckpt, device)
    if real:
        from ..data.realpairs import build_set as build_real
        rs = build_real(source, n, tile, scale, seed, root=root, log=log)
        samples = [{"x": rs.tiles[k].astype(np.float64), "y": rs.targets[k].astype(np.float64),
                    "capture": rs.captures[k]} for k in range(rs.n)]
        set_scale, records, skipped, preset = rs.scale, rs.records, rs.rejected, "real"
    else:
        hs = build_set(source, n, tile, seed, root, log)
        samples = render_set(hs, preset, ss)
        set_scale, records, skipped = hs.scale, hs.records, hs.skipped
    t0 = time.perf_counter()

    def restore(s):
        cap = None if blind else (s["capture"] if real else capture_of_params(s["a"]))
        return restore_frames(b, s["x"][None], white_level=1.0, capture=cap)[0]

    rep = score(samples, restore)
    outputs = rep.pop("outputs")
    rep.update(ckpt=str(ckpt), step=b.step, blind=blind, source=source, seed=seed, n=n, tile=tile, scale=set_scale,
               preset=preset, ss=ss, real=real, seconds=round(time.perf_counter() - t0, 1),
               records=[asdict(r) for r in records], skipped=skipped)
    if out is not None and save_tiles > 0:
        from ..io import save_frames
        k = min(save_tiles, len(samples))
        d = Path(out) / "tiles"
        save_frames(d, np.stack([s["x"] for s in samples[:k]]), "input")
        save_frames(d, np.stack(outputs[:k]), "model")
        save_frames(d, np.stack([s["y"] for s in samples[:k]]), "target")
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
             f"{rep['seconds']} s of inference and scoring.", "",
             "| metric | input (median) | model (median) | verdict | beats | ties | loses |",
             "|---|---:|---:|---|---:|---:|---:|"]
    for m, s in ms.items():
        lines.append(f"| {m} | {_fmt(s['input'])} | {_fmt(s['output'])} | {s['median_verdict'] or '-'} | "
                     f"{s['beats']} | {s['ties']} | {s['loses']} |")
    what = "real capture" if real else "simulated capture"
    against = ("the reference aligned and brought to the capture's exposure" if real
               else "the clean tile in the capture's rendering intent")
    lines += ["", f"On the medians the model beats the {what} on {won} of {len(ms)} metrics, "
                  f"ties {len(ms) - won - lost}, loses {lost}. Input and model are both scored against {against}; "
                  f"the per-tile tallies use the gates' tie margins."]
    if rep["skipped"]:
        lines += ["", f"{len(rep['skipped'])} {'pair(s) rejected' if real else 'file(s) skipped'} while building the set:"]
        lines += [f"- `{f}`: {why}" for f, why in rep["skipped"].items()]
    return "\n".join(lines) + "\n"
