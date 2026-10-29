"""The product: a camera's raw (or any photo) in, the restored photo out.

    python -m harness photo IN [IN ...] --ckpt RELEASE_DIR --out DIR

--ckpt takes a release directory (model export: model.safetensors and config.json, its SHA-256 checked) or a
training checkpoint (best.pt).

A raw (DNG, CR2/CR3, NEF, ARW, RAF, ORF, RW2, ...) decodes through LibRaw as the training pairs were decoded
(realpairs.decode_raw: camera white balance, linear sRGB primaries, 1.0 at the white level, no brightening),
box-downsampled by --scale (1, the default, keeps every pixel). Its capture descriptor is the one training saw for a
real pair (realpairs.capture_of): the noise law in that decode's units, here read blind from the frame itself
(data/blindnoise.py, calibrated on real train pairs at the decode scale), the bit depth of the sensor's white level,
Bayer, and no stated optics. A raw whose law does not fit, and any other image (an 8-bit sRGB file has been through
a camera's own processing), is restored blind (UNKNOWN_CAPTURE), as training covers for a web JPEG. --capture JSON
states the descriptor instead (model apply's keys). Each input writes

    <stem>.tif         16-bit sRGB-encoded TIFF: the photo to edit or print
    <stem>.jpg         8-bit sRGB preview (quality 92)
    <stem>.json        what was done: input, decode, descriptor and the law behind it, the DNG's own NoiseProfile
                       when it has one (beside the blind law, for comparison), checkpoint, exposure, timings
    <stem>_linear.tif  with --linear: the 32-bit float linear frame, 1.0 the sensor's clip (scene-referred)

--ev shifts the exposure of the encoded files by stops; `auto` lifts the 99th percentile of the brightest channel
to 1.0 and never darkens (LibRaw's auto-brightening rule). Values above 1.0 clip in the encoded files only.
--original also writes the decode unrestored (<stem>_original.tif/.jpg) at the same exposure, to compare."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

RAW_EXT = {".dng", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf", ".orf", ".rw2", ".rwl",
           ".pef", ".srw", ".3fr", ".fff", ".iiq", ".x3f", ".erf", ".kdc", ".dcr", ".mos", ".mrw", ".raw", ".mef"}
JPEG_QUALITY = 92
AUTO_PERCENTILE = 99.0


def inputs_of(paths) -> list[Path]:
    """Files as given, directories expanded to their raws and images (not recursive), sorted, each once."""
    from .io import IMAGE_EXT
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in RAW_EXT | IMAGE_EXT)
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(p)
    seen, uniq = set(), []
    for p in out:
        k = p.resolve()
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def is_raw(path: Path) -> bool:
    return Path(path).suffix.lower() in RAW_EXT


def dng_noise_profile(path: Path) -> dict | None:
    """The DNG's NoiseProfile as gain and read noise in electrons (dng.sensor_summary), None when absent."""
    if Path(path).suffix.lower() != ".dng":
        return None
    try:
        from .data.dng import read_tags, sensor_summary
        return sensor_summary(read_tags(path)).get("noise")
    except Exception:  # noqa: BLE001 - a comparison only
        return None


def load_photo(path: Path, scale: int = 1, decode=None) -> tuple[np.ndarray, dict]:
    """(H, W, 3) linear at 1.0 the clip and what the decode knows: raw or not, bit depth."""
    if is_raw(path):
        if decode is None:
            from .data.realpairs import decode_raw as decode
        x, bits = decode(Path(path), int(scale))
        return np.asarray(x, dtype=np.float64), {"raw": True, "bit_depth": int(bits), "scale": int(scale)}
    from .data.realpairs import box_downsample
    from .io import load_image
    x = load_image(path)
    return (box_downsample(x, int(scale)) if scale > 1 else x), {"raw": False, "bit_depth": 8, "scale": int(scale)}


def capture_for(x: np.ndarray, info: dict, stated: dict | None = None) -> tuple[np.ndarray | None, dict]:
    """The descriptor to condition on (None: withheld) and the record of how it was chosen."""
    from .model.data import capture_descriptor
    if stated is not None:
        return capture_descriptor(**stated), {"mode": "stated", "stated": stated}
    if not info["raw"]:
        return None, {"mode": "unknown", "why": "not a raw: the camera's own processing is unknown"}
    from .data.blindnoise import blind_law
    law = blind_law(x, scale=info["scale"])
    rec = {k: law.get(k) for k in ("a", "b", "electrons_at_clip", "read_noise_e", "bins", "flat_fraction",
                                    "blind_a", "blind_b", "var_scale", "smooth_sigma", "method")}
    if not law.get("electrons_at_clip"):
        return None, {"mode": "unknown", "why": "the noise law did not fit", "law": rec}
    cap = capture_descriptor(0.0, 0.0, law["electrons_at_clip"], law["read_noise_e"] or 0.0, info["bit_depth"], True)
    return cap, {"mode": "estimated", "law": rec}


def exposure_gain(x: np.ndarray, ev) -> float:
    if ev is None or str(ev) in ("", "0", "0.0"):
        return 1.0
    if str(ev) == "auto":
        p = float(np.percentile(np.asarray(x).max(axis=-1), AUTO_PERCENTILE))
        return max(1.0, 1.0 / p) if p > 0 else 1.0
    return float(2.0 ** float(ev))


def write_outputs(d: Path, stem: str, y: np.ndarray, gain: float, linear: bool, software: str) -> list[str]:
    from PIL import Image

    from .colorspace import linear_to_srgb
    from .tiff import write_tiff
    enc = linear_to_srgb(np.clip(np.asarray(y, dtype=np.float64) * gain, 0.0, 1.0))
    write_tiff(d / f"{stem}.tif", (enc * 65535.0 + 0.5).astype(np.uint16), software)
    Image.fromarray((enc * 255.0 + 0.5).astype(np.uint8)).save(d / f"{stem}.jpg", quality=JPEG_QUALITY)
    out = [f"{stem}.tif", f"{stem}.jpg"]
    if linear:
        write_tiff(d / f"{stem}_linear.tif", np.asarray(y, dtype=np.float32), software)
        out.append(f"{stem}_linear.tif")
    return out


def restore_photo(b, path: Path, out: Path, scale: int = 1, ev=None, target: dict | None = None,
                  stated: dict | None = None, linear: bool = False, original: bool = False, decode=None) -> dict:
    """One photo through the model; returns (and writes as <stem>.json) its record."""
    from .model.apply import restore_frames
    path, out = Path(path), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    x, info = load_photo(path, scale, decode)
    t1 = time.perf_counter()
    cap, how = capture_for(x, info, stated)
    t2 = time.perf_counter()
    y = restore_frames(b, x[None], target, 1.0, cap)[0]
    t3 = time.perf_counter()
    gain = exposure_gain(x, ev)
    software = f"harness photo (checkpoint step {b.step})"
    files = write_outputs(out, path.stem, y, gain, linear, software)
    if original:
        files += write_outputs(out, f"{path.stem}_original", x, gain, False, "harness photo (decode, unrestored)")
    rec = {"input": str(path), "size": [int(x.shape[1]), int(x.shape[0])], **info, "capture": how,
           "descriptor": None if cap is None else [float(v) for v in cap],
           "dng_noise_profile": dng_noise_profile(path), "checkpoint_step": int(b.step),
           "target": "stated" if target else "the mean restore target (calibration)", "exposure_gain": gain,
           "ev": ev, "files": files, "seconds": {"decode": round(t1 - t0, 2), "noise_law": round(t2 - t1, 2),
                                                 "restore": round(t3 - t2, 2),
                                                 "write": round(time.perf_counter() - t3, 2)}}
    (out / f"{path.stem}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def cmd_photo(args) -> int:
    from .model.apply import load_model
    files = inputs_of(args.inputs)
    if not files:
        print("no raw or image file among the inputs")
        return 2
    stems = [f.stem for f in files]
    if len(set(stems)) != len(stems):
        raise SystemExit("two inputs share a file stem; their outputs would collide in --out")
    target = json.loads(Path(args.target).read_text(encoding="utf-8")) if args.target else None
    stated = json.loads(Path(args.capture).read_text(encoding="utf-8")) if args.capture else None
    b = load_model(args.ckpt, args.device, args.bf16)
    print(f"checkpoint step {b.step} on {b.device}; {len(files)} photo(s) -> {args.out}")
    failed = 0
    for k, f in enumerate(files, 1):
        try:
            r = restore_photo(b, f, Path(args.out), args.scale, args.ev, target, stated, args.linear, args.original)
        except Exception as e:  # noqa: BLE001 - one bad file does not stop a folder
            failed += 1
            print(f"[{k}/{len(files)}] {f.name}: FAILED {type(e).__name__}: {e}")
            continue
        law = r["capture"].get("law") or {}
        noise = (f"{law['electrons_at_clip']:.0f} e at the clip, read {law['read_noise_e']:.1f} e"
                 if r["capture"]["mode"] == "estimated" else r["capture"]["mode"] + ": " + r["capture"].get("why", ""))
        print(f"[{k}/{len(files)}] {f.name}: {r['size'][0]}x{r['size'][1]}, {noise}; restored in "
              f"{r['seconds']['restore']:.1f} s -> {', '.join(r['files'])}")
    return 1 if failed else 0


def add_parser(sub) -> None:
    p = sub.add_parser("photo", help="restore camera raws or photos with a trained checkpoint: 16-bit TIFF, JPEG "
                                     "preview and a JSON record per input")
    p.add_argument("inputs", nargs="+", help="raw or image files, or directories of them")
    p.add_argument("--ckpt", required=True, help="a release directory (model export) or a training checkpoint (best.pt)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--scale", type=int, default=1, help="box-downsample the decode by this factor first (training's "
                                                       "real pairs are at 2)")
    p.add_argument("--ev", default=None, help="exposure shift in stops for the encoded files, or `auto`")
    p.add_argument("--linear", action="store_true", help="also write the 32-bit float linear TIFF")
    p.add_argument("--original", action="store_true", help="also write the unrestored decode, to compare")
    p.add_argument("--capture", help="JSON stating the capture descriptor (model apply's keys) instead of reading it")
    p.add_argument("--target", help="JSON of target metrics (an edit); default the mean restore target")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda when present)")
    p.add_argument("--bf16", action="store_true", help="bf16 autocast (faster; can band smooth gradients)")
    p.set_defaults(func=cmd_photo)
