"""A capture's noise law read from the capture alone, for a photo with no reference (python -m harness photo).

realpairs.noise_law measures var(x - t) on a real pair's flat pixels, t the gain-matched reference, and that law
is the capture descriptor training saw for a real pair (realpairs.capture_of). A photo has no t. Here t is the
capture smoothed by a Gaussian of `sigma` px, so x - t holds the noise's high frequencies, and the law is read from
that residual one of two ways:

    "mad"     noise_law's bins, flatness rule and robust (MAD) variance over every flat pixel of a bin
    "blocks"  the residual's variance in BLOCK x BLOCK cells, binned by the cell's smoothed luminance, a bin's
              variance the QUANTILE of its cells': a textured cell only raises the upper quantiles, so the floor
              the noise sets is read past the texture a flat-pixel rule lets through

and the law is multiplied by VAR_SCALE[method]: the median over a real store's train pairs of the measured slope
over the blind one (calibrate). That one number holds the share of the noise the smoothing keeps, the demosaic's
correlation between neighbours, the quantile's bias and what texture still adds; calibrate reports how far a
pair's blind electrons sit from its measured ones on the other split. A law that does not fit (fewer than three
bins, a slope at or below zero) is no law: the photo command then withholds the descriptor (UNKNOWN_CAPTURE), the
case training covers for a web JPEG."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SMOOTH_SIGMA = {"mad": 2.0, "blocks": 1.0}   # px: the Gaussian whose residual carries the noise
BLOCK = 8              # px: a "blocks" cell's side
QUANTILE = 0.10        # the quantile of a bin's cell variances read as its noise variance
MIN_CELLS = 24         # cells a "blocks" bin needs
METHOD = "blocks"
# measured variance over blind variance per method, the train medians of calibrate() on 2026-10-29: 289 stored
# pairs (up to 60 train and 60 val per source of rawnind, hdr-plus-burst and si-hdr), each read at its centre 1024
# crop. CALIBRATION, log10(blind / measured electrons) on the val pairs, p10 / median / p90:
#   blocks, sigma 1, q 0.10, 8 px  -0.569 / +0.018 / +0.271  (train scatter p90 - p10 1.677)  <- the default
#   blocks, sigma 1, q 0.25        -0.777 / +0.017 / +0.319  (1.800)
#   blocks, sigma 2, q 0.10        -0.705 / +0.027 / +0.307  (1.822)
#   blocks, sigma 2, q 0.25        -0.869 / +0.047 / +0.379  (2.032)
#   blocks, sigma 2, q 0.25, 16 px -0.908 / +0.059 / +0.411  (1.962)
#   mad, sigma 2                   -0.809 / +0.002 / +0.356  (1.744)
# chosen by the train scatter. The sources disagree on the scale (train medians rawnind 2.09, hdr-plus-burst 1.12,
# si-hdr 0.47 for the default), so a capture's error is mostly which camera and scene it is: val per source, the
# default: rawnind +0.19 (-0.32 to +0.31), hdr-plus-burst -0.13 (-0.85 to +0.05), si-hdr +0.12 (-0.87 to +0.23).
VAR_SCALE = {"mad": 0.5014, "blocks": 1.2310}
CALIB_SCALE = 2        # the decode scale the stores hold, at which VAR_SCALE was calibrated
# VAR_SCALE's factor at another decode scale (calibrate_decoded on train pairs decoded at that scale): at scale 1 the
# demosaic's correlation between neighbours is not averaged away by the 2 x 2 box, so the residual keeps less of
# the noise and the stored scale's VAR_SCALE reads a capture cleaner than it is. A scale not here uses CALIB_SCALE's.
# CALIBRATION (29 October 2026): blocks at scale 1 on 40 train pairs (centre 2048 px), log10(measured / blind a)
# median +0.231, p10 -0.159, p90 +0.461; per source rawnind 2.52 (n 20), hdr-plus-burst 1.42 (n 20).
SCALE_FACTOR: dict[tuple[str, int], float] = {("blocks", 1): 1.7019}
CALIB_CROP = 1024      # px: the centre crop a calibration pair is read at (memmapped, so only that much is read)
BIN_EDGES = np.geomspace(0.005, 0.9, 17)   # noise_law's luminance bins


def smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return np.stack([gaussian_filter(x[..., c], sigma, mode="reflect") for c in range(x.shape[-1])], axis=-1)


def fit_law(levels, variances, counts) -> dict:
    """var = a t + b by least squares weighted by sqrt(counts), as noise_law fits it; electrons at the clip 1/a,
    read noise sqrt(b)/a. Fewer than three bins, or a slope at or below zero, has no electrons."""
    out = {"a": None, "b": None, "electrons_at_clip": None, "read_noise_e": None, "bins": len(levels)}
    if len(levels) < 3:
        return out
    w = np.sqrt(np.asarray(counts, dtype=np.float64))
    design = np.stack([np.asarray(levels, dtype=np.float64), np.ones(len(levels))], axis=1) * w[:, None]
    (a, b), *_ = np.linalg.lstsq(design, np.asarray(variances, dtype=np.float64) * w, rcond=None)
    out.update(a=float(a), b=float(b))
    if a > 0:
        out.update(electrons_at_clip=float(1.0 / a), read_noise_e=float(math.sqrt(max(b, 0.0)) / a))
    return out


def blocks_law(x: np.ndarray, sigma: float = SMOOTH_SIGMA["blocks"], block: int = BLOCK,
               q: float = QUANTILE) -> dict:
    """The "blocks" reading (unscaled): cells whose capture is unclipped, binned by BIN_EDGES on the smoothed
    luminance, each bin's variance the q quantile of its cells' residual variance (mean over channels)."""
    from ..colorspace import luminance
    from .realpairs import CLIP
    x = np.asarray(x, dtype=np.float32)
    t = np.clip(smooth(x, sigma), 0.0, 1.0)
    hb, wb = x.shape[0] // block, x.shape[1] // block

    def cells(a):
        a = a[:hb * block, :wb * block]
        return a.reshape(hb, block, wb, block, -1).swapaxes(1, 2).reshape(hb, wb, block * block, a.shape[-1])

    v = cells(x - t).var(axis=2).mean(axis=-1)
    lum = cells(luminance(t)[..., None]).mean(axis=(2, 3))
    ok = cells(x).max(axis=(2, 3)) < CLIP
    idx = np.digitize(lum, BIN_EDGES)
    lv, vv, nn = [], [], []
    for b in range(1, len(BIN_EDGES)):
        sel = ok & (idx == b)
        n = int(sel.sum())
        if n < MIN_CELLS:
            continue
        lv.append(float(lum[sel].mean()))
        vv.append(float(np.quantile(v[sel], q)))
        nn.append(n)
    return {**fit_law(lv, vv, nn), "levels": lv, "variances": vv, "counts": nn, "flat_fraction": float(ok.mean())}


def mad_law(x: np.ndarray, sigma: float = SMOOTH_SIGMA["mad"]) -> dict:
    """The "mad" reading (unscaled): noise_law on the residual from the smoothed capture, robust."""
    from .realpairs import CLIP, noise_law
    x = np.asarray(x, dtype=np.float64)
    return noise_law(x, np.clip(smooth(x, sigma), 0.0, 1.0), x.max(axis=-1) < CLIP, robust=True)


def var_scale_for(method: str = METHOD, scale: int = CALIB_SCALE) -> float:
    """VAR_SCALE[method] for a frame decoded at `scale`."""
    return VAR_SCALE[method] * SCALE_FACTOR.get((method, int(scale)), 1.0)


def blind_law(x: np.ndarray, var_scale: float | None = None, sigma: float | None = None, method: str = METHOD,
              scale: int = CALIB_SCALE, **kw) -> dict:
    """(H, W, 3) linear at 1.0 the clip, decoded at `scale` -> the law var = a t + b with a and b times var_scale
    (var_scale_for(method, scale) by default), electrons at the clip and read noise from those, plus the unscaled
    fit (blind_a, blind_b)."""
    sigma = SMOOTH_SIGMA[method] if sigma is None else sigma
    var_scale = var_scale_for(method, scale) if var_scale is None else var_scale
    law = blocks_law(x, sigma, **kw) if method == "blocks" else mad_law(x, sigma)
    a, b = law["a"], law["b"]
    law.update(blind_a=a, blind_b=b, var_scale=float(var_scale), smooth_sigma=float(sigma), method=method,
               a=None, b=None, electrons_at_clip=None, read_noise_e=None)
    if a is not None and a > 0:
        a, b = a * var_scale, b * var_scale
        law.update(a=float(a), b=float(b), electrons_at_clip=float(1.0 / a),
                   read_noise_e=float(math.sqrt(max(b, 0.0)) / a))
    return law


def _spread(e) -> dict:
    e = np.asarray(e, dtype=np.float64)
    return ({"n": int(len(e)), "median": float(np.median(e)), "p10": float(np.percentile(e, 10)),
             "p90": float(np.percentile(e, 90))} if len(e) else {"n": 0})


def calibrate(sources: list[str], root: Path | None = None, n: int = 60, seed: int = 0, log=print,
              variants: dict | None = None, pairs: list[dict] | None = None) -> dict:
    """Blind laws (unscaled) of up to n stored inputs per source and split (train, val), each read at its centre
    CALIB_CROP crop, against the laws their pairs measured, for each of `variants` ({name: blind_law kwargs},
    default the two methods at their defaults). `pairs` ([{source, split, pair}], a report's rows) reads those
    pairs instead of drawing. Per variant: the scale (the median of measured a over blind a on the train pairs),
    the train pairs' scatter of log10(measured a / blind a) about it (p90 - p10, the number to choose a variant
    by), and on the val pairs the median and the 10th and 90th percentiles of log10(blind electrons / measured
    electrons) with that scale applied, per source and over all."""
    from .realstore import store_dir
    variants = variants or {"mad": {"method": "mad"}, "blocks": {"method": "blocks"}}
    if pairs is None:
        rng = np.random.default_rng(seed)
        pairs = []
        for source in sources:
            for split in ("train", "val"):
                metas = []
                for j in sorted(store_dir(source, root).glob("*/*.json")):
                    m = json.loads(j.read_text(encoding="utf-8"))
                    if m.get("split") == split and "rejected" not in m and (m.get("noise") or {}).get("a"):
                        metas.append(j)
                pairs += [{"source": source, "split": split,
                           "pair": metas[int(k)].relative_to(store_dir(source, root)).as_posix()}
                          for k in rng.permutation(len(metas))[:n]]
    rows = []
    for k, p in enumerate(pairs):
        j = store_dir(p["source"], root) / p["pair"]
        m = json.loads(j.read_text(encoding="utf-8"))
        x = np.load(j.with_suffix(".npy"), mmap_mode="r")
        h, w = x.shape[:2]
        c = min(CALIB_CROP, h, w)
        y0, x0 = (h - c) // 2, (w - c) // 2
        crop = np.asarray(x[y0:y0 + c, x0:x0 + c], dtype=np.float64)
        row = {**p, "measured_a": m["noise"]["a"], "measured_b": m["noise"]["b"]}
        for name, kw in variants.items():
            law = blind_law(crop, var_scale=1.0, **kw)
            row[name] = law["blind_a"] if law["blind_a"] and law["blind_a"] > 0 else None
        rows.append(row)
        if (k + 1) % 50 == 0:
            log(f"{k + 1}/{len(pairs)} pair(s) read")
    rep = {"crop": CALIB_CROP, "pairs": len(rows), "variants": {}, "rows": rows}
    for name, kw in variants.items():
        ok = [r for r in rows if r[name]]
        train = [math.log10(r["measured_a"] / r[name]) for r in ok if r["split"] == "train"]
        if not train:
            rep["variants"][name] = {"kwargs": kw, "fitted": 0}
            continue
        scale = 10.0 ** float(np.median(train))

        def err(rs):
            return _spread([math.log10(r["measured_a"] / (r[name] * scale)) for r in rs])

        val = [r for r in ok if r["split"] == "val"]
        rep["variants"][name] = {
            "kwargs": kw, "var_scale": scale, "fitted": len(ok), "of": len(rows),
            "train_scatter": float(np.percentile(train, 90) - np.percentile(train, 10)),
            "val_log10_electrons_error": err(val),
            "per_source": {s: {"train_scale": (10.0 ** float(np.median(
                                   [math.log10(r["measured_a"] / r[name]) for r in ok
                                    if r["source"] == s and r["split"] == "train"]))
                                   if any(r["source"] == s and r["split"] == "train" for r in ok) else None),
                               "val": err([r for r in val if r["source"] == s])} for s in sources}}
    return rep


def calibrate_decoded(sources: list[str], scale: int, n: int = 20, seed: int = 0, root: Path | None = None,
                      log=print, method: str = METHOD, splits=("train",)) -> dict:
    """SCALE_FACTOR[(method, scale)]: up to n pairs per source of `splits` (train: the held-out splits stay for
    capture-eval) decoded and matched at `scale` as capture-eval decodes them (realpairs.match_of), each read
    blind at its centre crop (CALIB_CROP at the stored scale's field of view) with var_scale_for(method,
    CALIB_SCALE), against the law the pair measures. The factor is the median of measured a over that blind a;
    the report has per pair the log10 ratio and per source its median."""
    from . import data_root
    from .realpairs import decode_raw, match_of, pairs_of
    root = Path(root or data_root())
    crop = CALIB_CROP * CALIB_SCALE // int(scale)
    rng = np.random.default_rng(seed)
    rows = []
    for source in sources:
        pairs = [p for p in pairs_of(source, root, log) if p.split in splits]
        got = 0
        for i in rng.permutation(len(pairs)):
            if got >= n:
                break
            p = pairs[int(i)]
            try:
                m, _, _, _ = match_of(p, root / "raw" / source, int(scale), decode_raw)
            except Exception as ex:  # noqa: BLE001 - not a pair at this scale; the next one is drawn
                log(f"{source}: {p.in_file}: {type(ex).__name__}: {str(ex)[:160]}")
                continue
            e = (m.get("noise") or {}).get("electrons_at_clip")
            if not e:
                continue
            x = np.asarray(m["x"], dtype=np.float32)
            h, w = x.shape[:2]
            c = min(crop, h, w)
            y0, x0 = (h - c) // 2, (w - c) // 2
            law = blind_law(x[y0:y0 + c, x0:x0 + c], var_scale=var_scale_for(method, CALIB_SCALE), method=method)
            del m, x
            if not law["a"]:
                continue
            got += 1
            r = math.log10((1.0 / e) / law["a"])
            rows.append({"source": source, "split": p.split, "in_file": p.in_file, "measured_electrons": e,
                         "blind_electrons": law["electrons_at_clip"], "log10_measured_over_blind_a": r})
            log(f"{source}: [{got}/{n}] {Path(p.in_file).name}: measured {e:.0f} e, blind {law['electrons_at_clip']:.0f} e")
    lr = [r["log10_measured_over_blind_a"] for r in rows]
    return {"scale": int(scale), "method": method, "crop": crop, "splits": list(splits), "pairs": len(rows),
            "factor": (10.0 ** float(np.median(lr))) if lr else None, "spread": _spread(lr),
            "per_source": {s: _spread([r["log10_measured_over_blind_a"] for r in rows if r["source"] == s])
                           for s in sources}, "rows": rows}
