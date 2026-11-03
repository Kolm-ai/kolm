"""The oracle linear frontier of an evaluation set: what the best shift-invariant linear filter reaches at each
noise level, the bar a restoration is judged against at its own shadow SNR (heldout.score's gates["frontier"]).

Why: on noisy real pairs the capture-relative gates ask for the impossible. Denoising trades detail for noise, so
even the best linear filter, fitted on the truth itself, falls below the noisy capture on mtf50 and texture (on
RawNIND: shadow SNR 14.7 -> 26.7 dB takes mtf50 0.39 -> 0.15, texture 0.67 -> 0.11), and CAPTURE_GATES' resolution
and detail checks fail any denoiser. The fair question is whether the restoration keeps more than the oracle does
at the same noise.

The ladder: per tile and channel, a parametric Wiener filter fitted on the truth, gains in NB radial bands
G = S_xy / (S_xy^2 / S_yy + lam (S_xx - S_xy^2 / S_yy)) from the tiles' Hann-windowed spectra (S_xy = Re<X Y*>,
the signal part of the input's power S_xy^2 / S_yy, the rest noise), clipped to [0, G_MAX], the lowest band kept at
1 so the tile's mean and shading stay the input's, applied on symmetric 2x padding. lam 1 is the MMSE filter, lam 0
the inverse filter (deblurs, keeps the noise), a larger lam trades more detail for less noise. Every rung is scored
with tile_metrics against the truth under the evaluation's noise reading and cached per set (a hash of its tiles),
noise reading and ladder.

The match is per tile: each tile's output shadow SNR is located on that tile's own ladder (linear between the rungs
that bracket it, clamped to the ends and counted), and every other metric is read off the ladder there; a tile
with no shadow SNR of its own takes the rung position where the set's median ladder meets the output's median SNR.
FRONTIER_GATES judge the output's medians against those matched medians, every metric tie or beat (mtf50 and
texture by distance to the truth), shadow SNR itself matched and so not judged."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..data.tilemetrics import tile_metrics
from ..gate import overall, run_gates
from ..metrics import STILL_METRICS, verdict

LAMS = (0.0, 0.03, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0, 8.0, 13.0, 20.0, 32.0)  # up to heavy smoothing
NB = 32            # radial bands from DC to Nyquist
G_MAX = 8.0        # the inverse filter's ceiling where the input holds almost none of the truth's signal
FRONTIER_DIR = "frontier"   # <data_root>/derived/frontier/<key>.json
VERSION = 1
MATCH = "shadow_snr_db"
TOB = "ties_or_beats_camera"


def _g(name: str, *metrics: str) -> dict:
    return {"name": name, "checks": [{"metric": m, "op": TOB} for m in metrics]}


FRONTIER_GATES = [
    _g("lighting_materials", "shading_rmse", "landmark_error_px"),
    _g("tone_color", "delta_e_mean", "delta_e_max", "dr_steps"),
    _g("artifacts", "edge_overshoot", "periodicity", "banding_score"),
    _g("resolution", "mtf50_cpp"),
    _g("micro_detail", "texture_hf_fidelity"),
]


def _bands(h: int, w: int, nb: int) -> np.ndarray:
    r = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :])
    return np.minimum((r / (0.5 / nb)).astype(int), nb - 1)


def gains(x: np.ndarray, y: np.ndarray, lam: float, nb: int = NB) -> np.ndarray:
    """(channels, nb) rung-lam gains of x (the input) towards y (the truth), both linear (h, w, c)"""
    h, w, ch = x.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    idx = _bands(h, w, nb).ravel()
    out = np.empty((ch, nb))
    for c in range(ch):
        X = np.fft.fft2((x[..., c] - x[..., c].mean()) * win)
        Y = np.fft.fft2((y[..., c] - y[..., c].mean()) * win)
        sxy = np.maximum(np.bincount(idx, np.real(X * np.conj(Y)).ravel(), nb), 0.0)
        sxx = np.bincount(idx, (np.abs(X) ** 2).ravel(), nb)
        syy = np.bincount(idx, (np.abs(Y) ** 2).ravel(), nb)
        sig = np.where(syy > 1e-30, sxy ** 2 / np.maximum(syy, 1e-30), 0.0)
        den = sig + lam * np.maximum(sxx - sig, 0.0)
        g = np.clip(np.where(den > 1e-30, sxy / np.maximum(den, 1e-30), 0.0), 0.0, G_MAX)
        g[0] = 1.0
        out[c] = g
    return out


def wiener(x: np.ndarray, y: np.ndarray, lam: float, nb: int = NB) -> np.ndarray:
    """x through the rung-lam filter fitted on y"""
    h, w, ch = x.shape
    g = gains(x, y, lam, nb)
    idx = _bands(2 * h, 2 * w, nb)
    o = np.empty(x.shape, dtype=np.float64)
    for c in range(ch):
        X = np.fft.fft2(np.pad(x[..., c], ((0, h), (0, w)), mode="symmetric"))
        o[..., c] = np.real(np.fft.ifft2(X * g[c][idx]))[:h, :w]
    return o


def ladder(samples: list[dict], noise: str = "residual", lams=LAMS, log=None) -> dict:
    """Every rung's tile metrics on every sample: {"lams", "noise", "version", "rows": [rung][tile] -> metrics}"""
    rows = []
    for lam in lams:
        rows.append([tile_metrics(wiener(s["x"], s["y"], lam), s["y"], noise) for s in samples])
        if log:
            log(f"  frontier lam {lam:g}: median {MATCH} {_median(r[MATCH] for r in rows[-1])}")
    return {"lams": list(lams), "noise": noise, "version": VERSION, "rows": rows}


def key(samples: list[dict], noise: str, lams=LAMS) -> str:
    """the cache key: the ladder's definition and every tile of the set"""
    h = hashlib.sha256(f"v{VERSION}|{noise}|{','.join(map(str, lams))}|{NB}|{G_MAX}".encode())
    for s in samples:
        for a in (s["x"], s["y"]):
            h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()[:20]


def cached_ladder(samples: list[dict], root: Path, noise: str = "residual", lams=LAMS, log=print) -> dict:
    """ladder(), read from <root>/derived/frontier/ when this set, noise reading and ladder were scored before"""
    p = Path(root) / "derived" / FRONTIER_DIR / f"{key(samples, noise, lams)}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    if log:
        log(f"frontier: {len(lams)} rungs on {len(samples)} tiles -> {p}")
    t = ladder(samples, noise, lams, log)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(t), encoding="utf-8")
    tmp.replace(p)
    return t


def _finite(v) -> bool:
    return v is not None and bool(np.isfinite(v))


def _median(vals) -> float | None:
    v = [x for x in vals if _finite(x)]
    return float(np.median(v)) if v else None


def matched(table: dict, outputs: list[dict]) -> tuple[list[dict], dict]:
    """Per tile, the ladder's metrics at that tile's output shadow SNR (module docstring), and how the match went:
    the output's median SNR, the median rung position (0 = the first rung, fractional between rungs), the median
    ladder's SNR range, tiles clamped to an end of their own ladder, tiles matched on the set's median instead and
    tiles left unmatched (no shadow SNR on the output or the ladder to place them by; their frontier rows are None)."""
    rows, nr = table["rows"], len(table["lams"])
    if any(len(r) != len(outputs) for r in rows):
        raise ValueError(f"the ladder has {len(rows[0])} tiles, the outputs {len(outputs)}")
    pos = np.arange(nr, dtype=np.float64)
    out_snr = _median(o.get(MATCH) for o in outputs)
    med_ladder = np.array([np.nan if (v := _median(r[k][MATCH] for k in range(len(outputs)))) is None else v
                           for r in rows])
    ok = np.isfinite(med_ladder)
    fallback = None
    if out_snr is not None and ok.sum() >= 2:
        o = np.argsort(med_ladder[ok])
        fallback = float(np.interp(out_snr, med_ladder[ok][o], pos[ok][o]))
    base, at, clamped, fell_back, unmatched = [], [], 0, 0, 0
    for k, out in enumerate(outputs):
        snr = np.array([rows[i][k][MATCH] for i in range(nr)], dtype=np.float64)
        fin = np.isfinite(snr)
        s = out.get(MATCH)
        if _finite(s) and fin.sum() >= 2:
            o = np.argsort(snr[fin])
            xs, ps = snr[fin][o], pos[fin][o]
            clamped += int(s < xs[0] or s > xs[-1])
            t = float(np.interp(s, xs, ps))
        elif fallback is not None:
            fell_back += 1
            t = fallback
        else:
            base.append({m: None for m in STILL_METRICS})
            unmatched += 1
            continue
        at.append(t)
        r = {}
        for m in STILL_METRICS:
            v = np.array([np.nan if rows[i][k][m] is None else rows[i][k][m] for i in range(nr)], dtype=np.float64)
            f = np.isfinite(v)
            r[m] = float(np.interp(t, pos[f], v[f])) if f.any() else None
        base.append(r)
    info = {"output_snr": out_snr, "rung": _median(at), "lams": table["lams"], "noise": table["noise"],
            "ladder_snr": [float(np.nanmin(med_ladder)), float(np.nanmax(med_ladder))] if ok.any() else None,
            "clamped": clamped, "set_median_match": fell_back, "unmatched": unmatched, "tiles": len(outputs)}
    return base, info


def gates(table: dict, outputs: list[dict], truths: list[dict]) -> dict:
    """FRONTIER_GATES on the output's medians against the matched frontier's, with the match's info and a verdict
    per metric (every metric, not only those up to the first failing gate)"""
    base, info = matched(table, outputs)
    med = {side: {m: _median(r[m] for r in rs) for m in STILL_METRICS}
           for side, rs in (("output", outputs), ("frontier", base), ("truth", truths))}
    res = run_gates(FRONTIER_GATES, med["output"], med["frontier"], med["truth"], "frontier")
    metrics = {}
    for m in STILL_METRICS:
        o, f, t = med["output"][m], med["frontier"][m], med["truth"][m]
        metrics[m] = {"frontier": f, "output": o, "truth": t,
                      "verdict": verdict(m, o, f, t)[0] if o is not None and f is not None and m != MATCH else None}
    return {"overall": overall(res), "gates": [g.to_dict() for g in res], "match": info, "metrics": metrics}
