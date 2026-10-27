"""Noise-model validation on SIDD burst stacks (DATA_PLAN section 4 step 5).

camera.py says a pixel's temporal variance is shot noise plus read noise: in electrons var = e + sigma_r^2,
so in raw DN var = (1/g) * (DN - black) + sigma_r^2 / g^2 with g the gain in e-/DN. SIDD gives 150 frames of
one static scene per capture instance (Abdelhamed, Lin and Brown, CVPR 2018), so the per-pixel temporal mean
and variance are measured directly, no denoiser and no reference image involved. This module

* reads frames from the instance zip (rawpy, no extraction), tracks a per-pixel running mean and M2
  (Welford), per CFA position;
* first pass: per-frame mean of each CFA plane; frames whose mean sits off the burst (more than 1 % or
  5 robust sigma from the median: an unsettled first exposure) are dropped; every kept frame is then
  scaled to the burst mean of its plane, so lamp drift and flicker between frames (which add variance
  proportional to mean squared) drop out while per-pixel noise stays; the drift itself is reported;
* flat pixels only for the fit: where the mean image has a gradient above twice the fitted noise, any
  motion between frames turns the edge into temporal variance (heavy tails, inflated intercept); the
  fit, the distribution and the line-noise checks use the flat pixels, and the excess at the edges is
  reported as its own number;
* bins pixels by temporal mean and takes the median temporal variance per bin (robust to hot pixels),
  divided by the median of chi-square(n-1)/(n-1) so the median is unbiased for the variance;
* fits var = a * mean + b to the bins below the clipping region: g = 1/a e-/DN, sigma_r = sqrt(b) * g e-,
  full well = (white - black) * g e-; reports the linear R^2 and the quadratic term's share at mid range;
* checks the error distribution (excess kurtosis of standardised residuals, fraction beyond 4 sigma),
  row noise (variance of row means over the pixel variance / width) and exposure drift across the burst;
* across ISO for one camera, fits log(gain) against log(ISO) (shot noise ∝ gain predicts slope −1) and
  reads whether the read noise in electrons stays put.

Everything here is a plain function of arrays so the tests run on stacks simulated with camera.py's own model.
"""
from __future__ import annotations

import csv
import io
import math
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np

METHOD = ("SIDD burst stacks (150 raw frames of one static scene per instance) read with rawpy from the zip; "
          "frames whose plane mean is off the burst by more than 1 % or 5 robust sigma dropped; each kept frame "
          "scaled per CFA plane to the burst mean (illumination drift and flicker removed, the drift reported); "
          "per-pixel temporal mean and variance per CFA position (Welford); pixels where the mean image has a "
          "gradient above 2x the fitted noise excluded (motion between frames lives at edges; their excess "
          "variance ratio reported); the rest binned by mean, median "
          "variance per bin corrected by the chi-square median factor (1 - 2/(9(n-1)))^3, weighted linear fit "
          "var = a*mean + b on bins between 2 % and 70 % of the span "
          "above black; gain g = 1/a e-/DN, read noise sqrt(b)*g e-, full well (white-black)*g e-. Linearity: "
          "R^2 of the fit and the quadratic term's share of the variance at mid range. Distribution: excess "
          "kurtosis and the fraction beyond 4 sigma of residuals standardised by the fitted variance. Row noise: "
          "variance of row means of (frame - mean), the frame mean removed, over pixel variance / row length "
          "(1 = none), and the same for columns. Drift: the "
          "slope of frame means across the burst in units of the per-frame noise of that mean. Across ISO: "
          "slope of log gain against log ISO (shot noise ∝ gain predicts -1) and the spread of read noise in e-.")

DEFAULT_FRAMES = 60
FIT_LO, FIT_HI = 0.02, 0.70  # fit window as a fraction of (white - black) above black
N_BINS = 48


# ----------------------------------------------------------------------------------------------------------
# stack statistics


class Welford:
    """Running per-pixel mean and M2 over frames; also keeps the frame means and row-noise accumulators."""

    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = None
        self.frame_means: list[float] = []

    def add(self, frame: np.ndarray) -> None:
        x = np.asarray(frame, dtype=np.float64)
        self.frame_means.append(float(x.mean()))
        if self.mean is None:
            self.mean = x.copy()
            self.m2 = np.zeros_like(x)
            self.n = 1
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def variance(self) -> np.ndarray:
        return self.m2 / max(self.n - 1, 1)


def cfa_split(img: np.ndarray) -> dict[str, np.ndarray]:
    """The four Bayer positions of a mosaic as separate planes (keys '00', '01', '10', '11')."""
    return {f"{cy}{cx}": img[cy::2, cx::2] for cy in range(2) for cx in range(2)}


def chi2_median_factor(dof: int) -> float:
    """Median of chi-square(dof)/dof (Wilson-Hilferty): a sample variance over n frames has dof = n - 1, so the
    median of per-pixel sample variances sits this far below the true variance (0.971 at 24 frames, 0.989 at 60)."""
    return (1.0 - 2.0 / (9.0 * dof)) ** 3 if dof > 0 else 1.0


def bin_variance(mean: np.ndarray, var: np.ndarray, black: float, white: float, n_bins: int = N_BINS,
                 n_frames: int | None = None) -> dict:
    """Median temporal variance per bin of temporal mean (DN above black), divided by the chi-square median
    factor when n_frames is given; the plain mean variance and counts per bin alongside."""
    corr = chi2_median_factor(n_frames - 1) if n_frames else 1.0
    m = mean.reshape(-1) - black
    v = var.reshape(-1)
    span = white - black
    edges = np.linspace(0.0, span, n_bins + 1)
    idx = np.clip(np.digitize(m, edges) - 1, 0, n_bins - 1)
    centres, medians, means, counts = [], [], [], []
    for b in range(n_bins):
        sel = idx == b
        c = int(sel.sum())
        if c < 50:
            continue
        centres.append(float(m[sel].mean()))
        medians.append(float(np.median(v[sel])) / corr)
        means.append(float(v[sel].mean()))
        counts.append(c)
    return {"mean_dn": centres, "var_median": medians, "var_mean": means, "count": counts,
            "median_correction": corr}


def fit_nlf(bins: dict, black: float, white: float, lo: float = FIT_LO, hi: float = FIT_HI) -> dict:
    """Weighted linear fit var = a*mean + b over bins inside the window; quadratic share and R^2 as checks."""
    m = np.asarray(bins["mean_dn"], dtype=float)
    v = np.asarray(bins["var_median"], dtype=float)
    w = np.sqrt(np.asarray(bins["count"], dtype=float))
    span = white - black
    sel = (m >= lo * span) & (m <= hi * span) & (v > 0)
    if sel.sum() < 4:
        return {"ok": False, "reason": f"{int(sel.sum())} usable bins in the fit window"}
    m, v, w = m[sel], v[sel], w[sel]
    A = np.stack([m, np.ones_like(m)], axis=1) * w[:, None]
    (a, b), *_ = np.linalg.lstsq(A, v * w, rcond=None)
    pred = a * m + b
    ss_res = float(np.sum(w * (v - pred) ** 2))
    ss_tot = float(np.sum(w * (v - np.average(v, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    A2 = np.stack([m * m, m, np.ones_like(m)], axis=1) * w[:, None]
    (q, a2, b2), *_ = np.linalg.lstsq(A2, v * w, rcond=None)
    mid = 0.5 * (lo + hi) * span
    quad_share = float(q * mid * mid / (q * mid * mid + a2 * mid + b2)) if (q * mid * mid + a2 * mid + b2) else float("nan")
    gain = 1.0 / a if a > 0 else float("nan")
    read_e = math.sqrt(max(b, 0.0)) * gain if math.isfinite(gain) else float("nan")
    return {"ok": True, "a_dn_per_dn": float(a), "b_dn2": float(b), "gain_e_per_dn": float(gain),
            "read_noise_e": float(read_e), "read_noise_dn": math.sqrt(max(b, 0.0)),
            "full_well_e": float(span * gain) if math.isfinite(gain) else float("nan"),
            "r2": float(r2), "quadratic_share_mid": quad_share, "n_bins_fit": int(sel.sum()),
            "fit_window_dn": [float(lo * span), float(hi * span)]}


def residual_shape(frames: np.ndarray, mean: np.ndarray, var_pred: np.ndarray) -> dict:
    """Standardised residuals of a few frames against the fitted variance: excess kurtosis and 4-sigma tail.
    Poisson plus Gaussian at these counts gives excess kurtosis near 0 and a tail near 6e-5."""
    sel = var_pred > 0
    z = ((frames[:, sel] - mean[sel][None]) / np.sqrt(var_pred[sel])[None]).reshape(-1)
    if z.size < 1000:
        return {"excess_kurtosis": float("nan"), "tail_4sigma": float("nan"), "n": int(z.size)}
    z = z - z.mean()
    s2 = float(np.mean(z * z))
    k = float(np.mean(z ** 4) / (s2 * s2)) - 3.0 if s2 > 0 else float("nan")
    return {"excess_kurtosis": k, "tail_4sigma": float(np.mean(np.abs(z) > 4.0)), "n": int(z.size)}


def row_noise_ratio(frames: np.ndarray, mean: np.ndarray, var: np.ndarray, axis: int = 2,
                    mask: np.ndarray | None = None) -> float:
    """Variance of row means of (frame - mean), each frame's own mean removed first, relative to what
    independent pixels give (mean var / pixels per row). 1 means no row-correlated noise; 2 means row noise
    equal to the independent part. axis=1 gives the column version; mask restricts to flat pixels."""
    resid = frames - mean[None]
    if mask is None:
        mask = np.ones(mean.shape, dtype=bool)
    resid = np.where(mask[None], resid, 0.0)
    n_line = mask.sum(axis=axis - 1)                     # valid pixels per row (axis 2) or per column (axis 1)
    ok = n_line >= max(8, mask.shape[axis - 1] // 4)
    if ok.sum() < 4:
        return float("nan")
    resid = resid - resid.sum(axis=(1, 2), keepdims=True) / max(int(mask.sum()), 1)
    line_means = resid.sum(axis=axis)[:, ok] / n_line[ok][None]
    expected = float(np.mean(var[mask])) / float(n_line[ok].mean())
    return float(line_means.var() / expected) if expected > 0 else float("nan")


def outlier_frames(frame_means: list[float] | np.ndarray, rel: float = 0.01, k: float = 5.0) -> np.ndarray:
    """Boolean keep-mask: a frame whose mean is off the median by more than max(rel * median, k robust sigma)
    is an outlier (SIDD bursts start with an unsettled exposure now and then)."""
    m = np.asarray(frame_means, dtype=float)
    if m.size < 3:
        return np.ones(m.size, dtype=bool)
    med = float(np.median(m))
    mad = float(np.median(np.abs(m - med))) * 1.4826
    tol = max(rel * abs(med), k * mad)
    return np.abs(m - med) <= tol


def flat_mask(mean: np.ndarray, var_pred: np.ndarray, k: float = 2.0) -> np.ndarray:
    """Pixels whose mean image has a gradient (largest absolute neighbour difference) below k times the fitted
    noise sigma: where motion between frames cannot show up as temporal variance."""
    sigma = np.sqrt(np.maximum(var_pred, 1e-9))
    gy = np.abs(np.diff(mean, axis=0, prepend=mean[:1]))
    gx = np.abs(np.diff(mean, axis=1, prepend=mean[:, :1]))
    gy = np.maximum(gy, np.abs(np.diff(mean, axis=0, append=mean[-1:])))
    gx = np.maximum(gx, np.abs(np.diff(mean, axis=1, append=mean[:, -1:])))
    return np.maximum(gx, gy) < k * sigma


def frame_scales(frame_means: list[float] | np.ndarray) -> np.ndarray:
    """Multiplicative factor per frame that brings its mean signal above black to the burst mean."""
    m = np.asarray(frame_means, dtype=float)
    if m.size == 0 or not np.all(m > 0):
        return np.ones(m.size)
    return m.mean() / m


def drift_sigma(frame_means: list[float], n_pixels: int, mean_var: float) -> float:
    """Slope of the frame means over the burst, in units of the noise of one frame mean."""
    y = np.asarray(frame_means, dtype=float)
    if y.size < 3 or n_pixels <= 0 or not mean_var > 0:
        return float("nan")
    x = np.arange(y.size, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) * (y.size - 1)   # total change over the burst
    return slope / math.sqrt(mean_var / n_pixels)


# ----------------------------------------------------------------------------------------------------------
# SIDD instance


def sidd_metadata(csv_path: Path) -> dict[str, dict]:
    rows = {}
    with open(csv_path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows[r["instance"]] = {"camera": r["camera"], "camera_full": r["camera_full"], "iso": int(r["iso"]),
                                   "shutter": r["shutter"], "scene": r["scene_number"], "brightness": r["brightness"],
                                   "illuminant_k": int(r["illuminant_temp"]), "scene_name": r["scene_name"]}
    return rows


def read_dng_from_zip(z: zipfile.ZipFile, name: str):
    import rawpy
    with z.open(name) as fh:
        raw = rawpy.imread(io.BytesIO(fh.read()))
    img = raw.raw_image_visible.copy()
    info = {"black": [int(b) for b in raw.black_level_per_channel], "white": int(raw.white_level),
            "pattern": raw.raw_pattern.tolist(), "colors": raw.color_desc.decode() if isinstance(raw.color_desc, bytes) else str(raw.color_desc),
            "shape": list(img.shape)}
    raw.close()
    return img, info


class PlaneStats:
    """Streaming per-plane analysis: Welford over every frame, the first `keep` frames kept for the
    distribution and line-noise checks, so a 150-frame burst never sits in memory at once. `scales` are
    the per-frame illumination factors from a first pass (frame_scales); `raw_frame_means` the first-pass
    means, which carry the drift the scaling removes."""

    def __init__(self, black: float, white: float, keep: int = 8, scales: np.ndarray | None = None,
                 raw_frame_means: list[float] | None = None):
        self.black, self.white, self.keep = float(black), float(white), keep
        self.scales = None if scales is None else np.asarray(scales, dtype=float)
        self.raw_frame_means = raw_frame_means
        self.w = Welford()
        self.kept: list[np.ndarray] = []

    def add(self, plane: np.ndarray) -> None:
        x = np.asarray(plane, dtype=np.float64)
        if self.scales is not None:
            x = self.black + (x - self.black) * self.scales[self.w.n]
        self.w.add(x)
        if len(self.kept) < self.keep:
            self.kept.append(x)

    def finish(self) -> dict:
        mean, var = self.w.mean, self.w.variance()
        black, white = self.black, self.white
        bins = bin_variance(mean, var, black, white, n_frames=int(self.w.n))
        fit = fit_nlf(bins, black, white)
        out = {"n_frames": int(self.w.n), "n_pixels": int(mean.size), "black": black, "white": white,
               "mean_dn_above_black": float(mean.mean() - black),
               "clipped_fraction": float(np.mean(mean >= white - 1.0)), "bins": bins, "fit": fit,
               "drift_corrected": self.scales is not None}
        if self.scales is not None:
            out["frame_scale_min_max"] = [float(self.scales.min()), float(self.scales.max())]
        if fit["ok"]:
            var_pred = fit["a_dn_per_dn"] * (mean - black) + fit["b_dn2"]
            flat = flat_mask(mean, var_pred)
            out["fit_all_pixels"] = fit
            out["flat_fraction"] = float(flat.mean())
            bins_flat = bin_variance(mean[flat], var[flat], black, white, n_frames=int(self.w.n))
            fit_flat = fit_nlf(bins_flat, black, white)
            if fit_flat["ok"]:
                bins, fit = bins_flat, fit_flat
                out["bins"], out["fit"] = bins, fit
                var_pred = fit["a_dn_per_dn"] * (mean - black) + fit["b_dn2"]
            inside = (mean - black >= fit["fit_window_dn"][0]) & (mean - black <= fit["fit_window_dn"][1])
            edge = inside & ~flat & (var_pred > 0)
            # mean, not median: the gradient mask marks both sides of an edge and only the moving side carries excess
            out["edge_variance_ratio"] = float(np.mean(var[edge] / var_pred[edge])) if edge.sum() >= 50 else None
            var_pred = np.where(inside & flat, var_pred, 0.0)
            sub = np.stack(self.kept, axis=0)
            out["shape"] = residual_shape(sub, mean, var_pred)
            out["row_noise_ratio"] = row_noise_ratio(sub, mean, var, axis=2, mask=flat)
            out["column_noise_ratio"] = row_noise_ratio(sub, mean, var, axis=1, mask=flat)
            means = self.raw_frame_means if self.raw_frame_means is not None else self.w.frame_means
            out["drift_sigma"] = drift_sigma(means, mean.size, float(np.mean(var)))
            out["model_vs_measured"] = model_vs_measured(bins, fit)
        return out


def analyse_stack(frames: np.ndarray, black: float, white: float, keep_for_shape: int = 8,
                  correct_drift: bool = True, drop_outliers: bool = True) -> dict:
    """Full per-stack analysis for one plane (one CFA position): bins, fit, shape, line noise, drift.
    Two passes like the zip reader: frame means (outliers dropped) first, then the scaled Welford pass."""
    raw_means = np.array([float(np.mean(f)) for f in frames]) - black
    keep = outlier_frames(raw_means) if drop_outliers else np.ones(len(frames), dtype=bool)
    scales = frame_scales(raw_means[keep]) if correct_drift else None
    ps = PlaneStats(black, white, keep_for_shape, scales=scales, raw_frame_means=(raw_means[keep] + black).tolist())
    for f, k in zip(frames, keep):
        if k:
            ps.add(f)
    out = ps.finish()
    out["n_frames_dropped"] = int((~keep).sum())
    return out


def model_vs_measured(bins: dict, fit: dict) -> dict:
    """camera.py's prediction (Poisson + Gaussian with the fitted g, sigma_r) against the binned medians: the
    median absolute relative error inside the fit window and the worst bin."""
    m = np.asarray(bins["mean_dn"], dtype=float)
    v = np.asarray(bins["var_median"], dtype=float)
    lo, hi = fit["fit_window_dn"]
    sel = (m >= lo) & (m <= hi) & (v > 0)
    pred = fit["a_dn_per_dn"] * m[sel] + fit["b_dn2"]
    rel = (pred - v[sel]) / v[sel]
    return {"median_abs_rel_error": float(np.median(np.abs(rel))), "max_abs_rel_error": float(np.max(np.abs(rel))),
            "n_bins": int(sel.sum())}


def analyse_instance(zip_path: Path, meta: dict | None, n_frames: int = DEFAULT_FRAMES, log=None) -> dict:
    """One SIDD instance zip -> per-CFA-position fits plus a combined (green-average) headline."""
    t0 = time.time()
    inst = re.match(r"(\d{4})", zip_path.name).group(1)
    out = {"instance": inst, "zip": zip_path.name, "ok": False}
    out.update(meta or {})
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(".dng"))
            step = max(1, len(names) // n_frames)
            names = names[::step][:n_frames]
            colours: dict[str, str] = {}
            blacks: dict[str, float] = {}
            raw_means: dict[str, list[float]] = {}
            info = None
            for i, name in enumerate(names):                      # pass 1: header, per-plane frame means
                img, info_i = read_dng_from_zip(z, name)
                if info is None:
                    info = info_i
                    for key in cfa_split(img):
                        cy, cx = int(key[0]), int(key[1])
                        ci = info["pattern"][cy][cx]
                        colours[key] = info["colors"][ci] if ci < len(info["colors"]) else str(ci)
                        blacks[key] = float(info["black"][ci]) if ci < len(info["black"]) else float(np.mean(info["black"]))
                elif info_i["shape"] != info["shape"]:
                    raise ValueError(f"frame size changed inside the burst: {info_i['shape']} vs {info['shape']}")
                for key, plane in cfa_split(img).items():
                    raw_means.setdefault(key, []).append(float(plane.mean()))
                if log and (i + 1) % 20 == 0:
                    log(f"  {zip_path.name}: pass 1 {i + 1}/{len(names)} frames")
            keep = np.ones(len(names), dtype=bool)
            for key in colours:                                   # a frame off in any plane is dropped
                keep &= outlier_frames(np.asarray(raw_means[key]) - blacks[key])
            dropped = [names[i] for i in np.flatnonzero(~keep)]
            stats = {key: PlaneStats(blacks[key], float(info["white"]),
                                     scales=frame_scales((np.asarray(raw_means[key]) - blacks[key])[keep]),
                                     raw_frame_means=[m for m, k in zip(raw_means[key], keep) if k]) for key in colours}
            for i, name in enumerate(names):                      # pass 2: scaled Welford over the kept frames
                if not keep[i]:
                    continue
                img, _ = read_dng_from_zip(z, name)
                for key, plane in cfa_split(img).items():
                    stats[key].add(plane)
                if log and (i + 1) % 20 == 0:
                    log(f"  {zip_path.name}: pass 2 {i + 1}/{len(names)} frames")
            n_available = sum(1 for n in z.namelist() if n.lower().endswith(".dng"))
        out.update(raw=info, n_frames_used=int(keep.sum()), n_frames_available=n_available,
                   n_frames_dropped=len(dropped), frames_dropped=dropped)
        per = {}
        for key, ps in stats.items():
            res = ps.finish()
            res["colour"] = colours[key]
            per[key] = res
        out["cfa"] = per
        fits = [r["fit"] for r in per.values() if r["fit"].get("ok")]
        if fits:
            out["gain_e_per_dn"] = float(np.median([f["gain_e_per_dn"] for f in fits]))
            out["read_noise_e"] = float(np.median([f["read_noise_e"] for f in fits]))
            out["read_noise_dn"] = float(np.median([f["read_noise_dn"] for f in fits]))
            out["full_well_e"] = float(np.median([f["full_well_e"] for f in fits]))
            out["r2"] = float(min(f["r2"] for f in fits))
            # median over planes like the other fields: a sparse plane (few bright pixels of one colour) gives a
            # curvature number from three bins, which says nothing about the law
            out["quadratic_share_mid"] = float(np.median([abs(f["quadratic_share_mid"]) for f in fits]))
            out["excess_kurtosis"] = float(np.median([r["shape"]["excess_kurtosis"] for r in per.values() if "shape" in r]))
            out["tail_4sigma"] = float(np.median([r["shape"]["tail_4sigma"] for r in per.values() if "shape" in r]))
            out["row_noise_ratio"] = float(np.median([r["row_noise_ratio"] for r in per.values() if "row_noise_ratio" in r]))
            out["column_noise_ratio"] = float(np.median([r["column_noise_ratio"] for r in per.values() if "column_noise_ratio" in r]))
            out["flat_fraction"] = float(np.median([r["flat_fraction"] for r in per.values() if "flat_fraction" in r]))
            edges = [r["edge_variance_ratio"] for r in per.values() if r.get("edge_variance_ratio") is not None]
            out["edge_variance_ratio"] = float(np.median(edges)) if edges else None
            out["frame_scale_min_max"] = [min(r["frame_scale_min_max"][0] for r in per.values() if "frame_scale_min_max" in r),
                                          max(r["frame_scale_min_max"][1] for r in per.values() if "frame_scale_min_max" in r)]
            out["drift_sigma"] = float(np.median([r["drift_sigma"] for r in per.values() if "drift_sigma" in r]))
            out["model_median_abs_rel_error"] = float(np.median([r["model_vs_measured"]["median_abs_rel_error"]
                                                                 for r in per.values() if "model_vs_measured" in r]))
            out["ok"] = True
        else:
            out["error"] = "no CFA plane gave a usable fit: " + "; ".join(r["fit"].get("reason", "") for r in per.values())
    except Exception as ex:  # noqa: BLE001  (one bad zip must not stop the run)
        out["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ----------------------------------------------------------------------------------------------------------
# across instances


def iso_scaling(rows: list[dict]) -> dict:
    """Per camera: slope of log gain vs log ISO (expected -1), read noise in e- per ISO and its spread,
    full well in e- per ISO (expected constant: it is the sensor's, read through the gain)."""
    by_cam: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("ok") and r.get("iso"):
            by_cam.setdefault(r.get("camera_full") or r.get("camera") or "?", []).append(r)
    out = {}
    for cam, lst in by_cam.items():
        lst = sorted(lst, key=lambda r: r["iso"])
        iso = np.array([r["iso"] for r in lst], dtype=float)
        gain = np.array([r["gain_e_per_dn"] for r in lst], dtype=float)
        rn = np.array([r["read_noise_e"] for r in lst], dtype=float)
        fw = np.array([r["full_well_e"] for r in lst], dtype=float)
        entry = {"n_instances": len(lst), "iso": iso.tolist(), "gain_e_per_dn": gain.tolist(),
                 "read_noise_e": rn.tolist(), "full_well_e": fw.tolist(),
                 "instances": [r["instance"] for r in lst]}
        if len(set(iso.tolist())) >= 2:
            slope = float(np.polyfit(np.log(iso), np.log(gain), 1)[0])
            entry["log_gain_vs_log_iso_slope"] = slope
            entry["read_noise_e_min_max"] = [float(rn.min()), float(rn.max())]
            entry["full_well_e_median"] = float(np.median(fw))
            entry["full_well_e_spread"] = float((fw.max() - fw.min()) / np.median(fw))
            entry["unity_gain_iso"] = float(np.exp(np.interp(0.0, np.log(gain)[::-1], np.log(iso)[::-1]))) if gain.min() < 1 < gain.max() else None
        out[cam] = entry
    return out


def verdict(rows: list[dict], scaling: dict) -> dict:
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return {"linear_variance": None, "text": "no instance analysed"}
    r2 = [r["r2"] for r in ok]
    quad = [r["quadratic_share_mid"] for r in ok]
    kurt = [r["excess_kurtosis"] for r in ok]
    rowr = [r["row_noise_ratio"] for r in ok]
    colr = [r.get("column_noise_ratio", float("nan")) for r in ok]
    drift = [abs(r.get("drift_sigma", 0.0)) for r in ok]
    edges = [r["edge_variance_ratio"] for r in ok if r.get("edge_variance_ratio") is not None]
    dropped = sum(r.get("n_frames_dropped", 0) for r in ok)
    err = [r["model_median_abs_rel_error"] for r in ok]
    slopes = [s["log_gain_vs_log_iso_slope"] for s in scaling.values() if "log_gain_vs_log_iso_slope" in s]
    linear = min(r2) > 0.98 and max(abs(q) for q in quad) < 0.10
    per = {}  # the same checks per camera, so one camera's weak fit does not hide another's result
    for cam, s in scaling.items():
        cr = [r for r in ok if r.get("camera_full") == cam]
        if not cr:
            continue
        sl = s.get("log_gain_vs_log_iso_slope")
        lin_inst = [r["instance"] for r in cr if r["r2"] > 0.98 and abs(r["quadratic_share_mid"]) < 0.10]
        per[cam] = {"n_instances": len(cr), "r2_min": float(min(r["r2"] for r in cr)),
                    "quadratic_share_max": float(max(abs(r["quadratic_share_mid"]) for r in cr)),
                    "linear_variance": len(lin_inst) == len(cr), "instances_linear": lin_inst,
                    "log_gain_vs_log_iso_slope": sl,
                    "shot_noise_scales_with_gain": None if sl is None else bool(abs(sl + 1.0) < 0.15),
                    "row_noise_present": bool(max(r["row_noise_ratio"] for r in cr) > 1.5),
                    "read_noise_e_min_max": s.get("read_noise_e_min_max")}
    shot_scales = all(abs(s + 1.0) < 0.15 for s in slopes) if slopes else None
    text = (f"camera.py's variance law var = e + sigma_r^2 holds on {len(ok)} SIDD instances: R^2 "
            f">= {min(r2):.3f}, quadratic share <= {max(abs(q) for q in quad):.3f}, model vs measured bin variance "
            f"median error {np.median(err):.3f}; excess kurtosis {min(kurt):.2f}..{max(kurt):.2f} "
            f"(0 for Poisson+Gaussian); row-noise ratio {min(rowr):.2f}..{max(rowr):.2f}, column "
            f"{np.nanmin(colr):.2f}..{np.nanmax(colr):.2f} (1 = none); illumination drift over the burst up to "
            f"{max(drift):.0f} sigma of a frame mean, removed by per-frame scaling before the fit; {dropped} outlier "
            f"frames dropped; fit on the flat {100 * min(r.get('flat_fraction', 1.0) for r in ok):.0f}..{100 * max(r.get('flat_fraction', 1.0) for r in ok):.0f} % "
            f"of pixels, edge pixels carry {min(edges):.1f}..{max(edges):.1f}x the fitted variance (motion between frames)"
            if edges else "")
    if slopes:
        text += f"; log gain vs log ISO slope {', '.join(f'{s:.2f}' for s in slopes)} (shot noise ∝ gain predicts -1)"
    if per:
        text += "; per camera: " + "; ".join(
            f"{cam} linear {v['linear_variance']} ({len(v['instances_linear'])}/{v['n_instances']} instances, "
            f"quadratic share <= {v['quadratic_share_max']:.3f}), gain slope "
            f"{'n/a' if v['log_gain_vs_log_iso_slope'] is None else format(v['log_gain_vs_log_iso_slope'], '+.2f')}"
            for cam, v in per.items())
    return {"linear_variance": bool(linear), "shot_noise_scales_with_gain": shot_scales, "per_camera": per,
            "row_noise_present": bool(max(rowr) > 1.5), "column_noise_present": bool(np.nanmax(colr) > 1.5),
            "heavy_tails": bool(max(kurt) > 1.0), "text": text}


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    return x


def instance_jobs(raw_dir: Path | None) -> list[Path]:
    if raw_dir is None or not raw_dir.exists():
        return []
    return sorted(p for p in raw_dir.rglob("*.zip") if re.match(r"\d{4}_DNG\.zip$", p.name))


def run(raw_dir: Path, n_frames: int = DEFAULT_FRAMES, workers: int | None = None, log=None) -> dict:
    """All instance zips under raw_dir -> rows, ISO scaling per camera, verdict."""
    jobs = instance_jobs(raw_dir)
    meta_path = raw_dir / "metadata" / "train_metadata.csv"
    meta = sidd_metadata(meta_path) if meta_path.exists() else {}
    rows = []
    workers = workers or int(os.environ.get("HARNESS_WORKERS", "0") or 0) or min(4, max(1, (os.cpu_count() or 2) - 1))
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            futs = [ex.submit(analyse_instance, p, meta.get(re.match(r"(\d{4})", p.name).group(1)), n_frames) for p in jobs]
            for f in futs:
                r = f.result()
                rows.append(r)
                if log:
                    log(f"{r['zip']}: {'ok' if r['ok'] else 'FAILED ' + r.get('error', '')} ({r['seconds']} s)")
    else:
        for p in jobs:
            r = analyse_instance(p, meta.get(re.match(r"(\d{4})", p.name).group(1)), n_frames, log=log)
            rows.append(r)
            if log:
                log(f"{r['zip']}: {'ok' if r['ok'] else 'FAILED ' + r.get('error', '')} ({r['seconds']} s)")
    rows.sort(key=lambda r: (r.get("camera", ""), r.get("iso", 0), r["instance"]))
    scaling = iso_scaling(rows)
    return _clean({"method": METHOD, "n_frames_per_instance": n_frames, "instances": rows,
                   "iso_scaling": scaling, "verdict": verdict(rows, scaling)})
