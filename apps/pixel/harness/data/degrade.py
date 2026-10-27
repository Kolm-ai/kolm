"""Synthetic degradation of clean tiles with every strength drawn from the measured tables in data/derived/.

A sample is (clean tile, parameter vector, seed) -> degraded tile. The vector is a pure function of
the seed and the measured distributions, so a label row re-renders bit for bit from
(source, file, crop, seed) and no image is ever stored. Any two vectors on one clean tile give a
training pair (state A to state B); the all-off vector `DegradeParams()` returns the tile unchanged.

Stages, in order, each skipped at its off value:
  prepare   scene range: an LDR tile is stretched in linear light until its p99.9/median matches a
            draw from the HDRI corpus (hdri_stats.json); exposure: the median luminance is placed at
            MID_GREY_FILL x clip (ISO 12232 metering, K = 12.5 as capture_stats uses) x 2^stops with
            stops drawn from the HDR+ BaselineExposure tags. 1.0 then means the sensor's clip level.
            The prepared tile is the clean target of the pair.
  optics    lateral chromatic shift: R and B translated by +-s at the tile's field position, s from
            TID2013 #23 (isp_stats.json); then harness.camera: Airy PSF at the preset's f-number and
            pitch, the residual aberration blur of a ray-traced lens at the drawn field
            (lens_traces.json, scaled to the sensor format), lens shading at that field position with
            the HDR+ corner falloff (capture_stats.json)
  sensor    harness.camera: shot noise, clip at the ISO-scaled well (noise_check.json: analog-gain
            sensors lose headroom, digital-gain sensors keep it), read noise at that ISO from the
            measured ladders, row noise, ADC quantisation, RGGB mosaic and bilinear demosaic
  isp       tone: contrast gain about mid-grey and mean shift in sRGB (TID2013 #17, #16); colour:
            saturation (TID2013 #18) and a cast equal to the ratio of two HDR+ bursts' as-shot
            neutrals; unsharp-mask halos at the KADID sharpen ladder's overshoot via the harness
            calibration (gates_fitted.json); posterisation to the bit depth that reproduces the KADID
            quantisation ladder's banding; JPEG at a quality fitted on TID2013 #10
Hand-set numbers are the mixture weights (which stages are on) and the uniform-in-area field draw;
every strength is an empirical draw from a measured list or an inversion of a measured ladder.
"""
from __future__ import annotations

import io
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from scipy.ndimage import shift as nd_shift

from . import data_root
from ..camera import CaptureConstraints, simulate_frame
from ..colorspace import linear_to_srgb, luminance, srgb_to_linear
from ..metrics.natural import quantize_srgb, unsharp

MID_GREY_FILL = 10.0 / 78.0   # ISO 12232: metered mid-grey at 10/ISO lx s, saturation at 78/ISO lx s
GAUSS_SIGMA_UM_PER_LPMM = math.sqrt(math.log(2.0) / (2.0 * math.pi ** 2)) * 1e3  # sigma_um = this / mtf50_lpmm
BASE_ISO = 100.0              # lowest ISO of the measured ladders; presets quote their clip level there
PEAK_PERCENTILE = 99.9        # the tile's "peak" for peak/median; one hot pixel is not the scene
MAX_STRETCH_GAMMA = 3.0       # cap on the luminance power law that stretches an LDR tile's range
FULL_FRAME_HALF_DIAG_MM = 21.6
P_CAMERA_ON = 0.8             # mixture weights: the only hand-set probabilities in the sampler
P_STAGE_ON = 0.5
PROFILES = ("iqa", "capture")  # sample_params profiles: every stage a coin at IQA-ladder levels, or a raw
                                  # capture: camera on, the ISP stages measured on real pipelines (isp_stats.json:
                                  # tone, saturation, cast, JPEG) as coins, no unsharp and no posterise: those two
                                  # take KADID ladder levels (median overshoot 1.33, gain 3.65) that no measured
                                  # ISP justifies, and a model trained on them reads every small bright patch as
                                  # a halo to remove (DEGRADATION_PLAN.md section 7)
P_EXPOSE_FOR_PEAK = 0.5       # the capture exposes for the highlights instead of for mid-grey (HDR+ style)
PEAK_FILL = (0.5, 1.0)        # where that policy places the scene peak, as a fraction of the clip level
EXPOSE_FOR_PEAK_STOPS = (-4.0, 1.0)  # the offset it may reach, relative to mid-grey at MID_GREY_FILL
UNSHARP_SIGMAS = (1.0, 1.5)   # the two kernel widths the overshoot calibration was measured at
POSTERISE_BITS = (3, 8)       # range the banding calibration is inverted over (measured at 5..8)
FORMAT_MATCH = 2.0            # a lens is drawn for a sensor whose half-diagonal is within this factor of its image height
MIN_FORMAT_LENSES = 5


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m harness data derive` first")
    return json.loads(path.read_text(encoding="utf-8"))


def _lens_fields(lens: dict) -> list[tuple[float, float]]:
    """(field fraction, residual aberration sigma in um) per traceable field of one ray-traced lens: the
    Gaussian that, combined with the lens's own diffraction MTF50, gives the traced MTF50 (geometric mean
    of tangential and sagittal). On axis this reproduces camera_presets' residual_blur_um."""
    diff = lens.get("diffraction_mtf50_lpmm")
    if not lens.get("ok") or not diff or diff <= 0:
        return []
    s_diff = GAUSS_SIGMA_UM_PER_LPMM / diff
    out = []
    for f in lens.get("fields", []):
        t, s = f.get("mtf50_t_lpmm"), f.get("mtf50_s_lpmm")
        if not t or not s or t <= 0 or s <= 0:
            continue
        s_tot = GAUSS_SIGMA_UM_PER_LPMM / math.sqrt(t * s)
        out.append((float(f["fraction"]), math.sqrt(max(s_tot * s_tot - s_diff * s_diff, 0.0))))
    return sorted(out)


def _ladder(gate: dict) -> tuple:
    return tuple((int(l["level"]), float(l["metric_median"]), float(l["metric_iqr"][0]), float(l["metric_iqr"][1]))
                 for l in gate["ladder"])


@dataclass(frozen=True)
class Distributions:
    """The measured lists the sampler draws from. Every array is the raw list of measured values; the
    sampler resamples them (empirical distribution), it never fits a parametric form to them."""
    presets: dict                      # camera_presets.json "presets" (20 sensors with lens rows)
    noise_ladders: dict                # camera -> (iso[], read_noise_e[], analog_gain) from noise_check
    iso_values: np.ndarray             # capture_stats frames: the ISO the phone chose
    exposure_offset_stops: np.ndarray  # capture_stats frames: DNG BaselineExposure, 0 when absent
    shading_stops: np.ndarray          # capture_stats frames: lens shading corner falloff in stops
    peak_over_median: np.ndarray       # hdri_stats maps: scene peak over median luminance
    lenses: tuple                      # (name, image_height_mm, ((fraction, residual_um), ...)) per traced lens
    sharpen_ladder: tuple              # (level, median, iqr_lo, iqr_hi) of edge_overshoot on KADID #24
    sharpen_slope: dict                # sigma_px -> overshoot per unit unsharp gain (harness calibration)
    banding_ladder: tuple              # (level, median, iqr_lo, iqr_hi) of banding_score on KADID #22
    banding_curve: tuple               # ((srgb_bits, banding metric), ...) (harness calibration)
    isp: dict                          # isp_stats.json: name -> measured values (jpeg, contrast, ...)

    @classmethod
    def load(cls, root: Path | None = None) -> "Distributions":
        root = Path(root) if root else data_root()
        d = root / "derived"
        presets = _read_json(d / "camera_presets.json")["presets"]
        ladders = {}
        for cam, v in _read_json(d / "noise_check.json")["iso_scaling"].items():
            iso, rn = np.asarray(v["iso"], float), np.asarray(v["read_noise_e"], float)
            order = np.argsort(iso)
            ladders[cam] = (iso[order], rn[order], bool(v.get("log_gain_vs_log_iso_slope", 0.0) < -0.5))
        frames = _read_json(d / "capture_stats.json").get("frames", [])
        iso = np.asarray([f["iso"] for f in frames if f.get("iso")], float)
        expo = np.asarray([float((f.get("sensor") or {}).get("baseline_exposure") or 0.0) for f in frames], float)
        shade = np.asarray([f["lens_shading"]["corner_falloff_stops"] for f in frames
                            if f.get("lens_shading") and f["lens_shading"].get("corner_falloff_stops") is not None],
                           float)
        maps = _read_json(d / "hdri_stats.json").get("maps", [])
        pom = np.asarray([m["peak_over_median"] for m in maps if m.get("peak_over_median")], float)
        traced = _read_json(d / "lens_traces.json")["lenses"]
        items = traced.items() if isinstance(traced, dict) else ((l["name"], l) for l in traced)
        lenses = []
        for name, lens in items:
            fl = _lens_fields(lens)
            ih = lens.get("image_height_mm")
            if fl and fl[0][0] == 0.0 and ih and ih > 0 and not (lens.get("focus") or {}).get("hit_scan_edge"):
                lenses.append((str(name), float(ih), tuple(fl)))
        gates = _read_json(d / "gates_fitted.json")["gates"]
        eo, bs = gates["edge_overshoot"], gates["banding_score"]
        slopes: dict[float, list[float]] = {}
        for c in eo["calibration"]["cases"]:
            slopes.setdefault(float(c["sigma_px"]), []).append(float(c["pair_whole_frame"]) / float(c["gain"]))
        curve = tuple(sorted((int(c["srgb_bits"]), float(c["pair_whole_frame"])) for c in bs["calibration"]["cases"]))
        isp = {}
        isp_path = d / "isp_stats.json"
        if isp_path.exists():
            for name, p in json.loads(isp_path.read_text(encoding="utf-8"))["parameters"].items():
                isp[name] = np.asarray(p["values"], float)
        if not lenses or iso.size == 0 or pom.size == 0 or shade.size == 0 or not presets:
            raise ValueError("derived tables present but empty; re-run `python -m harness data derive`")
        return cls(presets, ladders, iso, expo, shade, pom, tuple(lenses), _ladder(eo),
                   {s: float(np.mean(v)) for s, v in slopes.items()}, _ladder(bs), curve, isp)

    def ladder_for(self, preset_name: str, analog: bool):
        """The measured read-noise ladder of this preset (measured phones) or of the measured camera
        with the same gain behaviour (every other preset, scaled to its own base read noise)."""
        key = preset_name.lower()
        for cam, lad in self.noise_ladders.items():
            if cam.lower().replace(" ", "_") in key:
                return lad, True
        for cam, lad in self.noise_ladders.items():
            if lad[2] == analog:
                return lad, False
        return None, False

    def read_noise_at(self, preset_name: str, preset: dict, iso: float, analog: bool) -> float:
        base = float(preset.get("read_noise_e", 1.0))
        lad, own = self.ladder_for(preset_name, analog)
        if lad is None or lad[0].size < 2:
            return base
        k, v, _ = lad
        at = float(np.interp(math.log(min(max(iso, k[0]), k[-1])), np.log(k), v))
        return at if own else base * at / float(v[0])

    def lenses_for_format(self, half_diag_mm: float) -> tuple:
        """Traced lenses designed for roughly this sensor format (image height within a factor of
        FORMAT_MATCH of the half-diagonal): a lens is evaluated at its own designed field, its residual then
        scaled by the exact ratio. Falls back to every lens when fewer than MIN_FORMAT_LENSES match."""
        pool = tuple(l for l in self.lenses if 1.0 / FORMAT_MATCH <= half_diag_mm / l[1] <= FORMAT_MATCH)
        return pool if len(pool) >= MIN_FORMAT_LENSES else self.lenses

    def isp_values(self, name: str) -> np.ndarray:
        if name not in self.isp or self.isp[name].size == 0:
            raise KeyError(f"isp_stats.json has no measured '{name}' values; run harness.data.isp_stats.derive_isp_stats")
        return self.isp[name]


@dataclass(frozen=True)
class DegradeParams:
    """One point in degradation space. The defaults are the identity: degrade(tile, DegradeParams()) is the tile.
    Field groups: prepare (expose, exposure_stops, scene_peak_over_median), camera (camera_on .. ca_shift_px),
    isp (contrast_gain .. jpeg_quality). Flags are ints so a row serialises to plain columns."""
    expose: int = 0                      # 1: place the median luminance at MID_GREY_FILL x 2**exposure_stops
    exposure_stops: float = 0.0          # HDR+ BaselineExposure of a drawn frame, or the expose-for-the-peak offset
    scene_peak_over_median: float = 0.0  # LDR tiles are stretched to this; 0 = keep the tile's range
    camera_on: int = 0
    preset: str = ""                     # camera_presets.json key
    pixel_pitch_um: float = 0.0
    half_diag_mm: float = 0.0            # sensor format: sets the tile's angular extent for shading and CA
    focal_mm: float = 0.0
    f_number: float = 0.0
    iso: float = 0.0                     # from the HDR+ frames; scales the well or the digital gain
    analog_gain: int = 1                 # noise_check: 1 = clip level drops with ISO, 0 = well kept, gain digital
    full_well_e: float = 0.0             # electrons at the clip level at this ISO
    white_fill: float = 1.0              # fraction of the well that radiance 1.0 fills
    read_noise_e: float = 0.0            # at this ISO, from the measured ladder
    row_noise_e: float = 0.0
    bit_depth: int = 0
    bayer: int = 1
    lens: str = ""                       # lens_traces.json key
    field_fraction: float = 0.0          # where in the frame the tile sits, 0 centre .. 1 corner
    residual_blur_um: float = 0.0        # aberration blur of that lens at that field, scaled to this format
    shading_corner_stops: float = 0.0
    shading_corrected: int = 1
    ca_shift_px: float = 0.0             # lateral chromatic shift of R (+) and B (-) at the tile, px
    contrast_gain: float = 1.0           # sRGB contrast about 0.5 (TID2013 #17)
    mean_shift: float = 0.0              # sRGB offset (TID2013 #16)
    saturation: float = 1.0              # chroma ratio (TID2013 #18)
    cast_r: float = 1.0                  # ratio of two bursts' as-shot neutrals, R/G
    cast_b: float = 1.0                  # and B/G
    unsharp_gain: float = 0.0
    unsharp_sigma_px: float = 0.0
    posterise_bits: int = 0              # 0 = off
    jpeg_quality: int = 0                # 0 = off

    def is_identity(self) -> bool:
        return self == DegradeParams()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DegradeParams":
        names = {f.name: f.type for f in fields(cls)}
        kw = {}
        for k, v in d.items():
            if k in names and v is not None:
                kw[k] = int(v) if names[k] == "int" else (str(v) if names[k] == "str" else float(v))
        return cls(**kw)


PARAM_NAMES = tuple(f.name for f in fields(DegradeParams))


def _draw(rng: np.random.Generator, values: np.ndarray) -> float:
    return float(values[int(rng.integers(values.size))])


def preset_half_diag_mm(preset: dict) -> float:
    res = (preset.get("sensor") or {}).get("resolution_px")
    if res and len(res) == 2 and preset.get("pixel_pitch_um"):
        return 0.5 * float(preset["pixel_pitch_um"]) * 1e-3 * math.hypot(float(res[0]), float(res[1]))
    return FULL_FRAME_HALF_DIAG_MM


def draw_ladder_target(rng: np.random.Generator, ladder: tuple) -> tuple[float, int]:
    """A metric value a rated distortion ladder produced: level uniform, then log-uniform inside that level's
    interquartile range. Returns (target, level)."""
    level, _, lo, hi = ladder[int(rng.integers(len(ladder)))]
    lo, hi = max(lo, 1e-6), max(hi, 1e-6)
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi)))), int(level)


def bits_for_banding(target: float, curve: tuple) -> int:
    """sRGB bit depth whose posterisation gives `target` on banding_pair, from the log-linear fit of the
    harness calibration (bits 5..8), extrapolated and clamped to POSTERISE_BITS."""
    bits = np.asarray([b for b, _ in curve], float)
    logm = np.log([max(v, 1e-9) for _, v in curve])
    a, b = np.polyfit(bits, logm, 1)
    est = (math.log(max(target, 1e-9)) - b) / a
    return int(round(min(max(est, POSTERISE_BITS[0]), POSTERISE_BITS[1])))


def expose_for_peak_stops(peak_over_median: float, peak_fill: float) -> float:
    """Exposure offset (stops from mid-grey at MID_GREY_FILL) that places the scene peak at peak_fill of the clip
    level: a legitimately dark capture whose target is equally dark, so darkness alone is not a defect to undo.
    Clamped to EXPOSE_FOR_PEAK_STOPS."""
    stops = -math.log2(max(peak_over_median, 1e-9) * MID_GREY_FILL / peak_fill)
    return float(min(max(stops, EXPOSE_FOR_PEAK_STOPS[0]), EXPOSE_FOR_PEAK_STOPS[1]))


def sample_params(rng: np.random.Generator, dists: Distributions, profile: str = "iqa") -> DegradeParams:
    """One parameter vector. Every strength is a draw from a measured list; the coins are P_CAMERA_ON and
    P_STAGE_ON. The draw order is fixed, so a seed maps to one vector for a given data/derived/ and profile
    (the default "iqa" profile is unchanged; "capture" is the raw-pipeline profile of PROFILES)."""
    assert profile in PROFILES, profile
    capture = profile == "capture"
    kw: dict = dict(expose=1, exposure_stops=_draw(rng, dists.exposure_offset_stops),
                    scene_peak_over_median=_draw(rng, dists.peak_over_median))
    if rng.random() < P_EXPOSE_FOR_PEAK:
        kw["exposure_stops"] = expose_for_peak_stops(kw["scene_peak_over_median"], float(rng.uniform(*PEAK_FILL)))
    if capture or rng.random() < P_CAMERA_ON:
        names = sorted(dists.presets)
        name = names[int(rng.integers(len(names)))]
        p = dists.presets[name]
        analog = bool(p.get("iso_scaling_is_analog_gain", True))
        iso = _draw(rng, dists.iso_values)
        gain = max(iso, BASE_ISO) / BASE_ISO
        fw = float(p["full_well_e"])
        hd = preset_half_diag_mm(p)
        pool = dists.lenses_for_format(hd)
        lens_name, ih, fl = pool[int(rng.integers(len(pool)))]
        r = math.sqrt(rng.random())                       # uniform over the sensor's area
        resid = float(np.interp(r, [f for f, _ in fl], [s for _, s in fl])) * hd / ih
        ca_corner = _draw(rng, dists.isp_values("chromatic_aberration"))
        ca_on = rng.random() < P_STAGE_ON
        kw.update(camera_on=1, preset=name, pixel_pitch_um=float(p["pixel_pitch_um"]), half_diag_mm=hd,
                  focal_mm=float(p["focal_mm"]), f_number=float(p["focal_mm"]) / float(p["aperture_mm"]),
                  iso=iso, analog_gain=int(analog), full_well_e=fw / gain if analog else fw,
                  white_fill=1.0 if analog else 1.0 / gain,
                  read_noise_e=dists.read_noise_at(name, p, iso, analog),
                  row_noise_e=float(p.get("row_noise_e", 0.0)), bit_depth=int(p.get("bit_depth", 12)),
                  bayer=int(bool(p.get("bayer", True))), lens=lens_name, field_fraction=r,
                  residual_blur_um=resid, shading_corner_stops=_draw(rng, dists.shading_stops),
                  shading_corrected=int(rng.random() < P_STAGE_ON), ca_shift_px=ca_corner * r if ca_on else 0.0)
    if rng.random() < P_STAGE_ON:
        kw.update(contrast_gain=_draw(rng, dists.isp_values("contrast")), mean_shift=_draw(rng, dists.isp_values("mean_shift")))
    if rng.random() < P_STAGE_ON:
        kw.update(saturation=_draw(rng, dists.isp_values("saturation")))
    if rng.random() < P_STAGE_ON:
        wr, wb = dists.isp_values("wb_r_over_g"), dists.isp_values("wb_b_over_g")
        i, j = (int(x) for x in rng.integers(wr.size, size=2))
        kw.update(cast_r=float(wr[i] / wr[j]), cast_b=float(wb[i] / wb[j]))
    if not capture and rng.random() < P_STAGE_ON:
        target, _ = draw_ladder_target(rng, dists.sharpen_ladder)
        sigma = UNSHARP_SIGMAS[int(rng.integers(len(UNSHARP_SIGMAS)))]
        kw.update(unsharp_gain=target / dists.sharpen_slope[sigma], unsharp_sigma_px=sigma)
    if not capture and rng.random() < P_STAGE_ON:
        target, _ = draw_ladder_target(rng, dists.banding_ladder)
        kw.update(posterise_bits=bits_for_banding(target, dists.banding_curve))
    if rng.random() < P_STAGE_ON:
        kw.update(jpeg_quality=int(_draw(rng, dists.isp_values("jpeg"))))
    return DegradeParams(**kw)


# ---------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------
def prepare_tile(tile: np.ndarray, params: DegradeParams, is_hdr: bool = False) -> np.ndarray:
    """The clean target of a sample: the linear tile placed in the sensor's units (1.0 = clip level) with the
    drawn scene range. LDR tiles get a luminance power law (chromaticity kept) until p99.9/median reaches
    the drawn HDRI value; HDR tiles keep their own range. Identity params return the tile unchanged."""
    t = np.asarray(tile, dtype=np.float64)
    if params.scene_peak_over_median > 0 and not is_hdr:
        lum = luminance(t)
        med, peak = float(np.median(lum)), float(np.percentile(lum, PEAK_PERCENTILE))
        if med > 0 and peak > med * 1.001:
            gamma = min(max(math.log(params.scene_peak_over_median) / math.log(peak / med), 1.0), MAX_STRETCH_GAMMA)
            if gamma > 1.0:
                t = t * ((np.maximum(lum, 0.0) / med) ** (gamma - 1.0))[..., None]
    if params.expose:
        med = float(np.median(luminance(t)))
        if med > 0:
            t = t * (MID_GREY_FILL * 2.0 ** params.exposure_stops / med)
    return t


def _upsample(t: np.ndarray, ss: int) -> np.ndarray:
    return np.repeat(np.repeat(t, ss, axis=0), ss, axis=1) if ss > 1 else t


def _constraints(p: DegradeParams) -> CaptureConstraints:
    return CaptureConstraints(name=p.preset, aperture_mm=p.focal_mm / p.f_number, focal_mm=p.focal_mm,
                              pixel_pitch_um=p.pixel_pitch_um, full_well_e=p.full_well_e,
                              read_noise_e=p.read_noise_e, bit_depth=p.bit_depth, bayer=bool(p.bayer),
                              residual_blur_um=p.residual_blur_um, white_fill=p.white_fill,
                              shading_corner_stops=0.0, white_radiance=1.0, row_noise_e=p.row_noise_e)


def tile_shading(h: int, w: int, p: DegradeParams, ss: int = 1) -> np.ndarray | None:
    """Lens transmission across a tile centred at field fraction `field_fraction` along +x: the camera's
    quadratic-in-radius map (2**-stops at the frame corner) evaluated at the tile's pixels, whose extent in
    field units follows from the pitch and the sensor's half-diagonal."""
    if p.shading_corner_stops <= 0 or p.half_diag_mm <= 0:
        return None
    px_mm = p.pixel_pitch_um * 1e-3 / ss
    x = p.field_fraction * p.half_diag_mm + (np.arange(w) - (w - 1) / 2) * px_mm
    y = (np.arange(h) - (h - 1) / 2) * px_mm
    rho2 = (x[None, :] ** 2 + y[:, None] ** 2) / p.half_diag_mm ** 2
    return 2.0 ** (-p.shading_corner_stops * rho2)


def chromatic_shift(f: np.ndarray, shift_px: float) -> np.ndarray:
    """Lateral chromatic aberration as TID2013 #23 renders it: R translated by +shift, B by -shift (cubic)."""
    out = np.array(f, dtype=np.float64, copy=True)
    out[..., 0] = nd_shift(f[..., 0], (0.0, shift_px), order=3, mode="nearest")
    out[..., 2] = nd_shift(f[..., 2], (0.0, -shift_px), order=3, mode="nearest")
    return out


def _camera(t: np.ndarray, p: DegradeParams, rng: np.random.Generator, ss: int) -> np.ndarray:
    from ..scene import box_downsample
    f = _upsample(t, ss)
    if p.ca_shift_px:
        f = chromatic_shift(f, p.ca_shift_px * ss)
    shade = tile_shading(f.shape[0], f.shape[1], p, ss)
    if shade is not None:
        f = f * shade[..., None]
    out = simulate_frame(f, _constraints(p), ss, rng, white_radiance=1.0)
    if shade is not None and p.shading_corrected:
        out = out / box_downsample(shade, ss)[..., None]
    return np.clip(out, 0.0, 1.0)


def jpeg_roundtrip(linear: np.ndarray, quality: int) -> np.ndarray:
    """sRGB 8-bit JPEG at `quality` (4:4:4 at >= 90, else 4:2:0: the setting isp_stats fitted with)."""
    from PIL import Image
    s8 = np.clip(np.round(linear_to_srgb(np.clip(linear, 0.0, 1.0)) * 255.0), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(s8, "RGB").save(buf, format="JPEG", quality=int(quality), subsampling=0 if quality >= 90 else 2)
    buf.seek(0)
    return srgb_to_linear(np.asarray(Image.open(buf).convert("RGB"), dtype=np.float64) / 255.0)


def _isp(t: np.ndarray, p: DegradeParams) -> np.ndarray:
    f = t
    if p.contrast_gain != 1.0 or p.mean_shift != 0.0:
        s = linear_to_srgb(np.clip(f, 0.0, 1.0))
        f = srgb_to_linear(np.clip(0.5 + p.contrast_gain * (s - 0.5) + p.mean_shift, 0.0, 1.0))
    if p.saturation != 1.0:
        y = luminance(f)[..., None]
        f = np.clip(y + p.saturation * (f - y), 0.0, None)
    if p.cast_r != 1.0 or p.cast_b != 1.0:
        f = f * np.array([p.cast_r, 1.0, p.cast_b])
    if p.unsharp_gain:
        f = np.clip(unsharp(f, p.unsharp_gain, p.unsharp_sigma_px), 0.0, None)
    if p.posterise_bits:
        f = quantize_srgb(np.clip(f, 0.0, 1.0), p.posterise_bits)
    if p.jpeg_quality:
        f = jpeg_roundtrip(f, p.jpeg_quality)
    return f


def degrade(tile: np.ndarray, params: DegradeParams, seed: int = 0, ss: int = 1, is_hdr: bool = False,
            scene_gain: np.ndarray | None = None) -> np.ndarray:
    """(clean linear tile, params, seed) -> degraded linear tile, same shape. Deterministic in (params, seed).
    `ss` supersamples the optics (1 = tile resolution, the labeller's default for throughput). `scene_gain` (h, w)
    multiplies the prepared tile before the camera and the ISP: a lighting map that belongs to the scene. The
    model's pairs pass the input's uncorrected lens shading here so the target carries it where the input had
    it, ahead of the tone affine, which does not commute with a multiply."""
    t = prepare_tile(tile, params, is_hdr)
    if scene_gain is not None:
        t = t * np.asarray(scene_gain, dtype=np.float64)[..., None]
    if params.camera_on:
        t = _camera(t, params, np.random.default_rng(int(seed)), ss)
    return _isp(t, params)


def make_pair(tile: np.ndarray, a: DegradeParams, b: DegradeParams, seed_a: int, seed_b: int,
              ss: int = 1, is_hdr: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Two states of one clean tile: a training pair in either direction. With b = DegradeParams() the
    target is the clean tile itself (prepared the same way, so the pair shares its exposure placement)."""
    return degrade(tile, a, seed_a, ss, is_hdr), degrade(tile, b, seed_b, ss, is_hdr)
