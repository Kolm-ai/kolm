"""Best-case physical capture, simulated on the same supersampled render as the truth.

Pipeline: (motion already integrated by the scene at this camera's exposure) -> per-channel Airy PSF
-> residual aberration blur -> pixel-aperture box -> lens shading -> photon shot noise -> full-well clip
-> read noise (+ row noise) -> black pedestal -> ADC clip + quantise -> pedestal subtracted -> optional RGGB
mosaic + bilinear demosaic -> shading correction. Output is in the truth's radiance units.

Exposure. Two modes, chosen by `scene_white_cd_m2`:
  relative (0, the default): the exposure decision is `white_fill`: the scene's peak radiance (or `white_radiance`
    when set) lands at white_fill x full well, i.e. the best-case camera exposes for the brightest thing in the
    scene; exposure_s then only sets the motion blur.
  absolute (> 0): radiance 1.0 means this luminance in cd/m2 and the electron count follows the physics,
    e- = L * pi*T/(4N^2) * k * t * A_pixel * QE, so exposure_s, f-number, pixel pitch and QE all matter and the
    camera clips when the scene is brighter than it exposed for.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from scipy.signal import fftconvolve
from scipy.special import j1

from .scene import Scene, box_downsample

AIRY_FIRST_ZERO = 3.8317  # first zero of J1
# photons per (lux * s * m^2): 1 lx = 1/683 W/m^2 at 555 nm, photon energy hc/lambda = 3.58e-19 J. For a colour-filtered
# pixel the broadband excess of white light (~2.8x the 555 nm count for an equal-energy spectrum) and the filter's
# ~1/3 band roughly cancel. Check: with QE 0.6 the Pixel's measured ISO 100 clip (4,662 e-, 1.55 um pixel) gives a
# saturation-based ISO (ISO 12232, 78 lx*s / H_sat) of 99; see CaptureConstraints.saturation_iso.
PHOTONS_PER_LUX_S_M2 = 4.09e15
ISO_SATURATION_LUX_S = 78.0


@dataclass(frozen=True)
class CaptureConstraints:
    name: str = "custom"
    aperture_mm: float = 50 / 1.4
    focal_mm: float = 50.0
    pixel_pitch_um: float = 3.76
    exposure_s: float = 1 / 125
    full_well_e: float = 50_000.0
    read_noise_e: float = 1.5
    bit_depth: int = 14
    bayer: bool = True
    wavelengths_nm: tuple[float, float, float] = (610.0, 550.0, 465.0)
    residual_blur_um: float = 1.0  # Gaussian sigma for aberrations a real lens never fully removes
    white_fill: float = 0.8  # fraction of full well reached by radiance 1.0 at exposure_s
    shading_corner_stops: float = 0.0  # lens shading: corner transmission below the centre, in stops (HDR+ maps ~1.5)
    shading_corrected: bool = True  # the ISP divides the shading back out: equal brightness, less signal in the corners
    white_radiance: float | None = None  # relative mode: radiance placed at white_fill x FW; None = the scene's peak
    scene_white_cd_m2: float = 0.0  # > 0: absolute mode, radiance 1.0 is this luminance (HDR+ median scene: ~634)
    quantum_efficiency: float = 0.6  # absolute mode only; effective (filter-weighted) QE
    lens_transmission: float = 0.9  # absolute mode only; T in pi*T/(4N^2)
    black_pedestal_e: float | None = None  # ADC offset so the read-noise floor is not clipped at 0; None = 8 x read noise
    row_noise_e: float = 0.0  # sigma of a per-row offset shared along the row (horizontal streaks), electrons

    @property
    def f_number(self) -> float:
        return self.focal_mm / self.aperture_mm

    @property
    def pixel_area_m2(self) -> float:
        return (self.pixel_pitch_um * 1e-6) ** 2

    def electrons_per_unit_radiance(self, white_radiance: float = 1.0) -> float:
        """Electrons collected by one pixel per unit of scene radiance at this exposure (see the module docstring)."""
        if self.scene_white_cd_m2 > 0:
            illuminance_lux = self.scene_white_cd_m2 * math.pi * self.lens_transmission / (4.0 * self.f_number ** 2)
            return (illuminance_lux * PHOTONS_PER_LUX_S_M2 * self.exposure_s * self.pixel_area_m2
                    * self.quantum_efficiency)
        return self.white_fill * self.full_well_e / max(white_radiance, 1e-12)

    def exposure_for_peak(self, peak_radiance: float) -> float:
        """Absolute mode: the exposure time that puts `peak_radiance` at white_fill x full well (the best-case
        exposure decision made explicit). Relative mode: the current exposure (it does not set the exposure there)."""
        if self.scene_white_cd_m2 <= 0:
            return self.exposure_s
        per_s = self.electrons_per_unit_radiance() / self.exposure_s
        return self.white_fill * self.full_well_e / (per_s * max(peak_radiance, 1e-12))

    @property
    def saturation_iso(self) -> float:
        """ISO 12232 saturation-based speed, 78 lx*s / H_sat, from full well, pixel area and QE."""
        h_sat = self.full_well_e / (PHOTONS_PER_LUX_S_M2 * self.pixel_area_m2 * self.quantum_efficiency)
        return ISO_SATURATION_LUX_S / h_sat

    @property
    def pedestal_e(self) -> float:
        return 8.0 * self.read_noise_e if self.black_pedestal_e is None else float(self.black_pedestal_e)

    def airy_first_zero_um(self, wavelength_nm: float) -> float:
        return 1.22 * wavelength_nm * 1e-3 * self.f_number

    def replace(self, **kw) -> "CaptureConstraints":
        return dataclasses.replace(self, **kw)


PRESETS: dict[str, CaptureConstraints] = {
    "best_case_earth": CaptureConstraints(name="best_case_earth"),
    "phone": CaptureConstraints(name="phone", aperture_mm=6.8 / 1.8, focal_mm=6.8, pixel_pitch_um=1.2,
                                full_well_e=6_000.0, read_noise_e=2.0, bit_depth=12, residual_blur_um=0.6),
    "diffraction_only": CaptureConstraints(name="diffraction_only", bayer=False, residual_blur_um=0.0,
                                           read_noise_e=0.0, full_well_e=1e9, bit_depth=16),
}


def derived_preset(name: str) -> CaptureConstraints:
    """`derived:<id>` presets come from data/derived/camera_presets.json (python -m harness data derive),
    i.e. from sourced sensor + lens rows rather than the hand-set numbers in PRESETS."""
    import json
    from .data import data_root
    path = data_root() / "derived" / "camera_presets.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m harness data derive --ids camera_presets`")
    presets = json.loads(path.read_text(encoding="utf-8"))["presets"]
    if name not in presets:
        raise KeyError(f"derived camera preset {name!r}; have {sorted(presets)}")
    fields = {f.name for f in dataclasses.fields(CaptureConstraints)}
    kw = {k: v for k, v in presets[name].items() if k in fields and k != "name"}
    if "wavelengths_nm" in kw:
        kw["wavelengths_nm"] = tuple(float(x) for x in kw["wavelengths_nm"])
    return CaptureConstraints(name=f"derived:{name}", **kw)


def from_dict(d: dict) -> tuple[CaptureConstraints, int]:
    """Claim `camera` block -> (constraints, noise seed). Preset first, explicit keys override.
    `preset` is a PRESETS key or `derived:<id>`."""
    d = dict(d)
    seed = int(d.pop("seed", 0))
    preset = d.pop("preset", None)
    if preset is None:
        base = CaptureConstraints()
    elif preset.startswith("derived:"):
        base = derived_preset(preset.split(":", 1)[1])
    else:
        base = PRESETS[preset]
    c = base.replace(**d)
    if c.exposure_s <= 0:
        raise ValueError("camera exposure_s must be > 0")
    return c, seed


def airy_kernel(first_zero_px: float, zeros: float = 3.0) -> np.ndarray:
    if first_zero_px < 0.15:
        return np.ones((1, 1))
    half = int(math.ceil(zeros * first_zero_px))
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    arg = AIRY_FIRST_ZERO * np.hypot(x, y) / first_zero_px
    k = np.ones_like(arg)
    nz = arg > 1e-9
    k[nz] = (2 * j1(arg[nz]) / arg[nz]) ** 2
    return k / k.sum()


def bayer_mask(H: int, W: int) -> np.ndarray:
    """RGGB CFA as (H, W, 3) booleans."""
    y, x = np.mgrid[0:H, 0:W]
    r = (y % 2 == 0) & (x % 2 == 0)
    b = (y % 2 == 1) & (x % 2 == 1)
    return np.stack([r, ~r & ~b, b], axis=-1)


_DEMOSAIC_K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64)


def demosaic_bilinear(plane: np.ndarray, cfa: np.ndarray) -> np.ndarray:
    out = np.empty(plane.shape + (3,), dtype=np.float64)
    for ch in range(3):
        m = cfa[..., ch].astype(np.float64)
        num = convolve(plane * m, _DEMOSAIC_K, mode="reflect")
        den = convolve(m, _DEMOSAIC_K, mode="reflect")
        out[..., ch] = num / den
    return out


def _sensor(electrons: np.ndarray, c: CaptureConstraints, rng: np.random.Generator) -> np.ndarray:
    """Photo-electrons -> what the ISP sees, in electrons. The well clips the photo-electrons; readout adds read
    noise and row noise; a black pedestal keeps the noise floor above the ADC's zero (a real sensor's black level,
    subtracted again at the end); the ADC clips at full well + pedestal and quantises to bit_depth."""
    e = np.minimum(rng.poisson(np.clip(electrons, 0, None)).astype(np.float64), c.full_well_e)
    if c.read_noise_e > 0:
        e += rng.normal(0.0, c.read_noise_e, size=e.shape)
    if c.row_noise_e > 0:
        e += rng.normal(0.0, c.row_noise_e, size=(e.shape[0],) + (1,) * (e.ndim - 1))
    ped = c.pedestal_e
    top = c.full_well_e + ped
    e = np.clip(e + ped, 0.0, top)
    levels = 2 ** c.bit_depth - 1
    return np.round(e / top * levels) / levels * top - ped


def shading_map(H: int, W: int, corner_stops: float) -> np.ndarray | None:
    """Relative lens transmission: 1 at the centre, 2**-corner_stops at the corners, quadratic in the normalised
    radius (the small-angle form of cos^4). On the HDR+ shading maps this shape explains ~86 % of the map's
    variation with residuals under 0.45 stops (DATA_PLAN.md section 4 step 11). None when there is no shading."""
    if corner_stops <= 0:
        return None
    y = (np.arange(H) - (H - 1) / 2) / max((H - 1) / 2, 1)
    x = (np.arange(W) - (W - 1) / 2) / max((W - 1) / 2, 1)
    r2 = (y[:, None] ** 2 + x[None, :] ** 2) / 2.0  # 1.0 at the four corners
    return 2.0 ** (-corner_stops * r2)


def simulate_frame(frame_ss: np.ndarray, c: CaptureConstraints, ss: int, rng: np.random.Generator,
                   white_radiance: float = 1.0) -> np.ndarray:
    """One frame. `white_radiance` is the exposure decision in relative mode (simulate_sequence passes the scene's
    peak); ignored in absolute mode, where exposure_s and the f-number decide."""
    pitch_ss = c.pixel_pitch_um / ss
    optical = np.empty(frame_ss.shape, dtype=np.float64)
    for ch, lam in enumerate(c.wavelengths_nm):
        plane = frame_ss[..., ch].astype(np.float64)
        k = airy_kernel(c.airy_first_zero_um(lam) / pitch_ss)
        if k.shape != (1, 1):
            plane = fftconvolve(plane, k, mode="same")
        sigma = c.residual_blur_um / pitch_ss
        if sigma > 0.05:
            plane = gaussian_filter(plane, sigma, mode="nearest")
        optical[..., ch] = plane
    radiance = np.clip(box_downsample(optical, ss), 0.0, None)
    shade = shading_map(radiance.shape[0], radiance.shape[1], c.shading_corner_stops)
    if shade is not None:
        radiance = radiance * shade[..., None]  # less light reaches the corners before the sensor sees it
    gain = c.electrons_per_unit_radiance(white_radiance)  # electrons per unit radiance
    if c.bayer:
        cfa = bayer_mask(*radiance.shape[:2])
        plane = _sensor((radiance * cfa).sum(axis=-1) * gain, c, rng)
        out = demosaic_bilinear(plane, cfa)
    else:
        out = _sensor(radiance * gain, c, rng)
    if shade is not None and c.shading_corrected:
        out = out / shade[..., None]  # the ISP's correction: brightness restored, corner noise multiplied with it
    return out / gain


def simulate_sequence(scene: Scene, c: CaptureConstraints, seed: int = 0) -> np.ndarray:
    """Capture every frame of the scene at this camera's exposure. Noise seeded per frame."""
    spec = scene.spec
    white = c.white_radiance if c.white_radiance is not None else scene.peak_radiance()
    frames = []
    for k in range(spec.n_frames):
        rng = np.random.default_rng([seed, k])
        ss_frame = scene.render_ss_integrated(spec.frame_time(k), c.exposure_s)
        frames.append(simulate_frame(ss_frame, c, spec.supersample, rng, white_radiance=white))
    return np.stack(frames)
