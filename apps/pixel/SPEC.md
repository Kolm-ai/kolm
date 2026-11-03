# Verifiably better than any lens — process spec

Scope: still images and video. Everything here is a procedure a script can re-run on a new candidate and end with a number. Anything that cannot be re-run is out.

## 1. The claim

"Better than any lens" is undefined until it is a tuple. The harness refuses to run without one.

```json
{
  "qualities": { "<quality>": ["<metric>", ...] },
  "scene":     { "width", "height", "supersample", "seed", "n_frames", "fps", "exposure_s" },
  "camera":    { "preset" | explicit parameters, "exposure_s", "seed" },
  "display":   { "diagonal_in", "width_px", "height_px", "distance_m", "delivery" },
  "rendered_map": { "policy": "optional" | "required" },
  "gates":     [ { "name", "checks": [ { "metric", "op", "value" } ] }, ... ]
}
```

Result form: *candidate X exceeds capture constraints C on qualities Q for scene S at display D, by margins M.* Nothing broader is claimed.

`rendered_map` is the per-pixel provenance policy: a candidate may ship a map of the pixels it rendered rather than recovered (`run --rendered-map`). Without one it declares every pixel recovered, and all measured invention counts against it; under `required` a missing map leaves the provenance gate without data, so the verdict is INCOMPLETE.

## 2. Three regimes

| Regime | Provable? | How |
|---|---|---|
| Aesthetic preference | Yes | Blinded pairwise study, fitted model with intervals (§7) |
| Invented-world fidelity | Yes | Synthetic world with analytic ground truth; candidate vs camera-sim vs truth (§3–6) |
| Reality reconstruction | Not by generation alone | Information-bounded: ill-posed inverse, perception–distortion tradeoff. Needs more measurements, not more generation. Out of scope. |

## 3. Baseline camera model — the floor to beat

Simulated on the same supersampled render as the truth, so every difference is attributable to the capture model.

Pipeline: exposure-integrated motion → per-channel Airy PSF → residual aberration blur → pixel-aperture box → lens shading → photon shot noise → full-well clip → read noise → row noise → black pedestal → ADC clip → quantise → Bayer mosaic + bilinear demosaic (optional) → shading correction (the ISP's divide-back).

| Term | Model |
|---|---|
| Diffraction | Airy; first zero at r = 1.22 λ N on the sensor (N = f/D); angular θ = 1.22 λ / D |
| Photons, relative mode (default) | the scene's peak radiance (the wedge top, 2.0) lands at white_fill × FW, so the baseline is exposed for the scene it is handed; `white_radiance` pins another radiance there instead; shot noise ~ Poisson(e⁻) ⇒ SNR ∝ √e⁻ |
| Photons, absolute mode (`scene_white_cd_m2` > 0) | e⁻ = L · π · T / (4 N²) · k · t_exp · A_pix · QE, with L the luminance of radiance 1.0 in cd/m² (HDR+ median scene 634), T lens transmission 0.9, N = f/D, k = 4.09 × 10¹⁵ photons/(lx·s·m²) at 555 nm (a colour-filtered pixel differs by up to about 1.5×), QE 0.6; `exposure_for_peak` returns the t_exp that fills the well; saturation ISO = 78 lx·s / H_sat (ISO 12232), 99 for the Pixel at its ISO 100 clip of 4,662 e⁻, 1.55 µm, QE 0.6 |
| Read noise | Gaussian σ_r e⁻ added after the well clip; row noise Gaussian σ_row shared along each row (`row_noise_e`, from the measured row-noise ratio); a black pedestal of 8 σ_r keeps the shadow mean unbiased and the ADC clips at FW + pedestal |
| Lens shading | transmission 2^(−s·r²), r = 1 at the corners; s from the HDR+ shading maps (1.5 stops median, 1.78 for the Pixel family); the ISP divides it back out, so corner SNR falls by 2^(s/2) |
| Full well | photo-electrons clip at FW; engineering DR = log2(FW / σ_r); simulated DR = log2(FW / √(σ_r² + LSB² / 12)) with LSB = FW / (2^bits − 1), the range the linear single-gain model realises, which ranks the sourced frontier (`hamamatsu-orca-quest2`, 14.50 stops; a dual-gain HDR readout is not simulated) |
| Motion | render averaged over t_exp centred on frame time |
| Sampling | box over pixel pitch p; RGGB CFA optional |

Preset `best_case_earth`: f/1.4, λ = {610, 550, 465} nm, p = 3.76 µm, FW = 50 000 e⁻, σ_r = 1.5 e⁻, 14-bit, Bayer, residual blur 1.0 µm, white_fill 0.8: the scene's peak radiance fills 0.8 FW at t_exp (relative mode; set `scene_white_cd_m2` for absolute photometry). Change the preset, change the claim.

Residual blur floor from traced optics (DATA_PLAN.md §4 step 4): of 295 ray-traced prescriptions the sharpest full-frame lens leaves 0.27 µm on axis at F/1.5 (Strehl 0.83, MTF50 385 lp/mm); the median lens leaves 2.8 µm and the median edge MTF50 at the on-axis plane is 6 lp/mm. Preset `derived:best_case_traced` carries that floor on the best-DR sensor; it is design intent at the d line, lens only, so a claim against it is a claim against the sharpest lens that can be drawn, not the sharpest one built.

Noise law check (DATA_PLAN.md §4 step 5): on 4 Google Pixel raw bursts from SIDD (ISO 100 to 6400) the
per-pixel temporal variance follows var = e + σ_r² with R² ≥ 0.997 and gain ∝ ISO^-1.00; the residuals are Gaussian-shaped (excess kurtosis ≤ 0.10) once outlier frames, lamp drift and moving edges are taken out. Read noise is 1.5 DN at ISO 100 and 2.5 DN at ISO 400 (quantisation-limited), which is 7.2 e- at ISO 100, and 2.0 e- at ISO 6400: σ_r in the table above is a per-ISO number, and FW for a 10-bit phone raw is the ADC clip at base ISO (4,662 e-), not the pixel's well. Row-correlated noise (ratio 1.23 at ISO 100, 4.0 at ISO 6400) enters as `row_noise_e`, a lower bound of 0.055 e⁻ at ISO 100 from the ratio applied to the read noise. Preset `derived:measured_google_pixel` (alias `phone_measured`) carries the measured values. A second camera, iPhone 7 (ISO 100 to 1600, 5 bursts, preset `derived:measured_iphone_7`), fits the same law per ISO (R² ≥ 0.998) with read noise 1.7 to 2.0 e- and clip 1,385 e- at ISO 100, but its gain does not fall with ISO (slope −0.05), so its electron units are relative to the DNG scale: a floor to beat at that ISO, not a sensor characterisation.

Scene range from real environments (DATA_PLAN.md §4 step 7): on 997 Poly Haven HDRIs, solid-angle weighted, the sphere's p0.1 to p99.9 luminance range is 10.0 stops at the median (p10 7.1, p90 14.1); the brightest source sits 11.6 stops above the median pixel and the source-to-shade range is 16.7 stops (outdoor 18.7, indoor 16.2). A scene class S should quote its stop count from this table; the 14-step wedge spans 11.0 stops, `best_case_earth` holds 15.0 and the measured phone 9.3, and 424 of the 719 outdoor maps exceed 15 stops, so a sunlit exterior clips in every camera preset and the claim is decided at gate 3 unless S excludes the source.

Capture constraints from the field (DATA_PLAN.md §4 step 9): on the first raw frame of 153 HDR+ bursts the phone chose exposure median 1.7 ms (p10 0.4 ms, p90 30.3 ms; 9 % at 1/30 s or longer), ISO median 60 (p90 513), f/2.0 to f/2.4, in scenes of median 634 cd/m² (p10 5.7, p90 2904; inferred from the EXIF exposure through ISO 2720 with K = 12.5 assuming standard metering, and HDR+ deliberately underexposes to protect highlights, so the figure is biased high); the log-log slopes against luminance are -0.63 for time and -0.38 for ISO, so a darker scene is paid for mostly with exposure time. Its DNG noise profile sits within 10 % of the SIDD-measured gain, and the raw corners carry 1.5 stops less signal than the centre. A capture constraint C quoted for a phone should sit inside this distribution, and the motion-blur term uses the chosen exposure, not the frame rate.

## 4. Display ceiling

ppd = 1 / (2 · atan(p_d / 2d)) in degrees, with p_d = diagonal / √(W² + H²).

Visibility threshold ≈ 94 ppd mean, ≈ 120 ppd upper, for high-contrast foveal patterns. Master at ≥ 2× the display's pixel grid. Above threshold, resolution gains are invisible; spend on tone, colour, lighting, motion.

31.5″ 8K at 0.6 m → 115 ppd. Cameras already out-pixel screens (61–151 MP vs 8–33 MP). Pixel count is not the fight.

ITU design viewing distance (Rep. ITU-R BT.2246-9 Table 1, from Rec. ITU-R BT.1845): defined as one pixel per arc-minute, 60 ppd (3.18 H for 1080 lines, 1.59 H for 2160, 0.80 H for 4320); the table's rounded 3.1 / 1.5 / 0.75 H give 58.4 / 56.5 / 56.6 ppd at any panel size (presets `derived:hdtv-bt2246-design`, `derived:uhdtv-bt2246-design`, `derived:8k-bt2246-design`). A TV at its design distance sits below the 94 ppd threshold by definition, so a claim with D = television is decided at gates 1 to 4, never on resolution.

## 5. Metric suite — stills

Each metric maps (candidate, truth, scene regions) → scalar. Direction and tie margin are fixed in the registry (`harness/metrics/__init__.py`).

| Metric | Region | Method | Better |
|---|---|---|---|
| mtf50_cpp | slanted edge | ISO-12233-style SFR, 4× oversampled ESF | ↑ |
| texture_hf_fidelity | aperiodic texture | Re⟨C, T⟩ / ⟨T, T⟩ over f ≥ 0.25 c/px — true detail present, invented detail scores 0 | ↑ (≤ 1) |
| shadow_snr_db | wedge patches < 0.02 | mean/σ per patch, median | ↑ |
| highlight_loss_fraction | wedge patches > 1.0 | fraction of pixels < 0.95 · truth | ↓ |
| dr_steps | wedge | adjacent patches separated by > 2σ and ordered | ↑ |
| delta_e_mean / max | colour patches | CIEDE2000 vs known Lab | ↓ |
| edge_overshoot | slanted edge | ESF over/undershoot ÷ step | ↓ |
| banding_score | gradient | largest band step as Weber contrast: max positive second difference of the row means ÷ (L + 0.005 display black) | ↓ |
| periodicity | aperiodic texture | max normalised autocorrelation at lag ≥ 8 px | ↓ |
| landmark_error_px | stage | thresholded centroid vs known centre | ↓ |
| shading_rmse | disk interiors | RMSE vs Lambertian truth | ↓ |
| invented_detail_fraction | whole frame, 8 px border off, pixels under moving landmarks excluded | pixels carrying structured content no local linear filter of the truth explains and that is not noise: the truth is first blurred by the global Gaussian that best matches the candidate's high-pass content, then half-octave DoG bands (σ 0.7 to 2.8 px), one local gain per band fitted to the truth, residual mid-band energy against the candidate's own noise shape read on lit pixels where the truth is flat, Weber floor 0.02 on the high-pass residual, majority filter. Blur up to σ 3 px, linear sharpening and noise up to 5 % measure 0. Sensitivity on the 256×144 demo world with 32 px patches of band-limited texture: at 20 % Weber contrast recall 0.90 on colour patches, 1.00 on texture, 0.67 on the stage; at 10 % contrast 0.29, 0.91 and 0.37; false flags beyond 10 px of a patch under 0.7 %. The best-case camera measures 0 to 0.5 %, all at isoluminant colour edges where its demosaic and chromatic PSF create luminance structure the truth lacks; a JPEG q92 delivery of it measures 3 % (`harness/hallucination.py`) | ↓ |
| rendered_fraction | whole frame | pixels the candidate declares rendered in its map; 0 without a map | ↓ |
| undeclared_invention_fraction | whole frame | invented and not declared; equals invented_detail_fraction without a map | ↓ |

## 6. Metric suite — video

Adds to §5 (still metrics are averaged over frames).

| Metric | Method | Better |
|---|---|---|
| flicker | temporal σ of luminance over the never-moving mask ÷ mean | ↓ |
| flow_warped_excess | ‖warp_GTflow(cand[t+1]) − cand[t]‖ minus the same for truth; motion is integer-pixel per frame so the warp is exact | ↓ |
| motion_region_error | RMSE vs truth inside dilated moving masks | ↓ |
| trajectory_error_px | worst-frame landmark error | ↓ |

Ground-truth flow comes from the scene's motion model. The harness never estimates flow.

## 7. Human preference protocol

1. Items: candidate, camera-sim, truth, plus any others. Opaque ids; the key file is kept apart from the trial manifest.
2. Manifest: every pair, n trials each, side and order randomised by seed.
3. Viewing: one display, one distance, one ambient level, all stated in the claim. Two questions per trial — *more beautiful*, *more convincing* — recorded separately.
4. Fit: Bradley–Terry with a Rao–Kupper tie band (logistic link; a trial is a tie when the strength gap is inside θ) and one scale β per rater, by maximum likelihood with a weak ridge; bootstrap intervals. Without ties or rater ids it is Hunter's MM Bradley–Terry. Report P(candidate ≻ camera) with its interval. No interval, no claim.
5. The model was validated on five public vote sets (860,000 votes, DATA_PLAN.md §4 step 3): ties are 24–30 % of arena-style votes and need the band; rater scales move held-out likelihood by at most 0.012 nats per vote; bootstrap width matches between-fold spread (ratio 1.07–1.85). Held-out accuracy on generator-level votes is 0.51–0.69, so the gate reads the lower interval bound, never the point estimate.

## 8. Gates — default order and thresholds

Fail → stop, verdict `FAIL at <gate>`. A check without data (no human study, a metric the candidate lacks) is skipped and reported: when every gate that ran passed but a check was skipped the verdict is `INCOMPLETE (skipped: …)`, exit code 2, not a PASS. PASS needs every check of every gate evaluated and passed.

| # | Gate | Checks |
|---|---|---|
| 1 | composition | P(cand ≻ camera, convincing) lower bound > 0.5 — human; skipped without data |
| 2 | provenance | undeclared_invention_fraction ≤ 0.01: content the truth does not explain must be declared rendered, or there is at most 1 % of it. The best-case camera measures 0 to 0.5 % here, so a camera-faithful candidate passes; without a map every pixel is declared recovered; under `rendered_map.policy = required` a missing map skips the check |
| 3 | lighting_materials | shading_rmse ≤ 0.02 · landmark_error_px ≤ 0.5 |
| 4 | tone_color | delta_e_mean ≤ 2.0 · highlight_loss_fraction ≤ 0 · dr_steps ties or beats camera (the 14-step wedge spans 11 stops and sits inside a correctly exposed best-case camera's range, so the truth can only tie) |
| 5 | artifacts | edge_overshoot ≤ 0.02 (default and sourced; the KADID fit 0.22 is extrapolated, so it is reported beside the gate, not adopted, §10) · banding_score ≤ 0.01 (BT.2246 1/100; the sourced claim carries the KADID fit, §10) · periodicity ≤ 0.3 |
| 6 | resolution | display_ppd ≥ 94 · mtf50_cpp beats camera |
| 7 | micro_detail | texture_hf_fidelity beats camera |
| 8 | temporal (video) | flicker beats camera · flow_warped_excess ≤ 0.005 · motion_region_error beats camera |

Self-consistency (`python -m harness demo`): the truth as candidate passes every gate it has data for and ends `INCOMPLETE (skipped: composition)` (the still run also skips temporal): it beats the camera on resolution, shadows, geometry/lighting and, in video, temporal, and ties on highlights, colour and artifacts, since the correctly exposed baseline holds the whole wedge and there is nothing to beat. The camera-sim as candidate ties itself on every metric and fails at gate 6, resolution: `mtf50_cpp` must beat the camera and only ties it. A real candidate that only matches the best camera stops at the same place.

## 9. Report

`report.json` + `report.md`: the claim, per-metric {candidate, camera, truth, verdict}, per-quality verdict (worst of its metrics), the gate trail, delivery delta (metrics re-run after encoding), preference fit if present, and the per-pixel provenance block: measured invented, declared rendered and undeclared fractions for the frame and each scene region, the matched blur and noise shape per frame, with the map saved beside the report (`invented_detail.npy`, bool (T,H,W), and `invented_detail_0000.png`). Verdicts are *beats* / *ties* / *loses* with the margin stated.

## 10. Constant provenance

Every constant in §3, §4 and §8 resolves to a sourced row or is marked as an assumption:
sensors, lenses, displays and perceptual thresholds live in `harness/data/reference/*.csv`
(each row: `source_url` + `quote`), derived presets in `data/derived/` (`derived:<id>` in a
claim), the collection manifest in `harness/data/sources.json`, and the gap list in
`DATA_PLAN.md`. Two gates were put to human ratings rather than quoted; `fit_gates` (`harness/data/fit.py`)
regresses KADID-10k degradation category ratings (81 references × 5 levels, 30 raters each) on
the harness's own metric measured reference-to-distorted, then applies a stated support rule
(rank correlation ≤ −0.5, bootstrap interval ratio ≤ 3, estimator reads > 0 on ≥ 80 % of rated
images) before a fit may replace a sourced row. Results in `data/derived/gates_fitted.json`:

- `edge_overshoot` (sharpening ladder #24, `metrics/natural.edge_overshoot_pair`, calibrated
  against the slanted-edge metric): mean DCR 4.5 at overshoot 0.22, 95 % CI 0.19–0.26, rank
  correlation −0.80, extrapolated below the mildest rated level. Supported but extrapolated, so it is
  reported, not adopted: `claim_sourced.json` keeps 0.02 as the pass line and carries the fit with CI,
  condition and the note that says why.
- `banding_score` (quantization ladder #22, `metrics/natural.banding_pair`, Weber step contrast
  on smooth regions): rank correlation −0.31, interval spanning six decades, 18 % of rated images
  with no measurable band. Not supported: KADID's quantization posterises the whole image and the
  raters judged that, not banding on smooth gradients. The ratings only bound the threshold from
  above (level 1, median step 0.145, is already rated 4.17), so the sourced 0.01 (Rep. ITU-R
  BT.2246-9, 1/100) stays; TID2013 quantization noise (125 images) cross-checks with rank
  correlation −0.24 and says the same.
- Shift to generated content (`shift_check`, `harness/data/shift.py`; AIGCIQA2023, 2,400 images,
  three MOS dimensions): no rated set isolates halo or banding on generated images and none ships
  a reference, so both artifacts were read through no-reference proxies against the KADID
  references. The halo proxy does not calibrate on photographs (0.09 of the pair metric, rank
  correlation 0.37) and tracks sharpness on generated images (+0.34 with quality MOS), so the
  fitted gate's drift is unmeasured. Banded area is the one artifact the raters punish (−0.49),
  carried by one generator. `data/derived/shift_check.json`; DATA_PLAN.md §4 step 2.

## 11. Boundary

The harness measures candidates against a synthetic world it generated. It never infers truth from arbitrary footage. That is regime 3, and it stays out.
