# kolm pixel: the contract

Short version: a "better than any lens" claim is a tuple, not a sentence. Name
the regime, the qualities, the scene, the capture conditions and the display,
and this package measures it against a physical floor and a perceptual ceiling.
Without the tuple there is no claim and the harness refuses to run.

## 1. Claim

```json
{
  "claim_id": "...",
  "regime": "synthetic_world | capture_exceed | record_of_reality",
  "qualities": ["geometry", "tone_color", "artifacts", "resolution", "temporal"],
  "scene":     {"seed": 7, "width": 512, "height": 512, "frames": 1, "pan_px_per_frame": [1.37, -0.61]},
  "capture":   {"exposure_s": 0.01, "exposure_ev": 0.0, "motion_px_per_s": [120, 0], "camera": {...}},
  "display":   {"diagonal_in": 31.5, "width_px": 7680, "height_px": 4320, "distance_m": 0.6},
  "candidate": {"kind": "truth | camera | files", "path": "...", "pattern": "frame_{i:04d}.png"},
  "delivery":  {"format": "jpeg", "quality": 92},
  "preference": {"votes": "...", "key": "..."}
}
```

Regimes and what is provable:

| regime | meaning | provable here |
|---|---|---|
| `synthetic_world` | a world we specify completely | yes, every quality |
| `capture_exceed` | exceeding what the stated capture can record of that world | yes |
| `record_of_reality` | accuracy about a real event | no. The harness returns NOT PROVABLE and measures nothing |

A convincing image is not evidence of what happened. That regime needs
independent evidence, which no metric here can supply.

## 2. Floor: the best-case camera

`camera.BestCaseCamera` is a full-frame 61 MP class sensor (3.76 um pitch,
51 000 e- full well, 1.5 e- read noise, 14-bit) behind an aberration-free
f/2.8 lens. Its losses are all physical:

| stage | model |
|---|---|
| diffraction | Airy PSF per channel, 610 / 550 / 460 nm, supersampled onto pixels |
| motion | exposure-integrated blur along `motion_px_per_s * exposure_s` |
| photons | Poisson shot noise at `radiance * 2^ev * full_well` |
| saturation | hard clip at full well: one exposure, one well |
| read noise | Gaussian in electrons, then ADC quantisation |
| sampling | RGGB Bayer mosaic, bilinear demosaic |

Output is exposure-compensated back to scene units, then tonemapped and
sRGB-encoded with the same curve as the reference. The only difference between
camera and truth is what the sensor could record.

The **camera envelope** is the best value per metric over an exposure sweep
(-3..0 EV). A candidate "beats the floor" on a metric when it beats the
envelope, not one unlucky exposure. The single-exposure tradeoff is the whole
point: at 0 EV 75 % of the highlight wedge clips, at -3 EV shadow noise exceeds
one display code value. No exposure passes the tone gate.

## 3. Ceiling: the display

`display.Display` computes pixels per degree from diagonal, resolution and
distance. Measured acuity thresholds for high-contrast patterns are ~94 ppd
mean and ~120 ppd for the best observers (Ashraf et al. 2025). The reference
31.5" 8K at 0.6 m gives 115 ppd. Resolution beyond that buys nothing on that
screen; lighting, tone, colour and motion still do. Masters are rendered at
2x display size so crops and resampling do not cost resolution.

## 4. Ground truth

`scene.Scene(seed, width, height)` is an analytic HDR world: every element is a
function of continuous coordinates, band-limited (Gaussian-integrated edges,
rotated non-integer value-noise lattices, sinusoids below Nyquist), so a frame
at time `t` is exactly the frame at `t=0` translated by `t * pan`. That gives
exact flow for the temporal metrics and an unlimited supply of scenes at any
size. Nothing is stored.

| region | content | metric it serves |
|---|---|---|
| gradient | pure sky ramp | banding |
| highlight | log ramp 0.5..8 radiance, 25 % texture | highlight loss |
| patches | six flat ColorChecker-like colours | colour error |
| texture | full-band non-periodic texture | MTF50, tiling |
| edge | slanted step | halos |
| shadow | radiance 0.002, faint texture | shadow noise, crushed detail |
| landmarks | 3x3 Gaussian dots | geometry |

## 5. Metrics

| metric | better | what it measures |
|---|---|---|
| landmark_error_px | lower | centroid error of known dots |
| highlight_loss_fraction | lower | highlight pixels rendered flat where the truth has texture |
| shadow_noise_sigma | lower | error std in the shadow region, display units |
| shadow_detail_corr | higher | correlation with truth in the shadow region |
| banding_score | lower | lost distinct 8-bit levels in the smooth gradient |
| patch_delta_e | lower | CIEDE2000 of patch means |
| delta_e_p95 | lower | 95th percentile per-pixel CIEDE2000 |
| halo_score | lower | extra extrema next to edges (sharpening, ringing) |
| blocking_score | lower | spectral line at 8 px in the error gradient (codec grid) |
| tiling_score | lower | windowed 2-D autocorrelation peak (repeated texture) |
| mtf50 | higher | cross-spectrum MTF50 against truth, cycles/px |
| flicker_score | lower | std over time of mean-luminance error |
| temporal_noise | lower | per-pixel std over time of the error |
| motion_error_px | lower | estimated vs analytic frame-to-frame translation |

`metrics.generic` is the mask-free subset for arbitrary (degraded, clean)
pairs; it is what the labeller computes at scale.

## 6. Gates

Objective-first, halt on first failure, human preference last:

| order | gate | rule |
|---|---|---|
| 1 | claim | schema valid, regime provable |
| 2 | geometry | landmark_error_px <= 0.5 |
| 3 | tone_color | highlight_loss <= 0.02, shadow_noise <= 1/255, shadow_corr >= 0.8, banding <= 0.1, patch dE <= 2, dE p95 <= 4 |
| 4 | artifacts | halo <= 0.02, blocking <= 1.0, tiling <= 0.6 |
| 5 | resolution | mtf50 beats or ties the camera envelope |
| 6 | temporal (video) | flicker <= 1/255, temporal_noise <= 1.5/255, motion_error <= 0.25 px |
| 7 | delivery | gates 2..6 re-run on the exported file; a win that does not survive export is not a win |
| 8 | preference | blinded Bradley-Terry gap vs the camera with a bootstrap CI excluding zero |

Thresholds are visibility thresholds (one 8-bit code value, dE 2, half a
pixel), not tuned to any candidate. Production effort runs the other way:
subject and composition, then light and materials, then tone and colour, then
artifacts, then resolution, then micro-detail. Verification runs cheap and
objective first.

## 7. Preference study

`preference.make_manifest` emits blinded presentations (left/right, random
side, shuffled order) and a sealed key. Votes answer two questions separately:
more beautiful, more convincing. `bradley_terry` fits centred logit skills with
ties as half wins; `bootstrap` gives 95 % intervals on skills and on every
pairwise gap. Display conditions are fixed and stated. A preference without a
sealed key is not admissible.

## 8. Data pipeline

The only structural decision that matters: **store sources once, never store
derived data you can recompute.**

| stage | command | scales by |
|---|---|---|
| registry | `sources/registry.json` | adding entries, never items |
| collect | `python -m pixel collect` | threads; content-addressed, verified against publisher hashes, resumable, license-gated per item, byte budget |
| verify | `python -m pixel verify` | re-hash the store |
| shard | `python -m pixel shard` | tar shards in WebDataset layout |
| dedup | `python -m pixel dedup` | banded dHash, linear in corpus |
| label | `python -m pixel label` | processes x shards; one JSONL row per (source, seed) |
| budget | `python -m pixel plan-storage` | arithmetic |
| train | `python -m pixel train` | GPU; pairs regenerated from (source, seed) in the loader |

A training pair is `(source, seed)`: `degrade.recipe_from_seed(seed)` gives the
recipe, `degrade.apply` the degraded image, `degrade.direction` the label
vector. The labeller adds the generic metrics measured against the clean
source. On a 4 TB disk that is ~1.7 M LDR sources, ~2 000 4k HDRIs, ~55 M label
rows in ~16 GB, and zero stored degraded images.

## 9. Model

`train.PixelNet` is a small UNet conditioned on a direction vector with an
auxiliary head that predicts the direction of its input. Half the samples are
(degraded, -v) -> clean, half are (clean, +v) -> degraded, so one network moves
an image along every quality axis in both directions and doubles as a
no-reference quality estimator. `python -m pixel train --smoke` runs the whole
loop on procedural sources with no data.

## 10. Out of scope

Real-event accuracy, aesthetic judgement without a blinded study, perceptual
metrics that need a learned prior (LPIPS and friends), and any number produced
on an image whose claim tuple is missing.
