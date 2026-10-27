# Degradation plan: synthetic training data with harness metrics as offline labels

Status 2026-10-27. Code: [harness/data/degrade.py](harness/data/degrade.py) (parameter sampler, camera and ISP
model), [harness/data/tilemetrics.py](harness/data/tilemetrics.py) (the twelve still metrics on a natural tile
against its clean tile), [harness/data/label.py](harness/data/label.py) and the CLI
`python -m harness data label --sources <ids> --n <count> --out data/labels`. Tests:
[tests/test_degrade.py](tests/test_degrade.py). Sample shard: `data/labels/` (2048 rows, 18 shards, ledger).

## 1. What a sample is

One clean image on disk is cropped at `--crops` random positions (fixed by a hash of the item's identity) into
tiles of `--tile` px. Per tile, `--per-tile` parameter vectors are drawn from the measured distributions, each
rendered by the camera model plus ISP stages, and scored against the prepared clean tile. One row per draw:
`source, file, row, y0, x0, size, draw, seed, is_hdr, domain, scale, ss`, the parameter vector `p_*` (31 fields) and
the metric vector `m_*` (12 fields). No image is stored: the row reproduces the pair bit for bit, so storage is
a few hundred bytes per sample whatever the tile size.

Every input reaches the sampler as linear RGB, whatever the file held, so the metrics of a JPEG row and a raw
row read the same way; `domain` records the input class and `is_hdr` how the clean tile's range is prepared
(the table is the docstring of `harness/data/label.py`): `srgb8` (8-bit sRGB files and parquet blobs, sRGB
transfer inverted), `srgb16` (16-bit integer PNG/TIFF, same), `linear16` (camera raws through LibRaw: camera
white balance, sRGB primaries, linear output, no auto-brightening or sharpening, 1.0 = sensor white level;
`is_hdr`), `linear_float_hdr` (Radiance `.hdr` and OpenEXR `.exr`, plain or inside a zip: relative radiance as
stored; `is_hdr`). LDR domains get the drawn peak-over-median stretch, HDR domains keep their range, and every
draw re-exposes the tile median (`expose=1`), so a domain's absolute scale never reaches the metrics. A zip
member is an item of its own (`file` is `<zip>::<member>`), and a file no decoder can read is recorded in the
ledger entry of its chunk as `skipped` with a reason (`no_decoder:<package>`, `decode_error:<exception>`,
`too_small:HxW`); permanent reasons count as done until `--retry-skipped`, a missing decoder is retried.

Any two rows of the same tile are a training pair, state A to state B, with both metric vectors known. The
degraded-to-clean pair is the special case where B is the identity parameter vector, which
`degrade(tile, DegradeParams())` reproduces exactly (tested). Edit directions therefore include "less noise",
"less halo", "more contrast", "different white balance", not only "restore".

## 2. Where every parameter comes from

Nothing in the sampler is a hand-set range. Each draw is a value from a measured set in `data/derived/`:

| parameter | drawn from | file |
|---|---|---|
| camera preset (pixel pitch, full well, read noise, bit depth, bayer, aperture, focal) | one of 20 presets | camera_presets.json |
| ISO and read noise | 137 measured ISO values; read noise from the per-instance ladders, analog or digital gain by the measured log-gain slope | noise_check.json |
| exposure offset, lens shading corner stops | 153 HDR+ bursts | capture_stats.json |
| scene peak over median | 997 HDRIs | hdri_stats.json |
| lens, field fraction, residual blur | 281 traced lenses, 1011 field points, format-matched to the preset within a factor 2 of its half diagonal | lens_traces.json |
| unsharp gain | the fitted edge-overshoot ladder and its calibration slope | gates_fitted.json |
| posterisation bits | the fitted banding ladder through the bits-to-banding curve, clamped to 3..8 | gates_fitted.json |
| JPEG quality, mean shift, contrast, saturation, chromatic shift | 125 TID2013 reference-distortion values each | isp_stats.json |
| colour cast | ratio of two white-balance draws over the 153 HDR+ bursts | isp_stats.json |

The only hand-set numbers are the mixture coins, `P_CAMERA_ON = 0.8` (the camera stage is on),
`P_STAGE_ON = 0.5` (each ISP stage is on) and `P_EXPOSE_FOR_PEAK = 0.5` (the capture exposes for the
highlights), the peak fill that policy aims at, `PEAK_FILL = (0.5, 1.0)` of the clip level, clamped to
`EXPOSE_FOR_PEAK_STOPS = (-4, +1)` stops from mid-grey, the LDR stretch cap `MAX_STRETCH_GAMMA = 3`, and
the tile geometry.

The sampler has two profiles (`PROFILES` in `harness/data/degrade.py`, added 2026-10-28): the default `iqa`
profile above, and a `capture` profile in which the camera stage is always on and only the measured ISP stages
(tone affine, saturation, colour cast, JPEG) are coins, with no unsharp and no posterise, because
isp_stats.json holds no measured real-ISP sharpening and the KADID-ladder unsharp levels taught the model a
halo (section 7). The model's restore pairs draw the capture profile with `P_RESTORE_CAPTURE = 0.7`
(`harness/model/data.py`); the labels and the edit pairs keep the default profile, so existing label shards
and seeds map to the same vectors.

The expose-for-the-peak policy (added 2026-10-27) exists because the measured exposure offsets are all near
zero (q05/50/95 = 0/0/0.01 stops over the 153 HDR+ frames: BaselineExposure says nothing about the
under-exposure HDR+ applies before it), so the first model run saw every dark input as a mean-shift defect
and brightened the demo's best-case camera, which exposes for the highlights at a median 1.8 stops below
anything in the stream, by 2.2x. With the policy on, `exposure_stops = -log2(peak_over_median x
MID_GREY_FILL / fill)`, so the scene peak lands at `fill` of the clip level and the restore target (the same
prepare) is equally dark. The fill range is hand-set; HDR+ (Hasinoff et al. 2016) deliberately under-exposes
to protect highlights but publishes no fill distribution, so deriving one from the bursts' clip fractions is
an open item in section 7. Adding the coin changed the seed-to-vector mapping; rows of the sample shard
carry their own parameters and stay valid.
The sample shard reads camera on 0.79, JPEG on 0.50, posterise on 0.51.

Two properties of the measured sets that a consumer must know:

- The lens traces are flat-plane prescriptions, pessimistic off axis. Residual blur per field point reads
  p10/50/90/99 = 0.9/7.1/41/129 um, which at the presets' pitches is p50/90/99 = 1.3/10.5/46 px. A 46 px
  blur on a 256 px tile is close to featureless. Each row carries `p_lens`, `p_field_fraction` and
  `p_residual_blur_um`, so a consumer can filter (for example field fraction <= 0.5, or blur <= 4 px) without
  relabelling. Whether to cap the sampler instead is an open decision, see section 7.
- The ISP values are TID2013 distortion levels, so they cover the rated-visible range of each artefact, not
  the range a shipping ISP produces. That is intended: the labels must span what the metrics gate.

## 3. Clean corpus budget on D:

`HARNESS_DATA_ROOT` moves the whole data root. Plan: `HARNESS_DATA_ROOT=D:\lens-harness-data`, move the
present `data\raw`, `data\derived` and `data\ledger.jsonl` there (32 GB, one robocopy /MOVE), and fetch new
sources with `python -m harness data collect` into that root. D: has 1908 GB free.

| source | on disk | planned | role |
|---|---|---|---|
| polyhaven-hdris at 1k | 1.5 GB, 997 files | 2k: about 6 GB | HDR scenes, the over-range prior (peak over median) |
| ambientcg-hdris | 9 MB index only | 2k: 100 HDRIs, about 0.6 GB | HDR scenes, second licence and capture pipeline |
| hdr-plus-burst (153 bursts) | 3.2 GB | keep | real raw bursts: exposure, shading and white-balance statistics, and clean raw frames |
| hf-kadid10k-images | 2.9 GB, 81 refs | keep | the gate-fit references; also clean LDR tiles |
| hf-tid2013 | 1.5 GB, 25 refs | keep | ISP distortion statistics; clean LDR tiles |
| hf-sidd-dng-mirror | 21.5 GB | keep | noise calibration only, not a clean source |
| RawNIND | none | 120 GB | raw clean/noisy pairs: a real noise direction to check the synthetic one against |
| raw.pixls.us | 24.9 GB, 926 files (777 raws; the rest `*.tif` scan containers, excluded) | done | one raw per camera model: the held-out real-capture evaluation (`eval-raw`), never a training source |
| wikimedia-featured-pictures | 38 GB, 4,027 files (5,000 listed; 672 skipped for licence, type or size) | done | expert-reviewed camera originals; licence read per file by the API walk and kept as the first path component (CC0, public domain, CC BY, CC BY-SA; NC and ND skipped): the first large LDR photo corpus |
| wikimedia-quality-images | 6.7 GB, 914 files and collecting (20,000 listed, 2,475 skipped) | 100 GB by `--max-bytes` | the same review pipeline, 450 k files deep: the clean corpus now scales with the byte budget, not with a new source |
| MIT-Adobe FiveK | none | 51 GB, evaluation only | research-only non-commercial licence (DATA_PLAN principle 5): a real edit direction to evaluate against, never a training source |
| HDR+ full | subset above | stream, 3.7 TB | larger than D: (1.9 TB); fetch bursts on demand, never mirror |
| Hypersim | none | skip, 1.9 TB | would fill D: alone and is rendered without a sensor; stream frames only if a synthetic-scene prior is wanted later |

Planned total on D: about 32 + 6 + 1 + 120 + 25 + 40 + 100 = 324 GB of training-grade clean sources, FiveK's
51 GB beside them for evaluation, plus labels (section 4), leaving about 1.4 TB. C: keeps nothing of the data
root once the env var is set. Every pull is bounded (`--max-bytes`, `--param limit=N`) and resumable, so the
corpus grows by re-running the same command with a larger budget.

## 4. Label budget

Measured on the sample shard (2048 rows; parquet with pyarrow's default compression; 54 columns: 10 identity
fields, 31 parameters, 12 metrics, the per-row milliseconds):

| format | bytes per row | 10 M rows | 100 M rows |
|---|---|---|---|
| parquet (default) | 344 | 3.4 GB | 34 GB |
| JSONL | 1577 | 15.8 GB | 158 GB |

A shard holds `--chunk` images times crops times draws: 128 rows and 39 KB at the defaults. For 10 M rows use
`--chunk 64` (1024 rows, about 350 KB, 9.8 k shards); the ledger is one line per shard. Either way the labels
are a rounding error next to the clean corpus; the budget that matters is clean images, not label bytes.

Clean-image supply: at 4 crops times 4 draws a clean image yields 16 rows, so 10 M rows would need 625 k
images, more than any corpus here. The lever is rows per image. A 2k HDRI holds 32 disjoint 256 px tiles;
with `--crops 32 --per-tile 32` a thousand HDRIs give 1 M rows, and the full planned corpus (FiveK 5000 raws,
RawNIND, pixls, about 1100 HDRIs, the HDR+ bursts) gives 10 M at about 1 k rows per image. Scene diversity is
bounded by the corpus; the label count is not, and more draws per tile mostly add parameter coverage, which
is exactly what the pair construction of section 1 consumes.
Since the eighth run the sampler draws 16 tile areas per decoded image, each source is read at its own clean
scale with its `clean.exclude` patterns left out (section 3 rows), and the held-out split is by scene, so a
Poly Haven HDRI's resolutions and a burst's frames fall on one side of the hash.

## 5. Throughput

Measured: 16 processes on the 16-core box, 256 px tiles, `--ss 1`: 2048 rows in 36.6 s wall, 56 rows/s;
4.86 rows/s per process, 206 ms per row (p10/50/90 = 138/185/316 ms) with every core busy. A lone process
runs about twice as fast (metrics 37 ms plus degrade 12 to 72 ms, measured in isolation). The run had 18
chunks on 16 workers, so its tail ran with idle workers; a long run approaches 16 times the per-process rate.

| rows | at the measured 56 rows/s | at the steady-state 78 rows/s |
|---|---|---|
| 10 M | 50 h | 36 h |
| 100 M | 21 days | 15 days |

`--ss 2` roughly quadruples the camera stage; keep `--ss 1` for labels. Resume is a rerun of the same command:
shards whose ledger entry matches the configuration are skipped, a deleted shard is recomputed to identical
rows, and the rows do not depend on `--jobs` (all three tested). A worker process that dies takes the whole
pool down (`BrokenProcessPool`; on this Windows box a freshly spawned worker now and then starts with an
invalid queue handle, WinError 6, and dies on its first job): the chunks that had not finished are resubmitted
to a fresh pool, and a run gives up only after five pools in a row that finish nothing, so a finished chunk is
written once and no chunk is labelled twice. A chunk whose worker raises anything else (exhausted memory, a
bug) is logged as FAILED, counted in the summary and left for the next run, which plans its items again.

## 6. The metric vector: conditioning and evaluation, never the loss

Each of the twelve numbers is an estimator with a documented bias on natural tiles. A model trained to
minimise them learns the estimator, not the image. Use the vector to condition (the target state's vector),
to stratify and filter batches, and to evaluate, with the chart metrics in `harness.metrics` on the scene
as the held-out judge.

How the natural-tile estimators differ from the chart metrics ([harness/data/tilemetrics.py](harness/data/tilemetrics.py)):

- mtf50 and texture_hf_fidelity: shift-compensated cross spectrum, so a geometric shift does not read as blur;
  sharpening reads hf above 1 (p95 1.98 in the shard).
- shadow_snr_db and dr_steps: noise from the degraded tile's own high-pass residual, so blur does not deflate
  it; texture leaks in as a constant offset across draws on the same tile.
- delta_e: 8 px block means of in-range pixels only (clipped and shadow blocks excluded).
- edge_overshoot: after a sigma 8 px local gain match, so shading and tone curves do not read as halos.
- banding: smooth regions binned by clean level. periodicity: autocorrelation gain over the clean tile.
- landmark_error_px: per-channel phase shift of the whole tile. shading_rmse: 16 px block ratios.

Biases measured on the shard:

- highlight_loss_fraction is NaN on 27% of tiles (no over-range pixels) and about 1 whenever the camera stage
  is on, because the clean tile keeps values above 1.0 and a sensor output cannot. It separates camera-on from
  camera-off draws and little else.
- landmark_error_px is reliable when mtf50 > 0.05 cpp (p50/90 = 0.17/1.09 px) and meaningless in the
  heavy-blur tail (mtf50 <= 0.05 cpp on 24% of rows: p50 3.6 px, p90 32 px). Filter on mtf50 first.
- shadow_snr_db is NaN on 13% of tiles (no shadow pixels). periodicity rises with blur, as the chart metric
  does. NaN means "the tile lacks the pixels this metric needs"; JSONL stores it as null.

## 7. Synthetic-to-real gap

The camera stage is the harness's own model: Poisson shot noise, measured read-noise ladders, residual
Gaussian blur from the traces, lens shading, Bayer sampling, bit depth. Real raw files also carry fixed
pattern noise, PRNU, the demosaic of a specific ISP, lens-specific aberration shapes, motion blur and rolling
shutter, none of which is drawn. Real pipelines denoise and sharpen jointly; here they are independent coins.

What a tile can and cannot identify (learned from the first three model runs, 2026-10-27):

- Exposure. A dark input is either an under-exposed capture (the target is equally dark) or a mean-shift
  defect (the target is brighter). The measured offsets alone made every dark input a defect, so the first
  model brightened the demo camera 2.2x; the expose-for-the-peak policy in section 2 puts legitimately dark
  captures in the stream, which cut that to 1.07x by the third run. The residual is the ambiguity itself.
- Identity. Without (clean, clean) pairs the second run removed 30 % of a clean input's high-frequency energy
  and shifted its colour by delta_e 2.6; with `P_IDENTITY = 0.15` the third run changes a clean input by 4 %
  and keeps 95 % of that energy (`harness/model/data.py`).
- Lens shading. On one tile the shading map is a brightness gradient indistinguishable from scene lighting,
  and a target that removes it teaches the model to lift the dark side of any gradient (the third run lifted
  the demo's corners 8 to 21 % on a camera that has no shading). Real pipelines correct shading from
  calibration tables, not by inference, so from the fourth run the restore target keeps the input's
  uncorrected shading and the label reports it (`render_pair`): shading correction is a declared edit, never
  a default of restoration. If the whole-frame model is ever given a radial coordinate channel, this can be
  revisited, because the frame, unlike the tile, does identify the map.
- The target's exposure. The twelve metrics are measured against the target's own prepared reference, so
  they say nothing about how bright the target is. Edit pairs that drew the target's exposure independently
  gave the model an unobservable variable, and it regressed to the mean: at 9000 steps the third run made
  held-out restores 1.95x too bright when the target's median sat below 0.04 of the clip level and 0.75x too
  dark when it sat at mid-grey, which on the demo camera (median 0.036) was a 1.6x lift, delta_e 7.5 and a
  provenance fraction of 0.022. From the fourth run every target keeps the input's expose flag and exposure
  offset (`draw_pair`); an exposure change is a scalar multiply in linear light and belongs outside the model.
  The general rule: a conditioning vector must carry every variable the target depends on, or the pairs
  must hold that variable fixed.
- The target's global tone and colour. The ISP stage draws a contrast gain and a mean shift in sRGB, a
  saturation factor and a colour cast, and the restore target had none of them. A tile cannot identify an
  affine tone curve or a cast from the scene, so the fourth run, with exposure and shading held fixed,
  drifted as it trained: at step 1000 its demo output matched the camera's brightness (ratio 0.999, delta_e
  0.49), at step 4000 it was 1.06x too bright with the corners 1.09 to 1.15x, the blue channel 1.064x,
  delta_e 1.10 and shading_rmse 0.0204 (a loss). A lift that is largest where the frame is darkest is what a
  learned mean inverse of the mean shift looks like. From the fifth run every target keeps the input's eight
  global fields (`GLOBAL_FIELDS` in `harness/model/data.py`: expose, exposure_stops, scene_peak_over_median,
  contrast_gain, mean_shift, saturation, cast_r, cast_b) and the label is measured against the clean tile
  rendered in that intent; the tone affine, saturation and the cast are closed-form edits applied outside
  the model.
- The input's capture. With the target identified (the fifth run), what a tile still cannot identify is
  the input's own degradation: a 256 px tile does not separate a soft lens from soft subject matter, nor a
  noise level from fine texture, so an L1 restorer answers with the conditional median over every capture that
  could have produced the tile, which is smoother than any of them. The fifth run showed it on the demo's best-case camera: its step 1000 checkpoint tied the camera on resolution (mtf50 0.214 against 0.226) and lost shadows (15.6 against 26.2 dB); its step 9000 checkpoint, chosen by a validation loss on sampler pairs that fell monotonically from 0.098 to 0.067, sharpened the same frames to mtf50 0.443 against the camera's 0.220 (truth 0.600) and texture 0.351 against 0.223 with the brightness matched (luminance ratio 0.985), but with an edge overshoot of 0.092 against 0.005, 19 % of the highlight patch clipped, shadow SNR 12.0 against 26.2 dB and, on the video, 4.1 % undeclared invention (texture region 6.4 %, flat colour and wedge patches 11 %): a sharpening learned as the mean over every capture the sampler draws, applied to a camera that is nearly diffraction-limited and nearly noiseless, which that validation loss never sees.
  From the sixth run the conditioning also carries the input's capture descriptor (`capture_descriptor` in
  `harness/model/data.py`: the Airy first zero at 550 nm and the residual blur sigma in pixels, log10 of the
  shot noise at the clip, of the read noise and of one LSB relative to the clip, the Bayer flag and a known
  flag), which is what a calibrated pipeline knows about its own camera and what a deconvolution needs. It is
  withheld with probability 0.3 during training so the same model serves a frame of unknown origin blind;
  `eval-demo` passes the demo camera's descriptor (`--blind` withholds it) and `apply --capture` takes it as
  JSON. The sixth run measured it on the demo: with the descriptor its step 9000 checkpoint beats the camera on
  resolution by more than the fifth run did (mtf50 0.447 against 0.220, texture 0.522 against 0.223), halves
  the colour error (delta E 1.04 against 1.30), cuts the still's undeclared invention to 0.05 %, and still
  fails tone_color on the still (25 % of the highlight patch clipped against 19 %, shadow SNR 13.5 against
  26.2 dB, edge overshoot 0.103) and provenance on the video (7.5 %); blind it is gentler (mtf50 0.392,
  highlight loss 3 %, overshoot 0.058, video invention 3.0 %) and fails the same two gates. A clean ramp
  comes back at 0.96 to 0.98 of the input and a clean step edge shows no overshoot, while the demo wedge's
  deepest patch comes back 1.8x bright at 5.8 dB with the descriptor and at 2.2 dB blind: neither the halo
  nor the shadow floor is the conditioning's to fix.
- What the pairs teach and where the output can resolve. With the target and the input's capture identified
  (the fifth and sixth runs), two faults stayed at every checkpoint of both runs and no conditioning could
  carry them away. First, a halo prior: the IQA ladder's unsharp stage was on in half of all inputs at the
  KADID levels (gain q10/50/90 1.53/3.65/6.33 against an overshoot gate of 0.02 and a camera at 0.005), so the
  median restore target of a small bright patch between darker separators is darker than the input (on the
  sampler's own pairs the target-to-input ratio at 0.8 to 0.9 of the clip has median 1.0 but mean 0.914 and
  q10 0.477), and the demo wedge's bright patches came back about 5 % dark while a clean ramp came back at
  0.97 to 1.0: a prior, not a tone curve. From the seventh run the restore inputs are drawn with the sampler's
  `capture` profile (section 2) with probability 0.7, and the edits keep the IQA profile so a declared halo
  is still removed. Second, the output floor: the network's precision is absolute in the encoded domain (a
  flat mid-grey field comes back with std 0.001 to 0.002 on the sixth run's checkpoints, 0.0024 on the fifth) and in `log1p(8 x)`
  that floor was 20 to 50 % of the signal at 0.001 of the clip, where the deepest wedge patches sit (shadow
  SNR 12 dB on the fifth run and 4 to 7 dB on the sixth against the camera's 26 dB). L1 gradients are
  sign-based, so the domain decides where the output can resolve the signal, not a per-pixel weight. From the
  seventh run the domain is `log1p(4096 x)`: one 12-bit LSB is where the log turns linear, the clip encodes to
  8.3 with slope 1.0 there (0.9 before), and the floor at 0.001 of the clip is 100x finer. Measured on the seventh run (2026-10-28, step 9000 checkpoint): the control held, a flat mid-grey field comes back with std 0.0019 at step 1000 and 0.0038 at step 9000, so the finer domain did not make the mid-tones noisier; the deepest wedge patch at 0.001 of the clip comes back at 13.2 dB known and 15.0 dB blind against the camera's 20.8 (5.8 dB on the sixth run), and the still's shadow SNR rose from 13.5 to 22.8 dB known and 25.6 dB blind, which ties the camera's 26.2. The halo prior went with the ladder: a clean ramp comes back at 1.000 at mid-tones. The still still fails tone_color (known: mtf50 0.500 against 0.220, texture 0.472 against 0.223, landmark error 0.26 against 0.45 px, highlight loss 0.429, delta E 0.86, edge overshoot 0.178, banding 0.0067 against 0.0035; blind: mtf50 0.446, highlight loss 0.325, delta E 0.66, overshoot 0.130) and the video fails provenance (3.6 % known, 3.7 % blind): the brightest wedge patches come back 6 to 9 % dark and the mid and bright patches 10 to 20 dB noisier than the camera, the deconvolution amplifying noise where the camera has little. A clean ramp came back 5 % dark at every level at step 1000 and the shadows 13 % dark at step 2500; that darkening does not depend on the conditioning and the pairs' median target-to-input ratio is 1.0, and it was traced to a bug in how the pairs carried the input's lens shading into the target, the next bullet.

- The halo's price. The eighth run's bright-side overshoot (0.207 known on the still, and 0.029 on 45 of 64 real
  raw tiles whose input had none) is what L1 against a sharpened target buys: a thin overshoot costs one
  pixel-width of small error and pays for a steeper edge everywhere. The ninth run charges every output value
  above the target's 3 by 3 maximum or below its minimum once more (`RANGE_WEIGHT` in
  `harness/model/train.py`; step 6743 when the 75-minute budget ran out). Measured: overshoot 0.098 known and
  0.044 blind on the chart, 0.0003 and 0 on the real raws; the shadow SNR stops falling on the raws (14.6 dB
  against the input's 14.8, where the eighth run fell to 12.9) and ties the camera on the chart (25.6 against
  26.2 dB); the dynamic-range step the eighth run lost on the raws is kept; and about half of the eighth run's
  resolution gain is gone (mtf50 0.384 against 0.559 on the chart, 0.307 against 0.365 on the raws, the inputs
  at 0.220 and 0.254). A range clamp is not a model of an edge: the 0.098 that remains on the chart, against a
  gate of 0.01, is where the halo is wider than the 3 by 3 window.
- Synthetic to real, measured for the first time (`python -m harness model eval-raw`, `harness/model/heldout.py`,
  2026-10-28): 64 tiles of 256 px from raw.pixls.us at clean scale 4, one raw per camera model and never a
  training source, rendered through the demo claim's camera with the sampler's exposure policy and scored with
  the twelve tile metrics against the clean tile, input and model alike. Three things the chart did not show at
  this size: the eighth run adds noise in real shadows (shadow SNR 14.8 to 12.9 dB, losing on 19 tiles), puts a
  halo on edges that had none, and drops a dynamic-range step on 28 tiles. And known and blind differ by less
  than the tie margins on every metric for both runs, so the capture descriptor carries almost nothing on real
  captures. The references bound the score: each carries its own camera's noise and demosaic (scale 4 averages
  most of it away), so a model that removes real noise is charged for it. The real pairs of step 4 (RawNIND's
  clean mates, HDR+ burst merges) are the instrument without this bound.

So the real sources are the check, not decoration: label RawNIND's noisy raws and the HDR+ bursts against
their clean mates with the same tile metrics and compare those distributions with the synthetic rows at
matched ISO; FiveK's five retouches per raw are the real "state A to state B" direction. A model trained on
synthetic pairs is evaluated on those real pairs and on the chart metrics, never on its own synthetic labels.

Open decisions: cap or reweight the lens-trace blur tail (24% of rows near featureless); draw the ISP coins
from a measured ISP instead of 0.5 each; add motion blur and demosaic once measured; `--ss 2` for the
aliasing-sensitive metrics.

## 8. Steps

1. Set `HARNESS_DATA_ROOT=D:\lens-harness-data` and move the data root there.
2. Collect Poly Haven and ambientCG HDRIs at 2k, raw.pixls.us, RawNIND, Wikimedia featured pictures and
   quality images (licence per file), each under `--max-bytes`; FiveK separately and for evaluation only.
   Status 2026-10-28: Poly Haven at 1k, raw.pixls.us (25 GB), the featured pictures (38 GB) and 40 GB of RawNIND
   are on D:; the quality images are collecting.
3. `python -m harness data label --sources <all clean ids> --n 10000000 --chunk 64 --crops 16 --per-tile 16`
   (36 to 50 h on this machine; resumable per item: the ledger's shards say which files are labelled under the
   current tile, draw and crop settings, so a re-run after the corpus grew labels only the new files, into new
   chunks, and never rewrites a shard). Status 2026-10-28: 42,048 rows in 351 shards at tile 256, 4 draws, 4 crops;
   on this branch the pass stops after one chunk when a worker dies (`BrokenProcessPool`, a native decoder taking
   a worker down), and the pool retry that survives it is on the `pixel-label-raw-exr` branch, not yet merged.
4. Real-pair labels from RawNIND and HDR+ with the same tile metrics; compare distributions.
5. Train with the metric vector as conditioning; evaluate on the chart metrics and the real pairs.
   Started 2026-10-27 in `harness/model/` (pairs rendered on the fly by the sampler, conditioned residual UNet,
   `python -m harness model train|apply|eval-demo`); nine runs, six on 2026-10-27 and three on 2026-10-28, each fixing one unidentifiable
   variable found by the previous one (section 7); the measured demo-world verdicts per run are in README.md
   and LANDSCAPE.md section 6. Since the ninth run `python -m harness model eval-raw` scores a checkpoint on real
   raws no run trains on (raw.pixls.us through the demo camera, section 7); the real-pair evaluation waits on step 4.
