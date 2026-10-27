# Verifiably better than any lens — spec + harness

A claim that a generated image or clip is "better than any lens" is only checkable
as a tuple: **which qualities, on which scene, against which capture constraints, on
which display**. [SPEC.md](SPEC.md) is the contract; `harness/` measures it.

The harness builds a synthetic world it knows exactly (ground truth), simulates the
best physically possible camera on that same world (the baseline), then scores a
candidate against both and runs the gates in order. Stills and video, from the start.

## Install

```bash
pip install -r requirements.txt
```

Python ≥ 3.10, numpy, scipy, Pillow, pytest. Video *files* (mp4 in or out) additionally
need `imageio` + `imageio-ffmpeg`; `.npy` masters and PNG frame folders work without them.

## Self-test

```bash
python -m harness demo
```

Builds a still world (768×432) and a 24-frame clip (384×216), exports truth + camera-sim
frames and `world.json` for each, then evaluates two candidates:

| candidate | expected |
|---|---|
| ground truth itself | **INCOMPLETE (skipped: composition)** — beats the camera on resolution, shadows, geometry/lighting and temporal (video); ties on highlights, color and artifacts, because the baseline is exposed for the scene it is handed and holds the whole wedge; the composition gate needs a human study and a skipped check is not a pass (the still run also skips temporal) |
| the camera-sim output | **FAIL at resolution** — every metric ties itself, and `mtf50_cpp` has to beat the camera, not tie it |

Reports land in `demo_out/<still|video>/candidate_<truth|camera>/report.md`.
`--quick` shrinks everything for a smoke run.

## Evaluate your own candidate

1. Export the world your generator must reproduce:

   ```bash
   python -m harness render --claim claims/example.json --out world
   ```

   `world/truth.npy` and `world/truth_*.png` are the target; `world/camera.*` is the
   baseline; `world/world.json` lists the regions, landmark trajectories, wedge
   exposures and patch Labs. Feed whatever your model needs (the PNGs, the geometry,
   a text description of the layout) — but the harness never resamples, so the
   candidate must come back at exactly `width × height × n_frames`.

2. Evaluate:

   ```bash
   python -m harness run --claim claims/example.json --candidate cand.npy --out result
   ```

   Exit code 0 on PASS, 1 on FAIL, 2 on INCOMPLETE (every gate that ran passed but a check
   had no data, such as the human composition study: a skipped check is not a pass).
   Add `--rendered-map map.png` (or `.npy`, luminance > 0.5 = rendered, one frame or per
   frame) if the candidate declares which pixels it rendered rather than recovered: the
   `provenance` gate passes only when at most 1 % of the pixels carry structured content
   the truth does not explain that the candidate has not declared. Without a map every pixel
   counts as declared recovered; a claim with `"rendered_map": {"policy": "required"}` makes
   a missing map INCOMPLETE instead. The report's provenance block gives the invented,
   declared and undeclared fractions per scene region, the matched blur per frame, and
   `invented_detail.npy` beside it is the measured map. The detector finds pixels whose
   structure no linear filter of the truth explains: blur, sharpening and noise measure 0,
   the best-case camera 0 to 0.5 % at isoluminant colour edges, a JPEG q92 delivery of it
   3 %, and planted 32 px patches at 20 % contrast are found at 67 to 100 % recall. Pixels
   under moving landmarks are judged by the temporal gate instead (motion_region_error).
   `result/report.md` has the metric table (candidate / camera / truth / verdict), the
   per-quality verdicts, the gate trace, and the delivery check (metrics re-run after the
   display's encode).

   The claim's `camera` block is a preset plus overrides. By default the baseline exposes for
   the scene it is handed: the scene's peak radiance lands at `white_fill` × full well
   (`white_radiance` pins another radiance there). Set `scene_white_cd_m2` > 0 for absolute
   photometry: electrons follow luminance, exposure time, f-number, pixel area,
   `quantum_efficiency` (0.6) and `lens_transmission` (0.9), so an over- or under-exposed
   capture is simulated as such (`CaptureConstraints.exposure_for_peak` gives the exposure that
   fills the well, `saturation_iso` the ISO 12232 figure). `black_pedestal_e` (default 8 × read
   noise) keeps shadows unbiased and `row_noise_e` adds row-correlated noise; the old
   `sensitivity_e_per_s` field is gone.

Candidate formats: `.npy` `(T,H,W,3)` or `(H,W,3)` float linear RGB (the master format;
the only one that can carry HDR), a PNG/JPEG still, a folder of frames, or an mp4 with
imageio. **8-bit inputs are decoded as sRGB and clip at 1.0, so they fail
`highlight_loss_fraction` by construction** — a candidate that claims highlight
superiority has to arrive as a float master.

`claims/example_still.json` is the single-frame variant; edit `scene`, `camera`,
`display` and `gates` freely. `validate_claim` rejects unknown metrics and operators.

## The composition gate (human preference)

Composition and material plausibility are not measured, they are *preferred*: a blinded
pairwise study on the claimed display. The harness makes the manifest and fits the
model; any trial collector (a web form, a lab script) runs the trials.

```bash
python -m harness preference manifest --item candidate=cand.png --item camera=camera.png --item truth=truth.png --out study
# raters see study/stimuli/<opaque id>.png pairs per study/manifest.json; keep study/key.json away from them
# collector writes responses.csv with columns: trial,question,choice   (choice = left|right)
python -m harness preference fit --manifest study/manifest.json --key study/key.json --responses responses.csv --out fit.json
python -m harness run --claim claims/example.json --candidate cand.npy --out result --preference fit.json
```

The fit is Bradley–Terry with a Rao–Kupper tie band and one reliability scale per rater
(a `rater` column and `tie` choices are optional; without them it is plain Bradley–Terry),
with bootstrap intervals; the gate uses the **lower** 95 % bound of P(candidate ≻ camera)
for "convincing", so an underpowered study fails rather than passes. The same model is
validated on five public vote sets (860,000 votes) by `python -m harness data derive --ids
preference_public`; DATA_PLAN.md §4 step 3 has the held-out numbers.

## Reading a report

- **verdict** per metric is beats / ties / loses against the camera-sim, with a
  per-metric tie margin (`harness/metrics/__init__.py`); a quality's verdict is the
  worst of its metrics.
- **Gates** run in the order of the claim and stop at the first failure; a gate whose
  inputs are missing (no preference fit, still-only candidate on temporal) is `skip`,
  not `pass`.
- **Delivery** shows what survives the display's encode. The JPEG in the example claim
  destroys the highlight and shadow wins — that is the point of the check.

## Tests

```bash
python -m pytest -q
```

On a sandboxed Windows shell `tmp_path` may be unwritable; use
`mkdir -p .tmp/pt` then `python -m pytest -q -p no:cacheprovider --basetemp=.tmp/pt`, and delete `.tmp` after.

## The recovered layer (model)

`harness/model/` trains the restorer that LANDSCAPE.md section 6 calls the recovered layer. Training pairs are
never stored: each worker draws a clean tile, draws a degradation state from the measured distributions
(`harness/data/degrade.py`), renders input A and target B, and labels B with the twelve tile metrics. With
probability 0.15 the input is the clean prepared tile itself (identity: a clean input must come out unchanged);
otherwise with probability 0.7 the target is the clean prepared tile (restore; its input is drawn with the sampler's
raw `capture` profile 70 % of the time, DEGRADATION_PLAN.md section 2), else a second drawn state (an edit), so the model learns to move an image to a stated metric vector, not only to "better". The metric vector is the
conditioning and the judge, never the loss (the Goodhart rule in DEGRADATION_PLAN.md section 6). The network is a
conditioned residual UNet (8.2 M parameters, identity at initialisation) on log-encoded linear RGB (`log1p(4096 x)`: one 12-bit LSB is where the log turns linear, so above that floor an
L1 error is a relative error); the loss is a weighted L1 plus an edge L1 in that domain.

The first run (`recovered_v1`) brightened every frame of the demo's best-case camera by 2.2x and failed the
provenance gate, because the sampler's measured exposure offsets are all near zero while that camera exposes
for the highlights at a median 1.8 stops below mid-grey: to the model, a dark input was always a mean-shift
defect. The sampler now has an expose-for-the-peak policy (`P_EXPOSE_FOR_PEAK = 0.5`, peak placed at 0.5 to 1.0
of the clip level, DEGRADATION_PLAN.md section 2), whose restore target is equally dark, so darkness alone is
no longer something to undo. The measured verdicts of every run are in LANDSCAPE.md section 6.

The third run (`recovered_v3`, identity pairs added) lifted the demo camera's corners by 8 to 21 %: a single tile
cannot tell lens shading from a dark subject, and the restore target was the flat prepared tile. From the
fourth run the restore target keeps the input's uncorrected lens shading (`render_pair` in `harness/model/data.py`
multiplies the target by the input's shading map), so shading is scene to the model while the label still
reports it; an edit whose target state declares corrected shading still flattens. DEGRADATION_PLAN.md section 7
records what a tile can and cannot identify.

The fourth run (`recovered_v4`, exposure and shading held fixed) matched the camera's brightness at step 1000 and
then drifted as it trained, because the ISP's sRGB tone affine, saturation and colour cast were still drawn for
the input and absent from the target. From the fifth run (`recovered_v5`) every target keeps the input's eight
global fields (`GLOBAL_FIELDS` in `harness/model/data.py`) and the label is measured against the clean tile
rendered in that intent; exposure, tone, saturation and cast changes are closed-form edits applied outside the
model. The fifth run's step 9000 checkpoint is the first to beat the camera on resolution and texture (mtf50 0.443 against 0.220 on the still, texture 0.351 against 0.223, landmark error 0.30 against 0.45 px) with the brightness matched (luminance ratio 0.985), and the gates reject it anyway: the still fails tone_color (19 % of the highlight patch clipped, shadow SNR 12.0 against 26.2 dB, delta_e 1.30) with an edge overshoot of 0.092 against the camera's 0.005, and the video fails provenance (4.1 % undeclared invention against the 1 % gate; 0.3 % on the still). Its step 1000 checkpoint had tied resolution and failed only dr_steps, so the sharpening grew with training while the validation loss on sampler pairs fell from 0.098 to 0.067: that loss does not see the demo's camera.

The sixth run (`recovered_v6`) adds the input's capture descriptor to the conditioning (`capture_descriptor` in
`harness/model/data.py`: Airy radius, residual blur, shot and read noise and one LSB relative to the clip, Bayer
flag, known flag), withheld 30 % of the time so the model also works blind; `eval-demo` passes the demo camera's
descriptor unless `--blind`, and `apply --capture file.json` passes a real camera's. Its step 9000 checkpoint (validation loss 0.0667, 55 min on the RTX 5090) beats the camera on resolution by
more than the fifth run did (mtf50 0.447 against 0.220, texture 0.522 against 0.223, landmark error 0.29 against
0.45 px), halves the colour error (delta E 1.04) and keeps the still's undeclared invention at 0.05 %, but fails
tone_color on the still (highlight loss 0.250, shadow SNR 13.5 against 26.2 dB, edge overshoot 0.103) and
provenance on the video (7.5 %); blind, it fails the same two gates more gently (mtf50 0.392, highlight loss
0.031, overshoot 0.058, video invention 3.0 %).

The seventh run (`recovered_v7`) changes what the pairs teach and where the output can resolve, the two faults the
sixth run's conditioning could not carry away (DEGRADATION_PLAN.md section 7): restore inputs come from the
sampler's raw `capture` profile 70 % of the time (camera on, measured ISP stages only, no unsharp ladder, whose
halos had taught the model to darken small bright patches), and the domain is `log1p(4096 x)` instead of
`log1p(8 x)`, so the network's absolute output floor (std 0.001 to 0.002 at mid-grey) no longer swamps the deepest
shadows. Its step 9000 checkpoint (54.7 min on the RTX 5090; its validation loss, 0.1978, is in the new domain and not comparable to the sixth run's 0.0667) raises the still's shadow SNR from 13.5 to 22.8 dB known and 25.6 dB blind, which ties the camera's 26.2, keeps resolution and texture ahead (mtf50 0.500 against 0.220, texture 0.472 against 0.223, landmark error 0.26 against 0.45 px) and lowers the colour error to delta E 0.86 known and 0.66 blind, but still fails tone_color on the still because the brightest wedge patches come back 6 to 9 % dark (highlight loss 0.429 known, 0.325 blind) and the edge overshoot grew to 0.178 known and 0.130 blind; the video fails provenance at 3.6 % known and 3.7 % blind. A clean ramp came back 5 % dark at step 1000 of this run; the cause was a bug in how the pairs carried the input's lens shading into the target, fixed in the eighth run.

```bash
python -m harness model train --out data/models/recovered_v7 --steps 9000 --minutes 75 --batch 16 --tile 256 --workers 12
python -m harness model eval-demo --ckpt data/models/recovered_v7/best.pt --out demo_out_model
python -m harness model eval-demo --ckpt data/models/recovered_v7/best.pt --out demo_out_model_blind --blind
python -m harness model apply --ckpt data/models/recovered_v7/best.pt --input frames.npy --out restored --white-level 1.0 --capture capture.json
```

`eval-demo` builds the demo worlds, runs the camera-sim frames through the model at the camera's clip level and
judges the result through the gates like any candidate, so the report says on which gates the model beats the
best-case camera and whether it stays under the provenance limit. `train` resumes from `last.pt`, logs to
`train_log.jsonl` (loss, images per second, per-metric medians of input and output against the clean target on
the held-out restore set every checkpoint), and keeps `best.pt` by validation loss. Held-out items are a hash
split of the clean sources (one in sixteen).

## Data: no invented constants

Every number the harness used to make up (sensor pitch, full well, read noise, display size,
the ppd threshold, the ΔE / banding gates, viewing distance) now comes from a sourced row:

- `harness/data/reference/*.csv` — hand-verified rows (sensors, lenses, displays, vision
  thresholds). **No source, no row**: every row carries a `source_url` and a `quote`; the
  reader refuses a row without one.
- `harness/data/sources.json` — the manifest: 122 sources (spec PDFs, vendor pages, Zenodo
  records, Hugging Face datasets, CC0 asset libraries) with licence, size tier
  (S < 50 MB, M < 2 GB, L < 50 GB, XL), access kind and what each feeds.
- `harness/data/catalog/datasets.json` — 122 human-rated image/video datasets, with which
  gate each one can calibrate.
- `data/derived/*.json` — harness-ready tables built from the raw files by `derive`:
  `camera_presets.json` (use `"preset": "derived:<sensor id>"` in a claim),
  `displays.json` (`"preset": "derived:<display id>"`), `vision_thresholds.json`, the
  CIE / ColorChecker / camera-spectral tables, and `gates_fitted.json`: gates put to human
  ratings instead of quoted. `fit_gates` (`harness/data/fit.py`) fits KADID-10k ladders with the
  reference-based estimators in `harness/metrics/natural.py` and applies a stated support rule
  before a fit may replace a sourced value. Edge overshoot: 0.22 at mean DCR 4.5, 95 % CI
  0.19–0.26, rank correlation −0.80, supported but extrapolated below the mildest rated level, so
  it is reported beside the gate, not adopted as the pass line.
  Banding: not supported (rank correlation −0.31; KADID quantization posterises whole images, so
  the ratings bound the threshold only from above, below 0.145); the sourced 0.01 stays and
  TID2013 quantization noise agrees. `claim_sourced.json` records both outcomes and keeps
  0.02 / 0.01 as the pass lines (the default claim's engineering targets), each fit beside its gate. `shift_check.json`: the same two
  artifacts read without a reference (`shift_check`, `harness/data/shift.py`) on the 2,400
  AIGCIQA2023 images against their MOS and on the KADID references. The halo proxy does not
  calibrate on photographs and reads sharpness on generated images, so the fitted gate's drift is
  unmeasured; banded area is the one artifact the raters punish (rank correlation −0.49 with
  quality MOS, carried by one generator). DATA_PLAN.md §4 step 2 has the numbers.
  `lens_traces.json` (`lens_traces`, `harness/data/optics.py`, needs `pip install optiland`):
  295 of 304 lens prescriptions (Claff patent archive + DeepLens) ray-traced,
  refocused on axis, with centre Strehl, MTF50 at four fields and the lens-only residual blur. The
  sharpest full-frame lens (85 mm F/1.5, MTF50
  385 lp/mm, residual 0.27 µm) becomes the
  `best_case_traced` preset. DATA_PLAN.md §4 step 4 has the distribution and the caveats.
  `noise_check.json` (`noise_check`, `harness/data/noise.py`, needs `pip install rawpy`): `camera.py`'s
  shot + read noise law measured on 4 SIDD Google Pixel bursts (ISO 100 to 6400, 60 frames each):
  per-pixel temporal variance against the mean, outlier frames dropped, lamp drift scaled out, moving
  edges masked. R² ≥ 0.997, gain ∝ ISO^-1.00, read noise 7.2 e- at ISO 100 falling to
  2.0 e- at ISO 6400, clip 4,662 e- at base ISO; `camera_presets` turns it into
  `measured_google_pixel` (alias `phone_measured`). A second camera, iPhone 7 at ISO 100 to 1600 (5 bursts), gives
  `measured_iphone_7` (clip 1,385 e-, read 2.0 e-, 12-bit) whose gain does not fall with ISO (slope -0.05), so
  its electron units are relative to the DNG scale. DATA_PLAN.md §4 step 5 has the per-ISO numbers and the caveats.
  `hdri_stats.json` (`hdri_stats`, `harness/data/hdri.py`, numpy only): the luminance range of 997 real
  environments (every Poly Haven CC0 HDRI at 1k), solid-angle weighted. Scene range (p0.1 to p99.9)
  median 10.0 stops, 16.7 stops with the brightest source; outdoor
  9.1, indoor 12.8. This is where a claim's scene class S gets its stop count.
  `polyhaven_catalog.json` counts the CC0 asset libraries (4,858 assets) and what is on disk.
  DATA_PLAN.md §4 step 7 has the per-category numbers and the caveats.
  `capture_stats.json` (`capture_stats`, `harness/data/dng.py`, a struct-only DNG tag reader, no library): what a
  phone chose in the field on the first frame of 153 HDR+ bursts: exposure median 1.7 ms (9 % at
  1/30 s or longer), ISO median 60, scene luminance median 634 cd/m² (p10 5.7, p90 2904), the DNG
  noise profile within 10 % of the SIDD-measured gain for the Pixel, lens shading corner falloff 1.5 stops,
  which `camera.py` applies as `shading_corner_stops` (1.78 on the Pixel preset, from the Pixel and Pixel XL maps):
  the ISP's correction keeps the brightness and multiplies the corner noise, so corner SNR falls by 2^(stops/2).
  This is where a claim's capture constraints C get real values. DATA_PLAN.md §4 step 9 has the per-device
  numbers and the caveats.

```bash
python -m harness data list --tier L                 # what exists, what is on disk
python -m harness data collect --tier S              # fetch everything under 50 MB (resumable)
python -m harness data collect --ids hf-tid2013 --param limit=1 --max-bytes 500e6
python -m harness data status                        # ledger summary
python -m harness data collect --ids hf-kadid10k-images hf-iqa-dataset-labels --jobs 6 --max-bytes 4e9   # 3.1 GB, for fit_gates
python -m harness data collect --ids hf-aigciqa2023-images --max-bytes 1e9                           # 893 MB, for shift_check
python -m harness data collect --ids spring-flow --param pattern=test_cam_data --max-bytes 1e6        # Dataverse API, 3.8 KB check
python -m harness data collect --ids hdr-plus-burst --param 'pattern=bursts/0006_20160722_115157_431/' --max-bytes 1e9   # one HDR+ burst, GCS listing
python -m harness data collect --ids rawnind --param 'pattern=yaml$' --max-bytes 1e7                  # RawNIND indexes, then one scene by name
python -m harness data derive                        # rebuild data/derived (fit_gates needs the line above)
python -m harness data derive --ids lens_traces      # ray-trace every lens prescription (optiland, ~6 min on 15 workers)
python -m harness data derive --ids noise_check      # SIDD bursts (rawpy, ~33 s for four bursts on 4 workers)
python -m harness data collect --ids polyhaven-hdris --jobs 6 --max-bytes 2e9                       # 997 CC0 HDRIs at 1k, 1.6 GB
python -m harness data derive --ids hdri_stats polyhaven_index   # scene range from real environments + asset catalogue
python -m harness data collect --ids hf-sidd-dng-mirror --param 'pattern=data/train/(0028|0068|0032|0030|0088|0134)_DNG' --max-bytes 2e10   # iPhone 7 bursts, 14.1 GB
python -m harness data collect --ids hdr-plus-burst --param 'pattern=(payload|lens_shading_map)_N000[.](dng|tiff)$' --max-bytes 4e9   # HDR+ first frames, 3.3 GB
python -m harness data derive --ids capture_stats    # real capture constraints from the HDR+ first frames (about 20 s, no rawpy)
python -m harness data plan --out DATA_PLAN.md       # refresh the plan table
```

The collector is parallel, resumable (HTTP Range + append-only `data/ledger.jsonl`),
verifies sha256/md5 when the manifest states one, throttles per host and never retries a
host that refused it. Set `HARNESS_DATA_ROOT` to put the large tiers on a real disk; gated
Hugging Face sets read `HF_TOKEN`. `claims/example_sourced.json` is a claim whose camera,
display and gates all resolve to derived tables. [DATA_PLAN.md](DATA_PLAN.md) lists every
source, its status and the remaining gaps.

## Scope boundary

The harness scores candidates against **synthetic ground truth it generated**. It never
estimates flow, geometry or lighting from arbitrary footage: reconstructing reality from
a photograph is the information-bounded regime (SPEC §2) that no generator can win by
itself, and nothing here pretends otherwise.

## Layout

```
SPEC.md                 the contract (claim schema, regimes, baseline, metrics, gates, report)
claims/                 example claims (video, still)
harness/scene.py        procedural ground truth; analytic flow and landmarks
harness/camera.py       best-case camera: Airy PSF, shot/read noise, full well, lens shading, Bayer, motion blur
harness/display.py      ppd, mastering rule, encode round-trip
harness/metrics/        resolution, tone, color, artifacts, geometry, temporal, natural (reference-based)
harness/gate.py         ordered checks, stop on first fail
harness/report.py       report.json + report.md
harness/preference.py   blinded manifests, Bradley–Terry fit
harness/evaluate.py     world build + candidate evaluation
harness/cli.py          demo | run | render | preference | data | model
harness/data/           manifest, parallel resumable collector, expanders (HF, Zenodo, ...), derivers, gate fitting
harness/data/reference/ sourced CSVs: sensors, lenses, displays, vision thresholds
harness/data/catalog/   122 human-rated image/video datasets and what each can calibrate
harness/model/          the recovered layer: on-the-fly pairs from the sampler, conditioned UNet, trainer, inference
data/                   raw/<source>/..., ledger.jsonl, derived/*.json (created by `data collect|derive`), models/<run>/
DATA_PLAN.md            every source, tier, licence, status; gaps and the scaling plan
tests/                  physics sanity, metric sensitivity, pipeline
```
