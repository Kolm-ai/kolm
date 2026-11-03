# Verifiably better than any lens — spec + harness

A claim that a generated image or clip is "better than any lens" is only checkable
as a tuple: **which qualities, on which scene, against which capture constraints, on
which display**. [SPEC.md](SPEC.md) is the contract; `harness/` measures it.

The harness builds a synthetic world it knows exactly (ground truth), simulates the
best physically possible camera on that same world (the baseline), then scores a
candidate against both and runs the gates in order. Stills and video, from the start.

## Install

This app is `apps/pixel` of the kolm monorepo; every command below runs from that directory.

```bash
cd apps/pixel
pip install -e ".[data,model,dev]"
```

Python ≥ 3.10; numpy, scipy and Pillow are the core. Extras: `data` (requests, pyarrow: the collector and
parquet labels), `model` (torch: the recovered layer), `measure` (rawpy, optiland: `noise_check` and
`lens_traces`), `decode` (rawpy, OpenEXR: the labeller's camera-raw and EXR inputs, plain or inside zips;
without them those items are recorded as skipped with a reason, never a crash), `video` (imageio +
imageio-ffmpeg: mp4 in or out; `.npy` masters and PNG frame folders work without them), `dev` (pytest). The data root (ledgered sources, labels, checkpoints: tens of GB) lives outside
the repo: `HARNESS_DATA_ROOT=D:\lens-harness-data` on the GPU box; unset, it is `./data`, which git ignores.

## Restore a photo

The finished model is a release directory (`python -m harness model export`): `model.safetensors`, the EMA weights,
and `config.json`, which carries the network, the metric normaliser, the default restore conditioning, the blind
noise reading it was calibrated with and the weights' SHA-256, checked on load. Its model card states where it beats
the best-case camera and where it does not: [releases/kolm-pixel-v15.md](releases/kolm-pixel-v15.md).

```bash
pip install -e ".[model,decode]"
python -m harness photo IMG_0001.DNG more_raws/ --ckpt D:\lens-harness-data\release\pixel-v15 --out restored --original
```

Each input writes `<stem>.tif` (16-bit sRGB), `<stem>.jpg` (preview) and `<stem>.json` (the decode, the capture
descriptor and the noise law behind it, the DNG's own NoiseProfile beside it when it has one, timings); `--original`
adds the unrestored decode at the same exposure, `--linear` the 32-bit float linear frame, `--ev auto` lifts the
exposure of the encoded files (never darkens). A raw decodes through LibRaw as training's real pairs were decoded
and its noise law is read blind from the frame ([harness/data/blindnoise.py](harness/data/blindnoise.py)); a raw
whose law does not fit, and any other image, is restored with no descriptor; `--capture capture.json` states the law
instead. `--scale 2` restores at half resolution, the decode training's real pairs and their held-out verdicts use;
the default keeps every pixel (the blind reading carries a variance factor of 1.70 there, measured on 40 train pairs
decoded at full resolution; capture-eval's capture gates pass at scale 1 in all three modes, but no camera or
frontier verdict covers it). On the RTX 5090, beside a training run, a 50-megapixel Pixel 8 Pro raw restores in 33 to 44 s
and a 26-megapixel Sony ZV-E10 II raw in 18 to 22 s; `--device cpu` works, slowly.

A release from a run, its capture-descriptor check and its card:

```bash
python -m harness model export D:\lens-harness-data\models\recovered_v15\best.pt D:\lens-harness-data\release\pixel-v15 --name kolm-pixel-v15
python -m harness capture-eval --ckpt D:\lens-harness-data\release\pixel-v15 --out capture_eval_s1 --scale 1   # measured, blind-read and no descriptor on held-out real pairs
python -m harness model card D:\lens-harness-data\release\pixel-v15 --capture-eval capture_eval_s2 capture_eval_s1 --calibration blindnoise_calib.json blindnoise_calib_s1.json
```

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
CI (`.github/workflows/pixel-tests.yml`) runs the suite and the demo on every pull request that touches
`apps/pixel`; the tests that need the measured distributions skip when the data root has none.

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

The eighth run (`recovered_v8`) keeps the seventh's domain and pair recipe and fixes the pair bug the seventh exposed: the target now carries the input's uncorrected lens shading unchanged (`input_shading` in `harness/model/data.py`, shading treated as scene), so the model no longer learns to darken bright patches to pay for a second shading map. Its step 9000 checkpoint (58.1 min on the RTX 5090, validation loss 0.1845 against the seventh's 0.1978 in the same domain) cuts the still's highlight loss from 0.429 to 0.010 known and from 0.325 to 0.009 blind, lowers the colour error to delta E 0.60 known and 0.42 blind (max 1.69 and 1.28), extends the resolution lead (mtf50 0.559 known and 0.530 blind against the camera's 0.220, texture 0.587 and 0.480 against 0.223, landmark error 0.25 against 0.45 px) and brings the shading RMSE (0.011 known, 0.009 blind) to a tie with the camera. It still fails tone_color because that gate is highlight loss <= 0, the shadow SNR stayed at 22.5 dB known and 24.7 dB blind against the camera's 26.2, and the bright-side edge overshoot grew to 0.207 known (0.126 blind). The video fails provenance at 2.2 % known and 2.3 % blind against the 1 % gate (3.6 % on the seventh run); its flow and trajectory errors beat the camera, its flicker does not. The ninth run adds the local-range term (`RANGE_WEIGHT` in `harness/model/train.py`): every output value above the target's 3 by 3 maximum or below its minimum is charged once more, so a thin halo is no longer cheap.

The ninth run (`recovered_v9`) adds that term and stopped at step 6743 when the 75-minute budget ran out (the loader fell to about 12 images per second late in the run; its validation loss, 0.2825, includes the range term and is not comparable to the eighth's 0.1845). The term does what it was added for and costs what the halo prior was buying: on the still the edge overshoot falls from 0.207 to 0.098 known and from 0.126 to 0.044 blind, the highlight loss goes to 0, and the shadow SNR ties the camera known (25.6 against 26.2 dB) and beats it blind (27.9 dB); about half of the resolution lead is gone (mtf50 0.384 known and 0.284 blind against the camera's 0.220, texture 0.306 and 0.261 against 0.223, landmark error 0.29 against 0.45 px known), the banding rises to 0.0063 known against the camera's 0.0035 (0.0040 blind, a tie) and the colour error's maximum to 1.99 known (delta E 0.58 known, 0.43 blind). The still now fails at artifacts (the overshoot gate is 0.01) instead of tone_color; the video fails provenance at 1.68 % known and 2.47 % blind.

The tenth run (`recovered_v10`) keeps the ninth's loss and pair recipe and changes the targets: `--sources` adds the HDR+ merges and RawNIND's GT raws (DEGRADATION_PLAN.md step 4), 772 linear references among 6,341 training items drawn uniformly, so about one draw in eight has a linear, high-bit-depth target with measured noise instead of an ISP's output. It ran as two segments: the first died after step 4200 when a loader worker opened a 188-megapixel featured picture past Pillow's pixel cap (the sampler now drops such a file and did so twice in the second segment, that panorama and a truncated JPEG), and the second resumed from the step-4000 checkpoint with a fresh 75-minute budget and reached step 9000 (34 + 48 min on the RTX 5090 at 15 to 37 images per second; its validation loss, 0.2885, is on a draw that changed with the sources and again between the segments when the rest of RawNIND landed, so it is not comparable to the ninth's 0.2825). On the demo still it is sharper than the ninth (mtf50 0.429 known and 0.400 blind against the camera's 0.220, texture 0.346 and 0.296 against 0.223, landmark error 0.29 against 0.45 px), its shadow SNR ties the camera (26.97 and 26.84 against 26.17 dB) and the colour maximum falls (delta E 0.59 known and 0.55 blind, max 1.44 and 1.20 against the ninth's 1.99), but the highlight loss is back (0.163 known, 0.037 blind, against 0), the edge overshoot grew from 0.098 to 0.155 known (0.044 to 0.118 blind) and the banding to 0.0119 known (0.0084 blind) against the camera's 0.0035, so the still fails tone_color again; the video's undeclared invention falls to 1.02 % known and 1.06 % blind against the 1 % gate (1.68 % and 2.47 % for the ninth), the closest any run has come, and it fails provenance by that margin, with its shadow SNR 21.4 known and 23.6 blind against the camera's 26.7 dB. Where the linear references were meant to tell, the two real evaluations below, they did not.

Runs eleven to sixteen are recorded in [MODEL_PLAN.md](MODEL_PLAN.md) (sections 1, 15, 17 and 18): the 30M and 120M ladder (v14a and v14b), the 120M fifteenth run on every HDR+ train scene, sub-pixel registered real targets and the transfer term, and the sixteenth, queued behind it, with smooth resampling (`--resample smooth`: a binomial blur before each stride, bilinear upsampling) and two loss terms on the gates' own readings (`--overshoot-weight`, `--pyramid-weight`). Since the sixteenth, every held-out report reads noise as the residual no linear filter of the clean tile explains (`tilemetrics.residual_noise`), keeps its restored tiles (`outputs.npz`), and is judged a third way, against the oracle linear frontier at the model's own shadow SNR (`gates["frontier"]`, [harness/model/frontier.py](harness/model/frontier.py)), since no denoiser, the oracle included, keeps the noisy capture's mtf50 and texture. `model compare` puts every run's reports in one table per set.

`eval-raw` is the held-out real-capture evaluation (`harness/model/heldout.py`): 64 tiles of 256 px cut at clean scale 4 from `raw-pixls-us`, one CC0 raw per camera model (777 usable files) that no run trains on, rendered through the demo claim's `best_case_earth` camera with the sampler's exposure policy, restored known and blind, and scored with the twelve still metrics against the clean tile, input and model alike, with a per-tile verdict at each gate's tie margin. The set is a function of source, seed, count and tile size and is cached under `derived/heldout/`, so every checkpoint sees the same tiles; the report goes to `heldout_known/` or `heldout_blind/` beside the checkpoint. The references carry each camera's own noise and demosaic (scale 4 averages most of it away), which bounds the score. On it the eighth run sharpens (mtf50 0.254 to 0.365, texture 0.237 to 0.585, beating the capture on 47 of 64 tiles) but adds noise in the shadows (shadow SNR 14.8 to 12.9 dB, losing on 19 tiles), puts a halo on edges that had none (overshoot 0 to 0.029, losing on 45 tiles) and drops a dynamic-range step on 28 tiles; colour, banding, periodicity, landmark and shading tie: 3 beats, 5 ties, 4 losses on the medians. The ninth run's range term removes exactly those three: overshoot 0.0003 known and 0 blind, shadow SNR 14.6 known and 14.9 blind (ties), the dynamic-range step kept, at about half the sharpening gain (mtf50 0.307 known and 0.284 blind, texture 0.358 and 0.302): 3 beats, 8 ties, 1 loss (banding, 0.0095 to 0.013) known and 2, 8, 2 blind (banding and the colour maximum). Known and blind differ by less than the tie margins on every metric for both runs: on real captures the capture descriptor is carrying almost nothing. The tenth run is back at the eighth's tally, 3 beats, 5 ties, 4 losses known and blind alike: it sharpens more than the ninth (mtf50 0.317 known and 0.307 blind, texture 0.384 and 0.350, beating on 46 to 48 of 64 tiles) with the shadow SNR a tie (14.0 and 14.3 against 14.8 dB), but the halo is back (overshoot 0 to 0.023 known and 0.024 blind, losing on 42 and 41 tiles), a dynamic-range step is lost on 25 tiles, the banding rises to 0.0128 known and the colour maximum from 3.96 to 5.12.

`eval-raw --real --source <id>` is the second real evaluation, on the real capture/reference pairs that `data realpairs` builds (`harness/data/realpairs.py`): a burst's reference frame (`reference_frame.txt`) against its HDR+ merge, and each RawNIND noisy raw, and the second GT, against the first GT of the scene, decoded at the clean scale (2), aligned by phase correlation with a quadrant check, the reference brought to the capture's exposure by a per-channel gain fitted below clip, and the noise law (electrons at clip, read noise) fitted per tile from the residual against the reference. That noise law is the capture descriptor the model sees known; `--blind` withholds it. Input and model are scored against the aligned reference with the same twelve metrics and tie margins, and the report goes to `heldout_real_<source>_known/` or `_blind/` beside the checkpoint. The set is a function of source, seed, count, tile and scale, cached under `derived/realpairs/<source>/` with its rows as a label parquet; `--compare` puts the real rows' metric quantiles against the synthetic label rows' with a coverage per metric. Seed 0 gives 64 tiles of 256 px per source: the 10 HDR+ bursts of the val split, and 64 RawNIND pairs of 80 in 9 val and test scenes (3 rejected, the skull scene's captures being 4 to 7 px apart). The HDR+ pairs measure noise, colour and range and not sharpness (mtf50 saturates at 0.5 for frame and merge alike, and texture fidelity above 1 is the frame's noise counted as high-frequency energy); the RawNIND sweeps measure sharpness too (mtf50 0.05 to 0.5 across the tiles). On HDR+ the eighth run raises shadow SNR from 14.0 to 21.4 dB (beating on 40 of 64 tiles) and nothing else, losing texture (1.22 to 0.90), colour (ΔE 0.68 to 1.20), edges (overshoot 0.006 to 0.037) and landmarks (0.06 to 0.16 px): 1 beat, 4 ties, 7 losses on the medians; the ninth is the same picture with cleaner shadows (21.8 dB, 46 beats): 1, 5, 6 known and 1, 6, 5 blind. On RawNIND both runs raise shadow SNR by 7 to 8 dB (13.4 to 21.6 and 20.9 dB, beating on 43 of 64 tiles) and lower banding (0.021 to 0.015 and 0.014) while halving mtf50 (0.39 to 0.19 and 0.17, losing on 45 and 47 tiles) and cutting texture fidelity to a fifth (0.67 to 0.11 and 0.15): 2 beats, 7 ties, 3 losses known for both, 2, 5, 5 and 2, 6, 4 blind. The run that sharpens the raw.pixls.us tiles rendered through the demo camera blurs real noisy raws: on real noise it is a denoiser that pays in resolution, texture, colour and edges, known and blind alike. The tenth run, trained with the merges and the GT raws as about one target in eight, does not move that verdict: on HDR+ 1 beat, 4 ties, 7 losses known and 1, 7, 4 blind (shadow SNR 14.0 to 21.1 and 20.7 dB, texture 1.22 to 0.74 and 0.87, ΔE 0.68 to 1.30 and 1.10, overshoot 0.006 to 0.023 and 0.044, landmark error 0.06 to 0.16 and 0.12 px); on RawNIND 2, 6, 4 known and 2, 7, 3 blind (shadow SNR 13.4 to 20.2 and 19.2 dB, banding 0.021 to 0.017 and 0.018, mtf50 0.39 to 0.15 and 0.17, texture 0.67 to 0.18 and 0.26, periodicity 0.031 to 0.082 known, past the tie margin the ninth's 0.081 stayed inside). Known is worse than blind on both sets for the first time (texture 0.18 against 0.26 on RawNIND, 0.74 against 0.87 on HDR+): with the measured noise law in hand the model denoises harder. Against the synthetic label rows the real pairs put the sampler's optics, colour, periodicity and shading draws an order of magnitude above any real pair (mtf50 median 0.09 synthetic against 0.5 and 0.39 real, ΔE 9.8 against 0.7, shading 0.10 against 0.009) and its noise and dynamic range on top of the real ones (coverage 98 to 100 %).

```bash
python -m harness model train --out data/models/recovered_v9 --steps 9000 --minutes 75 --batch 16 --tile 256 --workers 12
python -m harness model train --out data/models/recovered_v10 --sources hf-kadid10k-images,hf-tid2013,polyhaven-hdris,wikimedia-featured-pictures,hdr-plus-burst,rawnind --steps 9000 --minutes 75 --batch 16 --tile 256 --workers 12   # the tenth run: the HDR+ merges and RawNIND GT raws as clean targets; the same command resumes from last.pt
python -m harness model eval-demo --ckpt data/models/recovered_v10/best.pt --out demo_out_model
python -m harness model eval-demo --ckpt data/models/recovered_v10/best.pt --out demo_out_model_blind --blind
python -m harness model eval-raw --ckpt data/models/recovered_v10/best.pt            # held-out real raws, capture known
python -m harness model eval-raw --ckpt data/models/recovered_v10/best.pt --blind    # the same tiles, capture withheld
python -m harness data realpairs --sources hdr-plus-burst rawnind --n 64 --compare   # real capture/reference sets (cached under derived/realpairs/), real-against-synthetic metric quantiles
python -m harness model eval-raw --ckpt data/models/recovered_v10/best.pt --real --source hdr-plus-burst   # the real pairs, the measured noise law as the capture; --blind withholds it
python -m harness model apply --ckpt data/models/recovered_v10/best.pt --input frames.npy --out restored --white-level 1.0 --capture capture.json
python -m harness model train --out data/models/recovered_v16 --sources hf-kadid10k-images,hf-tid2013,polyhaven-hdris,wikimedia-featured-pictures,hdr-plus-burst,rawnind,si-hdr,hf-pixelshift200 --real rawnind,hdr-plus-burst,si-hdr --p-real 0.5 --weights rawnind=6,hdr-plus-burst=6,polyhaven-hdris=2,si-hdr=4,hf-pixelshift200=16 --range-weight 3 --flat-weight 1 --texture-weight 1 --transfer-weight 0.1 --shift-real --noise-map --target-audit --resample smooth --overshoot-weight 1.5 --pyramid-weight 1 --warmup 1000 --lr-floor 0.003 --base 64 --mults 1,2,4,8 --blocks 2,3,6,8 --mid-blocks 12 --steps 120000 --minutes 480 --batch 16 --tile 256 --workers 12 --subpix-real --compile   # the sixteenth run (MODEL_PLAN.md section 18)
python -m harness model compare --out compare.md   # every run's held-out reports: verdicts, output / matched frontier
```

The model commands take explicit paths: with `HARNESS_DATA_ROOT` set, put the checkpoint directory under that
root (`D:\lens-harness-data\models\recovered_v7` on the GPU box) so checkpoints live beside the data they were
trained on and not in the repo.

`eval-demo` builds the demo worlds, runs the camera-sim frames through the model at the camera's clip level and
judges the result through the gates like any candidate, so the report says on which gates the model beats the
best-case camera and whether it stays under the provenance limit. `train` resumes from `last.pt`, logs to
`train_log.jsonl` (loss, images per second, per-metric medians of input and output against the clean target on
the held-out restore set every checkpoint), and keeps `best.pt` by validation loss. Held-out items are a hash
split of the clean sources by scene (one in sixteen) plus the scenes a source's `clean.holdout` names (RawNIND's
22 test-reserve scenes), which are split `test` and never trained or validated on; `eval-raw` scores real raws
no run trains on.

## Data: no invented constants

Every number the harness used to make up (sensor pitch, full well, read noise, display size,
the ppd threshold, the ΔE / banding gates, viewing distance) now comes from a sourced row:

- `harness/data/reference/*.csv` — hand-verified rows (sensors, lenses, displays, vision
  thresholds). **No source, no row**: every row carries a `source_url` and a `quote`; the
  reader refuses a row without one.
- `harness/data/sources.json` — the manifest: 124 sources (spec PDFs, vendor pages, Zenodo
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
python -m harness data collect --ids wikimedia-featured-pictures --param limit=5000 --max-bytes 40e9 --jobs 4   # 4,027 clean originals, licence per file, 38 GB
python -m harness data collect --ids raw-pixls-us --max-bytes 30e9                # one raw per camera model, 25 GB, held out for eval-raw
python -m harness data label --sources polyhaven-hdris hdr-plus-burst wikimedia-featured-pictures hf-kadid10k-images hf-tid2013 --n 2000000 --jobs 16   # offline metric labels, resumable per chunk
python -m harness data plan --out DATA_PLAN.md       # refresh the plan table
```

The collector is parallel, resumable (HTTP Range + append-only `data/ledger.jsonl`; the append holds a
cross-process lock on `ledger.lock` since two chains appending at once tore one record on 2026-10-28, and a load
skips a torn line and finds its file on disk), verifies sha256/md5 when the manifest states one, throttles per
host and never retries a host that refused it. Set `HARNESS_DATA_ROOT` to put the large tiers on a real disk; gated
Hugging Face sets read `HF_TOKEN`. `claims/example_sourced.json` is a claim whose camera,
display and gates all resolve to derived tables. [DATA_PLAN.md](DATA_PLAN.md) lists every
source, its status and the remaining gaps.

The recovered layer's clean sources (`DEFAULT_SOURCES` in `harness/model/train.py`) are the KADID-10k and TID2013
references, the Poly Haven HDRIs and the Wikimedia featured pictures: 4,027 expert-reviewed camera originals
(38 GB; the API walk listed 5,000 and skipped 672 for licence, type or size), the licence read per file and kept
as the first path component. Each source's `clean` block in `sources.json` says how the labeller and the sampler
read it: `scale` (the featured pictures at 2, and at `low_quality_scale` 4 when the JPEG quality is low;
raw.pixls.us at 4), `exclude` (the `*.tif` scan containers of raw.pixls.us), `include` (only RawNIND's GT raws
and the HDR+ merges are clean items; the noisy raws and the burst frames are the inputs of the real pairs), `scene`
(a regex whose first group is the scene key, so a RawNIND scene's GT and ISO sweep, and a burst's frames and
merges, split together) and `holdout` (scene keys split `test`). An item whose file the decoder rejects is dropped
from the sampler with its skip reason and named in the run's log by a warning (the 188-megapixel featured picture
past Pillow's pixel cap that ended the tenth run's first segment at step 4200 from a loader worker): one file
never ends a run.
raw.pixls.us (777 raws, 25 GB, one per camera model) is held out for `eval-raw` and never trained on; the
held-out split of the training sources is by scene, so a Poly Haven HDRI's resolutions and a burst's frames fall
on one side of the hash.

## Scope boundary

The harness scores candidates against **synthetic ground truth it generated**. It never
estimates flow, geometry or lighting from arbitrary footage: reconstructing reality from
a photograph is the information-bounded regime (SPEC §2) that no generator can win by
itself, and nothing here pretends otherwise.

## Layout

```
SPEC.md                 the contract (claim schema, regimes, baseline, metrics, gates, report)
DATA_PLAN.md            every source, tier, licence, status; gaps and the scaling plan
DEGRADATION_PLAN.md     degradation-as-labelling: where every parameter comes from, budgets, the real-pair gap
LANDSCAPE.md            what the field has, what the measurements say, the recovered layer's verdicts per run
pyproject.toml          kolm-pixel: core deps and the data / model / measure / video / dev extras
claims/                 example claims (video, still)
harness/scene.py        procedural ground truth; analytic flow and landmarks
harness/camera.py       best-case camera: Airy PSF, shot/read noise, full well, lens shading, Bayer, motion blur
harness/display.py      ppd, mastering rule, encode round-trip
harness/metrics/        resolution, tone, color, artifacts, geometry, temporal, natural (reference-based)
harness/gate.py         ordered checks, stop on first fail
harness/report.py       report.json + report.md
harness/preference.py   blinded manifests, Bradley–Terry fit
harness/evaluate.py     world build + candidate evaluation
harness/cli.py          demo | run | render | preference | data | model | photo | capture-eval
harness/photo.py        the product: a raw or photo in, the restored 16-bit TIFF and JPEG out
harness/tiff.py         16-bit and 32-bit float RGB TIFF writer
harness/data/           manifest, parallel resumable collector, expanders (HF, Zenodo, Wikimedia Commons, ...), derivers, gate fitting, labeller
harness/data/reference/ sourced CSVs: sensors, lenses, displays, vision thresholds
harness/data/catalog/   122 human-rated image/video datasets and what each can calibrate
harness/model/          the recovered layer: on-the-fly pairs from the sampler, conditioned UNet, trainer, inference, held-out raw evaluation, release export, model card, capture-descriptor evaluation
releases/               each release's model card (its weights live under $HARNESS_DATA_ROOT/release/<name>/)
$HARNESS_DATA_ROOT/     raw/<source>/..., ledger.jsonl, derived/*.json, derived/heldout/, derived/realpairs/<source>/, labels/, models/<run>/, release/<name>/ (outside the repo)
tests/                  physics sanity, metric sensitivity, pipeline
```
