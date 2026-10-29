# Model plan: the data, losses and architecture a model needs to beat the camera

Status 2026-10-29. This is a plan, not a record: nothing in it is built unless it says so. It draws on two kinds of evidence.

- **Six research passes run today:** data impact, dataset census, faithful architecture and loss, burst and video, the rendered layer, and evaluation with baselines. They read the web, so a paper's number is on that paper's benchmark, not on our gates. Figures marked (unverified) were not read at the primary source, and (est) marks an estimate.
- **Measurements on this machine:** the loader, the decoded store, the corpus, the gates and the thirteenth run.

The companion document is [DEGRADATION_PLAN.md](DEGRADATION_PLAN.md), which covers the sampler. The papers' numbers rank the levers against each other; they do not predict our gates.

## 0. The short version

1. **The first blocker is the claim's definition, not the model.** On the real pairs, two of the eight gates cannot be passed by any model (section 2.1).
   - `mtf50_cpp`: the noisy capture already sits at the metric's Nyquist ceiling (0.5).
   - `texture_hf_fidelity`: it rewards gain above the truth's, so sharpening passes and faithful restoration cannot.

   Fixing the definition is the user's call, and it comes before any amount of data or size.
2. **Then information.** A single frame through a network that minimises error ties the capture's coherent detail at best. Only more information beats it:
   - bursts: +1.5 to 2.5 dB on real data;
   - the raw mosaic: +2.9 to 4.5 dB over processed input.

   This is where "beats the best-case camera" comes from. It is not parameter count.
3. **Then the loss.** The three standard-deviation terms of the eleventh to thirteenth runs look only at output statistics. They cannot tell a correct edge from a halo, or coherent texture from invented texture. That is the trade-off the twelfth run measured. Replace them with residual-based and phase-aware terms (section 8).
4. **Then data quality before data volume.**
   - JPEG references below about q90 and sharpened references teach their artifacts: q75 costs 0.54 dB, q50 about 2.9.
   - Two training sources carry research-only licences.
   - Each real evaluation set has 9 to 10 scenes; a claim needs at least 60.
   - The noise model lacks the three parts that measured best: heavy-tailed read noise, a dark-frame bank, and unprocessing into camera RGB.
5. **Size: billions, where billions help.**

   | Layer | Size | Role |
   |---|---|---|
   | Faithful (recovered) | 30–120M, chosen by a measured ladder | single-frame restoration |
   | Burst/video | about 150M | the information lever |
   | Rendered | 6B (Z-Image) or 20B (Qwen-Image-Edit-2511), both Apache and already on this machine (section 16) | declared enhancement, masked and labelled |

   That is about 6.3B parameters in the product, or about 20.3B with the larger rendered base; nearly all of it sits in the declared layer. No faithful restorer in the literature gains from going past about 100M at our data scale. A 0.5–1.1B faithful network would see under a tenth of the samples published recipes use (section 3).
6. **Data to add.** The first seven sources in section 6 total about 0.94 TB and fit in D:'s 1.33 TB free, but only once we decide whether the grown corpus is also decoded into the store (section 13). They bring real scenes for evaluation, bursts, demosaic ground truth without demosaicing, tripod-averaged ground truth, HDR scenes, and roughly 8× the clean content.
7. **The ComfyUI install on this machine** already holds about 175 GB of Apache weights (and 21 GB of FLUX under a non-commercial licence) that the rendered layer and the generative baselines need, and the tools that train LoRAs on them. It is not a source of training images: generated images have no capture physics (section 16).

## 1. Where the model stands

The thirteenth run (`recovered_v13`, 8.2M parameters) is at step 24,000 of 30,000.

- **Real validation** (input → output): mtf50 0.50 → 0.38, texture 0.88 → 0.54, shadow SNR 13.7 → 20.9 dB, banding 0.023 → 0.013, edge overshoot 0.0049 → 0.0022, ΔE mean 0.78 → 0.52.
- **Synthetic validation loss:** 0.4515, against a best of 0.4511.

So it denoises well, keeps edges clean and improves colour. It loses about 40 % of the coherent high-frequency detail the capture holds. Every run since the ninth makes the same trade, with different weights on each side.

The furthest any run has reached is the twelfth: seven of eight gates on the demo still, failing at micro_detail. On the real pairs it beats on noise and banding and loses mtf50 and texture on every set. The thirteenth will be scored eight ways when it finishes (`.tmp/score_run.sh v13`).

## 2. Why it stalls, in order

### 2.1 Two real-pair gates cannot be passed by construction (found today, in the code)

**mtf50.** `tilemetrics.mtf50_cpp` returns 0.5 when the transfer never falls below one half. The noisy capture against its clean reference has a coherent transfer near 1 at every frequency, because noise averages out of the cross spectrum. So its median mtf50 is exactly 0.50, the ceiling.

`beats_camera` needs a candidate above 0.52 (tie band 0.02), which no output can reach. On real pairs this gate fails for every model, including a perfect one.

**texture.** `texture_hf_fidelity` is the same coherent ratio above the cutoff, registered with `higher_is_better=True` and no ceiling. The capture scores 0.88 (not 1, because of residual misregistration and demosaicing).

- A restorer that is optimal in mean squared error shrinks detail whose phase it cannot know, so it scores below the capture (Blau–Michaeli; v11 to v13 all do).
- A sharpener that multiplies coherent detail by 1.3 scores above it, and passes.

The gate therefore rewards the one behaviour the provenance and artifact gates exist to stop.

**On the demo still neither problem bites.** The simulated camera blurs, so its transfer is well below 1 and approaching the truth beats it. That is why the still is where the runs get furthest.

**Proposed fix (the claim's definition, so the user's call):**

1. **Score both transfer metrics by distance to the truth's**, so a value above the truth's counts as error just as a value below does:
   - `|1 − T|` for the texture ratio;
   - `|mtf50 − mtf50_truth|` for mtf50, or MTF50P together with overshoot, as ISO 12233 practice does.

   On the still this changes nothing for an honest model; on any set it closes the sharpening route.
2. **On real pairs, compare against two baselines.**
   - **The raw capture:** gate the transfer metrics as `ties_or_beats` within 1 JND, and require a beat on noise, banding, colour and dynamic range. That is "denoise without losing detail", which is the real achievement against a signal no camera ships.
   - **The camera's delivered image**, where one exists: the phone's `final.jpg` in the full HDR+ release (not in our lean pull: its `results_*` folders hold only `merged.dng` and `reference_frame.txt`), or a fixed reference ISP for RawNIND. Require a beat on every gate here. This is "the best-case camera", and the burst layer is the one expected to beat it.

**Done (2026-10-28), as proposed, with two deviations.**

1. **Distance to the truth.**
   - Each tile is also scored against itself (`tilemetrics.self_metrics`, the truth's own values).
   - `mtf50_cpp` and `texture_hf_fidelity` are judged by `|baseline − truth| − |candidate − truth|` (`MetricInfo.to_truth`), so detail beyond the reference counts as error.
   - A `beats_camera` check on a truth-bound metric passes on a tie when the baseline already sits at the truth (`at_truth`), and its detail says so. A gate with nothing left to win can now be passed, and only by matching the truth.
2. **Two baselines on real pairs**, judged on the medians (`heldout.CAPTURE_GATES`, `heldout.CAMERA_GATES`):
   - **against the capture:** shadow SNR and banding must beat it, everything else tie or beat;
   - **against the camera's photo:** beat it on every metric (mtf50 and texture by distance, tied at the truth), dynamic range tie or beat.
3. **The camera's photo.**
   - The HDR+ pull now keeps `final.jpg`.
   - `realpairs.match_camera` fits its rendering to the target on pixels blurred at 2 px: a per-channel isotonic tone curve, then a 3×3 colour matrix. A local luminance gain at σ 8 px, clamped to [0.25, 4], takes out its local tone mapping. What is compared is its detail, noise and artifacts, not its look.
   - Sets save the matched tiles, and `attach_cameras` fills them into sets cached before.
   - RawNIND has no camera photo, so its camera gates are absent, not passed.

**The deviations:**

- **Colour and dynamic range are tie-or-beat against the capture, not beat.** The captures' median delta E (0.68 HDR+, 0.74 RawNIND) is within the 0.5 tie margin of anything a candidate can reach, and their dynamic-range steps are at the tiles' ceiling. A required beat there would fail every model, the same trap as mtf50.
- **Highlight loss is out of both real gate sets.** It reads clean pixels above 1. A real target is clipped at 1, and tiles with clipped reference pixels are rejected, so the metric is always undefined and would hold every verdict at INCOMPLETE. The simulated gates keep it.

Tests: `tests/test_truthgates.py`.

**Still unfair to a denoiser, so a third gate set (10-29, section 18).** Tie-or-beat against the capture on mtf50 and texture still asks a denoiser to keep detail that the best linear filter, fitted on the truth itself, loses at the same noise. The frontier gates judge the output against that oracle at the output's own shadow SNR, and the capture gates stay as they are.

### 2.2 Information

A single-frame faithful restorer is bounded by what the frame holds. Published real-data figures:

- Merging a burst moves the bound: real BurstSR +1.5 to 2.5 dB over a single frame, with architecture choice under 1 dB of that, and alignment about 1.3 dB.
- Raw input before demosaicing moves it too: Zoom-to-Learn +4.5 dB over 8-bit input (confounded), and real raw +2.9 dB over synthetic raw.

Our recovered layer sees one demosaicked linear RGB frame, so demosaic errors are fixed before the network sees anything. Both levers are sections 8.2 and 9.

### 2.3 Loss

- **v11 range term** (3×3 local range excess) and **v12 flat term** (7×7 std excess on flat targets): they charge output statistics and never deficit. Where the model cannot tell the phase of fine detail, the cheapest output is a smooth one (v12: mtf50 0.337 → 0.240 on the still).
- **v13 texture term** (7×7 std deficit on textured targets): it rewards energy whatever its phase. The gate it is meant to serve counts only coherent detail, so the term pushes towards invented texture.

All three measure the output rather than the residual against the target. The replacements in section 8.3 measure the residual and its phase.

### 2.4 Data

Detail in sections 4 to 6:

- JPEG references from Commons at q75–q90 carry ringing and sharpening into the targets.
- KADID-10k and TID2013 references are in training under research-only terms.
- The HDR+ `merged.dng` targets cap the model at HDR+'s own quality and artifacts.
- Real pairs are aligned to the integer pixel.
- Each real evaluation set has 9 to 10 scenes.
- The noise model is missing heavy-tailed read noise, dark frames and camera-RGB unprocessing.

## 3. Size: the evidence and the three-layer system

### 3.1 What the literature shows for faithful restoration

- **No PSNR-track restorer of 2025–26 with hundreds of millions of parameters beats the 20–40M ones.** The NTIRE 2024 and 2026 fidelity podiums are 20–40M models or ensembles of them. The only entry above 100M was a 4B FLUX model in the perceptual track, at 25.6 dB.
- **Naive scaling loses.** Hi-IR: 15.7M → 57.6M lost 0.19 dB on Set5 and 0.57 dB on Urban100 at ×2. After architecture fixes (1×1 bottlenecks, dot-product attention), 39M beat 15M by 0.16–0.41 dB.
- **Depth saturates.** NAFNet on SIDD with 9, 18, 36 and 72 blocks: 39.78, 39.90, 39.96, 39.95 dB.
- **Narrower and deeper beats wider.** Restormer with channel width 48, 64, 80 scored 29.71, 29.63, 29.56.
- **Data, not parameters, is the binding constraint at our scale.**
  - Klug–Heckel: the power-law exponent drops from about 0.005 to 0.002 beyond about 6k images, and at 10k–30k images larger U-Nets gained nothing.
  - DiverSeg: going from 3k to 85k–260k filtered images bought +0.4 to +0.6 dB; 1.2M unfiltered images did worse than 259k filtered.
- **Compute.** 12 h on our GPU lets a 1.1B net see about 0.44M tiles, a 496M about 1.0M, and a 26M about 4.1M. Published recipes use 9.6M (X-Restormer) to 25.6M (NAFNet). AdamW state alone is 17.6 GB at 1.1B.
- **Our loader.** It renders 35–40 pairs/s, so 12 h holds at most about 1.6M distinct pairs until the degradation moves to the GPU (section 8.5).

### 3.2 Where billions do help

Billions help in generation: inventing plausible detail the input does not hold. That is the rendered layer, which must be declared and masked (section 10).

### 3.3 The system

| Layer | What it does | Size | Base and licence | What it may claim |
|---|---|---|---|---|
| Recovered (faithful) | single frame, raw or linear in, linear out, per-pixel variance out | 30–120M, by the ladder in 3.4 | NAFNet / X-Restormer blocks (MIT) with our FiLM conditioning | every gate; no invention (provenance ≤ 1 %) |
| Burst and video | 2–15 frames or a stream, aligned and merged | about 150M (est) | SEA-RAFT (BSD-3) flow, BasicVSR++-style recurrence (Apache) | every gate, plus the temporal gates and a multi-frame gain gate |
| Rendered (declared) | detail above the recovered layer's cutoff, inside a mask only | 6–20B | Z-Image 6B or Qwen-Image-Edit-2511 20B (Apache, on disk); video Wan2.2 A14B (Apache, on disk) or Wan2.1-1.3B | nothing hidden: area, bands and precision declared, C2PA region |

That totals about 6.3B parameters with Z-Image, or about 20.3B with Qwen-Image-Edit. A user gets a faithful image, and optionally a declared enhancement whose every rendered pixel is labelled.

### 3.4 The ladder

The faithful size is measured, not assumed:

- three points, about 8M, 30M and 120M;
- the same data, loss and wall time for each (245 min, the thirteenth run's budget);
- v13 is the 8M point.

**Stop rule:** if 120M's real and synthetic validation do not improve on 30M's by more than the spread between two seeds of one size, the faithful layer stays at 30M and further capacity goes to the burst layer.

This replaces the single 12 h run of shape J (496M) planned earlier, which the evidence above argues against.

## 4. Fixes to the data we already have

These come before any download, ranked by measured impact and cost.

| # | Fix | Why (evidence) | Cost |
|---|---|---|---|
| 1 | **Estimate each JPEG reference's quality from its quantisation tables.** q ≥ 90 at clean scale 2; 75–90 at scale 4; below 75 out of training (evaluation or rendered-layer data only). Add the HQ-50K / LucidFlux filters: short side ≥ 1024, Laplacian variance 150–8000, under 50 % flat patches. | DiverSeg: references at q95 cost 0.12–0.23 dB, q85 0.21–0.30, q75 0.54–0.56, q50 2.2–2.9. Sharpened references teach halos (Real-ESRGAN+). The model learns the target's artifacts. | a label column plus a sampler filter, no new data |
| 2 | **KADID-10k and TID2013 to evaluation only.** | 106 references (81 + 25), under 2 % of the clean set. Research-only terms; KADID references reportedly Pixabay-licensed, TID references mostly Kodak (both unverified). | drop two source ids from the run lines |
| 3 | **Scene split everywhere, plus perceptual-hash dedup against every evaluation set:** demo, raw.pixls.us, real val/test, SIDD val, RealSR and the rest. | Near-duplicate leakage reaches about 10 % in public sets. Commons Featured and Quality overlap; burst frames are near-duplicates by nature. | a hash pass over the corpus |
| 4 | **Real-pair hygiene:** sub-pixel registration (today: integer phase correlation refined to ±REFINE px, checked per quadrant, in `realpairs.py`); a per-pair 3×3 colour matrix; reject NCC below 0.9; mask clipped and misaligned pixels; drop the worst 2–3 % by loss. | BurstSR and RawNIND recipes. Misregistration lowers the coherent ratio at high frequency for both capture and model: it is part of why the capture scores 0.88, not 1. | `realpairs.py` and the real store |
| 5 | **Retire HDR+ `merged.dng` as a target once real averaged ground truth is in** (section 6: SIDD, SNIC). Until then, tag rows by target type. | A merged target caps the model at HDR+'s quality, artifacts included. Averaged static bursts are the published real ground truth. | a target-type field per row |
| 6 | **`wikimedia-quality-images` gets a clean block** (scale 2, low-quality scale 4). | It was read at scale 1. | done, uncommitted |
| 7 | **Real evaluation sets to ≥ 60 held-out scenes each** (today HDR+ 9–10, RawNIND 9). | Scene-level resampling: n_eff ≈ number of scenes. At about 97 % power per gate (what an all-gates claim needs), an effect of 0.5 SD needs about 59 scenes and 0.3 SD about 164. | HDR+ full release, SNIC, SIDD validation (section 6) |

## 5. What the degradation still lacks

[harness/data/degrade.py](harness/data/degrade.py) already has:

- shot noise;
- measured read-noise ladders;
- row noise;
- ADC quantisation;
- an RGGB mosaic with bilinear demosaic;
- posterisation;
- JPEG fitted to TID2013;
- a white-balance cast from HDR+ as-shot neutrals.

Missing, in order of measured impact:

| # | Addition | Evidence | Needs |
|---|---|---|---|
| 1 | **Dark-frame bank:** about 10 real lens-cap frames per ISO, added as sampled real dark noise (hot pixels, fixed pattern, dark shading). | SFRN / PMN: +2.2 to 2.9 dB over pixelwise noise models; dark shading alone +0.8 to 2.4 dB (extreme low light). | Frames from the user's own camera, which is a manual step. Dark frames inside the downloaded raw sets where they exist. |
| 2 | **Heavy-tailed read noise** (Tukey-lambda), fitted per camera from the bank. | ELD on SID: Gaussian 32.99, Poisson-Gaussian 33.93, true Poisson 35.36, + Tukey-lambda 37.08, + row 37.38, + quantisation 37.57, against real pairs 37.32. We have the row and quantisation terms; Tukey is the largest missing step. | the bank (row 1) |
| 3 | **Unprocessing into camera RGB.** Sample a camera colour matrix and white-balance gains, invert them, add noise in camera space, mosaic, then re-process. Today noise is added in linear sRGB primaries. | Brooks et al. on DND: removing unprocessing cost 1.33 dB, and a 4× larger model without it recovered only 0.33 dB. Additive Gaussian alone cost 1.70 dB. | Colour matrices from the training raws' DNG metadata (HDR+, RawNIND, SIDD), never from raw.pixls.us, which is held out |
| 4 | **Per-pixel noise variance as an input channel**, σ²(x) = k·x + σr² from the sampled parameters at training, and from metadata or an estimator at inference. | Brooks: +0.37 dB raw, +0.54 dB sRGB over blind. A 31-dimensional global FiLM vector cannot express signal-dependent per-pixel variance. | the sampler already knows k and σr |
| 5 | **Raw output of the sampler:** packed RGGB plus the noise map, for the raw path (section 8.2). | Zoom-to-Learn and RawNIND: linear MS-SSIM 0.956 against 0.928 developed. | the mosaic stage exists; emit it before demosaicing |
| 6 | **Augmentation:** flips and 90° rotations that keep Bayer phase, RGB channel permutation, CutBlur, progressive patches 128 → 256 → 384. No Cutout, no mixup. | RGB permutation +0.09/+0.13; CutBlur +0.05/+0.23; all together +0.09/+0.27; progressive patches +0.07; Cutout −0.1. | sampler and loader |

The measured read-noise ladders stay: measured beats the Brooks log-normal prior where we have the camera. The coverage diagnostic (`realpairs` against the label rows) is how to check the sampler spans the real pairs.

## 6. Data to add, ranked

Sizes are lean pulls (the census pass, corrected where it found errors). Licences are per source in [harness/data/sources.json](harness/data/sources.json) when added; ShareAlike rows carry a tag so they can be dropped later.

**Pulls, 2026-10-28.** The user gave one go-ahead for the whole of this section. Rows 1, 2, 3 and 5 are downloading to D:.

- **Collector floor.** The collector refuses a file that would take the drive below 150 GB free (`HARNESS_COLLECT_RESERVE_GB`).
- **Decoded store floor.** The decoded store refuses a write below 300 GB free (`HARNESS_DECODED_RESERVE_GB`), so the cache gives way before the raw data does.
- **Row 1, HDR+, in two stages.**
  - Stage 1 is every burst's merge, camera JPEG, reference-frame index and colour/timing text: 92 GB.
  - Stage 2 is each burst's own reference frame and its lens-shading map: about 73 GB. It is picked by an exact names file, built from stage 1's indexes (`names_file` on `gcs_bucket`).
  - The two neighbour frames wait for section 9.
- **Row 4, SNIC, is manual.** The Dataverse answers 403 to every datafile request the collector makes. A person fetches the files in a browser into `raw/snic/`, and they are checked against the record's MD5s.
- **Row 6, UHD-IQA.** Its host reset every connection on 2026-10-28. Retry it with the collector as it is.

| # | Source | Size | Licence | Role | Why |
|---|---|---|---|---|---|
| 1 | HDR+ full release: reference frame (plus 2 neighbours later), `merged.dng` and `final.jpg` per burst | 165 GB now (92 + 73); 350–400 GB with the neighbours | CC BY-SA 4.0 | real scenes for evaluation (3,640 bursts), bursts for section 9, the phone's own output as the best-case camera baseline | the single biggest step for the claim: scenes, bursts and a real baseline in one |
| 2 | PixelShift200 (merged RGGB `.mat`, camera ARWs, previews) | about 75 GB as listed | CC BY 4.0 | full-colour ground truth with no demosaicing, for the raw path | demosaic training +1.16 dB over DIV2K-derived targets (secondary) |
| 3 | Commons Featured (the remaining 15,955 as listed) plus Quality Images ≥ 8 MP (to ~40k files) | 164.5 + ~250 GB | CC BY / BY-SA / PD per file; attribution manifest | clean content: about 5k → about 45k images, after the section 4 filters | diversity over count (DiverSeg); the step that makes a 30–120M net worth having |
| 4 | SNIC subset (30 scenes; iPhone 11 Pro, RX100 IV, A7R III; ISO 100–12800, dark and grey frames); manual (403) | ~150 GB | MIT | tripod-averaged real ground truth on large sensors, plus dark frames for the bank | the only large permissive averaged-ground-truth set beyond SIDD |
| 5 | SI-HDR (raw stacks, references, inputs; not the other methods' reconstructions) | 24.5 GB | CC BY 4.0 | 181 HDR scenes, raw plus reference: highlights and dynamic range | the tone gates |
| 6 | UHD-IQA | 10.7 GB | CC0 | 6,073 images at 3840 px: clean content | permissive and large-format |
| 7 | SIDD-CC DNG bursts (10 of 115 instances on disk, 22 of 234.5 GB) | up to 235 GB; a subset is enough | MIT per the official page; the mirror says "sidd-terms-of-use" (to confirm) | 10 static indoor scenes × 5 phones × 150 frames: frame-averaged ground truth, N2N pairs, noise fitting. A noise and burst source, not a content source. | real averaged ground truth; split val/test by scene number |
| 8 | REDS train_orig (120 fps) | 180 GB | CC BY 4.0 | video, temporal training | the video layer |
| 9 | Tears of Steel footage and linear EXR, every 10th frame | cap 200 GB | CC BY 3.0 | linear video with real motion | linear ground truth for video |
| 10 | Hypersim final HDF5 | ~300 GB | CC BY-SA 3.0 | exact linear ground truth with depth and motion | synthetic bursts with known flow |
| 11 | Web pool (CommonCatalog CC-BY, PD12M, Megalith ≥ 8 MP), deduplicated | cap 300 GB | per row | content scale-up after 3 | only once 3 is exhausted |

**Do not train on:**

- **Held without a data licence:** SID, DRV/SMID, RawHDR, BurstSR, RealBSR.
- **Evaluation only:** LSDIR, DIV2K/DIV8K, Flickr2K, FiveK, RAISE, ELD, DND, PolyU.
- **Banned for ML scraping:** Pexels, Pixabay, Unsplash Full.
- **Mirrors whose licence tag differs from the origin's** (e.g. LSDIR on HF tagged Apache; origin academic): follow the origin.

**Netflix Open Content is not uniformly CC BY.** El Fuente, Meridian and Sol Levante are BY-NC-ND. Sparks needs its bundled licence read before any pull.

## 7. More labels from the same data

The user asked whether the same data can be labelled several ways, and whether video's 30–60 fps can be used. Yes, with two cautions.

**Many degradations per clean tile.** The sampler already draws `--per-tile` parameter vectors per crop, and a row reproduces its pair bit for bit, so K draws give K distinct pairs at no storage cost.

- This multiplies degradation diversity, which the noise results reward.
- It does not multiply content diversity, which the size results say is the binding constraint.
- With the degradation on the GPU (section 8.5) it is free: draw fresh parameters every step and never store pairs at all.

**Several targets per scene.** One scene can have:

- a linear reference (Poly Haven, SI-HDR);
- a raw ground truth (RawNIND);
- a burst merge (HDR+);
- a frame average (SIDD, SNIC);
- a pixel-shift capture (PixelShift200).

Label each row with its target type. Train the faithful layer only on targets at least as clean as the claim needs. Use the rest for pretraining or evaluation.

**Video frames: count per-shot content, not per-frame.**

- **As content,** consecutive frames are near-duplicates: about one frame per shot or per second adds diversity, and the rest is leakage across splits.
- **As supervision,** every frame counts:
  - static-camera clips give frame-averaged ground truth (the SIDD recipe);
  - moving clips give burst training samples with real motion;
  - flow-warped neighbours give MF2F self-supervised targets with occlusion masks, within about 0.35–0.7 dB of supervised training.

  Split video by clip, never by frame.

**Noise2Noise from bursts, raw domain only.** On SIDD raw, N2N matched supervised training (50.62 against 50.60 dB). In sRGB the correlated noise breaks it: neighbour methods score 28–34 dB against 40 supervised. So any raw burst, including the user's own camera's, becomes real training pairs without ground truth. Fixed-pattern noise survives averaging and needs the dark-frame bank to remove.

**Other cheap gains:**

| Gain | Measured |
|---|---|
| ×8 self-ensemble at inference | +0.05–0.22 dB |
| Loss-based pruning (keep the top half by loss minus the hardest 5 %) | +0.12 dB |
| EMA 0.9999 on long runs (we use 0.999) | +0.46 dB on GoPro |
| GainTuning test-time adaptation, out-of-distribution noise | +2.7–6.1 dB |
| GainTuning test-time adaptation, out-of-distribution content | +0.4–1.3 dB |

**Two traps.**

- **No learned image-quality score as a loss or a filter on the faithful layer.** CLIP-IQA rises while MANIQA and LPIPS fall: the model learns the metric, not the image.
- **No self-training on the model's own outputs.** Iterated, it collapses.

## 8. The recovered (faithful) layer

### 8.1 Architecture

Keep the U shape, the residual output, identity at initialisation and the 31-dimensional FiLM descriptor at every level. Change the blocks to NAFNet / X-Restormer style (MIT):

- simple gate;
- 1×1 → 3×3 → 1×1 bottleneck convolutions;
- channel attention at full resolution;
- dot-product (transposed) attention at the two lowest levels;
- deeper, not wider.

Parameters sit at the bottom levels, where they cost the least FLOPs. `net.CondUNet` already scales by width, levels and depth (`--base`, `--mults`, `--blocks`, `--mid-blocks`, `--grad-ckpt`), so the ladder can start on the current blocks and the block change follows as its own A/B.

### 8.2 Input

- **Today:** demosaicked linear RGB in `log1p(4096x)`. In the log domain shot-noise standard deviation goes as 1/√x, so an L1 loss there over-weights the shadows.
- **Add** the σ²(x) channel (section 5, row 4).
- **Ablate** a variance-stabilised domain (k-sigma or the generalised Anscombe transform): PMRID 39.76 against 39.65 for a concatenated map and 39.57 blind.
- **Then a raw path:** packed RGGB (4 channels at half resolution) plus the noise map, for every camera whose raw we can read (the decoders PR 15 added). The network denoises before demosaicing, which is where the published burst and raw winners do it.

### 8.3 Loss

Replace the range, flat and texture terms with:

| Term | Weight / setting | Evidence |
|---|---|---|
| Charbonnier (or L1) on the target | base | L1 34.42 dB against L2 33.18 (Zhao) |
| + MS-SSIM | α = 0.84 | 34.61 dB |
| + LDL artifact map | 7×7 local variance of the residual, scaled var^(1/5), penalised only where the current model is worse than its EMA (Apache) | RRDB+LDL against ESRGAN on DIV2K: 28.82 against 28.18 dB. It charges the residual, so a correct edge costs nothing and a halo costs its energy. |
| + high-band L1 on the residual's wavelet (SWT, one level) or FFT coefficients | about 0.05 | the NTIRE 2026 winner used this. It is phase-aware: detail in the wrong place costs, detail in the right place pays, which is what the texture gate counts. |
| + optional ΔE / Lab term | for the colour gate | |
| On real pairs: minimum over ±1 px shifts, or CoBi | | misregistration must not teach blur |

Avoid Gram and contextual losses, which match statistics, not alignment, and avoid any learned-IQA loss. The texture metric on the gates becomes the per-band coherent ratio, plus invention per band (section 2.1).

**Measured on v13 (2026-10-28).** Two of these rows were tried on the thirteenth run's checkpoint before v15.

- **MS-SSIM fails as a sharpening lever.** On the display rendering the fine bands sit far under SSIM's C2 constant, so the term barely sees them. No weight moved v13's best fine-band gain past 0.9 (0.7–0.9 across weights). The row is dropped; `--ssim-weight` stays at 0.
- **The coherent-transfer term replaces the high-band row.**
  - It is the texture gate as a loss: per radial band of 1/32 cycles per pixel, from 1/16 up, e = (⟨|Y|²⟩ − Re⟨P Y*⟩) / (⟨|Y|²⟩ + 10⁻³ × mean band power), under a Charbonnier with δ = 0.02. It reads the gate's texture value to 10⁻³.
  - It is two-sided: a band transferred above 1 costs as much as one below. Detail in the wrong place lowers Re⟨P Y*⟩, so it never pays.
  - v13 transfers 0.52–0.61 of the fine bands where the real capture carries 0.83–0.94.
  - At weight 1 its gradient is 2–7 times the base loss's.
  - RawNIND's best fine-band gain from a short fine-tune of v13: 0.8 at weight 0, then 0.9, 1.1, 1.2 and 1.4 at 0.03, 0.1, 0.3 and 1.
  - It is `--transfer-weight` (losses.transfer_loss).

### 8.4 Recipe

| Setting | Value |
|---|---|
| Batch | ≥ 32 (bf16 at 256 px fits at 30M; above that, gradient checkpointing or accumulation) |
| Optimiser | AdamW, lr 1e-3 → 3e-4 falling with size |
| Schedule | warmup 2–5 % of steps, cosine to 1e-6 |
| EMA | 0.999, 0.9999 on runs past about 100k steps |
| Patches | progressive, 128 → 256 → 384 |
| Seeds | ≥ 3 for any claimed number |
| Avoid | stochastic depth (−0.45 dB) |

Together these are +0.3 to 0.6 dB in the published ablations (est for us).

### 8.5 Loader: the degradation on the GPU

v13 is loader-bound.

- `render_pair` takes 0.17 s alone and 0.31 s with 12 workers busy, for 35–40 pairs/s.
- The GPU averages 15 % use, with bursts to 97 %.
- The costs are the FFT PSF (12 %), the sRGB transfer curves (16 %), the ISP (28 % cumulative) and the tile metrics (about 15 %). All of these are torch-native operations.
- JPEG on the GPU (torchvision nvjpeg) takes 0.69–0.85 ms per tile, against 1.1–1.9 ms in PIL. The bytes cross to the host between encode and decode.

Porting the sampler stage for stage to batched torch:

- removes the ceiling, which is what lets a 30–120M net see the published 10M+ samples;
- makes the fresh-draw-every-step labelling of section 7 free;
- is required for burst training, where N frames per sample multiply render cost by N.

The decoded store (EXR half ZIP, 3.4× smaller than float32, read in 0.086 s, about 170 GB for the current clean corpus) feeds it.

### 8.6 Inference

Today `apply` does EMA weights, 512 px tiles with 64 px raised-cosine overlap, and fp32.

Add:

- the ×8 self-ensemble (flips and rotations; Bayer phase preserved on raw);
- GainTuning for out-of-distribution noise;
- the per-pixel variance output.

**The fidelity dial.** Freirich et al. show that interpolating between the minimum-error estimate and a perceptual sample is optimal in the perception–distortion plane. The product exposes a single α, certified at the largest value whose invention stays at or below 1 % per band and whose overshoot stays at or below 0.005.

## 9. The burst and video layer

**Why.** It is the only lever that moves the bound on real data.

- Multi-frame gains +1.5 to 2.5 dB over single-frame on real BurstSR.
- Gains saturate near 8 frames: DBSR real with 1, 4, 8, 14 frames scored 46.41, 47.06, 47.52, 47.76 dB.
- A net trained only at N = 8 does worse at N = 2 than a single-frame net, so train with N varying from 2 to 15, weighted to 8.
- No published comparison exists of a learned merge against the HDR+ Wiener merge on real bursts against real ground truth. That is an open result we can fill.

**Sketch (about 150M, est; 0.15–0.4 s per 1080p frame in bf16, est).**

- **Inputs:** a reference frame and K−1 neighbours (packed Bayer or linear RGB), plus:
  - the noise map;
  - per-frame gain and exposure ratio;
  - the FiLM descriptor, extended with N, mode and timing.
- **Alignment:** SEA-RAFT flow (BSD-3) at half resolution, refined by flow-guided deformable alignment at quarter resolution.
- **Masks:** forward–backward consistency and photometric error against 2σ²(x).
- **Fusion:** per-pixel softmax weights. In video mode, a recurrent warped state: per FLOP, recurrence beats a sliding window (BasicVSR++ 7.3M scored 32.39 dB on REDS4, against VRT 35.6M at 32.19, and is 3× faster).
- **Trunk:** a U-Net with its parameters at 1/8 to 1/32 resolution.
- **Outputs:**
  - a linear image;
  - a log-variance;
  - the effective number of frames per pixel;
  - a prior-reliance map: |output − classical merge| / √variance.

**Losses:**

- Charbonnier in the log domain, plus FFT-magnitude L1;
- Gaussian negative log-likelihood on the variance head;
- a masked relative temporal loss: |W(y_{t−1}) − y_t| − |W(gt_{t−1}) − gt_t|;
- MF2F on real video;
- a high-frequency penalty where support is low.

**Data:**

- synthetic bursts from clean stills, with sampled homography plus local flow and per-frame noise, rendered on the GPU (section 8.5);
- HDR+ bursts (section 6, row 1);
- SIDD and SNIC static bursts with averaged ground truth;
- REDS and Tears of Steel for video;
- Hypersim for known flow.

**Gates added:**

- masked warping error (flow from a pinned SEA-RAFT);
- tOF;
- tLP;
- static-tile flicker;
- a multi-frame gain gate: the burst output must beat the single-frame net on the same reference frame.

Each temporal gate is paired with mtf50 and texture, because blur games warping error: VEnhancer has the best warping error and the worst PSNR.

**Licences.** HDR+'s IPOL code is AGPL, so we implement our own merge. timothybrooks/hdr-plus (Halide, MIT) runs as the classical baseline. Burstormer and BSRT are MIT; BasicVSR++ is Apache. DBSR, VRT and RVRT are non-commercial and stay out.

## 10. The rendered layer (declared)

**Bases (32 GB, clean licences).**

| Use | Base | Licence | Note |
|---|---|---|---|
| Stills, first | Z-Image 6B | Apache | on disk as Turbo (distilled); training needs the base release or a de-distillation adapter (check) |
| Stills, larger | Qwen-Image-Edit-2511 20B | Apache | on disk in fp8 (19 GB); takes a reference image natively; an 8-step Lightning LoRA (Apache) is on disk |
| Stills, smallest | FLUX.2 klein-base-4B | Apache | not on disk; the 9B is non-commercial; the FLUX 3 klein on disk is not Apache (section 16) |
| Video | Wan2.2 A14B low-noise expert, or Wan2.1-1.3B + a FlashVSR-style LoRA | Apache | Wan2.2 on disk in fp8 with 4-step lightx2v LoRAs; the 1.3B needs 18.3 GB at 768×1408 |
| Baseline now, off the shelf | SeedVR2-3B | Apache | over-generates on light degradations |

**Composition.** Y = R + M ⊙ w · (G − R), per half-octave band.

- R is the recovered output and G the generated one.
- The low bands always come from R. Only bands above the descriptor's cutoff are rendered.
- Outside the mask M, Y equals R exactly, so provenance there is R's.
- The knob w runs from 0 to 1 and is certified at 1.
- M comes from a band-gain detector on G against R. Its threshold is set by conformal risk control on about 1,000 ground-truth pairs per input domain, recalibrated on shift.
- M is exported as a C2PA region with a `digitalSourceType`.

**Limit.** No published method gives a per-image, per-pixel guarantee on invented detail: conformal masks are marginal and collapse under shift. We report declared area and declared precision, not a guarantee.

**Gate addition.** `evaluate.py` accepts a declared rendered map, with undeclared = flag & ~declared, but it puts no cap on how much may be declared. Add:

- a cap on rendered fraction, or at minimum report it alongside the precision of the declaration;
- a check of the detector by injecting known hallucinations and publishing its recall.

A huge M is a quality failure, not a pass.

**Training.** Condition the base on the VAE latent of R, plus the descriptor through an MLP into the timestep modulation, with a fixed empty text embedding.

- **Data:** ground truth → degrade → frozen recovered layer → R. Cache latents: about 1 MB per 1024², about 0.2 TB per 95k images.
- **Stage A:** a PMRF rectified-flow LoRA (rank 64–128), at 512 then 1024, with gradient checkpointing and 8-bit Adam.
- **Stage B:** one-step adversarial distillation (RpGAN, R1/R2, feature matching on frozen-base features).

That is about 150–300 hours on the 5090 for stills (est), and hundreds more for video.

**Licence traps.** FLUX.1-dev; FLUX.2 9B; FLUX 3 Image Klein (on disk; FLUX Kommunity License, section 16); OpenRAIL++ models (SD 2.x, SDXL, PixArt); SD3.5's revenue cap; LTX; HunyuanVideo; CogVideoX; S-Lab non-commercial models (InvSR, Upscale-A-Video); SUPIR; HYPIR; LucidFlux. Weights for ODTSR, TEASR, InfVSR and ReCaVSR are unconfirmed. SD-2.1 returned 404 on Hugging Face on 2026-10-28, so the default download for OSEDiff, PiSA-SR, SeeSR and DiffBIR is broken, and any mirror's provenance must be checked.

## 11. Evaluation

### 11.1 Statistics

- Resample by scene, not tile: 64 tiles from 9 scenes is n_eff ≈ 9.
- Paired cluster bootstrap with at least 10k resamples, reporting the one-sided 97.5 % lower bound per gate.
- The null hypothesis per gate is a margin of at most 1 JND, pre-registered.
- "All gates pass" is an intersection–union test, so it needs no multiplicity correction. But at 90 % power per gate, seven gates pass together only 48 % of the time, so each gate needs about 97 % power. That is where ≥ 60 scenes comes from (section 4, row 7). Per-gate claims use Holm.
- At least 3 seeds or checkpoints for our own model.
- The held-out sets are frozen before the run that claims on them.

### 11.2 Standard-conformant gates

| Gate | Standard / method |
|---|---|
| Resolution | ISO 12233:2023 e-SFR via the open SFRMAT5 Python port: MTF50P together with overshoot, on 4:1 edges at 3–7° |
| Micro-detail | ISO/TS 19567-2 dead leaves, the cross or intrinsic method with sub-pixel registration (0.15 px misregistration matters). The direct method is fooled by noise, denoising and sharpening; edge MTF and dead leaves diverge above about 0.3–0.4 c/px. |
| Dynamic range | ISO 15739:2023, stating the SNR threshold |
| Colour | ΔE2000 mean and p95 per patch |
| Optional | IEEE 1858-2023 CPIQ quality loss in JND, so "beats" means "by at least 1 JND" |

### 11.3 Baselines

Every baseline runs through the same gates; licences were read today.

| Kind | Models | Licences |
|---|---|---|
| Faithful | NAFNet-SIDD, Restormer, X-Restormer | MIT |
| | SwinIR, HAT / HAT-L, SCUNet | Apache |
| | Real-ESRGAN | BSD-3 |
| | BSRGAN | Apache |
| Classical ISP reference | timothybrooks/hdr-plus | MIT |
| Generative (they test the provenance gate) | OSEDiff, PiSA-SR, SeedVR2-3B | Apache |
| Generative, local through ComfyUI's API (section 16) | Qwen-Image-Edit-2511 + Lightning, Z-Image-Turbo, Wan2.2 | Apache |
| | FLUX 3 Image Klein, evaluation only | FLUX Kommunity |
| Evaluation only, outputs never in training | SUPIR, HYPIR, InvSR, DiT4SR | non-commercial |

Commercial tools are run by the user under their EULA, or the claim says they were not tested.

**Wired so far (10-29):** NAFNet-SIDD-width64, Restormer's real denoiser and SCUNet's real-noise PSNR model, through `python -m harness model baseline heldout|sidd` (baselines.py). Their weights come through the collector (sources.json `baseline-*`, hash-checked). Their architectures are vendored in `harness/model/baseline_archs/` from megvii-research/NAFNet 2b4af71e, swz30/Restormer 68dc6ac4 and cszn/SCUNet 52e440a8. Each file says what was changed and carries its licence text. The changes are imports, a print and demo code, plus NAFNet's LayerNorm2d and timm's DropPath re-expressed with the same forward. A CPU check on 16 evenly spaced SIDD validation blocks (10-30) gives 40.58 dB for NAFNet and 40.49 dB for Restormer, near their reported 40.30 and 40.02 on all 1,280. SCUNet's real model, which was not trained on SIDD, gives 35.25 dB. The input is 23.76 dB. They are scored blind on the held-out sets and on SIDD validation, and our releases are scored on SIDD too, in the gap after the burst run on v17 and before v18 (section 19).

**Public evaluation-only sets:**

- SIDD validation (1,280 blocks; current sRGB best about 40.5 dB);
- the SIDD benchmark (Kaggle);
- PolyU (non-commercial);
- RealSR test, DRealSR, RealLQ250;
- video: UDM10, REDS4, YouHQ40, VideoLQ.

Always report position on the perception–distortion plane, never one score. DND gets one submission, at the end.

### 11.4 Human study

- 2AFC with at least 20 screened observers and at least 50 scenes at 1:1, at a fixed viewing size.
- Honeypots; Bradley–Terry or JOD scaling.
- About 194 votes per comparison to detect 60 % against 50 %.
- A lab panel for micro-detail.

### 11.5 Claim wording

"On N held-out real scenes, … exceeds the reference camera and every evaluated open baseline [list] on gates A–G, each margin ≥ 1 JND at the one-sided 97.5 % scene-level paired-bootstrap lower bound; fidelity and perceptual scores both reported; invented detail < 1 % by a detector validated at Y % recall; commercial tools not tested."

Not "better than any lens on Earth": the still and video gates compare against a simulated camera.

## 12. What it ships as

The aim is "fully functioning and useful", and the gates are only half of that.

**Inputs:**

- camera raws (DNG and vendor formats, through the decoders PR 15 added);
- JPEG and HEIC;
- burst folders;
- video;
- a ComfyUI node wrapping `apply`, for people who already run those pipelines (section 16).

**Outputs:**

- linear EXR, 16-bit TIFF or JPEG;
- a per-pixel variance map;
- the effective-frames map (burst);
- a provenance map;
- a C2PA manifest, which declares the rendered region when the rendered layer is on.

**Controls:**

- the fidelity dial α (section 8.6);
- the rendered-layer knob w (section 10);
- descriptor targets, i.e. edits (`apply`'s `target_metrics`).

**Speed targets (est):**

- faithful layer at 30M: under 100 ms per 1080p frame in bf16 on the 5090;
- burst: 0.15–0.4 s per 1080p frame;
- rendered: seconds per still.

The speed table comes from `.tmp/bench_shapes.log` once it has run.

**What it will be best at:**

- faithful low-light and high-ISO restoration of raw captures, with a per-pixel uncertainty and a provenance record, which no open restorer ships;
- burst merging that beats the phone's own merge on real ground truth;
- optional, declared enhancement whose every rendered pixel is labelled.

The differentiator is the verified provenance, not raw PSNR.

## 13. Order, compute and disk

| Phase | Work | GPU | Needs the user |
|---|---|---|---|
| 0, now | Score v13. Decide the gate definitions (2.1). The data fixes in section 4 that need no download (rows 1–3, 6). Swap the loss (8.3). Add the σ² input. Port the degradation to the GPU (8.5). v14 = the ladder (3.4). | ladder: 2 × 245 min | gate decision; ladder go-ahead |
| 1 | Downloads 1–7 (about 0.94 TB). Real evaluation to ≥ 60 scenes. Dark-frame bank. Tukey read noise. Unprocessing. Real-pair hygiene. | none beyond scoring | downloads; lens-cap dark frames from their camera |
| 2 | Faithful layer at the ladder's size on the grown corpus, new loss and recipe, 3 seeds. The raw path. | 2–4 days (est) | |
| 3 | Burst and video layer. | 1–2 weeks (est) | video downloads (8–10) |
| 4 | Rendered layer, stills then video. | 150–300 h stills (est) on a 6B base, about 3–5× that on the 20B, more for video | the choice of base; no download for the on-disk bases, licences read 2026-10-28 |
| 5 | Full evaluation: baselines, public sets, human study, claim. | about 1 day for baselines | observers for the study |

**Disk.** On 10-28, D: had 1,332 GB free after the decoded store's 163 GB, and items 1–7 fit with about 0.39 TB to spare. On 10-29 at 05:30 it has 275 GB free of 1.9 TB:

- The decoded store holds 768 GB:

  | Source | Arrays | Size |
  |---|---|---|
  | Commons featured pictures | 10,198 | 555 GB |
  | HDR+ | 4,070 | 140 GB |
  | RawNIND | 1,116 | 34 GB |
  | Poly Haven | 1,994 | 30 GB |
  | PixelShift200 (float16) | 97 | 5.8 GB |
  | SI-HDR | 362 | 4.9 GB |

  RawNIND's and SI-HDR's counts include the hard-linked names described below.
- The store now keeps an array in float16 whenever every value is finite and within float16's range (`fits_half`), and in float32 otherwise. Arrays written before that change are float32.
  - `data decoded --compact` rewrites them in place. A file another process has mapped is left alone and counted busy.
  - PixelShift200's 97 arrays went from 12 GB to 5.8 GB.
  - The rest compact after v15's audit, beside its fill, and again once v15 has trained, since its sampler holds files open. Halving the float32 part would return up to about 380 GB (est).
- `data decoded --prune` removes the arrays no current item keys: the old names a key change leaves behind, and items dropped from a listing.
- The store's floor (`HARNESS_DECODED_RESERVE_GB`, default 300; v15's fill runs at 250) means it stops writing before the real store's 150 GB floor or the collector's.
- Video and synthetic sets (items 8–11, about 1 TB) need a second drive.

## 14. Risks

- **Gates that cannot pass** (2.1): fixed by definition or the claim fails whatever the model does.
- **Metric gaming:** every transfer gate is scored by distance to truth; every temporal gate is paired with detail; no learned IQA in any loss.
- **Leakage:** scene and clip splits, pHash dedup against every evaluation set, and evaluation sets frozen before the claiming run.
- **Targets that are not clean:** JPEG quality routing; `merged.dng` retired as a target once averaged ground truth is in.
- **Licences:**
  - per-row tags;
  - ShareAlike rows (HDR+, Hypersim, BY-SA Commons) droppable;
  - non-commercial sets for evaluation and fitting only;
  - base-model licences read at the primary source before any weight download;
  - outputs of non-commercial models never in training;
  - FLUX 3 klein outputs never in training, calibration or distillation (its licence's §4.b).
- **Compute:** the ladder and the GPU loader come before any large run; no 496M or 1.1B faithful run without ladder evidence.
- **Overclaiming:** the claim wording in 11.5; the rendered layer is always declared.

## 15. What changes for v14

The planned run was shape J (496M, `--base 64 --mults 1,2,4,8,16 --blocks 2,3,6,8,8 --mid-blocks 12`) at 245 min. The evidence in section 3 points to a ladder instead.

- **v14a, about 30M, and v14b, about 120M.** Shapes are picked from the bench so each lands at its target size.
- Both use the thirteenth run's data, loss, flags and 245 min, so size is the only change; v13 is the 8M point.
- Both are loader-bound at the same 35–40 pairs/s. They see the same pairs, which makes the comparison one of capacity at fixed data.
- **Then v15** takes the winning size with the section 4 data fixes, the section 8.3 loss and the σ² input, changed one at a time where the budget allows.

## 16. What the local ComfyUI install gives

A ComfyUI portable install at `C:\MiraiRender\ComfyUI_windows_portable` holds about 170 GB of weights:

- diffusion models 123.5 GB;
- text encoders 35.6 GB;
- LoRAs 10.2 GB;
- VAEs 1.1 GB.

`C:\MiraiRender\musubi_models` holds another 27 GB (the Wan2.2 T2V low-noise expert in fp16). The Hugging Face cache holds diffusers copies of FLUX 3 klein base, Qwen-Image-Edit-2511, Z-Image-Turbo, Wan2.2 T2V and Qwen3-VL-4B-Instruct.

Licences were read on 2026-10-28: FLUX's from the LICENSE.md in the local cache, the rest from each repository's Hugging Face licence tag.

| Weights here | Size | Licence | Use in this plan |
|---|---|---|---|
| Qwen-Image-Edit-2511 (20B, fp8 mixed) | 19.1 GB | Apache-2.0 | the larger rendered-layer base for stills; a generative baseline |
| Qwen-Image-2512 (20B, fp8) | 19.0 GB | Apache-2.0 | text-to-image; not needed |
| Qwen-Image and Qwen-Image-Edit Lightning LoRAs (4 and 8 steps) | in 10.2 GB | Apache-2.0 | shows these bases distil to few steps (Stage B); the baseline at 8 steps |
| Z-Image-Turbo (6B, bf16) | 11.5 GB | Apache-2.0 | the first rendered-layer base; distilled, so training needs the base release or a de-distillation adapter (check) |
| Wan2.2 T2V and I2V A14B, high- and low-noise experts (fp8) | 4 × 13.3 GB | Apache-2.0 | the video rendered-layer candidate (the low-noise expert as a refiner); the video generative baseline |
| Wan2.2 lightx2v LoRAs (4 steps) | in 10.2 GB | Apache-2.0 | few-step video distillation |
| FLUX 3 Image Klein, base and distilled (5.6B) | 10.3 GB each | FLUX Kommunity License v1.0 | evaluation only (see below) |
| Text encoders: Qwen2.5-VL-7B fp8, Qwen3-VL-4B, Qwen3-4B, UMT5-XXL fp8, Gemma | 35.6 GB | per base | loaded with their bases only |
| Qwen3-VL-4B-Instruct, transformers format (Hugging Face cache) | | Apache-2.0 | corpus tagging (item 5 below) |

**The FLUX licence.**

- It allows non-commercial use, plus commercial use of outputs by a Qualifying User: under US$5M gross annualised revenue with affiliates, who filters content and discloses AI generation.
- Derivatives (a fine-tune or a LoRA) may be made for non-commercial purposes only (§2.b).
- §4.b forbids using the model or its outputs "to improve any other artificial intelligence model" of similar function.

So a shipped rendered layer cannot be built on it, and its outputs never enter training, calibration or distillation. It may run as an evaluation baseline: testing and evaluation in a non-production environment is a listed non-commercial purpose.

It is not the FLUX.2 klein-base-4B that the research found. That model is Apache (tag read today), but it is not on disk.

**What it is not: training data for the faithful layer.**

- **Generated images carry no capture physics.** They have no photon noise, optics or demosaic, so they cannot be clean targets for a model that must undo a camera.
- **Their fine detail is invented.** It is the generator's invention, decoded through an 8× VAE whose texture fingerprint the restorer would learn as truth.
- **Training on generated images degrades successive models** (self-consuming loops). Every result in sections 5 and 6 rests on real photons.
- **Personal material stays untouched.** ComfyUI's output (555 files) and input folders, the `datasets` and `idtest` folders, and the personal LoRAs (`alexandra_*`, `lab_*`) are the user's own. This plan does not open them or use them for anything.
- **The VLM text encoders are not chat models.** They are encoder weights in ComfyUI's layout.

**What it is.**

1. **The rendered layer's bases, with no download.**
   - Section 10's 4B base needed a download. Z-Image and Qwen-Image-Edit-2511 are already here, under Apache.
   - Research deferred Qwen-Image at 20B because bf16 needs 40–78 GB. The fp8 copy here (19 GB) fits the 5090 for inference, and LoRA training fits with block swap (musubi-tuner documents it; measure before committing).
   - Qwen-Image-Edit takes a reference image natively, which is the restoration framing: R in, G out.
   - **Recommendation:** prototype Stage A on Z-Image (6B) for iteration speed. Move to Qwen-Image-Edit only if the 6B's rendered bands lose the human study (11.4).
2. **Training tools already working here.**
   - ai-toolkit and musubi-tuner are installed in `C:\MiraiRender` and have trained LoRAs on this GPU. Stage A uses them, not new code.
   - The pair is (R, ground truth): R is the frozen recovered layer's output on the degraded input. It goes in as the edit model's reference image, with a fixed prompt.
3. **Generative baselines through the headless API.**
   - ComfyUI serves an HTTP API on 127.0.0.1:8188: POST a workflow JSON to `/prompt`, then read the results from `/history`.
   - A small driver runs each held-out input through four models and scores the outputs on our gates:
     - Qwen-Image-Edit with Lightning (8 steps);
     - Z-Image-Turbo img2img at low denoise;
     - Wan2.2's low-noise expert, for video;
     - FLUX 3 klein, evaluation only.
   - That is the generative row of 11.3 without OSEDiff or PiSA-SR, whose SD-2.1 download is broken.
   - We expect these models to fail provenance. Measuring that is the point: it shows the gate separates faithful from invented.
4. **A provenance test set with known masks.**
   - Take real ground-truth images, mask regions of known size, and inpaint them with Qwen-Image-Edit or Z-Image. Both are Apache, so their outputs may feed our detector.
   - Check that `undeclared_invention` flags the inpainted regions. This gives the detector's recall and precision, which section 10's gate addition and the claim wording (11.5, "a detector validated at Y % recall") need.
   - The same G-over-R outputs are the calibration pairs for M's conformal threshold.
   - FLUX outputs stay out of both, because of §4.b.
5. **Corpus tagging with a real VLM.** Run Qwen3-VL-4B-Instruct from the cache over the clean corpus (about 17k images; throughput to measure). It serves two purposes:
   - **Content class** (faces, text, foliage, fabric, architecture, night, sky), for stratified evaluation and per-class reporting.
   - **Bad clean targets:** watermarks, overlays, borders, screenshots, composites, heavy edits and AI-generated images on Commons.

   Tags are metadata only. A flagged image is reviewed or dropped, never edited.
6. **Techniques that confirm the design.**
   - **Colour fix:** ComfyUI's colour-fix nodes (wavelet or AdaIN, used after SeedVR2 and SUPIR) take the low bands from the input and the high bands from the generator. That is the same split as section 10's composition, and it keeps low-band provenance.
   - **Tiling:** tiled VAE and tiled sampling with blended overlaps. The rendered layer needs this in latent space for full-resolution stills, as `apply.py` does in pixel space.
   - **Fitting 32 GB:** fp8 e4m3 weights with bf16 compute, plus block swap, are how 14–20B models run in 32 GB.
   - **Few steps:** the Lightning and lightx2v LoRAs reach 4–8 steps on exactly these bases, which de-risks Stage B.
7. **A ship channel.**
   - A ComfyUI custom node wrapping `apply` (raw or image in; image, variance, provenance map and C2PA out) puts the faithful layer in front of people who already run these pipelines.
   - ComfyUI is GPL-3.0. The node is a thin wrapper whose licence is chosen with that in mind, and the model and harness stay separate.
   - No custom nodes are installed now, so the SeedVR2 node is not here.

**What it needs from the user:**

- a go-ahead for the API driver and the provenance test set (both are evaluation, with no download);
- the choice of base, before any rendered-layer training.

## 17. What changes for v15

v15 takes whichever v14 shape wins the scoring: the fewest metrics lost to the real capture across the eval-raw sets (synthetic known and blind, HDR+ and RawNIND real), ties broken on metrics won.

- v14b won: across the six eval-raw sets it loses 11 median metrics against v14a's 12. Both win 15, and v14b loses 286 per-tile comparisons against v14a's 332.
- v15 therefore takes v14b's shape (about 120M: `--base 64 --mults 1,2,4,8 --blocks 2,3,6,8 --mid-blocks 12`).

- **Data.**
  - v13's sources and weights, as in v14.
  - The HDR+ real store grows from the 150 pairs v14 drew to every train and val scene of the corpus. It is 3,487 scenes to build, about 130 GB (est from 5.1 GB per 140).
  - The target audit (`python -m harness data audit`) reads every clean image for the halo it carries. `--target-audit` then draws only those under their source's `clean.max_halo` (about 0.044 for the two Commons sets, set from the audit's spread).
  - Two sources added on 10-29:
    - SI-HDR. Its merged EXR references are clean targets (weight 4, about 1.3 % of clean draws). Its exposure stacks are real pairs: each input is matched against the merge of its stack's longer frames, with the stack's own noise law. v15 draws whatever the store holds at launch.
    - PixelShift200's pixel-shift merges, which need no demosaic (CC BY 4.0, 5 scenes held out; weight 16, about 3.4 %).
  - Clean draw shares at launch: HDR+ 45 %, Commons featured pictures 37 %, Poly Haven 8 %, RawNIND 6.5 %, PixelShift200 3.4 %, SI-HDR 1.3 %, KADID and TID the rest.
  - SI-HDR's references read high on the halo audit (p75 0.072, p90 0.25, against the 0.044 cut). They are not filtered: the source has no `clean.max_halo`, and at 1.3 % of draws the effect is small. v16 sets a cut from its audit or leaves the source out.
  - The Commons featured-pictures pull is complete.
- **Pulled for v16, not in v15.** UHD-IQA's 900 test images (CC0), from V-Nova's lossless JPEG 2000 mirror (6.2 GB).
  - The archive is one uncompressed tar, read in place: label.py indexes its members and never extracts them.
  - Its JPG mirror is a uniform quality-80 re-encode. It is 41 dB from the originals, and its 8 × 8 blocking is still 4 % over a clean grid at scale 2.
  - The originals are JPEGs themselves (31 % blocking at full size), but within 1 % of a clean grid at scale 2. So they decode at scale 2 (1920 px), where the JPG mirror needed 4.
  - They are display-referred, so audited and filtered at the Commons cut.
  - The full 6,073-image set's host refused every connection on 10-28 and 10-29, so it is listed as manual.
- **RawNIND rejections, for v16.** The real store keeps 2,337 pairs over 286 scenes and rejects 26 (1.1 %).
  - 2 are files LibRaw cannot decode. The other 24 fail the alignment rule: the four quadrant shifts must agree within 1 px.
  - 6 are real changes: kitchen_light's three pairs change their lighting between frames (spread 65 px, correlation 0.46 to 0.63), and skull-C500D's three pairs sit 4 to 7 px off at correlation 1.00, which looks like a scale change.
  - 18 hold at correlation 0.79 to 0.99 while their quadrant shifts spread 2 to 3 px (10 pairs) or 9 to 57 px (8). 13 of the 18 are at ISO 6400 or above, so the quadrant estimate failing in the noise, or on a flat quadrant, is the likelier reading than a moved scene. v16 can estimate the quadrants on a denoised copy and drop a quadrant with too little texture to read. The 5 at ISO 50 to 3200 may be real misregistration and stay out unless the new estimate passes them.
  - Not changed for v15: the rule also builds the real-pair evaluation sets, and v15 freezes its code at launch.
  - RawNIND's listing now keeps only camera raws. Each scene also lists an `.xmp` sidecar and a `.jpg`; the `.xmp` would sort first as the reference. No stored scene changed, so the splits did not move.
- **Sub-pixel registration of the real targets, built into v15** (planned for v16; `harness/data/subpix.py`). The real store aligns each pair at an integer shift, and `--shift-real` picks each tile's best integer shift within a pixel. A probe measured what is left below a pixel: Fourier cross-correlation of band-passed log luminance, with a peak fit, on 512 px crops at the centre and the four quadrants.
  - RawNIND, 60 train pairs: a residual of 0.17 px median (p75 0.28, p90 0.45), a third of pairs over 0.25 px. At ISO below 1600 it is 0.18 px, and the quadrants agree within 0.11 px, so it is a shift of the whole frame, not noise in the estimate. Above ISO 6400 the estimate itself is unreliable (quadrant spread 0.7 px).
  - HDR+, 40 pairs: 0.024 px median, none over 0.25 px. Its targets are registered.
  - A residual of 0.2 px (as a Gaussian spread) takes the target's MTF to about 0.89 at 0.39 cycles a pixel. That is part, not all, of the resolution gate every run loses on RawNIND (v14a's mtf50 0.293, against the input's 0.387 and the truth's 0.5). The rest is the denoise trade.
  - The fix:
    - Each pair's residual is measured on the overlap (a central crop up to 3,072 px) and stored in the pair's record.
    - It is trusted when the quadrants agree within 0.3 px and the residual is under 1 px on both axes.
    - The reference tile is shifted by it in the Fourier domain. That shift is an all-pass, so it moves the target without blurring it the way interpolation would.
    - `--subpix-real` applies it in the loader. A real-pair evaluation set gets a `_sp` twin on the plain set's tiles, with the targets and the camera photos moved.
    - Every run is rescored on the `_sp` twins, since the evaluation truth moves too.
  - The backfill over the stored pairs (10-29):

    | Source | Measured | Trusted | Median residual |
    |---|---|---|---|
    | RawNIND | 2,337 | 2,135 | 0.159 px |
    | HDR+ | 1,117 | | 0.012 px |
    | SI-HDR | 88 | 57 | 0.092 px |

    The loader's train split applies it on 2,005 of RawNIND's 2,196 train pairs (median 0.162 px); the other 191 are untrusted and keep their integer alignment.
  - **Ringing probe**, on 150 RawNIND edge tiles with a residual of 0.15 px or more on an axis (from 1,365 draws).
    - Against the unmoved target, the Fourier move adds edge overshoot of 0.016 at the median edge window (0.097 at the 95th percentile). A cubic spline's move of the same residual adds 0.012 (0.087).
    - Against the capture, the moved target's overshoot is lower than the unmoved one's: 0.008 against 0.013. So most of what the move "adds" against the unmoved target is the misregistration it removes, not ringing.
    - Edge error against the capture: 0.0025 unmoved, 0.0009 after either move. The Fourier move is below the unmoved target on 98 % of tiles, and level with the cubic (below it on 42 %).
    - The Fourier move stays: the same error as the cubic, and no low-pass of the target.
- **Loss.** v13's, plus the transfer term at weight 0.1 (section 8.3).
  - Four fine-tunes of v13: 600 steps at batch 8, at weights 0, 0.1, 0.3 and 1. Each was scored on 64 known-capture tiles of each real set:

    | Weight | RawNIND MTF50 | RawNIND texture | RawNIND shadow SNR | RawNIND ΔE | RawNIND banding | HDR+ texture | HDR+ shadow SNR | HDR+ ΔE | HDR+ banding |
    |---|---|---|---|---|---|---|---|---|---|
    | 0 | 0.298 | 0.383 | 22.8 dB | 0.599 | 0.0117 | 1.085 | 19.9 dB | 0.711 | 0.0224 |
    | 0.1 | 0.322 | 0.585 | 21.9 dB | 0.651 | 0.0125 | 1.276 | 19.5 dB | 0.631 | 0.0241 |
    | 0.3 | 0.337 | 0.833 | 20.6 dB | 0.690 | 0.0154 | 1.329 | 18.7 dB | 0.678 | 0.0251 |
    | 1 | 0.379 | 1.054 | 19.1 dB | 0.734 | 0.0232 | 1.344 | 16.3 dB | 0.747 | 0.0280 |

    HDR+ MTF50 is 0.5 (the ceiling) at every weight. Texture's target is 1, and the capture's shadow SNR is 13.4 dB (RawNIND) and 14.0 dB (HDR+).
  - The rule written before the fine-tunes takes the largest weight that moves both real sets' texture toward 1 without spending colour, banding, edge overshoot or shadow SNR below the capture's.
  - No weight passes. Every weight takes HDR+ texture further from 1 (1.09 to 1.28–1.34), and 0.3 and 1 also cost RawNIND colour and banding. The rule's fallback is 0.1.
  - At 0.1:
    - RawNIND texture goes from 0.38 to 0.59, and MTF50 from 0.298 to 0.322.
    - RawNIND pays 0.9 dB of shadow SNR (still 8.5 dB over the capture) and 0.05 ΔE.
    - On HDR+, ΔE improves (0.71 to 0.63).
  - 600 steps is a short probe: the full run's own scores decide whether the term stays.
- **Flags.**
  - `--shift-real`: each real pair scored at its best integer shift within a pixel.
  - `--noise-map`: the network reads the capture's noise law per pixel (the σ² input of 8.2).
  - `--compile`: torch.compile of the training forward.
  - `--warmup 1000 --lr-floor 0.003`: a longer warmup, and a cosine that ends near zero (8.4).
- **One run, many changes.** Section 15 asked for these one at a time. The budget is one 245-minute run, so v15 changes them together and its record says so. If v15 regresses on a gate, the ablation runs from the fine-tune evidence first.
- **Process fixes found on the way.**
  - The priority helper (decodedstore._lower_priority) passes typed ctypes arguments; untyped, the process handle was truncated and the call failed silently.
  - Store builds and the audit run on a spawn `multiprocessing.Pool`. On this Windows Python a `ProcessPoolExecutor` worker can die on an invalid handle in its call queue (WinError 6), which broke the pool and failed every scene of the first full HDR+ store build.
  - The real store now has a free-space floor (`HARNESS_REALSTORE_RESERVE_GB`, default 150, the collector's).
    - The decoded cache stops at 300 GB, so it gives way first.
    - A write that would cross the floor stops the build, and the next run resumes there. The unwritten pairs carry no record, so they are not rejected.
  - Scoring a run skips a real-pair evaluation set that is not built yet. `eval-raw --real` builds a missing set, which takes hours for SI-HDR's stacks, so that set builds in its own low-priority job and the runs are scored on it once it exists.
  - Before v15's launch, only sources with a `clean.max_halo` need their new items audited; the loader leaves out an item of those that is unaudited. PixelShift200's merges, about 3 GB each to decode, decode before the launch, so the loader never meets one undecoded.
- **A held-out leak, found and fixed on 10-29 before v15's launch.**
  - **The cause.** RawNIND names its unknown-sensor files `Bayer_UNK_TEST_<scene>_…`, so `scene_of` keys six of the dataset's test-reserve scenes as `UNK_TEST_LucieB_<name>`: board, bw_drawing1, bw_drawing2, groceries, painted_plants and painted_wallpaper. The manifest's `clean.holdout` named them without the prefix, so they never matched.
  - **What leaked.** Their 10 GT raws fell in the clean sampler's train split. They were about 0.1 % of clean draws for v13, v14a, v14b and the fine-tunes, and possibly earlier runs.
  - **Scores are unaffected.** No score saw those scenes: the RawNIND real evaluation set draws its 64 tiles from 9 val scenes. `.tmp`'s leak rescoring confirms every run's RawNIND report has no tile from them. The scenes' own real test pairs (32 of the 148) are in no scored set.
  - **The fix.**
    - The manifest names the scenes by their keys.
    - `rawnind_pairs` now raises, instead of logging, when a test-reserve scene is not in `clean.holdout`.
    - A scan of the reserve against `split_of` finds 556 files held out and 0 leaking.
  - **Unchanged.** The scene regex keeps the `UNK_` prefix. Dropping it would rehash the splits of five unknown-sensor scenes outside the reserve, and move them against the real store.
- **The decoded key leaves out `holdout`**, as it already left out `max_halo`. Neither changes the decoded array, so a holdout edit no longer re-keys a source and orphans its arrays.
  - The leak fix re-keyed RawNIND, SI-HDR and PixelShift200. Their arrays were hard-linked under the new keys, so nothing decoded again, and `--prune` removes the old names once no process keys them.

## 18. What changes for v16

v16 is queued behind v15: `.tmp/v16_after_v15.sh` starts it once v15 is scored. It keeps v15's sources, weights, shape (about 120M) and flags, so its scores read against v15's. Four things change: how every run is scored, two loss terms, the network's resampling, and the budget.

- **Scoring (every run, from `.tmp/frozen_eval16`).**
  - **Noise is read as a residual** (`tilemetrics.residual_noise`, `noise="residual"`).
    - Shadow SNR and DR steps used to take the degraded tile's own high-pass as its noise. Clean texture leaks into that, so a busy tile read noisier than it is, and a blurred output read cleaner.
    - Evaluation now reads what of the degraded luminance no shift-invariant linear filter of the clean tile explains: one gain per radial band of 1/64 cpp, a 4 px border left out.
    - Blur, sharpening, a tone curve and a sub-pixel shift read as no noise; added noise reads as it does against the truth.
    - Labels and the loader's conditioning keep the high-pass reading (`hp`), so training is unchanged. Tests: `tests/test_residual_noise.py`.
  - **A third gate set, the oracle linear frontier** (`harness/model/frontier.py`, `gates["frontier"]`).
    - On noisy real pairs the capture gates ask for the impossible. Even the best linear filter, fitted on the truth itself, falls below the noisy capture on mtf50 and texture. On RawNIND, taking shadow SNR from 14.7 to 26.7 dB takes mtf50 from 0.39 to 0.15 and texture from 0.67 to 0.11.
    - So the output is judged against a parametric Wiener ladder fitted per tile on the truth: 16 rungs from λ 0 (the inverse filter) to 32 (heavy smoothing), 32 radial bands, gains clipped to [0, 8], the lowest band kept at 1.
    - Each tile's output shadow SNR is located on that tile's own ladder, and every other metric is read off the ladder there.
    - Every metric must tie or beat that matched point, mtf50 and texture by distance to the truth. Shadow SNR is the match, so it is not judged.
    - The ladder is cached per set, noise reading and ladder definition under `derived/frontier/`.
  - **Reports keep the restored tiles** (`outputs.npz` beside `heldout.json`), so a new reading or frontier rescores a run without its checkpoint or the GPU (`heldout.rescore`).
  - **`python -m harness model compare`** writes one table per set across runs: step, the capture, camera and frontier verdicts, and each metric as output / matched frontier.
  - **Every earlier run is rescored under both** once v15 is scored (`.tmp/rescore_v16.sh`, v14b first, then v13, v14a, v12, v11, v10), on the GPU beside v16. Each replaced report is kept as `<dir>_pre_v16`. Scores in sections 1 and 17 are under the old reading.
- **What v14b loses to the frontier on HDR+** (a probe on an earlier ladder; the rescore replaces these numbers):

  | Metric | v14b | Frontier | Note |
  |---|---|---|---|
  | edge overshoot | 0.0207 | 0 | the capture reads 0.0063 |
  | banding | 0.0199 | 0.0164 | |
  | periodicity | 0.022 | 0.0186 | |
  | shading | 0.0074 | 0.0062 | |

  These are the artifact and shading gates: v14b passes the noise a linear oracle reaches, and leaves halos and structure at 2 to 16 px.
- **Where the periodicity comes from** (`.tmp/chk/diag_period.py`, v14b on 24 known-capture tiles of each real set, CPU). The table gives the residual's luminance power (image minus truth) at each grid frequency on the axes, over the median of its ring at the same radius. It is the median over tiles, and the Wiener is matched per tile to the output's shadow SNR.

  | Set | Image | 1/2 cpp | 3/8 cpp | 1/4 cpp | 1/8 cpp | 1/16 cpp |
  |---|---|---|---|---|---|---|
  | RawNIND | capture | 3.84 | 2.55 | 2.50 | 2.79 | 2.10 |
  | RawNIND | v14b | **26.10** | 2.87 | **4.31** | 3.74 | 2.38 |
  | RawNIND | Wiener | 7.11 | 3.11 | 2.77 | 2.97 | 2.27 |
  | HDR+ | capture | 8.09 | 1.19 | 2.84 | 1.86 | 1.70 |
  | HDR+ | v14b | **14.81** | 2.36 | 3.78 | 2.89 | 2.18 |
  | HDR+ | Wiener | 3.88 | 2.14 | 2.82 | 2.07 | 1.92 |

  - The output's peak sits at the network's own strides: 1/2 cpp, and less at 1/4 and 1/8. There is none at 3/8, the JPEG 8 px block harmonic no power-of-2 stride makes. So it is the resampling grid, not the targets.
  - The lag-8 autocorrelation excess on RawNIND is in the Wiener too (its mean excess peaks at (±8, ∓2), 0.041 against the output's 0.018). That part is the data's, not the network's.
- **Smooth resampling** (`--resample smooth`; `net.BlurDown`, `net.SmoothUp`).
  - Down: a [1 2 1]² binomial low-pass on reflect padding before the strided 3×3 conv (Zhang 2019, anti-aliased CNNs). A bare strided conv aliases, so the level below sees a different signal at each phase of the tile against the stride.
  - Up: a 1×1 conv, then bilinear ×2, in place of the stride-2 transposed conv (Odena et al. 2016). A transposed conv gives each of the four output phases its own kernel.
  - The checkpoint's config records it, and `apply` rebuilds the matching net. `learned` stays the default, so every earlier checkpoint loads unchanged.
- **Two loss terms on the gates' own readings** (`harness/model/losses.py`).
  - **`overshoot_loss`, weight 1.5**: `edge_overshoot_pair` as a loss.
    - It works on clipped linear luminance, with the output's local gain against the target's at σ 8 px divided out under no gradient.
    - For each pixel and axis, a 9 px window gives the target's step and how far the output's maximum or minimum passes the target's range, over the step.
    - It is the mean over windows that hold an edge (step ≥ 0.1, inside the 0.98 clip guard).
    - It is zero for the identity and for any output inside the target's local range, so the perfect restoration is still the optimum.
    - It also charges what a blur carries into a window from beyond it: a 1 px Gaussian on dead leaves reads 0.025.
  - **`pyramid_loss`, weight 1**: the L1 of the encoded residual's block means at 2, 4, 8 and 16 px, each block weighted by its mean clip weight.
    - The encoding is near-logarithmic, so a block mean's residual reads as the Weber contrast `banding_score` and `shading_rmse` read.
    - The per-pixel L1 dilutes a residual that is coherent over a block; this term does not.
  - Both are zero at weight 0, where `pair_loss` is exactly v15's. Tests: `tests/test_losses.py`, and `tests/test_model.py` for the recorded terms and a rebuilt smooth net.
- **Budget: 480 minutes** (up to 120,000 steps), twice v15's. v15 trains at about 63 images a second at batch 16, so 480 minutes is about 110,000 steps.
- **Data: v15's sources and weights,** from the real stores as they stand at launch. The HDR+ and SI-HDR stores are still filling (1,576 of 3,487 and 57 of 163 on the morning of 10-29), so v16 draws more HDR+ pairs than v15 did. UHD-IQA's decode (100 of 845) and audit finish after the launch, so it goes into v17.
- **One run, several changes**, as for v15. If v16 regresses on a gate, the first ablation is resampling against the two loss terms, from v15's checkpoint.
- **A process fix, found when v15 failed to start (v15b and v15c).**
  - CPython 3.12's `_wmi` races when a query times out. The abandoned query thread later writes into, and closes, a handle number the process has reused by then. Upstream fixed it (gh-130727) after 3.12's last bugfix release.
  - Every spawned DataLoader worker asks WMI through `platform.uname()` when torch imports. Under a busy CPU the loader died at step 0 with `[WinError 6] The handle is invalid` on its queue semaphore.
  - `harness/__init__.py` turns WMI queries off on Windows, so `platform` answers from `sys.getwindowsversion()` (the same uname on this machine).
  - Loaders take `harness.loader_context()`, a spawn context whose Process class lives in `harness`. A worker unpickles that class first, so it imports `harness` before torch.
  - v15d trains on it. Tests: `tests/test_platform_wmi.py`.

## 19. The long run: three seeds on the full real stores (v18)

v18 is three runs of one recipe that differ only in the seed, 24 hours each on the one GPU, back to back. `.tmp/v18_seeds.sh` queues them behind the burst run on v17 and the baseline scoring (11.3). They train from `.tmp/frozen_v18`, the harness at 73758442.

- **Why three seeds.**
  - A held-out gate that passes on one run can pass by the draw: the initial weights and the order of the stream.
  - So a claim holds only if every seed passes it, and every score is reported with its spread.
- **What varies: `--seed 0`, `1`, `2`.**
  - The initial weights: `torch.manual_seed(seed)` runs before the network is built. Before 73758442 nothing seeded them.
  - The training stream: the PairStream at seed + step.
- **What stays fixed: `--val-seed 0`.**
  - The calibration draw (the normaliser) and the synthetic and real validation draws.
  - So every run chooses best.pt on one validation set, never on held-out.
- **Recipe: v17's or v15's, picked when v17 is scored (about 06:40 on 10-30).**
  - v17's if it passes the tone gate wherever v15 does and keeps every gate v15 passes.
  - Otherwise v15's: v17's flags without `--resample smooth`, `--overshoot-weight` and `--pyramid-weight`.
  - The pick is written to `.tmp/v18_recipe`; the queue waits for it.
  - The shape stays at about 120M (base 64, mults 1,2,4,8, blocks 2,3,6,8, 12 mid blocks). This run asks what more steps on real data buy, not what size does.
- **Data: the real stores as they stand at launch, plus v17's clean sources and weights.**

  | Store | Stored | v17 drew (train split) |
  |---|---|---|
  | HDR+ | 3,641 scenes | 3,126 pairs |
  | RawNIND | 287 scenes | 2,196 pairs |
  | SI-HDR | 164 scenes | 435 pairs |

  - **UHD-IQA joins** (`hf-uhd-iqa`, CC0) at weight 1: 845 decoded, 593 kept by the target audit (`max_halo` 0.044; 555 train, 38 val). It adds to the synthetic validation draw, so v18's validation losses do not read against v17's. And v18 against v17 measures the budget and UHD-IQA together. The release is judged on the gates and against the baselines, not against v17.
  - **The Commons quality images stay out.** Their audit is nearly done (7,700 of 8,557 read, none rejected), but D: is below the decoded store's 300 GB floor, so they cannot be decoded before the launch. They join a later run once the store has room.
  - **Not in v18, still open:** a halo cut for SI-HDR (its references read p75 0.072 and p90 0.25 on the audit, against the 0.044 cut; about 4 % of real draws), and RawNIND's quadrant alignment re-estimated on a denoised copy.
  - **Never in training:** raw.pixls.us (held-out), RawNIND's test_reserve and holdout scenes, the real-pair val and test scenes, SIDD, and generated images.
- **Budget: 24 hours a seed (`--minutes 1440 --steps 300000`).**
  - At v17's rate (about 56 images a second at batch 16, 3.5 steps a second), 24 hours is about 300,000 steps and 4.8M tiles of 256 px.
  - That is three times v17's 480 minutes (about 100,000 steps).
  - The cosine runs to the floor over whichever limit is reached first.
  - Each seed gets up to four tries, each resuming from last.pt; `--minutes` is summed across resumes.
- **Order and timing (est).**

  | Start | GPU work |
  |---|---|
  | 10-30 06:15 | v17 ends; scored by 06:40 |
  | 10-30 06:40 | the burst run on v17 (240 min), then its evaluation |
  | 10-30 about 10:45 | the baselines through the gates and SIDD validation (about 1 h), once their weights are pulled; otherwise after v18s2 |
  | 10-30 about 12:00 | v18s0, then its scoring (about 25 min) |
  | 10-31 about 12:30 | v18s1 |
  | 11-01 about 13:00 | v18s2 |
  | 11-02 about 13:30 | the comparison and the release seed |

- **Other splits, and why not.**
  - One 72-hour run gives no spread.
  - Three 72-hour runs take nine days.
  - Three at once would not fit: each takes 10.5 GB of GPU memory and 12 loader workers, one run already holds the GPU at 97 %, and RAM is tight.
- **Stop rules.**
  - A seed that dies four times is reported as dead, and the queue goes on to the next.
  - After v18s0 is scored, the queue checks it against v17 report by report (`.tmp/v18_gate_check.py`). If v18s0 fails a gate that v17 passes, the queue stops before v18s1 and waits for `.tmp/v18_go`. Three times the steps should not lose a gate; if they do, the cause is found before two more days go in.
- **Scoring, per seed.**
  - `score_run_v17.sh`, from `frozen_eval17`, so the scores read against v15's and v17's:
    - the demo and raw.pixls.us sets, known and blind;
    - the three real-pair sets, known and blind;
    - the sub-pixel sets.
  - SIDD validation, blind (`model baseline sidd --ckpt`). SIDD is sRGB phone crops and the model trains on linear data, so it is out of the model's domain: it is reported, not a gate.
  - Then `model compare --runs v15,v17,v18s0,v18s1,v18s2,nafnet,restormer,scunet`.
- **What is claimed.**
  - A gate passes for v18 only if it passes on all three seeds.
  - Each score is given as the median of the three, with the min and max.
  - The release is the seed with the best validation loss, not the best held-out score. It goes through export, capture-eval and the model card (section 12), and the burst layer is then trained on it.
- **Disk.** Each seed keeps two checkpoints of 1.9 GB plus its reports and their tiles, about 15 GB for all three. D: has 191 GB free.
