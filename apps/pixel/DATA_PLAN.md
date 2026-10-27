# Data plan — every constant the harness uses, where it comes from, and how the rest gets collected

The harness started with invented numbers: three camera presets, one display, a 94–120 ppd
visibility band, ΔE/overshoot/banding/flicker gate thresholds, a toy scene. This document is
the replacement process: **find** the published figure, **source** it (URL + quote), **derive**
the harness table from it, **collect** the bulk sets that let the thresholds be *fitted* instead
of guessed, and **scale** the collection without changing the code.

## 1. Principles

1. **No source, no row.** `harness/data/reference/*.csv` is hand-verified; every row carries
   `source_url` and `quote`. `read_reference` refuses a row without them, so an unsourced number
   cannot reach a derived table.
2. **Manifest, not scripts.** `harness/data/sources.json` describes every source once: licence,
   size tier, access kind (`direct`, `api`, `pip`, `docker`, `manual`, `page-only`,
   `scrape-tos-unclear`), the expander that turns an API/folder/Hub listing into file specs, and
   what it feeds. Adding a dataset is adding a row; nothing else changes.
3. **One collector, every tier.** `python -m harness data collect` is parallel, resumable
   (HTTP Range + `.part`), ledgered (append-only `data/ledger.jsonl`), checksum-verified
   (sha256/md5 when stated), per-host throttled, budgeted (`--max-bytes`) and bounded
   (`--param limit=N`). Re-running is free: finished files are skipped from the ledger.
4. **Derived, not copied.** `python -m harness data derive` rebuilds `data/derived/*.json`
   from raw files. Claims reference them as `derived:<id>`; the derivation is reproducible.
5. **Only what the site allows.** Plain fetches with the harness user agent; sites that refuse
   are listed as `page-only` or `manual`, never circumvented. Licences are recorded per source;
   non-commercial sets (CC BY-NC, research-only) are for evaluation and fitting only.

## 2. Assumption → sourced data → status

| Harness constant (was) | Replaced by | Source rows / derived table | Status |
|---|---|---|---|
| `best_case_earth` preset (pitch, full well, read noise) | 15 measured sensors; frontier = `derived:best_case_earth_sourced` by *simulated* DR (Hamamatsu ORCA-Quest 2: 4.6 µm, 7 ke⁻, 0.3 e⁻, 16-bit, 14.5 stops; the GSENSE6060BSI's 14.95 engineering stops need its dual-gain HDR readout, which the 12-bit linear model does not simulate: 13.66 stops; optics are the harness default where no lens row exists for the format, `optics_source` says so) | `reference/sensors.csv` → `derived/camera_presets.json` | done |
| aperture of the best lens | 3 measured lenses (MTF50 center, LensTip); full-frame sensors pair with Sony FE 50 f/1.2 GM | `reference/lenses.csv` → `camera_presets.json` | done (3 rows; more via `lens-designs-claff`, `deeplens-lenses`) |
| one hard-coded 31.5″ 8K display at 0.6 m | 5 monitors/phones with vendor pixel geometry + 4 headsets with stated ppd | `reference/displays.csv` → `derived/displays.json` (`derived:dell-up3218k` = 120.1 ppd at 0.625 m) | done |
| viewing distance 0.6 m | 0.625 m monitor (Rempel 2007), 0.362 m phone (Bababekova 2011), 1.5 H TV (EBU TR050), ITU design distances 0.75/1.5/3.1 H | `vision_thresholds.csv` rows `display.distance_m*`, `info.design_distance_*` | done |
| `VISIBILITY_PPD = (94, 120)` | 94 ppd lower bound (arXiv 2410.06068; 89 red-green, 53 yellow-violet) | row `gate.display_ppd` | done (upper bound 120 remains the practical ceiling of the sourced displays, not a JND) |
| ΔE gate 2.0 | JND ≈ 2.3 (CIE76 literature), 10-bit quantisation ≈ 1 | rows `gate.delta_e_mean`, `info.delta_e_jnd` | done |
| banding gate 0.01 | ITU-R BT.2246-9 contrast sensitivity 1/50–1/100; CAMBI TVI 0.019; 12-bit PQ for no banding (BT.2390). Put to KADID-10k Quantization (#22, 81 refs × 5 levels) with `natural.banding_pair`: rank correlation −0.31, fit **not supported**; the ratings bound the threshold below 0.145 (level 1 already rated 4.17); TID2013 quantization noise (125 images, ρ −0.24) agrees | rows `gate.banding_score`, `info.cambi_tvi_threshold`, `info.hdr_bits_no_banding`; `derived/gates_fitted.json` (record of the unsupported fit) | done: sourced 0.01 kept, fit recorded with its failed criteria |
| flicker / frame-rate gates | CFF saturation ≈ 90 Hz; harmful-flash limits (BT.1702-3); 120 Hz rank loss ≈ 1 (TR050) | rows `info.cff_saturation_hz`, `info.harmful_flash_*`, `info.frame_rate_rank_loss_120hz` | done (informational; temporal gate still compares against the camera sim) |
| edge-overshoot gate 0.02 | KADID-10k High_Sharpen ladder (#24, 81 refs × 5 levels, 30 DCRs each): fitted mean DCR 4.5 at overshoot **0.22** (95% CI 0.19–0.26, ρ −0.80, supported but extrapolated: reported beside the 0.02 pass line, not adopted); at DCR 4.0: 0.37 | `fit_gates` → `derived/gates_fitted.json` → `claim_sourced.json` | done, with caveats (§5): extrapolated below the mildest rated level (0.56, rated 3.58); the 0.02 default stays as a stricter engineering target |
| toy scene (`scene.py`) | CC0 HDRIs / PBR textures / models (Poly Haven, ambientCG); Infinigen / Kubric for flow ground truth | `polyhaven-*`, `ambientcg-*`, `gen-infinigen`, `gen-kubric` | catalogued; assets collected at small scale |
| colour targets (patch Lab) | X-Rite ColorChecker (Zenodo), BabelColor average, Munsell, Labsphere SRS-99, CIE 1931/2015 tables | `derived/colorchecker.json`, `spectra.json`, `cie_tables.json` | done |
| camera spectral sensitivities | Jiang 2013 camspec database (28 cameras) | `derived/camspec.json` | done |

## 3. Which datasets calibrate which gate

The catalogue (`harness/data/catalog/datasets.json`, 122 sets, groups A–D) was searched for
human-rated data per gate. What it found, and what it did not:

| Gate | Fit / validate with | Why these |
|---|---|---|
| artifacts (overshoot, banding, noise, JPEG) | **A**: TID2013, KADID-10k (only graded *sharpening* ladder), CSIQ, LIVE R2, BAND-2k (banding MOS), KonIQ++ (defect labels) | classical MOS over controlled distortion ladders: threshold = metric value where MOS drops one JND |
| shift of those thresholds to generated content | **B**: AIGCIQA2023 (done), AGIQA-3K, AIGCQA-30K labels, AGIN | MOS on AI images; no AIGI set grades halo/banding/tiling separately and none ships a reference, so A fits, B checks through no-reference proxies. Done on AIGCIQA2023 (`derived/shift_check.json`, §4 step 2): the halo proxy does not transfer, banded area is the one artifact the raters punish (ρ −0.49 with quality MOS) |
| artifact *regions* | RichHF-18K, Rapidata Rich HF (heat-maps), ArtiBench (boxes), X-AIGD, SynthScars, PAL4VST (masks) | localisation targets for region-level artifact metrics |
| composition (human preference) | Rapidata 700k (+ per-vote rater metadata), open-image-preferences (annotator ids, both-good/bad), GenAI-Bench arena, HPDv2/v3, Pick-a-Pic v2 mirror, datapointai 2M (gated) | raw pairwise votes with ties and rater ids → Bradley–Terry with rater effects, bootstrap CIs |
| temporal | VBench human annotation (gated), VideoGen-RewardBench, Rapidata T2V, VideoScore-Bench, VideoFeedback, KoNViD-1k, LIVE-Flicker, LIVE-YT-HFR, LIVE-YT-Banding | per-dimension video ratings; flicker / frame-rate / banding sets for the single-dimension checks |
| tone / colour on real captures | CID2013 (Zenodo), SPAQ attributes, PIQ23 (JOD scales), MIT-Adobe 5k retouch targets | attribute MOS and JOD scales on camera output |
| camera noise model (`camera.py`) | SIDD (NLF per capture), LRID / ELD / CRVD (calibrated K, read, row noise), RawNIND ISO sweeps, raw.pixls.us (one raw per camera) | measured shot/read noise vs ISO to validate the simulated floor |
| demosaic / resolution floor | PixelShift200 (full-colour GT vs single shot), MTF Mapper slanted-edge references | ground truth the Airy + Bayer sim must reproduce |

## 4. Scaling plan

Tiers decide *when*, not *how*; the same command scales from kilobytes to terabytes.

| Tier | Size | What is in it | How to run |
|---|---|---|---|
| S | < 50 MB | spec PDFs, vendor/Wikipedia pages, lens prescriptions, label-only tables, tiny HF sets | `data collect --tier S` (default), done on a laptop |
| M | < 2 GB | TID2013, AGIQA-3K, CID2013, IQA labels, HPDv2 test, arena battles | `data collect --tier M --jobs 8 --max-bytes 20e9`; sample first with `--param limit=1` |
| L | < 50 GB | KADID-10k, PIPAL, UHD-IQA, Rapidata rich feedback, VideoGen-RewardBench, VideoScore-Bench, KoNViD | `HARNESS_DATA_ROOT=/mnt/data data collect --tier L --jobs 16`; shards stream in parallel, resume on interruption |
| XL | ≥ 50 GB | HPDv3 (141 GB), Pick-a-Pic v2 mirror, LRID (596 GB), PixelShift200, SIDD DNG mirror, HDR+ bursts (gsutil), SNIC, RawNIND | `data collect --ids <one> --param limit=N` for samples; full pulls only on a machine with the disk, one source at a time |

Next steps in order:

1. **Fit, don't guess.** Done for both artifact gates with one deriver, `fit_gates`
   (`harness/data/fit.py`): read a KADID-10k ladder (lossless shards, row order re-checked on every
   reference, 81/81), measure the harness metric reference-to-distorted with the estimators in
   `metrics/natural.py` (each calibrated against its scene metric: overshoot within 1 % on the
   slanted edge, banding within 22 % on posterised frames, confounds recorded), join DMOS from
   `hf-iqa-dataset-labels`, fit a logistic in log(metric) pinned to the DCR bounds [1, 5] with a
   reference-level bootstrap, probe specificity on five other distortion types, and apply the
   support rule (ρ ≤ −0.5, CI ratio ≤ 3, ≥ 80 % measurable) before the value may replace a sourced
   row. Edge overshoot (#24): 0.22 at mean DCR 4.5, CI 0.19–0.26, ρ −0.80, supported but extrapolated (no rated
   image sits below the crossing), so `claim_sourced.json` carries it beside the gate and keeps 0.02 as the pass line. Banding (#22 Quantization): ρ −0.31, CI six decades wide, not supported;
   KADID's quantization posterises the whole image and the raters judged that, while the metric
   reads smooth regions only (18 % of rated images have none). The data bound the threshold below
   0.145 and the sourced 0.01 stays; TID2013 quantization noise (`hf-tid2013`, 125 images) is the
   cross-check (ρ −0.24, same conclusion). A banding *detection* threshold needs a banding-only
   set: BAND-2k and LIVE-YT-Banding are page-only (§6). Texture repetition has no rated source.
2. **Shift check.** Done with one deriver, `shift_check` (`harness/data/shift.py`), on the 2,400
   AIGCIQA2023 images (`hf-aigciqa2023-images`: six generators × 100 prompts × 4, MOS for quality,
   authenticity and correspondence; the index ↔ MOS ↔ generator alignment is verified from the
   repo's json and sheet, not assumed). No AI-image set grades halo or banding separately and none
   ships a reference, so the fitted gates cannot be re-run as such; two no-reference proxies
   (`metrics/natural.edge_overshoot_nr`, `banding_nr`) are read three ways: on the 81 KADID
   references (the pristine distribution), on the KADID #24 and #22 ladders next to the pair metrics
   and DMOS (calibration), and on the generated images against their MOS. `derived/shift_check.json`:
   - The halo proxy does not transfer to natural images: median NR/pair ratio 0.09 (0.7 on synthetic
     edges), ρ 0.37 with the pair metric and −0.29 with DMOS (pair metric −0.80). Its pristine
     baseline (p50 0.054, p95 0.125) lies above the NR equivalent of the fitted gate (0.019), so
     the 0.22 gate has no usable no-reference form and its drift on generated content is
     **unmeasured**. What is measured: generated images show no more edge overshoot than photographs
     (p50 0.016 vs 0.054; 1 % above the pristine 95th percentile), and the proxy correlates
     *positively* with quality MOS (ρ +0.34, n 2,394): on generated content it reads sharpness
     (Glide, the blurriest generator, reads 0 and rates lowest, 40.6).
   - The banding proxy transfers partially: ρ 0.55 with the pair metric on the #22 ladder (median
     ratio 0.22); against DMOS −0.05 for the step and −0.26 for banded area (pair metric −0.47).
     On generated images the step sits at the 8-bit floor on both sets (p50 0.011 vs pristine 0.017;
     4 % above the pristine 95th percentile), but banded *area* is the strongest quality signal in
     the set: ρ −0.49 with quality MOS, −0.36 with authenticity (n 2,360). It is a between-generator
     effect: Glide has 47 % of its images above the pristine 95th-percentile area (p50 0.24) and the
     lowest MOS; Controlnet, DALL·E and Stable Diffusion read 0 and rate 56–57; within one
     generator |ρ| ≤ 0.4.
   - Verdict: no evidence that generated images carry more halo or step banding than photographs;
     flat posterised regions (banded area) are the one artifact these proxies catch and the raters
     punish. The sourced 0.01 Weber gate lies below an 8-bit one-code step (0.013–0.03), so any
     undithered 8-bit deliverable fails it by construction: candidates are judged at the mastering
     bit depth or dithered (§5). Rerun: `python -m harness data collect --ids hf-aigciqa2023-images
     --max-bytes 1e9` (893 MB) then `python -m harness data derive --ids shift_check` (about five
     minutes on seven workers).
3. **Preference model at scale.** Done: `preference_public` (`harness/data/prefdata.py`) reads
   the five public vote sets on disk and fits each with `harness/preference.py`:
   Rapidata 700k (674,543 decisive votes over 4 generators, no rater ids, only a `userScore`),
   open-image-preferences (29,985 votes over 2 generators, 247 raters, 30 % ties), GenAI-Bench
   image (3,192 votes over 21 generators, 26 % ties), GenAI-Arena human eval (350 votes over 17
   generators, 36 judges, 24 % ties) and HPDv2 test (3,700 image sets ranked by 10 annotators,
   unrolled to 153,000 pairs). The model is Bradley–Terry with a Rao & Kupper (1967) tie band
   (logistic link, tie when |s_a − s_b| < θ) and one scale β per rater (Hunter 2004's MM fit is
   the no-tie, no-rater reference and `test_preference.py` checks the two agree). Validation is a
   20 % held-out split per set, a ten-bin calibration curve and a five-fold consistency check
   (bootstrap sd over between-fold sd; 1 is honest).
   - Ties are a third outcome in three sets; plain Bradley–Terry cannot score them at all. The
     tie band gives them a likelihood (θ 0.58–0.62) and leaves the decisive accuracy unchanged
     (0.514 / 0.691 / 0.692).
   - Rater scales change the held-out log-likelihood by −0.0010 to +0.0124 nats per vote on the
     three sets with rater ids. On open-image-preferences the fitted β tracks each rater's
     agreement with the majority (rank correlation 0.91 over 82 raters with 20+ votes); on
     HPDv2 all ten annotators read β 5–6.7 (they agree with the consensus 83–87 % of the time).
   - Calibration: ECE 0.005 (Rapidata), 0.020 (OIP), 0.055 (GenAI-Bench), 0.170 (arena, 52 test
     votes) and 0.187 (HPDv2: the within-set rankings are more decisive than a one-scale model
     predicts). Bootstrap sd over fold sd: median 1.11 across the five sets (1.07–1.85), so the
     harness intervals are not overconfident.
   - Cross-set check: GenAI-Arena and GenAI-Bench share 17 generators; the two fits agree in sign
     on 85 % of the 136 pairs (rank correlation 0.87). No other pair of sets shares an item.
   - Held-out accuracy on generator-level votes is 0.51–0.69: raters disagree more than the
     generators differ, which is the regime the composition gate will see. The harness therefore
     gates on the **lower** interval bound of P(candidate ≻ camera), never the point estimate.
   Output: `data/derived/preference_public.json` (per-set counts, fit, held-out report,
   calibration, fold consistency, rater check, `userScore` check, cross-set overlap, verdict).
   Rerun: `python -m harness data derive --ids preference_public` (HPDv2's 153,000 pairs dominate
   the run time; `HARNESS_PREF_SETS=genai_bench_image,genai_arena_human_eval` for a quick pass).
4. **Traced optics.** Done: `lens_traces` (`harness/data/optics.py`, optiland 0.6.3) ray-traces
   every prescription on disk: 300 Claff `.zmx` patent examples and 4 DeepLens JSON lenses
   (the DeepLens `.zmx` twins fail in optiland's reader). Each lens gets a paraxial image solve, an
   on-axis refocus scan at the d line, the centre Strehl ratio (FFT PSF), tangential and sagittal
   MTF50 at 0/0.5/0.7/1.0 of the field (FFT MTF to the diffraction cutoff), the diffraction MTF50
   0.404/(λN) and the residual Gaussian blur that reproduces the centre MTF50 in quadrature with
   the Airy limit (lens only, no sensor). 295/304 traced; 9 failed
   (3 unreadable by optiland; 2 no finite paraxial focal length or pupil; 2 extended asphere (xasphere) surface; 1 reader index error; 1 non-sequential file); 234 are full-frame (image height ≥ 21 mm).
   - Centre Strehl p10/p50/p90 0.03/0.34/0.98;
     25 % of the lenses are diffraction-limited on axis
     (Strehl ≥ 0.8). Centre MTF50 p50 62 lp/mm (p90 206, max 417);
     edge MTF50 p50 6 lp/mm, read at the on-axis plane, so it carries the field
     curvature of designs meant for a curved or refocused field. Residual blur p50
     2.8 µm (p10 0.23, p90 13).
   - By F-number band (medians): F/0-1.6: n 46, MTF50 56 lp/mm, Strehl 0.13, residual 3.4 µm; F/1.6-2.2: n 53, MTF50 39 lp/mm, Strehl 0.08, residual 4.8 µm; F/2.2-3.2: n 77, MTF50 58 lp/mm, Strehl 0.24, residual 3.2 µm; F/3.2-4.5: n 58, MTF50 67 lp/mm, Strehl 0.45, residual 2.6 µm; F/4.5-100: n 56, MTF50 84 lp/mm, Strehl 0.94, residual 0.6 µm. Residual alone rewards slow lenses at the diffraction
     limit, so the best lens is chosen by centre MTF50.
   - Sharpest full-frame lens: `KR1020200129711_Example02P.zmx` (84.5 mm F/1.48),
     Strehl 0.83, centre MTF50 385 lp/mm, edge
     6 lp/mm, residual 0.27 µm. `camera_presets`
     puts it on the best-DR sensor as `best_case_traced` (`"preset": "derived:best_case_traced"`):
     the lens-only floor is 0.27 µm against the 1.0 µm default of
     `best_case_earth`, and against `lenses.csv` residuals that still hold the sensor.
   - DeepLens cross-check (production-lens patents, JSON build): ef100mm_f2.8 Strehl
     0.85, MTF50 186 lp/mm, residual
     0.62 µm; ef50mm_f1.8 Strehl 0.05, MTF50
     55 lp/mm, residual 3.4 µm, which is the
     wide-open double-Gauss regime the Claff F/1.6–2.2 band shows.
   - Repairs the run makes and flags: 8 prescriptions carry a glass row with Abbe number 0
     (plane filter plates) that optiland's Buchdahl model turns into a nan index; the loader swaps
     those for a constant index at the filed nd (`repaired_glass`). 3 fast
     lenses whose marginal rays miss at the filed aperture are re-traced at a smaller entrance
     pupil, largest first (`stopped_down`, `aperture`). 5 lenses hit the edge
     of the refocus scan and are excluded from the best-lens pick.
   Output: `data/derived/lens_traces.json` (method, per-lens fields and focus record, summary with
   the failure list). Rerun: `python -m harness data derive --ids lens_traces` (about 6 minutes on
   15 workers; `HARNESS_LENS_LIMIT=20` for a quick pass, `HARNESS_WORKERS` for the pool size,
   `HARNESS_VERBOSE=1` for one line per lens).
5. **Noise model validation.** Done: `noise_check` (`harness/data/noise.py`) measures `camera.py`'s
   variance law var(e) = e + σ_r² on real raw bursts instead of quoting it: the SIDD DNG mirror, two cameras,
   Google Pixel at ISO 100, 400, 1600, 6400 and iPhone 7 at ISO 100, 200, 800, 1600, 60 of the 150 frames
   of each burst (17.6 GB pulled of the mirror's 234.5 GB; the Pixel raws are 10-bit, white 1023, black 63).
   The Pixel first:
   Method: per-pixel temporal mean and variance per CFA position (streaming Welford, so a 150-frame burst
   never sits in memory); frames whose plane mean is off the burst by more than 1 % or 5 robust σ are
   dropped; every kept frame is scaled to the burst mean of its plane (lamp drift and flicker add variance
   ∝ mean², per-pixel noise does not); pixels where the mean image has a gradient above 2× the fitted
   noise are excluded (motion between frames turns an edge into temporal variance, a flat pixel cannot
   move); the rest are binned by mean (48 bins), the median variance per bin is divided by the
   chi-square median factor (1 − 2/(9(n−1)))³ and fitted by a count-weighted line over 2 %–70 % of the
   range: gain = 1/slope e-/DN, read noise = √intercept × gain, clip = (white − black) × gain. The
   distribution check (excess kurtosis and 4σ tail of the standardised residuals, 0 and 6e-5 for
   Poisson + Gaussian) and the row / column noise ratios (1 = no line-correlated noise) run on the
   flat pixels of eight kept frames.
   Per instance (median over the four CFA planes):
   - ISO 100 (instance 0063, scene 003): gain 4.86 e-/DN, read noise 7.2 e- (1.5 DN), clip 4,662 e-, R² 0.9987, curvature share 0.009, excess kurtosis 0.10, 4σ tail 1.1e-04, row / column noise ratio 1.23 / 1.24, flat pixels 82 %, edge variance ratio 2.6, 1 of 60 frames dropped, lamp drift -188 σ of a frame mean.
   - ISO 400 (instance 0130, scene 006): gain 1.23 e-/DN, read noise 3.1 e- (2.5 DN), clip 1,185 e-, R² 0.9973, curvature share 0.009, excess kurtosis 0.08, 4σ tail 1.0e-04, row / column noise ratio 1.27 / 1.05, flat pixels 95 %, edge variance ratio 1.1, 0 of 60 frames dropped, lamp drift -58 σ of a frame mean.
   - ISO 1600 (instance 0064, scene 003): gain 0.31 e-/DN, read noise 2.2 e- (7.2 DN), clip 300 e-, R² 0.9994, curvature share 0.021, excess kurtosis 0.06, 4σ tail 1.2e-04, row / column noise ratio 1.51 / 1.06, flat pixels 96 %, edge variance ratio 1.1, 1 of 60 frames dropped, lamp drift -50 σ of a frame mean.
   - ISO 6400 (instance 0108, scene 005): gain 0.08 e-/DN, read noise 2.0 e- (26.8 DN), clip 72 e-, R² 0.9998, curvature share 0.017, excess kurtosis 0.09, 4σ tail 1.7e-04, row / column noise ratio 3.97 / 1.00, flat pixels 100 %, edge variance ratio 1.2, 9 of 60 frames dropped, lamp drift -10 σ of a frame mean.
   Across ISO: log gain vs log ISO slope -1.00 (shot noise ∝ gain predicts −1);
   unity-gain ISO 495; read noise is 1.5 DN at ISO 100 and 2.5 DN at ISO 400, under three codes of the
   10-bit ADC, while in electrons it falls from 7.2 e- at ISO 100 to 2.0 e- at ISO 6400 (26.8 DN), so at
   base ISO the ADC quantisation, not the pixel, sets the floor and `camera.py`'s σ_r is a per-ISO number. The clip
   at ISO 100 (4,662 e-, 9.3 stops over the read noise) is the 10-bit container's
   ceiling, a lower bound on the pixel's full well.
   Verdict for the Pixel: linear variance True (R² ≥ 0.997, curvature share ≤ 0.021, model vs
   measured bin variance median error ≤ 0.013); shot noise scales with gain True;
   heavy tails False (excess kurtosis 0.06–0.10); row noise present True
   (ratio 4.0 at ISO 6400, a real line-noise
   term `camera.py` does not model), column noise present False. 11 of
   240 frames dropped as outliers; fits on the flat 82–100 % of pixels.
   Output: `data/derived/noise_check.json` (method, per-instance and per-plane bins, fits on all and on flat
   pixels, dropped frame names, ISO scaling, verdict). `camera_presets` adds one preset per measured camera,
   `measured_google_pixel` (alias `phone_measured`, `"preset": "derived:phone_measured"`) and `measured_iphone_7`
   (below): the Pixel preset is clip 4,662 e- and read noise 7.2 e- at base ISO, 10-bit, read noise per ISO in its sensor block, optics 4.67 mm f/2.0 from the SIDD DNG Exif, pixel pitch 1.55 µm hand-set from Google's spec (step 14). Rerun:
   `python -m harness data derive --ids noise_check` (about 33 s per burst, four in parallel;
   `HARNESS_NOISE_FRAMES`, `HARNESS_WORKERS`, `HARNESS_VERBOSE=1` for one line per 20 frames).
   Second camera, iPhone 7 (the mirror's instances 0088, 0134, 0068, 0032, 0030 at ISO 100, 100, 200, 800, 1600,
   14.1 GB with the failing burst; instance 0028 at ISO 100 fails: a dark scene, at most 3 usable bins in the fit
   window where 4 are needed, plane means 37–91 DN above black; the raws are 12-bit, white 4095, black 528):
   - ISO 100 (instance 0088, scene 004): gain 0.39 e-/DN, read noise 2.0 e- (5.2 DN), clip 1,385 e-, R² 0.9988, curvature share 0.009, excess kurtosis 0.01, row / column noise ratio 1.22 / 1.04, flat pixels 83 %, 0 of 60 frames dropped.
   - ISO 100 (instance 0134, scene 006): gain 0.39 e-/DN, read noise 1.9 e- (5.0 DN), clip 1,387 e-, R² 0.9997, curvature share 0.011, excess kurtosis 0.02, row / column noise ratio 1.13 / 1.09, flat pixels 89 %, 0 of 60 frames dropped.
   - ISO 200 (instance 0068, scene 003): gain 0.22 e-/DN, read noise 1.9 e- (8.4 DN), clip 790 e-, R² 0.9993, curvature share 0.053, excess kurtosis -0.00, row / column noise ratio 1.27 / 1.06, flat pixels 91 %, 0 of 60 frames dropped.
   - ISO 800 (instance 0032, scene 001): gain 0.22 e-/DN, read noise 1.7 e- (7.6 DN), clip 776 e-, R² 0.9987, curvature share 0.204, excess kurtosis -0.04, row / column noise ratio 1.80 / 1.01, flat pixels 98 %, 0 of 60 frames dropped.
   - ISO 1600 (instance 0030, scene 001): gain 0.39 e-/DN, read noise 1.9 e- (4.9 DN), clip 1,388 e-, R² 0.9983, curvature share 0.009, excess kurtosis 0.01, row / column noise ratio 1.32 / 1.08, flat pixels 88 %, 0 of 60 frames dropped.
   Across ISO the iPhone's gain is flat: log gain vs log ISO slope −0.05 (0.39, 0.39, 0.22, 0.22, 0.39 e-/DN at
   ISO 100, 100, 200, 800, 1600), read noise 1.7–2.0 e-, clip 776–1,388 e-. The gain takes two values only, 0.39
   (ISO 100 and 1600) and 0.22 (ISO 200 and 800), which no analog-gain ladder produces.
   A gain that does not fall with ISO means these DN values are not the sensor's ADC output at a stepped analog
   gain (the mirror's DNG writer may rescale the data, or the ISO tag may not track the gain; the data cannot say
   which), so the electron units are relative to the DNG scale, unity-gain ISO is undefined and the preset
   `measured_iphone_7` (clip 1,385 e-, read noise 2.0 e-, 12-bit) is a floor to beat at
   ISO 100, not a sensor characterisation; `iso_scaling_is_analog_gain` is False on it and True on the Pixel.
   Per camera (`verdict.per_camera`): Google Pixel linear variance True on 4/4 instances, shot noise
   scales with gain True; iPhone 7 linear variance False on 4/5 instances (R² ≥ 0.998; the
   ISO 800 instance is a 4-bin fit with curvature share 0.20), shot noise scales with gain False. The
   overall flags are the AND over cameras, so `linear_variance` and `shot_noise_scales_with_gain` now read False
   with the per-camera split beside them.
6. **Expanders** (each is one function in `apis.py`, pure listings, downloads go through the
   ledgered collector with `--max-bytes`). `rpu_index` turns the raw.pixls.us JSON index into one
   sha256-verified CC0 raw per camera (926 cameras, 25 GB; `--param one_per_model=0` for all 1,874
   files; 3 fetched as the check). `dataverse_dataset` walks a Dataverse dataset through the native
   API (`server`, `doi`, `pattern`, `limit`), which publishes size and MD5 per file, so every
   download is verified: it serves `rawnind` (UCLouvain, 2,845 raw files, 120 GB), `snic` (Harvard,
   14 files, 473 GB) and `spring-flow` (DaRUS, 17 zips, 289 GB). `gcs_bucket` lists a public Google
   Cloud Storage bucket anonymously through the storage JSON API (`bucket`, `prefix`, `pattern`,
   `limit`; MD5 per object, no gsutil): `hdr-plus-burst` is the 153-burst HDR+ subset (4,056
   objects, 39.1 GB) and `hdrplus-bursts` the full 3.7 TB prefix. Checked 2026-10-27 with two
   bounded pulls: Spring's `test_cam_data.zip` (3.8 KB, 10 intrinsics files) and one HDR+ burst's
   `rgb2rgb.txt` and `timing.txt`, all MD5-verified in the ledger. Not added: the SIDD official
   page no longer carries Google Drive ids and its download host (130.63.97.225) closes plain
   connections, so `sidd-official` stays `manual` and the DNG bursts come from the Hugging Face
   mirror. Sites that refuse plain fetches stay `page-only` or `manual`; nothing is circumvented.
7. **Asset scale for scenes.** The CC0 libraries are catalogued in full and pulled where the harness
   has a use for them. `polyhaven_catalog.json`: 997 Poly Haven HDRIs, 869 texture sets,
   521 models, 2037 ambientCG materials and 434 ambientCG HDRIs
   (4,858 assets; the texture, model and material pulls stop at one asset each until a renderer consumes
   them, `--param limit=0` fetches a library). The HDRIs are consumed now: all 997 Poly Haven maps at
   1k (1.6 GB) are on disk and `hdri_stats` (`harness/data/hdri.py`, a numpy Radiance reader) measures the
   luminance range of each real environment, solid-angle weighted. Scene range (p0.1 to p99.9 of the
   sphere): 10.0 stops (p10 7.1, p90 14.1, n 997); with the brightest source and the deepest shade (max over p0.01):
   16.7 stops (p10 9.8, p90 24.7, n 997); the peak sits 11.6 stops above the median pixel. Outdoor 9.1 stops (p10 6.6, p90 12.1, n 719);
   indoor 12.8 stops (p10 10.3, p90 16.0, n 278); natural light 9.3 stops (p10 6.8, p90 12.6, n 760); artificial light 13.1 stops (p10 10.6, p90 16.6, n 148);
   night 9.5 stops (p10 6.5, p90 13.9, n 46); pure skies 8.1 stops (p10 5.7, p90 10.3, n 299). The generator's 14-step wedge spans 11.0 stops (0.001 to 2.0), which covers the median
   scene range but not the source-to-shade range of a sunlit exterior; `best_case_earth` holds
   15.0 stops (FW 50,000 e-, read 1.5 e-) and the measured phone 9.3 stops, so the camera baseline clips
   the brightest source in most outdoor maps and the harness's `highlight_loss_fraction` gate is the one that
   bites, as the demo shows. Output: `data/derived/hdri_stats.json` (method, per-map rows with Poly Haven
   categories, attributes and bracket count, summaries per category and attribute). Rerun:
   `python -m harness data derive --ids hdri_stats` (about 997 maps in under a minute on 16 workers;
   `HARNESS_HDRI_LIMIT`, `HARNESS_WORKERS`, `HARNESS_VERBOSE=1`). Scenes with exact flow stay with the
   harness's own generator: Infinigen (pip, bundles Blender) and Kubric (docker) are catalogued as generators
   with their outputs, not installed here, because the per-scene render cost has to be measured on the target
   machine before either is a scalable source.

8. **Second camera for the noise law.** Done, folded into step 5: the iPhone 7 bursts of the SIDD mirror
   (instances 0068, 0032, 0030, 0028, then 0088 and 0134 in step 12; 14.1 GB) run through the same `noise_check`;
   the law holds per ISO (R² ≥ 0.998) but the gain is flat across ISO (slope −0.05 over 100–1600), so the preset
   `measured_iphone_7` is a floor at ISO 100 with relative electron units, the verdict is per camera, and `camera_presets`
   writes one `measured_<camera>` preset per camera in `noise_check.json` (`measured_cameras` lists them).
9. **Real capture constraints.** Done: `capture_stats` (`harness/data/dng.py`, a struct-only TIFF/DNG tag
   reader, no library) reads the first raw frame and lens shading map of all 153 HDR+ bursts (3.3 GB of the
   39.1 GB subset, `--param 'pattern=(payload|lens_shading_map)_N000[.](dng|tiff)$'`). Per frame:
   ExposureTime, ISO and FNumber from the Exif IFD or, on the older Nexus files, IFD0; EV100 = log2(N²/t) −
   log2(ISO/100) and the scene luminance L = 2^EV100 × K/100 with K = 12.5 cd/m² (ISO 2720); the DNG
   NoiseProfile (scale, offset per plane) converted to gain 1/((white − black) × scale) e-/DN and read noise
   √offset / scale e-; the shading map's corner-to-centre gain in stops (Deflate-compressed maps included).
   141 of 153 frames carry an exposure time and 137 an ISO (16 of the 30 frames without
   Make/Model, 2014 bursts, hold zeros in one or both tags). What the
   phone chose: exposure median 1.7 ms (p10 0.4 ms, p90 30.3 ms, max 100.0 ms), 16 % of frames at
   1/60 s or longer, 9 % at 1/30 s, 2 % at 1/15 s; ISO median 60 (p10 40, p90 513, max 2056),
   5 % at ISO 800 or above; f-numbers 2.0, 2.4; scene luminance median 634 cd/m² (p10 5.7,
   p90 2904, max 9731; inferred from the AE choice through ISO 2720 with K = 12.5 assuming standard metering, and
   HDR+ deliberately underexposes to protect highlights, so the figure is biased high). Against luminance the log-log slopes are -0.63 for exposure time and -0.38 for
   ISO (an identity at a fixed f-number: exposure time × ISO must scale as 1/L, so the two slopes sum to −1 by
   construction): HDR+ pays for a darker scene mostly with time, and gain only
   past that. Noise profiles on 62 frames (the other 91, Nexus 5, most Nexus 6 and the unlabelled frames, carry none): the manufacturer's
   profile gain divided by the SIDD-measured gain at the same ISO is Pixel 0.91 (n 5, 0.90–0.97); Pixel XL 0.90 (n 9, 0.90–0.97),
   so the profile is a conservative envelope within 10 % of the measurement; its read noise
   (12.0 e- at the frames' ISO, median 50–51) sits above SIDD's 7.2 e- at ISO 100, the lowest ISO measured. Lens
   shading: corner falloff median 1.49 stops (p10 1.44, p90 1.60), full-map range median 1.80 stops, so
   the raw corners hold 1.5 stops less signal than the centre before the ISP's correction and corner SNR is
   where a phone loses first. Devices: Nexus 6 56 (noise profile on 10); Nexus 6P 30 (noise profile on 30); ? 30; Nexus 5 15; Pixel XL 9 (noise profile on 9); Nexus 5X 8 (noise profile on 8); Pixel 5 (noise profile on 5).
   Output: `data/derived/capture_stats.json` (method, summary, per-device rows, SIDD cross-check, per-frame
   rows). Rerun: `python -m harness data derive --ids capture_stats` (153 frames in about 20 s, no rawpy).
10. **Display gaps.** Partly done. Three format rows at the ITU design viewing distance
   (`hdtv-bt2246-design`, `uhdtv-bt2246-design`, `8k-bt2246-design`; Rep. ITU-R BT.2246-9 Table 1, from Rec.
   ITU-R BT.1845): the ITU defines the distance as one pixel per arc-minute, 60 ppd, which is 3.18 / 1.59 /
   0.80 H for 1080 / 2160 / 4320 lines; the table's rounded 3.1 / 1.5 / 0.75 H give
   58.4 / 56.5 / 56.6 ppd at any panel size, and that is what the rows carry. They are formats,
   not products: no peak luminance, and a TV at its design distance sits below the 94 ppd threshold by
   definition. The earlier rows cited "Table 2"; the PDF says TABLE 1, corrected in `vision_thresholds.csv`,
   `derive.py` and the derived JSON. Step 13 adds the Meta Quest 3 row (`meta-quest-3`, 25 ppd stated): Meta's compare page is
   script-rendered and the Wikipedia infobox gives the per-eye panel (2064×2208) but no ppd or FOV, so the
   angular resolution and the 110°×96° FOV are quoted from UploadVR's spec table, a review page, not the
   vendor. Still missing: a vendor TV row.
11. **Lens shading in the camera model.** Done: `CaptureConstraints.shading_corner_stops` (default 0, so every
   earlier result is bit-identical) and `shading_corrected` (default True). `simulate_frame` multiplies the radiance
   by 2^(−s·r²), r² normalised to 1 at the corners (the small-angle form of cos⁴), before the shot noise, and when
   corrected divides it back out after the sensor: brightness restored, corner noise multiplied with it, corner SNR
   down by 2^(s/2), 1.7× at 1.5 stops. Fit of that shape to all 153 HDR+ maps (least squares on −log2 of the map,
   through the centre): R² median 0.86 (min 0.75), largest residual median 0.39 stops (p90 0.45); the least-squares
   quadratic puts 1.77 stops at r = 1 while the measured corner is 1.49, so the real maps fall faster mid-field and
   flatten toward the corner. The presets carry the measured corner value, which under-states the mid-field loss by
   up to about 0.4 stops. `camera_presets` sets it per camera family from `capture_stats.json` (`MODEL_TO_SIDD`):
   `measured_google_pixel` 1.78 stops (median of Pixel 1.85, n 5, and Pixel XL 1.70, n 9); `measured_iphone_7` 0,
   with `shading_source` saying no map exists for it; the sourced sensor+lens presets carry 0 for the same reason.
   Tests: `tests/test_shading.py` (corner geometry, identical pipeline at 0 stops, brightness restored with a corner
   variance ratio near 1.9 at one stop, uncorrected darkening). SPEC.md §3 has the row.
12. **iPhone 7 at ISO 100.** Done: two brighter ISO 100 bursts (instances 0088 scene 004 and 0134 scene 006, 5.0 GB)
   fit (R² 0.9988 / 0.9997, gain 0.39 e-/DN, read noise 2.0 e-, clip 1,385 e-), so the 0028 failure was the dark
   scene, not the ISO. Five instances give log gain vs log ISO slope −0.05 and only two gain values (0.39 at ISO 100
   and 1600, 0.22 at ISO 200 and 800): the relative-units verdict stands, `measured_iphone_7` now sits at ISO 100
   (clip 1,385 e-, read 2.0 e-, 12-bit), 9 of 10 bursts fit overall. Rerun: `python -m harness data collect --ids
   hf-sidd-dng-mirror --param 'pattern=data/train/(0088|0134)_DNG' --max-bytes 6e9`, then `derive --ids noise_check`,
   `derive --ids capture_stats`, `derive --ids camera_presets`, one at a time in that order (the presets read both
   JSONs, and `--ids` runs in manifest order, not in the order given).
13. **Quest 3 display row.** Done: `meta-quest-3` (headset, 2064×2208 per eye, 25 ppd stated, FOV 110°×96°); the
   table holds 8 displays + 4 headsets, 3 of 12 reach 94 ppd. Meta's compare page is script-rendered (no spec text in
   a plain fetch with the harness UA, so it stays page-only, never circumvented); the Wikipedia infobox gives the panel
   but no ppd; UploadVR's spec table (`uploadvr_quest3_specs.html` under `vendor-spec-pages`, ledgered) gives the
   25 ppd and the FOV, a review figure, not the vendor's, as the row's condition says.
14. **Independent review fixes.** Done: a reviewer agent read SPEC, README, this plan, the harness and the derived JSON
   against the claim they make and reported nine findings; each was verified and resolved:
   - Exposure and aperture were absent from the noise model (electrons came from a fixed `sensitivity_e_per_s`).
     `camera.py` now has two modes: relative (default) exposes the scene's peak radiance at `white_fill` × full well;
     absolute (`scene_white_cd_m2` > 0) computes e⁻ = L·π·T/(4N²)·k·t·A_pix·QE with k = 4.09 × 10¹⁵ photons/(lx·s·m²),
     plus `exposure_for_peak` and `saturation_iso` (ISO 12232: 99 for the Pixel at its ISO 100 clip). Tests:
     `tests/test_camera_model.py`.
   - The baseline was not best-case exposed for the tone gate: it clipped the wedge top, so the demo's camera failed at
     tone_color for a reason of the harness's own making. Fixed by the relative mode; `dr_steps` became "ties or beats
     camera" because the 11-stop wedge sits inside a correctly exposed best-case camera's range.
   - The sourced frontier ranked by engineering DR, which put a 12-bit linear model of the GSENSE6060BSI (95 ke⁻, 3 e⁻,
     a 90 dB dual-gain HDR quote) first, with default optics and no label. `camera_presets` now reports
     `dr_stops_simulated` = log2(FW / √(σ_r² + LSB²/12)) per preset, ranks the frontier by it (ORCA-Quest 2, 14.50 stops)
     and labels `optics_source`.
   - The extrapolated KADID edge-overshoot fit (0.223, CI 0.19 to 0.26, n 405) was the pass line of `claim_sourced.json`.
     A supported but extrapolated fit is now reported beside the gate with a note; 0.02 stays the pass line.
   - A skipped check (no human study) still yielded PASS. `gate.overall` now returns `INCOMPLETE (skipped: …)` and the
     CLI exits 2; the demo's truth-as-candidate reads INCOMPLETE (skipped: composition).
   - The 634 cd/m² scene luminance carried no metering caveat; step 9 and SPEC §3 now say it is inferred through
     ISO 2720 assuming standard metering and is biased high because HDR+ underexposes.
   - No black pedestal and no row noise: `black_pedestal_e` (8 σ_r) keeps shadows unbiased, `row_noise_e` is derived
     from the measured row-noise ratio (Pixel 0.055 e⁻ at ISO 100, a lower bound), the well clips before read noise.
   - The `analog = slope < −0.7` split was unsourced: commented as a heuristic label midway between the pure
     analog-gain slope −1 (Pixel −1.00) and the digital-scaling slope 0 (iPhone 7 −0.05).
   - Nits: the ITU 8K row carries 56.6 ppd where recomputing from the rounded 0.6071 m gives 56.5 (rounding of the
     table's ratio); §6 counts use the last attempt per (source, path) and the ledger holds 12 duplicate rows from
     re-attempts, so its file counts are of distinct paths.
   - Found while checking the above: both measured presets inherited the generic phone preset's optics and 1.2 µm pixel
     pitch, so the Pixel's `saturation_iso` computed as 59, not the 99 the `camera.py` check states. `camera_presets` now
     reads f-number and focal length from the Exif of the SIDD instance the preset is built from (Pixel 4.67 mm f/2.0,
     iPhone 7 3.99 mm f/1.8, `optics_source`) and takes the pixel pitch from a hand-set, sourced table
     (`PHONE_PIXEL_PITCH_UM`: Pixel 1.55 µm from Google's spec, iPhone 7 1.22 µm from the Chipworks teardown,
     `pixel_pitch_source`); neither dataset's DNGs carry a FocalPlaneResolution tag. The Pixel now gives 98.6; the
     iPhone 7's 206 is not a sensor figure, since its electron units are relative (step 9).
   Outcome: the suite passes (96 tests), the demo's truth-as-candidate is INCOMPLETE (skipped: composition) and the
   camera-as-candidate fails at resolution (`mtf50_cpp` ties the camera and must beat it); the best-case camera no longer
   clips the wedge.

## 5. Gaps (honest list)

- The fitted edge-overshoot gate (0.22) is a crowd *tolerance* under KADID-10k's conditions
  (side-by-side DCR, 512×384 PNG, uncontrolled displays and distances), not a detection threshold
  on a mastering display, and it is extrapolated: KADID's mildest sharpening already has a median
  overshoot of 0.56 and a mean rating of 3.58, so no rated image sits in the imperceptible range.
  `DEFAULT_GATES` keeps 0.02 as the stricter engineering target; the data here neither support
  nor refute it. The estimator also reads 0.06–0.31 on pure noise (AWGN levels 1–5) and
  0.13–0.36 on posterisation, so a noisy or quantised candidate fails this gate for that reason,
  not for halos; `shadow_snr_db` and `banding_score` say which.
- The banding gate is sourced, not fitted: the only rated quantization ladders on disk (KADID #22,
  TID2013 #7) grade whole-image posterisation, and the smooth-region step metric explains little of
  their ratings (ρ −0.31 / −0.24). The fit is kept in `gates_fitted.json` with its failed criteria
  so nobody re-reads it as a threshold. The pair estimator also reads 0.03–0.09 on Gaussian blur
  of weak edges and 0.01–0.02 on noise σ 0.002–0.005, so a blurred or noisy candidate can trip it.
- The shift check is proxy-based. No AI-image set rates halo or banding on its own and none ships a
  reference, so the fitted gates were not re-evaluated; two no-reference proxies were. The halo
  proxy fails its calibration on natural images (0.09 of the pair metric, ρ 0.37) and reads
  sharpness on generated ones (ρ +0.34 with quality MOS), so the fitted 0.22 gate has no
  no-reference form and its drift is unmeasured. The banding proxy's step reads the 8-bit floor on
  photographs and generated images alike; only banded area separates generators (ρ −0.49 with
  quality MOS, carried by Glide). Both proxies are read against the pristine distribution, never as
  thresholds. A real re-evaluation needs generated images rated for halo and banding separately,
  which no public set provides.
- Hasselblad X2D and Fujifilm GFX100 II sensor rows are missing: vendor pages are JS-only and the
  cameras are absent from the Photons-to-Photos tables; Photons-to-Photos itself is cite-only
  (`scrape-tos-unclear`).
- No vendor TV display row: Samsung/LG pages are script-rendered or refuse plain fetches; the Quest 3 ppd
  (25) is a review-page figure (UploadVR) because Meta's own page is script-rendered. The three `*-bt2246-design` rows are formats at the ITU design distance, not products, with no
  peak luminance; they carry the table's rounded ratios (56.5–58.4 ppd), not the 60 ppd of the definition.
- `datapointai/text-2-image-human-preferences-2m` (best raw Bradley–Terry data) is gated with
  manual approval; `Vchitect/VBench_human_annotation` is gated (`HF_TOKEN`): its listing failure
  is ledgered as `<expand>` and shows in the table below until a token is set.
- Several catalogue entries are dead or unreachable (BID, RENOIR, SID-sharpened, NAM-CC) and are
  kept only as catalogue rows.
- Licence conflicts to resolve before any redistribution: HPDv3 (MIT vs CC BY-NC-SA),
  EvalMuse (BSD vs Apache), PIPAL (unstated; mirror tag unofficial).
- The 120 ppd upper bound is a practical ceiling from the display rows, not a measured JND.
- The preference validation is generator-level, not image-level, except HPDv2: Rapidata has no
  rater ids (its `userScore` moves agreement with the majority only from 0.55 to 0.57 across
  its quartiles), open-image-preferences has
  two generators, and only GenAI-Arena and GenAI-Bench overlap (17 generators). HPDv2's ten
  annotators rank whole sets, so its 153,000 pairs are not independent and its ECE (0.187) reads
  that; the arena set has 52 test votes, so its ECE (0.170) is noise. No public set votes on
  camera output against generated output, which is the harness's own pair.
- The traced lenses are patent examples, design intent rather than built lenses measured on a
  sensor, and the trace is monochromatic at the d line. 141/1180
  field points are untraceable under paraxial pupil aiming (wide-angle designs off axis) and are
  recorded as null; every field is read at the on-axis plane, so edge MTF50 reads field curvature;
  the DeepLens JSON build carries no vignetting apertures; polychromatic MTF is not computed.
  The 9 files optiland cannot read or does not support are listed by name in
  `lens_traces.json`.
- The noise check is one camera (Google Pixel) on three scenes at four ISO values, 60 of 150 frames per
  burst, 10-bit raws: the law and its ISO scaling are measured, the pixel's full well is not (the
  container clips at 4,662 e- at ISO 100). Frame 0 of the ISO 100 burst is 7 % dark and
  9 of 60 frames at ISO 6400 sit more than 1 % off the burst; all are dropped, not
  explained. Scene edges move between frames (2.6× the fitted variance at ISO 100)
  and are masked, so the fit is on 82–100 % of pixels. Read noise in electrons is
  ISO-dependent, so a preset's σ_r holds at one ISO. Row noise enters `camera.py` as a per-row Gaussian from the ISO 100 ratio (1.23, a
  lower bound of 0.055 e⁻ applied to the read noise); the ISO 6400 ratio (4.0×) is not represented at that ISO. No official SIDD noise-level file was used; LRID,
  CRVD and RawNIND are catalogued, not pulled.
- The HDRI ranges are relative radiance at 1k: Poly Haven maps carry no absolute calibration, so no cd/m² is reported, and at 1024×512 the sun is averaged over a 0.35° pixel, so `dr_stops_full` and `peak_over_median` are lower bounds. The maps are themselves bracketed captures (median 13 EVs captured), so the range cannot exceed the bracket, and 1 % of maps contain clipped-black pixels. The library is curated for usable lighting, not sampled from the world, so the per-category medians describe Poly Haven, not Earth.
- The iPhone 7 noise numbers are relative: the gain does not fall with ISO (slope −0.05 over five bursts) and
  takes only two values (0.22 and 0.39 e-/DN) that do not follow the ISO tag, so the electron units of
  `measured_iphone_7` are tied to the mirror's DNG scale, not to the pixel. The dark ISO 100 burst 0028 (at most
  3 usable bins in the fit window where 4 are needed, plane means 37–91 DN above black) does not fit; the two
  brighter ISO 100 bursts (0088, 0134) do. The camera, ISO and scene
  per instance are what the mirror's metadata says; the official SIDD site stays manual.
- The lens shading model is one quadratic in the radius with the measured corner value: it explains 86 % of the
  HDR+ maps' variation and misses the mid-field by up to 0.4 stops; it is isotropic and colour-independent (the
  maps carry four CFA planes); only the Pixel family has a value, the iPhone and the sourced presets carry 0.
- The HDR+ capture statistics describe one app's choices on 2014–2016 phones: 12 of 153 frames hold
  zero exposure or ISO tags (all among the 30 frames with no Make/Model), the scene luminance comes from the EXIF exposure through
  K = 12.5 cd/m² (a calibration constant that differs by up to a third of a stop between makers), and the
  first frame's exposure is the burst's per-frame choice before the merge, so it bounds motion blur per
  frame, not the merged result. The noise-profile cross-check covers the Pixel family only (n 14).
- Dual-gain HDR sensors are not simulated: a sensor whose vendor range needs two readout gains (GSENSE6060BSI,
  90 dB) is modelled as one linear 12-bit gain, so its simulated DR (13.66 stops) sits below its engineering DR
  (14.95) and it no longer heads the frontier; `dr_stops_simulated` says what the harness realises.
- The radiometric constant k = 4.09 × 10¹⁵ photons/(lx·s·m²) is the 555 nm value: for a colour-filtered pixel under
  a broadband source the photons per lux differ by up to about 1.5× (the broadband excess and the pass band roughly
  cancel), so absolute-mode electron counts and `saturation_iso` carry that uncertainty; QE 0.6 and transmission 0.9
  are typical values, not sourced per sensor.
- The frontier is by simulated DR, with the harness default optics where no lens row exists for the format
  (`optics_source`); the ORCA-Quest 2's 7 ke⁻ well caps its white SNR at 38 dB, so "best case" means best range, not
  best SNR at white; a claim that cares about white SNR should name a deeper-well sensor.
- Row noise is a lower bound: the ratio was measured over flat patches of every brightness and is applied to the
  read-noise floor, and the ISO 6400 ratio (4.0×) is not represented in the ISO 100 preset.

## 6. Sources and status (generated)

Refresh with `python -m harness data plan --out DATA_PLAN.md` after any collect.

<!-- plan:begin (generated by `python -m harness data plan`) -->

**122 sources** (S: 46, M: 21, L: 41, XL: 14); fetchable from here: 87, planned bytes 6,424.8 GB; on disk: 1799 files, 33.79 GB across 50 sources; 2 file(s)/listing(s) failed on the last attempt.
Catalogue: 122 datasets (A: 25, B: 38, C: 35, D: 24), 65 of them have a manifest source.

### Tier S

| id | access | licence | size | feeds | on disk |
|---|---|---|---:|---|---|
| `cie-colorimetry` | api | CC BY-SA 4.0 (each table's _ | 5 MB | spectral rendering of scene.colors and the camera's per-channel integration; exact Lab of any reflectance unde | 20 f / 0.6 MB |
| `zenodo-xrite-colorchecker` | api | not specified on Zenodo (X-R | 0 MB | known Lab of the colour-patch panel: the delta_e_mean gate is measured against these numbers | 6 f / 0.0 MB |
| `babelcolor-average` | direct | copyright Danny Pascale; fre | 0 MB | spectral reflectance of each patch, so patches can be rendered under any CIE illuminant and the Lab recomputed | 1 f / 0.2 MB |
| `zenodo-munsell-matt` | api | not specified on Zenodo (UEF | 3 MB | large known-spectra set for gamut coverage and metamerism tests of the tone_color gate | 3 f / 1.2 MB |
| `zenodo-labsphere-srs99` | api | CC BY 4.0 | 0 MB | white_fill / highlight reference: what a perfect diffuse white reflects per wavelength | 2 f / 0.0 MB |
| `zenodo-camspec-jiang2013` | api | CC BY-NC-SA 4.0 (non-commerc | 0 MB | per-channel integration of the camera baseline instead of the three fixed wavelengths in CaptureConstraints.wa | 4 f / 0.0 MB |
| `zenodo-lfdd-flicker` | api | CC BY 4.0 | 0 MB | sanity range for the flicker gate threshold (temporal gate) | 2 f / 0.0 MB |
| `gen-infinigen` | pip | BSD-3-Clause | ? | scalable replacement for the scene.py toy world: licence-clean realistic ground truth at any resolution | pip |
| `gen-kubric` | docker | Apache-2.0 | ? | exact optical-flow ground truth for the temporal gate on physically simulated motion | docker |
| `pkg-colour-science` | pip | BSD-3-Clause | 9 MB | cross-check of derived/cie_tables.npz and derived/colorchecker.json; no network needed | pip |
| `laval-photometric-hdr` | manual | research licence, request fo | ? | the only absolute-luminance environment set found: needed if the highlight/DR gates are ever stated in cd/m2 r | manual |
| `itu-bt2246-9` | direct | ITU-R publication, free down | 3 MB | vision_thresholds.csv rows itu-design-distance-*, de-uhdtv-10bit, banding-contrast-ratio, hfr-*, motion-blur-* | 1 f / 3.1 MB |
| `itu-bt500-15` | direct | ITU-R publication, free down | 4 MB | preference.py protocol reference; vision_thresholds row bt500-distance-8k | 1 f / 3.9 MB |
| `itu-bt2390-12` | direct | ITU-R publication, free down | 4 MB | vision_thresholds row pq-12bit-no-banding; future HDR tone gate | 1 f / 4.0 MB |
| `itu-bt1702-3` | direct | ITU-R publication, free down | 1 MB | vision_thresholds rows harmful-flash-*; temporal.py flicker safety ceiling | 1 f / 0.7 MB |
| `itu-bt2100-3` | direct | ITU-R publication, free down | 1 MB | future: HDR display delivery path in display.py | 1 f / 0.7 MB |
| `itu-bt2020-2` | direct | ITU-R publication, free down | 1 MB | colour conversion constants for tone/colour metrics | 1 f / 0.6 MB |
| `ebu-tr050` | direct | EBU technical report, free d | 1 MB | vision_thresholds rows tv-distance-ratio, hfr-quality-gain | 1 f / 1.3 MB |
| `netflix-cambi-doc` | direct | BSD-2-Clause-Patent (VMAF) | 0 MB | vision_thresholds row cambi-tvi; artifacts.py banding metric calibration | 1 f / 0.0 MB |
| `arxiv-2410-06068` | direct | arXiv (see paper licence) | 33 MB | VISIBILITY_PPD lower bound (gate.display_ppd) | 1 f / 32.9 MB |
| `basler-emva-overview` | direct | vendor datasheet, cite-only | 1 MB | bulk industrial-sensor rows for sensors.csv (table extraction pending) | 1 f / 1.3 MB |
| `kinetix-datasheet` | direct | vendor datasheet, cite-only | 1 MB | sensors.csv rows teledyne-kinetix-* | 1 f / 0.6 MB |
| `qhy600-datasheet` | direct | vendor datasheet, cite-only | 3 MB | sensors.csv row sony-imx455-qhy600 | 1 f / 3.5 MB |
| `flir-emva-imx183` | direct | vendor page, cite-only | 0 MB | sensors.csv row flir-imx183; template for every other FLIR model page (same URL pattern) | 1 f / 0.0 MB |
| `europepmc-viewing-distance` | api | Europe PMC REST (abstract me | 0 MB | vision_thresholds rows monitor-distance, phone-distance -> display.distance_m | 2 f / 0.0 MB |
| `pmc-ferry-porter` | direct | CC BY (PMC open access) | 0 MB | vision_thresholds row cff-saturation | 1 f / 0.0 MB |
| `wikipedia-camera-display-pages` | direct | CC BY-SA 4.0 | 2 MB | sensors.csv source_url_2 / quote_2; vision_thresholds rows de-* | 11 f / 2.9 MB |
| `vendor-spec-pages` | direct | vendor/review pages, cite-on | 3 MB | sensors/lenses/displays.csv quotes; re-fetch to re-verify every quote | 17 f / 4.4 MB |
| `photonstophotos-rn` | scrape-tos-unclear | site terms unclear; cite-onl | 0 MB | sensors.csv rows sony-a7r-v, sony-a1-ii, nikon-z8, nikon-z9, canon-r5-ii, canon-r1, leica-m11, phase-one-iq4-1 | scrape-tos-unclear |
| `lens-designs-claff` | direct | MIT (per the archive index) | 6 MB | traced by `lens_traces` (harness/data/optics.py): every .zmx through optiland, refocused on axis, centre Streh | 301 f / 5.8 MB |
| `deeplens-lenses` | direct | Apache-2.0 | 0 MB | traced by `lens_traces` from the JSON prescriptions (the .zmx twins fail in optiland's reader); cross-check of | 5 f / 0.0 MB |
| `pkg-optiland` | pip | MIT | ? | used: harness/data/optics.py does the sequential trace, image solve, FFT PSF (Strehl), FFT MTF and wavefront R | pip |
| `pkg-prysm` | pip | MIT | ? | not used: optiland's FFT PSF/MTF cover the monochromatic Strehl and MTF50; polychromatic MTF stays a gap | pip |
| `pkg-rayoptics` | pip | BSD-3-Clause | ? | not used: optiland reads 293 of 300 Claff files; the rest are listed in lens_traces.json rather than re-read | pip |
| `vesa-displayhdr` | page-only | VESA, page-only | ? | peak_nits sanity for displays.csv; HDR delivery tiers | page-only |
| `dxomark` | manual | proprietary, cite-only | ? | cross-check of P2P-derived DR; manual lookups only | manual |
| `band-2k` | page-only | unstated | ? | MOS 0-100 banding severity + binary patches: the direct calibration set for artifacts.py banding_score | page-only |
| `koniq-plus-plus` | page-only | freely available | 2 MB | defect distributions (blur, artifacts, noise, contrast) per image: multi-label artifact prior | page-only |
| `hf-aigcqa-30k-labels` | api | unknown (ModelScope page not | 4 MB | 14k MOS labels (no image bytes) for AIGCQA-30K: join with ModelScope images | 2 f / 0.5 MB |
| `pal4vst` | page-only | unknown (no licence file per | ? | per-pixel perceptual-artifact segmentation on synthesized images: artifact-map supervision | page-only |
| `hf-vbench-human` | api | MIT | 7 MB | per-dimension pairwise human annotations (temporal flicker, motion smoothness, ...): temporal-gate validation | **1 failed**: HTTPError: 404 Client Error: Not Found for url: https://data |
| `live-flicker` | page-only | UT Austin notice | ? | SSCQE continuous ratings of flicker/motion silencing: the only direct flicker-visibility set found | page-only |
| `live-yt-hfr` | page-only | UT Austin notice | ? | MOS across 24-120 fps versions: frame-rate / judder calibration for temporal.py | page-only |
| `live-yt-banding` | page-only | unknown | ? | MOS for banding in video: video banding threshold | page-only |
| `bvi-artefact` | page-only | unknown | ? | binary labels for 10 video artefact types (no MOS): artifact taxonomy coverage check | page-only |
| `slanted-edge-raws` | pip | n/a | ? | Imatest / MTF Mapper slanted-edge references: validate resolution.py's SFR against a reference implementation | pip |

### Tier M

| id | access | licence | size | feeds | on disk |
|---|---|---|---:|---|---|
| `polyhaven-hdris` | api | CC0 | 1,589 MB | environment lighting for rendered ground-truth worlds; relative (not absolute cd/m²) radiance | 998 f / 1597.6 MB |
| `hf-pickapic-v1` | api | MIT (dataset card) | 1,500 MB | validate the Bradley–Terry preference fit and its bootstrap coverage on real pairwise data | planned |
| `gen-mitsuba3` | pip | BSD-3-Clause | 100 MB | spectral radiance ground truth for the colour-patch and material panels using the CIE tables and measured refl | pip |
| `hf-iqa-dataset-labels` | api | unknown | 100 MB | MOS/DMOS labels for ~20 IQA sets in one table: join key for every Group-A fit | 2 f / 2.1 MB |
| `hf-tid2013` | api | none stated; freely availabl | 1,520 MB | cross-check for fit_gates: 24 distortion types x 5 levels with MOS 0-9; #7 quantization noise (125 images) che | 9 f / 1523.4 MB |
| `cid2013` | api | CC BY 4.0 | 610 MB | camera-captured MOS + sharpness/graininess/lightness/saturation sub-scores: calibrates tone/noise metrics on r | 2 f / 2.5 MB |
| `csiq` | page-only | copyright E. Larson; cite | 390 MB | DMOS for noise/blur/contrast/JPEG ladders (6 distortions): second Group-A fit target | page-only |
| `live-iqa-r2` | page-only | UT Austin copyright notice;  | 720 MB | DMOS 0-100 for JPEG/JP2K/noise/blur/fast-fading: classic sanity set for the artifact gates | page-only |
| `hf-agiqa-3k` | api | MIT (GitHub repo) | 100 MB | MOS quality + alignment on 3k AI-generated images: distribution-shift check of the Group-A thresholds on gener | 2 f / 0.1 MB |
| `hf-aigciqa2023` | api | apache-2.0 (HF tag on IntMeG | 500 MB | labels-only mirror (name, 3 MOS dims); images come from hf-aigciqa2023-images | planned |
| `hf-aigciqa2023-images` | api | apache-2.0 (HF repo tag) | 893 MB | shift_check (DATA_PLAN section 4 step 2), done: NR halo proxy p50 0.016 vs 0.054 on KADID references, rho +0.3 | 6 f / 893.0 MB |
| `hf-artibench` | api | cc-by-nc-4.0 (HF tag) | 500 MB | 1,000 images with artifact bounding boxes + explanations: localisation ground truth for artifacts.py region me | 2 f / 485.4 MB, 1 failed |
| `hf-richhf-18k` | api | unknown (GitHub repo contain | 200 MB | RichHF-18K artifact_map + misalignment_map heat-maps and 5-pt scores (labels only): artifact-region targets | 2 f / 141.5 MB |
| `hf-genai-bench-image` | api | cc-by-4.0 (HF tag on both) | 1,000 MB | 3,192 arena battles (left/right image, vote_type incl. tie/both-bad) for T2I: external BT validation | 2 f / 0.1 MB |
| `hf-genai-arena-human-eval` | api | cc-by-4.0 (HF tag on both) | 200 MB | 350 battles with judge id + anonymity flag: inter-rater agreement check | 2 f / 46.0 MB |
| `hf-mj-bench` | api | apache-2.0 (HF tag); GitHub  | 1,000 MB | binary pairwise preference across quality/safety/alignment perspectives: cross-check set | planned |
| `hf-hpdv2-test` | api | Apache-2.0 (card + GitHub) | 500 MB | HPDv2 test split with raw per-annotator rankings (user_hash): ranking-to-BT conversion test | 2 f / 0.1 MB |
| `hf-q-bench` | api | NOASSERTION on GitHub (S-Lab | 500 MB | low-level-vision multiple-choice questions with answers: sanity set for metric-vs-perception direction | planned |
| `hf-videofeedback` | api | apache-2.0 | 100 MB | 32.9k videos with 1-4 scores on visual quality, temporal consistency, dynamic degree, factual, alignment | 2 f / 11.1 MB |
| `hf-fetv` | api | cc-by-4.0 | 50 MB | 619 prompts with per-dimension ratings incl. temporal quality: small T2V sanity set | planned |
| `reallq250` | manual | unknown | 200 MB | no-reference rendered-layer comparison only; the harness has no truth for it, so it feeds the blinded preferen | manual |

### Tier L

| id | access | licence | size | feeds | on disk |
|---|---|---|---:|---|---|
| `polyhaven-textures` | api | CC0 | 4,000 MB | aperiodic micro-texture and material ground truth for the texture-fidelity and periodicity metrics | 4 f / 5.4 MB |
| `polyhaven-models` | api | CC0 | 10,000 MB | known geometry (landmarks, occlusion, shading) in rendered worlds | 6 f / 1.3 MB |
| `ambientcg-materials` | api | CC0 1.0 (docs.ambientcg.com/ | 15,685 MB | second CC0 material feed for texture-fidelity ground truth; variant chosen by params.variant | 3 f / 19.2 MB |
| `ambientcg-hdris` | api | CC0 1.0 (docs.ambientcg.com/ | 2,083 MB | environment lighting (relative radiance) for rendered ground-truth worlds | 3 f / 8.7 MB |
| `mpi-sintel` | manual | CC BY 3.0 (film); dataset te | 5,000 MB | second flow-GT benchmark for the temporal gate, with film-grade motion blur and atmospherics | manual |
| `hf-kadid10k-images` | api | no formal licence; freely av | 3,100 MB | fit_gates: #24 High_Sharpen ladder fits gate.edge_overshoot = 0.22 (CI 0.19-0.26, rho -0.80, supported, extrap | 7 f / 3049.4 MB |
| `kadid10k-official` | manual | no formal licence; freely av | 2,200 MB | DMOS 1-5 per image for the sharpening/blur/noise/colour/compression ladders; joins hf-kadid10k-images; mirrore | manual |
| `hf-pipal` | api | not stated (mirror tag apach | 5,840 MB | Elo scores for GAN/restoration outputs: the closest classical set to generative texture artifacts (micro_detai | planned |
| `hf-iqa-pytorch-datasets` | api | repo cc-by-sa-4.0; original  | 2,400 MB | bulk mirror of LIVE/CSIQ/LIVE-MD/BID/KonIQ/FLIVE/SPAQ archives (one tgz each) for the Group-A fits | planned |
| `hf-uhd-iqa` | api | images CC0 | 10,700 MB | 4K/UHD images with MOS 0-1 (CC0 images): resolution-gate realism check at display ppd | planned |
| `hf-koniq10k` | api | no formal licence (DaRUS mir | 5,610 MB | 10k in-the-wild images with 5-pt MOS; distribution prior for 'natural' sharpness/noise levels | planned |
| `hf-kadis700k` | api | freely available (research) | 44,600 MB | 700k pristine+distorted pairs without MOS: unsupervised pre-fit / sanity of artifact metrics at scale | planned |
| `spaq` | page-only | not stated | 34,840 MB | smartphone captures with MOS + brightness/colourfulness/contrast/noisiness/sharpness attribute MOS: real-captu | page-only |
| `piq23` | page-only | DXOMARK non-commercial resea | 5,000 MB | DXOMARK JOD (just-objectionable-difference) scales for details/exposure/overall on smartphone portraits: a JND | page-only |
| `hf-x-aigd` | api | cc-by-4.0 (HF tag) | 2,000 MB | pixel-level polygon masks for 7 artifact classes on generated images: artifact-map supervision | planned |
| `hf-synthscars` | api | apache-2.0 (HF tag) | 3,000 MB | 12k generated images with pixel artifact masks: second artifact-map source | planned |
| `hf-rapidata-rich-hf` | api | apache-2.0 (HF tag, both) | 17,900 MB | 13k images with word-level scores + coherence/alignment heat-maps from crowd clicks: artifact region + prefere | planned |
| `hf-rapidata-700k` | api | cdla-permissive-2.0 (HF tag, | 3,730 MB | 700k pairwise votes with per-vote rater metadata (country, userScore): primary Bradley-Terry fit data for pref | 27 f / 23.4 MB |
| `hf-rapidata-model-vs-model` | api | cdla-permissive-2.0 (per-mod | 5,000 MB | model-vs-model pairwise + coherence + alignment with per-vote demographics: rater-effect term for the BT model | planned |
| `hf-open-image-preferences` | api | apache-2.0 (HF tags) | 16,800 MB | 10k pairs with raw multi-annotator votes incl. both_good/both_bad and annotator ids: tie + rater modelling for | 2 f / 11.1 MB |
| `hf-genai-bench-1600` | api | apache-2.0 (HF tag) | 4,760 MB | per-image 1-5 image-text alignment Likert for 1,600 prompts x models: composition/alignment prior | planned |
| `hf-ava-aesthetics-10pct` | api | unknown (original AVA terms  | 2,000 MB | 20k images with 10-bin aesthetic vote histograms: prior for the composition gate's preference model | planned |
| `datapointai-t2i-preferences-2m` | manual | CC-BY-4.0 for votes/prompts/ | 34,900 MB | 216k pairs x 10 votes with rater ids and ties (CC-BY-4.0 votes): best Bradley-Terry raw data; gated manual app | manual |
| `imagerewarddb` | manual | Apache-2.0 (dataset card); G | 23,700 MB | 137k images with 7-pt Likert (overall, fidelity, alignment): absolute-scale anchor for BT scores | manual |
| `hf-videogen-rewardbench` | api | apache-2.0 | 13,400 MB | 25k pairwise votes per dimension (VQ, MQ, TA, overall) on generated video: BT fit for the video track | planned |
| `hf-rapidata-t2v` | api | apache-2.0 | 2,100 MB | 2.5k T2V pairs with raw per-voter data: rater modelling for video preference | planned |
| `hf-rapidata-t2v-rich` | api | apache-2.0 | 5,000 MB | Likert per dimension on generated video: temporal artifact priors | planned |
| `hf-videoscore-bench` | api | apache-2.0 | 28,800 MB | EvalCrafter/GenAI-Bench/VBench/VideoFeedback human ratings re-packaged with videos: one-stop temporal-gate ben | planned |
| `hf-videofeedback-mp4` | api | apache-2.0 | 8,500 MB | the mp4 files for hf-videofeedback | planned |
| `hf-aigve-bench` | api | other | 6,500 MB | 2,430 generated videos with per-dimension human scores | planned |
| `hf-konvid-1k-videos` | api | freely available (research) | 2,300 MB | 1,200 in-the-wild videos (MOS from the official csv): natural flicker/judder priors for temporal.py | planned |
| `hf-mj-bench-video` | api | MIT | 10,400 MB | pairwise video preference across perspectives | planned |
| `hf-lift-hra-20k` | api | MIT | 3,700 MB | 20k videos with per-dimension human ratings + rationales | planned |
| `hf-crvd` | api | CC BY-NC-SA 4.0 | 10,000 MB | calibrated shot/read noise per ISO for video raws: temporal noise model | planned |
| `hdr-plus-burst` | api | CC BY-SA 4.0 | 39,060 MB | real capture constraints C for the claim schema: `capture_stats` (harness/data/dng.py, struct-only DNG tag rea | 308 f / 3305.7 MB |
| `raw-pixls-us` | api | CC0 | 25,100 MB | one CC0 sample raw per camera model (~900 cameras): pixel pitch / bit depth / black level cross-check for sens | 4 f / 5.7 MB |
| `eld-noise-params` | page-only | code MIT; dataset terms unst | 11,460 MB | calibrated noise parameters (K, read, row, colour) per camera: direct inputs for camera.py presets | page-only |
| `realsr-iccv2019` | manual | not stated in the repository | 12,000 MB | rendered-layer evaluation set beside OSEDiff and PiSA-SR (LANDSCAPE section 6 item 5); real optical degradatio | manual |
| `drealsr-eccv2020` | manual | not stated in the repository | 20,000 MB | rendered-layer evaluation set with a wider lens and sensor mix than RealSR | manual |
| `hf-stablesr-testsets` | api | S-Lab License 1.0 (non-comme | 3,060 MB | the fetchable copy of the RealSR and DRealSR evaluation crops used by StableSR, OSEDiff and PiSA-SR, so render | planned |
| `ntire2026-up-isp` | manual | not stated; challenge terms  | 4,000 MB | recovered-layer external evidence: a real-phone RAW to ISP track to report beside the NTIRE results (LANDSCAPE | manual |

### Tier XL

| id | access | licence | size | feeds | on disk |
|---|---|---|---:|---|---|
| `spring-flow` | api | CC BY 4.0 (DaRUS doi:10.1841 | 288,900 MB | benchmark the harness's own flow GT and the flow_warp_error metric at sub-pixel accuracy | 2 f / 0.0 MB |
| `hypersim` | manual | CC BY-SA 3.0 | 1,900,000 MB | radiometric decomposition ground truth for lighting_materials; zip per scene 1-20 GB | manual |
| `hdrplus-bursts` | api | CC BY-SA 4.0 | 3,700,000 MB | real-camera raw noise / dynamic-range distribution to sanity-check the camera baseline's noise model | planned |
| `hf-hpdv3` | api | CONFLICT: MIT on HF card tag | 141,000 MB | 1.14M pairwise preferences with per-pair aggregates: the largest open T2I preference set | planned |
| `hf-pickapic-v2-mirror` | api | unknown (no licence tag on m | 250,000 MB | 959k pairwise votes with user_id + ranking_id + ties and image bytes: largest raw-vote set | planned |
| `evalmuse-40k` | page-only | CONFLICT: BSD-3-Clause on HF | 54,600 MB | 40k images with element-level alignment labels: composition gate (prompt faithfulness) | page-only |
| `hf-visionrewarddb-video` | api | apache-2.0 | 129,000 MB | checklist + pairwise annotations on generated video at scale | planned |
| `hf-lrid` | api | CC BY-NC 4.0 | 596,000 MB | calibration frames (dark/flat) across ISO: measured read noise + gain to validate camera.py's noise model | planned |
| `hf-pixelshift200` | api | CC BY 4.0 | 113,000 MB | pixel-shift full-colour ground truth vs single-shot raw: demosaic error floor for camera.py | planned |
| `hf-sidd-dng-mirror` | api | MIT (official site) | 234,500 MB | validates camera.py's shot + read noise law with `noise_check` (harness/data/noise.py): per-pixel temporal var | 12 f / 22587.4 MB |
| `hf-mit-adobe-5k` | api | research-only non-commercial | 50,830 MB | 5k raws + 5 expert retouch targets: tone-mapping preference prior for tone gate | planned |
| `sidd-official` | manual | MIT | 234,500 MB | SIDD medium/full with NLF per capture (MIT): primary noise-model calibration | manual |
| `rawnind` | api | CC BY-SA 4.0 | 120,170 MB | ISO sweeps with clean references (CC BY-SA 4.0): noise-vs-ISO curves | planned |
| `snic` | api | MIT | 473,360 MB | ISO series + calibration frames, MIT, Harvard Dataverse | planned |

<!-- plan:end -->
