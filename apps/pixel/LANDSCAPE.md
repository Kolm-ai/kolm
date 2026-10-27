# Landscape: AI photo quality maximisation and perfected alteration

Assessed 2026-10-27 from public sources (papers, challenge reports, vendor pages, trade press). Each claim carries its source; secondary or vendor sources are marked. The last section aligns this project to the direction the evidence supports. Companion documents: [SPEC.md](SPEC.md), [DATA_PLAN.md](DATA_PLAN.md), [README.md](README.md).

## 1. The field in one map

Three layers, each with a different mechanism and a different winner.

| Layer | What it does | Mechanism | Who leads (2026) |
|---|---|---|---|
| Capture | Gets more information out of the sensor | Multi-frame burst merge in raw, HDR+, Night Sight, cloud video processing | Google Pixel, Apple, Adobe Project Indigo |
| Restoration | Recovers or invents detail after capture | Learned raw denoise at demosaic time; one-step diffusion or adversarial restorers | DxO, Adobe (Topaz pending), ByteDance, academic groups |
| Editing | Changes the picture on purpose | Instruction-following image models; style-profile auto-retouch | Google (Nano Banana Pro), OpenAI, Black Forest Labs, Imagen, Aftershoot, Evoto |

Only the capture layer adds information. The restoration layer either recovers (bounded by what was captured) or renders (invents). The editing layer alters, and the pro segment pays per photo for alteration that matches a learned style.

## 2. Research: what works and what is measured

### 2.1 Raw-domain restoration and calibrated degradation

- The field has rejected additive white Gaussian noise as training noise. A CVPR 2020 paper shows that model "deviates strongly from realistic evaluation scenarios" and proposes a CMOS noise formation model ([Wei et al.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_A_Physics-Based_Noise_Formation_Model_for_Extreme_Low-Light_Raw_Denoising_CVPR_2020_paper.pdf)). ICCV 2023 work synthesises signal-dependent noise with physics and signal-independent noise by learning ([Zhang et al.](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Towards_General_Low-Light_Raw_Noise_Synthesis_and_Modeling_ICCV_2023_paper.pdf)). A 2026 paper maps sRGB back to raw, injects Poisson-Gaussian noise and re-projects through a forward ISP, and warns that learned sRGB-to-raw inversion "overfits specific ISP implementations" ([arXiv 2603.14304](https://arxiv.org/pdf/2603.14304)).
- Benchmarks: NTIRE 2025 RAW restoration and 2x RAW super-resolution drew 230 registrants and 45 submitting teams ([Conde, Timofte et al.](https://arxiv.org/html/2506.02197v1)). AIM 2025 ran low-light raw video denoising ([arXiv 2508.16830](https://arxiv.org/pdf/2508.16830)). NTIRE 2026 added Learned Smartphone ISP with Unpaired Data on the UP-ISP set, 13,420 raw patches ([entry paper, arXiv 2605.07495](https://arxiv.org/pdf/2605.07495)), and Efficient Burst HDR and Restoration ([call](https://mailman.srv.cs.cmu.edu/pipermail/connectionists/2026-January/043967.html)); the 2025 burst HDR winner reached 43.22 dB PSNR. RAIM 2026 multi-exposure fusion drew 114 teams and 987 submissions ([arXiv 2604.09030](https://arxiv.org/pdf/2604.09030)).
- Paired real-noise SOTA on SIDD sits near 40 dB and has moved by tenths of a dB since 2022; the test truth is withheld and scored online.

### 2.2 Generative restoration: one step, with knobs

| Method | Mechanism | Reported edge | Caveat |
|---|---|---|---|
| OSEDiff | One-step distilled diffusion SR | RealLQ250 PSNR 26.67 / LPIPS 0.312 vs SUPIR 25.73 / 0.391 at 50 steps ([RSD, arXiv 2503.13358](https://arxiv.org/pdf/2503.13358)) | SUPIR is non-commercial |
| PiSA-SR, CVPR 2025 | Two LoRAs on Stable Diffusion, one l2 pixel branch, one LPIPS plus score-distillation semantic branch, two guidance scales at inference | Beats OSEDiff on LPIPS, DISTS, CLIP-IQA, MUSIQ, MANIQA in one step ([arXiv 2412.03017](https://arxiv.org/pdf/2412.03017)) | Fidelity knob is a scalar, not a map |
| GramSR, 2026 | Visual-feature conditioning | Claims gains over PiSA-SR on RealSR and DRealSR ([arXiv 2604.25457](https://arxiv.org/pdf/2604.25457)) | Single-group claim |
| HYPIR, 2025 | Pretrained diffusion weights fine-tuned adversarially, no diffusion loss, one forward pass | About 0.1 to 0.2 s at 1024², lowest LPIPS 0.2022 in a secondary summary ([arXiv 2507.20590](https://arxiv.org/abs/2507.20590v2), [Emergent Mind](https://www.emergentmind.com/topics/hypir)) | Numbers not verified against the paper |
| SeedVR2, ByteDance 2025 | One-step video restoration, adversarial post-training on real data, adaptive window attention | "Comparable or better" than multi-step video restorers ([arXiv 2506.05301](https://arxiv.org/pdf/2506.05301)) | Model card: prototypes, "overly generate details on inputs with very light degradations" ([HF](https://huggingface.co/ByteDance-Seed/SeedVR2-7B)) |
| QUSR, 2026 | Uncertainty map sets spatially varying noise injection; MLLM supplies a degradation prior | Targets hallucination under non-uniform degradation ([arXiv 2603.09125](https://arxiv.org/pdf/2603.09125)) | Uses uncertainty to guide, not to flag |

Theory: reaching perfect perceptual quality requires at least twice the inherent uncertainty of the restoration problem ([arXiv 2405.16475](https://arxiv.org/abs/2405.16475v3)). The perception-distortion wall is a theorem. Any product that claims both maximal realism and maximal fidelity from one image is wrong about one of them.

All-in-one restoration surveys exist ([arXiv 2410.15067](https://arxiv.org/abs/2410.15067v2), [list](https://github.com/Harbinzzy/All-in-One-Image-Restoration-Survey)); their 2026 entries are mostly non-diffusion and weather or video oriented.

### 2.3 Reinforcement learning with quality rewards

- RealSR-R1 brought GRPO to real-world SR with a four-part reward including a visual expert model ([arXiv 2506.16796](https://arxiv.org/pdf/2506.16796)). FinPercep-RM at CVPR 2026 exists because global IQA scores "are insensitive to local, fine-grained distortions" and get hacked ([CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_FinPercep-RM_A_Fine-grained_Reward_Model_and_Co-evolutionary_Curriculum_for_RL-based_CVPR_2026_paper.html)). A GRPO survey catalogues the restoration branch ([arXiv 2603.06623](https://arxiv.org/pdf/2603.06623)).
- Judges: VisualQuality-R1 at NeurIPS 2025 trains a reasoning IQA model by RL-to-rank ([arXiv 2505.14460](https://arxiv.org/pdf/2505.14460)); PreResQ-R1 claims SOTA on 10 IQA and 5 VQA sets from 6K images ([arXiv 2511.05393](https://arxiv.org/pdf/2511.05393)); DeQA-Score and Q-Align are the MLLM scorer baselines; RALI at ICLR 2026 studies why RL IQA generalises. All numbers are self-reported on different training sets.

Lesson for this project: an IQA model is a judge, never a loss. Reward hacking of global scores is documented at CVPR 2026.

### 2.4 Hallucination measurement

- HalluGen synthesises controllable hallucinations with labels and shows PSNR, SSIM and LPIPS "often score hallucinated outputs highly" ([arXiv 2512.03345](https://arxiv.org/html/2512.03345v1)). sFRC detects hallucinations in undersampled medical imaging ([arXiv 2603.04673](https://arxiv.org/pdf/2603.04673)). No consumer-photo hallucination benchmark with ground truth was found. This project's synthetic-world harness can produce one, because it knows the truth per pixel.

### 2.5 Camera-parameter and physical-parameter control

- Generative Photography (Purdue, Chan) conditions generation on bokeh, focal length, shutter speed and colour temperature ([Purdue](https://engineering.purdue.edu/ECE/News/2025/generative-photography-lays-new-foundations-for-imaging)); Bokeh Diffusion conditions on a physical defocus parameter and supports real-image editing by inversion ([arXiv 2503.08434](https://arxiv.org/html/2503.08434v1)); video cinematic control ties blur to aperture, a focus plane, normalised focal length and exposure to shutter ([arXiv 2511.12921](https://arxiv.org/pdf/2511.12921)). A GPT-4o study found bokeh control good and focal-length control immature ([arXiv 2504.05979](https://arxiv.org/pdf/2504.05979)). Camera-parameter-conditioned denoising exists ([arXiv 2507.01587](https://arxiv.org/abs/2507.01587)).
- Relighting: Careaga and Aksoy, SIGGRAPH 2025 ([arXiv 2508.05626](https://arxiv.org/abs/2508.05626)); LightMover ([arXiv 2603.27209](https://arxiv.org/abs/2603.27209)); latent proxies ([arXiv 2603.15555](https://arxiv.org/abs/2603.15555)).
- Deep optics: curriculum-learned compound lenses ([arXiv 2302.01089](https://arxiv.org/html/2302.01089v3)) and learned restoration for mass-produced 10 mm metalenses ([Advanced Photonics 2024](https://spiedigitallibrary.org/journals/advanced-photonics/volume-6/issue-06/066002/Deep-learning-driven-end-to-end-metalens-imaging/10.1117/1.AP.6.6.066002.pdf)). Joint optics plus reconstruction is the frontier for the capture layer, not for a software product.

## 3. Products and what they actually do

### 3.1 Capture layer

| Product | Mechanism | Disclosure | Source |
|---|---|---|---|
| Google Pixel 10 Pro Zoom (was Pro Res Zoom) | Past about 30x, on-device generative model rebuilds regions; 4 to 5 s per shot | Metadata "Edited with AI tools"; C2PA written by Camera and Photos, removable with a CLI tool | [PhoneArena](https://www.phonearena.com/news/pixel-10-pro-res-zoom-when-is-a-photo-not-a-photo-anymore_id173458), [Android Authority](https://www.androidauthority.com/p-3590986/), [TechRepublic](https://www.techrepublic.com/article/google-pixel-10-pro-pro-zoom-name-change/) |
| Google Video Boost, Night Sight Video | On-device Tensor denoise, then cloud "HDR Plus" model; processed file replaces the original | None beyond Google's own description | [Google](https://store.google.com/magazine/video-boost?hl=en-US), [PhoneArena](https://www.phonearena.com/news/pixel-10s-video-feature-frustrating_id173655) |
| Adobe Project Indigo (Levoy, Kainz) | Bursts up to 32 frames merged for lower noise and higher DR, JPEG or raw, free on iPhone 12 Pro and up | Has an AI reflection-removal feature, so not AI-free | [9to5Google](https://9to5google.com/2025/06/18/adobe-project-indigo-pixel-camera-app/), [Thurrott](https://www.thurrott.com/?p=322344) |
| Apple Clean Up (iOS 26 to 27) | On-device infill; iOS 27 reportedly uses Gemini and cloud models do better | Weak at faces in iOS 26 | [Tom's Guide](https://www.tomsguide.com/phones/i-tried-ios-27-clean-up-vs-galaxy-ai-photo-assist-in-a-7-round-test-and-i-was-surprised-with-how-close-it-was), [Wccftech](https://wccftech.com/ios-27-vs-ios-26-clean-up-tool-comparison-major-improvements/amp/) |
| Samsung Galaxy AI Generative Edit | Remove, resize, relocate, add objects | S25 signs only AI-touched images | [SamMobile](https://www.sammobile.com/?p=16606689) |

### 3.2 Restoration layer

| Product | Mechanism | Standing | Source |
|---|---|---|---|
| DxO PureRAW 5 and 6, DeepPRIME XD3 | Learned denoise during demosaic, raw in, linear DNG out | Reviewers rate it ahead of Lightroom Denoise on detail and noise; slow, large files, high price | [Digital Camera World](https://digitalcameraworld.com/photography/photo-editing/dxo-pureraw-5-review), [DxO 5.3](https://www.dxo.com/en/news/introducing-dxo-pureraw-5-3), [PetaPixel](https://petapixel.com/2025/02/27/dxo-pureraw-5-features-local-adjustments-and-faster-denoising) |
| Adobe Lightroom Denoise, Reflection Removal, Generative Remove, AI Sharpen "powered by Topaz" | Denoise at demosaic; AI edits are layered and can be invalidated by later Denoise | Order-of-operations bug confirmed Feb 2026 | [Adobe release notes](https://helpx.adobe.com/uk/lightroom/desktop/introduction/release-notes.html), [forum](https://community.adobe.com/questions-675/reflection-removal-may-be-restored-by-denoise-1548689) |
| Topaz Photo AI: Standard vs Generative (Recover, Redefine) | Standard keeps fidelity; Generative is "highest quality and most creative" at the cost of speed and fidelity | Recover 326 s on an RTX 2070 Super; sky tiling and blotches reported; preview differs from result | [Topaz docs](https://developer.topazlabs.com/docs/image-models), [community](https://community.topazlabs.com/t/discussion-workflow-generative-models-results/84833), [AIarty](https://aiarty.com/ai-image-enhancer/gigapixel-ai-review.htm) |
| Magnific, Krea | Creative or "hallucination school" upscalers | Priced $20 to $199 per month depending on source | [Rangy](https://rangy.ai/blog/best-ai-photo-restoration-tools), [fast.io](https://fast.io/resources/best-ai-photo-enhancers-2026/) |
| Remini | Face and old-photo restoration, consumer mobile | Reoriented to social use | [MagicHour](https://magichour.ai/blog/best-ai-image-enhancers-2026) |
| Amped FIVE | Deterministic, documented, "does not use AI" (2021 statement); report lists every step | The forensic reference point | [Amped](https://blog.ampedsoftware.com/2021/11/20/amped-five-does-not-use-ai-and-implements-forensically-safe-algorithms-to-enhance-video-evidence), [product page](https://ampedsoftware.com/five) |

### 3.3 Editing layer

| Product | Mechanism | Measured standing | Source |
|---|---|---|---|
| Nano Banana Pro (Google) | Instruction image editing | WiseEdit top overall 82.4; GPT-image-1 77.6; Seedream 4.0 75.2; Qwen-Image-Edit 60.2 | [WiseEdit, arXiv 2512.00387](https://arxiv.org/pdf/2512.00387) |
| FLUX.2 (BFL) | Edits up to 4 MP | About 4.0 on Edit-Compass against Nano Banana's mid-4s | [Edit-Compass, arXiv 2605.13062](https://arxiv.org/pdf/2605.13062), [AIMultiple](https://research.aimultiple.com/ai-image-editing) |
| Imagen AI | Personal AI Profile learned from about 3,000 of the photographer's own edits in Lightroom; $0.22 to $0.32 per photo | Vendor-sourced | [Imagen](https://imagen-ai.com/post/best-ai-retouching-software-2026/) |
| Aftershoot | Culling plus style profile; offline desktop app, subscription | Vendor-sourced | [Imagen comparison](https://imagen-ai.com/post/best-evoto-alternatives-competitors/) |
| Evoto | Layered portrait retouch, per-export credits about $0.07 | Vendor-sourced | same |
| Retouch4me | Single-purpose Photoshop and Lightroom plugins, one-time licence | Vendor-sourced | same |

### 3.4 Market structure

- Adobe agreed on 2026-06-25 to acquire Topaz Labs, closing second half 2026 pending approval, terms undisclosed; Topaz models go into Firefly, Firefly Services, Photoshop, Lightroom and Premiere; standalone apps continue; NeuroStream runs large models locally ([Newsshooter](https://www.newsshooter.com/2026/06/25/adobe-acquires-topaz-labs-bringing-ai-upscaling-and-restoration-in-house/), [CineD](https://www.cined.com/adobe-to-acquire-topaz-labs-ai-upscaling-and-image-enhancement-on-device-models-standalone-apps-to-continue/), [TNW](https://thenextweb.com/news/adobe-acquires-topaz-labs-ai-enhancement)).
- Freepik acquired Magnific in 2024 ([CB Insights](https://www.cbinsights.com/company/magnific/financials)). One 2026 commentary argues the standalone upscaler category "collapsed" into editors and assistants ([Rioja](https://alejandrorioja.com/top-ai-image-enhancer-tools/)); it is opinion but matches the Topaz and Magnific exits.
- Read: the generic upscaler shelf is owned. Faithful restoration inside the editor will be Adobe's after Topaz. On-device generative zoom is Google's. Instruction editing is the frontier labs'. Pro auto-retouch is a per-photo business with style profiles and no physical model.

## 4. Provenance, forensics and disclosure

- C2PA cameras: Leica M11-P has signed every JPEG and DNG by default since October 2023; Sony has the broadest lineup but verification needs Sony's own infrastructure; Nikon's certificate was reported revoked in September 2025 with conflicting reports on current status; Pixel 10 signs every photo with hardware-backed keys; Samsung S25 signs only AI-touched images; Canon lags; as of 2026 no dedicated camera is certified under the C2PA Conformance Program ([Wikipedia](https://en.wikipedia.org/wiki/Content_Credentials), [c2paviewer devices](https://c2paviewer.com/supported-devices), [SoftwareSeni](https://softwareseni.com/c2pa-adoption-in-2026-hardware-platforms-and-verification-reality), [Digital Camera World](https://www.digitalcameraworld.com/cameras/these-are-the-best-c2pa-cameras-that-add-content-credentials-to-your-images-for-every-photographer-tired-of-the-question-is-this-ai)). Metadata is stripped on most uploads, so provenance must be re-asserted at the point of display, not only at capture.
- Forensics: SWGDE image processing guidelines require the original preserved, a working copy, and steps documented so a comparably trained person can reproduce them ([SWGDE 15-M-002](https://www.swgde.org/documents/published-complete-listing/15-m-002-swgde-image-processing-guidelines/)); a December 2025 commentary quotes SWGDE warning that machine learning makes it hard to identify and replicate applied processes ([Criminal Legal News](https://www.criminallegalnews.org/news/2025/dec/15/when-ai-invents-the-pixels-challenging-ai-enhanced-video-evidence-in-criminal-cases/)); State v. Puluka rejected generative-AI-enhanced evidence made with a Topaz tool, per a vendor-adjacent source ([Hackernoon](https://sia.hackernoon.com/when-ai-enhancement-becomes-evidence-why-deterministic-methods-are-quietly-changing-the-courtroom)). SWGDE requires any citation to carry the document version or date.

## 5. Where the field is going

1. One step, not fifty. Every 2025 to 2026 restorer that matters runs in one forward pass (OSEDiff, PiSA-SR, HYPIR, SeedVR2). Multi-step diffusion restoration is a research baseline now.
2. Knobs over prompts. Fidelity-versus-realism is exposed as an explicit control (PiSA-SR scales, QUSR uncertainty), and camera parameters are becoming conditioning inputs.
3. Physics-based degradation synthesis is the accepted way to make training data for real noise; AWGN and generic Real-ESRGAN-style pipelines are the thing papers now contrast against.
4. IQA models are becoming RL judges and RL rewards, and the reward-hacking problem is already published.
5. Disclosure is being forced: Pixel labels generative zoom, C2PA ships in phones, courts reject undocumented generative enhancement.
6. Consolidation: restoration moves inside the editor (Adobe plus Topaz) and inside the camera (Google, Apple, Samsung). Standalone generic enhancement is not a position.

## 6. Alignment: the direction this project takes

The evidence supports the two-layer plan already in [SPEC.md](SPEC.md) and the measured-degradation data plan in [DATA_PLAN.md](DATA_PLAN.md). It also says where to sharpen them.

### Keep

- The harness as the verifier with ground truth per pixel. Nothing in the market or the literature offers a consumer-photo hallucination benchmark with known truth; this project already generates one.
- The recovered layer in the raw or linear domain, trained on degradations sampled from measured camera distributions (noise law, lens traces, capture stats, HDRI scene range). This is exactly the physics-based synthesis the 2026 papers converge on.
- Metric vectors as conditioning and evaluation only, never as loss. CVPR 2026 reward hacking confirms the Goodhart rule in [DATA_PLAN.md](DATA_PLAN.md).

### Change or add

1. Rendered layer: do not train a base generative model. Adopt a one-step diffusion restorer with an explicit fidelity control (PiSA-SR-style dual branches or a HYPIR-style adversarial fine-tune of a permissively licensed base) and expose the knob plus an invented-detail map per output. SUPIR-class multi-step models are out on licence and speed.
2. Hallucination map as a product output. Label invented detail exactly against the synthetic truth (HalluGen-style controllable hallucinations, generated from our scene generator), train the map head on those labels, and report it per gate in the harness. This is the differentiator no reviewed product has.
3. Provenance per pixel. Emit a manifest that states which pixels came from the recovered layer and which from the rendered layer, in C2PA form where the host supports it and in the report always. The recovered layer then meets the SWGDE documentation test; the rendered layer is disclosed by construction.
4. Perfected alteration as parameter edits, not prompts. The "alter in perfected ways" product is a style profile learned from the user's own edits (the Imagen and Aftershoot mechanism) applied as changes to the virtual camera and lighting model: exposure, white balance, depth of field, relighting. Each edit is checked by the gate order, so the output is a photograph that a better camera in a better light could have taken, not a render.
5. External evidence of SOTA. Enter the NTIRE 2027 RAW restoration and burst HDR tracks with the recovered layer, and report the rendered layer on RealSR, DRealSR and RealLQ250 beside OSEDiff and PiSA-SR. SIDD stays the recovered layer's paired-noise check.
6. Judges: use VisualQuality-R1-class and DeQA-Score-class models only as blinded judges in the preference track, with the fine-grained-distortion caveat from FinPercep-RM.
7. Run local. Topaz NeuroStream and Aftershoot's offline desktop show that a local 20 to 60M parameter recovered layer is a selling point, and the one-step rendered layer fits one consumer GPU.

### Do not do

- A generic upscaler, a text-prompted semantic editor, or any forensic claim for the rendered layer.
- Cloud-only processing of originals, which Video Boost shows users distrust when the processed file replaces the source.

### Repository actions

| Action | Where | Status (2026-10-27) |
|---|---|---|
| Degradation sampler and offline metric labels | [harness/data/degrade.py](harness/data/degrade.py), [harness/data/tilemetrics.py](harness/data/tilemetrics.py), [harness/data/label.py](harness/data/label.py), [DEGRADATION_PLAN.md](DEGRADATION_PLAN.md) | Done: 31-field parameter vectors drawn from the measured distributions, 12 metric labels per draw, no images stored; 2048-row sample shard under `data/labels/`; tests in [tests/test_degrade.py](tests/test_degrade.py) |
| Invented-detail labels from the scene generator | [harness/hallucination.py](harness/hallucination.py), [tests/test_hallucination.py](tests/test_hallucination.py) | Done: per-pixel invented-detail map against the truth (global blur match, half-octave bands, local per-band gain, the candidate's own noise shape, moving landmarks excluded), `inject` plants labelled hallucinations; 14 tests; the best-case camera measures 0 to 0.5 %, 32 px patches at 20 % contrast are found at 67 to 100 % recall; the metric `invented_detail_fraction` is in the registry |
| Per-pixel provenance in the claim schema and report | [SPEC.md](SPEC.md) sections 1, 5, 8, 9; `rendered_map` claim block; `provenance` gate; `run --rendered-map` | Done: `rendered_fraction` and `undeclared_invention_fraction`, gate 2 `undeclared_invention_fraction <= 0.01`, report block per scene region with `invented_detail.npy` beside it |
| Evaluation sets RealSR, DRealSR, RealLQ250 and the NTIRE 2026 UP-ISP set in the manifest | [harness/data/sources.json](harness/data/sources.json) | Done: `realsr-iccv2019`, `drealsr-eccv2020` (manual, no licence stated), `hf-stablesr-testsets` (fetchable, S-Lab licence, the papers' evaluation crops), `ntire2026-up-isp` (manual, Codabench), `reallq250` (manual, origin unconfirmed) |
| Recovered layer trained on the sampler's pairs with the metric vector as conditioning | [harness/model/](harness/model/) (`data.py`, `net.py`, `train.py`, `apply.py`), `python -m harness model train\|apply\|eval-demo`, [tests/test_model.py](tests/test_model.py) | Done: seven runs, six on 2026-10-27 and the seventh on 2026-10-28, each fixing one variable the previous run showed a tile cannot identify (DEGRADATION_PLAN.md section 7); demo-world verdicts against the best-case camera: v1 FAIL at provenance (undeclared invention 0.125 still, 0.170 video, 2.2x too bright), v3 FAIL at provenance (0.022 and 0.011, 1.6x), v4 FAIL at lighting_materials (shading_rmse 0.023, 1.09x, delta_e 1.31, provenance 0), v5 FAIL at tone_color on the still (resolution and texture beat the camera for the first time, mtf50 0.443 against 0.220, luminance ratio 0.985, but 19 % of the highlight patch clipped, shadow SNR 12 against 26 dB, edge overshoot 0.092, provenance 0.3 %) and FAIL at provenance on the video (4.1 % undeclared invention), v6 FAIL at tone_color on the still (mtf50 0.447 against 0.220, texture 0.522 against 0.223, delta E 1.04, highlight loss 0.250, shadow SNR 13.5 against 26.2 dB, edge overshoot 0.103, provenance 0.05 %) and FAIL at provenance on the video (7.5 %); blind, FAIL at tone_color on the still (mtf50 0.392, highlight loss 0.031, overshoot 0.058, provenance 0.04 %) and FAIL at provenance on the video (3.0 %), v7 FAIL at tone_color on the still (mtf50 0.500 against 0.220, texture 0.472 against 0.223, shadow SNR 22.8 against 26.2 dB, highlight loss 0.429, delta E 0.86, edge overshoot 0.178, provenance 0.11 %) and FAIL at provenance on the video (3.6 %); blind, FAIL at tone_color on the still (mtf50 0.446, shadow SNR 25.6 ties the camera, highlight loss 0.325, delta E 0.66, overshoot 0.130, provenance 0.14 %) and FAIL at provenance on the video (3.7 %); 14 tests in [tests/test_model.py](tests/test_model.py) |
| Rendered-layer baseline run: OSEDiff and PiSA-SR weights through the harness on the demo scene | new `claims/rendered_baseline.json` | Not run: the third-party weights are not collected; the machine has an RTX 5090 (32 GB), so the earlier "needs a GPU" reason was wrong |

## 7. Source quality notes

Challenge reports and arXiv papers are primary. Topaz, DxO, Imagen and Google pages are vendor claims. Pricing figures conflict across review blogs and are not relied on. HYPIR speed and LPIPS numbers come from a secondary summary. The Puluka ruling is reported by a vendor-adjacent outlet. Nikon's C2PA status has conflicting reports. SWGDE's AI warning is quoted through a December 2025 commentary, not the original document.
