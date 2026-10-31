# kolm-pixel-v15

The recovered layer of kolm pixel: a conditional U-Net that takes a camera raw's linear decode (LibRaw's demosaic, camera white balance, linear sRGB primaries, 1.0 at the sensor's clip) and returns the scene it recorded: noise removed, detail and tone kept, conditioned on the capture (its noise law, bit depth, mosaic) and on target image metrics.

## Release

| | |
|---|---|
| step | 45500 |
| parameters | 120,535,171 (120.5M) |
| training time | 244 min |
| weights | `model.safetensors` (EMA, safetensors), SHA-256 `822593100eb0a8b5bcb344e711efba5a07fc368fdf4be05146ebafd290053e78` |
| format | `kolm-pixel-recovered/1` |
| network | CondUNet: base 64, mults [1, 2, 4, 8], blocks [2, 3, 6, 8], 12 mid blocks, conditioning 31 dims, noise map True |
| exported from | `D:/lens-harness-data/models/recovered_v15/best.pt` |

## Use

```bash
python -m harness photo IMG_0001.DNG --ckpt D:/lens-harness-data/release/pixel-v15 --out restored/ --original
```

Each input writes a 16-bit sRGB TIFF, a JPEG preview and a JSON record of the decode, the capture descriptor and the noise law behind it. A raw's noise law is read blind from the frame (`data/blindnoise.py`); any other image is restored with no descriptor. `--scale 2` restores at the scale training's real pairs were decoded at; `--capture JSON` states the descriptor instead of reading it. `apply.load_model` reads this directory, or a training checkpoint, the same way.

## Training data

60000 steps planned (the release is step 45500), batch 16, tiles of 256 px, learning rate 0.0002; real capture/reference pairs drawn with probability 0.5, degraded clean images otherwise.

| source | licence (manifest) | sampling weight | real pairs |
|---|---|---:|---|
| `hf-kadid10k-images` | no formal licence; freely available to the research community | 1 | - |
| `hf-tid2013` | none stated; freely available for scientific purposes | 1 | - |
| `polyhaven-hdris` | CC0 | 2.0 | - |
| `wikimedia-featured-pictures` | per file (LicenseShortName): CC0, public domain, CC BY, CC BY-SA kept and recorded as the first path component; NC, ND and GFDL-only files skipped | 1 | - |
| `hdr-plus-burst` | CC BY-SA 4.0 | 6.0 | yes |
| `rawnind` | CC BY-SA 4.0 | 6.0 | yes |
| `si-hdr` | CC BY 4.0 (dc.rights on the Apollo record, doi:10.17863/CAM.87333; read 2026-10-28) | 4.0 | yes |
| `hf-pixelshift200` | CC BY 4.0 | 16.0 | - |

Held out of training: every real pair's val and test scenes, RawNIND's test reserve, and raw-pixls-us (the raw evaluation set) entirely.

## Held-out results

Each set's tiles are scored by `model.heldout`: per metric, the restored tile against the truth beside the input's, and the gates in order (a set fails at its first failing gate; the gates after it are not reached). The gate sets compare against **capture**: the real capture (input); **camera**: the camera's own photo; **frontier**: the oracle linear frontier (the best linear filter fitted on the target). "known" conditions on the measured noise law (on the raw-pixls-us sets, the demo camera's), "blind" on no descriptor; "-" is a gate set the set has no reference for.

| set | what | tiles | capture | camera | frontier | first failing check per gate set |
|---|---|---:|---|---|---|---|
| `heldout_blind` | Held-out raw evaluation: raw-pixls-us at clean scale 4, 64 tiles of 256 px, seed 0; descriptor withheld | 64 | fail at denoise | - | fail at artifacts | capture: denoise: shadow_snr_db loses capture, margin -3.097 (tie within 1); frontier: artifacts: banding_score loses frontier, margin -0.004238 (tie within 0.002) |
| `heldout_known` | Held-out raw evaluation: raw-pixls-us at clean scale 4, 64 tiles of 256 px, seed 0; descriptor known | 64 | fail at artifacts | - | fail at artifacts | capture: artifacts: edge_overshoot loses capture, margin -0.02523 (tie within 0.01); frontier: artifacts: edge_overshoot loses frontier, margin -0.02523 (tie within 0.01) |
| `heldout_real_hdr-plus-burst_blind` | Real-pair evaluation: hdr-plus-burst at clean scale 2, 64 tiles of 256 px, seed 0; descriptor withheld | 64 | fail at artifacts | fail at artifacts | fail at artifacts | capture: artifacts: edge_overshoot loses capture, margin -0.02522 (tie within 0.01); camera: artifacts: edge_overshoot ties camera, margin +0.007922 (tie within 0.01); frontier: artifacts: edge_overshoot loses frontier, margin -0.03148 (tie within 0.01) |
| `heldout_real_hdr-plus-burst_known` | Real-pair evaluation: hdr-plus-burst at clean scale 2, 64 tiles of 256 px, seed 0; descriptor known | 64 | fail at micro_detail | **PASS** | fail at artifacts | capture: micro_detail: texture_hf_fidelity loses capture, margin -0.03954 (tie within 0.03, by distance to the truth 1); frontier: artifacts: edge_overshoot loses frontier, margin -0.0144 (tie within 0.01) |
| `heldout_real_rawnind_blind` | Real-pair evaluation: rawnind at clean scale 2, 64 tiles of 256 px, seed 0; descriptor withheld | 64 | fail at denoise | - | **PASS** | capture: denoise: banding_score ties capture, margin +0.001628 (tie within 0.002) |
| `heldout_real_rawnind_known` | Real-pair evaluation: rawnind at clean scale 2, 64 tiles of 256 px, seed 0; descriptor known | 64 | **PASS** | - | **PASS** | - |
| `heldout_real_rawnind_sp_blind` | Real-pair evaluation: rawnind at clean scale 2, 64 tiles of 256 px, seed 0; descriptor withheld; targets registered to a fraction of a pixel | 64 | fail at denoise | - | **PASS** | capture: denoise: banding_score ties capture, margin +0.0004804 (tie within 0.002) |
| `heldout_real_rawnind_sp_known` | Real-pair evaluation: rawnind at clean scale 2, 64 tiles of 256 px, seed 0; descriptor known; targets registered to a fraction of a pixel | 64 | fail at micro_detail | - | **PASS** | capture: micro_detail: texture_hf_fidelity loses capture, margin -0.05654 (tie within 0.03, by distance to the truth 1) |
| `heldout_real_si-hdr_blind` | Real-pair evaluation: si-hdr at clean scale 2, 64 tiles of 256 px, seed 0; descriptor withheld | 64 | fail at resolution | - | fail at micro_detail | capture: resolution: mtf50_cpp loses capture, margin -0.03328 (tie within 0.02, by distance to the truth 0.5); frontier: micro_detail: texture_hf_fidelity loses frontier, margin -0.08939 (tie within 0.03, by distance to the truth 1) |
| `heldout_real_si-hdr_known` | Real-pair evaluation: si-hdr at clean scale 2, 64 tiles of 256 px, seed 0; descriptor known | 64 | fail at resolution | - | fail at micro_detail | capture: resolution: mtf50_cpp loses capture, margin -0.09215 (tie within 0.02, by distance to the truth 0.5); frontier: micro_detail: texture_hf_fidelity loses frontier, margin -0.122 (tie within 0.03, by distance to the truth 1) |

Every gate set passes on 1 of 10 set(s): `heldout_real_rawnind_known`.

## The capture descriptor as the photo command reads it

Held-out real pairs restored with the law measured against the reference (what training saw), the law read blind from the input (what the photo command uses), and no descriptor, at each decode scale reported. These are capture-eval's own draws (up to 12 pairs a source, 4 tiles a pair) against the capture gates only, smaller than the held-out sets above, whose verdicts they do not replace.

### rawnind at scale 2: 12 pair(s), 48 tile(s)

estimated law fitted on 12/12 pair(s); log10(estimated / measured electrons) median +0.199, p10 +0.054, p90 +0.287

| metric | input | measured | estimated | unknown | truth |
|---|---|---|---|---|---|
| mtf50_cpp | 0.429 | 0.4425 | 0.4416 | 0.4433 | 0.5 |
| texture_hf_fidelity | 0.7448 | 0.9838 | 0.992 | 0.8925 | 1 |
| shadow_snr_db | 15.06 | 23.28 | 22.45 | 21.98 | 100 |
| highlight_loss_fraction | - | - | - | - | - |
| dr_steps | 3 | 3 | 3 | 3 | 3 |
| delta_e_mean | 0.8993 | 0.6846 | 0.6871 | 0.7084 | 0 |
| delta_e_max | 3.749 | 2.8 | 2.894 | 3.081 | 0 |
| edge_overshoot | 0 | 0 | 0 | 0 | 0 |
| banding_score | 0.03509 | 0.02176 | 0.02348 | 0.02557 | 0 |
| periodicity | 0.02981 | 0.05575 | 0.05486 | 0.05691 | 0 |
| landmark_error_px | 0.3709 | 0.3661 | 0.3756 | 0.3929 | 0 |
| shading_rmse | 0.01042 | 0.01224 | 0.01174 | 0.01298 | 0 |

| gate set | measured | estimated | unknown |
|---|---|---|---|
| capture | PASS | PASS | PASS |
| capture_hp | PASS | PASS | PASS |

### hdr-plus-burst at scale 2: 12 pair(s), 48 tile(s)

estimated law fitted on 12/12 pair(s); log10(estimated / measured electrons) median -0.112, p10 -0.549, p90 +0.027

| metric | input | measured | estimated | unknown | truth |
|---|---|---|---|---|---|
| mtf50_cpp | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| texture_hf_fidelity | 1.386 | 1.119 | 1.12 | 1.282 | 1 |
| shadow_snr_db | 14.14 | 22.06 | 22.84 | 22.26 | 100 |
| highlight_loss_fraction | - | - | - | - | - |
| dr_steps | 4 | 4 | 4 | 4 | 4 |
| delta_e_mean | 1.029 | 0.7911 | 0.789 | 0.879 | 0 |
| delta_e_max | 3.53 | 2.52 | 2.481 | 3.06 | 0 |
| edge_overshoot | 0.02201 | 0.01847 | 0.01542 | 0.0315 | 0 |
| banding_score | 0.0451 | 0.0303 | 0.02986 | 0.03312 | 0 |
| periodicity | 0.02551 | 0.02297 | 0.02398 | 0.03497 | 0 |
| landmark_error_px | 0.07647 | 0.0889 | 0.08758 | 0.1009 | 0 |
| shading_rmse | 0.005876 | 0.0072 | 0.007949 | 0.00964 | 0 |

| gate set | measured | estimated | unknown |
|---|---|---|---|
| capture | PASS | PASS | PASS |
| capture_hp | PASS | PASS | PASS |

### rawnind at scale 1: 12 pair(s), 48 tile(s)

estimated law fitted on 12/12 pair(s); log10(estimated / measured electrons) median +0.224, p10 +0.189, p90 +0.370

| metric | input | measured | estimated | unknown | truth |
|---|---|---|---|---|---|
| mtf50_cpp | 0.2151 | 0.2317 | 0.231 | 0.2371 | 0.5 |
| texture_hf_fidelity | 0.2449 | 0.2442 | 0.2541 | 0.283 | 1 |
| shadow_snr_db | 14.32 | 21.47 | 21.23 | 21.7 | 100 |
| highlight_loss_fraction | - | - | - | - | - |
| dr_steps | 2 | 2.5 | 2.5 | 2.5 | 2.5 |
| delta_e_mean | 1.593 | 1.474 | 1.516 | 1.489 | 0 |
| delta_e_max | 6.315 | 5.506 | 5.68 | 5.855 | 0 |
| edge_overshoot | 0 | 0 | 0 | 0 | 0 |
| banding_score | 0.04789 | 0.03163 | 0.03809 | 0.03473 | 0 |
| periodicity | 0.03734 | 0.06554 | 0.06609 | 0.08144 | 0 |
| landmark_error_px | 0.6146 | 0.6165 | 0.6282 | 0.595 | 0 |
| shading_rmse | 0.01465 | 0.01523 | 0.01524 | 0.01576 | 0 |

| gate set | measured | estimated | unknown |
|---|---|---|---|
| capture | PASS | PASS | PASS |
| capture_hp | PASS | PASS | PASS |

### hdr-plus-burst at scale 1: 12 pair(s), 48 tile(s)

estimated law fitted on 12/12 pair(s); log10(estimated / measured electrons) median -0.067, p10 -0.360, p90 -0.002

| metric | input | measured | estimated | unknown | truth |
|---|---|---|---|---|---|
| mtf50_cpp | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| texture_hf_fidelity | 1.513 | 1.151 | 1.111 | 1.194 | 1 |
| shadow_snr_db | 9.157 | 20.29 | 21.09 | 19.85 | 100 |
| highlight_loss_fraction | - | - | - | - | - |
| dr_steps | 3 | 4 | 4 | 3 | 4 |
| delta_e_mean | 1.659 | 1.18 | 1.198 | 1.312 | 0 |
| delta_e_max | 5.71 | 4.407 | 4.422 | 4.7 | 0 |
| edge_overshoot | 0.002597 | 0 | 0 | 0 | 0 |
| banding_score | 0.07691 | 0.04287 | 0.04423 | 0.04919 | 0 |
| periodicity | 0.03628 | 0.04221 | 0.04475 | 0.05577 | 0 |
| landmark_error_px | 0.1272 | 0.1924 | 0.1907 | 0.2086 | 0 |
| shading_rmse | 0.01157 | 0.01254 | 0.01322 | 0.01428 | 0 |

| gate set | measured | estimated | unknown |
|---|---|---|---|
| capture | PASS | PASS | PASS |
| capture_hp | PASS | PASS | PASS |

## Blind noise reading

Method `blocks` (Gaussian residual at sigma 1.0 px, 8 px cells, quantile 0.1), variance scale 1.231 at decode scale 2, times 1.702 at decode scale 1.

The factor at decode scale 1 is the median of measured over blind noise variance on 40 train pair(s) decoded at that scale (centre 2048 px crops): 1.702, log10 p10 -0.159, p90 +0.461; per source `rawnind` 2.524 (n 20), `hdr-plus-burst` 1.419 (n 20).

Calibrated on 289 stored real pairs (centre 1024 px crops): on the val pairs, log10(blind / measured electrons) has median +0.018, p10 -0.569, p90 +0.271 (n 122).

- `rawnind`: median +0.191, p10 -0.321, p90 +0.312 (n 57)
- `hdr-plus-burst`: median -0.132, p10 -0.847, p90 +0.047 (n 57)
- `si-hdr`: median +0.121, p10 -0.875, p90 +0.233 (n 8)

## Limitations

- It does not yet beat every reference on every set: 9 of 10 held-out set(s) fail a gate (above). The camera and frontier gates are the bar; a set that fails them is one where the model is not the best available result.
- Training's real pairs are decoded at scale 2 (box-downsampled), and every held-out verdict above is at that scale or coarser. Full resolution (`--scale 1`, the photo command's default) is measured only by capture-eval's capture gates where a scale-1 report is included; no camera or frontier verdict covers it.
- The descriptor the photo command reads is blind: where it is far from the measured law, the restore is the one the descriptor implies, not the one the capture needs. `--capture` states it when it is known.
- Inputs are LibRaw's demosaic of one frame: no burst merge, no lens correction beyond what the training degradations cover, and no claim about any image the held-out sets do not resemble.
- An 8-bit sRGB input has been through a camera's own processing; it is restored with no descriptor, and no held-out set above measures that case.

## Licence

Training data includes source(s) whose terms limit use to research or non-commercial purposes: `hf-kadid10k-images` (no formal licence; freely available to the research community), `hf-tid2013` (none stated; freely available for scientific purposes). These weights are for research use; a release for any other use needs a run trained without them.

Source(s) under CC BY-SA (`wikimedia-featured-pictures`, `hdr-plus-burst`, `rawnind`) are credited here; whether weights trained on them are adaptations under that licence is not settled, and this card does not decide it.
