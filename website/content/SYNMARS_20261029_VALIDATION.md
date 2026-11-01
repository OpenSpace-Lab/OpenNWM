# SynMars video-pose adaptation — 2026-10-29

This experiment adds paired Mars simulation reference/prediction clips to the
existing Earth and Moon gallery. It uses actual held-out SynMars-TW image
sequences and OpenNWM predictions. It does not use rendered illustrations or
repeat a single image as a purported reference video.

## Protocol

- Frozen pose annotator: VGGT-Omega-1B-512, official source revision
  `282ec70363edeff59424bf43731658092fba3d37`; weight SHA256
  `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.
- Each window contains 20 left-camera images, separated by four source-frame
  indices. The first four are observed context; the remaining 16 are targets.
- Train: six windows each from forward routes 1–5 (30 windows, 480 targets).
  Validation: three windows from route 8 (48 targets). Test: six windows from
  route 9 (96 targets). Both travel directions of each route belong to the same
  split; the reverse sequences are unused.
- Test starts are 1, 181, 361, 541, 721 and 901. Website starts are fixed at
  1, 361 and 721 before annotation, training or prediction. All six test windows
  are scored; gallery selection is not based on their scores.
- Invert estimated OpenCV world-to-camera matrices, project camera positions
  and headings onto the first-camera horizontal plane, then express each target
  relative to context frame 4. Positive x is forward, positive y/yaw is left.
  Normalize each window so its median adjacent planar translation is 0.25
  relative units. Keep yaw in radians.
- Initially estimate the selected 20 frames jointly. A step exceeding ten median
  steps triggers a retry using all 77 consecutive source frames in that same
  window. Select the original 20 poses afterwards and apply the same check.
  Retain both attempts and all additional image hashes. This heuristic checks
  continuity, not pose accuracy.
- Initialize from the original Nav15 Stage 2 joint100k OpenNWM checkpoint,
  SHA256 `aabff9b5a3ab62a9145073892653a81019782511272933b6acfb41e21b48b4f4`.
  Use the lunar adaptation recipe: 1,000 AdamW updates, learning rate 1e-5,
  effective batch 8, microbatch 2, EMA decay 0.99, seed 0, cached VAE posterior
  means. Only training pairs update weights. Validation is monitored every
  100 updates; the checkpoint is always the final update.
- Compare original and adapted EMA models with identical context, actions,
  per-sample random seeds, 250 diffusion steps and 224-pixel preprocessing.
  Each horizon is predicted directly from the four observed frames; generated
  frames are not fed back. Score PSNR, Gaussian RGB SSIM and AlexNet LPIPS on
  the 96 paired future images, with a repeat-last-context baseline.
- Website media contains the shared observed anchor followed by 16 paired
  targets, played at 4 FPS. Its 4.25-second duration is presentation timing.

## Interpretation

The action labels are offline visual estimates derived from the whole clip,
including future reference images. OpenNWM sees only the four observed RGB
frames and the estimated action vectors, but this is a retrospective
action-conditioned prediction test, not a demonstration of predicting unknown
future controls or executing a closed-loop navigation policy.

Translation scale is relative, not metres. Source timestamps and camera
calibration are unavailable. Roll, pitch and vertical movement are discarded;
large camera swings and low-overlap views can produce uncertain pose labels.
One training window (`synmars-forward-3-0161`) needed the dense-frame retry:
the largest selected-frame step fell from 10.77 to 1.72 median steps. A nearly
textureless sky view at intermediate source frame 194 still has an isolated
pose spike; this frame is not one of the 20 selected frames. Neighboring
sky-heavy views remain uncertain. Passing continuity checks does not establish
ground-truth pose accuracy.
Distinct route IDs establish route isolation, not geographically disjoint
terrain. This small, single-seed adaptation experiment is separate from the
paper's benchmark tables.

## Reproduction and integrity

`tools/prepare_synmars_pose_actions.py` records the fixed selection, camera
estimates, source-image hashes and data manifests. `tools/finetune_lusnar.py`
supports the same manifest format for this Mars run; its historical filename
does not select lunar data. `tools/evaluate_planetary.py` produces matched
before/after inference, and `tools/score_planetary_predictions.py` scores the
saved images without resizing them.

`tools/build_website_mars_gallery.py` validates completion, all six test IDs,
training and inference checkpoint hashes, source/pose hashes, disjoint routes,
the fixed gallery selection and pose-to-action consistency before exporting.
Each public clip manifest retains sanitized source IDs and hashes. Tests cover
coordinate signs, cumulative actions, gauge/scale invariance, invalid geometry,
split isolation and export tampering.

The annotation manifest SHA256 is
`92b2cfa9c17f4da9b636c666f2e487801427a9f9fdd8df115a9a1b4ad7268dd3`.
All 837 pose-input image hashes were checked, including the dense retry's
intermediate frames. The 660 cached images belong only to training and
validation windows; no test image is included in the training cache.

The final adapted checkpoint SHA256 is
`d51c8ee18e1a6b5c208a6a80e2bf98d6b617fc705eccf02d7e460950f040971f`.
Training completed successfully after 1,000 updates, using 234.51 seconds of
measured training/save time and a peak allocated GPU memory of 4.46 GiB.
Validation EMA loss decreased from 0.158346 to 0.126813. These are diffusion
training losses, not image-quality metrics or pose errors.

Before the adapted inference, the evaluator's descriptive `feedback` field was
corrected to state that direct horizons use fixed observed context. The actual
inference branches were unchanged. The original baseline manifest, source
snapshot and scores are retained; its corrected manifest was rescored, with
all 192 frame-metric rows (model and copy baseline) exactly unchanged.

## Results and website validation

Both inference runs completed all 96 future frames successfully. The following
means include all six predeclared held-out windows and all 16 horizons; the
shared observed anchor in website videos is excluded from scoring.

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: |
| Original OpenNWM | 15.250662 | 0.538051 | 0.561043 |
| Mars-adapted OpenNWM, final EMA | 17.814053 | 0.644203 | 0.451616 |
| Repeat last observed frame | 17.858030 | 0.600379 | 0.427776 |

Adaptation improves the three aggregate image metrics over the original model.
Against repeating the last observed frame, the adapted model has higher SSIM,
slightly lower PSNR, and worse LPIPS. These results do not establish superiority
over that simple baseline. Predicted terrain and rock geometry can still drift
from the reference, especially at longer horizons.

Per-window changes below are adapted minus original. Positive PSNR/SSIM and
negative LPIPS indicate improvement. SSIM and LPIPS improve for all six windows;
PSNR regresses slightly for window 0361, which remains in the predeclared gallery.

| Route 9 window start | Δ PSNR | Δ SSIM | Δ LPIPS | Adapted model worse than copy baseline |
| --- | ---: | ---: | ---: | --- |
| 0001 (gallery) | +1.988076 | +0.037757 | −0.069222 | PSNR, SSIM, LPIPS |
| 0181 | +4.375918 | +0.084156 | −0.177792 | PSNR, SSIM, LPIPS |
| 0361 (gallery) | −0.155832 | +0.095009 | −0.062817 | PSNR, LPIPS |
| 0541 | +1.035720 | +0.090412 | −0.054487 | None |
| 0721 (gallery) | +4.222410 | +0.106984 | −0.132359 | LPIPS |
| 0901 | +3.914052 | +0.222597 | −0.159884 | LPIPS |

The export completed successfully: three predeclared Mars sliders were added
and the 12 existing Earth/Moon cards were preserved. Each Mars slider contains
17 frames at 224 × 224, including its shared observed anchor, with a synchronized
estimated-path cursor. Public manifests retain checkpoint, pose, image and
media hashes without machine-local paths. The frozen media mirror predates the
new assets, so the Mars directory is served directly from the site using
`localAssetPrefixes`; other media continues to use the configured mirror.

Final Chromium validation completed successfully on 2026-10-29 at 03:35 UTC.
The actual media-delivery configuration was used without intercepting media
requests. All 15 reference/prediction pairs decoded and played, and all 60
referenced video/poster assets were available. The maximum sampled pair drift
was 8.745 ms overall and 0.465 ms for Mars. No JavaScript errors, failed requests
or HTTP errors were recorded in that run.

Both English and Chinese were checked at widths 1440, 768, 390 and 320 pixels,
with no horizontal overflow. All 24 Mars card/layout combinations passed
pointer and keyboard slider checks, translated labels, pose-estimation notes
and accessible descriptions. Offscreen pause/release and return-to-view resume
also passed. Mars requests used local assets directly; existing media continued
to use the configured mirror. Reduced-motion behavior was recorded under the
existing site policy, which still allows visible videos to autoplay.

An earlier run observed a transient stall in an existing lunar video served by
the external mirror; the final run played all three lunar pairs successfully.
An earlier mobile pointer assertion was traced to the harness clicking outside
the viewport. Centering the target and verifying the hit target fixed the
harness; no slider implementation change was required. These records are
retained alongside the passing run. The static-site checks mocked the live
inference API as offline and did not validate that separate service.

CPU checks passed: 18 pose/action tests, 13 Mars export tests and 8 existing
planetary-gallery tests, plus 65 media-routing cases and syntax checks for
eight JavaScript modules. The website source and exported media are packaged
together for publication at [OpenNWM](https://b0b8k1ng.github.io/OpenNWM/).
The public website repository's GitHub Pages workflow deploys the committed
static files, including all three Mars pairs and their provenance manifests.
