# LuSNAR curved-window gallery — 2026-10-29

The three lunar examples use recorded turning trajectories from the held-out **Moon_7, Moon_8 and Moon_9** scenes. These numbers are LuSNAR scene identifiers. The model was trained on Moon_1–5; Moon_6 was used for validation. No new training or checkpoint selection was performed for this gallery update.

## Selection before inference

For each held-out scene, enumerate complete windows containing four observed frames and sixteen future frames on the nominal 4 Hz grid. Require twenty distinct nearest source frames, each within 125 ms of its grid timestamp; retain the original evaluation-window exclusions and their sixteen-grid-step (4 s) buffers; require at least one meter of future travel. Rank by recorded future yaw range, rounded to two decimal degrees, breaking ties by the earliest grid anchor. Predictions are not consulted when choosing a window.

| Scene | Grid anchor | Source-frame anchor (zero-based) | Eligible windows | Recorded yaw range | Maximum absolute lateral displacement | Actual future span |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Moon_7 | 180 | 401 | 416 | 26.95° | 1.448 m | 3.9835 s |
| Moon_8 | 39 | 84 | 292 | 37.24° | 3.457 m | 3.9668 s |
| Moon_9 | 713 | 1589 | 615 | 52.12° | 5.167 m | 4.0954 s |

The curves are the dataset's recorded turns. No trajectory is bent, synthesized, interpolated, or selected for favorable prediction quality. These are qualitative examples selected for turning motion, not a random benchmark sample.

## Model and timing

- Original OpenNWM Nav15 Stage 2 checkpoint, fine-tuned on 30 Moon_1–5 windows / 480 context–target pairs for 1,000 updates; AdamW, learning rate 1e-5, effective batch size 8, EMA 0.99, seed 0.
- The fixed translation scale is **0.4863296751176136 m**, fitted only on the original training windows. Yaw remains in radians. No normalization is fitted on Moon_7–9.
- Direct prediction: every future horizon uses the same four observed context frames and the corresponding recorded anchor-relative body pose. These sixteen predictions are not an autoregressive rollout.
- DDPM 250 steps, seed 0, posterior-mean context encoding, 224 × 224 RGB.
- Each video contains the shared observed anchor plus all sixteen future frames, at 4 FPS: 17 frames / 4.25 s. The future horizon is nominally 4 s. Original nonuniform capture timestamps and offsets are retained separately.
- Trajectory positions are `[0, 0]` followed by the sixteen recorded forward/left coordinates in meters. Planar body poses do not include height, roll, pitch, or calibrated camera extrinsics.

Fine-tuned checkpoint SHA-256:
`1dbeb11d9d86915e18cbe9fa064f38f9f5e05c3e727244a7e026675a128a6b56`

Original training-data manifest SHA-256:
`8e14a6d7b65a277bce97869dd739316f34fe3d2a850682271956d24a1b8108b3`

Frozen curved-window samples manifest SHA-256:
`ff7ab57aff0fbd29d32c9df1bc1f802c90762b071473b0a39b6e500b11d41237`

## Reproduction and public provenance

The NAS experiment `lusnar_curved_gallery_20261029` retains the GT-only selection script, frozen samples, selection audit, execution command/PID/log/exit status, inference outputs, and media-validation results. The inference reuses `lusnar_fewshot_20261007/runs/opennwm` and passes the frozen `samples.json` to `tools/evaluate_planetary.py`.

The CPU exporter supports independent new assets and a separate catalog-items handoff:

```sh
"$NWMPY" tools/build_website_lunar_gallery.py \
  --direct-manifest "$CURVED_EXPERIMENT/direct/manifest.json" \
  --direct-only \
  --training-data-manifest "$LUSNAR_FINETUNE/data_manifest.json" \
  --asset-prefix assets/space/lusnar-curved-20261029 \
  --site-root "$CURVED_EXPERIMENT/site" \
  --items-output "$CURVED_EXPERIMENT/lunar-items.json"
```

It recomputes all metric and normalized actions from the original `gt.txt`, checks source timestamps and file hashes, verifies the training scale and held-out scenes, and retains the original exclusion buffers. The previous window00 export mode remains supported.

Public source manifests retain real sample identities, complete metric actions, timestamps, source-image and annotation hashes, checkpoint and inference hashes, and selection details:

- [Moon 7 source manifest](../assets/space/lusnar-curved-20261029/moon-7/manifest.json)
- [Moon 8 source manifest](../assets/space/lusnar-curved-20261029/moon-8/manifest.json)
- [Moon 9 source manifest](../assets/space/lusnar-curved-20261029/moon-9/manifest.json)
- [Recorded trajectory audit](../assets/space/lusnar-curved-20261029/trajectory-provenance.json)

## Validation results

Inference completed successfully (exit status 0): 48 future frames across the three fixed samples, with 98.4 s reported for inference after initialization. All six published videos fully decode and contain 17 frames at 4 FPS, 224 × 224 pixels, and 4.25 s duration. The seventeen plotted positions match the recorded metric actions exactly. Source timestamps, normalization, source-image hashes, annotation hashes, checkpoint identity, and held-out scene constraints passed validation.

Six deliberately altered inputs were rejected: metric poses, normalized actions, translation scale, GT timestamps, exclusion rules, and checkpoint identity. A separate legacy-mode export reproduced all three previous catalog entries exactly. Detailed results are retained in the experiment's `validation.json` and `validation.log`.

An early/middle/final GT–prediction contact sheet was inspected after the samples were fixed. Turning motion is visible, but later predictions show substantial drift in rock and crater-edge placement, particularly in Moon 8 and Moon 9. Samples were not replaced after this inspection. These examples demonstrate recorded turning trajectories and do not establish an improvement in prediction accuracy.
