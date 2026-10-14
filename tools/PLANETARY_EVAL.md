# LuSNAR / SynMars-TW exploratory evaluation

Run from the repository root with the existing environment and an available GPU:

```bash
source .env
CUDA_VISIBLE_DEVICES=4 "$NWM_PYTHON" -u tools/evaluate_planetary.py \
  --output-root /path/on/nas/planetary-eval --batch-size 8
"$NWM_PYTHON" -u tools/score_planetary_predictions.py \
  --output-root /path/on/nas/planetary-eval --device cpu
```

`--prepare-only` prepares the exact input manifest and reference images without
loading the model onto a GPU. `--run-dir` selects another imported Stage 2 run;
the default is `NWM_NAV15_STAGE2_RUN`. The directory must contain
`.hydra/config.yaml` and `checkpoints/latest.pth.tar`. Defaults match the Nav15
Stage 2 joint100k checkpoint: EMA, 224 pixels, four context images, 250 DDPM
steps, seed 0. VAE posterior means make context encoding deterministic. This
differs from the standard benchmark's posterior sampling. Saved 8-bit RGB is
used for autoregressive feedback, including on resume. Use a fresh output
directory after changing the checkpoint, input roots, protocol, or source.

Default sample selection is deterministic and does not use prediction quality:

| Dataset | Inputs | Protocol | Paired scoring |
| --- | --- | --- | --- |
| LuSNAR | Four disjoint windows in each of nine scenes | Fixed four-frame context, direct steps 1/4/8/16 | 36 windows × 4 targets |
| LuSNAR | First selected window in each scene | Four observed frames, 16 autoregressive predictions | 9 windows × 16 targets |
| SynMars-TW | 12 evenly spaced left test stills | Repeated initial context; straight/left/right controls, 16 generated steps | None |

LuSNAR timestamps are parsed as integer nanoseconds, duplicate identical GT
rows are removed, and image/pose associations must be within 1 ms. Actual
source timestamps are retained. Four-Hz nearest-image sampling selects only
unique frames within half a sample period of their target grid. Nominal
prediction horizons are 0.25/1/2/4 seconds, with the actual offsets in the
manifest. Translation is divided by each scene's median nonzero planar
displacement on that grid. GT quaternions are wxyz; the adapter explicitly
assumes AirSim body axes +X forward/+Y right/+Z down and reflects Y/yaw into
OpenNWM's forward/left convention. Published camera placement and a
forward-motion audit support the assumption. Actions describe rover-body XY
and yaw; camera mounting, terrain height, pitch and roll are not fully modeled.
These are dataset adaptation assumptions, not a calibrated 6-DoF benchmark.

SynMars-TW's release contains no pose, action, calibration or capture timestamp
truth. Filename order is not used as temporal ground truth. Each still receives
synthetic per-step controls `[0.5,0,0]`, `[0.5,0,0.05]`, and `[0.5,0,-0.05]` in
normalized waypoint units and yaw radians. Alternative controls share diffusion
noise for the same source and step. Four-FPS video playback does not imply a
physical four-second prediction. No future-image accuracy or navigation success
metric is assigned to these samples.

Outputs include `manifest.json`, `model_config.yaml`, source snapshots,
transformed reference images, and unmodified generated predictions. The scorer
produces per-frame PSNR, RGB Gaussian SSIM and LPIPS-Alex against supplied GT,
plus a copy-initial-image baseline with identical targets. Direct and
autoregressive metrics remain separate. Small-set FID is omitted.
`metrics.json`, `metrics.csv`, `overview.jpg`, `report.html`, and comparison
videos preserve both quantitative and qualitative results. Direct-horizon video
frames have unequal temporal gaps; captions identify the model step.

This evaluates one checkpoint and seed on a documented subset. It does not
establish whole-dataset performance or closed-loop navigation capability.

## LuSNAR few-shot comparison

`finetune_lusnar.py` adapts either the Nav15 Stage 2 EMA or the released NWM
CDiT-B EMA without changing the existing training or inference entry points.
The input run needs `.hydra/config.yaml` and `checkpoints/latest.pth.tar`;
use a separate run directory with a checkpoint symlink when importing weights.
Never overwrite the source run. Put outputs and the shared latent cache on NAS.

Build the fixed split with
`planetary_eval_data.build_lusnar_finetune_split(lusnar_root, previous_manifest_path)`
and save its returned dictionary as JSON. The previous evaluation manifest is
required to exclude its complete windows plus a four-second buffer. Selection
is independent of generated images and scores:

| Split | Scenes and temporal region | Windows | Future targets |
| --- | --- | ---: | ---: |
| Train | Moon_1–5, first 65% | 30 | 480 |
| Validation | Moon_6 | 8 | 128 |
| Same-scene test | Moon_1–5, after 75% | 10 | 160 |
| Cross-scene test | Moon_7–9 | 24 | 384 |

Every window contains four context frames and sixteen future frames. All 1,440
source images are distinct across windows and splits. One translation scale is
fitted to adjacent displacements inside the 30 training windows and reused for
validation and testing. Temporal separation does not rule out revisited places;
the cross-scene test is the primary generalization comparison. SynMars-TW is
excluded because it does not supply reliable action supervision.

```bash
source .env
# Replace these paths with an external output directory and an imported run.
lusnar_root=/path/on/nas/lusnar-fewshot
lusnar_run=/path/on/nas/imported-model
PYTHONDONTWRITEBYTECODE=1 "$NWM_PYTHON" tools/finetune_lusnar.py check \
  --data-manifest "$lusnar_root/data_manifest.json" --run-dir "$lusnar_run"
CUDA_VISIBLE_DEVICES=4 PYTHONDONTWRITEBYTECODE=1 "$NWM_PYTHON" tools/finetune_lusnar.py cache \
  --data-manifest "$lusnar_root/data_manifest.json" --run-dir "$lusnar_run" \
  --cache-root "$lusnar_root/cache" --microbatch 2 --memory-fraction .45
CUDA_VISIBLE_DEVICES=4 PYTHONDONTWRITEBYTECODE=1 "$NWM_PYTHON" tools/finetune_lusnar.py train \
  --data-manifest "$lusnar_root/data_manifest.json" --run-dir "$lusnar_run" \
  --cache-root "$lusnar_root/cache" --output-root "$lusnar_root/runs/model" \
  --steps 1000 --batch-size 8 --microbatch 2 --lr 1e-5 --ema-decay .99 \
  --seed 0 --memory-fraction .45
```

The common recipe freezes the VAE, caches posterior means for train/validation
only, and trains the active denoiser and action branch with AdamW. OpenNWM's
unused geometry/latent adapters stay frozen. NWM keeps its native XY min/max
normalization and yaw in radians; OpenNWM uses its existing real-action adapter.
Both use identical training-pair order, diffusion timesteps/noise, effective
batch size, learning rate, and update count. Validation monitors fixed-noise EMA
diffusion loss every 100 steps. The final step is fixed in advance; test scores
do not select checkpoints. Only the final model/EMA checkpoint is saved; an
interrupted training run must restart in a fresh output directory.

For held-out evaluation, write a JSON containing `samples`, `data_metadata`,
and `limitations`, then pass it to `evaluate_planetary.py --samples-manifest`
with `--model-label`. Direct samples use anchor-relative `actions`; autoregressive
samples must use `incremental_actions`, each expressed in the preceding body
frame. Select the same IDs/horizons for both models before and after fine-tuning.
`--memory-fraction .45` optionally caps CUDA allocator use without changing the
sampling protocol. The scorer reads the model label from the saved manifest.

### Website visualization export

Select `cross_scene_test`, window 0 in Moon_7–9 from the complete data manifest,
and run the fine-tuned OpenNWM checkpoint on all sixteen direct horizons with
`evaluate_planetary.py --samples-manifest`. These videos require actual dense
predictions for steps 1–16; the earlier 1/4/8/16 evaluation animation is not a
substitute. Export the completed direct manifest and the matching sixteen-step
autoregressive evaluation as follows:

```bash
source .env
PYTHONDONTWRITEBYTECODE=1 "$NWM_PYTHON" tools/build_website_space.py \
  --finetuned-eval "$lusnar_root/visualization_20261008/direct/manifest.json" \
  --autoregressive-eval "$lusnar_root/eval/opennwm_after/manifest.json"
```

The exporter adds three LuSNAR scenes, prediction/reference comparison videos,
and public provenance without replacing the original planetary examples.
Running the default builder later preserves these fine-tuned scenes. Temporary
frames are cleaned from NAS automatically; only final assets enter the website.
These fixed held-out windows are qualitative examples, not navigation results.

To include the equally fine-tuned NWM checkpoint and update the original
Yutu-2 / LuSNAR single-image examples, add all three comparison options:

```bash
PYTHONDONTWRITEBYTECODE=1 "$NWM_PYTHON" tools/build_website_space.py \
  --finetuned-eval "$lusnar_root/visualization_20261008/direct/manifest.json" \
  --autoregressive-eval "$lusnar_root/eval/opennwm_after/manifest.json" \
  --nwm-eval "$lusnar_root/website_model_comparison_20261008/nwm_release/manifest.json" \
  --nwm-autoregressive-eval "$lusnar_root/eval/nwm_release_after/manifest.json" \
  --single-image-eval "$lusnar_root/website_model_comparison_20261008/opennwm/manifest.json"
```

The two single-image inputs retain their original synthetic forward controls
and have no paired future ground truth; Moon_1 is a fine-tuning scene, so its
example is not a held-out benchmark. Both fine-tuned models share the new
evaluation noise and RGB-feedback protocol, which does not reproduce the
archived website predictions pixel for pixel. Default rebuilds preserve all
five upgraded lunar examples, and the two-option export retains their NWM
comparison links. Mars examples remain unchanged.
