# NWM

Training and evaluation retain CompACT's CDiT, Gaussian diffusion, SD-VAE, action encoding, losses, and EMA. The unified entry point adds validation monitoring, full checkpoints, and local training curves. [DMD2 four-step distillation](DMD2.md) provides a separate workflow for OpenNWM Stage 2 and original NWM weights.

## Setup

Complete the [installation and path setup](../README.md#environment-setup). Run commands from the repository root:

```bash
source .env
```

Paths can be overridden through environment variables or Hydra `key=value` arguments. Keep data and large weights outside the repository. Dependency changes may affect numerical reproducibility across PyTorch/CUDA versions.

Data follows `$NWM_DATA_ROOT/{recon,sacson,scand,tartan}/<trajectory>/{0.jpg,1.jpg,...,traj_data.pkl}`. `sacson` is HuRoN; the `tartan_drive` split uses the `tartan` data directory. Split files are `$NWM_SPLIT_ROOT/<dataset>/{train,val,test}/traj_names.txt`.

The historical training field `dataset.datasets.<name>.test` selects periodic validation and defaults to **val**. Final inference and evaluation use **test** separately. Override paths with `dataset.datasets.recon.train=...` or `.test=...`. Reuse existing validation splits when available; generating new held-out splits changes the data recipe.

## Training

Append Hydra overrides to any command. `--gpus` sets the task's GPU count; use `CUDA_VISIBLE_DEVICES` to select devices.

```bash
python run.py nwm train --mode baseline --gpus 1 --run-dir "$OPENNWM_WORK_ROOT/runs/baseline"
python run.py nwm train --mode original --gpus 8 --run-dir "$OPENNWM_WORK_ROOT/runs/nwm-original"
python run.py nwm train --resume "$OPENNWM_WORK_ROOT/runs/baseline" --gpus 1
```

`baseline` uses CompACT CDiT-B. `original` uses CDiT-XL with the original NWM dataset order, optimizer, sampling, and legacy action encoding. Validation defaults to `val`; reproducing historical test-set monitoring requires `NWM_MONITOR_SPLIT=test` and the original training manifests.

| Mode | Purpose |
| --- | --- |
| `timept` | Stage 1: action-free pretraining |
| `geopt` / `idmpt` | Stage 1: geometric / IDM offline proxy actions |
| `latentpt` / `latentonlypt` | Stage 1: LAM actions, mixed with action-free inputs / actions only |
| `time_reset` / `geo_reset` / `idm_reset` / `latent_reset` | Stage 2: new real-action adapter warmup, then joint finetuning |
| `latent_align` | Stage 2: LAM teacher embedding alignment, then joint finetuning |
| `latent_real_to_latent` | Stage 2: real-to-latent mapping, then joint finetuning |
| `latent_state_controller` | Stage 2: state-conditioned controller, then joint finetuning |
| `no_pretrain` | Random initialization without Stage 1 |
| `go2_adapt` | Adapt a compatible full NWM checkpoint to Go2 |

Stage 1 requires NavAnywhere frames, disjoint train/val recipes, and a training SD-VAE posterior cache. The default `seed=20260901` matches the bundled recipe; custom recipes must use their corresponding seed. Action-conditioned modes also need train/val proxy caches.

```bash
# Prepare the caches below first; .env supplies recipe and cache paths.
python run.py nwm train --mode timept --run-dir "$OPENNWM_WORK_ROOT/runs/timept"
python run.py nwm train --mode latentpt --run-dir "$OPENNWM_WORK_ROOT/runs/latentpt"
```

Stage 2 uses posterior caches for the four action datasets and the corresponding Stage 1 checkpoint:

```bash
python run.py nwm train --mode latent_reset --run-dir "$OPENNWM_WORK_ROOT/runs/reset" \
  finetune.stage1_checkpoint="$OPENNWM_WORK_ROOT/runs/latentpt/checkpoints/best.pth.tar"
python run.py nwm train --mode latent_align --run-dir "$OPENNWM_WORK_ROOT/runs/align" \
  finetune.stage1_checkpoint="$OPENNWM_WORK_ROOT/runs/latentpt/checkpoints/best.pth.tar"
python run.py nwm train --mode no_pretrain --run-dir "$OPENNWM_WORK_ROOT/runs/no-pretrain"
```

Periodic and stage-end validation use fixed samples and noise, restoring Python/NumPy/Torch RNG afterward. Defaults are `eval_every=5000` and `training.eval_num_batches=4`; reduce batch count and `training.eval_batch_size` for small validation sets. Outputs include `viz/` previews, `metrics.jsonl`, and `curves.png`. WandB is disabled by default.

Resume from `checkpoints/latest.pth.tar`; `best.pth.tar` tracks the lowest validation metric. Stage 2 also saves `best_warmup`, `best_joint`, and `transition.pth.tar`. The latest three periodic checkpoints are retained; `training.keep_periodic=0` keeps all.

Full resume restores optimizers, scheduler, scaler, EMA, per-device RNG, and data cursors. Exact continuation requires the same GPU count, batch size, data, and configuration. Older checkpoints can restore only the state they contain.

## Cache Preparation

Existing compatible caches can be reused. Cache generation saves complete trajectories atomically and skips completed trajectories on rerun. SD-VAE uses batch size 128, BF16, and cached posterior mean/logvar; training samples the posterior online.

```bash
python run.py nwm cache-vae --data-root "$NWM_NAVANYWHERE_ROOT" \
  --recipe "$NWM_NAVANYWHERE_RECIPE" --output-root "$NWM_NAVANYWHERE_VAE_LATENT_ROOT" --vae-model "$NWM_VAE_PATH"
python run.py nwm cache-vae --data-root "$NWM_DATA_ROOT" \
  --split-root "$NWM_SPLIT_ROOT" --output-root "$NWM_FINETUNE_VAE_LATENT_ROOT" --vae-model "$NWM_VAE_PATH"
# LAM cache generation requires the LAM environment.
"$LAM_PYTHON" nwm/prepare.py lam --data-root "$NWM_NAVANYWHERE_ROOT" \
  --recipe "$NWM_NAVANYWHERE_RECIPE" --output-root "$NWM_LATENT_PROXY_ROOT" --checkpoint "$LAM_PIXEL_ACTION_CHECKPOINT"
"$LAM_PYTHON" nwm/prepare.py lam --data-root "$NWM_NAVANYWHERE_ROOT" \
  --recipe "$NWM_NAVANYWHERE_VAL_RECIPE" --output-root "$NWM_NAVANYWHERE_VAL_PROXY_ROOT" --checkpoint "$LAM_PIXEL_ACTION_CHECKPOINT"
"$LAM_PYTHON" nwm/prepare.py lam --data-root "$NWM_DATA_ROOT" \
  --split-root "$NWM_SPLIT_ROOT" --output-root "$NWM_FINETUNE_LATENT_ROOT" --checkpoint "$LAM_PIXEL_ACTION_CHECKPOINT"
```

These commands use the PixelActionLAM checkpoint from `.env`. Use new cache directories when changing the checkpoint. VAE caches are needed only for training; validation encodes images online. Stage 1 needs separate train/val proxy caches; Stage 2 validation uses real actions.

Bundled train/val recipes are provided. For custom data, generate a manifest with `python run.py nwm recipe --root /data/navanywhere --output /data/recipe.json --seed 20260901`, then split by disjoint trajectories.

LAM caches store raw `z_mu` means for current-to-goal pairs with absolute offset ≤8: `{source}/{trajectory}.pt` contains `frame_pairs:[N,2]` and `motion:[N,32]`. Geo/IDM caches use the same interface with dimensions 3/7. Their external estimators are separate from NWM. Training offsets remain [-64,64] with the original local proxy mask.

## Evaluation

Import a downloaded checkpoint into a new inference directory:

```bash
python run.py nwm import-checkpoint --checkpoint /weights/joint_0100000.pth.tar --run-dir /runs/imported
# Add --config /weights/config.yaml if the checkpoint has no saved configuration.
python run.py nwm infer exp_dir=/runs/imported ckp=latest eval_type=time
```

Import uses the saved configuration, links the original weights, and resolves the VAE through `NWM_VAE_PATH`. The source experiment is unchanged.

Inference supports `time` and `rollout`, skips existing images, and derives noise from each sample. A run directory requires `.hydra/config.yaml` and a checkpoint. To generate ground truth, predictions, metrics, and plans:

```bash
python run.py nwm infer exp_dir="$OPENNWM_WORK_ROOT/runs/reset" ckp=best \
  eval_type=time gt=1 output_dir="$OPENNWM_WORK_ROOT/eval"
python run.py nwm infer exp_dir="$OPENNWM_WORK_ROOT/runs/reset" ckp=best \
  eval_type=time prediction_dir="$OPENNWM_WORK_ROOT/eval/pred"
python run.py nwm eval gt_dir="$OPENNWM_WORK_ROOT/eval/gt" exp_dir="$OPENNWM_WORK_ROOT/eval/pred" \
  'datasets_to_eval=[recon,scand,sacson]' eval_types=time
python run.py nwm plan exp_dir="$OPENNWM_WORK_ROOT/runs/reset" ckp=best \
  output_dir="$OPENNWM_WORK_ROOT/plan/reset"
```

For rollout, change both inference commands to `eval_type=rollout` and metrics to `eval_types=rollout`. Default rates are 1 and 4 fps; pass `'rollout_fps_values=[1,2,4]'` to all three commands to extend them.

Inference logs report the actual ground-truth and prediction paths. Metrics expect `<dataset>/time/<sample>/*.png` or `<dataset>/rollout_1fps/<sample>/*.png`. LPIPS, DreamSim, and FID preserve their original definitions; saved accumulators, FID statistics, and cursors allow resume. Completed JSON results are reused. Change the output directory or set `resume=false` when inputs or parameters change.

Inference and planning record checkpoint identity and configuration. A changed `latest` or `best` requires a new output directory. Planning also saves per-sample results for resume. Prepare public pretrained metric weights before offline use.

Navigation inference and planning require a Stage 2 checkpoint with a trained real-action adapter. Stage 1 is monitored using its matching proxy-action or action-free validation protocol.
