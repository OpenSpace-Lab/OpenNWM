# DMD2 Four-Step Distillation

A separate distillation workflow supports OpenNWM Stage 2 and original NWM CDiT-B weights. See the [pilot results](DMD2_RESULTS.md). Published bundles include student weights, portable configurations, the sampler, examples, and test results:

- [OpenNWM-DMD on ModelScope](https://modelscope.cn/models/LittleBoss/OpenNWM-DMD)
- [NWM-DMD on ModelScope](https://modelscope.cn/models/LittleBoss/NWM-DMD)

The implementation adapts [DMD2](https://github.com/tianweiy/DMD2/tree/8d8fa55633d47cfb81bbc7a892e7248f9518763f) to CDiT. It uses a frozen real-score teacher, a full-parameter student, a trainable fake-score CDiT, normalized distribution-matching gradients, a default 10:1 critic/student update ratio, and a logistic GAN loss. The GAN head pools final-layer CDiT tokens. Student training uniformly selects one of four steps, simulates earlier student steps without gradients, and backpropagates through the selected step only. This adaptation uses deterministic VAE posterior means and no text CFG.

The fixed schedule is `[999, 749, 499, 249]`: predict x0, add fresh noise at the next time, and return the final x0. The sampler reads only epsilon from CDiT's epsilon/variance output. **Use `dmd2.FourStepSchedule.sample` or the benchmark below.** Setting `eval_diffusion_steps=4` in the original DDPM entry point is not equivalent. Upstream SDXL distillation weights are incompatible with NWM.

## Prepare Data

Run from the repository root after setting data, VAE, and interpreter paths in `.env`. Set `NWM_NAV15_STAGE2_RUN` to the Stage 2 run directory. Check GPU availability with `nvidia-smi` and free space with `df -h`; keep large outputs outside the repository.

```bash
source .env
export DMD2_ROOT="$OPENNWM_WORK_ROOT/dmd2"
CUDA_VISIBLE_DEVICES=0 "$NWM_PYTHON" -u -B nwm/distill_dmd2.py prepare \
  --source-run "$NWM_NAV15_STAGE2_RUN" --output "$DMD2_ROOT/data"
```

Defaults select 128 train, 8 val, and 8 test trajectories from each of recon/sacson/scand, using one fixed window per trajectory and checking split separation. Predictions target 4 or 16 frames ahead. `manifest.json` records trajectories, frames, and source paths; `samples.pt` stores fixed VAE encodings and raw real actions. Original NWM applies its action normalization before model calls; OpenNWM uses its real-action adapter. This small sample is separate from the full official test protocol.

## Train Each Model

```bash
CUDA_VISIBLE_DEVICES=0 "$NWM_PYTHON" -u -B nwm/distill_dmd2.py train \
  --source-run "$NWM_NAV15_STAGE2_RUN" \
  --checkpoint "$NWM_NAV15_STAGE2_RUN/checkpoints/latest.pth.tar" \
  --data "$DMD2_ROOT/data/samples.pt" --output "$DMD2_ROOT/opennwm" \
  --generator-updates 200 --critic-ratio 10 --batch-size 2

export NWM_RELEASE_RUN="$OPENNWM_WORK_ROOT/runs/nwm-release"
CUDA_VISIBLE_DEVICES=0 "$NWM_PYTHON" -u -B nwm/distill_dmd2.py train \
  --source-run "$NWM_RELEASE_RUN" \
  --checkpoint "$NWM_RELEASE_RUN/checkpoints/0100000.pth.tar" \
  --data "$DMD2_ROOT/data/samples.pt" --output "$DMD2_ROOT/nwm" \
  --generator-updates 200 --critic-ratio 10 --batch-size 2
```

The NWM run must contain the official CDiT-B checkpoint `cdit_b_100000.pth.tar` under the configured checkpoint path. These short runs are pilots, not evidence of convergence. Add `--activation-checkpointing` or reduce batch size if needed. Logs record gradients, losses, update times, and memory; nonfinite losses or gradients stop training.

`latest.pt` contains the student, critic, both optimizers, and RNG state; `student.pt` contains final inference weights. `--resume` continues the same output directory after checking teacher/data hashes and key settings. Increase `--generator-updates` to extend training. The default save interval is 25 student updates. Keep `latest.pt` if further training is needed; source teacher weights are preserved.

## Quality and Speed

```bash
CUDA_VISIBLE_DEVICES=0 "$NWM_PYTHON" -u -B nwm/benchmark_dmd2.py \
  --source-run "$NWM_NAV15_STAGE2_RUN" \
  --checkpoint "$NWM_NAV15_STAGE2_RUN/checkpoints/latest.pth.tar" \
  --student "$DMD2_ROOT/opennwm/student.pt" --data "$DMD2_ROOT/data/samples.pt" \
  --output "$DMD2_ROOT/opennwm-test" --label OpenNWM --repeats 3
```

For NWM, replace the source run, checkpoint, student, output, and label. Four methods share inputs, actions, sample seeds, and initial noise: teacher 250-step DDPM, teacher 4-step DDPM, teacher 4-step DMD2, and student 4-step DMD2. Comparing the last two isolates the effect of distillation.

Each method is warmed up and runs in randomized order per sample. The default three repeats produce a per-sample median, then a mean across samples. Timing uses batch size 1, BF16 autocast, and CUDA synchronization. Reports separate denoising from total VAE encode/denoise/decode time, excluding loading, disk I/O, and metrics.

Quality metrics are PSNR, TorchMetrics Gaussian SSIM, and AlexNet LPIPS, with per-sample results, dataset summaries, and paired bootstrap intervals. FID is omitted for this small sample. Outputs are `report.md`, `results.json`, `comparison.png`, and sample images. This benchmark does not measure autoregressive rollouts or navigation success.

Run CPU tests with `CUDA_VISIBLE_DEVICES='' "$NWM_PYTHON" -B -m pytest -q nwm/tests/test_dmd2.py`.
