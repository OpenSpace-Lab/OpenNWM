# DMD2 Pilot Results — 2026-10-19

Four-step students were trained separately for OpenNWM Stage 2 and official NWM CDiT-B. **Inference became faster, but the short runs did not retain the 250-step teachers' perceptual quality.** PSNR/SSIM increased alongside smoother images, while LPIPS worsened.

## Protocol

- CDiT adaptation of [DMD2](https://github.com/tianweiy/DMD2/tree/8d8fa55633d47cfb81bbc7a892e7248f9518763f), with real/fake scores, a conditional GAN, student trajectory simulation, and a pooled-token GAN head.
- Per model: 384 training windows, 200 student updates, 2,000 critic updates, batch size 2, schedule `[999, 749, 499, 249]`.
- Test: 8 windows each from recon/sacson/scand, 24 total, predicting 4 or 16 frames ahead. Train/val/test trajectories are disjoint; test data was not used to tune parameters.
- NVIDIA L20, batch size 1, BF16 autocast, fixed inputs/actions/noise, and VAE posterior means. After warmup, randomized method order and CUDA synchronization were used. Timing averages per-sample medians over three repeats.
- Total latency includes online VAE encoding, denoising, and decoding; it excludes I/O, model loading, and metrics. No FID, long-horizon rollout, or navigation-success evaluation was performed.

## Before and After Distillation

Values compare the 250-step teacher with the four-step student.

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Total latency, ms | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenNWM | 14.215 → 15.613 | 0.4515 → 0.5314 | 0.3598 → 0.4506 | 3574.22 → 84.05 | 42.53× |
| NWM CDiT-B | 13.951 → 15.437 | 0.4460 → 0.5237 | 0.4033 → 0.4938 | 3608.12 → 87.41 | 41.28× |

## Four-Step Controls

| Model | Teacher DDPM LPIPS ↓ | Teacher DMD2 LPIPS ↓ | Student DMD2 LPIPS ↓ | Improvement over teacher DMD2 |
| --- | ---: | ---: | ---: | ---: |
| OpenNWM | 0.5008 | 0.4723 | 0.4506 | 4.58% |
| NWM | 0.5391 | 0.5095 | 0.4938 | 3.09% |

Distillation improves LPIPS over the teacher using the same four-step sampler, but remains below the 250-step teacher's quality. These measurements apply to this pilot configuration and do not establish convergence.

## Downloads and Reproduction

Download students and published test results from [OpenNWM-DMD](https://modelscope.cn/models/LittleBoss/OpenNWM-DMD) and [NWM-DMD](https://modelscope.cn/models/LittleBoss/NWM-DMD). Follow the [training and benchmark commands](DMD2.md) to reproduce the protocol. Students require `FourStepSchedule.sample`; reducing the original DDPM step count is not equivalent.

Benchmark outputs include per-sample metrics, dataset summaries, comparisons, and paired 95% bootstrap intervals. Training saves `latest.pt` for resume and `student.pt` for inference.
