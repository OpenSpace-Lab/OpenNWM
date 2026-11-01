# MarsLab physics-action adaptation — 2026-10-29

**Experiment status: capture, preparation, 1,000-update adaptation and scoring of all 96 held-out targets are complete. Gallery export and public-site browser checks are tracked separately below.**

火星示例使用 MarsLab 的地形与 Perseverance 资产，动作由物理仿真器的实际位姿计算。数据划分为 30 个训练片段、3 个验证片段、6 个测试片段；测试轨迹未参与训练，但与训练共用同一地形。1,000 次更新后，全部 96 个测试目标的平均指标优于原模型和重复最后一帧基线；部分窗口仍不如静帧基线，近景岩石几何与纹理也仍有误差。

## Simulator and assets

The experiment uses the official [MarsLab](https://github.com/kimhoyun-robotair/MarsLab) source archive at commit [`a406ca245bfbdf0faaff262a3c63d76110237e3d`](https://github.com/kimhoyun-robotair/MarsLab/commit/a406ca245bfbdf0faaff262a3c63d76110237e3d), its `main_crater` terrain and Perseverance rover assets. A minimal ROS-free Isaac Sim 4.1 recorder loads these assets and runs their physical rover controls. Upstream MarsLab targets Isaac Sim 5.x; this experiment does not reproduce the complete upstream application.

After initial placement, the rover moves through PhysX wheel drive, steering, contact and suspension dynamics. The recorded actions come from the realized body position and orientation, rather than image-estimated poses or commanded velocities. “Ground truth” here means simulator physics state, not measurements collected on Mars. The velocity commands are retained separately for audit.

Physics runs at 60 Hz with 3.72 m/s² gravity; synchronized RGB and body poses are sampled at 4 Hz. Physics uses the CPU and rendering uses the GPU. Images are 224 × 224, with a 90° horizontal field of view and 6.6° downward camera pitch. Lighting is a fixed sun and dome, with intensities 1000 and 250 respectively; the upstream atmospheric-scattering pipeline is not used.

Each episode has 3 seconds of settling and 6 seconds of driving before capture. A further 128 render updates load textures while physical time, physics step, body position and quaternion remain fixed. The exporter checks those values directly, including equality of camera and physics timestamps, and rejects missing or nonfinite audit data. These extra renders are excluded from the dataset.

An earlier recording version showed textures loading during its first 8–12 captured frames. That entire version was excluded from training, validation, testing and the gallery. All ten trajectories were rerecorded with the same byte-identical frozen episode plan and the uniform render warmup. Pilot recordings are also excluded from the model data; this correction did not change the split or select clips by prediction quality.

Isaac Sim 4.1 reproduced a native plugin-unloading crash after two earlier pilots. The final recorder therefore explicitly uses `process-exit`: verify image counts, command counts, synchronization and cadence; flush files and directories with `fsync`; record the completion status; then call `os._exit(0)`. All ten final external process exit codes are zero. The metadata retains `native_cleanup_completed=false` and `native_cleanup=not_attempted`; native plugin cleanup was not repaired or completed.

## Data and action definition

The frozen dataset contains 780 source RGB frames in ten independent trajectories. All windows contain 20 consecutive frames, with no overlapping windows. The first four frames are observed context; the fourth is the anchor and the remaining sixteen are future targets.

| Split | Trajectories | RGB frames | 20-frame windows | Target pairs |
| --- | --- | ---: | ---: | ---: |
| Train | `train_00`–`train_05`, 6 × 25 s | 600 | 30 | 480 |
| Validation | `val_00`, 15 s | 60 | 3 | 48 |
| Same-scene test | `test_00`–`test_02`, 3 × 10 s | 120 | 6 | 96 |

Each test trajectory supplies windows starting at source frames 0 and 20. The gallery selection is fixed to the first window of each test trajectory: `marslab-test_00-0000`, `marslab-test_01-0000` and `marslab-test_02-0000`. All six windows are evaluated. Training, validation and test do not reuse image paths or identical source-image bytes across splits. They share the same terrain, so visual overlap remains possible; this is not a cross-terrain generalization experiment.

For target position displacement `(Δx, Δy)` in the world plane and anchor heading `ψ`, the metric action is:

```text
forward =  cos(ψ) × Δx + sin(ψ) × Δy
left    = −sin(ψ) × Δx + cos(ψ) × Δy
yaw     = wrap(target_heading − ψ)
```

The saved `actions_meters_radians` preserve these metric values. To match the model's action scale, XY is multiplied by one shared training-only scale; yaw remains in radians. The median adjacent planar step across the six training trajectories is **0.1744947016343546 m**, giving **1.4327082579496493 model units per meter** (`0.25 / median_step`). Validation and test use this same scale. Public path plots use the unscaled metric actions. Height, pitch and roll are discarded by this SE(2) action interface.

## Frozen training and evaluation protocol

The run starts from the original OpenNWM Nav15 Stage 2 EMA model, using the existing joint real-action fine-tuning interface. The VAE is frozen; only training and validation images enter the posterior-mean cache, and only training pairs update model weights.

The fixed recipe is 1,000 AdamW updates, learning rate 1e-5, betas (0.9, 0.999), weight decay 0.01, effective batch 8, microbatch 2, EMA decay 0.99, gradient clipping at 2.0, seed 0, BF16 autocast with FP32 weights, and all 1,000 diffusion training timesteps. Training visits shuffled `(window, horizon)` rows and reshuffles after exhausting them. Validation is measured every 100 updates; the selected checkpoint is always the final 1,000-step EMA, without test-based checkpoint selection.

The world model has 196,675,460 parameters, of which 195,485,216 are trainable in this run. The unused geometry-action and latent-action adapter layers remain frozen, as does the separate VAE. This is a short fine-tuning run over 480 training pairs, not an adapter-only update.

Original and adapted models are evaluated on the same six test windows and all 96 future target images, with seed 0, 250 DDPM steps, VAE posterior mean and the same 4:3 center crop followed by 224 × 224 resize. Each future frame is predicted directly from the same four observed frames and its anchor-relative action. Generated images are not fed back as context. This is action-conditioned direct prediction, not a closed-loop navigation or navigation-success evaluation.

The gallery protocol includes the shared observed anchor plus sixteen future images, encoded as 17 frames at 4 FPS. Playback therefore matches **1× simulation time**: the anchor-to-last-target source span is 4 seconds, and the last frame is held for 0.25 seconds, giving a 4.25-second media duration. The anchor is excluded from image-quality metrics.

## Data integrity and reproducibility

`tools/record_marslab.py` and `tools/marslab_isaac4.py` record synchronized physics-state provenance. `tools/prepare_marslab_manifest.py` freezes the split, image hashes, timestamps, actions and train-only scale. `tools/finetune_lusnar.py` supplies the existing adaptation implementation; its historical filename does not select lunar data. `tools/evaluate_planetary.py` performs prediction, `tools/score_planetary_predictions.py` scores saved images, and `tools/build_website_marslab_gallery.py` validates and exports the fixed gallery.

The exporter recomputes every action from raw body poses, verifies all source-image hashes, rejects cross-split image reuse, requires all six test windows and 96 prediction images, and checks training/data/inference checkpoint consistency. Public provenance retains sanitized source identities, hashes, physical timestamps, true metric actions and the verified render-warmup audit, without local filesystem paths.

The following SHA-256 identities describe the frozen inputs. The rover entry is a small USD wrapper referencing the base rover and sensors, not a hash of the entire asset bundle.

| Artifact | SHA-256 |
| --- | --- |
| Official source archive | `3bfe76e60aa131be4d76c895d29fb153369e16a0b008b5becfc664b440aa49e4` |
| `main_crater.usdz` | `522b44b8e53dfc5c734b5caa42e01771084358684a9d8a1fc285892017087774` |
| `assets/robots/perseverance/m2020_lidar.usda` | `ae84052a9666b48faf0f05abe70ecc6b165c77014247300e2f24553d66ca9533` |
| Referenced `m2020.usd` | `f97c651331c7ceab6061bf39d20643010306d6921da3b6508718c4d23518071f` |
| `configs/robots/perseverance.yaml` | `18e3f1e6c32d2f14b70970c46ed14814b9776f51203cdd9e9a236dc7d62076da` |
| `tools/record_marslab.py` | `c93e1033fd661cdfbbf6392d250eb466af9bc75f2d0c93ee0f03a3172b745ae0` |
| `tools/marslab_isaac4.py` | `51696f9d989a0bf8204981ba8f009c3e6e83ddce7b3064240bd683840b0a394b` |
| `tools/prepare_marslab_manifest.py` | `22195590335115e77b8f5bb28bc366127bb93e45463e6c19ca9b82d8a4023ac0` |
| Frozen episode plan | `908532046807fa866901a02cfa8509be6a6cab5cea0c834c0fdf20d63b1a0afd` |
| Prepared data manifest, 39 windows | `3da3f945dd1a9865f9e47998c92d5f5413bc523422c8e9f6a71ce8e614e6115d` |
| Evaluation input manifest, 6 windows | `02667602215840591ef1186abf1d649a71c380ecbf2d3dcb7592799b2a28cf49` |

Independent CPU verification completed with exit code 0. All 780 PNGs decoded as 224 × 224 RGB, matched their stored hashes, and had distinct byte hashes. All 39 windows had unique source images, with no path or image-hash overlap across splits. The evaluation input is exactly the six held-out samples from the full data manifest. Raw poses reproduced every metric and model-scale action and the training-only calibration above.

All ten trajectories matched the frozen episode specifications, actual scene/rover/config/recorder/adapter/helper source hashes and shared capture settings. Every trajectory completed 128 render-only warmup updates with unchanged physical state; every recorded camera timestamp exactly equaled its physics timestamp. The ten external exit files and the recording runner each report zero. Completion records verify all image and command counts, synchronization/cadence and artifact `fsync`; the independent audit also checked actual PNG and command-line counts. These are data-integrity checks, not evidence of model accuracy.

## Completed experiment results

The training report is complete at update 1,000, with 480 training pairs and 48 validation pairs. Training took 238.59 seconds on an NVIDIA L20, with 4.457 GiB peak allocated and 4.924 GiB peak reserved GPU memory. EMA validation diffusion loss fell from **0.13847846** at update 0 to **0.10209152** at update 1,000, a 26.28% reduction. Update 900 had a slightly lower validation loss of 0.10176422; the frozen selection rule still uses update 1,000.

The baseline inference-and-scoring pipeline and the adaptation/inference/scoring pipeline both exited with code **0**. Each inference generated exactly **96 future RGB images across all six test windows**, with no omitted or unscored samples. The inference timers report 109.91 seconds for the original model and 110.13 seconds for the adapted model; these exclude scoring. Independent CPU verification checked both actual checkpoint hashes, both completed inference manifests, every prediction's dimensions and decoding, and byte-identical observed contexts and ground-truth targets between runs. All original action/timing fields match the evaluation input.

The table reports the arithmetic mean of all 96 per-image measurements, excluding the observed anchor. PSNR uses RGB mean squared error in [0, 1]; SSIM averages RGB channels with an 11 × 11 Gaussian window, sigma 1.5, valid-window population covariance, K1 = 0.01 and K2 = 0.03; LPIPS uses pretrained AlexNet v0.1 on RGB in [−1, 1]. Scoring does not crop or resize the saved images. The copy baseline repeats the exact last observed context frame for every target; its per-frame scores are identical in both evaluation runs.

| Method | Target frames | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| Original OpenNWM Nav15 Stage 2 EMA | 96 | 17.84384 | 0.59332 | 0.42952 |
| MarsLab adaptation, final 1,000-update EMA | 96 | 24.36038 | 0.70134 | 0.14795 |
| Repeat last observed frame | 96 | 21.87293 | 0.66319 | 0.16654 |

The adapted model improves all three averages over both comparisons. Improvements over the original model are +6.51654 dB PSNR, +0.10803 SSIM and −0.28157 LPIPS. These aggregate results do not mean every window beats a static image: `test_02-0020` is worse than the copy baseline on all three metrics, and `test_00-0020` has worse LPIPS than copying. The complete window breakdown below prevents the three fixed gallery windows from hiding those cases. Each entry is **PSNR / SSIM / LPIPS**, averaged over that window's 16 future frames.

| Held-out window | Original model | Adapted model | Repeat last frame |
| --- | --- | --- | --- |
| `test_00-0000` | 18.4777 / 0.6264 / 0.3995 | 25.9899 / 0.7545 / 0.1171 | 23.7687 / 0.7105 / 0.1255 |
| `test_00-0020` | 18.3731 / 0.6147 / 0.3990 | 24.8124 / 0.7026 / 0.1399 | 23.2131 / 0.6890 / 0.1297 |
| `test_01-0000` | 16.3277 / 0.5212 / 0.4997 | 22.4828 / 0.6439 / 0.1802 | 18.8723 / 0.5971 / 0.2328 |
| `test_01-0020` | 17.8128 / 0.5837 / 0.3892 | 24.8860 / 0.7313 / 0.1176 | 19.6704 / 0.6242 / 0.1939 |
| `test_02-0000` | 18.5318 / 0.6294 / 0.4014 | 26.1108 / 0.7416 / 0.1061 | 23.7522 / 0.7144 / 0.1180 |
| `test_02-0020` | 17.5399 / 0.5845 / 0.4883 | 21.8804 / 0.6341 / 0.2268 | 21.9610 / 0.6439 / 0.1993 |

Inspection of the completed overview of all six windows shows that adaptation preserves the crater setting, while the original model introduces indoor or terrestrial-looking structures at longer horizons. The adapted predictions still blur ground textures and misplace or deform nearby rocks; plausible terrain appearance does not guarantee correct geometry or motion. The static baseline remains strong because much of the scene changes little over four seconds.

This small, single-seed adaptation uses one terrain and simplified lighting. It does not establish navigation performance on real Mars, cross-scene generalization, or reliable long-horizon terrain geometry. These image-quality metrics do not measure temporal consistency or closed-loop success. The experiment is separate from the paper's benchmark tables.

| Completed artifact | SHA-256 |
| --- | --- |
| Original Nav15 Stage 2 checkpoint | `aabff9b5a3ab62a9145073892653a81019782511272933b6acfb41e21b48b4f4` |
| Final 1,000-update checkpoint | `c5f8ef71736db3f74f024f6a1f3ca9ee477f6c9946513d6435b8717ddc914b30` |
| Training metrics | `332223d49f4e32e60377020ec71aa3bbb52c92b4ce5dfc505a0bef5ca993ac24` |
| Original-model inference manifest | `d361e0193949b222fc28dd321d425ab5716aa0a01d0c280e037be583d29bada6` |
| Adapted-model inference manifest | `749bdd23851716d91c34e5eb415f04368c977870b9c0a99fe27bd2535f7a3419` |
| Original-model image metrics | `c7c6d887950c83d67895b1d4374ac7515f0861535e167c4119df4baaf08fee57` |
| Adapted-model image metrics | `d4b0bb0bf196896eaa4ba1b2754cac89966a6f9c47e001c14ae942a524a5b6f3` |

## Website validation status

The final export completed with exit code 0: three fixed test examples, six MP4 videos, six WebP posters and three public provenance manifests. Videos contain 17 frames at 4 FPS (the observed anchor followed by 16 future frames), with 1× simulation timing and a final 0.25-second frame hold. The 12 existing non-Mars gallery examples were preserved.

Local browser validation passed all 12 combinations of three Mars cards, English/Chinese, and desktop/mobile widths (1440/390 px). Real video decoding, natural replay, offscreen pause/resume, comparison-slider pointer and keyboard interaction, and trajectory progress passed. Maximum measured prediction/ground-truth playback drift was 1.484 milliseconds. There were no JavaScript errors, Mars resource failures, or layout overflows. Public provenance, metric ground-truth paths, warmup metadata, cache version and method links also passed. Public deployment and post-deployment checks are recorded separately from this pre-publication document.

Engineering tests passed: the full exporter contract suite (9/9, including six real MP4 encodes and decodes), targeted v2 warmup checks (7/7), and the updated full synthetic-input contract (1/1). These tests verify software contracts; the model-quality results above come from the completed real experiment.

## Asset attribution

The fixed upstream [DATA.md](https://github.com/kimhoyun-robotair/MarsLab/blob/a406ca245bfbdf0faaff262a3c63d76110237e3d/docs/demo/DATA.md) attributes Main Crater to HiRISE DEM `DTEEC_025735_2185_025801_2185_A01.tif`, NASA / JPL-Caltech / University of Arizona. The rover visual model is credited to NASA/JPL-Caltech's Zareh Gorjian, with URDF conversion from JPL RSVP. Mars Rocks are by Ivan Vakulko (milos4), under CC BY 4.0; this rock-asset license is not presented as the license of the entire MarsLab project.
