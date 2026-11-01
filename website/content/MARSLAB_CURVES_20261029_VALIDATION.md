# MarsLab curved trajectories — 2026-10-29

火星展示改为左弯、右弯和轻度 S 弯，动作来自仿真器实际物理位姿。三个展示窗口在训练前固定，全部 6 个测试窗口、96 个未来目标均参与评估。保持与上一轮相同的 30 个训练片段和 1,000 次更新规模，从原始 Stage 2 权重重新微调；原实验保留。测试轨迹独立，但与训练共用同一地形。弯曲程度增加不代表画质必然更好。

## Curves, capture and frozen selection

The [MarsLab](https://github.com/kimhoyun-robotair/MarsLab) assets remain fixed at source commit [`a406ca245bfbdf0faaff262a3c63d76110237e3d`](https://github.com/kimhoyun-robotair/MarsLab/commit/a406ca245bfbdf0faaff262a3c63d76110237e3d). A minimal ROS-free Isaac Sim 4.1 recorder uses the `main_crater` terrain and physical Perseverance wheel drive, steering, contact and suspension. This is not the complete upstream Isaac Sim 5.x application. Physics uses the CPU at 60 Hz and 3.72 m/s² gravity; GPU rendering produces synchronized 224 × 224 RGB at 4 Hz, with 90° horizontal FOV and a 6.6° downward camera pitch. Fixed sun/dome intensities are 1000/250.

The prior demonstration commands were near straight (yaw rates ±0.02 or 0 rad/s); the new commands use ±0.12–0.16 rad/s at 0.8 m/s. Test left/right commands use ±0.14 rad/s. The S route changes yaw-rate sign at recording times 2.5 and 6.5 s. Steering and wheel acceleration limit the realized response. Commands are retained for audit; they are never substituted for measured actions. The actual four-second first-window turns are +23.809° left, −25.362° right, and +11.127° followed by −7.222° for the S route. The S curve is gentle and does not cross the initial heading axis; both body-yaw and XY path-curvature signs reverse.

Physics-only safety pilots checked sustained movement, attitude and actual curve direction. Unsafe/stalled starts were adjusted before freezing; all pilots, including accepted pilots, are excluded from training and evaluation. The final ten-episode plan and the three gallery first windows were frozen at **2026-10-29 10:40:07.914 UTC**, before any model fitting or predictions. All ten formal episodes completed with external exit code 0, without post-training replacement. Selection was not based on image scores or prediction appearance.

Each episode advances physics for 3 seconds of settling and 6 seconds of driving before capture. It then performs **128 render-only warmup updates with physical time, physics step, body position and quaternion frozen**. These render updates load assets and are excluded from the data. The two warmup phases have different semantics: only the render-only phase freezes physics. All recorded camera timestamps equal the physics timestamps. Completion flushes artifacts with `fsync` and uses the existing `process-exit` shutdown; `native_cleanup_completed=false` is retained because native plugin unloading was not attempted.

| Split | Independent trajectories | Source RGB frames | 20-frame windows | Future targets |
| --- | ---: | ---: | ---: | ---: |
| Training | 6 × 25 s | 600 | 30 | 480 |
| Validation | 1 × 15 s | 60 | 3 | 48 |
| Same-terrain test | 3 × 10 s | 120 | 6 | 96 |

Every nonoverlapping window contains four observed frames and sixteen future frames. The fourth observed frame anchors the action. Gallery IDs are `marslab-test_00-0000`, `marslab-test_01-0000`, and `marslab-test_02-0000`; the second window of every test trajectory is also evaluated. No path or identical image bytes are shared across splits. The terrain itself is shared, so spatial/visual overlap is possible and this does not establish cross-terrain generalization.

Metric actions rotate target-minus-anchor world XY into the anchor's forward/left axes, with wrapped target-minus-anchor yaw in radians. The model multiplies XY by one training-only scale: **1.52680058131512 units/m**, from `0.25 / 0.1637410956345464 m` (the median adjacent planar step across training trajectories). Validation and test reuse that scale. The public path uses metric `actions_meters_radians`, never commands or display-normalized coordinates. The SE(2) interface discards height, pitch and roll.

Actual four-second motion, recomputed from raw PhysX state (positive yaw means left):

| Test window | Net yaw | Absolute yaw sum | Positive / negative yaw sum | XY path length (m) | Left-displacement range (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `test_00-0000` | +23.809° | 23.809° | +23.809° / +0.000° | 2.651 | 0.568 |
| `test_00-0020` | +24.879° | 24.879° | +24.879° / +0.000° | 2.783 | 0.629 |
| `test_01-0000` | -25.362° | 25.362° | +0.000° / -25.362° | 2.832 | 0.629 |
| `test_01-0020` | -25.103° | 25.103° | +0.000° / -25.103° | 2.809 | 0.622 |
| `test_02-0000` | +3.905° | 18.350° | +11.127° / -7.222° | 2.240 | 0.279 |
| `test_02-0020` | +7.679° | 17.702° | +12.691° / -5.012° | 2.153 | 0.062 |

## Fixed adaptation and evaluation

The run restarts from the original OpenNWM Nav15 Stage 2 EMA checkpoint (`aabff9b5a3ab62a9145073892653a81019782511272933b6acfb41e21b48b4f4`), not the earlier Mars adaptation. The recipe is unchanged: 1,000 AdamW updates, learning rate 1e−5, betas (0.9, 0.999), weight decay 0.01, effective batch 8, microbatch 2, EMA decay 0.99, gradient clipping 2, seed 0, BF16 autocast with FP32 weights and 1,000 diffusion training timesteps. Training shuffles all `(window, horizon)` rows and repeats after each epoch. The VAE stays frozen; its posterior-mean cache contains only the 660 training/validation source images. Only training pairs update weights. The world model has 196,675,460 parameters, with 195,485,216 trainable; unused geometry/latent action adapters remain frozen. Validation is measured every 100 updates; the checkpoint is always the final 1,000-update EMA.

Training completed in 242.71 s on NVIDIA L20, with 4.457 GiB peak allocated memory. EMA validation loss changed from **0.11672375** to **0.08654376**. Both original-model and adaptation/evaluation/scoring pipelines exited 0. Each model generated exactly 96 held-out images, using seed 0, 250 DDPM steps, VAE posterior means, four context images and the same 4:3 center crop followed by 224 × 224 resize.

Each future frame is predicted directly from the same four observed images and its anchor-relative actual action. Generated images are not fed back as context, and future reference images are not model inputs. This is action-conditioned direct prediction, not closed-loop navigation. The displayed trajectory is the known conditioning motion, not a model-estimated navigation path.

## All held-out results

All 96 future targets are scored; the observed anchor is excluded. Metrics are arithmetic means of per-image values. PSNR uses RGB MSE in [0,1]; SSIM uses the RGB mean, an 11 × 11 Gaussian window with sigma 1.5, valid-window population covariance and K1/K2 0.01/0.03; LPIPS uses pretrained AlexNet v0.1 on RGB [−1,1]. Saved images receive no additional crop or resize for scoring. The copy baseline repeats the exact fourth context image, and its per-frame results are byte-for-byte identical between both scoring reports.

| Method | Targets | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| Original OpenNWM Nav15 Stage 2 EMA | 96 | 17.28316 | 0.57186 | 0.48725 |
| Curved MarsLab adaptation, final 1,000-update EMA | 96 | 24.50011 | 0.68838 | 0.16523 |
| Repeat last observed frame | 96 | 22.06287 | 0.63860 | 0.20396 |

Per-window means below are **PSNR / SSIM / LPIPS**, each over sixteen targets. These are new measurements for the curved-route experiment; the earlier nearly straight-route scores are not reused.

| Held-out window | Original model | Adapted model | Repeat last frame |
| --- | --- | --- | --- |
| `test_00-0000` | 16.6129 / 0.5471 / 0.4584 | 23.7546 / 0.6785 / 0.1694 | 21.3473 / 0.6220 / 0.2227 |
| `test_00-0020` | 15.7550 / 0.4940 / 0.5318 | 22.0986 / 0.6096 / 0.2345 | 19.5621 / 0.5610 / 0.3001 |
| `test_01-0000` | 16.5536 / 0.5581 / 0.5500 | 25.4887 / 0.7047 / 0.1573 | 22.3356 / 0.6326 / 0.1962 |
| `test_01-0020` | 17.6109 / 0.5825 / 0.5336 | 25.6285 / 0.7109 / 0.1658 | 23.3528 / 0.6527 / 0.2059 |
| `test_02-0000` | 18.3006 / 0.6162 / 0.4326 | 24.7376 / 0.7073 / 0.1435 | 22.3643 / 0.6797 / 0.1594 |
| `test_02-0020` | 18.8661 / 0.6332 / 0.4171 | 25.2927 / 0.7193 / 0.1208 | 23.4152 / 0.6836 / 0.1394 |

This is a small, single-seed adaptation on one terrain with simplified lighting. It does not establish performance on real Mars, cross-scene generalization, reliable long-horizon geometry, temporal consistency or closed-loop success. The source textures are low resolution; generated texture, rock geometry and movement can still be inaccurate. Metrics do not guarantee that every predicted object follows the true motion. Results are separate from the paper's benchmark tables.

## Integrity and media

The complete capture audit decoded all 780 source PNGs, verified counts, hashes, cadence, commands and synchronization, and reported no errors or warnings. A separate CPU audit recomputed all source hashes and all 39 windows' actions from raw poses: maximum metric-action error was 1.33e−15 and model-action error 2.00e−15. Cross-split source path/inode/hash intersections were empty. All ten recordings passed frozen-state render-warmup checks. Both inference outputs have verified checkpoint hashes, all 192 prediction images decode as 224 × 224 RGB, and the original/adapted contexts and ground-truth targets match byte-for-byte. All six windows' action/timing fields match the evaluation input. Aggregate metrics were independently recomputed from the saved per-frame values.

The exporter revalidates raw poses, all source images, training/evaluation provenance and the fixed gallery selection. It exports six MP4s, six WebP posters and three public source manifests under `assets/space/marslab-curves-20261029/`. Each video is 17 frames at 4 FPS: the common observed anchor plus sixteen future targets. Playback is **1× simulation time**, covering four seconds of source motion with the final frame held another 0.25 s (4.25 s media duration). No frame interpolation is used. The three path overlays each contain exactly the same 17 synchronized timestamps and metric coordinates. All new media are served locally by the website, with a versioned asset path. Browser and deployment checks are recorded separately; this document does not claim an unperformed public deployment.

| Artifact | SHA-256 |
| --- | --- |
| Frozen episode plan | `94794061c2b2eba7e87444ac471af4a8ef7ed7e230fbeeacc72901e4edda1a7e` |
| Prepared 39-window data manifest | `28bc7c4e44594386ca13b379798319215472d5a1775ddef10aaf59484abe0f69` |
| Six-window evaluation input | `cf4b8670612c205772eb7aa7333c69c4869c0c95f1ef7f79a2216f5b88a79f49` |
| Final 1,000-update checkpoint | `194dcab1cbf2e0b1bb6c1c2ddde7f8b4ee7cc3c2ddab4261f19138977fdb1289` |
| Training report | `82376a87e789300dad72ca70ebe674a675282923b72990758e6e7fcd3af7658f` |
| Original-model inference manifest | `99448ce056d53bbff861b26df5a0b75944aba252f5ecd253fcf8ed2bd9781bf7` |
| Adapted-model inference manifest | `00e1e74837e4b3c433681ef79a18c38bab13b99522e3036ffad0f1056abd6b37` |
| Original-model image metrics | `32c4da05c959a22f9fc07742b53b29747ac104035dc54759faea7c6973f7575f` |
| Adapted-model image metrics | `15f753cd551df77d7c362d919ba7b49fb5f51070a1bf8d1cceb21366dbd59f64` |

## Attribution

The fixed [upstream DATA.md](https://github.com/kimhoyun-robotair/MarsLab/blob/a406ca245bfbdf0faaff262a3c63d76110237e3d/docs/demo/DATA.md) credits Main Crater to HiRISE DEM `DTEEC_025735_2185_025801_2185_A01.tif`, NASA / JPL-Caltech / University of Arizona. The rover visual model is credited to NASA/JPL-Caltech's Zareh Gorjian, with URDF conversion from JPL RSVP. Mars Rocks are by Ivan Vakulko (milos4), under CC BY 4.0; this license does not describe the entire MarsLab project. The earlier [MarsLab experiment record](MARSLAB_20261029_VALIDATION.md) preserves its separate dataset and results.
