# Planetary prediction examples

These qualitative examples use the OpenNWM Nav15 Stage 2 checkpoint, adapted separately to LuSNAR and MarsLab. They are separate from the paper benchmark tables and do not establish closed-loop navigation success.

Each example predicts sixteen future images directly from four fixed observed images and known anchor-relative actions. Generated images are not fed back as context. DDPM uses 250 steps and seed 0. Videos show the observed anchor followed by sixteen predictions at 4 FPS: 17 frames, 4.25 seconds.

## LuSNAR

The model was fine-tuned for 1,000 updates on 30 windows from Moon_1–5; Moon_6 was used for validation. Moon_7–9 were held out. AdamW used learning rate 1e-5, effective batch size 8 and EMA 0.99. The translation scale, fitted on training data only, is 0.4863296751176136 m; yaw remains in radians.

The three displayed windows were selected by recorded turning motion before inference, without consulting prediction quality. Each has twenty distinct source frames on the nominal 4 Hz grid, within 125 ms of its grid timestamp, and at least one meter of future travel. Original evaluation-window exclusions and their four-second buffers were retained.

| Scene | Grid anchor | Recorded yaw range | Actual future span |
| --- | ---: | ---: | ---: |
| Moon_7 | 180 | 26.95° | 3.9835 s |
| Moon_8 | 39 | 37.24° | 3.9668 s |
| Moon_9 | 713 | 52.12° | 4.0954 s |

The path overlays use recorded metric poses. Later predictions can drift in rock and crater placement; these turning examples do not establish improved accuracy. Source identities, timestamps, actions and hashes are retained in the [Moon 7](../assets/space/lusnar-curved-20261029/moon-7/manifest.json), [Moon 8](../assets/space/lusnar-curved-20261029/moon-8/manifest.json), [Moon 9](../assets/space/lusnar-curved-20261029/moon-9/manifest.json) and [trajectory audit](../assets/space/lusnar-curved-20261029/trajectory-provenance.json) manifests.

## MarsLab

The simulator uses the MarsLab `main_crater` terrain and Perseverance assets at commit `a406ca245bfbdf0faaff262a3c63d76110237e3d`, with an Isaac Sim 4.1 recorder and simplified lighting. RGB and physical body poses are synchronized at 4 Hz. Actions come from measured planar displacement and yaw, not commanded controls. The model's training-only scale is 1.52680058131512 units/m.

| Split | Trajectories | Twenty-frame windows | Future targets |
| --- | ---: | ---: | ---: |
| Training | 6 | 30 | 480 |
| Validation | 1 | 3 | 48 |
| Test | 3 | 6 | 96 |

The same terrain is shared across splits. The first test window of each trajectory was fixed before training for the left, right and S-curve gallery. All six test windows were evaluated. Adaptation restarted from the original Stage 2 checkpoint for 1,000 AdamW updates with learning rate 1e-5, effective batch size 8, EMA 0.99 and seed 0. The final EMA checkpoint was used without test-based selection.

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | ---: | ---: | ---: |
| Original OpenNWM | 17.28316 | 0.57186 | 0.48725 |
| Adapted OpenNWM | 24.50011 | 0.68838 | 0.16523 |
| Repeat last observed frame | 22.06287 | 0.63860 | 0.20396 |

Metrics are means over all 96 target images at 224 × 224 resolution. SSIM uses an 11 × 11 Gaussian window with sigma 1.5; LPIPS uses AlexNet v0.1. This small, single-seed, same-terrain experiment does not establish cross-terrain or real-Mars generalization.

The [left-turn](../assets/space/marslab-curves-20261029/marslab-test_00-0000/manifest.json), [right-turn](../assets/space/marslab-curves-20261029/marslab-test_01-0000/manifest.json) and [S-curve](../assets/space/marslab-curves-20261029/marslab-test_02-0000/manifest.json) manifests retain sample identities, timing, measured actions, checkpoint hashes and source provenance. The adapted checkpoint SHA-256 is `194dcab1cbf2e0b1bb6c1c2ddde7f8b4ee7cc3c2ddab4261f19138977fdb1289`.

## Attribution

[MarsLab asset attribution](https://github.com/kimhoyun-robotair/MarsLab/blob/a406ca245bfbdf0faaff262a3c63d76110237e3d/docs/demo/DATA.md) credits Main Crater to HiRISE DEM `DTEEC_025735_2185_025801_2185_A01.tif`, NASA / JPL-Caltech / University of Arizona. The rover model is credited to NASA/JPL-Caltech's Zareh Gorjian, with URDF conversion from JPL RSVP. Mars Rocks are by Ivan Vakulko (milos4), under CC BY 4.0; that license applies to those rock assets, not the entire MarsLab project.

Dataset and asset licenses remain with their respective authors. Per-sample source manifests retain the original provenance and are the reference for reproducing displayed media.
