# Prediction and opening-film revision — 2026-10-10

## Scope

- The opening film starts with the complete 225-tile video wall, with a fixed camera for exactly 3 seconds. Only Moon and Mars footage follows. The moving rover again indicates playback progress, with chapter and range seeking retained.
- The method section contains the first original paper architecture image and one concise explanation. The original image is unchanged (SHA256 `06feb9ee8a10c096120d1d51ddab38d1561831df089ea2887754b912c8036392`).
- Prediction replaces In motion. Its 9 unchanged Earth examples are followed by 3 lightly fine-tuned Moon examples, all with GT/OpenNWM sliders and Given path trajectories. Lunar cards use gray accents, a moon icon, and a brief adaptation label. The standalone Beyond Earth section is removed.
- The 8 navigation examples, both baseline comparison charts, Demo, and final dataset section remain. Navigation uses the semantically distinct Reference path label because the recorded path is not an input to its planner.

No training, inference, or GPU work was performed. Existing experiments and preview servers were left running.

## Opening-film evidence

Evidence directory: `/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/hero_refinement_20261010/`.

From `/file_system/vepfs/algorithm/dujun.nie/code/OpenNWM`:

```bash
OPENNWM_MEDIA_WORK_DIR=/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/hero_refinement_20261010 conda run --no-capture-output -n base python -u tools/build_website_cinematic.py render > /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/hero_refinement_20261010/render.log 2>&1
```

Render PID 237180 exited 0; validation PID 248749 exited 0. `validation.json` contains 17 passing checks. The final film is 1920×1080 H.264, 24 FPS, 606 frames, 25.25 seconds, and 14,819,671 bytes. Chapters: wall 0–3.00 s, Moon 3.00–13.44 s, Mars 13.44–25.25 s. Cache key: `fixed-wall-moon-mars-v5-c8e1deeae799`.

The first and last wall frames retain the complete composition; encoding PSNR against the unscaled source wall is 29.20 and 27.83 dB. Between 0 and 2 seconds, 222 of 225 tiles visibly change. At 3 seconds, the first lunar frame matches its source at 35.34 dB with no residual Earth transition. These PSNR values measure media encoding fidelity, not model accuracy. Timeline and full-wall review images were visually checked.

## Lunar gallery evidence

Evidence directory: `/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/prediction_20261010/`.

`tools/build_website_lunar_gallery.py` exports existing completed direct-inference Moon 7–9 results. It checks the original inference manifest, checkpoint, saved image hashes, pose hashes, test-scene exclusion, timestamps, and physical actions. Every paired stream contains the shared final context frame plus all 16 target frames: 17 frames, 224×224, 4 FPS, 4.25 seconds. Original nonuniform capture times are retained separately from playback times. The 9 Earth manifest objects are identical to the previous publication.

`media-report.json` verifies all 6 exported videos and source-frame concatenation hashes. `browser-report.json` records real paired playback, drift below 0.12 s, synchronized seeking at 0/1.1/4.05 s, trajectory positions, keyboard slider endpoints, and English/Chinese layouts at 1440/768/390/320 px. Browser PID 258283 exited 0; `browser.log` contains its output. No JavaScript errors were reported. Existing playback synchronization was retained.

## Integrated checks and publication

The integration harness is `prediction_20261010/integrated.cjs`, run from the source repository with the absolute Node runtime:

```bash
/file_system/vepfs/algorithm/dujun.nie/.npm-global/bin/node /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/prediction_20261010/integrated.cjs > /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/prediction_20261010/integrated-local.log 2>&1
```

An initial harness run (PID 262189) passed 20 structure/layout checks, then timed out because it awaited a decoded hero frame before requesting playback with reduced motion enabled. That incomplete result is retained as `local-report-before-harness-fix.json`. The harness now requests playback before waiting and explicitly awaits lazy image decoding before screenshots. No website code change was needed for that test correction.

A second harness run (PID 271315) likewise passed those 20 checks, but a direct `video.play()` was paused by the expected reduced-motion preference handler. It is retained as `local-report-before-ui-play-fix.json`. The final harness uses the actual page play button, matching user interaction.

The final local integrated run (PID 278483) exited 0: **47 checks passed**, including original method-image loading, section/copy changes, all 8 bilingual layouts, five hero seek positions, chapter seeking, visible moving rover, full-wall fitting on mobile, real hero playback, pointer/keyboard sliders and actual paired playback for an Earth example plus all three Moon examples, and both benchmark controls. No JavaScript or HTTP errors occurred. Only the read-only health API was requested and mocked offline; no inference endpoint was called. Screenshots of the method, lunar cards, and desktop/mobile hero were visually reviewed.

JavaScript syntax checks passed for app, research, cinematic, and motion-gallery modules. The original paper image hash is identical to the previous published copy.

The revision was published as `7d9582eb32f04b48677056674603cf28a2025880`; [Pages run 38052503067](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/38052503067) completed successfully. The first public hash run confirmed 25 of 26 files, including all six lunar videos, but the 14.8 MB hero transfer exceeded the 120-second per-attempt budget. The public integrated run passed its first 10 checks, then timed out waiting 30 seconds for the paper image. These incomplete runs are retained; they are not reported as full passes. Follow-up hero verification and the user's subsequent method-animation revision continue in `METHOD_ANIMATION_20261010_VALIDATION.md`.

The bounded follow-up completed successfully: PID 432636 verified the full hero SHA256 via 15 range responses, bringing the published-file total to **26/26**. Browser PID 449080 exited 0 after verifying actual decoded playback and the 3-second Moon/15-second Mars seek states with the corresponding rover positions. The public test connection buffered slowly, so this establishes playback and correct seeking rather than smooth uninterrupted playback. Evidence and the unlocated console 404 are recorded in `prediction_20261010/public-hero/VERIFICATION.md` and `public-hero/browser/BROWSER_VALIDATION.md`; no JavaScript page error or media error occurred.
