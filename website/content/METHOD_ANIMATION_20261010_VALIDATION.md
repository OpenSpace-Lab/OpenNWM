# Method animation — 2026-10-10

## Requested change and reference

The user requested an animated method figure in the style of https://knightdby.github.io/awam-demo/, allowing either the architecture or multi-stage training figure. This supersedes the earlier choice of a static architecture image while retaining a single concise method presentation.

The live reference uses native HTML/CSS sequential highlights and traveling dots, not a GIF or video. Its observed six-second loop informs the motion style. OpenNWM uses its own unchanged paper training image and training semantics, with stage selection and pause controls. The source image is `assets/paper/figure-3-training.png`, 2124×576; the original paper, image, and crop provenance remain intact.

Reference inspection evidence is in `/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/prediction_20261010/reference-method/REFERENCE_METHOD.md`, alongside screenshots, captured HTML, and browser measurements. The reference's mobile clipping and incomplete reduced-motion behavior are not copied.

## Prior revision deployment

The opening film and merged lunar Prediction gallery were published as `7d9582eb32f04b48677056674603cf28a2025880`, [Pages workflow](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/38052503067).

The initial public verification passed 25 files but exceeded its full-hero transfer budget. A read-only follow-up verified the entire hero through 15 HTTP range responses, with 14,819,671 bytes and SHA256 `a9d40bfbe3832fd55d882d2efa128fdf7fb3f1f7b8a5c176a001cf4132be14f7`, matching the deployed commit and checkout. It completed in 183.832 seconds, exit 0.

Working directory: `/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/prediction_20261010/public-hero/`.

```bash
/file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run --no-capture-output -n nwm python -u verify_hero.py > hash.log 2>&1
```

PID 432636. The bytes were held only in memory for verification. `hash-report.json`, `hash.log`, and `deployment-hashes-combined.json` preserve the follow-up and combined **26/26 verified files** without overwriting the earlier timeout record.

## Animation and final verification

The page presents one unchanged Figure 3 with a 10.8-second three-stage loop, moving markers on its arrows, and outlines around the blocks trained at each stage. Selecting a stage pauses the sequence. The world model remains unhighlighted during action-encoder warmup, matching the paper's frozen-model indicator. Desktop shows all three stages; narrow screens show one complete stage at a time. An expand button opens the original image.

Reduced motion initially pauses playback. Explicit playback enables this figure only; offscreen/hidden playback is suspended. Language updates preserve the current stage and playback state.

Local integrated browser regression passed **47/47 checks**, PID 472353, exit 0, at 13:10 UTC. This covered the single original figure, concise bilingual copy, EN/ZH layouts at 1440/768/390/320 px, the fixed wall/Moon/Mars film and rover, all 12 Prediction cards, real paired Earth/lunar playback and sliders, eight navigation examples, and both comparison charts. No JavaScript or HTTP errors were observed. Only a mocked GET health check was requested; no inference was invoked. Desktop and mobile method screenshots were inspected.

Working directory: `/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/method_animation_20261010/`.

```bash
/file_system/vepfs/algorithm/dujun.nie/.npm-global/bin/node /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/method_animation_20261010/integrated.cjs > /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/method_animation_20261010/integrated-local.log 2>&1
```

The log ends `FINISH {"passed":true,"checks":47,"failed":[]}`. `local-report.json` and `local-*.png` preserve measurements and screenshots.

Focused animation checks passed **54/54**, including 24 stage/language/viewport combinations, PID 489431, exit 0. This checked pause/resume elapsed time, offscreen and document-hidden suspension, keyboard stage selection, language changes preserving DOM/focus/state, reduced-motion default and explicit playback, and live preference changes. The live preference-change phase used the real browser clock. Screenshots of all three mobile stages and desktop warmup were inspected: original labels, fire/snowflake indicators, and column boundaries remained complete; overlays aligned with arrows and the correct trainable blocks.

Focused command, from `/file_system/vepfs/algorithm/dujun.nie/code/OpenNWM`:

```bash
conda run --no-capture-output -n base node /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/method_animation_20261010/focused.cjs > /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/method_animation_20261010/focused.log 2>&1
```

The log ends `no script errors {"passed":true}` and `COMPLETE exit=0`. Earlier focused runs (PIDs 474510, 480987, 483743) timed out at emulated preference changes with the virtual clock. The harness-editing command initially failed to apply because its Python heredoc input was captured. `focused-verification-notes.md` records those failures and their evidence limits; `virtual-clock-focused.*` preserves the latest failure. No website source change was required. The successful run uses real time for preference events.

Public commit `dfae157c0a209362ac5eaf2c134f2270400b248b` was pushed and successfully deployed by [Pages run 38055039324](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/38055039324). The API confirmed the exact head SHA and successful conclusion. Public browser verification passed **74/74 checks**, exit **0**, at 13:19:55 UTC.

Public file verification command, from the artifact directory above (PID 501793):

```bash
OPENNWM_DEPLOY_SHA=dfae157c0a209362ac5eaf2c134f2270400b248b OPENNWM_DEPLOY_RUN=38055039324 /file_system/vepfs/algorithm/dujun.nie/miniconda3/bin/conda run --no-capture-output -n nwm python -u verify_public_hashes.py > public-hashes.log 2>&1
```

The verification exited **0**, with **8/8 public files** matching the deployment checkout on their first request: all seven changed code/document files and the unchanged original training image. `public-hashes.json` and `public-hashes.log` record exact bytes, SHA256 values, and workflow identity. The log ends `passed=True exit=0`. The original PNG retains SHA256 `06f34ed26c6a7f39238449660cf767de54e6fd16e922425abe2053f2cd4f81eb`.

Public browser command, from its `public-method/` subdirectory (PID 502290):

```bash
OPENNWM_METHOD_DEPLOY_SHA=dfae157c0a209362ac5eaf2c134f2270400b248b /file_system/vepfs/algorithm/dujun.nie/.npm-global/bin/node verify.cjs > verify.log 2>&1
```

This focused public browser check blocks unrelated video/audio and images, allowing the original training image and all actual website HTML/JS/CSS/JSON/fonts to load from the public host. It does not retest the unchanged hero or Prediction media; those are covered by the prior public verification and current integrated local regression.

The public check confirmed the real-time 0→1→2→0 sequence, pause stability, manual stage selection, 1440/390/320 px layouts in both languages, the full-resolution original link, default static reduced-motion behavior and explicit playback. No JavaScript page errors occurred. Public desktop joint fine-tuning and mobile action-warmup screenshots were visually reviewed. Evidence: `public-method/report.json`, `public-method/verify.log`, and its stage screenshots. The log ends `COMPLETE passed=true checks=74/74 exit=0`. The public checkout is clean at the deployed commit.

`public-method/BROWSER_VALIDATION.md` records 19 screenshots, the expected static-site `/api/health` 404, and 171 intentionally blocked unrelated media/image requests. There were no unexpected transport failures or unresolved method-animation issues. Observed stage transitions occurred at approximately 7, 3709, 7303, and 10879 ms.
