# Website revision — 2026-10-09

Final local validation passed: motion and curved-navigation browser checks, visual review, final-rollout loss audit, exported-media verification, and actual media-stream interruption tests. The first public revision exposed a motion-video synchronization failure; the final fix waits for both streams and preserves their shared playback time through buffering. Final commit `d82e5fe7f573a1df24835879fc2bb3eb41c0d059` deployed successfully. Public acceptance coverage is complete through the main browser run's successful motion/navigation/layout checks and a separate passing playback-lifecycle check. The main run still records `passed: false` because its final reload check timed out during slow public media loading; it is not relabelled as a whole-run pass.

## Completed motion-gallery revision

- Nine cards show a recorded conditioning path and a synchronized Ground truth / OpenNWM wipe comparison: **18 displayed videos**, with no NWM baseline in the gallery.
- Each slider supports native pointer dragging and Home, End, and arrow keys. Its value changes the rendered GT clip-path; source videos retain their original proportions.
- Every SVG contains the recorded trajectory points from `conditioningTrajectory` in `content/demos.json`. The browser check confirms point counts, finite coordinates, provenance fields, and equal-scale geometry against the manifest.
- Cards support individual and global playback controls, reduced-motion defaults, offscreen pause, and resuming visible cards unless explicitly paused.

## Motion-only browser acceptance — PASS

Run from:

```text
/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/revision_20261009
```

Exact command:

```bash
SKIP_NAVIGATION=1 conda run --no-capture-output -n base node check.cjs
```

- Exit status: **0**, completed on 2026-10-09 at 03:18:27 UTC.
- Target: `http://127.0.0.1:8042/`.
- Nine cards / 18 videos passed keyboard and mouse crop checks, SVG/source geometry checks, playback, synchronized timing, and pause checks.
- Maximum observed playback drift: **0.008028 seconds**, sampled six times per card; acceptance limit 0.3 seconds.
- English and Chinese layouts passed at **320, 390, 768, and 1440 px** without document or card overflow.
- Offscreen pause/resume, global pause persistence, and per-card pause persistence passed. No page JavaScript errors, HTTP errors, or failed resource requests were recorded.
- The optional API health endpoint was mocked offline. This browser run did **not** call or validate GPU inference. Navigation was explicitly skipped while the new examples were being generated.
- Four screenshots were saved. Desktop and phone screenshots were inspected; a subsequent cosmetic fix keeps the wipe-handle arrows on one line. The final integrated run below includes that fix.

Evidence directory:

```text
/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/revision_20261009/2026-10-09T03-18-09-316Z-127.0.0.1/
```

It contains `check.log`, `report.json`, and English/Chinese screenshots at 390 and 1440 px. The reusable script is `revision_20261009/check.cjs`, uses the existing NAS `playwright-core` runtime and `/usr/bin/google-chrome` with software rendering, and waits for DOM content and relevant locators rather than network idle. `BASE_URL` selects the preview or public `/OpenNWM/` URL; Chrome uses `--no-proxy-server` for direct public access.

## Final local browser acceptance — PASS

- Exit status **0**, completed on 2026-10-09 at **04:17:36 UTC**, against `http://127.0.0.1:8042/`, after the final buffering-barrier fix.
- All **9 motion cards / 18 videos** passed native mouse/keyboard clipping, recorded-path geometry, playback synchronization, and pause checks. Maximum observed drift was **0.008309 seconds**.
- All **3 new curved-navigation cards** passed media playback, equal-scale SVG geometry, and video-time cursor alignment. Each planned path turns by at least **50°** and satisfies the selection gate.
- **16 layout checks** passed: both gallery sections, both languages, and widths of **320, 390, 768, and 1440 px**, without overflow.
- Offscreen pause/resume and both global and individual pause persistence passed. There were **no JavaScript, HTTP, or failed-request errors**. Desktop motion and navigation screenshots from the initial full local run were visually inspected.
- The inference-service health response remained mocked offline; this test does not exercise the optional live GPU service.

Evidence (`report.json`, `check.log`, and screenshots):

```text
/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/revision_20261009/2026-10-09T04-17-18-493Z-127.0.0.1/
```

The initial full local run at `2026-10-09T03-50-34-100Z-127.0.0.1/` also passed, with maximum drift **0.015346 seconds**. It preceded the public-network failure and does not replace the final run above.

## Actual media-stream interruption — PASS

- The prediction stream and ground-truth stream were separately delayed/interrupted for **7 seconds**. Both videos paused together and resumed together; measured pair-pause durations were **5.288 seconds** and **5.033 seconds**, respectively.
- Maximum observed drift was **0.000328 seconds** with prediction delayed and **0.000341 seconds** with ground truth delayed.
- Buffering produced **zero seek events** and **no timeline regression**. The final barrier preserves the shared playback time, waits for both videos before playback, and performs at most one seek per barrier.
- Manual pause, global pause, and offscreen cancellation passed. Cancelling a pending start left both videos paused at time zero.
- Seek and loop checks passed. The loop sample had prediction at **0.065946 seconds** and ground truth at **0.065962 seconds**, both playing.
- The report records `passed: true`. This controlled browser test uses actual media response interruption; final public-URL verification remains separate.

Evidence:

```text
/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/revision_20261009/motion-network-report.json
```

## New curved navigation — selection and loss audit completed

The new navigation pipeline uses independent goal-image CEM optimization. Inputs are the start RGB image and goal RGB image. It optimizes one speed and three yaw-rate segments, integrates body-frame actions through SE(2), and scores the final predicted image with LPIPS. Recorded poses are used only for choosing turning scenes and measuring errors afterwards, not as optimizer targets or controls. This is a qualitative offline planning demonstration, separate from the paper's benchmark protocol.

The `recon-078` pilot completed with exit status **0** (reported by the supervising agent). Its log records **150.2058 seconds** for the scene and these outputs:

- Total planned turn: **50.6618°**, with three qualifying turning segments.
- Endpoint error: **1.3269 m**; normalized endpoint error: **0.2978**.
- These are per-example diagnostics, not aggregate performance or a closed-loop success result. The pilot was subsequently included in the final three-example selection after visual review.

Working directory: `/file_system/vepfs/algorithm/dujun.nie/code/OpenNWM`. The successful retry log's first line reports Python PID `2867028` from `os.getpid()` and GPU selection `CUDA_VISIBLE_DEVICES=0`; no wrapper PID is inferred here. The earlier `pilot-recon078.log` reports a different Python PID (`2846515`) and a proxy failure, so it is not the successful pilot evidence.

The recorded command arguments correspond to:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n nwm python tools/build_website_curved_navigation.py run --ids recon-078 --population 32 --elites 6 --iterations 2 --steps 50 --batch-size 32
```

Successful pilot log:

```text
/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/curved_navigation_20261009/pilot-recon078-localvae.log
```

Final selection: **RECON 088, 046, and 078**, in that display order, exported as `curved-recon-088`, `curved-recon-046`, and `curved-recon-078`. All **17 frames per example** passed manual visual review. Both inference queues completed with exit status **0** and released their GPUs. The export gate requires at least two turn segments of 6° each, total turn ≥25°, normalized endpoint error ≤0.35, and full-frame visual review. Exported-media checks, navigation playback, and source-to-SVG verification have passed.

The final-rollout loss audit completed with exit status **0**. Search loss and the recomputed loss of each rendered batch-1 rollout are recorded separately; the audit did not change the planned paths.

| Selected example | Final rollout LPIPS goal loss | Final minus search loss |
| --- | ---: | ---: |
| `recon-088` | 0.3827298284 | +0.0358829796 |
| `recon-046` | 0.4219297171 | −0.0035668612 |
| `recon-078` | 0.2991186678 | +0.0099093616 |

Audit log:

```text
/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/curved_navigation_20261009/final-loss-audit.log
```

These losses describe the selected qualitative demonstration runs; they do not replace the paper's benchmark values.

## Navigation export and provenance — PASS

The independent export audit completed with exit status **0**:

- **81 source/media hashes** across the three examples match. The approximately **3.1 GB checkpoint** and inference/export-script hashes also match the recorded provenance.
- Saved float32 predictions reproduce the **17 source PNGs per example pixel for pixel**. Encoded videos are **224 × 224**, **17 frames**, **4 FPS**, and **4.25 seconds**; minimum framewise encoding PSNR is **36.50 dB**.
- Planned trajectories independently reconstruct from the recorded SE(2) controls. Reference paths independently reconstruct from the original `traj_data.pkl`; ATE and endpoint errors agree with the manifest.
- The first two selected paths each change turning direction once; the third is a sustained bend. The displayed paths are actual integrated controls, without substituting or smoothing reference trajectories.

Manifest SHA256:

```text
5b8a79ddfae489fb60553206c0694e58243a5c2ebe893311d528780732211909
```

Evidence:

```text
/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/curved_navigation_20261009/export_validation.json
/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/curved_navigation_20261009/export-validation.log
```

## Source and planning checks

Reported by the supervising agent and the independent planning audit:

- All **189** final JSON asset references resolve; the earlier pre-export snapshot had 201 references.
- Six JavaScript modules pass syntax checks; `git diff --check` passes. Runtime HTML/JavaScript/CSS/JSON contain no `/file_system`, `/home`, or `/tmp` paths.
- The public-checkout synchronization dry run lists only additions and updates, with no deletions. This does not establish that deployment has occurred.
- An independent CPU coordinate audit covered three straight/curved paths, four dataset waypoint spacings, and 16 steps. Reconstructing actions with the actual `BaseDataset._compute_actions` produced a maximum error of **4.768 × 10⁻⁹**, including yaw crossing ±π.
- Planned plot coordinates correspond to the SE(2)-integrated model actions. Recorded future poses do not enter the CEM proposals or loss; the goal RGB is used only by the goal-image LPIPS objective, and later contexts are model predictions. Fixed diffusion noise is consistent between candidate batches of 32 and the final batch of 1.
- The final rendered rollout's recomputed loss and search loss are distinct fields, with their measured differences recorded above; the separate exported-media audit also passes.

## Evidence boundaries

- Paper benchmark cards retain the published Table 2 / Table 10 values in `paper.json`; the new planner does not replace or remeasure them.
- Curated motion predictions and new navigation videos are qualitative examples. Their selection and per-example errors do not establish a new aggregate benchmark score or physical execution success rate.
- The prior 2026-10-08 record covers the broader site integration and its historical navigation samples. Those results do not validate the new navigation outputs.
- Public acceptance combines the two reports below. All covered behaviors have passing evidence, but no single final public report ran all checks to completion. Public media startup was slow in this test environment; passing playback behavior does not establish a loading-performance target.
- The optional live GPU inference service was not exercised by these browser runs; its health endpoint was mocked offline.

## First deployment — successful delivery, browser acceptance failed

- Deployed commit: `7cdf56ca8ff09f8f3efd0e0f42c40d9417b122f8`.
- GitHub Actions run **37881633118** completed successfully: <https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/37881633118>.
- SHA256 checks for **11 published files** matched the public checkout. Evidence: `revision_20261009/published-hashes.json` in the NAS website-artifacts directory.
- The public browser run beginning at **04:02:16 UTC** failed on `tartan-drive-36`: maximum GT/OpenNWM drift was **1.692 seconds**, above the 0.3-second acceptance limit. Different arrival times for the two media streams exposed a synchronization issue that the local server run did not reproduce.
- This failure led to paired loading and joint pause/resume during buffering. The final fix's local and actual interruption checks are recorded above; the first release's failed result is retained here.

Failed browser evidence:

```text
/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/revision_20261009/2026-10-09T04-02-16-588Z-b0b8k1ng.github.io/
```

Deployment and matching file hashes establish successful delivery of the first revision; they do not override the failed browser check.

## Intermediate and final synchronization releases

- Intermediate commit `54cd9fa43d10373746258dee503088bfe9aee424` deployed successfully through [Actions run 37882791524](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/37882791524). Its standard local browser run at `2026-10-09T04-11-23-818Z-127.0.0.1/` passed with maximum drift **0.006912 seconds**.
- The subsequent actual interruption check found that buffering paused both streams but reset ground truth to zero while prediction remained near 1.5 seconds. Rounding the buffering barrier's playback time to frame boundaries caused an unnecessary backwards seek. Passing the standard run did not override this failure.
- Final commit `d82e5fe7f573a1df24835879fc2bb3eb41c0d059` preserves shared playback time and limits seeks to at most one per barrier. It passed the final local and interruption checks above. [Actions run 37883217041](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/37883217041) deployed successfully; **12 public-file hashes** match the checkout, recorded in `revision_20261009/published-sync-hashes.json`.
- Its public-URL browser run at `2026-10-09T04-19-03-193Z-b0b8k1ng.github.io/` failed after waiting **30 seconds** for the first card to play. The report contains no HTTP errors, JavaScript errors, or failed requests. Subsequent Chrome network diagnostics in `motion-public-cdp.json` showed slow HTTP/2 media transfer, with only about **9.8 KB** of prediction media received during the first 50 seconds; this was not a paired-buffering barrier failure. That failed report remains part of the record.

## Public acceptance — combined coverage complete

Both runs targeted `https://b0b8k1ng.github.io/OpenNWM/` at deployed commit `d82e5fe7f573a1df24835879fc2bb3eb41c0d059`. No website source changed between these runs. The test harness increased media waits to **120,000 ms** to accommodate the measured public transfer rate.

### Main public run — motion, navigation, and layouts passed; whole run timed out

- Started **04:25:12 UTC**, finished **04:34:01 UTC** on 2026-10-09.
- All **9 motion cards / 18 videos** passed source-path geometry, native pointer/keyboard clipping, playback synchronization, and per-card pause checks. Maximum drift was **0.008296 seconds**.
- Motion startup waits ranged from **17.618 to 92.427 seconds**. The network delay affects initial loading; the videos remained synchronized after playback began.
- All **3 curved-navigation cards** passed manifest/SVG geometry, media playback, and video-time cursor checks.
- All **16 layout checks** passed across both sections, both languages, and **320, 390, 768, and 1440 px**.
- No JavaScript errors, HTTP errors, or failed requests were recorded. The supervising agent visually inspected the actual public `navigation-en-1440` and `motion-zh-390` screenshots and found them normal.
- The last automatic-playback check after reload still used the old **20-second** wait at `check.cjs:184` and timed out. Consequently, this report correctly retains **`passed: false`**; only its completed checks are counted as passing evidence.

Evidence:

```text
/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/revision_20261009/2026-10-09T04-25-12-505Z-b0b8k1ng.github.io/
```

### Public playback-lifecycle follow-up — PASS

The harness then applied the same 120-second media wait to the remaining lifecycle checks and ran only those checks, without repeating the already completed 9 motion, 3 navigation, and 16 layout checks.

Working directory:

```text
/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/revision_20261009
```

Exact command:

```bash
ONLY_LIFECYCLE=1 MEDIA_TIMEOUT_MS=120000 BASE_URL=https://b0b8k1ng.github.io/OpenNWM/ conda run --no-capture-output -n base node check.cjs
```

- Exit status **0**, report **`passed: true`**, started **04:34:40 UTC** and finished **04:36:26.843 UTC** on 2026-10-09.
- Automatic offscreen pause/resume, global pause persistence, and per-card pause persistence across scrolling all passed.
- No JavaScript errors, HTTP errors, or failed requests were recorded.
- `lifecycleOnly: true`; the empty motion/navigation/layout arrays indicate intentionally omitted checks, whose evidence is in the main run above.

Evidence:

```text
/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/revision_20261009/2026-10-09T04-34-40-053Z-b0b8k1ng.github.io/
```

Together, these public runs verify all requested browser behaviors in the acceptance script. They retain the main run's timeout rather than presenting it as a single complete passing run. Both runs used an offline health mock and did not test live GPU inference.

The machine-readable summary records `combinedAcceptancePassed: true`, `mainRunOverallPassed: false`, `lifecycleRunPassed: true`, and **12** matching published files:

```text
/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/revision_20261009/public-acceptance-summary.json
```

The public checkout is clean, and its final `git diff --check` exited **0** (reported by the supervising agent).
