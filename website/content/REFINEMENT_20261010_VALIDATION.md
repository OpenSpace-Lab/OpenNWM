# Website refinement — 2026-10-10

## Scope and status

This revision implements the five requested refinements:

1. Remove model parameter counts from the hero, result labels, accessibility text, and the 14 model-size fields in `content/paper.json`. The 17.5M dataset-frame count remains.
2. Replace navigation's number-only metric cards with baseline comparisons using the same chart, dataset selector, and metric controls as visual prediction. The charts sit side by side above 1100px and stack at or below 1100px.
3. Expand the selected navigation gallery from 3 to 8 examples, with full-frame visual review and recorded quality filters.
4. Remove visible table-number references and redundant result annotations. Source citations remain in the source records; benchmark values are unchanged.
5. Rename “THE IMAGINATION LAB” to “Demo”, including the navigation link and both language versions.

The local integrated acceptance result is **passed**. Deployment commit
`b3a99e28e5658618ddcb0897ddac190e39f4e829` completed successfully in the
[GitHub Pages workflow](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/38038052434).
Public-file verification passed for all 28 checked files, including the 5 newly
published navigation videos. Public browser checks passed the bilingual text,
structure, and chart controls. The default HTTP/2 browser connection did not
receive media body bytes, so public navigation playback was not confirmed in
that run. A subsequent HTTP/1.1 browser check passed actual public RECON 061
playback and trajectory synchronization. Both transport results are recorded below.

Earlier theme, structure, and playback records remain unchanged in
`REVISION_20261010_VALIDATION.md`, `STRUCTURE_20261010_VALIDATION.md`, and
`REVISION_20261009_VALIDATION.md`.

## Benchmark comparisons

Navigation supports RECON, SCAND, Office-Go2, and TartanDrive with ATE/RPE
controls. All numbers come from `paper.results.navigation`; missing values are
omitted, not converted to zero. The respective counts are 7, 6, 3, and 5 reported
models. OpenNWM is highlighted, and all bars start at zero. The previous standalone
OpenNWM metric cards and improvement sidecar are removed.

Focused checks verified all 8 dataset/metric combinations against the source JSON,
selection persistence across language changes, pressed states, and readable labels.
Model-size labels are absent from both visible prediction rows and their accessible
names. The NWM† definition remains to identify that baseline.

## Selected navigation examples

The manifest is version 3 and contains 8 selected examples from 23 completed
candidates. All selected examples use RECON; no cross-dataset diversity claim is
made for this curated gallery. Their exact source trajectories and frame windows
are recorded below. Each video has 17 frames, 224×224 resolution, 4 FPS, and a
4.25-second duration, with the recorded start as frame zero.

| Sample | RECON trajectory · frames | Offline ATE (m) | Endpoint error (m) | Final goal LPIPS | Inference snapshot |
| --- | --- | ---: | ---: | ---: | --- |
| `curved-recon-059` | `jackal_2019-08-06-15-48-05_13_r01` · 27–43 | 0.263 | 0.498 | 0.309 | `inference_source_reserve_20261010.py` |
| `curved-recon-088` | `jackal_2019-08-13-17-02-54_1_r04` · 3–19 | 0.638 | 0.770 | 0.383 | `inference_source.py` |
| `curved-recon-046` | `jackal_2019-10-20-16-29-40_11_r02` · 3–19 | 0.689 | 0.810 | 0.422 | `inference_source.py` |
| `curved-recon-025` | `jackal_2019-08-13-16-22-26_10_r02` · 59–75 | 0.565 | 0.951 | 0.352 | `inference_source_reserve_20261010.py` |
| `curved-recon-085` | `jackal_2019-09-17-17-35-40_7_r02` · 91–107 | 0.588 | 0.816 | 0.248 | `inference_source.py` |
| `curved-recon-074` | `jackal_2020-01-03-13-30-05_6_r04` · 27–43 | 0.296 | 0.571 | 0.565 | `inference_source_20261010.py` |
| `curved-recon-061` | `jackal_2019-09-12-16-20-46_14_r01` · 3–19 | 0.731 | 1.112 | 0.388 | `inference_source_20261010.py` |
| `curved-recon-078` | `jackal_2019-08-18-15-01-02_2_r00` · 3–19 | 0.807 | 1.327 | 0.299 | `inference_source.py` |

The selected quality ranges are:

- ATE: **0.263–0.807 m**; normalized ATE: **0.064–0.227**.
- Endpoint error: **0.498–1.327 m**; normalized endpoint error: **0.122–0.346**.
- Final goal LPIPS: **0.248–0.565**, evaluated on the retained final RGB separately from the search-batch loss.
- Total planned turn: **26.10–69.47°**, with at least two segments turning by 6° or more.

Selection requires endpoint error ≤1.5 m and normalized error ≤0.35, ATE ≤1 m
and normalized ATE ≤0.25, at least two 6° turning segments, total turn ≥25°, and
manual review of every predicted frame. The source manifest retains full precision.
These are curated qualitative offline plans, not closed-loop robot execution or
navigation success-rate measurements, and they do not replace the paper benchmarks.

The planner takes the start RGB repeated as four context frames and the goal RGB.
It uses independent goal-image CEM with one speed and three yaw-rate segments;
recorded actions and later observations are not planner inputs. Reference paths
are used only for candidate selection and post-hoc errors. Curves are integrated
from the optimized controls with exact SE(2) geometry, without smoothing or
substitution of ground-truth paths. Settings: seed 42, population 32, 6 elites,
2 CEM iterations, 50 diffusion steps, 16 predicted steps, and deterministic VAE
posterior means with autoregressive RGB feedback.

The checkpoint SHA256 is
`aabff9b5a3ab62a9145073892653a81019782511272933b6acfb41e21b48b4f4`.
The export validation passed source/checkpoint hashes, raw-control reconstruction,
original-reference geometry, frame counts, asset hashes, and encoded-frame fidelity.
Across all exported frames the minimum encoding PSNR is 36.499 dB; this measures
encoding fidelity to the saved predictions, not world-model accuracy. The manifest
SHA256 is `3e00994abf462ee8154fc83eb86c3b701d60155fef84a2573becbdfbec82bfb7`.

## Inference and export evidence

Archive root:
`/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/curved_navigation_20261009/`.
The historical archive name is retained; the expansion records are dated 2026-10-10.

Four expansion batches generated 16 additional candidates, all with exit status 0.
They ran directly in the `nwm` conda environment from
`/file_system/vepfs/algorithm/dujun.nie/code/OpenNWM`.

| GPU | Model PID | Candidate IDs | Log | Exit |
| --- | ---: | --- | --- | ---: |
| 0 | 2998124 | recon-061, recon-098, recon-020, recon-032, scand-047, huron-053 | `expansion-gpu0-20261010.log` | 0 |
| 1 | 2998230 | recon-044, recon-039, recon-068, recon-074, scand-046, huron-075 | `expansion-gpu1-20261010.log` | 0 |
| 2 | 3093859 | recon-025, recon-059 | `expansion-gpu2-20261010.log` | 0 |
| 3 | 3093981 | recon-084, recon-070 | `expansion-gpu3-20261010.log` | 0 |

`expansion-commands-20261010.json` preserves each exact command, runtime environment,
working directory, PID, GPU, log path, and exit status. The inference snapshots
listed in the sample table have individual hashes in `content/navigation.json`;
`inference_runs.json` maps retained samples to their actual source script.

Additional evidence in the archive:

- `expansion_selection_audit_20261010.json`: candidate metrics and selection criteria.
- `expansion_visual_review_20261010.json`: full-frame review decisions and exclusions.
- `runs/<sample>/all-predictions.jpg`: frame review sheets.
- `export_expansion_validation_20261010.json`: passed geometry, media, and provenance checks.
- `expansion-validation-20261010.log`: export validation output.

## Local integrated acceptance

Evidence root:
`/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/refinement_20261010/`.

`integrated-final-report.json` has `passed: true`, covering 97 assertions:
95 unchanged passing assertions from the complete run plus 2 focused passing
boundary checks. The original complete run is intentionally retained with exit 1
and `passed: false`: its only two failures incorrectly expected the charts to be
side by side at exactly 1100px. The specified layout stacks at 1100px and below.
No website source change was needed; `boundary-report.json` checks both languages
against that specification and passed with exit 0. The original failures have not
been erased or relabeled as a successful full run.

Coverage includes:

- English and Chinese layouts at 1440, 1100, 768, 390, and 320px, with no chart or page overflow.
- No model-size or table-number matches in visible text, relevant attributes, or accessibility snapshots; the 17.5M dataset count remains.
- Demo label, visible navigation access, anchor behavior, scene controls, and offline preset playback.
- Prediction and navigation benchmark controls.
- Exact agreement between all 8 manifest IDs and rendered cards.
- Real video playback for all 8 navigation examples, plus 24 cursor/progress checks at 0, 1.6, and 3.8 seconds.
- Gallery play/pause controls and no JavaScript, console, or HTTP errors.

The integrated check only routed the health endpoint to exercise offline Demo
behavior; no real-time inference request was sent. Earlier transient harness
assumptions about heading case, an obsolete domain option, and reduced-motion
autoplay were corrected before the retained 97-assertion report.

Files include `integrated-local-report.json`, `integrated-local.log`,
`integrated-local-exit-status`, `boundary-report.json`, `boundary.log`,
`boundary-exit-status`, and `integrated-final-report.json`. Screenshots include
`navigation-sample-1.png` through `navigation-sample-8.png`, bilingual
`nav-*.png` / `results-*.png`, and `results-*-1100-confirmed.png`.
`accessibility-en.json`, `accessibility-zh.json`, `body-en.txt`, and `body-zh.txt`
preserve the text audit.

The earlier focused chart evidence is in
`/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/navigation_chart_20261010/`.
JavaScript syntax, scoped whitespace checks, and formatting of the chart modules
passed. Existing method figures and the prediction-gallery synchronization were
preserved.

## Public acceptance

`public-hashes.json` records a successful check at 08:33 UTC on 2026-10-10:
28 of 28 files returned HTTP 200 on the first GET, with SHA256 matching the
published checkout. Coverage includes 8 code/JSON files, 5 new prediction videos,
and their 15 new start/goal/poster images. The newly published sample assets are
RECON 025, 059, 061, 074, and 085. The Pages workflow also reported
`completed/success` with the expected deployment SHA. The verification process
PID was 3209566 and its exit status was 0; the full request and hash records are
in `public-hashes.json`, with output in `public-hashes.log` under the evidence root.
It ran from `/file_system/vepfs/algorithm/dujun.nie/code/OpenNWM` with this command:

```bash
conda run --no-capture-output -n base python /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/refinement_20261010/verify_public_hashes.py > /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/refinement_20261010/public-hashes.log 2>&1
```

The first public browser acceptance check started as PID 3215364 from
`/file_system/vepfs/algorithm/dujun.nie/code/OpenNWM` with this exact command:

```bash
/file_system/vepfs/algorithm/dujun.nie/.npm-global/bin/node /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/refinement_20261010/check-published.cjs b3a99e28e5658618ddcb0897ddac190e39f4e829 > /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/refinement_20261010/published.log 2>&1
```

The first check passed all 9 bilingual structure/text/control checks and matched
10 loaded resources to the deployed commit. Its first RECON 061 video did not
start; the harness awaited an unbounded `play()` promise. The owned browser was
closed and the check exited 1, preserving `published-initial-report.json`.
RECON 078 was planned but not reached.

A focused media check ran as PID 3230701 from the same working directory:

```bash
/file_system/vepfs/algorithm/dujun.nie/.npm-global/bin/node /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/refinement_20261010/check-published-media.cjs b3a99e28e5658618ddcb0897ddac190e39f4e829 > /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/refinement_20261010/published-media.log 2>&1
```

It exited 1 after the RECON 061 gallery video and its direct public MP4 URL each
exceeded a 45-second playback limit. Both remained at currentTime 0, readyState 0,
and networkState 2, without a media error or loadedmetadata event. CDP recorded
HTTP/2 response headers (200/206) but zero media body bytes. No page JavaScript
or HTTP errors were recorded. The independent full-body GET and SHA256 checks
passed; these browser attempts did not establish playback or cursor success.
`published-media-report.json` and `published-final-report.json` preserve the
failure, while the local 8-video/24-cursor checks remain passed. Only the health
endpoint was mocked and no inference API was called.

### Transport diagnosis: HTTP/1.1 passed

The final bounded diagnosis added only `--disable-http2` to the same Chrome
launch flags. PID 3248456 ran from the same working directory and exited 0:

```bash
/file_system/vepfs/algorithm/dujun.nie/.npm-global/bin/node /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/refinement_20261010/check-http1-media.cjs b3a99e28e5658618ddcb0897ddac190e39f4e829 > /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/refinement_20261010/published-http1.log 2>&1
```

The actual protocol was `http/1.1`. The same public RECON 061 MP4 received all
237,926 bytes and played in both the direct URL and the real website gallery.
The gallery decoded at 224×224, duration 4.25 seconds, and reached readyState 4.
At 1.6 seconds and 3.8 seconds the markers exactly matched frames 6 and 15,
with 7 and 16 progress points respectively. This controlled comparison supports
a test-connection HTTP/2 media-transfer problem; no website change was needed.
The earlier HTTP/2 reports remain failed, and only this representative public
video is claimed as playback-verified in this revision; all 8 passed locally.
Evidence: `published-http1-report.json`, `published-http1-exit-status`,
`published-http1.log`, and visually inspected `published-http1-curved-recon-061.png`.
The exact public checkout remains clean at the deployment SHA.
