# Website revision — 2026-10-08

This revision is in the local OpenNWM workspace. It has not been published to GitHub Pages.

## Delivered

- Page order: opening → method → NavAnywhere → In motion → prediction benchmarks → navigation → interactive demo → Beyond Earth → resources.
- The opening screen links to Paper, Code, Demo, and **Dataset · NavAnywhere**.
- Method content follows paper §3.2–3.4, Figure 3, and Appendix C: dual-supervised latent actions; frozen-LAM video pretraining; physical-action alignment and CEM planning. World-model training phases are displayed separately.
- In motion has **9 scenes / 27 videos**, each comparing Ground truth, OpenNWM, and NWM. Three scene cards fit across the desktop layout, two on tablets, and one on phones. Visible cards play together; controls support individual and global pause.
- Navigation has **6 selected offline local plans** showing the recorded start, OpenNWM prediction, goal image, planned/reference trajectories, and goal error. Paper Tables 2 and 10 supply the separate benchmark cards. The selected videos are not physical closed-loop executions.
- Beyond Earth concludes the research story with **3 held-out simulated LuSNAR lunar scenes (Moon 7–9)**, following adaptation on 30 training windows for 1,000 updates. Direct and autoregressive comparisons are available. Earlier planetary stress-test assets remain in the source manifest but are not rendered in this section.
- Headings now use the normal-width Manrope sans serif face and stronger weights. Comparison metadata, resolution/FPS notes, provenance accordions, and repetitive descriptions have been removed from the displayed demos. Provenance remains in JSON and documentation.

## Media evidence

- Motion sources: `opennwm_pixel_corrected_20260924` inference archive. The seven new scenes retain the complete 16-second predictions, without temporal cherry-picking or interpolation. All 27 video files and 27 posters exist; video dimensions, frame rates, frame counts, durations, and newly exported file hashes were checked. Videos total 20.40 MiB.
- Navigation sources: `opennwm_navigation_gallery_20260924`, selecting RECON 003/000, SCAND 023/027, TartanDrive 001, and HuRoN 054 after metric and all-frame visual review. The six videos have 9 frames at 4 FPS and 224 × 224 resolution. Export hashes, trajectory ATE, and endpoint error were independently checked.
- Export commands run on CPU from the repository root:

  ```bash
  conda run --no-capture-output -n base python -u tools/build_website_motion_gallery.py
  conda run --no-capture-output -n base python tools/export_website_navigation.py
  ```

No model training or GPU inference was started for this webpage revision.

## Validation

- All 201 local `assets/` references in the content JSON resolve.
- JavaScript syntax checks pass for app, cinematic, playground, research, navigation, and motion-gallery modules.
- `git diff --check` passes for the website and new media export scripts.
- Desktop and mobile screenshots were visually reviewed for normal typography, compact comparisons, clear labels, and responsive layout.
- Browser integration completed with exit status **0**: all **48 media playback checks** passed (27 motion, 6 navigation, 6 lunar modes, and 9 interactive presets); both languages passed at 320, 390, 768, 1024, and 1440 px without document overflow. Dataset dialogs, benchmark controls, mobile navigation, reduced-motion behavior, automatic visible playback, and offscreen pause passed. There were no page JavaScript errors or missing-resource responses.
- Browser integration results and screenshots are recorded in the local evidence directory below. The test mocks the optional inference-service health response as offline and sends no generation requests; GPU inference was not exercised.

Local evidence (not needed by the website at runtime):

- `/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/revision_20261008/browser-report.json`
- `/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/revision_20261008/browser.log`
- `/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/revision_20261008/check.cjs`
- `/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/motion_review_20261008/export.log`

The browser test uses `playwright-core` from the NAS browser runtime and `/usr/bin/google-chrome`, with software rendering. Run from the repository root:

```bash
node /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/revision_20261008/check.cjs > /file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/revision_20261008/browser.log 2>&1
```
