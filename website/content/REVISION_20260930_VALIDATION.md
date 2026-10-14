# Website revision validation — 2026-09-30

Published at <https://b0b8k1ng.github.io/OpenNWM/>.
Public commit: `fa383fa3b4557a518925cc1d0ebb455258237df7`.
[GitHub Pages deployment 36725653537](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/36725653537): completed / success.

## Delivered

- One 62.708-second, 1920 × 1080, 24 FPS film: 12 Earth scenes, 2 Moon scenes, 2 Mars scenes, then a 225-tile wall with no text overlays. All CASIA-Nav office trajectories are excluded.
- Moon footage uses LuSNAR simulation. Mars footage uses continuous left-camera frames from the newly downloaded SynMars-TW `forward_9` testing sequence. Both are disclosed as simulation.
- Seekable Earth / Moon / Mars / Many Worlds chapters, 12 Earth stops and a moving robot progress marker.
- The title O contains half Earth, quarter Moon and quarter Mars textures. Every fresh page load defaults to English; manual Chinese switching remains available.
- Three selected demo scenes (Gravel trail, Garden path, Coral hallway), each with three real action-conditioned predictions. Glass corridor is removed from the interactive demo and model comparison.
- All 19 dataset cards play continuous-frame videos, with offscreen pause and video dialogs.

## Validation

- Combined film: all 1,505 frames decoded; 225 unique wall trajectories from 15 sources; no CASIA offices; continuous Mars source frames verified.
- Dataset media: 16 newly generated clips fully decoded, alongside 3 existing clips; all 19 cards decoded and played in the browser.
- Demo: all 9 presets decoded and played; 16 predicted frames, 4 FPS, 4 seconds each. Source hashes and action provenance checked. Directional outputs differ. New candidates were visually reviewed; a weaker plaza candidate was rejected.
- Backend: 12 CPU server tests passed. New inference used GPU 1, and completed successfully without stopping existing experiments.
- Final browser integration: chapter seeks and robot position, Earth stops, all dataset cards, all demo presets, dialogs, default English despite an old Chinese preference, manual Chinese, mobile 390 px layout and reduced motion passed. No JavaScript exceptions or missing resources. No GPU calls during this integration run.
- Public export: 101 media references resolve locally; no symlinks or private machine paths; all files below 100 MB; copied files match the source; `git diff --check` passed. Push and GitHub Pages deployment succeeded.
- Public smoke check: homepage and all three manifests returned HTTP 200. The homepage serves `earth-space-v3-20260930` and the planetary O; the cinematic manifest has the v3 combined film and four chapters; datasets has 19 videos; playground has 3 scenes and 9 presets. An initial proxy tunnel 502 cleared on the first retry. Full playback validation was performed locally before publication.

## Limits

The hero shows dataset inputs, not model predictions. Planetary predictions remain explicitly labeled stress-test failures because this checkpoint drifts into terrestrial environments. Curated demo predictions are qualitative and retain imperfect geometry and detail; they are not new benchmark claims.

GitHub Pages plays precomputed predictions. A custom mouse-drawn trajectory requires a connected GPU service; the offline UI does not substitute a preset as its prediction.

## Evidence (private NAS)

Root: `/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/`.

- `redesign_20260929/cinematic_v3/validation.json`: media/source audit.
- `redesign_20260929/browser/frontend-final-v3.json` and `.log`: final browser integration, exit 0; desktop/mobile screenshots beside the report.
- `revision_20260930/datasets/validation.json` and `build.log`: dataset generation and decode checks, exit 0.
- `revision_20260930/demo/validation.json`, `probe.log`, `probe.exit`: genuine GPU candidates and provenance checks, exit 0; reviewed contact sheets in the same directory.

These validation notes are internal and are not included in the public website export.
