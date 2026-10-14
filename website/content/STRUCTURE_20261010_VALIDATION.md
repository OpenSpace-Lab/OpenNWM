# Website structure revision — 2026-10-10

## Changes

- The actual hero MP4 now begins with the 18-second dataset video wall, followed by Earth, Moon and Mars. Original scene content, total 1,505 frames and 62.708333-second duration are preserved. Poster and all chapter/scene timestamps match the new order. The cinematic builder supports the new order on both full rebuild and existing-video reorder.
- Method uses exact high-resolution crops of paper Figure 2 (architecture, PDF page 3) and Figure 3 (training, page 7), with complete original artwork/colors and links to full resolution. No self-drawn method diagrams remain. Source PDF and output hashes are in `assets/paper/method-figures.json`.
- Section order: Method → Prediction visualization → Navigation visualization → Combined results → Interactive demo → Beyond Earth → Dataset → resource/footer utilities. All numbered headings and header anchors follow this order.
- Existing prediction metrics and four selected navigation benchmark cards now share Results. Navigation cards display images/video and paths, without goal-error numbers. Reported paper metrics and prediction/navigation media are unchanged.
- Moon 7–9 show recorded rover trajectories from `gt.txt`: anchor plus 16 target positions, equal metric scale, frame-aligned cursor in direct and autoregressive modes. Source validation is in `assets/space/lusnar-finetuned/trajectory-provenance.json`; synthetic controls from other space examples are not used.

## Validation

- Local browser acceptance exit 0: all six categories pass (section order/metrics; original figures; gallery/metric controls; hero order/chapter seeking/video-wall link; recorded space geometry and playback synchronization; eight English/Chinese responsive cases at 1440, 768, 390, 320 px).
- All three 17-point rendered space trajectories match the recorded positions up to one rigid reflection/rotation and a single uniform scale, with pairwise-distance error at most 4.27e-14 px. Direct and autoregressive Moon 7 playback markers matched frame indices with zero cursor-coordinate error.
- Dedicated method checks confirm readable desktop figures, complete mobile figures, and full-resolution PNG opening from a 320px viewport. No custom diagram nodes remain.
- Hero CPU reorder and full decode exit 0. Ten sampled positions, including the wall→Earth boundary and the final frame, correspond to the original content; the first ten wall frames are reconstructed without the old Mars transition. Poster matches the opening frame. Timeline consistency and idempotence checks pass.
- Existing nine GT/OpenNWM slider pairs and three curved navigation cases remain present. No new inference or GPU work was performed; no experiments were stopped.

Evidence:

- `/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/structure_20261010/acceptance.json` and `local.log`.
- `/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/paper_method_20261010/`.
- `/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/style_references_20261010/validation/hero-reorder/`.
- `/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/style_references_20261010/validation/space-trajectory-geometry.json`.

Some initial full-section screenshots captured lazy images before decoding following a language rerender. Dedicated method screenshots and checks wait for image decoding; those show both complete figures. No source change was required.

## Publication

Commit `44e3e5013643b65b971dd72c1fffe1f3d40cba76` was pushed to public `main`; [Pages workflow](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/38029698470) completed successfully.

Official URL: https://b0b8k1ng.github.io/OpenNWM/ . All 13 checked public files match the published checkout byte-for-byte (SHA256), including both original paper figures and the complete 40,008,733-byte hero video. The file verifier exited 0; `published-hashes.json` and `deployment.json` record the results.

Public browser checks passed section ordering, benchmark placement, original image decoding and full-resolution links, the nine prediction/three navigation cards, GT/OpenNWM sliders and metric controls. Dataset-wall and Earth chapter frames loaded; the Moon chapter also loaded. The first complete public run exited 1 after the Mars seek exceeded its 120-second loading timeout, so it is not recorded as a complete pass. The Moon boundary screenshot still shows the preceding Earth frame at the beginning of the existing 0.18-second crossfade; the full local video validation confirms the Moon imagery after that transition.

A separate public navigation check loaded all six start/goal images and all three posters, with no layout overflow or JavaScript errors. The first navigation video did not reach loadeddata within 120 seconds (readyState 0); that check exited 1 and is retained as a media-loading limitation. The optional `/api/health` returns the expected 404 on static Pages. Evidence is in `public-acceptance.json`, `public.log`, `public-navigation-media-report.json`, and `public-navigation-loaded.png` under the structure evidence directory. Remaining public trajectory/responsive checks are recorded separately.

The targeted remaining public checks exited 0: all three recorded space paths match source geometry (maximum error 4.27e-14 px); direct and autoregressive videos play with frame-aligned cursors and zero cursor-coordinate error. English and Chinese layouts pass at 1440, 768, 390 and 320 px, including paper-image decoding on mobile, benchmark rerendering and the mobile menu. No unexpected HTTP or JavaScript errors were recorded. Evidence: `public-remaining-acceptance.json`, `public-remaining.log`, and `public-remaining-*.png`. These successful targeted checks do not erase the initial media-loading timeouts.

A final targeted navigation check used the actual Play previews button with the same software-rendering browser flags. It exited 1 after 60 seconds: the first video emitted play/waiting/stalled, remained at currentTime 0 and readyState 0, and had networkState 2 with no media error. Media requests returned HTTP 206 `video/mp4`, but no body-data completion was recorded before the timeout. This remains an unresolved public media-loading limitation; no successful navigation-video playback is claimed for this revision's public checks. `public-navigation-playback-report.json` and `public-navigation-playback.log` preserve the evidence. Navigation images/posters, all local playback checks, and public space playback passed as recorded above. No source code was changed in response to these transfer timeouts.
