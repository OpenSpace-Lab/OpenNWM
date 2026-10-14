# Reference palette and layout — 2026-10-10

The public website is restyled using the colors and layout of [AWAM](https://knightdby.github.io/awam-demo/) and [Cosmos 3](https://research.nvidia.com/labs/cosmos-lab/cosmos3/). OpenNWM media, methods, data, and the existing 9 prediction / 3 navigation examples are preserved.

## Visual changes

- Charcoal `#0d0e10`, light text `#eeeeee`, muted `#a4a7ac`, lime `#b6ee65`.
- Arial/Helvetica system typography, 1120px content width, 12px cards, consistent section spacing.
- Inset round-corner video hero with centered title, paper subtitle, and four resource pills; compact film controls.
- Deep gray method diagrams, galleries, metrics, trajectory canvas, dataset modal, and citation card.
- Mobile resource buttons form a 2×2 grid; the navigation menu uses the same dark palette.
- Dataset-card scene subtitles are hidden to remove redundant fine print.
- No model inference, sample replacement, metrics change, or video synchronization change.

## Local checks

- Chromium visual checks at 1440, 768, 390, and 320px, in English and Chinese: no overflow or JavaScript errors. Desktop and mobile screenshots were inspected against both references.
- Corrected the mobile hero pause-button position, primary-button foreground/background contrast, and excess space below transfer cards.
- All 10 interaction checks are covered by passing reports: 9 motion cards / 18 videos with no baseline; native keyboard and pointer wipe clipping; representative video synchronization (maximum measured drift 0.000010s); 3 equal-scale curved navigation paths and representative video/cursor alignment; three method stages; benchmark controls; dataset filtering, all 19 sources, modal and Escape; playground presets, scene selection, drawing and clearing; language switching and mobile overflow.
- Dataset / Results / Resources and both mobile menus received an additional visual review.
- JS syntax checks and scoped whitespace checks passed. Optional GPU inference was not exercised.

## Evidence

Artifacts: `/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/style_references_20261010/`.

- `awam/` and `cosmos/`: reference screenshots and measured styles.
- `validation/visual-report.json`: eight language/viewport checks and screenshots.
- `validation/interaction-smoke-summary.json`: consolidated passing interaction evidence. Original test-assumption failures remain in individual logs (19 sources instead of 16; reduced-motion playback must use the UI; local SimpleHTTP has no Range support).
- `validation/review-dataset-results-resources.json`: focused secondary visual review.

## Publication

Commit `62012a11626a49bbd10836f841f2680c62536c38` was pushed to the public `main` branch. [GitHub Pages deployment](https://github.com/B0B8K1ng/b0b8k1ng.github.io/actions/runs/38021431606) completed successfully.

Actual URL: https://b0b8k1ng.github.io/OpenNWM/ . Ten published files match the public checkout by SHA-256 (`validation/published-hashes.json`).

Actual public browser checks at 1440px and 390px confirm the dark theme, Arial, 1120px desktop content, 36px/24px hero corners, four resource links, section order, 9 motion cards / 18 videos, 3 navigation cards, native slider clipping, hidden redundant dataset subtitles, no horizontal overflow, and a working Chinese mobile menu. Public desktop/mobile hero and motion screenshots were inspected. No JavaScript errors or missing website assets were observed.

The first public script queried the dataset before its asynchronous initialization and exited; a second run waited for all tested sections. Its visual and interaction assertions passed, but it returned exit 1 because its generic HTTP check included the existing optional `/OpenNWM/api/health` 404 on static Pages. `validation/public-acceptance-summary.json` classifies only that exact known response as expected and records passing website checks; raw `public-report.json`, both logs, and screenshots remain unchanged. The optional GPU service was not tested, and the previous full public-media synchronization acceptance remains in the 2026-10-09 record.
