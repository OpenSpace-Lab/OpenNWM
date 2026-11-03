# OpenNWM website

The [project page](https://openspace-lab.github.io/OpenNWM/) uses plain HTML, CSS and JavaScript. All public media and fonts live in `assets/`; the optional GPU service is separate.

## Preview and publish

From the repository root:

```bash
conda run --no-capture-output -n nwm python -m http.server 8000 \
  --directory website --bind 127.0.0.1
```

Open <http://127.0.0.1:8000/>. Use HTTP because the page loads JSON manifests.

`.github/workflows/pages.yml` tests and packages the static site, then deploys the artifact directly from the public `OpenSpace-Lab/OpenNWM` repository to <https://openspace-lab.github.io/OpenNWM/>. Deployment is skipped while the repository is private.

For initial setup, a repository administrator must make `OpenSpace-Lab/OpenNWM` public and select **Settings → Pages → Build and deployment → Source → GitHub Actions**. Allow the `github-pages` environment to deploy from `main`, then run **Build and publish public OpenNWM website** from the Actions tab. Making the repository public alone does not enable Pages. Later changes to website files on `main` trigger deployment automatically.

The workflow can also synchronize the artifact to [OpenNWM-site](https://github.com/OpenSpace-Lab/OpenNWM-site) when `OPENNWM_SITE_DEPLOY_KEY` is configured. This optional mirror is not required for the project page.

To build the same artifact locally with Python 3.11 or newer, choose an output directory outside the repository that does not yet exist:

```bash
conda run --no-capture-output -n base python tools/build_project_pages.py \
  --output-dir /path/to/output/opennwm-site \
  --report /path/to/output/opennwm-site-build.json
```

## Content

| Files | Purpose |
| --- | --- |
| `index.html`, `reference-theme.css` | Layout, authors and theme |
| `research.js`, `results.css`, `content/paper.json` | Paper results and citation |
| `content/go2.json`, `go2.js` | Selected robot navigation clips |
| `content/demos.json`, `content/navigation.json` | Prediction and offline planning examples |
| `content/playground.json`, `content/live-service.json` | Demo scenes, presets and API address |
| `content/datasets.json`, `content/cinematic.json` | Dataset previews and opening film |
| `media-scheduler.js`, `media-delivery.js` | Playback, mirror selection and fallback |

Keep paths, source IDs, timing, sampling protocols and hashes consistent when replacing media. The page is English only. Legacy `Zh` fields contain English for catalog compatibility.

The current method figure is the vector export documented in [METHOD_FIGURES.md](content/METHOD_FIGURES.md). Paper statistics and qualitative media are separate: see [paper sources](content/PAPER_SOURCES.md), the [Go2 gallery protocol](content/GO2_MANIFEST.md) and [planetary evaluation](content/PLANETARY_EVALUATION.md).

## Media delivery

`content/media-delivery.json` and the `media-delivery-config` block in `index.html` must match. The ModelScope mirror serves the original media bytes with HTTP Range support; failed requests fall back to local assets or the pinned source archive recorded in the manifest. Keep the original resolution, timing and prediction content.

`media-scheduler.js` loads visible videos first and pauses offscreen groups. `media-cache-worker.js` caches complete video bytes for repeated playback. The Go2 manifest selects reviewed videos and posters; superseded files are excluded from deployment.

## Live demo

The HTTPS endpoint in `content/live-service.json` accepts custom trajectories. The service uses the OpenNWM Stage 2 checkpoint with 50 DDPM steps and returns 16 frames at 4 FPS. Preset results retain their own sampling protocol. When the API is offline, preset videos remain available.

See the [inference service guide](server/README.md) for GPU setup, queues and API details. Static hosting does not execute the model. If the tunnel address changes, update and publish `content/live-service.json`.
