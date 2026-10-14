# Local trajectory demo

The standard-library HTTP server hosts the project website and its prediction API
on the same origin. The model loads on the first request and stays on one GPU.
It executes one request at a time, with at most three additional queued requests.
The GitHub Pages website uses genuine precomputed presets when the API is absent.
Static and generated media support single HTTP byte ranges, allowing browsers to
seek without first downloading the whole video.

From the repository root, choose an idle GPU after checking `nvidia-smi`:

```bash
source .env
CUDA_VISIBLE_DEVICES=4 "$NWM_PYTHON" -u website/server/serve.py --port 8765
```

Open `http://127.0.0.1:8765/`. For a remote machine, forward the port with SSH
(`ssh -L 8765:127.0.0.1:8765 your-host`). The default bind address is loopback.
This development server is intended for local/forwarded use; public hosting needs
an authenticated reverse proxy and HTTPS on the same origin. It does not create
an external endpoint or alter any existing experiment.

The model uses `NWM_NAV15_STAGE2_RUN`'s imported Stage 2 joint100k run by default
(override with `--run-dir`), the VAE/data paths in `.env`, and `ffmpeg` on `PATH`.
The Python runtime is the existing `NWM_PYTHON` environment. No additional web
framework is required. Generated job outputs go to NAS by default; use `--output`
to change that location. Only the latest 100 jobs remain in the in-memory API;
media/manifest files remain on disk for reproducibility.

## API

- `GET /api/health`: availability, lazy model state, and queued request count.
- `GET /api/scenes`: the same public catalog as `content/playground.json`.
- `POST /api/predict`: JSON `{ "sceneId": "indoor-huron", "points": [[0,0],[0,1]], "seed": 0 }`; returns HTTP 202 with `jobId`.
- `GET /api/jobs/{jobId}`: `status` (`queued`, `running`, `completed`, `failed`), `progress` in `[0,1]`, and completed `video`/`poster`/`manifest` URLs plus the inference `protocol`.
- `GET /api/media/{jobId}/manifest.json`: exact drawn trajectory, body-frame actions, source/checkpoint hashes, seed and protocol; machine-local paths are removed.

Requests accept only the six allowlisted scenes, 2–512 finite points, a path
starting at `[0,0]`, x in `[-1,1]`, y in `[-0.25,1.5]`, and integer seeds in
`[0,2147483647]`. HTTP bodies are capped at 32 KiB. The server rejects browser
writes from a different origin and returns HTTP 429 when its queue is full.

## Motion and inference

Drawings use x right and y forward. One canvas unit corresponds to 12 normalized
dataset waypoint-spacing units. The path is sampled uniformly by arc length at
16 steps, interpreted as a four-second prediction at 4 FPS. Heading follows the
path tangent, starting forward. Each displacement rotates into the current body
frame: forward x, left y, yaw radians. The model's real-action adapter applies its
own min/max normalization. This planar control has no height or aerial motion.

All presets use real checkpoint inference, seed 0, 250 DDPM steps, a single source
image repeated into four context frames, and autoregressive RGB re-encoding.
The VAE uses posterior means so action comparisons and live retries share a
deterministic image encoding. This differs from the posterior sampling in the
benchmark evaluator; these demos are qualitative outputs, not benchmark scores.
Sixteen predicted frames form a four-second MP4; the initial image is shown
separately. There is no interpolation or fabricated ground truth for drawn paths.

Moon and Mars use real mission source images. Their original capture intervals
are irregular, and translation is normalized by sequence median displacement.
The demo's four-second horizon is a model-step convention, not a claim about the
physical mission timeline or motion in meters.

The sixth scene uses a LuSNAR simulated lunar image and is labeled simulation.
Its controls use the same normalized drawing units; no calibrated camera/body
extrinsics or metric ground truth are implied.

The current checkpoint fails to preserve the lunar/Martian environment in these
space stress tests, instead hallucinating terrestrial structures. The website
labels these unmodified failures explicitly. This is not evidence of successful
space generalization. Diagnostic fixed-initial-context direct predictions are
retained separately on NAS; they also failed for both lunar inputs and Mars. They do not
replace the consistent autoregressive protocol used by the public playground.
`--mode direct --no-publish` selects that diagnostic protocol: cumulative
origin-frame action endpoints, horizon `1..16`, initial image held fixed, and
shared diffusion noise across horizons. It is not an autoregressive rollout.

Generate all 18 presets (three per scene), retaining private source/checkpoint
paths and exact actions on NAS while publishing sanitized provenance:

```bash
source .env
CUDA_VISIBLE_DEVICES=4 "$NWM_PYTHON" -u website/server/generate.py
"$NWM_PYTHON" -m unittest discover -s website/server -p 'test_*.py'
```

The generator skips completed scene batches. Each public preset links a JSON
manifest with its source image/weight SHA-256, canvas path, transformed actions,
seed, resolution implied by source/model, and diffusion configuration.

Use `--scenes <id> ...` to generate a subset without removing existing entries.
The static site shows only exact presets when no local API is available; arbitrary
drawings require live inference. Switching the frontend scene does not cancel a
GPU job already submitted. Old results can be ignored safely by the frontend.
