# Local trajectory demo

The standard-library HTTP server provides genuine OpenNWM DDPM inference.
The model loads on the first request and stays on one GPU.
It executes one request at a time, with at most three additional queued requests.
The GitHub Pages website reads the HTTPS API origin from `content/live-service.json`.
It submits arbitrary drawings, polls progress, and autoplays the returned video.
Genuine precomputed presets remain available when the API is offline; they never
substitute for a submitted custom trajectory.
Static and generated media support single HTTP byte ranges, allowing browsers to
seek without first downloading the whole video.

From the repository root, choose an idle GPU after checking `nvidia-smi`:

```bash
source .env
CUDA_VISIBLE_DEVICES=4 "$NWM_PYTHON" -u website/server/serve.py --port 8765
```

Open `http://127.0.0.1:8765/`. For a remote machine, forward the port with SSH
(`ssh -L 8765:127.0.0.1:8765 your-host`). The default bind address is loopback.
For local/forwarded use, omit `--api-only`. Public deployment uses `--api-only`
so that no static files, Python sources, or filesystem listings are exposed.
Only allowlisted API endpoints and three output filenames are served.

## Public single-GPU service

`bash tools/live_demo_start.sh` starts a detached API and an outbound HTTPS
Cloudflare quick tunnel. It sources `.env`, uses the explicit `NWM_PYTHON`
runtime, binds to loopback, and defaults to GPU 7 and 50 DDPM steps. Inspect GPU
availability first; the script prints GPU and disk state but never stops another
process. Output root: `/file_system/nas/algorithm/dujun.nie/opennwm_live_demo_20261029`.
Its `processes.json` records exact commands, GPU, PIDs and log paths.
The cloudflared 2026.10.5 Linux AMD64 binary was verified against the official
release SHA-256 before use.

The public website is [OpenNWM](https://openspace-lab.github.io/OpenNWM/).
Allowed browser origins: `https://openspace-lab.github.io`, the previous site's
`https://b0b8k1ng.github.io` for compatibility, and the local preview
`http://127.0.0.1:8766`. Origins contain no URL path. JSON CORS preflight and cross-origin media byte ranges
are supported. This is an intentionally public demo, not a private authenticated
API; CORS is not authentication. Admission is bounded to one active job, three
queued jobs, 20 requests per client per hour, at least 15 seconds between client
submissions, and 120 requests globally per hour. The tunnel supplies the client
address, accepted only from its loopback connector. No arbitrary image,
filesystem path, model, or executable input is accepted.

Keep both recorded processes running to retain the current public URL. Quick
tunnels have no uptime guarantee and their hostname changes when recreated;
after restarting one, update and publish `content/live-service.json` with its
new HTTPS origin. A stable production hostname requires a configured named
tunnel and domain, which are not currently present. See the official
[Cloudflare limitations](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/#limitations).

The model uses `NWM_NAV15_STAGE2_RUN`'s imported Stage 2 joint100k run by default
(override with `--run-dir`), the VAE/data paths in `.env`, and `ffmpeg` on `PATH`.
The Python runtime is the existing `NWM_PYTHON` environment. No additional web
framework is required. Generated job outputs go to NAS by default; use `--output`
to change that location. At most 100 finished jobs are retained, with a 24-hour
retention limit checked on submissions, completion and startup; expired media
directories are removed too. Completed jobs are restored after API restarts.

## API

- `GET /api/health`: availability, lazy model state, and queued request count.
- `GET /api/scenes`: the same public catalog as `content/playground.json`.
- `POST /api/predict`: JSON `{ "sceneId": "indoor-huron", "points": [[0,0],[0,1]], "seed": 0 }`; returns HTTP 202 with `jobId`.
- `GET /api/jobs/{jobId}`: `status` (`queued`, `running`, `completed`, `failed`), `progress` in `[0,1]`, and completed `video`/`poster`/`manifest` URLs plus the inference `protocol`.
- `GET /api/media/{jobId}/manifest.json`: exact drawn trajectory, body-frame actions, source/checkpoint hashes, seed and protocol; machine-local paths are removed.

Requests accept only allowlisted scenes, 2–512 finite points, a path
starting at `[0,0]`, x in `[-1,1]`, y in `[-0.25,1.5]`, and integer seeds in
`[0,2147483647]`. HTTP bodies are capped at 32 KiB. The server rejects unapproved browser
origins and returns HTTP 429 when its queue or rate budget is full.

## Motion and inference

Drawings use x right and y forward. One canvas unit corresponds to 12 normalized
dataset waypoint-spacing units. The path is sampled uniformly by arc length at
16 steps, interpreted as a four-second prediction at 4 FPS. Heading follows the
path tangent, starting forward. Each displacement rotates into the current body
frame: forward x, left y, yaw radians. The model's real-action adapter applies its
own min/max normalization. This planar control has no height or aerial motion.

Historical presets use real checkpoint inference, seed 0, 250 DDPM steps, a single source
image repeated into four context frames, and autoregressive RGB re-encoding.
The VAE uses posterior means so action comparisons and live retries share a
deterministic image encoding. This differs from the posterior sampling in the
benchmark evaluator; these demos are qualitative outputs, not benchmark scores.
Live custom requests use 50 DDPM steps, recorded in every returned manifest.
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
