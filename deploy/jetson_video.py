#!/usr/bin/env python3
"""Offline multi-trajectory video prediction from one captured RGB file."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
from jetson_infer import (REPO, load_models, add_sampler_arguments, validate_sampler_arguments,
                          checkpoint_relative_path, checkpoint_digest)

FPS = 4
FRAMES = 8
PATHS = ("stationary", "forward", "left_arc", "right_arc")


def trajectory_poses():
    """Origin/body SE(2) endpoints: x forward, y left, yaw counterclockwise."""
    result = {name: [] for name in PATHS}
    for frame in range(1, FRAMES + 1):
        fraction = frame / FRAMES
        distance = 2.0 * fraction
        result["stationary"].append([0., 0., 0.])
        result["forward"].append([distance, 0., 0.])
        radius = 2.0 / (math.pi / 4)
        angle = math.pi / 4 * fraction
        for name, sign in (("left_arc", 1), ("right_arc", -1)):
            result[name].append([radius * math.sin(angle),
                                 sign * radius * (1 - math.cos(angle)), sign * angle])
    return result


def write_video(path, frames, bitrate=1800):
    """Pipe RGB frames directly to H.264/MP4; retain no frame dumps."""
    import numpy as np

    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[0] != FRAMES or frames.shape[-1] != 3:
        raise ValueError("Expected exactly eight RGB frames")
    if path.exists():
        raise FileExistsError(path)
    height, width = frames.shape[1:3]
    temporary = path.with_suffix(".partial.mp4")
    command = ["gst-launch-1.0", "-q", "fdsrc", "fd=0", "!", "rawvideoparse",
               "format=rgb", "width=" + str(width), "height=" + str(height),
               "framerate=4/1", "!", "videoconvert", "!", "video/x-raw,format=I420",
               "!", "x264enc", "speed-preset=fast", "bitrate=" + str(bitrate),
               "key-int-max=8", "bframes=0", "!", "h264parse", "!", "mp4mux",
               "faststart=true", "!", "filesink", "location=" + str(temporary)]
    try:
        result = subprocess.run(command, input=frames.tobytes(), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=60)
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_trajectory_plot(output, trajectories):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    for name, values in trajectories.items():
        poses = [[0., 0., 0.]] + values
        ax.plot([-p[1] for p in poses], [p[0] for p in poses], "o-", label=name)
    ax.set(xlabel="Right (waypoint-spacing units)", ylabel="Forward (waypoint-spacing units)",
           title="Hypothetical model trajectories / 2 seconds")
    ax.axis("equal")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def comparison_frames(videos):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    result = []
    for frame in range(FRAMES):
        canvas = Image.new("RGB", (896, 528), (20, 24, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), "Predicted RGB | t=%.2f s | hypothetical trajectories" %
                  ((frame + 1) / FPS), fill="white", font=font)
        for row, model in enumerate(("opennwm", "nwm-b")):
            for column, name in enumerate(PATHS):
                x, y = column * 224, 32 + row * 248
                draw.text((x + 7, y + 4), model.upper() + " / " + name,
                          fill="white", font=font)
                canvas.paste(Image.fromarray(videos[model][name][frame]), (x, y + 24))
        result.append(np.asarray(canvas))
    return result


def generate(args):
    import numpy as np
    import torch
    from PIL import Image
    from isolated_nwm_infer import model_forward_wrapper
    from misc import get_transform, normalize_data

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if list(args.directory.glob("*.mp4")):
        raise FileExistsError("Choose a directory without existing prediction videos")
    source = args.directory / "source.png"
    manifest_path = args.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if "camera" in manifest:
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["camera"]["source_sha256"]
    trajectories = trajectory_poses()
    manifest.update(fps=FPS, frames=FRAMES, duration_seconds=FRAMES / FPS,
                    prediction_times_seconds=[i / FPS for i in range(1, FRAMES + 1)],
                    mode="direct_from_fixed_context", diffusion_steps=args.steps, seed=args.seed,
                    translation_units="training waypoint-spacing units; not calibrated meters",
                    rotation_units="radians", trajectories=trajectories,
                    context="One captured RGB repeated four times; deterministic VAE posterior mode",
                    noise="Same DDPM noise across trajectories, horizons and both models",
                    robot_motion_commands_sent=False, models={}, complete=False)
    if args.sampler != "ddpm":
        manifest["sampler"] = args.sampler
        manifest["noise"] = "Same seeded diffusion noise across trajectories, horizons and both models"
        if args.sampler == "ddim":
            manifest["ddim_eta"] = 0.0 if args.ddim_eta is None else args.ddim_eta
    write_trajectory_plot(args.directory / "trajectories.png", trajectories)
    videos = {}
    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    with torch.inference_mode():
        for model_name in ("opennwm", "nwm-b"):
            torch.manual_seed(args.seed)
            config, models = load_models(model_name, args.model_root, args.steps, device,
                                         sampler=args.sampler, ddim_eta=args.ddim_eta)
            model, diffusion, tokenizer = models
            transform = get_transform(config.dataset.image_size, config.dataset.mean, config.dataset.std)
            with Image.open(source) as image:
                normalized = transform(image.convert("RGB"))
            if model_name == "opennwm":
                pixels = ((normalized * .5 + .5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
                Image.fromarray(pixels).save(args.directory / "model-input.png")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                latent = tokenizer.vae.encode(normalized[None].to(device)).latent_dist.mode()
                latent = latent * tokenizer.scaling_factor
            checkpoint_name = checkpoint_relative_path(model_name, args.sampler)
            model_manifest = {"checkpoint": checkpoint_name, "sha256": checkpoint_digest(args.model_root / checkpoint_name),
                              "parameter_load": "strict EMA", "videos": {}}
            videos[model_name] = {}
            for name, poses in trajectories.items():
                frames = []
                started = time.monotonic()
                print("START %s / %s" % (model_name, name), flush=True)
                for first in range(0, FRAMES, args.batch_size):
                    last = min(first + args.batch_size, FRAMES)
                    values = np.asarray(poses[first:last], dtype=np.float32)
                    if model_name == "nwm-b":
                        stats = {k: np.asarray(v) for k, v in config.dataset.action_stats.items()}
                        values[:, :2] = normalize_data(values[:, :2], stats)
                    context = latent[:, None].repeat(last - first, int(config.dataset.context_size), 1, 1, 1)
                    prediction = model_forward_wrapper(
                        models, context, torch.as_tensor(values[:, None], device=device), None,
                        int(config.model.generator.input_size), device,
                        num_cond=int(config.dataset.context_size),
                        rel_t=torch.arange(first + 1, last + 1, device=device, dtype=torch.float32) / 128.,
                        skip_tokenizer=True, motion_type="real", sample_keys=[0] * (last - first),
                        noise_seed=args.seed, noise_stream="go2-camera/direct", progress=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        rgb = tokenizer.decode(prediction, denormalize=True).float()
                    if tuple(rgb.shape) != (last - first, 3, 224, 224) or not torch.isfinite(rgb).all():
                        raise RuntimeError("Invalid predicted RGB frames")
                    arrays = (rgb.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
                    frames.extend(arrays)
                    del prediction, rgb, context
                    print("FRAMES %s / %s %d/%d" % (model_name, name, last, FRAMES), flush=True)
                filename = "%s-%s.mp4" % (model_name, name)
                write_video(args.directory / filename, frames)
                videos[model_name][name] = frames
                model_manifest["videos"][name] = {"file": filename,
                    "generation_seconds": round(time.monotonic() - started, 3)}
                manifest["models"][model_name] = model_manifest
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
                print("VIDEO_READY " + filename, flush=True)
            del models, model, diffusion, tokenizer, latent
            gc.collect()
            torch.cuda.empty_cache()
    write_video(args.directory / "comparison.mp4", comparison_frames(videos), bitrate=6000)
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("COMPLETE " + str(args.directory), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True, help="Contains source.png and optional camera manifest")
    parser.add_argument("--model-root", type=Path, default=REPO.parents[1] / "models")
    parser.add_argument("--steps", type=int, default=None)
    add_sampler_arguments(parser, default_steps=50)
    parser.add_argument("--batch-size", type=int, default=4, choices=(1, 2, 4, 8))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    validate_sampler_arguments(parser, args)
    args.directory = args.directory.expanduser().resolve()
    args.model_root = args.model_root.expanduser().resolve()
    if not (args.directory / "source.png").is_file() or not 2 <= args.steps <= 1000:
        parser.error("source.png must exist and steps must be 2..1000")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    if shutil.which("gst-launch-1.0") is None:
        parser.error("GStreamer with rawvideoparse, x264enc, h264parse and mp4mux is required")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      MPLBACKEND="Agg", TQDM_MININTERVAL="15", NWM_VAE_PATH=str(args.model_root / "sd-vae-ft-ema"))
    sys.path.insert(0, str(REPO / "nwm"))
    with tempfile.TemporaryDirectory(prefix="opennwm-mpl-") as temporary:
        os.environ["MPLCONFIGDIR"] = temporary
        generate(args)


if __name__ == "__main__":
    main()
