#!/usr/bin/env python3
"""Rank saved hypothetical trajectories against a goal image; files only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
from jetson_infer import REPO

MODELS = ("opennwm", "nwm-b")
PATHS = ("stationary", "forward", "left_arc", "right_arc")


def last_frame(path):
    """Use the existing system OpenCV when the inference venv has no OpenCV."""
    from PIL import Image

    decoder = sys.executable if importlib.util.find_spec("cv2") else "/usr/bin/python3"
    code = """import cv2, sys
cap = cv2.VideoCapture(sys.argv[1])
fps = cap.get(cv2.CAP_PROP_FPS)
count, last = 0, None
while True:
    ok, frame = cap.read()
    if not ok: break
    assert frame.shape == (224, 224, 3), 'Unexpected video dimensions'
    count, last = count + 1, frame
cap.release()
assert fps == 4 and count == 8, 'Expected 8 frames at 4 FPS'
ok, encoded = cv2.imencode('.png', last)
assert ok
sys.stdout.buffer.write(encoded.tobytes())
"""
    result = subprocess.run([decoder, "-B", "-c", code, str(path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    return Image.open(io.BytesIO(result.stdout)).convert("RGB")


def read_candidates(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if not manifest.get("complete") or manifest.get("mode") != "direct_from_fixed_context":
        raise ValueError("A completed jetson_video.py direct prediction run is required")
    if (manifest["fps"], manifest["frames"], manifest["duration_seconds"]) != (4, 8, 2):
        raise ValueError("Expected a two-second, 4-FPS prediction run")
    if set(manifest["trajectories"]) != set(PATHS):
        raise ValueError("Expected the four documented candidate trajectories")
    for poses in manifest["trajectories"].values():
        if len(poses) != 8 or any(len(p) != 3 for p in poses):
            raise ValueError("Each trajectory must contain eight SE(2) poses")
        if not all(math.isfinite(v) for p in poses for v in p):
            raise ValueError("Trajectory poses must be finite")
    if manifest["prediction_times_seconds"] != [i / 4 for i in range(1, 9)]:
        raise ValueError("Unexpected prediction times")
    source_hash = hashlib.sha256((directory / "source.png").read_bytes()).hexdigest()
    if source_hash != manifest["camera"]["source_sha256"]:
        raise ValueError("Camera source differs from the generation manifest")
    frames, hashes = {}, {}
    for model in MODELS:
        frames[model] = {}
        for name in PATHS:
            filename = manifest["models"][model]["videos"][name]["file"]
            if Path(filename).name != filename or not filename.endswith(".mp4"):
                raise ValueError("Video filenames must be plain MP4 basenames")
            path = directory / filename
            frames[model][name] = last_frame(path)
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest, frames, hashes


def rank_candidates(scores, trajectories):
    if set(scores) != set(PATHS) or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("Expected one finite score per candidate")
    ranking = sorted(PATHS, key=lambda name: (scores[name], PATHS.index(name)))
    selected = ranking[0]
    return {"ranking": [{"candidate": name, "lpips": scores[name]} for name in ranking],
            "selected_candidate": selected,
            "first_waypoint": trajectories[selected][0],
            "terminal_waypoint": trajectories[selected][-1],
            "waypoints_relative_to_source": trajectories[selected],
            "best_second_score_gap": scores[ranking[1]] - scores[ranking[0]]}


def score_frames(frames, goal_path, root, device_name):
    import numpy as np
    import torch
    from PIL import Image

    calibration = root / "planning/lpips/v0.1/alex.pth"
    backbone = root / "planning/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
    if not calibration.is_file() or not backbone.is_file():
        raise FileNotFoundError("Local LPIPS and AlexNet weights are required; no download attempted")
    torch.hub.set_dir(str(root / "planning/torch/hub"))
    import lpips
    from misc import get_transform

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; explicitly choose --device cpu if appropriate")
    torch.set_num_threads(2)
    device = torch.device(device_name)
    scorer = lpips.LPIPS(net="alex", model_path=str(calibration), verbose=False).eval().to(device)
    transform = get_transform(224, [.5] * 3, [.5] * 3)
    with Image.open(goal_path) as goal_image:
        goal = transform(goal_image.convert("RGB"))
    goal_pixels = ((goal * .5 + .5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    scores = {}
    with torch.inference_mode():
        for model in MODELS:
            batch = torch.stack([torch.from_numpy(np.array(frames[model][name], copy=True))
                                 .permute(2, 0, 1).float() / 127.5 - 1 for name in PATHS]).to(device)
            values = scorer(batch, goal[None].to(device).expand_as(batch)).flatten().cpu().tolist()
            scores[model] = dict(zip(PATHS, values))
    provenance = {"lpips_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
                  "alexnet_sha256": hashlib.sha256(backbone.read_bytes()).hexdigest()}
    return scores, Image.fromarray(goal_pixels), provenance


def preview(source, goal, frames, selections):
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (896, 816), (20, 24, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 8), "Offline waypoint selection / NO ROBOT EXECUTION", fill="white", font=font)
    draw.text((8, 34), "Source model input", fill="white", font=font)
    draw.text((232, 34), "Goal model input" if goal else "Goal not supplied", fill="white", font=font)
    canvas.paste(source, (0, 56))
    if goal:
        canvas.paste(goal, (224, 56))
    for index, line in enumerate(("Future endpoint predictions: t = 2.0 s",
                                  "Score: LPIPS / AlexNet; lower is better",
                                  "XY: training waypoint units, NOT meters",
                                  "Collision / traversability: not assessed",
                                  "Prepared only: no selection" if not selections else
                                  "Selected per model; no cross-model winner")):
        draw.text((464, 62 + index * 25), line, fill="white", font=font)
    for row, model in enumerate(MODELS):
        for column, name in enumerate(PATHS):
            x, y = column * 224, 304 + row * 256
            draw.text((x + 6, y), model.upper() + " / " + name, fill="white", font=font)
            result = selections.get(model)
            label = "unscored"
            if result:
                score = next(v["lpips"] for v in result["ranking"] if v["candidate"] == name)
                label = "LPIPS %.4f%s" % (score, " | SELECTED" if result["selected_candidate"] == name else "")
            draw.text((x + 6, y + 16), label, fill="#9fe0ba", font=font)
            canvas.paste(frames[model][name], (x, y + 32))
    return canvas


def run(args):
    from PIL import Image

    manifest, frames, hashes = read_candidates(args.run_directory)
    selections, goal, scorer = {}, None, None
    if args.goal is not None:
        scores, goal, scorer = score_frames(frames, args.goal, args.model_root, args.device)
        selections = {model: rank_candidates(scores[model], manifest["trajectories"]) for model in MODELS}
    with Image.open(args.run_directory / "model-input.png") as source_file:
        source = source_file.convert("RGB")
    result = {"status": "ranked" if selections else "awaiting_goal_image",
              "method": "fixed-candidate terminal-image LPIPS ranking; not CEM",
              "prediction_source": "decoded H.264 terminal frames; compression can affect scores",
              "run_directory": str(args.run_directory), "source_sha256": manifest["camera"]["source_sha256"],
              "video_sha256": hashes, "diffusion_steps": manifest["diffusion_steps"],
              "prediction_mode": manifest["mode"], "prediction_times_seconds": manifest["prediction_times_seconds"],
              "translation_units": manifest["translation_units"], "rotation_units": "radians",
              "coordinate_frame": "hypothetical body frame at source timestamp; x forward, y left, yaw CCW",
              "camera_to_body_calibrated": False, "poses": "cumulative; not successive deltas",
              "trajectories": manifest["trajectories"], "goal": None,
              "goal_preprocessing": "repository 4:3 center crop and 224x224 resize; no VAE goal reconstruction",
              "generation_manifest_sha256": hashlib.sha256((args.run_directory / "manifest.json").read_bytes()).hexdigest(),
              "scorer": scorer, "selection": selections,
              "obstacle_check": "not_performed", "robot_motion_commands_sent": False,
              "eligible_for_execution": False}
    if args.goal is not None:
        result["goal"] = {"path": str(args.goal), "sha256": hashlib.sha256(args.goal.read_bytes()).hexdigest()}
    canvas = preview(source, goal, frames, selections)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        canvas.save(args.output / "selection-preview.png")
        shutil.copyfile(args.run_directory / "trajectories.png", args.output / "waypoints.png")
        (args.output / "selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except BaseException:
        for name in ("selection-preview.png", "waypoints.png", "selection.json"):
            (args.output / name).unlink(missing_ok=True)
        args.output.rmdir()
        raise
    print(json.dumps({"status": result["status"], "selection": selections, "output": str(args.output)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--goal", type=Path, help="Real target-view RGB photograph")
    mode.add_argument("--prepare-only", action="store_true", help="Preview candidates without selecting an action")
    parser.add_argument("--output", type=Path, required=True, help="New visualization directory; never overwritten")
    parser.add_argument("--model-root", type=Path, default=REPO.parents[1] / "models")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    for name in ("run_directory", "output", "model_root", "goal"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.output.exists():
        parser.error("Output directory already exists")
    if args.goal is not None and not args.goal.is_file():
        parser.error("Goal photograph does not exist")
    sys.path.insert(0, str(REPO / "nwm"))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MPLBACKEND="Agg",
                      TORCH_HOME=str(args.model_root / "planning/torch"))
    with tempfile.TemporaryDirectory(prefix="opennwm-select-") as temporary:
        os.environ["MPLCONFIGDIR"] = temporary
        run(args)


if __name__ == "__main__":
    main()
