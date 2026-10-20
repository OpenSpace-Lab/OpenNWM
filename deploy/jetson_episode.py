#!/usr/bin/env python3
"""File-only recorded-episode waypoint selection; no robot interfaces."""
from __future__ import annotations

import argparse
import bisect
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

sys.dont_write_bytecode = True
from jetson_infer import (REPO, load_models, add_sampler_arguments, validate_sampler_arguments,
                          checkpoint_relative_path, checkpoint_digest)
from jetson_video import trajectory_poses

MODEL_HORIZON_SECONDS = 1.0
MODEL_FRAME_OFFSET = MODEL_HORIZON_SECONDS * 4
NORMALIZED_K = MODEL_FRAME_OFFSET / 128


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def relative_pose(start, end):
    dx, dy = float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"])
    yaw = float(start["yaw"])
    angle = float(end["yaw"]) - yaw
    return [math.cos(yaw) * dx + math.sin(yaw) * dy,
            -math.sin(yaw) * dx + math.cos(yaw) * dy,
            math.atan2(math.sin(angle), math.cos(angle))]


def qualifies(pose):
    return (.25 <= math.hypot(*pose[:2]) <= .5
            and math.radians(15) <= abs(pose[2]) <= math.radians(30))


def candidates():
    """45 unique endpoints; retain the original seven names and trajectories."""
    paths = trajectory_poses()
    paths["forward_slow"] = [[p[0] / 2, p[1] / 2, p[2]] for p in paths["forward"]]
    for name, sign in (("rotate_left", 1), ("rotate_right", -1)):
        paths[name] = [[0., 0., sign * math.pi / 4 * k / 8] for k in range(1, 9)]
    for distance in (.5, 1., 1.5, 2.):
        distance_tag = "%g" % distance
        if distance not in (1., 2.):
            paths["forward_" + distance_tag] = [[distance * k / 8, 0., 0.] for k in range(1, 9)]
        for degrees in (15., 22.5, 30., 45.):
            if distance == 2. and degrees == 45.:
                continue  # Already present as left_arc/right_arc.
            angle = math.radians(degrees)
            radius = distance / angle
            for side, sign in (("left", 1), ("right", -1)):
                name = "%s_arc_%s_%g" % (side, distance_tag, degrees)
                paths[name] = [[radius * math.sin(angle * k / 8),
                                sign * radius * (1 - math.cos(angle * k / 8)),
                                sign * angle * k / 8] for k in range(1, 9)]
    for degrees in (15., 22.5, 30.):
        for side, sign in (("left", 1), ("right", -1)):
            paths["rotate_%s_%g" % (side, degrees)] = [
                [0., 0., sign * math.radians(degrees) * k / 8] for k in range(1, 9)]
    return paths


def prepare(args):
    from PIL import Image, ImageDraw

    with (args.cleaned / "aligned_pose.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    times = [int(row["timeline_ns"]) for row in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Non-monotonic frame timeline")
    with (args.raw / "aligned_pose.csv").open() as stream:
        raw_rows = {int(r["frame_index"]): r for r in csv.DictReader(stream)}

    def nearest(target):
        right = bisect.bisect_left(times, target)
        options = [i for i in (right - 1, right) if 0 <= i < len(rows)]
        return min(options, key=lambda i: abs(times[i] - target))

    possible = []
    for i, start in enumerate(rows):
        requested = [times[i] + int(s * 1e9) for s in (-.75, -.5, -.25, 0., 2.)]
        indices = [nearest(t) for t in requested]
        if any(abs(times[j] - t) > 40_000_000 for j, t in zip(indices, requested)):
            continue
        if len(set(indices)) != 5 or indices[3] != i:
            continue
        window = rows[indices[0]:indices[-1] + 1]
        if any(r["pair_valid"] != "True" or r["segment_id"] != start["segment_id"] for r in window):
            continue
        pose = relative_pose(start, rows[indices[-1]])
        if qualifies(pose):
            possible.append((indices, pose))
    # Choose both turn directions and a third disjoint window, before model scoring.
    choices = (
        ("left_turn", lambda p: p[2] > 0),
        ("right_turn", lambda p: p[2] < 0),
        ("turn_3", lambda p: True),
    )

    def cost(pose):
        return abs(math.hypot(*pose[:2]) - .375) / .125 + abs(abs(math.degrees(pose[2])) - 22.5) / 7.5

    pairs = []
    for name, accept in choices:
        eligible = [item for item in possible if accept(item[1]) and all(
            abs(times[item[0][3]] - p["start_timeline_ns"]) > 5_000_000_000 for p in pairs)]
        if not eligible:
            raise ValueError("No valid pair for " + name)
        indices, pose = min(eligible, key=lambda item: cost(item[1]))
        refs = []
        for index in indices:
            row = rows[index]
            original = int(row["original_frame_index"])
            raw = raw_rows[original]
            if int(raw["timeline_ns"]) != times[index] or raw["pair_valid"] != "True":
                raise ValueError("Raw/cleaned frame mapping differs")
            filename = raw["file"]
            path = (args.raw / filename).resolve()
            if args.raw not in path.parents:
                raise ValueError("Frame path escapes episode")
            refs.append({"file": filename, "sha256": digest(path), "original_frame_index": original,
                         "cleaned_frame_index": index, "timeline_ns": times[index],
                         "source_elapsed_s": (times[index] - times[0]) / 1e9})
        horizon = (times[indices[-1]] - times[indices[3]]) / 1e9
        pairs.append({"id": name, "segment_id": int(rows[indices[3]]["segment_id"]),
                      "start_timeline_ns": times[indices[3]], "history": refs[:4], "goal": refs[4],
                      "history_offsets_seconds": [(r["timeline_ns"] - times[indices[3]]) / 1e9 for r in refs[:4]],
                      "horizon_seconds": horizon,
                      "model_horizon_seconds": MODEL_HORIZON_SECONDS,
                      "model_frame_offset": MODEL_FRAME_OFFSET,
                      "normalized_k": NORMALIZED_K,
                      "measured_goal_relative_m_rad": pose,
                      "distance_m": math.hypot(*pose[:2]), "yaw_change_deg": math.degrees(pose[2]),
                      "threshold_passed": qualifies(pose)})
    result = {"episode": args.raw.name, "raw_csv_sha256": digest(args.raw / "aligned_pose.csv"),
              "cleaned_csv_sha256": digest(args.cleaned / "aligned_pose.csv"),
              "selection_rule": "0.25 <= distance_m <= 0.5 AND 15 <= abs(wrapped yaw difference_deg) <= 30",
              "pair_sampling": "Left turn, right turn, then another window; starts >5 s apart; prefer 0.375 m and 22.5 deg; not a random benchmark",
              "time_basis": "host receive timeline; closest frames within 40 ms; same continuous segment",
              "physical_camera_pose_sync_verified": False, "pose_source": "local robot odometry",
              "pairs": pairs, "robot_motion_commands_sent": False}
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "pairs.json", result)
    canvas = Image.new("RGB", (960, len(pairs) * 310), "#14181e")
    draw = ImageDraw.Draw(canvas)
    for i, pair in enumerate(pairs):
        y = i * 310
        draw.text((12, y + 8), "%s | %.3f m | yaw %+.2f deg | dt %.3f s" %
                  (pair["id"], pair["distance_m"], pair["yaw_change_deg"], pair["horizon_seconds"]), fill="white")
        for column, ref in enumerate((pair["history"][-1], pair["goal"])):
            with Image.open(args.raw / ref["file"]) as image:
                canvas.paste(image.convert("RGB").resize((480, 270)), (column * 480, y + 40))
            draw.text((column * 480 + 12, y + 24), "%s original frame %d" %
                      ("START" if column == 0 else "GOAL", ref["original_frame_index"]), fill="white")
    canvas.save(args.output / "pairs.png")
    print(json.dumps(result, ensure_ascii=False), flush=True)


def infer(args):
    import numpy as np
    import torch
    import lpips
    from PIL import Image, ImageDraw
    from isolated_nwm_infer import model_forward_wrapper
    from misc import get_transform, normalize_data

    plan = json.loads((args.output / "pairs.json").read_text())
    if digest(args.raw / "aligned_pose.csv") != plan["raw_csv_sha256"]:
        raise ValueError("Episode differs from pair-selection input")
    for pair in plan["pairs"]:
        if not qualifies(pair["measured_goal_relative_m_rad"]):
            raise ValueError("Pair does not meet distance/yaw threshold")
        for ref in pair["history"] + [pair["goal"]]:
            path = (args.raw / ref["file"]).resolve()
            if args.raw not in path.parents or digest(path) != ref["sha256"]:
                raise ValueError("Input image integrity mismatch")
    if (args.output / "results.json").exists():
        raise FileExistsError("This experiment already has results")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    paths = candidates()
    names = list(paths)
    device = torch.device("cuda:0")
    torch.set_num_threads(2)
    torch.hub.set_dir(str(args.model_root / "planning/torch/hub"))
    calibration = args.model_root / "planning/lpips/v0.1/alex.pth"
    backbone = args.model_root / "planning/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
    if not calibration.is_file() or not backbone.is_file():
        raise FileNotFoundError("Offline scoring weights are missing")
    scorer = lpips.LPIPS(net="alex", model_path=str(calibration), verbose=False).eval().requires_grad_(False).to(device)
    results = {"complete": False, "method": "%d fixed candidate endpoints ranked by LPIPS; not CEM or closed-loop navigation" % len(names),
               "candidate_count": len(names),
               "pairs_sha256": digest(args.output / "pairs.json"), "steps": args.steps, "seed": args.seed,
               "translation_units": "training waypoint-spacing units; NOT calibrated meters",
               "rotation_units": "radians", "coordinate_frame": "hypothetical source body frame: x forward, y left, yaw CCW",
               "context": "4 real RGB frames near -0.75,-0.5,-0.25,0 s; deterministic VAE posterior mode",
               "time_condition": "Both models use a fixed 1-second horizon: rel_t = 4 / 128 = 0.03125, independent of recorded frame intervals and diffusion sampling steps; horizon_seconds records the source interval only",
               "model_horizon_seconds": MODEL_HORIZON_SECONDS,
               "model_frame_offset": MODEL_FRAME_OFFSET, "normalized_k": NORMALIZED_K,
               "image_transform": "repository 4:3 center crop then 224x224 resize and [-1,1] normalization",
               "scoring": "float32 decoded predictions vs real goal; no video compression or goal VAE reconstruction",
               "noise": "same DDPM noise across candidates and models for each pair",
               "trajectory_note": "Only endpoints are predicted/scored; eight poses describe the predefined candidate path",
               "robot_motion_commands_sent": False, "eligible_for_execution": False,
               "scorer_sha256": {"lpips": digest(calibration), "alexnet": digest(backbone)},
               "code_sha256": {p.name: digest(p) for p in (Path(__file__), REPO / "deploy/jetson_infer.py", REPO / "deploy/jetson_video.py", REPO / "nwm/isolated_nwm_infer.py")},
               "trajectories": paths, "models": {}}
    if args.sampler != "ddpm":
        results["sampler"] = args.sampler
        results["noise"] = "Same seeded diffusion noise across candidates and models for each pair"
        if args.sampler == "ddim":
            results["ddim_eta"] = 0.0 if args.ddim_eta is None else args.ddim_eta
    with torch.inference_mode():
        for model_name in ("opennwm", "nwm-b"):
            torch.manual_seed(args.seed)
            config, models = load_models(model_name, args.model_root, args.steps, device,
                                         sampler=args.sampler, ddim_eta=args.ddim_eta)
            _, _, tokenizer = models
            if int(config.dataset.context_size) != 4:
                raise ValueError("Expected four-frame context")
            transform = get_transform(config.dataset.image_size, config.dataset.mean, config.dataset.std)
            checkpoint = checkpoint_relative_path(model_name, args.sampler)
            results["models"][model_name] = {"checkpoint": checkpoint, "sha256": checkpoint_digest(args.model_root / checkpoint), "pairs": {}}
            for pair_index, pair in enumerate(plan["pairs"]):
                started = time.monotonic()
                print("START %s / %s" % (model_name, pair["id"]), flush=True)
                images = []
                for ref in pair["history"] + [pair["goal"]]:
                    with Image.open(args.raw / ref["file"]) as image:
                        images.append(transform(image.convert("RGB")))
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    latent = tokenizer.vae.encode(torch.stack(images[:4]).to(device)).latent_dist.mode() * tokenizer.scaling_factor
                predictions = []
                for first in range(0, len(names), args.batch_size):
                    batch_names = names[first:first + args.batch_size]
                    values = np.asarray([paths[name][-1] for name in batch_names], dtype=np.float32)
                    if model_name == "nwm-b":
                        stats = {k: np.asarray(v) for k, v in config.dataset.action_stats.items()}
                        values[:, :2] = normalize_data(values[:, :2], stats)
                    context = latent[None].repeat(len(batch_names), 1, 1, 1, 1)
                    predicted = model_forward_wrapper(models, context, torch.as_tensor(values[:, None], device=device),
                        None, int(config.model.generator.input_size), device, num_cond=4,
                        rel_t=torch.full((len(batch_names),), NORMALIZED_K, device=device),
                        skip_tokenizer=True, motion_type="real", sample_keys=[pair_index] * len(batch_names),
                        noise_seed=args.seed, noise_stream="go2/episode/endpoint", progress=False)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        rgb = tokenizer.decode(predicted, denormalize=True).float().clamp(0, 1)
                    if tuple(rgb.shape) != (len(batch_names), 3, 224, 224) or not torch.isfinite(rgb).all():
                        raise ValueError("Invalid prediction")
                    predictions.append(rgb.cpu())
                    del context, predicted, rgb
                    print("BATCH %s / %s: %d/%d candidates" %
                          (model_name, pair["id"], first + len(batch_names), len(names)), flush=True)
                predicted = torch.cat(predictions)
                goal = images[-1][None].to(device)
                scores = []
                for first in range(0, len(names), args.batch_size):
                    batch = predicted[first:first + args.batch_size].to(device)
                    scores.extend(scorer(batch * 2 - 1, goal.expand(len(batch), -1, -1, -1)).flatten().cpu().tolist())
                    del batch
                if not all(math.isfinite(s) for s in scores):
                    raise ValueError("Invalid LPIPS score")
                order = sorted(range(len(names)), key=lambda i: (scores[i], i))
                chosen = names[order[0]]
                selected = {"selected_candidate": chosen, "terminal_waypoint": paths[chosen][-1],
                            "horizon_seconds": pair["horizon_seconds"],
                            "model_horizon_seconds": MODEL_HORIZON_SECONDS,
                            "model_frame_offset": MODEL_FRAME_OFFSET,
                            "normalized_k": NORMALIZED_K,
                            "waypoints_relative_to_start": paths[chosen],
                            "waypoint_times_seconds": [MODEL_HORIZON_SECONDS * k / 8 for k in range(1, 9)],
                            "ranking": [{"candidate": names[i], "lpips": scores[i]} for i in order],
                            "best_second_score_gap": scores[order[1]] - scores[order[0]],
                            "inference_and_scoring_seconds": round(time.monotonic() - started, 3)}
                results["models"][model_name]["pairs"][pair["id"]] = selected
                columns = 5
                canvas = Image.new("RGB", (224 * columns, 264 * math.ceil((len(names) + 2) / columns)), "#14181e")
                draw = ImageDraw.Draw(canvas)
                tiles = [("START", images[3] * .5 + .5), ("GOAL", images[4] * .5 + .5)]
                tiles += [("%s\nLPIPS %.4f%s" % (name, scores[i], " *" if name == chosen else ""), predicted[i]) for i, name in enumerate(names)]
                for i, (label, pixels) in enumerate(tiles):
                    x, y = i % columns * 224, i // columns * 264
                    draw.text((x + 4, y + 6), label, fill="white")
                    array = (pixels.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
                    canvas.paste(Image.fromarray(array), (x, y + 40))
                canvas.save(args.output / (pair["id"] + "-" + model_name + ".png"))
                save_json(args.output / "results.json", results)
                print(json.dumps({"model": model_name, "pair": pair["id"], **selected}, ensure_ascii=False), flush=True)
                del predicted, predictions, latent
            del models, tokenizer, _
            gc.collect()
            torch.cuda.empty_cache()
    results["complete"] = True
    save_json(args.output / "results.json", results)
    print("COMPLETE " + str(args.output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "infer"))
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cleaned", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-root", type=Path, default=REPO.parents[1] / "models")
    parser.add_argument("--steps", type=int, default=None)
    add_sampler_arguments(parser, default_steps=50)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4, 7), default=7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    validate_sampler_arguments(parser, args)
    if args.mode == "prepare" and (args.sampler != "ddpm" or args.ddim_eta is not None):
        parser.error("Sampler options require infer mode")
    for name in ("raw", "cleaned", "output", "model_root"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    if not 2 <= args.steps <= 1000 or not 0 <= args.seed < 2**32:
        parser.error("Invalid steps or seed")
    if args.mode == "prepare":
        if args.cleaned is None:
            parser.error("--cleaned is required for preparation")
        prepare(args)
    else:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                          MPLBACKEND="Agg", TORCH_HOME=str(args.model_root / "planning/torch"),
                          NWM_VAE_PATH=str(args.model_root / "sd-vae-ft-ema"))
        sys.path.insert(0, str(REPO / "nwm"))
        with tempfile.TemporaryDirectory(prefix="opennwm-episode-") as temporary:
            os.environ["MPLCONFIGDIR"] = temporary
            infer(args)


if __name__ == "__main__":
    main()
