#!/usr/bin/env python3
"""Record physical MarsLab rover trajectories with Isaac Sim 4.1, without ROS.

Run with Isaac's python.sh. One process records one trajectory so failed or
completed episodes never reuse live physics state. --write-specs needs no GPU.
RGB and body ground truth are acquired at the same simulation timestamp; the
recorder fails if the camera cannot deliver a matching rendered frame.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

LOG = logging.getLogger("record_marslab")
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "MarsLab"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_and_sync_capture(output: Path, manifest: dict, expected_frames: int, expected_commands: int) -> None:
    """Validate persisted outputs before an explicit process-exit shutdown."""
    from PIL import Image

    frames = manifest["frames"]
    if len(frames) != expected_frames or len(list((output / "rgb").glob("*.png"))) != expected_frames:
        raise RuntimeError("Capture validation failed: frame count mismatch")
    first_time = frames[0]["timestamp"]
    for index, frame in enumerate(frames):
        if frame["index"] != index or abs(frame["timestamp"] - first_time - index / manifest["fps"]) > 1e-5:
            raise RuntimeError("Capture validation failed: inconsistent frame cadence")
        if abs(frame["camera_timestamp"] - frame["timestamp"]) > 1e-5:
            raise RuntimeError("Capture validation failed: RGB/pose timestamp mismatch")
        path = output / frame["path"]
        with Image.open(path) as image:
            if list(image.size) != manifest["metadata"]["camera"]["resolution"] or image.mode != "RGB":
                raise RuntimeError("Capture validation failed: invalid RGB dimensions or mode")
            image.verify()
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    with (output / "commands.jsonl").open("rb") as handle:
        count = sum(1 for line in handle if json.loads(line))
        os.fsync(handle.fileno())
    if count != expected_commands:
        raise RuntimeError(f"Capture validation failed: {count} commands, expected {expected_commands}")
    manifest["capture_validation"] = {"png_files_verified": expected_frames,
                                      "commands_verified": count, "sync_and_cadence_verified": True,
                                      "artifacts_fsynced": True}


def default_specs() -> dict:
    trajectories = []
    # These are candidate starts, not a claim that all terrain is traversable.
    # Run pilots and freeze a validated plan before collecting evaluation data.
    for index, (split, duration, xy, yaw, turn) in enumerate([
        ("train", 25, [-12, -10], 0.0, 0.0),
        ("train", 25, [-12, -4], 0.12, 0.025),
        ("train", 25, [-12, 3], -0.12, -0.025),
        ("train", 25, [12, 10], math.pi, 0.0),
        ("train", 25, [12, 4], math.pi + 0.12, 0.025),
        ("train", 25, [12, -3], math.pi - 0.12, -0.025),
        ("val", 15, [-10, 13], -0.2, -0.02),
        ("test", 10, [-8, -14], 0.2, 0.02),
        ("test", 10, [10, 14], math.pi + 0.2, -0.02),
        ("test", 10, [8, -13], math.pi - 0.2, 0.0),
    ]):
        number = sum(item["split"] == split for item in trajectories)
        trajectories.append({
            "id": f"{split}_{number:02d}", "split": split, "seed": 1700 + index,
            "duration_s": duration,
            "spawn": {"mode": "dem_relative", "xy": xy, "yaw_rad": yaw},
            "commands": [{"start_s": 0.0, "v": 0.8, "w": turn}],
        })
    return {"schema_version": 1, "description": "Candidate independent trajectories; validate spawn safety before recording.", "trajectories": trajectories}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marslab-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--rover-usd", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--specs", type=Path)
    parser.add_argument("--write-specs", type=Path, help="Write the proposed train/val/test plan and exit without starting Isaac.")
    parser.add_argument("--episode", help="Trajectory id in --specs; each invocation records exactly one trajectory.")
    parser.add_argument("--pilot", action="store_true", help="Record two stationary seconds followed by two commanded seconds.")
    parser.add_argument("--duration", type=float, help="Override episode duration in simulation seconds.")
    parser.add_argument("--spawn-xy", type=float, nargs=2, help="Override DEM-relative XY start, in meters.")
    parser.add_argument("--yaw", type=float, help="Override yaw, in radians.")
    parser.add_argument("--speed", type=float, default=0.8, help="Pilot forward speed, m/s.")
    parser.add_argument("--yaw-rate", type=float, default=0.0, help="Pilot yaw rate, rad/s.")
    parser.add_argument("--gpu-id", type=int, default=0, help="Isaac renderer device index.")
    parser.add_argument("--physics-gpu-id", type=int, help="PhysX device index; defaults to --gpu-id.")
    parser.add_argument("--gpu-physics", action="store_true", help="Enable GPU dynamics; CPU PhysX is the default.")
    parser.add_argument("--resolution", type=int, default=224, choices=[224, 256])
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--physics-hz", type=int, default=60)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--warmup-seconds", type=float, default=0.0,
                        help="Drive with the first episode command before recording; all warmup commands are logged.")
    parser.add_argument("--render-warmup-frames", type=int, default=128,
                        help="Render at least this many frames at the recording pose with physics held fixed, before saving RGB.")
    parser.add_argument("--sun-intensity", type=float, default=1000.0)
    parser.add_argument("--dome-intensity", type=float, default=250.0)
    parser.add_argument("--shutdown-mode", choices=["native", "process-exit"], default="native",
                        help="process-exit validates/fsyncs outputs and exits without the broken Isaac 4.1 native plugin teardown.")
    parser.add_argument("--camera-fov", type=float, default=90.0, help="Horizontal pinhole field of view in degrees.")
    parser.add_argument("--camera-pitch", type=float, default=6.6, help="Downward camera pitch in degrees.")
    parser.add_argument("--camera-translation", type=float, nargs=3, default=[0.752094567, -0.774359584, 1.837690473])
    parser.add_argument("--gravity", type=float, default=3.72)
    args = parser.parse_args()
    if args.write_specs:
        if args.write_specs.exists():
            parser.error("--write-specs refuses to overwrite an existing file")
        write_json(args.write_specs, default_specs())
        return args
    if not args.scene or not args.output:
        parser.error("--scene and --output are required for recording")
    if not args.pilot and not (args.specs and args.episode):
        parser.error("Select --pilot or --specs FILE --episode ID")
    if args.fps <= 0 or args.physics_hz <= 0 or args.physics_hz % args.fps:
        parser.error("physics-hz must be a positive integer multiple of fps")
    if args.render_warmup_frames < 1:
        parser.error("render-warmup-frames must be positive")
    if not 1 < args.camera_fov < 175 or args.settle_seconds < 0 or args.warmup_seconds < 0:
        parser.error("invalid camera FOV or settling time")
    if not all(math.isfinite(value) and value >= 0 for value in
               (args.settle_seconds, args.warmup_seconds, args.sun_intensity, args.dome_intensity)):
        parser.error("pre-roll durations and light intensities must be finite and nonnegative")
    return args


def select_episode(args: argparse.Namespace) -> dict:
    if args.pilot:
        episode = {"id": args.episode or "pilot", "split": "pilot", "seed": 1700, "duration_s": 4,
                   "spawn": {"mode": "dem_relative", "xy": [0, 0], "yaw_rad": 0},
                   "commands": [{"start_s": 0, "v": 0, "w": 0}, {"start_s": 2, "v": args.speed, "w": args.yaw_rate}]}
    else:
        plan = json.loads(args.specs.read_text())
        identifiers = [item["id"] for item in plan["trajectories"]]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Trajectory ids must be unique")
        matches = [item for item in plan["trajectories"] if item["id"] == args.episode]
        if len(matches) != 1:
            raise ValueError(f"Episode {args.episode!r} not found exactly once")
        episode = matches[0]
    if Path(episode["id"]).name != episode["id"] or episode["id"] in ("", ".", ".."):
        raise ValueError("Episode id must be a single directory name")
    if args.duration is not None:
        episode["duration_s"] = args.duration
    if args.spawn_xy is not None:
        episode["spawn"].update(mode="dem_relative", xy=args.spawn_xy)
    if args.yaw is not None:
        episode["spawn"]["yaw_rad"] = args.yaw
    if episode["split"] not in ("train", "val", "test", "pilot"):
        raise ValueError("Invalid split")
    duration = float(episode["duration_s"])
    if not math.isfinite(duration) or duration <= 0 or not math.isclose(duration * args.fps, round(duration * args.fps)):
        raise ValueError("Duration must be positive and contain an integral number of frames")
    commands = episode["commands"]
    starts = [float(item["start_s"]) for item in commands]
    if not starts or starts[0] != 0 or starts != sorted(set(starts)):
        raise ValueError("Commands must start at zero with strictly increasing start_s")
    if not all(math.isfinite(float(item[key])) for item in commands for key in ("start_s", "v", "w")):
        raise ValueError("Nonfinite command")
    return episode


def command_at(episode: dict, seconds: float) -> tuple[float, float]:
    command = episode["commands"][0]
    for item in episode["commands"]:
        if float(item["start_s"]) > seconds + 1e-9:
            break
        command = item
    return float(command["v"]), float(command["w"])


def record(args: argparse.Namespace, episode: dict) -> None:
    import numpy as np
    import yaml
    from PIL import Image

    root = args.marslab_root.resolve()
    sys.path.insert(0, str(root))
    # Keep the compatibility boundary versioned with this recorder. The sibling
    # MarsLab checkout supplies original assets and simulator-independent helpers.
    import marslab_isaac4 as compatibility

    build_scene = compatibility.build_scene
    boot = compatibility.boot
    configure_control = compatibility.configure_control
    sample_camera = compatibility.sample_camera
    apply_control = compatibility.apply_control

    config_path = root / "configs/robots/perseverance.yaml"
    config = yaml.safe_load(config_path.read_text())["rover"]
    rover_usd = (args.rover_usd or (config_path.parent / config["usd_path"])).resolve()
    for path in (args.scene, rover_usd):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output.resolve() / episode["id"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "rgb").mkdir()
    np.random.seed(int(episode["seed"]))
    metadata = {
        "simulator": "Isaac Sim 4.1 / MarsLab physical Perseverance rover", "ground_truth": "PhysX rigid body state; no image-derived poses",
        "scene_path": str(args.scene.resolve()), "scene_sha256": sha256(args.scene),
        "rover_usd_path": str(rover_usd), "rover_usd_sha256": sha256(rover_usd),
        "rover_config_sha256": sha256(config_path), "recorder_sha256": sha256(Path(__file__)),
        "compatibility_sha256": sha256(Path(compatibility.__file__)),
        "marslab_helper_sha256": {name: sha256(root / name) for name in (
            "marslab/robots/differential.py", "marslab/robots/rover_control.py",
            "marslab/sim/physics_scene.py", "marslab/sim/point_instancer.py", "marslab/sim/spawn_clearance.py")},
        "specs_sha256": sha256(args.specs) if args.specs else None,
        "episode_spec": episode, "config": config, "gravity_m_s2": args.gravity,
        "coordinate_convention": "right-handed, X forward / Y left / Z up; quaternion wxyz; meters / seconds / radians",
        "gpu_id": args.gpu_id, "physics_gpu_id": args.physics_gpu_id if args.physics_gpu_id is not None else args.gpu_id,
        "gpu_physics": args.gpu_physics, "pid": os.getpid(),
        "camera": {"resolution": [args.resolution] * 2, "horizontal_fov_deg": args.camera_fov,
                   "local_translation_m": args.camera_translation, "pitch_down_deg": args.camera_pitch},
        "evaluation_protocol": "independent held-out trajectories in one scene; not cross-scene generalization",
        "settle_seconds": args.settle_seconds, "warmup_seconds": args.warmup_seconds,
        "render_warmup_frames": args.render_warmup_frames,
        "warmup_command_vw": list(command_at(episode, 0.0)),
        "shutdown_mode": args.shutdown_mode,
    }
    manifest = {"schema_version": 1, "status": "initializing", "trajectory_id": episode["id"],
                "split": episode["split"], "seed": int(episode["seed"]), "fps": args.fps,
                "physics_hz": args.physics_hz, "metadata": metadata, "frames": []}
    write_json(output / "trajectory.json", manifest)
    app = world = articulation = body = camera = control = None
    started = time.monotonic()
    try:
        app = boot(args)
        world, articulation, body, camera, details = build_scene(args, episode, config, rover_usd)
        metadata.update(details)
        control = configure_control(articulation, config["control"])
        frame_count = round(float(episode["duration_s"]) * args.fps)
        steps_per_frame = args.physics_hz // args.fps
        with (output / "commands.jsonl").open("w") as command_file:
            pre_steps = {"settle": round(args.settle_seconds * args.physics_hz),
                         "warmup": round(args.warmup_seconds * args.physics_hz)}
            remaining_steps = sum(pre_steps.values())
            metadata["pre_roll_start_sim_time"] = float(world.current_time)
            for phase, steps in pre_steps.items():
                requested = (0.0, 0.0) if phase == "settle" else command_at(episode, 0.0)
                for _ in range(steps):
                    applied = apply_control(articulation, control, requested, 1 / args.physics_hz)
                    command_file.write(json.dumps({"timestamp": float(world.current_time),
                                                   "episode_time": -remaining_steps / args.physics_hz,
                                                   "phase": phase, "command_vw": list(requested), **applied}) + "\n")
                    world.step(render=False)
                    remaining_steps -= 1
                LOG.info("%s finished after %.3f simulation seconds", phase, steps / args.physics_hz)
            metadata["render_warmup"] = compatibility.warmup_camera(
                world, body, camera, args.resolution, args.render_warmup_frames)
            metadata["recording_start_sim_time"] = float(world.current_time)
            metadata["recording_start_position_m"] = np.asarray(body.get_world_pose()[0], dtype=float).tolist()
            start_sim = float(world.current_time)
            manifest["status"] = "recording"
            for index in range(frame_count):
                if not app.is_running():
                    raise RuntimeError("SimulationApp stopped before recording completed")
                rgba, camera_time, render_frame = sample_camera(world, camera, args.resolution)
                position, quaternion = body.get_world_pose()
                position = np.asarray(position, dtype=float)
                quaternion = np.asarray(quaternion, dtype=float)
                if not np.isfinite(position).all() or not np.isfinite(quaternion).all() or abs(np.linalg.norm(quaternion) - 1) > 1e-3:
                    raise RuntimeError("Invalid ground-truth body pose")
                if abs(float(world.current_time) - (start_sim + index / args.fps)) > 1e-5:
                    raise RuntimeError("Physics acquisition cadence drifted")
                relative_path = f"rgb/{index:06d}.png"
                Image.fromarray(rgba[:, :, :3], "RGB").save(output / relative_path)
                requested = command_at(episode, index / args.fps)
                frame = {"index": index, "path": relative_path, "timestamp": float(world.current_time),
                         "episode_time": index / args.fps, "camera_timestamp": camera_time, "rendering_frame": render_frame,
                         "position_m": position.tolist(), "quaternion_wxyz": quaternion.tolist(),
                         "command_vw": list(requested),
                         "linear_velocity_world_m_s": np.asarray(body.get_linear_velocity(), dtype=float).tolist(),
                         "angular_velocity_world_rad_s": np.asarray(body.get_angular_velocity(), dtype=float).tolist()}
                manifest["frames"].append(frame)
                for substep in range(steps_per_frame):
                    episode_time = index / args.fps + substep / args.physics_hz
                    requested = command_at(episode, episode_time)
                    applied = apply_control(articulation, control, requested, 1 / args.physics_hz)
                    command_file.write(json.dumps({"timestamp": float(world.current_time), "episode_time": episode_time, "phase": "record",
                                                   "command_vw": list(requested), **applied}) + "\n")
                    world.step(render=False)
                if index % args.fps == 0:
                    command_file.flush()
                    write_json(output / "trajectory.json", manifest)
                    LOG.info("frame=%d/%d sim_time=%.3f body_xyz=%s", index + 1, frame_count, frame["timestamp"], frame["position_m"])
        positions = np.asarray([frame["position_m"] for frame in manifest["frames"]])
        validate_and_sync_capture(output, manifest, frame_count,
                                  sum(pre_steps.values()) + frame_count * steps_per_frame)
        manifest.update(status="captured", frame_count=len(manifest["frames"]), elapsed_wall_seconds=time.monotonic() - started,
                        path_length_m=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()))
        write_json(output / "trajectory.json", manifest)
        LOG.info("captured trajectory=%s frames=%d path_length_m=%.3f", output, frame_count, manifest["path_length_m"])
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_wall_seconds=time.monotonic() - started)
        write_json(output / "trajectory.json", manifest)
        LOG.exception("Trajectory capture failed")
        raise
    finally:
        if app is not None:
            if args.shutdown_mode == "process-exit":
                # Two pilots completed and validated acquisition but crashed in
                # Isaac 4.1 SimulationApp.close -> unload_all_plugins. An explicit
                # process boundary releases CUDA/PhysX through the OS, without
                # pretending native cleanup succeeded. Never use exit 0 for a
                # partial or unvalidated capture.
                successful = manifest["status"] == "captured" and bool(manifest.get("capture_validation"))
                manifest.update(status="complete" if successful else "failed",
                                shutdown_status="process_exit_requested", native_cleanup="not_attempted",
                                native_cleanup_completed=False)
                metadata["shutdown_reason"] = "Isaac Sim 4.1 native unload_all_plugins segfault reproduced after complete capture in two pilots"
                write_json(output / "trajectory.json", manifest)
                for directory in (output / "rgb", output):
                    fd = os.open(directory, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                LOG.info("Validated process-exit: capture_status=%s exit_code=%d; native teardown deliberately skipped",
                         manifest["status"], 0 if successful else 1)
                logging.shutdown()
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0 if successful else 1)
            # Isaac 4.1 can crash unloading native plugins with a live sensor
            # graph / simulation view. Release these while Kit is still alive.
            manifest["shutdown_status"] = "started"
            write_json(output / "trajectory.json", manifest)
            try:
                compatibility.release_scene(world, camera)
                world = articulation = body = camera = control = None
                gc.collect()
                app.close()
            except BaseException as exc:
                manifest.update(status="failed", shutdown_status="failed", shutdown_error=f"{type(exc).__name__}: {exc}")
                write_json(output / "trajectory.json", manifest)
                raise
            manifest["shutdown_status"] = "complete"
            if manifest["status"] == "captured":
                manifest["status"] = "complete"
            write_json(output / "trajectory.json", manifest)
            LOG.info("shutdown complete; trajectory status=%s", manifest["status"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.write_specs:
        LOG.info("Wrote candidate specs to %s", args.write_specs)
        return
    record(args, select_episode(args))


if __name__ == "__main__":
    main()
