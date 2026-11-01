#!/usr/bin/env python3
"""Annotate fixed SynMars clips with frozen VGGT-Omega relative camera actions.

Future RGB is used offline to estimate action labels, never as NWM context.
Translations have relative units, not metres; source FPS is unknown.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LIMITATIONS = [
    "Actions are frozen-model visual estimates, not recorded controls or ground-truth poses.",
    "Reference future RGB is used offline for action annotation; NWM receives only four observed RGB frames and estimated actions.",
    "Monocular translation is normalized per clip: median adjacent planar displacement equals 0.25 relative action units, not metres. Physical speed is unavailable.",
    "Source timestamps/FPS are unavailable. Indices are spaced by four source frames; 4 FPS is presentation speed only.",
    "Splits hold out route IDs including both directions. Separate route IDs do not establish geographically disjoint terrain.",
    "Pitch, roll and vertical translation are discarded when projecting camera poses to planar actions.",
]


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_cached_pose_inputs(saved, sample):
    """Revalidate every RGB input, including dense-only retry frames, on resume."""
    selected_indices = sample["frame_indices"]
    if saved.get("sample_id") != sample["id"] or saved.get("frame_indices") != selected_indices:
        raise ValueError("Existing pose sample or selected frame indices changed")
    dense_indices = list(range(selected_indices[0], selected_indices[-1] + 1))
    input_indices = saved.get("pose_input_frame_indices")
    if (not isinstance(input_indices, list)
            or any(type(index) is not int for index in input_indices)
            or input_indices not in (selected_indices, dense_indices)):
        raise ValueError("Existing pose input frame indices differ from the selected or dense window")
    first_path = Path(sample["context_source_paths"][0])
    expected_paths = [str(first_path.with_name(f"forward_{sample['route_id']}_{frame:04d}_l.png"))
                      for frame in input_indices]
    saved_hashes = saved.get("pose_input_source_sha256")
    if not isinstance(saved_hashes, dict) or set(saved_hashes) != set(expected_paths):
        raise ValueError("Existing pose input paths differ from the selected or dense window")
    current_hashes = {path: sha256(path) for path in expected_paths}
    if current_hashes != saved_hashes:
        raise ValueError("Existing pose input source content changed")
    return current_hashes, input_indices


def project_actions(extrinsics, nominal_step=.25):
    """Project to camera-0 horizontal plane, then form anchor-relative SE(2)."""
    if not np.isfinite(nominal_step) or nominal_step <= 0:
        raise ValueError("Nominal step must be finite and positive")
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.shape != (20, 3, 4) or not np.isfinite(extrinsics).all():
        raise ValueError("Expected twenty finite world-to-camera matrices")
    transforms = np.tile(np.eye(4), (len(extrinsics), 1, 1))
    transforms[:, :3] = extrinsics
    cameras = np.linalg.inv(transforms)
    rotation = extrinsics[0, :3, :3]
    centers = (rotation @ (cameras[:, :3, 3] - cameras[0, :3, 3]).T).T
    forwards = np.einsum("ij,nj->ni", rotation, cameras[:, :3, 2])
    poses = np.column_stack([centers[:, 2], -centers[:, 0], np.arctan2(-forwards[:, 0], forwards[:, 2])])
    lengths = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    if not np.isfinite(lengths).all() or np.median(lengths) <= 1e-6:
        raise ValueError("Degenerate camera translations; cannot normalize")
    scale = nominal_step / float(np.median(lengths))
    poses[:, :2] *= scale
    delta = poses[4:, :2] - poses[3, :2]
    c, s = np.cos(poses[3, 2]), np.sin(poses[3, 2])
    actions = np.column_stack([c * delta[:, 0] + s * delta[:, 1],
                               -s * delta[:, 0] + c * delta[:, 1],
                               (poses[4:, 2] - poses[3, 2] + np.pi) % (2 * np.pi) - np.pi])
    diagnostics = {
        "translation_scale": scale, "median_step_model_units": float(np.median(lengths)),
        "normalized_adjacent_steps": (lengths * scale).tolist(),
        "max_step_to_median_ratio": float(lengths.max() / np.median(lengths)),
        "total_yaw_range_degrees": float(np.ptp(np.unwrap(poses[:, 2])) * 180 / np.pi),
        "endpoint": actions[-1].tolist(),
    }
    # Explicit numerical checks catch corrupt geometry; diagnostics retain all motion.
    if (np.max(np.abs(extrinsics[:, :3, :3] @ extrinsics[:, :3, :3].transpose(0, 2, 1) - np.eye(3))) > 1e-3
            or np.max(np.abs(np.linalg.det(extrinsics[:, :3, :3]) - 1)) > 1e-3):
        raise ValueError("Invalid camera rotation")
    if diagnostics["max_step_to_median_ratio"] > 10:
        raise ValueError("Pose discontinuity exceeds ten median steps")
    return actions.tolist(), diagnostics


def select_samples(dataset_root):
    specs = [("train", route, start) for route in range(1, 6) for start in (1, 81, 161, 241, 321, 401)]
    specs += [("val", 8, start) for start in (1, 201, 401)]
    specs += [("cross_scene_test", 9, start) for start in (1, 181, 361, 541, 721, 901)]
    samples = []
    for split, route, start in specs:
        directory = "testing" if route == 9 else "training"
        indices = [start + 4 * offset for offset in range(20)]
        paths = [dataset_root / "images" / directory / f"forward_{route}_{index:04d}_l.png" for index in indices]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(next(path for path in paths if not path.is_file()))
        samples.append({
            "id": f"synmars-forward-{route}-{start:04d}", "dataset": "SynMars-TW", "split": split,
            "scene": f"forward_{route}", "source_scene": f"forward_{route}", "route_id": route,
            "protocol": "direct", "frame_indices": indices, "source_frame_stride": 4,
            "context_source_paths": [str(path) for path in paths[:4]],
            "gt_source_paths": [str(path) for path in paths[4:]],
            "prediction_steps": list(range(1, 17)), "action_source": "VGGT-Omega visual pose estimates",
            "action_units": "relative translation units; yaw radians", "source_fps": None,
            "gallery_selected": route == 9 and start in (1, 361, 721),
        })
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pose-code-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--limit", type=int, help="Pilot only; incomplete outputs cannot be trained/exported")
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.pose_code_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != "282ec70363edeff59424bf43731658092fba3d37":
        raise ValueError("Unexpected VGGT-Omega code revision")
    if subprocess.check_output(["git", "-C", str(args.pose_code_root), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("VGGT-Omega source has uncommitted modifications")
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934":
        raise ValueError("Unexpected VGGT-Omega checkpoint")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    poses_dir = root / "poses"
    poses_dir.mkdir(exist_ok=True)
    if (root / "data_manifest.json").exists():
        raise ValueError("Completed data manifest already exists; use a fresh output directory")
    selected = select_samples(args.dataset_root.resolve())
    # Record selection before any model output exists.
    selection = {"samples": selected, "gallery_sample_ids": [s["id"] for s in selected if s["gallery_selected"]],
                 "split_policy": {"train_routes": [1, 2, 3, 4, 5], "validation_routes": [8], "test_routes": [9]},
                 "limitations": LIMITATIONS}
    selection_path = root / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Existing selection differs")
    write_json(selection_path, selection)
    sys.path.insert(0, str(args.pose_code_root.resolve()))
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(.85)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = VGGTOmega().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.dense_head = None
    model.aggregator.cached_layer_indices = {model.aggregator.depth - 1}
    model = model.to("cuda")
    provenance = {
        "model": "VGGT-Omega-1B-512", "checkpoint_sha256": checkpoint_hash,
        "code_revision": revision,
        "resolution": args.resolution, "resize_mode": "max_size", "camera_convention": "OpenCV world-to-camera",
        "script_sha256": sha256(__file__), "torch_version": torch.__version__,
        "precision": "official model BF16 autocast", "seed": 0, "scale_policy": "median adjacent planar displacement = 0.25 relative units per clip",
        "pose_context_policy": "Jointly estimate the selected 20 frames. If a planar step exceeds ten median steps, retry all 77 consecutive source frames in the same window, then select the original 20 poses. The same quality threshold applies to the retry.",
    }
    started = time.monotonic()
    annotated = []
    for index, sample in enumerate(selected[:args.limit]):
        paths = sample["context_source_paths"] + sample["gt_source_paths"]
        output = poses_dir / f"{sample['id']}.json"
        source_hashes = {path: sha256(path) for path in paths}
        pose_input_hashes = source_hashes
        input_indices = sample["frame_indices"]
        if output.exists():
            saved = json.loads(output.read_text())
            if saved["provenance"] != provenance or saved["source_sha256"] != source_hashes:
                raise ValueError("Existing pose provenance changed")
            pose_input_hashes, input_indices = validate_cached_pose_inputs(saved, sample)
            extrinsics = np.array(saved["extrinsics_w2c"])
            intrinsics = saved["intrinsics"]
        else:
            images = load_and_preprocess_images(paths, mode="max_size", image_resolution=args.resolution).to("cuda")
            with torch.inference_mode():
                predictions = model(images)
                e, k = encoding_to_camera(predictions["pose_enc"], images.shape[-2:])
            extrinsics = e[0].float().cpu().numpy()
            intrinsics = k[0].float().cpu().tolist()
            del images, predictions, e, k
            try:
                project_actions(extrinsics)
            except ValueError as error:
                if "discontinuity" not in str(error):
                    raise
                sparse = {"extrinsics_w2c": extrinsics.tolist(), "intrinsics": intrinsics}
                print(json.dumps({"sample": sample["id"], "event": "retry_with_consecutive_pose_frames", "reason": str(error)}), flush=True)
                input_indices = list(range(sample["frame_indices"][0], sample["frame_indices"][-1] + 1))
                dense_paths = [str(Path(paths[0]).with_name(f"forward_{sample['route_id']}_{frame:04d}_l.png")) for frame in input_indices]
                pose_input_hashes = {path: sha256(path) for path in dense_paths}
                images = load_and_preprocess_images(dense_paths, mode="max_size", image_resolution=args.resolution).to("cuda")
                with torch.inference_mode():
                    predictions = model(images)
                    e, k = encoding_to_camera(predictions["pose_enc"], images.shape[-2:])
                extrinsics = e[0, ::4].float().cpu().numpy()
                intrinsics = k[0, ::4].float().cpu().tolist()
                write_json(poses_dir / f"{sample['id']}.dense_retry.json", {
                    "sample_id": sample["id"], "provenance": provenance,
                    "sparse_attempt": sparse, "pose_input_frame_indices": input_indices,
                    "pose_input_source_sha256": pose_input_hashes,
                    "extrinsics_w2c": e[0].float().cpu().tolist(), "intrinsics": k[0].float().cpu().tolist(),
                })
                del images, predictions, e, k
            write_json(output, {"sample_id": sample["id"], "provenance": provenance,
                                "frame_indices": sample["frame_indices"], "source_sha256": source_hashes,
                                "pose_input_frame_indices": input_indices, "pose_input_source_sha256": pose_input_hashes,
                                "extrinsics_w2c": extrinsics.tolist(), "intrinsics": intrinsics})
        actions, diagnostics = project_actions(extrinsics)
        sample.update(actions=actions, pose_provenance=str(output), pose_sha256=sha256(output),
                      pose_input_source_sha256=pose_input_hashes, pose_input_frame_indices=input_indices,
                      source_sha256=source_hashes, pose_diagnostics=diagnostics)
        annotated.append(sample)
        print(json.dumps({"sample": sample["id"], "index": index + 1, "total": len(selected),
                          "elapsed_seconds": round(time.monotonic() - started, 1), **diagnostics}), flush=True)
        write_json(root / "pose_progress.json", {"provenance": provenance, "samples": annotated})
    if len(annotated) != len(selected):
        print("Pilot completed; no training manifest written", flush=True)
        return
    data = {"schema_version": 1, "completed_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "SynMars-TW", "data_metadata": {"SynMars-TW": {"protocol": "paired direct video prediction with visually estimated relative actions", "limitations": LIMITATIONS}},
            "pose_model": provenance, "split_policy": selection["split_policy"], "selection_sha256": sha256(selection_path),
            "gallery_sample_ids": selection["gallery_sample_ids"], "limitations": LIMITATIONS, "samples": annotated}
    write_json(root / "data_manifest.json", data)
    write_json(root / "eval_manifest.json", {**data, "samples": [s for s in annotated if s["split"] == "cross_scene_test"]})
    print(json.dumps({"status": "complete", "samples": len(annotated), "elapsed_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
