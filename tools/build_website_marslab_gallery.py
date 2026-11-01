#!/usr/bin/env python3
"""Publish three predeclared MarsLab ground-truth-action prediction sliders.

CPU-only export after fine-tuning and all six held-out windows have completed.
Publication verifies raw PhysX poses, disjoint sources, model/data provenance,
preprocessed reference images, and the encoded media's real-time cadence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageChops, ImageOps

from build_website_motion_gallery import SITE, encode
from build_website_mars_gallery import read_json, sha256, write_public
from prepare_marslab_manifest import planar_poses, relative_actions


EXPECTED_SPLITS = {"train": 30, "val": 3, "same_scene_test": 6}
TRAJECTORIES = {**{f"train_{i:02d}": ("train", 100) for i in range(6)},
                "val_00": ("val", 60),
                **{f"test_{i:02d}": ("same_scene_test", 40) for i in range(3)}}
SELECTED_IDS = [f"marslab-test_{i:02d}-0000" for i in range(3)]
TEST_IDS = [f"marslab-test_{i:02d}-{start:04d}" for i in range(3) for start in (0, 20)]
ACTION_SOURCE = "MarsLab / Isaac Sim PhysX body ground truth"
PROTOCOL = (
    "Each of sixteen future views is predicted directly from the same four observed RGB "
    "frames and its anchor-relative realized PhysX body displacement. The future reference "
    "images are not model context. This is action-conditioned direct prediction, not an "
    "autoregressive rollout or a navigation success evaluation."
)
SELECTION = (
    "The first 20-frame window of test_00, test_01 and test_02, fixed before training; "
    "all six held-out windows are evaluated without ranking predictions. Independent "
    "held-out trajectories share the training terrain."
)
TIMING = (
    "MarsLab plays at 1× simulation time: 4 FPS, one shared observed anchor and sixteen "
    "future samples over 4 seconds; the final frame is held for 0.25 seconds."
)


def finite_array(value, shape, description):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Expected {description}: finite array of shape {shape}")
    return array


def validate_render_warmup(metadata):
    """Require v2 resource warmup and verify its recorded physical state directly."""
    warmup = metadata.get("render_warmup")
    if (metadata.get("render_warmup_frames") != 128 or not isinstance(warmup, dict)
            or warmup.get("requested_render_frames") != 128
            or type(warmup.get("render_updates")) is not int or warmup["render_updates"] < 128
            or warmup.get("physics_state_unchanged") is not True):
        raise ValueError("Expected v2 recording with 128 render-only warmup frames")
    required = ("simulation_time_before", "simulation_time_after", "physics_step_before",
                "physics_step_after", "body_position_before_m", "body_position_after_m",
                "body_quaternion_before_wxyz", "body_quaternion_after_wxyz", "camera_timestamp")
    if not all(key in warmup for key in required):
        raise ValueError("Incomplete render warmup physical-state audit")
    times = finite_array([warmup[key] for key in (
        "simulation_time_before", "simulation_time_after", "camera_timestamp")], (3,), "warmup timestamps")
    if times[0] < 0 or not np.all(times == times[0]):
        raise ValueError("Render warmup advanced physics time or camera timestamp is not synchronized")
    steps = [warmup[key] for key in ("physics_step_before", "physics_step_after")]
    if any(type(step) is not int or step < 0 for step in steps) or steps[0] != steps[1]:
        raise ValueError("Render warmup changed its physics step")
    for field, size in (("position", 3), ("quaternion", 4)):
        suffix = "m" if field == "position" else "wxyz"
        before, after = [finite_array(warmup[f"body_{field}_{when}_{suffix}"], (size,),
                                     f"warmup body {field}") for when in ("before", "after")]
        if not np.array_equal(before, after):
            raise ValueError(f"Render warmup changed body {field}")
        if field == "quaternion" and not np.isclose(np.linalg.norm(before), 1, rtol=0, atol=1e-4):
            raise ValueError("Render warmup body quaternion is not normalized")
    public = {key: warmup[key] for key in (
        "requested_render_frames", "render_updates", *required, "physics_state_unchanged")}
    for key in ("elapsed_wall_seconds", "rendering_frame_before", "rendering_frame_after"):
        if key in warmup and warmup[key] is not None:
            value = finite_array(warmup[key], (), f"warmup {key}")
            if value < 0 or (key != "elapsed_wall_seconds" and type(warmup[key]) is not int):
                raise ValueError(f"Invalid render warmup {key}")
            public[key] = warmup[key]
    return public


def source_name(path):
    """Use only a validated dataset-relative source identity in public metadata."""
    path = Path(path)
    if (path.parent.name != "rgb" or path.parent.parent.name not in TRAJECTORIES
            or not re.fullmatch(r"\d{6}\.png", path.name)):
        raise ValueError("Unexpected MarsLab source-image identity")
    return f"MarsLab/recordings/{path.parent.parent.name}/rgb/{path.name}"


def validate_data(data):
    """Validate the frozen split and action contract without model dependencies."""
    if data.get("dataset") != "MarsLab" or data.get("schema_version") != 1:
        raise ValueError("Expected the MarsLab ground-truth manifest schema")
    samples = data["samples"]
    by_id = {sample["id"]: sample for sample in samples}
    if len(by_id) != len(samples) or Counter(s["split"] for s in samples) != EXPECTED_SPLITS:
        raise ValueError("Expected 30 training, 3 validation and 6 held-out windows with unique IDs")
    expected_ids = {f"marslab-{route}-{start:04d}" for route, (_, count) in TRAJECTORIES.items()
                    for start in range(0, count, 20)}
    if set(by_id) != expected_ids:
        raise ValueError("Window identities differ from the predeclared trajectory split")
    scale = data["translation_scale"]
    if (scale["fit_splits"] != ["train"] or scale["nominal_step"] != .25
            or not np.isfinite(scale["model_units_per_meter"]) or scale["model_units_per_meter"] <= 0
            or not np.isfinite(scale["training_median_step_m"]) or scale["training_median_step_m"] <= 0):
        raise ValueError("Expected one positive translation scale fitted only on training poses")
    sources, content_splits = set(), {}
    for sample in samples:
        route, frames = sample["trajectory_id"], sample["frame_indices"]
        split, count = TRAJECTORIES[route]
        if (sample["split"] != split or len(frames) != 20 or frames[0] % 20
                or frames != list(range(frames[0], frames[0] + 20)) or frames[-1] >= count
                or sample["id"] != f"marslab-{route}-{frames[0]:04d}"):
            raise ValueError("Sample identity, split or source window differs from the frozen plan")
        if (sample["dataset"], sample["scene"], sample["source_scene"], sample["protocol"],
                sample["prediction_steps"], sample["source_fps"], sample["action_source"]) != (
                "MarsLab", "main_crater", "main_crater", "direct", list(range(1, 17)), 4, ACTION_SOURCE):
            raise ValueError("Expected sixteen direct MarsLab horizons with simulator ground-truth actions")
        if len(sample["context_source_paths"]) != 4 or len(sample["gt_source_paths"]) != 16:
            raise ValueError("Expected four observed contexts and sixteen reference targets")
        times = finite_array(sample["timestamps"], (20,), "source timestamps")
        if not np.allclose(np.diff(times), .25, atol=1e-5, rtol=0):
            raise ValueError("Source timestamps must have a real 4 Hz cadence")
        finite_array(sample["command_vw"], (20, 2), "recorded velocity commands")
        metric = finite_array(sample["actions_meters_radians"], (16, 3), "metric ground-truth actions")
        actions = finite_array(sample["actions"], (16, 3), "training-calibrated actions")
        expected = metric * np.array([scale["model_units_per_meter"], scale["model_units_per_meter"], 1])
        if not np.allclose(actions, expected, rtol=0, atol=1e-8):
            raise ValueError("Model actions do not match metric actions and the shared training scale")
        paths = sample["context_source_paths"] + sample["gt_source_paths"]
        if set(sample["source_sha256"]) != set(paths):
            raise ValueError("Source hashes must cover exactly the twenty input images")
        for frame, path in zip(frames, paths):
            identity = source_name(path)
            if identity != f"MarsLab/recordings/{route}/rgb/{frame:06d}.png":
                raise ValueError("Source image does not match the declared trajectory/frame")
            resolved = str(Path(path).resolve())
            if resolved in sources:
                raise ValueError("An image is reused across nonoverlapping windows or splits")
            sources.add(resolved)
            digest = sample["source_sha256"][path]
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Malformed source-image SHA-256")
            if digest in content_splits and content_splits[digest] != split:
                raise ValueError("Identical source-image bytes appear in different splits")
            content_splits[digest] = split
    if (data["gallery_sample_ids"] != SELECTED_IDS
            or sorted(s["id"] for s in samples if s.get("gallery_selected")) != SELECTED_IDS):
        raise ValueError("Gallery must use the three predeclared first test windows")
    return by_id


def validate_recordings(data):
    """Recompute every action from signed raw PhysX recordings, including train."""
    provenance = data["trajectory_provenance"]
    by_route = {entry["id"]: entry for entry in provenance}
    if len(by_route) != len(provenance) or set(by_route) != set(TRAJECTORIES):
        raise ValueError("Expected all ten original trajectory records")
    trajectories, train_steps, scene_hashes = {}, [], set()
    for route, (split, count) in TRAJECTORIES.items():
        entry = by_route[route]
        path = Path(entry["trajectory_path"])
        if (sha256(path) != entry["trajectory_sha256"]
                or sha256(path.parent / "commands.jsonl") != entry["commands_sha256"]):
            raise ValueError("Raw trajectory or command-log hash mismatch")
        trajectory = read_json(path)
        raw_split = "test" if split == "same_scene_test" else split
        if (trajectory.get("status"), trajectory["trajectory_id"], trajectory["split"],
                trajectory["fps"], entry["split"], entry["frames"]) != (
                "complete", route, raw_split, 4, raw_split, count):
            raise ValueError("Recorded trajectory identity, completion or cadence mismatch")
        frames, metadata = trajectory["frames"], trajectory["metadata"]
        validate_render_warmup(metadata)
        if len(frames) != count or [f["index"] for f in frames] != list(range(count)):
            raise ValueError("Recorded frames differ from the predeclared trajectory length")
        episode = metadata["episode_spec"]
        if (episode["id"], episode["split"], round(episode["duration_s"] * 4), metadata["specs_sha256"]) != (
                route, raw_split, count, data["specs_sha256"]):
            raise ValueError("Recording differs from the signed episode specification")
        for key in ("scene_sha256", "recorder_sha256", "compatibility_sha256"):
            if metadata[key] != entry[key]:
                raise ValueError("Recorder provenance differs from the prepared manifest")
        scene_hashes.add(metadata["scene_sha256"])
        times = finite_array([f["timestamp"] for f in frames], (count,), "physics timestamps")
        camera_times = finite_array([f["camera_timestamp"] for f in frames], (count,), "camera timestamps")
        if (not np.allclose(np.diff(times), .25, rtol=0, atol=1e-5)
                or np.max(np.abs(times - camera_times)) > 1e-5):
            raise ValueError("Camera and ground-truth poses are not synchronized at 4 Hz")
        poses = planar_poses(frames)
        steps = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
        if np.median(steps) <= 1e-4 or np.max(steps) > 1:
            raise ValueError("Stationary or discontinuous ground-truth recording")
        if split == "train":
            train_steps.extend(steps.tolist())
        trajectories[route] = trajectory
    if len(scene_hashes) != 1:
        raise ValueError("Expected one shared terrain for this same-scene evaluation")
    scale = data["translation_scale"]
    if (not np.isclose(np.median(train_steps), scale["training_median_step_m"], rtol=0, atol=1e-10)
            or not np.isclose(.25 / np.median(train_steps), scale["model_units_per_meter"], rtol=0, atol=1e-8)):
        raise ValueError("Translation scale does not match training-only raw poses")
    for sample in data["samples"]:
        route = sample["trajectory_id"]
        frames = [trajectories[route]["frames"][i] for i in sample["frame_indices"]]
        parent = Path(by_route[route]["trajectory_path"]).parent
        paths = [str((parent / f["path"]).resolve()) for f in frames]
        if paths != sample["context_source_paths"] + sample["gt_source_paths"]:
            raise ValueError("Prepared source images differ from recorded frame identities")
        if ([f["timestamp"] for f in frames] != sample["timestamps"]
                or [f["command_vw"] for f in frames] != sample["command_vw"]):
            raise ValueError("Prepared timestamps or commands differ from raw ground truth")
        poses = planar_poses(frames)
        for key, factor in (("actions_meters_radians", 1), ("actions", scale["model_units_per_meter"])):
            if not np.allclose(relative_actions(poses, factor), sample[key], rtol=0, atol=1e-8):
                raise ValueError("Published actions do not match the recorded PhysX body poses")
        for path, digest in sample["source_sha256"].items():
            if sha256(path) != digest:
                raise ValueError(f"Source-image hash mismatch: {Path(path).name}")
    return trajectories


def reference_pixels(path):
    """CPU equivalent of repository CenterCropAR + PIL bilinear 224 resize."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        crop_h, crop_w = (height, int(height * 4 / 3)) if width > height else (int(width / (4 / 3)), width)
        # torchvision center_crop pads before cropping if an aspect ratio needs it.
        pad_w, pad_h = max(crop_w - width, 0), max(crop_h - height, 0)
        if pad_w or pad_h:
            image = ImageOps.expand(image, (pad_w // 2, pad_h // 2, (pad_w + 1) // 2, (pad_h + 1) // 2))
        left, top = round((image.width - crop_w) / 2), round((image.height - crop_h) / 2)
        return image.crop((left, top, left + crop_w, top + crop_h)).resize((224, 224), Image.Resampling.BILINEAR)


def validate_inference_images(inference):
    predictions, references = set(), set()
    for sample in inference["samples"]:
        if (len(sample["context_paths"]), len(sample["gt_paths"]), len(sample["prediction_paths"]), sample["fps"]) != (4, 16, 16, 4):
            raise ValueError("Each completed test window requires 4 contexts and 16 targets/predictions")
        paths = sample["context_paths"] + sample["gt_paths"]
        sources = sample["context_source_paths"] + sample["gt_source_paths"]
        for source, path in zip(sources, paths):
            with Image.open(path) as actual:
                if actual.size != (224, 224) or ImageChops.difference(reference_pixels(source), actual.convert("RGB")).getbbox():
                    raise ValueError("Inference context/reference pixels differ from the signed source images")
            references.add(str(Path(path).resolve()))
        with Image.open(sample["initial_path"]) as anchor, Image.open(sample["context_paths"][-1]) as context:
            if anchor.size != (224, 224) or ImageChops.difference(anchor.convert("RGB"), context.convert("RGB")).getbbox():
                raise ValueError("Initial frame must equal the fourth observed context frame")
        references.add(str(Path(sample["initial_path"]).resolve()))
        for path in sample["prediction_paths"]:
            resolved = str(Path(path).resolve())
            if resolved in predictions:
                raise ValueError("Each prediction must have a distinct image artifact")
            predictions.add(resolved)
            with Image.open(path) as image:
                image.load()
                if image.size != (224, 224) or image.mode != "RGB":
                    raise ValueError("Expected a decodable 224×224 RGB prediction")
    if len(predictions) != 96 or predictions & references:
        raise ValueError("Expected all 96 completed prediction images, separate from references")


def validate_inputs(direct_path, training_path, data_path):
    inference, training, data = map(read_json, (direct_path, training_path, data_path))
    by_id = validate_data(data)
    if not inference.get("completed_utc") or training.get("status") != "complete":
        raise ValueError("Only completed training and inference can be published")
    if training["data_manifest_sha256"] != sha256(data_path):
        raise ValueError("Training data-manifest hash mismatch")
    signature = inference["signature"]
    checkpoint = (training_path.parent / "checkpoints/latest.pth.tar").resolve()
    if (signature["checkpoint_sha256"] != training["checkpoint_sha256"]
            or Path(signature["checkpoint"]).resolve() != checkpoint
            or sha256(checkpoint) != training["checkpoint_sha256"]):
        raise ValueError("Inference does not use the verified trained checkpoint")
    if sha256(training["source_checkpoint"]) != training["source_checkpoint_sha256"]:
        raise ValueError("Fine-tuning source-checkpoint hash mismatch")
    if (training["settings"]["steps"], training["training_pairs"], training["validation_pairs"]) != (1000, 480, 48):
        raise ValueError("Expected 1,000 updates on 480 training and 48 validation target pairs")
    if {k: v for k, v in training["split_windows"].items() if v} != EXPECTED_SPLITS:
        raise ValueError("Training split counts disagree with the frozen data manifest")
    for split, key in (("train", "train_sample_ids"), ("val", "val_sample_ids")):
        if sorted(training[key]) != sorted(s["id"] for s in data["samples"] if s["split"] == split):
            raise ValueError("Trained sample identities disagree with the signed manifest")
    if (signature["seed"], signature["diffusion_steps"], signature["vae_encoding"],
            signature["context_size"], signature["image_size"]) != (0, 250, "posterior_mean", 4, 224):
        raise ValueError("Unexpected inference sampling protocol")
    supplied = signature["samples_manifest"]
    if sha256(supplied["path"]) != supplied["sha256"]:
        raise ValueError("Inference source-manifest hash mismatch")
    evaluation = read_json(supplied["path"])
    if any(evaluation.get(key) != value for key, value in data.items() if key != "samples"):
        raise ValueError("Evaluation and training data provenance differ")
    for samples in (inference["samples"], evaluation["samples"]):
        if sorted(s["id"] for s in samples) != TEST_IDS:
            raise ValueError("Expected completed inference on all six held-out windows")
        for sample in samples:
            if any(sample.get(key) != value for key, value in by_id[sample["id"]].items()):
                raise ValueError(f"Inference sample differs from the signed data: {sample['id']}")
    if inference.get("generated_frames") != 96:
        raise ValueError("All 96 held-out predictions must be completed")
    trajectories = validate_recordings(data)
    validate_inference_images(inference)
    return inference, training, data, trajectories


def public_pose(sample, data, trajectory):
    entry = next(p for p in data["trajectory_provenance"] if p["id"] == sample["trajectory_id"])
    metadata = trajectory["metadata"]
    return {
        "estimated": False, "source": ACTION_SOURCE, "trajectory_id": sample["trajectory_id"],
        "raw_artifact_sha256": entry["trajectory_sha256"], "commands_sha256": entry["commands_sha256"],
        "scene_sha256": entry["scene_sha256"], "recorder_sha256": entry["recorder_sha256"],
        "compatibility_sha256": entry["compatibility_sha256"], "physics_hz": trajectory["physics_hz"],
        "simulator": metadata["simulator"], "ground_truth": metadata["ground_truth"],
        "gravity_m_s2": metadata["gravity_m_s2"], "coordinate_convention": metadata["coordinate_convention"],
        "render_warmup_frames": metadata["render_warmup_frames"],
        "render_warmup": validate_render_warmup(metadata),
        "frames": [{key: trajectory["frames"][i][key] for key in (
            "index", "timestamp", "camera_timestamp", "position_m", "quaternion_wxyz", "command_vw")}
            for i in sample["frame_indices"]],
    }


def verify_video(path):
    """Decode and probe each final file, including its final held-frame duration."""
    result = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames,duration", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    if (stream["width"], stream["height"], Fraction(stream["avg_frame_rate"]), int(stream["nb_read_frames"])) != (224, 224, 4, 17):
        raise ValueError("Encoded video must be 17 frames at 4 FPS and 224×224")
    if abs(float(stream["duration"]) - 4.25) > 1e-5:
        raise ValueError("Encoded video duration must be 4.25 seconds")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True, capture_output=True)


def export_gallery(direct_path, training_path, data_path, site=SITE,
                   asset_group="marslab-finetuned"):
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", asset_group):
        raise ValueError("Asset group must be a lowercase hyphen-separated directory name")
    inference, training, data, trajectories = validate_inputs(direct_path, training_path, data_path)
    signature = inference["signature"]
    inference_hash, metrics_hash = sha256(direct_path), sha256(training_path)
    catalog_path = site / "content/demos.json"
    catalog = read_json(catalog_path)
    original = [item for item in catalog["demos"] if item.get("environment") != "mars"]
    if len(original) != 12 or len({item["id"] for item in original}) != 12:
        raise ValueError("Expected to preserve exactly twelve distinct non-Mars gallery examples")
    by_id = {sample["id"]: sample for sample in inference["samples"]}
    additions = []
    with tempfile.TemporaryDirectory(prefix="marslab-website-", dir=direct_path.parent) as temporary:
        temporary = Path(temporary)
        for index, identity in enumerate(SELECTED_IDS, 1):
            sample = by_id[identity]
            directory = temporary / identity
            directory.mkdir()
            prefix = f"assets/space/{asset_group}/{identity}"
            times = sample["timestamps"][3:]
            offsets = [float(t - times[0]) for t in times]
            poses = public_pose(sample, data, trajectories[sample["trajectory_id"]])
            public = {
                "schema_version": 1, "model": "OpenNWM · MarsLab few-shot fine-tuned EMA",
                "checkpoint_sha256": signature["checkpoint_sha256"],
                "source_checkpoint_sha256": training["source_checkpoint_sha256"],
                "sample_id": identity, "dataset": "MarsLab", "source_scene": sample["source_scene"],
                "trajectory_id": sample["trajectory_id"], "split": sample["split"], "selection": SELECTION,
                "specs_sha256": data["specs_sha256"], "preparation_sha256": data["preparation_sha256"],
                "frame_indices": sample["frame_indices"], "source_frame_stride": 1,
                "source_anchor_frame": sample["frame_indices"][3],
                "context_source_paths": [source_name(path) for path in sample["context_source_paths"]],
                "gt_source_paths": [source_name(path) for path in sample["gt_source_paths"]],
                "source_sha256": {source_name(path): digest for path, digest in sample["source_sha256"].items()},
                "prediction_steps": sample["prediction_steps"], "actions": sample["actions"],
                "actions_meters_radians": sample["actions_meters_radians"], "translation_scale": data["translation_scale"],
                "action_translation_units": "Model inputs: shared training-calibrated units. Plotted path: meters.",
                "action_yaw_units": "Radians, positive left", "action_source": ACTION_SOURCE,
                "action_reference": "Realized target body pose relative to the fourth observed body pose, projected to SE(2).",
                "ground_truth_poses": poses, "protocol": PROTOCOL, "limitations": data["limitations"],
                "playback_fps": 4, "frame_count": 17, "frame_interpolation": False,
                "source_timestamps_available": True, "source_fps": 4, "playback_speed": 1,
                "source_timestamps_seconds": times, "sample_times_seconds": offsets,
                "source_span_seconds": offsets[-1], "media_duration_seconds": 4.25, "timing": TIMING,
                "sampling": {key: signature[key] for key in ("seed", "diffusion_steps", "vae_encoding", "image_size", "context_size")},
                "inference_manifest_sha256": inference_hash, "training_metrics_sha256": metrics_hash,
                "training": {"settings": training["settings"], "windows": 30, "target_pairs": 480,
                             "scenes": ["main_crater"], "trajectory_ids": [f"train_{i:02d}" for i in range(6)],
                             "split_windows": EXPECTED_SPLITS, "split_policy": data["split_policy"],
                             "data_manifest_sha256": training["data_manifest_sha256"]},
                "evaluation": {"windows": 6, "generated_frames": 96, "gallery_sample_ids": SELECTED_IDS},
                "asset_provenance": {},
            }
            item = {
                "id": f"{asset_group}-{identity}", "title": f"Mars crater · route {index}",
                "titleZh": f"Mars crater · route {index}", "dataset": "MarsLab", "datasetId": "marslab",
                "domain": "held-out trajectory · same terrain", "environment": "mars", "fineTuned": True,
                "sampleId": identity, "fps": 4, "duration": 4.25, "frameCount": 17,
                "width": 224, "height": 224, "includesInitialFrame": True, "posterTime": 1,
                "labels": {"gt": "Simulation reference", "prediction": "OpenNWM"},
                "model": public["model"], "checkpoint": {"sha256": public["checkpoint_sha256"]},
                "sourceManifest": f"{prefix}/manifest.json", "sourceInferenceManifestSha256": inference_hash,
                "protocol": PROTOCOL, "selection": SELECTION, "limitations": data["limitations"],
                "note": "Held-out trajectories on the same terrain, with simulator ground-truth poses. Fine-tuned on 30 clips for 1,000 updates; playback at 1× simulation time.",
                "noteZh": "Held-out trajectories on the same terrain, with simulator ground-truth poses. Fine-tuned on 30 clips for 1,000 updates; playback at 1× simulation time.",
                "training": {"steps": 1000, "windows": 30, "targetPairs": 480, "scenes": ["main_crater"]},
                "conditioningTrajectory": {
                    "estimated": False, "positions": [[0, 0]] + [a[:2] for a in sample["actions_meters_radians"]],
                    "sampleTimesSeconds": offsets, "axes": "Forward and left displacement in meters, relative to the observed anchor.",
                    "timeSemantics": "Offsets from synchronized simulator camera/PhysX timestamps; 1× simulation time.",
                    "sourceFile": f"{prefix}/manifest.json", "poseSource": ACTION_SOURCE,
                    "poseSourceSha256": poses["raw_artifact_sha256"],
                }, "assetProvenance": {},
            }
            for key, paths_key in (("gt", "gt_paths"), ("prediction", "prediction_paths")):
                target = directory / f"slider-{key}.mp4"
                provenance = encode([Path(sample["initial_path"])] + [Path(path) for path in sample[paths_key]], target, 4)
                verify_video(target)
                public["asset_provenance"][key] = item["assetProvenance"][key] = provenance
                item[key], item[f"{key}Poster"] = f"{prefix}/{target.name}", f"{prefix}/{target.with_suffix('.webp').name}"
            public_path = directory / "manifest.json"
            write_public(public_path, public)
            item["sourceManifestSha256"] = item["conditioningTrajectory"]["sourceSha256"] = sha256(public_path)
            item["poster"] = item["gtPoster"]
            additions.append(item)
        catalog["demos"] = original + additions
        obsolete = "SynMars source timing is unknown; its 17-frame media plays at 4 FPS for display only."
        timing = catalog.get("timingNote", "").replace(obsolete, "").strip()
        catalog["timingNote"] = timing if TIMING in timing else f"{timing} {TIMING}".strip()
        staged_catalog = temporary / "demos.json"
        write_public(staged_catalog, catalog)
        # Publish only after every input, image, video and public JSON has passed.
        for identity in SELECTED_IDS:
            destination = site / "assets/space" / asset_group / identity
            destination.mkdir(parents=True, exist_ok=True)
            for asset in (temporary / identity).iterdir():
                shutil.copyfile(asset, destination / asset.name)
        staged_catalog_copy = catalog_path.with_suffix(".json.tmp")
        shutil.copyfile(staged_catalog, staged_catalog_copy)
        staged_catalog_copy.replace(catalog_path)
    print(f"Published {len(additions)} MarsLab examples at 1×; preserved {len(original)} existing examples.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--site", type=Path, default=SITE)
    parser.add_argument("--asset-group", default="marslab-finetuned",
                        help="Versioned directory under assets/space; existing default is preserved.")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    paths = [path.resolve() for path in (args.direct_manifest, args.training_metrics, args.data_manifest)]
    if args.validate_only:
        validate_inputs(*paths)
        print("Validated all 39 windows, raw PhysX actions, checkpoint provenance and 96 held-out predictions.")
    else:
        export_gallery(*paths, site=args.site.resolve(), asset_group=args.asset_group)


if __name__ == "__main__":
    main()
