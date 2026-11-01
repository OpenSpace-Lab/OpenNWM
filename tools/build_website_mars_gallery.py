#!/usr/bin/env python3
"""Export three predeclared, held-out SynMars prediction/reference sliders.

CPU-only export of completed inference. Camera poses are visual estimates made
offline from each full clip; their relative actions are oracle annotations, not
recorded controls, metric trajectories, or predictions of future actions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageChops

from build_website_motion_gallery import SITE, encode
from prepare_synmars_pose_actions import project_actions


SELECTED_IDS = [f"synmars-forward-9-{start:04d}" for start in (1, 361, 721)]
TEST_IDS = [f"synmars-forward-9-{start:04d}" for start in (1, 181, 361, 541, 721, 901)]
EXPECTED_SPLITS = {"train": 30, "val": 3, "cross_scene_test": 6}
PROTOCOL = (
    "Each of 16 future views is predicted directly from the same four observed RGB frames "
    "and its anchor-relative visually estimated action. Future reference RGB is used only "
    "offline to annotate these oracle actions, never as image-model context."
)
LIMITATIONS = (
    "Actions are visual pose estimates, not recorded controls or ground-truth poses. "
    "Reference future RGB informs the offline oracle action annotation; pose estimation can "
    "also use intermediate RGB within the same source-frame interval. Monocular translation "
    "is normalized per clip to a median adjacent step of 0.25 relative units, not meters. "
    "Source timestamps and physical speed are unknown; 4 FPS is playback speed only. "
    "Held-out route IDs do not establish geographically separate terrain. Direct predictions "
    "are not an autoregressive rollout or evidence of physical navigation performance."
)
SELECTION = (
    "Predeclared test windows beginning at source frames 1, 361 and 721 of forward_9; "
    "selected before pose estimation, fine-tuning and inference, without ranking predictions."
)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_public(path, value):
    """Reject local absolute paths anywhere in public provenance."""
    def check(item):
        if isinstance(item, dict):
            for key, child in item.items():
                check(key)
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif isinstance(item, str) and (item.startswith("/") or "/file_system/" in item):
            raise ValueError("Public provenance contains a machine-local path")
    check(value)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_name(path):
    """Publish dataset-relative source identity, without machine paths."""
    path = Path(path)
    if path.parent.parent.name != "images" or path.parent.name not in {"training", "testing"}:
        raise ValueError(f"Unexpected SynMars image location: {path.name}")
    return f"SynMars-TW/images/{path.parent.name}/{path.name}"


def validate_data(data):
    if not data.get("completed_utc"):
        raise ValueError("Pose/action annotation has not completed")
    samples = data["samples"]
    by_id = {sample["id"]: sample for sample in samples}
    if len(by_id) != len(samples) or Counter(sample["split"] for sample in samples) != EXPECTED_SPLITS:
        raise ValueError("Expected 30 training, 3 validation and 6 held-out windows with unique IDs")
    routes = defaultdict(set)
    sources = {}
    for sample in samples:
        match = re.fullmatch(r"forward_(\d+)", sample["source_scene"])
        if not match:
            raise ValueError("Expected a named forward route")
        route = int(match[1])
        routes[sample["split"]].add(route)
        frames = sample["frame_indices"]
        if len(frames) != 20 or frames != list(range(frames[0], frames[0] + 80, 4)):
            raise ValueError("Each sample must contain 20 source images at stride four")
        if sample["id"] != f"synmars-forward-{route}-{frames[0]:04d}":
            raise ValueError("Sample identity does not match its source window")
        if (sample["dataset"], sample["protocol"], sample["prediction_steps"]) != (
                "SynMars-TW", "direct", list(range(1, 17))):
            raise ValueError("Expected sixteen direct SynMars prediction horizons")
        if len(sample["context_source_paths"]) != 4 or len(sample["gt_source_paths"]) != 16:
            raise ValueError("Expected four observed contexts and sixteen reference targets")
        actions = np.asarray(sample["actions"])
        if actions.shape != (16, 3) or not np.isfinite(actions).all():
            raise ValueError("Expected sixteen finite planar actions")
        paths = sample["context_source_paths"] + sample["gt_source_paths"]
        if set(sample["source_sha256"]) != set(paths):
            raise ValueError("Source hashes must cover all twenty input images")
        pose_frames = sample["pose_input_frame_indices"]
        dense_frames = list(range(frames[0], frames[-1] + 1))
        if pose_frames != frames and pose_frames != dense_frames:
            raise ValueError("Pose inputs must be the selected twenty frames or all 77 frames in the same window")
        pose_paths = [str(Path(paths[0]).with_name(f"forward_{route}_{frame:04d}_l.png"))
                      for frame in pose_frames]
        if set(sample["pose_input_source_sha256"]) != set(pose_paths):
            raise ValueError("Pose input hashes must match the declared route and frame interval")
        if any(sample["pose_input_source_sha256"].get(path) != digest
               for path, digest in sample["source_sha256"].items()):
            raise ValueError("Selected image hashes must be an unchanged subset of pose input hashes")
        for frame, path in zip(frames, paths):
            expected_name = f"forward_{route}_{frame:04d}_l.png"
            expected_split = "testing" if route == 9 else "training"
            if Path(path).name != expected_name or Path(path).parent.name != expected_split:
                raise ValueError("Source image does not match declared route/frame")
            source_name(path)
        for path in pose_paths:
            source_name(path)
            identity = str(Path(path).resolve())
            if identity in sources and sources[identity] != sample["split"]:
                raise ValueError("Source image overlaps different evaluation splits")
            sources[identity] = sample["split"]
    if (routes["train"] & routes["val"] or routes["train"] & routes["cross_scene_test"]
            or routes["val"] & routes["cross_scene_test"]):
        raise ValueError("Route numbers must be disjoint across train, validation and test")
    if dict(routes) != {"train": {1, 2, 3, 4, 5}, "val": {8}, "cross_scene_test": {9}}:
        raise ValueError("Route split differs from the predeclared few-shot protocol")
    if Counter(s["source_scene"] for s in samples if s["split"] == "train") != {
            f"forward_{route}": 6 for route in range(1, 6)}:
        raise ValueError("Expected six training windows per route")
    if sorted(s["id"] for s in samples if s["split"] == "cross_scene_test") != TEST_IDS:
        raise ValueError("Held-out window selection differs from the predeclared protocol")
    if data["gallery_sample_ids"] != SELECTED_IDS or sorted(
            s["id"] for s in samples if s.get("gallery_selected")) != SELECTED_IDS:
        raise ValueError("Gallery must use exactly the three predeclared test windows")
    return by_id


def validate_inputs(direct_path, training_path, data_path):
    inference, training, data = map(read_json, (direct_path, training_path, data_path))
    by_id = validate_data(data)
    if not inference.get("completed_utc") or training.get("status") != "complete":
        raise ValueError("Only completed training and inference can be published")
    if training["data_manifest_sha256"] != sha256(data_path):
        raise ValueError("Training data manifest hash mismatch")
    signature = inference["signature"]
    checkpoint_hash = training["checkpoint_sha256"]
    if signature["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Inference and training checkpoint hashes differ")
    checkpoint = (training_path.parent / "checkpoints/latest.pth.tar").resolve()
    if Path(signature["checkpoint"]).resolve() != checkpoint or sha256(checkpoint) != checkpoint_hash:
        raise ValueError("Inference does not use the verified trained checkpoint")
    if (training["settings"]["steps"], training["training_pairs"], training["validation_pairs"]) != (1000, 480, 48):
        raise ValueError("Expected 1,000 updates on 480 training and 48 validation target pairs")
    for split, count in EXPECTED_SPLITS.items():
        if training["split_windows"][split] != count:
            raise ValueError("Training split counts disagree with data manifest")
    for split, key in (("train", "train_sample_ids"), ("val", "val_sample_ids")):
        if sorted(training[key]) != sorted(s["id"] for s in data["samples"] if s["split"] == split):
            raise ValueError("Trained sample identities disagree with data manifest")
    if (signature["seed"], signature["diffusion_steps"], signature["vae_encoding"],
            signature["context_size"], signature["image_size"]) != (0, 250, "posterior_mean", 4, 224):
        raise ValueError("Unexpected inference sampling protocol")
    selection_path = data_path.parent / "selection.json"
    if sha256(selection_path) != data["selection_sha256"]:
        raise ValueError("Predeclared selection hash mismatch")
    selection = read_json(selection_path)
    if selection["gallery_sample_ids"] != SELECTED_IDS:
        raise ValueError("Predeclared gallery selection differs")
    if sorted(s["id"] for s in selection["samples"]) != sorted(by_id):
        raise ValueError("Data window identities differ from the predeclared selection")
    for sample in selection["samples"]:
        if any(by_id[sample["id"]].get(key) != value for key, value in sample.items()):
            raise ValueError("Data window differs from its predeclared source selection")
    supplied = signature["samples_manifest"]
    if sha256(supplied["path"]) != supplied["sha256"]:
        raise ValueError("Inference source manifest hash mismatch")
    eval_samples = read_json(supplied["path"])["samples"]
    samples = inference["samples"]
    if sorted(s["id"] for s in samples) != TEST_IDS or sorted(s["id"] for s in eval_samples) != TEST_IDS:
        raise ValueError("Expected completed inference on all six held-out windows")
    for sample in samples + eval_samples:
        original = by_id[sample["id"]]
        if any(sample.get(key) != value for key, value in original.items()):
            raise ValueError(f"Inference sample differs from the signed data: {sample['id']}")
    if inference.get("generated_frames") != 96:
        raise ValueError("All 96 held-out predictions must be completed")
    # Verify all data, not only the displayed subset, before publication.
    for sample in data["samples"]:
        for path, digest in sample["pose_input_source_sha256"].items():
            if sha256(path) != digest:
                raise ValueError(f"Source image hash mismatch: {Path(path).name}")
        if sha256(sample["pose_provenance"]) != sample["pose_sha256"]:
            raise ValueError("Estimated pose artifact hash mismatch")
        public_pose(sample, data["pose_model"])
    return inference, training, data


def public_pose(sample, pose_model):
    pose = read_json(sample["pose_provenance"])
    if (pose["sample_id"] != sample["id"] or pose["frame_indices"] != sample["frame_indices"]
            or pose["source_sha256"] != sample["source_sha256"] or pose["provenance"] != pose_model
            or pose["pose_input_frame_indices"] != sample["pose_input_frame_indices"]
            or pose["pose_input_source_sha256"] != sample["pose_input_source_sha256"]):
        raise ValueError("Estimated pose source does not match the sample")
    actions, diagnostics = project_actions(pose["extrinsics_w2c"])
    if not np.allclose(actions, sample["actions"], rtol=0, atol=1e-8):
        raise ValueError("Published actions do not match the estimated camera poses")
    intrinsics = np.asarray(pose["intrinsics"])
    if intrinsics.shape != (20, 3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("Expected twenty finite estimated intrinsic matrices")
    public = {"provenance": pose["provenance"], "raw_artifact_sha256": sample["pose_sha256"],
              "frame_indices": pose["frame_indices"], "extrinsics_w2c": pose["extrinsics_w2c"],
              "intrinsics": pose["intrinsics"], "diagnostics": diagnostics,
              "pose_input_frame_indices": pose["pose_input_frame_indices"],
              "pose_input_source_sha256": {source_name(path): digest
                                            for path, digest in pose["pose_input_source_sha256"].items()},
              "intrinsics_note": "Model estimates at pose-estimation preprocessing resolution, not calibrated source-camera intrinsics."}
    if pose["pose_input_frame_indices"] != pose["frame_indices"]:
        dense_path = Path(sample["pose_provenance"]).with_suffix(".dense_retry.json")
        dense = read_json(dense_path)
        if (dense["sample_id"] != sample["id"] or dense["provenance"] != pose_model
                or dense["pose_input_frame_indices"] != pose["pose_input_frame_indices"]
                or dense["pose_input_source_sha256"] != pose["pose_input_source_sha256"]):
            raise ValueError("Dense retry provenance differs from selected pose provenance")
        for key, shape in (("extrinsics_w2c", (77, 3, 4)), ("intrinsics", (77, 3, 3))):
            values = np.asarray(dense[key])
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError("Dense retry must contain 77 finite camera estimates")
            if not np.array_equal(values[::4], np.asarray(pose[key])):
                raise ValueError("Selected poses do not match the dense retry estimates")
        public["dense_retry"] = {
            "raw_artifact_sha256": sha256(dense_path), "pose_input_frame_count": 77,
            "selected_indices": list(range(0, 77, 4)),
            "note": "A retry used every source frame in this window. Published actions use only the original twenty selected estimates; these remain uncertain visual annotations.",
        }
    return public


def export_gallery(direct_path, training_path, data_path, site=SITE):
    inference, training, data = validate_inputs(direct_path, training_path, data_path)
    inference_hash, metrics_hash = sha256(direct_path), sha256(training_path)
    signature = inference["signature"]
    catalog_path = site / "content/demos.json"
    catalog = read_json(catalog_path)
    original = [item for item in catalog["demos"] if item.get("environment") != "mars"]
    additions = []
    by_id = {sample["id"]: sample for sample in inference["samples"]}
    with tempfile.TemporaryDirectory(prefix="mars-website-", dir=direct_path.parent) as temporary:
        temporary = Path(temporary)
        for index, identity in enumerate(SELECTED_IDS, 1):
            sample = by_id[identity]
            directory = temporary / identity
            directory.mkdir()
            prefix = f"assets/space/synmars-finetuned/{identity}"
            initial = Path(sample["initial_path"])
            with Image.open(initial) as anchor, Image.open(sample["context_paths"][-1]) as context:
                if ImageChops.difference(anchor.convert("RGB"), context.convert("RGB")).getbbox():
                    raise ValueError("Initial frame must equal the fourth observed context frame")
            public = {
                "schema_version": 1, "model": "OpenNWM · SynMars few-shot fine-tuned EMA",
                "checkpoint_sha256": signature["checkpoint_sha256"],
                "source_checkpoint_sha256": training["source_checkpoint_sha256"],
                "sample_id": identity, "dataset": "SynMars-TW", "source_scene": sample["source_scene"],
                "split": sample["split"], "selection": SELECTION,
                "selection_sha256": data["selection_sha256"], "frame_indices": sample["frame_indices"],
                "source_frame_stride": 4, "source_anchor_frame": sample["frame_indices"][3],
                "context_source_paths": [source_name(path) for path in sample["context_source_paths"]],
                "gt_source_paths": [source_name(path) for path in sample["gt_source_paths"]],
                "source_sha256": {source_name(path): digest for path, digest in sample["source_sha256"].items()},
                "prediction_steps": sample["prediction_steps"], "actions": sample["actions"],
                "action_translation_units": "Relative units; per-clip median adjacent planar step = 0.25. Not meters.",
                "action_yaw_units": "Radians, positive left",
                "action_reference": "Target camera pose relative to the fourth observed camera pose, projected to a plane.",
                "estimated_poses": public_pose(sample, data["pose_model"]),
                "protocol": PROTOCOL, "limitations": LIMITATIONS,
                "playback_fps": 4, "frame_count": 17, "frame_interpolation": False,
                "source_timestamps_available": False,
                "timing": "Shared observed anchor plus sixteen direct targets; 4.25 seconds of media is presentation duration only.",
                "sampling": {key: signature[key] for key in ("seed", "diffusion_steps", "vae_encoding", "image_size", "context_size")},
                "inference_manifest_sha256": inference_hash, "training_metrics_sha256": metrics_hash,
                "training": {"settings": training["settings"], "windows": 30, "target_pairs": 480,
                             "scenes": [f"forward_{route}" for route in range(1, 6)],
                             "split_windows": EXPECTED_SPLITS, "split_policy": data["split_policy"],
                             "data_manifest_sha256": training["data_manifest_sha256"]},
                "asset_provenance": {},
            }
            item = {
                "id": f"synmars-finetuned-{identity}", "title": f"Mars route 9 · clip {index}",
                "titleZh": f"Mars route 9 · clip {index}", "dataset": "SynMars-TW", "datasetId": "synmars",
                "domain": "held-out route", "environment": "mars", "fineTuned": True,
                "sampleId": identity, "fps": 4, "duration": 4.25, "frameCount": 17,
                "width": 224, "height": 224, "includesInitialFrame": True, "posterTime": 1,
                "labels": {"gt": "Simulation reference", "prediction": "OpenNWM"},
                "model": public["model"], "checkpoint": {"sha256": public["checkpoint_sha256"]},
                "sourceManifest": f"{prefix}/manifest.json", "sourceInferenceManifestSha256": inference_hash,
                "protocol": PROTOCOL, "selection": SELECTION, "limitations": LIMITATIONS,
                "training": {"steps": training["settings"]["steps"], "windows": 30,
                             "targetPairs": 480, "scenes": public["training"]["scenes"]},
                "conditioningTrajectory": {
                    "estimated": True, "positions": [[0, 0]] + [action[:2] for action in sample["actions"]],
                    "sampleTimesSeconds": [frame / 4 for frame in range(17)],
                    "axes": "Forward and left in relative action units; monocular estimated scale, not meters.",
                    "timeSemantics": "Playback offsets only. Source capture times and physical duration are unavailable.",
                    "sourceFile": f"{prefix}/manifest.json", "poseSource": "VGGT-Omega offline visual pose estimates",
                    "poseSourceSha256": sample["pose_sha256"],
                }, "assetProvenance": {},
            }
            for key, paths_key in (("gt", "gt_paths"), ("prediction", "prediction_paths")):
                future = [Path(path) for path in sample[paths_key]]
                if len(future) != 16:
                    raise ValueError("All sixteen target frames are required")
                target = directory / f"slider-{key}.mp4"
                provenance = encode([initial] + future, target, 4)
                public["asset_provenance"][key] = item["assetProvenance"][key] = provenance
                item[key] = f"{prefix}/{target.name}"
                item[f"{key}Poster"] = f"{prefix}/{target.with_suffix('.webp').name}"
            public_path = directory / "manifest.json"
            write_public(public_path, public)
            item["sourceManifestSha256"] = sha256(public_path)
            item["conditioningTrajectory"]["sourceSha256"] = sha256(public_path)
            item["poster"] = item["gtPoster"]
            additions.append(item)
        # Publish only after all three samples validate and encode successfully.
        for identity in SELECTED_IDS:
            destination = site / "assets/space/synmars-finetuned" / identity
            destination.mkdir(parents=True, exist_ok=True)
            for asset in (temporary / identity).iterdir():
                shutil.copyfile(asset, destination / asset.name)
    catalog["demos"] = original + additions
    timing = "SynMars source timing is unknown; its 17-frame media plays at 4 FPS for display only."
    if timing not in catalog.get("timingNote", ""):
        catalog["timingNote"] = (catalog.get("timingNote", "") + " " + timing).strip()
    write_public(catalog_path, catalog)
    print(f"Published {len(additions)} Mars examples; preserved {len(original)} existing examples.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    args = parser.parse_args()
    export_gallery(args.direct_manifest.resolve(), args.training_metrics.resolve(), args.data_manifest.resolve())


if __name__ == "__main__":
    main()
