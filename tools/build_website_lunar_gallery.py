"""Add held-out LuSNAR GT/OpenNWM sliders from completed direct inference.

CPU-only export. Each stream has the shared context anchor followed by all
sixteen paired target frames. No inference, interpolation, or synthetic path.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageChops

from build_website_motion_gallery import SITE, encode
from build_website_space import finetuned_samples


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_public(path, value):
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if "/file_system/" in serialized or '"/tmp/' in serialized:
        raise ValueError("Public gallery must not contain machine-local paths")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized)


def direct_only_sources(inference, inference_hash, training_path, site, asset_prefix):
    """Audit a new GT-selected manifest without borrowing old window00 provenance."""
    import numpy as np
    from planetary_eval_data import _read_lusnar_scene, local_action

    training_data = read_json(training_path)
    training = read_json(Path(inference["checkpoint"]).parent.parent / "metrics.json")
    metadata = inference["data_metadata"]
    if (training.get("status") != "complete"
            or sha256(training_path) != training["data_manifest_sha256"]
            or sha256(training_path) != metadata["training_data_manifest_sha256"]
            or training["checkpoint_sha256"] != inference["signature"]["checkpoint_sha256"]):
        raise ValueError("Training/checkpoint provenance mismatch")
    if (training["settings"]["steps"], training["split_windows"]["train"], training["training_pairs"]) != (1000, 30, 480):
        raise ValueError("Expected original LuSNAR fine-tuning protocol")
    signature = inference["signature"]
    if tuple(signature[key] for key in ("seed", "diffusion_steps", "vae_encoding", "context_size", "image_size")) != (0, 250, "posterior_mean", 4, 224):
        raise ValueError("Unexpected direct inference protocol")
    root = Path(training_data["metadata"]["lusnar_root"])
    spacing = training_data["metadata"]["waypoint_spacing_meters"]
    scenes = sorted({sample["scene"] for sample in training_data["samples"] if sample["split"] == "train"})
    public_name = lambda value: "LuSNAR/" + Path(value).relative_to(root).as_posix()
    result, audits = [], []
    for number in (7, 8, 9):
        scene = f"Moon_{number}"
        matches = [sample for sample in inference["samples"] if sample.get("dataset") == "LuSNAR"
                   and sample.get("scene") == scene and sample.get("split") == "cross_scene_test"
                   and sample.get("protocol") == "direct"]
        if len(matches) != 1 or scene in scenes:
            raise ValueError(f"Expected exactly one held-out direct sample for {scene}")
        sample = matches[0]
        if sample["prediction_steps"] != list(range(1, 17)) or sample["waypoint_spacing_meters"] != spacing:
            raise ValueError("All sixteen horizons and unchanged training-only action scale are required")
        paths, timestamps, positions, yaw, grid, indices, errors, _, _ = _read_lusnar_scene(root, scene, 4)
        anchor = sample["source_grid_anchor_index"]
        selected = metadata["selected_windows"][scene]
        if anchor != selected["source_grid_anchor_index"]:
            raise ValueError("Selected grid anchor differs from frozen selection")
        window = indices[anchor - 3:anchor + 17]
        if len(window) != 20 or len(set(window)) != 20 or max(errors[anchor - 3:anchor + 17]) > 125_000_000:
            raise ValueError("Invalid nearest-source 4 Hz window")
        exclusions = training_data["metadata"]["previous_evaluation_exclusion_ranges"][scene]
        if selected["previous_evaluation_exclusion_ranges"] != exclusions or any(
                anchor - 3 <= span["excluded_end_grid_index"] and anchor + 16 >= span["excluded_start_grid_index"]
                for span in exclusions):
            raise ValueError("Previous evaluation exclusion buffer is not preserved")
        context, future = window[:4], window[4:]
        for key, indexes in (("context", context), ("gt", future)):
            if sample[f"{key}_source_paths"] != [str(paths[index]) for index in indexes]:
                raise ValueError(f"Source paths do not match recorded {key} window")
            if sample[f"{key}_timestamps_ns"] != [int(timestamps[index]) for index in indexes]:
                raise ValueError(f"Recorded {key} timestamps do not match")
        start = indices[anchor]
        expected = [local_action(positions[start], yaw[start], positions[index], yaw[index], 1.0) for index in future]
        normalized = [[forward / spacing, left / spacing, angle] for forward, left, angle in expected]
        offsets = [(int(timestamps[index]) - int(timestamps[start])) / 1e9 for index in future]
        if not np.allclose(sample["metric_actions"], expected, atol=1e-9, rtol=0) or not np.allclose(
                sample["actions"], normalized, atol=1e-9, rtol=0) or offsets != sample["actual_prediction_offsets_seconds"]:
            raise ValueError("Actions or offsets differ from recorded GT poses")
        pose = Path(sample["pose_provenance"])
        annotations = {str(pose): sha256(pose), str(root / scene / "image0/color_timestamp.txt"):
                       sha256(root / scene / "image0/color_timestamp.txt")}
        if any(metadata["source_annotation_sha256"][path] != digest for path, digest in annotations.items()):
            raise ValueError("Source annotation hash mismatch")
        prefix = f"{asset_prefix}/moon-{number}"
        public = {
            "schema_version": 2, "model": "OpenNWM Nav15 Stage 2, LuSNAR fine-tuned EMA",
            "checkpoint_sha256": signature["checkpoint_sha256"],
            "source_checkpoint_sha256": training["source_checkpoint_sha256"],
            "training": {"settings": training["settings"], "windows": training["split_windows"]["train"],
                         "target_pairs": training["training_pairs"], "scenes": scenes,
                         "data_manifest_sha256": training["data_manifest_sha256"]},
            "sample_id": sample["id"], "dataset": "LuSNAR", "scene": scene,
            "split": sample["split"], "window": sample["window"], "selection": sample["selection"],
            "selection_audit": selected, "curvature_statistics": sample["curvature_statistics"],
            "source_samples_manifest_sha256": signature["samples_manifest"]["sha256"],
            "inference_code_sha256": signature["source_sha256"],
            "context_source_paths": list(map(public_name, sample["context_source_paths"])),
            "gt_source_paths": list(map(public_name, sample["gt_source_paths"])),
            "pose_source": public_name(pose), "pose_source_sha256": annotations[str(pose)],
            "source_annotation_sha256": {public_name(path): digest for path, digest in annotations.items()},
            "source_sha256": {public_name(path): digest for path, digest in sample["source_sha256"].items()},
            "playback_fps": 4, "frame_interpolation": False,
            "timing": "Nearest-source 4 Hz sampling; original nonuniform timestamps retained; shared anchor plus sixteen targets.",
            "sampling": {key: signature[key] for key in ("seed", "diffusion_steps", "vae_encoding", "image_size", "context_size")},
            "direct": {"protocol": "Each horizon is predicted from the same four observed context frames and anchor-relative target body pose.",
                       "actions": sample["actions"], "metric_actions": sample["metric_actions"],
                       "inference_manifest_sha256": inference_hash},
            "limitations": inference.get("limitations", []) + ["Ground-truth-curvature-selected qualitative examples; not a random benchmark or an accuracy improvement claim. No extra fine-tuning or prediction-based sample selection."],
        }
        for key in ("context_timestamps_ns", "gt_timestamps_ns", "actual_prediction_offsets_seconds",
                    "prediction_steps", "source_grid_anchor_index", "source_anchor_index", "anchor_timestamp_ns",
                    "waypoint_spacing_meters", "action_translation_units", "action_yaw_units", "coordinate_convention"):
            public[key] = sample[key]
        write_public(site / prefix / "manifest.json", public)
        audits.append({"scene": scene, "sample_id": sample["id"], "pose_source": public_name(pose),
                       "pose_source_sha256": annotations[str(pose)], "source_grid_anchor_index": anchor,
                       "validation": "All sixteen metric and normalized actions recomputed from recorded GT; timestamps, unique source frames, annotation hashes and prior exclusion buffer verified."})
        result.append(sample)
    write_public(site / asset_prefix / "trajectory-provenance.json", {
        "schema_version": 1, "source_inference_manifest_sha256": inference_hash,
        "selection": metadata["selection"], "scenes": audits,
        "display": "Anchor [0,0] followed by sixteen recorded body-frame metric XY positions. No trajectory synthesis or interpolation."})
    return result, {"scenes": audits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--site-root", type=Path, default=SITE)
    parser.add_argument("--items-output", type=Path, help="Write lunar items only; leave the shared demos catalog untouched")
    parser.add_argument("--direct-only", action="store_true", help="Audit/export new GT-selected direct windows with independent provenance")
    parser.add_argument("--training-data-manifest", type=Path)
    parser.add_argument("--asset-prefix", default="assets/space/lusnar-finetuned")
    args = parser.parse_args()
    prefix_path = Path(args.asset_prefix)
    if prefix_path.is_absolute() or ".." in prefix_path.parts:
        parser.error("--asset-prefix must remain inside --site-root")
    if args.direct_only and (not args.training_data_manifest or args.asset_prefix == "assets/space/lusnar-finetuned"):
        parser.error("--direct-only requires --training-data-manifest and an independent --asset-prefix")
    if not args.direct_only and args.asset_prefix != "assets/space/lusnar-finetuned":
        parser.error("A custom --asset-prefix requires --direct-only")
    inference = read_json(args.direct_manifest)
    if not inference.get("completed_utc"):
        raise ValueError("Only completed inference may be exported")
    inference_hash = sha256(args.direct_manifest)
    catalog_path = args.site_root / "content/demos.json"
    if args.items_output:
        catalog, original = None, []
    else:
        catalog = read_json(catalog_path)
        original = [item for item in catalog["demos"] if item.get("environment") != "lunar"]
    if args.direct_only:
        samples, pose_audit = direct_only_sources(inference, inference_hash, args.training_data_manifest,
                                                  args.site_root, args.asset_prefix)
    else:
        samples = finetuned_samples(inference, "direct")
        pose_audit = read_json(args.site_root / args.asset_prefix / "trajectory-provenance.json")
    additions = []
    for sample in samples:
        number = sample["scene"].split("_")[-1]
        prefix = f"{args.asset_prefix}/moon-{number}"
        directory = args.site_root / prefix
        public_path = directory / "manifest.json"
        public = read_json(public_path)
        if public["direct"]["inference_manifest_sha256"] != inference_hash:
            raise ValueError("Direct inference differs from the published source")
        if public["checkpoint_sha256"] != inference["signature"]["checkpoint_sha256"]:
            raise ValueError("Fine-tuned checkpoint differs from the published source")
        if sample["id"] != public["sample_id"] or public["scene"] in public["training"]["scenes"]:
            raise ValueError("Expected the existing held-out sample")
        for key in ("gt_timestamps_ns", "context_timestamps_ns", "actual_prediction_offsets_seconds"):
            if sample[key] != public[key]:
                raise ValueError(f"Source timing mismatch: {key}")
        actions = public["direct"]["metric_actions"]
        if actions != sample["metric_actions"] or len(actions) != 16 or any(
                len(point) != 3 or not all(math.isfinite(value) for value in point) for point in actions):
            raise ValueError("Invalid or mismatched target poses")
        audit = next(item for item in pose_audit["scenes"] if item["scene"] == sample["scene"])
        if sha256(Path(sample["pose_provenance"])) != audit["pose_source_sha256"]:
            raise ValueError("Recorded pose file differs from the audited source")
        for path, digest in sample["source_sha256"].items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"Source image hash mismatch: {Path(path).name}")
        initial = Path(sample["initial_path"])
        with Image.open(initial) as anchor, Image.open(sample["context_paths"][-1]) as context:
            if ImageChops.difference(anchor.convert("RGB"), context.convert("RGB")).getbbox():
                raise ValueError("Shared initial frame is not the last context frame")
        item = {
            "id": f"lusnar-finetuned-moon-{number}", "title": f"Moon {number}", "titleZh": f"月面 {number}",
            "dataset": "LuSNAR", "datasetId": "lusnar", "domain": "held-out scene",
            "environment": "lunar", "fineTuned": True, "sampleId": sample["id"],
            "fps": 4, "duration": 4.25, "horizon": 4, "frameCount": 17,
            "width": 224, "height": 224, "includesInitialFrame": True, "posterTime": 1,
            "labels": {"gt": "Ground truth", "prediction": "OpenNWM"},
            "model": public["model"], "checkpoint": {"sha256": public["checkpoint_sha256"]},
            "sourceManifest": f"{prefix}/manifest.json", "sourceManifestSha256": sha256(public_path),
            "sourceInferenceManifestSha256": inference_hash,
            "protocol": public["direct"]["protocol"], "selection": public["selection"],
            "training": {"steps": public["training"]["settings"]["steps"],
                         "windows": public["training"]["windows"],
                         "targetPairs": public["training"]["target_pairs"],
                         "scenes": public["training"]["scenes"]},
            "limitations": public["limitations"],
            "conditioningTrajectory": {
                "positions": [[0, 0]] + [[forward, left] for forward, left, _ in actions],
                "sampleTimesSeconds": [index / 4 for index in range(17)],
                "captureOffsetsSeconds": [0] + public["actual_prediction_offsets_seconds"],
                "axes": "Forward and left in meters relative to the last observed rover-body pose.",
                "timeSemantics": "Frame 0 is the shared context anchor; frames 1–16 use the paired target pose. Playback is 4 FPS; nonuniform source capture offsets are retained separately.",
                "sourceFile": f"{prefix}/manifest.json", "sourceSha256": sha256(public_path),
                "poseSource": public["pose_source"], "poseSourceSha256": audit["pose_source_sha256"],
            },
            "assetProvenance": {},
        }
        for key, paths_key in (("gt", "gt_paths"), ("prediction", "prediction_paths")):
            future = [Path(path) for path in sample[paths_key]]
            if len(future) != 16:
                raise ValueError("All sixteen target frames are required")
            target = directory / f"slider-{key}.mp4"
            item["assetProvenance"][key] = encode([initial] + future, target, 4)
            item[key] = target.relative_to(args.site_root).as_posix()
            item[f"{key}Poster"] = target.with_suffix(".webp").relative_to(args.site_root).as_posix()
        item["poster"] = item["gtPoster"]
        additions.append(item)
        print(f"Exported Moon {number}: 2 paired streams, 17 frames, 17 recorded positions", flush=True)
    if args.items_output:
        write_public(args.items_output, additions)
        print(f"Lunar items: {args.items_output}", flush=True)
        return
    catalog["demos"] = original + additions
    catalog["timingNote"] = (
        "A shared initial frame and all future frames are encoded at 4 FPS. "
        "16 s and nominal 4 s prediction horizons produce 16.25 s and 4.25 s media. "
        "LuSNAR uses nearest-source sampling; its original nonuniform capture offsets are retained in each sample."
    )
    write_public(catalog_path, catalog)
    print(f"Gallery: {len(original)} Earth + {len(additions)} lunar samples", flush=True)


if __name__ == "__main__":
    main()
