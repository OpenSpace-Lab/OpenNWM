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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    args = parser.parse_args()
    inference = read_json(args.direct_manifest)
    if not inference.get("completed_utc"):
        raise ValueError("Only completed inference may be exported")
    inference_hash = sha256(args.direct_manifest)
    catalog_path = SITE / "content/demos.json"
    catalog = read_json(catalog_path)
    original = [item for item in catalog["demos"] if item.get("environment") != "lunar"]
    pose_audit = read_json(SITE / "assets/space/lusnar-finetuned/trajectory-provenance.json")
    additions = []
    for sample in finetuned_samples(inference, "direct"):
        number = sample["scene"].split("_")[-1]
        prefix = f"assets/space/lusnar-finetuned/moon-{number}"
        directory = SITE / prefix
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
            item[key] = target.relative_to(SITE).as_posix()
            item[f"{key}Poster"] = target.with_suffix(".webp").relative_to(SITE).as_posix()
        item["poster"] = item["gtPoster"]
        additions.append(item)
        print(f"Exported Moon {number}: 2 paired streams, 17 frames, 17 recorded positions", flush=True)
    catalog["demos"] = original + additions
    catalog["timingNote"] = (
        "A shared initial frame and all future frames are encoded at 4 FPS. "
        "16 s and nominal 4 s prediction horizons produce 16.25 s and 4.25 s media. "
        "LuSNAR uses nearest-source sampling; its original nonuniform capture offsets are retained in each sample."
    )
    serialized = json.dumps(catalog, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if "/file_system/" in serialized:
        raise ValueError("Public gallery must not contain machine-local paths")
    catalog_path.write_text(serialized)
    print(f"Gallery: {len(original)} Earth + {len(additions)} lunar samples", flush=True)


if __name__ == "__main__":
    main()
