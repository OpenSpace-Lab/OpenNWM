"""Export visually reviewed model rollouts for the compact website gallery.

CPU-only: this reads completed inference outputs and never runs a model.
The archive contains future frames only; prepend its shared initial image.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "website"
DEFAULT_ARCHIVE = Path("/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/opennwm_pixel_corrected_20260924")
SELECTION = [
    ("recon", 25, "Open meadow", "开阔草地", "Grass and sky remain coherent during the turn; tree positions and terrain details drift."),
    ("scand", 133, "Campus path", "校园步道", "Preserves the tree-lined paved setting; the recorded pedestrians and canopy are not reconstructed precisely."),
    ("tartan_drive", 99, "Hillside trail", "山坡小径", "Preserves a vegetated hillside and visible motion; the path and camera pitch diverge over the long rollout."),
    ("recon", 55, "Garden path", "花园小径", "Retains the grass, path and small buildings; exact building geometry and motion differ."),
    ("scand", 5, "Shaded plaza", "林荫广场", "Preserves brick paving and tree cover with continuous motion; shadows and scene details change."),
    ("recon", 100, "Country buildings", "乡间建筑", "Maintains an outdoor building scene through the turn; view timing and building positions drift."),
    ("recon", 10, "Overcast field", "阴天田野", "Maintains the overcast field and tree silhouettes; the final building and plant geometry differ."),
]
DATASET_NAMES = {"recon": "RECON", "scand": "SCAND", "tartan_drive": "TartanDrive"}


def attach_trajectories(manifest, archive):
    """Preserve every recorded position aligned with the exported video frames."""
    for item in manifest["demos"]:
        # LuSNAR uses its own held-out inference and audited rover-body poses.
        # Its paired media/trajectory are rebuilt by build_website_lunar_gallery.py.
        if item.get("environment") == "lunar":
            continue
        path = archive / item["sourceExample"] / "trajectory.json"
        source = json.loads(path.read_text())
        points = source["positions_local"]
        times = source["sample_times_seconds"]
        if source["trajectory"] != item["trajectory"] or source["initial_frame"] != item["initialSourceFrame"]:
            raise ValueError(f"Trajectory identity does not match video: {item['id']}")
        if len(points) != item["frameCount"] or len(times) != len(points):
            raise ValueError(f"Trajectory and video frame counts disagree: {item['id']}")
        if any(len(point) != 2 or not all(math.isfinite(value) for value in point) for point in points):
            raise ValueError(f"Invalid trajectory coordinates: {item['id']}")
        if any(abs(time - index / item["fps"]) > 1e-8 for index, time in enumerate(times)):
            raise ValueError(f"Trajectory and playback timing disagree: {item['id']}")
        item["conditioningTrajectory"] = {
            "positions": points, "sampleTimesSeconds": times,
            "axes": source["axes"], "timeSemantics": source["time_semantics"],
            "sourceArchive": archive.name, "sourceFile": f"{item['sourceExample']}/trajectory.json",
            "sourceSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"Recorded trajectory: {item['id']}, {len(points)} positions", flush=True)


def encode(paths, target, fps):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", "224x224", "-framerate", str(fps),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
               "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(target)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    source_hash = hashlib.sha256()
    try:
        for path in paths:
            source_hash.update(path.read_bytes())
            with Image.open(path) as frame:
                if frame.size != (224, 224):
                    raise ValueError(f"Unexpected archived image size: {path}: {frame.size}")
                process.stdin.write(frame.convert("RGB").tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {target}")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    with Image.open(paths[min(fps, len(paths) - 1)]) as frame:
        frame.save(target.with_suffix(".webp"), quality=94)
    return {"sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "sourceFramesSha256": source_hash.hexdigest(), "frameCount": len(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--trajectories-only", action="store_true", help="Add recorded paths without re-encoding any media")
    args = parser.parse_args()
    manifest_path = SITE / "content/demos.json"
    manifest = json.loads(manifest_path.read_text())
    if args.trajectories_only:
        attach_trajectories(manifest, args.archive)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        return
    template = copy.deepcopy(next(item for item in manifest["demos"] if item["id"] == "tartan-drive-36"))
    existing = {item["id"]: item for item in manifest["demos"]}
    existing["tartan-drive-36"]["titleZh"] = "碎石小径"
    existing["unitree-go2-81"]["titleZh"] = "办公室漫步"
    out = SITE / "assets/demos"
    out.mkdir(exist_ok=True, parents=True)
    new_ids = []
    for dataset, sample_id, title, title_zh, limitations in SELECTION:
        example = args.archive / "examples" / dataset / f"id_{sample_id}"
        metrics = json.loads((example / "metrics.json").read_text())
        trajectory = json.loads((example / "trajectory.json").read_text())
        initial = args.archive / "initial" / dataset / f"id_{sample_id}.png"
        identifier = f"{dataset.replace('_', '-')}-{sample_id}"
        item = copy.deepcopy(template)
        item.update(id=identifier, title=title, titleZh=title_zh, dataset=DATASET_NAMES[dataset],
                    datasetId=dataset, sampleId=sample_id, domain=metrics["group"],
                    description=f"A {metrics['seconds']}-second action-conditioned rollout.",
                    horizon=metrics["seconds"], fps=metrics["fps"],
                    sourceArchive=args.archive.name, sourceExample=f"examples/{dataset}/id_{sample_id}",
                    trajectory=trajectory["trajectory"], initialSourceFrame=trajectory["initial_frame"],
                    scope=f"Full {metrics['seconds']}-second qualitative rollout.", limitations=limitations,
                    selection="Selected for scene diversity and visually coherent motion after reviewing the archived rollout. This is not an aggregate evaluation or a claim of superiority for every sample.",
                    visualReviewDate="2026-10-08", assetProvenance={})
        for key, model in [("gt", "GT"), ("prediction", "opennwm-finalLAM-100k"), ("baseline", "nwm-release")]:
            future = sorted((example / "frames" / model).glob("*.png"))
            expected = int(metrics["seconds"] * metrics["fps"])
            if len(future) != expected or [path.name for path in future] != [f"{i:03d}.png" for i in range(expected)]:
                raise ValueError(f"Incomplete inference frames: {example / model}")
            target = out / f"{identifier}-{key}.mp4"
            # Retain the existing prediction/baseline/gt field names for clients.
            item[key] = target.relative_to(SITE).as_posix()
            item[f"{key}Poster"] = target.with_suffix(".webp").relative_to(SITE).as_posix()
            item["assetProvenance"][key] = encode([initial] + future, target, metrics["fps"])
        item.update(poster=item["gtPoster"], frameCount=expected + 1,
                    duration=(expected + 1) / metrics["fps"], includesInitialFrame=True)
        existing[identifier] = item
        new_ids.append(identifier)
        print(f"Exported {identifier}: 3 streams, {item['frameCount']} frames each", flush=True)
    # Keep existing entries and their order; additions follow the reviewed order.
    order = [item["id"] for item in manifest["demos"]]
    order.extend(identifier for identifier in new_ids if identifier not in order)
    manifest["demos"] = [existing[identifier] for identifier in order]
    attach_trajectories(manifest, args.archive)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Gallery manifest: {len(manifest['demos'])} examples", flush=True)


if __name__ == "__main__":
    main()
