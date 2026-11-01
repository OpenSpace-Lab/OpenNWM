"""Export selected, already-computed goal-conditioned OpenNWM local plans.

Run with: conda run -n nwm python tools/export_website_navigation.py
No model inference is performed. Source PNGs remain unmodified apart from
lossless WebP encoding and H.264 video encoding.
"""

import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image


WEBSITE = Path(__file__).resolve().parents[1] / "website"
ARCHIVE = Path("/file_system/nas/algorithm/dujun.nie/nwm/results/nwm_benchmark/opennwm_navigation_gallery_20260924")
SELECTION = [
    ("recon", 3, "Residential garden", "Residential garden", "RECON"),
    ("scand", 23, "Campus walkway", "Campus walkway", "SCAND"),
    ("tartan_drive", 1, "Woodland road", "Woodland road", "TartanDrive"),
    ("recon", 0, "Across the lawn", "Across the lawn", "RECON"),
    ("scand", 27, "Pedestrian plaza", "Pedestrian plaza", "SCAND"),
    ("huron", 54, "Indoor corridor", "Indoor corridor", "HuRoN"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = json.loads((ARCHIVE / "selected_samples.json").read_text())
    samples = []
    for dataset, sample_id, en, zh, label in SELECTION:
        row = next(r for r in original if r["dataset"] == dataset and r["sample_id"] == sample_id)
        slug = f"{dataset}-{sample_id:03d}"
        source = ARCHIVE / "gallery" / f"{dataset}_{sample_id:03d}"
        output = WEBSITE / "assets" / "navigation" / slug
        output.mkdir(parents=True, exist_ok=True)
        for source_name, target_name in [("GT/000.png", "start.webp"), ("GT/008.png", "goal.webp"), ("OpenNWM/004.png", "poster.webp")]:
            with Image.open(source / source_name) as frame:
                frame.save(output / target_name, lossless=True)
        subprocess.run([
            "/usr/bin/ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", "4", "-start_number", "0", "-i", str(source / "OpenNWM" / "%03d.png"),
            "-frames:v", "9", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(output / "prediction.mp4"),
        ], check=True)
        samples.append({
            "id": slug, "dataset": label, "title": {"en": en, "zh": zh},
            "sample_id": sample_id, "trajectory": row["trajectory"],
            "start_frame": row["start_frame"], "goal_frame": row["goal_frame"],
            "start": f"assets/navigation/{slug}/start.webp",
            "goal": f"assets/navigation/{slug}/goal.webp",
            "poster": f"assets/navigation/{slug}/poster.webp",
            "video": f"assets/navigation/{slug}/prediction.mp4",
            "reference_xy_m": row["gt_xy_m"], "planned_xy_m": row["planned_xy_m"],
            "ate_m": row["ate_m"], "goal_error_m": row["goal_error_m"],
            "normalized_ate": row["normalized_ate"], "normalized_goal_error": row["normalized_goal_error"],
            "selection": "Low trajectory error and visually consistent predictions; manually reviewed all frames.",
            "source_frame_sha256": {f"{kind}/{i:03d}.png": digest(source / kind / f"{i:03d}.png") for kind in ["GT", "OpenNWM"] for i in range(9)},
            "asset_sha256": {p.name: digest(p) for p in sorted(output.iterdir())},
        })
    manifest = {
        "version": 1,
        "model": "OpenNWM Nav15 finalLAM joint100k",
        "checkpoint_sha256": "aabff9b5a3ab62a9145073892653a81019782511272933b6acfb41e21b48b4f4",
        "archive": "opennwm_navigation_gallery_20260924",
        "scope": "Selected offline goal-image-conditioned local plans; no closed-loop execution or success rate is measured.",
        "protocol": json.loads((ARCHIVE / "protocol.json").read_text()),
        "media": {"frame_count": 9, "predicted_frames": 8, "fps": 4, "duration_seconds": 2.25, "resolution": [224, 224], "frame_zero": "Recorded start shared by the two trajectories", "interpolation": False},
        "selection": {"candidate_count": 23, "archived_selected_count": 12, "website_selected_count": len(samples), "review": "Reviewed the full archived contact sheet and all nine prediction frames of the six published samples. Selected consistent scenes and low-error local plans. Go Stanford and Office-Go2 omitted because of visible scene drift; TUM and UZH were already rejected by the archive."},
        "samples": samples,
    }
    (WEBSITE / "content" / "navigation.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"samples": len(samples), "assets": len(samples) * 4, "bytes": sum(p.stat().st_size for p in (WEBSITE / "assets" / "navigation").rglob("*") if p.is_file())}))


if __name__ == "__main__":
    main()
