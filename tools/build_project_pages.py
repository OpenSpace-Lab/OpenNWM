#!/usr/bin/env python3
"""Build the reviewed static website for the OpenNWM project Pages URL.

Only tracked, explicitly allowed public files enter the artifact. No publishing,
network access, server execution, or inference happens here. Use a new output
directory; a failed build leaves no partial output directory. Requires Python 3.11+.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit

from publish_website_refinement_20261031 import identity, review_evidence


ROOT_FILES = frozenset("""
app.js cinematic.js go2.css go2.js index.html media-cache-worker.js
media-cache.js media-delivery.js media-scheduler.js method.css motion-gallery.css
motion-gallery.js navigation.css navigation.js playground.js reference-theme.css
research.js results.css space-trajectories.js space-trajectory.css style.css
""".split())
CONTENT_FILES = frozenset("content/" + name for name in """
GO2_MANIFEST.md LUNAR_CURVES_20261029_VALIDATION.md MARSLAB_20261029_VALIDATION.md
MARSLAB_CURVES_20261029_VALIDATION.md METHOD_FIGURES.md PAPER_SOURCES.md
SYNMARS_20261029_VALIDATION.md cinematic-display.json cinematic.json datasets.json
demos.json go2-selection.json go2.json live-service.json media-delivery.json
navigation.json paper.json playground.json space.json
""".split())
ASSET_DIRECTORIES = frozenset("space playground demos datasets navigation paper hero fonts".split())
ASSET_SUFFIXES = frozenset(".webp .mp4 .json .csv .svg .pdf .png .txt .ttf .md".split())
ROOT_ASSETS = frozenset("assets/" + name for name in (
    "mark.svg", "opennwm-logo.pdf", "opennwm-logo.svg", "opennwm-logo.zip"))
REVIEW = "assets/go2/privacy-review.json"
MAX_BYTES = 950_000_000


def source_file(site: Path, name: str) -> Path:
    """Reject path normalization, escape, and symlinks, including ancestors."""
    path = PurePosixPath(name)
    if (not isinstance(name, str) or path.is_absolute() or path.as_posix() != name
            or any(part in (".", "..") for part in path.parts)
            or any(char in name for char in ("\\", "?", "#"))):
        raise ValueError(f"Unsafe public path: {name!r}")
    candidate = site
    for part in path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"Symlink is not allowed: {name}")
    if not candidate.is_file() or not candidate.resolve().is_relative_to(site):
        raise ValueError(f"Missing or unsafe public file: {name}")
    return candidate


def tracked_files(site: Path) -> set[str]:
    raw = subprocess.check_output(
        ["git", "-C", str(site), "ls-files", "-z", "--", "."])
    return {name for name in raw.decode().split("\0") if name}


def ordinary_public_file(name: str) -> bool:
    path = PurePosixPath(name)
    return (name in ROOT_FILES or name in CONTENT_FILES or name in ROOT_ASSETS
            or (len(path.parts) >= 3 and path.parts[0] == "assets"
                and path.parts[1] in ASSET_DIRECTORIES
                and path.suffix in ASSET_SUFFIXES
                and not any(part.startswith(".") for part in path.parts)))


def public_files(site: Path) -> tuple[set[str], dict, dict]:
    tracked = tracked_files(site)
    required = ROOT_FILES | CONTENT_FILES | ROOT_ASSETS | {REVIEW, "assets/go2/report.json"}
    if missing := required - tracked:
        raise ValueError(f"Required public files are not tracked: {sorted(missing)}")
    files = {name for name in tracked if ordinary_public_file(name)} | required
    for name in files:
        source_file(site, name)
    html = source_file(site, "index.html").read_text()
    inline = re.search(r'<script\b[^>]*\bid="media-delivery-config"[^>]*>(.*?)</script>', html, re.S)
    if not inline or json.loads(inline.group(1)) != json.loads(
            source_file(site, "content/media-delivery.json").read_text()):
        raise ValueError("Inline media delivery configuration differs from its JSON file")
    go2 = json.loads(source_file(site, "content/go2.json").read_text())
    review_path = source_file(site, REVIEW)
    review = json.loads(review_path.read_text())
    clips = go2["clips"]
    if not clips or review.get("clip_count") != len(clips):
        raise ValueError("Privacy review clip count differs from the selected manifest")
    reviewed = {clip["id"]: clip for clip in review["clips"]}
    identities = {}
    fallbacks = []
    for clip in clips:
        if clip.get("privacy", {}).get("reviewed") is not True:
            raise ValueError(f"Clip is not privacy reviewed: {clip['id']}")
        row = reviewed[clip["id"]]
        for key, suffix in (("video", ".mp4"), ("poster", ".webp")):
            name = clip[key]
            path = PurePosixPath(name)
            if (len(path.parts) != 4 or path.parts[:2] != ("assets", "go2")
                    or path.parts[2] != clip["id"] or path.suffix != suffix or name not in tracked):
                raise ValueError(f"Invalid selected Go2 {key}: {name}")
            identities[name] = identity(source_file(site, name))
            files.add(name)
        for key, suffix in (("metrics", ".json"), ("trajectories", ".csv")):
            name = clip["downloads"][key]
            if (PurePosixPath(name).parent != PurePosixPath(clip["video"]).parent
                    or PurePosixPath(name).suffix != suffix or name not in tracked):
                raise ValueError(f"Invalid selected Go2 download: {name}")
            proof = row["downloads"][key]
            identities[name] = identity(source_file(site, name))
            if proof.get("path") != name or any(
                    proof.get(k) != identities[name][k] for k in ("bytes", "sha256")):
                raise ValueError(f"Privacy review does not match download: {name}")
            files.add(name)
        if fallback := clip.get("fallback_video"):
            url = urlsplit(fallback)
            expected_path = (r"/B0B8K1ng/b0b8k1ng\.github\.io/[0-9a-f]{40}/"
                             + re.escape(clip["video"]))
            if (url.scheme != "https" or url.netloc != "raw.githubusercontent.com"
                    or not re.fullmatch(expected_path, url.path) or url.query or url.fragment
                    or row.get("fallback_video") != fallback
                    or row.get("fallback_video_sha256") != identities[clip["video"]]["sha256"]):
                raise ValueError(f"Fallback is not the reviewed pinned public video: {clip['id']}")
            fallbacks.append(clip["video"])
    # Reuse the publisher's pure, offline verifier for all selected media hashes.
    review_evidence(review_path, go2, identities)
    files.difference_update(fallbacks)
    return files, identities, {
        "clips": len(clips), "selected_media_sha256_verified": len(clips) * 2,
        "download_sha256_verified": len(clips) * 2,
        "pinned_fallback_masters_omitted": len(fallbacks),
        "excluded_tracked_files": len(tracked - files),
        "inline_media_configuration_matches": True,
    }


def build(site: Path, output: Path) -> dict:
    if site.is_symlink():
        raise ValueError("Website root cannot be a symlink")
    site = site.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Use a new output directory: {output}")
    if output.resolve().is_relative_to(site) or site.is_relative_to(output.resolve()):
        raise ValueError("Source and output directories must not overlap")
    files, identities, checks = public_files(site)
    total = sum(source_file(site, name).stat().st_size for name in files)
    if total >= MAX_BYTES:
        raise ValueError(f"Static output exceeds the {MAX_BYTES} byte budget: {total}")
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    if free < total + 100_000_000:
        raise ValueError(f"Insufficient free disk: {free} bytes")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name in sorted(files):
            destination = temporary / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file(site, name), destination)
            if name in identities and identity(destination) != identities[name]:
                raise ValueError(f"Reviewed asset changed while copying: {name}")
        (temporary / ".nojekyll").touch()
        # Detect changes in the manifest/review during a local build, too.
        copied_files, _, copied_checks = public_files(site)
        if copied_files != files or copied_checks != checks:
            raise ValueError("Public selection changed during the build")
        for name in ("content/go2.json", REVIEW, "index.html", "content/media-delivery.json"):
            if identity(temporary / name) != identity(source_file(site, name)):
                raise ValueError(f"Public manifest changed while copying: {name}")
        actual_bytes = sum(path.stat().st_size for path in temporary.rglob("*") if path.is_file())
        if actual_bytes >= MAX_BYTES:
            raise ValueError(f"Copied output exceeds the byte budget: {actual_bytes}")
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"status": "passed", "source": str(site), "output": str(output),
            "files": len(files) + 1, "bytes": actual_bytes, "free_disk_before_copy": free,
            **checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-root", type=Path, default=Path(__file__).resolve().parents[1] / "website")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Write a JSON report outside the public output")
    args = parser.parse_args()
    if args.report and args.report.resolve().is_relative_to(args.output_dir.resolve()):
        parser.error("--report must be outside the public output")
    report = build(args.site_root, args.output_dir)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
