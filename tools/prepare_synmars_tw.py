#!/usr/bin/env python3
"""Extract and validate the complete SynMars-TW ZIP without converting its data.

--output is the dataset root; the archive's SynMars-TW/ prefix is stripped.
Existing files are read once to verify size/CRC. New files are read once from
the archive, whose CRC zipfile verifies while streaming, then installed without
overwriting an existing destination. The source ZIP is never removed.
"""

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import threading
import time
import zipfile
import zlib

from PIL import Image


SPLITS = {"train": 17000, "val": 2000, "test": 2000}
DIRECTORIES = {
    "images/training": ("images", "train"),
    "images/validation": ("images", "val"),
    "images/testing": ("images", "test"),
    "annotations/training_24": ("annotations", "train"),
    "annotations/validation_24": ("annotations", "val"),
    "annotations/testing_24": ("annotations", "test"),
    "depths/train-depth/train-depth": ("depths", "train"),
    "depths/validation-depth/validation-depth": ("depths", "val"),
    "depths/test-depth/test-depth": ("depths", "test"),
}
TOTAL_FILES = 63001
TOTAL_BYTES = 7539383135
CHUNK = 1024 * 1024


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def member_path(info):
    name = info.filename.rstrip("/") if info.is_dir() else info.filename
    parts = name.split("/")
    require("\\" not in name and "\x00" not in name and
            all(p not in ("", ".", "..") for p in parts),
            f"Unsafe archive path: {info.filename!r}")
    require(parts[0] == "SynMars-TW", f"Unexpected archive root: {name}")
    mode = info.external_attr >> 16
    require(stat.S_IFMT(mode) in (0, stat.S_IFDIR, stat.S_IFREG),
            f"Non-regular archive member: {name}")
    require(not info.flag_bits & 1, f"Encrypted archive member: {name}")
    return PurePosixPath(*parts[1:])


def safe_path(root, relative):
    target = root
    for part in relative.parts:
        target = target / part
        require(not target.is_symlink(), f"Refusing symlink: {target}")
    return target


def existing_matches(target, info):
    require(target.is_file() and not target.is_symlink(),
            f"Existing destination is not a regular file: {target}")
    require(target.stat().st_size == info.file_size,
            f"Existing file size differs; refusing to overwrite: {target}")
    crc = 0
    with target.open("rb") as stream:
        while block := stream.read(CHUNK):
            crc = zlib.crc32(block, crc)
    require(crc & 0xffffffff == info.CRC,
            f"Existing file CRC differs; refusing to overwrite: {target}")


def inspect_archive(archive, inventory_path):
    members = {}
    groups = defaultdict(dict)
    for info in archive.infolist():
        relative = member_path(info)
        if info.is_dir():
            continue
        name = relative.as_posix()
        require(name not in members, f"Duplicate ZIP destination: {name}")
        members[name] = info
        if name == "labels.txt":
            continue
        directory = relative.parent.as_posix()
        require(directory in DIRECTORIES, f"Unexpected data directory: {name}")
        modality, split = DIRECTORIES[directory]
        suffix = "_depth" if modality == "depths" else ""
        match = re.fullmatch(r"(.+)_([lr])" + suffix + r"\.png", relative.name)
        require(match is not None, f"Unexpected image filename: {name}")
        key = (match[1], match[2])
        require(key not in groups[(modality, split)], f"Duplicate image key: {name}")
        groups[(modality, split)][key] = name

    require(len(members) == TOTAL_FILES, f"Expected {TOTAL_FILES} files, got {len(members)}")
    require("labels.txt" in members, "labels.txt is missing")
    require(sum(i.file_size for i in members.values()) == TOTAL_BYTES,
            "Uncompressed total differs from the verified Kaggle inventory")
    pairs = {}
    for (modality, split), names in groups.items():
        require(len(names) == SPLITS[split], f"Wrong count for {modality}/{split}: {len(names)}")
        left = {stem for stem, side in names if side == "l"}
        right = {stem for stem, side in names if side == "r"}
        require(left == right, f"Unpaired left/right filenames: {modality}/{split}")
        pairs[f"{modality}/{split}"] = len(left)
    require(len(groups) == 9, "Expected all nine modality/split combinations")
    for split in SPLITS:
        reference = set(groups[("images", split)])
        for modality in ("annotations", "depths"):
            require(set(groups[(modality, split)]) == reference,
                    f"Cross-modality filenames differ: {modality}/{split}")

    if inventory_path:
        inventory = json.loads(inventory_path.read_text())
        expected = inventory["files"]
        require(len(expected) == len(members), "Inventory file count differs")
        require({item["name"] for item in expected} ==
                {info.filename for info in members.values()}, "Inventory paths differ")
        by_name = {info.filename: info for info in members.values()}
        for item in expected:
            info = by_name[item["name"]]
            require((info.file_size, info.compress_size, info.CRC) ==
                    (item["size"], item["compressed_size"], int(item["crc32"], 16)),
                    f"Inventory metadata differs: {info.filename}")
    return members, groups, pairs


def extract(archive_path, members, output, temp_dir, workers):
    present = set()
    for directory in DIRECTORIES:
        path = safe_path(output, PurePosixPath(directory))
        if path.exists():
            with os.scandir(path) as entries:
                present.update(f"{directory}/{entry.name}" for entry in entries)
    if safe_path(output, PurePosixPath("labels.txt")).exists():
        present.add("labels.txt")
    remaining = sum(info.file_size for name, info in members.items() if name not in present)
    free = shutil.disk_usage(output).free
    log(f"Disk free={free} bytes; remaining extraction={remaining} bytes")
    require(free >= remaining + CHUNK, "Insufficient disk space for extraction")
    for directory in DIRECTORIES:
        safe_path(output, PurePosixPath(directory)).mkdir(parents=True, exist_ok=True)
    local = threading.local()
    opened = []
    worker_dirs = []
    opened_lock = threading.Lock()

    def initialize():
        local.archive = zipfile.ZipFile(archive_path)
        with opened_lock:
            opened.append(local.archive)
        local.temp_dir = Path(tempfile.mkdtemp(prefix="worker-", dir=temp_dir))
        with opened_lock:
            worker_dirs.append(local.temp_dir)

    def extract_one(item):
        name, info = item
        # All parent directories were checked before starting the worker pool.
        target = output / name
        if target.exists():
            existing_matches(target, info)
            return "resumed", info.file_size
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=local.temp_dir, delete=False) as sink:
                temporary = Path(sink.name)
                written = 0
                with local.archive.open(info, "r") as source:
                    while block := source.read(CHUNK):
                        sink.write(block)
                        written += len(block)
                require(written == info.file_size, f"Extracted size differs: {name}")
            # Hard-link publication is atomic and refuses an existing target.
            os.link(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return "extracted", info.file_size

    counts = Counter(extracted=0, resumed=0)
    processed_bytes = 0
    previous_log = time.monotonic()
    index = 0
    pending = set()
    # Spread metadata operations over the nine modality/split directories.
    queues = defaultdict(deque)
    for name, info in members.items():
        queues[name.rpartition("/")[0]].append((name, info))

    def round_robin():
        while queues:
            for directory in tuple(queues):
                yield queues[directory].popleft()
                if not queues[directory]:
                    del queues[directory]

    items = round_robin()
    try:
        with ThreadPoolExecutor(max_workers=workers, initializer=initialize) as pool:
            try:
                for _ in range(workers * 2):
                    item = next(items, None)
                    if item is not None:
                        pending.add(pool.submit(extract_one, item))
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        action, size = future.result()
                        counts[action] += 1
                        processed_bytes += size
                        index += 1
                        now = time.monotonic()
                        if index % 1000 == 0 or now - previous_log >= 15 or index == len(members):
                            log(f"Verified {index}/{len(members)} files, {processed_bytes}/{TOTAL_BYTES} bytes; "
                                f"extracted={counts['extracted']}, resumed={counts['resumed']}")
                            previous_log = now
                        item = next(items, None)
                        if item is not None:
                            pending.add(pool.submit(extract_one, item))
            finally:
                for future in pending:
                    future.cancel()
    finally:
        for archive in opened:
            archive.close()
        for directory in worker_dirs:
            try:
                directory.rmdir()
            except OSError as error:
                log(f"Temporary directory retained: {directory}: {error}")
    return dict(counts)


def validate_output(output, members, groups):
    actual = {"labels.txt"}
    for modality in ("images", "annotations", "depths"):
        for directory, dirs, files in os.walk(output / modality, followlinks=False):
            for name in dirs + files:
                require(not (Path(directory) / name).is_symlink(),
                        f"Symlink in dataset: {Path(directory) / name}")
            actual.update((Path(directory) / name).relative_to(output).as_posix()
                          for name in files)
    require(actual == set(members), "Extracted file set differs from archive")
    actual_bytes = sum((output / name).stat().st_size for name in actual)
    require(actual_bytes == TOTAL_BYTES, "Extracted byte total differs")
    samples = []
    for (modality, split), names in sorted(groups.items()):
        # Both cameras at the first, middle and last stereo frame in each group.
        stems = sorted(stem for stem, side in names if side == "l")
        for index in sorted({0, len(stems) // 2, len(stems) - 1}):
            for side in ("l", "r"):
                name = names[(stems[index], side)]
                with Image.open(output / name) as picture:
                    picture.load()
                    require(picture.size == (512, 512), f"Unexpected dimensions: {name}")
                    samples.append({"file": name, "modality": modality, "split": split,
                                    "size": list(picture.size), "mode": picture.mode,
                                    "format": picture.format})
    return actual_bytes, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inventory", type=Path, help="Optional verified central-directory JSON")
    parser.add_argument("--workers", type=int, default=8, help="Independent ZIP readers (default: 8)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = safe_path(output, PurePosixPath("_download"))
    metadata.mkdir(exist_ok=True)
    temp_dir = safe_path(output, PurePosixPath("_download/.extract_tmp"))
    temp_dir.mkdir(exist_ok=True)
    report_path = safe_path(output, PurePosixPath("_download/validation.json"))
    started = time.monotonic()
    report = {"status": "running", "archive": str(args.archive.resolve()), "output": str(output),
              "started_utc": datetime.now(timezone.utc).isoformat()}
    exit_code = 0
    try:
        log(f"Reading ZIP central directory: {args.archive}")
        with zipfile.ZipFile(args.archive) as archive:
            members, groups, pairs = inspect_archive(archive, args.inventory)
            log(f"Archive structure verified: {len(members)} files; "
                "21,000 images per modality = 10,500 stereo pairs")
        progress = extract(args.archive, members, output, temp_dir, args.workers)
        actual_bytes, samples = validate_output(output, members, groups)
        report.update(status="passed", total_files=len(members), uncompressed_bytes=actual_bytes,
                      counts={f"{m}/{s}": len(v) for (m, s), v in sorted(groups.items())},
                      camera_counts={f"{m}/{s}": dict(Counter(side for _, side in v))
                                     for (m, s), v in sorted(groups.items())},
                      stereo_pairs=pairs, cross_modality_pairing="passed",
                      file_size_and_crc32="passed", extraction=progress, workers=args.workers,
                      image_samples=samples, sampled_images=len(samples),
                      inventory=str(args.inventory.resolve()) if args.inventory else None)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        log(report["error"])
        exit_code = 1
    report.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=round(time.monotonic() - started, 3))
    with tempfile.NamedTemporaryFile(mode="w", dir=metadata, delete=False) as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        temporary_report = Path(stream.name)
    os.replace(temporary_report, report_path)
    log(f"Validation {report['status']}: {report_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
