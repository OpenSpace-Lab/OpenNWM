#!/usr/bin/env python3
"""Create shared trajectory-level validation splits without reading any images."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import shutil


TRAIN_DATASETS = ("recon", "scand", "sacson", "tartan_drive")
TEST_FILES = {
    "traj_names.txt", "all_traj_names.txt", "rollout_traj_names.txt",
    "time.pkl", "rollout.pkl", "navigation_eval.pkl",
    "invalid_indices_in_time.txt", "invalid_indices_in_rollout.txt",
    "invalid_indices_in_navigation_eval.txt",
}


def make_split(source: Path, output: Path, fraction: float, seed: int) -> dict:
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("--output must be separate from the source split directory")
    if not 0 < fraction < 1:
        raise ValueError("--val-fraction must be between zero and one")
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists() and any(output.iterdir()):
        raise ValueError("--output must be empty; reuse a completed split directly")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"seed": seed, "val_fraction": fraction, "datasets": {}}
    # Only small split directories are listed. Image/data roots are never opened.
    for dataset in sorted(path for path in source.iterdir() if path.is_dir()):
        name = dataset.name
        counts = {}
        train_file = dataset / "train" / "traj_names.txt"
        if name in TRAIN_DATASETS and train_file.is_file():
            names = list(dict.fromkeys(line.strip() for line in train_file.read_text().splitlines() if line.strip()))
            if len(names) < 2:
                raise ValueError(f"{name}: at least two trajectories are required")
            # A dataset-specific string seed keeps this split unchanged when
            # another dataset is added, and is stable across Python processes.
            shuffled = sorted(names)
            random.Random(f"{seed}:{name}").shuffle(shuffled)
            val_count = min(len(names) - 1, max(1, math.ceil(len(names) * fraction)))
            validation = set(shuffled[:val_count])
            for split, selected in (("train", [item for item in names if item not in validation]),
                                    ("val", [item for item in names if item in validation])):
                destination = output / name / split / "traj_names.txt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text("\n".join(selected) + "\n")
                counts[split] = len(selected)
        test_dir = dataset / "test"
        copied = 0
        for filename in sorted(TEST_FILES):
            source_file = test_dir / filename
            if source_file.is_file():
                destination = output / name / "test" / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, destination)
                copied += 1
        if counts or copied:
            summary["datasets"][name] = {**counts, "test_files": copied}
    (output / "split.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "nwm/data_splits")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()
    summary = make_split(args.source, args.output, args.val_fraction, args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
