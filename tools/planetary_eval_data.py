"""Deterministic, traceable evaluation samples from raw planetary datasets.

LuSNAR supplies recorded rover-body poses; these become planar actions after a
documented axis conversion. SynMars-TW supplies no action or timestamp truth, so
its samples are independent initial-image stress tests with synthetic controls.
No source files are modified and no learned pose estimates are substituted.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


COORDINATE_NOTE = (
    "GT is rover body-to-world, quaternion wxyz. Assume AirSim-style body "
    "+X forward, +Y right, +Z down, supported by the published left camera "
    "translation [1,-0.155,-1.5] and forward-motion audit. Reflect world Y "
    "and negate yaw to obtain OpenNWM +X forward/+Y left, positive-left yaw. "
    "Actions use rover-body XY/yaw, not a calibrated optical-camera trajectory; "
    "height, roll and pitch are not conditioned on."
)


def wrap_angle(value):
    return np.arctan2(np.sin(value), np.cos(value))


def local_action(start_position, start_yaw, end_position, end_yaw, spacing=1.0):
    """Planar local displacement in spacing units, and wrapped yaw in radians."""
    dx, dy = np.asarray(end_position) - np.asarray(start_position)
    c, s = np.cos(start_yaw), np.sin(start_yaw)
    return np.asarray([(c * dx + s * dy) / spacing,
                       (-s * dx + c * dy) / spacing,
                       wrap_angle(end_yaw - start_yaw)], dtype=np.float64)


def _nearest_indices(sorted_ns, query_ns):
    """Use integer timestamps throughout nearest-neighbor association."""
    values = [int(x) for x in sorted_ns]
    result = []
    for query in query_ns:
        index = bisect.bisect_left(values, int(query))
        candidates = range(max(index - 1, 0), min(index + 1, len(values)))
        result.append(min(candidates, key=lambda i: (abs(values[i] - int(query)), i)))
    return np.asarray(result, dtype=np.int64)


def _read_lusnar_scene(root, name, fps):
    scene = root / name
    rows = list(csv.reader(line for line in (scene / "image0/color_timestamp.txt")
                           .read_text().splitlines() if line and not line.startswith("#")))
    timestamps = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    paths = [str(scene / "image0/color" / row[1].strip()) for row in rows]
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{name}: image timestamps must strictly increase")
    poses = {}
    duplicate_rows = 0
    for row in csv.reader(line for line in (scene / "gt.txt").read_text().splitlines()
                          if line and not line.startswith("#")):
        timestamp = int(row[0])
        values = np.asarray([float(x) for x in row[1:]], dtype=np.float64)
        if len(values) != 16 or not np.isfinite(values).all():
            raise ValueError(f"{name}: malformed pose at {timestamp}")
        if timestamp in poses:
            if not np.array_equal(poses[timestamp], values):
                raise ValueError(f"{name}: conflicting duplicate pose at {timestamp}")
            duplicate_rows += 1
        poses[timestamp] = values
    pose_ns = np.asarray(sorted(poses), dtype=np.int64)
    pose_values = np.asarray([poses[int(t)] for t in pose_ns])
    association = _nearest_indices(pose_ns, timestamps)
    alignment_ns = np.abs(pose_ns[association] - timestamps)
    if int(alignment_ns.max()) > 1_000_000:
        raise ValueError(f"{name}: image/GT alignment exceeds 1 ms")
    aligned = pose_values[association]
    quaternion = aligned[:, 3:7]
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(norms < 0.5) or np.any(norms > 1.5):
        raise ValueError(f"{name}: invalid quaternion norms")
    w, x, y, z = (quaternion / norms[:, None]).T
    body_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    position = aligned[:, :2] * np.asarray([1.0, -1.0])
    yaw = -body_yaw
    period_ns = int(round(1_000_000_000 / fps))
    grid_ns = np.arange(int(timestamps[0]), int(timestamps[-1]) + 1,
                        period_ns, dtype=np.int64)
    grid_indices = _nearest_indices(timestamps, grid_ns)
    errors_ns = np.abs(timestamps[grid_indices] - grid_ns)
    grid_position, grid_yaw = position[grid_indices], yaw[grid_indices]
    displacement = np.linalg.norm(np.diff(grid_position, axis=0), axis=1)
    valid_spacing = ((np.diff(grid_indices) > 0) & (errors_ns[:-1] <= period_ns // 2)
                     & (errors_ns[1:] <= period_ns // 2) & (displacement > 1e-6))
    spacing = float(np.median(displacement[valid_spacing]))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError(f"{name}: no valid nonzero displacement for normalization")
    raw_steps = np.diff(aligned[:, :2], axis=0)
    lengths = np.linalg.norm(raw_steps, axis=1)
    moving = lengths > 0.01
    forward = raw_steps[:, 0] * np.cos(body_yaw[:-1]) + raw_steps[:, 1] * np.sin(body_yaw[:-1])
    lateral = -raw_steps[:, 0] * np.sin(body_yaw[:-1]) + raw_steps[:, 1] * np.cos(body_yaw[:-1])
    audit = {
        "scene": name, "source_frames": len(paths), "resampled_grid_frames": len(grid_ns),
        "source_effective_fps": float((len(paths) - 1) * 1e9 / int(timestamps[-1] - timestamps[0])),
        "source_duration_seconds": float(int(timestamps[-1] - timestamps[0]) / 1e9),
        "duplicate_gt_rows_removed": duplicate_rows,
        "maximum_image_gt_alignment_ns": int(alignment_ns.max()),
        "waypoint_spacing_meters": spacing,
        "normalization": f"Scene median nonzero XY displacement between unique nearest frames on a {fps:g} Hz grid; heuristic dataset-specific action normalization, not camera calibration.",
        "moving_step_forward_fraction": float(np.mean(forward[moving] > 0)),
        "moving_step_median_absolute_lateral_ratio": float(np.median(np.abs(lateral[moving]) / lengths[moving])),
        "source_vertical_range_meters": float(np.ptp(aligned[:, 2])),
        "source_timestamp_path": str(scene / "image0/color_timestamp.txt"),
        "source_gt_path": str(scene / "gt.txt"),
        "coordinate_convention": COORDINATE_NOTE,
    }
    return paths, timestamps, position, yaw, grid_ns, grid_indices, errors_ns, spacing, audit


def _lusnar_samples(root, windows_per_scene, frames, fps):
    samples, audits = [], []
    period_ns = int(round(1e9 / fps))
    horizons = sorted(set(step for step in (1, 4, 8, frames) if step <= frames))
    for scene_number in range(1, 10):
        name = f"Moon_{scene_number}"
        paths, timestamps, position, yaw, grid, indices, error, spacing, audit = _read_lusnar_scene(root, name, fps)
        candidates = []
        for anchor in range(3, len(grid) - frames):
            window = slice(anchor - 3, anchor + frames + 1)
            if np.all(np.diff(indices[window]) > 0) and np.all(error[window] <= period_ns // 2):
                candidates.append(anchor)
        selected = []
        # Bin centers spread windows across the scene and avoid privileging the startup.
        targets = np.linspace(0, len(grid) - 1, windows_per_scene * 2 + 1)[1::2]
        for target in targets:
            available = [i for i in candidates if all(abs(i - old) > frames + 3 for old in selected)]
            if not available:
                raise ValueError(f"{name}: cannot select {windows_per_scene} disjoint valid windows")
            selected.append(min(available, key=lambda i: (abs(i - target), i)))
        selected.sort()
        audit["selected_grid_anchor_indices"] = selected
        audit["windows"] = len(selected)
        audits.append(audit)
        for window_number, anchor in enumerate(selected):
            source_anchor = int(indices[anchor])
            context_indices = indices[anchor - 3:anchor + 1]
            future_indices = indices[anchor + 1:anchor + frames + 1]
            context_paths = [paths[int(i)] for i in context_indices]
            for path in context_paths + [paths[int(i)] for i in future_indices]:
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
            prefix = f"lusnar_{name}_window{window_number:02d}"
            common = {
                "dataset": "LuSNAR", "scene": name, "window": window_number,
                "context_source_paths": context_paths, "noise_key": prefix,
                "waypoint_spacing_meters": spacing, "nominal_fps": fps,
                "action_translation_units": f"meters divided by scene median {fps:g} Hz waypoint spacing",
                "action_yaw_units": "radians, positive left",
                "coordinate_convention": COORDINATE_NOTE,
                "context_timestamps_ns": [int(timestamps[i]) for i in context_indices],
                "source_anchor_index": source_anchor, "anchor_timestamp_ns": int(timestamps[source_anchor]),
                "context_grid_offsets_seconds": [-3 / fps, -2 / fps, -1 / fps, 0.0],
                "pose_provenance": audit["source_gt_path"],
            }
            chosen = [int(future_indices[h - 1]) for h in horizons]
            samples.append({**common, "id": prefix + "_direct", "protocol": "direct",
                "prediction_steps": horizons,
                "gt_source_paths": [paths[i] for i in chosen],
                "gt_timestamps_ns": [int(timestamps[i]) for i in chosen],
                "actual_prediction_offsets_seconds": [int(timestamps[i] - timestamps[source_anchor]) / 1e9 for i in chosen],
                "actions": [local_action(position[source_anchor], yaw[source_anchor], position[i], yaw[i], spacing).tolist() for i in chosen],
                "metric_actions": [local_action(position[source_anchor], yaw[source_anchor], position[i], yaw[i]).tolist() for i in chosen],
                "action_reference": "Each target pose relative to the fixed current rover body pose"})
            if window_number == 0:
                pairs = zip(indices[anchor:anchor + frames], future_indices)
                actions = [local_action(position[a], yaw[a], position[b], yaw[b], spacing).tolist() for a, b in pairs]
                samples.append({**common, "id": prefix + "_autoregressive", "protocol": "autoregressive",
                    "prediction_steps": list(range(1, frames + 1)), "actions": actions,
                    "gt_source_paths": [paths[int(i)] for i in future_indices],
                    "gt_timestamps_ns": [int(timestamps[i]) for i in future_indices],
                    "actual_prediction_offsets_seconds": [int(timestamps[i] - timestamps[source_anchor]) / 1e9 for i in future_indices],
                    "action_reference": "Each next recorded body pose relative to the immediately preceding recorded body pose"})
    return samples, audits


def _synmars_samples(root, image_count, frames):
    inventory_path = root / "_download/archive_inventory.json"
    inventory = json.loads(inventory_path.read_text())
    groups = defaultdict(list)
    for record in inventory["files"]:
        relative = Path(record["name"])
        if relative.parts[:3] == ("SynMars-TW", "images", "testing") and relative.name.endswith("_l.png"):
            prefix, index, _ = relative.stem.rsplit("_", 2)
            groups[prefix].append((int(index), root / Path(*relative.parts[1:])))
    if not groups or image_count < len(groups):
        raise ValueError("SynMars test inventory absent or insufficient requested images")
    selected = []
    for group_number, (prefix, entries) in enumerate(sorted(groups.items())):
        entries.sort()
        count = image_count // len(groups) + (group_number < image_count % len(groups))
        if count > len(entries):
            raise ValueError("Requested more unique SynMars initial images than available")
        selected.extend((prefix, *entries[int(i)]) for i in np.linspace(0, len(entries) - 1, count).round().astype(int))
    samples = []
    for prefix, index, path in selected:
        if not path.is_file():
            raise FileNotFoundError(path)
        for control, action in (("straight", [0.5, 0.0, 0.0]), ("left", [0.5, 0.0, 0.05]), ("right", [0.5, 0.0, -0.05])):
            samples.append({
                "id": f"synmars_{prefix}_{index:04d}_{control}", "dataset": "SynMars-TW",
                "protocol": "synthetic_control", "control": control,
                "context_source_paths": [str(path)] * 4,
                "gt_source_paths": None, "actions": [action.copy() for _ in range(frames)],
                "prediction_steps": list(range(1, frames + 1)),
                "noise_key": f"synmars_{prefix}_{index:04d}",
                "source_split": "testing", "source_camera": "left", "source_prefix": prefix,
                "source_index": index, "source_timestamp": None,
                "action_translation_units": "model normalized waypoint units; no metric calibration",
                "action_yaw_units": "radians, positive left", "nominal_fps": None,
                "protocol_note": "Independent initial-image stress test: one RGB repeated four times, then generated context autoregression with synthetic controls. No GT target, action or capture time is available. Filename order is not treated as validated temporal adjacency.",
            })
    return samples, {"source_inventory": str(inventory_path), "testing_left_images": sum(map(len, groups.values())),
                     "testing_prefix_counts": {k: len(v) for k, v in sorted(groups.items())},
                     "selected_initial_images": len(selected), "controls_per_image": 3,
                     "ground_truth_actions_available": False, "capture_timestamps_available": False}


def build_samples(lusnar_root: Path, synmars_root: Path, windows_per_scene=4,
                  synmars_images=12, frames=16, fps=4):
    if windows_per_scene < 1 or synmars_images < 1 or frames < 1 or fps <= 0:
        raise ValueError("Sample counts, frames and fps must be positive")
    lunar, lunar_audit = _lusnar_samples(Path(lusnar_root), windows_per_scene, frames, fps)
    mars, mars_audit = _synmars_samples(Path(synmars_root), synmars_images, frames)
    samples = lunar + mars
    if len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("Duplicate sample identifiers")
    for sample in samples:
        action = np.asarray(sample["actions"])
        if action.shape != (len(sample["prediction_steps"]), 3) or not np.isfinite(action).all():
            raise ValueError(f"Invalid actions: {sample['id']}")
        if len(sample["context_source_paths"]) != 4:
            raise ValueError(f"Invalid context: {sample['id']}")
    metadata = {"schema_version": 1, "lusnar": lunar_audit, "synmars": mars_audit,
                "parameters": {"windows_per_scene": windows_per_scene, "synmars_images": synmars_images,
                               "frames": frames, "fps": fps},
                "limitations": ["Zero-shot evaluation, no adaptation on either dataset.",
                    "LuSNAR results depend on documented planar axis and scene-scale normalization assumptions.",
                    f"LuSNAR {fps:g} Hz is a nearest-frame resampling grid; actual capture offsets are retained.",
                    "SynMars supports qualitative synthetic-control stress testing, not GT action prediction or navigation success evaluation."]}
    return samples, metadata


def build_lusnar_finetune_split(lusnar_root, previous_manifest_path):
    """Build a fixed few-shot split without modifying the zero-shot protocol.

    Each sample contains four context images and all sixteen future endpoints.
    Translation uses one scale fitted only to the selected training windows.
    The previous evaluation manifest is required so its complete windows, with
    a four-second grid buffer on either side, cannot enter any new split.
    """
    root = Path(lusnar_root).absolute()
    previous_path = Path(previous_manifest_path).absolute()
    previous_bytes = previous_path.read_bytes()
    previous = json.loads(previous_bytes)
    previous_by_scene = defaultdict(list)
    for sample in previous["samples"]:
        if sample["dataset"] == "LuSNAR":
            previous_by_scene[sample["scene"]].append(sample)
    scene_names = [f"Moon_{i}" for i in range(1, 10)]
    if set(previous_by_scene) != set(scene_names):
        raise ValueError("Previous manifest must cover all nine LuSNAR scenes")

    fps, frames, buffer = 4, 16, 16
    period_ns = 250_000_000
    selected, scene_data, exclusions, anchors, audits = [], {}, {}, {}, []
    training_steps = []
    old_window_paths = set()

    def select_disjoint(candidates, count, description):
        # Earliest-finish greedy packing gives the maximum number of fixed-size
        # disjoint windows; evenly subsample this ordered packing deterministically.
        packed = []
        for anchor in candidates:
            if not packed or anchor - packed[-1] >= frames + 4:
                packed.append(anchor)
        if len(packed) < count:
            raise ValueError(f"{description}: need {count} windows, only {len(packed)} available")
        return [packed[int(i)] for i in np.linspace(0, len(packed) - 1, count).round().astype(int)]

    for number, name in enumerate(scene_names, 1):
        data = _read_lusnar_scene(root, name, fps)
        paths, timestamps, position, yaw, grid, indices, errors, _, audit = data
        scene_data[name] = data
        path_to_source = {str(Path(path).absolute()): i for i, path in enumerate(paths)}
        source_to_grid = defaultdict(list)
        for grid_index, source_index in enumerate(indices):
            source_to_grid[int(source_index)].append(grid_index)

        def previous_grid_index(path):
            source_index = path_to_source.get(str(Path(path).absolute()))
            matches = source_to_grid.get(source_index, [])
            if len(matches) != 1:
                raise ValueError(f"{name}: previous source is absent or ambiguous on the 4 Hz grid: {path}")
            return matches[0]

        old_intervals = {}
        for old in previous_by_scene[name]:
            context = old["context_source_paths"]
            targets = old.get("gt_source_paths") or []
            steps = old["prediction_steps"]
            if (len(context) != 4 or not targets or len(targets) != len(steps)
                    or old.get("nominal_fps") != fps
                    or any(not isinstance(step, int) or step < 1 for step in steps)):
                raise ValueError(f"{old['id']}: incompatible previous LuSNAR window")
            context_grid = [previous_grid_index(path) for path in context]
            anchor = context_grid[-1]
            if context_grid != list(range(anchor - 3, anchor + 1)):
                raise ValueError(f"{old['id']}: previous contexts do not form a consecutive window")
            if [previous_grid_index(path) for path in targets] != [anchor + step for step in steps]:
                raise ValueError(f"{old['id']}: previous target paths disagree with prediction steps")
            start, end = anchor - 3, anchor + max(steps)
            old_intervals.setdefault((start, end), []).append(old["id"])
            old_window_paths.update(paths[int(i)] for i in indices[start:end + 1])
        exclusions[name] = [
            {"old_window_start_grid_index": start, "old_window_end_grid_index": end,
             "excluded_start_grid_index": max(0, start - buffer),
             "excluded_end_grid_index": min(len(grid) - 1, end + buffer),
             "previous_sample_ids": sample_ids}
            for (start, end), sample_ids in sorted(old_intervals.items())
        ]
        valid = []
        for anchor in range(3, len(grid) - frames):
            start, end = anchor - 3, anchor + frames
            window = slice(start, end + 1)
            if (np.any(np.diff(indices[window]) <= 0)
                    or np.any(errors[window] > period_ns // 2)):
                continue
            if any(start <= item["excluded_end_grid_index"]
                   and end >= item["excluded_start_grid_index"] for item in exclusions[name]):
                continue
            valid.append(anchor)
        if number <= 5:
            train_end = int(0.65 * (len(grid) - 1))
            test_start = int(0.75 * (len(grid) - 1)) + 1
            choices = {
                "train": select_disjoint([i for i in valid if i + frames <= train_end], 6, f"{name}/train"),
                "same_scene_test": select_disjoint([i for i in valid if i - 3 >= test_start], 2, f"{name}/same_scene_test"),
            }
        else:
            split = "val" if number == 6 else "cross_scene_test"
            choices = {split: select_disjoint(valid, 8, f"{name}/{split}")}
        anchors[name] = choices
        for split, scene_anchors in choices.items():
            for window_number, anchor in enumerate(scene_anchors):
                selected.append((name, split, window_number, anchor))
                if split == "train":
                    xy = position[indices[anchor - 3:anchor + frames + 1]]
                    training_steps.extend(np.linalg.norm(np.diff(xy, axis=0), axis=1).tolist())
        audits.append(audit)

    nonzero_training_steps = [step for step in training_steps if step > 1e-6]
    spacing = float(np.median(nonzero_training_steps)) if nonzero_training_steps else float("nan")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Training windows contain no valid nonzero displacement")
    normalization = (
        "One fixed median nonzero XY step from adjacent 4 Hz frames inside the "
        "30 selected training windows; reused unchanged for validation and both test splits."
    )
    for audit in audits:
        audit["scene_median_waypoint_spacing_meters_for_audit_only"] = audit["waypoint_spacing_meters"]
        audit["waypoint_spacing_meters"] = spacing
        audit["normalization"] = normalization

    samples, path_owners = [], {}
    split_paths = defaultdict(set)
    for name, split, window_number, anchor in selected:
        paths, timestamps, position, yaw, grid, indices, _, _, audit = scene_data[name]
        source_anchor = int(indices[anchor])
        context_indices = indices[anchor - 3:anchor + 1]
        future_indices = indices[anchor + 1:anchor + frames + 1]
        context_paths = [paths[int(i)] for i in context_indices]
        target_paths = [paths[int(i)] for i in future_indices]
        prefix = f"lusnar_ft_{name}_{split}_window{window_number:02d}"
        for path in context_paths + target_paths:
            if path in path_owners or path in old_window_paths:
                raise ValueError(f"Source-frame leakage detected: {prefix}: {path}")
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            path_owners[path] = prefix
            split_paths[split].add(path)
        metric_actions = [local_action(position[source_anchor], yaw[source_anchor],
                                       position[int(i)], yaw[int(i)]) for i in future_indices]
        actions = np.asarray(metric_actions)
        actions[:, :2] /= spacing
        previous_indices = indices[anchor:anchor + frames]
        incremental_metric_actions = [local_action(position[int(a)], yaw[int(a)],
                                                   position[int(b)], yaw[int(b)])
                                      for a, b in zip(previous_indices, future_indices)]
        incremental_actions = np.asarray(incremental_metric_actions)
        incremental_actions[:, :2] /= spacing
        if (actions.shape != (frames, 3) or not np.isfinite(actions).all()
                or incremental_actions.shape != (frames, 3) or not np.isfinite(incremental_actions).all()):
            raise ValueError(f"Invalid actions: {prefix}")
        samples.append({
            "id": prefix + "_direct", "dataset": "LuSNAR", "split": split,
            "scene": name, "window": window_number, "protocol": "direct", "noise_key": prefix,
            "context_source_paths": context_paths, "gt_source_paths": target_paths,
            "context_timestamps_ns": [int(timestamps[i]) for i in context_indices],
            "gt_timestamps_ns": [int(timestamps[i]) for i in future_indices],
            "source_anchor_index": source_anchor, "source_grid_anchor_index": anchor,
            "anchor_timestamp_ns": int(timestamps[source_anchor]),
            "prediction_steps": list(range(1, frames + 1)), "actions": actions.tolist(),
            "metric_actions": [action.tolist() for action in metric_actions],
            "incremental_actions": incremental_actions.tolist(),
            "incremental_metric_actions": [action.tolist() for action in incremental_metric_actions],
            "incremental_action_reference": "Each next recorded body pose relative to the immediately preceding recorded body pose",
            "waypoint_spacing_meters": spacing, "nominal_fps": fps,
            "context_grid_offsets_seconds": [-3 / fps, -2 / fps, -1 / fps, 0.0],
            "actual_prediction_offsets_seconds": [int(timestamps[i] - timestamps[source_anchor]) / 1e9
                                                  for i in future_indices],
            "actual_incremental_offsets_seconds": [int(timestamps[b] - timestamps[a]) / 1e9
                                                   for a, b in zip(previous_indices, future_indices)],
            "action_translation_units": "meters divided by fixed training-only waypoint spacing",
            "action_yaw_units": "radians, positive left", "coordinate_convention": COORDINATE_NOTE,
            "pose_provenance": audit["source_gt_path"],
            "action_reference": "Each target pose relative to the fixed current rover body pose",
        })

    expected_counts = {"train": 30, "val": 8, "same_scene_test": 10, "cross_scene_test": 24}
    counts = {split: sum(sample["split"] == split for sample in samples) for split in expected_counts}
    if counts != expected_counts or len(path_owners) != 72 * 20:
        raise ValueError(f"Unexpected split size or source-frame overlap: {counts}")
    for left in expected_counts:
        for right in expected_counts:
            if left != right and split_paths[left].intersection(split_paths[right]):
                raise ValueError(f"Input/target leakage between {left} and {right}")
    gaps = {}
    for name in scene_names[:5]:
        _, timestamps, _, _, _, indices, _, _, _ = scene_data[name]
        last_train = indices[max(anchors[name]["train"]) + frames]
        first_test = indices[min(anchors[name]["same_scene_test"]) - 3]
        gaps[name] = int(timestamps[first_test] - timestamps[last_train]) / 1e9
    metadata = {
        "schema_version": 1, "protocol": "lusnar_few_shot_v1", "lusnar_root": str(root),
        "parameters": {"fps": fps, "context_frames": 4, "future_frames": frames,
                       "previous_window_buffer_grid_steps": buffer},
        "split_rules": {
            "train": "Moon_1..5, complete windows inside first 65% of each 4 Hz grid, six windows per scene",
            "val": "Moon_6, eight windows; never used to fit action scale",
            "same_scene_test": "Moon_1..5, complete windows strictly after 75% of each grid, two windows per scene",
            "cross_scene_test": "Moon_7..9, eight windows per scene",
            "selection": "Validate unique nearest-source frames within 125 ms; exclude old complete windows plus 16 grid steps on each side; earliest-finish disjoint packing, then rounded linspace subsampling",
            "no_leakage": "All 20 input/target source paths are unique across every selected window and split; no selected source occurs in any complete previous evaluation window",
        },
        "split_window_counts": counts,
        "split_target_pair_counts": {split: count * frames for split, count in counts.items()},
        "split_unique_source_image_counts": {split: len(split_paths[split]) for split in expected_counts},
        "selected_grid_anchor_indices": anchors, "total_unique_source_images": len(path_owners),
        "minimum_same_scene_train_test_gap_seconds": gaps,
        "waypoint_spacing_meters": spacing, "normalization": normalization,
        "training_scale_adjacent_step_count": len(training_steps),
        "training_scale_nonzero_step_count": len(nonzero_training_steps),
        "previous_manifest_path": str(previous_path),
        "previous_manifest_sha256": hashlib.sha256(previous_bytes).hexdigest(),
        "previous_evaluation_exclusion_ranges": exclusions,
        "source_annotation_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for name in scene_names
                                     for path in (root / name / "gt.txt", root / name / "image0/color_timestamp.txt")},
        "lusnar": audits,
        "limitations": [
            "Small, fixed exploratory split; temporal separation does not prove absence of revisited places within a scene.",
            "LuSNAR uses documented planar body-coordinate and axis assumptions; height, roll, pitch and camera mounting are not modeled.",
            "The nominal 4 Hz grid uses nearest captured images; actual timestamps and offsets are retained.",
            "No SynMars-TW images or pseudo-action labels are included.",
        ],
    }
    return {"samples": samples, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lusnar-root", type=Path, default=Path("/file_system/nas/algorithm/dujun.nie/datasets/LuSNAR"))
    parser.add_argument("--synmars-root", type=Path, default=Path("/file_system/nas/algorithm/dujun.nie/datasets/SynMars-TW"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows-per-scene", type=int, default=4)
    parser.add_argument("--synmars-images", type=int, default=12)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=4)
    args = parser.parse_args()
    samples, metadata = build_samples(args.lusnar_root, args.synmars_root, args.windows_per_scene,
                                      args.synmars_images, args.frames, args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"samples": samples, "metadata": metadata}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "samples": len(samples),
                      "lusnar": sum(x["dataset"] == "LuSNAR" for x in samples),
                      "synmars": sum(x["dataset"] == "SynMars-TW" for x in samples)}), flush=True)


if __name__ == "__main__":
    main()
