"""Reproducible, coverage-oriented sampling recipes for NavAnywhere.

The recipe is intentionally a small JSON inventory rather than an expanded list
of every training pair.  It freezes the trajectory/frame index set and the
versioned deterministic sampling algorithms.  TimePT, GeoPT, IDMPT, and
LatentPT can therefore consume the same file without coupling the recipe to an
action representation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


RECIPE_SCHEMA_VERSION = 1
RECIPE_FORMAT = "navanywhere_sampling_recipe"
OBSERVATION_SAMPLING = "source_trajectory_balanced_affine_v1"
GOAL_SAMPLING = "signed_offset_stratified_v1"
DEFAULT_GOAL_STRATA = ((-64, -9), (-8, -1), (0, 8), (9, 64))
FRAME_SCAN_ATTEMPTS = 3
FRAME_SCAN_RETRY_DELAY_SECONDS = 0.25


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_indices_sha256(frame_indices: Sequence[int] | np.ndarray) -> str:
    values = np.asarray(frame_indices, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _frame_index(filename: str) -> int:
    matches = re.findall(r"-?\d+", Path(filename).stem)
    if not matches:
        raise ValueError(f"Cannot infer integer frame index from {filename!r}")
    return int(matches[-1])


def _scan_trajectory_frames_once(path: str) -> list[tuple[int, str]]:
    # Some NAS clients can occasionally yield the same directory entry more
    # than once in a single readdir pass.  Collapse only byte-for-byte identical
    # records here; distinct paths that map to the same frame index are still
    # rejected by ``scan_trajectory_frames`` below.
    frames = list({
        (_frame_index(entry.name), os.path.realpath(entry.path))
        for entry in os.scandir(path)
        if entry.is_file(follow_symlinks=False)
        and entry.name.lower().endswith(".jpg")
    })
    frames.sort(key=lambda item: (item[0], item[1]))
    return frames


def scan_trajectory_frames(path: str | os.PathLike[str]) -> list[tuple[int, str]]:
    """Return sorted ``(frame_index, absolute_path)`` JPEG records.

    NAS directory listings can rarely return a transient duplicate entry. Retry
    that specific failure a few times, while still rejecting persistent duplicate
    frame indices as corrupt input.
    """
    path = os.path.realpath(os.path.expanduser(os.fspath(path)))
    duplicate_indices: list[int] = []
    for attempt in range(FRAME_SCAN_ATTEMPTS):
        frames = _scan_trajectory_frames_once(path)
        indices = [index for index, _ in frames]
        seen: set[int] = set()
        duplicates: set[int] = set()
        for index in indices:
            if index in seen:
                duplicates.add(index)
            seen.add(index)
        duplicate_indices = sorted(duplicates)
        if not duplicate_indices:
            return frames
        if attempt + 1 < FRAME_SCAN_ATTEMPTS:
            time.sleep(FRAME_SCAN_RETRY_DELAY_SECONDS * (attempt + 1))
    preview = duplicate_indices[:8]
    suffix = "..." if len(duplicate_indices) > len(preview) else ""
    raise ValueError(
        "Duplicate NavAnywhere frame indices under "
        f"{path}: {preview}{suffix}; persisted across {FRAME_SCAN_ATTEMPTS} scans"
    )


def discover_inventory(
    root: str | os.PathLike[str],
    *,
    context_size: int,
    source_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Discover the exact immediate ``source/trajectory/*.jpg`` inventory."""
    root = os.path.realpath(os.path.expanduser(os.fspath(root)))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"NavAnywhere root does not exist: {root}")
    if int(context_size) < 1:
        raise ValueError("context_size must be positive")
    selected_sources = None if source_ids is None else set(map(str, source_ids))
    source_entries = sorted(
        (
            entry
            for entry in os.scandir(root)
            if entry.is_dir(follow_symlinks=False)
            and (selected_sources is None or entry.name in selected_sources)
        ),
        key=lambda entry: entry.name,
    )
    if selected_sources is not None:
        found = {entry.name for entry in source_entries}
        missing = sorted(selected_sources - found)
        if missing:
            raise FileNotFoundError(f"NavAnywhere sources do not exist: {missing}")

    inventory: list[dict[str, Any]] = []
    progress = os.environ.get("NAVANYWHERE_RECIPE_PROGRESS", "0") == "1"
    started = time.monotonic()
    scanned_frames = 0
    last_report = started
    for source_number, source in enumerate(source_entries, 1):
        if progress:
            print(f"[recipe] source {source_number}/{len(source_entries)}: {source.name}; listing trajectories", file=sys.stderr, flush=True)
        trajectories = sorted(
            (
                entry
                for entry in os.scandir(source.path)
                if entry.is_dir(follow_symlinks=False)
            ),
            key=lambda entry: entry.name,
        )
        if progress:
            print(f"[recipe] {source.name}: 0/{len(trajectories)} trajectories", file=sys.stderr, flush=True)
        for trajectory_number, trajectory in enumerate(trajectories, 1):
            frames = scan_trajectory_frames(trajectory.path)
            scanned_frames += len(frames)
            now = time.monotonic()
            if progress and (now - last_report >= 30 or trajectory_number == len(trajectories)):
                print(
                    f"[recipe] source {source_number}/{len(source_entries)} {source.name}: "
                    f"{trajectory_number}/{len(trajectories)} trajectories; "
                    f"total_frames={scanned_frames}; elapsed={now - started:.0f}s",
                    file=sys.stderr, flush=True,
                )
                last_report = now
            observation_count = max(0, len(frames) - int(context_size) + 1)
            if observation_count == 0:
                continue
            indices = [index for index, _ in frames]
            inventory.append(
                {
                    "source_id": source.name,
                    "trajectory_id": trajectory.name,
                    "frame_count": len(frames),
                    "observation_count": observation_count,
                    "first_frame_index": int(indices[0]),
                    "last_frame_index": int(indices[-1]),
                    "frame_indices_sha256": frame_indices_sha256(indices),
                }
            )
    if not inventory:
        raise ValueError(f"No usable NavAnywhere trajectories found under {root}")
    return inventory


def build_sampling_recipe(
    root: str | os.PathLike[str],
    *,
    seed: int,
    context_size: int = 4,
    goals_per_obs: int = 4,
    min_frame_offset: int = -64,
    max_frame_offset: int = 64,
    samples_per_epoch: int = 0,
    source_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic recipe with equal source/trajectory exposure.

    Each logical epoch contains ``samples_per_epoch`` samples.  Sources are
    round-robin balanced; trajectories inside each source are round-robin
    balanced; observations inside a trajectory follow a full-cycle affine
    permutation.  DDP may shuffle these logical slots, but it cannot change the
    selected sample multiset or goal offsets.
    """
    context_size = int(context_size)
    goals_per_obs = int(goals_per_obs)
    if goals_per_obs < 1:
        raise ValueError("goals_per_obs must be positive")
    if (int(min_frame_offset), int(max_frame_offset)) != (-64, 64):
        raise ValueError("NavAnywhere recipes retain the complete [-64,64] range")
    inventory = discover_inventory(
        root, context_size=context_size, source_ids=source_ids
    )
    return build_sampling_recipe_from_inventory(
        inventory,
        seed=seed,
        context_size=context_size,
        goals_per_obs=goals_per_obs,
        samples_per_epoch=samples_per_epoch,
    )


def build_sampling_recipe_from_inventory(
    inventory: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    context_size: int = 4,
    goals_per_obs: int = 4,
    samples_per_epoch: int = 0,
) -> dict[str, Any]:
    """Build a recipe from an already fingerprinted trajectory inventory.

    This is used for deterministic trajectory-level train/validation splits.
    It deliberately retains the exact v1 sampling algorithms and recipe
    schema; only the supplied trajectory inventory differs.
    """

    context_size = int(context_size)
    goals_per_obs = int(goals_per_obs)
    if context_size < 1 or goals_per_obs < 1:
        raise ValueError("context_size and goals_per_obs must be positive")
    normalized = [dict(item) for item in inventory]
    identities = [
        (str(item.get("source_id")), str(item.get("trajectory_id")))
        for item in normalized
    ]
    if not normalized:
        raise ValueError("Sampling recipe inventory must not be empty")
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise ValueError("Sampling recipe inventory must be sorted and unique")
    for item, identity in zip(normalized, identities):
        frame_count = int(item.get("frame_count", 0))
        observation_count = int(item.get("observation_count", 0))
        if frame_count < context_size or observation_count < 1:
            raise ValueError(f"Invalid trajectory inventory entry {identity}")
        if observation_count != frame_count - context_size + 1:
            raise ValueError(
                f"Trajectory observation count does not match context size for {identity}"
            )
        if len(str(item.get("frame_indices_sha256", ""))) != 64:
            raise ValueError(f"Invalid frame fingerprint for {identity}")

    total_observations = sum(int(item["observation_count"]) for item in normalized)
    samples_per_epoch = int(samples_per_epoch) or total_observations
    if samples_per_epoch < 1:
        raise ValueError("samples_per_epoch must be positive")
    source_summary: dict[str, dict[str, int]] = {}
    for item in normalized:
        summary = source_summary.setdefault(
            item["source_id"], {"trajectories": 0, "observations": 0}
        )
        summary["trajectories"] += 1
        summary["observations"] += int(item["observation_count"])
    inventory_digest = hashlib.sha256(canonical_json(normalized)).hexdigest()
    return {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "format": RECIPE_FORMAT,
        "seed": int(seed),
        "context_size": context_size,
        "goals_per_obs": goals_per_obs,
        "frame_offset_range": [-64, 64],
        "samples_per_epoch": samples_per_epoch,
        "observation_sampling": {
            "strategy": OBSERVATION_SAMPLING,
            "source_weighting": "uniform",
            "trajectory_weighting_within_source": "uniform",
            "observation_cycle": "coprime_affine_permutation",
        },
        "goal_sampling": {
            "strategy": GOAL_SAMPLING,
            "strata": [list(bounds) for bounds in DEFAULT_GOAL_STRATA],
            "without_replacement_when_possible": True,
        },
        "inventory_sha256": inventory_digest,
        "totals": {
            "sources": len(source_summary),
            "trajectories": len(normalized),
            "observations": total_observations,
            "samples_per_epoch": samples_per_epoch,
        },
        "sources": source_summary,
        "trajectories": normalized,
    }


def validate_sampling_recipe(
    recipe: Mapping[str, Any],
    *,
    seed: int | None = None,
    context_size: int | None = None,
    goals_per_obs: int | None = None,
    frame_offset_range: Sequence[int] | None = None,
    samples_per_epoch: int | None = None,
) -> dict[str, Any]:
    """Validate structure, fingerprints, and optional runtime invariants."""
    value = dict(recipe)
    if int(value.get("schema_version", 0)) != RECIPE_SCHEMA_VERSION:
        raise ValueError("Unsupported NavAnywhere sampling recipe schema_version")
    if value.get("format") != RECIPE_FORMAT:
        raise ValueError("Invalid NavAnywhere sampling recipe format")
    trajectories = value.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("Sampling recipe trajectories must be a non-empty list")
    identities = []
    for item in trajectories:
        if not isinstance(item, Mapping):
            raise TypeError("Sampling recipe trajectory entries must be mappings")
        identity = (str(item.get("source_id")), str(item.get("trajectory_id")))
        identities.append(identity)
        if int(item.get("frame_count", 0)) < 1:
            raise ValueError(f"Invalid frame_count for recipe trajectory {identity}")
        if int(item.get("observation_count", 0)) < 1:
            raise ValueError(
                f"Invalid observation_count for recipe trajectory {identity}"
            )
        fingerprint = str(item.get("frame_indices_sha256", ""))
        if len(fingerprint) != 64:
            raise ValueError(
                f"Invalid frame index fingerprint for recipe trajectory {identity}"
            )
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise ValueError("Recipe trajectories must be sorted and unique")
    digest = hashlib.sha256(canonical_json(trajectories)).hexdigest()
    if digest != value.get("inventory_sha256"):
        raise ValueError("Sampling recipe inventory fingerprint is corrupt")
    if value.get("observation_sampling", {}).get("strategy") != OBSERVATION_SAMPLING:
        raise ValueError("Unsupported NavAnywhere observation sampling strategy")
    if value.get("goal_sampling", {}).get("strategy") != GOAL_SAMPLING:
        raise ValueError("Unsupported NavAnywhere goal sampling strategy")
    strata = value.get("goal_sampling", {}).get("strata")
    if strata != [list(bounds) for bounds in DEFAULT_GOAL_STRATA]:
        raise ValueError("NavAnywhere signed offset strata were changed")
    if int(value.get("samples_per_epoch", 0)) < 1:
        raise ValueError("Recipe samples_per_epoch must be positive")
    if samples_per_epoch is not None and int(value["samples_per_epoch"]) != int(
        samples_per_epoch
    ):
        raise ValueError(
            "Sampling recipe samples_per_epoch="
            f"{value['samples_per_epoch']!r}, expected {samples_per_epoch!r}"
        )
    expected = (
        ("seed", seed),
        ("context_size", context_size),
        ("goals_per_obs", goals_per_obs),
    )
    for field, requested in expected:
        if requested is not None and int(value.get(field)) != int(requested):
            raise ValueError(
                f"Sampling recipe {field}={value.get(field)!r}, expected {requested!r}"
            )
    if frame_offset_range is not None and list(map(int, frame_offset_range)) != list(
        map(int, value.get("frame_offset_range", ()))
    ):
        raise ValueError("Sampling recipe frame_offset_range mismatch")
    return value


def load_sampling_recipe(
    path: str | os.PathLike[str], **validation: Any
) -> tuple[dict[str, Any], str, str]:
    resolved = os.path.realpath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"NavAnywhere sampling recipe does not exist: {resolved}")
    with open(resolved, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError("NavAnywhere sampling recipe must contain a JSON object")
    return validate_sampling_recipe(value, **validation), resolved, sha256_file(resolved)


def write_sampling_recipe(
    path: str | os.PathLike[str], recipe: Mapping[str, Any], *, overwrite: bool = False
) -> str:
    value = validate_sampling_recipe(recipe)
    resolved = os.path.realpath(os.path.expanduser(os.fspath(path)))
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    if os.path.exists(resolved) and not overwrite:
        raise FileExistsError(
            f"Sampling recipe already exists: {resolved}; pass --overwrite explicitly"
        )
    temporary = f"{resolved}.tmp.{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return resolved


def coprime_stride(length: int, identity: str, seed: int) -> tuple[int, int]:
    """Return deterministic affine-permutation ``(stride, shift)`` values."""
    length = int(length)
    if length < 1:
        raise ValueError("Affine permutation length must be positive")
    digest = hashlib.sha256(f"{seed}\0{identity}".encode("utf-8")).digest()
    if length == 1:
        return 1, 0
    stride = int.from_bytes(digest[:8], "little") % length or 1
    while math.gcd(stride, length) != 1:
        stride = (stride + 1) % length or 1
    shift = int.from_bytes(digest[8:16], "little") % length
    return stride, shift


def affine_observation_slot(
    *,
    epoch: int,
    trajectory_draw: int,
    trajectory_draws_per_epoch: int,
    observation_count: int,
    stride: int,
    shift: int,
) -> int:
    """Return the recipe observation slot for one logical dataset index.

    This is shared by the training dataset and latent-action plan generation so
    the plan cannot drift from the affine observation cycle used in training.
    """

    if int(epoch) < 0 or int(trajectory_draw) < 0:
        raise ValueError("Recipe epoch and trajectory draw must be non-negative")
    if int(trajectory_draws_per_epoch) < 1 or int(observation_count) < 1:
        raise ValueError("Recipe draw and observation counts must be positive")
    global_draw = (
        int(epoch) * int(trajectory_draws_per_epoch) + int(trajectory_draw)
    )
    return (
        int(shift) + int(stride) * global_draw
    ) % int(observation_count)


def sample_recipe_goal_offsets(
    available_offsets: Sequence[int] | np.ndarray,
    *,
    seed: int,
    epoch: int,
    dataset_index: int,
    goals_per_obs: int,
    strata: Sequence[Sequence[int]] = DEFAULT_GOAL_STRATA,
) -> np.ndarray:
    """Sample the exact stratified offsets used by NavAnywhere training.

    Keep this function as the single source of truth for both
    ``NavAnywhereDataset`` and plan-only latent-action accounting.  In
    particular, it intentionally preserves the SeedSequence construction and
    the fallback behavior at trajectory boundaries.
    """

    available = np.asarray(available_offsets, dtype=np.int64)
    if available.ndim != 1 or available.size == 0:
        raise ValueError("available_offsets must be a non-empty vector")
    if np.any(np.diff(available) <= 0):
        raise ValueError("available_offsets must be strictly increasing")
    goals_per_obs = int(goals_per_obs)
    if goals_per_obs < 1:
        raise ValueError("goals_per_obs must be positive")
    normalized_strata = tuple(tuple(map(int, bounds)) for bounds in strata)
    if normalized_strata != DEFAULT_GOAL_STRATA:
        raise ValueError("NavAnywhere signed offset strata were changed")

    generator = np.random.default_rng(
        np.random.SeedSequence((int(seed), int(epoch), int(dataset_index)))
    )
    selected: list[int] = []
    unused = set(map(int, available.tolist()))
    available_values = list(map(int, available.tolist()))
    for goal_index in range(goals_per_obs):
        low, high = normalized_strata[goal_index % len(normalized_strata)]
        candidates = [value for value in unused if low <= value <= high]
        if not candidates:
            candidates = [
                value for value in available_values if low <= value <= high
            ]
        if not candidates:
            # Preserve the training dataset's boundary fallback ordering.  For
            # integer offsets this is deterministic in the pinned runtime and
            # is independently checked against completed TimePT/GeoPT logs by
            # the plan-only command.
            candidates = list(unused) or available_values
        candidates.sort()
        choice = candidates[int(generator.integers(0, len(candidates)))]
        selected.append(choice)
        unused.discard(choice)
    return np.asarray(selected, dtype=np.int64)


__all__ = [
    "DEFAULT_GOAL_STRATA",
    "GOAL_SAMPLING",
    "OBSERVATION_SAMPLING",
    "RECIPE_FORMAT",
    "RECIPE_SCHEMA_VERSION",
    "build_sampling_recipe",
    "affine_observation_slot",
    "canonical_json",
    "coprime_stride",
    "discover_inventory",
    "frame_indices_sha256",
    "load_sampling_recipe",
    "scan_trajectory_frames",
    "sample_recipe_goal_offsets",
    "sha256_file",
    "validate_sampling_recipe",
    "write_sampling_recipe",
]
