"""Navigation frame-pair data for labeled and frame-only training sources.

Pose-labeled sources are restricted to an explicit split manifest.  Frame-only
sources discover their immediate trajectory directories.  Mixed profiles use a
deterministic finite schedule that balances integer temporal offsets, applies a
square-root-style source weight, and visits valid anchors without replacement
inside each source/offset cycle.
"""

from __future__ import annotations

import bisect
import fcntl
import hashlib
import json
import math
import operator
import os
import pickle
import random
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

try:
    from lightning import LightningDataModule
except ImportError:  # pragma: no cover - exercised only in lightweight data-only installs
    # The LAM training environment installs Lightning.  Keeping the data-only
    # module importable is useful for dataset audits and unit tests on login
    # nodes where that optional training dependency is absent.
    class LightningDataModule:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def save_hyperparameters(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs


_ACTION_MIN = (-2.5, -4.0)
_ACTION_MAX = (5.0, 4.0)
_UINT64_MASK = (1 << 64) - 1
_INDEX_CACHE_VERSION = 3
_DINO_FEATURE_CACHE_VERSION = 1
_DINO_FEATURE_DTYPE = np.dtype("<f2")
_DINO_PATCH_COUNT = 300
_DINO_FEATURE_DIM = 1024
_DINO_MODEL_NAME = "dinov3_vitl16"
_DINO_WEIGHTS_SHA256 = (
    "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"
)
_DINO_SOURCE_REVISION = "346f38fee679c56a6888f91c51670fae61d364e0"


class NavigationFrameDecodeError(RuntimeError):
    """A single navigation frame exists but cannot be decoded for training."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Cannot decode navigation frame {path}: {reason}")


class NavigationTrajectoryValidationError(RuntimeError):
    """A trajectory recorded in an immutable index is no longer readable."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(reason)


def navigation_skip_bad_data_enabled() -> bool:
    """Return whether the opt-in navigation data recovery policy is active."""

    return os.environ.get("NAV_LAM_SKIP_BAD_DATA", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def report_navigation_data_warning(
    message: str,
    *,
    kind: str,
    source: str | None = None,
    path: Path | str | None = None,
) -> None:
    """Route one data warning to the suite JSONL without polluting stderr."""

    warning_log = os.environ.get("NAV_LAM_WARNING_LOG", "").strip()
    if not warning_log:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "kind": kind,
        "source": source,
        "path": None if path is None else str(path),
        "message": message,
    }
    warning_path = Path(warning_log).expanduser()
    try:
        warning_path.parent.mkdir(parents=True, exist_ok=True)
        with warning_path.open("a", encoding="utf-8", newline="\n") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (OSError, UnicodeError) as exc:
        warnings.warn(
            f"{message}; additionally failed to write dedicated navigation data "
            f"warning log {warning_path}: {exc}",
            RuntimeWarning,
            stacklevel=3,
        )


@dataclass(frozen=True)
class _Trajectory:
    name: str
    directory: Path
    positions: np.ndarray | None
    yaw: np.ndarray | None
    frame_count: int
    metadata_size: int
    metadata_mtime_ns: int
    directory_mtime_ns: int

    @property
    def length(self) -> int:
        return self.frame_count


@dataclass(frozen=True)
class _Source:
    display_name: str
    data_root: Path
    split_manifest: Path | None
    waypoint_spacing: float
    format: str
    has_action: bool
    trajectories: tuple[_Trajectory, ...]
    cumulative_anchor_counts: tuple[int, ...]

    @property
    def anchor_count(self) -> int:
        return self.cumulative_anchor_counts[-1]

    def anchor(self, source_anchor_index: int) -> tuple[_Trajectory, int]:
        trajectory_index = bisect.bisect_right(
            self.cumulative_anchor_counts, source_anchor_index
        )
        previous_count = (
            self.cumulative_anchor_counts[trajectory_index - 1]
            if trajectory_index
            else 0
        )
        return (
            self.trajectories[trajectory_index],
            source_anchor_index - previous_count,
        )

    def valid_anchor_count(self, offset: int) -> int:
        distance = abs(offset)
        return sum(max(trajectory.length - distance, 0) for trajectory in self.trajectories)

    def valid_anchor(self, offset: int, valid_index: int) -> tuple[_Trajectory, int]:
        distance = abs(offset)
        for trajectory in self.trajectories:
            count = max(trajectory.length - distance, 0)
            if valid_index < count:
                anchor = valid_index if offset >= 0 else valid_index + distance
                return trajectory, anchor
            valid_index -= count
        raise IndexError(valid_index)


class _DINOFeatureCache:
    """Validated, lazily mmap'ed monolithic FP16 feature files."""

    def __init__(
        self,
        root: Path,
        sources: Sequence[_Source],
        *,
        image_height: int,
        image_width: int,
    ) -> None:
        self.root = root
        metadata_path = root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Complete DINOv3 feature-cache metadata does not exist: {metadata_path}"
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot read DINOv3 feature-cache metadata {metadata_path}: {exc}"
            ) from exc
        if not isinstance(metadata, Mapping):
            raise TypeError(f"Malformed DINOv3 feature-cache metadata: {metadata_path}")

        expected_identity = {
            "version": _DINO_FEATURE_CACHE_VERSION,
            "complete": True,
            "model_name": _DINO_MODEL_NAME,
            "weights_sha256": _DINO_WEIGHTS_SHA256,
            "source_revision": _DINO_SOURCE_REVISION,
            "dtype": "float16",
            "patch_count": _DINO_PATCH_COUNT,
            "feature_dim": _DINO_FEATURE_DIM,
            "image_height": image_height,
            "image_width": image_width,
            "color_aug": False,
        }
        for key, expected in expected_identity.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"DINOv3 feature cache {metadata_path} has {key}="
                    f"{metadata.get(key)!r}; expected {expected!r}"
                )

        cached_sources = metadata.get("sources")
        if not isinstance(cached_sources, list) or len(cached_sources) != len(sources):
            raise RuntimeError(
                f"DINOv3 feature cache must describe {len(sources)} sources"
            )

        feature_paths: list[Path] = []
        trajectory_offsets: list[dict[str, int]] = []
        for source_id, (source, cached_source) in enumerate(
            zip(sources, cached_sources, strict=True)
        ):
            if not isinstance(cached_source, Mapping):
                raise TypeError(
                    f"Malformed DINOv3 feature-cache source {source_id}: {metadata_path}"
                )
            expected_source = {
                "dataset_id": source_id,
                "display_name": source.display_name,
                "data_root": str(source.data_root),
                "split_manifest": str(source.split_manifest),
                "anchor_count": source.anchor_count,
            }
            for key, expected in expected_source.items():
                if cached_source.get(key) != expected:
                    raise RuntimeError(
                        f"DINOv3 feature-cache source {source_id} has stale {key}: "
                        f"{cached_source.get(key)!r} != {expected!r}"
                    )

            cached_trajectories = cached_source.get("trajectories")
            if not isinstance(cached_trajectories, list) or len(
                cached_trajectories
            ) != len(source.trajectories):
                raise RuntimeError(
                    f"DINOv3 feature-cache source {source_id} trajectory count is stale"
                )
            offsets: dict[str, int] = {}
            expected_offset = 0
            for trajectory, cached_trajectory in zip(
                source.trajectories, cached_trajectories, strict=True
            ):
                expected_trajectory = {
                    "name": trajectory.name,
                    "offset": expected_offset,
                    "length": trajectory.length,
                    "metadata_size": trajectory.metadata_size,
                    "metadata_mtime_ns": trajectory.metadata_mtime_ns,
                    "directory_mtime_ns": trajectory.directory_mtime_ns,
                }
                if not isinstance(cached_trajectory, Mapping) or any(
                    cached_trajectory.get(key) != value
                    for key, value in expected_trajectory.items()
                ):
                    raise RuntimeError(
                        "DINOv3 feature cache is stale for trajectory "
                        f"{source.display_name}/{trajectory.name}"
                    )
                offsets[trajectory.name] = expected_offset
                expected_offset += trajectory.length

            feature_file = cached_source.get("feature_file")
            if (
                not isinstance(feature_file, str)
                or Path(feature_file).name != feature_file
            ):
                raise RuntimeError(
                    f"Unsafe DINOv3 feature-cache file for source {source_id}: "
                    f"{feature_file!r}"
                )
            feature_path = root / feature_file
            expected_size = (
                source.anchor_count
                * _DINO_PATCH_COUNT
                * _DINO_FEATURE_DIM
                * _DINO_FEATURE_DTYPE.itemsize
            )
            if not feature_path.is_file() or feature_path.stat().st_size != expected_size:
                actual = feature_path.stat().st_size if feature_path.exists() else None
                raise RuntimeError(
                    f"DINOv3 feature file has size {actual}, expected {expected_size}: "
                    f"{feature_path}"
                )
            feature_paths.append(feature_path)
            trajectory_offsets.append(offsets)

        self.metadata = dict(metadata)
        self._sources = tuple(sources)
        self._feature_paths = tuple(feature_paths)
        self._trajectory_offsets = tuple(trajectory_offsets)
        # Opening after DataLoader worker creation prevents inherited mmap state
        # and keeps the per-worker file-descriptor count fixed at three.
        self._arrays: list[np.memmap | None] = [None] * len(sources)

    def _array(self, source_id: int) -> np.memmap:
        array = self._arrays[source_id]
        if array is None:
            array = np.memmap(
                self._feature_paths[source_id],
                dtype=_DINO_FEATURE_DTYPE,
                mode="c",
                shape=(
                    self._sources[source_id].anchor_count,
                    _DINO_PATCH_COUNT,
                    _DINO_FEATURE_DIM,
                ),
            )
            self._arrays[source_id] = array
        return array

    def pair(
        self,
        source_id: int,
        trajectory: _Trajectory,
        anchor_frame: int,
        target_frame: int,
    ) -> Tensor:
        offset = self._trajectory_offsets[source_id][trajectory.name]
        # The DataLoader collator copies these two mmap views into its shared
        # batch buffer; there is no per-sample JPEG decode or DINO execution.
        array = self._array(source_id)
        return torch.stack(
            (
                torch.from_numpy(array[offset + anchor_frame]),
                torch.from_numpy(array[offset + target_frame]),
            ),
            dim=0,
        )


def _as_path(value: Any, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise TypeError(f"source_specs.{field} must be a non-empty path")
    return Path(value).expanduser().resolve()


def _source_root(raw_spec: Mapping[str, Any], source_id: int) -> Path:
    """Accept the public ``root`` spelling and the legacy ``data_root`` alias."""

    has_root = raw_spec.get("root") not in (None, "")
    has_data_root = raw_spec.get("data_root") not in (None, "")
    if not has_root and not has_data_root:
        raise ValueError(f"source_specs[{source_id}] is missing root/data_root")
    root = _as_path(
        raw_spec.get("root") if has_root else raw_spec.get("data_root"), "root"
    )
    if has_root and has_data_root:
        data_root = _as_path(raw_spec.get("data_root"), "data_root")
        if data_root != root:
            raise ValueError(
                f"source_specs[{source_id}] provides conflicting root and data_root"
            )
    return root


def _read_manifest(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Training split manifest does not exist: {path}")
    if any(part.lower() == "test" for part in path.parts) or path.name.lower().startswith(
        "test"
    ):
        raise ValueError(f"Refusing to use an apparent test manifest for training: {path}")

    try:
        names = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"Cannot read training split manifest {path}: {exc}") from exc
    if not names:
        raise ValueError(f"Training split manifest is empty: {path}")
    if len(set(names)) != len(names):
        raise ValueError(f"Training split manifest contains duplicate trajectories: {path}")
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe trajectory name {name!r} in manifest {path}")
    return names


def _read_action_label_manifest(path: Path) -> tuple[str, ...]:
    """Read an optional trajectory-level action-label allowlist.

    Unlike a training split, an action-label manifest is allowed to be empty:
    that is the explicit representation of the zero-label ablation.  Keeping
    this reader separate from :func:`_read_manifest` prevents an empty training
    split from ever being accepted accidentally.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Action-label manifest does not exist: {path}")
    try:
        names = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"Cannot read action-label manifest {path}: {exc}") from exc
    if len(set(names)) != len(names):
        raise ValueError(f"Action-label manifest contains duplicate trajectories: {path}")
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"Unsafe trajectory name {name!r} in action-label manifest {path}"
            )
    return names


def _load_trajectory(directory: Path, trajectory_name: str) -> _Trajectory:
    metadata_path = directory / "traj_data.pkl"
    if not metadata_path.is_file():
        raise RuntimeError(
            f"Existing trajectory is corrupt (missing traj_data.pkl): {directory}"
        )
    try:
        with metadata_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Cannot read trajectory metadata {metadata_path}: {exc}") from exc
    if not isinstance(payload, Mapping) or "position" not in payload or "yaw" not in payload:
        raise RuntimeError(
            f"Trajectory metadata must contain position and yaw: {metadata_path}"
        )

    try:
        positions = np.asarray(payload["position"], dtype=np.float64)
        yaw = np.asarray(payload["yaw"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid position/yaw arrays in {metadata_path}: {exc}") from exc
    if positions.ndim != 2 or positions.shape[1] < 2 or positions.shape[0] == 0:
        raise RuntimeError(
            f"position must have shape [L,2+] with L>0 in {metadata_path}; "
            f"got {positions.shape}"
        )
    if yaw.ndim == 2 and yaw.shape[1] == 1:
        yaw = yaw[:, 0]
    if yaw.ndim != 1 or yaw.shape[0] != positions.shape[0]:
        raise RuntimeError(
            f"yaw must have shape [L] or [L,1] matching position in {metadata_path}; "
            f"got position={positions.shape}, yaw={yaw.shape}"
        )
    if not np.isfinite(positions).all() or not np.isfinite(yaw).all():
        raise RuntimeError(f"position/yaw contains non-finite values: {metadata_path}")

    # ``position`` is authoritative in CompACT/NWM.  Do not glob images:
    # requiring exactly the expected 0..L-1 names catches internal gaps while
    # ignoring unrelated files in a trajectory directory.
    try:
        # A single directory enumeration is substantially cheaper than L
        # individual NAS lookups while still validating the authoritative
        # contiguous sequence.
        with os.scandir(directory) as entries:
            frame_names = {entry.name for entry in entries if entry.is_file()}
    except OSError as exc:
        raise RuntimeError(f"Cannot scan trajectory directory {directory}: {exc}") from exc
    for frame_index in range(int(positions.shape[0])):
        if f"{frame_index}.jpg" not in frame_names:
            raise RuntimeError(
                "Existing trajectory is corrupt (expected contiguous 0..L-1 "
                f"jpg files; missing frame {frame_index}): {directory}"
            )

    try:
        metadata_stat = metadata_path.stat()
        directory_stat = directory.stat()
    except OSError as exc:
        raise RuntimeError(f"Cannot stat validated trajectory {directory}: {exc}") from exc

    positions.setflags(write=False)
    yaw.setflags(write=False)
    return _Trajectory(
        trajectory_name,
        directory,
        positions,
        yaw,
        int(positions.shape[0]),
        metadata_stat.st_size,
        metadata_stat.st_mtime_ns,
        directory_stat.st_mtime_ns,
    )


def _load_frame_only_trajectory(directory: Path, trajectory_name: str) -> _Trajectory:
    """Audit one NavAnywhere trajectory containing contiguous numeric JPEGs."""

    try:
        with os.scandir(directory) as entries:
            numeric_frames = sorted(
                int(Path(entry.name).stem)
                for entry in entries
                if entry.is_file(follow_symlinks=False)
                and Path(entry.name).suffix.lower() == ".jpg"
                and Path(entry.name).stem.isdigit()
            )
        directory_stat = directory.stat()
    except OSError as exc:
        raise RuntimeError(f"Cannot scan frame-only trajectory {directory}: {exc}") from exc
    if not numeric_frames:
        raise RuntimeError(f"Frame-only trajectory contains no numeric JPEGs: {directory}")
    if numeric_frames != list(range(len(numeric_frames))):
        raise RuntimeError(
            "Frame-only trajectory must contain contiguous 0..L-1 JPEGs: "
            f"{directory} (first={numeric_frames[0]}, last={numeric_frames[-1]}, "
            f"count={len(numeric_frames)})"
        )
    return _Trajectory(
        trajectory_name,
        directory,
        None,
        None,
        len(numeric_frames),
        0,
        0,
        directory_stat.st_mtime_ns,
    )


def _source_format(raw_spec: Mapping[str, Any], source_id: int) -> tuple[str, bool]:
    source_format = str(raw_spec.get("format", "pose_labeled")).strip()
    if source_format not in {"pose_labeled", "frame_only"}:
        raise ValueError(
            f"source_specs[{source_id}].format must be pose_labeled or frame_only"
        )
    default_has_action = source_format == "pose_labeled"
    has_action = raw_spec.get("has_action", default_has_action)
    if not isinstance(has_action, bool):
        raise TypeError(f"source_specs[{source_id}].has_action must be bool")
    if has_action != default_has_action:
        raise ValueError(
            f"source_specs[{source_id}] has incompatible format={source_format!r} "
            f"and has_action={has_action!r}"
        )
    return source_format, has_action


def _trajectory_names(
    data_root: Path, raw_spec: Mapping[str, Any], source_format: str
) -> tuple[tuple[str, ...], Path | None]:
    manifest_value = raw_spec.get("split_manifest")
    if manifest_value not in (None, ""):
        manifest = _as_path(manifest_value, "split_manifest")
        return _read_manifest(manifest), manifest
    if source_format != "frame_only":
        raise ValueError("pose_labeled sources require split_manifest")
    try:
        names = tuple(
            sorted(
                entry.name
                for entry in os.scandir(data_root)
                if entry.is_dir(follow_symlinks=False)
            )
        )
    except OSError as exc:
        raise RuntimeError(f"Cannot discover trajectories below {data_root}: {exc}") from exc
    if not names:
        raise ValueError(f"Frame-only source has no trajectory directories: {data_root}")
    return names, None


def _source_specs_signature(source_specs: Sequence[Mapping[str, Any]]) -> str:
    """Hash the exact split inputs used to build a shared navigation index."""

    normalized = []
    for source_id, raw_spec in enumerate(source_specs):
        missing_fields = {"display_name"} - set(raw_spec)
        if missing_fields:
            raise ValueError(
                f"source_specs[{source_id}] is missing fields: {sorted(missing_fields)}"
            )
        data_root = _source_root(raw_spec, source_id)
        if not data_root.is_dir():
            raise FileNotFoundError(f"Navigation data root does not exist: {data_root}")
        source_format, has_action = _source_format(raw_spec, source_id)
        manifest_names, manifest = _trajectory_names(data_root, raw_spec, source_format)
        inventory = hashlib.sha256()
        direct_names = {name for name in manifest_names if len(Path(name).parts) == 1}
        direct_entries: dict[str, os.DirEntry[str]] = {}
        try:
            with os.scandir(data_root) as entries:
                direct_entries = {
                    entry.name: entry for entry in entries if entry.name in direct_names
                }
                # DirEntry stat data is no longer guaranteed usable after the
                # iterator closes on every platform, so consume it below while
                # retaining only primitive fingerprints.
                direct_fingerprints = {
                    name: (
                        entry.is_dir(follow_symlinks=False),
                        entry.stat(follow_symlinks=False).st_size,
                        entry.stat(follow_symlinks=False).st_mtime_ns,
                    )
                    for name, entry in direct_entries.items()
                }
        except OSError as exc:
            raise RuntimeError(f"Cannot inventory navigation data root {data_root}: {exc}") from exc
        for name in manifest_names:
            if name in direct_names:
                fingerprint = direct_fingerprints.get(name)
            else:
                path = data_root / name
                try:
                    path_stat = path.stat()
                    fingerprint = (
                        path.is_dir(),
                        path_stat.st_size,
                        path_stat.st_mtime_ns,
                    )
                except FileNotFoundError:
                    fingerprint = None
                except OSError as exc:
                    raise RuntimeError(
                        f"Cannot inventory navigation trajectory {path}: {exc}"
                    ) from exc
            inventory.update(name.encode("utf-8"))
            inventory.update(b"\0missing\0" if fingerprint is None else repr(fingerprint).encode("ascii"))
        normalized.append(
            {
                "data_root": str(data_root),
                # A trajectory appearing/disappearing below the source root
                # changes its directory mtime and invalidates the shared index.
                # Per-trajectory metadata/directory fingerprints are retained
                # in the cache and checked before that trajectory is sampled.
                "data_root_mtime_ns": data_root.stat().st_mtime_ns,
                "manifest_inventory_sha256": inventory.hexdigest(),
                "split_manifest": None if manifest is None else str(manifest),
                "split_sha256": (
                    None if manifest is None else hashlib.sha256(manifest.read_bytes()).hexdigest()
                ),
                "waypoint_spacing": float(raw_spec.get("waypoint_spacing", 1.0)),
                "display_name": str(raw_spec.get("display_name")),
                "format": source_format,
                "has_action": has_action,
            }
        )
    payload = json.dumps(
        {"version": _INDEX_CACHE_VERSION, "sources": normalized},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_spec_cache_identity(
    raw_spec: Mapping[str, Any], source_id: int
) -> tuple[str, str, str | None, float, str, bool]:
    """Return the cache-relevant source configuration without inspecting data."""

    if "display_name" not in raw_spec:
        raise ValueError(
            f"source_specs[{source_id}] is missing fields: ['display_name']"
        )
    display_name = str(raw_spec["display_name"]).strip()
    if not display_name:
        raise ValueError(f"source_specs[{source_id}].display_name must not be empty")
    data_root = _source_root(raw_spec, source_id)
    source_format, has_action = _source_format(raw_spec, source_id)
    manifest_value = raw_spec.get("split_manifest")
    if manifest_value in (None, ""):
        if source_format != "frame_only":
            raise ValueError("pose_labeled sources require split_manifest")
        manifest = None
    else:
        manifest = _as_path(manifest_value, "split_manifest")
    try:
        waypoint_spacing = float(raw_spec.get("waypoint_spacing", 1.0))
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"source_specs[{source_id}].waypoint_spacing must be numeric"
        ) from exc
    if not np.isfinite(waypoint_spacing) or waypoint_spacing <= 0:
        raise ValueError(
            f"source_specs[{source_id}].waypoint_spacing must be finite and positive"
        )
    return (
        display_name,
        str(data_root),
        None if manifest is None else str(manifest),
        waypoint_spacing,
        source_format,
        has_action,
    )


def _cached_sources_match_specs(
    sources: Sequence[_Source], source_specs: Sequence[Mapping[str, Any]]
) -> bool:
    """Check cache/config compatibility without walking trajectory directories."""

    if len(sources) != len(source_specs):
        return False
    requested = tuple(
        _source_spec_cache_identity(raw_spec, source_id)
        for source_id, raw_spec in enumerate(source_specs)
    )
    cached = tuple(
        (
            source.display_name,
            str(source.data_root),
            None if source.split_manifest is None else str(source.split_manifest),
            source.waypoint_spacing,
            source.format,
            source.has_action,
        )
        for source in sources
    )
    return cached == requested


def _read_index_cache(
    cache_path: Path,
    source_specs: Sequence[Mapping[str, Any]],
    *,
    expected_signature: str | None = None,
) -> tuple[tuple[_Source, ...], tuple[dict[str, Any], ...]] | None:
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Cannot read navigation index cache {cache_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"Malformed navigation index cache: {cache_path}")
    if payload.get("version") != _INDEX_CACHE_VERSION:
        return None
    if expected_signature is not None and payload.get("signature") != expected_signature:
        return None
    sources = payload.get("sources")
    audit_sources = payload.get("audit_sources")
    if (
        not isinstance(sources, tuple)
        or not sources
        or not all(isinstance(source, _Source) for source in sources)
        or not isinstance(audit_sources, tuple)
        or len(audit_sources) != len(sources)
    ):
        raise RuntimeError(f"Malformed navigation index cache contents: {cache_path}")
    if not _cached_sources_match_specs(sources, source_specs):
        return None
    return sources, audit_sources


def _write_index_cache(
    cache_path: Path,
    signature: str,
    sources: tuple[_Source, ...],
    audit_sources: tuple[dict[str, Any], ...],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            dir=cache_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            pickle.dump(
                {
                    "version": _INDEX_CACHE_VERSION,
                    "signature": signature,
                    "sources": sources,
                    "audit_sources": audit_sources,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, cache_path)
        os.chmod(cache_path, 0o644)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _seed_worker(worker_id: int, *, data_seed: int) -> None:
    # DataLoader assigns each worker a deterministic torch.initial_seed from
    # its generator.  Mix in the explicit data seed so this function remains
    # deterministic even if invoked outside DataLoader in a test/audit tool.
    seed = (torch.initial_seed() + int(data_seed) + int(worker_id)) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_navigation_frame(
    path: Path, *, image_height: int = 240, image_width: int = 320
) -> Tensor:
    """Decode and center-crop one frame exactly as navigation training does."""

    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise NavigationFrameDecodeError(path, str(exc)) from exc
    frame = torch.from_numpy(rgb).to(dtype=torch.float32).div_(255.0)
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise NavigationFrameDecodeError(path, f"invalid size {width}x{height}")
    target_ratio = 4.0 / 3.0
    current_ratio = width / height
    if current_ratio > target_ratio:
        crop_height = height
        crop_width = int(height * target_ratio)
    elif current_ratio < target_ratio:
        crop_height = int(width / target_ratio)
        crop_width = width
    else:
        crop_height, crop_width = height, width
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    frame = frame[top : top + crop_height, left : left + crop_width]
    frame = frame.permute(2, 0, 1).unsqueeze(0)
    frame = F.interpolate(
        frame,
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    )
    return frame.squeeze(0).permute(1, 2, 0).contiguous()


def _build_sources(
    source_specs: Sequence[Mapping[str, Any]],
) -> tuple[tuple[_Source, ...], tuple[dict[str, Any], ...]]:
    raw_audit_workers = os.environ.get("NAV_LAM_AUDIT_WORKERS", "1")
    try:
        audit_workers = int(raw_audit_workers)
    except ValueError as exc:
        raise ValueError("NAV_LAM_AUDIT_WORKERS must be a positive integer") from exc
    if audit_workers < 1:
        raise ValueError("NAV_LAM_AUDIT_WORKERS must be a positive integer")

    sources = []
    audit_sources = []
    for source_id, raw_spec in enumerate(source_specs):
        missing_fields = {"display_name"} - set(raw_spec)
        if missing_fields:
            raise ValueError(
                f"source_specs[{source_id}] is missing fields: {sorted(missing_fields)}"
            )
        data_root = _source_root(raw_spec, source_id)
        if not data_root.is_dir():
            raise FileNotFoundError(f"Navigation data root does not exist: {data_root}")
        source_format, has_action = _source_format(raw_spec, source_id)
        manifest_names, manifest = _trajectory_names(data_root, raw_spec, source_format)
        display_name = str(raw_spec["display_name"]).strip()
        if not display_name:
            raise ValueError(f"source_specs[{source_id}].display_name must not be empty")
        try:
            waypoint_spacing = float(raw_spec.get("waypoint_spacing", 1.0))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"source_specs[{source_id}].waypoint_spacing must be numeric"
            ) from exc
        if not np.isfinite(waypoint_spacing) or waypoint_spacing <= 0:
            raise ValueError(
                f"source_specs[{source_id}].waypoint_spacing must be finite and positive"
            )

        trajectories = []
        missing_names = []
        corrupt_trajectories: list[dict[str, str]] = []
        cumulative_counts = []
        anchor_count = 0

        def load_one_trajectory(
            trajectory_name: str,
        ) -> tuple[str, Path, _Trajectory | None, RuntimeError | None, bool]:
            trajectory_dir = (data_root / trajectory_name).resolve()
            try:
                trajectory_dir.relative_to(data_root)
            except ValueError as exc:
                raise ValueError(
                    f"Trajectory {trajectory_name!r} escapes data root {data_root}"
                ) from exc
            if not trajectory_dir.exists():
                return trajectory_name, trajectory_dir, None, None, True
            try:
                if not trajectory_dir.is_dir():
                    raise RuntimeError(
                        f"Existing trajectory path is not a directory: {trajectory_dir}"
                    )
                trajectory = (
                    _load_trajectory(trajectory_dir, trajectory_name)
                    if source_format == "pose_labeled"
                    else _load_frame_only_trajectory(trajectory_dir, trajectory_name)
                )
            except RuntimeError as exc:
                return trajectory_name, trajectory_dir, None, exc, False
            return trajectory_name, trajectory_dir, trajectory, None, False

        def consume_results(
            results: Any,
        ) -> None:
            nonlocal anchor_count
            for trajectory_name, trajectory_dir, trajectory, error, missing in results:
                if missing:
                    missing_names.append(trajectory_name)
                    continue
                if error is not None:
                    exc = error
                    if not navigation_skip_bad_data_enabled():
                        raise exc
                    message = (
                        f"Skipping corrupt navigation trajectory {trajectory_dir}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    corrupt_trajectories.append(
                        {
                            "trajectory": trajectory_name,
                            "path": str(trajectory_dir),
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    report_navigation_data_warning(
                        message,
                        kind="corrupt_trajectory",
                        source=display_name,
                        path=trajectory_dir,
                    )
                    continue
                assert trajectory is not None
                trajectories.append(trajectory)
                anchor_count += trajectory.length
                cumulative_counts.append(anchor_count)

        if audit_workers == 1 or len(manifest_names) <= 1:
            consume_results(map(load_one_trajectory, manifest_names))
        else:
            with ThreadPoolExecutor(
                max_workers=min(audit_workers, len(manifest_names)),
                thread_name_prefix="navigation-audit",
            ) as executor:
                consume_results(executor.map(load_one_trajectory, manifest_names))
        if not trajectories:
            raise ValueError(
                f"Source {display_name!r} has no usable trajectories "
                f"from {manifest if manifest is not None else data_root}"
            )

        source = _Source(
            display_name=display_name,
            data_root=data_root,
            split_manifest=manifest,
            waypoint_spacing=waypoint_spacing,
            format=source_format,
            has_action=has_action,
            trajectories=tuple(trajectories),
            cumulative_anchor_counts=tuple(cumulative_counts),
        )
        sources.append(source)
        audit = {
            "dataset_id": source_id,
            "display_name": display_name,
            "data_root": str(data_root),
            "split_manifest": None if manifest is None else str(manifest),
            "manifest_trajectory_count": len(manifest_names),
            "usable_trajectory_count": len(trajectories),
            "missing_trajectory_count": len(missing_names),
            "missing_trajectories": tuple(missing_names),
            "anchor_count": anchor_count,
            "waypoint_spacing": waypoint_spacing,
            "format": source_format,
            "has_action": has_action,
        }
        if navigation_skip_bad_data_enabled():
            audit.update(
                {
                    "corrupt_trajectory_count": len(corrupt_trajectories),
                    "corrupt_trajectories": tuple(corrupt_trajectories),
                }
            )
        audit_sources.append(audit)
    return tuple(sources), tuple(audit_sources)


class _SamplingSchedule:
    """Finite deterministic schedule with balanced offsets and weighted sources."""

    OFFSET_SLOTS = 4096

    def __init__(
        self,
        sources: Sequence[_Source],
        source_ids: Sequence[int],
        *,
        max_frame_offset: int,
        exponent: float,
        seed: int,
        group: int,
    ) -> None:
        if not source_ids:
            raise ValueError(f"sampling group {group} contains no sources")
        entries: list[tuple[int, int]] = []
        for offset in range(-max_frame_offset, max_frame_offset + 1):
            eligible = [
                source_id
                for source_id in source_ids
                if sources[source_id].valid_anchor_count(offset) > 0
            ]
            if not eligible:
                raise ValueError(f"No trajectory supports frame offset {offset}")
            weights = [sources[source_id].anchor_count**exponent for source_id in eligible]
            total = sum(weights)
            exact = [self.OFFSET_SLOTS * weight / total for weight in weights]
            counts = [int(value) for value in exact]
            remainder = self.OFFSET_SLOTS - sum(counts)
            order = sorted(
                range(len(eligible)),
                key=lambda index: (exact[index] - counts[index], -eligible[index]),
                reverse=True,
            )
            for index in order[:remainder]:
                counts[index] += 1
            for source_id, count in zip(eligible, counts, strict=True):
                entries.extend([(source_id, offset)] * count)
        rng = random.Random(seed + (group + 2) * 0x9E3779B1)
        rng.shuffle(entries)
        self.entries = tuple(entries)
        positions: dict[tuple[int, int], list[int]] = {}
        for position, category in enumerate(self.entries):
            positions.setdefault(category, []).append(position)
        self.positions = {category: tuple(values) for category, values in positions.items()}
        self.source_counts = {
            source_id: sum(count for (entry_source, _), count in self._category_counts().items() if entry_source == source_id)
            for source_id in source_ids
        }
        self.offset_counts = {
            offset: sum(count for (_, entry_offset), count in self._category_counts().items() if entry_offset == offset)
            for offset in range(-max_frame_offset, max_frame_offset + 1)
        }

    def _category_counts(self) -> dict[tuple[int, int], int]:
        return {category: len(values) for category, values in self.positions.items()}

    def sample(self, ordinal: int, epoch: int) -> tuple[int, int, int]:
        block_size = len(self.entries)
        shifted = ordinal + epoch * 104729
        cycle, position = divmod(shifted, block_size)
        source_id, offset = self.entries[position]
        category_positions = self.positions[(source_id, offset)]
        occurrence = cycle * len(category_positions) + bisect.bisect_left(
            category_positions, position
        )
        return source_id, offset, occurrence


def _permuted_anchor(ordinal: int, size: int, seed: int) -> int:
    """Map consecutive category occurrences bijectively over valid anchors."""

    if size <= 1:
        return 0
    rng = random.Random(seed & _UINT64_MASK)
    multiplier = rng.randrange(1, size)
    while math.gcd(multiplier, size) != 1:
        multiplier = (multiplier + 1) % size or 1
    increment = rng.randrange(size)
    return (multiplier * (ordinal % size) + increment) % size


class NavigationPairDataset(Dataset):
    """Navigation frame-pair dataset supporting labeled and frame-only sources.

    Args:
        source_specs: One or more source mappings. Legacy mappings default to
            ``pose_labeled``; ``frame_only`` mappings discover trajectory dirs.
        samples_per_epoch: Global logical epoch size across all DDP ranks.
        data_seed: Independent data RNG seed.  Pair offset and brightness are a
            deterministic function of this seed and logical index, so different
            model variants receive identical examples.
        max_frame_decode_retries: Maximum number of deterministic replacement
            samples tried after a frame decode failure.  Corrupt paths are
            reported once per DataLoader worker.
    """

    def __init__(
        self,
        source_specs: Sequence[Mapping[str, Any]],
        samples_per_epoch: int = 10_000_000,
        max_frame_offset: int = 64,
        data_seed: int = 0,
        color_aug: bool = True,
        image_height: int = 240,
        image_width: int = 320,
        action_min: Sequence[float] = _ACTION_MIN,
        action_max: Sequence[float] = _ACTION_MAX,
        print_audit: bool = True,
        index_cache_path: str | None = None,
        rebuild_index_cache: bool = False,
        dino_feature_cache_path: str | None = None,
        dataset_size_exponent: float = 0.0,
        uniform_offset_domain: bool = False,
        max_frame_decode_retries: int = 128,
        require_index_cache: bool = False,
    ) -> None:
        super().__init__()
        if not source_specs:
            raise ValueError("NavigationPairDataset requires at least one source")
        if not isinstance(samples_per_epoch, int) or samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be a positive integer")
        if not isinstance(max_frame_offset, int) or max_frame_offset < 0:
            raise ValueError("max_frame_offset must be a non-negative integer")
        if (
            not isinstance(max_frame_decode_retries, int)
            or max_frame_decode_retries < 0
        ):
            raise ValueError("max_frame_decode_retries must be a non-negative integer")
        if image_height <= 0 or image_width <= 0:
            raise ValueError("image_height and image_width must be positive")

        action_min_array = np.asarray(action_min, dtype=np.float64)
        action_max_array = np.asarray(action_max, dtype=np.float64)
        if action_min_array.shape != (2,) or action_max_array.shape != (2,):
            raise ValueError("action_min and action_max must each contain exactly dx and dy")
        if not np.all(action_max_array > action_min_array):
            raise ValueError("Each action_max value must be greater than action_min")

        self.samples_per_epoch = samples_per_epoch
        self.max_frame_offset = max_frame_offset
        self.max_frame_decode_retries = max_frame_decode_retries
        self.data_seed = int(data_seed)
        self.dataset_size_exponent = float(dataset_size_exponent)
        if not math.isfinite(self.dataset_size_exponent) or self.dataset_size_exponent < 0:
            raise ValueError("dataset_size_exponent must be finite and non-negative")
        self.uniform_offset_domain = bool(uniform_offset_domain)
        self.color_aug = bool(color_aug)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.action_min = action_min_array
        self.action_max = action_max_array
        self.index_cache_path = (
            None
            if index_cache_path is None
            else Path(index_cache_path).expanduser().resolve()
        )
        self.rebuild_index_cache = bool(rebuild_index_cache)
        self.require_index_cache = bool(require_index_cache)
        self.dino_feature_cache_path = (
            None
            if dino_feature_cache_path is None
            else Path(dino_feature_cache_path).expanduser().resolve()
        )
        if self.dino_feature_cache_path is not None and self.color_aug:
            raise ValueError(
                "Precomputed DINOv3 features require color_aug=False because "
                "image-space brightness augmentation cannot be applied after caching"
            )
        if self.rebuild_index_cache and self.index_cache_path is None:
            raise ValueError("rebuild_index_cache=True requires index_cache_path")
        if self.require_index_cache and self.index_cache_path is None:
            raise ValueError("require_index_cache=True requires index_cache_path")
        if self.require_index_cache and self.rebuild_index_cache:
            raise ValueError(
                "require_index_cache=True is incompatible with rebuild_index_cache=True"
            )
        self._loaded_from_index_cache = False
        self._validated_cached_trajectories: set[str] = set()
        # Each DataLoader worker has its own dataset copy.  Report a corrupt
        # path once per worker instead of flooding logs every time an epoch
        # revisits the same frame.
        self._reported_bad_frames: set[str] = set()
        # A shared scalar keeps persistent DataLoader workers in sync when the
        # sampler advances the epoch.  The default configuration does not use
        # persistent workers, but supporting it here avoids silently freezing
        # the pair/brightness RNG if that setting is changed for an audit.
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

        cached = None
        if self.index_cache_path is not None:
            if not self.require_index_cache:
                self.index_cache_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.rebuild_index_cache:
                # Immutable navigation snapshots are the common case.  Read
                # the complete cached index directly instead of making every
                # DDP rank walk and stat every NAS trajectory at startup.
                cached = _read_index_cache(self.index_cache_path, source_specs)
                self._loaded_from_index_cache = cached is not None
                if cached is None and self.require_index_cache:
                    raise RuntimeError(
                        "Required immutable navigation index cache is missing, stale, "
                        f"or incompatible: {self.index_cache_path}"
                    )
            if cached is None:
                lock_path = self.index_cache_path.with_suffix(
                    self.index_cache_path.suffix + ".lock"
                )
                with lock_path.open("a+b") as lock_file:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    if not self.rebuild_index_cache:
                        # Another rank may have completed the audit while this
                        # process waited for the lock.
                        cached = _read_index_cache(
                            self.index_cache_path, source_specs
                        )
                        self._loaded_from_index_cache = cached is not None
                    if cached is None:
                        # A cache miss, source-configuration change, or
                        # explicit rebuild is the only path that performs the
                        # expensive filesystem inventory and trajectory audit.
                        signature = _source_specs_signature(source_specs)
                        sources, audit_sources = _build_sources(source_specs)
                        _write_index_cache(
                            self.index_cache_path, signature, sources, audit_sources
                        )
                        cached = (sources, audit_sources)
        else:
            cached = _build_sources(source_specs)

        self._sources, audit_sources = cached
        # Action-label allowlists deliberately do not participate in the
        # index-cache identity.  They only mask supervision on pose-labeled
        # trajectories, so L0/L50/L100 can share the exact same cached source
        # index and, consequently, the exact same pixel sampling stream.
        action_label_names: list[frozenset[str] | None] = []
        augmented_audit_sources: list[dict[str, Any]] = []
        for source_id, (source, raw_spec, raw_audit) in enumerate(
            zip(self._sources, source_specs, audit_sources, strict=True)
        ):
            audit_source = dict(raw_audit)
            manifest_value = raw_spec.get("action_label_manifest")
            if manifest_value in (None, ""):
                selected_names = None
                visible_trajectories = source.trajectories if source.has_action else ()
                manifest_path = None
                manifest_sha256 = None
            else:
                if not source.has_action:
                    raise ValueError(
                        "action_label_manifest is only valid for pose-labeled "
                        f"sources; source_specs[{source_id}] is {source.format!r}"
                    )
                manifest_path = _as_path(manifest_value, "action_label_manifest")
                manifest_names = _read_action_label_manifest(manifest_path)
                usable_names = {trajectory.name for trajectory in source.trajectories}
                unknown_names = sorted(set(manifest_names) - usable_names)
                if unknown_names:
                    preview = ", ".join(repr(name) for name in unknown_names[:5])
                    suffix = "..." if len(unknown_names) > 5 else ""
                    raise ValueError(
                        f"Action-label manifest {manifest_path} contains "
                        f"{len(unknown_names)} trajectories absent from the usable "
                        f"training split: {preview}{suffix}"
                    )
                selected_names = frozenset(manifest_names)
                visible_trajectories = tuple(
                    trajectory
                    for trajectory in source.trajectories
                    if trajectory.name in selected_names
                )
                manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            action_label_names.append(selected_names)
            visible_anchor_count = sum(
                trajectory.length for trajectory in visible_trajectories
            )
            if manifest_path is not None:
                audit_source.update(
                    {
                        "action_label_manifest": str(manifest_path),
                        "action_label_manifest_sha256": manifest_sha256,
                        "action_label_trajectory_count": len(visible_trajectories),
                        "action_label_anchor_count": visible_anchor_count,
                        "action_label_anchor_fraction": (
                            visible_anchor_count / source.anchor_count
                            if source.has_action
                            else 0.0
                        ),
                    }
                )
            augmented_audit_sources.append(audit_source)
        self._action_label_names = tuple(action_label_names)
        audit_sources = tuple(augmented_audit_sources)
        self._dino_feature_cache = (
            None
            if self.dino_feature_cache_path is None
            else _DINOFeatureCache(
                self.dino_feature_cache_path,
                self._sources,
                image_height=self.image_height,
                image_width=self.image_width,
            )
        )
        self.source_anchor_counts = tuple(
            source.anchor_count for source in self._sources
        )
        self.action_source_ids = tuple(
            source_id for source_id, source in enumerate(self._sources) if source.has_action
        )
        self.frame_only_source_ids = tuple(
            source_id for source_id, source in enumerate(self._sources) if not source.has_action
        )
        explicit_mixed_source = any(
            "format" in spec or "has_action" in spec for spec in source_specs
        )
        self._mixed_sampling = (
            explicit_mixed_source
            or self.dataset_size_exponent != 0.0
            or self.uniform_offset_domain
        )
        self._schedules: dict[int, _SamplingSchedule] = {}
        if self._mixed_sampling:
            groups = {-1: tuple(range(len(self._sources)))}
            if self.action_source_ids:
                groups[1] = self.action_source_ids
            if self.frame_only_source_ids:
                groups[0] = self.frame_only_source_ids
            for group, source_ids in groups.items():
                self._schedules[group] = _SamplingSchedule(
                    self._sources,
                    source_ids,
                    max_frame_offset=self.max_frame_offset,
                    exponent=self.dataset_size_exponent,
                    seed=self.data_seed,
                    group=group,
                )
        self.audit: dict[str, Any] = {
            "samples_per_epoch": self.samples_per_epoch,
            "total_unique_anchors": sum(self.source_anchor_counts),
            "sources": audit_sources,
            "index_cache_path": (
                None if self.index_cache_path is None else str(self.index_cache_path)
            ),
            "loaded_from_index_cache": self._loaded_from_index_cache,
            "rebuild_index_cache": self.rebuild_index_cache,
            "dino_feature_cache_path": (
                None
                if self.dino_feature_cache_path is None
                else str(self.dino_feature_cache_path)
            ),
            "dataset_size_exponent": self.dataset_size_exponent,
            "uniform_offset_domain": self.uniform_offset_domain,
            "max_frame_decode_retries": self.max_frame_decode_retries,
            "action_source_ids": self.action_source_ids,
            "frame_only_source_ids": self.frame_only_source_ids,
            "sampling_schedules": {
                str(group): {
                    "cycle_samples": len(schedule.entries),
                    "source_counts": schedule.source_counts,
                    "offset_counts": schedule.offset_counts,
                    "anchor_repeat_periods": {
                        f"{source_id}:{offset}": self._sources[source_id].valid_anchor_count(offset)
                        for source_id, offset in schedule.positions
                    },
                }
                for group, schedule in self._schedules.items()
            },
        }
        if self.require_index_cache:
            self.audit["require_index_cache"] = True
        for source in audit_sources:
            if source["missing_trajectory_count"]:
                report_navigation_data_warning(
                    f"{source['display_name']} split is missing "
                    f"{source['missing_trajectory_count']} trajectories; they will be skipped",
                    kind="missing_trajectory",
                    source=source["display_name"],
                    path=source["split_manifest"] or source["data_root"],
                )
        if print_audit:
            self.print_audit()

    def print_audit(self) -> None:
        print(
            "[NavigationPairDataset] "
            f"unique_anchors={self.audit['total_unique_anchors']:,}, "
            f"logical_samples_per_epoch={self.samples_per_epoch:,}"
        )
        if self.index_cache_path is not None:
            cache_state = "loaded" if self._loaded_from_index_cache else "rebuilt"
            print(
                "[NavigationPairDataset] "
                f"index_cache={cache_state}: {self.index_cache_path}"
            )
        if self.dino_feature_cache_path is not None:
            print(
                "[NavigationPairDataset] "
                f"dino_feature_cache=loaded: {self.dino_feature_cache_path}"
            )
        for source in self.audit["sources"]:
            print(
                "[NavigationPairDataset] "
                f"source[{source['dataset_id']}]={source['display_name']}: "
                f"manifest={source['manifest_trajectory_count']}, "
                f"usable={source['usable_trajectory_count']}, "
                f"missing={source['missing_trajectory_count']}, "
                f"anchors={source['anchor_count']:,}"
                f", format={source['format']}, has_action={source['has_action']}"
            )
            if "action_label_manifest" in source:
                print(
                    "[NavigationPairDataset] "
                    f"source[{source['dataset_id']}] action_labels="
                    f"{source['action_label_trajectory_count']}/"
                    f"{source['usable_trajectory_count']} trajectories, "
                    f"anchor_fraction={source['action_label_anchor_fraction']:.6f}, "
                    f"manifest={source['action_label_manifest']}"
                )
        for group, schedule in self.audit["sampling_schedules"].items():
            source_counts = schedule["source_counts"]
            total = schedule["cycle_samples"]
            ratios = ", ".join(
                f"{source_id}:{count / total:.4f}"
                for source_id, count in sorted(source_counts.items())
            )
            print(
                "[NavigationPairDataset] "
                f"sampling_group={group}, schedule_cycle={total:,}, "
                f"source_ratios={ratios}, offsets=uniform[-{self.max_frame_offset},"
                f"{self.max_frame_offset}], anchor_repeat_period=valid_anchor_count"
            )

    def __len__(self) -> int:
        return self.samples_per_epoch

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    def set_epoch(self, epoch: int) -> None:
        epoch = operator.index(epoch)
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch.fill_(epoch)

    def _rng_for_index(self, index: int) -> random.Random:
        # SplitMix64 finalizer: stable across Python processes and independent
        # of global RNG state, unlike hash((seed, epoch, index)).  Epoch zero
        # intentionally preserves the original sequence used by the checked-in
        # variant-4 calibration result.
        value = (self.data_seed + index + 0x9E3779B97F4A7C15) & _UINT64_MASK
        if self.epoch:
            value = (
                value + self.epoch * 0xD1B54A32D192ED03
            ) & _UINT64_MASK
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
        value ^= value >> 31
        return random.Random(value)

    def _sample_offset(
        self, rng: random.Random, anchor_frame: int, trajectory_length: int
    ) -> int:
        lower = max(-self.max_frame_offset, -anchor_frame)
        upper = min(self.max_frame_offset, trajectory_length - anchor_frame - 1)
        return rng.randint(lower, upper)

    def _validate_cached_trajectory(self, trajectory: _Trajectory) -> None:
        """Ensure a cached immutable NAS trajectory has not changed in place."""

        if not self._loaded_from_index_cache:
            return
        cache_key = str(trajectory.directory)
        if cache_key in self._validated_cached_trajectories:
            return
        try:
            directory_stat = trajectory.directory.stat()
            metadata_stat = (
                None
                if trajectory.positions is None
                else (trajectory.directory / "traj_data.pkl").stat()
            )
        except OSError as exc:
            raise NavigationTrajectoryValidationError(
                trajectory.directory,
                f"Cached navigation trajectory is no longer readable: {trajectory.directory}",
            ) from exc
        if (
            (
                metadata_stat is not None
                and (
                    metadata_stat.st_size != trajectory.metadata_size
                    or metadata_stat.st_mtime_ns != trajectory.metadata_mtime_ns
                )
            )
            or directory_stat.st_mtime_ns != trajectory.directory_mtime_ns
        ):
            raise NavigationTrajectoryValidationError(
                trajectory.directory,
                "Navigation trajectory changed after the shared index was built: "
                f"{trajectory.directory}. Remove or relocate the configured index cache "
                "and restart so the dataset is audited again.",
            )
        self._validated_cached_trajectories.add(cache_key)

    def _load_frame(self, path: Path) -> Tensor:
        return load_navigation_frame(
            path,
            image_height=self.image_height,
            image_width=self.image_width,
        )

    def _action(
        self, source: _Source, trajectory: _Trajectory, start: int, target: int
    ) -> Tensor:
        if trajectory.positions is None or trajectory.yaw is None:
            raise RuntimeError("Cannot derive an action from a frame-only trajectory")
        displacement = trajectory.positions[target, :2] - trajectory.positions[start, :2]
        start_yaw = float(trajectory.yaw[start])
        cosine, sine = np.cos(start_yaw), np.sin(start_yaw)
        # This is CompACT's ``(position - curr_pos).dot(yaw_rotmat(curr_yaw))``
        # row-vector convention, restricted to planar navigation.
        local_xy = np.asarray(
            [
                displacement[0] * cosine + displacement[1] * sine,
                -displacement[0] * sine + displacement[1] * cosine,
            ],
            dtype=np.float64,
        )
        local_xy /= source.waypoint_spacing
        normalized_xy = (
            (local_xy - self.action_min) / (self.action_max - self.action_min) * 2.0
            - 1.0
        )
        yaw_delta = float(trajectory.yaw[target] - start_yaw)
        yaw_delta -= 2.0 * np.pi * np.floor((yaw_delta + np.pi) / (2.0 * np.pi))
        action = np.asarray(
            [normalized_xy[0], normalized_xy[1], yaw_delta], dtype=np.float32
        )
        return torch.from_numpy(action).unsqueeze(0)

    def _get_item_once(self, index: int, requested_group: int) -> dict[str, Tensor]:
        rng = self._rng_for_index(index)
        if self._mixed_sampling or requested_group != -1:
            try:
                schedule = self._schedules[requested_group]
            except KeyError as exc:
                label = "labeled" if requested_group == 1 else "frame-only"
                raise ValueError(f"Dataset has no {label} sampling sources") from exc
            source_id, offset, occurrence = schedule.sample(index, self.epoch)
            source = self._sources[source_id]
            valid_count = source.valid_anchor_count(offset)
            anchor_index = _permuted_anchor(
                occurrence,
                valid_count,
                self.data_seed
                + self.epoch * 0xD1B54A33
                + source_id * 0x9E3779B1
                + (offset + self.max_frame_offset) * 0x85EBCA77,
            )
            trajectory, anchor_frame = source.valid_anchor(offset, anchor_index)
        else:
            # Exact legacy behavior for the original three-source experiments.
            source_id = index % len(self._sources)
            source = self._sources[source_id]
            source_anchor_index = (index // len(self._sources)) % source.anchor_count
            trajectory, anchor_frame = source.anchor(source_anchor_index)
            offset = self._sample_offset(rng, anchor_frame, trajectory.length)
        self._validate_cached_trajectory(trajectory)
        target_frame = anchor_frame + offset

        selected_action_names = self._action_label_names[source_id]
        action_visible = source.has_action and (
            selected_action_names is None or trajectory.name in selected_action_names
        )

        item = {
            "actions": (
                self._action(source, trajectory, anchor_frame, target_frame)
                if action_visible
                else torch.zeros((1, 3), dtype=torch.float32)
            ),
            "action_mask": torch.tensor(action_visible, dtype=torch.bool),
            "dataset_id": torch.tensor(source_id, dtype=torch.int64),
            "frame_offset": torch.tensor(offset, dtype=torch.int64),
        }
        if self._dino_feature_cache is not None:
            item["dino_features"] = self._dino_feature_cache.pair(
                source_id, trajectory, anchor_frame, target_frame
            )
        else:
            videos = torch.stack(
                [
                    self._load_frame(trajectory.directory / f"{anchor_frame}.jpg"),
                    self._load_frame(trajectory.directory / f"{target_frame}.jpg"),
                ],
                dim=0,
            )
            if self.color_aug:
                brightness = rng.random() * 0.2 - 0.1
                videos = videos.add(brightness).clamp_(0.0, 1.0)
            item["videos"] = videos.to(dtype=torch.float32)
        return item

    def __getitem__(self, request: int | tuple[int, int]) -> dict[str, Tensor]:
        requested_group = -1
        if isinstance(request, tuple):
            if len(request) != 2:
                raise ValueError("navigation sample requests must be (ordinal, group)")
            index, requested_group = request
            if requested_group not in {-1, 0, 1}:
                raise ValueError(f"invalid navigation sampling group: {requested_group}")
        else:
            index = request
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        original_index = index
        # Consecutive logical indices are distinct until the finite dataset
        # wraps.  Keeping the group unchanged preserves labeled/frame-only
        # batch quotas while a corrupt sample is replaced.
        replacement_count = (
            len(self) - 1
            if navigation_skip_bad_data_enabled()
            else min(self.max_frame_decode_retries, len(self) - 1)
        )
        last_error: (
            NavigationFrameDecodeError | NavigationTrajectoryValidationError | None
        ) = None
        for attempt in range(replacement_count + 1):
            candidate_index = (original_index + attempt) % len(self)
            try:
                return self._get_item_once(candidate_index, requested_group)
            except NavigationTrajectoryValidationError as exc:
                if not navigation_skip_bad_data_enabled():
                    raise
                last_error = exc
                path_key = str(exc.path)
                if path_key not in self._reported_bad_frames:
                    self._reported_bad_frames.add(path_key)
                    report_navigation_data_warning(
                        f"Skipping stale or unreadable cached navigation trajectory "
                        f"{exc.path}: {exc.reason}",
                        kind="corrupt_cached_trajectory",
                        path=exc.path,
                    )
            except NavigationFrameDecodeError as exc:
                last_error = exc
                path_key = str(exc.path)
                if path_key not in self._reported_bad_frames:
                    self._reported_bad_frames.add(path_key)
                    if attempt < replacement_count:
                        outcome = (
                            "retrying with logical sample "
                            f"{(candidate_index + 1) % len(self)}"
                        )
                    else:
                        outcome = "no replacement attempts remain"
                    report_navigation_data_warning(
                        f"Skipping corrupt navigation frame {exc.path}: "
                        f"{exc.reason}; {outcome}",
                        kind="corrupt_frame",
                        path=exc.path,
                    )

        assert last_error is not None
        raise RuntimeError(
            "Unable to replace a navigation sample after trying "
            f"{replacement_count + 1} distinct logical samples "
            f"(original logical index {original_index}, sampling group "
            f"{requested_group}, last corrupt path {last_error.path}). "
            "Increase max_frame_decode_retries or repair the reported frames."
        ) from last_error


class _EpochAwareDistributedSampler(DistributedSampler):
    """Distributed sampler that also advances the dataset's stateless RNG."""

    dataset: NavigationPairDataset

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)
        super().set_epoch(epoch)


class _NavigationRequestSampler(Sampler[int | tuple[int, int]]):
    """Shard logical samples and optionally enforce a per-rank label quota."""

    def __init__(
        self,
        dataset: NavigationPairDataset,
        *,
        batch_size: int,
        labeled_batch_fraction: float | None,
        num_replicas: int,
        rank: int,
        seed: int,
        drop_last: bool,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        if labeled_batch_fraction is None:
            self.labeled_per_batch = None
        else:
            if not 0.0 < labeled_batch_fraction < 1.0:
                raise ValueError("labeled_batch_fraction must be between zero and one")
            self.labeled_per_batch = math.ceil(batch_size * labeled_batch_fraction)
            if not dataset.action_source_ids or not dataset.frame_only_source_ids:
                raise ValueError(
                    "labeled_batch_fraction requires both labeled and frame-only sources"
                )
        if drop_last:
            global_batch = batch_size * num_replicas
            self.num_samples = (len(dataset) // global_batch) * batch_size
        else:
            self.num_samples = math.ceil(len(dataset) / num_replicas)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = operator.index(epoch)
        self.dataset.set_epoch(self.epoch)

    def __iter__(self):
        if self.labeled_per_batch is None:
            for local_index in range(self.num_samples):
                ordinal = local_index * self.num_replicas + self.rank
                yield (ordinal % len(self.dataset), -1)
            return
        labeled = self.labeled_per_batch
        unlabeled = self.batch_size - labeled
        positions = list(range(self.batch_size))
        for local_index in range(self.num_samples):
            batch_index, position = divmod(local_index, self.batch_size)
            if position == 0:
                random.Random(self.seed + self.epoch * 104729 + batch_index).shuffle(
                    positions
                )
            quota_position = positions[position]
            global_batch = batch_index * self.num_replicas + self.rank
            if quota_position < labeled:
                ordinal = global_batch * labeled + quota_position
                yield (ordinal, 1)
            else:
                ordinal = global_batch * unlabeled + (quota_position - labeled)
                yield (ordinal, 0)


class NavigationPairDataModule(LightningDataModule):
    """Lightning data module for :class:`NavigationPairDataset` (train only)."""

    def __init__(
        self,
        source_specs: Sequence[Mapping[str, Any]],
        batch_size: int = 20,
        num_workers: int = 32,
        train_shuffle: bool = True,
        samples_per_epoch: int = 10_000_000,
        max_frame_offset: int = 64,
        data_seed: int = 0,
        color_aug: bool = True,
        image_height: int = 240,
        image_width: int = 320,
        action_min: Sequence[float] = _ACTION_MIN,
        action_max: Sequence[float] = _ACTION_MAX,
        print_audit: bool = True,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        drop_last: bool = False,
        index_cache_path: str | None = None,
        rebuild_index_cache: bool = False,
        dino_feature_cache_path: str | None = None,
        dataset_size_exponent: float = 0.0,
        uniform_offset_domain: bool = False,
        labeled_batch_fraction: float | None = None,
        max_frame_decode_retries: int = 128,
        require_index_cache: bool = False,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if max_frame_decode_retries < 0:
            raise ValueError("max_frame_decode_retries must be non-negative")
        self.source_specs = tuple(dict(spec) for spec in source_specs)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.train_shuffle = bool(train_shuffle)
        self.samples_per_epoch = int(samples_per_epoch)
        self.max_frame_offset = int(max_frame_offset)
        self.data_seed = int(data_seed)
        self.color_aug = bool(color_aug)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.action_min = tuple(float(value) for value in action_min)
        self.action_max = tuple(float(value) for value in action_max)
        self.print_audit_enabled = bool(print_audit)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers=True requires num_workers > 0")
        self.drop_last = bool(drop_last)
        self.index_cache_path = index_cache_path
        self.rebuild_index_cache = bool(rebuild_index_cache)
        self.require_index_cache = bool(require_index_cache)
        self.dino_feature_cache_path = dino_feature_cache_path
        self.dataset_size_exponent = float(dataset_size_exponent)
        self.uniform_offset_domain = bool(uniform_offset_domain)
        self.labeled_batch_fraction = labeled_batch_fraction
        self.max_frame_decode_retries = int(max_frame_decode_retries)
        self.train_dataset: NavigationPairDataset | None = None
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        if stage not in (None, "fit", "train"):
            return
        if self.train_dataset is None:
            dataset_kwargs = {
                "source_specs": self.source_specs,
                "samples_per_epoch": self.samples_per_epoch,
                "max_frame_offset": self.max_frame_offset,
                "data_seed": self.data_seed,
                "color_aug": self.color_aug,
                "image_height": self.image_height,
                "image_width": self.image_width,
                "action_min": self.action_min,
                "action_max": self.action_max,
                "print_audit": self.print_audit_enabled,
                "index_cache_path": self.index_cache_path,
                "require_index_cache": self.require_index_cache,
                "dino_feature_cache_path": self.dino_feature_cache_path,
                "dataset_size_exponent": self.dataset_size_exponent,
                "uniform_offset_domain": self.uniform_offset_domain,
                "max_frame_decode_retries": self.max_frame_decode_retries,
            }
            distributed = (
                dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            )
            if self.rebuild_index_cache and distributed:
                # Exactly one global rank performs the expensive forced audit;
                # broadcast its ordinary failure (if any), then let all other
                # ranks read the atomically written cache.
                rebuild_status: list[tuple[str, str] | None] = [None]
                rebuild_error: Exception | None = None
                if dist.get_rank() == 0:
                    try:
                        self.train_dataset = NavigationPairDataset(
                            **dataset_kwargs, rebuild_index_cache=True
                        )
                    except Exception as exc:  # noqa: BLE001 - broadcast data audit failures
                        rebuild_error = exc
                        rebuild_status[0] = (type(exc).__name__, str(exc))
                dist.broadcast_object_list(rebuild_status, src=0)
                if rebuild_status[0] is not None:
                    error_type, error_message = rebuild_status[0]
                    error = RuntimeError(
                        "rank 0 navigation index rebuild failed: "
                        f"{error_type}: {error_message}"
                    )
                    if rebuild_error is not None:
                        raise error from rebuild_error
                    raise error
                if dist.get_rank() != 0:
                    self.train_dataset = NavigationPairDataset(
                        **dataset_kwargs, rebuild_index_cache=False
                    )
            else:
                self.train_dataset = NavigationPairDataset(
                    **dataset_kwargs,
                    rebuild_index_cache=self.rebuild_index_cache,
                )

    @property
    def audit(self) -> dict[str, Any] | None:
        return None if self.train_dataset is None else self.train_dataset.audit

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        generator = torch.Generator()
        generator.manual_seed(self.data_seed)
        sampler = None
        shuffle = self.train_shuffle
        if self.train_shuffle:
            # Supplying our DistributedSampler subclass prevents Lightning
            # from replacing it and guarantees that Lightning's per-epoch
            # ``set_epoch`` call reaches the dataset RNG.  A one-replica
            # sampler gives the same behavior in ordinary single-process runs.
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
                rank = dist.get_rank()
            else:
                num_replicas = 1
                rank = 0
            if self.train_dataset._mixed_sampling or self.labeled_batch_fraction is not None:
                sampler = _NavigationRequestSampler(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    labeled_batch_fraction=self.labeled_batch_fraction,
                    num_replicas=num_replicas,
                    rank=rank,
                    seed=self.data_seed,
                    drop_last=self.drop_last,
                )
            else:
                sampler = _EpochAwareDistributedSampler(
                    self.train_dataset,
                    num_replicas=num_replicas,
                    rank=rank,
                    shuffle=True,
                    seed=self.data_seed,
                    drop_last=self.drop_last,
                )
            shuffle = False
        from .resumable_loader import ResumableDataLoader

        return ResumableDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            generator=generator,
            worker_init_fn=partial(_seed_worker, data_seed=self.data_seed),
        )


__all__ = [
    "NavigationFrameDecodeError",
    "NavigationTrajectoryValidationError",
    "NavigationPairDataModule",
    "NavigationPairDataset",
    "load_navigation_frame",
    "navigation_skip_bad_data_enabled",
    "report_navigation_data_warning",
]
