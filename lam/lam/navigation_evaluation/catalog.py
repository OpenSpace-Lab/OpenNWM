"""Read-only navigation data catalog used by reconstruction evaluation.

Unlike :mod:`lam.navigation_dataset`, this module accepts any number of
explicit dataset/split manifests and never discovers trajectories by walking a
data root.  The validation, image preprocessing, and SE(2) action convention
mirror navigation training exactly, while test manifests are intentionally
allowed.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch import Tensor

DEFAULT_OFFSETS: tuple[int, ...] = (-8, -6, -4, -2, 0, 2, 4, 6, 8)
DEFAULT_ACTION_MIN: tuple[float, float] = (-2.5, -4.0)
DEFAULT_ACTION_MAX: tuple[float, float] = (5.0, 4.0)
DEFAULT_WAYPOINT_SPACING: dict[str, float] = {
    "recon": 0.25,
    "huron": 0.255,
    "scand": 0.38,
    "tartan_drive": 0.72,
}

_DATASET_ALIASES = {
    "recon": "recon",
    "scand": "scand",
    "huron": "huron",
    "sacson": "huron",
    "tartan": "tartan_drive",
    "tartandrive": "tartan_drive",
    "tartan-drive": "tartan_drive",
    "tartan_drive": "tartan_drive",
}
_SPLITS = frozenset({"train", "test"})


def canonical_dataset_id(value: str) -> str:
    """Return a stable public dataset id, accepting documented path aliases.

    The four standard benchmark IDs retain fixed spellings. Additional
    explicitly configured domains use a conservative lowercase slug so the
    read-only catalog can scale without adding another hard-coded alias.
    """

    if not isinstance(value, str) or not value.strip():
        raise TypeError("dataset_id must be a non-empty string")
    normalized = value.strip().lower().replace(" ", "_")
    if normalized in _DATASET_ALIASES:
        return _DATASET_ALIASES[normalized]
    normalized = normalized.replace("-", "_")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", normalized):
        raise ValueError(
            f"Invalid navigation dataset id {value!r}; use lowercase letters, "
            "digits, and underscores"
        )
    return normalized


def canonical_split(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("split must be a non-empty string")
    split = value.strip().lower()
    if split not in _SPLITS:
        raise ValueError(
            f"Unsupported navigation split {value!r}; expected train or test"
        )
    return split


def validate_offsets(offsets: Iterable[int]) -> tuple[int, ...]:
    """Validate an ordered offset grid and return it as a tuple."""

    values: list[int] = []
    for value in offsets:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"Frame offsets must be integers; got {value!r}")
        values.append(int(value))
    result = tuple(values)
    if not result:
        raise ValueError("offsets must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"offsets contains duplicates: {result}")
    if tuple(sorted(result)) != result:
        raise ValueError(f"offsets must be strictly increasing: {result}")
    return result


@dataclass(frozen=True)
class DatasetSplitSpec:
    """One explicit dataset/split source; no directory discovery is performed."""

    dataset_id: str
    split: str
    data_root: Path
    split_manifest: Path
    waypoint_spacing: float
    display_name: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> DatasetSplitSpec:
        if not isinstance(raw, Mapping):
            raise TypeError("Each catalog source must be a mapping")
        raw_dataset = raw.get("dataset_id", raw.get("name", raw.get("display_name")))
        dataset_id = canonical_dataset_id(raw_dataset)
        raw_split = raw.get("split")
        if raw_split is None:
            raw_split = _split_from_manifest(raw.get("split_manifest"))
        split = canonical_split(raw_split)
        data_root = _as_path(raw.get("data_root"), "data_root")
        split_manifest = _as_path(raw.get("split_manifest"), "split_manifest")
        raw_spacing = raw.get("waypoint_spacing")
        if raw_spacing is None:
            raw_spacing = DEFAULT_WAYPOINT_SPACING.get(dataset_id)
        if raw_spacing is None:
            raise ValueError(
                f"waypoint_spacing is required for custom dataset {dataset_id!r}"
            )
        try:
            spacing = float(raw_spacing)
        except (TypeError, ValueError) as exc:
            raise TypeError("waypoint_spacing must be numeric") from exc
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("waypoint_spacing must be finite and positive")
        display_name = str(raw.get("display_name", dataset_id)).strip()
        if not display_name:
            raise ValueError("display_name must not be empty")
        return cls(dataset_id, split, data_root, split_manifest, spacing, display_name)

    @property
    def key(self) -> tuple[str, str]:
        return self.dataset_id, self.split


@dataclass(frozen=True)
class ActionRecord:
    """The same planar action represented in evaluation and training units.

    The third component is the wrapped yaw delta in radians in every
    representation.  Only XY changes units.
    """

    normalized: tuple[float, float, float]
    waypoint: tuple[float, float, float]
    meters: tuple[float, float, float]

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "normalized": list(self.normalized),
            "waypoint": list(self.waypoint),
            "meters": list(self.meters),
        }


@dataclass(frozen=True)
class TrajectoryRecord:
    dataset_id: str
    split: str
    name: str
    directory: Path
    waypoint_spacing: float
    positions: np.ndarray = field(repr=False, compare=False)
    yaw: np.ndarray = field(repr=False, compare=False)

    @property
    def length(self) -> int:
        return int(self.positions.shape[0])

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.dataset_id, self.split, self.name

    def frame_path(self, frame: int) -> Path:
        if frame < 0 or frame >= self.length:
            raise IndexError(
                f"Frame {frame} is outside trajectory {self.identity} of length {self.length}"
            )
        return self.directory / f"{frame}.jpg"


@dataclass(frozen=True)
class DatasetPartition:
    spec: DatasetSplitSpec
    trajectories: tuple[TrajectoryRecord, ...]
    manifest_sha256: str
    manifest_trajectory_count: int
    missing_trajectories: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.spec.key

    @property
    def legal_trajectory_count(self) -> int:
        return len(self.trajectories)


class EvaluationCatalog:
    """Validated, immutable catalog for arbitrary navigation domains."""

    def __init__(
        self,
        specs: Sequence[DatasetSplitSpec | Mapping[str, Any]],
        *,
        image_height: int = 240,
        image_width: int = 320,
        action_min: Sequence[float] = DEFAULT_ACTION_MIN,
        action_max: Sequence[float] = DEFAULT_ACTION_MAX,
    ) -> None:
        if not specs:
            raise ValueError("Evaluation catalog requires at least one explicit source")
        if image_height <= 0 or image_width <= 0:
            raise ValueError("image_height and image_width must be positive")
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.action_min = _xy_pair(action_min, "action_min")
        self.action_max = _xy_pair(action_max, "action_max")
        if not np.all(self.action_max > self.action_min):
            raise ValueError("Each action_max component must exceed action_min")

        partitions: dict[tuple[str, str], DatasetPartition] = {}
        for raw_spec in specs:
            spec = (
                raw_spec
                if isinstance(raw_spec, DatasetSplitSpec)
                else DatasetSplitSpec.from_mapping(raw_spec)
            )
            if spec.key in partitions:
                raise ValueError(
                    f"Duplicate catalog source for {spec.dataset_id}/{spec.split}"
                )
            partitions[spec.key] = _load_partition(spec)
        # A split label is meaningful only if it cannot silently reference the
        # same physical trajectory through two manifests.  Equal names below
        # distinct per-split roots remain valid because their resolved paths
        # differ; shared-root train/test leakage fails immediately.
        for dataset_id in {dataset for dataset, _ in partitions}:
            dataset_partitions = [
                partition
                for (dataset, _), partition in partitions.items()
                if dataset == dataset_id
            ]
            for left_index, left in enumerate(dataset_partitions):
                left_paths = {trajectory.directory for trajectory in left.trajectories}
                for right in dataset_partitions[left_index + 1 :]:
                    overlap = left_paths & {
                        trajectory.directory for trajectory in right.trajectories
                    }
                    if overlap:
                        examples = sorted(str(path) for path in overlap)[:3]
                        raise ValueError(
                            f"Dataset {dataset_id} split leakage between "
                            f"{left.spec.split} and {right.spec.split}; shared trajectories: "
                            f"{examples}"
                        )
        self._partitions = partitions

    @property
    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._partitions)

    @property
    def partitions(self) -> tuple[DatasetPartition, ...]:
        return tuple(self._partitions.values())

    def get(self, dataset_id: str, split: str) -> DatasetPartition:
        key = canonical_dataset_id(dataset_id), canonical_split(split)
        try:
            return self._partitions[key]
        except KeyError as exc:
            available = ", ".join(f"{d}/{s}" for d, s in self.keys)
            raise KeyError(
                f"Catalog has no {key[0]}/{key[1]}; available: {available}"
            ) from exc

    def load_frame(self, path: str | Path) -> Tensor:
        """Decode one image with the exact non-augmented training transform."""
        return load_navigation_frame(
            path, image_height=self.image_height, image_width=self.image_width
        )

    def action(
        self, trajectory: TrajectoryRecord, anchor: int, target: int
    ) -> ActionRecord:
        return compute_action(
            trajectory,
            anchor,
            target,
            action_min=self.action_min,
            action_max=self.action_max,
        )

    def audit(self) -> dict[str, Any]:
        return {
            "image_height": self.image_height,
            "image_width": self.image_width,
            "action_min": self.action_min.tolist(),
            "action_max": self.action_max.tolist(),
            "sources": [
                {
                    "dataset_id": partition.spec.dataset_id,
                    "split": partition.spec.split,
                    "display_name": partition.spec.display_name,
                    "data_root": str(partition.spec.data_root),
                    "split_manifest": str(partition.spec.split_manifest),
                    "manifest_sha256": partition.manifest_sha256,
                    "waypoint_spacing": partition.spec.waypoint_spacing,
                    "manifest_trajectory_count": partition.manifest_trajectory_count,
                    "trajectory_count": len(partition.trajectories),
                    "missing_trajectory_count": len(partition.missing_trajectories),
                    "missing_trajectories": list(partition.missing_trajectories),
                    "frame_count": sum(t.length for t in partition.trajectories),
                }
                for partition in self.partitions
            ],
        }


def compute_action(
    trajectory: TrajectoryRecord,
    anchor: int,
    target: int,
    *,
    action_min: Sequence[float] | np.ndarray = DEFAULT_ACTION_MIN,
    action_max: Sequence[float] | np.ndarray = DEFAULT_ACTION_MAX,
) -> ActionRecord:
    """Compute local SE(2) action using the navigation-training convention."""

    if anchor < 0 or anchor >= trajectory.length:
        raise IndexError(
            f"anchor={anchor} outside trajectory of length {trajectory.length}"
        )
    if target < 0 or target >= trajectory.length:
        raise IndexError(
            f"target={target} outside trajectory of length {trajectory.length}"
        )
    minimum = _xy_pair(action_min, "action_min")
    maximum = _xy_pair(action_max, "action_max")
    if not np.all(maximum > minimum):
        raise ValueError("Each action_max component must exceed action_min")

    displacement = trajectory.positions[target, :2] - trajectory.positions[anchor, :2]
    start_yaw = float(trajectory.yaw[anchor])
    cosine, sine = np.cos(start_yaw), np.sin(start_yaw)
    meters_xy = np.asarray(
        [
            displacement[0] * cosine + displacement[1] * sine,
            -displacement[0] * sine + displacement[1] * cosine,
        ],
        dtype=np.float64,
    )
    waypoint_xy = meters_xy / trajectory.waypoint_spacing
    normalized_xy = (waypoint_xy - minimum) / (maximum - minimum) * 2.0 - 1.0
    yaw_delta = wrap_yaw(float(trajectory.yaw[target] - start_yaw))

    def triple(xy: np.ndarray) -> tuple[float, float, float]:
        return float(xy[0]), float(xy[1]), float(yaw_delta)

    return ActionRecord(
        normalized=triple(normalized_xy),
        waypoint=triple(waypoint_xy),
        meters=triple(meters_xy),
    )


def load_navigation_frame(
    path: str | Path, *, image_height: int = 240, image_width: int = 320
) -> Tensor:
    """Load one manifest image without reconstructing the full catalog."""

    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_height and image_width must be positive")
    frame_path = Path(path)
    try:
        with Image.open(frame_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot decode navigation frame {frame_path}: {exc}"
        ) from exc
    frame = torch.from_numpy(rgb).to(dtype=torch.float32).div_(255.0)
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise RuntimeError(
            f"Navigation frame has invalid size {width}x{height}: {frame_path}"
        )
    target_ratio = 4.0 / 3.0
    current_ratio = width / height
    if current_ratio > target_ratio:
        crop_height, crop_width = height, int(height * target_ratio)
    elif current_ratio < target_ratio:
        crop_height, crop_width = int(width / target_ratio), width
    else:
        crop_height, crop_width = height, width
    if crop_height <= 0 or crop_width <= 0:
        raise RuntimeError(
            f"Navigation frame has invalid size {width}x{height}: {frame_path}"
        )
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    frame = frame[top : top + crop_height, left : left + crop_width]
    frame = frame.permute(2, 0, 1).unsqueeze(0)
    frame = F.interpolate(
        frame,
        size=(int(image_height), int(image_width)),
        mode="bilinear",
        align_corners=False,
    )
    return frame.squeeze(0).permute(1, 2, 0).contiguous()


def wrap_yaw(value: float | np.ndarray) -> float | np.ndarray:
    """Wrap radians to ``[-pi, pi)`` (the exact training boundary convention)."""

    array = np.asarray(value, dtype=np.float64)
    wrapped = array - 2.0 * np.pi * np.floor((array + np.pi) / (2.0 * np.pi))
    return float(wrapped) if wrapped.ndim == 0 else wrapped


def _as_path(value: Any, field_name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise TypeError(f"{field_name} must be a non-empty path")
    return Path(value).expanduser().resolve()


def _xy_pair(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric dx and dy") from exc
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain exactly two finite values")
    return result


def _split_from_manifest(value: Any) -> str:
    if not isinstance(value, (str, Path)):
        raise TypeError("split is required when split_manifest is not a path")
    parts = [part.lower() for part in Path(value).parts]
    matches = [split for split in _SPLITS if split in parts]
    if len(matches) != 1:
        raise ValueError(
            "split must be explicit unless split_manifest has exactly one train/test component"
        )
    return matches[0]


def _read_manifest(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation split manifest does not exist: {path}")
    try:
        names = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            f"Cannot read evaluation split manifest {path}: {exc}"
        ) from exc
    if not names:
        raise ValueError(f"Evaluation split manifest is empty: {path}")
    if len(names) != len(set(names)):
        raise ValueError(
            f"Evaluation split manifest contains duplicate trajectories: {path}"
        )
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe trajectory name {name!r} in manifest {path}")
    return names


def _load_partition(spec: DatasetSplitSpec) -> DatasetPartition:
    if not spec.data_root.is_dir():
        raise FileNotFoundError(
            f"Navigation data root does not exist: {spec.data_root}"
        )
    names = _read_manifest(spec.split_manifest)
    trajectories: list[TrajectoryRecord] = []
    missing: list[str] = []
    for name in names:
        directory = (spec.data_root / name).resolve()
        try:
            directory.relative_to(spec.data_root)
        except ValueError as exc:
            raise ValueError(
                f"Trajectory {name!r} escapes data root {spec.data_root}"
            ) from exc
        if not directory.exists():
            # CompACT's public HuRoN manifests intentionally contain a small
            # number of trajectories absent from the mounted release.  Match
            # training: audit those omissions, but never substitute unlisted
            # directories.  Any *existing* malformed entry still fails below.
            missing.append(name)
            continue
        if not directory.is_dir():
            raise RuntimeError(
                f"Existing trajectory path is not a directory: {directory}"
            )
        trajectories.append(_load_trajectory(spec, directory, name))
    if not trajectories:
        raise ValueError(
            f"Source {spec.dataset_id}/{spec.split} has no usable trajectories from "
            f"manifest {spec.split_manifest}"
        )
    return DatasetPartition(
        spec=spec,
        trajectories=tuple(trajectories),
        manifest_sha256=hashlib.sha256(spec.split_manifest.read_bytes()).hexdigest(),
        manifest_trajectory_count=len(names),
        missing_trajectories=tuple(missing),
    )


def _load_trajectory(
    spec: DatasetSplitSpec, directory: Path, trajectory_name: str
) -> TrajectoryRecord:
    metadata_path = directory / "traj_data.pkl"
    if not metadata_path.is_file():
        raise RuntimeError(
            f"Existing trajectory is corrupt (missing traj_data.pkl): {directory}"
        )
    try:
        with metadata_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read trajectory metadata {metadata_path}: {exc}"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or "position" not in payload
        or "yaw" not in payload
    ):
        raise RuntimeError(
            f"Trajectory metadata must contain position and yaw: {metadata_path}"
        )
    try:
        positions = np.asarray(payload["position"], dtype=np.float64)
        yaw = np.asarray(payload["yaw"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid position/yaw arrays in {metadata_path}: {exc}"
        ) from exc
    if positions.ndim != 2 or positions.shape[1] < 2 or positions.shape[0] == 0:
        raise RuntimeError(
            f"position must have shape [L,2+] with L>0 in {metadata_path}; got {positions.shape}"
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

    try:
        with os.scandir(directory) as entries:
            frame_names = {entry.name for entry in entries if entry.is_file()}
    except OSError as exc:
        raise RuntimeError(
            f"Cannot scan trajectory directory {directory}: {exc}"
        ) from exc
    for frame in range(int(positions.shape[0])):
        if f"{frame}.jpg" not in frame_names:
            raise RuntimeError(
                "Existing trajectory is corrupt (expected contiguous 0..L-1 jpg files; "
                f"missing frame {frame}): {directory}"
            )

    positions.setflags(write=False)
    yaw.setflags(write=False)
    return TrajectoryRecord(
        dataset_id=spec.dataset_id,
        split=spec.split,
        name=trajectory_name,
        directory=directory,
        waypoint_spacing=spec.waypoint_spacing,
        positions=positions,
        yaw=yaw,
    )
