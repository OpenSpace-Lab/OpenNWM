"""Offline proxy data support for two-stage NWM training.

This module is deliberately independent from the real-action navigation
datasets.  ``NavAnywhereDataset`` only reads RGB frames and offline proxy
records; it never imports or invokes a geometry, IDM, or latent-action model.
"""

from __future__ import annotations

import bisect
import json
import os
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from navanywhere_recipe import (
    DEFAULT_GOAL_STRATA,
    affine_observation_slot,
    coprime_stride,
    frame_indices_sha256,
    load_sampling_recipe,
    scan_trajectory_frames,
    sample_recipe_goal_offsets,
    validate_sampling_recipe,
)


PROXY_ACTION_MODES = ("geometry", "idm", "latent")
NAVANYWHERE_ACTION_MODES = ("none",) + PROXY_ACTION_MODES
LOCAL_PROXY_MAX_ABS_FRAME_OFFSET = 8


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def proxy_offset_mask(
    frame_offsets: Sequence[int] | np.ndarray | torch.Tensor,
    max_abs_frame_offset: int = 8,
) -> torch.Tensor:
    """Return the inclusive local-pair eligibility mask from raw offsets."""
    limit = int(max_abs_frame_offset)
    if limit < 0:
        raise ValueError("max_abs_frame_offset must be non-negative")
    offsets = torch.as_tensor(frame_offsets, dtype=torch.int64)
    return offsets.abs() <= limit


def format_proxy_sample_key(
    source_id: str,
    trajectory_id: str,
    current_frame: int,
    target_frame: int,
) -> str:
    """Format all four fields that uniquely identify an offline proxy record."""
    return (
        f"source_id={source_id!r}, trajectory_id={trajectory_id!r}, "
        f"current_frame={int(current_frame)}, target_frame={int(target_frame)}"
    )


class ProxyLookupError(RuntimeError):
    """Raised when strict offline proxy loading cannot satisfy a sample key."""


@dataclass(frozen=True, slots=True)
class ProxyLookupResult:
    """Result of one offline proxy lookup.

    ``found`` means that the requested frame-pair record was present.
    ``invalid`` distinguishes a malformed cache/record from an ordinary missing
    record.  ``proxy_action`` is a fixed-width zero placeholder unless ``valid``
    is true; callers must still apply the validity mask after action encoding.
    """

    proxy_action: torch.Tensor
    found: bool
    invalid: bool
    valid: bool
    sample_key: str
    error: str | None = None

    def __post_init__(self) -> None:
        if self.proxy_action.ndim != 1:
            raise ValueError("ProxyLookupResult.proxy_action must have shape [D]")
        if self.proxy_action.dtype != torch.float32:
            raise TypeError("ProxyLookupResult.proxy_action must be float32")
        if bool(self.valid) != (bool(self.found) and not bool(self.invalid)):
            raise ValueError("valid must equal found AND NOT invalid")
        if self.valid and not torch.isfinite(self.proxy_action).all():
            raise ValueError("A valid proxy action must contain only finite values")

    @property
    def missing(self) -> bool:
        return not self.found and not self.invalid

    @property
    def value(self) -> torch.Tensor:
        """Compatibility alias for callers that use store/value terminology."""
        return self.proxy_action

    @property
    def motion(self) -> torch.Tensor:
        return self.proxy_action

    @property
    def proxy_found(self) -> bool:
        return self.found

    @property
    def proxy_invalid(self) -> bool:
        return self.invalid

    @property
    def proxy_valid(self) -> bool:
        return self.valid


@dataclass(slots=True)
class _CachedTrajectory:
    path: str | None
    lookup: dict[tuple[int, int], int]
    values: torch.Tensor | None
    invalid_pairs: set[tuple[int, int]]
    global_error: str | None = None
    missing_file: bool = False


class OfflineProxyStore:
    """Lazy LRU reader for per-trajectory ``.pt`` or ``.npz`` proxy caches.

    The default file layout is ``root/source_id/trajectory_id.{pt,npz}``.
    ``file_pattern`` may include ``source_id``, ``trajectory_id``, and
    ``proxy_type`` fields.  If it has no suffix, exactly one of ``.pt`` and
    ``.npz`` must exist.  Canonical payload fields are ``frame_pairs`` with
    shape ``[N,2]`` and ``motion`` with shape ``[N,D]``.
    """

    def __init__(
        self,
        *,
        root: str | os.PathLike[str] | None = None,
        storage_path: str | os.PathLike[str] | None = None,
        proxy_type: str | None = None,
        motion_type: str | None = None,
        dim: int | None = None,
        input_dim: int | None = None,
        strict_loading: bool = False,
        file_pattern: str = "{source_id}/{trajectory_id}",
        pairs_key: str = "frame_pairs",
        values_key: str = "motion",
        validity_key: str | None = None,
        cache_size: int = 8,
        max_abs_frame_offset: int = 8,
    ) -> None:
        selected_root = root if root is not None else storage_path
        if selected_root is None:
            raise ValueError("OfflineProxyStore requires root or storage_path")
        if root is not None and storage_path is not None:
            if os.path.realpath(os.fspath(root)) != os.path.realpath(
                os.fspath(storage_path)
            ):
                raise ValueError("root and storage_path refer to different locations")
        self.root = os.path.realpath(os.path.expanduser(os.fspath(selected_root)))
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Offline proxy root does not exist: {self.root}")

        selected_type = proxy_type if proxy_type is not None else motion_type
        self.proxy_type = str(selected_type).lower() if selected_type is not None else ""
        if self.proxy_type not in PROXY_ACTION_MODES:
            raise ValueError(
                f"proxy_type must be one of {PROXY_ACTION_MODES}, got {selected_type!r}"
            )
        self.motion_type = self.proxy_type

        selected_dim = dim if dim is not None else input_dim
        if selected_dim is None or int(selected_dim) < 1:
            raise ValueError("OfflineProxyStore dim must be a positive integer")
        if dim is not None and input_dim is not None and int(dim) != int(input_dim):
            raise ValueError("dim and input_dim disagree")
        self.dim = int(selected_dim)
        self.input_dim = self.dim

        self.strict_loading = bool(strict_loading)
        self.file_pattern = str(file_pattern)
        self.pairs_key = str(pairs_key)
        self.values_key = str(values_key)
        self.validity_key = None if validity_key is None else str(validity_key)
        self.cache_size = int(cache_size)
        self.max_abs_frame_offset = int(max_abs_frame_offset)
        if self.cache_size < 1:
            raise ValueError("cache_size must be at least one")
        if self.max_abs_frame_offset < 0:
            raise ValueError("max_abs_frame_offset must be non-negative")
        self._cache: OrderedDict[tuple[str, str], _CachedTrajectory] = OrderedDict()

    @staticmethod
    def _scalar(value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            value = value.item()
        elif isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    def _safe_base_path(self, source_id: str, trajectory_id: str) -> str:
        try:
            relative = self.file_pattern.format(
                source_id=source_id,
                dataset_name=source_id,
                trajectory_id=trajectory_id,
                trajectory_name=trajectory_id,
                proxy_type=self.proxy_type,
                motion_type=self.proxy_type,
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid proxy file_pattern {self.file_pattern!r}: {exc}") from exc
        path = os.path.realpath(os.path.join(self.root, relative))
        try:
            common = os.path.commonpath((self.root, path))
        except ValueError as exc:
            raise ValueError("Proxy cache path is on a different filesystem root") from exc
        if common != self.root or path == self.root:
            raise ValueError(
                f"Unsafe proxy path for source={source_id!r}, trajectory={trajectory_id!r}"
            )
        return path

    def _resolve_path(
        self, source_id: str, trajectory_id: str
    ) -> tuple[str | None, str | None]:
        base = self._safe_base_path(source_id, trajectory_id)
        if base.endswith((".pt", ".npz")):
            candidates = [base]
        else:
            candidates = [base + ".pt", base + ".npz"]
        existing = [path for path in candidates if os.path.isfile(path)]
        if len(existing) == 1:
            return existing[0], None
        if len(existing) > 1:
            return None, (
                "Ambiguous proxy cache: both supported files exist: "
                + ", ".join(existing)
            )
        return None, "Proxy cache file is missing; checked " + ", ".join(candidates)

    def _read_payload(self, path: str) -> Mapping[str, Any]:
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as archive:
                return {key: archive[key] for key in archive.files}
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise TypeError("cache must contain a mapping")
        return payload

    @staticmethod
    def _integer_pairs(value: Any) -> torch.Tensor:
        pairs = torch.as_tensor(value)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(
                f"frame_pairs must have shape [N,2], got {tuple(pairs.shape)}"
            )
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if pairs.dtype not in integer_dtypes:
            raise TypeError(f"frame_pairs must use an integer dtype, got {pairs.dtype}")
        return pairs.to(dtype=torch.int64).contiguous()

    def _load_trajectory(self, source_id: str, trajectory_id: str) -> _CachedTrajectory:
        cache_key = (source_id, trajectory_id)
        cached = self._cache.pop(cache_key, None)
        if cached is not None:
            self._cache[cache_key] = cached
            return cached

        path, resolution_error = self._resolve_path(source_id, trajectory_id)
        if path is None:
            cached = _CachedTrajectory(
                path=None,
                lookup={},
                values=None,
                invalid_pairs=set(),
                global_error=resolution_error,
                missing_file=bool(resolution_error and resolution_error.startswith("Proxy cache file is missing")),
            )
            return self._remember(cache_key, cached)

        try:
            payload = self._read_payload(path)
        except Exception as exc:
            cached = _CachedTrajectory(
                path=path,
                lookup={},
                values=None,
                invalid_pairs=set(),
                global_error=f"Failed to read proxy cache {path}: {type(exc).__name__}: {exc}",
            )
            return self._remember(cache_key, cached)

        errors: list[str] = []
        for metadata_key in ("proxy_type", "motion_type", "action_mode"):
            if metadata_key in payload:
                actual_type = str(self._scalar(payload[metadata_key])).lower()
                if actual_type != self.proxy_type:
                    errors.append(
                        f"{metadata_key} mismatch: {actual_type!r} != {self.proxy_type!r}"
                    )
                break
        for metadata_key, expected in (
            ("source_id", source_id),
            ("trajectory_id", trajectory_id),
        ):
            if metadata_key in payload:
                actual = str(self._scalar(payload[metadata_key]))
                if actual != expected:
                    errors.append(
                        f"{metadata_key} mismatch: {actual!r} != {expected!r}"
                    )

        lookup: dict[tuple[int, int], int] = {}
        invalid_pairs: set[tuple[int, int]] = set()
        pairs: torch.Tensor | None = None
        if self.pairs_key not in payload:
            errors.append(f"required key {self.pairs_key!r} is missing")
        else:
            try:
                pairs = self._integer_pairs(payload[self.pairs_key])
                for row, raw_pair in enumerate(pairs.tolist()):
                    pair = (int(raw_pair[0]), int(raw_pair[1]))
                    if pair in lookup:
                        invalid_pairs.add(pair)
                    else:
                        lookup[pair] = row
            except Exception as exc:
                errors.append(f"invalid {self.pairs_key}: {type(exc).__name__}: {exc}")

        values: torch.Tensor | None = None
        if self.values_key not in payload:
            errors.append(f"required key {self.values_key!r} is missing")
        elif pairs is not None:
            try:
                values = torch.as_tensor(payload[self.values_key], dtype=torch.float32)
                expected_shape = (pairs.shape[0], self.dim)
                if tuple(values.shape) != expected_shape:
                    raise ValueError(
                        f"motion must have shape {expected_shape}, got {tuple(values.shape)}"
                    )
                finite_rows = torch.isfinite(values).all(dim=1)
                for row in torch.nonzero(~finite_rows, as_tuple=False).flatten().tolist():
                    pair = (int(pairs[row, 0]), int(pairs[row, 1]))
                    invalid_pairs.add(pair)
                values = values.contiguous()
            except Exception as exc:
                values = None
                errors.append(f"invalid {self.values_key}: {type(exc).__name__}: {exc}")

        detected_validity_key = self.validity_key
        if detected_validity_key is None:
            detected_validity_key = next(
                (
                    key
                    for key in ("proxy_valid", "motion_valid", "valid_mask")
                    if key in payload
                ),
                None,
            )
        if detected_validity_key is not None and pairs is not None:
            if detected_validity_key not in payload:
                errors.append(f"required key {detected_validity_key!r} is missing")
            else:
                try:
                    validity = torch.as_tensor(payload[detected_validity_key])
                    if validity.shape != (pairs.shape[0],):
                        raise ValueError(
                            f"{detected_validity_key} must have shape "
                            f"{(pairs.shape[0],)}, got {tuple(validity.shape)}"
                        )
                    if validity.dtype != torch.bool:
                        if not torch.all((validity == 0) | (validity == 1)):
                            raise ValueError(
                                f"{detected_validity_key} must be boolean or binary"
                            )
                        validity = validity.to(dtype=torch.bool)
                    for row in torch.nonzero(
                        ~validity, as_tuple=False
                    ).flatten().tolist():
                        pair = (int(pairs[row, 0]), int(pairs[row, 1]))
                        invalid_pairs.add(pair)
                except Exception as exc:
                    errors.append(
                        f"invalid {detected_validity_key}: "
                        f"{type(exc).__name__}: {exc}"
                    )

        if invalid_pairs:
            # Duplicate and non-finite rows remain pair-local failures.  This lets
            # tolerant training use other valid records in the same trajectory.
            duplicate_count = len(invalid_pairs)
            pair_error = f"{duplicate_count} duplicate or non-finite frame-pair record(s)"
        else:
            pair_error = None
        global_error = "; ".join(errors) if errors else None
        cached = _CachedTrajectory(
            path=path,
            lookup=lookup,
            values=values,
            invalid_pairs=invalid_pairs,
            global_error=global_error,
        )
        # Preserve a useful summary for debuggers without invalidating good rows.
        if pair_error and global_error:
            cached.global_error = global_error
        return self._remember(cache_key, cached)

    def _remember(
        self, key: tuple[str, str], value: _CachedTrajectory
    ) -> _CachedTrajectory:
        self._cache[key] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def _finish(
        self,
        *,
        sample_key: str,
        action: torch.Tensor | None,
        found: bool,
        invalid: bool,
        error: str | None,
    ) -> ProxyLookupResult:
        valid = bool(found and not invalid)
        result = ProxyLookupResult(
            proxy_action=(
                action.detach().clone().to(dtype=torch.float32)
                if valid and action is not None
                else torch.zeros(self.dim, dtype=torch.float32)
            ),
            found=bool(found),
            invalid=bool(invalid),
            valid=valid,
            sample_key=sample_key,
            error=error,
        )
        if self.strict_loading and not result.valid:
            state = "invalid" if result.invalid else "missing"
            raise ProxyLookupError(
                f"Offline {self.proxy_type} proxy is {state} for {sample_key}: {error}"
            )
        return result

    def lookup(
        self,
        source_id: str | None = None,
        trajectory_id: str | None = None,
        current_frame: int | None = None,
        target_frame: int | None = None,
        *,
        dataset_name: str | None = None,
        trajectory_name: str | None = None,
        dataset_or_source_id: str | None = None,
        trajectory_or_video_id: str | None = None,
    ) -> ProxyLookupResult:
        """Look up exactly one four-part sample key without fallback or extrapolation."""
        alias_source = dataset_name if dataset_name is not None else dataset_or_source_id
        if (
            dataset_name is not None
            and dataset_or_source_id is not None
            and str(dataset_name) != str(dataset_or_source_id)
        ):
            raise ValueError("dataset_name and dataset_or_source_id disagree")
        if source_id is None:
            source_id = alias_source
        elif alias_source is not None and str(source_id) != str(alias_source):
            raise ValueError("source_id and its dataset alias disagree")
        alias_trajectory = (
            trajectory_name
            if trajectory_name is not None
            else trajectory_or_video_id
        )
        if (
            trajectory_name is not None
            and trajectory_or_video_id is not None
            and str(trajectory_name) != str(trajectory_or_video_id)
        ):
            raise ValueError("trajectory_name and trajectory_or_video_id disagree")
        if trajectory_id is None:
            trajectory_id = alias_trajectory
        elif alias_trajectory is not None and str(trajectory_id) != str(alias_trajectory):
            raise ValueError("trajectory_id and its trajectory alias disagree")
        if source_id is None or trajectory_id is None:
            raise TypeError(
                "lookup requires source_id/dataset_name and "
                "trajectory_id/trajectory_name"
            )
        if current_frame is None or target_frame is None:
            raise TypeError("lookup requires current_frame and target_frame")
        source_id = str(source_id)
        trajectory_id = str(trajectory_id)
        current_frame = int(current_frame)
        target_frame = int(target_frame)
        sample_key = format_proxy_sample_key(
            source_id, trajectory_id, current_frame, target_frame
        )
        cached = self._load_trajectory(source_id, trajectory_id)
        pair = (current_frame, target_frame)
        found = pair in cached.lookup

        if cached.missing_file:
            return self._finish(
                sample_key=sample_key,
                action=None,
                found=False,
                invalid=False,
                error=cached.global_error,
            )
        if cached.global_error is not None:
            return self._finish(
                sample_key=sample_key,
                action=None,
                found=found,
                invalid=True,
                error=f"{cached.global_error}; path={cached.path}",
            )
        if not found:
            return self._finish(
                sample_key=sample_key,
                action=None,
                found=False,
                invalid=False,
                error=f"frame pair {pair} is absent from {cached.path}",
            )
        if pair in cached.invalid_pairs:
            return self._finish(
                sample_key=sample_key,
                action=None,
                found=True,
                invalid=True,
                error=f"frame pair {pair} failed offline validity checks in {cached.path}",
            )
        if cached.values is None:
            return self._finish(
                sample_key=sample_key,
                action=None,
                found=True,
                invalid=True,
                error=f"motion tensor is unavailable in {cached.path}",
            )
        row = cached.lookup[pair]
        return self._finish(
            sample_key=sample_key,
            action=cached.values[row],
            found=True,
            invalid=False,
            error=None,
        )

    def lookup_many(
        self,
        source_id: str | None = None,
        trajectory_id: str | None = None,
        current_frame: int | None = None,
        target_frames: Sequence[int] | None = None,
        *,
        dataset_name: str | None = None,
        trajectory_name: str | None = None,
        dataset_or_source_id: str | None = None,
        trajectory_or_video_id: str | None = None,
    ) -> list[ProxyLookupResult]:
        """Look up targets in caller order; no nearest-neighbour substitution occurs."""
        if target_frames is None:
            raise TypeError("lookup_many requires target_frames")
        return [
            self.lookup(
                source_id,
                trajectory_id,
                current_frame,
                target_frame,
                dataset_name=dataset_name,
                trajectory_name=trajectory_name,
                dataset_or_source_id=dataset_or_source_id,
                trajectory_or_video_id=trajectory_or_video_id,
            )
            for target_frame in target_frames
        ]


@dataclass(slots=True)
class _TrajectoryRecord:
    source_id: str
    trajectory_id: str
    frame_indices: np.ndarray
    frame_paths: tuple[str, ...]
    observation_positions: np.ndarray


class NavAnywhereDataset(Dataset):
    """Image-only NavAnywhere data with optional offline proxy actions.

    Expected image layout is ``root/source_id/trajectory_id/*.jpg``.  The
    returned ``video`` has context frames first and one target per goal after
    them.  Every goal-aligned tensor uses the same order.  Signed offsets are
    sampled from the full configured range (default ``[-64, 64]``); the local
    proxy range only controls whether the store is accessed.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        source_id: str | None = None,
        source_ids: str | Sequence[str] | None = None,
        manifest: str | os.PathLike[str] | Sequence[Any] | Mapping[str, Any] | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
        context_size: int = 1,
        goals_per_obs: int = 1,
        min_frame_offset: int = -64,
        max_frame_offset: int = 64,
        frame_offset_range: Sequence[int] | None = None,
        action_mode: str = "none",
        proxy_store: OfflineProxyStore | None = None,
        proxy: Any | None = None,
        proxy_dim: int | None = None,
        proxy_max_abs_frame_offset: int | None = None,
        strict_loading: bool | None = None,
        fixed_goal_offsets: Sequence[int] | None = None,
        seed: int | None = None,
        sampling_recipe: str | os.PathLike[str] | Mapping[str, Any] | None = None,
        precomputed_latent_root: str | os.PathLike[str] | None = None,
        precomputed_latent_cache_size: int = 8,
        precomputed_latent_metadata: Mapping[str, Any] | None = None,
        precomputed_latent_records: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.root = os.path.realpath(os.path.expanduser(os.fspath(root)))
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"NavAnywhere root does not exist: {self.root}")
        self.transform = transform
        self.context_size = int(context_size)
        self.goals_per_obs = int(goals_per_obs)
        if self.context_size < 1:
            raise ValueError("context_size must be at least one")
        if self.goals_per_obs < 1:
            raise ValueError("goals_per_obs must be at least one")

        if frame_offset_range is not None:
            if len(frame_offset_range) != 2:
                raise ValueError("frame_offset_range must contain [min, max]")
            min_frame_offset, max_frame_offset = map(int, frame_offset_range)
        self.min_frame_offset = int(min_frame_offset)
        self.max_frame_offset = int(max_frame_offset)
        if self.min_frame_offset > self.max_frame_offset:
            raise ValueError("min_frame_offset cannot exceed max_frame_offset")

        self.action_mode = str(action_mode).lower()
        if self.action_mode not in NAVANYWHERE_ACTION_MODES:
            raise ValueError(
                f"NavAnywhere action_mode must be one of {NAVANYWHERE_ACTION_MODES}"
            )
        self.seed = None if seed is None else int(seed)
        self.epoch = 0
        self.sampling_recipe: dict[str, Any] | None = None
        self.sampling_recipe_path: str | None = None
        self.sampling_recipe_sha256: str | None = None

        if source_id is not None:
            if source_ids is not None:
                raise ValueError("Pass source_id or source_ids, not both")
            source_ids = (source_id,)
        if isinstance(source_ids, str):
            source_ids = (source_ids,)
        self.source_ids = None if source_ids is None else tuple(map(str, source_ids))
        if self.source_ids is not None and len(set(self.source_ids)) != len(
            self.source_ids
        ):
            raise ValueError("source_ids contains duplicates")

        if sampling_recipe is not None:
            if manifest is not None:
                raise ValueError(
                    "Pass sampling_recipe or manifest for NavAnywhere, not both"
                )
            recipe_validation = {
                "seed": self.seed,
                "context_size": self.context_size,
                "goals_per_obs": self.goals_per_obs,
                "frame_offset_range": (
                    self.min_frame_offset,
                    self.max_frame_offset,
                ),
            }
            if isinstance(sampling_recipe, (str, os.PathLike)):
                (
                    self.sampling_recipe,
                    self.sampling_recipe_path,
                    self.sampling_recipe_sha256,
                ) = load_sampling_recipe(sampling_recipe, **recipe_validation)
            elif isinstance(sampling_recipe, Mapping):
                self.sampling_recipe = validate_sampling_recipe(
                    sampling_recipe, **recipe_validation
                )
            else:
                raise TypeError("sampling_recipe must be a path or mapping")
            recipe_sources = {
                str(item["source_id"])
                for item in self.sampling_recipe["trajectories"]
            }
            if self.source_ids is not None and recipe_sources != set(self.source_ids):
                raise ValueError(
                    "sampling_recipe sources do not exactly match source_ids: "
                    f"recipe={sorted(recipe_sources)}, configured={sorted(self.source_ids)}"
                )
            manifest = [
                {
                    "source_id": item["source_id"],
                    "trajectory_id": item["trajectory_id"],
                }
                for item in self.sampling_recipe["trajectories"]
            ]

        self.fixed_goal_offsets = (
            None
            if fixed_goal_offsets is None
            else np.asarray(fixed_goal_offsets, dtype=np.int64)
        )
        if self.fixed_goal_offsets is not None:
            if self.fixed_goal_offsets.shape != (self.goals_per_obs,):
                raise ValueError(
                    "fixed_goal_offsets must contain exactly goals_per_obs entries"
                )
            if np.any(self.fixed_goal_offsets < self.min_frame_offset) or np.any(
                self.fixed_goal_offsets > self.max_frame_offset
            ):
                raise ValueError(
                    "fixed_goal_offsets must stay inside the configured offset range"
                )

        self.proxy_store: OfflineProxyStore | None = None
        self.proxy_dim: int | None = None
        self.proxy_max_abs_frame_offset = LOCAL_PROXY_MAX_ABS_FRAME_OFFSET
        if self.action_mode in PROXY_ACTION_MODES:
            if proxy_store is None:
                storage_path = _config_get(proxy, "storage_path", _config_get(proxy, "root"))
                configured_dim = _config_get(proxy, "dim", proxy_dim)
                configured_strict = _config_get(proxy, "strict_loading", False)
                proxy_store = OfflineProxyStore(
                    storage_path=storage_path,
                    proxy_type=self.action_mode,
                    dim=configured_dim,
                    strict_loading=(
                        configured_strict if strict_loading is None else strict_loading
                    ),
                    file_pattern=str(
                        _config_get(proxy, "file_pattern", "{source_id}/{trajectory_id}")
                    ),
                    pairs_key=str(_config_get(proxy, "pairs_key", "frame_pairs")),
                    values_key=str(_config_get(proxy, "values_key", "motion")),
                    validity_key=_config_get(proxy, "validity_key"),
                    cache_size=int(_config_get(proxy, "cache_size", 8)),
                    max_abs_frame_offset=int(
                        _config_get(proxy, "max_abs_frame_offset", 8)
                    ),
                )
            if str(proxy_store.proxy_type) != self.action_mode:
                raise ValueError(
                    f"Proxy store type {proxy_store.proxy_type!r} does not match "
                    f"action_mode={self.action_mode!r}"
                )
            self.proxy_store = proxy_store
            self.proxy_dim = int(
                proxy_dim if proxy_dim is not None else proxy_store.input_dim
            )
            if self.proxy_dim != int(proxy_store.input_dim):
                raise ValueError("proxy_dim does not match the offline proxy store")
            self.proxy_max_abs_frame_offset = int(
                proxy_store.max_abs_frame_offset
                if proxy_max_abs_frame_offset is None
                else proxy_max_abs_frame_offset
            )
            if (
                self.proxy_max_abs_frame_offset
                != LOCAL_PROXY_MAX_ABS_FRAME_OFFSET
                or int(proxy_store.max_abs_frame_offset)
                != LOCAL_PROXY_MAX_ABS_FRAME_OFFSET
            ):
                raise ValueError(
                    "NavAnywhere proxy eligibility is fixed to "
                    "abs(frame_offset)<=8"
                )

        entries = self._read_manifest(manifest)
        if entries is None:
            entries = self._discover_trajectory_entries()
        if self.sampling_recipe is not None and precomputed_latent_root is not None:
            self._trajectories = self._build_precomputed_recipe_trajectories()
        else:
            self._trajectories = self._build_trajectories(entries)
        if not self._trajectories:
            raise ValueError(f"No usable NavAnywhere trajectories found under {self.root}")
        self._cumulative_observations: list[int] = []
        total = 0
        for record in self._trajectories:
            total += int(record.observation_positions.size)
            self._cumulative_observations.append(total)
        if total == 0:
            raise ValueError("No NavAnywhere observation has a valid configured goal")
        self._recipe_sources: tuple[tuple[_TrajectoryRecord, ...], ...] = ()
        self._recipe_affine: dict[tuple[str, str], tuple[int, int]] = {}
        if self.sampling_recipe is not None:
            self._configure_sampling_recipe()

        self.precomputed_latent_root = (
            None
            if precomputed_latent_root is None
            else os.path.realpath(
                os.path.expanduser(os.fspath(precomputed_latent_root))
            )
        )
        self.precomputed_latent_cache_size = int(precomputed_latent_cache_size)
        self.precomputed_latent_metadata = dict(
            precomputed_latent_metadata or {}
        )
        self.precomputed_latent_records = {
            str(source): {
                str(trajectory): dict(record)
                for trajectory, record in records.items()
            }
            for source, records in (precomputed_latent_records or {}).items()
        }
        self._latent_trajectory_cache: OrderedDict[
            tuple[str, str], dict[str, Any]
        ] = OrderedDict()
        if self.precomputed_latent_root is not None:
            if not os.path.isdir(self.precomputed_latent_root):
                raise FileNotFoundError(
                    "NavAnywhere precomputed latent root does not exist: "
                    f"{self.precomputed_latent_root}"
                )
            if self.precomputed_latent_cache_size < 1:
                raise ValueError("precomputed_latent_cache_size must be positive")
            if not self.precomputed_latent_records:
                raise ValueError(
                    "Validated precomputed latent manifest records are required"
                )

    @property
    def action_conditioning_enabled(self) -> bool:
        return self.action_mode != "none"

    @property
    def uses_precomputed_latents(self) -> bool:
        return self.precomputed_latent_root is not None

    @property
    def recipe_summary(self) -> dict[str, Any] | None:
        if self.sampling_recipe is None:
            return None
        return {
            "path": self.sampling_recipe_path,
            "sha256": self.sampling_recipe_sha256,
            "seed": int(self.sampling_recipe["seed"]),
            "samples_per_epoch": int(self.sampling_recipe["samples_per_epoch"]),
            "totals": dict(self.sampling_recipe["totals"]),
            "observation_sampling": self.sampling_recipe[
                "observation_sampling"
            ]["strategy"],
            "goal_sampling": self.sampling_recipe["goal_sampling"]["strategy"],
        }

    def set_epoch(self, epoch: int) -> None:
        """Change deterministic per-index goal sampling between training epochs."""
        self.epoch = int(epoch)

    def _configure_sampling_recipe(self) -> None:
        assert self.sampling_recipe is not None
        expected = {
            (str(item["source_id"]), str(item["trajectory_id"])): item
            for item in self.sampling_recipe["trajectories"]
        }
        actual = {
            (record.source_id, record.trajectory_id): record
            for record in self._trajectories
        }
        if set(expected) != set(actual):
            raise ValueError(
                "NavAnywhere inventory no longer matches the sampling recipe: "
                f"missing={sorted(set(expected) - set(actual))[:5]}, "
                f"unexpected={sorted(set(actual) - set(expected))[:5]}"
            )
        sources: dict[str, list[_TrajectoryRecord]] = {}
        for identity in sorted(expected):
            recipe_record = expected[identity]
            record = actual[identity]
            actual_hash = frame_indices_sha256(record.frame_indices)
            if actual_hash != recipe_record["frame_indices_sha256"]:
                raise ValueError(
                    "NavAnywhere frame inventory changed after recipe creation: "
                    f"source_id={identity[0]!r}, trajectory_id={identity[1]!r}"
                )
            if int(record.frame_indices.size) != int(recipe_record["frame_count"]):
                raise ValueError(f"Recipe frame_count mismatch for {identity}")
            if int(record.observation_positions.size) != int(
                recipe_record["observation_count"]
            ):
                raise ValueError(f"Recipe observation_count mismatch for {identity}")
            sources.setdefault(record.source_id, []).append(record)
            self._recipe_affine[identity] = coprime_stride(
                int(record.observation_positions.size),
                f"{record.source_id}/{record.trajectory_id}",
                int(self.sampling_recipe["seed"]),
            )
        self._recipe_sources = tuple(
            tuple(sources[source]) for source in sorted(sources)
        )
        if not self._recipe_sources:
            raise ValueError("Sampling recipe contains no usable sources")

    def _read_manifest(self, manifest: Any) -> list[Any] | None:
        if manifest is None:
            return None
        payload: Any = manifest
        if isinstance(manifest, (str, os.PathLike)):
            manifest_path = os.path.realpath(os.path.expanduser(os.fspath(manifest)))
            suffix = Path(manifest_path).suffix.lower()
            if suffix == ".json":
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            elif suffix == ".jsonl":
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    payload = [
                        json.loads(line)
                        for line in handle
                        if line.strip() and not line.lstrip().startswith("#")
                    ]
            else:
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    payload = [
                        line.strip()
                        for line in handle
                        if line.strip() and not line.lstrip().startswith("#")
                    ]
        if isinstance(payload, Mapping):
            if "trajectories" in payload:
                payload = payload["trajectories"]
            else:
                expanded = []
                for manifest_source, trajectories in payload.items():
                    for trajectory in trajectories:
                        if isinstance(trajectory, Mapping):
                            expanded.append({"source_id": manifest_source, **trajectory})
                        else:
                            expanded.append(
                                {
                                    "source_id": manifest_source,
                                    "trajectory_id": trajectory,
                                }
                            )
                payload = expanded
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
            raise TypeError("NavAnywhere manifest must resolve to a sequence")
        return list(payload)

    def _discover_trajectory_entries(self) -> list[dict[str, str]]:
        if self.source_ids is None:
            source_names = sorted(
                entry.name
                for entry in os.scandir(self.root)
                if entry.is_dir(follow_symlinks=False)
            )
        else:
            source_names = list(self.source_ids)
        entries: list[dict[str, str]] = []
        for source_name in source_names:
            source_path = self._safe_under_root(source_name)
            if not os.path.isdir(source_path):
                raise FileNotFoundError(
                    f"NavAnywhere source directory does not exist: {source_path}"
                )
            for trajectory in sorted(
                os.scandir(source_path), key=lambda entry: entry.name
            ):
                if trajectory.is_dir(follow_symlinks=False):
                    entries.append(
                        {
                            "source_id": source_name,
                            "trajectory_id": trajectory.name,
                        }
                    )
        return entries

    def _safe_under_root(self, *parts: str) -> str:
        path = os.path.realpath(os.path.join(self.root, *map(str, parts)))
        if os.path.commonpath((self.root, path)) != self.root or path == self.root:
            raise ValueError(f"Unsafe NavAnywhere path components: {parts!r}")
        return path

    def _normalise_entry(self, entry: Any) -> dict[str, Any]:
        if isinstance(entry, str):
            cleaned = entry.replace("\\", "/").strip("/")
            parts = cleaned.split("/", 1)
            if len(parts) == 1:
                if self.source_ids is None or len(self.source_ids) != 1:
                    raise ValueError(
                        f"Manifest trajectory {entry!r} needs an explicit source_id"
                    )
                return {"source_id": self.source_ids[0], "trajectory_id": parts[0]}
            return {"source_id": parts[0], "trajectory_id": parts[1]}
        if not isinstance(entry, Mapping):
            raise TypeError(f"Invalid NavAnywhere manifest entry: {entry!r}")
        source = entry.get(
            "source_id",
            entry.get("dataset_or_source_id", entry.get("dataset_name", entry.get("source"))),
        )
        trajectory = entry.get(
            "trajectory_id",
            entry.get(
                "trajectory_or_video_id",
                entry.get("trajectory_name", entry.get("video_id", entry.get("trajectory"))),
            ),
        )
        if source is None or trajectory is None:
            raise ValueError(
                "Each manifest entry requires source_id and trajectory_id"
            )
        return {**entry, "source_id": str(source), "trajectory_id": str(trajectory)}

    @staticmethod
    def _frame_index(path_or_name: str, explicit_index: Any = None) -> int:
        if explicit_index is not None:
            return int(explicit_index)
        stem = Path(path_or_name).stem
        matches = re.findall(r"-?\d+", stem)
        if not matches:
            raise ValueError(f"Cannot infer integer frame index from {path_or_name!r}")
        return int(matches[-1])

    def _frames_from_entry(
        self, entry: Mapping[str, Any], trajectory_path: str
    ) -> list[tuple[int, str]]:
        supplied = entry.get("frames", entry.get("frame_paths"))
        if supplied is None:
            # Keep training inventory discovery identical to recipe generation
            # and latent precomputation, including retries for transient NAS
            # directory-listing duplicates.
            return scan_trajectory_frames(trajectory_path)

        if isinstance(supplied, Mapping):
            supplied = [
                {"frame_index": frame_index, "path": frame_path}
                for frame_index, frame_path in supplied.items()
            ]
        if not isinstance(supplied, Sequence) or isinstance(supplied, (str, bytes)):
            raise TypeError("Manifest frames must be a sequence or index/path mapping")
        frame_pattern = str(entry.get("frame_pattern", "{frame_index}.jpg"))
        frames: list[tuple[int, str]] = []
        for item in supplied:
            explicit_index = None
            if isinstance(item, Mapping):
                explicit_index = item.get("frame_index", item.get("index"))
                frame_name = item.get("path", item.get("filename"))
                if frame_name is None:
                    if explicit_index is None:
                        raise ValueError("Manifest frame requires a path or frame_index")
                    frame_name = frame_pattern.format(frame_index=int(explicit_index))
            elif isinstance(item, (int, np.integer)):
                explicit_index = int(item)
                frame_name = frame_pattern.format(frame_index=explicit_index)
            else:
                frame_name = str(item)
            frame_index = self._frame_index(str(frame_name), explicit_index)
            frame_path = (
                os.path.realpath(str(frame_name))
                if os.path.isabs(str(frame_name))
                else os.path.realpath(os.path.join(trajectory_path, str(frame_name)))
            )
            if os.path.commonpath((self.root, frame_path)) != self.root:
                raise ValueError(f"Manifest frame escapes NavAnywhere root: {frame_name}")
            if not frame_path.lower().endswith(".jpg"):
                raise ValueError(f"NavAnywhere frame must be a .jpg file: {frame_path}")
            frames.append((frame_index, frame_path))
        return frames

    def _observation_positions(self, frame_indices: np.ndarray) -> np.ndarray:
        candidates = np.arange(
            self.context_size - 1, frame_indices.size, dtype=np.int64
        )
        if candidates.size == 0:
            return candidates
        current = frame_indices[candidates]
        if self.fixed_goal_offsets is not None:
            valid = np.ones(candidates.size, dtype=bool)
            for offset in self.fixed_goal_offsets:
                valid &= np.isin(current + int(offset), frame_indices)
            return candidates[valid]
        if self.min_frame_offset <= 0 <= self.max_frame_offset:
            return candidates
        valid = np.zeros(candidates.size, dtype=bool)
        for offset in range(self.min_frame_offset, self.max_frame_offset + 1):
            valid |= np.isin(current + offset, frame_indices)
        return candidates[valid]

    def _build_trajectories(self, raw_entries: Sequence[Any]) -> list[_TrajectoryRecord]:
        records: list[_TrajectoryRecord] = []
        seen: set[tuple[str, str]] = set()
        allowed_sources = None if self.source_ids is None else set(self.source_ids)
        for raw_entry in raw_entries:
            entry = self._normalise_entry(raw_entry)
            source = entry["source_id"]
            trajectory = entry["trajectory_id"]
            if allowed_sources is not None and source not in allowed_sources:
                continue
            identity = (source, trajectory)
            if identity in seen:
                raise ValueError(f"Duplicate NavAnywhere manifest trajectory: {identity}")
            seen.add(identity)
            trajectory_path = self._safe_under_root(source, trajectory)
            if not os.path.isdir(trajectory_path):
                raise FileNotFoundError(
                    f"NavAnywhere trajectory directory does not exist: {trajectory_path}"
                )
            frames = sorted(self._frames_from_entry(entry, trajectory_path))
            if not frames:
                continue
            indices = np.asarray([frame[0] for frame in frames], dtype=np.int64)
            if np.unique(indices).size != indices.size:
                raise ValueError(
                    f"Duplicate frame indices in NavAnywhere trajectory {identity}"
                )
            positions = self._observation_positions(indices)
            if positions.size == 0:
                continue
            records.append(
                _TrajectoryRecord(
                    source_id=source,
                    trajectory_id=trajectory,
                    frame_indices=indices,
                    frame_paths=tuple(frame[1] for frame in frames),
                    observation_positions=positions,
                )
            )
        return records

    def _build_precomputed_recipe_trajectories(self) -> list[_TrajectoryRecord]:
        """Restore a frozen contiguous inventory without rescanning raw JPEGs.

        Precomputed-latent training never reads ``frame_paths``.  The validated
        recipe and latent manifests are therefore the source of truth, and
        reconstructing their contiguous indices avoids every DDP rank listing
        millions of raw files on NAS during startup.
        """
        assert self.sampling_recipe is not None
        records: list[_TrajectoryRecord] = []
        for item in self.sampling_recipe["trajectories"]:
            source = str(item["source_id"])
            trajectory = str(item["trajectory_id"])
            first = int(item["first_frame_index"])
            last = int(item["last_frame_index"])
            frame_count = int(item["frame_count"])
            if last - first + 1 != frame_count:
                raise ValueError(
                    "Precomputed NavAnywhere startup requires contiguous recipe "
                    f"frame indices for {source!r}/{trajectory!r}"
                )
            indices = np.arange(first, last + 1, dtype=np.int64)
            if frame_indices_sha256(indices) != str(item["frame_indices_sha256"]):
                raise ValueError(
                    "Reconstructed NavAnywhere frame inventory does not match "
                    f"the frozen recipe for {source!r}/{trajectory!r}"
                )
            positions = self._observation_positions(indices)
            if positions.size == 0:
                continue
            records.append(
                _TrajectoryRecord(
                    source_id=source,
                    trajectory_id=trajectory,
                    frame_indices=indices,
                    frame_paths=(),
                    observation_positions=positions,
                )
            )
        return records

    def __len__(self) -> int:
        if self.sampling_recipe is not None:
            return int(self.sampling_recipe["samples_per_epoch"])
        return self._cumulative_observations[-1]

    def _locate(self, index: int) -> tuple[_TrajectoryRecord, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.sampling_recipe is not None:
            source_count = len(self._recipe_sources)
            source_slot = int(index) % source_count
            source_draw = int(index) // source_count
            trajectories = self._recipe_sources[source_slot]
            trajectory_count = len(trajectories)
            trajectory_slot = source_draw % trajectory_count
            trajectory_draw = source_draw // trajectory_count
            record = trajectories[trajectory_slot]
            observations = record.observation_positions
            # Continue each trajectory's affine cycle exactly where the prior
            # epoch ended.  Uneven final round-robin slots receive one fewer
            # draw, so a shared ceil(epoch_size / sources / trajectories)
            # would skip cycle positions and could repeat before full cover.
            source_draws_per_epoch = (
                (len(self) - 1 - source_slot) // source_count + 1
                if source_slot < len(self)
                else 0
            )
            trajectory_draws_per_epoch = (
                (source_draws_per_epoch - 1 - trajectory_slot)
                // trajectory_count
                + 1
                if trajectory_slot < source_draws_per_epoch
                else 0
            )
            stride, shift = self._recipe_affine[
                (record.source_id, record.trajectory_id)
            ]
            observation_slot = affine_observation_slot(
                epoch=self.epoch,
                trajectory_draw=trajectory_draw,
                trajectory_draws_per_epoch=trajectory_draws_per_epoch,
                observation_count=int(observations.size),
                stride=stride,
                shift=shift,
            )
            return record, int(observations[observation_slot])
        trajectory_position = bisect.bisect_right(
            self._cumulative_observations, index
        )
        previous = (
            0
            if trajectory_position == 0
            else self._cumulative_observations[trajectory_position - 1]
        )
        record = self._trajectories[trajectory_position]
        observation_position = int(record.observation_positions[index - previous])
        return record, observation_position

    def _available_offsets(
        self, record: _TrajectoryRecord, current_position: int
    ) -> np.ndarray:
        current_frame = int(record.frame_indices[current_position])
        candidates = np.arange(
            self.min_frame_offset, self.max_frame_offset + 1, dtype=np.int64
        )
        target_indices = current_frame + candidates
        positions = np.searchsorted(record.frame_indices, target_indices)
        valid = positions < record.frame_indices.size
        valid[valid] &= record.frame_indices[positions[valid]] == target_indices[valid]
        return candidates[valid]

    def _sample_offsets(
        self, record: _TrajectoryRecord, current_position: int, dataset_index: int
    ) -> np.ndarray:
        if self.fixed_goal_offsets is not None:
            return self.fixed_goal_offsets.copy()
        available = self._available_offsets(record, current_position)
        if available.size == 0:
            raise RuntimeError("Indexed observation unexpectedly has no valid goals")
        if self.sampling_recipe is not None:
            strata = tuple(
                tuple(map(int, bounds))
                for bounds in self.sampling_recipe["goal_sampling"]["strata"]
            )
            if strata != DEFAULT_GOAL_STRATA:
                raise RuntimeError("Validated goal strata changed unexpectedly")
            return sample_recipe_goal_offsets(
                available,
                seed=int(self.sampling_recipe["seed"]),
                epoch=self.epoch,
                dataset_index=int(dataset_index),
                goals_per_obs=self.goals_per_obs,
                strata=strata,
            )
        if self.seed is None:
            selected = np.random.randint(0, available.size, size=self.goals_per_obs)
        else:
            generator = np.random.default_rng(
                np.random.SeedSequence((self.seed, self.epoch, int(dataset_index)))
            )
            selected = generator.integers(0, available.size, size=self.goals_per_obs)
        return available[selected]

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB").copy()

    def _transform_image(self, path: str) -> torch.Tensor:
        image = self._load_rgb(path)
        if self.transform is None:
            array = np.array(image, dtype=np.uint8, copy=True)
            return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
        transformed = self.transform(image)
        if isinstance(transformed, Mapping):
            if "image" not in transformed:
                raise ValueError("Image transform mapping must contain an 'image' tensor")
            transformed = transformed["image"]
        tensor = torch.as_tensor(transformed, dtype=torch.float32)
        if tensor.ndim != 3:
            raise ValueError(
                f"NavAnywhere transform must return [C,H,W], got {tuple(tensor.shape)}"
            )
        return tensor

    def _latent_path(self, source_id: str, trajectory_id: str) -> str:
        if self.precomputed_latent_root is None:
            raise RuntimeError("Precomputed NavAnywhere latents are disabled")
        path = os.path.realpath(
            os.path.join(
                self.precomputed_latent_root,
                str(source_id),
                f"{trajectory_id}.pt",
            )
        )
        if (
            os.path.commonpath((self.precomputed_latent_root, path))
            != self.precomputed_latent_root
            or path == self.precomputed_latent_root
        ):
            raise ValueError(
                "Unsafe NavAnywhere latent path for "
                f"source_id={source_id!r}, trajectory_id={trajectory_id!r}"
            )
        return path

    def _load_latent_trajectory(
        self, record: _TrajectoryRecord
    ) -> dict[str, Any]:
        identity = (record.source_id, record.trajectory_id)
        cached = self._latent_trajectory_cache.pop(identity, None)
        if cached is not None:
            self._latent_trajectory_cache[identity] = cached
            return cached

        manifest_record = self.precomputed_latent_records.get(
            record.source_id, {}
        ).get(record.trajectory_id)
        if manifest_record is None:
            raise KeyError(
                "Validated NavAnywhere latent manifest has no record for "
                f"source_id={record.source_id!r}, "
                f"trajectory_id={record.trajectory_id!r}"
            )
        path = self._latent_path(record.source_id, record.trajectory_id)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "NavAnywhere latent trajectory file is missing after manifest "
                f"validation: {path}"
            )
        try:
            payload = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        except TypeError:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise TypeError(f"NavAnywhere latent cache must be a mapping: {path}")
        for key in (
            "schema_version",
            "format",
            "dataset_name",
            "trajectory_name",
            "frame_indices",
            "posterior_mean",
            "posterior_logvar",
            "metadata",
        ):
            if key not in payload:
                raise KeyError(f"NavAnywhere latent cache {path} is missing {key!r}")
        if int(payload["schema_version"]) != 1:
            raise ValueError(f"Unsupported latent cache schema_version in {path}")
        if str(payload["format"]) != "sd_vae_posterior_stats":
            raise ValueError(f"Invalid NavAnywhere latent cache format in {path}")
        if str(payload["dataset_name"]) != record.source_id:
            raise ValueError(f"Latent cache source mismatch in {path}")
        if str(payload["trajectory_name"]) != record.trajectory_id:
            raise ValueError(f"Latent cache trajectory mismatch in {path}")

        frame_indices = torch.as_tensor(payload["frame_indices"])
        expected_indices = torch.from_numpy(record.frame_indices)
        if frame_indices.dtype != torch.int64 or not torch.equal(
            frame_indices, expected_indices
        ):
            raise ValueError(
                f"Latent cache frame inventory differs from NavAnywhere recipe: {path}"
            )
        mean = torch.as_tensor(payload["posterior_mean"])
        logvar = torch.as_tensor(payload["posterior_logvar"])
        if mean.dtype != torch.bfloat16 or logvar.dtype != torch.bfloat16:
            raise TypeError(
                f"NavAnywhere latent posterior tensors must be bfloat16: {path}"
            )
        if mean.ndim != 4 or mean.shape != logvar.shape:
            raise ValueError(
                "NavAnywhere latent posterior tensors must have identical "
                f"[N,C,H,W] shapes: {path}"
            )
        if mean.shape[0] != frame_indices.numel() or mean.shape[1] != 4:
            raise ValueError(f"Invalid NavAnywhere latent posterior shape in {path}")
        manifest_shape = manifest_record.get("posterior_shape")
        if manifest_shape is not None and list(mean.shape) != list(manifest_shape):
            raise ValueError(f"Latent manifest posterior_shape mismatch for {path}")
        if int(manifest_record.get("frame_count", -1)) != frame_indices.numel():
            raise ValueError(f"Latent manifest frame_count mismatch for {path}")
        if str(manifest_record.get("frame_indices_sha256", "")) != (
            frame_indices_sha256(record.frame_indices)
        ):
            raise ValueError(f"Latent manifest frame fingerprint mismatch for {path}")
        file_metadata = payload["metadata"]
        if not isinstance(file_metadata, Mapping):
            raise TypeError(f"Latent cache metadata must be a mapping: {path}")
        for key, expected in self.precomputed_latent_metadata.items():
            if file_metadata.get(key) != expected:
                raise ValueError(
                    "NavAnywhere latent file metadata mismatch for "
                    f"{key}: {file_metadata.get(key)!r} != {expected!r}; file={path}"
                )

        value = {
            "frame_indices": frame_indices,
            "posterior_mean": mean,
            "posterior_logvar": logvar,
        }
        self._latent_trajectory_cache[identity] = value
        while len(self._latent_trajectory_cache) > self.precomputed_latent_cache_size:
            self._latent_trajectory_cache.popitem(last=False)
        return value

    def _get_precomputed_posteriors(
        self, record: _TrajectoryRecord, frame_indices: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._load_latent_trajectory(record)
        cached_indices = cached["frame_indices"]
        requested = torch.as_tensor(frame_indices, dtype=torch.int64)
        positions = torch.searchsorted(cached_indices, requested)
        if bool((positions >= cached_indices.numel()).any()) or not torch.equal(
            cached_indices[positions], requested
        ):
            raise KeyError(
                "Precomputed NavAnywhere latent cache does not contain every "
                f"requested frame for {record.source_id}/{record.trajectory_id}: "
                f"{requested.tolist()}"
            )
        mean = cached["posterior_mean"].index_select(0, positions).clone()
        logvar = cached["posterior_logvar"].index_select(0, positions).clone()
        if not torch.isfinite(mean).all() or not torch.isfinite(logvar).all():
            raise ValueError(
                "Non-finite NavAnywhere latent posterior selected for "
                f"{record.source_id}/{record.trajectory_id}"
            )
        return mean, logvar

    def __getitem__(self, index: int) -> dict[str, Any]:
        record, current_position = self._locate(int(index))
        current_frame = int(record.frame_indices[current_position])
        frame_offsets = self._sample_offsets(record, current_position, int(index))
        target_frames = current_frame + frame_offsets
        target_positions = np.searchsorted(record.frame_indices, target_frames)
        if np.any(target_positions >= record.frame_indices.size) or not np.array_equal(
            record.frame_indices[target_positions], target_frames
        ):
            raise RuntimeError("Goal sampling produced a nonexistent target frame")

        context_positions = range(
            current_position - self.context_size + 1, current_position + 1
        )
        selected_positions = list(context_positions)
        selected_positions.extend(int(position) for position in target_positions)

        offsets = torch.as_tensor(frame_offsets, dtype=torch.int64)
        rel_t = offsets.to(dtype=torch.float32) / 128.0
        eligible = torch.zeros(self.goals_per_obs, dtype=torch.bool)
        found = torch.zeros_like(eligible)
        invalid = torch.zeros_like(eligible)
        valid = torch.zeros_like(eligible)
        missing = torch.zeros_like(eligible)
        sample: dict[str, Any] = {
            "k": rel_t,
            "rel_t": rel_t.clone(),
            "frame_offset": offsets,
            "target_frame": torch.as_tensor(target_frames, dtype=torch.int64),
            "current_frame": torch.tensor(current_frame, dtype=torch.int64),
            "source_id": record.source_id,
            "trajectory_id": record.trajectory_id,
            "motion_type": self.action_mode,
            "proxy_eligible": eligible,
            "proxy_found": found,
            "proxy_invalid": invalid,
            "proxy_valid": valid,
            "proxy_missing": missing,
            "action_conditioning_enabled": self.action_conditioning_enabled,
        }
        if self.uses_precomputed_latents:
            selected_frames = [
                int(record.frame_indices[position]) for position in selected_positions
            ]
            posterior_mean, posterior_logvar = self._get_precomputed_posteriors(
                record, selected_frames
            )
            sample["posterior_mean"] = posterior_mean
            sample["posterior_logvar"] = posterior_logvar
        else:
            image_paths = [record.frame_paths[position] for position in selected_positions]
            sample["video"] = torch.stack(
                [self._transform_image(path) for path in image_paths]
            )

        if self.action_mode == "none":
            return sample

        assert self.proxy_store is not None and self.proxy_dim is not None
        eligible.copy_(proxy_offset_mask(offsets, self.proxy_max_abs_frame_offset))
        motion = torch.zeros(
            (self.goals_per_obs, self.proxy_dim), dtype=torch.float32
        )
        for goal_index in torch.nonzero(eligible, as_tuple=False).flatten().tolist():
            result = self.proxy_store.lookup(
                record.source_id,
                record.trajectory_id,
                current_frame,
                int(target_frames[goal_index]),
            )
            if result.proxy_action.shape != (self.proxy_dim,):
                raise ValueError(
                    f"Proxy result for {result.sample_key} has shape "
                    f"{tuple(result.proxy_action.shape)}, expected {(self.proxy_dim,)}"
                )
            found[goal_index] = result.found
            invalid[goal_index] = result.invalid
            valid[goal_index] = result.valid
            missing[goal_index] = result.missing
            if result.valid:
                motion[goal_index] = result.proxy_action

        if not torch.equal(valid, eligible & found & ~invalid):
            raise RuntimeError("Internal proxy validity invariant was violated")
        if not torch.equal(missing, eligible & ~found & ~invalid):
            raise RuntimeError("Internal proxy missing-record invariant was violated")
        goal_count = self.goals_per_obs
        if not all(
            tensor.shape[0] == goal_count
            for tensor in (
                offsets,
                rel_t,
                eligible,
                found,
                invalid,
                valid,
                missing,
                motion,
            )
        ):
            raise RuntimeError("Goal-aligned NavAnywhere fields have inconsistent shapes")
        sample["motion"] = motion
        sample["proxy_action"] = motion.clone()
        sample["motion_mask"] = valid.clone()
        return sample


__all__ = [
    "NAVANYWHERE_ACTION_MODES",
    "PROXY_ACTION_MODES",
    "NavAnywhereDataset",
    "OfflineProxyStore",
    "ProxyLookupError",
    "ProxyLookupResult",
    "format_proxy_sample_key",
    "proxy_offset_mask",
]
