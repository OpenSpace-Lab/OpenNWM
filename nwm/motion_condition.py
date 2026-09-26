"""Shared motion-condition utilities for NWM training and inference.

The model-facing representation is grouped by motion type instead of padded to
the largest input dimension.  Consequently a ``none`` sample has no action
tensor and never passes through a zero/null action embedding.
"""

from __future__ import annotations

import os
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import default_collate

MOTION_TYPES = ("none", "real", "geometry", "idm", "latent")
MOTION_ADAPTER_TYPES = MOTION_TYPES[1:]
OFFLINE_MOTION_SCHEMA_VERSION = 1


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def motion_input_dim(config: Any, motion_type: str) -> int:
    """Read and validate the configured input dimension for one adapter."""
    if motion_type not in MOTION_ADAPTER_TYPES:
        raise ValueError(f"No input dimension exists for motion type {motion_type!r}")
    type_config = _config_get(config, motion_type)
    dim_key = f"{motion_type}_dim"
    input_dim = _config_get(type_config, dim_key)
    if input_dim is None:
        raise ValueError(f"motion_condition.{motion_type}.{dim_key} is required")
    input_dim = int(input_dim)
    if input_dim < 1:
        raise ValueError(f"{dim_key} must be positive, got {input_dim}")
    if motion_type in {"real", "geometry"} and input_dim != 3:
        raise ValueError(
            f"{dim_key} must be 3 for planar (delta_x, delta_y, delta_yaw) motion"
        )
    return input_dim


def motion_offset_mask(
    frame_offsets: Sequence[int] | np.ndarray | torch.Tensor,
    max_frame_offset: int,
) -> torch.Tensor:
    """Select labels whose signed frame offset is inside an inclusive range."""
    limit = int(max_frame_offset)
    if limit < 0:
        raise ValueError("max_frame_offset must be non-negative")
    offsets = torch.as_tensor(frame_offsets, dtype=torch.int64)
    return offsets.abs() <= limit


def validate_motion_types(config: Any, *, training: bool = False) -> tuple[str, ...]:
    """Validate available/train motion types and return the selected list."""
    available = tuple(_config_get(config, "available_types", MOTION_TYPES))
    if not available:
        raise ValueError("motion_condition.available_types cannot be empty")
    unknown = set(available).difference(MOTION_TYPES)
    if unknown:
        raise ValueError(f"Unknown available motion type(s): {sorted(unknown)}")
    if len(set(available)) != len(available):
        raise ValueError("motion_condition.available_types contains duplicates")

    selected = (
        tuple(_config_get(config, "train_types", ("real",))) if training else available
    )
    if not selected:
        raise ValueError("motion_condition.train_types cannot be empty")
    unknown = set(selected).difference(available)
    if unknown:
        raise ValueError(
            "motion_condition.train_types must be a subset of available_types; "
            f"got {sorted(unknown)}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("motion_condition.train_types contains duplicates")
    return selected


def resolve_dataset_motion_map(
    config: Any, dataset_names: Sequence[str]
) -> dict[str, str] | None:
    """Resolve an optional fixed dataset -> motion type training assignment."""
    configured = _config_get(config, "dataset_motion_types", {})
    if configured is None or len(configured) == 0:
        return None
    if not isinstance(configured, Mapping):
        raise TypeError("motion_condition.dataset_motion_types must be a mapping")

    names = tuple(str(name) for name in dataset_names)
    if len(set(names)) != len(names):
        raise ValueError("Training dataset names must be unique")
    resolved = {str(name): str(motion_type) for name, motion_type in configured.items()}
    missing = set(names).difference(resolved)
    unknown = set(resolved).difference(names)
    if missing or unknown:
        raise ValueError(
            "motion_condition.dataset_motion_types must cover every training dataset "
            f"exactly; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    train_types = set(validate_motion_types(config, training=True))
    invalid = set(resolved.values()).difference(train_types)
    if invalid:
        raise ValueError(
            "Every dataset motion type must be listed in train_types; "
            f"invalid={sorted(invalid)}"
        )
    unused = train_types.difference(resolved.values())
    if unused:
        raise ValueError(
            "motion_condition.train_types contains types unused by "
            "dataset_motion_types: "
            f"{sorted(unused)}"
        )
    return {name: resolved[name] for name in names}


class MotionAdapter(nn.Module):
    """Map one motion representation into the shared DiT condition space."""

    def __init__(self, input_dim: int, adapter_dim: int, condition_dim: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.condition_dim = int(condition_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, int(adapter_dim)),
            nn.SiLU(),
            nn.Linear(int(adapter_dim), self.condition_dim),
            nn.LayerNorm(self.condition_dim),
        )

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        if motion.ndim != 2 or motion.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected motion [N, {self.input_dim}], got {tuple(motion.shape)}"
            )
        return self.net(motion)


class RealToLatentAdapter(nn.Module):
    """Small real-action to latent-action mapper used by finetune scheme C.

    The mapper intentionally receives only the real action.  Relative time is
    embedded by CDiT's existing time embedder and is never concatenated here.
    """

    def __init__(self, real_action_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        if min(real_action_dim, hidden_dim, latent_dim) < 1:
            raise ValueError("real, hidden, and latent dimensions must be positive")
        self.real_action_dim = int(real_action_dim)
        self.latent_dim = int(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(self.real_action_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), self.latent_dim),
        )

    def forward(self, real_action: torch.Tensor) -> torch.Tensor:
        if real_action.ndim != 2 or real_action.shape[-1] != self.real_action_dim:
            raise ValueError(
                f"Expected real action [N, {self.real_action_dim}], got "
                f"{tuple(real_action.shape)}"
            )
        return self.net(real_action)


class _StateSelfAttentionBlock(nn.Module):
    """Pre-norm self-attention block used only by the state controller."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        mlp_dim = int(hidden_dim * float(mlp_ratio))
        if mlp_dim < 1:
            raise ValueError("controller MLP width must be positive")
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(tokens)
        tokens = tokens + self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )[0]
        return tokens + self.mlp(self.norm2(tokens))


class StateConditionedLatentController(nn.Module):
    """Predict a latent action from current visual state and a real action.

    The visual tokens pass through two self-attention blocks.  A three-layer
    action MLP supplies one cross-attention query, and the attended visual
    value is projected directly to the latent-action coordinate system.  There
    is deliberately no action-only output or residual bypass around the
    cross-attention operation.
    """

    def __init__(
        self,
        real_action_dim: int,
        hidden_dim: int,
        latent_dim: int,
        *,
        num_heads: int,
        num_state_blocks: int = 2,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        if min(real_action_dim, hidden_dim, latent_dim, num_heads) < 1:
            raise ValueError("controller dimensions and num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("controller hidden_dim must be divisible by num_heads")
        if int(num_state_blocks) != 2:
            raise ValueError("state-conditioned controller requires two state blocks")
        self.real_action_dim = int(real_action_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.state_blocks = nn.ModuleList(
            [
                _StateSelfAttentionBlock(
                    self.hidden_dim,
                    int(num_heads),
                    float(mlp_ratio),
                )
                for _ in range(int(num_state_blocks))
            ]
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(self.real_action_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.state_norm = nn.LayerNorm(self.hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_dim,
            int(num_heads),
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.latent_head = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(
        self,
        state_tokens: torch.Tensor,
        normalized_real_action: torch.Tensor,
    ) -> torch.Tensor:
        if state_tokens.ndim != 3 or state_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                "Expected state tokens [N, P, "
                f"{self.hidden_dim}], got {tuple(state_tokens.shape)}"
            )
        if (
            normalized_real_action.ndim != 2
            or normalized_real_action.shape
            != (state_tokens.shape[0], self.real_action_dim)
        ):
            raise ValueError(
                "Expected normalized real action "
                f"[{state_tokens.shape[0]}, {self.real_action_dim}], got "
                f"{tuple(normalized_real_action.shape)}"
            )
        for block in self.state_blocks:
            state_tokens = block(state_tokens)
        query = self.query_norm(self.action_mlp(normalized_real_action)).unsqueeze(1)
        state_tokens = self.state_norm(state_tokens)
        attended = self.cross_attention(
            query=query,
            key=state_tokens,
            value=state_tokens,
            need_weights=False,
        )[0].squeeze(1)
        return self.latent_head(self.output_norm(attended))


class MotionInputNormalizer(nn.Module):
    """Parameter-free normalization applied immediately before an adapter."""

    def __init__(self, input_dim: int, config: Any):
        super().__init__()
        mode = str(_config_get(config, "mode", "identity")).lower()
        self.mode = mode
        self.input_dim = int(input_dim)

        if mode == "layer_norm":
            self.layer_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        elif mode in {"mean_std", "standard"}:
            mean = self._vector(config, "mean")
            std = self._vector(config, "std")
            if torch.any(std <= 0):
                raise ValueError("Motion normalization std values must be positive")
            self.register_buffer("offset", mean)
            self.register_buffer("scale", std)
        elif mode == "minmax":
            minimum = self._vector(config, "min")
            maximum = self._vector(config, "max")
            if torch.any(maximum <= minimum):
                raise ValueError("Motion normalization max must be greater than min")
            self.register_buffer("offset", (minimum + maximum) / 2)
            self.register_buffer("scale", (maximum - minimum) / 2)
        elif mode == "identity":
            pass
        else:
            raise ValueError(f"Unsupported motion normalization mode: {mode!r}")

    def _vector(self, config: Any, name: str) -> torch.Tensor:
        value = _config_get(config, name)
        if value is None:
            raise ValueError(f"Motion normalization {self.mode}.{name} is required")
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.shape != (self.input_dim,):
            raise ValueError(
                f"Motion normalization {name} must have shape [{self.input_dim}], "
                f"got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Motion normalization {name} contains non-finite values")
        return tensor

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        if self.mode == "layer_norm":
            return self.layer_norm(motion)
        if self.mode in {"mean_std", "standard", "minmax"}:
            return (motion - self.offset.to(motion)) / self.scale.to(motion)
        return motion


class MotionConditionEncoder(nn.Module):
    """Route heterogeneous motion values through independent adapters."""

    def __init__(self, condition_dim: int, config: Any):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.available_types = validate_motion_types(config)
        adapter_base_dim = int(
            _config_get(config, "adapter_hidden_dim", self.condition_dim)
        )
        if adapter_base_dim < 1:
            raise ValueError("motion_condition.adapter_hidden_dim must be positive")
        balance_params = bool(_config_get(config, "balance_parameter_count", True))
        reference_dim = int(_config_get(config, "parameter_count_reference_dim", 3))
        target_params = (
            adapter_base_dim * (reference_dim + self.condition_dim + 1)
            + 3 * self.condition_dim
        )

        self.input_dims: dict[str, int] = {}
        self.adapter_dims: dict[str, int] = {}
        for motion_type in MOTION_ADAPTER_TYPES:
            if motion_type not in self.available_types:
                continue
            input_dim = motion_input_dim(config, motion_type)
            adapter_dim = adapter_base_dim
            if balance_params:
                adapter_dim = max(
                    1,
                    round(
                        (target_params - 3 * self.condition_dim)
                        / (input_dim + self.condition_dim + 1)
                    ),
                )
            type_config = _config_get(config, motion_type)
            normalization = _config_get(type_config, "normalization", {})
            if (
                motion_type == "latent"
                and str(_config_get(normalization, "mode", "")).lower() != "layer_norm"
            ):
                raise ValueError(
                    "motion_condition.latent.normalization.mode must be layer_norm"
                )

            setattr(
                self,
                f"{motion_type}_motion_normalizer",
                MotionInputNormalizer(input_dim, normalization),
            )
            setattr(
                self,
                f"{motion_type}_action_adapter",
                MotionAdapter(input_dim, adapter_dim, self.condition_dim),
            )
            self.input_dims[motion_type] = input_dim
            self.adapter_dims[motion_type] = adapter_dim

    def adapter_parameter_counts(self) -> dict[str, int]:
        return {
            motion_type: sum(
                parameter.numel()
                for parameter in getattr(
                    self, f"{motion_type}_action_adapter"
                ).parameters()
            )
            for motion_type in self.input_dims
        }

    def encode(self, motion_type: str, values: torch.Tensor) -> torch.Tensor:
        """Encode dense values for one motion type without adding a condition.

        Keeping this operation separate lets the two-stage trainer apply a
        validity mask *after* the adapter (including its bias), and lets the
        latent adapter act as the frozen teacher in alignment finetuning.
        """
        if motion_type not in self.input_dims:
            raise ValueError(
                f"Motion type {motion_type!r} is not enabled; available adapter "
                f"types are {sorted(self.input_dims)}"
            )
        if values.ndim != 2 or values.shape[-1] != self.input_dims[motion_type]:
            raise ValueError(
                f"Expected {motion_type} values [N, {self.input_dims[motion_type]}], "
                f"got {tuple(values.shape)}"
            )
        adapter = getattr(self, f"{motion_type}_action_adapter")
        adapter_parameter = next(adapter.parameters())
        values = values.to(
            device=adapter_parameter.device, dtype=adapter_parameter.dtype
        )
        return adapter(self.normalize(motion_type, values))

    def normalize(self, motion_type: str, values: torch.Tensor) -> torch.Tensor:
        """Apply the configured input normalization without action encoding."""
        if motion_type not in self.input_dims:
            raise ValueError(f"Motion type {motion_type!r} is not enabled")
        if values.ndim != 2 or values.shape[-1] != self.input_dims[motion_type]:
            raise ValueError(
                f"Expected {motion_type} values [N, {self.input_dims[motion_type]}], "
                f"got {tuple(values.shape)}"
            )
        normalizer = getattr(self, f"{motion_type}_motion_normalizer")
        parameter = next(
            getattr(self, f"{motion_type}_action_adapter").parameters()
        )
        return normalizer(values.to(device=parameter.device, dtype=parameter.dtype))

    def forward(
        self,
        base_condition: torch.Tensor,
        motion: Mapping[str, Mapping[str, torch.Tensor]] | None,
    ) -> torch.Tensor:
        if motion is None or len(motion) == 0:
            return base_condition
        if not isinstance(motion, Mapping):
            raise TypeError("motion must be a mapping grouped by motion type")

        condition = base_condition
        for motion_type, payload in motion.items():
            if motion_type == "none":
                raise ValueError("none samples must be omitted from the motion mapping")
            if motion_type not in self.input_dims:
                raise ValueError(
                    f"Motion type {motion_type!r} is not enabled; available adapter "
                    f"types are {sorted(self.input_dims)}"
                )
            if not isinstance(payload, Mapping):
                raise TypeError(f"motion[{motion_type!r}] must be a mapping")
            indices = payload.get("indices")
            values = payload.get("values")
            if not isinstance(indices, torch.Tensor) or not isinstance(
                values, torch.Tensor
            ):
                raise TypeError(
                    f"motion[{motion_type!r}] requires tensor indices and values"
                )
            if indices.ndim != 1 or indices.dtype != torch.int64:
                raise ValueError(f"motion[{motion_type!r}].indices must be int64 [N]")
            if values.ndim != 2 or values.shape[0] != indices.numel():
                raise ValueError(
                    f"motion[{motion_type!r}].values must be [N, D] aligned with indices"
                )
            if values.shape[1] != self.input_dims[motion_type]:
                raise ValueError(
                    f"motion[{motion_type!r}] expected dim {self.input_dims[motion_type]}, "
                    f"got {values.shape[1]}"
                )
            if indices.numel() == 0:
                continue

            adapter = getattr(self, f"{motion_type}_action_adapter")
            values = values.to(device=condition.device)
            indices = indices.to(device=condition.device)
            embeddings = self.encode(motion_type, values).to(condition.dtype)
            # index_add leaves every omitted (none) row exactly as timestep + k.
            condition = condition.index_add(0, indices, embeddings)
        return condition


class OfflineMotionStore:
    """Lazy per-trajectory reader for offline geometry or LAM actions.

    Canonical files are either ``.pt`` dictionaries or ``.npz`` archives with
    schema/type metadata, ``frame_pairs`` shaped ``[N,2]`` and ``motion`` shaped
    ``[N,D]``. A pair is ``(current_frame, target_frame)``. Strict geometry
    metadata also fixes coordinate frame, translation/yaw units and component
    order. Configurable keys/file patterns keep extractors decoupled.
    """

    def __init__(
        self,
        *,
        root: str,
        dataset_name: str,
        motion_type: str,
        input_dim: int,
        file_pattern: str = "{dataset_name}/{trajectory_name}.pt",
        pairs_key: str = "frame_pairs",
        values_key: str = "motion",
        cache_size: int = 8,
        strict_metadata: bool = True,
        translation_unit: str | None = None,
    ):
        if motion_type not in {"geometry", "latent"}:
            raise ValueError("OfflineMotionStore only supports geometry and latent")
        if not root:
            raise ValueError(f"A cache root is required for {motion_type} motion")
        self.root = os.path.realpath(os.path.expanduser(str(root)))
        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                f"Offline {motion_type} motion root does not exist: {self.root}"
            )
        self.dataset_name = str(dataset_name)
        self.motion_type = motion_type
        self.input_dim = int(input_dim)
        self.file_pattern = str(file_pattern)
        self.pairs_key = str(pairs_key)
        self.values_key = str(values_key)
        self.cache_size = int(cache_size)
        self.strict_metadata = bool(strict_metadata)
        self.translation_unit = translation_unit
        if (
            self.motion_type == "geometry"
            and self.strict_metadata
            and self.translation_unit
            not in {
                "waypoint_spacing_units",
                "meters",
            }
        ):
            raise ValueError(
                "Geometry translation_unit must be 'waypoint_spacing_units' or 'meters'"
            )
        if self.cache_size < 1:
            raise ValueError("Offline motion cache_size must be at least 1")
        self._cache: OrderedDict[
            str, tuple[torch.Tensor, dict[tuple[int, int], int]]
        ] = OrderedDict()

    def _path(self, trajectory_name: str) -> str:
        relative = self.file_pattern.format(
            dataset_name=self.dataset_name,
            trajectory_name=trajectory_name,
            motion_type=self.motion_type,
        )
        path = os.path.realpath(os.path.join(self.root, relative))
        if path != self.root and not path.startswith(self.root + os.sep):
            raise ValueError(f"Unsafe offline motion path for {trajectory_name!r}")
        return path

    def _load_payload(self, path: str) -> Mapping[str, Any]:
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as archive:
                return {key: archive[key] for key in archive.files}
        return torch.load(path, map_location="cpu", weights_only=True)

    @staticmethod
    def _scalar(value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            return value.item()
        if isinstance(value, np.ndarray) and value.ndim == 0:
            return value.item()
        return value

    def _load_trajectory(
        self, trajectory_name: str
    ) -> tuple[torch.Tensor, dict[tuple[int, int], int]]:
        cached = self._cache.pop(trajectory_name, None)
        if cached is not None:
            self._cache[trajectory_name] = cached
            return cached

        path = self._path(trajectory_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {self.motion_type} motion cache for "
                f"{self.dataset_name}/{trajectory_name}: {path}"
            )
        payload = self._load_payload(path)
        if not isinstance(payload, Mapping):
            raise TypeError(f"Offline motion cache must contain a mapping: {path}")
        if self.strict_metadata:
            schema_version = self._scalar(payload.get("schema_version"))
            payload_type = self._scalar(payload.get("motion_type"))
            if schema_version != OFFLINE_MOTION_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported offline motion schema in {path}: {schema_version!r}"
                )
            if payload_type != self.motion_type:
                raise ValueError(
                    f"Offline motion type mismatch in {path}: {payload_type!r}"
                )
            for key, expected in (
                ("dataset_name", self.dataset_name),
                ("trajectory_name", trajectory_name),
            ):
                if key in payload and self._scalar(payload[key]) != expected:
                    raise ValueError(f"Offline motion {key} mismatch in {path}")

            expected_metadata: dict[str, Any] = {
                "pair_direction": "current_to_goal",
                "normalization": "raw",
            }
            if self.motion_type == "geometry":
                expected_metadata.update(
                    {
                        "coordinate_frame": "current_navigation_frame",
                        "translation_unit": self.translation_unit,
                        "yaw_unit": "radians",
                        "components": ["delta_x", "delta_y", "delta_yaw"],
                    }
                )
            for key, expected in expected_metadata.items():
                if key not in payload:
                    raise ValueError(
                        f"Offline {self.motion_type} metadata {key!r} is missing "
                        f"in {path}"
                    )
                actual = payload[key]
                if isinstance(actual, (torch.Tensor, np.ndarray)):
                    actual = actual.item() if actual.ndim == 0 else actual.tolist()
                if actual != expected:
                    raise ValueError(
                        f"Offline {self.motion_type} metadata {key!r} mismatch "
                        f"in {path}: {actual!r} != {expected!r}"
                    )

        if self.pairs_key not in payload or self.values_key not in payload:
            raise KeyError(
                f"Offline motion cache {path} requires {self.pairs_key!r} and "
                f"{self.values_key!r}"
            )
        pairs = torch.as_tensor(payload[self.pairs_key], dtype=torch.int64)
        values = torch.as_tensor(payload[self.values_key], dtype=torch.float32)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"frame pairs must have shape [N,2] in {path}")
        if values.ndim != 2 or values.shape != (pairs.shape[0], self.input_dim):
            raise ValueError(
                f"motion must have shape [{pairs.shape[0]},{self.input_dim}] in {path}; "
                f"got {tuple(values.shape)}"
            )
        if not torch.isfinite(values).all():
            raise ValueError(f"Offline motion contains non-finite values in {path}")

        lookup: dict[tuple[int, int], int] = {}
        for row, pair in enumerate(pairs.tolist()):
            key = (int(pair[0]), int(pair[1]))
            if key in lookup:
                raise ValueError(f"Duplicate frame pair {key} in {path}")
            lookup[key] = row

        cached = (values.contiguous(), lookup)
        self._cache[trajectory_name] = cached
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return cached

    def get(
        self, trajectory_name: str, current_frame: int, target_frames: Sequence[int]
    ) -> torch.Tensor:
        values, lookup = self._load_trajectory(str(trajectory_name))
        rows = []
        missing = []
        for target_frame in target_frames:
            pair = (int(current_frame), int(target_frame))
            row = lookup.get(pair)
            if row is None:
                missing.append(pair)
            else:
                rows.append(row)
        if missing:
            raise KeyError(
                f"{self.motion_type} cache for {self.dataset_name}/{trajectory_name} "
                f"does not contain frame pair(s): {missing}"
            )
        return values.index_select(0, torch.as_tensor(rows, dtype=torch.int64))


def motion_condition_collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate heterogeneous samples without padding or inventing null actions."""
    if not samples:
        raise ValueError("Cannot collate an empty motion-condition batch")
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise TypeError("Motion-condition datasets must return dictionaries")

    reserved_keys = {"motion_type", "motion", "motion_mask"}
    data_keys = set(samples[0]).difference(reserved_keys)
    for sample in samples[1:]:
        if set(sample).difference(reserved_keys) != data_keys:
            raise ValueError(
                "All samples in a batch must use the same image input mode"
            )
    batch = {
        key: default_collate([sample[key] for sample in samples]) for key in data_keys
    }

    motion_types: list[str] = []
    grouped: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"sample_indices": [], "values": [], "masks": []}
    )
    for sample_index, sample in enumerate(samples):
        motion_type = str(sample.get("motion_type", ""))
        if motion_type not in MOTION_TYPES:
            raise ValueError(f"Invalid sample motion_type: {motion_type!r}")
        motion_types.append(motion_type)
        if motion_type == "none":
            if "motion" in sample:
                raise ValueError("none samples must not contain a motion field")
            continue
        if "motion" not in sample:
            raise ValueError(f"{motion_type} sample is missing its motion field")
        values = torch.as_tensor(sample["motion"], dtype=torch.float32)
        if values.ndim != 2:
            raise ValueError(
                "Each sample motion must have shape [num_goals, input_dim]"
            )
        if "k" in sample and values.shape[0] != torch.as_tensor(sample["k"]).numel():
            raise ValueError(
                "Sample motion and temporal k must have the same num_goals"
            )
        grouped[motion_type]["sample_indices"].append(sample_index)
        grouped[motion_type]["values"].append(values)
        mask = torch.as_tensor(
            sample.get("motion_mask", torch.ones(values.shape[0], dtype=torch.bool)),
            dtype=torch.bool,
        )
        if mask.shape != (values.shape[0],):
            raise ValueError("Each sample motion_mask must have shape [num_goals]")
        grouped[motion_type]["masks"].append(mask)

    batch["motion_type"] = motion_types
    if grouped:
        batch["motion"] = {
            motion_type: {
                "sample_indices": torch.as_tensor(
                    payload["sample_indices"], dtype=torch.int64
                ),
                "values": default_collate(payload["values"]),
                "masks": default_collate(payload["masks"]),
            }
            for motion_type, payload in grouped.items()
        }
    return batch


def flatten_motion_groups(
    grouped: Mapping[str, Mapping[str, torch.Tensor]] | None,
    *,
    batch_size: int,
    num_goals: int,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]] | None:
    """Convert collated ``[sample, goal]`` groups to model ``[B*G]`` groups."""
    if grouped is None or len(grouped) == 0:
        return None
    result: dict[str, dict[str, torch.Tensor]] = {}
    seen_indices = []
    goal_indices = torch.arange(num_goals, dtype=torch.int64)
    for motion_type, payload in grouped.items():
        sample_indices = payload["sample_indices"]
        values = payload["values"]
        masks = payload.get("masks")
        if sample_indices.ndim != 1 or sample_indices.dtype != torch.int64:
            raise ValueError("Collated motion sample_indices must be int64 [N]")
        if values.ndim != 3 or values.shape[:2] != (
            sample_indices.numel(),
            num_goals,
        ):
            raise ValueError(
                f"Collated {motion_type} motion must have shape [N,{num_goals},D]"
            )
        flat_indices = (
            sample_indices[:, None] * num_goals + goal_indices[None, :]
        ).reshape(-1)
        flat_values = values.flatten(0, 1)
        if masks is not None:
            if masks.dtype != torch.bool or masks.shape != values.shape[:2]:
                raise ValueError(
                    f"Collated {motion_type} motion masks must be bool [N,{num_goals}]"
                )
            flat_mask = masks.reshape(-1)
            flat_indices = flat_indices[flat_mask]
            flat_values = flat_values[flat_mask]
        seen_indices.append(flat_indices)
        result[motion_type] = {
            "indices": flat_indices.to(device=device, non_blocking=True),
            "values": flat_values.to(device=device, non_blocking=True),
        }

    all_indices = torch.cat(seen_indices)
    if all_indices.numel() != torch.unique(all_indices).numel():
        raise ValueError("A flattened sample is assigned to more than one motion type")
    if torch.any(all_indices < 0) or torch.any(all_indices >= batch_size * num_goals):
        raise ValueError("Flattened motion index is outside the model batch")
    return result


def make_motion_group(
    motion_type: str, values: torch.Tensor
) -> dict[str, dict[str, torch.Tensor]] | None:
    """Build a uniform model-facing group, primarily for navigation inference."""
    if motion_type == "none":
        return None
    if motion_type not in MOTION_ADAPTER_TYPES:
        raise ValueError(f"Invalid motion type: {motion_type!r}")
    if values.ndim != 2:
        raise ValueError("Uniform motion values must have shape [N,D]")
    return {
        motion_type: {
            "indices": torch.arange(values.shape[0], device=values.device),
            "values": values,
        }
    }
