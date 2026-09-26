"""Strict checkpoint helpers for the opt-in two-stage NWM training path.

This module deliberately does not replace :mod:`train_utils` checkpoint loading.
Legacy configurations and checkpoints continue through that unchanged path.  The
helpers here distinguish two operations that must not be conflated:

* initializing stage 2 from a stage-1 checkpoint, where a small, explicitly
  named set of newly initialized adapter parameters may be absent; and
* resuming a two-stage substage, where model loading is exact and optimizer
  state is restored only for the same optimizer phase.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn

logger = logging.getLogger(__name__)

TWO_STAGE_METADATA_KEY = "two_stage_metadata"
TWO_STAGE_CHECKPOINT_SCHEMA_VERSION = 1
DATA_RESUME_FINGERPRINT_SCHEMA_VERSION = 1

_REAL_ADAPTER_PREFIXES = (
    "motion_condition_encoder.real_action_adapter.",
    "motion_condition_encoder.real_motion_normalizer.",
    "real_action_encoder.",
    "real_action_embedder.",
    "E_real.",
)
_REAL_TO_LATENT_PREFIXES = (
    "real_to_latent.",
    "real_to_latent_adapter.",
    "real_to_latent_encoder.",
    "G_real_to_latent.",
)
_STATE_CONTROLLER_PREFIXES = (
    "state_conditioned_controller.",
)


def capture_rng_state() -> dict[str, Any]:
    """Capture the process-local RNG state needed for an exact continuation.

    Data-loader worker state is reconstructed by replaying the saved batch
    cursor from a deterministic per-epoch worker seed.  The state captured here
    covers model-side Python, NumPy, CPU torch, and the current CUDA device.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }
    # Do not initialize CUDA merely because a CPU unit test saves a checkpoint.
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state().cpu().clone()
        state["cuda_device"] = int(torch.cuda.current_device())
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""
    if state is None:
        return
    if not isinstance(state, Mapping):
        raise TypeError("rng_state must be a mapping")
    required = {"python", "numpy", "torch"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"rng_state is missing required fields: {missing}")
    random.setstate(state["python"])
    np.random.set_state(tuple(state["numpy"]))
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint has CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state(
            torch.as_tensor(state["cuda"], dtype=torch.uint8).cpu(),
            device=torch.cuda.current_device(),
        )


def _motion_adapter_prefixes(motion_type: str) -> tuple[str, ...]:
    return (
        f"motion_condition_encoder.{motion_type}_action_adapter.",
        f"motion_condition_encoder.{motion_type}_motion_normalizer.",
        f"{motion_type}_action_encoder.",
        f"E_{motion_type}.",
    )


_INACTIVE_PROXY_PREFIXES = (
    *_motion_adapter_prefixes("geometry"),
    *_motion_adapter_prefixes("idm"),
)
_ALL_PROXY_PREFIXES = (
    *_INACTIVE_PROXY_PREFIXES,
    *_motion_adapter_prefixes("latent"),
)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _path(value: Any, *keys: str, default: Any = None) -> Any:
    current = value
    for key in keys:
        current = _get(current, key, None)
        if current is None:
            return default
    return current


def _two_stage_value(config: Any, key: str, default: Any = None) -> Any:
    """Read both supported config layouts without imposing one on Hydra."""
    direct = _get(config, key, None)
    if direct is not None:
        return direct
    return _get(_get(config, "two_stage", None), key, default)


def _finetune_config(config: Any) -> Any:
    return _two_stage_value(config, "finetune", {})


def _proxy_config(config: Any) -> Any:
    return _two_stage_value(config, "proxy", {})


def _serializable(value: Any) -> Any:
    """Convert config/tensor leaves to deterministic checkpoint-safe values."""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _serializable(value.tolist())
        except (TypeError, ValueError):
            pass
    return value


def _data_resume_substage(value: Any) -> str:
    substage = _normalize_substage(value)
    if substage is None:
        raise ValueError("A data-resume fingerprint requires a training substage")
    # A transition artifact is the epoch-zero entry point to joint training;
    # it must therefore carry the same data contract as a joint checkpoint.
    return "joint" if substage == "transition" else substage


def _fingerprint_digest(payload: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            _serializable(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Data-resume fingerprint contains a non-serializable value"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def build_data_resume_fingerprint(
    config: Any,
    substage: str,
    *,
    dataset_length: int | None = None,
    loader_length: int | None = None,
) -> dict[str, Any]:
    """Fingerprint every config value that controls cursor replay.

    Step targets and optimizer settings are intentionally excluded, which lets
    a phase-final checkpoint continue when its target is extended. Dataset,
    sampler, worker, and batch settings are included because changing any of
    them makes an in-epoch cursor refer to different samples.
    """
    normalized_substage = _data_resume_substage(substage)
    training = _get(config, "training", {})
    finetune = _finetune_config(config)
    payload = _serializable(
        {
            "seed": int(_get(config, "seed", 0)),
            "training_stage": str(
                _two_stage_value(config, "training_stage", "legacy")
            ).strip().lower(),
            "action_mode": str(
                _two_stage_value(config, "action_mode", "none")
            ).strip().lower(),
            "resume_substage": normalized_substage,
            "finetune_scheme": (
                _normalize_scheme(_get(finetune, "scheme", "reset"))
                if normalized_substage in {"warmup", "joint"}
                else None
            ),
            "batch_size": int(_get(training, "batch_size", 1)),
            "num_workers": int(_get(training, "num_workers", 0)),
            "loader_contract": {
                "sampler": "DistributedSampler",
                "shuffle": True,
                "drop_last": True,
                "persistent_workers": False,
                # The worker start method is deliberately not part of the
                # sample/cursor contract.  With an explicit per-epoch loader
                # generator and non-persistent workers, fork and spawn assign
                # the same sampler indices and worker RNG seeds.  Keeping this
                # operational detail out lets existing exact-resume
                # checkpoints continue under the CUDA-safe spawn context.
            },
            "dataset": _get(config, "dataset", None),
            "dataset_selection": _get(config, "dataset_selection", None),
            "proxy": _proxy_config(config),
            "motion_condition": _get(config, "motion_condition", None),
        }
    )
    assert isinstance(payload, Mapping)
    reference_batch_size = _get(training, "reference_batch_size", None)
    if reference_batch_size not in (None, "", "null"):
        # Keep fingerprints for all existing/default loaders byte-for-byte
        # compatible; the extra cursor contract exists only for opt-in runs.
        payload = dict(payload)
        payload["reference_batch_size"] = int(reference_batch_size)
    if normalized_substage == "pretrain":
        sample_target = _get(training, "target_samples_per_rank", None)
    elif normalized_substage == "warmup":
        sample_target = _get(finetune, "warmup_samples_per_rank", None)
    else:
        sample_target = _get(finetune, "joint_samples_per_rank", None)
    if sample_target not in (None, "", "null"):
        # A final cropped optimizer batch consumes a complete DataLoader cursor.
        # Pinning the budget prevents an extended resume from silently skipping
        # the unused suffix of that final fetched batch.
        payload = dict(payload)
        payload["target_samples_per_rank"] = int(sample_target)
    fingerprint: dict[str, Any] = {
        "schema_version": DATA_RESUME_FINGERPRINT_SCHEMA_VERSION,
        "sha256": _fingerprint_digest(payload),
        "payload": dict(payload),
    }
    if dataset_length is not None:
        if int(dataset_length) < 0:
            raise ValueError("dataset_length must be non-negative")
        fingerprint["dataset_length"] = int(dataset_length)
    if loader_length is not None:
        if int(loader_length) < 0:
            raise ValueError("loader_length must be non-negative")
        fingerprint["loader_length"] = int(loader_length)
    return fingerprint


def validate_data_resume_fingerprint(
    fingerprint: Mapping[str, Any] | None,
    config: Any | None,
    substage: str,
    *,
    dataset_length: int | None = None,
    loader_length: int | None = None,
    required: bool = False,
) -> None:
    """Strictly validate the saved data/cursor replay contract.

    Schema-v1 checkpoints created before fingerprints existed remain accepted
    at an epoch boundary (``batch_in_epoch == 0``). An in-epoch checkpoint can
    never be replayed exactly without this contract and is therefore rejected.
    """
    if fingerprint is None:
        if required:
            raise ValueError(
                "In-epoch checkpoint has no data resume fingerprint; exact "
                "cursor replay is not possible"
            )
        return
    if not isinstance(fingerprint, Mapping):
        raise TypeError("Checkpoint data resume fingerprint must be a mapping")
    try:
        schema_version = int(fingerprint.get("schema_version", 0))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Checkpoint data resume fingerprint schema_version must be an integer"
        ) from error
    if schema_version != DATA_RESUME_FINGERPRINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported data resume fingerprint schema_version: "
            f"{schema_version}"
        )
    payload = fingerprint.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint data resume fingerprint is missing its payload")
    saved_digest = str(fingerprint.get("sha256", ""))
    if not saved_digest or saved_digest != _fingerprint_digest(payload):
        raise ValueError("Checkpoint data resume fingerprint is corrupt")

    if config is None:
        if required:
            raise ValueError(
                "In-epoch resume requires the current config to validate its "
                "data resume fingerprint"
            )
    else:
        expected = build_data_resume_fingerprint(config, substage)
        if saved_digest != expected["sha256"]:
            expected_payload = expected["payload"]
            changed = sorted(
                key
                for key in set(payload) | set(expected_payload)
                if payload.get(key) != expected_payload.get(key)
            )
            raise ValueError(
                "Checkpoint data resume fingerprint mismatch; cursor replay "
                f"would use different data settings (changed: {changed})"
            )

    for name, current_value in (
        ("dataset_length", dataset_length),
        ("loader_length", loader_length),
    ):
        if current_value is None:
            continue
        saved_value = fingerprint.get(name)
        if saved_value is None:
            if required:
                raise ValueError(
                    f"In-epoch checkpoint data resume fingerprint has no {name}"
                )
            continue
        if int(saved_value) != int(current_value):
            raise ValueError(
                f"Checkpoint data resume {name} mismatch: "
                f"checkpoint={int(saved_value)}, current={int(current_value)}"
            )


def _normalize_scheme(value: Any) -> str:
    scheme = str(value or "reset").strip().lower().replace("-", "_")
    aliases = {
        "a": "reset",
        "alignment": "embedding_align",
        "align": "embedding_align",
        "embedding_alignment": "embedding_align",
        "b": "embedding_align",
        "real2latent": "real_to_latent",
        "real_action_to_latent": "real_to_latent",
        "c": "real_to_latent",
        "d": "state_conditioned_controller",
        "state_controller": "state_conditioned_controller",
        "latent_state_controller": "state_conditioned_controller",
    }
    scheme = aliases.get(scheme, scheme)
    if scheme not in {
        "reset",
        "embedding_align",
        "real_to_latent",
        "state_conditioned_controller",
    }:
        raise ValueError(f"Unsupported finetune scheme: {value!r}")
    return scheme


def _normalize_substage(value: Any) -> str | None:
    if value is None:
        return None
    substage = str(value).strip().lower().replace("-", "_")
    aliases = {
        "step1": "warmup",
        "step_1": "warmup",
        "adapter_warmup": "warmup",
        "step2": "joint",
        "step_2": "joint",
        "joint_finetune": "joint",
        "joint_fine_tuning": "joint",
        "proxy_pretrain": "pretrain",
    }
    substage = aliases.get(substage, substage)
    if substage not in {"pretrain", "warmup", "transition", "joint"}:
        raise ValueError(f"Unsupported finetune substage: {value!r}")
    return substage


def _unwrap_model(model: nn.Module | None) -> nn.Module | None:
    if model is None:
        return None
    while True:
        if hasattr(model, "module") and isinstance(model.module, nn.Module):
            model = model.module
            continue
        if hasattr(model, "_orig_mod") and isinstance(model._orig_mod, nn.Module):
            model = model._orig_mod
            continue
        return model


def _model_signature(config: Any, model: nn.Module | None) -> dict[str, Any]:
    model = _unwrap_model(model)
    generator = _path(config, "model", "generator", default={})
    model_target = _get(generator, "_target_", None)
    model_size = (
        _get(generator, "model_size", None)
        or _get(generator, "name", None)
        or model_target
    )

    hidden_dim = _get(generator, "hidden_size", None)
    if model is not None:
        hidden_dim = getattr(model, "hidden_size", hidden_dim)
        if hidden_dim is None and hasattr(model, "pos_embed"):
            pos_embed = getattr(model, "pos_embed")
            if isinstance(pos_embed, torch.Tensor) and pos_embed.ndim > 0:
                hidden_dim = int(pos_embed.shape[-1])

    context_size = _path(config, "dataset", "context_size", default=None)
    if model is not None:
        context_size = getattr(model, "context_size", context_size)

    return {
        "model_target": None if model_target is None else str(model_target),
        "model_size": None if model_size is None else str(model_size),
        "hidden_dim": None if hidden_dim is None else int(hidden_dim),
        "context_size": None if context_size is None else int(context_size),
    }


def _latent_dim(config: Any, model: nn.Module | None = None) -> int | None:
    model = _unwrap_model(model)
    if model is not None and hasattr(model, "motion_condition_encoder"):
        input_dims = getattr(model.motion_condition_encoder, "input_dims", None)
        if isinstance(input_dims, Mapping) and input_dims.get("latent") is not None:
            return int(input_dims["latent"])

    configured = _path(
        config, "motion_condition", "latent", "latent_dim", default=None
    )
    if configured is None:
        configured = _get(_proxy_config(config), "dim", None)
    return None if configured is None else int(configured)


def _proxy_dim(config: Any, action_mode: str, model: nn.Module | None) -> int | None:
    configured = _get(_proxy_config(config), "dim", None)
    if configured is not None:
        return int(configured)
    if action_mode == "none":
        return None
    if action_mode == "latent":
        return _latent_dim(config, model)
    configured = _path(
        config,
        "motion_condition",
        action_mode,
        f"{action_mode}_dim",
        default=None,
    )
    if configured is not None:
        return int(configured)
    model = _unwrap_model(model)
    if model is not None and hasattr(model, "motion_condition_encoder"):
        dims = getattr(model.motion_condition_encoder, "input_dims", {})
        if isinstance(dims, Mapping) and action_mode in dims:
            return int(dims[action_mode])
    return None


def _latent_normalization(config: Any) -> dict[str, Any]:
    normalization = _path(
        config, "motion_condition", "latent", "normalization", default={}
    )
    proxy = _proxy_config(config)
    mode = _get(normalization, "mode", None)
    identifier = (
        _get(normalization, "identifier", None)
        or _get(proxy, "normalization_identifier", None)
        or mode
    )
    normalization_path = (
        _get(normalization, "path", None)
        or _get(proxy, "normalization_path", None)
    )
    statistics = _get(normalization, "statistics", None)
    if statistics is None:
        statistics = {
            key: _get(normalization, key, None)
            for key in ("mean", "std", "min", "max")
            if _get(normalization, key, None) is not None
        }
    return {
        "identifier": None if identifier is None else str(identifier),
        "path": None if normalization_path is None else str(normalization_path),
        "statistics": _serializable(statistics or {}),
        "mode": None if mode is None else str(mode),
    }


def _state_controller_contract(
    config: Any,
    model: nn.Module | None = None,
) -> dict[str, Any]:
    """Serialize architecture and optimization semantics that affect Scheme D."""
    model = _unwrap_model(model)
    finetune = _finetune_config(config)
    controller = (
        getattr(model, "state_conditioned_controller", None)
        if model is not None
        else None
    )
    state_blocks = int(_get(finetune, "controller_state_blocks", 2))
    num_heads = _get(finetune, "controller_num_heads", None)
    mlp_ratio = float(_get(finetune, "controller_mlp_ratio", 4.0))
    if controller is not None:
        state_blocks = len(controller.state_blocks)
        num_heads = int(controller.cross_attention.num_heads)
        first_mlp = controller.state_blocks[0].mlp[0]
        mlp_ratio = float(first_mlp.out_features) / float(controller.hidden_dim)
    return _serializable(
        {
            "state_blocks": state_blocks,
            "num_heads": None if num_heads is None else int(num_heads),
            "mlp_ratio": mlp_ratio,
            "latent_l2_weight": float(_get(finetune, "latent_l2_weight", 1.0)),
            "controller_pretrain_lr": float(
                _get(finetune, "controller_pretrain_lr", 1.0e-3)
            ),
            "controller_pretrain_weight_decay": float(
                _get(finetune, "controller_pretrain_weight_decay", 0.04)
            ),
            "controller_lr_warmup_steps": int(
                _get(finetune, "controller_lr_warmup_steps", 300)
            ),
            "joint_lr": float(_get(finetune, "joint_lr", 1.0e-4)),
        }
    )


def build_checkpoint_metadata(
    config: Any,
    finetune_substage: str | None,
    completed_warmup_steps: int,
    completed_joint_steps: int,
    model: nn.Module | None = None,
) -> dict[str, Any]:
    """Build the complete, serializable contract stored by two-stage runs."""
    training_stage = str(
        _two_stage_value(config, "training_stage", "legacy")
    ).lower()
    action_mode = str(_two_stage_value(config, "action_mode", "none")).lower()
    proxy = _proxy_config(config)
    proxy_type = str(_get(proxy, "type", action_mode)).lower()
    if action_mode == "none":
        proxy_type = "none"
    # A proxy-pretraining checkpoint has no fine-tuning scheme yet. Recording
    # ``reset`` here made stage-1 provenance look like a stage-2 reset run, so
    # keep the required field but store its accurate null value until stage 2.
    if training_stage == "proxy_pretrain":
        scheme = None
        finetune_initialization = None
    else:
        finetune = _finetune_config(config)
        scheme = _normalize_scheme(
            _get(finetune, "scheme", "reset")
        )
        finetune_initialization = (
            "random"
            if bool(_get(finetune, "random_init", False))
            else "init_checkpoint" if (_get(finetune, "init_checkpoint", None)
                                        or _get(finetune, "dataset_protocol", "paper") == "go2")
            else "stage1_checkpoint"
        )
    substage = _normalize_substage(finetune_substage)
    warmup_steps = int(completed_warmup_steps)
    joint_steps = int(completed_joint_steps)
    if warmup_steps < 0 or joint_steps < 0:
        raise ValueError("Completed warmup/joint step counts must be non-negative")

    signature = _model_signature(config, model)
    max_abs_frame_offset = _get(proxy, "max_abs_frame_offset", 8)
    if max_abs_frame_offset is None:
        max_abs_frame_offset = 8
    metadata = {
        "schema_version": TWO_STAGE_CHECKPOINT_SCHEMA_VERSION,
        "training_stage": training_stage,
        "action_mode": action_mode,
        "proxy_type": proxy_type,
        "proxy_dim": _proxy_dim(config, action_mode, model),
        "proxy_max_abs_frame_offset": int(max_abs_frame_offset),
        "proxy_relative_time_mode": str(
            _get(proxy, "relative_time_mode", "always")
        ).strip().lower(),
        "latent_dim": _latent_dim(config, model),
        "latent_normalization": _latent_normalization(config),
        "finetune_scheme": scheme,
        "finetune_initialization": finetune_initialization,
        "finetune_substage": substage,
        "completed_warmup_steps": warmup_steps,
        "completed_joint_steps": joint_steps,
        **signature,
    }
    if scheme == "state_conditioned_controller":
        metadata["state_controller"] = _state_controller_contract(config, model)
    initialization = _get(_finetune_config(config), "initialization", None)
    if initialization is not None:
        metadata["initialization"] = _serializable(initialization)
    return _serializable(metadata)


def extract_checkpoint_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Extract metadata while accepting early two-stage key spellings."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")
    for key in (TWO_STAGE_METADATA_KEY, "metadata", "checkpoint_metadata"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and "training_stage" in value:
            return dict(value)
    # A flat metadata dictionary is convenient for validation/tests, but an old
    # checkpoint without these fields is intentionally not guessed to be stage 1.
    if "training_stage" in checkpoint:
        return dict(checkpoint)
    raise ValueError(
        "Checkpoint has no two-stage metadata. Legacy checkpoints must be loaded "
        "through the legacy train_utils path, not the two-stage resume loader."
    )


def _metadata_dict(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if any(
        key in metadata
        for key in (TWO_STAGE_METADATA_KEY, "metadata", "checkpoint_metadata")
    ):
        return extract_checkpoint_metadata(metadata)
    if "training_stage" not in metadata:
        raise ValueError("Stage-1 checkpoint metadata is missing training_stage")
    return dict(metadata)


def _assert_same(
    field: str, source: Any, expected: Any, *, required: bool = False
) -> None:
    if required and source is None:
        raise ValueError(f"Stage-1 checkpoint metadata is missing {field}")
    if source is not None and expected is not None and source != expected:
        raise ValueError(
            f"Stage-1 checkpoint {field} mismatch: checkpoint={source!r}, "
            f"current={expected!r}"
        )


def validate_stage1_checkpoint(
    metadata: Mapping[str, Any],
    config: Any,
    model: nn.Module | None = None,
) -> dict[str, Any]:
    """Validate the stage-1 contract before mutating a stage-2 model."""
    source = _metadata_dict(metadata)
    schema_version = int(
        source.get("schema_version", TWO_STAGE_CHECKPOINT_SCHEMA_VERSION)
    )
    if schema_version != TWO_STAGE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported two-stage checkpoint metadata schema_version: "
            f"{schema_version}"
        )
    stage = str(source.get("training_stage", "")).lower()
    if stage != "proxy_pretrain":
        raise ValueError(
            "Stage-2 initialization requires a proxy_pretrain checkpoint; "
            f"got training_stage={stage!r}"
        )

    scheme = _normalize_scheme(_get(_finetune_config(config), "scheme", "reset"))
    action_mode = str(source.get("action_mode", source.get("proxy_type", ""))).lower()
    proxy_type = str(source.get("proxy_type", action_mode)).lower()
    if action_mode != proxy_type:
        raise ValueError(
            "Stage-1 checkpoint action/proxy provenance is inconsistent: "
            f"action_mode={action_mode!r}, proxy_type={proxy_type!r}"
        )

    if scheme in {
        "embedding_align",
        "real_to_latent",
        "state_conditioned_controller",
    } and (
        action_mode != "latent" or proxy_type != "latent"
    ):
        raise ValueError(
            f"finetune scheme {scheme!r} requires an NWM-LatentPT checkpoint; "
            f"got action_mode={action_mode!r}, proxy_type={proxy_type!r}"
        )

    configured_proxy_type = _get(_proxy_config(config), "type", None)
    if configured_proxy_type is not None:
        configured_proxy_type = str(configured_proxy_type).lower()
        _assert_same("proxy_type", proxy_type, configured_proxy_type, required=True)

    expected_proxy_window = _get(
        _proxy_config(config), "max_abs_frame_offset", None
    )
    if expected_proxy_window is not None:
        _assert_same(
            "proxy_max_abs_frame_offset",
            source.get("proxy_max_abs_frame_offset"),
            int(expected_proxy_window),
            required=True,
        )

    # Report the semantically important latent width first. Without this
    # ordering the same incompatibility surfaced as a generic proxy_dim error.
    if proxy_type == "latent":
        _assert_same(
            "latent_dim",
            source.get("latent_dim", source.get("proxy_dim")),
            _latent_dim(config, model),
            required=True,
        )

    expected_proxy_dim = _proxy_dim(config, proxy_type, model)
    # TimePT has no action tensor, so null is the canonical dimension. Other
    # proxy sources must record and match their actual cached vector width.
    if proxy_type != "none" or expected_proxy_dim is not None:
        _assert_same(
            "proxy_dim",
            source.get("proxy_dim"),
            expected_proxy_dim,
            required=True,
        )

    expected_signature = _model_signature(config, model)
    for field in ("hidden_dim", "model_target", "model_size", "context_size"):
        _assert_same(
            field,
            source.get(field),
            expected_signature[field],
            required=expected_signature[field] is not None,
        )

    # A LatentPT source carries an input-coordinate contract even when scheme A
    # discards the encoder in its forward path. Check it for every LatentPT
    # transition so an incompatible retained adapter cannot be serialized.
    if proxy_type == "latent":
        source_latent_dim = source.get("latent_dim", source.get("proxy_dim"))
        expected_latent_dim = _latent_dim(config, model)
        _assert_same(
            "latent_dim", source_latent_dim, expected_latent_dim, required=True
        )
        source_normalization = _serializable(source.get("latent_normalization"))
        expected_normalization = _latent_normalization(config)
        _assert_same(
            "latent_normalization",
            source_normalization,
            expected_normalization,
            required=True,
        )

    return {
        "training_stage": stage,
        "source_action_mode": action_mode,
        "source_proxy_type": proxy_type,
        "finetune_scheme": scheme,
        "metadata": source,
    }


def validate_resume_checkpoint(
    metadata: Mapping[str, Any],
    config: Any | None = None,
    model: nn.Module | None = None,
    expected_substage: str | None = None,
) -> dict[str, Any]:
    """Validate provenance and phase counters before an exact resume load.

    ``config`` is optional for API compatibility. The two-stage trainer passes
    it so proxy source/window/normalization are checked in addition to the exact
    model state dict. A resume must represent the same experiment, unlike a
    stage-1-to-stage-2 initialization which deliberately adds new modules.
    """
    source = _metadata_dict(metadata)
    schema_version = int(
        source.get("schema_version", TWO_STAGE_CHECKPOINT_SCHEMA_VERSION)
    )
    if schema_version != TWO_STAGE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported two-stage checkpoint metadata schema_version: "
            f"{schema_version}"
        )

    stage = str(source.get("training_stage", "")).strip().lower()
    if stage not in {"proxy_pretrain", "real_finetune"}:
        raise ValueError(
            "Two-stage resume requires training_stage=proxy_pretrain or "
            f"real_finetune; got {stage!r}"
        )

    recorded_substage = _normalize_substage(source.get("finetune_substage"))
    if recorded_substage is None:
        raise ValueError("Two-stage resume metadata is missing finetune_substage")
    allowed_substages = (
        {"pretrain"}
        if stage == "proxy_pretrain"
        else {"warmup", "transition", "joint"}
    )
    if recorded_substage not in allowed_substages:
        raise ValueError(
            "Checkpoint training_stage/substage mismatch: "
            f"training_stage={stage!r}, finetune_substage={recorded_substage!r}"
        )

    expected = _normalize_substage(expected_substage)
    if expected is not None:
        compatible = recorded_substage == expected or (
            recorded_substage == "transition" and expected == "joint"
        )
        if not compatible:
            raise ValueError(
                "Checkpoint finetune_substage mismatch: "
                f"checkpoint={recorded_substage!r}, expected={expected!r}"
            )

    try:
        warmup_steps = int(source.get("completed_warmup_steps", 0))
        joint_steps = int(source.get("completed_joint_steps", 0))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Completed warmup/joint step counts must be integers"
        ) from error
    if warmup_steps < 0 or joint_steps < 0:
        raise ValueError("Completed warmup/joint step counts must be non-negative")
    if stage == "proxy_pretrain" and (warmup_steps != 0 or joint_steps != 0):
        raise ValueError(
            "Proxy-pretrain resume metadata cannot contain fine-tuning steps"
        )
    if recorded_substage in {"warmup", "transition"} and joint_steps != 0:
        raise ValueError(
            f"{recorded_substage} checkpoint cannot contain completed joint steps"
        )

    if stage == "proxy_pretrain":
        # New checkpoints store null; accept the historical schema-v1 spelling
        # ``reset`` so earlier opt-in pretraining runs remain resumable.
        source_scheme = source.get("finetune_scheme")
        if source_scheme not in {None, "reset"}:
            raise ValueError(
                "Proxy-pretrain checkpoint must not name a fine-tuning scheme; "
                f"got {source_scheme!r}"
            )
    else:
        source_scheme = _normalize_scheme(source.get("finetune_scheme"))
        # Schema-v1 stage-2 checkpoints written before this field existed were
        # necessarily initialized from stage 1: random-init was not supported.
        source_initialization = str(
            source.get("finetune_initialization", "stage1_checkpoint")
        ).strip().lower()
        if source_initialization not in {"random", "stage1_checkpoint", "init_checkpoint"}:
            raise ValueError(
                "Unknown checkpoint finetune_initialization: "
                f"{source_initialization!r}"
            )

    if config is not None:
        expected_stage = str(
            _two_stage_value(config, "training_stage", "legacy")
        ).strip().lower()
        _assert_same("training_stage", stage, expected_stage, required=True)

        expected_action_mode = str(
            _two_stage_value(config, "action_mode", "none")
        ).strip().lower()
        _assert_same(
            "action_mode",
            source.get("action_mode"),
            expected_action_mode,
            required=True,
        )

        proxy = _proxy_config(config)
        expected_proxy_type = str(
            _get(proxy, "type", expected_action_mode)
        ).strip().lower()
        _assert_same(
            "proxy_type",
            source.get("proxy_type"),
            expected_proxy_type,
            required=True,
        )
        expected_window = _get(proxy, "max_abs_frame_offset", 8)
        _assert_same(
            "proxy_max_abs_frame_offset",
            source.get("proxy_max_abs_frame_offset"),
            int(expected_window),
            required=True,
        )
        expected_relative_time_mode = str(
            _get(proxy, "relative_time_mode", "always")
        ).strip().lower()
        _assert_same(
            "proxy_relative_time_mode",
            source.get("proxy_relative_time_mode", "always"),
            expected_relative_time_mode,
            required=True,
        )
        expected_proxy_dim = _proxy_dim(config, expected_proxy_type, model)
        if expected_proxy_type != "none" or expected_proxy_dim is not None:
            _assert_same(
                "proxy_dim",
                source.get("proxy_dim"),
                expected_proxy_dim,
                required=True,
            )

        expected_signature = _model_signature(config, model)
        for field in ("hidden_dim", "model_target", "model_size", "context_size"):
            _assert_same(
                field,
                source.get(field),
                expected_signature[field],
                required=expected_signature[field] is not None,
            )

        if expected_proxy_type == "latent":
            _assert_same(
                "latent_dim",
                source.get("latent_dim", source.get("proxy_dim")),
                _latent_dim(config, model),
                required=True,
            )
            _assert_same(
                "latent_normalization",
                _serializable(source.get("latent_normalization")),
                _latent_normalization(config),
                required=True,
            )

        finetune = _finetune_config(config)
        if stage == "real_finetune":
            expected_scheme = _normalize_scheme(_get(finetune, "scheme", "reset"))
            _assert_same(
                "finetune_scheme", source_scheme, expected_scheme, required=True
            )
            if expected_scheme == "state_conditioned_controller":
                _assert_same(
                    "state_controller",
                    _serializable(source.get("state_controller")),
                    _state_controller_contract(config, model),
                    required=True,
                )
            expected_initialization = (
                "random"
                if bool(_get(finetune, "random_init", False))
                else "init_checkpoint" if (_get(finetune, "init_checkpoint", None)
                                            or _get(finetune, "dataset_protocol", "paper") == "go2")
                else "stage1_checkpoint"
            )
            _assert_same(
                "finetune_initialization",
                source_initialization,
                expected_initialization,
                required=True,
            )
            configured_warmup_steps = _get(finetune, "warmup_steps", None)
            configured_joint_steps = _get(finetune, "joint_steps", None)
            if (
                recorded_substage in {"transition", "joint"}
                and configured_warmup_steps is not None
                and warmup_steps != int(configured_warmup_steps)
            ):
                raise ValueError(
                    "Checkpoint completed_warmup_steps mismatch: "
                    f"checkpoint={warmup_steps}, expected="
                    f"{int(configured_warmup_steps)} before {recorded_substage}"
                )
            if (
                configured_warmup_steps is not None
                and warmup_steps > int(configured_warmup_steps)
            ):
                raise ValueError(
                    "Checkpoint completed_warmup_steps exceeds configured target"
                )
            if (
                configured_joint_steps is not None
                and joint_steps > int(configured_joint_steps)
            ):
                raise ValueError(
                    "Checkpoint completed_joint_steps exceeds configured target"
                )

    return {
        "metadata": source,
        "training_stage": stage,
        "finetune_substage": recorded_substage,
        "resume_substage": (
            "joint" if recorded_substage == "transition" else recorded_substage
        ),
        "completed_warmup_steps": warmup_steps,
        "completed_joint_steps": joint_steps,
        "finetune_initialization": (
            source_initialization if stage == "real_finetune" else None
        ),
    }


def _load_checkpoint(checkpoint_or_path: Any) -> dict[str, Any]:
    if isinstance(checkpoint_or_path, Mapping):
        return dict(checkpoint_or_path)
    path = os.fspath(checkpoint_or_path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older PyTorch versions
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must contain a mapping: {path}")
    return dict(checkpoint)


def _normalize_state_key(key: str) -> str:
    # torch.compile may insert this component at the root or below a wrapper.
    normalized = str(key).replace("_orig_mod.", "")
    if normalized.startswith("module."):
        normalized = normalized[len("module.") :]
    return normalized


def _model_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model")
    if state is None:
        state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint does not contain a model/state_dict mapping")
    normalized = {_normalize_state_key(key): value for key, value in state.items()}
    if len(normalized) != len(state):
        raise ValueError("Checkpoint model keys collide after wrapper-prefix removal")
    return normalized


def _matches_prefix(key: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        key == prefix.rstrip(".") or key.startswith(prefix) for prefix in prefixes
    )


def _module_names(keys: list[str]) -> list[str]:
    modules = set()
    for key in keys:
        parts = key.split(".")
        modules.add(".".join(parts[:-1]) if len(parts) > 1 else parts[0])
    return sorted(modules)


def load_finetune_weights(
    model: nn.Module, checkpoint_or_path: Any, config: Any,
) -> dict[str, Any]:
    """Initialize from a trained real-action EMA without importing training state.

    Unlike stage-1 transfer this requires the entire state, including inactive
    adapters and normalization buffers. Legacy real-action checkpoints provide
    their architecture contract through config instead of two_stage_metadata.
    """
    checkpoint = _load_checkpoint(checkpoint_or_path)
    source = checkpoint.get("config")
    if not isinstance(source, Mapping):
        raise ValueError("Complete initialization requires the source checkpoint config")
    if _get(source, "training_stage", "legacy") not in {"legacy", "real_finetune"}:
        raise ValueError("Initialization must come from a trained real-action model")
    if _get(source, "action_mode", "real") != "real":
        raise ValueError("Initialization requires action_mode=real")
    if _normalize_scheme(
        _get(_finetune_config(source), "scheme", "reset")
    ) in {"real_to_latent", "state_conditioned_controller"}:
        raise ValueError(
            "Controller checkpoints need their original inference interface"
        )
    metadata = checkpoint.get(TWO_STAGE_METADATA_KEY, {})
    if metadata and (metadata.get("training_stage") != "real_finetune"
                     or metadata.get("action_mode") != "real"):
        raise ValueError("Source metadata must describe real-action fine-tuning")
    fields = (
        ("model", "generator"),
        ("dataset", "context_size"), ("dataset", "image_size"),
        ("motion_condition", "enabled"),
        ("motion_condition", "real", "real_dim"),
        ("motion_condition", "real", "normalization"),
    )
    for keys in fields:
        expected, actual = _serializable(_path(config, *keys)), _serializable(_path(source, *keys))
        if expected is None or actual != expected:
            raise ValueError(f"Initialization config mismatch at {'.'.join(keys)}: {actual} != {expected}")
    state = checkpoint.get("ema")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Initialization requires a nonempty ema state")
    normalized = {_normalize_state_key(k): v for k, v in state.items()}
    if len(normalized) != len(state):
        raise ValueError("EMA keys collide after removing wrapper prefixes")
    target = _unwrap_model(model)
    current = target.state_dict()
    if current.keys() != normalized.keys():
        raise ValueError(f"EMA state keys differ: missing={sorted(current.keys() - normalized.keys())}, "
                         f"unexpected={sorted(normalized.keys() - current.keys())}")
    for key, value in current.items():
        other = normalized[key]
        if not isinstance(other, torch.Tensor) or value.shape != other.shape:
            raise ValueError(f"EMA parameter shape mismatch: {key}")
        if "motion_normalizer." in key and not torch.equal(value.cpu(), other.cpu()):
            raise ValueError(f"EMA normalization buffer mismatch: {key}")
    source_path = None
    checksum = None
    if not isinstance(checkpoint_or_path, Mapping):
        source_path = str(Path(checkpoint_or_path).resolve())
        digest = hashlib.sha256()
        with open(source_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        checksum = digest.hexdigest()
        expected_hash = _get(_finetune_config(config), "init_sha256", None)
        if expected_hash and checksum != expected_hash:
            raise ValueError("Initialization checkpoint SHA-256 changed")
    target.load_state_dict(normalized, strict=True)
    return {"path": source_path, "sha256": checksum, "weight_key": "ema",
            "source_train_steps": int(checkpoint.get("train_steps", 0))}


def load_stage1_weights(
    model: nn.Module,
    checkpoint_or_path: Any,
    config: Any,
) -> dict[str, Any]:
    """Initialize stage 2 while allowing only scheme-specific new modules."""
    checkpoint = _load_checkpoint(checkpoint_or_path)
    metadata = extract_checkpoint_metadata(checkpoint)
    validation = validate_stage1_checkpoint(metadata, config, model=model)
    scheme = validation["finetune_scheme"]

    if scheme == "reset":
        new_module_prefixes = _REAL_ADAPTER_PREFIXES
        optional_inactive_prefixes = _ALL_PROXY_PREFIXES
    elif scheme == "embedding_align":
        new_module_prefixes = _REAL_ADAPTER_PREFIXES
        optional_inactive_prefixes = _INACTIVE_PROXY_PREFIXES
    elif scheme == "real_to_latent":
        new_module_prefixes = _REAL_TO_LATENT_PREFIXES
        optional_inactive_prefixes = (
            *_REAL_ADAPTER_PREFIXES,
            *_INACTIVE_PROXY_PREFIXES,
        )
    else:
        new_module_prefixes = _STATE_CONTROLLER_PREFIXES
        optional_inactive_prefixes = (
            *_REAL_ADAPTER_PREFIXES,
            *_INACTIVE_PROXY_PREFIXES,
        )
    allowed_missing = (*new_module_prefixes, *optional_inactive_prefixes)

    source_state = _model_state(checkpoint)
    target = _unwrap_model(model)
    assert target is not None
    target_keys = set(target.state_dict())
    # New stage-2 modules are always reset, even when a broadly configured
    # stage-1 model happened to serialize an unused copy. Other inactive proxy
    # modules are retained when both models contain them and ignored only when
    # the stage-2 architecture intentionally omits them.
    skipped_keys = []
    for key in source_state:
        reset_new_module = _matches_prefix(key, new_module_prefixes)
        absent_inactive_module = (
            key not in target_keys
            and _matches_prefix(key, optional_inactive_prefixes)
        )
        if reset_new_module or absent_inactive_module:
            skipped_keys.append(key)
    filtered_state = {
        key: value
        for key, value in source_state.items()
        if key not in skipped_keys
    }
    try:
        result = target.load_state_dict(filtered_state, strict=False)
    except RuntimeError as error:
        raise RuntimeError(
            "Stage-1 model tensor mismatch while loading stage 2. "
            "Check model size, hidden_dim, latent_dim, and context_size.\n"
            f"{error}"
        ) from error

    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)
    invalid_missing = [
        key for key in missing_keys if not _matches_prefix(key, allowed_missing)
    ]
    if invalid_missing or unexpected_keys:
        raise RuntimeError(
            "Unexpected stage-1 -> stage-2 state-dict mismatch: "
            f"missing={invalid_missing}, unexpected={unexpected_keys}. "
            f"Only these missing prefixes are allowed for {scheme}: "
            f"{list(allowed_missing)}"
        )

    loaded_keys = [key for key in filtered_state if key in target_keys]
    summary = {
        **validation,
        "allowed_missing_prefixes": list(allowed_missing),
        "loaded_keys": loaded_keys,
        "loaded_modules": _module_names(loaded_keys),
        "skipped_keys": skipped_keys,
        "skipped_modules": _module_names(skipped_keys),
        "expected_missing_keys": missing_keys,
    }
    logger.info(
        "Loaded stage-1 modules for %s: %s; skipped/new modules: %s",
        scheme,
        summary["loaded_modules"],
        summary["skipped_modules"] or _module_names(missing_keys),
    )
    return summary


def load_two_stage_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    checkpoint_or_path: Any,
    expected_substage: str | None = None,
    config: Any | None = None,
) -> dict[str, Any]:
    """Strictly resume warm-up/joint work, or consume a transition checkpoint.

    A transition contains final warm-up model weights but intentionally does
    not restore a warm-up optimizer into the newly constructed joint optimizer.
    """
    checkpoint = _load_checkpoint(checkpoint_or_path)
    metadata = extract_checkpoint_metadata(checkpoint)
    validation = validate_resume_checkpoint(
        metadata,
        config=config,
        model=model,
        expected_substage=expected_substage,
    )
    recorded_substage = validation["finetune_substage"]

    target = _unwrap_model(model)
    assert target is not None
    state = _model_state(checkpoint)
    try:
        target.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(f"Exact two-stage resume model mismatch:\n{error}") from error

    optimizer_loaded = False
    if optimizer is not None and recorded_substage != "transition":
        optimizer_state = checkpoint.get("opt", checkpoint.get("optimizer"))
        if optimizer_state is None:
            raise ValueError(
                f"{recorded_substage!r} resume checkpoint has no optimizer state"
            )
        optimizer.load_state_dict(optimizer_state)
        optimizer_loaded = True

    progress = checkpoint.get("data_progress", {})
    if progress is None:
        progress = {}
    if not isinstance(progress, Mapping):
        raise TypeError("Checkpoint data_progress must be a mapping")
    try:
        batch_in_epoch = int(
            progress.get("batch_in_epoch", checkpoint.get("batch_in_epoch", 0))
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint batch_in_epoch must be an integer") from error
    if batch_in_epoch < 0:
        raise ValueError("Checkpoint batch_in_epoch must be non-negative")

    data_fingerprint = progress.get(
        "data_fingerprint", checkpoint.get("data_fingerprint")
    )
    validate_data_resume_fingerprint(
        data_fingerprint,
        config,
        validation["resume_substage"],
        required=batch_in_epoch > 0,
    )

    runtime_states = checkpoint.get("rank_runtime_states")
    runtime_state = None
    runtime_world_size = int(progress.get("world_size", 0) or 0)
    if runtime_states is not None:
        if not isinstance(runtime_states, Sequence) or isinstance(
            runtime_states, (str, bytes)
        ):
            raise TypeError("Checkpoint rank_runtime_states must be a sequence")
        runtime_states = list(runtime_states)
        if not runtime_states:
            raise ValueError("Checkpoint rank_runtime_states cannot be empty")
        if runtime_world_size not in {0, len(runtime_states)}:
            raise ValueError(
                "Checkpoint data_progress.world_size does not match "
                "rank_runtime_states"
            )
        runtime_world_size = len(runtime_states)
        current_world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        current_rank = (
            dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        )
        if batch_in_epoch > 0 and runtime_world_size != current_world_size:
            raise ValueError(
                "An in-epoch checkpoint must resume with the same distributed "
                "world size: checkpoint="
                f"{runtime_world_size}, current={current_world_size}"
            )
        if current_rank < runtime_world_size:
            selected = runtime_states[current_rank]
            if not isinstance(selected, Mapping):
                raise TypeError("Each rank runtime state must be a mapping")
            runtime_state = dict(selected)
    elif batch_in_epoch > 0:
        raise ValueError(
            "In-epoch checkpoint has no rank runtime state; exact continuation "
            "is not possible"
        )

    return {
        **validation,
        "finetune_substage": recorded_substage,
        "optimizer_loaded": optimizer_loaded,
        "epoch": int(checkpoint.get("epoch", 0)),
        "train_steps": int(checkpoint.get("train_steps", 0)),
        "batch_in_epoch": batch_in_epoch,
        "runtime_world_size": runtime_world_size,
        "runtime_state": runtime_state,
        "data_fingerprint": data_fingerprint,
    }


__all__ = [
    "DATA_RESUME_FINGERPRINT_SCHEMA_VERSION",
    "TWO_STAGE_CHECKPOINT_SCHEMA_VERSION",
    "TWO_STAGE_METADATA_KEY",
    "build_checkpoint_metadata",
    "build_data_resume_fingerprint",
    "capture_rng_state",
    "extract_checkpoint_metadata",
    "load_stage1_weights",
    "load_two_stage_resume",
    "restore_rng_state",
    "validate_data_resume_fingerprint",
    "validate_resume_checkpoint",
    "validate_stage1_checkpoint",
]
