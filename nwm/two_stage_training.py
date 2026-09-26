"""Opt-in training loop for two-stage NWM experiments.

The legacy loop in :mod:`train` remains the default.  This loop is entered only
when ``training_stage`` is explicitly ``proxy_pretrain`` or ``real_finetune``.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from time import time
from typing import Any

import torch
import torch.distributed as dist
import wandb
from einops import repeat
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra.utils import get_original_cwd

from data_utils import create_dataloader, prepare_datasets
from checkpoint_io import log_metrics, point_checkpoint, prune_periodic
from misc import get_unnormalize
from train_utils import (
    evaluate,
    sample_precomputed_vae_posterior,
    setup_diffusion,
    setup_model,
    setup_optimizer,
    setup_scheduler,
    setup_tokenizer,
    update_ema,
)
from two_stage_checkpoint import (
    TWO_STAGE_METADATA_KEY,
    build_checkpoint_metadata,
    build_data_resume_fingerprint,
    capture_rng_state,
    extract_checkpoint_metadata,
    load_stage1_weights,
    load_finetune_weights,
    load_two_stage_resume,
    restore_rng_state,
    validate_data_resume_fingerprint,
)
from two_stage_nwm import (
    ProxyMetrics,
    alignment_loss,
    assert_goal_alignment,
    build_optimizer_param_groups,
    configure_trainable_parameters,
    dense_motion_from_collated,
    flatten_goal_tensor,
    latent_action_l2_loss,
    LOCAL_PROXY_MAX_ABS_FRAME_OFFSET,
    normalize_finetune_scheme,
    predicted_latent_metrics,
    trainability_report,
    validate_two_stage_config,
)

logger = logging.getLogger(__name__)


class _PrecomputedPosteriorTokenizer(torch.nn.Module):
    """Minimal tokenizer contract when training consumes cached posteriors."""

    def __init__(self, scaling_factor: float):
        super().__init__()
        self.scaling_factor = float(scaling_factor)

    def encode(self, _value: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "Online VAE encoding is disabled because precomputed_latents.enabled=true"
        )


def _setup_training_tokenizer(config: Any, device: torch.device, log: Any):
    latent_config = config.dataset.get("precomputed_latents", {})
    if bool(latent_config.get("enabled", False)):
        scaling_factor = float(latent_config.get("scaling_factor", 0.18215))
        log.info(
            "Using cached VAE posterior statistics (scaling_factor=%s); "
            "the VAE model is not loaded on training GPUs",
            scaling_factor,
        )
        return _PrecomputedPosteriorTokenizer(scaling_factor).to(device)
    return setup_tokenizer(config, device)


def _wandb_active(config: Any, rank: int) -> bool:
    return bool(
        rank == 0
        and config.training.get("wandb_enabled", False)
        and wandb.run is not None
    )


def _wandb_log(config: Any, rank: int, values: Mapping[str, Any], step: int) -> None:
    if not _wandb_active(config, rank):
        return
    wandb.log(dict(values), step=int(step))


def _resolve_user_path(value: Any) -> str:
    """Resolve CLI/config paths against the launch directory, not Hydra's run dir."""
    path = os.path.expanduser(os.fspath(value))
    if not os.path.isabs(path):
        try:
            launch_dir = get_original_cwd()
        except ValueError:
            launch_dir = os.getcwd()
        path = os.path.join(launch_dir, path)
    return os.path.realpath(path)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while True:
        if isinstance(model, DDP):
            model = model.module
            continue
        if hasattr(model, "_orig_mod") and isinstance(model._orig_mod, torch.nn.Module):
            model = model._orig_mod
            continue
        return model


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    config: Any,
    substage: str,
    completed_warmup_steps: int,
    completed_joint_steps: int,
    train_steps: int,
    epoch: int,
    include_optimizer: bool,
    batch_in_epoch: int = 0,
    runtime_states: Sequence[Mapping[str, Any]] | None = None,
    data_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw_model = _unwrap_model(model)
    batch_in_epoch = int(batch_in_epoch)
    if batch_in_epoch < 0:
        raise ValueError("batch_in_epoch must be non-negative")
    if runtime_states is None:
        runtime_states = ({"rng_state": capture_rng_state()},)
    runtime_states = [dict(state) for state in runtime_states]
    if not runtime_states:
        raise ValueError("runtime_states cannot be empty")
    if data_fingerprint is None:
        data_fingerprint = build_data_resume_fingerprint(config, substage)
    if batch_in_epoch > 0:
        missing_lengths = [
            name
            for name in ("dataset_length", "loader_length")
            if data_fingerprint.get(name) is None
        ]
        if missing_lengths:
            raise ValueError(
                "An in-epoch checkpoint requires data fingerprint lengths: "
                f"{missing_lengths}"
            )
    payload = {
        "model": raw_model.state_dict(),
        "ema": _unwrap_model(ema).state_dict(),
        "config": OmegaConf.to_container(config, resolve=True),
        "epoch": int(epoch),
        "train_steps": int(train_steps),
        "data_progress": {
            "batch_in_epoch": batch_in_epoch,
            "world_size": len(runtime_states),
            "data_fingerprint": dict(data_fingerprint),
        },
        "rank_runtime_states": runtime_states,
        "best_metrics": dict(config.training.get("best_metrics", {})),
        TWO_STAGE_METADATA_KEY: build_checkpoint_metadata(
            config,
            substage,
            completed_warmup_steps,
            completed_joint_steps,
            model=raw_model,
        ),
    }
    if include_optimizer and optimizer is not None:
        payload["opt"] = optimizer.state_dict()
    if include_optimizer and scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    # GradScaler is continuous across warm-up and joint. Transition artifacts
    # intentionally omit the phase-specific optimizer/scheduler but retain AMP
    # scale history so external transition resume matches uninterrupted runs.
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    return payload


def save_two_stage_checkpoint(path: str, **kwargs: Any) -> str:
    """Atomically save one exact substage checkpoint (rank zero only)."""
    final_path = os.path.abspath(os.fspath(path))
    checkpoint_dir = os.path.dirname(final_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    temporary_path = os.path.join(
        checkpoint_dir,
        f".{os.path.basename(final_path)}.{uuid.uuid4().hex}.tmp",
    )
    payload = _checkpoint_payload(**kwargs)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, final_path)
    finally:
        if os.path.lexists(temporary_path):
            os.unlink(temporary_path)
        del payload
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return path


def _periodic_checkpoint_due(
    *,
    training_stage: str,
    current_steps: int,
    global_steps: int,
    target_steps: int,
    interval: int,
) -> bool:
    """Return whether this step should publish a periodic checkpoint."""
    if interval <= 0:
        raise ValueError("ckpt_every must be positive")
    if training_stage == "real_finetune":
        return (
            current_steps > 0
            and current_steps % interval == 0
            and current_steps < target_steps
        )
    return (
        global_steps > 0
        and global_steps % interval == 0
        and current_steps < target_steps
    )


def _evaluation_due(
    *,
    global_steps: int,
    interval: int,
    eval_at_first_step: bool,
) -> bool:
    """Match the legacy training-loop evaluation schedule."""
    global_steps = int(global_steps)
    interval = int(interval)
    if global_steps < 0:
        raise ValueError("global_steps must be non-negative")
    return bool(
        (interval > 0 and global_steps % interval == 0)
        or (eval_at_first_step and global_steps == 1)
    )


def _phase_evaluation_due(
    *,
    training_stage: str,
    scheme: str | None,
    substage: str,
    current_steps: int,
    global_steps: int,
    config: Any,
) -> bool:
    """Keep legacy scheduling, except for the controller's independent axes."""
    if (
        training_stage == "real_finetune"
        and scheme == "state_conditioned_controller"
    ):
        interval = int(
            config.finetune.get("controller_eval_every", 1000)
            if substage == "warmup"
            else config.get("eval_every", 5000)
        )
        return _evaluation_due(
            global_steps=current_steps,
            interval=interval,
            eval_at_first_step=bool(config.get("eval_at_first_step", True)),
        )
    return _evaluation_due(
        global_steps=global_steps,
        interval=int(config.get("eval_every", 0)),
        eval_at_first_step=bool(config.get("eval_at_first_step", True)),
    )


def _controller_phase_name(scheme: str | None, substage: str) -> str | None:
    if scheme != "state_conditioned_controller":
        return None
    return "controller_pretrain" if substage == "warmup" else "joint"


def _define_controller_wandb_metrics(config: Any, rank: int) -> None:
    if not _wandb_active(config, rank):
        return
    for phase in ("controller_pretrain", "joint"):
        step_name = f"{phase}/step"
        wandb.define_metric(step_name)
        wandb.define_metric(f"{phase}/train/*", step_metric=step_name)
        wandb.define_metric(f"{phase}/eval/*", step_metric=step_name)


def _distributed_cuda_memory_gib(device: torch.device) -> dict[str, float] | None:
    """Return the largest current/peak CUDA footprint across all ranks."""

    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize(device)
    values = torch.tensor(
        [
            torch.cuda.memory_allocated(device),
            torch.cuda.memory_reserved(device),
            torch.cuda.max_memory_allocated(device),
            torch.cuda.max_memory_reserved(device),
        ],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
    current_allocated, current_reserved, peak_allocated, peak_reserved = (
        value / 1024**3 for value in values.tolist()
    )
    return {
        "current_allocated_gib": current_allocated,
        "current_reserved_gib": current_reserved,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
    }


def _reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _retained_stage2_checkpoint_path(
    checkpoint_dir: str,
    substage: str,
    current_steps: int,
) -> str:
    """Build the stable per-substage checkpoint path used by Stage 2."""
    if substage not in {"warmup", "joint"}:
        raise ValueError(f"Unsupported retained stage-2 substage: {substage!r}")
    if current_steps < 0:
        raise ValueError("current_steps must be non-negative")
    return os.path.join(
        checkpoint_dir,
        f"{substage}_{current_steps:07d}.pth.tar",
    )


def _retained_stage1_checkpoint_path(
    checkpoint_dir: str,
    current_steps: int,
) -> str:
    """Build a retained periodic Stage-1 checkpoint filename."""
    if current_steps < 0:
        raise ValueError("current_steps must be non-negative")
    return os.path.join(
        checkpoint_dir,
        f"pretrain_{current_steps:07d}.pth.tar",
    )


def _atomic_update_latest_checkpoint(target_path: str, latest_path: str) -> None:
    """Atomically point latest at a retained checkpoint using a relative link."""
    target_path = os.path.abspath(os.fspath(target_path))
    latest_path = os.path.abspath(os.fspath(latest_path))
    latest_dir = os.path.dirname(latest_path)
    os.makedirs(latest_dir, exist_ok=True)
    temporary_link = os.path.join(
        latest_dir,
        f".{os.path.basename(latest_path)}.{uuid.uuid4().hex}.tmp",
    )
    try:
        os.symlink(os.path.relpath(target_path, latest_dir), temporary_link)
        os.replace(temporary_link, latest_path)
    finally:
        if os.path.lexists(temporary_link):
            os.unlink(temporary_link)


def _save_training_checkpoint(
    checkpoint_dir: str,
    *,
    training_stage: str,
    substage: str,
    current_steps: int,
    **kwargs: Any,
) -> str:
    """Save a retained numbered checkpoint and atomically update latest."""
    latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
    if training_stage == "real_finetune":
        checkpoint_path = _retained_stage2_checkpoint_path(
            checkpoint_dir,
            substage,
            current_steps,
        )
    elif training_stage == "proxy_pretrain":
        checkpoint_path = _retained_stage1_checkpoint_path(
            checkpoint_dir,
            current_steps,
        )
    else:
        raise ValueError(f"Unsupported training_stage: {training_stage!r}")
    save_two_stage_checkpoint(
        checkpoint_path,
        substage=substage,
        **kwargs,
    )
    _atomic_update_latest_checkpoint(checkpoint_path, latest_path)
    prune_periodic(checkpoint_dir, kwargs["config"].training.get("keep_periodic", 3))
    return checkpoint_path


def _proxy_metrics_state(metrics: ProxyMetrics) -> dict[str, float]:
    return {
        name: float(getattr(metrics, name))
        for name in metrics.__dataclass_fields__
    }


def _restore_proxy_metrics(state: Any) -> ProxyMetrics:
    metrics = ProxyMetrics()
    if state is None:
        return metrics
    if not isinstance(state, Mapping):
        raise TypeError("Saved proxy_metrics state must be a mapping")
    expected = set(metrics.__dataclass_fields__)
    unexpected = sorted(set(state) - expected)
    if unexpected:
        raise ValueError(f"Unknown saved proxy metric fields: {unexpected}")
    for name in expected:
        value = float(state.get(name, 0.0))
        if value < 0:
            raise ValueError(f"Saved proxy metric {name} must be non-negative")
        setattr(metrics, name, value)
    return metrics


def _gather_runtime_states(
    proxy_metrics: ProxyMetrics,
    valid_alignment_batches: int,
    completed_samples_per_rank: int | None = None,
) -> list[dict[str, Any]]:
    local_state = {
        "rng_state": capture_rng_state(),
        "proxy_metrics": _proxy_metrics_state(proxy_metrics),
        "valid_alignment_batches": int(valid_alignment_batches),
    }
    if completed_samples_per_rank is not None:
        completed_samples_per_rank = int(completed_samples_per_rank)
        if completed_samples_per_rank < 0:
            raise ValueError("completed_samples_per_rank must be non-negative")
        local_state["completed_samples_per_rank"] = completed_samples_per_rank
    if not dist.is_available() or not dist.is_initialized():
        return [local_state]
    states: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(states, local_state)
    return [dict(state) for state in states]


def _phase_target_samples_per_rank(
    config: Any,
    training_stage: str,
    substage: str,
) -> int | None:
    """Return the opt-in exact local sample budget for one training phase."""
    if training_stage == "proxy_pretrain":
        value = config.training.get("target_samples_per_rank", None)
    elif substage == "warmup":
        value = config.finetune.get("warmup_samples_per_rank", None)
    elif substage == "joint":
        value = config.finetune.get("joint_samples_per_rank", None)
    else:
        value = None
    if value in (None, "", "null"):
        return None
    value = int(value)
    if value < 0:
        raise ValueError("Phase target_samples_per_rank must be non-negative")
    return value


def _phase_complete(
    current_steps: int,
    target_steps: int,
    completed_samples_per_rank: int,
    target_samples_per_rank: int | None,
) -> bool:
    if target_samples_per_rank is None:
        return int(current_steps) >= int(target_steps)
    return int(completed_samples_per_rank) >= int(target_samples_per_rank)


def _training_batch_size(batch: Mapping[str, Any]) -> int:
    if "k" not in batch:
        raise KeyError("Training batches must contain k for sample accounting")
    shape = torch.as_tensor(batch["k"]).shape
    if not shape or int(shape[0]) < 1:
        raise ValueError("Training batch k must have a positive batch dimension")
    return int(shape[0])


def _truncate_batch_value(value: Any, batch_size: int, limit: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and int(value.shape[0]) == batch_size:
            return value[:limit]
        return value
    if isinstance(value, Mapping):
        return {
            key: _truncate_batch_value(item, batch_size, limit)
            for key, item in value.items()
        }
    if isinstance(value, list):
        if len(value) == batch_size:
            return value[:limit]
        return [_truncate_batch_value(item, batch_size, limit) for item in value]
    if isinstance(value, tuple):
        if len(value) == batch_size:
            return value[:limit]
        return tuple(
            _truncate_batch_value(item, batch_size, limit) for item in value
        )
    return value


def _truncate_motion_groups(
    groups: Mapping[str, Mapping[str, Any]],
    limit: int,
) -> dict[str, dict[str, Any]]:
    truncated: dict[str, dict[str, Any]] = {}
    for motion_type, payload in groups.items():
        if not isinstance(payload, Mapping) or "sample_indices" not in payload:
            raise ValueError("Collated motion groups require sample_indices")
        indices = torch.as_tensor(payload["sample_indices"], dtype=torch.int64)
        if indices.ndim != 1:
            raise ValueError("Collated motion sample_indices must be one-dimensional")
        keep = indices < int(limit)
        if not bool(keep.any()):
            continue
        selected: dict[str, Any] = {}
        for name, value in payload.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and int(
                value.shape[0]
            ) == int(indices.numel()):
                selected[name] = value[keep]
            elif isinstance(value, list) and len(value) == int(indices.numel()):
                selected[name] = [
                    item for item, retain in zip(value, keep.tolist()) if retain
                ]
            else:
                selected[name] = value
        truncated[str(motion_type)] = selected
    return truncated


def _truncate_training_batch(
    batch: Mapping[str, Any],
    limit: int,
) -> dict[str, Any]:
    """Take a local batch prefix while preserving collated motion groups."""
    batch_size = _training_batch_size(batch)
    limit = int(limit)
    if limit < 1 or limit > batch_size:
        raise ValueError(
            f"Training batch limit must be within [1,{batch_size}], got {limit}"
        )
    if limit == batch_size:
        return dict(batch)
    truncated = {
        key: (
            _truncate_motion_groups(value, limit)
            if key == "motion" and isinstance(value, Mapping)
            else _truncate_batch_value(value, batch_size, limit)
        )
        for key, value in batch.items()
    }
    if _training_batch_size(truncated) != limit:
        raise RuntimeError("Training batch truncation produced the wrong size")
    return truncated


def _budget_training_batch(
    batch: Mapping[str, Any],
    completed_samples_per_rank: int,
    target_samples_per_rank: int | None,
) -> tuple[dict[str, Any] | Mapping[str, Any], int]:
    """Crop only the final optimizer batch to hit an exact sample budget."""
    batch_size = _training_batch_size(batch)
    if target_samples_per_rank is None:
        return batch, batch_size
    remaining = int(target_samples_per_rank) - int(completed_samples_per_rank)
    if remaining < 1:
        raise RuntimeError("Sample-budgeted phase received a batch after completion")
    if batch_size > remaining:
        batch = _truncate_training_batch(batch, remaining)
        batch_size = remaining
    return batch, batch_size


def _set_loader_epoch_seed(loader, config_seed: int, rank: int, epoch: int) -> None:
    """Make worker RNG replayable without touching legacy DataLoaders."""
    # Keep the seed within torch.Generator's signed 64-bit range.
    seed = (
        int(config_seed)
        + 1_000_003 * int(rank)
        + 1_000_000_007 * int(epoch)
    ) % (2**63 - 1)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader.generator = generator


def _phase_progress_after_iteration(
    epoch: int,
    batch_in_epoch: int,
    loader_length: int,
) -> tuple[int, int, bool]:
    """Advance epochs only after the iterator was actually exhausted.

    A phase target may be reached in the middle of an epoch. Keeping that
    cursor in the phase-final checkpoint is what permits an exact continuation
    when the configured target is later extended.
    """
    epoch = int(epoch)
    batch_in_epoch = int(batch_in_epoch)
    loader_length = int(loader_length)
    if loader_length < 1:
        raise ValueError("loader_length must be positive")
    if batch_in_epoch < 0 or batch_in_epoch > loader_length:
        raise ValueError(
            "batch_in_epoch must be within the current DataLoader: "
            f"cursor={batch_in_epoch}, batches={loader_length}"
        )
    if batch_in_epoch == loader_length:
        return epoch + 1, 0, True
    return epoch, batch_in_epoch, False


def _restore_completed_phase_rng(
    current_steps: int,
    target_steps: int,
    rng_state: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Restore a resumed phase RNG even when its training loop is skipped."""
    if int(current_steps) >= int(target_steps) and rng_state is not None:
        restore_rng_state(rng_state)
        return None
    return rng_state


def _make_optimizer(
    config: Any,
    model: torch.nn.Module,
    *,
    training_stage: str,
    scheme: str | None,
    substage: str | None,
) -> torch.optim.Optimizer:
    finetune = config.get("finetune", {})
    base_lr = float(config.training.optimizer.get("lr", 1e-4))
    # Stage 1 has no adapter/backbone two-rate policy: every selected module
    # follows the ordinary pretraining optimizer learning rate.
    adapter_lr = base_lr
    backbone_lr = base_lr
    if training_stage == "real_finetune":
        adapter_lr = float(finetune.get("adapter_lr", base_lr))
        backbone_lr = float(finetune.get("backbone_lr", base_lr))
        if scheme == "reset":
            phase_key = "warmup_adapter_lr" if substage == "warmup" else "joint_adapter_lr"
            adapter_lr = float(finetune.get(phase_key, adapter_lr))
        if scheme == "state_conditioned_controller":
            if substage == "warmup":
                adapter_lr = float(
                    finetune.get("controller_pretrain_lr", 1.0e-3)
                )
            else:
                joint_lr = float(finetune.get("joint_lr", 1.0e-4))
                adapter_lr = joint_lr
                backbone_lr = joint_lr
    groups = build_optimizer_param_groups(
        model,
        training_stage=training_stage,
        finetune_scheme=scheme,
        finetune_substage=substage,
        adapter_lr=adapter_lr,
        backbone_lr=backbone_lr,
    )
    if (
        training_stage == "real_finetune"
        and scheme == "state_conditioned_controller"
        and substage == "warmup"
    ):
        controller_weight_decay = float(
            finetune.get("controller_pretrain_weight_decay", 0.04)
        )
        for group in groups:
            group["weight_decay"] = controller_weight_decay
    return setup_optimizer(config, groups)


def _make_scheduler(
    config: Any,
    optimizer: torch.optim.Optimizer,
    *,
    training_stage: str,
    scheme: str | None,
    substage: str | None,
):
    """Use the paper schedule only for controller pretraining."""
    if (
        training_stage != "real_finetune"
        or scheme != "state_conditioned_controller"
    ):
        return setup_scheduler(config, optimizer)
    if substage == "joint":
        # The requested joint protocol is a fixed 1e-4, matching nwm-real.
        return None
    finetune = config.get("finetune", {})
    total_steps = int(finetune.get("warmup_steps", 0))
    lr_warmup_steps = int(finetune.get("controller_lr_warmup_steps", 300))
    if total_steps < 1:
        raise ValueError("Controller pretraining requires warmup_steps > 0")
    if not 0 <= lr_warmup_steps <= total_steps:
        raise ValueError(
            "controller_lr_warmup_steps must be within [0, warmup_steps]"
        )

    def multiplier(completed_steps: int) -> float:
        completed_steps = int(completed_steps)
        if lr_warmup_steps > 0 and completed_steps < lr_warmup_steps:
            return float(completed_steps + 1) / float(lr_warmup_steps)
        cosine_steps = total_steps - lr_warmup_steps
        if cosine_steps <= 0:
            return 1.0
        progress = (completed_steps - lr_warmup_steps) / float(cosine_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _log_trainability(model, optimizer, stage, substage, log):
    report = trainability_report(model, optimizer)
    log.info(
        "Trainability stage=%s substage=%s: trainable_names=%s",
        stage,
        substage,
        report["trainable_parameter_names"],
    )
    log.info(
        "Parameter counts: frozen=%d trainable=%d optimizer_groups=%s",
        report["frozen_parameter_count"],
        report["trainable_parameter_count"],
        report["optimizer_groups"],
    )
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if optimizer_ids != expected_ids:
        raise AssertionError("Optimizer parameters do not exactly match requires_grad=True")


def _encode_batch(tokenizer, batch, config, device):
    use_precomputed = bool(
        config.dataset.get("precomputed_latents", {}).get("enabled", False)
    )
    with torch.no_grad():
        if use_precomputed:
            posterior_mean = batch["posterior_mean"].to(device, non_blocking=True)
            posterior_logvar = batch["posterior_logvar"].to(device, non_blocking=True)
            sequence = sample_precomputed_vae_posterior(
                posterior_mean,
                posterior_logvar,
                tokenizer.scaling_factor,
            )
        else:
            video = batch["video"].to(device, non_blocking=True)
            if video.ndim != 5:
                raise ValueError(f"video must be [B,T,C,H,W], got {tuple(video.shape)}")
            batch_size, sequence_length = video.shape[:2]
            sequence = tokenizer.encode(video.flatten(0, 1)).unflatten(
                0, (batch_size, sequence_length)
            )

    batch_size, sequence_length = sequence.shape[:2]
    context_size = int(config.dataset.context_size)
    num_goals = sequence_length - context_size
    if num_goals < 1:
        raise ValueError("A training sequence must contain at least one target goal")
    rel_t = torch.as_tensor(batch["k"], dtype=torch.float32)
    frame_offset = torch.as_tensor(batch["frame_offset"], dtype=torch.int64)
    rel_t = flatten_goal_tensor(
        rel_t, batch_size=batch_size, num_goals=num_goals, name="rel_t"
    ).to(device, non_blocking=True)
    frame_offset = flatten_goal_tensor(
        frame_offset,
        batch_size=batch_size,
        num_goals=num_goals,
        name="frame_offset",
    ).to(device, non_blocking=True)
    target = sequence[:, context_size:].flatten(0, 1)
    context = repeat(
        sequence[:, :context_size], "b t ... -> (b g) t ...", g=num_goals
    )
    assert_goal_alignment(
        target_batch=target, rel_t=rel_t, frame_offset=frame_offset
    )
    return target, context, rel_t, frame_offset, batch_size, num_goals


def _flatten_optional(batch, key, batch_size, num_goals, device, dtype=None):
    if key not in batch:
        return None
    value = flatten_goal_tensor(
        batch[key], batch_size=batch_size, num_goals=num_goals, name=key
    )
    return value.to(device=device, dtype=dtype, non_blocking=True)


def _real_action(batch, raw_model, batch_size, num_goals, device):
    if "real_action" in batch:
        action = _flatten_optional(
            batch, "real_action", batch_size, num_goals, device, torch.float32
        )
        valid = torch.ones(action.shape[0], dtype=torch.bool, device=device)
        return action, valid
    dim = raw_model.motion_condition_encoder.input_dims["real"]
    return dense_motion_from_collated(
        batch.get("motion"),
        motion_type="real",
        batch_size=batch_size,
        num_goals=num_goals,
        input_dim=dim,
        device=device,
        require_all=True,
    )


def _proxy_action(batch, raw_model, action_mode, batch_size, num_goals, device):
    dim = raw_model.motion_condition_encoder.input_dims[action_mode]
    if "proxy_action" in batch:
        action = _flatten_optional(
            batch, "proxy_action", batch_size, num_goals, device, torch.float32
        )
        valid = _flatten_optional(
            batch, "proxy_valid", batch_size, num_goals, device, torch.bool
        )
        if action.shape != (batch_size * num_goals, dim):
            raise ValueError(
                f"proxy_action must flatten to [{batch_size * num_goals},{dim}], "
                f"got {tuple(action.shape)}"
            )
        return action, valid
    return dense_motion_from_collated(
        batch.get("motion"),
        motion_type=action_mode,
        batch_size=batch_size,
        num_goals=num_goals,
        input_dim=dim,
        device=device,
        require_all=False,
    )


def _alignment_targets(batch, raw_model, batch_size, num_goals, device):
    latent_dim = raw_model.motion_condition_encoder.input_dims["latent"]
    teacher = _flatten_optional(
        batch, "teacher_latent", batch_size, num_goals, device, torch.float32
    )
    valid = _flatten_optional(
        batch, "latent_valid", batch_size, num_goals, device, torch.bool
    )
    if teacher is None or valid is None:
        raise ValueError(
            "Latent-teacher training requires teacher_latent and latent_valid "
            "dataset fields"
        )
    if teacher.shape != (batch_size * num_goals, latent_dim):
        raise ValueError(
            f"teacher_latent must flatten to [{batch_size * num_goals},{latent_dim}], "
            f"got {tuple(teacher.shape)}"
        )
    return teacher, valid


def _finite_nonzero_gradient(named_parameters, prefix):
    gradients = [
        parameter.grad
        for name, parameter in named_parameters
        if name.startswith(prefix) and parameter.requires_grad
    ]
    return any(
        gradient is not None
        and bool(torch.isfinite(gradient).all())
        and bool((gradient != 0).any())
        for gradient in gradients
    )


def _assert_first_step_gradients(raw_model, scheme, substage):
    named = list(raw_model.named_parameters())
    if scheme in {"reset", "embedding_align"}:
        adapter_prefix = "motion_condition_encoder.real_"
    elif scheme == "real_to_latent":
        adapter_prefix = "real_to_latent."
    else:
        adapter_prefix = "state_conditioned_controller."
    if not _finite_nonzero_gradient(named, adapter_prefix):
        raise AssertionError(
            f"{adapter_prefix} has no nonzero finite gradient in {substage}"
        )
    if substage == "warmup":
        illegal = [
            name
            for name, parameter in named
            if not name.startswith(adapter_prefix) and parameter.grad is not None
        ]
        if illegal:
            raise AssertionError(f"Frozen parameters received warmup gradients: {illegal}")
    else:
        backbone = [
            (name, p)
            for name, p in named
            if p.requires_grad
            and not name.startswith(adapter_prefix)
            and not name.startswith("motion_condition_encoder.")
        ]
        if not any(
            p.grad is not None
            and bool(torch.isfinite(p.grad).all())
            and bool((p.grad != 0).any())
            for _, p in backbone
        ):
            raise AssertionError("Joint NWM backbone has no nonzero finite gradient")
        if scheme == "real_to_latent" and not _finite_nonzero_gradient(
            named, "motion_condition_encoder.latent_"
        ):
            raise AssertionError("Joint E_z has no nonzero finite gradient")
    if scheme in {"embedding_align", "state_conditioned_controller"} or (
        scheme == "real_to_latent" and substage == "warmup"
    ):
        teacher_grads = [
            name
            for name, parameter in named
            if name.startswith("motion_condition_encoder.latent_")
            and parameter.grad is not None
        ]
        if teacher_grads:
            raise AssertionError(f"Frozen E_z received gradients: {teacher_grads}")


def _should_check_first_step_gradients(
    *,
    check_gradients: bool,
    training_stage: str,
    scheme: str | None,
    substage: str,
    local_alignment_has_target: bool | None,
) -> bool:
    """Return whether this rank can require a nonzero adapter gradient.

    EmbeddingAlign uses a globally normalized loss so all ranks must execute
    backward/step when any rank has a target. A rank whose local valid mask is
    empty still participates with a zero graph, but it cannot be required to
    demonstrate a locally nonzero E_real gradient.
    """
    if not check_gradients or training_stage != "real_finetune":
        return False
    if scheme in {"embedding_align", "state_conditioned_controller"} and substage == "warmup":
        if local_alignment_has_target is None:
            raise AssertionError(
                "Latent-teacher warm-up gradient validation needs the local "
                "teacher-valid mask"
            )
        return bool(local_alignment_has_target)
    return True


def _defer_random_init_gradient_check(config, substage, completed_substage_steps):
    """Allow fresh zero-initialized CDiT layers to open before checking E_real.

    On the first joint backward only the zero-initialized output projection can
    receive a gradient.  The second backward opens the zero-initialized adaLN
    paths, and the third is the first one on which E_real can be required to
    have a nonzero gradient.
    """
    random_init = bool(config.get("finetune", {}).get("random_init", False))
    return (
        random_init
        and substage == "joint"
        and int(completed_substage_steps) < 2
    )


def _validate_resume_metadata(metadata, training_stage, action_mode, scheme):
    recorded_stage = str(metadata.get("training_stage", ""))
    if recorded_stage != training_stage:
        raise ValueError(
            "Resume checkpoint training_stage mismatch: "
            f"checkpoint={recorded_stage!r}, current={training_stage!r}"
        )
    recorded_mode = str(metadata.get("action_mode", ""))
    if recorded_mode != action_mode:
        raise ValueError(
            "Resume checkpoint action_mode mismatch: "
            f"checkpoint={recorded_mode!r}, current={action_mode!r}"
        )
    if training_stage == "real_finetune":
        recorded_scheme = normalize_finetune_scheme(
            metadata.get("finetune_scheme", "reset")
        )
        if recorded_scheme != scheme:
            raise ValueError(
                "Resume checkpoint finetune_scheme mismatch: "
                f"checkpoint={recorded_scheme!r}, current={scheme!r}"
            )


def _set_dataset_epoch(dataset, epoch):
    """Propagate epochs through ConcatDataset without changing legacy loaders."""
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    for child in getattr(dataset, "datasets", ()):
        _set_dataset_epoch(child, epoch)


def two_stage_train_step(
    *,
    model,
    ema,
    diffusion,
    tokenizer,
    batch,
    device,
    optimizer,
    scheduler,
    scaler,
    config,
    training_stage,
    scheme,
    substage,
    check_gradients=False,
):
    """Run one step and return scalar logs plus proxy accounting tensors."""
    raw_model = _unwrap_model(model)
    bfloat = bool(config.get("bfloat16", False))
    logs: dict[str, Any] = {}
    proxy_accounting = None
    local_alignment_has_target: bool | None = None
    optimizer.zero_grad(set_to_none=True)

    teacher_warmup = bool(
        training_stage == "real_finetune"
        and scheme in {"embedding_align", "state_conditioned_controller"}
        and substage == "warmup"
    )
    if teacher_warmup:
        rel_shape = torch.as_tensor(batch["k"]).shape
        if len(rel_shape) != 2:
            raise ValueError("Warmup temporal tensor must be [B,G]")
        batch_size, num_goals = rel_shape
        action, _ = _real_action(batch, raw_model, batch_size, num_goals, device)
        teacher, valid = _alignment_targets(
            batch, raw_model, batch_size, num_goals, device
        )
        frame_offset = _flatten_optional(
            batch,
            "frame_offset",
            batch_size,
            num_goals,
            device,
            torch.int64,
        )
        if frame_offset is None:
            raise ValueError("Latent-teacher warm-up requires raw frame_offset")
        valid = valid & (
            frame_offset.abs() <= LOCAL_PROXY_MAX_ABS_FRAME_OFFSET
        )
        local_alignment_has_target = bool(valid.any().item())
        if scheme == "embedding_align":
            auxiliary = model(
                None,
                None,
                action=action,
                teacher_latent=teacher,
                teacher_valid=valid,
                alignment_only=True,
            )
            loss_terms = alignment_loss(
                auxiliary["real_embedding"],
                auxiliary["target_embedding"],
                auxiliary["alignment_valid"],
                cosine_weight=float(config.finetune.get("align_cos_weight", 1.0)),
                l1_weight=float(config.finetune.get("align_l1_weight", 0.0)),
                distributed_mean=True,
            )
            loss = loss_terms["alignment"]
        else:
            with torch.amp.autocast(
                device_type="cuda",
                enabled=bfloat and device.type == "cuda",
                dtype=torch.bfloat16,
            ):
                (
                    _,
                    context,
                    _,
                    encoded_frame_offset,
                    encoded_batch_size,
                    encoded_num_goals,
                ) = _encode_batch(tokenizer, batch, config, device)
                if (encoded_batch_size, encoded_num_goals) != (
                    batch_size,
                    num_goals,
                ):
                    raise ValueError("Controller warm-up batch layout changed")
                if not torch.equal(encoded_frame_offset, frame_offset):
                    raise ValueError("Controller warm-up frame offsets are misaligned")
                auxiliary = model(
                    None,
                    None,
                    x_cond=context,
                    action=action,
                    state_controller_only=True,
                )
                loss_terms = latent_action_l2_loss(
                    auxiliary["predicted_latent"],
                    teacher,
                    valid,
                    distributed_mean=True,
                )
                loss = loss_terms["latent_l2"]
            logs.update(predicted_latent_metrics(auxiliary["predicted_latent"]))
        logs.update({key: value.detach() for key, value in loss_terms.items()})
        logs["diffusion"] = loss.detach() * 0
    else:
        with torch.amp.autocast(
            device_type="cuda",
            enabled=bfloat and device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            target, context, rel_t, frame_offset, batch_size, num_goals = _encode_batch(
                tokenizer, batch, config, device
            )
            timesteps = torch.randint(
                0, diffusion.num_timesteps, (target.shape[0],), device=device
            )
            kwargs = {"x_cond": context, "rel_t": rel_t}
            if training_stage == "proxy_pretrain":
                eligible = (
                    frame_offset.abs() <= LOCAL_PROXY_MAX_ABS_FRAME_OFFSET
                )
                if str(config.action_mode) == "none":
                    kwargs["conditioning_mode"] = "none"
                    valid = torch.zeros_like(frame_offset, dtype=torch.bool)
                    action = None
                else:
                    action, valid = _proxy_action(
                        batch,
                        raw_model,
                        str(config.action_mode),
                        batch_size,
                        num_goals,
                        device,
                    )
                    # The raw integer offset is authoritative; a collated
                    # placeholder can never make a long-range proxy valid.
                    valid = valid & eligible
                    kwargs.update(
                        action=action,
                        action_valid=valid,
                        conditioning_mode=str(config.action_mode),
                    )
                found = _flatten_optional(
                    batch, "proxy_found", batch_size, num_goals, device, torch.bool
                )
                invalid = _flatten_optional(
                    batch, "proxy_invalid", batch_size, num_goals, device, torch.bool
                )
                if found is None:
                    found = valid.clone()
                if invalid is None:
                    invalid = eligible & found & ~valid
                proxy_accounting = {
                    "frame_offset": frame_offset.detach(),
                    "eligible": eligible.detach(),
                    "found": found.detach(),
                    "invalid": invalid.detach(),
                    "used": valid.detach(),
                }
            else:
                action, action_valid = _real_action(
                    batch, raw_model, batch_size, num_goals, device
                )
                kwargs.update(action=action, action_valid=action_valid)
                if scheme == "embedding_align":
                    teacher, latent_valid = _alignment_targets(
                        batch, raw_model, batch_size, num_goals, device
                    )
                    latent_valid = latent_valid & (
                        frame_offset.abs() <= LOCAL_PROXY_MAX_ABS_FRAME_OFFSET
                    )
                    kwargs.update(
                        teacher_latent=teacher,
                        teacher_valid=latent_valid,
                        return_aux=True,
                    )
                elif scheme == "real_to_latent":
                    kwargs["return_aux"] = True
                elif scheme == "state_conditioned_controller":
                    teacher, latent_valid = _alignment_targets(
                        batch, raw_model, batch_size, num_goals, device
                    )
                    latent_valid = latent_valid & (
                        frame_offset.abs() <= LOCAL_PROXY_MAX_ABS_FRAME_OFFSET
                    )
                    kwargs["return_aux"] = True

            losses = diffusion.training_losses(model, target, timesteps, kwargs)
            diffusion_loss = losses["loss"].mean()
            loss = diffusion_loss
            logs["diffusion"] = diffusion_loss.detach()
            auxiliary = losses.get("model_aux", {})
            if scheme == "embedding_align":
                loss_terms = alignment_loss(
                    auxiliary["real_embedding"],
                    auxiliary["target_embedding"],
                    auxiliary["alignment_valid"],
                    cosine_weight=float(config.finetune.get("align_cos_weight", 1.0)),
                    l1_weight=float(config.finetune.get("align_l1_weight", 0.0)),
                    distributed_mean=True,
                )
                loss = loss + loss_terms["alignment"]
                logs.update({key: value.detach() for key, value in loss_terms.items()})
            if scheme == "real_to_latent" and "predicted_latent" in auxiliary:
                logs.update(predicted_latent_metrics(auxiliary["predicted_latent"]))
            if scheme == "state_conditioned_controller":
                loss_terms = latent_action_l2_loss(
                    auxiliary["predicted_latent"],
                    teacher,
                    latent_valid,
                    distributed_mean=True,
                )
                latent_weight = float(config.finetune.get("latent_l2_weight", 1.0))
                weighted_latent = latent_weight * loss_terms["latent_l2"]
                loss = loss + weighted_latent
                logs.update(
                    {key: value.detach() for key, value in loss_terms.items()}
                )
                logs["latent_l2_weighted"] = weighted_latent.detach()
                logs.update(predicted_latent_metrics(auxiliary["predicted_latent"]))

    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite two-stage loss: {loss.detach().item()}")
    should_check_gradients = _should_check_first_step_gradients(
        check_gradients=check_gradients,
        training_stage=training_stage,
        scheme=scheme,
        substage=substage,
        local_alignment_has_target=local_alignment_has_target,
    )
    if scaler is None:
        loss.backward()
        if should_check_gradients:
            _assert_first_step_gradients(raw_model, scheme, substage)
            logs["gradient_check_performed"] = True
        if float(config.training.get("grad_clip_val", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad],
                float(config.training.grad_clip_val),
            )
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if should_check_gradients:
            _assert_first_step_gradients(raw_model, scheme, substage)
            logs["gradient_check_performed"] = True
        if float(config.training.get("grad_clip_val", 0.0)) > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad],
                float(config.training.grad_clip_val),
            )
        scaler.step(optimizer)
        scaler.update()
    if scheduler is not None:
        scheduler.step()
    update_ema(ema, raw_model, decay=float(config.training.get("ema_decay", 0.9999)))
    logs["loss"] = loss.detach()
    return logs, proxy_accounting


def _load_ema_and_training_state(checkpoint_path, ema, scheduler, scaler):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "ema" in checkpoint:
        ema.load_state_dict(
            {key.replace("_orig_mod.", ""): value for key, value in checkpoint["ema"].items()},
            strict=True,
        )
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def _prepare_loader(
    config,
    rank,
    substage,
    *,
    include_eval=False,
):
    stage = str(config.training_stage)
    if stage == "proxy_pretrain":
        from data_utils import (
            prepare_proxy_pretrain_dataset,
            prepare_proxy_pretrain_validation_dataset,
        )

        dataset = prepare_proxy_pretrain_dataset(config)
        eval_dataset = (
            prepare_proxy_pretrain_validation_dataset(config)
            if include_eval
            else None
        )
    else:
        dataset, eval_dataset = prepare_datasets(
            config,
            finetune_substage=substage,
            include_test=bool(include_eval),
        )
    loader, sampler = create_dataloader(dataset, config, rank, is_train=True)
    eval_loader = None
    if eval_dataset is not None:
        eval_loader, _ = create_dataloader(
            eval_dataset, config, rank, is_train=False
        )
    return loader, sampler, dataset, eval_loader, eval_dataset


def run_two_stage_training(config, device, rank, local_gpu, experiment_dir, log=None):
    """Run proxy pretraining or the resumable warmup→joint stage-2 flow."""
    log = log or logger
    training_stage, action_mode, scheme = validate_two_stage_config(config)
    if training_stage == "legacy":
        raise ValueError("run_two_stage_training must not be used for legacy configs")
    if scheme == "state_conditioned_controller":
        _define_controller_wandb_metrics(config, rank)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    tokenizer = _setup_training_tokenizer(config, device, log)
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    raw_model = setup_model(config, device)

    resume_path = config.training.get("from_checkpoint", None)
    resume_metadata = None
    OmegaConf.update(config, "training.best_metrics", {}, force_add=True)
    resume_info = None
    if resume_path:
        resume_path = _resolve_user_path(resume_path)
        resume_checkpoint = torch.load(
            str(resume_path), map_location="cpu", weights_only=False
        )
        OmegaConf.update(config, "training.best_metrics", resume_checkpoint.get("best_metrics", {}), force_add=True)
        resume_metadata = extract_checkpoint_metadata(resume_checkpoint)
        if resume_metadata.get("initialization") is not None:
            OmegaConf.update(config, "finetune.initialization", resume_metadata["initialization"], force_add=True)
        _validate_resume_metadata(
            resume_metadata, training_stage, action_mode, scheme
        )
        # Load weights before constructing EMA or selecting substages. This is
        # essential when the recorded phase has already reached a configured
        # zero/finished target and its loop is skipped before joint training.
        resume_info = load_two_stage_resume(
            raw_model,
            None,
            str(resume_path),
            expected_substage=None,
            config=config,
        )
    elif training_stage == "real_finetune":
        stage1_path = config.finetune.get("stage1_checkpoint", None)
        init_path = config.finetune.get("init_checkpoint", None)
        random_init = bool(config.finetune.get("random_init", False))
        if init_path:
            summary = load_finetune_weights(raw_model, _resolve_user_path(init_path), config)
            OmegaConf.update(config, "finetune.initialization", summary, force_add=True)
            log.info("Complete EMA initialization (fresh training state): %s", summary)
        elif random_init:
            # setup_model constructs CDiT and every motion adapter with their
            # standard fresh initialization. Deliberately do not touch any NWM
            # checkpoint here; validate_two_stage_config also rejects a path.
            log.info(
                "Strict no-pretrain initialization: fresh CDiT and real-action "
                "adapter weights (seed=%s); no NWM checkpoint loaded",
                config.get("seed", 0),
            )
        elif not stage1_path:
            raise ValueError("real_finetune requires finetune.stage1_checkpoint")
        else:
            stage1_path = _resolve_user_path(stage1_path)
            summary = load_stage1_weights(raw_model, stage1_path, config)
            log.info(
                "Stage-1 initialization loaded modules=%s skipped=%s",
                summary["loaded_modules"],
                summary["skipped_modules"],
            )

    ema = deepcopy(raw_model).to(device).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    diffusion = setup_diffusion(config, for_eval=False, device=device)
    scaler = (
        torch.amp.GradScaler("cuda")
        if bool(config.get("bfloat16", False))
        else None
    )
    if resume_path:
        _load_ema_and_training_state(resume_path, ema, None, scaler)

    eval_interval = int(config.get("eval_every", 0))
    controller_eval_interval = int(
        config.get("finetune", {}).get("controller_eval_every", 0)
    )
    eval_at_first_step = bool(config.get("eval_at_first_step", True))
    log_cuda_memory = bool(config.get("log_cuda_memory", False))
    eval_offload_models = bool(config.get("eval_offload_models", False))
    proxy_validation_enabled = bool(
        training_stage == "proxy_pretrain"
        and config.dataset.get("validation", {}).get("enabled", False)
    )
    eval_enabled = bool(
        (training_stage == "real_finetune" or proxy_validation_enabled)
        and (
            eval_interval > 0
            or (
                scheme == "state_conditioned_controller"
                and controller_eval_interval > 0
            )
            or eval_at_first_step
        )
    )
    eval_tokenizer = None
    eval_tokenizer_offloaded = False
    if eval_enabled:
        # Cached posteriors are a training-only optimization. The legacy
        # evaluator needs RGB inputs and a real VAE to encode conditions and
        # decode predictions, while training keeps its lightweight tokenizer.
        if bool(
            config.dataset.get("precomputed_latents", {}).get("enabled", False)
        ):
            training_rng_state = capture_rng_state()
            try:
                eval_tokenizer = setup_tokenizer(config, device)
            finally:
                restore_rng_state(training_rng_state)
        else:
            eval_tokenizer = tokenizer
        eval_tokenizer.eval()
        for parameter in eval_tokenizer.parameters():
            parameter.requires_grad_(False)
        if eval_offload_models and eval_tokenizer is not tokenizer:
            eval_tokenizer.to("cpu")
            eval_tokenizer_offloaded = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        log.info(
            "Evaluation enabled: stage=%s first_step=%s interval=%d "
            "controller_interval=%d model=EMA offload_models=%s",
            training_stage,
            eval_at_first_step,
            eval_interval,
            controller_eval_interval,
            eval_offload_models,
        )

    completed_warmup = int((resume_metadata or {}).get("completed_warmup_steps", 0))
    completed_joint = int((resume_metadata or {}).get("completed_joint_steps", 0))
    if training_stage == "proxy_pretrain":
        substages = [("pretrain", int(config.get("max_train_steps", 0)))]
    else:
        warmup_steps = int(config.finetune.get("warmup_steps", 0))
        joint_steps = int(config.finetune.get("joint_steps", 0))
        recorded = str((resume_metadata or {}).get("finetune_substage", "warmup"))
        if recorded in {"joint", "transition"}:
            substages = [("joint", joint_steps)]
        else:
            substages = [("warmup", warmup_steps), ("joint", joint_steps)]

    if training_stage == "proxy_pretrain" and resume_path:
        global_steps = int(resume_checkpoint.get("train_steps", 0))
    else:
        global_steps = int(
            (resume_metadata or {}).get("completed_warmup_steps", 0)
        ) + int((resume_metadata or {}).get("completed_joint_steps", 0))
    for substage, target_steps in substages:
        target_samples_per_rank = _phase_target_samples_per_rank(
            config, training_stage, substage
        )
        if (
            target_samples_per_rank is not None
            and target_samples_per_rank > 0
            and target_steps <= 0
        ):
            raise ValueError(
                "A positive sample target requires a positive optimizer-step cap"
            )
        if target_steps <= 0 or target_samples_per_rank == 0:
            if training_stage == "real_finetune" and substage == "warmup":
                # A zero-length/already-disabled warm-up still has to carry the
                # saved per-rank RNG into joint setup. Joint restores it only
                # after iterator creation, matching transition resume.
                runtime_state = dict((resume_info or {}).get("runtime_state") or {})
                runtime_state.setdefault("rng_state", capture_rng_state())
                runtime_state.setdefault(
                    "proxy_metrics", _proxy_metrics_state(ProxyMetrics())
                )
                runtime_state.setdefault("valid_alignment_batches", 0)
                resume_info = {
                    "resume_substage": "joint",
                    "epoch": 0,
                    "batch_in_epoch": 0,
                    "runtime_state": runtime_state,
                    "data_fingerprint": build_data_resume_fingerprint(
                        config, "joint"
                    ),
                }
            continue
        configured_substage = substage if training_stage == "real_finetune" else None
        configure_trainable_parameters(
            raw_model,
            training_stage=training_stage,
            action_mode=action_mode,
            finetune_scheme=scheme,
            finetune_substage=configured_substage,
        )
        optimizer = _make_optimizer(
            config,
            raw_model,
            training_stage=training_stage,
            scheme=scheme,
            substage=configured_substage,
        )
        scheduler = _make_scheduler(
            config,
            optimizer,
            training_stage=training_stage,
            scheme=scheme,
            substage=configured_substage,
        )
        current = (
            completed_warmup
            if substage == "warmup"
            else completed_joint
            if substage == "joint"
            else global_steps
        )
        if resume_path:
            recorded = str(resume_metadata.get("finetune_substage"))
            expected = substage
            if training_stage == "proxy_pretrain":
                expected = "pretrain"
            if recorded == substage or (recorded == "transition" and substage == "joint") or (
                recorded == "pretrain" and expected == "pretrain"
            ):
                resume = load_two_stage_resume(
                    raw_model,
                    optimizer,
                    str(resume_path),
                    expected_substage=expected,
                    config=config,
                )
                resume_info = resume
                _load_ema_and_training_state(resume_path, ema, scheduler, scaler)
                log.info(
                    "Resumed %s checkpoint: %s",
                    substage,
                    {key: value for key, value in resume.items() if key != "runtime_state"},
                )
                resume_path = None

        _log_trainability(raw_model, optimizer, training_stage, substage, log)
        compiled = torch.compile(raw_model) if bool(config.get("torch_compile", False)) else raw_model
        model = DDP(compiled, device_ids=[local_gpu], find_unused_parameters=False)
        model.train()
        loader, sampler, dataset, eval_loader, eval_dataset = _prepare_loader(
            config,
            rank,
            substage,
            include_eval=(
                training_stage == "real_finetune" or proxy_validation_enabled
            ),
        )
        if rank == 0 and getattr(dataset, "recipe_summary", None) is not None:
            recipe_summary = dataset.recipe_summary
            log.info("Frozen NavAnywhere sampling recipe: %s", recipe_summary)
            if _wandb_active(config, rank):
                wandb.config.update(
                    {"navanywhere_sampling_recipe": recipe_summary},
                    allow_val_change=True,
                )
                for key, value in recipe_summary.items():
                    if key != "totals":
                        wandb.run.summary[f"sampling_recipe/{key}"] = value
                for key, value in recipe_summary["totals"].items():
                    wandb.run.summary[f"sampling_recipe/totals/{key}"] = value
        if rank == 0 and getattr(eval_dataset, "recipe_summary", None) is not None:
            val_recipe_summary = eval_dataset.recipe_summary
            log.info(
                "Frozen NavAnywhere validation recipe: %s", val_recipe_summary
            )
            if _wandb_active(config, rank):
                wandb.config.update(
                    {"navanywhere_validation_recipe": val_recipe_summary},
                    allow_val_change=True,
                )
        if len(loader) == 0:
            raise ValueError(
                "Training DataLoader has no full batch; lower training.batch_size "
                "or provide more samples"
            )
        if log_cuda_memory:
            _reset_cuda_peak_memory(device)
        phase_resume = (
            resume_info
            if resume_info is not None
            and str(resume_info.get("resume_substage")) == substage
            else None
        )
        runtime_state = dict((phase_resume or {}).get("runtime_state") or {})
        batch_in_epoch = int((phase_resume or {}).get("batch_in_epoch", 0))
        if batch_in_epoch > len(loader):
            raise ValueError(
                "Checkpoint batch cursor exceeds DataLoader length: "
                f"cursor={batch_in_epoch}, batches={len(loader)}"
            )
        phase_data_fingerprint = build_data_resume_fingerprint(
            config,
            substage,
            dataset_length=len(dataset),
            loader_length=len(loader),
        )
        if phase_resume is not None:
            validate_data_resume_fingerprint(
                phase_resume.get("data_fingerprint"),
                config,
                substage,
                dataset_length=len(dataset),
                loader_length=len(loader),
                required=batch_in_epoch > 0,
            )
        proxy_metrics = _restore_proxy_metrics(runtime_state.get("proxy_metrics"))
        epoch = int((phase_resume or {}).get("epoch", 0))
        resume_rng_state = runtime_state.get("rng_state")
        saved_samples = runtime_state.get("completed_samples_per_rank", None)
        if (
            target_samples_per_rank is not None
            and current > 0
            and saved_samples is None
        ):
            raise ValueError(
                "A sample-budgeted resume checkpoint must record "
                "completed_samples_per_rank"
            )
        completed_samples_per_rank = int(saved_samples or 0)
        if completed_samples_per_rank < 0:
            raise ValueError("Saved completed_samples_per_rank must be non-negative")
        if (
            target_samples_per_rank is not None
            and completed_samples_per_rank > target_samples_per_rank
        ):
            raise ValueError(
                "Checkpoint completed_samples_per_rank exceeds configured target"
            )
        resumed_valid_alignment_batches = int(
            runtime_state.get("valid_alignment_batches", 0)
        )
        if resumed_valid_alignment_batches < 0:
            raise ValueError("Saved valid_alignment_batches must be non-negative")
        valid_alignment_batches = (
            resumed_valid_alignment_batches if batch_in_epoch else 0
        )
        # A completed phase does not enter the iterator loop, so restore its
        # model-side RNG here. The transition runtime state then carries this
        # exact point into joint and restores it after joint iterator setup.
        if target_samples_per_rank is None:
            resume_rng_state = _restore_completed_phase_rng(
                current, target_steps, resume_rng_state
            )
        elif _phase_complete(
            current,
            target_steps,
            completed_samples_per_rank,
            target_samples_per_rank,
        ) and resume_rng_state is not None:
            restore_rng_state(resume_rng_state)
            resume_rng_state = None
        log.info(
            "Phase budget: substage=%s steps=%d/%d samples_per_rank=%d/%s",
            substage,
            current,
            target_steps,
            completed_samples_per_rank,
            (
                "step-controlled"
                if target_samples_per_rank is None
                else str(target_samples_per_rank)
            ),
        )
        checked_gradients = False
        # Match the legacy nwm-real logging contract: each published train
        # metric is an average over all optimizer steps since the previous
        # log point, rather than the loss of one sampled minibatch.
        running_log_sums: dict[str, torch.Tensor] = {}
        running_log_counts: dict[str, int] = {}
        while not _phase_complete(
            current,
            target_steps,
            completed_samples_per_rank,
            target_samples_per_rank,
        ):
            if current >= target_steps:
                raise RuntimeError(
                    "Optimizer-step cap was reached before the exact sample "
                    "budget; increase the configured phase step target"
                )
            sampler.set_epoch(epoch)
            _set_dataset_epoch(dataset, epoch)
            _set_loader_epoch_seed(loader, int(config.seed), rank, epoch)
            batch_iterator = iter(loader)
            if batch_in_epoch:
                for _ in range(batch_in_epoch):
                    try:
                        next(batch_iterator)
                    except StopIteration as error:
                        raise ValueError(
                            "Checkpoint batch cursor cannot be replayed by the "
                            "current DataLoader"
                        ) from error
                log.info(
                    "Replayed %d batches to resume epoch %d exactly",
                    batch_in_epoch,
                    epoch,
                )
            # Iterator creation and cursor replay may consume RNG (notably when
            # num_workers=0). Restore model-side RNG only after that replay.
            if resume_rng_state is not None:
                restore_rng_state(resume_rng_state)
                resume_rng_state = None
            for batch in batch_iterator:
                batch_in_epoch += 1
                batch, local_batch_size = _budget_training_batch(
                    batch,
                    completed_samples_per_rank,
                    target_samples_per_rank,
                )
                if (
                    training_stage == "real_finetune"
                    and scheme
                    in {"embedding_align", "state_conditioned_controller"}
                    and substage == "warmup"
                ):
                    raw_latent_valid = torch.as_tensor(
                        batch["latent_valid"], dtype=torch.bool
                    )
                    raw_frame_offset = torch.as_tensor(
                        batch["frame_offset"], dtype=torch.int64
                    )
                    local_valid = torch.tensor(
                        int(
                            bool(
                                (
                                    raw_latent_valid
                                    & (
                                        raw_frame_offset.abs()
                                        <= LOCAL_PROXY_MAX_ABS_FRAME_OFFSET
                                    )
                                ).any()
                            )
                        ),
                        device=device,
                    )
                    dist.all_reduce(local_valid, op=dist.ReduceOp.SUM)
                    if int(local_valid.item()) == 0:
                        # All ranks skip together, so DDP collectives remain
                        # aligned. If another rank has a target, every rank runs
                        # the step and its zero graph participates normally.
                        continue
                    valid_alignment_batches += 1
                logs, accounting = two_stage_train_step(
                    model=model,
                    ema=ema,
                    diffusion=diffusion,
                    tokenizer=tokenizer,
                    batch=batch,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    training_stage=training_stage,
                    scheme=scheme,
                    substage=substage,
                    check_gradients=(
                        not checked_gradients
                        and not _defer_random_init_gradient_check(
                            config, substage, current
                        )
                    ),
                )
                checked_gradients = checked_gradients or bool(
                    logs.pop("gradient_check_performed", False)
                )
                current += 1
                global_steps += 1
                completed_samples_per_rank += local_batch_size
                if substage == "warmup":
                    completed_warmup = current
                elif substage == "joint":
                    completed_joint = current
                if accounting is not None:
                    proxy_metrics.update(**accounting)
                for key, value in logs.items():
                    detached = torch.as_tensor(value, device=device).detach().float()
                    if key in running_log_sums:
                        running_log_sums[key] += detached
                    else:
                        running_log_sums[key] = detached.clone()
                    running_log_counts[key] = running_log_counts.get(key, 0) + 1
                phase_name = _controller_phase_name(scheme, substage)
                log_step = current if phase_name is not None else global_steps
                log_due = log_step % int(config.get("log_every", 100)) == 0
                evaluation_due = bool(
                    eval_enabled
                    and _phase_evaluation_due(
                        training_stage=training_stage,
                        scheme=scheme,
                        substage=substage,
                        current_steps=current,
                        global_steps=global_steps,
                        config=config,
                    )
                )
                if log_due:
                    scalar_logs = {}
                    for key, value in running_log_sums.items():
                        reduced = value / running_log_counts[key]
                        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                        reduced /= dist.get_world_size()
                        scalar_logs[key] = reduced.item()
                    log.info(
                        "stage=%s substage=%s step=%d metrics=%s",
                        training_stage,
                        substage,
                        global_steps,
                        scalar_logs,
                    )
                    if rank == 0:
                        wandb_values = {
                            f"train/{key}": value
                            for key, value in scalar_logs.items()
                        }
                        wandb_values["train/samples_per_rank"] = (
                            completed_samples_per_rank
                        )
                        wandb_values.update(
                            {
                                f"train/lr_group_{group_index}": float(
                                    group["lr"]
                                )
                                for group_index, group in enumerate(
                                    optimizer.param_groups
                                )
                            }
                        )
                        if config.get("finetune", {}).get("dataset_protocol", "paper") == "go2":
                            wandb_values["train/joint_phase"] = int(substage == "joint")
                        if phase_name is not None:
                            wandb_values[f"{phase_name}/step"] = current
                            wandb_values.update(
                                {
                                    f"{phase_name}/train/{key}": value
                                    for key, value in scalar_logs.items()
                                }
                            )
                            wandb_values.update(
                                {
                                    f"{phase_name}/train/lr_group_{group_index}": float(
                                        group["lr"]
                                    )
                                    for group_index, group in enumerate(
                                        optimizer.param_groups
                                    )
                                }
                            )
                        log_metrics(experiment_dir, {"step": global_steps, "phase": substage, **wandb_values})
                        _wandb_log(
                            config, rank, wandb_values, global_steps
                        )
                    running_log_sums.clear()
                    running_log_counts.clear()
                if log_cuda_memory and (log_due or evaluation_due):
                    train_memory = _distributed_cuda_memory_gib(device)
                    if train_memory is not None and rank == 0:
                        log.info(
                            "CUDA training memory (max across ranks): "
                            "current_allocated=%.2f GiB current_reserved=%.2f GiB "
                            "peak_allocated=%.2f GiB peak_reserved=%.2f GiB",
                            train_memory["current_allocated_gib"],
                            train_memory["current_reserved_gib"],
                            train_memory["peak_allocated_gib"],
                            train_memory["peak_reserved_gib"],
                        )
                        _wandb_log(
                            config,
                            rank,
                            {
                                f"memory/train_{key}": value
                                for key, value in train_memory.items()
                            },
                            global_steps,
                        )
                    _reset_cuda_peak_memory(device)
                if _periodic_checkpoint_due(
                    training_stage=training_stage,
                    current_steps=current,
                    global_steps=global_steps,
                    target_steps=target_steps,
                    interval=int(config.get("ckpt_every", 10000)),
                ):
                    runtime_states = _gather_runtime_states(
                        proxy_metrics,
                        valid_alignment_batches,
                        completed_samples_per_rank,
                    )
                    if rank == 0:
                        _save_training_checkpoint(
                            checkpoint_dir,
                            training_stage=training_stage,
                            current_steps=current,
                            model=model,
                            ema=ema,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            config=config,
                            substage=substage,
                            completed_warmup_steps=completed_warmup,
                            completed_joint_steps=completed_joint,
                            train_steps=global_steps,
                            epoch=epoch,
                            include_optimizer=True,
                            batch_in_epoch=batch_in_epoch,
                            runtime_states=runtime_states,
                            data_fingerprint=phase_data_fingerprint,
                        )
                    dist.barrier()
                if eval_enabled and _phase_complete(current, target_steps, completed_samples_per_rank, target_samples_per_rank):
                    evaluation_due = True
                if evaluation_due:
                    evaluation_jobs = []
                    if eval_loader is not None:
                        evaluation_jobs.append(
                            (
                                (
                                    "navanywhere"
                                    if training_stage == "proxy_pretrain"
                                    else "nwm_real"
                                ),
                                eval_loader,
                                int(config.seed),
                            )
                        )
                    if not evaluation_jobs or eval_tokenizer is None:
                        raise RuntimeError(
                            "Evaluation is enabled but no pixel eval loader or "
                            "tokenizer was created"
                        )
                    eval_start_time = time()
                    # This step's gradients have already been consumed. Free
                    # them before loading the infrequently used evaluation
                    # VAE and DreamSim ensemble.
                    optimizer.zero_grad(set_to_none=True)
                    logs.clear()
                    accounting = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _reset_cuda_peak_memory(device)
                    if eval_tokenizer_offloaded:
                        eval_tokenizer.to(device)
                    # Sampling consumes Python/NumPy/torch RNG. Restore every
                    # stream afterward so eval cannot alter training or exact
                    # checkpoint-resume behavior.
                    training_rng_state = capture_rng_state()
                    eval_scores = {}
                    eval_times = {}
                    try:
                        for (
                            eval_name,
                            current_eval_loader,
                            eval_seed,
                        ) in evaluation_jobs:
                            current_eval_start = time()
                            save_dir = (
                                os.path.join(
                                    experiment_dir,
                                    "viz",
                                    eval_name,
                                    str(global_steps),
                                )
                                if training_stage == "proxy_pretrain"
                                else os.path.join(
                                    experiment_dir,
                                    "viz",
                                    phase_name,
                                    str(current),
                                )
                                if phase_name is not None
                                else os.path.join(
                                    experiment_dir, "viz", str(global_steps)
                                )
                            )
                            eval_scores[eval_name] = evaluate(
                                ema,
                                eval_tokenizer,
                                diffusion,
                                current_eval_loader,
                                rank,
                                int(config.model.generator.input_size),
                                device,
                                save_dir,
                                eval_seed,
                                bool(config.get("bfloat16", False)),
                                int(config.dataset.context_size),
                                get_unnormalize(
                                    config.dataset.mean, config.dataset.std
                                ),
                                offload_model=eval_offload_models,
                                num_batches=config.training.get("eval_num_batches", 1),
                            )
                            eval_times[eval_name] = time() - current_eval_start
                    finally:
                        restore_rng_state(training_rng_state)
                        if eval_tokenizer_offloaded:
                            eval_tokenizer.to("cpu")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    dist.barrier()
                    eval_time = time() - eval_start_time
                    eval_memory = (
                        _distributed_cuda_memory_gib(device)
                        if log_cuda_memory
                        else None
                    )
                    for eval_name, sim_score in eval_scores.items():
                        log.info(
                            "(step=%07d) %s Perceptual Loss: %.4f, "
                            "Eval Time: %.2f",
                            global_steps,
                            eval_name,
                            float(sim_score),
                            eval_times[eval_name],
                        )
                    if eval_memory is not None and rank == 0:
                        log.info(
                            "CUDA evaluation memory (max across ranks): "
                            "peak_allocated=%.2f GiB peak_reserved=%.2f GiB",
                            eval_memory["peak_allocated_gib"],
                            eval_memory["peak_reserved_gib"],
                        )
                    if rank == 0:
                        eval_values = {
                            "eval/eval_time": eval_time,
                            "eval/step": global_steps,
                            "epoch": epoch,
                        }
                        if training_stage == "proxy_pretrain":
                            eval_values.update(
                                {
                                    f"eval/{name}_perceptual_loss": score
                                    for name, score in eval_scores.items()
                                }
                            )
                            eval_values.update(
                                {
                                    f"eval/{name}_time": eval_times[name]
                                    for name in eval_scores
                                }
                            )
                            # Backward-compatible primary curve.
                            if "navanywhere" in eval_scores:
                                eval_values["eval/perceptual_loss"] = (
                                    eval_scores["navanywhere"]
                                )
                        else:
                            eval_values["eval/perceptual_loss"] = eval_scores[
                                "nwm_real"
                            ]
                            if phase_name is not None:
                                eval_values[f"{phase_name}/step"] = current
                                eval_values[
                                    f"{phase_name}/eval/perceptual_loss"
                                ] = eval_scores["nwm_real"]
                                eval_values[
                                    f"{phase_name}/eval/eval_time"
                                ] = eval_time
                        if eval_memory is not None:
                            eval_values.update(
                                {
                                    f"memory/eval_{key}": value
                                    for key, value in eval_memory.items()
                                }
                            )
                        _wandb_log(
                            config,
                            rank,
                            eval_values,
                            global_steps,
                        )
                    score = float(next(iter(eval_scores.values())))
                    metric_key = substage if training_stage == "real_finetune" else "pretrain"
                    previous = float(config.training.best_metrics.get(metric_key, float("inf")))
                    if score < previous:
                        OmegaConf.update(config, f"training.best_metrics.{metric_key}", score, force_add=True)
                        runtime_states = _gather_runtime_states(proxy_metrics, valid_alignment_batches, completed_samples_per_rank)
                        if rank == 0:
                            best_path = os.path.join(checkpoint_dir, f"best_{metric_key}.pth.tar")
                            save_two_stage_checkpoint(
                                best_path, model=model, ema=ema, optimizer=optimizer, scheduler=scheduler,
                                scaler=scaler, config=config, substage=substage,
                                completed_warmup_steps=completed_warmup, completed_joint_steps=completed_joint,
                                train_steps=global_steps, epoch=epoch, include_optimizer=True,
                                batch_in_epoch=batch_in_epoch, runtime_states=runtime_states,
                                data_fingerprint=phase_data_fingerprint)
                            point_checkpoint(best_path, os.path.join(checkpoint_dir, "best.pth.tar"))
                        dist.barrier()
                    if rank == 0:
                        log_metrics(experiment_dir, {"step": global_steps, "phase": metric_key,
                                    "eval/perceptual_loss": score, "best/perceptual_loss": float(config.training.best_metrics[metric_key])})
                    _reset_cuda_peak_memory(device)
                if _phase_complete(
                    current,
                    target_steps,
                    completed_samples_per_rank,
                    target_samples_per_rank,
                ):
                    break
            finished_epoch = epoch
            epoch, batch_in_epoch, epoch_exhausted = _phase_progress_after_iteration(
                epoch, batch_in_epoch, len(loader)
            )
            if epoch_exhausted:
                if (
                    training_stage == "real_finetune"
                    and scheme
                    in {"embedding_align", "state_conditioned_controller"}
                    and substage == "warmup"
                    and valid_alignment_batches == 0
                ):
                    raise RuntimeError(
                        "Latent-teacher warm-up found no valid local target in "
                        "an entire epoch; verify the cache keys or enable strict_loading"
                    )
                if training_stage == "proxy_pretrain":
                    proxy_metrics.distributed_reduce(device)
                    if rank == 0:
                        metrics = proxy_metrics.as_log_dict()
                        metrics["action_conditioning_enabled"] = action_mode != "none"
                        log.info(
                            "Epoch %d proxy metrics: %s", finished_epoch, metrics
                        )
                        _wandb_log(config, rank, metrics, global_steps)
                    proxy_metrics = ProxyMetrics()
                valid_alignment_batches = 0
            elif not _phase_complete(
                current,
                target_steps,
                completed_samples_per_rank,
                target_samples_per_rank,
            ):
                raise RuntimeError(
                    "Training DataLoader stopped before its reported length; "
                    "exact cursor accounting is impossible"
                )

        if (
            target_samples_per_rank is not None
            and completed_samples_per_rank != target_samples_per_rank
        ):
            raise RuntimeError(
                "Phase ended without reaching its exact sample budget: "
                f"{completed_samples_per_rank} != {target_samples_per_rank}"
            )
        runtime_states = _gather_runtime_states(
            proxy_metrics,
            valid_alignment_batches,
            completed_samples_per_rank,
        )
        if rank == 0:
            _save_training_checkpoint(
                checkpoint_dir,
                training_stage=training_stage,
                current_steps=current,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                substage=substage,
                completed_warmup_steps=completed_warmup,
                completed_joint_steps=completed_joint,
                train_steps=global_steps,
                epoch=epoch,
                include_optimizer=True,
                batch_in_epoch=batch_in_epoch,
                runtime_states=runtime_states,
                data_fingerprint=phase_data_fingerprint,
            )
        dist.barrier()
        del (
            loader,
            sampler,
            dataset,
            eval_loader,
            eval_dataset,
            model,
            compiled,
            optimizer,
            scheduler,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if training_stage == "real_finetune" and substage == "warmup":
            runtime_states = _gather_runtime_states(ProxyMetrics(), 0, 0)
            transition_data_fingerprint = build_data_resume_fingerprint(
                config, "joint"
            )
            if rank == 0:
                save_two_stage_checkpoint(
                    os.path.join(checkpoint_dir, "transition.pth.tar"),
                    model=raw_model,
                    ema=ema,
                    optimizer=None,
                    scheduler=None,
                    scaler=scaler,
                    config=config,
                    substage="transition",
                    completed_warmup_steps=completed_warmup,
                    completed_joint_steps=completed_joint,
                    train_steps=global_steps,
                    epoch=0,
                    include_optimizer=False,
                    batch_in_epoch=0,
                    runtime_states=runtime_states,
                    data_fingerprint=transition_data_fingerprint,
                )
            dist.barrier()
            runtime_rank = (
                dist.get_rank()
                if dist.is_available() and dist.is_initialized()
                else 0
            )
            resume_info = {
                "resume_substage": "joint",
                "epoch": 0,
                "batch_in_epoch": 0,
                "runtime_world_size": len(runtime_states),
                "runtime_state": dict(runtime_states[runtime_rank]),
                "data_fingerprint": transition_data_fingerprint,
            }

    log.info(
        "Two-stage training complete: stage=%s warmup=%d joint=%d total=%d",
        training_stage,
        completed_warmup,
        completed_joint,
        global_steps,
    )


__all__ = [
    "run_two_stage_training",
    "save_two_stage_checkpoint",
    "two_stage_train_step",
]
