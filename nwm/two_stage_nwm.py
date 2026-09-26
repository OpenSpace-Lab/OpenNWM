"""Shared mechanics for proxy pretraining and two-step real-action finetuning.

This module deliberately contains no dataset I/O and no diffusion-model
forward implementation.  It centralizes validation, alignment loss,
trainability, optimizer grouping, shape checks, and proxy accounting so the
legacy training entry point can remain unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


TRAINING_STAGES = ("legacy", "proxy_pretrain", "real_finetune")
ACTION_MODES = ("none", "real", "geometry", "idm", "latent")
FINETUNE_SCHEMES = (
    "reset",
    "embedding_align",
    "real_to_latent",
    "state_conditioned_controller",
)
FINETUNE_SUBSTAGES = ("warmup", "joint")
LOCAL_PROXY_MAX_ABS_FRAME_OFFSET = 8

_SCHEME_ALIASES = {
    "a": "reset",
    "b": "embedding_align",
    "c": "real_to_latent",
    "align": "embedding_align",
    "alignment": "embedding_align",
    "embedding_alignment": "embedding_align",
    "embedding_align": "embedding_align",
    "real-to-latent": "real_to_latent",
    "real_to_latent": "real_to_latent",
    "d": "state_conditioned_controller",
    "state_controller": "state_conditioned_controller",
    "latent_state_controller": "state_conditioned_controller",
    "state_conditioned_controller": "state_conditioned_controller",
    "reset": "reset",
}


def config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def normalize_finetune_scheme(value: Any) -> str:
    scheme = str(value or "reset").strip().lower().replace("-", "_")
    try:
        return _SCHEME_ALIASES[scheme]
    except KeyError as exc:
        raise ValueError(
            f"Unknown finetune scheme {scheme!r}; expected {FINETUNE_SCHEMES}"
        ) from exc


def validate_two_stage_config(config: Any) -> tuple[str, str, str | None]:
    """Validate only new semantics; absent fields always resolve to legacy."""
    stage = str(config_get(config, "training_stage", "legacy"))
    if stage not in TRAINING_STAGES:
        raise ValueError(
            f"training_stage must be one of {TRAINING_STAGES}, got {stage!r}"
        )
    action_mode = str(config_get(config, "action_mode", "real"))
    if action_mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}")
    if stage == "legacy":
        return stage, action_mode, None
    proxy = config_get(config, "proxy", {})
    relative_time_mode = str(
        config_get(proxy, "relative_time_mode", "always")
    ).strip().lower()
    if relative_time_mode not in {"always", "fallback"}:
        raise ValueError(
            "proxy.relative_time_mode must be 'always' or 'fallback', got "
            f"{relative_time_mode!r}"
        )
    proxy_limit = int(
        config_get(
            proxy,
            "max_abs_frame_offset",
            LOCAL_PROXY_MAX_ABS_FRAME_OFFSET,
        )
    )
    if proxy_limit != LOCAL_PROXY_MAX_ABS_FRAME_OFFSET:
        raise ValueError(
            "Two-stage proxy/teacher eligibility is fixed to abs(frame_offset)<=8; "
            f"proxy.max_abs_frame_offset={proxy_limit} is not permitted"
        )
    dataset = config_get(config, "dataset", {})
    distance = config_get(dataset, "distance", {})
    minimum = int(config_get(distance, "min_dist_cat", -64))
    maximum = int(config_get(distance, "max_dist_cat", 64))
    if (minimum, maximum) != (-64, 64):
        raise ValueError(
            "Two-stage training must retain the complete frame-offset range "
            f"[-64, 64], got [{minimum}, {maximum}]"
        )
    motion = config_get(config, "motion_condition", {})
    if not bool(config_get(motion, "enabled", False)):
        raise ValueError("Two-stage training requires motion_condition.enabled=true")
    if relative_time_mode == "fallback" and not (
        stage == "proxy_pretrain" and action_mode == "latent"
    ):
        raise ValueError(
            "proxy.relative_time_mode=fallback is reserved for latent-only "
            "pretraining and requires training_stage=proxy_pretrain with "
            "action_mode=latent"
        )
    if stage == "proxy_pretrain":
        if action_mode not in {"none", "geometry", "idm", "latent"}:
            raise ValueError(
                "proxy_pretrain action_mode must be none, geometry, idm, or latent"
            )
        if not bool(config_get(proxy, "use_precomputed_only", True)):
            raise ValueError(
                "proxy_pretrain forbids online proxy extraction; "
                "proxy.use_precomputed_only must be true"
            )
        proxy_type = str(config_get(proxy, "type", action_mode))
        if proxy_type != action_mode:
            raise ValueError(
                f"proxy.type={proxy_type!r} must match action_mode={action_mode!r}"
            )
        if action_mode != "none":
            proxy_dim = int(config_get(proxy, "dim", 0))
            type_config = config_get(motion, action_mode, {})
            dimension_key = {
                "geometry": "geometry_dim",
                "idm": "idm_dim",
                "latent": "latent_dim",
            }[action_mode]
            encoder_dim = int(config_get(type_config, dimension_key, 0))
            if proxy_dim < 1 or encoder_dim != proxy_dim:
                raise ValueError(
                    f"proxy.dim={proxy_dim} must equal "
                    f"motion_condition.{action_mode}.{dimension_key}={encoder_dim}"
                )
        return stage, action_mode, None
    if action_mode != "real":
        raise ValueError("real_finetune must use action_mode=real")
    finetune = config_get(config, "finetune", {})
    scheme = normalize_finetune_scheme(config_get(finetune, "scheme", "reset"))
    random_init = bool(config_get(finetune, "random_init", False))
    protocol = str(config_get(finetune, "dataset_protocol", "paper"))
    if protocol not in {"paper", "go2"}:
        raise ValueError(f"Unknown finetune.dataset_protocol: {protocol}")
    if config_get(finetune, "init_checkpoint", None):
        if random_init or config_get(finetune, "stage1_checkpoint", None):
            raise ValueError("init_checkpoint forbids stage1_checkpoint and random_init")
        if scheme != "reset":
            raise ValueError("Complete real-action initialization requires scheme=reset")
    if protocol == "go2":
        training = config_get(config, "training", {})
        if not (config_get(training, "wandb_enabled", False)
                and config_get(training, "wandb_required", False)):
            raise ValueError("Go2 adaptation requires online WandB training curves")
        if not config_get(finetune, "init_checkpoint", None) and not config_get(
            config_get(config, "training", {}), "from_checkpoint", None
        ):
            raise ValueError("Go2 adaptation requires init_checkpoint or from_checkpoint")
        if scheme != "reset" or random_init or config_get(finetune, "stage1_checkpoint", None):
            raise ValueError("Go2 adaptation must preserve the existing real-action adapter")
        if int(config_get(config, "eval_every", 0)) > 0 or bool(
            config_get(config, "eval_at_first_step", False)
        ):
            raise ValueError("Go2 adaptation must not evaluate the test set during training")
    decay = float(config_get(config_get(config, "training", {}), "ema_decay", 0.9999))
    if not 0 <= decay < 1:
        raise ValueError("training.ema_decay must be in [0, 1)")
    if random_init:
        if scheme != "reset":
            raise ValueError(
                "finetune.random_init=true requires finetune.scheme=reset"
            )
        if str(config_get(proxy, "type", "none")) != "none":
            raise ValueError("finetune.random_init=true requires proxy.type=none")
        if config_get(finetune, "stage1_checkpoint", None):
            raise ValueError(
                "finetune.random_init=true forbids finetune.stage1_checkpoint"
            )
        if int(config_get(finetune, "warmup_steps", 0)) != 0:
            raise ValueError(
                "finetune.random_init=true requires finetune.warmup_steps=0: "
                "a fresh CDiT's zero-initialized output/adaLN layers block "
                "adapter-only warmup gradients"
            )
    if scheme in {
        "embedding_align",
        "real_to_latent",
        "state_conditioned_controller",
    }:
        if str(config_get(proxy, "type", "")) != "latent":
            raise ValueError(f"{scheme} requires proxy.type=latent")
        latent_config = config_get(motion, "latent", {})
        latent_dim = int(config_get(latent_config, "latent_dim", 0))
        proxy_dim = int(config_get(proxy, "dim", 0))
        if latent_dim < 1 or proxy_dim != latent_dim:
            raise ValueError(
                f"proxy.dim={proxy_dim} must equal "
                f"motion_condition.latent.latent_dim={latent_dim}"
            )
    if scheme in {"embedding_align", "state_conditioned_controller"} and not bool(
        config_get(proxy, "use_precomputed_only", True)
    ):
        raise ValueError(f"{scheme} teacher latents must be read offline")
    if scheme == "state_conditioned_controller":
        if int(config_get(finetune, "controller_state_blocks", 2)) != 2:
            raise ValueError(
                "state_conditioned_controller requires controller_state_blocks=2"
            )
        if float(config_get(finetune, "latent_l2_weight", 1.0)) < 0:
            raise ValueError("finetune.latent_l2_weight must be non-negative")
        if int(config_get(finetune, "controller_lr_warmup_steps", 300)) < 0:
            raise ValueError(
                "finetune.controller_lr_warmup_steps must be non-negative"
            )
    return stage, action_mode, scheme


def alignment_loss(
    real_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
    valid: torch.Tensor,
    *,
    cosine_weight: float,
    l1_weight: float,
    distributed_mean: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute valid-only embedding alignment without empty-mask NaNs.

    When DDP is active, each rank is scaled by ``world_size/global_count``.
    DDP's gradient averaging then produces the exact global mean even when the
    number of valid local pairs differs between ranks.
    """
    if real_embedding.ndim != 2 or real_embedding.shape != target_embedding.shape:
        raise ValueError(
            "Alignment inputs must be equal [N,D] tensors; got "
            f"{tuple(real_embedding.shape)} and {tuple(target_embedding.shape)}"
        )
    valid = torch.as_tensor(valid, dtype=torch.bool, device=real_embedding.device)
    if valid.shape != (real_embedding.shape[0],):
        raise ValueError(
            f"Alignment valid mask must be [{real_embedding.shape[0]}], got "
            f"{tuple(valid.shape)}"
        )
    local_count = valid.sum()
    global_count = local_count.detach().clone()
    world_size = 1
    if distributed_mean and dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    if int(global_count.item()) > 0:
        selected_real = real_embedding[valid]
        selected_target = target_embedding[valid].to(selected_real)
        cosine_sum = (
            1.0 - F.cosine_similarity(selected_real, selected_target, dim=-1)
        ).sum()
        l1_sum = (selected_real - selected_target).abs().sum()
        cosine = cosine_sum * (float(world_size) / global_count.to(cosine_sum))
        l1 = l1_sum * (
            float(world_size)
            / (global_count.to(l1_sum) * real_embedding.shape[-1])
        )
    else:
        # Retain a zero-gradient graph to E_real for DDP without dividing by zero.
        cosine = real_embedding.sum() * 0.0
        l1 = real_embedding.sum() * 0.0
    total = float(cosine_weight) * cosine + float(l1_weight) * l1
    return {
        "alignment": total,
        "alignment_cosine": cosine,
        "alignment_l1": l1,
        "alignment_valid_count": global_count,
    }


def latent_action_l2_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    distributed_mean: bool = False,
) -> dict[str, torch.Tensor]:
    """Valid-only latent MSE with exact global normalization under DDP."""
    if predicted.ndim != 2 or predicted.shape != target.shape:
        raise ValueError(
            "Latent L2 inputs must be equal [N,D] tensors; got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}"
        )
    valid = torch.as_tensor(valid, dtype=torch.bool, device=predicted.device)
    if valid.shape != (predicted.shape[0],):
        raise ValueError(
            f"Latent L2 valid mask must be [{predicted.shape[0]}], got "
            f"{tuple(valid.shape)}"
        )
    local_count = valid.sum()
    global_count = local_count.detach().clone()
    world_size = 1
    if distributed_mean and dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    if int(global_count.item()) > 0:
        selected_prediction = predicted[valid]
        selected_target = target.detach()[valid].to(selected_prediction)
        squared_sum = (selected_prediction - selected_target).square().sum()
        mse = squared_sum * (
            float(world_size)
            / (global_count.to(squared_sum) * predicted.shape[-1])
        )
    else:
        # Every rank retains the controller graph so DDP collectives stay aligned.
        mse = predicted.sum() * 0.0
    return {
        "latent_l2": mse,
        "latent_l2_valid_count": global_count,
    }


def flatten_goal_tensor(
    value: torch.Tensor,
    *,
    batch_size: int,
    num_goals: int,
    name: str,
) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.shape[:2] != (batch_size, num_goals):
        raise ValueError(
            f"{name} must begin with [B,G]=[{batch_size},{num_goals}], got "
            f"{tuple(value.shape)}"
        )
    return value.flatten(0, 1)


def assert_goal_alignment(
    *,
    target_batch: torch.Tensor,
    rel_t: torch.Tensor,
    frame_offset: torch.Tensor,
    action: torch.Tensor | None = None,
    action_valid: torch.Tensor | None = None,
) -> None:
    expected = target_batch.shape[0]
    named = {"rel_t": rel_t, "frame_offset": frame_offset}
    if action is not None:
        named["action"] = action
    if action_valid is not None:
        named["action_valid"] = action_valid
    mismatched = {
        name: tuple(tensor.shape)
        for name, tensor in named.items()
        if tensor.shape[0] != expected
    }
    if mismatched:
        raise ValueError(
            f"Flattened goal fields do not match target batch {expected}: {mismatched}"
        )


def dense_motion_from_collated(
    grouped: Mapping[str, Mapping[str, torch.Tensor]] | None,
    *,
    motion_type: str,
    batch_size: int,
    num_goals: int,
    input_dim: int,
    device: torch.device,
    require_all: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover a B-major/G-minor dense action and validity tensor."""
    count = batch_size * num_goals
    dense = torch.zeros(count, input_dim, dtype=torch.float32, device=device)
    valid = torch.zeros(count, dtype=torch.bool, device=device)
    if not grouped or motion_type not in grouped:
        if require_all:
            raise ValueError(f"Batch has no {motion_type!r} motion group")
        return dense, valid
    payload = grouped[motion_type]
    sample_indices = torch.as_tensor(payload["sample_indices"], dtype=torch.int64)
    values = torch.as_tensor(payload["values"], dtype=torch.float32)
    masks = torch.as_tensor(
        payload.get("masks", torch.ones(values.shape[:2], dtype=torch.bool)),
        dtype=torch.bool,
    )
    if values.shape != (sample_indices.numel(), num_goals, input_dim):
        raise ValueError(
            f"Collated {motion_type} values must be "
            f"[{sample_indices.numel()},{num_goals},{input_dim}], got {tuple(values.shape)}"
        )
    if masks.shape != values.shape[:2]:
        raise ValueError(f"Collated {motion_type} mask shape is invalid")
    goals = torch.arange(num_goals, dtype=torch.int64)
    indices = (sample_indices[:, None] * num_goals + goals[None, :]).reshape(-1)
    flat_mask = masks.reshape(-1)
    indices = indices[flat_mask].to(device)
    flat_values = values.flatten(0, 1)[flat_mask].to(device)
    if bool(((indices < 0) | (indices >= count)).any()):
        raise ValueError(
            f"Collated {motion_type} sample_indices are outside batch size {batch_size}"
        )
    if indices.numel() != torch.unique(indices).numel():
        raise ValueError(f"Duplicate flattened {motion_type} indices")
    dense.index_copy_(0, indices, flat_values)
    valid[indices] = True
    if require_all and not bool(valid.all()):
        missing = torch.nonzero(~valid, as_tuple=False).flatten().tolist()
        raise ValueError(
            f"Real-action finetuning requires every goal action; missing={missing[:16]}"
        )
    return dense, valid


def _is_real_adapter(name: str) -> bool:
    return name.startswith("motion_condition_encoder.real_")


def _is_latent_adapter(name: str) -> bool:
    return name.startswith("motion_condition_encoder.latent_")


def _is_any_action_adapter(name: str) -> bool:
    return name.startswith("motion_condition_encoder.")


def _is_state_controller(name: str) -> bool:
    return name.startswith("state_conditioned_controller.")


def configure_trainable_parameters(
    model: torch.nn.Module,
    *,
    training_stage: str,
    action_mode: str,
    finetune_scheme: str | None = None,
    finetune_substage: str | None = None,
) -> dict[str, Any]:
    """Set exact trainability for stage 1 and both finetuning substages."""
    if training_stage not in {"proxy_pretrain", "real_finetune"}:
        raise ValueError("configure_trainable_parameters is for two-stage training")
    scheme = (
        normalize_finetune_scheme(finetune_scheme)
        if training_stage == "real_finetune"
        else None
    )
    if training_stage == "real_finetune" and finetune_substage not in FINETUNE_SUBSTAGES:
        raise ValueError(f"finetune_substage must be one of {FINETUNE_SUBSTAGES}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for name, parameter in model.named_parameters():
        train = False
        if training_stage == "proxy_pretrain":
            if not _is_any_action_adapter(name):
                train = True
            elif action_mode != "none" and name.startswith(
                f"motion_condition_encoder.{action_mode}_"
            ):
                train = True
        elif finetune_substage == "warmup":
            if scheme in {"reset", "embedding_align"}:
                train = _is_real_adapter(name)
            elif scheme == "real_to_latent":
                train = name.startswith("real_to_latent.")
            else:
                train = _is_state_controller(name)
        else:  # joint
            if scheme in {"reset", "embedding_align"}:
                train = not _is_any_action_adapter(name) or _is_real_adapter(name)
            elif scheme == "real_to_latent":
                train = (
                    not _is_any_action_adapter(name)
                    or _is_latent_adapter(name)
                    or name.startswith("real_to_latent.")
                )
            else:
                # The controller and CDiT backbone train jointly.  The shared
                # image patch embedder and every motion adapter (including E_z)
                # retain their Stage-1 weights exactly.
                train = _is_state_controller(name) or (
                    not _is_any_action_adapter(name)
                    and not name.startswith("x_embedder.")
                    and not _is_state_controller(name)
                )
        parameter.requires_grad_(train)

    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters remain after stage configuration")
    return {
        "trainable_names": trainable,
        "frozen_names": frozen,
        "trainable_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_count": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }


def build_optimizer_param_groups(
    model: torch.nn.Module,
    *,
    training_stage: str,
    finetune_scheme: str | None,
    finetune_substage: str | None,
    adapter_lr: float,
    backbone_lr: float,
) -> list[dict[str, Any]]:
    """Build named, filtered groups and assert that frozen teachers are absent."""
    scheme = normalize_finetune_scheme(finetune_scheme) if finetune_scheme else None
    adapter_parameters = []
    backbone_parameters = []
    optimizer_names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if scheme in {"reset", "embedding_align"}:
            is_adapter = _is_real_adapter(name)
        elif scheme == "real_to_latent":
            is_adapter = name.startswith("real_to_latent.")
        else:
            is_adapter = _is_state_controller(name)
        (adapter_parameters if is_adapter else backbone_parameters).append(parameter)
        optimizer_names.append(name)
    groups = []
    if adapter_parameters:
        groups.append(
            {"params": adapter_parameters, "lr": float(adapter_lr), "name": "adapter"}
        )
    if backbone_parameters:
        groups.append(
            {"params": backbone_parameters, "lr": float(backbone_lr), "name": "backbone"}
        )
    if not groups:
        raise RuntimeError("Optimizer would have no parameters")
    for group in groups:
        if not all(parameter.requires_grad for parameter in group["params"]):
            raise AssertionError("Optimizer received a frozen parameter")
    if scheme in {"embedding_align", "state_conditioned_controller"} or (
        scheme == "real_to_latent" and finetune_substage == "warmup"
    ):
        forbidden = [name for name in optimizer_names if _is_latent_adapter(name)]
        if forbidden:
            raise AssertionError(f"Frozen latent teacher entered optimizer: {forbidden}")
    return groups


def trainability_report(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups = [
        {
            "name": group.get("name", f"group_{index}"),
            "lr": float(group["lr"]),
            "parameter_count": sum(p.numel() for p in group["params"]),
        }
        for index, group in enumerate(optimizer.param_groups)
    ]
    return {
        "trainable_parameter_names": names,
        "frozen_parameter_count": frozen_count,
        "trainable_parameter_count": trainable_count,
        "optimizer_groups": groups,
    }


@dataclass
class ProxyMetrics:
    total_samples: float = 0.0
    eligible_samples: float = 0.0
    ineligible_long_range_samples: float = 0.0
    found_samples: float = 0.0
    missing_samples: float = 0.0
    invalid_samples: float = 0.0
    used_samples: float = 0.0
    abs_offset_used_sum: float = 0.0
    abs_offset_time_only_sum: float = 0.0
    time_only_samples: float = 0.0

    def update(
        self,
        frame_offset: torch.Tensor,
        *,
        eligible: torch.Tensor,
        found: torch.Tensor,
        invalid: torch.Tensor,
        used: torch.Tensor,
    ) -> None:
        offsets = torch.as_tensor(frame_offset, dtype=torch.int64).reshape(-1)
        tensors = {
            "eligible": torch.as_tensor(eligible, dtype=torch.bool).reshape(-1),
            "found": torch.as_tensor(found, dtype=torch.bool).reshape(-1),
            "invalid": torch.as_tensor(invalid, dtype=torch.bool).reshape(-1),
            "used": torch.as_tensor(used, dtype=torch.bool).reshape(-1),
        }
        if any(value.numel() != offsets.numel() for value in tensors.values()):
            raise ValueError("Proxy metric masks must align with frame_offset")
        eligible_t = tensors["eligible"]
        found_t = tensors["found"]
        invalid_t = tensors["invalid"]
        used_t = tensors["used"]
        if bool((used_t & ~eligible_t).any()):
            raise ValueError("Used proxy samples must be eligible")
        self.total_samples += offsets.numel()
        self.eligible_samples += eligible_t.sum().item()
        self.ineligible_long_range_samples += (~eligible_t).sum().item()
        self.found_samples += (eligible_t & found_t).sum().item()
        self.missing_samples += (eligible_t & ~found_t).sum().item()
        self.invalid_samples += (eligible_t & found_t & invalid_t).sum().item()
        self.used_samples += used_t.sum().item()
        self.abs_offset_used_sum += offsets[used_t].abs().sum().item()
        time_only = ~used_t
        self.time_only_samples += time_only.sum().item()
        self.abs_offset_time_only_sum += offsets[time_only].abs().sum().item()

    def distributed_reduce(self, device: torch.device) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        names = tuple(self.__dataclass_fields__)
        values = torch.tensor([getattr(self, name) for name in names], device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        for name, value in zip(names, values.tolist()):
            setattr(self, name, float(value))

    def as_log_dict(self) -> dict[str, float]:
        total = max(self.total_samples, 1.0)
        used = max(self.used_samples, 1.0)
        time_only = max(self.time_only_samples, 1.0)
        return {
            "proxy/total_samples": self.total_samples,
            "proxy/eligible_samples": self.eligible_samples,
            "proxy/ineligible_long_range_samples": self.ineligible_long_range_samples,
            "proxy/found_samples": self.found_samples,
            "proxy/missing_samples": self.missing_samples,
            "proxy/invalid_samples": self.invalid_samples,
            "proxy/used_samples": self.used_samples,
            "proxy/eligible_rate": self.eligible_samples / total,
            "proxy/used_rate": self.used_samples / total,
            "proxy/mean_abs_offset_used": (
                self.abs_offset_used_sum / used if self.used_samples else 0.0
            ),
            "proxy/mean_abs_offset_time_only": (
                self.abs_offset_time_only_sum / time_only
                if self.time_only_samples
                else 0.0
            ),
        }


def predicted_latent_metrics(value: torch.Tensor) -> dict[str, float]:
    detached = value.detach().float()
    if detached.numel() == 0:
        return {key: 0.0 for key in (
            "predicted_latent/mean",
            "predicted_latent/std",
            "predicted_latent/norm",
            "predicted_latent/max_abs",
        )}
    return {
        "predicted_latent/mean": detached.mean().item(),
        "predicted_latent/std": detached.std(unbiased=False).item(),
        "predicted_latent/norm": detached.norm(dim=-1).mean().item(),
        "predicted_latent/max_abs": detached.abs().max().item(),
    }
