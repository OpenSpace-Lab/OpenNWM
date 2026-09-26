"""Callbacks used only by the isolated navigation LAM training entry point."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from lam.modules.blocks import SelfAttention, SpatioBlock, SpatioTemporalBlock
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def enable_fused_sdpa(module: nn.Module, *, require_match: bool = True) -> int:
    """Idempotently enable PyTorch SDPA on every LAM attention module.

    The patch replaces only a bound Python callable and therefore adds no
    checkpoint state.  It is shared by the training callback and standalone
    evaluation, which prevents the two paths from quietly using different
    attention implementations.

    Returns:
        Number of newly patched :class:`SelfAttention` modules.  A second call
        on the same module tree returns zero.
    """

    matched = 0
    patched = 0
    for attention in module.modules():
        if not isinstance(attention, SelfAttention):
            continue
        matched += 1
        if getattr(attention, "_navigation_fused_attention", False):
            continue

        def fused_attention(
            query: Tensor,
            key: Tensor,
            value: Tensor,
            is_causal: bool = False,
            *,
            _attention: SelfAttention = attention,
        ) -> Tensor:
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=_attention.scale,
            )

        attention.scaled_dot_product_attention = fused_attention  # type: ignore[method-assign]
        attention._navigation_fused_attention = True  # type: ignore[attr-defined]
        patched += 1

    if require_match and matched == 0:
        raise RuntimeError("LAM fused attention found no SelfAttention modules")
    return patched


class LAMFusedAttention(Callback):
    """Use PyTorch fused SDPA for navigation LAM attention without new state."""

    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.patched_modules = 0

    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
    ) -> None:
        del trainer
        if not self.enabled or stage != "fit":
            return
        self.patched_modules += enable_fused_sdpa(pl_module, require_match=True)
        rank_zero_info(
            "Enabled PyTorch fused SDPA for "
            f"{self.patched_modules} navigation LAM attention modules."
        )


class LAMActivationCheckpointing(Callback):
    """Checkpoint LAM transformer blocks without changing model/state-dict keys.

    The callback replaces only the Python ``forward`` callables of existing
    LAM blocks. It adds no modules, parameters, or buffers. During training with
    gradients enabled, block activations are recomputed in backward; eval and
    no-grad execution continue to call the original forward directly.
    """

    def __init__(self, enabled: bool = True, use_reentrant: bool = False) -> None:
        super().__init__()
        self.enabled = enabled
        self.use_reentrant = use_reentrant
        self.patched_modules = 0

    def _wrap(self, module: nn.Module) -> None:
        if getattr(module, "_navigation_activation_checkpointed", False):
            return
        original_forward: Callable[..., Tensor] = module.forward

        def checkpointed_forward(*args: Any, **kwargs: Any) -> Tensor:
            if module.training and torch.is_grad_enabled():
                return checkpoint(
                    original_forward,
                    *args,
                    use_reentrant=self.use_reentrant,
                    **kwargs,
                )
            return original_forward(*args, **kwargs)

        module.forward = checkpointed_forward  # type: ignore[method-assign]
        module._navigation_activation_checkpointed = True  # type: ignore[attr-defined]
        self.patched_modules += 1

    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str,
    ) -> None:
        del trainer
        if not self.enabled or stage != "fit":
            return
        for module in pl_module.modules():
            if isinstance(module, (SpatioBlock, SpatioTemporalBlock)):
                self._wrap(module)
        if self.patched_modules == 0:
            raise RuntimeError(
                "LAM activation checkpointing found no SpatioBlock or "
                "SpatioTemporalBlock modules"
            )
        rank_zero_info(
            "Enabled activation checkpointing for "
            f"{self.patched_modules} navigation LAM transformer blocks."
        )


class LAMFinalCheckpoint(Callback):
    """Save the non-interval terminal step as a full resumable checkpoint."""

    def __init__(self, dirpath: str, final_step: int) -> None:
        super().__init__()
        if final_step <= 0:
            raise ValueError("final_step must be positive")
        self.dirpath = Path(dirpath).expanduser().resolve(strict=False)
        self.final_step = int(final_step)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if trainer.global_step != self.final_step:
            return
        checkpoint = self.dirpath / f"step={self.final_step}.ckpt"
        self.dirpath.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(checkpoint, weights_only=False)
        trainer.strategy.barrier("navigation-final-checkpoint")
        if trainer.is_global_zero:
            temporary = self.dirpath / f".last.ckpt.{os.getpid()}.tmp"
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(checkpoint.name)
            os.replace(temporary, self.dirpath / "last.ckpt")
            rank_zero_info(f"Saved terminal navigation checkpoint: {checkpoint}")
        trainer.strategy.barrier("navigation-final-checkpoint-link")


__all__ = [
    "LAMActivationCheckpointing",
    "LAMFinalCheckpoint",
    "LAMFusedAttention",
    "enable_fused_sdpa",
]
