from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from lam.model import LAM
from lam.modules.blocks import PositionalEncoding
from lam.navigation_callbacks import (
    LAMActivationCheckpointing,
    LAMFinalCheckpoint,
    LAMFusedAttention,
    enable_fused_sdpa,
)


def _small_model() -> LAM:
    return LAM(
        image_channels=3,
        lam_model_dim=8,
        lam_latent_dim=4,
        lam_patch_size=2,
        lam_enc_blocks=1,
        lam_dec_blocks=1,
        lam_num_heads=2,
        lam_dropout=0.0,
        beta=1e-6,
    )


def test_activation_checkpointing_preserves_state_outputs_loss_and_gradients(
    monkeypatch,
):
    monkeypatch.setattr(
        PositionalEncoding,
        "forward",
        lambda self, value: (
            value + self.pos_enc[: value.shape[2]].to(value.device, value.dtype)
        ),
    )
    torch.manual_seed(12)
    original = _small_model().train()
    torch.manual_seed(12)
    checkpointed = _small_model().train()

    callback = LAMActivationCheckpointing()
    state_keys_before = tuple(checkpointed.state_dict())
    callback.setup(None, checkpointed, "fit")
    assert callback.patched_modules == 2
    assert tuple(checkpointed.state_dict()) == state_keys_before
    assert tuple(checkpointed.state_dict()) == tuple(original.state_dict())

    torch.manual_seed(21)
    batch = {"videos": torch.rand(2, 2, 4, 4, 3)}
    torch.manual_seed(34)
    original_outputs = original.lam(batch)
    torch.manual_seed(34)
    checkpointed_outputs = checkpointed.lam(batch)
    for key in original_outputs:
        torch.testing.assert_close(
            original_outputs[key], checkpointed_outputs[key], rtol=0, atol=0
        )

    original_loss = original_outputs["recon"].sum() + original_outputs["z_mu"].sum()
    checkpointed_loss = (
        checkpointed_outputs["recon"].sum() + checkpointed_outputs["z_mu"].sum()
    )
    torch.testing.assert_close(original_loss, checkpointed_loss, rtol=0, atol=0)
    original_loss.backward()
    checkpointed_loss.backward()
    for original_parameter, checkpointed_parameter in zip(
        original.parameters(), checkpointed.parameters()
    ):
        assert (original_parameter.grad is None) == (
            checkpointed_parameter.grad is None
        )
        if original_parameter.grad is not None:
            torch.testing.assert_close(
                original_parameter.grad,
                checkpointed_parameter.grad,
                rtol=1e-5,
                atol=1e-6,
            )


def test_activation_checkpointing_is_not_applied_outside_fit():
    model = _small_model()
    callback = LAMActivationCheckpointing()
    callback.setup(None, model, "test")
    assert callback.patched_modules == 0
    assert not any(
        getattr(module, "_navigation_activation_checkpointed", False)
        for module in model.modules()
    )


def test_fused_attention_preserves_state_and_attention_semantics(monkeypatch):
    monkeypatch.setattr(
        PositionalEncoding,
        "forward",
        lambda self, value: (
            value + self.pos_enc[: value.shape[2]].to(value.device, value.dtype)
        ),
    )
    torch.manual_seed(45)
    original = _small_model().train()
    torch.manual_seed(45)
    fused = _small_model().train()
    state_keys_before = tuple(fused.state_dict())

    callback = LAMFusedAttention()
    callback.setup(None, fused, "fit")
    assert callback.patched_modules == 3
    assert tuple(fused.state_dict()) == state_keys_before
    assert tuple(fused.state_dict()) == tuple(original.state_dict())

    torch.manual_seed(51)
    batch = {"videos": torch.rand(2, 2, 4, 4, 3)}
    torch.manual_seed(63)
    original_outputs = original.lam(batch)
    torch.manual_seed(63)
    fused_outputs = fused.lam(batch)
    for key in original_outputs:
        torch.testing.assert_close(
            original_outputs[key], fused_outputs[key], rtol=1e-5, atol=1e-6
        )

    fused_loss = fused_outputs["recon"].sum() + fused_outputs["z_mu"].sum()
    fused_loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in fused.parameters()
    )


def test_fused_attention_helper_is_idempotent():
    model = _small_model()
    state_keys = tuple(model.state_dict())
    assert enable_fused_sdpa(model) == 3
    assert enable_fused_sdpa(model) == 0
    assert tuple(model.state_dict()) == state_keys


def test_final_checkpoint_saves_only_the_requested_terminal_step(tmp_path: Path):
    saved: list[tuple[Path, bool]] = []
    barriers: list[str] = []

    def save_checkpoint(path: Path, *, weights_only: bool) -> None:
        saved.append((path, weights_only))
        path.write_bytes(b"full checkpoint")

    trainer = SimpleNamespace(
        global_step=62_500,
        is_global_zero=True,
        save_checkpoint=save_checkpoint,
        strategy=SimpleNamespace(barrier=barriers.append),
    )
    callback = LAMFinalCheckpoint(str(tmp_path), 62_500)
    callback.on_train_end(trainer, None)

    checkpoint = tmp_path / "step=62500.ckpt"
    assert saved == [(checkpoint, False)]
    assert barriers == [
        "navigation-final-checkpoint",
        "navigation-final-checkpoint-link",
    ]
    assert (tmp_path / "last.ckpt").is_symlink()
    assert (tmp_path / "last.ckpt").resolve() == checkpoint

    trainer.global_step = 60_000
    saved.clear()
    barriers.clear()
    callback.on_train_end(trainer, None)
    assert saved == []
    assert barriers == []
