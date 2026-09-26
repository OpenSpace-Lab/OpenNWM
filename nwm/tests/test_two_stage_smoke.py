"""End-to-end CPU smoke tests for the opt-in two-stage NWM trainer.

Unlike the focused unit tests in :mod:`test_two_stage_nwm`, these tests run the
production ``two_stage_train_step`` with the real Gaussian diffusion object.
The model, images, and batches are deliberately tiny, but the batch layout is
the real ``[B, context + goals, C, H, W]`` layout used by the trainer.
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion import create_gaussian_diffusion
from models import CDiT
from two_stage_checkpoint import load_two_stage_resume
from two_stage_nwm import (
    build_optimizer_param_groups,
    configure_trainable_parameters,
)
from two_stage_training import (
    _defer_random_init_gradient_check,
    save_two_stage_checkpoint,
    two_stage_train_step,
)


SMOKE_DEVICE = torch.device(os.environ.get("TWO_STAGE_SMOKE_DEVICE", "cpu"))
if SMOKE_DEVICE.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError(
        "TWO_STAGE_SMOKE_DEVICE requested CUDA, but CUDA is unavailable"
    )


MOTION_CONFIG = {
    "enabled": True,
    "available_types": ["none", "real", "geometry", "idm", "latent"],
    "train_types": ["none", "real", "geometry", "idm", "latent"],
    "adapter_hidden_dim": 6,
    "balance_parameter_count": False,
    "real": {"real_dim": 3, "normalization": {"mode": "identity"}},
    "geometry": {
        "geometry_dim": 3,
        "normalization": {"mode": "identity"},
    },
    "idm": {"idm_dim": 5, "normalization": {"mode": "identity"}},
    "latent": {
        "latent_dim": 4,
        # This is the production latent-coordinate contract.
        "normalization": {"mode": "layer_norm"},
    },
}


class IdentityTokenizer:
    """Minimal tokenizer preserving the production encode call contract."""

    scaling_factor = 1.0

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return value


def _open_zero_gates(model: CDiT) -> None:
    """Open fresh CDiT zero gates so adapter gradients are observable."""

    generator = torch.Generator(device="cpu").manual_seed(20260901)
    with torch.no_grad():
        for block in model.blocks:
            block.adaLN_modulation[-1].weight.copy_(
                torch.randn(
                    block.adaLN_modulation[-1].weight.shape,
                    generator=generator,
                )
                * 0.04
            )
            block.adaLN_modulation[-1].bias.copy_(
                torch.randn(
                    block.adaLN_modulation[-1].bias.shape,
                    generator=generator,
                )
                * 0.04
            )
        model.final_layer.adaLN_modulation[-1].weight.copy_(
            torch.randn(
                model.final_layer.adaLN_modulation[-1].weight.shape,
                generator=generator,
            )
            * 0.04
        )
        model.final_layer.adaLN_modulation[-1].bias.copy_(
            torch.randn(
                model.final_layer.adaLN_modulation[-1].bias.shape,
                generator=generator,
            )
            * 0.04
        )
        model.final_layer.linear.weight.copy_(
            torch.randn(
                model.final_layer.linear.weight.shape,
                generator=generator,
            )
            * 0.04
        )
        model.final_layer.linear.bias.zero_()


def _config(
    *,
    training_stage: str,
    action_mode: str,
    scheme: str = "reset",
    relative_time_mode: str = "always",
):
    proxy_type = action_mode if training_stage == "proxy_pretrain" else "latent"
    proxy = {
        "type": proxy_type,
        "max_abs_frame_offset": 8,
        "use_precomputed_only": True,
        "relative_time_mode": relative_time_mode,
    }
    if proxy_type != "none":
        proxy["dim"] = 4
    return OmegaConf.create(
        {
            "training_stage": training_stage,
            "action_mode": action_mode,
            "bfloat16": False,
            "dataset": {
                "context_size": 2,
                "precomputed_latents": {"enabled": False},
                "distance": {"min_dist_cat": -64, "max_dist_cat": 64},
            },
            "motion_condition": copy.deepcopy(MOTION_CONFIG),
            "proxy": proxy,
            "finetune": {
                "scheme": scheme,
                "warmup_steps": 2,
                "joint_steps": 2,
                "adapter_lr": 2.0e-3,
                "backbone_lr": 1.0e-3,
                "align_cos_weight": 1.0,
                "align_l1_weight": 0.25,
                "real_to_latent_hidden_dim": 6,
                "freeze_latent_action_encoder": True,
            },
            "training": {
                "grad_clip_val": 0.0,
                "optimizer": {"lr": 1.0e-3},
            },
            # Checkpoint metadata reads this signature. The actual model is
            # instantiated directly below to keep the smoke independent of Hydra.
            "model": {
                "generator": {
                    "_target_": "models.CDiT",
                    "hidden_size": 8,
                }
            },
        }
    )


def _model(config, *, open_zero_gates: bool = True) -> CDiT:
    torch.manual_seed(1103)
    model = CDiT(
        input_size=2,
        context_size=2,
        patch_size=1,
        in_channels=1,
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=1.0,
        learn_sigma=False,
        motion_condition=MOTION_CONFIG,
        training_stage=str(config.training_stage),
        action_mode=str(config.action_mode),
        proxy_relative_time_mode=str(config.proxy.relative_time_mode),
        finetune=OmegaConf.to_container(config.finetune, resolve=True),
    ).to(SMOKE_DEVICE)
    if open_zero_gates:
        _open_zero_gates(model)
    return model


def _diffusion(*, use_kl: bool = False):
    return create_gaussian_diffusion(
        timestep_respacing="",
        noise_schedule="squaredcos_cap_v2",
        use_kl=use_kl,
        learn_sigma=False,
        diffusion_steps=12,
    )


def _batch(*, include_teacher: bool = True) -> dict[str, torch.Tensor]:
    """Return B=2, G=2 with one local and one long goal per sample."""

    generator = torch.Generator(device="cpu").manual_seed(2207)
    result = {
        "video": torch.randn(2, 4, 1, 2, 2, generator=generator),
        "k": torch.tensor([[-0.125, 1.0], [0.015625, -1.0]]),
        "frame_offset": torch.tensor([[-8, 64], [1, -64]], dtype=torch.int64),
        "real_action": torch.randn(2, 2, 3, generator=generator),
        "proxy_action": torch.randn(2, 2, 4, generator=generator),
        # Deliberately mark long-range placeholders valid. The trainer must
        # intersect this with the raw integer offset before model.forward.
        "proxy_valid": torch.ones(2, 2, dtype=torch.bool),
        "proxy_found": torch.ones(2, 2, dtype=torch.bool),
        "proxy_invalid": torch.zeros(2, 2, dtype=torch.bool),
    }
    if include_teacher:
        result.update(
            teacher_latent=torch.randn(2, 2, 4, generator=generator),
            # As above, the trainer owns the authoritative local-range mask.
            latent_valid=torch.ones(2, 2, dtype=torch.bool),
        )
    return result


def _ema(model: CDiT) -> CDiT:
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    return ema


def _optimizer(
    model: CDiT,
    *,
    training_stage: str,
    action_mode: str,
    scheme: str | None,
    substage: str | None,
) -> torch.optim.Optimizer:
    configure_trainable_parameters(
        model,
        training_stage=training_stage,
        action_mode=action_mode,
        finetune_scheme=scheme,
        finetune_substage=substage,
    )
    groups = build_optimizer_param_groups(
        model,
        training_stage=training_stage,
        finetune_scheme=scheme,
        finetune_substage=substage,
        adapter_lr=2.0e-3,
        backbone_lr=1.0e-3,
    )
    return torch.optim.AdamW(groups, weight_decay=0.0)


def _step(
    *,
    model: CDiT,
    ema: CDiT,
    diffusion,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    config,
    scheme: str | None,
    substage: str | None,
    check_gradients: bool,
):
    logs, proxy = two_stage_train_step(
        model=model,
        ema=ema,
        diffusion=diffusion,
        tokenizer=IdentityTokenizer(),
        batch=batch,
        device=SMOKE_DEVICE,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        config=config,
        training_stage=str(config.training_stage),
        scheme=scheme,
        substage=substage,
        check_gradients=check_gradients,
    )
    if not bool(torch.isfinite(torch.as_tensor(logs["loss"]))):
        raise AssertionError(f"non-finite smoke loss: {logs}")
    return logs, proxy


def _state_copy(model: CDiT, prefix: str) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith(prefix)
    }


def _any_changed(before: dict[str, torch.Tensor], model: CDiT) -> bool:
    current = dict(model.named_parameters())
    return any(not torch.equal(value, current[name].detach()) for name, value in before.items())


class TwoStageProductionSmokeTests(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        torch.manual_seed(3301)

    def test_no_pretrain_opens_standard_zero_init_before_gradient_check(self):
        config = _config(
            training_stage="real_finetune", action_mode="real", scheme="reset"
        )
        config.finetune.random_init = True
        config.finetune.warmup_steps = 0
        config.finetune.joint_steps = 3
        model = _model(config, open_zero_gates=False)
        ema = _ema(model)
        optimizer = _optimizer(
            model,
            training_stage="real_finetune",
            action_mode="real",
            scheme="reset",
            substage="joint",
        )
        before = _state_copy(model, "motion_condition_encoder.real_")

        for current in range(3):
            logs, _ = _step(
                model=model,
                ema=ema,
                diffusion=_diffusion(),
                optimizer=optimizer,
                batch=_batch(),
                config=config,
                scheme="reset",
                substage="joint",
                check_gradients=not _defer_random_init_gradient_check(
                    config, "joint", current
                ),
            )
            self.assertEqual(
                bool(logs.get("gradient_check_performed", False)), current == 2
            )

        self.assertTrue(_any_changed(before, model))

    def test_timept_three_steps_checkpoint_and_exact_resume(self):
        config = _config(
            training_stage="proxy_pretrain", action_mode="none"
        )
        model = _model(config)
        ema = _ema(model)
        optimizer = _optimizer(
            model,
            training_stage="proxy_pretrain",
            action_mode="none",
            scheme=None,
            substage=None,
        )
        diffusion = _diffusion()
        batch = _batch()

        for _ in range(2):
            logs, proxy = _step(
                model=model,
                ema=ema,
                diffusion=diffusion,
                optimizer=optimizer,
                batch=batch,
                config=config,
                scheme=None,
                substage=None,
                check_gradients=False,
            )
            self.assertTrue(bool(torch.isfinite(logs["loss"])))
            self.assertEqual(int(proxy["used"].sum()), 0)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = str(Path(directory) / "timept-step2.pt")
            save_two_stage_checkpoint(
                checkpoint_path,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=None,
                scaler=None,
                config=config,
                substage="pretrain",
                completed_warmup_steps=0,
                completed_joint_steps=0,
                train_steps=2,
                epoch=3,
                include_optimizer=True,
            )

            resumed = _model(config)
            resumed_optimizer = _optimizer(
                resumed,
                training_stage="proxy_pretrain",
                action_mode="none",
                scheme=None,
                substage=None,
            )
            info = load_two_stage_resume(
                resumed,
                resumed_optimizer,
                checkpoint_path,
                expected_substage="pretrain",
                config=config,
            )
            self.assertTrue(info["optimizer_loaded"])
            self.assertEqual(info["train_steps"], 2)
            self.assertEqual(info["epoch"], 3)
            self.assertTrue(resumed_optimizer.state)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(resumed.state_dict()[name], value)

            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            resumed_ema = _ema(resumed)
            resumed_ema.load_state_dict(checkpoint["ema"], strict=True)
            logs, proxy = _step(
                model=resumed,
                ema=resumed_ema,
                diffusion=diffusion,
                optimizer=resumed_optimizer,
                batch=batch,
                config=config,
                scheme=None,
                substage=None,
                check_gradients=False,
            )
            self.assertTrue(bool(torch.isfinite(logs["loss"])))
            self.assertEqual(int(proxy["used"].sum()), 0)

    def test_latentpt_two_steps_mix_local_and_long_range(self):
        config = _config(
            training_stage="proxy_pretrain", action_mode="latent"
        )
        model = _model(config)
        ema = _ema(model)
        optimizer = _optimizer(
            model,
            training_stage="proxy_pretrain",
            action_mode="latent",
            scheme=None,
            substage=None,
        )
        diffusion = _diffusion()
        before = _state_copy(model, "motion_condition_encoder.latent_")
        expected = torch.tensor([True, False, True, False])
        for _ in range(2):
            logs, proxy = _step(
                model=model,
                ema=ema,
                diffusion=diffusion,
                optimizer=optimizer,
                batch=_batch(),
                config=config,
                scheme=None,
                substage=None,
                check_gradients=False,
            )
            self.assertTrue(torch.equal(proxy["eligible"].cpu(), expected))
            self.assertTrue(torch.equal(proxy["used"].cpu(), expected))
            self.assertEqual(int(proxy["frame_offset"].abs().max()), 64)
            self.assertTrue(bool(torch.isfinite(logs["loss"])))
        self.assertTrue(_any_changed(before, model))

    def test_latentonlypt_two_steps_mix_local_latent_and_long_range_time(self):
        config = _config(
            training_stage="proxy_pretrain",
            action_mode="latent",
            relative_time_mode="fallback",
        )
        model = _model(config)
        self.assertEqual(model.proxy_relative_time_mode, "fallback")
        ema = _ema(model)
        optimizer = _optimizer(
            model,
            training_stage="proxy_pretrain",
            action_mode="latent",
            scheme=None,
            substage=None,
        )
        logs, proxy = _step(
            model=model,
            ema=ema,
            diffusion=_diffusion(),
            optimizer=optimizer,
            batch=_batch(),
            config=config,
            scheme=None,
            substage=None,
            check_gradients=False,
        )
        expected = torch.tensor([True, False, True, False])
        self.assertTrue(torch.equal(proxy["used"].cpu(), expected))
        self.assertTrue(bool(torch.isfinite(logs["loss"])))

    def _run_finetune_pair(self, scheme: str):
        config = _config(
            training_stage="real_finetune",
            action_mode="real",
            scheme=scheme,
        )
        model = _model(config)
        ema = _ema(model)
        diffusion = _diffusion()
        batch = _batch(include_teacher=scheme != "real_to_latent")

        warmup_optimizer = _optimizer(
            model,
            training_stage="real_finetune",
            action_mode="real",
            scheme=scheme,
            substage="warmup",
        )
        warmup = None
        for _ in range(2):
            warmup, _ = _step(
                model=model,
                ema=ema,
                diffusion=diffusion,
                optimizer=warmup_optimizer,
                batch=batch,
                config=config,
                scheme=scheme,
                substage="warmup",
                check_gradients=True,
            )
        assert warmup is not None

        # Exercise the real transition-checkpoint contract on the main B path.
        if scheme == "embedding_align":
            with tempfile.TemporaryDirectory() as directory:
                checkpoint_path = str(Path(directory) / "transition.pt")
                save_two_stage_checkpoint(
                    checkpoint_path,
                    model=model,
                    ema=ema,
                    optimizer=None,
                    scheduler=None,
                    scaler=None,
                    config=config,
                    substage="transition",
                    completed_warmup_steps=2,
                    completed_joint_steps=0,
                    train_steps=2,
                    epoch=0,
                    include_optimizer=False,
                )
                resumed = _model(config)
                info = load_two_stage_resume(
                    resumed,
                    None,
                    checkpoint_path,
                    expected_substage="joint",
                    config=config,
                )
                self.assertEqual(info["resume_substage"], "joint")
                self.assertFalse(info["optimizer_loaded"])
                model = resumed
                ema = _ema(model)

        joint_optimizer = _optimizer(
            model,
            training_stage="real_finetune",
            action_mode="real",
            scheme=scheme,
            substage="joint",
        )
        joint = None
        for _ in range(2):
            joint, _ = _step(
                model=model,
                ema=ema,
                diffusion=diffusion,
                optimizer=joint_optimizer,
                batch=batch,
                config=config,
                scheme=scheme,
                substage="joint",
                check_gradients=True,
            )
        assert joint is not None
        return warmup, joint

    def test_reset_warmup_and_joint_real_gaussian_steps(self):
        warmup, joint = self._run_finetune_pair("reset")
        self.assertNotIn("alignment", warmup)
        self.assertNotIn("alignment", joint)
        self.assertTrue(warmup["gradient_check_performed"])
        self.assertTrue(joint["gradient_check_performed"])

    def test_embedding_align_warmup_transition_resume_and_joint(self):
        warmup, joint = self._run_finetune_pair("embedding_align")
        self.assertEqual(int(warmup["alignment_valid_count"]), 2)
        self.assertEqual(int(joint["alignment_valid_count"]), 2)
        self.assertEqual(float(warmup["diffusion"]), 0.0)
        self.assertTrue(bool(torch.isfinite(warmup["alignment"])))
        self.assertTrue(bool(torch.isfinite(joint["alignment"])))

    def test_real_to_latent_warmup_and_joint_without_latent_targets(self):
        warmup, joint = self._run_finetune_pair("real_to_latent")
        for logs in (warmup, joint):
            self.assertNotIn("alignment", logs)
            self.assertIn("predicted_latent/norm", logs)
            self.assertTrue(torch.isfinite(torch.as_tensor(logs["predicted_latent/norm"])))
            self.assertTrue(logs["gradient_check_performed"])

    def test_gaussian_kl_tuple_auxiliary_reaches_alignment_loss(self):
        config = _config(
            training_stage="real_finetune",
            action_mode="real",
            scheme="embedding_align",
        )
        model = _model(config)
        ema = _ema(model)
        optimizer = _optimizer(
            model,
            training_stage="real_finetune",
            action_mode="real",
            scheme="embedding_align",
            substage="joint",
        )
        logs, _ = _step(
            model=model,
            ema=ema,
            diffusion=_diffusion(use_kl=True),
            optimizer=optimizer,
            batch=_batch(),
            config=config,
            scheme="embedding_align",
            substage="joint",
            check_gradients=True,
        )
        # This fails with KeyError if Gaussian KL drops CDiT's tuple auxiliary.
        self.assertIn("alignment", logs)
        self.assertEqual(int(logs["alignment_valid_count"]), 2)
        self.assertTrue(bool(torch.isfinite(logs["loss"])))


if __name__ == "__main__":
    unittest.main()
