"""Focused regression tests for the state-conditioned latent controller."""

from __future__ import annotations

import copy
import unittest

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion import create_gaussian_diffusion
from models import CDiT
from two_stage_checkpoint import (
    TWO_STAGE_METADATA_KEY,
    build_checkpoint_metadata,
    load_stage1_weights,
    validate_resume_checkpoint,
)
from two_stage_nwm import (
    configure_trainable_parameters,
    latent_action_l2_loss,
    validate_two_stage_config,
)
from two_stage_training import (
    _make_optimizer,
    _make_scheduler,
    _phase_evaluation_due,
    two_stage_train_step,
)


MOTION_CONFIG = {
    "enabled": True,
    "available_types": ["none", "real", "latent"],
    "train_types": ["real"],
    "adapter_hidden_dim": 8,
    "balance_parameter_count": False,
    "real": {
        "real_dim": 3,
        "normalization": {
            "mode": "minmax",
            "min": [-2.5, -4.0, -3.141592653589793],
            "max": [5.0, 4.0, 3.141592653589793],
        },
    },
    "latent": {
        "latent_dim": 4,
        "normalization": {"mode": "layer_norm"},
    },
}


class _IdentityTokenizer:
    scaling_factor = 1.0

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return value


def _model(*, stage: str = "real_finetune", seed: int = 17) -> CDiT:
    torch.manual_seed(seed)
    finetune = {
        "scheme": "state_conditioned_controller",
        "controller_state_blocks": 2,
        "controller_mlp_ratio": 2.0,
    }
    return CDiT(
        input_size=2,
        context_size=2,
        patch_size=1,
        in_channels=1,
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=1.0,
        learn_sigma=False,
        motion_condition=copy.deepcopy(MOTION_CONFIG),
        training_stage=stage,
        action_mode="latent" if stage == "proxy_pretrain" else "real",
        finetune=finetune if stage == "real_finetune" else None,
    ).cpu()


def _config(*, stage: str = "real_finetune"):
    action_mode = "latent" if stage == "proxy_pretrain" else "real"
    return OmegaConf.create(
        {
            "seed": 0,
            "training_stage": stage,
            "action_mode": action_mode,
            "bfloat16": False,
            "eval_every": 5000,
            "eval_at_first_step": True,
            "dataset": {
                "context_size": 2,
                "distance": {"min_dist_cat": -64, "max_dist_cat": 64},
                "precomputed_latents": {"enabled": False},
            },
            "proxy": {
                "type": "latent",
                "dim": 4,
                "max_abs_frame_offset": 8,
                "storage_path": "/unused/unit-test-cache",
                "use_precomputed_only": True,
            },
            "motion_condition": copy.deepcopy(MOTION_CONFIG),
            "finetune": {
                "scheme": "state_conditioned_controller",
                "warmup_steps": 3000,
                "joint_steps": 200000,
                "controller_pretrain_lr": 1.0e-3,
                "controller_pretrain_weight_decay": 0.04,
                "controller_lr_warmup_steps": 300,
                "controller_eval_every": 1000,
                "joint_lr": 1.0e-4,
                "adapter_lr": 1.0e-4,
                "backbone_lr": 1.0e-4,
                "latent_l2_weight": 1.0,
                "controller_state_blocks": 2,
                "controller_mlp_ratio": 2.0,
            },
            "training": {
                "grad_clip_val": 0.0,
                "ema_decay": 0.9999,
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "lr": 1.0e-4,
                    "betas": [0.9, 0.999],
                    "weight_decay": 0.01,
                },
            },
            "model": {
                "generator": {
                    "_target_": "models.CDiT",
                    "hidden_size": 8,
                }
            },
        }
    )


def _diffusion():
    return create_gaussian_diffusion(
        timestep_respacing="",
        noise_schedule="squaredcos_cap_v2",
        use_kl=False,
        learn_sigma=False,
        diffusion_steps=12,
    )


class StateControllerModelTests(unittest.TestCase):
    def test_uses_only_last_context_frame_and_supports_grouped_inference(self):
        model = _model().eval()
        generator = torch.Generator().manual_seed(29)
        batch_size = 3
        context = torch.randn(batch_size, 2, 1, 2, 2, generator=generator)
        action = torch.randn(batch_size, 3, generator=generator)

        with torch.no_grad():
            first = model(
                None,
                None,
                x_cond=context,
                action=action,
                state_controller_only=True,
            )["predicted_latent"]
            changed_history = context.clone()
            changed_history[:, 0] += 1000.0
            second = model(
                None,
                None,
                x_cond=changed_history,
                action=action,
                state_controller_only=True,
            )["predicted_latent"]
            changed_state = context.clone()
            changed_state[:, -1] += 1.0
            third = model(
                None,
                None,
                x_cond=changed_state,
                action=action,
                state_controller_only=True,
            )["predicted_latent"]
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertFalse(torch.equal(first, third))

        x = torch.randn(batch_size, 1, 2, 2, generator=generator)
        timestep = torch.arange(batch_size, dtype=torch.float32)
        rel_t = torch.linspace(-0.5, 0.5, batch_size)
        grouped = {
            "real": {
                "indices": torch.arange(batch_size, dtype=torch.int64),
                "values": action,
            }
        }
        with torch.no_grad():
            dense_output, dense_aux = model(
                x,
                timestep,
                x_cond=context,
                rel_t=rel_t,
                action=action,
                return_aux=True,
            )
            grouped_output, grouped_aux = model(
                x,
                timestep,
                x_cond=context,
                rel_t=rel_t,
                motion=grouped,
                return_aux=True,
            )
        torch.testing.assert_close(dense_output, grouped_output)
        torch.testing.assert_close(
            dense_aux["predicted_latent"], grouped_aux["predicted_latent"]
        )
        torch.testing.assert_close(first, dense_aux["predicted_latent"])

    def test_exact_freezing_and_phase_optimizer_contract(self):
        config = _config()
        model = _model()
        warmup = configure_trainable_parameters(
            model,
            training_stage="real_finetune",
            action_mode="real",
            finetune_scheme="state_conditioned_controller",
            finetune_substage="warmup",
        )
        self.assertTrue(warmup["trainable_names"])
        self.assertTrue(
            all(
                name.startswith("state_conditioned_controller.")
                for name in warmup["trainable_names"]
            )
        )
        optimizer = _make_optimizer(
            config,
            model,
            training_stage="real_finetune",
            scheme="state_conditioned_controller",
            substage="warmup",
        )
        scheduler = _make_scheduler(
            config,
            optimizer,
            training_stage="real_finetune",
            scheme="state_conditioned_controller",
            substage="warmup",
        )
        self.assertEqual(optimizer.param_groups[0]["initial_lr"], 1.0e-3)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.04)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0e-3 / 300)
        self.assertIsNotNone(scheduler)

        joint = configure_trainable_parameters(
            model,
            training_stage="real_finetune",
            action_mode="real",
            finetune_scheme="state_conditioned_controller",
            finetune_substage="joint",
        )
        trainability = dict(model.named_parameters())
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in trainability.items()
                if name.startswith("x_embedder.")
                or name.startswith("motion_condition_encoder.")
            )
        )
        self.assertIn("pos_embed", joint["trainable_names"])
        joint_optimizer = _make_optimizer(
            config,
            model,
            training_stage="real_finetune",
            scheme="state_conditioned_controller",
            substage="joint",
        )
        self.assertEqual(
            [group["name"] for group in joint_optimizer.param_groups],
            ["adapter", "backbone"],
        )
        self.assertTrue(
            all(group["lr"] == 1.0e-4 for group in joint_optimizer.param_groups)
        )
        self.assertTrue(
            all(group["weight_decay"] == 0.01 for group in joint_optimizer.param_groups)
        )
        self.assertIsNone(
            _make_scheduler(
                config,
                joint_optimizer,
                training_stage="real_finetune",
                scheme="state_conditioned_controller",
                substage="joint",
            )
        )


class StateControllerLossAndTrainerTests(unittest.TestCase):
    def test_latent_l2_ignores_invalid_targets(self):
        prediction = torch.randn(5, 4, requires_grad=True)
        target = torch.randn(5, 4)
        valid = torch.tensor([True, False, True, False, False])
        first = latent_action_l2_loss(prediction, target, valid)["latent_l2"]
        changed = target.clone()
        changed[~valid] += 100000.0
        second = latent_action_l2_loss(prediction, changed, valid)["latent_l2"]
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        first.backward()
        self.assertTrue(torch.equal(prediction.grad[~valid], torch.zeros(3, 4)))
        self.assertGreater(float(prediction.grad[valid].abs().sum()), 0.0)

    def test_controller_only_then_joint_production_steps(self):
        config = _config()
        model = _model().train()
        ema = copy.deepcopy(model).eval()
        for parameter in ema.parameters():
            parameter.requires_grad_(False)

        generator = torch.Generator().manual_seed(47)
        base_batch = {
            "video": torch.randn(2, 4, 1, 2, 2, generator=generator),
            "k": torch.tensor([[-0.0625, 0.0625], [-0.03125, 0.03125]]),
            "real_action": torch.randn(2, 2, 3, generator=generator),
            "teacher_latent": torch.randn(2, 2, 4, generator=generator),
            "latent_valid": torch.ones(2, 2, dtype=torch.bool),
        }

        configure_trainable_parameters(
            model,
            training_stage="real_finetune",
            action_mode="real",
            finetune_scheme="state_conditioned_controller",
            finetune_substage="warmup",
        )
        warmup_optimizer = _make_optimizer(
            config,
            model,
            training_stage="real_finetune",
            scheme="state_conditioned_controller",
            substage="warmup",
        )
        warmup_batch = dict(base_batch)
        warmup_batch["frame_offset"] = torch.tensor([[-8, 8], [-4, 4]])
        warmup_logs, _ = two_stage_train_step(
            model=model,
            ema=ema,
            diffusion=_diffusion(),
            tokenizer=_IdentityTokenizer(),
            batch=warmup_batch,
            device=torch.device("cpu"),
            optimizer=warmup_optimizer,
            scheduler=None,
            scaler=None,
            config=config,
            training_stage="real_finetune",
            scheme="state_conditioned_controller",
            substage="warmup",
            check_gradients=True,
        )
        self.assertEqual(float(warmup_logs["diffusion"]), 0.0)
        self.assertEqual(int(warmup_logs["latent_l2_valid_count"]), 4)
        self.assertTrue(warmup_logs["gradient_check_performed"])

        configure_trainable_parameters(
            model,
            training_stage="real_finetune",
            action_mode="real",
            finetune_scheme="state_conditioned_controller",
            finetune_substage="joint",
        )
        joint_optimizer = _make_optimizer(
            config,
            model,
            training_stage="real_finetune",
            scheme="state_conditioned_controller",
            substage="joint",
        )
        joint_batch = dict(base_batch)
        joint_batch["frame_offset"] = torch.tensor([[-8, 64], [1, -64]])
        joint_logs, _ = two_stage_train_step(
            model=model,
            ema=ema,
            diffusion=_diffusion(),
            tokenizer=_IdentityTokenizer(),
            batch=joint_batch,
            device=torch.device("cpu"),
            optimizer=joint_optimizer,
            scheduler=None,
            scaler=None,
            config=config,
            training_stage="real_finetune",
            scheme="state_conditioned_controller",
            substage="joint",
            check_gradients=True,
        )
        self.assertTrue(torch.isfinite(joint_logs["diffusion"]))
        self.assertEqual(int(joint_logs["latent_l2_valid_count"]), 2)
        self.assertTrue(joint_logs["gradient_check_performed"])
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if name.startswith("x_embedder.")
                or name.startswith("motion_condition_encoder.latent_")
            )
        )


class StateControllerIntegrationTests(unittest.TestCase):
    def test_overlay_and_phase_local_eval_schedule(self):
        from pathlib import Path

        config_dir = Path(__file__).resolve().parents[1] / "conf"
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            config = compose(
                config_name="nwm",
                overrides=["two_stage=latent_state_controller"],
            )
        self.assertEqual(
            validate_two_stage_config(config),
            ("real_finetune", "real", "state_conditioned_controller"),
        )
        self.assertEqual(config.finetune.warmup_steps, 3000)
        self.assertEqual(config.finetune.joint_steps, 200000)
        self.assertEqual(config.finetune.joint_lr, 1.0e-4)
        for step in (1, 1000, 2000, 3000):
            self.assertTrue(
                _phase_evaluation_due(
                    training_stage="real_finetune",
                    scheme="state_conditioned_controller",
                    substage="warmup",
                    current_steps=step,
                    global_steps=step,
                    config=config,
                )
            )
        self.assertFalse(
            _phase_evaluation_due(
                training_stage="real_finetune",
                scheme="state_conditioned_controller",
                substage="warmup",
                current_steps=999,
                global_steps=999,
                config=config,
            )
        )
        self.assertTrue(
            _phase_evaluation_due(
                training_stage="real_finetune",
                scheme="state_conditioned_controller",
                substage="joint",
                current_steps=5000,
                global_steps=8000,
                config=config,
            )
        )

    def test_stage1_load_allows_only_new_controller(self):
        source = _model(stage="proxy_pretrain", seed=71)
        target = _model(stage="real_finetune", seed=73)
        source_config = _config(stage="proxy_pretrain")
        target_config = _config(stage="real_finetune")
        checkpoint = {
            "model": copy.deepcopy(source.state_dict()),
            TWO_STAGE_METADATA_KEY: build_checkpoint_metadata(
                source_config,
                "pretrain",
                completed_warmup_steps=0,
                completed_joint_steps=0,
                model=source,
            ),
        }
        controller_before = {
            key: value.clone()
            for key, value in target.state_dict().items()
            if key.startswith("state_conditioned_controller.")
        }
        summary = load_stage1_weights(target, checkpoint, target_config)
        self.assertTrue(
            all(
                key.startswith("state_conditioned_controller.")
                for key in summary["expected_missing_keys"]
            )
        )
        for key, value in controller_before.items():
            torch.testing.assert_close(target.state_dict()[key], value)
        for key, value in source.state_dict().items():
            if key.startswith("motion_condition_encoder.latent_"):
                torch.testing.assert_close(target.state_dict()[key], value)

    def test_resume_metadata_locks_controller_contract(self):
        model = _model()
        config = _config()
        metadata = build_checkpoint_metadata(
            config,
            "warmup",
            completed_warmup_steps=10,
            completed_joint_steps=0,
            model=model,
        )
        validated = validate_resume_checkpoint(
            metadata,
            config=config,
            model=model,
            expected_substage="warmup",
        )
        self.assertEqual(validated["completed_warmup_steps"], 10)
        changed = copy.deepcopy(config)
        changed.finetune.latent_l2_weight = 2.0
        with self.assertRaisesRegex(ValueError, "state_controller mismatch"):
            validate_resume_checkpoint(
                metadata,
                config=changed,
                model=model,
                expected_substage="warmup",
            )


if __name__ == "__main__":
    unittest.main()
