"""Focused CPU tests for proxy pretraining and two-step NWM finetuning.

The tests intentionally open CDiT's zero-initialized modulation/output gates.
Without that test-only change, a freshly initialized DiT produces exactly zero
and cannot demonstrate that condition gradients reach the action adapters.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import CDiT
from motion_condition import motion_offset_mask
from two_stage_nwm import (
    alignment_loss,
    assert_goal_alignment,
    build_optimizer_param_groups,
    configure_trainable_parameters,
    dense_motion_from_collated,
    flatten_goal_tensor,
    validate_two_stage_config,
)


MOTION_CONFIG = {
    "enabled": True,
    "available_types": ["none", "real", "geometry", "idm", "latent"],
    "train_types": ["none", "real", "geometry", "idm", "latent"],
    "adapter_hidden_dim": 12,
    "balance_parameter_count": False,
    "real": {
        "real_dim": 3,
        "normalization": {"mode": "identity"},
    },
    "geometry": {
        "geometry_dim": 3,
        "normalization": {"mode": "identity"},
    },
    "idm": {
        "idm_dim": 5,
        "normalization": {"mode": "identity"},
    },
    "latent": {
        "latent_dim": 4,
        "normalization": {"mode": "layer_norm"},
    },
}


def _open_condition_gates(model: CDiT) -> None:
    """Make a fresh tiny CDiT sensitive to its condition in a stable way."""
    generator = torch.Generator(device="cpu").manual_seed(20260901)
    with torch.no_grad():
        for block in model.blocks:
            block.adaLN_modulation[-1].weight.copy_(
                torch.randn(
                    block.adaLN_modulation[-1].weight.shape, generator=generator
                )
                * 0.05
            )
            block.adaLN_modulation[-1].bias.copy_(
                torch.randn(
                    block.adaLN_modulation[-1].bias.shape, generator=generator
                )
                * 0.05
            )
        model.final_layer.adaLN_modulation[-1].weight.copy_(
            torch.randn(
                model.final_layer.adaLN_modulation[-1].weight.shape,
                generator=generator,
            )
            * 0.05
        )
        model.final_layer.adaLN_modulation[-1].bias.copy_(
            torch.randn(
                model.final_layer.adaLN_modulation[-1].bias.shape,
                generator=generator,
            )
            * 0.05
        )
        model.final_layer.linear.weight.copy_(
            torch.randn(
                model.final_layer.linear.weight.shape, generator=generator
            )
            * 0.05
        )
        model.final_layer.linear.bias.zero_()


def _model(
    *,
    training_stage: str = "proxy_pretrain",
    action_mode: str = "latent",
    scheme: str = "reset",
    proxy_relative_time_mode: str = "always",
) -> CDiT:
    torch.manual_seed(101)
    model = CDiT(
        input_size=4,
        context_size=2,
        patch_size=2,
        in_channels=2,
        hidden_size=16,
        depth=1,
        num_heads=4,
        mlp_ratio=1.0,
        learn_sigma=False,
        motion_condition=MOTION_CONFIG,
        training_stage=training_stage,
        action_mode=action_mode,
        proxy_relative_time_mode=proxy_relative_time_mode,
        finetune={"scheme": scheme, "real_to_latent_hidden_dim": 8},
    ).cpu()
    _open_condition_gates(model)
    return model


def _inputs(batch_size: int = 4) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(303)
    return {
        "x": torch.randn(batch_size, 2, 4, 4, generator=generator),
        "t": torch.linspace(0, 999, batch_size),
        "x_cond": torch.randn(batch_size, 2, 2, 4, 4, generator=generator),
        "rel_t": torch.tensor([-0.5, -0.0625, 0.0625, 0.5])[:batch_size],
        "action": torch.randn(batch_size, 3, generator=generator),
        "latent": torch.randn(batch_size, 4, generator=generator),
        "target": torch.randn(batch_size, 2, 4, 4, generator=generator),
    }


def _diffusion_like_loss(
    model: CDiT,
    batch: dict[str, torch.Tensor],
    *,
    teacher_latent: torch.Tensor | None = None,
    teacher_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output, auxiliary = model(
        x=batch["x"],
        t=batch["t"],
        x_cond=batch["x_cond"],
        rel_t=batch["rel_t"],
        action=batch["action"],
        teacher_latent=teacher_latent,
        teacher_valid=teacher_valid,
        return_aux=True,
    )
    return F.mse_loss(output, batch["target"]), auxiliary


def _parameters_with_prefix(model: CDiT, prefix: str):
    return [(name, parameter) for name, parameter in model.named_parameters() if name.startswith(prefix)]


def _backbone_parameters(model: CDiT):
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("motion_condition_encoder.")
        and not name.startswith("real_to_latent.")
    ]


def _has_finite_nonzero_gradient(named_parameters) -> bool:
    return any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.abs().sum()) > 0.0
        for _, parameter in named_parameters
    )


def _assert_all_grad_none(test: unittest.TestCase, named_parameters) -> None:
    for name, parameter in named_parameters:
        test.assertIsNone(parameter.grad, name)


class ProxyConditionTests(unittest.TestCase):
    def test_fallback_relative_time_mode_is_latent_only(self):
        config = {
            "training_stage": "proxy_pretrain",
            "action_mode": "geometry",
            "proxy": {
                "type": "geometry",
                "dim": 3,
                "max_abs_frame_offset": 8,
                "relative_time_mode": "fallback",
                "use_precomputed_only": True,
            },
            "dataset": {"distance": {"min_dist_cat": -64, "max_dist_cat": 64}},
            "motion_condition": {
                "enabled": True,
                "geometry": {"geometry_dim": 3},
            },
        }
        with self.assertRaisesRegex(ValueError, "action_mode=latent"):
            validate_two_stage_config(config)

    def test_context_length_mismatch_fails_before_positional_addition(self):
        model = _model().eval()
        batch = _inputs()
        with self.assertRaisesRegex(
            ValueError, "context length does not match its positional embeddings"
        ):
            model(
                batch["x"],
                batch["t"],
                x_cond=batch["x_cond"][:, :1],
                rel_t=batch["rel_t"],
                conditioning_mode="none",
            )

    def test_proxy_offset_eligibility_is_inclusive_and_keeps_long_range(self):
        offsets = torch.tensor([-64, -9, -8, -1, 0, 1, 8, 9, 64])
        expected = torch.tensor([False, False, True, True, True, True, True, False, False])
        self.assertTrue(torch.equal(motion_offset_mask(offsets, 8), expected))
        self.assertEqual(int(offsets.min()), -64)
        self.assertEqual(int(offsets.max()), 64)

    def test_invalid_proxy_is_time_only_and_independent_of_placeholder(self):
        model = _model().eval()
        batch = _inputs()
        valid = torch.zeros(4, dtype=torch.bool)
        action_a = batch["latent"]
        action_b = action_a * 100000.0 + 777.0
        with torch.no_grad():
            output_a = model(
                batch["x"],
                batch["t"],
                x_cond=batch["x_cond"],
                rel_t=batch["rel_t"],
                action=action_a,
                action_valid=valid,
                conditioning_mode="latent",
            )
            output_b = model(
                batch["x"],
                batch["t"],
                x_cond=batch["x_cond"],
                rel_t=batch["rel_t"],
                action=action_b,
                action_valid=valid,
                conditioning_mode="latent",
            )
            output_none = model(
                batch["x"],
                batch["t"],
                x_cond=batch["x_cond"],
                rel_t=batch["rel_t"],
                conditioning_mode="none",
            )
        torch.testing.assert_close(output_a, output_b, rtol=0.0, atol=0.0)
        torch.testing.assert_close(output_a, output_none, rtol=0.0, atol=0.0)

    def test_latentonly_uses_latent_locally_and_time_as_fallback(self):
        model = _model(proxy_relative_time_mode="fallback").eval()
        generator = torch.Generator(device="cpu").manual_seed(404)
        diffusion = torch.randn(4, model.hidden_size, generator=generator)
        relative_time = torch.randn(4, model.hidden_size, generator=generator)
        latent = torch.randn(4, 4, generator=generator)
        valid = torch.tensor([True, False, True, False])

        condition, _ = model._compute_condition(
            diffusion,
            relative_time,
            action=latent,
            action_valid=valid,
            conditioning_mode="latent",
        )
        latent_embedding = model.motion_condition_encoder.encode("latent", latent)
        expected = diffusion + relative_time * (~valid).float().unsqueeze(-1)
        expected = expected + latent_embedding * valid.float().unsqueeze(-1)
        torch.testing.assert_close(condition, expected)

        changed_time = relative_time + 1000.0
        changed_condition, _ = model._compute_condition(
            diffusion,
            changed_time,
            action=latent,
            action_valid=valid,
            conditioning_mode="latent",
        )
        torch.testing.assert_close(condition[valid], changed_condition[valid])
        self.assertFalse(torch.equal(condition[~valid], changed_condition[~valid]))

    def test_latentonly_grouped_motion_matches_dense_conditioning(self):
        model = _model(proxy_relative_time_mode="fallback").eval()
        generator = torch.Generator(device="cpu").manual_seed(405)
        diffusion = torch.randn(4, model.hidden_size, generator=generator)
        relative_time = torch.randn(4, model.hidden_size, generator=generator)
        latent = torch.randn(4, 4, generator=generator)
        valid = torch.tensor([True, False, True, False])
        grouped = {
            "latent": {
                "indices": torch.tensor([0, 2], dtype=torch.int64),
                "values": latent[valid],
            }
        }
        dense_condition, _ = model._compute_condition(
            diffusion,
            relative_time,
            action=latent,
            action_valid=valid,
            conditioning_mode="latent",
        )
        grouped_condition, _ = model._compute_condition(
            diffusion,
            relative_time,
            motion=grouped,
            conditioning_mode="latent",
        )
        torch.testing.assert_close(dense_condition, grouped_condition)

    def test_timept_ignores_any_accidentally_supplied_action_tensor(self):
        model = _model(action_mode="none").eval()
        batch = _inputs()
        with torch.no_grad():
            without_action = model(
                batch["x"],
                batch["t"],
                x_cond=batch["x_cond"],
                rel_t=batch["rel_t"],
                conditioning_mode="none",
            )
            with_action_a = model(
                batch["x"],
                batch["t"],
                x_cond=batch["x_cond"],
                rel_t=batch["rel_t"],
                action=batch["latent"],
                action_valid=torch.ones(4, dtype=torch.bool),
                conditioning_mode="none",
            )
            with_action_b = model(
                batch["x"],
                batch["t"],
                x_cond=batch["x_cond"],
                rel_t=batch["rel_t"],
                action=batch["latent"] * -10000.0 + 321.0,
                action_valid=torch.zeros(4, dtype=torch.bool),
                conditioning_mode="none",
            )
        torch.testing.assert_close(without_action, with_action_a, rtol=0.0, atol=0.0)
        torch.testing.assert_close(without_action, with_action_b, rtol=0.0, atol=0.0)

    def test_stage1_proxy_rejects_mismatched_grouped_action_type(self):
        model = _model(action_mode="latent").eval()
        batch = _inputs()
        real_group = {
            "real": {
                "indices": torch.arange(4, dtype=torch.int64),
                "values": batch["action"],
            }
        }
        with self.assertRaisesRegex(
            ValueError, "Run real-action inference from a stage-2 checkpoint"
        ):
            model(
                batch["x"],
                batch["t"],
                x_cond=batch["x_cond"],
                rel_t=batch["rel_t"],
                motion=real_group,
            )

    def test_action_encoder_gradient_respects_post_encoder_mask(self):
        model = _model().train()
        batch = _inputs()
        latent_parameters = _parameters_with_prefix(
            model, "motion_condition_encoder.latent_"
        )

        model.zero_grad(set_to_none=True)
        output = model(
            batch["x"],
            batch["t"],
            x_cond=batch["x_cond"],
            rel_t=batch["rel_t"],
            action=batch["latent"],
            action_valid=torch.zeros(4, dtype=torch.bool),
            conditioning_mode="latent",
        )
        F.mse_loss(output, batch["target"]).backward()
        for name, parameter in latent_parameters:
            if parameter.grad is not None:
                self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)), name)
        self.assertTrue(_has_finite_nonzero_gradient(_backbone_parameters(model)))

        valid = torch.tensor([True, False, True, False])
        gradients = []
        for invalid_fill in (0.0, 12345.0):
            model.zero_grad(set_to_none=True)
            values = batch["latent"].clone()
            values[~valid] = invalid_fill
            output = model(
                batch["x"],
                batch["t"],
                x_cond=batch["x_cond"],
                rel_t=batch["rel_t"],
                action=values,
                action_valid=valid,
                conditioning_mode="latent",
            )
            F.mse_loss(output, batch["target"]).backward()
            gradients.append(
                [
                    None if parameter.grad is None else parameter.grad.detach().clone()
                    for _, parameter in latent_parameters
                ]
            )
        for first, second in zip(*gradients):
            if first is None or second is None:
                self.assertIs(first, second)
            else:
                torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertTrue(_has_finite_nonzero_gradient(latent_parameters))

    def test_multigoal_flatten_is_b_major_goal_minor_for_every_field(self):
        batch_size, num_goals = 2, 3
        target = torch.tensor([[[10.0], [11.0], [12.0]], [[20.0], [21.0], [22.0]]])
        rel_t = torch.tensor([[-0.5, -0.25, 0.0], [0.25, 0.5, 0.75]])
        offset = torch.tensor([[-64, -8, 0], [1, 8, 64]])
        action = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
        valid = torch.tensor([[False, True, True], [True, True, False]])

        flattened = {
            "target": flatten_goal_tensor(target, batch_size=batch_size, num_goals=num_goals, name="target"),
            "rel_t": flatten_goal_tensor(rel_t, batch_size=batch_size, num_goals=num_goals, name="rel_t"),
            "offset": flatten_goal_tensor(offset, batch_size=batch_size, num_goals=num_goals, name="frame_offset"),
            "action": flatten_goal_tensor(action, batch_size=batch_size, num_goals=num_goals, name="action"),
            "valid": flatten_goal_tensor(valid, batch_size=batch_size, num_goals=num_goals, name="proxy_valid"),
        }
        self.assertEqual(flattened["target"].squeeze(-1).tolist(), [10, 11, 12, 20, 21, 22])
        self.assertEqual(flattened["offset"].tolist(), [-64, -8, 0, 1, 8, 64])
        self.assertEqual(flattened["valid"].tolist(), [False, True, True, True, True, False])
        self.assertTrue(torch.equal(flattened["action"], action.reshape(6, 3)))

        dense, dense_valid = dense_motion_from_collated(
            {
                "real": {
                    "sample_indices": torch.tensor([0, 1]),
                    "values": action,
                    "masks": valid,
                }
            },
            motion_type="real",
            batch_size=batch_size,
            num_goals=num_goals,
            input_dim=3,
            device=torch.device("cpu"),
            require_all=False,
        )
        self.assertTrue(torch.equal(dense_valid, flattened["valid"]))
        self.assertTrue(torch.equal(dense[dense_valid], flattened["action"][flattened["valid"]]))
        assert_goal_alignment(
            target_batch=flattened["target"],
            rel_t=flattened["rel_t"],
            frame_offset=flattened["offset"],
            action=dense,
            action_valid=dense_valid,
        )


class AlignmentLossTests(unittest.TestCase):
    def test_alignment_uses_only_valid_local_pairs_and_empty_mask_is_finite(self):
        generator = torch.Generator().manual_seed(13)
        real = torch.randn(5, 8, generator=generator, requires_grad=True)
        target = torch.randn(5, 8, generator=generator)
        offsets = torch.tensor([-9, -8, 1, 8, 9])
        latent_valid = torch.tensor([True, True, False, True, True])
        valid = motion_offset_mask(offsets, 8) & latent_valid
        first = alignment_loss(
            real,
            target,
            valid,
            cosine_weight=0.75,
            l1_weight=0.25,
        )
        changed_target = target.clone()
        changed_target[~valid] = 100000.0
        second = alignment_loss(
            real,
            changed_target,
            valid,
            cosine_weight=0.75,
            l1_weight=0.25,
        )
        torch.testing.assert_close(first["alignment"], second["alignment"])
        self.assertEqual(int(first["alignment_valid_count"]), 2)

        empty_real = real.detach().clone().requires_grad_(True)
        empty = alignment_loss(
            empty_real,
            target,
            torch.zeros(5, dtype=torch.bool),
            cosine_weight=1.0,
            l1_weight=1.0,
        )
        self.assertTrue(bool(torch.isfinite(empty["alignment"])))
        self.assertEqual(float(empty["alignment"].detach()), 0.0)
        empty["alignment"].backward()
        self.assertTrue(torch.equal(empty_real.grad, torch.zeros_like(empty_real)))


class ResetFinetuneTests(unittest.TestCase):
    def test_reset_warmup_and_joint_trainability_and_gradients(self):
        batch = _inputs()
        model = _model(training_stage="real_finetune", action_mode="real", scheme="reset")

        report = configure_trainable_parameters(
            model,
            training_stage="real_finetune",
            action_mode="real",
            finetune_scheme="reset",
            finetune_substage="warmup",
        )
        self.assertTrue(report["trainable_names"])
        self.assertTrue(all(name.startswith("motion_condition_encoder.real_") for name in report["trainable_names"]))
        groups = build_optimizer_param_groups(
            model,
            training_stage="real_finetune",
            finetune_scheme="reset",
            finetune_substage="warmup",
            adapter_lr=1e-3,
            backbone_lr=1e-4,
        )
        self.assertEqual([group["name"] for group in groups], ["adapter"])
        model.zero_grad(set_to_none=True)
        loss, _ = _diffusion_like_loss(model, batch)
        loss.backward()
        self.assertTrue(_has_finite_nonzero_gradient(_parameters_with_prefix(model, "motion_condition_encoder.real_")))
        _assert_all_grad_none(self, _backbone_parameters(model))

        configure_trainable_parameters(
            model,
            training_stage="real_finetune",
            action_mode="real",
            finetune_scheme="reset",
            finetune_substage="joint",
        )
        model.zero_grad(set_to_none=True)
        loss, _ = _diffusion_like_loss(model, batch)
        loss.backward()
        self.assertTrue(_has_finite_nonzero_gradient(_parameters_with_prefix(model, "motion_condition_encoder.real_")))
        self.assertTrue(_has_finite_nonzero_gradient(_backbone_parameters(model)))
        _assert_all_grad_none(self, _parameters_with_prefix(model, "motion_condition_encoder.latent_"))


class AlignmentFinetuneTests(unittest.TestCase):
    def test_alignment_warmup_and_joint_use_real_condition_and_freeze_teacher(self):
        batch = _inputs()
        model = _model(
            training_stage="real_finetune",
            action_mode="real",
            scheme="embedding_align",
        )
        local_valid = torch.tensor([False, True, True, False])

        report = configure_trainable_parameters(
            model,
            training_stage="real_finetune",
            action_mode="real",
            finetune_scheme="embedding_align",
            finetune_substage="warmup",
        )
        self.assertTrue(all(name.startswith("motion_condition_encoder.real_") for name in report["trainable_names"]))
        model.zero_grad(set_to_none=True)
        auxiliary = model(
            None,
            None,
            action=batch["action"],
            teacher_latent=batch["latent"],
            teacher_valid=local_valid,
            alignment_only=True,
        )
        warmup_loss = alignment_loss(
            auxiliary["real_embedding"],
            auxiliary["target_embedding"],
            auxiliary["alignment_valid"],
            cosine_weight=1.0,
            l1_weight=0.5,
        )["alignment"]
        warmup_loss.backward()
        self.assertTrue(_has_finite_nonzero_gradient(_parameters_with_prefix(model, "motion_condition_encoder.real_")))
        _assert_all_grad_none(self, _parameters_with_prefix(model, "motion_condition_encoder.latent_"))
        _assert_all_grad_none(self, _backbone_parameters(model))

        configure_trainable_parameters(
            model,
            training_stage="real_finetune",
            action_mode="real",
            finetune_scheme="embedding_align",
            finetune_substage="joint",
        )
        model.zero_grad(set_to_none=True)
        diffusion, auxiliary = _diffusion_like_loss(
            model,
            batch,
            teacher_latent=batch["latent"],
            teacher_valid=local_valid,
        )
        align = alignment_loss(
            auxiliary["real_embedding"],
            auxiliary["target_embedding"],
            auxiliary["alignment_valid"],
            cosine_weight=1.0,
            l1_weight=0.5,
        )["alignment"]
        (diffusion + align).backward()
        self.assertTrue(_has_finite_nonzero_gradient(_parameters_with_prefix(model, "motion_condition_encoder.real_")))
        self.assertTrue(_has_finite_nonzero_gradient(_backbone_parameters(model)))
        _assert_all_grad_none(self, _parameters_with_prefix(model, "motion_condition_encoder.latent_"))

        model.eval()
        with torch.no_grad():
            prediction_a, _ = model(
                batch["x"],
                batch["t"],
                x_cond=batch["x_cond"],
                rel_t=batch["rel_t"],
                action=batch["action"],
                teacher_latent=batch["latent"],
                teacher_valid=local_valid,
                return_aux=True,
            )
            prediction_b, _ = model(
                batch["x"],
                batch["t"],
                x_cond=batch["x_cond"],
                rel_t=batch["rel_t"],
                action=batch["action"],
                teacher_latent=batch["latent"] * 999.0,
                teacher_valid=local_valid,
                return_aux=True,
            )
        # The teacher supplies only L_align; it never conditions diffusion.
        torch.testing.assert_close(prediction_a, prediction_b, rtol=0.0, atol=0.0)


class RealToLatentFinetuneTests(unittest.TestCase):
    def test_real_to_latent_warmup_and_joint_gradient_contract(self):
        batch = _inputs()
        model = _model(
            training_stage="real_finetune",
            action_mode="real",
            scheme="real_to_latent",
        )

        report = configure_trainable_parameters(
            model,
            training_stage="real_finetune",
            action_mode="real",
            finetune_scheme="real_to_latent",
            finetune_substage="warmup",
        )
        self.assertTrue(report["trainable_names"])
        self.assertTrue(all(name.startswith("real_to_latent.") for name in report["trainable_names"]))
        model.zero_grad(set_to_none=True)
        loss, auxiliary = _diffusion_like_loss(model, batch)
        self.assertIn("predicted_latent", auxiliary)
        self.assertNotIn("real_embedding", auxiliary)
        self.assertNotIn("alignment_valid", auxiliary)
        loss.backward()
        self.assertTrue(_has_finite_nonzero_gradient(_parameters_with_prefix(model, "real_to_latent.")))
        _assert_all_grad_none(self, _parameters_with_prefix(model, "motion_condition_encoder.latent_"))
        _assert_all_grad_none(self, _backbone_parameters(model))

        configure_trainable_parameters(
            model,
            training_stage="real_finetune",
            action_mode="real",
            finetune_scheme="real_to_latent",
            finetune_substage="joint",
        )
        model.zero_grad(set_to_none=True)
        loss, auxiliary = _diffusion_like_loss(model, batch)
        self.assertEqual(set(auxiliary), {"predicted_latent"})
        loss.backward()
        self.assertTrue(_has_finite_nonzero_gradient(_parameters_with_prefix(model, "real_to_latent.")))
        self.assertTrue(_has_finite_nonzero_gradient(_parameters_with_prefix(model, "motion_condition_encoder.latent_")))
        self.assertTrue(_has_finite_nonzero_gradient(_backbone_parameters(model)))
        _assert_all_grad_none(self, _parameters_with_prefix(model, "motion_condition_encoder.real_"))


if __name__ == "__main__":
    unittest.main()
