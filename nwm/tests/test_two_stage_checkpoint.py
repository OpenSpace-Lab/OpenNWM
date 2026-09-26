"""Unit tests for the opt-in two-stage checkpoint contract."""

from __future__ import annotations

import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from two_stage_checkpoint import (
    TWO_STAGE_METADATA_KEY,
    build_checkpoint_metadata,
    build_data_resume_fingerprint,
    capture_rng_state,
    load_stage1_weights,
    load_two_stage_resume,
    restore_rng_state,
    validate_data_resume_fingerprint,
    validate_resume_checkpoint,
    validate_stage1_checkpoint,
)
from two_stage_training import (
    _atomic_update_latest_checkpoint,
    _evaluation_due,
    _load_ema_and_training_state,
    _periodic_checkpoint_due,
    _phase_progress_after_iteration,
    _prepare_loader,
    _retained_stage1_checkpoint_path,
    _retained_stage2_checkpoint_path,
    _restore_completed_phase_rng,
    _save_training_checkpoint,
    _should_check_first_step_gradients,
    save_two_stage_checkpoint,
)
from train_utils import validate_model_context_sizes


class _TinyMotionEncoder(nn.Module):
    def __init__(self, latent_dim: int = 4, hidden_dim: int = 6):
        super().__init__()
        self.input_dims = {"real": 3, "latent": latent_dim}
        self.real_action_adapter = nn.Linear(3, hidden_dim)
        self.latent_action_adapter = nn.Linear(latent_dim, hidden_dim)


class _TinyTwoStageModel(nn.Module):
    def __init__(self, latent_dim: int = 4, hidden_dim: int = 6, context_size: int = 2):
        super().__init__()
        self.hidden_size = hidden_dim
        self.context_size = context_size
        self.backbone = nn.Linear(hidden_dim, hidden_dim)
        self.motion_condition_encoder = _TinyMotionEncoder(latent_dim, hidden_dim)
        self.real_to_latent = nn.Linear(3, latent_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.backbone(value)


class _StatefulScaler:
    def __init__(self, state: dict | None = None):
        self.state = copy.deepcopy(state or {})

    def state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, state):
        self.state = copy.deepcopy(state)


class Stage2CheckpointPublicationTests(unittest.TestCase):
    def test_stage2_eval_schedule_matches_legacy_global_steps(self):
        self.assertTrue(
            _evaluation_due(
                global_steps=1,
                interval=5000,
                eval_at_first_step=True,
            )
        )
        self.assertTrue(
            _evaluation_due(
                global_steps=5000,
                interval=5000,
                eval_at_first_step=True,
            )
        )
        self.assertFalse(
            _evaluation_due(
                global_steps=5001,
                interval=5000,
                eval_at_first_step=True,
            )
        )
        self.assertFalse(
            _evaluation_due(
                global_steps=1,
                interval=0,
                eval_at_first_step=False,
            )
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _evaluation_due(
                global_steps=-1,
                interval=5000,
                eval_at_first_step=True,
            )

    def test_finetune_loader_builds_a_separate_pixel_eval_loader(self):
        config = SimpleNamespace(training_stage="real_finetune")
        with mock.patch(
            "two_stage_training.prepare_datasets",
            return_value=("latent-train-dataset", "pixel-eval-dataset"),
        ) as prepare, mock.patch(
            "two_stage_training.create_dataloader",
            side_effect=[
                ("train-loader", "train-sampler"),
                ("eval-loader", "eval-sampler"),
            ],
        ) as create:
            result = _prepare_loader(
                config,
                rank=3,
                substage="joint",
                include_eval=True,
            )

        self.assertEqual(
            result,
            (
                "train-loader",
                "train-sampler",
                "latent-train-dataset",
                "eval-loader",
                "pixel-eval-dataset",
            ),
        )
        prepare.assert_called_once_with(
            config,
            finetune_substage="joint",
            include_test=True,
        )
        self.assertEqual(create.call_count, 2)
        self.assertTrue(create.call_args_list[0].kwargs["is_train"])
        self.assertFalse(create.call_args_list[1].kwargs["is_train"])

    def test_stage1_loader_builds_navanywhere_validation(self):
        config = SimpleNamespace(training_stage="proxy_pretrain")
        with mock.patch(
            "data_utils.prepare_proxy_pretrain_dataset",
            return_value="nav-train-dataset",
        ), mock.patch(
            "data_utils.prepare_proxy_pretrain_validation_dataset",
            return_value="nav-eval-dataset",
        ), mock.patch(
            "two_stage_training.create_dataloader",
            side_effect=[
                ("train-loader", "train-sampler"),
                ("nav-eval-loader", "nav-eval-sampler"),
            ],
        ) as create:
            result = _prepare_loader(
                config,
                rank=2,
                substage="pretrain",
                include_eval=True,
            )

        self.assertEqual(
            result,
            (
                "train-loader",
                "train-sampler",
                "nav-train-dataset",
                "nav-eval-loader",
                "nav-eval-dataset",
            ),
        )
        self.assertEqual(create.call_count, 2)
        self.assertFalse(create.call_args_list[1].kwargs["is_train"])

    def test_stage2_checkpoint_names_include_substage_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                Path(
                    _retained_stage2_checkpoint_path(
                        directory, "warmup", 10_000
                    )
                ).name,
                "warmup_0010000.pth.tar",
            )
            self.assertEqual(
                Path(
                    _retained_stage2_checkpoint_path(
                        directory, "joint", 100_000
                    )
                ).name,
                "joint_0100000.pth.tar",
            )
            self.assertEqual(
                Path(
                    _retained_stage1_checkpoint_path(directory, 10_000)
                ).name,
                "pretrain_0010000.pth.tar",
            )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            _retained_stage2_checkpoint_path(".", "transition", 10_000)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _retained_stage2_checkpoint_path(".", "joint", -1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _retained_stage1_checkpoint_path(".", -1)

    def test_stage2_periodicity_uses_substage_steps_and_leaves_final_save_once(
        self,
    ):
        self.assertTrue(
            _periodic_checkpoint_due(
                training_stage="real_finetune",
                current_steps=10,
                global_steps=13,
                target_steps=100,
                interval=10,
            )
        )
        self.assertFalse(
            _periodic_checkpoint_due(
                training_stage="proxy_pretrain",
                current_steps=100,
                global_steps=100,
                target_steps=100,
                interval=10,
            )
        )
        self.assertFalse(
            _periodic_checkpoint_due(
                training_stage="real_finetune",
                current_steps=7,
                global_steps=20,
                target_steps=100,
                interval=10,
            )
        )
        self.assertFalse(
            _periodic_checkpoint_due(
                training_stage="real_finetune",
                current_steps=100,
                global_steps=110,
                target_steps=100,
                interval=10,
            )
        )
        self.assertTrue(
            _periodic_checkpoint_due(
                training_stage="proxy_pretrain",
                current_steps=7,
                global_steps=20,
                target_steps=100,
                interval=10,
            )
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            _periodic_checkpoint_due(
                training_stage="real_finetune",
                current_steps=1,
                global_steps=1,
                target_steps=100,
                interval=0,
            )

    def test_latest_checkpoint_is_an_atomically_replaced_relative_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            latest = checkpoint_dir / "latest.pth.tar"
            warmup = checkpoint_dir / "warmup_0010000.pth.tar"
            joint = checkpoint_dir / "joint_0010000.pth.tar"
            latest.write_text("legacy", encoding="utf-8")
            warmup.write_text("warmup", encoding="utf-8")
            joint.write_text("joint", encoding="utf-8")

            _atomic_update_latest_checkpoint(str(warmup), str(latest))
            self.assertTrue(latest.is_symlink())
            self.assertEqual(latest.readlink(), Path(warmup.name))
            self.assertEqual(latest.read_text(encoding="utf-8"), "warmup")

            _atomic_update_latest_checkpoint(str(joint), str(latest))
            self.assertTrue(latest.is_symlink())
            self.assertEqual(latest.readlink(), Path(joint.name))
            self.assertEqual(latest.read_text(encoding="utf-8"), "joint")
            self.assertEqual(
                list(checkpoint_dir.glob(".latest.pth.tar.*.tmp")),
                [],
            )

    def test_both_stages_retain_numbered_files_and_update_latest(self):
        stage2_config = OmegaConf.create(
            _config(
                stage="real_finetune",
                action_mode="real",
                scheme="reset",
                proxy_type="temporal_distance",
            )
        )
        stage2_model = _TinyTwoStageModel()
        with tempfile.TemporaryDirectory() as directory:
            retained = Path(
                _save_training_checkpoint(
                    directory,
                    training_stage="real_finetune",
                    substage="joint",
                    current_steps=10_000,
                    model=stage2_model,
                    ema=copy.deepcopy(stage2_model),
                    optimizer=None,
                    scheduler=None,
                    scaler=None,
                    config=stage2_config,
                    completed_warmup_steps=10_000,
                    completed_joint_steps=10_000,
                    train_steps=20_000,
                    epoch=1,
                    include_optimizer=False,
                )
            )
            latest = Path(directory) / "latest.pth.tar"
            self.assertEqual(retained.name, "joint_0010000.pth.tar")
            self.assertTrue(retained.is_file())
            self.assertTrue(latest.is_symlink())
            self.assertEqual(latest.resolve(), retained.resolve())

        stage1_config = OmegaConf.create(
            _config(stage="proxy_pretrain", action_mode="latent")
        )
        stage1_model = _TinyTwoStageModel()
        with tempfile.TemporaryDirectory() as directory:
            saved = Path(
                _save_training_checkpoint(
                    directory,
                    training_stage="proxy_pretrain",
                    substage="pretrain",
                    current_steps=10_000,
                    model=stage1_model,
                    ema=copy.deepcopy(stage1_model),
                    optimizer=None,
                    scheduler=None,
                    scaler=None,
                    config=stage1_config,
                    completed_warmup_steps=0,
                    completed_joint_steps=0,
                    train_steps=10_000,
                    epoch=1,
                    include_optimizer=False,
                )
            )
            latest = Path(directory) / "latest.pth.tar"
            self.assertEqual(saved.name, "pretrain_0010000.pth.tar")
            self.assertTrue(saved.is_file())
            self.assertTrue(latest.is_symlink())
            self.assertEqual(latest.resolve(), saved.resolve())
            self.assertEqual(
                list(Path(directory).glob("pretrain_*.pth.tar")), [saved]
            )


class InferenceContextValidationTests(unittest.TestCase):
    def test_matching_context_sizes_are_accepted(self):
        config = OmegaConf.create(
            {
                "dataset": {"context_size": 4},
                "eval_context_size": 4,
                "trajectory_eval_context_size": 4,
            }
        )
        self.assertEqual(
            validate_model_context_sizes(
                config, "eval_context_size", "trajectory_eval_context_size"
            ),
            4,
        )

    def test_context_override_mismatch_is_actionable(self):
        config = OmegaConf.create(
            {"dataset": {"context_size": 4}, "eval_context_size": 2}
        )
        with self.assertRaisesRegex(
            ValueError,
            "dataset.context_size=4, eval_context_size=2",
        ):
            validate_model_context_sizes(config, "eval_context_size")


def _config(
    *,
    stage: str,
    action_mode: str,
    scheme: str = "reset",
    latent_dim: int = 4,
    proxy_type: str | None = None,
    proxy_window: int = 8,
    warmup_steps: int | None = None,
    joint_steps: int | None = None,
    relative_time_mode: str = "always",
) -> dict:
    proxy_type = action_mode if proxy_type is None else proxy_type
    finetune = {"scheme": scheme}
    if warmup_steps is not None:
        finetune["warmup_steps"] = warmup_steps
    if joint_steps is not None:
        finetune["joint_steps"] = joint_steps
    return {
        "seed": 1701,
        "training_stage": stage,
        "action_mode": action_mode,
        "training": {"batch_size": 2, "num_workers": 0},
        "proxy": {
            "type": proxy_type,
            "dim": latent_dim if proxy_type == "latent" else (0 if proxy_type == "none" else 3),
            "max_abs_frame_offset": proxy_window,
            "relative_time_mode": relative_time_mode,
            "normalization_path": "stats/navanywhere-1000h.json",
        },
        "finetune": finetune,
        "dataset": {"context_size": 2},
        "model": {"generator": {"_target_": "models.TinyCDiT-B"}},
        "motion_condition": {
            "latent": {
                "latent_dim": latent_dim,
                "normalization": {
                    "mode": "layer_norm",
                    "identifier": "navanywhere-1000h-v1",
                },
            }
        },
    }


def _stage1_checkpoint(
    action_mode: str = "latent",
    *,
    latent_dim: int = 4,
    compiled_prefix: bool = False,
) -> tuple[_TinyTwoStageModel, dict]:
    model = _TinyTwoStageModel(latent_dim=latent_dim)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters(), start=1):
            parameter.fill_(index / 10.0)
    metadata = build_checkpoint_metadata(
        _config(
            stage="proxy_pretrain",
            action_mode=action_mode,
            latent_dim=latent_dim,
        ),
        finetune_substage="pretrain",
        completed_warmup_steps=0,
        completed_joint_steps=0,
        model=model,
    )
    state = model.state_dict()
    if compiled_prefix:
        state = {f"_orig_mod.{key}": value for key, value in state.items()}
    return model, {"model": state, TWO_STAGE_METADATA_KEY: metadata}


class Stage1LoadingTests(unittest.TestCase):
    def test_metadata_contains_required_contract(self):
        model, checkpoint = _stage1_checkpoint()
        metadata = checkpoint[TWO_STAGE_METADATA_KEY]
        self.assertEqual(metadata["training_stage"], "proxy_pretrain")
        self.assertEqual(metadata["action_mode"], "latent")
        self.assertEqual(metadata["proxy_type"], "latent")
        self.assertEqual(metadata["proxy_dim"], 4)
        self.assertEqual(metadata["proxy_max_abs_frame_offset"], 8)
        self.assertEqual(metadata["proxy_relative_time_mode"], "always")
        self.assertEqual(metadata["latent_dim"], 4)
        self.assertEqual(metadata["hidden_dim"], model.hidden_size)
        self.assertEqual(metadata["context_size"], model.context_size)
        self.assertIsNone(metadata["finetune_scheme"])
        self.assertIn("identifier", metadata["latent_normalization"])
        self.assertIn("path", metadata["latent_normalization"])
        self.assertIn("statistics", metadata["latent_normalization"])

    def test_latent_checkpoint_loads_reset_alignment_and_real_to_latent(self):
        source, checkpoint = _stage1_checkpoint(compiled_prefix=True)
        for scheme in ("reset", "embedding_align", "real_to_latent"):
            with self.subTest(scheme=scheme):
                torch.manual_seed(13)
                target = _TinyTwoStageModel()
                real_before = copy.deepcopy(
                    target.motion_condition_encoder.real_action_adapter.state_dict()
                )
                generator_before = copy.deepcopy(target.real_to_latent.state_dict())
                summary = load_stage1_weights(
                    target,
                    checkpoint,
                    _config(
                        stage="real_finetune",
                        action_mode="real",
                        scheme=scheme,
                        proxy_type="latent",
                    ),
                )
                torch.testing.assert_close(
                    target.backbone.weight, source.backbone.weight
                )
                torch.testing.assert_close(
                    target.motion_condition_encoder.latent_action_adapter.weight,
                    source.motion_condition_encoder.latent_action_adapter.weight,
                )
                if scheme in {"reset", "embedding_align"}:
                    for key, value in real_before.items():
                        torch.testing.assert_close(
                            target.motion_condition_encoder.real_action_adapter.state_dict()[
                                key
                            ],
                            value,
                        )
                    self.assertTrue(
                        any("real_action_adapter" in key for key in summary["skipped_keys"])
                    )
                else:
                    for key, value in generator_before.items():
                        torch.testing.assert_close(target.real_to_latent.state_dict()[key], value)
                    self.assertTrue(
                        any("real_to_latent" in key for key in summary["skipped_keys"])
                    )

    def test_nonlatent_checkpoint_is_rejected_for_b_and_c(self):
        for action_mode in ("none", "geometry", "idm"):
            _, checkpoint = _stage1_checkpoint(action_mode=action_mode)
            for scheme in ("embedding_align", "real_to_latent"):
                with self.subTest(action_mode=action_mode, scheme=scheme):
                    with self.assertRaisesRegex(ValueError, "NWM-LatentPT"):
                        validate_stage1_checkpoint(
                            checkpoint[TWO_STAGE_METADATA_KEY],
                            _config(
                                stage="real_finetune",
                                action_mode="real",
                                scheme=scheme,
                                proxy_type="latent",
                            ),
                            model=_TinyTwoStageModel(),
                        )

    def test_reset_accepts_all_matching_stage1_sources(self):
        for source_type in ("none", "geometry", "idm", "latent"):
            with self.subTest(source_type=source_type):
                _, checkpoint = _stage1_checkpoint(action_mode=source_type)
                target = _TinyTwoStageModel()
                summary = load_stage1_weights(
                    target,
                    checkpoint,
                    _config(
                        stage="real_finetune",
                        action_mode="real",
                        scheme="reset",
                        proxy_type=source_type,
                    ),
                )
                self.assertEqual(summary["source_action_mode"], source_type)

    def test_reset_rejects_wrong_configured_stage1_source(self):
        _, checkpoint = _stage1_checkpoint(action_mode="geometry")
        with self.assertRaisesRegex(ValueError, "proxy_type mismatch"):
            validate_stage1_checkpoint(
                checkpoint[TWO_STAGE_METADATA_KEY],
                _config(
                    stage="real_finetune",
                    action_mode="real",
                    scheme="reset",
                    proxy_type="idm",
                ),
                model=_TinyTwoStageModel(),
            )

    def test_model_and_context_metadata_mismatches_are_rejected(self):
        _, checkpoint = _stage1_checkpoint()
        for field, bad_value in (
            ("hidden_dim", 99),
            ("model_target", "models.OtherCDiT"),
            ("context_size", 7),
        ):
            with self.subTest(field=field):
                corrupted = copy.deepcopy(checkpoint[TWO_STAGE_METADATA_KEY])
                corrupted[field] = bad_value
                with self.assertRaisesRegex(ValueError, f"{field} mismatch"):
                    validate_stage1_checkpoint(
                        corrupted,
                        _config(
                            stage="real_finetune",
                            action_mode="real",
                            scheme="embedding_align",
                            proxy_type="latent",
                        ),
                        model=_TinyTwoStageModel(),
                    )

    def test_latent_dimension_mismatch_has_clear_error(self):
        _, checkpoint = _stage1_checkpoint(latent_dim=4)
        with self.assertRaisesRegex(ValueError, "latent_dim mismatch"):
            validate_stage1_checkpoint(
                checkpoint[TWO_STAGE_METADATA_KEY],
                _config(
                    stage="real_finetune",
                    action_mode="real",
                    scheme="embedding_align",
                    latent_dim=5,
                    proxy_type="latent",
                ),
                model=_TinyTwoStageModel(latent_dim=5),
            )

    def test_unexpected_missing_key_is_not_silenced(self):
        _, checkpoint = _stage1_checkpoint()
        del checkpoint["model"]["backbone.bias"]
        with self.assertRaisesRegex(RuntimeError, "missing=.*backbone.bias"):
            load_stage1_weights(
                _TinyTwoStageModel(),
                checkpoint,
                _config(
                    stage="real_finetune",
                    action_mode="real",
                    scheme="embedding_align",
                    proxy_type="latent",
                ),
            )


class ResumeTests(unittest.TestCase):
    @staticmethod
    def _optimizer_with_state(model: nn.Module) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        model(torch.ones(2, model.hidden_size)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return optimizer

    def _resume_checkpoint(self, substage: str, include_optimizer: bool = True):
        model = _TinyTwoStageModel()
        optimizer = self._optimizer_with_state(model)
        metadata = build_checkpoint_metadata(
            _config(
                stage="real_finetune",
                action_mode="real",
                scheme="embedding_align",
                proxy_type="latent",
            ),
            finetune_substage=substage,
            completed_warmup_steps=7,
            completed_joint_steps=3 if substage == "joint" else 0,
            model=model,
        )
        checkpoint = {
            "model": model.state_dict(),
            TWO_STAGE_METADATA_KEY: metadata,
            "epoch": 2,
            "train_steps": 10,
        }
        if include_optimizer:
            checkpoint["opt"] = optimizer.state_dict()
        return model, checkpoint

    def test_warmup_and_joint_resume_restore_optimizer(self):
        for substage in ("warmup", "joint"):
            with self.subTest(substage=substage):
                source, checkpoint = self._resume_checkpoint(substage)
                target = _TinyTwoStageModel()
                optimizer = torch.optim.AdamW(target.parameters(), lr=1e-2)
                config = _config(
                    stage="real_finetune",
                    action_mode="real",
                    scheme="embedding_align",
                    proxy_type="latent",
                )
                summary = load_two_stage_resume(
                    target,
                    optimizer,
                    checkpoint,
                    expected_substage=substage,
                    config=config,
                )
                self.assertTrue(summary["optimizer_loaded"])
                self.assertEqual(summary["finetune_substage"], substage)
                torch.testing.assert_close(target.backbone.weight, source.backbone.weight)
                self.assertEqual(optimizer.param_groups[0]["lr"], 3e-4)

    def test_saved_batch_cursor_runtime_state_and_rng_restore_exactly(self):
        config = OmegaConf.create(
            _config(
                stage="real_finetune",
                action_mode="real",
                scheme="embedding_align",
                proxy_type="latent",
                warmup_steps=3,
            )
        )
        source = _TinyTwoStageModel()
        ema = copy.deepcopy(source)
        optimizer = self._optimizer_with_state(source)

        random.seed(1201)
        np.random.seed(1202)
        torch.manual_seed(1203)
        saved_rng_state = capture_rng_state()
        expected_python = [random.random() for _ in range(4)]
        expected_numpy = np.random.random_sample(4)
        expected_torch = torch.rand(4)

        runtime_state = {
            "rng_state": saved_rng_state,
            "proxy_metrics": {
                "total_samples": 17.0,
                "eligible_samples": 9.0,
                "used_samples": 7.0,
            },
            "valid_alignment_batches": 2,
        }
        data_fingerprint = build_data_resume_fingerprint(
            config,
            "warmup",
            dataset_length=20,
            loader_length=10,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "in-epoch.pth.tar"
            save_two_stage_checkpoint(
                str(path),
                model=source,
                ema=ema,
                optimizer=optimizer,
                scheduler=None,
                scaler=None,
                config=config,
                substage="warmup",
                completed_warmup_steps=3,
                completed_joint_steps=0,
                train_steps=11,
                epoch=4,
                include_optimizer=True,
                batch_in_epoch=5,
                runtime_states=[runtime_state],
                data_fingerprint=data_fingerprint,
            )

            # Perturb all three RNGs and initialize fresh objects before the
            # public restore helper is exercised.
            random.random()
            np.random.random_sample(7)
            torch.rand(9)
            target = _TinyTwoStageModel()
            target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-2)
            summary = load_two_stage_resume(
                target,
                target_optimizer,
                path,
                expected_substage="warmup",
                config=config,
            )
            extended_config = copy.deepcopy(config)
            extended_config.finetune.warmup_steps = 6
            extended_summary = load_two_stage_resume(
                _TinyTwoStageModel(),
                None,
                path,
                expected_substage="warmup",
                config=extended_config,
            )

        self.assertEqual(summary["batch_in_epoch"], 5)
        self.assertEqual(summary["runtime_world_size"], 1)
        self.assertEqual(summary["runtime_state"]["proxy_metrics"], runtime_state["proxy_metrics"])
        self.assertEqual(summary["runtime_state"]["valid_alignment_batches"], 2)
        self.assertEqual(summary["data_fingerprint"]["dataset_length"], 20)
        self.assertEqual(summary["data_fingerprint"]["loader_length"], 10)
        self.assertTrue(summary["optimizer_loaded"])
        self.assertEqual(extended_summary["epoch"], 4)
        self.assertEqual(extended_summary["batch_in_epoch"], 5)

        restore_rng_state(summary["runtime_state"]["rng_state"])
        self.assertEqual([random.random() for _ in range(4)], expected_python)
        np.testing.assert_array_equal(np.random.random_sample(4), expected_numpy)
        torch.testing.assert_close(
            torch.rand(4), expected_torch, rtol=0.0, atol=0.0
        )

    def test_old_two_stage_checkpoint_defaults_to_epoch_boundary_cursor(self):
        source, checkpoint = self._resume_checkpoint("warmup")
        self.assertNotIn("data_progress", checkpoint)
        self.assertNotIn("rank_runtime_states", checkpoint)
        target = _TinyTwoStageModel()
        summary = load_two_stage_resume(
            target,
            None,
            checkpoint,
            expected_substage="warmup",
            config=_config(
                stage="real_finetune",
                action_mode="real",
                scheme="embedding_align",
                proxy_type="latent",
            ),
        )
        self.assertEqual(summary["batch_in_epoch"], 0)
        self.assertEqual(summary["runtime_world_size"], 0)
        self.assertIsNone(summary["runtime_state"])
        torch.testing.assert_close(target.backbone.weight, source.backbone.weight)

    def test_old_explicit_zero_cursor_without_fingerprint_remains_compatible(self):
        source, checkpoint = self._resume_checkpoint("warmup")
        checkpoint["data_progress"] = {"batch_in_epoch": 0, "world_size": 1}
        checkpoint["rank_runtime_states"] = [
            {"rng_state": capture_rng_state()}
        ]
        target = _TinyTwoStageModel()
        summary = load_two_stage_resume(
            target,
            None,
            checkpoint,
            expected_substage="warmup",
            config=_config(
                stage="real_finetune",
                action_mode="real",
                scheme="embedding_align",
                proxy_type="latent",
            ),
        )
        self.assertEqual(summary["batch_in_epoch"], 0)
        self.assertIsNone(summary["data_fingerprint"])
        self.assertIsNotNone(summary["runtime_state"])
        torch.testing.assert_close(target.backbone.weight, source.backbone.weight)

    def test_in_epoch_resume_rejects_data_config_changes(self):
        base_config = OmegaConf.create(
            _config(
                stage="real_finetune",
                action_mode="real",
                scheme="embedding_align",
                proxy_type="latent",
            )
        )
        source = _TinyTwoStageModel()
        fingerprint = build_data_resume_fingerprint(
            base_config,
            "warmup",
            dataset_length=12,
            loader_length=6,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cursor.pth.tar"
            save_two_stage_checkpoint(
                str(path),
                model=source,
                ema=copy.deepcopy(source),
                optimizer=self._optimizer_with_state(source),
                scheduler=None,
                scaler=None,
                config=base_config,
                substage="warmup",
                completed_warmup_steps=2,
                completed_joint_steps=0,
                train_steps=2,
                epoch=1,
                include_optimizer=True,
                batch_in_epoch=3,
                runtime_states=[{"rng_state": capture_rng_state()}],
                data_fingerprint=fingerprint,
            )

            cases = []
            changed_seed = copy.deepcopy(base_config)
            changed_seed.seed += 1
            cases.append((changed_seed, "seed"))
            changed_batch = copy.deepcopy(base_config)
            changed_batch.training.batch_size = 4
            cases.append((changed_batch, "batch_size"))
            changed_dataset = copy.deepcopy(base_config)
            changed_dataset.dataset["sampling_variant"] = "changed-order"
            cases.append((changed_dataset, "dataset"))
            for changed_config, changed_field in cases:
                with self.subTest(changed_field=changed_field):
                    with self.assertRaisesRegex(
                        ValueError, "data resume fingerprint mismatch"
                    ):
                        load_two_stage_resume(
                            _TinyTwoStageModel(),
                            None,
                            path,
                            expected_substage="warmup",
                            config=changed_config,
                        )

    def test_in_epoch_checkpoint_without_fingerprint_is_rejected(self):
        _, checkpoint = self._resume_checkpoint("warmup")
        checkpoint["data_progress"] = {"batch_in_epoch": 1, "world_size": 1}
        checkpoint["rank_runtime_states"] = [
            {"rng_state": capture_rng_state()}
        ]
        with self.assertRaisesRegex(ValueError, "no data resume fingerprint"):
            load_two_stage_resume(
                _TinyTwoStageModel(),
                None,
                checkpoint,
                expected_substage="warmup",
                config=_config(
                    stage="real_finetune",
                    action_mode="real",
                    scheme="embedding_align",
                    proxy_type="latent",
                ),
            )

    def test_data_fingerprint_validates_observed_dataset_and_loader_lengths(self):
        config = _config(
            stage="real_finetune",
            action_mode="real",
            scheme="embedding_align",
            proxy_type="latent",
        )
        fingerprint = build_data_resume_fingerprint(
            config,
            "warmup",
            dataset_length=12,
            loader_length=6,
        )
        validate_data_resume_fingerprint(
            fingerprint,
            config,
            "warmup",
            dataset_length=12,
            loader_length=6,
            required=True,
        )
        with self.assertRaisesRegex(ValueError, "dataset_length mismatch"):
            validate_data_resume_fingerprint(
                fingerprint,
                config,
                "warmup",
                dataset_length=13,
                loader_length=6,
                required=True,
            )

    def test_phase_final_keeps_partial_cursor_for_extended_target(self):
        self.assertEqual(
            _phase_progress_after_iteration(4, 3, 8),
            (4, 3, False),
        )
        self.assertEqual(
            _phase_progress_after_iteration(4, 8, 8),
            (5, 0, True),
        )

    def test_completed_phase_restores_rng_before_joint_transition(self):
        random.seed(2201)
        np.random.seed(2202)
        torch.manual_seed(2203)
        saved = capture_rng_state()
        expected = (random.random(), np.random.random(), torch.rand(3))
        random.random()
        np.random.random(4)
        torch.rand(5)

        remaining = _restore_completed_phase_rng(7, 7, saved)

        self.assertIsNone(remaining)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0.0, atol=0.0)

    def test_transition_preserves_scaler_and_starts_joint_at_epoch_zero(self):
        config = OmegaConf.create(
            _config(
                stage="real_finetune",
                action_mode="real",
                scheme="embedding_align",
                proxy_type="latent",
                warmup_steps=7,
                joint_steps=4,
            )
        )
        source = _TinyTwoStageModel()
        scaler_state = {"scale": 4096.0, "growth_tracker": 17}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transition.pth.tar"
            save_two_stage_checkpoint(
                str(path),
                model=source,
                ema=copy.deepcopy(source),
                optimizer=None,
                scheduler=None,
                scaler=_StatefulScaler(scaler_state),
                config=config,
                substage="transition",
                completed_warmup_steps=7,
                completed_joint_steps=0,
                train_steps=7,
                epoch=0,
                include_optimizer=False,
                batch_in_epoch=0,
                runtime_states=[{"rng_state": capture_rng_state()}],
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertNotIn("opt", checkpoint)
            self.assertNotIn("scheduler", checkpoint)
            self.assertEqual(checkpoint["scaler"], scaler_state)

            restored_scaler = _StatefulScaler()
            _load_ema_and_training_state(
                path,
                copy.deepcopy(source),
                None,
                restored_scaler,
            )
            self.assertEqual(restored_scaler.state, scaler_state)
            summary = load_two_stage_resume(
                _TinyTwoStageModel(),
                None,
                path,
                expected_substage="joint",
                config=config,
            )

        self.assertEqual(summary["epoch"], 0)
        self.assertEqual(summary["batch_in_epoch"], 0)
        self.assertEqual(
            summary["data_fingerprint"]["payload"]["resume_substage"],
            "joint",
        )

    def test_ddp_alignment_gradient_assertion_uses_local_validity(self):
        common = {
            "check_gradients": True,
            "training_stage": "real_finetune",
            "scheme": "embedding_align",
            "substage": "warmup",
        }
        # Another rank may make the global alignment count nonzero; this rank's
        # empty mask must still skip the local nonzero-gradient assertion while
        # participating in the synchronized backward/optimizer step.
        self.assertFalse(
            _should_check_first_step_gradients(
                **common, local_alignment_has_target=False
            )
        )
        self.assertTrue(
            _should_check_first_step_gradients(
                **common, local_alignment_has_target=True
            )
        )
        self.assertTrue(
            _should_check_first_step_gradients(
                check_gradients=True,
                training_stage="real_finetune",
                scheme="embedding_align",
                substage="joint",
                local_alignment_has_target=None,
            )
        )

    def test_transition_enters_joint_without_loading_warmup_optimizer(self):
        source, checkpoint = self._resume_checkpoint(
            "transition", include_optimizer=False
        )
        target = _TinyTwoStageModel()
        optimizer = torch.optim.AdamW(target.parameters(), lr=9e-4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transition.pth.tar"
            torch.save(checkpoint, path)
            summary = load_two_stage_resume(
                target,
                optimizer,
                path,
                expected_substage="joint",
                config=_config(
                    stage="real_finetune",
                    action_mode="real",
                    scheme="embedding_align",
                    proxy_type="latent",
                ),
            )
        self.assertFalse(summary["optimizer_loaded"])
        self.assertEqual(summary["resume_substage"], "joint")
        self.assertEqual(optimizer.param_groups[0]["lr"], 9e-4)
        torch.testing.assert_close(target.backbone.weight, source.backbone.weight)

    def test_substage_mismatch_is_rejected(self):
        _, checkpoint = self._resume_checkpoint("warmup")
        with self.assertRaisesRegex(ValueError, "substage mismatch"):
            load_two_stage_resume(
                _TinyTwoStageModel(), None, checkpoint, expected_substage="joint"
            )

    def test_resume_rejects_proxy_provenance_window_and_normalization_changes(self):
        _, checkpoint = self._resume_checkpoint("warmup")
        base_config = _config(
            stage="real_finetune",
            action_mode="real",
            scheme="embedding_align",
            proxy_type="latent",
        )
        cases = []

        wrong_source = copy.deepcopy(base_config)
        wrong_source["proxy"]["type"] = "geometry"
        wrong_source["proxy"]["dim"] = 3
        cases.append((wrong_source, "proxy_type mismatch"))

        wrong_window = copy.deepcopy(base_config)
        wrong_window["proxy"]["max_abs_frame_offset"] = 7
        cases.append((wrong_window, "proxy_max_abs_frame_offset mismatch"))

        wrong_normalization = copy.deepcopy(base_config)
        wrong_normalization["motion_condition"]["latent"]["normalization"][
            "identifier"
        ] = "different-train-split"
        cases.append((wrong_normalization, "latent_normalization mismatch"))

        for config, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_resume_checkpoint(
                        checkpoint[TWO_STAGE_METADATA_KEY],
                        config=config,
                        model=_TinyTwoStageModel(),
                        expected_substage="warmup",
                    )

    def test_resume_phase_counter_contract(self):
        _, transition = self._resume_checkpoint(
            "transition", include_optimizer=False
        )
        transition[TWO_STAGE_METADATA_KEY]["completed_joint_steps"] = 1
        with self.assertRaisesRegex(ValueError, "cannot contain completed joint"):
            validate_resume_checkpoint(
                transition[TWO_STAGE_METADATA_KEY], expected_substage="joint"
            )

        _, valid_joint = self._resume_checkpoint("joint")
        config = _config(
            stage="real_finetune",
            action_mode="real",
            scheme="embedding_align",
            proxy_type="latent",
            warmup_steps=7,
            joint_steps=4,
        )
        summary = validate_resume_checkpoint(
            valid_joint[TWO_STAGE_METADATA_KEY],
            config=config,
            model=_TinyTwoStageModel(),
            expected_substage="joint",
        )
        self.assertEqual(summary["completed_warmup_steps"], 7)
        self.assertEqual(summary["completed_joint_steps"], 3)

        bad_target = copy.deepcopy(config)
        bad_target["finetune"]["warmup_steps"] = 8
        with self.assertRaisesRegex(ValueError, "completed_warmup_steps mismatch"):
            validate_resume_checkpoint(
                valid_joint[TWO_STAGE_METADATA_KEY],
                config=bad_target,
                model=_TinyTwoStageModel(),
                expected_substage="joint",
            )

    def test_proxy_pretrain_resume_has_null_scheme_and_no_finetune_counts(self):
        model, checkpoint = _stage1_checkpoint()
        validation = validate_resume_checkpoint(
            checkpoint[TWO_STAGE_METADATA_KEY],
            config=_config(
                stage="proxy_pretrain",
                action_mode="latent",
                proxy_type="latent",
            ),
            model=model,
            expected_substage="pretrain",
        )
        self.assertEqual(validation["finetune_substage"], "pretrain")

        latentonly_config = _config(
            stage="proxy_pretrain",
            action_mode="latent",
            proxy_type="latent",
            relative_time_mode="fallback",
        )
        with self.assertRaisesRegex(ValueError, "proxy_relative_time_mode"):
            validate_resume_checkpoint(
                checkpoint[TWO_STAGE_METADATA_KEY],
                config=latentonly_config,
                model=model,
                expected_substage="pretrain",
            )

        corrupted = copy.deepcopy(checkpoint[TWO_STAGE_METADATA_KEY])
        corrupted["completed_warmup_steps"] = 1
        with self.assertRaisesRegex(ValueError, "cannot contain fine-tuning"):
            validate_resume_checkpoint(corrupted, expected_substage="pretrain")

    def test_old_checkpoint_is_left_for_legacy_loader(self):
        with self.assertRaisesRegex(ValueError, "Legacy checkpoints"):
            load_two_stage_resume(
                _TinyTwoStageModel(), None, {"model": _TinyTwoStageModel().state_dict()}
            )

    def test_old_checkpoint_still_loads_through_legacy_loader(self):
        from train_utils import load_checkpoint

        source = _TinyTwoStageModel()
        ema_source = copy.deepcopy(source)
        optimizer_source = self._optimizer_with_state(source)
        checkpoint = {
            "model": source.state_dict(),
            "ema": ema_source.state_dict(),
            "opt": optimizer_source.state_dict(),
            "epoch": 4,
            "train_steps": 19,
        }
        target = _TinyTwoStageModel()
        ema_target = _TinyTwoStageModel()
        optimizer_target = torch.optim.AdamW(target.parameters(), lr=1e-2)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "legacy.pth.tar"
            torch.save(checkpoint, checkpoint_path)
            config = SimpleNamespace(
                training=SimpleNamespace(from_checkpoint=str(checkpoint_path))
            )
            start_epoch, train_steps = load_checkpoint(
                directory,
                config,
                target,
                ema_target,
                optimizer_target,
            )
        self.assertEqual(start_epoch, 5)
        self.assertEqual(train_steps, 19)
        torch.testing.assert_close(target.backbone.weight, source.backbone.weight)
        torch.testing.assert_close(
            ema_target.backbone.weight, ema_source.backbone.weight
        )


if __name__ == "__main__":
    unittest.main()
