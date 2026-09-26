"""Contract tests for the strict random-initialization baseline."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from two_stage_checkpoint import (
    build_checkpoint_metadata,
    validate_resume_checkpoint,
)
from two_stage_nwm import validate_two_stage_config


class NoPretrainConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
            cls.no_pretrain = compose(
                config_name="nwm", overrides=["two_stage=no_pretrain"]
            )
            cls.time_reset = compose(
                config_name="nwm", overrides=["two_stage=time_reset"]
            )

    def test_uses_fresh_weights_and_same_stage2_budget(self):
        stage, action_mode, scheme = validate_two_stage_config(self.no_pretrain)
        self.assertEqual(
            (stage, action_mode, scheme), ("real_finetune", "real", "reset")
        )
        self.assertTrue(self.no_pretrain.finetune.random_init)
        self.assertIsNone(self.no_pretrain.finetune.stage1_checkpoint)

        self.assertEqual(self.no_pretrain.finetune.warmup_steps, 0)
        self.assertEqual(
            self.no_pretrain.finetune.joint_steps,
            self.time_reset.finetune.warmup_steps
            + self.time_reset.finetune.joint_steps,
        )
        for key in ("adapter_lr", "backbone_lr"):
            with self.subTest(key=key):
                self.assertEqual(
                    self.no_pretrain.finetune[key], self.time_reset.finetune[key]
                )
        self.assertEqual(
            self.no_pretrain.dataset_selection.finetune,
            self.time_reset.dataset_selection.finetune,
        )
        self.assertEqual(
            self.no_pretrain.training.batch_size,
            self.time_reset.training.batch_size,
        )
        self.assertEqual(self.no_pretrain.training.num_workers, 8)
        for key in (
            "seed",
            "bfloat16",
            "ckpt_every",
            "eval_every",
            "eval_at_first_step",
            "dataset.image_size",
            "dataset.context_size",
            "model.generator",
            "training.optimizer",
        ):
            with self.subTest(key=key):
                no_pretrain_value = OmegaConf.select(self.no_pretrain, key)
                time_reset_value = OmegaConf.select(self.time_reset, key)
                if OmegaConf.is_config(no_pretrain_value):
                    no_pretrain_value = OmegaConf.to_container(
                        no_pretrain_value, resolve=False
                    )
                    time_reset_value = OmegaConf.to_container(
                        time_reset_value, resolve=False
                    )
                self.assertEqual(no_pretrain_value, time_reset_value)

    def test_stage1_checkpoint_is_rejected(self):
        invalid = copy.deepcopy(self.no_pretrain)
        invalid.finetune.stage1_checkpoint = "/tmp/stage1.pth.tar"
        with self.assertRaisesRegex(ValueError, "forbids finetune.stage1_checkpoint"):
            validate_two_stage_config(invalid)

    def test_adapter_only_warmup_is_rejected(self):
        invalid = copy.deepcopy(self.no_pretrain)
        invalid.finetune.warmup_steps = 1
        with self.assertRaisesRegex(ValueError, "requires finetune.warmup_steps=0"):
            validate_two_stage_config(invalid)

    def test_checkpoint_records_random_provenance_and_rejects_pt_resume(self):
        metadata = build_checkpoint_metadata(
            self.no_pretrain,
            finetune_substage="joint",
            completed_warmup_steps=0,
            completed_joint_steps=3,
        )
        self.assertEqual(metadata["finetune_initialization"], "random")
        summary = validate_resume_checkpoint(
            metadata,
            config=self.no_pretrain,
            expected_substage="joint",
        )
        self.assertEqual(summary["finetune_initialization"], "random")

        pretrained_metadata = dict(metadata)
        pretrained_metadata["finetune_initialization"] = "stage1_checkpoint"
        with self.assertRaisesRegex(ValueError, "finetune_initialization mismatch"):
            validate_resume_checkpoint(
                pretrained_metadata,
                config=self.no_pretrain,
                expected_substage="joint",
            )


if __name__ == "__main__":
    unittest.main()
