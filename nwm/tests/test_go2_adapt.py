"""CPU checks for the shipped Go2 adaptation recipe and initialization contract."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from two_stage_nwm import validate_two_stage_config


class Go2AdaptConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
            cls.recipe = compose(config_name="nwm", overrides=["two_stage=go2_adapt"])

    def initialized_config(self):
        config = copy.deepcopy(self.recipe)
        config.finetune.init_checkpoint = "/weights/real-action.pth.tar"
        return config

    def test_shipped_recipe_allows_local_logging_and_validation(self):
        config = self.initialized_config()
        self.assertFalse(config.training.wandb_enabled)
        self.assertFalse(config.training.wandb_required)
        self.assertTrue(config.eval_at_first_step)
        self.assertGreater(config.eval_every, 0)
        self.assertEqual(Path(config.dataset.datasets.go2.test).name, "val")
        self.assertEqual(
            validate_two_stage_config(config), ("real_finetune", "real", "reset")
        )

    def test_online_logging_remains_optional(self):
        config = self.initialized_config()
        config.training.wandb_enabled = True
        config.training.wandb_required = True
        self.assertEqual(
            validate_two_stage_config(config), ("real_finetune", "real", "reset")
        )

    def test_resume_can_supply_initialization(self):
        config = copy.deepcopy(self.recipe)
        config.training.from_checkpoint = "/runs/go2/checkpoints/latest.pth.tar"
        self.assertEqual(
            validate_two_stage_config(config), ("real_finetune", "real", "reset")
        )

    def test_missing_complete_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires init_checkpoint or from_checkpoint"):
            validate_two_stage_config(self.recipe)

    def test_go2_must_preserve_the_trained_real_adapter(self):
        for field, value in (
            ("random_init", True),
            ("stage1_checkpoint", "/weights/stage1.pth.tar"),
            ("scheme", "real_to_latent"),
        ):
            with self.subTest(field=field):
                config = copy.deepcopy(self.recipe)
                config.training.from_checkpoint = "/runs/go2/checkpoints/latest.pth.tar"
                config.finetune[field] = value
                with self.assertRaisesRegex(ValueError, "preserve the existing real-action adapter"):
                    validate_two_stage_config(config)

    def test_legacy_original_nwm_interface_is_not_accepted(self):
        config = self.initialized_config()
        config.motion_condition.enabled = False
        with self.assertRaisesRegex(ValueError, "requires motion_condition.enabled=true"):
            validate_two_stage_config(config)


if __name__ == "__main__":
    unittest.main()
