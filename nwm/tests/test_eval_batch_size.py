"""Evaluation batch overrides must leave the training loader unchanged."""
import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf

from data_utils import create_dataloader


class EvalBatchSizeTests(unittest.TestCase):
    def test_independent_batch_and_legacy_fallback(self):
        for override in ({}, {"eval_batch_size": None}, {"eval_batch_size": 4}):
            for reference in ({}, {"reference_batch_size": 16}):
                with self.subTest(override=override, reference=reference):
                    config = OmegaConf.create({
                        "seed": 1,
                        "training": {"batch_size": 48, "num_workers": 0,
                                     **override, **reference},
                    })
                    dataset = torch.arange(192)
                    with mock.patch("data_utils.dist.get_world_size", return_value=1):
                        train, _ = create_dataloader(dataset, config, 0, True)
                        evaluation, _ = create_dataloader(dataset, config, 0, False)
                    self.assertEqual(len(next(iter(train))), 48)
                    self.assertEqual(len(next(iter(evaluation))),
                                     override.get("eval_batch_size") or 48)


if __name__ == "__main__":
    unittest.main()
