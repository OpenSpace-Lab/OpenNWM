"""Evaluation consumes distinct batches and averages over all goal images."""
import unittest
from unittest import mock

import torch

import train_utils


class Loader(list):
    def __init__(self, batches):
        super().__init__(batches)
        self.sampler = mock.Mock()


class Scores(torch.nn.Module):
    def forward(self, targets, predictions):
        return targets[:, 0, 0, 0]


class EvalBatchesTests(unittest.TestCase):
    def run_eval(self, sizes, *, dictionary=False, num_batches=None):
        batches = []
        start = 1
        for size in sizes:
            video = torch.arange(start, start + size, dtype=torch.float32)
            video = video[:, None, None, None, None].expand(size, 3, 3, 2, 2)
            offsets = torch.zeros(size, 2)
            batches.append({"video": video, "k": offsets} if dictionary else
                           (video, torch.zeros(size, 2, 2), offsets))
            start += size
        options = {} if num_batches is None else {"num_batches": num_batches}
        with mock.patch.object(train_utils, "_eval_model_cache", Scores()), \
             mock.patch.object(train_utils.dist, "all_reduce"), \
             mock.patch("isolated_nwm_infer.model_forward_wrapper",
                        side_effect=lambda models, x, y, **kw:
                        torch.zeros(x.shape[0] * 2, 3, 2, 2)) as forward:
            score = train_utils.evaluate(
                None, None, None, Loader(batches), rank=1, latent_size=2,
                device=torch.device("cpu"), save_dir="unused", seed=42,
                bfloat_enable=False, num_cond=1, unnormalize_fn=lambda x: x,
                **options,
            )
        return score.item(), forward.call_count

    def test_four_batches_evaluate_sixteen_distinct_observations(self):
        for dictionary in (False, True):
            with self.subTest(dictionary=dictionary):
                score, calls = self.run_eval([4] * 4, dictionary=dictionary,
                                             num_batches=4)
                self.assertEqual(score, 8.5)
                self.assertEqual(calls, 4)

    def test_partial_batch_is_weighted_by_sample_count(self):
        score, calls = self.run_eval([4, 1], num_batches=2)
        self.assertEqual(score, 3.0)
        self.assertEqual(calls, 2)

    def test_default_still_evaluates_one_batch(self):
        score, calls = self.run_eval([4] * 4)
        self.assertEqual(score, 2.5)
        self.assertEqual(calls, 1)

    def test_rejects_invalid_or_unavailable_batch_count(self):
        for count in (0, -1, 1.5, 5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.run_eval([4] * 4, num_batches=count)


if __name__ == "__main__":
    unittest.main()
