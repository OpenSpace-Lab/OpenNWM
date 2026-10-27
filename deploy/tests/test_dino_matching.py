"""Metric semantics independent of downloaded weights or robot hardware."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dino_matching import DinoMatcher


class DinoDistance(unittest.TestCase):
    def test_identical_orthogonal_and_opposite_features(self):
        features = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        torch.testing.assert_close(DinoMatcher.distances(features, features[:1]),
                                   torch.tensor([[0.], [1.], [2.]]))

    def test_cls_normalization_and_imagenet_preprocessing(self):
        matcher = DinoMatcher.__new__(DinoMatcher)
        matcher.mean = torch.tensor([.485, .456, .406])[None, :, None, None]
        matcher.std = torch.tensor([.229, .224, .225])[None, :, None, None]
        matcher.model = MagicMock(return_value=SimpleNamespace(last_hidden_state=torch.tensor([[[3., 4.], [9., 9.]]])))
        pixels = torch.ones(1, 3, 224, 224)
        torch.testing.assert_close(matcher.encode(pixels), torch.tensor([[.6, .8]]))
        torch.testing.assert_close(matcher.model.call_args.kwargs['pixel_values'], (pixels-matcher.mean)/matcher.std)
        matcher.model.return_value.last_hidden_state[:] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'Invalid DINOv2'):
            matcher.encode(pixels)


if __name__ == '__main__':
    unittest.main()
