import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jetson_navigate as nav


class MatchingBatches(unittest.TestCase):
    def engine(self):
        e = nav.Engine.__new__(nav.Engine)
        e.torch, e.np, e.device = torch, np, torch.device('cpu')
        e.args = SimpleNamespace(batch_size=1)
        return e

    def test_cartesian_pair_order_and_partial_batch(self):
        e = self.engine()
        e.scorer = MagicMock(side_effect=lambda a,b: ((a-b)**2).mean((1,2,3)))
        a = torch.arange(2.).reshape(2,1,1,1)
        b = torch.arange(3.).reshape(3,1,1,1)
        torch.testing.assert_close(e.lpips_distances(a,b), torch.tensor([[0.,1.,4.],[1.,0.,1.]]))
        self.assertEqual([len(c.args[0]) for c in e.scorer.call_args_list], [3,3])

    def test_default_seven_candidates_follow_actual_target_count(self):
        e = self.engine()
        e.args.batch_size = 7
        for count, targets, expected in ((7,10,[70]), (7,3,[21]), (8,3,[21,3]), (1,10,[10])):
            with self.subTest(count=count, targets=targets):
                e.scorer = MagicMock(side_effect=lambda a,b: ((a-b)**2).mean((1,2,3)))
                scores = e.lpips_distances(torch.zeros(count,1,1,1), torch.zeros(targets,1,1,1))
                self.assertEqual(tuple(scores.shape),(count,targets))
                self.assertEqual([len(c.args[0]) for c in e.scorer.call_args_list],expected)

    def test_target_cache_reuses_overlap_and_handles_changed_image(self):
        e = self.engine()
        e.dino = SimpleNamespace(encode=MagicMock(side_effect=lambda x:x.mean((2,3))))
        red, blue = Image.new('RGB',(224,224),'red'), Image.new('RGB',(224,224),'blue')
        first = e.dino_target_features([(1,red),(2,blue)])
        reordered = e.dino_target_features([(2,blue),(1,red)])
        torch.testing.assert_close(first.flip(0), reordered)
        self.assertEqual(e.dino.encode.call_count, 1)
        e.dino_target_features([(1,Image.new('RGB',(224,224),'green'))])
        self.assertEqual(e.dino.encode.call_count, 2)
        self.assertEqual(len(e.dino.encode.call_args.args[0]), 1)


if __name__ == '__main__':
    unittest.main()
