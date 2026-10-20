"""Verify actual inference tensors, not merely the default constant."""
import contextlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jetson_navigate as nav
from calibrated_actions import calibrated_proposals
from calibration_fixture import calibration_fixture


class FixedNavigationTime(unittest.TestCase):
    def test_all_models_receive_same_half_second_for_all_candidates(self):
        calibration = calibration_fixture()
        for i, entry in enumerate(calibration['actions'].values()):
            entry['summary']['observation_horizon_s'] = 1. + i
        proposals = calibrated_proposals(calibration)
        for p in proposals.values():
            self.assertEqual(p['normalized_k'], .015625)
            self.assertEqual(p['model_horizon_seconds'], .5)
        paths = {name: p['effective_path'] for name, p in proposals.items()}
        from itertools import product
        for model, motion_min, batch_size in product(('opennwm', 'nwm-b', 'opennwm-dmd', 'nwm-dmd'), (.1, 1.), (14, 15)):
            with self.subTest(model=model, motion_min=motion_min, batch_size=batch_size), tempfile.TemporaryDirectory() as tmp:
                engine = nav.Engine.__new__(nav.Engine)
                engine.args = SimpleNamespace(model=model, batch_size=batch_size, seed=0)
                engine.torch, engine.np, engine.device = torch, np, torch.device('cpu')
                engine.config = SimpleNamespace(model=SimpleNamespace(generator=SimpleNamespace(input_size=28)),
                                                dataset=SimpleNamespace(action_stats={}))
                engine.models = ()
                engine.transform = lambda im: torch.zeros(3,224,224)
                engine.encode_history = lambda images: (torch.zeros(4,4,28,28),1)
                engine.distance = MagicMock(return_value=.4)
                engine.tokenizer = SimpleNamespace(decode=MagicMock(return_value=torch.zeros(14,3,224,224)))
                engine.scorer = lambda a,b: torch.arange(14,dtype=torch.float32) + motion_min
                forward = MagicMock(return_value=torch.zeros(14,4,28,28))
                image = Image.new('RGB',(224,224),(70,120,160))
                with patch.dict(sys.modules, {'isolated_nwm_infer':SimpleNamespace(model_forward_wrapper=forward),
                                              'misc':SimpleNamespace(normalize_data=lambda x,stats:x)}), \
                     patch.object(torch.cuda,'synchronize'), \
                     patch.object(torch.amp,'autocast',side_effect=lambda *a,**kw:contextlib.nullcontext()):
                    result,chosen = engine.select([image]*4,image,paths,1,Path(tmp))
                forward.assert_called_once()
                actual = forward.call_args.kwargs['rel_t']
                torch.testing.assert_close(actual,torch.full((14,),.015625),rtol=0,atol=0)
                self.assertEqual(result['normalized_k'],.015625)
                self.assertEqual(result['model_horizon_seconds'],.5)
                self.assertEqual(result['model_frame_offset'],2.)
                self.assertEqual(result['candidate_count'],15)
                self.assertEqual(result['inferred_candidate_count'],14)
                self.assertEqual(result['current_goal_lpips'],.4)
                engine.distance.assert_called_once_with(image,image)
                engine.tokenizer.decode.assert_called_once()
                self.assertEqual(len(result['ranking']),15)
                stop = next(r for r in result['ranking'] if r['candidate']=='stationary')
                self.assertEqual(stop,{'candidate':'stationary','lpips':.4,'score_source':'current_observation'})
                expected = np.asarray([paths[k][-1] for k in paths if k!='stationary'],dtype=np.float32)
                np.testing.assert_array_equal(forward.call_args.args[2].numpy()[:,0],expected)
                if motion_min > .4:
                    self.assertEqual(result['selected'],'stationary')
                    self.assertEqual(result['selected_image_source'],'current_observation')
                    self.assertEqual(chosen.tobytes(),nav.model_frame(image).tobytes())
                else:
                    self.assertEqual(result['selected'],'forward_25')
                    self.assertEqual(result['selected_image_source'],'model_prediction')


if __name__=='__main__':unittest.main()
