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
        for model, motion_min, batch_size, constrained in product(('opennwm', 'nwm-b', 'opennwm-dmd', 'nwm-dmd'), (.1, 1.), (6, 7, 14, 15), (False,True)):
            with self.subTest(model=model, motion_min=motion_min, batch_size=batch_size), tempfile.TemporaryDirectory() as tmp:
                paths = {name: p['effective_path'] for name, p in proposals.items()
                         if not constrained or p['nominal']['distance_m'] > 0 or p['nominal']['yaw_rad'] >= 0}
                motion_count=len(paths)
                engine = nav.Engine.__new__(nav.Engine)
                engine.args = SimpleNamespace(model=model, batch_size=batch_size, seed=0)
                engine.torch, engine.np, engine.device = torch, np, torch.device('cpu')
                engine.config = SimpleNamespace(model=SimpleNamespace(generator=SimpleNamespace(input_size=28)),
                                                dataset=SimpleNamespace(action_stats={}))
                engine.models = ()
                engine.transform = lambda im: torch.zeros(3,224,224)
                engine.encode_history = lambda images: (torch.zeros(4,4,28,28),1)
                engine.distance = MagicMock(return_value=.4)
                engine.tokenizer = SimpleNamespace(decode=MagicMock(side_effect=lambda output, **kw: torch.zeros(len(output),3,224,224)))
                candidate_scores = torch.arange(motion_count,dtype=torch.float32)+motion_min
                candidate_scores[0] = .5
                candidate_scores[1] = motion_min
                chunks = [candidate_scores[i:i+batch_size,None] for i in range(0,motion_count,batch_size)]
                engine.lpips_distances = MagicMock(side_effect=[torch.tensor([[.4]],dtype=torch.float64)]+chunks)
                forward = MagicMock(side_effect=lambda models, context, *a, **kw: torch.zeros(len(context),4,28,28))
                image = Image.new('RGB',(224,224),(70,120,160))
                with patch.dict(sys.modules, {'isolated_nwm_infer':SimpleNamespace(model_forward_wrapper=forward),
                                              'misc':SimpleNamespace(normalize_data=lambda x,stats:x)}), \
                     patch.object(torch.cuda,'synchronize'), \
                     patch.object(torch.amp,'autocast',side_effect=lambda *a,**kw:contextlib.nullcontext()):
                    result,chosen = engine.select([image]*4,image,paths,1,Path(tmp))
                self.assertEqual(forward.call_count, len(chunks))
                actual = torch.cat([call.kwargs['rel_t'] for call in forward.call_args_list])
                torch.testing.assert_close(actual,torch.full((motion_count,),.015625),rtol=0,atol=0)
                self.assertEqual(result['normalized_k'],.015625)
                self.assertEqual(result['model_horizon_seconds'],.5)
                self.assertEqual(result['model_frame_offset'],2.)
                self.assertEqual(result['candidate_count'],len(paths))
                self.assertEqual(result['inferred_candidate_count'],motion_count)
                self.assertEqual(result['current_goal_lpips'],.4)
                engine.distance.assert_not_called()
                self.assertEqual(engine.lpips_distances.call_count,1+len(chunks))
                self.assertEqual(engine.tokenizer.decode.call_count,len(chunks))
                self.assertEqual(len(result['ranking']),len(paths))
                stop = next(r for r in result['ranking'] if r['candidate']=='stationary')
                self.assertEqual(stop,{'candidate':'stationary','score':.5,'lpips':.5,'score_source':'model_prediction'})
                expected = np.asarray([paths[k][-1] for k in paths],dtype=np.float32)
                np.testing.assert_array_equal(torch.cat([call.args[2] for call in forward.call_args_list]).numpy()[:,0],expected)
                if motion_min > .4:
                    self.assertEqual(result['selected'],'stationary')
                    self.assertEqual(result['selected_image_source'],'model_prediction')
                    self.assertEqual(chosen.getpixel((0,0)),(0,0,0))
                else:
                    self.assertEqual(result['selected'],'forward_25')
                    self.assertEqual(result['selected_image_source'],'model_prediction')


if __name__=='__main__':unittest.main()
