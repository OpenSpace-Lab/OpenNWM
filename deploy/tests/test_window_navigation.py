"""Window selection tests without GPU, DDS or robot movement."""
import contextlib
import itertools
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jetson_navigate as nav


class LoopAvoidance(unittest.TestCase):
    def setUp(self):
        self.guard = nav.NavigationLoopGuard()
        self.pose = SimpleNamespace(x=0., y=0., yaw=0.)
        self.ranking = [dict(target_index=6, candidate='stationary', lpips=.27),
                        dict(target_index=6, candidate='forward', lpips=.3),
                        dict(target_index=7, candidate='stationary', lpips=.4)]

    def test_first_repeat_is_blocked_for_exactly_one_selection(self):
        chosen, event = self.guard.choose(self.ranking, self.pose)
        self.assertIsNone(event)
        self.guard.observe(6, 'stationary', self.pose, 3)
        chosen, event = self.guard.choose(self.ranking, self.pose)
        self.assertEqual(chosen['candidate'], 'forward')
        self.assertEqual(event['previous_step'], 3)
        self.guard.observe(6, 'forward', self.pose, 4)
        moved_pose = SimpleNamespace(x=.1, y=0., yaw=0.)
        chosen, event = self.guard.choose(self.ranking, moved_pose)
        self.assertEqual(chosen['candidate'], 'stationary')
        self.assertIsNone(event)

    def test_multi_step_cycle_is_detected_on_first_return(self):
        self.guard.observe(6, 'stationary', self.pose, 3)
        self.guard.observe(7, 'forward', SimpleNamespace(x=.2, y=0., yaw=0.), 4)
        self.assertIsNotNone(self.guard.choose(self.ranking, self.pose)[1])

    def test_rollback_at_changed_pose_is_allowed(self):
        self.guard.observe(6, 'stationary', self.pose, 3)
        for pose in (SimpleNamespace(x=.1, y=0., yaw=0.),
                     SimpleNamespace(x=0., y=0., yaw=.2)):
            self.assertIsNone(self.guard.choose(self.ranking, pose)[1])

    def test_all_stationary_targets_are_excluded_for_this_selection(self):
        self.guard.observe(6, 'stationary', self.pose, 3)
        ranking = [self.ranking[0], dict(target_index=5, candidate='stationary', lpips=.28),
                   self.ranking[1]]
        chosen, _ = self.guard.choose(ranking, self.pose)
        self.assertEqual(chosen['candidate'], 'forward')
        # A different, unrepeated target may still arrive normally next step.
        chosen, event = self.guard.choose(ranking[1:], self.pose)
        self.assertEqual(chosen['candidate'], 'stationary')
        self.assertIsNone(event)

    def test_failed_escape_triggers_again_without_forgetting_history(self):
        self.guard.observe(6, 'stationary', self.pose, 3)
        self.guard.choose(self.ranking, self.pose)
        self.guard.observe(6, 'forward', self.pose, 4)
        chosen, event = self.guard.choose(self.ranking, self.pose)
        self.assertEqual(chosen['candidate'], 'forward')
        self.assertIsNotNone(event)

    def test_repeated_motion_is_not_blocked(self):
        self.guard.observe(6, 'forward', self.pose, 3)
        chosen, event = self.guard.choose(self.ranking[1:], self.pose)
        self.assertEqual(chosen['candidate'], 'forward')
        self.assertIsNone(event)

    def test_only_stationary_candidates_fails_closed(self):
        self.guard.observe(6, 'stationary', self.pose, 3)
        with self.assertRaisesRegex(RuntimeError, 'no eligible motion'):
            self.guard.choose([self.ranking[0], self.ranking[2]], self.pose)


class WindowNavigation(unittest.TestCase):
    def test_time_sampling_uses_fixed_ticks_not_pose_thresholds(self):
        rows = [{'timeline_ns': int(t * 1e9)} for t in (0, .2, .51, .9, 1.01, 1.49, 1.51)]
        self.assertEqual([i for i, _ in nav.select_timed_targets(rows, .5)], [0, 2, 4, 6])

    def test_window_center_and_edges(self):
        self.assertEqual(nav.target_window(5, 12, 5), [5, 6, 7, 8, 9])
        self.assertEqual(nav.target_window(1, 12, 5), [1, 2, 3, 4, 5])
        self.assertEqual(nav.target_window(11, 12, 5), [11])
        self.assertEqual(nav.target_window(1, 3, 5), [1, 2])

    def test_ten_frame_window_center_and_edges(self):
        self.assertEqual(nav.target_window(7, 20, 10), list(range(7, 17)))
        self.assertEqual(nav.target_window(1, 20, 10), list(range(1, 11)))
        self.assertEqual(nav.target_window(19, 20, 10), [19])
        self.assertEqual(nav.target_window(1, 3, 10), [1, 2])

    def test_joint_minimum_all_models_single_generation_and_fixed_time(self):
        for model, metric in itertools.product(('opennwm', 'opennwm-dmd', 'nwm-b', 'nwm-dmd'), ('lpips', 'dinov2')):
            for stop_wins, reselect in ((False, False), (True, False), (True, True)):
                with self.subTest(model=model, metric=metric, stop_wins=stop_wins, reselect=reselect), tempfile.TemporaryDirectory() as tmp:
                    engine = nav.Engine.__new__(nav.Engine)
                    engine.args = SimpleNamespace(model=model, batch_size=14, seed=0, matching_metric=metric)
                    engine.torch, engine.np, engine.device = torch, np, torch.device('cpu')
                    engine.config = SimpleNamespace(model=SimpleNamespace(generator=SimpleNamespace(input_size=28)),
                                                    dataset=SimpleNamespace(action_stats={}))
                    engine.models = ()
                    engine.transform = lambda im: torch.zeros(3, 224, 224)
                    engine.encode_history = MagicMock(return_value=(torch.zeros(4, 4, 28, 28), 1))
                    engine.lpips_distances = MagicMock(side_effect=[torch.tensor([[.01 if stop_wins else .9, .8, .7]]),
                        torch.tensor([[.01 if stop_wins else .9, .8, .7], [.6, .4, .2], [.5, .3, .1]])])
                    engine.dino_target_features = MagicMock(return_value=torch.zeros(3, 384))
                    engine.tokenizer = SimpleNamespace(decode=MagicMock(return_value=torch.zeros(3, 3, 224, 224)))
                    engine.scorer = MagicMock(side_effect=[torch.tensor([.6, .5]), torch.tensor([.4, .3]), torch.tensor([.2, .1])])
                    engine.dino = MagicMock()
                    engine.dino.distances.side_effect = [torch.tensor([[.01 if stop_wins else .9, .8, .7]]),
                                                        torch.tensor([[.01 if stop_wins else .9, .8, .7], [.6, .4, .2], [.5, .3, .1]])]
                    forward = MagicMock(return_value=torch.zeros(3, 4, 28, 28))
                    im = Image.new('RGB', (224, 224), 'red')
                    paths = {'stationary': [[0, 0, 0]], 'left': [[0, 0, .2]], 'forward': [[.2, 0, 0]]}
                    guard = nav.NavigationLoopGuard()
                    pose = SimpleNamespace(x=0., y=0., yaw=0.)
                    if reselect: guard.observe(3, 'stationary', pose, 0)
                    with patch.dict(sys.modules, {'isolated_nwm_infer': SimpleNamespace(model_forward_wrapper=forward),
                                                  'misc': SimpleNamespace(normalize_data=lambda x, stats: x)}), \
                         patch.object(torch.cuda, 'synchronize'), \
                         patch.object(torch.amp, 'autocast', side_effect=lambda *a, **kw: contextlib.nullcontext()):
                        result, prediction = engine.select([im]*4, im, paths, 1, Path(tmp),
                                                  window_goals=[(3, im), (4, im), (5, im)],
                                                  loop_guard=guard, source_pose=pose)
                    self.assertEqual(result['matching_metric'], metric)
                    if metric == 'dinov2':
                        engine.scorer.assert_not_called()
                        self.assertEqual(engine.dino.encode.call_count, 2)
                        self.assertNotIn('selected_lpips', result)
                    forward.assert_called_once()
                    engine.tokenizer.decode.assert_called_once()
                    engine.encode_history.assert_called_once()
                    torch.testing.assert_close(forward.call_args.kwargs['rel_t'], torch.full((3,), .015625))
                    self.assertEqual(len(result['ranking']), 9)
                    self.assertEqual(result['selected'], 'stationary' if stop_wins and not reselect else 'forward')
                    self.assertEqual(result['selected_target_index'], 3 if stop_wins and not reselect else 5)
                    self.assertAlmostEqual(result['current_goal_lpips'], .01 if stop_wins and not reselect else .7)
                    self.assertEqual(result['inferred_candidate_count'], 3)
                    self.assertEqual(result['loop_avoidance'] is not None, reselect)
                    self.assertAlmostEqual(result['selected_score'], .01 if stop_wins and not reselect else .1)
                    self.assertEqual(result['selected_image_source'],
                                     'model_prediction')
                    expected = (0, 0, 0)
                    self.assertEqual(prediction.getpixel((0, 0)), expected)
                    self.assertFalse(list(Path(tmp).glob('*.png')))


if __name__ == '__main__':
    unittest.main()
