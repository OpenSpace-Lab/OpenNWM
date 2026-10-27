"""Stable-camera recovery without robot hardware or wall-clock sleeps."""
import itertools
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jetson_navigate as nav


class AutoRecovery(unittest.TestCase):
    def run_recovery(self, scores, fault=None, duplicate=False):
        clock = [0.]
        pose = SimpleNamespace(x=0., y=0., yaw=0., source=0., received=0., speed=0., yaw_speed=0.)
        io = MagicMock(); io.read_pose.return_value = pose
        if fault: io.read_pose.side_effect = fault
        capture = MagicMock(); sequence = itertools.count(1)
        image = Image.new('RGB', (224, 224))
        capture.latest.side_effect = lambda **kw: (image, {'sequence': 1 if duplicate else next(sequence),
                                                         'timestamp_monotonic': clock[0]})
        engine = MagicMock(); engine.distance.side_effect = scores
        with tempfile.TemporaryDirectory() as tmp, patch.object(nav, 'stopped_pose', return_value=pose), \
             patch.object(nav.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(nav.time, 'sleep', side_effect=lambda dt: clock.__setitem__(0, clock[0]+dt)):
            current, history = nav.automatic_replan_context(capture, io, engine, Path(tmp))
            io.stop.assert_called_once()
            self.assertFalse(list(Path(tmp).glob('*.png')))
            self.assertTrue(history.context()[1]['cold_start_repetition'])
            self.assertEqual(history.context()[1]['unique_real_frames'], 1)
            self.assertGreaterEqual(clock[0], .4)
            return clock[0]

    def test_stable_frames_rebuild_context(self):
        self.run_recovery(itertools.repeat(0.))

    def test_changed_frame_restarts_stability_period(self):
        elapsed = self.run_recovery(itertools.chain([.2], itertools.repeat(0.)))
        self.assertGreaterEqual(elapsed, .6)

    def test_unstable_scene_times_out(self):
        with self.assertRaisesRegex(RuntimeError, '10 秒'):
            self.run_recovery(itertools.repeat(.2))

    def test_remote_takeover_never_retries(self):
        with self.assertRaisesRegex(RuntimeError, 'physical_remote_takeover'):
            self.run_recovery(itertools.repeat(0.), fault=RuntimeError('physical_remote_takeover'))

    def test_duplicate_camera_frame_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'fresh camera'):
            self.run_recovery(itertools.repeat(0.), duplicate=True)

    def test_nonfinite_distance_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Invalid recovery'):
            self.run_recovery(itertools.repeat(float('nan')))


if __name__ == '__main__':
    unittest.main()
