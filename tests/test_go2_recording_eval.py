"""CPU checks for the recording adapter's geometry and timestamp provenance."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / 'tools/evaluate_go2_recording.py'
spec = importlib.util.spec_from_file_location('go2_eval', MODULE)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class GeometryTests(unittest.TestCase):
    def test_fixed_clips_keep_grid_ids_and_deduplicate_overlap(self):
        manifest = dict(aligned_duration_s=60, clips=[
            dict(id='a', start_s=4, duration_s=20), dict(id='b', start_s=20, duration_s=20)])
        schedule, _ = adapter.window_schedule(manifest, 2, True)
        self.assertEqual([r['id'] for r in schedule], list(range(2, 20)))
        self.assertEqual(schedule[8]['clip_ids'], ['a', 'b'])
        shorter, _ = adapter.window_schedule(manifest, 1, True)
        self.assertEqual(len(shorter), 36)
        self.assertEqual(shorter[-1]['offset'], 39)
        with self.assertRaises(ValueError):
            adapter.window_schedule(dict(aligned_duration_s=60,
                clips=[dict(id='bad', start_s=3, duration_s=20)]), 2, True)

    def test_full_recording_preserves_nondivisible_tail(self):
        schedule, _ = adapter.window_schedule(dict(aligned_duration_s=5.5), 2)
        self.assertEqual([r['offset'] for r in schedule], [0, 2])

    def test_current_body_steps_rotate_into_initial_frame(self):
        actual = adapter.integrate_body_deltas([[1, 0, np.pi / 2], [1, 0, 0], [1, 0, 0]])
        np.testing.assert_allclose(actual[:, :2], [[0, 0], [1, 0], [1, 1], [1, 2]], atol=1e-12)
        np.testing.assert_allclose(actual[:, 2], [0, np.pi / 2, np.pi / 2, np.pi / 2])

    def test_local_frame_and_wrapped_yaw(self):
        xy, yaw = adapter.local_poses([[10, 4], [10, 5], [9, 5]], [np.pi / 2, np.pi, -np.pi / 2])
        np.testing.assert_allclose(xy, [[0, 0], [1, 0], [1, 1]], atol=1e-12)
        np.testing.assert_allclose(yaw, [0, np.pi / 2, -np.pi])

    def test_invalid_pair_uses_real_nearest_valid_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            alignment = root / 'alignment.json'
            alignment.write_text(json.dumps({'aligned_start_host_s': 1, 'aligned_duration_s': 2}))
            with (source / 'aligned_pose.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['frame_index', 'file', 'timeline_ns', 'pair_valid', 'x', 'y', 'yaw'])
                writer.writeheader()
                for i in range(61):
                    filename = f'{i:03d}.jpg'
                    (source / filename).touch()
                    writer.writerow(dict(frame_index=i, file=filename, timeline_ns=i * 50000000,
                                         pair_valid=i != 40, x=i * .01, y=0, yaw=0))
            adapter.prepare(argparse.Namespace(root=root, source=source, alignment=alignment, spacing=adapter.SPACING))
            result = json.loads((root / 'evaluation/windows.json').read_text())
            self.assertEqual(result['skipped_windows'], [])
            window = result['windows'][0]
            self.assertEqual(window['source_frame_indices'][4], 39)
            self.assertEqual(window['invalid_pair_fallbacks'][0]['rejected_source_frame_index'], 40)
            self.assertEqual(window['context_source_frame_indices'], [5, 10, 15, 20])
            self.assertTrue(all(index <= window['source_frame_indices'][0]
                                for index in window['context_source_frame_indices']))
            self.assertAlmostEqual(window['reference_xy_m'][4][0], .19)
            adapter.prepare(argparse.Namespace(root=root, source=source, alignment=alignment,
                spacing=adapter.SPACING, horizon_seconds=1, evaluation_name='evaluation_1s'))
            shorter = json.loads((root / 'evaluation_1s/windows.json').read_text())
            self.assertEqual(shorter['horizon'], 4)
            self.assertEqual(shorter['goal_interval_s'], 1)
            self.assertEqual(len(shorter['windows']), 2)
            self.assertEqual(len(shorter['windows'][0]['frames']), 5)
            self.assertEqual(shorter['windows'][1]['end_time_s'], 2)


if __name__ == '__main__':
    unittest.main()
