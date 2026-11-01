"""CPU regression check for clip-wide metric overlays and unmodified predictions."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


MODULE = Path(__file__).resolve().parents[1] / 'tools/render_go2_gallery.py'
spec = importlib.util.spec_from_file_location('go2_clip_renderer', MODULE)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class ClipMeanTests(unittest.TestCase):
    def test_shortened_clips_require_explicit_parent_bounds(self):
        valid = {'start_s': 71, 'end_s': 80,
                 'shortened_from': {'id': 'sand', 'start_s': 57, 'end_s': 82}}
        renderer.validate_duration(valid)
        for invalid in ({'start_s': 71, 'end_s': 80},
                        {**valid, 'start_s': 56},
                        {**valid, 'end_s': 77},
                        {**valid, 'end_s': 83}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                renderer.validate_duration(invalid)

    def test_overlay_is_clip_mean_across_local_plans_without_changing_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / 'evaluation'
            frame_paths = []

            def write_json(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))

            def write_image(path, seed, size):
                path.parent.mkdir(parents=True, exist_ok=True)
                # Spatial detail makes interpolation or unexpected image edits detectable.
                y, x = np.indices((size[1], size[0]))
                pixels = np.stack(((x + seed) % 256, (y * 2 + seed) % 256,
                                   (x + y * 3 + seed * 11) % 256), axis=-1).astype(np.uint8)
                Image.fromarray(pixels).save(path)
                frame_paths.append(path)

            write_json(root / 'alignment_manifest.json', {
                'third_video': str(root / 'third.mp4'), 'ego_video': str(root / 'ego.mp4'),
                'alignment': {'offset_s': 0},
            })
            protocol = {'diffusion_sampler': 'DDPM', 'horizon': 2, 'fps': 0.2,
                        'goal_interval_s': 10}
            for model in renderer.MODELS:
                write_json(evaluation / 'outputs' / model / 'protocol.json', protocol)

            # Only the two middle windows belong to the selected 20-second clip.
            # Excluded windows have no result files, so including them is also a failure.
            rows = []
            metrics = {
                1: {'opennwm': {'ate_m': 0.1, 'rpe_m': 0.05},
                    'nwm': {'ate_m': 0.3, 'rpe_m': 0.1}},
                2: {'opennwm': {'ate_m': 0.9, 'rpe_m': 0.45},
                    'nwm': {'ate_m': 1.7, 'rpe_m': 0.7}},
            }
            for window in range(4):
                frames = [root / 'recorded' / f'{window}_{step}.png' for step in range(3)]
                reference = [[0, 0], [0.1, 0.02 * window], [0.2, 0.04 * window]]
                rows.append({'id': window, 'start_time_s': window * 10,
                             'end_time_s': (window + 1) * 10,
                             'frames': list(map(str, frames)), 'reference_xy_m': reference,
                             'reference_world_xy_m': reference})
                if window not in metrics:
                    continue
                for step, path in enumerate(frames):
                    write_image(path, window * 30 + step, (320, 240))
                for model_index, model in enumerate(renderer.MODELS):
                    directory = evaluation / 'outputs' / model / f'{window:06d}'
                    write_json(directory / 'result.json', {
                        'metrics': metrics[window][model], 'planned_xy_m': reference,
                        'planned_yaw_rad': [0, 0, 0], 'reference_xy_m': reference,
                    })
                    for step in range(2):
                        write_image(directory / 'predicted_frames' / f'{step:03d}.png',
                                    100 * model_index + 30 * window + step, (224, 224))

            write_json(evaluation / 'windows.json', {'fps': 0.2, 'horizon': 2, 'windows': rows})
            clip = renderer.Clip({'id': 'mean_test', 'episode_id': 'synthetic_episode',
                                  'episode_root': str(root), 'evaluation_root': str(evaluation),
                                  'start_s': 10, 'end_s': 30})
            source_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in frame_paths}
            expected = {'opennwm': {'ate_m': 0.5, 'rpe_m': 0.25},
                        'nwm': {'ate_m': 1.0, 'rpe_m': 0.4}}
            report = clip.report()
            for model in renderer.MODELS:
                for metric, value in expected[model].items():
                    self.assertAlmostEqual(report['metrics'][model][metric], value)
            self.assertEqual([plan['id'] for plan in report['plans']], [1, 2])
            self.assertEqual(report['metric_aggregation']['plan_count'], 2)
            self.assertEqual(report['metric_aggregation']['display'], 'constant_for_entire_clip')

            third = Image.new('RGB', renderer.THIRD, (72, 120, 160))
            third_before = third.tobytes()
            overlays, metric_text = [], []
            for window, time in ((1, 12.5), (2, 22.5)):
                with patch.object(clip, 'text', wraps=clip.text) as draw_text:
                    composite, frames, frame_index = clip.compose(time, third)
                self.assertEqual(frame_index, 1)
                overlays.append(composite.crop((1248, 800, 1896, 1062)).tobytes())
                metric_text.append([call.args[2] for call in draw_text.call_args_list
                                    if call.args[1][0] >= 1248 and call.args[1][1] >= 800])
                for model, x in (('opennwm', 432), ('nwm', 840)):
                    source = evaluation / 'outputs' / model / f'{window:06d}' / 'predicted_frames/000.png'
                    with Image.open(source) as image:
                        expected_frame = image.convert('RGB')
                    self.assertEqual(frames[model][frame_index].tobytes(), expected_frame.tobytes())
                    expected_panel = expected_frame.resize((384, 288), Image.Resampling.LANCZOS)
                    self.assertEqual(composite.crop((x, 774, x + 384, 1062)).tobytes(),
                                     expected_panel.tobytes())

            self.assertEqual(metric_text[0], ['Clip mean', 'ATE (m)', 'RPE (m)',
                                            'OpenNWM', '0.500', '0.250', 'NWM', '1.000', '0.400'])
            self.assertEqual(metric_text[0], metric_text[1])
            self.assertEqual(overlays[0], overlays[1])
            self.assertEqual(third.tobytes(), third_before)
            self.assertEqual(source_hashes,
                             {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in frame_paths})


if __name__ == '__main__':
    unittest.main()
