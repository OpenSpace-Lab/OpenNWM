"""Offline calibration contracts, frozen commands and rejection of invalid measurements."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibrated_actions import (candidate_specs, nominal_proposals, load_calibration,
    calibrated_proposals, build_calibration, validate_schedule)
from calibration_fixture import calibration_fixture
import jetson_calibrate_actions as c


class Calibration(unittest.TestCase):
    def load(self, data):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'calibration.json'
            path.write_text(json.dumps(data))
            return load_calibration(path)

    def test_exact_geometry_units_and_measured_not_nominal_condition(self):
        specs = candidate_specs()
        self.assertEqual(len(specs), 7)
        self.assertEqual(len({s['name'] for s in specs}), 7)
        self.assertEqual({s['distance_m'] for s in specs}, {0., .25, .5})
        self.assertEqual({abs(s['yaw_rad']) for s in specs}, {0., math.pi/9, math.pi/4})
        p = nominal_proposals()['rotate_left_pi4']
        self.assertEqual(p['metric_target'], [0., 0., math.pi/4])
        self.assertTrue(all(not (s['distance_m'] and s['yaw_rad']) for s in specs))
        calibrated = calibrated_proposals(self.load(calibration_fixture()))
        self.assertAlmostEqual(calibrated['forward_50']['effective_candidate'][0], 1.92)
        self.assertNotEqual(calibrated['forward_50']['effective_candidate'], p['raw_candidate'])
        self.assertEqual(calibrated['stationary']['effective_candidate'], [0.,0.,0.])

    def test_incomplete_or_forged_ready_rejected(self):
        data = calibration_fixture()
        data['actions']['forward_25']['samples'].pop()
        with self.assertRaises(ValueError): self.load(data)
        data = calibration_fixture(); data['actions'].pop('stationary')
        with self.assertRaises(ValueError): self.load(data)
        data = calibration_fixture(); data['waypoint_spacing_m'] = .3
        with self.assertRaises(ValueError): self.load(data)

    def test_bad_feedback_nonfinite_bounds_and_variance_rejected(self):
        for mutation in ('failed', 'nan', 'path', 'turn', 'spread', 'short', 'wrong_direction'):
            data = calibration_fixture()
            sample = data['actions']['forward_25']['samples'][0]
            if mutation == 'failed': sample['status'] = 'failed'
            if mutation == 'nan': sample['measured_m_rad'][0] = float('nan')
            if mutation == 'path': sample['measured_path_m'] = 1.01
            if mutation == 'turn': sample['measured_accumulated_turn_rad'] = math.pi/3
            if mutation == 'spread': sample['measured_m_rad'][1] = .1
            if mutation == 'short':
                for v in data['actions']['forward_25']['samples']: v['measured_m_rad'][0] = .1
            if mutation == 'wrong_direction':
                for v in data['actions']['forward_25']['samples']: v['measured_m_rad'][0] = -.25
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.load(data)

    def test_sample_schedule_and_stationary_cannot_be_overridden(self):
        data = calibration_fixture()
        data['actions']['forward_25']['samples'][0]['proposal']['schedule']['duration_s'] += 1
        with self.assertRaises(ValueError): self.load(data)
        data = calibration_fixture()
        data['actions']['stationary']['proposal']['schedule']['vx_m_s'] = .1
        with self.assertRaises(ValueError): self.load(data)

    def test_summary_is_recomputed_and_tuned_commands_are_frozen(self):
        data = calibration_fixture()
        data['actions']['forward_25']['summary']['model_action'] = [100,0,0]
        schedule = data['actions']['forward_25']['proposal']['schedule']
        schedule['duration_s'] *= 1.1
        for sample in data['actions']['forward_25']['samples']:
            sample['proposal']['schedule'] = schedule.copy()
        checked = self.load(data)
        p = calibrated_proposals(checked)['forward_25']
        self.assertEqual(p['effective_candidate'], [1.,0.,0.])
        self.assertEqual(p['schedule'], schedule)
        self.assertEqual(data['actions']['forward_25']['summary']['model_action'], [100,0,0])

    def test_offline_tuning_is_bounded_and_stall_not_amplified(self):
        proposal = nominal_proposals()['forward_25']
        with self.assertRaises(RuntimeError): c.tune_schedule(proposal, [0.,0.,0.])
        adjusted = c.tune_schedule(proposal, [.18,0.,0.])
        validate_schedule(adjusted)
        self.assertAlmostEqual(adjusted['duration_s'], proposal['schedule']['duration_s']*1.2)
        self.assertEqual(adjusted['vx_m_s'], .30)
        turn = nominal_proposals()['rotate_left_pi9']
        with self.assertRaises(RuntimeError): c.tune_schedule(turn, [0.,0.,-.1])

    def test_plan_is_file_only_and_not_navigation_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'plan'
            result = subprocess.run([sys.executable, str(Path(c.__file__)), 'plan', '--output', str(out)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads((out/'calibration.json').read_text())['ready'])
            self.assertEqual(len(json.loads((out/'plan.json').read_text())['candidates']), 7)

    def test_wrong_move_token_has_no_control_side_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            io, capture = MagicMock(), MagicMock()
            args = type('Args', (), {'output':Path(tmp)})()
            with patch('builtins.input', return_value='yes'), self.assertRaises(KeyboardInterrupt):
                c.trial(args, io, capture, nominal_proposals()['forward_25'], 1, 'repeat')
            self.assertEqual(io.mock_calls, [])
            self.assertEqual(capture.mock_calls, [])

    def test_collect_freezes_final_trials_and_always_releases_robot(self):
        from types import SimpleNamespace
        fixture = calibration_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp), repeats=3, max_tuning_rounds=3)
            io, capture, ros = MagicMock(), MagicMock(), MagicMock()
            def execute(args, io, capture, proposal, number, phase):
                sample = copy.deepcopy(fixture['actions'][proposal['candidate']]['samples'][0])
                sample['proposal'] = copy.deepcopy(proposal)
                sample['phase'] = phase
                return sample
            with patch.dict(sys.modules, {'rclpy':ros}), patch.object(c.nav, 'robot_adapter', return_value=io), \
                 patch.object(c.nav, 'Capture', return_value=capture), patch.object(c, 'trial', side_effect=execute) as run:
                samples, status = {}, {}
                c.collect(args, samples, nominal_proposals(), status)
                self.assertEqual(status['status'], 'complete')
                self.assertEqual(run.call_count, 24)  # 6 pilots + 18 moving repeats; stationary is built in.
                self.assertTrue(all(len(v)==3 for v in samples.values()))
                self.assertTrue(all(s['phase'].startswith('repeat') for v in samples.values() for s in v))
                io.stop.assert_called_once(); io.close.assert_called_once(); capture.close.assert_called_once()
                ros.shutdown.assert_called_once()

    def test_collect_failure_stops_and_cleans_up(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp), repeats=3, max_tuning_rounds=3)
            io, capture, ros = MagicMock(), MagicMock(), MagicMock()
            with patch.dict(sys.modules, {'rclpy':ros}), patch.object(c.nav, 'robot_adapter', return_value=io), \
                 patch.object(c.nav, 'Capture', return_value=capture), patch.object(c, 'trial', side_effect=RuntimeError('camera lost')):
                with self.assertRaisesRegex(RuntimeError, 'camera lost'):
                    c.collect(args, {}, nominal_proposals(), {})
                io.stop.assert_called_once(); io.close.assert_called_once(); capture.close.assert_called_once()
                ros.shutdown.assert_called_once()


if __name__ == '__main__': unittest.main()
