import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jetson_calibrate_actions as c
from calibration_fixture import calibration_fixture


class AdaptiveCalibration(unittest.TestCase):
    def test_fixed_linear_speed_changes_time(self):
        for name in ('forward_25','forward_50'):
            p = c.initial_proposals()[name]
            for gain in (.7, 1.3):
                measured = c.nav.twist_endpoint([p['nominal']['distance_m']*gain, 0., p['nominal']['yaw_rad']*.9])
                result = c.tune_schedule(p, measured)
                self.assertEqual(result['vx_m_s'], .30)
                self.assertEqual(result['vy_m_s'], 0.)
                if gain < 1:
                    self.assertGreater(result['duration_s'], p['schedule']['duration_s'])
                else:
                    self.assertLess(result['duration_s'], p['schedule']['duration_s'])
                self.assertLessEqual(abs(result['yaw_rad_s']), .75)

    def test_migration_preserves_fixed_speed_samples_and_recollects_old_speeds(self):
        fixture = calibration_fixture()
        proposals = c.initial_proposals()
        samples = {k: copy.deepcopy(v['samples']) for k,v in fixture['actions'].items()}
        name = 'forward_50'
        proposals[name]['schedule'].update(vx_m_s=.25, duration_s=1.8)
        old = copy.deepcopy(samples[name])
        forward = copy.deepcopy((proposals['forward_25'], samples['forward_25']))
        changes = c.upgrade_fixed_translation(proposals, samples)
        self.assertEqual([x['candidate'] for x in changes], [name])
        self.assertEqual(changes[0]['archived_samples'], old)
        self.assertEqual(samples[name], [])
        self.assertEqual((proposals['forward_25'], samples['forward_25']), forward)
        self.assertEqual(proposals[name]['schedule']['vx_m_s'], .30)
        self.assertAlmostEqual(proposals[name]['schedule']['duration_s'], 1.5)
        proposals[name]['schedule']['duration_s'] = 1.8
        self.assertEqual(c.upgrade_fixed_translation(proposals, samples), [])
        self.assertEqual(proposals[name]['schedule']['duration_s'], 1.8)

    def run_collection(self, make_trial, proposals, samples):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output=Path(tmp), repeats=3, max_tuning_rounds=0)
            status = {}
            with patch.dict(sys.modules, {'rclpy': MagicMock()}), \
                 patch.object(c.nav, 'robot_adapter', return_value=MagicMock()) as adapter, \
                 patch.object(c.nav, 'Capture', return_value=MagicMock()), \
                 patch.object(c, 'trial', side_effect=make_trial):
                c.collect(args, samples, proposals, status)
                adapter.return_value.stop.assert_called_once()
            archives = [json.loads(p.read_text()) for p in Path(tmp).glob('rejected-*.json')]
            saved = json.loads((Path(tmp)/'calibration.json').read_text())
            return status, archives, saved

    def test_failed_repeats_retune_and_start_new_frozen_group(self):
        fixture = calibration_fixture()['actions']['forward_25']['samples'][0]
        proposals = {'forward_25': c.initial_proposals()['forward_25']}
        samples = {}; calls = []
        def trial(args, io, capture, p, number, phase):
            calls.append((number, phase, copy.deepcopy(p['schedule'])))
            result = copy.deepcopy(fixture)
            result['proposal'] = copy.deepcopy(p)
            if number in (2,3,4): result['measured_m_rad'] = [.35, 0., 0.]
            return result
        status, archives, saved = self.run_collection(trial, proposals, samples)
        self.assertEqual(status['status'], 'complete')
        self.assertEqual(len(calls), 8)
        self.assertEqual(len(archives), 1)
        self.assertEqual(len(archives[0]['samples']), 3)
        self.assertEqual(archives[0]['next_schedule']['vx_m_s'], .30)
        self.assertLess(archives[0]['next_schedule']['duration_s'], 2.)
        self.assertTrue(all(x['measured_m_rad'][0] == .25 for x in samples['forward_25']))
        self.assertEqual(calls[5][2], calls[6][2])
        self.assertEqual(calls[6][2], calls[7][2])
        self.assertEqual(len(saved['actions']['forward_25']['samples']), 3)

    def test_default_can_continue_beyond_old_eight_pilot_limit(self):
        fixture = calibration_fixture()['actions']['forward_25']['samples'][0]
        proposals = {'forward_25': c.initial_proposals()['forward_25']}
        calls = []
        def trial(args, io, capture, p, number, phase):
            calls.append(number)
            result = copy.deepcopy(fixture); result['proposal'] = copy.deepcopy(p)
            if number <= 9: result['measured_m_rad'] = [.31, 0., 0.]
            return result
        status, _, _ = self.run_collection(trial, proposals, {})
        self.assertEqual(status['status'], 'complete')
        self.assertEqual(len(calls), 13)

    def test_scatter_retry_keeps_linear_speed_and_faults_cannot_retry(self):
        p = c.initial_proposals()['forward_50']
        records = calibration_fixture()['actions']['forward_50']['samples']
        result = c.retry_schedule(p, records, 'not repeatable', 1)
        self.assertEqual(result['vx_m_s'], .30)
        self.assertEqual(result['vy_m_s'], 0.)
        for flag in ('error','avoidance_intervened','feedback_error','budget_error'):
            bad = copy.deepcopy(records); bad[0][flag] = True
            with self.assertRaises(RuntimeError): c.retry_schedule(p, bad, 'not repeatable', 1)
        bad = copy.deepcopy(records); bad[0]['status'] = 'stopped_at_reserve'
        with self.assertRaises(RuntimeError): c.retry_schedule(p, bad, 'not repeatable', 1)

    def test_only_forward_and_in_place_turn_commands_are_created(self):
        proposals=c.initial_proposals()
        self.assertEqual(len(proposals),7)
        for p in proposals.values():
            s=p['schedule']
            self.assertEqual(s['vy_m_s'],0.)
            self.assertFalse(s['vx_m_s'] and s['yaw_rad_s'])
            self.assertNotIn('_arc_',p['candidate'])
