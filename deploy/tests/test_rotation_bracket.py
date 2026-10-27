import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jetson_calibrate_actions as c

class RotationBracket(unittest.TestCase):
    def setUp(self):
        self.p = c.nominal_proposals()['rotate_right_pi9']
        self.p['schedule'] = dict(vx_m_s=0., vy_m_s=0., yaw_rad_s=-.35, duration_s=1.4844432269)
        self.history = []
        for duration, yaw in [(1.55104917, -.27733016), (1.75702660, -.35830644),
                              (1.54054220, -.26698041), (1.81277570, -.38364571)]:
            p = copy.deepcopy(self.p)
            p['schedule']['duration_s'] = duration
            self.history.append(dict(proposal=p, status='succeeded', phase='pilot-0',
                controller={'completion_basis': 'elapsed_command_time'},
                measured_m_rad=[.017, -.020, yaw], measured_path_m=.03,
                measured_accumulated_turn_rad=abs(yaw)))

    def test_real_oscillation_interpolates_inside_bracket(self):
        result = c.bracket_rotation_schedule(self.p, self.history)
        # The 20-degree target lies near the high edge; keep the 75% interior cap.
        self.assertAlmostEqual(result['duration_s'], 1.55104917 + .75*(1.75702660-1.55104917))
        self.assertEqual(result['yaw_rad_s'], -.35)
        self.assertEqual(self.p['schedule']['duration_s'], 1.4844432269)

    def test_excludes_faults_reserve_wrong_direction_drift_and_speed(self):
        for field, value in [('avoidance_intervened', True), ('status', 'stopped_at_reserve'),
                             ('measured_m_rad', [0, 0, .3]), ('measured_m_rad', [.05, 0, -.3])]:
            records = copy.deepcopy(self.history)
            for r in records: r[field] = value
            self.assertIsNone(c.bracket_rotation_schedule(self.p, records))
        records = copy.deepcopy(self.history)
        for r in records: r['proposal']['schedule']['yaw_rad_s'] = -.30
        self.assertIsNone(c.bracket_rotation_schedule(self.p, records))
        self.assertIsNone(c.bracket_rotation_schedule(self.p, self.history[:1]))

    def test_history_survives_resume_without_duplicate_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'pilot-history.json').write_text(json.dumps(self.history))
            trial = root/'026-rotate_right_pi9-pilot-7'
            trial.mkdir()
            (trial/'execution.json').write_text(json.dumps(self.history[-1]))
            restored = c.restore_pilot_history(root/'calibration.json')
            self.assertEqual(len(restored), 4)
            self.assertEqual(c.bracket_rotation_schedule(self.p, restored),
                             c.bracket_rotation_schedule(self.p, self.history))

if __name__ == '__main__': unittest.main()
