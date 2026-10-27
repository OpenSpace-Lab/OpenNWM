import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jetson_calibrate_actions as c
from calibrated_actions import nominal_proposals, validate_schedule


class ForwardSpeed(unittest.TestCase):
    def test_new_forward_seed_executes_and_tunes_at_higher_speed(self):
        for name in ('forward_25', 'forward_50'):
            p = c.initial_proposals()[name]
            schedule = p['schedule']
            validate_schedule(schedule)
            guarded = MagicMock()
            now = [0.]
            c.nav.execute_schedule(guarded, schedule, lambda: now[0],
                                   lambda dt: now.__setitem__(0, now[0]+dt))
            self.assertAlmostEqual(now[0], 2.)
            for call in guarded.move.call_args_list:
                self.assertEqual(call.args, (.30, 0., 0.))
            guarded.pause_motion.assert_called_once()
            guarded.stop.assert_not_called()
            tuned = c.tune_schedule(p, [p['nominal']['distance_m']*.9, 0., 0.])
            self.assertAlmostEqual(tuned['vx_m_s'], .30)

    def test_stationary_preserved_and_forward_tuning_uses_fixed_speed(self):
        baseline = nominal_proposals()
        for name, proposal in c.initial_proposals().items():
            if name == 'stationary':
                self.assertEqual(proposal, baseline[name])
        tuned = c.tune_schedule(baseline['forward_25'], [.22, 0., 0.])
        self.assertEqual(tuned['vx_m_s'], .30)
        self.assertAlmostEqual(tuned['duration_s'], baseline['forward_25']['schedule']['duration_s']*.25/.22)

    def test_unsafe_translation_speeds_rejected(self):
        for vx, vy, yaw in ((.301, 0., 0.), (-.18, 0., 0.), (0., .18, 0.), (.301, 0., .1)):
            schedule = dict(vx_m_s=vx, vy_m_s=vy, yaw_rad_s=yaw, duration_s=1.)
            with self.assertRaises(ValueError):
                validate_schedule(schedule)
            with self.assertRaises(ValueError):
                c.nav.execute_schedule(MagicMock(), schedule)
