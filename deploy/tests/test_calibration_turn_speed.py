from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jetson_calibrate_actions as c
from calibrated_actions import nominal_proposals, validate_schedule, TURN_CALIBRATION_RATE, ARC_CALIBRATION_SPEED, angular_speed_limit


class TurnSpeed(unittest.TestCase):
    def test_all_turns_start_faster_and_execute_within_new_limits(self):
        old=nominal_proposals()
        for name,p in c.initial_proposals().items():
            if not p['nominal']['yaw_rad']:continue
            schedule=p['schedule']; baseline=old[name]['schedule']
            if not p['nominal']['distance_m']:
                self.assertAlmostEqual(schedule['yaw_rad_s']/baseline['yaw_rad_s'], .45/.35)
            else:
                self.assertEqual(schedule['vx_m_s'], .30)
                self.assertAlmostEqual(schedule['yaw_rad_s']*schedule['duration_s'], p['nominal']['yaw_rad'])
            self.assertLessEqual(abs(schedule['yaw_rad_s']),angular_speed_limit(schedule['vx_m_s'], schedule['vy_m_s'])+1e-6)
            validate_schedule(schedule)
            guarded=MagicMock();now=[0.]
            c.nav.execute_schedule(guarded,schedule,lambda:now[0],lambda dt:now.__setitem__(0,now[0]+dt))
            self.assertTrue(guarded.move.called)
            guarded.pause_motion.assert_called_once()
            guarded.stop.assert_not_called()
            if p['nominal']['distance_m']:
                self.assertGreater(schedule['vx_m_s'],baseline['vx_m_s'])
                self.assertLessEqual(schedule['vx_m_s'],ARC_CALIBRATION_SPEED+1e-6)

    def test_rotation_tuning_does_not_reset_to_old_angular_limit(self):
        p=c.initial_proposals()['rotate_left_pi4']
        tuned=c.tune_schedule(p,[0.,0.,.6])
        self.assertAlmostEqual(tuned['yaw_rad_s'],.45)

    def test_excess_turn_rate_still_rejected(self):
        schedule=dict(vx_m_s=0.,vy_m_s=0.,yaw_rad_s=.451,duration_s=1.)
        with self.assertRaises(ValueError):validate_schedule(schedule)
        with self.assertRaises(ValueError):c.nav.execute_schedule(MagicMock(),schedule)
