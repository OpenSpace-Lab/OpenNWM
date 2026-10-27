"""Forward direction diagnostics removed; rotation drift and motion validity remain."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jetson_calibrate_actions as c
from calibrated_actions import nominal_proposals, summarize_samples
from calibration_fixture import calibration_fixture

class Direction(unittest.TestCase):
    def test_forward_lateral_motion_no_longer_triggers_direction_diagnostic(self):
        p=nominal_proposals()['forward_25']
        for measured in ([.0297,-.0605,.0276],[.0374,-.138,-.0192],[.0238,-.1822,-.0202]):
            c.require_correctable_direction(p, measured)
            c.tune_schedule(p, measured)

    def test_rotation_drift_still_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'原地转向平移超限'):
            c.tune_schedule(nominal_proposals()['rotate_left_pi9'],[-.067,-.104,.246])

    def test_forward_repeat_set_no_longer_rejected_for_lateral_or_yaw_offset(self):
        samples=copy.deepcopy(calibration_fixture()['actions']['forward_25']['samples'])
        for sample in samples:sample['measured_m_rad']=[.24,.06,.06]
        summarize_samples('forward_25',samples,3)

    def test_nonfinite_feedback_still_rejected(self):
        with self.assertRaises(RuntimeError):
            c.require_correctable_direction(nominal_proposals()['forward_25'],[float('nan'),0.,0.])

    def test_recorded_forward_group_keeps_yaw_variation_without_rejecting(self):
        import math
        samples=copy.deepcopy(calibration_fixture()['actions']['forward_25']['samples'])
        measurements=[(.2660,.0086,2.268),(.2516,.0308,8.424),(.2279,-.0062,1.8)]
        for sample,(x,y,yaw) in zip(samples,measurements):
            sample['measured_m_rad']=[x,y,math.radians(yaw)]
        summary=summarize_samples('forward_25',samples,3)
        self.assertAlmostEqual(summary['mean_m_rad'][0],.2485)
        self.assertGreater(summary['max_yaw_deviation_rad'],math.radians(3))
        self.assertFalse(summary['yaw_repeatability_required'])
        self.assertTrue(summary['warnings'])

    def test_forward_position_repeatability_still_required(self):
        samples=copy.deepcopy(calibration_fixture()['actions']['forward_25']['samples'])
        samples[0]['measured_m_rad'][0]=.35
        with self.assertRaisesRegex(ValueError,'not repeatable'):
            summarize_samples('forward_25',samples,3)

    def test_turn_yaw_repeatability_still_required(self):
        import math
        samples=copy.deepcopy(calibration_fixture()['actions']['rotate_left_pi9']['samples'])
        samples[0]['measured_m_rad'][2]+=math.radians(8)
        with self.assertRaisesRegex(ValueError,'not repeatable'):
            summarize_samples('rotate_left_pi9',samples,3)
