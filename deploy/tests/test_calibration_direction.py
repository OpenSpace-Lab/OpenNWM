"""Recorded forward/right drift must stop calibration instead of extending commands."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jetson_calibrate_actions as c
from calibrated_actions import nominal_proposals, summarize_samples
from calibration_fixture import calibration_fixture

class Direction(unittest.TestCase):
    def test_recorded_wrong_axis_not_tuned(self):
        p=nominal_proposals()['forward_25']; before=copy.deepcopy(p)
        for measured in ([.0297,-.0605,.0276],[.0374,-.138,-.0192],[.0238,-.1822,-.0202]):
            with self.assertRaisesRegex(RuntimeError,'直行侧移超限'):
                c.tune_schedule(p,measured)
        self.assertEqual(p,before)

    def test_rotation_drift_not_tuned(self):
        with self.assertRaisesRegex(RuntimeError,'原地转向平移超限'):
            c.tune_schedule(nominal_proposals()['rotate_left_pi12'],[-.067,-.104,.246])

    def test_first_bad_pilot_stops_and_keeps_original_command(self):
        p=nominal_proposals()['forward_25'];before=copy.deepcopy(p)
        sample=copy.deepcopy(calibration_fixture()['actions']['forward_25']['samples'][0])
        sample['measured_m_rad']=[.0297,-.0605,.0276]
        with tempfile.TemporaryDirectory() as tmp:
            io=MagicMock(); trials=MagicMock(return_value=sample)
            with patch.dict(sys.modules,{'rclpy':MagicMock()}),patch.object(c.nav,'robot_adapter',return_value=io),patch.object(c.nav,'Capture',return_value=MagicMock()),patch.object(c,'trial',trials):
                with self.assertRaisesRegex(RuntimeError,'直行侧移超限'):
                    c.collect(SimpleNamespace(output=Path(tmp),repeats=3,max_tuning_rounds=8),{}, {'forward_25':p},{})
            self.assertEqual(trials.call_count,1)
            io.stop.assert_called_once()
            self.assertFalse((Path(tmp)/'pilot-tuning.json').exists())
            self.assertEqual(p,before)

    def test_repeatable_sideways_cannot_be_accepted_as_forward(self):
        samples=copy.deepcopy(calibration_fixture()['actions']['forward_25']['samples'])
        for s in samples:s['measured_m_rad']=[.05,-.24,0.]
        with self.assertRaisesRegex(ValueError,'lateral drift'):
            summarize_samples('forward_25',samples,3)

if __name__=='__main__':unittest.main()
