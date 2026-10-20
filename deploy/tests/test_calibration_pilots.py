"""Regression for B74bgd: correct right turns converging too slowly for three pilots."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jetson_calibrate_actions as c
from calibrated_actions import nominal_proposals
from calibration_fixture import calibration_fixture

NAME='right_arc_25_pi12'
MEASURED=[
 [.27096861645797216,-.011363059586198487,-.13927200064063075],
 [.24971870082382236,-.017655099378319987,-.20049522072076797],
 [.24777792710780622,-.028195676186997462,-.23764744400978088]]


class PilotBudget(unittest.TestCase):
    def run_trials(self, rounds, fail=False):
        fixture=calibration_fixture()['actions'][NAME]['samples'][0]
        proposal=nominal_proposals()[NAME]
        proposal['schedule']=dict(vx_m_s=.12,vy_m_s=0.,yaw_rad_s=-.12848637506404037,duration_s=2.9340941266564227)
        proposals={NAME:proposal}; samples={}; status={}; pilots=[]
        def trial(args,io,capture,p,number,phase):
            result=copy.deepcopy(fixture);result['proposal']=copy.deepcopy(p)
            if phase.startswith('pilot'):
                index=len(pilots);pilots.append(copy.deepcopy(p['schedule']))
                if index<3:result['measured_m_rad']=MEASURED[index]
                if fail:result['avoidance_intervened']=True
            return result
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(output=Path(tmp),repeats=3,max_tuning_rounds=rounds)
            with patch.dict(sys.modules,{'rclpy':MagicMock()}), \
                 patch.object(c.nav,'robot_adapter',return_value=MagicMock()) as adapter, \
                 patch.object(c.nav,'Capture',return_value=MagicMock()),patch.object(c,'trial',side_effect=trial):
                error=None
                try:c.collect(args,samples,proposals,status)
                except RuntimeError as exc:error=exc
                adapter.return_value.stop.assert_called_once()
            audit=json.loads((Path(tmp)/'pilot-tuning.json').read_text()) if (Path(tmp)/'pilot-tuning.json').exists() else None
        return samples,status,pilots,proposal,error,audit

    def test_more_than_three_pilots_can_converge_without_accepting_short_turns(self):
        samples,status,pilots,proposal,error,audit=self.run_trials(8)
        self.assertIsNone(error)
        self.assertEqual(status['status'],'complete')
        self.assertEqual(len(pilots),4)
        self.assertEqual(len(samples[NAME]),3)
        self.assertTrue(all(s['measured_m_rad'] not in MEASURED for s in samples[NAME]))
        self.assertTrue(all(s['proposal']['schedule']==proposal['schedule'] for s in samples[NAME]))
        for a,b in zip(pilots,pilots[1:]):
            old=abs(a['yaw_rad_s'])*a['duration_s'];new=abs(b['yaw_rad_s'])*b['duration_s']
            self.assertLessEqual(new/old,1.200001)
            self.assertLessEqual(abs(b['yaw_rad_s']),.35)
            self.assertLessEqual(b['vx_m_s'],.120001)

    def test_exhaustion_saves_next_unverified_schedule_and_no_samples(self):
        samples,status,pilots,proposal,error,audit=self.run_trials(3)
        self.assertIsInstance(error,c.CalibrationNeedsMoreTrials)
        self.assertEqual(samples,{})
        self.assertFalse(audit['next_schedule_verified'])
        self.assertEqual(audit['previous_schedule'],pilots[-1])
        self.assertEqual(audit['next_schedule'],proposal['schedule'])
        self.assertGreater(abs(proposal['schedule']['yaw_rad_s']),abs(pilots[-1]['yaw_rad_s']))

    def test_pilot_16_48_degrees_does_not_freeze_boundary_command(self):
        proposal=nominal_proposals()['left_arc_50_pi12']
        sample=copy.deepcopy(calibration_fixture()['actions']['left_arc_50_pi12']['samples'][0])
        sample['measured_m_rad']=[.5066563386910296,.0810734068398721,.28757496178150177]
        with self.assertRaisesRegex(ValueError,'margin insufficient'):
            c.validate_pilot(proposal,sample)
        sample['measured_m_rad'][2]=.3141592653589793
        c.validate_pilot(proposal,sample)

    def test_flagged_avoidance_is_not_used_to_increase_commands(self):
        samples,status,pilots,proposal,error,audit=self.run_trials(8,fail=True)
        self.assertIsInstance(error,RuntimeError)
        self.assertNotIsInstance(error,c.CalibrationNeedsMoreTrials)
        self.assertEqual(len(pilots),1)
        self.assertEqual(proposal['schedule'],pilots[0])
        self.assertIsNone(audit)

if __name__=='__main__':unittest.main()
