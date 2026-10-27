import copy,json,math,sys,tempfile
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from calibrated_actions import load_calibration
from calibration_fixture import calibration_fixture
from jetson_calibrate_actions import retune_recollection

class Recollection(unittest.TestCase):
    def test_explicit_redo_preserves_other_sets_and_source_file(self):
        data=calibration_fixture();name='rotate_left_pi9';data['ready']=False
        data['actions'][name]['samples'][0]['measured_m_rad'][2]+=.15
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'calibration.json';p.write_text(json.dumps(data));original=p.read_bytes()
            with self.assertRaises(ValueError):load_calibration(p,require_ready=False)
            checked=load_calibration(p,require_ready=False,redo_candidates=[name])
            self.assertEqual(checked['actions'][name]['samples'],[])
            self.assertEqual(sum(len(v['samples']) for v in checked['actions'].values()),15)
            self.assertFalse(checked['ready'])
            self.assertEqual(checked['recollection']['discarded_from_new_run_only'],{name:3})
            self.assertEqual(original,p.read_bytes())
            with self.assertRaises(ValueError):load_calibration(p,redo_candidates=[name])
            with self.assertRaises(ValueError):load_calibration(p,require_ready=False,redo_candidates=['typo'])
            data['actions']['forward_25']['samples'][0]['status']='failed';p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):load_calibration(p,require_ready=False,redo_candidates=[name])

    def test_stable_repeat_mean_underturn_adjusts_next_command_only(self):
        entry=calibration_fixture()['actions']['rotate_left_pi4']
        proposal=entry['proposal']; samples=entry['samples']
        for sample in samples:
            sample['controller']={'completion_basis':'elapsed_command_time'}
            sample['measured_m_rad']=[0.,0.,.6497116635]
        original_samples=copy.deepcopy(samples); original_schedule=proposal['schedule'].copy()
        result=retune_recollection(proposal,samples)
        self.assertTrue(result['applied'])
        self.assertFalse(result['next_schedule_verified'])
        old_turn=original_schedule['yaw_rad_s']*original_schedule['duration_s']
        new_turn=proposal['schedule']['yaw_rad_s']*proposal['schedule']['duration_s']
        self.assertAlmostEqual(new_turn/old_turn,1.2)
        self.assertEqual(samples,original_samples)
        self.assertLessEqual(abs(proposal['schedule']['yaw_rad_s']),.45)

    def test_unstable_or_intervened_repeats_do_not_adjust_commands(self):
        for kind in ['unstable','avoidance','reserve','different_schedule']:
            entry=calibration_fixture()['actions']['rotate_left_pi4']
            proposal=entry['proposal']; samples=entry['samples']
            for sample in samples:
                sample['controller']={'completion_basis':'elapsed_command_time'}
                sample['measured_m_rad']=[0.,0.,.6497]
            if kind=='unstable': samples[0]['measured_m_rad'][2]=.4
            if kind=='avoidance': samples[0]['avoidance_intervened']=True
            if kind=='reserve': samples[0]['status']='stopped_at_reserve'
            if kind=='different_schedule': samples[0]['proposal']['schedule']['duration_s']+=1
            before=proposal['schedule'].copy()
            result=retune_recollection(proposal,samples)
            self.assertFalse(result['applied'],kind)
            self.assertEqual(proposal['schedule'],before)

if __name__=='__main__':unittest.main()
