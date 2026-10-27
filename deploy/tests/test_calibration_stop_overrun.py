import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import calibrated_actions as a
import jetson_calibrate_actions as c
import jetson_navigate as n
from calibration_fixture import calibration_fixture


def pose(x, source):
    return SimpleNamespace(x=x,y=0.,yaw=0.,source=source)


class StopOverrun(unittest.TestCase):
    def test_settling_enforces_reserve_immediately_and_hard_limit_after_stop(self):
        io=MagicMock();capture=MagicMock()
        io.read_pose.side_effect=[pose(.87,1.),pose(.89,2.),pose(.95,3.),pose(1.01,4.)]
        budget=n.MotionBudget(pose(0.,0.),c.initial_proposals()['forward_50'])
        record=dict(status='succeeded',controller=dict(completion_basis='elapsed_command_time'))
        guard=n.SettlingIO(io,capture,budget,record)
        guard.read_pose();io.stop.assert_not_called()
        guard.read_pose();io.stop.assert_called_once()
        self.assertEqual(record['status'],'stopped_at_reserve')
        self.assertEqual(record['controller']['completion_basis'],'motion_reserve_during_settling')
        guard.read_pose();io.stop.assert_called_once()
        with self.assertRaisesRegex(RuntimeError,'hard limit'):
            guard.read_pose()
        self.assertEqual(record['status'],'failed')
        self.assertIn('1.0100 >= 1.0000',record['budget_error'])

    def test_single_feedback_jump_past_both_limits_is_failed_immediately(self):
        io=MagicMock();io.read_pose.return_value=pose(1.02,1.)
        record=dict(status='succeeded')
        guard=n.SettlingIO(io,MagicMock(),n.MotionBudget(pose(0.,0.),c.initial_proposals()['forward_50']),record)
        with self.assertRaises(RuntimeError):guard.read_pose()
        io.stop.assert_called_once()
        self.assertEqual(record['status'],'failed')

    def test_forward50_seed_migrates_once_and_does_not_reset_accepted_commands(self):
        proposals=c.initial_proposals();samples={}
        self.assertEqual(proposals['forward_50']['schedule']['duration_s'],2.)
        proposals['forward_50']['schedule']['duration_s']=2.8333333333333335
        original=copy.deepcopy(proposals['forward_25'])
        change=c.upgrade_forward50_seed(proposals,samples,'fixed_translation_030_v3')
        self.assertEqual(change['next_schedule']['duration_s'],2.)
        self.assertEqual(proposals['forward_25'],original)
        proposals['forward_50']['schedule']['duration_s']=2.2
        self.assertIsNone(c.upgrade_forward50_seed(proposals,samples,c.TUNING_POLICY))
        samples['forward_50']=[{'status':'succeeded'}]
        self.assertIsNone(c.upgrade_forward50_seed(proposals,samples,None))

    def test_auto_resume_migrates_70cm_limits_and_preserves_successful_short_action(self):
        data=calibration_fixture();old=a.nominal_proposals()
        for name,p in old.items():p['hard_distance_m']=.7
        data['contract_sha256']=a.contract_hash(old)
        for name,entry in data['actions'].items():
            entry['proposal']['hard_distance_m']=.7
            if name!='forward_25':entry['samples']=[]
            for sample in entry['samples']:
                sample['proposal']['hard_distance_m']=.7
                sample['controller']={'completion_basis':'elapsed_command_time'}
        data['ready']=False
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'wifi-calibration-001';directory.mkdir()
            path=directory/'calibration.json';path.write_text(json.dumps(data));original=path.read_bytes()
            (directory/'session.json').write_text(json.dumps(dict(mode='collect')))
            self.assertEqual(c.find_auto_resume(Path(tmp),3),(path,[]))
            checked=a.load_calibration(path,require_ready=False)
            self.assertEqual(checked['actions']['forward_25']['samples'],data['actions']['forward_25']['samples'])
            self.assertEqual(checked['limit_migration']['from_m_rad'],[.7,math.pi/3])
            self.assertEqual(checked['limit_migration']['to_m_rad'],[1.,math.pi/3])
            self.assertTrue(all(e['proposal']['hard_distance_m']==1. for e in checked['actions'].values()))
            self.assertEqual(path.read_bytes(),original)
