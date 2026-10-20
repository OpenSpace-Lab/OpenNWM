import copy,json,math,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from calibrated_actions import nominal_proposals,contract_hash,load_calibration,calibrated_proposals
from calibration_fixture import calibration_fixture
from jetson_navigate import MotionBudget,MotionReserveStop

class Limits(unittest.TestCase):
    def legacy(self):
        data=calibration_fixture();old=nominal_proposals()
        for p in old.values():p.update(hard_distance_m=.5,hard_turn_rad=math.pi/4)
        data['contract_sha256']=contract_hash(old)
        for name,entry in data['actions'].items():
            entry['proposal'].update(hard_distance_m=.5,hard_turn_rad=math.pi/4)
            for sample in entry['samples']:
                sample['proposal'].update(hard_distance_m=.5,hard_turn_rad=math.pi/4)
                sample['controller']={'completion_basis':'elapsed_command_time'}
        return data
    def load(self,data,**kw):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'calibration.json';p.write_text(json.dumps(data));return load_calibration(p,**kw)
    def test_all_navigation_proposals_use_new_hard_limits(self):
        ready=self.load(calibration_fixture())
        for proposal in calibrated_proposals(ready).values():
            self.assertEqual(proposal['hard_distance_m'],.7)
            self.assertEqual(proposal['hard_turn_rad'],math.pi/3)
        self.assertEqual({p['nominal']['distance_m'] for p in nominal_proposals().values()},{0.,.25,.5})
    def test_old_timed_samples_migrate_without_changing_commands_or_measurements(self):
        data=self.legacy();original=copy.deepcopy(data)
        with self.assertRaises(ValueError):self.load(data)
        checked=self.load(data,require_ready=False)
        self.assertTrue(checked['ready'])
        self.assertEqual(checked['limit_migration']['recollect_candidates'],[])
        for name,entry in checked['actions'].items():
            self.assertEqual(entry['samples'],original['actions'][name]['samples'])
            self.assertEqual(entry['proposal']['schedule'],original['actions'][name]['proposal']['schedule'])
        self.load(checked)  # Saved migrated file is usable by navigation.
    def test_old_reserve_stopped_candidate_requires_recollection(self):
        data=self.legacy();data['actions']['forward_50']['samples'][0]['status']='stopped_at_reserve'
        checked=self.load(data,require_ready=False)
        self.assertEqual(checked['actions']['forward_50']['samples'],[])
        self.assertEqual(checked['limit_migration']['recollect_candidates'],['forward_50'])
        self.assertFalse(checked['ready'])
    def test_failed_old_samples_not_promoted(self):
        data=self.legacy();data['actions']['forward_50']['samples'][0]['status']='failed'
        with self.assertRaises(ValueError):self.load(data,require_ready=False)
    def test_guard_still_enforces_new_limit_and_retains_reserve(self):
        pose=lambda x,yaw,source:SimpleNamespace(x=x,y=0.,yaw=yaw,source=source)
        proposal=nominal_proposals()['left_arc_25_pi4']
        budget=MotionBudget(pose(0,0,0),proposal)
        budget.check(pose(.6,math.radians(46.69),1),stopped=True)
        with self.assertRaises(RuntimeError):budget.check(pose(.7,math.radians(46.69),2),stopped=True)
        budget=MotionBudget(pose(0,0,0),proposal)
        with self.assertRaises(MotionReserveStop):budget.check(pose(0,math.radians(56.1),1))
        budget=MotionBudget(pose(0,0,0),proposal)
        with self.assertRaises(RuntimeError):budget.check(pose(0,math.pi/3,1),stopped=True)
if __name__=='__main__':unittest.main()
