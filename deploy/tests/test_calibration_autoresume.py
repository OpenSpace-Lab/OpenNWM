import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jetson_calibrate_actions as c
from calibrated_actions import build_calibration, calibrated_proposals, load_calibration
from calibration_fixture import calibration_fixture


class AutoResume(unittest.TestCase):
    def save(self,root,name,data,mode='collect'):
        directory=root/name;directory.mkdir()
        (directory/'session.json').write_text(json.dumps({'mode':mode}))
        path=directory/'calibration.json';path.write_text(json.dumps(data));return path

    def test_stationary_has_zero_measured_samples_and_does_not_block_readiness(self):
        data=calibration_fixture()
        entry=data['actions']['stationary']
        self.assertEqual(entry['samples'],[])
        self.assertEqual(entry['summary']['sample_count'],0)
        self.assertEqual(entry['source'],'builtin_zero_action')
        self.assertTrue(data['ready'])
        self.assertEqual(calibrated_proposals(data)['stationary']['effective_candidate'],[0.,0.,0.])

    def test_latest_compatible_wireless_collect_is_selected_and_invalid_group_redone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=calibration_fixture()
            first=self.save(root,'wifi-calibration-001',data)
            failed=copy.deepcopy(data);failed['ready']=False
            for sample in failed['actions']['forward_25']['samples']:
                sample['measured_m_rad']=[.36,0.,0.]
            latest=self.save(root,'wifi-calibration-002',failed)
            self.save(root,'wifi-calibration-003',data,mode='plan')
            wrong=copy.deepcopy(data);wrong['contract_sha256']='old-gait'
            self.save(root,'wifi-calibration-004',wrong)
            self.save(root,'wired-calibration-999',data)
            source,redo=c.find_auto_resume(root,3)
            self.assertEqual(source,latest)
            self.assertEqual(redo,['forward_25'])
            resumed=load_calibration(source,require_ready=False,redo_candidates=redo)
            self.assertEqual(resumed['actions']['forward_25']['samples'],[])
            self.assertEqual(len(resumed['actions']['forward_50']['samples']),3)
            self.assertEqual(json.loads(latest.read_text()),failed)
            self.assertEqual(c.find_auto_resume(root,5),(None,[]))

    def test_old_15_degree_files_are_skipped_before_sample_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old=calibration_fixture()
            for side in ('left', 'right'):
                old['actions']['rotate_'+side+'_pi12']=old['actions'].pop('rotate_'+side+'_pi9')
            old['contract_sha256']='previous-15-degree-contract'
            path=self.save(root,'wifi-calibration-002',old)
            before=path.read_bytes()
            with patch.object(c,'summarize_samples',side_effect=AssertionError('old samples must not be interpreted')):
                self.assertEqual(c.find_auto_resume(root,3),(None,[]))
            current=self.save(root,'wifi-calibration-001',calibration_fixture())
            self.assertEqual(c.find_auto_resume(root,3),(current,[]))
            self.assertEqual(path.read_bytes(),before)

    def test_unknown_sample_candidate_raises_validation_error(self):
        with self.assertRaisesRegex(ValueError,'Unknown calibration candidate'):
            c.summarize_samples('rotate_left_pi12',[],3)

    def test_completed_actions_skip_all_robot_initialization(self):
        data=calibration_fixture();status={}
        samples={k:v['samples'] for k,v in data['actions'].items()}
        proposals={k:v['proposal'] for k,v in data['actions'].items()}
        with patch.object(c.nav,'robot_adapter') as robot, patch.object(c,'trial') as trial:
            c.collect(SimpleNamespace(repeats=3),samples,proposals,status)
        robot.assert_not_called();trial.assert_not_called()
        self.assertEqual(status['status'],'complete')

    def test_session_marker_survives_missing_final_status_after_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=calibration_fixture()
            data['actions']['forward_25']['samples']=data['actions']['forward_25']['samples'][:1]
            path=self.save(root,'wifi-calibration-001',data)
            self.assertEqual(c.find_auto_resume(root,3),(path,[]))
