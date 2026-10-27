import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import calibrated_actions as a
import jetson_calibrate_actions as c
from calibration_fixture import calibration_fixture


class ForwardTurnSet(unittest.TestCase):
    def legacy(self):
        data=calibration_fixture()
        previous=a._legacy_arc_proposals()
        for name,p in previous.items():
            if name not in data['actions']:
                # Removed arcs may be unfinished or failed; they are not reused.
                data['actions'][name]=dict(proposal=p,samples=[dict(status='failed')])
        data.update(schema='opennwm.fixed15.v1',contract_sha256=a.contract_hash(previous),ready=False)
        return data

    def test_old_angles_rejected_for_navigation_and_resume(self):
        data=self.legacy()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'calibration.json';path.write_text(json.dumps(data));before=path.read_bytes()
            for ready in (True, False):
                with self.assertRaisesRegex(ValueError,'20°/45°'):
                    a.load_calibration(path, require_ready=ready)
            self.assertEqual(path.read_bytes(), before)

    def test_old_control_policy_cannot_be_promoted_by_removing_arcs(self):
        data=self.legacy();previous=a._legacy_arc_proposals()
        for p in previous.values():p['gait_policy']='preserve_current'
        data['contract_sha256']=a.contract_hash(previous)
        with self.assertRaisesRegex(ValueError,'不兼容'):
            a.restrict_legacy_actions(data)

    def test_new_file_cannot_reintroduce_arcs_or_hide_arc_commands_as_forward(self):
        for kind in ('extra','forward_yaw','rotation_vx'):
            data=calibration_fixture()
            if kind=='extra':data['actions']['left_arc_25_pi12']=copy.deepcopy(data['actions']['forward_25'])
            if kind=='forward_yaw':data['actions']['forward_25']['proposal']['schedule']['yaw_rad_s']=.1
            if kind=='rotation_vx':data['actions']['rotate_left_pi9']['proposal']['schedule']['vx_m_s']=.1
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'calibration.json';path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):a.load_calibration(path)
