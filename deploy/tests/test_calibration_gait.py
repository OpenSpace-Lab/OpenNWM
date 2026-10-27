import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jetson_navigate as n
import jetson_calibrate_actions as c
from calibrated_actions import nominal_proposals, contract_hash, load_calibration
from calibration_fixture import calibration_fixture


def pose(x=0., y=0., source=0.):
    return SimpleNamespace(x=x, y=y, yaw=0., speed=0., yaw_speed=0., received=source, source=source)


class CalibrationGait(unittest.TestCase):
    def test_calibration_and_navigation_select_static_walk_with_avoidance_and_stop(self):
        p=c.initial_proposals()['forward_25']
        io=MagicMock(); io.read_pose.return_value=pose()
        with tempfile.TemporaryDirectory() as tmp, patch.object(n, 'execute_schedule', return_value={'status':'succeeded'}) as execute, patch.object(n.time,'sleep'), patch.object(n,'stopped_pose',return_value=pose(.09,.01,1.)):
            result=n.do_motion(SimpleNamespace(),io,MagicMock(),p,Path(tmp))
        io.static_walk.assert_called_once()
        io.enable_avoidance.assert_called_once()
        io.arm_navigation_motion.assert_called_once()
        io.disarm_navigation_motion.assert_called_once()
        io.stop.assert_not_called()
        io.pause_motion.assert_called_once()
        io.release_motion_control.assert_called_once()
        execute.assert_called_once()
        self.assertEqual(result['execution_mode'], 'static_walk_timed')
        self.assertEqual(result['gait_transition_policy'], 'session_static_walk_v2')
        from calibrated_actions import calibrated_proposals
        for proposal in calibrated_proposals(calibration_fixture()).values():
            self.assertEqual(proposal['gait_policy'], 'static_walk')
            self.assertEqual(proposal['gait_transition_policy'], 'session_static_walk_v2')

    def test_waits_after_gait_switch_before_velocity_command(self):
        p=c.initial_proposals()['forward_25']
        io=MagicMock(); io.read_pose.return_value=pose()
        events=[]
        io.static_walk.side_effect=lambda: events.append('static')
        def settle(guarded, **kwargs):
            if isinstance(guarded,n.GuardedIO):
                self.assertEqual(kwargs,dict(timeout=6.,stable_seconds=.8,min_wait=2.,position_tolerance=.005))
                events.append('stable')
            return pose()
        def execute(*args):
            events.append('move')
            return {'status':'succeeded'}
        with tempfile.TemporaryDirectory() as tmp, patch.object(n,'execute_schedule',side_effect=execute), patch.object(n,'stopped_pose',side_effect=settle):
            n.do_motion(SimpleNamespace(),io,MagicMock(),p,Path(tmp))
        self.assertEqual(events,['static','stable','move'])

    def test_transition_fault_never_sends_velocity(self):
        p=c.initial_proposals()['forward_25']
        io=MagicMock(); io.read_pose.return_value=pose()
        def settle(guarded, **kwargs):
            if isinstance(guarded,n.GuardedIO):
                raise RuntimeError('Robot did not become stationary')
            return pose()
        with tempfile.TemporaryDirectory() as tmp, patch.object(n,'execute_schedule') as execute, patch.object(n,'stopped_pose',side_effect=settle):
            with self.assertRaisesRegex(RuntimeError,'stationary'):
                n.do_motion(SimpleNamespace(),io,MagicMock(),p,Path(tmp))
        execute.assert_not_called()
        io.stop.assert_called()

    def test_forward_direction_diagnostic_removed_but_budget_still_active(self):
        for x,y in ((.02,-.041),(.02,.041),(-.031,0.)):
            io=MagicMock(); io.read_pose.return_value=pose(x,y,1.)
            guard=n.GuardedIO(io,MagicMock(),c.initial_proposals()['forward_25'],pose())
            guard.read_pose()
        io.read_pose.return_value=pose(1.01,0.,2.)
        with self.assertRaises(n.MotionReserveStop):
            guard.read_pose()

    def test_previous_preserved_gait_contract_cannot_resume_or_navigate(self):
        data=calibration_fixture(); previous=nominal_proposals()
        for p in previous.values():
            p['gait_policy']='preserve_current'
            p.pop('gait_transition_policy')
        data['contract_sha256']=contract_hash(previous)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'old.json';p.write_text(json.dumps(data))
            for ready in (False,True):
                with self.assertRaisesRegex(ValueError,'步态策略'):
                    load_calibration(p,require_ready=ready)
