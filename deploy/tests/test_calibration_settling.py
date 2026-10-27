import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import jetson_navigate as n


class Settling(unittest.TestCase):
    def test_zero_velocity_during_retraction_is_not_enough(self):
        now=[0.]; io=MagicMock()
        def pose():
            return SimpleNamespace(x=max(0., .06-.03*now[0]),y=0.,yaw=0.,speed=0.,yaw_speed=0.)
        io.read_pose.side_effect=pose
        with patch.object(n.time,'monotonic',side_effect=lambda:now[0]), patch.object(n.time,'sleep',side_effect=lambda dt:now.__setitem__(0,now[0]+dt)):
            result=n.stopped_pose(io,timeout=6.,stable_seconds=.8,min_wait=2.,position_tolerance=.005)
        self.assertGreater(now[0],2.5)
        self.assertAlmostEqual(result.x,0.)

    def test_stable_robot_still_has_minimum_post_stop_wait(self):
        now=[0.]; io=MagicMock()
        io.read_pose.return_value=SimpleNamespace(x=0.,y=0.,yaw=0.,speed=0.,yaw_speed=0.)
        with patch.object(n.time,'monotonic',side_effect=lambda:now[0]), patch.object(n.time,'sleep',side_effect=lambda dt:now.__setitem__(0,now[0]+dt)):
            n.stopped_pose(io,timeout=6.,stable_seconds=.8,min_wait=2.,position_tolerance=.005)
        self.assertGreaterEqual(now[0],2.)

    def test_never_accepts_continually_drifting_pose(self):
        now=[0.];io=MagicMock()
        io.read_pose.side_effect=lambda:SimpleNamespace(x=.03*now[0],y=0.,yaw=0.,speed=0.,yaw_speed=0.)
        with patch.object(n.time,'monotonic',side_effect=lambda:now[0]), patch.object(n.time,'sleep',side_effect=lambda dt:now.__setitem__(0,now[0]+dt)):
            with self.assertRaisesRegex(RuntimeError,'stationary'):
                n.stopped_pose(io,timeout=6.,stable_seconds=.8,min_wait=2.,position_tolerance=.005)

    def test_fast_forward_reserves_twelve_cm_before_hard_limit(self):
        from jetson_calibrate_actions import initial_proposals
        proposal=initial_proposals()['forward_25']
        proposal['hard_distance_m']=.30
        pose=lambda x,t:SimpleNamespace(x=x,y=0.,yaw=0.,source=t)
        budget=n.MotionBudget(pose(0.,0.),proposal)
        budget.check(pose(.179,1.))
        with self.assertRaises(n.MotionReserveStop):
            budget.check(pose(.181,2.))
        budget.check(pose(.275,3.),stopped=True)
        with self.assertRaises(RuntimeError):
            budget.check(pose(.301,4.),stopped=True)
