import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navigation_constraints import TurnReversalGuard
from calibrated_actions import nominal_proposals


class TurnConstraints(unittest.TestCase):
    def setUp(self):
        self.guard=TurnReversalGuard()
        self.proposals=nominal_proposals()

    def observe(self,name,pose,status='succeeded'):
        return self.guard.observe(self.proposals[name],{'status':status,'measured_m_rad':pose},1)

    def test_only_opposite_in_place_rotations_blocked_symmetrically(self):
        for side,sign in (('left',1),('right',-1)):
            for angle,label in ((math.pi/9,'pi9'),(math.pi/4,'pi4')):
                self.guard=TurnReversalGuard()
                self.observe('rotate_'+side+'_'+label,[0,0,sign*angle])
                eligible,report=self.guard.eligible(self.proposals)
                self.assertEqual(len(eligible),5)
                self.assertEqual(len(report['blocked']),2)
                self.assertTrue(all(p['nominal']['distance_m']>0 or p['nominal']['yaw_rad']*sign>=0 for p in eligible.values()))
                self.assertFalse(any('_arc_' in name for name in eligible))
                self.assertIn('stationary',eligible)
                self.assertIn('forward_25',eligible)

    def test_stops_never_clear_turn_memory(self):
        self.observe('rotate_left_pi9',[0,0,math.pi/12])
        before=self.guard.snapshot()
        for i in range(10):
            self.observe('stationary',[0,0,0])
            self.guard.eligible(self.proposals)
        self.assertEqual(self.guard.snapshot(),before)

    def test_measured_straight_progress_releases_not_nominal_distance(self):
        self.observe('rotate_left_pi9',[0,0,math.pi/12])
        self.observe('forward_50',[0,0,0])
        self.observe('forward_25',[.09,0,0])
        self.observe('stationary',[0,0,0])
        self.assertNotIn('rotate_right_pi9',self.guard.eligible(self.proposals)[0])
        event=self.observe('forward_25',[.11,0,0])
        self.assertEqual(event['event'],'released_by_translation')
        self.assertEqual(len(self.guard.eligible(self.proposals)[0]),7)

    def test_forward_can_release_rotation_guard_without_arc_candidates(self):
        self.observe('rotate_left_pi9',[0,0,math.pi/12])
        eligible,_=self.guard.eligible(self.proposals)
        self.assertFalse(any('_arc_' in name for name in eligible))
        self.assertNotIn('rotate_right_pi9',eligible)
        event=self.observe('forward_25',[.23,0.,0.])
        self.assertEqual(event['event'],'released_by_translation')
        self.assertEqual(len(self.guard.eligible(self.proposals)[0]),7)

    def test_backward_lateral_failed_or_invalid_feedback_cannot_release(self):
        self.observe('rotate_left_pi9',[0,0,math.pi/12])
        before=self.guard.snapshot()
        for pose,status in (([-.3,0,0],'succeeded'),([0,.3,0],'succeeded'),([.3,0,0],'failed'),([float('nan'),0,0],'succeeded')):
            self.observe('forward_50',pose,status)
        self.assertEqual(self.guard.snapshot(),before)

    def test_actual_direction_and_reserve_stop_are_used(self):
        self.observe('rotate_left_pi9',[0,0,-math.pi/12],'stopped_at_reserve')
        eligible,_=self.guard.eligible(self.proposals)
        self.assertNotIn('rotate_left_pi9',eligible)
        self.assertIn('rotate_right_pi9',eligible)

    def test_tiny_yaw_drift_does_not_create_guard(self):
        self.observe('forward_25',[.21,0,.02])
        self.assertIsNone(self.guard.last_turn)

if __name__=='__main__':unittest.main()
