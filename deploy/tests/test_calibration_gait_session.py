import sys
import threading
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gait_session import SessionGaitControl
import jetson_navigate as n
import jetson_calibrate_actions as c


class Base:
    def __init__(self):
        self.lock = threading.RLock()
        self.execute = True
        self.fault = ''
        self.active = self.has_avoidance_control = False
        self.sport = (0., SimpleNamespace(gait_type=1))
        self.events = []
        self.heartbeat = 0

    def static_walk(self): self.events.append('static')
    def arm_navigation_motion(self): pass
    def disarm_navigation_motion(self): self.events.append('disarm')
    def enable_avoidance(self): self.has_avoidance_control = True

    def move(self, x, y, yaw):
        if self.fault: raise RuntimeError(self.fault)
        if not self.has_avoidance_control: raise RuntimeError('No control')
        self.active = True
        self.heartbeat += 1
        self.events.append(('move', x, y, yaw))

    def read_pose(self):
        if self.fault: raise RuntimeError(self.fault)
        return SimpleNamespace(x=0., y=0., yaw=0., speed=0., yaw_speed=0., source=0., received=0.)

    def publish_avoidance(self, api, payload, **kwargs):
        self.events.append(('api', api, payload))

    def stop(self):
        self.events.append('emergency-stop')
        self.active = self.has_avoidance_control = False


class Adapter(SessionGaitControl, Base): pass


class SessionTests(unittest.TestCase):
    def test_select_once_until_remote_or_gait_change(self):
        io = Adapter()
        self.assertTrue(io.static_walk())
        io.confirm_static_gait()
        self.assertFalse(io.static_walk())
        io.invalidate_gait()
        self.assertTrue(io.static_walk())
        io.confirm_static_gait()
        io.sport[1].gait_type = 2
        self.assertTrue(io.static_walk())
        self.assertEqual(io.events.count('static'), 3)

    def test_zero_refresh_keeps_watchdog_active_then_release_preserves_gait(self):
        io = Adapter(); io.static_walk(); io.confirm_static_gait(); io.enable_avoidance()
        io.pause_motion()
        self.assertTrue(io.active)
        self.assertTrue(io.has_avoidance_control)
        heartbeat = io.heartbeat
        io.read_pose()
        self.assertGreater(io.heartbeat, heartbeat)
        io.release_motion_control()
        self.assertFalse(io.active)
        self.assertFalse(io.has_avoidance_control)
        self.assertFalse(io.static_walk())
        self.assertNotIn('emergency-stop', io.events)
        self.assertIn(('api', 1004, {'is_remote_commands_from_api': False}), io.events)

    def test_fault_stop_invalidates_gait_and_cannot_resume(self):
        io=Adapter(); io.static_walk(); io.confirm_static_gait(); io.enable_avoidance(); io.pause_motion()
        io.fault='physical_remote_takeover'
        with self.assertRaises(RuntimeError):io.read_pose()
        io.stop()  # Original watchdog/cleanup entrypoint.
        self.assertFalse(io._static_gait_ready)
        self.assertFalse(io._settling_at_zero)
        self.assertFalse(io.active)
        with self.assertRaises(RuntimeError):io.static_walk()
        with self.assertRaises(RuntimeError):io.release_motion_control()

    def test_release_without_normal_zero_stop_is_rejected(self):
        io=Adapter(); io.enable_avoidance()
        with self.assertRaises(RuntimeError):io.release_motion_control()
        self.assertTrue(io.has_avoidance_control)

    def test_two_actions_switch_once_and_release_only_after_settling(self):
        io=Adapter(); p=c.initial_proposals()['forward_25']
        def settle(reader, **kwargs):
            if isinstance(reader,n.SettlingIO):
                self.assertTrue(io.active)
                self.assertTrue(io.has_avoidance_control)
                io.events.append('settled')
            return reader.read_pose()
        with tempfile.TemporaryDirectory() as tmp, patch.object(n,'stopped_pose',side_effect=settle), patch.object(n,'execute_schedule',return_value={'status':'succeeded'}):
            for number in range(2):
                path=Path(tmp)/str(number); path.mkdir()
                result=n.do_motion(SimpleNamespace(),io,MagicMock(),p,path)
                self.assertEqual(result['status'],'succeeded')
                self.assertEqual(result['gait_switch_sent'],number==0)
                self.assertFalse(io.has_avoidance_control)
        self.assertEqual(io.events.count('static'),1)
        self.assertNotIn('emergency-stop',io.events)
        for i,event in enumerate(io.events):
            if isinstance(event,tuple) and event[0]=='api':
                self.assertIn('settled',io.events[:i])

    def test_normal_zero_publish_failure_uses_emergency_stop(self):
        guarded=MagicMock(); now=[0.]
        guarded.pause_motion.side_effect=RuntimeError('publish failed')
        with self.assertRaisesRegex(RuntimeError,'publish failed'):
            n.execute_schedule(guarded,dict(vx_m_s=.3,vy_m_s=0.,yaw_rad_s=0.,duration_s=.1),
                               lambda:now[0],lambda dt:now.__setitem__(0,now[0]+dt))
        guarded.stop.assert_called_once()
