"""Idle operator reposition versus protected calibration; no ROS or motion."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jetson_navigate as n
import jetson_calibrate_actions as c
from calibrated_actions import load_calibration, nominal_proposals
from calibration_fixture import calibration_fixture


class FakeROS:
    def __init__(self, execute):
        self.lock = threading.RLock()
        self.fault = ''
        self.active = self.has_avoidance_control = False
        self.subs = []
        self.node = MagicMock()
    def on_remote(self, msg):
        if msg.keys or max(abs(msg.lx),abs(msg.ly),abs(msg.rx),abs(msg.ry)) > .1:
            self.fault = 'remote_controller_takeover'
    def read_pose(self):
        if self.fault: raise RuntimeError(self.fault)
        return SimpleNamespace(speed=0.,yaw_speed=0.)


def modules():
    return {'motion_control.ros_adapter':SimpleNamespace(Go2ROS=FakeROS),
            'go2_capture.core':SimpleNamespace(decode_remote_packet=lambda p:p),
            'unitree_go.msg':SimpleNamespace(LowState=object),
            'rclpy.qos':SimpleNamespace(QoSProfile=lambda **kw:kw,
                                       ReliabilityPolicy=SimpleNamespace(BEST_EFFORT=1))}


class RemoteCalibration(unittest.TestCase):
    def setUp(self):
        self.now = 1.
        self.mods = patch.dict(sys.modules, modules()); self.mods.start(); self.addCleanup(self.mods.stop)
        self.clock = patch.object(n.time, 'monotonic', side_effect=lambda:self.now)
        self.clock.start(); self.addCleanup(self.clock.stop)
        self.io = n.robot_adapter(True, allow_idle_remote=True)

    def packet(self, active=False, low=True, wireless=True):
        if low: self.io.on_low(SimpleNamespace(wireless_remote={'active':active}))
        if wireless: self.io.on_remote(SimpleNamespace(keys=0,lx=.2 if active else 0.,ly=0.,rx=0.,ry=0.))

    def neutral(self):
        for t in [2.,2.2,2.4,2.6]:
            self.now = t; self.packet()

    def test_idle_reposition_both_streams_then_fresh_arm(self):
        self.packet(True)
        self.assertEqual(self.io.fault, '')
        self.assertEqual(self.io.remote_monitor.fault, '')
        with self.assertRaises(n.RemoteTelemetryNotReady): self.io.begin_calibration_trial()
        self.neutral(); self.io.begin_calibration_trial()
        self.assertTrue(self.io.calibration_trial_armed)
        self.io.end_calibration_trial()
        self.now += .1; self.packet(True)
        self.assertEqual(self.io.fault, '')
        with self.assertRaises(n.RemoteTelemetryNotReady): self.io.begin_calibration_trial()

    def test_each_stream_latches_takeover_during_trial_even_before_velocity(self):
        for low,wireless in [(True,False),(False,True)]:
            self.io = n.robot_adapter(True, allow_idle_remote=True)
            self.neutral(); self.io.begin_calibration_trial()
            self.now += .01; self.packet(True,low,wireless)
            self.assertTrue(self.io.fault)
            with self.assertRaises(RuntimeError): self.io.end_calibration_trial()
            self.now += .1; self.packet(False)
            with self.assertRaises(RuntimeError): self.io.begin_calibration_trial()

    def test_default_navigation_still_latches_idle_takeover(self):
        self.io = n.robot_adapter(True)
        self.packet(True)
        self.assertTrue(self.io.fault)
        self.assertEqual(self.io.remote_monitor.fault, 'physical_remote_takeover')

    def test_sparse_wireless_does_not_block_fresh_neutral_lowstate(self):
        self.neutral()
        self.now = 3.2; self.packet(low=True,wireless=False)
        self.now = 3.4; self.packet(low=True,wireless=False)
        self.now = 3.6; self.packet(low=True,wireless=False)
        self.now = 3.8; self.packet(low=True,wireless=False)
        self.io.begin_calibration_trial()
        self.assertTrue(self.io.calibration_trial_armed)

    def test_no_wireless_messages_still_requires_continuous_lowstate_neutrality(self):
        for t in [1.,1.2,1.4,1.6]:
            self.now=t; self.packet(wireless=False)
        self.io.begin_calibration_trial()
        self.assertTrue(self.io.calibration_trial_armed)

    def test_stale_lowstate_blocks_even_with_fresh_wireless(self):
        self.neutral()
        self.now=3.2; self.packet(low=False)
        with self.assertRaises(n.RemoteTelemetryNotReady): self.io.begin_calibration_trial()
        self.assertFalse(self.io.calibration_trial_armed)

    def test_recent_wireless_event_vetoes_lowstate_neutral_then_expires(self):
        self.neutral()
        self.now=2.7; self.packet(True,low=False)
        self.now=2.9; self.packet(wireless=False)
        with self.assertRaises(n.RemoteTelemetryNotReady): self.io.begin_calibration_trial()
        self.now=3.1; self.packet(wireless=False)
        self.now=3.3; self.packet(wireless=False)
        self.io.begin_calibration_trial()
        self.assertTrue(self.io.calibration_trial_armed)

    def test_existing_fault_is_never_cleared_by_idle_remote(self):
        self.io.fault = 'localization failure'
        self.packet(True); self.neutral()
        self.assertEqual(self.io.fault, 'localization failure')
        with self.assertRaisesRegex(RuntimeError,'localization'): self.io.begin_calibration_trial()

    def test_protected_even_when_api_control_owned_but_not_trial_armed(self):
        self.io.has_avoidance_control = True
        self.packet(True)
        self.assertTrue(self.io.fault)

    def test_confirm_before_release_requires_new_confirmation_and_no_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            io = MagicMock(); capture=MagicMock()
            io.begin_calibration_trial.side_effect=[n.RemoteTelemetryNotReady('release sticks'),None]
            with patch('builtins.input',side_effect=['MOVE 001','MOVE 001']) as prompt, \
                 patch.object(c,'measure_trial',return_value={'status':'ok'}) as measure:
                result=c.trial(SimpleNamespace(output=Path(tmp)),io,capture,nominal_proposals()['forward_25'],1,'repeat-1')
            self.assertEqual(prompt.call_count,2); measure.assert_called_once()
            io.end_calibration_trial.assert_called_once()
            self.assertEqual(result,{'status':'ok'})

    def test_final_measurement_exception_disarms_but_does_not_swallow(self):
        with tempfile.TemporaryDirectory() as tmp:
            io=MagicMock()
            with patch('builtins.input',return_value='MOVE 001'), \
                 patch.object(c,'measure_trial',side_effect=RuntimeError('camera failure')):
                with self.assertRaisesRegex(RuntimeError,'camera failure'):
                    c.trial(SimpleNamespace(output=Path(tmp)),io,MagicMock(),nominal_proposals()['stationary'],1,'repeat-1')
            io.end_calibration_trial.assert_called_once()


class ResumeCalibration(unittest.TestCase):
    def partial(self):
        data=calibration_fixture(); data['ready']=False
        for name,entry in data['actions'].items():
            if name=='stationary': continue
            entry['samples']=entry['samples'][:1] if name=='forward_25' else []
        return data

    def test_partial_resume_validates_and_preserves_only_existing_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'calibration.json'; data=self.partial(); path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): load_calibration(path)
            resumed=load_calibration(path,require_ready=False)
            self.assertEqual(sum(len(v['samples']) for v in resumed['actions'].values()),4)
            self.assertFalse(resumed['ready'])
            data['actions']['forward_25']['samples'][0]['status']='failed'; path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): load_calibration(path,require_ready=False)

    def test_collect_resume_keeps_frozen_schedule_and_skips_completed_actions(self):
        fixture=calibration_fixture(); data=self.partial()
        proposals={k:v['proposal'] for k,v in data['actions'].items()}
        samples={k:v['samples'] for k,v in data['actions'].items()}
        original=copy.deepcopy(samples['forward_25'][0]); trial_names=[]
        def trial(args,io,capture,proposal,number,phase):
            trial_names.append((proposal['candidate'],phase))
            sample=copy.deepcopy(fixture['actions'][proposal['candidate']]['samples'][0])
            sample['proposal']=copy.deepcopy(proposal)
            return sample
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(output=Path(tmp),repeats=3,max_tuning_rounds=3)
            with patch.dict(sys.modules,{'rclpy':MagicMock()}), \
                 patch.object(n,'robot_adapter',return_value=MagicMock()) as adapter, \
                 patch.object(n,'Capture',return_value=MagicMock()),patch.object(c,'trial',side_effect=trial):
                c.collect(args,samples,proposals,{})
                adapter.assert_called_once_with(execute=True,allow_idle_remote=True)
            self.assertEqual(samples['forward_25'][0],original)
            self.assertTrue(all(name!='stationary' for name,phase in trial_names))
            self.assertEqual([phase for name,phase in trial_names if name=='forward_25'],['repeat-2','repeat-3'])
            self.assertEqual(len(trial_names),54)  # No repeated stationary, pilot or accepted forward sample.

if __name__=='__main__': unittest.main()
