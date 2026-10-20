"""Offline navigation guards. No DDS, camera, GPU or real robot commands."""
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import queue
from collections import deque
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jetson_navigate as n


def row(x=0., y=0., yaw=0., t=0.):
    return {'x': x, 'y': y, 'yaw': yaw, 'timeline_ns': int(t*1e9)}


def pose(x=0., y=0., yaw=0., source=0.):
    return SimpleNamespace(x=x, y=y, yaw=yaw, source=source, received=source, speed=0., yaw_speed=0.)


class Targets(unittest.TestCase):
    def test_translation_only_and_rotation_only(self):
        rows = [row(), row(x=.25,t=1), row(x=.25,yaw=math.radians(15),t=2)]
        self.assertEqual([i for i,_ in n.select_targets(rows,'either')],[0,1,2])
        self.assertEqual([i for i,_ in n.select_targets(rows,'strict')],[0,2])

    def test_both_upper_bounds_are_required(self):
        for r in (row(x=.51,yaw=math.radians(15)), row(x=.25,yaw=math.radians(31))):
            with self.assertRaises(ValueError):
                n.select_targets([row(),r],'either')

    def test_no_unqualified_tail(self):
        self.assertEqual(n.select_targets([row(),row(x=.24,yaw=math.radians(14))],'either'),[(0,'start')])

    def test_wrapped_yaw(self):
        rows=[row(yaw=math.radians(175)),row(yaw=math.radians(-169))]
        self.assertEqual(len(n.select_targets(rows,'either')),2)

    def test_source_pose_displacement_rejects(self):
        with self.assertRaises(RuntimeError): n.assert_source_unchanged(pose(),pose(x=.031))
        with self.assertRaises(RuntimeError): n.assert_source_unchanged(pose(),pose(yaw=math.radians(6)))
        n.assert_source_unchanged(pose(),pose(x=.01,yaw=.01))

    def test_source_diagnostics_and_wrapped_yaw(self):
        with self.assertRaisesRegex(n.StaleProposal, r'0.0400 m / 6.00 deg'):
            n.assert_source_unchanged(pose(), pose(x=.04, yaw=math.radians(6)))
        n.assert_source_unchanged(pose(yaw=math.radians(179)), pose(yaw=math.radians(-179)))
        with self.assertRaisesRegex(RuntimeError, 'Invalid pose'):
            n.assert_source_unchanged(pose(), pose(x=float('nan')))


class History(unittest.TestCase):
    def capture_stub(self):
        capture=n.Capture.__new__(n.Capture)
        capture.lock=threading.Lock(); capture.closed=threading.Event()
        capture.frames=deque(maxlen=40); capture.queue=queue.Queue(maxsize=30)
        capture.fault=''; capture.timestamp_drops=0; capture.consecutive_timestamp_drops=0
        return capture

    def frame(self, stamp):
        return SimpleNamespace(timeline_ns=stamp,sequence=stamp,rtp_timestamp=stamp,rtp_ssrc=1)

    def test_duplicate_and_brief_reordering_drop_without_poisoning_new_frames(self):
        c=self.capture_stub()
        for stamp in (1_000_000_000,1_000_000_000,990_000_000,1_067_000_000):
            c.on_frame(self.frame(stamp))
        self.assertEqual(c.fault,'')
        self.assertEqual(c.timestamp_drops,2)
        self.assertEqual(c.consecutive_timestamp_drops,0)
        self.assertEqual(c.queue.qsize(),2)
        self.assertEqual([f.timeline_ns for _,f in c.frames],[1_000_000_000,1_067_000_000])

    def test_large_or_repeated_timestamp_regression_remains_fatal(self):
        for stamps in ((2_000_000_000,1_000_000_000), (2_000_000_000,)*6):
            c=self.capture_stub()
            for stamp in stamps: c.on_frame(self.frame(stamp))
            self.assertTrue(c.fault)
            c.on_frame(self.frame(2_100_000_000))
            self.assertTrue(c.fault)

    def test_action_history_padding_and_latest_four_actions_ignore_time_gaps(self):
        from PIL import Image
        h=n.ActionHistory()
        for i,stamp in enumerate((0.,100.,100.01,700.,1000.)):
            im=Image.new('RGB',(16,16),(i*40,0,0))
            h.append(im,{'sequence':i+1,'timestamp_monotonic':stamp},action_step=i*3)
            images,meta=h.context()
            expected=[0]*(max(0,3-i))+list(range(max(0,i-3),i+1))
            self.assertEqual(meta['action_steps'],[v*3 for v in expected])
            self.assertEqual([im.getpixel((0,0))[0] for im in images],[v*40 for v in expected])
            self.assertEqual(meta['unique_real_frames'],min(i+1,4))
        self.assertEqual(meta['action_steps'],[3,6,9,12])

    def test_history_requires_new_frame_and_does_not_mutate_on_read(self):
        from PIL import Image
        h=n.ActionHistory(); image=Image.new('RGB',(8,8),'red')
        h.append(image,{'sequence':1,'timestamp_monotonic':1.},0)
        before=h.context()[1]
        image.paste('blue',(0,0,8,8))
        self.assertEqual(h.context()[0][-1].getpixel((0,0)),(255,0,0))
        self.assertEqual(h.context()[1],before)
        with self.assertRaises(RuntimeError):h.append(image,{'sequence':1,'timestamp_monotonic':2.},1)

    def test_model_geometry_is_square_and_keeps_center_content(self):
        from PIL import Image
        image=Image.new('RGB',(800,300),'red')
        image.paste('green',(200,0,600,300))
        result=n.model_frame(image)
        self.assertEqual(result.size,(224,224))
        self.assertEqual(result.getpixel((0,0)),(0,128,0))
        self.assertEqual(result.getpixel((223,223)),(0,128,0))

    def test_post_action_capture_waits_for_a_frame_after_completion(self):
        import numpy as np
        c=self.capture_stub(); c.check=MagicMock()
        frame=SimpleNamespace(image=np.zeros((8,8,3),dtype=np.uint8),sequence=1)
        c.frames.append((1.,frame)); now=[1.]
        def sleep(dt):
            now[0]+=dt
            if now[0]>=1.1:
                c.frames.append((now[0],SimpleNamespace(image=frame.image,sequence=2)))
        with patch.object(n.time,'monotonic',side_effect=lambda:now[0]),patch.object(n.time,'sleep',side_effect=sleep):
            _,metadata=c.latest(after=1.1)
        self.assertEqual(metadata['sequence'],2)
        self.assertGreaterEqual(metadata['timestamp_monotonic'],1.1)


class Mapping(unittest.TestCase):
    def test_bootstrap_nine_nonstationary_candidates(self):
        paths=n.allowed_candidates(n.OnlineScale())
        self.assertEqual(len(paths),9)
        self.assertNotIn('stationary',paths)
        for path in paths.values():
            self.assertLessEqual(math.hypot(*path[-1][:2]),.50001)
            self.assertLessEqual(abs(path[-1][2]),math.radians(22.5)+1e-9)

    def test_all_candidates_limited_even_for_large_gains(self):
        s=n.OnlineScale(); s.xy=1.; s.yaw=1.5
        for full in (False,True):
            for name,path in n.candidates().items():
                p=n.make_proposal(name,path,s,full)
                self.assertLessEqual(p['mapped_path_length_m'],p['hard_distance_m']-.025+1e-9)
                self.assertLessEqual(abs(p['metric_target'][2]),p['hard_turn_rad']-math.radians(5)+1e-9)

    def test_stationary_does_not_train(self):
        s=n.OnlineScale(); p=n.make_proposal('stationary',n.candidates()['stationary'],s)
        s.update(p,[0,0,0],True)
        self.assertEqual(s.stats()['eligible_samples'],0)
        self.assertFalse(s.stats()['full_candidates_ready'])

    def test_nonfinite_rejected(self):
        s=n.OnlineScale(); p=n.make_proposal('forward',n.candidates()['forward'],s)
        with self.assertRaises(ValueError): s.update(p,[float('nan'),0,0],True)

    def test_learns_bounded_linear_mapping_and_requires_both_turn_directions(self):
        s=n.OnlineScale()
        for i in range(40):
            name='left_arc_0.5_15' if i%2 else 'right_arc_0.5_15'
            p=n.make_proposal(name,n.candidates()[name],s)
            x,y,a=p['effective_candidate']
            s.update(p,[.4*x,.4*y,.8*a],True)
        self.assertAlmostEqual(s.xy,.4,delta=.002)
        self.assertAlmostEqual(s.yaw,.8,delta=.002)
        self.assertTrue(s.stats()['full_candidates_ready'])
        self.assertEqual(len(n.allowed_candidates(s)),45)
        s.update(p,[0,0,0],False)
        self.assertFalse(s.stats()['full_candidates_ready'])
        self.assertEqual(len(n.allowed_candidates(s)),9)

    def test_one_direction_cannot_unlock_full_set(self):
        s=n.OnlineScale()
        for _ in range(30):
            p=n.make_proposal('left_arc_0.5_15',n.candidates()['left_arc_0.5_15'],s)
            s.update(p,p['metric_target'],True)
        self.assertFalse(s.stats()['full_candidates_ready'])

    def test_relaxed_unlock_residuals_still_have_rmse_and_outlier_limits(self):
        s=n.OnlineScale()
        s.history.extend([(.30,1.)]*8)
        s.observations=[{'fit_eligible':True, 'effective_candidate':[.5,0.,.2 if i%2 else -.2],
                         'xy_error_m':.06,'yaw_error_rad':math.radians(4.5)} for i in range(12)]
        self.assertTrue(s.stats()['full_candidates_ready'])
        s.observations[-1]['xy_error_m']=.101
        self.assertFalse(s.stats()['full_candidates_ready'])
        s.observations[-1]['xy_error_m']=.06
        s.observations[-1]['yaw_error_rad']=math.radians(8.1)
        self.assertFalse(s.stats()['full_candidates_ready'])
        for item in s.observations:
            item.update(xy_error_m=.081,yaw_error_rad=math.radians(4.5))
        self.assertFalse(s.stats()['full_candidates_ready'])
        for item in s.observations:
            item.update(xy_error_m=.06,yaw_error_rad=math.radians(5.1))
        self.assertFalse(s.stats()['full_candidates_ready'])

    def test_completed_action_with_nine_cm_error_can_train(self):
        s=n.OnlineScale(); p=n.make_proposal('forward_0.5',n.candidates()['forward_0.5'],s)
        target=p['metric_target']
        self.assertTrue(s.update(p,[target[0]-.09,target[1],target[2]],True)['fit_eligible'])
        self.assertFalse(s.update(p,[target[0]-.121,target[1],target[2]],True)['fit_eligible'])

    def test_clipped_input_not_raw_input_used_for_fit(self):
        s=n.OnlineScale(); s.xy=1.
        p=n.make_proposal('forward',n.candidates()['forward'],s,True)
        self.assertLess(p['xy_scale_factor'],1)
        observation=s.update(p,p['metric_target'],True)
        self.assertAlmostEqual(observation['effective_candidate'][0],.475)
        self.assertEqual(observation['raw_candidate'][0],2)
        self.assertAlmostEqual(observation['xy_error_m'],0)



    def test_failed_feedback_is_recorded_without_training(self):
        s=n.OnlineScale(); p=n.make_proposal('forward',n.candidates()['forward'],s)
        s.record_failed(p,'Telemetry unavailable after stop')
        self.assertEqual(s.stats()['observed_actions'],1)
        self.assertEqual(s.stats()['eligible_samples'],0)
        self.assertIsNone(s.observations[-1]['measured_m_rad'])


class Limits(unittest.TestCase):
    def proposal(self):
        return {'hard_distance_m':.25,'hard_turn_rad':math.radians(25)}

    def test_navigation_time_condition_is_half_second(self):
        self.assertEqual(n.MODEL_HORIZON_SECONDS, .5)
        self.assertEqual(n.MODEL_FRAME_OFFSET, 2.)
        self.assertEqual(n.NORMALIZED_K, .015625)

    def test_stopped_pose_inside_braking_reserve_is_valid(self):
        for final in (pose(yaw=math.radians(22.1),source=1), pose(x=.24,source=1)):
            with self.subTest(final=final):
                with self.assertRaisesRegex(RuntimeError, 'stopping reserve'):
                    n.MotionBudget(pose(),self.proposal()).check(final)
                n.MotionBudget(pose(),self.proposal()).check(final,stopped=True)

    def test_stopped_pose_still_enforces_hard_limits_and_cumulative_turn(self):
        for final in (pose(yaw=math.radians(25),source=1), pose(x=.25,source=1)):
            with self.assertRaisesRegex(RuntimeError, 'hard limit after stop'):
                n.MotionBudget(pose(),self.proposal()).check(final,stopped=True)
        guard=n.MotionBudget(pose(),self.proposal())
        guard.check(pose(yaw=math.radians(14),source=1))
        with self.assertRaisesRegex(RuntimeError, 'hard limit after stop'):
            guard.check(pose(source=2),stopped=True)

    def test_final_braking_drift_does_not_fail_completed_motion(self):
        for angle, expected in ((22.1,'succeeded'), (25.1,'failed')):
            io=MagicMock(); io.read_pose.return_value=pose()
            p=dict(self.proposal(),schedule={'vx_m_s':0.,'vy_m_s':0.,'yaw_rad_s':.3,'duration_s':1.})
            final=pose(x=.009,y=.007,yaw=math.radians(angle),source=1)
            with tempfile.TemporaryDirectory() as tmp, patch.object(n,'execute_schedule',return_value={'status':'succeeded'}), \
                 patch.object(n.time,'sleep'), patch.object(n,'stopped_pose',return_value=final):
                result=n.do_motion(SimpleNamespace(),io,MagicMock(),p,Path(tmp))
                saved=json.loads((Path(tmp)/'execution.json').read_text())
            self.assertEqual(result['status'],expected)
            self.assertEqual(saved['status'],expected)
            if expected == 'failed':
                self.assertIn('hard limit after stop',result['budget_error'])
            io.stop.assert_called(); io.disarm_navigation_motion.assert_called_once()

    def test_motion_fault_is_not_cleared_by_valid_final_pose(self):
        io=MagicMock(); io.read_pose.return_value=pose()
        p=dict(self.proposal(),schedule={'vx_m_s':0.,'vy_m_s':0.,'yaw_rad_s':.3,'duration_s':1.})
        with tempfile.TemporaryDirectory() as tmp, patch.object(n,'execute_schedule',side_effect=RuntimeError('motion fault')), \
             patch.object(n.time,'sleep'), patch.object(n,'stopped_pose',return_value=pose()):
            with self.assertRaisesRegex(RuntimeError,'motion fault'):
                n.do_motion(SimpleNamespace(),io,MagicMock(),p,Path(tmp))
            saved=json.loads((Path(tmp)/'execution.json').read_text())
        self.assertEqual(saved['status'],'failed')
        self.assertEqual(saved['error'],'motion fault')

    def test_cumulative_path_even_if_returning_to_origin(self):
        guard=n.MotionBudget(pose(),self.proposal())
        guard.check(pose(x=.12,source=1))
        with self.assertRaises(RuntimeError): guard.check(pose(x=0,source=2))

    def test_cumulative_turn_even_if_returning_to_initial_heading(self):
        guard=n.MotionBudget(pose(),self.proposal())
        guard.check(pose(yaw=math.radians(12),source=1))
        with self.assertRaises(RuntimeError): guard.check(pose(yaw=0,source=2))

    def test_small_move_and_duplicate_telemetry(self):
        guard=n.MotionBudget(pose(),self.proposal())
        p=pose(x=.1,source=1); guard.check(p); guard.check(p)
        self.assertEqual(guard.distance,.1)

    def test_sensor_failure_prevents_velocity(self):
        io=MagicMock(); capture=MagicMock(); capture.check.side_effect=RuntimeError('camera stale')
        guard=n.GuardedIO(io,capture,self.proposal(),pose())
        with self.assertRaises(RuntimeError): guard.move(.1,0,0)
        io.move.assert_not_called()



    def test_guard_does_not_suppress_motion_when_at_waypoint(self):
        io=MagicMock(); initial=pose()
        proposal=dict(self.proposal(),metric_target=[.21,0.,0.])
        io.read_pose.return_value=pose(x=.21,source=1)
        guard=n.GuardedIO(io,MagicMock(),proposal,initial)
        guard.read_pose(); guard.move(.12,0.,0.)
        io.move.assert_called_once_with(.12,0.,0.)

    def test_motion_selects_static_walk_and_uses_timed_schedule(self):
        io=MagicMock(); io.read_pose.return_value=pose()
        p=n.make_proposal('forward_0.5',n.candidates()['forward_0.5'],n.OnlineScale())
        with tempfile.TemporaryDirectory() as tmp, patch.object(n,'execute_schedule',return_value={'status':'succeeded'}) as execute, \
             patch.object(n.time,'sleep'), patch.object(n,'stopped_pose',return_value=pose(x=.06,source=1)):
            result=n.do_motion(SimpleNamespace(),io,MagicMock(),p,Path(tmp))
        io.static_walk.assert_called_once(); io.enable_avoidance.assert_called_once()
        execute.assert_called_once()
        self.assertEqual(execute.call_args.args[1],p['schedule'])
        self.assertFalse(result['waypoint_verified'])
        self.assertEqual(result['status'],'succeeded')  # Endpoint error is deliberately not a completion gate.
        self.assertAlmostEqual(result['measured_m_rad'][0],.06)


class TimedMotion(unittest.TestCase):
    def test_minimum_commands_and_upper_bounds_for_all_scale_extremes(self):
        for xy in (.05,.3,1.):
            for yaw in (.5,1.,1.5):
                s=n.OnlineScale(); s.xy=xy; s.yaw=yaw
                for full in (False,True):
                    for name,path in n.candidates().items():
                        p=n.make_proposal(name,path,s,full)
                        x,y,a=p['metric_target']; distance=math.hypot(x,y)
                        if distance: self.assertGreaterEqual(distance,.21-1e-9)
                        if a: self.assertGreaterEqual(abs(a),math.radians(15)-1e-9)
                        self.assertLessEqual(p['mapped_path_length_m'],p['hard_distance_m']-.025+1e-9)
                        self.assertLessEqual(abs(a),p['hard_turn_rad']-math.radians(5)+1e-9)
                        schedule=p['schedule']; t=schedule['duration_s']
                        twist=[schedule[k]*t for k in ('vx_m_s','vy_m_s','yaw_rad_s')]
                        for expected,actual in zip(p['metric_target'],n.twist_endpoint(twist)):
                            self.assertAlmostEqual(expected,actual)
                        for expected,actual in zip(p['metric_target'],s.predict(p['effective_path'][-1])):
                            self.assertAlmostEqual(expected,actual)

    def test_duration_controls_completion_without_using_endpoint_error(self):
        guarded=MagicMock(); now=[0.]
        schedule=n.timed_schedule([.21,0.,0.])
        result=n.execute_schedule(guarded,schedule,lambda:now[0],lambda dt:now.__setitem__(0,now[0]+dt))
        self.assertAlmostEqual(now[0],schedule['duration_s'])
        self.assertFalse(result['waypoint_verified'])
        self.assertGreater(guarded.move.call_count,20)
        for call in guarded.move.call_args_list:
            self.assertEqual(call.args,(.12,0.,0.))
        guarded.stop.assert_called_once()

    def test_budget_or_sensor_failure_stops_without_sending_more(self):
        guarded=MagicMock(); guarded.read_pose.side_effect=RuntimeError('limit')
        with self.assertRaises(RuntimeError):
            n.execute_schedule(guarded,n.timed_schedule([.21,0.,0.]))
        guarded.move.assert_not_called(); guarded.stop.assert_called_once()

    def test_reserve_stops_action_without_raising_or_learning(self):
        guarded=MagicMock(); now=[0.]
        guarded.read_pose.side_effect=[pose(), n.MotionReserveStop('stopping reserve')]
        result=n.execute_schedule(guarded,n.timed_schedule([.21,0.,0.]),
                                  lambda:now[0],lambda dt:now.__setitem__(0,now[0]+dt))
        self.assertEqual(result['status'],'stopped_at_reserve')
        self.assertEqual(result['command_count'],1)
        guarded.stop.assert_called_once()
        c=n.ExecutionCompensation()
        proposal={'candidate':'test','metric_target':[.21,0.,0.],'schedule':{}}
        update=c.update(proposal,dict(result,controller=result,measured_m_rad=[.24,0.,0.]))
        self.assertEqual(c.xy,1.)
        self.assertIn('不更新补偿',update['reason'])

    def test_reserve_stop_still_validates_final_hard_limit(self):
        for distance, expected in ((.2425,'stopped_at_reserve'),(.251,'failed')):
            io=MagicMock(); io.read_pose.return_value=pose()
            p=dict(hard_distance_m=.25,hard_turn_rad=math.radians(25),schedule=n.timed_schedule([.21,0.,0.]))
            stopped={'status':'stopped_at_reserve','stop_reason':'stopping reserve'}
            with tempfile.TemporaryDirectory() as tmp, patch.object(n,'execute_schedule',return_value=stopped), \
                 patch.object(n.time,'sleep'), patch.object(n,'stopped_pose',return_value=pose(x=distance,source=1)):
                result=n.do_motion(SimpleNamespace(),io,MagicMock(),p,Path(tmp))
            self.assertEqual(result['status'],expected)
            io.stop.assert_called()

    def test_delayed_loop_stops_instead_of_extending_duration(self):
        guarded=MagicMock(); now=[0.]
        with self.assertRaises(RuntimeError):
            n.execute_schedule(guarded,n.timed_schedule([.21,0.,0.]),lambda:now[0],
                               lambda dt:now.__setitem__(0,now[0]+.3))
        self.assertEqual(guarded.move.call_count,1)
        guarded.stop.assert_called_once()


class RemotePackets(unittest.TestCase):
    @staticmethod
    def decode(packet):
        if packet not in (b'neutral',b'active'):
            raise ValueError('invalid header')
        return {'active':packet==b'active'}

    def test_idle_bad_packet_recovers_only_after_continuously_valid_data(self):
        m=n.RemotePacketMonitor(self.decode)
        m.accept(b'neutral',1.); m.accept(b'neutral',1.25); m.check(1.25)
        m.accept(bytes(40),1.3)
        with self.assertRaises(n.RemoteTelemetryNotReady):m.check(1.3)
        m.accept(b'neutral',1.31)
        with self.assertRaises(n.RemoteTelemetryNotReady):m.check(1.4)
        m.accept(b'neutral',1.52); m.check(1.52)
        self.assertEqual(m.invalid_count,1); self.assertEqual(m.fault,'')

    def test_invalid_packets_do_not_refresh_freshness_and_gap_requires_recovery(self):
        m=n.RemotePacketMonitor(self.decode)
        m.accept(b'neutral',1.); m.accept(b'bad',1.6)
        self.assertEqual(m.last_valid,1.)
        with self.assertRaises(n.RemoteTelemetryNotReady):m.check(1.6)
        m.accept(b'neutral',2.)
        with self.assertRaises(n.RemoteTelemetryNotReady):m.check(2.)
        m.accept(b'neutral',2.21); m.check(2.21)
        m.accept(b'neutral',3.)
        with self.assertRaises(n.RemoteTelemetryNotReady):m.check(3.)

    def test_bad_packet_after_motion_armed_latches_even_if_data_recovers(self):
        m=n.RemotePacketMonitor(self.decode)
        m.accept(b'neutral',1.); m.accept(b'neutral',1.25)
        m.accept(b'bad',1.3,moving=True)
        m.accept(b'neutral',1.4); m.accept(b'neutral',1.65)
        with self.assertRaisesRegex(RuntimeError,'during_motion'):m.check(1.65)

    def test_actual_remote_takeover_is_always_latched(self):
        for moving in (False,True):
            m=n.RemotePacketMonitor(self.decode)
            m.accept(b'active',1.,moving=moving)
            m.accept(b'neutral',1.1);m.accept(b'neutral',1.4)
            with self.assertRaisesRegex(RuntimeError,'takeover'):m.check(1.4)

    def test_takeover_logs_first_packet_and_preserves_it_after_release(self):
        event={'active':True,'keys':4,'lx':.12,'ly':0.,'rx':0.,'ry':0.,'l2':0.}
        decoder=MagicMock(side_effect=[event,{'active':False},{'active':True,'keys':8}])
        m=n.RemotePacketMonitor(decoder)
        with self.assertLogs(n.LOG,level='ERROR') as logs:
            m.accept(b'packet',10.,moving=False)
            m.accept(b'packet',10.1)
            m.accept(b'packet',10.2)
        self.assertEqual(len(logs.output),1)
        self.assertEqual(m.stats()['first_takeover']['remote'],event)
        self.assertEqual(m.stats()['first_takeover']['host_monotonic'],10.)
        with self.assertRaisesRegex(RuntimeError,'takeover'):m.check(10.2)

    def test_stationary_wait_can_recover_but_never_ignores_latched_fault(self):
        now=[0.]; io=MagicMock()
        io.read_pose.side_effect=[n.RemoteTelemetryNotReady('recovering')]+[pose()]*100
        with patch.object(n.time,'monotonic',side_effect=lambda:now[0]), \
             patch.object(n.time,'sleep',side_effect=lambda dt:now.__setitem__(0,now[0]+dt)):
            n.stopped_pose(io)
        self.assertGreater(now[0],.4)
        io.read_pose.side_effect=RuntimeError('physical_remote_takeover')
        with self.assertRaisesRegex(RuntimeError,'takeover'):n.stopped_pose(io)


class Display(unittest.TestCase):
    def test_read_only_dashboard_serves_live_target_and_exact_four_inputs(self):
        import io
        import time
        import urllib.request
        import urllib.error
        import numpy as np
        from PIL import Image
        frame=SimpleNamespace(image=np.zeros((32,48,3),dtype=np.uint8),sequence=12)
        capture=SimpleNamespace(lock=threading.Lock(),frames=deque([(time.monotonic(),frame)]),
                                fault='',video=SimpleNamespace(error=''))
        display=n.LiveDisplay(capture,0)
        url='http://127.0.0.1:%d' % display.port
        images=[Image.new('RGB',(48,32),color) for color in ('red','green','blue','white')]
        try:
            display.update(goal=images[0],images=images,history={'unique_real_frames':4},
                           goal_number=2,goal_total=8,frame_index=200,reference_time_s=15.,phase='测试')
            with urllib.request.urlopen(url+'/state.json') as response:
                state=json.load(response)
            self.assertEqual(state['history']['unique_real_frames'],4)
            self.assertEqual(state['goal_number'],2)
            self.assertEqual(state['camera_sequence'],12)
            for i in range(4):
                with urllib.request.urlopen(url+'/history/%d.jpg'%i) as response:
                    pixel=Image.open(io.BytesIO(response.read())).getpixel((10,10))
                for a,b in zip(pixel,images[i].getpixel((10,10))): self.assertLessEqual(abs(a-b),3)
            with urllib.request.urlopen(url+'/live.jpg') as response:
                self.assertEqual(response.headers['Content-Type'],'image/jpeg')
                self.assertEqual(Image.open(io.BytesIO(response.read())).size,(224,224))
            for path in ('/move','/../jetson_navigate.py'):
                with self.assertRaises(urllib.error.HTTPError): urllib.request.urlopen(url+path)
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(urllib.request.Request(url+'/move',data=b'MOVE 001'))
            display.update(goal=images[1],goal_number=3,clear_inputs=True)
            self.assertEqual(display.snapshot()['assets'],['/goal.jpg'])
            capture.frames[-1]=(time.monotonic()-1.,frame)
            self.assertTrue(display.snapshot()['camera_error'])
        finally:
            display.close()


class Compensation(unittest.TestCase):
    def proposal(self, comp, name='forward_0.5'):
        return comp.apply(n.make_proposal(name,n.candidates()[name],n.OnlineScale()))

    def execution(self, measured, **kw):
        return dict(status='succeeded', measured_m_rad=measured,
                    controller={'completion_basis':'elapsed_command_time'}, **kw)

    def test_underexecution_learns_despite_calibration_rejection(self):
        c=n.ExecutionCompensation(); p=self.proposal(c)
        self.assertFalse(n.OnlineScale().update(p,[.05,0,0],True)['fit_eligible'])
        c.update(p,self.execution([.05,0,0]))
        self.assertAlmostEqual(c.xy,1.2); self.assertEqual(c.yaw,1.)
        new=self.proposal(c)
        self.assertEqual(new['metric_target'],p['metric_target'])
        self.assertAlmostEqual(new['schedule']['duration_s'],p['schedule']['duration_s']*1.2)
        self.assertAlmostEqual(new['schedule']['vx_m_s'],p['schedule']['vx_m_s'])

    def test_gains_bounded_and_no_progress_never_increases(self):
        c=n.ExecutionCompensation()
        for _ in range(30): c.update(self.proposal(c),self.execution([.03,0,0]))
        self.assertEqual(c.xy,2.)
        for measured in ([0,0,0],[-.1,0,0],[.01,0,0],[.05,.1,0],[.05,0,.3]):
            c.update(self.proposal(c),self.execution(measured))
            self.assertEqual(c.xy,2.)

    def test_failure_or_known_intervention_does_not_learn(self):
        for extra in ({'status':'failed'}, {'budget_error':'limit'}, {'feedback_error':'stale'},
                      {'avoidance_intervened':True}, {'controller':None}, {'measured_m_rad':[float('nan'),0,0]}):
            c=n.ExecutionCompensation(); execution=self.execution([.05,0,0]); execution.update(extra)
            c.update(self.proposal(c),execution)
            self.assertEqual((c.xy,c.yaw),(1.,1.))

    def test_turn_gain_independent_and_correct_sign(self):
        c=n.ExecutionCompensation()
        p=self.proposal(c,'left_arc_0.5_15')
        target=p['metric_target']; measured=target[:2]+[target[2]*.5]
        c.update(p,self.execution(measured))
        self.assertAlmostEqual(c.yaw,1.2)
        previous=c.yaw
        c.update(self.proposal(c,'left_arc_0.5_15'),self.execution(target[:2]+[-target[2]]))
        self.assertEqual(c.yaw,previous)

    def test_overexecution_reduces_compensation(self):
        c=n.ExecutionCompensation(); c.xy=1.8
        p=self.proposal(c); c.update(p,self.execution([.23,0,0]))
        self.assertLess(c.xy,1.8); self.assertGreaterEqual(c.xy,1.8*.8)

    def test_all_schedules_respect_speed_time_and_original_limits(self):
        for gx,ga in ((.5,2.),(2.,.5),(2.,2.)):
            c=n.ExecutionCompensation(); c.xy=gx; c.yaw=ga
            for full in (False,True):
                for name,path in n.candidates().items():
                    p=n.make_proposal(name,path,n.OnlineScale(),full)
                    original=p['metric_target'][:]; hard=(p['hard_distance_m'],p['hard_turn_rad'])
                    c.apply(p); s=p['schedule']
                    self.assertLessEqual(math.hypot(s['vx_m_s'],s['vy_m_s']),.120001)
                    self.assertLessEqual(abs(s['yaw_rad_s']),.350001)
                    self.assertLessEqual(s['duration_s'],15.)
                    self.assertEqual(p['metric_target'],original)
                    self.assertEqual((p['hard_distance_m'],p['hard_turn_rad']),hard)

    def test_model_scale_stays_fixed_but_accuracy_statistics_update(self):
        s=n.OnlineScale(); c=n.ExecutionCompensation(); p=self.proposal(c)
        observation=s.update(p,[.15,0,0],True,learn_mapping=False)
        self.assertTrue(observation['fit_eligible'])
        self.assertEqual((s.xy,s.yaw),(.3,1.))
        self.assertEqual(s.stats()['eligible_samples'],1)


class Confirmation(unittest.TestCase):
    def test_start_case_insensitive_and_typo_reprompts(self):
        for answer in ('START','start','Start'):
            capture=MagicMock()
            with patch('builtins.input',side_effect=['yes','',answer]) as prompt:
                self.assertTrue(n.confirm_start(capture))
                self.assertEqual(prompt.call_count,3)
        with patch('builtins.input',return_value='q'):
            self.assertFalse(n.confirm_start(MagicMock()))

    def test_start_does_not_bypass_camera_fault(self):
        capture=MagicMock(); capture.check.side_effect=[None,RuntimeError('camera stale')]
        with patch('builtins.input',return_value='start'), self.assertRaises(RuntimeError):
            n.confirm_start(capture)

    def run_fake(self, tokens, selected='forward_25', scene_score=0., poses=None,
                 target_count=1, max_actions=2, move_status='succeeded', confirmation_age=0.):
        from PIL import Image
        im=Image.new('RGB',(224,224))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); im.save(root/'goal.png')
            target={'file':'goal.png','frame_index':238,'elapsed_seconds':15.893}; plan={'targets':[target]*(target_count+1)}
            from calibration_fixture import calibration_fixture
            calibration_path = root/'fixed15.json'
            calibration_path.write_text(json.dumps(calibration_fixture()))
            args=SimpleNamespace(mode='navigate',output=root,max_actions=max_actions,model='opennwm',display_port=0,
                                 action_calibration=calibration_path, batch_size=15)
            capture=MagicMock()
            observation_count=[0]
            def latest(*a,**k):
                observation_count[0]+=1
                return im,{'sequence':observation_count[0],'timestamp_monotonic':float(observation_count[0])}
            capture.latest.side_effect=latest
            engine=MagicMock(); engine.select.side_effect=lambda *a,**kw: ({'selected':selected,'ranking':[], 'current_goal_lpips':.5},im)
            engine.distance.return_value=scene_score
            if isinstance(scene_score, Exception):
                engine.distance.side_effect=scene_score
            io=MagicMock(); io.walking_readiness.return_value={'ready':True}
            fake_ros=MagicMock()
            scale=n.OnlineScale()
            if selected=='stationary':
                statistics=scale.stats(); statistics['full_candidates_ready']=True
                scale.stats=MagicMock(return_value=statistics)

            answers=iter(tokens); prompts=[]; clock=[0.]
            def answer(prompt):
                token=next(answers)
                prompts.append({'prompt':prompt, 'answer':token, 'inferences':engine.select.call_count,
                                'comparisons':engine.distance.call_count,
                                'saved_comparisons':[json.loads(p.read_text()) for p in sorted(root.glob('step-*/image-comparison.json'))]})
                if token == 'MOVE':
                    clock[0]+=confirmation_age
                return token
            with patch.dict(sys.modules,{'rclpy':fake_ros}), patch.object(n,'robot_adapter',return_value=io), \
                 patch.object(n,'Capture',return_value=capture), patch.object(n,'Engine',return_value=engine), \
                 patch.object(n,'OnlineScale',return_value=scale), \
                 patch.object(n,'stopped_pose',return_value=pose(),side_effect=poses), patch.object(n,'do_motion') as move, \
                 patch.object(n.os,'open',return_value=123456), patch.object(n.os,'close'), patch.object(n.fcntl,'flock'), \
                 patch.object(n.time,'monotonic',side_effect=lambda:clock[0]), \
                 patch('builtins.input',side_effect=answer):
                move.return_value={'status':move_status,'measured_m_rad':[.05,0,0],'finished_monotonic':2.,
                                   'controller':{'completion_basis':'elapsed_command_time'}}
                status={}; n.run_robot(args,plan,status)
                status['test_histories']=[json.loads(p.read_text()) for p in sorted(root.glob('step-*/history.json'))]
                status['test_compensation_exists']=(root/'execution-compensation.json').exists()
                status['test_batch_inputs']=[{'count':len(call.args[2]), 'kwargs':call.kwargs} for call in engine.select.call_args_list]
                status['test_proposals']=[json.loads(p.read_text())['proposal'] for p in sorted(root.glob('step-*/selection.json'))]
                status['test_prompts']=prompts
                status['test_rejections']=[json.loads(p.read_text()) for p in sorted(root.glob('step-*/proposal-rejection.json'))]
                status['test_arrivals']=[json.loads(p.read_text()) for p in sorted(root.glob('step-*/arrival.json'))]
                status['test_inferences']=engine.select.call_count
                count=move.call_count
                io.close.assert_called_once(); capture.close.assert_called_once()
                return count,status

    def test_q_no_movement(self):
        count,status=self.run_fake(['START','q'])
        self.assertEqual(count,0); self.assertEqual(status['status'],'stopped_by_operator')

    def test_blank_or_yes_is_not_confirmation(self):
        for token in ('','yes','MOVE 001','MOVE 002'):
            count,_=self.run_fake(['START',token,'q'])
            self.assertEqual(count,0)

    def test_plain_move_executes_current_proposal(self):
        count,status=self.run_fake(['START','MOVE','','q'])
        self.assertEqual(count,1)
        self.assertEqual(status['test_histories'][0]['action_steps'],[0,0,0,0])
        self.assertEqual(status['test_histories'][1]['action_steps'],[0,0,0,1])
        self.assertFalse(status['test_compensation_exists'])
        self.assertTrue(all(v['count']==15 and not v['kwargs'] for v in status['test_batch_inputs']))
        first,second=status['test_proposals']
        self.assertEqual(first['metric_target'],second['metric_target'])
        self.assertEqual(second['schedule'],first['schedule'])

    def test_rejected_confirmation_does_not_add_history(self):
        count,status=self.run_fake(['START','yes','q'])
        self.assertEqual(count,0)
        self.assertEqual(status['test_histories'][0],status['test_histories'][1])

    def test_operator_arrival_advances_without_movement(self):
        count,status=self.run_fake(['START','arrived'])
        self.assertEqual(count,0); self.assertEqual(status['status'],'all_targets_confirmed')

    def test_stationary_does_not_move_or_auto_advance(self):
        count,status=self.run_fake(['START','MOVE','q'],selected='stationary')
        self.assertEqual(count,0); self.assertEqual(status['status'],'stopped_by_operator')

    def test_changed_scene_can_stop_without_moving_or_replanning(self):
        count,status=self.run_fake(['START','MOVE','q'],scene_score=.2)
        self.assertEqual(count,0)
        self.assertEqual(status['test_inferences'],1)
        self.assertEqual(status['test_rejections'][0]['resolution'],'stopped_by_operator')

    def test_pose_change_requires_explicit_replan_and_new_move_token(self):
        count,status=self.run_fake(['START','MOVE','MOVE','replan','MOVE','arrived'],
                                  poses=[pose(),pose(x=.04),pose(x=.04),pose(x=.04),pose(x=.09)])
        self.assertEqual(count,1)
        self.assertEqual(status['status'],'all_targets_confirmed')
        first,second=status['test_histories']
        self.assertEqual(first['sequences'],[1]*4)
        self.assertEqual(second['sequences'],[2]*4)
        self.assertEqual(second['action_steps'],[0]*4)
        self.assertTrue(second['cold_start_repetition'])
        rejection=status['test_rejections'][0]
        self.assertEqual(rejection['resolution'],'operator_replan_new_context')
        self.assertAlmostEqual(rejection['delta_m_rad'][0],.04)

    def test_scene_change_replan_does_not_execute_old_proposal(self):
        count,status=self.run_fake(['START','MOVE','replan','q'],scene_score=.2)
        self.assertEqual(count,0)
        self.assertEqual(status['test_inferences'],2)
        self.assertNotEqual(status['test_histories'][0]['sequences'],status['test_histories'][1]['sequences'])

    def test_confirmation_timeout_rejects_without_motion(self):
        count,status=self.run_fake(['START','MOVE','q'],confirmation_age=121.)
        self.assertEqual(count,0)
        self.assertIn('expired',status['test_rejections'][0]['reason'])

    def test_unrelated_runtime_error_is_not_recoverable_proposal(self):
        with self.assertRaisesRegex(RuntimeError,'GPU failure'):
            self.run_fake(['START','MOVE','replan'],scene_score=RuntimeError('GPU failure'))

    def test_post_action_lpips_saved_before_arrival_without_new_inference(self):
        with self.assertLogs(n.LOG, level='INFO') as logs:
            count,status=self.run_fake(['START','MOVE','arrived'],scene_score=.1234)
        self.assertTrue(any('执行后实际图像与目标 LPIPS 0.1234' in line for line in logs.output))
        self.assertEqual(count,1)
        self.assertEqual(status['status'],'all_targets_confirmed')
        self.assertEqual(status['test_inferences'],1)
        self.assertIn('本次动作已结束',status['test_prompts'][-1]['prompt'])
        self.assertEqual(status['test_prompts'][-1]['comparisons'],2)  # Scene guard + actual vs current goal.
        self.assertEqual(status['test_prompts'][-1]['saved_comparisons'], [{'actual_vs_reference_goal_lpips': .1234}])
        self.assertEqual(status['test_arrivals'][0]['phase'],'after_action')

    def test_post_action_q_does_not_infer_again(self):
        count,status=self.run_fake(['START','MOVE','q'])
        self.assertEqual(count,1)
        self.assertEqual(status['status'],'stopped_by_operator')
        self.assertEqual(status['test_inferences'],1)
        self.assertEqual(status['test_prompts'][-1]['comparisons'],2)
        self.assertTrue(status['test_prompts'][-1]['saved_comparisons'])

    def test_post_action_invalid_input_reprompts_without_inference(self):
        count,status=self.run_fake(['START','MOVE','MOVE','yes','arrived'])
        self.assertEqual(count,1)
        self.assertEqual(status['test_inferences'],1)
        self.assertTrue(all(p['comparisons']==2 for p in status['test_prompts'][2:]))

    def test_arrival_at_last_action_is_still_asked(self):
        count,status=self.run_fake(['START','MOVE','arrived'],max_actions=1)
        self.assertEqual(count,1)
        self.assertEqual(status['status'],'all_targets_confirmed')

    def test_arrival_advances_goal_before_next_inference(self):
        count,status=self.run_fake(['START','MOVE','arrived','arrived'],target_count=2)
        self.assertEqual(count,1)
        self.assertEqual([a['goal'] for a in status['test_arrivals']],[1,2])
        self.assertEqual(status['targets_confirmed'],2)

    def test_failed_motion_does_not_offer_arrival_or_continue(self):
        with self.assertRaisesRegex(RuntimeError,'Action did not complete'):
            self.run_fake(['START','MOVE'],move_status='failed')


if __name__=='__main__':
    unittest.main()
