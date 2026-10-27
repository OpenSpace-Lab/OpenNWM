#!/usr/bin/env python3
"""Single confirmed forward probe with DDS request tracing; no calibration/tuning."""
import argparse
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import signal
import tempfile
import threading
import time

import jetson_navigate as nav
import jetson_calibrate_actions as calibration
import calibrated_actions as action_contract
from calibrated_actions import nominal_proposals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='允许单次试验；仍需现场输入 MOVE 001')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gait', choices=['static', 'keep'], default='keep', help='static 切换碎步；keep 保留现场遥控验证的当前步态')
    parser.add_argument('--duration', type=float, default=1.4, help='单次前进时长，0.5–3.0 秒')
    parser.add_argument('--speed', type=float, choices=[.18, .25, .30], default=.18, help='仅本次诊断的前进速度')
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not .5 <= args.duration <= 3.:
        parser.error('duration 必须为 0.5–3.0 秒')
    print('单次诊断：vx=%.2f m/s，vy=0，yaw=0，最多 %.2f 秒；保留避障及遥控接管。' % (args.speed, args.duration))
    print('步态：' + ('保持当前步态（请先用遥控器确认前进正常，然后松开）' if args.gait == 'keep' else '切换碎步（旧流程，仅作对照）'))
    print('位移/路程上限 0.30 m、转角上限 10°，提前预留刹停。')
    print('限位依赖里程计与控制器响应，不能保证零超调。不会自动重试，也不会生成标定。')
    if not args.execute:
        print('仅显示诊断计划，未连接机器人。'); return
    # Explicit diagnostic-only limit; this process never emits a usable calibration.
    action_contract.FORWARD_CALIBRATION_SPEED = args.speed
    args.interface = 'eth10'
    args.runtime_root = Path('/home/unitree/ndj/code/go2-runtime')
    args.ros_setup = Path('/home/unitree/ndj/envs/go2-ros/install/setup.bash')
    args.entry_script = Path(__file__).resolve()
    nav.ensure_robot_environment(args)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(args.output/'diagnostic.log')])
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from unitree_api.msg import Request, Response
    from unitree_go.msg import SportModeState, LowState
    rclpy.init()
    io = capture = motion_lock = None
    trace = (args.output/'wire.jsonl').open('w')
    trace_lock = threading.Lock()
    def write(topic, data):
        with trace_lock:
            trace.write(json.dumps(dict(t=time.monotonic(),topic=topic,**data),ensure_ascii=False)+'\n')
            trace.flush()
    def request(topic, msg):
        write(topic, {'id':msg.header.identity.id,'api':msg.header.identity.api_id,'parameter':msg.parameter})
    def response(topic, msg):
        write(topic, {'id':msg.header.identity.id,'api':msg.header.identity.api_id,'code':msg.header.status.code,'data':msg.data})
    last_state = [0.]
    def state(msg):
        now=time.monotonic()
        if now-last_state[0]<.02:return
        last_state[0]=now
        write('/sportmodestate', dict(position=[float(x) for x in msg.position],rpy=[float(x) for x in msg.imu_state.rpy],
              velocity=[float(x) for x in msg.velocity],yaw_speed=float(msg.yaw_speed),mode=msg.mode,gait_type=msg.gait_type,error_code=msg.error_code,
              foot_force=[int(x) for x in msg.foot_force], foot_position_body=[float(x) for x in msg.foot_position_body],
              foot_speed_body=[float(x) for x in msg.foot_speed_body], foot_raise_height=float(msg.foot_raise_height)))
    last_low = [0.]
    def low(msg):
        now=time.monotonic()
        if now-last_low[0]<.02:return
        last_low[0]=now
        write('/lowstate/legs', dict(foot_force=[int(x) for x in msg.foot_force],
              q=[float(x.q) for x in msg.motor_state[:12]], dq=[float(x.dq) for x in msg.motor_state[:12]]))
    def interrupted(signum, frame):raise KeyboardInterrupt('Signal %d'%signum)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        motion_lock=os.open(str(Path(tempfile.gettempdir())/('go2-motion-%d.lock'%os.getuid())),os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        fcntl.flock(motion_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        io=nav.robot_adapter(execute=True,allow_idle_remote=True)
        qos=QoSProfile(depth=200,reliability=ReliabilityPolicy.BEST_EFFORT)
        for service in ('sport','obstacles_avoid'):
            req='/api/'+service+'/request';res='/api/'+service+'/response'
            io.subs.append(io.node.create_subscription(Request,req,lambda m,t=req:request(t,m),qos))
            io.subs.append(io.node.create_subscription(Response,res,lambda m,t=res:response(t,m),qos))
        io.subs.append(io.node.create_subscription(SportModeState,'/sportmodestate',state,qos))
        io.subs.append(io.node.create_subscription(LowState,'/lowstate',low,qos))
        write('diagnostic/config', {'gait': args.gait, 'vx': args.speed, 'duration': args.duration})
        io.wait_ready(seconds=10)
        capture=nav.Capture(args,io);capture.wait_ready()
        proposal=nominal_proposals()['forward_25']
        proposal['gait_policy'] = 'preserve_current' if args.gait == 'keep' else 'static_walk'
        proposal['schedule']=dict(vx_m_s=args.speed,vy_m_s=0.,yaw_rad_s=0.,duration_s=args.duration)
        proposal.update(hard_distance_m=.30,hard_turn_rad=math.radians(10))
        record=calibration.trial(args,io,capture,proposal,1,'diagnostic')
        record['diagnostic_gait'] = args.gait
        print('动作已停止，继续只读观察 2 秒，检查身体回缩。', flush=True)
        time.sleep(2.)
        settled=io.read_pose()
        record['late_settled_pose']=nav.pose_dict(settled)
        record['late_settled_delta_m_rad']=nav.relative_pose(record['start_pose'],nav.pose_dict(settled))
        nav.save_json(args.output/'result.json',record)
        print('实测位移/转角：',record.get('measured_m_rad'))
    finally:
        try:
            if io:io.stop()
        finally:
            try:
                if capture:capture.close()
            finally:
                try:
                    if io:io.close()
                finally:
                    trace.close()
                    if motion_lock is not None:os.close(motion_lock)
                    rclpy.shutdown()
        print('诊断记录：',args.output)


if __name__=='__main__':main()
