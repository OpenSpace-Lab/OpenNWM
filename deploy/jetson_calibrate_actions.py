#!/usr/bin/env python3
"""Offline fixed15 calibration. plan is file-only; collect needs a MOVE token per trial."""
import argparse
import copy
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
import time

import jetson_navigate as nav
from calibrated_actions import (SPACING, candidate_specs, nominal_proposals, build_calibration,
                                summarize_samples, validate_schedule, load_calibration, HARD_DISTANCE, HARD_TURN)

LOG = logging.getLogger('go2.calibrate')


class CalibrationNeedsMoreTrials(RuntimeError):
    """Bounded pilot budget used up; preserve progress without treating it as a crash."""



class CalibrationNeedsRecollection(RuntimeError):
    """A fixed-command repeat set failed quality checks; explicit redo is required."""


def require_correctable_direction(proposal, measured):
    """Reject direction/drift faults before any duration or speed tuning."""
    if len(measured) != 3 or not all(math.isfinite(v) for v in measured):
        raise RuntimeError('Invalid measured motion; refusing to adjust command')
    nominal = proposal['nominal']
    if nominal['distance_m'] and not nominal['yaw_rad'] and abs(measured[1]) > .04:
        raise RuntimeError('直行侧移超限：dy=%+.3f m，限值 0.04 m；停止标定，不延长指令，请检查底层控制/避障/地面与定位' % measured[1])
    if not nominal['distance_m'] and nominal['yaw_rad'] and math.hypot(*measured[:2]) > .04:
        raise RuntimeError('原地转向平移超限：%.3f m，限值 0.04 m；停止标定，不继续试调' % math.hypot(*measured[:2]))


def tune_schedule(proposal, measured):
    """Offline pilot adjustment only; final repeat trials always use one frozen schedule."""
    if len(measured) != 3 or not all(math.isfinite(v) for v in measured):
        raise RuntimeError('Invalid measured motion; refusing to adjust command')
    require_correctable_direction(proposal, measured)
    nominal = proposal['nominal']
    actual = nav.body_twist(measured)
    schedule = proposal['schedule']
    if nominal['distance_m'] and (math.hypot(*actual[:2]) < .02 or actual[0] <= 0):
        raise RuntimeError('No forward progress; refusing to amplify command')
    if nominal['yaw_rad'] and (actual[2]*nominal['yaw_rad'] <= 0 or abs(actual[2]) < math.pi/180):
        raise RuntimeError('No correctly directed rotation; refusing to amplify command')
    xy_gain = nominal['distance_m']/math.hypot(*actual[:2]) if nominal['distance_m'] else 1.
    # Aiming slightly above the lower edge avoids accepting sub-15-degree averages.
    yaw_target = max(abs(nominal['yaw_rad']), math.radians(18))
    yaw_gain = yaw_target/abs(actual[2]) if nominal['yaw_rad'] else 1.
    xy_gain, yaw_gain = [min(1.2, max(.8, v)) for v in (xy_gain, yaw_gain)]
    integrated = [schedule['vx_m_s']*schedule['duration_s']*xy_gain,
                  schedule['vy_m_s']*schedule['duration_s']*xy_gain,
                  schedule['yaw_rad_s']*schedule['duration_s']*yaw_gain]
    duration = max(1., math.hypot(*integrated[:2])/.12, abs(integrated[2])/.35)
    result = dict(zip(('vx_m_s', 'vy_m_s', 'yaw_rad_s'), [v/duration for v in integrated]), duration_s=duration)
    validate_schedule(result)
    return result


def bracket_rotation_schedule(proposal, history):
    """Interpolate clean pure-rotation pilots at the same angular speed.

    Pilots are tuning evidence only, never accepted repetition samples.
    Keep the next duration in the interior of the tightest observed bracket.
    """
    if proposal['nominal']['distance_m'] or not proposal['nominal']['yaw_rad']:
        return None
    schedule = proposal['schedule']
    target = max(abs(proposal['nominal']['yaw_rad']), math.radians(18))
    points = []
    for record in history:
        try:
            old = record['proposal']
            command = old['schedule']
            validate_schedule(command)
            measured = record['measured_m_rad']
            if (old['candidate'] != proposal['candidate'] or old['nominal'] != proposal['nominal']
                    or record.get('status') != 'succeeded'
                    or record.get('controller', {}).get('completion_basis') != 'elapsed_command_time'
                    or any(record.get(k) for k in ('error', 'budget_error', 'feedback_error', 'avoidance_intervened'))
                    or not str(record.get('phase', '')).startswith('pilot-')
                    or command['vx_m_s'] != 0 or command['vy_m_s'] != 0
                    or not math.isclose(command['yaw_rad_s'], schedule['yaw_rad_s'], abs_tol=1e-9)
                    or len(measured) != 3 or not all(math.isfinite(v) for v in measured)
                    or measured[2]*proposal['nominal']['yaw_rad'] <= 0
                    or math.hypot(*measured[:2]) > .04
                    or not 0 <= record['measured_path_m'] < HARD_DISTANCE
                    or not abs(measured[2]) <= record['measured_accumulated_turn_rad'] < HARD_TURN):
                continue
            points.append((command['duration_s'], abs(measured[2])))
        except (KeyError, TypeError, ValueError):
            continue
    brackets = [(lo, hi) for lo in points for hi in points
                if lo[1] < target < hi[1] and lo[0] < hi[0]]
    if not brackets:
        return None
    lo, hi = min(brackets, key=lambda pair: pair[1][0]-pair[0][0])
    fraction = min(.75, max(.25, (target-lo[1])/(hi[1]-lo[1])))
    duration = lo[0]+fraction*(hi[0]-lo[0])
    # Retain the existing per-adjustment bound even when restoring old pilots.
    if not .8*schedule['duration_s'] <= duration <= 1.2*schedule['duration_s']:
        return None
    result = dict(schedule, duration_s=duration)
    validate_schedule(result)
    return result


def restore_pilot_history(source):
    """Read the previous run's carried history and trial records, without ROS IO."""
    records = []
    carried = source.parent / 'pilot-history.json'
    if carried.exists():
        records.extend(json.loads(carried.read_text()))
    for path in sorted(source.parent.glob('*-pilot-*/execution.json')):
        records.append(json.loads(path.read_text()))
    unique = {json.dumps(r, sort_keys=True): r for r in records}
    return list(unique.values())


def retune_recollection(proposal, old_samples):
    """Use a stable rejected repeat set to correct bias before a fresh pilot/set.

    Never learn from unstable, failed, reserve-stopped or known-intervened trials.
    The old trials are provenance only and are not added to the new calibration.
    """
    report = {'candidate': proposal['candidate'], 'applied': False}
    if len(old_samples) < 3:
        return dict(report, reason='Fewer than three previous repetitions')
    for sample in old_samples:
        if (sample.get('status') != 'succeeded'
                or sample.get('controller', {}).get('completion_basis') != 'elapsed_command_time'
                or sample.get('proposal', {}).get('schedule') != proposal['schedule']
                or any(sample.get(k) for k in ('error', 'budget_error', 'feedback_error', 'avoidance_intervened'))):
            return dict(report, reason='Previous trials were not identical clean timed commands')
    try:
        summarize_samples(proposal['candidate'], old_samples, len(old_samples))
        return dict(report, reason='Previous repeat set already met quality requirements')
    except ValueError as exc:
        reason = str(exc)
        detail = reason.split(': ', 1)[-1]
        if not detail.startswith(('mean turn ', 'turn differs from nominal', 'distance differs from nominal')):
            return dict(report, reason='Not a stable correctable magnitude bias: ' + reason)
    mean = [sum(s['measured_m_rad'][i] for s in old_samples)/len(old_samples) for i in range(3)]
    previous = proposal['schedule'].copy()
    adjusted = tune_schedule(proposal, mean)
    proposal['schedule'] = adjusted
    return dict(report, applied=True, reason=reason, source_repetitions=len(old_samples),
                measured_mean_m_rad=mean, previous_schedule=previous,
                next_schedule=adjusted, next_schedule_verified=False)


def validate_pilot(proposal, pilot):
    require_correctable_direction(proposal, pilot['measured_m_rad'])
    name = proposal['candidate']
    summary = summarize_samples(name, [pilot], 1)
    # The lower yaw boundary is 15 degrees. A single barely passing pilot
    # cannot establish that repeated runs will remain above that boundary.
    if (math.isclose(abs(proposal['nominal']['yaw_rad']), math.pi/12)
            and abs(summary['mean_m_rad'][2]) < math.radians(17)):
        raise ValueError('%s: pilot turn margin insufficient (%.3f deg; need >=17 deg before repeats; tune toward 18 deg)'
                         % (name, math.degrees(abs(summary['mean_m_rad'][2]))))
    return summary


def trial(args, io, capture, proposal, number, phase):
    name = proposal['candidate']
    directory = args.output / ('%03d-%s-%s' % (number, name, phase))
    directory.mkdir()
    schedule = proposal['schedule']
    LOG.info('试验 %03d %s %s | 名义路程 %.2fm，转角 %+.4fπ | vx %.3f vy %.3f yaw %.3f，%.3fs',
             number, name, phase, proposal['nominal']['distance_m'], proposal['nominal']['yaw_rad']/math.pi,
             schedule['vx_m_s'], schedule['vy_m_s'], schedule['yaw_rad_s'], schedule['duration_s'])
    while True:
        token = input('可先用遥控器移到空旷处；松开摇杆/按键并停稳后输入 MOVE %03d；q 退出：' % number).strip()
        if token != 'MOVE %03d' % number:
            raise KeyboardInterrupt('Operator quit or did not enter exact trial token')
        capture.check()
        try:
            io.begin_calibration_trial()
            break
        except nav.RemoteTelemetryNotReady as exc:
            LOG.warning('%s；尚未发送运动指令，请重新输入本次 MOVE', exc)
    try:
        return measure_trial(args, io, capture, proposal, phase, directory)
    finally:
        # Keep remote takeover protection through stop, settling and the final image.
        io.end_calibration_trial()


def measure_trial(args, io, capture, proposal, phase, directory):
    name, schedule = proposal['candidate'], proposal['schedule']
    capture.check()
    initial = nav.stopped_pose(io)
    before, before_meta = capture.latest(after=time.monotonic())
    before.save(directory / 'before.png')
    if name == 'stationary':
        # Read-only observation; no walking mode, avoidance or velocity request.
        until = time.monotonic()+schedule['duration_s']
        while time.monotonic() < until:
            capture.check()
            io.read_pose()
            time.sleep(.02)
        final = nav.stopped_pose(io)
        record = dict(status='succeeded', proposal=copy.deepcopy(proposal),
                      measured_m_rad=nav.relative_pose(nav.pose_dict(initial), nav.pose_dict(final)),
                      measured_path_m=math.hypot(final.x-initial.x, final.y-initial.y),
                      measured_accumulated_turn_rad=abs(nav.wrap(final.yaw-initial.yaw)),
                      finished_monotonic=time.monotonic(), control_sent=False)
    else:
        record = nav.do_motion(args, io, capture, proposal, directory)
    after, after_meta = capture.latest(after=record['finished_monotonic'])
    io.read_pose()  # Reject a takeover/fault during settling or final-image capture.
    after.save(directory / 'after.png')
    record.update(observation_horizon_s=after_meta['timestamp_monotonic']-before_meta['timestamp_monotonic'],
                  before_image=before_meta, after_image=after_meta, phase=phase,
                  physical_camera_pose_sync_verified=False)
    nav.save_json(directory / 'execution.json', record)
    if record['status'] not in ('succeeded', 'stopped_at_reserve'):
        raise RuntimeError('Trial failed; inspect ' + str(directory / 'execution.json'))
    LOG.info('实测 dx %.4fm dy %.4fm yaw %+.4fπ；图像间隔 %.3fs',
             *record['measured_m_rad'][:2], record['measured_m_rad'][2]/math.pi, record['observation_horizon_s'])
    return record


def collect(args, samples, proposals, status):
    import rclpy
    rclpy.init()
    io = capture = None
    lock = None
    number = 0
    history = getattr(args, 'pilot_history', [])
    try:
        lock = os.open(str(Path(tempfile.gettempdir()) / ('go2-motion-%d.lock' % os.getuid())),
                       os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        io = nav.robot_adapter(execute=True, allow_idle_remote=True)
        io.wait_ready(seconds=10)
        capture = nav.Capture(args, io)
        capture.wait_ready()
        for name, proposal in proposals.items():
            existing = samples.get(name, [])
            if existing:
                summarize_samples(name, existing, len(existing))
                if len(existing) >= args.repeats:
                    LOG.info('%s: 保留已有 %d 个合格样本，跳过', name, len(existing))
                    continue
                LOG.info('%s: 保留 %d 个样本，沿用冻结指令继续', name, len(existing))
            # Pilots are saved separately; no changing schedules inside the final repeat set.
            if name != 'stationary' and not existing:
                for tuning_round in range(args.max_tuning_rounds):
                    number += 1
                    pilot = trial(args, io, capture, proposal, number, 'pilot-%d' % tuning_round)
                    history.append(copy.deepcopy(pilot))
                    nav.save_json(args.output / 'pilot-history.json', history)
                    try:
                        validate_pilot(proposal, pilot)
                        break
                    except ValueError as exc:
                        LOG.warning('Pilot needs offline adjustment: %s', exc)
                        if any(pilot.get(k) for k in ('error', 'budget_error', 'feedback_error', 'avoidance_intervened')):
                            raise RuntimeError('Invalid feedback; no tuning')
                        previous_schedule = proposal['schedule'].copy()
                        proposal['schedule'] = (bracket_rotation_schedule(proposal, history)
                                                or tune_schedule(proposal, pilot['measured_m_rad']))
                        nav.save_json(args.output / 'pilot-tuning.json', {
                            'candidate': name, 'reason': str(exc), 'measured_m_rad': pilot['measured_m_rad'],
                            'previous_schedule': previous_schedule, 'next_schedule': proposal['schedule'],
                            'next_schedule_verified': False})
                        nav.save_json(args.output / 'frozen-command-candidates.json', proposals)
                        LOG.info('已保存下一次待验证指令：%s；当前实测转角 %+.4fπ / 名义 %+.4fπ',
                                 name, pilot['measured_m_rad'][2]/math.pi, proposal['nominal']['yaw_rad']/math.pi)
                        if tuning_round+1 == args.max_tuning_rounds:
                            raise CalibrationNeedsMoreTrials(
                                '%s：本轮 %d 次试调已用完，仍未达标；已保存进度和下一次待验证指令，可用 --resume 继续'
                                % (name, args.max_tuning_rounds))
            samples.setdefault(name, [])
            for repetition in range(len(samples[name]), args.repeats):
                number += 1
                result = trial(args, io, capture, proposal, number, 'repeat-%d' % (repetition+1))
                samples[name].append(result)
                nav.save_json(args.output / 'calibration.json', build_calibration(samples, args.repeats, proposals))
            try:
                summarize_samples(name, samples[name], args.repeats)
            except ValueError as exc:
                raise CalibrationNeedsRecollection(
                    '%s；本组重复测量不合格，进度已保存。续跑时加 --redo-candidate %s，仅重测此候选' % (exc, name)) from exc
            LOG.info('%s: %d repetitions accepted; command frozen', name, args.repeats)
        status['status'] = 'complete'
    finally:
        try:
            if io:
                io.stop()
        finally:
            try:
                if capture:
                    capture.close()
            finally:
                try:
                    if io:
                        io.close()
                finally:
                    if lock is not None:
                        os.close(lock)
                    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'collect'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--resume', type=Path, help='Read accepted samples from an earlier calibration.json; always write a new output directory')
    parser.add_argument('--redo-candidate', action='append', default=[],
                        help='With --resume, recollect this named candidate in the new output; repeat flag for multiple names')
    parser.add_argument('--max-tuning-rounds', type=int, default=8,
                        help='Pilot attempts per unfinished candidate in this run; default 8, maximum 12')
    parser.add_argument('--runtime-root', type=Path, default=nav.REPO.parent / 'go2-runtime')
    parser.add_argument('--ros-setup', type=Path, default=nav.REPO.parents[1] / 'envs/go2-ros/install/setup.bash')
    parser.add_argument('--interface', default='eth10')
    args = parser.parse_args()
    if not 3 <= args.repeats <= 20 or not 1 <= args.max_tuning_rounds <= 12:
        parser.error('repeats must be 3..20; max-tuning-rounds 1..12')
    if not re.fullmatch(r'[a-zA-Z0-9_.:-]+', args.interface):
        parser.error('Invalid network interface')
    for name in ('output', 'runtime_root', 'ros_setup'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.redo_candidate and args.resume is None:
        parser.error('--redo-candidate requires --resume')
    resumed = None
    if args.resume is not None:
        if args.mode != 'collect':
            parser.error('--resume requires collect')
        args.resume = args.resume.expanduser().resolve()
        resumed = load_calibration(args.resume, require_ready=False, redo_candidates=args.redo_candidate)
        if resumed['repeats'] != args.repeats:
            parser.error('--repeats must match the source calibration')
    if args.mode == 'collect':
        if not sys.stdin.isatty():
            parser.error('collect requires an interactive terminal and individual MOVE confirmations')
        if args.interface == 'lo' or not Path('/sys/class/net', args.interface).exists():
            parser.error('A real robot network interface is required')
        args.entry_script = Path(__file__).resolve()
        nav.ensure_robot_environment(args)
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output / 'calibration.log')])
    proposals, samples = nominal_proposals(), {}
    if resumed is not None:
        proposals = {name: entry['proposal'] for name, entry in resumed['actions'].items()}
        samples = {name: entry['samples'] for name, entry in resumed['actions'].items()}
        source_data = json.loads(args.resume.read_text())
        adjustments = [retune_recollection(proposals[name], source_data['actions'][name]['samples'])
                       for name in dict.fromkeys(args.redo_candidate)]
        if adjustments:
            nav.save_json(args.output / 'recollection-adjustments.json', adjustments)
            for adjustment in adjustments:
                LOG.info('重采前按旧重复测量均值校正：%s', json.dumps(adjustment, ensure_ascii=False))
        nav.save_json(args.output / 'resume-source.json', {
            'path': str(args.resume), 'sha256': nav.digest(args.resume),
            'samples_preserved': {name: len(v) for name, v in samples.items()},
            'limit_migration': resumed.get('limit_migration'),
            'recollection': resumed.get('recollection')})
        if resumed.get('recollection'):
            LOG.info('仅在新运行中重采所选候选，源文件不变：%s', resumed['recollection'])
        if resumed.get('limit_migration'):
            LOG.info('硬限位已迁移至 0.70 m / π/3；需重新采集的旧限位停止候选：%s',
                     resumed['limit_migration']['recollect_candidates'])
    args.pilot_history = restore_pilot_history(args.resume) if resumed is not None else []
    # An explicit redo starts a new tuning experiment for that candidate.
    args.pilot_history = [r for r in args.pilot_history
                          if r.get('proposal', {}).get('candidate') not in args.redo_candidate]
    for name, proposal in proposals.items():
        if not samples.get(name):
            adjusted = bracket_rotation_schedule(proposal, args.pilot_history)
            if adjusted is not None:
                LOG.info('恢复历史试调区间：%s；下一次待验证时长 %.3f s', name, adjusted['duration_s'])
                proposal['schedule'] = adjusted
    nav.save_json(args.output / 'pilot-history.json', args.pilot_history)
    nav.save_json(args.output / 'plan.json', dict(waypoint_spacing_m=SPACING,
        candidates=candidate_specs(), proposals=proposals, repetitions=args.repeats,
        fixed_repeat_trials=15*args.repeats, moving_repeat_trials=14*args.repeats,
        additional_pilots_min=14, additional_pilots_max=14*args.max_tuning_rounds,
        nominal_distance_is_arc_length=True, hard_path_limit_m=HARD_DISTANCE, hard_turn_limit_rad=HARD_TURN,
        stopping_reserve_m=.015, stopping_reserve_rad=math.radians(4)))
    status = dict(status='plan_only_no_robot_io', mode=args.mode)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Signal %d' % signum)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.mode == 'collect':
            status['status'] = 'collecting'
            collect(args, samples, proposals, status)
    except CalibrationNeedsRecollection as exc:
        status.update(status='needs_recollection', error=str(exc))
        LOG.warning('%s', exc)
    except CalibrationNeedsMoreTrials as exc:
        status.update(status='needs_more_pilots', error=str(exc))
        LOG.warning('%s', exc)
    except KeyboardInterrupt as exc:
        status.update(status='interrupted', error=str(exc))
        LOG.warning('Stopped; partial calibration is not usable for navigation')
    except Exception as exc:
        status.update(status='failed', error=str(exc))
        LOG.exception('Calibration stopped')
        raise
    finally:
        calibration = build_calibration(samples, args.repeats, proposals)
        if args.mode == 'collect' and status['status'] != 'complete':
            calibration['ready'] = False
            calibration['problems'].append('Collection did not finish cleanly: ' + status['status'])
        nav.save_json(args.output / 'calibration.json', calibration)
        status['ready'] = calibration['ready']
        nav.save_json(args.output / 'status.json', status)
        LOG.info('Calibration ready=%s; %s', calibration['ready'], args.output)


if __name__ == '__main__':
    main()
