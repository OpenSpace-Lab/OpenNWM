#!/usr/bin/env python3
"""Offline forward/turn calibration. plan is file-only; collect needs MOVE per trial."""
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
                                summarize_samples, validate_schedule, load_calibration, HARD_DISTANCE, HARD_TURN,
                                FORWARD_CALIBRATION_SPEED, TURN_CALIBRATION_RATE, ARC_CALIBRATION_SPEED,
                                ARC_CALIBRATION_RATE, SCHEMA, contract_hash, restrict_legacy_actions)

LOG = logging.getLogger('go2.calibrate')
TUNING_POLICY = 'fixed_translation_030_forward50_seed2_v4'


def initial_proposals():
    proposals = nominal_proposals()
    for name in ('forward_25', 'forward_50'):
        proposal = proposals[name]
        proposal['schedule'] = dict(vx_m_s=FORWARD_CALIBRATION_SPEED, vy_m_s=0.,
                                    yaw_rad_s=0., duration_s=2.)
    for proposal in proposals.values():
        if proposal['nominal']['yaw_rad']:
            schedule = proposal['schedule']
            if proposal['nominal']['distance_m']:
                distance, yaw = proposal['nominal']['distance_m'], proposal['nominal']['yaw_rad']
                duration = max(1., distance/FORWARD_CALIBRATION_SPEED, abs(yaw)/ARC_CALIBRATION_RATE)
                proposal['schedule'] = dict(vx_m_s=FORWARD_CALIBRATION_SPEED, vy_m_s=0.,
                                            yaw_rad_s=yaw/duration, duration_s=duration)
                validate_schedule(proposal['schedule'])
                continue
            factor = TURN_CALIBRATION_RATE / .35
            for key in ('vx_m_s', 'vy_m_s', 'yaw_rad_s'):
                schedule[key] *= factor
            schedule['duration_s'] = max(1., schedule['duration_s']/factor)
            validate_schedule(schedule)
    return proposals


class CalibrationNeedsMoreTrials(RuntimeError):
    """Bounded pilot budget used up; preserve progress without treating it as a crash."""



class CalibrationNeedsRecollection(RuntimeError):
    """A fixed-command repeat set failed quality checks; explicit redo is required."""


def require_correctable_direction(proposal, measured):
    """Reject direction/drift faults before any duration or speed tuning."""
    if len(measured) != 3 or not all(math.isfinite(v) for v in measured):
        raise RuntimeError('Invalid measured motion; refusing to adjust command')
    nominal = proposal['nominal']
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
    # Tune to the nominal 20/45-degree angle.
    yaw_target = abs(nominal['yaw_rad'])
    yaw_gain = yaw_target/abs(actual[2]) if nominal['yaw_rad'] else 1.
    xy_gain, yaw_gain = [min(1.2, max(.8, v)) for v in (xy_gain, yaw_gain)]
    if nominal['distance_m']:
        # Linear velocity is invariant; distance feedback changes time only.
        duration = max(.25, schedule['duration_s']*xy_gain)
        yaw = schedule['yaw_rad_s']*schedule['duration_s']*yaw_gain/duration if nominal['yaw_rad'] else 0.
        yaw = max(-ARC_CALIBRATION_RATE, min(ARC_CALIBRATION_RATE, yaw))
        result = dict(vx_m_s=FORWARD_CALIBRATION_SPEED, vy_m_s=0., yaw_rad_s=yaw, duration_s=duration)
    else:
        # Pure rotation retains its angular speed and tunes elapsed time.
        result = dict(schedule, duration_s=max(.25, schedule['duration_s']*yaw_gain))
    validate_schedule(result)
    return result


def upgrade_fixed_translation(proposals, samples):
    """Recollect only commands whose linear velocity conflicts with the new policy."""
    changes = []
    for name, proposal in proposals.items():
        if not proposal['nominal']['distance_m']:
            continue
        old = proposal['schedule']
        if old['vx_m_s'] == FORWARD_CALIBRATION_SPEED and old['vy_m_s'] == 0.:
            continue
        # Keep the old command integrals as the unverified tuning seed.
        duration = max(.25, math.hypot(old['vx_m_s'], old['vy_m_s'])*old['duration_s']/FORWARD_CALIBRATION_SPEED)
        yaw = old['yaw_rad_s']*old['duration_s']/duration
        new = dict(vx_m_s=FORWARD_CALIBRATION_SPEED, vy_m_s=0.,
                   yaw_rad_s=max(-ARC_CALIBRATION_RATE, min(ARC_CALIBRATION_RATE, yaw)), duration_s=duration)
        validate_schedule(new)
        changes.append(dict(candidate=name, previous_schedule=copy.deepcopy(old),
                            next_schedule=copy.deepcopy(new),
                            archived_samples=copy.deepcopy(samples.get(name, [])),
                            next_schedule_verified=False))
        proposal['schedule'] = new
        samples[name] = []
    return changes


def upgrade_forward50_seed(proposals, samples, source_policy):
    """Replace an old, unaccepted long seed once; never relabel failed trials."""
    name = 'forward_50'
    if source_policy == TUNING_POLICY or name not in proposals or samples.get(name):
        return None
    schedule = proposals[name]['schedule']
    if schedule['duration_s'] <= 2.:
        return None
    previous = copy.deepcopy(schedule)
    schedule['duration_s'] = 2.
    validate_schedule(schedule)
    return dict(candidate=name, previous_schedule=previous, next_schedule=copy.deepcopy(schedule),
                next_schedule_verified=False, reason='Replace unaccepted long forward_50 seed; no failed samples reused')


def retry_schedule(proposal, rejected, reason, cycle):
    """Retune from the measured mean without varying the fixed linear speed."""
    for record in rejected:
        if record.get('status') not in ('succeeded', 'stopped_at_reserve') or any(
                record.get(k) for k in ('error', 'budget_error', 'feedback_error', 'avoidance_intervened')):
            raise RuntimeError('Execution/feedback fault; refusing automatic retry')
    if any(r['status'] == 'stopped_at_reserve' for r in rejected):
        raise RuntimeError('Motion reserve reached; inspect before further tuning')
    mean = [sum(r['measured_m_rad'][i] for r in rejected)/len(rejected) for i in range(3)]
    require_correctable_direction(proposal, mean)
    nominal = proposal['nominal']
    if nominal['distance_m'] and (mean[0] <= 0 or math.hypot(*mean[:2]) < .02):
        raise RuntimeError('No forward progress; refusing automatic retry')
    if nominal['yaw_rad'] and mean[2]*nominal['yaw_rad'] <= 0:
        raise RuntimeError('Incorrect rotation direction; refusing automatic retry')
    return tune_schedule(proposal, mean)


def bracket_rotation_schedule(proposal, history):
    """Interpolate clean pure-rotation pilots at the same angular speed.

    Pilots are tuning evidence only, never accepted repetition samples.
    Keep the next duration in the interior of the tightest observed bracket.
    """
    if proposal['nominal']['distance_m'] or not proposal['nominal']['yaw_rad']:
        return None
    schedule = proposal['schedule']
    target = abs(proposal['nominal']['yaw_rad'])
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


def find_auto_resume(root, repeats, requested_redo=()):
    """Choose the latest compatible wireless collection, never a plan or wired run."""
    for path in sorted(root.glob('wifi-calibration-*/calibration.json'), key=lambda p: p.parent.name, reverse=True):
        try:
            data = restrict_legacy_actions(json.loads(path.read_text()))
            marker = path.parent / 'session.json'
            if not marker.exists():
                marker = path.parent / 'status.json'
            metadata = json.loads(marker.read_text())
            if metadata.get('mode') != 'collect':
                continue
            if data.get('schema') != SCHEMA or data.get('repeats') != repeats:
                continue
            # Validate the contract before interpreting samples by current action names.
            # Clearing samples here is in-memory only; rejected groups are handled below.
            load_calibration(path, require_ready=False, redo_candidates=list(data['actions']))
            redo = list(requested_redo)
            for name, entry in data['actions'].items():
                if name == 'stationary' or not entry['samples']:
                    continue
                try:
                    summarize_samples(name, entry['samples'], len(entry['samples']))
                except ValueError:
                    redo.append(name)
            redo = list(dict.fromkeys(redo))
            # Check definitions, schedules and every retained sample before selection.
            load_calibration(path, require_ready=False, redo_candidates=redo)
            return path, redo
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None, list(requested_redo)


def collect(args, samples, proposals, status):
    moving = [name for name in proposals if name != 'stationary']
    if all(len(samples.get(name, [])) >= args.repeats for name in moving):
        for name in moving:
            summarize_samples(name, samples[name], args.repeats)
        LOG.info('所有运动动作已合格，跳过采集，不连接运动接口')
        status['status'] = 'complete'
        return
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
            if name == 'stationary':
                LOG.info('stationary: 使用内置零动作，跳过采集')
                continue
            existing = samples.get(name, [])
            if existing:
                summarize_samples(name, existing, len(existing))
                if len(existing) >= args.repeats:
                    LOG.info('%s: 保留已有 %d 个合格样本，跳过', name, len(existing))
                    continue
                LOG.info('%s: 保留 %d 个样本，沿用冻结指令继续', name, len(existing))
            cycle = 0
            while True:
                # Pilots are saved separately; no changing schedules inside the final repeat set.
                if name != 'stationary' and not existing:
                    tuning_round = 0
                    while True:
                        number += 1
                        pilot = trial(args, io, capture, proposal, number, 'pilot-%d-%d' % (cycle, tuning_round))
                        history.append(copy.deepcopy(pilot))
                        nav.save_json(args.output / 'pilot-history.json', history)
                        try:
                            validate_pilot(proposal, pilot)
                            break
                        except ValueError as exc:
                            LOG.warning('Pilot needs offline adjustment: %s', exc)
                            if pilot.get('status') != 'succeeded' or any(pilot.get(k) for k in ('error', 'budget_error', 'feedback_error', 'avoidance_intervened')):
                                raise RuntimeError('Invalid feedback or motion reserve reached; no tuning')
                            previous_schedule = proposal['schedule'].copy()
                            proposal['schedule'] = (bracket_rotation_schedule(proposal, history)
                                                    or tune_schedule(proposal, pilot['measured_m_rad']))
                            nav.save_json(args.output / 'pilot-tuning.json', {
                                'candidate': name, 'reason': str(exc), 'measured_m_rad': pilot['measured_m_rad'],
                                'previous_schedule': previous_schedule, 'next_schedule': proposal['schedule'],
                                'next_schedule_verified': False})
                            nav.save_json(args.output / 'frozen-command-candidates.json', proposals)
                            nav.save_json(args.output / 'calibration.json', build_calibration(samples, args.repeats, proposals))
                            LOG.info('已保存下一次待验证指令：%s；当前实测转角 %+.4fπ / 名义 %+.4fπ',
                                     name, pilot['measured_m_rad'][2]/math.pi, proposal['nominal']['yaw_rad']/math.pi)
                            tuning_round += 1
                            if args.max_tuning_rounds and tuning_round >= args.max_tuning_rounds:
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
                    summary = summarize_samples(name, samples[name], args.repeats)
                    for warning in summary.get('warnings', []):
                        LOG.warning('%s: %s', name, warning)
                except ValueError as exc:
                    cycle += 1
                    # Archive every rejected sample before changing the frozen command.
                    previous = copy.deepcopy(proposal['schedule'])
                    next_schedule = retry_schedule(proposal, samples[name], str(exc), cycle)
                    nav.save_json(args.output / ('rejected-%s-%03d.json' % (name, cycle)),
                                  dict(candidate=name, reason=str(exc), samples=samples[name],
                                       previous_schedule=previous, next_schedule=next_schedule,
                                       next_schedule_verified=False))
                    if args.max_tuning_rounds and cycle >= args.max_tuning_rounds:
                        raise CalibrationNeedsRecollection('%s；达到指定重试组数，进度已保存' % exc) from exc
                    proposal['schedule'] = next_schedule
                    samples[name] = []
                    existing = []
                    nav.save_json(args.output / 'calibration.json', build_calibration(samples, args.repeats, proposals))
                    nav.save_json(args.output / 'frozen-command-candidates.json', proposals)
                    LOG.warning('%s；失败组已留档，继续第 %d 轮试调并重新测量 %d 次；下一指令 %s',
                                exc, cycle+1, args.repeats, next_schedule)
                    continue
                LOG.info('%s: %d repetitions accepted; command frozen', name, args.repeats)
                break
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
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument('--resume', type=Path, help='Read accepted samples from an earlier calibration.json; always write a new output directory')
    restart.add_argument('--auto-resume', action='store_true', help='Resume latest compatible wireless collection; recollect rejected candidates')
    parser.add_argument('--redo-candidate', action='append', default=[],
                        help='With --resume, recollect this named candidate in the new output; repeat flag for multiple names')
    parser.add_argument('--max-tuning-rounds', type=int, default=0,
                        help='0 (default): retry until accepted or operator exits; 1..12 limits pilots per group and rejected groups')
    parser.add_argument('--runtime-root', type=Path, default=nav.REPO.parent / 'go2-runtime')
    parser.add_argument('--ros-setup', type=Path, default=nav.REPO.parents[1] / 'envs/go2-ros/install/setup.bash')
    parser.add_argument('--interface', default='eth10')
    args = parser.parse_args()
    if not 3 <= args.repeats <= 20 or not 0 <= args.max_tuning_rounds <= 12:
        parser.error('repeats must be 3..20; max-tuning-rounds 0..12')
    if not re.fullmatch(r'[a-zA-Z0-9_.:-]+', args.interface):
        parser.error('Invalid network interface')
    for name in ('output', 'runtime_root', 'ros_setup'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.auto_resume:
        if args.mode != 'collect':
            parser.error('--auto-resume requires collect')
        args.resume, args.redo_candidate = find_auto_resume(args.output.parent, args.repeats, args.redo_candidate)
        print('自动续跑：' + (str(args.resume) if args.resume else '无兼容无线进度，从头开始'), flush=True)
        if args.redo_candidate:
            print('本次重测：' + ', '.join(args.redo_candidate), flush=True)
    if args.redo_candidate and args.resume is None:
        parser.error('--redo-candidate requires a compatible resume source')
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
    proposals, samples = initial_proposals(), {}
    if resumed is not None:
        proposals = {name: entry['proposal'] for name, entry in resumed['actions'].items()}
        samples = {name: entry['samples'] for name, entry in resumed['actions'].items()}
        source_data = json.loads(args.resume.read_text())
        upgrades = upgrade_fixed_translation(proposals, samples)
        upgraded = {item['candidate'] for item in upgrades}
        if upgrades:
            nav.save_json(args.output / 'fixed-speed-migration.json', upgrades)
            LOG.info('旧线速度切换为固定 0.30 m/s，对应旧样本留档并重新试调：%s', ', '.join(sorted(upgraded)))
        adjustments = [retune_recollection(proposals[name], source_data['actions'][name]['samples'])
                       for name in dict.fromkeys(args.redo_candidate) if name not in upgraded]
        if adjustments:
            nav.save_json(args.output / 'recollection-adjustments.json', adjustments)
            for adjustment in adjustments:
                LOG.info('重采前按旧重复测量均值校正：%s', json.dumps(adjustment, ensure_ascii=False))
        nav.save_json(args.output / 'resume-source.json', {
            'path': str(args.resume), 'sha256': nav.digest(args.resume),
            'samples_preserved': {name: len(v) for name, v in samples.items()},
            'limit_migration': resumed.get('limit_migration'),
            'action_set_migration': resumed.get('action_set_migration'),
            'recollection': resumed.get('recollection')})
        if resumed.get('recollection'):
            LOG.info('仅在新运行中重采所选候选，源文件不变：%s', resumed['recollection'])
        if resumed.get('limit_migration'):
            LOG.info('硬限位已迁移至 1.00 m / π/3；需重新采集的旧限位停止候选：%s',
                     resumed['limit_migration']['recollect_candidates'])
        marker = args.resume.parent / 'session.json'
        source_policy = json.loads(marker.read_text()).get('tuning_policy') if marker.exists() else None
        seed_change = upgrade_forward50_seed(proposals, samples, source_policy)
        if seed_change:
            nav.save_json(args.output / 'forward50-seed-migration.json', seed_change)
            LOG.info('forward_50 尚未验收：起始时长由 %.3f s 缩短为 2.000 s，需重新试调',
                     seed_change['previous_schedule']['duration_s'])
    args.pilot_history = restore_pilot_history(args.resume) if resumed is not None else []
    # An explicit redo starts a new tuning experiment for that candidate.
    args.pilot_history = [r for r in args.pilot_history
                          if r.get('proposal', {}).get('candidate') not in set(args.redo_candidate) | (upgraded if resumed is not None else set())]
    for name, proposal in proposals.items():
        if not samples.get(name):
            adjusted = bracket_rotation_schedule(proposal, args.pilot_history)
            if adjusted is not None:
                LOG.info('恢复历史试调区间：%s；下一次待验证时长 %.3f s', name, adjusted['duration_s'])
                proposal['schedule'] = adjusted
    nav.save_json(args.output / 'pilot-history.json', args.pilot_history)
    moving_count = sum(name != 'stationary' for name in proposals)
    nav.save_json(args.output / 'plan.json', dict(waypoint_spacing_m=SPACING,
        candidates=candidate_specs(), proposals=proposals, repetitions=args.repeats,
        fixed_repeat_trials=moving_count*args.repeats, moving_repeat_trials=moving_count*args.repeats,
        stationary_policy='builtin_zero_action_no_collection',
        additional_pilots_min=moving_count, additional_pilots_max=(moving_count*args.max_tuning_rounds if args.max_tuning_rounds else None),
        tuning_policy=TUNING_POLICY, retry_until_accepted=args.max_tuning_rounds == 0,
        nominal_distance_is_arc_length=True, hard_path_limit_m=HARD_DISTANCE, hard_turn_limit_rad=HARD_TURN,
        stopping_reserve_m=.015, stopping_reserve_rad=math.radians(4)))
    nav.save_json(args.output / 'session.json', dict(mode=args.mode, interface=args.interface, repeats=args.repeats, tuning_policy=TUNING_POLICY))
    nav.save_json(args.output / 'calibration.json', build_calibration(samples, args.repeats, proposals))
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
