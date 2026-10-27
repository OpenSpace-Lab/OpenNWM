#!/usr/bin/env python3
"""One entry point for reference preparation and confirmed Go2 visual navigation."""
from __future__ import annotations

import argparse
from collections import deque, OrderedDict
import csv
from dataclasses import asdict
import fcntl
import hashlib
import io as image_io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time

sys.dont_write_bytecode = True
from jetson_infer import REPO, MODEL_CHOICES, load_models, add_sampler_arguments, validate_sampler_arguments
from jetson_episode import candidates, digest, relative_pose
from calibrated_actions import (load_calibration, calibrated_proposals, HARD_DISTANCE, HARD_TURN,
                                MODEL_HORIZON_SECONDS, MODEL_FRAME_OFFSET, NORMALIZED_K)

# All navigation candidates use 0.5 seconds at the training convention of 4 FPS.
# Measured observation intervals are retained for diagnostics only.

from navigation_constraints import TurnReversalGuard

LOG = logging.getLogger('go2.navigate')
MIN_TRANSLATION_M = .21
MIN_TURN_RAD = math.radians(15)
PLANNING_DISTANCE_RESERVE_M = .025
STOP_DISTANCE_RESERVE_M = .015


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def pose_dict(pose):
    return {'x': pose.x, 'y': pose.y, 'yaw': pose.yaw, 'received': pose.received,
            'source': pose.source, 'speed': pose.speed, 'yaw_speed': pose.yaw_speed}


def model_frame(image):
    """Same PIL geometry as misc.CenterCropAR + torchvision bilinear Resize, once only."""
    from PIL import Image, ImageOps
    image = image.convert('RGB')
    w, h = image.size
    crop_w, crop_h = (int(h*4/3), h) if w > h else (w, int(w/(4/3)))
    pad_w, pad_h = max(0, crop_w-w), max(0, crop_h-h)
    if pad_w or pad_h:
        image = ImageOps.expand(image, (pad_w//2, pad_h//2, (pad_w+1)//2, (pad_h+1)//2), fill=0)
    w, h = image.size
    left, top = int(round((w-crop_w)/2)), int(round((h-crop_h)/2))
    return image.crop((left, top, left+crop_w, top+crop_h)).resize((224,224), Image.Resampling.BILINEAR)


class ActionHistory:
    """Initial observation plus one real observation per completed action; no time resampling."""
    def __init__(self, update_mode="action"):
        self.update_mode = update_mode
        self.events = deque(maxlen=4)
        self.total_events = 0

    def append(self, image, metadata, action_step):
        if self.events and metadata['sequence'] <= self.events[-1][1]['sequence']:
            raise RuntimeError('Post-action history requires a new camera frame')
        self.events.append((image.copy(), dict(metadata, action_step=action_step, event_index=self.total_events)))
        self.total_events += 1

    def context(self):
        if not self.events:
            raise RuntimeError('Action history has no initial frame')
        events = list(self.events)
        padded = [events[0]]*(4-len(events))+events
        return [e[0].copy() for e in padded], {
            'selection_rule': 'initial_then_post_' + self.update_mode + '_latest_four',
            'update_mode': self.update_mode,
            'timestamps_monotonic': [e[1]['timestamp_monotonic'] for e in padded],
            'sequences': [e[1]['sequence'] for e in padded],
            'action_steps': [e[1]['action_step'] for e in padded],
            'event_indices': [e[1]['event_index'] for e in padded],
            'unique_real_frames': len(events), 'left_padding_slots': 4-len(events),
            'cold_start_repetition': self.total_events == 1,
            'total_observation_events': self.total_events, 'physical_camera_pose_sync_verified': False}


def finish_observation_round(args, capture, history, step, directory):
    """Refresh real observations for a completed no-motion round, when enabled."""
    if getattr(args, 'history_update', 'action') == 'round':
        history.update_mode = 'round'
        image, metadata = capture.latest(after=time.monotonic())
        history.append(image, metadata, action_step=step)
        image.save(directory / 'round-observation.png')
        save_json(directory / 'post-round-history.json', history.context()[1])
    return history.context()


def reference_rows(episode, seconds):
    with (episode / 'aligned_pose.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Reference contains no frames')
    origin = int(rows[0]['timeline_ns'])
    rows = [r for r in rows if int(r['timeline_ns']) <= origin + seconds * 1e9]
    previous = None
    for row in rows:
        stamp = int(row['timeline_ns'])
        if row['pair_valid'] != 'True' or not all(math.isfinite(float(row[k])) for k in ('x', 'y', 'yaw')):
            raise ValueError('Invalid reference pose at frame ' + row['frame_index'])
        if previous is not None:
            if not 0 < stamp - int(previous['timeline_ns']) <= 500_000_000:
                raise ValueError('Reference has a timeline gap/reset; use a continuous episode')
            if row.get('clock_epoch') != previous.get('clock_epoch'):
                raise ValueError('Reference odometry epoch changed')
        previous = row
    if len(rows) < 2 or (int(rows[-1]['timeline_ns']) - origin) / 1e9 < seconds - .2:
        raise ValueError('Reference does not cover the requested duration')
    return rows


def select_targets(rows, rule):
    """Strict AND rule, or explicitly requested motion-wise coverage segmentation."""
    chosen = [0]
    reasons = ['start']
    for index in range(1, len(rows)):
        pose = relative_pose(rows[chosen[-1]], rows[index])
        distance, angle = math.hypot(*pose[:2]), abs(math.degrees(pose[2]))
        eps = 1e-9  # Include exact boundaries after floating-point yaw wrapping.
        strict = .25-eps <= distance <= .5+eps and 15-eps <= angle <= 30+eps
        coverage = (distance >= .25-eps or angle >= 15-eps) and distance <= .5+eps and angle <= 30+eps
        if (strict if rule == 'strict' else coverage):
            chosen.append(index)
            reasons.append('distance_and_angle' if strict else 'translation_or_rotation')
        elif rule != 'strict' and (distance > .5 or angle > 30):
            raise ValueError('Reference jumps over segmentation bounds')
    return list(zip(chosen, reasons))


def select_timed_targets(rows, interval):
    """Take the first available frame at/after each fixed timeline tick."""
    origin = int(rows[0]['timeline_ns'])
    chosen = [(0, 'start')]
    tick = 1
    for i, row in enumerate(rows[1:], 1):
        elapsed = (int(row['timeline_ns']) - origin) / 1e9
        if elapsed + 1e-9 >= tick * interval:
            chosen.append((i, 'fixed_time_interval'))
            tick = math.floor((elapsed + 1e-9) / interval) + 1
    return chosen


def target_window(center, count, size):
    # Never include targets before the current index, including at the end.
    first = max(0, center)
    return list(range(first, min(first + size, count)))


def prepare_reference(args):
    from PIL import Image, ImageDraw, ImageFont
    rows = reference_rows(args.reference, args.reference_seconds)
    selected = (select_timed_targets(rows, args.target_interval) if getattr(args, 'window_navigation', False)
                else select_targets(rows, args.target_rule))
    if len(selected) < 2:
        raise ValueError('No target satisfies the selected rule')
    directory = args.output / 'reference'
    directory.mkdir()
    targets = []
    for number, (index, reason) in enumerate(selected):
        row = rows[index]
        source = (args.reference / row['file']).resolve()
        if args.reference not in source.parents:
            raise ValueError('Reference image path escapes episode')
        with Image.open(source) as image:
            image.convert('RGB').save(directory / ('%03d.png' % number))
        previous = rows[selected[max(number - 1, 0)][0]]
        targets.append({'number': number, 'frame_index': int(row['frame_index']),
                        'elapsed_seconds': (int(row['timeline_ns']) - int(rows[0]['timeline_ns'])) / 1e9,
                        'file': 'reference/%03d.png' % number, 'source_file': row['file'],
                        'source_sha256': digest(source), 'pose': {k: float(row[k]) for k in ('x', 'y', 'yaw')},
                        'relative_to_previous_m_rad': relative_pose(previous, row), 'reason': reason})
    last = selected[-1][0]
    plan = {'episode': str(args.reference), 'reference_seconds': args.reference_seconds,
            'rule': 'fixed_time_interval' if getattr(args, 'window_navigation', False) else args.target_rule,
            'target_interval_s': getattr(args, 'target_interval', None), 'source_csv_sha256': digest(args.reference / 'aligned_pose.csv'),
            'targets': targets, 'target_count': len(targets) - 1,
            'covered_until_seconds': targets[-1]['elapsed_seconds'],
            'uncovered_tail_seconds': (int(rows[-1]['timeline_ns']) - int(rows[last]['timeline_ns'])) / 1e9,
            'pose_source': 'local robot odometry; not external ground truth'}
    save_json(args.output / 'reference.json', plan)
    width, tile_height = 960, 220
    canvas = Image.new('RGB', (width, tile_height * math.ceil(len(targets) / 4)), '#14181e')
    draw = ImageDraw.Draw(canvas)
    for i, target in enumerate(targets):
        x, y = i % 4 * 240, i // 4 * tile_height
        with Image.open(args.output / target['file']) as im:
            im.thumbnail((232, 160))
            canvas.paste(im, (x + 4, y + 48))
        p = target['relative_to_previous_m_rad']
        draw.text((x + 4, y + 4), '#%d t=%.2fs\nd=%.3fm yaw=%+.1f deg' %
                  (i, target['elapsed_seconds'], math.hypot(*p[:2]), math.degrees(p[2])), fill='white')
    canvas.save(args.output / 'reference.png')
    LOG.debug('Reference: %d targets, rule=%s, covered %.2fs, remaining %.2fs',
             plan['target_count'], plan['rule'], plan['covered_until_seconds'], plan['uncovered_tail_seconds'])
    return plan


class OnlineScale:
    """Bounded zero-intercept least squares, XY isotropic + yaw; no claim of metric model calibration."""
    def __init__(self):
        self.xy, self.yaw = .30, 1.0
        self.observations = []
        self.history = deque(maxlen=8)

    def predict(self, candidate):
        return [self.xy * candidate[0], self.xy * candidate[1], self.yaw * candidate[2]]

    def stats(self):
        valid = [v for v in self.observations if v['fit_eligible']][-60:]
        recent = valid[-8:]
        translation = [v for v in valid if math.hypot(*v['effective_candidate'][:2]) > .1]
        rotation = [v for v in valid if abs(v['effective_candidate'][2]) > .1]
        xy_rmse = math.sqrt(sum(v['xy_error_m'] ** 2 for v in recent) / len(recent)) if recent else None
        yaw_rmse = math.sqrt(sum(v['yaw_error_rad'] ** 2 for v in recent) / len(recent)) if recent else None
        stable = len(self.history) == 8 and all(
            max(v[k] for v in self.history) - min(v[k] for v in self.history) <= .1 * self.history[-1][k]
            for k in (0, 1))
        ready = (len(valid) >= 12 and len(translation) >= 8 and len(rotation) >= 8 and stable
                 and any(v['effective_candidate'][2] > .1 for v in rotation)
                 and any(v['effective_candidate'][2] < -.1 for v in rotation)
                 and xy_rmse <= .08 and yaw_rmse <= math.radians(5)
                 and all(v['xy_error_m'] <= .10 and v['yaw_error_rad'] <= math.radians(8) for v in recent)
                 and all(v['fit_eligible'] for v in self.observations[-8:]))
        return {'xy_m_per_training_unit': self.xy, 'yaw_rad_per_model_rad': self.yaw,
                'observed_actions': len(self.observations), 'eligible_samples': len(valid),
                'recent_preupdate_xy_rmse_m': xy_rmse, 'recent_preupdate_yaw_rmse_rad': yaw_rmse,
                'full_candidates_ready': bool(ready), 'semantic_metric_calibration_verified': False,
                'meaning': 'Execution-scale consistency on tested actions; not independent identification of model training units'}

    def record_failed(self, proposal, reason):
        self.observations.append({'raw_candidate': proposal['raw_candidate'],
            'effective_candidate': proposal['effective_candidate'], 'metric_target': proposal['metric_target'],
            'measured_m_rad': None, 'fit_eligible': False, 'reason': reason})

    def update(self, proposal, measured, succeeded, learn_mapping=True):
        if not all(math.isfinite(v) for v in measured):
            raise ValueError('Nonfinite measured motion')
        effective = proposal['effective_candidate']
        prediction = proposal['metric_target']
        xy_error = math.hypot(measured[0] - prediction[0], measured[1] - prediction[1])
        yaw_error = abs(wrap(measured[2] - prediction[2]))
        reasons = []
        if not succeeded: reasons.append('execution_failed')
        if xy_error > .12: reasons.append('translation_error_over_0.12m')
        if yaw_error > math.radians(10): reasons.append('yaw_error_over_10deg')
        if math.hypot(*effective[:2]) <= .1 and abs(effective[2]) <= .1:
            reasons.append('stationary_or_too_small')
        eligible = not reasons
        labels = {'execution_failed':'执行失败', 'translation_error_over_0.12m':'平移误差超过 0.12 m',
                  'yaw_error_over_10deg':'角度误差超过 10°', 'stationary_or_too_small':'静止或动作过小'}
        item = {'raw_candidate': proposal['raw_candidate'], 'effective_candidate': effective,
                'metric_target': prediction, 'measured_m_rad': measured,
                'xy_error_m': xy_error, 'yaw_error_rad': yaw_error, 'fit_eligible': eligible,
                'reason': '执行成功且误差在标定样本门槛内' if eligible else '；'.join(labels[r] for r in reasons),
                'exclusion_reasons': reasons}
        self.observations.append(item)
        valid = [v for v in self.observations if v['fit_eligible']][-60:]
        # Small fixed priors regularize sparse samples without fitting a stationary bias.
        xx, xy, aa, ay = .05, .05 * .30, .01, .01
        for v in valid:
            x, y, a = v['effective_candidate']; mx, my, ma = v['measured_m_rad']
            xx += x*x + y*y; xy += x*mx + y*my
            aa += a*a; ay += a*ma
        if eligible:
            if learn_mapping:
                self.xy = min(1., max(.05, xy / xx))
                self.yaw = min(1.5, max(.5, ay / aa))
            self.history.append((self.xy, self.yaw))
        return item


class ExecutionCompensation:
    """Per-run bounded feedforward gains, independent of model-unit conversion.

    No endpoint servo, automatic retry, or claim to identify obstacle intervention.
    Learn from successful command intervals only; hard odometry guards stay active.
    """
    def __init__(self):
        self.xy = self.yaw = 1.
        self.observations = []

    def stats(self):
        return {'translation_gain': self.xy, 'turn_gain': self.yaw,
                'gain_bounds': [.5, 2.], 'max_relative_change_per_action': .2,
                'observations': len(self.observations), 'persistence': 'current_run_only',
                'model_mapping_learning': False}

    def apply(self, proposal):
        target = proposal['metric_target']
        twist = body_twist(target)
        adjusted = [twist[0]*self.xy, twist[1]*self.xy, twist[2]*self.yaw]
        base = proposal['schedule'].copy()
        duration = max(base['duration_s'], math.hypot(*adjusted[:2])/.12,
                       abs(adjusted[2])/.35)
        proposal['uncompensated_schedule'] = base
        proposal['execution_compensation'] = self.stats().copy()
        if duration > 0:
            proposal['schedule'] = dict(zip(('vx_m_s', 'vy_m_s', 'yaw_rad_s'),
                                            [v/duration for v in adjusted]), duration_s=duration)
        # The generated condition is the intended target, not a measured outcome.
        proposal['prediction_matches_executed_condition'] = False
        proposal['prediction_condition'] = 'intended_metric_target_before_execution_compensation'
        return proposal

    def update(self, proposal, execution):
        before = self.stats()
        result = {'before': before, 'axes': {}, 'status': execution.get('status'),
                  'candidate': proposal['candidate'], 'metric_target': proposal['metric_target'],
                  'measured_m_rad': execution.get('measured_m_rad'), 'schedule': proposal['schedule']}
        measured = execution.get('measured_m_rad')
        controller = execution.get('controller') or {}
        if (execution.get('status') != 'succeeded' or measured is None
                or len(measured) != 3 or not all(math.isfinite(v) for v in measured)
                or controller.get('completion_basis') != 'elapsed_command_time'
                or any(execution.get(k) for k in ('error', 'budget_error', 'feedback_error', 'avoidance_intervened'))):
            result['reason'] = '执行失败、反馈无效、未正常完成定时指令或已知避障干预，不更新补偿'
        else:
            desired, actual = body_twist(proposal['metric_target']), body_twist(measured)
            length = math.hypot(*desired[:2])
            along = ((actual[0]*desired[0]+actual[1]*desired[1])/length) if length > 1e-9 else 0.
            cross = abs(actual[0]*desired[1]-actual[1]*desired[0])/length if length > 1e-9 else 0.
            used = proposal['execution_compensation']
            for axis, target, progress, gain in (
                    ('xy', length, along, used['translation_gain']),
                    ('yaw', abs(desired[2]), actual[2]* (1 if desired[2] >= 0 else -1), used['turn_gain'])):
                minimum = .02 if axis == 'xy' else math.radians(2)
                if target < minimum:
                    reason = '此轴没有足够的目标动作'
                elif progress < minimum:
                    reason = '无明显进展或方向相反，不增加补偿'
                elif axis == 'xy' and cross > max(.03, .5*along):
                    reason = '横向偏差过大，不更新平移补偿'
                elif axis == 'xy' and abs(wrap(measured[2]-proposal['metric_target'][2])) > math.radians(10):
                    reason = '转向偏差过大，不更新平移补偿'
                else:
                    estimate = gain*target/progress
                    old = getattr(self, axis)
                    smoothed = .75*old + .25*estimate
                    new = min(2., max(.5, min(old*1.2, max(old*.8, smoothed))))
                    setattr(self, axis, new)
                    reason = '按实际进展更新；单次变化 ≤ 20%，倍率范围 0.5–2.0'
                result['axes'][axis] = {'target': target if axis == 'xy' else math.degrees(target),
                                       'actual_progress': progress if axis == 'xy' else math.degrees(progress),
                                       'unit': 'm' if axis == 'xy' else 'deg', 'reason': reason}
            result['reason'] = '平移和转向分别评估；不因误差超过标定门槛而丢弃欠执行样本'
        self.observations.append(result)
        result['after'] = self.stats()
        LOG.debug('独立执行补偿：平移 %.3f → %.3f，转向 %.3f → %.3f | %s | %s',
                 before['translation_gain'], self.xy, before['turn_gain'], self.yaw,
                 result['reason'], json.dumps(result['axes'], ensure_ascii=False))
        return result


def allowed_candidates(scale):
    paths = candidates()
    if scale.stats()['full_candidates_ready']:
        return paths
    return {name: path for name, path in paths.items()
            if name != 'stationary'
            and sum(math.dist(a[:2], b[:2]) for a, b in zip([[0, 0, 0]] + path[:-1], path)) <= .50001
            and abs(path[-1][2]) <= math.radians(22.5) + 1e-9}


def body_twist(target):
    """Integrated constant body velocity whose ideal SE(2) endpoint is target."""
    x, y, angle = target
    if abs(angle) < 1e-9:
        return [x, y, angle]
    a, b = math.sin(angle)/angle, (1-math.cos(angle))/angle
    determinant = a*a+b*b
    return [(a*x+b*y)/determinant, (-b*x+a*y)/determinant, angle]


def twist_endpoint(twist, fraction=1.):
    x, y, angle = [v*fraction for v in twist]
    if abs(angle) < 1e-9:
        return [x, y, angle]
    a, b = math.sin(angle)/angle, (1-math.cos(angle))/angle
    return [a*x-b*y, b*x+a*y, angle]


def timed_schedule(target):
    twist = body_twist(target)
    if max(abs(v) for v in twist) < 1e-9:
        return {'duration_s': 0., 'vx_m_s': 0., 'vy_m_s': 0., 'yaw_rad_s': 0.}
    duration = max(math.hypot(*twist[:2])/.12, abs(twist[2])/.35, 1.)
    return dict(zip(('vx_m_s', 'vy_m_s', 'yaw_rad_s'), [v/duration for v in twist]), duration_s=duration)


def make_proposal(name, path, scale, full=False):
    hard_distance, hard_angle = (.5, math.radians(50)) if full else (.25, math.radians(25))
    raw = scale.predict(path[-1])
    norm = math.hypot(*raw[:2])
    angle = math.copysign(min(max(abs(raw[2]), MIN_TURN_RAD), hard_angle-math.radians(5)), raw[2]) if abs(raw[2]) > 1e-9 else 0.
    # A constant body twist traces an arc: constrain its path length, not just its chord.
    chord_ratio = 2*math.sin(abs(angle)/2)/abs(angle) if abs(angle) > 1e-9 else 1.
    max_chord = (hard_distance-PLANNING_DISTANCE_RESERVE_M)*chord_ratio
    distance = min(max(norm, MIN_TRANSLATION_M), max_chord) if norm > 1e-9 else 0.
    target = [raw[0]*distance/norm, raw[1]*distance/norm, angle] if norm > 1e-9 else [0., 0., angle]
    twist = body_twist(target)
    effective_path = [[p[0]/scale.xy, p[1]/scale.xy, p[2]/scale.yaw]
                      for p in (twist_endpoint(twist, i/len(path)) for i in range(1, len(path)+1))]
    return {'candidate': name, 'raw_candidate': path[-1], 'effective_candidate': effective_path[-1],
            'effective_path': effective_path, 'metric_target': target,
            'xy_scale_factor': distance/norm if norm > 1e-9 else 1.,
            'yaw_scale_factor': angle/raw[2] if abs(raw[2]) > 1e-9 else 1.,
            'mapped_path_length_m': math.hypot(*twist[:2]), 'hard_distance_m': hard_distance,
            'hard_turn_rad': hard_angle, 'bootstrap': not full,
            'execution_mode': 'static_walk_timed', 'schedule': timed_schedule(target),
            'prediction_matches_executed_condition': True}


MONITOR_HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenNWM · 导航实时监看</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#10151b;color:#e9eef5}
*{box-sizing:border-box}body{margin:0;padding:24px;max-width:1600px;margin:auto}
header{display:flex;justify-content:space-between;align-items:center;gap:16px}h1{font-size:24px;margin:0}
.sub,.meta{color:#a8b6c7;font-size:13px;line-height:1.7}.sub{margin:8px 0 20px}
.badge{background:#223b37;color:#9ceacb;padding:8px 14px;border-radius:20px;font-size:13px}
.bad{background:#532c2c!important;color:#ffc4b8!important}.panels{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.card{border:1px solid #303c4c;background:#19212c;border-radius:12px;overflow:hidden;min-width:0}
.caption{padding:14px 16px;font-size:15px;min-height:55px}.caption b{display:block;margin-bottom:4px}
.image{width:100%;aspect-ratio:4/3;object-fit:contain;background:#0b1016;display:block}
.panels .image{height:min(30vh,320px);aspect-ratio:auto}.history .image{height:min(18vh,180px);aspect-ratio:auto}
.empty{opacity:.25}.history{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
h2{font-size:17px;margin:24px 0 6px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0}
.metric{padding:14px 16px;background:#19212c;border:1px solid #303c4c;border-radius:10px;line-height:1.65;font-size:14px}
.metric span{display:block;color:#a8b6c7;font-size:12px}.note{padding:12px 16px;border-left:3px solid #8acbb3;background:#172c29;font-size:13px;line-height:1.7}
@media(max-width:850px){.panels{grid-template-columns:1fr 1fr}.panels>.card:first-child{grid-column:1/-1}.history{grid-template-columns:1fr 1fr}.metrics{grid-template-columns:1fr}body{padding:14px}}
</style><header><h1 id="model-title">OpenNWM / 导航实时监看</h1><div class="badge" id="phase">正在连接…</div></header>
<p class="sub">全部画面统一为模型视图 224×224 · 实时摄像头持续刷新 · 历史按动作完成顺序积累 · 所有动作仍须在终端逐次确认</p>
<div class="panels">
<section class="card"><div class="caption"><b>实时摄像头</b><span class="meta" id="camera-status">等待视频</span></div><img class="image" id="live" src="/placeholder.svg" alt="等待摄像头画面"></section>
<section class="card"><div class="caption"><b id="goal-title">当前目标帧</b><span class="meta" id="goal-meta">等待目标</span></div><img class="image" id="goal" src="/placeholder.svg" alt="等待目标帧"></section>
<section class="card"><div class="caption"><b>选中动作的预测画面</b><span class="meta" id="prediction-meta">推理完成后显示</span></div><img class="image" id="prediction" src="/placeholder.svg" alt="等待预测"></section>
</div><div class="metrics"><div class="metric"><span>候选与阶段</span><div id="candidates">等待推理</div></div>
<div class="metric"><span>本次指令 / 执行方式</span><div id="action">尚未选择动作</div></div>
<div class="metric"><span>硬限位 / 命令下限</span><div id="limits">等待配置</div></div></div>
<h2>本轮使用的全部历史帧</h2><p class="sub" id="history-meta">尚未采样；进入推理时显示实际输入的 4 个槽位。</p>
<div class="history" id="history"></div><p class="note" id="note">本页面只读。切换目标请输入 arrived；执行请输入 MOVE；终端 q 退出。</p>
<script>
const byId=id=>document.getElementById(id);let revision=-1,liveURL=null;
for(let i=0;i<4;i++){const c=document.createElement('section');c.className='card';const t=document.createElement('div');t.className='caption';t.id='hlabel'+i;t.textContent='历史 '+(i+1)+' · 等待输入';const im=document.createElement('img');im.className='image';im.id='h'+i;im.src='/placeholder.svg';im.alt='等待历史帧 '+(i+1);c.append(t,im);byId('history').append(c)}
async function live(){try{const r=await fetch('/live.jpg?t='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error(r.status);const u=URL.createObjectURL(await r.blob());byId('live').src=u;if(liveURL)URL.revokeObjectURL(liveURL);liveURL=u}catch(e){byId('camera-status').textContent='摄像头暂不可用'}finally{setTimeout(live,150)}}
async function state(){try{const r=await fetch('/state.json',{cache:'no-store'});if(!r.ok)throw Error(r.status);const s=await r.json();byId('phase').textContent=s.phase;byId('model-title').textContent=(s.model==='nwm-b'?'NWM-B':'OpenNWM')+' / 导航实时监看';byId('phase').classList.toggle('bad',!!s.camera_error);byId('camera-status').textContent=s.camera_error||('画面年龄 '+s.camera_age_s.toFixed(2)+' s · 帧 '+s.camera_sequence);byId('camera-status').classList.toggle('bad',!!s.camera_error);
byId('goal-title').textContent='当前目标 '+s.goal_number+' / '+s.goal_total;byId('goal-meta').textContent='原视频帧 '+s.frame_index+' · '+Number(s.reference_time_s).toFixed(3)+' s';
byId('candidates').textContent=s.candidate_count==null?'等待推理':s.candidate_count+' / 7 · 提案 '+s.step+' · 固定离线标定';
byId('action').textContent=s.action_text||'尚未选择动作';byId('limits').textContent=s.limit_text||'前进/原地转向与静止共 7 个选项；名义路程 0.25/0.50 m，转角 π/12 或 π/4';
byId('note').textContent=s.note||'本页面只读。切换目标请输入 arrived；执行请输入 MOVE；终端 q 退出。';
if(revision!==s.revision){revision=s.revision;for(const key of ['goal','prediction']){const im=byId(key);if(s.assets.includes('/'+key+'.jpg'))im.src='/'+key+'.jpg?r='+revision;else im.src='/placeholder.svg'}
const h=s.history||{};byId('history-meta').textContent=h.unique_real_frames?('4 个输入槽位 · '+h.unique_real_frames+' 张不同真实帧 · '+(h.cold_start_repetition?'冷启动：同一帧重复填充':((h.update_mode==='round'?'按每轮结束顺序':'按动作完成顺序')+' · 左侧补齐 '+(h.left_padding_slots||0)+' 个槽位'))):'尚未采样';
for(let i=0;i<4;i++){const im=byId('h'+i),path='/history/'+i+'.jpg';if(s.assets.includes(path)){im.src=path+'?r='+revision;const ts=h.timestamps_monotonic||[];byId('hlabel'+i).textContent='历史 '+(i+1)+' · 帧 '+(h.sequences||[])[i]+' · '+((h.action_steps||[])[i]===0?'初始帧':(h.update_mode==='round'?'轮次 ':'动作 ')+(h.action_steps||[])[i]+' 后')}else{im.src='/placeholder.svg';byId('hlabel'+i).textContent='历史 '+(i+1)+' · 等待输入'}}byId('prediction-meta').textContent=s.selected||'推理完成后显示'}
}catch(e){byId('phase').textContent='连接已断开 / 程序已结束';byId('phase').classList.add('bad');byId('camera-status').textContent='画面已停止更新';byId('camera-status').classList.add('bad')}finally{setTimeout(state,500)}}live();state();
</script></html>'''


class LiveDisplay:
    """Read-only browser monitor; never exposes robot command endpoints."""
    def __init__(self, capture, port):
        self.capture = capture
        self.lock = threading.Lock()
        self.assets = {}
        self.data = {'revision': 0, 'phase': '初始化', 'history': {}}
        display = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                path = self.path.split('?', 1)[0]
                try:
                    if path == '/':
                        body, mime = MONITOR_HTML.encode('utf-8'), 'text/html; charset=utf-8'
                    elif path == '/placeholder.svg':
                        body = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 220">'
                                '<rect width="360" height="220" fill="#0b1016"/>'
                                '<text x="180" y="115" text-anchor="middle" fill="#7f91a5" '
                                'font-family="sans-serif" font-size="16">等待画面</text></svg>').encode('utf-8')
                        mime = 'image/svg+xml'
                    elif path == '/state.json':
                        body, mime = json.dumps(display.snapshot(), ensure_ascii=False).encode('utf-8'), 'application/json'
                    elif path == '/live.jpg':
                        with capture.lock:
                            frame = capture.frames[-1][1] if capture.frames else None
                        if frame is None:
                            self.send_error(503, 'No camera frame'); return
                        from PIL import Image
                        body, mime = display.jpeg(Image.fromarray(frame.image[:, :, ::-1])), 'image/jpeg'
                    else:
                        with display.lock:
                            body = display.assets.get(path)
                        if body is None:
                            self.send_error(404); return
                        mime = 'image/jpeg'
                    self.send_response(200)
                    self.send_header('Content-Type', mime)
                    self.send_header('Content-Length', str(len(body)))
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('X-Content-Type-Options', 'nosniff')
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        LOG.info('实时监看（只读）：http://127.0.0.1:%d；远程电脑使用 SSH -L %d:127.0.0.1:%d 转发',
                 self.port, self.port, self.port)

    @staticmethod
    def jpeg(image, prepared=False):
        image = image.convert('RGB') if prepared else model_frame(image)
        stream = image_io.BytesIO()
        image.save(stream, format='JPEG', quality=85)
        return stream.getvalue()

    def update(self, goal=None, images=None, prediction=None, clear_inputs=False, **values):
        assets = {}
        if goal is not None: assets['/goal.jpg'] = self.jpeg(goal)
        if prediction is not None: assets['/prediction.jpg'] = self.jpeg(prediction, prepared=True)
        if images is not None:
            assets.update(('/history/%d.jpg' % i, self.jpeg(im)) for i, im in enumerate(images))
        with self.lock:
            if clear_inputs:
                self.assets = {k:v for k,v in self.assets.items() if k == '/goal.jpg'}
                self.data.update(history={}, selected=None, action_text=None)
            self.assets.update(assets)
            self.data.update(values)
            self.data['revision'] += 1

    def snapshot(self):
        with self.lock:
            data = dict(self.data, assets=list(self.assets))
        with self.capture.lock:
            last = self.capture.frames[-1] if self.capture.frames else None
        age = time.monotonic()-last[0] if last else None
        data.update(camera_age_s=age, camera_sequence=last[1].sequence if last else None,
                    camera_error=self.capture.fault or self.capture.video.error or
                    ('相机缺失或画面已过期' if age is None or age > .5 else ''))
        return data

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class Capture:
    """Continuous timestamped camera recording, bounded queues, actual-frame history."""
    def __init__(self, args, io):
        from go2_capture.video import RtpVideo
        self.args, self.io = args, io
        self.lock = threading.Lock()
        self.frames = deque(maxlen=2)  # ActionHistory owns the sparse post-action frames.
        self.queue = queue.Queue(maxsize=30)
        self.closed = threading.Event()
        self.fault = ''
        self.timestamp_drops = 0
        self.consecutive_timestamp_drops = 0
        self.first_timestamp = None
        self.video = RtpVideo({'interface': args.interface,
                              'video': {'address': '230.1.1.1', 'port': 1720,
                                        'jitter_ms': 100, 'socket_buffer_bytes': 4194304}}, self.on_frame)
        self.worker = threading.Thread(target=self.record, daemon=True)
        self.worker.start()
        self.video.start()
        self.telemetry = threading.Thread(target=self.record_pose, daemon=True)
        self.telemetry.start()

    def on_frame(self, frame):
        stamp = frame.timeline_ns / 1e9
        if self.closed.is_set():
            return
        with self.lock:
            if self.frames and frame.timeline_ns <= self.frames[-1][1].timeline_ns:
                previous = self.frames[-1][1]
                backwards_ns = previous.timeline_ns - frame.timeline_ns
                self.timestamp_drops += 1
                self.consecutive_timestamp_drops += 1
                # Reception order can differ from decode order. Never invent timestamps
                # or feed these frames into history/recording; require fresh valid frames.
                fatal = backwards_ns > 500_000_000 or self.consecutive_timestamp_drops >= 5
                if fatal:
                    self.fault = 'Camera timestamps repeatedly invalid or regressed by more than 0.5 s'
                (LOG.error if fatal else LOG.warning)(
                    '丢弃相机时间戳异常帧：sequence=%s receive_ns=%s previous_receive_ns=%s '
                    'rtp=%s previous_rtp=%s ssrc=%s previous_ssrc=%s backwards_ms=%.3f '
                    'consecutive=%d total=%d fatal=%s',
                    frame.sequence, frame.timeline_ns, previous.timeline_ns, frame.rtp_timestamp,
                    previous.rtp_timestamp, frame.rtp_ssrc, previous.rtp_ssrc, backwards_ns / 1e6,
                    self.consecutive_timestamp_drops, self.timestamp_drops, fatal)
                return
            self.consecutive_timestamp_drops = 0
            self.frames.append((stamp, frame))
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            self.fault = 'Video recording queue overflow'

    def record_pose(self):
        try:
            with (self.args.output / 'trajectory.csv').open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['host_monotonic', 'source', 'x', 'y', 'yaw',
                                                            'speed', 'yaw_speed', 'received', 'error'])
                writer.writeheader()
                while not self.closed.wait(.05):
                    row = {'host_monotonic': time.monotonic()}
                    try:
                        row.update(pose_dict(self.io.read_pose()))
                    except RuntimeError as exc:
                        row['error'] = str(exc)
                    writer.writerow(row)
                    stream.flush()
        except Exception as exc:
            self.fault = 'Trajectory recording failed: ' + str(exc)

    def record(self):
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        pipeline = None
        try:
            with (self.args.output / 'video-timestamps.csv').open('x', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['frame', 'host_receive_ns', 'decoded_ns', 'recording_pts_ns', 'rtp_timestamp', 'ssrc'])
                count = 0
                while not self.closed.is_set() or not self.queue.empty():
                    try:
                        frame = self.queue.get(timeout=.1)
                    except queue.Empty:
                        continue
                    if pipeline is None:
                        h, w = frame.image.shape[:2]
                        pipeline = Gst.parse_launch(
                            'appsrc name=source format=time is-live=false block=true max-bytes=8388608 '
                            '! videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=8000 '
                            '! h264parse ! matroskamux ! filesink name=output')
                        app = pipeline.get_by_name('source')
                        app.set_property('caps', Gst.Caps.from_string(
                            'video/x-raw,format=BGR,width=%d,height=%d,framerate=15/1' % (w, h)))
                        pipeline.get_by_name('output').set_property('location', str(self.args.output / 'navigation.mkv'))
                        pipeline.set_state(Gst.State.PLAYING)
                        self.first_timestamp = frame.timeline_ns
                    data = frame.image.tobytes()
                    buf = Gst.Buffer.new_allocate(None, len(data), None)
                    buf.fill(0, data)
                    buf.pts = frame.timeline_ns - self.first_timestamp
                    buf.dts = buf.pts
                    buf.duration = int(Gst.SECOND / 15)
                    if app.emit('push-buffer', buf) != Gst.FlowReturn.OK:
                        raise RuntimeError('Video encoder rejected a frame')
                    writer.writerow([count, frame.timeline_ns, frame.decoded_ns, buf.pts,
                                     frame.rtp_timestamp, frame.rtp_ssrc])
                    stream.flush()
                    count += 1
                    error = pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
                    if error:
                        raise RuntimeError(str(error.parse_error()[0]))
                if pipeline:
                    app.emit('end-of-stream')
                    message = pipeline.get_bus().timed_pop_filtered(5 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
                    if not message or message.type != Gst.MessageType.EOS:
                        raise RuntimeError('Video finalization failed')
        except Exception as exc:
            self.fault = 'Video recording failed: ' + str(exc)
            LOG.exception(self.fault)
        finally:
            if pipeline:
                pipeline.set_state(Gst.State.NULL)

    def check(self):
        if self.fault or self.video.error:
            raise RuntimeError(self.fault or self.video.error)
        with self.lock:
            if not self.frames or time.monotonic() - self.frames[-1][0] > .5:
                raise RuntimeError('Live camera is missing/stale')
        if not self.worker.is_alive() or not self.telemetry.is_alive():
            raise RuntimeError('Recorder thread stopped')
        if __import__('shutil').disk_usage(self.args.output).free < 2 * 1024**3:
            raise RuntimeError('Recording disk has less than 2 GiB free')

    def latest(self, after=None):
        from PIL import Image
        deadline = time.monotonic()+1.
        while True:
            self.check()
            with self.lock:
                stamp, frame = self.frames[-1]
            if after is None or stamp >= after:
                image = Image.fromarray(frame.image[:, :, ::-1].copy())
                return image, {'timestamp_monotonic':stamp, 'sequence':frame.sequence,
                               'physical_camera_pose_sync_verified':False}
            if time.monotonic() >= deadline:
                raise RuntimeError('No new camera frame after action completion')
            time.sleep(.02)

    def wait_ready(self):
        until = time.monotonic() + 10
        while time.monotonic() < until:
            try:
                self.check()
                self.latest()
                return
            except RuntimeError:
                if self.fault or self.video.error:
                    raise
                time.sleep(.1)
        raise RuntimeError('No fresh camera frame within 10 seconds')

    def close(self):
        self.video.close()
        self.closed.set()
        self.worker.join(timeout=8)
        self.telemetry.join(timeout=2)
        if self.worker.is_alive():
            self.fault = 'Video finalization thread did not finish'
        LOG.debug('相机时间戳异常丢帧累计：%d', self.timestamp_drops)
        if self.fault:
            raise RuntimeError(self.fault)


class MotionReserveStop(RuntimeError):
    """End this action at the braking reserve; navigation can continue after stopping."""


class MotionBudget:
    def __init__(self, initial, proposal):
        self.initial = self.previous = initial
        self.proposal = proposal
        self.distance = self.turn = 0.

    def check(self, pose, *, stopped=False):
        # While moving reserve space for braking; after stopping enforce the hard limits.
        if pose.source > self.previous.source:
            self.distance += math.hypot(pose.x - self.previous.x, pose.y - self.previous.y)
            self.turn += abs(wrap(pose.yaw - self.previous.yaw))
            self.previous = pose
        displacement = math.hypot(pose.x - self.initial.x, pose.y - self.initial.y)
        yaw = abs(wrap(pose.yaw - self.initial.yaw))
        reserve = STOP_DISTANCE_RESERVE_M
        schedule = self.proposal.get('schedule', {})
        speed = math.hypot(schedule.get('vx_m_s', 0.), schedule.get('vy_m_s', 0.))
        if speed > .18:
            # The zero-stop forward_50 run exceeded 0.70 m after its timed
            # command ended. Reserve 12 cm at 0.30 m/s, including settling.
            reserve = max(reserve, speed * .4)
        distance_limit = self.proposal['hard_distance_m'] - (0. if stopped else reserve)
        turn_reserve = math.radians(4)
        if abs(schedule.get('yaw_rad_s', 0.)) > .45:
            turn_reserve = max(turn_reserve, abs(schedule['yaw_rad_s']) * .2)
        turn_limit = self.proposal['hard_turn_rad'] - (0. if stopped else turn_reserve)
        limit_kind = 'hard limit after stop' if stopped else 'stopping reserve'
        if max(self.distance, displacement) >= distance_limit:
            raise (RuntimeError if stopped else MotionReserveStop)('Measured path/displacement reached %s: %.4f >= %.4f m' %
                               (limit_kind, max(self.distance, displacement), distance_limit))
        if max(self.turn, yaw) >= turn_limit:
            raise (RuntimeError if stopped else MotionReserveStop)('Measured accumulated turn reached %s: %.2f >= %.2f deg' %
                               (limit_kind, math.degrees(max(self.turn, yaw)), math.degrees(turn_limit)))


class GuardedIO:
    def __init__(self, io, capture, proposal, initial):
        self.io, self.capture = io, capture
        self.budget = MotionBudget(initial, proposal)
        self.current = initial

    def read_pose(self):
        self.capture.check()
        pose = self.io.read_pose()
        self.budget.check(pose)
        self.current = pose
        return pose

    def move(self, vx, vy, yaw):
        self.capture.check()
        self.io.move(vx, vy, yaw)

    def stop(self):
        self.io.stop()

    def pause_motion(self):
        self.io.pause_motion()


class SettlingIO:
    """Monitor every coast/settling pose; escalate to emergency stop at reserve."""
    def __init__(self, io, capture, budget, record):
        self.io, self.capture, self.budget, self.record = io, capture, budget, record
        self.emergency = record.get('status') != 'succeeded'
        self.current = None

    def read_pose(self):
        self.capture.check()
        pose = self.io.read_pose()
        self.current = pose
        try:
            self.budget.check(pose, stopped=self.emergency)
        except MotionReserveStop as exc:
            self.emergency = True
            self.record.update(status='stopped_at_reserve', stop_reason=str(exc))
            controller = self.record.setdefault('controller', {})
            controller.update(status='stopped_at_reserve', completion_basis='motion_reserve_during_settling')
            LOG.warning('等待停稳期间达到刹停预留，立即强制停止：%s', exc)
            self.io.stop()
            # A large feedback jump can cross both limits in the same sample.
            try:
                self.budget.check(pose, stopped=True)
            except RuntimeError as hard:
                self.record.update(status='failed', budget_error=str(hard))
                raise
        except RuntimeError as exc:
            self.record.update(status='failed', budget_error=str(exc))
            self.io.stop()
            raise
        return pose


class NavigationLoopGuard:
    """On a repeated stationary state, select a motion candidate for this step only."""

    def __init__(self):
        self.history = []

    def observe(self, target, action, pose, step):
        # Only confirmed actions enter history; cancelled proposals cannot form a loop.
        self.history.append((target, action, pose.x, pose.y, pose.yaw, step))

    def choose(self, ranking, pose):
        first = ranking[0]
        if first['candidate'] != 'stationary':
            return first, None
        for target, action, x, y, yaw, step in reversed(self.history):
            if (target == first['target_index'] and action == first['candidate']
                    and math.hypot(pose.x-x, pose.y-y) <= .03
                    and abs(wrap(pose.yaw-yaw)) <= math.radians(5)):
                alternatives = [r for r in ranking if r['candidate'] != 'stationary']
                if not alternatives:
                    raise RuntimeError('Stationary loop: no eligible motion candidate')
                # No persistent ban: the next decision checks for a new loop normally.
                return alternatives[0], {
                    'repeated_target_index': target, 'blocked_candidate': 'stationary',
                    'previous_step': step, 'scope': 'this_selection_only',
                    'position_tolerance_m': .03, 'yaw_tolerance_deg': 5,
                }
        return first, None


class Engine:
    def __init__(self, args):
        import numpy as np
        import torch
        import lpips
        from torchvision import transforms
        self.args, self.np, self.torch = args, np, torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required')
        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        self.device = torch.device('cuda:0')
        LOG.info('加载模型 · %s · %d 步 %s', args.model, args.steps, args.sampler.upper())
        self.config, self.models = load_models(args.model, args.model_root, args.steps, self.device,
                                               sampler=args.sampler, ddim_eta=args.ddim_eta)
        if int(self.config.dataset.context_size) != 4:
            raise RuntimeError('This entry expects the released four-frame checkpoint')
        self.tokenizer = self.models[2]
        if int(self.config.dataset.image_size) != 224:
            raise RuntimeError('Monitor/model preprocessing expects the released 224x224 checkpoint')
        self.tensor_transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(self.config.dataset.mean, self.config.dataset.std, inplace=True)])
        self.transform = lambda im: self.tensor_transform(model_frame(im))
        self.history_latents = OrderedDict()
        torch.hub.set_dir(str(args.model_root / 'planning/torch/hub'))
        self.scorer = lpips.LPIPS(net='alex', model_path=str(args.model_root / 'planning/lpips/v0.1/alex.pth'),
                                 verbose=False).eval().requires_grad_(False).to(self.device)
        if getattr(args, 'matching_metric', 'lpips') == 'dinov2':
            from dino_matching import DinoMatcher
            self.dino = DinoMatcher(args.dinov2_model or args.model_root / 'planning/dinov2-small', self.device)
        LOG.info('模型就绪 · 标定位移 · 全部候选（含静止）预测 0.5 s（k=0.015625）')

    def distance(self, a, b, a_prepared=False, b_prepared=False):
        torch = self.torch
        with torch.inference_mode():
            ta = self.tensor_transform(a) if a_prepared else self.transform(a)
            tb = self.tensor_transform(b) if b_prepared else self.transform(b)
            score = self.scorer(ta[None].to(self.device), tb[None].to(self.device)).item()
        if not math.isfinite(score):
            raise RuntimeError('Invalid LPIPS score')
        return score

    def lpips_distances(self, candidates, targets):
        """Cartesian candidate/target scores in bounded GPU batches, candidate-major."""
        torch = self.torch
        count, target_count = len(candidates), len(targets)
        limit = self.args.batch_size * target_count
        scores = []
        with torch.inference_mode():
            for first in range(0, count * target_count, limit):
                pairs = torch.arange(first, min(first + limit, count * target_count), device=self.device)
                scores.append(self.scorer(candidates[pairs // target_count],
                                          targets[pairs % target_count]).flatten())
        return torch.cat(scores).reshape(count, target_count)

    def dino_target_features(self, goals):
        """Cache reference features across overlapping windows; content keys avoid stale reuse."""
        torch, np = self.torch, self.np
        if not hasattr(self, 'target_features'):
            self.target_features = OrderedDict()
        prepared = [model_frame(im) for _, im in goals]
        keys = [hashlib.sha256(im.tobytes()).hexdigest() for im in prepared]
        missing = OrderedDict((key, im) for key, im in zip(keys, prepared) if key not in self.target_features)
        if missing:
            pixels = torch.stack([torch.from_numpy(np.array(im, dtype=np.float32)).permute(2, 0, 1) / 255.
                                  for im in missing.values()]).to(self.device)
            for key, feature in zip(missing, self.dino.encode(pixels)):
                self.target_features[key] = feature.detach().clone()
        result = torch.stack([self.target_features[key] for key in keys])
        for key in keys: self.target_features.move_to_end(key)
        while len(self.target_features) > 128: self.target_features.popitem(last=False)
        return result

    def encode_history(self, images):
        prepared = [model_frame(im) for im in images]
        keys = [hashlib.sha256(im.tobytes()).hexdigest() for im in prepared]
        missing = OrderedDict((key, im) for key, im in zip(keys, prepared) if key not in self.history_latents)
        if missing:
            inputs = self.torch.stack([self.tensor_transform(im) for im in missing.values()]).to(self.device)
            with self.torch.amp.autocast('cuda', dtype=self.torch.bfloat16):
                encoded = self.tokenizer.vae.encode(inputs).latent_dist.mode() * self.tokenizer.scaling_factor
            for key, latent in zip(missing, encoded):
                self.history_latents[key] = latent.detach().clone()
        for key in keys: self.history_latents.move_to_end(key)
        result = self.torch.stack([self.history_latents[key] for key in keys])
        while len(self.history_latents) > 8: self.history_latents.popitem(last=False)
        return result, len(missing)

    def select(self, images, goal, paths, step, directory, *, window_goals=None, loop_guard=None, source_pose=None):
        from PIL import Image
        from misc import normalize_data
        from isolated_nwm_infer import model_forward_wrapper
        torch, np = self.torch, self.np
        names, scores, best = list(paths), [], None
        predictions = {}
        goals = window_goals if window_goals is not None else [(None, goal)]
        metric = getattr(self.args, 'matching_metric', 'lpips')
        began = time.monotonic()
        with torch.inference_mode():
            targets = torch.stack([self.transform(im) for _, im in goals]).to(self.device)
            current_scores = self.lpips_distances(self.transform(images[-1])[None].to(self.device), targets)[0].tolist()
            current_lpips = current_scores
            if metric == 'dinov2':
                def to_tensor(im):
                    return torch.from_numpy(np.array(im, dtype=np.float32)).permute(2, 0, 1) / 255.
                target_features = self.dino_target_features(goals)
                current_features = self.dino.encode(to_tensor(model_frame(images[-1]))[None].to(self.device))
                current_scores = self.dino.distances(current_features, target_features)[0].tolist()
            for (number, im), score in zip(goals, current_scores):
                if not math.isfinite(score):
                    raise RuntimeError('Invalid current-goal matching score')
            latent, encoded_count = self.encode_history(images)
            torch.cuda.synchronize()
            history_seconds = time.monotonic()-began
            LOG.debug('历史按所选更新模式输入；本轮新增 VAE 编码 %d 张 / 4 槽位，用时 %.3f s', encoded_count, history_seconds)
            for first in range(0, len(names), self.args.batch_size):
                batch_names = names[first:first + self.args.batch_size]
                values = np.asarray([paths[name][-1] for name in batch_names], dtype=np.float32)
                if self.args.model in ('nwm-b', 'nwm-dmd'):
                    stats = {k: np.asarray(v) for k, v in self.config.dataset.action_stats.items()}
                    values[:, :2] = normalize_data(values[:, :2], stats)
                context = latent[None].repeat(len(batch_names), 1, 1, 1, 1)
                output = model_forward_wrapper(self.models, context, torch.as_tensor(values[:, None], device=self.device),
                    None, int(self.config.model.generator.input_size), self.device, num_cond=4,
                    rel_t=torch.full((len(batch_names),), NORMALIZED_K, device=self.device),
                    skip_tokenizer=True, motion_type='real', sample_keys=[step] * len(batch_names),
                    noise_seed=self.args.seed, noise_stream='go2/navigation', progress=False)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    rgb = self.tokenizer.decode(output, denormalize=True).float().clamp(0, 1)
                if tuple(rgb.shape) != (len(batch_names), 3, 224, 224) or not torch.isfinite(rgb).all():
                    raise RuntimeError('Invalid prediction shape or nonfinite pixels')
                for name, pixels in zip(batch_names, rgb):
                    predictions[name] = (pixels.cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
                # Generate/decode each action only once; reuse it for every target.
                pair_scores = (self.dino.distances(self.dino.encode(rgb), target_features)
                               if metric == 'dinov2' else self.lpips_distances(rgb * 2 - 1, targets)).tolist()
                for target_pos, (number, _) in enumerate(goals):
                    batch_scores = [row[target_pos] for row in pair_scores]
                    for name, score, pixels in zip(batch_names, batch_scores, rgb):
                        if not math.isfinite(score):
                            raise RuntimeError('Nonfinite candidate score')
                        row = {'candidate': name, 'score': score, metric: score, 'score_source': 'model_prediction'}
                        if window_goals is not None: row['target_index'] = number
                        scores.append(row)
                        if best is None or score < best[0]:
                            best = (score, name, predictions[name], number)
                LOG.debug('Step %d: generated/scored %d/%d candidates; best %s distance %.5f',
                         step, first + len(batch_names), len(names), best[1], best[0])
                del context, output, rgb
            ranking = sorted(scores, key=lambda v: v['score'])
            chosen, loop_report = (loop_guard.choose(ranking, source_pose)
                                   if loop_guard is not None else (ranking[0], None))
            best = (chosen['score'], chosen['candidate'], predictions[chosen['candidate']],
                    chosen.get('target_index'))
            prediction = Image.fromarray(best[2])
        current_goal_lpips = current_lpips[[number for number, _ in goals].index(best[3])]
        return {'selected': best[1], 'selected_target_index': best[3], 'ranking': ranking, 'inference_seconds': time.monotonic() - began,
                'history_vae_encoded_images': encoded_count, 'history_preparation_seconds': history_seconds,
                'current_goal_lpips': current_goal_lpips,
                'inferred_candidate_count': len(names),
                'loop_avoidance': loop_report, 'matching_metric': metric, 'selected_score': best[0],
                'current_goal_score': current_scores[[number for number, _ in goals].index(best[3])],
                **({'selected_lpips': best[0]} if metric == 'lpips' else {'selected_dinov2': best[0]}),
                'stationary_score_source': 'model_prediction',
                'selected_image_source': 'model_prediction',
                'normalized_k': NORMALIZED_K,
                'model_horizon_seconds': MODEL_HORIZON_SECONDS,
                'model_frame_offset': MODEL_FRAME_OFFSET,
                'target_count': len(goals), 'candidate_target_pair_count': len(scores),
                'candidate_count': len(names)}, prediction


def ensure_robot_environment(args):
    if os.environ.get('OPENNWM_ROBOT_ENV') != '1':
        setups = [Path('/opt/ros/foxy/setup.bash'), Path('/home/unitree/cyclonedds_ws/install/setup.bash'), args.ros_setup]
        for path in setups:
            if not path.is_file():
                raise RuntimeError('Missing ROS setup: ' + str(path))
        command = ' && '.join('source ' + shlex.quote(str(p)) for p in setups)
        command += ' && export OPENNWM_ROBOT_ENV=1 && exec ' + ' '.join(
            shlex.quote(v) for v in [sys.executable, '-B', str(getattr(args, 'entry_script', Path(__file__).resolve()))] + sys.argv[1:])
        # JetPack 5 needs libgomp loaded before Torch/GStreamer consume static TLS.
        gomp = Path('/lib/aarch64-linux-gnu/libgomp.so.1')
        if gomp.is_file():
            os.environ['LD_PRELOAD'] = str(gomp) + (':' + os.environ['LD_PRELOAD'] if os.environ.get('LD_PRELOAD') else '')
        os.execv('/bin/bash', ['/bin/bash', '-c', command])
    # Keep venv NumPy/Torch ahead of system GI and native OpenCV packages.
    sys.path.append('/usr/lib/python3/dist-packages')
    sys.path.insert(0, str(args.runtime_root))
    os.environ.update(ROS_DOMAIN_ID='0', RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',
        CYCLONEDDS_URI='<CycloneDDS><Domain Id="any"><General><Interfaces><NetworkInterface name="%s"/></Interfaces>'
                       '<AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>' % args.interface)
    os.environ.pop('ROS_LOCALHOST_ONLY', None)


class RemoteTelemetryNotReady(RuntimeError):
    """Recoverable only while stationary; never a reason to continue motion."""


class RemotePacketMonitor:
    def __init__(self, decoder, allow_idle_remote=False):
        self.decoder = decoder
        self.allow_idle_remote = allow_idle_remote
        self.neutral_since = None
        self.last_valid = None
        self.healthy_since = None
        self.valid_count = self.invalid_count = self.invalid_streak = 0
        self.recovering = False
        self.fault = ''
        self.takeover = None

    def accept(self, packet, now, moving=False):
        try:
            remote = self.decoder(packet)
        except Exception as exc:
            self.neutral_since = None
            self.invalid_count += 1
            self.invalid_streak += 1
            self.healthy_since = None
            self.recovering = True
            if moving and not self.fault:
                self.fault = 'invalid_remote_packet_during_motion: ' + str(exc)
            if self.invalid_streak in (1, 10, 100) or self.invalid_streak % 1000 == 0:
                try:
                    data = bytes(packet)
                    detail = 'length=%d header=%s all_zero=%s' % (len(data), data[:2].hex(), not any(data))
                except Exception:
                    detail = 'unreadable packet'
                (LOG.error if moving else LOG.warning)(
                    '遥控器无效包：%s reason=%s consecutive=%d total=%d；%s',
                    detail, exc, self.invalid_streak, self.invalid_count,
                    '执行已授权，锁存故障并停止' if moving else '等待阶段丢弃，必须恢复连续有效数据后才能执行')
            return
        self.valid_count += 1
        self.invalid_streak = 0
        if self.last_valid is not None and now-self.last_valid > .5:
            self.neutral_since = None
            self.healthy_since = None
            self.recovering = True
        self.last_valid = now
        if self.healthy_since is None:
            self.healthy_since = now
        if remote['active']:
            self.neutral_since = None
        elif self.neutral_since is None:
            self.neutral_since = now
        if remote['active'] and (moving or not self.allow_idle_remote) and not self.fault:
            self.fault = 'physical_remote_takeover'
            self.takeover = {'host_monotonic': now, 'motion_armed_or_active': bool(moving),
                             'remote': dict(remote)}
            LOG.error('检测到遥控器接管，当前导航授权已失效：%s。松开后不会自动恢复；'
                      '请退出，重新对齐现场并重启。', json.dumps(self.takeover, ensure_ascii=False))
        if self.recovering and now-self.healthy_since >= .2 and not self.fault:
            self.recovering = False
            LOG.info('遥控数据已恢复：连续有效 ≥ 0.2 s，累计丢弃 %d 个无效包；不自动发起或重试动作', self.invalid_count)

    def check(self, now):
        if self.fault:
            raise RuntimeError(self.fault)
        if self.last_valid is None or now-self.last_valid > .5:
            raise RemoteTelemetryNotReady('Physical remote telemetry is missing/stale (>0.5 s since valid packet)')
        if self.healthy_since is None or now-self.healthy_since < .2:
            raise RemoteTelemetryNotReady('Waiting for 0.2 s of continuously valid remote telemetry')

    def check_neutral(self, now):
        self.check(now)
        if self.neutral_since is None or now-self.neutral_since < .5:
            raise RemoteTelemetryNotReady('松开遥控器摇杆/按键并保持中立至少 0.5 s，再重新确认')

    def stats(self):
        return {'allow_idle_remote': self.allow_idle_remote, 'neutral_since': self.neutral_since,
                'valid_packets': self.valid_count, 'invalid_packets': self.invalid_count,
                'invalid_streak': self.invalid_streak, 'recovering': self.recovering, 'fault': self.fault,
                'first_takeover': self.takeover}


def robot_adapter(execute, allow_idle_remote=False):
    from motion_control.ros_adapter import Go2ROS
    from gait_session import SessionGaitControl
    from go2_capture.core import decode_remote_packet
    from unitree_go.msg import LowState
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    class MonitoredROS(SessionGaitControl, Go2ROS):
        def __init__(self):
            self.remote_monitor = RemotePacketMonitor(decode_remote_packet, allow_idle_remote=allow_idle_remote)
            self.navigation_motion_pending = False
            self.calibration_trial_armed = False
            self.wireless_last_valid = self.wireless_neutral_since = None
            self.wireless_last_activity = None
            super().__init__(execute=execute)
            self.subs.append(self.node.create_subscription(LowState, '/lowstate', self.on_low,
                             QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)))

        def remote_protected(self):
            return (self.active or self.has_avoidance_control or self.navigation_motion_pending
                    or self.calibration_trial_armed)

        def on_remote(self, msg):
            if not allow_idle_remote:
                return super().on_remote(msg)
            with self.lock:
                now = time.monotonic()
                axes = [msg.lx, msg.ly, msg.rx, msg.ry]
                if not all(math.isfinite(v) for v in axes):
                    self.wireless_last_activity = now
                    self.wireless_neutral_since = None
                    if self.remote_protected() and not self.fault:
                        self.fault = 'invalid_wireless_controller_during_calibration'
                    return
                if self.wireless_last_valid is None or now-self.wireless_last_valid > .5:
                    self.wireless_neutral_since = None
                self.wireless_last_valid = now
                active = bool(msg.keys) or max(abs(v) for v in axes) > .1
                if active:
                    self.invalidate_gait()
                    self.wireless_last_activity = now
                    self.wireless_neutral_since = None
                    if self.remote_protected() and not self.fault:
                        self.fault = 'remote_controller_takeover'
                elif self.wireless_neutral_since is None:
                    self.wireless_neutral_since = now
                # Idle calibration explicitly belongs to the operator. Do not clear
                # any existing fault and never release a latched moving takeover.

        def begin_calibration_trial(self):
            with self.lock:
                if not allow_idle_remote:
                    raise RuntimeError('Idle reposition is only enabled for calibration')
                if self.fault:
                    raise RuntimeError(self.fault)
                now = time.monotonic()
                self.remote_monitor.check_neutral(now)
                # /wirelesscontroller is intermittent on this robot. Continuous
                # neutrality/freshness comes from the decoded 500 Hz /lowstate.
                # Wireless events remain a veto, never a required heartbeat.
                if self.wireless_last_activity is not None and now-self.wireless_last_activity < .5:
                    raise RemoteTelemetryNotReady('最近 0.5 s 收到 /wirelesscontroller 活动或异常事件；松开后重新确认')
                self.calibration_trial_armed = True

        def end_calibration_trial(self):
            with self.lock:
                self.calibration_trial_armed = False
                if self.fault:
                    raise RuntimeError(self.fault)

        def on_low(self, msg):
            with self.lock:
                self.remote_monitor.accept(msg.wireless_remote, time.monotonic(),
                    moving=self.remote_protected())
                if self.remote_monitor.neutral_since is None:
                    self.invalidate_gait()
                if self.remote_monitor.fault and not self.fault:
                    self.fault = self.remote_monitor.fault

        def arm_navigation_motion(self):
            with self.lock:
                if self.fault:
                    raise RuntimeError(self.fault)
                self.remote_monitor.check(time.monotonic())
                self.navigation_motion_pending = True

        def disarm_navigation_motion(self):
            with self.lock:
                self.navigation_motion_pending = False

        def close(self):
            with self.lock:
                LOG.debug('遥控数据统计：%s', json.dumps(self.remote_monitor.stats(), ensure_ascii=False))
            super().close()

        def walking_readiness(self):
            try:
                super().read_pose()
                return {'ready': True}
            except RuntimeError as exc:
                return {'ready': False, 'reason': str(exc)}

        def read_pose(self):
            if execute:
                pose = super().read_pose()
            else:
                with self.lock:
                    if self.fault:
                        raise RuntimeError(self.fault)
                    pose = self.pose
                    if pose is None or time.monotonic() - pose.received > .25:
                        raise RuntimeError('Read-only odometry is missing/stale')
                    if not all(math.isfinite(v) for v in (pose.x, pose.y, pose.yaw, pose.speed, pose.yaw_speed)):
                        raise RuntimeError('Nonfinite read-only odometry')
            with self.lock:
                self.remote_monitor.check(time.monotonic())
            return pose

    return MonitoredROS()


def decision_image(images, goal, prediction, proposal, path):
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (672, 278), '#14181e')
    d = ImageDraw.Draw(canvas)
    for i, (label, image) in enumerate((('CURRENT', images[-1]), ('REFERENCE GOAL', goal),
                                       ('CURRENT (STOP)' if proposal.get('candidate') == 'stationary' else 'PREDICTED', prediction))):
        im = image.copy()
        im = model_frame(image) if i < 2 else image.resize((224,224))
        canvas.paste(im, (224*i, 26))
        d.text((224*i+5, 6), label, fill='white')
    d.text((5, 255), '%s | target: %.3f, %.3f m, %+.1f deg' %
           (proposal['candidate'], *proposal['metric_target'][:2], math.degrees(proposal['metric_target'][2])), fill='white')
    canvas.save(path)


def stopped_pose(io, timeout=2., *, stable_seconds=.4, min_wait=0., position_tolerance=None):
    began = time.monotonic()
    deadline = began + timeout
    stable = anchor = None
    remote_error = None
    while time.monotonic() < deadline:
        try:
            p = io.read_pose()
        except RemoteTelemetryNotReady as exc:
            # Only used before/after motion, never in the active command loop.
            stable = anchor = None
            remote_error = exc
            time.sleep(.02)
            continue
        remote_error = None
        if p.speed < .025 and abs(p.yaw_speed) < .04:
            now = time.monotonic()
            if stable is None:
                stable, anchor = now, p
            if position_tolerance is not None and (math.hypot(p.x-anchor.x, p.y-anchor.y) > position_tolerance
                                                   or abs(wrap(p.yaw-anchor.yaw)) > math.radians(1)):
                stable, anchor = now, p
            if now - stable >= stable_seconds and now - began >= min_wait:
                return p
        else:
            stable = anchor = None
        time.sleep(.02)
    if remote_error is not None:
        raise RemoteTelemetryNotReady('Remote telemetry did not recover within %.1f s: %s' % (timeout, remote_error))
    raise RuntimeError('Robot did not become stationary')


class StaleProposal(RuntimeError):
    """An unexecuted proposal no longer matches its observation."""


def assert_source_unchanged(source, current):
    delta = relative_pose(pose_dict(source), pose_dict(current))
    if not all(math.isfinite(v) for v in delta):
        raise RuntimeError('Invalid pose delta before execution')
    distance, angle = math.hypot(*delta[:2]), abs(delta[2])
    if distance > .03 or angle > math.radians(5):
        raise StaleProposal('Robot moved since inference: %.4f m / %.2f deg '
                            '(limits 0.0300 m / 5.00 deg); discard the proposal' %
                            (distance, math.degrees(angle)))


def confirm_after_action(capture, io):
    while True:
        capture.check()
        io.read_pose()
        answer = input('  › 本次动作已结束 · arrived 到达 / 回车继续 / q 退出：').strip().casefold()
        LOG.debug('Post-action response: %s', answer)
        if answer == 'q':
            return answer
        capture.check()
        io.read_pose()
        if answer in ('arrived', '', 'continue'):
            return answer or 'continue'
        LOG.info('请输入 arrived、回车/continue 或 q；尚未开始下一次推理。')


def recover_stale_proposal(capture, io, display, directory, source, current, reason):
    """Never reuse a rejected MOVE; reset temporal context only on explicit replan."""
    record = {'status': 'discarded_no_motion', 'reason': str(reason),
              'source_pose': pose_dict(source), 'current_pose': pose_dict(current),
              'delta_m_rad': relative_pose(pose_dict(source), pose_dict(current)),
              'translation_limit_m': .03, 'yaw_limit_rad': math.radians(5)}
    save_json(directory / 'proposal-rejection.json', record)
    LOG.warning('本次提案已作废，未执行运动：%s', reason)
    if display:
        display.update(phase='提案已作废，等待 replan / q', selected=None, action_text='未执行运动')
    while True:
        capture.check()
        io.read_pose()
        answer = input('输入 replan 从当前位置重新采集初始帧并规划（重置模型历史，保留执行记录）；q 停止：').strip().casefold()
        LOG.debug('Recovery response: %s', answer)
        if answer == 'q':
            record['resolution'] = 'stopped_by_operator'
            save_json(directory / 'proposal-rejection.json', record)
            return None
        if answer != 'replan':
            LOG.info('旧 MOVE 已失效；请输入 replan 或 q。')
            continue
        current = stopped_pose(io)
        image, metadata = capture.latest(after=time.monotonic())
        # A fresh context starts at this observation; prior execution files stay intact.
        history = ActionHistory()
        history.append(image, metadata, action_step=0)
        record.update(resolution='operator_replan_new_context',
                      recovery_pose=pose_dict(current), recovery_frame=metadata)
        save_json(directory / 'proposal-rejection.json', record)
        LOG.info('已从当前位置重建模型历史；重新推理后仍需再次输入 MOVE 确认。')
        return current, history


def automatic_replan_context(capture, io, engine, directory):
    """Stay stopped and require fresh, stable observations before discarding history."""
    io.stop()
    anchor_pose = stopped_pose(io)
    anchor, metadata = capture.latest(after=time.monotonic())
    anchor_time = metadata['timestamp_monotonic']
    previous_sequence = metadata['sequence']
    deadline = time.monotonic() + 10.
    while time.monotonic() < deadline:
        time.sleep(.2)
        capture.check()
        current = io.read_pose()  # Remote takeover/faults still propagate immediately.
        image, metadata = capture.latest(after=time.monotonic())
        if metadata['sequence'] <= previous_sequence:
            raise RuntimeError('Recovery requires fresh camera frames')
        previous_sequence = metadata['sequence']
        score = engine.distance(anchor, image)
        if not math.isfinite(score):
            raise RuntimeError('Invalid recovery scene distance')
        stationary = current.speed < .025 and abs(current.yaw_speed) < .04
        try:
            assert_source_unchanged(anchor_pose, current)
        except StaleProposal:
            stationary = False
        if not stationary or score > .05:
            anchor, anchor_pose = image, current
            anchor_time = metadata['timestamp_monotonic']
            continue
        if metadata['timestamp_monotonic'] - anchor_time >= .4:
            history = ActionHistory()
            history.append(image, metadata, action_step=0)
            save_json(directory / 'recovery.json', {'resolution': 'automatic_replan_new_context',
                'pose': pose_dict(current), 'frame': metadata, 'stable_seconds': metadata['timestamp_monotonic'] - anchor_time,
                'anchor_lpips': score, 'stability_lpips_limit': .05})
            return current, history
    raise RuntimeError('自动重规划停止：10 秒内未采集到稳定画面')


def confirm_start(capture, automatic=False):
    while True:
        capture.check()
        prompt = ('确认现场已对齐参考起点、机器人可行走，输入 START/start 启动自动导航（将自动运动）；q 退出：'
                  if automatic else '确认现场已对齐参考起点、机器人可行走，输入 START/start 开始选点（不移动）；q 退出：')
        answer = input(prompt).strip()
        LOG.debug('Startup response: %s', answer)
        if answer.casefold() == 'start':
            capture.check()
            return True
        if answer.casefold() == 'q':
            return False
        LOG.info('未开始，也未执行运动。请输入 START/start，或 q 退出。')


def execute_schedule(guarded, schedule, clock=time.monotonic, sleep=time.sleep):
    """Timed velocity commands. Pose feedback guards limits; it never closes endpoint error."""
    duration = schedule['duration_s']
    velocity = [schedule[k] for k in ('vx_m_s', 'vy_m_s', 'yaw_rad_s')]
    if not all(math.isfinite(v) for v in [duration]+velocity) or not 0 < duration <= 15:
        raise ValueError('Invalid timed motion schedule')
    from calibrated_actions import translation_speed_limit, angular_speed_limit
    if math.hypot(*velocity[:2]) > translation_speed_limit(*velocity) + 1e-6 or abs(velocity[2]) > angular_speed_limit(*velocity[:2]) + 1e-6:
        raise ValueError('Timed velocity exceeds configured limits')
    began = previous = clock()
    count = 0
    completed = False
    try:
        while True:
            now = clock()
            if now-previous > .2:
                raise RuntimeError('Timed command loop delayed; stop rather than extend motion')
            if now-began >= duration:
                break
            previous = now
            guarded.read_pose()
            # Include guard/telemetry time in the duration; never send after the deadline.
            remaining = duration-(clock()-began)
            if remaining <= 0:
                break
            guarded.move(*velocity)
            count += 1
            sleep(min(.05, remaining))
        completed = True
        return {'status': 'succeeded', 'completion_basis': 'elapsed_command_time',
                'waypoint_verified': False, 'elapsed_s': clock()-began, 'command_count': count}
    except MotionReserveStop as exc:
        return {'status': 'stopped_at_reserve', 'completion_basis': 'motion_reserve',
                'stop_reason': str(exc), 'waypoint_verified': False,
                'elapsed_s': clock()-began, 'command_count': count}
    finally:
        if completed:
            try:
                guarded.pause_motion()
            except BaseException:
                guarded.stop()
                raise
        else:
            guarded.stop()


def do_motion(args, io, capture, proposal, directory):
    gait_policy = proposal.get('gait_policy', 'static_walk')
    if gait_policy not in ('static_walk', 'preserve_current'):
        raise ValueError('Unknown gait policy: ' + str(gait_policy))
    initial = io.read_pose()
    guarded = GuardedIO(io, capture, proposal, initial)
    record = {'status': 'starting', 'proposal': proposal, 'start_pose': pose_dict(initial),
              'execution_mode': gait_policy + '_timed', 'gait_policy': gait_policy, 'waypoint_verified': False,
              'confirmation': ('automatic after terminal START' if getattr(args, 'automatic', False)
                               else 'explicit terminal MOVE token'), 'started_monotonic': time.monotonic()}
    save_json(directory / 'execution.json', record)
    try:
        # These are the first control side effects, strictly AFTER terminal confirmation.
        capture.check()
        io.arm_navigation_motion()
        if gait_policy == 'static_walk':
            switched = io.static_walk() is not False
            if switched:
                LOG.info('本会话请求小碎步；等待机身稳定后执行速度指令')
                ready = stopped_pose(guarded, timeout=6., stable_seconds=.8,
                                     min_wait=2., position_tolerance=.005)
                io.confirm_static_gait()
                record['gait_ready_pose'] = pose_dict(ready)
            else:
                LOG.info('沿用本会话小碎步，不重复切换')
            record['gait_switch_sent'] = switched
            record['gait_transition_policy'] = 'session_static_walk_v2'
        else:
            time.sleep(.5)
        guarded.read_pose()
        io.enable_avoidance()
        LOG.debug('定时执行（%s）：vx=%+.3f m/s vy=%+.3f m/s 转速=%+.2f°/s，持续 %.3f s；不检验 waypoint 到达',
                 gait_policy, proposal['schedule']['vx_m_s'], proposal['schedule']['vy_m_s'],
                 math.degrees(proposal['schedule']['yaw_rad_s']), proposal['schedule']['duration_s'])
        result = execute_schedule(guarded, proposal['schedule'])
        record['controller'] = result
        record['status'] = result['status']
        if result['status'] == 'stopped_at_reserve':
            record['stop_reason'] = result['stop_reason']
            LOG.warning('本次动作达到刹停预留阈值，停止本次动作并保留导航流程：%s', result['stop_reason'])
    except MotionReserveStop as exc:
        record.update(status='stopped_at_reserve', stop_reason=str(exc))
        LOG.warning('动作准备阶段达到刹停预留阈值，停止本次动作：%s', exc)
    except BaseException as exc:
        record.update(status='failed', error=str(exc) or type(exc).__name__)
        raise
    finally:
        settling = SettlingIO(io, capture, guarded.budget, record)
        try:
            if record['status'] == 'succeeded':
                io.pause_motion()
            else:
                io.stop()
            if proposal.get('settling_policy') == 'pose_stable_2s_v1':
                final = stopped_pose(settling, timeout=6., stable_seconds=.8, min_wait=2., position_tolerance=.005)
            else:
                final = stopped_pose(settling)
            record['end_pose'] = pose_dict(final)
            record['measured_m_rad'] = relative_pose(record['start_pose'], record['end_pose'])
            try:
                guarded.budget.check(final, stopped=True)
            except RuntimeError as exc:
                record.update(status='failed', budget_error=str(exc))
            record['measured_path_m'] = guarded.budget.distance
            record['measured_accumulated_turn_rad'] = guarded.budget.turn
            if record['status'] == 'succeeded':
                io.release_motion_control()
                record['stop_policy'] = 'zero_then_release_after_settle_v1'
            else:
                io.stop()
        except Exception as exc:
            record.update(status='failed', feedback_error=str(exc))
            io.stop()
        except BaseException:
            io.stop()
            raise
        finally:
            record['measured_path_m'] = guarded.budget.distance
            record['measured_accumulated_turn_rad'] = guarded.budget.turn
            if 'end_pose' not in record and settling.current is not None:
                record['end_pose'] = pose_dict(settling.current)
                record['measured_m_rad'] = relative_pose(record['start_pose'], record['end_pose'])
            io.disarm_navigation_motion()
        if record['status'] == 'failed':
            LOG.error('动作失败详情：error=%s | budget_error=%s | feedback_error=%s',
                      record.get('error'), record.get('budget_error'), record.get('feedback_error'))
        record['finished_monotonic'] = time.monotonic()
        save_json(directory / 'execution.json', record)
    return record


def run_robot(args, plan, status):
    import rclpy
    from PIL import Image
    automatic = getattr(args, 'automatic', False)
    rclpy.init()
    io = capture = display = None
    lock = None
    try:
        if args.mode == 'navigate':
            lock = os.open(str(Path(tempfile.gettempdir()) / ('go2-motion-%d.lock' % os.getuid())),
                           os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        io = robot_adapter(execute=args.mode == 'navigate')
        io.wait_ready(seconds=10)
        capture = Capture(args, io)
        capture.wait_ready()
        status['walking_readiness'] = io.walking_readiness()
        status['avoidance_service_subscribers'] = int(io.node.count_subscribers('/api/obstacles_avoid/request'))
        LOG.debug('Pose, camera and physical-remote telemetry ready; walking readiness: %s', status['walking_readiness'])
        if args.mode == 'probe':
            time.sleep(2)
            capture.check()
            io.read_pose()
            capture.latest()[0].save(args.output / 'camera.png')
            status['status'] = 'probe_passed_no_motion' if status['walking_readiness']['ready'] else 'sensors_ready_walking_blocked'
            return
        if getattr(args, 'display_port', 8765):
            display = LiveDisplay(capture, args.display_port)
            target = plan['targets'][1]
            with Image.open(args.output / target['file']) as image:
                display.update(goal=image, model=args.model, goal_number=1, goal_total=len(plan['targets'])-1,
                               frame_index=target['frame_index'], reference_time_s=target['elapsed_seconds'],
                               phase='加载模型')
            status['monitor_url'] = 'http://127.0.0.1:%d' % display.port
        engine = Engine(args)
        calibration = load_calibration(args.action_calibration)
        fixed_proposals = calibrated_proposals(calibration)
        save_json(args.output / 'action-calibration.json', calibration)
        LOG.info('标定就绪 · %d 个动作 · %d 个样本 · %s', len(calibration['actions']),
                 sum(len(v['samples']) for v in calibration['actions'].values()), args.action_calibration.name)
        if args.mode == 'navigate':
            LOG.info('起点参考 · %s', args.output / plan['targets'][0]['file'])
            LOG.info('请保持机器人站立并对齐起点；START 后自动连续导航，静止胜出切换目标，Ctrl+C 停止。'
                     if automatic else '请保持机器人站立并对齐起点；每步输入 MOVE 确认，arrived 确认到达，q 退出。')
            if display: display.update(phase='等待 START / start')
            if not confirm_start(capture, automatic=automatic):
                status['status'] = 'cancelled_before_navigation'
                return
        source_pose = stopped_pose(io)
        action_history = ActionHistory(getattr(args, 'history_update', 'action'))
        initial_image, initial_metadata = capture.latest()
        action_history.append(initial_image, initial_metadata, action_step=0)
        images, history = action_history.context()
        turn_guard = TurnReversalGuard()
        loop_guard = NavigationLoopGuard() if getattr(args, 'window_navigation', False) else None
        index, step = 1, 0
        stale_rejections = 0
        while index < len(plan['targets']) and step < args.max_actions:
            capture.check()
            proposal_index = index
            target = plan['targets'][index]
            with Image.open(args.output / target['file']) as f:
                goal = f.convert('RGB')
            step += 1
            directory = args.output / ('step-%03d' % step)
            directory.mkdir()
            save_json(directory / 'history.json', history)
            proposals, constraint_report = turn_guard.eligible(fixed_proposals)
            save_json(directory / 'candidate-constraints.json', constraint_report)
            if constraint_report['blocked']:
                previous = constraint_report['state']['last_turn']
                LOG.info('约束 · 第 %03d 步实测%s转 %.1f°；排除 %d 个原地%s转候选（弧线仍允许）。静止不解除，累计实测平移 %.2f/%.2f m 后解除',
                         previous['step'], '左' if previous['direction'] > 0 else '右',
                         abs(math.degrees(previous['measured_yaw_rad'])), len(constraint_report['blocked']),
                         '右' if previous['direction'] > 0 else '左', turn_guard.translation_progress_m,
                         turn_guard.release_distance_m)
                LOG.debug('排除候选：%s', ', '.join(constraint_report['blocked']))
            paths = {name: p['effective_path'] for name, p in proposals.items()}
            save_json(directory / 'candidate-proposals.json', proposals)
            if display:
                display.update(goal=goal, images=images, history=history, clear_inputs=True,
                               goal_number=index, goal_total=len(plan['targets'])-1,
                               frame_index=target['frame_index'], reference_time_s=target['elapsed_seconds'],
                               phase='%d 个候选推理（含静止）' % len(paths),
                               step=step, candidate_count=len(paths),
                               limit_text='前进/原地转向与静止共7个选项；名义路程 0.25/0.50 m；硬限位 1.00 m / π/3（60°）')
            LOG.info('── 第 %03d 步 · 目标 %d/%d · 评估 %d/7 个动作 ──',
                     step, index, len(plan['targets'])-1, len(paths))
            window_goals = None
            if getattr(args, 'window_navigation', False):
                window_goals = []
                for number in target_window(index, len(plan['targets']), args.target_window):
                    with Image.open(args.output / plan['targets'][number]['file']) as im:
                        window_goals.append((number, im.convert('RGB')))
                save_json(directory / 'target-window.json', {'center': index, 'indices': [n for n, _ in window_goals]})
            selection, prediction = engine.select(images, goal, paths, step, directory,
                **({'window_goals': window_goals, 'loop_guard': loop_guard,
                    'source_pose': source_pose} if window_goals is not None else {}))
            if selection.get('loop_avoidance'):
                event = selection['loop_avoidance']
                LOG.info('循环 · 首次重复第 %03d 步目标 %d 的静止状态；本轮排除全部 stationary，重选目标 %d + %s；下轮恢复静止候选',
                         event['previous_step'], event['repeated_target_index'],
                         selection['selected_target_index'], selection['selected'])
            if window_goals is not None:
                previous_index = index
                index = selection['selected_target_index']
                if index not in [n for n, _ in window_goals]:
                    raise RuntimeError('Selected target is outside the target window')
                target = plan['targets'][index]
                goal = next(im for n, im in window_goals if n == index)
                LOG.info('窗口 · 目标 %d → %d · 候选 %s · %s %.4f',
                         previous_index, index, selection['selected'], selection.get('matching_metric', 'lpips'),
                         selection.get('selected_score', selection.get('selected_lpips')))
                if display:
                    display.update(goal=goal, goal_number=index, frame_index=target['frame_index'],
                                   reference_time_s=target['elapsed_seconds'])
            # A remote takeover during lengthy inference invalidates this proposal.
            # Check before presenting a MOVE prompt; callback logs also cover input waits.
            io.read_pose()
            if selection['selected'] not in proposals:
                raise RuntimeError('Selected action violates candidate constraints: ' + selection['selected'])
            proposal = proposals[selection['selected']]
            selection.update(reference_target=target, proposal=proposal, history=history,
                             candidate_constraints=constraint_report,
                             source_pose=pose_dict(source_pose), calibration_sha256=digest(args.action_calibration))
            save_json(directory / 'target.json', {'target_index': index, 'reference_file': target['file'],
                'frame_index': target['frame_index']})
            save_json(directory / 'selection.json', selection)
            save_json(args.output / 'run-status.json', dict(status, current_step=step, current_goal=index))
            LOG.debug('Selected %s; nominal model action=%s; calibrated mean endpoint: dx=%.3fm dy=%.3fm yaw=%+.2fdeg; hard path<=%.2fm / turn<=%.1fdeg',
                selection['selected'], proposal['raw_candidate'], *proposal['metric_target'][:2],
                math.degrees(proposal['metric_target'][2]), proposal['hard_distance_m'], math.degrees(proposal['hard_turn_rad']))
            LOG.debug('Current-vs-goal LPIPS %.5f', selection['current_goal_lpips'])
            dx, dy, dyaw = proposal['metric_target']
            schedule = proposal['schedule']
            LOG.debug('固定候选 %s：实测平均终点 [%+.4f m, %+.4f m, %+.4fπ]；'
                     '离线观测时间 %.3fs；命令 vx=%+.3f vy=%+.3f yaw=%+.3f，持续 %.3fs；无在线学习',
                     selection['selected'], dx, dy, dyaw/math.pi, proposal['observation_horizon_s'],
                     schedule['vx_m_s'], schedule['vy_m_s'], schedule['yaw_rad_s'], schedule['duration_s'])
            LOG.info('选择 · %s | 标定位移 (%+.3f, %+.3f) m · 转角 %+.1f° · 执行 %.2f s',
                     selection['selected'], dx, dy, math.degrees(dyaw), schedule['duration_s'])
            LOG.info('距离 · 当前图像与目标 LPIPS %.4f', selection['current_goal_lpips'])
            if display:
                display.update(prediction=prediction, selected=selection['selected'],
                               phase='只读预览完成' if args.mode == 'preview' else ('自动执行前检查' if automatic else '等待 MOVE 确认'),
                               note=('自动导航：动作结束后继续推理；静止胜出切换目标；Ctrl+C 停止。' if automatic else
                                     '全部候选（含静止）使用模型预测图像评分。终端 MOVE 执行，arrived 到达，q 退出。'),
                               action_text='净距离 %.3f m / 转角 %+.1f° · 碎步 %.2f s' %
                                           (math.hypot(dx,dy),math.degrees(dyaw),schedule['duration_s']))
            if args.mode == 'preview':
                finish_observation_round(args, capture, action_history, step, directory)
                status.update(status='preview_complete_no_motion', completed_proposals=step)
                return
            prompted = time.monotonic()
            token = 'MOVE' if automatic else input('  › MOVE 执行 / arrived 到达 / q 退出：').strip()
            LOG.debug('Terminal response: %s', token)
            if token == 'q':
                status.update(status='stopped_by_operator', completed_proposals=step)
                return
            if token == 'arrived':
                save_json(directory / 'arrival.json', {'confirmed_by': 'operator', 'goal': index})
                LOG.info('到达 · 目标 %d/%d 已人工确认', index, len(plan['targets'])-1)
                index += 1
                if display and index < len(plan['targets']):
                    next_target = plan['targets'][index]
                    with Image.open(args.output / next_target['file']) as image:
                        display.update(goal=image, clear_inputs=True, goal_number=index,
                                       frame_index=next_target['frame_index'], reference_time_s=next_target['elapsed_seconds'],
                                       phase='目标已切换，采集历史')
                images, history = finish_observation_round(args, capture, action_history, step, directory)
                continue
            if token != 'MOVE':
                LOG.info('未确认动作 · 保持静止')
                images, history = finish_observation_round(args, capture, action_history, step, directory)
                continue
            capture.check()
            current = stopped_pose(io)
            try:
                if time.monotonic() - prompted > 120:
                    raise StaleProposal('Confirmation expired after 120 s')
                assert_source_unchanged(source_pose, current)
                latest, _ = capture.latest()
                scene_distance = engine.distance(images[-1], latest)
                if not math.isfinite(scene_distance):
                    raise RuntimeError('Invalid scene distance before execution')
                if scene_distance > .15:
                    raise StaleProposal('Scene changed since inference: LPIPS %.4f > 0.15' % scene_distance)
            except StaleProposal as exc:
                if automatic:
                    io.stop()
                    stale_rejections += 1
                    index = proposal_index  # A rejected match must not advance the target.
                    save_json(directory / 'proposal-rejection.json', {'reason': str(exc), 'automatic': True,
                        'status': 'discarded_no_motion', 'consecutive_rejections': stale_rejections,
                        'resolution': 'rejection_limit' if stale_rejections >= 3 else 'automatic_replan',
                        'resume_target_index': index})
                    if stale_rejections >= 3:
                        raise RuntimeError('自动导航停止：连续 3 次提案失效') from exc
                    LOG.warning('提案失效，保持停止并重新采集稳定画面（连续 %d/3）：%s', stale_rejections, exc)
                    if display:
                        display.update(phase='保持停止，采集稳定画面后重新推理', selected=None, action_text='旧动作已作废')
                    source_pose, action_history = automatic_replan_context(capture, io, engine, directory)
                    action_history.update_mode = getattr(args, 'history_update', 'action')
                    images, history = action_history.context()
                    continue
                recovered = recover_stale_proposal(capture, io, display, directory, source_pose, current, exc)
                if recovered is None:
                    status.update(status='stopped_by_operator', completed_proposals=step, targets_confirmed=index-1)
                    return
                source_pose, action_history = recovered
                action_history.update_mode = getattr(args, 'history_update', 'action')
                images, history = action_history.context()
                continue
            stale_rejections = 0
            if loop_guard is not None:
                loop_guard.observe(index, selection['selected'], source_pose, step)
            if max(abs(v) for v in proposal['metric_target']) < 1e-9:
                latest.save(directory / 'actual-after.png')
                if automatic:
                    save_json(directory / 'arrival.json', {
                        'confirmed_by': 'stationary_selected', 'goal': index,
                        'current_goal_lpips': selection['current_goal_lpips']})
                    LOG.info('到达 · 目标 %d/%d 静止胜出，%s', index, len(plan['targets'])-1,
                             '全部目标完成' if index == len(plan['targets'])-1 else '切换下一个目标')
                    index += 1
                    status.update(status='running', targets_confirmed=index-1)
                    save_json(args.output / 'run-status.json', status)
                    if display and index == len(plan['targets']):
                        display.update(phase='全部目标已完成（静止判定）')
                else:
                    LOG.info('选择静止 · 未移动；到达仍需输入 arrived 确认')
                images, history = finish_observation_round(args, capture, action_history, step, directory)
                continue
            status['motion_attempts'] = status.get('motion_attempts', 0) + 1
            if display: display.update(phase='执行固定标定指令')
            execution = do_motion(args, io, capture, proposal, directory)
            if display: display.update(phase='动作已结束，记录实测位移；不更新标定')
            if 'measured_m_rad' in execution:
                measured = execution['measured_m_rad']
                LOG.info('执行 · %s | 实测位移 (%+.3f, %+.3f) m · 转角 %+.1f°',
                         execution['status'], measured[0], measured[1], math.degrees(measured[2]))
                save_json(directory / 'calibration-deviation.json', {
                    'measured_m_rad': measured, 'offline_mean_m_rad': proposal['metric_target'],
                    'xy_error_m': math.dist(measured[:2], proposal['metric_target'][:2]),
                    'yaw_error_rad': wrap(measured[2]-proposal['metric_target'][2]),
                    'online_update': False})
            source_pose = stopped_pose(io)
            after_image, after_metadata = capture.latest(after=execution.get('finished_monotonic'))
            action_history.update_mode = getattr(args, 'history_update', 'action')
            action_history.append(after_image, after_metadata, action_step=step)
            images, history = action_history.context()
            images[-1].save(directory / 'actual-after.png')
            save_json(directory / 'post-action-history.json', history)
            if getattr(args, 'history_update', 'action') == 'round':
                save_json(directory / 'post-round-history.json', history)
            if execution['status'] not in ('succeeded', 'stopped_at_reserve'):
                reasons = '; '.join('%s: %s' % (key, execution[key])
                                    for key in ('error', 'budget_error', 'feedback_error') if execution.get(key))
                raise RuntimeError('Action did not complete; stopped rather than proceeding: ' +
                                   (reasons or 'unknown failure; inspect execution.json'))
            constraint_event = turn_guard.observe(proposal, execution, step)
            status['turn_reversal_guard'] = turn_guard.snapshot()
            save_json(directory / 'constraint-state-after-action.json',
                      {'event': constraint_event, 'state': turn_guard.snapshot()})
            if constraint_event and constraint_event['event'] == 'released_by_translation':
                LOG.info('约束解除 · 累计实测平移 %.2f m，解除原地反向转动限制',
                         constraint_event['previous']['translation_progress_m'])
            # Score the completed action against this goal before asking the operator.
            comparison = {'actual_vs_reference_goal_lpips': engine.distance(images[-1], goal)}
            save_json(directory / 'image-comparison.json', comparison)
            LOG.info('距离 · 执行后实际图像与目标 LPIPS %.4f', comparison['actual_vs_reference_goal_lpips'])
            if automatic:
                comparison['prediction_vs_actual_lpips'] = engine.distance(prediction, images[-1], a_prepared=True)
                save_json(directory / 'image-comparison.json', comparison)
                status.update(status='running', current_step=step, current_goal=index, targets_confirmed=index-1)
                save_json(args.output / 'run-status.json', status)
                if display: display.update(phase='动作已结束，继续推理')
                continue
            # Manual navigation waits for the operator before the next inference.
            if display:
                display.update(phase='动作已结束，等待到达确认 / 继续 / q')
            status.update(current_step=step, current_goal=index, targets_confirmed=index-1,
                          status='awaiting_arrival_confirmation')
            save_json(args.output / 'run-status.json', status)
            answer = confirm_after_action(capture, io)
            if answer == 'q':
                status.update(status='stopped_by_operator', completed_proposals=step)
                return
            if answer == 'arrived':
                save_json(directory / 'arrival.json', {'confirmed_by': 'operator', 'goal': index,
                                                       'phase': 'after_action'})
                LOG.info('到达 · 目标 %d/%d 已人工确认', index, len(plan['targets'])-1)
                index += 1
                status.update(status='running', targets_confirmed=index-1)
                if display and index == len(plan['targets']):
                    display.update(phase='全部目标已确认到达')
                continue
            status['status'] = 'running'
            comparison['prediction_vs_actual_lpips'] = engine.distance(prediction, images[-1], a_prepared=True)
            save_json(directory / 'image-comparison.json', comparison)
        status.update(status='all_targets_confirmed' if index == len(plan['targets']) else 'proposal_limit_reached',
                      completed_proposals=step, targets_confirmed=index-1)
    finally:
        # Stop/release BEFORE recorder teardown, even on logging or disk errors.
        try:
            if io:
                io.stop()
        finally:
            try:
                if capture:
                    primary_error = sys.exc_info()[1]
                    try:
                        try:
                            if display: display.close()
                        finally:
                            capture.close()
                    except Exception as exc:
                        status['capture_cleanup_error'] = str(exc)
                        if primary_error is None:
                            raise
                        LOG.exception('录像收尾也发生异常；保留原始停止原因：%s', primary_error)
            finally:
                try:
                    if io:
                        io.close()
                finally:
                    if lock is not None:
                        os.close(lock)
                    rclpy.shutdown()



class ConsoleFormatter(logging.Formatter):
    def format(self, record):
        # Format a copy so the detailed file handler retains traceback information.
        import copy
        compact = copy.copy(record)
        if compact.exc_info:
            compact.msg = compact.getMessage() + ' · ' + str(compact.exc_info[1])
            compact.args = ()
        compact.exc_info = compact.exc_text = compact.stack_info = None
        label = '错误 · ' if compact.levelno >= logging.ERROR else '警告 · ' if compact.levelno >= logging.WARNING else ''
        return label + super().format(compact)


def configure_logging(output, verbose=False):
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not verbose:
        console.addFilter(lambda record: record.name.startswith('go2.') or record.levelno >= logging.WARNING)
    console.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s') if verbose
                         else ConsoleFormatter('%(message)s'))
    logfile = logging.FileHandler(output / 'navigation.log', encoding='utf-8')
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    logging.basicConfig(level=logging.INFO, handlers=[console, logfile], force=True)
    LOG.setLevel(logging.DEBUG)


class Tee:
    """Also persist library stdout/stderr without recursing through the logger."""
    def __init__(self, original, handler):
        self.original, self.handler = original, handler
        self.encoding = getattr(original, 'encoding', 'utf-8')

    def write(self, text):
        self.handler.acquire()
        try:
            self.original.write(text)
            self.handler.stream.write(text)
            self.handler.stream.flush()
        finally:
            self.handler.release()
        return len(text)

    def flush(self):
        self.original.flush()
        self.handler.flush()

    def isatty(self):
        return self.original.isatty()

    def fileno(self):
        return self.original.fileno()


def main(*, automatic=False, entry_script=None, window_navigation=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'probe', 'preview', 'navigate'))
    parser.add_argument('--reference', type=Path, default=REPO.parents[1] / 'data/go2-workspace/datasets/episode_20261017T050607_8c26ec7b')
    parser.add_argument('--reference-seconds', type=float, default=30.)
    parser.add_argument('--target-rule', choices=('either', 'strict'), default='either')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=MODEL_CHOICES, default='opennwm')
    parser.add_argument('--model-root', type=Path, default=REPO.parents[1] / 'models')
    parser.add_argument('--runtime-root', type=Path, default=REPO.parent / 'go2-runtime')
    parser.add_argument('--ros-setup', type=Path, default=REPO.parents[1] / 'envs/go2-ros/install/setup.bash')
    parser.add_argument('--interface', default='eth10')
    parser.add_argument('--display-port', type=int, default=8765, help='Read-only localhost monitor port; 0 disables display')
    parser.add_argument('--steps', type=int, default=None)
    add_sampler_arguments(parser, default_steps=50)
    parser.add_argument('--batch-size', type=int, choices=(6,7,14,15), default=7,
                        help='At most 7 candidates including stationary; batch size limits each inference call')
    parser.add_argument('--history-update', choices=('action', 'round'), default='action',
                        help='action: append after motion; round: append after every round, waiting for motion when executed')
    parser.add_argument('--action-calibration', type=Path, default=Path(__file__).with_name('default-action-calibration.json'),
                        help='Validated forward/turn calibration; compatible old files retain only forward/turn actions')
    parser.add_argument('--verbose', action='store_true', help='Show detailed terminal logs; navigation.log always retains details')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--matching-metric', choices=('lpips', 'dinov2'), default='lpips',
                        help='Candidate/target matching distance; scene-change safety remains LPIPS')
    parser.add_argument('--dinov2-model', type=Path, help='Local Hugging Face DINOv2 model directory')
    parser.add_argument('--max-actions', type=int, default=50, help='Maximum proposals, including cancelled proposals')
    if window_navigation:
        parser.add_argument('--target-interval', type=float, default=.5, help='Reference sampling interval in seconds')
        parser.add_argument('--target-window', type=int, default=10, help='Number of reference frames in the current-and-future window (at least 3)')
    args = parser.parse_args()
    args.window_navigation = window_navigation
    if window_navigation and (not math.isfinite(args.target_interval) or args.target_interval <= 0
                              or args.target_window < 3):
        parser.error('Target interval must be positive; target window must be at least 3')
    args.automatic = automatic
    args.entry_script = Path(entry_script or __file__).resolve()
    validate_sampler_arguments(parser, args)
    if args.mode in ('prepare', 'probe') and (args.sampler != 'ddpm' or args.ddim_eta is not None):
        parser.error('Sampler options require preview or navigate mode')
    for name in ('reference', 'output', 'model_root', 'runtime_root', 'ros_setup'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not math.isfinite(args.reference_seconds) or args.reference_seconds <= 0 or not 2 <= args.steps <= 1000:
        parser.error('Invalid duration or diffusion steps')
    if not 0 <= args.display_port <= 65535:
        parser.error('Invalid display port')
    if not 0 <= args.seed < 2**32 or not 1 <= args.max_actions <= 500:
        parser.error('Invalid seed or proposal limit')
    if not re.fullmatch(r'[a-zA-Z0-9_.:-]+', args.interface):
        parser.error('Invalid network interface')
    if args.mode in ('preview', 'navigate'):
        if args.action_calibration is None:
            parser.error('--action-calibration is required; run jetson_calibrate_actions.py first')
        args.action_calibration = args.action_calibration.expanduser().resolve()
        load_calibration(args.action_calibration)
    if args.mode == 'navigate' and not sys.stdin.isatty():
        parser.error('navigate requires a real interactive terminal for START')
    if args.mode != 'prepare':
        if args.interface == 'lo' or not Path('/sys/class/net', args.interface).exists():
            parser.error('A real robot-facing network interface is required')
        ensure_robot_environment(args)
    args.output.mkdir(parents=True, exist_ok=False)
    configure_logging(args.output, args.verbose)
    # Route warnings and uncaught exceptions through the same terminal + file logger.
    logging.captureWarnings(True)
    handler = next(h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler))
    sys.stdout, sys.stderr = Tee(sys.stdout, handler), Tee(sys.stderr, handler)
    status = {'status': 'starting', 'mode': args.mode, 'model': args.model,
              'time_condition': 'fixed_model_horizon_0.5s', 'candidate_count': 7,
              'max_inferred_candidate_count': 7, 'stationary_score_source': 'model_prediction',
              'candidate_constraints_policy': 'no_opposite_in_place_turn_until_0.20m_measured_translation',
              'model_horizon_seconds': MODEL_HORIZON_SECONDS, 'normalized_k': NORMALIZED_K,
              'waypoint_spacing_m': None if window_navigation else .25, 'online_calibration': False,
              'target_selection': 'fixed_time_forward_window' if window_navigation else 'pose_threshold',
              'hard_distance_m': HARD_DISTANCE, 'hard_turn_rad': HARD_TURN,
              'physical_collision_free_guaranteed': False, 'motion_attempts': 0,
              'script_sha256': digest(Path(__file__)),
              'runtime_code_sha256': {str(p.relative_to(args.runtime_root)): digest(p) for p in
                  [args.runtime_root / name for name in ('motion_control/controller.py', 'motion_control/ros_adapter.py',
                                                        'go2_capture/core.py', 'go2_capture/video.py')] if p.is_file()},
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    save_json(args.output / 'run-status.json', status)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Received signal %d' % signum)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        plan = prepare_reference(args)
        if args.mode == 'prepare':
            status['status'] = 'reference_prepared_no_motion'
            return
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                          MPLBACKEND='Agg', TORCH_HOME=str(args.model_root / 'planning/torch'),
                          NWM_VAE_PATH=str(args.model_root / 'sd-vae-ft-ema'))
        sys.path.insert(0, str(REPO / 'nwm'))
        with tempfile.TemporaryDirectory(prefix='opennwm-navigation-') as temporary:
            os.environ['MPLCONFIGDIR'] = temporary
            os.environ['GST_REGISTRY'] = str(Path(temporary) / 'gstreamer-registry.bin')
            run_robot(args, plan, status)
    except KeyboardInterrupt:
        status['status'] = 'interrupted'
        LOG.warning('已中断 · 停止导航')
    except Exception as exc:
        status.update(status='failed', error=str(exc))
        LOG.exception('导航异常停止')
        raise
    finally:
        save_json(args.output / 'run-status.json', status)
        LOG.info('结束 · %s\n记录 · %s', status['status'], args.output)


if __name__ == '__main__':
    main()
