#!/usr/bin/env python3
"""Render selected Go2 clips without altering recorded or predicted pixels.

CPU runtime: conda run --no-capture-output -n base python tools/render_go2_gallery.py
The plan names each episode, evaluation directory and preselected time range.
Only clips are exported; full-route films are deliberately not produced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/file_system/nas/algorithm/dujun.nie/opennwm_go2_showcase_20261027')
SIZE = (1920, 1080)
THIRD = (1200, 675)
MODELS = ('opennwm', 'nwm')
BG, SURFACE, LINE = '#0d1113', '#171e21', '#2c373b'
TEXT, MUTED = '#f2f5f0', '#b5c0bf'
COLORS = {'gt': '#eef2ed', 'opennwm': '#b6ee65', 'nwm': '#87b8ff'}


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def public(value):
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if not k.startswith('_')}
    if isinstance(value, list):
        return [public(v) for v in value]
    if isinstance(value, str) and value.startswith('/'):
        return Path(value).name
    return value


def run(command):
    print('COMMAND', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames,duration', '-of', 'json', str(path)]))['streams'][0]


def code(seconds):
    return f'{int(seconds) // 60:02d}:{int(seconds) % 60:02d}'


def camera_frame(path, seconds, size=THIRD):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-ss', str(seconds), '-i', str(path),
        '-frames:v', '1', '-vf', f'scale={size[0]}:{size[1]}', '-threads', '2', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return Image.frombytes('RGB', size, raw)


def recorded_image(path):
    """Keep the recorded image sharp while matching the models' 4:3 crop."""
    image = Image.open(path).convert('RGB')
    w, h = image.size
    cw, ch = (int(h * 4 / 3), h) if w > h else (w, int(w / (4 / 3)))
    x, y = int(round((w - cw) / 2)), int(round((h - ch) / 2))
    return image.crop((x, y, x + cw, y + ch))


def validate_duration(spec):
    """Permit documented continuous excerpts; keep ordinary clips at 20–30 s."""
    start, end = float(spec['start_s']), float(spec['end_s'])
    original = spec.get('shortened_from')
    if original is None:
        if not 20 <= end - start <= 30:
            raise ValueError('Selected clips must last 20–30 seconds')
        return
    parent_start, parent_end = float(original['start_s']), float(original['end_s'])
    if (not original.get('id') or not 20 <= parent_end - parent_start <= 30
            or not parent_start <= start < end <= parent_end
            or not 8 <= end - start < parent_end - parent_start):
        raise ValueError('Shortened clips require an 8 s or longer continuous excerpt of a documented 20–30 s clip')


class Encoder:
    def __init__(self, path, size, fps, crf):
        self.path = path
        args = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{size[0]}x{size[1]}', '-r', str(fps), '-i', '-', '-an',
            '-c:v', 'libx264', '-threads', '4', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p',
            '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
            '-movflags', '+faststart', str(path)]
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE)
        print('ENCODER', self.process.pid, ' '.join(args), flush=True)

    def write(self, image):
        self.process.stdin.write(image.tobytes())

    def close(self):
        if not self.process.stdin.closed:
            self.process.stdin.close()
        status = self.process.wait()
        print('ENCODER_EXIT', status, self.path, flush=True)
        if status:
            raise RuntimeError(f'Encoder failed: {self.path}')


class Clip:
    def __init__(self, spec):
        self.spec = spec
        self.root = Path(spec['episode_root'])
        self.evaluation = Path(spec['evaluation_root'])
        self.alignment = load(self.root / 'alignment_manifest.json')
        self.windows = load(self.evaluation / 'windows.json')
        self.start, self.end = float(spec['start_s']), float(spec['end_s'])
        self.duration = self.end - self.start
        self.fps = self.windows['fps']
        self.horizon = self.windows['horizon']
        self.interval = self.horizon / self.fps
        self.rows = sorted([r for r in self.windows['windows'] if r['start_time_s'] >= self.start - 1e-6
            and r['end_time_s'] <= self.end + 1e-6], key=lambda r: r['start_time_s'])
        if not self.rows or not math.isclose(self.rows[0]['start_time_s'], self.start, abs_tol=1e-6):
            raise ValueError(f'Clip must start on a complete evaluation window: {spec["id"]}')
        if not math.isclose(self.rows[-1]['end_time_s'], self.end, abs_tol=1e-6):
            raise ValueError(f'Clip must end on a complete evaluation window: {spec["id"]}')
        for previous, current in zip(self.rows, self.rows[1:]):
            if not math.isclose(previous['end_time_s'], current['start_time_s'], abs_tol=1e-6):
                raise ValueError('Unevaluated gap inside selected clip')
        validate_duration(spec)
        self.results, self.protocols, self.fonts, self.cache = {}, {}, {}, {}
        self.third = Path(self.alignment['third_video'])
        self.ego = Path(self.alignment['ego_video'])
        for model in MODELS:
            self.protocols[model] = load(self.evaluation / 'outputs' / model / 'protocol.json')
        for key in ('candidates', 'elites', 'horizon', 'fps', 'goal_interval_s', 'repeats', 'updates', 'diffusion_sampler', 'diffusion_steps', 'waypoint_spacing_m'):
            if self.protocols['opennwm'].get(key) != self.protocols['nwm'].get(key):
                raise ValueError(f'Model protocol mismatch: {key}')
        if self.protocols['opennwm'].get('diffusion_sampler') != 'DDPM':
            raise ValueError('Only original DDPM model results are supported')
        for row in self.rows:
            if len(row['frames']) != self.horizon + 1:
                raise ValueError('Reference frame count differs from planning horizon')
            self.results[row['id']] = {}
            for model in MODELS:
                path = self.evaluation / 'outputs' / model / f'{row["id"]:06d}' / 'result.json'
                result = load(path)
                for metric in ('ate_m', 'rpe_m'):
                    if not math.isfinite(result['metrics'][metric]):
                        raise ValueError(f'Invalid metric in {path}')
                planned = np.asarray(result['planned_xy_m'])
                if planned.shape != (self.horizon + 1, 2) or not np.isfinite(planned).all():
                    raise ValueError(f'Invalid trajectory in {path}')
                if not np.allclose(result['reference_xy_m'], row['reference_xy_m'], atol=1e-7):
                    raise ValueError(f'Reference mismatch in {path}')
                for n in range(self.horizon):
                    if not (path.parent / 'predicted_frames' / f'{n:03d}.png').is_file():
                        raise FileNotFoundError(f'Missing prediction {path}:{n}')
                result['_directory'] = path.parent
                result['source_result_sha256'] = digest(path)
                self.results[row['id']][model] = result
        # Keep the navigation scores stable throughout this displayed clip.
        # Every complete, equal-duration local plan contributes once.
        self.clip_metrics = self.summary()

    @staticmethod
    def screen_axes(path):
        return np.column_stack((-path[:, 1], -path[:, 0]))

    def text(self, draw, xy, value, size=30, color=TEXT, anchor=None):
        if size not in self.fonts:
            font = ImageFont.truetype(str(REPO / 'website/assets/fonts/Manrope.ttf'), size=size)
            font.set_variation_by_axes([550])
            self.fonts[size] = font
        draw.text(xy, value, font=self.fonts[size], fill=color, anchor=anchor)

    def row_at(self, time):
        return next(r for r in self.rows if r['start_time_s'] - 1e-7 <= time < r['end_time_s'] - 1e-7)

    def frames(self, row):
        if row['id'] not in self.cache:
            self.cache.clear()
            observed = [recorded_image(p) for p in row['frames']]
            frames = {'gt': observed}
            for model in MODELS:
                directory = self.results[row['id']][model]['_directory']
                # The initial model view is the same observation fed to the model.
                frames[model] = [observed[0].resize((224, 224), Image.Resampling.BILINEAR)] + [
                    Image.open(directory / 'predicted_frames' / f'{i:03d}.png').convert('RGB')
                    for i in range(self.horizon)]
            self.cache[row['id']] = frames
        return self.cache[row['id']]

    def plot(self, image, row, index):
        d = ImageDraw.Draw(image)
        x, y, w, h = 1248, 48, 648, 480
        d.rectangle((x, y, x + w - 1, y + h - 1), fill=SURFACE)
        self.text(d, (x + 24, y + 16), f'{self.interval:g} s local plan', 33)
        for offset, key, label in [(24, 'gt', 'GT'), (166, 'opennwm', 'OpenNWM'), (436, 'nwm', 'NWM')]:
            d.line((x + offset, y + 86, x + offset + 20, y + 86), fill=COLORS[key], width=4)
            self.text(d, (x + offset + 30, y + 65), label, 28, COLORS[key])
        left, top, right, bottom = x + 24, y + 112, x + w - 24, y + 434
        for gx in range(left, right + 1, 60):
            d.line((gx, top, gx, bottom), fill='#253035', width=1)
        for gy in range(top, bottom + 1, 54):
            d.line((left, gy, right, gy), fill='#253035', width=1)
        origin = np.array([(left + right) / 2, (top + bottom) / 2])
        paths = {'gt': row['reference_xy_m']}
        paths.update({m: self.results[row['id']][m]['planned_xy_m'] for m in MODELS})
        all_points = np.concatenate([self.screen_axes(np.asarray(p)) for p in paths.values()])
        lo, hi = all_points.min(0), all_points.max(0)
        center = (lo + hi) / 2
        scale = min(552 / max(hi[0] - lo[0], .4), 298 / max(hi[1] - lo[1], .4))
        for key, path in paths.items():
            points = [tuple(p) for p in (self.screen_axes(np.asarray(path)) - center) * scale + origin]
            faint = {'gt': '#596260', 'opennwm': '#526840', 'nwm': '#405a78'}[key]
            d.line(points, fill=faint, width=4, joint='curve')
            if index > 0:
                d.line(points[:index + 1], fill=COLORS[key], width=6, joint='curve')
            px, py = points[index]
            d.ellipse((px - 7, py - 7, px + 7, py + 7), fill=COLORS[key], outline=SURFACE, width=2)
            if key == 'gt':
                gx, gy = points[-1]
                d.ellipse((gx - 9, gy - 9, gx + 9, gy + 9), outline=COLORS[key], width=2)
        meters = 10 ** math.floor(math.log10(max(65 / scale, .001)))
        if meters * scale < 35:
            meters *= 2
        d.line((x + 24, y + h - 22, x + 24 + meters * scale, y + h - 22), fill=MUTED, width=3)
        self.text(d, (x + 36 + meters * scale, y + h - 41), f'{meters:g} m', 25, MUTED)

    def compose(self, time, third):
        row = self.row_at(time)
        index = min(self.horizon, max(0, int(math.floor((time - row['start_time_s']) * self.fps + .5))))
        frames = self.frames(row)
        image = Image.new('RGB', SIZE, BG)
        d = ImageDraw.Draw(image)
        self.text(d, (24, 4), 'Third-person GT', 30)
        self.text(d, (1224, 7), f'{code(time - self.start)} / {code(self.duration)}', 26, MUTED, 'ra')
        image.paste(third, (24, 48))
        d.rectangle((24, 720, 1223, 723), fill=LINE)
        d.rectangle((24, 720, 24 + int(1199 * (time - self.start) / self.duration), 723), fill=COLORS['gt'])
        for x, key, label in [(24, 'gt', 'First-person GT'), (432, 'opennwm', 'OpenNWM'), (840, 'nwm', 'NWM')]:
            d.line((x, 752, x + 18, 752), fill=COLORS[key], width=4)
            self.text(d, (x + 28, 729), label, 31, COLORS[key])
            image.paste(frames[key][index].resize((384, 288), Image.Resampling.LANCZOS), (x, 774))
        self.plot(image, row, index)
        d = ImageDraw.Draw(image)
        x, y, w, h = 1248, 552, 648, 510
        d.rectangle((x, y, x + w - 1, y + h - 1), fill=SURFACE)
        self.text(d, (x + 24, y + 22), 'Visual goal', 34)
        self.text(d, (x + 24, y + 80), f'+{self.interval:g} s', 38, MUTED)
        image.paste(frames['gt'][-1].resize((240, 180), Image.Resampling.LANCZOS), (x + w - 264, y + 24))
        d.line((x + 24, y + 234, x + w - 24, y + 234), fill=LINE, width=2)
        self.text(d, (x + 24, y + 258), 'Clip mean', 28, MUTED)
        for right, label in ((x + 430, 'ATE (m)'), (x + w - 24, 'RPE (m)')):
            self.text(d, (right - 25, y + 258), label, 28, MUTED, 'ra')
            # Draw the direction marker explicitly so every font renders it clearly.
            arrow_x, arrow_y = right - 8, y + 267
            d.line((arrow_x, arrow_y, arrow_x, arrow_y + 20), fill=MUTED, width=3)
            d.line((arrow_x - 6, arrow_y + 14, arrow_x, arrow_y + 20,
                    arrow_x + 6, arrow_y + 14), fill=MUTED, width=3)
        for off, model, label in [(328, 'opennwm', 'OpenNWM'), (414, 'nwm', 'NWM')]:
            metrics = self.clip_metrics[model]
            self.text(d, (x + 24, y + off), label, 32, COLORS[model])
            self.text(d, (x + 430, y + off - 3), f'{metrics["ate_m"]:.3f}', 38, TEXT, 'ra')
            self.text(d, (x + w - 24, y + off - 3), f'{metrics["rpe_m"]:.3f}', 38, TEXT, 'ra')
        return image, frames, index

    def summary(self):
        return {model: {key: float(np.mean([self.results[row['id']][model]['metrics'][key] for row in self.rows]))
            for key in ('ate_m', 'rpe_m')} for model in MODELS}

    def report(self):
        plans = []
        for row in self.rows:
            plans.append(dict(id=row['id'], start_time_s=row['start_time_s'], end_time_s=row['end_time_s'],
                reference_xy_m=row['reference_xy_m'], reference_world_xy_m=row['reference_world_xy_m'],
                models={m: {k: self.results[row['id']][m][k] for k in
                    ('metrics', 'planned_xy_m', 'planned_yaw_rad', 'source_result_sha256')} for m in MODELS}))
        return dict(id=self.spec['id'], episode=self.spec['episode_id'], start_s=self.start, end_s=self.end,
            goal_interval_s=self.interval, metrics=self.clip_metrics, plans=plans,
            metric_aggregation={'method': 'arithmetic_mean_of_complete_local_plans',
                'plan_count': len(self.rows), 'display': 'constant_for_entire_clip',
                'metrics': ['ate_m', 'rpe_m'], 'units': 'meters'},
            alignment=public(self.alignment.get('alignment', {})),
            alignment_sha256=digest(self.root / 'alignment_manifest.json'),
            evaluation_windows_sha256=digest(self.evaluation / 'windows.json'),
            model_protocols=public(self.protocols))


def export_trajectories(clip, directory):
    rows = []
    for window in clip.rows:
        for key in ('gt', *MODELS):
            path = window['reference_xy_m'] if key == 'gt' else clip.results[window['id']][key]['planned_xy_m']
            for step, xy in enumerate(path):
                rows.append(dict(series=key, window_id=window['id'], clip_time_s=window['start_time_s'] - clip.start + step / clip.fps,
                    plan_time_s=step / clip.fps, forward_m=xy[0], left_m=xy[1]))
    with (directory / 'trajectories.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(clip, directory, fps, crf):
    directory.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-ss', str(clip.start), '-i', str(clip.third),
        '-t', str(clip.duration), '-an', '-vf', f'fps={fps},scale=1200:675:flags=lanczos', '-threads', '2',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    decoder = subprocess.Popen(command, stdout=subprocess.PIPE)
    print('DECODER', decoder.pid, ' '.join(command), flush=True)
    encoders = {'comparison': Encoder(directory / 'comparison.mp4', SIZE, fps, crf)}
    encoders.update({m: Encoder(directory / f'{m}.mp4', (224, 224), fps, 16) for m in MODELS})
    count = round(clip.duration * fps)
    try:
        for n in range(count):
            raw = decoder.stdout.read(THIRD[0] * THIRD[1] * 3)
            if len(raw) != THIRD[0] * THIRD[1] * 3:
                raise RuntimeError(f'Third-person decoder ended at {n}/{count}')
            image, frames, index = clip.compose(clip.start + n / fps, Image.frombytes('RGB', THIRD, raw))
            encoders['comparison'].write(image)
            for model in MODELS:
                encoders[model].write(frames[model][index])
            if n == round(.75 * fps):
                image.save(directory / 'poster.webp', quality=92)
            if n % (fps * 5) == 0:
                print('RENDER', clip.spec['id'], n, count, flush=True)
        decoder.stdout.close()
        if decoder.wait():
            raise RuntimeError('Third-person decoder failed')
    finally:
        for encoder in encoders.values():
            encoder.close()
        if decoder.poll() is None:
            decoder.terminate()
            decoder.wait()
    # Full-resolution aligned views, independent of the dashboard layout.
    for source, name in [(clip.ego, 'gt_ego.mp4'), (clip.third, 'gt_third_person.mp4')]:
        run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-ss', str(clip.start), '-i', source,
            '-t', str(clip.duration), '-an', '-c:v', 'libx264', '-threads', '4', '-preset', 'slow', '-crf', '17',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', directory / name])
    export_trajectories(clip, directory)
    save(directory / 'metrics.json', clip.report())
    for name in ['comparison.mp4', 'opennwm.mp4', 'nwm.mp4', 'gt_ego.mp4', 'gt_third_person.mp4']:
        info = probe(directory / name)
        if abs(float(info['duration']) - clip.duration) > .08:
            raise ValueError(f'Duration mismatch: {directory / name}')
    print('CLIP_COMPLETE', clip.spec['id'], flush=True)


def publish_manifest(plan, output, fps):
    clips, reports = [], []
    for spec in plan['clips']:
        clip = Clip(spec)
        prefix = f'assets/go2/{spec["id"]}'
        files = ('comparison.mp4', 'poster.webp', 'gt_ego.mp4', 'gt_third_person.mp4', 'opennwm.mp4', 'nwm.mp4', 'metrics.json', 'trajectories.csv')
        if any(not (output / spec['id'] / name).is_file() for name in files):
            raise FileNotFoundError(f'Incomplete clip export: {spec["id"]}')
        clips.append(dict(id=spec['id'], episode=spec['episode_id'], route_id=f"route_{spec['route']:02d}",
            title=spec['title'], title_zh=spec['title_zh'],
            duration_s=clip.duration, start_time_s=clip.start, goal_interval_s=clip.interval,
            video=f'{prefix}/comparison.mp4', poster=f'{prefix}/poster.webp', metrics=clip.summary(),
            metric_aggregation=clip.report()['metric_aggregation'],
            evaluated_goal_ids=[r['id'] for r in clip.rows],
            downloads=dict(ego=f'{prefix}/gt_ego.mp4', third_person=f'{prefix}/gt_third_person.mp4',
                opennwm=f'{prefix}/opennwm.mp4', nwm=f'{prefix}/nwm.mp4', trajectories=f'{prefix}/trajectories.csv',
                metrics=f'{prefix}/metrics.json')))
        reports.append(clip.report())
    manifest = dict(version=3, status='ready', clips=clips, report='assets/go2/report.json')
    if 'routes' in plan:
        manifest['routes'] = plan['routes']
    if 'selection_basis' in plan:
        manifest['selection'] = {'criteria': plan['selection_basis'], 'display_clip_count': len(clips)}
    report = dict(schema_version=3, episodes=list(dict.fromkeys(s['episode_id'] for s in plan['clips'])), clips=reports,
        protocol=dict(evaluation='Offline visual-goal planning; predictions were not executed on the robot.',
            reference='Recorded onboard odometry in meters; approximate visual camera alignment.',
            selection=plan.get('selection_basis', 'Continuous 20–30 s clips preselected for scene coverage and unobstructed third-person view before evaluation.'),
            planning='Each local plan starts from recorded context; both models share goals, horizon and CEM budget within each clip.',
            metrics='Paper ATE/RPE definitions, averaged across complete local plans. Different planning horizons are reported separately.',
            sampling='Original DDPM checkpoints; no DMD2. Budget and exact settings are retained per clip.'),
        visualization=dict(layout_version=4, resolution=list(SIZE), fps=fps, model_sample_fps=4, interpolation=False,
            navigation_metrics='Constant arithmetic mean of all complete local plans within this clip, in meters.',
            third_person='Dominant full-aspect video; high-quality SDR conversion where required.',
            gt_ego='High-resolution recorded frames with the same center crop as model input, shown at matched 4 Hz timestamps.',
            model='Exact saved generated frames, repeated at output FPS; no image enhancement or replacement.',
            sampling='Nearest 4 Hz sample shared by GT/model/trajectory; third-person follows aligned continuous time.',
            trajectory='Forward up, physical left to the left. Equal metric scale shared by all three paths, fit per local plan; no smoothing.'))
    for key in ('presentation_order', 'summary', 'quality'):
        if key in plan:
            report[key] = plan[key]
    save(output / 'website_manifest.json', manifest)
    save(output / 'report.json', report)
    inventory = {str(p.relative_to(output)): dict(bytes=p.stat().st_size, sha256=digest(p))
        for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'inventory.json'}
    save(output / 'inventory.json', inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--clip', help='Render only this clip ID, useful for parallel CPU jobs.')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--preview-time', type=float, default=.75, help='Seconds relative to the clip start.')
    parser.add_argument('--manifest-only', action='store_true')
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--crf', type=int, default=18)
    args = parser.parse_args()
    output = args.output or args.root / 'gallery'
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 4 * 1024**3:
        raise RuntimeError('At least 4 GiB free disk required')
    if args.fps <= 0 or args.fps % 4:
        parser.error('--fps must be a positive multiple of 4')
    plan = load(args.plan or args.root / 'gallery_plan.json')
    selected = [s for s in plan['clips'] if not args.clip or s['id'] == args.clip]
    if not selected:
        parser.error('Requested clip does not exist')
    if not args.manifest_only:
        for spec in selected:
            clip = Clip(spec)
            if args.preview:
                time = clip.start + args.preview_time
                if not clip.start <= time < clip.end:
                    raise ValueError('Preview time is outside the clip')
                image, _, _ = clip.compose(time, camera_frame(clip.third, time))
                path = args.root / '_work' / f'preview_{spec["id"]}.jpg'
                path.parent.mkdir(exist_ok=True)
                image.save(path, quality=95)
                print('PREVIEW', path, flush=True)
            else:
                render(clip, output / spec['id'], args.fps, args.crf)
    if args.manifest_only or (not args.preview and not args.clip):
        publish_manifest(plan, output, args.fps)
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
