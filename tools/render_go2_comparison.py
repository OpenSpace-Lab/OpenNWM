#!/usr/bin/env python3
"""Compose real Go2/CEM evaluation artifacts into a synchronized comparison film.

CPU only. Original/aligned recordings and evaluator outputs are read-only.
Example:
  conda run --no-capture-output -n base python tools/render_go2_comparison.py --preview-only
  conda run --no-capture-output -n base python tools/render_go2_comparison.py --publish
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/file_system/nas/algorithm/dujun.nie/opennwm_go2_20261026')
BG = '#0d1113'
SURFACE = '#161d20'
LINE = '#2d393c'
TEXT = '#f0f3ee'
MUTED = '#a1afad'
GT = '#e8eeea'
GREEN = '#b6ee65'
BLUE = '#87b8ff'
MODELS = ('opennwm', 'nwm')
COLORS = {'gt': GT, 'opennwm': GREEN, 'nwm': BLUE}
CLIPS = [
    ('clip_01_office_exit', 16, 'Leaving the office', 'Leaving the office'),
    ('clip_02_corridor', 40, 'Along the corridor', 'Along the corridor'),
    ('clip_03_corner', 58, 'Around the corner', 'Around the corner'),
    ('clip_04_foyer_exit', 76, 'Leaving the foyer', 'Leaving the foyer'),
    ('clip_05_glass_door', 110, 'At the glass door', 'At the glass door'),
    ('clip_06_turnaround', 132, 'Turning back', 'Turning back'),
    ('clip_07_return', 156, 'Returning to the office', 'Returning to the office'),
]


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def public_metadata(value):
    """Retain reproducibility facts while removing private filesystem locations."""
    if isinstance(value, dict):
        return {key: public_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    if isinstance(value, str) and value.startswith('/'):
        return Path(value).name
    return value


def command(args):
    print('COMMAND', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def timecode(seconds):
    return f'{int(seconds) // 60:02d}:{int(seconds) % 60:02d}'


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)]))


class Encoder:
    def __init__(self, path, size, fps, crf=21):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                        '-s', f'{size[0]}x{size[1]}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                        '-threads', '4', '-preset', 'fast', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
        print('ENCODER', ' '.join(self.command), flush=True)
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE)
        print('ENCODER_PID', self.process.pid, str(path), flush=True)

    def write(self, frame):
        self.process.stdin.write(frame.tobytes())

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        status = self.process.wait()
        print('ENCODER_EXIT', status, str(self.path), flush=True)
        if status:
            raise RuntimeError(f'ffmpeg exited {status}: {self.path}')


def ffmpeg_frame(path, time, size=(496, 279)):
    raw = subprocess.check_output(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', str(time), '-i', str(path),
                                   '-frames:v', '1', '-vf', f'scale={size[0]}:{size[1]}', '-threads', '2',
                                   '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
    return Image.frombytes('RGB', size, raw)


def model_image(path):
    """Exact PIL-side equivalent of CenterCropAR(4/3), Resize(224,224)."""
    image = Image.open(path).convert('RGB')
    w, h = image.size
    crop_w, crop_h = (int(h * 4 / 3), h) if w > h else (w, int(w / (4 / 3)))
    # torchvision center_crop uses rounded half offsets, including any padding.
    left, top = int(round((w - crop_w) / 2)), int(round((h - crop_h) / 2))
    return image.crop((left, top, left + crop_w, top + crop_h)).resize((224, 224), Image.Resampling.BILINEAR)


class Composer:
    def __init__(self, root):
        self.root = root
        self.windows = load_json(root / 'evaluation/windows.json')
        self.alignment = load_json(root / 'alignment_manifest.json')
        self.rows = sorted(self.windows['windows'], key=lambda r: r['start_time_s'])
        self.duration = float(self.windows['duration_s'])
        self.third = root / 'aligned/third_person.mp4'
        self.ego = root / 'aligned/ego.mp4'
        self.fonts = {}
        self.results = {}
        self.cache = {}
        self.missing = []
        self.protocols = {}
        for model in MODELS:
            path = root / 'evaluation/outputs' / model / 'protocol.json'
            if path.exists():
                self.protocols[model] = load_json(path)
        if len(self.protocols) == 2:
            for key in ('candidates', 'elites', 'horizon', 'fps', 'goal_interval_s', 'repeats', 'updates', 'diffusion_sampler', 'diffusion_steps', 'waypoint_spacing_m'):
                if self.protocols['opennwm'].get(key) != self.protocols['nwm'].get(key):
                    raise ValueError(f'Planning budget mismatch for {key}')
            if self.protocols['opennwm'].get('diffusion_sampler') != 'DDPM':
                raise ValueError('This showcase requires the standard DDPM models, not a distilled sampler.')
        for row in self.rows:
            rid = row['id']
            self.results[rid] = {}
            for model in MODELS:
                path = root / 'evaluation/outputs' / model / f'{rid:06d}/result.json'
                if not path.exists():
                    self.missing.append(f'{model}/{rid:06d}')
                    continue
                result = load_json(path)
                for metric in ('ate_m', 'rpe_m'):
                    if not math.isfinite(float(result['metrics'][metric])):
                        raise ValueError(f'Nonfinite metric: {path} {metric}')
                for index in range(8):
                    frame = path.parent / 'predicted_frames' / f'{index:03d}.png'
                    if not frame.is_file():
                        raise FileNotFoundError(frame)
                result['_directory'] = path.parent
                result['_sha256'] = digest(path)
                self.results[rid][model] = result

    def font(self, size):
        if size not in self.fonts:
            font = ImageFont.truetype(str(REPO / 'website/assets/fonts/Manrope.ttf'), size=size)
            font.set_variation_by_axes([500])
            self.fonts[size] = font
        return self.fonts[size]

    def text(self, draw, xy, text, size=16, color=TEXT, anchor=None):
        draw.text(xy, text, font=self.font(size), fill=color, anchor=anchor)

    def find_window(self, time):
        for row in self.rows:
            if row['start_time_s'] <= time < row['end_time_s']:
                return row
        if time >= self.rows[-1]['end_time_s'] and time <= self.duration:
            return self.rows[-1]
        raise ValueError(f'No evaluated window at aligned time {time}')

    def frames(self, row):
        rid = row['id']
        if rid not in self.cache:
            if len(self.cache) > 2:
                self.cache.clear()
            observed = [model_image(path) for path in row['frames']]
            value = {'gt': observed}
            for model in MODELS:
                result = self.results[rid].get(model)
                if result is None:
                    raise ValueError(f'Missing evaluated result: {model}/{rid:06d}')
                value[model] = [observed[0]] + [Image.open(result['_directory'] / 'predicted_frames' / f'{index:03d}.png').convert('RGB') for index in range(8)]
            self.cache[rid] = value
        return self.cache[rid]

    def selected_index(self, row, time):
        return min(8, max(0, int(math.floor((time - row['start_time_s']) * 4 + .5))))

    def plot(self, image, row, index):
        d = ImageDraw.Draw(image)
        x, y, w, h = 1008, 64, 568, 440
        d.rectangle((x, y, x + w - 1, y + h - 1), fill=SURFACE)
        self.text(d, (x + 24, y + 17), '2 s local plan', 21)
        for offset, key, label in [(24, 'gt', 'GT'), (164, 'opennwm', 'OpenNWM'), (374, 'nwm', 'NWM')]:
            lx = x + offset
            d.line((lx, y + 65, lx + 19, y + 65), fill=COLORS[key], width=3)
            self.text(d, (lx + 28, y + 54), label, 15)
        paths = {'gt': np.asarray(row['reference_xy_m'], dtype=float)}
        paths.update({model: np.asarray(self.results[row['id']][model]['planned_xy_m'], dtype=float) for model in MODELS})
        # Current robot frame: forward is upward; physical left is screen left.
        paths = {key: np.stack((-path[:, 1], -path[:, 0]), axis=1) for key, path in paths.items()}
        all_points = np.concatenate(list(paths.values()))
        lower, upper = all_points.min(0), all_points.max(0)
        span = np.maximum(upper - lower, .15)
        left, top, right, bottom = x + 28, y + 96, x + w - 28, y + h - 48
        scale = min((right - left - 24) / span[0], (bottom - top - 24) / span[1])
        center = (lower + upper) / 2
        def mapped(path):
            return [tuple(p) for p in ((path - center) * scale + np.array([(left + right) / 2, (top + bottom) / 2]))]
        for gx in range(left, right + 1, 48):
            d.line((gx, top, gx, bottom), fill='#263034', width=1)
        for gy in range(top, bottom + 1, 48):
            d.line((left, gy, right, gy), fill='#263034', width=1)
        for key in ('gt', 'opennwm', 'nwm'):
            points = mapped(paths[key])
            faint = {'gt': '#53605e', 'opennwm': '#4e6534', 'nwm': '#3b5272'}[key]
            d.line(points, fill=faint, width=3, joint='curve')
            if index:
                d.line(points[:index + 1], fill=COLORS[key], width=4, joint='curve')
            px, py = points[index]
            d.ellipse((px - 5, py - 5, px + 5, py + 5), fill=COLORS[key], outline=SURFACE, width=1)
        start = mapped(paths['gt'])[0]
        goal = mapped(paths['gt'])[-1]
        d.ellipse((start[0] - 5, start[1] - 5, start[0] + 5, start[1] + 5), fill=SURFACE, outline=GT, width=2)
        d.ellipse((goal[0] - 7, goal[1] - 7, goal[0] + 7, goal[1] + 7), outline=GT, width=2)
        meters = 10 ** math.floor(math.log10(max(60 / scale, .001)))
        if meters * scale < 30:
            meters *= 2
        d.line((x + 28, y + h - 22, x + 28 + meters * scale, y + h - 22), fill=MUTED, width=2)
        self.text(d, (x + 35 + meters * scale, y + h - 33), f'{meters:g} m', 13, MUTED)
        self.text(d, (x + w - 24, y + h - 33), 'Local frame · equal scale', 13, MUTED, 'ra')

    def compose(self, time, third, title='Full recording'):
        row = self.find_window(time)
        index = self.selected_index(row, time)
        frames = self.frames(row)
        image = Image.new('RGB', (1600, 900), BG)
        d = ImageDraw.Draw(image)
        self.text(d, (24, 17), 'Recorded third person', 24)
        self.text(d, (984, 26), f'Go2  /  {timecode(time)} — {timecode(self.duration)}', 15, MUTED, 'ra')
        self.text(d, (1008, 24), 'VISUAL-GOAL NAVIGATION', 13, MUTED)
        self.text(d, (1576, 18), 'OpenNWM', 24, GREEN, 'ra')
        # The complete 16:9 third-person recording is the visual anchor.
        image.paste(third.resize((960, 540), Image.Resampling.BILINEAR), (24, 64))
        # Three equally sized 4:3 views share one baseline and sample index.
        for x, key, label in [(24, 'gt', 'First-person GT'), (348, 'opennwm', 'OpenNWM'), (672, 'nwm', 'NWM')]:
            d.line((x, 628, x + 15, 628), fill=COLORS[key], width=3)
            self.text(d, (x + 24, 616), label, 18, COLORS[key])
            image.paste(frames[key][index].resize((312, 234), Image.Resampling.BICUBIC), (x, 646))
        self.plot(image, row, index)
        d = ImageDraw.Draw(image)
        x, y, w, h = 1008, 520, 568, 360
        d.rectangle((x, y, x + w - 1, y + h - 1), fill=SURFACE)
        image.paste(frames['gt'][-1].resize((176, 132), Image.Resampling.BICUBIC), (x + w - 200, y + 24))
        self.text(d, (x + 24, y + 24), f'Goal at {timecode(row["end_time_s"])}', 23)
        self.text(d, (x + 24, y + 65), f'Plan {row["id"] + 1:02d} / {len(self.rows)}   ·   +{index / 4:.2f} s', 15, MUTED)
        self.text(d, (x + 24, y + 98), '2 s context reset · offline CEM', 14, MUTED)
        d.line((x + 24, y + 178, x + w - 24, y + 178), fill=LINE, width=1)
        self.text(d, (x + 24, y + 193), 'PLAN ERROR', 12, MUTED)
        self.text(d, (x + 350, y + 193), 'ATE (m)', 13, MUTED, 'ra')
        self.text(d, (x + w - 24, y + 193), 'RPE (m)', 13, MUTED, 'ra')
        for offset, model, label in [(224, 'opennwm', 'OpenNWM'), (268, 'nwm', 'NWM')]:
            metrics = self.results[row['id']][model]['metrics']
            d.line((x + 24, y + offset + 14, x + 41, y + offset + 14), fill=COLORS[model], width=3)
            self.text(d, (x + 50, y + offset), label, 19, COLORS[model])
            self.text(d, (x + 350, y + offset - 2), f'{metrics["ate_m"]:.3f}', 23, TEXT, 'ra')
            self.text(d, (x + w - 24, y + offset - 2), f'{metrics["rpe_m"]:.3f}', 23, TEXT, 'ra')
        self.text(d, (x + 24, y + 329), 'Lower is better · GT: onboard odometry', 13, MUTED)
        return image, frames, index

    def metric_summary(self, start, end):
        selected = [r for r in self.rows if r['start_time_s'] >= start - 1e-6 and r['end_time_s'] <= end + 1e-6]
        if not selected:
            raise ValueError('No full two-second plans in requested clip')
        result = {model: {metric: float(np.mean([self.results[r['id']][model]['metrics'][metric] for r in selected])) for metric in ('ate_m', 'rpe_m')} for model in MODELS}
        return result, [r['id'] for r in selected]


def cut(source, target, start, duration, scale=None):
    args = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-ss', str(start), '-i', str(source), '-t', str(duration), '-an']
    if scale:
        args += ['-vf', f'scale={scale}:trunc(ow/a/2)*2']
    args += ['-c:v', 'libx264', '-threads', '4', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)]
    command(args)


def render(args, composer, output, start=0, end=None, preview=False):
    end = composer.duration if end is None else end
    if composer.missing and not preview:
        raise ValueError(f'Incomplete evaluation ({len(composer.missing)} missing): {composer.missing[:10]}')
    if composer.windows.get('skipped_windows') and not preview:
        raise ValueError('A continuous full film needs all evaluation windows; inspect skipped windows before export.')
    count = round((end - start) * args.fps)
    suffix = f'preview_{start:07.2f}' if preview else 'full'
    decoder_args = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-ss', str(start), '-i', str(composer.third), '-t', str(end - start), '-an', '-vf', f'fps={args.fps},scale=960:540', '-threads', '2', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    print('DECODER', ' '.join(decoder_args), flush=True)
    decoder = subprocess.Popen(decoder_args, stdout=subprocess.PIPE)
    print('DECODER_PID', decoder.pid, flush=True)
    encoders = {'comparison': Encoder(output / f'comparison_{suffix}.mp4', (1600, 900), args.fps, args.crf)}
    encoders.update({model: Encoder(output / f'{model}_{suffix}.mp4', (224, 224), args.fps, 18) for model in MODELS})
    snapshots = set(round(t * args.fps) for t in args.preview_times)
    try:
        for number in range(count):
            raw = decoder.stdout.read(960 * 540 * 3)
            if len(raw) != 960 * 540 * 3:
                raise RuntimeError(f'Third-person video ended at frame {number}/{count}')
            time = start + number / args.fps
            third = Image.frombytes('RGB', (960, 540), raw)
            image, frames, index = composer.compose(time, third, 'Pilot preview' if preview else 'Full recording')
            encoders['comparison'].write(image)
            for model in MODELS:
                encoders[model].write(frames[model][index])
            if number in snapshots:
                image.save(output / 'previews' / f'frame_{time:07.2f}.jpg', quality=93)
            if number % (args.fps * 10) == 0:
                print(f'RENDER {time:.1f}/{end:.1f}s ({number}/{count})', flush=True)
        decoder.stdout.close()
        status = decoder.wait()
        if status:
            raise RuntimeError(f'Third-person decoder exited {status}')
    finally:
        for encoder in encoders.values():
            encoder.close()
        if decoder.poll() is None:
            decoder.terminate()
            decoder.wait()
    print('FILM_DONE', output / f'comparison_{suffix}.mp4', flush=True)


def publish_assets(args, composer, output):
    film = output / 'comparison_full.mp4'
    duration = float(probe(film)['format']['duration'])
    if abs(duration - composer.duration) > 1 / args.fps + .001:
        raise ValueError(f'Duration mismatch: {duration} != {composer.duration}')
    poster = ffmpeg_frame(film, min(18.75, duration / 2), (1600, 900))
    poster.save(output / 'poster_full.webp', quality=88)
    metrics, ids = composer.metric_summary(0, composer.duration)
    manifest = dict(version=1, status='ready', recorded_date='2026-10-26', episode_id='episode_20261026T091121_148a01b5',
        full=dict(id='full', title='Full comparison', title_zh='Full comparison', description=f'{len(ids)} visual goals across the recorded route. Both models receive the same context, goal images and CEM budget.',
                  description_zh=f'{len(ids)} visual goals across the recorded route. Both models receive the same context, goal images and CEM budget.', duration_s=duration,
                  video='assets/go2/comparison_full.mp4', poster='assets/go2/poster_full.webp'),
        clips=[], summary=dict(metrics=metrics, evaluated_goal_count=len(ids)),
        protocol=dict(evaluation='Offline visual-goal navigation', evaluation_zh='Offline visual-goal navigation', goal_interval_s=2, planner='CEM',
            note='Independent 2 s plans reset to recorded context; predicted paths were not executed on the robot. GT uses onboard odometry; cameras are approximately aligned. ATE/RPE are mean errors in meters, using the paper definitions. This recording uses 32 CEM candidates, 5 elites, 2 updates and 50-step DDPM: a practical budget, not an exact rerun of the paper benchmark.',
            note_zh='Independent 2 s plans reset to recorded context; predicted paths were not executed on the robot. GT uses onboard odometry; cameras are approximately aligned. ATE/RPE are mean errors in meters, using the paper definitions. This recording uses 32 CEM candidates, 5 elites, 2 updates and 50-step DDPM: a practical budget, not an exact rerun of the paper benchmark.'),
        report='assets/go2/report.json')
    for clip_id, start, title, title_zh in CLIPS:
        if start + 8 > duration:
            continue
        directory = output / clip_id
        directory.mkdir(exist_ok=True)
        for source, filename in [(film, 'comparison.mp4'), (composer.ego, 'gt_ego.mp4'), (composer.third, 'gt_third_person.mp4'),
                                 (output / 'opennwm_full.mp4', 'opennwm.mp4'), (output / 'nwm_full.mp4', 'nwm.mp4')]:
            cut(source, directory / filename, start, 8, scale=960 if filename.startswith('gt_') else None)
        ffmpeg_frame(directory / 'comparison.mp4', .75, (1600, 900)).save(directory / 'poster.webp', quality=88)
        clip_metrics, clip_ids = composer.metric_summary(start, start + 8)
        prefix = f'assets/go2/{clip_id}'
        manifest['clips'].append(dict(id=clip_id, title=title, title_zh=title_zh, duration_s=8, start_time_s=start,
            description=f'{timecode(start)}–{timecode(start + 8)} · {len(clip_ids)} consecutive visual goals. Recorded motion and both model futures share one timeline.',
            description_zh=f'{timecode(start)}–{timecode(start + 8)} · {len(clip_ids)} consecutive visual goals. Recorded motion and both model futures share one timeline.',
            video=f'{prefix}/comparison.mp4', poster=f'{prefix}/poster.webp', metrics=clip_metrics, evaluated_goal_ids=clip_ids,
            downloads=dict(ego=f'{prefix}/gt_ego.mp4', third_person=f'{prefix}/gt_third_person.mp4', opennwm=f'{prefix}/opennwm.mp4', nwm=f'{prefix}/nwm.mp4')))
    reports = []
    for row in composer.rows:
        result = dict(id=row['id'], start_time_s=row['start_time_s'], end_time_s=row['end_time_s'],
                      reference_xy_m=row['reference_xy_m'], reference_world_xy_m=row.get('reference_world_xy_m'), models={})
        for model in MODELS:
            data = composer.results[row['id']][model]
            result['models'][model] = {key: data[key] for key in ('metrics', 'planned_xy_m', 'planned_yaw_rad')}
            result['models'][model]['source_result_sha256'] = data['_sha256']
        reports.append(result)
    report = dict(schema_version=1, episode_id=manifest['episode_id'], recorded_date=manifest['recorded_date'],
                  scope=composer.windows.get('scope'), reference=composer.windows.get('reference'),
                  metric_aggregation='Unweighted arithmetic mean of complete 2 s plans; ATE and RPE are computed by the repository paper evaluator, then converted to meters with the fixed Office-Go2 waypoint scale.',
                  waypoint_spacing_m=composer.windows.get('waypoint_spacing_m'),
                  spacing_policy=composer.windows.get('spacing_policy'),
                  evaluation_windows_sha256=digest(composer.root / 'evaluation/windows.json'),
                  alignment_manifest_sha256=digest(composer.root / 'alignment_manifest.json'),
                  camera_alignment=dict(aligned_start_host_s=composer.alignment.get('aligned_start_host_s'),
                      duration_s=composer.duration, **public_metadata(composer.alignment.get('alignment', {}))),
                  visualization=dict(layout_version=2, layout='Third-person main view; equal-size first-person GT and model views below; trajectory, goal and metrics in a fixed right column.', fps=args.fps, model_sample_fps=4, interpolation=False,
                    sample_selection='Nearest 4 Hz sample; both GT ego and model frames share sample index. Third-person video follows continuous aligned time.',
                    aspect_ratio='Model 224x224 and identically preprocessed GT are displayed at 4:3, reversing the training square resize.',
                    trajectory_plot='Current robot frame; forward upward, left leftward; equal metric scale; no path smoothing.',
                    context_reset_s=2), summary=manifest['summary'], clips=manifest['clips'], plans=reports)
    report['model_protocols'] = public_metadata(composer.protocols)
    write_json(output / 'report.json', report)
    write_json(output / 'website_manifest.json', manifest)
    inventory = {str(path.relative_to(output)): dict(bytes=path.stat().st_size, sha256=digest(path)) for path in output.rglob('*') if path.suffix in ('.mp4', '.webp', '.json') and path.name != 'inventory.json'}
    write_json(output / 'inventory.json', inventory)
    if args.publish:
        public = args.website / 'assets/go2'
        public.mkdir(parents=True, exist_ok=True)
        paths = ['comparison_full.mp4', 'poster_full.webp', 'report.json']
        for clip in manifest['clips']:
            paths.extend(str(path.relative_to(output)) for path in (output / clip['id']).iterdir() if path.is_file())
        for relative in paths:
            source, target = output / relative, public / relative
            if source.stat().st_size >= 95 * 1024**2:
                raise ValueError(f'Public asset exceeds 95 MiB: {source}')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        write_json(args.website / 'content/go2.json', manifest)
        print('WEBSITE_ASSETS_READY', public, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--website', type=Path, default=REPO / 'website')
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--crf', type=int, default=21)
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--preview-range', type=float, nargs=2, metavar=('START', 'END'), help='Render a nonpublished segment using already completed paired results.')
    parser.add_argument('--preview-times', type=float, nargs='+', default=[18.75, 42.75, 60.75, 78.75, 112.75, 134.75, 158.75])
    parser.add_argument('--export-only', action='store_true', help='Reuse completed full films; export clips, posters and reports.')
    parser.add_argument('--publish', action='store_true', help='Copy verified artifacts to local website; does not push or deploy.')
    args = parser.parse_args()
    if args.fps <= 0 or args.fps % 4:
        parser.error('--fps must be a positive multiple of 4')
    output = args.output or args.root / 'visualizations'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'previews').mkdir(exist_ok=True)
    free = shutil.disk_usage(output).free
    print(f'OUTPUT {output}\nFREE_DISK_BYTES {free}', flush=True)
    if free < 2 * 1024**3:
        raise RuntimeError('At least 2 GiB free disk is required.')
    composer = Composer(args.root)
    if args.preview_range:
        start, end = args.preview_range
        if start < 0 or end <= start or end > composer.duration:
            parser.error('--preview-range must lie within the aligned recording')
        if args.publish:
            parser.error('--preview-range cannot be published')
        render(args, composer, output, start, end, preview=True)
        return
    if args.preview_only:
        for time in args.preview_times:
            row = composer.find_window(time)
            if len(composer.results[row['id']]) != 2:
                print('SKIP_INCOMPLETE_PREVIEW', time, row['id'], flush=True)
                continue
            third = ffmpeg_frame(composer.third, time, (960, 540))
            image, _, _ = composer.compose(time, third)
            path = output / 'previews' / f'frame_{time:07.2f}.jpg'
            image.save(path, quality=93)
            print('PREVIEW', path, flush=True)
        return
    if not args.export_only:
        render(args, composer, output)
    publish_assets(args, composer, output)
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
