#!/usr/bin/env python3
"""Check sampled gallery pixels against source cameras and saved predictions.

This is a content correspondence audit, separate from full decoding and checksum
checks. Three timestamps per clip cover its beginning, middle and end. Lossy
H.264 pixels are compared with an explicit mean absolute error tolerance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time as clock

import numpy as np
from PIL import Image


def read(path):
    return json.loads(Path(path).read_text())


def extract(path, indices, size, start=0, source_fps=None):
    select = '+'.join(f'eq(n\\,{i})' for i in indices)
    filters = ([f'fps={source_fps}'] if source_fps else []) + [
        f'select={select}', f'scale={size[0]}:{size[1]}:flags=lanczos']
    command = ['ffmpeg', '-v', 'error', '-threads', '2', '-ss', str(start),
               '-i', str(path), '-vf', ','.join(filters), '-frames:v', str(len(indices)),
               '-vsync', '0', '-threads', '2', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    raw = subprocess.check_output(command)
    expected = len(indices) * size[0] * size[1] * 3
    if len(raw) != expected:
        raise AssertionError(f'Incomplete sample decode: {path}, {len(raw)} != {expected}')
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(indices), size[1], size[0], 3)


def check(actual, expected, label, maximum_mae=10):
    error = actual.astype(np.float32) - np.asarray(expected, dtype=np.float32)
    mae = float(np.abs(error).mean())
    mse = float(np.square(error).mean())
    psnr = float(10 * np.log10(255 ** 2 / mse)) if mse else None
    if mae > maximum_mae:
        raise AssertionError(f'{label}: pixel mismatch MAE={mae:.4f} > {maximum_mae}')
    return {'label': label, 'mae_8bit': mae, 'psnr_db': psnr}


def cropped_observation(path):
    image = Image.open(path).convert('RGB')
    width, height = image.size
    cw, ch = (int(height * 4 / 3), height) if width > height else (width, int(width * 3 / 4))
    x, y = round((width - cw) / 2), round((height - ch) / 2)
    return image.crop((x, y, x + cw, y + ch))


def audit_clip(spec, gallery, wait_ready_s=0):
    directory = gallery / spec['id']
    deadline = clock.monotonic() + wait_ready_s
    while not (directory / 'metrics.json').is_file():
        if clock.monotonic() >= deadline:
            raise FileNotFoundError(f'Clip render not complete: {directory}')
        clock.sleep(1)
    start, end = spec['start_s'], spec['end_s']
    duration = end - start
    times = [.25, int(duration / 2) + .5, duration - .25]
    indices = [round(t * 24) for t in times]
    comparison = extract(directory / 'comparison.mp4', indices, (1920, 1080))
    windows = read(Path(spec['evaluation_root']) / 'windows.json')
    alignment = read(Path(spec['episode_root']) / 'alignment_manifest.json')
    checks = []
    samples = []
    for i, time in enumerate(times):
        absolute = start + time
        row = next(r for r in windows['windows'] if r['start_time_s'] <= absolute < r['end_time_s'])
        step = int(np.floor((absolute - row['start_time_s']) * windows['fps'] + .5))
        if step < 1:
            raise AssertionError('Audit must inspect generated frames, not initial context')
        observed = cropped_observation(row['frames'][step]).resize((384, 288), Image.Resampling.LANCZOS)
        checks.append(check(comparison[i, 774:1062, 24:408], observed, f'GT panel @ {time}s'))
        samples.append((row, step))
    for model, x in [('opennwm', 432), ('nwm', 840)]:
        standalone = extract(directory / f'{model}.mp4', indices, (224, 224))
        for i, (row, step) in enumerate(samples):
            prediction = Image.open(Path(spec['evaluation_root']) / 'outputs' / model /
                                    f'{row["id"]:06d}' / 'predicted_frames' / f'{step - 1:03d}.png').convert('RGB')
            checks.append(check(standalone[i], prediction, f'{model} standalone @ {times[i]}s'))
            checks.append(check(comparison[i, 774:1062, x:x + 384],
                                prediction.resize((384, 288), Image.Resampling.LANCZOS),
                                f'{model} panel @ {times[i]}s'))
    for key, filename in [('ego_video', 'gt_ego.mp4'), ('third_video', 'gt_third_person.mp4')]:
        # Both source and cut are sampled through the same 24 Hz temporal grid.
        source = extract(alignment[key], indices, (1200, 675), start=start, source_fps=24)
        cut = extract(directory / filename, indices, (1200, 675), source_fps=24)
        for i, time in enumerate(times):
            checks.append(check(cut[i], source[i], f'{filename} source @ {time}s'))
            if key == 'third_video':
                checks.append(check(comparison[i, 48:723, 24:1224], source[i], f'Third-person panel @ {time}s'))
    return {'id': spec['id'], 'passed': True, 'sample_times_s': times,
            'maximum_mae_8bit': max(c['mae_8bit'] for c in checks), 'checks': checks}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--gallery', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--clip')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--wait-ready-s', type=float, default=0,
                    help='Wait for the final metrics export, written after video encoders close.')
    args = ap.parse_args()
    specs = [s for s in read(args.plan)['clips'] if not args.clip or s['id'] == args.clip]
    if not specs:
        ap.error('No selected clips')
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for result in pool.map(lambda spec: audit_clip(spec, args.gallery, args.wait_ready_s), specs):
            results.append(result)
            print(json.dumps({k: result[k] for k in ('id', 'passed', 'maximum_mae_8bit')}), flush=True)
    result = {'passed': True, 'scope': 'Three timestamps per clip; source cameras, GT panel, both model panels and standalone model videos. This is a sampled pixel correspondence check, not an exhaustive all-frame identity proof.',
              'threshold_mae_8bit': 10, 'clips': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
