#!/usr/bin/env python3
"""Export GT and independently anchored Go2 CEM plans as PNG, SVG and CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np


DEFAULT_ROOT = Path('/file_system/nas/algorithm/dujun.nie/opennwm_go2_20261026')
COLORS = {'gt': '#354350', 'opennwm': '#68a330', 'nwm': '#4f83c7'}


def world_coordinates(local_xy, anchor_xy, anchor_yaw):
    """Invert the evaluator's initial-pose-relative SE(2) coordinate transform."""
    c, s = np.cos(anchor_yaw), np.sin(anchor_yaw)
    return np.asarray(local_xy) @ np.array([[c, s], [-s, c]]) + np.asarray(anchor_xy)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_inputs(root, allow_partial):
    path = root / 'evaluation' / 'windows.json'
    data = json.loads(path.read_text())
    windows = {int(w['id']): w for w in data['windows']}
    results, missing = {}, {}
    for model in ['opennwm', 'nwm']:
        values = [json.loads(p.read_text()) for p in sorted((root / 'evaluation' / 'outputs' / model).glob('*/result.json'))]
        results[model] = {int(r['id']): r for r in values}
        missing[model] = sorted(set(windows) - set(results[model]))
        unexpected = set(results[model]) - set(windows)
        if unexpected:
            raise ValueError(f'{model}: unexpected window IDs {unexpected}')
        for sample, result in results[model].items():
            window = windows[sample]
            anchor, yaw = window['reference_world_xy_m'][0], window['reference_world_yaw_rad'][0]
            expected = world_coordinates(result['reference_xy_m'], anchor, yaw)
            if not np.allclose(expected, window['reference_world_xy_m'], atol=1e-7):
                raise ValueError(f'{model}/{sample}: local/world ground-truth transform mismatch')
            if np.asarray(result['planned_xy_m']).shape != (9, 2):
                raise ValueError(f'{model}/{sample}: expected nine saved plan points')
            if not np.isfinite(result['planned_xy_m']).all():
                raise ValueError(f'{model}/{sample}: nonfinite saved plan')
    complete = not any(missing.values())
    if not complete and not allow_partial:
        counts = {m: len(v) for m, v in results.items()}
        raise SystemExit(f'Waiting for complete evaluation: {counts}, expected {len(windows)} each. No final plot written.')
    return data, windows, results, missing, complete


def plot(root, allow_partial=False):
    data, windows, results, missing, complete = read_inputs(root, allow_partial)
    out = root / 'visualizations'
    out.mkdir(exist_ok=True)
    stem = 'trajectory_overview' if complete else 'trajectory_overview_partial'
    world = {}
    csv_rows = []
    gt_points = []
    for sample, window in windows.items():
        reference = np.asarray(window['reference_world_xy_m'])
        gt_points.extend(reference)
        for step, xy in enumerate(reference):
            csv_rows.append(dict(series='gt_odometry', window_id=sample,
                window_start_aligned_s=window['start_time_s'], step=step,
                aligned_time_s=window['start_time_s'] + step / 4,
                source_host_time_s=window['source_host_times_s'][step],
                source_frame_index=window['source_frame_indices'][step],
                local_x_m=window['reference_xy_m'][step][0], local_y_m=window['reference_xy_m'][step][1],
                world_x_m=xy[0], world_y_m=xy[1],
                world_yaw_rad=window['reference_world_yaw_rad'][step]))
    for model, model_results in results.items():
        world[model] = {}
        for sample, result in model_results.items():
            window = windows[sample]
            anchor, yaw = window['reference_world_xy_m'][0], window['reference_world_yaw_rad'][0]
            xy = world_coordinates(result['planned_xy_m'], anchor, yaw)
            world[model][sample] = xy
            for step, point in enumerate(xy):
                csv_rows.append(dict(series=model, window_id=sample,
                    window_start_aligned_s=window['start_time_s'], step=step,
                    aligned_time_s=window['start_time_s'] + step / 4,
                    source_host_time_s=window['source_host_times_s'][step],
                    source_frame_index=window['source_frame_indices'][step],
                    local_x_m=result['planned_xy_m'][step][0], local_y_m=result['planned_xy_m'][step][1],
                    world_x_m=point[0], world_y_m=point[1],
                    world_yaw_rad=yaw + result['planned_yaw_rad'][step]))
    gt = np.asarray(gt_points)
    all_points = [gt] + [segment for values in world.values() for segment in values.values()]
    combined = np.concatenate(all_points)
    xmin, ymin = combined.min(0) - .8
    xmax, ymax = combined.max(0) + .8
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#c5cdd4', 'axes.labelcolor': '#52606b',
        'xtick.color': '#65727c', 'ytick.color': '#65727c',
        'grid.color': '#e6eaee', 'grid.linewidth': .65, 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 6.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.055, right=.985, bottom=.19, top=.78, wspace=.12)
    fig.text(.055, .945, 'Go2 navigation: recorded route and local plans', fontsize=21, weight='semibold', color='#1f2c36')
    detail = f"{data['duration_s']:.0f} seconds  /  {len(windows)} visual goals  /  2-second CEM horizon  /  shared metric scale"
    if not complete:
        detail += '  /  PARTIAL PREVIEW'
    fig.text(.055, .895, detail, fontsize=10.5, color='#63707a')
    gt_segments = [np.asarray(w['reference_world_xy_m']) for w in windows.values()]
    for ax, series, title in zip(axes, ['gt', 'opennwm', 'nwm'], ['Recorded odometry', 'OpenNWM', 'NWM']):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.grid(True, zorder=0)
        ax.set_xlabel('x (m)', labelpad=8)
        ax.add_collection(LineCollection(gt_segments,
            colors=COLORS['gt'] if series == 'gt' else '#c6cdd4',
            linewidths=2.3 if series == 'gt' else 1.6,
            alpha=1 if series == 'gt' else .8, zorder=2))
        ax.set_title(title, loc='left', fontsize=15, color=COLORS[series], pad=29, weight='semibold')
        if series == 'gt':
            length = sum(w['reference_distance_m'] for w in windows.values())
            subtitle = f'{length:.1f} m measured path'
        else:
            segments = list(world[series].values())
            if segments:
                ax.add_collection(LineCollection(segments, colors=COLORS[series], linewidths=1.6, alpha=.92, zorder=3))
                endpoints = np.asarray([s[-1] for s in segments])
                ax.scatter(endpoints[:, 0], endpoints[:, 1], s=7, c=COLORS[series], alpha=.8, linewidth=0, zorder=4)
            count = len(results[series])
            ate = np.mean([r['metrics']['ate_m'] for r in results[series].values()]) if count else np.nan
            rpe = np.mean([r['metrics']['rpe_m'] for r in results[series].values()]) if count else np.nan
            subtitle = f'{count}/{len(windows)} plans  ·  ATE {ate:.3f} m  ·  RPE {rpe:.3f} m'
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=8.8, color='#63707a')
        ax.scatter(gt[0, 0], gt[0, 1], marker='o', s=42, facecolor='white', edgecolor='#354350', lw=1.3, zorder=6)
        ax.scatter(gt[-1, 0], gt[-1, 1], marker='s', s=33, facecolor='#354350', edgecolor='white', lw=.7, zorder=6)
    axes[0].set_ylabel('y (m)', labelpad=8)
    legend = [Line2D([0], [0], color=COLORS['gt'], lw=2, label='Recorded odometry'),
        Line2D([0], [0], color=COLORS['opennwm'], lw=2, label='OpenNWM plan'),
        Line2D([0], [0], color=COLORS['nwm'], lw=2, label='NWM plan'),
        Line2D([0], [0], marker='o', color='none', markeredgecolor='#354350', markerfacecolor='white', label='Start'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='#354350', label='Finish')]
    fig.legend(handles=legend, loc='lower left', bbox_to_anchor=(.05, .065), ncol=5, frameon=False, fontsize=9)
    fig.text(.055, .043, 'Colored segments are independent 2 s plans, each anchored to the recorded pose. No robot execution.', fontsize=9, color='#52606b')
    fig.text(.055, .018, 'GT reference: robot local odometry. Translation metrics describe each local plan; they do not measure a free-running route.', fontsize=8.3, color='#7b8790')
    fig.savefig(out / f'{stem}.png', dpi=300, facecolor='white')
    fig.savefig(out / f'{stem}.svg', facecolor='white')
    plt.close(fig)
    with (out / f'{stem}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    (out / f'{stem}.json').write_text(json.dumps({
        'complete': complete, 'expected_windows_per_model': len(windows),
        'saved_windows_per_model': {m: len(v) for m, v in results.items()},
        'missing_windows': missing,
        'windows_json_sha256': digest(root / 'evaluation' / 'windows.json'),
        'coordinate_transform': 'world_xy = local_xy @ [[cos(yaw0), sin(yaw0)], [-sin(yaw0), cos(yaw0)]] + recorded_start_xy',
        'scope': 'Independent 2 s offline plans anchored to recorded starts. Segments are not concatenated into a claimed executed trajectory.',
        'gt_reference': data['reference'], 'csv_rows': len(csv_rows),
    }, indent=2) + '\n')
    print(json.dumps({'stem': str(out / stem), 'complete': complete,
                      'counts': {m: len(v) for m, v in results.items()}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--allow-partial', action='store_true', help='Write clearly named partial-preview assets using available results only.')
    parser.add_argument('--wait', action='store_true', help='Wait for all saved result files before exporting.')
    parser.add_argument('--wait-timeout', type=float, default=3600, help='Maximum wait in seconds.')
    args = parser.parse_args()
    if args.wait:
        expected = {int(w['id']) for w in json.loads((args.root / 'evaluation' / 'windows.json').read_text())['windows']}
        started, previous = time.monotonic(), None
        while True:
            found = {m: {int(p.parent.name) for p in (args.root / 'evaluation' / 'outputs' / m).glob('*/result.json')}
                     for m in ['opennwm', 'nwm']}
            counts = {m: len(expected & values) for m, values in found.items()}
            if counts != previous:
                print('Waiting for saved results:', counts, 'expected', len(expected), flush=True)
                previous = counts
            if all(expected <= values for values in found.values()):
                break
            if time.monotonic() - started > args.wait_timeout:
                raise SystemExit('Timed out waiting for full evaluation; final plots were not generated.')
            time.sleep(30)
    plot(args.root, args.allow_partial)


if __name__ == '__main__':
    main()
