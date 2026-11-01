#!/usr/bin/env python3
"""Prepare disjoint 4-context/16-future windows from MarsLab PhysX recordings.

The action is realized body displacement, not the commanded wheel velocity.
One translation scale is fitted on training trajectories and shared by all splits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def planar_poses(frames):
    positions = np.asarray([f['position_m'] for f in frames], dtype=np.float64)
    quaternions = np.asarray([f['quaternion_wxyz'] for f in frames], dtype=np.float64)
    if positions.shape != (len(frames), 3) or quaternions.shape != (len(frames), 4):
        raise ValueError('Malformed position/quaternion')
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError('Nonfinite pose')
    if np.max(np.abs(np.linalg.norm(quaternions, axis=1) - 1)) > 1e-3:
        raise ValueError('Non-unit quaternion')
    w, x, y, z = quaternions.T
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return np.column_stack([positions[:, :2], yaw])


def relative_actions(poses, translation_scale):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (20, 3) or not np.isfinite(poses).all():
        raise ValueError('Expected 20 finite planar poses')
    if not np.isfinite(translation_scale) or translation_scale <= 0:
        raise ValueError('Invalid translation scale')
    delta = (poses[4:, :2] - poses[3, :2]) * translation_scale
    c, s = np.cos(poses[3, 2]), np.sin(poses[3, 2])
    return np.column_stack([c*delta[:, 0] + s*delta[:, 1],
                            -s*delta[:, 0] + c*delta[:, 1],
                            (poses[4:, 2] - poses[3, 2] + np.pi) % (2*np.pi) - np.pi])


def prepare(recordings, specs_path, output, nominal_step=.25):
    specs = json.loads(specs_path.read_text())
    episodes = specs['trajectories']
    if len({e['id'] for e in episodes}) != len(episodes):
        raise ValueError('Duplicate trajectory IDs')
    if set(e['split'] for e in episodes) != {'train', 'val', 'test'}:
        raise ValueError('Plan requires train, val, and test trajectories')
    loaded, train_steps, provenance, all_paths = [], [], [], set()
    scene_hashes = set()
    for episode in episodes:
        path = recordings / episode['id'] / 'trajectory.json'
        trajectory = json.loads(path.read_text())
        if trajectory.get('status') != 'complete':
            raise ValueError(f'Incomplete trajectory: {path}')
        if trajectory['trajectory_id'] != episode['id'] or trajectory['split'] != episode['split']:
            raise ValueError('Trajectory identity or split differs from frozen plan')
        metadata = trajectory['metadata']
        warmup = metadata.get('render_warmup', {})
        if (metadata.get('render_warmup_frames') != 128
                or warmup.get('requested_render_frames') != 128
                or warmup.get('render_updates', 0) < 128
                or warmup.get('physics_state_unchanged') is not True):
            raise ValueError('Missing validated v2 rendering warmup; v1 recordings are excluded')
        for before, after in [('simulation_time_before', 'simulation_time_after'),
                              ('physics_step_before', 'physics_step_after'),
                              ('body_position_before_m', 'body_position_after_m'),
                              ('body_quaternion_before_wxyz', 'body_quaternion_after_wxyz')]:
            values = np.asarray(warmup[before], dtype=float)
            if not np.isfinite(values).all() or not np.array_equal(values, np.asarray(warmup[after], dtype=float)):
                raise ValueError('Rendering warmup changed physics state')
        if (not np.isfinite(warmup['camera_timestamp'])
                or abs(warmup['camera_timestamp'] - warmup['simulation_time_after']) > 1e-5):
            raise ValueError('Rendering warmup camera timestamp mismatch')
        if metadata['episode_spec'] != episode or metadata['specs_sha256'] != sha256(specs_path):
            raise ValueError('Recorded episode differs from frozen plan')
        frames = trajectory['frames']
        if trajectory['fps'] != 4 or len(frames) != round(episode['duration_s'] * 4) or len(frames) % 20:
            raise ValueError('Expected 4 Hz recordings with complete nonoverlapping 20-frame windows')
        times = np.asarray([f['timestamp'] for f in frames])
        camera_times = np.asarray([f['camera_timestamp'] for f in frames])
        if not np.isfinite(times).all() or not np.isfinite(camera_times).all():
            raise ValueError('Nonfinite timestamp')
        if not np.allclose(np.diff(times), .25, atol=1e-5, rtol=0) or np.max(np.abs(times-camera_times)) > 1e-5:
            raise ValueError('Camera/pose cadence or timestamp mismatch')
        if [f['index'] for f in frames] != list(range(len(frames))):
            raise ValueError('Frame indices are not contiguous')
        poses = planar_poses(frames)
        steps = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
        if np.median(steps) <= 1e-4 or np.max(steps) > 1.0:
            raise ValueError(f'Stationary or discontinuous recording: {episode["id"]}')
        paths = [(path.parent / f['path']).resolve() for f in frames]
        if len(set(paths)) != len(paths) or any(p in all_paths for p in paths):
            raise ValueError('Images reused across trajectories')
        for image_path in paths:
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
        all_paths.update(paths)
        if episode['split'] == 'train':
            train_steps.extend(steps.tolist())
        scene_hashes.add(metadata['scene_sha256'])
        provenance.append({'id': episode['id'], 'split': episode['split'], 'trajectory_path': str(path),
                           'trajectory_sha256': sha256(path), 'commands_sha256': sha256(path.parent / 'commands.jsonl'),
                           'scene_sha256': metadata['scene_sha256'], 'recorder_sha256': metadata['recorder_sha256'],
                           'compatibility_sha256': metadata['compatibility_sha256'],
                           'frames': len(frames), 'planar_path_length_m': float(steps.sum()),
                           'median_step_m': float(np.median(steps)),
                           'max_sync_error_seconds': float(np.max(np.abs(times-camera_times)))})
        loaded.append((episode, frames, paths, poses))
    if len(scene_hashes) != 1:
        raise ValueError('This protocol expects one shared scene')
    scale = float(nominal_step / np.median(train_steps))
    samples = []
    for episode, frames, paths, poses in loaded:
        split = 'same_scene_test' if episode['split'] == 'test' else episode['split']
        for start in range(0, len(frames), 20):
            window = paths[start:start+20]
            actions_m = relative_actions(poses[start:start+20], 1)
            samples.append({
                'id': f'marslab-{episode["id"]}-{start:04d}', 'dataset': 'MarsLab', 'split': split,
                'scene': 'main_crater', 'source_scene': 'main_crater', 'trajectory_id': episode['id'],
                'protocol': 'direct', 'frame_indices': list(range(start, start+20)),
                'context_source_paths': [str(p) for p in window[:4]],
                'gt_source_paths': [str(p) for p in window[4:]],
                'prediction_steps': list(range(1, 17)),
                'actions': relative_actions(poses[start:start+20], scale).tolist(),
                'actions_meters_radians': actions_m.tolist(),
                'command_vw': [f['command_vw'] for f in frames[start:start+20]],
                'timestamps': [f['timestamp'] for f in frames[start:start+20]],
                'action_source': 'MarsLab / Isaac Sim PhysX body ground truth',
                'action_units': 'training-calibrated translation units; yaw radians',
                'source_fps': 4, 'gallery_selected': split == 'same_scene_test' and start == 0,
                'source_sha256': {str(p): sha256(p) for p in window},
            })
    limitations = [
        'Independent held-out trajectories share the main_crater terrain; this is not cross-scene generalization.',
        'Actions are realized PhysX body displacements, not a claim that commanded velocity exactly equals traveled motion.',
        'A single translation scale fitted only on training poses is shared by validation and test; metric poses and commands are retained.',
        'The SE(2) action interface discards vertical motion, pitch, and roll.',
        'MarsLab scene and rover assets are run with a minimal Isaac Sim 4.1 recorder, not the complete official Isaac Sim 5.x application.',
    ]
    data = {'schema_version': 1, 'dataset': 'MarsLab', 'specs_sha256': sha256(specs_path),
            'preparation_sha256': sha256(__file__), 'trajectory_provenance': provenance,
            'data_metadata': {'MarsLab': {'protocol': 'paired direct prediction with simulator ground-truth actions', 'limitations': limitations}},
            'translation_scale': {'model_units_per_meter': scale, 'training_median_step_m': float(np.median(train_steps)),
                                  'nominal_step': nominal_step, 'fit_splits': ['train']},
            'split_policy': 'independent trajectories; nonoverlapping 20-frame windows; test first windows selected before training',
            'gallery_sample_ids': [s['id'] for s in samples if s['gallery_selected']],
            'limitations': limitations, 'samples': samples}
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'data_manifest.json').exists():
        raise FileExistsError('Refusing to overwrite a completed manifest')
    write_json(output / 'data_manifest.json', data)
    write_json(output / 'eval_manifest.json', {**data, 'samples': [s for s in samples if s['split'] == 'same_scene_test']})
    print(json.dumps({'status': 'complete', 'samples': dict(Counter(s['split'] for s in samples)),
                      'translation_scale': data['translation_scale'], 'gallery_sample_ids': data['gallery_sample_ids']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', type=Path, required=True)
    parser.add_argument('--specs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nominal-step', type=float, default=.25)
    args = parser.parse_args()
    if not np.isfinite(args.nominal_step) or args.nominal_step <= 0:
        parser.error('--nominal-step must be finite and positive')
    prepare(args.recordings.resolve(), args.specs.resolve(), args.output.resolve(), args.nominal_step)
