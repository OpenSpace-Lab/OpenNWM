#!/usr/bin/env python3
"""Reuse exact precomputed windows, verifying inputs and preserving provenance."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_go2_recording import ROOT, aggregate, digest, write_json


def reuse(args):
    original = args.original / 'windows.json'
    destination = args.root / args.evaluation_name
    old, new = json.loads(original.read_text()), json.loads((destination / 'windows.json').read_text())
    for key in ('horizon', 'fps', 'goal_interval_s', 'waypoint_spacing_m', 'pose_sha256'):
        if old[key] != new[key]:
            raise ValueError(f'Incompatible {key}')
    old_windows = {row['id']: row for row in old['windows']}
    selected = new['windows']
    if args.indices:
        indices = {int(value) for value in args.indices.split(',')}
        selected = [row for row in selected if row['id'] in indices]
        if {row['id'] for row in selected} != indices:
            raise ValueError('Requested reuse index absent from destination windows')
    verified = []
    hash_cache = {}
    def checked_hash(path):
        resolved = str(Path(path).resolve())
        if resolved not in hash_cache:
            hash_cache[resolved] = digest(resolved)
        return hash_cache[resolved]
    for row in selected:
        previous = old_windows[row['id']]
        for key in ('source_frame_indices', 'context_source_frame_indices'):
            if row[key] != previous[key]:
                raise ValueError(f'{row["id"]}: {key} changed')
        for key in ('start_time_s', 'end_time_s', 'source_host_times_s', 'reference_xy_m',
                    'reference_yaw_rad', 'reference_world_xy_m', 'reference_world_yaw_rad'):
            np.testing.assert_allclose(row[key], previous[key], rtol=0, atol=1e-10, err_msg=f'{row["id"]}: {key}')
        for key in ('frames', 'context_frames'):
            if len(row[key]) != len(previous[key]):
                raise ValueError(f'{row["id"]}: {key} length changed')
            for current_path, old_path in zip(row[key], previous[key]):
                if checked_hash(current_path) != checked_hash(old_path):
                    raise ValueError(f'{row["id"]}: source image differs: {current_path}')
        verified.append(row['id'])
    for model in ([args.model] if args.model else ('opennwm', 'nwm')):
        original_out, new_out = args.original / 'outputs' / model, destination / 'outputs' / model
        old_protocol = original_out / 'protocol.json'
        contract = json.loads(old_protocol.read_text())
        expected = dict(candidates=32, elites=5, updates=2, repeats=1,
                        diffusion_steps=50, diffusion_sampler='DDPM', seed=42)
        for key, value in expected.items():
            if contract[key] != value:
                raise ValueError(f'Original protocol {key} differs from requested comparison')
        contract['windows_sha256'] = digest(destination / 'windows.json')
        if args.resume_compatible:
            if contract['runner_source_sha256'] != digest(ROOT / 'tools/evaluate_go2_recording.py'):
                raise ValueError('Resume-compatible reuse requires the current exact frozen evaluator')
            if contract['planner_source_sha256'] != digest(ROOT / 'nwm/planning_eval.py'):
                raise ValueError('Resume-compatible reuse requires the current exact frozen planner')
            if 'reuse' in contract:
                raise ValueError('Cannot resume from an already rebased historical protocol')
        else:
            contract['reuse'] = dict(original_protocol=str(old_protocol), original_protocol_sha256=digest(old_protocol),
                original_windows_sha256=digest(original), method='Exact source images, poses, context, IDs and timing verified; raw artifacts symlinked unchanged.')
        existing_protocol = new_out / 'protocol.json'
        if existing_protocol.exists() and json.loads(existing_protocol.read_text()) != contract:
            raise ValueError('Refusing to change the existing destination protocol')
        write_json(new_out / 'protocol.json', contract)
        for row in selected:
            source_dir = original_out / f'{row["id"]:06d}'
            target_dir = new_out / f'{row["id"]:06d}'
            result_path = source_dir / 'result.json'
            result = json.loads(result_path.read_text())
            target_dir.mkdir(parents=True, exist_ok=True)
            for name in ('predicted_frames', 'rollout.npz', 'rollout.mp4', 'candidate_audit.json'):
                target, source = target_dir / name, source_dir / name
                if not source.exists():
                    raise FileNotFoundError(source)
                if not target.exists():
                    target.symlink_to(source)
                elif target.resolve() != source.resolve():
                    raise ValueError(f'Refusing to replace existing artifact {target}')
            result.update(horizon=new['horizon'], fps=new['fps'], goal_interval_s=new['goal_interval_s'],
                clip_ids=row.get('clip_ids', []), initial_frame=row['frames'][0], goal_frame=row['frames'][-1],
                rollout_npz=str(target_dir / 'rollout.npz'), rollout_video=str(target_dir / 'rollout.mp4'),
                candidate_audit=str(target_dir / 'candidate_audit.json'))
            result['reuse_provenance'] = dict(original_result=str(result_path), original_result_sha256=digest(result_path),
                original_protocol=str(old_protocol), original_protocol_sha256=digest(old_protocol),
                original_runner_source_sha256=contract['runner_source_sha256'],
                raw_rollout_sha256=digest(source_dir / 'rollout.npz'), inputs_verified=True,
                elapsed_s_note='Original measured generation time; no GPU inference repeated during reuse.')
            write_json(target_dir / 'result.json', result)
        args.model = model
        aggregate(args)
    write_json(destination / 'reuse_manifest.json', dict(original_evaluation=str(args.original),
        destination_evaluation=str(destination), verified_ids=verified,
        verified_source_image_count=len(hash_cache), source_images_byte_identical=True,
        original_runner_preserved_in_protocol=True, current_reuse_script_sha256=digest(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--evaluation-name', default='evaluation_2s')
    parser.add_argument('--model', choices=['opennwm', 'nwm'], help='Default: reuse both models')
    parser.add_argument('--indices', help='Reuse only these exact window IDs; default: all destination windows')
    parser.add_argument('--resume-compatible', action='store_true',
                        help='Require identical current runner/planner, allowing GPU completion of remaining windows')
    reuse(parser.parse_args())
