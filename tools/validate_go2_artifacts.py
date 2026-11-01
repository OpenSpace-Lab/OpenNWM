#!/usr/bin/env python3
"""Independently audit saved Go2 navigation results and published media.

This consumes archived outputs without loading models or modifying source data.
ATE and translation RPE are recomputed in meters from the saved positions,
following the paper's unaligned, origin-excluded, consecutive-pose convention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def probe(path):
    value = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,pix_fmt,width,height,nb_frames,avg_frame_rate,duration:format=duration',
        '-of', 'json', str(path)], text=True))
    s = value['streams'][0]
    s['duration_s'] = float(s.get('duration', value['format']['duration']))
    return s


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def proposal_actions(parameters, horizon):
    """Reconstruct the frozen Go2 proposal contract without importing the planner.

    Both archived model configurations use XY bounds [-2.5, -4] to [5, 4]
    in waypoint units. The first step turns toward the displacement; the final
    step adds the proposal's yaw/pi adjustment. Float32 matches saved actions.
    """
    parameters = np.asarray(parameters, dtype=np.float32)
    xy = np.repeat(parameters[:, None, :2], horizon, axis=1)
    body_xy = (xy + np.float32(1)) / np.float32(2)
    body_xy = body_xy * np.array([7.5, 8], dtype=np.float32) + np.array([-2.5, -4], dtype=np.float32)
    yaw = np.zeros((*xy.shape[:2], 1), dtype=np.float32)
    yaw[:, 0, 0] = np.arctan2(body_xy[:, 0, 1], body_xy[:, 0, 0])
    yaw[:, -1, 0] += parameters[:, 2] * np.float32(np.pi)
    return np.concatenate((xy, yaw), axis=-1), np.concatenate((body_xy, yaw), axis=-1)


def validate_cem(audit, raw, protocol, horizon, label):
    """Bind CEM evidence to the exact actions archived with model predictions."""
    require(audit['gt_pose_used_for_optimization'] is False, f'{label}: GT optimization flag')
    require(len(audit['iterations']) == protocol['updates'], f'{label}: CEM update count')
    require(protocol['updates'] > 0 and 0 < protocol['elites'] <= protocol['candidates'],
            f'{label}: invalid CEM protocol')
    for iteration in audit['iterations']:
        parameters = np.asarray(iteration['candidate_parameters'], dtype=float)
        require(parameters.shape == (protocol['candidates'], 3) and np.isfinite(parameters).all(),
                f'{label}: invalid candidate parameters')
        deltas, _ = proposal_actions(parameters, horizon)
        require(np.asarray(iteration['candidate_deltas_normalized']).shape == deltas.shape,
                f'{label}: candidate action shape')
        np.testing.assert_allclose(iteration['candidate_deltas_normalized'], deltas,
                                   rtol=2e-5, atol=2e-6, err_msg=f'{label}: candidate action conversion')
        losses = np.asarray(iteration['mean_candidate_losses'], dtype=float)
        require(losses.shape == (protocol['candidates'],) and np.isfinite(losses).all(),
                f'{label}: invalid candidate losses')
        repeat_losses = np.asarray(iteration['candidate_losses_by_repeat'], dtype=float)
        require(repeat_losses.shape == (protocol['repeats'], protocol['candidates']) and
                np.isfinite(repeat_losses).all() and protocol['repeats'] > 0,
                f'{label}: invalid repeated candidate losses')
        # The stored LPIPS means may have been rounded to BF16 after reduction.
        # A single repeat has no reduction error and must be exactly preserved.
        tolerance = 0 if protocol['repeats'] == 1 else 2 ** -8
        np.testing.assert_allclose(losses, repeat_losses.mean(0), rtol=tolerance, atol=0,
                                   err_msg=f'{label}: repeated loss mean')
        elite = iteration['elite_candidate_ids']
        require(len(elite) == protocol['elites'] and len(set(elite)) == len(elite) and
                all(type(i) is int and 0 <= i < protocol['candidates'] for i in elite),
                f'{label}: elite indices')
        # BF16 LPIPS can produce ties; any tied candidate is admissible.
        np.testing.assert_allclose(np.sort(losses[elite]), np.sort(losses)[:protocol['elites']],
                                   atol=0, rtol=0, err_msg=f'{label}: elite ranking')
    final_iteration = audit['iterations'][-1]
    elite_parameters = np.asarray(final_iteration['candidate_parameters'])[final_iteration['elite_candidate_ids']]
    final = np.asarray(audit['final_parameters'], dtype=float)
    require(final.shape == (3,) and np.isfinite(final).all(), f'{label}: invalid final parameters')
    np.testing.assert_allclose(final, elite_parameters.mean(0), rtol=2e-5, atol=2e-6,
                               err_msg=f'{label}: final elite mean')
    deltas, body = proposal_actions(final[None], horizon)
    for value, expected, name in (
        (audit['final_planner_deltas'], deltas[0], 'audit final actions'),
        (raw['planner_deltas'], deltas[0], 'rollout final actions'),
        (raw['body_actions_waypoint_units'], body[0], 'rollout body actions'),
    ):
        require(np.asarray(value).shape == (horizon, 3), f'{label}: {name} shape')
        np.testing.assert_allclose(value, expected, rtol=2e-5, atol=2e-6, err_msg=f'{label}: {name}')
    np.testing.assert_allclose(raw['model_conditioning'], audit['model_conditioning'],
                               rtol=2e-5, atol=2e-6, err_msg=f'{label}: saved conditioning')
    expected_conditioning = body[0] if protocol['model'] == 'opennwm' else deltas[0]
    require(np.asarray(raw['model_conditioning']).shape == expected_conditioning.shape and
            np.asarray(audit['model_conditioning']).shape == expected_conditioning.shape,
            f'{label}: conditioning shape')
    np.testing.assert_allclose(raw['model_conditioning'], expected_conditioning,
                               rtol=2e-5, atol=2e-6, err_msg=f'{label}: conditioning adapter')


def validate(root, website=None, partial=False, evaluation_name='evaluation'):
    evaluation = root / evaluation_name
    windows = read(evaluation / 'windows.json')
    horizon = windows['horizon']
    duration = horizon / windows['fps']
    window_rows = {row['id']: row for row in windows['windows']}
    expected = {row['id'] for row in windows['windows']}
    require(len(expected) == len(windows['windows']), 'Duplicate evaluation windows')
    summary = {'expected_windows': len(expected), 'models': {}, 'media': {}}
    protocols = {}
    for model in ('opennwm', 'nwm'):
        directory = evaluation / 'outputs' / model
        protocol = read(directory / 'protocol.json')
        protocols[model] = protocol
        require(protocol['model'] == model, f'{model}: mismatched model protocol')
        require(protocol['horizon'] == horizon and protocol['fps'] == windows['fps'],
                f'{model}: mismatched window protocol')
        require(protocol['windows_sha256'] == sha256(evaluation / 'windows.json'),
                f'{model}: changed evaluation windows')
        require('dmd' not in protocol['checkpoint']['path'].lower(), f'{model}: distilled checkpoint')
        results = [read(p) for p in sorted(directory.glob('*/result.json'))]
        ids = [r['id'] for r in results]
        require(len(ids) == len(set(ids)), f'{model}: duplicate goal')
        require(set(ids) <= expected, f'{model}: unknown goals')
        if not partial:
            require(set(ids) == expected, f'{model}: incomplete coverage')
        for result in results:
            label = f'{model}/{result["id"]:06d}'
            ref = np.asarray(result['reference_xy_m'], dtype=float)
            pred = np.asarray(result['planned_xy_m'], dtype=float)
            row = window_rows[result['id']]
            require(result['model'] == model and result['horizon'] == horizon and
                    result['fps'] == windows['fps'] and
                    result['start_time_s'] == row['start_time_s'] and
                    result['end_time_s'] == row['end_time_s'], f'{label}: result identity/timing')
            np.testing.assert_allclose(ref, row['reference_xy_m'], atol=0, rtol=0, err_msg=label)
            require(result['initial_frame'] == row['frames'][0] and result['goal_frame'] == row['frames'][-1],
                    f'{label}: changed input frames')
            require(ref.shape == pred.shape == (horizon + 1, 2), f'{label}: pose shape')
            require(np.isfinite(ref).all() and np.isfinite(pred).all(), f'{label}: nonfinite poses')
            np.testing.assert_allclose(ref[0], 0, atol=1e-8, err_msg=label)
            np.testing.assert_allclose(pred[0], 0, atol=1e-8, err_msg=label)
            # Independently integrate exactly the actions used to condition the
            # video model, rather than accepting the saved trajectory as truth.
            body = np.asarray(result['body_actions_m_rad'], dtype=float)
            require(body.shape == (horizon, 3) and np.isfinite(body).all(), f'{label}: body actions')
            headings = np.r_[0., np.cumsum(body[:, 2])]
            cosine, sine = np.cos(headings[:-1]), np.sin(headings[:-1])
            world_delta = np.column_stack((cosine * body[:, 0] - sine * body[:, 1],
                                           sine * body[:, 0] + cosine * body[:, 1]))
            integrated = np.vstack((np.zeros(2), np.cumsum(world_delta, axis=0)))
            np.testing.assert_allclose(pred, integrated, rtol=2e-5, atol=2e-6, err_msg=label)
            np.testing.assert_allclose(result['planned_yaw_rad'], headings,
                                       rtol=2e-5, atol=2e-6, err_msg=label)
            # The historical paper evaluator supplies future poses only to EVO.
            error = pred[1:] - ref[1:]
            ate = np.sqrt(np.mean(np.sum(error**2, axis=1)))
            relative_error = np.diff(pred[1:], axis=0) - np.diff(ref[1:], axis=0)
            rpe = np.sqrt(np.mean(np.sum(relative_error**2, axis=1)))
            endpoint = np.linalg.norm(error[-1])
            metric = result['metrics']
            np.testing.assert_allclose([metric['ate_m'], metric['rpe_m'], metric['endpoint_m']],
                                       [ate, rpe, endpoint], rtol=2e-5, atol=2e-6, err_msg=label)
            require(all(np.isfinite(float(v)) for v in metric.values()), f'{label}: nonfinite metrics')
            output = directory / f'{result["id"]:06d}'
            raw = np.load(output / 'rollout.npz')
            rgb = raw['predicted_rgb']
            require(rgb.shape == (horizon, 3, 224, 224), f'{label}: rollout shape')
            require(np.isfinite(rgb).all(), f'{label}: nonfinite rollout')
            require(rgb.min() >= 0 and rgb.max() <= 1.0001, f'{label}: out-of-range rollout')
            np.testing.assert_allclose(raw['body_actions_waypoint_units'] *
                                       [protocol['waypoint_spacing_m'], protocol['waypoint_spacing_m'], 1],
                                       body, rtol=2e-5, atol=2e-6, err_msg=label)
            frames = sorted((output / 'predicted_frames').glob('*.png'))
            require(len(frames) == horizon, f'{label}: missing rollout PNGs')
            for frame, saved in zip(rgb, frames):
                expected_rgb = (np.clip(frame.transpose(1, 2, 0), 0, 1) * 255).round().astype('uint8')
                require(np.array_equal(np.asarray(Image.open(saved).convert('RGB')), expected_rgb),
                        f'{label}: edited or mismatched predicted frame {saved.name}')
            media = probe(output / 'rollout.mp4')
            require(abs(media['duration_s'] - duration) < .01, f'{label}: rollout duration')
            audit = read(output / 'candidate_audit.json')
            validate_cem(audit, raw, protocol, horizon, label)
        summary['models'][model] = {
            'count': len(results), 'missing': sorted(expected - set(ids)),
            'means': {k: float(np.mean([r['metrics'][k] for r in results]))
                      for k in results[0]['metrics']} if results else {},
            'checkpoint_sha256': protocol['checkpoint']['sha256'],
        }
    for field in ('seed', 'candidates', 'elites', 'horizon', 'fps', 'goal_interval_s',
                  'repeats', 'updates', 'diffusion_sampler', 'diffusion_steps',
                  'candidate_microbatch', 'windows_sha256', 'waypoint_spacing_m'):
        require(protocols['opennwm'][field] == protocols['nwm'][field], f'Unmatched protocol: {field}')
    aligned = [root / 'aligned/ego.mp4', root / 'aligned/third_person.mp4']
    values = [probe(path) for path in aligned]
    require(abs(values[0]['duration_s'] - values[1]['duration_s']) < .001, 'Aligned durations differ')
    require(values[0]['nb_frames'] == values[1]['nb_frames'], 'Aligned frame counts differ')
    summary['media']['aligned'] = {p.name: v for p, v in zip(aligned, values)}
    if website:
        manifest = read(website / 'content/go2.json')
        require(manifest['status'] == 'ready', 'Website results not ready')
        public_text = json.dumps(manifest) + (website / manifest['report']).read_text()
        require('/file_system/' not in public_text, 'Private filesystem path in public report')
        public_report = read(website / manifest['report'])
        require(len(public_report['plans']) == len(expected), 'Public report coverage mismatch')
        for model in ('opennwm', 'nwm'):
            for key in ('ate_m', 'rpe_m'):
                np.testing.assert_allclose(manifest['summary']['metrics'][model][key],
                                           summary['models'][model]['means'][key],
                                           rtol=1e-9, atol=1e-9, err_msg='Website summary mismatch')
        for clip in manifest['clips']:
            media = probe(website / clip['video'])
            require(5 <= media['duration_s'] <= 10.01, f'{clip["id"]}: duration outside 5–10s')
            require(abs(media['duration_s'] - float(clip['duration_s'])) < .05, 'Manifest duration mismatch')
            for name, path in clip['downloads'].items():
                item = probe(website / path)
                require(abs(item['duration_s'] - media['duration_s']) < .05, f'{clip["id"]}/{name}: duration mismatch')
            for model in ('opennwm', 'nwm'):
                selected = [read(evaluation / 'outputs' / model / f'{i:06d}/result.json')
                            for i in clip['evaluated_goal_ids']]
                require(len(selected) == 4, f'{clip["id"]}: expected four complete goals')
                for key in ('ate_m', 'rpe_m'):
                    np.testing.assert_allclose(clip['metrics'][model][key],
                                               np.mean([r['metrics'][key] for r in selected]),
                                               rtol=1e-9, atol=1e-9, err_msg='Website clip metric mismatch')
            summary['media'][clip['id']] = {**media, 'sha256': sha256(website / clip['video'])}
        full = probe(website / manifest['full']['video'])
        require(abs(full['duration_s'] - windows['duration_s']) < .05, 'Full comparison coverage mismatch')
        summary['media']['full'] = full
    summary['passed'] = True
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/file_system/nas/algorithm/dujun.nie/opennwm_go2_20261026'))
    parser.add_argument('--website', type=Path)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--evaluation-name', default='evaluation')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate(args.root, args.website, args.partial, args.evaluation_name)
    output = args.output or args.root / 'validation.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
