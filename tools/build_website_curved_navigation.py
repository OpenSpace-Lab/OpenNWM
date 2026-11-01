"""Independent goal-image CEM demonstrations; does not change paper evaluation.

The optimizer receives the starting RGB and goal RGB only. Recorded poses are
used for selecting turning scenes and for post-hoc error measurement, never as
proposal means, controls, optimization targets, or intermediate observations.
Run in the nwm environment. Inference outputs and complete audits live on NAS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import shutil

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path('/file_system/nas/algorithm/dujun.nie/nwm/opennwm_website_artifacts/curved_navigation_20261009')
DEFAULT_RUN = Path('/file_system/nas/algorithm/dujun.nie/nwm/opennwm/runs/imported-nav15-stage2')
DATA = Path('/file_system/nas/algorithm/dujun.nie/nwm/data')
DATASETS = {
    'recon': ('recon', 'recon', .25, .875),
    'scand': ('scand', 'scand', .38, .45),
    'huron': ('sacson', 'sacson', .255, .65),
    'tartan_drive': ('tartan', 'tartan_drive', .72, 3.125),
}
TITLES = {
    'recon-078': ('Garden bend', 'Garden bend'),
    'recon-085': ('Around the cottage', 'Around the cottage'),
    'recon-099': ('Meadow crossing', 'Meadow crossing'),
    'recon-046': ('Between the buildings', 'Between the buildings'),
    'recon-088': ('Across the garden', 'Across the garden'),
    'scand-011': ('Indoor passage', 'Indoor passage'),
    'huron-044': ('Coral corridor', 'Coral corridor'),
    'recon-061': ('Past the courtyard', 'Past the courtyard'),
    'recon-098': ('Along the garden path', 'Along the garden path'),
    'recon-020': ('Toward the hedge', 'Toward the hedge'),
    'recon-032': ('Around the outbuilding', 'Around the outbuilding'),
    'recon-044': ('Beside the long building', 'Beside the long building'),
    'recon-039': ('Between the sheds', 'Between the sheds'),
    'recon-068': ('Across the meadow', 'Across the meadow'),
    'recon-074': ('Through the tall grass', 'Through the tall grass'),
    'scand-047': ('Through the entrance', 'Through the entrance'),
    'scand-046': ('Along the sidewalk', 'Along the sidewalk'),
    'huron-053': ('Around the corridor corner', 'Around the corridor corner'),
    'huron-075': ('Into the hallway', 'Into the hallway'),
    'recon-025': ('Past the garden sheds', 'Past the garden sheds'),
    'recon-059': ('Across the courtyard', 'Across the courtyard'),
    'recon-084': ('Meadow S-bend', 'Meadow S-bend'),
    'recon-070': ('Along the field edge', 'Along the field edge'),
}
LABELS = {'recon': 'RECON', 'scand': 'SCAND', 'huron': 'HuRoN', 'tartan_drive': 'TartanDrive'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def curvature(xy):
    xy = np.asarray(xy, dtype=float)
    delta = np.diff(xy, axis=0)
    length = np.linalg.norm(delta, axis=1).sum()
    headings = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    changes = np.diff(headings)
    chord = xy[-1] - xy[0]
    distance = np.linalg.norm(chord)
    relative = xy - xy[0]
    deviation = np.abs(chord[0] * relative[:, 1] - chord[1] * relative[:, 0]).max() / max(distance, 1e-8)
    # A distinct turn needs an accumulated 6 degree change. This avoids calling
    # numerical jitter a bend while accepting a sustained multi-step curve.
    segment_turns = [float(np.rad2deg(v.sum())) for v in np.array_split(changes, 3)]
    signs = [int(np.sign(v)) for v in segment_turns if abs(v) >= 6]
    return dict(total_turn_deg=float(np.rad2deg(np.abs(changes).sum())),
                net_turn_deg=float(np.rad2deg(headings[-1] - headings[0])),
                max_step_turn_deg=float(np.rad2deg(np.abs(changes).max())),
                segment_turns_deg=segment_turns,
                turning_segments=sum(abs(v) >= 6 for v in segment_turns),
                turn_direction_changes=sum(a != b for a, b in zip(signs, signs[1:])),
                max_chord_deviation_m=float(deviation), length_m=float(length),
                chord_length_m=float(distance), deviation_ratio=float(deviation / max(length, 1e-8)))


def prepare(args):
    rows = []
    for dataset, (folder, split, spacing, _) in DATASETS.items():
        index_path = REPO / 'nwm/data_splits' / split / 'test/navigation_eval.pkl'
        index = pickle.loads(index_path.read_bytes())
        cache = {}
        for sample_id, (name, start, *_) in enumerate(index):
            directory = DATA / folder / name
            try:
                if name not in cache:
                    cache[name] = pickle.loads((directory / 'traj_data.pkl').read_bytes())
                pose = cache[name]
                xy = np.asarray(pose['position'], dtype=float)[start:start + args.horizon + 1, :2]
                yaw = np.asarray(pose['yaw'], dtype=float).reshape(-1)[start:start + args.horizon + 1]
                if len(xy) != args.horizon + 1 or not all((directory / f'{t}.jpg').is_file() for t in (start, start + args.horizon // 2, start + args.horizon)):
                    continue
                speed = np.linalg.norm(np.diff(xy, axis=0), axis=1)
                if speed.min() < .005:
                    continue
                c, s = np.cos(yaw[0]), np.sin(yaw[0])
                local = (xy - xy[0]) @ np.array([[c, -s], [s, c]])
                shape = curvature(local)
                if not (22 <= shape['total_turn_deg'] <= 180 and shape['max_step_turn_deg'] <= 35 and shape['chord_length_m'] >= .5 and shape['deviation_ratio'] >= .035):
                    continue
                rows.append(dict(id=f'{dataset}-{sample_id:03d}', dataset=dataset, sample_id=sample_id,
                    folder=folder, trajectory=name, source_dir=str(directory), source_index=str(index_path),
                    start_frame=int(start), goal_frame=int(start + args.horizon), horizon=args.horizon,
                    spacing_m=spacing, reference_xy_m=local.tolist(),
                    reference_yaw=(np.unwrap(yaw) - yaw[0]).tolist(), reference_curvature=shape,
                    mean_step_units=float(speed.mean() / spacing)))
            except (OSError, ValueError, IndexError) as exc:
                print(f'Skip {dataset}/{sample_id}: {exc}', flush=True)
    rows.sort(key=lambda row: abs(abs(row['reference_curvature']['net_turn_deg']) - 55) + max(0, row['reference_curvature']['total_turn_deg'] - 100) * .3)
    write_json(args.output / 'candidates.json', rows)
    chosen = []
    for row in rows:
        if sum(item['dataset'] == row['dataset'] for item in chosen) < 7:
            chosen.append(row)
    sheet = Image.new('RGB', (840, len(chosen) * 172), 'white')
    draw = ImageDraw.Draw(sheet)
    for n, row in enumerate(chosen):
        y = n * 172
        shape = row['reference_curvature']
        draw.text((5, y + 2), f"{row['id']} turn {shape['net_turn_deg']:.0f}/{shape['total_turn_deg']:.0f} step {row['mean_step_units']:.2f}", fill='black')
        for col, frame in enumerate([row['start_frame'], row['start_frame'] + args.horizon // 2, row['goal_frame']]):
            with Image.open(Path(row['source_dir']) / f'{frame}.jpg') as image:
                image = image.convert('RGB'); image.thumbnail((180, 140)); sheet.paste(image, (col * 185, y + 23))
        xy = np.asarray(row['reference_xy_m']); lo = xy.min(0); span = np.ptp(xy, axis=0)
        scale = min(220 / max(span[0], .01), 125 / max(span[1], .01))
        points = [(590 + (p[0] - lo[0]) * scale, y + 155 - (p[1] - lo[1]) * scale) for p in xy]
        draw.line(points, fill='green', width=3)
    sheet.save(args.output / 'candidates.jpg')
    print(json.dumps([dict(id=row['id'], turn=row['reference_curvature']['net_turn_deg'], total=row['reference_curvature']['total_turn_deg'], mean_step_units=row['mean_step_units']) for row in rows[:40]], indent=2), flush=True)


def controls_to_actions(parameters, horizon):
    """Three constant yaw-rate segments, one shared forward speed.

    Distances are model waypoint-spacing units. Arc midpoint translations are
    expressed in the *current* body frame and SE(2)-integrated for plotting.
    """
    parameters = np.asarray(parameters, dtype=float)
    batch = len(parameters)
    actions = np.zeros((batch, horizon, 3), dtype=np.float32)
    endpoints = np.zeros((batch, horizon + 1, 3), dtype=np.float64)
    boundaries = np.linspace(0, horizon, parameters.shape[1], dtype=int)
    for step in range(horizon):
        segment = min(np.searchsorted(boundaries[1:], step, side='right'), parameters.shape[1] - 2)
        yaw_delta = parameters[:, segment + 1]
        actions[:, step, 0] = parameters[:, 0] * np.cos(yaw_delta / 2)
        actions[:, step, 1] = parameters[:, 0] * np.sin(yaw_delta / 2)
        actions[:, step, 2] = yaw_delta
        yaw = endpoints[:, step, 2]
        dx, dy = actions[:, step, 0], actions[:, step, 1]
        endpoints[:, step + 1, 0] = endpoints[:, step, 0] + np.cos(yaw) * dx - np.sin(yaw) * dy
        endpoints[:, step + 1, 1] = endpoints[:, step, 1] + np.sin(yaw) * dx + np.cos(yaw) * dy
        endpoints[:, step + 1, 2] = yaw + yaw_delta
    return actions, endpoints


class Planner:
    def __init__(self, args):
        sys.path.insert(0, str(REPO / 'website/server'))
        from inference import Predictor
        self.predictor = Predictor(args.run_dir, args.steps)
        self.predictor.load()
        self.args = args
        import torch
        import lpips
        from misc import get_transform, get_normalize
        torch.set_num_threads(4)
        self.torch = torch
        self.device = self.predictor.device
        self.config = self.predictor.config
        self.transform = get_transform(224, self.config.dataset.mean, self.config.dataset.std)
        self.normalize = get_normalize(self.config.dataset.mean, self.config.dataset.std)
        self.lpips = lpips.LPIPS(net='alex').to(self.device).eval()

    def rollout(self, initial, actions, scene_id, retain=False):
        """Only the initial RGB enters the model; every later view is predicted."""
        from isolated_nwm_infer import model_forward_wrapper
        torch = self.torch
        controls = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        batch, horizon = controls.shape[:2]
        context = initial[None, None].repeat(batch, int(self.config.dataset.context_size), 1, 1, 1).to(self.device)
        frames = []
        with torch.inference_mode():
            for step in range(horizon):
                pixels = model_forward_wrapper(self.predictor.models, context, controls[:, step:step + 1], 1,
                    int(self.config.model.generator.input_size), self.device,
                    num_cond=int(self.config.dataset.context_size), motion_type='real',
                    sample_keys=[0] * batch, noise_seed=self.args.seed,
                    noise_stream=f'curved-navigation/{scene_id}/step={step}')
                pixels = pixels.clamp(0, 1)
                context = torch.cat((context[:, 1:], self.normalize(pixels).unsqueeze(1)), dim=1)
                if retain:
                    frames.append(pixels.detach().float().cpu().numpy())
                if step == 0 or step == horizon - 1:
                    print(f'  rollout step={step + 1}/{horizon} batch={batch}', flush=True)
        return pixels, np.stack(frames, axis=1) if retain else None

    def run_case(self, row):
        torch = self.torch
        args = self.args
        directory = args.output / 'runs' / row['id']
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / 'result.json').is_file() and not args.overwrite:
            print(f"Already complete: {row['id']}", flush=True)
            return
        started = time.monotonic()
        source = Path(row['source_dir'])
        with Image.open(source / f"{row['start_frame']}.jpg") as image:
            initial = self.transform(image.convert('RGB'))
        with Image.open(source / f"{row['goal_frame']}.jpg") as image:
            goal = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
        # Reconstructed target matches the model output image domain. It is
        # never supplied as context or used to extract a ground-truth action.
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vae = self.predictor.models[2]
            target = vae.decode(vae.encode(goal), denormalize=True).clamp(0, 1)
        prior_speed = DATASETS[row['dataset']][3]
        mu = np.array([prior_speed, 0., 0., 0.])
        sigma = np.array([prior_speed * .40, .09, .09, .09])
        rng = np.random.default_rng(args.seed)
        best = None
        audit = []
        for iteration in range(args.iterations):
            parameters = rng.normal(mu, sigma, size=(args.population, 4))
            parameters[:, 0] = np.clip(parameters[:, 0], prior_speed * .25, prior_speed * 2.2)
            parameters[:, 1:] = np.clip(parameters[:, 1:], -.20, .20)
            parameters[0] = mu
            if best is not None:
                parameters[1] = best['parameters']
            actions, endpoints = controls_to_actions(parameters, row['horizon'])
            losses = []
            for start in range(0, args.population, args.batch_size):
                stop = min(start + args.batch_size, args.population)
                pred, _ = self.rollout(initial, actions[start:stop], row['id'])
                with torch.inference_mode():
                    loss = self.lpips(pred.float() * 2 - 1, target.float().repeat(stop - start, 1, 1, 1) * 2 - 1).flatten()
                losses.extend(loss.cpu().tolist())
            order = np.argsort(losses)
            elite = parameters[order[:args.elites]]
            mu = elite.mean(0)
            sigma = np.maximum(elite.std(0), np.array([prior_speed * .06, .016, .016, .016]))
            if best is None or losses[order[0]] < best['loss']:
                best = dict(parameters=parameters[order[0]].tolist(), loss=losses[order[0]])
            audit.append(dict(iteration=iteration, parameters=parameters.tolist(), losses=losses,
                              elite_indices=order[:args.elites].tolist(), mean=mu.tolist(), std=sigma.tolist()))
            write_json(directory / 'optimization.json', audit)
            print(json.dumps(dict(scene=row['id'], iteration=iteration, best=best, elapsed_seconds=time.monotonic() - started)), flush=True)
        actions, endpoints = controls_to_actions([best['parameters']], row['horizon'])
        final_pixels, frames = self.rollout(initial, actions, row['id'], retain=True)
        with torch.inference_mode():
            final_loss = float(self.lpips(final_pixels.float() * 2 - 1, target.float() * 2 - 1).item())
        initial_pixels = ((initial * .5 + .5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
        Image.fromarray(initial_pixels).save(directory / 'start.webp', lossless=True)
        goal_pixels = ((goal[0].cpu() * .5 + .5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
        Image.fromarray(goal_pixels).save(directory / 'goal.webp', lossless=True)
        frame_dir = directory / 'frames'; frame_dir.mkdir(exist_ok=True)
        Image.fromarray(initial_pixels).save(frame_dir / '000.png')
        for index, frame in enumerate(frames[0], 1):
            pixels = (frame.transpose(1, 2, 0).clip(0, 1) * 255).astype('uint8')
            Image.fromarray(pixels).save(frame_dir / f'{index:03d}.png')
        np.savez_compressed(directory / 'rollout.npz', predicted_rgb=frames[0], actions_body_waypoint_units=actions[0], poses_origin_waypoint_units=endpoints[0])
        planned = endpoints[0, :, :2] * row['spacing_m']
        reference = np.asarray(row['reference_xy_m'])
        errors = np.linalg.norm(planned[1:] - reference[1:], axis=1)
        result = dict(**row, planned_xy_m=planned.tolist(), planned_curvature=curvature(planned),
            actions_body_waypoint_units=actions[0].tolist(), poses_origin_waypoint_units=endpoints[0].tolist(),
            parameters=best['parameters'], search_goal_loss=best['loss'], final_goal_loss=final_loss,
            search_to_final_loss_delta=final_loss - best['loss'], ate_m=float(np.sqrt(np.mean(errors ** 2))),
            goal_error_m=float(errors[-1]), normalized_goal_error=float(errors[-1] / max(np.linalg.norm(reference[-1]), .01)),
            elapsed_seconds=time.monotonic() - started,
            checkpoint=self.predictor.checkpoint,
            protocol=dict(planner='independent piecewise SE(2) goal-image CEM demo',
                input='single start RGB repeated four times and goal RGB; no recorded actions or later observations',
                objective='LPIPS between final model prediction and VAE-reconstructed goal RGB',
                proposal='one speed and three constant yaw-rate segments; dataset-only speed prior',
                feedback='autoregressive decoded RGB, deterministic VAE posterior mean',
                coordinates='each action is current-body dx,dy in dataset waypoint units,dyaw radians; exact SE(2) integration, multiplied by metric waypoint spacing for meters',
                seed=args.seed, population=args.population, elites=args.elites, iterations=args.iterations,
                diffusion_steps=args.steps, batch_size=args.batch_size, horizon=row['horizon'], fps=4,
                shared_candidate_noise=True, ground_truth_usage='scene selection and post-hoc errors only',
                scope='qualitative offline imagined planning; no execution or success rate; separate from paper benchmark'))
        write_json(directory / 'result.json', result)
        render_review(directory, result)
        print(json.dumps({k: result[k] for k in ['id', 'goal_error_m', 'normalized_goal_error', 'planned_curvature', 'elapsed_seconds']}), flush=True)


def render_review(directory, row):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    horizon = row['horizon']
    indices = np.linspace(0, horizon, 9, dtype=int)
    fig, axes = plt.subplots(3, 9, figsize=(18, 7))
    for col, step in enumerate(indices):
        for line, path in [(0, Path(row['source_dir']) / f"{row['start_frame'] + step}.jpg"), (1, directory / 'frames' / f'{step:03d}.png')]:
            with Image.open(path) as image:
                axes[line, col].imshow(image)
            axes[line, col].axis('off')
        axes[0, col].set_title(f'{step / 4:g} s')
        axes[2, col].axis('off')
    axes[0, 0].text(-.05, .5, 'Recorded', transform=axes[0, 0].transAxes, rotation=90, va='center')
    axes[1, 0].text(-.05, .5, 'Planned', transform=axes[1, 0].transAxes, rotation=90, va='center')
    ax = fig.add_axes([.4, .035, .22, .25])
    for key, color, label in [('reference_xy_m', '#777', 'Recorded'), ('planned_xy_m', '#1a8266', 'OpenNWM')]:
        xy = np.asarray(row[key]); ax.plot(xy[:, 0], xy[:, 1], marker='.', color=color, label=label)
    ax.set_aspect('equal', adjustable='datalim'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle(f"{row['id']} | error {row['goal_error_m']:.2f}m ({row['normalized_goal_error']:.0%}) | turn {row['planned_curvature']['total_turn_deg']:.0f} deg")
    fig.savefig(directory / 'review.jpg', dpi=100, bbox_inches='tight'); plt.close(fig)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_final_losses(args):
    """Measure saved final RGB separately from the search batch predictions."""
    planner = Planner(args)
    torch = planner.torch
    keys = args.ids or [path.parent.name for path in sorted((args.output / 'runs').glob('*/result.json'))]
    for key in keys:
        directory = args.output / 'runs' / key
        row = json.loads((directory / 'result.json').read_text())
        with Image.open(Path(row['source_dir']) / f"{row['goal_frame']}.jpg") as image:
            goal = planner.transform(image.convert('RGB')).unsqueeze(0).to(planner.device)
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            vae = planner.predictor.models[2]
            target = vae.decode(vae.encode(goal), denormalize=True).clamp(0, 1)
        data = np.load(directory / 'rollout.npz')
        final = torch.as_tensor(data['predicted_rgb'][-1:]).to(planner.device)
        with torch.inference_mode():
            loss = float(planner.lpips(final.float() * 2 - 1, target.float() * 2 - 1).item())
        row['search_goal_loss'] = row.pop('lpips_goal_loss', row.get('search_goal_loss'))
        row['final_goal_loss'] = loss
        row['search_to_final_loss_delta'] = loss - row['search_goal_loss']
        row['planned_curvature'] = curvature(row['planned_xy_m'])
        row['reference_curvature'] = curvature(row['reference_xy_m'])
        row['final_loss_audit'] = 'Recomputed from the archived raw float32 final RGB, not the search batch; same deterministic VAE-reconstructed goal and LPIPS alex.'
        write_json(directory / 'result.json', row)
        print(json.dumps({name: row[name] for name in ['id', 'search_goal_loss', 'final_goal_loss', 'search_to_final_loss_delta']}), flush=True)


def publish(args):
    """Export explicitly selected, visually reviewed examples without inference."""
    if not args.ids:
        raise ValueError('Select reviewed IDs explicitly with --ids')
    samples = []
    checkpoints = set()
    source_map_path = args.output / 'inference_runs.json'
    source_map = json.loads(source_map_path.read_text()) if source_map_path.is_file() else {}
    inference_sources = {}
    for key in args.ids:
        source = args.output / 'runs' / key
        row = json.loads((source / 'result.json').read_text())
        # Guard against accidentally publishing a good-looking straight path,
        # corrupt action integration, or an implausibly remote final endpoint.
        planned = np.asarray(row['planned_xy_m'])
        shape = curvature(planned)
        assert shape['turning_segments'] >= 2 and shape['total_turn_deg'] >= 25
        assert row['normalized_goal_error'] <= .35
        assert row['goal_error_m'] <= 1.5 and row['ate_m'] <= 1.0
        actions, poses = controls_to_actions([row['parameters']], row['horizon'])
        np.testing.assert_allclose(planned, poses[0, :, :2] * row['spacing_m'], atol=1e-7)
        np.testing.assert_allclose(actions[0], row['actions_body_waypoint_units'], atol=1e-7)
        slug = f'curved-{key}'
        target = REPO / 'website/assets/navigation' / slug
        target.mkdir(parents=True, exist_ok=True)
        for name in ('start.webp', 'goal.webp'):
            shutil.copyfile(source / name, target / name)
        with Image.open(source / 'frames' / f"{row['horizon'] // 2:03d}.png") as image:
            image.save(target / 'poster.webp', lossless=True)
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', '4',
            '-i', str(source / 'frames/%03d.png'), '-frames:v', str(row['horizon'] + 1),
            '-c:v', 'libx264', '-crf', '16', '-preset', 'medium', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', '-an', str(target / 'prediction.mp4')], check=True)
        title = TITLES[key]
        checkpoints.add(row['checkpoint']['sha256'])
        distance = np.linalg.norm(np.asarray(row['reference_xy_m'])[-1])
        assert row['ate_m'] / max(distance, .01) <= .25
        source_script = source_map.get(key, 'inference_source.py')
        source_hash = digest(args.output / source_script)
        inference_sources[source_script] = source_hash
        sample = {name: row[name] for name in ('sample_id', 'trajectory', 'start_frame', 'goal_frame',
            'reference_xy_m', 'planned_xy_m', 'ate_m', 'goal_error_m', 'normalized_goal_error',
            'reference_curvature', 'planned_curvature', 'actions_body_waypoint_units',
            'poses_origin_waypoint_units', 'parameters', 'search_goal_loss', 'final_goal_loss',
            'search_to_final_loss_delta', 'protocol')}
        sample.update(id=slug, dataset=LABELS[row['dataset']], title=dict(en=title[0], zh=title[1]),
            start=f'assets/navigation/{slug}/start.webp', goal=f'assets/navigation/{slug}/goal.webp',
            poster=f'assets/navigation/{slug}/poster.webp', video=f'assets/navigation/{slug}/prediction.mp4',
            normalized_ate=row['ate_m'] / max(distance, .01), waypoint_spacing_m=row['spacing_m'],
            fps=4, frame_count=row['horizon'] + 1, duration_seconds=(row['horizon'] + 1) / 4,
            selection='Manually reviewed all predicted frames for scene consistency; selected real multi-segment turns and low post-hoc endpoint error.',
            final_loss_audit=row.get('final_loss_audit', 'Evaluated on the final retained single-candidate rollout RGB, separately from search-batch predictions; deterministic VAE-reconstructed goal and LPIPS alex.'),
            inference_source=source_script, inference_source_sha256=source_hash,
            source_archive=args.output.name, source_example=f'runs/{key}',
            source_frame_sha256={f'prediction/{index:03d}.png': digest(source / 'frames' / f'{index:03d}.png') for index in range(row['horizon'] + 1)},
            source_observation_sha256={name: digest(Path(row['source_dir']) / f"{row[field]}.jpg") for name, field in [('start', 'start_frame'), ('goal', 'goal_frame')]},
            source_result_sha256=digest(source / 'result.json'),
            source_optimization_sha256=digest(source / 'optimization.json'),
            source_index_sha256=digest(Path(row['source_index'])),
            rollout_sha256=digest(source / 'rollout.npz'),
            asset_sha256={path.name: digest(path) for path in sorted(target.iterdir())})
        samples.append(sample)
    assert len(checkpoints) == 1
    completed = list((args.output / 'runs').glob('*/result.json'))
    manifest = dict(version=3, model='OpenNWM Nav15 finalLAM joint100k',
        checkpoint_sha256=checkpoints.pop(), archive=args.output.name,
        inference_source_sha256=digest(args.output / 'inference_source.py'),
        inference_sources=inference_sources,
        export_source_sha256=digest(Path(__file__)),
        scope='Selected offline goal-image-conditioned sequence plans; independently generated qualitative demonstrations, separate from paper benchmarks. No closed-loop execution or success rate measured.',
        protocol=samples[0]['protocol'],
        media=dict(fps=4, resolution=[224, 224], frame_zero='Recorded start', interpolation=False),
        selection=dict(completed_candidate_count=len(completed), website_selected_count=len(samples),
            criterion='At least two turning segments of six degrees, total turn >=25 degrees, endpoint error <=1.5 m and <=0.35 normalized, ATE <=1 m and <=0.25 normalized, and manual visual review of the full prediction.',
            note='Curves are exact SE(2) integrations of the optimized model controls; no trajectory smoothing or ground-truth path substitution.'),
        samples=samples)
    payload = json.dumps(manifest, indent=2, ensure_ascii=False) + '\n'
    assert '/file_system/' not in payload
    (REPO / 'website/content/navigation.json').write_text(payload)
    print(json.dumps(dict(published=args.ids, samples=len(samples))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'publish', 'audit'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--ids', nargs='+', default=[])
    parser.add_argument('--horizon', type=int, default=16)
    parser.add_argument('--population', type=int, default=32)
    parser.add_argument('--elites', type=int, default=6)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'publish':
        publish(args)
    elif args.command == 'audit':
        audit_final_losses(args)
    else:
        rows = json.loads((args.output / 'candidates.json').read_text())
        selected = [next(row for row in rows if row['id'] == key) for key in args.ids]
        if not selected:
            parser.error('Select candidate IDs with --ids')
        print(json.dumps(dict(pid=os.getpid(), cwd=str(Path.cwd()), argv=sys.argv, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))), flush=True)
        planner = Planner(args)
        for row in selected:
            planner.run_case(row)


if __name__ == '__main__':
    main()
