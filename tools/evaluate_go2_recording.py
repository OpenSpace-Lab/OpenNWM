#!/usr/bin/env python3
"""Auditable offline CEM comparison for a synchronized Go2 recording.

The existing paper planner supplies the proposal family, world-model rollouts,
LPIPS cost and EVO metrics. CEM refits raw proposal parameters to preserve yaw
units across multiple iterations. Recorded future poses are used only for
scoring. Each one- or two-second window starts from recorded context; this is an offline
navigation comparison, not a robot execution experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path('/file_system/nas/algorithm/dujun.nie/opennwm_go2_20261026')
MODEL_RUNS = {
    'opennwm': Path('/file_system/nas/algorithm/dujun.nie/nwm/opennwm/runs/imported-nav15-stage2'),
    'nwm': Path('/file_system/nas/algorithm/dujun.nie/nwm/benchmark_models/nwm-release'),
}
CHECKPOINTS = {'opennwm': 'latest', 'nwm': '0100000'}
SPACING = 0.07610250112606785


def evaluation_directory(args):
    return args.root / getattr(args, 'evaluation_name', 'evaluation')


def window_schedule(manifest, seconds, selected_clips=False):
    """Freeze windows from recording/clip times, independently of model results."""
    duration = float(manifest['aligned_duration_s'])
    clips = manifest.get('clips', []) if selected_clips else [dict(id='full_recording', start_s=0, duration_s=duration)]
    if not clips:
        raise ValueError('Selected-clip evaluation needs a nonempty manifest.clips list')
    schedule = {}
    for clip in clips:
        start, end = float(clip['start_s']), float(clip['start_s']) + float(clip['duration_s'])
        if start < -1e-6 or end > duration + 1e-6:
            raise ValueError(f'Clip outside aligned recording: {clip}')
        if abs(start / seconds - round(start / seconds)) > 1e-6:
            raise ValueError(f'Clip start {start} must lie on the {seconds:g}s planning grid')
        if selected_clips and abs((end - start) / seconds - round((end - start) / seconds)) > 1e-6:
            raise ValueError(f'Clip duration must be divisible by {seconds:g}s for complete coverage')
        for offset in np.arange(start, end - seconds + 1e-6, seconds):
            sample_id = int(round(float(offset) / seconds))
            row = schedule.setdefault(sample_id, dict(id=sample_id, offset=float(offset), clip_ids=[]))
            row['clip_ids'].append(clip['id'])
    return [schedule[key] for key in sorted(schedule)], clips


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def local_poses(xy, yaw):
    """Same right-handed local frame and wrapped yaw as nwm.datasets."""
    xy, yaw = np.asarray(xy, float), np.asarray(yaw, float)
    c, s = np.cos(yaw[0]), np.sin(yaw[0])
    local = (xy - xy[0]) @ np.array([[c, -s], [s, c]])
    relative_yaw = np.arctan2(np.sin(yaw - yaw[0]), np.cos(yaw - yaw[0]))
    return local, relative_yaw


def nearest_indices(times, requested):
    right = np.searchsorted(times, requested).clip(0, len(times) - 1)
    left = (right - 1).clip(0, len(times) - 1)
    return np.where(np.abs(times[left] - requested) <= np.abs(times[right] - requested), left, right)


def integrate_body_deltas(deltas):
    """Integrate current-body dx/dy and yaw into the initial observation frame."""
    poses = np.zeros((len(deltas) + 1, 3), dtype=float)
    for i, (dx, dy, dyaw) in enumerate(np.asarray(deltas)):
        c, s = np.cos(poses[i, 2]), np.sin(poses[i, 2])
        poses[i + 1] = poses[i] + [c * dx - s * dy, s * dx + c * dy, dyaw]
    return poses


def prepare(args):
    """Create read-only source references; no images or poses are synthesized."""
    manifest = json.loads(args.alignment.read_text())
    start = float(manifest['aligned_start_host_s'])
    duration = float(manifest['aligned_duration_s'])
    seconds = float(getattr(args, 'horizon_seconds', 2))
    horizon = int(round(seconds * 4))
    selected_clips = bool(getattr(args, 'selected_clips', False))
    schedule, clips = window_schedule(manifest, seconds, selected_clips)
    source = args.source.resolve()
    with (source / 'aligned_pose.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    origin_ns = int(rows[0]['timeline_ns'])
    times = np.array([(int(r['timeline_ns']) - origin_ns) * 1e-9 for r in rows])
    position = np.array([[float(r[k]) if r[k] else np.nan for k in ('x', 'y')] for r in rows])
    yaw = np.array([float(r['yaw']) if r['yaw'] else np.nan for r in rows])
    valid = np.array([r['pair_valid'].lower() == 'true' for r in rows]) & np.isfinite(position).all(1) & np.isfinite(yaw)
    windows, skipped = [], []
    valid_indices = np.flatnonzero(valid)
    for scheduled in schedule:
        sample, offset = scheduled['id'], scheduled['offset']
        requested = start + offset + np.arange(horizon + 1) / 4
        indices = nearest_indices(times, requested)
        # Past context can precede the display crop, but never the recording.
        context_requested = start + offset + np.arange(-3, 1) / 4
        context = nearest_indices(times, context_requested.clip(times[0], times[-1]))
        fallbacks = []
        for kind, chosen, desired in [('future', indices, requested), ('context', context, context_requested)]:
            for j in np.flatnonzero(~valid[chosen]):
                replacement = valid_indices[nearest_indices(times[valid_indices], [desired[j]])[0]]
                error = float(times[replacement] - desired[j])
                if abs(error) <= .10:
                    fallbacks.append(dict(kind=kind, sample_index=int(j),
                        rejected_source_frame_index=int(rows[chosen[j]]['frame_index']),
                        replacement_source_frame_index=int(rows[replacement]['frame_index']),
                        replacement_time_error_s=error, reason='Nearest valid measured pose within100ms; no interpolation.'))
                    chosen[j] = replacement
        reasons = []
        if not valid[indices].all():
            reasons.append('invalid_pose_pair')
        if not valid[context].all():
            reasons.append('invalid_context_pose_pair')
        if np.abs(times[indices] - requested).max() > .10:
            reasons.append('frame_sampling_gap_over_100ms')
        if reasons:
            skipped.append(dict(id=sample, start_time_s=float(offset), reasons=reasons,
                                source_frame_indices=indices.tolist()))
            continue
        local, relative_yaw = local_poses(position[indices], yaw[indices])
        frames = [str(source / rows[i]['file']) for i in indices]
        contexts = [str(source / rows[i]['file']) for i in context]
        missing = [p for p in frames + contexts if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        distance = float(np.linalg.norm(np.diff(local, axis=0), axis=1).sum())
        windows.append(dict(id=sample, start_time_s=float(offset), end_time_s=float(offset + seconds),
                            clip_ids=scheduled['clip_ids'],
                            source_start_host_s=float(start + offset),
                            requested_host_times_s=requested.tolist(),
                            source_host_times_s=times[indices].tolist(),
                            source_frame_indices=[int(rows[i]['frame_index']) for i in indices],
                            frames=frames, context_frames=contexts,
                            context_source_frame_indices=[int(rows[i]['frame_index']) for i in context],
                            context_repeated_at_recording_start=bool((context_requested < times[0]).any()),
                            invalid_pair_fallbacks=fallbacks,
                            reference_xy_m=local.tolist(), reference_yaw_rad=relative_yaw.tolist(),
                            reference_world_xy_m=position[indices].tolist(),
                            reference_world_yaw_rad=yaw[indices].tolist(),
                            reference_distance_m=distance,
                            stationary_reference=distance < .02,
                            max_frame_time_error_s=float(np.abs(times[indices] - requested).max())))
    pilot_ids = set()
    for clip in clips:
        available = [w['id'] for w in windows if clip['id'] in w['clip_ids']]
        if available:
            pilot_ids.update([available[0], available[len(available)//2]])
    output = evaluation_directory(args) / 'windows.json'
    value = dict(schema_version=2, source=str(source), alignment_manifest=str(args.alignment.resolve()),
                 alignment_sha256=digest(args.alignment), pose_sha256=digest(source / 'aligned_pose.csv'),
                 fps=4, horizon=horizon, goal_interval_s=seconds, waypoint_spacing_m=args.spacing,
                 spacing_policy='Frozen pretrained Office-Go2 scale; not fitted to this evaluation recording.',
                 reference='Go2 sport local odometry (meters/radians), not independent motion-capture ground truth.',
                 physical_synchronization_verified=False,
                 scope=f'Offline image-goal planning; recorded context resets every {seconds:g} seconds; no robot execution.',
                 selected_clips_only=selected_clips, selected_clips=clips,
                 pilot_window_ids=sorted(pilot_ids),
                 pilot_selection='First and middle complete planning window of each fixed clip; independent of model output.',
                 duration_s=duration, windows=windows, skipped_windows=skipped,
                 evaluated_duration_s=len(windows) * seconds,
                 uncovered_tail_s=float(duration - (windows[-1]['end_time_s'] if windows else 0)) if not selected_clips else None)
    write_json(output, value)
    print(json.dumps(dict(output=str(output), windows=len(windows), skipped=skipped,
                          stationary=[w['id'] for w in windows if w['stationary_reference']],
                          pilot_window_ids=sorted(pilot_ids),
                          max_frame_error_s=max((w['max_frame_time_error_s'] for w in windows), default=None)), indent=2), flush=True)


def make_evaluator(args):
    sys.path.insert(0, str(ROOT / 'nwm'))
    import torch
    import lpips
    from omegaconf import OmegaConf
    from hydra_utils import load_experiment_config
    from train_utils import setup_model, setup_tokenizer, setup_diffusion
    from misc import get_normalize, get_unnormalize, get_transform, calculate_delta_yaw, unnormalize_data, get_action_torch
    from scripts.benchmark_reproducibility import samplewise_randn
    from planning_eval import WM_Planning_Evaluator

    class RecordedPlanner(WM_Planning_Evaluator):
        def __init__(self):
            self.config = OmegaConf.merge(load_experiment_config(str(MODEL_RUNS[args.model])),
                                         OmegaConf.load(ROOT / 'nwm/conf/plan_config.yaml'))
            self.config.exp_dir = str(MODEL_RUNS[args.model])
            self.config.ckp = CHECKPOINTS[args.model]
            self.config.model.tokenizer.model_path = os.environ['NWM_VAE_PATH']
            self.config.model.diffusion.eval_timestep_respacing = args.steps
            self.config.planning_microbatch_size = args.microbatch
            self.config.num_samples = args.candidates
            self.config.topk = args.elites
            self.config.opt_steps = args.updates
            self.config.num_repeat_eval = args.repeats
            self.config.save_navigation_rollout = True
            self.config.save_planned_trajectories = False
            self.config.torch_compile = args.compile
            self.config.seed = args.seed
            self.config.planning_sample_seed = args.seed
            self.device = torch.device('cuda:0')
            self.logger = logging.getLogger('go2_evaluation')
            self.vae = setup_tokenizer(self.config, self.device).eval()
            self.model = setup_model(self.config, self.device).eval()
            checkpoint = MODEL_RUNS[args.model] / 'checkpoints' / f'{CHECKPOINTS[args.model]}.pth.tar'
            weights = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
            print(f'Strict EMA load: {self.model.load_state_dict(weights["ema"], strict=True)}', flush=True)
            del weights
            self.checkpoint = dict(path=str(checkpoint.resolve()), sha256=digest(checkpoint), weight_key='ema')
            self.parameter_count = sum(p.numel() for p in self.model.parameters())
            if args.compile:
                self.model = torch.compile(self.model)
            self.diffusion = setup_diffusion(self.config, True, self.device)
            self.latent_size = self.config.model.generator.input_size
            self.num_cond = 4
            self.mode = 'cem'
            self.num_samples, self.topk = args.candidates, args.elites
            self.opt_steps, self.num_repeat_eval = args.updates, args.repeats
            self.action_dim = 3
            self.unnormalize_fn = get_unnormalize(self.config.dataset.mean, self.config.dataset.std)
            self.transform = get_transform(self.config.dataset.image_size, self.config.dataset.mean, self.config.dataset.std)
            self.cost_fn = 'lpips'
            self.lpips = lpips.LPIPS(net='alex').to(self.device).eval()
            self.lpips_normalize = get_normalize([.5] * 3, [.5] * 3)
            self.compute_cost_with_recon = True
            self.action_stats = {k: torch.tensor(v) for k, v in self.config.dataset.action_stats.items()}
            self.motion_condition_enabled = bool(self.config.get('motion_condition', {}).get('enabled', False))
            self.audit_rollouts, self.audit_costs = [], []

        def autoregressive_rollout_without_intermediate_decoding(self, obs_image, deltas, rollout_stride, **kwargs):
            self.audit_rollouts.append(dict(keys=kwargs.get('sample_keys'), stream=kwargs.get('noise_stream'),
                                           deltas=deltas.detach().float().cpu().tolist()))
            started = time.monotonic()
            result = super().autoregressive_rollout_without_intermediate_decoding(obs_image, deltas, rollout_stride, **kwargs)
            print(f'Rollout batch={len(deltas)} seconds={time.monotonic()-started:.2f} stream={kwargs.get("noise_stream")}', flush=True)
            return result

        def compute_cost(self, *inputs):
            loss = super().compute_cost(*inputs)
            self.audit_costs.append(loss.detach().float().cpu().tolist())
            return loss

        @torch.no_grad()
        def generate_actions(self, output, dataset_name, idxs, obs_image, goal_image, gt_actions, length):
            """CEM with proposal yaw kept in its original yaw/pi units.

            The historical implementation refits from converted terminal deltas,
            which incorrectly reapplies pi at the next iteration and final plan.
            Only raw Gaussian parameters are refitted here. Future poses are not
            read: gt_actions exists solely for API compatibility.
            """
            sample_id = int(idxs.flatten()[0])
            mu, sigma = self.init_mu_sigma(obs_image, length, dataset_name)
            mu, sigma = mu[0].to(self.device), sigma[0].to(self.device)
            target_latent = self.vae.encode(goal_image.flatten(0, 1).to(self.device))
            target = self.vae.decode(target_latent, denormalize=True).float()
            target_code = self.vae.get_code_from_latent(target_latent)
            iterations = []

            def to_deltas(parameters):
                xy = parameters[:, None, :2].repeat(1, length, 1)
                yaw = calculate_delta_yaw(unnormalize_data(xy, self.action_stats))
                deltas = torch.cat((xy, yaw), dim=-1)
                deltas[:, -1, 2] += parameters[:, 2] * np.pi
                return deltas

            for iteration in range(self.opt_steps):
                keys = [f'{dataset_name}/navigation/sample={sample_id}/opt={iteration}/candidate={n}' for n in range(self.num_samples)]
                parameters = samplewise_randn(keys, (3,), base_seed=args.seed,
                    stream=f'{dataset_name}/navigation/candidates', device=self.device) * sigma + mu
                deltas = to_deltas(parameters)
                repeat_costs = []
                for repeat in range(self.num_repeat_eval):
                    costs = []
                    for begin in range(0, self.num_samples, args.microbatch):
                        end = min(begin + args.microbatch, self.num_samples)
                        count = end - begin
                        pred, code = self.autoregressive_rollout_without_intermediate_decoding(
                            obs_image.repeat(count, 1, 1, 1, 1), deltas[begin:end], 1,
                            only_final=True, return_codes=True, sample_keys=keys[begin:end],
                            noise_stream=f'{dataset_name}/navigation/sample={sample_id}/opt={iteration}/repeat={repeat}')
                        costs.append(self.compute_cost(pred[:, -1], target.repeat(count, 1, 1, 1), code,
                                                       target_code.repeat(count, 1, 1, 1)))
                    repeat_costs.append(torch.cat(costs))
                mean_cost = torch.stack(repeat_costs).mean(0)
                elite = mean_cost.argsort()[:self.topk]
                iterations.append(dict(iteration=iteration, candidate_parameters=parameters.float().cpu().tolist(),
                    parameter_units=['normalized_dx', 'normalized_dy', 'terminal_yaw_over_pi'],
                    candidate_deltas_normalized=deltas.float().cpu().tolist(),
                    candidate_losses_by_repeat=torch.stack(repeat_costs).float().cpu().tolist(),
                    mean_candidate_losses=mean_cost.float().cpu().tolist(), elite_candidate_ids=elite.cpu().tolist()))
                mu = parameters[elite].mean(0)
                sigma = parameters[elite].std(0).clamp_min(1e-5)
            final = to_deltas(mu[None])
            pred, code = self.autoregressive_rollout_without_intermediate_decoding(
                obs_image, final, 1, only_final=False, return_codes=True,
                sample_keys=[f'{dataset_name}/navigation/sample={sample_id}/final'],
                noise_stream=f'{dataset_name}/navigation/final')
            final_cost = self.compute_cost(pred[:, -1], target, code[:, -1], target_code)
            directory = Path(output) / 'navigation_rollouts' / f'{sample_id:06d}'
            directory.mkdir(parents=True, exist_ok=True)
            body = torch.cat((unnormalize_data(final[:, :, :2], self.action_stats), final[:, :, 2:]), dim=-1)
            body_np = body[0].float().cpu().numpy()
            integrated = integrate_body_deltas(body_np)
            self.last_legacy_actions = get_action_torch(final[:, :, :2], self.action_stats).float().cpu()
            np.savez_compressed(directory / 'rollout.npz', predicted_rgb=pred[0].float().cpu().numpy(),
                context_rgb=self.unnormalize_fn(obs_image[0]).float().cpu().numpy(),
                planner_deltas=final[0].float().cpu().numpy(),
                body_actions_waypoint_units=body_np,
                integrated_poses_waypoint_units=integrated,
                model_conditioning=self._real_motion_for_model(final)[0].float().cpu().numpy())
            self.last_audit = dict(iterations=iterations, final_parameters=mu.float().cpu().tolist(),
                final_planner_deltas=final[0].float().cpu().tolist(),
                final_proposal_loss=final_cost.float().cpu().tolist(),
                final_proposal='Mean of elite raw Gaussian parameters with yaw/pi units preserved',
                historical_yaw_refit_bug_corrected=True, gt_pose_used_for_optimization=False,
                model_conditioning=self._real_motion_for_model(final)[0].float().cpu().tolist(),
                model_conditioning_units=('waypoint_dx,waypoint_dy,radians' if self.motion_condition_enabled
                                          else 'normalized_dx,normalized_dy,radians'))
            return torch.tensor(integrated[None, 1:, :2], dtype=torch.float32), final[:, :, 2].sum(1)

    return RecordedPlanner()


def encode_video(frames_dir, output, frame_count=8):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        ffmpeg = '/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/nwm/bin/ffmpeg'
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-framerate', '4',
                    '-i', str(frames_dir / '%03d.png'), '-frames:v', str(frame_count), '-c:v', 'libx264',
                    '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], check=True)


def protocol(args, windows, evaluator):
    import torch
    return dict(model=args.model, checkpoint=evaluator.checkpoint, parameter_count=evaluator.parameter_count,
                runtime=dict(python_executable=sys.executable, python_version=sys.version.split()[0],
                             torch_version=torch.__version__, torch_cuda_version=torch.version.cuda),
                windows_sha256=digest(evaluation_directory(args) / 'windows.json'),
                planner_source_sha256=digest(ROOT / 'nwm/planning_eval.py'),
                runner_source_sha256=digest(__file__), seed=args.seed,
                candidates=args.candidates, elites=args.elites, horizon=windows['horizon'], fps=4, goal_interval_s=windows['goal_interval_s'],
                repeats=args.repeats, updates=args.updates, diffusion_sampler='DDPM', diffusion_steps=args.steps,
                candidate_microbatch=args.microbatch, compile=args.compile,
                vae_posterior='stochastic, inherited from paper implementation',
                cost='LPIPS AlexNet against VAE-reconstructed goal',
                candidate_prior='Frozen Office-Go2 normalized mean [-0.1,0,0], std [0.1,0.15,0.1]',
                proposal_family=f'Historical planner: constant XY delta for {windows["horizon"]} steps and terminal yaw adjustment.',
                final_action='Elite mean of raw Gaussian parameters; final stochastic rollout is rerun from this action.',
                historical_yaw_refit_correction='Refit raw terminal_yaw/pi instead of converted radians; prevents repeated pi multiplication.',
                trajectory_integration='SE(2) integration of current-body model actions; historical unrotated cumsum metrics also exported separately.',
                metric_definition=f'ATE: translation RMSE over {windows["horizon"]} future positions. RPE: adjacent-position displacement RMSE over {windows["horizon"]-1} pairs, identity pose orientations, no alignment or scale correction.',
                context=f'Four recorded past frames; autoregressive latent feedback within each {windows["goal_interval_s"]:g}-second window.',
                reference=windows['reference'], scope=windows['scope'],
                waypoint_spacing_m=windows['waypoint_spacing_m'],
                historical_reproduction=False,
                note='New held-out recording. Practical CEM budget explicitly differs from archived Table 2; same ATE/RPE metric definitions.')


def save_runtime_provenance(args, evaluator, contract):
    """Save actual resolved inference sections without environment files or secrets."""
    from omegaconf import OmegaConf
    destination = evaluation_directory(args) / 'provenance'
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in [('evaluate_go2_recording.py', Path(__file__)),
                         ('planning_eval.py', ROOT / 'nwm/planning_eval.py'),
                         ('plan_config.source.yaml', ROOT / 'nwm/conf/plan_config.yaml')]:
        temporary = destination / f'{name}.tmp.{os.getpid()}'
        shutil.copyfile(source, temporary)
        temporary.replace(destination / name)
    resolved = {key: OmegaConf.to_container(evaluator.config[key], resolve=True)
                for key in ('model', 'dataset', 'motion_condition') if key in evaluator.config}
    write_json(destination / f'{args.model}.model_runtime.resolved.json', resolved)
    write_json(destination / f'{args.model}.planning_runtime.resolved.json', contract)


def run(args, evaluator=None):
    import torch
    windows = json.loads((evaluation_directory(args) / 'windows.json').read_text())
    horizon = int(windows['horizon'])
    selected = windows['windows']
    if getattr(args, 'pilot_only', False):
        selected = [w for w in selected if w['id'] in windows['pilot_window_ids']]
    if args.indices:
        ids = set(int(v) for v in args.indices.split(','))
        selected = [w for w in selected if w['id'] in ids]
        if {w['id'] for w in selected} != ids:
            raise ValueError('Requested missing/invalid window index')
    selected = selected[args.shard_index::args.num_shards]
    if not selected:
        raise ValueError('No requested windows')
    print(f'GPU visible={os.environ.get("CUDA_VISIBLE_DEVICES")} device={torch.cuda.get_device_name(0)}', flush=True)
    torch.set_num_threads(4)
    evaluator = evaluator or make_evaluator(args)
    from misc import seed_everything
    out = evaluation_directory(args) / 'outputs' / args.model
    out.mkdir(parents=True, exist_ok=True)
    contract = protocol(args, windows, evaluator)
    protocol_file = out / 'protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != contract:
        raise ValueError('Existing output protocol differs; choose a separate --root output tree')
    write_json(protocol_file, contract)
    save_runtime_provenance(args, evaluator, contract)
    for row in selected:
        directory = out / f'{row["id"]:06d}'
        if (directory / 'result.json').exists() and (directory / 'rollout.mp4').exists():
            print(f'Skip completed {args.model}/{row["id"]}', flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        seed_everything(args.seed + row['id'])
        evaluator.audit_rollouts, evaluator.audit_costs = [], []
        obs = torch.stack([evaluator.transform(Image.open(p).convert('RGB')) for p in row['context_frames']])[None]
        goal = evaluator.transform(Image.open(row['frames'][-1]).convert('RGB'))[None, None]
        reference = np.array(row['reference_xy_m'])
        reference_yaw = np.array(row['reference_yaw_rad'])
        spacing = windows['waypoint_spacing_m']
        gt = torch.tensor(np.c_[reference[1:] / spacing, reference_yaw[1:]], dtype=torch.float32)[None]
        idxs = torch.tensor([[row['id']]], dtype=torch.float32)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            planned, planned_yaw = evaluator.generate_actions(str(directory), 'unitree_go2', idxs, obs, goal, gt, horizon)
        planned = planned.detach().float().cpu()
        measured = evaluator.eval_metrics(evaluator.actions_to_traj(gt[0, :, :2]), evaluator.actions_to_traj(planned[0]))
        legacy = evaluator.last_legacy_actions
        legacy_measured = evaluator.eval_metrics(evaluator.actions_to_traj(gt[0, :, :2]), evaluator.actions_to_traj(legacy[0]))
        legacy_endpoint = float(torch.linalg.vector_norm(legacy[0, -1] - gt[0, -1, :2]))
        endpoint = float(torch.linalg.vector_norm(planned[0, -1] - gt[0, -1, :2]))
        yaw_error = float(planned_yaw.cpu()[0]) - reference_yaw[-1]
        yaw_error = float(abs(np.arctan2(np.sin(yaw_error), np.cos(yaw_error))))
        raw = directory / 'navigation_rollouts' / f'{row["id"]:06d}' / 'rollout.npz'
        target = directory / 'rollout.npz'
        raw.replace(target)
        data = np.load(target)
        predicted = data['predicted_rgb']
        frame_dir = directory / 'predicted_frames'
        frame_dir.mkdir(exist_ok=True)
        for frame, rgb in enumerate(predicted):
            Image.fromarray((np.clip(rgb.transpose(1, 2, 0), 0, 1) * 255).round().astype('uint8')).save(frame_dir / f'{frame:03d}.png')
        encode_video(frame_dir, directory / 'rollout.mp4', horizon)
        write_json(directory / 'candidate_audit.json', evaluator.last_audit)
        planned_m = np.r_[np.zeros((1, 2)), planned[0].numpy() * spacing]
        result = dict(id=row['id'], model=args.model, start_time_s=row['start_time_s'], end_time_s=row['end_time_s'],
                      horizon=horizon, fps=4, goal_interval_s=windows['goal_interval_s'], clip_ids=row.get('clip_ids', []),
                      metrics=dict(ate=float(measured[0]), rpe_trans=float(measured[1]), pos_diff_norm=endpoint,
                                   yaw_diff_norm=yaw_error, ate_m=float(measured[0]) * spacing,
                                   rpe_m=float(measured[1]) * spacing, endpoint_m=endpoint * spacing,
                                   endpoint_yaw_deg=float(np.rad2deg(yaw_error))),
                      legacy_paper_cumsum_metrics=dict(ate=float(legacy_measured[0]), rpe_trans=float(legacy_measured[1]),
                                   pos_diff_norm=legacy_endpoint, ate_m=float(legacy_measured[0]) * spacing,
                                   rpe_m=float(legacy_measured[1]) * spacing, endpoint_m=legacy_endpoint * spacing),
                      reference_xy_m=reference.tolist(), planned_xy_m=planned_m.tolist(),
                      body_actions_m_rad=(data['body_actions_waypoint_units'] * [spacing, spacing, 1]).tolist(),
                      reference_yaw_rad=reference_yaw.tolist(),
                      planned_yaw_rad=np.r_[0, np.cumsum(data['planner_deltas'][:, 2])].tolist(),
                      reference_world_xy_m=row['reference_world_xy_m'],
                      reference_world_yaw_rad=row['reference_world_yaw_rad'],
                      elapsed_s=time.monotonic() - started, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      rollout_npz=str(target), rollout_video=str(directory / 'rollout.mp4'),
                      predicted_frame_time_offsets_s=(np.arange(1, horizon + 1)/4).tolist(),
                      initial_frame=row['frames'][0], goal_frame=row['frames'][-1],
                      candidate_audit=str(directory / 'candidate_audit.json'),
                      reference=windows['reference'], scope=windows['scope'])
        write_json(directory / 'result.json', result)
        print(json.dumps(dict(completed=row['id'], model=args.model, elapsed_s=result['elapsed_s'], metrics=result['metrics'])), flush=True)
    aggregate(args)
    return evaluator


def aggregate(args):
    windows = json.loads((evaluation_directory(args) / 'windows.json').read_text())
    out = evaluation_directory(args) / 'outputs' / args.model
    results = [json.loads(p.read_text()) for p in sorted(out.glob('*/result.json'))]
    missing = sorted({r['id'] for r in windows['windows']} - {r['id'] for r in results})
    value = dict(model=args.model, count=len(results), expected=len(windows['windows']), complete=not missing,
                 missing_indices=missing,
                 means={k: float(np.mean([r['metrics'][k] for r in results])) for k in results[0]['metrics']} if results else {},
                 total_window_seconds=sum(r['elapsed_s'] for r in results),
                 reference=windows['reference'], scope=windows['scope'])
    write_json(out / 'aggregate.json', value)
    print(json.dumps(value, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run', 'aggregate'])
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--additional-roots', type=Path, nargs='*', default=[], help='Additional episode roots, reusing the loaded model for run')
    p.add_argument('--evaluation-name', default='evaluation', help='Evaluation directory relative to each episode root')
    p.add_argument('--horizon-seconds', type=int, choices=[1, 2], default=2, help='Preparation horizon; run reads the saved windows contract')
    p.add_argument('--selected-clips', action='store_true', help='Prepare only the fixed manifest.clips ranges')
    p.add_argument('--pilot-only', action='store_true', help='Run the predetermined first/middle window of each fixed clip')
    p.add_argument('--alignment', type=Path)
    p.add_argument('--source', type=Path)
    p.add_argument('--spacing', type=float, default=SPACING)
    p.add_argument('--model', choices=list(MODEL_RUNS), default='opennwm')
    p.add_argument('--indices', help='Comma-separated window IDs; omitted means all valid windows')
    p.add_argument('--steps', type=int, default=250)
    p.add_argument('--microbatch', type=int, default=40)
    p.add_argument('--candidates', type=int, default=80)
    p.add_argument('--elites', type=int, default=5)
    p.add_argument('--updates', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    args = p.parse_args()
    if not 1 <= args.microbatch <= 80 or not 0 <= args.shard_index < args.num_shards:
        p.error('Invalid microbatch or shard configuration')
    if not 1 < args.elites <= args.candidates or min(args.updates, args.repeats) < 1:
        p.error('Invalid CEM configuration')
    if args.command == 'prepare' and args.additional_roots and (args.alignment or args.source):
        p.error('Explicit --alignment/--source apply to one root; omit them for multi-root preparation')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    evaluator = None
    explicit_alignment, explicit_source = args.alignment, args.source
    for episode_root in [args.root] + args.additional_roots:
        args.root = episode_root
        if args.command == 'prepare':
            args.alignment = explicit_alignment or episode_root / 'alignment_manifest.json'
            manifest = json.loads(args.alignment.read_text())
            args.source = explicit_source or Path(manifest.get('source_directory',
                episode_root / 'source/episode_20261026T091121_148a01b5'))
        if args.command == 'run':
            evaluator = run(args, evaluator)
        else:
            globals()[args.command](args)


if __name__ == '__main__':
    main()
