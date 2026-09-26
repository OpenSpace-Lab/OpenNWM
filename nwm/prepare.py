#!/usr/bin/env python3
"""Prepare resumable trajectory caches from a NavAnywhere recipe or NWM splits.

Each completed trajectory is atomically saved; rerunning skips its file. No
whole-dataset preflight scan or model-file checksum is needed.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import pickle
import sys
import numpy as np
import torch
from PIL import Image
from checkpoint_io import atomic_save
from misc import get_transform
from navanywhere_recipe import load_sampling_recipe, scan_trajectory_frames, frame_indices_sha256

DATASETS = {'recon': 'recon', 'sacson': 'sacson', 'scand': 'scand', 'tartan_drive': 'tartan'}


def tasks(args):
    if args.recipe:
        recipe, _, _ = load_sampling_recipe(args.recipe)
        for row in recipe['trajectories']:
            source, trajectory = row['source_id'], row['trajectory_id']
            frames = scan_trajectory_frames(args.data_root/source/trajectory)
            yield source, trajectory, frames, True
    else:
        for source in args.datasets.split(','):
            seen = set()
            for split in args.splits.split(','):
                names = args.split_root/source/split/'traj_names.txt'
                for trajectory in names.read_text().splitlines():
                    trajectory = trajectory.strip()
                    if not trajectory or trajectory in seen:
                        continue
                    seen.add(trajectory)
                    root = args.data_root/DATASETS[source]/trajectory
                    with (root/'traj_data.pkl').open('rb') as stream:
                        count = len(pickle.load(stream)['position'])
                    yield source, trajectory, [(i, str(root/f'{i}.jpg')) for i in range(count)], False


def local_pairs(indices, navanywhere):
    available = set(map(int, indices))
    # Exact local teacher windows used by the original two preparation scripts.
    currents = indices[3:] if navanywhere else indices[3:max(3, len(indices)-64)]
    return np.asarray([(int(current), target) for current in currents
                       for target in range(int(current)-8, int(current)+9)
                       if target in available], dtype=np.int64).reshape(-1, 2)


def encode_vae(vae, frame_paths, args, device):
    transform = get_transform(224)
    means, logvars = [], []
    for start in range(0, len(frame_paths), args.batch_size):
        images = []
        for path in frame_paths[start:start+args.batch_size]:
            with Image.open(path) as image:
                images.append(transform(image.convert('RGB')))
        valid = len(images)
        batch = torch.stack(images)
        if valid < args.batch_size:
            batch = torch.cat([batch, batch[-1:].expand(args.batch_size-valid, -1, -1, -1)])
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            posterior = vae.encode(batch.to(device)).latent_dist
            means.append(posterior.mean[:valid].to(device='cpu', dtype=torch.bfloat16))
            logvars.append(posterior.logvar[:valid].clamp(-30, 20).to(device='cpu', dtype=torch.bfloat16))
    return torch.cat(means).contiguous(), torch.cat(logvars).contiguous()


def encode_lam(adapter, frames, pairs, args, load_frame):
    paths = dict(frames)
    hparams = adapter.checkpoint_metadata['hparams']
    height, width = int(hparams.get('image_height', 240)), int(hparams.get('image_width', 320))
    outputs = []
    dtype = {'bf16-mixed': torch.bfloat16, '16-mixed': torch.float16}.get(args.precision)
    # Bound CPU/GPU frame memory by a chunk while preserving original pair order.
    for start in range(0, len(pairs), 4096):
        chunk = pairs[start:start+4096]
        indices, inverse = np.unique(chunk, return_inverse=True)
        bank = torch.stack([load_frame(paths[int(i)], image_height=height, image_width=width) for i in indices])
        positions = torch.as_tensor(inverse.reshape(-1, 2), device=adapter.device)
        bank = bank.to(adapter.device)
        for offset in range(0, len(chunk), args.batch_size):
            selected = positions[offset:offset+args.batch_size]
            video = torch.stack((bank[selected[:, 0]], bank[selected[:, 1]]), dim=1)
            autocast = torch.autocast(adapter.device.type, dtype=dtype) if dtype else nullcontext()
            with torch.inference_mode(), autocast:
                latent = adapter.encode(video)
            outputs.append(latent[:, 0, 0].detach().float().cpu())
    dim = int(hparams['lam_latent_dim'])
    return torch.cat(outputs).contiguous() if outputs else torch.empty((0, dim))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['vae', 'lam'])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, help='NavAnywhere recipe; omit for NWM data')
    parser.add_argument('--split-root', type=Path, default=Path(os.environ.get('NWM_SPLIT_ROOT', 'data_splits')))
    parser.add_argument('--splits', default='train')
    parser.add_argument('--datasets', default=','.join(DATASETS))
    parser.add_argument('--vae-model', default='stabilityai/sd-vae-ft-ema')
    parser.add_argument('--checkpoint', type=Path, help='LAM checkpoint for lam mode')
    parser.add_argument('--batch-size', type=int, help='Defaults: VAE 128, LAM 4')
    parser.add_argument('--precision', choices=['32', 'bf16-mixed', '16-mixed'], default='bf16-mixed')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.batch_size = args.batch_size or (128 if args.kind == 'vae' else 4)
    if args.kind == 'lam' and args.checkpoint is None:
        parser.error('--checkpoint is required for lam')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    model = None
    manifests = {}
    metadata = {'format': 'sd_vae_posterior_stats' if args.kind == 'vae' else 'lam_action_proxy',
                'schema_version': 1, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    metadata_path = args.output_root/'metadata.json'
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous.get('arguments') != metadata['arguments']:
            raise ValueError('Cache arguments differ; use another output root')
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    for source, trajectory, frames, navanywhere in tasks(args):
        output = args.output_root/source/f'{trajectory}.pt'
        manifest = args.output_root/source/'manifest.jsonl'
        if source not in manifests:
            manifests[source] = {json.loads(line)['trajectory_name'] for line in manifest.read_text().splitlines()} if manifest.exists() else set()
        if output.exists() and trajectory in manifests[source]:
            continue
        if model is None:
            if args.kind == 'vae':
                from diffusers import AutoencoderKL
                model = AutoencoderKL.from_pretrained(args.vae_model).eval().to(device)
                model.requires_grad_(False)
            else:
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'lam'))
                from lam.navigation_evaluation.adapters import load_evaluation_adapter
                from lam.navigation_evaluation.catalog import load_navigation_frame
                model = load_evaluation_adapter(args.checkpoint, device=device, fused_attention=True)
        indices = np.asarray([item[0] for item in frames], dtype=np.int64)
        record = {'trajectory_name': trajectory, 'trajectory_id': trajectory, 'source_id': source,
                  'num_frames': len(frames), 'frame_count': len(frames), 'frame_indices_sha256': frame_indices_sha256(indices)}
        if args.kind == 'vae':
            mean, logvar = encode_vae(model, [item[1] for item in frames], args, device)
            payload = {'schema_version': 1, 'format': 'sd_vae_posterior_stats',
                       'dataset_name': source, 'trajectory_name': trajectory,
                       'frame_indices': torch.from_numpy(indices), 'posterior_mean': mean, 'posterior_logvar': logvar,
                       'metadata': {'num_frames': len(frames), 'scaling_factor': 0.18215, 'image_size': 224,
                                    'storage_dtype': 'bfloat16', 'compute_dtype': 'bfloat16', 'vae_batch_size': args.batch_size}}
            record['posterior_shape'] = list(mean.shape)
        else:
            pairs = local_pairs(indices, navanywhere)
            motion = encode_lam(model, frames, pairs, args, load_navigation_frame)
            payload = {'frame_pairs': torch.from_numpy(pairs), 'motion': motion,
                       'source_id': source, 'trajectory_id': trajectory, 'proxy_type': 'latent',
                       'latent_value': 'z_mu', 'pair_direction': 'current_to_goal', 'normalization': 'raw'}
        atomic_save(payload, output)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        manifests[source].add(trajectory)
        print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
