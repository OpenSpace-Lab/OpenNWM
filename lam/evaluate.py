#!/usr/bin/env python3
"""Original LAM evaluation protocols with small, resumable batch files."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from lam.navigation_evaluation.adapters import load_evaluation_adapter
from lam.navigation_evaluation.assets import evaluation_torch_home
from lam.navigation_evaluation.catalog import load_navigation_frame
from lam.navigation_evaluation.config import load_evaluation_config
from lam.navigation_evaluation.io import atomic_write_json
from lam.navigation_evaluation.manifest import build_evaluation_manifest
from lam.navigation_evaluation.reporting import aggregate_rows, write_run_reports, write_gallery
from lam.navigation_evaluation.runner import _RecordFrames, _process_batch, _render_condition_visuals, _reference_frames
from lam.navigation_evaluation.visualization import FeaturePCA

ROOT = Path(__file__).resolve().parent


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def build_pca(adapter, records, config, output):
    path = output / 'feature_pca.npz'
    if path.exists():
        return FeaturePCA.load(path)
    vis, sampling = config['visualization'], config['sampling']
    references, _ = _reference_frames(records, int(vis['pca_reference_frames_per_dataset_split']), int(config['seed']))
    max_tokens = int(vis['pca_max_tokens'])
    per_frame = max(1, math.ceil(max_tokens / len(references)))
    rng = np.random.default_rng(int(config['seed']))
    features = []
    batch_size = int(config['runtime']['batch_size'])
    for start in range(0, len(references), batch_size):
        frames = torch.stack([load_navigation_frame(Path(path), image_height=sampling['image_height'],
                              image_width=sampling['image_width']) for _, _, path in references[start:start + batch_size]])
        extracted = adapter.extract_features(frames).detach().float().cpu()
        for feature in extracted.unbind(0):
            indices = rng.choice(feature.shape[0], size=min(per_frame, feature.shape[0]), replace=False)
            features.append(feature[torch.as_tensor(indices)].clone())
    pca = FeaturePCA.fit(features, seed=config['seed'], max_tokens=max_tokens,
                         lower_percentile=vis['pca_lower_percentile'], upper_percentile=vis['pca_upper_percentile'])
    pca.save(path)
    return pca


def run(config, checkpoint, manifest, output, *, model_overrides=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint).resolve()
    records = read_records(manifest)
    # Resume identity is a small config/path/stat record, not a content scan.
    identity = {'checkpoint': str(checkpoint), 'checkpoint_mtime_ns': checkpoint.stat().st_mtime_ns,
                'checkpoint_size': checkpoint.stat().st_size,
                'manifest': str(Path(manifest).resolve()), 'manifest_mtime_ns': Path(manifest).stat().st_mtime_ns,
                'config': config, 'model_overrides': model_overrides or {}}
    identity_path = output / 'run.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('This output directory belongs to different inputs. Choose a new --output-dir.')
    atomic_write_json(identity_path, identity)
    if config['runtime'].get('deterministic', True):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    adapter = load_evaluation_adapter(checkpoint, device=config['runtime']['device'],
                                      fused_attention=config['runtime']['fused_sdpa'], model_overrides=model_overrides)
    lpips = None
    if 'rgb' in adapter.capabilities and config['metrics']['lpips']['enabled']:
        import piq
        with evaluation_torch_home(config['metrics']['lpips']['asset_root']):
            lpips = piq.LPIPS(reduction='none').eval().requires_grad_(False).to(adapter.device)
    pca = build_pca(adapter, records, config, output) if 'feature' in adapter.capabilities and any(r['is_visual'] for r in records) else None
    grouped = defaultdict(list)
    for record in records:
        if record['protocol'] in config['protocols']:
            grouped[record['condition_id']].append(record)
    all_rows, gallery = [], []
    sampling, runtime = config['sampling'], config['runtime']
    for condition, condition_records in grouped.items():
        directory = output / 'conditions' / condition
        directory.mkdir(parents=True, exist_ok=True)
        visuals = {}
        batch_size = int(runtime['batch_size'])
        # Restart at a saved batch boundary, loading no already-finished images.
        for start in range(0, len(condition_records), batch_size):
            batch_path = directory / f'batch-{start:08d}.pt'
            if batch_path.exists():
                saved = torch.load(batch_path, map_location='cpu', weights_only=True)
                rows, batch_visuals = saved['rows'], saved['visuals']
            else:
                batch_records = condition_records[start:start + batch_size]
                dataset = _RecordFrames(batch_records, int(sampling['image_height']), int(sampling['image_width']))
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=int(runtime['num_workers']))
                _, source_anchor, source_target, recipient_anchor, recipient_target = next(iter(loader))
                rows, batch_visuals = _process_batch(adapter, batch_records, source_anchor, source_target,
                                                     recipient_anchor, recipient_target,
                                                     lpips_model=lpips, pca=pca, config=config)
                temporary = batch_path.with_suffix('.tmp')
                torch.save({'rows': rows, 'visuals': batch_visuals}, temporary)
                temporary.replace(batch_path)
            all_rows.extend(rows)
            visuals.update(batch_visuals)
        entries = _render_condition_visuals(adapter, condition, condition_records, visuals, directory, config)
        for entry in entries:
            entry['path'] = str(directory / entry['path'])
        gallery.extend(entries)
        print(f'[eval] {condition}: {len(condition_records)} pairs', flush=True)
    metadata_path = Path(manifest).with_suffix('.metadata.json')
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    summary = aggregate_rows(all_rows, offsets=sampling['offsets'],
                             bootstrap_samples=int(config['metrics']['bootstrap_samples']),
                             confidence=float(config['metrics']['bootstrap_confidence']), seed=int(config['seed']),
                             primary_excludes_zero=bool(config['metrics']['primary_excludes_zero_offset']),
                             condition_stats=metadata.get('conditions'))
    write_run_reports(output, all_rows, summary, save_parquet=config['output']['save_parquet'])
    write_gallery(output, gallery)
    atomic_write_json(output / 'complete.json', {'pairs': len(all_rows), 'conditions': len(grouped)})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare-manifest', 'run', 'all'):
        command = sub.add_parser(name)
        command.add_argument('--config', type=Path, default=ROOT / 'config/navigation_eval.yaml')
        command.add_argument('--data-root', type=Path, default=Path(os.environ.get('NWM_DATA_ROOT', 'data')))
        command.add_argument('--split-root', type=Path, default=Path(os.environ.get('NWM_SPLIT_ROOT', str(ROOT.parent / 'nwm/data_splits'))))
        command.add_argument('--manifest', type=Path)
        command.add_argument('--output-dir', type=Path, default=Path('runs/lam/eval'))
        command.add_argument('--num-metric', type=int)
        command.add_argument('--num-visual', type=int)
        command.add_argument('--protocols')
        if name != 'prepare-manifest':
            command.add_argument('--checkpoint', type=Path, required=True)
            command.add_argument('--device', default='cuda:0')
            command.add_argument('--precision', choices=('32', '16-mixed', 'bf16-mixed'), default='16-mixed')
            command.add_argument('--batch-size', type=int, default=8)
            command.add_argument('--num-workers', type=int, default=0)
            command.add_argument('--disable-lpips', action='store_true')
            command.add_argument('--asset-root', type=Path)
            command.add_argument('--dino-repo', type=Path, default=Path(os.environ['DINO_REPO_PATH']) if os.environ.get('DINO_REPO_PATH') else None)
            command.add_argument('--dino-weights', type=Path, default=Path(os.environ['DINO_WEIGHTS_PATH']) if os.environ.get('DINO_WEIGHTS_PATH') else None)
    assets = sub.add_parser('prepare-assets')
    assets.add_argument('--asset-root', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare-assets':
        import piq
        with evaluation_torch_home(args.asset_root):
            piq.LPIPS(reduction='none')
        return 0
    config = load_evaluation_config(args.config)
    if args.data_root:
        for dataset in config['datasets'].values():
            dataset['data_root'] = str(args.data_root.resolve() / dataset['path_alias'])
    if args.split_root:
        for name, dataset in config['datasets'].items():
            directory = 'tartan_drive' if name == 'tartan_drive' else dataset['path_alias']
            for split in dataset['splits']:
                dataset['splits'][split] = str(args.split_root.resolve() / directory / split / 'traj_names.txt')
    for source in config.get('evaluation', {}).get('datasets', []):
        if args.data_root:
            source['data_root'] = str(args.data_root.resolve() / Path(source['data_root']).name)
        if args.split_root:
            old = Path(source['split_manifest'])
            source['split_manifest'] = str(args.split_root.resolve() / old.parts[-3] / old.parts[-2] / old.name)
    for field, argument in [('metric_count', args.num_metric), ('visual_count', args.num_visual)]:
        if argument is not None:
            config['sampling'][field] = argument
    if args.protocols:
        config['protocols'] = args.protocols.split(',')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest.resolve() if args.manifest else output / 'sample_manifest.jsonl'
    if not manifest.exists():
        if args.command == 'run':
            parser.error(f'Manifest missing: {manifest}. Use all or prepare-manifest first.')
        records, metadata = build_evaluation_manifest(config)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in records))
        temporary.replace(manifest)
        atomic_write_json(manifest.with_suffix('.metadata.json'), metadata)
    if args.command == 'prepare-manifest':
        print(manifest)
        return 0
    config['runtime'].update(device=args.device, precision=args.precision,
                             batch_size=args.batch_size, num_workers=args.num_workers)
    if args.disable_lpips:
        config['metrics']['lpips']['enabled'] = False
    if args.asset_root:
        config['metrics']['lpips']['asset_root'] = str(args.asset_root.resolve())
    overrides = {}
    if args.dino_repo:
        overrides['dino_repo_path'] = str(args.dino_repo.resolve())
    if args.dino_weights:
        overrides['dino_weights_path'] = str(args.dino_weights.resolve())
    run(config, args.checkpoint, manifest, output, model_overrides=overrides)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
