#!/usr/bin/env python3
"""Portable entry point: train, eval, extract, calibrate, prepare-dino, precompute-dino."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parent
VARIANTS = {1: 'action', 2: 'pixel', 3: 'dino', 4: 'pixel_action'}


def merge(base, overlay):
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def read_config(path):
    return yaml.safe_load(Path(path).read_text()) or {}


def build_train_config(args):
    from navigation_data_profile import build_profile, build_legacy_profile, NAVANYWHERE15_DATASETS, NAVANYWHERE_DATASETS

    profile = args.profile or ('legacy' if args.variant == 1 else 'navanywhere')
    if args.variant == 1 and profile != 'legacy':
        raise ValueError('Action-only LAM uses the labeled legacy profile')
    run = args.run_dir.expanduser().resolve()
    kwargs = dict(variant=args.variant, data_root=args.data_root.resolve(),
                  split_root=args.split_root.resolve(), index_cache_path=(args.index_cache or run / 'index.pkl').resolve(),
                  dino_repo_path=args.dino_repo.resolve(), dino_weights_path=args.dino_weights.resolve())
    config = read_config(ROOT / 'config/navigation_base.yaml')
    merge(config, read_config(ROOT / f'config/navigation_{args.variant}_{VARIANTS[args.variant]}.yaml'))
    if profile == 'legacy':
        overlay = build_legacy_profile(**kwargs, dino_feature_cache_path=args.dino_cache.resolve())
    else:
        overlay = build_profile(**kwargs, navanywhere_datasets=(NAVANYWHERE15_DATASETS if profile == 'navanywhere15' else NAVANYWHERE_DATASETS))
    merge(config, overlay)
    if args.variant == 4:
        suffix = '' if profile == 'legacy' else f'_{profile}'
        merge(config, read_config(args.calibration or ROOT / f'config/generated/navigation_4{suffix}_action_weight.yaml'))
    for path in args.config:
        merge(config, read_config(path))
    config['model']['init_args']['log_path'] = str(run / 'images')
    data, trainer = config['data'], config['trainer']
    for key, value in [('batch_size', args.batch_size), ('num_workers', args.num_workers)]:
        if value is not None:
            data[key] = value
    if data['num_workers'] == 0:
        data['persistent_workers'] = False
    if args.max_steps is not None:
        trainer['max_steps'] = args.max_steps
        if args.variant != 1:
            config['model']['init_args']['scheduler_total_steps'] = args.max_steps
    if args.warmup_steps is not None and args.variant != 1:
        config['model']['init_args']['scheduler_warmup_steps'] = args.warmup_steps
    if args.lr is not None:
        config['model']['init_args']['optimizer']['init_args']['lr'] = args.lr
    if args.precision:
        trainer['precision'] = args.precision
    if args.grad_accum is not None:
        trainer['accumulate_grad_batches'] = args.grad_accum
    trainer.update(accelerator=args.accelerator, devices=args.devices, num_nodes=args.nodes,
                   strategy='auto' if args.devices * args.nodes == 1 else 'ddp_find_unused_parameters_false',
                   default_root_dir=str(run), enable_checkpointing=False,
                   limit_val_batches=0, num_sanity_val_steps=0)
    if args.eval_config:
        validation = read_config(args.eval_config)
        sources = validation['source_specs']
    else:
        sources = []
        for source in data['source_specs']:
            if source.get('has_action', True):
                source = dict(source)
                old = Path(source['split_manifest'])
                source['split_manifest'] = str(old.parent.parent / args.eval_split / old.name)
                source['data_root'] = source.get('data_root', source.get('root'))
                sources.append(source)
        validation = {}
    callback = dict(run_dir=str(run), source_specs=sources,
                    every_n_steps=args.eval_every, save_every_n_steps=args.save_every,
                    periodic_every_n_steps=args.periodic_every, sample_count=args.eval_samples,
                    batch_size=args.eval_batch_size, keep_periodic=args.keep_periodic, seed=config['seed_everything'],
                    image_height=data['image_height'], image_width=data['image_width'],
                    max_frame_offset=data['max_frame_offset'], action_min=data['action_min'], action_max=data['action_max'])
    callback.update({key: value for key, value in validation.items() if key != 'source_specs'})
    trainer['callbacks'] = [
        {'class_path': 'lam.navigation_callbacks.LAMFusedAttention', 'init_args': {'enabled': True}},
        {'class_path': 'lam.navigation_callbacks.LAMActivationCheckpointing', 'init_args': {'enabled': args.activation_checkpointing}},
        {'class_path': 'lam.training_runtime.PeriodicEvaluation', 'init_args': callback},
    ]
    trainer['logger'] = [{'class_path': 'lightning.pytorch.loggers.TensorBoardLogger',
                          'init_args': {'save_dir': str(run), 'name': 'tensorboard', 'version': ''}}]
    if args.wandb:
        trainer['logger'].append({'class_path': 'lightning.pytorch.loggers.WandbLogger',
                                 'init_args': {'project': args.wandb, 'save_dir': str(run),
                                               'id': run.name, 'resume': 'allow'}})
    return config


def train(argv):
    parser = argparse.ArgumentParser(description='LAM training with fixed validation and latest/best/periodic checkpoints')
    parser.add_argument('--variant', type=int, choices=VARIANTS, required=True)
    parser.add_argument('--profile', choices=('legacy', 'navanywhere', 'navanywhere15'))
    parser.add_argument('--data-root', type=Path, default=Path(os.environ.get('NWM_DATA_ROOT', os.environ.get('OPENNWM_DATA_ROOT', 'data'))))
    parser.add_argument('--split-root', type=Path, default=Path(os.environ.get('NWM_SPLIT_ROOT', str(ROOT.parent / 'nwm/data_splits'))))
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--index-cache', type=Path)
    parser.add_argument('--dino-repo', type=Path, default=Path(os.environ.get('DINO_REPO_PATH', 'third_party/dinov3')))
    parser.add_argument('--dino-weights', type=Path, default=Path(os.environ.get('DINO_WEIGHTS_PATH', 'weights/dinov3.pth')))
    parser.add_argument('--dino-cache', type=Path, default=Path('cache/dino'))
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--config', type=Path, action='append', default=[])
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--node-rank', type=int, default=int(os.environ.get('NODE_RANK', 0)))
    parser.add_argument('--master-addr', default=os.environ.get('MASTER_ADDR', '127.0.0.1'))
    parser.add_argument('--master-port', type=int, default=int(os.environ.get('MASTER_PORT', 29501)))
    parser.add_argument('--accelerator', choices=('cpu', 'gpu'), default='gpu')
    parser.add_argument('--precision', choices=('32-true', '16-mixed', 'bf16-mixed'))
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--warmup-steps', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--grad-accum', type=int)
    parser.add_argument('--activation-checkpointing', action='store_true')
    parser.add_argument('--eval-split', default='val')
    parser.add_argument('--eval-config', type=Path, help='YAML with held-out source_specs and optional validation settings')
    parser.add_argument('--eval-every', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--periodic-every', type=int, default=10000)
    parser.add_argument('--keep-periodic', type=int, default=3, help='retained step checkpoints; 0 keeps all')
    parser.add_argument('--eval-samples', type=int, default=192)
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--wandb', help='optional W&B project; TensorBoard is always enabled')
    parser.add_argument('--resume', nargs='?', const='latest', help='latest or checkpoint path')
    parser.add_argument('--dry-run', action='store_true', help='write resolved config without starting training')
    args, extra = parser.parse_known_args(argv)
    config = build_train_config(args)
    run = args.run_dir.expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    config_path = run / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    checkpoint = None
    if args.resume:
        checkpoint = run / 'checkpoints/latest.ckpt' if args.resume == 'latest' else Path(args.resume).resolve()
    command = [sys.executable]
    if args.devices * args.nodes > 1:
        command += ['-m', 'torch.distributed.run', f'--nnodes={args.nodes}', f'--nproc_per_node={args.devices}',
                    f'--node_rank={args.node_rank}', f'--master_addr={args.master_addr}', f'--master_port={args.master_port}']
    command += [str(ROOT / 'navigation_main.py'), 'fit', '--config', str(config_path)]
    if checkpoint:
        command += ['--ckpt_path', str(checkpoint)]
    command += extra
    print(' '.join(command), flush=True)
    if args.dry_run:
        return 0
    for source in config['trainer']['callbacks'][-1]['init_args']['source_specs']:
        if not Path(source['split_manifest']).is_file():
            parser.error(f"Validation list missing: {source['split_manifest']}. Supply --eval-config with independent held-out trajectories; see lam/README.md.")
    if checkpoint and not checkpoint.is_file():
        parser.error(f'Checkpoint not found: {checkpoint}')
    return subprocess.call(command, cwd=ROOT)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ('-h', '--help'):
        print(__doc__)
        print('Use: python lam/run.py <command> --help')
        return 0
    command, arguments = argv[0], argv[1:]
    if command == 'train':
        return train(arguments)
    scripts = {'eval': 'evaluate.py', 'extract': 'navigation_latent_extract.py',
               'calibrate': 'calibrate_action_weight.py', 'prepare-dino': 'prepare_dinov3.py',
               'precompute-dino': 'precompute_dinov3_features.py'}
    if command not in scripts:
        raise SystemExit(f'Unknown LAM command: {command}')
    if command == 'extract' and (not arguments or arguments[0] not in ('extract', 'benchmark', 'validate')):
        arguments = ['extract'] + arguments
    return subprocess.call([sys.executable, str(ROOT / scripts[command]), *arguments])


if __name__ == '__main__':
    raise SystemExit(main())
