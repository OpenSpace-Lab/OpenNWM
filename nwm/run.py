#!/usr/bin/env python3
"""Small entry point; accepts ordinary Hydra key=value overrides after options."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODES = ['baseline', 'original'] + sorted(p.stem for p in (ROOT/'conf/two_stage').glob('*.yaml'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='Train baseline, Stage 1, or Stage 2')
    train.add_argument('--mode', choices=MODES, default='baseline')
    train.add_argument('--run-dir', type=Path)
    train.add_argument('--resume', type=Path, help='Run directory or complete checkpoint')
    train.add_argument('--gpus', type=int, default=int(os.environ.get('NPROC_PER_NODE', '1')))
    for name in ('infer', 'eval', 'plan'):
        commands.add_parser(name, help=f'Run {name}; accepts Hydra key=value arguments')
    for name in ('recipe', 'cache-vae', 'cache-lam'):
        commands.add_parser(name, add_help=False, help=f'Prepare {name}; use {name} --help')
    args, extra = parser.parse_known_args(argv)
    python = os.environ.get('NWM_PYTHON', sys.executable)
    if args.command == 'train':
        settings = []
        if args.mode == 'original':
            settings += ['training=nwm_original', 'model/generator=cdit_xl', 'motion_condition.enabled=false']
        elif args.mode != 'baseline':
            settings += [f'two_stage={args.mode}']
        if args.run_dir:
            settings += [f'hydra.run.dir={args.run_dir.expanduser().resolve()}']
        if args.resume:
            source = args.resume.expanduser().absolute()
            checkpoint = source/'checkpoints/latest.pth.tar' if source.is_dir() else source
            run_dir = checkpoint.parent.parent
            if not checkpoint.is_file():
                parser.error(f'Resume checkpoint does not exist: {checkpoint}')
            if not (run_dir/'.hydra/config.yaml').is_file():
                parser.error(f'Missing saved configuration: {run_dir}/.hydra/config.yaml')
            settings = [f'--config-path={run_dir}/.hydra', '--config-name=config',
                        f'training.from_checkpoint={checkpoint}',
                        f'hydra.run.dir={(args.run_dir or run_dir).expanduser().resolve()}',
                        'hydra.job.chdir=true']
        command = [python, '-m', 'torch.distributed.run', '--standalone',
                   f'--nproc_per_node={args.gpus}', str(ROOT/'train.py'), *settings, *extra]
    elif args.command == 'recipe':
        command = [python, str(ROOT/'scripts/build_navanywhere_sampling_recipe.py'), *extra]
    elif args.command.startswith('cache-'):
        command = [python, str(ROOT/'prepare.py'), args.command[6:], *extra]
    else:
        entry = {'infer': 'isolated_nwm_infer.py', 'eval': 'isolated_nwm_eval.py', 'plan': 'planning_eval.py'}[args.command]
        command = [python, str(ROOT/entry), *extra]
    print('Working directory:', ROOT, flush=True)
    print('Command:', __import__('shlex').join(command), flush=True)
    return subprocess.call(command, cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
