#!/usr/bin/env python3
"""Sequentially score archived or fresh Table 1 predictions with the old evaluator.

Default action is a CPU-only plan. ``run`` requires an explicit GPU and records
every command, working directory, PID, log, exit status and evaluator hash.
No inference is performed; old predictions/results are always read-only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess


HOME_ROOT = Path('/file_system/vepfs/algorithm/dujun.nie')
NAS = Path('/file_system/nas/algorithm/dujun.nie/nwm')
ROOT = NAS / 'opennwm_reproduction/dino_main_tables_20261007'
RUNTIME = HOME_ROOT / 'code/CompACT-eval-huron-fix'
PYTHON = HOME_ROOT / 'miniconda3/envs/nwm/bin/python'
BENCH = NAS / 'results/nwm_benchmark'
LATER = BENCH / 'a800_full_evaluations/full_8xa800_seed0_v1'
ARCHIVE_PRED = BENCH / 'finalLAM_reset_checkpoint_sweep_20260922/predictions/finalLAM-reset-joint0100000'
DATASETS = ('recon', 'scand', 'huron', 'tartan_drive', 'go_stanford',
            'tum_rgbd', 'planetary_rover', 'unitree_go2')
LATER_REUSED = ('recon', 'scand', 'tartan_drive', 'go_stanford', 'tum_rgbd', 'unitree_go2')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def plan(args):
    if args.source == 'archive' and args.model != 'baseline':
        raise ValueError('Archive source is only available for baseline')
    if args.protocol == 'l20_batch32' and (args.source != 'fresh' or args.datasets != ['huron']):
        raise ValueError('l20_batch32 scores require fresh HuRoN only; other datasets are explicitly shared in the report')
    if args.source == 'fresh' and args.protocol in ('corrected', 'l20_batch32') and args.pred_root is None:
        raise ValueError('Corrected fresh evaluation requires an explicit --pred-root')
    output_dir = args.output_dir
    if output_dir is None:
        if args.source == 'archive':
            output_dir = args.root / ('archive_rescore' if args.protocol == 'paper_historical' else 'archive_rescore_corrected')
        else:
            output_dir = args.root / 'fresh_scores' / args.protocol / args.model
    evaluator = args.runtime.resolve() / 'scripts/evaluate_nwm_predictions.py'
    if not evaluator.is_file() or not args.python.is_file():
        raise FileNotFoundError('Explicit runtime Python or original metric evaluator missing')
    jobs = []
    for dataset in args.datasets:
        count = 10 if dataset == 'planetary_rover' else (329 if dataset == 'huron' and args.protocol in ('corrected', 'l20_batch32') else 500)
        endpoint = '16obs' if dataset == 'planetary_rover' and args.protocol == 'corrected' else '4s'
        if args.gt_root is not None:
            gt = args.gt_root.resolve() / dataset / 'time'
        elif args.protocol in ('corrected', 'l20_batch32'):
            gt = LATER / 'gt' / dataset / 'time'
        elif dataset == 'recon':
            gt = NAS / 'results/release_eval_20260820/gt/recon/time'
        else:
            gt = BENCH / 'gt' / dataset / 'time'
        if args.pred_root is not None:
            pred = args.pred_root.resolve() / dataset / 'time'
        elif args.source == 'archive':
            base = ARCHIVE_PRED if args.protocol == 'paper_historical' else LATER / 'predictions/opennwm-finalLAM-100k'
            pred = base / dataset / 'time'
        else:
            part = 'rover' if dataset == 'planetary_rover' else 'main'
            pred = args.root / args.model / part / 'merged_predictions' / dataset / 'time'
        output = output_dir.resolve() / f'{dataset}.json'
        input_mapping = None
        if args.source == 'archive' and args.protocol == 'corrected':
            # The later archive mixes old id_N_4 directories (4s only) with
            # id_N directories (other horizons). Pair by explicit original ID;
            # never let the evaluator naturally sort this mixed population.
            view = output_dir.resolve() / 'canonical_views' / dataset
            input_mapping = {'gt_source': str(gt), 'pred_source': str(pred),
                             'gt_directory_template': 'id_{sample_id}',
                             'pred_directory_template': 'id_{sample_id}_4' if dataset in LATER_REUSED else 'id_{sample_id}',
                             'endpoint_filename': '4.png', 'view_root': str(view),
                             'mapping_manifest': str(view / 'mapping.json')}
            gt, pred = view / 'gt', view / 'pred'
        batch = 10 if dataset == 'planetary_rover' else 32
        command = [str(args.python.resolve()), str(evaluator), '--gt-dir', str(gt),
                   '--pred-dir', str(pred), '--output', str(output), '--frames', f'{endpoint}:4',
                   '--dataset', dataset, '--eval-type', 'time', '--eval-name', 'time',
                   '--batch-size', str(batch), '--device', 'cuda',
                   '--dreamsim-cache', str(args.dreamsim_cache.resolve()),
                   '--inference-backend', 'nwm', '--sampler', 'ddpm', '--sampling-steps', '250', '--seed', '0']
        jobs.append({'dataset': dataset, 'protocol': args.protocol, 'model': args.model,
                     'origin': 'archive_rescore' if args.source == 'archive' else 'fresh_inference',
                     'endpoint': endpoint, 'frame_index': 4, 'expected_count': count,
                     'metric_batch_size': batch, 'sample_order': 'natural_sorted',
                     'gt_dir': str(gt), 'pred_dir': str(pred), 'output': str(output),
                     'log': str(output_dir.resolve() / 'logs' / f'{dataset}.log'),
                     'status_path': str(output_dir.resolve() / 'status' / f'{dataset}.json'),
                     'working_directory': str(args.runtime.resolve()), 'command': command,
                     'shell_command': shlex.join(command), 'state': 'planned',
                     'input_mapping': input_mapping,
                     'evaluator_sha256': sha256(evaluator), 'launcher_sha256': sha256(__file__)})
        if args.protocol == 'l20_batch32':
            jobs[-1]['inference_metadata_path'] = str(args.pred_root.resolve().parent / 'metadata.json')
    return output_dir.resolve(), jobs


def validate_inference_metadata(job):
    path = job.get('inference_metadata_path')
    if path is None:
        return
    metadata = read_json(path)
    expected = {'protocol': 'l20_batch32', 'batch_size': 32, 'archive_batch_size': 128,
                'new_protocol': True, 'accept_batch_change': True, 'model': job['model'],
                'state': 'completed', 'exit_code': 0, 'expected_full_count': 329,
                'sample_ids': list(range(329)), 'datasets': ['huron']}
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError(f'L20 HuRoN inference provenance mismatch: {mismatches}')
    if Path(job['pred_dir']).resolve() != Path(metadata['run_directory']).resolve() / 'predictions/huron/time':
        raise ValueError('L20 HuRoN prediction path does not match inference metadata')
    job['inference_metadata_sha256'] = sha256(path)


def prepare_views(job):
    mapping = job.get('input_mapping')
    if not mapping:
        return
    records = []
    for index in range(job['expected_count']):
        row = {'sample_id': index}
        for kind in ('gt', 'pred'):
            name = mapping[f'{kind}_directory_template'].format(sample_id=index)
            original = Path(mapping[f'{kind}_source']) / name / '4.png'
            if not original.is_file():
                raise FileNotFoundError(f'Explicit endpoint mapping is missing: {original}')
            row[kind] = {'source': str(original), 'resolved_source': str(original.resolve()),
                         'bytes': original.stat().st_size}
        records.append(row)
    # Validate the complete explicitly selected ID population. id_N sibling
    # folders in the six reused archives contain other horizons and are ignored.
    for kind in ('gt', 'pred'):
        root = Path(mapping[f'{kind}_source'])
        template = mapping[f'{kind}_directory_template']
        expected = {template.format(sample_id=i) for i in range(job['expected_count'])}
        actual = {p.name for p in root.iterdir() if p.is_dir()}
        allowed_other = {f'id_{i}' for i in range(job['expected_count'])} if template.endswith('_4') else set()
        if actual != expected | (actual & allowed_other):
            raise ValueError(f'Explicit source population mismatch: {root}')
    for row in records:
        for kind in ('gt', 'pred'):
            target = Path(job[f'{kind}_dir']) / f'id_{row["sample_id"]}' / '4.png'
            original = Path(row[kind]['source'])
            if target.is_symlink():
                if target.resolve() != original.resolve():
                    raise ValueError(f'Existing canonical view points at a different image: {target}')
            elif target.exists():
                raise FileExistsError(f'Refusing non-symlink canonical image: {target}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(original)
    write_json(mapping['mapping_manifest'], {'dataset': job['dataset'], 'endpoint': job['endpoint'],
                                             'selection': mapping, 'samples': records})
    job['mapping_manifest_sha256'] = sha256(mapping['mapping_manifest'])


def validate_inputs(job):
    expected = {f'id_{i}' for i in range(job['expected_count'])}
    inputs = {}
    for kind in ('gt', 'pred'):
        root = Path(job[f'{kind}_dir'])
        names = {p.name for p in root.iterdir() if p.is_dir()}
        if names != expected:
            raise ValueError(f'{kind} population mismatch: {root}; expected={len(expected)} actual={len(names)}')
        missing = [name for name in expected if not (root / name / '4.png').is_file()]
        if missing:
            raise ValueError(f'{kind} missing endpoint 4.png: {missing[:10]}')
        inputs[kind] = {'directory': str(root), 'count': len(names),
                        'sample_names_sha256': hashlib.sha256('\n'.join(sorted(names)).encode()).hexdigest()}
    return inputs


def validate_score(job):
    result = read_json(job['output'])
    checks = {'dataset': job['dataset'], 'eval_name': 'time', 'eval_type': 'time',
              'sample_order': 'natural_sorted', 'frame_indices': {job['endpoint']: 4}}
    for key, value in checks.items():
        if result.get(key) != value:
            raise ValueError(f'Output {key} mismatch: {job["output"]}')
    for key, kind in (('gt_eval_dir', 'gt'), ('pred_eval_dir', 'pred')):
        if Path(result[key]).resolve() != Path(job[f'{kind}_dir']).resolve():
            raise ValueError(f'Output source mismatch: {key}')
    metric = result['metrics'][job['endpoint']]
    if metric['sample_count'] != job['expected_count']:
        raise ValueError('Output sample_count mismatch')
    if not all(math.isfinite(metric[key]) for key in ('lpips_alex', 'dreamsim', 'psnr')):
        raise ValueError('Nonfinite score')
    return metric


def gpu_preflight(gpu, min_free_mib):
    command = ['nvidia-smi', '--id', gpu, '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free',
               '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True)
    record = {'command': command, 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    if result.returncode:
        raise RuntimeError(f'GPU inspection failed: {record}')
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1 or float(lines[0].split(',')[-1].strip()) < min_free_mib:
        raise RuntimeError(f'Expected one GPU with at least {min_free_mib} MiB free: {record}')
    return record


def run(args, output_dir, jobs):
    if not args.gpu:
        raise ValueError('run requires explicit --gpu')
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / '.scores.lock'
    # A stale lock is never silently removed: inspect the recorded PID first.
    with lock.open('x') as stream:
        json.dump({'pid': os.getpid(), 'created_utc': now()}, stream)
    manifest_path = output_dir / 'scores_manifest.json'
    manifest = {'schema_version': 1, 'created_utc': now(), 'origin': jobs[0]['origin'], 'results': [], 'jobs': jobs}
    if manifest_path.exists():
        previous = read_json(manifest_path)
        # Preserve completed records outside this invocation's selected datasets.
        manifest['results'] = [item for item in previous.get('results', [])
                               if item['dataset'] not in args.datasets]
    try:
        launcher_snapshot = output_dir / 'sources' / f'reproduce_paper_scores.{sha256(__file__)}.py'
        launcher_snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not launcher_snapshot.exists():
            launcher_snapshot.write_bytes(Path(__file__).read_bytes())
        manifest['launcher_snapshot'] = str(launcher_snapshot)
        for job in jobs:
            validate_inference_metadata(job)
            prepare_views(job)
            job['input_validation'] = validate_inputs(job)
            output = Path(job['output'])
            if output.exists():
                if not args.skip_existing:
                    raise FileExistsError(f'Refusing to overwrite {output}; use --skip-existing to validate/reuse')
                metric = validate_score(job)
                job.update(state='reused_existing_score', output_sha256=sha256(output))
            else:
                disk = shutil.disk_usage(output_dir)
                if disk.free < 100 * 1024**2:
                    raise RuntimeError('Need at least 100 MiB free before scoring/log writes')
                job.update(free_disk_bytes=disk.free, gpu_preflight=gpu_preflight(args.gpu, args.min_free_mib),
                           environment_overrides={'CUDA_VISIBLE_DEVICES': args.gpu,
                                                  'OMP_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'})
                log_path = Path(job['log'])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                print(json.dumps({k: job[k] for k in ('dataset', 'working_directory', 'shell_command', 'log')}), flush=True)
                with log_path.open('x') as log:
                    process = subprocess.Popen(job['command'], cwd=job['working_directory'],
                                               env=dict(os.environ, **job['environment_overrides']),
                                               stdout=log, stderr=subprocess.STDOUT)
                    job.update(state='running', pid=process.pid, started_utc=now())
                    write_json(job['status_path'], job)
                    write_json(manifest_path, manifest)
                    print(f"PID={process.pid}; log={log_path}", flush=True)
                    exit_code = process.wait()
                job.update(exit_code=exit_code, finished_utc=now(), state='completed' if exit_code == 0 else 'failed')
                print(f"Exit status: {exit_code}; log tail ({job['dataset']}):", flush=True)
                print('\n'.join(log_path.read_text(errors='replace').splitlines()[-24:]), flush=True)
                if exit_code:
                    write_json(job['status_path'], job)
                    write_json(manifest_path, manifest)
                    return exit_code
                metric = validate_score(job)
                job['output_sha256'] = sha256(output)
            job['validated_metrics'] = metric
            write_json(job['status_path'], job)
            manifest['results'].append({'protocol': job['protocol'], 'model': job['model'], 'table': 1,
                                        'dataset': job['dataset'], 'endpoint': job['endpoint'],
                                        'path': job['output'], 'complete': True,
                                        'origin': job['origin'], 'status_path': job['status_path']})
            if 'inference_metadata_path' in job:
                manifest['results'][-1].update(inference_metadata_path=job['inference_metadata_path'],
                                                inference_metadata_sha256=job['inference_metadata_sha256'])
            write_json(manifest_path, manifest)
        print(f'Completed selected scores: {manifest_path}', flush=True)
        return 0
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run'))
    parser.add_argument('--source', choices=('archive', 'fresh'), required=True)
    parser.add_argument('--protocol', choices=('paper_historical', 'corrected', 'l20_batch32'), default='paper_historical')
    parser.add_argument('--model', choices=('baseline', 'dino'), default='baseline')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--datasets', nargs='+', choices=(*DATASETS, 'uzh_fpv'), default=list(DATASETS))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--pred-root', type=Path)
    parser.add_argument('--gt-root', type=Path)
    parser.add_argument('--runtime', type=Path, default=RUNTIME)
    parser.add_argument('--python', type=Path, default=PYTHON)
    parser.add_argument('--dreamsim-cache', type=Path, default=NAS / 'compact/cache/models')
    parser.add_argument('--gpu')
    parser.add_argument('--min-free-mib', type=int, default=2500)
    parser.add_argument('--skip-existing', action='store_true', help='Validate and reuse existing matching scores without overwriting')
    args = parser.parse_args()
    if len(args.datasets) != len(set(args.datasets)):
        raise ValueError('Duplicate datasets are not allowed')
    output_dir, jobs = plan(args)
    if args.action == 'plan':
        print(json.dumps({'output_directory': str(output_dir), 'jobs': jobs}, indent=2))
        return 0
    return run(args, output_dir, jobs)


if __name__ == '__main__':
    raise SystemExit(main())
