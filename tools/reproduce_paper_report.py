#!/usr/bin/env python3
"""CPU-only reference audit for OpenNWM Tables 1/2; never run inference.

The output deliberately separates published numbers, archived measurements,
rescoring archived images, and fresh inference. Optional fresh-result manifests
must name their protocol and model explicitly. Merely finding a metrics file
does not establish successful reproduction.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[1]
NAS = Path('/file_system/nas/algorithm/dujun.nie/nwm')
ROOT = NAS / 'opennwm_reproduction/dino_main_tables_20261007'
SWEEP = NAS / 'results/nwm_benchmark/finalLAM_reset_checkpoint_sweep_20260922/benchmark_results.json'
LATER = NAS / 'results/nwm_benchmark/a800_full_evaluations/full_8xa800_seed0_v1/report.json'
HIST_MODEL = 'finalLAM-reset-joint0100000'
LATER_MODEL = 'opennwm-finalLAM-100k'
DINO = NAS / 'compact/runs/navanywherev1_latentpt_dino_ft/nwm-latentpt-dino-ft/checkpoints/joint_0100000.pth.tar'
PDF_SHA = '91b306f869f96d0890d83f1a7125b4476291d0e93f36c27524b68334e6ee7292'
ID_DATASETS = ('recon', 'scand', 'huron', 'tartan_drive')
LATER_SHARED_DATASETS = ('recon', 'scand', 'tartan_drive', 'go_stanford', 'tum_rgbd', 'unitree_go2')
PROTOCOLS = ('paper_historical', 'corrected', 'l20_batch32')
RESULT_PROTOCOLS = ('paper_historical', 'l20_batch32')
PROTOCOL_LABELS = {'paper_historical': '论文历史口径', 'l20_batch32': 'L20 后续口径（HuRoN batch32）'}
OOD_FIELDS = {'go_stanford_psnr': 'go_stanford', 'tum_psnr': 'tum_rgbd',
              'planetary_rover_psnr': 'planetary_rover', 'office_go2_psnr': 'unitree_go2'}
NAV_DATASETS = {'recon': 'recon', 'scand': 'scand', 'office_go2': 'unitree_go2'}
TABLE1_FIELDS = ('id_lpips', 'id_dreamsim', 'id_psnr', *OOD_FIELDS)
TABLE2_FIELDS = tuple(f'{dataset}_{metric}' for dataset in NAV_DATASETS for metric in ('ate', 'rpe'))
LABELS = {'recon': 'RECON', 'scand': 'SCAND', 'huron': 'HuRoN',
          'tartan_drive': 'TartanDrive', 'go_stanford': 'Go Stanford',
          'tum_rgbd': 'TUM RGB-D', 'planetary_rover': 'Planetary Rover',
          'unitree_go2': 'Office-Go2', 'uzh_fpv': 'UZH FPV（主表外）'}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source(path, *, hash_file=True):
    path = Path(path).resolve()
    result = {'path': str(path), 'exists': path.is_file()}
    if result['exists']:
        result.update(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
        if hash_file:
            result['sha256'] = sha256(path)
    return result


def rounding(value, places, mode=ROUND_HALF_UP):
    return float(Decimal(str(value)).quantize(Decimal(10) ** -places, rounding=mode))


def table1_summary(datasets):
    metrics = {'id_lpips': 'lpips_alex', 'id_dreamsim': 'dreamsim', 'id_psnr': 'psnr'}
    result = {field: sum(datasets[d]['metrics'][metric] for d in ID_DATASETS) / 4
              for field, metric in metrics.items()}
    result.update({field: datasets[d]['metrics']['psnr'] for field, d in OOD_FIELDS.items()})
    return result


def checks(target, actual, fields, places):
    result = {}
    for field in fields:
        rounded = rounding(actual[field], places)
        truncated = rounding(actual[field], places, ROUND_DOWN)
        result[field] = {'paper': target[field], 'archive_raw': actual[field],
                         'delta_raw_minus_paper': actual[field] - target[field],
                         'round_half_up': rounded, 'truncate': truncated,
                         'matches_round_half_up': rounded == target[field],
                         'matches_truncate': truncated == target[field]}
    return result


def reuse_table1_fresh(root, fresh):
    """Share fresh measurements only where the full endpoint audit permits it."""
    audit_path = root / 'audit/table1_later_protocol.json'
    if not audit_path.is_file():
        return fresh
    evidence = read_json(audit_path).get('reused_historical_predictions', {})
    if set(evidence) != set(LATER_SHARED_DATASETS):
        raise ValueError('Later Table1 reuse requires all six audited datasets')
    for dataset, row in evidence.items():
        if any(row.get(key) != 500 for key in ('sample_count', 'gt_byte_identical',
                                               'gt_pixel_identical', 'same_historical_source_links')):
            raise ValueError(f'Incomplete endpoint/link identity evidence for {dataset}')
        historical_hash = row.get('historical_gt_manifest_sha256', '')
        later_hash = row.get('later_gt_manifest_sha256', '')
        if (len(historical_hash) != 64 or historical_hash != later_hash
                or any(c not in '0123456789abcdef' for c in historical_hash)):
            raise ValueError(f'GT manifest hashes do not prove endpoint identity for {dataset}')
    independent = {(item['protocol'], item['model'], item.get('dataset')) for item in fresh
                   if item['table'] == 1 and item['protocol'] != 'paper_historical'}
    shared = []
    for item in fresh:
        dataset = item.get('dataset')
        if (item['table'] != 1 or item['protocol'] != 'paper_historical'
                or dataset not in LATER_SHARED_DATASETS):
            continue
        metrics = item['data'].get('metrics', {}).get('4s', {})
        if (item.get('complete') is not True or metrics.get('sample_count') != 500
                or not all(isinstance(metrics.get(key), (float, int)) and math.isfinite(metrics[key])
                           for key in ('lpips_alex', 'dreamsim', 'psnr'))):
            continue
        reuse = {'audit_source': source(audit_path), 'dataset_evidence': evidence[dataset],
                 'from_protocol': 'paper_historical', 'endpoint': '4s',
                 'rule': "This model's complete fresh historical predictions and scores are shared; no archived scalar is substituted."}
        item['shared_between_protocol_views'] = True
        item['reuse_evidence'] = reuse
        for protocol in ('corrected', 'l20_batch32'):
            if (protocol, item['model'], dataset) not in independent:
                shared.append({**item, 'protocol': protocol, 'shared_between_protocol_views': True,
                               'reuse_evidence': reuse})
    for item in fresh:
        if (item['table'] != 1 or item['protocol'] != 'corrected' or item.get('dataset') != 'planetary_rover'
                or ('l20_batch32', item['model'], 'planetary_rover') in independent):
            continue
        metrics = item['data'].get('metrics', {}).get('16obs', {})
        if (item.get('complete') is not True or metrics.get('sample_count') != 10
                or not all(isinstance(metrics.get(key), (float, int)) and math.isfinite(metrics[key])
                           for key in ('lpips_alex', 'dreamsim', 'psnr'))):
            continue
        metadata_path = root / 'corrected' / item['model'] / 'planetary_rover/metadata.json'
        metadata = read_json(metadata_path)
        if metadata.get('state') != 'completed' or metadata.get('exit_code') != 0 or metadata.get('model') != item['model']:
            raise ValueError('Rover sharing requires this model\'s completed L20 inference')
        shared.append({**item, 'protocol': 'l20_batch32', 'shared_between_protocol_views': True,
                       'reuse_evidence': {'from_protocol': 'corrected', 'endpoint': '16obs',
                                         'inference_metadata_source': source(metadata_path),
                                         'archive_equivalence': 'L20 baseline did not match A800 archive',
                                         'rule': 'Share completed L20 Rover fresh inference; batch32 applies only to HuRoN.'}})
    return fresh + shared


def optional_results(root, manifests):
    rescored = []
    for protocol, directory in (('paper_historical', 'archive_rescore'), ('corrected', 'archive_rescore_corrected')):
        for path in sorted((root / directory).glob('*.json')):
            if path.stem not in LABELS:
                continue
            try:
                rescored.append({'source': source(path), 'origin': 'archive_rescore',
                                 'protocol': protocol, 'dataset': path.stem, 'data': read_json(path)})
            except (ValueError, OSError) as exc:
                rescored.append({'source': source(path), 'origin': 'archive_rescore', 'error': str(exc)})
    fresh = []
    manifest_paths = sorted({Path(path).resolve() for path in manifests} |
                            set((root / 'fresh_scores').glob('*/*/scores_manifest.json')))
    for manifest_path in manifest_paths:
        manifest = read_json(manifest_path)
        for item in manifest['results']:
            if item.get('origin', manifest.get('origin', 'fresh_inference')) != 'fresh_inference':
                raise ValueError(f'Archive scores cannot be registered as fresh: {manifest_path}')
            if item['protocol'] not in PROTOCOLS:
                raise ValueError(f'Unknown fresh protocol: {item["protocol"]}')
            if item['model'] not in ('baseline', 'dino') or item['table'] not in (1, 2):
                raise ValueError('Fresh results require model baseline/dino and table 1/2')
            path = Path(item['path'])
            if not path.is_absolute():
                path = Path(manifest_path).resolve().parent / path
            record = {**item, 'origin': 'fresh_inference', 'source': source(path),
                      'manifest_source': source(manifest_path), 'data': read_json(path)}
            if item['protocol'] == 'l20_batch32':
                metadata_path = Path(item['inference_metadata_path'])
                metadata = read_json(metadata_path)
                expected = {'protocol': 'l20_batch32', 'model': item['model'], 'batch_size': 32,
                            'archive_batch_size': 128, 'new_protocol': True, 'accept_batch_change': True,
                            'state': 'completed', 'exit_code': 0, 'expected_full_count': 329,
                            'datasets': ['huron'], 'sample_ids': list(range(329))}
                if (item['table'] != 1 or item.get('dataset') != 'huron'
                        or any(metadata.get(k) != v for k, v in expected.items())
                        or sha256(metadata_path) != item['inference_metadata_sha256']):
                    raise ValueError('L20 batch32 score has invalid inference provenance')
                record['inference_metadata_source'] = source(metadata_path)
                record['new_protocol_not_exact_reproduction'] = True
            fresh.append(record)
    for path in sorted((root / 'table2').glob('*/*/mb80/aggregate.json')):
        data = read_json(path)
        if data['model'] not in ('baseline', 'dino') or data['dataset'] not in NAV_DATASETS.values():
            raise ValueError(f'Unexpected Table2 aggregate identity: {path}')
        if data['microbatch_size'] != 80:
            raise ValueError('Table2 historical protocol requires candidate microbatch 80')
        # HuRoN/Rover differences affect Table1 only. Table2 is the same CEM80
        # evaluation in both views; reuse these measurements transparently.
        for protocol in PROTOCOLS:
            fresh.append({'origin': 'fresh_inference', 'protocol': protocol, 'model': data['model'],
                          'table': 2, 'dataset': data['dataset'], 'source': source(path),
                          'data': data, 'complete': data['complete'],
                          'shared_between_protocol_views': True})
    return {'archive_rescore': rescored, 'fresh_inference': reuse_table1_fresh(root, fresh),
            'fresh_completion_verified_by_this_tool': False,
            'note': 'Files are evidence attachments. Completeness, population, RNG protocol and pixel/metric agreement require separate verification.'}


def comparison_tables(ref):
    """Never substitute archive values for missing fresh measurements."""
    normalized = {}
    for item in ref['current_run']['fresh_inference']:
        data, table, protocol = item['data'], item['table'], item['protocol']
        dataset = item.get('dataset', data.get('dataset'))
        if not dataset:
            raise ValueError(f'Fresh result dataset required: {item["source"]["path"]}')
        if table == 1:
            endpoint = '16obs' if dataset == 'planetary_rover' and protocol != 'paper_historical' else '4s'
            metrics = data['metrics'].get(endpoint, {})
            count = metrics.get('sample_count', 0)
            expected = 10 if dataset == 'planetary_rover' else (329 if dataset == 'huron' and protocol != 'paper_historical' else 500)
            values = {k: metrics.get(k) for k in ('lpips_alex', 'dreamsim', 'psnr')}
        else:
            means = data.get('means', data.get('metrics', data))
            values = {'ate': means.get('ate', means.get(f'{dataset}_ate')),
                      'rpe': means.get('rpe_trans', means.get(f'{dataset}_rpe_trans'))}
            count = data.get('count', data.get('sample_count', 0))
            expected = 100
        complete = (item.get('complete') is True and count == expected
                    and all(isinstance(v, (float, int)) and math.isfinite(v) for v in values.values()))
        key = (protocol, item['model'], table, dataset)
        entry = {'metrics': values, 'count': count, 'expected_count': expected,
                 'complete': complete, 'source': item['source'],
                 'historical_equivalence': data.get('historical_equivalence', 'not_checked'),
                 'shared_between_protocol_views': item.get('shared_between_protocol_views', False)}
        if table == 1 and protocol == 'paper_historical' and item['model'] == 'baseline':
            evidence = ref['current_run'].get('baseline_table1_pixel_evidence', {}).get(dataset, {})
            if (complete and evidence.get('complete_pixel_replay') is True
                    and evidence.get('canonical_merge_verified') is True
                    and data.get('pred_eval_dir') == evidence.get('merged_dataset_directory')):
                entry['historical_equivalence'] = 'full_dataset_archive_pixels_identical'
                entry['historical_equivalence_evidence'] = evidence
        if 'reuse_evidence' in item:
            entry['reuse_evidence'] = item['reuse_evidence']
        if key in normalized and normalized[key]['source']['path'] != entry['source']['path']:
            raise ValueError(f'Multiple fresh sources for {key}; remove ambiguity explicitly')
        normalized[key] = entry

    def measurement(protocol, model, table, dataset, metric):
        entry = normalized.get((protocol, model, table, dataset))
        if entry is None:
            return None, '待运行/待评分'
        if not entry['complete']:
            return None, f"未完整（{entry['count']}/{entry['expected_count']}）"
        return entry['metrics'][metric], '已评分，复现核验另列'

    paper1 = next(row for row in ref['paper_targets']['table1']['rows'] if row['id'] == 'opennwm')
    paper2 = next(row for row in ref['paper_targets']['table2']['rows'] if row['id'] == 'opennwm')
    results = {}
    for protocol, archive_key in (('paper_historical', 'paper_historical_table1'), ('corrected', 'later_corrected_table1'),
                                  ('l20_batch32', 'later_corrected_table1')):
        table1 = []
        for field in TABLE1_FIELDS:
            row = {'field': field, 'paper': paper1[field], 'archive': ref['archive'][archive_key]['summary'][field]}
            for model in ('baseline', 'dino'):
                if field.startswith('id_'):
                    metric = {'id_lpips': 'lpips_alex', 'id_dreamsim': 'dreamsim', 'id_psnr': 'psnr'}[field]
                    values = [measurement(protocol, model, 1, d, metric) for d in ID_DATASETS]
                    available = [v for v, _ in values if v is not None]
                    value = sum(available) / 4 if len(available) == 4 else None
                    status = '已评分，复现核验另列' if value is not None else f'待完整评分（ID {len(available)}/4 数据集）'
                else:
                    value, status = measurement(protocol, model, 1, OOD_FIELDS[field], 'psnr')
                row[f'{model}_fresh'] = value
                row[f'{model}_status'] = status
            row['delta_dino_minus_baseline'] = (row['dino_fresh'] - row['baseline_fresh']
                                                  if row['dino_fresh'] is not None and row['baseline_fresh'] is not None else None)
            row['delta_baseline_minus_archive'] = (row['baseline_fresh'] - row['archive']
                                                      if row['baseline_fresh'] is not None else None)
            table1.append(row)
        table2 = []
        for field in TABLE2_FIELDS:
            name, metric = field.rsplit('_', 1)
            row = {'field': field, 'paper': paper2[field], 'archive': ref['archive']['table2']['summary'][field]}
            for model in ('baseline', 'dino'):
                value, status = measurement(protocol, model, 2, NAV_DATASETS[name], metric)
                row[f'{model}_fresh'], row[f'{model}_status'] = value, status
            row['delta_dino_minus_baseline'] = (row['dino_fresh'] - row['baseline_fresh']
                                                  if row['dino_fresh'] is not None and row['baseline_fresh'] is not None else None)
            row['delta_baseline_minus_archive'] = (row['baseline_fresh'] - row['archive']
                                                      if row['baseline_fresh'] is not None else None)
            table2.append(row)
        results[protocol] = {'table1': table1, 'table2': table2}
    results['fresh_per_dataset'] = [{'protocol': k[0], 'model': k[1], 'table': k[2], 'dataset': k[3], **v}
                                     for k, v in sorted(normalized.items())]
    return results


def historical_pixel_evidence(root, pixel_checks):
    """Use canonical shard/merge records; never count probes or moved attempts."""
    root = root.resolve()
    evidence = {}
    for dataset in (*ID_DATASETS, *OOD_FIELDS.values()):
        part, total, world = ('rover', 10, 1) if dataset == 'planetary_rover' else ('main', 500, 4)
        shards, sample_ids, reasons = [], [], []
        for item in pixel_checks:
            comparison = item['data']
            if dataset not in comparison['counts']:
                continue
            directory = Path(item['source']['path']).parent
            metadata_path = directory / 'metadata.json'
            if not metadata_path.is_file():
                reasons.append(f'Missing shard metadata: {metadata_path}')
                continue
            metadata = read_json(metadata_path)
            rank = metadata.get('virtual_rank')
            valid = (type(rank) is int and 0 <= rank < world
                     and directory == root / 'baseline' / part / f'rank_{rank}'
                     and metadata.get('state') == 'completed' and metadata.get('exit_code') == 0
                     and metadata.get('model') == 'baseline' and metadata.get('part') == part
                     and metadata.get('protocol') == 'paper_table1_historical_direct_4s'
                     and metadata.get('virtual_world_size') == world
                     and metadata.get('expected_full_count') == total
                     and metadata.get('horizons_seconds') == [4]
                     and dataset in metadata.get('datasets', [])
                     and metadata.get('equivalence_status') == 'all_pixels_identical'
                     and comparison.get('run_directory') == str(directory)
                     and metadata.get('run_directory') == str(directory)
                     and comparison.get('archive') == metadata.get('archive_predictions')
                     and comparison.get('all_pixels_identical') is True
                     and comparison.get('mismatched_files') == [])
            ids = list(range(rank, total, world)) if type(rank) is int and 0 <= rank < world else []
            counts = comparison['counts'][dataset]
            valid = valid and counts.get('compared') == counts.get('pixel_identical') == len(ids)
            if not valid:
                reasons.append(f'Shard identity/completeness/equality not proven: {directory}')
                continue
            shards.append({'directory': str(directory), 'metadata_source': source(metadata_path),
                           'comparison_source': item['source'], 'rank': rank, 'count': len(ids)})
            sample_ids.extend(ids)
        complete = sorted(sample_ids) == list(range(total)) and len(shards) == world
        merge_path = root / 'baseline' / part / 'merge_manifest.json'
        merged = root / 'baseline' / part / 'merged_predictions'
        merge_source = source(merge_path)
        merge = read_json(merge_path) if merge_path.is_file() else {}
        merge_valid = (complete and merge.get('count') == total
                       and merge.get('merged_predictions') == str(merged)
                       and sorted(merge.get('shards', [])) == sorted(s['directory'] for s in shards))
        evidence[dataset] = {'expected_count': total, 'verified_count': len(sample_ids),
                             'complete_pixel_replay': complete, 'canonical_merge_verified': merge_valid,
                             'merged_dataset_directory': str(merged / dataset / 'time'),
                             'shards': shards, 'merge_source': merge_source, 'unverified_reasons': reasons,
                             'scope': 'Decoded pixels only; metric equality and paper rounding remain separately reported.'}
    return evidence


def historical_rank1_diagnostics(root):
    path = root / 'diagnostics/rank1_logical_cuda_index_diagnosis.json'
    if not path.is_file():
        return {}
    data = read_json(path)
    old, restored = (data['probes'][name] for name in ('rank1_vae_seed_probe', 'rank1_cuda1_probe'))
    # The cuda0 probe also contains a seed0 variant; use the first, unchanged
    # occurrence of each event, not the later diagnostic intervention.
    def event(probe, name):
        return next(item for item in probe['events'] if item['event'] == name)
    fields = {'before_first_batch': ('observations_sha256', 'actions_sha256'),
              'before_vae_encode': ('input_sha256', 'cuda_rng_sha256'),
              'after_vae_encode': ('latent_sha256', 'cuda_rng_sha256')}
    controls = {f'{name}/{field}': event(old, name)[field] == event(restored, name)[field]
                for name, names in fields.items() for field in names}
    selected = [probe['environment']['CUDA_VISIBLE_DEVICES'].split(',')[event(probe, 'before_first_batch')['cuda_device']]
                for probe in (old, restored)]
    controls['same_physical_gpu'] = selected[0] == selected[1]
    controls['same_rank_seed_1'] = all(event(probe, 'before_first_batch')['cuda_initial_seed'] == 1
                                       for probe in (old, restored))
    old_comparison, restored_comparison = old['comparison']['predictions_vs_archive'], restored['comparison']['archive']
    verified = (all(controls.values()) and old['exit_code'] == restored['exit_code'] == 0
                and old_comparison['count'] == restored_comparison['count'] == 32
                and old_comparison['pixel_identical'] == old_comparison['byte_identical'] == 0
                and restored_comparison['pixel_identical'] == restored_comparison['byte_identical'] == 32
                and [event(probe, 'before_first_batch')['cuda_device'] for probe in (old, restored)] == [0, 1])
    relocation_path = root / 'audit/rank1_cuda0_relocation.json'
    relocation = read_json(relocation_path) if relocation_path.is_file() else {}
    return {'source': source(path), 'markdown_source': source(path.with_suffix('.md')),
            'scope': 'rank1_first_batch_only_not_full_replay', 'canonical_fresh_inclusion': False,
            'controlled_first_batch_result_verified': verified,
            'selected_physical_gpus': selected, 'controls_equal': controls,
            'logical_cuda_indices': [event(probe, 'before_first_batch')['cuda_device'] for probe in (old, restored)],
            'cuda0_vs_archive': old_comparison, 'cuda1_vs_archive': restored_comparison,
            'probe_exit_codes': [probe['exit_code'] for probe in (old, restored)],
            'interpretation': data['interpretation'], 'limitations': data['limitations'],
            'relocation_source': source(relocation_path),
            'relocation': {key: value for key, value in relocation.items() if key != 'files'},
            'relocation_files_rehashed_by_report': False,
            'exclusion_rule': 'Diagnostics and attempts are outside canonical baseline shard discovery. Old metadata paths remain original; identify the moved attempt by its relocation manifest and recorded hashes.'}


def enrich_current(ref, root):
    ref['operational_limits'] = {
        'later_huron': {
            'historical_batch_size': 128, 'recorded_peak_allocated_gib': 56.5,
            'available_device_family': 'NVIDIA L20',
            'unchanged_historical_batch_fits_current_device': False,
            'user_decision': 'L20 HuRoN batch32 authorized; separate new protocol',
            'note': 'Running a smaller inference batch changes stochastic VAE RNG consumption and is a different protocol. Do not silently label such scores as unchanged later-protocol reproduction.'}}
    decisions = root / 'evaluation_decisions.json'
    if decisions.is_file():
        ref['sources']['evaluation_decisions'] = source(decisions)
        ref['operational_limits']['user_decisions'] = read_json(decisions)
        ref['operational_limits']['later_huron']['user_decision'] = 'see_evaluation_decisions'
    l20_selection = root / 'audit/l20_user_selection.json'
    if l20_selection.is_file():
        ref['sources']['l20_user_selection'] = source(l20_selection)
        ref['operational_limits']['l20_user_selection'] = read_json(l20_selection)
    identity = root / 'checkpoint_runtime_identity.json'
    if identity.is_file():
        data = read_json(identity)
        ref['sources']['checkpoint_runtime_identity'] = source(identity)
        ref['runtime_packages'] = data.get('packages', {})
        for model in ('baseline', 'dino'):
            current = data['checkpoints'][model]
            checkpoint = ref['models'][model]['checkpoint']
            if Path(current['path']).resolve() != Path(checkpoint['path']).resolve() or current['bytes'] != checkpoint['bytes']:
                raise ValueError(f'Checkpoint identity metadata mismatch: {model}')
            ref['models'][model]['checkpoint_sha256_recomputed_this_run'] = current['sha256']
        ref['models']['baseline']['recomputed_sha256_matches_archive'] = (
            data['checkpoints']['baseline']['sha256'] == ref['models']['baseline']['checkpoint_sha256_from_archive'])
    pixel_checks = []
    for path in sorted((root / 'baseline').glob('*/rank_*/archive_pixel_comparison.json')):
        pixel_checks.append({'source': source(path), 'data': read_json(path)})
    required_datasets = (*ID_DATASETS, *OOD_FIELDS.values())
    pixel_counts = {d: {'compared': 0, 'pixel_identical': 0} for d in required_datasets}
    for item in pixel_checks:
        for dataset, values in item['data']['counts'].items():
            if dataset in pixel_counts:
                for metric in pixel_counts[dataset]:
                    pixel_counts[dataset][metric] += values[metric]
    ref['current_run']['baseline_table1_archive_pixel_checks'] = pixel_checks
    ref['current_run']['baseline_table1_pixel_coverage'] = pixel_counts
    pixel_evidence = historical_pixel_evidence(root, pixel_checks)
    ref['current_run']['baseline_table1_pixel_evidence'] = pixel_evidence
    ref['current_run']['baseline_table1_complete_pixel_replay'] = all(
        row['complete_pixel_replay'] for row in pixel_evidence.values())
    ref['comparisons'] = comparison_tables(ref)
    ref['current_run']['historical_table1_rank1_diagnostics'] = historical_rank1_diagnostics(root)
    later_checks = []
    for path in sorted((root / 'corrected/baseline').glob('*/archive_pixel_comparison.json')):
        later_checks.append({'dataset': path.parent.name, 'source': source(path), 'data': read_json(path)})
    ref['current_run']['baseline_table1_later_archive_pixel_checks'] = later_checks
    diagnostics = {}
    for name, relative in (
            ('rover_native_comparison', 'rover_direct_native/comparison.json'),
            ('rover_native_run', 'rover_direct_native/metadata.json'),
            ('baseline_checkpoint_hash', 'baseline_checkpoint_hash.json'),
            ('cuda_noise_launch_policy', 'cuda_noise_launch_policy.json')):
        path = root / 'diagnostics' / relative
        if path.is_file():
            diagnostics[name] = {'source': source(path), 'data': read_json(path)}
    ref['current_run']['later_rover_diagnostics'] = diagnostics
    verification_paths = (set((root / 'table2').glob('*/*/mb80/subset_verification.json')) |
                          set((root / 'table2').glob('*/*/mb80/l20_subset_verification.json')))
    pilot_queue = root / 'queues/table2_pilot.json'
    pending_specs = {}
    if pilot_queue.is_file():
        for job in read_json(pilot_queue).get('jobs', []):
            command = job.get('argv', [])
            if all(flag in command for flag in ('--model', '--dataset', '--indices')):
                model, dataset, indices = (command[command.index(flag) + 1]
                                           for flag in ('--model', '--dataset', '--indices'))
                path = root / 'table2' / model / dataset / 'mb80/subset_verification.json'
                verification_paths.add(path)
                pending_specs[path] = {'model': model, 'dataset': dataset, 'requested_indices_spec': indices,
                                       'queue_source': source(pilot_queue)}
    if not verification_paths:
        verification_paths.add(root / 'table2/baseline/recon/mb80/subset_verification.json')
    if l20_selection.is_file():
        verification_paths.add(root / 'table2/baseline/recon/mb80/l20_subset_verification.json')
    verifications = []
    for path in sorted(verification_paths):
        verifications.append({'source': source(path), 'scope': 'subset_only_not_full_reproduction',
                              'verification_policy': 'L20 differences accepted; provenance/coverage required' if path.name.startswith('l20_') else 'strict exact metric equality',
                              **pending_specs.get(path, {}),
                              'data': read_json(path) if path.is_file() else {'state': 'pending'}})
    ref['current_run']['table2_subset_verifications'] = verifications
    table2_mapping_path = root / 'diagnostics/table2_pilot_archive_mapping.json'
    if table2_mapping_path.is_file():
        mapping = read_json(table2_mapping_path)
        ref['sources']['table2_pilot_archive_mapping'] = source(table2_mapping_path)
        ref['current_run']['table2_pilot_archive_mapping'] = mapping
        ref['archive']['table2']['source_provenance'] = {
            'recon_reference_has_mixed_hardware_sources': mapping['conclusion']['historical_recon_reference_is_mixed'],
            'recon_population': mapping['reference_population'],
            'hardware_causality_established': mapping['conclusion']['hardware_causality_established'],
            'evidence_source': ref['sources']['table2_pilot_archive_mapping']}
        ref['operational_limits']['table2_pilot'] = mapping['pilot_timing']
    later_audit = root / 'audit/table1_later_protocol.json'
    if later_audit.is_file():
        ref['sources']['later_protocol_audit'] = source(later_audit)
        ref['current_run']['later_protocol_audit'] = read_json(later_audit)
    rescored = []
    for item in ref['current_run']['archive_rescore']:
        if 'data' not in item:
            continue
        protocol, dataset = item['protocol'], item['dataset']
        endpoint = '16obs' if dataset == 'planetary_rover' and protocol == 'corrected' else '4s'
        archive_key = 'paper_historical_table1' if protocol == 'paper_historical' else 'later_corrected_table1'
        metrics = item['data'].get('metrics', {}).get(endpoint)
        archive = ref['archive'][archive_key]['datasets'].get(dataset)
        if not metrics or not archive:
            continue
        original = archive['metrics']
        complete = metrics.get('sample_count') == original['sample_count']
        deltas = {key: metrics[key] - original[key] for key in ('lpips_alex', 'dreamsim', 'psnr')}
        rescored.append({'protocol': protocol, 'dataset': dataset, 'count_matches': complete,
                         'metrics': metrics, 'delta_from_archive': deltas,
                         'all_three_exactly_equal': complete and all(v == 0 for v in deltas.values()),
                         'source': item['source']})
    ref['current_run']['archive_rescore_checks'] = rescored
    add_result_summary(ref)


def add_result_summary(ref):
    """Completion describes measured coverage; reproduction is a separate state."""
    entries = {(e['protocol'], e['model'], e['table'], e['dataset']): e
               for e in ref['comparisons']['fresh_per_dataset']}
    progress, rows = [], []
    for protocol in RESULT_PROTOCOLS:
        for table, datasets, fields in ((1, (*ID_DATASETS, *OOD_FIELDS.values()), TABLE1_FIELDS),
                                         (2, tuple(NAV_DATASETS.values()), TABLE2_FIELDS)):
            comparisons = ref['comparisons'][protocol][f'table{table}']
            for model in ('baseline', 'dino'):
                complete_entries = [entries[(protocol, model, table, d)] for d in datasets
                                    if entries.get((protocol, model, table, d), {}).get('complete') is True]
                progress.append({'protocol': protocol, 'table': table, 'model': model,
                                 'complete_datasets': len(complete_entries), 'expected_datasets': len(datasets),
                                 'complete_metrics': sum(row[f'{model}_fresh'] is not None for row in comparisons),
                                 'expected_metrics': len(fields),
                                 'complete_samples': sum(e['count'] for e in complete_entries),
                                 'expected_samples': (300 if table == 2 else 3510 if protocol == 'paper_historical' else 3339),
                                 'complete': len(complete_entries) == len(datasets)})
            for row in comparisons:
                field = row['field']
                if table == 1:
                    required = ID_DATASETS if field.startswith('id_') else (OOD_FIELDS[field],)
                else:
                    required = (NAV_DATASETS[field.rsplit('_', 1)[0]],)
                record = {'protocol': protocol, 'view': PROTOCOL_LABELS[protocol], 'table': table,
                          'metric': field, 'direction': 'higher' if 'psnr' in field else 'lower',
                          'paper': row['paper'], 'archive': row['archive'],
                          'baseline': row['baseline_fresh'], 'dino': row['dino_fresh'],
                          'delta_dino_minus_baseline': row['delta_dino_minus_baseline'],
                          'delta_baseline_minus_archive': row['delta_baseline_minus_archive'],
                          'comparison_complete': row['baseline_fresh'] is not None and row['dino_fresh'] is not None,
                          'table2_shared_measurement': table == 2}
                for model in ('baseline', 'dino'):
                    relevant = [entries.get((protocol, model, table, d), {}) for d in required]
                    complete = row[f'{model}_fresh'] is not None
                    record[f'{model}_complete'] = complete
                    record[f'{model}_completion'] = 'completed' if complete else 'pending'
                    record[f'{model}_complete_datasets'] = sum(e.get('complete') is True for e in relevant)
                    record[f'{model}_expected_datasets'] = len(required)
                    record[f'{model}_sources'] = ';'.join(e['source']['path'] for e in relevant if 'source' in e)
                    if not complete:
                        state = 'pending_complete_measurements'
                    elif model == 'dino':
                        state = 'model_comparison_not_baseline_reproduction'
                    elif table == 2:
                        state = relevant[0].get('historical_equivalence', 'not_checked')
                    elif protocol == 'l20_batch32':
                        state = 'l20_protocol_archive_equivalence_not_claimed'
                    elif all(e.get('historical_equivalence') == 'full_dataset_archive_pixels_identical' for e in relevant):
                        state = ('full_pixels_match_archive_value_exact' if row['baseline_fresh'] == row['archive']
                                 else 'full_pixels_match_archive_value_differs')
                    else:
                        state = 'pending_full_archive_pixel_verification'
                    record[f'{model}_reproduction_status'] = state
                record['baseline_matches_paper_display'] = (rounding(row['baseline_fresh'], 3 if table == 1 else 2) == row['paper']
                                                            if row['baseline_fresh'] is not None else None)
                rows.append(record)
    complete = all(item['complete'] for item in progress)
    nav_states = {dataset: entries.get(('paper_historical', 'baseline', 2, dataset), {}).get('historical_equivalence', 'pending')
                  for dataset in NAV_DATASETS.values()}
    nav_verified = all(value in ('all_four_saved_metrics_exactly_equal_on_all_100_windows',
                                'replay_metrics_differ_from_archive') for value in nav_states.values())
    pixel_complete = ref['current_run']['baseline_table1_complete_pixel_replay']
    any_measurements = any(item['complete_datasets'] for item in progress)
    if complete and pixel_complete and nav_verified:
        status = 'evaluation_complete_with_documented_protocol_differences'
    elif complete:
        status = 'measurements_complete_reproduction_checks_pending'
    elif any_measurements:
        status = 'evaluation_in_progress'
    else:
        status = 'reference_audit_only_measurements_pending'
    ref['status'] = status
    ref['result_summary'] = {'display_protocols': list(RESULT_PROTOCOLS), 'progress': progress, 'rows': rows,
                             'all_requested_measurements_complete': complete,
                             'historical_table1_full_pixel_replay': pixel_complete,
                             'table2_full_archive_checks_complete': nav_verified,
                             'table2_archive_reproduction_by_dataset': nav_states,
                             'table2_all_saved_metrics_exact': nav_verified and all(
                                 value == 'all_four_saved_metrics_exactly_equal_on_all_100_windows' for value in nav_states.values()),
                             'l20_protocol_is_exact_a800_reproduction': False,
                             'table2_shared_between_views': True,
                             'completion_rule': 'All scores complete in both requested views, historical baseline Table1 full pixel checks, and full Table2 per-sample archive checks. Completed does not mean archive differences are zero.'}


def comparison_csv(ref):
    stream = io.StringIO(newline='')
    rows = ref['result_summary']['rows']
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    # csv writes Python's round-trip float representation; absent measured
    # values stay empty, with explicit pending completion fields beside them.
    writer.writerows(rows)
    return stream.getvalue()


def table2_pilot_notes(ref):
    mapping = ref['current_run'].get('table2_pilot_archive_mapping')
    if not mapping:
        return []
    timing, verification = mapping['pilot_timing'], mapping['strict_verification']
    count = len(mapping['windows'])
    same = sum(window['all_four_metrics_exact'] is True for window in mapping['windows'])
    validity = lambda value: '通过' if value else '未通过'
    return [
        '**Table 2 存档来源：**后续报告中的 RECON 100 窗口为混合来源：'
        '窗口 0–15 复用早期 L20 `navigation_largebatch_20260921`，窗口 16–99 来自后续 A800 运行。'
        '不能把该汇总统称为纯 A800 实测。pilot 窗口 0 与存档精确一致、窗口 16 有差异；'
        '窗口 16 新旧均为逻辑 CUDA0、seed58，导航 split 与 planner 源码相同。'
        '硬件、环境和首窗口编译状态没有被独立控制，这两个窗口不能证明 GPU 型号是差异原因。',
        f"**Table 2 pilot 核验：**来源核验（provenance）{validity(verification['provenance_valid'])}，"
        f"窗口覆盖核验（coverage）{validity(verification['coverage_valid'])}；"
        f"四项保存指标精确相等 {same}/{count}，有差异 {verification['metric_difference_sample_count']}/{count}。"
        f"严格逐项相等核验状态为 `{verification['state']}`、退出码 {verification['exit_code']}，"
        '精确复现核验未通过。该结果只覆盖 pilot 两窗口，完整 Table 2 指标仍须各模型每数据集 100 窗口完成后汇总。',
        f"**Table 2 pilot 耗时与显存：**窗口 0 为 {timing['window_0_seconds']:.4f} 秒；"
        f"窗口 16 为 {timing['window_16_seconds_inferred']:.4f} 秒（由日志滚动均值差分，文件时间交叉核验）。"
        f"pilot 结束时尚余 {timing['remaining_new_windows_after_pilot']} 个新窗口，按该两窗口初估约 "
        f"{timing['remaining_598_gpu_hours_using_window_16']:.2f}–{timing['remaining_598_gpu_hours_using_two_window_mean']:.2f} GPU 小时，"
        '约 5–5.5 天单 L20；未计各组启动/编译和 HuRoN 前置等待，RECON 基线速度不能代表其他数据集或 DINO。'
        f"PyTorch 分配峰值 {timing['pytorch_max_memory_mib']} MiB，"
        f"nvidia-smi 观测设备已用峰值 {timing['nvidia_smi_max_observed_used_mib']} MiB。",
        f"[Table 2 pilot 来源、指标与耗时证据]({ref['sources']['table2_pilot_archive_mapping']['path']})",
    ]


def results_markdown(ref, output):
    summary = ref['result_summary']
    status_labels = {'evaluation_in_progress': '评测进行中',
                     'reference_audit_only_measurements_pending': '参考核验已准备，实测待完成',
                     'measurements_complete_reproduction_checks_pending': '全部评分已完成，复现核验待完成',
                     'evaluation_complete_with_documented_protocol_differences': '评测及差异核验已完成（含已注明的协议变化与存档差异）'}
    labels = {'id_lpips': 'ID LPIPS↓', 'id_dreamsim': 'ID DreamSim↓', 'id_psnr': 'ID PSNR↑',
              'go_stanford_psnr': 'Go Stanford PSNR↑', 'tum_psnr': 'TUM RGB-D PSNR↑',
              'planetary_rover_psnr': 'Rover PSNR↑', 'office_go2_psnr': 'Office-Go2 PSNR↑'}
    for name, dataset in NAV_DATASETS.items():
        labels.update({f'{name}_ate': f'{LABELS[dataset]} ATE↓', f'{name}_rpe': f'{LABELS[dataset]} RPE↓'})
    def cell(value, signed=False):
        return '待完成' if value is None else format(value, '+.6f' if signed else '.6f')
    def metric_table(protocol, table):
        data = ref['comparisons'][protocol][f'table{table}']
        return md_table(['指标', '论文', '存档', 'OpenNWM 实测', 'DINO 实测', 'Δ DINO−OpenNWM', 'Δ OpenNWM−存档'],
                        [[labels[r['field']], fmt(r['paper'], 3 if table == 1 else 2), fmt(r['archive'], 6),
                          cell(r['baseline_fresh']), cell(r['dino_fresh']), cell(r['delta_dino_minus_baseline'], True),
                          cell(r['delta_baseline_minus_archive'], True)] for r in data])
    sections = ['# OpenNWM / DINO 主表评测结果',
                f"**{status_labels[ref['status']]}**。更新：{ref['generated_at']}（Asia/Taipei）。",
                'DINO：NWM Stage 2 joint 100k。空缺均标为“待完成”；存档列是参考值，未替代任何实测。'
                'LPIPS、DreamSim、ATE、RPE 越低越好，PSNR 越高越好。',
                '## 完成度（完整评分的数据集/主表指标）',
                md_table(['视图/表', '模型', '数据集', '主表指标', '已完整评分样本'],
                         [[('共用 Table 2' if p['table'] == 2 else PROTOCOL_LABELS[p['protocol']] + ' / Table 1'),
                           'OpenNWM' if p['model'] == 'baseline' else 'DINO',
                           f"{p['complete_datasets']}/{p['expected_datasets']}",
                           f"{p['complete_metrics']}/{p['expected_metrics']}",
                           f"{p['complete_samples']}/{p['expected_samples']}"]
                          for p in summary['progress'] if p['table'] == 1 or p['protocol'] == 'paper_historical']),
                'Table 1 的 ID 指标须 RECON、SCAND、HuRoN、TartanDrive 全部完成后等权宏平均；部分分片不计为完整评分。'
                '样本数按每模型、每视图列出，共享结果不代表重复运行。']
    for protocol in RESULT_PROTOCOLS:
        sections.extend([f'## Table 1：{PROTOCOL_LABELS[protocol]}', metric_table(protocol, 1)])
    sections.extend(['L20 后续口径：HuRoN 独立使用 329 样本、batch32（A800 存档为 batch128）；'
                     '六个不变数据集共享同模型完整历史实测，Rover 使用后续 horizons 的 L20 实测。'
                     '该视图用于同配置模型对比，不能称为 A800 精确复现。',
                     '## Table 2：两视图共用同一批 L20 实测',
                     '保持 CEM80、candidate microbatch80，每数据集每模型 100 个窗口；此表只展示一次。'
                     '与历史存档的差值保留，运行完成不等于精确复现。', metric_table('paper_historical', 2)])
    sections.extend(table2_pilot_notes(ref))
    sections.append('## 历史复现证据与存档差异')
    evidence = ref['current_run']['baseline_table1_pixel_evidence']
    sections.append(md_table(['历史 Table 1 / OpenNWM', '已验证像素相同', '完整像素重放'],
                             [[LABELS[d], f"{e['verified_count']}/{e['expected_count']}",
                               '通过' if e['complete_pixel_replay'] else '待完成'] for d, e in evidence.items()]))
    sections.append('历史存档的 Table 1 七项汇总与论文三位小数一致；本次是否复现由上面的新推理证据与实测差值分别判断。')
    nav_labels = {'pending': '待完成', 'not_checked': '待核验',
                  'not established by aggregate means alone': '待完整逐窗口核验',
                  'replay_metrics_differ_from_archive': '全量已核验，与历史存档有差异',
                  'all_four_saved_metrics_exactly_equal_on_all_100_windows': '100 窗口四项保存指标精确相等'}
    sections.append(md_table(['Table 2 / OpenNWM', '与历史存档逐窗口核验'],
                             [[LABELS[d], nav_labels.get(state, state)]
                              for d, state in summary['table2_archive_reproduction_by_dataset'].items()]))
    diag = ref['current_run']['historical_table1_rank1_diagnostics']
    if diag.get('controlled_first_batch_result_verified'):
        rank1_verified = all(any(s['rank'] == 1 for s in evidence[d]['shards'])
                             for d in (*ID_DATASETS, 'go_stanford', 'tum_rgbd', 'unitree_go2'))
        sections.append('历史 rank1 控制实验：同 GPU 恢复逻辑 cuda:1 后，首批 32/32 与存档字节及像素一致；'
                        '输入、VAE latent 和 RNG hash 不变。' +
                        ('随后完整 rank1 的主表相关 7 组均通过 125/125 像素核验；其他 rank 以全量覆盖表为准。'
                         if rank1_verified else '完整 rank1 仍以正式分片核验为准。') +
                        '旧 cuda:0 尝试已隔离，不计入实测。此证据与后续 Rover 的 A800/L20 差异分别记录。')
    sections.extend(['## 必要说明',
                     '- Table 2 论文 RECON/SCAND RPE 为 0.35/0.27；存档原值 0.35580294906679705/0.277794077525373，'
                     '常规四舍五入为 0.36/0.28，论文显示与截断一致。保留原值，不改指标。\n'
                     '- 训练条件不同：OpenNWM 为 Nav15 Pixel+Action，60k 预训练 + 3k warmup + 100k joint；'
                     'DINO 为 NavAnywhere v1，200k + 10k + 100k。结果是系统对比，不能把差异全部归因于 DINO 目标。',
                     f'[原始精度 CSV]({output / "comparison.csv"}) · [完整证据 JSON]({output / "reference.json"}) · '
                     f'[详细审计报告]({output / "preflight.md"})'])
    return '\n\n'.join(sections) + '\n'


def build(args):
    paper_path = REPO / 'website/content/paper.json'
    pdf_path = REPO / 'website/assets/paper/OpenNWM.pdf'
    paper, sweep, later = read_json(paper_path), read_json(args.sweep), read_json(args.later)
    pdf_source = source(pdf_path)
    if pdf_source.get('sha256') != PDF_SHA:
        raise ValueError('PDF differs from the user-confirmed audited paper')
    hist_model = sweep['models'][HIST_MODEL]
    historical = {}
    for dataset, value in hist_model['results']['direct_prediction'].items():
        record = value['time']
        historical[dataset] = {'metrics': record['metrics']['4s'],
                               'source_audit': source(record['source_audit']),
                               'inference': record['inference'], 'frame_label': '4s',
                               'frame_index': record['frame_indices']['4s']}
    corrected, navigation = {}, {}
    for key, group in later['groups'].items():
        if group.get('model') != LATER_MODEL or not group.get('metrics'):
            continue
        dataset, metrics = group['dataset'], group['metrics']
        if group['kind'] == 'direct':
            endpoint = '16obs' if dataset == 'planetary_rover' else '4s'
            corrected[dataset] = {'metrics': metrics['metrics'][endpoint],
                                  'endpoint': endpoint, 'frame_index': metrics['frame_indices'][endpoint],
                                  'inference': metrics['inference'], 'group': key,
                                  'metric_source': source(group['metric_path']),
                                  'gt_eval_dir': metrics['gt_eval_dir'],
                                  'pred_eval_dir': metrics['pred_eval_dir']}
        elif group['kind'] == 'navigation' and dataset in NAV_DATASETS.values():
            navigation[dataset] = {'metrics': {k: v for k, v in metrics.items() if k != 'sample_sources'},
                                   'metric_source': source(group['metric_path']), 'group': key,
                                   'sample_count': group['sample_count'],
                                   'expected_items': group['expected_items'],
                                   'complete_items': group['complete_items'],
                                   'reused_old_items': group['reused_old_items'],
                                   'sample_sources': metrics.get('sample_sources', [])}
    direct_rows = paper['results']['direct_prediction']['rows']
    nav_rows = [{k: v for k, v in row.items() if not k.startswith('tartan_drive_')}
                for row in paper['results']['navigation']['rows']]
    targets = {1: next(row for row in direct_rows if row['id'] == 'opennwm'),
               2: next(row for row in nav_rows if row['id'] == 'opennwm')}
    hist_summary, corrected_summary = table1_summary(historical), table1_summary(corrected)
    nav_summary = {f'{name}_{metric}': navigation[d]['metrics'][f'{d}_{suffix}']
                   for name, d in NAV_DATASETS.items()
                   for metric, suffix in (('ate', 'ate'), ('rpe', 'rpe_trans'))}
    code_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    split_paths = {
        'paper_huron_refreshed_500': Path('/file_system/vepfs/algorithm/dujun.nie/code/CompACT-eval-huron-fix/data_splits/sacson/test/time.pkl'),
        'later_huron_original_500_filter_329': Path('/file_system/vepfs/algorithm/dujun.nie/code/CompACT/data_splits/sacson/test/time.pkl'),
    }
    return {
        'schema_version': 1,
        'generated_at': datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
        'status': 'reference_audit_only_fresh_reproduction_not_established',
        'confirmed_scope': {'paper_tables': [1, 2], 'pdf_page': 7,
                            'protocols': list(PROTOCOLS),
                            'dino_stage': 'NWM Stage 2', 'dino_joint_steps': 100000},
        'sources': {'paper_pdf': pdf_source, 'paper_transcription': source(paper_path),
                    'historical_table1': source(args.sweep), 'later_direct_and_navigation': source(args.later),
                    'report_generator': source(__file__), 'repo_head': code_head,
                    'split_files': {name: source(path) for name, path in split_paths.items()}},
        'models': {
            'baseline': {'checkpoint': source(hist_model['checkpoint'], hash_file=False),
                         'checkpoint_sha256_from_archive': hist_model['sha256'],
                         'checkpoint_sha256_recomputed_this_audit': False,
                         'data': 'NavAnywhere 15-source Pixel+Action LAM pipeline',
                         'pretraining_steps': 60000, 'adapter_warmup_steps': 3000, 'joint_steps': 100000},
            'dino': {'checkpoint': source(DINO, hash_file=False),
                     'checkpoint_sha256_recomputed_this_audit': False,
                     'data': 'NavAnywhere v1 DINO-LAM pipeline',
                     'pretraining_steps': 200000, 'adapter_warmup_steps': 10000, 'joint_steps': 100000},
            'comparison_caveat': 'Training data and schedules differ: this is a system comparison, not a controlled DINO-objective-only ablation.'},
        'paper_targets': {'table1': {'rows': direct_rows, 'fields': TABLE1_FIELDS},
                          'table2': {'rows': nav_rows, 'fields': TABLE2_FIELDS},
                          'excluded_appendix_fields': 'TartanDrive navigation belongs to Table 10, not Table 2.'},
        'archive': {
            'paper_historical_table1': {'origin': 'archive', 'model': HIST_MODEL,
                                        'datasets': historical, 'summary': hist_summary,
                                        'paper_display_checks': checks(targets[1], hist_summary, TABLE1_FIELDS, 3)},
            'later_corrected_table1': {'origin': 'archive', 'model': LATER_MODEL,
                                      'datasets': corrected, 'summary': corrected_summary,
                                      'paper_display_checks': checks(targets[1], corrected_summary, TABLE1_FIELDS, 3)},
            'table2': {'origin': 'archive', 'model': LATER_MODEL, 'datasets': navigation,
                       'summary': nav_summary, 'paper_display_checks': checks(targets[2], nav_summary, TABLE2_FIELDS, 2)}},
        'aggregation': {'id': 'unweighted macro mean across RECON, SCAND, HuRoN, TartanDrive',
                        'psnr': 'mean of per-sample RGB PSNR in dB, not PSNR of globally pooled MSE',
                        'navigation': 'mean of 100 per-sample metrics for each main-table dataset'},
        'protocol_differences': {
            'huron': {'paper_population': 500, 'paper_split_sha256': 'd285d1b93967b992380f90b6e92edc2742577b251a863e88638e1f22d3ecfe6d',
                      'later_population': 329, 'later_raw_population': 500,
                      'later_split_sha256': later['huron']['time']['split_sha256'],
                      'later_effective_mapping_sha256': later['huron']['time']['effective_mapping_sha256'],
                      'note': 'Paper uses a refreshed 500-window split; later evaluation filters a different original 500-row split to 329 valid rows. This does not establish that the paper population was invalid.',
                      'historical_protocol_registry_caveat': 'The sweep global protocols section records the original split SHA. It is not authoritative evidence of the refreshed HuRoN split used by the paper run.'},
            'rover': {'paper_label': '4s', 'corrected_label': '16obs', 'population': 10,
                      'same_ground_truth_endpoint': True,
                      'paper_requested_horizons': [4], 'later_requested_horizons': [1, 2, 4],
                      'note': 'The same ten ground-truth endpoints are compared. Requested horizons and stochastic VAE RNG consumption require separate verification; matching requested horizons alone has not yet reproduced the later archive. Observed frames do not imply verified physical seconds.'},
            'table2_rounding': 'RECON/SCAND RPE .355802949/.277794078 display as .35/.27 in the PDF; these match truncation, whereas normal rounding gives .36/.28. Preserve raw values; do not silently adjust measurements.'},
        'current_run': optional_results(args.root, args.fresh_manifest),
    }


def md_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows])


def fmt(value, places=3):
    return '—' if value is None else f'{value:.{places}f}'


def fresh_cell(row, model):
    value = row[f'{model}_fresh']
    return f'{value:.9f}' if value is not None else row[f'{model}_status']


def fresh_tables(ref):
    sections = ['## 本次 OpenNWM / DINO 完整对比表（随评分更新）',
                'Δ = DINO 新测 − OpenNWM 新测；PSNR 越高越好，其他指标越低越好。'
                '只有完整样本评分才填入数值；待运行、部分样本和未评分保留显式状态。'
                '新测评分数值与“复现通过”是两项独立结论。']
    for protocol, label in (('paper_historical', '论文历史口径'), ('corrected', '后续存档协议（HuRoN batch128）'),
                            ('l20_batch32', 'L20 后续口径（HuRoN batch32 新协议）')):
        for table in (1, 2):
            sections.append(f'### {label} / Table {table}')
            sections.append(md_table(['指标', '论文 OpenNWM', '对应 archive', '本次 OpenNWM',
                                      '本次 DINO', 'Δ DINO−OpenNWM', 'Δ 新测OpenNWM−archive'],
                                     [[r['field'], fmt(r['paper'], 3 if table == 1 else 2),
                                       fmt(r['archive'], 9), fresh_cell(r, 'baseline'), fresh_cell(r, 'dino'),
                                       '待两模型完整' if r['delta_dino_minus_baseline'] is None else f"{r['delta_dino_minus_baseline']:+.9f}",
                                       '待OpenNWM完整' if r['delta_baseline_minus_archive'] is None else f"{r['delta_baseline_minus_archive']:+.9f}"]
                                      for r in ref['comparisons'][protocol][f'table{table}']]))
    sections.append('L20 后续口径：六个不变数据集共享同模型完整的历史口径新测；Rover 共享已完成的 L20 后续新测，'
                    '其 OpenNWM 结果尚未匹配 A800 存档；HuRoN 必须完成独立 batch32 的 329 个样本评分。'
                    '此视图是用户授权的新协议比较，不能称作 A800 精确复现。')
    sections.append('Table 2 各视图透明共用同一批 L20 新测，保持 CEM80、candidate microbatch=80。'
                    '新测与历史存档的差异逐项另列；RECON 存档混合早期 L20 与后续 A800 来源，'
                    '小样本允许继续运行不表示精确复现通过。')
    entries = ref['comparisons']['fresh_per_dataset']
    if entries:
        equivalence_labels = {'full_dataset_archive_pixels_identical': '全量像素与历史存档一致（指标另核验）',
                              'not_checked': '尚无对应的完整等价核验'}
        sections.append(md_table(['口径', '表', '模型', '数据集', '覆盖', '评分完整', '历史等价核验'],
                                 [[e['protocol'], e['table'], e['model'], LABELS.get(e['dataset'], e['dataset']),
                                   f"{e['count']}/{e['expected_count']}", '是' if e['complete'] else '否',
                                   equivalence_labels.get(e['historical_equivalence'], e['historical_equivalence'])]
                                  for e in entries]))
    return sections


def markdown(ref):
    old = ref['archive']['paper_historical_table1']
    later = ref['archive']['later_corrected_table1']
    nav = ref['archive']['table2']
    current = ref['current_run']
    sections = [
        '# OpenNWM / DINO 主表评测：运行前审计与参考值',
        f"生成时间：{ref['generated_at']}（Asia/Taipei）。",
        '**本文件首先记录论文目标和历史归档核验；归档指标不代表本次重新推理已完成。** '
        '本次复现完成与否必须根据新推理、样本覆盖、协议及指标核验判断。',
        '已确认范围：仓库 OpenNWM.pdf 第 7 页 Table 1 和 Table 2；DINO 版本 NWM Stage 2 的 joint 100k；'
        '同时做论文历史口径复现核验、后续口径（此前称“修正口径”）OpenNWM/DINO 对比。'
        '后续口径包含不同的采样窗口和随机数消费顺序，不代表已证明论文口径有错。',
        f"DINO checkpoint：`{DINO}`。\n\n论文 SHA256：`{PDF_SHA}`。",
        '## 论文 Table 1 目标（3 位小数）',
        md_table(['模型', 'ID LPIPS↓', 'ID DreamSim↓', 'ID PSNR↑', 'Go Stanford PSNR↑',
                  'TUM PSNR↑', 'Rover PSNR↑', 'Office-Go2 PSNR↑'],
                 [[row['model'], *(fmt(row[f]) for f in TABLE1_FIELDS)]
                  for row in ref['paper_targets']['table1']['rows']]),
        'ID 是 RECON、SCAND、HuRoN、TartanDrive 四个数据集等权宏平均。每个数据集 PSNR 为逐样本 dB 的算术平均。'
        'Rover 的论文栏头为 4s；后续口径按 16 个观测帧标注，物理秒数不能由索引单独确认。',
        '## OpenNWM Table 1：归档原值与论文核验',
        md_table(['指标', '论文', '论文历史 archive 原值', '常规 round 一致', '后续 archive 原值'],
                 [[field, fmt(v['paper']), f"{v['archive_raw']:.12f}",
                   '是' if v['matches_round_half_up'] else '否', f"{later['summary'][field]:.12f}"]
                  for field, v in old['paper_display_checks'].items()]),
        '历史 archive 的 7 项汇总指标均在常规三位小数显示下与论文一致。'
        '这建立了论文数字与历史结果的对应关系；不是本次新测结果。',
        md_table(['数据集', '历史样本数', '历史 LPIPS', '历史 DreamSim', '历史 PSNR',
                  '后续样本数', '后续 LPIPS', '后续 DreamSim', '后续 PSNR'],
                 [[LABELS[d], v['metrics']['sample_count'],
                   *(fmt(v['metrics'][k], 9) for k in ('lpips_alex', 'dreamsim', 'psnr')),
                   later['datasets'][d]['metrics']['sample_count'],
                   *(fmt(later['datasets'][d]['metrics'][k], 9) for k in ('lpips_alex', 'dreamsim', 'psnr'))]
                  for d, v in old['datasets'].items() if d in later['datasets']]),
        '## 论文 Table 2 目标（2 位小数）',
        md_table(['模型', 'RECON ATE↓', 'RECON RPE↓', 'SCAND ATE↓', 'SCAND RPE↓',
                  'Office-Go2 ATE↓', 'Office-Go2 RPE↓'],
                 [[row['model'], *(fmt(row[f], 2) for f in TABLE2_FIELDS)]
                  for row in ref['paper_targets']['table2']['rows']]),
        '网站 JSON 中的 TartanDrive 导航属于附录 Table 10，未混入 Table 2。',
        '## OpenNWM Table 2：归档原值与显示差异',
        md_table(['指标', '论文', 'archive 原值', '常规 round', '截断', '常规 round 一致'],
                 [[field, fmt(v['paper'], 2), f"{v['archive_raw']:.12f}", fmt(v['round_half_up'], 2),
                   fmt(v['truncate'], 2), '是' if v['matches_round_half_up'] else '否']
                  for field, v in nav['paper_display_checks'].items()]),
        '各数据集 100 个样本。RECON RPE=0.35580294906679705、SCAND RPE=0.277794077525373；'
        '正常四舍五入显示为 0.36/0.28，论文 0.35/0.27 与截断一致。此处保留真实原值，不为了匹配论文改动指标。',
        '## 两套口径的边界',
        '- HuRoN：论文采用 refreshed 500 窗口，SHA256 `d285d1b93967b992380f90b6e92edc2742577b251a863e88638e1f22d3ecfe6d`；'
        '后续从另一个 original 500 窗口过滤为 329，SHA256 `9d1c5239dfbf0f73d4aaee29c5abda31f28e97bbaf02f5bfc76cfd4f7f63953e`。'
        '新旧不是同一批窗口，不能据此说论文 500 样本原本无效。历史 sweep 的全局 protocol registry 仍写 original SHA，需以运行级证据识别实际 split。\n'
        '- Rover：两套比较的 10 个 GT endpoint 相同；历史只请求 horizons=[4]，后续请求 [1,2,4]，'
        '存档 PSNR 为 12.574183369 vs 12.968571091。horizon 配置及 VAE 随机数流需分别核验；'
        '仅设置后续 horizons 并不足以证明已重放后续存档。\n'
        '- 宏平均：后续口径仍按四个 ID 数据集等权，不将 HuRoN 329 与其余 500 混成逐样本加权平均。\n'
        '- 六组共享新测：RECON、SCAND、TartanDrive、Go Stanford、TUM RGB-D、Office-Go2 的每组 500 张 GT '
        '字节/像素及 manifest hash 一致，后续 archive 的 endpoint 链接指向历史源。审计证据完整时，'
        '后续与 L20 视图引用同模型完整的本次历史口径预测和指标，记录 shared_between_protocol_views 与 reuse_evidence；'
        '不借用 archive 数值。若另有独立后续口径新测，优先采用独立新测。\n'
        '- 训练条件：论文 OpenNWM 使用 Nav15 Pixel+Action、60k 预训练 + 3k warmup + 100k joint；'
        'DINO 使用 NavAnywhere v1、200k 预训练 + 10k warmup + 100k joint。数据和 schedule 并未完全控制，'
        '只能解释为系统对比，不能把全部差异归因于 DINO 目标。',
        '## 设备选择与协议变化',
        '后续 HuRoN 历史推理 batch=128，记录的峰值显存约 56.5 GiB，当前 L20 无法原样容纳。'
        '要保持原协议，需要足够显存的设备；改小 batch 会改变随机 VAE 的随机数消费，必须作为新协议另列。'
        '用户已授权使用 L20：HuRoN 两模型使用独立 batch32 新协议，Table 2 保持 candidate microbatch80。'
        '新测与对应历史存档差异另列，不能把这项授权写成精确复现通过，也不能用 archive 代替 DINO 新测。'
        + (' 用户决定记录已另附 `evaluation_decisions.json`，以下述记录为准：' +
           json.dumps(ref['operational_limits']['user_decisions'], ensure_ascii=False)
           if 'user_decisions' in ref['operational_limits'] else ''),
        '## 本次执行证据（与 archive 分开）',
        f"已发现 archive 重评分文件 {len(current['archive_rescore'])} 个；显式提供的新推理结果记录 {len(current['fresh_inference'])} 个。",
    ]
    if not current['fresh_inference']:
        sections.append('**新测结果待运行/纳入：目前没有可据以发布 DINO 对比的本次完整结果。**')
    if current['archive_rescore_checks']:
        sections.append(md_table(['重评分口径', '数据集', '样本数一致', 'ΔLPIPS', 'ΔDreamSim', 'ΔPSNR', '三项完全相等'],
                                 [[r['protocol'], LABELS[r['dataset']], '是' if r['count_matches'] else '否',
                                   *(f"{r['delta_from_archive'][k]:+.12g}" for k in ('lpips_alex', 'dreamsim', 'psnr')),
                                   '是' if r['all_three_exactly_equal'] else '否']
                                  for r in current['archive_rescore_checks']]))
    if current['baseline_table1_archive_pixel_checks']:
        sections.append(md_table(['本次OpenNWM重新推理像素核验', '已比较', '像素完全相同', '完整目标数'],
                                 [[LABELS[d], v['compared'], v['pixel_identical'], 10 if d == 'planetary_rover' else 500]
                                  for d, v in current['baseline_table1_pixel_coverage'].items()]))
        sections.append('全部 Table 1 主表样本像素级重放已完成：' +
                        ('是' if current['baseline_table1_complete_pixel_replay'] else '否；当前为部分证据') + '。')
    historical_diagnostics = current['historical_table1_rank1_diagnostics']
    if historical_diagnostics:
        sections.append('### 历史 Table 1 rank1：逻辑 CUDA index 控制实验')
        if historical_diagnostics['controlled_first_batch_result_verified']:
            gpu = historical_diagnostics['selected_physical_gpus'][0]
            sections.append(f'同一物理 L20 GPU{gpu}、相同 rank seed=1 的首批 32 个样本：'
                            '使用逻辑 cuda:0 时与存档 0/32 相同，恢复历史逻辑 cuda:1 后 32/32 像素及 PNG 字节完全相同。'
                            '两次输入观测、action、VAE 输入与 latent，以及 VAE 前后的 CUDA RNG hash 均一致；'
                            '差异出现在 VAE 编码之后，并依赖逻辑设备编号。精确编译内核机制尚未证实，不能归因于 VAE seed。')
        else:
            sections.append('控制实验记录尚不足以确认预期的 0/32→32/32 结果；见下列逐项证据。')
        sections.append('控制实验本身仅覆盖 rank1 首批 32 张；正式重放的覆盖范围以上面的分片像素核验为准，'
                        '不能由首批结果推断整个主表已完整重放。该诊断独立于后续 Rover 的 A800/L20 差异。')
        relocation = historical_diagnostics['relocation']
        sections.append(f"旧 cuda:0 尝试迁移状态：{relocation.get('state', '待记录')}；"
                        f"归档目录：`{relocation.get('relocated_directory', '待记录')}`。"
                        '旧尝试与诊断输出不计入 canonical fresh 结果。旧 metadata 保留原路径，'
                        '应按迁移清单和已记录 SHA256 定位旧文件；本报告仅记录证据文件的 hash，不重新计算迁移清单中全部文件的 hash。')
        sections.append(md_table(['历史 rank1 诊断证据', '路径', 'SHA256'],
                                 [[name, historical_diagnostics[key]['path'], historical_diagnostics[key].get('sha256', '—')]
                                  for name, key in (('控制实验 JSON', 'source'), ('控制实验说明', 'markdown_source'),
                                                    ('旧尝试迁移记录', 'relocation_source'))]))
    if current['baseline_table1_later_archive_pixel_checks']:
        sections.append(md_table(['后续口径 OpenNWM 像素核验', 'horizon', '已比较', '像素完全相同'],
                                 [[LABELS[item['dataset']], horizon, values['compared'], values['pixel_identical']]
                                  for item in current['baseline_table1_later_archive_pixel_checks']
                                  for horizon, values in item['data']['counts_by_horizon'].items()]))
        failed = [LABELS[item['dataset']] for item in current['baseline_table1_later_archive_pixel_checks']
                  if not item['data']['all_pixels_identical']]
        if failed:
            sections.append('**后续口径重新推理尚未匹配存档：' + '、'.join(failed) +
                            '。下表新测数值可用于本次相同运行配置下的模型对比；不能据此宣称后续存档已复现一致。**')
    diagnostics = current['later_rover_diagnostics']
    if 'rover_native_comparison' in diagnostics:
        comparison = diagnostics['rover_native_comparison']['data']
        sections.append(md_table(['Rover 原入口诊断比较对象', 'horizon', '图像数', '字节完全相同'],
                                 [[target, horizon, values['compared'], values['byte_identical']]
                                  for target, row in comparison.items()
                                  for horizon, values in row['by_horizon'].items()]))
        run = diagnostics.get('rover_native_run', {}).get('data', {})
        checkpoint = diagnostics.get('baseline_checkpoint_hash', {}).get('data', {})
        sections.append(f"原入口诊断退出码：{run.get('exit_code', '待记录')}；"
                        f"checkpoint 完整 SHA256 匹配存档：{checkpoint.get('matches_archive', '待记录')}。"
                        '原入口与本次封装输出相同，说明此次差异不来自封装入口或复制 split。')
    if 'cuda_noise_launch_policy' in diagnostics:
        sections.append('CUDA 随机数源码证据表明，大张量 randn 的线程排列依赖设备属性；'
                        '实测 L20 的 grid=552，按 A800 参考属性推算为 864。A800 属性不是该历史运行的直接观测，'
                        '这是一项具体机制候选；尚未分别量化它与 BF16、TF32、编译内核数值差异对指标的影响。'
                        '后续存档严格复现宜使用原 A800 环境，不能把当前 L20 结果声明为已经一致。')
    if diagnostics:
        sections.append(md_table(['诊断证据', '路径', 'SHA256'],
                                 [[name, item['source']['path'], item['source']['sha256']]
                                  for name, item in diagnostics.items()]))
    sections.append('### Table 2 小样本验证状态')
    sections.extend(table2_pilot_notes(ref))
    sections.append('仅核验请求的窗口及四项保存指标（ATE、RPE、位置差、角度差），不代表每个数据集 100 窗口的全量复现。')
    sections.append(md_table(['模型/数据集', '策略', '请求窗口', '已核验数', '执行状态', '四项 all_metrics_exact', 'provenance_valid', '核验结论'],
                             [[f"{item['data'].get('model', item.get('model', 'baseline'))}/{item['data'].get('dataset', item.get('dataset', 'recon'))}",
                               item['verification_policy'],
                               item['data'].get('requested_indices', item.get('requested_indices_spec', '待记录')),
                               item['data'].get('count', 0), item['data']['state'],
                               item['data'].get('all_metrics_exact', '待验证'), item['data'].get('provenance_valid', '待验证'),
                               item['data'].get('verification_outcome', '逐项精确性见 all_metrics_exact')]
                              for item in current['table2_subset_verifications']]))
    sections.append('L20 允许指标差异的核验即使执行状态 completed，也只表示来源、覆盖及继续运行条件通过；'
                    '是否与存档精确相等仍以 all_metrics_exact 为准。strict 与 L20 核验文件独立保留。')
    for item in current['table2_subset_verifications']:
        if item['data'].get('error'):
            sections.append('小样本验证未通过：' + item['data']['error'])
    sections.extend(fresh_tables(ref))
    for category in ('archive_rescore', 'fresh_inference'):
        if current[category]:
            sections.append(md_table(['来源分类', '路径', 'SHA256'],
                                     [[category, item['source']['path'], item['source'].get('sha256', '—')]
                                      for item in current[category]]))
    sections += [
        '本脚本仅收录新证据，不自动宣称复现通过。需要继续核对：checkpoint、split/有效样本映射、'
        'horizon、batch/rank/seed、VAE 随机数消费、完整样本数、像素/指标差异。'
        '历史预测重评分只能验证评分路径，不能代替重新推理。',
        '## 可追溯来源',
        md_table(['来源', '绝对路径', 'SHA256'],
                 [[name, value['path'], value.get('sha256', '—')]
                  for name, value in ref['sources'].items() if isinstance(value, dict) and 'path' in value]),
        f"当前代码 HEAD：`{ref['sources']['repo_head']}`。"
        + ('本次单独的 checkpoint_runtime_identity.json 已重新计算两个 checkpoint SHA256；'
           f"baseline 与 archive 一致={ref['models']['baseline']['recomputed_sha256_matches_archive']}。"
           if 'checkpoint_runtime_identity' in ref['sources'] else
           'checkpoint 本审计只记录路径/大小/时间，baseline 已有 archive SHA256；尚未纳入本次重算记录。') +
        '完整原始数值、每个 metrics/source_audit 的文件 hash、导航逐样本来源及新增证据见 `reference.json`。',
    ]
    return '\n\n'.join(sections) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--sweep', type=Path, default=SWEEP)
    parser.add_argument('--later', type=Path, default=LATER)
    parser.add_argument('--fresh-manifest', action='append', type=Path, default=[],
                        help='JSON fresh results with protocol paper_historical/corrected/l20_batch32; fresh_scores manifests and mb80 navigation aggregates are auto-discovered')
    args = parser.parse_args()
    output = args.output_dir or args.root / 'audit'
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ref = build(args)
        enrich_current(ref, args.root)
        for name, contents in (('reference.json', json.dumps(ref, ensure_ascii=False, indent=2) + '\n'),
                               ('preflight.md', markdown(ref)),
                               ('comparison.csv', comparison_csv(ref)),
                               ('RESULTS.md', results_markdown(ref, output.resolve()))):
            path = output / name
            with tempfile.NamedTemporaryFile(mode='w', dir=output, prefix=name + '.', suffix='.tmp', delete=False) as stream:
                stream.write(contents)
                temporary = Path(stream.name)
            temporary.replace(path)
            print(path)
    print('Report status: ' + ref['status'] + '; measurement completion and archive agreement are reported separately.')


if __name__ == '__main__':
    main()
