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
PROTOCOL_LABELS = {'paper_historical': 'Historical paper protocol', 'l20_batch32': 'Later L20 protocol (HuRoN batch32)'}
OOD_FIELDS = {'go_stanford_psnr': 'go_stanford', 'tum_psnr': 'tum_rgbd',
              'planetary_rover_psnr': 'planetary_rover', 'office_go2_psnr': 'unitree_go2'}
NAV_DATASETS = {'recon': 'recon', 'scand': 'scand', 'office_go2': 'unitree_go2'}
TABLE1_FIELDS = ('id_lpips', 'id_dreamsim', 'id_psnr', *OOD_FIELDS)
TABLE2_FIELDS = tuple(f'{dataset}_{metric}' for dataset in NAV_DATASETS for metric in ('ate', 'rpe'))
LABELS = {'recon': 'RECON', 'scand': 'SCAND', 'huron': 'HuRoN',
          'tartan_drive': 'TartanDrive', 'go_stanford': 'Go Stanford',
          'tum_rgbd': 'TUM RGB-D', 'planetary_rover': 'Planetary Rover',
          'unitree_go2': 'Office-Go2', 'uzh_fpv': 'UZH FPV (outside main tables)'}


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
            return None, 'Pending inference/scoring'
        if not entry['complete']:
            return None, f"Incomplete ({entry['count']}/{entry['expected_count']})"
        return entry['metrics'][metric], 'Scored; reproduction checks reported separately'

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
                    status = 'Scored; reproduction checks reported separately' if value is not None else f'Pending complete scoring (ID {len(available)}/4 datasets)'
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
    validity = lambda value: 'Passed' if value else 'Failed'
    return [
        '**Table 2 archive provenance:** The later report combines two sources for its 100 RECON windows: '
        'windows 0–15 reuse the earlier L20 `navigation_largebatch_20260921` run; windows 16–99 come from a later A800 run. '
        'The aggregate is not an A800-only measurement. Pilot window 0 matches the archive exactly; window 16 differs. '
        'Both window 16 runs used logical CUDA0, seed58, the same navigation split and planner source. '
        'Hardware, environment and first-window compilation were not independently controlled, so these two windows cannot establish GPU type as the cause.',
        f"**Table 2 pilot verification:** provenance {validity(verification['provenance_valid'])}; "
        f"coverage {validity(verification['coverage_valid'])}; "
        f"all four saved metrics match in {same}/{count} windows and differ in {verification['metric_difference_sample_count']}/{count}. "
        f"Strict equality check: `{verification['state']}`, exit code {verification['exit_code']}. "
        'Exact reproduction failed. These checks cover only two pilot windows; full Table 2 aggregates require 100 windows per model and dataset.',
        f"**Table 2 pilot time and memory:** window 0: {timing['window_0_seconds']:.4f} seconds; "
        f"window 16: {timing['window_16_seconds_inferred']:.4f} seconds (inferred from rolling log means and checked against file timestamps). "
        f"The pilot left {timing['remaining_new_windows_after_pilot']} new windows, roughly "
        f"{timing['remaining_598_gpu_hours_using_window_16']:.2f}–{timing['remaining_598_gpu_hours_using_two_window_mean']:.2f} GPU hours "
        '(5–5.5 days on one L20), excluding startup, compilation and HuRoN prerequisites. RECON baseline timing does not represent other datasets or DINO. '
        f"Peak PyTorch allocation: {timing['pytorch_max_memory_mib']} MiB; "
        f"peak device usage observed by nvidia-smi: {timing['nvidia_smi_max_observed_used_mib']} MiB.",
        f"[Table 2 pilot provenance, metrics and timing]({ref['sources']['table2_pilot_archive_mapping']['path']})",
    ]


def results_markdown(ref, output):
    summary = ref['result_summary']
    status_labels = {'evaluation_in_progress': 'Evaluation in progress',
                     'reference_audit_only_measurements_pending': 'Reference audit ready; measurements pending',
                     'measurements_complete_reproduction_checks_pending': 'Scoring complete; reproduction checks pending',
                     'evaluation_complete_with_documented_protocol_differences': 'Evaluation and difference audit complete, with documented protocol changes and archive differences'}
    labels = {'id_lpips': 'ID LPIPS↓', 'id_dreamsim': 'ID DreamSim↓', 'id_psnr': 'ID PSNR↑',
              'go_stanford_psnr': 'Go Stanford PSNR↑', 'tum_psnr': 'TUM RGB-D PSNR↑',
              'planetary_rover_psnr': 'Rover PSNR↑', 'office_go2_psnr': 'Office-Go2 PSNR↑'}
    for name, dataset in NAV_DATASETS.items():
        labels.update({f'{name}_ate': f'{LABELS[dataset]} ATE↓', f'{name}_rpe': f'{LABELS[dataset]} RPE↓'})
    def cell(value, signed=False):
        return 'Pending' if value is None else format(value, '+.6f' if signed else '.6f')
    def metric_table(protocol, table):
        data = ref['comparisons'][protocol][f'table{table}']
        return md_table(['Metric', 'Paper', 'Archive', 'OpenNWM measured', 'DINO measured', 'Δ DINO−OpenNWM', 'Δ OpenNWM−archive'],
                        [[labels[r['field']], fmt(r['paper'], 3 if table == 1 else 2), fmt(r['archive'], 6),
                          cell(r['baseline_fresh']), cell(r['dino_fresh']), cell(r['delta_dino_minus_baseline'], True),
                          cell(r['delta_baseline_minus_archive'], True)] for r in data])
    sections = ['# OpenNWM / DINO main-table evaluation results',
                f"**{status_labels[ref['status']]}**. Updated: {ref['generated_at']} (Asia/Taipei).",
                'DINO: NWM Stage 2 joint 100k. Missing results are marked Pending; archive columns are references and never replace measurements. '
                'Lower LPIPS, DreamSim, ATE and RPE are better; higher PSNR is better.',
                '## Progress (fully scored datasets and main-table metrics)',
                md_table(['View/table', 'Model', 'Dataset', 'Main-table metrics', 'Fully scored samples'],
                         [[('Shared Table 2' if p['table'] == 2 else PROTOCOL_LABELS[p['protocol']] + ' / Table 1'),
                           'OpenNWM' if p['model'] == 'baseline' else 'DINO',
                           f"{p['complete_datasets']}/{p['expected_datasets']}",
                           f"{p['complete_metrics']}/{p['expected_metrics']}",
                           f"{p['complete_samples']}/{p['expected_samples']}"]
                          for p in summary['progress'] if p['table'] == 1 or p['protocol'] == 'paper_historical']),
                'Table 1 ID metrics require complete RECON, SCAND, HuRoN and TartanDrive results, then an equal-weight macro average. Partial shards are incomplete. '
                'Sample counts are listed by model and view; shared results do not imply repeated runs.']
    for protocol in RESULT_PROTOCOLS:
        sections.extend([f'## Table 1: {PROTOCOL_LABELS[protocol]}', metric_table(protocol, 1)])
    sections.extend(['Later L20 protocol: HuRoN uses an independent 329-sample batch32 run (the A800 archive used batch128); '
                     'six unchanged datasets share complete historical-protocol measurements for the same model, while Rover uses later-horizon L20 measurements. '
                     'This view compares models under the same configuration; it is not exact A800 reproduction.',
                     '## Table 2: shared L20 measurements for both views',
                     'CEM80, candidate microbatch80, 100 windows per dataset and model; this table appears once. '
                     'Differences from the historical archive remain visible. Completing a run does not establish exact reproduction.', metric_table('paper_historical', 2)])
    sections.extend(table2_pilot_notes(ref))
    sections.append('## Historical reproduction evidence and archive differences')
    evidence = ref['current_run']['baseline_table1_pixel_evidence']
    sections.append(md_table(['Historical Table 1 / OpenNWM', 'Verified identical pixels', 'Complete pixel replay'],
                             [[LABELS[d], f"{e['verified_count']}/{e['expected_count']}",
                               'Passed' if e['complete_pixel_replay'] else 'Pending'] for d, e in evidence.items()]))
    sections.append('The seven historical Table 1 aggregates match the paper at three decimals. Current reproduction is assessed from fresh inference evidence and measurement differences above.')
    nav_labels = {'pending': 'Pending', 'not_checked': 'Verification pending',
                  'not established by aggregate means alone': 'Complete per-window verification pending',
                  'replay_metrics_differ_from_archive': 'Fully checked; differs from the historical archive',
                  'all_four_saved_metrics_exactly_equal_on_all_100_windows': 'All four saved metrics match exactly in all 100 windows'}
    sections.append(md_table(['Table 2 / OpenNWM', 'Per-window historical archive comparison'],
                             [[LABELS[d], nav_labels.get(state, state)]
                              for d, state in summary['table2_archive_reproduction_by_dataset'].items()]))
    diag = ref['current_run']['historical_table1_rank1_diagnostics']
    if diag.get('controlled_first_batch_result_verified'):
        rank1_verified = all(any(s['rank'] == 1 for s in evidence[d]['shards'])
                             for d in (*ID_DATASETS, 'go_stanford', 'tum_rgbd', 'unitree_go2'))
        sections.append('Historical rank1 control: restoring logical cuda:1 on the same GPU produced 32/32 byte- and pixel-identical first-batch samples; '
                        'inputs, VAE latents and RNG hashes were unchanged. ' +
                        ('All seven relevant rank1 groups subsequently passed 125/125 pixel checks; see the full coverage table for other ranks. '
                         if rank1_verified else 'Complete rank1 coverage remains subject to formal shard verification. ') +
                        'The earlier cuda:0 attempt is isolated and excluded from measurements. This evidence is separate from the later Rover A800/L20 difference.')
    sections.extend(['## Notes',
                     '- Table 2 paper RECON/SCAND RPE is 0.35/0.27; archive values are 0.35580294906679705/0.277794077525373. '
                     'Normal rounding gives 0.36/0.28; the paper matches truncation. Raw metrics are preserved.\n'
                     '- Training conditions differ: OpenNWM uses Nav15 Pixel+Action, 60k pretraining + 3k warmup + 100k joint; '
                     'DINO uses NavAnywhere v1, 200k + 10k + 100k. This is a system comparison; differences cannot be attributed solely to the DINO objective.',
                     f'[Full-precision CSV]({output / "comparison.csv"}) · [Complete evidence JSON]({output / "reference.json"}) · '
                     f'[Detailed audit]({output / "preflight.md"})'])
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
    sections = ['## Current OpenNWM / DINO comparison (updated as scoring completes)',
                'Δ = fresh DINO − fresh OpenNWM; higher PSNR and lower other metrics are better. '
                'Values appear only after complete scoring; pending, partial and unscored results retain explicit status. '
                'Fresh measurements and successful reproduction are separate conclusions.']
    for protocol, label in (('paper_historical', 'Historical paper protocol'), ('corrected', 'Later archive protocol (HuRoN batch128)'),
                            ('l20_batch32', 'Later L20 protocol (new HuRoN batch32 protocol)')):
        for table in (1, 2):
            sections.append(f'### {label} / Table {table}')
            sections.append(md_table(['Metric', 'Paper OpenNWM', 'Matching archive', 'Fresh OpenNWM',
                                      'Fresh DINO', 'Δ DINO−OpenNWM', 'Δ fresh OpenNWM−archive'],
                                     [[r['field'], fmt(r['paper'], 3 if table == 1 else 2),
                                       fmt(r['archive'], 9), fresh_cell(r, 'baseline'), fresh_cell(r, 'dino'),
                                       'Both models incomplete' if r['delta_dino_minus_baseline'] is None else f"{r['delta_dino_minus_baseline']:+.9f}",
                                       'OpenNWM incomplete' if r['delta_baseline_minus_archive'] is None else f"{r['delta_baseline_minus_archive']:+.9f}"]
                                      for r in ref['comparisons'][protocol][f'table{table}']]))
    sections.append('Later L20 protocol: six unchanged datasets share complete fresh historical-protocol results for the same model; Rover shares completed later L20 measurements, '
                    'whose OpenNWM results do not yet match the A800 archive. HuRoN requires independent scoring of all 329 batch32 samples. '
                    'This authorized protocol comparison is not exact A800 reproduction.')
    sections.append('Table 2 views share the same fresh L20 measurements with CEM80 and candidate microbatch=80. '
                    'Fresh/archive differences are listed separately. The RECON archive combines earlier L20 and later A800 runs; '
                    'permission to continue after a small pilot does not establish exact reproduction.')
    entries = ref['comparisons']['fresh_per_dataset']
    if entries:
        equivalence_labels = {'full_dataset_archive_pixels_identical': 'All pixels match the historical archive (metrics checked separately)',
                              'not_checked': 'Complete equivalence check unavailable'}
        sections.append(md_table(['Protocol', 'Table', 'Model', 'Dataset', 'Coverage', 'Scoring complete', 'Historical equivalence check'],
                                 [[e['protocol'], e['table'], e['model'], LABELS.get(e['dataset'], e['dataset']),
                                   f"{e['count']}/{e['expected_count']}", 'Yes' if e['complete'] else 'No',
                                   equivalence_labels.get(e['historical_equivalence'], e['historical_equivalence'])]
                                  for e in entries]))
    return sections


def markdown(ref):
    old = ref['archive']['paper_historical_table1']
    later = ref['archive']['later_corrected_table1']
    nav = ref['archive']['table2']
    current = ref['current_run']
    sections = [
        '# OpenNWM / DINO main-table evaluation: preflight audit and references',
        f"Generated: {ref['generated_at']} (Asia/Taipei).",
        '**This report records paper targets and historical archive checks; archive metrics do not establish completed fresh inference.** '
        'Reproduction requires fresh inference, sample coverage, protocol and metric verification.',
        'Scope: Tables 1 and 2 on page 7 of the repository OpenNWM.pdf; DINO NWM Stage 2 joint 100k; '
        'historical paper-protocol reproduction checks and a later-protocol (previously called corrected) OpenNWM/DINO comparison. '
        'The later protocol changes sampled windows and RNG consumption order; this does not establish an error in the paper protocol.',
        f"DINO checkpoint: `{DINO}`.\n\nPaper SHA256: `{PDF_SHA}`.",
        '## Paper Table 1 targets (three decimals)',
        md_table(['Model', 'ID LPIPS↓', 'ID DreamSim↓', 'ID PSNR↑', 'Go Stanford PSNR↑',
                  'TUM PSNR↑', 'Rover PSNR↑', 'Office-Go2 PSNR↑'],
                 [[row['model'], *(fmt(row[f]) for f in TABLE1_FIELDS)]
                  for row in ref['paper_targets']['table1']['rows']]),
        'ID is an equal-weight macro average of RECON, SCAND, HuRoN and TartanDrive. Dataset PSNR is the arithmetic mean of per-sample dB values. '
        'The paper labels Rover as 4s; the later protocol labels 16 observed frames. Frame indices alone do not establish physical seconds.',
        '## OpenNWM Table 1: raw archive values and paper checks',
        md_table(['Metric', 'Paper', 'Historical raw archive', 'Normal rounding matches', 'Later raw archive'],
                 [[field, fmt(v['paper']), f"{v['archive_raw']:.12f}",
                   'Yes' if v['matches_round_half_up'] else 'No', f"{later['summary'][field]:.12f}"]
                  for field, v in old['paper_display_checks'].items()]),
        'All seven historical archive aggregates match the paper when normally rounded to three decimals. '
        'This links the paper numbers to historical results; these are not fresh measurements.',
        md_table(['Dataset', 'Historical samples', 'Historical LPIPS', 'Historical DreamSim', 'Historical PSNR',
                  'Later samples', 'Later LPIPS', 'Later DreamSim', 'Later PSNR'],
                 [[LABELS[d], v['metrics']['sample_count'],
                   *(fmt(v['metrics'][k], 9) for k in ('lpips_alex', 'dreamsim', 'psnr')),
                   later['datasets'][d]['metrics']['sample_count'],
                   *(fmt(later['datasets'][d]['metrics'][k], 9) for k in ('lpips_alex', 'dreamsim', 'psnr'))]
                  for d, v in old['datasets'].items() if d in later['datasets']]),
        '## Paper Table 2 targets (two decimals)',
        md_table(['Model', 'RECON ATE↓', 'RECON RPE↓', 'SCAND ATE↓', 'SCAND RPE↓',
                  'Office-Go2 ATE↓', 'Office-Go2 RPE↓'],
                 [[row['model'], *(fmt(row[f], 2) for f in TABLE2_FIELDS)]
                  for row in ref['paper_targets']['table2']['rows']]),
        'TartanDrive navigation in the website JSON belongs to Appendix Table 10 and is excluded from Table 2.',
        '## OpenNWM Table 2: raw archive values and display differences',
        md_table(['Metric', 'Paper', 'Raw archive', 'Normal rounding', 'Truncation', 'Normal rounding matches'],
                 [[field, fmt(v['paper'], 2), f"{v['archive_raw']:.12f}", fmt(v['round_half_up'], 2),
                   fmt(v['truncate'], 2), 'Yes' if v['matches_round_half_up'] else 'No']
                  for field, v in nav['paper_display_checks'].items()]),
        'Each dataset has 100 samples. RECON RPE=0.35580294906679705; SCAND RPE=0.277794077525373. '
        'Normal rounding gives 0.36/0.28; the paper values 0.35/0.27 match truncation. Raw values are preserved without adjusting metrics to match the paper.',
        '## Protocol boundaries',
        '- HuRoN: the paper uses refreshed 500 windows, SHA256 `d285d1b93967b992380f90b6e92edc2742577b251a863e88638e1f22d3ecfe6d`; '
        'the later protocol filters a different original 500-window split to 329, SHA256 `9d1c5239dfbf0f73d4aaee29c5abda31f28e97bbaf02f5bfc76cfd4f7f63953e`. '
        'These are different populations, so this does not invalidate the paper samples. The historical sweep registry still records the original SHA; run-level evidence identifies the actual split.\n'
        '- Rover: both comparisons use the same 10 GT endpoints. Historical requests use horizons=[4], later requests use [1,2,4]; '
        'archive PSNR is 12.574183369 vs 12.968571091. Horizon settings and VAE RNG streams require separate checks; '
        'matching later horizons alone does not prove replay of the later archive.\n'
        '- Macro averaging: the later protocol retains equal weights for four ID datasets, without sample-weighting HuRoN 329 against the other 500-sample datasets.\n'
        '- Six shared measurements: the 500 GT images in each of RECON, SCAND, TartanDrive, Go Stanford, TUM RGB-D and Office-Go2 '
        'have identical bytes/pixels and manifest hashes; later archive endpoint links point to historical sources. With complete audit evidence, '
        'later and L20 views reuse the same model\'s complete fresh historical-protocol predictions and metrics, recording shared_between_protocol_views and reuse_evidence. '
        'Archive values are never substituted. Independent fresh later-protocol measurements take precedence when available.\n'
        '- Training: paper OpenNWM uses Nav15 Pixel+Action, 60k pretraining + 3k warmup + 100k joint; '
        'DINO uses NavAnywhere v1, 200k pretraining + 10k warmup + 100k joint. Data and schedules are not fully controlled; '
        'this is a system comparison and differences cannot be attributed solely to the DINO objective.',
        '## Hardware and protocol changes',
        'Later historical HuRoN inference used batch128 with roughly 56.5 GiB peak memory, exceeding the current L20 capacity. '
        'Preserving that protocol requires sufficient device memory. Reducing the batch changes stochastic VAE RNG consumption and requires a separate protocol. '
        'The authorized L20 runs use independent batch32 HuRoN evaluation for both models; Table 2 retains candidate microbatch80. '
        'Differences from matching archives are reported separately. Authorization does not establish exact reproduction, and archive values cannot replace fresh DINO measurements.'
        + (' User decisions are recorded separately in `evaluation_decisions.json`: ' +
           json.dumps(ref['operational_limits']['user_decisions'], ensure_ascii=False)
           if 'user_decisions' in ref['operational_limits'] else ''),
        '## Current execution evidence (separate from archives)',
        f"Found {len(current['archive_rescore'])} archive rescoring files and {len(current['fresh_inference'])} explicitly supplied fresh inference records.",
    ]
    if not current['fresh_inference']:
        sections.append('**Fresh results pending: no complete current results support publishing the DINO comparison yet.**')
    if current['archive_rescore_checks']:
        sections.append(md_table(['Rescoring protocol', 'Dataset', 'Sample counts match', 'ΔLPIPS', 'ΔDreamSim', 'ΔPSNR', 'All three exactly equal'],
                                 [[r['protocol'], LABELS[r['dataset']], 'Yes' if r['count_matches'] else 'No',
                                   *(f"{r['delta_from_archive'][k]:+.12g}" for k in ('lpips_alex', 'dreamsim', 'psnr')),
                                   'Yes' if r['all_three_exactly_equal'] else 'No']
                                  for r in current['archive_rescore_checks']]))
    if current['baseline_table1_archive_pixel_checks']:
        sections.append(md_table(['Fresh OpenNWM pixel comparison', 'Compared', 'Pixel-identical', 'Expected total'],
                                 [[LABELS[d], v['compared'], v['pixel_identical'], 10 if d == 'planetary_rover' else 500]
                                  for d, v in current['baseline_table1_pixel_coverage'].items()]))
        sections.append('Pixel replay complete for all Table 1 main-table samples: ' +
                        ('Yes' if current['baseline_table1_complete_pixel_replay'] else 'No; partial evidence only') + '. ')
    historical_diagnostics = current['historical_table1_rank1_diagnostics']
    if historical_diagnostics:
        sections.append('### Historical Table 1 rank1: logical CUDA index control')
        if historical_diagnostics['controlled_first_batch_result_verified']:
            gpu = historical_diagnostics['selected_physical_gpus'][0]
            sections.append(f'First 32 samples on physical L20 GPU{gpu}, with rank seed=1: '
                            'logical cuda:0 matched 0/32 archive samples; restoring historical logical cuda:1 matched 32/32 pixels and PNG bytes. '
                            'Input observations, actions, VAE inputs/latents and CUDA RNG hashes before/after the VAE were identical; '
                            'the difference appears after VAE encoding and depends on logical device index. The exact compiled-kernel mechanism is unconfirmed and cannot be attributed to the VAE seed.')
        else:
            sections.append('The control records do not yet confirm the expected 0/32→32/32 result; see the evidence below.')
        sections.append('The control covers only the first 32 rank1 images. Formal replay coverage follows the shard checks above; '
                        'the first batch cannot establish full-table replay. This diagnosis is separate from the later Rover A800/L20 difference.')
        relocation = historical_diagnostics['relocation']
        sections.append(f"Earlier cuda:0 attempt relocation: {relocation.get('state', 'Unrecorded')}; "
                        f"archive directory: `{relocation.get('relocated_directory', 'Unrecorded')}`. "
                        'Earlier attempts and diagnostic outputs are excluded from canonical fresh results. Historical metadata retains original paths; '
                        'use the relocation manifest and recorded SHA256 to locate files. This report hashes evidence files, not every file in the relocation manifest.')
        sections.append(md_table(['Historical rank1 diagnostic evidence', 'Path', 'SHA256'],
                                 [[name, historical_diagnostics[key]['path'], historical_diagnostics[key].get('sha256', '—')]
                                  for name, key in (('Control JSON', 'source'), ('Control notes', 'markdown_source'),
                                                    ('Attempt relocation record', 'relocation_source'))]))
    if current['baseline_table1_later_archive_pixel_checks']:
        sections.append(md_table(['Later-protocol OpenNWM pixel comparison', 'horizon', 'Compared', 'Pixel-identical'],
                                 [[LABELS[item['dataset']], horizon, values['compared'], values['pixel_identical']]
                                  for item in current['baseline_table1_later_archive_pixel_checks']
                                  for horizon, values in item['data']['counts_by_horizon'].items()]))
        failed = [LABELS[item['dataset']] for item in current['baseline_table1_later_archive_pixel_checks']
                  if not item['data']['all_pixels_identical']]
        if failed:
            sections.append('**Later-protocol fresh inference does not yet match the archive: ' + ', '.join(failed) +
                            '. Fresh values below support model comparison under the same current configuration; they do not establish reproduction of the later archive.**')
    diagnostics = current['later_rover_diagnostics']
    if 'rover_native_comparison' in diagnostics:
        comparison = diagnostics['rover_native_comparison']['data']
        sections.append(md_table(['Rover native-entry diagnostic target', 'horizon', 'Images', 'Byte-identical'],
                                 [[target, horizon, values['compared'], values['byte_identical']]
                                  for target, row in comparison.items()
                                  for horizon, values in row['by_horizon'].items()]))
        run = diagnostics.get('rover_native_run', {}).get('data', {})
        checkpoint = diagnostics.get('baseline_checkpoint_hash', {}).get('data', {})
        sections.append(f"Native-entry diagnostic exit code: {run.get('exit_code', 'Unrecorded')}; "
                        f"full checkpoint SHA256 matches archive: {checkpoint.get('matches_archive', 'Unrecorded')}. "
                        'The native entry and current wrapper produce identical output, ruling out the wrapper entry or copied split as the cause of this difference.')
    if 'cuda_noise_launch_policy' in diagnostics:
        sections.append('CUDA RNG source shows that large-tensor randn thread layout depends on device properties; '
                        'the observed L20 grid is 552, versus 864 inferred from reference A800 properties. Those A800 properties were not directly observed in the historical run. '
                        'This is a candidate mechanism; its contribution relative to BF16, TF32 and compiled-kernel numerical differences has not been quantified. '
                        'Use the original A800 environment for strict later-archive reproduction; current L20 results are not established as identical.')
    if diagnostics:
        sections.append(md_table(['Diagnostic evidence', 'Path', 'SHA256'],
                                 [[name, item['source']['path'], item['source']['sha256']]
                                  for name, item in diagnostics.items()]))
    sections.append('### Table 2 subset verification')
    sections.extend(table2_pilot_notes(ref))
    sections.append('Checks cover only requested windows and four saved metrics (ATE, RPE, position and angle differences), not all 100 windows per dataset.')
    sections.append(md_table(['Model/dataset', 'Policy', 'Requested windows', 'Checked', 'Execution status', 'Four-metric all_metrics_exact', 'provenance_valid', 'Verification outcome'],
                             [[f"{item['data'].get('model', item.get('model', 'baseline'))}/{item['data'].get('dataset', item.get('dataset', 'recon'))}",
                               item['verification_policy'],
                               item['data'].get('requested_indices', item.get('requested_indices_spec', 'Unrecorded')),
                               item['data'].get('count', 0), item['data']['state'],
                               item['data'].get('all_metrics_exact', 'Unverified'), item['data'].get('provenance_valid', 'Unverified'),
                               item['data'].get('verification_outcome', 'See all_metrics_exact for individual equality')]
                              for item in current['table2_subset_verifications']]))
    sections.append('An L20 check allowing metric differences can complete after provenance, coverage and continuation checks pass; '
                    'archive equality still depends on all_metrics_exact. Strict and L20 verification records are retained separately.')
    for item in current['table2_subset_verifications']:
        if item['data'].get('error'):
            sections.append('Subset verification failed: ' + item['data']['error'])
    sections.extend(fresh_tables(ref))
    for category in ('archive_rescore', 'fresh_inference'):
        if current[category]:
            sections.append(md_table(['Source category', 'Path', 'SHA256'],
                                     [[category, item['source']['path'], item['source'].get('sha256', '—')]
                                      for item in current[category]]))
    sections += [
        'This script records fresh evidence without automatically declaring reproduction. Check checkpoint, split/effective mapping, '
        'horizon, batch/rank/seed, VAE RNG consumption, full sample counts and pixel/metric differences. '
        'Rescoring historical predictions validates scoring, not fresh inference.',
        '## Sources',
        md_table(['Source', 'Absolute path', 'SHA256'],
                 [[name, value['path'], value.get('sha256', '—')]
                  for name, value in ref['sources'].items() if isinstance(value, dict) and 'path' in value]),
        f"Current code HEAD: `{ref['sources']['repo_head']}`. "
        + ('The separate checkpoint_runtime_identity.json recomputed both checkpoint SHA256 hashes; '
           f"baseline matches archive={ref['models']['baseline']['recomputed_sha256_matches_archive']}. "
           if 'checkpoint_runtime_identity' in ref['sources'] else
           'This audit records checkpoint path/size/time and the existing baseline archive SHA256; recomputed hashes have not yet been included. ') +
        'See `reference.json` for full-precision values, metrics/source_audit file hashes, per-window navigation provenance and fresh evidence.',
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
