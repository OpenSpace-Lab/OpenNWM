#!/usr/bin/env python3
"""CPU-only observer of existing OpenNWM queues; never starts or retries experiments.

Run once, or use --watch --interval 300. Only meaningful state changes write
audit/RUN_STATE.md and queue_snapshot.json. New completion/failure events refresh
the CPU report. Two observations confirm stale running PIDs before treating an
unattended queue as terminal. No signal is ever sent to an experiment process.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo


REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/file_system/nas/algorithm/dujun.nie/nwm/opennwm_reproduction/dino_main_tables_20261007')
QUEUE_NAMES = ('table1_l20_huron', 'table2_l20_full')
MODELS = ('baseline', 'dino')
NAV_DATASETS = ('recon', 'scand', 'unitree_go2')
MAIN_DATASETS = ('recon', 'scand', 'huron', 'tartan_drive', 'go_stanford', 'tum_rgbd', 'planetary_rover', 'unitree_go2')
METRICS = ('ate', 'rpe_trans', 'pos_diff_norm', 'yaw_diff_norm')
EVAL_NAME = 'CEM_N80_K5_RS1_rep3_OPT1_COST-lpips-RECON-True'
TERMINAL = {'completed', 'failed', 'interrupted', 'error'}


def now():
    return datetime.now(ZoneInfo('Asia/Taipei')).isoformat()


def read_json(path, errors):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        errors.append({'path': str(path), 'error': str(exc)})
        return None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def meaningful(value):
    if isinstance(value, dict):
        return {key: meaningful(item) for key, item in value.items() if key not in ('observer', 'os_state')}
    if isinstance(value, list):
        return [meaningful(item) for item in value]
    return value


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def process_info(pid, expected_argv=None, required_tokens=()):
    result = {'pid': pid, 'alive': False, 'identity_matches': False}
    if type(pid) is not int or pid <= 0:
        return result
    proc = Path('/proc') / str(pid)
    try:
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        argv = [part.decode(errors='replace') for part in (proc / 'cmdline').read_bytes().split(b'\0') if part]
        result.update(os_state=stat[0], start_ticks=int(stat[19]), argv=argv,
                      cwd=os.readlink(proc / 'cwd'))
        for field, suffix in (('stdout', 'fd/1'), ('stderr', 'fd/2')):
            try:
                result[field] = os.readlink(proc / suffix)
            except OSError:
                pass
        result['alive'] = stat[0] not in ('Z', 'X') and bool(argv)
        result['identity_matches'] = result['alive'] and (
            argv == expected_argv if expected_argv is not None else all(token in argv for token in required_tokens))
    except FileNotFoundError:
        result['unavailable_reason'] = 'process_absent'
    except OSError as exc:
        result['unavailable_reason'] = str(exc)
    return result


def metadata_summary(path, errors):
    value = read_json(path, errors)
    if value is None:
        return {'path': str(path), 'state': 'missing_or_unreadable'}
    return {'path': str(path), **{key: value.get(key) for key in ('state', 'exit_code', 'pid', 'log', 'error')}}


def queue_snapshot(root, name, previous, errors, *, recheck=True):
    path = root / 'queues' / (name + '.state.json')
    state = read_json(path, errors)
    if state is None:
        return {'name': name, 'state_path': str(path), 'recorded_status': 'missing_or_unreadable',
                'effective_status': 'unknown', 'terminal': False, 'jobs': [], 'issues': ['Queue state unavailable']}
    jobs_file = state.get('jobs_file', str(path.with_name(name + '.json')))
    process = process_info(state.get('queue_pid'), required_tokens=(str(REPO / 'tools/reproduce_paper_queue.py'), '--jobs', jobs_file))
    jobs, issues = [], []
    for item in state.get('jobs', []):
        spec = item.get('specification', {})
        status = item.get('status', 'unknown')
        job = {key: item.get(key) for key in ('id', 'status', 'pid', 'exit_code', 'error')}
        job.update(argv=spec.get('argv', []), cwd=spec.get('cwd'), log=spec.get('log'),
                   gpu=spec.get('gpu'), environment_overrides=spec.get('env', {}))
        if status not in TERMINAL and status != 'pending':
            job['prerequisites'] = [metadata_summary(p, errors) for p in spec.get('prerequisite_metadata', [])]
            job['completion_metadata'] = [metadata_summary(p, errors) for p in spec.get('completion_metadata', [])]
        if status == 'running':
            job['process'] = process_info(item.get('pid'), expected_argv=spec.get('argv'))
            if not job['process']['identity_matches']:
                issues.append(f"Job {item['id']} records running but PID is absent or has different argv")
        jobs.append(job)
    recorded = state.get('status', 'unknown')
    if recorded not in TERMINAL and not process['identity_matches']:
        issues.append('Queue records nonterminal state but PID is absent or has different argv')
    # A child can exit immediately before the queue writes its completion state.
    if issues and recheck:
        time.sleep(0.2)
        return queue_snapshot(root, name, previous, errors, recheck=False)
    anomaly_key = digest({'pid': state.get('queue_pid'), 'issues': issues}) if issues else None
    confirmed = bool(issues and previous.get('anomaly_key') == anomaly_key)
    live_child = any(j.get('process', {}).get('identity_matches') for j in jobs)
    if recorded in TERMINAL:
        effective, terminal = recorded, True
    elif not process['identity_matches']:
        effective = 'orphan_child_running' if live_child else 'stale_queue_confirmed' if confirmed else 'stale_queue_suspected'
        terminal = confirmed and not live_child
    elif issues:
        effective, terminal = ('job_process_mismatch_confirmed' if confirmed else 'job_process_mismatch_suspected'), False
    else:
        active = next((j['status'] for j in jobs if j['status'] not in TERMINAL and j['status'] != 'pending'), recorded)
        effective, terminal = active, False
    observed = process if process['identity_matches'] else previous.get('last_verified_process')
    launch_path = root / 'queues' / (name + '.launch.json')
    launch = read_json(launch_path, errors) or {}
    launch_valid = (launch.get('pid') == state.get('queue_pid') and isinstance(launch.get('command'), list)
                    and str(REPO / 'tools/reproduce_paper_queue.py') in launch['command']
                    and '--jobs' in launch['command'] and jobs_file in launch['command'])
    invocation = ({'argv': observed['argv'], 'cwd': observed['cwd'], 'log': observed.get('stdout'),
                   'evidence': 'current_or_last_verified_proc'} if observed else
                  {'argv': launch['command'], 'cwd': launch['cwd'], 'log': launch['log'],
                   'evidence': str(launch_path)} if launch_valid else {})
    return {'name': name, 'state_path': str(path), 'jobs_file': jobs_file,
            'recorded_status': recorded, 'effective_status': effective, 'terminal': terminal,
            'queue_pid': state.get('queue_pid'), 'process': process, 'last_verified_process': observed,
            'invocation': invocation,
            'queue_log': invocation.get('log', str(root / 'queues' / (name + '.queue.log'))),
            'exit_code': state.get('exit_code'), 'error': state.get('error'), 'failed_job': state.get('failed_job'),
            'jobs': jobs, 'issues': issues, 'anomaly_key': anomaly_key, 'anomaly_confirmed': confirmed}


def score_complete(root, protocol, model, dataset, errors):
    directory = root / 'fresh_scores' / protocol / model
    manifest = read_json(directory / 'scores_manifest.json', errors) or {}
    record = next((v for v in manifest.get('results', []) if v.get('dataset') == dataset), {})
    path = directory / (dataset + '.json')
    score = read_json(path, errors) or {}
    status = read_json(directory / 'status' / (dataset + '.json'), errors) or {}
    endpoint = '16obs' if dataset == 'planetary_rover' and protocol != 'paper_historical' else '4s'
    values = score.get('metrics', {}).get(endpoint, {})
    expected = 10 if dataset == 'planetary_rover' else 329 if dataset == 'huron' and protocol != 'paper_historical' else 500
    return (record.get('complete') is True and record.get('origin') == 'fresh_inference'
            and record.get('model') == model and record.get('protocol') == protocol
            and record.get('path') == str(path) and values.get('sample_count') == expected
            and status.get('state') == 'completed' and status.get('exit_code') == 0
            and all(finite(values.get(key)) for key in ('lpips_alex', 'dreamsim', 'psnr')))


def table1_snapshot(root, errors):
    rows = []
    for view in ('paper_historical', 'l20_batch32'):
        for model in MODELS:
            done = []
            for dataset in MAIN_DATASETS:
                protocol = ('l20_batch32' if dataset == 'huron' else 'corrected' if dataset == 'planetary_rover' else 'paper_historical') if view == 'l20_batch32' else view
                if score_complete(root, protocol, model, dataset, errors):
                    done.append(dataset)
            rows.append({'view': view, 'model': model, 'complete_datasets': done, 'expected_datasets': 8,
                         'complete': len(done) == 8, 'pending_datasets': [d for d in MAIN_DATASETS if d not in done]})
    proof_path = root / 'audit/table1_final_metrics_validation.json'
    proof = read_json(proof_path, errors) or {}
    verified = proof.get('all_checks_passed') is True and proof.get('historical_baseline_all_seven_raw_archive_values_exact') is True
    for model in MODELS:
        for dataset in MAIN_DATASETS:
            path = root / 'fresh_scores/paper_historical' / model / (dataset + '.json')
            expected_hash = proof.get('source_sha256', {}).get(str(path))
            verified = verified and path.is_file() and expected_hash == hashlib.sha256(path.read_bytes()).hexdigest()
    return {'progress': rows, 'historical_baseline_seven_metrics_exact_archive_verified': verified,
            'validation_path': str(proof_path)}


def table2_snapshot(root, errors):
    groups = []
    for model in MODELS:
        for dataset in NAV_DATASETS:
            directory = root / 'table2' / model / dataset / 'mb80'
            samples = directory / 'outputs' / dataset / EVAL_NAME / 'sample_metrics'
            valid, invalid = {}, []
            for path in sorted(samples.glob('*.json')):
                value = read_json(path, errors)
                if (len(path.stem) != 6 or not path.stem.isdigit() or not 0 <= int(path.stem) < 100
                        or not isinstance(value, dict) or set(value) != set(METRICS)
                        or not all(finite(value[key]) for key in METRICS)):
                    invalid.append(str(path))
                else:
                    valid[int(path.stem)] = value
            aggregate_path = directory / 'aggregate.json'
            aggregate = read_json(aggregate_path, errors)
            complete = bool(aggregate and len(valid) == 100 and not invalid and aggregate.get('complete') is True
                            and aggregate.get('count') == 100 and aggregate.get('missing_indices') == []
                            and aggregate.get('model') == model and aggregate.get('dataset') == dataset
                            and aggregate.get('microbatch_size') == 80 and aggregate.get('sample_metrics') == str(samples)
                            and all(finite(aggregate.get('means', {}).get(key)) and math.isclose(
                                aggregate['means'][key], math.fsum(v[key] for v in valid.values()) / 100,
                                rel_tol=0, abs_tol=1e-12) for key in METRICS))
            groups.append({'model': model, 'dataset': dataset, 'sample_directory': str(samples),
                           'complete_saved_windows': len(valid), 'expected_windows': 100, 'indices': sorted(valid),
                           'invalid_sample_files': invalid, 'aggregate_path': str(aggregate_path),
                           'full_100_window_aggregate_complete': complete,
                           'aggregate_state': 'complete' if complete else 'not_complete_or_inconsistent' if aggregate else 'pending',
                           'archive_equivalence': aggregate.get('historical_equivalence') if complete else 'pending_full_aggregate'})
    return groups


def snapshot(root, previous, watch):
    errors = []
    old = {q['name']: q for q in previous.get('queues', [])}
    queues = [queue_snapshot(root, name, old.get(name, {}), errors) for name in QUEUE_NAMES]
    result = {'schema_version': 1, 'observer': {'pid': os.getpid(), 'watch': watch,
              'argv': [sys.executable, *sys.argv], 'cwd': str(Path.cwd())},
              'queues': queues, 'table1': table1_snapshot(root, errors),
              'table2': table2_snapshot(root, errors), 'read_errors': errors}
    result['all_queues_terminal'] = all(q['terminal'] for q in queues)
    result['has_failure'] = bool(errors or any(q['recorded_status'] in {'failed', 'error', 'interrupted', 'missing_or_unreadable'}
                                 or (q['recorded_status'] == 'completed' and q.get('exit_code') != 0)
                                 or q.get('anomaly_confirmed') for q in queues))
    result['has_unconfirmed_process_anomaly'] = any(q.get('issues') and not q.get('anomaly_confirmed') for q in queues)
    events = []
    for q in queues:
        if q['recorded_status'] in TERMINAL or q.get('anomaly_confirmed'):
            events.append(digest([q['name'], q['recorded_status'], q['exit_code'], q.get('anomaly_key')]))
        for j in q['jobs']:
            if j['status'] in TERMINAL:
                events.append(digest([q['name'], j['id'], j['status'], j['exit_code'], j.get('error')]))
    result['terminal_events'] = sorted(events)
    return result


def markdown(data, root):
    lines = ['# OpenNWM evaluation status', f"Snapshot updated on request: {data['updated_at']} (Asia/Taipei); unchanged snapshots retain their timestamp.",
             'This page observes existing processes and outputs without starting, stopping or retrying experiments. A complete window has four saved metrics; window counts do not establish a full 100-window aggregate or exact reproduction.',
             '## Table 1', '| Protocol | Model | Fully scored datasets | Status |', '| --- | --- | --- | --- |']
    for row in data['table1']['progress']:
        lines.append(f"| {row['view']} | {row['model']} | {len(row['complete_datasets'])}/8 | {'Complete' if row['complete'] else 'Pending: ' + ', '.join(row['pending_datasets'])} |")
    lines.append('Historical OpenNWM seven-metric exact archive equality: ' + ('Passed' if data['table1']['historical_baseline_seven_metrics_exact_archive_verified'] else 'Verification pending') + '. ')
    lines.extend(['## Table 2', '| Model | Dataset | Complete saved windows | Full 100-window aggregate |', '| --- | --- | --- | --- |'])
    for row in data['table2']:
        lines.append(f"| {row['model']} | {row['dataset']} | {row['complete_saved_windows']}/100 | {'Complete' if row['full_100_window_aggregate_complete'] else 'Pending'} |")
        if row['invalid_sample_files']:
            lines.append('Incomplete or invalid JSON: ' + ', '.join(row['invalid_sample_files']))
    lines.extend(['The pilot left 598 windows, roughly 121–130 GPU hours (5–5.5 days on one L20) based on two RECON baseline windows. '
                  'This is not a live completion estimate. It excludes startup, compilation and HuRoN prerequisites; two windows do not represent all datasets or DINO timing.',
                  'The RECON archive combines earlier L20 windows 0–15 and later A800 windows 16–99. The strict pilot equality check failed (1/2 windows differed), which does not establish hardware causality.',
                  '## Observed queues'])
    for q in data['queues']:
        p = q.get('process', {})
        observed = q.get('invocation') or {}
        lines.extend([f"### {q['name']}",
                      f"Status: `{q['effective_status']}`; recorded status `{q['recorded_status']}`; PID {q.get('queue_pid')}; "
                      f"process identity: {'active with matching argv' if p.get('identity_matches') else 'activity unconfirmed'}; exit code {q.get('exit_code')}. ",
                      f"Working directory: `{observed.get('cwd', 'Not captured')}`; queue log: [view]({q.get('queue_log', '')}). ",
                      f"Exact queue argv (evidence: {observed.get('evidence', 'Not captured')}): \n\n```sh\n" + (shlex.join(observed.get('argv', [])) or 'Not captured; command not inferred') + '\n```'])
        if q.get('error') or q.get('issues'):
            lines.append('Error: ' + str(q.get('error') or '') + '; '.join(q.get('issues', [])))
        active = [j for j in q['jobs'] if j['status'] not in ('completed', 'pending')]
        for j in active:
            lines.extend([f"Current job `{j['id']}`: {j['status']}; PID {j['pid']}; exit code {j['exit_code']}. ",
                          f"cwd: `{j['cwd']}`; log: [view]({j['log']}). ",
                          'Exact job argv (queue specification): \n\n```sh\n' + shlex.join(j['argv']) + '\n```'])
            if j.get('prerequisites'):
                lines.append('Prerequisites: \n\n' + '\n'.join(f"- `{m['path']}`: {m['state']}, exit={m.get('exit_code')}" for m in j['prerequisites']))
            if j.get('error'):
                lines.append('Job error: ' + str(j['error']))
        lines.append(f"Queue jobs completed: {sum(j['status'] == 'completed' for j in q['jobs'])}/{len(q['jobs'])}. ")
    if data['read_errors']:
        lines.append('Read errors (not success): \n\n' + '\n'.join(f"- {e['path']}: {e['error']}" for e in data['read_errors']))
    if data.get('report_refresh'):
        lines.append('Latest CPU report refresh: exit code ' + str(data['report_refresh']['exit_code']) + '; [log](' + data['report_refresh']['log'] + '). ')
    lines.append(f'[Main-table results]({root / "audit/RESULTS.md"}) · [Full status JSON]({root / "audit/queue_snapshot.json"})')
    blocks = []
    for line in lines:
        if line.startswith('|') and blocks and blocks[-1].startswith('|'):
            blocks[-1] += '\n' + line
        else:
            blocks.append(line)
    return '\n\n'.join(blocks) + '\n'


def atomic_write(path, contents):
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as out:
        out.write(contents)
        out.flush()
        os.fsync(out.fileno())
        temporary = Path(out.name)
    temporary.replace(path)


def refresh_report(root):
    command = [sys.executable, str(REPO / 'tools/reproduce_paper_report.py'), '--root', str(root)]
    log = root / 'audit/status_report_refresh.log'
    with log.open('a') as out:
        out.write(now() + ' ' + shlex.join(command) + '\n')
        out.flush()
        process = subprocess.run(command, cwd=REPO, stdout=out, stderr=subprocess.STDOUT)
    return {'time': now(), 'argv': command, 'cwd': str(REPO), 'log': str(log), 'exit_code': process.returncode}


def update(root, watch):
    audit = root / 'audit'
    audit.mkdir(parents=True, exist_ok=True)
    with (audit / '.status.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = read_json(audit / 'queue_snapshot.json', []) or {}
        current = snapshot(root, previous, watch)
        # Acquisition time, file mtimes, queue heartbeats, and log growth are
        # deliberately absent. Only meaningful status/progress changes count.
        fingerprint = digest(meaningful(current))
        if fingerprint == previous.get('fingerprint'):
            return previous, False
        old_events = set(previous.get('refreshed_terminal_events', []))
        new_events = set(current['terminal_events']) - old_events
        refresh = refresh_report(root) if new_events else previous.get('report_refresh')
        current.update(fingerprint=fingerprint, updated_at=now(),
                       refreshed_terminal_events=sorted(old_events | set(current['terminal_events'])),
                       report_refresh=refresh)
        if refresh and refresh['exit_code'] != 0:
            current['has_failure'] = True
        atomic_write(audit / 'queue_snapshot.json', json.dumps(current, ensure_ascii=False, indent=2) + '\n')
        atomic_write(audit / 'RUN_STATE.md', markdown(current, root))
        return current, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=300)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    root = args.root.resolve()
    (root / 'audit').mkdir(parents=True, exist_ok=True)
    with (root / 'audit/.status_watch.lock').open('a+') as observer_lock:
        if args.watch:
            try:
                fcntl.flock(observer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.exit(2, 'Another status observer already holds the watch lock.\n')
        while True:
            data, changed = update(root, args.watch)
            if changed:
                print(json.dumps({'time': data['updated_at'], 'event': 'status_changed',
                                  'queues': {q['name']: q['effective_status'] for q in data['queues']},
                                  'table2_windows': sum(g['complete_saved_windows'] for g in data['table2']),
                                  'has_failure': data['has_failure']}, ensure_ascii=False), flush=True)
            if not args.watch or data['all_queues_terminal']:
                return 1 if data['has_failure'] else 2 if data['has_unconfirmed_process_anomaly'] else 0
            deadline = time.monotonic() + args.interval
            while time.monotonic() < deadline:
                time.sleep(min(60, max(0, deadline - time.monotonic())))


if __name__ == '__main__':
    raise SystemExit(main())
