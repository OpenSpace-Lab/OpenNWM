#!/usr/bin/env python3
"""Prepare compact Go2 web films and publish them to one or more local websites.

This does not commit, push, or deploy. NAS masters and standalone recordings /
model rollouts are read-only. Run with an explicit Python runtime; see
website/content/GO2_MANIFEST.md for the publication contract and command.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import fcntl
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import struct
import subprocess
import tempfile


FFMPEG = Path('/file_system/vepfs/algorithm/dujun.nie/miniconda3/envs/fastwam/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
FFPROBE = Path('/usr/bin/ffprobe')
STANDALONES = {'gt_ego.mp4', 'gt_third_person.mp4', 'opennwm.mp4', 'nwm.mp4'}
COPY_FILES = ('poster.webp', 'metrics.json', 'trajectories.csv')
MAX_VIDEO_BYTES = 95 * 1024**2
ENCODER_VERSION = 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    return {'bytes': path.stat().st_size, 'sha256': sha256(path)}


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def checked_destination(website, relative):
    """Never follow a destination symlink while replacing or deleting files."""
    relative = PurePosixPath(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, f'Unsafe destination: {relative}')
    path = website
    for part in relative.parts:
        path = path / part
        require(not path.is_symlink(), f'Destination contains a symlink: {path}')
    return path


def atomic_copy(source, destination, expected):
    if destination.is_file() and identity(destination) == expected:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.' + destination.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
        shutil.copyfile(source, temporary)
        require(identity(temporary) == expected, f'Staged copy differs: {source}')
        os.replace(temporary, destination)
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def probe(path, ffprobe):
    data = json.loads(subprocess.check_output([str(ffprobe), '-v', 'error', '-show_streams',
        '-show_format', '-of', 'json', str(path)]))
    streams = [s for s in data['streams'] if s['codec_type'] == 'video']
    require(len(streams) == 1, f'Expected one video stream: {path}')
    stream = streams[0]
    return {'width': stream['width'], 'height': stream['height'],
            'codec': stream['codec_name'], 'pixel_format': stream.get('pix_fmt'),
            'fps': stream['avg_frame_rate'], 'frames': int(stream['nb_frames']),
            'duration_s': float(stream.get('duration', data['format']['duration'])),
            'stream_count': len(data['streams'])}


def faststart(path):
    """Require the MP4 movie index before media data (HTTP progressive playback)."""
    with path.open('rb') as stream:
        while header := stream.read(8):
            require(len(header) == 8, f'Truncated MP4 atom: {path}')
            size, kind = struct.unpack('>I4s', header)
            header_size = 8
            if size == 1:
                extended = stream.read(8)
                require(len(extended) == 8, f'Truncated MP4 atom: {path}')
                size, header_size = struct.unpack('>Q', extended)[0], 16
            if kind == b'moov':
                return True
            if kind == b'mdat' or size == 0:
                return False
            require(size >= header_size, f'Invalid MP4 atom: {path}')
            stream.seek(size - header_size, 1)
    return False


def check_media_info(info, original, duration, path):
    """Check timing/format invariants for either fresh probes or cached receipts."""
    require((info['width'], info['height']) == (1920, 1080), f'Web film must remain 1080p: {path}')
    require(info['codec'] == 'h264' and info['pixel_format'] == 'yuv420p', f'Incompatible web video: {path}')
    require(info['stream_count'] == 1, f'Unexpected audio / data stream: {path}')
    require(info['frames'] == original['frames'] and Fraction(info['fps']) == Fraction(original['fps']),
            f'Frame timing changed: {path}')
    require(abs(info['duration_s'] - duration) < .08 and
            abs(info['duration_s'] - original['duration_s']) < .001, f'Duration changed: {path}')
    require(path.stat().st_size < MAX_VIDEO_BYTES, f'Web film exceeds 95 MiB: {path}')
    require(faststart(path), f'MP4 is missing faststart: {path}')


def validate_video(path, original, duration, ffmpeg, ffprobe):
    info = probe(path, ffprobe)
    check_media_info(info, original, duration, path)
    subprocess.run([str(ffmpeg), '-v', 'error', '-xerror', '-err_detect', 'explode', '-threads', '2',
        '-i', str(path), '-map', '0:v:0', '-an', '-f', 'null', '-'], check=True)
    return info


def encode_clip(clip, source, cache, settings, args):
    clip_id = clip['id']
    master = source / clip_id / 'comparison.mp4'
    master_identity = identity(master)
    directory = cache / clip_id
    directory.mkdir(parents=True, exist_ok=True)
    video, receipt = directory / 'comparison.mp4', directory / 'validation.json'
    original = probe(master, args.ffprobe)
    require((original['width'], original['height']) == (1920, 1080), f'Master is not 1080p: {master}')
    require(abs(original['duration_s'] - clip['duration_s']) < .08, f'Master duration mismatch: {master}')
    if receipt.is_file() and video.is_file():
        old = read(receipt)
        if (old.get('settings') == settings and
                old.get('master') == master_identity and old.get('output') == identity(video)):
            certified = old.get('passed') is True and old.get('fully_decoded') is True
            if certified:
                try:
                    check_media_info(old['media'], original, clip['duration_s'], video)
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    certified = False
            if not certified:
                # A matching hash proves identity, not that the prior validation completed.
                info = validate_video(video, original, clip['duration_s'], args.ffmpeg, args.ffprobe)
                require(identity(master) == master_identity, f'Master changed during validation: {master}')
                old = {**old, 'passed': True, 'fully_decoded': True, 'media': info}
                atomic_bytes(receipt, json_bytes(old))
                print('CACHE_REVALIDATED', clip_id, flush=True)
            print('CACHE_HIT', clip_id, flush=True)
            return {**old, 'cache_hit': True}
    temporary = directory / 'comparison.encoding.mp4'
    log_path = directory / 'encode.log'
    command = [str(args.ffmpeg), '-hide_banner', '-loglevel', 'warning', '-y', '-threads', '2',
        '-i', str(master), '-map', '0:v:0', '-map_metadata', '-1', '-an', '-sn', '-dn',
        '-c:v', 'libx264', '-threads', '4', '-preset', args.preset, '-crf', str(args.crf),
        '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough', '-movflags', '+faststart', str(temporary)]
    try:
        with log_path.open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=source)
            print('ENCODE', json.dumps({'clip': clip_id, 'cwd': str(source), 'pid': process.pid,
                'command': shlex.join(command), 'log': str(log_path)}), flush=True)
            status = process.wait()
        print('ENCODE_EXIT', clip_id, status, flush=True)
        require(status == 0, f'Encoder failed ({status}): {log_path}\n{log_path.read_text()[-3000:]}')
        info = validate_video(temporary, original, clip['duration_s'], args.ffmpeg, args.ffprobe)
        require(identity(master) == master_identity, f'Master changed during encoding: {master}')
        output = identity(temporary)
        os.replace(temporary, video)
        result = {'passed': True, 'clip': clip_id, 'settings': settings, 'master': master_identity,
                  'output': output, 'media': info, 'fully_decoded': True}
        atomic_bytes(receipt, json_bytes(result))
        return {**result, 'cache_hit': False}
    finally:
        temporary.unlink(missing_ok=True)


def validate_clip_mean(clip, report):
    """Require a constant clip mean, bound to every complete exported plan."""
    label = clip['id']
    plans = report.get('plans', [])
    expected = {'method': 'arithmetic_mean_of_complete_local_plans',
                'plan_count': len(plans), 'display': 'constant_for_entire_clip',
                'metrics': ['ate_m', 'rpe_m'], 'units': 'meters'}
    require(bool(plans) and report.get('metric_aggregation') == expected and
            clip.get('metric_aggregation') == expected, f'{label}: clip metric aggregation contract')
    require(report.get('id') == label and report.get('episode') == clip.get('episode'),
            f'{label}: clip report identity differs')
    ids = [row['id'] for row in plans]
    require(len(ids) == len(set(ids)) and ids == clip.get('evaluated_goal_ids'),
            f'{label}: clip plan coverage/order differs')
    start, duration, interval = clip['start_time_s'], clip['duration_s'], clip['goal_interval_s']
    require(report['start_s'] == start and report['end_s'] == start + duration and
            report['goal_interval_s'] == interval and interval > 0, f'{label}: clip report timing differs')
    require(plans[0]['start_time_s'] == start and plans[-1]['end_time_s'] == start + duration and
            all(row['end_time_s'] - row['start_time_s'] == interval for row in plans) and
            all(a['end_time_s'] == b['start_time_s'] for a, b in zip(plans, plans[1:])),
            f'{label}: incomplete or non-contiguous local plans')
    require(clip['metrics'] == report['metrics'] and set(report['metrics']) == {'opennwm', 'nwm'},
            f'{label}: clip summary differs')
    for model in ('opennwm', 'nwm'):
        require(set(report['metrics'][model]) == {'ate_m', 'rpe_m'}, f'{label}: unexpected summary metrics')
        for metric in ('ate_m', 'rpe_m'):
            values = [row['models'][model]['metrics'][metric] for row in plans]
            actual = report['metrics'][model][metric]
            require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and
                        math.isfinite(v) and v >= 0 for v in [actual, *values]),
                    f'{label}: invalid metric values')
            require(math.isclose(actual, math.fsum(values) / len(values), rel_tol=1e-10, abs_tol=1e-12),
                    f'{label}: summary is not arithmetic mean of complete local plans')


def load_manifest(source):
    manifest = read(source / 'website_manifest.json')
    require(manifest.get('version') in (2, 3) and manifest.get('status') == 'ready',
            'A ready v2 or v3 manifest is required')
    require(manifest.get('clips'), 'The gallery is empty')
    ids = [clip['id'] for clip in manifest['clips']]
    require(len(ids) == len(set(ids)), 'Duplicate clip IDs')
    public = copy.deepcopy(manifest)
    public.pop('full', None)
    public['report'] = 'assets/go2/report.json'
    for clip in public['clips']:
        clip_id = clip['id']
        require(re.fullmatch(r'[A-Za-z0-9_-]+', clip_id), f'Unsafe clip ID: {clip_id}')
        duration = clip['duration_s']
        require(isinstance(duration, (int, float)) and math.isfinite(duration) and duration > 0,
                f'Invalid duration: {clip_id}')
        prefix = f'assets/go2/{clip_id}'
        require(clip['video'] == f'{prefix}/comparison.mp4' and clip['poster'] == f'{prefix}/poster.webp',
                f'Unexpected master path: {clip_id}')
        for name in ('comparison.mp4', *COPY_FILES):
            require((source / clip_id / name).is_file(), f'Missing master asset: {clip_id}/{name}')
        clip['downloads'] = {'metrics': f'{prefix}/metrics.json', 'trajectories': f'{prefix}/trajectories.csv'}
    require((source / 'report.json').is_file(), 'Missing gallery report')
    report = read(source / 'report.json')
    if manifest['version'] >= 3:
        require(report.get('schema_version') == 3, 'Gallery report schema must match v3 manifest')
        reports = {row['id']: row for row in report['clips']}
        require(len(reports) == len(report['clips']) and set(ids) <= set(reports),
                'Gallery report clip coverage or duplicate mismatch')
        for clip in public['clips']:
            individual = read(source / clip['id'] / 'metrics.json')
            require(individual == reports[clip['id']], f'{clip["id"]}: individual and combined reports differ')
            validate_clip_mean(clip, individual)
        if 'display_clip_count' in manifest.get('selection', {}):
            require(manifest['selection']['display_clip_count'] == len(ids), 'Display clip count differs')
    # Reports are already scrubbed by the renderer. Do not rewrite metrics or protocol.
    for value in (public, report, *[read(source / c / 'metrics.json') for c in ids]):
        require('/file_system/' not in json.dumps(value), 'Private NAS path in public metadata')
    return public


def deletion_plan(website, source, websites=(), retained_sources=()):
    """Only remove known standalone exports after matching an intact NAS original."""
    result = []
    asset_root = checked_destination(website, 'assets/go2')
    for path in sorted(asset_root.rglob('*.mp4')):
        if path.name not in STANDALONES:
            continue
        relative = path.relative_to(asset_root)
        require(len(relative.parts) == 2, f'Unexpected standalone location: {path}')
        checked_destination(website, path.relative_to(website).as_posix())
        recorded = identity(path)
        original = None
        for candidate_root in (source, *retained_sources):
            candidate = candidate_root / relative
            if not candidate.is_file():
                continue
            resolved = candidate.resolve(strict=True)
            require(not any(w == resolved or w in resolved.parents for w in websites),
                    f'NAS original points into a website: {candidate}')
            if identity(candidate) == recorded:
                original = candidate
                break
        require(original is not None, f'No byte-identical NAS original; refusing removal: {path}')
        result.append({'path': path.relative_to(website).as_posix(), 'original': str(original), **recorded})
    return result


def deployed_files(website):
    """Match the current Pages workflow; exclude .git and local development files."""
    files = {p for pattern in ('*.html', '*.css', '*.js', '.nojekyll') for p in website.glob(pattern) if p.is_file()}
    for directory in ('assets', 'content'):
        files.update(p for p in (website / directory).rglob('*') if p.is_file())
    return {p.relative_to(website).as_posix(): p.stat().st_size for p in files}


def size_plan(website, desired, removals, manifest_bytes, budget):
    before = deployed_files(website)
    after = dict(before)
    for row in removals:
        after.pop(row['path'], None)
    for relative, entry in desired.items():
        after[relative] = entry['bytes']
    after['content/go2.json'] = len(manifest_bytes)
    # Include the two redirect pages generated outside the copied site folder.
    projected = sum(after.values()) + 4096
    require(projected <= budget,
            f'Site budget exceeded at {website}: {projected:,} > {budget:,} bytes; no website files changed')
    return {'before_bytes': sum(before.values()), 'projected_bytes': projected,
            'budget_bytes': budget, 'removed_bytes': sum(r['bytes'] for r in removals)}


def publish(args):
    source = args.source.resolve(strict=True)
    websites = [p.resolve(strict=True) for p in args.website]
    retained_sources = [p.resolve(strict=True) for p in args.retained_source]
    require(len(set(websites)) == len(websites), 'Repeated website destination')
    cache = (args.cache or source.parent / 'web_gallery_cache' / source.name /
             f'crf{args.crf}_{args.preset}').resolve()
    validation_path = (args.validation or source.parent / 'webpublish_validation.json').resolve()
    for path in [cache, validation_path, *websites]:
        require(path != source and source not in path.parents and path not in source.parents,
                f'Output must be separate from NAS masters: {path}')
    for website in websites:
        require((website / 'index.html').is_file(), f'Not a website root: {website}')
        require(website not in cache.parents and cache not in website.parents and website != cache,
                'Cache must be outside website roots')
        require(website != validation_path and website not in validation_path.parents,
                'Private validation report must be outside website roots')
        checked_destination(website, 'content/go2.json')
    manifest = load_manifest(source)
    source_manifest = identity(source / 'website_manifest.json')
    manifest_data = json_bytes(manifest)
    for clip in manifest['clips']:
        for name in ('comparison.mp4', *COPY_FILES):
            resolved = (source / clip['id'] / name).resolve(strict=True)
            require(not any(w == resolved or w in resolved.parents for w in websites),
                    f'Master asset points into a website: {resolved}')
    # Hash-check removals in all destinations before any encoding or site changes.
    removals = {str(w): deletion_plan(w, source, websites, retained_sources) for w in websites}
    master_bytes = sum((source / c['id'] / 'comparison.mp4').stat().st_size for c in manifest['clips'])
    cache.parent.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(cache.parent).free > master_bytes + 512 * 1024**2,
            'Insufficient free space for the separate web encoding cache')
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / '.publish.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings = {'version': ENCODER_VERSION, 'crf': args.crf, 'preset': args.preset,
                    'ffmpeg': str(args.ffmpeg), 'ffmpeg_sha256': sha256(args.ffmpeg),
                    'resolution': [1920, 1080], 'frame_timing': 'passthrough', 'audio': False}
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            encoded = list(pool.map(lambda c: encode_clip(c, source, cache, settings, args), manifest['clips']))
        desired = {}
        for clip, result in zip(manifest['clips'], encoded):
            clip_id = clip['id']
            desired[f'assets/go2/{clip_id}/comparison.mp4'] = {
                'source': cache / clip_id / 'comparison.mp4', **result['output']}
            for name in COPY_FILES:
                path = source / clip_id / name
                desired[f'assets/go2/{clip_id}/{name}'] = {'source': path, **identity(path)}
        desired['assets/go2/report.json'] = {'source': source / 'report.json', **identity(source / 'report.json')}
        plans = []
        # Finish all budget/path checks before touching either website.
        for website in websites:
            for relative in desired:
                checked_destination(website, relative)
            sizes = size_plan(website, desired, removals[str(website)], manifest_data, args.max_site_mb * 1_000_000)
            require(shutil.disk_usage(website).free > sum(d['bytes'] for d in desired.values()) + 128 * 1024**2,
                    f'Insufficient space for atomic website copies: {website}')
            plans.append({'website': str(website), 'size': sizes, 'removals': removals[str(website)]})
        # Recheck all originals after a potentially long encode, before destructive work.
        require(identity(source / 'website_manifest.json') == source_manifest,
                'Source manifest changed during encoding; refusing publication')
        for plan in plans:
            website = Path(plan['website'])
            for row in plan['removals']:
                expected = {k: row[k] for k in ('bytes', 'sha256')}
                require(identity(Path(row['original'])) == expected and
                        identity(website / row['path']) == expected,
                        f'Standalone changed after preflight: {row["path"]}')
        report = {'version': 1, 'passed': False, 'source': str(source), 'cache': str(cache),
                  'source_manifest': source_manifest,
                  'retained_sources': list(map(str, retained_sources)), 'published': False,
                  'started_at': datetime.now(timezone.utc).isoformat(), 'clip_count': len(encoded),
                  'settings': settings, 'clips': encoded, 'websites': plans,
                  'masters_modified': False, 'standalone_masters_removed': False}
        atomic_bytes(validation_path, json_bytes(report))
        if args.prepare_only:
            report.update(passed=True, prepare_only=True, completed_at=datetime.now(timezone.utc).isoformat())
            atomic_bytes(validation_path, json_bytes(report))
            print(json.dumps({'passed': True, 'published': False, 'prepare_only': True,
                              'clips': len(encoded), 'cache': str(cache),
                              'validation': str(validation_path)}), flush=True)
            return
        for plan in plans:
            website = Path(plan['website'])
            changed = 0
            for relative, entry in desired.items():
                changed += atomic_copy(entry['source'], checked_destination(website, relative),
                                       {k: entry[k] for k in ('bytes', 'sha256')})
            # The manifest is last among new assets, so it never references partial files.
            atomic_bytes(checked_destination(website, 'content/go2.json'), manifest_data)
            # Removed downloads no longer appear in the manifest at this point.
            for row in plan['removals']:
                checked_destination(website, row['path']).unlink()
            for relative, entry in desired.items():
                require(identity(website / relative) == {k: entry[k] for k in ('bytes', 'sha256')},
                        f'Published checksum mismatch: {website / relative}')
            actual_bytes = sum(deployed_files(website).values()) + 4096
            require(actual_bytes <= args.max_site_mb * 1_000_000, f'Final site exceeds budget: {website}')
            require(read(website / 'content/go2.json') == manifest, 'Published manifest mismatch')
            plan.update(passed=True, changed_assets=changed, actual_bytes=actual_bytes,
                        manifest_sha256=sha256(website / 'content/go2.json'))
            atomic_bytes(validation_path, json_bytes(report))
        report.update(passed=True, published=True, completed_at=datetime.now(timezone.utc).isoformat(),
                      published_assets={k: {f: v[f] for f in ('bytes', 'sha256')} for k, v in desired.items()})
        atomic_bytes(validation_path, json_bytes(report))
        print(json.dumps({'passed': True, 'clips': len(encoded), 'validation': str(validation_path),
                          'websites': [{'path': p['website'], 'bytes': p['actual_bytes']} for p in plans]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='NAS gallery directory with website_manifest.json')
    parser.add_argument('--website', type=Path, action='append', required=True, help='Repeat for each website checkout')
    parser.add_argument('--retained-source', type=Path, action='append', default=[],
                        help='Additional NAS master gallery to prove existing website copies are safely retained; repeatable')
    parser.add_argument('--prepare-only', action='store_true',
                        help='Encode, validate and check budgets without copying or deleting any website file')
    parser.add_argument('--crf', type=int, default=22)
    parser.add_argument('--preset', choices=('medium', 'slow', 'slower'), default='medium')
    parser.add_argument('--jobs', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--max-site-mb', type=int, default=950, help='Decimal MB, all deployed files (default: 950)')
    parser.add_argument('--cache', type=Path, help='Separate NAS cache; never inside masters or a website')
    parser.add_argument('--validation', type=Path, help='Default: source parent / webpublish_validation.json')
    parser.add_argument('--ffmpeg', type=Path, default=FFMPEG)
    parser.add_argument('--ffprobe', type=Path, default=FFPROBE)
    args = parser.parse_args()
    if not 0 <= args.crf <= 51:
        parser.error('--crf must be 0–51')
    if args.max_site_mb <= 0:
        parser.error('--max-site-mb must be positive')
    for binary in (args.ffmpeg, args.ffprobe):
        require(binary.is_absolute() and binary.is_file() and os.access(binary, os.X_OK),
                f'An executable absolute tool path is required: {binary}')
    publish(args)


if __name__ == '__main__':
    main()
