#!/usr/bin/env python3
"""Stage, publish, and verify the incremental OpenNWM website media v7 release.

Use --stage-only after privacy review has completed, then --upload explicitly.
No command changes GitHub Pages or uploads private local source paths.
Only changed selected assets and the two delivery manifests are uploaded.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RELEASES = Path('/file_system/nas/algorithm/dujun.nie/nwm/releases')
REPO = 'LittleBoss/OpenNWM-Website'
TAG = 'site-20261031-v7'
BASE_TAG = 'site-20261030-v6'
ORIGIN = 'https://b0b8k1ng.github.io'
LOGO_ASSETS = ('assets/paper/figure-2-framework-dark.svg',
               'assets/opennwm-logo.svg', 'assets/opennwm-logo.pdf')


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    write_text_durable(temporary, json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def write_text_durable(path, value):
    with path.open('w') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(path):
    before = path.stat()
    result = {'bytes': before.st_size, 'sha256': digest(path)}
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        # NAS can finalize a freshly written file's mtime on its first read.
        # Accept that metadata transition only when two complete reads agree
        # byte-for-byte and metadata remains stable for the second read.
        second = {'bytes': after.st_size, 'sha256': digest(path)}
        final = path.stat()
        if (second != result or
                (after.st_size, after.st_mtime_ns) != (final.st_size, final.st_mtime_ns)):
            raise RuntimeError(f'Source changed during hashing: {path}')
    return result


def asset_path(value):
    if not isinstance(value, str):
        raise ValueError(f'Expected a relative website asset path: {value!r}')
    path = PurePosixPath(value)
    if (not value.startswith('assets/')
            or path.is_absolute() or any(part in ('.', '..') for part in path.parts)
            or path.as_posix() != value or '\\' in value or '?' in value or '#' in value):
        raise ValueError(f'Expected a relative website asset path: {value!r}')
    return path.as_posix()


def connect(args):
    # Must be set before importing the SDK, which captures this setting at import.
    os.environ['UPLOAD_LFS_ENFORCE_THRESHOLD'] = '1'
    if args.direct:
        for key in list(os.environ):
            if key.lower() in {'http_proxy', 'https_proxy', 'all_proxy'}:
                os.environ.pop(key)
        os.environ['NO_PROXY'] = '*'
    from modelscope_hub import HubApi
    from modelscope_hub.constants import UPLOAD_LFS_ENFORCE_THRESHOLD
    if UPLOAD_LFS_ENFORCE_THRESHOLD != 1:
        raise RuntimeError('SDK LFS threshold must be exactly 1 byte')
    api = HubApi()
    if api.whoami().username != REPO.split('/')[0]:
        raise RuntimeError('Unexpected authenticated ModelScope account owner')
    return api


def snapshot(api, revision):
    result = {}
    for entry in api.list_repo_files(REPO, 'model', revision=revision):
        if entry.is_dir:
            continue
        sha = entry.blob_id
        if not isinstance(sha, str) or len(sha) != 64:
            if entry.size > 32 * 1024 * 1024:
                raise RuntimeError(f'No SHA256 metadata for large file: {entry.path}')
            sha = digest(Path(api.download_file(REPO, 'model', entry.path, revision=revision)))
        result[entry.path] = {'bytes': entry.size, 'sha256': sha}
    return result


def require_match(actual, expected, label, exact=True):
    mismatches = sorted(p for p, row in expected.items() if actual.get(p) != row)
    extras = sorted(set(actual) - set(expected)) if exact else []
    if mismatches or extras:
        raise RuntimeError(f'{label} differs: mismatches={mismatches[:12]}, unexpected={extras[:12]}')
    return {'revision': label, 'files': len(expected), 'size_verified': len(expected),
            'sha256_verified': len(expected), 'verified_at': now()}


def has_tag(api):
    def contains(value):
        if isinstance(value, str):
            return value == TAG
        if isinstance(value, dict):
            return any(contains(item) for item in value.values())
        if isinstance(value, list):
            return any(contains(item) for item in value)
        return False
    return contains(api.list_repo_revisions(REPO, 'model'))


def review_evidence(path, go2=None, sources=None):
    original = identity(path)
    evidence = read_json(path)
    if evidence.get('status') != 'passed' or evidence.get('selection_preserved') is not True:
        raise RuntimeError('Privacy review must report status="passed" and selection_preserved=true')
    if go2 is not None:
        rows = evidence.get('clips', [])
        reviewed = {row['id']: row for row in rows}
        selected = {row['id']: row for row in go2['clips']}
        if (len(reviewed) != len(rows) or len(selected) != len(go2['clips'])
                or set(reviewed) != set(selected)):
            raise RuntimeError('Privacy review must cover every selected clip exactly once')
        for clip_id, clip in selected.items():
            row = reviewed[clip_id]
            for key in ('video', 'poster'):
                path_value = asset_path(clip[key])
                if (row.get(key) != path_value
                        or row.get(key + '_sha256') != sources[path_value]['sha256']):
                    raise RuntimeError(f'Privacy review does not match {clip_id} {key}')
    if identity(path) != original:
        raise RuntimeError('Privacy review changed while being read')
    return {'path': str(path.resolve()), **original}


def stage(args, api):
    release, payload = args.release_dir, args.release_dir / 'payload'
    if (release / 'stage.json').exists():
        raise RuntimeError('A frozen stage already exists; use --upload or a fresh release directory')
    if has_tag(api):
        raise RuntimeError(f'Immutable target tag already exists: {TAG}')
    base = read_json(args.base_release / 'payload-files.json')
    previous_public = read_json(args.base_release / 'public-verification.json')
    if previous_public.get('ready_for_website') is not True:
        raise RuntimeError('The base release did not pass public delivery verification')
    old_manifest = args.base_release / 'payload/media-manifest.json'
    if identity(old_manifest) != base['media-manifest.json']:
        raise RuntimeError('Base media-manifest.json no longer matches its saved identity')
    manifest = read_json(old_manifest)
    if manifest['release'] != BASE_TAG or manifest['repo_id'] != REPO:
        raise RuntimeError('Unexpected base media manifest identity')
    require_match({row['path']: {k: row[k] for k in ('bytes', 'sha256')}
                   for row in manifest['files']},
                  {p: row for p, row in base.items() if p.startswith('assets/')},
                  'base manifest assets')
    base_remote = snapshot(api, BASE_TAG)
    require_match(base_remote, base, BASE_TAG, exact=False)
    require_match(snapshot(api, 'master'), base_remote, 'master before staging')
    go2_path = args.site_root / 'content/go2.json'
    selected_identity = identity(go2_path)
    go2 = read_json(go2_path)
    if go2.get('status') != 'ready' or not go2.get('clips'):
        raise RuntimeError('The selected Go2 gallery is not ready')
    candidates = set(LOGO_ASSETS)
    for clip in go2['clips']:
        candidates.update(asset_path(clip[key]) for key in ('video', 'poster'))
        candidates.update(asset_path(value) for value in clip.get('downloads', {}).values())
    candidates.update(asset_path(value) for value in args.extra_asset)
    if not args.privacy_review:
        raise RuntimeError('--privacy-review is required to freeze a publishable stage')
    # A public report stored under assets is part of the incremental release.
    try:
        report_asset = args.privacy_review.resolve().relative_to(args.site_root.resolve()).as_posix()
    except ValueError:
        report_asset = None
    if report_asset and report_asset.startswith('assets/'):
        candidates.add(asset_path(report_asset))
    sources = {p: identity(args.site_root / p) for p in sorted(candidates)}
    changes = {p: row for p, row in sources.items() if base_remote.get(p) != row}
    review = review_evidence(args.privacy_review, go2, sources)
    if report_asset in sources and sources[report_asset] != {k: review[k] for k in ('bytes', 'sha256')}:
        raise RuntimeError('Privacy report changed during source validation')
    # Media under review is never implicitly published by staging.
    if not changes:
        raise RuntimeError('No changed/new selected assets were found')
    release.mkdir(parents=True, exist_ok=True)
    required = 2 * sum(row['bytes'] for row in changes.values()) + 32 * 1024 * 1024
    free = shutil.disk_usage(release).free
    print(f'Disk free={free:,} bytes; staging required={required:,} bytes', flush=True)
    if free < required:
        raise RuntimeError('Insufficient NAS staging capacity')
    print(f'Staging {len(changes)} changed assets, including '
          f'{sum(p.endswith(".mp4") for p in changes)} videos', flush=True)
    for path, row in changes.items():
        target = payload / path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.copying')
        # Flush the NAS write before hashing: otherwise delayed metadata updates
        # can change mtime during the first read even though content is stable.
        with (args.site_root / path).open('rb') as source, temporary.open('wb') as destination:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        if identity(temporary) != row or identity(args.site_root / path) != row:
            raise RuntimeError(f'Source changed or copy failed: {path}')
        temporary.replace(target)
    expected = {**base, **changes}
    manifest['release'] = TAG
    manifest['real_world_clips'] = [clip['id'] for clip in go2['clips']]
    manifest['encoding'] = ('Changed assets are copied byte-for-byte from the reviewed website '
                            'sources. Other assets retain their site-20261030-v6 bytes.')
    manifest['files'] = [{'path': p, **row} for p, row in sorted(expected.items())
                         if p.startswith('assets/')]
    manifest['delivery_provenance'] = {
        'base_release': BASE_TAG,
        'change': 'Privacy-reviewed real-world gallery and green N logo in the method figure.',
        'changed_asset_count': len(changes),
        'unchanged_asset_count': sum(p.startswith('assets/') and p not in changes for p in base),
        'replacements': [{'path': p, 'previous_sha256': base_remote.get(p, {}).get('sha256'),
                          **row} for p, row in sorted(changes.items())],
    }
    write_json(payload / 'media-manifest.json', manifest)
    expected['media-manifest.json'] = identity(payload / 'media-manifest.json')
    write_text_durable(payload / 'SHA256SUMS', ''.join(
        f"{row['sha256']}  {p}\n" for p, row in sorted(expected.items()) if p != 'SHA256SUMS'))
    expected['SHA256SUMS'] = identity(payload / 'SHA256SUMS')
    upload_paths = sorted(set(changes) | {'media-manifest.json', 'SHA256SUMS'})
    remote_expected = {**base_remote, **expected}
    require_match({p: identity(payload / p) for p in upload_paths},
                  {p: expected[p] for p in upload_paths}, 'staged payload')
    write_json(release / 'payload-files.json', expected)
    write_json(release / 'baseline-verification.json', {
        'base_revision': BASE_TAG, 'remote_files': base_remote, 'verified_at': now()})
    if identity(go2_path) != selected_identity or review_evidence(args.privacy_review, go2, sources) != review:
        raise RuntimeError('Gallery or privacy review changed during staging')
    frozen = {
        'version': 1, 'repo_id': REPO, 'revision': TAG, 'base_revision': BASE_TAG,
        'site_root': str(args.site_root.resolve()), 'created_at': now(),
        'selected_go2': {'path': str(go2_path.resolve()), **selected_identity},
        'privacy_review': review, 'source_identities': sources,
        'changed_assets': sorted(changes), 'upload_paths': upload_paths,
        'remote_expected': remote_expected,
        'payload_files_identity': identity(release / 'payload-files.json'),
    }
    write_json(release / 'inheritance.json', {
        'base_revision': BASE_TAG, 'base_release_dir': str(args.base_release.resolve()),
        'inherited_files': sorted(set(base_remote) - set(upload_paths)),
        'changed_files': upload_paths})
    write_json(release / 'stage.json', frozen)
    write_json(release / 'public-verification.json', {
        'revision': TAG, 'ready_for_website': False, 'status': 'staged_only',
        'changed_assets': sorted(changes), 'created_at': now()})
    print(f'Frozen stage: {release}; {len(upload_paths)} upload paths; no remote writes', flush=True)


def load_stage(args, require_source):
    frozen = read_json(args.release_dir / 'stage.json')
    if (frozen['repo_id'], frozen['revision'], frozen['base_revision']) != (REPO, TAG, BASE_TAG):
        raise RuntimeError('Unexpected frozen release identity')
    if identity(args.release_dir / 'payload-files.json') != frozen['payload_files_identity']:
        raise RuntimeError('Frozen payload-files.json was modified')
    files = read_json(args.release_dir / 'payload-files.json')
    require_match({p: identity(args.release_dir / 'payload' / p) for p in frozen['upload_paths']},
                  {p: files[p] for p in frozen['upload_paths']}, 'frozen payload')
    if require_source:
        review = frozen.get('privacy_review')
        selected = frozen['selected_go2']
        if identity(Path(selected['path'])) != {k: selected[k] for k in ('bytes', 'sha256')}:
            raise RuntimeError('The selected gallery changed after staging')
        site = Path(frozen['site_root'])
        require_match({p: identity(site / p) for p in frozen['source_identities']},
                      frozen['source_identities'], 'source assets after staging')
        if not review or review_evidence(Path(review['path']), read_json(Path(selected['path'])),
                                         frozen['source_identities']) != review:
            raise RuntimeError('Completed privacy-review evidence is required before upload')
    return frozen, files


def public_url(path):
    return f'https://modelscope.cn/models/{REPO}/resolve/{TAG}/' + urllib.parse.quote(path, safe='/')


def request_bytes(path, expected, local, start=None, end=None):
    headers = {'Origin': ORIGIN, 'Accept-Encoding': 'identity'}
    if start is not None:
        headers['Range'] = f'bytes={start}-{end}'
    request = urllib.request.Request(public_url(path), headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        status, cors = response.status, response.headers.get('Access-Control-Allow-Origin')
        content_range = response.headers.get('Content-Range')
        limit = expected['bytes'] if start is None else end - start + 1
        body = response.read(limit + 1)
    if cors not in ('*', ORIGIN):
        raise RuntimeError(f'Public CORS mismatch: {path}: {cors}')
    if start is None:
        if status != 200 or len(body) != expected['bytes'] or hashlib.sha256(body).hexdigest() != expected['sha256']:
            raise RuntimeError(f'Public full-file identity mismatch: {path}')
    else:
        if status != 206 or content_range != f'bytes {start}-{end}/{expected["bytes"]}':
            raise RuntimeError(f'Public Range mismatch: {path}: {status}, {content_range}')
        with local.open('rb') as stream:
            stream.seek(start)
            sample = stream.read(limit)
        if len(body) != limit or body != sample:
            raise RuntimeError(f'Public Range bytes differ: {path}: {start}-{end}')
    return {'path': path, 'status': status, 'cors': cors,
            'request_range': headers.get('Range'), 'content_range': content_range,
            'sample_bytes_verified': len(body), 'sample_sha256': hashlib.sha256(body).hexdigest(),
            'full_file_verified': start is None}


def public_checks(args, api, frozen, files):
    metadata = require_match(snapshot(api, TAG), frozen['remote_expected'], TAG)
    write_json(args.release_dir / 'pinned-verification.json', metadata)
    jobs = []
    for path in frozen['upload_paths']:
        row, local = files[path], args.release_dir / 'payload' / path
        if path.endswith('.mp4'):
            jobs.extend([(path, row, local, 0, min(65535, row['bytes'] - 1)),
                         (path, row, local, max(0, row['bytes'] - 65536), row['bytes'] - 1)])
        else:
            jobs.append((path, row, local, None, None))
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(request_bytes, *job): job[0] for job in jobs}
        for future in as_completed(futures):
            path = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({'path': path, 'error': str(exc)})
                print(f'Public verification FAILED: {path}: {exc}', flush=True)
    receipt = {
        'revision': TAG, 'base_revision': BASE_TAG, 'repo_id': REPO,
        'asset_base_url': f'https://modelscope.cn/models/{REPO}/resolve/{TAG}/',
        'manifest_sha256': files['media-manifest.json']['sha256'], 'verified_at': now(),
        'remote_size_sha256_files': metadata['files'],
        'changed_video_count': sum(p.endswith('.mp4') for p in frozen['changed_assets']),
        'unchanged_files': len(frozen['remote_expected']) - len(frozen['upload_paths'])
                           - int('sdk_generated_gitattributes' in frozen),
        'public_probes': len(results), 'sample_bytes_verified': sum(r['sample_bytes_verified'] for r in results),
        'results': sorted(results, key=lambda r: (r['path'], r['request_range'] or '')),
        'failures': failures, 'ready_for_website': not failures,
        'changed_assets': [{'path': p, **files[p]} for p in frozen['changed_assets']],
    }
    write_json(args.release_dir / 'public-verification.json', receipt)
    if failures:
        raise RuntimeError(f'{len(failures)} public checks failed; website must retain its current media configuration')
    print(f'Public release verified: {TAG}; ready_for_website=true; {len(results)} HTTP probes', flush=True)


def reconcile_lfs_metadata(args, api, frozen, files, baseline):
    """Account for the SDK's exact path-specific LFS rules, not arbitrary drift."""
    original = Path(api.download_file(REPO, 'model', '.gitattributes',
                                      revision=BASE_TAG)).read_bytes()
    original_identity = {'bytes': len(original), 'sha256': hashlib.sha256(original).hexdigest()}
    if original_identity != baseline['.gitattributes']:
        raise RuntimeError('Base .gitattributes bytes do not match the frozen baseline')
    lines = set(original.decode().splitlines())
    additions = [f'"/{path}" filter=lfs diff=lfs merge=lfs -text'
                 for path in frozen['upload_paths']
                 if f'"/{path}" filter=lfs diff=lfs merge=lfs -text' not in lines]
    generated = '\n'.join(original.decode().splitlines() + additions).encode()
    row = {'bytes': len(generated), 'sha256': hashlib.sha256(generated).hexdigest()}
    if frozen['remote_expected']['.gitattributes'] == row:
        return
    if frozen['remote_expected']['.gitattributes'] != original_identity:
        raise RuntimeError('Unexpected frozen .gitattributes identity')
    # Only auxiliary repository metadata and its checksum list change here.
    # The media manifest and every reviewed source asset remain frozen.
    frozen['sdk_generated_gitattributes'] = {
        'baseline': original_identity, 'expected': row, 'added_rules': additions}
    files['.gitattributes'] = row
    frozen['remote_expected']['.gitattributes'] = row
    sums = args.release_dir / 'payload/SHA256SUMS'
    frozen['pre_lfs_checksum_identity'] = files['SHA256SUMS']
    write_text_durable(sums, ''.join(
        f"{value['sha256']}  {path}\n" for path, value in sorted(files.items())
        if path != 'SHA256SUMS'))
    files['SHA256SUMS'] = identity(sums)
    frozen['remote_expected']['SHA256SUMS'] = files['SHA256SUMS']
    write_json(args.release_dir / 'payload-files.json', files)
    frozen['payload_files_identity'] = identity(args.release_dir / 'payload-files.json')
    write_json(args.release_dir / 'lfs-metadata-adjustment.json',
               {'verified_at': now(), **frozen['sdk_generated_gitattributes']})
    inheritance = read_json(args.release_dir / 'inheritance.json')
    inheritance['inherited_files'] = [p for p in inheritance['inherited_files'] if p != '.gitattributes']
    inheritance['sdk_generated_files'] = ['.gitattributes']
    write_json(args.release_dir / 'inheritance.json', inheritance)
    write_json(args.release_dir / 'stage.json', frozen)
    print(f'Accounted for {len(additions)} SDK-generated LFS rules; media manifest unchanged', flush=True)


def upload(args, api):
    frozen, files = load_stage(args, require_source=True)
    baseline = read_json(args.release_dir / 'baseline-verification.json')['remote_files']
    require_match(snapshot(api, BASE_TAG), baseline, BASE_TAG)
    reconcile_lfs_metadata(args, api, frozen, files, baseline)
    if not has_tag(api):
        actual = snapshot(api, 'master')
        expected = frozen['remote_expected']
        # A retry may resume a partial upload, but never overwrite unrelated changes.
        if set(actual) - set(expected) or set(baseline) - set(actual):
            raise RuntimeError('Remote master gained or lost unexpected files')
        for path, row in actual.items():
            allowed = ((baseline.get(path), expected.get(path))
                       if path in frozen['upload_paths'] or path == '.gitattributes'
                       else (baseline.get(path),))
            if path == 'SHA256SUMS':
                allowed += (frozen.get('pre_lfs_checksum_identity'),)
            if row not in allowed:
                raise RuntimeError(f'Remote master has an unrelated change: {path}')
        if actual != expected:
            pending = [p for p in frozen['upload_paths'] if actual.get(p) != expected[p]]
            print(f'Uploading exactly {len(pending)} changed paths; LFS threshold=1', flush=True)
            api.upload_folder(REPO, 'model', args.release_dir / 'payload',
                               allow_patterns=pending, max_workers=args.workers,
                              use_cache=False, disable_tqdm=True, sync_remote_repo=False,
                              commit_message=f'Privacy-reviewed gallery and green N logo: {TAG}')
        write_json(args.release_dir / 'master-verification.json',
                   require_match(snapshot(api, 'master'), expected, 'master after upload'))
        api.create_repo_tag(REPO, 'model', TAG, revision='master')
    public_checks(args, api, frozen, files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--stage-only', action='store_true', help='Read-only remote checks and local NAS staging')
    mode.add_argument('--upload', action='store_true', help='Publish the existing frozen stage, tag, and verify')
    mode.add_argument('--verify-only', action='store_true', help='Retry public checks against the existing immutable tag')
    parser.add_argument('--site-root', type=Path, default=ROOT / 'website')
    parser.add_argument('--base-release', type=Path, default=RELEASES / 'OpenNWM-Website-20261030-v6')
    parser.add_argument('--release-dir', type=Path, default=RELEASES / 'OpenNWM-Website-20261031-v7')
    parser.add_argument('--privacy-review', type=Path,
                        help='Completed JSON with status=passed and selection_preserved=true; required to stage')
    parser.add_argument('--extra-asset', action='append', default=[], help='Additional relative assets/... path, repeat as needed')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--direct', action='store_true', help='Bypass inherited proxy settings, as used for v6')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    api = connect(args)
    if args.stage_only:
        stage(args, api)
    elif args.upload:
        upload(args, api)
    else:
        frozen, files = load_stage(args, require_source=False)
        public_checks(args, api, frozen, files)


if __name__ == '__main__':
    main()
