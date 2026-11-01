#!/usr/bin/env python3
"""Publish byte-identical website assets to the public OpenNWM media mirror.

Run with modelscope_hub installed. Uses existing ModelScope login credentials;
credentials and local source paths are never included in the public payload.
Only the real-world clips selected by content/go2.json are mirrored.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = Path('/file_system/nas/algorithm/dujun.nie/nwm/releases/OpenNWM-Website-20261029')


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def stage(args):
    site = args.site_root.resolve()
    selected = {clip['id'] for clip in json.loads((site / 'content/go2.json').read_text())['clips']}
    sources = []
    for source in sorted((site / 'assets').rglob('*')):
        if not source.is_file():
            continue
        relative = source.relative_to(site)
        if relative.parts[1] == 'go2' and (len(relative.parts) < 4 or relative.parts[2] not in selected):
            continue
        if any(part.startswith('.') for part in relative.parts):
            continue
        sources.append((source, relative))
    if not sources or not selected:
        raise ValueError('Expected website assets and selected real-world clips')
    total = sum(source.stat().st_size for source, _ in sources)
    args.release_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.release_dir).free
    if free < total * 2:
        raise RuntimeError(f'Insufficient staging space: {free:,} free, {total:,} needed')
    payload = args.release_dir / 'payload'
    payload.mkdir(exist_ok=True)
    files = {}
    for source, relative in sources:
        target = payload / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = source.stat()
        source_hash = digest(source)
        if not target.is_file() or target.stat().st_size != before.st_size or digest(target) != source_hash:
            shutil.copyfile(source, target)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or digest(target) != source_hash:
            raise RuntimeError(f'Source changed or staged hash differs: {relative}')
        files[relative.as_posix()] = {'bytes': before.st_size, 'sha256': source_hash}
    write_json(payload / 'media-manifest.json', {
        'version': 1, 'release': args.tag, 'repo_id': args.repo_id,
        'website': 'https://openspace-lab.github.io/OpenNWM/',
        'encoding': 'Original website bytes, without media re-encoding',
        'real_world_clips': sorted(selected),
        'files': [{'path': name, **value} for name, value in files.items()],
    })
    (payload / 'README.md').write_text('''---
license: cc-by-nc-4.0
tags:
- OpenNWM
- website-media
---

# OpenNWM website media

Original media files for the [OpenNWM project website](https://openspace-lab.github.io/OpenNWM/).
This public delivery mirror preserves the exact website file bytes; it does not
re-encode videos or reduce image quality. See `media-manifest.json` for sizes and
SHA256 hashes. Original sources retain their respective licenses.
''')
    write_json(payload / 'configuration.json', {'framework': 'pytorch', 'model': {'type': 'OpenNWM-Website-Media'}})
    for name in ('media-manifest.json', 'README.md', 'configuration.json'):
        file = payload / name
        files[name] = {'bytes': file.stat().st_size, 'sha256': digest(file)}
    (payload / 'SHA256SUMS').write_text(''.join(f"{value['sha256']}  {name}\n" for name, value in sorted(files.items())))
    checksum_file = payload / 'SHA256SUMS'
    files['SHA256SUMS'] = {'bytes': checksum_file.stat().st_size, 'sha256': digest(checksum_file)}
    write_json(args.release_dir / 'payload-files.json', files)
    print(f'Staged {len(files)} files, {sum(row["bytes"] for row in files.values()):,} bytes; {len(selected)} real-world clips', flush=True)
    return payload, files


def upload(args, payload, files):
    if args.direct:
        for key in list(os.environ):
            if key.lower() in {'http_proxy', 'https_proxy', 'all_proxy'}:
                os.environ.pop(key)
        os.environ['NO_PROXY'] = '*'
    from modelscope_hub import HubApi
    api = HubApi()
    try:
        user = api.whoami()
    except Exception:
        token = os.environ.get('MODELSCOPE_API_TOKEN')
        if not token:
            raise RuntimeError('ModelScope login unavailable; use the SDK login cache or MODELSCOPE_API_TOKEN') from None
        user = api.login(token)
    if user.username != args.repo_id.split('/')[0]:
        raise ValueError('Authenticated ModelScope account does not own the requested namespace')
    if not api.repo_exists(args.repo_id, 'model'):
        api.create_repo(args.repo_id, 'model', visibility='public', license='cc-by-nc-4.0',
                        description='Byte-identical media delivery mirror for the OpenNWM project website')
    print(f'Uploading {len(files)} files to https://modelscope.cn/models/{args.repo_id}', flush=True)
    api.upload_folder(args.repo_id, 'model', payload, allow_patterns=list(files),
                      max_workers=args.workers, use_cache=True, disable_tqdm=True,
                      commit_message=f'Publish byte-identical OpenNWM website media {args.tag}')
    remote = {entry.path: entry for entry in api.list_repo_files(args.repo_id, 'model') if not entry.is_dir}
    verified = []
    for name, expected in files.items():
        entry = remote.get(name)
        if entry is None or entry.size != expected['bytes']:
            raise RuntimeError(f'Remote file missing or size mismatch: {name}')
        remote_hash = getattr(entry, 'sha256', None)
        blob = getattr(entry, 'blob_id', None)
        if not remote_hash and isinstance(blob, str) and len(blob) == 64:
            remote_hash = blob
        lfs = getattr(entry, 'lfs', None)
        if not remote_hash and lfs:
            remote_hash = lfs.get('oid') or lfs.get('sha256')
            if isinstance(remote_hash, str):
                remote_hash = remote_hash.removeprefix('sha256:')
        if not remote_hash:
            if expected['bytes'] > 32 * 1024 * 1024:
                raise RuntimeError(f'Remote SHA256 not available for large asset: {name}')
            remote_hash = digest(Path(api.download_file(args.repo_id, 'model', name)))
        if remote_hash != expected['sha256']:
            raise RuntimeError(f'Remote SHA256 mismatch: {name}')
        verified.append({'path': name, **expected})
    revisions = api.list_repo_revisions(args.repo_id, 'model')
    if args.tag not in json.dumps(revisions):
        api.create_repo_tag(args.repo_id, 'model', args.tag, revision='master')
    pinned_manifest = api.download_file(args.repo_id, 'model', 'media-manifest.json', revision=args.tag)
    if digest(Path(pinned_manifest)) != files['media-manifest.json']['sha256']:
        raise RuntimeError(f'Release tag {args.tag!r} already points to different assets; choose a new tag')
    base = f'https://modelscope.cn/models/{args.repo_id}/resolve/{args.tag}/'
    receipt = {
        'repo_id': args.repo_id, 'asset_base_url': base,
        'verified_at': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': files['media-manifest.json']['sha256'],
        'files': verified,
    }
    write_json(args.release_dir / 'upload-verification.json', receipt)
    print(f'Verified {len(verified)} remote file sizes and SHA256 hashes', flush=True)
    print(f'Asset base URL: {base}', flush=True)
    print(f'Media manifest SHA256: {receipt["manifest_sha256"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site-root', type=Path, default=ROOT / 'website')
    parser.add_argument('--release-dir', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--repo-id', default='LittleBoss/OpenNWM-Website')
    parser.add_argument('--tag', default='site-20261029-v3')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--direct', action='store_true')
    args = parser.parse_args()
    payload, files = stage(args)
    if args.upload:
        upload(args, payload, files)


if __name__ == '__main__':
    main()
