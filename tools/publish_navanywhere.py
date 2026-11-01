#!/usr/bin/env python3
"""Stage an audited ModelScope snapshot and publish it to Hugging Face.

The inventory is a JSON list of {path, size, sha256} for source repository files.
Large archives are symlinked, never repacked. Only inventory-listed files are
published. Uploads use huggingface_hub's resumable Xet pipeline. No credentials
are accepted on the command line or written into the release directory.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time


NWM = Path('/file_system/nas/algorithm/dujun.nie/nwm')
DEFAULT_RELEASE = NWM / 'releases/NavAnywhere-20261016'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def safe_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name:
        raise ValueError(f'Unsafe repository path: {name!r}')
    return path


def network_retry(label, operation, attempts):
    for attempt in range(attempts):
        try:
            return operation()
        except ValueError:
            # Local manifest or checksum failures need investigation, not retry.
            raise
        except Exception as error:
            response = getattr(error, 'response', None)
            status = getattr(response, 'status_code', None)
            # Do not log exception text: it can contain signed upload URLs.
            print(f'{label} attempt {attempt + 1} failed: '
                  f'{type(error).__name__}; HTTP status: {status}', flush=True)
            if status in (401, 403, 404, 413, 422) or attempt + 1 == attempts:
                raise SystemExit(f'{label} incomplete. Rerun the same command to resume.') from None
            time.sleep(min(60, 5 * (2 ** attempt)))


def dataset_card(source, repo_id, modelscope_repo):
    body = source.split('---', 2)[2].lstrip() if source.startswith('---\n') else source
    body = body.replace('# LAM-Data', '# NavAnywhere', 1)
    body = body.replace('LittleBoss/LAM-Data', modelscope_repo)
    body = body.replace('/path/to/LAM-Data', '/path/to/NavAnywhere')
    introduction = (
        '\nNavAnywhere 是 OpenNWM 使用的导航视频数据发布。此仓库镜像原 ModelScope '
        'LAM-Data 的完整已发布文件，包括 NavAnywhere 快照与历史动作数据。'
        '压缩包名称、包内路径和数据校验值保持原样。\n\n'
        f'- ModelScope：[源仓库](https://modelscope.cn/datasets/{modelscope_repo})\n'
        '- 代码：[OpenNWM](https://github.com/OpenSpace-Lab/OpenNWM)\n\n'
        '### Hugging Face 下载\n\n```bash\n'
        f'hf download {repo_id} --repo-type dataset --local-dir /path/to/NavAnywhere\n'
        '```\n\n中断后重跑同一命令即可续传。数据以归档文件发布，请按下方说明解压。\n'
    )
    body = body.replace('# NavAnywhere\n', '# NavAnywhere\n' + introduction, 1)
    return ('---\nlicense: other\nlicense_name: source-specific-terms\n'
            f'license_link: https://huggingface.co/datasets/{repo_id}/blob/main/README.md\n'
            'pretty_name: NavAnywhere\n'
            'task_categories:\n- other\ntags:\n- visual-navigation\n- robotics\n'
            '- world-model\n---\n\n' + body)


def prepare(args):
    release = args.release_root.resolve()
    inventory_path = args.inventory or release / 'modelscope-files.json'
    inventory = json.loads(inventory_path.read_text())
    if not inventory or not isinstance(inventory, list):
        raise ValueError('An audited nonempty source inventory is required')
    names = [item['path'] for item in inventory]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate paths in inventory')
    if 'NavAnywhere/manifests/snapshot-v3.json' not in names or 'README.md' not in names:
        raise ValueError('Inventory does not contain the complete published v3 snapshot')
    roots = [release / 'source-metadata'] + (args.source_root or [
        NWM / 'modelscope_staging/great-outdoors-20260914',
        NWM / 'modelscope_staging/ego4d-20260908',
        NWM / 'modelscope_staging/LAM-Data',
        NWM / '.lam_data_upload',
    ])
    release.mkdir(parents=True, exist_ok=True)
    print(f'Free release disk bytes: {shutil.disk_usage(release).free}', flush=True)
    folder = release / 'dataset'
    folder.mkdir(exist_ok=True)
    rows = []
    for item in sorted(inventory, key=lambda x: x['path']):
        name = item['path']
        safe_path(name)
        source = next((root / name for root in roots
                       if (root / name).is_file()
                       and (root / name).stat().st_size == int(item['size'])), None)
        if source is None:
            raise FileNotFoundError(f'Missing source file with matching size: {name}')
        expected = item.get('sha256', '')
        # Verify small metadata now. Archive checksums are checked against the
        # server after upload; --hash-all additionally reads all local archives.
        if int(item['size']) < 100_000_000 or args.hash_all or not expected:
            actual = digest(source)
            if expected and actual != expected:
                raise ValueError(f'Source checksum mismatch: {name}')
            expected = actual
        destination = folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name == 'README.md':
            if destination.is_symlink():
                destination.unlink()
            destination.write_text(dataset_card(source.read_text(), args.repo_id,
                                                args.modelscope_repo))
        elif destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError(f'Existing stage symlink targets a different file: {name}')
        elif destination.exists():
            raise ValueError(f'Refusing to replace unexpected staged file: {name}')
        else:
            destination.symlink_to(source.resolve())
        rows.append({'path': name, 'size': destination.stat().st_size,
                     'sha256': digest(destination) if name == 'README.md' else expected})
    allowed = set(names)
    unexpected = [str(p.relative_to(folder)) for p in folder.rglob('*')
                  if p.is_file() and str(p.relative_to(folder)) not in allowed
                  and '.cache' not in p.relative_to(folder).parts]
    if unexpected:
        raise ValueError(f'Unexpected staged files: {unexpected}')
    snapshot = json.loads((folder / 'NavAnywhere/manifests/snapshot-v3.json').read_text())
    for pack in snapshot['packs']:
        row = next((x for x in rows if x['path'] == pack['repository_path']), None)
        if not row or row['size'] != pack['compressed_bytes'] or row['sha256'] != pack['sha256']:
            raise ValueError(f'Snapshot pack absent or inconsistent: {pack["repository_path"]}')
    manifest = {'repo_id': args.repo_id, 'source_repo': args.modelscope_repo,
                'files': rows, 'total_bytes': sum(row['size'] for row in rows)}
    write_json(release / 'upload-manifest.json', manifest)
    print(f'Staged {len(rows)} files, {manifest["total_bytes"]:,} bytes, '
          f'{len(snapshot["packs"])} NavAnywhere packs: {folder}', flush=True)
    return manifest


def verify(args, api, manifest, *, skip_readme=False):
    from huggingface_hub import RepoFile, hf_hub_download
    remote = {f.path: f for f in api.list_repo_tree(
        args.repo_id, repo_type='dataset', recursive=True) if isinstance(f, RepoFile)}
    problems = []
    for row in manifest['files']:
        if skip_readme and row['path'] == 'README.md':
            continue
        entry = remote.get(row['path'])
        if entry is None or entry.size != row['size']:
            problems.append({'path': row['path'], 'error': 'missing or size mismatch'})
            continue
        if entry.lfs:
            actual = entry.lfs.sha256
        elif row['size'] < 100_000_000:
            downloaded = hf_hub_download(args.repo_id, row['path'], repo_type='dataset',
                                         cache_dir=args.release_root / 'verification-cache')
            actual = digest(Path(downloaded))
        else:
            problems.append({'path': row['path'], 'error': 'server has no SHA256 metadata'})
            continue
        if actual != row['sha256']:
            problems.append({'path': row['path'], 'error': 'SHA256 mismatch'})
    checked = len(manifest['files']) - int(skip_readme)
    report = {'repo_id': args.repo_id, 'checked_files': checked,
              'ok': not problems, 'problems': problems}
    write_json(args.release_root / 'verification.json', report)
    if problems:
        raise ValueError(f'Remote verification failed for {len(problems)} files; see verification.json')
    print(f'Verified {checked} remote files by size and SHA256.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'upload', 'finalize', 'verify'])
    parser.add_argument('--repo-id', required=True, help='Hugging Face owner/NavAnywhere')
    parser.add_argument('--release-root', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--source-root', type=Path, action='append')
    parser.add_argument('--modelscope-repo', default='LittleBoss/NavAnywhere')
    parser.add_argument('--hash-all', action='store_true')
    parser.add_argument('--attempts', type=int, default=5)
    parser.add_argument('--direct-storage', action='store_true',
                        help='Bypass the proxy for Xet storage hosts, keeping the Hub API proxy')
    args = parser.parse_args()
    if args.repo_id.count('/') != 1 or args.repo_id.split('/')[1] != 'NavAnywhere':
        parser.error('--repo-id must be owner/NavAnywhere')
    if args.attempts < 1:
        parser.error('--attempts must be positive')
    if args.action == 'prepare':
        prepare(args)
        return
    if args.direct_storage:
        hosts = 'cas-server.xethub.hf.co,transfer.xethub.hf.co,cas-bridge.xethub.hf.co'
        for key in ('NO_PROXY', 'no_proxy'):
            os.environ[key] = ','.join(filter(None, [os.environ.get(key, ''), hosts]))
    from huggingface_hub import HfApi
    api = HfApi(endpoint='https://huggingface.co')
    identity = network_retry('Authentication', api.whoami, args.attempts)
    print(f'Hugging Face account: {identity["name"]}; destination: {args.repo_id}', flush=True)
    manifest = json.loads((args.release_root / 'upload-manifest.json').read_text())
    if manifest['repo_id'] != args.repo_id:
        raise ValueError('Prepared repo_id differs; rerun prepare with the intended destination')
    if args.action == 'upload':
        from packaging.version import Version
        from huggingface_hub.utils._runtime import is_xet_available
        if Version(importlib.metadata.version('huggingface_hub')) < Version('1.27.0') or not is_xet_available():
            raise RuntimeError('Use huggingface_hub >= 1.27.0 with hf_xet for resumable streaming uploads; '
                               'the nwm conda environment has these dependencies.')
        # Authenticate before creating or writing a repository. Creating public
        # storage does not enable paid storage or change billing settings.
        network_retry('Repository initialization', lambda: api.create_repo(
            args.repo_id, repo_type='dataset', private=False, exist_ok=True), args.attempts)
        info = network_retry('Repository inspection', lambda: api.repo_info(
            args.repo_id, repo_type='dataset'), args.attempts)
        if info.private:
            raise ValueError('Existing destination is private; refusing an unintended private TB upload')
        network_retry('Archive upload', lambda: api.upload_folder(
            repo_id=args.repo_id, repo_type='dataset',
            folder_path=args.release_root / 'dataset',
            allow_patterns=[r['path'] for r in manifest['files'] if r['path'] != 'README.md'],
            commit_message='Publish NavAnywhere source archives and metadata'), args.attempts)
    if args.action in ('upload', 'finalize'):
        network_retry('Archive verification', lambda: verify(
            args, api, manifest, skip_readme=True), args.attempts)
        network_retry('Dataset card finalization', lambda: api.upload_file(
                        repo_id=args.repo_id, repo_type='dataset',
                        path_or_fileobj=args.release_root / 'dataset/README.md',
                        path_in_repo='README.md',
                        commit_message='Finalize verified NavAnywhere dataset card'), args.attempts)
    network_retry('Final verification', lambda: verify(args, api, manifest), args.attempts)
    print(f'https://huggingface.co/datasets/{args.repo_id}', flush=True)


if __name__ == '__main__':
    main()
