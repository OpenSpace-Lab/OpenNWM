"""Mirror the verified OpenNWM release to Hugging Face without converting weights."""
import json
import os
from pathlib import Path
import shutil

from publish_navanywhere import digest, network_retry, write_json


def main():
    release = Path('/file_system/nas/algorithm/dujun.nie/nwm/releases/OpenNWM-20261016')
    target = 'LittleB0sS/OpenNWM'
    print(f'PID: {os.getpid()}; free bytes: {shutil.disk_usage(release).free}', flush=True)
    for key in ('NO_PROXY', 'no_proxy'):
        os.environ[key] = ','.join(filter(None, [os.environ.get(key, ''),
            'cas-server.xethub.hf.co,transfer.xethub.hf.co,cas-bridge.xethub.hf.co']))
    from huggingface_hub import HfApi, RepoFile, hf_hub_download
    api = HfApi()
    retry = lambda label, fn: network_retry(label, fn, 10)
    identity = retry('Authentication', api.whoami)
    if identity['name'] != 'LittleB0sS':
        raise ValueError('Logged-in account differs from the intended owner')
    source = release / 'payload'
    folder = release / 'hf-payload'
    folder.mkdir(exist_ok=True)
    manifest = json.loads((release / 'payload-files.json').read_text())
    for name, row in manifest.items():
        path = source / name
        if path.stat().st_size != row['bytes']:
            raise ValueError(f'Source size mismatch: {name}')
        if row['bytes'] < 100_000_000 and digest(path) != row['sha256']:
            raise ValueError(f'Source checksum mismatch: {name}')
        if name in ('README.md', 'SHA256SUMS', 'manifest.json'):
            continue
        dst = folder / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink():
            if dst.resolve() != path.resolve():
                raise ValueError(f'Unexpected symlink: {name}')
        elif dst.exists():
            raise ValueError(f'Unexpected file: {name}')
        else:
            dst.symlink_to(path.resolve())
    public_manifest = json.loads((source / 'manifest.json').read_text())
    public_manifest['source_code'] = 'https://github.com/OpenSpace-Lab/OpenNWM'
    # Replace an older staging symlink without writing through it into the source release.
    write_json(folder / 'manifest.json', public_manifest)
    manifest['manifest.json'] = {'bytes': (folder / 'manifest.json').stat().st_size,
                                 'sha256': digest(folder / 'manifest.json')}
    card = (source / 'README.md').read_text()
    card = card.replace('B0B8K1ng/OpenNWM', 'OpenSpace-Lab/OpenNWM')
    card = card.replace('frameworks:\n- pytorch', 'library_name: pytorch')
    card = card.replace('- Repository:', '- ModelScope mirror:')
    card = card.replace('https://modelscope.cn/datasets/LittleBoss/NavAnywhere',
                        'https://huggingface.co/datasets/LittleB0sS/NavAnywhere')
    card = card.replace('from modelscope import snapshot_download',
                        'from huggingface_hub import snapshot_download')
    card = card.replace('snapshot_download("LittleBoss/OpenNWM",',
                        'snapshot_download("LittleB0sS/OpenNWM",')
    (folder / 'README.md').write_text(card)
    manifest['README.md'] = {'bytes': (folder / 'README.md').stat().st_size,
                              'sha256': digest(folder / 'README.md')}
    (folder / 'SHA256SUMS').write_text(''.join(
        f"{row['sha256']}  {name}\n" for name, row in sorted(manifest.items())
        if name != 'SHA256SUMS'))
    manifest['SHA256SUMS'] = {'bytes': (folder / 'SHA256SUMS').stat().st_size,
                              'sha256': digest(folder / 'SHA256SUMS')}
    write_json(release / 'hf-payload-files.json', manifest)
    retry('Create repository', lambda: api.create_repo(target, private=False, exist_ok=True))
    info = retry('Repository inspection', lambda: api.repo_info(target))
    if info.private:
        raise ValueError('Existing repository is private')
    print(f'Uploading {len(manifest)} files to {target}', flush=True)
    retry('Upload', lambda: api.upload_folder(repo_id=target, folder_path=folder,
        allow_patterns=list(manifest), commit_message='Publish original OpenNWM Nav15 checkpoints and configs'))
    def verify():
        info = api.repo_info(target)
        revision = info.sha
        files = {f.path: f for f in api.list_repo_tree(target, recursive=True, revision=revision)
                 if isinstance(f, RepoFile)}
        for name, row in manifest.items():
            f = files.get(name)
            if f is None or f.size != row['bytes']:
                raise ValueError(f'Remote size mismatch: {name}')
            actual = f.lfs.sha256 if f.lfs else digest(Path(hf_hub_download(
                target, name, revision=revision, cache_dir=release / 'hf-verification-cache')))
            if actual != row['sha256']:
                raise ValueError(f'Remote checksum mismatch: {name}')
        write_json(release / 'hf-verification.json', {
            'repo_id': target, 'revision': revision, 'ok': True,
            'verified_files': len(manifest), 'bytes': sum(x['bytes'] for x in manifest.values())})
        print(f'Verified all {len(manifest)} files by size and SHA256: https://huggingface.co/{target}', flush=True)
    retry('Verification', verify)


if __name__ == '__main__':
    main()
