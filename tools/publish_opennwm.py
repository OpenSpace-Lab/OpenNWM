#!/usr/bin/env python3
"""Stage, publish and verify the unchanged OpenNWM Nav15 paper checkpoints.

Run with an environment containing PyYAML and modelscope_hub (for upload).
Credentials are read from MODELSCOPE_API_TOKEN or the ModelScope login cache.
No credentials or local source paths are written to the public payload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path('/file_system/nas/algorithm/dujun.nie/nwm')
DEFAULT_RELEASE = DEFAULT_SOURCE / 'releases/OpenNWM-20261016'
ASSETS = {
    'opennwm-nav15-stage2-joint100k.pth.tar': {
        'source': 'compact/runs/nav15_pa_step60000_latentpt_reset_ft/nwm-latentpt-pixel-action-finalLAM-ft-reset/checkpoints/joint_0100000.pth.tar',
        'bytes': 3137820686,
        'role': 'Stage 2 real-action navigation; 3,000 adapter warmup + 100,000 joint steps',
        'state_dict': 'ema',
        'config': 'configs/opennwm-nav15-stage2.yaml',
    },
    'opennwm-nav15-stage1-pretrain60k.pth.tar': {
        'source': 'compact/runs/nav15_pa_step60000_latentpt/nwm-latentpt-pixel-action-finalLAM/checkpoints/pretrain_0060000.pth.tar',
        'bytes': 3137820971,
        'role': 'Stage 1 latent-action pretraining; 60,000 steps',
        'state_dict': 'ema',
        'config': 'configs/opennwm-nav15-stage1.yaml',
    },
    'lam-nav15-pixel-action-step60k.ckpt': {
        'source': 'weights/navigation_lam/variant_4_pixel_action/nav15-pixel-action/checkpoints/step=60000.ckpt',
        'bytes': 8518856077,
        'role': 'Nav15 PixelActionLAM; 60,000 steps; 32-dimensional latent actions',
        'state_dict': 'state_dict',
    },
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path, progress=False):
    sha = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        while block := stream.read(16 * 1024 * 1024):
            sha.update(block)
            count += len(block)
            if progress and count % (512 * 1024 * 1024) == 0:
                print(f'Hashing {path.name}: {count:,}/{path.stat().st_size:,} bytes', flush=True)
    return sha.hexdigest()


def portable_config(source, stage):
    import yaml
    config = yaml.safe_load(source.read_text())
    config['model']['tokenizer']['model_path'] = '${oc.env:NWM_VAE_PATH,stabilityai/sd-vae-ft-ema}'
    config['training']['results_dir'] = '${oc.env:OPENNWM_WORK_ROOT,./runs}'
    config['training']['run_name'] = f'opennwm-nav15-stage{stage}'
    config['training']['wandb_enabled'] = False
    config['training']['from_checkpoint'] = None
    config['dataset']['precomputed_latents']['enabled'] = False
    config['dataset']['precomputed_latents']['root'] = '${oc.env:NWM_VAE_LATENT_ROOT,null}'
    config['finetune']['stage1_checkpoint'] = None
    config['resume_from_dir'] = None
    if stage == 1:
        config['dataset']['sampling_recipe']['path'] = '${oc.env:NWM_NAVANYWHERE_RECIPE}'
    else:
        for name, entry in config['dataset']['datasets'].items():
            for split in ('train', 'test'):
                if split in entry:
                    entry[split] = '${oc.env:NWM_SPLIT_ROOT}/' + name + '/' + split + '/'
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    # Every private training path must be explicitly handled above, never silently leaked.
    if '/file_system/' in rendered or '/home/' in rendered:
        raise ValueError(f'Unmapped private path in config {source.name}')
    return rendered


def model_card(repo_id, rows):
    table = '\n'.join(f"| `{row['path']}` | {row['role']} | {row['bytes']:,} |" for row in rows)
    return f'''---
license: cc-by-nc-4.0
language:
- en
tags:
- OpenNWM
- navigation
- world-model
- latent-action-model
- NavAnywhere
frameworks:
- pytorch
---

# OpenNWM

The Nav15 Pixel+Action release of OpenNWM: a latent action model (LAM), a
Stage 1 navigation world model pretrained with latent actions, and a Stage 2
world model adapted to real navigation actions.

- Code and setup: [OpenSpace-Lab/OpenNWM](https://github.com/OpenSpace-Lab/OpenNWM)
- Repository: [{repo_id}](https://modelscope.cn/models/{repo_id})
- Training data: [NavAnywhere](https://modelscope.cn/datasets/LittleBoss/NavAnywhere)

## Checkpoints

| File | Use | Bytes |
| --- | --- | ---: |
{table}

Use **Stage 2** for real-action prediction and planning. Stage 1 uses latent
actions and has not been adapted for direct real-action navigation. The LAM
maps image pairs to 32-dimensional latent actions. These are the final Nav15
checkpoints, distinct from the older Nav1 weights in `LittleBoss/LAM-weights`.

Files preserve their original checkpoint format and bytes, including training
state and historical configuration metadata. No optimizer tensors were removed
and no weight conversion was performed. Use the separate portable YAML configs
below instead of relying on historical machine paths embedded in checkpoints.
All payload files are covered by `SHA256SUMS`; `manifest.json` records checkpoint
roles, byte sizes and hashes.

## Model and inference settings

The world models use CDiT-B/2 with four 224×224 context frames and
`stabilityai/sd-vae-ft-ema` (latent scaling 0.18215). The VAE is an external
dependency and is downloaded separately. Stage 2 uses the checkpoint's `ema`
state, three real action dimensions, and 250 DDPM sampling steps. The real-action
normalization bounds are [-2.5, -4, -π] to [5, 4, π]. LAM images use a 4:3 center
crop followed by resize to 240×320; use the repository preprocessing code.

## Download and use

```python
from modelscope import snapshot_download
snapshot_download("{repo_id}", local_dir="./weights/OpenNWM")
```

Install OpenNWM and configure its data and runtime paths following the code
repository README. From the OpenNWM source repository, import Stage 2 into a
new inference directory with its portable config:

```bash
python run.py nwm import-checkpoint \\
  --checkpoint /path/to/weights/OpenNWM/opennwm-nav15-stage2-joint100k.pth.tar \\
  --config /path/to/weights/OpenNWM/configs/opennwm-nav15-stage2.yaml \\
  --run-dir /path/to/runs/opennwm-nav15-stage2
python run.py nwm infer exp_dir=/path/to/runs/opennwm-nav15-stage2 ckp=latest eval_type=time
```

`NWM_VAE_PATH` can point to a local VAE or use the default public identifier.
Dataset evaluation also requires `NWM_DATA_ROOT` and `NWM_SPLIT_ROOT`; use the
original split lists for historical score reproduction. The portable configs
are for inference and do not claim exact training resumption. Refer to the
repository NWM README for prediction, evaluation and planning, and the LAM
README for using `lam-nav15-pixel-action-step60k.ckpt` with LAM evaluation or
latent extraction. Stage 1's config additionally uses
`NWM_NAVANYWHERE_ROOT`, `NWM_NAVANYWHERE_RECIPE` and `NWM_LATENT_PROXY_ROOT`.

## Provenance and licenses

NWM code derives from Meta NWM and CompACT; its upstream **CC BY-NC 4.0**
license is preserved in `licenses/NWM-CC-BY-NC-4.0.txt`. DreamDojo LAM code's
**Apache 2.0** license is preserved in `licenses/DreamDojo-Apache-2.0.txt`.
The repository code provenance pins DreamDojo `8f3eeb7`, CompACT `8367a5e`,
and Meta NWM `3f6cd8e`. This release retains both upstream notices and does
not relicense upstream material. External VAE and data sources retain their
respective terms.

The weights support research on visual navigation and future observation
prediction. They are not a complete robot controller; predictions can drift
outside the training distribution. No new benchmark scores or deployment
claims are introduced by this release.

'''


def stage(args):
    release = args.release_dir.expanduser().absolute()
    release.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(release)
    print(f'Free disk: {disk.free:,} bytes; checkpoint staging uses symlinks', flush=True)
    payload = release / 'payload'
    payload.mkdir(exist_ok=True)
    cache_path = release / 'local-provenance.json'
    previous = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    provenance = {}
    rows = []
    for name, spec in ASSETS.items():
        source = (args.source_root / spec['source']).resolve(strict=True)
        stat = source.stat()
        if stat.st_size != spec['bytes']:
            raise ValueError(f'Unexpected checkpoint size: {source}')
        target = payload / name
        if target.is_symlink():
            if target.resolve() != source:
                raise ValueError(f'Existing staging symlink points elsewhere: {target}')
        elif target.exists():
            raise ValueError(f'Refusing to overwrite existing file: {target}')
        else:
            target.symlink_to(source)
        identity = {'source': str(source), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        cached = previous.get(name, {})
        sha = cached.get('sha256') if all(cached.get(k) == v for k, v in identity.items()) else None
        if not sha or args.rehash:
            sha = digest(source, progress=True)
        if source.stat().st_mtime_ns != stat.st_mtime_ns or source.stat().st_size != stat.st_size:
            raise RuntimeError(f'Checkpoint changed while hashing: {source}')
        provenance[name] = dict(identity, sha256=sha)
        write_json(cache_path, {**previous, **provenance})
        row = {k: v for k, v in spec.items() if k != 'source'}
        rows.append(dict(path=name, sha256=sha, **row))
        print(f'Checkpoint verified: {name} {sha}', flush=True)
        if 'config' in spec:
            stage_number = 2 if 'stage2' in name else 1
            config_source = source.parent.parent / '.hydra/config.yaml'
            target_config = payload / spec['config']
            target_config.parent.mkdir(exist_ok=True)
            target_config.write_text(portable_config(config_source, stage_number))
    for source in (REPO_ROOT / 'licenses').glob('*.txt'):
        destination = payload / 'licenses' / source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
    (payload / 'README.md').write_text(model_card(args.repo_id, rows))
    write_json(payload / 'configuration.json', {'framework': 'pytorch', 'model': {'type': 'OpenNWM'}})
    commit = subprocess.check_output(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    write_json(payload / 'manifest.json', {
        'schema_version': 1, 'release': 'OpenNWM Nav15 Pixel+Action', 'repo_id': args.repo_id,
        'source_code': 'https://github.com/OpenSpace-Lab/OpenNWM', 'source_code_commit': commit,
        'checkpoint_format': 'original, unmodified training checkpoints', 'checkpoints': rows,
    })
    files = {}
    for path in sorted(payload.rglob('*')):
        if path.is_file() and '.ms_upload_cache' not in path.parts and path.name != 'SHA256SUMS':
            name = path.relative_to(payload).as_posix()
            sha = provenance[name]['sha256'] if name in provenance else digest(path)
            files[name] = {'bytes': path.stat().st_size, 'sha256': sha}
    (payload / 'SHA256SUMS').write_text(''.join(f"{value['sha256']}  {name}\n" for name, value in files.items()))
    files['SHA256SUMS'] = {'bytes': (payload / 'SHA256SUMS').stat().st_size, 'sha256': digest(payload / 'SHA256SUMS')}
    write_json(release / 'payload-files.json', files)
    print(f'Staged {len(files)} files, {sum(f["bytes"] for f in files.values()):,} bytes at {payload}', flush=True)
    return payload, files


def upload(args, payload, files):
    if args.direct:
        for key in list(os.environ):
            if key.lower() in {'http_proxy', 'https_proxy', 'all_proxy'}:
                os.environ.pop(key)
    from modelscope_hub import HubApi
    api = HubApi()
    try:
        user = api.whoami()
    except Exception:
        token = os.environ.get('MODELSCOPE_API_TOKEN')
        if not token:
            raise RuntimeError('ModelScope login unavailable. Set a valid MODELSCOPE_API_TOKEN or log in with the ModelScope SDK, then rerun --upload.') from None
        user = api.login(token)
    owner = args.repo_id.split('/')[0]
    if user.username != owner:
        raise ValueError(f'Authenticated account {user.username!r} differs from requested owner {owner!r}')
    if not api.repo_exists(args.repo_id, 'model'):
        api.create_repo(args.repo_id, 'model', visibility='public', license='cc-by-nc-4.0',
                        description='OpenNWM Nav15 Pixel+Action: LAM, Stage 1 and Stage 2 navigation world model checkpoints')
    print(f'Uploading to https://modelscope.cn/models/{args.repo_id}', flush=True)
    api.upload_folder(args.repo_id, 'model', payload, max_workers=args.workers,
                      allow_patterns=list(files), use_cache=True, disable_tqdm=True,
                      commit_message='Release original OpenNWM Nav15 checkpoints with portable configs and SHA256 manifest')
    remote = {item.path: item for item in api.list_repo_files(args.repo_id, 'model') if not item.is_dir}
    verified = []
    for name, expected in files.items():
        item = remote.get(name)
        if item is None or item.size != expected['bytes']:
            raise RuntimeError(f'Remote file missing or size mismatch: {name}')
        remote_sha = item.sha256
        if not remote_sha and item.lfs:
            remote_sha = item.lfs.get('oid') or item.lfs.get('sha256')
            if isinstance(remote_sha, str):
                remote_sha = remote_sha.removeprefix('sha256:')
        if not remote_sha:
            # Small Git files do not always expose SHA256; download and hash them.
            if expected['bytes'] > 32 * 1024 * 1024:
                raise RuntimeError(f'Remote server omitted SHA256 for large checkpoint: {name}')
            downloaded = api.download_file(args.repo_id, 'model', name)
            remote_sha = digest(Path(downloaded))
        if remote_sha != expected['sha256']:
            raise RuntimeError(f'Remote SHA256 mismatch: {name}')
        verified.append({'path': name, **expected})
    receipt = {'repo_id': args.repo_id, 'url': f'https://modelscope.cn/models/{args.repo_id}',
               'verified_at': datetime.now(timezone.utc).isoformat(), 'files': verified}
    write_json(args.release_dir / 'upload-verification.json', receipt)
    print(f'Upload verified: {len(verified)} remote file sizes and SHA256 hashes match', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--release-dir', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--repo-id', default='LittleBoss/OpenNWM')
    parser.add_argument('--upload', action='store_true', help='Publish public model repo, then verify remote size/SHA256')
    parser.add_argument('--direct', action='store_true', help='Ignore proxy environment variables for upload')
    parser.add_argument('--rehash', action='store_true', help='Recompute checkpoint hashes even if source identity is unchanged')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    payload, files = stage(args)
    if args.upload:
        upload(args, payload, files)


if __name__ == '__main__':
    main()
