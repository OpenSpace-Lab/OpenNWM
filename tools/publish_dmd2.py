#!/usr/bin/env python3
"""Package DMD2 inference weights and publish verified ModelScope releases."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('/file_system/nas/algorithm/dujun.nie/nwm/dmd2_4step_20261019')
RELEASE = Path('/file_system/nas/algorithm/dujun.nie/nwm/releases/DMD2-20261019')
MODELS = {'opennwm': 'OpenNWM-DMD', 'nwm': 'NWM-DMD'}


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            sha.update(block)
    return sha.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def card(name, results):
    rows = []
    for method in ('teacher_250_ddpm', 'teacher_4_ddpm', 'teacher_4_dmd2', 'student_4_dmd2'):
        r = results['summary'][method]
        rows.append(f"| {method} | {r['psnr']:.3f} | {r['ssim']:.4f} | {r['lpips']:.4f} | {r['denoise_ms']:.2f} | {r['end_to_end_ms']:.2f} |")
    return f'''---
license: cc-by-nc-4.0
language:
- en
- zh
tags:
- navigation
- world-model
- DMD2
frameworks:
- pytorch
---

# {name}

Four-step DMD2 distillation pilot of {'OpenNWM Nav15 Stage 2' if name == 'OpenNWM-DMD' else 'the official NWM CDiT-B checkpoint'}.
Code and dependencies: [OpenNWM](https://github.com/B0B8K1ng/OpenNWM).
Method: [DMD2](https://github.com/tianweiy/DMD2/tree/8d8fa55633d47cfb81bbc7a892e7248f9518763f).

**This is a short training pilot, not a converged or quality-preserving release.**
Each student received 200 generator updates and 2,000 critic updates with batch 2
on 384 training windows. DMD2 is adapted to conditional CDiT with a pooled-token
GAN head, normalized distribution-matching gradients, a 10:1 critic/student ratio,
and backward simulation of the student's preceding steps. No text CFG is used.

## Files and compatibility

- `student.pt`: inference weights under `ema` (final student, not a temporal EMA),
  plus public provenance. All tensor names, values, shapes and dtypes match the
  trained local checkpoint exactly; machine-specific metadata has been removed.
- `config.yaml`: portable inference configuration; the external VAE is
  `stabilityai/sd-vae-ft-ema`, overridable with `NWM_VAE_PATH`.
- `dmd2.py`: the exact four-step sampler used in the experiment.
- `results.json`, `samples.jsonl`, `comparison.png`: aggregate/individual test
  results and six comparison rows (two per dataset).
- `SHA256SUMS`, `manifest.json`: release integrity and provenance.

The weights require **four steps `[999, 749, 499, 249]` with x0 prediction and
fresh noise between steps**. Setting the old DDPM sampler to four steps is not
equivalent. No teacher, critic, optimizer, VAE weights or dataset is bundled.
This release does not replace the original teacher repository.

## Download and inference

```python
from modelscope import snapshot_download
snapshot_download('LittleBoss/{name}', local_dir='./weights/{name}')
```

Install the dependencies of the OpenNWM code repository and run from that
repository's root. The example below consumes `conditioning.pt`, a local
dictionary containing:

- `context`: float32 `[B, 4, 3, 224, 224]` images in `[-1, 1]`, with the original
  NWM resize/crop preprocessing;
- `action`: float32 `[B, 3]` endpoint motion `(x, y, yaw)`, in the local frame and
  waypoint-spacing units produced by `BaseDataset._compute_actions`, **before**
  the legacy NWM min/max normalization;
- `rel_t`: float32 `[B]`, future frame offset divided by 128.

```python
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf

weights = Path('./weights/{name}')
sys.path.insert(0, str(Path('nwm').resolve()))
sys.path.insert(0, str(weights.resolve()))  # use this release's sampler
from train_utils import setup_model, setup_tokenizer, setup_diffusion
from motion_condition import make_motion_group
from dmd2 import FourStepSchedule

config = OmegaConf.load(weights / 'config.yaml')
model = setup_model(config, torch.device('cpu'))
model.load_state_dict(torch.load(weights / 'student.pt', map_location='cpu',
                                 weights_only=True)['ema'], strict=True)
model = model.cuda().eval().requires_grad_(False)
vae = setup_tokenizer(config, torch.device('cuda')).eval().requires_grad_(False)
diffusion = setup_diffusion(config, False, torch.device('cuda'))
schedule = FourStepSchedule(diffusion.alphas_cumprod).cuda()
inputs = torch.load('conditioning.pt', map_location='cuda', weights_only=True)
context, action = inputs['context'], inputs['action'].clone()
torch.manual_seed(20261019)
with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
    latents = vae.vae.encode(context.flatten(0, 1)).latent_dist.mode() * vae.scaling_factor
    condition = dict(x_cond=latents.unflatten(0, context.shape[:2]).float(),
                     rel_t=inputs['rel_t'])
    if model.motion_condition_enabled:
        condition['motion'] = make_motion_group('real', action)
    else:
        low = action.new_tensor(config.dataset.action_stats.min)
        high = action.new_tensor(config.dataset.action_stats.max)
        action[:, :2] = 2 * (action[:, :2] - low) / (high - low) - 1
        condition['y'] = action
    noise = torch.randn(context.shape[0], 4, 28, 28, device='cuda')
    predicted = schedule.sample(model, noise, condition)
    images = vae.decode(predicted, denormalize=True).float().cpu()
torch.save(images, 'predictions.pt')  # [B, 3, 224, 224], values in [0, 1]
```

## Small-sample results

24 held-out windows: 8 each from recon, sacson and scand. Train/validation/test
trajectories are disjoint. Future horizons are 4/16 frames. All methods use the
same inputs, actions and initial noise; VAE posterior means are deterministic.
NVIDIA L20, batch 1, BF16 autocast; one warmup per method, randomized method
order, CUDA synchronization, three repetitions (median per sample, then mean).
End-to-end includes online VAE encoding, denoising and decoding, excludes I/O,
model loading and metrics. No FID, rollout or navigation-success evaluation.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Denoising ms ↓ | Including VAE ms ↓ |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Measured student speedup including VAE: **{results['speedup']['end_to_end_ms']:.2f}×**.
PSNR/SSIM increase while LPIPS worsens relative to the 250-step teacher;
outputs are smoother and lose fine detail. The student improves LPIPS over
the untrained four-step sampler but does not retain the teacher's perceptual
quality. The small test set is not a full benchmark or a convergence claim.
Paired bootstrap 95% intervals and per-dataset summaries are in `results.json`.

![Comparison](comparison.png)

## License and provenance

NWM-derived model weights/code retain CC BY-NC 4.0 terms; see
`licenses/NWM-CC-BY-NC-4.0.txt`. The sampler is a local CDiT adaptation of the
referenced DMD2 method, not upstream SDXL weights. External VAE and datasets
retain their respective terms. Teacher/source hashes and the original local
student checksum are recorded in `manifest.json`.

## 中文说明

这是 200 次学生更新的小规模四步蒸馏权重，尚未收敛，不能视为无损加速。
实测速度明显提高，但纹理更模糊、LPIPS 变差。必须使用附带的四步采样器；
不能仅把原 DDPM 推理步数设为 4。权重张量保持原样，公开包已移除本机路径。
'''


def stage():
    import torch
    import yaml
    torch.set_num_threads(4)
    RELEASE.mkdir(parents=True, exist_ok=True)
    print(f'Free disk bytes: {shutil.disk_usage(RELEASE).free}', flush=True)
    for key, name in MODELS.items():
        folder = RELEASE / name
        folder.mkdir(exist_ok=True)
        original = SOURCE / key / 'student.pt'
        saved = torch.load(original, map_location='cpu', weights_only=False)
        meta = saved['metadata']
        config = copy.deepcopy(meta['config'])
        config = {k: config[k] for k in ('model', 'dataset', 'motion_condition',
                  'training_stage', 'action_mode', 'proxy', 'finetune', 'original_nwm') if k in config}
        config['dataset'] = {k: config['dataset'][k] for k in
            ('image_size', 'context_size', 'mean', 'std', 'normalize', 'action_stats')}
        config['model']['generator']['input_size'] = 28
        config['model']['diffusion']['eval_timestep_respacing'] = 250
        config['model']['tokenizer']['model_path'] = '${oc.env:NWM_VAE_PATH,stabilityai/sd-vae-ft-ema}'
        if 'finetune' in config:
            config['finetune']['stage1_checkpoint'] = None
        portable = yaml.safe_dump(config, sort_keys=False)
        if '/file_system/' in portable or '/home/' in portable:
            raise ValueError('Private path remains in public configuration')
        (folder / 'config.yaml').write_text(portable)
        public_meta = {k: meta[k] for k in ('method', 'timesteps', 'teacher_sha256',
                      'data_sha256', 'adaptation', 'reference')}
        public_meta['generator_updates'] = saved['generator_updates']
        public_meta['pilot'] = True
        torch.save(dict(ema=saved['ema'], metadata=public_meta,
                        generator_updates=saved['generator_updates']), folder / 'student.pt')
        reread = torch.load(folder / 'student.pt', map_location='cpu', weights_only=True)
        assert saved['ema'].keys() == reread['ema'].keys()
        assert all(torch.equal(value, reread['ema'][k]) and value.dtype == reread['ema'][k].dtype
                   for k, value in saved['ema'].items())
        del reread, saved
        results = json.loads((SOURCE / (key + '-test') / 'results.json').read_text())
        results.pop('training_metadata')
        results['student_sha256'] = digest(folder / 'student.pt')
        results['source_student_sha256'] = digest(original)
        write_json(folder / 'results.json', results)
        for asset in ('samples.jsonl', 'comparison.png'):
            shutil.copyfile(SOURCE / (key + '-test') / asset, folder / asset)
        shutil.copyfile(REPO / 'nwm/dmd2.py', folder / 'dmd2.py')
        (folder / 'licenses').mkdir(exist_ok=True)
        shutil.copyfile(REPO / 'licenses/NWM-CC-BY-NC-4.0.txt', folder / 'licenses/NWM-CC-BY-NC-4.0.txt')
        (folder / 'README.md').write_text(card(name, results))
        write_json(folder / 'configuration.json', {'framework': 'pytorch', 'task': 'image-to-image'})
        manifest = dict(repo_id='LittleBoss/' + name, source_student_sha256=digest(original),
                        tensor_equality_verified=True, metadata=public_meta)
        write_json(folder / 'manifest.json', manifest)
        files = {}
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or '.cache' in path.parts or path.name == 'SHA256SUMS':
                continue
            if path.suffix in ('.json', '.jsonl', '.yaml', '.md', '.py', '.txt'):
                content = path.read_text()
                if '/file_system/' in content or '/home/' in content:
                    raise ValueError(f'Private path remains in {path.name}')
            files[str(path.relative_to(folder))] = dict(bytes=path.stat().st_size, sha256=digest(path))
        sums = folder / 'SHA256SUMS'
        sums.write_text(''.join(f"{r['sha256']}  {p}\n" for p, r in files.items()))
        files['SHA256SUMS'] = dict(bytes=sums.stat().st_size, sha256=digest(sums))
        write_json(RELEASE / (name + '-files.json'), files)
        print(f'Staged {name}: {len(files)} files; exact tensor equality verified', flush=True)


def validate():
    import sys
    import re
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(REPO / 'nwm'))
    from train_utils import setup_model, setup_diffusion
    torch.set_num_threads(4)
    for name in MODELS.values():
        folder = RELEASE / name
        config = OmegaConf.load(folder / 'config.yaml')
        model = setup_model(config, torch.device('cpu'))
        checkpoint = torch.load(folder / 'student.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['ema'], strict=True)
        assert len(setup_diffusion(config, False, torch.device('cpu')).alphas_cumprod) == 1000
        for example in re.findall(r'```python\n(.*?)```', (folder / 'README.md').read_text(), re.S):
            compile(example, str(folder / 'README.md'), 'exec')
        print(f'VALIDATED {name}: portable config, strict weights, 1000-step base schedule, example syntax', flush=True)
        del model, checkpoint


def upload():
    # Explicitly bypass environment proxies for both API and object storage.
    for key in list(os.environ):
        if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy'):
            os.environ.pop(key)
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = '*'
    from modelscope_hub import HubApi
    api = HubApi()
    user = api.whoami()
    if user.username != 'LittleBoss':
        raise ValueError('Authenticated account differs from intended namespace')
    for name in MODELS.values():
        repo_id = 'LittleBoss/' + name
        files = json.loads((RELEASE / (name + '-files.json')).read_text())
        folder = RELEASE / name
        for path, expected in files.items():
            if (folder / path).stat().st_size != expected['bytes'] or digest(folder / path) != expected['sha256']:
                raise ValueError(f'Staged file changed: {path}')
        if not api.repo_exists(repo_id, 'model'):
            api.create_repo(repo_id, 'model', visibility='public', license='cc-by-nc-4.0',
                            description=name + ': four-step DMD2 distillation pilot, weights and measured quality/speed')
        print(f'Uploading directly: {repo_id}', flush=True)
        started = time.monotonic()
        api.upload_folder(repo_id, 'model', folder, max_workers=4, allow_patterns=list(files),
                          use_cache=True, disable_tqdm=True,
                          commit_message='Release four-step DMD2 pilot weights, sampler, config and measured results')
        remote = {p.path: p for p in api.list_repo_files(repo_id, 'model') if not p.is_dir}
        for path, expected in files.items():
            item = remote.get(path)
            if item is None or item.size != expected['bytes']:
                raise RuntimeError(f'Remote size mismatch: {path}')
            sha = getattr(item, 'sha256', None)
            # Current ModelScope FileInfo exposes the content SHA256 as blob_id.
            blob_id = getattr(item, 'blob_id', None)
            if not sha and isinstance(blob_id, str) and len(blob_id) == 64:
                sha = blob_id
            if not sha and item.lfs:
                sha = item.lfs.get('oid') or item.lfs.get('sha256')
                if sha:
                    sha = sha.removeprefix('sha256:')
            if not sha:
                if expected['bytes'] > 32 << 20:
                    raise RuntimeError(f'Missing remote large-file hash: {path}')
                sha = digest(api.download_file(repo_id, 'model', path))
            if sha != expected['sha256']:
                raise RuntimeError(f'Remote hash mismatch: {path}')
        write_json(RELEASE / (name + '-verification.json'), dict(repo_id=repo_id,
                   url=f'https://modelscope.cn/models/{repo_id}', files=files,
                   elapsed_seconds=time.monotonic()-started, direct=True, verified=True))
        print(f'VERIFIED {repo_id}: {len(files)} sizes/SHA256 hashes match; {time.monotonic()-started:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['stage', 'validate', 'upload'])
    args = parser.parse_args()
    print(f'PID={os.getpid()} cwd={Path.cwd()} command={args.command}', flush=True)
    {'stage': stage, 'validate': validate, 'upload': upload}[args.command]()
