#!/usr/bin/env python3
"""Make an inference run directory from a standalone NWM checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf


def import_checkpoint(checkpoint, run_dir, config_path=None):
    checkpoint, run_dir = Path(checkpoint).expanduser().resolve(), Path(run_dir).expanduser().absolute()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f'Choose a new empty run directory: {run_dir}')
    if config_path:
        config = OmegaConf.load(config_path)
    else:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        if 'config' not in payload:
            raise ValueError('Checkpoint has no saved configuration; provide --config config.yaml')
        config = OmegaConf.create(payload['config'])
        del payload
    OmegaConf.update(config, 'model.tokenizer.model_path', '${oc.env:NWM_VAE_PATH,stabilityai/sd-vae-ft-ema}', force_add=True)
    OmegaConf.update(config, 'training.results_dir', str(run_dir.parent), force_add=True)
    OmegaConf.update(config, 'training.run_name', run_dir.name, force_add=True)
    (run_dir/'.hydra').mkdir(parents=True, exist_ok=True)
    (run_dir/'checkpoints').mkdir(exist_ok=True)
    OmegaConf.save(config, run_dir/'.hydra/config.yaml', resolve=False)
    (run_dir/'checkpoints/latest.pth.tar').symlink_to(checkpoint)
    print(f'Imported {checkpoint}\nRun directory: {run_dir}\nInference: exp_dir={run_dir} ckp=latest', flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    import_checkpoint(args.checkpoint, args.run_dir, args.config)


if __name__ == '__main__':
    main()
