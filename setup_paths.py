#!/usr/bin/env python3
"""Write machine-local paths and prepare the small, fixed sampling lists."""
import argparse
import gzip
from pathlib import Path
import shlex
import shutil

from tools.make_validation_split import make_split

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--split-root', type=Path, help='Existing train/val/test lists; otherwise hold out 5%% of train.')
    parser.add_argument('--lam-python', type=Path, required=True)
    parser.add_argument('--nwm-python', type=Path, required=True)
    parser.add_argument('--nav-profile', choices=['13', '15'], default='13')
    parser.add_argument('--vae', default='stabilityai/sd-vae-ft-ema')
    parser.add_argument('--dino-repo', type=Path)
    parser.add_argument('--dino-weights', type=Path)
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    args = parser.parse_args()
    work = args.work_root.expanduser().resolve()
    data = args.data_root.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    splits = args.split_root.expanduser().resolve() if args.split_root else work / 'splits'
    if not args.split_root and not (splits / 'split.json').exists():
        make_split(ROOT / 'nwm/data_splits', splits, 0.05, 32)
    profile = 'v1_13src' if args.nav_profile == '13' else 'v2_15src'
    recipes = work / 'recipes'
    recipes.mkdir(exist_ok=True)
    recipe_paths = {}
    for split in ('train', 'val'):
        name = f'navanywhere_{profile}_{split}_seed20260901.json'
        destination = recipes / name
        if not destination.exists():
            with gzip.open(ROOT / 'resources/recipes' / (name + '.gz'), 'rb') as source:
                with destination.open('wb') as output:
                    shutil.copyfileobj(source, output)
        recipe_paths[split] = destination
    values = {
        'LAM_PYTHON': args.lam_python.expanduser().absolute(),
        'NWM_PYTHON': args.nwm_python.expanduser().absolute(),
        'NWM_DATA_ROOT': data, 'NWM_SPLIT_ROOT': splits,
        'OPENNWM_WORK_ROOT': work, 'NWM_OUTPUT_ROOT': work / 'runs/nwm',
        'NWM_VAE_PATH': str(Path(args.vae).expanduser().resolve()) if Path(args.vae).expanduser().exists() else args.vae, 'NWM_NAVANYWHERE_ROOT': data / 'NavAnywhere',
        'NWM_NAVANYWHERE_RECIPE': recipe_paths['train'],
        'NWM_NAVANYWHERE_VAL_RECIPE': recipe_paths['val'],
        'NWM_NAVANYWHERE_VAE_LATENT_ROOT': work / 'cache/nav-vae',
        'NWM_LATENT_PROXY_ROOT': work / 'cache/nav-train-lam',
        'NWM_NAVANYWHERE_VAL_PROXY_ROOT': work / 'cache/nav-val-lam',
        'NWM_FINETUNE_VAE_LATENT_ROOT': work / 'cache/nwm-vae',
        'NWM_FINETUNE_LATENT_ROOT': work / 'cache/nwm-lam',
        'LAM_ACTION_CHECKPOINT': work / 'models/lam-action-100000.ckpt',
        'LAM_PIXEL_ACTION_CHECKPOINT': work / 'models/lam-pixel-action-100000.ckpt',
        'NWM_STAGE1_CHECKPOINT': work / 'models/nwm-latentpt-200000.pth.tar',
        'DINO_REPO_PATH': args.dino_repo.expanduser().resolve() if args.dino_repo else work / 'models/dinov3',
        'DINO_WEIGHTS_PATH': args.dino_weights.expanduser().resolve() if args.dino_weights else work / 'models/dinov3_vitl16.pth',
        'HF_HOME': work / 'cache/huggingface', 'TORCH_HOME': work / 'cache/torch',
        'LAM_EVAL_ASSET_ROOT': work / 'cache/lam-eval',
        'NWM_MODEL_CACHE': work / 'cache/dreamsim', 'WANDB_MODE': 'offline',
    }
    env = args.env_file.expanduser().absolute()
    env.parent.mkdir(parents=True, exist_ok=True)
    env.write_text('# Local paths. Load with: source .env\n' + ''.join(
        f'export {key}={shlex.quote(str(value))}\n' for key, value in values.items()))
    print(f'Paths saved to {env}\nLoad with: source {shlex.quote(str(env))}')
    print(f'Splits: {splits}\nNavAnywhere recipe: {args.nav_profile} sources, seed 20260901')


if __name__ == '__main__':
    main()
