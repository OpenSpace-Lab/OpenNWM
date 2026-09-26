"""Reference checks for the original Meta NWM checkout, separate from CompACT."""
import ast
from collections import OrderedDict
import copy
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from compare_original import gradients, initial_state, seed, snapshot, state


def original_step():
    """Execute the original main() training block without its GPU/DDP launcher."""
    tree = ast.parse(Path('train.py').read_text())
    update = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'update_ema')
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    loop = next(node for node in ast.walk(main) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Tuple)
                and [item.id for item in node.target.elts] == ['x', 'y', 'rel_t'])
    body = []
    for node in loop.body:
        body.append(node)
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == 'update_ema'):
            break
    signature = ast.parse('def step(model, ema, diffusion, tokenizer, batch, device, opt, scaler, config):\n pass').body[0]
    signature.body = ast.parse('x, y, rel_t = batch\nbfloat_enable = False\nnum_cond = 2').body + body
    signature.body += ast.parse('return loss.detach().item()').body
    module = ast.fix_missing_locations(ast.Module(body=[update, signature], type_ignores=[]))
    namespace = {'torch': torch, 'OrderedDict': OrderedDict}
    exec(compile(module, 'original_train_block', 'exec'), namespace)
    return namespace['step']


class _Posterior:
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value + torch.randn_like(self.value) * 0.1


class _OriginalVAE:
    def encode(self, value):
        return SimpleNamespace(latent_dist=_Posterior(value))

    def decode(self, value):
        self.last_decoded = value.clone()
        return SimpleNamespace(sample=value)


class _WrappedVAE(_OriginalVAE):
    scaling_factor = 0.18215

    def encode(self, value):
        return super().encode(value).latent_dist.sample().mul_(self.scaling_factor)

    def decode(self, value, denormalize=False):
        value = super().decode(value / self.scaling_factor).sample
        return (value * 0.5 + 0.5).clamp(0, 1) if denormalize else value


class _ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def dataset_records(candidate, artifact):
    from datasets import TrainingDataset, EvalDataset
    import misc

    directory = artifact / ('candidate_input' if candidate else 'original_input')
    data, splits = directory / 'data', directory / 'splits'
    (data / 'trajectory').mkdir(parents=True, exist_ok=True)
    splits.mkdir(parents=True, exist_ok=True)
    (splits / 'traj_names.txt').write_text('trajectory\n')
    frames = 20
    positions = np.stack((np.arange(frames) * 0.25, np.sin(np.arange(frames) / 5)), axis=1)
    with (data / 'trajectory/traj_data.pkl').open('wb') as handle:
        pickle.dump({'position': positions, 'yaw': np.arange(frames) * 0.07}, handle)
    for frame in range(frames):
        image = (np.arange(12 * 16 * 3).reshape(12, 16, 3) + frame * 7).astype(np.uint8)
        Image.fromarray(image).save(data / 'trajectory' / f'{frame}.jpg')
    if candidate:
        # Hydra's original cwd is the current project root outside a Hydra run.
        import datasets
        datasets.get_original_cwd = lambda: str(Path.cwd())
    kwargs = dict(data_folder=str(data), data_split_folder=str(splits), dataset_name='recon',
                  image_size=224, min_dist_cat=-8, max_dist_cat=8, len_traj_pred=4,
                  traj_stride=1, context_size=2, goals_per_obs=2,
                  traj_names='traj_names.txt',
                  transform=misc.get_transform() if candidate else misc.transform)
    if candidate:
        kwargs.update(action_stats={'min': [-2.5, -4], 'max': [5, 4]}, waypoint_spacing=0.25)
    train, evaluation = TrainingDataset(**kwargs), EvalDataset(**kwargs)
    seed(700)
    return {'train': snapshot([train[index] for index in (0, 5, 11)]),
            'evaluation': snapshot([evaluation[index] for index in (0, 5, 11)])}


def original_nwm_cases(reference, artifact):
    from models import CDiT
    from isolated_nwm_infer import model_forward_wrapper
    candidate = reference is not None
    if candidate:
        from diffusion import create_gaussian_diffusion as create_diffusion
        from train_utils import train_step
        from omegaconf import OmegaConf
    else:
        from diffusion import create_diffusion
        train_step = original_step()
    seed(101)
    model = CDiT(input_size=2, context_size=2, patch_size=1, in_channels=4,
                 hidden_size=12, depth=1, num_heads=2, mlp_ratio=1, learn_sigma=True)
    model.original_nwm = True
    generator = torch.Generator().manual_seed(991)
    with torch.no_grad():
        for layer in [model.blocks[0].adaLN_modulation[-1], model.final_layer.adaLN_modulation[-1], model.final_layer.linear]:
            for parameter in layer.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.04)
    initialized = initial_state(model, None if reference is None else reference['legacy_nwm'])
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0)
    diffusion = create_diffusion('', diffusion_steps=12, noise_schedule='squaredcos_cap_v2')
    tokenizer = _WrappedVAE() if candidate else _OriginalVAE()
    seed(202)
    batch = (torch.randn(2, 4, 4, 2, 2), torch.randn(2, 2, 3),
             torch.tensor([[-0.0625, 0.5], [0.0078125, -0.5]]))
    records = []
    for step in range(2):
        forwards = []
        handle = model.register_forward_hook(lambda _module, _inputs, output: forwards.append(snapshot(output)))
        seed(300 + step)
        if candidate:
            config = OmegaConf.create({'dataset': {'context_size': 2}, 'training': {'grad_clip_val': 10}})
            loss = train_step(_ModelWrapper(model), ema, diffusion, tokenizer, batch, torch.device('cpu'),
                              optimizer, None, False, None, config)
        else:
            loss = train_step(_ModelWrapper(model), ema, diffusion, tokenizer, batch, torch.device('cpu'),
                              optimizer, None, {'grad_clip_val': 10})
        handle.remove()
        records.append({'loss': loss, 'forwards': forwards, 'gradients': gradients(model),
                        'parameters': state(model), 'ema': state(ema), 'optimizer': snapshot(optimizer.state_dict())})
    model.eval()
    seed(407)
    pixels = model_forward_wrapper((model, diffusion, tokenizer), batch[0][:, :2], batch[1],
                                   num_timesteps=8, latent_size=2, device=torch.device('cpu'),
                                   num_cond=2, num_goals=2, rel_t=batch[2].flatten())
    if not candidate:
        pixels = pixels * 0.5 + 0.5
    return {'legacy_nwm': {'initial': initialized, 'batch': snapshot(batch), 'steps': records,
                           'eval_pixels': snapshot(pixels), 'eval_decoded': snapshot(tokenizer.last_decoded),
                           'dataset': dataset_records(candidate, artifact)}}
