"""Recovery must preserve accumulated metrics and retained usable checkpoints."""
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import pytest
import torch
from omegaconf import OmegaConf

from checkpoint_io import atomic_save, point_checkpoint, prune_periodic, output_identity
import distributed
import isolated_nwm_eval as evaluation
from isolated_nwm_infer import model_forward_wrapper
from diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType
from import_checkpoint import import_checkpoint
from train_utils import save_checkpoint, load_checkpoint


class FakeFID:
    def __init__(self):
        self.device = torch.device('cpu')
        self.values = {'real': torch.tensor(0.), 'fake': torch.tensor(0.)}

    def update(self, images, is_real):
        self.values['real' if is_real else 'fake'] += images.sum()

    def compute(self):
        return (self.values['real'] - self.values['fake']).abs()

    def state_dict(self):
        return {key: value.clone() for key, value in self.values.items()}

    def load_state_dict(self, values):
        self.values = values

    def to(self, device):
        self.device = device
        return self


def test_dreamsim_preserves_torchvision_cache_directory(tmp_path, monkeypatch):
    original = torch.hub.get_dir()
    def dreamsim(**kwargs):
        torch.hub.set_dir(str(tmp_path/'dreamsim'))
        return object(), object()
    monkeypatch.setattr(evaluation, 'dreamsim', dreamsim)
    monkeypatch.setattr(evaluation, 'get_original_cwd', lambda: str(tmp_path))
    evaluation.get_loss_fn('dreamsim', [1], 'cpu')
    assert torch.hub.get_dir() == original


@pytest.mark.parametrize('original_nwm', [False, True])
def test_inference_preserves_unbounded_vae_latents(original_nwm):
    class FixedCleanLatent(torch.nn.Module):
        def forward(self, x, t, **kwargs):
            return torch.full_like(x, 2.0)

    model = FixedCleanLatent()
    model.original_nwm = original_nwm
    diffusion = GaussianDiffusion(
        betas=np.array([0.1, 0.2]), model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL, loss_type=LossType.MSE,
    )
    # Both original projects diffuse VAE latents, whose range exceeds [-1, 1].
    prediction = model_forward_wrapper(
        (model, diffusion, None), torch.zeros(1, 4, 4, 2, 2), None,
        num_timesteps=1, latent_size=2, device='cpu', num_cond=4,
        skip_tokenizer=True,
    )
    torch.testing.assert_close(prediction, torch.full_like(prediction, 2.0))


def test_metrics_resume_matches_uninterrupted_batches(tmp_path):
    gt, pred = tmp_path/'gt', tmp_path/'pred'
    for episode in range(5):
        for root, multiplier in ((gt, 4), (pred, 2)):
            directory = root/f'episode_{episode}'
            directory.mkdir(parents=True)
            for sec in (1, 2):
                Image.fromarray(np.full((2, 2, 3), (episode+sec)*multiplier, np.uint8)).save(directory/f'{sec}.png')
    def loss(left, right):
        return torch.tensor(np.mean([np.asarray(Image.open(a), dtype=float).mean()-np.asarray(Image.open(b), dtype=float).mean() for a, b in zip(left, right)]))
    secs = np.array([1, 2])
    reference = distributed.MetricLogger()
    config = OmegaConf.create({'batch_size': 2, 'exp_dir': str(tmp_path/'reference'), 'resume': True})
    Path(config.exp_dir).mkdir()
    evaluation.evaluate(config, 'fixture', 'time', reference, (loss, loss, {s: FakeFID() for s in secs}), str(gt), str(pred), secs)
    config.exp_dir = str(tmp_path/'resumed')
    Path(config.exp_dir).mkdir()
    calls = 0
    def interrupted_loss(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('simulated interruption')
        return loss(*args)
    with pytest.raises(RuntimeError, match='interruption'):
        evaluation.evaluate(config, 'fixture', 'time', distributed.MetricLogger(), (loss, interrupted_loss, {s: FakeFID() for s in secs}), str(gt), str(pred), secs)
    recovered = distributed.MetricLogger()
    evaluation.evaluate(config, 'fixture', 'time', recovered, (loss, loss, {s: FakeFID() for s in secs}), str(gt), str(pred), secs)
    assert {key: value.global_avg for key, value in recovered.meters.items()} == {key: value.global_avg for key, value in reference.meters.items()}
    config.batch_size = 1
    with pytest.raises(ValueError, match='arguments differ'):
        evaluation.evaluate(config, 'fixture', 'time', distributed.MetricLogger(), (loss, loss, {s: FakeFID() for s in secs}), str(gt), str(pred), secs)


def test_retention_protects_best_latest_and_transition(tmp_path):
    for step in range(6):
        atomic_save({'step': step}, tmp_path/f'pretrain_{step:07d}.pth.tar')
    atomic_save({}, tmp_path/'transition.pth.tar')
    point_checkpoint(tmp_path/'pretrain_0000000.pth.tar', tmp_path/'best.pth.tar')
    point_checkpoint(tmp_path/'pretrain_0000005.pth.tar', tmp_path/'latest.pth.tar')
    prune_periodic(tmp_path, keep=2)
    assert {p.name for p in tmp_path.glob('pretrain_*.pth.tar')} == {'pretrain_0000000.pth.tar', 'pretrain_0000004.pth.tar', 'pretrain_0000005.pth.tar'}
    assert (tmp_path/'transition.pth.tar').is_file()
    assert (tmp_path/'best.pth.tar').is_file()
    assert (tmp_path/'latest.pth.tar').is_file()


def test_baseline_checkpoint_keeps_cursor_optimizer_and_best(tmp_path):
    config = OmegaConf.create({'training': {'from_checkpoint': None, 'keep_periodic': 3}})
    model, ema = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    opt.step()
    state = {'batch_in_epoch': 7, 'rng_state': {'torch': torch.get_rng_state()}, 'epoch_start_rng': {'example': 2}}
    save_checkpoint(model, ema, opt, config, 3, 19, False, None, None, tmp_path, runtime_states=[state], best_score=0.125)
    restored, restored_ema = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    restored_opt = torch.optim.AdamW(restored.parameters())
    with mock.patch('train_utils.dist.get_world_size', return_value=1), mock.patch('train_utils.dist.get_rank', return_value=0):
        epoch, step, runtime = load_checkpoint(tmp_path, config, restored, restored_ema, restored_opt, return_runtime=True)
    assert (epoch, step, runtime['batch_in_epoch'], runtime['best_score']) == (3, 19, 7, 0.125)
    torch.testing.assert_close(restored.weight, model.weight)
    assert int(next(iter(restored_opt.state.values()))['step']) == 1


def test_output_cache_rejects_changed_checkpoint(tmp_path):
    weight = tmp_path/'weight.pth'
    atomic_save({'v': 1}, weight)
    config = OmegaConf.create({'seed': 42})
    output_identity(tmp_path, config, checkpoint=weight)
    output_identity(tmp_path, config, checkpoint=weight)
    atomic_save({'v': torch.ones(100)}, weight)
    with pytest.raises(ValueError, match='changed'):
        output_identity(tmp_path, config, checkpoint=weight)


def test_import_standalone_checkpoint_preserves_source_and_rejects_old_run(tmp_path):
    checkpoint = tmp_path/'source.pth.tar'
    atomic_save({'config': {'model': {'tokenizer': {'model_path': '/old/vae'}}, 'training': {}}, 'ema': {}}, checkpoint)
    run = import_checkpoint(checkpoint, tmp_path/'imported')
    assert (run/'checkpoints/latest.pth.tar').resolve() == checkpoint
    config = OmegaConf.load(run/'.hydra/config.yaml')
    assert config.model.tokenizer.model_path == 'stabilityai/sd-vae-ft-ema'
    with pytest.raises(ValueError, match='new empty'):
        import_checkpoint(checkpoint, run)
