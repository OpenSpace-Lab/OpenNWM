"""Algorithm contracts that catch silent errors in few-step distillation."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dmd2 import Critic, FourStepSchedule, distribution_matching_loss, enable_checkpointing, TIMESTEPS


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.01))
        self.calls = []

    def forward(self, x, t, **kwargs):
        self.calls.append((int(t[0]), torch.is_grad_enabled()))
        return torch.cat((x * self.weight, torch.ones_like(x) * 100), dim=1)


def schedule():
    return FourStepSchedule(np.cumprod(1-np.linspace(.0001, .02, 1000)))


def test_four_evaluations_and_reproducible_renoising():
    model = Predictor()
    noise = torch.randn(2, 4, 3, 3)
    a = schedule().sample(model, noise, {}, generator=torch.Generator().manual_seed(2))
    assert [t for t, _ in model.calls] == list(TIMESTEPS)
    assert not any(grad for _, grad in model.calls)
    b = schedule().sample(model, noise, {}, generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("step", range(4))
def test_backward_simulation_only_last_step_has_gradient(step):
    model = Predictor()
    sample = schedule().sample(model, torch.randn(2, 4, 3, 3), {}, stop_step=step, last_grad=True)
    assert model.calls == [(t, i == step) for i, t in enumerate(TIMESTEPS[:step+1])]
    sample.square().mean().backward()
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad)


def test_epsilon_conversion_ignores_learned_variance():
    model = Predictor()
    x = torch.ones(2, 4, 3, 3)
    t = torch.tensor([249, 999])
    s = schedule()
    a, sigma = s.coefficients(t, x)
    clean = s.predict_x0(model, x, t, {})
    torch.testing.assert_close(s.add_noise(clean, x * model.weight, t), x)


def test_dm_gradient_sign_normalization_and_detachment():
    generated = torch.tensor([[[[3.]]]], requires_grad=True)
    teacher = torch.tensor([[[[1.]]]], requires_grad=True)
    fake = torch.tensor([[[[2.]]]], requires_grad=True)
    loss, _ = distribution_matching_loss(generated, teacher, fake)
    loss.backward()
    torch.testing.assert_close(generated.grad, torch.tensor([[[[.5]]]]))
    assert teacher.grad is None and fake.grad is None


def test_equal_scores_give_zero_gradient():
    x = torch.ones(2, 4, 3, 3, requires_grad=True)
    loss, norm = distribution_matching_loss(x, x.detach(), x.detach())
    loss.backward()
    assert norm == 0 and x.grad.count_nonzero() == 0


def test_nonfinite_score_fails_instead_of_saving_broken_model():
    with pytest.raises(FloatingPointError):
        distribution_matching_loss(torch.ones(1, 4, 3, 3), torch.ones(1, 4, 3, 3),
                                   torch.full((1, 4, 3, 3), float("nan")))


def test_real_cdit_gan_head_and_checkpointing_preserve_gradients_and_state():
    from models import CDiT
    torch.set_num_threads(2)
    model = CDiT(input_size=8, patch_size=2, in_channels=4, hidden_size=32,
                 depth=2, num_heads=4, context_size=2)
    original_keys = set(model.state_dict())
    enable_checkpointing(model)
    assert set(model.state_dict()) == original_keys
    critic = Critic(model)
    condition = dict(x_cond=torch.randn(2, 2, 4, 8, 8), y=torch.randn(2, 3), rel_t=torch.ones(2))
    noise = torch.randn(2, 4, 8, 8, requires_grad=True)
    logits = critic.logits(noise, torch.tensor([249, 499]), condition)
    logits.sum().backward()
    assert model.x_embedder.proj.weight.grad is not None
    assert critic.head[-1].weight.grad.abs().sum() > 0
    assert noise.grad is not None and noise.grad.abs().sum() > 0
    # Generator GAN update freezes critic weights but must differentiate its input.
    critic.requires_grad_(False)
    new_input = torch.randn_like(noise, requires_grad=True)
    critic.logits(new_input, torch.tensor([249, 499]), condition).sum().backward()
    assert new_input.grad.abs().sum() > 0


def test_legacy_action_normalization_does_not_mutate_adapter_inputs():
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from distill_dmd2 import condition_for
    config = OmegaConf.create({"dataset": {"action_stats": {"min": [-2.5, -4], "max": [5, 4]}}})
    action = torch.tensor([[-2.5, 4., .3], [5., -4., -.2]])
    batch = dict(action=action.clone(), context=torch.randn(2, 4, 4, 8, 8), rel_t=torch.ones(2)/128)
    legacy = condition_for(SimpleNamespace(motion_condition_enabled=False), batch, config)
    torch.testing.assert_close(legacy["y"], torch.tensor([[-1., 1., .3], [1., -1., -.2]]))
    adapter = condition_for(SimpleNamespace(motion_condition_enabled=True), batch, config)
    torch.testing.assert_close(adapter["motion"]["real"]["values"], action)
    torch.testing.assert_close(batch["action"], action)
