"""DMD2 adaptation for conditional CDiT epsilon predictors.

Method: Yin et al., NeurIPS 2024, https://github.com/tianweiy/DMD2
Reference revision: 8d8fa55633d47cfb81bbc7a892e7248f9518763f.
No changes to the original training or DDPM inference paths are required.
"""
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


TIMESTEPS = (999, 749, 499, 249)


class FourStepSchedule(nn.Module):
    def __init__(self, alphas_cumprod):
        super().__init__()
        if len(alphas_cumprod) != 1000:
            raise ValueError("This four-step schedule requires 1000 training timesteps")
        self.register_buffer("alpha", torch.as_tensor(alphas_cumprod, dtype=torch.float32))

    def coefficients(self, t, x):
        a = self.alpha[t].reshape(-1, *([1] * (x.ndim - 1)))
        return a.sqrt(), (1 - a).sqrt()

    def add_noise(self, x, noise, t):
        a, s = self.coefficients(t, x)
        return a * x + s * noise

    def predict_x0(self, model, x, t, condition):
        # CDiT's second C channels predict DDPM variance; DMD2 uses epsilon only.
        eps = model(x, t, **condition)[:, :x.shape[1]].float()
        a, s = self.coefficients(t, x)
        return (x.float() - s * eps) / a

    def sample(self, model, noise, condition, *, stop_step=3, last_grad=False,
               generator=None):
        """Simulate inference, differentiating only the selected last step.

        During training sample stop_step uniformly from 0..3. Earlier steps
        use the student's own predictions, addressing train/inference mismatch.
        Inference uses all four steps and three fresh re-noising draws.
        """
        if stop_step not in range(4):
            raise ValueError("stop_step must be in [0, 3]")
        x = noise
        for i, value in enumerate(TIMESTEPS[:stop_step + 1]):
            t = torch.full((x.shape[0],), value, device=x.device, dtype=torch.long)
            context = nullcontext() if last_grad and i == stop_step else torch.no_grad()
            with context:
                clean = self.predict_x0(model, x, t, condition)
            if i < stop_step:
                next_t = torch.full_like(t, TIMESTEPS[i + 1])
                eps = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
                x = self.add_noise(clean.detach(), eps, next_t)
        return clean


def distribution_matching_loss(sample, teacher_x0, fake_x0):
    """Normalized score difference; target is detached, no teacher regression."""
    with torch.no_grad():
        scale = (sample.float() - teacher_x0.float()).abs().mean(
            dim=tuple(range(1, sample.ndim)), keepdim=True).clamp_min(1e-6)
        grad = (fake_x0.float() - teacher_x0.float()) / scale
        if not torch.isfinite(grad).all():
            raise FloatingPointError("Non-finite DMD2 score difference")
        target = sample.float() - grad
    return 0.5 * F.mse_loss(sample.float(), target), grad.norm().detach()


class Critic(nn.Module):
    """Trainable fake score CDiT with a conditional GAN head on its tokens.

    CDiT has transformer tokens rather than the UNet feature maps in DMD2;
    pool final-block tokens and classify them with an MLP. The head and the
    score backbone both learn from the real/fake logistic loss.
    """
    def __init__(self, score):
        super().__init__()
        self.score = score
        width = score.pos_embed.shape[-1]
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 256),
                                  nn.SiLU(), nn.Linear(256, 1))

    def logits(self, x, t, condition):
        features = []
        handle = self.score.blocks[-1].register_forward_hook(
            lambda module, args, output: features.append(output))
        try:
            self.score(x, t, **condition)
        finally:
            handle.remove()
        return self.head(features[0].mean(dim=1)).float()


def enable_checkpointing(model):
    """Checkpoint block activations without changing checkpoint parameter names."""
    from torch.utils.checkpoint import checkpoint
    for block in model.blocks:
        original = block.forward

        def forward(*args, _original=original, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)
            return _original(*args, **kwargs)

        block.forward = forward
