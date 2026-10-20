"""DMD2 inference adapter for the published LittleBoss four-step students.

Matches the released dmd2.py FourStepSchedule: epsilon -> unclipped x0,
then fresh noise at the next original training timestep. This is not DDIM.
"""
import torch


class DMDInferenceSampler:
    timesteps = (999, 749, 499, 249)
    num_timesteps = 4
    uses_step_noise = True
    sampler = "dmd"

    def __init__(self, alphas_cumprod, device):
        if len(alphas_cumprod) != 1000:
            raise ValueError("DMD requires the full 1000-step training schedule")
        self.alpha = torch.as_tensor(alphas_cumprod, dtype=torch.float32, device=device)

    @classmethod
    def from_config(cls, config, device):
        from hydra.utils import instantiate
        from diffusion.gaussian_diffusion import ModelMeanType
        full = instantiate(config.model.diffusion, for_eval=False,
                           timestep_respacing="", _recursive_=False)
        if full.model_mean_type != ModelMeanType.EPSILON:
            raise ValueError("Released DMD students require epsilon prediction")
        return cls(full.alphas_cumprod, device)

    @torch.no_grad()
    def p_sample_loop(self, model, shape, noise=None, clip_denoised=False,
                      denoised_fn=None, cond_fn=None, model_kwargs=None,
                      device=None, progress=False, step_noises=None):
        if clip_denoised or denoised_fn is not None or cond_fn is not None:
            raise ValueError("DMD does not support x0 clipping or extra guidance")
        if noise is not None and tuple(noise.shape) != tuple(shape):
            raise ValueError("Initial DMD noise must match sample shape")
        # The shared wrapper supplies four draws, as for a four-step DDPM.
        # Only the first three are used: the last step returns clean x0.
        if step_noises is not None and (
            len(step_noises) not in (3, 4)
            or any(tuple(value.shape) != tuple(shape) for value in step_noises)
        ):
            raise ValueError("DMD needs three re-noising draws matching the sample shape")
        device = device or (noise.device if noise is not None else self.alpha.device)
        x = noise if noise is not None else torch.randn(*shape, device=device)
        alpha = self.alpha.to(device)
        for i, value in enumerate(self.timesteps):
            t = torch.full((shape[0],), value, device=device, dtype=torch.long)
            eps = model(x, t, **(model_kwargs or {}))[:, :shape[1]].float()
            a = alpha[t].reshape(-1, *([1] * (x.ndim - 1)))
            clean = (x.float() - (1 - a).sqrt() * eps) / a.sqrt()
            if i < 3:
                next_a = alpha[self.timesteps[i + 1]]
                fresh = torch.randn_like(x) if step_noises is None else step_noises[i]
                x = next_a.sqrt() * clean + (1 - next_a).sqrt() * fresh
        return clean
