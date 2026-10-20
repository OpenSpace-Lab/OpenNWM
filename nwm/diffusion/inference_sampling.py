"""Opt-in inference samplers; training and the legacy DDPM factory stay intact.

The model still predicts its checkpoint's epsilon (or x0) and optional sigma.
SpacedDiffusion owns output conversion, clipping and the training-index mapping.
Only the already converted C-channel x0 is passed to DPM-Solver++.
"""

import inspect
import logging
import math

import numpy as np
import torch
from hydra.utils import instantiate

from .gaussian_diffusion import ModelMeanType, _extract_into_tensor
from .respace import SpacedDiffusion

logger = logging.getLogger(__name__)
SAMPLERS = ("ddpm", "dpmpp_2m", "ddim")


def validate_sampler_options(sampler="ddpm", ddim_eta=None):
    if sampler not in SAMPLERS:
        raise ValueError("Unknown inference_sampler %r; choose %s" % (sampler, SAMPLERS))
    if ddim_eta is not None:
        if sampler != "ddim":
            raise ValueError("ddim_eta is only supported with inference_sampler=ddim")
        if isinstance(ddim_eta, bool) or not isinstance(ddim_eta, (int, float)):
            raise ValueError("ddim_eta must be a finite number in [0, 1]")
        if not math.isfinite(ddim_eta) or not 0 <= ddim_eta <= 1:
            raise ValueError("ddim_eta must be a finite number in [0, 1]")


def setup_inference_diffusion(config, device):
    """Use existing eval_timestep_respacing; never modify the training config."""
    sampler = config.get("inference_sampler", "ddpm")
    eta = config.get("ddim_eta")
    validate_sampler_options(sampler, eta)
    diffusion_config = config.model.diffusion
    if sampler == "ddpm":
        # Exactly the old setup_diffusion(..., for_eval=True) instantiation.
        # No extra scheduler, logging or random draws on this path.
        return instantiate(diffusion_config, for_eval=True, _recursive_=False)

    if diffusion_config.get("_target_") != "diffusion.create_gaussian_diffusion":
        raise ValueError("Accelerated sampling requires diffusion.create_gaussian_diffusion")
    requested = diffusion_config.get("eval_timestep_respacing", 250)
    if isinstance(requested, bool) or not (
        isinstance(requested, int) or isinstance(requested, str) and requested.isdigit()
    ):
        raise ValueError("Accelerated eval_timestep_respacing must be an integer step count")
    steps = int(requested)
    train_steps = int(diffusion_config.get("diffusion_steps", 1000))
    if not 2 <= steps <= train_steps:
        raise ValueError("Accelerated inference steps must be in [2, %d]" % train_steps)

    # Rebuild from the training factory, not from an already respaced 50-step
    # schedule. Both instances retain the configured beta schedule and semantics.
    diffusion = instantiate(diffusion_config, for_eval=True, _recursive_=False)
    full_diffusion = instantiate(
        diffusion_config, for_eval=False, timestep_respacing="", _recursive_=False
    )
    result = InferenceSampler(diffusion, full_diffusion, sampler, 0.0 if eta is None else eta)
    logger.info(
        "Inference sampler=%s steps=%d training_steps=%d mapped_endpoints=%s ddim_eta=%s",
        sampler, result.num_timesteps, train_steps,
        (diffusion.timestep_map[-1], diffusion.timestep_map[0]),
        result.eta if sampler == "ddim" else "not applicable",
    )
    return result


class InferenceSampler:
    """The p_sample_loop contract used by model_forward_wrapper, inference only."""

    def __init__(self, diffusion, full_diffusion, sampler, eta=0.0):
        validate_sampler_options(sampler, eta if sampler == "ddim" else None)
        if sampler == "ddpm":
            raise ValueError("Use the original SpacedDiffusion for DDPM")
        if sampler != "ddim" and eta != 0:
            raise ValueError("eta is only supported by DDIM")
        if type(diffusion) is not SpacedDiffusion or type(full_diffusion) is not SpacedDiffusion:
            raise ValueError("Accelerated sampling requires the repository's SpacedDiffusion")
        if diffusion.model_mean_type not in (ModelMeanType.EPSILON, ModelMeanType.START_X):
            raise ValueError("Accelerated sampling supports this repository's epsilon/x0 models only")
        self.diffusion = diffusion
        self.sampler = sampler
        self.eta = eta
        self.num_timesteps = diffusion.num_timesteps
        self.uses_step_noise = sampler == "ddim" and eta > 0
        self.trained_betas = full_diffusion.betas.copy()
        self.timesteps = list(reversed(diffusion.timestep_map))
        if (
            full_diffusion.timestep_map != list(range(full_diffusion.original_num_steps))
            or self.num_timesteps < 2
            or self.timesteps[0] != full_diffusion.num_timesteps - 1
            or self.timesteps[-1] != 0
            or any(a <= b for a, b in zip(self.timesteps, self.timesteps[1:]))
            or not np.allclose(
                diffusion.alphas_cumprod,
                full_diffusion.alphas_cumprod[diffusion.timestep_map],
                rtol=1e-12, atol=1e-12,
            )
        ):
            raise ValueError("Inference schedule must span the complete training schedule with matching alphas")
        self.scheduler_class = None
        if sampler == "dpmpp_2m":
            # Optional for legacy DDPM/DDIM; reuse the existing declared dependency.
            try:
                from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
            except ImportError as exc:
                raise ImportError("dpmpp_2m requires the project's declared Diffusers dependency") from exc
            if "timesteps" not in inspect.signature(DPMSolverMultistepScheduler.set_timesteps).parameters:
                raise RuntimeError("dpmpp_2m requires Diffusers with custom timesteps (declared 0.30.3/0.35.1)")
            self.scheduler_class = DPMSolverMultistepScheduler

    def _new_solver(self, device):
        # A fresh instance per generation: no multistep history shared between
        # batches, candidate chunks, observations, rollout frames or requests.
        solver = self.scheduler_class(
            num_train_timesteps=len(self.trained_betas),
            trained_betas=self.trained_betas,
            algorithm_type="dpmsolver++", solver_order=2, solver_type="midpoint",
            lower_order_final=True, thresholding=False,
            prediction_type="sample", variance_type=None,
            use_karras_sigmas=False, use_lu_lambdas=False,
            final_sigmas_type="zero", rescale_betas_zero_snr=False,
        )
        # `sample` describes the adapter's converted x0, NOT the checkpoint's
        # output. Never send raw epsilon or the 2C epsilon/sigma output here.
        solver.set_timesteps(timesteps=self.timesteps, device=device)
        return solver

    @torch.no_grad()
    def p_sample_loop(
        self, model, shape, noise=None, clip_denoised=True, denoised_fn=None,
        cond_fn=None, model_kwargs=None, device=None, progress=False, step_noises=None,
    ):
        if self.sampler == "dpmpp_2m" and cond_fn is not None:
            raise ValueError("dpmpp_2m does not support gradient cond_fn guidance; use DDPM or DDIM")
        if step_noises is not None:
            if not self.uses_step_noise:
                raise ValueError("Per-step noise is only used by stochastic DDIM (ddim_eta > 0)")
            if len(step_noises) != self.num_timesteps or any(
                tuple(value.shape) != tuple(shape) for value in step_noises
            ):
                raise ValueError("DDIM step noise count/shape must match the inference schedule/sample")
        if noise is not None and tuple(noise.shape) != tuple(shape):
            raise ValueError("Initial noise must match the requested sample shape")
        if device is None:
            device = noise.device if noise is not None else next(model.parameters()).device
        img = noise if noise is not None else torch.randn(*shape, device=device)
        solver = self._new_solver(device) if self.sampler == "dpmpp_2m" else None
        indices = list(range(self.num_timesteps - 1, -1, -1))
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        for step_index, i in enumerate(indices):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            # This call maps i -> training timestep exactly once, splits learned
            # sigma at C (latent C is not RGB's 3), converts epsilon -> x0 and
            # applies the original denoised_fn / clip_denoised semantics.
            out = self.diffusion.p_mean_variance(
                model, img, t, clip_denoised=clip_denoised,
                denoised_fn=denoised_fn, model_kwargs=model_kwargs,
            )
            if solver is not None:
                # Diffusers returns the adapter output's dtype. Keep the latent
                # sample dtype even for an x0 model evaluated under autocast.
                x0 = out["pred_xstart"].to(dtype=img.dtype)
                img = solver.step(x0, self.timesteps[step_index], img).prev_sample
                continue
            if cond_fn is not None:
                out = self.diffusion.condition_score(
                    cond_fn, out, img, t, model_kwargs={} if model_kwargs is None else model_kwargs
                )
            # GaussianDiffusion.ddim_sample Eq. 12, with caller-owned noise for
            # the repository's per-sample RNG contract; eta=0 draws no step noise.
            x0 = out["pred_xstart"]
            eps = self.diffusion._predict_eps_from_xstart(img, t, x0)
            alpha = _extract_into_tensor(self.diffusion.alphas_cumprod, t, img.shape)
            previous = _extract_into_tensor(self.diffusion.alphas_cumprod_prev, t, img.shape)
            sigma = self.eta * ((1 - previous) / (1 - alpha)).sqrt() * (1 - alpha / previous).sqrt()
            img = previous.sqrt() * x0 + (1 - previous - sigma**2).clamp_min(0).sqrt() * eps
            if self.uses_step_noise and i != 0:
                step_noise = torch.randn_like(img) if step_noises is None else step_noises[step_index]
                img = img + sigma * step_noise
        return img
