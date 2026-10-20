#!/usr/bin/env python3
"""Predict one RGB image on Jetson from files only; no robot interfaces."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time
import types

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
MODEL_CHOICES = ("opennwm", "nwm-b", "opennwm-dmd", "nwm-dmd")


def add_sampler_arguments(parser, default_steps=250):
    parser.set_defaults(_default_steps=default_steps)
    parser.add_argument("--sampler", choices=("ddpm", "dpmpp_2m", "ddim", "dmd"), default=None,
                        help="Default: DDPM for teacher models, four-step DMD for *-dmd models")
    parser.add_argument("--ddim-eta", type=float, default=None,
                        help="DDIM only: stochasticity in [0, 1], default 0")


def validate_sampler_arguments(parser, args):
    try:
        args.sampler, args.steps = resolve_sampling(
            getattr(args, "model", "opennwm"), args.steps, args.sampler,
            args.ddim_eta, getattr(args, "_default_steps", 250))
    except ValueError as exc:
        parser.error(str(exc))
    if args.ddim_eta is not None:
        if args.sampler != "ddim":
            parser.error("--ddim-eta requires --sampler ddim")
        if not math.isfinite(args.ddim_eta) or not 0 <= args.ddim_eta <= 1:
            parser.error("--ddim-eta must be finite and in [0, 1]")


def resolve_sampling(model_name, steps=None, sampler=None, ddim_eta=None, default_steps=250):
    if model_name not in MODEL_CHOICES:
        raise ValueError("Unknown model: " + model_name)
    distilled = model_name.endswith("-dmd")
    sampler = sampler or ("dmd" if distilled else "ddpm")
    if distilled and sampler != "dmd":
        raise ValueError("DMD weights require --sampler dmd")
    if sampler == "dmd":
        if steps not in (None, 4) or ddim_eta is not None:
            raise ValueError("DMD requires exactly --steps 4 and does not accept --ddim-eta")
        return sampler, 4
    if sampler not in ("ddpm", "ddim", "dpmpp_2m"):
        raise ValueError("Unknown sampler: " + sampler)
    return sampler, default_steps if steps is None else steps


def checkpoint_relative_path(model_name, sampler):
    if sampler == "dmd":
        return ("NWM-DMD" if model_name in ("nwm-b", "nwm-dmd") else "OpenNWM-DMD") + "/student.pt"
    return ("NWM-B/cdit_b_100000.pth.tar" if model_name == "nwm-b"
            else "OpenNWM/opennwm-nav15-stage2-joint100k.pth.tar")


def checkpoint_digest(path):
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _Numpy1Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # The released Stage-2 checkpoint includes a NumPy-2 RNG ndarray.
        # JetPack's Python 3.8 uses NumPy 1.24; the ndarray format is compatible.
        if module == "numpy._core.multiarray":
            module = "numpy.core.multiarray"
        return super().find_class(module, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--goal", type=Path, help="Optional RGB file for LPIPS scoring")
    parser.add_argument("--output", type=Path, help="Optional prediction PNG; never overwritten")
    parser.add_argument("--model-root", type=Path, default=REPO.parents[1] / "models")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="opennwm")
    parser.add_argument("--steps", type=int, default=None)
    add_sampler_arguments(parser)
    parser.add_argument("--frames-ahead", type=int, default=1, help="Model horizon at 4 fps")
    parser.add_argument("--delta", nargs=3, type=float, default=(0., 0., 0.),
                        metavar=("X", "Y", "YAW"),
                        help="Hypothetical relative pose: waypoint-spacing units and radians")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    validate_sampler_arguments(parser, args)
    if not 2 <= args.steps <= 1000 or not 1 <= args.frames_ahead <= 64:
        parser.error("steps must be 2..1000; frames-ahead must be 1..64")
    if not all(math.isfinite(value) for value in args.delta):
        parser.error("delta must contain finite values")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    for path in (args.image, args.goal):
        if path is not None and not path.is_file():
            parser.error("Image does not exist: " + str(path))
    if args.output is not None and args.output.exists():
        parser.error("Output already exists: " + str(args.output))

    root = args.model_root.expanduser().resolve()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", MPLBACKEND="Agg",
                      TORCH_HOME=str(root / "planning/torch"),
                      NWM_VAE_PATH=str(root / "sd-vae-ft-ema"))
    sys.path.insert(0, str(REPO / "nwm"))
    with tempfile.TemporaryDirectory(prefix="opennwm-mpl-") as mpl_cache:
        os.environ["MPLCONFIGDIR"] = mpl_cache
        predict(args, root)


def predict(args, root):
    import numpy as np
    import torch
    from PIL import Image
    from isolated_nwm_infer import model_forward_wrapper
    from misc import get_transform, normalize_data

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use the Jetson opennwm environment.")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    started = time.monotonic()
    config, (model, diffusion, tokenizer) = load_models(
        args.model, root, args.steps, device, sampler=args.sampler, ddim_eta=args.ddim_eta
    )
    transform = get_transform(config.dataset.image_size, config.dataset.mean, config.dataset.std)
    with Image.open(args.image) as source:
        image = transform(source.convert("RGB"))
    # Cold-start context, matching the repository's single-image inference mode.
    context = image[None, None].repeat(1, int(config.dataset.context_size), 1, 1, 1).to(device)
    delta = np.asarray(args.delta, dtype=np.float32)[None, None, :]
    if args.model in ("nwm-b", "nwm-dmd"):
        stats = {key: np.asarray(value) for key, value in config.dataset.action_stats.items()}
        delta[..., :2] = normalize_data(delta[..., :2], stats)
    condition = torch.as_tensor(delta, device=device)
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    print("Loaded %s EMA strictly; predicting with %d %s steps." %
          (args.model, diffusion.num_timesteps, args.sampler.upper()), flush=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        # DMD release uses posterior mode, not the legacy stochastic VAE encode.
        if args.sampler == "dmd":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                latent = tokenizer.vae.encode(context.flatten(0, 1)).latent_dist.mode()
                context = (latent * tokenizer.scaling_factor).unflatten(0, (1, int(config.dataset.context_size)))
        prediction = model_forward_wrapper(
            (model, diffusion, tokenizer), context, condition, args.frames_ahead,
            int(config.model.generator.input_size), device,
            num_cond=int(config.dataset.context_size), motion_type="real",
            sample_keys=[0], noise_seed=args.seed, noise_stream="jetson/offline",
            skip_tokenizer=args.sampler == "dmd",
        ).float()
        if args.sampler == "dmd":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                prediction = tokenizer.decode(prediction, denormalize=True).float()
        torch.cuda.synchronize()
        inference_seconds = time.monotonic() - started
        expected_shape = (1, 3, int(config.dataset.image_size), int(config.dataset.image_size))
        if tuple(prediction.shape) != expected_shape or not torch.isfinite(prediction).all():
            raise RuntimeError("Prediction has invalid shape or non-finite pixels.")
        result = {"model": args.model, "gpu": torch.cuda.get_device_name(),
                  "shape": list(prediction.shape), "steps": diffusion.num_timesteps,
                  "load_seconds": round(load_seconds, 3),
                  "inference_seconds": round(inference_seconds, 3),
                  "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
        if args.sampler != "ddpm":
            result["sampler"] = args.sampler
            if args.sampler == "ddim":
                result["ddim_eta"] = 0.0 if args.ddim_eta is None else args.ddim_eta
        if args.goal is not None:
            import lpips
            scorer = lpips.LPIPS(net="alex", model_path=str(root / "planning/lpips/v0.1/alex.pth"),
                                 verbose=False).eval().to(device)
            with Image.open(args.goal) as source:
                goal = transform(source.convert("RGB"))[None].to(device)
            score = scorer(prediction * 2 - 1, goal).item()
            if not math.isfinite(score):
                raise RuntimeError("LPIPS score is non-finite.")
            result["lpips"] = score
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pixels = (prediction[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            with args.output.open("xb") as output:
                Image.fromarray(pixels).save(output, format="PNG")
            result["output"] = str(args.output.resolve())
    print(json.dumps(result, ensure_ascii=False), flush=True)


def load_models(model_name, root, steps, device, sampler=None, ddim_eta=None):
    """Load the same strict EMA, VAE and sampler for stills and video."""
    import torch
    from omegaconf import OmegaConf
    import hydra_utils  # Registers the checkpoint's divide resolver.
    from train_utils import setup_model, setup_tokenizer
    from diffusion.inference_sampling import setup_inference_diffusion, validate_sampler_options

    sampler, steps = resolve_sampling(model_name, steps, sampler, ddim_eta)
    checkpoint = root / checkpoint_relative_path(model_name, sampler)
    if sampler == "dmd":
        from dmd_sampler import DMDInferenceSampler
        config = OmegaConf.load(checkpoint.parent / "config.yaml")
        config.model.tokenizer.model_path = str(root / "sd-vae-ft-ema")
    else:
        validate_sampler_options(sampler, ddim_eta)
        config = OmegaConf.load(root / "OpenNWM/configs/opennwm-nav15-stage2.yaml")
    config.model.diffusion.eval_timestep_respacing = steps
    config.inference_sampler = sampler
    config.ddim_eta = ddim_eta
    if model_name == "nwm-b" and sampler != "dmd":
        config.motion_condition.enabled = False
        config.training_stage = "legacy"
        config.original_nwm = True
    model = setup_model(config, torch.device("cpu"))
    checkpoint_pickle = types.ModuleType("jetson_checkpoint_pickle")
    checkpoint_pickle.__dict__.update(vars(pickle))
    checkpoint_pickle.Unpickler = _Numpy1Unpickler
    if sampler == "dmd":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    else:
        payload = torch.load(checkpoint, map_location="cpu", pickle_module=checkpoint_pickle,
                             weights_only=False)
    model.load_state_dict(payload["ema"], strict=True)
    del payload
    gc.collect()
    model = model.eval().requires_grad_(False).to(device)
    tokenizer = setup_tokenizer(config, device).eval().requires_grad_(False)
    diffusion = (DMDInferenceSampler.from_config(config, device) if sampler == "dmd"
                 else setup_inference_diffusion(config, device))
    return config, (model, diffusion, tokenizer)


if __name__ == "__main__":
    main()
