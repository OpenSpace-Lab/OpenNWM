#!/usr/bin/env python3
"""Small, isolated LuSNAR adaptation with identical data/noise for two checkpoints.

The source run and raw images are read-only. Cache contains posterior means for
train/validation images only; test images never enter optimization or selection.
All generated files belong in the supplied NAS directories, outside this repo.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "nwm"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_torch(path, value):
    import torch

    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_inputs(args):
    from hydra_utils import load_experiment_config

    data = json.loads(args.data_manifest.read_text())
    samples = data.get("samples", [])
    groups = {name: [] for name in ("train", "val", "same_scene_test", "cross_scene_test")}
    seen_ids, image_splits = set(), {}
    for sample in samples:
        split = sample.get("split")
        if split not in groups or sample["id"] in seen_ids:
            raise ValueError("Every sample needs a unique ID and a recognized split")
        seen_ids.add(sample["id"])
        context, targets = sample["context_source_paths"], sample["gt_source_paths"]
        steps, actions = sample["prediction_steps"], sample["actions"]
        if len(context) != 4 or not targets or len(targets) != len(steps) or len(actions) != len(steps):
            raise ValueError(f"Malformed sample: {sample['id']}")
        if steps != sorted(set(steps)) or any(int(step) != step or step <= 0 for step in steps):
            raise ValueError("Prediction steps must be positive, unique, increasing integers")
        if any(len(action) != 3 or not all(math.isfinite(v) for v in action) for action in actions):
            raise ValueError("Actions must contain three finite values")
        for source in context + targets:
            path = str(Path(source).resolve())
            previous = image_splits.setdefault(path, split)
            if previous != split:
                raise ValueError(f"Image shared across splits: {path}")
            if split in {"train", "val"} and not Path(path).is_file():
                raise FileNotFoundError(path)
        groups[split].append(sample)
    if not groups["train"] or not groups["val"]:
        raise ValueError("Nonempty train and val splits are required")
    config = load_experiment_config(str(args.run_dir.resolve()))
    if config is None:
        raise ValueError("Source run must provide a model configuration")
    if int(config.dataset.context_size) != 4 or int(config.dataset.image_size) != 224:
        raise ValueError("This protocol requires four 224-pixel context images")
    return data, groups, config


def cache_contract(args, groups, config):
    from omegaconf import OmegaConf
    import diffusers
    import torch
    import torchvision

    tokenizer = OmegaConf.to_container(config.model.tokenizer, resolve=True)
    vae_dir = Path(tokenizer.pop("model_path")).expanduser().resolve()
    weight = vae_dir / "diffusion_pytorch_model.safetensors"
    if not weight.is_file():
        weight = vae_dir / "diffusion_pytorch_model.bin"
    if not weight.is_file() or not (vae_dir / "config.json").is_file():
        raise ValueError("Use an existing local SD-VAE directory with weights and config.json")
    paths = sorted({str(Path(p).resolve()) for split in ("train", "val")
                    for sample in groups[split]
                    for p in sample["context_source_paths"] + sample["gt_source_paths"]})
    contract = {
        "schema_version": 1,
        "data_manifest_sha256": sha256(args.data_manifest),
        "vae_weights_sha256": sha256(weight),
        "vae_config_sha256": sha256(vae_dir / "config.json"),
        "tokenizer": tokenizer,
        "image_size": int(config.dataset.image_size),
        "mean": list(config.dataset.mean), "std": list(config.dataset.std),
        "transform_source_sha256": sha256(REPO / "nwm/misc.py"),
        "tokenizer_source_sha256": sha256(REPO / "nwm/tokenizer_wrapper.py"),
        "posterior": "mean; BF16 autocast encoding; scaling_factor=0.18215; stored float32",
        "included_splits": ["train", "val"],
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "diffusers": diffusers.__version__},
        "images": [{"path": p, "bytes": Path(p).stat().st_size,
                    "mtime_ns": Path(p).stat().st_mtime_ns} for p in paths],
    }
    return contract, paths


def cuda_device(fraction):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for cache/train; use check for CPU validation")
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(json.dumps({"pid": os.getpid(), "gpu": torch.cuda.get_device_name(device),
                      "memory_fraction": fraction}), flush=True)
    return device


def prepare_cache(args, groups, config):
    from PIL import Image
    import torch
    from misc import get_transform
    from train_utils import setup_tokenizer

    contract, paths = cache_contract(args, groups, config)
    root = args.cache_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path, latent_path = root / "manifest.json", root / "latents.pt"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["signature"] != signature(contract):
            raise ValueError("Cache contract changed; use a different cache directory")
        if latent_path.is_file() and sha256(latent_path) == saved.get("latents_sha256"):
            print(f"Verified existing shared cache: {root}", flush=True)
            return
        raise ValueError("Existing cache is incomplete or corrupted; remove that cache directory before rebuilding")
    if latent_path.exists():
        raise ValueError("Uncommitted cache exists; remove latents.pt before rebuilding")
    device = cuda_device(args.memory_fraction)
    vae = setup_tokenizer(config, device).eval().requires_grad_(False)
    transform = get_transform(config.dataset.image_size, config.dataset.mean, config.dataset.std)
    latents = []
    started = time.monotonic()
    with torch.inference_mode():
        for begin in range(0, len(paths), args.microbatch):
            pixels = []
            for path in paths[begin:begin + args.microbatch]:
                with Image.open(path) as image:
                    pixels.append(transform(image.convert("RGB")))
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                encoded = vae.vae.encode(torch.stack(pixels).to(device)).latent_dist.mode()
                encoded = encoded.mul(vae.scaling_factor)
            if not torch.isfinite(encoded).all():
                raise FloatingPointError("Nonfinite VAE posterior means")
            latents.append(encoded.float().cpu())
            if begin == 0 or (begin // args.microbatch + 1) % 25 == 0:
                print(json.dumps({"cache_images": min(begin + args.microbatch, len(paths)),
                                  "total_images": len(paths),
                                  "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    tensor = torch.cat(latents)
    save_torch(latent_path, {"signature": signature(contract), "paths": paths, "latents": tensor})
    write_json(manifest_path, {"signature": signature(contract), "contract": contract,
                              "latents_sha256": sha256(latent_path), "shape": list(tensor.shape),
                              "encoding_microbatch": args.microbatch})
    print(f"Saved {len(paths)} train/val posterior means at {root}", flush=True)


def load_model(args, config, device):
    import torch
    from train_utils import setup_model
    from two_stage_nwm import configure_trainable_parameters

    checkpoint = (args.run_dir / "checkpoints/latest.pth.tar").resolve()
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = weights.get("ema")
    if not state:
        raise ValueError("Source checkpoint must contain a nonempty EMA state")
    normalized = {key.replace("_orig_mod.", ""): value for key, value in state.items()}
    if len(normalized) != len(state):
        raise ValueError("Source EMA keys collide after removing compile prefixes")
    model = setup_model(config, torch.device("cpu"))
    model.load_state_dict(normalized, strict=True)
    del normalized, state, weights
    if model.motion_condition_enabled:
        if str(config.get("training_stage", "legacy")) != "real_finetune" or str(config.finetune.scheme) != "reset":
            raise ValueError("Only the existing reset-scheme real-action Stage 2 model is supported")
        configure_trainable_parameters(model, training_stage="real_finetune", action_mode="real",
                                       finetune_scheme="reset", finetune_substage="joint")
    else:
        model.requires_grad_(True)
    return model.to(device), checkpoint


def model_kwargs(model, config, context, actions, steps):
    import torch

    kwargs = {"x_cond": context, "rel_t": steps.float() / 128.0}
    if model.motion_condition_enabled:
        kwargs["action"] = actions
    else:
        actions = actions.clone()
        lower = torch.as_tensor(config.dataset.action_stats.min, device=actions.device)
        upper = torch.as_tensor(config.dataset.action_stats.max, device=actions.device)
        if lower.shape != (2,) or upper.shape != (2,) or not torch.all(upper > lower):
            raise ValueError("Invalid legacy XY normalization bounds")
        actions[:, :2] = 2 * (actions[:, :2] - lower) / (upper - lower) - 1
        kwargs["y"] = actions
    return kwargs


def parameter_report(model):
    return {"total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_parameter_names": [n for n, p in model.named_parameters() if not p.requires_grad],
            "action_interface": "action/raw XY and yaw" if model.motion_condition_enabled
            else "y/minmax XY without clipping; yaw in radians"}


def check_inputs(args, groups, config):
    import torch
    from train_utils import setup_diffusion

    model, checkpoint = load_model(args, config, torch.device("cpu"))
    sample = groups["train"][0]
    size = int(config.model.generator.input_size)
    kwargs = model_kwargs(model, config, torch.zeros(1, 4, 4, size, size),
                          torch.tensor([sample["actions"][0]], dtype=torch.float32),
                          torch.tensor([sample["prediction_steps"][0]]))
    with torch.inference_mode():
        output = model.eval()(torch.zeros(1, 4, size, size), torch.tensor([500]), **kwargs)
    if output.shape != (1, 8, size, size) or not torch.isfinite(output).all():
        raise ValueError("CPU forward returned an invalid shape or nonfinite values")
    diffusion = setup_diffusion(config, False, torch.device("cpu"))
    if diffusion.num_timesteps != 1000:
        raise ValueError("Comparison requires all 1000 diffusion training timesteps")
    print(json.dumps({"cpu_check": "passed", "checkpoint": str(checkpoint),
                      "split_windows": {k: len(v) for k, v in groups.items()},
                      "model_kwargs": sorted(kwargs), "output_shape": list(output.shape),
                      **parameter_report(model)}, indent=2), flush=True)


def training_rows(samples, path_index):
    rows = []
    for sample in samples:
        context = [path_index[str(Path(path).resolve())] for path in sample["context_source_paths"]]
        for step, action, path in zip(sample["prediction_steps"], sample["actions"], sample["gt_source_paths"]):
            rows.append({"context": context, "target": path_index[str(Path(path).resolve())],
                         "step": step, "action": action, "sample_id": sample["id"]})
    return rows


def materialize(rows, indices, latents, device):
    import torch

    batch = [rows[int(i)] for i in indices]
    context = latents[torch.tensor([r["context"] for r in batch])].to(device)
    targets = latents[torch.tensor([r["target"] for r in batch])].to(device)
    actions = torch.tensor([r["action"] for r in batch], dtype=torch.float32, device=device)
    steps = torch.tensor([r["step"] for r in batch], dtype=torch.float32, device=device)
    return context, targets, actions, steps


def validate(model, config, diffusion, rows, latents, args, device):
    import torch

    # A private CPU RNG leaves training order and diffusion draws untouched.
    generator = torch.Generator().manual_seed(args.seed + 1_000_003)
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for start in range(0, len(rows), args.microbatch):
            ids = range(start, min(start + args.microbatch, len(rows)))
            context, target, action, steps = materialize(rows, ids, latents, device)
            times = torch.randint(1000, (len(target),), generator=generator).to(device)
            noise = torch.randn(target.shape, generator=generator).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                terms = diffusion.training_losses(model, target, times,
                    model_kwargs(model, config, context, action, steps), noise=noise)
            if not torch.isfinite(terms["loss"]).all():
                raise FloatingPointError("Nonfinite validation loss")
            total += terms["loss"].double().sum().item()
    return total / len(rows)


def train(args, groups, config):
    import torch
    from omegaconf import OmegaConf, open_dict
    from train_utils import setup_diffusion, update_ema

    contract, _ = cache_contract(args, groups, config)
    cache_manifest = json.loads((args.cache_root / "manifest.json").read_text())
    latent_path = args.cache_root / "latents.pt"
    if cache_manifest["signature"] != signature(contract) or sha256(latent_path) != cache_manifest["latents_sha256"]:
        raise ValueError("Shared cache signature or content hash mismatch")
    cache = torch.load(latent_path, map_location="cpu", weights_only=True)
    if cache["signature"] != cache_manifest["signature"]:
        raise ValueError("Latent payload signature mismatch")
    latents = cache["latents"]
    if not torch.isfinite(latents).all() or latents.ndim != 4 or latents.shape[1:] != (4, 28, 28):
        raise ValueError("Invalid latent cache")
    path_index = {p: i for i, p in enumerate(cache["paths"])}
    rows = {split: training_rows(groups[split], path_index) for split in ("train", "val")}
    root = args.output_root.resolve()
    if root == args.run_dir.resolve() or args.run_dir.resolve() in root.parents:
        raise ValueError("Output must be separate from the source checkpoint run")
    if (root / "metrics.json").exists() or (root / "checkpoints/latest.pth.tar").exists():
        raise ValueError("Output already contains a training run; choose a fresh directory")
    root.mkdir(parents=True, exist_ok=True)
    device = cuda_device(args.memory_fraction)
    torch.manual_seed(args.seed)
    model, source = load_model(args, config, device)
    ema = deepcopy(model).eval().requires_grad_(False)
    diffusion = setup_diffusion(config, False, device)
    if diffusion.num_timesteps != 1000:
        raise ValueError("Training must use all 1000 diffusion timesteps")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    settings = {"steps": args.steps, "effective_batch_size": args.batch_size,
                "microbatch": args.microbatch, "learning_rate": args.lr,
                "optimizer": "AdamW", "betas": [0.9, 0.999], "weight_decay": 0.01,
                "ema_decay": args.ema_decay, "grad_clip": 2.0, "seed": args.seed,
                "diffusion_training_steps": 1000, "precision": "BF16 autocast; FP32 weights",
                "posterior": "cached mean", "validation_every": 100,
                "selection": "fixed final step; validation monitored but no test-based selection",
                "sampling": "shuffled exhaustive (window, horizon) rows; repeat after each epoch",
                "resume": "no optimizer checkpoint; an interrupted run must restart in a fresh output directory"}
    source_files = ["tools/finetune_lusnar.py", "nwm/models.py", "nwm/motion_condition.py",
                    "nwm/train_utils.py", "nwm/two_stage_nwm.py", "nwm/diffusion/__init__.py",
                    "nwm/diffusion/gaussian_diffusion.py", "nwm/diffusion/respace.py"]
    metrics = {"status": "running", "source_run": str(args.run_dir.resolve()),
               "source_checkpoint": str(source), "source_checkpoint_sha256": sha256(source),
               "data_manifest_sha256": sha256(args.data_manifest),
               "cache_signature": cache_manifest["signature"], "settings": settings,
               "source_sha256": {name: sha256(REPO / name) for name in source_files},
               "parameters": parameter_report(model),
               "split_windows": {k: len(v) for k, v in groups.items()},
               "training_pairs": len(rows["train"]), "validation_pairs": len(rows["val"]),
               "train_sample_ids": [s["id"] for s in groups["train"]],
               "val_sample_ids": [s["id"] for s in groups["val"]],
               "training": [], "validation": []}
    with open_dict(config):
        config.seed = args.seed
        config.max_train_steps = args.steps
        config.training.run_name = root.name
        config.training.results_dir = str(root.parent)
        config.training.from_checkpoint = None
        config.training.wandb_enabled = False
        config.training.batch_size = args.batch_size
        config.training.ema_decay = args.ema_decay
        config.training.optimizer.lr = args.lr
        config.lusnar_adaptation = settings
    (root / ".hydra").mkdir(exist_ok=True)
    OmegaConf.save(config, root / ".hydra/config.yaml", resolve=False)
    write_json(root / "metrics.json", metrics)
    sampler = torch.Generator().manual_seed(args.seed)
    noise_rng = torch.Generator().manual_seed(args.seed + 17)
    order, cursor = [], 0
    started = time.monotonic()
    initial_loss = validate(ema, config, diffusion, rows["val"], latents, args, device)
    metrics["validation"].append({"step": 0, "ema_loss": initial_loss})
    print(json.dumps({"step": 0, "validation_ema_loss": initial_loss}), flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        indices = []
        while len(indices) < args.batch_size:
            if cursor == len(order):
                order = torch.randperm(len(rows["train"]), generator=sampler).tolist()
                cursor = 0
            take = min(args.batch_size - len(indices), len(order) - cursor)
            indices.extend(order[cursor:cursor + take])
            cursor += take
        # Draw for the effective batch before splitting it, independently of model state.
        times = torch.randint(1000, (args.batch_size,), generator=noise_rng)
        noise = torch.randn((args.batch_size, 4, 28, 28), generator=noise_rng)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for begin in range(0, args.batch_size, args.microbatch):
            stop = min(begin + args.microbatch, args.batch_size)
            context, target, action, offsets = materialize(rows["train"], indices[begin:stop], latents, device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                terms = diffusion.training_losses(model, target, times[begin:stop].to(device),
                    model_kwargs(model, config, context, action, offsets), noise=noise[begin:stop].to(device))
                loss = terms["loss"].sum() / args.batch_size
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step}")
            loss.backward()
            loss_value += loss.detach().item()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 2.0, error_if_nonfinite=True)
        optimizer.step()
        update_ema(ema, model, decay=args.ema_decay)
        entry = {"step": step, "loss": loss_value, "grad_norm": grad_norm.item(),
                 "elapsed_seconds": round(time.monotonic() - started, 2)}
        metrics["training"].append(entry)
        if step == 1 or step % 10 == 0:
            print(json.dumps({**entry, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30}), flush=True)
        if step % 100 == 0 or step == args.steps:
            val = validate(ema, config, diffusion, rows["val"], latents, args, device)
            metrics["validation"].append({"step": step, "ema_loss": val})
            metrics["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            metrics["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
            write_json(root / "metrics.json", metrics)
            print(json.dumps({"step": step, "validation_ema_loss": val,
                              "peak_allocated_gib": metrics["peak_allocated_gib"]}), flush=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    checkpoint_path = root / "checkpoints/latest.pth.tar"
    save_torch(checkpoint_path, {"model": {k: v.cpu() for k, v in model.state_dict().items()},
                                "ema": {k: v.cpu() for k, v in ema.state_dict().items()},
                                "train_steps": args.steps,
                                "config": OmegaConf.to_container(config, resolve=False),
                                "lusnar_adaptation": metrics})
    metrics.update(status="complete", elapsed_seconds=round(time.monotonic() - started, 2),
                   checkpoint_sha256=sha256(checkpoint_path))
    write_json(root / "metrics.json", metrics)
    print(json.dumps({"status": "complete", "checkpoint": str(checkpoint_path),
                      "initial_validation_loss": initial_loss,
                      "final_validation_loss": metrics["validation"][-1]["ema_loss"],
                      "elapsed_seconds": metrics["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "cache", "train"))
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.microbatch) < 1 or args.lr <= 0:
        parser.error("Steps, batch sizes and learning rate must be positive")
    if not 0 <= args.ema_decay < 1 or not 0 < args.memory_fraction <= 1:
        parser.error("Invalid EMA decay or memory fraction")
    if args.command != "check" and args.cache_root is None:
        parser.error("cache/train requires --cache-root")
    if args.command == "train" and args.output_root is None:
        parser.error("train requires --output-root")
    for artifact_root in (args.cache_root, args.output_root):
        if artifact_root is not None and (artifact_root.resolve() == REPO or REPO in artifact_root.resolve().parents):
            parser.error("Generated cache/checkpoint/metrics directories must be outside the source repository")
    import torch

    torch.set_num_threads(4)
    _, groups, config = load_inputs(args)
    if args.command == "check":
        check_inputs(args, groups, config)
    elif args.command == "cache":
        prepare_cache(args, groups, config)
    else:
        train(args, groups, config)


if __name__ == "__main__":
    main()
