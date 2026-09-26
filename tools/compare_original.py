#!/usr/bin/env python3
"""Compare the original and extracted implementations in isolated CPU processes.

The original checkout is optional for ordinary training/testing; only this
one-time migration check needs it. Both subprocesses use this Python interpreter.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def snapshot(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(snapshot(item) for item in value)
    return value


def seed(value):
    import numpy as np
    import torch

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def state(model):
    return snapshot(model.state_dict())


def gradients(model):
    return {name: snapshot(parameter.grad) for name, parameter in model.named_parameters()}


def initial_state(model, reference):
    initialized = state(model)
    if reference is not None:
        model.load_state_dict(reference["initial"], strict=True)
    return initialized


def lam_cases(reference):
    import torch
    from lam.navigation_variants import ActionOnlyLAM, DINOFeatureLAM, PixelActionLAM, PixelLAM

    results = {}
    for name, model_class in (
        ("action", ActionOnlyLAM), ("pixel", PixelLAM),
        ("dino", DINOFeatureLAM), ("pixel_action", PixelActionLAM),
    ):
        seed(101)
        kwargs = dict(image_channels=3, lam_model_dim=8, lam_latent_dim=4,
                      lam_patch_size=2, lam_enc_blocks=1, lam_dec_blocks=1,
                      lam_num_heads=2, lam_dropout=0.0, beta=0.25, log_interval=100)
        if name == "pixel_action":
            kwargs["action_loss_weight"] = 2.5
        if name == "dino":
            kwargs.update(lam_patch_size=16, use_precomputed_dino_features=True,
                          dino_weights_sha256="2" * 64)
        model = model_class(**kwargs).cpu()
        original = None if reference is None else reference[name]
        initialized = initial_state(model, original)
        seed(202)
        if name == "dino":
            batch = {"dino_features": torch.randn(2, 2, 300, 1024)}
        else:
            batch = {"videos": torch.rand(2, 2, 12, 16, 3),
                     "actions": torch.rand(2, 1, 3)}
        optimizer = model.configure_optimizers()
        if isinstance(optimizer, dict):
            optimizer = optimizer["optimizer"]
        records = []
        for step in range(2):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            seed(300 + step)
            outputs, loss, components = model.shared_step(batch)
            loss.backward()
            record = {"outputs": snapshot(outputs), "loss": snapshot(loss),
                      "components": snapshot(dict(components)), "gradients": gradients(model)}
            optimizer.step()
            record.update(parameters=state(model), optimizer=snapshot(optimizer.state_dict()))
            records.append(record)
        model.eval()
        with torch.no_grad():
            outputs, loss, components = model.shared_step(batch)
        results[name] = {"initial": initialized, "batch": snapshot(batch), "steps": records,
                         "evaluation": snapshot({"outputs": outputs, "loss": loss,
                                                 "components": dict(components)})}
        print(f"completed lam/{name}", flush=True)
    return results


def nwm_cases(reference):
    import torch
    from omegaconf import OmegaConf
    from diffusion import create_gaussian_diffusion
    from isolated_nwm_infer import model_forward_wrapper
    from models import CDiT
    from two_stage_nwm import configure_trainable_parameters
    from two_stage_training import _make_optimizer, _make_scheduler, two_stage_train_step

    class IdentityTokenizer:
        scaling_factor = 1.0

        def encode(self, value):
            return value

    motion = {
        "enabled": True, "available_types": ["none", "real", "geometry", "idm", "latent"],
        "train_types": ["none", "real", "geometry", "idm", "latent"],
        "adapter_hidden_dim": 8, "balance_parameter_count": False,
        "real": {"real_dim": 3, "normalization": {"mode": "identity"}},
        "geometry": {"geometry_dim": 3, "normalization": {"mode": "identity"}},
        "idm": {"idm_dim": 5, "normalization": {"mode": "identity"}},
        "latent": {"latent_dim": 4, "normalization": {"mode": "layer_norm"}},
    }
    cases = [(f"pretrain_{mode}", "proxy_pretrain", mode, None, None, "always")
             for mode in ("none", "geometry", "idm", "latent")]
    cases += [("pretrain_latentonly", "proxy_pretrain", "latent", None, None, "fallback")]
    cases += [(f"{scheme}_{phase}", "real_finetune", "real", scheme, phase, "always")
              for scheme in ("reset", "embedding_align", "real_to_latent", "state_conditioned_controller")
              for phase in ("warmup", "joint")]
    cases += [("no_pretrain", "real_finetune", "real", "reset", "joint", "always")]
    results = {}
    for name, stage, mode, scheme, phase, relative_time in cases:
        seed(101)
        finetune = {
            "scheme": scheme or "reset", "warmup_steps": 3, "joint_steps": 3,
            "adapter_lr": 2e-3, "backbone_lr": 1e-3,
            "align_cos_weight": 1.0, "align_l1_weight": 0.25,
            "real_to_latent_hidden_dim": 8, "freeze_latent_action_encoder": True,
            "controller_state_blocks": 2, "controller_mlp_ratio": 2.0,
            "controller_pretrain_lr": 1e-3, "controller_pretrain_weight_decay": 0.04,
            "controller_lr_warmup_steps": 1, "joint_lr": 1e-4, "latent_l2_weight": 1.0,
        }
        if name == "no_pretrain":
            finetune.update(random_init=True, stage1_checkpoint=None, warmup_steps=0)
        config = OmegaConf.create({
            "training_stage": stage, "action_mode": mode, "bfloat16": False,
            "dataset": {"context_size": 2, "precomputed_latents": {"enabled": False},
                        "distance": {"min_dist_cat": -64, "max_dist_cat": 64}},
            "motion_condition": motion,
            "proxy": {"type": mode if stage == "proxy_pretrain" else "latent",
                      "dim": {"geometry": 3, "idm": 5}.get(mode, 4),
                      "max_abs_frame_offset": 8, "use_precomputed_only": True,
                      "relative_time_mode": relative_time},
            "finetune": finetune,
            "training": {"grad_clip_val": 0.0, "ema_decay": 0.9999,
                         "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1e-3,
                                       "betas": [0.9, 0.999], "weight_decay": 0.01}},
            "scheduler": {"scheduler": {"_target_": None}},
            "model": {"generator": {"_target_": "models.CDiT", "hidden_size": 8}},
        })
        model = CDiT(input_size=2, context_size=2, patch_size=1, in_channels=1,
                     hidden_size=8, depth=1, num_heads=2, mlp_ratio=1.0,
                     learn_sigma=False, motion_condition=copy.deepcopy(motion),
                     training_stage=stage, action_mode=mode,
                     proxy_relative_time_mode=relative_time, finetune=finetune).cpu()
        # Fresh DiT gates otherwise hide conditioning gradients. The scratch
        # baseline deliberately retains production zero initialization.
        if name != "no_pretrain":
            generator = torch.Generator().manual_seed(991)
            layers = [block.adaLN_modulation[-1] for block in model.blocks]
            layers += [model.final_layer.adaLN_modulation[-1], model.final_layer.linear]
            with torch.no_grad():
                for layer in layers:
                    for parameter in layer.parameters():
                        parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.04)
        original = None if reference is None else reference[name]
        initialized = initial_state(model, original)
        configure_trainable_parameters(model, training_stage=stage, action_mode=mode,
                                       finetune_scheme=scheme, finetune_substage=phase)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = _make_optimizer(config, model, training_stage=stage, scheme=scheme, substage=phase)
        scheduler = _make_scheduler(config, optimizer, training_stage=stage, scheme=scheme, substage=phase)
        diffusion = create_gaussian_diffusion(timestep_respacing="", noise_schedule="squaredcos_cap_v2",
                                               use_kl=False, learn_sigma=False, diffusion_steps=12)
        seed(202)
        batch = {"video": torch.randn(2, 4, 1, 2, 2),
                 "k": torch.tensor([[-0.125, 1.0], [0.015625, -1.0]]),
                 "frame_offset": torch.tensor([[-8, 64], [1, -64]]),
                 "real_action": torch.randn(2, 2, 3),
                 "proxy_action": torch.randn(2, 2, {"geometry": 3, "idm": 5}.get(mode, 4)),
                 "proxy_valid": torch.ones(2, 2, dtype=torch.bool),
                 "proxy_found": torch.ones(2, 2, dtype=torch.bool),
                 "proxy_invalid": torch.zeros(2, 2, dtype=torch.bool),
                 "teacher_latent": torch.randn(2, 2, 4),
                 "latent_valid": torch.ones(2, 2, dtype=torch.bool)}
        records = []
        for step in range(2):
            forwards = []
            handle = model.register_forward_hook(lambda _module, _inputs, output: forwards.append(snapshot(output)))
            seed(300 + step)
            try:
                logs, accounting = two_stage_train_step(
                    model=model, ema=ema, diffusion=diffusion, tokenizer=IdentityTokenizer(), batch=batch,
                    device=torch.device("cpu"), optimizer=optimizer, scheduler=scheduler, scaler=None,
                    config=config, training_stage=stage, scheme=scheme, substage=phase, check_gradients=False)
            finally:
                handle.remove()
            records.append({"forwards": forwards, "logs": snapshot(logs), "proxy": snapshot(accounting),
                            "gradients": gradients(model), "parameters": state(model), "ema": state(ema),
                            "optimizer": snapshot(optimizer.state_dict()),
                            "scheduler": None if scheduler is None else snapshot(scheduler.state_dict())})
        results[name] = {"initial": initialized, "batch": snapshot(batch), "steps": records,
                         "trainable": [key for key, value in model.named_parameters() if value.requires_grad]}
        model.eval()
        action = batch["proxy_action"] if stage == "proxy_pretrain" else batch["real_action"]
        # The production inference wrapper, real twelve-step DDPM chain and
        # sample-key noise schedule run on CPU. Only the external VAE is bypassed.
        results[name]["evaluation"] = snapshot(model_forward_wrapper(
            (model, diffusion, IdentityTokenizer()), batch["video"][:, :2], action,
            num_timesteps=8, latent_size=2, device=torch.device("cpu"), num_cond=2,
            num_goals=2, rel_t=batch["k"], skip_tokenizer=True, motion_type=mode,
            sample_keys=["parity/0", "parity/1"], noise_seed=407, noise_stream="parity"))
        print(f"completed nwm/{name}", flush=True)
    return results


def worker(args):
    import torch

    root = Path(args.root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    reference = None if args.reference is None else torch.load(args.reference, map_location="cpu", weights_only=False)
    result = (lam_cases if args.project == "lam" else nwm_cases)(None if reference is None else reference["cases"])
    module_names = ("lam.navigation_variants", "lam.model") if args.project == "lam" else ("models", "two_stage_training", "two_stage_nwm", "isolated_nwm_infer")
    source_files = {}
    for module_name in module_names:
        path = Path(sys.modules[module_name].__file__).resolve()
        if root not in path.parents:
            raise RuntimeError(f"Imported {module_name} from outside selected checkout: {path}")
        source_files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    torch.save({"cases": result, "environment": {"python": sys.version, "torch": torch.__version__,
                                                 "device": "cpu", "threads": 1},
                "root": str(root), "source_files": source_files}, args.artifact)


def compare_tree(reference, candidate, *, rtol, atol):
    import torch

    stats = {"tensor_count": 0, "scalar_count": 0, "max_absolute_difference": 0.0}

    def visit(left, right, path):
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or left.shape != right.shape:
                raise AssertionError(f"{path}: tensor type, shape or dtype differs")
            if left.is_floating_point() and (not torch.isfinite(left).all() or not torch.isfinite(right).all()):
                raise AssertionError(f"{path}: nonfinite tensor")
            torch.testing.assert_close(left, right, rtol=rtol, atol=atol, msg=lambda message: f"{path}: {message}")
            if left.numel():
                difference = (left.to(torch.float64) - right.to(torch.float64)).abs().max().item()
                stats["max_absolute_difference"] = max(stats["max_absolute_difference"], difference)
            stats["tensor_count"] += 1
        elif isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                raise AssertionError(f"{path}: dictionary keys differ")
            for key in left:
                visit(left[key], right[key], f"{path}/{key}")
        elif isinstance(left, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                raise AssertionError(f"{path}: sequence shape differs")
            for index, (item, other) in enumerate(zip(left, right)):
                visit(item, other, f"{path}/{index}")
        else:
            if type(left) is not type(right) or left != right:
                raise AssertionError(f"{path}: {left!r} != {right!r}")
            stats["scalar_count"] += 1

    visit(reference, candidate, "case")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=("lam", "nwm"), required=True)
    parser.add_argument("--original", help="Original project root (contains lam/ or models.py)")
    parser.add_argument("--candidate", help="New project root; defaults to this repository's project subdirectory")
    parser.add_argument("--output-dir", help="Directory for CPU artifacts, subprocess logs and report.json")
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", help=argparse.SUPPRESS)
    parser.add_argument("--artifact", help=argparse.SUPPRESS)
    parser.add_argument("--reference", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return 0
    if not args.original or not args.output_dir:
        parser.error("--original and --output-dir are required")
    if args.rtol < 0 or args.atol < 0:
        parser.error("tolerances must be nonnegative")
    import torch

    candidate = Path(args.candidate or Path(__file__).resolve().parents[1] / args.project).resolve()
    original = Path(args.original).resolve()
    for root in (original, candidate):
        entry = root / ("lam/model.py" if args.project == "lam" else "models.py")
        if not entry.is_file():
            parser.error(f"project entry is missing: {entry}")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"project": args.project, "status": "running", "original": str(original),
              "candidate": str(candidate), "rtol": args.rtol, "atol": args.atol,
              "device": "cpu", "cases": {}, "commands": []}
    report_path = output / "report.json"
    started = time.monotonic()
    try:
        for label, root in (("original", original), ("candidate", candidate)):
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--project", args.project,
                       "--root", str(root), "--artifact", str(output / f"{label}.pt")]
            if label == "candidate":
                command += ["--reference", str(output / "original.pt")]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                               PYTHONHASHSEED="0", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
            environment.pop("PYTHONPATH", None)
            log_path = output / f"{label}.log"
            report["commands"].append({"argv": command, "cwd": str(root), "log": str(log_path)})
            with log_path.open("w") as log:
                process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
                print(f"{label}: pid={process.pid}, cwd={root}, log={log_path}", flush=True)
                status = process.wait()
            report["commands"][-1]["exit_status"] = status
            if status:
                raise RuntimeError(f"{label} exited with {status}; see {log_path}")
        left = torch.load(output / "original.pt", map_location="cpu", weights_only=False)
        right = torch.load(output / "candidate.pt", map_location="cpu", weights_only=False)
        report["environment"] = left["environment"]
        if left["environment"] != right["environment"]:
            raise AssertionError("original and candidate environments differ")
        report["source_files"] = {"original": left["source_files"], "candidate": right["source_files"]}
        if left["cases"].keys() != right["cases"].keys():
            raise AssertionError("case sets differ")
        for name, reference in left["cases"].items():
            try:
                stats = compare_tree(reference, right["cases"][name], rtol=args.rtol, atol=args.atol)
                report["cases"][name] = {"status": "passed", **stats}
            except AssertionError as error:
                report["cases"][name] = {"status": "failed", "error": str(error)}
        report["status"] = "passed" if all(case["status"] == "passed" for case in report["cases"].values()) else "failed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"{report['status']}: {report_path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
