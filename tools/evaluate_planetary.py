#!/usr/bin/env python3
"""Reproducible exploratory OpenNWM prediction on LuSNAR and SynMars-TW.

Uses the repository's original model/diffusion implementation. This is a
documented out-of-domain protocol, not an official dataset benchmark.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "nwm"))
from planetary_eval_data import build_samples


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def feedback_description(samples):
    """Describe the feedback branches used by the selected sample protocols."""
    direct = any(sample["protocol"] == "direct" for sample in samples)
    rollout = any(sample["protocol"] != "direct" for sample in samples)
    descriptions = []
    if direct:
        descriptions.append(
            "Direct horizons use the same four observed context frames and "
            "anchor-relative cumulative actions, with no generated-frame feedback."
        )
    if rollout:
        descriptions.append(
            "Autoregressive and synthetic-control samples re-encode normalized "
            "generated RGB using current-body-frame actions at each step."
        )
    return " ".join(descriptions)


def load_samples_manifest(path, context_size):
    """Read raw source samples without reselecting data or changing controls."""
    manifest = json.loads(path.read_text())
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("--samples-manifest must contain a nonempty samples list")
    identities = set()
    for sample in samples:
        identity = sample.get("id")
        if (not isinstance(identity, str) or not identity or
                Path(identity).name != identity or identity in {".", ".."} or
                identity in identities):
            raise ValueError(f"Invalid or duplicate sample id: {identity!r}")
        identities.add(identity)
        if not sample.get("dataset") or sample.get("protocol") not in {"direct", "autoregressive", "synthetic_control"}:
            raise ValueError(f"Sample {identity}: dataset and supported protocol are required")
        contexts = sample.get("context_source_paths")
        steps = sample.get("prediction_steps")
        actions = sample.get("actions")
        targets = sample.get("gt_source_paths") or []
        if not isinstance(contexts, list) or len(contexts) != context_size:
            raise ValueError(f"Sample {identity}: expected {context_size} context_source_paths")
        if (not isinstance(steps, list) or not steps or
                any(type(step) is not int or step < 1 for step in steps) or
                steps != sorted(set(steps))):
            raise ValueError(f"Sample {identity}: prediction_steps must be positive, unique and increasing")
        if sample["protocol"] != "direct" and steps != list(range(1, len(steps) + 1)):
            raise ValueError(f"Sample {identity}: autoregressive steps must be consecutive from 1")
        if (not isinstance(actions, list) or len(actions) != len(steps) or
                any(not isinstance(action, list) or len(action) != 3 or
                    any(type(value) not in (int, float) or not math.isfinite(value) for value in action)
                    for action in actions)):
            raise ValueError(f"Sample {identity}: actions must have shape [prediction_steps, 3]")
        if not isinstance(targets, list) or (targets and len(targets) != len(steps)):
            raise ValueError(f"Sample {identity}: gt_source_paths must align with prediction_steps")
        for source in contexts + targets:
            if not isinstance(source, str) or not Path(source).is_absolute() or not Path(source).is_file():
                raise ValueError(f"Sample {identity}: expected an existing absolute source path: {source!r}")
    return samples, manifest.get("data_metadata", {}), manifest.get("limitations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=os.environ.get("NWM_NAV15_STAGE2_RUN"))
    parser.add_argument("--lusnar-root", type=Path, default=Path("/file_system/nas/algorithm/dujun.nie/datasets/LuSNAR"))
    parser.add_argument("--synmars-root", type=Path, default=Path("/file_system/nas/algorithm/dujun.nie/datasets/SynMars-TW"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-fraction", type=float,
                        help="Optional CUDA allocator memory fraction, in (0, 1]")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=250)
    parser.add_argument("--windows-per-scene", type=int, default=4)
    parser.add_argument("--synmars-images", type=int, default=12)
    parser.add_argument("--samples-manifest", type=Path,
                        help="Reuse raw source samples and actions from a JSON samples manifest")
    parser.add_argument("--model-label", help="Model/checkpoint label used in reports")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.run_dir is None or args.batch_size < 1:
        parser.error("--run-dir and a positive --batch-size are required")
    if args.memory_fraction is not None and not 0 < args.memory_fraction <= 1:
        parser.error("--memory-fraction must be in (0, 1]")

    import numpy as np
    from PIL import Image
    import torch
    from hydra_utils import load_experiment_config
    from misc import get_transform, get_normalize
    from omegaconf import open_dict, OmegaConf
    from train_utils import setup_model, setup_tokenizer, setup_diffusion
    from isolated_nwm_infer import model_forward_wrapper

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = load_experiment_config(str(args.run_dir.resolve()))
    if config is None:
        raise ValueError("Missing model configuration")
    with open_dict(config):
        config.model.diffusion.eval_timestep_respacing = args.diffusion_steps
    checkpoint = (args.run_dir / "checkpoints/latest.pth.tar").resolve()
    signature = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "seed": args.seed, "diffusion_steps": args.diffusion_steps,
        "windows_per_scene": args.windows_per_scene, "synmars_images": args.synmars_images,
        "lusnar_root": str(args.lusnar_root.resolve()), "synmars_root": str(args.synmars_root.resolve()),
        "source_sha256": {name: digest(REPO / name) for name in [
            "tools/evaluate_planetary.py", "tools/planetary_eval_data.py",
            "nwm/isolated_nwm_infer.py", "nwm/motion_condition.py", "nwm/models.py"]},
        "vae": str(config.model.tokenizer.model_path), "vae_encoding": "posterior_mean",
        "image_size": int(config.dataset.image_size), "context_size": int(config.dataset.context_size),
    }
    if args.samples_manifest is not None:
        args.samples_manifest = args.samples_manifest.expanduser().resolve()
        signature["samples_manifest"] = {
            "path": str(args.samples_manifest), "sha256": digest(args.samples_manifest),
        }
    if args.model_label is not None:
        if not args.model_label.strip():
            parser.error("--model-label must not be empty")
        signature["model_label"] = args.model_label
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] != signature:
            raise ValueError("Protocol/checkpoint/source changed: use a new output directory")
        print("Resuming identical protocol", flush=True)
    else:
        supplied_limitations = None
        if args.samples_manifest is None:
            samples, metadata = build_samples(args.lusnar_root, args.synmars_root,
                windows_per_scene=args.windows_per_scene, synmars_images=args.synmars_images, frames=16, fps=4)
        else:
            samples, metadata, supplied_limitations = load_samples_manifest(
                args.samples_manifest, int(config.dataset.context_size))
        manifest = {
            "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "signature": signature, "checkpoint": signature["checkpoint"],
            "protocol_details": {
                "model": "OpenNWM Nav15 Stage 2 joint100k EMA; CDiT-B/2",
                "sampling": f"DDPM {args.diffusion_steps}; seed {args.seed}; deterministic per-sample diffusion noise",
                "vae": "Posterior means for deterministic contexts, differs from repository benchmark posterior sampling",
                "preprocessing": "Repository get_transform: center crop to 4:3 then resize to 224x224, RGB",
                "lusnar": "4 observed context frames; 4 FPS nearest-timestamp sampling; direct horizons 1/4/8/16 steps and autoregressive 16 steps",
                "synmars": "12 test left stills; repeat each initial image four times; straight/left/right synthetic controls; 16 model steps, no known physical duration",
                "feedback": feedback_description(samples),
            },
            "data_metadata": metadata,
            "limitations": [
                "Exploratory OOD evaluation of one checkpoint and seed; not an official benchmark or navigation success rate.",
                "LuSNAR uses rover planar body poses; pitch/roll/height and camera mounting are not fully modeled.",
                "LuSNAR coordinate reflection and per-scene median-step normalization are explicit dataset-adaptation assumptions.",
                "SynMars-TW lacks action/pose/timestamp truth; generated controls cannot be scored against future GT.",
                "Small sample evaluation: no FID and no claim of whole-dataset coverage.",
            ],
            "samples": samples,
        }
        if args.model_label is not None:
            manifest["model_label"] = args.model_label
            manifest["protocol_details"]["model"] = args.model_label
        if args.samples_manifest is not None:
            manifest["protocol_details"]["samples"] = "Contexts, targets, horizons and raw actions from the signed external samples manifest"
            manifest["protocol_details"]["lusnar"] = "Observed context frames and paired targets at the horizons specified by each external sample"
            manifest["protocol_details"]["synmars"] = (
                "Contexts, paired targets when supplied, and action provenance are specified by each external sample; "
                "actions may be estimated or synthetic and are not dataset pose ground truth. "
                "Frame order does not establish capture timestamps or physical duration."
            )
            datasets = {sample["dataset"] for sample in samples}
            for dataset, key in (("LuSNAR", "lusnar"), ("SynMars-TW", "synmars")):
                if dataset not in datasets:
                    manifest["protocol_details"].pop(key, None)
                    manifest["limitations"] = [item for item in manifest["limitations"] if not item.startswith(dataset)]
            if supplied_limitations is not None:
                manifest["limitations"] = supplied_limitations
            elif "SynMars-TW" in datasets:
                manifest["limitations"] = [
                    "SynMars-TW has no pose/action/timestamp truth; external sample metadata identifies "
                    "estimated or synthetic actions and any paired source-image references."
                    if item.startswith("SynMars-TW") else item for item in manifest["limitations"]
                ]
        for sample in samples:
            sample["gt_source_paths"] = sample.get("gt_source_paths") or []
            directory = root / "samples" / sample["id"]
            directory.mkdir(parents=True, exist_ok=True)
            sample.setdefault("fps", 4)
            sample["initial_path"] = str(directory / "initial.png")
            sample["context_paths"] = [str(directory / f"context_{i:02d}.png") for i in range(len(sample["context_source_paths"]))]
            sample["gt_paths"] = [str(directory / f"gt_{i:03d}.png") for i in range(len(sample.get("gt_source_paths", [])))]
            sample["prediction_paths"] = [str(directory / f"prediction_{i:03d}.png") for i in range(len(sample["prediction_steps"]))]
            sample["source_sha256"] = {path: digest(path) for path in dict.fromkeys(sample["context_source_paths"] + sample.get("gt_source_paths", []))}
        write_json(manifest_path, manifest)
        OmegaConf.save(config, root / "model_config.yaml")
        source_dir = root / "source"
        source_dir.mkdir(exist_ok=True)
        for name in signature["source_sha256"]:
            shutil.copy2(REPO / name, source_dir / Path(name).name)

    transform = get_transform(config.dataset.image_size, config.dataset.mean, config.dataset.std)
    normalize = get_normalize(config.dataset.mean, config.dataset.std)

    def load_source(path):
        with Image.open(path) as img:
            return transform(img.convert("RGB"))

    def save_pixels(pixels, path):
        array = (pixels.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        temporary = Path(path).with_suffix(".tmp.png")
        Image.fromarray(array).save(temporary)
        temporary.replace(path)

    for sample in manifest["samples"]:
        for source, target in zip(sample["context_source_paths"], sample["context_paths"]):
            if not Path(target).exists():
                save_pixels(load_source(source) * .5 + .5, target)
        if not Path(sample["initial_path"]).exists():
            shutil.copyfile(sample["context_paths"][-1], sample["initial_path"])
        for source, target in zip(sample.get("gt_source_paths", []), sample["gt_paths"]):
            if not Path(target).exists():
                save_pixels(load_source(source) * .5 + .5, target)
    print(f"Prepared {len(manifest['samples'])} samples at {root}", flush=True)
    if args.prepare_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    if args.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    print(f"GPU={torch.cuda.get_device_name(device)} PID={os.getpid()} loading EMA", flush=True)
    vae = setup_tokenizer(config, device).eval()
    vae.encode = lambda images: vae.vae.encode(images).latent_dist.mode().mul_(vae.scaling_factor)
    model = setup_model(config, device)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(weights["ema"], strict=True)
    del weights
    model.eval()
    all_models = (model, setup_diffusion(config, True, device), vae)
    groups = defaultdict(list)
    for sample in manifest["samples"]:
        groups[(sample["dataset"], sample["protocol"], tuple(sample["prediction_steps"]))].append(sample)

    def load_saved(path):
        with Image.open(path) as image:
            return torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1).float() / 255

    started = time.monotonic()
    with torch.inference_mode():
        for (dataset, protocol, steps), group in groups.items():
            for begin in range(0, len(group), args.batch_size):
                batch = group[begin:begin + args.batch_size]
                if all(all(Path(p).is_file() for p in s["prediction_paths"]) for s in batch):
                    continue
                context = torch.stack([torch.stack([load_source(p) for p in s["context_source_paths"]]) for s in batch]).to(device)
                actions = torch.tensor([s["actions"] for s in batch], dtype=torch.float32, device=device)
                if not model.motion_condition_enabled:
                    # Legacy NWM expects normalized XY and yaw in radians;
                    # OpenNWM's real adapter normalizes raw actions internally.
                    minimum = actions.new_tensor(config.dataset.action_stats.min)
                    maximum = actions.new_tensor(config.dataset.action_stats.max)
                    actions[..., :2] = 2 * (actions[..., :2] - minimum) / (maximum - minimum) - 1
                direct = protocol == "direct"
                for index, step in enumerate(steps):
                    destinations = [s["prediction_paths"][index] for s in batch]
                    if all(Path(p).is_file() for p in destinations):
                        pixels = torch.stack([load_saved(p) for p in destinations]).to(device)
                    else:
                        # Synthetic action alternatives share the source's noise_key,
                        # isolating control changes from stochastic noise changes.
                        pixels = model_forward_wrapper(all_models, context, actions[:, index:index + 1],
                            step if direct else 1, int(config.model.generator.input_size), device,
                            num_cond=int(config.dataset.context_size), motion_type="real",
                            sample_keys=[s.get("noise_key", s["id"]) for s in batch], noise_seed=args.seed,
                            noise_stream=f"planetary/{dataset}/{protocol}/step={step}")
                        for p, path in zip(pixels, destinations):
                            save_pixels(p, path)
                        # Quantized saved output is fed back so resume and fresh
                        # execution use exactly the same visible RGB context.
                        pixels = torch.stack([load_saved(p) for p in destinations]).to(device)
                    if not direct:
                        context = torch.cat((context[:, 1:], normalize(pixels).unsqueeze(1)), dim=1)
                    progress = {"dataset": dataset, "protocol": protocol, "batch_start": begin,
                        "batch_size": len(batch), "step": step, "elapsed_seconds": round(time.monotonic() - started, 1)}
                    write_json(root / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["inference_seconds_this_invocation"] = time.monotonic() - started
    manifest["generated_frames"] = sum(len(s["prediction_paths"]) for s in manifest["samples"])
    write_json(manifest_path, manifest)
    print(f"COMPLETE frames={manifest['generated_frames']} elapsed={time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
