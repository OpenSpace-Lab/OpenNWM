"""Isolated DMD2 preparation, full-parameter training, and paired evaluation.

Run from the repository root after sourcing .env; see DMD2.md.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf, open_dict

from dmd2 import Critic, FourStepSchedule, TIMESTEPS, distribution_matching_loss, enable_checkpointing
from hydra_utils import load_experiment_config
from train_utils import setup_model, setup_tokenizer, setup_diffusion
from motion_condition import make_motion_group


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(run):
    config = load_experiment_config(str(run))
    if config is None:
        raise ValueError(f"Missing source config: {run}")
    with open_dict(config):
        config.model.tokenizer.model_path = os.environ.get(
            "NWM_VAE_PATH", config.model.tokenizer.model_path)
    return config


def load_weights(config, path, device):
    model = setup_model(config, torch.device("cpu"))
    weights = torch.load(path, map_location="cpu", weights_only=False)
    state = weights.get("ema", weights.get("model", weights))
    model.load_state_dict(state, strict=True)
    del weights, state
    return model.to(device).eval()


def condition_for(model, batch, config):
    action = batch["action"].clone()
    result = {"x_cond": batch["context"], "rel_t": batch["rel_t"]}
    if model.motion_condition_enabled:
        result["motion"] = make_motion_group("real", action)
    else:
        # Exact legacy TrainingDataset endpoint normalization (no delta actions).
        low = action.new_tensor(config.dataset.action_stats.min)
        high = action.new_tensor(config.dataset.action_stats.max)
        action[:, :2] = 2 * (action[:, :2] - low) / (high - low) - 1
        result["y"] = action
    return result


def select_records(config, args):
    """One deterministic window per trajectory; train/val/test stay disjoint."""
    rng = np.random.default_rng(args.seed)
    records = []
    for name in args.datasets:
        entry = config.dataset.datasets[name]
        root = Path(args.data_root) / ("tartan" if name == "tartan_drive" else name)
        splits = {}
        for split in ("train", "val", "test"):
            path = Path(args.split_root) / name / split / "traj_names.txt"
            splits[split] = [x for x in path.read_text().splitlines() if x]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if set(splits[a]) & set(splits[b]):
                raise ValueError(f"Trajectory leakage: {name} {a}/{b}")
        for split, count in (("train", args.train_per_dataset),
                             ("val", args.eval_per_dataset),
                             ("test", args.eval_per_dataset)):
            found = 0
            for trajectory in rng.permutation(splits[split]):
                trajectory = str(trajectory)
                path = root / trajectory / "traj_data.pkl"
                if not path.is_file():
                    continue
                with path.open("rb") as f:
                    traj = pickle.load(f)
                n = len(traj["position"])
                horizon = int(args.horizons[found % len(args.horizons)])
                minimum = int(config.dataset.context_size) - 1
                if n <= minimum + horizon:
                    continue
                current = int(rng.integers(minimum, n - horizon))
                frames = list(range(current - minimum, current + 1)) + [current + horizon]
                if not all((root / trajectory / f"{i}.jpg").is_file() for i in frames):
                    continue
                records.append(dict(dataset=name, split=split, trajectory=trajectory,
                                    current=current, horizon=horizon, frames=frames,
                                    root=str(root), waypoint_spacing=float(entry.metric_waypoint_spacing)))
                found += 1
                if found == count:
                    break
            if found != count:
                raise ValueError(f"Only {found}/{count} valid trajectories for {name}/{split}")
    return records


@torch.no_grad()
def prepare(args):
    from datasets import BaseDataset
    from misc import get_transform
    config = load_config(args.source_run)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "samples.pt").exists():
        raise FileExistsError("Prepared data exists; use a new output directory")
    records = select_records(config, args)
    write_json(output / "manifest.json", records)
    vae = setup_tokenizer(config, torch.device("cuda")).eval().requires_grad_(False)
    transform = get_transform(config.dataset.image_size, config.dataset.mean, config.dataset.std)
    buffers = {s: {k: [] for k in ("context", "target", "action", "rel_t", "pixels", "context_pixels")}
               for s in ("train", "val", "test")}
    for index, record in enumerate(records):
        root = Path(record["root"]) / record["trajectory"]
        with (root / "traj_data.pkl").open("rb") as f:
            traj = {k: np.asarray(v).astype(float) for k, v in pickle.load(f).items()}
        helper = SimpleNamespace(len_traj_pred=record["horizon"], normalize=bool(config.dataset.normalize),
                                 WAYPOINT_SPACING=record["waypoint_spacing"])
        _, action = BaseDataset._compute_actions(helper, traj, record["current"],
                                                 np.array([record["frames"][-1]]))
        images = []
        for frame in record["frames"]:
            with Image.open(root / f"{frame}.jpg") as img:
                images.append(transform(img.convert("RGB")))
        pixels = torch.stack(images)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Fixed posterior means shared across all methods and both teachers.
            latents = vae.vae.encode(pixels.cuda()).latent_dist.mode() * vae.scaling_factor
        b = buffers[record["split"]]
        b["context"].append(latents[:-1].float().cpu())
        b["target"].append(latents[-1].float().cpu())
        b["action"].append(torch.tensor(action[0], dtype=torch.float32))
        b["rel_t"].append(torch.tensor(record["horizon"] / 128.))
        if record["split"] != "train":
            b["pixels"].append(((pixels[-1] * .5 + .5).clamp(0, 1) * 255).round().byte())
            b["context_pixels"].append(((pixels[:-1] * .5 + .5).clamp(0, 1) * 255).round().byte())
        if (index + 1) % 32 == 0:
            print(f"encoded {index + 1}/{len(records)}", flush=True)
    data = {split: {k: torch.stack(v) for k, v in b.items() if v} for split, b in buffers.items()}
    data["metadata"] = dict(seed=args.seed, image_size=int(config.dataset.image_size),
                            context_size=int(config.dataset.context_size),
                            manifest_sha256=digest(output / "manifest.json"),
                            vae_path=str(config.model.tokenizer.model_path),
                            posterior="mode", datasets=args.datasets, horizons=args.horizons)
    save_torch(output / "samples.pt", data)
    print("Prepared", {s: len(data[s]["target"]) for s in buffers}, flush=True)


def check_data(config, data):
    for key in ("image_size", "context_size"):
        if int(config.dataset[key]) != data["metadata"][key]:
            raise ValueError(f"Data/teacher mismatch: {key}")
    if list(config.dataset.mean) != [.5]*3 or list(config.dataset.std) != [.5]*3:
        raise ValueError("This prepared data requires the standard NWM pixel normalization")


def take_batch(data, indices, device="cuda"):
    return {k: v[indices].to(device) for k, v in data.items() if k not in ("pixels", "context_pixels")}


def train(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest.pt"
    if latest.exists() and not args.resume:
        raise FileExistsError("Training checkpoint exists; pass --resume or choose a new directory")
    config = load_config(args.source_run)
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    check_data(config, data)
    teacher = load_weights(config, args.checkpoint, "cuda").requires_grad_(False)
    student = copy.deepcopy(teacher)
    fake = copy.deepcopy(teacher)
    # Restore trainability from a fresh model; keep fixed positional embeddings fixed.
    prototype = setup_model(config, torch.device("cpu"))
    trainable = {n for n, p in prototype.named_parameters() if p.requires_grad}
    del prototype
    for model in (student, fake):
        for n, p in model.named_parameters():
            p.requires_grad_(n in trainable)
        if args.activation_checkpointing:
            enable_checkpointing(model)
    critic = Critic(fake).cuda()
    g_params = [p for p in student.parameters() if p.requires_grad]
    d_params = [p for p in critic.parameters() if p.requires_grad]
    g_opt = torch.optim.AdamW(g_params, lr=args.generator_lr, betas=(0.9, 0.999), weight_decay=0)
    d_opt = torch.optim.AdamW(d_params, lr=args.critic_lr, betas=(0.9, 0.999), weight_decay=0)
    diffusion = setup_diffusion(config, False, torch.device("cuda"))
    schedule = FourStepSchedule(diffusion.alphas_cumprod).cuda()
    metadata = dict(method="DMD2-CDiT", timesteps=list(TIMESTEPS), teacher=str(Path(args.checkpoint).resolve()),
                    teacher_sha256=digest(args.checkpoint), data_sha256=digest(args.data),
                    # Preserve Hydra-only run-name resolvers such as ${now:...};
                    # model and tokenizer instantiation already resolve their own fields.
                    source_run=str(Path(args.source_run).resolve()), config=OmegaConf.to_container(config, resolve=False),
                    arguments=vars(args), trainable_parameters=sum(p.numel() for p in g_params),
                    adaptation="conditional CDiT token GAN head; full parameter epsilon models; no CFG; deterministic VAE means",
                    reference="https://github.com/tianweiy/DMD2/tree/8d8fa55633d47cfb81bbc7a892e7248f9518763f")
    start = 0
    if args.resume:
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        for key in ("teacher_sha256", "data_sha256"):
            if saved["metadata"][key] != metadata[key]:
                raise ValueError(f"Resume mismatch: {key}")
        for key in ("seed", "batch_size", "critic_ratio", "generator_lr", "critic_lr", "gan_weight"):
            if saved["metadata"]["arguments"][key] != vars(args)[key]:
                raise ValueError(f"Resume argument changed: {key}")
        student.load_state_dict(saved["student"])
        critic.load_state_dict(saved["critic"])
        g_opt.load_state_dict(saved["g_opt"])
        d_opt.load_state_dict(saved["d_opt"])
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        start = saved["iteration"]
        del saved
        # A crash can leave log rows newer than the last atomic checkpoint.
        log_path = output / "metrics.jsonl"
        if log_path.exists():
            kept = []
            for line in log_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row["iteration"] <= start:
                    kept.append(line)
            temporary = log_path.with_suffix(".tmp")
            temporary.write_text("\n".join(kept) + ("\n" if kept else ""))
            temporary.replace(log_path)
    write_json(output / "run.json", metadata)
    total = args.generator_updates * args.critic_ratio
    if start > total:
        raise ValueError("Requested update limit is older than the resumed checkpoint")
    print(f"TRAIN pid={os.getpid()} device={torch.cuda.get_device_name()} iterations={start}..{total} "
          f"generator_parameters={metadata['trainable_parameters']}", flush=True)
    begun = time.monotonic()
    for iteration in range(start, total):
        ids = torch.randint(len(data["train"]["target"]), (args.batch_size,))
        batch = take_batch(data["train"], ids)
        condition = condition_for(student, batch, config)
        target = batch["target"]
        selected = int(torch.randint(4, ()).item())
        update_g = (iteration + 1) % args.critic_ratio == 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated = schedule.sample(student, torch.randn_like(target), condition,
                                        stop_step=selected, last_grad=update_g)
            # Critic always sees detached generated images, never teacher trajectories.
            t = torch.randint(0, 1000, (len(ids),), device="cuda")
            eps = torch.randn_like(target)
            noisy = schedule.add_noise(generated.detach(), eps, t)
            predicted = fake(noisy, t, **condition)[:, :target.shape[1]]
            score_loss = F.mse_loss(predicted.float(), eps)
            gan_t = torch.randint(0, 1000, (len(ids),), device="cuda")
            gan_noise = torch.randn_like(target)
            fake_logits = critic.logits(schedule.add_noise(generated.detach(), gan_noise, gan_t), gan_t, condition)
            real_logits = critic.logits(schedule.add_noise(target, gan_noise, gan_t), gan_t, condition)
            d_gan = F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()
            d_loss = score_loss + args.gan_weight * d_gan
        if not torch.isfinite(d_loss):
            raise FloatingPointError(f"Non-finite critic loss at {iteration + 1}")
        d_opt.zero_grad(set_to_none=True)
        d_loss.backward()
        d_norm = torch.nn.utils.clip_grad_norm_(d_params, 1., error_if_nonfinite=True)
        d_opt.step()
        d_opt.zero_grad(set_to_none=True)
        row = dict(iteration=iteration + 1, generator_updates=(iteration + 1)//args.critic_ratio,
                   score_loss=score_loss.item(), discriminator_loss=d_gan.item(),
                   critic_grad_norm=float(d_norm), selected_step=selected)
        if update_g:
            for p in d_params:
                p.requires_grad_(False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    t = torch.randint(20, 981, (len(ids),), device="cuda")
                    noisy = schedule.add_noise(generated.detach(), torch.randn_like(target), t)
                    real_x0 = schedule.predict_x0(teacher, noisy, t, condition)
                    fake_x0 = schedule.predict_x0(fake, noisy, t, condition)
                dm, dm_norm = distribution_matching_loss(generated, real_x0, fake_x0)
                logits = critic.logits(schedule.add_noise(generated, gan_noise, gan_t), gan_t, condition)
                g_gan = F.softplus(-logits).mean()
                g_loss = dm + args.gan_weight * g_gan
            if not torch.isfinite(g_loss):
                raise FloatingPointError(f"Non-finite generator loss at {iteration + 1}")
            g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            g_norm = torch.nn.utils.clip_grad_norm_(g_params, 1., error_if_nonfinite=True)
            g_opt.step()
            g_opt.zero_grad(set_to_none=True)
            for p in d_params:
                p.requires_grad_(True)
            row.update(dm_loss=dm.item(), generator_gan_loss=g_gan.item(), dm_gradient_norm=float(dm_norm),
                       generator_grad_norm=float(g_norm), elapsed_seconds=time.monotonic()-begun,
                       peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if (iteration + 1) % (args.save_every * args.critic_ratio) == 0 or iteration + 1 == total:
            save_torch(latest, dict(student=student.state_dict(), critic=critic.state_dict(),
                g_opt=g_opt.state_dict(), d_opt=d_opt.state_dict(), iteration=iteration + 1,
                rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), numpy_rng=np.random.get_state(),
                python_rng=random.getstate(), metadata=metadata))
    # Separate inference-only weights: no optimizer, fake score, or GAN head needed.
    save_torch(output / "student.pt", dict(ema=student.state_dict(), metadata=metadata,
                                         generator_updates=args.generator_updates))
    print(f"COMPLETE generator_updates={args.generator_updates} elapsed={time.monotonic()-begun:.1f}s", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-run", required=True)
    prep.add_argument("--data-root", default=os.environ.get("NWM_DATA_ROOT"), required="NWM_DATA_ROOT" not in os.environ)
    prep.add_argument("--split-root", default=os.environ.get("NWM_SPLIT_ROOT"), required="NWM_SPLIT_ROOT" not in os.environ)
    prep.add_argument("--datasets", nargs="+", default=["recon", "sacson", "scand"])
    prep.add_argument("--train-per-dataset", type=int, default=128)
    prep.add_argument("--eval-per-dataset", type=int, default=8)
    prep.add_argument("--horizons", type=int, nargs="+", default=[4, 16])
    prep.add_argument("--output", required=True)
    train_parser = sub.add_parser("train")
    for p in (train_parser,):
        p.add_argument("--source-run", required=True)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--data", required=True)
        p.add_argument("--output", required=True)
    train_parser.add_argument("--generator-updates", type=int, default=200)
    train_parser.add_argument("--critic-ratio", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--generator-lr", type=float, default=1e-6)
    train_parser.add_argument("--critic-lr", type=float, default=1e-5)
    train_parser.add_argument("--gan-weight", type=float, default=.005)
    train_parser.add_argument("--save-every", type=int, default=25)
    train_parser.add_argument("--activation-checkpointing", action="store_true")
    train_parser.add_argument("--resume", action="store_true")
    for p in (prep, train_parser):
        p.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    for key in ("train_per_dataset", "eval_per_dataset", "generator_updates", "critic_ratio", "batch_size", "save_every"):
        if hasattr(args, key) and getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if hasattr(args, "horizons") and any(x < 1 for x in args.horizons):
        parser.error("horizons must be positive")
    return args


if __name__ == "__main__":
    args = parse_args()
    torch.set_num_threads(4)
    seed_all(args.seed)
    {"prepare": prepare, "train": train}[args.command](args)
