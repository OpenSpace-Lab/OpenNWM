"""Small persistent training/evaluation state helpers."""
from __future__ import annotations

import json
import os
import uuid

import torch


def atomic_save(value, path):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def point_checkpoint(target, path):
    path = os.path.abspath(os.fspath(path))
    temporary = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        os.symlink(os.path.relpath(target, os.path.dirname(path)), temporary)
        os.replace(temporary, path)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)


def log_metrics(run_dir, values):
    """Append readable local curves even when WandB is disabled."""
    with open(os.path.join(run_dir, "metrics.jsonl"), "a", encoding="utf-8") as stream:
        stream.write(json.dumps(values, allow_nan=False) + "\n")
    if any(key.startswith("eval/") for key in values):
        plot_metrics(run_dir)


def prune_periodic(checkpoint_dir, keep=3):
    """Prune only known numbered snapshots in this run, preserving all links."""
    import re
    from pathlib import Path
    if int(keep) <= 0:
        return
    root = Path(checkpoint_dir)
    protected = {p.resolve() for p in root.iterdir() if p.is_symlink()}
    numbered = [p for p in root.iterdir()
                if re.fullmatch(r"(?:(?:pretrain|warmup|joint)_)?[0-9]{7,}\.pth\.tar", p.name)]
    phases = {"pretrain": 0, "warmup": 1, "joint": 2}
    numbered.sort(key=lambda p: (p.stat().st_mtime_ns, phases.get(p.name.split("_")[0], 0),
                                int(re.search(r"([0-9]+)\.pth", p.name).group(1))), reverse=True)
    for path in numbered[int(keep):]:
        if path.resolve() not in protected:
            path.unlink()


def plot_metrics(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(os.path.join(run_dir, "metrics.jsonl"), encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    keys = sorted({key for row in rows for key in row
                   if key.startswith("eval/") or key in {"train/loss", "train/total_loss"}})
    if not keys:
        return
    figure, axes = plt.subplots(len(keys), 1, figsize=(8, 3 * len(keys)), squeeze=False)
    for axis, key in zip(axes[:, 0], keys):
        points = [(row["step"], row[key]) for row in rows if key in row]
        axis.plot(*zip(*points), marker=".")
        axis.set(xlabel="Optimizer step", ylabel=key)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(os.path.join(run_dir, "curves.png"), dpi=130)
    plt.close(figure)


def output_identity(directory, config, *, checkpoint=None, name="run"):
    """Reject stale cached results after a checkpoint/config change (stat only)."""
    from omegaconf import OmegaConf
    fields = ("seed", "batch_size", "eval_type", "input_fps", "num_sec_eval",
              "time_horizons_seconds", "eval_diffusion_steps", "eval_context_size",
              "eval_len_traj_pred", "rollout_fps_values", "use_efficient_rollout",
              "single_image_context", "eval_sample_indices", "evaluation_datasets",
              "trajectory_eval_distance", "trajectory_eval_len_traj_pred",
              "traj_stride", "opt_steps", "planning_sample_seed", "model", "motion_condition")
    def plain(value):
        return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    identity = {key: plain(config[key]) for key in fields if key in config}
    # Omit legacy defaults so existing DDPM results remain resumable. Different
    # accelerated samplers/eta must never reuse a DDPM (or each other's) cache.
    sampler = config.get("inference_sampler", "ddpm")
    if sampler != "ddpm":
        identity["inference_sampler"] = sampler
        if sampler == "ddim":
            eta = config.get("ddim_eta")
            identity["ddim_eta"] = 0.0 if eta is None else eta
    if checkpoint is not None:
        stat = os.stat(checkpoint)
        identity["checkpoint"] = {"path": os.path.realpath(checkpoint),
                                  "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    path = os.path.join(directory, f".{name}.identity.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as stream:
            if json.load(stream) != identity:
                raise ValueError(f"Weights or evaluation settings changed for {directory}; choose a new output directory")
    else:
        atomic_json(identity, path)
    return identity


def atomic_json(value, path):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
