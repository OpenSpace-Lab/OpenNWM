#!/usr/bin/env python3
"""Replay the archived OpenNWM Table 1 protocol without editing its runtime.

The historical main run had four DDP ranks. Each ``infer`` invocation keeps a
real, one-rank process group and exposes a virtual rank only to the inference
entry point. This preserves the historical sampler, batch sizes, dataset order,
and rank seed while allowing one GPU to replay the four streams serially.
The selected GPU must retain its historical logical CUDA index: expose enough
devices and use --physical-local-rank for nonzero virtual ranks. Only that
selected device is used for computation.
Pixel equality against the archive must be checked before claiming exact replay.

Use the explicit nwm Python environment. ``--help`` and ``compare`` do not use a
GPU. All output paths are new, per-rank directories; resuming partially produced
images is deliberately unsupported because that would change the VAE RNG stream.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback


HOME_ROOT = Path("/file_system/vepfs/algorithm/dujun.nie")
NAS_ROOT = Path("/file_system/nas/algorithm/dujun.nie/nwm")
PYTHON = HOME_ROOT / "miniconda3/envs/nwm/bin/python"
ARCHIVE = NAS_ROOT / "results/nwm_benchmark/finalLAM_reset_checkpoint_sweep_20260922"
ARCHIVE_PREDICTIONS = ARCHIVE / "predictions/finalLAM-reset-joint0100000"
RUNTIMES = {
    "main": HOME_ROOT / "code/CompACT-eval-huron-fix",
    "rover": HOME_ROOT / "code/CompACT",
}
MODELS = {
    "baseline": NAS_ROOT / "compact/runs/nav15_pa_step60000_latentpt_reset_ft"
    / "nwm-latentpt-pixel-action-finalLAM-ft-reset",
    "dino": NAS_ROOT / "compact/runs/navanywherev1_latentpt_dino_ft/nwm-latentpt-dino-ft",
}
MAIN_DATASETS = (
    "recon", "scand", "huron", "tartan_drive", "go_stanford", "unitree_go2",
    "tum_rgbd", "uzh_fpv",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_capture(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return {"command": command, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def inspect_gpu(gpu, minimum_free_gib):
    result = run_capture([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits", "--id", gpu,
    ])
    if result["exit_code"]:
        raise RuntimeError(f"Cannot inspect requested GPU: {result}")
    rows = list(csv.reader(result["stdout"].strip().splitlines()))
    if len(rows) != 1:
        raise ValueError("Specify exactly one physical GPU index or UUID")
    free_gib = float(rows[0][-1].strip()) / 1024
    result.update(free_gib=free_gib, minimum_free_gib=minimum_free_gib)
    if free_gib < minimum_free_gib:
        raise RuntimeError(f"GPU {gpu} has {free_gib:.2f} GiB free; "
                           f"need {minimum_free_gib:.2f} GiB. No run outputs created.")
    return result


def runtime_fingerprint(runtime):
    relative_files = (
        "isolated_nwm_infer.py", "datasets.py", "distributed.py", "misc.py",
        "hydra_utils.py", "train_utils.py", "motion_condition.py",
        "conf/infer_config.yaml", "scripts/benchmark_reproducibility.py",
        "scripts/evaluate_nwm_predictions.py", ".runtime/final_lam_eval/isolated_entry.py",
    )
    return {
        "git_head": run_capture(["git", "rev-parse", "HEAD"], runtime),
        "git_status": run_capture(["git", "status", "--short"], runtime),
        "sha256": {name: sha256(runtime / name) for name in relative_files
                   if (runtime / name).is_file()},
    }


def datasets_for(part):
    return list(MAIN_DATASETS) if part == "main" else ["planetary_rover"]


def rank_directory(args):
    return args.output_root.resolve() / args.model / args.part / f"rank_{args.virtual_rank}"


def cuda_mapping(args, *, enforce_historical_index=True):
    """Validate the logical index before GPU inspection or output creation."""
    gpu = args.gpu.strip()
    visible = (args.cuda_visible_devices or gpu).split(",")
    visible = [item.strip() for item in visible]
    local_rank = args.physical_local_rank
    if local_rank is None:
        local_rank = 0
    if not all(visible) or len(set(visible)) != len(visible):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain distinct nonempty GPU identifiers")
    if not 0 <= local_rank < len(visible):
        raise ValueError("--physical-local-rank must index --cuda-visible-devices")
    if visible[local_rank] != gpu:
        raise ValueError("The selected CUDA index must map to the preflight --gpu")
    if enforce_historical_index and local_rank != args.virtual_rank:
        raise ValueError(
            f"Historical rank {args.virtual_rank} requires logical cuda:{args.virtual_rank}; "
            "set --cuda-visible-devices and --physical-local-rank explicitly")
    return {"physical_gpu": gpu, "physical_local_rank": local_rank,
            "cuda_visible_devices": ",".join(visible),
            "historical_cuda_index": args.virtual_rank,
            "historical_cuda_index_preserved": local_rank == args.virtual_rank}


def shard_cuda_mapping(metadata):
    """Read old rank-zero provenance and reject old nonzero cuda:0 shards."""
    visible = metadata["environment_overrides"]["CUDA_VISIBLE_DEVICES"].split(",")
    local_rank = metadata.get("physical_local_rank", 0)
    if not 0 <= local_rank < len(visible) or local_rank != metadata["virtual_rank"]:
        raise RuntimeError(
            f"Shard {metadata['virtual_rank']} did not preserve its historical CUDA index")
    selected = visible[local_rank]
    if metadata.get("physical_gpu", selected) != selected:
        raise RuntimeError("Shard GPU mapping contradicts the GPU preflight selection")
    return {"physical_local_rank": local_rank, "physical_gpu": selected,
            "cuda_visible_devices": ",".join(visible),
            "provenance": "explicit" if "physical_local_rank" in metadata else "legacy_rank0"}


def prepare(args, *, enforce_historical_cuda_index=True):
    world = args.virtual_world_size
    required_world = 4 if args.part == "main" else 1
    if world != required_world or not 0 <= args.virtual_rank < world:
        raise ValueError(f"{args.part} requires world size {required_world} and a valid rank")
    mapping = cuda_mapping(args, enforce_historical_index=enforce_historical_cuda_index)
    runtime = (args.runtime or RUNTIMES[args.part]).resolve()
    data_root = args.data_root.resolve()
    experiment = MODELS[args.model]
    checkpoint = experiment / "checkpoints/joint_0100000.pth.tar"
    for path in (runtime / "isolated_nwm_infer.py", args.python, checkpoint,
                 experiment / ".hydra/config.yaml", data_root):
        if not path.exists():
            raise FileNotFoundError(path)
    run_dir = rank_directory(args)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to mix or resume outputs: {run_dir}")
    # Free space is checked before predictions or copies are written.
    existing = run_dir.parent
    while not existing.exists():
        existing = existing.parent
    disk = shutil.disk_usage(existing)
    if disk.free < 2 * 1024**3:
        raise RuntimeError(f"Need at least 2 GiB free for this shard; available={disk.free}")
    minimum_free_gib = args.min_gpu_free_gib
    if minimum_free_gib is None:
        minimum_free_gib = 22 if args.part == "main" else 8
    gpu = inspect_gpu(args.gpu, minimum_free_gib)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "predictions").mkdir()
    split_paths = {}
    split_hashes = {}
    for dataset in datasets_for(args.part):
        split = "sacson" if dataset == "huron" else dataset
        original = runtime / "data_splits" / split / "test"
        copied = run_dir / "splits" / split / "test"
        # Only manifests used by the time protocol are needed. Do not copy old
        # generated cache/index files, which may be large and are not inputs.
        copied.mkdir(parents=True)
        for name in ("time.pkl", "traj_names.txt"):
            source = original / name
            if not source.exists():
                if name == "time.pkl":
                    raise FileNotFoundError(source)
                continue
            shutil.copy2(source, copied / name)
            split_hashes[f"{split}/{name}"] = sha256(source)
        split_paths[split] = str(copied)
    batch = 32 if args.part == "main" else 10
    count = 500 if args.part == "main" else 10
    hydra_args = [
        f"exp_dir={experiment}", "ckp=joint_0100000",
        f"output_dir={run_dir}", f"prediction_dir={run_dir / 'predictions'}",
        f"datasets_to_eval=[{','.join(datasets_for(args.part))}]",
        "eval_type=time", f"eval_expected_full_count={count}",
        "eval_diffusion_steps=250", f"batch_size={batch}", "num_workers=4",
        "pin_memory=false", "seed=0", "eval_len_traj_pred=16",
        "time_horizons_seconds=[4]", f"hydra.run.dir={run_dir / 'hydra'}",
    ]
    command = [str(args.python), "-m", "torch.distributed.run", "--standalone",
               "--nproc-per-node=1", str(Path(__file__).resolve()), "_entry",
               "--metadata", str(run_dir / "metadata.json"), "--", *hydra_args]
    metadata = {
        "schema_version": 1, "created_utc": now(), "state": "prepared",
        "protocol": "paper_table1_historical_direct_4s",
        "model": args.model, "part": args.part, "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "experiment_config_sha256": sha256(experiment / ".hydra/config.yaml"),
        "runtime": str(runtime), "runtime_fingerprint": runtime_fingerprint(runtime),
        "launcher_sha256": sha256(__file__), "python": str(args.python),
        "virtual_rank": args.virtual_rank, "virtual_world_size": world,
        **mapping,
        "physical_world_size": 1, "seed": 0, "rank_seed": args.virtual_rank,
        "datasets": datasets_for(args.part), "batch_size": batch,
        "num_workers": 4, "expected_full_count": count,
        "diffusion_steps": 250, "horizons_seconds": [4], "eval_len_traj_pred": 16,
        "vae_posterior": "historical stochastic encode, unchanged",
        "split_paths": split_paths, "split_sha256": split_hashes,
        "archive_predictions": str(ARCHIVE_PREDICTIONS),
        "data_root": str(data_root), "run_directory": str(run_dir),
        "log": str(run_dir / "inference.log"), "command": command,
        "shell_command": shlex.join(command), "free_disk_bytes": disk.free,
        "gpu_preflight": gpu,
        "environment_overrides": {
            "CUDA_VISIBLE_DEVICES": mapping["cuda_visible_devices"], "NWM_DATA_ROOT": str(data_root),
            "NWM_INDEX_ROOT": str(run_dir / "index_cache"),
            "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        },
        "equivalence_status": "unverified_until_archive_pixel_comparison",
    }
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def infer(args):
    metadata = prepare(args)
    run_dir = Path(metadata["run_directory"])
    environment = dict(os.environ, **metadata["environment_overrides"])
    print(json.dumps({key: metadata[key] for key in
                      ("runtime", "shell_command", "log", "gpu_preflight")}, indent=2),
          flush=True)
    with Path(metadata["log"]).open("w") as log:
        process = subprocess.Popen(metadata["command"], cwd=metadata["runtime"],
                                   env=environment, stdout=log, stderr=subprocess.STDOUT)
        metadata.update(state="running", pid=process.pid, started_utc=now())
        write_json(run_dir / "metadata.json", metadata)
        print(f"PID={process.pid}; log={metadata['log']}", flush=True)
        return_code = process.wait()
    metadata.update(state="completed" if return_code == 0 else "failed",
                    exit_code=return_code, finished_utc=now())
    write_json(run_dir / "metadata.json", metadata)
    print(f"Exit status: {return_code}; log tail:", flush=True)
    print("\n".join(Path(metadata["log"]).read_text(errors="replace").splitlines()[-30:]))
    return return_code


def entry(args):
    metadata = json.loads(args.metadata.read_text())
    # torchrun sets LOCAL_RANK=0 for its single real process. Restore only the
    # device selector, leaving real RANK=0 and WORLD_SIZE=1 unchanged. Legacy
    # rank-zero metadata (including the later corrected protocol) stays valid.
    local_rank = metadata.get("physical_local_rank", 0)
    os.environ["LOCAL_RANK"] = str(local_rank)
    runtime = Path(metadata["runtime"])
    os.chdir(runtime)
    sys.path.insert(0, str(runtime))
    import isolated_nwm_infer as inference

    real_distributed = inference.dist

    class VirtualInferenceDistributed:
        # Only this module reference is replaced. torch.distributed and the
        # actual DDP process group remain world_size=1, rank=0.
        def __getattr__(self, name):
            return getattr(real_distributed, name)

        def init_distributed(self, *positional, **keyword):
            _, _, device, initialized = real_distributed.init_distributed(*positional, **keyword)
            if device != local_rank:
                raise RuntimeError(f"Runtime selected CUDA index {device}; expected {local_rank}")
            write_json(Path(metadata["run_directory"]) / "execution_device.json", {
                "created_utc": now(), "physical_local_rank": device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "real_rank": real_distributed.get_rank(),
                "real_world_size": real_distributed.get_world_size(),
                "virtual_rank": metadata["virtual_rank"],
                "virtual_world_size": metadata["virtual_world_size"],
            })
            return self.get_world_size(), self.get_rank(), device, initialized

        def get_rank(self):
            return metadata["virtual_rank"]

        def get_world_size(self):
            return metadata["virtual_world_size"]

    inference.dist = VirtualInferenceDistributed()
    original_dataset = inference.EvalDataset

    def isolated_dataset(*positional, **keyword):
        split = Path(keyword["data_split_folder"]).parent.name
        copied = Path(metadata["split_paths"][split])
        keyword["data_split_folder"] = str(copied)
        if keyword.get("predefined_index"):
            keyword["predefined_index"] = str(copied / Path(keyword["predefined_index"]).name)
        # Equivalent to final_lam_eval/isolated_entry.py's pin for HuRoN:
        # sacson/test/time.pkl, now copied into the output directory.
        return original_dataset(*positional, **keyword)

    inference.EvalDataset = isolated_dataset
    hydra_args = args.hydra_args
    if hydra_args and hydra_args[0] == "--":
        hydra_args = hydra_args[1:]
    # Imported Hydra entry points resolve a relative config_path as a Python
    # package. The historical conf directory is not a package, so make the
    # original script's filesystem config location explicit.
    sys.argv = [str(runtime / "isolated_nwm_infer.py"), "--config-path",
                str(runtime / "conf"), *hydra_args]
    inference.main()
    return 0


def prediction_files(metadata):
    root = Path(metadata["run_directory"]) / "predictions"
    expected = {
        Path(dataset) / "time" / f"id_{index}" / "4.png"
        for dataset in metadata["datasets"]
        for index in range(metadata["virtual_rank"], metadata["expected_full_count"],
                           metadata["virtual_world_size"])
    }
    actual = {path.relative_to(root) for path in root.rglob("*.png")}
    if actual != expected:
        raise RuntimeError(f"Incomplete/unexpected predictions: missing={len(expected - actual)}, "
                           f"extra={len(actual - expected)}")
    return root, sorted(expected)


def compare(args):
    from PIL import Image, ImageChops
    run_dir = rank_directory(args)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if metadata["model"] != "baseline":
        raise ValueError("Archive equivalence is meaningful only for --model baseline")
    root, files = prediction_files(metadata)
    archive = args.archive_predictions.resolve()
    counts = {}
    mismatches = []
    for relative in files:
        dataset = relative.parts[0]
        counts.setdefault(dataset, {"compared": 0, "pixel_identical": 0, "byte_identical": 0})
        predicted = root / relative
        historical = archive / relative
        if not historical.is_file():
            raise FileNotFoundError(historical)
        with Image.open(predicted) as left, Image.open(historical) as right:
            same = (left.mode == right.mode and left.size == right.size
                    and ImageChops.difference(left, right).getbbox() is None)
        byte_same = sha256(predicted) == sha256(historical)
        counts[dataset]["compared"] += 1
        counts[dataset]["pixel_identical"] += int(same)
        counts[dataset]["byte_identical"] += int(byte_same)
        if not same:
            mismatches.append(str(relative))
    report = {
        "created_utc": now(), "archive": str(archive), "run_directory": str(run_dir),
        "counts": counts, "all_pixels_identical": not mismatches,
        "mismatched_files": mismatches,
    }
    write_json(run_dir / "archive_pixel_comparison.json", report)
    metadata["equivalence_status"] = "all_pixels_identical" if not mismatches else "pixel_mismatch"
    write_json(run_dir / "metadata.json", metadata)
    print(json.dumps(report, indent=2))
    return 0 if not mismatches else 1


def merge(args):
    part_dir = args.output_root.resolve() / args.model / args.part
    expected_world = 4 if args.part == "main" else 1
    shards = []
    for rank in range(expected_world):
        metadata = json.loads((part_dir / f"rank_{rank}" / "metadata.json").read_text())
        if metadata["state"] != "completed" or metadata.get("exit_code") != 0:
            raise RuntimeError(f"Shard {rank} did not complete successfully")
        if metadata["virtual_rank"] != rank or metadata["virtual_world_size"] != expected_world:
            raise RuntimeError(f"Shard {rank} rank configuration does not match protocol")
        shard_cuda_mapping(metadata)
        if metadata.get("physical_local_rank") is not None:
            executed = json.loads((part_dir / f"rank_{rank}" / "execution_device.json").read_text())
            if (executed["physical_local_rank"] != rank or executed["real_rank"] != 0
                    or executed["real_world_size"] != 1
                    or executed["cuda_visible_devices"] != metadata["environment_overrides"]["CUDA_VISIBLE_DEVICES"]):
                raise RuntimeError(f"Shard {rank} execution device does not match recorded protocol")
        if args.model == "baseline":
            comparison = json.loads((part_dir / f"rank_{rank}" / "archive_pixel_comparison.json").read_text())
            if metadata.get("equivalence_status") != "all_pixels_identical" or not comparison["all_pixels_identical"]:
                raise RuntimeError(f"Baseline shard {rank} has not passed archive pixel equality")
        if shards:
            for key in ("model", "part", "checkpoint", "checkpoint_bytes", "checkpoint_mtime_ns",
                        "experiment_config_sha256", "runtime_fingerprint", "datasets",
                        "batch_size", "num_workers", "split_sha256", "data_root"):
                if metadata[key] != shards[0][0][key]:
                    raise RuntimeError(f"Shard {rank} differs in {key}")
        root, files = prediction_files(metadata)
        shards.append((metadata, root, files))
    merged = part_dir / "merged_predictions"
    if merged.exists():
        raise FileExistsError(f"Refusing to mix merged outputs: {merged}")
    merged.mkdir()
    for _, root, files in shards:
        for relative in files:
            target = merged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(root / relative)
    write_json(part_dir / "merge_manifest.json", {
        "created_utc": now(), "merged_predictions": str(merged),
        "shards": [metadata["run_directory"] for metadata, _, _ in shards],
        "cuda_mappings": [shard_cuda_mapping(metadata) for metadata, _, _ in shards],
        "count": sum(len(files) for _, _, files in shards),
    })
    print(merged)
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = result.add_subparsers(dest="operation", required=True)
    for name in ("infer", "compare", "merge"):
        current = subparsers.add_parser(name)
        current.add_argument("--model", choices=MODELS, required=True)
        current.add_argument("--part", choices=RUNTIMES, default="main")
        current.add_argument("--output-root", type=Path, required=True)
        if name != "merge":
            current.add_argument("--virtual-rank", type=int, default=0)
        if name == "infer":
            current.add_argument("--virtual-world-size", type=int, default=4,
                                 help="4 for main; explicitly set 1 for rover")
            current.add_argument("--gpu", required=True, help="One physical GPU index or UUID")
            current.add_argument("--cuda-visible-devices",
                                 help="Explicit visible GPU list; only the selected index is used")
            current.add_argument("--physical-local-rank", type=int,
                                 help="Selected logical CUDA index; must equal virtual rank (default: 0)")
            current.add_argument("--min-gpu-free-gib", type=float,
                                 help="Fail before writing below this free memory (main:22, rover:8)")
            current.add_argument("--data-root", type=Path, required=True)
            current.add_argument("--python", type=Path, default=PYTHON)
            current.add_argument("--runtime", type=Path,
                                 help="Defaults to historical runtime for the selected part")
        elif name == "compare":
            current.add_argument("--archive-predictions", type=Path, default=ARCHIVE_PREDICTIONS)
    internal = subparsers.add_parser("_entry", help=argparse.SUPPRESS)
    internal.add_argument("--metadata", type=Path, required=True)
    internal.add_argument("hydra_args", nargs=argparse.REMAINDER)
    return result


def main():
    args = parser().parse_args()
    return {"infer": infer, "compare": compare, "merge": merge, "_entry": entry}[args.operation](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
