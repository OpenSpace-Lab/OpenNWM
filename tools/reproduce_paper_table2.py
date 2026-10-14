#!/usr/bin/env python3
"""Auditable replay of the archived OpenNWM Table 2 navigation protocol.

This launcher executes the original planning evaluator without changing it.
Each window has its original seed, so index sharding and resume are supported.
Candidate microbatch changes are a separate protocol: stochastic VAE encoding
uses the global CUDA RNG, even though candidate and diffusion noise use keys.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


HOME_ROOT = Path("/file_system/vepfs/algorithm/dujun.nie")
NAS_ROOT = Path("/file_system/nas/algorithm/dujun.nie/nwm")
PYTHON = HOME_ROOT / "miniconda3/envs/nwm/bin/python"
RUNTIME = HOME_ROOT / "code/CompACT_a800_original_seed0_v1"
EXPECTED_PLANNER_SHA = "cff9deb6b64c6064cd6672aae1891c07f947e033114fae9d0d07534a92b3c080"
OUTPUT_ROOT = NAS_ROOT / "opennwm_reproduction/dino_main_tables_20261007"
ARCHIVE = NAS_ROOT / "results/nwm_benchmark/a800_full_evaluations/full_8xa800_seed0_v1"
MODELS = {
    "baseline": NAS_ROOT / "compact/runs/nav15_pa_step60000_latentpt_reset_ft"
    / "nwm-latentpt-pixel-action-finalLAM-ft-reset",
    "dino": NAS_ROOT / "compact/runs/navanywherev1_latentpt_dino_ft/nwm-latentpt-dino-ft",
}
DATASETS = ("recon", "scand", "unitree_go2")
EVAL_NAME = "CEM_N80_K5_RS1_rep3_OPT1_COST-lpips-RECON-True"
ARCHIVE_MEANS = {
    "recon": {"ate": 1.121662379, "rpe_trans": 0.355802949},
    "scand": {"ate": 0.981816295, "rpe_trans": 0.277794078},
    "unitree_go2": {"ate": 2.886700621, "rpe_trans": 0.650186650},
}
PAPER_VALUES = {
    "recon": {"ate": 1.12, "rpe_trans": 0.35},
    "scand": {"ate": 0.98, "rpe_trans": 0.27},
    "unitree_go2": {"ate": 2.89, "rpe_trans": 0.65},
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def capture(command, cwd=None):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    return {"command": command, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def selected_indices(args):
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
    if args.indices is None:
        indices = list(range(100))
    elif ":" in args.indices:
        start, stop = args.indices.split(":")
        indices = list(range(int(start), int(stop)))
    else:
        indices = [int(value) for value in args.indices.split(",")]
    if (not indices or len(set(indices)) != len(indices)
            or any(value < 0 or value >= 100 for value in indices)):
        raise ValueError("indices must be nonempty, unique positions in [0,100)")
    indices = indices[args.shard_index::args.num_shards]
    if not indices:
        raise ValueError("This shard has no requested windows")
    return indices


def protocol_directory(args):
    return args.output_root.resolve() / "table2" / args.model / args.dataset / f"mb{args.microbatch}"


def source_fingerprint(runtime):
    source_files = [
        "planning_eval.py", "isolated_nwm_infer.py", "isolated_nwm_eval.py",
        "datasets.py", "tokenizer_wrapper.py", "models.py", "motion_condition.py",
        "train_utils.py", "hydra_utils.py", "misc.py", "distributed.py",
        "conf/plan_config.yaml", "scripts/benchmark_reproducibility.py",
    ]
    source_files.extend(str(path.relative_to(runtime))
                        for path in sorted((runtime / "diffusion").glob("*.py")))
    return {name: sha256(runtime / name) for name in source_files
            if (runtime / name).is_file()}


def fixed_parameters():
    return {
        "num_samples": 80, "topk": 5, "horizon": 8, "rollout_stride": 1,
        "repeat": 3, "optimization_steps": 1, "ddpm_steps": 250,
        "seed": 42, "planning_sample_seed": 42, "batch_size": 1,
        "num_workers": 4, "cost": "lpips_alex", "cost_with_reconstruction": True,
        "precision": "bfloat16 autocast, TF32 enabled", "compile": True,
    }


def prepare(args, lock_held=False):
    indices = selected_indices(args)
    if not 1 <= args.microbatch <= 80:
        raise ValueError("microbatch must be in [1,80]")
    if args.microbatch != 80 and not args.accept_microbatch_change:
        raise ValueError("Historical microbatch is 80. Changing it changes VAE RNG "
                         "mapping; explicitly pass --accept-microbatch-change only "
                         "for an authorized separate-protocol comparison.")
    runtime = args.runtime.resolve()
    experiment = MODELS[args.model]
    checkpoint = experiment / "checkpoints/joint_0100000.pth.tar"
    for path in (args.python, runtime / "planning_eval.py", checkpoint,
                 experiment / ".hydra/config.yaml", args.data_root):
        if not path.exists():
            raise FileNotFoundError(path)
    planner_sha = sha256(runtime / "planning_eval.py")
    if planner_sha != EXPECTED_PLANNER_SHA:
        raise ValueError(f"Archived planner fingerprint changed: {planner_sha}")
    protocol_dir = protocol_directory(args)
    index_key = hashlib.sha256(json.dumps(indices).encode()).hexdigest()[:12]
    job_dir = protocol_dir / "shards" / index_key
    metadata_path = job_dir / "metadata.json"
    lock_path = job_dir / "active.lock"
    if lock_path.exists() and not lock_held:
        with lock_path.open("r") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Shard is already locked: {lock_path}") from error
    existing = protocol_dir
    while not existing.exists():
        existing = existing.parent
    disk = shutil.disk_usage(existing)
    if disk.free < 1024**3:
        raise RuntimeError("Need at least 1 GiB free for logs and compiler cache")
    # Per-shard copies prevent generated invalid-index audit files from writing
    # into source splits or colliding with another evaluator.
    job_dir.mkdir(parents=True, exist_ok=True)
    copied_split = job_dir / "data_splits" / args.dataset / "test"
    copied_split.mkdir(parents=True, exist_ok=True)
    split_hashes = {}
    for name in ("navigation_eval.pkl", "rollout_traj_names.txt"):
        source = runtime / "data_splits" / args.dataset / "test" / name
        copied = copied_split / name
        original_sha = sha256(source)
        if copied.exists() and sha256(copied) != original_sha:
            raise ValueError(f"Existing copied split differs from source: {copied}")
        if not copied.exists():
            shutil.copy2(source, copied)
        split_hashes[name] = original_sha
    fingerprint = source_fingerprint(runtime)
    hydra_args = [
        f"exp_dir={experiment}", "ckp=joint_0100000",
        f"datasets_to_eval=[{args.dataset}]", f"hydra.run.dir={job_dir / 'hydra'}",
        f"output_dir={protocol_dir / 'outputs'}",
        "planning_sample_indices=[" + ",".join(map(str, indices)) + "]",
        "planning_write_aggregate=false", "batch_size=1", "num_workers=4",
        "num_samples=80", "topk=5", "rollout_stride=1", "opt_steps=1",
        "num_repeat_eval=3", "seed=42", "+planning_sample_seed=42",
        f"planning_microbatch_size={args.microbatch}", "cost_fn=lpips",
        "compute_cost_with_recon=true", "save_preds=false", "plot=false",
        "resume_planning_samples=true",
        f"evaluation_datasets.{args.dataset}.test={copied_split}",
        f"evaluation_datasets.{args.dataset}.navigation_index={copied_split / 'navigation_eval.pkl'}",
    ]
    command = [str(args.python), "-m", "torch.distributed.run", "--standalone",
               "--nproc-per-node=1", str(runtime / "planning_eval.py"), *hydra_args]
    manifest = {
        "schema_version": 1, "created_utc": now(), "state": "prepared",
        "protocol": ("paper_table2_historical" if args.microbatch == 80
                     else "table2_changed_candidate_microbatch"),
        "model": args.model, "dataset": args.dataset, "indices": indices,
        "full_dataset_count": 100, "checkpoint": str(checkpoint), "weight_key": "ema",
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "experiment_config_sha256": sha256(experiment / ".hydra/config.yaml"),
        "runtime": str(runtime), "source_sha256": fingerprint,
        "launcher_sha256": sha256(__file__), "split_sha256": split_hashes,
        "copied_split": str(copied_split), "python": str(args.python),
        "protocol_directory": str(protocol_dir), "working_directory": str(job_dir),
        "metadata_path": str(metadata_path), "log": str(job_dir / "planning.log"),
        "command": command, "shell_command": shlex.join(command),
        "microbatch_size": args.microbatch, "free_disk_bytes": disk.free,
        "fixed_parameters": fixed_parameters(),
        "randomness": {
            "window_seed": "42 + sample_index",
            "candidate_and_ddpm_noise": "stable semantic sample keys",
            "vae_posterior": "global CUDA RNG; microbatch shape must remain fixed",
        },
        "environment_overrides": {
            "CUDA_VISIBLE_DEVICES": args.gpu, "NWM_DATA_ROOT": str(args.data_root.resolve()),
            "NWM_INDEX_ROOT": str(job_dir / "index_cache"), "OMP_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
        },
        "equivalence_status": "unverified_until_per_sample_archive_comparison",
    }
    protocol_path = protocol_dir / "protocol.json"
    # Every shard writing into a common sample directory must use the same
    # checkpoint, source, population, and parameters, including on resume.
    protocol = {key: manifest[key] for key in
                ("protocol", "model", "dataset", "checkpoint", "checkpoint_bytes",
                 "checkpoint_mtime_ns", "experiment_config_sha256", "source_sha256",
                 "split_sha256", "microbatch_size", "fixed_parameters")}
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError(f"Cross-shard protocol mismatch: {protocol_path}")
    else:
        write_json(protocol_path, protocol)
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        for key in ("protocol", "model", "dataset", "indices", "checkpoint",
                    "checkpoint_bytes", "checkpoint_mtime_ns", "experiment_config_sha256",
                    "source_sha256", "split_sha256", "microbatch_size", "fixed_parameters"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Resume provenance mismatch for {key}: {metadata_path}")
        # Preserve completed-run history when preparing a resume.
        manifest["previous_attempts"] = previous.get("previous_attempts", [])
        if "started_utc" in previous:
            manifest["previous_attempts"].append({key: previous.get(key) for key in
                ("started_utc", "finished_utc", "pid", "exit_code", "state")})
    write_json(metadata_path, manifest)
    return manifest


def run(args):
    index_key = hashlib.sha256(json.dumps(selected_indices(args)).encode()).hexdigest()[:12]
    job_dir = protocol_directory(args) / "shards" / index_key
    job_dir.mkdir(parents=True, exist_ok=True)
    lock = (job_dir / "active.lock").open("a+")
    # The kernel releases flock even if this launcher crashes; the persistent
    # lock file is only a PID hint and never requires deleting for resume.
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    try:
        manifest = prepare(args, lock_held=True)
        metadata_path = Path(manifest["metadata_path"])
        gpu = capture(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free",
                       "--format=csv,noheader,nounits", "--id", args.gpu])
        if gpu["exit_code"]:
            raise RuntimeError(f"Cannot inspect requested GPU: {gpu}")
        manifest["gpu_preflight"] = gpu
        print(json.dumps({key: manifest[key] for key in
                          ("working_directory", "shell_command", "log", "gpu_preflight")},
                         indent=2), flush=True)
        with Path(manifest["log"]).open("a") as log:
            log.write(f"\n=== launch {now()} ===\n")
            log.flush()
            process = subprocess.Popen(manifest["command"], cwd=manifest["working_directory"],
                                       env=dict(os.environ, **manifest["environment_overrides"]),
                                       stdout=log, stderr=subprocess.STDOUT)
            manifest.update(state="running", pid=process.pid, started_utc=now())
            write_json(metadata_path, manifest)
            print(f"PID={process.pid}; log={manifest['log']}", flush=True)
            return_code = process.wait()
        manifest.update(state="completed" if return_code == 0 else "failed",
                        exit_code=return_code, finished_utc=now())
        write_json(metadata_path, manifest)
        print(f"Exit status: {return_code}; log tail:", flush=True)
        print("\n".join(Path(manifest["log"]).read_text(errors="replace").splitlines()[-25:]))
        return return_code
    finally:
        lock.close()


def cpu_check(args):
    manifest = prepare(args)
    os.environ.update(manifest["environment_overrides"])
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.chdir(manifest["working_directory"])
    sys.path.insert(0, manifest["runtime"])
    import torch
    from omegaconf import OmegaConf
    from hydra_utils import merge_configs, load_experiment_config
    from train_utils import setup_model, validate_model_context_sizes
    from planning_eval import get_dataset_eval
    import datasets as runtime_datasets

    torch.set_num_threads(2)
    # The CPU preflight is outside Hydra's decorated main. Absolute split/data
    # paths are retained; supply its equivalent cwd without affecting GPU runs.
    runtime_datasets.get_original_cwd = os.getcwd
    plan_config = OmegaConf.load(Path(manifest["runtime"]) / "conf/plan_config.yaml")
    plan_config.pop("defaults", None)
    plan_config.pop("hydra", None)
    config = merge_configs(load_experiment_config(str(MODELS[args.model])), plan_config)
    config.exp_dir = str(MODELS[args.model])
    config.ckp = "joint_0100000"
    config.evaluation_datasets[args.dataset].test = manifest["copied_split"]
    config.evaluation_datasets[args.dataset].navigation_index = str(
        Path(manifest["copied_split"]) / "navigation_eval.pkl")
    validate_model_context_sizes(config, "eval_context_size", "trajectory_eval_context_size")
    if int(config.model.diffusion.eval_timestep_respacing) != 250:
        raise ValueError("The checkpoint config does not use the archived 250 DDPM steps")
    model = setup_model(config, torch.device("cpu"))
    checkpoint = torch.load(manifest["checkpoint"], map_location="cpu", mmap=True,
                            weights_only=False)
    status = model.load_state_dict(checkpoint["ema"], strict=True, assign=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del checkpoint, model
    dataset = get_dataset_eval(config, args.dataset, predefined_index=True)
    sample_shapes = {}
    for index in manifest["indices"]:
        sample = dataset[index]
        shapes = [list(value.shape) for value in sample]
        if shapes != [[1], [4, 3, 224, 224], [1, 3, 224, 224], [8, 3], [1, 3]]:
            raise ValueError(f"Unexpected sample shapes at {index}: {shapes}")
        if not all(torch.isfinite(value).all().item() for value in sample):
            raise ValueError(f"Nonfinite values in sample {index}")
        sample_shapes[str(index)] = shapes
    result = {"created_utc": now(), "model": args.model, "dataset": args.dataset,
              "strict_load": str(status), "parameter_count": parameter_count,
              "dataset_count": len(dataset), "checked_indices": manifest["indices"],
              "sample_shapes": sample_shapes, "gpu_initialized": torch.cuda.is_initialized(),
              "metadata": manifest["metadata_path"]}
    path = Path(manifest["working_directory"]) / "cpu_check.json"
    write_json(path, result)
    print(json.dumps({**result, "sample_shapes": f"{len(sample_shapes)} checked; see {path}"}, indent=2))
    return 0


def aggregate(args):
    protocol_dir = protocol_directory(args)
    samples = protocol_dir / "outputs" / args.dataset / EVAL_NAME / "sample_metrics"
    keys = ("ate", "rpe_trans", "pos_diff_norm", "yaw_diff_norm")
    values = {}
    for path in sorted(samples.glob("*.json")):
        index = int(path.stem)
        record = json.loads(path.read_text())
        if not 0 <= index < 100 or set(record) != set(keys):
            raise ValueError(f"Unexpected sample metric record: {path}")
        if any(not math.isfinite(record[key]) for key in keys):
            raise ValueError(f"Nonfinite sample metrics: {path}")
        values[index] = record
    missing = sorted(set(range(100)) - set(values))
    means = {key: sum(item[key] for item in values.values()) / len(values)
             for key in keys} if values else {}
    archive = json.loads(args.archive_report.read_text())
    archive_group = archive["groups"][f"navigation/opennwm-finalLAM-100k/{args.dataset}/cem80"]
    archived_metrics = archive_group["metrics"]
    archived_paths = {int(Path(source).stem): Path(source)
                      for source in archived_metrics["sample_sources"]}
    if set(archived_paths) != set(range(100)):
        raise ValueError("Archive sample_sources must contain exactly the 100 original windows")
    if args.archive_samples is not None:
        archived_paths = {index: args.archive_samples / f"{index:06d}.json"
                          for index in range(100)}
    result = {
        "created_utc": now(), "model": args.model, "dataset": args.dataset,
        "microbatch_size": args.microbatch, "count": len(values), "missing_indices": missing,
        "complete": not missing, "means": means,
        "paper_values": PAPER_VALUES[args.dataset],
        "archived_reference_means": {key: archived_metrics[f"{args.dataset}_{key}"]
                                      for key in keys},
        "archive_report": str(args.archive_report),
        "rounding_note": "Paper RECON/SCAND RPE .35/.27 do not use nearest rounding of .355802949/.277794078.",
        "historical_equivalence": "not established by aggregate means alone",
        "sample_metrics": str(samples),
    }
    deltas = []
    for index, record in values.items():
        archived_path = archived_paths[index]
        archived = json.loads(archived_path.read_text())
        if set(archived) != set(keys) or any(not math.isfinite(archived[key]) for key in keys):
            raise ValueError(f"Unexpected archived sample metric: {archived_path}")
        deltas.append({"index": index, "archive_source": str(archived_path),
                       "new": record, "archived": archived,
                       "difference": {key: record[key] - float(archived[key]) for key in keys}})
    result["per_sample_archive_comparison"] = {
        "kind": "reproduction_check" if args.model == "baseline" else "dino_minus_archived_baseline",
        "count": len(deltas),
        "max_absolute_difference": {key: max((abs(row["difference"][key]) for row in deltas), default=None)
                                    for key in keys},
        "mean_difference": {key: (sum(row["difference"][key] for row in deltas) / len(deltas)
                                  if deltas else None) for key in keys},
        "exact_equal_sample_count": sum(all(row["difference"][key] == 0 for key in keys)
                                        for row in deltas),
    }
    if args.model == "baseline" and args.microbatch == 80 and not missing:
        result["historical_equivalence"] = (
            "all_four_saved_metrics_exactly_equal_on_all_100_windows"
            if result["per_sample_archive_comparison"]["exact_equal_sample_count"] == 100
            else "replay_metrics_differ_from_archive")
    protocol_dir.mkdir(parents=True, exist_ok=True)
    write_json(protocol_dir / "per_sample_archive_comparison.json", deltas)
    write_json(protocol_dir / "aggregate.json", result)
    print(json.dumps(result, indent=2))
    return 0 if not missing else 2


def verify_subset(args):
    """Validate provenance, coverage, and saved metrics for selected windows.

    This is a CPU provenance/JSON check. It neither prepares a new inference
    shard nor promotes a subset comparison to a full reproduction claim.
    Metric differences fail by default; an explicit option permits them for
    the authorized L20 comparison while retaining every recorded difference.
    """
    protocol_dir = protocol_directory(args)
    allow_differences = args.allow_metric_differences
    result = {
        "created_utc": now(), "state": "failed", "exit_code": 1,
        "model": args.model, "dataset": args.dataset, "microbatch_size": args.microbatch,
        "count": 0, "requested_indices": [], "per_sample_deltas": [],
        "all_metrics_exact": False, "provenance_valid": False, "coverage_valid": False,
        "allow_metric_differences": allow_differences,
        "acceptance_policy": ("provenance_and_coverage_allow_metric_differences"
                              if allow_differences else "exact_metrics_required"),
        "verification_outcome": "not_verified",
        "scope": "Only requested windows are verified; this is not a full-dataset reproduction claim.",
        "archive_report": str(args.archive_report.resolve()),
    }
    try:
        if args.model != "baseline" or args.microbatch != 80:
            raise ValueError("verify-subset requires --model baseline and --microbatch 80")
        if args.archive_samples is not None:
            raise ValueError("verify-subset uses the archived report's original sample_sources only")
        indices = selected_indices(args)
        result["requested_indices"] = indices
        protocol_path = protocol_dir / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        runtime = args.runtime.resolve()
        experiment = MODELS["baseline"]
        checkpoint = experiment / "checkpoints/joint_0100000.pth.tar"
        fingerprint = source_fingerprint(runtime)
        if fingerprint.get("planning_eval.py") != EXPECTED_PLANNER_SHA:
            raise ValueError("Original planning evaluator fingerprint does not match the pinned archive")
        expected = {
            "protocol": "paper_table2_historical", "model": "baseline", "dataset": args.dataset,
            "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
            "experiment_config_sha256": sha256(experiment / ".hydra/config.yaml"),
            "source_sha256": fingerprint,
            "split_sha256": {name: sha256(runtime / "data_splits" / args.dataset / "test" / name)
                              for name in ("navigation_eval.pkl", "rollout_traj_names.txt")},
            "microbatch_size": 80, "fixed_parameters": fixed_parameters(),
        }
        if protocol != expected:
            mismatches = [key for key in set(protocol) | set(expected)
                          if protocol.get(key) != expected.get(key)]
            raise ValueError(f"Protocol provenance mismatch: {sorted(mismatches)}")
        # Every requested window must be attributable to a successful shard
        # with the same protocol and intact private copies of the split files.
        covered_by = {index: [] for index in indices}
        for path in sorted((protocol_dir / "shards").glob("*/metadata.json")):
            metadata = json.loads(path.read_text())
            overlap = set(indices).intersection(metadata.get("indices", []))
            if not overlap or metadata.get("state") != "completed" or metadata.get("exit_code") != 0:
                continue
            for key, expected_value in expected.items():
                if metadata.get(key) != expected_value:
                    raise ValueError(f"Completed shard provenance mismatch for {key}: {path}")
            if (metadata.get("weight_key") != "ema" or metadata.get("full_dataset_count") != 100
                    or metadata.get("runtime") != str(runtime)
                    or metadata.get("environment_overrides", {}).get("NWM_DATA_ROOT")
                    != str(args.data_root.resolve())):
                raise ValueError(f"Completed shard runtime/data/weight provenance mismatch: {path}")
            copied_split = Path(metadata["copied_split"])
            for name, digest in expected["split_sha256"].items():
                if sha256(copied_split / name) != digest:
                    raise ValueError(f"Copied split provenance mismatch: {copied_split / name}")
            for index in overlap:
                covered_by[index].append(str(path))
        uncovered = [index for index, paths in covered_by.items() if not paths]
        if uncovered:
            raise ValueError(f"Requested windows lack successful, provenance-matched shards: {uncovered}")
        result.update(provenance_valid=True, protocol_file=str(protocol_path),
                      protocol_sha256=sha256(protocol_path),
                      checkpoint_validation="Recorded path, size, mtime_ns, experiment config SHA256 and EMA key",
                      source_and_split_validation="Current SHA256 equals the recorded protocol and completed shards",
                      completed_shards_by_index=covered_by)
        report = json.loads(args.archive_report.read_text())
        sources = report["groups"][f"navigation/opennwm-finalLAM-100k/{args.dataset}/cem80"]["metrics"]["sample_sources"]
        archived_paths = {int(Path(source).stem): Path(source) for source in sources}
        if len(sources) != 100 or set(archived_paths) != set(range(100)):
            raise ValueError("Archive must identify exactly 100 unique original navigation windows")
        result["archive_report_sha256"] = sha256(args.archive_report)
        keys = ("ate", "rpe_trans", "pos_diff_norm", "yaw_diff_norm")
        sample_dir = protocol_dir / "outputs" / args.dataset / EVAL_NAME / "sample_metrics"
        for index in indices:
            path = sample_dir / f"{index:06d}.json"
            archived_path = archived_paths[index]
            current = json.loads(path.read_text())
            archived = json.loads(archived_path.read_text())
            for label, record in ((str(path), current), (str(archived_path), archived)):
                if (set(record) != set(keys) or any(type(record[key]) not in (int, float)
                        or not math.isfinite(record[key]) for key in keys)):
                    raise ValueError(f"Expected exactly four finite numeric planning metrics: {label}")
            differences = {key: current[key] - archived[key] for key in keys}
            result["per_sample_deltas"].append({
                "index": index, "sample_path": str(path), "sample_sha256": sha256(path),
                "archive_source": str(archived_path), "archive_sample_sha256": sha256(archived_path),
                "new": current, "archived": archived, "difference": differences,
                "all_metrics_exact": all(value == 0 for value in differences.values()),
            })
            result["count"] += 1
        result["coverage_valid"] = result["count"] == len(indices)
        result["all_metrics_exact"] = all(row["all_metrics_exact"] for row in result["per_sample_deltas"])
        result["metric_difference_sample_count"] = sum(
            not row["all_metrics_exact"] for row in result["per_sample_deltas"])
        if result["all_metrics_exact"]:
            result.update(state="completed", exit_code=0,
                          verification_outcome="requested_subset_metrics_exact")
        elif allow_differences:
            result.update(state="completed", exit_code=0,
                          verification_outcome="metrics_differ_from_archive_continuation_authorized",
                          comparison_note="Metric differences are retained for the authorized L20 comparison; "
                                          "this is not a successful exact historical reproduction.")
        else:
            result["verification_outcome"] = "metrics_differ_from_archive"
            result["error"] = "At least one requested window differs from its archived saved metrics"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["finished_utc"] = now()
    protocol_dir.mkdir(parents=True, exist_ok=True)
    filename = "l20_subset_verification.json" if allow_differences else "subset_verification.json"
    write_json(protocol_dir / filename, result)
    print(json.dumps(result, indent=2))
    return result["exit_code"]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("prepare", "run", "cpu-check", "aggregate", "verify-subset"))
    result.add_argument("--model", choices=MODELS, required=True)
    result.add_argument("--dataset", choices=DATASETS, required=True)
    result.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    result.add_argument("--runtime", type=Path, default=RUNTIME)
    result.add_argument("--python", type=Path, default=PYTHON)
    result.add_argument("--data-root", type=Path, default=NAS_ROOT / "data")
    result.add_argument("--gpu", default="4")
    result.add_argument("--indices", help="Comma-separated indices or start:stop (exclusive); default 0:100")
    result.add_argument("--num-shards", type=int, default=1)
    result.add_argument("--shard-index", type=int, default=0)
    result.add_argument("--microbatch", type=int, default=80)
    result.add_argument("--accept-microbatch-change", action="store_true")
    result.add_argument("--allow-metric-differences", action="store_true",
                        help="verify-subset only: permit recorded archive metric differences for the authorized "
                             "L20 comparison; provenance and full requested coverage must still pass")
    result.add_argument("--archive-samples", type=Path,
                        help="Optional original per-sample metric directory for exact comparison")
    result.add_argument("--archive-report", type=Path, default=ARCHIVE / "report.json",
                        help="Original report resolves mixed RECON sample source directories")
    return result


def main():
    argument_parser = parser()
    args = argument_parser.parse_args()
    if args.allow_metric_differences and args.action != "verify-subset":
        argument_parser.error("--allow-metric-differences is only valid for verify-subset")
    if args.action == "prepare":
        print(json.dumps(prepare(args), indent=2))
        return 0
    return {"run": run, "cpu-check": cpu_check, "aggregate": aggregate,
            "verify-subset": verify_subset}[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
