#!/usr/bin/env python3
"""Replay only the two newly inferred Table 1 groups in the later archive.

The later archive reused historical 4s predictions for the other six Table 1
datasets. It independently reran HuRoN with batch 128 and all five horizons,
and Planetary Rover with all three horizons. This launcher preserves those
commands, including RNG-relevant earlier/later horizons and effective sample IDs.
It does not silently reduce the HuRoN batch when GPU memory is insufficient.
The explicit --batch-size 32 --accept-batch-change option writes a separate
L20 protocol, retaining the complete original batch128 job and metric reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shlex
import shutil
import subprocess
import sys

import reproduce_paper_table1 as historical


RUNTIME = historical.HOME_ROOT / "code/CompACT_a800_original_seed0_v1"
LATER = historical.NAS_ROOT / "results/nwm_benchmark/a800_full_evaluations/full_8xa800_seed0_v1"
LATER_MODEL = "opennwm-finalLAM-100k"
JOBS = {
    "huron": "direct_opennwm-finalLAM-100k_huron_time_0-328_ed085d1c.json",
    "planetary_rover": "direct_opennwm-finalLAM-100k_planetary_rover_time_0-9_69330f32.json",
}
REUSED = ("recon", "scand", "tartan_drive", "go_stanford", "tum_rgbd", "unitree_go2")


def read_json(path):
    return json.loads(Path(path).read_text())


def source(path):
    return {"path": str(path), "sha256": historical.sha256(path)}


def verify_runtime():
    contract = read_json(LATER / "contract.json")
    checked = {}
    for name in ("isolated_nwm_infer.py", "conf/infer_config.yaml"):
        actual = historical.sha256(RUNTIME / name)
        if actual != contract["entrypoint_sha256"][name]:
            raise RuntimeError(f"Later runtime changed: {RUNTIME / name}")
        checked[name] = actual
    return checked


def huron_population(data_root):
    """Apply the archived loader validity rule, without modifying its split."""
    split = RUNTIME / "data_splits/sacson/test/time.pkl"
    rows = pickle.loads(split.read_bytes())
    lengths, valid, effective = {}, [], []
    missing = short = 0
    for raw_index, (trajectory, current, lower, upper, *_) in enumerate(rows):
        if trajectory not in lengths:
            path = data_root / "sacson" / trajectory / "traj_data.pkl"
            lengths[trajectory] = len(pickle.loads(path.read_bytes())["position"]) if path.is_file() else None
        length = lengths[trajectory]
        if length is None:
            missing += 1
        elif current + lower < 0 or current + upper >= length:
            short += 1
        else:
            valid.append(raw_index)
            effective.append([raw_index, trajectory, current, lower, upper, length])
    mapping = json.dumps(effective, sort_keys=True, separators=(",", ":"))
    result = {
        "raw": len(rows), "valid": len(valid), "missing_trajectory_rows": missing,
        "out_of_range_rows": short, "effective_raw_row_indices": valid,
        "effective_mapping_sha256": hashlib.sha256(mapping.encode()).hexdigest(),
        "trajectory_lengths_sha256": hashlib.sha256(
            json.dumps(sorted(lengths.items()), separators=(",", ":")).encode()).hexdigest(),
        "split_sha256": historical.sha256(split),
        "sample_id_semantics": "position_after_loader_filtering",
    }
    expected = read_json(LATER / "report.json")["huron"]["time"]
    if result != expected:
        changed = [key for key in set(expected) | set(result) if expected.get(key) != result.get(key)]
        raise RuntimeError(f"HuRoN effective population changed in fields: {changed}")
    return result


def archived_job(dataset):
    path = LATER / "jobs" / JOBS[dataset]
    job = read_json(path)
    if job["status"] != "complete" or job["exit_code"] != 0:
        raise RuntimeError(f"Archived job is not complete: {path}")
    expected_count = 329 if dataset == "huron" else 10
    expected_horizons = [1, 2, 4, 8, 16] if dataset == "huron" else [1, 2, 4]
    if (job["job"]["ids"] != list(range(expected_count))
            or job["job"]["endpoints"] != expected_horizons or job["batch"] != 128):
        raise RuntimeError(f"Archived command contract changed: {path}")
    return path, job


def run_directory(args):
    batch = validated_batch(args)
    protocol = "l20_batch32" if batch == 32 else "corrected"
    return args.output_root.resolve() / protocol / args.model / args.dataset


def validated_batch(args):
    batch = getattr(args, "batch_size", 128)
    accepted = getattr(args, "accept_batch_change", False)
    if batch == 32:
        if args.dataset != "huron" or not accepted:
            raise ValueError("batch32 is a new HuRoN protocol and requires --accept-batch-change")
    elif batch != 128 or accepted:
        raise ValueError("Unchanged archive batch is 128; explicit batch-change acceptance is only for HuRoN batch32")
    return batch


def inference_command(args, job, run_dir, experiment):
    replacements = {
        "exp_dir": str(experiment), "output_dir": str(run_dir),
        "prediction_dir": str(run_dir / "predictions"),
        "hydra.run.dir": str(run_dir / "hydra"),
        "batch_size": str(validated_batch(args)),
    }
    hydra_args = []
    for argument in job["command"][4:]:
        key = argument.split("=", 1)[0]
        hydra_args.append(f"{key}={replacements[key]}" if key in replacements else argument)
    return [str(historical.PYTHON), "-m", "torch.distributed.run", "--standalone",
            "--nproc-per-node=1", str(Path(historical.__file__).resolve()), "_entry",
            "--metadata", str(run_dir / "metadata.json"), "--", *hydra_args]


def plan(args):
    batch = validated_batch(args)
    checked = verify_runtime()
    _, job = archived_job(args.dataset)
    population = huron_population(args.data_root.resolve()) if args.dataset == "huron" else None
    command = inference_command(args, job, run_directory(args), historical.MODELS[args.model])
    print(json.dumps({"run_directory": str(run_directory(args)), "batch_size": batch,
                      "archive_batch_size": job["batch"], "new_protocol": batch != job["batch"],
                      "expected_count": len(job["job"]["ids"]),
                      "population_mapping_sha256": population["effective_mapping_sha256"] if population else None,
                      "runtime_hashes": checked, "runtime": str(RUNTIME),
                      "command": command, "shell_command": shlex.join(command)}, indent=2))
    return 0


def prepare(args):
    batch = validated_batch(args)
    checked = verify_runtime()
    job_path, job = archived_job(args.dataset)
    data_root = args.data_root.resolve()
    run_dir = run_directory(args)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to mix or resume predictions: {run_dir}")
    parent = run_dir.parent
    while not parent.exists():
        parent = parent.parent
    disk = shutil.disk_usage(parent)
    if disk.free < 2 * 1024**3:
        raise RuntimeError("Need at least 2 GiB free for prediction outputs")
    # The recorded HuRoN run peaked at 56.52 GiB. Keep a modest buffer and
    # require actual memory availability instead of quietly changing batch 128.
    minimum_free_gib = 22 if batch == 32 else (60 if args.dataset == "huron" else 8)
    gpu = historical.inspect_gpu(args.gpu, minimum_free_gib)
    population = huron_population(data_root) if args.dataset == "huron" else None
    experiment = historical.MODELS[args.model]
    checkpoint = experiment / "checkpoints/joint_0100000.pth.tar"
    for path in (checkpoint, experiment / ".hydra/config.yaml", historical.PYTHON, data_root):
        if not path.exists():
            raise FileNotFoundError(path)
    split = "sacson" if args.dataset == "huron" else "planetary_rover"
    original_split = RUNTIME / "data_splits" / split / "test"
    split_sha = historical.sha256(original_split / "time.pkl")
    expected_split = read_json(LATER / "contract.json")["contracts"][f"{LATER_MODEL}/{args.dataset}/time"]["split_sha256"]
    if split_sha != expected_split:
        raise RuntimeError("Later split hash no longer matches archived contract")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "predictions").mkdir()
    copied = run_dir / "splits" / split / "test"
    copied.mkdir(parents=True)
    split_hashes = {}
    for name in ("time.pkl", "traj_names.txt"):
        original = original_split / name
        if original.exists():
            shutil.copy2(original, copied / name)
            split_hashes[name] = historical.sha256(original)
    command = inference_command(args, job, run_dir, experiment)
    metadata = {
        "schema_version": 1, "created_utc": historical.now(), "state": "prepared",
        "protocol": "l20_batch32" if batch == 32 else "corrected_later_archive", "model": args.model,
        "new_protocol": batch != job["batch"], "accept_batch_change": args.accept_batch_change,
        "archive_batch_size": job["batch"], "archive_job": job,
        "archive_reference_group": read_json(LATER / "report.json")["groups"][f"direct/{LATER_MODEL}/{args.dataset}/time"],
        "protocol_changes": ({"batch_size": {"archive": 128, "current": 32},
                              "device_family": {"archive": "A800", "current": "L20"}} if batch == 32 else {}),
        "part": args.dataset, "datasets": [args.dataset],
        "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "experiment_config_sha256": historical.sha256(experiment / ".hydra/config.yaml"),
        "runtime": str(RUNTIME), "runtime_fingerprint": historical.runtime_fingerprint(RUNTIME),
        "verified_archive_runtime_hashes": checked,
        "launcher_sha256": historical.sha256(__file__),
        "entry_launcher_sha256": historical.sha256(historical.__file__),
        "source_job": source(job_path), "source_job_command": job["command"],
        "virtual_rank": 0, "virtual_world_size": 1, "physical_world_size": 1,
        "seed": 0, "rank_seed": 0, "batch_size": batch, "num_workers": 4,
        "expected_full_count": len(job["job"]["ids"]), "sample_ids": job["job"]["ids"],
        "diffusion_steps": 250, "horizons_seconds": job["job"]["endpoints"],
        "eval_len_traj_pred": max(job["job"]["endpoints"]) * 4,
        "split_paths": {split: str(copied)}, "split_sha256": split_hashes,
        "effective_population": population, "vae_posterior": "historical stochastic encode, unchanged",
        "data_root": str(data_root), "run_directory": str(run_dir),
        "log": str(run_dir / "inference.log"), "command": command,
        "shell_command": shlex.join(command), "free_disk_bytes": disk.free,
        "gpu_preflight": gpu,
        "archive_predictions": str(LATER / "predictions" / LATER_MODEL),
        "environment_overrides": {
            "CUDA_VISIBLE_DEVICES": args.gpu, "NWM_DATA_ROOT": str(data_root),
            "NWM_INDEX_ROOT": str(run_dir / "index_cache"),
            "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        },
        "equivalence_status": "new_protocol_not_exact_archive_reproduction" if batch == 32 else "unverified_until_archive_pixel_comparison",
    }
    historical.write_json(run_dir / "metadata.json", metadata)
    return metadata


def infer(args):
    metadata = prepare(args)
    run_dir = Path(metadata["run_directory"])
    print(json.dumps({key: metadata[key] for key in ("runtime", "shell_command", "log")}, indent=2), flush=True)
    with Path(metadata["log"]).open("w") as log:
        process = subprocess.Popen(metadata["command"], cwd=RUNTIME,
                                   env=dict(os.environ, **metadata["environment_overrides"]),
                                   stdout=log, stderr=subprocess.STDOUT)
        metadata.update(state="running", pid=process.pid, started_utc=historical.now())
        historical.write_json(run_dir / "metadata.json", metadata)
        print(f"PID={process.pid}; log={metadata['log']}", flush=True)
        return_code = process.wait()
    metadata.update(state="completed" if return_code == 0 else "failed",
                    exit_code=return_code, finished_utc=historical.now())
    historical.write_json(run_dir / "metadata.json", metadata)
    print(f"Exit status: {return_code}; log tail:", flush=True)
    print("\n".join(Path(metadata["log"]).read_text(errors="replace").splitlines()[-30:]))
    return return_code


def compare(args):
    from PIL import Image, ImageChops
    run_dir = run_directory(args)
    metadata = read_json(run_dir / "metadata.json")
    if metadata.get("state") != "completed" or metadata.get("exit_code") != 0:
        raise RuntimeError("Prediction comparison requires a successful completed run")
    if args.model != "baseline":
        raise ValueError("Archive equivalence comparison requires --model baseline")
    root = run_dir / "predictions"
    expected = {
        Path(args.dataset) / "time" / f"id_{sample}" / f"{horizon}.png"
        for sample in metadata["sample_ids"] for horizon in metadata["horizons_seconds"]
    }
    actual = {path.relative_to(root) for path in root.rglob("*.png")}
    if actual != expected:
        raise RuntimeError(f"Prediction coverage mismatch: missing={len(expected - actual)}, extra={len(actual - expected)}")
    counts, mismatches = {}, []
    for relative in sorted(expected):
        horizon = relative.stem
        counts.setdefault(horizon, {"compared": 0, "pixel_identical": 0, "byte_identical": 0})
        archived = Path(metadata["archive_predictions"]) / relative
        with Image.open(root / relative) as left, Image.open(archived) as right:
            same = (left.mode == right.mode and left.size == right.size
                    and ImageChops.difference(left, right).getbbox() is None)
        counts[horizon]["compared"] += 1
        counts[horizon]["pixel_identical"] += int(same)
        counts[horizon]["byte_identical"] += int(historical.sha256(root / relative) == historical.sha256(archived))
        if not same:
            mismatches.append(str(relative))
    report = {"created_utc": historical.now(), "counts_by_horizon": counts,
              "all_pixels_identical": not mismatches, "mismatched_files": mismatches,
              "run_directory": str(run_dir), "archive": metadata["archive_predictions"],
              "protocol": metadata["protocol"], "new_protocol": metadata.get("new_protocol", False)}
    historical.write_json(run_dir / "archive_pixel_comparison.json", report)
    metadata["equivalence_status"] = ("new_protocol_not_exact_archive_reproduction" if metadata.get("new_protocol")
                                      else ("all_pixels_identical" if not mismatches else "pixel_mismatch"))
    historical.write_json(run_dir / "metadata.json", metadata)
    print(json.dumps(report, indent=2))
    return int(bool(mismatches))


def audit(args):
    from PIL import Image, ImageChops
    checks = verify_runtime()
    populations = huron_population(args.data_root.resolve())
    reused = {}
    for dataset in REUSED:
        matches = []
        historical_audit = historical.ARCHIVE_PREDICTIONS / f"{dataset}_time_direct_4s_v1_audit.json"
        original_gt = Path(read_json(historical_audit)["gt_eval_dir"])
        later_gt = LATER / "gt" / dataset / "time"
        gt_pixels_identical, gt_bytes_identical = 0, 0
        original_gt_digest, later_gt_digest = hashlib.sha256(), hashlib.sha256()
        for sample in range(500):
            # The later archive stores legacy 4s symlinks under id_N_4; its
            # scorer resolves the original source through per-sample state.
            pointer = LATER / "predictions" / LATER_MODEL / dataset / "time" / f"id_{sample}_4/4.png"
            original = historical.ARCHIVE_PREDICTIONS / dataset / "time" / f"id_{sample}/4.png"
            matches.append(pointer.is_symlink() and pointer.resolve() == original.resolve()
                           and original.is_file())
            relative = Path(f"id_{sample}") / "4.png"
            left_path, right_path = original_gt / relative, later_gt / relative
            with Image.open(left_path) as left, Image.open(right_path) as right:
                same = (left.mode == right.mode and left.size == right.size
                        and ImageChops.difference(left, right).getbbox() is None)
            left_hash, right_hash = historical.sha256(left_path), historical.sha256(right_path)
            gt_pixels_identical += int(same)
            gt_bytes_identical += int(left_hash == right_hash)
            original_gt_digest.update(f"{relative}:{left_hash}\n".encode())
            later_gt_digest.update(f"{relative}:{right_hash}\n".encode())
        if not all(matches):
            raise RuntimeError(f"Cannot prove historical prediction reuse for {dataset}")
        if gt_pixels_identical != 500:
            raise RuntimeError(f"Historical and later GT endpoints differ for {dataset}")
        reused[dataset] = {"sample_count": 500, "same_historical_source_links": sum(matches),
                           "historical_gt_directory": str(original_gt),
                           "later_gt_directory": str(later_gt),
                           "gt_pixel_identical": gt_pixels_identical,
                           "gt_byte_identical": gt_bytes_identical,
                           "historical_gt_manifest_sha256": original_gt_digest.hexdigest(),
                           "later_gt_manifest_sha256": later_gt_digest.hexdigest(),
                           "reuse_rule": "Use this model's current historical-protocol predictions for both protocols."}
        print(f"{dataset}: 500 historical prediction links; "
              f"GT pixels identical={gt_pixels_identical}/500, "
              f"bytes identical={gt_bytes_identical}/500", flush=True)
    report = {
        "created_utc": historical.now(), "source_report": source(LATER / "report.json"),
        "runtime": str(RUNTIME), "runtime_hashes_match_contract": checks,
        "huron_population": populations, "reused_historical_predictions": reused,
        "independent_jobs": {dataset: read_json(LATER / "jobs" / name) for dataset, name in JOBS.items()},
        "minimal_new_inference": ["huron", "planetary_rover"],
        "later_huron_memory": "Archived batch128 peak56.52GiB; launcher requires60GiB free. No batch reduction.",
        "provenance_rule": "Current historical-protocol predictions may serve both protocols on six unchanged datasets. Archived predictions and archived rescoring remain separately labeled; HuRoN/Rover require their own fresh inference.",
        "metric_rule": "Score matching GT endpoints with identical metric implementation/settings for baseline and DINO; no borrowed baseline scalar substitutes for a new prediction.",
    }
    path = args.output_root.resolve() / "audit" / "table1_later_protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    historical.write_json(path, report)
    print(path)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("infer", "compare", "audit", "plan"):
        current = commands.add_parser(name)
        current.add_argument("--output-root", type=Path, required=True)
        if name != "audit":
            current.add_argument("--model", choices=historical.MODELS, required=True)
            current.add_argument("--dataset", choices=JOBS, required=True)
            current.add_argument("--batch-size", type=int, choices=(32, 128), default=128)
            current.add_argument("--accept-batch-change", action="store_true",
                                 help="Explicitly accept the independent L20 HuRoN batch32 protocol")
        if name == "infer":
            current.add_argument("--gpu", required=True)
        if name != "compare":
            current.add_argument("--data-root", type=Path,
                                 default=historical.NAS_ROOT / "data")
    args = parser.parse_args()
    return {"infer": infer, "compare": compare, "audit": audit, "plan": plan}[args.operation](args)


if __name__ == "__main__":
    raise SystemExit(main())
