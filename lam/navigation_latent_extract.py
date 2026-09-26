#!/usr/bin/env python3
"""Extract navigation-LAM posterior means for CompACT offline motion training.

The output contract is intentionally the small, model-agnostic schema consumed
by ``CompACT.motion_condition.OfflineMotionStore``.  One atomic ``.pt`` file is
written per trajectory, including empty files for trajectories that cannot
produce a CompACT training sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from lam.navigation_evaluation.adapters import (
    inspect_evaluation_checkpoint,
    load_evaluation_adapter,
)
from lam.navigation_evaluation.catalog import load_navigation_frame
from lam.navigation_evaluation.io import (
    atomic_write_json,
    sha256_file,
)

SCHEMA_VERSION = 1
FORMAT = "compact_offline_lam_action"
DEFAULT_DATASET_NAME = "tartan_drive"
PAIR_DIRECTION = "current_to_goal"
NORMALIZATION = "raw"
IMAGE_PREPROCESSING = "center_crop_4:3_then_bilinear_resize"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract deterministic LAM z_mu caches for CompACT NWM-latent"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--input-manifest", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
        command.add_argument("--checkpoint-sha256")
        command.add_argument("--device", default="cuda:0")
        command.add_argument(
            "--precision", choices=("32", "16-mixed", "bf16-mixed"), default="16-mixed"
        )
        command.add_argument("--batch-size", type=int, default=32)
        command.add_argument("--image-height", type=int, default=240)
        command.add_argument("--image-width", type=int, default=320)
        command.add_argument("--min-offset", type=int, default=-64)
        command.add_argument("--max-offset", type=int, default=64)
        command.add_argument("--context-size", type=int, default=4)
        command.add_argument("--len-traj-pred", type=int, default=64)

    extract = subparsers.add_parser("extract", help="extract one resumable shard")
    common(extract)
    extract.add_argument("--shard-index", type=int, default=0)
    extract.add_argument("--shard-count", type=int, default=1)
    extract.add_argument("--overwrite", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark", help="measure safe encoder throughput without writing caches"
    )
    common(benchmark)
    benchmark.add_argument("--batch-sizes", default="16,32,64")
    benchmark.add_argument("--warmup-steps", type=int, default=2)
    benchmark.add_argument("--measure-steps", type=int, default=5)

    validate = subparsers.add_parser(
        "validate", help="validate every trajectory and publish completion markers"
    )
    validate.add_argument("--checkpoint", type=Path, required=True)
    validate.add_argument("--input-manifest", type=Path, required=True)
    validate.add_argument("--output-root", type=Path, required=True)
    validate.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    validate.add_argument("--checkpoint-sha256")
    validate.add_argument("--image-height", type=int, default=240)
    validate.add_argument("--image-width", type=int, default=320)
    validate.add_argument("--min-offset", type=int, default=-64)
    validate.add_argument("--max-offset", type=int, default=64)
    validate.add_argument("--context-size", type=int, default=4)
    validate.add_argument("--len-traj-pred", type=int, default=64)
    validate.add_argument("--expected-shards", type=int)
    return parser


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dataset_name(args: argparse.Namespace) -> str:
    name = str(args.dataset_name)
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"dataset name must be one safe path component: {name!r}")
    return name


def _manifest(path: Path, dataset_name: str) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    raw = resolved.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"input manifest must be a JSON mapping: {resolved}")
    if value.get("complete") is not True:
        raise RuntimeError(f"input manifest is not complete: {resolved}")
    if value.get("dataset_name") != dataset_name:
        raise ValueError(
            f"input manifest dataset={value.get('dataset_name')!r}, expected {dataset_name!r}"
        )
    trajectories = value.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("input manifest has no trajectories")
    names: set[str] = set()
    for record in trajectories:
        if not isinstance(record, dict):
            raise TypeError("trajectory manifest records must be mappings")
        name = record.get("trajectory_name")
        frames = record.get("frame_records")
        if not isinstance(name, str) or not name or "/" in name or ".." in name:
            raise ValueError(f"unsafe trajectory name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate trajectory in input manifest: {name}")
        names.add(name)
        if not isinstance(frames, list) or len(frames) != record.get("num_frames"):
            raise ValueError(f"malformed frame inventory for {name}")
        expected_ids = list(range(len(frames)))
        actual_ids = [frame.get("frame_id") for frame in frames]
        if actual_ids != expected_ids:
            raise ValueError(f"non-contiguous frame ids for {name}")
    if value.get("num_trajectories") != len(trajectories):
        raise ValueError("input manifest trajectory count mismatch")
    return value, hashlib.sha256(raw).hexdigest()


def build_compact_frame_pairs(
    num_frames: int,
    *,
    min_offset: int,
    max_offset: int,
    context_size: int,
    len_traj_pred: int,
) -> torch.Tensor:
    """Mirror CompACT ``BaseDataset._build_index`` exactly."""

    if context_size < 1 or len_traj_pred < 0:
        raise ValueError("context_size must be positive and len_traj_pred non-negative")
    if min_offset > max_offset:
        raise ValueError("min_offset must not exceed max_offset")
    rows: list[tuple[int, int]] = []
    for current in range(context_size - 1, num_frames - len_traj_pred):
        lower = max(0, current + min_offset)
        upper = min(num_frames - 1, current + max_offset)
        rows.extend((current, target) for target in range(lower, upper + 1))
    if not rows:
        return torch.empty((0, 2), dtype=torch.int64)
    return torch.tensor(rows, dtype=torch.int64)


def _pairs(record: Mapping[str, Any], args: argparse.Namespace) -> torch.Tensor:
    return build_compact_frame_pairs(
        int(record["num_frames"]),
        min_offset=int(args.min_offset),
        max_offset=int(args.max_offset),
        context_size=int(args.context_size),
        len_traj_pred=int(args.len_traj_pred),
    )


def _checkpoint_identity(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    actual_sha256 = args.checkpoint_sha256 or sha256_file(checkpoint)
    if args.checkpoint_sha256:
        expected = str(args.checkpoint_sha256).lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("--checkpoint-sha256 must be a lowercase SHA-256 digest")
        actual_sha256 = expected
    metadata = inspect_evaluation_checkpoint(checkpoint)
    hparams = metadata.get("hparams", {})
    data_hparams = metadata.get("datamodule_hparams", {})
    required = {
        "lam_latent_dim": 32,
        "lam_patch_size": 16,
    }
    for key, expected in required.items():
        if hparams.get(key) != expected:
            raise ValueError(
                f"checkpoint {key}={hparams.get(key)!r}, expected {expected!r}"
            )
    for key, expected in (
        ("image_height", int(args.image_height)),
        ("image_width", int(args.image_width)),
        ("max_frame_offset", max(abs(int(args.min_offset)), abs(int(args.max_offset)))),
    ):
        if data_hparams.get(key) != expected:
            raise ValueError(
                f"checkpoint data contract {key}={data_hparams.get(key)!r}, expected {expected!r}"
            )
    return metadata, actual_sha256


def _checkpoint_identity_for_validation(
    args: argparse.Namespace, output_root: Path
) -> tuple[dict[str, Any], str, bool]:
    """Recover immutable checkpoint metadata after a retention-policy rotation.

    Every extracted file and shard receipt is bound to the checkpoint digest.
    Validation normally re-inspects the source checkpoint, but a long-running
    extraction may outlive a trainer policy that removes old checkpoints.  In
    that case an explicitly supplied digest plus the cache's repeated metadata
    is sufficient to reconstruct the already-published contract.
    """

    try:
        metadata, digest = _checkpoint_identity(args)
        return metadata, digest, True
    except FileNotFoundError:
        if not args.checkpoint_sha256:
            raise
    candidates = sorted((output_root / _dataset_name(args)).glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            "checkpoint was rotated and no extracted cache exists to recover metadata"
        )
    payload = torch.load(
        candidates[0], map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(payload, Mapping):
        raise TypeError(f"cache payload is not a mapping: {candidates[0]}")
    digest = str(args.checkpoint_sha256).lower()
    if payload.get("checkpoint_sha256") != digest:
        raise RuntimeError(
            "rotated-checkpoint validation digest does not match extracted cache"
        )
    required = {
        "checkpoint_class_path": str,
        "checkpoint_global_step": int,
        "checkpoint_epoch": int,
        "latent_dim": int,
    }
    for key, expected_type in required.items():
        if not isinstance(payload.get(key), expected_type):
            raise TypeError(f"cache metadata {key!r} is missing or malformed")
    metadata = {
        "path": str(args.checkpoint.expanduser().resolve()),
        "class_path": payload["checkpoint_class_path"],
        "global_step": payload["checkpoint_global_step"],
        "epoch": payload["checkpoint_epoch"],
        "hparams": {"lam_latent_dim": payload["latent_dim"]},
    }
    return metadata, digest, False


def _contract(
    args: argparse.Namespace,
    checkpoint_metadata: Mapping[str, Any],
    checkpoint_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "format": FORMAT,
        "dataset_name": _dataset_name(args),
        "motion_type": "latent",
        "pair_direction": PAIR_DIRECTION,
        "normalization": NORMALIZATION,
        "latent_dim": int(checkpoint_metadata["hparams"]["lam_latent_dim"]),
        "checkpoint_path": str(args.checkpoint.expanduser().resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_class_path": checkpoint_metadata["class_path"],
        "checkpoint_global_step": int(checkpoint_metadata["global_step"]),
        "checkpoint_epoch": int(checkpoint_metadata["epoch"]),
        "input_manifest_path": str(args.input_manifest.expanduser().resolve()),
        "input_manifest_sha256": manifest_sha256,
        "image_height": int(args.image_height),
        "image_width": int(args.image_width),
        "image_preprocessing": IMAGE_PREPROCESSING,
        "min_offset": int(args.min_offset),
        "max_offset": int(args.max_offset),
        "context_size": int(args.context_size),
        "len_traj_pred": int(args.len_traj_pred),
        "deterministic_posterior": "z_mu",
    }


def _payload_valid(
    path: Path,
    record: Mapping[str, Any],
    pairs: torch.Tensor,
    contract: Mapping[str, Any],
) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return False, f"unreadable: {exc}"
    if not isinstance(payload, Mapping):
        return False, "not a mapping"
    for key, value in contract.items():
        if payload.get(key) != value:
            return False, f"metadata mismatch: {key}"
    expected = {
        "trajectory_name": record["trajectory_name"],
        "split": record["split"],
        "source_frame_list_sha256": record["frame_list_sha256"],
        "num_frames": int(record["num_frames"]),
        "num_pairs": int(pairs.shape[0]),
        "complete": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            return False, f"trajectory metadata mismatch: {key}"
    stored_pairs = payload.get("frame_pairs")
    motion = payload.get("motion")
    if not isinstance(stored_pairs, torch.Tensor) or stored_pairs.dtype != torch.int64:
        return False, "frame_pairs must be an int64 tensor"
    if not torch.equal(stored_pairs, pairs):
        return False, "frame_pairs mismatch"
    latent_dim = int(contract["latent_dim"])
    if not isinstance(motion, torch.Tensor):
        return False, "motion is not a tensor"
    if motion.dtype != torch.float32 or tuple(motion.shape) != (len(pairs), latent_dim):
        return False, "motion dtype/shape mismatch"
    if not torch.isfinite(motion).all():
        return False, "motion contains non-finite values"
    return True, "ok"


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(dict(value), handle)
            handle.flush()
            os.fchmod(handle.fileno(), 0o640)
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_shared_json(path: Path, value: Any) -> Path:
    output = atomic_write_json(path, value)
    output.chmod(0o640)
    return output


def _output_path(root: Path, dataset_name: str, trajectory_name: str) -> Path:
    resolved_root = root.expanduser().resolve()
    path = (resolved_root / dataset_name / f"{trajectory_name}.pt").resolve()
    path.relative_to(resolved_root)
    return path


def _shard_status_path(root: Path, args: argparse.Namespace, shard_index: int) -> Path:
    dataset_name = _dataset_name(args)
    directory = root / "_shards"
    if dataset_name != DEFAULT_DATASET_NAME:
        directory /= dataset_name
    return directory / f"shard-{shard_index:04d}.json"


def _completion_path(root: Path, args: argparse.Namespace, filename: str) -> Path:
    dataset_name = _dataset_name(args)
    if dataset_name == DEFAULT_DATASET_NAME:
        return root / filename
    path = Path(filename)
    return root / f"{path.stem}.{dataset_name}{path.suffix}"


def _assigned_records(
    records: Sequence[Mapping[str, Any]], args: argparse.Namespace
) -> list[Mapping[str, Any]]:
    shard_count = int(args.shard_count)
    shard_index = int(args.shard_index)
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard index must be in [0, shard count)")
    weighted = sorted(
        ((len(_pairs(record, args)), str(record["trajectory_name"]), record) for record in records),
        key=lambda item: (-item[0], item[1]),
    )
    loads = [0] * shard_count
    assignments: list[list[Mapping[str, Any]]] = [[] for _ in range(shard_count)]
    for cost, _, record in weighted:
        destination = min(range(shard_count), key=lambda index: (loads[index], index))
        assignments[destination].append(record)
        loads[destination] += cost
    return assignments[shard_index]


def _configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(32)
    np.random.seed(32)
    torch.manual_seed(32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(32)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False


def _autocast(device: torch.device, precision: str):
    if precision == "32" or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if precision == "16-mixed" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _load_frames(record: Mapping[str, Any], manifest: Mapping[str, Any], args: argparse.Namespace) -> torch.Tensor:
    data_root = Path(str(manifest["data_root"])).expanduser().resolve()
    frames: list[torch.Tensor] = []
    for frame_record in record["frame_records"]:
        path = (data_root / str(frame_record["relative_path"])).resolve()
        path.relative_to(data_root)
        if not path.is_file() or path.stat().st_size != int(frame_record["size_bytes"]):
            raise RuntimeError(f"source frame missing or size changed: {path}")
        frames.append(
            load_navigation_frame(
                path,
                image_height=int(args.image_height),
                image_width=int(args.image_width),
            )
        )
    return torch.stack(frames).contiguous()


def _encode_pairs(
    adapter: Any,
    frames: torch.Tensor,
    pairs: torch.Tensor,
    *,
    batch_size: int,
    precision: str,
) -> torch.Tensor:
    if len(pairs) == 0:
        latent_dim = int(adapter.checkpoint_metadata["hparams"]["lam_latent_dim"])
        return torch.empty((0, latent_dim), dtype=torch.float32)
    # A trajectory contains at most a few hundred resized frames.  Keeping that
    # bank on the inference GPU avoids repeating large CPU index_select/stack
    # copies for every pair batch, which otherwise starves multi-GPU workers.
    frame_bank = frames.to(device=adapter.device)
    pair_bank = pairs.to(device=adapter.device)
    outputs: list[torch.Tensor] = []
    for start in range(0, len(pairs), batch_size):
        selection = pair_bank[start : start + batch_size]
        current = frame_bank.index_select(0, selection[:, 0])
        goal = frame_bank.index_select(0, selection[:, 1])
        videos = torch.stack((current, goal), dim=1)
        with _autocast(adapter.device, precision):
            z_mu = adapter.encode(videos)
        outputs.append(z_mu[:, 0, 0].detach().float().cpu())
    return torch.cat(outputs, dim=0).contiguous()


def _load_adapter(args: argparse.Namespace) -> Any:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full navigation LAM extraction requires CUDA")
    return load_evaluation_adapter(
        args.checkpoint.expanduser().resolve(),
        device=device,
        fused_attention=True,
    )


def _extract(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    _configure_determinism()
    dataset_name = _dataset_name(args)
    manifest, manifest_sha256 = _manifest(args.input_manifest, dataset_name)
    checkpoint_metadata, checkpoint_sha256 = _checkpoint_identity(args)
    contract = _contract(args, checkpoint_metadata, checkpoint_sha256, manifest_sha256)
    records = _assigned_records(manifest["trajectories"], args)
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[Mapping[str, Any], torch.Tensor, Path]] = []
    resumed = 0
    for record in records:
        pairs = _pairs(record, args)
        path = _output_path(root, dataset_name, str(record["trajectory_name"]))
        if not args.overwrite:
            valid, _ = _payload_valid(path, record, pairs, contract)
            if valid:
                resumed += 1
                continue
        pending.append((record, pairs, path))

    nonempty = sum(len(pairs) > 0 for _, pairs, _ in pending)
    print(
        json.dumps(
            {
                "event": "shard_start",
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "assigned_trajectories": len(records),
                "pending_trajectories": len(pending),
                "pending_nonempty": nonempty,
                "pending_pairs": sum(len(pairs) for _, pairs, _ in pending),
                "resumed_trajectories": resumed,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    adapter = _load_adapter(args) if nonempty else None
    started = time.time()
    written_pairs = 0
    written_trajectories = 0
    for record, pairs, path in pending:
        trajectory_started = time.time()
        if len(pairs):
            frames = _load_frames(record, manifest, args)
            motion = _encode_pairs(
                adapter,
                frames,
                pairs,
                batch_size=int(args.batch_size),
                precision=str(args.precision),
            )
            del frames
        else:
            motion = torch.empty((0, int(contract["latent_dim"])), dtype=torch.float32)
        payload = {
            **contract,
            "trajectory_name": record["trajectory_name"],
            "split": record["split"],
            "source_frame_list_sha256": record["frame_list_sha256"],
            "num_frames": int(record["num_frames"]),
            "num_pairs": len(pairs),
            "frame_pairs": pairs,
            "motion": motion,
            "complete": True,
        }
        _atomic_torch_save(path, payload)
        valid, reason = _payload_valid(path, record, pairs, contract)
        if not valid:
            raise RuntimeError(f"post-write validation failed for {path}: {reason}")
        written_pairs += len(pairs)
        written_trajectories += 1
        print(
            json.dumps(
                {
                    "event": "trajectory_complete",
                    "trajectory": record["trajectory_name"],
                    "pairs": len(pairs),
                    "seconds": round(time.time() - trajectory_started, 3),
                    "written_trajectories": written_trajectories,
                    "pending_trajectories": len(pending),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    status = {
        "schema_version": 1,
        "complete": True,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "contract_sha256": _canonical_json_sha256(contract),
        "assigned_trajectories": len(records),
        "assigned_pairs": sum(len(_pairs(record, args)) for record in records),
        "written_trajectories": written_trajectories,
        "written_pairs": written_pairs,
        "resumed_trajectories": resumed,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_shared_json(
        _shard_status_path(root, args, int(args.shard_index)), status
    )
    print(json.dumps({"event": "shard_complete", **status}, sort_keys=True), flush=True)


def _benchmark(args: argparse.Namespace) -> None:
    _configure_determinism()
    manifest, _ = _manifest(args.input_manifest, _dataset_name(args))
    _checkpoint_identity(args)
    candidates = sorted(
        manifest["trajectories"], key=lambda record: int(record["num_frames"]), reverse=True
    )
    record = next(record for record in candidates if len(_pairs(record, args)) > 0)
    pairs = _pairs(record, args)
    frames = _load_frames(record, manifest, args)
    adapter = _load_adapter(args)
    results = []
    for batch_size in [int(value) for value in str(args.batch_sizes).split(",")]:
        if batch_size < 1:
            raise ValueError("benchmark batch sizes must be positive")
        needed = batch_size * (int(args.warmup_steps) + int(args.measure_steps))
        repeats = math.ceil(needed / len(pairs))
        selection = pairs.repeat((repeats, 1))[:needed]
        try:
            _encode_pairs(
                adapter,
                frames,
                selection[: batch_size * int(args.warmup_steps)],
                batch_size=batch_size,
                precision=str(args.precision),
            )
            torch.cuda.synchronize(adapter.device)
            started = time.perf_counter()
            _encode_pairs(
                adapter,
                frames,
                selection[batch_size * int(args.warmup_steps) :],
                batch_size=batch_size,
                precision=str(args.precision),
            )
            torch.cuda.synchronize(adapter.device)
            elapsed = time.perf_counter() - started
            result = {
                "batch_size": batch_size,
                "status": "ok",
                "pairs": batch_size * int(args.measure_steps),
                "seconds": elapsed,
                "pairs_per_second": batch_size * int(args.measure_steps) / elapsed,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(adapter.device),
            }
        except torch.cuda.OutOfMemoryError as exc:
            result = {"batch_size": batch_size, "status": "oom", "error": str(exc)}
            torch.cuda.empty_cache()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps({"event": "benchmark_complete", "results": results}, sort_keys=True))


def _validate(args: argparse.Namespace) -> None:
    dataset_name = _dataset_name(args)
    manifest, manifest_sha256 = _manifest(args.input_manifest, dataset_name)
    root = args.output_root.expanduser().resolve()
    checkpoint_metadata, checkpoint_sha256, checkpoint_available = (
        _checkpoint_identity_for_validation(args, root)
    )
    contract = _contract(args, checkpoint_metadata, checkpoint_sha256, manifest_sha256)
    if args.expected_shards is not None:
        for index in range(int(args.expected_shards)):
            status_path = _shard_status_path(root, args, index)
            if not status_path.is_file():
                raise FileNotFoundError(f"shard completion marker is missing: {status_path}")
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("complete") is not True or status.get("contract_sha256") != _canonical_json_sha256(contract):
                raise RuntimeError(f"invalid shard completion marker: {status_path}")
    failures = []
    total_pairs = 0
    files_sha256 = hashlib.sha256()
    split_counts: dict[str, dict[str, int]] = {}
    started = time.time()
    for index, record in enumerate(manifest["trajectories"], start=1):
        pairs = _pairs(record, args)
        path = _output_path(root, dataset_name, str(record["trajectory_name"]))
        valid, reason = _payload_valid(path, record, pairs, contract)
        if not valid:
            failures.append({"trajectory": record["trajectory_name"], "reason": reason})
        else:
            digest = sha256_file(path)
            files_sha256.update(str(record["trajectory_name"]).encode("utf-8"))
            files_sha256.update(b"\0")
            files_sha256.update(digest.encode("ascii"))
            files_sha256.update(b"\0")
        total_pairs += len(pairs)
        split = str(record["split"])
        split_counts.setdefault(split, {"trajectories": 0, "pairs": 0})
        split_counts[split]["trajectories"] += 1
        split_counts[split]["pairs"] += len(pairs)
        if index % 100 == 0:
            print(
                json.dumps(
                    {"event": "validation_progress", "checked": index, "total": len(manifest["trajectories"])},
                    sort_keys=True,
                ),
                flush=True,
            )
    if failures:
        _atomic_shared_json(
            _completion_path(root, args, "validation_failures.json"), failures
        )
        raise RuntimeError(f"latent cache validation failed for {len(failures)} trajectories")
    metadata = {
        **contract,
        "status": "complete",
        "complete": True,
        "num_trajectories": len(manifest["trajectories"]),
        "num_pairs": total_pairs,
        "split_counts": split_counts,
        "trajectory_files_sha256": files_sha256.hexdigest(),
        "checkpoint_available_at_validation": checkpoint_available,
        "checkpoint_validation_note": (
            "source checkpoint re-inspected"
            if checkpoint_available
            else "source checkpoint rotated after extraction; identity recovered from digest-bound cache and shard receipts"
        ),
        "validated_at_unix": time.time(),
        "validation_seconds": time.time() - started,
    }
    metadata_path = _atomic_shared_json(
        _completion_path(root, args, "metadata.json"), metadata
    )
    success = {
        "schema_version": 1,
        "complete": True,
        "format": FORMAT,
        "dataset_name": dataset_name,
        "checkpoint_sha256": checkpoint_sha256,
        "metadata_sha256": sha256_file(metadata_path),
        "trajectory_files_sha256": files_sha256.hexdigest(),
        "num_trajectories": len(manifest["trajectories"]),
        "num_pairs": total_pairs,
    }
    _atomic_shared_json(_completion_path(root, args, "_SUCCESS.json"), success)
    print(json.dumps({"event": "validation_complete", **success}, sort_keys=True), flush=True)


def main() -> None:
    args = _parser().parse_args()
    if args.command == "extract":
        _extract(args)
    elif args.command == "benchmark":
        _benchmark(args)
    elif args.command == "validate":
        _validate(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
