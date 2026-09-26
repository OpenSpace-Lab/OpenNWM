"""Precompute the complete navigation-train DINOv3 patch-token cache.

The cache stores one contiguous FP16 file per navigation source.  Run with
``torchrun`` to split whole trajectories across GPUs; interrupted jobs resume
at trajectory granularity without exposing a partial cache to training.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Dataset

from lam.navigation_dataset import NavigationPairDataset, load_navigation_frame
from prepare_dinov3 import (
    DEFAULT_DINO_REPO,
    DEFAULT_OUTPUT,
    EXPECTED_DINO_REVISION,
    EXPECTED_SHA256,
    load_strict_backbone,
    validate_checkpoint,
    validate_source,
)

DEFAULT_BASE_CONFIG = Path(__file__).parent / "config" / "navigation_base.yaml"
DEFAULT_CACHE_ROOT = Path("cache/dino")
CACHE_VERSION = 1
PATCH_COUNT = 300
FEATURE_DIM = 1024
FEATURE_DTYPE = np.dtype("<f2")


@dataclass(frozen=True)
class _WorkUnit:
    source_id: int
    trajectory_index: int
    status_index: int
    offset: int
    length: int
    directory: Path


class _AssignedFrames(Dataset):
    def __init__(self, units: Sequence[_WorkUnit]) -> None:
        self.units = tuple(units)
        cumulative = []
        total = 0
        for unit in self.units:
            total += unit.length
            cumulative.append(total)
        self.cumulative = tuple(cumulative)

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, index: int):
        unit_index = bisect.bisect_right(self.cumulative, index)
        previous = self.cumulative[unit_index - 1] if unit_index else 0
        frame_index = index - previous
        unit = self.units[unit_index]
        return (
            unit.source_id,
            unit.offset + frame_index,
            load_navigation_frame(unit.directory / f"{frame_index}.jpg"),
        )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_data_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Cannot read navigation base config {path}: {exc}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), Mapping):
        raise TypeError(f"Navigation base config has no data mapping: {path}")
    return dict(payload["data"])


def _cache_payload(sources: Sequence[Any]) -> dict[str, Any]:
    cached_sources = []
    status_index = 0
    for source_id, source in enumerate(sources):
        trajectories = []
        offset = 0
        for trajectory in source.trajectories:
            trajectories.append(
                {
                    "name": trajectory.name,
                    "offset": offset,
                    "length": trajectory.length,
                    "metadata_size": trajectory.metadata_size,
                    "metadata_mtime_ns": trajectory.metadata_mtime_ns,
                    "directory_mtime_ns": trajectory.directory_mtime_ns,
                    "status_index": status_index,
                }
            )
            offset += trajectory.length
            status_index += 1
        cached_sources.append(
            {
                "dataset_id": source_id,
                "display_name": source.display_name,
                "data_root": str(source.data_root),
                "split_manifest": str(source.split_manifest),
                "manifest_sha256": hashlib.sha256(
                    source.split_manifest.read_bytes()
                ).hexdigest(),
                "anchor_count": source.anchor_count,
                "feature_file": f"source_{source_id}.fp16",
                "trajectories": trajectories,
            }
        )
    return {
        "version": CACHE_VERSION,
        "complete": True,
        "model_name": "dinov3_vitl16",
        "weights_sha256": EXPECTED_SHA256,
        "source_revision": EXPECTED_DINO_REVISION,
        "dtype": "float16",
        "patch_count": PATCH_COUNT,
        "feature_dim": FEATURE_DIM,
        "image_height": 240,
        "image_width": 320,
        "color_aug": False,
        "sources": cached_sources,
        "trajectory_count": status_index,
        "frame_count": sum(source.anchor_count for source in sources),
    }


def _cache_key(payload: Mapping[str, Any]) -> str:
    identity = dict(payload)
    identity.pop("complete", None)
    identity.pop("completed_at_utc", None)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _work_units(sources: Sequence[Any], payload: Mapping[str, Any]) -> list[_WorkUnit]:
    units = []
    for source, cached_source in zip(sources, payload["sources"], strict=True):
        for trajectory_index, (trajectory, cached_trajectory) in enumerate(
            zip(source.trajectories, cached_source["trajectories"], strict=True)
        ):
            units.append(
                _WorkUnit(
                    source_id=cached_source["dataset_id"],
                    trajectory_index=trajectory_index,
                    status_index=cached_trajectory["status_index"],
                    offset=cached_trajectory["offset"],
                    length=trajectory.length,
                    directory=trajectory.directory,
                )
            )
    return units


def _balanced_assignments(
    units: Sequence[_WorkUnit], world_size: int
) -> tuple[tuple[_WorkUnit, ...], ...]:
    assignments: list[list[_WorkUnit]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for unit in sorted(units, key=lambda item: (-item.length, item.status_index)):
        rank = min(range(world_size), key=lambda item: (loads[item], item))
        assignments[rank].append(unit)
        loads[rank] += unit.length
    for assignment in assignments:
        assignment.sort(key=lambda item: (item.source_id, item.offset))
    return tuple(tuple(assignment) for assignment in assignments)


def _validate_complete_cache(root: Path, payload: Mapping[str, Any]) -> bool:
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        return False
    existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    if _cache_key(existing) != _cache_key(payload) or existing.get("complete") is not True:
        raise RuntimeError(f"Existing DINOv3 cache metadata does not match: {metadata_path}")
    for source in payload["sources"]:
        path = root / source["feature_file"]
        expected_size = (
            source["anchor_count"]
            * PATCH_COUNT
            * FEATURE_DIM
            * FEATURE_DTYPE.itemsize
        )
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(f"Complete DINOv3 cache has a bad feature file: {path}")
    return True


def _prepare_storage(root: Path, payload: Mapping[str, Any]) -> str:
    root.mkdir(parents=True, exist_ok=True)
    if _validate_complete_cache(root, payload):
        return "complete"

    in_progress_path = root / "in_progress.json"
    key = _cache_key(payload)
    if in_progress_path.exists():
        existing = json.loads(in_progress_path.read_text(encoding="utf-8"))
        if existing.get("cache_key") != key:
            raise RuntimeError(
                f"Partial DINOv3 cache belongs to different inputs: {in_progress_path}"
            )
        return "resume"

    unexpected = [path for path in root.iterdir() if path.name != "metadata.json"]
    if unexpected:
        raise RuntimeError(
            f"Refusing to overwrite untracked files in new cache root {root}: "
            f"{unexpected[:3]}"
        )
    for source in payload["sources"]:
        path = root / source["feature_file"]
        size = (
            source["anchor_count"]
            * PATCH_COUNT
            * FEATURE_DIM
            * FEATURE_DTYPE.itemsize
        )
        with path.open("xb") as handle:
            handle.truncate(size)
        os.chmod(path, 0o644)
    progress_path = root / "trajectory_progress.uint8"
    with progress_path.open("xb") as handle:
        handle.truncate(payload["trajectory_count"])
    os.chmod(progress_path, 0o644)
    _atomic_json(
        in_progress_path,
        {
            "cache_key": key,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "frame_count": payload["frame_count"],
            "trajectory_count": payload["trajectory_count"],
        },
    )
    return "new"


def _distributed_status(callable_) -> Any:
    status: list[Any] = [None]
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            status[0] = (True, callable_())
        except Exception as exc:  # noqa: BLE001 - propagate rank-zero setup failures
            status[0] = (False, type(exc).__name__, str(exc))
    if dist.is_initialized():
        dist.broadcast_object_list(status, src=0)
    assert status[0] is not None
    if not status[0][0]:
        raise RuntimeError(f"rank-zero setup failed: {status[0][1]}: {status[0][2]}")
    return status[0][1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--dino-repo", type=Path, default=DEFAULT_DINO_REPO)
    parser.add_argument("--weights", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Flush completed trajectories every N batches for bounded resume loss.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if (
        args.batch_size <= 0
        or args.num_workers < 0
        or args.log_every <= 0
        or args.checkpoint_every <= 0
    ):
        raise ValueError(
            "batch-size/log-every/checkpoint-every must be positive and "
            "num-workers non-negative"
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("DINOv3 feature precomputation requires CUDA")
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    base_config = args.base_config.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    dino_repo = args.dino_repo.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    data_config = _load_data_config(base_config)
    dataset = NavigationPairDataset(
        data_config["source_specs"],
        samples_per_epoch=3,
        color_aug=False,
        image_height=240,
        image_width=320,
        print_audit=rank == 0,
        index_cache_path=data_config.get("index_cache_path"),
    )
    sources = dataset._sources
    payload = _cache_payload(sources)

    def validate_inputs() -> str:
        validate_source(dino_repo, EXPECTED_DINO_REVISION)
        validate_checkpoint(weights)
        return _prepare_storage(cache_root, payload)

    storage_state = _distributed_status(validate_inputs)
    if storage_state == "complete":
        if rank == 0:
            print(f"DINOv3 feature cache is already complete: {cache_root}")
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    if dist.is_initialized():
        dist.barrier()

    progress = np.memmap(
        cache_root / "trajectory_progress.uint8",
        dtype=np.uint8,
        mode="r+",
        shape=(payload["trajectory_count"],),
    )
    units = _work_units(sources, payload)
    assignments = _balanced_assignments(units, world_size)
    assigned = tuple(unit for unit in assignments[rank] if progress[unit.status_index] == 0)
    frames = _AssignedFrames(assigned)
    if rank == 0:
        completed = int(np.count_nonzero(progress))
        gib = payload["frame_count"] * PATCH_COUNT * FEATURE_DIM * 2 / 2**30
        print(
            f"DINOv3 cache {storage_state}: {payload['frame_count']:,} frames, "
            f"{payload['trajectory_count']:,} trajectories, {gib:.2f} GiB; "
            f"resuming after {completed:,} completed trajectories on {world_size} GPUs"
        )

    arrays = [
        np.memmap(
            cache_root / source["feature_file"],
            dtype=FEATURE_DTYPE,
            mode="r+",
            shape=(source["anchor_count"], PATCH_COUNT, FEATURE_DIM),
        )
        for source in payload["sources"]
    ]
    loader = DataLoader(
        frames,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    model = load_strict_backbone(dino_repo, weights, device)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    processed = 0
    durable_unit_cursor = 0

    def checkpoint_completed_trajectories(force: bool = False) -> None:
        nonlocal durable_unit_cursor
        completed_cursor = bisect.bisect_right(frames.cumulative, processed)
        if (
            completed_cursor == durable_unit_cursor
            or (not force and batch_index % args.checkpoint_every != 0)
        ):
            return
        # Publication order matters: feature pages must be durable before their
        # trajectory completion bytes become visible to a resumed launch.
        for array in arrays:
            array.flush()
        for unit in assigned[durable_unit_cursor:completed_cursor]:
            progress[unit.status_index] = 1
        progress.flush()
        durable_unit_cursor = completed_cursor

    batch_index = 0
    with torch.inference_mode():
        for batch_index, (source_ids, indices, images) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True).permute(0, 3, 1, 2)
            images = (images - mean) / std
            with torch.autocast("cuda", dtype=torch.float16):
                tokens = model.forward_features(images)["x_norm_patchtokens"]
            expected_shape = (images.shape[0], PATCH_COUNT, FEATURE_DIM)
            if tuple(tokens.shape) != expected_shape or not torch.isfinite(tokens).all():
                raise RuntimeError(
                    f"Bad DINOv3 output on rank {rank}: {tuple(tokens.shape)}"
                )
            tokens_numpy = tokens.to(dtype=torch.float16).cpu().numpy()
            source_numpy = source_ids.numpy()
            index_numpy = indices.numpy()
            for source_id in np.unique(source_numpy):
                mask = source_numpy == source_id
                arrays[int(source_id)][index_numpy[mask]] = tokens_numpy[mask]
            processed += images.shape[0]
            checkpoint_completed_trajectories()
            if batch_index % args.log_every == 0:
                elapsed = time.monotonic() - started
                print(
                    f"[rank {rank}] {processed:,}/{len(frames):,} frames "
                    f"({processed / elapsed:.1f} frames/s)"
                )

    checkpoint_completed_trajectories(force=True)
    if durable_unit_cursor != len(assigned):
        raise RuntimeError(
            f"rank {rank} processed all frames but only made "
            f"{durable_unit_cursor}/{len(assigned)} trajectories durable"
        )
    if dist.is_initialized():
        dist.barrier()

    def finalize() -> str:
        final_progress = np.memmap(
            cache_root / "trajectory_progress.uint8",
            dtype=np.uint8,
            mode="r",
            shape=(payload["trajectory_count"],),
        )
        missing = int(np.count_nonzero(final_progress == 0))
        if missing:
            raise RuntimeError(f"DINOv3 cache still has {missing} incomplete trajectories")
        final_payload = dict(payload)
        final_payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(cache_root / "metadata.json", final_payload)
        (cache_root / "in_progress.json").unlink(missing_ok=True)
        return str(cache_root / "metadata.json")

    metadata_path = _distributed_status(finalize)
    if rank == 0:
        elapsed = time.monotonic() - started
        print(
            f"Complete DINOv3 feature cache: {metadata_path}; "
            f"this launch processed {sum(len(_AssignedFrames(a)) for a in assignments):,} "
            f"assigned frames in {elapsed / 60:.1f} min"
        )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
