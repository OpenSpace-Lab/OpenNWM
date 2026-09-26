from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from lam.navigation_dataset import (  # noqa: E402
    NavigationPairDataset,
    _NavigationRequestSampler,
    _permuted_anchor,
)


def _frames(directory: Path, count: int, value: int = 32) -> None:
    directory.mkdir(parents=True)
    for index in range(count):
        Image.new("RGB", (16, 12), (value + index % 16,) * 3).save(
            directory / f"{index}.jpg"
        )


def _frame_source(tmp_path: Path, name: str, lengths: tuple[int, ...]) -> dict:
    root = tmp_path / name
    for index, length in enumerate(lengths):
        _frames(root / f"trajectory-{index}", length, 16 + index * 32)
    return {
        "data_root": str(root),
        "display_name": name,
        "format": "frame_only",
        "has_action": False,
        "waypoint_spacing": 1.0,
    }


def _action_source(tmp_path: Path, name: str, count: int) -> dict:
    root = tmp_path / name
    trajectory = root / "trajectory"
    _frames(trajectory, count, 96)
    with (trajectory / "traj_data.pkl").open("wb") as handle:
        pickle.dump(
            {
                "position": np.stack((np.arange(count), np.zeros(count)), axis=1),
                "yaw": np.zeros(count),
            },
            handle,
        )
    manifest = tmp_path / "splits" / name / "train" / "traj_names.txt"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("trajectory\n", encoding="utf-8")
    return {
        "data_root": str(root),
        "split_manifest": str(manifest),
        "display_name": name,
        "format": "pose_labeled",
        "has_action": True,
        "waypoint_spacing": 1.0,
    }


def test_frame_only_source_accepts_public_root_key(tmp_path: Path) -> None:
    spec = _frame_source(tmp_path, "root-alias", (4,))
    spec["root"] = spec.pop("data_root")
    dataset = NavigationPairDataset(
        [spec],
        samples_per_epoch=8,
        max_frame_offset=1,
        color_aug=False,
        print_audit=False,
        dataset_size_exponent=0.5,
        uniform_offset_domain=True,
    )
    assert dataset.source_anchor_counts == (4,)


def _mixed_dataset(tmp_path: Path) -> NavigationPairDataset:
    return NavigationPairDataset(
        [
            _frame_source(tmp_path, "small", (16,)),
            _frame_source(tmp_path, "large", (64,)),
            _action_source(tmp_path, "action", 16),
        ],
        samples_per_epoch=131_072,
        max_frame_offset=8,
        data_seed=32,
        color_aug=False,
        print_audit=False,
        dataset_size_exponent=0.5,
        uniform_offset_domain=True,
    )


def test_mixed_schedule_has_uniform_17_offsets_and_sqrt_source_weights(
    tmp_path: Path,
) -> None:
    dataset = _mixed_dataset(tmp_path)
    schedule = dataset._schedules[-1]

    assert len(schedule.entries) == 17 * 4096
    assert set(schedule.offset_counts) == set(range(-8, 9))
    assert set(schedule.offset_counts.values()) == {4096}
    ratios = {
        source_id: count / len(schedule.entries)
        for source_id, count in schedule.source_counts.items()
    }
    # Anchor counts are 16:64:16, therefore the intended sqrt weights are 1:2:1.
    assert abs(ratios[0] - 0.25) < 0.002
    assert abs(ratios[1] - 0.50) < 0.002
    assert abs(ratios[2] - 0.25) < 0.002


def test_frame_only_samples_use_zero_action_and_labeled_samples_keep_action(
    tmp_path: Path,
) -> None:
    dataset = _mixed_dataset(tmp_path)
    unlabeled = dataset[(0, 0)]
    labeled = dataset[(0, 1)]

    assert unlabeled["action_mask"].item() is False
    assert torch.equal(unlabeled["actions"], torch.zeros(1, 3))
    assert labeled["action_mask"].item() is True
    assert labeled["actions"].shape == (1, 3)
    assert unlabeled["videos"].shape == labeled["videos"].shape == (2, 240, 320, 3)
    assert -8 <= unlabeled["frame_offset"].item() <= 8
    assert -8 <= labeled["frame_offset"].item() <= 8


def test_long_trajectory_receives_more_distinct_anchor_samples(tmp_path: Path) -> None:
    dataset = NavigationPairDataset(
        [_frame_source(tmp_path, "frames", (9, 90))],
        samples_per_epoch=4096,
        max_frame_offset=0,
        color_aug=False,
        print_audit=False,
        dataset_size_exponent=0.5,
        uniform_offset_domain=True,
    )
    source = dataset._sources[0]
    seen = Counter()
    for occurrence in range(source.valid_anchor_count(0)):
        anchor_index = _permuted_anchor(occurrence, source.valid_anchor_count(0), 123)
        trajectory, _ = source.valid_anchor(0, anchor_index)
        seen[trajectory.name] += 1
    assert seen == {"trajectory-0": 9, "trajectory-1": 90}


def test_variant4_sampler_is_exactly_25_percent_per_rank_and_reproducible(
    tmp_path: Path,
) -> None:
    dataset = _mixed_dataset(tmp_path)
    rank_requests = []
    for rank in range(2):
        sampler = _NavigationRequestSampler(
            dataset,
            batch_size=32,
            labeled_batch_fraction=0.25,
            num_replicas=2,
            rank=rank,
            seed=32,
            drop_last=True,
        )
        sampler.set_epoch(3)
        requests = list(sampler)[:64]
        assert all(
            sum(group == 1 for _, group in requests[start : start + 32]) == 8
            for start in (0, 32)
        )
        duplicate = _NavigationRequestSampler(
            dataset,
            batch_size=32,
            labeled_batch_fraction=0.25,
            num_replicas=2,
            rank=rank,
            seed=32,
            drop_last=True,
        )
        duplicate.set_epoch(3)
        assert list(duplicate)[:64] == requests
        rank_requests.append(set(requests))
    assert rank_requests[0].isdisjoint(rank_requests[1])


def test_frame_only_index_cache_requires_explicit_rebuild_for_added_trajectory(
    tmp_path: Path,
) -> None:
    spec = _frame_source(tmp_path, "frames", (9,))
    cache = tmp_path / "index.pkl"
    first = NavigationPairDataset(
        [spec],
        samples_per_epoch=32,
        max_frame_offset=0,
        color_aug=False,
        print_audit=False,
        index_cache_path=str(cache),
    )
    second = NavigationPairDataset(
        [spec],
        samples_per_epoch=32,
        max_frame_offset=0,
        color_aug=False,
        print_audit=False,
        index_cache_path=str(cache),
    )
    assert first.audit["loaded_from_index_cache"] is False
    assert second.audit["loaded_from_index_cache"] is True

    _frames(Path(spec["data_root"]) / "trajectory-new", 11)
    changed = NavigationPairDataset(
        [spec],
        samples_per_epoch=32,
        max_frame_offset=0,
        color_aug=False,
        print_audit=False,
        index_cache_path=str(cache),
    )
    assert changed.audit["loaded_from_index_cache"] is True
    assert changed.source_anchor_counts == (9,)

    rebuilt = NavigationPairDataset(
        [spec],
        samples_per_epoch=32,
        max_frame_offset=0,
        color_aug=False,
        print_audit=False,
        index_cache_path=str(cache),
        rebuild_index_cache=True,
    )
    assert rebuilt.audit["loaded_from_index_cache"] is False
    assert rebuilt.source_anchor_counts == (20,)
