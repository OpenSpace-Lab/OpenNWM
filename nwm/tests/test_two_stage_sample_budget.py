"""CPU tests for opt-in large-batch sample accounting."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from omegaconf import OmegaConf

from data_utils import ReferenceBatchSampler
from motion_condition import flatten_motion_groups, motion_condition_collate
from two_stage_checkpoint import (
    build_data_resume_fingerprint,
    validate_data_resume_fingerprint,
)
from two_stage_nwm import ProxyMetrics
from two_stage_training import (
    _budget_training_batch,
    _gather_runtime_states,
    _phase_complete,
    _phase_target_samples_per_rank,
    _truncate_training_batch,
)


ROOT = Path(__file__).resolve().parents[1]


class _RangeSampler:
    def __init__(self, length: int) -> None:
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        return iter(range(self.length))


def _sample(index: int, motion_type: str) -> dict:
    sample = {
        "image": torch.tensor([float(index)]),
        "k": torch.tensor([-1, 1]),
        "motion_type": motion_type,
        "sample_name": f"sample-{index}",
    }
    if motion_type == "latent":
        sample["motion"] = torch.full((2, 4), float(index))
        sample["motion_mask"] = torch.tensor([True, index % 2 == 0])
    elif motion_type == "real":
        sample["motion"] = torch.full((2, 3), float(index))
    return sample


def test_final_batch_crop_preserves_motion_group_indices() -> None:
    batch = motion_condition_collate(
        [
            _sample(0, "latent"),
            _sample(1, "none"),
            _sample(2, "real"),
            _sample(3, "latent"),
            _sample(4, "real"),
        ]
    )
    cropped = _truncate_training_batch(batch, 3)

    assert tuple(cropped["k"].shape) == (3, 2)
    assert cropped["motion_type"] == ["latent", "none", "real"]
    assert cropped["sample_name"] == ["sample-0", "sample-1", "sample-2"]
    assert torch.equal(
        cropped["motion"]["latent"]["sample_indices"], torch.tensor([0])
    )
    assert torch.equal(
        cropped["motion"]["real"]["sample_indices"], torch.tensor([2])
    )
    flattened = flatten_motion_groups(
        cropped["motion"],
        batch_size=3,
        num_goals=2,
        device=torch.device("cpu"),
    )
    assert flattened is not None
    assert torch.equal(flattened["latent"]["indices"], torch.tensor([0, 1]))
    assert torch.equal(flattened["real"]["indices"], torch.tensor([4, 5]))


def test_budget_helper_hits_exact_samples_with_partial_final_batch() -> None:
    batch = motion_condition_collate([_sample(i, "real") for i in range(5)])
    cropped, local_batch_size = _budget_training_batch(
        batch,
        completed_samples_per_rank=8,
        target_samples_per_rank=10,
    )
    assert local_batch_size == 2
    assert tuple(cropped["k"].shape) == (2, 2)
    assert _phase_complete(3, 4, 10, 10)
    assert not _phase_complete(3, 4, 9, 10)
    assert _phase_complete(4, 4, 0, None)


def test_nav1_reference_rebatch_has_exact_33336_step_budget() -> None:
    # DistributedSampler pads 4,132,468 observations to 516,559/rank. The old
    # batch16/drop-last contract keeps 516,544 of them in every full epoch.
    sampler = ReferenceBatchSampler(
        _RangeSampler(516_559),
        batch_size=96,
        reference_batch_size=16,
    )
    epoch_batch_sizes = [len(batch) for batch in sampler]
    assert len(epoch_batch_sizes) == 5_381
    assert epoch_batch_sizes[-1] == 64
    assert sum(epoch_batch_sizes) == 516_544

    completed = 0
    steps = 0
    while completed < 3_200_000:
        for batch_size in epoch_batch_sizes:
            used = min(batch_size, 3_200_000 - completed)
            completed += used
            steps += 1
            if completed == 3_200_000:
                break
    assert completed == 3_200_000
    assert steps == 33_336




def test_phase_targets_and_runtime_checkpoint_counter_are_opt_in() -> None:
    config = OmegaConf.create(
        {
            "training": {"target_samples_per_rank": 3_200_000},
            "finetune": {
                "warmup_samples_per_rank": 160_000,
                "joint_samples_per_rank": 1_600_000,
            },
        }
    )
    assert (
        _phase_target_samples_per_rank(config, "proxy_pretrain", "pretrain")
        == 3_200_000
    )
    assert (
        _phase_target_samples_per_rank(config, "real_finetune", "warmup")
        == 160_000
    )
    assert (
        _phase_target_samples_per_rank(config, "real_finetune", "joint")
        == 1_600_000
    )
    states = _gather_runtime_states(ProxyMetrics(), 0, 123_456)
    assert states[0]["completed_samples_per_rank"] == 123_456
    config.training.batch_size = 96
    config.training.num_workers = 8
    config.training.reference_batch_size = 16
    config.seed = 20260901
    config.training_stage = "proxy_pretrain"
    config.action_mode = "latent"
    config.dataset = {"fixture": True}
    config.dataset_selection = {"pretrain": ["navanywhere"]}
    config.proxy = {}
    config.motion_condition = {"enabled": True}
    fingerprint = build_data_resume_fingerprint(config, "pretrain")
    assert fingerprint["payload"]["target_samples_per_rank"] == 3_200_000
    changed = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    changed.training.target_samples_per_rank += 16
    try:
        validate_data_resume_fingerprint(
            fingerprint, changed, "pretrain", required=True
        )
    except ValueError as error:
        assert "fingerprint mismatch" in str(error)
    else:
        raise AssertionError("A changed exact sample budget must reject resume")


