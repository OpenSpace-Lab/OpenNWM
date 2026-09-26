#!/usr/bin/env python3
"""Deterministic, topology-independent randomness for NWM benchmarks.

Benchmark samples must keep the same stochastic input when the number of GPUs,
the rank assignment, or the batch size changes.  The helpers in this module
derive one CUDA/CPU generator seed from the semantic identity of each sample
instead of consuming a process-global random stream.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

import torch


MAX_TORCH_SEED = 2**63 - 1


def stable_seed(base_seed: int, *parts: Any) -> int:
    """Return a stable torch seed for an ordered semantic identity."""

    digest = hashlib.sha256()
    for value in (int(base_seed), *parts):
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "little") % MAX_TORCH_SEED


def normalize_sample_keys(sample_keys: Any) -> list[str]:
    """Normalize tensor/list sample identities without losing their order."""

    if isinstance(sample_keys, torch.Tensor):
        values = sample_keys.detach().cpu().reshape(-1).tolist()
    elif isinstance(sample_keys, (str, bytes)):
        values = [sample_keys]
    elif isinstance(sample_keys, Iterable):
        values = list(sample_keys)
    else:
        values = [sample_keys]
    normalized = []
    for value in values:
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        normalized.append(str(value))
    return normalized


def samplewise_randn(
    sample_keys: Any,
    sample_shape: Sequence[int],
    *,
    base_seed: int,
    stream: str,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate one independent normal tensor per semantic sample key."""

    keys = normalize_sample_keys(sample_keys)
    target_device = torch.device(device)
    samples = []
    for key in keys:
        generator = torch.Generator(device=target_device)
        generator.manual_seed(stable_seed(base_seed, stream, key))
        samples.append(
            torch.randn(
                tuple(int(value) for value in sample_shape),
                generator=generator,
                device=target_device,
                dtype=dtype,
            )
        )
    if not samples:
        return torch.empty((0, *sample_shape), device=target_device, dtype=dtype)
    return torch.stack(samples, dim=0)


def samplewise_noise_schedule(
    sample_keys: Any,
    sample_shape: Sequence[int],
    *,
    draws: int,
    base_seed: int,
    stream: str,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``[draw, sample, ...]`` noise independent of batch partitioning."""

    if draws < 1:
        raise ValueError(f"draws must be positive, got {draws}")
    keys = normalize_sample_keys(sample_keys)
    target_device = torch.device(device)
    per_sample = []
    draw_shape = (int(draws), *(int(value) for value in sample_shape))
    for key in keys:
        generator = torch.Generator(device=target_device)
        generator.manual_seed(stable_seed(base_seed, stream, key))
        per_sample.append(
            torch.randn(
                draw_shape,
                generator=generator,
                device=target_device,
                dtype=dtype,
            )
        )
    if not per_sample:
        return torch.empty(
            (draws, 0, *sample_shape), device=target_device, dtype=dtype
        )
    return torch.stack(per_sample, dim=1)


def expand_sample_keys(sample_keys: Any, repeats: int) -> list[str]:
    """Expand batch keys with stable child identities for multi-goal inference."""

    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")
    return [
        f"{sample_key}/goal={goal_index}"
        for sample_key in normalize_sample_keys(sample_keys)
        for goal_index in range(repeats)
    ]
