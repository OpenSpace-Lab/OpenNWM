"""Fast CPU tests for NavAnywhere offline proxy data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import default_collate

import navanywhere_recipe
import two_stage_data
from navanywhere_recipe import build_sampling_recipe, frame_indices_sha256
from two_stage_data import (
    NavAnywhereDataset,
    OfflineProxyStore,
    ProxyLookupError,
    proxy_offset_mask,
)


def _write_frames(
    root: Path,
    *,
    source: str = "nav_source",
    trajectory: str = "trajectory_001",
    first: int = 0,
    last: int = 128,
    pixel_base: int = 0,
) -> Path:
    trajectory_path = root / source / trajectory
    trajectory_path.mkdir(parents=True)
    for frame_index in range(first, last + 1):
        value = pixel_base + frame_index
        Image.new("RGB", (3, 2), color=(value, value, value)).save(
            trajectory_path / f"{frame_index:06d}.jpg",
            quality=100,
            subsampling=0,
        )
    return trajectory_path


def _pixel_transform(image: Image.Image) -> torch.Tensor:
    return torch.tensor(float(image.getpixel((0, 0))[0])).reshape(1, 1, 1)


def _write_pt_cache(
    root: Path,
    pairs: list[list[int]],
    motion: list[list[float]] | torch.Tensor,
    *,
    proxy_type: str = "latent",
    source: str = "nav_source",
    trajectory: str = "trajectory_001",
) -> Path:
    path = root / source / f"{trajectory}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "proxy_type": proxy_type,
            "source_id": source,
            "trajectory_id": trajectory,
            "frame_pairs": torch.tensor(pairs, dtype=torch.int64),
            "motion": torch.as_tensor(motion, dtype=torch.float32),
        },
        path,
    )
    return path


def test_dataset_retries_transient_duplicate_directory_entries(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "NavAnywhere"
    trajectory = _write_frames(data_root, first=0, last=1)
    real_scan = navanywhere_recipe._scan_trajectory_frames_once
    calls = 0

    def flaky_scan(path: str):
        nonlocal calls
        calls += 1
        frames = real_scan(path)
        if Path(path) == trajectory and calls == 1:
            return [frames[0], frames[0], *frames[1:]]
        return frames

    monkeypatch.setattr(navanywhere_recipe, "_scan_trajectory_frames_once", flaky_scan)
    monkeypatch.setattr(navanywhere_recipe, "FRAME_SCAN_RETRY_DELAY_SECONDS", 0)

    dataset = NavAnywhereDataset(
        data_root,
        source_id="nav_source",
        context_size=1,
        fixed_goal_offsets=[0],
    )

    assert dataset._trajectories[0].frame_indices.tolist() == [0, 1]
    assert calls == 2


def test_precomputed_recipe_startup_does_not_rescan_raw_frames(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "NavAnywhere"
    _write_frames(data_root, first=0, last=5)
    recipe = build_sampling_recipe(
        data_root,
        seed=23,
        context_size=1,
        goals_per_obs=1,
    )
    latent_root = tmp_path / "latents"
    latent_root.mkdir()

    def fail_scan(path: str):
        raise AssertionError(f"raw frame scan was called for {path}")

    monkeypatch.setattr(two_stage_data, "scan_trajectory_frames", fail_scan)
    dataset = NavAnywhereDataset(
        data_root,
        sampling_recipe=recipe,
        context_size=1,
        goals_per_obs=1,
        seed=23,
        precomputed_latent_root=latent_root,
        precomputed_latent_records={"nav_source": {"trajectory_001": {}}},
    )

    assert dataset._trajectories[0].frame_indices.tolist() == list(range(6))
    assert dataset._trajectories[0].frame_paths == ()


def test_proxy_offset_mask_keeps_full_signed_range_but_marks_only_local() -> None:
    offsets = torch.tensor([-64, -9, -8, -1, 0, 1, 8, 9, 64])
    assert proxy_offset_mask(offsets, 8).tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize(
    ("proxy_type", "extension"),
    [
        ("geometry", ".pt"),
        ("idm", ".npz"),
        ("latent", ".pt"),
    ],
)
def test_store_reads_pt_and_npz_for_each_proxy_type(
    tmp_path: Path, proxy_type: str, extension: str
) -> None:
    cache_root = tmp_path / "proxy"
    cache_path = cache_root / "source_a" / f"traj_a{extension}"
    cache_path.parent.mkdir(parents=True)
    payload = {
        "proxy_type": proxy_type,
        "source_id": "source_a",
        "trajectory_id": "traj_a",
        "frame_pairs": np.asarray([[10, 12], [10, 11]], dtype=np.int64),
        "motion": np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
    }
    if extension == ".pt":
        torch.save(
            {
                key: torch.from_numpy(value)
                if isinstance(value, np.ndarray) and value.ndim > 0
                else value
                for key, value in payload.items()
            },
            cache_path,
        )
    else:
        np.savez(cache_path, **payload)

    store = OfflineProxyStore(
        root=cache_root,
        proxy_type=proxy_type,
        dim=2,
        strict_loading=True,
    )
    result = store.lookup(
        dataset_name="source_a",
        trajectory_name="traj_a",
        current_frame=10,
        target_frame=11,
    )
    assert result.valid and result.found and not result.invalid
    assert torch.equal(result.proxy_action, torch.tensor([4.0, 5.0]))
    assert "source_id='source_a'" in result.sample_key
    assert "trajectory_id='traj_a'" in result.sample_key


def test_tolerant_store_distinguishes_found_missing_and_invalid(tmp_path: Path) -> None:
    cache_root = tmp_path / "proxy"
    _write_pt_cache(
        cache_root,
        [[10, 11], [10, 12]],
        [[1.0, 2.0], [float("nan"), 3.0]],
    )
    store = OfflineProxyStore(
        root=cache_root,
        proxy_type="latent",
        dim=2,
        strict_loading=False,
    )

    found = store.lookup("nav_source", "trajectory_001", 10, 11)
    invalid = store.lookup("nav_source", "trajectory_001", 10, 12)
    missing_record = store.lookup("nav_source", "trajectory_001", 10, 13)
    missing_file = store.lookup("nav_source", "another_trajectory", 10, 11)

    assert found.valid and found.found and not found.invalid and not found.missing
    assert invalid.found and invalid.invalid and not invalid.valid
    assert not missing_record.found and not missing_record.invalid
    assert missing_record.missing
    assert not missing_file.found and not missing_file.invalid
    assert missing_file.missing
    assert torch.equal(invalid.proxy_action, torch.zeros(2))
    assert torch.equal(missing_record.proxy_action, torch.zeros(2))


def test_store_honours_precomputed_per_record_validity_mask(tmp_path: Path) -> None:
    cache_root = tmp_path / "proxy"
    path = _write_pt_cache(
        cache_root,
        [[10, 11], [10, 12]],
        [[1.0, 2.0], [3.0, 4.0]],
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["proxy_valid"] = torch.tensor([False, True])
    torch.save(payload, path)
    store = OfflineProxyStore(
        root=cache_root,
        proxy_type="latent",
        dim=2,
        strict_loading=False,
    )

    rejected = store.lookup("nav_source", "trajectory_001", 10, 11)
    accepted = store.lookup("nav_source", "trajectory_001", 10, 12)
    assert rejected.found and rejected.invalid and not rejected.valid
    assert accepted.valid
    assert torch.equal(accepted.proxy_action, torch.tensor([3.0, 4.0]))


def test_wrong_shape_is_invalid_and_strict_error_has_complete_sample_key(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "proxy"
    _write_pt_cache(cache_root, [[3, 4]], [[1.0, 2.0, 3.0]])
    tolerant = OfflineProxyStore(
        root=cache_root,
        proxy_type="latent",
        dim=2,
        strict_loading=False,
    )
    result = tolerant.lookup("nav_source", "trajectory_001", 3, 4)
    assert result.found and result.invalid and not result.valid
    assert "shape" in (result.error or "")

    strict = OfflineProxyStore(
        root=cache_root,
        proxy_type="latent",
        dim=2,
        strict_loading=True,
    )
    with pytest.raises(ProxyLookupError) as error:
        strict.lookup("nav_source", "trajectory_001", 3, 4)
    message = str(error.value)
    assert "source_id='nav_source'" in message
    assert "trajectory_id='trajectory_001'" in message
    assert "current_frame=3" in message
    assert "target_frame=4" in message

    with pytest.raises(ProxyLookupError) as missing_error:
        strict.lookup("missing_source", "missing_trajectory", 7, 8)
    missing_message = str(missing_error.value)
    for fragment in (
        "source_id='missing_source'",
        "trajectory_id='missing_trajectory'",
        "current_frame=7",
        "target_frame=8",
    ):
        assert fragment in missing_message


def test_navanywhere_local_proxy_lookup_and_long_range_zero_io(tmp_path: Path) -> None:
    data_root = tmp_path / "NavAnywhere"
    _write_frames(data_root)
    cache_root = tmp_path / "proxy"
    _write_pt_cache(
        cache_root,
        [[64, 56], [64, 72]],
        [[-8.0, 56.0], [8.0, 72.0]],
    )
    store = OfflineProxyStore(
        root=cache_root,
        proxy_type="latent",
        dim=2,
        strict_loading=True,
    )
    calls: list[tuple[object, ...]] = []
    original_lookup = store.lookup

    def recording_lookup(*args: object, **kwargs: object):
        calls.append(args)
        return original_lookup(*args, **kwargs)

    store.lookup = recording_lookup  # type: ignore[method-assign]
    dataset = NavAnywhereDataset(
        data_root,
        source_id="nav_source",
        context_size=1,
        goals_per_obs=4,
        fixed_goal_offsets=[-9, -8, 8, 9],
        action_mode="latent",
        proxy_store=store,
        transform=_pixel_transform,
    )

    # With all four fixed offsets available, local dataset index 55 is frame 64.
    sample = dataset[55]
    assert sample["current_frame"].item() == 64
    assert sample["frame_offset"].tolist() == [-9, -8, 8, 9]
    assert sample["k"].tolist() == pytest.approx(
        [-9 / 128, -8 / 128, 8 / 128, 9 / 128]
    )
    assert sample["proxy_eligible"].tolist() == [False, True, True, False]
    assert sample["proxy_found"].tolist() == [False, True, True, False]
    assert sample["proxy_invalid"].tolist() == [False, False, False, False]
    assert sample["proxy_valid"].tolist() == [False, True, True, False]
    assert sample["motion_mask"].tolist() == [False, True, True, False]
    assert torch.equal(
        sample["motion"],
        torch.tensor([[0.0, 0.0], [-8.0, 56.0], [8.0, 72.0], [0.0, 0.0]]),
    )
    assert torch.equal(sample["proxy_action"], sample["motion"])
    assert [int(round(value)) for value in sample["video"][1:, 0, 0, 0].tolist()] == [
        55,
        56,
        72,
        73,
    ]
    assert [int(call[3]) for call in calls] == [56, 72]


def test_all_long_range_goals_do_not_touch_proxy_store(tmp_path: Path) -> None:
    data_root = tmp_path / "NavAnywhere"
    _write_frames(data_root)
    cache_root = tmp_path / "empty_proxy"
    cache_root.mkdir()
    store = OfflineProxyStore(
        root=cache_root,
        proxy_type="geometry",
        dim=3,
        strict_loading=True,
    )
    calls = 0
    original_lookup = store.lookup

    def recording_lookup(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original_lookup(*args, **kwargs)

    store.lookup = recording_lookup  # type: ignore[method-assign]
    dataset = NavAnywhereDataset(
        data_root,
        source_id="nav_source",
        context_size=1,
        goals_per_obs=2,
        fixed_goal_offsets=[-64, 64],
        action_mode="geometry",
        proxy_store=store,
        transform=_pixel_transform,
    )
    sample = dataset[0]
    assert sample["current_frame"].item() == 64
    assert sample["frame_offset"].tolist() == [-64, 64]
    assert not sample["proxy_eligible"].any()
    assert not sample["proxy_valid"].any()
    assert torch.equal(sample["motion"], torch.zeros(2, 3))
    assert calls == 0


def test_multigoal_default_collation_preserves_every_alignment(tmp_path: Path) -> None:
    data_root = tmp_path / "NavAnywhere"
    trajectory_path = _write_frames(data_root, first=0, last=5)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "trajectories": [
                    {
                        "source_id": "nav_source",
                        "trajectory_id": "trajectory_001",
                        "frames": [path.name for path in sorted(trajectory_path.glob("*.jpg"))],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cache_root = tmp_path / "proxy"
    pairs = []
    values = []
    for current in range(1, 5):
        for offset in (-1, 1):
            target = current + offset
            pairs.append([current, target])
            values.append([float(target), float(offset)])
    _write_pt_cache(cache_root, pairs, values, proxy_type="idm")
    store = OfflineProxyStore(
        root=cache_root,
        proxy_type="idm",
        dim=2,
        strict_loading=True,
    )
    dataset = NavAnywhereDataset(
        data_root,
        manifest=manifest_path,
        context_size=1,
        goals_per_obs=2,
        fixed_goal_offsets=[-1, 1],
        action_mode="idm",
        proxy_store=store,
        transform=_pixel_transform,
    )

    batch = default_collate([dataset[0], dataset[1]])
    target_pixels = batch["video"][:, 1:, 0, 0, 0].reshape(-1)
    target_frames = batch["target_frame"].reshape(-1).float()
    frame_offsets = batch["frame_offset"].reshape(-1).float()
    rel_t = batch["rel_t"].reshape(-1)
    proxy_target = batch["proxy_action"][..., 0].reshape(-1)
    proxy_offset = batch["proxy_action"][..., 1].reshape(-1)
    proxy_valid = batch["proxy_valid"].reshape(-1)

    assert torch.allclose(target_pixels, target_frames, atol=1.0)
    assert torch.equal(proxy_target, target_frames)
    assert torch.equal(proxy_offset, frame_offsets)
    assert torch.allclose(rel_t, frame_offsets / 128.0)
    assert proxy_valid.tolist() == [True, True, True, True]


def test_none_mode_has_full_default_offset_range_and_no_motion_fields(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "NavAnywhere"
    _write_frames(data_root, first=0, last=2)
    dataset = NavAnywhereDataset(
        data_root,
        source_id="nav_source",
        action_mode="none",
        context_size=1,
        goals_per_obs=1,
        seed=123,
        transform=_pixel_transform,
    )
    sample = dataset[0]
    assert dataset.min_frame_offset == -64
    assert dataset.max_frame_offset == 64
    assert sample["motion_type"] == "none"
    assert sample["action_conditioning_enabled"] is False
    assert "motion" not in sample
    assert "motion_mask" not in sample
    assert "proxy_action" not in sample
    assert not sample["proxy_valid"].any()


def test_shared_recipe_balances_sources_and_replays_identical_pairs(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "NavAnywhere"
    _write_frames(data_root, source="source_a", trajectory="a0", last=32)
    _write_frames(data_root, source="source_a", trajectory="a1", last=15)
    _write_frames(data_root, source="source_b", trajectory="b0", last=48)
    recipe = build_sampling_recipe(
        data_root,
        seed=31415,
        context_size=1,
        goals_per_obs=4,
        samples_per_epoch=96,
    )
    stores = {}
    for mode, dim in (("geometry", 3), ("idm", 7), ("latent", 32)):
        root = tmp_path / f"{mode}_proxy"
        root.mkdir()
        stores[mode] = OfflineProxyStore(
            root=root,
            proxy_type=mode,
            dim=dim,
            strict_loading=False,
        )
    datasets = {
        "none": NavAnywhereDataset(
            data_root,
            context_size=1,
            goals_per_obs=4,
            action_mode="none",
            seed=31415,
            sampling_recipe=recipe,
            transform=_pixel_transform,
        )
    }
    for mode, store in stores.items():
        datasets[mode] = NavAnywhereDataset(
            data_root,
            context_size=1,
            goals_per_obs=4,
            action_mode=mode,
            proxy_store=store,
            seed=31415,
            sampling_recipe=recipe,
            transform=_pixel_transform,
        )

    reference = []
    source_counts: dict[str, int] = {}
    trajectory_counts: dict[tuple[str, str], int] = {}
    observations: dict[tuple[str, str], set[int]] = {}
    for index in range(len(datasets["none"])):
        sample = datasets["none"][index]
        identity = (sample["source_id"], sample["trajectory_id"])
        source_counts[identity[0]] = source_counts.get(identity[0], 0) + 1
        trajectory_counts[identity] = trajectory_counts.get(identity, 0) + 1
        observations.setdefault(identity, set()).add(int(sample["current_frame"]))
        reference.append(
            (
                *identity,
                int(sample["current_frame"]),
                tuple(sample["target_frame"].tolist()),
                tuple(sample["frame_offset"].tolist()),
            )
        )
    assert source_counts == {"source_a": 48, "source_b": 48}
    assert trajectory_counts[("source_a", "a0")] == 24
    assert trajectory_counts[("source_a", "a1")] == 24
    assert trajectory_counts[("source_b", "b0")] == 48
    for identity, draws in trajectory_counts.items():
        recipe_record = next(
            item
            for item in recipe["trajectories"]
            if (item["source_id"], item["trajectory_id"]) == identity
        )
        assert len(observations[identity]) == min(
            draws, int(recipe_record["observation_count"])
        )

    for mode in ("geometry", "idm", "latent"):
        replay = []
        for index in range(len(datasets[mode])):
            sample = datasets[mode][index]
            replay.append(
                (
                    sample["source_id"],
                    sample["trajectory_id"],
                    int(sample["current_frame"]),
                    tuple(sample["target_frame"].tolist()),
                    tuple(sample["frame_offset"].tolist()),
                )
            )
        assert replay == reference


def test_recipe_stratifies_past_future_local_and_long_goals(tmp_path: Path) -> None:
    data_root = tmp_path / "NavAnywhere"
    _write_frames(data_root, first=0, last=140)
    recipe = build_sampling_recipe(
        data_root,
        seed=7,
        context_size=1,
        goals_per_obs=4,
        samples_per_epoch=141,
    )
    dataset = NavAnywhereDataset(
        data_root,
        context_size=1,
        goals_per_obs=4,
        action_mode="none",
        seed=7,
        sampling_recipe=recipe,
        transform=_pixel_transform,
    )
    record = dataset._trajectories[0]
    current_position = int(np.flatnonzero(record.frame_indices == 70)[0])
    offsets = dataset._sample_offsets(record, current_position, 19)
    assert -64 <= int(offsets[0]) <= -9
    assert -8 <= int(offsets[1]) <= -1
    assert 0 <= int(offsets[2]) <= 8
    assert 9 <= int(offsets[3]) <= 64


def test_recipe_affine_cycle_continues_without_epoch_boundary_gaps(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "NavAnywhere"
    _write_frames(data_root, source="source_a", trajectory="a0", first=0, last=10)
    _write_frames(data_root, source="source_a", trajectory="a1", first=0, last=10)
    _write_frames(data_root, source="source_a", trajectory="a2", first=0, last=10)
    recipe = build_sampling_recipe(
        data_root,
        seed=19,
        context_size=1,
        goals_per_obs=4,
        # Five slots give trajectories 2/2/1 draws per epoch. The last
        # trajectory is the case that a shared ceil epoch span gets wrong.
        samples_per_epoch=5,
    )
    dataset = NavAnywhereDataset(
        data_root,
        context_size=1,
        goals_per_obs=4,
        action_mode="none",
        seed=19,
        sampling_recipe=recipe,
        transform=_pixel_transform,
    )
    observed: list[int] = []
    for epoch in range(11):
        dataset.set_epoch(epoch)
        sample = dataset[2]
        assert sample["trajectory_id"] == "a2"
        observed.append(int(sample["current_frame"]))
    assert len(set(observed)) == 11


def test_navanywhere_precomputed_posteriors_bypass_jpeg_decode(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "NavAnywhere"
    _write_frames(data_root, first=0, last=8)
    recipe = build_sampling_recipe(
        data_root,
        seed=11,
        context_size=2,
        goals_per_obs=4,
        samples_per_epoch=9,
    )
    record = recipe["trajectories"][0]
    latent_root = tmp_path / "latents"
    latent_path = latent_root / record["source_id"] / f"{record['trajectory_id']}.pt"
    latent_path.parent.mkdir(parents=True)
    file_metadata = {
        "vae_fingerprint": "vae-test",
        "transform_fingerprint": "transform-test",
        "encoding_fingerprint": "encoding-test",
        "sampling_recipe_sha256": "recipe-test",
        "storage_dtype": "bfloat16",
    }
    frame_count = int(record["frame_count"])
    mean = torch.arange(frame_count, dtype=torch.bfloat16).reshape(
        frame_count, 1, 1, 1
    ).expand(frame_count, 4, 28, 28).contiguous()
    torch.save(
        {
            "schema_version": 1,
            "format": "sd_vae_posterior_stats",
            "dataset_name": record["source_id"],
            "trajectory_name": record["trajectory_id"],
            "frame_indices": torch.arange(frame_count, dtype=torch.int64),
            "posterior_mean": mean,
            "posterior_logvar": torch.zeros_like(mean),
            "metadata": {**file_metadata, "source_fingerprint": "source-test"},
        },
        latent_path,
    )
    manifest_records = {
        record["source_id"]: {
            record["trajectory_id"]: {
                "frame_count": frame_count,
                "frame_indices_sha256": frame_indices_sha256(range(frame_count)),
                "posterior_shape": list(mean.shape),
            }
        }
    }
    dataset = NavAnywhereDataset(
        data_root,
        context_size=2,
        goals_per_obs=4,
        action_mode="none",
        seed=11,
        sampling_recipe=recipe,
        precomputed_latent_root=latent_root,
        precomputed_latent_metadata=file_metadata,
        precomputed_latent_records=manifest_records,
        transform=lambda _: (_ for _ in ()).throw(
            AssertionError("JPEG transform must not run in precomputed mode")
        ),
    )
    sample = dataset[0]
    assert "video" not in sample
    assert sample["posterior_mean"].shape == (6, 4, 28, 28)
    assert sample["posterior_logvar"].shape == (6, 4, 28, 28)
    assert torch.isfinite(sample["posterior_mean"]).all()
