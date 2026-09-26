"""CPU smoke tests for heterogeneous CDiT motion conditioning."""

from functools import lru_cache

import pytest
import torch

from models import CDiT
from motion_condition import (
    MotionConditionEncoder,
    OfflineMotionStore,
    flatten_motion_groups,
    make_motion_group,
    motion_condition_collate,
    motion_offset_mask,
    resolve_dataset_motion_map,
)

MOTION_CONFIG = {
    "enabled": True,
    "available_types": ["real", "geometry", "latent", "none"],
    "train_types": ["real", "geometry", "latent", "none"],
    "adapter_hidden_dim": 8,
    "balance_parameter_count": True,
    "parameter_count_reference_dim": 3,
    "real": {
        "real_dim": 3,
        "normalization": {
            "mode": "mean_std",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
    },
    "geometry": {
        "geometry_dim": 3,
        "normalization": {
            "mode": "mean_std",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
    },
    "latent": {
        "latent_dim": 32,
        "normalization": {"mode": "layer_norm"},
    },
}


@lru_cache(maxsize=1)
def _tiny_model() -> CDiT:
    torch.manual_seed(0)
    model = CDiT(
        input_size=4,
        context_size=2,
        patch_size=2,
        in_channels=4,
        hidden_size=16,
        depth=1,
        num_heads=4,
        mlp_ratio=1.0,
        learn_sigma=False,
        motion_condition=MOTION_CONFIG,
    )
    return model.cpu().eval()


def _assert_forward(motion, batch_size: int) -> None:
    model = _tiny_model()
    x = torch.randn(batch_size, 4, 4, 4)
    x_cond = torch.randn(batch_size, 2, 4, 4, 4)
    timestep = torch.arange(batch_size, dtype=torch.float32)
    temporal_k = torch.arange(1, batch_size + 1, dtype=torch.float32)

    with torch.no_grad():
        output = model(
            x,
            timestep,
            x_cond=x_cond,
            rel_t=temporal_k,
            motion=motion,
        )

    assert output.shape == (batch_size, 4, 4, 4)
    assert torch.isfinite(output).all()


def test_disabled_framework_preserves_legacy_checkpoint_structure() -> None:
    kwargs = {
        "input_size": 4,
        "context_size": 2,
        "patch_size": 2,
        "in_channels": 4,
        "hidden_size": 16,
        "depth": 1,
        "num_heads": 4,
        "mlp_ratio": 1.0,
        "learn_sigma": False,
    }
    torch.manual_seed(7)
    legacy = CDiT(**kwargs)
    torch.manual_seed(7)
    explicitly_disabled = CDiT(**kwargs, motion_condition={"enabled": False})

    assert legacy.state_dict().keys() == explicitly_disabled.state_dict().keys()
    assert all(
        torch.equal(legacy.state_dict()[key], explicitly_disabled.state_dict()[key])
        for key in legacy.state_dict()
    )


def test_dataset_motion_map_assigns_common_real_and_proxy_datasets() -> None:
    config = {
        **MOTION_CONFIG,
        "train_types": ["real", "latent"],
        "dataset_motion_types": {
            "recon": "real",
            "sacson": "real",
            "scand": "latent",
        },
    }
    resolved = resolve_dataset_motion_map(config, ["recon", "sacson", "scand"])
    assert resolved == {
        "recon": "real",
        "sacson": "real",
        "scand": "latent",
    }

    with pytest.raises(ValueError, match="cover every training dataset"):
        resolve_dataset_motion_map(config, ["recon", "sacson", "scand", "extra"])


def test_real_action_forward() -> None:
    motion = make_motion_group("real", torch.randn(2, 3))
    _assert_forward(motion, batch_size=2)


def test_latent_action_forward_with_configured_32_dimensions() -> None:
    motion = make_motion_group("latent", torch.randn(2, 32))
    _assert_forward(motion, batch_size=2)


def test_geometry_action_forward() -> None:
    motion = make_motion_group("geometry", torch.randn(2, 3))
    _assert_forward(motion, batch_size=2)


def test_independent_adapters_have_nearly_equal_parameter_counts() -> None:
    production_config = {**MOTION_CONFIG, "adapter_hidden_dim": 768}
    encoder = MotionConditionEncoder(condition_dim=768, config=production_config)
    counts = encoder.adapter_parameter_counts()
    assert counts["real"] == counts["geometry"]
    assert (max(counts.values()) - min(counts.values())) / min(counts.values()) < 0.01
    parameter_ids = [
        {
            id(parameter)
            for parameter in getattr(encoder, f"{name}_action_adapter").parameters()
        }
        for name in ("real", "geometry", "latent")
    ]
    assert parameter_ids[0].isdisjoint(parameter_ids[1])
    assert parameter_ids[0].isdisjoint(parameter_ids[2])
    assert parameter_ids[1].isdisjoint(parameter_ids[2])


def test_none_condition_forward_leaves_base_condition_unchanged() -> None:
    model = _tiny_model()
    base_condition = torch.randn(2, 16)

    # ``none`` has no group and therefore returns before any adapter is invoked.
    encoded = model.motion_condition_encoder(base_condition, motion=None)
    assert encoded is base_condition
    assert torch.equal(encoded, base_condition)

    _assert_forward(motion=None, batch_size=2)


def test_mixed_real_latent_and_none_forward() -> None:
    # Batch row 2 is the none sample and is deliberately absent from the mapping.
    motion = {
        "real": {
            "indices": torch.tensor([0], dtype=torch.int64),
            "values": torch.randn(1, 3),
        },
        "latent": {
            "indices": torch.tensor([1], dtype=torch.int64),
            "values": torch.randn(1, 32),
        },
    }
    base_condition = torch.randn(3, 16)
    encoded = _tiny_model().motion_condition_encoder(base_condition, motion)
    assert torch.equal(encoded[2], base_condition[2])
    _assert_forward(motion, batch_size=3)


def test_collate_and_flatten_preserve_mixed_sample_goal_indices() -> None:
    latent_values = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    real_values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    samples = [
        {
            "image": torch.tensor([10.0]),
            "k": torch.tensor([1, 2]),
            "motion_type": "latent",
            "motion": latent_values,
        },
        {
            "image": torch.tensor([20.0]),
            "k": torch.tensor([1, 2]),
            "motion_type": "none",
        },
        {
            "image": torch.tensor([30.0]),
            "k": torch.tensor([1, 2]),
            "motion_type": "real",
            "motion": real_values,
        },
    ]

    batch = motion_condition_collate(samples)
    assert batch["motion_type"] == ["latent", "none", "real"]
    assert "motion" not in samples[1]
    assert "none" not in batch["motion"]

    flattened = flatten_motion_groups(
        batch["motion"], batch_size=3, num_goals=2, device=torch.device("cpu")
    )
    assert flattened is not None
    assert "none" not in flattened
    assert torch.equal(flattened["latent"]["indices"], torch.tensor([0, 1]))
    assert torch.equal(flattened["real"]["indices"], torch.tensor([4, 5]))
    assert torch.equal(flattened["latent"]["values"], latent_values)
    assert torch.equal(flattened["real"]["values"], real_values)


def test_collate_and_flatten_omit_individual_unlabelled_goals() -> None:
    latent_values = torch.arange(96, dtype=torch.float32).reshape(3, 32)
    samples = [
        {
            "image": torch.tensor([10.0]),
            "k": torch.tensor([-9, 0, 8]),
            "motion_type": "latent",
            "motion": latent_values,
            "motion_mask": torch.tensor([False, True, True]),
        },
        {
            "image": torch.tensor([20.0]),
            "k": torch.tensor([-8, 9, 1]),
            "motion_type": "latent",
            "motion": latent_values + 100,
            "motion_mask": torch.tensor([True, False, True]),
        },
    ]

    batch = motion_condition_collate(samples)
    flattened = flatten_motion_groups(
        batch["motion"], batch_size=2, num_goals=3, device=torch.device("cpu")
    )
    assert flattened is not None
    assert torch.equal(flattened["latent"]["indices"], torch.tensor([1, 2, 3, 5]))
    assert torch.equal(
        flattened["latent"]["values"],
        torch.stack(
            (
                latent_values[1],
                latent_values[2],
                latent_values[0] + 100,
                latent_values[2] + 100,
            )
        ),
    )


def test_motion_offset_mask_uses_inclusive_signed_range() -> None:
    assert torch.equal(
        motion_offset_mask([-9, -8, 0, 8, 9], 8),
        torch.tensor([False, True, True, True, False]),
    )


def test_offline_motion_store_uses_current_to_goal_pairs(tmp_path) -> None:
    cache_dir = tmp_path / "recon"
    cache_dir.mkdir()
    torch.save(
        {
            "schema_version": 1,
            "motion_type": "latent",
            "dataset_name": "recon",
            "trajectory_name": "traj_1",
            "pair_direction": "current_to_goal",
            "normalization": "raw",
            "frame_pairs": torch.tensor([[4, 9], [4, 2]]),
            "motion": torch.stack((torch.arange(32), torch.arange(32) + 100)).float(),
        },
        cache_dir / "traj_1.pt",
    )
    store = OfflineMotionStore(
        root=str(tmp_path),
        dataset_name="recon",
        motion_type="latent",
        input_dim=32,
    )

    result = store.get("traj_1", current_frame=4, target_frames=[2, 9])
    assert torch.equal(result[0], torch.arange(32).float() + 100)
    assert torch.equal(result[1], torch.arange(32).float())


def test_geometry_store_validates_coordinate_and_unit_metadata(tmp_path) -> None:
    cache_dir = tmp_path / "recon"
    cache_dir.mkdir()
    torch.save(
        {
            "schema_version": 1,
            "motion_type": "geometry",
            "dataset_name": "recon",
            "trajectory_name": "traj_2",
            "pair_direction": "current_to_goal",
            "normalization": "raw",
            "coordinate_frame": "current_navigation_frame",
            "translation_unit": "waypoint_spacing_units",
            "yaw_unit": "radians",
            "components": ["delta_x", "delta_y", "delta_yaw"],
            "frame_pairs": torch.tensor([[3, 7]]),
            "motion": torch.tensor([[1.0, 2.0, 0.5]]),
        },
        cache_dir / "traj_2.pt",
    )
    store = OfflineMotionStore(
        root=str(tmp_path),
        dataset_name="recon",
        motion_type="geometry",
        input_dim=3,
        translation_unit="waypoint_spacing_units",
    )

    result = store.get("traj_2", current_frame=3, target_frames=[7])
    assert torch.equal(result, torch.tensor([[1.0, 2.0, 0.5]]))
