from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lam.navigation_dataset as navigation_dataset_module
from lam.navigation_dataset import (
    NavigationFrameDecodeError,
    NavigationPairDataModule,
    NavigationPairDataset,
    _EpochAwareDistributedSampler,
)


def _write_trajectory(
    root: Path,
    name: str,
    positions: np.ndarray,
    yaw: np.ndarray,
    *,
    base_value: int = 20,
    image_size: tuple[int, int] = (16, 12),
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    with (directory / "traj_data.pkl").open("wb") as handle:
        pickle.dump({"position": positions, "yaw": yaw}, handle)
    for frame_index in range(len(positions)):
        value = min(255, base_value + frame_index * 20)
        Image.new("RGB", image_size, (value, value, value)).save(
            directory / f"{frame_index}.jpg"
        )
    return directory


def _source_spec(
    tmp_path: Path,
    source_id: int,
    names: list[str],
    *,
    spacing: float = 1.0,
    display_name: str | None = None,
) -> dict:
    root = tmp_path / "data" / f"source_{source_id}"
    root.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "splits" / f"source_{source_id}" / "train" / "traj_names.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(names) + "\n", encoding="utf-8")
    return {
        "data_root": str(root),
        "split_manifest": str(manifest),
        "waypoint_spacing": spacing,
        "display_name": display_name or f"source-{source_id}",
    }


def _make_three_sources(
    tmp_path: Path,
    positions: np.ndarray,
    yaw: np.ndarray,
    *,
    spacing: float = 1.0,
    image_size: tuple[int, int] = (16, 12),
) -> list[dict]:
    specs = []
    for source_id in range(3):
        spec = _source_spec(tmp_path, source_id, ["train_traj"], spacing=spacing)
        _write_trajectory(
            Path(spec["data_root"]),
            "train_traj",
            positions,
            yaw,
            base_value=20 + source_id * 60,
            image_size=image_size,
        )
        specs.append(spec)
    return specs


def _write_dino_cache(root: Path, dataset: NavigationPairDataset) -> None:
    sources = []
    status_index = 0
    for source_id, source in enumerate(dataset._sources):
        feature_file = f"source_{source_id}.fp16"
        features = np.memmap(
            root / feature_file,
            dtype="<f2",
            mode="w+",
            shape=(source.anchor_count, 300, 1024),
        )
        trajectories = []
        offset = 0
        for trajectory in source.trajectories:
            for frame_index in range(trajectory.length):
                features[offset + frame_index] = source_id * 10 + frame_index
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
            status_index += 1
            offset += trajectory.length
        features.flush()
        sources.append(
            {
                "dataset_id": source_id,
                "display_name": source.display_name,
                "data_root": str(source.data_root),
                "split_manifest": str(source.split_manifest),
                "anchor_count": source.anchor_count,
                "feature_file": feature_file,
                "trajectories": trajectories,
            }
        )
    metadata = {
        "version": 1,
        "complete": True,
        "model_name": "dinov3_vitl16",
        "weights_sha256": (
            "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"
        ),
        "source_revision": "346f38fee679c56a6888f91c51670fae61d364e0",
        "dtype": "float16",
        "patch_count": 300,
        "feature_dim": 1024,
        "image_height": 240,
        "image_width": 320,
        "color_aug": False,
        "sources": sources,
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_round_robin_sources_and_complete_frame_anchor_cycle(tmp_path: Path) -> None:
    positions = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    specs = _make_three_sources(tmp_path, positions, np.zeros(3))
    dataset = NavigationPairDataset(
        specs,
        samples_per_epoch=18,
        max_frame_offset=0,
        color_aug=False,
        print_audit=False,
    )

    assert dataset.source_anchor_counts == (3, 3, 3)
    assert dataset.audit["total_unique_anchors"] == 9
    assert [dataset[index]["dataset_id"].item() for index in range(9)] == [
        0,
        1,
        2,
        0,
        1,
        2,
        0,
        1,
        2,
    ]
    for source_id in range(3):
        means = [
            dataset[source_id + 3 * anchor]["videos"][0].mean().item()
            for anchor in range(3)
        ]
        assert means[0] < means[1] < means[2]
        # The next logical source sample wraps only after all source anchors.
        assert torch.equal(
            dataset[source_id]["videos"], dataset[source_id + 9]["videos"]
        )


def test_offset_is_stateless_reproducible_uniform_domain_and_in_bounds(
    tmp_path: Path,
) -> None:
    positions = np.stack([np.arange(7), np.zeros(7)], axis=1).astype(np.float64)
    specs = _make_three_sources(tmp_path, positions, np.zeros(7))
    first = NavigationPairDataset(
        specs,
        samples_per_epoch=600,
        max_frame_offset=4,
        data_seed=1234,
        print_audit=False,
    )
    second = NavigationPairDataset(
        specs,
        samples_per_epoch=600,
        max_frame_offset=4,
        data_seed=1234,
        print_audit=False,
    )

    observed_center_offsets = set()
    # source 0, anchor 3; adding 3*7 revisits that anchor under a new logical idx.
    for repetition in range(20):
        index = 3 * (3 + repetition * 7)
        item = first[index]
        offset = item["frame_offset"].item()
        observed_center_offsets.add(offset)
        assert -3 <= offset <= 3
        assert offset == second[index]["frame_offset"].item()
        assert torch.equal(item["videos"], second[index]["videos"])
    assert any(value < 0 for value in observed_center_offsets)
    assert 0 in observed_center_offsets
    assert any(value > 0 for value in observed_center_offsets)

    # Boundary anchors can only point to frames in their own trajectory.
    for anchor in (0, 6):
        item = first[3 * anchor]
        target = anchor + item["frame_offset"].item()
        assert 0 <= target < 7


def test_epoch_changes_pairs_but_is_reproducible_across_variants(
    tmp_path: Path,
) -> None:
    positions = np.stack([np.arange(11), np.zeros(11)], axis=1).astype(np.float64)
    specs = _make_three_sources(tmp_path, positions, np.zeros(11))
    first = NavigationPairDataset(
        specs,
        samples_per_epoch=330,
        max_frame_offset=5,
        data_seed=1234,
        print_audit=False,
    )
    second = NavigationPairDataset(
        specs,
        samples_per_epoch=330,
        max_frame_offset=5,
        data_seed=1234,
        print_audit=False,
    )

    epoch_zero = [first[index]["frame_offset"].item() for index in range(90)]
    first.set_epoch(1)
    second.set_epoch(1)
    first_epoch_one = [first[index]["frame_offset"].item() for index in range(90)]
    second_epoch_one = [second[index]["frame_offset"].item() for index in range(90)]

    assert first.epoch == second.epoch == 1
    assert first_epoch_one == second_epoch_one
    assert first_epoch_one != epoch_zero


def test_eight_way_sampler_has_exact_global_coverage(tmp_path: Path) -> None:
    positions = np.stack([np.arange(5), np.zeros(5)], axis=1).astype(np.float64)
    specs = _make_three_sources(tmp_path, positions, np.zeros(5))
    dataset = NavigationPairDataset(
        specs,
        samples_per_epoch=120,
        max_frame_offset=0,
        color_aug=False,
        print_audit=False,
    )
    shards = []
    for rank in range(8):
        sampler = _EpochAwareDistributedSampler(
            dataset,
            num_replicas=8,
            rank=rank,
            shuffle=True,
            seed=777,
            drop_last=False,
        )
        sampler.set_epoch(3)
        shards.append(list(sampler))

    flattened = [index for shard in shards for index in shard]
    assert all(len(shard) == 15 for shard in shards)
    assert len(flattened) == len(set(flattened)) == 120
    assert sorted(flattened) == list(range(120))
    for source_id in range(3):
        covered_anchors = {
            (index // 3) % dataset.source_anchor_counts[source_id]
            for index in flattened
            if index % 3 == source_id
        }
        assert covered_anchors == set(range(5))
    assert dataset.epoch == 3


def test_action_matches_nwm_local_se2_normalization_and_yaw_wrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    positions = np.asarray([[0.0, 0.0], [0.0, 2.0]])
    yaw = np.asarray([[np.pi / 2], [np.pi / 2 + 2 * np.pi + 0.25]])
    specs = _make_three_sources(tmp_path, positions, yaw, spacing=2.0)
    dataset = NavigationPairDataset(
        specs,
        samples_per_epoch=3,
        max_frame_offset=1,
        color_aug=False,
        print_audit=False,
    )
    monkeypatch.setattr(dataset, "_sample_offset", lambda rng, anchor, length: 1)

    item = dataset[0]
    # Global [0,2], seen at heading pi/2, is local [2,0].  Dividing by spacing
    # gives [1,0], then NWM maps x from [-2.5,5] and y from [-4,4] to [-1,1].
    expected = torch.tensor([[(-1.0 / 15.0), 0.0, 0.25]], dtype=torch.float32)
    torch.testing.assert_close(item["actions"], expected, atol=1e-6, rtol=0)
    assert item["actions"].shape == (1, 3)
    assert item["actions"].dtype == torch.float32


def test_output_contract_crop_resize_and_pair_shared_brightness(tmp_path: Path) -> None:
    positions = np.zeros((1, 2), dtype=np.float64)
    specs = _make_three_sources(tmp_path, positions, np.zeros(1), image_size=(30, 10))
    dataset = NavigationPairDataset(
        specs,
        samples_per_epoch=3,
        max_frame_offset=64,
        data_seed=99,
        color_aug=True,
        print_audit=False,
    )
    item = dataset[0]

    assert item["videos"].shape == (2, 240, 320, 3)
    assert item["videos"].dtype == torch.float32
    assert torch.equal(item["videos"][0], item["videos"][1])
    assert item["frame_offset"].dtype == torch.int64
    assert item["frame_offset"].item() == 0
    assert item["dataset_id"].dtype == torch.int64


def test_zero_byte_frame_is_reported_and_replaced(tmp_path: Path) -> None:
    positions = np.zeros((1, 2), dtype=np.float64)
    specs = _make_three_sources(tmp_path, positions, np.zeros(1))
    corrupt_path = Path(specs[0]["data_root"]) / "train_traj" / "0.jpg"
    corrupt_path.write_bytes(b"")
    dataset = NavigationPairDataset(
        specs,
        samples_per_epoch=3,
        max_frame_offset=0,
        max_frame_decode_retries=2,
        color_aug=False,
        print_audit=False,
    )

    with pytest.warns(RuntimeWarning, match=r"Skipping corrupt navigation frame.*0\.jpg"):
        item = dataset[0]

    # Legacy logical index 0 is source 0; its deterministic replacement is
    # index 1 from source 1.
    assert item["dataset_id"].item() == 1
    assert item["videos"].shape == (2, 240, 320, 3)


def test_ablation_routes_corrupt_frame_to_dedicated_log_and_keeps_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    warning_log = tmp_path / "warnings" / "data.jsonl"
    monkeypatch.setenv("NAV_LAM_SKIP_BAD_DATA", "1")
    monkeypatch.setenv("NAV_LAM_WARNING_LOG", str(warning_log))
    positions = np.zeros((1, 2), dtype=np.float64)
    specs = _make_three_sources(tmp_path, positions, np.zeros(1))
    corrupt_path = Path(specs[0]["data_root"]) / "train_traj" / "0.jpg"
    corrupt_path.write_bytes(b"")
    dataset = NavigationPairDataset(
        specs,
        samples_per_epoch=3,
        max_frame_offset=0,
        max_frame_decode_retries=0,
        color_aug=False,
        print_audit=False,
    )

    item = dataset[0]

    assert item["dataset_id"].item() == 1
    assert "Skipping corrupt navigation frame" not in capsys.readouterr().err
    records = [
        json.loads(line)
        for line in warning_log.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["kind"] for record in records] == ["corrupt_frame"]
    assert records[0]["path"] == str(corrupt_path)


def test_decode_retry_exhaustion_reports_all_bad_frames(tmp_path: Path) -> None:
    positions = np.zeros((1, 2), dtype=np.float64)
    specs = _make_three_sources(tmp_path, positions, np.zeros(1))
    for spec in specs:
        (Path(spec["data_root"]) / "train_traj" / "0.jpg").write_bytes(b"")
    dataset = NavigationPairDataset(
        specs,
        samples_per_epoch=3,
        max_frame_offset=0,
        max_frame_decode_retries=2,
        color_aug=False,
        print_audit=False,
    )

    with pytest.warns(RuntimeWarning) as reported:
        with pytest.raises(
            RuntimeError, match="after trying 3 distinct logical samples"
        ):
            dataset[0]

    assert len(reported) == 3


def test_corrupt_replacement_preserves_requested_sampling_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    positions = np.zeros((1, 2), dtype=np.float64)
    specs = _make_three_sources(tmp_path, positions, np.zeros(1))[:2]
    specs[0].update(format="pose_labeled", has_action=True)
    specs[1].update(format="frame_only", has_action=False)
    dataset = NavigationPairDataset(
        specs,
        samples_per_epoch=4,
        max_frame_offset=0,
        max_frame_decode_retries=1,
        color_aug=False,
        print_audit=False,
    )

    calls: list[tuple[int, int]] = []
    original = dataset._get_item_once

    def fail_first(index: int, group: int) -> dict[str, torch.Tensor]:
        calls.append((index, group))
        if len(calls) == 1:
            raise NavigationFrameDecodeError(tmp_path / "broken.jpg", "synthetic")
        return original(index, group)

    monkeypatch.setattr(dataset, "_get_item_once", fail_first)
    with pytest.warns(RuntimeWarning, match="broken.jpg"):
        item = dataset[(0, 0)]

    assert calls == [(0, 0), (1, 0)]
    assert item["action_mask"].item() is False


def test_precomputed_dino_cache_skips_jpegs_and_returns_fp16_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    positions = np.stack([np.arange(3), np.zeros(3)], axis=1).astype(np.float64)
    specs = _make_three_sources(tmp_path, positions, np.zeros(3))
    uncached = NavigationPairDataset(
        specs, samples_per_epoch=9, max_frame_offset=0, color_aug=False, print_audit=False
    )
    cache_root = tmp_path / "dino_cache"
    cache_root.mkdir()
    _write_dino_cache(cache_root, uncached)
    cached = NavigationPairDataset(
        specs,
        samples_per_epoch=9,
        max_frame_offset=0,
        color_aug=False,
        dino_feature_cache_path=str(cache_root),
        print_audit=False,
    )
    monkeypatch.setattr(
        cached,
        "_load_frame",
        lambda path: (_ for _ in ()).throw(AssertionError(f"decoded {path}")),
    )

    item = cached[3 * 2]
    assert "videos" not in item
    assert item["dino_features"].shape == (2, 300, 1024)
    assert item["dino_features"].dtype == torch.float16
    torch.testing.assert_close(
        item["dino_features"].float().mean(), torch.tensor(2.0)
    )


def test_precomputed_dino_cache_rejects_image_space_augmentation(tmp_path: Path) -> None:
    positions = np.zeros((1, 2), dtype=np.float64)
    specs = _make_three_sources(tmp_path, positions, np.zeros(1))
    with pytest.raises(ValueError, match="color_aug=False"):
        NavigationPairDataset(
            specs,
            samples_per_epoch=3,
            color_aug=True,
            dino_feature_cache_path=str(tmp_path / "cache"),
            print_audit=False,
        )


def test_manifest_is_exclusive_and_missing_directories_are_audited(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    positions = np.zeros((2, 2), dtype=np.float64)
    specs = []
    for source_id in range(3):
        spec = _source_spec(
            tmp_path, source_id, ["missing_public_traj", "listed_train_traj"]
        )
        root = Path(spec["data_root"])
        _write_trajectory(root, "listed_train_traj", positions, np.zeros(2))
        # This trajectory exists on disk but is absent from the train manifest.
        _write_trajectory(root, "unlisted_test_traj", positions, np.zeros(2))
        specs.append(spec)

    dataset = NavigationPairDataset(specs, samples_per_epoch=6, print_audit=True)
    output = capsys.readouterr().out
    assert "missing=1" in output
    assert dataset.source_anchor_counts == (2, 2, 2)
    for audit_source in dataset.audit["sources"]:
        assert audit_source["missing_trajectories"] == ("missing_public_traj",)
        assert audit_source["usable_trajectory_count"] == 1


@pytest.mark.parametrize("corruption", ["metadata", "image_gap", "yaw_length"])
def test_existing_corrupt_trajectory_fails_fast(
    tmp_path: Path, corruption: str
) -> None:
    specs = []
    for source_id in range(3):
        spec = _source_spec(tmp_path, source_id, ["trajectory"])
        directory = _write_trajectory(
            Path(spec["data_root"]),
            "trajectory",
            np.zeros((2, 2), dtype=np.float64),
            np.zeros(2),
        )
        specs.append(spec)
        if source_id == 0:
            if corruption == "metadata":
                (directory / "traj_data.pkl").unlink()
            elif corruption == "image_gap":
                (directory / "1.jpg").unlink()
            else:
                with (directory / "traj_data.pkl").open("wb") as handle:
                    pickle.dump(
                        {
                            "position": np.zeros((2, 2), dtype=np.float64),
                            "yaw": np.zeros(1),
                        },
                        handle,
                    )

    with pytest.raises(RuntimeError):
        NavigationPairDataset(specs, samples_per_epoch=3, print_audit=False)


def test_ablation_skips_missing_and_corrupt_trajectories_into_warning_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    warning_log = tmp_path / "data_warnings.jsonl"
    monkeypatch.setenv("NAV_LAM_SKIP_BAD_DATA", "1")
    monkeypatch.setenv("NAV_LAM_WARNING_LOG", str(warning_log))
    positions = np.zeros((2, 2), dtype=np.float64)
    specs = []
    for source_id in range(3):
        names = ["good"]
        if source_id == 0:
            names = ["missing", "corrupt", "good"]
        spec = _source_spec(tmp_path, source_id, names)
        _write_trajectory(
            Path(spec["data_root"]), "good", positions, np.zeros(2)
        )
        if source_id == 0:
            corrupt = Path(spec["data_root"]) / "corrupt"
            corrupt.mkdir()
            (corrupt / "0.jpg").write_bytes(b"not-an-image")
        specs.append(spec)

    dataset = NavigationPairDataset(
        specs,
        samples_per_epoch=6,
        max_frame_offset=0,
        print_audit=False,
    )

    first_audit = dataset.audit["sources"][0]
    assert first_audit["usable_trajectory_count"] == 1
    assert first_audit["missing_trajectories"] == ("missing",)
    assert first_audit["corrupt_trajectory_count"] == 1
    assert first_audit["corrupt_trajectories"][0]["trajectory"] == "corrupt"
    assert "Skipping corrupt navigation trajectory" not in capsys.readouterr().err
    records = [
        json.loads(line)
        for line in warning_log.read_text(encoding="utf-8").splitlines()
    ]
    assert {record["kind"] for record in records} == {
        "missing_trajectory",
        "corrupt_trajectory",
    }


def test_apparent_test_manifest_is_rejected(tmp_path: Path) -> None:
    positions = np.zeros((1, 2), dtype=np.float64)
    specs = _make_three_sources(tmp_path, positions, np.zeros(1))
    test_manifest = tmp_path / "splits" / "source_0" / "test" / "traj_names.txt"
    test_manifest.parent.mkdir(parents=True)
    test_manifest.write_text("train_traj\n", encoding="utf-8")
    specs[0]["split_manifest"] = str(test_manifest)
    with pytest.raises(ValueError, match="test manifest"):
        NavigationPairDataset(specs, samples_per_epoch=3, print_audit=False)


def test_data_module_recreates_deterministic_loader_and_has_no_validation_data(
    tmp_path: Path,
) -> None:
    positions = np.stack([np.arange(3), np.zeros(3)], axis=1).astype(np.float64)
    specs = _make_three_sources(tmp_path, positions, np.zeros(3))
    kwargs = {
        "source_specs": specs,
        "batch_size": 3,
        "num_workers": 0,
        "train_shuffle": False,
        "samples_per_epoch": 9,
        "data_seed": 777,
        "print_audit": False,
        "pin_memory": False,
    }
    first = NavigationPairDataModule(**kwargs)
    second = NavigationPairDataModule(**kwargs)
    first_batch = next(iter(first.train_dataloader()))
    second_batch = next(iter(second.train_dataloader()))

    assert first.audit is not None
    assert not hasattr(first, "val_dataset")
    for key in ("videos", "actions", "dataset_id", "frame_offset"):
        assert torch.equal(first_batch[key], second_batch[key])


def test_shared_index_cache_reads_without_rescan_and_detects_changed_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    positions = np.stack([np.arange(3), np.zeros(3)], axis=1).astype(np.float64)
    specs = _make_three_sources(tmp_path, positions, np.zeros(3))
    cache_path = tmp_path / "cache" / "navigation_index.pkl"

    first = NavigationPairDataset(
        specs,
        samples_per_epoch=9,
        print_audit=False,
        index_cache_path=str(cache_path),
    )

    def unexpected_full_scan(*args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "a valid index cache must be read without a filesystem scan"
        )

    monkeypatch.setattr(
        navigation_dataset_module, "_source_specs_signature", unexpected_full_scan
    )
    monkeypatch.setattr(
        navigation_dataset_module, "_build_sources", unexpected_full_scan
    )
    second = NavigationPairDataset(
        specs,
        samples_per_epoch=9,
        print_audit=False,
        index_cache_path=str(cache_path),
    )
    assert cache_path.is_file()
    assert not first.audit["loaded_from_index_cache"]
    assert second.audit["loaded_from_index_cache"]
    assert first.source_anchor_counts == second.source_anchor_counts == (3, 3, 3)

    metadata = Path(specs[0]["data_root"]) / "train_traj" / "traj_data.pkl"
    with metadata.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(RuntimeError, match="changed after the shared index"):
        second[0]


def test_cached_trajectory_change_is_checked_lazily_and_rebuild_reaudits(
    tmp_path: Path,
) -> None:
    positions = np.stack([np.arange(3), np.zeros(3)], axis=1).astype(np.float64)
    specs = _make_three_sources(tmp_path, positions, np.zeros(3))
    cache_path = tmp_path / "cache" / "navigation_index.pkl"
    NavigationPairDataset(
        specs,
        samples_per_epoch=9,
        print_audit=False,
        index_cache_path=str(cache_path),
    )

    broken_frame = Path(specs[0]["data_root"]) / "train_traj" / "1.jpg"
    broken_frame.unlink()
    cached = NavigationPairDataset(
        specs,
        samples_per_epoch=9,
        print_audit=False,
        index_cache_path=str(cache_path),
    )
    assert cached.audit["loaded_from_index_cache"]
    with pytest.raises(RuntimeError, match="changed after the shared index"):
        cached[0]

    with pytest.raises(RuntimeError, match="missing frame 1"):
        NavigationPairDataset(
            specs,
            samples_per_epoch=9,
            print_audit=False,
            index_cache_path=str(cache_path),
            rebuild_index_cache=True,
        )


def test_ablation_skips_cached_trajectory_that_changed_after_index_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    positions = np.stack([np.arange(3), np.zeros(3)], axis=1).astype(np.float64)
    specs = _make_three_sources(tmp_path, positions, np.zeros(3))
    cache_path = tmp_path / "cache" / "navigation_index.pkl"
    NavigationPairDataset(
        specs,
        samples_per_epoch=9,
        max_frame_offset=0,
        print_audit=False,
        index_cache_path=str(cache_path),
    )
    (Path(specs[0]["data_root"]) / "train_traj" / "1.jpg").unlink()
    warning_log = tmp_path / "data_warnings.jsonl"
    monkeypatch.setenv("NAV_LAM_SKIP_BAD_DATA", "1")
    monkeypatch.setenv("NAV_LAM_WARNING_LOG", str(warning_log))
    cached = NavigationPairDataset(
        specs,
        samples_per_epoch=9,
        max_frame_offset=0,
        max_frame_decode_retries=0,
        print_audit=False,
        index_cache_path=str(cache_path),
    )

    item = cached[0]

    assert item["dataset_id"].item() == 1
    records = [
        json.loads(line)
        for line in warning_log.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["kind"] for record in records] == [
        "corrupt_cached_trajectory"
    ]


def test_explicit_index_cache_rebuild_reaudits_valid_in_place_changes(
    tmp_path: Path,
) -> None:
    positions = np.stack([np.arange(3), np.zeros(3)], axis=1).astype(np.float64)
    specs = _make_three_sources(tmp_path, positions, np.zeros(3))
    cache_path = tmp_path / "cache" / "navigation_index.pkl"
    NavigationPairDataset(
        specs,
        samples_per_epoch=9,
        print_audit=False,
        index_cache_path=str(cache_path),
    )

    metadata = Path(specs[0]["data_root"]) / "train_traj" / "traj_data.pkl"
    with metadata.open("wb") as handle:
        pickle.dump(
            {
                "position": positions + np.asarray([10.0, 0.0]),
                "yaw": np.zeros(3),
            },
            handle,
        )
    rebuilt = NavigationPairDataset(
        specs,
        samples_per_epoch=9,
        print_audit=False,
        index_cache_path=str(cache_path),
        rebuild_index_cache=True,
    )
    assert not rebuilt.audit["loaded_from_index_cache"]
    assert rebuilt.audit["rebuild_index_cache"]


@pytest.mark.parametrize("rank", [0, 1])
def test_distributed_rebuild_propagates_rank_zero_failure_to_every_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rank: int
) -> None:
    data = NavigationPairDataModule(
        source_specs=[{}, {}, {}],
        index_cache_path=str(tmp_path / "cache.pkl"),
        rebuild_index_cache=True,
        print_audit=False,
    )
    monkeypatch.setattr(navigation_dataset_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(navigation_dataset_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(navigation_dataset_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(navigation_dataset_module.dist, "get_rank", lambda: rank)

    def construct_dataset(**kwargs):
        del kwargs
        if rank == 0:
            raise ValueError("synthetic corrupt trajectory")
        raise AssertionError("rank 1 must wait for rank 0's rebuild status")

    def broadcast_status(status, src):
        assert src == 0
        if rank == 1:
            status[0] = ("ValueError", "synthetic corrupt trajectory")

    monkeypatch.setattr(
        navigation_dataset_module, "NavigationPairDataset", construct_dataset
    )
    monkeypatch.setattr(
        navigation_dataset_module.dist, "broadcast_object_list", broadcast_status
    )
    with pytest.raises(
        RuntimeError,
        match="rank 0 navigation index rebuild failed: ValueError: synthetic",
    ):
        data.setup("fit")
