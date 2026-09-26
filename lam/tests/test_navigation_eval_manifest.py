from __future__ import annotations

import itertools
import json
import math
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lam.navigation_dataset import NavigationPairDataset
from lam.navigation_evaluation import manifest as manifest_module
from lam.navigation_evaluation.catalog import (
    DEFAULT_OFFSETS,
    EvaluationCatalog,
    canonical_dataset_id,
    load_navigation_frame,
    wrap_yaw,
)
from lam.navigation_evaluation.manifest import (
    InsufficientSamplesError,
    _allocate_nonvisual_donors,
    _CandidateEdge,
    _joint_select_visual_cross_matching,
    _match_cross_offsets,
    _minimum_cost_unique_matching,
    build_evaluation_manifest,
    expected_condition_ids,
    load_manifest,
    manifest_records_sha256,
    sample_balanced_windows,
    write_manifest,
)


def _write_trajectory(
    root: Path,
    name: str,
    *,
    length: int = 12,
    speed: float = 1.0,
    yaw: np.ndarray | None = None,
    image_size: tuple[int, int] = (16, 12),
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    positions = np.stack(
        [np.arange(length, dtype=np.float64) * speed, np.zeros(length)], axis=1
    )
    yaw_array = np.zeros(length, dtype=np.float64) if yaw is None else np.asarray(yaw)
    with (directory / "traj_data.pkl").open("wb") as handle:
        pickle.dump({"position": positions, "yaw": yaw_array}, handle)
    for frame in range(length):
        Image.new("RGB", image_size, (10 + frame, 30 + frame, 50 + frame)).save(
            directory / f"{frame}.jpg"
        )
    return directory


def _catalog_spec(
    tmp_path: Path,
    dataset_id: str,
    split: str,
    names: list[str],
    *,
    spacing: float = 1.0,
) -> tuple[dict, Path]:
    root = tmp_path / "data" / dataset_id / split
    root.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "splits" / dataset_id / split / "traj_names.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(names) + "\n", encoding="utf-8")
    return (
        {
            "dataset_id": dataset_id,
            "split": split,
            "data_root": str(root),
            "split_manifest": str(manifest),
            "waypoint_spacing": spacing,
            "display_name": dataset_id,
        },
        root,
    )


def _formal_dataset(
    tmp_path: Path,
    dataset_id: str,
    names: list[str],
    *,
    speed: float = 1.0,
) -> dict:
    root = tmp_path / "formal" / dataset_id
    root.mkdir(parents=True)
    manifest = tmp_path / "formal_splits" / dataset_id / "train" / "traj_names.txt"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("\n".join(names) + "\n", encoding="utf-8")
    for name in names:
        _write_trajectory(root, name, speed=speed)
    return {
        "display_name": dataset_id,
        "data_root": str(root),
        "waypoint_spacing": 1.0,
        "splits": {"train": str(manifest)},
    }


def _config(
    tmp_path: Path,
    *,
    protocols: list[str],
    id_datasets: tuple[str, ...] = ("recon",),
    ood_datasets: tuple[str, ...] = ("tartan_drive",),
    metric_count: int = 3,
    visual_count: int = 1,
    dataset_names: tuple[str, ...] = ("a", "b", "c"),
    speed: float = 1.0,
) -> dict:
    datasets = {}
    for dataset_id in {*id_datasets, *ood_datasets}:
        datasets[dataset_id] = _formal_dataset(
            tmp_path, dataset_id, list(dataset_names), speed=speed
        )
    return {
        "version": 1,
        "seed": 32,
        "domains": {"train": list(id_datasets), "ood": list(ood_datasets)},
        "datasets": datasets,
        "protocols": protocols,
        "splits": ["train"],
        "sampling": {
            "offsets": [-2, 0, 2],
            "metric_count": metric_count,
            "visual_count": visual_count,
            "image_height": 12,
            "image_width": 16,
            "action_min": [-2.5, -4.0],
            "action_max": [5.0, 4.0],
        },
        "matching": {
            "translation_tolerance_waypoint": 1.0,
            "yaw_tolerance_degrees": 15.0,
            "candidate_pool_multiplier": 4,
            "nearest_candidates": max(4, metric_count),
            "recipient_pair_unique": True,
        },
    }


def test_catalog_aliases_explicit_splits_and_training_action(tmp_path: Path) -> None:
    assert canonical_dataset_id("SACSoN") == "huron"
    assert canonical_dataset_id("tartan") == "tartan_drive"
    assert canonical_dataset_id("Campus-Night") == "campus_night"
    train_spec, train_root = _catalog_spec(tmp_path, "recon", "train", ["train_a"])
    test_spec, test_root = _catalog_spec(tmp_path, "recon", "test", ["test_a"])
    yaw = np.asarray([np.pi / 2, np.pi / 2 + 2 * np.pi + 0.25])
    train_directory = _write_trajectory(train_root, "train_a", length=2, yaw=yaw)
    # Replace positions so global +Y becomes local +X at heading pi/2.
    with (train_directory / "traj_data.pkl").open("wb") as handle:
        pickle.dump(
            {"position": np.asarray([[0.0, 0.0], [0.0, 2.0]]), "yaw": yaw}, handle
        )
    _write_trajectory(test_root, "test_a", length=2)
    train_spec["waypoint_spacing"] = 2.0
    catalog = EvaluationCatalog(
        [train_spec, test_spec], image_height=12, image_width=16
    )

    assert catalog.keys == (("recon", "train"), ("recon", "test"))
    trajectory = catalog.get("recon", "train").trajectories[0]
    action = catalog.action(trajectory, 0, 1)
    np.testing.assert_allclose(action.meters, [2.0, 0.0, 0.25], atol=1e-7)
    np.testing.assert_allclose(action.waypoint, [1.0, 0.0, 0.25], atol=1e-7)
    np.testing.assert_allclose(action.normalized, [-1.0 / 15.0, 0.0, 0.25], atol=1e-7)
    assert wrap_yaw(np.pi) == pytest.approx(-np.pi)


def test_catalog_audits_missing_but_fails_existing_corruption(tmp_path: Path) -> None:
    spec, root = _catalog_spec(tmp_path, "huron", "train", ["missing", "good"])
    _write_trajectory(root, "good")
    catalog = EvaluationCatalog([spec])
    partition = catalog.get("sacson", "train")
    assert partition.missing_trajectories == ("missing",)
    assert partition.manifest_trajectory_count == 2
    assert [trajectory.name for trajectory in partition.trajectories] == ["good"]

    corrupt_spec, corrupt_root = _catalog_spec(tmp_path, "scand", "train", ["corrupt"])
    corrupt = _write_trajectory(corrupt_root, "corrupt")
    (corrupt / "3.jpg").unlink()
    with pytest.raises(RuntimeError, match="missing frame 3"):
        EvaluationCatalog([corrupt_spec])


def test_catalog_rejects_physical_train_test_trajectory_leakage(tmp_path: Path) -> None:
    train_spec, root = _catalog_spec(tmp_path, "recon", "train", ["shared"])
    _write_trajectory(root, "shared")
    test_manifest = tmp_path / "splits" / "recon" / "test" / "traj_names.txt"
    test_manifest.parent.mkdir(parents=True)
    test_manifest.write_text("shared\n", encoding="utf-8")
    test_spec = {
        **train_spec,
        "split": "test",
        "split_manifest": str(test_manifest),
    }
    with pytest.raises(ValueError, match="split leakage"):
        EvaluationCatalog([train_spec, test_spec])


def test_standalone_frame_loader_exactly_matches_training_transform(
    tmp_path: Path,
) -> None:
    pixels = np.zeros((6, 18, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(18, dtype=np.uint8)
    pixels[:, :, 1] = 100
    path = tmp_path / "wide.jpg"
    Image.fromarray(pixels).save(path, quality=100, subsampling=0)
    actual = load_navigation_frame(path, image_height=9, image_width=12)
    expected = NavigationPairDataset._load_frame(
        SimpleNamespace(image_height=9, image_width=12), path
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (9, 12, 3)
    assert actual.dtype == torch.float32
    assert 0 <= actual.min() <= actual.max() <= 1


def test_balanced_sampler_is_seeded_unique_and_visual_prefix_is_distinct(
    tmp_path: Path,
) -> None:
    spec, root = _catalog_spec(tmp_path, "recon", "train", ["a", "b", "c"])
    for name in ("a", "b", "c"):
        _write_trajectory(root, name)
    partition = EvaluationCatalog([spec]).get("recon", "train")
    first = sample_balanced_windows(
        partition, 7, [-2, 0, 2], seed=123, require_distinct_first=3
    )
    second = sample_balanced_windows(
        partition, 7, [-2, 0, 2], seed=123, require_distinct_first=3
    )
    assert [sample.key for sample in first] == [sample.key for sample in second]
    assert len({sample.key for sample in first}) == 7
    assert len({sample.trajectory.name for sample in first[:3]}) == 3
    assert all(2 <= sample.anchor <= 9 for sample in first)


def test_default_grid_requires_anchor_8_through_length_minus_9(
    tmp_path: Path,
) -> None:
    spec, root = _catalog_spec(tmp_path, "recon", "train", ["exact"])
    _write_trajectory(root, "exact", length=17)
    partition = EvaluationCatalog([spec]).get("recon", "train")
    samples = sample_balanced_windows(partition, 1, DEFAULT_OFFSETS, seed=1)
    assert samples[0].anchor == 8
    with pytest.raises(InsufficientSamplesError, match="only 1 anchors"):
        sample_balanced_windows(partition, 2, DEFAULT_OFFSETS, seed=1)


def test_formal_config_builds_id_ood_and_visual_subset_of_metric(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        protocols=["id", "ood"],
        metric_count=4,
        visual_count=2,
    )
    first, metadata = build_evaluation_manifest(config)
    second, second_metadata = build_evaluation_manifest(config)

    assert first == second
    assert metadata["records_sha256"] == second_metadata["records_sha256"]
    assert metadata["condition_count"] == 2
    assert metadata["expected_condition_ids"] == list(expected_condition_ids(config))
    assert len(first) == 2 * 4 * 3
    assert {record["protocol"] for record in first} == {"id", "ood"}
    for condition_id in {record["condition_id"] for record in first}:
        rows = [record for record in first if record["condition_id"] == condition_id]
        visual = [record for record in rows if record["is_visual"]]
        assert len(visual) == 2 * 3
        assert {record["visual_index"] for record in visual} == {0, 1}
        assert len({record["source_trajectory"] for record in visual}) == 2
        assert all(
            record["recipient_anchor_path"] == record["source_anchor_path"]
            for record in rows
        )
        assert all(
            record["recipient_action"] == record["source_action"] for record in rows
        )


def test_expected_condition_ids_has_standard_19_condition_test_only_layout() -> None:
    config = {
        "domains": {
            "train": ["recon", "huron", "scand"],
            "ood": ["tartan_drive"],
        },
        "protocols": ["id", "id_cd", "ood", "ood_cd"],
        "splits": ["test"],
    }
    condition_ids = expected_condition_ids(config)
    assert len(condition_ids) == len(set(condition_ids)) == 19
    assert sum(value.startswith("id__") for value in condition_ids) == 3
    assert sum(value.startswith("id_cd__") for value in condition_ids) == 9
    assert sum(value.startswith("ood__") for value in condition_ids) == 1
    assert sum(value.startswith("ood_cd__") for value in condition_ids) == 6
    assert condition_ids[0] == "id__recon__test"
    assert condition_ids[-1] == "ood_cd__tartan_drive_test__to__scand_test"


def test_cross_manifest_rules_and_complete_visual_window(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        protocols=["id_cd"],
        ood_datasets=("tartan_drive",),
        metric_count=2,
        visual_count=1,
    )
    records, metadata = build_evaluation_manifest(config)

    assert metadata["condition_count"] == 1
    assert len(records) == 2 * 3
    visual = [record for record in records if record["is_visual"]]
    nonvisual = [record for record in records if not record["is_visual"]]
    assert len(visual) == 3
    assert (
        len(
            {
                (record["source_trajectory"], record["source_anchor"])
                for record in visual
            }
        )
        == 1
    )
    assert {record["source_offset"] for record in visual} == {-2, 0, 2}
    assert all(
        (record["recipient_offset"] == 0) == (record["source_offset"] == 0)
        for record in records
    )
    assert all(
        record["source_trajectory"] != record["recipient_trajectory"]
        for record in records
    )
    for source_offset in (-2, 0, 2):
        offset_records = [
            record for record in records if record["source_offset"] == source_offset
        ]
        assert len(
            {
                (
                    record["recipient_dataset"],
                    record["recipient_split"],
                    record["recipient_trajectory"],
                    record["recipient_anchor"],
                    record["recipient_target"],
                )
                for record in offset_records
            }
        ) == len(offset_records)
    assert len(
        {
            (
                record["source_dataset"],
                record["source_split"],
                record["source_trajectory"],
                record["source_anchor"],
            )
            for record in nonvisual
        }
    ) == len(nonvisual)
    assert all(record["match_translation_waypoint_l2"] <= 1.0 for record in records)
    assert all(record["match_yaw_degrees"] <= 15.0 for record in records)
    assert metadata["conditions"][0]["unique_recipient_pairs"] == len(records)
    assert all(
        stats["unique_recipient_pairs"] == 2
        for stats in metadata["conditions"][0]["offsets"].values()
    )
    assert all(
        stats["candidate_coverage"] == pytest.approx(1.0)
        and stats["no_match_rejection_rate"] == pytest.approx(0.0)
        and stats["rejection_rate"] == pytest.approx(0.0)
        for stats in metadata["conditions"][0]["offsets"].values()
    )


def test_cross_fails_before_inference_when_same_domain_has_no_other_trajectory(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        protocols=["id_cd"],
        metric_count=1,
        visual_count=1,
        dataset_names=("only",),
    )
    with pytest.raises(InsufficientSamplesError, match="valid action match"):
        build_evaluation_manifest(config)


def test_global_matching_avoids_cost_101_augmenting_path_counterexample() -> None:
    def edge(recipient: int, cost: float) -> _CandidateEdge:
        return _CandidateEdge(recipient, cost, 0.0, cost)

    # The former donor-ordered DFS first took d0->r0, then rerouted d0->r1
    # when d1 also preferred r0, returning 100+1=101. The global optimum is
    # d0->r0 plus d1->r1, with total cost 2.
    assignments = _minimum_cost_unique_matching(
        [
            [edge(0, 1.0), edge(1, 100.0)],
            [edge(0, 1.0), edge(1, 1.0)],
        ],
        required_prefix=0,
        target_count=2,
        condition="counterexample",
    )
    assert math.fsum(value.cost for value in assignments.values()) == pytest.approx(2.0)
    assert assignments[0].recipient_index == 0
    assert assignments[1].recipient_index == 1


def test_global_matching_forces_mandatory_donor_and_selects_best_subset() -> None:
    def edge(recipient: int, cost: float) -> _CandidateEdge:
        return _CandidateEdge(recipient, cost, 0.0, cost)

    edges = [
        [edge(0, 5.0)],  # mandatory despite its higher cost
        [edge(1, 2.0)],
        [edge(2, 1.0)],
    ]
    first = _minimum_cost_unique_matching(
        edges,
        required_prefix=1,
        target_count=2,
        condition="mandatory-subset",
    )
    second = _minimum_cost_unique_matching(
        edges,
        required_prefix=1,
        target_count=2,
        condition="mandatory-subset",
    )
    assert first == second
    assert set(first) == {0, 2}
    assert math.fsum(value.cost for value in first.values()) == pytest.approx(6.0)

    with pytest.raises(InsufficientSamplesError, match="infeasible"):
        _minimum_cost_unique_matching(
            [[edge(0, 1.0)], [edge(0, 1.0)], [edge(1, 1.0)]],
            required_prefix=2,
            target_count=2,
            condition="mandatory-conflict",
        )


def test_joint_donor_allocation_avoids_offset_order_false_shortage() -> None:
    def edge(recipient: int, cost: float) -> _CandidateEdge:
        return _CandidateEdge(recipient, cost, 0.0, cost)

    # Offset A can use d0 or d1, while offset B can only use d0. Sequentially
    # matching A first at minimum local cost would consume d0 and falsely fail
    # B. The joint capacity solution must reserve d0 for B.
    offset_a = [[edge(0, 0.0)], [edge(0, 1.0)]]
    offset_b = [[edge(0, 0.0)], []]
    allocation = _allocate_nonvisual_donors(
        [offset_a, offset_b],
        visual_count=0,
        per_offset_count=1,
        condition="offset-capacity-counterexample",
    )
    assert allocation == [[1], [0]]

    # Reversing offset enumeration changes only result row order, not the
    # logical donor reservation.
    reversed_allocation = _allocate_nonvisual_donors(
        [offset_b, offset_a],
        visual_count=0,
        per_offset_count=1,
        condition="offset-capacity-counterexample-reversed",
    )
    assert reversed_allocation == [[0], [1]]


def test_cross_offset_matching_is_always_exact_with_mandatory_visual() -> None:
    def edge(recipient: int, cost: float) -> _CandidateEdge:
        return _CandidateEdge(recipient, cost, 0.0, cost)

    # d0 is the mandatory visual donor. Although d1 has the cheapest local
    # edge, it collides on r0, so the exact joint flow selects d2 for r1.
    assignments, strategy = _match_cross_offsets(
        [
            [
                [edge(0, 0.0)],
                [edge(0, 0.0)],
                [edge(1, 1.0)],
            ]
        ],
        visual_count=1,
        target_count=2,
        condition="recipient-collision-fallback",
    )
    assert strategy == "exact_joint_min_cost_flow_lp"
    assert set(assignments[0]) == {0, 2}
    assert {edge.recipient_index for edge in assignments[0].values()} == {0, 1}
    assert math.fsum(edge.cost for edge in assignments[0].values()) == pytest.approx(
        1.0
    )


def test_cross_matching_jointly_optimizes_donor_subset_and_recipients() -> None:
    def edge(recipient: int, cost: float) -> _CandidateEdge:
        return _CandidateEdge(recipient, cost, 0.0, cost)

    # A donor-minimum preselection chooses d0+d1 because their independent
    # minima cost 0+1, then pays 100+1 to make recipients unique. The exact
    # donor/recipient flow instead chooses d0+d2 for global cost 0+2=2.
    assignments, strategy = _match_cross_offsets(
        [
            [
                [edge(0, 0.0), edge(1, 100.0)],
                [edge(0, 1.0)],
                [edge(1, 2.0)],
            ]
        ],
        visual_count=0,
        target_count=2,
        condition="joint-donor-recipient-optimum",
    )
    assert strategy == "exact_joint_min_cost_flow_lp"
    assert set(assignments[0]) == {0, 2}
    assert math.fsum(edge.cost for edge in assignments[0].values()) == pytest.approx(
        2.0
    )


def test_joint_visual_selection_replaces_conflicting_prefix() -> None:
    def edge(recipient: int, cost: float) -> _CandidateEdge:
        return _CandidateEdge(recipient, cost, 0.0, cost)

    # The first two complete donors cannot both be visual because they collide
    # on r0 at every offset. A later donor makes the requested complete visual
    # set feasible and must be selected instead of reporting false shortage.
    graph = [
        [
            [edge(0, 0.0)],
            [edge(0, 0.1)],
            [edge(1, 0.2)],
            [edge(2, 5.0)],
        ],
        [
            [edge(0, 0.0)],
            [edge(0, 0.1)],
            [edge(1, 0.2)],
            [edge(2, 5.0)],
        ],
    ]
    assignments, visual, strategy = _joint_select_visual_cross_matching(
        graph,
        visual_count=2,
        target_count=2,
        condition="replace-conflicting-visual-prefix",
    )
    repeated = _joint_select_visual_cross_matching(
        graph,
        visual_count=2,
        target_count=2,
        condition="replace-conflicting-visual-prefix",
    )

    assert strategy == "exact_joint_visual_selection_milp"
    assert repeated == (assignments, visual, strategy)
    assert 2 in visual
    assert len(set(visual) & {0, 1}) == 1
    assert all(set(visual) == set(assignment) for assignment in assignments)
    assert all(
        len({edge.recipient_index for edge in assignment.values()}) == 2
        for assignment in assignments
    )


def test_joint_visual_selection_matches_random_small_bruteforce() -> None:
    rng = np.random.default_rng(73)
    for case in range(6):
        graph: list[list[list[_CandidateEdge]]] = []
        for _ in range(2):
            offset_edges: list[list[_CandidateEdge]] = []
            for _ in range(4):
                offset_edges.append(
                    [
                        _CandidateEdge(
                            recipient,
                            0.0,
                            0.0,
                            float(rng.integers(0, 11)),
                        )
                        for recipient in range(3)
                        if rng.random() < 0.7
                    ]
                )
            graph.append(offset_edges)

        eligible_visual = [
            donor
            for donor in range(4)
            if all(graph[offset][donor] for offset in range(2))
        ]
        brute: list[float] = []
        for visual_tuple in itertools.combinations(eligible_visual, 1):
            visual = set(visual_tuple)
            per_offset_options: list[list[dict[int, _CandidateEdge]]] = []
            for offset_edges in graph:
                options: list[dict[int, _CandidateEdge]] = []
                for donors in itertools.combinations(range(4), 2):
                    if not visual.issubset(donors):
                        continue
                    for selected_edges in itertools.product(
                        *(offset_edges[donor] for donor in donors)
                    ):
                        if len({edge.recipient_index for edge in selected_edges}) != 2:
                            continue
                        options.append(dict(zip(donors, selected_edges, strict=True)))
                per_offset_options.append(options)
            for pair in itertools.product(*per_offset_options):
                nonvisual = [
                    donor
                    for assignment in pair
                    for donor in assignment
                    if donor not in visual
                ]
                if len(nonvisual) != len(set(nonvisual)):
                    continue
                brute.append(
                    math.fsum(
                        edge.cost for assignment in pair for edge in assignment.values()
                    )
                )

        if not brute:
            with pytest.raises(InsufficientSamplesError):
                _joint_select_visual_cross_matching(
                    graph,
                    visual_count=1,
                    target_count=2,
                    condition=f"random-visual-bruteforce-{case}",
                )
            continue
        assignments, visual, _ = _joint_select_visual_cross_matching(
            graph,
            visual_count=1,
            target_count=2,
            condition=f"random-visual-bruteforce-{case}",
        )
        assert all(set(visual).issubset(assignment) for assignment in assignments)
        actual = math.fsum(
            edge.cost for assignment in assignments for edge in assignment.values()
        )
        assert actual == pytest.approx(min(brute))


def test_exact_joint_flow_falls_back_to_integral_primary_on_tie_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scipy.optimize

    real_linprog = scipy.optimize.linprog
    calls = 0

    def fail_secondary(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls % 2 == 0:
            return SimpleNamespace(success=False, message="synthetic equality issue")
        return real_linprog(*args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "linprog", fail_secondary)
    edge = _CandidateEdge(0, 0.0, 0.0, 0.0)
    assignments, strategy = _match_cross_offsets(
        [[[edge]]],
        visual_count=0,
        target_count=1,
        condition="secondary-tie-fallback",
    )
    assert assignments == [{0: edge}]
    assert strategy == "exact_joint_min_cost_flow_lp_primary_tie_fallback"
    repeated, repeated_strategy = _match_cross_offsets(
        [[[edge]]],
        visual_count=0,
        target_count=1,
        condition="secondary-tie-fallback-repeat",
    )
    assert repeated == assignments
    assert repeated_strategy == strategy


def test_exact_joint_flow_handles_nine_offset_capacity_shape() -> None:
    offset_count = 9
    target_count = 8
    donor_count = 1 + offset_count * (target_count - 1)
    graph = [
        [
            [
                _CandidateEdge(
                    recipient,
                    0.0,
                    0.0,
                    ((offset * 19 + donor * 13 + recipient * 7) % 101) / 100.0,
                )
                for recipient in range(target_count)
            ]
            for donor in range(donor_count)
        ]
        for offset in range(offset_count)
    ]
    assignments, _ = _match_cross_offsets(
        graph,
        visual_count=1,
        target_count=target_count,
        condition="nine-offset-capacity-shape",
    )
    assert all(len(assignment) == target_count for assignment in assignments)
    assert all(0 in assignment for assignment in assignments)
    nonvisual = [
        donor for assignment in assignments for donor in assignment if donor != 0
    ]
    assert len(nonvisual) == len(set(nonvisual)) == donor_count - 1


def test_exact_joint_flow_matches_random_small_bruteforce() -> None:
    rng = np.random.default_rng(19)
    for case in range(12):
        graph: list[list[list[_CandidateEdge]]] = []
        for _ in range(2):
            offset_edges: list[list[_CandidateEdge]] = []
            for _ in range(5):
                donor_edges = [
                    _CandidateEdge(recipient, float(rng.integers(0, 9)), 0.0, 0.0)
                    for recipient in range(3)
                    if rng.random() < 0.72
                ]
                # Keep the actual matching objective in ``cost`` while the gap
                # fields remain finite audit values.
                donor_edges = [
                    _CandidateEdge(
                        edge.recipient_index,
                        edge.cost,
                        0.0,
                        edge.translation_gap,
                    )
                    for edge in donor_edges
                ]
                offset_edges.append(donor_edges)
            graph.append(offset_edges)

        per_offset_options: list[list[dict[int, _CandidateEdge]]] = []
        for offset_edges in graph:
            options: list[dict[int, _CandidateEdge]] = []
            for donors in itertools.combinations(range(5), 2):
                for selected_edges in itertools.product(
                    *(offset_edges[donor] for donor in donors)
                ):
                    if len({edge.recipient_index for edge in selected_edges}) != 2:
                        continue
                    options.append(dict(zip(donors, selected_edges, strict=True)))
            per_offset_options.append(options)
        brute_costs = [
            math.fsum(edge.cost for assignment in pair for edge in assignment.values())
            for pair in itertools.product(*per_offset_options)
            if set(pair[0]).isdisjoint(pair[1])
        ]
        if not brute_costs:
            with pytest.raises(InsufficientSamplesError):
                _match_cross_offsets(
                    graph,
                    visual_count=0,
                    target_count=2,
                    condition=f"random-bruteforce-{case}",
                )
            continue
        assignments, _ = _match_cross_offsets(
            graph,
            visual_count=0,
            target_count=2,
            condition=f"random-bruteforce-{case}",
        )
        actual = math.fsum(
            edge.cost for assignment in assignments for edge in assignment.values()
        )
        assert actual == pytest.approx(min(brute_costs))


def test_exact_joint_flow_scopes_recipient_uniqueness_to_each_offset() -> None:
    edge = _CandidateEdge(0, 0.0, 0.0, 0.0)
    assignments, _ = _match_cross_offsets(
        [
            [[edge], []],
            [[], [edge]],
        ],
        visual_count=0,
        target_count=1,
        condition="recipient-offset-scope",
    )
    assert [next(iter(values.values())).recipient_index for values in assignments] == [
        0,
        0,
    ]
    assert set(assignments[0]).isdisjoint(assignments[1])


def test_cross_recipient_uniqueness_resets_per_source_offset(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        protocols=["id_cd"],
        metric_count=2,
        visual_count=1,
        speed=0.0,
    )
    records, metadata = build_evaluation_manifest(config)
    pairs_by_offset: dict[int, set[tuple[object, ...]]] = {}
    for source_offset in (-2, 0, 2):
        selected = [
            record for record in records if record["source_offset"] == source_offset
        ]
        pairs_by_offset[source_offset] = {
            (
                record["recipient_dataset"],
                record["recipient_split"],
                record["recipient_trajectory"],
                record["recipient_anchor"],
                record["recipient_target"],
            )
            for record in selected
        }
        assert len(pairs_by_offset[source_offset]) == len(selected) == 2

    # Cross-offset reuse is permitted but not required; the exact joint tie
    # policy may select either disjoint or repeated physical recipient pairs.
    condition = metadata["conditions"][0]
    assert condition["recipient_pair_cross_offset_reuse_count"] >= 0
    assert condition["unique_recipient_pairs"] == len(records)


def test_cross_builder_adaptively_expands_deterministic_source_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path,
        protocols=["id_cd"],
        metric_count=2,
        visual_count=1,
        speed=0.0,
    )
    config["matching"]["candidate_pool_multiplier"] = 1
    original_match = manifest_module._match_cross_offsets
    donor_counts: list[int] = []

    def require_expansion(edges_by_offset, **kwargs):
        donor_counts.append(len(edges_by_offset[0]))
        if len(edges_by_offset[0]) <= 4:
            raise InsufficientSamplesError("synthetic prefix shortage")
        return original_match(edges_by_offset, **kwargs)

    monkeypatch.setattr(manifest_module, "_match_cross_offsets", require_expansion)
    records, metadata = build_evaluation_manifest(config)
    condition = metadata["conditions"][0]

    assert len(records) == 6
    assert donor_counts[:2] == [4, 8]
    assert condition["source_pool_initial_count"] == 4
    assert condition["source_pool_count"] == 8
    assert condition["source_pool_expansions"] == 1
    assert condition["source_pool_exhausted"] is False


def test_cross_builder_only_reports_shortage_after_full_source_exhaustion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path,
        protocols=["id_cd"],
        metric_count=2,
        visual_count=1,
        speed=0.0,
    )
    config["matching"]["candidate_pool_multiplier"] = 1
    fixed_counts: list[int] = []
    selection_counts: list[int] = []

    def fail_fixed(edges_by_offset, **kwargs):
        fixed_counts.append(len(edges_by_offset[0]))
        raise InsufficientSamplesError("synthetic fixed-visual shortage")

    def fail_selection(edges_by_offset, **kwargs):
        selection_counts.append(len(edges_by_offset[0]))
        raise InsufficientSamplesError("synthetic full-pool shortage")

    monkeypatch.setattr(manifest_module, "_match_cross_offsets", fail_fixed)
    monkeypatch.setattr(
        manifest_module,
        "_joint_select_visual_cross_matching",
        fail_selection,
    )
    with pytest.raises(InsufficientSamplesError, match="exhaustive matching"):
        build_evaluation_manifest(config)

    assert fixed_counts == [4, 8, 16, 24]
    assert selection_counts == [24]


def test_manifest_jsonl_is_stable_and_hash_verified(tmp_path: Path) -> None:
    config = _config(tmp_path, protocols=["id"], metric_count=2, visual_count=1)
    records, metadata = build_evaluation_manifest(config)
    path = tmp_path / "sample_manifest.jsonl"
    written = write_manifest(records, path, metadata)
    first_bytes = path.read_bytes()
    write_manifest(records, path, metadata)
    assert path.read_bytes() == first_bytes
    loaded, loaded_metadata = load_manifest(path)
    assert loaded == records
    assert loaded_metadata["records_sha256"] == manifest_records_sha256(records)
    assert written["record_count"] == len(records)

    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["source_anchor"] += 1
    lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_manifest(path)
