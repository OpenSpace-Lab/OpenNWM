"""Deterministic sampling and JSONL manifests for navigation LAM evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import (
    DEFAULT_ACTION_MAX,
    DEFAULT_ACTION_MIN,
    DEFAULT_OFFSETS,
    ActionRecord,
    DatasetPartition,
    DatasetSplitSpec,
    EvaluationCatalog,
    TrajectoryRecord,
    canonical_dataset_id,
    canonical_split,
    validate_offsets,
    wrap_yaw,
)
from .integrity import (
    evaluation_manifest_sha256,
    manifest_metadata_sha256,
)

MANIFEST_SCHEMA_VERSION = 1
PROTOCOL_ORDER: tuple[str, ...] = ("id", "id_cd", "ood", "ood_cd")


class InsufficientSamplesError(ValueError):
    """Raised before inference when a requested condition cannot be populated."""


@dataclass(frozen=True)
class WindowSample:
    trajectory: TrajectoryRecord
    anchor: int

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (*self.trajectory.identity, self.anchor)


@dataclass(frozen=True)
class _RecipientCandidate:
    trajectory: TrajectoryRecord
    anchor: int
    offset: int
    waypoint_action: tuple[float, float, float]

    @property
    def target(self) -> int:
        return self.anchor + self.offset

    @property
    def key(self) -> tuple[str, str, str, int, int]:
        return (*self.trajectory.identity, self.anchor, self.target)


@dataclass(frozen=True)
class _CandidateEdge:
    recipient_index: int
    translation_gap: float
    yaw_gap: float
    cost: float


def sample_balanced_windows(
    partition: DatasetPartition,
    count: int,
    offsets: Sequence[int] = DEFAULT_OFFSETS,
    *,
    seed: int = 32,
    require_distinct_first: int = 0,
) -> list[WindowSample]:
    """Sample unique legal anchors by seeded trajectory round-robin.

    Every returned anchor admits every requested offset.  The first pass takes
    at most one anchor from each shuffled trajectory, which makes the first
    ``require_distinct_first`` samples come from different trajectories.
    """

    offsets = validate_offsets(offsets)
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if (
        isinstance(require_distinct_first, bool)
        or not isinstance(require_distinct_first, int)
        or require_distinct_first < 0
        or require_distinct_first > count
    ):
        raise ValueError("require_distinct_first must be an integer in [0, count]")

    lower = max(0, -min(offsets))
    trajectories = [
        trajectory
        for trajectory in partition.trajectories
        if trajectory.length - 1 - max(offsets) >= lower
    ]
    legal_counts = {
        trajectory.identity: trajectory.length - max(offsets) - lower
        for trajectory in trajectories
    }
    total = sum(legal_counts.values())
    condition = f"{partition.spec.dataset_id}/{partition.spec.split}"
    if len(trajectories) < require_distinct_first:
        raise InsufficientSamplesError(
            f"{condition}: requested {require_distinct_first} visual windows from "
            f"different trajectories, but only {len(trajectories)} trajectories admit "
            f"all offsets {list(offsets)}"
        )
    if total < count:
        raise InsufficientSamplesError(
            f"{condition}: requested {count} unique windows, but only {total} anchors "
            f"admit all offsets {list(offsets)}"
        )

    rng = random.Random(_stable_seed(seed, condition, offsets))
    rng.shuffle(trajectories)
    anchor_orders: dict[tuple[str, str, str], list[int]] = {}
    for trajectory in trajectories:
        anchors = list(range(lower, lower + legal_counts[trajectory.identity]))
        rng.shuffle(anchors)
        anchor_orders[trajectory.identity] = anchors
    next_anchor = {trajectory.identity: 0 for trajectory in trajectories}
    result: list[WindowSample] = []
    while len(result) < count:
        made_progress = False
        for trajectory in trajectories:
            if len(result) >= count:
                break
            index = next_anchor[trajectory.identity]
            anchors = anchor_orders[trajectory.identity]
            if index >= len(anchors):
                continue
            anchor = anchors[index]
            next_anchor[trajectory.identity] = index + 1
            result.append(WindowSample(trajectory, anchor))
            made_progress = True
        if (
            not made_progress
        ):  # defensive; the total-count check should make this unreachable
            raise RuntimeError(f"Balanced sampler stalled for {condition}")

    if (
        len({sample.trajectory.identity for sample in result[:require_distinct_first]})
        != require_distinct_first
    ):
        raise RuntimeError(
            "Balanced sampler violated its distinct-trajectory prefix invariant"
        )
    return result


def expected_condition_ids(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Derive the canonical ordered condition set without touching any data.

    This is shared by manifest preparation, CLI audits, and tests so the
    standard test-only 3 ID + 9 ID-CD + 1 OOD + 6 OOD-CD layout cannot silently
    drift
    between configuration and sampling code.
    """

    if not isinstance(config, Mapping):
        raise TypeError("Evaluation manifest config must be a mapping")
    raw = config.get("evaluation", config)
    if not isinstance(raw, Mapping):
        raise TypeError("config.evaluation must be a mapping")
    protocols = _protocols(raw.get("protocols", PROTOCOL_ORDER))
    splits = tuple(
        canonical_split(value) for value in raw.get("splits", ("test",))
    )
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("splits must be a non-empty sequence without duplicates")
    domains = raw.get("domains", {})
    if not isinstance(domains, Mapping):
        raise TypeError("domains must be a mapping")
    id_datasets = _dataset_ids(
        raw.get("id_datasets", domains.get("train", ("recon", "huron", "scand"))),
        "domains.train",
    )
    ood_datasets = _dataset_ids(
        raw.get("ood_datasets", domains.get("ood", ("tartan_drive",))),
        "domains.ood",
    )
    overlap = set(id_datasets) & set(ood_datasets)
    if overlap:
        raise ValueError(f"ID and OOD dataset sets overlap: {sorted(overlap)}")

    condition_ids: list[str] = []
    if "id" in protocols:
        condition_ids.extend(
            f"id__{dataset_id}__{split}"
            for dataset_id in id_datasets
            for split in splits
        )
    if "id_cd" in protocols:
        condition_ids.extend(
            f"id_cd__{source_dataset}_{source_split}__to__"
            f"{recipient_dataset}_{recipient_split}"
            for source_dataset in id_datasets
            for recipient_dataset in id_datasets
            for source_split in splits
            for recipient_split in splits
        )
    if "ood" in protocols:
        condition_ids.extend(
            f"ood__{dataset_id}__{split}"
            for dataset_id in ood_datasets
            for split in splits
        )
    if "ood_cd" in protocols:
        for id_dataset in id_datasets:
            for ood_dataset in ood_datasets:
                for id_split in splits:
                    for ood_split in splits:
                        condition_ids.extend(
                            (
                                (
                                    f"ood_cd__{id_dataset}_{id_split}__to__"
                                    f"{ood_dataset}_{ood_split}"
                                ),
                                (
                                    f"ood_cd__{ood_dataset}_{ood_split}__to__"
                                    f"{id_dataset}_{id_split}"
                                ),
                            )
                        )
    if len(condition_ids) != len(set(condition_ids)):
        raise RuntimeError("Evaluation configuration produced duplicate condition ids")
    return tuple(condition_ids)


def build_evaluation_manifest(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build every requested ID/OOD condition without loading a checkpoint.

    Supported config shape (directly or below an ``evaluation`` key)::

        datasets:
          - {dataset_id, split, data_root, split_manifest, waypoint_spacing}
        protocols: [id, id_cd, ood, ood_cd]
        id_datasets: [recon, scand, huron]
        ood_datasets: [tartan_drive]
        splits: [test]
        offsets: [-8, -6, -4, -2, 0, 2, 4, 6, 8]
        num_metric: 100
        num_visual: 10
        seed: 32

    ``datasets`` may alternatively be a nested ``dataset -> split -> spec``
    mapping.  Returned records contain only ordinary JSON-compatible values.
    """

    if not isinstance(config, Mapping):
        raise TypeError("Evaluation manifest config must be a mapping")
    raw = config.get("evaluation", config)
    if not isinstance(raw, Mapping):
        raise TypeError("config.evaluation must be a mapping")
    sampling = raw.get("sampling", {})
    if not isinstance(sampling, Mapping):
        raise TypeError("evaluation.sampling must be a mapping")

    matching = raw.get("matching", {})
    if not isinstance(matching, Mapping):
        raise TypeError("evaluation.matching must be a mapping")
    if matching.get("recipient_pair_unique", True) is not True:
        raise ValueError(
            "matching.recipient_pair_unique must remain true for standard evaluation"
        )
    specs = _catalog_specs(config, raw)
    offsets = validate_offsets(_setting(raw, sampling, "offsets", DEFAULT_OFFSETS))
    seed = _integer_setting(raw, sampling, "seed", 32, minimum=0)
    num_metric = _integer_alias_setting(
        raw, sampling, ("metric_count", "num_metric"), 100, minimum=1
    )
    num_visual = _integer_alias_setting(
        raw, sampling, ("visual_count", "num_visual"), 10, minimum=0
    )
    if num_visual > num_metric:
        raise ValueError("num_visual must not exceed num_metric")
    cross_candidate_multiplier = _integer_alias_setting(
        raw,
        matching,
        ("candidate_pool_multiplier", "cross_candidate_multiplier"),
        20,
        minimum=1,
    )
    nearest_candidates = _integer_alias_setting(
        raw, matching, ("nearest_candidates",), 64, minimum=1
    )
    translation_tolerance = _float_alias_setting(
        raw,
        matching,
        ("translation_tolerance_waypoint", "translation_tolerance_waypoints"),
        1.0,
        positive=True,
    )
    yaw_tolerance_degrees = _float_alias_setting(
        raw, matching, ("yaw_tolerance_degrees",), 15.0, positive=True
    )
    yaw_tolerance = math.radians(yaw_tolerance_degrees)
    image_height = _integer_setting(raw, sampling, "image_height", 240, minimum=1)
    image_width = _integer_setting(raw, sampling, "image_width", 320, minimum=1)
    action_min = _setting(raw, sampling, "action_min", DEFAULT_ACTION_MIN)
    action_max = _setting(raw, sampling, "action_max", DEFAULT_ACTION_MAX)

    protocols = _protocols(raw.get("protocols", PROTOCOL_ORDER))
    splits = tuple(
        canonical_split(value) for value in raw.get("splits", ("test",))
    )
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("splits must be a non-empty sequence without duplicates")
    domains = raw.get("domains", {})
    if not isinstance(domains, Mapping):
        raise TypeError("domains must be a mapping")
    id_datasets = _dataset_ids(
        raw.get("id_datasets", domains.get("train", ("recon", "huron", "scand"))),
        "domains.train",
    )
    ood_datasets = _dataset_ids(
        raw.get("ood_datasets", domains.get("ood", ("tartan_drive",))),
        "domains.ood",
    )
    overlap = set(id_datasets) & set(ood_datasets)
    if overlap:
        raise ValueError(f"ID and OOD dataset sets overlap: {sorted(overlap)}")

    catalog = EvaluationCatalog(
        specs,
        image_height=image_height,
        image_width=image_width,
        action_min=action_min,
        action_max=action_max,
    )
    builder = _ManifestBuilder(
        catalog=catalog,
        offsets=offsets,
        seed=seed,
        num_metric=num_metric,
        num_visual=num_visual,
        cross_candidate_multiplier=cross_candidate_multiplier,
        nearest_candidates=nearest_candidates,
        translation_tolerance=translation_tolerance,
        yaw_tolerance=yaw_tolerance,
    )
    records: list[dict[str, Any]] = []
    condition_stats: list[dict[str, Any]] = []

    if "id" in protocols:
        for dataset_id in id_datasets:
            for split in splits:
                condition_records, stats = builder.build_direct("id", dataset_id, split)
                records.extend(condition_records)
                condition_stats.append(stats)
    if "id_cd" in protocols:
        for source_dataset in id_datasets:
            for recipient_dataset in id_datasets:
                for source_split in splits:
                    for recipient_split in splits:
                        condition_records, stats = builder.build_cross(
                            "id_cd",
                            source_dataset,
                            source_split,
                            recipient_dataset,
                            recipient_split,
                        )
                        records.extend(condition_records)
                        condition_stats.append(stats)
    if "ood" in protocols:
        for dataset_id in ood_datasets:
            for split in splits:
                condition_records, stats = builder.build_direct(
                    "ood", dataset_id, split
                )
                records.extend(condition_records)
                condition_stats.append(stats)
    if "ood_cd" in protocols:
        for id_dataset in id_datasets:
            for ood_dataset in ood_datasets:
                for id_split in splits:
                    for ood_split in splits:
                        for (
                            source_dataset,
                            source_split,
                            recipient_dataset,
                            recipient_split,
                        ) in (
                            (id_dataset, id_split, ood_dataset, ood_split),
                            (ood_dataset, ood_split, id_dataset, id_split),
                        ):
                            condition_records, stats = builder.build_cross(
                                "ood_cd",
                                source_dataset,
                                source_split,
                                recipient_dataset,
                                recipient_split,
                            )
                            records.extend(condition_records)
                            condition_stats.append(stats)

    expected_ids = expected_condition_ids(config)
    actual_ids = tuple(str(stats["condition_id"]) for stats in condition_stats)
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        unexpected = sorted(set(actual_ids) - set(expected_ids))
        raise RuntimeError(
            "Manifest builder condition set drifted from the canonical configuration: "
            f"missing={missing}, unexpected={unexpected}, order_matches=False"
        )
    _validate_records(records)
    metadata: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "seed": seed,
        "offsets": list(offsets),
        "num_metric": num_metric,
        "num_visual": num_visual,
        "protocols": list(protocols),
        "id_datasets": list(id_datasets),
        "ood_datasets": list(ood_datasets),
        "splits": list(splits),
        "translation_tolerance_waypoints": translation_tolerance,
        "yaw_tolerance_degrees": yaw_tolerance_degrees,
        "cross_candidate_multiplier": cross_candidate_multiplier,
        "nearest_candidates": nearest_candidates,
        "catalog": catalog.audit(),
        "conditions": condition_stats,
        "expected_condition_ids": list(expected_ids),
        "condition_count": len(condition_stats),
        "record_count": len(records),
        "records_sha256": manifest_records_sha256(records),
    }
    return records, metadata


class _ManifestBuilder:
    def __init__(
        self,
        *,
        catalog: EvaluationCatalog,
        offsets: tuple[int, ...],
        seed: int,
        num_metric: int,
        num_visual: int,
        cross_candidate_multiplier: int,
        nearest_candidates: int,
        translation_tolerance: float,
        yaw_tolerance: float,
    ) -> None:
        self.catalog = catalog
        self.offsets = offsets
        self.seed = seed
        self.num_metric = num_metric
        self.num_visual = num_visual
        self.cross_candidate_multiplier = cross_candidate_multiplier
        self.nearest_candidates = nearest_candidates
        self.translation_tolerance = translation_tolerance
        self.yaw_tolerance = yaw_tolerance
        self._recipient_pools: dict[
            tuple[str, str, tuple[int, ...]], _RecipientPool
        ] = {}

    def build_direct(
        self, protocol: str, dataset_id: str, split: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        partition = self.catalog.get(dataset_id, split)
        condition_id = f"{protocol}__{dataset_id}__{split}"
        windows = sample_balanced_windows(
            partition,
            self.num_metric,
            self.offsets,
            seed=_stable_seed(self.seed, condition_id),
            require_distinct_first=self.num_visual,
        )
        records: list[dict[str, Any]] = []
        for window_index, window in enumerate(windows):
            for offset_index, offset in enumerate(self.offsets):
                record = self._source_record(
                    protocol,
                    condition_id,
                    window,
                    offset,
                    is_visual=window_index < self.num_visual,
                    visual_index=(
                        window_index if window_index < self.num_visual else None
                    ),
                    window_index=window_index,
                    offset_index=offset_index,
                )
                record.update(
                    {
                        "recipient_dataset": window.trajectory.dataset_id,
                        "recipient_split": window.trajectory.split,
                        "recipient_trajectory": window.trajectory.name,
                        "recipient_anchor": window.anchor,
                        "recipient_target": window.anchor + offset,
                        "recipient_offset": offset,
                        "recipient_anchor_path": str(
                            window.trajectory.frame_path(window.anchor)
                        ),
                        "recipient_target_path": str(
                            window.trajectory.frame_path(window.anchor + offset)
                        ),
                        "recipient_waypoint_spacing": window.trajectory.waypoint_spacing,
                        "recipient_action": record["source_action"],
                        "recipient_action_normalized": record[
                            "source_action_normalized"
                        ],
                        "recipient_action_waypoint": record["source_action_waypoint"],
                        "recipient_action_meters": record["source_action_meters"],
                    }
                )
                record["record_id"] = _record_id(record)
                records.append(record)
        stats = {
            "condition_id": condition_id,
            "protocol": protocol,
            "source_dataset": dataset_id,
            "source_split": split,
            "record_count": len(records),
            "metric_window_count": len(windows),
            "visual_window_count": self.num_visual,
            "unique_source_trajectories": len(
                {window.trajectory.name for window in windows}
            ),
            "per_offset_count": {
                str(offset): self.num_metric for offset in self.offsets
            },
        }
        return records, stats

    def build_cross(
        self,
        protocol: str,
        source_dataset: str,
        source_split: str,
        recipient_dataset: str,
        recipient_split: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        source = self.catalog.get(source_dataset, source_split)
        recipient = self.catalog.get(recipient_dataset, recipient_split)
        condition_id = (
            f"{protocol}__{source_dataset}_{source_split}__to__"
            f"{recipient_dataset}_{recipient_split}"
        )
        source_total = _legal_window_count(source, self.offsets)
        required_distinct_source_windows = self.num_visual + (
            self.num_metric - self.num_visual
        ) * len(self.offsets)
        if source_total < required_distinct_source_windows:
            raise InsufficientSamplesError(
                f"{condition_id}: requested {required_distinct_source_windows} distinct "
                "donor anchor-windows (complete visual donors are reused across offsets), "
                f"but only {source_total} anchors admit all offsets {list(self.offsets)}"
            )
        initial_source_pool_count = min(
            source_total,
            max(
                self.num_metric * self.cross_candidate_multiplier,
                required_distinct_source_windows,
            ),
        )
        # Build one stable ordering of all legal donors. Metric matching uses
        # only the configured prefix, while complete visual-window discovery
        # may continue beyond it. Thus a small smoke-test metric count cannot
        # create an artificial coverage failure.
        all_source_windows = sample_balanced_windows(
            source,
            source_total,
            self.offsets,
            seed=_stable_seed(self.seed, condition_id, "source_pool"),
        )
        offset_pools = [
            self._pool(
                recipient,
                (0,)
                if offset == 0
                else tuple(value for value in self.offsets if value != 0),
            )
            for offset in self.offsets
        ]
        edge_cache: list[dict[tuple[str, str, str, int], list[_CandidateEdge]]] = [
            {} for _ in self.offsets
        ]

        def edges_for(window: WindowSample, offset_index: int) -> list[_CandidateEdge]:
            cached = edge_cache[offset_index].get(window.key)
            if cached is not None:
                return cached
            offset = self.offsets[offset_index]
            action = self.catalog.action(
                window.trajectory, window.anchor, window.anchor + offset
            )
            # Keeping at least N neighbors is exact for an N-edge matching: if
            # a selected edge ranked below N, the other N-1 pairs can occupy at
            # most N-1 of that donor's first N recipients, leaving one
            # no-more-expensive replacement.
            cached = offset_pools[offset_index].edges(
                action,
                source_trajectory=window.trajectory,
                enforce_different_trajectory=(source_dataset == recipient_dataset),
                translation_tolerance=self.translation_tolerance,
                yaw_tolerance=self.yaw_tolerance,
                used=frozenset(),
                max_neighbors=max(self.nearest_candidates, self.num_metric),
            )
            edge_cache[offset_index][window.key] = cached
            return cached

        # Expand a deterministic donor prefix only when the exact graph is
        # infeasible. The full-pool visual-selection MILP is the final proof
        # before reporting true data insufficiency.
        source_pool_count = initial_source_pool_count
        source_pool_expansions = 0
        last_failure: Exception | None = None
        while True:
            source_windows = all_source_windows[:source_pool_count]
            compatible = [
                window
                for window in source_windows
                if all(
                    edges_for(window, offset_index)
                    for offset_index in range(len(self.offsets))
                )
            ]
            if len(compatible) >= self.num_visual:
                visual_windows = compatible[: self.num_visual]
                visual_keys = {window.key for window in visual_windows}
                ordered_windows = visual_windows + [
                    window for window in source_windows if window.key not in visual_keys
                ]
                edges_by_offset = [
                    [edges_for(window, offset_index) for window in ordered_windows]
                    for offset_index in range(len(self.offsets))
                ]
                try:
                    assignments_by_offset, matching_strategy = _match_cross_offsets(
                        edges_by_offset,
                        visual_count=self.num_visual,
                        target_count=self.num_metric,
                        condition=condition_id,
                    )
                    break
                except InsufficientSamplesError as fixed_visual_failure:
                    last_failure = fixed_visual_failure
                    visual_conflict = False
                    for offset_index, donor_edges in enumerate(edges_by_offset):
                        try:
                            _minimum_cost_unique_matching(
                                donor_edges[: self.num_visual],
                                required_prefix=self.num_visual,
                                target_count=self.num_visual,
                                condition=(
                                    f"{condition_id}/visual-offset-index={offset_index}"
                                ),
                            )
                        except InsufficientSamplesError:
                            visual_conflict = True
                            break
                    if visual_conflict or source_pool_count == source_total:
                        try:
                            (
                                assignments_by_offset,
                                selected_visual_indices,
                                selection_strategy,
                            ) = _joint_select_visual_cross_matching(
                                edges_by_offset,
                                visual_count=self.num_visual,
                                target_count=self.num_metric,
                                condition=condition_id,
                            )
                            visual_windows = [
                                ordered_windows[index]
                                for index in selected_visual_indices
                            ]
                            visual_keys = {window.key for window in visual_windows}
                            matching_strategy = selection_strategy
                            break
                        except InsufficientSamplesError as selection_failure:
                            last_failure = selection_failure
            else:
                last_failure = InsufficientSamplesError(
                    f"{condition_id}: only {len(compatible)} of "
                    f"{source_pool_count} donor windows have a candidate at every "
                    f"offset; requested {self.num_visual} complete visual windows"
                )

            if source_pool_count == source_total:
                raise InsufficientSamplesError(
                    f"{condition_id}: exhaustive matching over all {source_total} legal "
                    "donor windows is infeasible; no valid action matching exists "
                    "and no tolerance was relaxed. "
                    f"last_diagnostic={last_failure}"
                ) from last_failure
            source_pool_count = min(
                source_total, max(source_pool_count + 1, source_pool_count * 2)
            )
            source_pool_expansions += 1

        visual_index = {
            window.key: index for index, window in enumerate(visual_windows)
        }

        records: list[dict[str, Any]] = []
        recipient_pair_occurrences: list[tuple[str, str, str, int, int]] = []
        used_nonvisual_source_windows: set[tuple[str, str, str, int]] = set()
        offset_stats: dict[str, Any] = {}
        for offset_index, offset in enumerate(self.offsets):
            # Each source offset is an independent experiment. Recipient pair
            # uniqueness intentionally resets here; donor uniqueness does not.
            used_recipient_pairs: set[tuple[str, str, str, int, int]] = set()
            pool = offset_pools[offset_index]
            donor_edges = edges_by_offset[offset_index]
            assignments = assignments_by_offset[offset_index]
            matched_indices = sorted(
                assignments,
                key=lambda index: (
                    0 if ordered_windows[index].key in visual_keys else 1,
                    visual_index.get(ordered_windows[index].key, index),
                    index,
                ),
            )
            matched_indices = matched_indices[: self.num_metric]
            if len(matched_indices) != self.num_metric:
                raise InsufficientSamplesError(
                    f"{condition_id}/offset={offset}: matched {len(matched_indices)} unique "
                    f"recipient pairs, requested {self.num_metric}"
                )
            if not visual_keys.issubset(
                {ordered_windows[index].key for index in matched_indices}
            ):
                raise InsufficientSamplesError(
                    f"{condition_id}/offset={offset}: unique matching could not reserve every "
                    "complete visual donor window"
                )

            for window_index, donor_index in enumerate(matched_indices):
                window = ordered_windows[donor_index]
                edge = assignments[donor_index]
                candidate = pool.candidates[edge.recipient_index]
                if candidate.key in used_recipient_pairs:
                    raise RuntimeError("Cross matching reused a recipient pair")
                used_recipient_pairs.add(candidate.key)
                recipient_pair_occurrences.append(candidate.key)
                is_visual = window.key in visual_keys
                if not is_visual:
                    if window.key in used_nonvisual_source_windows:
                        raise RuntimeError(
                            "Cross matching reused a non-visual donor anchor-window"
                        )
                    used_nonvisual_source_windows.add(window.key)
                record = self._source_record(
                    protocol,
                    condition_id,
                    window,
                    offset,
                    is_visual=is_visual,
                    visual_index=visual_index.get(window.key),
                    window_index=window_index,
                    offset_index=offset_index,
                )
                recipient_action = self.catalog.action(
                    candidate.trajectory, candidate.anchor, candidate.target
                )
                record.update(
                    {
                        "recipient_dataset": recipient_dataset,
                        "recipient_split": recipient_split,
                        "recipient_trajectory": candidate.trajectory.name,
                        "recipient_anchor": candidate.anchor,
                        "recipient_target": candidate.target,
                        "recipient_offset": candidate.offset,
                        "recipient_anchor_path": str(
                            candidate.trajectory.frame_path(candidate.anchor)
                        ),
                        "recipient_target_path": str(
                            candidate.trajectory.frame_path(candidate.target)
                        ),
                        "recipient_waypoint_spacing": candidate.trajectory.waypoint_spacing,
                        "recipient_action": recipient_action.as_dict(),
                        "recipient_action_normalized": list(
                            recipient_action.normalized
                        ),
                        "recipient_action_waypoint": list(recipient_action.waypoint),
                        "recipient_action_meters": list(recipient_action.meters),
                        "match_translation_waypoint_l2": edge.translation_gap,
                        "match_yaw_radians": edge.yaw_gap,
                        "match_yaw_degrees": math.degrees(edge.yaw_gap),
                        "match_cost": edge.cost,
                    }
                )
                record["record_id"] = _record_id(record)
                records.append(record)

            nonempty = sum(bool(edges) for edges in donor_edges)
            candidate_coverage = nonempty / len(donor_edges)
            no_match_rejection_rate = 1.0 - candidate_coverage
            offset_stats[str(offset)] = {
                "pair_count": self.num_metric,
                "donor_candidate_count": len(donor_edges),
                "candidate_coverage": candidate_coverage,
                "no_match_rejection_rate": no_match_rejection_rate,
                # Backward-compatible alias with corrected semantics.
                "rejection_rate": no_match_rejection_rate,
                "donor_unselected_rate": (1.0 - self.num_metric / len(donor_edges)),
                "recipient_candidate_count": len(pool.candidates),
                "recipient_offsets": list(pool.offsets),
                "unique_recipient_pairs": len(used_recipient_pairs),
                "matching_strategy": matching_strategy,
            }

        unique_recipient_pairs_across_offsets = len(set(recipient_pair_occurrences))
        stats = {
            "condition_id": condition_id,
            "protocol": protocol,
            "source_dataset": source_dataset,
            "source_split": source_split,
            "recipient_dataset": recipient_dataset,
            "recipient_split": recipient_split,
            "record_count": len(records),
            "visual_window_count": self.num_visual,
            "source_pool_initial_count": initial_source_pool_count,
            "source_pool_count": source_pool_count,
            "source_pool_total": source_total,
            "source_pool_expansions": source_pool_expansions,
            "source_pool_exhausted": source_pool_count == source_total,
            "unique_source_trajectories": len(
                {record["source_trajectory"] for record in records}
            ),
            "unique_source_windows": len(
                {
                    (
                        record["source_dataset"],
                        record["source_split"],
                        record["source_trajectory"],
                        record["source_anchor"],
                    )
                    for record in records
                }
            ),
            "unique_recipient_trajectories": len(
                {record["recipient_trajectory"] for record in records}
            ),
            # Sum of the independently unique per-offset pair sets. This equals
            # record_count by construction; the next two fields quantify valid
            # cross-offset reuse rather than treating it as leakage.
            "unique_recipient_pairs": sum(
                int(offset_summary["unique_recipient_pairs"])
                for offset_summary in offset_stats.values()
            ),
            "unique_recipient_pairs_across_offsets": (
                unique_recipient_pairs_across_offsets
            ),
            "recipient_pair_cross_offset_reuse_count": (
                len(recipient_pair_occurrences) - unique_recipient_pairs_across_offsets
            ),
            "matching_strategy": matching_strategy,
            "offsets": offset_stats,
        }
        return records, stats

    def _pool(
        self, partition: DatasetPartition, offsets: tuple[int, ...]
    ) -> _RecipientPool:
        key = (*partition.key, offsets)
        if key not in self._recipient_pools:
            self._recipient_pools[key] = _RecipientPool(
                partition, offsets, self.offsets
            )
        return self._recipient_pools[key]

    def _source_record(
        self,
        protocol: str,
        condition_id: str,
        window: WindowSample,
        offset: int,
        *,
        is_visual: bool,
        visual_index: int | None,
        window_index: int,
        offset_index: int,
    ) -> dict[str, Any]:
        target = window.anchor + offset
        action = self.catalog.action(window.trajectory, window.anchor, target)
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "condition_id": condition_id,
            "protocol": protocol,
            "source_dataset": window.trajectory.dataset_id,
            "source_split": window.trajectory.split,
            "source_trajectory": window.trajectory.name,
            "source_anchor": window.anchor,
            "source_target": target,
            "source_offset": offset,
            # Short aliases make direct/non-cross batches convenient while the
            # explicit source_* names remain unambiguous for cross evaluation.
            "anchor": window.anchor,
            "target": target,
            "offset": offset,
            "source_anchor_path": str(window.trajectory.frame_path(window.anchor)),
            "source_target_path": str(window.trajectory.frame_path(target)),
            "source_waypoint_spacing": window.trajectory.waypoint_spacing,
            "source_action": action.as_dict(),
            "source_action_normalized": list(action.normalized),
            "source_action_waypoint": list(action.waypoint),
            "source_action_meters": list(action.meters),
            "window_index": window_index,
            "offset_index": offset_index,
            "is_visual": is_visual,
            "visual_index": visual_index,
        }


class _RecipientPool:
    """CPU action index for one recipient split and an allowed offset group."""

    def __init__(
        self,
        partition: DatasetPartition,
        offsets: tuple[int, ...],
        full_offsets: tuple[int, ...],
    ) -> None:
        self.partition = partition
        self.offsets = offsets
        lower = max(0, -min(full_offsets))
        upper_margin = max(full_offsets)
        candidates: list[_RecipientCandidate] = []
        features: list[tuple[float, float, float]] = []
        for trajectory in partition.trajectories:
            upper = trajectory.length - 1 - upper_margin
            if upper < lower:
                continue
            anchors = np.arange(lower, upper + 1, dtype=np.int64)
            start_positions = trajectory.positions[anchors, :2]
            start_yaw = trajectory.yaw[anchors]
            cosine, sine = np.cos(start_yaw), np.sin(start_yaw)
            for offset in offsets:
                target_positions = trajectory.positions[anchors + offset, :2]
                displacement = target_positions - start_positions
                waypoint_x = (
                    displacement[:, 0] * cosine + displacement[:, 1] * sine
                ) / trajectory.waypoint_spacing
                waypoint_y = (
                    -displacement[:, 0] * sine + displacement[:, 1] * cosine
                ) / trajectory.waypoint_spacing
                yaw = wrap_yaw(trajectory.yaw[anchors + offset] - start_yaw)
                for index, anchor in enumerate(anchors):
                    action = (
                        float(waypoint_x[index]),
                        float(waypoint_y[index]),
                        float(yaw[index]),
                    )
                    candidates.append(
                        _RecipientCandidate(
                            trajectory, int(anchor), int(offset), action
                        )
                    )
                    features.append(action)
        if not candidates:
            raise InsufficientSamplesError(
                f"{partition.spec.dataset_id}/{partition.spec.split}: no recipient pairs "
                f"admit full offset grid {list(full_offsets)}"
            )
        self.candidates = tuple(candidates)
        self._features = np.asarray(features, dtype=np.float64)
        self._tree_cache: dict[tuple[float, float], tuple[Any, np.ndarray]] = {}

    def edges(
        self,
        source_action: ActionRecord,
        *,
        source_trajectory: TrajectoryRecord,
        enforce_different_trajectory: bool,
        translation_tolerance: float,
        yaw_tolerance: float,
        used: frozenset[tuple[str, str, str, int, int]],
        max_neighbors: int,
    ) -> list[_CandidateEdge]:
        source = np.asarray(source_action.waypoint, dtype=np.float64)
        if self.offsets == (0,) and np.allclose(source, 0.0, atol=1e-12, rtol=0.0):
            # Every zero-offset action is exactly [0,0,0] in waypoint space.
            # Avoid materializing/sorting tens of thousands of equal KD-tree
            # neighbors for each donor while retaining manifest-order tie-breaks.
            zero_edges: list[_CandidateEdge] = []
            for recipient_index, candidate in enumerate(self.candidates):
                if candidate.key in used:
                    continue
                if (
                    enforce_different_trajectory
                    and candidate.trajectory.identity == source_trajectory.identity
                ):
                    continue
                zero_edges.append(_CandidateEdge(recipient_index, 0.0, 0.0, 0.0))
                if len(zero_edges) >= max_neighbors:
                    break
            return zero_edges
        query_limit = min(len(self.candidates) * 3, max(256, max_neighbors * 4))
        edges: list[_CandidateEdge] = []
        while True:
            candidate_indices, exhausted = self._nearby_indices(
                source,
                translation_tolerance,
                yaw_tolerance,
                query_limit=query_limit,
            )
            edges = []
            for recipient_index in candidate_indices:
                candidate = self.candidates[recipient_index]
                if candidate.key in used:
                    continue
                if (
                    enforce_different_trajectory
                    and candidate.trajectory.identity == source_trajectory.identity
                ):
                    continue
                feature = self._features[recipient_index]
                translation_gap = float(np.linalg.norm(feature[:2] - source[:2]))
                yaw_gap = abs(float(wrap_yaw(feature[2] - source[2])))
                if translation_gap > translation_tolerance or yaw_gap > yaw_tolerance:
                    continue
                cost = math.hypot(
                    translation_gap / translation_tolerance, yaw_gap / yaw_tolerance
                )
                edges.append(
                    _CandidateEdge(recipient_index, translation_gap, yaw_gap, cost)
                )
            if len(edges) >= max_neighbors or exhausted:
                break
            query_limit = min(len(self.candidates) * 3, query_limit * 2)
        edges.sort(
            key=lambda edge: (
                edge.cost,
                edge.translation_gap,
                edge.yaw_gap,
                self.candidates[edge.recipient_index].key,
            )
        )
        return edges[:max_neighbors]

    def _nearby_indices(
        self,
        source: np.ndarray,
        translation_tolerance: float,
        yaw_tolerance: float,
        *,
        query_limit: int,
    ) -> tuple[list[int], bool]:
        key = float(translation_tolerance), float(yaw_tolerance)
        cached = self._tree_cache.get(key)
        if cached is None:
            scaled = self._features.copy()
            scaled[:, :2] /= translation_tolerance
            scaled[:, 2] /= yaw_tolerance
            expanded = np.concatenate(
                [
                    scaled + np.asarray([0.0, 0.0, -2.0 * np.pi / yaw_tolerance]),
                    scaled,
                    scaled + np.asarray([0.0, 0.0, 2.0 * np.pi / yaw_tolerance]),
                ],
                axis=0,
            )
            original = np.tile(np.arange(len(self.candidates), dtype=np.int64), 3)
            try:
                from scipy.spatial import cKDTree

                tree: Any = cKDTree(expanded)
            except (
                ImportError
            ):  # pragma: no cover - the project environment includes scipy
                tree = expanded
            cached = tree, original
            self._tree_cache[key] = cached
        tree, original = cached
        query = source.copy()
        query[:2] /= translation_tolerance
        query[2] /= yaw_tolerance
        radius = math.sqrt(2.0) + 1e-12
        query_count = min(int(query_limit), len(original))
        if isinstance(tree, np.ndarray):
            distances = np.linalg.norm(tree - query, axis=1)
            within = np.flatnonzero(distances <= radius)
            order = within[np.argsort(distances[within], kind="stable")]
            expanded_indices = order[:query_count]
            exhausted = len(within) <= query_count
        else:
            distances, expanded_indices = tree.query(
                query, k=query_count, distance_upper_bound=radius
            )
            distances = np.atleast_1d(distances)
            expanded_indices = np.atleast_1d(expanded_indices)
            valid = np.isfinite(distances) & (expanded_indices < len(original))
            expanded_indices = expanded_indices[valid]
            exhausted = int(valid.sum()) < query_count or query_count == len(original)
        # Preserve nearest-neighbor order here; final hard-threshold edges are
        # sorted by exact cost and stable recipient identity in ``edges``.
        seen: set[int] = set()
        result: list[int] = []
        for expanded_index in expanded_indices:
            candidate_index = int(original[int(expanded_index)])
            if candidate_index not in seen:
                seen.add(candidate_index)
                result.append(candidate_index)
        return result, exhausted


def _minimum_cost_unique_matching(
    edges: Sequence[Sequence[_CandidateEdge]],
    *,
    required_prefix: int,
    target_count: int,
    condition: str,
) -> dict[int, _CandidateEdge]:
    """Solve a fixed-cardinality global minimum-cost bipartite matching.

    Donors ``[0, required_prefix)`` are equality-constrained to participate.
    Every other donor may be omitted, every recipient is used at most once,
    and exactly ``target_count`` edges are selected.  The bipartite matching
    LP is integral; we still validate the returned vertex rather than silently
    rounding.  A second optimization over the primary-optimal face applies a
    stable hash-based tie policy, making equal-cost manifests reproducible.
    """

    donor_count = len(edges)
    if (
        isinstance(required_prefix, bool)
        or not isinstance(required_prefix, int)
        or required_prefix < 0
        or required_prefix > donor_count
    ):
        raise ValueError("required_prefix must be an integer in [0, donor_count]")
    if (
        isinstance(target_count, bool)
        or not isinstance(target_count, int)
        or target_count < required_prefix
        or target_count > donor_count
    ):
        raise ValueError(
            "target_count must be an integer in [required_prefix, donor_count]"
        )
    if target_count == 0:
        return {}

    # Canonicalize duplicate donor->recipient edges by retaining their true
    # minimum cost, then enumerate variables in stable donor/recipient order.
    canonical_edges: list[list[_CandidateEdge]] = []
    for donor, donor_edges in enumerate(edges):
        by_recipient: dict[int, _CandidateEdge] = {}
        for edge in donor_edges:
            if edge.recipient_index < 0:
                raise ValueError(
                    f"{condition}: donor {donor} has negative recipient index "
                    f"{edge.recipient_index}"
                )
            if not all(
                math.isfinite(value)
                for value in (edge.cost, edge.translation_gap, edge.yaw_gap)
            ):
                raise ValueError(f"{condition}: donor {donor} has a non-finite edge")
            previous = by_recipient.get(edge.recipient_index)
            if previous is None or (
                edge.cost,
                edge.translation_gap,
                edge.yaw_gap,
            ) < (
                previous.cost,
                previous.translation_gap,
                previous.yaw_gap,
            ):
                by_recipient[edge.recipient_index] = edge
        canonical_edges.append(
            sorted(
                by_recipient.values(),
                key=lambda edge: (
                    edge.recipient_index,
                    edge.cost,
                    edge.translation_gap,
                    edge.yaw_gap,
                ),
            )
        )

    empty_mandatory = [
        donor for donor in range(required_prefix) if not canonical_edges[donor]
    ]
    covered_donors = sum(bool(donor_edges) for donor_edges in canonical_edges)
    recipient_ids = sorted(
        {
            edge.recipient_index
            for donor_edges in canonical_edges
            for edge in donor_edges
        }
    )
    if empty_mandatory:
        raise InsufficientSamplesError(
            f"{condition}: mandatory visual donors have no candidate edges: "
            f"{empty_mandatory}"
        )
    if covered_donors < target_count or len(recipient_ids) < target_count:
        raise InsufficientSamplesError(
            f"{condition}: cannot match {target_count} unique pairs; "
            f"candidate_donors={donor_count}, covered_donors={covered_donors}, "
            f"unique_recipients={len(recipient_ids)}"
        )

    variables: list[tuple[int, _CandidateEdge]] = [
        (donor, edge)
        for donor, donor_edges in enumerate(canonical_edges)
        for edge in donor_edges
    ]
    recipient_row = {
        recipient: donor_count + index for index, recipient in enumerate(recipient_ids)
    }
    variable_count = len(variables)
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix, vstack
    except ImportError as exc:  # pragma: no cover - scipy is a project dependency
        raise RuntimeError(
            "Global minimum-cost action matching requires scipy.optimize"
        ) from exc

    ub_rows: list[int] = []
    ub_columns: list[int] = []
    for column, (donor, edge) in enumerate(variables):
        ub_rows.extend((donor, recipient_row[edge.recipient_index]))
        ub_columns.extend((column, column))
    a_ub = coo_matrix(
        (
            np.ones(len(ub_rows), dtype=np.float64),
            (np.asarray(ub_rows), np.asarray(ub_columns)),
        ),
        shape=(donor_count + len(recipient_ids), variable_count),
    ).tocsr()
    b_ub = np.ones(a_ub.shape[0], dtype=np.float64)

    eq_rows: list[int] = []
    eq_columns: list[int] = []
    for column, (donor, _) in enumerate(variables):
        eq_rows.append(0)  # fixed cardinality
        eq_columns.append(column)
        if donor < required_prefix:
            eq_rows.append(1 + donor)
            eq_columns.append(column)
    a_eq = coo_matrix(
        (
            np.ones(len(eq_rows), dtype=np.float64),
            (np.asarray(eq_rows), np.asarray(eq_columns)),
        ),
        shape=(1 + required_prefix, variable_count),
    ).tocsr()
    b_eq = np.concatenate(
        ([float(target_count)], np.ones(required_prefix, dtype=np.float64))
    )
    primary_cost = np.asarray([edge.cost for _, edge in variables], dtype=np.float64)
    solver_options = {
        "presolve": True,
        "dual_feasibility_tolerance": 1e-9,
        "primal_feasibility_tolerance": 1e-9,
    }
    primary = linprog(
        primary_cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, 1.0),
        method="highs",
        options=solver_options,
    )
    if not primary.success:
        raise InsufficientSamplesError(
            f"{condition}: global matching is infeasible for cardinality "
            f"{target_count} with {required_prefix} mandatory donors; "
            f"candidate_donors={donor_count}, covered_donors={covered_donors}, "
            f"unique_recipients={len(recipient_ids)}, solver={primary.message}"
        )

    primary_values = np.asarray(primary.x, dtype=np.float64)
    primary_rounded = np.rint(primary_values)
    if not np.allclose(primary_values, primary_rounded, rtol=0.0, atol=1e-6):
        fractional = [
            (index, float(value))
            for index, value in enumerate(primary_values)
            if abs(value - round(float(value))) > 1e-6
        ][:10]
        raise RuntimeError(
            f"{condition}: primary integral bipartite LP returned fractional "
            f"values: {fractional}"
        )

    # The primary optimum defines a face of the integral matching polytope.
    # Optimize a deterministic secondary objective on that face. Hash ranks
    # avoid additive donor/recipient rank ties while remaining independent of
    # Python's randomized hash seed.
    tie_cost = np.empty(variable_count, dtype=np.float64)
    for column, (donor, edge) in enumerate(variables):
        digest = hashlib.sha256(
            f"{donor}:{edge.recipient_index}".encode("ascii")
        ).digest()
        tie_cost[column] = int.from_bytes(digest[:8], "big") / float(1 << 64)
    optimum = float(primary.fun)
    optimum_row = coo_matrix(primary_cost.reshape(1, -1)).tocsr()
    secondary = linprog(
        tie_cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=vstack((a_eq, optimum_row), format="csr"),
        b_eq=np.concatenate((b_eq, [optimum])),
        bounds=(0.0, 1.0),
        method="highs",
        options=solver_options,
    )
    rounded = primary_rounded
    if secondary.success:
        secondary_values = np.asarray(secondary.x, dtype=np.float64)
        secondary_rounded = np.rint(secondary_values)
        if np.allclose(secondary_values, secondary_rounded, rtol=0.0, atol=1e-6):
            secondary_columns = np.flatnonzero(secondary_rounded > 0.5)
            secondary_cost = math.fsum(
                primary_cost[int(column)] for column in secondary_columns
            )
            if math.isclose(secondary_cost, optimum, rel_tol=1e-8, abs_tol=1e-8):
                rounded = secondary_rounded
    selected_columns = np.flatnonzero(rounded > 0.5)
    if len(selected_columns) != target_count:
        raise RuntimeError(
            f"{condition}: solver selected {len(selected_columns)} edges, expected "
            f"{target_count}"
        )
    assignment = {
        variables[int(column)][0]: variables[int(column)][1]
        for column in selected_columns
    }
    if len(assignment) != target_count:
        raise RuntimeError(f"{condition}: solver reused a donor")
    selected_recipients = [edge.recipient_index for edge in assignment.values()]
    if len(selected_recipients) != len(set(selected_recipients)):
        raise RuntimeError(f"{condition}: solver reused a recipient")
    if any(donor not in assignment for donor in range(required_prefix)):
        raise RuntimeError(f"{condition}: solver omitted a mandatory visual donor")
    selected_cost = math.fsum(edge.cost for edge in assignment.values())
    if not math.isclose(selected_cost, optimum, rel_tol=1e-8, abs_tol=1e-8):
        raise RuntimeError(
            f"{condition}: selected cost {selected_cost} differs from global optimum "
            f"{optimum}"
        )
    return assignment


def _allocate_nonvisual_donors(
    edges_by_offset: Sequence[Sequence[Sequence[_CandidateEdge]]],
    *,
    visual_count: int,
    per_offset_count: int,
    condition: str,
) -> list[list[int]]:
    """Diagnostic relaxation for donor-capacity allocation tests.

    This is a sparse bipartite b-matching between offsets (each with capacity
    ``per_offset_count``) and donors (each with capacity one).  Its LP
    relaxation is integral, so solving all offsets together avoids the false
    shortages produced by offset-ordered greedy consumption. Production
    matching does not use this relaxation because independently minimizing a
    donor's cheapest recipient can select a globally suboptimal donor subset;
    :func:`_match_cross_offsets` solves donor and recipient jointly instead.
    """

    offset_count = len(edges_by_offset)
    if offset_count == 0:
        raise ValueError("edges_by_offset must contain at least one offset")
    donor_count = len(edges_by_offset[0])
    if any(len(offset_edges) != donor_count for offset_edges in edges_by_offset):
        raise ValueError("Every offset must use the same ordered donor set")
    if (
        isinstance(visual_count, bool)
        or not isinstance(visual_count, int)
        or visual_count < 0
        or visual_count > donor_count
    ):
        raise ValueError("visual_count must be an integer in [0, donor_count]")
    if (
        isinstance(per_offset_count, bool)
        or not isinstance(per_offset_count, int)
        or per_offset_count < 0
        or per_offset_count > donor_count - visual_count
    ):
        raise ValueError(
            "per_offset_count must be an integer in [0, donor_count - visual_count]"
        )
    if per_offset_count == 0:
        return [[] for _ in range(offset_count)]

    required_total = offset_count * per_offset_count
    nonvisual_count = donor_count - visual_count
    if required_total > nonvisual_count:
        raise InsufficientSamplesError(
            f"{condition}: joint donor-capacity allocation needs {required_total} "
            f"globally unique non-visual donors, but only {nonvisual_count} exist"
        )

    variables: list[tuple[int, int]] = []
    primary_cost_values: list[float] = []
    covered_per_offset: list[int] = []
    for offset_index, offset_edges in enumerate(edges_by_offset):
        covered = 0
        for donor_index in range(visual_count, donor_count):
            donor_edges = offset_edges[donor_index]
            for edge in donor_edges:
                if edge.recipient_index < 0 or not all(
                    math.isfinite(value)
                    for value in (edge.cost, edge.translation_gap, edge.yaw_gap)
                ):
                    raise ValueError(
                        f"{condition}: offset {offset_index}, donor {donor_index} "
                        "has an invalid candidate edge"
                    )
            if not donor_edges:
                continue
            covered += 1
            variables.append((offset_index, donor_index))
            primary_cost_values.append(min(edge.cost for edge in donor_edges))
        covered_per_offset.append(covered)
    deficient_offsets = [
        (index, covered)
        for index, covered in enumerate(covered_per_offset)
        if covered < per_offset_count
    ]
    if deficient_offsets:
        raise InsufficientSamplesError(
            f"{condition}: offsets lack enough candidate-aware non-visual donors; "
            f"required_per_offset={per_offset_count}, coverage={deficient_offsets}"
        )
    globally_covered = len({donor for _, donor in variables})
    if globally_covered < required_total:
        raise InsufficientSamplesError(
            f"{condition}: only {globally_covered} non-visual donors have candidate "
            f"edges, but {required_total} globally unique donors are required"
        )

    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix, vstack
    except ImportError as exc:  # pragma: no cover - scipy is a project dependency
        raise RuntimeError(
            "Joint cross-offset donor allocation requires scipy.optimize"
        ) from exc

    variable_count = len(variables)
    columns = np.arange(variable_count, dtype=np.int64)
    donor_rows = np.asarray(
        [donor - visual_count for _, donor in variables], dtype=np.int64
    )
    offset_rows = np.asarray([offset for offset, _ in variables], dtype=np.int64)
    a_ub = coo_matrix(
        (np.ones(variable_count, dtype=np.float64), (donor_rows, columns)),
        shape=(nonvisual_count, variable_count),
    ).tocsr()
    b_ub = np.ones(nonvisual_count, dtype=np.float64)
    a_eq = coo_matrix(
        (np.ones(variable_count, dtype=np.float64), (offset_rows, columns)),
        shape=(offset_count, variable_count),
    ).tocsr()
    b_eq = np.full(offset_count, float(per_offset_count), dtype=np.float64)
    primary_cost = np.asarray(primary_cost_values, dtype=np.float64)
    solver_options = {
        "presolve": True,
        "dual_feasibility_tolerance": 1e-9,
        "primal_feasibility_tolerance": 1e-9,
    }
    primary = linprog(
        primary_cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, 1.0),
        method="highs",
        options=solver_options,
    )
    if not primary.success:
        diagnostic = (
            f"{condition}: joint donor-capacity allocation failed; "
            f"offsets={offset_count}, required_per_offset={per_offset_count}, "
            f"covered_per_offset={covered_per_offset}, "
            f"globally_covered_donors={globally_covered}, solver={primary.message}"
        )
        if getattr(primary, "status", None) == 2:
            raise InsufficientSamplesError(diagnostic)
        raise RuntimeError(diagnostic)

    # Resolve equal-cost allocations on the primary-optimal face.  Because the
    # primary equality is a supporting face of an integral b-matching
    # polytope, its vertices remain integral.
    tie_cost = np.empty(variable_count, dtype=np.float64)
    for column, (offset_index, donor_index) in enumerate(variables):
        digest = hashlib.sha256(
            f"{offset_index}:{donor_index}".encode("ascii")
        ).digest()
        tie_cost[column] = int.from_bytes(digest[:8], "big") / float(1 << 64)
    optimum = float(primary.fun)
    optimum_row = coo_matrix(primary_cost.reshape(1, -1)).tocsr()
    secondary = linprog(
        tie_cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=vstack((a_eq, optimum_row), format="csr"),
        b_eq=np.concatenate((b_eq, [optimum])),
        bounds=(0.0, 1.0),
        method="highs",
        options=solver_options,
    )
    if not secondary.success:
        raise RuntimeError(
            f"{condition}: deterministic joint donor-allocation tie-break failed "
            f"after primary optimum {optimum}: {secondary.message}"
        )
    values = np.asarray(secondary.x, dtype=np.float64)
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0.0, atol=1e-6):
        fractional = [
            (variables[index], float(value))
            for index, value in enumerate(values)
            if abs(value - round(float(value))) > 1e-6
        ][:10]
        raise RuntimeError(
            f"{condition}: integral donor-allocation LP returned fractional "
            f"values: {fractional}"
        )

    selected: list[list[int]] = [[] for _ in range(offset_count)]
    for column in np.flatnonzero(rounded > 0.5):
        offset_index, donor_index = variables[int(column)]
        selected[offset_index].append(donor_index)
    for donor_indices in selected:
        donor_indices.sort()
    if any(len(donors) != per_offset_count for donors in selected):
        raise RuntimeError(
            f"{condition}: joint allocator violated per-offset cardinality: "
            f"{[len(donors) for donors in selected]}"
        )
    flattened = [donor for donors in selected for donor in donors]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError(f"{condition}: joint allocator reused a non-visual donor")
    selected_cost = math.fsum(
        primary_cost_values[index] for index, value in enumerate(rounded) if value > 0.5
    )
    if not math.isclose(selected_cost, optimum, rel_tol=1e-8, abs_tol=1e-8):
        raise RuntimeError(
            f"{condition}: selected donor-allocation cost {selected_cost} differs "
            f"from global optimum {optimum}"
        )
    return selected


def _match_cross_offsets(
    edges_by_offset: Sequence[Sequence[Sequence[_CandidateEdge]]],
    *,
    visual_count: int,
    target_count: int,
    condition: str,
) -> tuple[list[dict[int, _CandidateEdge]], str]:
    """Jointly minimize donor/recipient cost over every source offset."""

    return _solve_joint_minimum_cost_cross_matching(
        edges_by_offset,
        visual_count=visual_count,
        target_count=target_count,
        condition=condition,
    )


def _solve_joint_minimum_cost_cross_matching(
    edges_by_offset: Sequence[Sequence[Sequence[_CandidateEdge]]],
    *,
    visual_count: int,
    target_count: int,
    condition: str,
) -> tuple[list[dict[int, _CandidateEdge]], str]:
    """Solve exact joint matching as an integral sparse min-cost-flow LP.

    Non-visual donors have one global unit of capacity, visual donors are
    copied per offset and forced to send one unit, recipient nodes have unit
    capacity within each offset, and every offset sends ``target_count`` units
    to the sink.  This is a single-commodity network-flow polytope, so its LP
    vertices are integral. Sparse arrays avoid Python tuples per candidate on
    million-edge formal graphs.
    """

    offset_count = len(edges_by_offset)
    if offset_count == 0:
        raise ValueError("edges_by_offset must contain at least one offset")
    donor_count = len(edges_by_offset[0])
    if any(len(offset_edges) != donor_count for offset_edges in edges_by_offset):
        raise ValueError("Every offset must use the same ordered donor set")
    if (
        isinstance(visual_count, bool)
        or not isinstance(visual_count, int)
        or visual_count < 0
        or visual_count > donor_count
    ):
        raise ValueError("visual_count must be an integer in [0, donor_count]")
    if (
        isinstance(target_count, bool)
        or not isinstance(target_count, int)
        or target_count < visual_count
        or target_count > donor_count
    ):
        raise ValueError(
            "target_count must be an integer in [visual_count, donor_count]"
        )

    required_nonvisual = offset_count * (target_count - visual_count)
    if required_nonvisual > donor_count - visual_count:
        raise InsufficientSamplesError(
            f"{condition}: exact joint matching needs {required_nonvisual} globally "
            f"unique non-visual donors, but only {donor_count - visual_count} exist"
        )

    # Canonicalize duplicate donor/recipient edges before allocating solver
    # arrays, retaining the true minimum edge under the same ordering as the
    # single-offset matcher.
    canonical: list[list[list[_CandidateEdge]]] = []
    covered_per_offset: list[int] = []
    recipients_per_offset: list[list[int]] = []
    for offset_index, offset_edges in enumerate(edges_by_offset):
        canonical_offset: list[list[_CandidateEdge]] = []
        for donor_index, donor_edges in enumerate(offset_edges):
            by_recipient: dict[int, _CandidateEdge] = {}
            for edge in donor_edges:
                if edge.recipient_index < 0 or not all(
                    math.isfinite(value)
                    for value in (edge.cost, edge.translation_gap, edge.yaw_gap)
                ):
                    raise ValueError(
                        f"{condition}: offset {offset_index}, donor {donor_index} "
                        "has an invalid candidate edge"
                    )
                previous = by_recipient.get(edge.recipient_index)
                if previous is None or (
                    edge.cost,
                    edge.translation_gap,
                    edge.yaw_gap,
                ) < (
                    previous.cost,
                    previous.translation_gap,
                    previous.yaw_gap,
                ):
                    by_recipient[edge.recipient_index] = edge
            canonical_offset.append(
                sorted(
                    by_recipient.values(),
                    key=lambda edge: (
                        edge.recipient_index,
                        edge.cost,
                        edge.translation_gap,
                        edge.yaw_gap,
                    ),
                )
            )
        empty_visual = [
            donor for donor in range(visual_count) if not canonical_offset[donor]
        ]
        if empty_visual:
            raise InsufficientSamplesError(
                f"{condition}/offset_index={offset_index}: mandatory visual donors "
                f"have no candidate edges: {empty_visual}"
            )
        covered = sum(bool(donor_edges) for donor_edges in canonical_offset)
        recipients = sorted(
            {
                edge.recipient_index
                for donor_edges in canonical_offset
                for edge in donor_edges
            }
        )
        if covered < target_count or len(recipients) < target_count:
            raise InsufficientSamplesError(
                f"{condition}/offset_index={offset_index}: exact joint graph cannot "
                f"provide {target_count} pairs; covered_donors={covered}, "
                f"unique_recipients={len(recipients)}"
            )
        canonical.append(canonical_offset)
        covered_per_offset.append(covered)
        recipients_per_offset.append(recipients)

    variable_count = sum(
        len(donor_edges) for offset_edges in canonical for donor_edges in offset_edges
    )
    if variable_count == 0:
        if target_count == 0:
            return ([{} for _ in range(offset_count)], "exact_joint_min_cost_flow_lp")
        raise InsufficientSamplesError(f"{condition}: exact joint graph has no edges")

    variable_offsets = np.empty(variable_count, dtype=np.int32)
    variable_donors = np.empty(variable_count, dtype=np.int32)
    variable_edge_indices = np.empty(variable_count, dtype=np.int32)
    variable_recipients = np.empty(variable_count, dtype=np.int64)
    primary_cost = np.empty(variable_count, dtype=np.float64)
    cursor = 0
    for offset_index, offset_edges in enumerate(canonical):
        for donor_index, donor_edges in enumerate(offset_edges):
            count = len(donor_edges)
            if count == 0:
                continue
            selection = slice(cursor, cursor + count)
            variable_offsets[selection] = offset_index
            variable_donors[selection] = donor_index
            variable_edge_indices[selection] = np.arange(count, dtype=np.int32)
            variable_recipients[selection] = [
                edge.recipient_index for edge in donor_edges
            ]
            primary_cost[selection] = [edge.cost for edge in donor_edges]
            cursor += count
    if cursor != variable_count:
        raise RuntimeError(f"{condition}: joint variable enumeration drifted")

    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix, vstack
    except ImportError as exc:  # pragma: no cover - scipy is a project dependency
        raise RuntimeError(
            "Exact joint cross matching requires scipy.optimize"
        ) from exc

    nonvisual_count = donor_count - visual_count
    recipient_maps: list[dict[int, int]] = []
    recipient_bases: list[int] = []
    next_row = nonvisual_count
    for recipients in recipients_per_offset:
        recipient_bases.append(next_row)
        recipient_maps.append(
            {recipient: index for index, recipient in enumerate(recipients)}
        )
        next_row += len(recipients)

    columns = np.arange(variable_count, dtype=np.int64)
    recipient_rows = np.empty(variable_count, dtype=np.int64)
    for offset_index in range(offset_count):
        indices = np.flatnonzero(variable_offsets == offset_index)
        mapping = recipient_maps[offset_index]
        recipient_rows[indices] = [
            recipient_bases[offset_index] + mapping[int(variable_recipients[index])]
            for index in indices
        ]
    nonvisual_columns = np.flatnonzero(variable_donors >= visual_count)
    nonvisual_donor_rows = (
        variable_donors[nonvisual_columns].astype(np.int64) - visual_count
    )
    a_ub = coo_matrix(
        (
            np.ones(variable_count + len(nonvisual_columns), dtype=np.float64),
            (
                np.concatenate((recipient_rows, nonvisual_donor_rows)),
                np.concatenate((columns, nonvisual_columns)),
            ),
        ),
        shape=(next_row, variable_count),
    ).tocsr()
    b_ub = np.ones(next_row, dtype=np.float64)

    visual_columns = np.flatnonzero(variable_donors < visual_count)
    visual_rows = variable_offsets[visual_columns].astype(
        np.int64
    ) * visual_count + variable_donors[visual_columns].astype(np.int64)
    cardinality_rows = offset_count * visual_count + variable_offsets.astype(np.int64)
    equality_count = offset_count * visual_count + offset_count
    a_eq = coo_matrix(
        (
            np.ones(variable_count + len(visual_columns), dtype=np.float64),
            (
                np.concatenate((cardinality_rows, visual_rows)),
                np.concatenate((columns, visual_columns)),
            ),
        ),
        shape=(equality_count, variable_count),
    ).tocsr()
    b_eq = np.concatenate(
        (
            np.ones(offset_count * visual_count, dtype=np.float64),
            np.full(offset_count, float(target_count), dtype=np.float64),
        )
    )
    solver_options = {
        "presolve": True,
        "dual_feasibility_tolerance": 1e-9,
        "primal_feasibility_tolerance": 1e-9,
    }
    primary = linprog(
        primary_cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, 1.0),
        method="highs",
        options=solver_options,
    )
    if not primary.success:
        diagnostic = (
            f"{condition}: exact joint donor/recipient matching failed; "
            f"offsets={offset_count}, target_per_offset={target_count}, "
            f"covered_per_offset={covered_per_offset}, variables={variable_count}, "
            f"solver={primary.message}"
        )
        if getattr(primary, "status", None) == 2:
            raise InsufficientSamplesError(diagnostic)
        raise RuntimeError(diagnostic)

    primary_values = np.asarray(primary.x, dtype=np.float64)
    primary_rounded = np.rint(primary_values)
    if not np.allclose(primary_values, primary_rounded, rtol=0.0, atol=1e-6):
        fractional = [
            (int(index), float(value))
            for index, value in enumerate(primary_values)
            if abs(value - round(float(value))) > 1e-6
        ][:10]
        raise RuntimeError(
            f"{condition}: primary integral min-cost-flow LP returned fractional "
            f"values: {fractional}"
        )

    # SplitMix64-style vectorized ranks give a stable tie policy without a
    # Python SHA-256 object per edge on million-variable graphs.
    mixed = (variable_offsets.astype(np.uint64) + np.uint64(1)) * np.uint64(
        0x9E3779B97F4A7C15
    )
    mixed ^= (variable_donors.astype(np.uint64) + np.uint64(1)) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    mixed ^= (variable_recipients.astype(np.uint64) + np.uint64(1)) * np.uint64(
        0x94D049BB133111EB
    )
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    tie_cost = (mixed >> np.uint64(11)).astype(np.float64) / float(1 << 53)

    optimum = float(primary.fun)
    secondary_eq = vstack(
        (a_eq, coo_matrix(primary_cost.reshape(1, -1)).tocsr()), format="csr"
    )
    secondary = linprog(
        tie_cost,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=secondary_eq,
        b_eq=np.concatenate((b_eq, [optimum])),
        bounds=(0.0, 1.0),
        method="highs",
        options=solver_options,
    )
    strategy = "exact_joint_min_cost_flow_lp"
    rounded = primary_rounded
    if secondary.success:
        secondary_values = np.asarray(secondary.x, dtype=np.float64)
        secondary_rounded = np.rint(secondary_values)
        secondary_integral = np.allclose(
            secondary_values, secondary_rounded, rtol=0.0, atol=1e-6
        )
        if secondary_integral:
            secondary_columns = np.flatnonzero(secondary_rounded > 0.5)
            secondary_cost = math.fsum(
                primary_cost[int(column)] for column in secondary_columns
            )
            if math.isclose(secondary_cost, optimum, rel_tol=1e-8, abs_tol=1e-8):
                rounded = secondary_rounded
            else:
                strategy = "exact_joint_min_cost_flow_lp_primary_tie_fallback"
        else:
            strategy = "exact_joint_min_cost_flow_lp_primary_tie_fallback"
    else:
        # An exact floating-point equality at ``primary.fun`` can occasionally
        # make HiGHS reject an otherwise valid secondary solve. The primary
        # vertex was already checked integral and globally optimal, so retain
        # it rather than misreporting data insufficiency.
        strategy = "exact_joint_min_cost_flow_lp_primary_tie_fallback"

    assignments: list[dict[int, _CandidateEdge]] = [{} for _ in range(offset_count)]
    recipients_used: list[set[int]] = [set() for _ in range(offset_count)]
    nonvisual_used: set[int] = set()
    for column in np.flatnonzero(rounded > 0.5):
        index = int(column)
        offset_index = int(variable_offsets[index])
        donor_index = int(variable_donors[index])
        edge_index = int(variable_edge_indices[index])
        edge = canonical[offset_index][donor_index][edge_index]
        if donor_index in assignments[offset_index]:
            raise RuntimeError(f"{condition}: joint solver reused an offset donor")
        if edge.recipient_index in recipients_used[offset_index]:
            raise RuntimeError(f"{condition}: joint solver reused an offset recipient")
        if donor_index >= visual_count:
            if donor_index in nonvisual_used:
                raise RuntimeError(
                    f"{condition}: joint solver reused a non-visual donor"
                )
            nonvisual_used.add(donor_index)
        assignments[offset_index][donor_index] = edge
        recipients_used[offset_index].add(edge.recipient_index)

    for offset_index, assignment in enumerate(assignments):
        if len(assignment) != target_count:
            raise RuntimeError(
                f"{condition}: joint solver selected {len(assignment)} pairs for "
                f"offset {offset_index}, expected {target_count}"
            )
        if any(donor not in assignment for donor in range(visual_count)):
            raise RuntimeError(
                f"{condition}: joint solver omitted a mandatory visual donor at "
                f"offset {offset_index}"
            )
    selected_cost = math.fsum(
        primary_cost[int(column)] for column in np.flatnonzero(rounded > 0.5)
    )
    if not math.isclose(selected_cost, optimum, rel_tol=1e-8, abs_tol=1e-8):
        raise RuntimeError(
            f"{condition}: selected exact-joint cost {selected_cost} differs from "
            f"global optimum {optimum}"
        )
    return assignments, strategy


def _joint_minimum_cost_cross_matching(
    edges_by_offset: Sequence[Sequence[Sequence[_CandidateEdge]]],
    *,
    visual_count: int,
    target_count: int,
    condition: str,
) -> list[dict[int, _CandidateEdge]]:
    """Backward-compatible assignment-only wrapper for the exact solver."""

    assignments, _ = _solve_joint_minimum_cost_cross_matching(
        edges_by_offset,
        visual_count=visual_count,
        target_count=target_count,
        condition=condition,
    )
    return assignments


def _joint_select_visual_cross_matching(
    edges_by_offset: Sequence[Sequence[Sequence[_CandidateEdge]]],
    *,
    visual_count: int,
    target_count: int,
    condition: str,
) -> tuple[list[dict[int, _CandidateEdge]], list[int], str]:
    """Jointly choose complete visual donors and all recipient assignments.

    The regular min-cost-flow path receives an already fixed visual prefix.
    This exact binary fallback is used only when that prefix has a visual Hall
    conflict, or after the deterministic donor prefix has expanded to the full
    legal source inventory. Binary ``y[d]`` variables select the donors reused
    once at every offset; all donors with ``y[d] == 0`` retain global capacity
    one. Consequently an infeasible full-pool solve is a proof of true sample
    insufficiency under the configured tolerances.
    """

    offset_count = len(edges_by_offset)
    if offset_count == 0:
        raise ValueError("edges_by_offset must contain at least one offset")
    donor_count = len(edges_by_offset[0])
    if any(len(offset_edges) != donor_count for offset_edges in edges_by_offset):
        raise ValueError("Every offset must use the same ordered donor set")
    if (
        isinstance(visual_count, bool)
        or not isinstance(visual_count, int)
        or visual_count < 0
        or visual_count > donor_count
    ):
        raise ValueError("visual_count must be an integer in [0, donor_count]")
    if (
        isinstance(target_count, bool)
        or not isinstance(target_count, int)
        or target_count < visual_count
        or target_count > donor_count
    ):
        raise ValueError(
            "target_count must be an integer in [visual_count, donor_count]"
        )
    required_distinct = visual_count + offset_count * (target_count - visual_count)
    if required_distinct > donor_count:
        raise InsufficientSamplesError(
            f"{condition}: joint visual selection needs {required_distinct} distinct "
            f"donor anchor-windows, but only {donor_count} exist"
        )
    if visual_count == 0:
        assignments, strategy = _solve_joint_minimum_cost_cross_matching(
            edges_by_offset,
            visual_count=0,
            target_count=target_count,
            condition=condition,
        )
        return assignments, [], strategy

    canonical: list[list[list[_CandidateEdge]]] = []
    recipients_per_offset: list[list[int]] = []
    covered_per_offset: list[int] = []
    for offset_index, offset_edges in enumerate(edges_by_offset):
        canonical_offset: list[list[_CandidateEdge]] = []
        for donor_index, donor_edges in enumerate(offset_edges):
            by_recipient: dict[int, _CandidateEdge] = {}
            for edge in donor_edges:
                if edge.recipient_index < 0 or not all(
                    math.isfinite(value)
                    for value in (edge.cost, edge.translation_gap, edge.yaw_gap)
                ):
                    raise ValueError(
                        f"{condition}: offset {offset_index}, donor {donor_index} "
                        "has an invalid candidate edge"
                    )
                previous = by_recipient.get(edge.recipient_index)
                if previous is None or (
                    edge.cost,
                    edge.translation_gap,
                    edge.yaw_gap,
                ) < (
                    previous.cost,
                    previous.translation_gap,
                    previous.yaw_gap,
                ):
                    by_recipient[edge.recipient_index] = edge
            canonical_offset.append(
                sorted(
                    by_recipient.values(),
                    key=lambda edge: (
                        edge.recipient_index,
                        edge.cost,
                        edge.translation_gap,
                        edge.yaw_gap,
                    ),
                )
            )
        covered = sum(bool(donor_edges) for donor_edges in canonical_offset)
        recipients = sorted(
            {
                edge.recipient_index
                for donor_edges in canonical_offset
                for edge in donor_edges
            }
        )
        if covered < target_count or len(recipients) < target_count:
            raise InsufficientSamplesError(
                f"{condition}/offset_index={offset_index}: joint visual graph "
                f"cannot provide {target_count} pairs; covered_donors={covered}, "
                f"unique_recipients={len(recipients)}"
            )
        canonical.append(canonical_offset)
        recipients_per_offset.append(recipients)
        covered_per_offset.append(covered)

    eligible_visual = [
        donor_index
        for donor_index in range(donor_count)
        if all(canonical[offset][donor_index] for offset in range(offset_count))
    ]
    if len(eligible_visual) < visual_count:
        raise InsufficientSamplesError(
            f"{condition}: only {len(eligible_visual)} donors have candidate edges "
            f"at every offset; requested {visual_count} complete visual donors"
        )

    edge_variable_count = sum(
        len(donor_edges) for offset_edges in canonical for donor_edges in offset_edges
    )
    variable_offsets = np.empty(edge_variable_count, dtype=np.int32)
    variable_donors = np.empty(edge_variable_count, dtype=np.int32)
    variable_edge_indices = np.empty(edge_variable_count, dtype=np.int32)
    variable_recipients = np.empty(edge_variable_count, dtype=np.int64)
    edge_cost = np.empty(edge_variable_count, dtype=np.float64)
    cursor = 0
    for offset_index, offset_edges in enumerate(canonical):
        for donor_index, donor_edges in enumerate(offset_edges):
            count = len(donor_edges)
            if not count:
                continue
            selection = slice(cursor, cursor + count)
            variable_offsets[selection] = offset_index
            variable_donors[selection] = donor_index
            variable_edge_indices[selection] = np.arange(count, dtype=np.int32)
            variable_recipients[selection] = [
                edge.recipient_index for edge in donor_edges
            ]
            edge_cost[selection] = [edge.cost for edge in donor_edges]
            cursor += count
    if cursor != edge_variable_count:
        raise RuntimeError(f"{condition}: visual MILP variable enumeration drifted")

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix, vstack
    except ImportError as exc:  # pragma: no cover - scipy is a project dependency
        raise RuntimeError(
            "Exact complete-visual selection requires scipy.optimize.milp"
        ) from exc

    eligible_visual_array = np.asarray(eligible_visual, dtype=np.int32)
    eligible_rank = np.full(donor_count, -1, dtype=np.int64)
    eligible_rank[eligible_visual_array] = np.arange(
        len(eligible_visual), dtype=np.int64
    )
    y_columns = edge_variable_count + np.arange(len(eligible_visual), dtype=np.int64)
    variable_count = edge_variable_count + len(eligible_visual)
    x_columns = np.arange(edge_variable_count, dtype=np.int64)

    recipient_bases: list[int] = []
    recipient_maps: list[dict[int, int]] = []
    next_row = donor_count
    for recipients in recipients_per_offset:
        recipient_bases.append(next_row)
        recipient_maps.append(
            {recipient: index for index, recipient in enumerate(recipients)}
        )
        next_row += len(recipients)
    activation_base = next_row
    activation_count = len(eligible_visual) * offset_count
    cardinality_base = activation_base + activation_count
    visual_count_row = cardinality_base + offset_count
    row_count = visual_count_row + 1

    recipient_rows = np.empty(edge_variable_count, dtype=np.int64)
    for offset_index in range(offset_count):
        indices = np.flatnonzero(variable_offsets == offset_index)
        mapping = recipient_maps[offset_index]
        recipient_rows[indices] = [
            recipient_bases[offset_index] + mapping[int(variable_recipients[index])]
            for index in indices
        ]
    cardinality_rows = cardinality_base + variable_offsets.astype(np.int64)
    eligible_x = eligible_rank[variable_donors] >= 0
    eligible_x_columns = x_columns[eligible_x]
    eligible_activation_rows = (
        activation_base
        + eligible_rank[variable_donors[eligible_x]] * offset_count
        + variable_offsets[eligible_x].astype(np.int64)
    )
    y_activation_rows = (
        activation_base
        + np.repeat(np.arange(len(eligible_visual), dtype=np.int64), offset_count)
        * offset_count
        + np.tile(np.arange(offset_count, dtype=np.int64), len(eligible_visual))
    )
    y_activation_columns = np.repeat(y_columns, offset_count)

    rows = np.concatenate(
        (
            variable_donors.astype(np.int64),
            recipient_rows,
            cardinality_rows,
            eligible_activation_rows,
            eligible_visual_array.astype(np.int64),
            y_activation_rows,
            np.full(len(eligible_visual), visual_count_row, dtype=np.int64),
        )
    )
    columns = np.concatenate(
        (
            x_columns,
            x_columns,
            x_columns,
            eligible_x_columns,
            y_columns,
            y_activation_columns,
            y_columns,
        )
    )
    data = np.concatenate(
        (
            np.ones(edge_variable_count * 3, dtype=np.float64),
            -np.ones(len(eligible_x_columns), dtype=np.float64),
            np.full(len(eligible_visual), -(offset_count - 1), dtype=np.float64),
            np.ones(len(y_activation_columns), dtype=np.float64),
            np.ones(len(eligible_visual), dtype=np.float64),
        )
    )
    constraint_matrix = coo_matrix(
        (data, (rows, columns)), shape=(row_count, variable_count)
    ).tocsr()
    lower = np.full(row_count, -np.inf, dtype=np.float64)
    upper = np.full(row_count, np.inf, dtype=np.float64)
    upper[:donor_count] = 1.0
    upper[donor_count:activation_base] = 1.0
    upper[activation_base:cardinality_base] = 0.0
    lower[cardinality_base:visual_count_row] = float(target_count)
    upper[cardinality_base:visual_count_row] = float(target_count)
    lower[visual_count_row] = float(visual_count)
    upper[visual_count_row] = float(visual_count)

    primary_cost = np.concatenate(
        (edge_cost, np.zeros(len(eligible_visual), dtype=np.float64))
    )
    integrality = np.ones(variable_count, dtype=np.uint8)
    variable_bounds = Bounds(
        np.zeros(variable_count, dtype=np.float64),
        np.ones(variable_count, dtype=np.float64),
    )
    primary = milp(
        c=primary_cost,
        integrality=integrality,
        bounds=variable_bounds,
        constraints=LinearConstraint(constraint_matrix, lower, upper),
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not primary.success:
        diagnostic = (
            f"{condition}: exact joint complete-visual selection failed; "
            f"offsets={offset_count}, target_per_offset={target_count}, "
            f"visual_count={visual_count}, eligible_visual="
            f"{len(eligible_visual)}, covered_per_offset={covered_per_offset}, "
            f"variables={variable_count}, solver={primary.message}"
        )
        if getattr(primary, "status", None) == 2:
            raise InsufficientSamplesError(diagnostic)
        raise RuntimeError(diagnostic)

    primary_values = np.asarray(primary.x, dtype=np.float64)
    primary_rounded = np.rint(primary_values)
    if not np.allclose(primary_values, primary_rounded, rtol=0.0, atol=1e-6):
        fractional = [
            (int(index), float(value))
            for index, value in enumerate(primary_values)
            if abs(value - round(float(value))) > 1e-6
        ][:10]
        raise RuntimeError(
            f"{condition}: primary complete-visual MILP returned fractional "
            f"values: {fractional}"
        )

    mixed_x = (variable_offsets.astype(np.uint64) + np.uint64(1)) * np.uint64(
        0x9E3779B97F4A7C15
    )
    mixed_x ^= (variable_donors.astype(np.uint64) + np.uint64(1)) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    mixed_x ^= (variable_recipients.astype(np.uint64) + np.uint64(1)) * np.uint64(
        0x94D049BB133111EB
    )
    mixed_x ^= mixed_x >> np.uint64(30)
    mixed_x *= np.uint64(0xBF58476D1CE4E5B9)
    mixed_x ^= mixed_x >> np.uint64(27)
    mixed_x *= np.uint64(0x94D049BB133111EB)
    mixed_x ^= mixed_x >> np.uint64(31)
    mixed_y = (eligible_visual_array.astype(np.uint64) + np.uint64(1)) * np.uint64(
        0xD6E8FEB86659FD93
    )
    mixed_y ^= mixed_y >> np.uint64(32)
    mixed_y *= np.uint64(0x9E3779B97F4A7C15)
    mixed_y ^= mixed_y >> np.uint64(29)
    tie_cost = np.concatenate(
        (
            (mixed_x >> np.uint64(11)).astype(np.float64) / float(1 << 53),
            (mixed_y >> np.uint64(11)).astype(np.float64) / float(1 << 53),
        )
    )
    optimum = float(primary.fun)
    secondary_matrix = vstack(
        (constraint_matrix, coo_matrix(primary_cost.reshape(1, -1)).tocsr()),
        format="csr",
    )
    secondary = milp(
        c=tie_cost,
        integrality=integrality,
        bounds=variable_bounds,
        constraints=LinearConstraint(
            secondary_matrix,
            np.concatenate((lower, [optimum])),
            np.concatenate((upper, [optimum])),
        ),
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    strategy = "exact_joint_visual_selection_milp"
    rounded = primary_rounded
    if secondary.success:
        secondary_values = np.asarray(secondary.x, dtype=np.float64)
        secondary_rounded = np.rint(secondary_values)
        secondary_integral = np.allclose(
            secondary_values, secondary_rounded, rtol=0.0, atol=1e-6
        )
        if secondary_integral:
            selected_edge_columns = np.flatnonzero(
                secondary_rounded[:edge_variable_count] > 0.5
            )
            secondary_cost = math.fsum(
                edge_cost[int(column)] for column in selected_edge_columns
            )
            if math.isclose(secondary_cost, optimum, rel_tol=1e-8, abs_tol=1e-8):
                rounded = secondary_rounded
            else:
                strategy += "_primary_tie_fallback"
        else:
            strategy += "_primary_tie_fallback"
    else:
        strategy += "_primary_tie_fallback"

    selected_visual_columns = np.flatnonzero(rounded[edge_variable_count:] > 0.5)
    selected_visual = sorted(
        int(eligible_visual[int(index)]) for index in selected_visual_columns
    )
    if len(selected_visual) != visual_count:
        raise RuntimeError(
            f"{condition}: visual MILP selected {len(selected_visual)} complete "
            f"donors, expected {visual_count}"
        )
    selected_visual_set = set(selected_visual)
    assignments: list[dict[int, _CandidateEdge]] = [{} for _ in range(offset_count)]
    recipients_used: list[set[int]] = [set() for _ in range(offset_count)]
    nonvisual_used: set[int] = set()
    selected_x_columns = np.flatnonzero(rounded[:edge_variable_count] > 0.5)
    for column in selected_x_columns:
        index = int(column)
        offset_index = int(variable_offsets[index])
        donor_index = int(variable_donors[index])
        edge = canonical[offset_index][donor_index][int(variable_edge_indices[index])]
        if donor_index in assignments[offset_index]:
            raise RuntimeError(f"{condition}: visual MILP reused an offset donor")
        if edge.recipient_index in recipients_used[offset_index]:
            raise RuntimeError(f"{condition}: visual MILP reused an offset recipient")
        if donor_index not in selected_visual_set:
            if donor_index in nonvisual_used:
                raise RuntimeError(
                    f"{condition}: visual MILP reused a non-visual donor"
                )
            nonvisual_used.add(donor_index)
        assignments[offset_index][donor_index] = edge
        recipients_used[offset_index].add(edge.recipient_index)

    for offset_index, assignment in enumerate(assignments):
        if len(assignment) != target_count:
            raise RuntimeError(
                f"{condition}: visual MILP selected {len(assignment)} pairs for "
                f"offset {offset_index}, expected {target_count}"
            )
        if not selected_visual_set.issubset(assignment):
            raise RuntimeError(
                f"{condition}: visual MILP omitted a selected complete donor at "
                f"offset {offset_index}"
            )
    selected_cost = math.fsum(edge_cost[int(column)] for column in selected_x_columns)
    if not math.isclose(selected_cost, optimum, rel_tol=1e-8, abs_tol=1e-8):
        raise RuntimeError(
            f"{condition}: selected complete-visual cost {selected_cost} differs "
            f"from global optimum {optimum}"
        )
    return assignments, selected_visual, strategy


def manifest_records_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(_canonical_json(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_manifest(
    records: Sequence[Mapping[str, Any]],
    path: str | Path,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write stable JSONL plus ``<stem>.metadata.json`` sidecar."""

    materialized = [dict(record) for record in records]
    _validate_records(materialized)
    output = Path(path).expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = _metadata_path(output)
    digest = manifest_records_sha256(materialized)
    resolved_metadata = dict(metadata or {})
    resolved_metadata.update(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "record_count": len(materialized),
            "records_sha256": digest,
            "manifest_path": str(output),
        }
    )
    resolved_metadata["metadata_sha256"] = manifest_metadata_sha256(resolved_metadata)
    if isinstance(resolved_metadata.get("data_integrity"), Mapping):
        resolved_metadata["evaluation_manifest_sha256"] = evaluation_manifest_sha256(
            resolved_metadata
        )
    else:
        resolved_metadata.pop("evaluation_manifest_sha256", None)
    lines = "".join(_canonical_json(record) + "\n" for record in materialized)
    _atomic_text(output, lines)
    _atomic_text(
        sidecar,
        json.dumps(resolved_metadata, sort_keys=True, indent=2, allow_nan=False) + "\n",
    )
    return resolved_metadata


def load_manifest(
    path: str | Path, *, verify_hash: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load JSONL records and metadata, verifying count and SHA-256 by default."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest does not exist: {manifest_path}")
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"line {line_number} is not a JSON object")
            records.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(
            f"Cannot read evaluation manifest {manifest_path}: {exc}"
        ) from exc
    _validate_records(records)
    sidecar = _metadata_path(manifest_path)
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"Evaluation manifest metadata does not exist: {sidecar}"
        )
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read evaluation manifest metadata {sidecar}: {exc}"
        ) from exc
    if not isinstance(metadata, dict):
        raise TypeError(f"Evaluation manifest metadata must be an object: {sidecar}")
    if verify_hash:
        expected_count = metadata.get("record_count")
        if expected_count != len(records):
            raise RuntimeError(
                f"Manifest record count mismatch: metadata={expected_count}, actual={len(records)}"
            )
        actual = manifest_records_sha256(records)
        if metadata.get("records_sha256") != actual:
            raise RuntimeError(
                f"Manifest SHA-256 mismatch: metadata={metadata.get('records_sha256')}, actual={actual}"
            )
        actual_metadata_sha = manifest_metadata_sha256(metadata)
        if metadata.get("metadata_sha256") != actual_metadata_sha:
            raise RuntimeError(
                "Manifest semantic metadata SHA-256 mismatch: "
                f"metadata={metadata.get('metadata_sha256')}, "
                f"actual={actual_metadata_sha}"
            )
        if isinstance(metadata.get("data_integrity"), Mapping):
            actual_evaluation_sha = evaluation_manifest_sha256(metadata)
            if metadata.get("evaluation_manifest_sha256") != actual_evaluation_sha:
                raise RuntimeError(
                    "Evaluation manifest composite SHA-256 mismatch: "
                    f"metadata={metadata.get('evaluation_manifest_sha256')}, "
                    f"actual={actual_evaluation_sha}"
                )
    return records, metadata


def _catalog_specs(
    config: Mapping[str, Any], raw: Mapping[str, Any]
) -> list[DatasetSplitSpec]:
    catalog_raw = raw.get("datasets", raw.get("catalog"))
    if catalog_raw is None:
        data = config.get("data", {})
        if isinstance(data, Mapping):
            catalog_raw = data.get("source_specs")
    if catalog_raw is None:
        raise ValueError("Evaluation config requires explicit datasets/catalog entries")
    expanded: list[Mapping[str, Any]] = []
    if isinstance(catalog_raw, Sequence) and not isinstance(catalog_raw, (str, bytes)):
        expanded = list(catalog_raw)
    elif isinstance(catalog_raw, Mapping):
        for raw_dataset, raw_splits in catalog_raw.items():
            if not isinstance(raw_splits, Mapping):
                raise TypeError(f"datasets.{raw_dataset} must be a split mapping")
            # Public YAML: one dataset block owns shared root/spacing and a
            # ``splits: {train: manifest, test: manifest}`` mapping.
            if "splits" in raw_splits:
                split_values = raw_splits["splits"]
                if not isinstance(split_values, Mapping):
                    raise TypeError(f"datasets.{raw_dataset}.splits must be a mapping")
                common = {
                    key: value for key, value in raw_splits.items() if key != "splits"
                }
                for raw_split, raw_manifest in split_values.items():
                    item = dict(common)
                    item.setdefault("dataset_id", raw_dataset)
                    item["split"] = raw_split
                    if isinstance(raw_manifest, Mapping):
                        item.update(raw_manifest)
                    else:
                        item["split_manifest"] = raw_manifest
                    expanded.append(item)
                continue
            if "data_root" in raw_splits:
                item = dict(raw_splits)
                item.setdefault("dataset_id", raw_dataset)
                expanded.append(item)
                continue
            for raw_split, raw_spec in raw_splits.items():
                if not isinstance(raw_spec, Mapping):
                    raise TypeError(
                        f"datasets.{raw_dataset}.{raw_split} must be a mapping"
                    )
                item = dict(raw_spec)
                item.setdefault("dataset_id", raw_dataset)
                item.setdefault("split", raw_split)
                expanded.append(item)
    else:
        raise TypeError("datasets/catalog must be a sequence or nested mapping")
    return [DatasetSplitSpec.from_mapping(item) for item in expanded]


def _setting(
    raw: Mapping[str, Any], sampling: Mapping[str, Any], name: str, default: Any
) -> Any:
    return raw[name] if name in raw else sampling.get(name, default)


def _integer_setting(
    raw: Mapping[str, Any],
    sampling: Mapping[str, Any],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    value = _setting(raw, sampling, name, default)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _integer_alias_setting(
    raw: Mapping[str, Any],
    group: Mapping[str, Any],
    names: Sequence[str],
    default: int,
    *,
    minimum: int,
) -> int:
    found = [name for name in names if name in raw or name in group]
    if len(found) > 1:
        values = {name: raw[name] if name in raw else group[name] for name in found}
        if len({int(value) for value in values.values()}) > 1:
            raise ValueError(f"Conflicting aliases for {names[0]}: {values}")
    name = found[0] if found else names[0]
    value = raw[name] if name in raw else group.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _float_setting(
    raw: Mapping[str, Any],
    sampling: Mapping[str, Any],
    name: str,
    default: float,
    *,
    positive: bool,
) -> float:
    value = _setting(raw, sampling, name, default)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}")
    return result


def _float_alias_setting(
    raw: Mapping[str, Any],
    group: Mapping[str, Any],
    names: Sequence[str],
    default: float,
    *,
    positive: bool,
) -> float:
    found = [name for name in names if name in raw or name in group]
    if len(found) > 1:
        values = {name: raw[name] if name in raw else group[name] for name in found}
        if len({float(value) for value in values.values()}) > 1:
            raise ValueError(f"Conflicting aliases for {names[0]}: {values}")
    name = found[0] if found else names[0]
    value = raw[name] if name in raw else group.get(name, default)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}")
    return result


def _dataset_ids(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(canonical_dataset_id(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be non-empty without aliases/duplicates")
    return result


def _protocols(values: Any) -> tuple[str, ...]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Sequence):
        raise TypeError("protocols must be a sequence")
    normalized = {str(value).strip().lower().replace("-", "_") for value in values}
    unknown = normalized - set(PROTOCOL_ORDER)
    if unknown:
        raise ValueError(f"Unknown evaluation protocols: {sorted(unknown)}")
    if not normalized:
        raise ValueError("protocols must not be empty")
    return tuple(protocol for protocol in PROTOCOL_ORDER if protocol in normalized)


def _legal_window_count(partition: DatasetPartition, offsets: Sequence[int]) -> int:
    lower = max(0, -min(offsets))
    upper_margin = max(offsets)
    return sum(
        max(0, trajectory.length - upper_margin - lower)
        for trajectory in partition.trajectories
    )


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = _canonical_json([int(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _record_id(record: Mapping[str, Any]) -> str:
    identity = {
        key: record.get(key)
        for key in (
            "condition_id",
            "source_dataset",
            "source_split",
            "source_trajectory",
            "source_anchor",
            "source_target",
            "recipient_dataset",
            "recipient_split",
            "recipient_trajectory",
            "recipient_anchor",
            "recipient_target",
        )
    }
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:24]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    ids: set[str] = set()
    required = {
        "condition_id",
        "protocol",
        "source_dataset",
        "source_split",
        "source_trajectory",
        "source_anchor",
        "source_target",
        "source_offset",
        "is_visual",
        "record_id",
    }
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"Manifest record {index} is not a mapping")
        missing = required - set(record)
        if missing:
            raise ValueError(
                f"Manifest record {index} is missing fields: {sorted(missing)}"
            )
        record_id = record["record_id"]
        if not isinstance(record_id, str) or not record_id:
            raise TypeError(
                f"Manifest record {index}.record_id must be a non-empty string"
            )
        if record_id in ids:
            raise ValueError(f"Duplicate manifest record_id: {record_id}")
        ids.add(record_id)
        _canonical_json(record)  # rejects NaN/Inf and non-JSON values


def _metadata_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f"{manifest_path.stem}.metadata.json")


def _atomic_text(path: Path, text: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
