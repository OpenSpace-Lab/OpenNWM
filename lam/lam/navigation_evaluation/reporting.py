"""Metric aggregation and portable reports for navigation evaluation."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from .io import atomic_write_json, atomic_write_text, write_csv

# Manifest/provenance values remain in per-pair output but are never metrics.
IDENTITY_COLUMNS = {
    "schema_version",
    "record_id",
    "condition_id",
    "protocol",
    "source_dataset",
    "source_split",
    "source_trajectory",
    "source_anchor",
    "source_target",
    "source_offset",
    "source_window_id",
    "source_anchor_path",
    "source_target_path",
    "source_waypoint_spacing",
    "source_action",
    "source_action_normalized",
    "source_action_waypoint",
    "source_action_meters",
    "recipient_dataset",
    "recipient_split",
    "recipient_trajectory",
    "recipient_anchor",
    "recipient_target",
    "recipient_offset",
    "recipient_anchor_path",
    "recipient_target_path",
    "recipient_waypoint_spacing",
    "recipient_action",
    "recipient_action_normalized",
    "recipient_action_waypoint",
    "recipient_action_meters",
    # Legacy aliases accepted when comparing an earlier report.
    "target_dataset",
    "target_split",
    "target_trajectory",
    "target_anchor",
    "target_target",
    "target_offset",
    "anchor",
    "target",
    "offset",
    "window_index",
    "offset_index",
    "visual_index",
    "is_visual",
}

_METRIC_SCOPE = re.compile(
    r"^[a-z][a-z0-9]*_(?:swapped_vs_(?:recipient|donor)|"
    r"swapped|anchor_copy|oracle|degradation)_.+$"
)
_CROSS_PROTOCOLS = {"id_cd", "ood_cd"}
_MAX_BOOTSTRAP_INDEX_ELEMENTS = 2_000_000
_NONFINITE_KINDS = {"positive_infinity", "negative_infinity", "nan"}
_PAIR_MATCH_DIAGNOSTICS = (
    "match_cost",
    "match_translation_waypoint_l2",
    "match_yaw_degrees",
)
_MODALITIES = ("rgb", "feature", "action")
_RGB_SCOPES = ("anchor_copy", "oracle", "swapped", "degradation")
_RGB_METRICS = ("mse", "mae", "psnr", "ssim", "lpips")
_FEATURE_METRICS = ("mse", "mae", "cosine_similarity")
_ACTION_SCOPES = (
    "swapped_vs_donor",
    "swapped_vs_recipient",
    "oracle",
    "degradation",
)
_ACTION_METRICS = (
    "normalized_mse",
    "normalized_mae",
    "translation_epe_waypoints",
    "translation_epe_meters",
    "yaw_mae_degrees",
    "yaw_rmse_degrees",
)
STANDARD_METRICS_BY_MODALITY = {
    "rgb": tuple(
        f"rgb_{scope}_{metric}" for scope in _RGB_SCOPES for metric in _RGB_METRICS
    ),
    "feature": tuple(
        f"feature_{scope}_{metric}"
        for scope in _RGB_SCOPES
        for metric in _FEATURE_METRICS
    ),
    "action": tuple(
        f"action_{scope}_{metric}"
        for scope in _ACTION_SCOPES
        for metric in _ACTION_METRICS
        if not (scope == "degradation" and metric == "yaw_rmse_degrees")
    ),
}


def _seed_for(base_seed: int, *parts: object) -> int:
    payload = "\0".join([str(base_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def _is_cross_protocol(protocol: str) -> bool:
    return protocol in _CROSS_PROTOCOLS or protocol.endswith("_cd")


def _metric_aggregation(metric: str) -> str:
    return (
        "root_mean_square"
        if metric.endswith("_yaw_rmse_degrees")
        else "arithmetic_mean"
    )


def metric_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Discover metrics via the public ``modality_scope_name`` convention.

    Name-based discovery prevents numeric anchor indices, waypoint spacing and
    match diagnostics from silently contaminating aggregate tables. A future
    modality works unchanged when it uses one of the standard metric scopes.
    """

    keys: set[str] = set()
    for row in rows:
        for raw_key in row:
            key = str(raw_key)
            if key not in IDENTITY_COLUMNS and _METRIC_SCOPE.fullmatch(key):
                keys.add(key)
    return sorted(keys)


def _metric_modality(metric: str) -> str:
    for modality in _MODALITIES:
        if metric.startswith(f"{modality}_"):
            return modality
    return "other"


def _condition_capabilities(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    observed = metric_columns(rows)
    output: dict[str, dict[str, Any]] = {}
    for modality in _MODALITIES:
        modality_metrics = [
            metric for metric in observed if _metric_modality(metric) == modality
        ]
        raw_statuses = {
            str(row[f"{modality}_status"]).strip()
            for row in rows
            if row.get(f"{modality}_status") is not None
            and str(row[f"{modality}_status"]).strip()
        }
        if modality_metrics and raw_statuses:
            raise ValueError(
                f"condition mixes {modality} metrics with unavailable status values"
            )
        if len(raw_statuses) > 1:
            raise ValueError(
                f"condition contains inconsistent {modality}_status values: "
                f"{sorted(raw_statuses)}"
            )
        if modality_metrics:
            output[modality] = {
                "available": True,
                "status": "available",
                "metrics": modality_metrics,
            }
        else:
            status = next(iter(raw_statuses), None)
            output[modality] = {
                "available": False,
                "status": status or f"N/A: no {modality} metrics or status recorded",
                "metrics": [],
            }
    return output


def _metric_observation(row: Mapping[str, Any], metric: str) -> Any:
    value = row.get(metric)
    if value is not None:
        return value
    diagnostics = row.get("metric_nonfinite")
    if isinstance(diagnostics, Mapping):
        kind = diagnostics.get(metric)
        if kind in _NONFINITE_KINDS:
            return kind
    return value


def _numeric_kind(value: Any) -> tuple[str, float | None]:
    if isinstance(value, str) and value in _NONFINITE_KINDS:
        return value, None
    if value is None or isinstance(value, (bool, np.bool_)):
        return "missing", None
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return "missing", None
        value = value.item()
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return "missing", None
    if math.isnan(converted):
        return "nan", None
    if converted == math.inf:
        return "positive_infinity", None
    if converted == -math.inf:
        return "negative_infinity", None
    return "finite", converted


def _numeric_profile(values: Iterable[Any]) -> tuple[np.ndarray, dict[str, int]]:
    output: list[float] = []
    counts = {
        "sample_count": 0,
        "finite_count": 0,
        "nonfinite_count": 0,
        "positive_infinity_count": 0,
        "negative_infinity_count": 0,
        "nan_count": 0,
        "missing_count": 0,
    }
    for value in values:
        counts["sample_count"] += 1
        kind, converted = _numeric_kind(value)
        if kind == "finite":
            counts["finite_count"] += 1
            if converted is None:  # pragma: no cover - guarded by _numeric_kind.
                raise AssertionError("finite numeric observation has no value")
            output.append(converted)
        elif kind == "missing":
            counts["missing_count"] += 1
        else:
            counts["nonfinite_count"] += 1
            counts[f"{kind}_count"] += 1
    return np.asarray(output, dtype=np.float64), counts


def _finite(values: Iterable[Any]) -> np.ndarray:
    return _numeric_profile(values)[0]


def _nonfinite_status(counts: Mapping[str, int]) -> str:
    positive = int(counts.get("positive_infinity_count", 0))
    negative = int(counts.get("negative_infinity_count", 0))
    nan_count = int(counts.get("nan_count", 0))
    if positive and not negative and not nan_count:
        return "positive_infinity"
    if negative and not positive and not nan_count:
        return "negative_infinity"
    if positive or negative:
        return "mixed_nonfinite"
    if nan_count:
        return "nonfinite_values"
    return "ok"


def _validate_bootstrap(samples: int, confidence: float) -> None:
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
        raise ValueError("bootstrap samples must be a non-negative integer")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("bootstrap confidence must be in (0, 1)")


def _bootstrap_draw_means(
    array: np.ndarray, samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ordinary bootstrap means with bounded temporary index memory."""

    count = int(array.size)
    means = np.empty(samples, dtype=np.float64)
    if samples == 0:
        return means
    if count <= _MAX_BOOTSTRAP_INDEX_ELEMENTS:
        draw_chunk = max(
            1,
            min(
                samples,
                4096,
                _MAX_BOOTSTRAP_INDEX_ELEMENTS // max(1, count),
            ),
        )
        for start in range(0, samples, draw_chunk):
            stop = min(samples, start + draw_chunk)
            indices = rng.integers(0, count, size=(stop - start, count))
            means[start:stop] = array[indices].mean(axis=1)
        return means

    # Even one [draw, observation] index array would exceed the memory budget.
    for draw in range(samples):
        total = 0.0
        remaining = count
        while remaining:
            chunk = min(remaining, _MAX_BOOTSTRAP_INDEX_ELEMENTS)
            indices = rng.integers(0, count, size=chunk)
            total += float(array[indices].sum(dtype=np.float64))
            remaining -= chunk
        means[draw] = total / count
    return means


def _bootstrap_draw_root_mean_squares(
    array: np.ndarray, samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap RMS values without materializing all squared observations."""

    count = int(array.size)
    values = np.empty(samples, dtype=np.float64)
    if samples == 0:
        return values
    if count <= _MAX_BOOTSTRAP_INDEX_ELEMENTS:
        draw_chunk = max(
            1,
            min(
                samples,
                4096,
                _MAX_BOOTSTRAP_INDEX_ELEMENTS // max(1, count),
            ),
        )
        for start in range(0, samples, draw_chunk):
            stop = min(samples, start + draw_chunk)
            indices = rng.integers(0, count, size=(stop - start, count))
            selected = array[indices]
            values[start:stop] = np.sqrt(np.square(selected).mean(axis=1))
        return values

    for draw in range(samples):
        squared_total = 0.0
        remaining = count
        while remaining:
            chunk = min(remaining, _MAX_BOOTSTRAP_INDEX_ELEMENTS)
            indices = rng.integers(0, count, size=chunk)
            squared_total += float(np.square(array[indices]).sum(dtype=np.float64))
            remaining -= chunk
        values[draw] = math.sqrt(squared_total / count)
    return values


def _ci_from_draws(
    draws: np.ndarray, confidence: float, fallback: float
) -> tuple[float, float]:
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return fallback, fallback
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(finite, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    _validate_bootstrap(samples, confidence)
    array = _finite(values)
    if array.size == 0:
        return math.nan, math.nan
    observed = float(array.mean())
    if array.size == 1 or samples == 0:
        return observed, observed
    rng = np.random.default_rng(seed)
    return _ci_from_draws(
        _bootstrap_draw_means(array, samples, rng), confidence, observed
    )


def describe_values(
    values: Sequence[float] | np.ndarray,
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    aggregation: str = "arithmetic_mean",
) -> dict[str, Any]:
    if aggregation not in {"arithmetic_mean", "root_mean_square"}:
        raise ValueError(f"unsupported aggregation {aggregation!r}")
    array, counts = _numeric_profile(values)
    base: dict[str, Any] = {
        # ``count`` remains a compatibility alias for finite_count.
        "count": counts["finite_count"],
        **counts,
    }
    if aggregation == "root_mean_square":
        base["aggregation"] = aggregation
    if counts["nonfinite_count"]:
        return {
            **base,
            "status": _nonfinite_status(counts),
            "mean": None,
            "std": None,
            "median": None,
            "ci_lower": None,
            "ci_upper": None,
        }
    if array.size == 0:
        return {
            **base,
            "status": "no_values",
            "mean": None,
            "std": None,
            "median": None,
            "ci_lower": None,
            "ci_upper": None,
        }
    if aggregation == "root_mean_square":
        observed = float(np.sqrt(np.square(array).mean()))
        _validate_bootstrap(bootstrap_samples, confidence)
        draws = _bootstrap_draw_root_mean_squares(
            array, bootstrap_samples, np.random.default_rng(seed)
        )
        lower, upper = _ci_from_draws(draws, confidence, observed)
    else:
        observed = float(array.mean())
        lower, upper = bootstrap_mean_ci(
            array,
            samples=bootstrap_samples,
            confidence=confidence,
            seed=seed,
        )
    return {
        **base,
        "status": "partial_missing" if counts["missing_count"] else "ok",
        "mean": observed,
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "ci_lower": lower,
        "ci_upper": upper,
    }


def _source_offset(row: Mapping[str, Any]) -> int:
    value = row.get("source_offset", row.get("offset"))
    if isinstance(value, bool):
        raise TypeError("source_offset must be an integer, not bool")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid source_offset {value!r}") from exc
    if isinstance(value, (float, np.floating)) and float(value) != converted:
        raise ValueError(f"source_offset must be integral, got {value!r}")
    return converted


def _cluster_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    if row.get("source_window_id") is not None:
        return ("source_window_id", str(row["source_window_id"]))
    required = (
        row.get("source_dataset"),
        row.get("source_split"),
        row.get("source_trajectory"),
        row.get("source_anchor", row.get("anchor")),
    )
    if required[2] is not None and required[3] is not None:
        return ("anchor_window", *required)
    if row.get("window_index") is not None:
        return ("window_index", row.get("window_index"))
    if row.get("record_id") is not None:
        return ("record_id", str(row["record_id"]))
    raise ValueError(
        "direct bootstrap requires source_window_id, source trajectory/anchor, "
        "window_index, or record_id"
    )


def _direct_cluster_draws(
    rows: Sequence[Mapping[str, Any]],
    offsets: Sequence[int],
    metric: str,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    aggregation = _metric_aggregation(metric)
    values: dict[tuple[tuple[Any, ...], int], list[float]] = defaultdict(list)
    clusters: set[tuple[Any, ...]] = set()
    allowed = set(offsets)
    for row in rows:
        offset = _source_offset(row)
        if offset not in allowed:
            continue
        cluster = _cluster_key(row)
        clusters.add(cluster)
        finite = _finite([_metric_observation(row, metric)])
        if finite.size:
            values[(cluster, offset)].append(float(finite[0]))
    ordered_clusters = sorted(clusters, key=repr)
    active_offsets = [
        offset
        for offset in offsets
        if any(values.get((cluster, offset)) for cluster in ordered_clusters)
    ]
    if not ordered_clusters or not active_offsets or samples == 0:
        return np.empty(0, dtype=np.float64)
    matrix = np.full(
        (len(ordered_clusters), len(active_offsets)), np.nan, dtype=np.float64
    )
    for cluster_index, cluster in enumerate(ordered_clusters):
        for offset_index, offset in enumerate(active_offsets):
            cell = values.get((cluster, offset))
            if cell:
                cell_array = np.asarray(cell, dtype=np.float64)
                matrix[cluster_index, offset_index] = float(
                    np.square(cell_array).mean()
                    if aggregation == "root_mean_square"
                    else cell_array.mean()
                )

    rng = np.random.default_rng(seed)
    cluster_count = len(ordered_clusters)
    draw_chunk = max(
        1,
        min(
            samples,
            4096,
            _MAX_BOOTSTRAP_INDEX_ELEMENTS // max(1, cluster_count),
        ),
    )
    output = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, draw_chunk):
        stop = min(samples, start + draw_chunk)
        indices = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
        macro_sum = np.zeros(stop - start, dtype=np.float64)
        macro_count = np.zeros(stop - start, dtype=np.int64)
        for offset_index in range(len(active_offsets)):
            selected = matrix[indices, offset_index]
            finite = np.isfinite(selected)
            count = finite.sum(axis=1)
            mean = np.divide(
                np.where(finite, selected, 0.0).sum(axis=1),
                count,
                out=np.full(stop - start, np.nan),
                where=count > 0,
            )
            if aggregation == "root_mean_square":
                mean = np.sqrt(mean)
            valid = np.isfinite(mean)
            macro_sum[valid] += mean[valid]
            macro_count[valid] += 1
        output[start:stop] = np.divide(
            macro_sum,
            macro_count,
            out=np.full(stop - start, np.nan),
            where=macro_count > 0,
        )
    return output


def _stratified_offset_draws(
    rows: Sequence[Mapping[str, Any]],
    offsets: Sequence[int],
    metric: str,
    *,
    samples: int,
    seed: int,
) -> np.ndarray:
    if samples == 0:
        return np.empty(0, dtype=np.float64)
    arrays = [
        _finite(
            _metric_observation(row, metric)
            for row in rows
            if _source_offset(row) == offset
        )
        for offset in offsets
    ]
    arrays = [array for array in arrays if array.size]
    if not arrays:
        return np.empty(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    accumulated = np.zeros(samples, dtype=np.float64)
    for array in arrays:
        if _metric_aggregation(metric) == "root_mean_square":
            accumulated += _bootstrap_draw_root_mean_squares(array, samples, rng)
        else:
            accumulated += _bootstrap_draw_means(array, samples, rng)
    return accumulated / len(arrays)


def _macro_summary(
    rows: Sequence[Mapping[str, Any]],
    offset_summaries: Mapping[str, Any],
    offsets: Sequence[int],
    metric: str,
    *,
    protocol: str,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    selected = [offset_summaries[str(offset)][metric] for offset in offsets]
    offset_means = _finite(item["mean"] for item in selected)
    counts = {
        key: sum(int(item.get(key, 0)) for item in selected)
        for key in (
            "sample_count",
            "finite_count",
            "nonfinite_count",
            "positive_infinity_count",
            "negative_infinity_count",
            "nan_count",
            "missing_count",
        )
    }
    semantic_offset_count = sum(
        int(item.get("finite_count", 0)) + int(item.get("nonfinite_count", 0)) > 0
        for item in selected
    )
    finite_offset_count = sum(int(item.get("finite_count", 0)) > 0 for item in selected)
    nonfinite_offset_count = sum(
        int(item.get("nonfinite_count", 0)) > 0 for item in selected
    )
    method = (
        "offset_stratified_pair"
        if _is_cross_protocol(protocol)
        else "anchor_window_cluster"
    )
    base = {
        # Compatibility alias: count is the number of finite pair values.
        "count": counts["finite_count"],
        **counts,
        "configured_offset_count": len(offsets),
        "offset_count": semantic_offset_count,
        "finite_offset_count": finite_offset_count,
        "nonfinite_offset_count": nonfinite_offset_count,
        "bootstrap_unit": method,
    }
    if _metric_aggregation(metric) == "root_mean_square":
        base["aggregation"] = "root_mean_square"
    if counts["nonfinite_count"]:
        return {
            **base,
            "status": _nonfinite_status(counts),
            "macro_mean": None,
            "macro_std": None,
            "macro_median": None,
            "ci_lower": None,
            "ci_upper": None,
        }
    if semantic_offset_count != len(offsets):
        return {
            **base,
            "status": "incomplete_offsets" if semantic_offset_count else "no_values",
            "macro_mean": None,
            "macro_std": None,
            "macro_median": None,
            "ci_lower": None,
            "ci_upper": None,
        }
    observed = float(offset_means.mean())
    if _is_cross_protocol(protocol):
        draws = _stratified_offset_draws(
            rows, offsets, metric, samples=bootstrap_samples, seed=seed
        )
    else:
        draws = _direct_cluster_draws(
            rows, offsets, metric, samples=bootstrap_samples, seed=seed
        )
    lower, upper = _ci_from_draws(draws, confidence, observed)
    return {
        **base,
        "status": "partial_missing" if counts["missing_count"] else "ok",
        "macro_mean": observed,
        "macro_std": (
            float(offset_means.std(ddof=1)) if offset_means.size > 1 else 0.0
        ),
        "macro_median": float(np.median(offset_means)),
        "ci_lower": lower,
        "ci_upper": upper,
    }


def _condition_stats_by_id(
    raw: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> dict[str, Mapping[str, Any]]:
    if raw is None:
        return {}
    entries: Any = raw
    if isinstance(raw, Mapping):
        if "conditions" in raw:
            entries = raw["conditions"]
        elif "condition_id" in raw:
            entries = [raw]
        else:
            result: dict[str, Mapping[str, Any]] = {}
            for condition_id, value in raw.items():
                if not isinstance(value, Mapping):
                    continue
                result[str(condition_id)] = value
            return result
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError("condition_stats must be a condition mapping or sequence")
    result = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise TypeError(f"condition_stats entry {index} is not a mapping")
        condition_id = entry.get("condition_id")
        if not isinstance(condition_id, str) or not condition_id:
            raise ValueError(f"condition_stats entry {index} has no condition_id")
        if condition_id in result:
            raise ValueError(f"condition_stats contains duplicate {condition_id!r}")
        result[condition_id] = entry
    return result


def _matching_manifest_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    offsets: Sequence[int],
    stats: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_windows = {
        (
            row.get("source_dataset"),
            row.get("source_split"),
            row.get("source_trajectory"),
            row.get("source_anchor", row.get("anchor")),
        )
        for row in rows
    }
    source_trajectories = {
        str(row["source_trajectory"])
        for row in rows
        if row.get("source_trajectory") is not None
    }
    recipient_trajectories = {
        str(row.get("recipient_trajectory", row.get("target_trajectory")))
        for row in rows
        if row.get("recipient_trajectory", row.get("target_trajectory")) is not None
    }
    recipient_pairs_by_offset: dict[int, set[tuple[Any, ...]]] = defaultdict(set)
    all_recipient_pairs: set[tuple[Any, ...]] = set()
    for row in rows:
        pair = (
            row.get("recipient_dataset", row.get("target_dataset")),
            row.get("recipient_split", row.get("target_split")),
            row.get("recipient_trajectory", row.get("target_trajectory")),
            row.get("recipient_anchor", row.get("target_anchor")),
            row.get("recipient_target", row.get("target_target")),
        )
        recipient_pairs_by_offset[_source_offset(row)].add(pair)
        all_recipient_pairs.add(pair)
    derived = {
        "unique_source_windows": len(source_windows),
        "unique_source_trajectories": len(source_trajectories),
        "unique_recipient_trajectories": len(recipient_trajectories),
        "unique_recipient_pairs": sum(
            len(recipient_pairs_by_offset[offset]) for offset in offsets
        ),
        "unique_recipient_pairs_across_offsets": len(all_recipient_pairs),
        "recipient_pair_cross_offset_reuse_count": (
            sum(len(recipient_pairs_by_offset[offset]) for offset in offsets)
            - len(all_recipient_pairs)
        ),
    }
    overall = dict(derived)
    sidecar_offsets: Mapping[str, Any] = {}
    status = "unavailable"
    if stats is not None:
        status = "ok"
        if isinstance(stats.get("offsets"), Mapping):
            sidecar_offsets = stats["offsets"]
    offset_output: dict[str, Any] = {}
    for offset in offsets:
        sidecar = sidecar_offsets.get(str(offset), {})
        if not isinstance(sidecar, Mapping):
            sidecar = {}
        offset_output[str(offset)] = {
            "status": "ok" if sidecar else "unavailable",
            "candidate_coverage": sidecar.get("candidate_coverage"),
            "no_match_rejection_rate": sidecar.get(
                "no_match_rejection_rate", sidecar.get("rejection_rate")
            ),
            "rejection_rate": sidecar.get("rejection_rate"),
            "donor_unselected_rate": sidecar.get("donor_unselected_rate"),
            "unique_recipient_pairs": len(recipient_pairs_by_offset[offset]),
        }
    return {"status": status, "overall": overall, "offsets": offset_output}


def _cross_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    offsets: Sequence[int],
    primary_offsets: Sequence[int],
    protocol: str,
    condition_id: str,
    stats: Mapping[str, Any] | None,
    *,
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    pair: dict[str, Any] = {}
    for diagnostic in _PAIR_MATCH_DIAGNOSTICS:
        if not any(diagnostic in row for row in rows):
            continue
        offset_summaries = {
            str(offset): describe_values(
                [row.get(diagnostic) for row in rows if _source_offset(row) == offset],
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
                seed=_seed_for(seed, condition_id, "diagnostic", offset, diagnostic),
            )
            for offset in offsets
        }
        pair[diagnostic] = {
            "offsets": offset_summaries,
            "primary": _macro_summary(
                rows,
                {key: {diagnostic: value} for key, value in offset_summaries.items()},
                primary_offsets,
                diagnostic,
                protocol=protocol,
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
                seed=_seed_for(seed, condition_id, "diagnostic_primary", diagnostic),
            ),
            "secondary_all_offsets": _macro_summary(
                rows,
                {key: {diagnostic: value} for key, value in offset_summaries.items()},
                offsets,
                diagnostic,
                protocol=protocol,
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
                seed=_seed_for(seed, condition_id, "diagnostic_secondary", diagnostic),
            ),
        }
    return {
        "pair_gaps": pair,
        "manifest_matching": _matching_manifest_diagnostics(rows, offsets, stats),
    }


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    offsets: Sequence[int],
    bootstrap_samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 32,
    primary_excludes_zero: bool = True,
    condition_stats: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate rows using cluster bootstrap for direct protocols and
    offset-stratified bootstrap for cross-domain protocols."""

    _validate_bootstrap(bootstrap_samples, confidence)
    normalized_offsets = tuple(int(offset) for offset in offsets)
    if not normalized_offsets or len(set(normalized_offsets)) != len(
        normalized_offsets
    ):
        raise ValueError("offsets must be a non-empty sequence of unique integers")
    metrics = metric_columns(rows)
    stats_by_condition = _condition_stats_by_id(condition_stats)
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    allowed_offsets = set(normalized_offsets)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"row {index} is not a mapping")
        if not str(row.get("condition_id", "")).strip():
            raise ValueError(f"row {index} has no condition_id")
        offset = _source_offset(row)
        if offset not in allowed_offsets:
            raise ValueError(
                f"row {index} source_offset={offset} is not in configured offsets"
            )
        by_condition[str(row["condition_id"])].append(row)

    primary_offsets = (
        tuple(offset for offset in normalized_offsets if offset != 0)
        if primary_excludes_zero
        else normalized_offsets
    )
    conditions: dict[str, Any] = {}
    for condition_id in sorted(by_condition):
        condition_rows = by_condition[condition_id]
        first = condition_rows[0]
        protocol = str(first.get("protocol", ""))
        if any(str(row.get("protocol", "")) != protocol for row in condition_rows):
            raise ValueError(f"condition {condition_id!r} mixes protocols")
        offset_summaries: dict[str, Any] = {}
        for offset in normalized_offsets:
            selected = [row for row in condition_rows if _source_offset(row) == offset]
            offset_summaries[str(offset)] = {
                metric: describe_values(
                    [_metric_observation(row, metric) for row in selected],
                    bootstrap_samples=bootstrap_samples,
                    confidence=confidence,
                    seed=_seed_for(seed, condition_id, "offset", offset, metric),
                    aggregation=_metric_aggregation(metric),
                )
                for metric in metrics
            }

        recipient_dataset = first.get("recipient_dataset", first.get("target_dataset"))
        recipient_split = first.get("recipient_split", first.get("target_split"))
        source_trajectories = {
            str(row["source_trajectory"])
            for row in condition_rows
            if row.get("source_trajectory") is not None
        }
        recipient_trajectories = {
            str(row.get("recipient_trajectory", row.get("target_trajectory")))
            for row in condition_rows
            if row.get("recipient_trajectory", row.get("target_trajectory")) is not None
        }
        conditions[condition_id] = {
            "identity": {
                "protocol": protocol,
                "source_dataset": first.get("source_dataset"),
                "source_split": first.get("source_split"),
                "recipient_dataset": recipient_dataset,
                "recipient_split": recipient_split,
            },
            "row_count": len(condition_rows),
            "visual_row_count": sum(
                bool(row.get("is_visual")) for row in condition_rows
            ),
            "unique_source_trajectories": len(source_trajectories),
            "unique_recipient_trajectories": len(recipient_trajectories),
            "capabilities": _condition_capabilities(condition_rows),
            "offsets": offset_summaries,
            "primary": {
                metric: _macro_summary(
                    condition_rows,
                    offset_summaries,
                    primary_offsets,
                    metric,
                    protocol=protocol,
                    bootstrap_samples=bootstrap_samples,
                    confidence=confidence,
                    seed=_seed_for(seed, condition_id, "primary", metric),
                )
                for metric in metrics
            },
            "secondary_all_offsets": {
                metric: _macro_summary(
                    condition_rows,
                    offset_summaries,
                    normalized_offsets,
                    metric,
                    protocol=protocol,
                    bootstrap_samples=bootstrap_samples,
                    confidence=confidence,
                    seed=_seed_for(seed, condition_id, "secondary_all_offsets", metric),
                )
                for metric in metrics
            },
        }
        if _is_cross_protocol(protocol):
            conditions[condition_id]["cross_diagnostics"] = _cross_diagnostics(
                condition_rows,
                normalized_offsets,
                primary_offsets,
                protocol,
                condition_id,
                stats_by_condition.get(condition_id),
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
                seed=seed,
            )
    return {
        "schema_version": 1,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_confidence": confidence,
        "primary_offsets": list(primary_offsets),
        "secondary_offsets": list(normalized_offsets),
        "metrics": metrics,
        "standard_metrics_by_modality": {
            modality: list(STANDARD_METRICS_BY_MODALITY[modality])
            for modality in _MODALITIES
        },
        "conditions": conditions,
    }


def _summary_long_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition_id in sorted(summary.get("conditions", {})):
        condition = summary["conditions"][condition_id]
        identity = condition["identity"]
        capabilities = condition.get("capabilities", {})
        if isinstance(capabilities, Mapping):
            for modality in _MODALITIES:
                capability = capabilities.get(modality)
                if not isinstance(capability, Mapping):
                    continue
                output.append(
                    {
                        "condition_id": condition_id,
                        **identity,
                        "entry_type": "capability",
                        "scope": "capability",
                        "offset": None,
                        "metric": modality,
                        "available": capability.get("available"),
                        "status": capability.get("status", "N/A"),
                        "mean": None,
                    }
                )
        for offset in summary.get("secondary_offsets", ()):
            metric_map = condition.get("offsets", {}).get(str(offset), {})
            for metric in summary.get("metrics", ()):
                values = metric_map.get(metric)
                if values is not None:
                    output.append(
                        {
                            "condition_id": condition_id,
                            **identity,
                            "entry_type": "metric",
                            "scope": "offset",
                            "offset": int(offset),
                            "metric": metric,
                            **values,
                        }
                    )
        for scope in ("primary", "secondary_all_offsets"):
            for metric in summary.get("metrics", ()):
                values = condition.get(scope, {}).get(metric)
                if values is not None:
                    output.append(
                        {
                            "condition_id": condition_id,
                            **identity,
                            "entry_type": "metric",
                            "scope": scope,
                            "offset": None,
                            "metric": metric,
                            **values,
                        }
                    )
        diagnostics = condition.get("cross_diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        pair_gaps = diagnostics.get("pair_gaps", {})
        if isinstance(pair_gaps, Mapping):
            for diagnostic, values in pair_gaps.items():
                if not isinstance(values, Mapping):
                    continue
                diagnostic_offsets = values.get("offsets", {})
                for offset in summary.get("secondary_offsets", ()):
                    offset_values = (
                        diagnostic_offsets.get(str(offset))
                        if isinstance(diagnostic_offsets, Mapping)
                        else None
                    )
                    if isinstance(offset_values, Mapping):
                        output.append(
                            {
                                "condition_id": condition_id,
                                **identity,
                                "entry_type": "diagnostic_pair_gap",
                                "scope": "offset",
                                "offset": int(offset),
                                "metric": str(diagnostic),
                                **offset_values,
                            }
                        )
                for scope in ("primary", "secondary_all_offsets"):
                    scope_values = values.get(scope)
                    if isinstance(scope_values, Mapping):
                        output.append(
                            {
                                "condition_id": condition_id,
                                **identity,
                                "entry_type": "diagnostic_pair_gap",
                                "scope": scope,
                                "offset": None,
                                "metric": str(diagnostic),
                                **scope_values,
                            }
                        )
        matching = diagnostics.get("manifest_matching", {})
        if not isinstance(matching, Mapping):
            continue
        matching_offsets = matching.get("offsets", {})
        if isinstance(matching_offsets, Mapping):
            for offset in summary.get("secondary_offsets", ()):
                values = matching_offsets.get(str(offset), {})
                if not isinstance(values, Mapping):
                    continue
                for diagnostic in (
                    "candidate_coverage",
                    "no_match_rejection_rate",
                    "rejection_rate",
                    "donor_unselected_rate",
                    "unique_recipient_pairs",
                ):
                    value = values.get(diagnostic)
                    output.append(
                        {
                            "condition_id": condition_id,
                            **identity,
                            "entry_type": "diagnostic_manifest",
                            "scope": "offset",
                            "offset": int(offset),
                            "metric": diagnostic,
                            "status": values.get("status", "unavailable"),
                            "mean": value,
                            "std": None,
                            "median": value,
                            "ci_lower": None,
                            "ci_upper": None,
                        }
                    )
        overall = matching.get("overall", {})
        if isinstance(overall, Mapping):
            for diagnostic, value in overall.items():
                output.append(
                    {
                        "condition_id": condition_id,
                        **identity,
                        "entry_type": "diagnostic_manifest",
                        "scope": "condition",
                        "offset": None,
                        "metric": str(diagnostic),
                        "status": matching.get("status", "unavailable"),
                        "mean": value,
                        "std": None,
                        "median": value,
                        "ci_lower": None,
                        "ci_upper": None,
                    }
                )
    return output


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, np.floating)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (RuntimeError, TypeError, ValueError):
            pass
    return str(value)


def _result_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a strict-JSON row while preserving metric nonfinite semantics."""

    diagnostics: dict[str, str] = {}
    existing = row.get("metric_nonfinite")
    if isinstance(existing, Mapping):
        diagnostics.update(
            {
                str(metric): str(kind)
                for metric, kind in existing.items()
                if kind in _NONFINITE_KINDS
            }
        )
    for metric in metric_columns([row]):
        kind, _ = _numeric_kind(row.get(metric))
        if kind in _NONFINITE_KINDS:
            diagnostics[metric] = kind
    result = _json_safe(dict(row))
    if not isinstance(result, dict):  # pragma: no cover - dict input is guaranteed.
        raise TypeError("serialized result row is not a dictionary")
    result["metric_nonfinite"] = dict(sorted(diagnostics.items()))
    return result


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(
                    json.dumps(
                        _result_row(row),
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def _tabular_safe(value: Any) -> Any:
    safe = _json_safe(value)
    if isinstance(safe, (dict, list)):
        return json.dumps(
            safe, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
    return "" if safe is None else safe


def _tabular_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {str(key): _tabular_safe(value) for key, value in row.items()} for row in rows
    ]


def _parquet_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Preserve numeric/null columns while flattening nested JSON values."""

    output: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            safe = _json_safe(value)
            converted[str(key)] = (
                json.dumps(
                    safe,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if isinstance(safe, (dict, list))
                else safe
            )
        output.append(converted)
    return output


def _record_report_failure(output: Path, message: str) -> None:
    """Make a previous or partial run status unambiguously incomplete."""

    status_path = output / "run_status.json"
    status: dict[str, Any] = {}
    if status_path.is_file():
        try:
            loaded = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, Mapping):
            status.update(loaded)
    status.setdefault("schema_version", 1)
    status.update(
        {
            "complete": False,
            "selection_complete": False,
            "full_manifest_complete": False,
            "report_error": message,
        }
    )
    atomic_write_json(status_path, _json_safe(status))


def _heatmap_records(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in _summary_long_rows(summary):
        scope = str(row["scope"])
        if scope == "offset":
            scope = f"offset:{int(row['offset']):+d}"
            value = row.get("mean")
        else:
            value = row.get("macro_mean")
        records.append(
            {
                "protocol": str(row.get("protocol") or "unknown"),
                "scope": scope,
                "metric": str(row["metric"]),
                "source": (f"{row.get('source_dataset')}/{row.get('source_split')}"),
                "recipient": (
                    f"{row.get('recipient_dataset', row.get('target_dataset'))}/"
                    f"{row.get('recipient_split', row.get('target_split'))}"
                ),
                "condition_id": str(row["condition_id"]),
                "value": _json_safe(value),
                "status": str(row.get("status") or "unknown"),
            }
        )
    return records


def _script_json(value: Any) -> str:
    """Serialize JSON safely inside a script element."""

    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def write_heatmap(output_dir: str | Path, summary: Mapping[str, Any]) -> dict[str, str]:
    """Write a dependency-free, filterable source-to-recipient heatmap."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = _heatmap_records(_json_safe(summary))
    data_path = atomic_write_json(output / "heatmap.json", records)
    payload = _script_json(records)
    html_text = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Navigation LAM metric heatmap</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#17202a;background:#f7f8fa}}
.controls{{position:sticky;top:0;z-index:1;display:flex;gap:16px;flex-wrap:wrap;background:white;padding:14px;border:1px solid #d8dee6}}
label{{display:grid;gap:4px;font-size:13px;font-weight:600}}select{{min-width:180px;padding:6px}}
.panel{{margin-top:18px;background:white;padding:16px;border:1px solid #d8dee6;overflow:auto}}
table{{border-collapse:collapse;min-width:720px}}th,td{{border:1px solid #c9d1da;padding:8px;text-align:center;white-space:nowrap}}
th{{background:#eef2f6}}th:first-child{{position:sticky;left:0;z-index:1}}td:first-child{{position:sticky;left:0;background:#f8fafc;font-weight:600}}
td.metric{{min-width:105px;font-variant-numeric:tabular-nums}}td.na{{background:#eceff1;color:#687078}}
.legend{{margin:10px 0;font-size:13px;color:#53606d}}#message{{padding:12px;color:#8a2830}}
</style></head><body>
<h1>Navigation LAM metric heatmap</h1>
<div class="controls">
<label>Protocol<select id="protocol"></select></label>
<label>Scope / source offset<select id="scope"></select></label>
<label>Metric<select id="metric"></select></label>
</div>
<div class="panel"><h2 id="heading"></h2>
<div class="legend">Sequential color shows the selected matrix's numeric minimum (light) to maximum (dark); it does not imply whether higher or lower is better.</div>
<div id="message" hidden></div><table id="matrix"></table></div>
<script>
const DATA={payload};
const protocol=document.getElementById('protocol'),scope=document.getElementById('scope'),metric=document.getElementById('metric');
const unique=(key)=>[...new Set(DATA.map(x=>x[key]))].sort((a,b)=>a.localeCompare(b,undefined,{{numeric:true}}));
const fill=(select,values)=>{{for(const value of values){{const option=document.createElement('option');option.value=value;option.textContent=value;select.append(option)}}}};
fill(protocol,unique('protocol'));fill(scope,unique('scope'));fill(metric,unique('metric'));
const color=(value,min,max)=>{{if(value===null)return null;const t=max===min?0.55:(value-min)/(max-min);const light=96-52*t;return `hsl(194 68% ${{light}}%)`}};
function render(){{
 const selected=DATA.filter(x=>x.protocol===protocol.value&&x.scope===scope.value&&x.metric===metric.value);
 const sources=[...new Set(selected.map(x=>x.source))].sort(),recipients=[...new Set(selected.map(x=>x.recipient))].sort();
 const table=document.getElementById('matrix'),message=document.getElementById('message');table.replaceChildren();
 document.getElementById('heading').textContent=`${{protocol.value}} · ${{scope.value}} · ${{metric.value}}`;
 if(!selected.length){{message.hidden=false;message.textContent='No values exist for this filter combination.';return}}message.hidden=true;
 const lookup=new Map(selected.map(x=>[JSON.stringify([x.source,x.recipient]),x]));
 const finite=selected.map(x=>x.value).filter(Number.isFinite),min=Math.min(...finite),max=Math.max(...finite);
 const head=document.createElement('tr'),corner=document.createElement('th');corner.textContent='source ↓ / recipient →';head.append(corner);
 for(const name of recipients){{const th=document.createElement('th');th.textContent=name;head.append(th)}}table.append(head);
 for(const sourceName of sources){{const tr=document.createElement('tr'),label=document.createElement('td');label.textContent=sourceName;tr.append(label);
  for(const recipientName of recipients){{const td=document.createElement('td'),item=lookup.get(JSON.stringify([sourceName,recipientName]));td.className='metric';
   if(!item||!Number.isFinite(item.value)){{td.textContent='N/A';td.classList.add('na')}}else{{td.textContent=Number(item.value).toPrecision(6);td.style.background=color(item.value,min,max);td.title=`${{item.condition_id}}: ${{item.value}}`}}tr.append(td)}}table.append(tr)}}
}}
for(const select of [protocol,scope,metric])select.addEventListener('change',render);
if(DATA.length){{const initial=DATA.find(x=>x.scope==='primary'&&Number.isFinite(x.value))||DATA.find(x=>Number.isFinite(x.value))||DATA[0];protocol.value=initial.protocol;scope.value=initial.scope;metric.value=initial.metric;render()}}else{{document.getElementById('message').hidden=false;document.getElementById('message').textContent='No metric data available.'}}
</script></body></html>
"""
    html_path = atomic_write_text(output / "heatmap.html", html_text)
    return {"html": str(html_path), "json": str(data_path)}


def write_run_reports(
    output_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    save_parquet: bool = True,
) -> dict[str, str]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_rows = [_result_row(row) for row in rows]
    tabular_rows = _tabular_rows(result_rows)
    safe_summary = _json_safe(summary)
    paths = {
        "per_pair_jsonl": str(
            _write_jsonl_atomic(output / "per_pair.jsonl", result_rows)
        ),
        "per_pair_csv": str(write_csv(output / "per_pair.csv", tabular_rows)),
        "summary_json": str(atomic_write_json(output / "summary.json", safe_summary)),
    }
    long_rows = _summary_long_rows(safe_summary)
    paths["summary_csv"] = str(
        write_csv(output / "summary.csv", _tabular_rows(long_rows))
    )
    paths["summary_markdown"] = str(
        atomic_write_text(output / "summary.md", summary_markdown(safe_summary))
    )
    heatmap = write_heatmap(output, safe_summary)
    paths["heatmap_html"] = heatmap["html"]
    paths["heatmap_json"] = heatmap["json"]
    if save_parquet:
        temporary: Path | None = None
        try:
            import pandas as pd

            parquet_path = output / "per_pair.parquet"
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output,
                prefix=".per_pair.parquet.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            pd.DataFrame(_parquet_rows(result_rows)).to_parquet(temporary, index=False)
            os.replace(temporary, parquet_path)
            temporary = None
            paths["per_pair_parquet"] = str(parquet_path)
        except Exception as exc:
            message = (
                "save_parquet=true requires a successfully written per_pair.parquet; "
                f"writing {output / 'per_pair.parquet'} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            _record_report_failure(output, message)
            raise RuntimeError(message) from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return paths


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)):
        converted = float(value)
        return f"{converted:.6g}" if math.isfinite(converted) else "N/A"
    return str(value)


def _md_cell(value: Any) -> str:
    text = html.escape(_fmt(value), quote=False)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Navigation LAM evaluation summary",
        "",
        f"Primary offsets: `{summary.get('primary_offsets', [])}`  ",
        f"Secondary offsets: `{summary.get('secondary_offsets', [])}`",
        "",
        "| condition | protocol | source | recipient | metric | primary | primary status | all offsets | all status |",
        "|---|---|---|---|---|---:|---|---:|---|",
    ]
    for condition_id in sorted(summary.get("conditions", {})):
        condition = summary["conditions"][condition_id]
        identity = condition["identity"]
        source = f"{identity.get('source_dataset')}/{identity.get('source_split')}"
        recipient = (
            f"{identity.get('recipient_dataset', identity.get('target_dataset'))}/"
            f"{identity.get('recipient_split', identity.get('target_split'))}"
        )
        for metric in summary.get("metrics", ()):
            primary = condition.get("primary", {}).get(metric, {}).get("macro_mean")
            secondary = (
                condition.get("secondary_all_offsets", {})
                .get(metric, {})
                .get("macro_mean")
            )
            primary_status = condition.get("primary", {}).get(metric, {}).get("status")
            secondary_status = (
                condition.get("secondary_all_offsets", {}).get(metric, {}).get("status")
            )
            cells = (
                condition_id,
                identity.get("protocol"),
                source,
                recipient,
                metric,
                primary,
                primary_status,
                secondary,
                secondary_status,
            )
            lines.append("| " + " | ".join(_md_cell(cell) for cell in cells) + " |")
    capability_rows = [
        row
        for row in _summary_long_rows(summary)
        if row.get("entry_type") == "capability"
    ]
    if capability_rows:
        lines.extend(
            [
                "",
                "## Modality capabilities",
                "",
                "| condition | protocol | source | recipient | modality | available | status |",
                "|---|---|---|---|---|---:|---|",
            ]
        )
        for row in capability_rows:
            source = f"{row.get('source_dataset')}/{row.get('source_split')}"
            recipient = (
                f"{row.get('recipient_dataset', row.get('target_dataset'))}/"
                f"{row.get('recipient_split', row.get('target_split'))}"
            )
            cells = (
                row.get("condition_id"),
                row.get("protocol"),
                source,
                recipient,
                row.get("metric"),
                row.get("available"),
                row.get("status"),
            )
            lines.append("| " + " | ".join(_md_cell(cell) for cell in cells) + " |")
    diagnostic_rows = [
        row
        for row in _summary_long_rows(summary)
        if str(row.get("entry_type", "")).startswith("diagnostic_")
    ]
    if diagnostic_rows:
        lines.extend(
            [
                "",
                "## Cross-domain matching diagnostics",
                "",
                "| condition | protocol | source | recipient | type | scope | offset | diagnostic | count | mean/value | std | median | CI lower | CI upper | status |",
                "|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in diagnostic_rows:
            source = f"{row.get('source_dataset')}/{row.get('source_split')}"
            recipient = (
                f"{row.get('recipient_dataset', row.get('target_dataset'))}/"
                f"{row.get('recipient_split', row.get('target_split'))}"
            )
            cells = (
                row.get("condition_id"),
                row.get("protocol"),
                source,
                recipient,
                row.get("entry_type"),
                row.get("scope"),
                row.get("offset"),
                row.get("metric"),
                row.get("count"),
                row.get("macro_mean", row.get("mean")),
                row.get("macro_std", row.get("std")),
                row.get("macro_median", row.get("median")),
                row.get("ci_lower"),
                row.get("ci_upper"),
                row.get("status"),
            )
            lines.append("| " + " | ".join(_md_cell(cell) for cell in cells) + " |")
    return "\n".join(lines) + "\n"


def _gallery_href(path_value: Any, output: Path) -> tuple[Path, str]:
    raw = Path(str(path_value)).expanduser()
    if raw.is_absolute():
        relative = Path(os.path.relpath(raw.resolve(), output))
        stem_source = raw
    else:
        relative = Path(os.path.normpath(str(raw)))
        stem_source = raw
    return stem_source, quote(relative.as_posix(), safe="/._-~")


def _md_alt(value: str) -> str:
    return value.replace("\\", "\\\\").replace("]", "\\]").replace("\n", " ")


def write_gallery(
    output_dir: str | Path,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Write portable, relative-path HTML and CommonMark gallery indexes."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    md = ["# Evaluation visualization gallery", ""]
    cards: list[str] = []
    ordered = sorted(
        entries, key=lambda item: (str(item.get("condition_id")), str(item.get("path")))
    )
    for entry in ordered:
        if not str(entry.get("path", "")).strip():
            raise ValueError("every gallery entry must contain a non-empty path")
        stem_source, href = _gallery_href(entry["path"], output)
        title = str(entry.get("title") or entry.get("condition_id") or stem_source.stem)
        md.extend(
            [
                f"## {_md_cell(title)}",
                "",
                f"![{_md_alt(title)}](<{href}>)",
                "",
            ]
        )
        searchable = " ".join(
            [
                *(str(value) for key, value in entry.items() if key != "path"),
                href,
            ]
        )
        cards.append(
            '<article class="card" '
            f'data-search="{html.escape(searchable, quote=True)}">'
            f"<h2>{html.escape(title)}</h2>"
            f'<a href="{html.escape(href, quote=True)}">'
            f'<img loading="lazy" src="{html.escape(href, quote=True)}" '
            f'alt="{html.escape(title, quote=True)}"></a></article>'
        )
    html_text = (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Navigation LAM gallery</title>\n"
        "<style>body{font-family:sans-serif;margin:24px;background:#f6f7f9}"
        ".toolbar{position:sticky;top:0;background:#fff;padding:12px;z-index:1}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:18px}"
        ".card{background:#fff;padding:14px;border:1px solid #ddd;overflow:auto}"
        ".card img{max-width:100%;height:auto}</style>\n"
        '</head><body><div class="toolbar"><label>Filter '
        '<input id="filter" type="search"></label></div><main class="grid">\n'
        + "\n".join(cards)
        + "</main>\n<script>const q=document.getElementById('filter');"
        "q.addEventListener('input',()=>{for(const c of document.querySelectorAll('.card'))"
        "c.hidden=!c.dataset.search.toLowerCase().includes(q.value.toLowerCase())});"
        "</script>\n</body></html>\n"
    )
    return {
        "markdown": str(atomic_write_text(output / "gallery.md", "\n".join(md))),
        "html": str(atomic_write_text(output / "gallery.html", html_text)),
    }


def os_path_rel(path: Path, start: Path) -> str:
    """Backward-compatible relative-path helper."""

    return os.path.relpath(path.resolve(), start.resolve())


def _required_text(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{path}: provenance is missing non-empty {field}")
    return value


def _layout_integer_sequence(value: Any, field: str, path: Path) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{path}: summary.{field} must be an integer array")
    output: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{path}: summary.{field}[{index}] must be an integer")
        output.append(item)
    if len(output) != len(set(output)):
        raise RuntimeError(f"{path}: summary.{field} contains duplicate offsets")
    return tuple(output)


def _summary_layout(summary: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Validate and describe the report axes required for a fair comparison."""

    primary_offsets = _layout_integer_sequence(
        summary.get("primary_offsets"), "primary_offsets", path
    )
    secondary_offsets = _layout_integer_sequence(
        summary.get("secondary_offsets"), "secondary_offsets", path
    )
    if not set(primary_offsets).issubset(secondary_offsets):
        raise RuntimeError(
            f"{path}: summary.primary_offsets must be a subset of secondary_offsets"
        )

    raw_metrics = summary.get("metrics")
    if not isinstance(raw_metrics, Sequence) or isinstance(raw_metrics, (str, bytes)):
        raise TypeError(f"{path}: summary.metrics must be a string array")
    metrics = tuple(raw_metrics)
    if any(not isinstance(metric, str) or not metric for metric in metrics):
        raise TypeError(f"{path}: summary.metrics must contain non-empty strings")
    if len(metrics) != len(set(metrics)):
        raise RuntimeError(f"{path}: summary.metrics contains duplicates")

    standard_layout = summary.get("standard_metrics_by_modality")
    if not isinstance(standard_layout, Mapping):
        raise TypeError(
            f"{path}: summary.standard_metrics_by_modality must be an object"
        )
    for modality in _MODALITIES:
        expected = STANDARD_METRICS_BY_MODALITY[modality]
        actual = standard_layout.get(modality)
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
            raise TypeError(
                f"{path}: standard metric layout for {modality} must be an array"
            )
        if tuple(actual) != expected:
            raise RuntimeError(
                f"{path}: standard metric layout for {modality} is incompatible"
            )

    conditions = summary.get("conditions")
    if not isinstance(conditions, Mapping):
        raise TypeError(f"{path}: summary.conditions must be an object")
    condition_ids = tuple(sorted(str(condition_id) for condition_id in conditions))
    if len(condition_ids) != len(conditions):
        raise RuntimeError(f"{path}: summary.conditions has ambiguous condition IDs")

    expected_metric_keys = set(metrics)
    expected_offset_keys = {str(offset) for offset in secondary_offsets}
    availability: list[tuple[str, tuple[tuple[str, bool], ...]]] = []
    for condition_id in condition_ids:
        condition = conditions[condition_id]
        if not isinstance(condition, Mapping):
            raise TypeError(
                f"{path}: summary.conditions[{condition_id!r}] must be an object"
            )
        if not isinstance(condition.get("identity"), Mapping):
            raise TypeError(
                f"{path}: condition {condition_id!r} is missing identity metadata"
            )
        capabilities = condition.get("capabilities")
        if not isinstance(capabilities, Mapping) or set(capabilities) != set(
            _MODALITIES
        ):
            raise RuntimeError(
                f"{path}: condition {condition_id!r} must explicitly report "
                "rgb/feature/action capabilities"
            )
        condition_availability: list[tuple[str, bool]] = []
        for modality in _MODALITIES:
            capability = capabilities[modality]
            if not isinstance(capability, Mapping):
                raise TypeError(
                    f"{path}: condition {condition_id!r} capability {modality} "
                    "must be an object"
                )
            available = capability.get("available")
            if not isinstance(available, bool):
                raise TypeError(
                    f"{path}: condition {condition_id!r} capability {modality} "
                    "must contain a boolean available flag"
                )
            status = capability.get("status")
            if not isinstance(status, str) or not status:
                raise TypeError(
                    f"{path}: condition {condition_id!r} capability {modality} "
                    "must contain a non-empty status"
                )
            capability_metrics = capability.get("metrics")
            if not isinstance(capability_metrics, Sequence) or isinstance(
                capability_metrics, (str, bytes)
            ):
                raise TypeError(
                    f"{path}: condition {condition_id!r} capability {modality} "
                    "metrics must be an array"
                )
            if any(
                metric not in expected_metric_keys
                or _metric_modality(str(metric)) != modality
                for metric in capability_metrics
            ):
                raise RuntimeError(
                    f"{path}: condition {condition_id!r} capability {modality} "
                    "lists metrics outside summary.metrics"
                )
            if available != bool(capability_metrics):
                raise RuntimeError(
                    f"{path}: condition {condition_id!r} capability {modality} "
                    "availability disagrees with its metric list"
                )
            condition_availability.append((modality, available))
        availability.append((condition_id, tuple(condition_availability)))
        offset_map = condition.get("offsets")
        if not isinstance(offset_map, Mapping):
            raise TypeError(
                f"{path}: condition {condition_id!r}.offsets must be an object"
            )
        actual_offset_keys = {str(offset) for offset in offset_map}
        if actual_offset_keys != expected_offset_keys:
            raise RuntimeError(
                f"{path}: condition {condition_id!r} offset layout does not match "
                "summary.secondary_offsets"
            )
        for offset in secondary_offsets:
            metric_map = offset_map.get(str(offset))
            if not isinstance(metric_map, Mapping):
                raise TypeError(
                    f"{path}: condition {condition_id!r} offset {offset:+d} "
                    "must contain a metric object"
                )
            if set(metric_map) != expected_metric_keys:
                raise RuntimeError(
                    f"{path}: condition {condition_id!r} offset {offset:+d} "
                    "metric layout does not match summary.metrics"
                )
        for scope in ("primary", "secondary_all_offsets"):
            metric_map = condition.get(scope)
            if not isinstance(metric_map, Mapping):
                raise TypeError(
                    f"{path}: condition {condition_id!r}.{scope} must be an object"
                )
            if set(metric_map) != expected_metric_keys:
                raise RuntimeError(
                    f"{path}: condition {condition_id!r}.{scope} metric layout "
                    "does not match summary.metrics"
                )

    return {
        "condition_ids": condition_ids,
        "primary_offsets": primary_offsets,
        "secondary_offsets": secondary_offsets,
        "metrics": metrics,
        "metrics_by_modality": tuple(
            (
                modality,
                tuple(
                    metric for metric in metrics if _metric_modality(metric) == modality
                ),
            )
            for modality in (*_MODALITIES, "other")
        ),
        "availability": tuple(availability),
    }


def _compare_run_metadata(
    summary_path: Path,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    # Read each artifact exactly once. Hashes and parsed values therefore
    # describe the same byte snapshots even if another process begins a new
    # atomic publication while comparison is validating this run.
    summary_bytes = summary_path.read_bytes()
    loaded = json.loads(summary_bytes.decode("utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError(f"summary must contain a JSON object: {summary_path}")
    provenance_path = summary_path.parent / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"comparison requires provenance.json beside summary: {provenance_path}"
        )
    provenance_bytes = provenance_path.read_bytes()
    provenance = json.loads(provenance_bytes.decode("utf-8"))
    if not isinstance(provenance, Mapping):
        raise TypeError(f"provenance must contain a JSON object: {provenance_path}")
    status_path = summary_path.parent / "run_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(
            f"comparison requires run_status.json beside summary: {status_path}"
        )
    status_bytes = status_path.read_bytes()
    status = json.loads(status_bytes.decode("utf-8"))
    if not isinstance(status, Mapping):
        raise TypeError(f"run status must contain a JSON object: {status_path}")
    if status.get("full_manifest_complete") is not True:
        raise RuntimeError(
            f"{status_path}: comparison requires full_manifest_complete=true"
        )
    identities = (
        _required_text(loaded.get("run_identity"), "run_identity", summary_path),
        _required_text(provenance.get("run_identity"), "run_identity", provenance_path),
        _required_text(status.get("run_identity"), "run_identity", status_path),
    )
    if len(set(identities)) != 1:
        raise RuntimeError(
            f"{summary_path.parent}: mixed report run identities: {identities}"
        )
    generations = (
        _required_text(
            loaded.get("publication_generation"),
            "publication_generation",
            summary_path,
        ),
        _required_text(
            provenance.get("publication_generation"),
            "publication_generation",
            provenance_path,
        ),
        _required_text(
            status.get("publication_generation"),
            "publication_generation",
            status_path,
        ),
    )
    if len(set(generations)) != 1:
        raise RuntimeError(
            f"{summary_path.parent}: mixed publication generations: {generations}"
        )
    expected_summary_hash = _required_text(
        status.get("summary_sha256"), "summary_sha256", status_path
    )
    expected_provenance_hash = _required_text(
        status.get("provenance_sha256"), "provenance_sha256", status_path
    )
    actual_summary_hash = hashlib.sha256(summary_bytes).hexdigest()
    actual_provenance_hash = hashlib.sha256(provenance_bytes).hexdigest()
    if actual_summary_hash != expected_summary_hash:
        raise RuntimeError(
            f"{summary_path}: SHA-256 mismatch against run_status.json; "
            f"expected {expected_summary_hash}, actual {actual_summary_hash}"
        )
    if actual_provenance_hash != expected_provenance_hash:
        raise RuntimeError(
            f"{provenance_path}: SHA-256 mismatch against run_status.json; "
            f"expected {expected_provenance_hash}, actual {actual_provenance_hash}"
        )
    manifest = provenance.get("manifest")
    checkpoint = provenance.get("checkpoint")
    if not isinstance(manifest, Mapping):
        raise TypeError(f"{provenance_path}: provenance.manifest must be an object")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{provenance_path}: provenance.checkpoint must be an object")
    manifest_sha = _required_text(
        manifest.get("sha256"), "manifest.sha256", provenance_path
    )
    checkpoint_class = _required_text(
        checkpoint.get("class_path"), "checkpoint.class_path", provenance_path
    )
    checkpoint_sha = _required_text(
        checkpoint.get("sha256"), "checkpoint.sha256", provenance_path
    )
    config_sha = _required_text(
        provenance.get("evaluation_config_sha256"),
        "evaluation_config_sha256",
        provenance_path,
    )
    evaluator_sha = _required_text(
        provenance.get("evaluator_sha256"), "evaluator_sha256", provenance_path
    )
    execution_environment_sha = _required_text(
        provenance.get("execution_environment_sha256"),
        "execution_environment_sha256",
        provenance_path,
    )
    asset_sha = _required_text(
        provenance.get("asset_sha256"), "asset_sha256", provenance_path
    )
    global_step = checkpoint.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise TypeError(f"{provenance_path}: checkpoint.global_step must be an integer")
    identity_payload = {
        "asset_sha256": asset_sha,
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config_sha,
        "evaluator_sha256": evaluator_sha,
        "execution_environment_sha256": execution_environment_sha,
        "manifest_sha256": manifest_sha,
    }
    computed_identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if computed_identity != identities[0]:
        raise RuntimeError(
            f"{provenance_path}: run_identity does not match immutable inputs; "
            f"expected {computed_identity}, actual {identities[0]}"
        )
    run_id = (
        f"{checkpoint_class.rsplit('.', maxsplit=1)[-1]}@step={global_step}:"
        f"ckpt={checkpoint_sha[:12]}:config={config_sha[:12]}"
    )
    return loaded, {
        "run_id": run_id,
        "run_identity_sha256": identities[0],
        "publication_generation": generations[0],
        "summary_sha256": actual_summary_hash,
        "provenance_sha256": actual_provenance_hash,
        "summary_path": str(summary_path),
        "provenance_path": str(provenance_path),
        "run_status_path": str(status_path),
        "manifest_sha256": manifest_sha,
        "checkpoint_class": checkpoint_class,
        "checkpoint_global_step": global_step,
        "checkpoint_sha256": checkpoint_sha,
        "evaluation_config_sha256": config_sha,
        "evaluator_sha256": evaluator_sha,
        "execution_environment_sha256": execution_environment_sha,
        "asset_sha256": asset_sha,
    }


def compare_runs(
    summary_paths: Sequence[str | Path], output_dir: str | Path
) -> dict[str, str]:
    """Compare complete summaries sharing one immutable sample manifest."""

    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    run_counts: dict[str, int] = defaultdict(int)
    expected_manifest_sha: str | None = None
    expected_config_sha: str | None = None
    expected_evaluator_sha: str | None = None
    expected_execution_environment_sha: str | None = None
    expected_layout: dict[str, Any] | None = None
    expected_modality_metrics: dict[str, tuple[str, ...]] = {}
    for raw_path in summary_paths:
        path = Path(raw_path).expanduser().resolve()
        loaded, metadata = _compare_run_metadata(path)
        layout = _summary_layout(loaded, path)
        if expected_layout is None:
            expected_layout = layout
        else:
            for field in (
                "condition_ids",
                "primary_offsets",
                "secondary_offsets",
            ):
                if layout[field] != expected_layout[field]:
                    label = (
                        "condition ID set"
                        if field == "condition_ids"
                        else f"{field} layout"
                    )
                    raise RuntimeError(
                        f"cannot compare summaries with different {label}: {path}"
                    )
        actual_availability = {
            condition_id: dict(values)
            for condition_id, values in layout["availability"]
        }
        actual_metrics = dict(layout["metrics_by_modality"])
        for modality in (*_MODALITIES, "other"):
            available = (
                bool(actual_metrics[modality])
                if modality == "other"
                else any(
                    values.get(modality, False)
                    for values in actual_availability.values()
                )
            )
            if not available:
                continue
            reference = expected_modality_metrics.setdefault(
                modality, actual_metrics[modality]
            )
            if actual_metrics[modality] != reference:
                raise RuntimeError(
                    "cannot compare summaries with different metric layout "
                    f"for jointly available modality {modality}: {path}"
                )
        manifest_sha = str(metadata["manifest_sha256"])
        if expected_manifest_sha is None:
            expected_manifest_sha = manifest_sha
        elif manifest_sha != expected_manifest_sha:
            raise RuntimeError(
                "cannot compare runs built from different sample manifests: "
                f"expected {expected_manifest_sha}, got {manifest_sha} at {path}"
            )
        config_sha = str(metadata["evaluation_config_sha256"])
        if expected_config_sha is None:
            expected_config_sha = config_sha
        elif config_sha != expected_config_sha:
            raise RuntimeError(
                "cannot compare runs built with different semantic evaluation "
                f"configs: expected {expected_config_sha}, got {config_sha} at {path}"
            )
        evaluator_sha = str(metadata["evaluator_sha256"])
        if expected_evaluator_sha is None:
            expected_evaluator_sha = evaluator_sha
        elif evaluator_sha != expected_evaluator_sha:
            raise RuntimeError(
                "cannot compare runs produced by different evaluator code: "
                f"expected {expected_evaluator_sha}, got {evaluator_sha} at {path}"
            )
        execution_environment_sha = str(metadata["execution_environment_sha256"])
        if expected_execution_environment_sha is None:
            expected_execution_environment_sha = execution_environment_sha
        elif execution_environment_sha != expected_execution_environment_sha:
            raise RuntimeError(
                "cannot compare runs produced in different execution environments: "
                f"expected {expected_execution_environment_sha}, "
                f"got {execution_environment_sha} at {path}"
            )
        base_run_id = str(metadata["run_id"])
        run_counts[base_run_id] += 1
        run_id = (
            base_run_id
            if run_counts[base_run_id] == 1
            else f"{base_run_id}#{run_counts[base_run_id]}"
        )
        metadata = {**metadata, "run_id": run_id}
        layout_payload = {key: list(value) for key, value in layout.items()}
        metadata["summary_layout_sha256"] = hashlib.sha256(
            json.dumps(layout_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        runs.append(metadata)
        for row in _summary_long_rows(loaded):
            rows.append({**metadata, **_json_safe(row)})
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(output / "comparison.csv", _tabular_rows(rows))
    json_path = atomic_write_json(output / "comparison.json", _json_safe(rows))
    runs_path = atomic_write_json(output / "comparison_runs.json", _json_safe(runs))
    md = [
        "# Navigation LAM run comparison",
        "",
        f"Manifest SHA-256: `{expected_manifest_sha or 'N/A'}`",
        "",
        "## Runs",
        "",
        "| run | checkpoint class | global step | checkpoint SHA-256 | config SHA-256 | evaluator SHA-256 | execution environment SHA-256 |",
        "|---|---|---:|---|---|---|---|",
    ]
    for run in runs:
        cells = (
            run["run_id"],
            run["checkpoint_class"],
            run["checkpoint_global_step"],
            run["checkpoint_sha256"],
            run["evaluation_config_sha256"],
            run["evaluator_sha256"],
            run["execution_environment_sha256"],
        )
        md.append("| " + " | ".join(_md_cell(cell) for cell in cells) + " |")
    md.extend(
        [
            "",
            "## Metrics",
            "",
            "| run | entry type | protocol | source dataset | source split | recipient dataset | recipient split | scope | offset | metric | status | value |",
            "|---|---|---|---|---|---|---|---|---:|---|---|---:|",
        ]
    )
    for row in rows:
        value = row.get("macro_mean", row.get("mean"))
        cells = (
            row["run_id"],
            row.get("entry_type"),
            row.get("protocol"),
            row.get("source_dataset"),
            row.get("source_split"),
            row.get("recipient_dataset", row.get("target_dataset")),
            row.get("recipient_split", row.get("target_split")),
            row["scope"],
            row.get("offset"),
            row["metric"],
            row.get("status"),
            value,
        )
        md.append("| " + " | ".join(_md_cell(cell) for cell in cells) + " |")
    md_path = atomic_write_text(output / "comparison.md", "\n".join(md) + "\n")
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "runs_json": str(runs_path),
        "markdown": str(md_path),
    }


__all__ = [
    "aggregate_rows",
    "bootstrap_mean_ci",
    "compare_runs",
    "describe_values",
    "metric_columns",
    "os_path_rel",
    "summary_markdown",
    "write_gallery",
    "write_heatmap",
    "write_run_reports",
]
