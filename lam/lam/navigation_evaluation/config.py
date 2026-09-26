"""Configuration loading and validation for navigation LAM evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .catalog import canonical_dataset_id

PROTOCOLS = ("id", "id_cd", "ood", "ood_cd")
REQUIRED_OFFSETS = (-8, -6, -4, -2, 0, 2, 4, 6, 8)
REQUIRED_METRICS = {
    "rgb": ("mse", "mae", "psnr", "ssim", "lpips"),
    "feature": ("mse", "mae", "cosine_similarity"),
    "action": (
        "normalized_mse",
        "normalized_mae",
        "translation_epe_waypoints",
        "translation_epe_meters",
        "yaw_mae_degrees",
        "yaw_rmse_degrees",
    ),
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def config_digest(config: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for a resolved configuration."""

    return hashlib.sha256(_canonical_json(config)).hexdigest()


def _merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = _merge(dict(base[key]), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def validate_evaluation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the public evaluation YAML contract."""

    resolved = copy.deepcopy(dict(config))
    if resolved.get("version") != 1:
        raise ValueError("navigation evaluation config version must be 1")

    datasets = resolved.get("datasets")
    if not isinstance(datasets, Mapping) or not datasets:
        raise ValueError("datasets must be a non-empty mapping")
    for dataset_id, raw in datasets.items():
        canonical_id = canonical_dataset_id(str(dataset_id))
        if canonical_id != dataset_id:
            raise ValueError(
                f"datasets key {dataset_id!r} is not canonical; use {canonical_id!r}"
            )
        if not isinstance(raw, Mapping):
            raise TypeError(f"datasets.{dataset_id} must be a mapping")
        for field in ("display_name", "data_root", "waypoint_spacing", "splits"):
            if field not in raw:
                raise ValueError(f"datasets.{dataset_id}.{field} is required")
        spacing = float(raw["waypoint_spacing"])
        if spacing <= 0:
            raise ValueError(f"datasets.{dataset_id}.waypoint_spacing must be positive")
        splits = raw["splits"]
        if not isinstance(splits, Mapping) or set(splits) != {"train", "test"}:
            raise ValueError(
                f"datasets.{dataset_id}.splits must contain exactly train and test"
            )

    domains = resolved.get("domains", {})
    if not isinstance(domains, Mapping):
        raise TypeError("domains must be a mapping")
    train_domains = tuple(
        canonical_dataset_id(value) for value in domains.get("train", ())
    )
    ood_domains = tuple(canonical_dataset_id(value) for value in domains.get("ood", ()))
    if not train_domains:
        raise ValueError("domains.train must contain at least one explicit dataset")
    if not ood_domains:
        raise ValueError("domains.ood must contain at least one explicit dataset")
    if len(set(train_domains)) != len(train_domains) or len(set(ood_domains)) != len(
        ood_domains
    ):
        raise ValueError("domains.train/domains.ood cannot contain duplicates")
    overlap = set(train_domains) & set(ood_domains)
    if overlap:
        raise ValueError(f"ID and OOD domains overlap: {sorted(overlap)}")
    missing_domains = (set(train_domains) | set(ood_domains)) - set(datasets)
    if missing_domains:
        raise ValueError(
            f"domains reference missing dataset configs: {sorted(missing_domains)}"
        )
    resolved["domains"] = {
        "train": list(train_domains),
        "ood": list(ood_domains),
    }
    splits = tuple(resolved.get("splits", ("test",)))
    if (
        not splits
        or len(set(splits)) != len(splits)
        or any(split not in {"train", "test"} for split in splits)
    ):
        raise ValueError("splits must be a non-empty unique subset of train/test")
    resolved["splits"] = list(splits)

    sampling = resolved.get("sampling")
    if not isinstance(sampling, Mapping):
        raise TypeError("sampling must be a mapping")
    offsets = tuple(int(value) for value in sampling.get("offsets", ()))
    if offsets != REQUIRED_OFFSETS:
        raise ValueError(f"sampling.offsets must be exactly {list(REQUIRED_OFFSETS)}")
    _positive_int(sampling.get("metric_count"), "sampling.metric_count")
    _nonnegative_int(sampling.get("visual_count"), "sampling.visual_count")
    if sampling["visual_count"] > sampling["metric_count"]:
        raise ValueError("sampling.visual_count cannot exceed metric_count")
    _positive_int(sampling.get("image_height"), "sampling.image_height")
    _positive_int(sampling.get("image_width"), "sampling.image_width")
    _nonnegative_int(resolved.get("seed"), "seed")

    matching = resolved.get("matching")
    if not isinstance(matching, Mapping):
        raise TypeError("matching must be a mapping")
    matching = dict(matching)
    matching.setdefault("nearest_candidates", 64)
    matching.setdefault("candidate_pool_multiplier", 20)
    if float(matching.get("translation_tolerance_waypoint", 0)) <= 0:
        raise ValueError("matching translation tolerance must be positive")
    yaw_tolerance = float(matching.get("yaw_tolerance_degrees", 0))
    if not 0 < yaw_tolerance < 180:
        raise ValueError("matching yaw tolerance must be in (0, 180) degrees")
    if matching.get("recipient_pair_unique", True) is not True:
        raise ValueError("matching.recipient_pair_unique must remain true")
    _positive_int(matching["nearest_candidates"], "matching.nearest_candidates")
    _positive_int(
        matching["candidate_pool_multiplier"], "matching.candidate_pool_multiplier"
    )
    resolved["matching"] = matching

    requested_protocols = tuple(resolved.get("protocols", ()))
    if not requested_protocols or any(p not in PROTOCOLS for p in requested_protocols):
        raise ValueError(f"protocols must be selected from {list(PROTOCOLS)}")
    if len(set(requested_protocols)) != len(requested_protocols):
        raise ValueError("protocols cannot contain duplicates")

    runtime = resolved.get("runtime")
    if not isinstance(runtime, Mapping):
        raise TypeError("runtime must be a mapping")
    _positive_int(runtime.get("batch_size"), "runtime.batch_size")
    _nonnegative_int(runtime.get("num_workers"), "runtime.num_workers")
    _positive_int(runtime.get("shard_count"), "runtime.shard_count")
    shard_index = int(runtime.get("shard_index", -1))
    if not 0 <= shard_index < int(runtime["shard_count"]):
        raise ValueError("runtime.shard_index must be in [0, shard_count)")
    precision = runtime.get("precision")
    if precision not in {"32", "16-mixed", "bf16-mixed"}:
        raise ValueError("runtime.precision must be 32, 16-mixed, or bf16-mixed")
    device = str(runtime.get("device", "")).strip()
    if not device:
        raise ValueError("runtime.device must be configured")
    if precision != "32" and not (device == "cuda" or device.startswith("cuda:")):
        raise ValueError("mixed precision requires a CUDA runtime.device")
    if not isinstance(runtime.get("verify_data_files", True), bool):
        raise TypeError("runtime.verify_data_files must be boolean")

    metrics = resolved.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    for modality, required in REQUIRED_METRICS.items():
        configured = metrics.get(modality)
        if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
            raise TypeError(f"metrics.{modality} must be a sequence")
        if tuple(configured) != required:
            raise ValueError(
                f"metrics.{modality} is fixed for comparable evaluation and must "
                f"equal {list(required)}"
            )
    _positive_int(metrics.get("bootstrap_samples"), "metrics.bootstrap_samples")
    confidence = float(metrics.get("bootstrap_confidence", 0))
    if not 0 < confidence < 1:
        raise ValueError("metrics.bootstrap_confidence must be in (0, 1)")
    lpips = metrics.get("lpips")
    if not isinstance(lpips, Mapping):
        raise TypeError("metrics.lpips must be a mapping")
    if bool(lpips.get("enabled", True)):
        if not bool(lpips.get("offline", False)):
            raise ValueError("formal LPIPS evaluation must be offline")
        if not str(lpips.get("asset_root", "")).strip():
            raise ValueError("metrics.lpips.asset_root is required")

    visualization = resolved.get("visualization")
    if not isinstance(visualization, Mapping):
        raise TypeError("visualization must be a mapping")
    for field in (
        "thumbnail_width",
        "thumbnail_height",
        "row_label_width",
        "font_size",
        "pca_reference_frames_per_dataset_split",
    ):
        _positive_int(visualization.get(field), f"visualization.{field}")
    _positive_int(
        visualization.get("pca_max_tokens", 50_000),
        "visualization.pca_max_tokens",
    )
    lower = float(visualization.get("pca_lower_percentile", -1))
    upper = float(visualization.get("pca_upper_percentile", -1))
    if not 0 <= lower < upper <= 100:
        raise ValueError("DINO PCA percentiles must satisfy 0 <= lower < upper <= 100")

    output = resolved.get("output")
    if not isinstance(output, Mapping) or not str(output.get("root", "")).strip():
        raise ValueError("output.root must be configured")
    if output.get("save_visual_predictions_only") is not True:
        raise ValueError("output.save_visual_predictions_only must remain true")
    if not isinstance(output.get("save_parquet"), bool):
        raise TypeError("output.save_parquet must be boolean")
    return resolved


def load_evaluation_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load YAML, apply a nested override mapping, and validate it."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"evaluation config does not exist: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError(f"evaluation config must contain a mapping: {config_path}")
    resolved = dict(loaded)
    if overrides:
        resolved = _merge(resolved, overrides)
    resolved = validate_evaluation_config(resolved)
    resolved["_config_path"] = str(config_path)
    return resolved


def parse_csv_protocols(value: str | Sequence[str] | None) -> list[str] | None:
    """Parse a CLI protocol override while preserving canonical order."""

    if value is None:
        return None
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        values = [str(part).strip() for part in value if str(part).strip()]
    unknown = set(values) - set(PROTOCOLS)
    if unknown:
        raise ValueError(f"unknown protocols: {sorted(unknown)}")
    return [protocol for protocol in PROTOCOLS if protocol in values]
