"""Calibrate the fixed action-loss weight for navigation LAM variant 4.

This command only instantiates an untrained model and runs inference.  It never
constructs a Trainer, optimizer, or checkpoint, and never calls backward().
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import SequentialSampler

from lam.navigation_dataset import NavigationPairDataModule
from lam.navigation_variants import PixelActionLAM

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = PROJECT_DIR / "config/navigation_base.yaml"
DEFAULT_VARIANT_CONFIG = PROJECT_DIR / "config/navigation_4_pixel_action.yaml"
DEFAULT_OUTPUT_CONFIG = PROJECT_DIR / "config/generated/navigation_4_action_weight.yaml"
CALIBRATION_SEED = 32
DEFAULT_BATCHES = 256
DEFAULT_BATCH_SIZE = 8
PIXEL_ACTION_CLASS = "lam.navigation_variants.PixelActionLAM"


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"Config must contain a mapping: {path}")
    return dict(value)


def _parse_named_paths(values: Sequence[str], option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects DATASET=/absolute/path, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path:
            raise ValueError(f"{option} expects DATASET=/absolute/path, got {value!r}")
        parsed[name.casefold()] = str(Path(raw_path).expanduser().resolve())
    return parsed


def _override_source_paths(
    source_specs: list[dict[str, Any]],
    data_roots: Mapping[str, str],
    split_manifests: Mapping[str, str],
) -> None:
    available = {str(spec["display_name"]).casefold(): spec for spec in source_specs}
    unknown = (set(data_roots) | set(split_manifests)) - set(available)
    if unknown:
        names = ", ".join(sorted(unknown))
        expected = ", ".join(str(spec["display_name"]) for spec in source_specs)
        raise ValueError(f"Unknown dataset override(s): {names}. Expected one of: {expected}")
    for name, path in data_roots.items():
        spec = available[name]
        if "root" in spec:
            spec["root"] = path
            spec.pop("data_root", None)
        else:
            spec["data_root"] = path
    for name, path in split_manifests.items():
        available[name]["split_manifest"] = path


def _seed_everything() -> None:
    random.seed(CALIBRATION_SEED)
    np.random.seed(CALIBRATION_SEED)
    torch.manual_seed(CALIBRATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CALIBRATION_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _move_tensors(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _build_components(
    config: Mapping[str, Any],
    *,
    batch_size: int | None,
    num_workers: int | None,
    data_roots: Mapping[str, str],
    split_manifests: Mapping[str, str],
) -> tuple[PixelActionLAM, NavigationPairDataModule]:
    model_config = copy.deepcopy(dict(config["model"]))
    if model_config.get("class_path") != PIXEL_ACTION_CLASS:
        raise ValueError(
            f"Variant config must select {PIXEL_ACTION_CLASS}, got "
            f"{model_config.get('class_path')!r}"
        )
    model_args = dict(model_config.get("init_args", {}))
    # Optimizer construction is irrelevant to a no-grad forward calibration.
    model_args.pop("optimizer", None)
    # The calibrated value cannot be known yet; it does not affect forward outputs.
    model_args["action_loss_weight"] = 1.0

    data_args = copy.deepcopy(dict(config["data"]))
    source_specs = [dict(spec) for spec in data_args["source_specs"]]
    _override_source_paths(source_specs, data_roots, split_manifests)
    data_args["source_specs"] = source_specs
    if data_roots or split_manifests:
        # Do not replace the shared default-mount audit cache with an index for
        # an ad-hoc calibration mount.  Callers can provide a dedicated cache
        # in a custom base config when desired.
        data_args["index_cache_path"] = None
    data_args["data_seed"] = CALIBRATION_SEED
    data_args["train_shuffle"] = data_args.get("labeled_batch_fraction") is not None
    if batch_size is not None:
        data_args["batch_size"] = batch_size
    if num_workers is not None:
        data_args["num_workers"] = num_workers

    model = PixelActionLAM(**model_args)
    data = NavigationPairDataModule(**data_args)
    return model, data


def calibrate(
    model: PixelActionLAM,
    data: NavigationPairDataModule,
    *,
    num_batches: int,
    device: torch.device,
    mixed_precision: bool,
    autocast_dtype: torch.dtype = torch.float16,
) -> dict[str, int | float]:
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")

    data.setup("fit")
    loader = data.train_dataloader()
    if getattr(data, "labeled_batch_fraction", None) is None and not isinstance(
        loader.sampler, SequentialSampler
    ):
        raise TypeError("Calibration without a label quota requires a sequential DataLoader")
    # This generator is separate from training and fixes worker base seeds.
    loader.generator = torch.Generator().manual_seed(CALIBRATION_SEED)

    model.to(device)
    # Match the formal training forward path (including VAE sampling) while
    # keeping the entire calibration graph-free and deterministic from seed 32.
    model.train()

    pixel_squared_error = 0.0
    action_squared_error = 0.0
    pixel_elements = 0
    action_elements = 0
    completed_batches = 0

    autocast_enabled = mixed_precision and device.type == "cuda"
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(loader):
            if batch_index >= num_batches:
                break
            batch = _move_tensors(cpu_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                outputs = model(batch)

            pixel_error = outputs["recon"] - batch["videos"][:, 1:]
            action_mask = batch.get("action_mask")
            if action_mask is None:
                action_mask = torch.ones(
                    outputs["action_recon"].shape[0], dtype=torch.bool, device=device
                )
            else:
                action_mask = action_mask.to(dtype=torch.bool).reshape(-1)
            action_error = (
                outputs["action_recon"][action_mask] - batch["actions"][action_mask]
            )
            pixel_squared_error += pixel_error.square().sum(dtype=torch.float64).item()
            action_squared_error += action_error.square().sum(dtype=torch.float64).item()
            pixel_elements += pixel_error.numel()
            action_elements += action_error.numel()
            completed_batches += 1

    if completed_batches != num_batches:
        raise RuntimeError(
            f"DataLoader ended after {completed_batches} batches; requested {num_batches}"
        )
    if pixel_elements == 0 or action_elements == 0:
        raise RuntimeError("Calibration observed no pixel or action elements")

    pixel_mse = pixel_squared_error / pixel_elements
    action_mse = action_squared_error / action_elements
    if not math.isfinite(pixel_mse) or not math.isfinite(action_mse) or action_mse <= 0:
        raise RuntimeError(
            f"Invalid calibration means: pixel_mse={pixel_mse}, action_mse={action_mse}"
        )

    return {
        "action_loss_weight": pixel_mse / action_mse,
        "pixel_mse": pixel_mse,
        "action_mse": action_mse,
        "pixel_elements": pixel_elements,
        "action_elements": action_elements,
        "batches": completed_batches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--variant-config", type=Path, default=DEFAULT_VARIANT_CONFIG)
    parser.add_argument("--profile-config", type=Path)
    parser.add_argument("--output-config", type=Path, default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument("--num-batches", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Calibration device (default: cuda; use cpu only for small smoke configs).",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Override a configured source root by display_name; repeat as needed.",
    )
    parser.add_argument(
        "--split-manifest",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Override a configured train manifest by display_name; repeat as needed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA calibration was requested, but CUDA is unavailable. Use --device cpu only for smoke tests.")

    base_path = args.base_config.expanduser().resolve()
    variant_path = args.variant_config.expanduser().resolve()
    output_path = args.output_config.expanduser().resolve()
    data_roots = _parse_named_paths(args.data_root, "--data-root")
    split_manifests = _parse_named_paths(args.split_manifest, "--split-manifest")
    config = _deep_merge(_load_yaml(base_path), _load_yaml(variant_path))
    profile_path = None
    profile_config = getattr(args, "profile_config", None)
    if profile_config is not None:
        profile_path = profile_config.expanduser().resolve()
        config = _deep_merge(config, _load_yaml(profile_path))

    _seed_everything()
    model, data = _build_components(
        config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_roots=data_roots,
        split_manifests=split_manifests,
    )
    precision = str(config.get("trainer", {}).get("precision", "32-true"))
    statistics = calibrate(
        model,
        data,
        num_batches=args.num_batches,
        device=device,
        mixed_precision=precision in {"16-mixed", "bf16-mixed"},
        autocast_dtype=(torch.bfloat16 if precision == "bf16-mixed" else torch.float16),
    )

    overlay = {
        "model": {
            "init_args": {
                "action_loss_weight": statistics["action_loss_weight"],
            }
        }
    }
    metadata = {
        **statistics,
        "seed": CALIBRATION_SEED,
        "precision": precision,
        "base_config": str(base_path),
        "variant_config": str(variant_path),
        "profile_config": None if profile_path is None else str(profile_path),
        "data_root_overrides": data_roots,
        "split_manifest_overrides": split_manifests,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")

    overlay_text = (
        "# Generated by calibrate_action_weight.py; do not tune during training.\n"
        f"# seed: {CALIBRATION_SEED}\n"
        f"# batches: {statistics['batches']}\n"
        f"# pixel_mse: {statistics['pixel_mse']:.17g}\n"
        f"# action_mse: {statistics['action_mse']:.17g}\n"
        + yaml.safe_dump(overlay, sort_keys=False)
    )
    _atomic_write_text(output_path, overlay_text)
    _atomic_write_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"pixel_mse={statistics['pixel_mse']:.12g}")
    print(f"action_mse={statistics['action_mse']:.12g}")
    print(f"action_loss_weight={statistics['action_loss_weight']:.12g}")
    print(f"Wrote Lightning overlay: {output_path}")
    print(f"Wrote calibration metadata: {metadata_path}")


if __name__ == "__main__":
    main()
