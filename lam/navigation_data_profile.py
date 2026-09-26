"""Generate the portable NavAnywhere mixed-training Lightning overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

NAVANYWHERE_DATASETS = (
    "BotanicGarden",
    "CASIA-Nav",
    "CityWalker",
    "DL3DV-10K",
    "EgoWalk",
    "KrishnaCam",
    "LAVN",
    "ROVER",
    "RealEstate10K",
    "SANPO",
    "Walking_Tours",
    "i2Nav-Robot",
    "uB-VisioGeoloc",
)
NAVANYWHERE15_ADDITIONAL_DATASETS = (
    "The_Great_Outdoors",
    "ego4d",
)
NAVANYWHERE15_DATASETS = NAVANYWHERE_DATASETS + NAVANYWHERE15_ADDITIONAL_DATASETS
ACTION_SOURCES = (
    ("recon", "RECON", 0.25),
    ("sacson", "SACSoN", 0.255),
    ("scand", "SCAND", 0.38),
)
NAVANYWHERE_ABLATION_FRACTIONS = (25, 50, 75, 100)
PHYSICAL_LABEL_FRACTIONS = (0, 50, 100)
NAVANYWHERE_ABLATION_SAMPLES_PER_EPOCH = 2_560_000
NAVANYWHERE_ABLATION_SEED = 32
NAVANYWHERE_ABLATION_MAX_FRAME_OFFSET = 8
NAVANYWHERE_ABLATION_BUNDLE_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_digest(metadata: Mapping[str, Any]) -> str:
    payload = dict(metadata)
    payload.pop("bundle_sha256", None)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _require_ablation_bundle_file(
    bundle: Path, metadata: Mapping[str, Any], relative_path: Path
) -> Path:
    """Validate one immutable bundle artifact against ``metadata.json``."""

    relative_name = relative_path.as_posix()
    files = metadata.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("Ablation bundle metadata.files must be a mapping")
    identity = files.get(relative_name)
    if identity is None:
        raise RuntimeError(
            f"Ablation bundle metadata does not describe {relative_name}"
        )
    if not isinstance(identity, Mapping):
        raise TypeError(
            f"Ablation bundle metadata has malformed identity for {relative_name}"
        )
    artifact = bundle / relative_path
    if not artifact.is_file():
        raise FileNotFoundError(f"Ablation bundle artifact does not exist: {artifact}")
    expected_size = identity.get("size")
    if not isinstance(expected_size, int) or expected_size < 0:
        raise TypeError(
            f"Ablation bundle metadata has invalid size for {relative_name}"
        )
    actual_size = artifact.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Ablation bundle artifact size mismatch for {relative_name}: "
            f"expected {expected_size}, got {actual_size}"
        )
    expected_sha256 = identity.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise TypeError(
            f"Ablation bundle metadata has invalid sha256 for {relative_name}"
        )
    actual_sha256 = _sha256(artifact)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Ablation bundle artifact SHA-256 mismatch for {relative_name}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return artifact


def _ablation_paths(
    *,
    bundle: Path,
    data_root: Path,
    split_root: Path,
    navanywhere_fraction: int,
    physical_label_fraction: int,
) -> tuple[dict[str, Path], dict[str, Path], Path]:
    """Validate a prepared bundle and return the requested immutable inputs."""

    if not bundle.is_absolute():
        raise ValueError("nav_ablation_bundle must be an absolute path")
    bundle = bundle.expanduser().resolve(strict=False)
    if not bundle.is_dir():
        raise FileNotFoundError(f"Ablation bundle directory does not exist: {bundle}")
    metadata_path = bundle / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Complete ablation bundle metadata does not exist: {metadata_path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read ablation bundle metadata {metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, Mapping):
        raise TypeError(f"Malformed ablation bundle metadata: {metadata_path}")
    expected_identity = {
        "version": NAVANYWHERE_ABLATION_BUNDLE_VERSION,
        "complete": True,
        "seed": NAVANYWHERE_ABLATION_SEED,
        "max_frame_offset": NAVANYWHERE_ABLATION_MAX_FRAME_OFFSET,
    }
    for key, expected in expected_identity.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"Ablation bundle {metadata_path} has {key}={metadata.get(key)!r}; "
                f"expected {expected!r}"
            )
    bundle_sha256 = metadata.get("bundle_sha256")
    if (
        not isinstance(bundle_sha256, str)
        or len(bundle_sha256) != 64
        or any(character not in "0123456789abcdef" for character in bundle_sha256)
    ):
        raise RuntimeError(
            f"Ablation bundle {metadata_path} has an invalid bundle_sha256"
        )
    if bundle_sha256 != _metadata_digest(metadata):
        raise RuntimeError(
            f"Ablation bundle metadata digest mismatch: {metadata_path}"
        )

    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError("Ablation bundle metadata.inputs must be a mapping")
    expected_inputs = {
        "data_root": str(data_root.expanduser().resolve(strict=False)),
        "split_root": str(split_root.expanduser().resolve(strict=False)),
        "navanywhere_datasets": list(NAVANYWHERE15_DATASETS),
        "action_sources": [
            {
                "directory": directory,
                "display_name": display_name,
                "waypoint_spacing": spacing,
            }
            for directory, display_name, spacing in ACTION_SOURCES
        ],
    }
    for key, expected in expected_inputs.items():
        if inputs.get(key) != expected:
            raise RuntimeError(
                f"Ablation bundle {metadata_path} has stale inputs.{key}: "
                f"{inputs.get(key)!r} != {expected!r}"
            )

    nav_manifests = {
        name: _require_ablation_bundle_file(
            bundle,
            metadata,
            Path("manifests")
            / "navanywhere"
            / f"u{navanywhere_fraction}"
            / f"{name}.txt",
        )
        for name in NAVANYWHERE15_DATASETS
    }
    action_manifests = {
        directory: _require_ablation_bundle_file(
            bundle,
            metadata,
            Path("manifests")
            / "physical"
            / f"l{physical_label_fraction}"
            / f"{directory}.txt",
        )
        for directory, _, _ in ACTION_SOURCES
    }
    index_cache = _require_ablation_bundle_file(
        bundle,
        metadata,
        Path("indices") / f"navigation_u{navanywhere_fraction}.pkl",
    )
    return nav_manifests, action_manifests, index_cache


def build_profile(
    *,
    variant: int,
    data_root: Path,
    split_root: Path,
    index_cache_path: Path,
    dino_repo_path: Path,
    dino_weights_path: Path,
    navanywhere_datasets: tuple[str, ...] = NAVANYWHERE_DATASETS,
    navanywhere_fraction: int | None = None,
    physical_label_fraction: int | None = None,
    nav_ablation_bundle: Path | None = None,
) -> dict[str, Any]:
    if variant not in {2, 3, 4}:
        raise ValueError("The NavAnywhere mixed profile supports variants 2, 3, and 4")
    ablation_values = (
        navanywhere_fraction,
        physical_label_fraction,
        nav_ablation_bundle,
    )
    ablation_requested = any(value is not None for value in ablation_values)
    if ablation_requested and not all(value is not None for value in ablation_values):
        raise ValueError(
            "navanywhere_fraction, physical_label_fraction, and "
            "nav_ablation_bundle must be provided together"
        )
    nav_manifests: dict[str, Path] | None = None
    action_manifests: dict[str, Path] | None = None
    if ablation_requested:
        if variant != 4:
            raise ValueError("NavAnywhere data ablation is only supported for variant 4")
        if tuple(navanywhere_datasets) != NAVANYWHERE15_DATASETS:
            raise ValueError(
                "NavAnywhere data ablation requires the 15-source NavAnywhere profile"
            )
        if navanywhere_fraction not in NAVANYWHERE_ABLATION_FRACTIONS:
            raise ValueError(
                "navanywhere_fraction must be one of "
                f"{NAVANYWHERE_ABLATION_FRACTIONS}"
            )
        if physical_label_fraction not in PHYSICAL_LABEL_FRACTIONS:
            raise ValueError(
                "physical_label_fraction must be one of "
                f"{PHYSICAL_LABEL_FRACTIONS}"
            )
        assert nav_ablation_bundle is not None
        nav_manifests, action_manifests, index_cache_path = _ablation_paths(
            bundle=nav_ablation_bundle,
            data_root=data_root,
            split_root=split_root,
            navanywhere_fraction=navanywhere_fraction,
            physical_label_fraction=physical_label_fraction,
        )
    source_specs: list[dict[str, Any]] = [
        {
            "root": str(data_root / "NavAnywhere" / name),
            **(
                {"split_manifest": str(nav_manifests[name])}
                if nav_manifests is not None
                else {}
            ),
            "display_name": name,
            "format": "frame_only",
            "has_action": False,
            "waypoint_spacing": 1.0,
        }
        for name in navanywhere_datasets
    ]
    source_specs.extend(
        {
            "root": str(data_root / directory),
            "split_manifest": str(split_root / directory / "train" / "traj_names.txt"),
            **(
                {"action_label_manifest": str(action_manifests[directory])}
                if action_manifests is not None
                else {}
            ),
            "display_name": display_name,
            "format": "pose_labeled",
            "has_action": True,
            "waypoint_spacing": spacing,
        }
        for directory, display_name, spacing in ACTION_SOURCES
    )
    model_args: dict[str, Any] = {
        "scheduler_total_steps": 50_000,
        "scheduler_warmup_steps": 2_000,
        "scheduler_min_lr_ratio": 0.1,
        "optimizer": {
            "class_path": "torch.optim.AdamW",
            "init_args": {"lr": 6.25e-5, "weight_decay": 1e-2},
        },
    }
    data: dict[str, Any] = {
        "source_specs": source_specs,
        "batch_size": 32,
        "num_workers": 8,
        "train_shuffle": True,
        "samples_per_epoch": (
            NAVANYWHERE_ABLATION_SAMPLES_PER_EPOCH
            if ablation_requested
            else 5_120_000
        ),
        "max_frame_offset": 8,
        "max_frame_decode_retries": 128,
        "data_seed": 32,
        "color_aug": True,
        "image_height": 240,
        "image_width": 320,
        "action_min": [-2.5, -4.0],
        "action_max": [5.0, 4.0],
        "print_audit": True,
        "index_cache_path": str(index_cache_path),
        "rebuild_index_cache": False,
        "pin_memory": True,
        "persistent_workers": True,
        "drop_last": True,
        "dataset_size_exponent": 0.5,
        "uniform_offset_domain": True,
        "labeled_batch_fraction": 0.25 if variant == 4 else None,
        "dino_feature_cache_path": None,
    }
    if variant == 3:
        model_args.update(
            {
                "dino_repo_path": str(dino_repo_path),
                "dino_weights_path": str(dino_weights_path),
                "use_precomputed_dino_features": False,
            }
        )
    if ablation_requested:
        # Zero is an explicit, backwards-compatible image-logging off switch;
        # the base model default remains 1000 for every existing profile.
        model_args["log_interval"] = 0
        # Bundle indices are immutable experiment inputs.  A cache mismatch
        # must fail closed instead of rebuilding in place and invalidating the
        # already-recorded bundle/contract hashes.
        data["require_index_cache"] = True
    return {
        "model": {"init_args": model_args},
        "data": data,
        "trainer": {
            "max_epochs": -1,
            "max_steps": 50_000,
            "precision": "bf16-mixed",
            "log_every_n_steps": 100,
            "accumulate_grad_batches": 1,
            "gradient_clip_val": 0.3,
            "use_distributed_sampler": False,
        },
    }


def build_legacy_profile(
    *,
    variant: int,
    data_root: Path,
    split_root: Path,
    index_cache_path: Path,
    dino_repo_path: Path,
    dino_weights_path: Path,
    dino_feature_cache_path: Path,
) -> dict[str, Any]:
    """Override only mount-dependent paths while retaining legacy sampling/training."""

    source_specs = [
        {
            "data_root": str(data_root / directory),
            "split_manifest": str(split_root / directory / "train" / "traj_names.txt"),
            # Keep the legacy source schema byte-for-byte in spirit: omitting
            # the new format/has_action fields is what selects the original
            # three-source index cycling and per-anchor random offset path.
            "display_name": "HuRoN" if directory == "sacson" else display_name,
            "waypoint_spacing": spacing,
        }
        for directory, display_name, spacing in ACTION_SOURCES
    ]
    config: dict[str, Any] = {
        "data": {
            "source_specs": source_specs,
            "index_cache_path": str(index_cache_path),
        }
    }
    if variant == 3:
        config["model"] = {
            "init_args": {
                "dino_repo_path": str(dino_repo_path),
                "dino_weights_path": str(dino_weights_path),
            }
        }
        config["data"]["dino_feature_cache_path"] = str(dino_feature_cache_path)
    return config


def write_profile(config: dict[str, Any], output: Path) -> Path:
    output = output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        yaml.safe_dump(config, handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=int, required=True, choices=(1, 2, 3, 4))
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--legacy", action="store_true")
    profile.add_argument(
        "--navanywhere15",
        action="store_true",
        help=(
            "Use the 15-source NavAnywhere profile; the default remains the "
            "original 13-source profile."
        ),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--index-cache-path", type=Path, required=True)
    parser.add_argument("--dino-repo-path", type=Path, required=True)
    parser.add_argument("--dino-weights-path", type=Path, required=True)
    parser.add_argument("--dino-feature-cache-path", type=Path, required=True)
    parser.add_argument(
        "--navanywhere-fraction",
        type=int,
        choices=NAVANYWHERE_ABLATION_FRACTIONS,
    )
    parser.add_argument(
        "--physical-label-fraction",
        type=int,
        choices=PHYSICAL_LABEL_FRACTIONS,
    )
    parser.add_argument("--nav-ablation-bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    ablation_values = (
        args.navanywhere_fraction,
        args.physical_label_fraction,
        args.nav_ablation_bundle,
    )
    if any(value is not None for value in ablation_values):
        if not all(value is not None for value in ablation_values):
            parser.error(
                "--navanywhere-fraction, --physical-label-fraction, and "
                "--nav-ablation-bundle must be provided together"
            )
        if args.variant != 4 or not args.navanywhere15 or args.legacy:
            parser.error(
                "data ablation requires --variant 4 and --navanywhere15"
            )
    common = {
        "variant": args.variant,
        "data_root": args.data_root.expanduser().resolve(),
        "split_root": args.split_root.expanduser().resolve(),
        "index_cache_path": args.index_cache_path.expanduser().resolve(),
        "dino_repo_path": args.dino_repo_path.expanduser().resolve(),
        "dino_weights_path": args.dino_weights_path.expanduser().resolve(),
    }
    if args.legacy:
        config = build_legacy_profile(
            **common,
            dino_feature_cache_path=args.dino_feature_cache_path.expanduser().resolve(),
        )
    else:
        config = build_profile(
            **common,
            navanywhere_datasets=(
                NAVANYWHERE15_DATASETS
                if args.navanywhere15
                else NAVANYWHERE_DATASETS
            ),
            navanywhere_fraction=args.navanywhere_fraction,
            physical_label_fraction=args.physical_label_fraction,
            nav_ablation_bundle=(
                None
                if args.nav_ablation_bundle is None
                else args.nav_ablation_bundle.expanduser()
            ),
        )
    write_profile(
        config,
        args.output,
    )


if __name__ == "__main__":
    main()
