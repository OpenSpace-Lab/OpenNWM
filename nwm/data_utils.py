import hashlib
import json
import logging
import os
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler, Sampler
from datasets import TrainingDataset
from misc import CenterCropAR, IMAGE_ASPECT_RATIO, get_transform
from hydra.utils import get_original_cwd
import time

from motion_condition import (
    motion_condition_collate,
    resolve_dataset_motion_map,
    validate_motion_types,
)


logger = logging.getLogger(__name__)
LATENT_SCHEMA_VERSION = 1
LATENT_FORMAT = "sd_vae_posterior_stats"
# CUDA/PyTorch can report a few MiB less or more usable memory for the same GPU
# model after a driver, firmware, or ECC configuration change.  That difference
# does not affect the numerical contents of an already extracted posterior
# cache, so keep the hardware-family checks strict while allowing only a small
# capacity-reporting drift at runtime.
RUNTIME_GPU_MEMORY_TOLERANCE_BYTES = 64 * 1024 * 1024


class ReferenceBatchSampler(Sampler):
    """Rebatch a sampler without changing a smaller reference-batch prefix.

    NavAnywhere's strict offline action cache was planned with per-rank batch
    16 and ``drop_last=True``. A larger ordinary DataLoader batch changes the
    epoch prefix and can therefore request pairs outside that frozen plan. This
    opt-in sampler first keeps exactly the samples that the reference batch
    contract would have consumed, then merges consecutive reference batches
    into larger optimizer batches. A final reference-aligned partial batch is
    emitted instead of discarded.
    """

    def __init__(
        self,
        sampler,
        *,
        batch_size: int,
        reference_batch_size: int,
    ) -> None:
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.reference_batch_size = int(reference_batch_size)
        if self.batch_size < 1 or self.reference_batch_size < 1:
            raise ValueError("Batch sizes must be positive")
        if self.batch_size < self.reference_batch_size:
            raise ValueError(
                "training.batch_size must be at least reference_batch_size"
            )
        if self.batch_size % self.reference_batch_size:
            raise ValueError(
                "training.batch_size must be a multiple of "
                "training.reference_batch_size"
            )
        self.usable_samples = (
            len(self.sampler) // self.reference_batch_size
        ) * self.reference_batch_size
        if self.usable_samples < self.reference_batch_size:
            raise ValueError(
                "Distributed sampler has no full reference batch; lower "
                "training.reference_batch_size or provide more samples"
            )

    def __iter__(self):
        batch = []
        for sample_index, index in enumerate(self.sampler):
            if sample_index >= self.usable_samples:
                break
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            # usable_samples and batch_size are both reference aligned.
            assert len(batch) % self.reference_batch_size == 0
            yield batch

    def __len__(self) -> int:
        return (self.usable_samples + self.batch_size - 1) // self.batch_size


def _original_cwd() -> str:
    try:
        return get_original_cwd()
    except ValueError:
        # Enables standalone cache validation tests before Hydra initializes.
        return os.getcwd()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to read JSON file {path}: {exc}") from exc


def validate_precomputed_latent_cache(config: DictConfig) -> dict:
    """Read the existing cache manifests; tensors are loaded only on demand."""
    root = os.path.abspath(os.path.expanduser(str(config.dataset.precomputed_latents.root)))
    metadata = _read_json(os.path.join(root, "metadata.json"))
    records = {}
    for name, entry in config.dataset.datasets.items():
        if not entry.get("enabled", True) or entry.get("evaluation_only", False):
            continue
        with open(os.path.join(root, name, "manifest.jsonl"), encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        records[name] = {str(row["trajectory_name"]): row for row in rows}
    return {"root": root, "metadata": metadata, "manifest_records": records,
            "expected_file_metadata": {"scaling_factor": float(config.dataset.precomputed_latents.scaling_factor),
                                       "image_size": int(config.dataset.image_size)}}


def prepare_datasets(
    config: DictConfig,
    *,
    finetune_substage: str | None = None,
    include_test: bool = True,
):
    """Create and prepare training and test datasets."""
    train_dataset = []
    test_dataset = []
    train_motion_audit = []

    training_stage = str(config.get("training_stage", "legacy"))
    latent_config = config.dataset.get("precomputed_latents", {})
    use_precomputed_latents = bool(latent_config.get("enabled", False))
    validated_latents = None
    if use_precomputed_latents:
        validated_latents = validate_precomputed_latent_cache(config)
        logger.info(
            "Precomputed SD-VAE posterior mode enabled for training only: "
            f"root={validated_latents['root']}, "
            f"worker_lru_trajectories={int(latent_config.cache_size)}"
        )
    else:
        logger.info(
            "Precomputed latent mode disabled; training images will be decoded "
            "and encoded by the VAE online"
        )

    active_datasets = {
        str(name): dataset_config
        for name, dataset_config in config.dataset.datasets.items()
        if bool(dataset_config.get("enabled", True))
    }
    if not active_datasets:
        raise ValueError("At least one dataset must be enabled")

    if training_stage == "real_finetune":
        protocol = str(config.get("finetune", {}).get("dataset_protocol", "paper"))
        if protocol not in {"paper", "go2"}:
            raise ValueError(f"Unknown finetune.dataset_protocol: {protocol}")
        expected_training_datasets = (
            {"go2"} if protocol == "go2"
            else {"recon", "scand", "tartan_drive", "sacson"}
        )
        selected_training_datasets = {
            name
            for name, dataset_config in active_datasets.items()
            if "train" in dataset_config
            and not bool(dataset_config.get("evaluation_only", False))
        }
        if selected_training_datasets != expected_training_datasets:
            raise ValueError(
                f"real_finetune protocol={protocol} must train on exactly "
                f"{sorted(expected_training_datasets)}; got "
                f"{sorted(selected_training_datasets)}"
            )
        declared_selection = config.get("dataset_selection", {}).get(
            "finetune", None
        )
        if declared_selection is not None and set(map(str, declared_selection)) != (
            expected_training_datasets
        ):
            raise ValueError(
                f"dataset_selection.finetune must be exactly {sorted(expected_training_datasets)}"
            )
        global_distance = (
            int(config.dataset.distance.min_dist_cat),
            int(config.dataset.distance.max_dist_cat),
        )
        if global_distance != (-64, 64):
            raise ValueError(
                "real_finetune must retain the complete frame-offset range "
                f"[-64, 64], got {list(global_distance)}"
            )
        for name in selected_training_datasets:
            data_config = active_datasets[name]
            effective_distance = (
                int(
                    data_config.get("distance", {}).get(
                        "min_dist_cat", global_distance[0]
                    )
                ),
                int(
                    data_config.get("distance", {}).get(
                        "max_dist_cat", global_distance[1]
                    )
                ),
            )
            if effective_distance != (-64, 64):
                raise ValueError(
                    f"real_finetune dataset {name!r} must sample [-64, 64], "
                    f"got {list(effective_distance)}"
                )

    motion_config = config.get("motion_condition", None)
    use_motion_condition = bool(
        motion_config is not None and motion_config.get("enabled", False)
    )
    train_motion_types = None
    eval_motion_types = None
    dataset_motion_map = None
    if use_motion_condition:
        train_motion_types = validate_motion_types(motion_config, training=True)
        training_dataset_names = [
            str(name)
            for name, dataset_config in active_datasets.items()
            if "train" in dataset_config
            and not bool(dataset_config.get("evaluation_only", False))
        ]
        dataset_motion_map = resolve_dataset_motion_map(
            motion_config, training_dataset_names
        )
        eval_motion_type = str(motion_config.get("eval_type", "real"))
        available_types = validate_motion_types(motion_config)
        if eval_motion_type not in available_types:
            raise ValueError(
                f"motion_condition.eval_type={eval_motion_type!r} is not available"
            )
        eval_motion_types = (eval_motion_type,)
        logger.info(
            f"Motion condition enabled: train_types={train_motion_types}, "
            f"eval_type={eval_motion_type}"
        )
        if dataset_motion_map is not None:
            logger.info(
                "Using fixed dataset motion assignments: "
                f"{dataset_motion_map}"
            )
        if eval_motion_type not in train_motion_types:
            logger.warning(
                f"Evaluation uses the {eval_motion_type} adapter, but it is not in "
                f"train_types={train_motion_types}; that adapter will remain at its "
                "seeded initialization unless a checkpoint initializes it."
            )

    for dataset_name, data_config in active_datasets.items():

        split_types = ["train", "test"] if include_test else ["train"]
        for data_split_type in split_types:
            if data_split_type in data_config:
                if data_split_type == "train" and bool(
                    data_config.get("evaluation_only", False)
                ):
                    logger.info(
                        f"Skipping evaluation-only dataset {dataset_name} for training"
                    )
                    continue
                goals_per_obs = int(data_config["goals_per_obs"])
                if data_split_type == "test":
                    goals_per_obs = 4  # standardize testing

                min_dist_cat = data_config.get("distance", {}).get(
                    "min_dist_cat", config.dataset.distance.min_dist_cat
                )
                max_dist_cat = data_config.get("distance", {}).get(
                    "max_dist_cat", config.dataset.distance.max_dist_cat
                )
                len_traj_pred = data_config.get(
                    "len_traj_pred", config.dataset.len_traj_pred
                )
                if data_split_type == "train":
                    split_motion_types = (
                        (dataset_motion_map[dataset_name],)
                        if dataset_motion_map is not None
                        else train_motion_types
                    )
                else:
                    split_motion_types = eval_motion_types

                dataset = TrainingDataset(
                    data_folder=data_config["data_folder"],
                    data_split_folder=data_config[data_split_type],
                    dataset_name=dataset_name,
                    image_size=config.dataset.image_size,
                    min_dist_cat=min_dist_cat,
                    max_dist_cat=max_dist_cat,
                    len_traj_pred=len_traj_pred,
                    context_size=config.dataset.context_size,
                    normalize=config.dataset.normalize,
                    goals_per_obs=goals_per_obs,
                    transform=get_transform(
                        config.dataset.image_size,
                        config.dataset.mean,
                        config.dataset.std,
                    ),
                    action_stats=config.dataset.action_stats,
                    waypoint_spacing=data_config.metric_waypoint_spacing,
                    predefined_index=None,
                    traj_stride=1,
                    precomputed_latent_root=(
                        validated_latents["root"]
                        if data_split_type == "train" and validated_latents
                        else None
                    ),
                    precomputed_latent_cache_size=int(
                        latent_config.get("cache_size", 8)
                    ),
                    precomputed_latent_metadata=(
                        validated_latents["expected_file_metadata"]
                        if data_split_type == "train" and validated_latents
                        else None
                    ),
                    precomputed_latent_records=(
                        validated_latents["manifest_records"][dataset_name]
                        if data_split_type == "train" and validated_latents
                        else None
                    ),
                    motion_condition=motion_config,
                    motion_types=split_motion_types,
                    two_stage_config=(
                        config
                        if data_split_type == "train"
                        and str(config.get("training_stage", "legacy")) != "legacy"
                        else None
                    ),
                    finetune_substage=(
                        finetune_substage if data_split_type == "train" else None
                    ),
                )

                if data_split_type == "train":
                    train_dataset.append(dataset)
                    if use_motion_condition:
                        train_motion_audit.append(
                            (dataset_name, len(dataset), tuple(split_motion_types))
                        )
                else:
                    test_dataset.append(dataset)

                motion_label = (
                    tuple(split_motion_types) if use_motion_condition else "legacy-real"
                )
                print(
                    f"Dataset: {dataset_name} ({data_split_type}), size: {len(dataset)}, "
                    f"input={'precomputed posterior' if dataset.uses_precomputed_latents else 'pixels'}, "
                    f"motion={motion_label}"
                )

    # Combine all datasets from different robots
    if train_motion_audit:
        total_samples = sum(size for _, size, _ in train_motion_audit)
        for dataset_name, size, motion_types in train_motion_audit:
            logger.info(
                f"Training mix: dataset={dataset_name}, motion={motion_types}, "
                f"samples={size}, expected_share={size / total_samples:.2%}"
            )
    print(f"Combining {len(train_dataset)} datasets.")
    train_dataset = ConcatDataset(train_dataset)
    test_dataset = ConcatDataset(test_dataset) if test_dataset else None

    return train_dataset, test_dataset


def prepare_proxy_pretrain_dataset(config: DictConfig):
    """Build the NavAnywhere-only image dataset used by stage-1 training."""
    from navanywhere_latent_cache import validate_navanywhere_latent_cache
    from two_stage_data import NavAnywhereDataset

    if str(config.get("training_stage", "legacy")) != "proxy_pretrain":
        raise ValueError(
            "prepare_proxy_pretrain_dataset requires training_stage=proxy_pretrain"
        )
    datasets_config = config.dataset.datasets
    enabled = [
        str(name)
        for name, dataset_config in datasets_config.items()
        if bool(dataset_config.get("enabled", True))
    ]
    if enabled != ["navanywhere"]:
        raise ValueError(
            "Stage-1 training must enable NavAnywhere only; "
            f"enabled datasets are {enabled}"
        )
    data_config = datasets_config.navanywhere
    root = data_config.get("root", data_config.get("data_folder"))
    if root is None:
        raise ValueError("dataset.datasets.navanywhere.root is required")
    root = os.path.expanduser(str(root))
    if not os.path.isabs(root):
        root = os.path.join(_original_cwd(), root)
    source_ids = data_config.get("source_ids", None)
    if source_ids is not None:
        source_ids = list(source_ids)
    manifest = data_config.get("manifest", None)
    if manifest is not None and str(manifest).strip().lower() in {"", "none", "null"}:
        manifest = None
    elif isinstance(manifest, (str, os.PathLike)):
        manifest = os.path.expanduser(os.fspath(manifest))
        if not os.path.isabs(manifest):
            manifest = os.path.join(_original_cwd(), manifest)
    recipe_config = config.dataset.get("sampling_recipe", {})
    sampling_recipe = recipe_config.get("path", None)
    if sampling_recipe is not None and str(sampling_recipe).strip().lower() in {
        "",
        "none",
        "null",
    }:
        sampling_recipe = None
    elif sampling_recipe is not None:
        sampling_recipe = os.path.expanduser(os.fspath(sampling_recipe))
        if not os.path.isabs(sampling_recipe):
            sampling_recipe = os.path.join(_original_cwd(), sampling_recipe)
        sampling_recipe = os.path.realpath(sampling_recipe)
        if not os.path.isfile(sampling_recipe):
            raise FileNotFoundError(
                f"NavAnywhere sampling recipe does not exist: {sampling_recipe}"
            )
        configured_recipe_sha = recipe_config.get("sha256", None)
        actual_recipe_sha = _sha256_file(sampling_recipe)
        if configured_recipe_sha not in (None, "", "null") and str(
            configured_recipe_sha
        ) != actual_recipe_sha:
            raise ValueError(
                "dataset.sampling_recipe.sha256 does not match the recipe file: "
                f"{configured_recipe_sha!r} != {actual_recipe_sha!r}"
            )
        config.dataset.sampling_recipe.sha256 = actual_recipe_sha
    if sampling_recipe is not None and manifest is not None:
        raise ValueError(
            "Use dataset.sampling_recipe.path or NavAnywhere manifest, not both"
        )
    frame_range = data_config.get("frame_offset_range", None)
    min_offset = int(
        data_config.get(
            "min_frame_offset",
            config.dataset.distance.get("min_dist_cat", -64),
        )
    )
    max_offset = int(
        data_config.get(
            "max_frame_offset",
            config.dataset.distance.get("max_dist_cat", 64),
        )
    )
    if frame_range is not None:
        min_offset, max_offset = map(int, frame_range)
    if (min_offset, max_offset) != (-64, 64):
        raise ValueError(
            "Two-stage NavAnywhere pretraining must retain frame offsets [-64, 64]; "
            f"got [{min_offset}, {max_offset}]"
        )
    action_mode = str(config.get("action_mode", "none"))
    raw_proxy_config = config.get("proxy", {})
    proxy_config = (
        OmegaConf.to_container(raw_proxy_config, resolve=True)
        if OmegaConf.is_config(raw_proxy_config)
        else dict(raw_proxy_config)
    )
    if not isinstance(proxy_config, dict):
        raise TypeError("proxy configuration must be a mapping")
    if not bool(proxy_config.get("use_precomputed_only", True)):
        raise ValueError(
            "Stage-1 training only supports offline proxy caches; "
            "proxy.use_precomputed_only must be true"
        )
    proxy_limit = int(proxy_config.get("max_abs_frame_offset", 8))
    if proxy_limit != 8:
        raise ValueError(
            "Stage-1 proxy eligibility is fixed to abs(frame_offset)<=8"
        )
    if action_mode != "none":
        storage_path = proxy_config.get("storage_path")
        if storage_path is None:
            raise ValueError(
                f"action_mode={action_mode!r} requires proxy.storage_path"
            )
        storage_path = os.path.expanduser(str(storage_path))
        if not os.path.isabs(storage_path):
            storage_path = os.path.join(_original_cwd(), storage_path)
        proxy_config["storage_path"] = storage_path
        proxy_type = str(proxy_config.get("type", action_mode))
        if proxy_type != action_mode:
            raise ValueError(
                f"proxy.type={proxy_type!r} must match action_mode={action_mode!r}"
            )
    latent_config = config.dataset.get("precomputed_latents", {})
    validated_latents = None
    if bool(latent_config.get("enabled", False)):
        if sampling_recipe is None:
            raise ValueError(
                "NavAnywhere precomputed latents require "
                "dataset.sampling_recipe.path so cache/frame identity is frozen"
            )
        if latent_config.get("root", None) in (None, "", "null"):
            raise ValueError(
                "dataset.precomputed_latents.root is required when enabled"
            )
        validated_latents = validate_navanywhere_latent_cache(
            config,
            recipe_path=sampling_recipe,
            launch_dir=_original_cwd(),
        )
        logger.info(
            "NavAnywhere precomputed SD-VAE posteriors enabled: root=%s, "
            "recipe_sha256=%s, worker_lru_trajectories=%d",
            validated_latents["root"],
            validated_latents["recipe_sha256"],
            int(latent_config.get("cache_size", 8)),
        )
    dataset = NavAnywhereDataset(
        root=root,
        source_ids=source_ids,
        manifest=manifest,
        sampling_recipe=sampling_recipe,
        transform=get_transform(
            config.dataset.image_size,
            config.dataset.mean,
            config.dataset.std,
        ),
        context_size=int(config.dataset.context_size),
        goals_per_obs=int(data_config.get("goals_per_obs", 1)),
        min_frame_offset=min_offset,
        max_frame_offset=max_offset,
        action_mode=action_mode,
        proxy=proxy_config,
        proxy_dim=(
            None
            if action_mode == "none"
            else int(proxy_config.get("dim"))
        ),
        proxy_max_abs_frame_offset=proxy_limit,
        strict_loading=bool(proxy_config.get("strict_loading", False)),
        seed=int(config.seed),
        precomputed_latent_root=(
            validated_latents["root"] if validated_latents else None
        ),
        precomputed_latent_cache_size=int(latent_config.get("cache_size", 8)),
        precomputed_latent_metadata=(
            validated_latents["expected_file_metadata"]
            if validated_latents
            else None
        ),
        precomputed_latent_records=(
            validated_latents["manifest_records"] if validated_latents else None
        ),
    )
    logger.info(
        "Stage-1 dataset: NavAnywhere only, samples=%d, action_mode=%s, "
        "offset_range=[%d,%d]",
        len(dataset),
        action_mode,
        min_offset,
        max_offset,
    )
    if dataset.recipe_summary is not None:
        logger.info("NavAnywhere sampling recipe: %s", dataset.recipe_summary)
    return dataset


def prepare_proxy_pretrain_validation_dataset(config: DictConfig):
    """Build the opt-in, trajectory-disjoint pixel validation dataset.

    Stage-1 training may use cached SD-VAE posteriors, but evaluation needs
    pixels for prediction decoding and perceptual scoring.  The validation
    recipe and proxy root are therefore separate from their training peers.
    """

    from navanywhere_recipe import load_sampling_recipe
    from two_stage_data import NavAnywhereDataset

    if str(config.get("training_stage", "legacy")) != "proxy_pretrain":
        raise ValueError(
            "prepare_proxy_pretrain_validation_dataset requires "
            "training_stage=proxy_pretrain"
        )
    validation = config.dataset.get("validation", {})
    if not bool(validation.get("enabled", False)):
        raise ValueError("NavAnywhere Stage-1 validation is not enabled")
    if int(validation.get("batches", 1)) < 1:
        raise ValueError(
            ""
            "dataset.validation.batches must be positive"
        )

    data_config = config.dataset.datasets.navanywhere
    root = data_config.get("root", data_config.get("data_folder"))
    if root is None:
        raise ValueError("dataset.datasets.navanywhere.root is required")
    root = os.path.expanduser(str(root))
    if not os.path.isabs(root):
        root = os.path.join(_original_cwd(), root)

    recipe_config = validation.get("sampling_recipe", {})
    recipe_path = recipe_config.get("path", None)
    if recipe_path is None or str(recipe_path).strip().lower() in {
        "",
        "none",
        "null",
    }:
        raise ValueError("dataset.validation.sampling_recipe.path is required")
    recipe_path = os.path.expanduser(os.fspath(recipe_path))
    if not os.path.isabs(recipe_path):
        recipe_path = os.path.join(_original_cwd(), recipe_path)
    recipe_path = os.path.realpath(recipe_path)
    recipe, _, recipe_sha = load_sampling_recipe(
        recipe_path,
        seed=int(config.seed),
        context_size=int(config.dataset.context_size),
        goals_per_obs=int(data_config.get("goals_per_obs", 1)),
        frame_offset_range=(-64, 64),
    )
    configured_sha = recipe_config.get("sha256", None)
    if configured_sha not in (None, "", "null") and str(configured_sha) != recipe_sha:
        raise ValueError(
            "dataset.validation.sampling_recipe.sha256 does not match the "
            f"recipe file: {configured_sha!r} != {recipe_sha!r}"
        )
    config.dataset.validation.sampling_recipe.sha256 = recipe_sha

    train_recipe_path = config.dataset.get("sampling_recipe", {}).get("path", None)
    if train_recipe_path not in (None, "", "null"):
        train_recipe_path = os.path.expanduser(os.fspath(train_recipe_path))
        if not os.path.isabs(train_recipe_path):
            train_recipe_path = os.path.join(_original_cwd(), train_recipe_path)
        train_recipe_path = os.path.realpath(train_recipe_path)
        train_recipe, _, _ = load_sampling_recipe(
            train_recipe_path,
            seed=int(config.seed),
            context_size=int(config.dataset.context_size),
            goals_per_obs=int(data_config.get("goals_per_obs", 1)),
            frame_offset_range=(-64, 64),
        )
        train_ids = {
            (str(item["source_id"]), str(item["trajectory_id"]))
            for item in train_recipe["trajectories"]
        }
        val_ids = {
            (str(item["source_id"]), str(item["trajectory_id"]))
            for item in recipe["trajectories"]
        }
        overlap = sorted(train_ids & val_ids)
        if overlap:
            raise ValueError(
                "NavAnywhere train/validation recipes overlap by trajectory: "
                f"{overlap[:5]}"
            )

    action_mode = str(config.get("action_mode", "none"))
    raw_proxy = config.get("proxy", {})
    proxy_config = (
        OmegaConf.to_container(raw_proxy, resolve=True)
        if OmegaConf.is_config(raw_proxy)
        else dict(raw_proxy)
    )
    if not isinstance(proxy_config, dict):
        raise TypeError("proxy configuration must be a mapping")
    if action_mode != "none":
        validation_proxy_root = validation.get("proxy_root", None)
        if validation_proxy_root in (None, "", "null"):
            raise ValueError(
                "dataset.validation.proxy_root is required for action-conditioned "
                "Stage-1 validation"
            )
        validation_proxy_root = os.path.expanduser(str(validation_proxy_root))
        if not os.path.isabs(validation_proxy_root):
            validation_proxy_root = os.path.join(
                _original_cwd(), validation_proxy_root
            )
        proxy_config["storage_path"] = validation_proxy_root

    source_ids = data_config.get("source_ids", None)
    if source_ids is not None:
        source_ids = list(source_ids)
    dataset = NavAnywhereDataset(
        root=root,
        source_ids=source_ids,
        sampling_recipe=recipe_path,
        transform=get_transform(
            config.dataset.image_size,
            config.dataset.mean,
            config.dataset.std,
        ),
        context_size=int(config.dataset.context_size),
        goals_per_obs=int(data_config.get("goals_per_obs", 1)),
        min_frame_offset=-64,
        max_frame_offset=64,
        action_mode=action_mode,
        proxy=proxy_config,
        proxy_dim=(None if action_mode == "none" else int(proxy_config["dim"])),
        proxy_max_abs_frame_offset=int(proxy_config.get("max_abs_frame_offset", 8)),
        strict_loading=bool(proxy_config.get("strict_loading", False)),
        seed=int(config.seed),
    )
    logger.info(
        "Stage-1 validation dataset: samples=%d, trajectories=%d, "
        "sources=%d, action_mode=%s, recipe_sha256=%s",
        len(dataset),
        int(recipe["totals"]["trajectories"]),
        int(recipe["totals"]["sources"]),
        action_mode,
        recipe_sha,
    )
    return dataset


def create_dataloader(dataset, config: DictConfig, rank, is_train=True):
    """Create a deterministic, CUDA-safe dataloader for training or testing.

    Two-stage training initializes CUDA before the loader iterator is created.
    Linux's default ``fork`` context would therefore copy CUDA and pinned-host
    allocator state into every worker.  A worker-side Python GC can later try
    to release that inherited state and abort while reinitializing CUDA.  Spawn
    workers start with a clean runtime and never inherit the parent's CUDA
    state.

    Workers intentionally remain non-persistent.  The training loop updates
    the dataset epoch and the loader generator before every iterator is made;
    recreating workers is what propagates both values and makes an in-epoch
    checkpoint replay the same worker RNG streams from the start of the epoch.
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=is_train,
        seed=config.seed,
    )

    num_workers = int(config.training.num_workers)
    worker_options = {}
    if num_workers > 0:
        worker_options["multiprocessing_context"] = "spawn"

    loader_options = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": False,
        "collate_fn": (
            motion_condition_collate
            if bool(
                config.get("motion_condition", {}).get("enabled", False)
            )
            else None
        ),
        **worker_options,
    }
    reference_batch_size = config.training.get("reference_batch_size", None)
    if is_train and reference_batch_size not in (None, "", "null"):
        batch_sampler = ReferenceBatchSampler(
            sampler,
            batch_size=int(config.training.batch_size),
            reference_batch_size=int(reference_batch_size),
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **loader_options,
        )
    else:
        batch_size = config.training.batch_size
        if not is_train:
            eval_batch_size = config.training.get("eval_batch_size", None)
            if eval_batch_size is not None:
                batch_size = eval_batch_size
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=True,
            **loader_options,
        )

    return loader, sampler


