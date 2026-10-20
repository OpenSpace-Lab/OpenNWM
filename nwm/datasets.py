# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

import logging
import random
import numpy as np
import torch
import os
from collections import OrderedDict
from PIL import Image
from typing import Any, Dict, Optional, Tuple
import pickle
import tqdm
from torch.utils.data import Dataset
from misc import (
    angle_difference,
    get_data_path,
    get_delta_np,
    normalize_data,
    to_local_coords,
)
from hydra.utils import get_original_cwd
from typing import List, Sequence, Union, Any
from torchvision import transforms
from omegaconf import DictConfig
import json
import base64

from motion_condition import (
    OfflineMotionStore,
    motion_input_dim,
    motion_offset_mask,
    validate_motion_types,
)

logger = logging.getLogger(__name__)


def load_image(path):
    """Load an image and return a copy, ensuring file handle is properly closed."""
    with Image.open(path) as img:
        # Convert to RGB and copy to ensure data is loaded into memory
        return img.convert("RGB").copy()


class NumpySerializedList:
    def __init__(self, lst: list):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        print(
            "Serializing {} elements to byte tensors and concatenating them all ...".format(
                len(lst)
            )
        )
        self._lst = [_serialize(x) for x in lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = np.cumsum(self._addr)
        self._lst = np.concatenate(self._lst)
        print("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr])
        return pickle.loads(bytes)


class TorchSerializedList(NumpySerializedList):
    def __init__(self, lst: list):
        super().__init__(lst)
        self._addr = torch.from_numpy(self._addr)
        self._lst = torch.from_numpy(self._lst)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        return pickle.loads(bytes)


class DualNormalizationTransform:
    """
    Transform class that applies dual normalization to images:
    - One normalization for the base tokenizer
    - Another normalization for DINOv2 (always ImageNet stats)

    Returns a dictionary with both normalized versions.
    """

    def __init__(
        self,
        image_size: int,
        dinov2_image_size: int,
        use_augmentation: bool = False,
        tokenizer_mean: List[float] = [0.5, 0.5, 0.5],
        tokenizer_std: List[float] = [0.5, 0.5, 0.5],
        dinov2_mean: List[float] = [0.485, 0.456, 0.406],
        dinov2_std: List[float] = [0.229, 0.224, 0.225],
    ):
        self.image_size = image_size
        self.dinov2_image_size = dinov2_image_size
        self.use_augmentation = use_augmentation

        # Determine the larger size to avoid upsampling
        max_size = max(image_size, dinov2_image_size)

        # Common transforms applied once (including random augmentations)
        self.common_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    max_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                lambda image: image.clamp(0, 1),
                transforms.RandomCrop(max_size)
                if use_augmentation
                else transforms.CenterCrop(max_size),
                transforms.RandomHorizontalFlip(0.5)
                if use_augmentation
                else lambda image: image,
            ]
        )

        # Tokenizer-specific resize (if needed) and normalization
        self.tokenizer_post_transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size, interpolation=transforms.InterpolationMode.BICUBIC
                )
                if image_size != max_size
                else lambda x: x,
                transforms.Normalize(mean=tokenizer_mean, std=tokenizer_std),
            ]
        )

        # DINOv2-specific resize (if needed) and normalization
        self.dinov2_post_transform = transforms.Compose(
            [
                transforms.Resize(
                    dinov2_image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
                if dinov2_image_size != max_size
                else lambda x: x,
                transforms.Normalize(mean=dinov2_mean, std=dinov2_std),
            ]
        )

    def __call__(self, image):
        """Apply dual normalization to the image."""
        # Apply common transforms once (including random augmentations)
        augmented_image = self.common_transforms(image)

        # Apply specific resizing (if needed) and normalizations to the same augmented image
        tokenizer_image = self.tokenizer_post_transform(augmented_image.clone())
        dinov2_image = self.dinov2_post_transform(augmented_image.clone())

        # Return the tokenizer-normalized image as the main output
        # Store the DINOv2 version as additional metadata
        return {"image": tokenizer_image, "dinov2_image": dinov2_image}


class BaseDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int,
        context_size: int,
        transform: object,
        action_stats: dict,
        waypoint_spacing: float,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        self.data_folder = os.path.join(get_original_cwd(), data_folder)
        self.data_split_folder = os.path.join(get_original_cwd(), data_split_folder)
        self.dataset_name = dataset_name
        self.goals_per_obs = goals_per_obs
        self.missing_trajectories = []
        self.rebuild_index = os.environ.get("NWM_REBUILD_INDEX", "0") == "1"

        # Split manifests are source-controlled, but generated indexes and
        # audit files can be large and belong with the dataset on NAS.
        index_root = os.environ.get("NWM_INDEX_ROOT")
        if index_root:
            split_name = os.path.basename(os.path.normpath(self.data_split_folder))
            self.index_cache_dir = os.path.join(
                index_root, self.dataset_name, split_name
            )
            os.makedirs(self.index_cache_dir, exist_ok=True)
        else:
            self.index_cache_dir = self.data_split_folder

        traj_names_file = os.path.join(self.data_split_folder, traj_names)
        if os.path.isfile(traj_names_file):
            with open(traj_names_file, "r") as f:
                file_lines = f.read()
            self.traj_names = file_lines.split("\n")
            if "" in self.traj_names:
                self.traj_names.remove("")
        elif predefined_index:
            # Some official evaluation-only splits (notably Go Stanford) ship
            # only a predefined pickle. Derive the scan manifest from that
            # immutable index instead of requiring a redundant text file.
            with open(predefined_index, "rb") as f:
                raw_index_to_data = pickle.load(f)
            self.traj_names = sorted({entry[0] for entry in raw_index_to_data})
            logger.info(
                "Derived %d trajectory names from predefined index %s",
                len(self.traj_names),
                predefined_index,
            )
        else:
            raise FileNotFoundError(
                f"Trajectory manifest does not exist: {traj_names_file}"
            )

        self.existing_traj_names = []
        training_index_cache = os.path.join(
            self.index_cache_dir,
            f"dataset_dist_{min_dist_cat}_to_{max_dist_cat}_n{context_size}_len_traj_pred_{len_traj_pred}.pkl",
        )
        if (
            not predefined_index
            and not self.rebuild_index
            and os.path.isfile(training_index_cache)
        ):
            logger.info(
                f"Found cached dataset index for {self.dataset_name}; skipping trajectory scan"
            )
        else:
            # Scan for existing trajectories before building or filtering an index.
            logger.info(f"Scanning for existing trajectories in {self.dataset_name}...")
            for traj_name in tqdm.tqdm(self.traj_names):
                traj_path = os.path.join(self.data_folder, traj_name, "traj_data.pkl")
                if os.path.exists(traj_path):
                    self.existing_traj_names.append(traj_name)
                else:
                    self.missing_trajectories.append(traj_name)

            logger.info(
                f"Found {len(self.existing_traj_names)}/{len(self.traj_names)} trajectories in {self.dataset_name}"
            )

            # Save missing trajectories to a file
            if self.missing_trajectories:
                missing_trajs_file = os.path.join(
                    self.index_cache_dir,
                    f"missing_trajectories_{self.dataset_name}.txt",
                )
                with open(missing_trajs_file, "w") as f:
                    for traj in self.missing_trajectories:
                        f.write(f"{traj}\n")
                logger.info(
                    f"Saved {len(self.missing_trajectories)} missing trajectories to {missing_trajs_file}"
                )

        self.image_size = image_size
        self.distance_categories = list(range(min_dist_cat, max_dist_cat + 1))
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.len_traj_pred = len_traj_pred
        self.traj_stride = traj_stride

        self.context_size = context_size
        self.normalize = normalize

        # use this index to retrieve the dataset name from the data_config.yaml
        self.transform = transform
        self._load_index(predefined_index)
        self.ACTION_STATS = {}
        for key in action_stats:
            self.ACTION_STATS[key] = np.expand_dims(action_stats[key], axis=0)
        self.WAYPOINT_SPACING = waypoint_spacing

    def _is_index_valid(
        self, traj_name, curr_state_time, min_offset_bound, max_offset_bound
    ):
        traj = self._get_trajectory(traj_name)
        min_bound = curr_state_time + min_offset_bound
        max_bound = curr_state_time + max_offset_bound
        is_index_valid = min_bound >= 0 and max_bound < len(traj["position"])
        return is_index_valid

    def _load_index(self, predefined_index) -> None:
        """
        Generates a list of tuples of (obs_traj_name, curr_state_time, min_offset_bound, max_offset_bound) for each observation in the dataset
        """
        if predefined_index:
            logger.info(
                f"****** Using a predefined evaluation index... {predefined_index}******"
            )
            with open(predefined_index, "rb") as f:
                raw_index_to_data = pickle.load(f)
            index_to_data = [
                x for x in raw_index_to_data if x[0] not in self.missing_trajectories
            ]
            invalid_indices = [x for x in index_to_data if not self._is_index_valid(*x)]
            self.index_to_data = TorchSerializedList(
                [x for x in index_to_data if x not in invalid_indices]
            )

            logger.info(
                f"Found {len(self.index_to_data)} / {len(raw_index_to_data)} valid indices"
            )
            # Save invalid indices to a file
            invalid_indices_file = os.path.join(
                self.index_cache_dir,
                f"invalid_indices_in_{os.path.splitext(os.path.basename(predefined_index))[0]}.txt",
            )
            with open(invalid_indices_file, "w") as f:
                for idx in invalid_indices:
                    f.write(f"{idx[0]}, {idx[1]}, {idx[2]}, {idx[3]}\n")
            logger.info(
                f"Saved {len(invalid_indices)} invalid indices to {invalid_indices_file}"
            )
        else:
            logger.info("****** Evaluating from NON PREDEFINED index... ******")
            index_to_data_path = os.path.join(
                self.index_cache_dir,
                f"dataset_dist_{self.min_dist_cat}_to_{self.max_dist_cat}_n{self.context_size}_len_traj_pred_{self.len_traj_pred}.pkl",
            )

            if not self.rebuild_index and os.path.isfile(index_to_data_path):
                logger.info(f"Loading cached dataset index from {index_to_data_path}")
                with open(index_to_data_path, "rb") as f:
                    self.index_to_data, self.goals_index = pickle.load(f)
                return

            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _build_index(self, use_tqdm: bool = False):
        """
        NOTE: This function is only used when pre-defined index is not provided; meaning that it is only called once before the training, and doesnt used in evaluation since index is already provided.

        Build an index consisting of tuples (obs_traj_name, curr_state_time, min_offset_bound, max_offset_bound)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(
            self.existing_traj_names, disable=not use_tqdm, dynamic_ncols=True
        ):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])
            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size - 1
            end_time = traj_len - self.len_traj_pred
            for curr_time in range(begin_time, end_time, self.traj_stride):
                max_goal_distance = min(self.max_dist_cat, traj_len - curr_time - 1)
                min_goal_distance = max(self.min_dist_cat, -curr_time)
                samples_index.append(
                    (traj_name, curr_time, min_goal_distance, max_goal_distance)
                )

        return TorchSerializedList(samples_index), TorchSerializedList(goals_index)

    def _get_trajectory(self, trajectory_name):
        traj_path = os.path.join(self.data_folder, trajectory_name, "traj_data.pkl")
        with open(traj_path, "rb") as f:
            traj_data = pickle.load(f)
        if self.dataset_name == "tum_rgbd":
            from scripts.prepare_tum_camera_heading import heading_from_metadata
            traj_data["yaw"] = heading_from_metadata(
                os.path.join(self.data_folder, trajectory_name, "frame_metadata.jsonl")
            ).copy()
        for k, v in traj_data.items():
            traj_data[k] = v.astype("float", copy=False)
        return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1
        yaw = traj_data["yaw"][start_index:end_index]
        positions = traj_data["position"][start_index:end_index]
        goal_pos = traj_data["position"][goal_time]
        goal_yaw = traj_data["yaw"][goal_time]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            raise ValueError("is used?")
            # const_len = self.len_traj_pred + 1 - yaw.shape[0]
            # yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            # positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        # Navigation motion is planar even when a source trajectory stores xyz.
        waypoints_pos = to_local_coords(positions, positions[0], yaw[0])[..., :2]
        waypoints_yaw = angle_difference(yaw[0], yaw)
        actions = np.concatenate([waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)
        actions = actions[1:]

        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])[..., :2]
        goal_yaw = angle_difference(yaw[0], goal_yaw)

        if self.normalize:
            actions[:, :2] /= self.WAYPOINT_SPACING
            goal_pos[:, :2] /= self.WAYPOINT_SPACING

        goal_pos = np.concatenate([goal_pos, goal_yaw.reshape(-1, 1)], axis=-1)
        return actions, goal_pos


class TrainingDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int,
        context_size: int,
        transform: object,
        action_stats: dict,
        waypoint_spacing: float,
        traj_names: str = "traj_names.txt",
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
        precomputed_latent_root: Optional[str] = None,
        precomputed_latent_cache_size: int = 8,
        precomputed_latent_metadata: Optional[Dict[str, Any]] = None,
        precomputed_latent_records: Optional[Dict[str, Dict[str, Any]]] = None,
        motion_condition: Optional[Any] = None,
        motion_types: Optional[Sequence[str]] = None,
        two_stage_config: Optional[Any] = None,
        finetune_substage: Optional[str] = None,
    ):
        # Evaluation on Jetson Python 3.8 must not import training-only
        # dataclasses that require a newer Python runtime.
        from two_stage_data import OfflineProxyStore

        super().__init__(
            data_folder,
            data_split_folder,
            dataset_name,
            image_size,
            min_dist_cat,
            max_dist_cat,
            len_traj_pred,
            traj_stride,
            context_size,
            transform,
            action_stats,
            waypoint_spacing,
            traj_names,
            normalize,
            predefined_index,
            goals_per_obs,
        )
        self.precomputed_latent_root = precomputed_latent_root
        self.precomputed_latent_cache_size = int(precomputed_latent_cache_size)
        self.precomputed_latent_metadata = precomputed_latent_metadata
        self.precomputed_latent_records = precomputed_latent_records or {}
        self._latent_trajectory_cache = OrderedDict()
        self.motion_condition = motion_condition
        self.two_stage_config = two_stage_config
        self.training_stage = str(
            (two_stage_config or {}).get("training_stage", "legacy")
        )
        self.finetune_config = (two_stage_config or {}).get("finetune", {})
        raw_finetune_scheme = (
            str(self.finetune_config.get("scheme", "reset"))
            .strip()
            .lower()
            .replace("-", "_")
        )
        self.finetune_scheme = {
            "align": "embedding_align",
            "alignment": "embedding_align",
            "embedding_alignment": "embedding_align",
            "d": "state_conditioned_controller",
            "state_controller": "state_conditioned_controller",
            "latent_state_controller": "state_conditioned_controller",
        }.get(raw_finetune_scheme, raw_finetune_scheme)
        self.finetune_substage = finetune_substage
        self.alignment_warmup = bool(
            self.training_stage == "real_finetune"
            and self.finetune_scheme
            in {"embedding_align", "state_conditioned_controller"}
            and finetune_substage == "warmup"
        )
        self.alignment_proxy_store: Optional[OfflineProxyStore] = None
        self.alignment_latent_dim = 0
        self.alignment_max_abs_frame_offset = 8
        if (
            self.training_stage == "real_finetune"
            and self.finetune_scheme
            in {"embedding_align", "state_conditioned_controller"}
        ):
            proxy_config = (two_stage_config or {}).get("proxy", {})
            storage_path = proxy_config.get("storage_path")
            if storage_path is None:
                raise ValueError(
                    f"{self.finetune_scheme} requires proxy.storage_path for the "
                    "offline latent teacher cache"
                )
            storage_path = os.path.expanduser(str(storage_path))
            if not os.path.isabs(storage_path):
                try:
                    original_cwd = get_original_cwd()
                except ValueError:
                    original_cwd = os.getcwd()
                storage_path = os.path.join(original_cwd, storage_path)
            self.alignment_latent_dim = int(proxy_config.get("dim", 0))
            if self.alignment_latent_dim < 1:
                raise ValueError(
                    f"{self.finetune_scheme} requires a positive proxy.dim"
                )
            self.alignment_max_abs_frame_offset = int(
                proxy_config.get("max_abs_frame_offset", 8)
            )
            if self.alignment_max_abs_frame_offset != 8:
                raise ValueError(
                    "Latent teacher eligibility is fixed to "
                    "abs(frame_offset)<=8"
                )
            self.alignment_proxy_store = OfflineProxyStore(
                storage_path=storage_path,
                proxy_type="latent",
                dim=self.alignment_latent_dim,
                strict_loading=bool(proxy_config.get("strict_loading", True)),
                file_pattern=str(
                    proxy_config.get(
                        "file_pattern", "{source_id}/{trajectory_id}"
                    )
                ),
                pairs_key=str(proxy_config.get("pairs_key", "frame_pairs")),
                values_key=str(proxy_config.get("values_key", "motion")),
                validity_key=proxy_config.get("validity_key"),
                cache_size=int(proxy_config.get("cache_size", 8)),
                max_abs_frame_offset=self.alignment_max_abs_frame_offset,
            )
        self.motion_condition_enabled = bool(
            motion_condition is not None
            and motion_condition.get("enabled", False)
        )
        self.motion_types: tuple[str, ...] = ()
        self.motion_sampling_strategy = "round_robin"
        self.offline_motion_stores: Dict[str, OfflineMotionStore] = {}
        self.motion_max_frame_offsets: Dict[str, int] = {}

        if self.motion_condition_enabled:
            available_types = validate_motion_types(motion_condition)
            self.motion_types = tuple(
                str(value)
                for value in (
                    motion_types
                    if motion_types is not None
                    else validate_motion_types(motion_condition, training=True)
                )
            )
            if not self.motion_types or not set(self.motion_types).issubset(
                available_types
            ):
                raise ValueError(
                    f"Invalid dataset motion types {self.motion_types}; "
                    f"available={available_types}"
                )
            self.motion_sampling_strategy = str(
                motion_condition.get("sampling_strategy", "round_robin")
            )
            if self.motion_sampling_strategy != "round_robin":
                raise ValueError(
                    "Only deterministic round_robin motion sampling is supported"
                )

            for motion_type in ("geometry", "latent"):
                if motion_type not in self.motion_types:
                    continue
                type_config = motion_condition[motion_type]
                max_frame_offset = type_config.get("max_frame_offset")
                if max_frame_offset is not None:
                    max_frame_offset = int(max_frame_offset)
                    if max_frame_offset < 0:
                        raise ValueError(
                            f"motion_condition.{motion_type}.max_frame_offset "
                            "must be non-negative"
                        )
                    self.motion_max_frame_offsets[motion_type] = max_frame_offset
                offline_config = type_config.get("offline", {})
                root = offline_config.get("root")
                if root is None:
                    raise ValueError(
                        f"motion_condition.{motion_type}.offline.root is required "
                        f"when training/evaluating with {motion_type} motion"
                    )
                root = os.path.expanduser(str(root))
                if not os.path.isabs(root):
                    root = os.path.join(get_original_cwd(), root)
                self.offline_motion_stores[motion_type] = OfflineMotionStore(
                    root=root,
                    dataset_name=self.dataset_name,
                    motion_type=motion_type,
                    input_dim=motion_input_dim(motion_condition, motion_type),
                    file_pattern=str(
                        offline_config.get(
                            "file_pattern", "{dataset_name}/{trajectory_name}.pt"
                        )
                    ),
                    pairs_key=str(offline_config.get("pairs_key", "frame_pairs")),
                    values_key=str(offline_config.get("values_key", "motion")),
                    cache_size=int(offline_config.get("cache_size", 8)),
                    strict_metadata=bool(
                        offline_config.get("strict_metadata", True)
                    ),
                    translation_unit=(
                        "waypoint_spacing_units" if self.normalize else "meters"
                    ),
                )

        if self.precomputed_latent_root is not None:
            if self.precomputed_latent_cache_size < 1:
                raise ValueError(
                    "precomputed_latent_cache_size must be at least 1 when "
                    "precomputed latent loading is enabled"
                )
            self.precomputed_latent_dataset_root = os.path.realpath(
                os.path.join(self.precomputed_latent_root, self.dataset_name)
            )
        else:
            self.precomputed_latent_dataset_root = None

    @property
    def uses_precomputed_latents(self) -> bool:
        return self.precomputed_latent_root is not None

    def _latent_path(self, trajectory_name: str) -> str:
        """Resolve a trajectory cache path without allowing path traversal."""
        path = os.path.realpath(
            os.path.join(
                self.precomputed_latent_dataset_root, f"{trajectory_name}.pt"
            )
        )
        root_prefix = self.precomputed_latent_dataset_root + os.sep
        if not path.startswith(root_prefix):
            raise ValueError(
                f"Unsafe trajectory name in latent cache: {trajectory_name!r}"
            )
        return path

    def _validate_latent_trajectory(
        self, payload: Any, trajectory_name: str, path: str
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(f"Latent cache is not a dictionary: {path}")

        required_keys = {
            "schema_version",
            "format",
            "dataset_name",
            "trajectory_name",
            "frame_indices",
            "posterior_mean",
            "posterior_logvar",
            "metadata",
        }
        missing_keys = required_keys.difference(payload)
        if missing_keys:
            raise ValueError(
                f"Latent cache {path} is missing keys: {sorted(missing_keys)}"
            )
        if payload["schema_version"] != 1:
            raise ValueError(
                f"Unsupported latent cache schema_version in {path}: "
                f"{payload['schema_version']!r} (expected 1)"
            )
        if payload["format"] != "sd_vae_posterior_stats":
            raise ValueError(
                f"Unsupported latent cache format in {path}: {payload['format']!r}"
            )
        if payload["dataset_name"] != self.dataset_name:
            raise ValueError(
                f"Latent cache dataset mismatch in {path}: "
                f"{payload['dataset_name']!r} != {self.dataset_name!r}"
            )
        if payload["trajectory_name"] != trajectory_name:
            raise ValueError(
                f"Latent cache trajectory mismatch in {path}: "
                f"{payload['trajectory_name']!r} != {trajectory_name!r}"
            )

        frame_indices = payload["frame_indices"]
        mean = payload["posterior_mean"]
        logvar = payload["posterior_logvar"]
        if not isinstance(frame_indices, torch.Tensor):
            raise TypeError(f"frame_indices must be a tensor in {path}")
        if frame_indices.dtype != torch.int64 or frame_indices.ndim != 1:
            raise ValueError(
                f"frame_indices must be int64 [N] in {path}; got "
                f"dtype={frame_indices.dtype}, shape={tuple(frame_indices.shape)}"
            )
        if frame_indices.numel() == 0:
            raise ValueError(f"Latent cache has no frames: {path}")
        if torch.any(frame_indices < 0):
            raise ValueError(f"Latent cache contains negative frame indices: {path}")
        if frame_indices.numel() > 1 and not torch.all(
            frame_indices[1:] > frame_indices[:-1]
        ):
            raise ValueError(
                f"frame_indices must be strictly increasing with no duplicates: {path}"
            )
        if not torch.equal(
            frame_indices, torch.arange(frame_indices.numel(), dtype=torch.int64)
        ):
            raise ValueError(
                f"frame_indices must cover the contiguous authoritative range "
                f"0..N-1 in {path}"
            )

        latent_hw = int(self.image_size) // 8
        expected_shape = (frame_indices.numel(), 4, latent_hw, latent_hw)
        for name, tensor in (("posterior_mean", mean), ("posterior_logvar", logvar)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor in {path}")
            if tensor.device.type != "cpu":
                raise ValueError(f"{name} must be stored on CPU in {path}")
            if tensor.dtype != torch.bfloat16:
                raise ValueError(
                    f"{name} must be bfloat16 in {path}; got {tensor.dtype}"
                )
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name} has shape {tuple(tensor.shape)} in {path}; "
                    f"expected {expected_shape}"
                )
        file_metadata = payload["metadata"]
        if not isinstance(file_metadata, dict):
            raise TypeError(f"metadata must be a dictionary in {path}")
        if file_metadata.get("num_frames") != frame_indices.numel():
            raise ValueError(
                f"metadata.num_frames mismatch in {path}: "
                f"{file_metadata.get('num_frames')!r} != {frame_indices.numel()}"
            )
        manifest_record = self.precomputed_latent_records.get(trajectory_name)
        if manifest_record is None:
            raise ValueError(
                f"Trajectory {self.dataset_name}/{trajectory_name} is absent "
                "from the validated latent manifest"
            )
        if manifest_record.get("num_frames") != frame_indices.numel():
            raise ValueError(
                f"Manifest/file frame count mismatch in {path}: "
                f"{manifest_record.get('num_frames')!r} != {frame_indices.numel()}"
            )
        expected_metadata = self.precomputed_latent_metadata or {}
        # These fields define the numerical meaning of the cached tensors. The
        # source section is trajectory-specific and is validated by the frame
        # index/name checks above.
        for key in (
            "scaling_factor",
            "image_size",
            "storage_dtype",
            "compute_dtype",
            "vae_batch_size",
        ):
            if key not in file_metadata:
                raise ValueError(f"metadata.{key} is missing in {path}")
            if key in expected_metadata and file_metadata[key] != expected_metadata[key]:
                raise ValueError(
                    f"metadata.{key} mismatch in {path}: "
                    f"{file_metadata[key]!r} != {expected_metadata[key]!r}"
                )

        return payload

    def _get_latent_trajectory(self, trajectory_name: str) -> Dict[str, Any]:
        """Load and validate one trajectory, with a small per-worker CPU LRU."""
        cached = self._latent_trajectory_cache.pop(trajectory_name, None)
        if cached is not None:
            self._latent_trajectory_cache[trajectory_name] = cached
            return cached

        path = self._latent_path(trajectory_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Precomputed latent cache is incomplete; missing {path}"
            )
        try:
            payload = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load latent cache {path}: {exc}") from exc
        payload = self._validate_latent_trajectory(payload, trajectory_name, path)

        self._latent_trajectory_cache[trajectory_name] = payload
        while len(self._latent_trajectory_cache) > self.precomputed_latent_cache_size:
            self._latent_trajectory_cache.popitem(last=False)
        return payload

    def _get_precomputed_posteriors(
        self, trajectory_name: str, frame_times: Sequence[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        payload = self._get_latent_trajectory(trajectory_name)
        frame_indices = payload["frame_indices"]
        requested = torch.as_tensor(frame_times, dtype=torch.int64)
        locations = torch.searchsorted(frame_indices, requested)

        in_range = locations < frame_indices.numel()
        matches = torch.zeros_like(in_range)
        matches[in_range] = frame_indices[locations[in_range]] == requested[in_range]
        if not torch.all(matches):
            missing = requested[~matches].tolist()
            raise IndexError(
                f"Precomputed latent cache for {self.dataset_name}/{trajectory_name} "
                f"does not contain requested frame(s): {missing}"
            )

        selected_mean = payload["posterior_mean"].index_select(0, locations)
        selected_logvar = payload["posterior_logvar"].index_select(0, locations)
        # Touch and validate only the frames used by this sample. A full finite
        # scan here would defeat mmap by paging an entire trajectory into RAM on
        # every LRU miss; extraction verification plus _SUCCESS covers the file.
        if not torch.isfinite(selected_mean).all() or not torch.isfinite(
            selected_logvar
        ).all():
            raise ValueError(
                f"Precomputed posterior contains non-finite values for "
                f"{self.dataset_name}/{trajectory_name}, frames={list(frame_times)}"
            )
        return selected_mean, selected_logvar

    def _motion_type_for_index(self, index: int) -> str:
        # Stable assignment makes proxy mixtures reproducible across workers and
        # epochs; DistributedSampler still shuffles observation order.
        return self.motion_types[int(index) % len(self.motion_types)]

    def _get_motion(
        self,
        motion_type: str,
        trajectory_name: str,
        current_frame: int,
        target_frames: Sequence[int],
    ) -> Optional[torch.Tensor]:
        if motion_type == "none":
            return None
        if motion_type == "real":
            trajectory = self._get_trajectory(trajectory_name)
            _, goal_pose = self._compute_actions(
                trajectory, current_frame, np.asarray(target_frames)
            )
            return torch.as_tensor(goal_pose, dtype=torch.float32)
        return self.offline_motion_stores[motion_type].get(
            trajectory_name, current_frame, target_frames
        )

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            f_curr = str(f_curr)
            curr_time, min_goal_dist, max_goal_dist = (
                int(curr_time),
                int(min_goal_dist),
                int(max_goal_dist),
            )

            sample_min, sample_max = min_goal_dist, max_goal_dist
            if self.alignment_warmup:
                local_limit = self.alignment_max_abs_frame_offset
                sample_min = max(sample_min, -local_limit)
                sample_max = min(sample_max, local_limit)
                if sample_min > sample_max:
                    raise ValueError(
                        f"No local alignment goal for {self.dataset_name}/{f_curr} "
                        f"frame={curr_time}, available=[{min_goal_dist},{max_goal_dist}], "
                        f"required=[{-local_limit},{local_limit}]"
                    )
            goal_offset = np.random.randint(
                sample_min, sample_max + 1, size=(self.goals_per_obs)
            )
            goal_time = (curr_time + goal_offset).astype("int")
            rel_time = (goal_offset).astype("float") / (
                128.0
            )  # TODO: refactor, currently a fixed const

            context_times = list(
                range(curr_time - self.context_size + 1, curr_time + 1)
            )
            all_frame_times = context_times + goal_time.tolist()
            if self.uses_precomputed_latents:
                posterior_mean, posterior_logvar = self._get_precomputed_posteriors(
                    f_curr, all_frame_times
                )
            else:
                context = [(f_curr, t) for t in all_frame_times]
                obs_image = torch.stack(
                    [
                        self.transform(
                            load_image(get_data_path(self.data_folder, f, t))
                        )
                        for f, t in context
                    ]
                )

            if self.motion_condition_enabled:
                motion_type = self._motion_type_for_index(i)
                sample = {
                    "k": torch.as_tensor(rel_time, dtype=torch.float32),
                    "motion_type": motion_type,
                }
                if self.training_stage != "legacy":
                    sample["frame_offset"] = torch.as_tensor(
                        goal_offset, dtype=torch.int64
                    )
                if self.uses_precomputed_latents:
                    sample.update(
                        {
                            "posterior_mean": posterior_mean,
                            "posterior_logvar": posterior_logvar,
                        }
                    )
                else:
                    sample["video"] = torch.as_tensor(
                        obs_image, dtype=torch.float32
                    )

                motion_mask = None
                max_frame_offset = self.motion_max_frame_offsets.get(motion_type)
                if max_frame_offset is not None:
                    motion_mask = motion_offset_mask(goal_offset, max_frame_offset)
                    selected_targets = goal_time[motion_mask.numpy()].tolist()
                    selected_motion = (
                        self._get_motion(
                            motion_type, f_curr, curr_time, selected_targets
                        )
                        if selected_targets
                        else None
                    )
                    motion = torch.zeros(
                        (
                            self.goals_per_obs,
                            motion_input_dim(self.motion_condition, motion_type),
                        ),
                        dtype=torch.float32,
                    )
                    if selected_motion is not None:
                        motion[motion_mask] = selected_motion
                    sample["motion_mask"] = motion_mask
                else:
                    motion = self._get_motion(
                        motion_type, f_curr, curr_time, goal_time.tolist()
                    )
                if motion is not None:
                    sample["motion"] = motion
                if self.alignment_proxy_store is not None:
                    latent = torch.zeros(
                        (self.goals_per_obs, self.alignment_latent_dim),
                        dtype=torch.float32,
                    )
                    latent_eligible = motion_offset_mask(
                        goal_offset, self.alignment_max_abs_frame_offset
                    )
                    latent_found = torch.zeros(
                        self.goals_per_obs, dtype=torch.bool
                    )
                    latent_invalid = torch.zeros_like(latent_found)
                    latent_valid = torch.zeros_like(latent_found)
                    # The exact integer offset gates I/O. Long-range examples
                    # still carry real action and diffusion loss, but never
                    # touch the DreamDojo cache.
                    for goal_index in torch.nonzero(
                        latent_eligible, as_tuple=False
                    ).flatten().tolist():
                        result = self.alignment_proxy_store.lookup(
                            self.dataset_name,
                            f_curr,
                            curr_time,
                            int(goal_time[goal_index]),
                        )
                        latent_found[goal_index] = result.found
                        latent_invalid[goal_index] = result.invalid
                        latent_valid[goal_index] = result.valid
                        if result.valid:
                            latent[goal_index] = result.proxy_action
                    sample.update(
                        teacher_latent=latent,
                        latent_eligible=latent_eligible,
                        latent_found=latent_found,
                        latent_invalid=latent_invalid,
                        latent_valid=latent_valid,
                    )
                return sample

            # Legacy real-action tuple path, kept for old configs/checkpoints.
            curr_traj_data = self._get_trajectory(f_curr)
            _, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
            goal_pos[:, :2] = normalize_data(goal_pos[:, :2], self.ACTION_STATS)

            if self.uses_precomputed_latents:
                # Four elements are an explicit marker for train_step. Keeping
                # the batch dimension first also preserves train.py's throughput
                # accounting without special collation or extra copies.
                return (
                    posterior_mean,
                    posterior_logvar,
                    torch.as_tensor(goal_pos, dtype=torch.float32),
                    torch.as_tensor(rel_time, dtype=torch.float32),
                )
            return (
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
                torch.as_tensor(rel_time, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            if self.training_stage != "legacy":
                raise
            raise Exception(e)


class EvalDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int,
        context_size: int,
        transform: object,
        action_stats: dict,
        waypoint_spacing: float,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
        motion_condition_enabled: bool = False,
        wrap_delta_yaw: bool = False,
    ):
        super().__init__(
            data_folder,
            data_split_folder,
            dataset_name,
            image_size,
            min_dist_cat,
            max_dist_cat,
            len_traj_pred,
            traj_stride,
            context_size,
            transform,
            action_stats,
            waypoint_spacing,
            traj_names,
            normalize,
            predefined_index,
            goals_per_obs,
        )
        self.motion_condition_enabled = bool(motion_condition_enabled)
        self.wrap_delta_yaw = bool(wrap_delta_yaw)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            # Goal min max bound offset is not used in EvalDataset
            f_curr, curr_time, _, _ = self.index_to_data[i]
            f_curr = str(f_curr)
            curr_time = int(curr_time)

            context_times = list(
                range(curr_time - self.context_size + 1, curr_time + 1)
            )
            pred_times = list(range(curr_time + 1, curr_time + self.len_traj_pred + 1))

            context = [(f_curr, t) for t in context_times]
            pred = [(f_curr, t) for t in pred_times]

            obs_image = torch.stack(
                [
                    self.transform(load_image(get_data_path(self.data_folder, f, t)))
                    for f, t in context
                ]
            )
            pred_image = torch.stack(
                [
                    self.transform(load_image(get_data_path(self.data_folder, f, t)))
                    for f, t in pred
                ]
            )

            # Compute actions
            curr_traj_data = self._get_trajectory(f_curr)
            actions, _ = self._compute_actions(
                curr_traj_data, curr_time, np.array([curr_time + 1])
            )  # last argument is dummy goal
            # New adapters own normalization, while legacy checkpoints keep the
            # historical pre-normalized action contract.
            if not self.motion_condition_enabled:
                actions[:, :2] = normalize_data(actions[:, :2], self.ACTION_STATS)
            delta = get_delta_np(actions, wrap_yaw=self.wrap_delta_yaw)

            return (
                torch.tensor([i], dtype=torch.float32),  # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(pred_image, dtype=torch.float32),
                torch.as_tensor(delta, dtype=torch.float32),
            )
        except Exception as e:
            logger.error(f"Exception in {self.dataset_name}", e)
            raise Exception(e)


class TrajectoryEvalDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int,
        context_size: int,
        transform: object,
        action_stats: dict,
        waypoint_spacing: float,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(
            data_folder,
            data_split_folder,
            dataset_name,
            image_size,
            min_dist_cat,
            max_dist_cat,
            len_traj_pred,
            traj_stride,
            context_size,
            transform,
            action_stats,
            waypoint_spacing,
            traj_names,
            normalize,
            predefined_index,
            goals_per_obs,
        )

    def _sample_goal(self, trajectory_name, curr_time, min_goal_dist, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1)
        goal_time = curr_time + int(goal_offset)
        return trajectory_name, goal_time, False

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]

            f_goal, goal_time, _ = self._sample_goal(
                f_curr, curr_time, min_goal_dist, max_goal_dist
            )

            context_times = list(
                range(curr_time - self.context_size + 1, curr_time + 1)
            )
            context = [(f_curr, t) for t in context_times]

            obs_image = torch.stack(
                [
                    self.transform(load_image(get_data_path(self.data_folder, f, t)))
                    for f, t in context
                ]
            )
            goal_image = self.transform(
                load_image(get_data_path(self.data_folder, f_goal, goal_time))
            ).unsqueeze(0)

            curr_traj_data = self._get_trajectory(f_curr)
            actions, goal_pos = self._compute_actions(
                curr_traj_data, curr_time, np.array([goal_time])
            )

            return (
                torch.tensor([i], dtype=torch.float32),  # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_image, dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)


