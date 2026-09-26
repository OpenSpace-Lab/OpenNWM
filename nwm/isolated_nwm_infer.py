# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from einops import repeat
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os
import numpy as np
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict
from checkpoint_io import output_identity

import misc
import distributed as dist
from datasets import EvalDataset
from PIL import Image
from hydra_utils import load_experiment_config, merge_configs
from train_utils import (
    setup_tokenizer,
    setup_model,
    print_config,
    create_logger,
    setup_diffusion,
    validate_model_context_sizes,
)
from motion_condition import make_motion_group
from scripts.rollout_actions import body_frame_rollout_actions
from scripts.benchmark_reproducibility import (
    expand_sample_keys,
    samplewise_noise_schedule,
)


def _model_flag(model, name):
    """Read the flag through optional DDP and torch.compile wrappers."""
    current = model
    visited = set()
    while id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, name):
            return bool(getattr(current, name))
        if hasattr(current, "module"):
            current = current.module
        elif hasattr(current, "_orig_mod"):
            current = current._orig_mod
        else:
            break
    return False


def _uses_motion_condition(model):
    return _model_flag(model, "motion_condition_enabled")


def save_image(output_file, img):
    img = img.detach().cpu()
    img = img * 255
    img = img.byte()
    image = Image.fromarray(img.permute(1, 2, 0).numpy(), mode="RGB")

    temporary = f"{output_file}.tmp.{os.getpid()}.png"
    image.save(temporary, format="PNG")
    os.replace(temporary, output_file)


def valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (FileNotFoundError, OSError, ValueError):
        return False


def select_missing(output_dir, idxs, frames, *tensors):
    """Keep original split IDs while removing fully written samples."""
    positions = [
        offset
        for offset, value in enumerate(idxs.reshape(-1))
        if not all(
            valid_image(os.path.join(output_dir, f"id_{int(value.item())}", f"{frame}.png"))
            for frame in frames
        )
    ]
    if not positions:
        return None
    selected = torch.as_tensor(positions, dtype=torch.long)
    return (idxs.index_select(0, selected.to(idxs.device)),) + tuple(
        tensor.index_select(0, selected.to(tensor.device)) for tensor in tensors
    )


def get_dataset_eval(config, dataset_name, eval_type, predefined_index=True):
    # Explicit evaluation entries override experiment/training dataset entries.
    # This keeps held-out split controls (for example a fixed 500-window index)
    # from being shadowed by an older training config stored in a checkpoint run.
    if dataset_name in config.evaluation_datasets:
        data_config = config.evaluation_datasets[dataset_name]
    elif dataset_name in config.dataset.datasets:
        data_config = config.dataset.datasets[dataset_name]
    else:
        raise KeyError(f"Unknown evaluation dataset: {dataset_name}")
    split_name = str(data_config.get("split_name", dataset_name))
    loader_name = str(data_config.get("loader_name", dataset_name))
    if predefined_index and bool(data_config.get("use_predefined_index", True)):
        predefined_index = os.path.join(data_config.test, f"{eval_type}.pkl")
    else:
        predefined_index = None

    predefined_index = data_config.get("predefined_index_path", predefined_index)

    dataset = EvalDataset(
        data_folder=data_config.data_folder,
        data_split_folder=data_config.test,
        dataset_name=loader_name,
        image_size=config.dataset.image_size,
        min_dist_cat=config.eval_distance.eval_min_dist_cat,
        max_dist_cat=config.eval_distance.eval_max_dist_cat,
        len_traj_pred=config.eval_len_traj_pred,
        traj_stride=config.traj_stride,
        context_size=config.eval_context_size,
        normalize=config.dataset.normalize,
        action_stats=config.dataset.action_stats,
        waypoint_spacing=data_config.metric_waypoint_spacing,
        transform=misc.get_transform(
            config.dataset.image_size, config.dataset.mean, config.dataset.std
        ),
        goals_per_obs=4,
        predefined_index=predefined_index,
        traj_names=str(
            data_config.get(
                "rollout_traj_names" if eval_type == "rollout" else "traj_names",
                "rollout_traj_names.txt" if eval_type == "rollout" else "traj_names.txt",
            )
        ),
        motion_condition_enabled=bool(
            config.get("motion_condition", {}).get("enabled", False)
        ),
        wrap_delta_yaw=bool(data_config.get("wrap_delta_yaw", False)),
    )

    return dataset


def resolve_time_horizons(config):
    """Resolve and validate direct-prediction horizons without changing legacy defaults."""

    configured = config.get("time_horizons_seconds")
    if configured is None:
        horizons = [2**index for index in range(int(config.num_sec_eval))]
    else:
        horizons = [int(value) for value in configured]
        if any(float(value) != int(value) for value in configured):
            raise ValueError("time_horizons_seconds must contain integer seconds")
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("time_horizons_seconds must contain positive values")
    if len(horizons) != len(set(horizons)) or horizons != sorted(horizons):
        raise ValueError("time_horizons_seconds must be unique and increasing")
    required_frames = horizons[-1] * int(config.input_fps)
    if required_frames > int(config.eval_len_traj_pred):
        raise ValueError(
            "Direct-prediction horizon exceeds eval_len_traj_pred: "
            f"{horizons[-1]}s * {config.input_fps}fps = {required_frames} frames, "
            f"but eval_len_traj_pred={config.eval_len_traj_pred}"
        )
    return np.asarray(horizons, dtype=np.int64)


@torch.no_grad()
def model_forward_wrapper(
    all_models,
    curr_obs,
    curr_delta,
    num_timesteps,
    latent_size,
    device,
    num_cond,
    num_goals=1,
    rel_t=None,
    progress=False,
    skip_tokenizer=False,
    motion=None,
    motion_type="real",
    sample_keys=None,
    noise_seed=0,
    noise_stream="nwm",
):
    model, diffusion, tokenizer = all_models
    original_nwm = _model_flag(model, "original_nwm")
    x = curr_obs.to(device)
    y = curr_delta.to(device) if curr_delta is not None else None

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        B, T = x.shape[:2]
        model_batch = B * num_goals

        if rel_t is None:
            if num_timesteps is None:
                raise ValueError("num_timesteps is required when rel_t is omitted")
            rel_t = torch.full(
                (model_batch,),
                float(num_timesteps) / 128.0,
                device=device,
            )
        else:
            rel_t = rel_t.to(device).reshape(-1)
            if rel_t.numel() != model_batch:
                raise ValueError(
                    f"rel_t must contain B*num_goals={model_batch} values, "
                    f"got {rel_t.numel()}"
                )

        if not skip_tokenizer:
            x = x.flatten(0, 1)
            x = tokenizer.encode(x).unflatten(0, (B, T))

        x_cond = repeat(x[:, :num_cond], "b t ... -> (b g) t ...", g=num_goals)
        latent_shape = (model_batch, *x.shape[2:])
        initial_noise = None
        step_noises = None
        if sample_keys is None:
            # Keep the historical global RNG position for training/planning
            # callers that have not opted into the benchmark noise contract.
            first_noise = torch.randn(*latent_shape, device=device)
            if original_nwm:
                initial_noise = first_noise
        else:
            expanded_keys = expand_sample_keys(sample_keys, num_goals)
            if len(expanded_keys) != model_batch:
                raise ValueError(
                    f"sample_keys expand to {len(expanded_keys)} rows, expected "
                    f"B*num_goals={model_batch}"
                )
            noise_schedule = samplewise_noise_schedule(
                expanded_keys,
                x.shape[2:],
                draws=diffusion.num_timesteps + 1,
                base_seed=int(noise_seed),
                stream=str(noise_stream),
                device=device,
            )
            initial_noise = noise_schedule[0]
            step_noises = noise_schedule[1:]
        if y is not None:
            if y.ndim == 3:
                y = y.flatten(0, 1)
            elif y.ndim != 2:
                raise ValueError("Action must have shape [B,G,D] or [B*G,D]")
            if y.shape[0] != model_batch:
                raise ValueError(
                    f"Action must contain B*num_goals={model_batch} rows, "
                    f"got {y.shape[0]}"
                )

        if _uses_motion_condition(model):
            if motion is None and y is not None:
                # Navigation inference is deliberately real-action conditioned
                # for every training variant. An empty mapping distinguishes an
                # explicit none request from an omitted legacy argument.
                motion = (
                    {}
                    if motion_type == "none"
                    else make_motion_group(motion_type, y)
                )
            elif motion is not None:
                motion = {
                    key: {
                        "indices": value["indices"].to(device),
                        "values": value["values"].to(device),
                    }
                    for key, value in motion.items()
                }
            model_kwargs = {
                "x_cond": x_cond,
                "rel_t": rel_t,
                "motion": motion,
            }
        else:
            # Preserve the original kwargs contract for legacy continuous and
            # discrete checkpoints, whose forward methods do not accept motion.
            model_kwargs = {"y": y, "x_cond": x_cond, "rel_t": rel_t}
        samples = diffusion.p_sample_loop(
            model.forward,
            latent_shape,
            noise=initial_noise,
            model_kwargs=model_kwargs,
            progress=progress,
            device=device,
            step_noises=step_noises,
            clip_denoised=False,
        )

        if not skip_tokenizer:
            samples = tokenizer.decode(
                samples, denormalize=True
            )  # Unnormalize to [0, 1] range
            return samples  # Already in [0, 1] range after unnormalization

        return samples


def generate_rollout_efficient(
    config,
    output_dir,
    rollout_fps,
    idxs,
    all_models,
    obs_image,
    gt_image,
    delta,
    num_cond,
    device,
    dataset_name,
):
    """
    Efficient autoregressive rollout that operates in latent space.

    This version encodes observations once and performs the entire rollout
    in latent space, only decoding when needed for visualization.
    This is significantly faster than encoding/decoding at each step.
    """
    model, diffusion, tokenizer = all_models
    rollout_stride = config.input_fps // rollout_fps
    gt_image = gt_image[:, rollout_stride - 1 :: rollout_stride]
    delta = delta.unflatten(1, (-1, rollout_stride)).sum(2)

    if config.gt:
        # Ground truth mode - just visualize GT frames
        for i in range(gt_image.shape[1]):
            x_pred_pixels = gt_image[:, i].clone().to(device)
            x_pred_pixels = misc.get_unnormalize(
                config.dataset.mean, config.dataset.std
            )(x_pred_pixels)
            visualize_preds(output_dir, idxs, i, x_pred_pixels)
    else:
        # Encode initial observations to latent space
        curr_obs = obs_image.clone().to(device)
        B, T = curr_obs.shape[:2]
        curr_obs_latents = tokenizer.encode(curr_obs.flatten(0, 1)).unflatten(0, (B, T))

        # Perform rollout in latent space
        for i in range(gt_image.shape[1]):
            curr_delta = delta[:, i : i + 1].to(device)

            # Generate next frame in latent space
            x_pred_latents = model_forward_wrapper(
                all_models,
                curr_obs_latents,
                curr_delta,
                rollout_stride,
                config.latent_size,
                num_cond=num_cond,
                num_goals=1,
                device=device,
                motion_type="real",
                skip_tokenizer=True,  # Work directly with latents
                sample_keys=idxs,
                noise_seed=config.seed,
                noise_stream=f"{dataset_name}/rollout_{rollout_fps}fps/step={i}",
            )

            # Decode for visualization
            x_pred_pixels = tokenizer.decode(x_pred_latents, denormalize=True)

            # Update latent observation window
            x_pred_latents = x_pred_latents.unsqueeze(1)
            curr_obs_latents = torch.cat((curr_obs_latents, x_pred_latents), dim=1)
            curr_obs_latents = curr_obs_latents[:, 1:]  # Remove first observation

            # Visualize the decoded prediction
            visualize_preds(output_dir, idxs, i, x_pred_pixels)


def generate_rollout(
    config,
    output_dir,
    rollout_fps,
    idxs,
    all_models,
    obs_image,
    gt_image,
    delta,
    num_cond,
    device,
    dataset_name,
):
    """
    Produce an autoregressive rollout video at a downsampled frame rate.

    Starting from the context frames, the function repeatedly predicts the *next*
    time-step image (or copies ground-truth when `config.gt` is set), feeds
    that prediction back into the sliding window of observations, and writes each
    visualised frame to disk.
    In short: **step-by-step future synthesis, so error can accumulate and be
    observed over an entire trajectory**.

    Note: Consider using generate_rollout_efficient for better performance.
    """
    rollout_stride = config.input_fps // rollout_fps
    gt_image = gt_image[:, rollout_stride - 1 :: rollout_stride]
    delta = body_frame_rollout_actions(
        delta, rollout_stride,
        legacy_stats=None if config.get("motion_condition", {}).get("enabled", False)
        else config.dataset.action_stats,
    )
    # Diagnostic override only; benchmark runs use the unmodified real action.
    delta = delta * float(config.get("diagnostic_rollout_action_scale", 1.0))
    curr_obs = obs_image.clone().to(device)

    for i in range(gt_image.shape[1]):
        curr_delta = delta[:, i : i + 1].to(device)
        if config.gt:
            x_pred_pixels = gt_image[:, i].clone().to(device)
            x_pred_pixels = misc.get_unnormalize(
                config.dataset.mean, config.dataset.std
            )(x_pred_pixels)
        else:
            x_pred_pixels = model_forward_wrapper(
                all_models,
                curr_obs,
                curr_delta,
                rollout_stride,
                config.latent_size,
                num_cond=num_cond,
                num_goals=1,
                device=device,
                motion_type="real",
                sample_keys=idxs,
                noise_seed=config.seed,
                noise_stream=f"{dataset_name}/rollout_{rollout_fps}fps/step={i}",
            )

        # model_forward_wrapper returns display-space pixels in [0, 1], while
        # EvalDataset observations are normalized before entering the VAE.
        # Restore that input contract before recursively feeding predictions
        # back into the context window.
        x_pred_context = misc.get_normalize(
            config.dataset.mean, config.dataset.std
        )(x_pred_pixels)
        curr_obs = torch.cat(
            (curr_obs, x_pred_context.unsqueeze(1)), dim=1
        )  # append current prediction
        curr_obs = curr_obs[
            :, 1:
        ]  # remove first observation => moving window of context
        visualize_preds(
            output_dir,
            idxs,
            i,
            x_pred_pixels,
        )


def generate_time(
    config,
    output_dir,
    idxs,
    all_models,
    obs_image,
    gt_output,
    delta,
    secs,
    num_cond,
    device,
    dataset_name,
):
    """
    Predict future *snapshots* at a list of absolute times given in seconds.

    For every timestep within the requested horizon (hz is the input fps), the function sums all motion deltas from the
    start up to that instant, performs a single model call (or fetches the
    ground-truth frame), and saves the resulting image.
    In short: **one-shot forecasts at specific future times, without feeding
    predictions back into the model**.

    when config.gt is set, the function will use the ground truth frames as the future frames
    """
    eval_timesteps = [sec * config.input_fps for sec in secs]
    for sec, timestep in zip(secs, eval_timesteps):
        selected = select_missing(output_dir, idxs, (sec,), obs_image, gt_output, delta)
        if selected is None:
            continue
        selected_ids, selected_obs, selected_gt, selected_delta = selected
        curr_delta = selected_delta[:, :timestep].sum(dim=1, keepdim=True)
        if config.gt:
            x_pred_pixels = selected_gt[:, timestep - 1].clone().to(device)
            x_pred_pixels = misc.get_unnormalize(
                config.dataset.mean, config.dataset.std
            )(x_pred_pixels)
        else:
            x_pred_pixels = model_forward_wrapper(
                all_models,
                selected_obs,
                curr_delta,
                timestep,
                config.latent_size,
                num_cond=num_cond,
                num_goals=1,
                device=device,
                motion_type="real",
                sample_keys=selected_ids,
                noise_seed=config.seed,
                noise_stream=f"{dataset_name}/time/{int(sec)}s",
            )
        visualize_preds(
            output_dir,
            selected_ids,
            sec,
            x_pred_pixels,
        )


def visualize_preds(output_dir, idxs, sec, x_pred_pixels):
    for batch_idx, sample_idx in enumerate(idxs.reshape(-1)):
        sample_idx = int(sample_idx.item())
        sample_folder = os.path.join(output_dir, f"id_{sample_idx}")
        os.makedirs(sample_folder, exist_ok=True)
        image_file = os.path.join(sample_folder, f"{sec}.png")
        save_image(image_file, x_pred_pixels[batch_idx])


@torch.no_grad()
@hydra.main(version_base=None, config_path="conf", config_name="infer_config")
def main(config: DictConfig):
    # Restore original working directory
    os.chdir(get_original_cwd())

    # Setup distributed environment
    _, rank, device, _ = dist.init_distributed()
    # Create logger
    logger = create_logger(os.getcwd())

    # Print configuration
    if rank == 0:
        print_config(config)

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    # Keep stochastic diffusion samples reproducible across checkpoints while
    # still assigning a distinct random stream to every distributed rank.
    misc.seed_everything(config.seed * num_tasks + global_rank)

    # Validate required parameters
    if config.output_dir is None:
        raise ValueError("output_dir must be specified")
    if config.exp_dir is None and not config.gt:
        raise ValueError("exp_dir must be specified")
    if config.datasets_to_eval is None:
        raise ValueError("datasets must be specified")
    if config.eval_type is None:
        raise ValueError("eval_type must be specified (must be 'time' or 'rollout')")

    # Process config values
    rollout_fps_values = config.rollout_fps_values
    dataset_names = config.datasets_to_eval

    # Output directory setup
    if config.gt:
        save_output_dir = os.path.join(config.output_dir, "gt")
    elif config.prediction_dir is not None:
        save_output_dir = os.path.abspath(config.prediction_dir)
    else:
        save_output_dir = os.path.join(get_original_cwd(), config.exp_dir, "results")
        save_output_dir = save_output_dir + f"_{config.ckp}"

    os.makedirs(save_output_dir, exist_ok=True)

    # Load experiment configuration
    exp_config = None
    if os.path.exists(config.exp_dir):
        exp_config = load_experiment_config(config.exp_dir)
        if exp_config:
            logger.info(
                f"Loaded experiment config from {config.exp_dir}/.hydra/config.yaml"
            )
            config = merge_configs(exp_config, config)
            logger.info("Experiment configuration merged with inference config")
    else:
        raise ValueError(f"Experiment directory {config.exp_dir} does not exist")

    requested_diffusion_steps = config.get("eval_diffusion_steps")
    if requested_diffusion_steps is not None:
        requested_diffusion_steps = int(requested_diffusion_steps)
        if requested_diffusion_steps <= 0:
            raise ValueError("eval_diffusion_steps must be positive")
        with open_dict(config):
            config.model.diffusion.eval_timestep_respacing = requested_diffusion_steps
        logger.info(
            "Pinned evaluation diffusion steps to %d", requested_diffusion_steps
        )

    output_identity(save_output_dir, config,
                    checkpoint=None if config.gt else f"{config.exp_dir}/checkpoints/{config.ckp}.pth.tar",
                    name=str(config.eval_type))

    validate_model_context_sizes(config, "eval_context_size")

    # Get number of context frames
    num_cond = config.dataset.context_size

    # Load model if not generating ground truth
    model_lst = (None, None, None)
    if not config.gt:
        logger.info("Setting up model and tokenizer...")

        # Setup tokenizer using helper function
        vae = setup_tokenizer(config, device)
        # Setup model using helper function
        model = setup_model(config, device)
        # Load checkpoint
        checkpoint_path = f"{config.exp_dir}/checkpoints/{config.ckp}.pth.tar"
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint {checkpoint_path} does not exist")
        ckp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        logger.info(
            f"Loading model weights: {model.load_state_dict(ckp['ema'], strict=True)}"
        )
        model.eval()
        model.to(device)
        if config.get("torch_compile", False):
            model = torch.compile(model)

        # Create diffusion model
        diffusion = setup_diffusion(config, for_eval=True, device=device)

        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=False
        )
        model_lst = (model, diffusion, vae)
        logger.info(
            f"Model setup complete. Parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

    # Print configuration
    if rank == 0:
        print_config(config)

    # Set latent size
    with open_dict(config):
        config.latent_size = config.model.generator.input_size

    # Load datasets
    datasets = {}
    for dataset_name in dataset_names:
        logger.info(f"Loading dataset: {dataset_name}")
        dataset_val = get_dataset_eval(
            config, dataset_name, config.eval_type, predefined_index=True
        )

        expected_full_count = config.get("eval_expected_full_count")
        if expected_full_count is not None and len(dataset_val) != int(expected_full_count):
            raise RuntimeError(
                f"Evaluation split size changed for {dataset_name}/{config.eval_type}: "
                f"expected {expected_full_count}, got {len(dataset_val)}"
            )

        sample_indices = config.get("eval_sample_indices")
        if sample_indices is not None:
            sample_indices = [int(index) for index in sample_indices]
            if len(sample_indices) != len(set(sample_indices)):
                raise ValueError("eval_sample_indices must be unique")
            invalid = [
                index for index in sample_indices if index < 0 or index >= len(dataset_val)
            ]
            if invalid:
                raise IndexError(
                    f"eval_sample_indices are outside [0, {len(dataset_val)}): {invalid}"
                )
            dataset_val = torch.utils.data.Subset(dataset_val, sample_indices)
            logger.info(
                "Selected %d fixed samples for %s/%s: %s",
                len(sample_indices),
                dataset_name,
                config.eval_type,
                sample_indices,
            )

        # Exact strided sharding avoids DistributedSampler padding and works for
        # every GPU count. Sample-keyed noise makes rank ownership irrelevant.
        sampler_val = list(range(global_rank, len(dataset_val), num_tasks))

        curr_data_loader = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
        )
        datasets[dataset_name] = curr_data_loader
        logger.info(f"Dataset {dataset_name} loaded, size: {len(dataset_val)}")

    print_freq = 1
    header = "Inference: "
    metric_logger = dist.MetricLogger(delimiter="  ")

    # Run inference for each dataset
    for dataset_name in dataset_names:
        dataset_save_output_dir = os.path.join(save_output_dir, dataset_name)
        os.makedirs(dataset_save_output_dir, exist_ok=True)
        curr_data_loader = datasets[dataset_name]
        logger.info(f"Running inference on {dataset_name}...")

        for data_iter_step, (idxs, obs_image, gt_image, delta) in enumerate(
            metric_logger.log_every(curr_data_loader, print_freq, header)
        ):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                obs_image = obs_image[:, -num_cond:].to(device)
                gt_image = gt_image.to(device)

                # NOTE Evaluation is done only in forward direction (negative min bound offset is not used)
                if config.eval_type == "rollout":
                    for rollout_fps in rollout_fps_values:
                        # Following experiment in the paper; rollout is evaluated at 1 and 4 fps
                        curr_rollout_output_dir = os.path.join(
                            dataset_save_output_dir, f"rollout_{rollout_fps}fps"
                        )
                        os.makedirs(curr_rollout_output_dir, exist_ok=True)
                        selected = select_missing(
                            curr_rollout_output_dir,
                            idxs,
                            range(gt_image.shape[1] // (int(config.input_fps) // int(rollout_fps))),
                            obs_image,
                            gt_image,
                            delta,
                        )
                        if selected is None:
                            continue
                        selected_ids, selected_obs, selected_gt, selected_delta = selected

                        if bool(config.get("single_image_context", False)) and not config.gt:
                            if bool(config.get("save_initial_image", False)):
                                initial_dir = os.path.join(config.output_dir, "initial", dataset_name)
                                initial_pixels = misc.get_unnormalize(
                                    config.dataset.mean, config.dataset.std
                                )(selected_obs[:, -1])
                                for offset, sample_index in enumerate(selected_ids.reshape(-1)):
                                    initial_path = os.path.join(
                                        initial_dir, f"id_{int(sample_index.item())}.png"
                                    )
                                    os.makedirs(initial_dir, exist_ok=True)
                                    save_image(initial_path, initial_pixels[offset])
                            # Match the official single-image interactive demo:
                            # repeat the current frame to fill the model's context.
                            selected_obs = selected_obs[:, -1:].expand(
                                -1, num_cond, -1, -1, -1
                            ).contiguous()

                        # Use efficient rollout if configured (default: True for better performance)
                        use_efficient_rollout = getattr(
                            config, "use_efficient_rollout", True
                        )
                        rollout_fn = (
                            generate_rollout_efficient
                            if use_efficient_rollout
                            else generate_rollout
                        )

                        rollout_fn(
                            config,
                            curr_rollout_output_dir,
                            rollout_fps,
                            selected_ids,
                            model_lst,
                            selected_obs,
                            selected_gt,
                            selected_delta,
                            num_cond,
                            device,
                            dataset_name,
                        )

                elif config.eval_type == "time":
                    secs = resolve_time_horizons(config)
                    curr_time_output_dir = os.path.join(dataset_save_output_dir, "time")
                    os.makedirs(curr_time_output_dir, exist_ok=True)
                    generate_time(
                        config,
                        curr_time_output_dir,
                        idxs,
                        model_lst,
                        obs_image,
                        gt_image,
                        delta,
                        secs,
                        num_cond,
                        device,
                        dataset_name,
                    )
                else:
                    raise ValueError(
                        f"Unknown eval_type: {config.eval_type}, must be 'time' or 'rollout'"
                    )
            if torch.cuda.is_available():
                logger.info(
                    "eval_progress dataset=%s batch=%d/%d samples=%d peak_memory_gib=%.2f",
                    dataset_name,
                    data_iter_step + 1,
                    len(curr_data_loader),
                    len(idxs),
                    torch.cuda.max_memory_allocated(device) / 1024**3,
                )

    logger.info(f"Inference completed. Results saved to {save_output_dir}")


if __name__ == "__main__":
    main()
