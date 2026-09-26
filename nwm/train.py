# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

import torch
import wandb
from datetime import datetime
import json

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import matplotlib

matplotlib.use("Agg")
from copy import deepcopy
from time import time
import os
import gc
import hydra
from omegaconf import DictConfig, OmegaConf

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from distributed import init_distributed
from train_utils import (
    setup_tokenizer,
    setup_model,
    setup_optimizer,
    setup_scheduler,
    requires_grad,
    print_config,
    create_logger,
    cleanup,
    save_checkpoint,
    evaluate,
    train_step,
    load_checkpoint,
    setup_diffusion,
)
from data_utils import (
    create_dataloader,
    prepare_datasets,
)
import hydra_utils  # noqa: F401 — registers OmegaConf resolvers
from misc import seed_everything, get_unnormalize
from two_stage_checkpoint import capture_rng_state, restore_rng_state
from checkpoint_io import log_metrics
from two_stage_training import _set_loader_epoch_seed, _set_dataset_epoch

#################################################################################
#                                  Training Loop                                #
#################################################################################


def initialize_wandb(config, experiment_dir):
    """Keep legacy logging optional; Go2 requires a persistent online run."""
    options = {}
    required = config.training.get("wandb_required", False)
    state_path = os.path.join(experiment_dir, "wandb_state.json")
    if required:
        if config.training.get("from_checkpoint"):
            with open(state_path) as stream:
                saved = json.load(stream)
            options.update(id=saved["id"], resume="must")
        options["mode"] = "online"
    run = wandb.init(
        project=config.training.get("wandb_project", "nwm-training"),
        entity=config.training.get("wandb_entity", None),
        name=config.training.get("run_name", None),
        notes=config.training.get("wandb_notes", None),
        tags=config.training.get("wandb_tags", []),
        config=OmegaConf.to_container(config, resolve=False),
        dir=experiment_dir,
        group=config.training.get("wandb_group", None),
        **options,
    )
    if required:
        if run is None or run.settings.mode != "online":
            raise RuntimeError("Go2 requires an online WandB run")
        temporary = state_path + ".tmp"
        with open(temporary, "w") as stream:
            json.dump({"id": run.id, "url": run.url, "path": run.path,
                       "status": "running"}, stream, indent=2)
        os.replace(temporary, state_path)
    return run


@hydra.main(version_base=None, config_path="conf", config_name="nwm")
def main(config: DictConfig):
    """
    Trains a new CDiT model using Hydra for configuration.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    # Initialize distributed; get local GPU index and create a proper torch.device
    _, rank, local_gpu, _ = init_distributed()
    torch_device = torch.device(f"cuda:{local_gpu}")
    seed = config.seed * dist.get_world_size() + rank
    seed_everything(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Print config for debugging
    if dist.get_rank() == 0:
        print_config(config)

    # Use Hydra's current working directory as experiment directory
    # Hydra has already created the timestamped directory based on run_name
    experiment_dir = os.getcwd()

    if rank == 0:
        logger = create_logger(experiment_dir)
        logger.info(f"Using Hydra experiment directory: {experiment_dir}")

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize WandB logging (only on rank 0)
        wandb_run_name = None
        if config.training.get("wandb_enabled", False):
            try:
                wandb_state_path = os.path.join(experiment_dir, "wandb_state.json")
                wandb_run_name = initialize_wandb(config, experiment_dir).name
                logger.info(
                    f"WandB logging initialized with run name: {wandb_run_name}"
                )
            except Exception as e:
                if config.training.get("wandb_required", False):
                    # Go2 uses one GPU. Other ranks are terminated by torchrun
                    # if a caller explicitly launches a distributed variant.
                    raise RuntimeError("Required online WandB initialization failed") from e
                logger.warning(
                    f"Failed to initialize WandB: {e}. Continuing without WandB logging."
                )

        # Create experiment note with wandb run name, date, and wandb_notes
        current_date_for_note = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_note = {
            "date": current_date_for_note,
            "wandb_run_name": wandb_run_name,
            "wandb_notes": config.training.get("wandb_notes", None),
            "experiment_dir": experiment_dir,
            "config_name": config.get("_target_", "unknown"),
            "model_type": config.model.generator.get("_target_", "unknown"),
        }

        experiment_note_path = os.path.join(experiment_dir, "experiment_note.json")
        with open(experiment_note_path, "w") as f:
            json.dump(experiment_note, f, indent=2)
        logger.info(f"Experiment note saved to {experiment_note_path}")
    else:
        logger = create_logger(None)

    # New two-stage experiments are opt-in.  Keeping the branch here leaves the
    # complete legacy construction, checkpoint, evaluation, and planning path
    # below unchanged when training_stage is absent (or explicitly legacy).
    if str(config.get("training_stage", "legacy")) != "legacy":
        from two_stage_training import run_two_stage_training

        run_two_stage_training(
            config,
            torch_device,
            rank,
            local_gpu,
            experiment_dir,
            logger,
        )
        if rank == 0 and config.training.get("wandb_enabled", False):
            try:
                wandb.finish()
                if config.training.get("wandb_required", False):
                    with open(wandb_state_path) as stream:
                        saved_wandb = json.load(stream)
                    saved_wandb["status"] = "sdk_finished"
                    with open(wandb_state_path, "w") as stream:
                        json.dump(saved_wandb, stream, indent=2)
            except Exception as exc:
                if config.training.get("wandb_required", False):
                    raise
                logger.warning(f"Failed to finish WandB run: {exc}")
        cleanup()
        return

    # Create model using the utility functions
    tokenizer = setup_tokenizer(config, torch_device)
    tokenizer.eval()
    latent_size = config.model.generator.input_size
    assert config.dataset.image_size % 8 == 0, (
        "Image size must be divisible by 8 (for the VAE encoder)."
    )

    model = setup_model(config, torch_device)
    ema = deepcopy(model).to(
        torch_device
    )  # Create an EMA of the model for use after training
    for p in ema.parameters():
        p.detach_()
    requires_grad(ema, False)

    # Setup optimizer using Hydra instantiation
    opt = setup_optimizer(config, model.parameters())
    scheduler = setup_scheduler(config, opt)

    bfloat_enable = bool(config.bfloat16)
    scaler = None
    if bfloat_enable:
        scaler = torch.amp.GradScaler()

    # Load existing checkpoint
    start_epoch, train_steps, resume_state = load_checkpoint(
        checkpoint_dir,
        config,
        model,
        ema,
        opt,
        scheduler=scheduler,
        bfloat_enable=bfloat_enable,
        scaler=scaler if bfloat_enable else None,
        return_runtime=True,
    )

    # ~40% speedup but might leads to worse performance depending on pytorch version
    if config.torch_compile:
        model = torch.compile(model)

    # Wrap model for DDP using the local GPU index; keep a torch.device for tensor moves
    model = DDP(
        model,
        device_ids=[local_gpu],
        # Single-type and none batches intentionally leave motion adapters out
        # of the graph, so their unused parameters must be discovered by DDP.
        find_unused_parameters=bool(
            config.get("motion_condition", {}).get("enabled", False)
        ),
    )
    diffusion = setup_diffusion(config, for_eval=False, device=torch_device)
    logger.info(f"CDiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets
    train_dataset, test_dataset = prepare_datasets(config)
    loader, sampler = create_dataloader(train_dataset, config, rank, is_train=True)
    eval_loader, eval_sampler = create_dataloader(
        test_dataset, config, rank, is_train=False
    )

    logger.info(f"Dataset contains {len(train_dataset):,} images")
    logger.info(f"Eval dataset contains {len(test_dataset):,} images")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    best_score = float(resume_state.get("best_score", float("inf")))
    epoch = start_epoch
    batch_in_epoch = int(resume_state.get("batch_in_epoch", 0))
    epoch_start_rng = resume_state.get("epoch_start_rng")
    max_train_steps = config.get("max_train_steps", None)
    stop_training = max_train_steps is not None and train_steps >= int(max_train_steps)

    def checkpoint(best=False):
        local_state = {"rng_state": capture_rng_state(), "epoch_start_rng": epoch_start_rng,
                       "batch_in_epoch": batch_in_epoch}
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, local_state)
        if rank == 0:
            path = save_checkpoint(model, ema, opt, config, epoch, train_steps,
                                   bfloat_enable, scheduler, scaler, checkpoint_dir,
                                   runtime_states=states, best_score=best_score, best=best)
            logger.info("Saved checkpoint to %s", path)
        dist.barrier()

    logger.info(f"Training for {config.epochs} epochs...")
    for epoch in range(start_epoch, config.epochs):
        if stop_training:
            break
        sampler.set_epoch(epoch)
        _set_dataset_epoch(train_dataset, epoch)
        _set_loader_epoch_seed(loader, int(config.seed), rank, epoch)
        if resume_state.get("epoch_start_rng") is not None:
            restore_rng_state(resume_state["epoch_start_rng"])
        epoch_start_rng = capture_rng_state()
        iterator = iter(loader)
        cursor = int(resume_state.get("batch_in_epoch", 0))
        for _ in range(cursor):
            next(iterator)
        if resume_state.get("rng_state") is not None:
            restore_rng_state(resume_state["rng_state"])
        resume_state = {}
        batch_in_epoch = cursor
        logger.info(f"Beginning epoch {epoch} at batch {cursor}...")
        for batch_idx, batch in enumerate(iterator, start=cursor):
            batch_in_epoch = batch_idx + 1
            loss = train_step(
                model,
                ema,
                diffusion,
                tokenizer,
                batch,
                torch_device,
                opt,
                scheduler,
                bfloat_enable,
                scaler,
                config,
            )

            # Log loss values:
            running_loss += loss
            log_steps += 1
            train_steps += 1
            if train_steps % config.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                batch_video = (
                    batch.get("video", batch.get("posterior_mean"))
                    if isinstance(batch, dict)
                    else batch[0]
                )
                samples_per_sec = (
                    dist.get_world_size()
                    * batch_video.shape[0]
                    * (batch_video.shape[1] - config.dataset.context_size)
                    * steps_per_sec
                )
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=torch_device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, Samples/Sec: {samples_per_sec:.2f}"
                )
                if config.get("log_cuda_memory", False) and rank == 0:
                    peak_allocated = torch.cuda.max_memory_allocated(torch_device)
                    peak_reserved = torch.cuda.max_memory_reserved(torch_device)
                    logger.info(
                        "CUDA peak memory: "
                        f"allocated={peak_allocated / 1024**3:.2f} GiB, "
                        f"reserved={peak_reserved / 1024**3:.2f} GiB"
                    )
                gc.collect()

                # Log to WandB (only on rank 0)
                if rank == 0 and config.training.get("wandb_enabled", False):
                    try:
                        wandb.log(
                            {
                                "train/loss": avg_loss,
                                "train/learning_rate": opt.param_groups[0]["lr"],
                                "train/steps_per_sec": steps_per_sec,
                                "train/samples_per_sec": samples_per_sec,
                                "train/step": train_steps,
                                "epoch": epoch,
                            },
                            step=train_steps,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log to WandB: {e}")
                if rank == 0:
                    log_metrics(experiment_dir, {"step": train_steps, "train/loss": avg_loss,
                                "train/learning_rate": opt.param_groups[0]["lr"]})
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % int(config.ckpt_every) == 0:
                checkpoint()
            at_limit = max_train_steps is not None and train_steps >= int(max_train_steps)
            at_final_epoch = epoch + 1 == int(config.epochs) and batch_in_epoch == len(loader)
            if ((int(config.eval_every) > 0 and train_steps % int(config.eval_every) == 0)
                or (train_steps == 1 and config.get("eval_at_first_step", True))
                or at_limit or at_final_epoch):
                eval_start_time = time()
                save_dir = os.path.join(experiment_dir, "viz", str(train_steps))
                # Evaluation uses a fixed sample/noise stream for comparable
                # curves. Restore every RNG afterward so it cannot alter the
                # subsequent training trajectory.
                training_rng_state = capture_rng_state()
                try:
                    sim_score = evaluate(
                        ema,
                        tokenizer,
                        diffusion,
                        eval_loader,
                        rank,
                        latent_size,
                        torch_device,
                        save_dir,
                        config.seed,
                        bfloat_enable,
                        config.dataset.context_size,
                        get_unnormalize(config.dataset.mean, config.dataset.std),
                        num_batches=int(config.training.get("eval_num_batches", 1)),
                    )
                finally:
                    restore_rng_state(training_rng_state)
                dist.barrier()
                eval_end_time = time()
                eval_time = eval_end_time - eval_start_time
                logger.info(
                    f"(step={train_steps:07d}) Perceptual Loss: {sim_score:.4f}, Eval Time: {eval_time:.2f}"
                )

                if float(sim_score) < best_score:
                    best_score = float(sim_score)
                    checkpoint(best=True)
                if rank == 0:
                    log_metrics(experiment_dir, {"step": train_steps, "epoch": epoch,
                                "eval/perceptual_loss": float(sim_score), "best/perceptual_loss": best_score})

                # Log to WandB (only on rank 0)
                if rank == 0 and config.training.get("wandb_enabled", False):
                    try:
                        wandb.log(
                            {
                                "eval/perceptual_loss": sim_score,
                                "eval/eval_time": eval_time,
                                "eval/step": train_steps,
                                "epoch": epoch,
                            },
                            step=train_steps,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to log evaluation metrics to WandB: {e}"
                        )

            if at_limit:
                stop_training = True
                break
        if stop_training:
            break

    checkpoint()
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")

    # Finish WandB run (only on rank 0)
    if rank == 0 and config.training.get("wandb_enabled", False):
        try:
            wandb.finish()
            logger.info("WandB logging finished")
        except Exception as e:
            logger.warning(f"Failed to finish WandB run: {e}")

    cleanup()


if __name__ == "__main__":
    main()
