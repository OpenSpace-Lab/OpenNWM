"""Navigation-specific latent action model training variants.

This module intentionally lives alongside, rather than inside, the original LAM
implementation.  In particular, :class:`PixelActionLAM` runs the unmodified
``LatentActionModel`` pixel path and only adds an action prediction head.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from pathlib import Path

import torch
import torch.nn.functional as F
from lam.model import LAM
from lam.modules.blocks import SpatioTemporalTransformer, SpatioTransformer, patchify
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

OptimizerCallable = Callable[[Iterable[nn.Parameter]], Optimizer]
_MIXED_SOURCE_COUNT = 16
_MIXED_MIN_OFFSET = -8
_MIXED_MAX_OFFSET = 8


def _log_sampling_statistics(module: LightningModule, batch: Mapping[str, Tensor]) -> None:
    """Log the realized per-rank mixed-data composition every 100 steps."""

    if module.global_step % 100 or "dataset_id" not in batch or "frame_offset" not in batch:
        return
    dataset_ids = batch["dataset_id"].reshape(-1)
    offsets = batch["frame_offset"].reshape(-1)
    batch_size = max(dataset_ids.numel(), 1)
    source_count = _MIXED_SOURCE_COUNT
    trainer = getattr(module, "_trainer", None)
    source_specs = getattr(
        getattr(trainer, "datamodule", None), "source_specs", None
    )
    if source_specs is not None:
        source_count = max(source_count, len(source_specs))
    metrics = {
        f"sampling/source_{source_id}_fraction": (dataset_ids == source_id).sum()
        / batch_size
        for source_id in range(source_count)
    }
    metrics.update(
        {
            f"sampling/offset_{offset:+d}_fraction": (offsets == offset).sum()
            / batch_size
            for offset in range(_MIXED_MIN_OFFSET, _MIXED_MAX_OFFSET + 1)
        }
    )
    action_mask = batch.get("action_mask")
    if action_mask is not None:
        metrics["sampling/action_fraction"] = action_mask.float().mean()
    module.log_dict(
        metrics,
        logger=True,
        prog_bar=False,
        on_step=True,
        on_epoch=False,
        sync_dist=True,
    )


def _optimizer_with_schedule(
    optimizer_factory: OptimizerCallable,
    parameters: Iterable[nn.Parameter],
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
):
    optimizer = optimizer_factory(parameter for parameter in parameters if parameter.requires_grad)
    if total_steps <= 0:
        return optimizer
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("scheduler_warmup_steps must be in [0, scheduler_total_steps)")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("scheduler_min_lr_ratio must be between zero and one")

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0 / warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = LambdaLR(optimizer, lr_lambda=factor)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }


def _kl_loss(outputs: Mapping[str, Tensor]) -> Tensor:
    """KL term used by the original LAM implementation."""

    return (
        -0.5
        * torch.sum(
            1 + outputs["z_var"] - outputs["z_mu"] ** 2 - outputs["z_var"].exp(),
            dim=1,
        ).mean()
    )


def _action_targets(batch: Mapping[str, Tensor], prediction: Tensor) -> Tensor:
    if "actions" not in batch:
        raise KeyError("navigation LAM batches must contain an 'actions' tensor")
    target = batch["actions"]
    if target.ndim == 2:
        target = target.unsqueeze(1)
    if target.shape != prediction.shape:
        raise ValueError(
            "actions must have shape [B, T-1, action_dim]; "
            f"got {tuple(target.shape)}, expected {tuple(prediction.shape)}"
        )
    return target.to(device=prediction.device, dtype=prediction.dtype)


def _masked_action_mse(
    batch: Mapping[str, Tensor], prediction: Tensor, target: Tensor
) -> Tensor:
    """Mean action error over labeled samples only; old batches are all labeled."""

    mask = batch.get("action_mask")
    if mask is None:
        return F.mse_loss(prediction, target)
    mask = mask.to(device=prediction.device, dtype=torch.bool).reshape(-1)
    if mask.numel() != prediction.shape[0]:
        raise ValueError(
            f"action_mask must contain one value per sample; got {mask.numel()} "
            f"for batch size {prediction.shape[0]}"
        )
    if not torch.any(mask):
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[mask], target[mask])


def _make_action_decoder(
    latent_dim: int, model_dim: int, action_dim: int
) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(latent_dim),
        nn.Linear(latent_dim, model_dim),
        nn.GELU(),
        nn.Linear(model_dim, action_dim),
    )


class _NavigationLAMBase(LightningModule):
    """Training/logging shared by non-pixel navigation variants."""

    log_interval: int
    optimizer: OptimizerCallable

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        del batch_idx
        _, loss, aux_losses = self.shared_step(batch)
        self.log_dict(
            {"train_loss": loss, **{f"train/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        _log_sampling_statistics(self, batch)
        return loss

    @torch.no_grad()
    def test_step(self, batch: dict, batch_idx: int) -> Tensor:
        del batch_idx
        _, loss, aux_losses = self.shared_step(batch)
        self.log_dict(
            {"test_loss": loss, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    scheduler_total_steps: int = 0
    scheduler_warmup_steps: int = 0
    scheduler_min_lr_ratio: float = 0.1

    def configure_optimizers(self):
        return _optimizer_with_schedule(
            self.optimizer,
            self.parameters(),
            total_steps=self.scheduler_total_steps,
            warmup_steps=self.scheduler_warmup_steps,
            min_lr_ratio=self.scheduler_min_lr_ratio,
        )


class PixelLAM(LAM):
    """Original pixel LAM with an optional step-based warmup/cosine schedule."""

    def __init__(
        self,
        image_channels: int = 3,
        lam_model_dim: int = 512,
        lam_latent_dim: int = 32,
        lam_patch_size: int = 16,
        lam_enc_blocks: int = 8,
        lam_dec_blocks: int = 8,
        lam_num_heads: int = 8,
        lam_dropout: float = 0.0,
        beta: float = 0.01,
        log_interval: int = 1000,
        log_path: str = "log_imgs",
        optimizer: OptimizerCallable = AdamW,
        scheduler_total_steps: int = 0,
        scheduler_warmup_steps: int = 0,
        scheduler_min_lr_ratio: float = 0.1,
    ) -> None:
        super().__init__(
            image_channels=image_channels,
            lam_model_dim=lam_model_dim,
            lam_latent_dim=lam_latent_dim,
            lam_patch_size=lam_patch_size,
            lam_enc_blocks=lam_enc_blocks,
            lam_dec_blocks=lam_dec_blocks,
            lam_num_heads=lam_num_heads,
            lam_dropout=lam_dropout,
            beta=beta,
            log_interval=log_interval,
            log_path=log_path,
            optimizer=optimizer,
        )
        self.scheduler_total_steps = int(scheduler_total_steps)
        self.scheduler_warmup_steps = int(scheduler_warmup_steps)
        self.scheduler_min_lr_ratio = float(scheduler_min_lr_ratio)
        self.save_hyperparameters(
            {
                "scheduler_total_steps": self.scheduler_total_steps,
                "scheduler_warmup_steps": self.scheduler_warmup_steps,
                "scheduler_min_lr_ratio": self.scheduler_min_lr_ratio,
            }
        )

    def configure_optimizers(self):
        return _optimizer_with_schedule(
            self.optimizer,
            self.parameters(),
            total_steps=self.scheduler_total_steps,
            warmup_steps=self.scheduler_warmup_steps,
            min_lr_ratio=self.scheduler_min_lr_ratio,
        )

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        loss = super().training_step(batch, batch_idx)
        _log_sampling_statistics(self, batch)
        return loss


class _VisualLatentEncoder(nn.Module):
    """The encoder/VAE half of ``LatentActionModel``, without pixel modules."""

    def __init__(
        self,
        in_dim: int,
        model_dim: int,
        latent_dim: int,
        patch_size: int,
        enc_blocks: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        patch_token_dim = in_dim * patch_size**2

        # Keep the construction order and initialization identical to the
        # corresponding prefix of LatentActionModel.__init__.
        self.action_prompt = nn.Parameter(torch.empty(1, 1, 1, patch_token_dim))
        nn.init.uniform_(self.action_prompt, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=patch_token_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.fc = nn.Linear(model_dim, latent_dim * 2)
        self.mu_record = None

    def encode(self, videos: Tensor) -> dict[str, Tensor]:
        batch_size, sequence_length = videos.shape[:2]
        if sequence_length < 2:
            raise ValueError("LAM requires at least two frames per sample")
        patches = patchify(videos, self.patch_size)
        action_pad = self.action_prompt.expand(batch_size, sequence_length, -1, -1)
        padded_patches = torch.cat([action_pad, patches], dim=2)

        z = self.encoder(padded_patches)
        z = z[:, 1:, 0]
        z = z.reshape(batch_size * (sequence_length - 1), self.model_dim)
        moments = self.fc(z)
        z_mu, z_var = torch.chunk(moments, 2, dim=1)
        if self.training:
            z_rep = z_mu + torch.randn_like(z_var) * torch.exp(0.5 * z_var)
        else:
            z_rep = z_mu
        z_rep = z_rep.reshape(batch_size, sequence_length - 1, 1, self.latent_dim)

        if not self.training:
            if self.mu_record is None:
                self.mu_record = z_mu
            else:
                self.mu_record = torch.cat([self.mu_record, z_mu], dim=0)

        return {"z_rep": z_rep, "z_mu": z_mu, "z_var": z_var}

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return self.encode(batch["videos"])


class ActionOnlyLAM(_NavigationLAMBase):
    """Infer a latent action from frames and decode it to navigation action."""

    def __init__(
        self,
        image_channels: int = 3,
        lam_model_dim: int = 512,
        lam_latent_dim: int = 32,
        lam_patch_size: int = 16,
        lam_enc_blocks: int = 8,
        lam_dec_blocks: int = 8,
        lam_num_heads: int = 8,
        lam_dropout: float = 0.0,
        beta: float = 0.01,
        log_interval: int = 1000,
        log_path: str = "log_imgs",
        optimizer: OptimizerCallable = AdamW,
        action_dim: int = 3,
    ) -> None:
        super().__init__()
        # lam_dec_blocks and log_path remain constructor-compatible with LAM;
        # this variant intentionally does not instantiate a pixel decoder.
        self.lam = _VisualLatentEncoder(
            in_dim=image_channels,
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            patch_size=lam_patch_size,
            enc_blocks=lam_enc_blocks,
            num_heads=lam_num_heads,
            dropout=lam_dropout,
        )
        self.action_decoder = _make_action_decoder(
            lam_latent_dim, lam_model_dim, action_dim
        )
        self.beta = beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.save_hyperparameters()

    def forward(self, batch: dict) -> dict[str, Tensor]:
        outputs = self.lam(batch)
        outputs["action_recon"] = self.action_decoder(outputs["z_rep"].squeeze(2))
        return outputs

    def loss_components(
        self, batch: dict
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        outputs = self(batch)
        target = _action_targets(batch, outputs["action_recon"])
        return outputs, {
            "action_mse": _masked_action_mse(
                batch, outputs["action_recon"], target
            ),
            "kl_loss": _kl_loss(outputs),
        }

    def shared_step(self, batch: dict) -> tuple:
        outputs, losses = self.loss_components(batch)
        loss = losses["action_mse"] + self.beta * losses["kl_loss"]
        return outputs, loss, tuple(losses.items())


class PixelActionLAM(LAM):
    """Original pixel-reconstruction LAM with an additional action head."""

    def __init__(
        self,
        image_channels: int = 3,
        lam_model_dim: int = 512,
        lam_latent_dim: int = 32,
        lam_patch_size: int = 16,
        lam_enc_blocks: int = 8,
        lam_dec_blocks: int = 8,
        lam_num_heads: int = 8,
        lam_dropout: float = 0.0,
        beta: float = 0.01,
        log_interval: int = 1000,
        log_path: str = "log_imgs",
        optimizer: OptimizerCallable = AdamW,
        action_dim: int = 3,
        action_loss_weight: float | None = None,
        scheduler_total_steps: int = 0,
        scheduler_warmup_steps: int = 0,
        scheduler_min_lr_ratio: float = 0.1,
    ) -> None:
        if action_loss_weight is None:
            raise ValueError(
                "PixelActionLAM requires a calibrated action_loss_weight; run "
                "calibrate_action_weight.py and load its generated YAML overlay"
            )
        if not math.isfinite(action_loss_weight) or action_loss_weight <= 0:
            raise ValueError("action_loss_weight must be finite and positive")
        # Calling the original constructor first preserves both module
        # initialization order and all pixel-path state-dict keys.
        super().__init__(
            image_channels=image_channels,
            lam_model_dim=lam_model_dim,
            lam_latent_dim=lam_latent_dim,
            lam_patch_size=lam_patch_size,
            lam_enc_blocks=lam_enc_blocks,
            lam_dec_blocks=lam_dec_blocks,
            lam_num_heads=lam_num_heads,
            lam_dropout=lam_dropout,
            beta=beta,
            log_interval=log_interval,
            log_path=log_path,
            optimizer=optimizer,
        )
        self.action_decoder = _make_action_decoder(
            lam_latent_dim, lam_model_dim, action_dim
        )
        self.action_loss_weight = action_loss_weight
        self.scheduler_total_steps = int(scheduler_total_steps)
        self.scheduler_warmup_steps = int(scheduler_warmup_steps)
        self.scheduler_min_lr_ratio = float(scheduler_min_lr_ratio)
        self.save_hyperparameters(
            {
                "action_dim": action_dim,
                "action_loss_weight": action_loss_weight,
                "scheduler_total_steps": self.scheduler_total_steps,
                "scheduler_warmup_steps": self.scheduler_warmup_steps,
                "scheduler_min_lr_ratio": self.scheduler_min_lr_ratio,
            }
        )

    def forward(self, batch: dict) -> dict[str, Tensor]:
        outputs = self.lam(batch)
        outputs["action_recon"] = self.action_decoder(outputs["z_rep"].squeeze(2))
        return outputs

    def loss_components(
        self, batch: dict
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Return unweighted losses, also used by action-weight calibration."""

        outputs = self(batch)
        target = _action_targets(batch, outputs["action_recon"])
        return outputs, {
            "mse_loss": F.mse_loss(outputs["recon"], batch["videos"][:, 1:]),
            "action_mse": _masked_action_mse(
                batch, outputs["action_recon"], target
            ),
            "kl_loss": _kl_loss(outputs),
        }

    def shared_step(self, batch: dict) -> tuple:
        outputs, losses = self.loss_components(batch)
        loss = (
            losses["mse_loss"]
            + self.action_loss_weight * losses["action_mse"]
            + self.beta * losses["kl_loss"]
        )
        aux_losses = (
            ("mse_loss", losses["mse_loss"]),
            ("kl_loss", losses["kl_loss"]),
            ("action_mse", losses["action_mse"]),
        )
        return outputs, loss, aux_losses

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        loss = super().training_step(batch, batch_idx)
        _log_sampling_statistics(self, batch)
        return loss

    def configure_optimizers(self):
        return _optimizer_with_schedule(
            self.optimizer,
            self.parameters(),
            total_steps=self.scheduler_total_steps,
            warmup_steps=self.scheduler_warmup_steps,
            min_lr_ratio=self.scheduler_min_lr_ratio,
        )


class _FeatureLatentActionModel(nn.Module):
    """Latent action VAE operating directly on DINO patch tokens."""

    def __init__(
        self,
        token_dim: int,
        model_dim: int,
        latent_dim: int,
        enc_blocks: int,
        dec_blocks: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.latent_dim = latent_dim

        self.action_prompt = nn.Parameter(torch.empty(1, 1, 1, token_dim))
        nn.init.uniform_(self.action_prompt, a=-1, b=1)
        self.encoder = SpatioTemporalTransformer(
            in_dim=token_dim,
            model_dim=model_dim,
            out_dim=model_dim,
            num_blocks=enc_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.fc = nn.Linear(model_dim, latent_dim * 2)
        self.patch_up = nn.Linear(token_dim, model_dim)
        self.action_up = nn.Linear(latent_dim, model_dim)
        self.decoder = SpatioTransformer(
            in_dim=model_dim,
            model_dim=model_dim,
            out_dim=token_dim,
            num_blocks=dec_blocks,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.mu_record = None

    def encode(self, features: Tensor) -> dict[str, Tensor]:
        batch_size, sequence_length = features.shape[:2]
        if sequence_length < 2:
            raise ValueError("LAM requires at least two frames per sample")
        action_pad = self.action_prompt.expand(batch_size, sequence_length, -1, -1)
        padded_features = torch.cat([action_pad, features], dim=2)
        z = self.encoder(padded_features)
        z = z[:, 1:, 0]
        z = z.reshape(batch_size * (sequence_length - 1), self.model_dim)
        moments = self.fc(z)
        z_mu, z_var = torch.chunk(moments, 2, dim=1)
        if self.training:
            z_rep = z_mu + torch.randn_like(z_var) * torch.exp(0.5 * z_var)
        else:
            z_rep = z_mu
        z_rep = z_rep.reshape(batch_size, sequence_length - 1, 1, self.latent_dim)

        if not self.training:
            if self.mu_record is None:
                self.mu_record = z_mu
            else:
                self.mu_record = torch.cat([self.mu_record, z_mu], dim=0)

        return {
            "features": features,
            "z_rep": z_rep,
            "z_mu": z_mu,
            "z_var": z_var,
        }

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        outputs = self.encode(features)
        start_features = self.patch_up(outputs["features"][:, :-1])
        action_features = self.action_up(outputs["z_rep"])
        feature_recon = self.decoder(start_features + action_features)
        del outputs["features"]
        outputs["feature_recon"] = feature_recon
        return outputs


class DINOFeatureLAM(_NavigationLAMBase):
    """Reconstruct frozen DINOv3 patch features instead of RGB pixels."""

    DINO_FEATURE_DIM = 1024
    DINO_PATCH_COUNT = 300
    INPUT_HEIGHT = 240
    INPUT_WIDTH = 320

    def __init__(
        self,
        image_channels: int = 3,
        lam_model_dim: int = 512,
        lam_latent_dim: int = 32,
        lam_patch_size: int = 16,
        lam_enc_blocks: int = 8,
        lam_dec_blocks: int = 8,
        lam_num_heads: int = 8,
        lam_dropout: float = 0.0,
        beta: float = 0.01,
        log_interval: int = 1000,
        log_path: str = "log_imgs",
        optimizer: OptimizerCallable = AdamW,
        dino_repo_path: str = "",
        dino_weights_path: str = "",
        dino_model_name: str = "dinov3_vitl16",
        dino_repo_revision: str = "",
        dino_weights_sha256: str = "",
        use_precomputed_dino_features: bool = False,
        scheduler_total_steps: int = 0,
        scheduler_warmup_steps: int = 0,
        scheduler_min_lr_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        if image_channels != 3:
            raise ValueError("DINOv3 expects exactly three image channels")
        if lam_patch_size != 16:
            raise ValueError("DINOv3 ViT-L/16 requires lam_patch_size=16")
        if dino_model_name != "dinov3_vitl16":
            raise ValueError(
                "DINOFeatureLAM is fixed to the official dinov3_vitl16 backbone"
            )

        self.dino_repo_path = str(Path(dino_repo_path).expanduser().resolve())
        self.dino_weights_path = str(Path(dino_weights_path).expanduser().resolve())
        self.dino_model_name = dino_model_name
        self.dino_repo_revision = dino_repo_revision.strip().lower()
        self.dino_weights_sha256 = dino_weights_sha256.strip().lower()
        self.use_precomputed_dino_features = bool(use_precomputed_dino_features)
        self.scheduler_total_steps = int(scheduler_total_steps)
        self.scheduler_warmup_steps = int(scheduler_warmup_steps)
        self.scheduler_min_lr_ratio = float(scheduler_min_lr_ratio)
        self._weights_sha256: str | None = None
        if self.use_precomputed_dino_features:
            if not self.dino_weights_sha256:
                raise ValueError(
                    "use_precomputed_dino_features requires dino_weights_sha256"
                )
            self._validate_dino_weights_hash_format()
        self.dino_encoder: nn.Module | None = None
        if not self.use_precomputed_dino_features:
            self._ensure_dino_encoder()

        self.lam = _FeatureLatentActionModel(
            token_dim=self.DINO_FEATURE_DIM,
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            enc_blocks=lam_enc_blocks,
            dec_blocks=lam_dec_blocks,
            num_heads=lam_num_heads,
            dropout=lam_dropout,
        )
        self.beta = beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer
        self.register_buffer(
            "_image_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_image_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.save_hyperparameters()

    def _validate_dino_paths(self) -> None:
        repo_path = Path(self.dino_repo_path)
        weights_path = Path(self.dino_weights_path)
        if not repo_path.is_dir() or not (repo_path / "hubconf.py").is_file():
            raise FileNotFoundError(
                f"DINOv3 local repository (including hubconf.py) not found: {repo_path}"
            )
        if not weights_path.is_file():
            raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")

    def _validate_dino_weights_hash_format(self) -> None:
        if len(self.dino_weights_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.dino_weights_sha256
        ):
            raise ValueError("dino_weights_sha256 must contain 64 lowercase hex digits")

    @staticmethod
    def _load_dino(repo_path: str, weights_path: str, model_name: str) -> nn.Module:
        # Import the backbone factory directly from the configured local source.
        # DINOv3's hubconf eagerly imports optional segmentation dependencies;
        # the backbone itself does not need them.  Loading the core factory also
        # avoids all torch.hub URL/checkpoint-cache behavior.
        sys.path.insert(0, repo_path)
        try:
            backbones = importlib.import_module("dinov3.hub.backbones")
        finally:
            sys.path.remove(repo_path)
        module_path = Path(backbones.__file__).resolve()
        if not module_path.is_relative_to(Path(repo_path)):
            raise RuntimeError(
                "Imported DINOv3 from an unexpected location: "
                f"{module_path} (configured repo: {repo_path})"
            )
        factory = getattr(backbones, model_name, None)
        if factory is None:
            raise ValueError(f"DINOv3 backbone factory does not exist: {model_name}")
        model = factory(pretrained=False)
        if Path(weights_path).suffix != ".pth":
            raise ValueError(
                "DINOFeatureLAM requires Meta's original .pth state-dict layout; "
                f"got {weights_path}"
            )
        try:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with old PyTorch
            state = torch.load(weights_path, map_location="cpu")
        if not isinstance(state, Mapping):
            raise TypeError(
                f"DINOv3 weights must contain a state-dict mapping: {weights_path}"
            )
        model.load_state_dict(state, strict=True)
        architecture = {
            "embed_dim": getattr(model, "embed_dim", None),
            "blocks": len(getattr(model, "blocks", ())),
            "patch_size": getattr(model, "patch_size", None),
            "storage_tokens": getattr(model, "n_storage_tokens", None),
            "attention_heads": getattr(model.blocks[0].attn, "num_heads", None),
        }
        expected = {
            "embed_dim": 1024,
            "blocks": 24,
            "patch_size": 16,
            "storage_tokens": 4,
            "attention_heads": 16,
        }
        if architecture != expected:
            raise RuntimeError(
                f"Unexpected DINOv3 ViT-L/16 architecture: {architecture}; "
                f"expected {expected}"
            )
        return model

    def train(self, mode: bool = True) -> DINOFeatureLAM:
        super().train(mode)
        # Lightning repeatedly toggles the parent module.  The frozen teacher
        # must never enter train mode (including stochastic depth/dropout).
        if self.dino_encoder is not None:
            self.dino_encoder.eval()
        return self

    def _ensure_dino_encoder(self, device: torch.device | None = None) -> nn.Module:
        if self.dino_encoder is None:
            self._validate_dino_paths()
            encoder = self._load_dino(
                self.dino_repo_path, self.dino_weights_path, self.dino_model_name
            )
            encoder.requires_grad_(False)
            encoder.eval()
            if device is not None:
                encoder.to(device)
            self.dino_encoder = encoder
        elif device is not None:
            parameter = next(self.dino_encoder.parameters(), None)
            if parameter is not None and parameter.device != device:
                self.dino_encoder.to(device)
        return self.dino_encoder

    @torch.no_grad()
    def extract_dino_features(self, videos: Tensor) -> Tensor:
        if videos.ndim != 5:
            raise ValueError("videos must have shape [B, T, H, W, C]")
        batch_size, sequence_length, height, width, channels = videos.shape
        expected = (self.INPUT_HEIGHT, self.INPUT_WIDTH, 3)
        if (height, width, channels) != expected:
            raise ValueError(
                "DINOFeatureLAM expects videos shaped [B,T,240,320,3]; "
                f"got spatial/channel shape {(height, width, channels)}"
            )
        images = videos.reshape(batch_size * sequence_length, height, width, channels)
        images = images.permute(0, 3, 1, 2)
        images = (images - self._image_mean) / self._image_std
        encoder = self._ensure_dino_encoder(images.device)
        feature_dict = encoder.forward_features(images)
        if (
            not isinstance(feature_dict, Mapping)
            or "x_norm_patchtokens" not in feature_dict
        ):
            raise RuntimeError(
                "DINOv3 forward_features() did not return x_norm_patchtokens"
            )
        tokens = feature_dict["x_norm_patchtokens"]
        expected_shape = (
            batch_size * sequence_length,
            self.DINO_PATCH_COUNT,
            self.DINO_FEATURE_DIM,
        )
        if tuple(tokens.shape) != expected_shape:
            raise RuntimeError(
                "unexpected DINOv3 patch-token shape: "
                f"got {tuple(tokens.shape)}, expected {expected_shape}"
            )
        return tokens.reshape(
            batch_size,
            sequence_length,
            self.DINO_PATCH_COUNT,
            self.DINO_FEATURE_DIM,
        )

    def forward(self, batch: dict) -> dict[str, Tensor]:
        if "dino_features" in batch:
            target_features = batch["dino_features"]
            if target_features.ndim != 4 or tuple(target_features.shape[1:]) != (
                2,
                self.DINO_PATCH_COUNT,
                self.DINO_FEATURE_DIM,
            ):
                raise ValueError(
                    "dino_features must have shape [B,2,300,1024]; "
                    f"got {tuple(target_features.shape)}"
                )
            if not target_features.is_floating_point():
                raise TypeError("dino_features must be a floating-point tensor")
            if not torch.is_autocast_enabled():
                # Lightning's mixed-precision context handles this in training;
                # keep direct CPU/debug calls usable as well.
                target_features = target_features.to(dtype=self.lam.patch_up.weight.dtype)
        elif "videos" in batch:
            target_features = self.extract_dino_features(batch["videos"])
        else:
            raise KeyError("DINOFeatureLAM requires 'dino_features' or 'videos'")
        outputs = self.lam(target_features)
        outputs["target_features"] = target_features[:, 1:]
        return outputs

    def loss_components(
        self, batch: dict
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        outputs = self(batch)
        return outputs, {
            "feature_mse": F.mse_loss(
                outputs["feature_recon"], outputs["target_features"]
            ),
            "kl_loss": _kl_loss(outputs),
        }

    def shared_step(self, batch: dict) -> tuple:
        outputs, losses = self.loss_components(batch)
        loss = losses["feature_mse"] + self.beta * losses["kl_loss"]
        cosine = F.cosine_similarity(
            outputs["feature_recon"], outputs["target_features"], dim=-1
        ).mean()
        return (
            outputs,
            loss,
            (
                ("feature_mse", losses["feature_mse"]),
                ("kl_loss", losses["kl_loss"]),
                ("patch_cosine_similarity", cosine),
            ),
        )

    def _weights_hash(self) -> str:
        if self._weights_sha256 is None:
            digest = hashlib.sha256()
            with open(self.dino_weights_path, "rb") as weights_file:
                for chunk in iter(lambda: weights_file.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            self._weights_sha256 = digest.hexdigest()
        return self._weights_sha256

    def on_save_checkpoint(self, checkpoint: MutableMapping) -> None:
        checkpoint["dino"] = {
            "model_name": self.dino_model_name,
            "repo_path": self.dino_repo_path,
            "repo_revision": self.dino_repo_revision,
            "weights_path": self.dino_weights_path,
            # Production cached-feature profiles already carry the audited hash
            # and stay I/O-free here.  Ad-hoc online-teacher profiles without a
            # configured identity still get a self-contained checkpoint.
            "weights_sha256": self.dino_weights_sha256 or self._weights_hash(),
        }

    def on_load_checkpoint(self, checkpoint: Mapping) -> None:
        metadata = checkpoint.get("dino")
        if metadata is None:
            raise RuntimeError("DINOFeatureLAM checkpoint is missing DINO metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError("DINOFeatureLAM checkpoint has malformed DINO metadata")
        if metadata.get("model_name") != self.dino_model_name:
            raise RuntimeError(
                "DINO model mismatch: checkpoint uses "
                f"{metadata.get('model_name')!r}, configured {self.dino_model_name!r}"
            )
        if metadata.get("repo_revision") != self.dino_repo_revision:
            raise RuntimeError(
                "DINO source revision mismatch: checkpoint uses "
                f"{metadata.get('repo_revision')!r}, configured "
                f"{self.dino_repo_revision!r}"
            )
        actual_hash = self.dino_weights_sha256 or self._weights_hash()
        if metadata.get("weights_sha256") != actual_hash:
            raise RuntimeError(
                "DINO weights SHA-256 mismatch: checkpoint requires "
                f"{metadata.get('weights_sha256')!r}, configured weights have "
                f"{actual_hash!r}"
            )

    def state_dict(
        self, destination=None, prefix: str = "", keep_vars: bool = False
    ) -> OrderedDict:
        state = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        dino_prefix = f"{prefix}dino_encoder."
        for key in tuple(state):
            if key.startswith(dino_prefix):
                del state[key]
        return state

    def load_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        # Ignore any legacy/foreign serialized teacher tensors, then inject the
        # currently reconstructed external teacher state so strict loading still
        # verifies every trainable LAM tensor.
        merged = OrderedDict(
            (key, value)
            for key, value in state_dict.items()
            if not key.startswith("dino_encoder.")
        )
        if hasattr(state_dict, "_metadata"):
            merged._metadata = state_dict._metadata  # type: ignore[attr-defined]
        full_current_state = nn.Module.state_dict(self)
        for key, value in full_current_state.items():
            if key.startswith("dino_encoder."):
                merged[key] = value
        return nn.Module.load_state_dict(self, merged, strict=strict, assign=assign)


__all__ = ["ActionOnlyLAM", "DINOFeatureLAM", "PixelActionLAM", "PixelLAM"]
