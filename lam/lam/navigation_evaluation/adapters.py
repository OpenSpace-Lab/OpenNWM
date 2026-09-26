"""Strict, deterministic model adapters for navigation LAM evaluation.

The training variants expose similar latent-action components through three
different forward paths.  This module gives the evaluator one tensor contract
without teaching the sampling/reporting code about those implementation
details.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from lam.model import LAM
from lam.modules.blocks import patchify, unpatchify
from lam.navigation_callbacks import enable_fused_sdpa
from lam.navigation_variants import (
    ActionOnlyLAM,
    DINOFeatureLAM,
    PixelActionLAM,
    PixelLAM,
)

Capability = str
AdapterFactory = Callable[
    [nn.Module, frozenset[Capability], Mapping[str, Any]], "EvaluationAdapter"
]


@dataclass(frozen=True)
class _AdapterRegistration:
    model_class: type[nn.Module]
    capabilities: frozenset[Capability]
    factory: AdapterFactory


_ADAPTER_REGISTRY: dict[str, _AdapterRegistration] = {}


def register_evaluation_adapter(
    class_path: str,
    model_class: type[nn.Module],
    capabilities: set[Capability] | frozenset[Capability],
    factory: AdapterFactory | None = None,
    *,
    replace: bool = False,
) -> None:
    """Explicitly add a checkpoint class to the evaluation whitelist.

    Registering any future training paradigm is intentionally explicit: an
    arbitrary ``_class_path`` found in a pickle is never imported or executed.
    A custom factory may return an :class:`EvaluationAdapter` subclass when a
    future model needs different decoding semantics.
    """

    if not isinstance(class_path, str) or not class_path.strip():
        raise ValueError("class_path must be a non-empty string")
    if not issubclass(model_class, nn.Module):
        raise TypeError("model_class must be a torch.nn.Module subclass")
    normalized = frozenset(capabilities)
    unknown = normalized - {"rgb", "feature", "action"}
    if unknown:
        raise ValueError(f"unknown evaluation capabilities: {sorted(unknown)}")
    if class_path in _ADAPTER_REGISTRY and not replace:
        raise ValueError(f"evaluation adapter is already registered: {class_path}")
    _ADAPTER_REGISTRY[class_path] = _AdapterRegistration(
        model_class=model_class,
        capabilities=normalized,
        factory=EvaluationAdapter if factory is None else factory,
    )


def registered_checkpoint_classes() -> tuple[str, ...]:
    """Return the exact class paths accepted by the checkpoint loader."""

    return tuple(sorted(_ADAPTER_REGISTRY))


def checkpoint_capabilities(class_path: str) -> frozenset[Capability]:
    """Return registered capabilities without constructing the checkpoint model."""

    try:
        return _ADAPTER_REGISTRY[class_path].capabilities
    except KeyError as exc:
        raise ValueError(f"unsupported checkpoint class {class_path!r}") from exc


class EvaluationAdapter:
    """Capability-aware deterministic interface around one navigation LAM.

    Tensor contract:

    * ``encode(videos[B,T,H,W,C]) -> z_mu[B,T-1,1,D]``.
    * RGB/feature decoders accept either ``anchor[B,H,W,C]`` or
      ``anchor[B,K,H,W,C]`` and return the same leading rank.  ``K`` must match
      the latent sequence length.
    * ``decode_action(z) -> action[B,K,A]``.

    Every public inference method runs under ``torch.inference_mode``.  The
    model stays in eval mode, VAE sampling is therefore replaced by ``z_mu``,
    and the training-time ``mu_record`` accumulator is cleared after each call.
    """

    def __init__(
        self,
        model: nn.Module,
        capabilities: frozenset[Capability],
        checkpoint_metadata: Mapping[str, Any],
    ) -> None:
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.capabilities = frozenset(capabilities)
        self.checkpoint_metadata = dict(checkpoint_metadata)
        try:
            first_parameter = next(self.model.parameters())
        except StopIteration:  # pragma: no cover - all current LAMs have state
            self.device = torch.device("cpu")
            self.dtype = torch.float32
        else:
            self.device = first_parameter.device
            self.dtype = first_parameter.dtype

    def _require(self, capability: Capability) -> None:
        if capability not in self.capabilities:
            class_path = self.checkpoint_metadata.get("class_path", type(self.model))
            raise NotImplementedError(
                f"checkpoint {class_path!r} has no {capability!r} decoder"
            )

    def _input(self, value: Tensor) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError("evaluation model inputs must be torch tensors")
        if value.is_floating_point():
            return value.to(device=self.device, dtype=self.dtype)
        return value.to(device=self.device)

    @property
    def _lam(self) -> nn.Module:
        lam = getattr(self.model, "lam", None)
        if not isinstance(lam, nn.Module):
            raise TypeError("registered evaluation model has no .lam module")
        return lam

    def _clear_mu_record(self) -> None:
        if hasattr(self._lam, "mu_record"):
            self._lam.mu_record = None

    @staticmethod
    def _anchors(anchor: Tensor) -> tuple[Tensor, bool]:
        if anchor.ndim == 4:
            return anchor.unsqueeze(1), True
        if anchor.ndim == 5:
            return anchor, False
        raise ValueError(
            "anchor must have shape [B,H,W,C] or [B,K,H,W,C]; "
            f"got {tuple(anchor.shape)}"
        )

    @staticmethod
    def _latent(z_mu: Tensor, *, batch: int, steps: int) -> Tensor:
        if z_mu.ndim == 2:
            z_mu = z_mu.unsqueeze(1).unsqueeze(2)
        elif z_mu.ndim == 3:
            z_mu = z_mu.unsqueeze(2)
        if z_mu.ndim != 4 or z_mu.shape[2] != 1:
            raise ValueError(
                "latent action must have shape [B,D], [B,K,D], or [B,K,1,D]; "
                f"got {tuple(z_mu.shape)}"
            )
        if z_mu.shape[:2] != (batch, steps):
            raise ValueError(
                "latent leading dimensions must match anchors: "
                f"latent={tuple(z_mu.shape[:2])}, anchor={(batch, steps)}"
            )
        return z_mu

    @torch.inference_mode()
    def encode(self, videos: Tensor) -> Tensor:
        """Encode frame pairs to deterministic posterior means.

        Args:
            videos: RGB tensor shaped ``[B,T,H,W,C]`` with ``T >= 2``.

        Returns:
            Posterior mean shaped ``[B,T-1,1,latent_dim]``.
        """

        if videos.ndim != 5 or videos.shape[1] < 2:
            raise ValueError(
                "videos must have shape [B,T,H,W,C] with at least two frames; "
                f"got {tuple(videos.shape)}"
            )
        videos = self._input(videos)
        batch, frames = videos.shape[:2]
        self.model.eval()
        self._clear_mu_record()
        try:
            if "feature" in self.capabilities:
                if not isinstance(self.model, DINOFeatureLAM):
                    raise TypeError("feature adapter requires DINOFeatureLAM semantics")
                features = self.model.extract_dino_features(videos)
                outputs = self._lam.encode(features)
            else:
                outputs = self._lam.encode(videos)
            z_mu = outputs["z_mu"]
            expected = batch * (frames - 1)
            if z_mu.ndim != 2 or z_mu.shape[0] != expected:
                raise RuntimeError(
                    "LAM encoder returned malformed z_mu: "
                    f"got {tuple(z_mu.shape)}, expected first dimension {expected}"
                )
            return z_mu.reshape(batch, frames - 1, 1, z_mu.shape[-1])
        finally:
            self._clear_mu_record()

    @torch.inference_mode()
    def decode_rgb(self, anchor: Tensor, z_mu: Tensor) -> Tensor:
        """Decode latent actions against RGB anchors.

        A rank-four anchor returns ``[B,H,W,C]``; a rank-five anchor returns
        ``[B,K,H,W,C]``.
        """

        self._require("rgb")
        anchors, squeeze_time = self._anchors(self._input(anchor))
        batch, steps, height, width = anchors.shape[:4]
        latent = self._latent(self._input(z_mu), batch=batch, steps=steps)
        lam = self._lam
        required = ("patch_size", "patch_up", "action_up", "decoder")
        if any(not hasattr(lam, name) for name in required):
            raise RuntimeError("registered RGB adapter lacks pixel decoder modules")
        anchor_patches = lam.patch_up(patchify(anchors, lam.patch_size))
        action_patches = lam.action_up(latent)
        reconstruction = torch.sigmoid(lam.decoder(anchor_patches + action_patches))
        rgb = unpatchify(reconstruction, lam.patch_size, height, width)
        return rgb[:, 0] if squeeze_time else rgb

    @torch.inference_mode()
    def extract_features(self, frames: Tensor) -> Tensor:
        """Extract frozen DINO patch tokens from RGB frames.

        Rank-four input returns ``[B,N,D]`` and rank-five input returns
        ``[B,K,N,D]``.  This is used for GT targets and anchor-copy controls;
        metrics are always computed in the unprojected teacher feature space.
        """

        self._require("feature")
        if not isinstance(self.model, DINOFeatureLAM):
            raise TypeError("feature adapter requires DINOFeatureLAM semantics")
        frames, squeeze_time = self._anchors(self._input(frames))
        features = self.model.extract_dino_features(frames)
        return features[:, 0] if squeeze_time else features

    @torch.inference_mode()
    def decode_feature(self, anchor: Tensor, z_mu: Tensor) -> Tensor:
        """Decode DINO features using RGB anchors and supplied latent actions.

        A rank-four anchor returns ``[B,N,D]``; a rank-five anchor returns
        ``[B,K,N,D]``.
        """

        self._require("feature")
        if not isinstance(self.model, DINOFeatureLAM):
            raise TypeError("feature adapter requires DINOFeatureLAM semantics")
        anchors, squeeze_time = self._anchors(self._input(anchor))
        batch, steps = anchors.shape[:2]
        latent = self._latent(self._input(z_mu), batch=batch, steps=steps)
        anchor_features = self.model.extract_dino_features(anchors)
        lam = self._lam
        reconstruction = lam.decoder(
            lam.patch_up(anchor_features) + lam.action_up(latent)
        )
        return reconstruction[:, 0] if squeeze_time else reconstruction

    @torch.inference_mode()
    def decode_action(self, z_mu: Tensor) -> Tensor:
        """Decode normalized navigation actions as ``[B,K,action_dim]``."""

        self._require("action")
        if z_mu.ndim == 2:
            z_mu = z_mu.unsqueeze(1).unsqueeze(2)
        elif z_mu.ndim == 3:
            z_mu = z_mu.unsqueeze(2)
        if z_mu.ndim != 4 or z_mu.shape[2] != 1:
            raise ValueError(
                "latent action must have shape [B,D], [B,K,D], or [B,K,1,D]; "
                f"got {tuple(z_mu.shape)}"
            )
        decoder = getattr(self.model, "action_decoder", None)
        if not isinstance(decoder, nn.Module):
            raise TypeError("registered action adapter lacks action_decoder")
        return decoder(self._input(z_mu).squeeze(2))

    @torch.inference_mode()
    def native(self, videos: Tensor) -> dict[str, Tensor]:
        """Run the model's native eval forward for oracle/equivalence checks."""

        if videos.ndim != 5 or videos.shape[1] < 2:
            raise ValueError("videos must have shape [B,T,H,W,C] with T >= 2")
        self.model.eval()
        self._clear_mu_record()
        try:
            batch = {"videos": self._input(videos)}
            # The original pixel Lightning wrapper predates an explicit
            # forward(); its native inference entry point is the contained
            # LatentActionModel.  Navigation variants do define forward().
            if type(self.model) in {LAM, PixelLAM}:
                outputs = self._lam(batch)
            else:
                outputs = self.model(batch)
            if not isinstance(outputs, dict):
                raise TypeError("navigation LAM forward must return a dict")
            return outputs
        finally:
            self._clear_mu_record()


def _load_checkpoint_mapping(path: Path) -> Mapping[str, Any]:
    """Load a Lightning checkpoint safely while leaving tensor storage mmapped."""

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError as exc:  # pragma: no cover - unsupported old PyTorch
        raise RuntimeError(
            "navigation evaluation requires torch.load(weights_only=True, mmap=True)"
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must contain a mapping: {path}")
    return checkpoint


def _checkpoint_access_path(checkpoint_path: str | Path) -> Path:
    """Preserve ``/proc/self/fd`` paths instead of resolving their symlink.

    The evaluation runner deliberately keeps one source inode open while a
    mutable name such as ``last.ckpt`` may be atomically replaced.  Resolving
    the procfs link after replacement can produce a synthetic ``(deleted)``
    pathname and would defeat that binding.
    """

    path = Path(checkpoint_path).expanduser()
    if str(path).startswith("/proc/self/fd/"):
        return path
    return path.resolve()


def _checkpoint_identity(
    checkpoint: Mapping[str, Any], path: Path
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    raw_hparams = checkpoint.get("hyper_parameters")
    if not isinstance(raw_hparams, Mapping):
        raise TypeError(f"checkpoint is missing hyper_parameters mapping: {path}")
    hparams = dict(raw_hparams)
    class_path = hparams.get("_class_path")
    if not isinstance(class_path, str) or not class_path:
        raise RuntimeError(
            f"checkpoint hyper_parameters is missing _class_path: {path}"
        )
    if class_path not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"unsupported checkpoint class {class_path!r}; allowed classes are "
            f"{list(registered_checkpoint_classes())}"
        )
    metadata = {
        "path": str(path),
        "class_path": class_path,
        "global_step": checkpoint.get("global_step"),
        "epoch": checkpoint.get("epoch"),
        "lightning_version": checkpoint.get("pytorch-lightning_version"),
        "hparams": hparams,
    }
    datamodule_hparams = checkpoint.get("datamodule_hyper_parameters")
    if datamodule_hparams is not None:
        if not isinstance(datamodule_hparams, Mapping):
            raise RuntimeError(
                f"checkpoint has malformed datamodule_hyper_parameters: {path}"
            )
        metadata["datamodule_hparams"] = dict(datamodule_hparams)
    if class_path == "lam.navigation_variants.DINOFeatureLAM":
        metadata["dino"] = dict(checkpoint.get("dino", {}))
    return class_path, hparams, metadata


def inspect_evaluation_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Read whitelisted checkpoint metadata without constructing the model."""

    path = _checkpoint_access_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"navigation checkpoint not found: {path}")
    checkpoint = _load_checkpoint_mapping(path)
    _, _, metadata = _checkpoint_identity(checkpoint, path)
    return metadata


def _model_kwargs(
    model_class: type[nn.Module],
    hparams: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(model_class.__init__)
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unknown_overrides = set(overrides) - accepted
    if unknown_overrides:
        raise ValueError(
            f"unknown constructor overrides for {model_class.__name__}: "
            f"{sorted(unknown_overrides)}"
        )
    # Optimizer configs serialized by Lightning/jsonargparse are not callable
    # and are irrelevant to inference.  Let each model use its constructor
    # default; no optimizer is ever instantiated by this loader.
    kwargs = {
        key: value
        for key, value in hparams.items()
        if key in accepted and key != "optimizer"
    }
    kwargs.update(overrides)
    return kwargs


def load_evaluation_adapter(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    *,
    dtype: torch.dtype | None = None,
    fused_attention: bool = True,
    model_overrides: Mapping[str, Any] | None = None,
) -> EvaluationAdapter:
    """Strictly restore a whitelisted checkpoint as an evaluation adapter.

    The checkpoint is opened with ``weights_only=True`` and ``mmap=True`` so
    serialized optimizer tensors are never copied into resident CPU memory.
    State restoration is strict.  DINO checkpoints additionally execute the
    model's metadata hook, which audits source revision and external teacher
    weights before loading trainable tensors.
    """

    path = _checkpoint_access_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"navigation checkpoint not found: {path}")
    checkpoint = _load_checkpoint_mapping(path)
    class_path, hparams, metadata = _checkpoint_identity(checkpoint, path)
    registration = _ADAPTER_REGISTRY[class_path]
    kwargs = _model_kwargs(
        registration.model_class,
        hparams,
        {} if model_overrides is None else model_overrides,
    )
    model = registration.model_class(**kwargs)
    if isinstance(model, DINOFeatureLAM):
        # This validates checkpoint DINO metadata against the just-audited
        # local source tree and external weights.
        model.on_load_checkpoint(checkpoint)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"checkpoint is missing a state_dict mapping: {path}")
    # Keep checkpoint tensors mmap-backed, but copy only the model state into
    # the freshly allocated module.  ``assign=True`` would leave CPU
    # evaluation parameters backed by the checkpoint mapping, allowing a
    # later in-place file write to silently alter inference weights.  The
    # optimizer payload remains mmap-only and is never copied.
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        # Strict loading normally raises first; retain an explicit invariant in
        # case a future model overrides load_state_dict incorrectly.
        raise RuntimeError(
            "strict checkpoint restore returned incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    target_device = torch.device(device)
    if dtype is None:
        model = model.to(device=target_device)
    else:
        if not dtype.is_floating_point:
            raise ValueError("evaluation dtype must be floating point")
        model = model.to(device=target_device, dtype=dtype)
    model.eval().requires_grad_(False)
    if fused_attention:
        enable_fused_sdpa(model, require_match=True)
    metadata["device"] = str(target_device)
    metadata["dtype"] = str(next(model.parameters()).dtype)
    metadata["fused_attention"] = bool(fused_attention)
    return registration.factory(model, registration.capabilities, metadata)


def _default_factory(
    model: nn.Module,
    capabilities: frozenset[Capability],
    checkpoint_metadata: Mapping[str, Any],
) -> EvaluationAdapter:
    return EvaluationAdapter(model, capabilities, checkpoint_metadata)


for _class_path, _model_class, _capabilities in (
    ("lam.model.LAM", LAM, {"rgb"}),
    ("lam.navigation_variants.PixelLAM", PixelLAM, {"rgb"}),
    ("lam.navigation_variants.ActionOnlyLAM", ActionOnlyLAM, {"action"}),
    ("lam.navigation_variants.DINOFeatureLAM", DINOFeatureLAM, {"feature"}),
    (
        "lam.navigation_variants.PixelActionLAM",
        PixelActionLAM,
        {"rgb", "action"},
    ),
):
    register_evaluation_adapter(
        _class_path,
        _model_class,
        _capabilities,
        _default_factory,
    )


__all__ = [
    "EvaluationAdapter",
    "checkpoint_capabilities",
    "inspect_evaluation_checkpoint",
    "load_evaluation_adapter",
    "register_evaluation_adapter",
    "registered_checkpoint_classes",
]
