"""Per-sample reconstruction metrics and cross-domain control helpers."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MetricDict = dict[str, Tensor]


def _mean_nonbatch(value: Tensor) -> Tensor:
    if value.ndim == 1:
        return value
    return value.mean(dim=tuple(range(1, value.ndim)))


def _check_pair(prediction: Tensor, target: Tensor, name: str) -> None:
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError(f"{name} prediction and target must be torch tensors")
    if prediction.shape != target.shape:
        raise ValueError(
            f"{name} prediction and target shapes differ: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}"
        )
    if prediction.ndim < 2:
        raise ValueError(f"{name} tensors must include a batch and feature dimensions")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError(f"{name} tensors must be floating point")


def _rgb_images(value: Tensor) -> tuple[Tensor, int, int]:
    """Convert NHWC/NCHW images (optionally temporal) to flattened NCHW."""

    if value.ndim not in (4, 5):
        raise ValueError(
            "RGB tensors must have shape [B,H,W,C], [B,C,H,W], "
            "[B,T,H,W,C], or [B,T,C,H,W]"
        )
    batch = value.shape[0]
    steps = 1 if value.ndim == 4 else value.shape[1]
    if value.shape[-1] in (1, 3):
        if value.ndim == 4:
            images = value.permute(0, 3, 1, 2)
        else:
            images = value.permute(0, 1, 4, 2, 3).reshape(
                batch * steps, value.shape[-1], value.shape[2], value.shape[3]
            )
    else:
        channel_axis = 1 if value.ndim == 4 else 2
        if value.shape[channel_axis] not in (1, 3):
            raise ValueError(
                "RGB tensor channel dimension must contain one or three channels"
            )
        images = (
            value
            if value.ndim == 4
            else value.reshape(
                batch * steps, value.shape[2], value.shape[3], value.shape[4]
            )
        )
    return images.float(), batch, steps


def _per_sequence(value: Tensor, batch: int, steps: int) -> Tensor:
    if value.numel() != batch * steps:
        raise RuntimeError(
            f"metric returned {value.numel()} values for {batch * steps} images"
        )
    return value.reshape(batch, steps).mean(dim=1)


def _gaussian_ssim(prediction: Tensor, target: Tensor, data_range: float) -> Tensor:
    """Standard local SSIM, returned independently for every NCHW image."""

    _, channels, height, width = prediction.shape
    kernel_size = min(11, height, width)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    if kernel_size < 3:
        reduce = (2, 3)
        mu_x = prediction.mean(dim=reduce)
        mu_y = target.mean(dim=reduce)
        var_x = ((prediction - mu_x[:, :, None, None]) ** 2).mean(dim=reduce)
        var_y = ((target - mu_y[:, :, None, None]) ** 2).mean(dim=reduce)
        covariance = (
            (prediction - mu_x[:, :, None, None]) * (target - mu_y[:, :, None, None])
        ).mean(dim=reduce)
        score = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
            (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
        )
        return score.mean(dim=1)

    coordinates = torch.arange(
        kernel_size, device=prediction.device, dtype=prediction.dtype
    )
    coordinates -= (kernel_size - 1) / 2
    gaussian = torch.exp(-(coordinates.square()) / (2 * 1.5**2))
    gaussian /= gaussian.sum()
    kernel_2d = gaussian[:, None] * gaussian[None, :]
    kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size)

    mu_x = F.conv2d(prediction, kernel, groups=channels)
    mu_y = F.conv2d(target, kernel, groups=channels)
    mu_x_sq, mu_y_sq = mu_x.square(), mu_y.square()
    mu_xy = mu_x * mu_y
    var_x = F.conv2d(prediction.square(), kernel, groups=channels) - mu_x_sq
    var_y = F.conv2d(target.square(), kernel, groups=channels) - mu_y_sq
    covariance = F.conv2d(prediction * target, kernel, groups=channels) - mu_xy
    score = ((2 * mu_xy + c1) * (2 * covariance + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (var_x + var_y + c2)
    )
    return score.mean(dim=(1, 2, 3))


@torch.inference_mode()
def rgb_reconstruction_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    lpips_model: nn.Module | Callable[[Tensor, Tensor], Tensor] | None,
    data_range: float = 1.0,
    require_lpips: bool = True,
    metric_names: Sequence[str] | None = None,
) -> MetricDict:
    """Compute requested RGB reconstruction metrics per batch sample.

    Inputs may be channel-last or channel-first and may contain a temporal
    dimension.  Temporal image scores are averaged within each batch sample.
    By default, MSE, MAE, PSNR, SSIM and LPIPS are returned for backward
    compatibility. ``metric_names`` can select a strict subset when a smaller
    evaluation contract is desired.
    ``lpips_model`` is injected by the offline asset loader; if it is omitted,
    an evaluation requesting LPIPS fails by default rather than silently
    downloading or dropping it. Tests/diagnostics may set
    ``require_lpips=False``, in which case a per-sample NaN tensor is returned
    under the ``lpips`` key.
    """

    _check_pair(prediction, target, "RGB")
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")
    supported = ("mse", "mae", "psnr", "ssim", "lpips")
    requested = set(supported if metric_names is None else metric_names)
    if not requested:
        raise ValueError("metric_names must contain at least one RGB metric")
    unknown = requested - set(supported)
    if unknown:
        raise ValueError(f"unsupported RGB metrics: {sorted(unknown)}")
    pred, batch, steps = _rgb_images(prediction)
    gt, target_batch, target_steps = _rgb_images(target)
    if (batch, steps) != (target_batch, target_steps):  # pragma: no cover
        raise AssertionError("RGB layout conversion changed leading dimensions")
    difference = pred - gt
    metrics: MetricDict = {}
    if requested & {"mse", "psnr"}:
        mse_images = difference.square().mean(dim=(1, 2, 3))
        if "mse" in requested:
            metrics["mse"] = _per_sequence(mse_images, batch, steps)
        if "psnr" in requested:
            psnr_images = torch.where(
                mse_images == 0,
                torch.full_like(mse_images, float("inf")),
                10.0 * torch.log10((data_range**2) / mse_images),
            )
    if "mae" in requested:
        mae_images = difference.abs().mean(dim=(1, 2, 3))
        metrics["mae"] = _per_sequence(mae_images, batch, steps)
    if "psnr" in requested:
        metrics["psnr"] = _per_sequence(psnr_images, batch, steps)
    if "ssim" in requested:
        ssim_images = _gaussian_ssim(pred, gt, data_range)
        metrics["ssim"] = _per_sequence(ssim_images, batch, steps)

    if "lpips" not in requested:
        return metrics
    if lpips_model is None:
        if require_lpips:
            raise RuntimeError(
                "LPIPS metric is required but unavailable; prepare and verify "
                "offline assets before evaluation"
            )
        metrics["lpips"] = torch.full(
            (batch,), float("nan"), device=pred.device, dtype=pred.dtype
        )
    else:
        lpips_value = lpips_model(pred.clamp(0, data_range), gt.clamp(0, data_range))
        if not isinstance(lpips_value, Tensor):
            raise TypeError("LPIPS model must return a torch tensor")
        if lpips_value.ndim == 0:
            if batch * steps != 1:
                raise RuntimeError(
                    "LPIPS model reduced across the batch; construct it with "
                    "reduction='none'"
                )
            lpips_value = lpips_value.unsqueeze(0)
        lpips_images = lpips_value.reshape(lpips_value.shape[0], -1).mean(dim=1)
        metrics["lpips"] = _per_sequence(lpips_images, batch, steps)
    return metrics


@torch.inference_mode()
def feature_reconstruction_metrics(prediction: Tensor, target: Tensor) -> MetricDict:
    """Compute raw-token feature MSE, MAE and mean cosine per sample."""

    _check_pair(prediction, target, "feature")
    if prediction.ndim < 3:
        raise ValueError("feature tensors must end in a token feature dimension")
    pred = prediction.float()
    gt = target.float()
    reduce_dimensions = tuple(range(1, pred.ndim))
    cosine = F.cosine_similarity(pred, gt, dim=-1)
    cosine_reduce = tuple(range(1, cosine.ndim))
    return {
        "feature_mse": (pred - gt).square().mean(dim=reduce_dimensions),
        "feature_mae": (pred - gt).abs().mean(dim=reduce_dimensions),
        "feature_cosine_similarity": cosine.mean(dim=cosine_reduce),
    }


def wrapped_angle_difference(prediction: Tensor, target: Tensor) -> Tensor:
    """Return signed circular ``prediction-target`` differences in radians."""

    return torch.atan2(torch.sin(prediction - target), torch.cos(prediction - target))


def denormalize_translation(
    normalized_xy: Tensor,
    action_min: Sequence[float] | Tensor = (-2.5, -4.0),
    action_max: Sequence[float] | Tensor = (5.0, 4.0),
) -> Tensor:
    """Convert normalized XY action coordinates back to waypoint units."""

    minimum = torch.as_tensor(
        action_min, device=normalized_xy.device, dtype=normalized_xy.dtype
    )
    maximum = torch.as_tensor(
        action_max, device=normalized_xy.device, dtype=normalized_xy.dtype
    )
    if minimum.shape != (2,) or maximum.shape != (2,):
        raise ValueError("action_min and action_max must each contain two values")
    if not torch.all(maximum > minimum):
        raise ValueError("every action_max value must exceed action_min")
    return (normalized_xy + 1.0) * 0.5 * (maximum - minimum) + minimum


@torch.inference_mode()
def action_reconstruction_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    waypoint_spacing: float | Tensor,
    action_min: Sequence[float] | Tensor = (-2.5, -4.0),
    action_max: Sequence[float] | Tensor = (5.0, 4.0),
) -> MetricDict:
    """Compute normalized, physical translation and circular-yaw errors.

    The first two action coordinates use the training normalization and the
    last coordinate is an unnormalized yaw delta in radians.  Translation EPE
    is reported in waypoint units and, using the recipient domain's spacing,
    meters.  All returned tensors have shape ``[B]``.
    """

    _check_pair(prediction, target, "action")
    if prediction.shape[-1] != 3:
        raise ValueError(
            f"navigation actions must have final dimension 3, got {prediction.shape[-1]}"
        )
    pred = prediction.float()
    gt = target.float()
    reduce_dimensions = tuple(range(1, pred.ndim))
    difference = pred - gt
    # Yaw is stored in radians rather than linearly normalized.  Use its
    # circular residual in the aggregate normalized-space MSE/MAE as well, so
    # +pi and -pi do not appear maximally different.
    difference[..., 2] = wrapped_angle_difference(pred[..., 2], gt[..., 2])
    pred_waypoints = denormalize_translation(pred[..., :2], action_min, action_max)
    gt_waypoints = denormalize_translation(gt[..., :2], action_min, action_max)
    waypoint_epe = torch.linalg.vector_norm(pred_waypoints - gt_waypoints, dim=-1)
    waypoint_per_sample = _mean_nonbatch(waypoint_epe)

    spacing = torch.as_tensor(
        waypoint_spacing,
        device=waypoint_per_sample.device,
        dtype=waypoint_per_sample.dtype,
    )
    if spacing.ndim == 0:
        spacing = spacing.expand(pred.shape[0])
    elif spacing.shape != (pred.shape[0],):
        raise ValueError(
            "waypoint_spacing must be a scalar or one value per batch sample; "
            f"got {tuple(spacing.shape)}"
        )
    if not torch.isfinite(spacing).all() or not torch.all(spacing > 0):
        raise ValueError("waypoint_spacing must be finite and positive")

    yaw_error_deg = torch.rad2deg(wrapped_angle_difference(pred[..., 2], gt[..., 2]))
    return {
        "action_normalized_mse": difference.square().mean(dim=reduce_dimensions),
        "action_normalized_mae": difference.abs().mean(dim=reduce_dimensions),
        "translation_epe_waypoints": waypoint_per_sample,
        "translation_epe_meters": waypoint_per_sample * spacing,
        "yaw_mae_degrees": _mean_nonbatch(yaw_error_deg.abs()),
        "yaw_rmse_degrees": _mean_nonbatch(yaw_error_deg.square()).sqrt(),
    }


def _higher_is_better(metric_name: str) -> bool:
    name = metric_name.lower()
    return "cosine" in name or name in {"psnr", "ssim"}


def positive_degradation(
    swapped_metrics: Mapping[str, Tensor],
    oracle_metrics: Mapping[str, Tensor],
    *,
    higher_is_better: set[str] | frozenset[str] | None = None,
) -> MetricDict:
    """Return positive values when swapped reconstruction is worse than oracle."""

    if set(swapped_metrics) != set(oracle_metrics):
        raise ValueError("swapped and oracle metric dictionaries must have equal keys")
    explicit = None if higher_is_better is None else set(higher_is_better)
    degradation: MetricDict = {}
    for name in swapped_metrics:
        swapped = swapped_metrics[name]
        oracle = oracle_metrics[name]
        if swapped.shape != oracle.shape:
            raise ValueError(f"metric shape mismatch for {name!r}")
        high = _higher_is_better(name) if explicit is None else name in explicit
        degradation[name] = oracle - swapped if high else swapped - oracle
    return degradation


@torch.inference_mode()
def reconstruction_controls(
    metric_function: Callable[[Tensor, Tensor], Mapping[str, Tensor]],
    *,
    target: Tensor,
    anchor: Tensor,
    oracle: Tensor,
    swapped: Tensor,
    higher_is_better: set[str] | frozenset[str] | None = None,
) -> dict[str, MetricDict]:
    """Compute anchor-copy, oracle, swapped, and swapped degradation metrics."""

    anchor_metrics = dict(metric_function(anchor, target))
    oracle_metrics = dict(metric_function(oracle, target))
    swapped_metrics = dict(metric_function(swapped, target))
    return {
        "anchor_copy": anchor_metrics,
        "oracle": oracle_metrics,
        "swapped": swapped_metrics,
        "degradation": positive_degradation(
            swapped_metrics,
            oracle_metrics,
            higher_is_better=higher_is_better,
        ),
    }


# Short aliases keep runner configuration concise while the descriptive names
# remain the primary public API.
rgb_metrics = rgb_reconstruction_metrics
feature_metrics = feature_reconstruction_metrics
action_metrics = action_reconstruction_metrics


__all__ = [
    "action_metrics",
    "action_reconstruction_metrics",
    "denormalize_translation",
    "feature_metrics",
    "feature_reconstruction_metrics",
    "positive_degradation",
    "reconstruction_controls",
    "rgb_metrics",
    "rgb_reconstruction_metrics",
    "wrapped_angle_difference",
]
