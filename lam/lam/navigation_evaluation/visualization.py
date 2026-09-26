"""Publication-readable visualizations for navigation LAM evaluation."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _image(value: Any, width: int, height: int) -> Image.Image:
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
    else:
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
        array = np.asarray(value)
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 2:
            array = array[..., None]
        if (
            array.ndim == 3
            and array.shape[-1] not in (1, 3, 4)
            and array.shape[0] in (1, 3, 4)
        ):
            array = np.moveaxis(array, 0, -1)
        if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
            raise ValueError(f"expected HWC image, got {array.shape}")
        if np.issubdtype(array.dtype, np.floating):
            array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
            array = (np.clip(array, 0, 1) * 255.0).round().astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        image = Image.fromarray(array).convert("RGB")
    return image.resize((width, height), Image.Resampling.BILINEAR)


def _wrapped_lines(text: str, width: int, font: ImageFont.ImageFont) -> str:
    """Wrap rendered text, including long path-like strings without spaces."""

    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    available = max(1, width - 20)
    output: list[str] = []
    for source_line in str(text).splitlines() or [""]:
        current = ""
        for word in source_line.split(" "):
            candidate = word if not current else f"{current} {word}"
            if probe.textlength(candidate, font=font) <= available:
                current = candidate
                continue
            if current:
                output.append(current)
            current = ""
            for character in word:
                candidate = current + character
                if current and probe.textlength(candidate, font=font) > available:
                    output.append(current)
                    current = character
                else:
                    current = candidate
        output.append(current)
    return "\n".join(output)


def _multiline_cell(
    text: str, width: int, height: int, *, font_size: int, background=(250, 250, 250)
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    font = _font(font_size)
    draw.multiline_text(
        (10, 8),
        _wrapped_lines(text, width, font),
        fill=(25, 25, 25),
        font=font,
        spacing=4,
    )
    return canvas


def _sheet(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[tuple[str, str]],
    output: str | Path,
    *,
    title: str,
    width: int = 320,
    height: int = 240,
    label_width: int = 300,
    font_size: int = 20,
    max_pixels: int = 50_000_000,
) -> Path:
    if width <= 0 or height <= 0 or label_width <= 0 or font_size <= 0:
        raise ValueError("sheet dimensions and font_size must be positive")
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    header_height = 58
    title_font = _font(font_size + 5)
    title_text = _wrapped_lines(title, label_width + len(columns) * width, title_font)
    title_height = max(52, 18 + len(title_text.splitlines()) * (font_size + 9))
    row_height = height + 72
    total_width = label_width + len(columns) * width
    total_height = title_height + header_height + len(rows) * row_height
    pixel_count = total_width * total_height
    if pixel_count > max_pixels:
        raise MemoryError(
            f"visualization sheet would contain {pixel_count:,} pixels; "
            f"limit is {max_pixels:,}. Render fewer rows or use a smaller layout."
        )
    canvas = Image.new("RGB", (total_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (14, 10),
        title_text,
        fill="black",
        font=title_font,
    )
    y = title_height
    draw.rectangle((0, y, total_width, y + header_height), fill=(232, 237, 243))
    draw.text((12, y + 14), "offset / action", fill="black", font=_font(font_size))
    for column_index, (heading, _) in enumerate(columns):
        heading_cell = _multiline_cell(
            heading,
            width,
            header_height,
            font_size=font_size,
            background=(232, 237, 243),
        )
        canvas.paste(heading_cell, (label_width + column_index * width, y))
    y += header_height
    for row_index, row in enumerate(rows):
        background = (248, 249, 251) if row_index % 2 == 0 else (238, 242, 246)
        draw.rectangle((0, y, total_width, y + row_height), fill=background)
        if row.get("label"):
            label = str(row["label"])
        elif isinstance(row.get("offset"), (int, np.integer)):
            label = f"offset={int(row['offset']):+d}"
        else:
            label = f"row={row_index}"
        label_cell = _multiline_cell(
            label,
            label_width,
            row_height,
            font_size=max(13, font_size - 2),
            background=background,
        )
        canvas.paste(label_cell, (0, y))
        for column_index, (_, key) in enumerate(columns):
            x = label_width + column_index * width
            value = row.get(key)
            if value is None:
                cell = _multiline_cell(
                    "N/A",
                    width,
                    height,
                    font_size=font_size,
                    background=(225, 225, 225),
                )
            elif isinstance(value, str):
                cell = _multiline_cell(value, width, height, font_size=font_size)
            else:
                cell = _image(value, width, height)
            canvas.paste(cell, (x, y))
            caption_value = row.get(f"{key}_caption")
            caption = "" if caption_value is None else str(caption_value)
            if caption:
                caption_cell = _multiline_cell(
                    caption,
                    width,
                    row_height - height,
                    font_size=max(12, font_size - 4),
                    background=background,
                )
                canvas.paste(caption_cell, (x, y + height))
        y += row_height
    output_path = Path(output).expanduser().resolve()
    if output_path.suffix.lower() != ".png":
        raise ValueError("visualization sheets must use a .png output path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            canvas.save(handle, format="PNG")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output_path


def render_pixel_sheet(
    rows: Sequence[dict[str, Any]],
    output: str | Path,
    *,
    title: str,
    cross: bool,
    **layout: Any,
) -> Path:
    has_action = any(
        row.get("action_overlay") is not None
        or row.get("donor_action_overlay") is not None
        or row.get("recipient_action_overlay") is not None
        for row in rows
    )
    if cross:
        columns = (
            ("donor anchor", "donor_anchor"),
            ("donor target", "donor_target"),
            ("recipient anchor", "recipient_anchor"),
            ("pred recipient", "prediction"),
            ("GT recipient", "target"),
        )
        if has_action:
            columns += (
                ("donor GT / predicted action", "donor_action_overlay"),
                ("recipient GT / predicted action", "recipient_action_overlay"),
            )
    else:
        columns = (
            ("anchor", "anchor"),
            ("pred target", "prediction"),
            ("GT target", "target"),
        )
        if has_action:
            columns += (("GT / predicted action", "action_overlay"),)
    return _sheet(rows, columns, output, title=title, **layout)


@dataclass
class FeaturePCA:
    """A fixed three-channel PCA color basis fitted only on GT features."""

    mean: np.ndarray
    components: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.components = np.asarray(self.components, dtype=np.float32)
        self.lower = np.asarray(self.lower, dtype=np.float32)
        self.upper = np.asarray(self.upper, dtype=np.float32)
        if self.mean.ndim != 1:
            raise ValueError("FeaturePCA mean must be one-dimensional")
        if self.components.shape != (self.mean.size, 3):
            raise ValueError("FeaturePCA components must have shape [feature_dim, 3]")
        if self.lower.shape != (3,) or self.upper.shape != (3,):
            raise ValueError("FeaturePCA bounds must each have shape [3]")
        if not all(
            np.isfinite(value).all()
            for value in (self.mean, self.components, self.lower, self.upper)
        ):
            raise ValueError("FeaturePCA parameters must be finite")
        if not np.all(self.upper > self.lower):
            raise ValueError("FeaturePCA upper bounds must exceed lower bounds")

    @classmethod
    def fit(
        cls,
        feature_frames: Sequence[np.ndarray | torch.Tensor],
        *,
        seed: int = 32,
        max_tokens: int = 50_000,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
    ) -> FeaturePCA:
        if not feature_frames:
            raise ValueError("FeaturePCA requires at least one GT feature frame")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 3
        ):
            raise ValueError("max_tokens must be an integer of at least 3")
        if not 0 <= lower_percentile < upper_percentile <= 100:
            raise ValueError("PCA percentiles must satisfy 0 <= lower < upper <= 100")
        rng = np.random.default_rng(seed)
        per_frame = max(1, math.ceil(max_tokens / len(feature_frames)))
        samples: list[np.ndarray] = []
        feature_dim: int | None = None
        for raw in feature_frames:
            array = (
                raw.detach().float().cpu().numpy()
                if isinstance(raw, torch.Tensor)
                else np.asarray(raw, dtype=np.float32)
            )
            if array.ndim < 2 or array.shape[-1] == 0:
                raise ValueError(
                    "feature frames must have at least two dimensions and a non-empty feature axis"
                )
            array = array.reshape(-1, array.shape[-1])
            feature_dim = array.shape[-1] if feature_dim is None else feature_dim
            if array.shape[-1] != feature_dim:
                raise ValueError("all feature frames must share feature dimension")
            array = array[np.isfinite(array).all(axis=1)]
            if len(array) == 0:
                continue
            count = min(per_frame, len(array))
            indices = rng.choice(len(array), size=count, replace=False, shuffle=False)
            samples.append(array[indices])
        if not samples:
            raise ValueError("FeaturePCA found no finite GT feature tokens")
        matrix = np.concatenate(samples, axis=0)
        if len(matrix) > max_tokens:
            matrix = matrix[
                rng.choice(len(matrix), size=max_tokens, replace=False, shuffle=False)
            ]
        if matrix.shape[0] < 3 or matrix.shape[1] < 3:
            raise ValueError(
                "FeaturePCA requires at least 3 tokens and 3 feature dimensions"
            )
        tensor = torch.from_numpy(matrix)
        mean = tensor.mean(dim=0)
        tensor.sub_(mean)
        # pca_lowrank is randomized. fork_rng makes the result reproducible
        # without perturbing the process-wide evaluation RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            _, _, components = torch.pca_lowrank(tensor, q=3, center=False, niter=4)
        # Remove PCA sign ambiguity: the largest-magnitude loading is positive.
        for component in range(components.shape[1]):
            column = components[:, component]
            pivot = torch.argmax(column.abs())
            if column[pivot] < 0:
                components[:, component].neg_()
        projected = tensor @ components
        lower = torch.quantile(projected, lower_percentile / 100.0, dim=0)
        upper = torch.quantile(projected, upper_percentile / 100.0, dim=0)
        upper = torch.maximum(upper, lower + 1e-6)
        return cls(
            mean.numpy().copy(),
            components.numpy().copy(),
            lower.numpy().copy(),
            upper.numpy().copy(),
        )

    def project(self, features: np.ndarray | torch.Tensor) -> np.ndarray:
        array = (
            features.detach().float().cpu().numpy()
            if isinstance(features, torch.Tensor)
            else np.asarray(features, dtype=np.float32)
        )
        if array.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f"feature dim {array.shape[-1]} does not match PCA {self.mean.shape[0]}"
            )
        projected = (array - self.mean) @ self.components
        return np.clip((projected - self.lower) / (self.upper - self.lower), 0, 1)

    def patch_image(
        self,
        features: np.ndarray | torch.Tensor,
        *,
        patch_height: int = 15,
        patch_width: int = 20,
    ) -> np.ndarray:
        if patch_height <= 0 or patch_width <= 0:
            raise ValueError("patch dimensions must be positive")
        projected = self.project(features)
        token_count = int(np.prod(projected.shape[:-1]))
        expected = patch_height * patch_width
        if token_count != expected:
            raise ValueError(
                f"feature token count {token_count} does not match patch grid "
                f"{patch_height}x{patch_width} ({expected})"
            )
        projected = projected.reshape(patch_height, patch_width, 3)
        return projected.astype(np.float32)

    def save(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(
                    handle,
                    schema_version=np.asarray(1),
                    mean=self.mean,
                    components=self.components,
                    lower=self.lower,
                    upper=self.upper,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return output

    @classmethod
    def load(cls, path: str | Path) -> FeaturePCA:
        with np.load(Path(path), allow_pickle=False) as payload:
            if int(payload["schema_version"]) != 1:
                raise RuntimeError("unsupported FeaturePCA schema")
            return cls(
                payload["mean"],
                payload["components"],
                payload["lower"],
                payload["upper"],
            )


def cosine_error_map(
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    *,
    patch_height: int = 15,
    patch_width: int = 20,
) -> np.ndarray:
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError("patch dimensions must be positive")
    pred = torch.as_tensor(prediction, dtype=torch.float32).detach().cpu()
    gt = torch.as_tensor(target, dtype=torch.float32).detach().cpu()
    if pred.shape != gt.shape or pred.ndim < 2:
        raise ValueError(
            "prediction and target features must have the same shape ending in feature_dim"
        )
    expected = patch_height * patch_width
    token_count = int(np.prod(pred.shape[:-1]))
    if token_count != expected:
        raise ValueError(
            f"feature token count {token_count} does not match patch grid "
            f"{patch_height}x{patch_width} ({expected})"
        )
    finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(gt).all(dim=-1)
    safe_pred = torch.nan_to_num(pred)
    safe_gt = torch.nan_to_num(gt)
    similarity = torch.nn.functional.cosine_similarity(safe_pred, safe_gt, dim=-1)
    both_zero = (safe_pred.norm(dim=-1) == 0) & (safe_gt.norm(dim=-1) == 0)
    similarity = torch.where(both_zero, torch.ones_like(similarity), similarity)
    # Non-finite feature tokens are deliberately rendered as maximum error,
    # rather than becoming an invalid image or being mistaken for a good fit.
    similarity = torch.where(finite, similarity, -torch.ones_like(similarity))
    error = (1.0 - similarity).clamp(0, 2).reshape(patch_height, patch_width)
    normalized = (error / 2.0).cpu().numpy()
    # Blue -> yellow -> red, intentionally independent of matplotlib.
    red = np.clip(normalized * 2.0, 0, 1)
    green = np.clip(1.0 - np.abs(normalized - 0.5) * 2.0, 0, 1)
    blue = np.clip(1.0 - normalized * 2.0, 0, 1)
    return np.stack([red, green, blue], axis=-1).astype(np.float32)


def render_feature_sheet(
    rows: Sequence[dict[str, Any]],
    output: str | Path,
    *,
    title: str,
    cross: bool,
    **layout: Any,
) -> Path:
    columns = (
        (
            ("donor anchor RGB", "donor_anchor"),
            ("donor target RGB", "donor_target"),
            ("recipient anchor RGB", "recipient_anchor"),
            ("recipient target RGB", "recipient_target"),
            ("recipient anchor DINO PCA", "anchor_pca"),
            ("pred DINO PCA", "prediction_pca"),
            ("GT DINO PCA", "target_pca"),
        )
        if cross
        else (
            ("anchor RGB", "anchor"),
            ("target RGB", "target_rgb"),
            ("anchor DINO PCA", "anchor_pca"),
            ("pred DINO PCA", "prediction_pca"),
            ("GT DINO PCA", "target_pca"),
        )
    )
    return _sheet(rows, columns, output, title=title, **layout)


def action_overlay(
    ground_truth: Sequence[float],
    prediction: Sequence[float],
    *,
    size: tuple[int, int] = (320, 240),
) -> Image.Image:
    """Draw waypoint XY vectors and yaw values without implying an RGB prediction."""

    gt = np.asarray(ground_truth, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    if gt.shape != (3,) or pred.shape != (3,):
        raise ValueError("action_overlay expects [dx,dy,dyaw] for GT and prediction")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("action overlay size must be positive")
    canvas = Image.new("RGB", size, (249, 249, 249))
    draw = ImageDraw.Draw(canvas)
    if not np.isfinite(gt).all() or not np.isfinite(pred).all():
        draw.multiline_text(
            (12, 12),
            "Action unavailable\n(non-finite GT or prediction)",
            fill=(150, 35, 35),
            font=_font(16),
            spacing=5,
        )
        return canvas
    origin = (width // 2, height - 38)
    draw.line((origin[0], 12, origin[0], height - 20), fill=(180, 180, 180), width=2)
    draw.line((18, origin[1], width - 18, origin[1]), fill=(180, 180, 180), width=2)
    scale = 0.75 * min(
        width / (2 * max(1.0, abs(gt[1]), abs(pred[1]))),
        height / max(2.0, abs(gt[0]), abs(pred[0])),
    )

    def arrow(action: np.ndarray, color: tuple[int, int, int], label: str) -> None:
        endpoint = (
            origin[0] - int(action[1] * scale),
            origin[1] - int(action[0] * scale),
        )
        draw.line((*origin, *endpoint), fill=color, width=5)
        radius = 5
        draw.ellipse(
            (
                endpoint[0] - radius,
                endpoint[1] - radius,
                endpoint[0] + radius,
                endpoint[1] + radius,
            ),
            fill=color,
        )
        draw.text(
            (endpoint[0] + 6, endpoint[1] - 12), label, fill=color, font=_font(14)
        )

    arrow(gt, (30, 110, 210), "GT")
    arrow(pred, (220, 80, 35), "pred")
    draw.text(
        (8, 6),
        f"GT yaw {math.degrees(gt[2]):+.1f}°\npred yaw {math.degrees(pred[2]):+.1f}°",
        fill=(25, 25, 25),
        font=_font(14),
    )
    return canvas


def render_action_sheet(
    rows: Sequence[dict[str, Any]],
    output: str | Path,
    *,
    title: str,
    cross: bool,
    **layout: Any,
) -> Path:
    columns = (
        (
            ("donor anchor", "donor_anchor"),
            ("donor target", "donor_target"),
            ("recipient anchor", "recipient_anchor"),
            ("recipient target", "recipient_target"),
            ("donor GT / decoded latent", "donor_action_overlay"),
            ("recipient GT / decoded latent", "recipient_action_overlay"),
        )
        if cross
        else (
            ("anchor", "anchor"),
            ("target", "target"),
            ("GT / decoded action", "action_overlay"),
        )
    )
    return _sheet(rows, columns, output, title=title, **layout)


__all__ = [
    "FeaturePCA",
    "action_overlay",
    "cosine_error_map",
    "render_action_sheet",
    "render_feature_sheet",
    "render_pixel_sheet",
]
