"""Original LAM batch metrics and rendering; orchestration lives in evaluate.py."""
from __future__ import annotations
import copy
import hashlib
import math
import random
import json
import numpy as np
import torch
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from torch import Tensor
from torch.utils.data import Dataset
from .adapters import EvaluationAdapter
from .catalog import load_navigation_frame
from .metrics import action_metrics, denormalize_translation, feature_metrics, positive_degradation, reconstruction_controls, rgb_metrics
from .visualization import FeaturePCA, action_overlay, render_action_sheet, render_feature_sheet, render_pixel_sheet

class _RecordFrames(Dataset):
    def __init__(self, records: Sequence[Mapping[str, Any]], height: int, width: int):
        self.records = records
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, path: str) -> Tensor:
        return load_navigation_frame(
            path, image_height=self.height, image_width=self.width
        )

    def __getitem__(self, index: int) -> tuple[int, Tensor, Tensor, Tensor, Tensor]:
        record = self.records[index]
        source_anchor = self._load(str(record["source_anchor_path"]))
        source_target = self._load(str(record["source_target_path"]))
        if (
            record["recipient_anchor_path"] == record["source_anchor_path"]
            and record["recipient_target_path"] == record["source_target_path"]
        ):
            recipient_anchor, recipient_target = source_anchor, source_target
        else:
            recipient_anchor = self._load(str(record["recipient_anchor_path"]))
            recipient_target = self._load(str(record["recipient_target_path"]))
        return index, source_anchor, source_target, recipient_anchor, recipient_target


def _autocast(device: torch.device, precision: str):
    if precision == "32":
        return nullcontext()
    if device.type != "cuda":
        # CPU smoke tests remain portable and deterministic.
        return nullcontext()
    dtype = torch.float16 if precision == "16-mixed" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _tensor_values(value: Tensor) -> list[float | None]:
    result: list[float | None] = []
    for item in value.detach().float().cpu().reshape(-1):
        number = float(item)
        result.append(number if math.isfinite(number) else None)
    return result


def _append_metric_group(
    rows: list[dict[str, Any]], prefix: str, metrics: Mapping[str, Tensor]
) -> None:
    raw_values = {
        name: [float(item) for item in value.detach().float().cpu().reshape(-1)]
        for name, value in metrics.items()
    }
    for batch_index, row in enumerate(rows):
        for name, per_sample in raw_values.items():
            metric_name = name.removeprefix("feature_").removeprefix("action_")
            column = f"{prefix}_{metric_name}"
            value = per_sample[batch_index]
            if math.isfinite(value):
                row[column] = value
            else:
                row[column] = None
                status = (
                    "nan"
                    if math.isnan(value)
                    else ("positive_infinity" if value > 0 else "negative_infinity")
                )
                row.setdefault("metric_nonfinite", {})[column] = status


def _base_result_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    result.update(
        {
            "target_dataset": record["recipient_dataset"],
            "target_split": record["recipient_split"],
            "target_trajectory": record["recipient_trajectory"],
            "target_anchor": record["recipient_anchor"],
            "target_target": record["recipient_target"],
            "target_offset": record["recipient_offset"],
            "source_window_id": (
                f"{record['source_dataset']}/{record['source_split']}/"
                f"{record['source_trajectory']}@{record['source_anchor']}"
            ),
        }
    )
    return result


def _action_tensor(
    records: Sequence[Mapping[str, Any]], side: str, device: torch.device
) -> Tensor:
    values = [record[f"{side}_action_normalized"] for record in records]
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _spacing_tensor(
    records: Sequence[Mapping[str, Any]], side: str, device: torch.device
) -> Tensor:
    return torch.as_tensor(
        [float(record[f"{side}_waypoint_spacing"]) for record in records],
        dtype=torch.float32,
        device=device,
    )


def _decoded_waypoint(
    action: Tensor, action_min: Sequence[float], action_max: Sequence[float]
) -> np.ndarray:
    action = action.detach().float().cpu()
    xy = denormalize_translation(action[:2], action_min, action_max)
    yaw = torch.atan2(torch.sin(action[2:3]), torch.cos(action[2:3]))
    return torch.cat([xy, yaw]).numpy()


def _visual_label(record: Mapping[str, Any]) -> str:
    source = record["source_action_waypoint"]
    recipient = record["recipient_action_waypoint"]
    lines = [
        f"donor offset {int(record['source_offset']):+d}",
        f"donor wp=({source[0]:+.2f},{source[1]:+.2f}) yaw={math.degrees(source[2]):+.1f} deg",
    ]
    if record["protocol"].endswith("_cd"):
        lines.extend(
            [
                f"recipient offset {int(record['recipient_offset']):+d}",
                f"recipient wp=({recipient[0]:+.2f},{recipient[1]:+.2f}) yaw={math.degrees(recipient[2]):+.1f} deg",
                (
                    f"gap={float(record['match_translation_waypoint_l2']):.3f} wp, "
                    f"{float(record['match_yaw_degrees']):.2f} deg"
                ),
            ]
        )
    return "\n".join(lines)


def _visual_layout(config: Mapping[str, Any]) -> dict[str, int]:
    value = config["visualization"]
    return {
        "width": int(value["thumbnail_width"]),
        "height": int(value["thumbnail_height"]),
        "label_width": int(value["row_label_width"]),
        "font_size": int(value["font_size"]),
    }


def _metric_function_rgb(lpips_model: torch.nn.Module | None):
    return lambda prediction, target: rgb_metrics(
        prediction,
        target,
        lpips_model=lpips_model,
        require_lpips=lpips_model is not None,
    )


def _process_batch(
    adapter: EvaluationAdapter,
    records: Sequence[Mapping[str, Any]],
    source_anchor: Tensor,
    source_target: Tensor,
    recipient_anchor: Tensor,
    recipient_target: Tensor,
    *,
    lpips_model: torch.nn.Module | None,
    pca: FeaturePCA | None,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return JSON-compatible rows and visualization payloads keyed by record id."""

    device = adapter.device
    source_pair = torch.stack([source_anchor, source_target], dim=1)
    recipient_pair = torch.stack([recipient_anchor, recipient_target], dim=1)
    is_direct = all(
        record["source_anchor_path"] == record["recipient_anchor_path"]
        and record["source_target_path"] == record["recipient_target_path"]
        for record in records
    )
    precision = str(config["runtime"]["precision"])
    with _autocast(device, precision):
        source_z = adapter.encode(source_pair)
        recipient_z = source_z if is_direct else adapter.encode(recipient_pair)
    result_rows = [_base_result_record(record) for record in records]
    visuals: dict[str, dict[str, Any]] = {}

    if "rgb" in adapter.capabilities:
        with _autocast(device, precision):
            swapped = adapter.decode_rgb(recipient_anchor, source_z)
            oracle = (
                swapped
                if is_direct
                else adapter.decode_rgb(recipient_anchor, recipient_z)
            )
        controls = reconstruction_controls(
            _metric_function_rgb(lpips_model),
            target=recipient_target.to(device=device, dtype=torch.float32),
            anchor=recipient_anchor.to(device=device, dtype=torch.float32),
            oracle=oracle.float(),
            swapped=swapped.float(),
        )
        for group, metrics in controls.items():
            _append_metric_group(result_rows, f"rgb_{group}", metrics)
        for index, record in enumerate(records):
            if not record["is_visual"]:
                continue
            visuals[str(record["record_id"])] = {
                "label": _visual_label(record),
                "anchor": recipient_anchor[index].cpu(),
                "donor_anchor": source_anchor[index].cpu(),
                "donor_target": source_target[index].cpu(),
                "recipient_anchor": recipient_anchor[index].cpu(),
                "prediction": swapped[index].detach().float().cpu(),
                "target": recipient_target[index].cpu(),
            }
    else:
        for row in result_rows:
            row["rgb_status"] = "N/A: checkpoint has no RGB decoder"

    if "feature" in adapter.capabilities:
        if pca is None:
            raise RuntimeError("DINO feature evaluation requires a fixed GT PCA asset")
        target_features = adapter.extract_features(recipient_target)
        anchor_features = adapter.extract_features(recipient_anchor)
        with _autocast(device, precision):
            swapped_features = adapter.decode_feature(recipient_anchor, source_z)
            oracle_features = (
                swapped_features
                if is_direct
                else adapter.decode_feature(recipient_anchor, recipient_z)
            )
        controls = reconstruction_controls(
            feature_metrics,
            target=target_features,
            anchor=anchor_features,
            oracle=oracle_features,
            swapped=swapped_features,
        )
        for group, metrics in controls.items():
            _append_metric_group(result_rows, f"feature_{group}", metrics)
        for index, record in enumerate(records):
            if not record["is_visual"]:
                continue
            prediction = swapped_features[index].detach().float().cpu()
            target = target_features[index].detach().float().cpu()
            anchor_feature = anchor_features[index].detach().float().cpu()
            visuals[str(record["record_id"])] = {
                "label": _visual_label(record),
                "anchor": recipient_anchor[index].cpu(),
                "target_rgb": recipient_target[index].cpu(),
                "donor_anchor": source_anchor[index].cpu(),
                "donor_target": source_target[index].cpu(),
                "recipient_anchor": recipient_anchor[index].cpu(),
                "recipient_target": recipient_target[index].cpu(),
                "anchor_pca": pca.patch_image(anchor_feature),
                "prediction_pca": pca.patch_image(prediction),
                "target_pca": pca.patch_image(target),
            }
    else:
        for row in result_rows:
            row["feature_status"] = "N/A: checkpoint has no feature decoder"

    if "action" in adapter.capabilities:
        with _autocast(device, precision):
            source_decoded = adapter.decode_action(source_z).squeeze(1).float()
            recipient_decoded = (
                source_decoded
                if is_direct
                else adapter.decode_action(recipient_z).squeeze(1).float()
            )
        source_gt = _action_tensor(records, "source", device)
        recipient_gt = _action_tensor(records, "recipient", device)
        source_metrics = action_metrics(
            source_decoded,
            source_gt,
            waypoint_spacing=_spacing_tensor(records, "source", device),
            action_min=config["sampling"]["action_min"],
            action_max=config["sampling"]["action_max"],
        )
        swapped_metrics = action_metrics(
            source_decoded,
            recipient_gt,
            waypoint_spacing=_spacing_tensor(records, "recipient", device),
            action_min=config["sampling"]["action_min"],
            action_max=config["sampling"]["action_max"],
        )
        oracle_metrics = action_metrics(
            recipient_decoded,
            recipient_gt,
            waypoint_spacing=_spacing_tensor(records, "recipient", device),
            action_min=config["sampling"]["action_min"],
            action_max=config["sampling"]["action_max"],
        )
        degradation = positive_degradation(swapped_metrics, oracle_metrics)
        # RMSE is nonlinear across pairs: a per-pair difference of absolute yaw
        # errors cannot be aggregated into ``swapped RMSE - oracle RMSE``. Keep
        # both auditable controls and omit this misleading derived column; the
        # linear MAE/MSE/EPE degradations remain well-defined.
        degradation.pop("yaw_rmse_degrees", None)
        _append_metric_group(result_rows, "action_swapped_vs_donor", source_metrics)
        _append_metric_group(
            result_rows, "action_swapped_vs_recipient", swapped_metrics
        )
        _append_metric_group(result_rows, "action_oracle", oracle_metrics)
        _append_metric_group(result_rows, "action_degradation", degradation)
        action_min = config["sampling"]["action_min"]
        action_max = config["sampling"]["action_max"]
        overlay_size = (
            int(config["visualization"]["thumbnail_width"]),
            int(config["visualization"]["thumbnail_height"]),
        )
        for index, record in enumerate(records):
            decoded = _decoded_waypoint(source_decoded[index], action_min, action_max)
            normalized = source_decoded[index].detach().float().cpu().clone()
            normalized[2] = torch.atan2(
                torch.sin(normalized[2]), torch.cos(normalized[2])
            )
            recipient_spacing = float(record["recipient_waypoint_spacing"])
            source_spacing = float(record["source_waypoint_spacing"])
            decoded_recipient_meters = decoded.copy()
            decoded_recipient_meters[:2] *= recipient_spacing
            decoded_source_meters = decoded.copy()
            decoded_source_meters[:2] *= source_spacing
            result_rows[index]["decoded_action_normalized"] = _tensor_values(normalized)
            result_rows[index]["decoded_action_waypoint"] = decoded.tolist()
            result_rows[index]["decoded_action_meters"] = (
                decoded_recipient_meters.tolist()
            )
            result_rows[index]["decoded_action_meters_recipient_semantics"] = (
                decoded_recipient_meters.tolist()
            )
            result_rows[index]["decoded_action_meters_donor_semantics"] = (
                decoded_source_meters.tolist()
            )
            if not record["is_visual"]:
                continue
            donor_gt = np.asarray(record["source_action_waypoint"], dtype=np.float32)
            recipient_gt_wp = np.asarray(
                record["recipient_action_waypoint"], dtype=np.float32
            )
            payload = visuals.setdefault(
                str(record["record_id"]),
                {
                    "label": _visual_label(record),
                    "anchor": recipient_anchor[index].cpu(),
                    "target": recipient_target[index].cpu(),
                    "donor_anchor": source_anchor[index].cpu(),
                    "donor_target": source_target[index].cpu(),
                    "recipient_anchor": recipient_anchor[index].cpu(),
                    "recipient_target": recipient_target[index].cpu(),
                },
            )
            payload["action_overlay"] = action_overlay(
                recipient_gt_wp, decoded, size=overlay_size
            )
            payload["donor_action_overlay"] = action_overlay(
                donor_gt, decoded, size=overlay_size
            )
            payload["recipient_action_overlay"] = payload["action_overlay"]
            payload["action_overlay_caption"] = (
                f"decoded=({decoded[0]:+.2f},{decoded[1]:+.2f},"
                f"{math.degrees(decoded[2]):+.1f} deg)"
            )
            # PixelAction sheets remain image-centric, but the action head is
            # still visible numerically below the predicted recipient frame.
            payload["prediction_caption"] = payload["action_overlay_caption"]
    else:
        for row in result_rows:
            row["action_status"] = "N/A: checkpoint has no action decoder"

    return result_rows, visuals


def _render_condition_visuals(
    adapter: EvaluationAdapter,
    condition_id: str,
    condition_records: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    visual_records = [record for record in condition_records if record["is_visual"]]
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in visual_records:
        groups[int(record["visual_index"])].append(record)
    offsets = list(config["sampling"]["offsets"])
    cross = str(condition_records[0]["protocol"]).endswith("_cd")
    entries: list[dict[str, Any]] = []
    for visual_index in sorted(groups):
        records = sorted(
            groups[visual_index], key=lambda item: int(item["offset_index"])
        )
        present = [int(record["source_offset"]) for record in records]
        if present != offsets:
            raise RuntimeError(
                f"{condition_id}: visual group {visual_index} has offsets {present}, "
                f"expected {offsets}"
            )
        rows = [dict(payloads[str(record["record_id"])]) for record in records]
        path = output_dir / "visualizations" / f"visual_{visual_index:03d}.png"
        title = f"{condition_id} | visual window {visual_index}"
        layout = _visual_layout(config)
        if "rgb" in adapter.capabilities:
            render_pixel_sheet(rows, path, title=title, cross=cross, **layout)
        elif "feature" in adapter.capabilities:
            render_feature_sheet(rows, path, title=title, cross=cross, **layout)
        elif "action" in adapter.capabilities:
            render_action_sheet(rows, path, title=title, cross=cross, **layout)
        else:  # pragma: no cover - adapter registration forbids this state
            raise RuntimeError("adapter has no evaluable capability")
        entries.append(
            {
                "condition_id": condition_id,
                "visual_index": visual_index,
                "title": title,
                "path": path.relative_to(output_dir).as_posix(),
            }
        )
    expected = int(config["sampling"]["visual_count"])
    if len(groups) != expected:
        raise RuntimeError(
            f"{condition_id}: generated {len(groups)} visual windows, expected {expected}"
        )
    return entries


def _reference_frames(
    records: Sequence[Mapping[str, Any]], count: int, seed: int
) -> tuple[list[tuple[str, str, str]], str]:
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        grouped[(str(record["source_dataset"]), str(record["source_split"]))].add(
            str(record["source_target_path"])
        )
        grouped[(str(record["recipient_dataset"]), str(record["recipient_split"]))].add(
            str(record["recipient_target_path"])
        )
    selected: list[tuple[str, str, str]] = []
    for key in sorted(grouped):
        paths = sorted(grouped[key])
        rng = random.Random(
            int.from_bytes(
                hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode()).digest()[:8],
                "little",
            )
        )
        rng.shuffle(paths)
        selected.extend((key[0], key[1], path) for path in paths[:count])
    digest = hashlib.sha256(_canonical_json(selected).encode("utf-8")).hexdigest()
    return selected, digest



def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
