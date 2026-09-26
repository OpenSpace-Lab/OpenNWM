from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from lam.navigation_evaluation import reporting
from lam.navigation_evaluation.reporting import (
    aggregate_rows,
    bootstrap_mean_ci,
    compare_runs,
    metric_columns,
    summary_markdown,
    write_gallery,
    write_run_reports,
)
from lam.navigation_evaluation.visualization import (
    FeaturePCA,
    action_overlay,
    cosine_error_map,
    render_action_sheet,
    render_feature_sheet,
    render_pixel_sheet,
)


def _feature_frames() -> list[np.ndarray]:
    rng = np.random.default_rng(12)
    scale = np.asarray([9.0, 5.0, 3.0, 1.5, 0.8, 0.4, 0.2, 0.1])
    return [
        (rng.normal(size=(4, 5, 8)) * scale).astype(np.float32),
        (rng.normal(size=(6, 5, 8)) * scale + 0.3).astype(np.float32),
    ]


def test_feature_pca_is_deterministic_canonical_and_does_not_change_torch_rng():
    frames = _feature_frames()
    torch.manual_seed(91)
    state = torch.random.get_rng_state().clone()
    first = FeaturePCA.fit(frames, seed=7, max_tokens=50)
    assert torch.equal(torch.random.get_rng_state(), state)
    second = FeaturePCA.fit(frames, seed=7, max_tokens=50)

    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.components, second.components)
    np.testing.assert_array_equal(first.lower, second.lower)
    np.testing.assert_array_equal(first.upper, second.upper)
    for component in first.components.T:
        pivot = int(np.argmax(np.abs(component)))
        assert component[pivot] > 0

    projected = first.project(frames[0])
    assert projected.shape == (4, 5, 3)
    assert np.isfinite(projected).all()
    assert projected.min() >= 0 and projected.max() <= 1
    assert first.patch_image(frames[0], patch_height=4, patch_width=5).shape == (
        4,
        5,
        3,
    )


def test_feature_pca_save_load_exact_path_and_shape_validation(tmp_path: Path):
    fitted = FeaturePCA.fit(_feature_frames(), seed=11, max_tokens=40)
    path = tmp_path / "fixed_pca_basis"
    assert fitted.save(path) == path.resolve()
    assert path.is_file()
    assert not (tmp_path / "fixed_pca_basis.npz").exists()
    loaded = FeaturePCA.load(path)
    np.testing.assert_array_equal(loaded.components, fitted.components)
    np.testing.assert_array_equal(
        loaded.project(_feature_frames()[0]), fitted.project(_feature_frames()[0])
    )

    with pytest.raises(ValueError, match="token count"):
        fitted.patch_image(_feature_frames()[0], patch_height=3, patch_width=3)
    with pytest.raises(ValueError, match="feature dim"):
        fitted.project(np.zeros((5, 9), dtype=np.float32))
    with pytest.raises(ValueError, match="percentiles"):
        FeaturePCA.fit(_feature_frames(), lower_percentile=99, upper_percentile=1)


def test_cosine_error_map_known_colors_zero_vectors_nan_and_shapes():
    prediction = np.asarray(
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0], [math.nan, 0, 0]],
        dtype=np.float32,
    )
    target = np.asarray(
        [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0]],
        dtype=np.float32,
    )
    image = cosine_error_map(prediction, target, patch_height=1, patch_width=5)
    assert image.shape == (1, 5, 3)
    np.testing.assert_allclose(image[0, 0], [0, 0, 1])
    np.testing.assert_allclose(image[0, 1], [1, 1, 0])
    np.testing.assert_allclose(image[0, 2], [1, 0, 0])
    np.testing.assert_allclose(image[0, 3], [0, 0, 1])
    np.testing.assert_allclose(image[0, 4], [1, 0, 0])
    with pytest.raises(ValueError, match="same shape"):
        cosine_error_map(prediction, target[:-1], patch_height=1, patch_width=5)
    with pytest.raises(ValueError, match="token count"):
        cosine_error_map(prediction, target, patch_height=2, patch_width=3)


def _rgb(value: float, *, chw: bool = False) -> np.ndarray:
    result = np.full((18, 24, 3), value, dtype=np.float32)
    return np.moveaxis(result, -1, 0) if chw else result


def _assert_sheet(path: Path, expected_width: int) -> None:
    assert path.is_file()
    with Image.open(path) as image:
        image.load()
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.width == expected_width
        assert image.height == 52 + 58 + (36 + 72)
        assert np.asarray(image).std() > 0


def test_pixel_feature_and_action_sheets_are_clear_and_support_na(tmp_path: Path):
    layout = {"width": 48, "height": 36, "label_width": 90, "font_size": 9}
    pixel = render_pixel_sheet(
        [
            {
                "offset": -16,
                "label": "offset=-16\ndonor action=(1, 2, 3)",
                "donor_anchor": _rgb(0.1, chw=True),
                "donor_target": _rgb(0.2),
                "recipient_anchor": _rgb(0.3),
                "prediction": _rgb(0.4),
                "target": None,
            }
        ],
        tmp_path / "pixel.png",
        title="pixel cross",
        cross=True,
        **layout,
    )
    _assert_sheet(pixel, 90 + 5 * 48)

    pixel_action = render_pixel_sheet(
        [
            {
                "offset": -16,
                "donor_anchor": _rgb(0.1),
                "donor_target": _rgb(0.2),
                "recipient_anchor": _rgb(0.3),
                "prediction": _rgb(0.4),
                "target": _rgb(0.5),
                "donor_action_overlay": action_overlay(
                    [1.0, 0.5, 0.1], [0.8, 0.4, 0.2], size=(48, 36)
                ),
                "recipient_action_overlay": action_overlay(
                    [0.9, 0.4, 0.1], [0.8, 0.4, 0.2], size=(48, 36)
                ),
            }
        ],
        tmp_path / "pixel-action.png",
        title="pixel action cross",
        cross=True,
        **layout,
    )
    _assert_sheet(pixel_action, 90 + 7 * 48)

    feature = render_feature_sheet(
        [
            {
                "offset": 0,
                "anchor": _rgb(0.1),
                "target_rgb": _rgb(0.2),
                "anchor_pca": _rgb(0.3),
                "prediction_pca": _rgb(0.4),
                "target_pca": _rgb(0.5),
            }
        ],
        tmp_path / "feature.png",
        title="feature direct",
        cross=False,
        **layout,
    )
    _assert_sheet(feature, 90 + 5 * 48)

    feature_cross = render_feature_sheet(
        [
            {
                "offset": 0,
                "donor_anchor": _rgb(0.1),
                "donor_target": _rgb(0.2),
                "recipient_anchor": _rgb(0.3),
                "recipient_target": _rgb(0.4),
                "anchor_pca": _rgb(0.5),
                "prediction_pca": _rgb(0.6),
                "target_pca": _rgb(0.7),
            }
        ],
        tmp_path / "feature-cross.png",
        title="feature cross",
        cross=True,
        **layout,
    )
    _assert_sheet(feature_cross, 90 + 7 * 48)

    donor_overlay = action_overlay([1, 0.5, 0.1], [0.8, 0.4, 0.2], size=(48, 36))
    recipient_overlay = action_overlay([0.9, 0.4, 0.1], [0.8, 0.4, 0.2], size=(48, 36))
    action = render_action_sheet(
        [
            {
                "offset": 16,
                "donor_anchor": _rgb(0.1),
                "donor_target": _rgb(0.2),
                "recipient_anchor": _rgb(0.3),
                "recipient_target": _rgb(0.4),
                "donor_action_overlay": donor_overlay,
                "recipient_action_overlay": recipient_overlay,
            }
        ],
        tmp_path / "action.png",
        title="action cross",
        cross=True,
        **layout,
    )
    _assert_sheet(action, 90 + 6 * 48)
    assert action_overlay([1, 2, math.nan], [1, 2, 3], size=(48, 36)).size == (
        48,
        36,
    )
    with pytest.raises(MemoryError, match="Render fewer rows"):
        render_pixel_sheet(
            [{"offset": 0}],
            tmp_path / "too_large.png",
            title="large",
            cross=False,
            max_pixels=10,
        )


def _direct_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    offset_base = {-1: 1.0, 0: 100.0, 1: 3.0}
    for window in range(4):
        adjustment = window - 1.5
        for offset, base in offset_base.items():
            rows.append(
                {
                    "record_id": f"direct-{window}-{offset}",
                    "condition_id": "id|recon",
                    "protocol": "id",
                    "source_dataset": "recon",
                    "source_split": "test",
                    "source_trajectory": f"trajectory-{window}",
                    "source_anchor": 100 + window,
                    "source_offset": offset,
                    "recipient_dataset": "recon",
                    "recipient_split": "test",
                    "recipient_trajectory": f"trajectory-{window}",
                    "window_index": window,
                    "is_visual": window == 0,
                    "source_waypoint_spacing": 0.25,
                    "match_cost": 0.01,
                    "candidate_coverage": 0.9,
                    "rgb_swapped_mse": base + adjustment,
                    "rgb_oracle_lpips": math.nan if window == 0 else None,
                }
            )
    return rows


def test_metric_whitelist_cluster_bootstrap_and_primary_zero_policy():
    rows = _direct_rows()
    assert metric_columns(rows) == ["rgb_oracle_lpips", "rgb_swapped_mse"]
    first = aggregate_rows(rows, offsets=[-1, 0, 1], bootstrap_samples=300, seed=8)
    second = aggregate_rows(rows, offsets=[-1, 0, 1], bootstrap_samples=300, seed=8)
    assert first == second
    condition = first["conditions"]["id|recon"]
    metric = condition["primary"]["rgb_swapped_mse"]
    assert metric["macro_mean"] == pytest.approx(2.0)
    assert metric["bootstrap_unit"] == "anchor_window_cluster"
    assert condition["secondary_all_offsets"]["rgb_swapped_mse"][
        "macro_mean"
    ] == pytest.approx((1 + 100 + 3) / 3)
    assert condition["offsets"]["0"]["rgb_oracle_lpips"] == {
        "count": 0,
        "sample_count": 4,
        "finite_count": 0,
        "nonfinite_count": 1,
        "positive_infinity_count": 0,
        "negative_infinity_count": 0,
        "nan_count": 1,
        "missing_count": 3,
        "status": "nonfinite_values",
        "mean": None,
        "std": None,
        "median": None,
        "ci_lower": None,
        "ci_upper": None,
    }
    assert first["primary_offsets"] == [-1, 1]
    assert first["secondary_offsets"] == [-1, 0, 1]


def test_positive_infinity_psnr_is_semantic_but_never_silently_averaged(
    tmp_path: Path,
):
    offsets = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
    rows = []
    for window in range(3):
        for offset in offsets:
            rows.append(
                {
                    "record_id": f"psnr-{window}-{offset}",
                    "condition_id": "id__recon__test",
                    "protocol": "id",
                    "source_dataset": "recon",
                    "source_split": "test",
                    "source_trajectory": f"trajectory-{window}",
                    "source_anchor": 100 + window,
                    "source_offset": offset,
                    "recipient_dataset": "recon",
                    "recipient_split": "test",
                    "recipient_trajectory": f"trajectory-{window}",
                    "rgb_swapped_psnr": (
                        math.inf if offset == 0 else 30.0 + window + abs(offset) / 64
                    ),
                }
            )
    summary = aggregate_rows(rows, offsets=offsets, bootstrap_samples=30, seed=7)
    condition = summary["conditions"]["id__recon__test"]
    zero = condition["offsets"]["0"]["rgb_swapped_psnr"]
    assert zero["status"] == "positive_infinity"
    assert zero["finite_count"] == 0
    assert zero["nonfinite_count"] == zero["positive_infinity_count"] == 3
    assert zero["mean"] is None

    primary = condition["primary"]["rgb_swapped_psnr"]
    assert primary["status"] == "ok"
    assert primary["configured_offset_count"] == primary["offset_count"] == 8
    assert primary["nonfinite_offset_count"] == 0
    assert primary["macro_mean"] is not None

    secondary = condition["secondary_all_offsets"]["rgb_swapped_psnr"]
    assert secondary["status"] == "positive_infinity"
    assert secondary["configured_offset_count"] == secondary["offset_count"] == 9
    assert secondary["finite_offset_count"] == 8
    assert secondary["nonfinite_offset_count"] == 1
    assert secondary["finite_count"] == 24
    assert secondary["nonfinite_count"] == 3
    assert secondary["macro_mean"] is None
    assert secondary["ci_lower"] is None and secondary["ci_upper"] is None

    outputs = write_run_reports(tmp_path, rows, summary, save_parquet=False)
    serialized = [
        json.loads(line)
        for line in Path(outputs["per_pair_jsonl"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    zero_rows = [row for row in serialized if row["source_offset"] == 0]
    assert all(row["rgb_swapped_psnr"] is None for row in zero_rows)
    assert all(
        row["metric_nonfinite"] == {"rgb_swapped_psnr": "positive_infinity"}
        for row in zero_rows
    )
    reconstructed = aggregate_rows(
        serialized, offsets=offsets, bootstrap_samples=30, seed=7
    )
    assert reconstructed == summary
    markdown = Path(outputs["summary_markdown"]).read_text(encoding="utf-8")
    assert "positive_infinity" in markdown


def test_yaw_rmse_uses_root_mean_square_at_offset_and_macro_levels():
    rows = []
    values = {
        -1: (0.0, 4.0),
        0: (3.0, 3.0),
        1: (0.0, 0.0),
    }
    for offset, observations in values.items():
        for window, observation in enumerate(observations):
            rows.append(
                {
                    "record_id": f"yaw-{offset}-{window}",
                    "condition_id": "id__recon__test",
                    "protocol": "id",
                    "source_dataset": "recon",
                    "source_split": "test",
                    "source_trajectory": f"trajectory-{window}",
                    "source_anchor": 100 + window,
                    "source_offset": offset,
                    "recipient_dataset": "recon",
                    "recipient_split": "test",
                    "recipient_trajectory": f"trajectory-{window}",
                    "action_swapped_vs_recipient_yaw_rmse_degrees": observation,
                    "rgb_status": "N/A: checkpoint has no RGB decoder",
                    "feature_status": "N/A: checkpoint has no feature decoder",
                }
            )
    first = aggregate_rows(rows, offsets=[-1, 0, 1], bootstrap_samples=200, seed=9)
    second = aggregate_rows(rows, offsets=[-1, 0, 1], bootstrap_samples=200, seed=9)
    assert first == second
    condition = first["conditions"]["id__recon__test"]
    negative = condition["offsets"]["-1"][
        "action_swapped_vs_recipient_yaw_rmse_degrees"
    ]
    assert negative["aggregation"] == "root_mean_square"
    assert negative["mean"] == pytest.approx(math.sqrt(8.0))
    assert negative["mean"] != pytest.approx(2.0)
    primary = condition["primary"]["action_swapped_vs_recipient_yaw_rmse_degrees"]
    assert primary["aggregation"] == "root_mean_square"
    assert primary["macro_mean"] == pytest.approx(math.sqrt(8.0) / 2.0)
    secondary = condition["secondary_all_offsets"][
        "action_swapped_vs_recipient_yaw_rmse_degrees"
    ]
    assert secondary["macro_mean"] == pytest.approx((math.sqrt(8.0) + 3.0) / 3.0)
    assert negative["ci_lower"] is not None and negative["ci_upper"] is not None


def test_cross_bootstrap_is_offset_stratified_and_bootstrap_memory_path_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    rows = []
    for offset in (-1, 0, 1):
        for pair in range(5):
            rows.append(
                {
                    "record_id": f"cross-{offset}-{pair}",
                    "condition_id": "id_cd__recon__scand",
                    "protocol": "id_cd",
                    "source_dataset": "recon",
                    "source_split": "train",
                    "source_trajectory": f"source-{pair}",
                    "source_anchor": 70 + pair,
                    "source_offset": offset,
                    "recipient_dataset": "scand",
                    "recipient_split": "test",
                    "recipient_trajectory": f"recipient-{pair}",
                    "feature_swapped_mse": offset + pair + 5.0,
                    "action_swapped_vs_donor_yaw_rmse_degrees": float(pair),
                    "match_cost": 0.05 * (pair + 1),
                    "match_translation_waypoint_l2": 0.1 * (pair + 1),
                    "match_yaw_degrees": 1.0 + pair,
                }
            )
    condition_stats = [
        {
            "condition_id": "id_cd__recon__scand",
            "offsets": {
                str(offset): {
                    "candidate_coverage": 0.8 + 0.05 * (offset + 1),
                    "no_match_rejection_rate": 0.2 - 0.05 * (offset + 1),
                    "rejection_rate": 0.2 - 0.05 * (offset + 1),
                    "donor_unselected_rate": 0.6,
                }
                for offset in (-1, 0, 1)
            },
        }
    ]
    summary = aggregate_rows(
        rows,
        offsets=[-1, 0, 1],
        bootstrap_samples=250,
        seed=4,
        condition_stats=condition_stats,
    )
    scope = summary["conditions"]["id_cd__recon__scand"]["primary"][
        "feature_swapped_mse"
    ]
    assert scope["bootstrap_unit"] == "offset_stratified_pair"
    assert scope["offset_count"] == 2
    yaw_scope = summary["conditions"]["id_cd__recon__scand"]["primary"][
        "action_swapped_vs_donor_yaw_rmse_degrees"
    ]
    assert yaw_scope["aggregation"] == "root_mean_square"
    assert yaw_scope["macro_mean"] == pytest.approx(math.sqrt(6.0))
    assert yaw_scope["bootstrap_unit"] == "offset_stratified_pair"
    diagnostics = summary["conditions"]["id_cd__recon__scand"]["cross_diagnostics"]
    match_cost = diagnostics["pair_gaps"]["match_cost"]
    assert match_cost["offsets"]["-1"]["mean"] == pytest.approx(0.15)
    assert match_cost["primary"]["status"] == "ok"
    translation = diagnostics["pair_gaps"]["match_translation_waypoint_l2"]
    assert translation["offsets"]["-1"]["mean"] == pytest.approx(0.3)
    assert translation["offsets"]["-1"]["std"] is not None
    assert translation["offsets"]["-1"]["median"] == pytest.approx(0.3)
    assert translation["offsets"]["-1"]["ci_lower"] is not None
    assert translation["secondary_all_offsets"]["status"] == "ok"
    matching = diagnostics["manifest_matching"]
    assert matching["status"] == "ok"
    assert matching["offsets"]["-1"]["candidate_coverage"] == pytest.approx(0.8)
    assert matching["offsets"]["0"]["no_match_rejection_rate"] == pytest.approx(0.15)
    assert matching["offsets"]["0"]["rejection_rate"] == pytest.approx(0.15)
    assert matching["offsets"]["0"]["donor_unselected_rate"] == pytest.approx(0.6)
    assert matching["overall"]["unique_source_windows"] == 5
    assert matching["overall"]["unique_source_trajectories"] == 5
    assert matching["overall"]["unique_recipient_pairs"] == 15
    assert matching["overall"]["unique_recipient_trajectories"] == 5

    reports = write_run_reports(tmp_path, rows, summary, save_parquet=False)
    summary_csv = Path(reports["summary_csv"]).read_text(encoding="utf-8")
    summary_md = Path(reports["summary_markdown"]).read_text(encoding="utf-8")
    for diagnostic in (
        "match_cost",
        "match_translation_waypoint_l2",
        "match_yaw_degrees",
        "candidate_coverage",
        "no_match_rejection_rate",
        "rejection_rate",
        "donor_unselected_rate",
        "unique_recipient_pairs",
        "unique_source_windows",
        "unique_source_trajectories",
        "unique_recipient_trajectories",
    ):
        assert diagnostic in summary_csv
        assert diagnostic in summary_md

    monkeypatch.setattr(reporting, "_MAX_BOOTSTRAP_INDEX_ELEMENTS", 4)
    values = np.arange(10, dtype=np.float64)
    assert bootstrap_mean_ci(values, samples=7, confidence=0.9, seed=22) == (
        bootstrap_mean_ci(values, samples=7, confidence=0.9, seed=22)
    )


def _strict_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def test_reports_are_strict_json_csv_markdown_and_parquet_required_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rows = _direct_rows()
    summary = aggregate_rows(rows, offsets=[-1, 0, 1], bootstrap_samples=20)
    outputs = write_run_reports(tmp_path / "success", rows, summary, save_parquet=True)
    jsonl_path = Path(outputs["per_pair_jsonl"])
    parsed = [
        json.loads(
            line,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
    ]
    assert parsed[0]["rgb_oracle_lpips"] is None
    assert parsed[0]["metric_nonfinite"] == {"rgb_oracle_lpips": "nan"}
    _strict_json(Path(outputs["summary_json"]))
    assert Path(outputs["per_pair_csv"]).is_file()
    assert Path(outputs["summary_csv"]).is_file()
    assert "N/A" in Path(outputs["summary_markdown"]).read_text(encoding="utf-8")
    heatmap_records = _strict_json(Path(outputs["heatmap_json"]))
    assert {
        (record["protocol"], record["scope"], record["metric"])
        for record in heatmap_records
    } >= {
        ("id", "primary", "rgb_swapped_mse"),
        ("id", "secondary_all_offsets", "rgb_swapped_mse"),
        ("id", "offset:+0", "rgb_swapped_mse"),
    }
    heatmap_html = Path(outputs["heatmap_html"]).read_text(encoding="utf-8")
    assert '<select id="protocol">' in heatmap_html
    assert '<select id="scope">' in heatmap_html
    assert '<select id="metric">' in heatmap_html
    assert "source ↓ / recipient →" in heatmap_html
    assert "rgb_swapped_mse" in heatmap_html
    assert "NaN" not in heatmap_html and "Infinity" not in heatmap_html
    assert Path(outputs["per_pair_parquet"]).is_file()
    assert "per_pair_parquet_error" not in outputs

    class _BrokenFrame:
        def to_parquet(self, *_args, **_kwargs):
            raise RuntimeError("missing parquet test engine")

    monkeypatch.setitem(
        sys.modules,
        "pandas",
        SimpleNamespace(DataFrame=lambda _rows: _BrokenFrame()),
    )
    stale_status = {
        "complete": True,
        "selection_complete": True,
        "full_manifest_complete": True,
    }
    failure_dir = tmp_path / "failure"
    failure_dir.mkdir()
    (failure_dir / "run_status.json").write_text(
        json.dumps(stale_status), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="save_parquet=true"):
        write_run_reports(failure_dir, rows[:1], summary, save_parquet=True)
    failed_status = _strict_json(failure_dir / "run_status.json")
    assert failed_status["complete"] is False
    assert failed_status["selection_complete"] is False
    assert failed_status["full_manifest_complete"] is False
    assert "missing parquet test engine" in failed_status["report_error"]
    assert not (tmp_path / "failure" / "per_pair.parquet").exists()
    assert not list((tmp_path / "failure").glob(".per_pair.parquet.*"))


def _write_comparison_run(
    path: Path,
    summary: dict,
    *,
    manifest_sha: str,
    checkpoint_class: str,
    global_step: int,
    checkpoint_sha: str,
    config_sha: str,
    evaluator_sha: str = "e" * 64,
    execution_environment_sha: str = "c" * 64,
    full_manifest_complete: bool = True,
    write_status: bool = True,
) -> None:
    path.parent.mkdir(parents=True)
    asset_sha = "b" * 64
    identity_payload = {
        "asset_sha256": asset_sha,
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config_sha,
        "evaluator_sha256": evaluator_sha,
        "execution_environment_sha256": execution_environment_sha,
        "manifest_sha256": manifest_sha,
    }
    run_identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    generation = f"generation-{global_step}-{checkpoint_sha[:8]}"
    published_summary = {
        **summary,
        "run_identity": run_identity,
        "publication_generation": generation,
    }
    path.write_text(json.dumps(published_summary), encoding="utf-8")
    provenance = {
        "run_identity": run_identity,
        "publication_generation": generation,
        "manifest": {"sha256": manifest_sha},
        "checkpoint": {
            "class_path": checkpoint_class,
            "global_step": global_step,
            "sha256": checkpoint_sha,
        },
        "evaluation_config_sha256": config_sha,
        "evaluator_sha256": evaluator_sha,
        "execution_environment_sha256": execution_environment_sha,
        "asset_sha256": asset_sha,
    }
    provenance_path = path.parent / "provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    if write_status:
        status = {
            "schema_version": 2,
            "run_identity": run_identity,
            "publication_generation": generation,
            "full_manifest_complete": full_manifest_complete,
            "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "provenance_sha256": hashlib.sha256(
                provenance_path.read_bytes()
            ).hexdigest(),
        }
        (path.parent / "run_status.json").write_text(
            json.dumps(status), encoding="utf-8"
        )


def test_gallery_paths_escaping_commonmark_and_compare(tmp_path: Path):
    visual = tmp_path / "visuals" / "space & name.png"
    visual.parent.mkdir()
    Image.new("RGB", (8, 8), "red").save(visual)
    gallery = write_gallery(
        tmp_path / "report",
        [
            {
                "condition_id": "id_cd|recon",
                "title": "A | [B] & <C>",
                "path": visual,
            },
            {
                "condition_id": "relative",
                "path": "assets/local image.png",
            },
        ],
    )
    html_text = Path(gallery["html"]).read_text(encoding="utf-8")
    markdown = Path(gallery["markdown"]).read_text(encoding="utf-8")
    assert str(tmp_path) not in html_text
    assert "../visuals/space%20%26%20name.png" in html_text
    assert "A | [B] &amp; &lt;C&gt;" in html_text
    assert "assets/local%20image.png" in markdown
    assert "\n\n![" in markdown

    summary = aggregate_rows(_direct_rows(), offsets=[-1, 0, 1], bootstrap_samples=10)
    first = tmp_path / "one" / "same" / "summary.json"
    second = tmp_path / "two" / "same" / "summary.json"
    manifest_sha = "f" * 64
    _write_comparison_run(
        first,
        summary,
        manifest_sha=manifest_sha,
        checkpoint_class="lam.model.LAM",
        global_step=100,
        checkpoint_sha="1" * 64,
        config_sha="a" * 64,
    )
    _write_comparison_run(
        second,
        summary,
        manifest_sha=manifest_sha,
        checkpoint_class="lam.navigation_variants.PixelActionLAM",
        global_step=200,
        checkpoint_sha="2" * 64,
        config_sha="a" * 64,
    )
    compared = compare_runs([first, second], tmp_path / "comparison")
    comparison_rows = _strict_json(Path(compared["json"]))
    assert {row["checkpoint_global_step"] for row in comparison_rows} == {100, 200}
    assert {row["manifest_sha256"] for row in comparison_rows} == {manifest_sha}
    assert {row["checkpoint_sha256"] for row in comparison_rows} == {
        "1" * 64,
        "2" * 64,
    }
    run_metadata = _strict_json(Path(compared["runs_json"]))
    assert len(run_metadata) == 2
    assert all(run["run_identity_sha256"] for run in run_metadata)
    assert len({run["summary_layout_sha256"] for run in run_metadata}) == 1
    assert Path(compared["csv"]).is_file()
    comparison_md = Path(compared["markdown"]).read_text(encoding="utf-8")
    for heading in (
        "protocol",
        "source dataset",
        "source split",
        "recipient dataset",
        "recipient split",
        "scope",
        "offset",
        "metric",
    ):
        assert heading in comparison_md
    assert "| offset | -1 |" in comparison_md
    assert "| primary | N/A |" in comparison_md
    assert summary_markdown(summary).endswith("\n")

    mismatched = tmp_path / "three" / "summary.json"
    _write_comparison_run(
        mismatched,
        summary,
        manifest_sha="e" * 64,
        checkpoint_class="lam.model.LAM",
        global_step=300,
        checkpoint_sha="3" * 64,
        config_sha="c" * 64,
    )
    with pytest.raises(RuntimeError, match="different sample manifests"):
        compare_runs([first, mismatched], tmp_path / "bad-comparison")

    missing_provenance = tmp_path / "missing" / "summary.json"
    missing_provenance.parent.mkdir()
    missing_provenance.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="requires provenance"):
        compare_runs([missing_provenance], tmp_path / "missing-comparison")


def test_compare_requires_full_run_status(tmp_path: Path):
    summary = aggregate_rows(_direct_rows(), offsets=[-1, 0, 1], bootstrap_samples=5)
    missing = tmp_path / "missing-status" / "summary.json"
    _write_comparison_run(
        missing,
        summary,
        manifest_sha="f" * 64,
        checkpoint_class="lam.model.LAM",
        global_step=1,
        checkpoint_sha="1" * 64,
        config_sha="a" * 64,
        write_status=False,
    )
    with pytest.raises(FileNotFoundError, match="requires run_status"):
        compare_runs([missing], tmp_path / "missing-status-comparison")

    partial = tmp_path / "partial-status" / "summary.json"
    _write_comparison_run(
        partial,
        summary,
        manifest_sha="f" * 64,
        checkpoint_class="lam.model.LAM",
        global_step=2,
        checkpoint_sha="2" * 64,
        config_sha="b" * 64,
        full_manifest_complete=False,
    )
    with pytest.raises(RuntimeError, match="full_manifest_complete=true"):
        compare_runs([partial], tmp_path / "partial-status-comparison")

    first = tmp_path / "config" / "first" / "summary.json"
    second = tmp_path / "config" / "second" / "summary.json"
    _write_comparison_run(
        first,
        summary,
        manifest_sha="f" * 64,
        checkpoint_class="lam.model.LAM",
        global_step=1,
        checkpoint_sha="1" * 64,
        config_sha="a" * 64,
    )
    _write_comparison_run(
        second,
        summary,
        manifest_sha="f" * 64,
        checkpoint_class="lam.model.LAM",
        global_step=2,
        checkpoint_sha="2" * 64,
        config_sha="b" * 64,
    )
    with pytest.raises(RuntimeError, match="different semantic evaluation configs"):
        compare_runs([first, second], tmp_path / "config" / "comparison")


def test_compare_rejects_different_evaluator_code(tmp_path: Path) -> None:
    summary = aggregate_rows(_direct_rows(), offsets=[-1, 0, 1], bootstrap_samples=5)
    first = tmp_path / "evaluator" / "first" / "summary.json"
    second = tmp_path / "evaluator" / "second" / "summary.json"
    common = {
        "manifest_sha": "f" * 64,
        "checkpoint_class": "lam.model.LAM",
        "config_sha": "a" * 64,
    }
    _write_comparison_run(
        first,
        summary,
        global_step=1,
        checkpoint_sha="1" * 64,
        evaluator_sha="e" * 64,
        **common,
    )
    _write_comparison_run(
        second,
        summary,
        global_step=2,
        checkpoint_sha="2" * 64,
        evaluator_sha="d" * 64,
        **common,
    )
    with pytest.raises(RuntimeError, match="different evaluator code"):
        compare_runs([first, second], tmp_path / "evaluator" / "comparison")


@pytest.mark.parametrize("field", ["evaluator_sha256", "execution_environment_sha256"])
def test_compare_requires_immutable_identity_field(tmp_path: Path, field: str) -> None:
    summary = aggregate_rows(_direct_rows(), offsets=[-1, 0, 1], bootstrap_samples=5)
    summary_path = tmp_path / f"missing-{field}" / "summary.json"
    _write_comparison_run(
        summary_path,
        summary,
        manifest_sha="f" * 64,
        checkpoint_class="lam.model.LAM",
        global_step=1,
        checkpoint_sha="1" * 64,
        config_sha="a" * 64,
    )
    provenance_path = summary_path.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.pop(field)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    status_path = summary_path.parent / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["provenance_sha256"] = hashlib.sha256(
        provenance_path.read_bytes()
    ).hexdigest()
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(RuntimeError, match=field):
        compare_runs([summary_path], summary_path.parent / "comparison")


def test_compare_rejects_different_execution_environments(tmp_path: Path) -> None:
    summary = aggregate_rows(_direct_rows(), offsets=[-1, 0, 1], bootstrap_samples=5)
    first = tmp_path / "environment" / "first" / "summary.json"
    second = tmp_path / "environment" / "second" / "summary.json"
    common = {
        "manifest_sha": "f" * 64,
        "checkpoint_class": "lam.model.LAM",
        "config_sha": "a" * 64,
        "evaluator_sha": "e" * 64,
    }
    _write_comparison_run(
        first,
        summary,
        global_step=1,
        checkpoint_sha="1" * 64,
        execution_environment_sha="c" * 64,
        **common,
    )
    _write_comparison_run(
        second,
        summary,
        global_step=2,
        checkpoint_sha="2" * 64,
        execution_environment_sha="d" * 64,
        **common,
    )
    with pytest.raises(RuntimeError, match="different execution environments"):
        compare_runs([first, second], tmp_path / "environment" / "comparison")


@pytest.mark.parametrize(
    ("artifact", "field", "message"),
    [
        ("summary.json", "publication_generation", "mixed publication generations"),
        ("provenance.json", "run_identity", "mixed report run identities"),
    ],
)
def test_compare_rejects_mixed_atomic_publication_snapshots(
    tmp_path: Path, artifact: str, field: str, message: str
) -> None:
    summary = aggregate_rows(_direct_rows(), offsets=[-1, 0, 1], bootstrap_samples=5)
    summary_path = tmp_path / "mixed" / "summary.json"
    _write_comparison_run(
        summary_path,
        summary,
        manifest_sha="f" * 64,
        checkpoint_class="lam.model.LAM",
        global_step=1,
        checkpoint_sha="1" * 64,
        config_sha="a" * 64,
    )
    artifact_path = summary_path.parent / artifact
    value = json.loads(artifact_path.read_text(encoding="utf-8"))
    value[field] = f"different-{field}"
    artifact_path.write_text(json.dumps(value), encoding="utf-8")
    # Update the status hash so the test reaches the cross-document identity
    # check rather than failing at the independent byte-integrity check.
    status_path = summary_path.parent / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["summary_sha256" if artifact == "summary.json" else "provenance_sha256"] = (
        hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    )
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        compare_runs([summary_path], tmp_path / "mixed-comparison")


def test_compare_hashes_the_same_bytes_it_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = aggregate_rows(_direct_rows(), offsets=[-1, 0, 1], bootstrap_samples=5)
    summary_path = tmp_path / "snapshot" / "summary.json"
    _write_comparison_run(
        summary_path,
        summary,
        manifest_sha="f" * 64,
        checkpoint_class="lam.model.LAM",
        global_step=1,
        checkpoint_sha="1" * 64,
        config_sha="a" * 64,
    )
    watched = {
        summary_path,
        summary_path.parent / "provenance.json",
        summary_path.parent / "run_status.json",
    }
    calls = {path: 0 for path in watched}
    original = Path.read_bytes

    def tracked(path: Path) -> bytes:
        if path in calls:
            calls[path] += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    reporting._compare_run_metadata(summary_path)
    assert calls == {path: 1 for path in watched}

    # A later byte change is rejected against the status-bound digest.
    monkeypatch.setattr(Path, "read_bytes", original)
    summary_path.write_bytes(summary_path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        compare_runs([summary_path], tmp_path / "tampered-comparison")


@pytest.mark.parametrize(
    ("layout_part", "message"),
    [
        ("condition_ids", "condition ID set"),
        ("primary_offsets", "primary_offsets layout"),
        ("secondary_offsets", "secondary_offsets layout"),
        ("metrics", "metric layout"),
    ],
)
def test_compare_rejects_different_summary_layouts(
    tmp_path: Path, layout_part: str, message: str
):
    summary = aggregate_rows(_direct_rows(), offsets=[-1, 0, 1], bootstrap_samples=5)
    changed = copy.deepcopy(summary)
    if layout_part == "condition_ids":
        condition = changed["conditions"].pop("id|recon")
        changed["conditions"]["different-condition"] = condition
    elif layout_part == "primary_offsets":
        changed["primary_offsets"] = [0]
    elif layout_part == "secondary_offsets":
        changed["secondary_offsets"] = [0, -1, 1]
    elif layout_part == "metrics":
        changed["metrics"] = list(reversed(changed["metrics"]))
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(layout_part)

    first = tmp_path / layout_part / "first" / "summary.json"
    second = tmp_path / layout_part / "second" / "summary.json"
    common = {
        "manifest_sha": "f" * 64,
        "checkpoint_class": "lam.model.LAM",
        "config_sha": "a" * 64,
    }
    _write_comparison_run(
        first,
        summary,
        global_step=1,
        checkpoint_sha="1" * 64,
        **common,
    )
    _write_comparison_run(
        second,
        changed,
        global_step=2,
        checkpoint_sha="2" * 64,
        **common,
    )
    with pytest.raises(RuntimeError, match=message):
        compare_runs([first, second], tmp_path / layout_part / "comparison")


def test_capability_na_is_explicit_and_compare_allows_different_modalities(
    tmp_path: Path,
):
    pixel_rows = copy.deepcopy(_direct_rows())
    for row in pixel_rows:
        row["feature_status"] = "N/A: checkpoint has no feature decoder"
        row["action_status"] = "N/A: checkpoint has no action decoder"
    action_rows = copy.deepcopy(_direct_rows())
    for row in action_rows:
        row.pop("rgb_swapped_mse")
        row.pop("rgb_oracle_lpips")
        row["rgb_status"] = "N/A: checkpoint has no RGB decoder"
        row["feature_status"] = "N/A: checkpoint has no feature decoder"
        row["action_swapped_vs_recipient_yaw_mae_degrees"] = 2.0

    pixel = aggregate_rows(pixel_rows, offsets=[-1, 0, 1], bootstrap_samples=5)
    action = aggregate_rows(action_rows, offsets=[-1, 0, 1], bootstrap_samples=5)
    pixel_capabilities = pixel["conditions"]["id|recon"]["capabilities"]
    action_capabilities = action["conditions"]["id|recon"]["capabilities"]
    assert pixel_capabilities["rgb"]["available"] is True
    assert pixel_capabilities["action"]["status"].startswith("N/A:")
    assert action_capabilities["rgb"]["available"] is False
    assert action_capabilities["action"]["available"] is True
    assert "Modality capabilities" in summary_markdown(pixel)

    first = tmp_path / "capability" / "pixel" / "summary.json"
    second = tmp_path / "capability" / "action" / "summary.json"
    common = {"manifest_sha": "f" * 64, "config_sha": "a" * 64}
    _write_comparison_run(
        first,
        pixel,
        checkpoint_class="lam.navigation_variants.PixelLAM",
        global_step=1,
        checkpoint_sha="1" * 64,
        **common,
    )
    _write_comparison_run(
        second,
        action,
        checkpoint_class="lam.navigation_variants.ActionOnlyLAM",
        global_step=2,
        checkpoint_sha="2" * 64,
        **common,
    )
    compared = compare_runs([first, second], tmp_path / "capability" / "comparison")
    comparison = _strict_json(Path(compared["json"]))
    capability_rows = [
        row for row in comparison if row.get("entry_type") == "capability"
    ]
    assert len(capability_rows) == 6
    assert any(
        row["metric"] == "action" and str(row["status"]).startswith("N/A:")
        for row in capability_rows
    )
    markdown = Path(compared["markdown"]).read_text(encoding="utf-8")
    assert "entry type" in markdown
    assert "N/A: checkpoint has no action decoder" in markdown
