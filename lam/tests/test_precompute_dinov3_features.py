from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from precompute_dinov3_features import (  # noqa: E402
    FEATURE_DIM,
    PATCH_COUNT,
    _WorkUnit,
    _balanced_assignments,
    _cache_key,
    _prepare_storage,
)


def test_balancing_is_deterministic_complete_and_bounded(tmp_path: Path):
    units = [
        _WorkUnit(0, index, index, index * 10, length, tmp_path)
        for index, length in enumerate((11, 9, 7, 5, 3, 1))
    ]
    first = _balanced_assignments(units, 3)
    second = _balanced_assignments(units, 3)
    assert first == second
    assert sorted(unit.status_index for rank in first for unit in rank) == list(range(6))
    loads = [sum(unit.length for unit in rank) for rank in first]
    assert max(loads) - min(loads) <= max(unit.length for unit in units)


def test_cache_key_ignores_publication_state_and_completion_time():
    payload = {"complete": True, "sources": [1, 2, 3], "frame_count": 7}
    completed = {**payload, "complete": False, "completed_at_utc": "later"}
    assert _cache_key(payload) == _cache_key(completed)


def test_storage_is_preallocated_once_and_then_resumable(tmp_path: Path):
    payload = {
        "complete": True,
        "frame_count": 1,
        "trajectory_count": 1,
        "sources": [
            {
                "anchor_count": 1,
                "feature_file": "source_0.fp16",
            }
        ],
    }
    assert _prepare_storage(tmp_path, payload) == "new"
    assert (tmp_path / "source_0.fp16").stat().st_size == PATCH_COUNT * FEATURE_DIM * 2
    assert (tmp_path / "trajectory_progress.uint8").stat().st_size == 1
    assert _prepare_storage(tmp_path, payload) == "resume"
