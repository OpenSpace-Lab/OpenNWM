"""Content identity for every data file referenced by an evaluation manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .io import sha256_file

DATA_INTEGRITY_SCHEMA_VERSION = 1
EVALUATION_MANIFEST_IDENTITY_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_METADATA_IDENTITY_EXCLUDED_FIELDS = frozenset(
    {
        "evaluation_manifest_sha256",
        "manifest_path",
        "metadata_sha256",
        # Kept in the sidecar for audit, but it contains device, batching, and
        # output settings that do not define checkpoint-independent sampling.
        "resolved_config_sha256",
    }
)
_FRAME_PATH_FIELDS = (
    "source_anchor_path",
    "source_target_path",
    "recipient_anchor_path",
    "recipient_target_path",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _referenced_paths(records: Sequence[Mapping[str, Any]]) -> list[Path]:
    paths: set[Path] = set()
    frame_cache: dict[str, Path] = {}
    trajectory_metadata_cache: dict[Path, Path] = {}
    for index, record in enumerate(records):
        for field in _FRAME_PATH_FIELDS:
            raw = record.get(field)
            if not isinstance(raw, str) or not raw:
                raise ValueError(f"manifest record {index} is missing {field}")
            frame = frame_cache.get(raw)
            if frame is None:
                frame = Path(raw).expanduser().resolve(strict=True)
                frame_cache[raw] = frame
            if not frame.is_file():
                raise FileNotFoundError(f"referenced frame is not a file: {frame}")
            paths.add(frame)
            trajectory_metadata = trajectory_metadata_cache.get(frame.parent)
            if trajectory_metadata is None:
                trajectory_metadata = (frame.parent / "traj_data.pkl").resolve(
                    strict=True
                )
                trajectory_metadata_cache[frame.parent] = trajectory_metadata
            if not trajectory_metadata.is_file():
                raise FileNotFoundError(
                    f"trajectory metadata for referenced frame is missing: "
                    f"{trajectory_metadata}"
                )
            paths.add(trajectory_metadata)
    return sorted(paths, key=str)


def _fingerprint(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    stable_fields = ("st_size", "st_mtime_ns", "st_ctime_ns", "st_dev", "st_ino")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise RuntimeError(f"data file changed while hashing: {path}")
    return {
        "path": str(path),
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
        "device_id": after.st_dev,
        "inode": after.st_ino,
        "sha256": digest,
    }


def _stable_hash(path: Path) -> tuple[os.stat_result, str]:
    """Hash one path and reject a concurrent replacement or mutation."""

    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    stable_fields = ("st_size", "st_mtime_ns", "st_ctime_ns", "st_dev", "st_ino")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise RuntimeError(f"data file changed while verifying: {path}")
    return after, digest


def _content_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    identity = [
        {
            "path": str(entry["path"]),
            "size": int(entry["size"]),
            "sha256": str(entry["sha256"]),
        }
        for entry in entries
    ]
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def manifest_metadata_sha256(metadata: Mapping[str, Any]) -> str:
    """Hash every semantic sidecar field without self-referential identities."""

    semantic = {
        key: value
        for key, value in metadata.items()
        if key not in _METADATA_IDENTITY_EXCLUDED_FIELDS
    }
    return hashlib.sha256(_canonical_json(semantic).encode("utf-8")).hexdigest()


def build_data_integrity(
    records: Sequence[Mapping[str, Any]], *, workers: int | None = None
) -> dict[str, Any]:
    """Hash all unique referenced frames and trajectory metadata once."""

    paths = _referenced_paths(records)
    worker_count = min(16, max(1, os.cpu_count() or 1)) if workers is None else workers
    if (
        isinstance(worker_count, bool)
        or not isinstance(worker_count, int)
        or worker_count < 1
    ):
        raise ValueError("data integrity workers must be a positive integer")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        entries = list(executor.map(_fingerprint, paths))
    return {
        "schema_version": DATA_INTEGRITY_SCHEMA_VERSION,
        "file_count": len(entries),
        "total_size_bytes": sum(int(entry["size"]) for entry in entries),
        "content_sha256": _content_digest(entries),
        "files": entries,
    }


def verify_data_integrity(
    inventory: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify inventory coverage and current referenced-file identities.

    Matching size/mtime/ctime/device/inode values provide a fast verification
    path for immutable NAS data.  A changed stat identity triggers a stable
    content re-hash, so an atomic replacement with identical bytes remains
    valid while any byte drift fails before checkpoint inspection.
    """

    schema_version = inventory.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != DATA_INTEGRITY_SCHEMA_VERSION
    ):
        raise RuntimeError("manifest data integrity metadata is missing or unsupported")
    raw_entries = inventory.get("files")
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
        raise TypeError("manifest data integrity files must be an array")
    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    fast_verified = 0
    rehashed = 0
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise TypeError(f"data integrity entry {index} must be an object")
        raw_path = raw_entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise TypeError(f"data integrity entry {index} has no valid path")
        path = Path(raw_path).expanduser().resolve(strict=False)
        canonical_path = str(path)
        if raw_path != canonical_path:
            raise RuntimeError(
                f"data integrity entry {index} path is not canonical: {raw_path!r}"
            )
        if canonical_path in seen_paths:
            raise RuntimeError(f"duplicate data integrity path: {canonical_path}")
        seen_paths.add(canonical_path)
        if not path.is_file():
            raise FileNotFoundError(f"manifest data asset disappeared: {path}")
        expected_hash = raw_entry.get("sha256")
        if not isinstance(expected_hash, str) or not _SHA256_PATTERN.fullmatch(
            expected_hash
        ):
            raise RuntimeError(
                f"data integrity entry {index} has no valid SHA-256: {expected_hash!r}"
            )
        integer_fields = ("size", "mtime_ns", "ctime_ns", "device_id", "inode")
        for field in integer_fields:
            value = raw_entry.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(
                    f"data integrity entry {index}.{field} must be a non-negative integer"
                )
        entry = {
            "path": canonical_path,
            "size": int(raw_entry["size"]),
            "mtime_ns": int(raw_entry["mtime_ns"]),
            "ctime_ns": int(raw_entry["ctime_ns"]),
            "device_id": int(raw_entry["device_id"]),
            "inode": int(raw_entry["inode"]),
            "sha256": expected_hash,
        }
        stat = path.stat()
        expected_stat = (
            entry.get("size"),
            entry.get("mtime_ns"),
            entry.get("ctime_ns"),
            entry.get("device_id"),
            entry.get("inode"),
        )
        current_stat = (
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_dev,
            stat.st_ino,
        )
        if current_stat == expected_stat:
            fast_verified += 1
        else:
            stable_stat, actual_hash = _stable_hash(path)
            rehashed += 1
            if stable_stat.st_size != entry["size"] or actual_hash != entry["sha256"]:
                raise RuntimeError(
                    f"manifest data asset content changed: {path}; "
                    f"expected sha256={entry['sha256']}, actual={actual_hash}"
                )
        entries.append(entry)
    expected_paths = [str(path) for path in _referenced_paths(records)]
    actual_paths = [entry["path"] for entry in entries]
    if actual_paths != sorted(actual_paths):
        raise RuntimeError("manifest data integrity files are not canonically sorted")
    if actual_paths != expected_paths:
        missing = sorted(set(expected_paths) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(expected_paths))
        raise RuntimeError(
            "manifest data integrity inventory does not exactly cover referenced "
            f"files: missing={missing}, extra={extra}"
        )
    file_count = inventory.get("file_count")
    if isinstance(file_count, bool) or not isinstance(file_count, int):
        raise TypeError("manifest data integrity file_count must be an integer")
    if file_count != len(entries):
        raise RuntimeError(
            f"manifest data integrity file count mismatch: "
            f"metadata={file_count}, actual={len(entries)}"
        )
    expected_total_size = sum(entry["size"] for entry in entries)
    total_size = inventory.get("total_size_bytes")
    if isinstance(total_size, bool) or not isinstance(total_size, int):
        raise TypeError("manifest data integrity total_size_bytes must be an integer")
    if total_size != expected_total_size:
        raise RuntimeError(
            "manifest data integrity total size mismatch: "
            f"metadata={total_size}, actual={expected_total_size}"
        )
    expected_digest = _content_digest(entries)
    content_sha = inventory.get("content_sha256")
    if not isinstance(content_sha, str) or not _SHA256_PATTERN.fullmatch(content_sha):
        raise RuntimeError("manifest data integrity content SHA-256 is malformed")
    if content_sha != expected_digest:
        raise RuntimeError("manifest data integrity aggregate SHA-256 mismatch")
    return {
        "status": "verified",
        "file_count": len(entries),
        "total_size_bytes": expected_total_size,
        "content_sha256": expected_digest,
        "fast_stat_verified": fast_verified,
        "content_rehashed": rehashed,
    }


def evaluation_manifest_sha256(metadata: Mapping[str, Any]) -> str:
    """Bind sample/match rows and referenced data bytes into one run identity."""

    records_sha = metadata.get("records_sha256")
    metadata_sha = metadata.get("metadata_sha256")
    inventory = metadata.get("data_integrity")
    if not isinstance(records_sha, str) or not _SHA256_PATTERN.fullmatch(records_sha):
        raise RuntimeError("manifest metadata has no valid records SHA-256")
    if not isinstance(inventory, Mapping):
        raise TypeError("manifest metadata has no data integrity inventory")
    if not isinstance(metadata_sha, str) or not _SHA256_PATTERN.fullmatch(metadata_sha):
        raise RuntimeError("manifest metadata has no valid semantic metadata SHA-256")
    data_sha = inventory.get("content_sha256")
    if not isinstance(data_sha, str) or not _SHA256_PATTERN.fullmatch(data_sha):
        raise RuntimeError("manifest data integrity has no valid content SHA-256")
    payload = {
        "schema_version": EVALUATION_MANIFEST_IDENTITY_SCHEMA_VERSION,
        "records_sha256": records_sha,
        "metadata_sha256": metadata_sha,
        "data_content_sha256": data_sha,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def verify_evaluation_manifest_integrity(
    records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    verify_data_files: bool = True,
) -> dict[str, Any]:
    """Verify the stored inventory and composite evaluation-manifest identity."""

    inventory = metadata.get("data_integrity")
    if not isinstance(inventory, Mapping):
        raise TypeError("manifest metadata has no data integrity inventory")
    if verify_data_files:
        verification = verify_data_integrity(inventory, records=records)
    else:
        file_count = inventory.get("file_count")
        total_size = inventory.get("total_size_bytes")
        content_sha = inventory.get("content_sha256")
        if isinstance(file_count, bool) or not isinstance(file_count, int):
            raise TypeError("manifest data integrity file_count must be an integer")
        if isinstance(total_size, bool) or not isinstance(total_size, int):
            raise TypeError(
                "manifest data integrity total_size_bytes must be an integer"
            )
        if not isinstance(content_sha, str) or not _SHA256_PATTERN.fullmatch(
            content_sha
        ):
            raise RuntimeError("manifest data integrity content SHA-256 is malformed")
        verification = {
            "status": "trusted_manifest_inventory",
            "data_files_verified": False,
            "file_count": file_count,
            "total_size_bytes": total_size,
            "content_sha256": content_sha,
            "fast_stat_verified": 0,
            "content_rehashed": 0,
        }
    expected_metadata_sha = manifest_metadata_sha256(metadata)
    stored_metadata_sha = metadata.get("metadata_sha256")
    if stored_metadata_sha != expected_metadata_sha:
        raise RuntimeError(
            "manifest semantic metadata SHA-256 mismatch: "
            f"metadata={stored_metadata_sha!r}, actual={expected_metadata_sha}"
        )
    actual = evaluation_manifest_sha256(metadata)
    stored = metadata.get("evaluation_manifest_sha256")
    if stored != actual:
        raise RuntimeError(
            "evaluation manifest composite SHA-256 mismatch: "
            f"metadata={stored!r}, actual={actual}"
        )
    return {
        **verification,
        "metadata_sha256": expected_metadata_sha,
        "evaluation_manifest_sha256": actual,
    }


__all__ = [
    "build_data_integrity",
    "evaluation_manifest_sha256",
    "manifest_metadata_sha256",
    "verify_data_integrity",
    "verify_evaluation_manifest_integrity",
]
