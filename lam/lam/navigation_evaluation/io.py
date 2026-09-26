"""Small atomic-I/O and provenance helpers used by evaluation commands."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def atomic_write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    return atomic_write_text(
        path,
        "".join(
            json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
    )


def write_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> Path:
    output = Path(path).expanduser().resolve()
    if not rows:
        return atomic_write_text(output, "")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _untracked_content_hash(repo_root: Path) -> tuple[str | None, int]:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None, 0
    relative_paths = sorted(path for path in result.stdout.split(b"\0") if path)
    digest = hashlib.sha256()
    hashed_count = 0
    for relative_bytes in relative_paths:
        relative = relative_bytes.decode("utf-8", errors="surrogateescape")
        path = (repo_root / relative).resolve()
        try:
            path.relative_to(repo_root)
        except ValueError:
            continue
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
        elif path.is_file():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            digest.update(b"<missing-or-special>")
        digest.update(b"\0")
        hashed_count += 1
    return digest.hexdigest(), hashed_count


def runtime_provenance(
    repo_root: str | Path, *, device: str | torch.device | None = None
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    diff = _git(root, "diff", "--binary", "HEAD")
    status = _git(root, "status", "--short")
    tracked_diff_hash = (
        None if diff is None else hashlib.sha256(diff.encode("utf-8")).hexdigest()
    )
    untracked_hash, untracked_count = _untracked_content_hash(root)
    dirty_identity = None
    if tracked_diff_hash is not None and untracked_hash is not None:
        dirty_identity = hashlib.sha256(
            f"{tracked_diff_hash}:{untracked_hash}".encode("ascii")
        ).hexdigest()
    dependencies: dict[str, str | None] = {}
    for distribution in (
        "numpy",
        "pillow",
        "piq",
        "torchvision",
        "lightning",
        "pandas",
        "pyarrow",
        "scipy",
        "PyYAML",
    ):
        try:
            dependencies[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            dependencies[distribution] = None
    gpu: dict[str, Any] | None = None
    requested_device = None if device is None else torch.device(device)
    if requested_device is not None and requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"cannot record provenance for unavailable CUDA device {requested_device}"
            )
        index = (
            torch.cuda.current_device()
            if requested_device.index is None
            else requested_device.index
        )
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        }
    elif requested_device is None and torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        }
    return {
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status": status,
        "git_diff_sha256": tracked_diff_hash,
        "git_untracked_sha256": untracked_hash,
        "git_untracked_file_count": untracked_count,
        "git_dirty_state_sha256": dirty_identity,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "dependencies": dependencies,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "requested_device": None if requested_device is None else str(requested_device),
        "gpu": gpu,
    }
