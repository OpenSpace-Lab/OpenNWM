"""Explicit preparation and offline verification of evaluation assets."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import piq
import torch
import torchvision

from .io import atomic_write_json, sha256_file

ASSET_METADATA_NAME = "assets.json"
LPIPS_FILENAMES = ("lpips_weights.pt", "vgg16-397923af.pth")


def _runtime_versions() -> dict[str, str]:
    return {
        "piq": str(getattr(piq, "__version__", "unknown")),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
    }


@contextmanager
def _deny_torch_hub_downloads() -> Iterator[None]:
    """Make an accidental cache miss fail instead of reaching the network."""

    original = torch.hub.download_url_to_file

    def denied(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(
            "formal LPIPS evaluation is strictly offline; run prepare-assets first"
        )

    torch.hub.download_url_to_file = denied
    try:
        yield
    finally:
        torch.hub.download_url_to_file = original


@contextmanager
def evaluation_torch_home(asset_root: str | Path) -> Iterator[Path]:
    """Point torch hub at an isolated, auditable asset directory."""

    root = Path(asset_root).expanduser().resolve()
    torch_home = root / "torch"
    torch_home.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("TORCH_HOME")
    os.environ["TORCH_HOME"] = str(torch_home)
    try:
        yield torch_home
    finally:
        if previous is None:
            os.environ.pop("TORCH_HOME", None)
        else:
            os.environ["TORCH_HOME"] = previous


def _checkpoint_inventory(torch_home: Path) -> dict[str, dict[str, Any]]:
    checkpoint_dir = torch_home / "hub" / "checkpoints"
    inventory: dict[str, dict[str, Any]] = {}
    for name in LPIPS_FILENAMES:
        path = checkpoint_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"LPIPS evaluation asset is missing after preparation: {path}"
            )
        inventory[name] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return inventory


def prepare_lpips_assets(asset_root: str | Path) -> dict[str, Any]:
    """Download LPIPS/VGG once and write a hash inventory.

    Network access is confined to this explicit command.  Normal evaluation
    calls :func:`verify_lpips_assets` before constructing the metric.
    """

    root = Path(asset_root).expanduser().resolve()
    with evaluation_torch_home(root) as torch_home:
        piq.LPIPS(reduction="none")
        files = _checkpoint_inventory(torch_home)
    metadata = {
        "schema_version": 1,
        **_runtime_versions(),
        "lpips_url": getattr(piq.LPIPS, "_weights_url", None),
        "files": files,
    }
    atomic_write_json(root / ASSET_METADATA_NAME, metadata)
    return metadata


def verify_lpips_assets(asset_root: str | Path) -> dict[str, Any]:
    """Verify every prepared asset without attempting a download."""

    import json

    root = Path(asset_root).expanduser().resolve()
    metadata_path = root / ASSET_METADATA_NAME
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"LPIPS assets are not prepared: {metadata_path}. "
            "Run navigation_eval.py prepare-assets first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise RuntimeError(f"unsupported LPIPS asset metadata: {metadata_path}")
    versions = _runtime_versions()
    mismatched_versions = {
        name: {"prepared": metadata.get(name), "runtime": version}
        for name, version in versions.items()
        if metadata.get(name) != version
    }
    if mismatched_versions:
        raise RuntimeError(
            "LPIPS asset/runtime version mismatch; rerun prepare-assets in the current "
            f"environment: {mismatched_versions}"
        )
    files = metadata.get("files")
    if not isinstance(files, dict):
        raise TypeError(f"malformed LPIPS asset metadata: {metadata_path}")
    for name in LPIPS_FILENAMES:
        entry = files.get(name)
        if not isinstance(entry, dict):
            raise TypeError(f"LPIPS asset metadata is missing {name}")
        path = Path(entry["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"LPIPS asset escapes configured root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"prepared LPIPS asset disappeared: {path}")
        if path.stat().st_size != int(entry["size"]):
            raise RuntimeError(f"LPIPS asset size mismatch: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"LPIPS asset hash mismatch: {path}")
    return metadata


def make_offline_lpips(asset_root: str | Path, device: torch.device) -> torch.nn.Module:
    """Construct one LPIPS instance after strict cache verification."""

    verify_lpips_assets(asset_root)
    with evaluation_torch_home(asset_root), _deny_torch_hub_downloads():
        metric = piq.LPIPS(reduction="none")
    metric.eval().requires_grad_(False)
    return metric.to(device)


__all__ = [
    "evaluation_torch_home",
    "make_offline_lpips",
    "prepare_lpips_assets",
    "verify_lpips_assets",
]
