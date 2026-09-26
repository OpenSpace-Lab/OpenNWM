"""Shared Torch Hub cache location for evaluation weights."""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def evaluation_torch_home(asset_root: str | Path) -> Iterator[Path]:
    """Use the requested cache for this operation, then restore TORCH_HOME."""
    torch_home = Path(asset_root).expanduser().resolve() / "torch"
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
