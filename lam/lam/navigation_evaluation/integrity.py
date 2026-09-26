"""Small legacy manifest identifiers; no file or dataset scanning."""
from __future__ import annotations
import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

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

def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )

def manifest_metadata_sha256(metadata: Mapping[str, Any]) -> str:
    """Hash every semantic sidecar field without self-referential identities."""

    semantic = {
        key: value
        for key, value in metadata.items()
        if key not in _METADATA_IDENTITY_EXCLUDED_FIELDS
    }
    return hashlib.sha256(_canonical_json(semantic).encode("utf-8")).hexdigest()

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
