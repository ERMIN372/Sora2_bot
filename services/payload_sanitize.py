"""Utilities for sanitising provider payloads."""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Set

log = logging.getLogger(__name__)


def _str_or_none(value: Any) -> Optional[str]:
    """Return ``None`` if *value* is ``None`` otherwise ``str(value)``."""

    return None if value is None else str(value)


def normalize_sora_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *payload* with Sora-specific type coercions applied."""

    data: Dict[str, Any] = dict(payload or {})

    if "duration_sec" in data:
        data["duration_sec"] = _str_or_none(data.get("duration_sec"))

    for key in ("aspect_ratio", "format"):
        if key in data:
            data[key] = _str_or_none(data.get(key))

    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
    else:
        metadata = {}

    if "duration_sec" in metadata:
        metadata["duration_sec"] = _str_or_none(metadata.get("duration_sec"))

    if "seed" in metadata and metadata["seed"] is not None and not isinstance(
        metadata["seed"], (int, float, str)
    ):
        metadata["seed"] = str(metadata["seed"])

    data["metadata"] = metadata

    top_duration = data.get("duration_sec")
    metadata_duration = metadata.get("duration_sec")

    if top_duration and not metadata_duration:
        metadata["duration_sec"] = top_duration
    elif metadata_duration and not top_duration:
        data["duration_sec"] = metadata_duration

    return data


def sanitize_payload(payload: Optional[Mapping[str, Any]], allowed: Set[str]) -> Dict[str, Any]:
    """Return a shallow copy of *payload* limited to *allowed* keys."""

    if not payload:
        return {}
    return {key: value for key, value in payload.items() if key in allowed}


def log_removed_keys(
    tag: str,
    original: Optional[Mapping[str, Any]],
    cleaned: Mapping[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log keys present in *original* but missing in *cleaned*."""

    source: Mapping[str, Any] = original or {}
    removed = sorted({key for key in source.keys() if key not in cleaned})
    if not removed:
        return
    target_logger = logger or log
    target_logger.info("[%s] removed unsupported keys: %s", tag, removed)

__all__ = ["sanitize_payload", "log_removed_keys", "normalize_sora_payload"]
