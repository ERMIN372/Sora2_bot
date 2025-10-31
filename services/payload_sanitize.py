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

    data: Dict[str, Any] = {key: value for key, value in dict(payload or {}).items() if value is not None}

    if "duration_sec" in data:
        data["duration_sec"] = _str_or_none(data.get("duration_sec"))

    for key in ("aspect_ratio", "format"):
        if key in data:
            data[key] = _str_or_none(data.get(key))

    metadata = data.get("metadata")
    if isinstance(metadata, Mapping):
        duration_from_meta = metadata.get("duration_sec")
        if duration_from_meta is not None and "duration_sec" not in data:
            data["duration_sec"] = _str_or_none(duration_from_meta)
        seed_from_meta = metadata.get("seed")
        if seed_from_meta is not None and "seed" not in data:
            if not isinstance(seed_from_meta, (int, float, str)):
                data["seed"] = str(seed_from_meta)
            else:
                data["seed"] = seed_from_meta

    seed = data.get("seed")
    if seed is not None and not isinstance(seed, (int, float, str)):
        data["seed"] = str(seed)

    # Legacy metadata payloads are no longer supported; drop any provided metadata.
    if "metadata" in data:
        data.pop("metadata", None)

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
