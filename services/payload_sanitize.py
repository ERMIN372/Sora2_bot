"""Utilities for sanitising provider payloads."""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Set

log = logging.getLogger(__name__)


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


__all__ = ["sanitize_payload", "log_removed_keys"]
