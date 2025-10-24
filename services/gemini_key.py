from __future__ import annotations

import logging
import os
from typing import Optional, Set, Tuple

from config import Config

log = logging.getLogger(__name__)

_LOGGED_CONTEXTS: Set[Tuple[str, str, str]] = set()


def mask_gemini_key(api_key: str) -> str:
    """Return a masked representation of ``api_key`` suitable for logs."""

    key = (api_key or "").strip()
    if not key:
        return ""
    prefix = key[:4]
    if len(prefix) < 4:
        prefix = (prefix + "****")[:4]
    return f"{prefix}…XXXX"


def ensure_gemini_key_logged(
    config: Config,
    *,
    context: str,
    process_id: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Ensure the Gemini API key mask is logged for *context* and return it."""

    target_logger = logger or log
    api_key = (config.gemini_api_key or "").strip()
    environment = (config.environment or "dev").lower()
    pid_label = process_id or f"pid={os.getpid()}"
    mask = mask_gemini_key(api_key)
    if not api_key:
        target_logger.error(
            "Gemini API key missing context=%s env=%s process=%s",
            context,
            environment,
            pid_label,
        )
        return mask
    signature = (environment, mask, pid_label)
    if signature not in _LOGGED_CONTEXTS:
        _LOGGED_CONTEXTS.add(signature)
        target_logger.info(
            "Gemini API key mask: %s env=%s process=%s context=%s",
            mask,
            environment,
            pid_label,
            context,
        )
    return mask


def current_key_mask(config: Config) -> str:
    """Return the current Gemini key mask (may be empty)."""

    return mask_gemini_key(config.gemini_api_key)


__all__ = ["current_key_mask", "ensure_gemini_key_logged", "mask_gemini_key"]
