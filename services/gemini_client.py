"""Shared Gemini client factory."""
from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Tuple

from google.genai import Client as _GenAIClient
from google.genai._api_client import HttpOptions as _HttpOptions

from config import Config
from services.gemini_key import ensure_gemini_key_logged

log = logging.getLogger(__name__)

_CLIENT_CACHE: Dict[Tuple[str, str, Tuple[int, int, int]], _GenAIClient] = {}
_CACHE_LOCK = threading.Lock()


def _build_http_options(config: Config) -> _HttpOptions:
    timeout_seconds: float | None = None
    candidates = []
    for value in (
        config.request_timeout,
        config.request_read_timeout,
        config.request_connect_timeout,
    ):
        if value:
            try:
                candidates.append(float(value))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
    if candidates:
        timeout_seconds = max(candidates)
    timeout_ms = int(timeout_seconds * 1000) if timeout_seconds else None
    api_version = (config.gemini_api_version or "").strip()
    return _HttpOptions(timeout=timeout_ms, api_version=api_version or None)


def get_gemini_client(config: Config) -> _GenAIClient:
    """Return a cached :class:`google.genai.Client` instance."""

    api_key = (config.gemini_api_key or "").strip()
    if not api_key:
        raise RuntimeError("Gemini API key is required to use Gemini services")
    ensure_gemini_key_logged(
        config,
        context="gemini_client",
        process_id=f"pid={os.getpid()}",
        logger=log,
    )
    cache_key = (
        api_key,
        (config.gemini_api_version or "v1").strip() or "v1",
        (
            int((config.request_timeout or 0) * 1000),
            int((config.request_read_timeout or 0) * 1000),
            int((config.request_connect_timeout or 0) * 1000),
        ),
    )
    with _CACHE_LOCK:
        client = _CLIENT_CACHE.get(cache_key)
        if client is None:
            http_options = _build_http_options(config)
            client = _GenAIClient(api_key=api_key, http_options=http_options)
            _CLIENT_CACHE[cache_key] = client
    return client


__all__ = ["get_gemini_client"]
