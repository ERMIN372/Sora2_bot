"""Shared Gemini client factory."""
from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Tuple

from google.genai import Client as _GenAIClient
from google.genai import types as _genai_types

from config import Config
from services.gemini_key import ensure_gemini_key_logged

log = logging.getLogger(__name__)

_CLIENT_CACHE: Dict[Tuple[str, str, Tuple[int, int, int]], _GenAIClient] = {}
_CACHE_LOCK = threading.Lock()


def _build_http_options(config: Config, *, api_version: str) -> _genai_types.HTTPOptions:
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
    version = (api_version or "").strip() or None
    return _genai_types.HTTPOptions(timeout=timeout_ms, api_version=version)


def _get_client(config: Config, *, api_version: str) -> _GenAIClient:
    """Return a cached :class:`google.genai.Client` instance for *api_version*."""

    api_key = (config.gemini_api_key or "").strip()
    if not api_key:
        raise RuntimeError("Gemini API key is required to use Gemini services")
    ensure_gemini_key_logged(
        config,
        context="gemini_client",
        process_id=f"pid={os.getpid()}",
        logger=log,
    )
    version = api_version.strip() or "v1"
    cache_key = (
        api_key,
        version,
        (
            int((config.request_timeout or 0) * 1000),
            int((config.request_read_timeout or 0) * 1000),
            int((config.request_connect_timeout or 0) * 1000),
        ),
    )
    with _CACHE_LOCK:
        client = _CLIENT_CACHE.get(cache_key)
        if client is None:
            http_options = _build_http_options(config, api_version=version)
            client = _GenAIClient(api_key=api_key, http_options=http_options)
            _CLIENT_CACHE[cache_key] = client
    return client


def get_text_client(config: Config) -> _GenAIClient:
    """Return a Gemini client configured for text models (v1)."""

    return _get_client(config, api_version="v1")


def get_media_client(config: Config) -> _GenAIClient:
    """Return a Gemini client configured for media models (v1beta)."""

    return _get_client(config, api_version="v1beta")


def get_gemini_client(config: Config, *, api_version: str | None = None) -> _GenAIClient:
    """Backward compatible wrapper for legacy call sites."""

    version = (api_version or "v1").strip().lower()
    if version in {"v1", "text"}:
        return get_text_client(config)
    if version in {"v1beta", "media"}:
        return get_media_client(config)
    raise ValueError(f"Unsupported Gemini API version: {api_version}")


__all__ = ["get_gemini_client", "get_media_client", "get_text_client"]
