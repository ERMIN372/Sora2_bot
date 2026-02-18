"""Shared Gemini client factory."""
from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Tuple

from google import genai  # type: ignore[import-untyped]
from google.genai import types

from config import Config, RoutingCfg
from services.gemini_key import ensure_gemini_key_logged

log = logging.getLogger(__name__)

_CLIENT_CACHE: Dict[Tuple[str, str, str, str, Tuple[int, int, int]], genai.Client] = {}
_CACHE_LOCK = threading.Lock()


def _build_http_options(config: Config, *, api_version: str) -> types.HttpOptions:
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
    return types.HttpOptions(timeout=timeout_ms, api_version=version)


def _get_client(config: Config, *, api_version: str) -> genai.Client:
    """Return a cached :class:`google.genai.Client` instance for *api_version*."""

    api_key = (config.gemini_api_key or "").strip()
    mode = (config.gemini_api_mode or "developer").strip().lower()
    if mode != "vertex" and not api_key:
        raise RuntimeError("Gemini API key is required to use Gemini services")
    ensure_gemini_key_logged(
        config,
        context="gemini_client",
        process_id=f"pid={os.getpid()}",
        logger=log,
    )
    version = api_version.strip() or "v1"
    vertex_project = (config.vertex_project_id or "").strip()
    vertex_location = (config.vertex_location or "").strip()
    cache_key = (
        api_key if mode != "vertex" else "",
        version,
        mode,
        f"{vertex_project}:{vertex_location}",
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
            if mode == "vertex":
                if not vertex_project or not vertex_location:
                    raise RuntimeError(
                        "VERTEX_PROJECT_ID and VERTEX_LOCATION are required when GEMINI_API_MODE=vertex"
                    )
                client = genai.Client(
                    vertexai=True,
                    project=vertex_project,
                    location=vertex_location,
                    http_options=http_options,
                )
            else:
                client = genai.Client(api_key=api_key, http_options=http_options)
            _CLIENT_CACHE[cache_key] = client
    return client


def get_text_client(config: Config) -> genai.Client:
    """Return a Gemini client configured for text models (v1)."""

    return _get_client(config, api_version=RoutingCfg.TEXT_API_VERSION)


def get_media_client(config: Config) -> genai.Client:
    """Return a Gemini client configured for media models (v1beta)."""

    return _get_client(config, api_version=RoutingCfg.MEDIA_API_VERSION)


def get_gemini_client(config: Config, *, api_version: str | None = None) -> genai.Client:
    """Backward compatible wrapper for legacy call sites."""

    version = (api_version or "v1").strip().lower()
    if version in {"v1", "text"}:
        return get_text_client(config)
    if version in {"v1beta", "media"}:
        return get_media_client(config)
    raise ValueError(f"Unsupported Gemini API version: {api_version}")


__all__ = ["get_gemini_client", "get_media_client", "get_text_client"]
