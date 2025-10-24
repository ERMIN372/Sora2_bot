"""Helpers for querying Gemini model catalog."""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Iterable, List, Optional

from google.genai import errors as _genai_errors

from config import Config
from services.gemini_client import get_media_client, get_text_client

log = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 300.0
_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, List[str]]] = {}
_SUPPORTED_PREFIXES = ("veo-3.0-", "veo-3.1-")


def _is_supported_model(name: str) -> bool:
    value = (name or "").strip().lower()
    return any(value.startswith(prefix) for prefix in _SUPPORTED_PREFIXES)


def _extract_model_name(entry: object) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):  # type: ignore[redundant-expr]
        for key in ("name", "model", "id"):
            value = entry.get(key)  # type: ignore[union-attr]
            if isinstance(value, str) and value:
                return value
        return ""
    for attribute in ("name", "model", "id"):
        value = getattr(entry, attribute, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _iter_models(response: object) -> Iterable[object]:
    if response is None:
        return ()
    if isinstance(response, dict):
        models = response.get("models")
        if isinstance(models, (list, tuple)):
            return models
    if hasattr(response, "models"):
        models_obj = getattr(response, "models")
        if isinstance(models_obj, (list, tuple)):
            return models_obj
    if isinstance(response, (list, tuple)):
        return response
    return (response,)


def _iter_response(obj: object) -> Iterable[object]:
    if obj is None:
        return ()
    if hasattr(obj, "pages"):
        try:
            pages = getattr(obj, "pages")
            for page in pages:
                for entry in _iter_models(page):
                    yield entry
            return
        except Exception:  # pragma: no cover - defensive against SDK quirks
            log.debug("Failed to iterate Gemini model pages", exc_info=True)
    if isinstance(obj, (list, tuple)):
        yield from obj
        return
    if isinstance(obj, dict):
        yield from _iter_models(obj)
        return
    if isinstance(obj, str):
        yield obj
        return
    try:
        iterator = iter(obj)  # type: ignore[arg-type]
    except TypeError:
        yield from _iter_models(obj)
        return
    for entry in iterator:
        yield entry


def _normalise_model_name(name: str) -> str:
    cleaned = (name or "").strip()
    if "/" in cleaned:
        cleaned = cleaned.rsplit("/", 1)[-1]
    return cleaned


def _normalise_method_name(method: object) -> str:
    if not isinstance(method, str):
        method = str(method or "")
    text = method.strip()
    if not text:
        return ""
    if text.startswith("generate") and any(char.isupper() for char in text[8:]):
        text = re.sub(r"([A-Z])", r"_\1", text).lower()
    return text.replace("-", "_").strip().lower()


def _pick_client(config: Config, capability_key: str):
    if capability_key in {"generate_videos", "generate_images"}:
        return get_media_client(config)
    return get_text_client(config)


def list_models_with_capability(
    config: Config,
    *,
    capability: str,
) -> List[str]:
    """Return model names that advertise *capability* in the catalog."""

    capability_key = _normalise_method_name(capability)
    if not capability_key:
        return []

    try:
        client = _pick_client(config, capability_key)
    except Exception:
        log.warning("Gemini client initialisation failed for capability=%s", capability_key, exc_info=True)
        return []

    responses: List[object] = []
    page_token: Optional[str] = None
    while True:
        try:
            if page_token:
                response = client.models.list(page_token=page_token)
            else:
                response = client.models.list()
        except _genai_errors.APIError as exc:  # pragma: no cover - network guard
            log.warning("Gemini models.list failed capability=%s: %s", capability_key, exc, exc_info=True)
            return []
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Unexpected error listing Gemini models capability=%s", capability_key, exc_info=True)
            return []
        responses.append(response)
        next_token = None
        for attr in ("next_page_token", "nextPageToken"):
            candidate = getattr(response, attr, None)
            if isinstance(candidate, str) and candidate:
                next_token = candidate
                break
            if isinstance(response, dict):
                candidate = response.get(attr)
                if isinstance(candidate, str) and candidate:
                    next_token = candidate
                    break
        if not next_token:
            break
        if next_token == page_token:
            break
        page_token = next_token

    names: List[str] = []
    seen: set[str] = set()

    def _register(name: str) -> None:
        if not name:
            return
        if name not in seen:
            names.append(name)
            seen.add(name)

    for response in responses:
        for entry in _iter_response(response):
            supported = getattr(entry, "supported_generation_methods", None)
            if supported is None and isinstance(entry, dict):
                supported = entry.get("supported_generation_methods")
        method_names = {
            _normalise_method_name(value)
            for value in (supported or [])
        }
        if capability_key not in method_names:
            continue
        raw_name = _extract_model_name(entry)
        if not raw_name:
            continue
        short = _normalise_model_name(raw_name)
        if short:
            _register(short)
        _register(raw_name)

    return names


def list_veo_video_models(config: Config) -> List[str]:
    """Return available Veo 3.x model identifiers from Developer API."""

    if not config.gemini_video_enabled:
        return []

    cache_key = (config.gemini_api_key or "").strip()
    now = time.time()
    with _CACHE_LOCK:
        cached = _CACHE.get(cache_key)
        if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
            return list(cached[1])

    names = list_models_with_capability(
        config,
        capability="generateVideos",
    )
    filtered: List[str] = []
    for name in names:
        normalised = _normalise_model_name(name)
        if _is_supported_model(normalised) and normalised not in filtered:
            filtered.append(normalised)

    unique_sorted = sorted(filtered)
    with _CACHE_LOCK:
        _CACHE[cache_key] = (now, unique_sorted)
    return list(unique_sorted)


__all__ = ["list_models_with_capability", "list_veo_video_models"]
