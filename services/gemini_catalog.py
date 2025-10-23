"""Helpers for querying Gemini model catalog."""
from __future__ import annotations

import logging
import threading
import time
from typing import Iterable, List

from google.genai import errors as _genai_errors

from config import Config
from services.gemini_client import get_gemini_client

log = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 300.0
_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, List[str]]] = {}


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


def _normalise_model_name(name: str) -> str:
    cleaned = (name or "").strip()
    if "/" in cleaned:
        cleaned = cleaned.rsplit("/", 1)[-1]
    return cleaned


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

    client = get_gemini_client(config)
    try:
        response = client.models.list()
    except _genai_errors.APIError as exc:  # pragma: no cover - network guard
        log.warning("Failed to list Gemini models: %s", exc, exc_info=True)
        return []
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Unexpected error listing Gemini models", exc_info=True)
        return []

    def _iter_response(obj: object) -> Iterable[object]:
        if obj is None:
            return ()
        # The google-genai SDK returns a pager object that is already iterable.
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

    names: List[str] = []
    for item in _iter_response(response):
        name = _extract_model_name(item)
        if not name:
            continue
        normalised = _normalise_model_name(name)
        if normalised.startswith("veo-3"):
            names.append(normalised)

    unique_sorted = sorted(dict.fromkeys(names))
    with _CACHE_LOCK:
        _CACHE[cache_key] = (now, unique_sorted)
    return list(unique_sorted)


__all__ = ["list_veo_video_models"]
