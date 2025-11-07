"""Clients for Google Gemini generative models."""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import os
import random
import re
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from uuid import uuid4

from google import genai  # type: ignore[import-untyped]
from google.genai import errors as genai_errors, types

if not hasattr(types, "GenerateContentConfig"):
    types.GenerateContentConfig = types.GenerateImagesConfig  # type: ignore[attr-defined]

from config import (
    Config,
    DEBUG_GEMINI,
    GEMINI_PREDICT_DISABLE_TTL_SEC,
    GEMINI_TRACE,
    GEMINI_TRACE_CURL,
    GEMINI_TRACE_HEADERS,
    GEMINI_TRACE_SAVE_B64,
    GEMINI_TRACE_SAVE_JSON,
    SafetyCfg,
)
from services.gemini_key import ensure_gemini_key_logged
from services.gemini_client import get_media_client, get_text_client
from services.gemini_router import (
    GeminiRoutingError,
    RouteDecision,
    get_gemini_router,
)
from services.logging_safety import extract_block_info
from services.payload_sanitize import log_removed_keys, sanitize_payload
from services.payload_whitelists import GEMINI_IMAGE_ALLOWED, GEMINI_TEXT_ALLOWED
from services.prompt_filters import (
    attach_negative_prompt,
    default_rephraser,
    scrub_brands_and_persons,
)
from services.retry_policy import RetryDecision, next_attempt
from services.rate_limiter import AsyncRateLimiter
from observability import increment_metric, record_timing_metric

from .api_error_handler import ApiErrorHandler
from .base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
    _sanitize_headers,
)
from .logx import get_logger, kv, mask

log = get_logger("providers.gemini")


_TRACE_DIR = Path("logs") / "gemini"
_TRACE_CASES_DIR = _TRACE_DIR / "cases"
_TRACE_INLINE_DIR = _TRACE_DIR / "inline"

_TRACE_HEADERS_ALL = any(header == "*" for header in GEMINI_TRACE_HEADERS)
_TRACE_HEADER_WHITELIST = {
    header.strip().lower() for header in GEMINI_TRACE_HEADERS if header and header != "*"
}


_MODEL_CAPABILITY_TTL = 3600.0
_MODEL_CAPABILITIES: Dict[str, Dict[str, Any]] = {}


_TRACE_HISTORY: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)


_IMAGE_RETRYABLE_REASONS = {"IMAGE_OTHER", "NO_IMAGE", "STOP"}
_IMAGE_BLOCK_REASONS = {"IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT"}
_IMAGE_RETRY_DELAYS = (1.5, 3.0)


class GeminiImageEmptyError(ProviderAPIError):
    """Raised when Gemini returns a response without inline image data."""

    def __init__(
        self,
        *,
        provider: str,
        finish_reason: Optional[str],
        response_id: Optional[str],
        headers: Optional[Mapping[str, Any]],
        prompt_feedback: Optional[Mapping[str, Any]] = None,
        safety_ratings: Optional[Any] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        message = "Gemini не вернул изображение"
        super().__init__(
            provider=provider,
            status_code=200,
            message=message,
            error_type="gemini_empty_image",
            provider_message=message,
            retryable=False,
            duration_ms=duration_ms,
        )
        self.finish_reason = finish_reason
        self.response_id = response_id
        self.headers = dict(headers or {})
        self.prompt_feedback = prompt_feedback
        self.safety_ratings = safety_ratings



def _has_generate_content_flag(model_meta: dict) -> bool:
    """Return True if the model metadata advertises generateContent support."""

    # The public Google Generative Language API changed the casing of the
    # metadata keys a couple of times (``supportedGenerationMethods`` in v1,
    # ``supported_generation_methods`` in beta) and occasionally nests the
    # information under ``capabilities``.  Normalise everything into a single
    # iterable before checking for the ``generateContent`` flag so the
    # healthcheck does not start failing when the schema shifts again.
    candidate_keys = (
        "supported_generation_methods",
        "supportedGenerationMethods",
        "generation_methods",
        "generationMethods",
        "capabilities",
    )

    methods = []
    for key in candidate_keys:
        value = model_meta.get(key)
        if not value:
            continue
        if isinstance(value, dict):
            methods.extend(value.keys())
        else:
            methods.extend(value if isinstance(value, (list, tuple, set, frozenset)) else [value])

    normalised = {str(item).lower().replace("_", "").replace("-", "") for item in methods}
    return "generatecontent" in normalised


_SAFETY_CATEGORIES: Tuple[types.HarmCategory, ...] = (
    types.HarmCategory.HARM_CATEGORY_HARASSMENT,
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
)

_SAFETY_DEFAULT_THRESHOLD = types.HarmBlockThreshold.BLOCK_NONE
_SAFETY_FALLBACK_THRESHOLD = types.HarmBlockThreshold.BLOCK_ONLY_HIGH


def _enum_code(value: Any) -> str:
    """Return a stable string representation for Gemini enums."""

    if value is None:
        return ""
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def _default_safety_settings() -> List[types.SafetySetting]:
    """Return a permissive Gemini safety configuration for text calls."""

    return [
        types.SafetySetting(
            category=category, threshold=_SAFETY_DEFAULT_THRESHOLD
        )
        for category in _SAFETY_CATEGORIES
    ]


def _normalise_inline_blob(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert inline or image payload to a normalised metadata structure."""

    if not isinstance(data, dict):
        return None

    mime = (
        data.get("mime_type")
        or data.get("mimeType")
        or data.get("mime")
        or data.get("content_type")
        or "application/octet-stream"
    )

    raw = (
        data.get("data")
        or data.get("image_bytes")
        or data.get("imageBytes")
        or data.get("inlineData")
        or data.get("inline_data")
        or data.get("base64")
    )

    binary: Optional[bytes] = None
    if isinstance(raw, (bytes, bytearray)):
        binary = bytes(raw)
    elif isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("data:"):
            header, _, payload = stripped.partition(",")
            if header.startswith("data:"):
                mime = header[5:].split(";", 1)[0] or mime
            if payload:
                try:
                    binary = base64.b64decode(payload, validate=True)
                except (binascii.Error, ValueError):
                    binary = payload.encode("utf-8")
        else:
            try:
                binary = base64.b64decode(stripped, validate=True)
            except (binascii.Error, ValueError):
                binary = stripped.encode("utf-8")

    if binary is None:
        extra = data.get("data_bytes") or data.get("bytes")
        if isinstance(extra, (bytes, bytearray)):
            binary = bytes(extra)

    if binary is None:
        return None

    encoded = base64.b64encode(binary).decode("ascii")
    data_url = f"data:{mime};base64,{encoded}"
    return {
        "inline": True,
        "kind": data.get("kind") or "inline",
        "mime": mime,
        "bytes": len(binary),
        "base64": encoded,
        "data_url": data_url,
        "data_uri": data_url,
    }


def _serialise_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _normalise_setting_key(name: str) -> str:
    if not name:
        return ""
    normalised = re.sub(r"[\s-]+", "_", name)
    normalised = re.sub(r"(?<!^)(?=[A-Z])", "_", normalised)
    normalised = normalised.lower()
    normalised = normalised.replace("_j_s_o_n_", "_json_")
    normalised = normalised.replace("_u_r_i_", "_uri_")
    normalised = re.sub(r"__+", "_", normalised)
    return normalised.strip("_")


def _ensure_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return value
    return (value,)


def _extract_prompt_feedback(payload: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return None
    feedback = payload.get("prompt_feedback") or payload.get("promptFeedback")
    return feedback if isinstance(feedback, Mapping) else None


def _extract_safety_ratings(feedback: Optional[Mapping[str, Any]]) -> Optional[Any]:
    if not isinstance(feedback, Mapping):
        return None
    ratings = feedback.get("safety_ratings") or feedback.get("safetyRatings")
    return ratings


def _extract_response_headers(response: Any) -> Dict[str, str]:
    sdk = getattr(response, "sdk_http_response", None)
    headers = getattr(sdk, "headers", None)
    if not headers:
        return {}
    if hasattr(headers, "items"):
        try:
            raw = {str(k).lower(): str(v) for k, v in headers.items()}
        except Exception:  # pragma: no cover - defensive
            raw = {}
    else:
        try:
            raw = {str(k).lower(): str(headers[k]) for k in headers}
        except Exception:  # pragma: no cover - defensive
            raw = {}
    if _TRACE_HEADERS_ALL:
        filtered = raw
    else:
        filtered = {key: raw[key] for key in raw if key in _TRACE_HEADER_WHITELIST}
    return _sanitize_headers(filtered)


def _ensure_trace_dir(path: Path) -> Optional[Path]:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - filesystem guard
        log.warning("Failed to initialise trace directory path=%s", path, exc_info=True)
        return None
    return path


def _trace_key(prefix: str, corr_id: Optional[str]) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    safe_corr = (corr_id or "no_corr").replace(os.sep, "_")[:64]
    return f"{timestamp}_{prefix}_{safe_corr}"


def _coerce_json(value: Any, depth: int = 0) -> Any:
    if depth > 5:  # pragma: no cover - defensive recursion guard
        return str(value)
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, Mapping):
        return {str(k): _coerce_json(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_coerce_json(item, depth + 1) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")  # type: ignore[call-arg]
        except TypeError:
            return dump()  # type: ignore[misc]
        except Exception:  # pragma: no cover - defensive serialisation
            return str(value)
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        try:
            return dict_method()  # type: ignore[call-arg]
        except Exception:  # pragma: no cover - defensive serialisation
            return str(value)
    return str(value)


def _register_trace_event(corr_id: Optional[str], payload: Dict[str, Any]) -> None:
    if not corr_id:
        corr_id = "no_corr"
    history = _TRACE_HISTORY[corr_id]
    history.append(payload)
    if len(history) > 20:
        del history[:-20]


def _trace_log(level: int, message: str, **kwargs: Any) -> None:
    # DEBUG: structured tracing helper
    if not GEMINI_TRACE:
        return
    log.log(level, "gemini.trace %s %s", message, kv(**kwargs))


def _normalise_model_name(model: Optional[str]) -> str:
    return (model or "").strip().lower()


def _get_capability_entry(model: Optional[str]) -> Optional[Dict[str, Any]]:
    key = _normalise_model_name(model)
    if not key:
        return None
    entry = _MODEL_CAPABILITIES.get(key)
    if not entry:
        return None
    if time.time() - entry.get("last_checked", 0.0) > _MODEL_CAPABILITY_TTL:
        return None
    return entry


def get_predict_capability(model: Optional[str]) -> Optional[bool]:
    entry = _get_capability_entry(model)
    if not entry:
        return None
    return entry.get("predict_works")


def mark_predict_capability(model: Optional[str], works: bool) -> None:
    key = _normalise_model_name(model)
    if not key:
        return
    _MODEL_CAPABILITIES[key] = {
        "predict_works": bool(works),
        "last_checked": time.time(),
    }


def capability_snapshot() -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    now = time.time()
    for key, value in list(_MODEL_CAPABILITIES.items()):
        if now - value.get("last_checked", 0.0) > _MODEL_CAPABILITY_TTL:
            continue
        snapshot[key] = dict(value)
    return snapshot


def should_probe_predict(model: Optional[str]) -> bool:
    return get_predict_capability(model) is None


def _method_operation(method: str) -> str:
    if method == "generate_images":
        return "generateImages"
    if method == "generate_content":
        return "generateContent"
    return method


def _summarise_prompt_payload(
    request_kwargs: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    prompt_text = ""
    if isinstance(request_kwargs.get("prompt"), str):
        prompt_text = request_kwargs.get("prompt") or ""
    elif isinstance(request_kwargs.get("contents"), list):
        for item in request_kwargs["contents"]:
            if not isinstance(item, Mapping):
                continue
            parts = item.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    prompt_text += part["text"]
    elif isinstance(payload.get("contents"), list):
        for item in payload["contents"]:
            if not isinstance(item, Mapping):
                continue
            parts = item.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    prompt_text += part["text"]
    elif isinstance(payload.get("prompt"), str):
        prompt_text = payload.get("prompt", "") or ""

    assets = payload.get("assets")
    asset_count = len(assets) if isinstance(assets, list) else 0
    inline_refs = payload.get("reference_inline_data")
    inline_count = len(inline_refs) if isinstance(inline_refs, list) else 0
    return {
        "prompt_chars": len(prompt_text or ""),
        "asset_count": asset_count,
        "inline_refs": inline_count,
    }


def _trace_request(
    *,
    client: "GeminiGenerativeClient",
    decision: "RouteDecision",
    method: str,
    api_version: Optional[str],
    attempt: int,
    corr_id: Optional[str],
    request_kwargs: Mapping[str, Any],
    payload: Mapping[str, Any],
    force_mode: Optional[str],
    payload_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if not GEMINI_TRACE:
        return {}
    _ensure_trace_dir(_TRACE_DIR)
    summary = _summarise_prompt_payload(request_kwargs, payload)
    operation = _method_operation(method)
    version = (api_version or client._api_version or "").strip("/") or "v1beta"
    endpoint = f"{client._base_url}/{version}/models/{decision.model}:{operation}"
    event = {
        "ts": time.time(),
        "phase": "request",
        "model": decision.model,
        "method": method,
        "operation": operation,
        "api_version": version,
        "attempt": attempt,
        "corr_id": corr_id or "",
        "url": endpoint,
        "force_mode": force_mode or "",
        "saved_request": str(payload_path) if payload_path else "",
    }
    event.update(summary)
    _trace_log(
        logging.INFO,
        "request",
        model=decision.model,
        method=method,
        version=version,
        attempt=attempt,
        corr_id=corr_id or "",
        url=endpoint,
        prompt_chars=summary["prompt_chars"],
        asset_count=summary["asset_count"],
        inline_refs=summary["inline_refs"],
        force_mode=force_mode or "",
        saved_request=str(payload_path) if payload_path else "",
    )
    _register_trace_event(corr_id, event)
    return event


def _trace_save_request_payload(
    *,
    corr_id: Optional[str],
    method: str,
    payload: Mapping[str, Any],
) -> Optional[Path]:
    if not GEMINI_TRACE_CURL:
        return None
    trace_dir = _ensure_trace_dir(_TRACE_DIR)
    if trace_dir is None:
        return None
    key = _trace_key(f"request_{method}", corr_id)
    path = trace_dir / f"{key}.payload.json"
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_coerce_json(payload), handle, ensure_ascii=False, indent=2)
    except Exception:  # pragma: no cover - filesystem guard
        log.warning("Failed to persist Gemini request payload path=%s", path, exc_info=True)
        return None
    return path


def _trace_log_curl(
    *,
    client: "GeminiGenerativeClient",
    decision: "RouteDecision",
    method: str,
    api_version: Optional[str],
    payload_path: Optional[Path],
    corr_id: Optional[str],
) -> None:
    if not GEMINI_TRACE_CURL or payload_path is None:
        return
    operation = _method_operation(method)
    version = (api_version or client._api_version or "").strip("/") or "v1beta"
    endpoint = f"{client._base_url}/{version}/models/{decision.model}:{operation}"
    curl = (
        f"curl -X POST '{endpoint}' -H 'Content-Type: application/json' "
        f"-H 'x-goog-api-key:***' --data @{payload_path.name}"
    )
    _trace_log(
        logging.INFO,
        "curl",
        corr_id=corr_id or "",
        model=decision.model,
        method=method,
        version=version,
        payload_file=str(payload_path),
        command=curl,
    )


def _collect_inline_blobs(payload_json: Mapping[str, Any]) -> List[Tuple[str, str]]:
    blobs: List[Tuple[str, str]] = []

    def _consume_inline(data: Mapping[str, Any]) -> None:
        inline = data.get("inline_data") or data.get("inlineData")
        if isinstance(inline, Mapping):
            data_field = inline.get("data") or inline.get("data_base64")
            if isinstance(data_field, str) and data_field.strip():
                mime = (
                    inline.get("mime_type")
                    or inline.get("mimeType")
                    or inline.get("mime")
                    or "image/png"
                )
                blobs.append((str(mime), data_field))

    candidates = payload_json.get("candidates") if isinstance(payload_json, Mapping) else None
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            content = candidate.get("content")
            if isinstance(content, Mapping):
                parts = content.get("parts")
            elif isinstance(content, list):
                parts = content
            else:
                parts = None
            if isinstance(parts, list):
                for part in parts:
                    if isinstance(part, Mapping):
                        _consume_inline(part)
    inline_direct = payload_json.get("inline_data") or payload_json.get("inlineData")
    if isinstance(inline_direct, Mapping):
        data_field = inline_direct.get("data") or inline_direct.get("data_base64")
        if isinstance(data_field, str) and data_field.strip():
            mime = (
                inline_direct.get("mime_type")
                or inline_direct.get("mimeType")
                or inline_direct.get("mime")
                or "image/png"
            )
            blobs.append((str(mime), data_field))
    return blobs


def _trace_save_inline_blobs(
    *,
    payload_json: Mapping[str, Any],
    response_id: Optional[str],
) -> List[Path]:
    if not GEMINI_TRACE_SAVE_B64:
        return []
    trace_dir = _ensure_trace_dir(_TRACE_INLINE_DIR)
    if trace_dir is None:
        return []
    saved: List[Path] = []
    for index, (mime, blob) in enumerate(_collect_inline_blobs(payload_json)):
        filename = _trace_key(response_id or "resp", f"inline{index}")
        extension = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
        }.get(mime.lower(), ".bin")
        path = trace_dir / f"{filename}{extension}"
        try:
            binary = base64.b64decode(blob, validate=False)
        except (binascii.Error, ValueError):  # pragma: no cover - defensive
            binary = blob.encode("utf-8")
        try:
            with path.open("wb") as handle:
                handle.write(binary)
        except Exception:  # pragma: no cover - filesystem guard
            log.warning("Failed to persist inline blob path=%s", path, exc_info=True)
            continue
        saved.append(path)
    return saved


def _trace_save_response_json(
    *,
    payload_json: Mapping[str, Any],
    corr_id: Optional[str],
    method: str,
    version: Optional[str],
    status: int,
) -> Optional[Path]:
    if not GEMINI_TRACE_SAVE_JSON:
        return None
    trace_dir = _ensure_trace_dir(_TRACE_DIR)
    if trace_dir is None:
        return None
    key = _trace_key(f"response_{method}_{status}", corr_id)
    path = trace_dir / f"{key}.json"
    try:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_coerce_json(payload_json), handle, ensure_ascii=False, indent=2)
    except Exception:  # pragma: no cover - filesystem guard
        log.warning("Failed to persist Gemini response payload path=%s", path, exc_info=True)
        return None
    return path


def _trace_response(
    *,
    client: "GeminiGenerativeClient",
    decision: "RouteDecision",
    method: str,
    api_version: Optional[str],
    corr_id: Optional[str],
    status: int,
    headers: Mapping[str, Any],
    duration_ms: int,
    payload_json: Mapping[str, Any],
    request_event: Mapping[str, Any],
    response_id: Optional[str],
) -> None:
    if not GEMINI_TRACE:
        return
    version = (api_version or client._api_version or "").strip("/") or "v1beta"
    inline_paths = _trace_save_inline_blobs(
        payload_json=payload_json,
        response_id=response_id,
    )
    response_path = _trace_save_response_json(
        payload_json=payload_json,
        corr_id=corr_id,
        method=method,
        version=version,
        status=status,
    )
    event = {
        "ts": time.time(),
        "phase": "response",
        "model": decision.model,
        "method": method,
        "api_version": version,
        "status": status,
        "corr_id": corr_id or "",
        "duration_ms": duration_ms,
        "response_id": response_id or "",
        "headers": dict(headers),
        "saved_json": str(response_path) if response_path else "",
        "saved_inline": [str(path) for path in inline_paths],
    }
    event.update({
        "prompt_chars": request_event.get("prompt_chars"),
        "asset_count": request_event.get("asset_count"),
        "inline_refs": request_event.get("inline_refs"),
    })
    _trace_log(
        logging.INFO,
        "response",
        model=decision.model,
        method=method,
        version=version,
        status=status,
        corr_id=corr_id or "",
        duration_ms=duration_ms,
        response_id=response_id or "",
        headers=headers,
        saved_json=str(response_path) if response_path else "",
        saved_inline=[str(path) for path in inline_paths],
    )
    _register_trace_event(corr_id, event)


def dump_gemini_case(corr_id: str) -> Optional[Path]:
    """Bundle the trace artefacts for *corr_id* into a ZIP archive."""

    if not corr_id:
        return None
    case_dir = _ensure_trace_dir(_TRACE_CASES_DIR)
    if case_dir is None:
        return None
    events = list(_TRACE_HISTORY.get(corr_id, []))
    if not events:
        return None
    archive_name = _trace_key("case", corr_id) + ".zip"
    archive_path = case_dir / archive_name
    try:
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            summary = {
                "corr_id": corr_id,
                "events": events,
            }
            archive.writestr(
                "summary.json",
                json.dumps(summary, ensure_ascii=False, indent=2),
            )
            for event in events:
                saved_request = event.get("saved_request")
                if saved_request:
                    path = Path(saved_request)
                    if path.exists():
                        archive.write(path, arcname=path.name)
                saved_json = event.get("saved_json")
                if saved_json:
                    path = Path(saved_json)
                    if path.exists():
                        archive.write(path, arcname=path.name)
                for inline_path in event.get("saved_inline", []):
                    path = Path(inline_path)
                    if path.exists():
                        archive.write(path, arcname=path.name)
    except Exception:  # pragma: no cover - filesystem guard
        log.warning("Failed to create Gemini debug archive corr_id=%s", corr_id, exc_info=True)
        return None
    return archive_path


def _prepare_images_assets(
    response: Any,
    *,
    provider_name: str,
    duration_ms: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, Dict[str, Any]]]:
    images = getattr(response, "images", None) or []
    inline_assets: List[Dict[str, Any]] = []
    asset_map: Dict[str, str] = {}
    asset_meta: Dict[str, Dict[str, Any]] = {}
    for index, img in enumerate(images):
        data = getattr(img, "data", None)
        mime = getattr(img, "mime_type", None)
        if data is None:
            data = getattr(img, "image_bytes", None)
        if data is None and hasattr(img, "image"):
            nested = getattr(img, "image")
            data = getattr(nested, "data", None)
            if data is None:
                data = getattr(nested, "image_bytes", None)
            mime = mime or getattr(nested, "mime_type", None)
        if isinstance(data, str):
            try:
                data_bytes = base64.b64decode(data)
            except Exception:
                data_bytes = data.encode("utf-8")
        else:
            data_bytes = bytes(data or b"")
        if not data_bytes:
            continue
        encoded = base64.b64encode(data_bytes).decode("ascii")
        mime = (mime or getattr(img, "mime_type", None) or "image/png") or "image/png"
        data_uri = f"data:{mime};base64,{encoded}"
        key = f"inline_{index}"
        inline_assets.append(
            {
                "key": key,
                "kind": "image",
                "mime": mime,
                "bytes": len(data_bytes),
                "data_uri": data_uri,
                "data_url": data_uri,
            }
        )
        asset_map[key] = data_uri
        asset_meta[key] = {
            "inline": True,
            "kind": "image",
            "mime": mime,
            "bytes": len(data_bytes),
        }
    if inline_assets:
        return inline_assets, asset_map, asset_meta
    raise GeminiImageEmptyError(
        provider=provider_name,
        finish_reason=None,
        response_id=getattr(response, "response_id", None),
        headers=_extract_response_headers(response),
        duration_ms=duration_ms,
    )


def _extract_images_from_gemini(
    response: Any,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    candidates = getattr(response, "candidates", None)
    finish_reason: Optional[str] = None
    response_id = getattr(response, "response_id", None)
    if candidates:
        primary = candidates[0]
        finish_reason = getattr(primary, "finish_reason", None)
        content_obj = getattr(primary, "content", None)
        parts = getattr(content_obj, "parts", None)
    else:
        parts = None
    if not parts and isinstance(response, Mapping):
        candidates_json = response.get("candidates")
        if isinstance(candidates_json, list) and candidates_json:
            first = candidates_json[0]
            if isinstance(first, Mapping):
                finish_reason = first.get("finishReason") or first.get("finish_reason")
                content_json = first.get("content")
                if isinstance(content_json, Mapping):
                    parts = content_json.get("parts")
    images: List[Dict[str, Any]] = []
    if isinstance(parts, list):
        for part in parts:
            inline = None
            if hasattr(part, "inline_data"):
                inline = getattr(part, "inline_data")
            elif isinstance(part, Mapping):
                inline = part.get("inline_data") or part.get("inlineData")
            if inline is None:
                continue
            data = getattr(inline, "data", None)
            if data is None and isinstance(inline, Mapping):
                data = inline.get("data")
            if data in (None, "", b""):
                continue
            if isinstance(data, (bytes, bytearray)):
                encoded = base64.b64encode(bytes(data)).decode("ascii")
            elif isinstance(data, str):
                stripped = data.strip()
                if stripped.startswith("data:") and "," in stripped:
                    _, _, payload = stripped.partition(",")
                    encoded = payload
                else:
                    encoded = stripped
            else:
                try:
                    encoded = base64.b64encode(bytes(data)).decode("ascii")
                except Exception:
                    continue
            mime = (
                getattr(inline, "mime_type", None)
                or getattr(inline, "mimeType", None)
                if not isinstance(inline, Mapping)
                else inline.get("mime_type")
                or inline.get("mimeType")
                or getattr(inline, "mime", None)
            )
            images.append({"data_b64": encoded, "mime": mime or "image/png"})
    if not finish_reason and candidates:
        primary = candidates[0]
        if isinstance(primary, Mapping):
            finish_reason = primary.get("finishReason") or primary.get("finish_reason")
    return images, finish_reason, response_id


def _prepare_flash_image_assets(
    response: Any,
    payload_json: Mapping[str, Any],
    *,
    provider_name: str,
    duration_ms: int,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, str],
    Dict[str, Dict[str, Any]],
    Optional[str],
    Optional[str],
]:
    images, finish_reason, response_id = _extract_images_from_gemini(response)
    inline_assets: List[Dict[str, Any]] = []
    asset_map: Dict[str, str] = {}
    asset_meta: Dict[str, Dict[str, Any]] = {}
    for index, image in enumerate(images):
        raw_b64 = image.get("data_b64")
        if not raw_b64:
            continue
        try:
            binary = base64.b64decode(str(raw_b64), validate=False)
        except (binascii.Error, ValueError):
            binary = str(raw_b64).encode("utf-8")
        encoded = base64.b64encode(binary).decode("ascii")
        mime = image.get("mime") or image.get("mime_type") or "image/png"
        data_uri = f"data:{mime};base64,{encoded}"
        key = f"inline_{index}"
        entry = {
            "key": key,
            "kind": "image",
            "mime": mime,
            "bytes": len(binary),
            "data_uri": data_uri,
            "data_url": data_uri,
        }
        if GEMINI_TRACE_SAVE_B64:
            trace_dir = _ensure_trace_dir(_TRACE_INLINE_DIR)
            if trace_dir is not None:
                filename = _trace_key(response_id or "resp", f"inline{index}")
                extension = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                }.get(str(mime).lower(), ".bin")
                trace_path = trace_dir / f"{filename}{extension}"
                try:
                    with trace_path.open("wb") as handle:
                        handle.write(binary)
                except Exception:  # pragma: no cover - filesystem guard
                    log.warning(
                        "Failed to persist inline asset trace path=%s", trace_path, exc_info=True
                    )
                else:
                    entry.setdefault("trace_path", str(trace_path))
        inline_assets.append(entry)
        asset_map[key] = data_uri
        asset_meta[key] = {
            "inline": True,
            "kind": "image",
            "mime": mime,
            "bytes": len(binary),
        }
    if inline_assets:
        return inline_assets, asset_map, asset_meta, finish_reason, response_id
    prompt_feedback = _extract_prompt_feedback(payload_json)
    safety_ratings = _extract_safety_ratings(prompt_feedback)
    headers = _extract_response_headers(response)
    raise GeminiImageEmptyError(
        provider=provider_name,
        finish_reason=finish_reason,
        response_id=getattr(response, "response_id", None),
        headers=headers,
        prompt_feedback=prompt_feedback,
        safety_ratings=safety_ratings,
        duration_ms=duration_ms,
    )


def _analyse_empty_image_response(
    payload_json: Mapping[str, Any],
    *,
    response: Any,
    error: GeminiImageEmptyError,
) -> Tuple[Optional[str], Optional[str], bool]:
    headers = {}
    if hasattr(error, "headers") and isinstance(error.headers, Mapping):
        headers = {str(k): str(error.headers[k]) for k in error.headers}
    else:
        headers = _extract_response_headers(response)
    header_lookup = {str(key).lower(): str(value) for key, value in headers.items()}
    finish_code = (error.finish_reason or "").upper() or None
    header_candidates = (
        "x-generative-ai-finish-reason",
        "x-generative-ai-output-status",
        "x-goog-rai-filtered-reason",
        "x-goog-image-response-status",
        "x-goog-ai-response-code",
    )
    for name in header_candidates:
        header_value = header_lookup.get(name)
        if header_value:
            finish_code = str(header_value).upper()
            break
    provider_reasons: List[str] = []
    candidate: Optional[Mapping[str, Any]] = None
    if isinstance(payload_json, Mapping):
        for key in ("status", "finish_reason", "finishReason"):
            value = payload_json.get(key)
            if value and not finish_code:
                finish_code = str(value).upper()
        candidates_payload = payload_json.get("candidates")
        if isinstance(candidates_payload, list) and candidates_payload:
            first_candidate = candidates_payload[0]
            if isinstance(first_candidate, Mapping):
                candidate = first_candidate
        generated_images = payload_json.get("generated_images") or payload_json.get("generatedImages")
        if isinstance(generated_images, list):
            for item in generated_images:
                if not isinstance(item, Mapping):
                    continue
                reason = item.get("rai_filtered_reason") or item.get("raiFilteredReason")
                if reason:
                    text = str(reason)
                    provider_reasons.append(text)
                    if not finish_code:
                        finish_code = text.upper()
    merged_reason = None
    if provider_reasons:
        merged_reason = "; ".join(dict.fromkeys(provider_reasons))
    reason_upper = (finish_code or "").upper()
    safety_trigger = False
    if reason_upper in _IMAGE_BLOCK_REASONS:
        safety_trigger = True
    elif provider_reasons:
        for reason in provider_reasons:
            lowered = reason.lower()
            if "safety" in lowered or "prohibit" in lowered or "policy" in lowered:
                safety_trigger = True
                break
    if reason_upper == "STOP":
        safety_trigger = False

    candidate_dump: Optional[Dict[str, Any]] = None
    if isinstance(candidate, Mapping):
        candidate_dump = _coerce_json(candidate)
        if isinstance(candidate_dump, dict):
            content = candidate_dump.get("content")
            parts = None
            if isinstance(content, dict):
                parts = content.get("parts")
            elif isinstance(content, list):
                parts = content
            if isinstance(parts, list):
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    inline_data = part.get("inline_data") or part.get("inlineData")
                    if isinstance(inline_data, dict):
                        data_value = inline_data.get("data") or inline_data.get("data_base64")
                        if isinstance(data_value, str):
                            inline_data["data"] = f"<len={len(data_value)}>"
                            inline_data.pop("data_base64", None)

    prompt_feedback = payload_json.get("prompt_feedback") or payload_json.get("promptFeedback")
    safety_ratings = _extract_safety_ratings(prompt_feedback)
    assets_payload = payload_json.get("generated_images") or payload_json.get("generatedImages")
    assets_count = len(assets_payload) if isinstance(assets_payload, list) else 0
    block_reason = payload_json.get("block_reason") or payload_json.get("blockReason")
    context = {
        "finish": reason_upper or "",
        "response_id": error.response_id or getattr(response, "response_id", ""),
        "headers": header_lookup,
        "candidate": candidate_dump,
        "prompt_feedback": _coerce_json(prompt_feedback)
        if isinstance(prompt_feedback, Mapping)
        else None,
        "safety_ratings": _coerce_json(safety_ratings) if safety_ratings is not None else None,
        "assets_count": assets_count,
        "block_reason": block_reason,
    }
    setattr(error, "analysis_context", context)
    _trace_log(
        logging.WARNING,
        "empty_image.analysis",
        finish=reason_upper or "",
        response_id=context["response_id"],
        assets=assets_count,
        block_reason=block_reason or "",
    )
    if DEBUG_GEMINI:
        log.debug(
            "gemini.empty.analyse %s",
            kv(
                finish=finish_code,
                safety=safety_trigger,
                headers=header_lookup,
            ),
        )
    return finish_code, merged_reason, safety_trigger


def _convert_setting_list(value: Any, model_cls: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    converted: List[Any] = []
    for item in _ensure_iterable(value):
        if isinstance(item, model_cls):
            converted.append(item)
        elif isinstance(item, dict):
            try:
                converted.append(model_cls.model_validate(item))
            except Exception as exc:  # pragma: no cover - defensive parsing
                log.warning("Failed to parse %s: %s", model_cls.__name__, exc, exc_info=True)
        else:
            converted.append(item)
    return converted


def _maybe_model_validate(value: Any, model_cls: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, model_cls):
        return value
    if isinstance(value, dict):
        try:
            return model_cls.model_validate(value)
        except Exception as exc:  # pragma: no cover - defensive parsing
            log.warning("Failed to parse %s: %s", model_cls.__name__, exc, exc_info=True)
    return value


def _infer_aspect_ratio(size: Optional[str]) -> Optional[str]:
    if not size or "x" not in size.lower():
        return None
    try:
        width_str, height_str = size.lower().split("x", 1)
        width = int(width_str)
        height = int(height_str)
        if width <= 0 or height <= 0:
            return None
    except (ValueError, ZeroDivisionError):
        return None
    ratio = Fraction(width, height).limit_denominator(48)
    return f"{ratio.numerator}:{ratio.denominator}"


def _truncate_preview(text: Optional[str], limit: int = 160) -> Optional[str]:
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}…(+{len(cleaned) - limit} chars)"


class GeminiGenerativeClient(BaseProviderClient):
    """Base client for synchronous Gemini API calls."""

    # Ensure debug flag exists even for instances created via ``object.__new__``
    # in unit-tests that bypass ``__init__``.
    _debug_enabled: bool = False

    def __init__(
        self,
        *,
        config: Config,
        api_key: str,
        model: str,
        provider_name: str,
        task: str,
        api_version: str | None,
        safety_settings: Optional[List[types.SafetySetting]] = None,
    ) -> None:
        base_url = self._resolve_base_url(config)
        super().__init__(
            config=config,
            base_url=base_url,
            api_key=api_key,
            provider_name=provider_name,
        )
        self._debug_enabled = bool(getattr(config, "debug_gemini", False))
        if self._debug_enabled and log.getEffectiveLevel() > logging.DEBUG:
            log.setLevel(logging.DEBUG)
        self._task = task
        resolved_version = self._resolve_api_version(
            task=self._task,
            model=model,
            api_version=api_version,
        )
        http_opts = types.HttpOptions(api_version=resolved_version)

        self._client = genai.Client(
            api_key=api_key,
            http_options=http_opts,
        )

        self._api_key = api_key
        parsed_safety = _convert_setting_list(safety_settings, types.SafetySetting)
        self._safety_settings = parsed_safety or _default_safety_settings()
        self._model = model
        self._api_version = resolved_version
        self._router = get_gemini_router(config)
        self._available_models: List[str] = []
        self._available_model_ids: Set[str] = set()
        self._media_safety_threshold = self._parse_safety_threshold(
            config.gemini_safety_threshold
        )
        self._media_safety_settings = [
            types.SafetySetting(
                category=category, threshold=self._media_safety_threshold
            )
            for category in _SAFETY_CATEGORIES
        ]
        self._pending: Dict[str, Tuple[Dict[str, Any], int, int, Dict[str, Any]]] = {}
        self._environment = (config.environment or "dev").lower()
        self._key_mask = ensure_gemini_key_logged(
            config,
            context="gemini_generative_client",
            process_id=f"pid={os.getpid()}",  # pragma: no cover - runtime value
            logger=log,
        )
        self._predict_block_until: Optional[float] = None
        if DEBUG_GEMINI:
            log.info(
                "gemini.init %s",
                kv(
                    provider=self.provider_name,
                    model=self._model,
                    env=self._environment,
                ),
            )
        self._log_debug_event(
            "gemini.config",
            {
                "provider": provider_name,
                "task": task,
                "model": model,
                "api_version": self._api_version,
                "environment": self._environment,
                "api_mode": (config.gemini_api_mode or "developer"),
                "media_safety": _enum_code(self._media_safety_threshold),
                "text_safety": [
                    _enum_code(getattr(entry, "threshold", ""))
                    for entry in self._safety_settings
                ],
                "fallback_models": list(getattr(config, "fallback_image_models", []) or []),
                "key_mask": self._key_mask or mask(self._api_key),
            },
            level=logging.INFO if self._debug_enabled else logging.DEBUG,
        )
        self._pending_dir = Path("attached_assets") / "pending" / provider_name
        try:
            self._pending_dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - filesystem guard
            log.warning(
                "Failed to initialise pending cache directory dir=%s", self._pending_dir,
                exc_info=True,
            )
        self._load_available_models()
        self._error_handler = ApiErrorHandler(
            provider=provider_name,
            logger=log,
            supported_models=self._available_models,
        )
        limit_per_minute = max(1, config.gemini_requests_per_minute)
        self._rate_limiter = AsyncRateLimiter(limit_per_minute, 60.0)
        if self._available_model_ids and self._model.lower() not in self._available_model_ids:
            raise ProviderAPIError(
                provider=self.provider_name,
                status_code=404,
                message=f"Модель {self._model} недоступна для Gemini",
                error_type="invalid_model",
                provider_message="model_not_available",
            )

    def _predict_temporarily_blocked(self) -> bool:
        return bool(self._predict_block_until and self._predict_block_until > time.time())

    def _block_predict_now(self, ttl: int) -> None:
        self._predict_block_until = time.time() + max(60, int(ttl or 0))
        if DEBUG_GEMINI:
            log.warning("gemini.predict.disabled %s", kv(ttl_sec=ttl))

    def _save_no_image_diag(
        self,
        *,
        decision: RouteDecision,
        method: str,
        payload_json: Mapping[str, Any],
        headers: Mapping[str, Any],
        response: Any,
        finish: Optional[str],
        safety_feedback: Any,
        error: GeminiImageEmptyError,
        duration_ms: int,
        corr_id: str,
    ) -> Optional[Path]:
        try:
            diag_dir = Path("tmp") / "gemini_diag"
            diag_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - filesystem guard
            log.warning("gemini.diag.save_failed %s", kv(error=exc))
            return None
        diag_id = corr_id or getattr(response, "response_id", "") or ""
        safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", diag_id or uuid4().hex)
        path = diag_dir / f"{safe_id}.json"
        candidates_list: List[Any] = []
        if isinstance(payload_json, Mapping):
            raw_candidates = payload_json.get("candidates")
            if isinstance(raw_candidates, list):
                candidates_list = raw_candidates
        parts_total = 0
        had_inline = bool(
            isinstance(payload_json, Mapping)
            and (payload_json.get("inline_data") or payload_json.get("inlineData"))
        )
        for candidate in candidates_list:
            if not isinstance(candidate, Mapping):
                continue
            content_json = candidate.get("content")
            parts_json = None
            if isinstance(content_json, Mapping):
                parts_json = content_json.get("parts")
            elif isinstance(content_json, list):
                parts_json = content_json
            if isinstance(parts_json, list):
                parts_total += len(parts_json)
                if not had_inline:
                    for part in parts_json:
                        if not isinstance(part, Mapping):
                            continue
                        if part.get("inline_data") or part.get("inlineData"):
                            had_inline = True
                            break
        images_total = 0
        image_fields = ("images", "generated_images", "generatedImages")
        for key in image_fields:
            maybe_images = payload_json.get(key) if isinstance(payload_json, Mapping) else None
            if isinstance(maybe_images, list):
                images_total = len(maybe_images)
                if images_total:
                    break
        if not images_total:
            response_images = getattr(response, "images", None)
            if response_images:
                try:
                    images_total = len(response_images)
                except TypeError:
                    images_total = 1
        diag_payload = {
            "model": decision.model,
            "method": method,
            "api_version": decision.api_version,
            "finish": finish,
            "headers": dict(headers or {}),
            "prompt_preview": _truncate_preview(decision.prompt),
            "counts": {
                "candidates": len(candidates_list),
                "parts": parts_total,
                "images": images_total,
            },
            "had_inline_data": had_inline,
            "prompt_feedback": self._sanitize_debug_value(safety_feedback),
            "safety_ratings": self._sanitize_debug_value(error.safety_ratings),
            "response_id": getattr(response, "response_id", None),
            "latency_ms": duration_ms,
        }
        try:
            path.write_text(json.dumps(diag_payload, ensure_ascii=False, indent=2))
            size = path.stat().st_size
            log.info("gemini.diag.saved %s", kv(path=str(path), size=size))
            return path
        except Exception as exc:  # pragma: no cover - filesystem guard
            log.warning("gemini.diag.save_failed %s", kv(error=exc))
            return None

    def _sanitize_debug_value(self, value: Any, depth: int = 0) -> Any:
        if value is None:
            return None
        if isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            if value.startswith("data:") and ";base64," in value:
                return f"data-uri({len(value)})"
            if len(value) > 160:
                return f"{value[:160]}…(+{len(value) - 160} chars)"
            return value
        if isinstance(value, (list, tuple)):
            items = [self._sanitize_debug_value(item, depth + 1) for item in list(value)[:5]]
            if len(value) > 5:
                items.append(f"…(+{len(value) - 5} items)")
            return items
        if isinstance(value, dict):
            result: Dict[str, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                if index >= 8:
                    result["…"] = f"(+{len(value) - index} keys)"
                    break
                result[str(key)] = self._sanitize_debug_value(item, depth + 1)
            return result
        return str(value)

    def _log_debug_event(
        self,
        name: str,
        details: Optional[Mapping[str, Any]] = None,
        *,
        level: Optional[int] = None,
    ) -> None:
        if not details:
            payload: Dict[str, Any] = {}
        else:
            payload = {
                key: self._sanitize_debug_value(value)
                for key, value in details.items()
                if value not in (None, "", [], {})
            }
        if not payload:
            payload = {}
        message = f"{name} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
        debug_enabled = getattr(self, "_debug_enabled", False)
        target_level = level or (logging.INFO if debug_enabled else logging.DEBUG)
        log.log(target_level, message)

    @staticmethod
    def _resolve_api_version(
        *, task: str, model: str, api_version: str | None
    ) -> str:
        """Pick the correct Gemini API version for the requested task/model."""

        requested = (api_version or "").strip().lower()
        if requested in {"text", "v1"}:
            return "v1"
        if requested in {"media", "v1beta"}:
            return "v1beta"
        if requested in {"auto", "default"}:
            requested = ""

        default_version = "v1beta" if task in {"image", "video"} else "v1"
        if requested:
            # Unknown explicit value – fall back to default to stay operational.
            return default_version

        return default_version

    def _resolve_base_url(self, config: Config) -> str:
        endpoint = (config.gemini_api_endpoint or "").strip()
        if endpoint:
            return endpoint.rstrip("/")
        mode = (config.gemini_api_mode or "developer").strip().lower()
        if mode == "vertex":
            region = (config.vertex_location or "").strip()
            project = (config.vertex_project_id or "").strip()
            if region and project:
                return (
                    f"https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/google/models"
                )
            log.warning(
                "Gemini Vertex API selected but VERTEX_LOCATION or VERTEX_PROJECT_ID is missing; falling back to public endpoint"
            )
        return "https://generativelanguage.googleapis.com"

    def _parse_safety_threshold(
        self, value: Optional[str]
    ) -> types.HarmBlockThreshold:
        if not value:
            return _SAFETY_FALLBACK_THRESHOLD
        normalised = value.strip().upper()
        if not normalised:
            return _SAFETY_FALLBACK_THRESHOLD
        direct = getattr(types.HarmBlockThreshold, normalised, None)
        if isinstance(direct, types.HarmBlockThreshold):
            return direct
        for candidate in types.HarmBlockThreshold:
            if normalised in {candidate.name.upper(), str(candidate.value).upper()}:
                return candidate
        log.warning(
            "Unknown Gemini safety threshold %s; defaulting to %s",
            value,
            _SAFETY_FALLBACK_THRESHOLD.name,
        )
        return _SAFETY_FALLBACK_THRESHOLD

    def _load_available_models(self) -> None:
        candidates: Set[str] = set()
        identifiers: Set[str] = set()
        clients = []
        own_client = getattr(self, "_client", None)
        if own_client is not None:
            clients.append(own_client)
        router_clients = getattr(self._router, "_clients", None)
        if isinstance(router_clients, dict):
            for entry in router_clients.values():
                client_obj = getattr(entry, "_client", None) or getattr(entry, "client", None)
                if client_obj is not None and client_obj not in clients:
                    clients.append(client_obj)
        if not clients:
            for getter in (get_media_client, get_text_client):
                try:
                    client_obj = getter(self._config)
                except Exception as exc:  # pragma: no cover - defensive
                    log.debug(
                        "Failed to build Gemini client for model discovery getter=%s error=%s",
                        getattr(getter, "__name__", str(getter)),
                        exc,
                        exc_info=True,
                    )
                    continue
                if client_obj not in clients:
                    clients.append(client_obj)
        for client_obj in clients:
            try:
                models_iterable = list(client_obj.models.list())
            except Exception as exc:  # pragma: no cover - network guard
                log.warning(
                    "Failed to fetch Gemini models list client=%s error=%s",
                    getattr(client_obj, "__class__", type(client_obj)).__name__,
                    exc,
                    exc_info=True,
                )
                continue
            for model in models_iterable:
                name = getattr(model, "name", None) or getattr(model, "model", None)
                if not isinstance(name, str) or not name:
                    continue
                variants = {name.strip()}
                if "/" in name:
                    variants.add(name.rsplit("/", 1)[-1].strip())
                for variant in variants:
                    if not variant:
                        continue
                    candidates.add(variant)
                    identifiers.add(variant.lower())
        self._available_models = sorted(candidates)
        self._available_model_ids = identifiers
        if self._available_models:
            log.info(
                "Gemini available models loaded count=%s", len(self._available_models)
            )

    def _extract_safety_feedback(self, response: Any, payload: Dict[str, Any]) -> Any:
        if hasattr(response, "prompt_feedback"):
            feedback = getattr(response, "prompt_feedback")
            if hasattr(feedback, "model_dump"):
                try:
                    return feedback.model_dump(mode="json")
                except Exception:  # pragma: no cover - defensive serialisation
                    return feedback
            return feedback
        prompt_feedback = payload.get("prompt_feedback") or payload.get("promptFeedback")
        return prompt_feedback

    def _log_generation_feedback(
        self,
        *,
        status_code: int,
        provider_message: Optional[str],
        safety_feedback: Any,
        corr_id: Optional[str],
        payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        block_reason: str = ""
        safety_ratings: Any = None
        finish_reasons: List[str] = []
        if isinstance(payload, Mapping):
            prompt_feedback = (
                payload.get("prompt_feedback")
                or payload.get("promptFeedback")
                or payload.get("safety_feedback")
                or payload.get("safetyFeedback")
            )
            if isinstance(prompt_feedback, Mapping):
                block_reason = str(
                    prompt_feedback.get("block_reason")
                    or prompt_feedback.get("blockReason")
                    or ""
                )
                safety_ratings = (
                    prompt_feedback.get("safety_ratings")
                    or prompt_feedback.get("safetyRatings")
                )
            candidates = payload.get("candidates")
            if isinstance(candidates, list):
                for candidate in candidates:
                    if not isinstance(candidate, Mapping):
                        continue
                    finish = candidate.get("finishReason") or candidate.get("finish_reason")
                    if not finish:
                        continue
                    finish_text = str(finish).strip()
                    if not finish_text:
                        continue
                    finish_reasons.append(finish_text.upper())
        log.info(
            "gemini.response status=%s block_reason=%s safety_ratings=%s finish_reasons=%s provider_message=%s safety_feedback=%s corr_id=%s",
            status_code,
            block_reason or "",
            _serialise_json(safety_ratings) if safety_ratings not in (None, "") else "",
            ",".join(dict.fromkeys(finish_reasons)) if finish_reasons else "",
            (provider_message or ""),
            _serialise_json(safety_feedback) if safety_feedback not in (None, "") else "",
            corr_id or "",
        )

    def _build_contents(
        self,
        *,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = [{"text": prompt}]
        if settings:
            reference = settings.get("reference_inline_data")
            if isinstance(reference, dict):
                inline_meta = _normalise_inline_blob(reference)
                if inline_meta and inline_meta.get("base64"):
                    raw_bytes = base64.b64decode(inline_meta["base64"], validate=False)
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": inline_meta.get("mime", "image/jpeg"),
                                "data": raw_bytes,
                            }
                        }
                    )
        return [{"role": "user", "parts": parts}]

    def _build_generation_config(
        self,
        settings: Optional[Dict[str, Any]],
        *,
        method: str,
    ) -> Any:
        if method == "generate_images":
            return self._build_images_config(settings)

        kwargs: Dict[str, Any] = {}
        safety_settings: Optional[List[types.SafetySetting]] = None
        if self._task == "text":
            safety_settings = list(self._safety_settings)
        elif self._task == "video":
            safety_settings = list(self._media_safety_settings)
        if not settings:
            if self._task == "image":
                kwargs.setdefault("response_mime_type", "image/png")
            if safety_settings is not None:
                kwargs["safety_settings"] = safety_settings
            return types.GenerateContentConfig(**kwargs)

        allowed_fields = set(types.GenerateContentConfig.model_fields.keys())
        skip_fields = {"http_options", "should_return_http_response"}
        image_config: Dict[str, Any] = {}
        for raw_key, value in settings.items():
            if value is None:
                continue
            key = _normalise_setting_key(raw_key)
            if key in {"reference_inline_data", "image_file_id", "size"}:
                continue
            if key in skip_fields or key not in allowed_fields:
                continue
            if key == "safety_settings":
                parsed = _convert_setting_list(value, types.SafetySetting)
                if parsed is not None and self._task == "text":
                    safety_settings = parsed
                continue
            if key == "tools":
                if self._task == "image":
                    continue
                parsed = _convert_setting_list(value, types.Tool)
                if parsed is not None:
                    kwargs["tools"] = parsed
                continue
            if key == "tool_config":
                if self._task == "image":
                    continue
                parsed = _maybe_model_validate(value, types.ToolConfig)
                if parsed is not None:
                    kwargs["tool_config"] = parsed
                continue
            if key == "response_schema":
                if self._task == "image":
                    continue
                parsed = _maybe_model_validate(value, types.Schema)
                if parsed is not None:
                    kwargs["response_schema"] = parsed
                continue
            if key == "image_config":
                kwargs["image_config"] = value
                continue
            if key in {"stop_sequences", "response_modalities"} and isinstance(value, tuple):
                kwargs[key] = list(value)
                continue
            kwargs[key] = value

        if self._task == "image":
            aspect = settings.get("aspect_ratio")
            if not aspect:
                inferred = _infer_aspect_ratio(str(settings.get("size"))) if settings.get("size") else None
                if inferred:
                    aspect = inferred
            if aspect:
                image_config.setdefault("aspect_ratio", aspect)
            mime = settings.get("mime_type") or settings.get("response_mime_type")
            if method == "generate_content":
                if "response_mime_type" in kwargs:
                    kwargs.pop("response_mime_type", None)
                if mime:
                    image_config.setdefault("mime_type", mime)
                response_modalities = kwargs.get("response_modalities")
                if not response_modalities:
                    kwargs["response_modalities"] = ["IMAGE"]
                elif isinstance(response_modalities, tuple):
                    kwargs["response_modalities"] = list(response_modalities)
            else:
                if mime:
                    kwargs.setdefault("response_mime_type", mime)
                else:
                    kwargs.setdefault("response_mime_type", "image/png")
            if image_config:
                existing_config = kwargs.get("image_config")
                if isinstance(existing_config, dict):
                    merged = dict(existing_config)
                    merged.update({k: v for k, v in image_config.items() if v is not None})
                    kwargs["image_config"] = merged
                else:
                    kwargs["image_config"] = {
                        k: v for k, v in image_config.items() if v is not None
                    }

        if safety_settings is not None:
            kwargs["safety_settings"] = safety_settings
        return types.GenerateContentConfig(**kwargs)

    def _build_images_config(
        self, settings: Optional[Dict[str, Any]]
    ) -> types.GenerateImagesConfig:
        kwargs: Dict[str, Any] = {}

        if isinstance(settings, Mapping) and not isinstance(settings, dict):
            settings = dict(settings.items())
        if isinstance(settings, dict):
            settings = {k: v for k, v in settings.items() if k != "model"}

        model_fields = getattr(
            types.GenerateImagesConfig, "model_fields", None
        ) or getattr(types.GenerateImagesConfig, "__fields__", {})
        allowed_fields = set(model_fields.keys())
        allowed_fields.discard("model")

        if not settings:
            config_fields: Dict[str, Any] = {}
            config = types.GenerateImagesConfig(**config_fields)
            return config

        for raw_key, value in settings.items():
            if value is None:
                continue
            key = _normalise_setting_key(raw_key)
            if key in {"reference_inline_data", "image_file_id"}:
                continue
            if key == "model":
                continue
            if key == "safety_settings":
                # The images API does not accept per-request safety settings.
                continue
            if key == "response_mime_type":
                key = "mime_type"
            if key == "aspectratio":
                key = "aspect_ratio"
            if key not in allowed_fields:
                continue
            kwargs[key] = value

        size_value = settings.get("size") if isinstance(settings, dict) else None
        if (
            size_value
            and "size" in allowed_fields
            and "size" not in kwargs
        ):
            kwargs["size"] = size_value
            inferred = _infer_aspect_ratio(str(size_value))
            if (
                inferred
                and "aspect_ratio" in allowed_fields
                and "aspect_ratio" not in kwargs
            ):
                kwargs["aspect_ratio"] = inferred

        aspect = settings.get("aspect_ratio") if isinstance(settings, dict) else None
        if (
            aspect
            and "aspect_ratio" in allowed_fields
            and "aspect_ratio" not in kwargs
        ):
            kwargs["aspect_ratio"] = aspect

        mime = None
        if isinstance(settings, dict):
            mime = settings.get("mime_type") or settings.get("response_mime_type")
            if (
                mime
                and "mime_type" in allowed_fields
                and "mime_type" not in kwargs
            ):
                kwargs["mime_type"] = mime

        config_fields = {
            key: value
            for key, value in kwargs.items()
            if key in allowed_fields and key not in {"model", "safety_settings"}
        }
        config = types.GenerateImagesConfig(**config_fields)
        return config

    def _prepare_generate_call(
        self,
        *,
        decision: "RouteDecision",
        prompt: str,
        settings: Dict[str, Any],
        payload: Dict[str, Any],
        force_mode: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        contents = payload.get("contents")
        config = payload.get("config")
        method = force_mode or decision.method
        predict_target = (
            decision.task == "image"
            and isinstance(decision.model, str)
            and decision.model.lower() == "gemini-2.5-flash-image"
        )
        capability = get_predict_capability(decision.model)
        if predict_target and self._predict_temporarily_blocked():
            ttl_left = max(0, int((self._predict_block_until or 0) - time.time()))
            method = "generate_content"
            if DEBUG_GEMINI:
                log.warning(
                    "gemini.predict.skip %s",
                    kv(
                        model=decision.model,
                        task=decision.task,
                        ttl_left=ttl_left,
                    ),
                )
        elif predict_target and capability is False:
            method = "generate_content"
        elif predict_target and not force_mode:
            method = "generate_images"
        prompt_text = prompt
        if contents is None:
            contents = self._build_contents(prompt=prompt, settings=settings)
        else:
            text_parts: List[str] = []
            for entry in contents:
                if not isinstance(entry, Mapping):
                    continue
                parts = entry.get("parts")
                if not isinstance(parts, list):
                    continue
                for part in parts:
                    if not isinstance(part, Mapping):
                        continue
                    text_value = part.get("text")
                    if isinstance(text_value, str) and text_value:
                        text_parts.append(text_value)
            if text_parts:
                prompt_text = "\n".join(text_parts)
        if prompt_text:
            prompt_text = re.sub(r"(?im)^\s*negative\s*:\s*.*(?:$|\n)", "", prompt_text)
            prompt_text = re.sub(r"\n{2,}", "\n", prompt_text).strip()
        if not prompt_text:
            prompt_text = prompt or ""
        expected_config = (
            types.GenerateImagesConfig
            if method == "generate_images"
            else types.GenerateContentConfig
        )
        if method == "generate_images":
            if not isinstance(config, types.GenerateImagesConfig):
                config = self._build_generation_config(settings, method=method)
        else:
            if not isinstance(config, expected_config):
                config = self._build_generation_config(settings, method=method)
        if method == "generate_images":
            request_kwargs = {
                "model": decision.model,
                "prompt": prompt_text,
                "config": config,
            }
        else:
            request_kwargs = {
                "model": decision.model,
                "contents": contents,
                "config": config,
            }
            if self._task != "image":
                if "tools" in payload:
                    request_kwargs["tools"] = payload["tools"]
                if "system_instruction" in payload:
                    request_kwargs["system_instruction"] = payload["system_instruction"]
        request_meta = {
            "mode": method,
            "api_version": decision.api_version,
            "config": config.model_dump(mode="json") if hasattr(config, "model_dump") else config,
        }
        settings_dict = dict(settings or {})
        payload_keys = sorted(payload.keys()) if isinstance(payload, dict) else []
        self._log_debug_event(
            "gemini.prepare",
            {
                "task": decision.task,
                "model": decision.model,
                "method": method,
                "router_method": decision.method,
                "api_version": decision.api_version,
                "force_mode": force_mode or "",
                "prompt_chars": len(prompt_text or ""),
                "prompt_preview": _truncate_preview(prompt_text),
                "settings_keys": sorted(settings_dict.keys()),
                "size": settings_dict.get("size"),
                "aspect_ratio": settings_dict.get("aspect_ratio"),
                "mime_type": settings_dict.get("mime_type")
                or settings_dict.get("response_mime_type"),
                "payload_keys": payload_keys,
            },
        )
        if DEBUG_GEMINI:
            log.debug(
                "gemini.prepare %s",
                kv(
                    task=decision.task,
                    model=decision.model,
                    api_version=decision.api_version,
                    mode=method,
                    force_mode=force_mode or "",
                    contents_present=bool(contents),
                    prompt_preview=(prompt_text[:140] if prompt_text else ""),
                ),
            )
        generator = getattr(decision.client.models, method, None)
        if generator is None:
            # Some unit tests stub only the router-selected method. If the
            # capability cache forces a fallback method that the stub does not
            # implement we try the router suggestion before erroring out.
            generator = getattr(decision.client.models, decision.method, None)
        if generator is None:
            raise AttributeError(
                f"Gemini client has no generator for method={method}"
            )
        return generator, request_kwargs, request_meta

    def _map_error(
        self,
        exc: Exception,
        *,
        duration_ms: int,
        decision: Optional[RouteDecision] = None,
    ) -> Tuple[ProviderAPIError, Optional[str], Optional[str], Optional[str]]:
        if isinstance(exc, genai_errors.APIError):
            status_code = int(getattr(exc, "code", 0) or 0)
            message = getattr(exc, "message", "Gemini API error") or "Gemini API error"
            retryable = status_code in {408, 409, 429, 500, 502, 503, 504}
            provider_message = getattr(exc, "details", None)
            if not provider_message and hasattr(exc, "response"):
                provider_message = getattr(exc.response, "text", None)
            status_name = str(getattr(exc, "status", "") or "")
            provider_text = ""
            if provider_message is not None:
                try:
                    provider_text = json.dumps(provider_message, ensure_ascii=False)
                except (TypeError, ValueError):
                    provider_text = str(provider_message)
            error_type = "api"
            if status_code in {400, 404}:
                message = (
                    "Эта модель не поддерживает данный метод. Поменяй модель или метод."
                )
                if decision and decision.task in {"image", "video"}:
                    message = (
                        "Проверь версию API: text → v1, video → v1beta, а модели gemini-*-image работают через v1."
                    )
                if status_code == 404 and decision and decision.task == "image":
                    error_type = "provider_misconfigured"
                    retryable = False
                    message = "Модель временно недоступна из-за конфигурации. Попробуй позже, кредиты вернули."
            policy_trigger = (
                "policy" in provider_text.lower()
                or "safety" in provider_text.lower()
                or "policy" in status_name.lower()
                or "safety" in status_name.lower()
            )
            if policy_trigger:
                message = (
                    "Запрос отклонён политикой. Попробуем смягчить формулировку или уберём прямые названия брендов/фильмов."
                )
            if status_code == 403 and "download" in provider_text.lower():
                error_type = "download_failed"
                retryable = False
                message = "Не удалось скачать файл Gemini. Попробуем ещё раз позже и вернём кредиты."
            raw_body = provider_text or str(message)
            error_code = status_name or (str(status_code) if status_code else None)
            return (
                ProviderAPIError(
                    provider=self.provider_name,
                    status_code=status_code,
                    message=message,
                    error_type=error_type,
                    error_code=error_code,
                    provider_message=raw_body,
                    retryable=retryable,
                    duration_ms=duration_ms,
                ),
                error_code,
                message,
                raw_body,
            )
        if isinstance(exc, ValueError):
            text = str(exc) or "Invalid Gemini request"
            return (
                ProviderAPIError(
                    provider=self.provider_name,
                    status_code=400,
                    message=text,
                    error_type="validation",
                    provider_message=text,
                    retryable=False,
                    duration_ms=duration_ms,
                ),
                "value_error",
                text,
                text,
            )
        text = str(exc) or "Gemini request failed"
        return (
            ProviderAPIError(
                provider=self.provider_name,
                status_code=0,
                message=text,
                error_type="network",
                provider_message=text,
                retryable=True,
                duration_ms=duration_ms,
            ),
            exc.__class__.__name__,
            text,
            text,
        )

    def _extract_threshold_error_category(self, exc: Exception) -> Optional[str]:
        if not isinstance(exc, genai_errors.APIError):
            return None
        message = str(getattr(exc, "message", "") or "")
        if not message:
            message = str(getattr(exc, "details", "") or "")
        if not message:
            message = str(exc)
        match = re.search(r"category\s+([A-Z_]+)", message, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    def _downgrade_safety_config(
        self,
        config: Any,
        category: str,
    ) -> Optional[types.GenerateContentConfig]:
        if not isinstance(config, types.GenerateContentConfig):
            return None
        try:
            dump = config.model_dump(mode="json")
        except Exception:  # pragma: no cover - defensive
            return None
        settings = dump.get("safety_settings")
        if not isinstance(settings, list):
            return None
        changed = False
        target = category.upper()
        fallback_value = _enum_code(_SAFETY_FALLBACK_THRESHOLD)
        for entry in settings:
            if not isinstance(entry, dict):
                continue
            cat = str(entry.get("category") or "").upper()
            if cat.endswith(target) or cat == target:
                current = str(entry.get("threshold") or "").upper()
                if current == fallback_value.upper():
                    return None
                entry["threshold"] = fallback_value
                changed = True
                break
        if not changed:
            return None
        try:
            return types.GenerateContentConfig.model_validate(dump)
        except Exception:  # pragma: no cover - defensive revalidation
            log.warning(
                "Failed to downgrade safety config for category=%s", category, exc_info=True
            )
            return None

    async def _submit_single_attempt(
        self,
        *,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        **_: Any,
    ) -> ProviderJobSubmission:
        if not self._api_key:
            raise ProviderAPIError(
                provider=self.provider_name,
                status_code=401,
                message="Gemini API key is not configured",
                error_type="auth",
            )

        request_settings = dict(settings or {})
        payload_dict = dict(payload or {})
        allowed_keys: Optional[Set[str]] = None
        if self.provider_name == "gemini-image":
            allowed_keys = GEMINI_IMAGE_ALLOWED
        elif self.provider_name == "gemini-text":
            allowed_keys = GEMINI_TEXT_ALLOWED
        if allowed_keys is not None:
            cleaned_payload = sanitize_payload(payload_dict, allowed_keys)
            log_removed_keys(
                f"{self.provider_name}", payload_dict, cleaned_payload, logger=log
            )
            payload_dict = cleaned_payload

        attempt = 0
        delay = self._config.retry_backoff
        last_error: Optional[ProviderAPIError] = None
        downgraded_categories: Set[str] = set()
        server_retry_index = 0
        image_retry_count = 0
        force_mode: Optional[str] = None
        forced_api_version: Optional[str] = None
        fallback_to_content = False
        attempted_api_versions: Set[str] = set()
        capability_forced = False
        capability_hint: Optional[bool] = None
        no_image_streak = 0
        while attempt <= self._config.request_retries:
            model_override = request_settings.get("model") or self._model
            if (
                self._task == "image"
                and isinstance(model_override, str)
                and model_override.lower() == "gemini-2.5-flash-image"
            ):
                capability_hint = get_predict_capability(model_override)
                if capability_hint is False:
                    if force_mode != "generate_content":
                        force_mode = "generate_content"
                        capability_forced = True
                        if not forced_api_version:
                            forced_api_version = "v1beta"
                        _trace_log(
                            logging.INFO,
                            "capability.predict_disabled",
                            model=model_override,
                            corr_id=idempotency_key or "",
                            source="cache",
                        )
                elif capability_hint is True and capability_forced:
                    capability_forced = False
            else:
                capability_hint = None
            try:
                decision = self._router.route(
                    task=self._task,
                    model=model_override,
                    prompt=prompt,
                    assets=request_settings,
                    forced_api_version=forced_api_version,
                )
            except GeminiRoutingError as exc:
                raise ProviderAPIError(
                    provider=self.provider_name,
                    status_code=400,
                    message=str(exc),
                    error_type="routing",
                    retryable=False,
                ) from exc

            self._log_debug_event(
                "gemini.attempt",
                {
                    "attempt": attempt + 1,
                    "task": self._task,
                    "decision_model": decision.model,
                    "decision_method": decision.method,
                    "forced_mode": force_mode or "",
                    "forced_version": forced_api_version or "",
                    "fallback_to_content": fallback_to_content,
                },
            )
            generator, request_kwargs, request_meta = self._prepare_generate_call(
                decision=decision,
                prompt=decision.prompt,
                settings=request_settings,
                payload=payload_dict,
                force_mode=force_mode,
            )
            decision_version = (decision.api_version or "").strip().lower()
            if decision_version:
                attempted_api_versions.add(decision_version)
            actual_method = request_meta.get("mode", decision.method)
            request_meta.update(
                {
                    "mode": actual_method,
                    "method": actual_method,
                    "model": decision.model,
                    "api_version": decision.api_version,
                    "task": decision.task,
                }
            )
            if decision.rewrite_notes:
                request_meta["prompt_notes"] = decision.rewrite_notes
            request_meta.setdefault("prompt", decision.prompt)
            start = time.monotonic()
            corr_id = idempotency_key or ""
            if DEBUG_GEMINI:
                log.info(
                    "gemini.generate start %s",
                    kv(
                        model=decision.model,
                        method=actual_method,
                        version=decision.api_version,
                        env=self._environment,
                        key_mask=self._key_mask or "",
                        corr_id=corr_id,
                        predict_blocked=self._predict_temporarily_blocked(),
                        attempt=attempt + 1,
                        retry_index=image_retry_count,
                        force_mode=force_mode or "",
                    ),
                )
            self._log_debug_event(
                "gemini.call.start",
                {
                    "model": decision.model,
                    "method": actual_method,
                    "api_version": decision.api_version,
                    "task": decision.task,
                    "corr_id": idempotency_key,
                    "request_meta": request_meta,
                },
            )
            trace_payload_path: Optional[Path] = None
            trace_event: Dict[str, Any] = {}
            if GEMINI_TRACE or GEMINI_TRACE_CURL:
                trace_payload_path = _trace_save_request_payload(
                    corr_id=corr_id or idempotency_key,
                    method=actual_method,
                    payload=request_kwargs,
                )
            if GEMINI_TRACE:
                trace_event = _trace_request(
                    client=self,
                    decision=decision,
                    method=actual_method,
                    api_version=decision.api_version,
                    attempt=attempt + 1,
                    corr_id=corr_id,
                    request_kwargs=request_kwargs,
                    payload=payload_dict,
                    force_mode=force_mode,
                    payload_path=trace_payload_path,
                )
            if GEMINI_TRACE_CURL and trace_payload_path is not None:
                _trace_log_curl(
                    client=self,
                    decision=decision,
                    method=actual_method,
                    api_version=decision.api_version,
                    payload_path=trace_payload_path,
                    corr_id=corr_id,
                )
            try:
                await self._rate_limiter.acquire()
                if (
                    self._task == "image"
                    and isinstance(decision.model, str)
                    and decision.model.lower() == "gemini-2.5-flash-image"
                    and actual_method == "generate_content"
                ):
                    log.info(
                        "gemini.sync.request.start",
                        extra={
                            "model": decision.model,
                            "task": decision.task,
                            "provider": self.provider_name,
                        },
                    )
                response = await asyncio.to_thread(generator, **request_kwargs)
                duration_ms = int((time.monotonic() - start) * 1000)
                if hasattr(response, "model_dump"):
                    payload_json = response.model_dump(mode="json")
                elif isinstance(response, dict):
                    payload_json = response
                else:
                    payload_json = {"response": response}
                safety_feedback = self._extract_safety_feedback(response, payload_json)
                headers = _extract_response_headers(response)
                response_id = getattr(response, "response_id", None)
                candidates_len = 0
                if isinstance(payload_json, Mapping):
                    candidates_payload = payload_json.get("candidates")
                    if isinstance(candidates_payload, list):
                        candidates_len = len(candidates_payload)
                response_candidates = getattr(response, "candidates", None)
                if response_candidates and not candidates_len:
                    try:
                        candidates_len = len(response_candidates)
                    except TypeError:
                        candidates_len = 0
                response_images = getattr(response, "images", None)
                images_len = 0
                if response_images:
                    try:
                        images_len = len(response_images)
                    except TypeError:
                        images_len = 1
                elif isinstance(payload_json, Mapping):
                    for key in ("images", "generated_images", "generatedImages"):
                        maybe_images = payload_json.get(key)
                        if isinstance(maybe_images, list):
                            images_len = len(maybe_images)
                            if images_len:
                                break
                finish_reason = None
                try:
                    if getattr(response, "candidates", None):
                        primary_candidate = response.candidates[0]
                        finish_reason = getattr(primary_candidate, "finish_reason", None)
                    elif isinstance(payload_json, dict):
                        first_candidate = (payload_json.get("candidates") or [{}])[0]
                        if isinstance(first_candidate, Mapping):
                            finish_reason = first_candidate.get("finishReason") or first_candidate.get(
                                "finish_reason"
                            )
                except Exception:  # pragma: no cover - defensive parsing
                    finish_reason = None

                finish_code_upper = (
                    str(finish_reason).upper() if finish_reason else ""
                )

                if GEMINI_TRACE:
                    trace_payload = (
                        payload_json if isinstance(payload_json, Mapping) else {}
                    )
                    _trace_response(
                        client=self,
                        decision=decision,
                        method=actual_method,
                        api_version=decision.api_version,
                        corr_id=corr_id,
                        status=200,
                        headers=headers,
                        duration_ms=duration_ms,
                        payload_json=trace_payload,
                        request_event=trace_event,
                        response_id=response_id,
                    )

                inline_present = False
                parts_count = 0
                if isinstance(payload_json, Mapping):
                    inline_present = bool(
                        payload_json.get("inline_data") or payload_json.get("inlineData")
                    )
                    if not inline_present:
                        candidates_payload = payload_json.get("candidates")
                        if isinstance(candidates_payload, list):
                            for candidate in candidates_payload:
                                if not isinstance(candidate, Mapping):
                                    continue
                                content_json = candidate.get("content")
                                parts_json = None
                                if isinstance(content_json, Mapping):
                                    parts_json = content_json.get("parts")
                                elif isinstance(content_json, list):
                                    parts_json = content_json
                                if isinstance(parts_json, list):
                                    parts_count += len(parts_json)
                                    for part in parts_json:
                                        if isinstance(part, Mapping) and (
                                            part.get("inline_data") or part.get("inlineData")
                                        ):
                                            inline_present = True
                                            break
                                    if inline_present:
                                        break
                if not inline_present and getattr(response, "candidates", None):
                    for candidate in response.candidates or []:
                        content_obj = getattr(candidate, "content", None)
                        parts = getattr(content_obj, "parts", None)
                        if not parts:
                            continue
                        try:
                            parts_count += len(parts)
                        except TypeError:
                            parts_count += 0
                        for part in parts:
                            data_obj = getattr(part, "inline_data", None) or getattr(
                                part, "inlineData", None
                            )
                            if data_obj:
                                inline_present = True
                                break
                        if inline_present:
                            break

                safety_summary = None
                if isinstance(safety_feedback, Mapping):
                    ratings = safety_feedback.get("safety_ratings") or safety_feedback.get("safetyRatings")
                    if isinstance(ratings, list):
                        categories: List[str] = []
                        for rating in ratings:
                            if isinstance(rating, Mapping):
                                category = rating.get("category") or rating.get("category_name")
                                if isinstance(category, Mapping):
                                    category = category.get("name") or category.get("value")
                                if category:
                                    categories.append(str(category))
                        if categories:
                            unique_categories = sorted({entry for entry in categories if entry})
                            safety_summary = ",".join(unique_categories[:6])
                        else:
                            safety_summary = f"len={len(ratings)}"
                    elif ratings is not None:
                        safety_summary = str(ratings)
                elif safety_feedback:
                    safety_summary = str(type(safety_feedback).__name__)

                candidates_payload = []
                if isinstance(payload_json, Mapping):
                    raw_candidates = payload_json.get("candidates")
                    if isinstance(raw_candidates, list):
                        candidates_payload = raw_candidates
                if DEBUG_GEMINI:
                    log.info(
                        "gemini.sync.result.meta %s",
                        kv(
                            headers=headers,
                            candidates=len(candidates_payload),
                            images=images_len,
                            finish=finish_reason,
                            response_id=getattr(response, "response_id", None),
                        ),
                    )
                self._log_debug_event(
                    "gemini.call.result",
                    {
                        "model": decision.model,
                        "method": actual_method,
                        "api_version": decision.api_version,
                        "duration_ms": duration_ms,
                        "status_code": 200,
                        "finish_reason": (
                            payload_json.get("finish_reason")
                            or payload_json.get("finishReason")
                            or getattr(response, "finish_reason", None)
                        ),
                        "candidate_count": len(payload_json.get("candidates", []) or []),
                        "image_count": len(payload_json.get("images", []) or []),
                        "generated_images": len(
                            payload_json.get("generated_images", [])
                            or payload_json.get("generatedImages", [])
                        ),
                        "prompt_feedback": bool(
                            payload_json.get("prompt_feedback")
                            or payload_json.get("promptFeedback")
                        ),
                    },
                )
                is_flash_image = (
                    self._task == "image"
                    and isinstance(decision.model, str)
                    and decision.model.lower() == "gemini-2.5-flash-image"
                )
                if is_flash_image:
                    try:
                        if actual_method == "generate_images":
                            inline_assets, asset_map, asset_meta = _prepare_images_assets(
                                response,
                                provider_name=self.provider_name,
                                duration_ms=duration_ms,
                            )
                            finish_reason: Optional[str] = None
                            response_id = getattr(response, "response_id", None)
                        else:
                            (
                                inline_assets,
                                asset_map,
                                asset_meta,
                                finish_reason,
                                response_id,
                            ) = _prepare_flash_image_assets(
                                response,
                                payload_json,
                                provider_name=self.provider_name,
                                duration_ms=duration_ms,
                            )
                        finish_code_upper = (
                            finish_reason or finish_code_upper or ""
                        ).upper()
                        no_image_streak = 0
                        if actual_method == "generate_images":
                            mark_predict_capability(decision.model, True)
                        self._log_debug_event(
                            "gemini.inline_assets",
                            {
                                "corr_id": idempotency_key,
                                "inline_count": len(inline_assets),
                                "asset_bytes": [asset.get("bytes") for asset in inline_assets],
                                "asset_mime": [asset.get("mime") for asset in inline_assets],
                                "method": actual_method,
                            },
                        )
                        if DEBUG_GEMINI and inline_assets:
                            total_bytes = 0
                            mimes: Set[str] = set()
                            for asset in inline_assets:
                                try:
                                    total_bytes += int(asset.get("bytes") or 0)
                                except Exception:
                                    continue
                                mime_val = asset.get("mime")
                                if isinstance(mime_val, str) and mime_val:
                                    mimes.add(mime_val)
                            log.info(
                                "gemini.image.assets %s",
                                kv(
                                    count=len(inline_assets),
                                    total_bytes=total_bytes,
                                    mimes=";".join(sorted(mimes)) if mimes else "",
                                ),
                            )
                    except GeminiImageEmptyError as image_error:
                        finish_code, provider_message, safety_block = _analyse_empty_image_response(
                            payload_json if isinstance(payload_json, Mapping) else {},
                            response=response,
                            error=image_error,
                        )
                        self._log_debug_event(
                            "gemini.empty_image",
                            {
                                "method": actual_method,
                                "finish_code": finish_code,
                                "safety_block": safety_block,
                                "response_id": image_error.response_id,
                                "prompt_feedback": bool(
                                    payload_json.get("prompt_feedback")
                                    if isinstance(payload_json, Mapping)
                                    else False
                                ),
                            },
                        )
                        if DEBUG_GEMINI:
                            log.warning(
                                "gemini.sync.result.empty %s",
                                kv(
                                    finish=finish_code,
                                    safety=safety_block,
                                    provider_message=provider_message,
                                    response_id=image_error.response_id
                                    or getattr(response, "response_id", None),
                                ),
                            )
                        finish_code_upper = (finish_code or "").upper()
                        log.warning(
                            "gemini.sync.result.empty finish_reason=%s response_id=%s",
                            finish_code_upper or "",
                            image_error.response_id
                            or getattr(response, "response_id", "")
                            or "",
                        )
                        self._log_generation_feedback(
                            status_code=200,
                            provider_message=provider_message or image_error.message,
                            safety_feedback=safety_feedback,
                            corr_id=idempotency_key,
                            payload=payload_json if isinstance(payload_json, Mapping) else None,
                        )
                        if finish_code_upper == "NO_IMAGE":
                            no_image_streak += 1
                            increment_metric(
                                "gemini_no_image_total",
                                tags={"model": str(decision.model)},
                            )
                            summary = _summarise_prompt_payload(
                                request_kwargs,
                                payload_dict,
                            )
                            _trace_log(
                                logging.WARNING,
                                "no_image",
                                corr_id=corr_id,
                                model=decision.model,
                                method=actual_method,
                                version=decision.api_version or "",
                                latency_ms=duration_ms,
                                attempt=attempt + 1,
                                retry=image_retry_count,
                                prompt_chars=summary["prompt_chars"],
                                assets=summary["asset_count"],
                                inline_refs=summary["inline_refs"],
                            )
                            log.warning(
                                "NO_IMAGE with 200 OK %s",
                                kv(
                                    corr_id=corr_id,
                                    model=decision.model,
                                    method=actual_method,
                                    version=decision.api_version or "",
                                    latency_ms=duration_ms,
                                    attempt=attempt + 1,
                                    retry=image_retry_count,
                                    prompt_chars=summary["prompt_chars"],
                                    assets=summary["asset_count"],
                                    inline_refs=summary["inline_refs"],
                                ),
                            )
                            self._save_no_image_diag(
                                decision=decision,
                                method=actual_method,
                                payload_json=
                                payload_json if isinstance(payload_json, Mapping) else {},
                                headers=headers,
                                response=response,
                                finish=finish_code_upper,
                                safety_feedback=safety_feedback,
                                error=image_error,
                                duration_ms=duration_ms,
                                corr_id=corr_id,
                            )
                            if not safety_block:
                                if no_image_streak == 1:
                                    image_retry_count = 0
                                    continue
                                if no_image_streak == 2:
                                    jitter = random.uniform(0.3, 0.8)
                                    await asyncio.sleep(jitter)
                                    image_retry_count = 0
                                    continue
                                provider_text = provider_message or image_error.message
                                no_image_error = ProviderAPIError(
                                    provider=self.provider_name,
                                    status_code=503,
                                    message="Изображение не готово, попробуй ещё раз чуть позже.",
                                    error_type="provider_unavailable",
                                    provider_message=provider_text,
                                    retryable=False,
                                    duration_ms=duration_ms,
                                )
                                setattr(no_image_error, "finish_reason", finish_code_upper or "NO_IMAGE")
                                setattr(no_image_error, "corr_id", corr_id)
                                raise no_image_error from image_error
                        else:
                            no_image_streak = 0
                        if (
                            actual_method == "generate_images"
                            and not safety_block
                            and not fallback_to_content
                        ):
                            log.warning(
                                "gemini.image.empty -> fallback: switching method generate_images -> generate_content; reason=%s",
                                finish_code_upper or "NO_IMAGE",
                            )
                            self._log_debug_event(
                                "gemini.fallback",
                                {
                                    "from": "generate_images",
                                    "to": "generate_content",
                                    "finish_code": finish_code_upper or "",
                                    "reason": "empty_image",
                                },
                            )
                            force_mode = "generate_content"
                            fallback_to_content = True
                            image_retry_count = 0
                            no_image_streak = 0
                            continue
                        if safety_block or finish_code_upper in _IMAGE_BLOCK_REASONS:
                            increment_metric(
                                "gemini_safety_blocks_total",
                                tags={
                                    "model": str(decision.model),
                                    "category": finish_code_upper or "unknown",
                                },
                            )
                            raise ProviderAPIError(
                                provider=self.provider_name,
                                status_code=400,
                                message=(
                                    "Запрос отклонён политикой безопасности. "
                                    "Попробуй безопасную формулировку: плюшевая капибара в игрушечном "
                                    "авто на закрытой площадке, статичный кадр."
                                ),
                                error_type="safety",
                                provider_message=provider_message or image_error.message,
                                retryable=False,
                            ) from image_error
                        retriable = (
                            finish_code_upper in _IMAGE_RETRYABLE_REASONS or not finish_code_upper
                        )
                        if (
                            actual_method == "generate_images"
                            and retriable
                            and image_retry_count < len(_IMAGE_RETRY_DELAYS)
                        ):
                            retry_delay = _IMAGE_RETRY_DELAYS[image_retry_count]
                            image_retry_count += 1
                            log.warning(
                                "Retrying Gemini image request finish_reason=%s delay_ms=%s attempt=%s/%s",
                                finish_code_upper or "",
                                int(retry_delay * 1000),
                                image_retry_count,
                                len(_IMAGE_RETRY_DELAYS),
                            )
                            self._log_debug_event(
                                "gemini.retry",
                                {
                                    "reason": finish_code_upper or "",
                                    "delay_ms": int(retry_delay * 1000),
                                    "attempt": image_retry_count,
                                    "max_attempts": len(_IMAGE_RETRY_DELAYS),
                                },
                            )
                            await asyncio.sleep(retry_delay)
                            continue
                        raise ProviderAPIError(
                            provider=self.provider_name,
                            status_code=503,
                            message=(
                                "Модель не вернула изображение. Такое бывает при перегрузке. "
                                "Попробуй ещё раз или переформулируй запрос."
                            ),
                            error_type="provider_unavailable",
                            provider_message=provider_message or image_error.message,
                            retryable=False,
                        ) from image_error
                    else:
                        payload_json = dict(payload_json)
                        payload_json.setdefault("inline_assets", inline_assets)
                        if asset_meta:
                            assets_meta = payload_json.setdefault("assets_meta", {})
                            if isinstance(assets_meta, dict):
                                assets_meta.update(asset_meta)
                            else:
                                payload_json["assets_meta"] = dict(asset_meta)
                        if finish_reason:
                            payload_json.setdefault("finish_reason", finish_reason)
                        if response_id:
                            payload_json.setdefault("response_id", response_id)
                        log.info(
                            "gemini.sync.result.ok",
                            extra={"assets": len(asset_map), "response_id": response_id or ""},
                        )
                        log.info(
                            "gemini.image.inline assets_count=%s inline_count=%s finish_reason=%s response_id=%s",
                            len(asset_map),
                            len(inline_assets),
                            (finish_reason or "") or "",
                            response_id or "",
                        )
                        if DEBUG_GEMINI:
                            total_bytes = sum(
                                int(asset.get("bytes") or 0) for asset in inline_assets
                            )
                            mimes = sorted(
                                {
                                    str(asset.get("mime"))
                                    for asset in inline_assets
                                    if asset.get("mime")
                                }
                            )
                            log.info(
                                "gemini.image.assets %s",
                                kv(
                                    count=len(inline_assets),
                                    total_bytes=total_bytes,
                                    mimes=";".join(mimes),
                                ),
                            )
                        size = request_settings.get("size")
                        corr_id = idempotency_key or response_id or "inline"
                        self._log_generation_feedback(
                            status_code=200,
                            provider_message=None,
                            safety_feedback=safety_feedback,
                            corr_id=corr_id,
                            payload=payload_json,
                        )
                        job_id = str(uuid4())
                        assets_meta: Dict[str, Any] = dict(asset_meta)
                        meta = {
                            "prompt": decision.prompt,
                            "settings": request_settings or {},
                            "payload": payload_json,
                            "assets_meta": assets_meta,
                            "request": request_meta,
                            "request_mode": actual_method,
                            "task": decision.task,
                            "key_mask": self._key_mask,
                            "model": decision.model,
                            "api_version": decision.api_version,
                            "headers": dict(headers or {}),
                        }
                        if response_id:
                            meta["response_id"] = response_id
                        if idempotency_key:
                            meta["idempotency_key"] = idempotency_key
                        self._pending[job_id] = (payload_json, 200, duration_ms, meta)
                        self._persist_pending_record(
                            job_id,
                            payload=payload_json,
                            status_code=200,
                            duration_ms=duration_ms,
                            meta=meta,
                        )
                        self._log_debug_event(
                            "gemini.pending.store",
                            {
                                "job_id": job_id,
                                "corr_id": corr_id,
                                "status_code": 200,
                                "duration_ms": duration_ms,
                                "inline_assets": len(inline_assets),
                                "has_asset_meta": bool(asset_meta),
                            },
                        )
                        size = request_settings.get("size")
                        corr_id = idempotency_key or response_id or job_id
                        self._log_generation_feedback(
                            status_code=200,
                            provider_message=None,
                            safety_feedback=safety_feedback,
                            corr_id=corr_id,
                            payload=payload_json,
                        )
                        metric_tags = {
                            "model": str(decision.model),
                            "method": actual_method,
                            "version": decision.api_version or "",
                            "status": "200",
                        }
                        increment_metric("gemini_calls_total", tags=metric_tags)
                        record_timing_metric(
                            "gemini_latency_ms",
                            duration_ms,
                            tags={
                                "model": metric_tags["model"],
                                "method": metric_tags["method"],
                                "version": metric_tags["version"],
                            },
                        )
                        if finish_code_upper:
                            increment_metric(
                                "metric_gemini_finish_reason",
                                tags={"reason": finish_code_upper},
                            )
                            if finish_code_upper == "STOP":
                                increment_metric(
                                    "gemini_stop_total",
                                    tags={"model": metric_tags["model"]},
                                )
                        log.info(
                            "gemini.call corr_id=%s provider=%s model=%s method=%s version=%s size=%s status=%s latency_ms=%s",
                            corr_id or "",
                            self.provider_name,
                            decision.model,
                            actual_method,
                            decision.api_version,
                            size or "",
                            200,
                            duration_ms,
                        )
                        log.info(
                            "gemini.generate done model=%s method=%s version=%s env=%s key_mask=%s corr_id=%s latency_ms=%s",
                            decision.model,
                            actual_method,
                            decision.api_version,
                            self._environment,
                            self._key_mask or "",
                            corr_id or "",
                            duration_ms,
                        )
                        self._log_debug_event(
                            "gemini.inline_return",
                            {
                                "corr_id": corr_id,
                                "job_id": job_id,
                                "inline_count": len(inline_assets),
                                "asset_bytes": [asset.get("bytes") for asset in inline_assets],
                                "asset_mime": [asset.get("mime") for asset in inline_assets],
                                "duration_ms": duration_ms,
                            },
                        )
                        return ProviderJobSubmission(
                            job_id=job_id,
                            status_code=200,
                            duration_ms=duration_ms,
                            data={
                                "inline_assets": inline_assets,
                                "assets_meta": asset_meta,
                                "payload": payload_json,
                            },
                        )
                job_id = str(uuid4())
                meta = {
                    "prompt": decision.prompt,
                    "settings": request_settings or {},
                    "payload": payload_json,
                    "assets_meta": {},
                    "request": request_meta,
                    "request_mode": actual_method,
                    "task": decision.task,
                    "key_mask": self._key_mask,
                    "model": decision.model,
                    "api_version": decision.api_version,
                    "headers": dict(headers or {}),
                }
                if idempotency_key:
                    meta["idempotency_key"] = idempotency_key
                self._pending[job_id] = (payload_json, 200, duration_ms, meta)
                self._persist_pending_record(
                    job_id,
                    payload=payload_json,
                    status_code=200,
                    duration_ms=duration_ms,
                    meta=meta,
                )
                self._log_debug_event(
                    "gemini.pending.store",
                    {
                        "job_id": job_id,
                        "corr_id": idempotency_key,
                        "status_code": 200,
                        "duration_ms": duration_ms,
                        "inline_assets": len(
                            payload_json.get("inline_assets", [])
                            if isinstance(payload_json, Mapping)
                            else []
                        ),
                        "has_asset_meta": bool(meta.get("assets_meta")),
                    },
                )
                size = request_settings.get("size")
                self._log_generation_feedback(
                    status_code=200,
                    provider_message=None,
                    safety_feedback=safety_feedback,
                    corr_id=idempotency_key or job_id,
                    payload=payload_json,
                )
                no_image_streak = 0
                metric_tags = {
                    "model": str(decision.model),
                    "method": actual_method,
                    "version": decision.api_version or "",
                    "status": "200",
                }
                increment_metric("gemini_calls_total", tags=metric_tags)
                record_timing_metric(
                    "gemini_latency_ms",
                    duration_ms,
                    tags={
                        "model": metric_tags["model"],
                        "method": metric_tags["method"],
                        "version": metric_tags["version"],
                    },
                )
                if finish_code_upper:
                    increment_metric(
                        "metric_gemini_finish_reason",
                        tags={"reason": finish_code_upper},
                    )
                    if finish_code_upper == "STOP":
                        increment_metric(
                            "gemini_stop_total",
                            tags={"model": metric_tags["model"]},
                        )
                log.info(
                    "gemini.call corr_id=%s provider=%s model=%s method=%s version=%s size=%s status=%s latency_ms=%s",
                    idempotency_key or job_id,
                    self.provider_name,
                    decision.model,
                    actual_method,
                    decision.api_version,
                    size or "",
                    200,
                    duration_ms,
                )
                log.info(
                    "gemini.generate done model=%s method=%s version=%s env=%s key_mask=%s corr_id=%s latency_ms=%s",
                    decision.model,
                    actual_method,
                    decision.api_version,
                    self._environment,
                    self._key_mask or "",
                    idempotency_key or job_id,
                    duration_ms,
                )
                return ProviderJobSubmission(
                    job_id=job_id,
                    status_code=200,
                    duration_ms=duration_ms,
                    data=payload_json,
                )
            except genai_errors.APIError as exc:
                duration_ms = int((time.monotonic() - start) * 1000)
                category = self._extract_threshold_error_category(exc)
                if (
                    category
                    and category.upper() not in downgraded_categories
                    and actual_method == "generate_content"
                ):
                    downgraded = self._downgrade_safety_config(
                        request_kwargs.get("config"), category
                    )
                    if downgraded is not None:
                        downgraded_categories.add(category.upper())
                        request_kwargs["config"] = downgraded
                        payload_dict["config"] = downgraded
                        request_meta["config"] = (
                            downgraded.model_dump(mode="json")
                            if hasattr(downgraded, "model_dump")
                            else downgraded
                        )
                        log.warning(
                            "gemini.safety downgrade model=%s category=%s threshold=%s",
                            decision.model,
                            category,
                            _enum_code(_SAFETY_FALLBACK_THRESHOLD),
                        )
                        continue
                provider_error, error_code, error_message, raw_body = self._map_error(
                    exc,
                    duration_ms=duration_ms,
                    decision=decision,
                )
                error_metric_tags = {
                    "model": str(decision.model),
                    "method": actual_method,
                    "version": decision.api_version or "",
                    "status": str(provider_error.status_code or 0),
                }
                increment_metric("gemini_calls_total", tags=error_metric_tags)
                record_timing_metric(
                    "gemini_latency_ms",
                    duration_ms,
                    tags={
                        "model": error_metric_tags["model"],
                        "method": error_metric_tags["method"],
                        "version": error_metric_tags["version"],
                    },
                )
                if (
                    actual_method == "generate_images"
                    and provider_error.status_code in {401, 403, 404}
                ):
                    self._block_predict_now(GEMINI_PREDICT_DISABLE_TTL_SEC)
                    if provider_error.status_code == 404:
                        mark_predict_capability(decision.model, False)
                        capability_forced = True
                        increment_metric(
                            "gemini_predict_404_total",
                            tags={"model": str(decision.model)},
                        )
                    force_mode = "generate_content"
                    fallback_to_content = True
                    image_retry_count = 0
                    if DEBUG_GEMINI:
                        log.warning(
                            "gemini.predict.blocked.retry %s",
                            kv(status=provider_error.status_code, method=actual_method),
                        )
                    continue
                if (
                    decision.task == "image"
                    and provider_error.status_code in {400, 404}
                ):
                    alt_version = "v1beta" if decision_version == "v1" else "v1"
                    if alt_version not in attempted_api_versions:
                        log.warning(
                            "gemini.api.version_switch task=%s model=%s from=%s to=%s",
                            decision.task,
                            decision.model,
                            decision.api_version,
                            alt_version,
                        )
                        if DEBUG_GEMINI:
                            log.warning(
                                "gemini.api.version_switch %s",
                                kv(
                                    task=self._task,
                                    model=decision.model,
                                    from_version=decision.api_version,
                                    to_version=alt_version,
                                    status=provider_error.status_code,
                                ),
                            )
                        self._log_debug_event(
                            "gemini.api.version_switch",
                            {
                                "task": decision.task,
                                "model": decision.model,
                                "from": decision.api_version,
                                "to": alt_version,
                                "status_code": provider_error.status_code,
                            },
                        )
                        forced_api_version = alt_version
                        force_mode = None
                        fallback_to_content = False
                        image_retry_count = 0
                        continue
                is_flash_image = (
                    self._task == "image"
                    and isinstance(decision.model, str)
                    and decision.model.lower() == "gemini-2.5-flash-image"
                )
                if (
                    is_flash_image
                    and actual_method == "generate_images"
                    and provider_error.status_code in {400, 404}
                    and not fallback_to_content
                ):
                    log.warning(
                        "gemini.image.fallback %s",
                        kv(
                            method=actual_method,
                            status=provider_error.status_code,
                            switching_to="generate_content",
                            attempt=f"{image_retry_count + 1}",
                        ),
                    )
                    self._log_debug_event(
                        "gemini.fallback",
                        {
                            "from": "generate_images",
                            "to": "generate_content",
                            "status_code": provider_error.status_code,
                            "reason": "api_error",
                        },
                    )
                    force_mode = "generate_content"
                    fallback_to_content = True
                    image_retry_count = 0
                    continue
                log.error(
                    "gemini.api.error status=%s code=%s message=%s",
                    provider_error.status_code,
                    error_code or provider_error.error_code,
                    error_message or provider_error.message,
                )
                resolution = self._error_handler.resolve(
                    status_code=provider_error.status_code,
                    error_code=error_code or provider_error.error_code,
                    error_message=error_message or provider_error.message,
                )
                if resolution:
                    if (
                        provider_error.status_code == 429
                        and resolution.cooldown_seconds
                    ):
                        await self._rate_limiter.penalize(resolution.cooldown_seconds)
                    self._error_handler.log_admin_hint(resolution)
                    provider_error = self._error_handler.apply(
                        base_error=provider_error,
                        resolution=resolution,
                        raw_body=raw_body,
                        fallback_code=error_code or provider_error.error_code,
                    )
                else:
                    log.error(
                        "gemini.api.unknown status=%s body=%s",
                        provider_error.status_code,
                        raw_body,
                    )
                    provider_error = ProviderAPIError(
                        provider=self.provider_name,
                        status_code=provider_error.status_code,
                        message=provider_error.message,
                        error_type="unknown",
                        error_code=error_code or provider_error.error_code,
                        provider_message=raw_body or provider_error.provider_message,
                        retryable=False,
                        duration_ms=provider_error.duration_ms,
                    )
                self._log_generation_feedback(
                    status_code=provider_error.status_code,
                    provider_message=provider_error.provider_message,
                    safety_feedback=None,
                    corr_id=idempotency_key,
                    payload=None,
                )
                if (
                    provider_error.status_code
                    in self._error_handler.SERVER_ERROR_STATUSES
                ):
                    delay_override = self._error_handler.server_retry_delay(
                        server_retry_index
                    )
                    if delay_override is not None:
                        server_retry_index += 1
                        log.warning(
                            "Retrying Gemini request after server error status=%s attempt=%s delay=%s",
                            provider_error.status_code,
                            server_retry_index,
                            delay_override,
                        )
                        await asyncio.sleep(delay_override)
                        continue
                log.warning(
                    "gemini.generate error model=%s method=%s version=%s env=%s key_mask=%s corr_id=%s status=%s latency_ms=%s",
                    decision.model,
                    actual_method,
                    decision.api_version,
                    self._environment,
                    self._key_mask or "",
                    idempotency_key or "",
                    provider_error.status_code,
                    duration_ms,
                )
                last_error = provider_error
                if not provider_error.retryable or attempt == self._config.request_retries:
                    raise provider_error
                attempt += 1
                log.warning(
                    "Retrying Gemini request provider=%s model=%s method=%s attempt=%s/%s error=%s",
                    self.provider_name,
                    decision.model,
                    actual_method,
                    attempt + 1,
                    self._config.request_retries + 1,
                    provider_error,
                )
                self._log_debug_event(
                    "gemini.retry",
                    {
                        "provider": self.provider_name,
                        "model": decision.model,
                        "method": actual_method,
                        "attempt": attempt + 1,
                        "max_attempts": self._config.request_retries + 1,
                        "error_type": provider_error.error_type,
                        "status_code": provider_error.status_code,
                    },
                )
                await asyncio.sleep(delay)
                delay *= max(1.0, float(self._config.retry_backoff))
                continue
            except Exception as exc:  # pragma: no cover - network guard
                duration_ms = int((time.monotonic() - start) * 1000)
                provider_error, error_code, error_message, raw_body = self._map_error(
                    exc,
                    duration_ms=duration_ms,
                    decision=decision,
                )
                error_metric_tags = {
                    "model": str(decision.model),
                    "method": actual_method,
                    "version": decision.api_version or "",
                    "status": str(provider_error.status_code or 0),
                }
                increment_metric("gemini_calls_total", tags=error_metric_tags)
                record_timing_metric(
                    "gemini_latency_ms",
                    duration_ms,
                    tags={
                        "model": error_metric_tags["model"],
                        "method": error_metric_tags["method"],
                        "version": error_metric_tags["version"],
                    },
                )
                if (
                    actual_method == "generate_images"
                    and provider_error.status_code in {401, 403, 404}
                ):
                    self._block_predict_now(GEMINI_PREDICT_DISABLE_TTL_SEC)
                    if provider_error.status_code == 404:
                        mark_predict_capability(decision.model, False)
                        capability_forced = True
                        increment_metric(
                            "gemini_predict_404_total",
                            tags={"model": str(decision.model)},
                        )
                    force_mode = "generate_content"
                    fallback_to_content = True
                    image_retry_count = 0
                    if DEBUG_GEMINI:
                        log.warning(
                            "gemini.predict.blocked.retry %s",
                            kv(status=provider_error.status_code, method=actual_method),
                        )
                    continue
                log.error(
                    "gemini.api.error status=%s code=%s message=%s",
                    provider_error.status_code,
                    error_code or provider_error.error_code,
                    error_message or provider_error.message,
                )
                resolution = self._error_handler.resolve(
                    status_code=provider_error.status_code,
                    error_code=error_code or provider_error.error_code,
                    error_message=error_message or provider_error.message,
                )
                if resolution:
                    if (
                        provider_error.status_code == 429
                        and resolution.cooldown_seconds
                    ):
                        await self._rate_limiter.penalize(resolution.cooldown_seconds)
                    self._error_handler.log_admin_hint(resolution)
                    provider_error = self._error_handler.apply(
                        base_error=provider_error,
                        resolution=resolution,
                        raw_body=raw_body,
                        fallback_code=error_code or provider_error.error_code,
                    )
                elif 400 <= provider_error.status_code <= 599:
                    log.error(
                        "gemini.api.unknown status=%s body=%s",
                        provider_error.status_code,
                        raw_body,
                    )
                    provider_error = ProviderAPIError(
                        provider=self.provider_name,
                        status_code=provider_error.status_code,
                        message=provider_error.message,
                        error_type="unknown",
                        error_code=error_code or provider_error.error_code,
                        provider_message=raw_body or provider_error.provider_message,
                        retryable=False,
                        duration_ms=provider_error.duration_ms,
                    )
                log.warning(
                    "gemini.generate error model=%s method=%s version=%s env=%s key_mask=%s corr_id=%s status=%s latency_ms=%s",
                    decision.model,
                    actual_method,
                    decision.api_version,
                    self._environment,
                    self._key_mask or "",
                    idempotency_key or "",
                    provider_error.status_code,
                    duration_ms,
                )
                last_error = provider_error
                if not provider_error.retryable or attempt == self._config.request_retries:
                    raise provider_error
                attempt += 1
                log.warning(
                    "Retrying Gemini request provider=%s model=%s method=%s attempt=%s/%s error=%s",
                    self.provider_name,
                    decision.model,
                    actual_method,
                    attempt + 1,
                    self._config.request_retries + 1,
                    provider_error,
                )
                self._log_debug_event(
                    "gemini.retry",
                    {
                        "provider": self.provider_name,
                        "model": decision.model,
                        "method": actual_method,
                        "attempt": attempt + 1,
                        "max_attempts": self._config.request_retries + 1,
                        "error_type": provider_error.error_type,
                        "status_code": provider_error.status_code,
                    },
                )
                await asyncio.sleep(delay)
                delay *= self._config.retry_backoff
        assert last_error is not None  # pragma: no cover
        raise last_error

    async def enqueue_job(
        self,
        *,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        **_: Any,
    ) -> ProviderJobSubmission:
        original_prompt = prompt

        def _noop_attach(text: str, _neg: str) -> str:
            return text

        use_negative = SafetyCfg.ENABLE_NEGATIVE_PROMPT and self._task in {"image", "video"}
        attach_neg_fn = attach_negative_prompt if use_negative else _noop_attach
        current_prompt = (
            attach_neg_fn(prompt, SafetyCfg.NEGATIVE_PROMPT_BASE)
            if SafetyCfg.ENABLE_NEGATIVE_PROMPT
            else prompt
        )
        attempt = 0
        rephraser_fn = default_rephraser(self._api_key)

        while True:
            submission = await self._submit_single_attempt(
                prompt=current_prompt,
                settings=settings,
                payload=payload,
                idempotency_key=idempotency_key,
            )

            pending_record = self._pending.get(submission.job_id)
            payload_json: Dict[str, Any] = {}
            meta: Dict[str, Any] = {}
            if pending_record:
                stored_payload, status_code, duration_ms, stored_meta = pending_record  # type: ignore[misc]
                if isinstance(stored_payload, dict):
                    payload_json = stored_payload
                if isinstance(stored_meta, dict):
                    meta = stored_meta
            elif isinstance(submission.data, dict):
                payload_json = submission.data

            info = extract_block_info(payload_json)
            info_reason = str(info.get("reason") or "").strip()
            reason_upper = info_reason.upper()
            finish_reasons: List[str] = []
            if isinstance(payload_json, dict):
                candidates = payload_json.get("candidates")
                if isinstance(candidates, list):
                    for candidate in candidates:
                        if not isinstance(candidate, dict):
                            continue
                        finish = candidate.get("finishReason") or candidate.get("finish_reason")
                        finish_text = str(finish or "").strip().upper()
                        if not finish_text:
                            continue
                        finish_reasons.append(finish_text)
            prompt_feedback = None
            if isinstance(payload_json, dict):
                prompt_feedback = (
                    payload_json.get("prompt_feedback")
                    or payload_json.get("promptFeedback")
                )
            ratings_block = False
            if isinstance(prompt_feedback, Mapping):
                blocked_flag = prompt_feedback.get("blocked")
                if isinstance(blocked_flag, bool) and blocked_flag:
                    ratings_block = True
                ratings = prompt_feedback.get("safety_ratings") or prompt_feedback.get("safetyRatings")
                if isinstance(ratings, list):
                    for rating in ratings:
                        if not isinstance(rating, Mapping):
                            continue
                        if rating.get("blocked") is True:
                            ratings_block = True
                            break
                        probability = str(rating.get("probability") or "").upper()
                        if probability and probability not in {"LOW", "VERY_LOW", "NEGLIGIBLE", "BLOCK_NONE"}:
                            ratings_block = True
                            break
                        if "BLOCK" in probability:
                            ratings_block = True
                            break
            headers_meta = {}
            if isinstance(meta, Mapping):
                headers_value = meta.get("headers")
                if isinstance(headers_value, Mapping):
                    headers_meta = {str(k).lower(): str(headers_value[k]) for k in headers_value}
            header_block = False
            header_reason = None
            for name, value in headers_meta.items():
                upper_value = str(value or "").upper()
                if not upper_value:
                    continue
                if name in {
                    "x-goog-rai-filtered-reason",
                    "x-goog-ai-response-code",
                    "x-goog-image-response-status",
                } and upper_value:
                    header_block = True
                    header_reason = f"{name}:{upper_value}"
                    break
                if name in {
                    "x-generative-ai-output-status",
                    "x-generative-ai-finish-reason",
                } and any(token in upper_value for token in {"FILTER", "BLOCK", "SAFETY"}):
                    header_block = True
                    header_reason = f"{name}:{upper_value}"
                    break
            finish_block = any(
                finish and finish not in {"STOP", "SUCCESS"} and (
                    finish in _IMAGE_BLOCK_REASONS or "SAFETY" in finish or "BLOCK" in finish
                )
                for finish in finish_reasons
            )
            reason_block = bool(reason_upper and reason_upper not in {"STOP", "SUCCESS"})
            blocked = ratings_block or header_block or finish_block or reason_block
            if blocked:
                if reason_block:
                    info["reason"] = reason_upper
                elif header_reason:
                    info["reason"] = header_reason
                elif finish_block and finish_reasons:
                    info["reason"] = ",".join(
                        finish for finish in finish_reasons if finish not in {"STOP", "SUCCESS"}
                    )
                elif ratings_block:
                    info["reason"] = "SAFETY_RATING"
            if not blocked:
                return submission

            if pending_record:
                self._pending.pop(submission.job_id, None)

            model_name = meta.get("model") or self._model
            api_version = meta.get("api_version") or self._api_version
            log.warning(
                "gemini.safety block provider=%s task=%s model=%s api_version=%s attempt=%s reason=%s categories=%s finish=%s original_prompt=%r sanitized_prompt=%r",
                self.provider_name,
                self._task,
                model_name,
                api_version,
                attempt + 1,
                info.get("reason"),
                info.get("categories"),
                ",".join(dict.fromkeys(finish_reasons)) if finish_reasons else "",
                original_prompt,
                current_prompt,
            )

            if attempt >= SafetyCfg.MAX_RETRIES:
                raise ProviderAPIError(
                    provider=self.provider_name,
                    status_code=400,
                    message="Запрос отклонён политиками Gemini",
                    error_type="safety",
                    provider_message=str(info.get("reason")),
                    retryable=False,
                )

            decision: RetryDecision = await asyncio.to_thread(
                next_attempt,
                original_prompt=original_prompt,
                last_prompt=current_prompt,
                last_response_json=payload_json or {},
                cfg=SafetyCfg,
                rephraser=rephraser_fn,
                scrubber=scrub_brands_and_persons,
                attach_neg=attach_neg_fn,
            )

            if not decision.do_retry:
                raise ProviderAPIError(
                    provider=self.provider_name,
                    status_code=400,
                    message="Gemini заблокировал запрос",
                    error_type="safety",
                    provider_message=str(info.get("reason")),
                    retryable=False,
                )

            current_prompt = decision.next_prompt
            attempt += 1

    async def get_job_status(self, job_id: str) -> ProviderJobStatus:
        record = self._pending.pop(job_id, None)
        source = "memory" if record is not None else "disk"
        if record is None:
            record = self._load_pending_record(job_id)
        if record is None:
            self._log_debug_event(
                "gemini.pending.miss",
                {"job_id": job_id},
                level=logging.DEBUG,
            )
            return ProviderJobStatus(
                job_id=job_id,
                status="failed",
                error="Result not found",
                assets={},
                data={},
                status_code=404,
                duration_ms=0,
            )
        self._delete_pending_record(job_id)
        payload, status_code, duration_ms, meta = record
        status, error, assets, inline_assets, asset_meta = self._extract_result(payload)
        self._log_debug_event(
            "gemini.pending.load",
            {
                "job_id": job_id,
                "source": source,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "inline_count": len(inline_assets),
                "asset_keys": list(assets.keys())[:5],
            },
        )
        inline_assets_copy = [dict(asset) for asset in inline_assets]
        assets_meta = meta.get("assets_meta")
        if not isinstance(assets_meta, dict):
            assets_meta = {}
            meta["assets_meta"] = assets_meta
        for idx, asset in enumerate(inline_assets_copy):
            key = asset.get("key") or f"inline_{idx}"
            asset["key"] = key
            assets_meta[key] = {**asset, "inline": True}
        for key, info in asset_meta.items():
            assets_meta[key] = info
        if status != "completed" and not error:
            error = "Модель не вернула результат"
        meta.setdefault("request_mode", meta.get("request_mode") or meta.get("request", {}).get("mode"))
        return ProviderJobStatus(
            job_id=job_id,
            status=status,
            error=error,
            assets=assets,
            data={
                "response": payload,
                "meta": meta,
                "inline_assets": inline_assets_copy,
            },
            status_code=status_code,
            duration_ms=duration_ms,
        )

    def _pending_path(self, job_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id or "job") or "job"
        return self._pending_dir / f"{safe_id}.json"

    def _persist_pending_record(
        self,
        job_id: str,
        *,
        payload: Dict[str, Any],
        status_code: int,
        duration_ms: int,
        meta: Dict[str, Any],
    ) -> None:
        path = self._pending_path(job_id)
        record = {
            "payload": payload,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "meta": meta,
        }
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False)
        except Exception:  # pragma: no cover - filesystem guard
            log.warning("Failed to persist pending Gemini result job_id=%s", job_id, exc_info=True)

    def _load_pending_record(
        self, job_id: str
    ) -> Optional[Tuple[Dict[str, Any], int, int, Dict[str, Any]]]:
        path = self._pending_path(job_id)
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return None
        except Exception:  # pragma: no cover - filesystem guard
            log.warning("Failed to load pending Gemini result job_id=%s", job_id, exc_info=True)
            return None
        payload = data.get("payload") if isinstance(data, dict) else {}
        if not isinstance(payload, dict):
            payload = {}
        status_code = int(data.get("status_code", 200)) if isinstance(data, dict) else 200
        duration_ms = int(data.get("duration_ms", 0)) if isinstance(data, dict) else 0
        meta = data.get("meta") if isinstance(data, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        return payload, status_code, duration_ms, meta

    def _delete_pending_record(self, job_id: str) -> None:
        path = self._pending_path(job_id)
        try:
            path.unlink(missing_ok=True)
        except Exception:  # pragma: no cover - filesystem guard
            log.debug("Failed to remove pending Gemini cache job_id=%s", job_id, exc_info=True)

    def _extract_result(
        self, payload: Dict[str, Any]
    ) -> Tuple[str, Optional[str], Dict[str, str], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        if not isinstance(payload, dict):
            return "failed", "Invalid response", {}, [], {}
        assets: Dict[str, str] = {}
        inline_assets: List[Dict[str, Any]] = []
        asset_meta: Dict[str, Dict[str, Any]] = {}
        error: Optional[str] = None
        status = "completed"

        inline_inline = payload.get("inline_assets")
        if isinstance(inline_inline, list) and inline_inline:
            for index, item in enumerate(inline_inline):
                if not isinstance(item, dict):
                    continue
                entry = dict(item)
                key = entry.get("key") or f"inline_{index}"
                entry["key"] = key
                data_uri = entry.get("data_uri") or entry.get("data_url")
                if data_uri:
                    assets[key] = str(data_uri)
                inline_assets.append(entry)
                asset_meta[key] = {
                    "inline": True,
                    "kind": entry.get("kind") or "image",
                    "mime": entry.get("mime"),
                    "bytes": entry.get("bytes"),
                }
            if inline_assets:
                return "completed", None, assets, inline_assets, asset_meta

        prompt_feedback = payload.get("prompt_feedback") or payload.get("promptFeedback")
        if isinstance(prompt_feedback, dict):
            block_reason = prompt_feedback.get("block_reason") or prompt_feedback.get("blockReason")
            block_message = prompt_feedback.get("block_reason_message") or prompt_feedback.get("blockReasonMessage")
            if block_reason or block_message:
                status = "failed"
                parts = [str(part) for part in (block_reason, block_message) if part]
                error = " ".join(parts) if parts else str(block_reason)

        candidates = payload.get("candidates")
        if isinstance(candidates, list):
            counters: Dict[str, int] = {
                "text": 0,
                "function_call": 0,
                "function_response": 0,
                "code_execution": 0,
                "part": 0,
            }
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                finish_reason = candidate.get("finishReason") or candidate.get("finish_reason")
                if finish_reason and not error and str(finish_reason).upper() in {"SAFETY", "STOP"}:
                    status = "failed"
                    error = str(finish_reason)
                content = candidate.get("content")
                parts: Sequence[Any] = []
                if isinstance(content, dict):
                    parts = content.get("parts") or []
                elif isinstance(content, list):
                    parts = content
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    inline_data = part.get("inline_data") or part.get("inlineData")
                    text_value = part.get("text")
                    function_call = part.get("function_call") or part.get("functionCall")
                    function_response = part.get("function_response") or part.get("functionResponse")
                    code_result = part.get("code_execution_result") or part.get("codeExecutionResult")

                    if inline_data:
                        meta = _normalise_inline_blob(inline_data)
                        if meta:
                            key = f"inline_{len(inline_assets)}"
                            inline_entry = {
                                "key": key,
                                "kind": meta.get("kind", "inline"),
                                "mime": meta.get("mime"),
                                "bytes": meta.get("bytes"),
                                "data_uri": meta.get("data_uri"),
                                "data_url": meta.get("data_url"),
                            }
                            inline_assets.append(inline_entry)
                            data_uri = inline_entry.get("data_uri") or inline_entry.get("data_url")
                            if data_uri:
                                assets[key] = data_uri
                            asset_meta[key] = {
                                "inline": True,
                                "kind": inline_entry.get("kind"),
                                "mime": inline_entry.get("mime"),
                                "bytes": inline_entry.get("bytes"),
                            }
                    elif isinstance(text_value, str) and text_value.strip():
                        key = f"text_{counters['text']}"
                        counters["text"] += 1
                        assets[key] = text_value
                        asset_meta.setdefault(
                            key,
                            {"inline": False, "kind": "text", "length": len(text_value)},
                        )
                    elif function_call:
                        key = f"function_call_{counters['function_call']}"
                        counters["function_call"] += 1
                        assets[key] = _serialise_json(function_call)
                        asset_meta[key] = {
                            "inline": False,
                            "kind": "function_call",
                            "payload": function_call,
                        }
                    elif function_response:
                        key = f"function_response_{counters['function_response']}"
                        counters["function_response"] += 1
                        assets[key] = _serialise_json(function_response)
                        asset_meta[key] = {
                            "inline": False,
                            "kind": "function_response",
                            "payload": function_response,
                        }
                    elif code_result:
                        key = f"code_execution_{counters['code_execution']}"
                        counters["code_execution"] += 1
                        assets[key] = _serialise_json(code_result)
                        asset_meta[key] = {
                            "inline": False,
                            "kind": "code_execution_result",
                            "payload": code_result,
                        }
                    else:
                        cleaned = {
                            key: value
                            for key, value in part.items()
                            if value not in (None, "", [], {})
                        }
                        if cleaned:
                            key = f"part_{counters['part']}"
                            counters["part"] += 1
                            assets[key] = _serialise_json(cleaned)
                            asset_meta[key] = {
                                "inline": False,
                                "kind": "part",
                                "payload": cleaned,
                            }

        generated_images = payload.get("generated_images") or payload.get("generatedImages")
        if isinstance(generated_images, list):
            return self._extract_legacy_image_result(payload, generated_images)

        if status == "completed" and not assets:
            status = "failed"
            if not error:
                error = "Модель не вернула результат"
        return status, error, assets, inline_assets, asset_meta

    def _extract_legacy_image_result(
        self,
        payload: Dict[str, Any],
        generated_images: Sequence[Any],
    ) -> Tuple[str, Optional[str], Dict[str, str], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        assets: Dict[str, str] = {}
        inline_assets: List[Dict[str, Any]] = []
        asset_meta: Dict[str, Dict[str, Any]] = {}
        error: Optional[str] = None
        status = "completed"

        prompt_feedback = payload.get("prompt_feedback") or payload.get("promptFeedback")
        if isinstance(prompt_feedback, dict):
            block_reason = prompt_feedback.get("block_reason") or prompt_feedback.get("blockReason")
            block_message = prompt_feedback.get("block_reason_message") or prompt_feedback.get("blockReasonMessage")
            if block_reason or block_message:
                status = "failed"
                parts = [str(part) for part in (block_reason, block_message) if part]
                error = " ".join(parts) if parts else str(block_reason)

        filtered_reasons: List[str] = []
        for index, item in enumerate(generated_images):
            if not isinstance(item, dict):
                continue
            rai_reason = item.get("rai_filtered_reason") or item.get("raiFilteredReason")
            if rai_reason:
                filtered_reasons.append(str(rai_reason))
                continue
            image_obj = item.get("image") or {}
            meta = _normalise_inline_blob(image_obj)
            key = f"inline_{index}"
            if not meta:
                continue
            data_uri = meta.get("data_uri") or meta.get("data_url")
            inline_entry = {
                "key": key,
                "kind": "image",
                "mime": meta.get("mime"),
                "bytes": meta.get("bytes"),
                "data_uri": data_uri,
                "data_url": data_uri,
            }
            inline_assets.append(inline_entry)
            if data_uri:
                assets[key] = data_uri
            asset_meta[key] = {
                "inline": True,
                "kind": inline_entry.get("kind"),
                "mime": inline_entry.get("mime"),
                "bytes": inline_entry.get("bytes"),
            }

        if not inline_assets and filtered_reasons and not assets:
            status = "failed"
            error = "; ".join(filtered_reasons)
        if status == "completed" and not assets:
            status = "failed"
            if not error:
                error = "Модель не вернула результат"
        return status, error, assets, inline_assets, asset_meta


class GeminiTextClient(GeminiGenerativeClient):
    """Generate text content using Gemini."""

    def __init__(self, *, config: Config) -> None:
        super().__init__(
            config=config,
            api_key=config.gemini_api_key,
            model=config.gemini_model_text,
            provider_name="gemini-text",
            task="text",
            api_version="v1",
        )


class GeminiImageClient(GeminiGenerativeClient):
    """Generate images using Gemini."""

    def __init__(self, *, config: Config) -> None:
        super().__init__(
            config=config,
            api_key=config.gemini_api_key,
            model=config.gemini_model_image,
            provider_name="gemini-image",
            task="image",
            api_version=None,
        )

__all__ = [
    "GeminiGenerativeClient",
    "GeminiImageClient",
    "GeminiTextClient",
    "_has_generate_content_flag",
    "GeminiImageEmptyError",
]
