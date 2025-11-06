"""Clients for Google Gemini generative models."""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import re
import time
from pathlib import Path
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from uuid import uuid4

from google import genai
from google.genai import errors as genai_errors, types

from config import Config, SafetyCfg
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

from .api_error_handler import ApiErrorHandler
from .base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
    _sanitize_headers,
)

log = logging.getLogger(__name__)


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
    sdk_response = getattr(response, "sdk_http_response", None)
    headers = getattr(sdk_response, "headers", None)
    if headers is None:
        return {}
    if isinstance(headers, Mapping):
        raw = {str(key): headers[key] for key in headers}
    elif hasattr(headers, "items"):
        try:
            raw = {str(key): value for key, value in headers.items()}
        except Exception:  # pragma: no cover - defensive
            raw = {}
    else:
        raw = {}
    return _sanitize_headers({k: str(v) for k, v in raw.items()})


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
    if isinstance(payload_json, Mapping):
        for key in ("status", "finish_reason", "finishReason"):
            value = payload_json.get(key)
            if value and not finish_code:
                finish_code = str(value).upper()
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


class GeminiGenerativeClient(BaseProviderClient):
    """Base client for synchronous Gemini API calls."""

    def __init__(
        self,
        *,
        config: Config,
        api_key: str,
        model: str,
        provider_name: str,
        task: str,
        api_version: str,
        safety_settings: Optional[List[types.SafetySetting]] = None,
    ) -> None:
        base_url = self._resolve_base_url(config)
        super().__init__(
            config=config,
            base_url=base_url,
            api_key=api_key,
            provider_name=provider_name,
        )
        self._task = task
        if self._task in ("image", "video"):
            resolved_version = "v1beta"
            http_opts = types.HttpOptions(api_version="v1beta")
        else:
            resolved_version = "v1"
            http_opts = types.HttpOptions(api_version="v1")

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
        if (
            decision.task == "image"
            and isinstance(decision.model, str)
            and decision.model.lower() == "gemini-2.5-flash-image"
            and not force_mode
        ):
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
        generator = getattr(decision.client.models, method)
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
                        "Для изображений и видео используй v1beta; для текста — v1."
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
        fallback_to_content = False
        while attempt <= self._config.request_retries:
            model_override = request_settings.get("model") or self._model
            try:
                decision = self._router.route(
                    task=self._task,
                    model=model_override,
                    prompt=prompt,
                    assets=request_settings,
                )
            except GeminiRoutingError as exc:
                raise ProviderAPIError(
                    provider=self.provider_name,
                    status_code=400,
                    message=str(exc),
                    error_type="routing",
                    retryable=False,
                ) from exc

            generator, request_kwargs, request_meta = self._prepare_generate_call(
                decision=decision,
                prompt=decision.prompt,
                settings=request_settings,
                payload=payload_dict,
                force_mode=force_mode,
            )
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
            log.info(
                "gemini.generate start model=%s method=%s version=%s env=%s key_mask=%s corr_id=%s",
                decision.model,
                actual_method,
                decision.api_version,
                self._environment,
                self._key_mask or "",
                idempotency_key or "",
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
                    except GeminiImageEmptyError as image_error:
                        finish_code, provider_message, safety_block = _analyse_empty_image_response(
                            payload_json if isinstance(payload_json, Mapping) else {},
                            response=response,
                            error=image_error,
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
                        if safety_block or finish_code_upper in _IMAGE_BLOCK_REASONS:
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
                        size = request_settings.get("size")
                        corr_id = idempotency_key or response_id or "inline"
                        self._log_generation_feedback(
                            status_code=200,
                            provider_message=None,
                            safety_feedback=safety_feedback,
                            corr_id=corr_id,
                            payload=payload_json,
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
                        return ProviderJobSubmission(
                            job_id=None,
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
                size = request_settings.get("size")
                self._log_generation_feedback(
                    status_code=200,
                    provider_message=None,
                    safety_feedback=safety_feedback,
                    corr_id=idempotency_key or job_id,
                    payload=payload_json,
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
                        "gemini.image.fallback method=generate_images status=%s switching_to=generate_content",
                        provider_error.status_code,
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
                await asyncio.sleep(delay)
                delay *= self._config.retry_backoff
                continue
            except Exception as exc:  # pragma: no cover - network guard
                duration_ms = int((time.monotonic() - start) * 1000)
                provider_error, error_code, error_message, raw_body = self._map_error(
                    exc,
                    duration_ms=duration_ms,
                    decision=decision,
                )
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
                stored_payload, _, _, stored_meta = pending_record
                if isinstance(stored_payload, dict):
                    payload_json = stored_payload
                if isinstance(stored_meta, dict):
                    meta = stored_meta
            elif isinstance(submission.data, dict):
                payload_json = submission.data

            info = extract_block_info(payload_json)
            blocked = bool(info.get("reason"))
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
                        if finish_text in {"STOP", "SAFETY"} and not blocked:
                            info["reason"] = finish_text
                            blocked = True
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
        if record is None:
            record = self._load_pending_record(job_id)
        if record is None:
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
            api_version="v1beta",
        )

__all__ = [
    "GeminiGenerativeClient",
    "GeminiImageClient",
    "GeminiTextClient",
    "_has_generate_content_flag",
    "GeminiImageEmptyError",
]
