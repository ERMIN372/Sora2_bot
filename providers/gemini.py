"""Clients for Google Gemini generative models."""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import random
import re
import time
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from uuid import uuid4

from google.genai import errors as _genai_errors, types as _genai_types

from config import Config
from services.gemini_key import ensure_gemini_key_logged
from services.gemini_router import (
    GeminiRoutingError,
    RouteDecision,
    get_gemini_router,
)

from .base import BaseProviderClient, ProviderAPIError, ProviderJobStatus, ProviderJobSubmission

log = logging.getLogger(__name__)


_SAFETY_CATEGORIES: Tuple[_genai_types.HarmCategory, ...] = (
    _genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
    _genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    _genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    _genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    _genai_types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
)

_SAFETY_DEFAULT_THRESHOLD = _genai_types.HarmBlockThreshold.BLOCK_NONE
_SAFETY_FALLBACK_THRESHOLD = _genai_types.HarmBlockThreshold.BLOCK_ONLY_HIGH


def _enum_code(value: Any) -> str:
    """Return a stable string representation for Gemini enums."""

    if value is None:
        return ""
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def _default_safety_settings() -> List[_genai_types.SafetySetting]:
    """Return a permissive Gemini safety configuration for text calls."""

    return [
        _genai_types.SafetySetting(
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
        safety_settings: Optional[List[_genai_types.SafetySetting]] = None,
    ) -> None:
        super().__init__(
            config=config,
            base_url="",
            api_key=api_key,
            provider_name=provider_name,
        )
        self._api_key = api_key
        parsed_safety = _convert_setting_list(safety_settings, _genai_types.SafetySetting)
        self._safety_settings = parsed_safety or _default_safety_settings()
        self._model = model
        self._task = task
        self._api_version = api_version
        self._router = get_gemini_router(config)
        self._pending: Dict[str, Tuple[Dict[str, Any], int, int, Dict[str, Any]]] = {}
        self._environment = (config.environment or "dev").lower()
        self._key_mask = ensure_gemini_key_logged(
            config,
            context="gemini_generative_client",
            process_id=f"pid={os.getpid()}",  # pragma: no cover - runtime value
            logger=log,
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
        self, settings: Optional[Dict[str, Any]]
    ) -> _genai_types.GenerateContentConfig:
        kwargs: Dict[str, Any] = {}
        safety_settings: Optional[List[_genai_types.SafetySetting]] = None
        if self._task == "text":
            safety_settings = list(self._safety_settings)
        if not settings:
            if safety_settings is not None:
                kwargs["safety_settings"] = safety_settings
            return _genai_types.GenerateContentConfig(**kwargs)

        allowed_fields = set(_genai_types.GenerateContentConfig.model_fields.keys())
        skip_fields = {"http_options", "should_return_http_response"}
        for raw_key, value in settings.items():
            if value is None:
                continue
            key = _normalise_setting_key(raw_key)
            if key in {"reference_inline_data", "image_file_id", "size"}:
                continue
            if key in skip_fields or key not in allowed_fields:
                continue
            if key == "safety_settings":
                parsed = _convert_setting_list(value, _genai_types.SafetySetting)
                if parsed is not None:
                    safety_settings = parsed if self._task == "text" else None
                continue
            if key == "tools":
                parsed = _convert_setting_list(value, _genai_types.Tool)
                if parsed is not None:
                    kwargs["tools"] = parsed
                continue
            if key == "tool_config":
                parsed = _maybe_model_validate(value, _genai_types.ToolConfig)
                if parsed is not None:
                    kwargs["tool_config"] = parsed
                continue
            if key == "response_schema":
                parsed = _maybe_model_validate(value, _genai_types.Schema)
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

        if safety_settings is not None:
            kwargs["safety_settings"] = safety_settings
        return _genai_types.GenerateContentConfig(**kwargs)

    def _prepare_generate_call(
        self,
        *,
        decision: "RouteDecision",
        prompt: str,
        settings: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        contents = payload.get("contents")
        config = payload.get("config")
        if contents is None:
            contents = self._build_contents(prompt=prompt, settings=settings)
        if not isinstance(config, _genai_types.GenerateContentConfig):
            config = self._build_generation_config(settings)
        request_kwargs = {
            "model": decision.model,
            "contents": contents,
            "config": config,
        }
        request_meta = {
            "mode": "generate_content",
            "api_version": decision.api_version,
            "config": config.model_dump(mode="json") if hasattr(config, "model_dump") else config,
        }
        generator = getattr(decision.client.models, decision.method)
        return generator, request_kwargs, request_meta

    def _map_error(
        self,
        exc: Exception,
        *,
        duration_ms: int,
        decision: Optional[RouteDecision] = None,
    ) -> ProviderAPIError:
        if isinstance(exc, _genai_errors.APIError):
            status_code = int(getattr(exc, "code", 0) or 0)
            message = getattr(exc, "message", "Gemini API error") or "Gemini API error"
            retryable = status_code in {408, 409, 429, 500, 502, 503, 504}
            provider_message = getattr(exc, "details", None)
            if not provider_message and hasattr(exc, "response"):
                provider_message = getattr(exc.response, "text", None)
            status_name = str(getattr(exc, "status", "") or "")
            provider_text = str(provider_message or "")
            error_type = "api"
            if status_code in {400, 404}:
                message = (
                    "Эта модель не поддерживает данный метод. Поменяй модель или метод."
                )
                if decision and decision.method in {"generate_images", "generate_videos"}:
                    message = (
                        "Для изображений и видео используй v1beta; для текста — v1."
                    )
                if (
                    status_code == 404
                    and decision
                    and decision.method == "generate_images"
                ):
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
            return ProviderAPIError(
                provider=self.provider_name,
                status_code=status_code,
                message=message,
                error_type=error_type,
                error_code=str(getattr(exc, "status", "")) or None,
                provider_message=str(provider_message or message),
                retryable=retryable,
                duration_ms=duration_ms,
            )
        if isinstance(exc, ValueError):
            return ProviderAPIError(
                provider=self.provider_name,
                status_code=400,
                message=str(exc) or "Invalid Gemini request",
                error_type="validation",
                provider_message=str(exc),
                retryable=False,
                duration_ms=duration_ms,
            )
        return ProviderAPIError(
            provider=self.provider_name,
            status_code=0,
            message=str(exc) or "Gemini request failed",
            error_type="network",
            provider_message=str(exc),
            retryable=True,
            duration_ms=duration_ms,
        )

    def _extract_threshold_error_category(self, exc: Exception) -> Optional[str]:
        if not isinstance(exc, _genai_errors.APIError):
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
    ) -> Optional[_genai_types.GenerateContentConfig]:
        if not isinstance(config, _genai_types.GenerateContentConfig):
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
            return _genai_types.GenerateContentConfig.model_validate(dump)
        except Exception:  # pragma: no cover - defensive revalidation
            log.warning(
                "Failed to downgrade safety config for category=%s", category, exc_info=True
            )
            return None

    async def enqueue_job(
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

        attempt = 0
        delay = self._config.retry_backoff
        last_error: Optional[ProviderAPIError] = None
        downgraded_categories: Set[str] = set()
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
            )
            request_meta.update(
                {
                    "mode": decision.method,
                    "method": decision.method,
                    "model": decision.model,
                    "api_version": decision.api_version,
                }
            )
            if decision.rewrite_notes:
                request_meta["prompt_notes"] = decision.rewrite_notes
            request_meta.setdefault("prompt", decision.prompt)
            start = time.monotonic()
            log.info(
                "gemini.generate start model=%s method=%s version=%s env=%s key_mask=%s corr_id=%s",
                decision.model,
                decision.method,
                decision.api_version,
                self._environment,
                self._key_mask or "",
                idempotency_key or "",
            )
            try:
                response = await asyncio.to_thread(generator, **request_kwargs)
                duration_ms = int((time.monotonic() - start) * 1000)
                if hasattr(response, "model_dump"):
                    payload_json = response.model_dump(mode="json")
                elif isinstance(response, dict):
                    payload_json = response
                else:
                    payload_json = {"response": response}
                job_id = idempotency_key or str(uuid4())
                meta = {
                    "prompt": decision.prompt,
                    "settings": request_settings or {},
                    "payload": payload_json,
                    "assets_meta": {},
                    "request": request_meta,
                    "request_mode": decision.method,
                    "key_mask": self._key_mask,
                }
                self._pending[job_id] = (payload_json, 200, duration_ms, meta)
                size = request_settings.get("size")
                log.info(
                    "gemini.call corr_id=%s provider=%s model=%s method=%s version=%s size=%s status=%s latency_ms=%s",
                    idempotency_key or job_id,
                    self.provider_name,
                    decision.model,
                    decision.method,
                    decision.api_version,
                    size or "",
                    200,
                    duration_ms,
                )
                log.info(
                    "gemini.generate done model=%s method=%s version=%s env=%s key_mask=%s corr_id=%s latency_ms=%s",
                    decision.model,
                    decision.method,
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
            except _genai_errors.APIError as exc:
                duration_ms = int((time.monotonic() - start) * 1000)
                category = self._extract_threshold_error_category(exc)
                if (
                    category
                    and category.upper() not in downgraded_categories
                    and decision.method == "generate_content"
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
                provider_error = self._map_error(
                    exc,
                    duration_ms=duration_ms,
                    decision=decision,
                )
                log.warning(
                    "gemini.generate error model=%s method=%s version=%s env=%s key_mask=%s corr_id=%s status=%s latency_ms=%s",
                    decision.model,
                    decision.method,
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
                    decision.method,
                    attempt + 1,
                    self._config.request_retries + 1,
                    provider_error,
                )
                await asyncio.sleep(delay)
                delay *= self._config.retry_backoff
                continue
            except Exception as exc:  # pragma: no cover - network guard
                duration_ms = int((time.monotonic() - start) * 1000)
                provider_error = self._map_error(
                    exc,
                    duration_ms=duration_ms,
                    decision=decision,
                )
                log.warning(
                    "gemini.generate error model=%s method=%s version=%s env=%s key_mask=%s corr_id=%s status=%s latency_ms=%s",
                    decision.model,
                    decision.method,
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
                    decision.method,
                    attempt + 1,
                    self._config.request_retries + 1,
                    provider_error,
                )
                await asyncio.sleep(delay)
                delay *= self._config.retry_backoff
        assert last_error is not None  # pragma: no cover
        raise last_error

    async def get_job_status(self, job_id: str) -> ProviderJobStatus:
        record = self._pending.pop(job_id, None)
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

    async def enqueue_job(
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
        corr_id = idempotency_key or ""
        max_attempts = 4
        last_error: Optional[ProviderAPIError] = None

        for attempt in range(1, max_attempts + 1):
            try:
                decision = self._router.route(
                    task=self._task,
                    model=request_settings.get("model") or self._model,
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

            generator = getattr(decision.client.models, "generate_images", None)
            if generator is None:
                raise ProviderAPIError(
                    provider=self.provider_name,
                    status_code=500,
                    message="Gemini SDK is missing generate_images",
                    error_type="configuration",
                    retryable=False,
                )

            request_kwargs = {"model": decision.model, "prompt": decision.prompt}
            request_meta = {
                "mode": "generate_images",
                "method": "generate_images",
                "model": decision.model,
                "api_version": decision.api_version,
                "prompt": decision.prompt,
            }
            log.info(
                "gemini.generateImages start corr_id=%s model=%s version=%s env=%s key_mask=%s",
                corr_id or "",
                decision.model,
                decision.api_version,
                self._environment,
                self._key_mask or "",
            )
            start = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(generator, **request_kwargs),
                    timeout=60.0,
                )
            except asyncio.TimeoutError as exc:
                duration_ms = int((time.monotonic() - start) * 1000)
                provider_error = ProviderAPIError(
                    provider=self.provider_name,
                    status_code=504,
                    message="Gemini image request timed out",
                    error_type="timeout",
                    retryable=True,
                    duration_ms=duration_ms,
                )
                last_error = provider_error
                log.warning(
                    "gemini.generateImages timeout corr_id=%s model=%s version=%s latency_ms=%s",
                    corr_id or "",
                    decision.model,
                    decision.api_version,
                    duration_ms,
                )
                if attempt == max_attempts:
                    provider_error.error_type = "provider_unavailable"
                    raise provider_error from exc
                log.warning(
                    "Retrying Gemini image request corr_id=%s model=%s attempt=%s/%s reason=timeout",
                    corr_id or "",
                    decision.model,
                    attempt + 1,
                    max_attempts,
                )
                await self._sleep_with_backoff(attempt)
                continue
            except _genai_errors.APIError as exc:
                duration_ms = int((time.monotonic() - start) * 1000)
                provider_error = self._map_error(
                    exc,
                    duration_ms=duration_ms,
                    decision=decision,
                )
                last_error = provider_error
                log.warning(
                    "gemini.generateImages error corr_id=%s model=%s version=%s status=%s latency_ms=%s",
                    corr_id or "",
                    decision.model,
                    decision.api_version,
                    provider_error.status_code,
                    duration_ms,
                )
                if (
                    provider_error.status_code in {429, 500, 503}
                    and attempt < max_attempts
                ):
                    log.warning(
                        "Retrying Gemini image request corr_id=%s model=%s attempt=%s/%s status=%s",
                        corr_id or "",
                        decision.model,
                        attempt + 1,
                        max_attempts,
                        provider_error.status_code,
                    )
                    await self._sleep_with_backoff(attempt)
                    continue
                if provider_error.status_code in {429, 500, 503}:
                    provider_error.error_type = provider_error.error_type or "provider_unavailable"
                raise provider_error from exc
            except Exception as exc:
                duration_ms = int((time.monotonic() - start) * 1000)
                provider_error = self._map_error(
                    exc,
                    duration_ms=duration_ms,
                    decision=decision,
                )
                last_error = provider_error
                log.warning(
                    "gemini.generateImages error corr_id=%s model=%s version=%s status=%s latency_ms=%s",
                    corr_id or "",
                    decision.model,
                    decision.api_version,
                    provider_error.status_code,
                    duration_ms,
                )
                if (
                    provider_error.status_code in {429, 500, 503}
                    and attempt < max_attempts
                ):
                    log.warning(
                        "Retrying Gemini image request corr_id=%s model=%s attempt=%s/%s status=%s",
                        corr_id or "",
                        decision.model,
                        attempt + 1,
                        max_attempts,
                        provider_error.status_code,
                    )
                    await self._sleep_with_backoff(attempt)
                    continue
                if provider_error.status_code in {429, 500, 503}:
                    provider_error.error_type = provider_error.error_type or "provider_unavailable"
                raise provider_error

            duration_ms = int((time.monotonic() - start) * 1000)
            payload_json, inline_assets, asset_meta = self._serialise_image_response(
                response
            )
            job_id = idempotency_key or str(uuid4())
            meta = {
                "prompt": decision.prompt,
                "settings": request_settings or {},
                "payload": payload_json.get("response"),
                "assets_meta": {key: dict(value) for key, value in asset_meta.items()},
                "request": request_meta,
                "request_mode": "generate_images",
                "key_mask": self._key_mask,
            }
            if inline_assets:
                meta.setdefault("inline_assets", inline_assets)

            self._pending[job_id] = (payload_json, 200, duration_ms, meta)
            log.info(
                "gemini.generateImages success corr_id=%s job_id=%s model=%s status=%s duration_ms=%s",
                corr_id or job_id,
                job_id,
                decision.model,
                200,
                duration_ms,
            )
            return ProviderJobSubmission(
                job_id=job_id,
                status_code=200,
                duration_ms=duration_ms,
                data=payload_json,
            )

        if last_error is not None:
            if last_error.status_code in {429, 500, 503} and not last_error.error_type:
                last_error.error_type = "provider_unavailable"
            raise last_error
        raise ProviderAPIError(
            provider=self.provider_name,
            status_code=500,
            message="Gemini image request failed",
            error_type="provider_unavailable",
        )

    async def _sleep_with_backoff(self, attempt: int) -> None:
        base_delay = 1.0 * (2 ** max(attempt - 1, 0))
        jitter = random.uniform(0.5, 1.5)
        await asyncio.sleep(min(base_delay * jitter, 15.0))

    def _serialise_image_response(
        self, response: Any
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        if hasattr(response, "model_dump"):
            try:
                base_payload = response.model_dump(mode="json")
            except Exception:  # pragma: no cover - defensive serialisation
                base_payload = {"raw": str(response)}
        elif isinstance(response, dict):
            base_payload = dict(response)
        else:
            base_payload = {"raw": str(response)}

        inline_assets, asset_meta = self._extract_image_inline_assets(response)
        payload = {"response": base_payload}
        if inline_assets:
            payload["inline_assets"] = [dict(item) for item in inline_assets]
        if asset_meta:
            payload["assets_meta"] = {key: dict(value) for key, value in asset_meta.items()}
        return payload, inline_assets, asset_meta

    def _extract_image_inline_assets(
        self, response: Any
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        inline_assets: List[Dict[str, Any]] = []
        asset_meta: Dict[str, Dict[str, Any]] = {}

        candidates: Iterable[Any]
        if hasattr(response, "images"):
            candidates = getattr(response, "images") or []
        elif isinstance(response, dict):
            candidates = response.get("images") or []
        else:
            candidates = []

        for index, item in enumerate(candidates):
            data = getattr(item, "data", None)
            if data is None:
                data = getattr(item, "image_bytes", None)
            if data is None and isinstance(item, dict):
                data = item.get("data") or item.get("image_bytes")
            if isinstance(data, str):
                try:
                    decoded = base64.b64decode(data)
                except (binascii.Error, ValueError):
                    decoded = b""
                data_bytes = decoded
            else:
                data_bytes = bytes(data) if isinstance(data, (bytes, bytearray)) else b""
            if not data_bytes:
                continue
            mime = (
                getattr(item, "mime_type", None)
                or getattr(item, "mime", None)
                or (item.get("mime_type") if isinstance(item, dict) else None)
                or "image/png"
            )
            base64_data = base64.b64encode(data_bytes).decode("ascii")
            data_uri = f"data:{mime};base64,{base64_data}"
            key = f"inline_{index}"
            inline_entry = {
                "key": key,
                "kind": "image",
                "mime": mime,
                "bytes": len(data_bytes),
                "data_uri": data_uri,
                "data_url": data_uri,
            }
            uri_value = getattr(item, "uri", None) or (
                item.get("uri") if isinstance(item, dict) else None
            )
            if uri_value:
                inline_entry["uri"] = uri_value
            inline_assets.append(inline_entry)
            asset_meta[key] = {
                "inline": True,
                "kind": "image",
                "mime": mime,
                "bytes": len(data_bytes),
            }

        return inline_assets, asset_meta

    def _build_images_config(
        self,
        settings: Dict[str, Any],
        existing: Optional[Any] = None,
    ) -> _genai_types.GenerateImagesConfig:
        kwargs: Dict[str, Any]
        if isinstance(existing, _genai_types.GenerateImagesConfig):
            kwargs = existing.model_dump(mode="json", exclude_none=True)
        elif isinstance(existing, dict):
            kwargs = dict(existing)
        else:
            kwargs = {}

        allowed_fields = set(_genai_types.GenerateImagesConfig.model_fields.keys())
        skip_fields = {"http_options", "safety_settings"}
        for raw_key, value in settings.items():
            if value is None:
                continue
            key = _normalise_setting_key(raw_key)
            if key in {"reference_inline_data", "image_file_id"}:
                continue
            if key in skip_fields or key not in allowed_fields:
                continue
            if key == "size":
                continue
            kwargs[key] = value

        if "aspect_ratio" not in kwargs:
            aspect = settings.get("aspect_ratio")
            if not aspect:
                inferred = _infer_aspect_ratio(str(settings.get("size"))) if settings.get("size") else None
                if inferred:
                    aspect = inferred
            if aspect:
                kwargs["aspect_ratio"] = aspect

        labels = kwargs.get("labels")
        if isinstance(labels, dict):
            kwargs["labels"] = {str(k): str(v) for k, v in labels.items()}

        references: List[_genai_types.Image] = []
        raw_reference = settings.get("reference_inline_data")
        for entry in _ensure_iterable(raw_reference):
            if not isinstance(entry, dict):
                continue
            meta = _normalise_inline_blob(entry)
            if not meta or not meta.get("base64"):
                continue
            try:
                data = base64.b64decode(meta["base64"], validate=False)
            except (ValueError, binascii.Error):  # pragma: no cover - defensive decode
                continue
            mime = meta.get("mime") or entry.get("mime_type") or "image/jpeg"
            try:
                references.append(
                    _genai_types.Image(image_bytes=data, mime_type=mime)
                )
            except Exception:  # pragma: no cover - defensive instantiation
                log.debug("Failed to encode reference image", exc_info=True)
        if references:
            kwargs["reference_images"] = references

        kwargs.setdefault("mime_type", "image/png")
        return _genai_types.GenerateImagesConfig(**kwargs)

    def _prepare_generate_call(
        self,
        *,
        decision: "RouteDecision",
        prompt: str,
        settings: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        config = payload.get("config")
        images_config = self._build_images_config(settings, existing=config)
        request_kwargs = {
            "model": decision.model,
            "prompt": prompt,
            "config": images_config,
        }
        request_meta = {
            "mode": "generate_images",
            "api_version": decision.api_version,
            "config": images_config.model_dump(mode="json") if hasattr(images_config, "model_dump") else images_config,
        }
        generator = getattr(decision.client.models, decision.method)
        return generator, request_kwargs, request_meta

    def _extract_result(
        self, payload: Dict[str, Any]
    ) -> Tuple[str, Optional[str], Dict[str, str], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        inline_inline = payload.get("inline_assets")
        if isinstance(inline_inline, list) and inline_inline:
            assets: Dict[str, str] = {}
            inline_assets: List[Dict[str, Any]] = []
            asset_meta: Dict[str, Dict[str, Any]] = {}
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
        generated = payload.get("generated_images") or payload.get("generatedImages")
        if isinstance(generated, list):
            return self._extract_images(payload)
        return super()._extract_result(payload)

    def _extract_images(
        self, payload: Dict[str, Any]
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

        generated_images = payload.get("generated_images") or payload.get("generatedImages") or []
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
            if meta:
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

__all__ = [
    "GeminiGenerativeClient",
    "GeminiImageClient",
    "GeminiTextClient",
]
