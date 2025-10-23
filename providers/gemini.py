"""Clients for Google Gemini generative models."""
from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from google.genai import Client as _GenAIClient, errors as _genai_errors, types as _genai_types
from google.genai._api_client import HttpOptions as _HttpOptions

from config import Config

from .base import BaseProviderClient, ProviderAPIError, ProviderJobStatus, ProviderJobSubmission

log = logging.getLogger(__name__)


def _default_safety_settings() -> List[_genai_types.SafetySetting]:
    """Return a defensive Gemini safety configuration."""

    threshold = "BLOCK_MEDIUM_AND_ABOVE"
    categories = [
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_CIVIC_INTEGRITY",
    ]
    return [
        _genai_types.SafetySetting(category=category, threshold=threshold)
        for category in categories
    ]


def _build_http_options(config: Config) -> _HttpOptions:
    timeout: Optional[float | Tuple[float, float]] = None
    if config.request_connect_timeout or config.request_read_timeout:
        timeout = (
            float(config.request_connect_timeout),
            float(config.request_read_timeout),
        )
    elif config.request_timeout:
        timeout = float(config.request_timeout)
    return _HttpOptions(timeout=timeout)


def _normalise_inline_blob(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert inline_data payload to a normalised metadata structure."""

    if not isinstance(data, dict):
        return None
    mime = (
        data.get("mime_type")
        or data.get("mimeType")
        or data.get("mime")
        or "application/octet-stream"
    )
    raw = data.get("data")
    encoded: Optional[str] = None
    binary: Optional[bytes] = None
    if isinstance(raw, (bytes, bytearray)):
        binary = bytes(raw)
    elif isinstance(raw, str):
        stripped = raw.strip()
        try:
            binary = base64.b64decode(stripped, validate=True)
        except (binascii.Error, ValueError):
            binary = stripped.encode("utf-8")
    if not binary:
        return None
    encoded = base64.b64encode(binary).decode("ascii")
    meta: Dict[str, Any] = {
        "inline": True,
        "kind": "inline",
        "mime": mime,
        "bytes": len(binary),
        "base64": encoded,
        "data_url": f"data:{mime};base64,{encoded}",
    }
    return meta


class GeminiGenerativeClient(BaseProviderClient):
    """Base client for synchronous Gemini generate_content calls."""

    def __init__(
        self,
        *,
        config: Config,
        api_key: str,
        model: str,
        provider_name: str,
        safety_settings: Optional[List[_genai_types.SafetySetting]] = None,
    ) -> None:
        super().__init__(
            config=config,
            base_url="",
            api_key=api_key,
            provider_name=provider_name,
        )
        http_options = _build_http_options(config)
        self._client = _GenAIClient(api_key=api_key, http_options=http_options)
        self._model = model
        self._safety_settings = safety_settings or _default_safety_settings()
        self._pending: Dict[str, Tuple[Dict[str, Any], int, int, Dict[str, Any]]] = {}

    def _build_contents(
        self,
        *,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = [{"text": prompt}]
        if settings:
            reference = settings.get("reference_inline_data")
            if isinstance(reference, dict) and reference.get("data"):
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
        kwargs: Dict[str, Any] = {"safetySettings": list(self._safety_settings)}
        if not settings:
            return _genai_types.GenerateContentConfig(**kwargs)
        mapping = {
            "temperature": "temperature",
            "top_p": "topP",
            "topP": "topP",
            "top_k": "topK",
            "topK": "topK",
            "max_output_tokens": "maxOutputTokens",
            "maxOutputTokens": "maxOutputTokens",
            "candidate_count": "candidateCount",
            "candidateCount": "candidateCount",
        }
        for key, target in mapping.items():
            if key in settings and settings[key] is not None:
                kwargs[target] = settings[key]
        return _genai_types.GenerateContentConfig(**kwargs)

    def _map_error(
        self,
        exc: Exception,
        *,
        duration_ms: int,
    ) -> ProviderAPIError:
        if isinstance(exc, _genai_errors.APIError):
            status_code = int(getattr(exc, "code", 0) or 0)
            message = getattr(exc, "message", "Gemini API error") or "Gemini API error"
            retryable = status_code in {408, 429, 500, 502, 503, 504}
            return ProviderAPIError(
                provider=self.provider_name,
                status_code=status_code,
                message=message,
                error_type="api",
                error_code=str(getattr(exc, "status", "")) or None,
                provider_message=str(getattr(exc, "details", message)),
                retryable=retryable,
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
        body_contents = payload.get("contents") if isinstance(payload, dict) else None
        body_config = payload.get("config") if isinstance(payload, dict) else None
        if body_contents is None:
            body_contents = self._build_contents(prompt=prompt, settings=settings)
        if body_config is None:
            body_config = self._build_generation_config(settings)
        attempt = 0
        delay = self._config.retry_backoff
        last_error: Optional[ProviderAPIError] = None
        while attempt <= self._config.request_retries:
            start = time.monotonic()
            try:
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self._model,
                    contents=body_contents,
                    config=body_config,
                )
                duration_ms = int((time.monotonic() - start) * 1000)
                payload_json = response.model_dump(mode="json")
                job_id = idempotency_key or str(uuid4())
                meta = {
                    "prompt": prompt,
                    "settings": settings or {},
                    "payload": payload_json,
                    "assets_meta": {},
                }
                self._pending[job_id] = (payload_json, 200, duration_ms, meta)
                size = (settings or {}).get("size")
                log.info(
                    "gemini.call corr_id=%s provider=%s model=%s size=%s status=%s latency_ms=%s",
                    idempotency_key or job_id,
                    self.provider_name,
                    self._model,
                    size or "",
                    200,
                    duration_ms,
                )
                return ProviderJobSubmission(
                    job_id=job_id,
                    status_code=200,
                    duration_ms=duration_ms,
                    data=payload_json,
                )
            except Exception as exc:  # pragma: no cover - network guard
                duration_ms = int((time.monotonic() - start) * 1000)
                provider_error = self._map_error(exc, duration_ms=duration_ms)
                last_error = provider_error
                if not provider_error.retryable or attempt == self._config.request_retries:
                    raise provider_error
                attempt += 1
                log.warning(
                    "Retrying Gemini request provider=%s model=%s attempt=%s/%s error=%s",
                    self.provider_name,
                    self._model,
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
        status, error, assets, inline_assets = self._extract_result(payload)
        if meta.get("assets_meta") is None:
            meta["assets_meta"] = {}
        for idx, asset in enumerate(inline_assets):
            meta["assets_meta"][f"inline_{idx}"] = asset
        if status != "completed" and not error:
            error = "Модель не вернула результат"
        return ProviderJobStatus(
            job_id=job_id,
            status=status,
            error=error,
            assets=assets,
            data={
                "response": payload,
                "meta": meta,
                "inline_assets": inline_assets,
            },
            status_code=status_code,
            duration_ms=duration_ms,
        )

    def _extract_result(
        self, payload: Dict[str, Any]
    ) -> Tuple[str, Optional[str], Dict[str, str], List[Dict[str, Any]]]:
        if not isinstance(payload, dict):
            return "failed", "Invalid response", {}, []
        assets: Dict[str, str] = {}
        inline_assets: List[Dict[str, Any]] = []
        error: Optional[str] = None
        status = "completed"

        prompt_feedback = payload.get("promptFeedback") or payload.get("prompt_feedback")
        if isinstance(prompt_feedback, dict):
            block_reason = prompt_feedback.get("blockReason") or prompt_feedback.get("block_reason")
            if block_reason:
                status = "failed"
                error = str(block_reason)

        candidates = payload.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                finish_reason = candidate.get("finishReason") or candidate.get("finish_reason")
                if finish_reason and not error and str(finish_reason).upper() in {"SAFETY", "STOP"}:
                    status = "failed"
                    error = str(finish_reason)
                content = candidate.get("content")
                parts: List[Any] = []
                if isinstance(content, dict):
                    parts = content.get("parts") or []
                elif isinstance(content, list):
                    parts = content
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    inline_data = part.get("inline_data") or part.get("inlineData")
                    text_value = part.get("text")
                    if inline_data:
                        meta = _normalise_inline_blob(inline_data)
                        if meta:
                            key = f"inline_{len(inline_assets)}"
                            inline_assets.append(meta)
                            data_url = meta.get("data_url")
                            if data_url:
                                assets[key] = data_url
                    elif isinstance(text_value, str) and text_value.strip():
                        key = f"text_{len(assets)}"
                        assets[key] = text_value

        if status == "completed" and not assets:
            status = "failed"
            if not error:
                error = "Модель не вернула результат"
        return status, error, assets, inline_assets


class GeminiTextClient(GeminiGenerativeClient):
    """Generate text content using Gemini."""

    def __init__(self, *, config: Config) -> None:
        super().__init__(
            config=config,
            api_key=config.gemini_api_key,
            model=config.gemini_model_text,
            provider_name="gemini-text",
        )


class GeminiImageClient(GeminiGenerativeClient):
    """Generate images using Gemini."""

    def __init__(self, *, config: Config) -> None:
        super().__init__(
            config=config,
            api_key=config.gemini_api_key,
            model=config.gemini_model_image,
            provider_name="gemini-image",
        )


__all__ = [
    "GeminiGenerativeClient",
    "GeminiImageClient",
    "GeminiTextClient",
]
