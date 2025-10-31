"""OpenAI Sora video generation client."""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, Iterable, Optional, Tuple

from config import Config

from .base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
    _serialise_for_log,
)
from services.payload_sanitize import (
    log_removed_keys,
    normalize_sora_payload,
    sanitize_payload,
)
from services.payload_whitelists import SORA_VIDEO_ALLOWED

log = logging.getLogger(__name__)


AVAILABLE_SORA_MODELS = ["sora-2", "sora-2-fast", "sora-2-pro"]


_SIZE_PATTERN = re.compile(r"^(\d+)[xX](\d+)$")
_TERMINAL_STATUSES = {"completed", "succeeded", "failed", "errored", "cancelled", "canceled"}
_VIDEO_MIME_PREFIXES = ("video/", "application/octet-stream")


def _infer_aspect_ratio(size: Optional[str]) -> Optional[str]:
    if not size:
        return None
    match = _SIZE_PATTERN.match(size.strip())
    if not match:
        return None
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    divisor = math.gcd(width, height) or 1
    return f"{width // divisor}:{height // divisor}"


def _iter_nodes(value: Any) -> Iterable[Dict[str, Any]]:
    stack: list[Any] = [value]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            identifier = id(current)
            if identifier in seen:
                continue
            seen.add(identifier)
            yield current
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)


def _extract_url(entry: Dict[str, Any]) -> Optional[str]:
    for key in ("download_url", "url", "video_url", "asset_url"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    for nested_key in ("file", "video", "asset"):
        nested = entry.get(nested_key)
        if isinstance(nested, dict):
            for key in ("download_url", "url"):
                value = nested.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _extract_base64(entry: Dict[str, Any]) -> Optional[str]:
    for key in ("b64_json", "base64", "data", "inline_data"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            payload = value.get("data") or value.get("base64")
            if isinstance(payload, str) and payload:
                return payload
    return None


def _guess_mime(entry: Dict[str, Any]) -> Optional[str]:
    for key in ("mime_type", "content_type", "mime"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    nested = entry.get("file") or entry.get("video")
    if isinstance(nested, dict):
        for key in ("mime_type", "content_type", "mime"):
            value = nested.get(key)
            if isinstance(value, str) and value:
                return value
    return None


class SoraVideoClient(BaseProviderClient):
    """Thin wrapper around OpenAI's Responses API for Sora generation."""

    def __init__(self, *, config: Config) -> None:
        api_key = config.openai_key
        if not api_key:
            raise RuntimeError("Sora API key is not configured; set SORA_API_KEY or OPENAI_API_KEY")
        base_url = (config.openai_api_base or "https://api.openai.com/v1").rstrip("/")
        default_headers = {"OpenAI-Beta": "assistants=v2"}
        super().__init__(
            config=config,
            base_url=base_url,
            api_key=api_key,
            provider_name="sora",
            default_headers=default_headers,
        )
        self._supported_models = set(AVAILABLE_SORA_MODELS)
        configured_model = (config.sora_model_video or "").strip()
        fallback_model = AVAILABLE_SORA_MODELS[0]
        if configured_model and configured_model in self._supported_models:
            self._default_model = configured_model
        else:
            if configured_model:
                log.warning(
                    "Unsupported default Sora model configured model=%s; falling back to '%s'",
                    configured_model,
                    fallback_model,
                )
            self._default_model = fallback_model
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # High level operations
    # ------------------------------------------------------------------

    async def enqueue_job(
        self,
        *,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        **_: Any,
    ) -> ProviderJobSubmission:
        settings = dict(settings or {})
        requested_model = (settings.get("model") or self._default_model or "").strip()
        model_name = requested_model or self._default_model
        if model_name not in self._supported_models:
            available = ", ".join(AVAILABLE_SORA_MODELS)
            log.warning(
                "Unsupported Sora model requested model=%s supported=%s",
                model_name,
                AVAILABLE_SORA_MODELS,
            )
            raise ProviderAPIError(
                provider="sora",
                status_code=400,
                message=(
                    "Выбранная модель Sora не поддерживается, используйте одну из: "
                    f"{available}"
                ),
                error_type="invalid_model",
                error_code="invalid_model",
                provider_message=json.dumps(
                    {
                        "error": {
                            "code": "invalid_model",
                            "message": "Unsupported Sora model",
                        }
                    },
                    ensure_ascii=False,
                ),
            )
        settings["model"] = model_name
        request_payload = self._build_request_payload(
            prompt=prompt,
            settings=settings,
            payload=payload or {},
        )
        endpoint = "/responses"
        url = f"{self._base_url}{endpoint}"
        serialised_payload = _serialise_for_log(request_payload, limit=2048)
        log.debug("Sora request: url=%s, payload=%s", url, serialised_payload)
        log.info(
            "sora.enqueue start endpoint=%s method=%s model=%s payload=%s",
            endpoint,
            "POST",
            model_name,
            _serialise_for_log(request_payload, limit=2048),
        )
        data, status_code, duration_ms = await self._request(
            "POST",
            endpoint,
            json=request_payload,
            idempotency_key=idempotency_key,
        )
        error_code = None
        error_type = None
        error_message = None
        error_payload = data.get("error") if isinstance(data, dict) else None
        if isinstance(error_payload, dict):
            error_code = error_payload.get("code") or error_payload.get("error_code")
            error_type = error_payload.get("type") or error_payload.get("error_type")
            error_message = error_payload.get("message") or error_payload.get("detail")
            log.error(
                "Sora API error in response code=%s message=%s", error_code, error_message
            )
        serialised_response = _serialise_for_log(data, limit=2048)
        log.debug("Sora response: status=%s, body=%s", status_code, serialised_response)
        log.info(
            "sora.enqueue response endpoint=%s status=%s duration_ms=%s error_type=%s error_code=%s error_message=%s json=%s",
            endpoint,
            status_code,
            duration_ms,
            error_type,
            error_code,
            error_message,
            serialised_response,
        )
        job_id = self._resolve_job_id(data)
        if not job_id:
            provider_message = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else str(data)
            raise ProviderAPIError(
                provider="sora",
                status_code=status_code,
                message="Sora API did not return a response id",
                error_type=error_type or "protocol",
                error_code=error_code,
                provider_message=provider_message,
            )
        self._cache[job_id] = data
        return ProviderJobSubmission(
            job_id=job_id,
            status_code=status_code,
            duration_ms=duration_ms,
            data=data,
        )

    async def get_job_status(self, job_id: str) -> ProviderJobStatus:
        cached = self._cache.get(job_id)
        if cached and self._is_terminal(cached):
            data = cached
            status_code = 200
            duration_ms = 0
        else:
            endpoint = f"/responses/{job_id}"
            url = f"{self._base_url}{endpoint}"
            log.debug("Sora request: url=%s, payload=%s", url, "")
            log.info(
                "sora.status start endpoint=%s method=%s",
                endpoint,
                "GET",
            )
            data, status_code, duration_ms = await self._request("GET", endpoint)
            error_type = None
            error_code = None
            error_message = None
            if isinstance(data, dict):
                error_payload = data.get("error")
                if isinstance(error_payload, dict):
                    error_type = error_payload.get("type")
                    error_code = error_payload.get("code")
                    error_message = error_payload.get("message") or error_payload.get("detail")
                    log.error(
                        "Sora API error in response code=%s message=%s",
                        error_code,
                        error_message,
                    )
            serialised_response = _serialise_for_log(data, limit=2048)
            log.debug("Sora response: status=%s, body=%s", status_code, serialised_response)
            log.info(
                "sora.status response endpoint=%s status=%s duration_ms=%s error_type=%s error_code=%s error_message=%s json=%s",
                endpoint,
                status_code,
                duration_ms,
                error_type,
                error_code,
                error_message,
                serialised_response,
            )
            self._cache[job_id] = data
        status = self._extract_status(data)
        error = self._extract_error(data)
        assets = self._extract_assets(data)
        return ProviderJobStatus(
            job_id=job_id,
            status=status,
            assets=assets,
            error=error,
            data=data,
            status_code=status_code,
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_request_payload(
        self,
        *,
        prompt: str,
        settings: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not prompt:
            raise ValueError("prompt must not be empty")
        normalized_payload = normalize_sora_payload(payload or {})
        cleaned_payload = sanitize_payload(normalized_payload, SORA_VIDEO_ALLOWED)
        log_removed_keys("sora-video", normalized_payload, cleaned_payload, logger=log)
        prompt_value = cleaned_payload.get("prompt")
        if isinstance(prompt_value, str) and prompt_value.strip():
            prompt_text = prompt_value.strip()
        else:
            prompt_text = prompt
        model_name = (settings.get("model") or self._default_model or "sora").strip()
        content: list[Dict[str, Any]] = [
            {"type": "input_text", "text": prompt_text},
        ]
        reference_url = settings.get("reference_image_url") or settings.get("reference_url")
        if isinstance(reference_url, str) and reference_url.strip():
            content.append(
                {
                    "type": "input_image",
                    "image_url": {"url": reference_url.strip()},
                }
            )
        reference_file = settings.get("reference_file_id") or settings.get("image_file_id")
        if isinstance(reference_file, str) and reference_file.strip():
            content.append(
                {
                    "type": "input_image",
                    "image_file": {"file_id": reference_file.strip()},
                }
            )
        inline_data = settings.get("reference_inline_data")
        if isinstance(inline_data, dict):
            encoded = inline_data.get("data") or inline_data.get("base64")
            if isinstance(encoded, str) and encoded.strip():
                entry: Dict[str, Any] = {"type": "input_image", "image_base64": encoded.strip()}
                mime = inline_data.get("mime_type") or inline_data.get("mime")
                if isinstance(mime, str) and mime:
                    entry["mime_type"] = mime
                content.append(entry)
        if payload.get("content") and isinstance(payload["content"], list):
            for item in payload["content"]:
                if isinstance(item, dict):
                    content.append(item)
        request: Dict[str, Any] = {
            "model": model_name,
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
        metadata: Dict[str, Any] = {}
        aspect_ratio = (
            settings.get("aspect_ratio")
            or cleaned_payload.get("aspect_ratio")
            or _infer_aspect_ratio(str(settings.get("size") or ""))
        )
        if aspect_ratio:
            metadata["aspect_ratio"] = aspect_ratio
        duration = (
            settings.get("duration")
            or settings.get("duration_sec")
            or settings.get("duration_seconds")
            or cleaned_payload.get("duration_sec")
        )
        if isinstance(duration, (int, float)) and duration > 0:
            metadata["duration_sec"] = str(int(duration))
        elif isinstance(duration, str) and duration.isdigit():
            metadata["duration_sec"] = str(int(duration))
        if cleaned_payload.get("format"):
            metadata["format"] = cleaned_payload["format"]
        if cleaned_payload.get("seed") is not None:
            metadata["seed"] = cleaned_payload["seed"]
        if metadata:
            request["metadata"] = metadata
        return request

    async def _request(
        self,
        method: str,
        path: str,
        *,
        idempotency_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], int, int]:
        url = f"{self._base_url}{path}"
        params_payload = kwargs.get("params")
        json_payload = kwargs.get("json")
        data_payload = kwargs.get("data")
        payload = json_payload or data_payload or params_payload
        log.debug("Sora request: url=%s, payload=%s", url, _serialise_for_log(payload, limit=2048))
        try:
            data, status_code, duration_ms = await super()._request(
                method,
                path,
                idempotency_key=idempotency_key,
                **kwargs,
            )
        except ProviderAPIError as exc:
            raw_body = exc.provider_message or ""
            error_json: Dict[str, Any]
            try:
                error_json = json.loads(raw_body) if raw_body else {}
            except (TypeError, ValueError):
                error_json = {"raw": raw_body}
            error_code = None
            error_type = exc.error_type
            error_message = None
            if isinstance(error_json, dict):
                error_payload = error_json.get("error")
                if isinstance(error_payload, dict):
                    error_code = error_payload.get("code")
                    error_type = error_payload.get("type") or error_type
                    error_message = error_payload.get("message") or error_payload.get("detail")
                else:
                    error_code = error_json.get("code")
                    error_message = error_json.get("message")
                    error_type = error_json.get("type") or error_type
            serialised_error = _serialise_for_log(error_json, limit=2048)
            log.debug(
                "Sora response: status=%s, body=%s",
                exc.status_code,
                serialised_error,
            )
            if error_code or error_message:
                log.error(
                    "Sora API error in response code=%s message=%s",
                    error_code,
                    error_message,
                )
            log.warning(
                "sora.http.error method=%s url=%s status=%s duration_ms=%s error_code=%s error_message=%s text=%s json=%s",
                method,
                url,
                exc.status_code,
                exc.duration_ms or 0,
                error_code,
                error_message,
                raw_body,
                serialised_error,
            )
            model_not_found_codes = {
                "model_not_found",
                "invalid_model",
                "unsupported_model",
            }
            if error_code in model_not_found_codes:
                user_message = "Модель не найдена. Проверьте имя модели и попробуйте снова"
            else:
                provider_details = error_message or raw_body or str(exc)
                if error_code:
                    provider_details = f"[{error_code}] {provider_details}" if provider_details else f"[{error_code}]"
                user_message = f"Sora API error (status {exc.status_code}): {provider_details}"
            raise ProviderAPIError(
                provider="sora",
                status_code=exc.status_code,
                message=user_message,
                error_type=error_type or exc.error_type,
                error_code=error_code or exc.error_code,
                provider_message=raw_body or exc.provider_message,
                retryable=exc.retryable,
                duration_ms=exc.duration_ms,
            ) from exc
        error_code = None
        error_type = None
        error_message = None
        if isinstance(data, dict):
            error_payload = data.get("error")
            if isinstance(error_payload, dict):
                error_code = error_payload.get("code")
                error_type = error_payload.get("type")
                error_message = error_payload.get("message") or error_payload.get("detail")
                log.error(
                    "Sora API error in response code=%s message=%s",
                    error_code,
                    error_message,
                )
        serialised_response = _serialise_for_log(data, limit=2048)
        log.debug("Sora response: status=%s, body=%s", status_code, serialised_response)
        log.debug(
            "sora.http.response method=%s url=%s status=%s duration_ms=%s error_type=%s error_code=%s error_message=%s json=%s",
            method,
            url,
            status_code,
            duration_ms,
            error_type,
            error_code,
            error_message,
            serialised_response,
        )
        return data, status_code, duration_ms

    def _resolve_job_id(self, payload: Dict[str, Any]) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        for key in ("id", "response_id"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        response = payload.get("response")
        if isinstance(response, dict):
            value = response.get("id")
            if isinstance(value, str) and value:
                return value
        return None

    def _extract_status(self, payload: Dict[str, Any]) -> str:
        raw: Optional[str] = None
        if isinstance(payload, dict):
            candidate = payload.get("status")
            if isinstance(candidate, str) and candidate:
                raw = candidate
            elif isinstance(payload.get("response"), dict):
                response_status = payload["response"].get("status")
                if isinstance(response_status, str) and response_status:
                    raw = response_status
        if not raw:
            return "unknown"
        normalised = raw.strip().lower()
        if normalised in {"succeeded", "success"}:
            return "completed"
        if normalised in {"queued", "pending", "in_progress", "running"}:
            return "running"
        if normalised in {"failed", "errored", "error", "cancelled", "canceled"}:
            return "failed"
        if normalised == "completed":
            return "completed"
        return normalised or "unknown"

    def _extract_error(self, payload: Dict[str, Any]) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        error = payload.get("error")
        if isinstance(error, dict):
            detail = error.get("message") or error.get("detail") or error.get("code")
            if isinstance(detail, str) and detail:
                return detail
            return str(error)
        if isinstance(error, str) and error:
            return error
        last_error = payload.get("last_error")
        if isinstance(last_error, dict):
            message = last_error.get("message") or last_error.get("code")
            if isinstance(message, str) and message:
                return message
        return None

    def _extract_assets(self, payload: Dict[str, Any]) -> Dict[str, str]:
        assets: Dict[str, str] = {}
        if not isinstance(payload, dict):
            return assets
        for entry in _iter_nodes(payload):
            if not isinstance(entry, dict):
                continue
            entry_type = str(entry.get("type") or entry.get("kind") or "").lower()
            mime = _guess_mime(entry) or ""
            is_video_type = entry_type in {"output_video", "video", "video_asset"}
            is_video_mime = any(mime.startswith(prefix) for prefix in _VIDEO_MIME_PREFIXES)
            if not is_video_type and not is_video_mime:
                continue
            key = entry.get("file_id") or entry.get("id")
            if isinstance(key, str) and not key.strip():
                key = None
            url = _extract_url(entry)
            if not url:
                base64_payload = _extract_base64(entry)
                if base64_payload:
                    mime_type = mime or "video/mp4"
                    url = f"data:{mime_type};base64,{base64_payload}"
            if not isinstance(url, str) or not url:
                continue
            asset_key = str(key or f"asset_{len(assets)}")
            if asset_key not in assets:
                assets[asset_key] = url
        return assets

    def _is_terminal(self, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        status = self._extract_status(payload)
        return status in _TERMINAL_STATUSES
