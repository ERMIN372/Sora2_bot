"""OpenAI Sora video generation client."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import aiohttp

from config import DEBUG_SORA, Config, SORA_DEFAULT_MODEL, SORA_SUPPORTED_MODELS

from .api_error_handler import ApiErrorHandler
from .base import (
    BaseProviderClient,
    ModelUnavailable,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
    _extract_request_id,
    _join_url,
    _mask_secret,
    _sanitize_headers,
    _serialise_for_log,
    _truncate,
    extract_url,
    iter_nodes,
    stream_response_to_file,
    validate_download_url,
)
from services.payload_sanitize import (
    log_removed_keys,
    normalize_sora_payload,
    sanitize_payload,
)
from services.payload_whitelists import SORA_VIDEO_ALLOWED
from services.rate_limiter import AsyncRateLimiter

log = logging.getLogger(__name__)

BASE_URL = "https://api.openai.com/v1"
ENDPOINT = f"{BASE_URL}/videos"

_DEBUG_SORA_ENABLED_DEFAULT = bool(DEBUG_SORA)
_DEBUG_PREVIEW_LIMIT = 3072


_MODEL_CACHE_TTL = 600.0
_MODEL_CACHE: Dict[Tuple[str, str, str], Tuple[float, bool]] = {}
_MODEL_CACHE_LOCK = asyncio.Lock()


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


_iter_nodes = iter_nodes
_extract_url = extract_url


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


def _parse_error_payload(raw: Any) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str, Any]]:
    payload: Dict[str, Any]
    if isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            payload = {"raw": raw}
        else:
            payload = parsed if isinstance(parsed, dict) else {"raw": parsed}
    else:
        payload = {}

    error_code: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    if isinstance(payload, dict):
        error_payload = payload.get("error")
        if isinstance(error_payload, dict):
            error_code = (
                error_payload.get("code")
                or error_payload.get("error_code")
                or error_payload.get("status")
            )
            error_type = error_payload.get("type") or error_payload.get("error_type")
            error_message = (
                error_payload.get("message")
                or error_payload.get("detail")
                or error_payload.get("title")
            )
        else:
            if isinstance(payload.get("code"), str):
                error_code = payload.get("code")
            if isinstance(payload.get("type"), str):
                error_type = payload.get("type")
            raw_message = payload.get("message") or payload.get("error")
            if isinstance(raw_message, str):
                error_message = raw_message
    return error_code, error_type, error_message, payload


async def _has_model(
    token: str,
    org: Optional[str],
    project: Optional[str],
    model_id: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    session: Optional[aiohttp.ClientSession] = None,
    base_url: Optional[str] = None,
) -> bool:
    model_name = (model_id or "").strip()
    if not model_name:
        return False
    cache_key = ((org or "").strip(), (project or "").strip(), model_name)
    now = time.monotonic()
    async with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached and now - cached[0] < _MODEL_CACHE_TTL:
            return cached[1]

    request_headers = dict(headers or {})
    if token and not request_headers.get("Authorization"):
        request_headers["Authorization"] = f"Bearer {token}"
    if org and "OpenAI-Organization" not in request_headers:
        request_headers["OpenAI-Organization"] = org
    if project and "OpenAI-Project" not in request_headers:
        request_headers["OpenAI-Project"] = project
    request_headers.setdefault("Content-Type", "application/json")

    base = (base_url or BASE_URL).rstrip("/")
    models_url = _join_url(base, "v1", "models")

    close_session = False
    client_session = session
    if client_session is None:
        timeout = aiohttp.ClientTimeout(total=10)
        client_session = aiohttp.ClientSession(timeout=timeout)
        close_session = True

    payload: Dict[str, Any] = {}
    request_method_name: Optional[str] = None
    request_method = getattr(client_session, "request", None)
    if callable(request_method):
        request_method_name = "request"
    else:
        get_method = getattr(client_session, "get", None)
        if callable(get_method):
            request_method_name = "get"
            request_method = get_method
    if not request_method_name or request_method is None:
        log.debug("sora.model_preflight.unsupported_session session=%s", type(client_session))
        return True

    try:
        if request_method_name == "request":
            context_manager = request_method("GET", models_url, headers=request_headers)
        else:
            context_manager = request_method(models_url, headers=request_headers)
        async with context_manager as response:
            if response.status != 200:
                body_preview = await response.text()
                log.warning(
                    "sora.model_preflight.unexpected_status status=%s body=%s",
                    response.status,
                    _truncate(body_preview, limit=256),
                )
                return True
            try:
                payload = await response.json()
            except (aiohttp.ContentTypeError, json.JSONDecodeError):
                log.warning("sora.model_preflight.invalid_json")
                return True
    except asyncio.TimeoutError:
        log.warning("sora.model_preflight.timeout url=%s", models_url)
        return True
    except aiohttp.ClientError as exc:
        log.warning("sora.model_preflight.client_error error=%s", exc)
        return True
    finally:
        if close_session and client_session is not None:
            await client_session.close()

    result = False
    data_payload = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data_payload, list):
        for entry in data_payload:
            if isinstance(entry, dict) and (entry.get("id") or "").strip() == model_name:
                result = True
                break

    async with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[cache_key] = (time.monotonic(), result)
    if not result:
        log.info("sora.model_preflight.missing model=%s", model_name)
    return result


class SoraVideoClient(BaseProviderClient):
    """Thin wrapper around OpenAI's Videos API for Sora generation."""

    def __init__(self, *, config: Config) -> None:
        api_key = config.openai_key
        if not api_key:
            raise RuntimeError("Sora API key is not configured; set SORA_API_KEY or OPENAI_API_KEY")
        base_url = (config.openai_api_base or "https://api.openai.com").rstrip("/")
        raw_version = (config.openai_api_version_video or "v1").strip().strip("/")
        if not raw_version:
            raw_version = "v1"
        normalised_version = raw_version.lower()
        if normalised_version != "v1":
            log.warning(
                "Unsupported Sora API version requested=%s; forcing v1 for videos endpoint",
                raw_version,
            )
        api_version = "v1"
        beta_header = (config.openai_beta_header or "video=1").strip()
        default_headers: Dict[str, str] = {}
        if beta_header:
            default_headers["OpenAI-Beta"] = beta_header
        if config.openai_org_id:
            default_headers["OpenAI-Organization"] = config.openai_org_id
        project_header = (
            getattr(config, "openai_project_id", None)
            or os.getenv("OPENAI_PROJECT_ID")
            or os.getenv("OPENAI_PROJECT")
            or ""
        ).strip()
        if project_header:
            default_headers["OpenAI-Project"] = project_header
        super().__init__(
            config=config,
            base_url=base_url,
            api_key=api_key,
            provider_name="sora",
            default_headers=default_headers,
        )
        log.debug(
            "SoraVideoClient configured base_url=%s api_version=%s beta=%s org_id=%s api_key=%s",
            base_url,
            api_version,
            beta_header or "default",
            _mask_secret(config.openai_org_id),
            _mask_secret(api_key),
        )
        self._beta_header = beta_header
        self._organization_id = (config.openai_org_id or "").strip()
        self._project_id = project_header
        self._api_version = api_version
        self._debug_enabled = bool(config.debug_sora or _DEBUG_SORA_ENABLED_DEFAULT)
        self._last_download_meta: Dict[str, Any] = {}
        self._supported_models = set(SORA_SUPPORTED_MODELS)
        configured_model = config.resolved_sora_video_model
        fallback_model = SORA_DEFAULT_MODEL
        if configured_model:
            self._supported_models.add(configured_model)
        if configured_model in self._supported_models:
            self._default_model = configured_model
        else:
            if configured_model and configured_model != fallback_model:
                log.warning(
                    "Unsupported default Sora model configured model=%s; falling back to '%s'",
                    configured_model,
                    fallback_model,
                )
            self._default_model = fallback_model
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._error_handler = ApiErrorHandler(
            provider="sora", logger=log, supported_models=SORA_SUPPORTED_MODELS
        )
        limit_per_minute = max(1, config.sora_requests_per_minute)
        self._rate_limiter = AsyncRateLimiter(limit_per_minute, 60.0)

    # ------------------------------------------------------------------
    # High level operations
    # ------------------------------------------------------------------

    @property
    def supported_models(self) -> Tuple[str, ...]:
        return tuple(sorted(self._supported_models))

    async def _ensure_model_access(self, model_id: str) -> None:
        if not model_id:
            return
        session = await self._ensure_session()
        headers = self._build_headers()
        has_access = await _has_model(
            self._api_key,
            self._organization_id or None,
            self._project_id or None,
            model_id,
            headers=headers,
            session=session,
            base_url=self._base_url,
        )
        if not has_access:
            raise ModelUnavailable(
                f"Model not found: {model_id}",
                provider=self.provider_name,
            )

    def _build_endpoint_segments(self, *suffix: str) -> Tuple[str, ...]:
        segments: List[str] = [self._api_version]
        for part in suffix:
            text = str(part or "").strip("/")
            if not text:
                continue
            segments.extend(fragment for fragment in text.split("/") if fragment)
        return tuple(segments)

    def _endpoint_path(self, *suffix: str) -> str:
        segments = self._build_endpoint_segments(*suffix)
        return "/" + "/".join(segments)

    def _resolve_target(
        self, path: Union[str, Iterable[str], None]
    ) -> Tuple[Tuple[str, ...], str, str]:
        raw_segments: List[str] = []
        if isinstance(path, str):
            raw_segments.extend(
                fragment
                for fragment in path.strip("/").split("/")
                if fragment
            )
        elif isinstance(path, Iterable):
            for part in path:
                text = str(part or "").strip("/")
                if not text:
                    continue
                raw_segments.extend(fragment for fragment in text.split("/") if fragment)
        if not raw_segments:
            raw_segments = [self._api_version]
        elif raw_segments[0] not in {"v1", "v1beta"}:
            raw_segments.insert(0, self._api_version)
        segments = tuple(raw_segments)
        endpoint = "/" + "/".join(segments)
        url = _join_url(self._base_url, *segments)
        return segments, endpoint, url

    def responses_url(self) -> str:
        """Return the fully-qualified Videos API URL used for Sora."""

        _, _, url = self._resolve_target(self._build_endpoint_segments("videos"))
        return url

    def responses_path(self) -> str:
        """Return the relative Videos API path used for Sora."""

        return self._endpoint_path("videos")

    def _log_debug(self, event: str, payload: Dict[str, Any]) -> None:
        if not self._debug_enabled:
            return
        entry = dict(payload)
        entry.setdefault("provider", self.provider_name)
        entry.setdefault("event", event)
        try:
            message = json.dumps(entry, ensure_ascii=False, default=str)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            message = json.dumps(
                {
                    "event": event,
                    "provider": self.provider_name,
                    "unserializable": True,
                    "payload_repr": repr(payload),
                },
                ensure_ascii=False,
            )
        log.info(message)

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
        if requested_model not in self._supported_models:
            if requested_model:
                log.warning(
                    "Unsupported Sora model requested model=%s; using default=%s",
                    requested_model,
                    self._default_model,
                )
            model_name = self._default_model
        else:
            model_name = requested_model
        settings["model"] = model_name
        await self._ensure_model_access(model_name)
        request_payload = self._build_request_payload(
            prompt=prompt,
            settings=settings,
            payload=payload or {},
        )
        endpoint_segments = self._build_endpoint_segments("videos")
        endpoint = "/" + "/".join(endpoint_segments)
        url = _join_url(self._base_url, *endpoint_segments)
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
            endpoint_segments,
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
            endpoint_segments = self._build_endpoint_segments("videos", job_id)
            endpoint = "/" + "/".join(endpoint_segments)
            url = _join_url(self._base_url, *endpoint_segments)
            log.debug("Sora request: url=%s, payload=%s", url, "")
            log.info(
                "sora.status start endpoint=%s method=%s",
                endpoint,
                "GET",
            )
            data, status_code, duration_ms = await self._request("GET", endpoint_segments)
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
        request: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt_text,
        }
        size_value = settings.get("size") or cleaned_payload.get("size")
        if isinstance(size_value, str) and size_value.strip():
            request["size"] = size_value.strip()
        duration = (
            settings.get("seconds")
            or cleaned_payload.get("seconds")
            or settings.get("duration")
            or settings.get("duration_sec")
            or settings.get("duration_seconds")
            or cleaned_payload.get("duration_sec")
            or cleaned_payload.get("duration_seconds")
        )
        seconds_text: Optional[str] = None
        if isinstance(duration, (int, float)) and duration > 0:
            seconds_text = str(int(duration))
        elif isinstance(duration, str) and duration.strip():
            seconds_text = duration.strip()
        if seconds_text:
            request["seconds"] = seconds_text
        return request

    async def _request(
        self,
        method: str,
        path: Union[str, Iterable[str]],
        *,
        idempotency_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], int, int]:
        _, endpoint, url = self._resolve_target(path)
        request_kwargs = dict(kwargs)
        extra_headers = request_kwargs.pop("headers", {})
        params_payload = request_kwargs.get("params")
        json_payload = request_kwargs.get("json")
        data_payload = request_kwargs.get("data")
        payload = json_payload or data_payload or params_payload
        payload_preview = _serialise_for_log(payload, limit=_DEBUG_PREVIEW_LIMIT)
        log.debug("Sora request: url=%s, payload=%s", url, payload_preview)

        headers = self._build_headers()
        headers.pop("OpenAI-Version", None)
        headers.pop("openai-version", None)
        if idempotency_key:
            headers.setdefault("Idempotency-Key", idempotency_key)
        headers.update(extra_headers)
        headers.pop("OpenAI-Version", None)
        headers.pop("openai-version", None)
        safe_headers = _sanitize_headers(dict(headers))

        attempt = 0
        backoff_delay = self._config.retry_backoff
        server_retry_index = 0

        while True:
            await self._rate_limiter.acquire()
            self._log_debug(
                "sora.debug.request",
                {
                    "method": method,
                    "url": url,
                    "endpoint": endpoint,
                    "api_version": self._api_version,
                    "attempt": attempt + 1,
                    "headers": safe_headers,
                    "idempotency_key": headers.get("Idempotency-Key", ""),
                    "body_preview": payload_preview,
                    "base": self._base_url,
                },
            )
            try:
                data, status_code, duration_ms = await super()._perform_request(
                    method,
                    url,
                    headers=headers,
                    **request_kwargs,
                )
            except ProviderAPIError as exc:
                raw_body = exc.provider_message or ""
                error_code, error_type, error_message, error_payload = _parse_error_payload(raw_body)
                serialised_error = _serialise_for_log(error_payload, limit=_DEBUG_PREVIEW_LIMIT)
                log.error(
                    "sora.api.error status=%s code=%s message=%s",
                    exc.status_code,
                    error_code or exc.error_code,
                    error_message or str(exc),
                )
                log.warning(
                    "sora.http.error method=%s url=%s status=%s duration_ms=%s error_type=%s error_code=%s error_message=%s text=%s json=%s",
                    method,
                    url,
                    exc.status_code,
                    exc.duration_ms or 0,
                    error_type or exc.error_type,
                    error_code or exc.error_code,
                    error_message or str(exc),
                    raw_body,
                    serialised_error,
                )
                self._log_debug(
                    "sora.debug.error",
                    {
                        "method": method,
                        "url": url,
                        "endpoint": endpoint,
                        "status": exc.status_code,
                        "duration_ms": exc.duration_ms or 0,
                        "error_type": error_type or exc.error_type,
                        "error_code": error_code or exc.error_code,
                        "body_preview": _truncate(raw_body, limit=_DEBUG_PREVIEW_LIMIT),
                    },
                )

                error_type_text = (error_type or exc.error_type or "").strip()
                error_message_text = error_message or str(exc)
                if (
                    exc.status_code == 404
                    and error_type_text == "invalid_request_error"
                    and isinstance(error_message_text, str)
                    and "Model not found" in error_message_text
                ):
                    log.warning(
                        "sora.enqueue failure reason=no_access_or_wrong_endpoint status=%s message=%s",
                        exc.status_code,
                        error_message_text,
                    )
                    raise ModelUnavailable(
                        error_message_text,
                        provider=self.provider_name,
                        status_code=exc.status_code,
                        error_type=error_type_text,
                        error_code=error_code or exc.error_code,
                        provider_message=raw_body or exc.provider_message,
                        request_id=exc.request_id,
                    ) from exc

                if exc.status_code == 404 and "/v1/v1beta/" in url:
                    log.critical(
                        "suspect_misconfigured_endpoint provider=%s method=%s url=%s",
                        self.provider_name,
                        method,
                        url,
                    )
                if exc.status_code == 401:
                    log.error("sora.auth_error method=%s url=%s hint=check OPENAI_API_KEY", method, url)

                if exc.status_code in self._error_handler.SERVER_ERROR_STATUSES:
                    delay = self._error_handler.server_retry_delay(server_retry_index)
                    if delay is not None:
                        server_retry_index += 1
                        log.warning(
                            "sora.http.retry server_error status=%s attempt=%s delay=%s",
                            exc.status_code,
                            server_retry_index,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                resolution = self._error_handler.resolve(
                    status_code=exc.status_code,
                    error_code=error_code or exc.error_code,
                    error_message=error_message or str(exc),
                )
                if resolution:
                    if exc.status_code == 429 and resolution.cooldown_seconds:
                        await self._rate_limiter.penalize(resolution.cooldown_seconds)
                    self._error_handler.log_admin_hint(resolution)
                    raise self._error_handler.apply(
                        base_error=exc,
                        resolution=resolution,
                        raw_body=raw_body,
                        fallback_code=error_code or exc.error_code,
                    ) from exc

                if (
                    exc.retryable
                    and attempt < self._config.request_retries
                    and exc.status_code not in self._error_handler.SERVER_ERROR_STATUSES
                ):
                    attempt += 1
                    log.warning(
                        "sora.http.retry status=%s attempt=%s/%s delay=%s",
                        exc.status_code,
                        attempt,
                        self._config.request_retries,
                        backoff_delay,
                    )
                    await asyncio.sleep(backoff_delay)
                    backoff_delay *= self._config.retry_backoff
                    continue

                log.error("sora.http.unknown_error status=%s body=%s", exc.status_code, raw_body)

                friendly_code: Optional[str] = None
                friendly_message: Optional[str] = None

                if exc.status_code == 0:
                    friendly_code = error_code or exc.error_code or "network_error"
                    friendly_message = "Sora API временно недоступен, попробуйте позже"
                    raise ProviderAPIError(
                        provider="sora",
                        status_code=exc.status_code,
                        message=friendly_message,
                        error_type=exc.error_type or "network",
                        error_code=friendly_code,
                        provider_message=raw_body or exc.provider_message,
                        retryable=True,
                        duration_ms=exc.duration_ms,
                        code=friendly_code,
                    ) from exc

                if exc.status_code == 404 and "/v1/v1beta/" in url:
                    friendly_code = "endpoint_misconfigured"
                    friendly_message = "Sora endpoint misconfigured (double version prefix)"
                elif exc.status_code == 401:
                    friendly_code = "auth_failed"
                    friendly_message = "Sora authentication failed; check OPENAI_API_KEY"
                elif exc.status_code == 429:
                    friendly_code = "rate_limited"
                    friendly_message = "Sora API rate limit exceeded"
                elif exc.status_code >= 500:
                    friendly_code = "server_error"
                    friendly_message = "Sora API server error"

                raise ProviderAPIError(
                    provider="sora",
                    status_code=exc.status_code,
                    message=friendly_message
                    or error_message
                    or str(exc)
                    or "Sora API error",
                    error_type=friendly_code or error_type or exc.error_type or "unknown",
                    error_code=friendly_code or error_code or exc.error_code or "unknown_error",
                    provider_message=raw_body or exc.provider_message,
                    retryable=exc.retryable,
                    duration_ms=exc.duration_ms,
                    code=friendly_code or error_code or exc.error_code,
                    body=raw_body,
                    request_id=exc.request_id,
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
            serialised_response = _serialise_for_log(data, limit=_DEBUG_PREVIEW_LIMIT)
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
            meta = dict(self._last_response_meta or {})
            response_headers = meta.get("headers") or {}
            request_id = meta.get("request_id") or ""
            content_length = (
                response_headers.get("Content-Length")
                or response_headers.get("content-length")
                or ""
            )
            self._log_debug(
                "sora.debug.response",
                {
                    "method": method,
                    "url": url,
                    "endpoint": endpoint,
                    "status": status_code,
                    "duration_ms": duration_ms,
                    "request_id": request_id,
                    "content_length": content_length,
                    "body_preview": serialised_response,
                },
            )
            return data, status_code, duration_ms

    async def download_content(self, video_id: str, format: str = "mp4") -> Path:
        if not video_id:
            raise ValueError("video_id must not be empty")
        allowed_formats = {"mp4", "webm"}
        fmt = (format or "mp4").strip().lower()
        if fmt not in allowed_formats:
            fmt = "mp4"
        endpoint_segments = self._build_endpoint_segments("videos", video_id, "content")
        endpoint = "/" + "/".join(endpoint_segments)
        url = _join_url(self._base_url, *endpoint_segments)
        params = {"format": fmt}
        headers = self._build_headers()
        headers.pop("Content-Type", None)
        headers["Accept"] = "video/mp4" if fmt == "mp4" else "video/webm"
        validate_download_url(url)
        session = await self._ensure_session()
        timeout = aiohttp.ClientTimeout(
            total=self._config.request_read_timeout or None,
            sock_read=self._config.request_read_timeout or None,
        )
        delays = (0, 1, 3, 7, 15)
        for attempt, delay in enumerate(delays, start=1):
            if delay:
                await asyncio.sleep(delay)
            await self._rate_limiter.acquire()
            started = time.monotonic()
            try:
                async with session.get(
                    url,
                    params=params,
                    headers=headers,
                    allow_redirects=True,
                    timeout=timeout,
                ) as response:
                    duration_ms = int((time.monotonic() - started) * 1000)
                    status = int(response.status)
                    response_headers = dict(response.headers)
                    request_id = _extract_request_id(response_headers) or ""
                    meta: Dict[str, Any] = {
                        "video_id": video_id,
                        "status_code": status,
                        "request_id": request_id,
                        "duration_ms": duration_ms,
                        "attempt": attempt,
                        "url": url,
                        "format": fmt,
                        "timestamp": time.time(),
                    }
                    if status == 200:
                        target_dir = Path(tempfile.mkdtemp(prefix="sora-video-"))
                        file_path = target_dir / f"{video_id}.{fmt}"
                        size = await stream_response_to_file(response, file_path)
                        meta["bytes"] = size
                        meta["file_path"] = str(file_path)
                        meta["note"] = "success"
                        self._last_download_meta = meta
                        log.info(
                            "sora.download.success video_id=%s status=%s bytes=%s request_id=%s",
                            video_id,
                            status,
                            size,
                            request_id,
                        )
                        return file_path
                    if status in (404, 409):
                        note = "content_not_ready"
                        meta["note"] = note
                        self._last_download_meta = meta
                        log.debug(
                            "sora.download.retry video_id=%s status=%s attempt=%s delay=%s",
                            video_id,
                            status,
                            attempt,
                            delay,
                        )
                        if attempt < len(delays):
                            continue
                        break
                    if status in (429, 500, 502, 503, 504):
                        meta["note"] = "server_retry"
                        self._last_download_meta = meta
                        log.warning(
                            "sora.download.server_retry video_id=%s status=%s attempt=%s",
                            video_id,
                            status,
                            attempt,
                        )
                        if attempt < len(delays):
                            continue
                    body = await response.text()
                    meta["note"] = "error"
                    self._last_download_meta = meta
                    raise ProviderAPIError(
                        provider=self.provider_name,
                        status_code=status,
                        message=f"Sora content download failed with status {status}",
                        error_type="http",
                        error_code=str(status),
                        provider_message=body,
                        retryable=status in (429, 500, 502, 503, 504),
                        duration_ms=duration_ms,
                    )
            except asyncio.TimeoutError as exc:
                duration_ms = int((time.monotonic() - started) * 1000)
                self._last_download_meta = {
                    "video_id": video_id,
                    "status_code": 0,
                    "request_id": "",
                    "duration_ms": duration_ms,
                    "attempt": attempt,
                    "url": url,
                    "format": fmt,
                    "timestamp": time.time(),
                    "note": "timeout",
                }
                raise ProviderAPIError(
                    provider=self.provider_name,
                    status_code=0,
                    message="Sora content download timed out",
                    error_type="timeout",
                    error_code="timeout",
                    provider_message=str(exc),
                    retryable=True,
                    duration_ms=duration_ms,
                ) from exc
            except aiohttp.ClientError as exc:
                duration_ms = int((time.monotonic() - started) * 1000)
                self._last_download_meta = {
                    "video_id": video_id,
                    "status_code": 0,
                    "request_id": "",
                    "duration_ms": duration_ms,
                    "attempt": attempt,
                    "url": url,
                    "format": fmt,
                    "timestamp": time.time(),
                    "note": "client_error",
                }
                raise ProviderAPIError(
                    provider=self.provider_name,
                    status_code=0,
                    message="Sora content download failed",
                    error_type="network",
                    error_code="network_error",
                    provider_message=str(exc),
                    retryable=True,
                    duration_ms=duration_ms,
                ) from exc
        self._last_download_meta = {
            "video_id": video_id,
            "status_code": 0,
            "request_id": "",
            "duration_ms": 0,
            "attempt": len(delays),
            "url": url,
            "format": fmt,
            "timestamp": time.time(),
            "note": "content_not_ready",
        }
        raise ProviderAPIError(
            provider=self.provider_name,
            status_code=0,
            message="Контент ещё не готов, попробуйте позже",
            error_type="content_not_ready",
            error_code="content_not_ready",
            provider_message="",
            retryable=True,
            code="content_not_ready",
        )

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


async def sora_self_test(config: Config) -> Tuple[str, Dict[str, Any]]:
    """Execute a minimal self-test request against the Sora Videos API."""

    client = SoraVideoClient(config=config)
    payload = client._build_request_payload(  # type: ignore[attr-defined]
        prompt="ping",
        settings={"model": client._default_model, "duration": 1},
        payload={},
    )
    target_segments = client._build_endpoint_segments("videos")
    url = _join_url(client._base_url, *target_segments)
    log.info("sora.self_test.start url=%s", url)
    try:
        data, status_code, duration_ms = await client._request(
            "POST",
            target_segments,
            json=payload,
        )
        log.info(
            "sora.self_test.success url=%s status=%s duration_ms=%s",
            url,
            status_code,
            duration_ms,
        )
        return "ok", {
            "url": url,
            "status": status_code,
            "duration_ms": duration_ms,
            "response": data,
        }
    except ProviderAPIError as exc:
        details = {
            "url": url,
            "status": exc.status_code,
            "error_type": exc.error_type,
            "error_code": exc.error_code,
            "message": str(exc),
        }
        if exc.status_code == 404 and "/v1/v1beta/" in url:
            log.error("sora.self_test.endpoint_misconfigured url=%s", url)
            return "endpoint_misconfigured", details
        log.error(
            "sora.self_test.error url=%s status=%s error_type=%s error_code=%s",
            url,
            exc.status_code,
            exc.error_type,
            exc.error_code,
        )
        return "error", details
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("sora.self_test.exception url=%s", url)
        return "exception", {"url": url, "message": str(exc)}
    finally:
        await client.close()
