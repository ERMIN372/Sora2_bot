"""Client for the OpenAI Videos API (Sora 2 models)."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from config import Config

from .api_error_handler import ApiErrorHandler
from .base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
    _mask_secret,
    _serialise_for_log,
)
from services.payload_sanitize import log_removed_keys, normalize_sora_payload, sanitize_payload
from services.payload_whitelists import SORA_VIDEO_ALLOWED


_PROVIDERS_LOG = logging.getLogger("providers")
log = _PROVIDERS_LOG.getChild("openai_video") if _PROVIDERS_LOG else logging.getLogger(__name__)


_DEFAULT_VIDEO_MODELS: Tuple[str, ...] = ("sora-2", "sora-2-fast", "sora-2-pro")
_VIDEO_MIME_PREFIXES = ("video/", "application/octet-stream")


def _iter_nodes(value: Any) -> Iterable[Dict[str, Any]]:
    stack: List[Any] = [value]
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


def _extract_url(entry: Mapping[str, Any]) -> Optional[str]:
    for key in ("download_url", "url", "video_url", "asset_url", "content_url"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = entry.get("file") or entry.get("video") or entry.get("asset")
    if isinstance(nested, Mapping):
        for key in ("download_url", "url", "content_url"):
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_base64(entry: Mapping[str, Any]) -> Optional[str]:
    for key in ("b64_json", "base64", "data", "inline_data"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            nested = value.get("data") or value.get("base64")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
    return None


def _guess_mime(entry: Mapping[str, Any]) -> Optional[str]:
    for key in ("mime_type", "content_type", "mime"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = entry.get("file") or entry.get("video")
    if isinstance(nested, Mapping):
        for key in ("mime_type", "content_type", "mime"):
            value = nested.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


class OpenAIVideoClient(BaseProviderClient):
    """Client for the OpenAI Videos API used by Sora 2 models."""

    def __init__(self, *, config: Config) -> None:
        api_key = (config.openai_key or "").strip()
        if not api_key:
            raise RuntimeError("OpenAI API key is required for video generation")
        base_url = (config.openai_api_base or "https://api.openai.com/v1").rstrip("/")
        beta_header = (config.openai_beta_header or "video=1").strip()
        default_headers: Dict[str, str] = {}
        if beta_header:
            default_headers["OpenAI-Beta"] = beta_header
        if config.openai_org_id:
            default_headers["OpenAI-Organization"] = config.openai_org_id
        super().__init__(
            config=config,
            base_url=base_url,
            api_key=api_key,
            provider_name="openai-video",
            default_headers=default_headers,
        )
        self._supported_models: set[str] = set(_DEFAULT_VIDEO_MODELS)
        configured_model = (config.sora_model_video or "").strip()
        if configured_model:
            self._supported_models.add(configured_model)
        self._default_model = configured_model or _DEFAULT_VIDEO_MODELS[0]
        self._api_version = (config.openai_api_version_video or "").strip()
        self._default_params: Dict[str, Any] = {}
        if self._api_version:
            self._default_params["api-version"] = self._api_version
        self._error_handler = ApiErrorHandler(
            provider="openai-video", logger=log, supported_models=sorted(self._supported_models)
        )
        log.debug(
            "OpenAIVideoClient configured base_url=%s api_version=%s beta=%s org_id=%s api_key=%s",
            base_url,
            self._api_version or "default",
            beta_header or "default",
            _mask_secret(config.openai_org_id),
            _mask_secret(api_key),
        )

    # ------------------------------------------------------------------
    # High level API
    # ------------------------------------------------------------------

    @property
    def supported_models(self) -> Tuple[str, ...]:
        return tuple(sorted(self._supported_models))

    async def create(
        self,
        *,
        prompt: str,
        settings: Optional[Mapping[str, Any]] = None,
        payload: Optional[Mapping[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        return_meta: bool = False,
    ) -> Any:
        request = self._build_create_payload(
            prompt=prompt,
            settings=dict(settings or {}),
            payload=dict(payload or {}),
        )
        data, status_code, duration_ms = await self._request_json(
            "POST",
            self._videos_path(),
            json=request,
            idempotency_key=idempotency_key,
        )
        log.info(
            "openai.video.create status=%s duration_ms=%s model=%s", status_code, duration_ms, request.get("model")
        )
        return (data, status_code, duration_ms) if return_meta else data

    async def remix(
        self,
        *,
        video_id: str,
        prompt: Optional[str] = None,
        settings: Optional[Mapping[str, Any]] = None,
        payload: Optional[Mapping[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        return_meta: bool = False,
    ) -> Any:
        if not video_id:
            raise ValueError("video_id must not be empty")
        request = self._build_create_payload(
            prompt=prompt or "",
            settings=dict(settings or {}),
            payload=dict(payload or {}),
            allow_empty_prompt=True,
        )
        data, status_code, duration_ms = await self._request_json(
            "POST",
            f"{self._videos_path()}/{video_id}/remix",
            json=request,
            idempotency_key=idempotency_key,
        )
        log.info(
            "openai.video.remix status=%s duration_ms=%s video_id=%s", status_code, duration_ms, video_id
        )
        return (data, status_code, duration_ms) if return_meta else data

    async def list(
        self,
        *,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        return_meta: bool = False,
    ) -> Any:
        params: Dict[str, Any] = {}
        if limit is not None:
            try:
                params["limit"] = int(limit)
            except (TypeError, ValueError):
                log.debug("Invalid limit provided to openai.video.list", exc_info=True)
        if order:
            params["order"] = str(order)
        if after:
            params["after"] = str(after)
        if before:
            params["before"] = str(before)
        data, status_code, duration_ms = await self._request_json(
            "GET", self._videos_path(), params=params or None
        )
        log.debug(
            "openai.video.list status=%s duration_ms=%s params=%s",
            status_code,
            duration_ms,
            _serialise_for_log(params, limit=512),
        )
        return (data, status_code, duration_ms) if return_meta else data

    async def retrieve(self, video_id: str, *, return_meta: bool = False) -> Any:
        if not video_id:
            raise ValueError("video_id must not be empty")
        data, status_code, duration_ms = await self._request_json(
            "GET", f"{self._videos_path()}/{video_id}"
        )
        log.debug(
            "openai.video.retrieve status=%s duration_ms=%s video_id=%s",
            status_code,
            duration_ms,
            video_id,
        )
        return (data, status_code, duration_ms) if return_meta else data

    async def content(
        self,
        video_id: str,
        *,
        asset_id: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[bytes, Dict[str, str]]:
        if not video_id:
            raise ValueError("video_id must not be empty")
        path = f"{self._videos_path()}/{video_id}/content"
        if asset_id:
            path += f"/{asset_id}"
        body, headers = await self._request_binary("GET", path, params=params or None)
        log.debug(
            "openai.video.content video_id=%s asset_id=%s bytes=%s",
            video_id,
            asset_id,
            len(body),
        )
        return body, headers

    async def list_models(self) -> List[str]:
        data, status_code, duration_ms = await self._request_json("GET", "/models")
        log.debug("openai.video.list_models status=%s duration_ms=%s", status_code, duration_ms)
        models: List[str] = []
        if isinstance(data, Mapping):
            entries = data.get("data")
            if isinstance(entries, list):
                for item in entries:
                    if isinstance(item, Mapping):
                        model_id = item.get("id")
                        if isinstance(model_id, str) and model_id:
                            models.append(model_id)
        if models:
            for model in models:
                if model.startswith("sora") or model.startswith("gpt"):
                    self._supported_models.add(model)
        return models

    def get_video_object(self, payload: Any) -> Optional[Dict[str, Any]]:
        if isinstance(payload, Mapping):
            if str(payload.get("object")) == "video":
                return dict(payload)
            for key in ("data", "items", "results"):
                nested = payload.get(key)
                if isinstance(nested, list):
                    for entry in nested:
                        result = self.get_video_object(entry)
                        if result:
                            return result
            for nested_key in ("video", "asset", "response"):
                nested = payload.get(nested_key)
                result = self.get_video_object(nested)
                if result:
                    return result
        elif isinstance(payload, list):
            for item in payload:
                result = self.get_video_object(item)
                if result:
                    return result
        return None

    # ------------------------------------------------------------------
    # Job queue compatibility
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
        data, status_code, duration_ms = await self.create(
            prompt=prompt,
            settings=settings,
            payload=payload,
            idempotency_key=idempotency_key,
            return_meta=True,
        )
        job_id = self._resolve_job_id(data)
        if not job_id:
            raise ProviderAPIError(
                provider=self.provider_name,
                status_code=status_code,
                message="OpenAI Video API did not return a job id",
                error_type="protocol",
                provider_message=json.dumps(data),
            )
        return ProviderJobSubmission(
            job_id=job_id,
            status_code=status_code,
            duration_ms=duration_ms,
            data=data,
        )

    async def get_job_status(self, job_id: str) -> ProviderJobStatus:
        data, status_code, duration_ms = await self.retrieve(job_id, return_meta=True)
        status = self._extract_status(data)
        error = self._extract_error(data)
        assets = self._extract_assets(data)
        return ProviderJobStatus(
            job_id=job_id,
            status=status,
            error=error,
            assets=assets,
            data=data,
            status_code=status_code,
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _videos_path(self) -> str:
        return "/videos"

    def _build_create_payload(
        self,
        *,
        prompt: str,
        settings: Dict[str, Any],
        payload: Dict[str, Any],
        allow_empty_prompt: bool = False,
    ) -> Dict[str, Any]:
        normalized_payload = normalize_sora_payload(dict(payload or {}))
        sanitized_payload = sanitize_payload(normalized_payload, SORA_VIDEO_ALLOWED)
        log_removed_keys("openai-video", normalized_payload, sanitized_payload, logger=log)
        request: Dict[str, Any] = dict(sanitized_payload)
        prompt_text = str(request.get("prompt") or "").strip()
        if not prompt_text and prompt:
            prompt_text = str(prompt).strip()
        if not prompt_text and not allow_empty_prompt:
            raise ValueError("prompt must not be empty")
        if prompt_text:
            request["prompt"] = prompt_text
        model_name = (settings.get("model") or request.get("model") or self._default_model).strip()
        if not model_name:
            model_name = self._default_model
        request["model"] = model_name
        self._supported_models.add(model_name)
        metadata = dict(request.get("metadata") or {})
        aspect_ratio = settings.get("aspect_ratio") or request.get("aspect_ratio")
        if aspect_ratio and isinstance(aspect_ratio, str):
            metadata.setdefault("aspect_ratio", aspect_ratio)
        duration = (
            settings.get("duration")
            or settings.get("duration_seconds")
            or settings.get("duration_sec")
            or request.get("duration_sec")
        )
        if isinstance(duration, (int, float)) and duration > 0:
            metadata.setdefault("duration_sec", str(int(duration)))
        elif isinstance(duration, str) and duration.isdigit():
            metadata.setdefault("duration_sec", duration)
        if metadata:
            request["metadata"] = metadata
        for key, value in settings.items():
            if value is None:
                continue
            if key in {"model", "prompt", "duration", "duration_sec", "duration_seconds", "aspect_ratio"}:
                continue
            request.setdefault(key, value)
        return request

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        idempotency_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], int, int]:
        params = kwargs.pop("params", None)
        merged_params = self._merge_params(params)
        if merged_params:
            kwargs["params"] = merged_params
        try:
            data, status_code, duration_ms = await super()._request(
                method, path, idempotency_key=idempotency_key, **kwargs
            )
        except ProviderAPIError as exc:
            raise self._normalise_error(exc) from exc
        return data, status_code, duration_ms

    async def _request_binary(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[bytes, Dict[str, str]]:
        session = await self._ensure_session()
        url = f"{self._base_url}{path}"
        headers = self._build_headers()
        merged_params = self._merge_params(params)
        if merged_params is None:
            merged_params = {}
        log.debug(
            "openai.video.binary_request method=%s url=%s params=%s",
            method,
            url,
            _serialise_for_log(dict(merged_params), limit=512),
        )
        start = time.monotonic()
        try:
            async with session.request(
                method, url, headers=headers, params=dict(merged_params)
            ) as response:
                body = await response.read()
                duration_ms = int((time.monotonic() - start) * 1000)
                if response.status >= 400:
                    message = body.decode("utf-8", errors="ignore")
                    raise ProviderAPIError(
                        provider=self.provider_name,
                        status_code=response.status,
                        message="OpenAI Video API returned an error",
                        error_type="http",
                        error_code=str(response.status),
                        provider_message=message,
                        retryable=response.status >= 500,
                        duration_ms=duration_ms,
                    )
                return body, dict(response.headers)
        except asyncio.TimeoutError as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            raise ProviderAPIError(
                provider=self.provider_name,
                status_code=0,
                message="OpenAI Video request timed out",
                error_type="timeout",
                provider_message="",
                retryable=False,
                duration_ms=duration_ms,
            ) from exc

    def _merge_params(self, params: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        merged: Dict[str, Any] = {}
        if self._default_params:
            merged.update(self._default_params)
        if params:
            for key, value in params.items():
                if value is None:
                    continue
                merged[key] = value
        return merged or None

    def _normalise_error(self, error: ProviderAPIError) -> ProviderAPIError:
        raw_body = error.provider_message or ""
        error_code: Optional[str] = None
        error_message: Optional[str] = None
        error_type: Optional[str] = None
        try:
            payload = json.loads(raw_body) if raw_body else {}
        except (TypeError, ValueError):
            payload = {"raw": raw_body}
        if isinstance(payload, Mapping):
            nested = payload.get("error")
            if isinstance(nested, Mapping):
                error_code = nested.get("code") or nested.get("error_code")
                error_message = nested.get("message") or nested.get("detail")
                error_type = nested.get("type") or nested.get("error_type")
            else:
                if isinstance(payload.get("code"), str):
                    error_code = payload.get("code")
                if isinstance(payload.get("message"), str):
                    error_message = payload.get("message")
        log.warning(
            "openai.video.error status=%s error_code=%s error_type=%s error_message=%s body=%s",
            error.status_code,
            error_code or error.error_code,
            error_type or error.error_type,
            error_message or str(error),
            raw_body,
        )
        resolution = self._error_handler.resolve(
            status_code=error.status_code,
            error_code=error_code or error.error_code,
            error_message=error_message or str(error),
        )
        if resolution:
            self._error_handler.log_admin_hint(resolution)
            return self._error_handler.apply(
                base_error=error,
                resolution=resolution,
                raw_body=raw_body,
                fallback_code=error_code or error.error_code,
            )
        return ProviderAPIError(
            provider=error.provider,
            status_code=error.status_code,
            message=error_message or str(error) or "OpenAI Video API error",
            error_type=error_type or error.error_type or "unknown",
            error_code=error_code or error.error_code,
            provider_message=raw_body or error.provider_message,
            retryable=error.retryable,
            duration_ms=error.duration_ms,
        )

    def _resolve_job_id(self, payload: Mapping[str, Any]) -> Optional[str]:
        if not isinstance(payload, Mapping):
            return None
        for key in ("id", "job_id", "video_id"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        response = payload.get("response")
        if isinstance(response, Mapping):
            candidate = response.get("id")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, Mapping):
                    candidate = item.get("id")
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
        return None

    def _extract_status(self, payload: Mapping[str, Any]) -> str:
        if not isinstance(payload, Mapping):
            return "unknown"
        raw_status = payload.get("status")
        if isinstance(raw_status, str) and raw_status.strip():
            status = raw_status.strip().lower()
        else:
            response = payload.get("response")
            if isinstance(response, Mapping):
                status = str(response.get("status") or "").lower()
            else:
                status = ""
        if status in {"queued", "pending", "in_progress", "processing", "running"}:
            return "running"
        if status in {"completed", "succeeded", "ready", "finished"}:
            return "completed"
        if status in {"failed", "errored", "error", "cancelled", "canceled"}:
            return "failed"
        return status or "unknown"

    def _extract_error(self, payload: Mapping[str, Any]) -> Optional[str]:
        if not isinstance(payload, Mapping):
            return None
        error = payload.get("error")
        if isinstance(error, Mapping):
            message = error.get("message") or error.get("detail") or error.get("code")
            if isinstance(message, str) and message.strip():
                return message.strip()
            return json.dumps(error, ensure_ascii=False)
        if isinstance(error, str) and error.strip():
            return error.strip()
        last_error = payload.get("last_error")
        if isinstance(last_error, Mapping):
            message = last_error.get("message") or last_error.get("code")
            if isinstance(message, str) and message.strip():
                return message.strip()
        return None

    def _extract_assets(self, payload: Mapping[str, Any]) -> Dict[str, str]:
        assets: Dict[str, str] = {}
        if not isinstance(payload, Mapping):
            return assets
        container = payload.get("assets")
        if isinstance(container, Mapping):
            for key, value in container.items():
                if isinstance(value, Mapping):
                    url = _extract_url(value)
                    if url:
                        assets[str(key)] = url
        for entry in _iter_nodes(payload):
            if not isinstance(entry, Mapping):
                continue
            mime = _guess_mime(entry) or ""
            is_video_type = str(entry.get("type") or "").lower() in {"video", "output_video", "video_asset"}
            is_video_mime = any(mime.startswith(prefix) for prefix in _VIDEO_MIME_PREFIXES)
            url = _extract_url(entry)
            if not url and not is_video_type and not is_video_mime:
                continue
            if not url:
                base64_data = _extract_base64(entry)
                if base64_data:
                    assets[str(entry.get("id") or f"asset_{len(assets)}")] = base64_data
                    continue
            if url:
                key = entry.get("id") or entry.get("name") or entry.get("asset_id")
                if isinstance(key, str) and key.strip():
                    assets[key.strip()] = url
                else:
                    assets[f"asset_{len(assets)}"] = url
        return assets


__all__ = ["OpenAIVideoClient"]

