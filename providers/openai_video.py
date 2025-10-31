"""Client for the OpenAI Videos API (Sora 2 models)."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from config import Config

from .api_error_handler import ApiErrorHandler
from .base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
    _mask_secret,
    _sanitize_headers,
    _serialise_for_log,
)


_PROVIDERS_LOG = logging.getLogger("providers")
log = _PROVIDERS_LOG.getChild("openai_video") if _PROVIDERS_LOG else logging.getLogger(__name__)


_DEFAULT_VIDEO_MODELS: Tuple[str, ...] = ("sora-2", "sora-2-fast", "sora-2-pro")
_VIDEO_MIME_PREFIXES = ("video/", "application/octet-stream")

ALLOWED_CREATE_FIELDS: FrozenSet[str] = frozenset({
    "prompt",
    "input_reference",
    "model",
    "seconds",
    "size",
})
ALLOWED_REMIX_FIELDS: FrozenSet[str] = frozenset({"prompt"})
ALLOWED_LIST_FIELDS: FrozenSet[str] = frozenset({"after", "limit", "order"})
ALLOWED_RETRIEVE_FIELDS: FrozenSet[str] = frozenset()
ALLOWED_CONTENT_FIELDS: FrozenSet[str] = frozenset({"variant"})
_LOG_PAYLOAD_LIMIT = 3072


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


@dataclass(frozen=True)
class PreparedCreateRequest:
    request: Dict[str, Any]
    extras: Dict[str, Any]
    dropped_fields: Tuple[str, ...]
    correlation_id: Optional[str]


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
        org_id = (config.openai_org_id or "").strip()
        if org_id:
            default_headers["OpenAI-Organization"] = org_id
        super().__init__(
            config=config,
            base_url=base_url,
            api_key=api_key,
            provider_name="openai-video",
            default_headers=default_headers,
        )
        self._beta_header = beta_header
        self._organization_id = org_id
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
        self._last_request: Dict[str, Any] = {}
        self._diagnostic_history: Deque[Dict[str, Any]] = deque(maxlen=10)
        self._error_snapshots: Deque[Dict[str, Any]] = deque(maxlen=10)
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

    @property
    def last_request(self) -> Dict[str, Any]:
        return dict(self._last_request)

    @property
    def diagnostic_history(self) -> List[Dict[str, Any]]:
        history: List[Dict[str, Any]] = []
        for entry in self._diagnostic_history:
            item = dict(entry)
            dropped = item.get("dropped_fields")
            if isinstance(dropped, (list, tuple, set)):
                item["dropped_fields"] = tuple(dropped)
            error_info = item.get("error")
            if isinstance(error_info, Mapping):
                item["error"] = dict(error_info)
            history.append(item)
        return history

    def get_diagnostics(self) -> Dict[str, Any]:
        safe_headers = _sanitize_headers(self._build_headers())
        return {
            "api_version": self._api_version or "",
            "beta_header": self._beta_header,
            "organization_id": self._organization_id,
            "headers": safe_headers,
            "error_snapshots": [dict(snapshot) for snapshot in self._error_snapshots],
        }

    async def create(
        self,
        *,
        prompt: str,
        settings: Optional[Mapping[str, Any]] = None,
        payload: Optional[Mapping[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        return_meta: bool = False,
        prepared: Optional[PreparedCreateRequest] = None,
    ) -> Any:
        if prepared is None:
            prepared = self.prepare_create_request(
                prompt=prompt,
                settings=settings,
                payload=payload,
            )
        request = dict(prepared.request)
        if "prompt" not in request:
            prompt_text = str(prompt or "").strip()
            if prompt_text:
                request["prompt"] = prompt_text
        dropped_fields = list(prepared.dropped_fields)
        correlation_id = prepared.correlation_id
        path = self._videos_path()
        headers = self._build_headers()
        if idempotency_key:
            headers.setdefault("Idempotency-Key", idempotency_key)
        safe_headers = _sanitize_headers(headers)
        corr_label = correlation_id or idempotency_key or ""
        self._record_last_request(
            method="POST",
            path=path,
            payload=request,
            headers=safe_headers,
            dropped_fields=dropped_fields,
            correlation_id=corr_label,
        )
        self._log_request(
            method="POST",
            path=path,
            payload=request,
            headers=safe_headers,
            dropped_fields=dropped_fields,
            correlation_id=corr_label,
        )
        data, status_code, duration_ms = await self._request_json(
            "POST",
            path,
            json=request,
            idempotency_key=idempotency_key,
            diagnostic={"dropped_fields": dropped_fields},
        )
        log.info(
            "openai.video.create status=%s duration_ms=%s model=%s", status_code, duration_ms, request.get("model")
        )
        return (data, status_code, duration_ms) if return_meta else data

    def prepare_create_request(
        self,
        *,
        prompt: str,
        settings: Optional[Mapping[str, Any]] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> PreparedCreateRequest:
        request, extras, dropped_fields, correlation_id = self._prepare_create_request(
            prompt=prompt,
            settings=settings,
            payload=payload,
        )
        return PreparedCreateRequest(
            request=dict(request),
            extras=dict(extras),
            dropped_fields=tuple(sorted(set(dropped_fields))),
            correlation_id=correlation_id,
        )

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
        request, dropped_fields, correlation_id = self._build_remix_payload(
            prompt=prompt or "",
            settings=dict(settings or {}),
            payload=dict(payload or {}),
            allow_empty_prompt=True,
        )
        path = f"{self._videos_path()}/{video_id}/remix"
        headers = self._build_headers()
        if idempotency_key:
            headers.setdefault("Idempotency-Key", idempotency_key)
        safe_headers = _sanitize_headers(headers)
        corr_label = correlation_id or idempotency_key or ""
        self._record_last_request(
            method="POST",
            path=path,
            payload=request,
            headers=safe_headers,
            dropped_fields=dropped_fields,
            correlation_id=corr_label,
        )
        self._log_request(
            method="POST",
            path=path,
            payload=request,
            headers=safe_headers,
            dropped_fields=dropped_fields,
            correlation_id=corr_label,
        )
        data, status_code, duration_ms = await self._request_json(
            "POST",
            path,
            json=request,
            idempotency_key=idempotency_key,
            diagnostic={"dropped_fields": dropped_fields},
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
        raw_params: Dict[str, Any] = {
            "limit": limit,
            "order": order,
            "after": after,
        }
        if before is not None:
            raw_params["before"] = before
        dropped_fields: List[str] = []
        filtered_params = self._filter_allowed_fields(
            raw_params, ALLOWED_LIST_FIELDS, dropped_fields, source="params"
        )
        params: Dict[str, Any] = {}
        if "limit" in filtered_params:
            try:
                params["limit"] = int(filtered_params["limit"])
            except (TypeError, ValueError):
                dropped_fields.append("params.limit")
        if "order" in filtered_params and filtered_params["order"] is not None:
            params["order"] = str(filtered_params["order"])
        if "after" in filtered_params and filtered_params["after"] is not None:
            params["after"] = str(filtered_params["after"])
        self._record_last_request(
            method="GET",
            path=self._videos_path(),
            params=params or None,
            dropped_fields=dropped_fields,
            correlation_id="",
        )
        if dropped_fields:
            log.warning(
                "openai.video.list dropped_fields=%s params=%s",
                sorted(set(dropped_fields)),
                _serialise_for_log(params or {}, limit=512),
            )
        data, status_code, duration_ms = await self._request_json(
            "GET",
            self._videos_path(),
            params=params or None,
            diagnostic={"dropped_fields": dropped_fields},
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
        self._record_last_request(
            method="GET",
            path=f"{self._videos_path()}/{video_id}",
            params=None,
            dropped_fields=[],
            correlation_id="",
        )
        data, status_code, duration_ms = await self._request_json(
            "GET",
            f"{self._videos_path()}/{video_id}",
            diagnostic={"dropped_fields": []},
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
        raw_params = dict(params or {})
        dropped_fields: List[str] = []
        filtered_params = self._filter_allowed_fields(
            raw_params, ALLOWED_CONTENT_FIELDS, dropped_fields, source="params"
        )
        query: Dict[str, Any] = {}
        if "variant" in filtered_params and filtered_params["variant"] is not None:
            query["variant"] = str(filtered_params["variant"])
        self._record_last_request(
            method="GET",
            path=path,
            params=query or None,
            dropped_fields=dropped_fields,
            correlation_id="",
        )
        if dropped_fields:
            log.warning(
                "openai.video.content dropped_fields=%s params=%s video_id=%s asset_id=%s",
                sorted(set(dropped_fields)),
                _serialise_for_log(query or {}, limit=512),
                video_id,
                asset_id,
            )
        body, headers = await self._request_binary(
            "GET",
            path,
            params=query or None,
            diagnostic={"dropped_fields": dropped_fields},
        )
        log.debug(
            "openai.video.content video_id=%s asset_id=%s bytes=%s",
            video_id,
            asset_id,
            len(body),
        )
        return body, headers

    async def list_models(self) -> List[str]:
        data, status_code, duration_ms = await self._request_json(
            "GET", "/models", diagnostic={"dropped_fields": []}
        )
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
    ) -> Tuple[Dict[str, Any], List[str], Optional[str]]:
        request, _extras, dropped_fields, corr_id = self._prepare_create_request(
            prompt=prompt,
            settings=settings,
            payload=payload,
            allow_empty_prompt=allow_empty_prompt,
        )
        return request, sorted(set(dropped_fields)), corr_id

    def _prepare_create_request(
        self,
        *,
        prompt: str,
        settings: Optional[Mapping[str, Any]],
        payload: Optional[Mapping[str, Any]],
        allow_empty_prompt: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], Optional[str]]:
        original_settings = dict(settings or {})
        original_payload = dict(payload or {})
        prepared_settings, prepared_payload, dropped_fields, corr_id = self._prepare_request_data(
            original_settings,
            original_payload,
        )
        request: Dict[str, Any] = {}
        request.update(
            self._filter_allowed_fields(prepared_payload, ALLOWED_CREATE_FIELDS, dropped_fields, source="payload")
        )
        allowed_settings = self._filter_allowed_fields(
            prepared_settings, ALLOWED_CREATE_FIELDS, dropped_fields, source="settings"
        )
        for key, value in allowed_settings.items():
            request.setdefault(key, value)
        prompt_text = str(request.get("prompt") or prompt or "").strip()
        if not prompt_text and not allow_empty_prompt:
            raise ValueError("prompt must not be empty")
        if prompt_text:
            request["prompt"] = prompt_text
        model_value = request.get("model") or self._default_model
        model_text = str(model_value or self._default_model).strip()
        if not model_text:
            model_text = self._default_model
        request["model"] = model_text
        self._supported_models.add(model_text)
        if "seconds" in request:
            normalised_seconds = self._normalise_seconds(request["seconds"])
            if normalised_seconds is not None:
                request["seconds"] = normalised_seconds
            else:
                request.pop("seconds", None)
        if "size" in request:
            size_value = str(request["size"]).strip()
            if size_value:
                request["size"] = size_value
            else:
                request.pop("size", None)
        if "input_reference" in request and request["input_reference"] in {None, ""}:
            request.pop("input_reference", None)
        extras = self._extract_local_extras(
            original_settings=original_settings,
            original_payload=original_payload,
            request=request,
        )
        return request, extras, dropped_fields, corr_id

    def _build_remix_payload(
        self,
        *,
        prompt: str,
        settings: Dict[str, Any],
        payload: Dict[str, Any],
        allow_empty_prompt: bool = False,
    ) -> Tuple[Dict[str, Any], List[str], Optional[str]]:
        prepared_settings, prepared_payload, dropped_fields, corr_id = self._prepare_request_data(
            settings, payload
        )
        allowed_payload = self._filter_allowed_fields(
            prepared_payload, ALLOWED_REMIX_FIELDS, dropped_fields, source="payload"
        )
        allowed_settings = self._filter_allowed_fields(
            prepared_settings, ALLOWED_REMIX_FIELDS, dropped_fields, source="settings"
        )
        prompt_text = str(
            allowed_payload.get("prompt")
            or allowed_settings.get("prompt")
            or prompt
            or ""
        ).strip()
        if not prompt_text and not allow_empty_prompt:
            raise ValueError("prompt must not be empty")
        request: Dict[str, Any] = {}
        if prompt_text:
            request["prompt"] = prompt_text
        return request, sorted(set(dropped_fields)), corr_id

    def _record_last_request(
        self,
        *,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        dropped_fields: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "method": method,
            "path": path,
            "endpoint": path,
            "dropped_fields": tuple(sorted(set(dropped_fields or []))),
            "correlation_id": correlation_id or "",
            "timestamp": time.monotonic(),
        }
        if payload is not None:
            entry["payload"] = payload
        if params is not None:
            entry["params"] = params
        if headers is not None:
            entry["headers"] = headers
        self._last_request = entry
        self._diagnostic_history.append(entry)

    def _update_last_diagnostic(self, **updates: Any) -> None:
        if not self._last_request:
            return
        normalised = dict(updates)
        if "dropped_fields" in normalised:
            fields = normalised.get("dropped_fields") or ()
            if isinstance(fields, (list, tuple, set)):
                normalised["dropped_fields"] = tuple(
                    sorted({str(field) for field in fields if field})
                )
            elif isinstance(fields, str):
                normalised["dropped_fields"] = (fields,) if fields else ()
            else:
                normalised.pop("dropped_fields", None)
        normalised.setdefault("endpoint", self._last_request.get("path", ""))
        normalised.setdefault("method", self._last_request.get("method", ""))
        if "status" not in normalised and "status_code" in normalised:
            normalised["status"] = normalised["status_code"]
        if "request_id" in normalised:
            normalised["request_id"] = str(normalised.get("request_id") or "")
        self._last_request.update(normalised)
        if self._diagnostic_history:
            self._diagnostic_history[-1].update(normalised)
        error_info = normalised.get("error")
        if error_info:
            if isinstance(error_info, Mapping):
                snapshot = dict(error_info)
            else:
                snapshot = {"message": str(error_info)}
            status_code = normalised.get("status_code")
            if status_code is not None and "status_code" not in snapshot:
                snapshot["status_code"] = status_code
            self._error_snapshots.append(snapshot)

    def _record_response_history(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        duration_ms: int,
        request_id: str,
        diagnostic: Optional[Mapping[str, Any]] = None,
    ) -> None:
        dropped_fields: Any = ()
        if diagnostic and diagnostic.get("dropped_fields"):
            dropped_fields = diagnostic.get("dropped_fields")
        elif self._last_request.get("dropped_fields"):
            dropped_fields = self._last_request["dropped_fields"]
        updates = {
            "method": method,
            "endpoint": path,
            "status_code": status_code,
            "status": status_code,
            "duration_ms": duration_ms,
            "request_id": request_id or "",
            "dropped_fields": dropped_fields,
        }
        self._update_last_diagnostic(**updates)

    def _log_request(
        self,
        *,
        method: str,
        path: str,
        payload: Optional[Mapping[str, Any]],
        headers: Mapping[str, str],
        dropped_fields: List[str],
        correlation_id: str,
    ) -> None:
        log.warning(
            "openai.video.request corr_id=%s method=%s endpoint=%s payload=%s headers=%s dropped_fields=%s",
            correlation_id,
            method,
            path,
            _serialise_for_log(payload or {}, limit=_LOG_PAYLOAD_LIMIT),
            _serialise_for_log(headers, limit=1024),
            sorted(set(dropped_fields)),
        )

    def _prepare_request_data(
        self,
        settings: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], Optional[str]]:
        dropped_fields: List[str] = []
        settings_copy = dict(settings or {})
        payload_copy = dict(payload or {})
        corr_id = self._extract_correlation_id(settings_copy, payload_copy, dropped_fields)
        self._apply_aliases(settings_copy, "settings", dropped_fields)
        self._apply_aliases(payload_copy, "payload", dropped_fields)
        return settings_copy, payload_copy, dropped_fields, corr_id

    def _extract_local_extras(
        self,
        *,
        original_settings: Mapping[str, Any],
        original_payload: Mapping[str, Any],
        request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        extras: Dict[str, Any] = {}
        seconds_value: Any = request.get("seconds")
        if seconds_value is None:
            seconds_value = self._search_nested(original_settings, ("seconds", "duration", "duration_seconds", "duration_sec"))
            if seconds_value is None:
                seconds_value = self._search_nested(
                    original_payload, ("seconds", "duration", "duration_seconds", "duration_sec")
                )
        normalised_seconds = self._normalise_seconds(seconds_value)
        if normalised_seconds is not None:
            try:
                numeric_seconds = int(float(normalised_seconds))
            except (TypeError, ValueError):
                numeric_seconds = None
            if numeric_seconds and numeric_seconds > 0:
                extras["duration_seconds"] = numeric_seconds
            else:
                extras["duration_seconds"] = normalised_seconds
        aspect_ratio = self._search_nested(
            original_settings, ("aspect_ratio", "aspectRatio")
        )
        if aspect_ratio is None:
            aspect_ratio = self._search_nested(original_payload, ("aspect_ratio", "aspectRatio"))
        if isinstance(aspect_ratio, str):
            aspect_value = aspect_ratio.strip()
        elif aspect_ratio is not None:
            aspect_value = str(aspect_ratio)
        else:
            aspect_value = ""
        if aspect_value:
            extras["aspect_ratio"] = aspect_value
        size_value: Any = request.get("size")
        if not size_value:
            size_value = self._search_nested(original_settings, ("size",))
            if not size_value:
                size_value = self._search_nested(original_payload, ("size",))
        if isinstance(size_value, str):
            size_text = size_value.strip()
        elif size_value is not None:
            size_text = str(size_value)
        else:
            size_text = ""
        if size_text:
            extras.setdefault("size", size_text)
        return extras

    def _filter_allowed_fields(
        self,
        data: Mapping[str, Any],
        allowed: FrozenSet[str],
        dropped_fields: List[str],
        *,
        source: str,
    ) -> Dict[str, Any]:
        filtered: Dict[str, Any] = {}
        for key, value in data.items():
            if value is None:
                continue
            if key in allowed:
                filtered[key] = value
            else:
                dropped_fields.append(f"{source}.{key}")
        return filtered

    def _extract_correlation_id(
        self,
        settings: Dict[str, Any],
        payload: Dict[str, Any],
        dropped_fields: List[str],
    ) -> Optional[str]:
        correlation_id: Optional[str] = None
        for container, label in ((settings, "settings"), (payload, "payload")):
            for key in ("corr_id", "correlation_id", "correlationId"):
                if key in container:
                    value = container.pop(key)
                    if value and correlation_id is None:
                        correlation_id = str(value)
                    dropped_fields.append(f"{label}.{key}")
        return correlation_id

    def _apply_aliases(
        self,
        data: Dict[str, Any],
        label: str,
        dropped_fields: List[str],
    ) -> None:
        alias_map = {
            "duration": "seconds",
            "duration_sec": "seconds",
            "duration_seconds": "seconds",
        }
        for alias, target in alias_map.items():
            if alias in data:
                value = data.pop(alias)
                if value is not None and target not in data:
                    data[target] = value
                dropped_fields.append(f"{label}.{alias}")

    def _normalise_seconds(self, value: Any) -> Optional[str]:
        if isinstance(value, (int, float)):
            if value <= 0:
                return None
            return str(int(value))
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.isdigit():
                return str(int(text))
            return text
        return None

    def _search_nested(self, data: Any, keys: Iterable[str]) -> Optional[Any]:
        if isinstance(data, Mapping):
            for key in keys:
                if key in data:
                    value = data[key]
                    if value not in (None, ""):
                        return value
            for value in data.values():
                nested = self._search_nested(value, keys)
                if nested is not None:
                    return nested
        elif isinstance(data, (list, tuple)):
            for item in data:
                nested = self._search_nested(item, keys)
                if nested is not None:
                    return nested
        return None

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        idempotency_key: Optional[str] = None,
        diagnostic: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], int, int]:
        params = kwargs.pop("params", None)
        merged_params = self._merge_params(params)
        if merged_params:
            kwargs["params"] = merged_params
        self._set_request_context(diagnostic)
        try:
            data, status_code, duration_ms = await super()._request(
                method, path, idempotency_key=idempotency_key, **kwargs
            )
        except ProviderAPIError as exc:
            meta = getattr(self, "_last_response_meta", {}) or {}
            self._record_response_history(
                method=method,
                path=path,
                status_code=exc.status_code,
                duration_ms=int(meta.get("duration_ms") or exc.duration_ms or 0),
                request_id=meta.get("request_id", ""),
                diagnostic=diagnostic,
            )
            raise self._normalise_error(exc) from exc
        meta = getattr(self, "_last_response_meta", {}) or {}
        self._record_response_history(
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            request_id=meta.get("request_id", ""),
            diagnostic=diagnostic,
        )
        return data, status_code, duration_ms

    async def _request_binary(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        diagnostic: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bytes, Dict[str, str]]:
        session = await self._ensure_session()
        url = f"{self._base_url}{path}"
        headers = self._build_headers()
        merged_params = self._merge_params(params)
        if merged_params is None:
            merged_params = {}
        diagnostic_fields = tuple(
            sorted(set(diagnostic.get("dropped_fields", ())))
        ) if diagnostic else ()
        if diagnostic_fields:
            self._update_last_diagnostic(dropped_fields=diagnostic_fields)
        log.debug(
            "openai.video.binary_request method=%s url=%s params=%s dropped_fields=%s",
            method,
            url,
            _serialise_for_log(dict(merged_params), limit=512),
            diagnostic_fields,
        )
        start = time.monotonic()
        try:
            async with session.request(
                method, url, headers=headers, params=dict(merged_params)
            ) as response:
                body = await response.read()
                duration_ms = int((time.monotonic() - start) * 1000)
                self._record_response_meta(
                    method=method,
                    url=url,
                    status=response.status,
                    duration_ms=duration_ms,
                    headers=dict(response.headers),
                    context=diagnostic,
                )
                meta = getattr(self, "_last_response_meta", {}) or {}
                if response.status >= 400:
                    message = body.decode("utf-8", errors="ignore")
                    api_error = ProviderAPIError(
                        provider=self.provider_name,
                        status_code=response.status,
                        message="OpenAI Video API returned an error",
                        error_type="http",
                        error_code=str(response.status),
                        provider_message=message,
                        retryable=response.status >= 500,
                        duration_ms=duration_ms,
                    )
                    self._record_response_history(
                        method=method,
                        path=path,
                        status_code=response.status,
                        duration_ms=duration_ms,
                        request_id=meta.get("request_id", ""),
                        diagnostic=diagnostic,
                    )
                    self._update_last_diagnostic(
                        error={
                            "status_code": api_error.status_code,
                            "error_code": api_error.error_code,
                            "error_type": api_error.error_type,
                            "message": str(api_error),
                        },
                    )
                    raise api_error
                self._record_response_history(
                    method=method,
                    path=path,
                    status_code=response.status,
                    duration_ms=duration_ms,
                    request_id=meta.get("request_id", ""),
                    diagnostic=diagnostic,
                )
                return body, dict(response.headers)
        except asyncio.TimeoutError as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            api_error = ProviderAPIError(
                provider=self.provider_name,
                status_code=0,
                message="OpenAI Video request timed out",
                error_type="timeout",
                provider_message="",
                retryable=False,
                duration_ms=duration_ms,
            )
            self._update_last_diagnostic(
                error={
                    "status_code": api_error.status_code,
                    "error_code": api_error.error_code,
                    "error_type": api_error.error_type,
                    "message": str(api_error),
                },
            )
            self._record_response_history(
                method=method,
                path=path,
                status_code=api_error.status_code,
                duration_ms=duration_ms,
                request_id=(self._last_response_meta or {}).get("request_id", ""),
                diagnostic=diagnostic,
            )
            raise api_error from exc
        except ProviderAPIError:
            raise
        except Exception:
            self._record_response_history(
                method=method,
                path=path,
                status_code=0,
                duration_ms=int((time.monotonic() - start) * 1000),
                request_id=(self._last_response_meta or {}).get("request_id", ""),
                diagnostic=diagnostic,
            )
            raise

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
        offending_params: set[str] = set()
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
                for key in (
                    "param",
                    "parameter",
                    "field",
                    "params",
                    "parameters",
                    "unknown",
                    "unknown_params",
                    "unknown_parameters",
                ):
                    if key in nested:
                        self._collect_offending_params(nested[key], offending_params)
            else:
                if isinstance(payload.get("code"), str):
                    error_code = payload.get("code")
                if isinstance(payload.get("message"), str):
                    error_message = payload.get("message")
            for key in (
                "param",
                "params",
                "parameters",
                "unknown",
                "unknown_params",
                "unknown_parameters",
            ):
                if key in payload:
                    self._collect_offending_params(payload[key], offending_params)
        log.warning(
            "openai.video.error status=%s error_code=%s error_type=%s error_message=%s body=%s",
            error.status_code,
            error_code or error.error_code,
            error_type or error.error_type,
            error_message or str(error),
            raw_body,
        )
        message_candidates = [error_message or "", raw_body]
        unknown_parameter = any(
            "unknown parameter" in str(candidate).lower() for candidate in message_candidates if candidate
        )
        if unknown_parameter:
            for candidate in message_candidates:
                self._collect_offending_params(candidate, offending_params)
            dropped = tuple(self._last_request.get("dropped_fields", ()))
            params_text = ", ".join(sorted(offending_params)) if offending_params else "—"
            dropped_text = ", ".join(dropped) if dropped else "—"
            message = (
                f"Unknown parameter(s) reported by OpenAI: {params_text}. "
                f"Dropped fields: {dropped_text}."
            )
            error_snapshot = {
                "status_code": error.status_code,
                "error_code": "invalid_request",
                "error_type": "invalid_request",
                "message": message,
                "provider_message": raw_body,
                "offending_parameters": tuple(sorted(offending_params)),
            }
            self._update_last_diagnostic(
                status_code=error.status_code, error=error_snapshot
            )
            return ProviderAPIError(
                provider=error.provider,
                status_code=error.status_code,
                message=message,
                error_type="invalid_request",
                error_code="invalid_request",
                provider_message=raw_body or error.provider_message,
                retryable=False,
                duration_ms=error.duration_ms,
            )
        resolution = self._error_handler.resolve(
            status_code=error.status_code,
            error_code=error_code or error.error_code,
            error_message=error_message or str(error),
        )
        if resolution:
            self._error_handler.log_admin_hint(resolution)
            resolved = self._error_handler.apply(
                base_error=error,
                resolution=resolution,
                raw_body=raw_body,
                fallback_code=error_code or error.error_code,
            )
            self._update_last_diagnostic(
                status_code=resolved.status_code,
                error={
                    "status_code": resolved.status_code,
                    "error_code": resolved.error_code,
                    "error_type": resolved.error_type,
                    "message": str(resolved),
                    "provider_message": raw_body,
                },
            )
            return resolved
        message = error_message or str(error) or "OpenAI Video API error"
        fallback_error = ProviderAPIError(
            provider=error.provider,
            status_code=error.status_code,
            message=message,
            error_type=error_type or error.error_type or "unknown",
            error_code=error_code or error.error_code,
            provider_message=raw_body or error.provider_message,
            retryable=error.retryable,
            duration_ms=error.duration_ms,
        )
        self._update_last_diagnostic(
            status_code=fallback_error.status_code,
            error={
                "status_code": fallback_error.status_code,
                "error_code": fallback_error.error_code,
                "error_type": fallback_error.error_type,
                "message": str(fallback_error),
                "provider_message": raw_body,
            },
        )
        return fallback_error

    def _collect_offending_params(self, value: Any, result: set[str]) -> None:
        if value is None:
            return
        if isinstance(value, str):
            matches = re.findall(r"['\"]([^'\"]+)['\"]", value)
            if matches:
                for item in matches:
                    candidate = item.strip()
                    if candidate:
                        result.add(candidate)
                return
            for token in re.split(r"[,\s]+", value):
                candidate = token.strip(" '\"")
                if candidate:
                    result.add(candidate)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                self._collect_offending_params(item, result)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                self._collect_offending_params(item, result)

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


__all__ = ["OpenAIVideoClient", "PreparedCreateRequest"]

