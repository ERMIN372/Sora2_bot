"""Shared provider client utilities."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import urlparse

import aiohttp

from config import Config

log = logging.getLogger(__name__)

_SENSITIVE_HEADERS = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
    "x-anthropic-api-key",
}


def _mask_secret(value: Any, visible: int = 6) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) <= visible:
        return "*" * len(text)
    return f"{text[:visible]}{'*' * (len(text) - visible)}"


def _sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    safe: Dict[str, str] = {}
    for key, value in headers.items():
        if isinstance(value, str) and key.lower() in _SENSITIVE_HEADERS:
            safe[key] = _mask_secret(value)
        else:
            safe[key] = value
    return safe


def _join_url(base: str, *parts: str) -> str:
    base_clean = (base or "").rstrip("/")
    segments: List[str] = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip("/")
        if not text:
            continue
        segments.extend(segment for segment in text.split("/") if segment)

    if segments:
        first_segment = segments[0]
        if first_segment in {"v1", "v1beta"}:
            if base_clean.endswith("/v1"):
                base_clean = base_clean[: -len("/v1")]
            elif base_clean.endswith("/v1beta"):
                base_clean = base_clean[: -len("/v1beta")]
        joined = "/".join(segments)
        url = f"{base_clean}/{joined}" if base_clean else joined
    else:
        url = base_clean

    if not url:
        return ""

    # Guard against accidental double version segments specific to the responses API.
    if "/v1/v1beta/" in url:
        url = url.replace("/v1/v1beta/", "/v1beta/", 1)
    if url.endswith("/v1/v1beta"):
        url = url[: -len("/v1/v1beta")] + "/v1beta"
    if url.endswith("/v1/v1beta/responses"):
        url = url[: -len("/v1/v1beta/responses")] + "/v1beta/responses"

    return url


_DOWNLOAD_ALLOWED_SCHEMES = {"https"}
_DOWNLOAD_BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "[::1]", "metadata.google.internal"}


def validate_download_url(url: str) -> None:
    """Raise ValueError if *url* points to a non-HTTPS or internal target."""
    parsed = urlparse(url)
    if parsed.scheme not in _DOWNLOAD_ALLOWED_SCHEMES:
        raise ValueError(f"Blocked non-HTTPS download URL: scheme={parsed.scheme!r}")
    hostname = (parsed.hostname or "").lower()
    if hostname in _DOWNLOAD_BLOCKED_HOSTS or hostname.endswith(".internal"):
        raise ValueError(f"Blocked download to internal host: {hostname!r}")


def _extract_request_id(headers: Mapping[str, str]) -> Optional[str]:
    if not headers:
        return None
    lowered: Dict[str, Any] = {}
    for key, value in headers.items():
        try:
            lowered[str(key).lower()] = value
        except Exception:  # pragma: no cover - defensive
            continue
    for candidate in (
        "request-id",
        "x-request-id",
        "x-requestid",
        "openai-request-id",
        "x-openai-request-id",
        "openai-requestid",
    ):
        header_value = lowered.get(candidate)
        if header_value:
            return str(header_value)
    return None


def iter_nodes(value: Any) -> Iterable[Dict[str, Any]]:
    """Walk a nested dict/list structure and yield every dict node."""
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


_URL_KEYS = ("download_url", "url", "video_url", "asset_url", "content_url")
_NESTED_KEYS = ("file", "video", "asset")


def extract_url(entry: Mapping[str, Any]) -> Optional[str]:
    """Extract a download URL from a provider response entry."""
    for key in _URL_KEYS:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for nested_key in _NESTED_KEYS:
        nested = entry.get(nested_key)
        if isinstance(nested, Mapping):
            for key in _URL_KEYS:
                value = nested.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


async def stream_response_to_file(response: Any, file_path: Path) -> int:
    """Stream an aiohttp response body to *file_path*, return bytes written.

    On error the parent directory is cleaned up to prevent temp-dir leaks.
    """
    import shutil

    size = 0
    try:
        with open(file_path, "wb") as output:
            if hasattr(response, "aiter_bytes"):
                async for chunk in response.aiter_bytes():
                    if chunk:
                        output.write(chunk)
                        size += len(chunk)
            else:
                async for chunk in response.content.iter_chunked(65536):
                    if chunk:
                        output.write(chunk)
                        size += len(chunk)
    except BaseException:
        # Clean up the temp directory to avoid disk leaks.
        parent = file_path.parent
        if parent != Path("/") and parent.exists():
            shutil.rmtree(parent, ignore_errors=True)
        raise
    return size


def _serialise_for_log(data: Any, *, limit: int = 3072) -> str:
    if data is None:
        return ""
    try:
        serialised = json.dumps(data, ensure_ascii=False)
    except (TypeError, ValueError):
        serialised = str(data)
    if len(serialised) > limit:
        return f"{serialised[:limit]}…(+{len(serialised) - limit} chars)"
    return serialised


def _truncate(text: Optional[str], limit: int = 3072) -> str:
    if not text:
        return ""
    if len(text) > limit:
        return f"{text[:limit]}…(+{len(text) - limit} chars)"
    return text


@dataclass(slots=True)
class ProviderRequestMeta:
    status_code: int
    duration_ms: int


@dataclass(slots=True)
class ProviderJobSubmission(ProviderRequestMeta):
    job_id: str
    data: Dict[str, Any]


@dataclass(slots=True)
class ProviderJobStatus(ProviderRequestMeta):
    job_id: str
    status: str
    assets: Dict[str, str]
    error: Optional[str]
    data: Dict[str, Any]


class ProviderAPIError(RuntimeError):
    """Structured representation of provider API errors."""

    def __init__(
        self,
        *,
        provider: str,
        status_code: int,
        message: str,
        error_type: Optional[str] = None,
        error_code: Optional[str] = None,
        provider_message: Optional[str] = None,
        retryable: bool = False,
        duration_ms: Optional[int] = None,
        code: Optional[str] = None,
        body: Optional[str] = None,
        request_id: Optional[str] = None,
        status: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code
        self.provider_message = provider_message
        self.retryable = retryable
        self.duration_ms = duration_ms
        self.code = code or error_code
        self.body = body or provider_message
        self.request_id = request_id
        self.status = status if status is not None else status_code


class ModelUnavailable(ProviderAPIError):
    """Error raised when a requested model is unavailable for the caller."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int = 404,
        error_type: Optional[str] = None,
        error_code: Optional[str] = None,
        provider_message: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            provider=provider,
            status_code=status_code,
            message=message,
            error_type=error_type or "model_unavailable",
            error_code=error_code or "model_not_found",
            provider_message=provider_message,
            request_id=request_id,
        )


class BaseProviderClient:
    """Base HTTP client for generation providers."""

    def __init__(
        self,
        *,
        config: Config,
        base_url: str,
        api_key: str,
        provider_name: str,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self._config = config
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._provider_name = provider_name
        self._default_headers = default_headers or {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self._pending_log_context: Optional[Dict[str, Any]] = None
        self._last_response_meta: Dict[str, Any] = {}

    @property
    def provider_name(self) -> str:
        return self._provider_name

    async def _ensure_session(self) -> aiohttp.ClientSession:
        async with self._lock:
            if self._session is None or self._session.closed:
                total_timeout = None
                if self._config.provider_timeout_s > 0:
                    total_timeout = self._config.provider_timeout_s
                timeout = aiohttp.ClientTimeout(
                    total=total_timeout,
                    sock_connect=self._config.request_connect_timeout,
                    sock_read=self._config.request_read_timeout,
                )
                self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # API surface for subclasses
    # ------------------------------------------------------------------

    async def enqueue_job(
        self,
        *,
        idempotency_key: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderJobSubmission:
        payload = self._build_enqueue_payload(**kwargs)
        return await self._submit_payload(payload=payload, idempotency_key=idempotency_key)

    def _build_enqueue_payload(self, **kwargs: Any) -> Dict[str, Any]:
        payload = kwargs.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("Provider payload must be provided as a dict")
        return payload

    async def _submit_payload(
        self,
        *,
        payload: Dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> ProviderJobSubmission:
        data, status_code, duration_ms = await self._request(
            "POST",
            self._jobs_path(),
            json=payload,
            idempotency_key=idempotency_key,
        )
        job_id = self._extract_job_id(data)
        if not job_id:
            raise ProviderAPIError(
                provider=self._provider_name,
                status_code=status_code,
                message="Provider API did not return a job id",
                error_type="protocol",
                provider_message=json.dumps(data),
            )
        return ProviderJobSubmission(
            job_id=str(job_id),
            status_code=status_code,
            duration_ms=duration_ms,
            data=data,
        )

    async def get_job_status(self, job_id: str) -> ProviderJobStatus:
        data, status_code, duration_ms = await self._request("GET", f"{self._jobs_path()}/{job_id}")
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

    async def download_asset(self, asset_url: str) -> bytes:
        session = await self._ensure_session()
        start = time.monotonic()
        log.debug(
            "Downloading provider asset provider=%s url=%s",
            self._provider_name,
            asset_url,
        )
        try:
            async with session.get(asset_url) as response:
                body = await response.read()
                if response.status >= 400:
                    log.warning(
                        "Failed to download provider asset provider=%s url=%s status=%s",
                        self._provider_name,
                        asset_url,
                        response.status,
                    )
                    raise ProviderAPIError(
                        provider=self._provider_name,
                        status_code=response.status,
                        message=f"Failed to download asset: {response.status}",
                        provider_message=body.decode("utf-8", errors="ignore"),
                    )
                return body
        except asyncio.TimeoutError as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            log.warning(
                "Provider asset download timed out provider=%s url=%s duration_ms=%s",
                self._provider_name,
                asset_url,
                duration_ms,
            )
            raise ProviderAPIError(
                provider=self._provider_name,
                status_code=0,
                message="Asset download timed out",
                error_type="timeout",
                duration_ms=duration_ms,
            ) from exc
        finally:
            duration_ms = int((time.monotonic() - start) * 1000)
            log.debug(
                "Finished provider asset download provider=%s url=%s duration_ms=%s",
                self._provider_name,
                asset_url,
                duration_ms,
            )

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _jobs_path(self) -> str:
        return "/jobs"

    def _extract_job_id(self, payload: Dict[str, Any]) -> Optional[str]:
        return str(payload.get("id")) if payload.get("id") else None

    def _extract_status(self, payload: Dict[str, Any]) -> str:
        return str(payload.get("status", "unknown"))

    def _extract_error(self, payload: Dict[str, Any]) -> Optional[str]:
        return payload.get("error")

    def _extract_assets(self, payload: Dict[str, Any]) -> Dict[str, str]:
        result = payload.get("result")
        if isinstance(result, dict):
            assets = {}
            for key, value in result.items():
                if isinstance(value, str):
                    assets[key] = value
            return assets
        return {}

    def _build_headers(self) -> Dict[str, str]:
        headers = dict(self._default_headers)
        headers.setdefault("Authorization", f"Bearer {self._api_key}")
        headers.setdefault("Content-Type", "application/json")
        return headers

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        idempotency_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], int, int]:
        url = f"{self._base_url}{path}"
        headers = self._build_headers()
        if idempotency_key:
            headers.setdefault("Idempotency-Key", idempotency_key)
        headers.update(kwargs.pop("headers", {}))

        attempt = 0
        delay = self._config.retry_backoff
        last_error: Optional[ProviderAPIError] = None
        while attempt <= self._config.request_retries:
            try:
                log.debug(
                    "Provider request attempt=%s provider=%s method=%s url=%s idempotency_key=%s",
                    attempt + 1,
                    self._provider_name,
                    method,
                    url,
                    headers.get("Idempotency-Key"),
                )
                return await self._perform_request(method, url, headers=headers, **kwargs)
            except ProviderAPIError as exc:
                last_error = exc
                if not exc.retryable or attempt == self._config.request_retries:
                    log.warning(
                        "Provider request failed provider=%s method=%s url=%s status=%s error_type=%s error_code=%s message=%s",
                        self._provider_name,
                        method,
                        url,
                        exc.status_code,
                        exc.error_type,
                        exc.error_code,
                        exc,
                    )
                    raise
                attempt += 1
                log.warning(
                    "Retrying provider request provider=%s method=%s url=%s attempt=%s/%s retryable=%s",
                    self._provider_name,
                    method,
                    url,
                    attempt + 1,
                    self._config.request_retries + 1,
                    exc.retryable,
                )
                await asyncio.sleep(delay)
                delay *= self._config.retry_backoff
        assert last_error is not None  # pragma: no cover
        raise last_error

    async def _perform_request(
        self,
        method: str,
        url: str,
        *,
        headers: Dict[str, str],
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], int, int]:
        session = await self._ensure_session()
        start = time.monotonic()
        json_payload = kwargs.get("json")
        data_payload = kwargs.get("data")
        params_payload = kwargs.get("params")
        context = dict(self._pending_log_context or {})
        context_text = _serialise_for_log(context) if context else ""
        log.debug(
            "Sending provider request provider=%s method=%s url=%s headers=%s params=%s json=%s data=%s context=%s",
            self._provider_name,
            method,
            url,
            _sanitize_headers(headers),
            _serialise_for_log(params_payload),
            _serialise_for_log(json_payload),
            _serialise_for_log(data_payload),
            context_text,
        )
        try:
            # fal.ai requires GET/HEAD requests to be sent without Content-Type
            # when there is no request body.
            method_upper = method.upper()
            if method_upper in {"GET", "HEAD"}:
                if json_payload in (None, {}):
                    kwargs.pop("json", None)
                    json_payload = None
                if data_payload in (None, b"", ""):
                    kwargs.pop("data", None)
                    data_payload = None
            if method_upper in {"GET", "HEAD"} and not json_payload and not data_payload:
                headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
            if "queue.fal.run" in url:
                log.info(
                    "fal.http.request provider=%s method=%s url=%s has_body=%s header_names=%s",
                    self._provider_name,
                    method_upper,
                    url,
                    bool(json_payload) or bool(data_payload),
                    sorted(headers.keys()),
                )
            if "queue.fal.run" in url and (self._config.environment or "").strip().lower() == "dev":
                log.debug(
                    "fal.queue.request provider=%s method=%s url=%s has_json=%s has_data=%s sent_content_type=%s",
                    self._provider_name,
                    method_upper,
                    url,
                    bool(json_payload),
                    bool(data_payload),
                    bool(
                        next((v for k, v in headers.items() if k.lower() == "content-type"), None)
                    ),
                )
            async with session.request(method, url, headers=headers, **kwargs) as response:
                text = await response.text()
                duration_ms = int((time.monotonic() - start) * 1000)
                status = response.status
                response_headers = dict(response.headers)
                request_id = _extract_request_id(response_headers)
                safe_response_headers = _sanitize_headers(response_headers)
                self._record_response_meta(
                    method=method,
                    url=url,
                    status=status,
                    duration_ms=duration_ms,
                    headers=response_headers,
                    context=context,
                )
                log.debug(
                    "Provider response provider=%s method=%s url=%s status=%s duration_ms=%s request_id=%s headers=%s context=%s body=%s",
                    self._provider_name,
                    method,
                    url,
                    status,
                    duration_ms,
                    request_id or "",
                    safe_response_headers,
                    context_text,
                    _truncate(text),
                )
                if status >= 400:
                    log.warning(
                        "Provider responded with error provider=%s method=%s url=%s status=%s duration_ms=%s request_id=%s headers=%s context=%s body=%s",
                        self._provider_name,
                        method,
                        url,
                        status,
                        duration_ms,
                        request_id or "",
                        safe_response_headers,
                        context_text,
                        _truncate(text, limit=3072),
                    )
                    raise self._build_error(status, text, duration_ms)
                if not text:
                    return {}, status, duration_ms
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as exc:
                    log.error(
                        "Failed to decode provider response provider=%s method=%s url=%s status=%s request_id=%s headers=%s context=%s body=%s",
                        self._provider_name,
                        method,
                        url,
                        status,
                        request_id or "",
                        safe_response_headers,
                        context_text,
                        _truncate(text),
                    )
                    raise ProviderAPIError(
                        provider=self._provider_name,
                        status_code=status,
                        message="Unable to decode provider response",
                        error_type="decode",
                        provider_message=text,
                        retryable=False,
                        duration_ms=duration_ms,
                    ) from exc
                return data, status, duration_ms
        except asyncio.TimeoutError as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            log.warning(
                "Provider request timed out provider=%s method=%s url=%s duration_ms=%s",
                self._provider_name,
                method,
                url,
                duration_ms,
            )
            raise ProviderAPIError(
                provider=self._provider_name,
                status_code=0,
                message="Provider request timed out",
                error_type="timeout",
                duration_ms=duration_ms,
            ) from exc
        finally:
            self._pending_log_context = None

    def _record_response_meta(
        self,
        *,
        method: str,
        url: str,
        status: int,
        duration_ms: int,
        headers: Mapping[str, str],
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        request_id = _extract_request_id(headers)
        meta = {
            "method": method,
            "url": url,
            "status": status,
            "duration_ms": duration_ms,
            "request_id": request_id,
            "headers": _sanitize_headers(dict(headers)),
            "context": dict(context or {}),
        }
        self._last_response_meta = meta
        self._after_response_meta(meta)

    def _after_response_meta(self, meta: Dict[str, Any]) -> None:
        """Hook for subclasses to react to response metadata."""
        return

    def _set_request_context(self, context: Optional[Mapping[str, Any]]) -> None:
        self._pending_log_context = dict(context or {}) if context else None

    def _build_error(self, status: int, text: str, duration_ms: int) -> ProviderAPIError:
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {"raw": text}
        message = "Provider API error"
        error_type = None
        error_code = None
        retryable = status >= 500
        error_context: Dict[str, Any] = {}
        if isinstance(payload, dict):
            message = payload.get("message") or payload.get("error") or message
            error_type = payload.get("type") or payload.get("error_type")
            error_code = payload.get("code") or payload.get("error_code")
            retryable = bool(payload.get("retryable", retryable))
            nested_error = payload.get("error")
            if isinstance(nested_error, dict):
                message = nested_error.get("message") or message
                error_type = (
                    nested_error.get("type") or nested_error.get("error_type") or error_type
                )
                error_code = (
                    nested_error.get("code") or nested_error.get("error_code") or error_code
                )
                for key in ("param", "detail", "details", "help", "url"):
                    value = nested_error.get(key)
                    if value:
                        error_context[key] = value
                extra_fields = {
                    key: value
                    for key, value in nested_error.items()
                    if key
                    not in {
                        "message",
                        "type",
                        "error_type",
                        "code",
                        "error_code",
                        "param",
                        "detail",
                        "details",
                        "help",
                        "url",
                    }
                }
                if extra_fields:
                    error_context.setdefault("extra", extra_fields)
        if error_context:
            log.warning(
                "Provider error details provider=%s status=%s type=%s code=%s context=%s",
                self._provider_name,
                status,
                error_type,
                error_code,
                _serialise_for_log(error_context, limit=2048),
            )
        return ProviderAPIError(
            provider=self._provider_name,
            status_code=status,
            message=message,
            error_type=error_type,
            error_code=error_code,
            provider_message=text,
            retryable=retryable,
            duration_ms=duration_ms,
        )


__all__ = [
    "BaseProviderClient",
    "ProviderAPIError",
    "ModelUnavailable",
    "ProviderJobSubmission",
    "ProviderJobStatus",
]
