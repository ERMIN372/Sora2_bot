"""Shared provider client utilities."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import aiohttp

from config import Config

log = logging.getLogger(__name__)


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
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code
        self.provider_message = provider_message
        self.retryable = retryable
        self.duration_ms = duration_ms


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

    @property
    def provider_name(self) -> str:
        return self._provider_name

    async def _ensure_session(self) -> aiohttp.ClientSession:
        async with self._lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(
                    total=None,
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
        return await self._submit_payload(
            payload=payload, idempotency_key=idempotency_key
        )

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
        data, status_code, duration_ms = await self._request(
            "GET", f"{self._jobs_path()}/{job_id}"
        )
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
        try:
            async with session.get(asset_url) as response:
                body = await response.read()
                if response.status >= 400:
                    raise ProviderAPIError(
                        provider=self._provider_name,
                        status_code=response.status,
                        message=f"Failed to download asset: {response.status}",
                        provider_message=body.decode("utf-8", errors="ignore"),
                    )
                return body
        except asyncio.TimeoutError as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            raise ProviderAPIError(
                provider=self._provider_name,
                status_code=0,
                message="Asset download timed out",
                error_type="timeout",
                duration_ms=duration_ms,
            ) from exc

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
                return await self._perform_request(
                    method, url, headers=headers, **kwargs
                )
            except ProviderAPIError as exc:
                last_error = exc
                if not exc.retryable or attempt == self._config.request_retries:
                    raise
                attempt += 1
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
        try:
            async with session.request(method, url, headers=headers, **kwargs) as response:
                text = await response.text()
                duration_ms = int((time.monotonic() - start) * 1000)
                status = response.status
                if status >= 400:
                    raise self._build_error(status, text, duration_ms)
                if not text:
                    return {}, status, duration_ms
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as exc:
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
            raise ProviderAPIError(
                provider=self._provider_name,
                status_code=0,
                message="Provider request timed out",
                error_type="timeout",
                duration_ms=duration_ms,
            ) from exc

    def _build_error(self, status: int, text: str, duration_ms: int) -> ProviderAPIError:
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {"raw": text}
        message = "Provider API error"
        error_type = None
        error_code = None
        retryable = status >= 500
        if isinstance(payload, dict):
            message = payload.get("message") or payload.get("error") or message
            error_type = payload.get("type") or payload.get("error_type")
            error_code = payload.get("code") or payload.get("error_code")
            retryable = bool(payload.get("retryable", retryable))
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
    "ProviderJobSubmission",
    "ProviderJobStatus",
]
