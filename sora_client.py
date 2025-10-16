"""Async client for the Sora video generation API."""
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
class SoraRequestMeta:
    status_code: int
    duration_ms: int


@dataclass(slots=True)
class SoraJobSubmission(SoraRequestMeta):
    job_id: str
    data: Dict[str, Any]


@dataclass(slots=True)
class SoraJobResult(SoraRequestMeta):
    job_id: str
    status: str
    video_url: Optional[str]
    error: Optional[str]
    data: Dict[str, Any]


class SoraAPIError(RuntimeError):
    """Structured representation of Sora API errors."""

    def __init__(
        self,
        *,
        status_code: int,
        message: str,
        error_type: Optional[str] = None,
        error_code: Optional[str] = None,
        provider_message: Optional[str] = None,
        retryable: bool = False,
        duration_ms: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code
        self.provider_message = provider_message
        self.retryable = retryable
        self.duration_ms = duration_ms


class SoraClient:
    """Thin wrapper around the Sora API with fine-grained retries."""

    def __init__(self, *, config: Config) -> None:
        self._config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

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

    async def submit_job(
        self,
        *,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> SoraJobSubmission:
        payload = {"prompt": prompt, "settings": settings or {}}
        data, status_code, duration_ms = await self._request(
            "POST",
            "/jobs",
            json=payload,
            idempotency_key=idempotency_key,
        )
        job_id = data.get("id")
        if not job_id:
            raise SoraAPIError(
                status_code=status_code,
                message="Sora API did not return a job id",
                error_type="protocol",
                provider_message=json.dumps(data),
            )
        return SoraJobSubmission(
            job_id=str(job_id),
            status_code=status_code,
            duration_ms=duration_ms,
            data=data,
        )

    async def fetch_job(self, job_id: str) -> SoraJobResult:
        data, status_code, duration_ms = await self._request("GET", f"/jobs/{job_id}")
        status = data.get("status", "unknown")
        video_url = data.get("result", {}).get("video_url") if isinstance(data.get("result"), dict) else None
        error = data.get("error")
        return SoraJobResult(
            job_id=job_id,
            status=status,
            video_url=video_url,
            error=error,
            data=data,
            status_code=status_code,
            duration_ms=duration_ms,
        )

    async def list_models(self) -> Dict[str, Any]:
        data, _, _ = await self._request("GET", "/models")
        return data

    async def _request(
        self,
        method: str,
        path: str,
        *,
        idempotency_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], int, int]:
        url = f"{self._config.sora_api_url.rstrip('/')}{path}"
        headers = kwargs.pop("headers", {})
        headers.setdefault("Authorization", f"Bearer {self._config.sora_api_key}")
        headers.setdefault("Content-Type", "application/json")
        if idempotency_key:
            headers.setdefault("Idempotency-Key", idempotency_key)

        attempt = 0
        delay = self._config.retry_backoff
        last_error: Optional[SoraAPIError] = None
        while attempt <= self._config.request_retries:
            try:
                return await self._perform_request(method, url, headers=headers, **kwargs)
            except SoraAPIError as exc:
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
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                    raise SoraAPIError(
                        status_code=status,
                        message="Unable to decode Sora API response",
                        error_type="decode",
                        provider_message=text,
                        retryable=False,
                        duration_ms=duration_ms,
                    ) from exc
                return data, status, duration_ms
        except asyncio.TimeoutError as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            raise SoraAPIError(
                status_code=0,
                message="Sora API request timed out",
                error_type="timeout",
                provider_message=str(exc),
                retryable=True,
                duration_ms=duration_ms,
            ) from exc
        except aiohttp.ClientError as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            raise SoraAPIError(
                status_code=0,
                message="Network error communicating with Sora",
                error_type="network",
                provider_message=str(exc),
                retryable=True,
                duration_ms=duration_ms,
            ) from exc

    def _build_error(self, status: int, body: str, duration_ms: int) -> SoraAPIError:
        error_type: Optional[str] = None
        error_code: Optional[str] = None
        message = f"Sora API error {status}"
        retryable = status == 429 or 500 <= status < 600
        detail: Optional[str] = None
        if body:
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                error_obj = parsed.get("error")
                if isinstance(error_obj, dict):
                    detail = error_obj.get("message") or error_obj.get("detail")
                    error_type = error_obj.get("type") or error_obj.get("code")
                    error_code = error_obj.get("code")
                elif isinstance(error_obj, str):
                    detail = error_obj
                elif parsed.get("detail"):
                    detail = parsed.get("detail") if isinstance(parsed.get("detail"), str) else json.dumps(parsed.get("detail"))
                if parsed.get("type") and not error_type:
                    error_type = str(parsed.get("type"))
                if parsed.get("code") and not error_code:
                    error_code = str(parsed.get("code"))
            else:
                detail = body.strip()
        provider_message = detail or body
        if status in {401, 403}:
            retryable = False
            error_type = error_type or "auth"
        if status == 400:
            retryable = False
            error_type = error_type or "validation"
        message = detail or message
        return SoraAPIError(
            status_code=status,
            message=message,
            error_type=error_type,
            error_code=error_code,
            provider_message=provider_message,
            retryable=retryable,
            duration_ms=duration_ms,
        )


__all__ = ["SoraAPIError", "SoraClient", "SoraJobResult", "SoraJobSubmission", "SoraRequestMeta"]
