"""Async client for the Sora video generation API."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp

from config import Config
from utils import async_retry

log = logging.getLogger(__name__)


@dataclass
class SoraJobResult:
    job_id: str
    status: str
    video_url: Optional[str]
    error: Optional[str]


class SoraClient:
    """Thin wrapper around the Sora API with retry semantics."""

    def __init__(self, *, config: Config) -> None:
        self._config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        async with self._lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=self._config.request_timeout)
                self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def submit_job(self, *, prompt: str, settings: Optional[Dict[str, Any]] = None) -> str:
        payload = {"prompt": prompt, "settings": settings or {}}
        response = await self._request("POST", "/jobs", json=payload)
        job_id = response.get("id")
        if not job_id:
            raise RuntimeError("Sora API did not return a job id")
        return str(job_id)

    async def fetch_job(self, job_id: str) -> SoraJobResult:
        response = await self._request("GET", f"/jobs/{job_id}")
        status = response.get("status", "unknown")
        video_url = response.get("result", {}).get("video_url")
        error = response.get("error")
        return SoraJobResult(job_id=job_id, status=status, video_url=video_url, error=error)

    async def _request(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        url = f"{self._config.sora_api_url.rstrip('/')}{path}"
        headers = kwargs.pop("headers", {})
        headers.setdefault("Authorization", f"Bearer {self._config.sora_api_key}")
        headers.setdefault("Content-Type", "application/json")

        session = await self._ensure_session()

        async def _do_request() -> Dict[str, Any]:
            async with session.request(method, url, headers=headers, **kwargs) as response:
                body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"Sora API error {response.status}: {body}")
                if not body:
                    return {}
                try:
                    return json.loads(body)
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                    raise RuntimeError("Unable to decode Sora API response") from exc

        return await async_retry(
            _do_request,
            retries=self._config.request_retries,
            backoff=self._config.retry_backoff,
        )


__all__ = ["SoraClient", "SoraJobResult"]
