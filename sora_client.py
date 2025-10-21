"""Async client for the Sora video generation API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import Config
from providers.base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
)


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


class SoraAPIError(ProviderAPIError):
    """Sora-specific provider error for backwards compatibility."""

    def __init__(self, *, status_code: int, message: str, **kwargs: Any) -> None:
        super().__init__(
            provider="sora",
            status_code=status_code,
            message=message,
            **kwargs,
        )


class SoraClient(BaseProviderClient):
    """Thin wrapper around the Sora API with fine-grained retries."""

    def __init__(self, *, config: Config) -> None:
        super().__init__(
            config=config,
            base_url=config.sora_api_url,
            api_key=config.sora_api_key,
            provider_name="sora",
        )

    def _build_error(self, status: int, text: str, duration_ms: int) -> ProviderAPIError:
        error = super()._build_error(status, text, duration_ms)
        return SoraAPIError(
            status_code=error.status_code,
            message=str(error),
            error_type=error.error_type,
            error_code=error.error_code,
            provider_message=error.provider_message,
            retryable=error.retryable,
            duration_ms=duration_ms,
        )

    def _build_enqueue_payload(
        self,
        *,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        if payload is not None:
            return payload
        body = {"prompt": prompt, "settings": settings or {}}
        return body

    def _extract_job_id(self, payload: Dict[str, Any]) -> Optional[str]:
        job_id = payload.get("id")
        return str(job_id) if job_id else None

    def _extract_error(self, payload: Dict[str, Any]) -> Optional[str]:
        error = payload.get("error")
        if isinstance(error, dict):
            return error.get("message") or error.get("detail")
        return error

    async def submit_job(
        self,
        *,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> SoraJobSubmission:
        submission: ProviderJobSubmission = await super().enqueue_job(
            prompt=prompt,
            settings=settings,
            idempotency_key=idempotency_key,
        )
        return SoraJobSubmission(
            job_id=submission.job_id,
            status_code=submission.status_code,
            duration_ms=submission.duration_ms,
            data=submission.data,
        )

    async def fetch_job(self, job_id: str) -> SoraJobResult:
        status: ProviderJobStatus = await super().get_job_status(job_id)
        video_url = None
        if status.assets:
            video_url = status.assets.get("video_url") or next(
                (value for key, value in status.assets.items() if key.endswith("_url")),
                None,
            )
        return SoraJobResult(
            job_id=job_id,
            status=status.status,
            video_url=video_url,
            error=status.error,
            data=status.data,
            status_code=status.status_code,
            duration_ms=status.duration_ms,
        )

    async def list_models(self) -> Dict[str, Any]:
        data, _, _ = await self._request("GET", "/models")
        return data


__all__ = [
    "SoraAPIError",
    "SoraClient",
    "SoraJobResult",
    "SoraJobSubmission",
    "SoraRequestMeta",
]
