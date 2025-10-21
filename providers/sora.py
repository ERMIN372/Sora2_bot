"""Client for the Sora video generation API."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from config import Config

from .base import BaseProviderClient, ProviderJobSubmission

log = logging.getLogger(__name__)


class SoraClient(BaseProviderClient):
    """Thin wrapper around the Sora API with telemetry logging."""

    def __init__(self, *, config: Config) -> None:
        super().__init__(
            config=config,
            base_url=config.sora_api_base,
            api_key=config.sora_api_key,
            provider_name="sora",
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
        config = settings or {}
        model_name = config.get("model") or self._config.sora_model
        body: Dict[str, Any] = {
            "prompt": prompt,
            "model": model_name,
            "settings": config,
        }
        return body

    async def enqueue_job(
        self,
        *,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderJobSubmission:
        submission = await super().enqueue_job(
            prompt=prompt,
            settings=settings,
            payload=payload,
            idempotency_key=idempotency_key,
            **kwargs,
        )
        config = settings or {}
        model_name = config.get("model") or self._config.sora_model
        size = config.get("size")
        duration = config.get("duration")
        log.info(
            "sora.call corr_id=%s provider=%s model=%s size=%s duration=%s status=%s latency_ms=%s",
            idempotency_key or submission.job_id,
            self.provider_name,
            model_name,
            size or "",
            duration or "",
            submission.status_code,
            submission.duration_ms,
        )
        return submission

    async def list_models(self) -> Dict[str, Any]:
        data, _, _ = await self._request("GET", "/models")
        return data


__all__ = ["SoraClient"]
