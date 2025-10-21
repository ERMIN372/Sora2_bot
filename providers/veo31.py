"""Veo 3.1 video generation client."""
from __future__ import annotations

from typing import Any, Dict, Optional

from config import Config

from .base import BaseProviderClient, ProviderJobSubmission


class Veo31Client(BaseProviderClient):
    def __init__(self, *, config: Config) -> None:
        api_key = config.veo31_api_key or config.veo3_api_key or config.sora_api_key
        super().__init__(
            config=config,
            base_url=config.veo31_api_url,
            api_key=api_key,
            provider_name="veo3.1",
        )

    def _build_enqueue_payload(
        self,
        *,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        preset: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        if payload is not None:
            return payload
        body: Dict[str, Any] = {"prompt": prompt}
        if settings:
            body["settings"] = settings
        if preset:
            body["preset"] = preset
        return body
