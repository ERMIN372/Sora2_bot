"""NanoBanana image generation client."""
from __future__ import annotations

from typing import Any, Dict, Optional

from config import Config

from .base import BaseProviderClient, ProviderJobSubmission


class NanoBananaClient(BaseProviderClient):
    def __init__(self, *, config: Config) -> None:
        api_key = config.nanobanana_api_key or config.sora_api_key
        super().__init__(
            config=config,
            base_url=config.nanobanana_api_url,
            api_key=api_key,
            provider_name="nanobanana",
        )

    def _build_enqueue_payload(
        self,
        *,
        prompt: str,
        preset: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        if payload is not None:
            return payload
        body: Dict[str, Any] = {"prompt": prompt}
        if preset:
            body["preset"] = preset
        if settings:
            body["settings"] = settings
        return body
