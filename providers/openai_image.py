"""OpenAI Images API client for fallback generation."""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from config import Config
from providers.base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
)

log = logging.getLogger(__name__)


class OpenAIImageClient(BaseProviderClient):
    """Minimal client for OpenAI image generation used as a fallback."""

    def __init__(self, *, config: Config) -> None:
        api_key = (config.openai_key or "").strip()
        if not api_key:
            raise RuntimeError("OpenAI API key is required for image generation")
        base_url = (config.openai_api_base or "https://api.openai.com/v1").rstrip("/")
        super().__init__(
            config=config,
            base_url=base_url,
            api_key=api_key,
            provider_name="openai-image",
            default_headers={"OpenAI-Beta": "assistants=v2"},
        )
        fallback_models = [
            model for model in config.fallback_image_models if isinstance(model, str) and model
        ]
        self._default_model = fallback_models[0] if fallback_models else "dall-e-3"
        self._pending: Dict[str, Dict[str, Any]] = {}

    def _jobs_path(self) -> str:  # pragma: no cover - unused by enqueue
        return "/images/generations"

    async def enqueue_job(
        self,
        *,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        **_: Any,
    ) -> ProviderJobSubmission:
        if not self._api_key:
            raise ProviderAPIError(
                provider=self.provider_name,
                status_code=401,
                message="OpenAI API key is not configured",
                error_type="auth",
            )
        options = dict(settings or {})
        model = (options.get("model") or self._default_model or "dall-e-3").strip()
        if not model:
            model = "dall-e-3"
        size = options.get("size")
        quality = options.get("quality")
        response_format = options.get("response_format") or "b64_json"
        user = options.get("user")
        request: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "response_format": response_format,
        }
        if size:
            request["size"] = str(size)
        if quality:
            request["quality"] = quality
        if user:
            request["user"] = str(user)
        if payload and isinstance(payload, dict):
            request.setdefault("prompt", prompt)
        data, status_code, duration_ms = await self._request(
            "POST",
            "/images/generations",
            json=request,
            idempotency_key=idempotency_key,
        )
        job_id = str(uuid4())
        inline_assets: List[Dict[str, Any]] = []
        assets_meta: List[Dict[str, Any]] = []
        asset_map: Dict[str, str] = {}
        images = data.get("data") if isinstance(data, dict) else None
        if isinstance(images, list):
            for index, item in enumerate(images):
                if not isinstance(item, dict):
                    continue
                key = f"image_{index}"
                if item.get("b64_json"):
                    raw = str(item["b64_json"]) or ""
                    try:
                        binary = base64.b64decode(raw)
                    except Exception:
                        binary = b""
                    data_uri = f"data:image/png;base64,{raw}" if raw else ""
                    inline_assets.append(
                        {
                            "key": key,
                            "kind": "image",
                            "mime": "image/png",
                            "bytes": len(binary),
                            "data_uri": data_uri,
                            "data_url": data_uri,
                        }
                    )
                    if data_uri:
                        asset_map[key] = data_uri
                        assets_meta.append(
                            {
                                "key": key,
                                "inline": True,
                                "mime": "image/png",
                                "bytes": len(binary),
                            }
                        )
                elif item.get("url"):
                    url = str(item.get("url"))
                    if url:
                        asset_map[key] = url
                        assets_meta.append({"key": key, "inline": False, "url": url})
        record = {
            "payload": data,
            "inline_assets": inline_assets,
            "assets": assets_meta,
            "asset_map": asset_map,
            "status_code": status_code,
            "duration_ms": duration_ms,
        }
        self._pending[job_id] = record
        result_payload = {
            "payload": data,
            "inline_assets": inline_assets,
            "assets": assets_meta,
        }
        return ProviderJobSubmission(
            job_id=job_id,
            status_code=status_code,
            duration_ms=duration_ms,
            data=result_payload,
        )

    async def get_job_status(self, job_id: str) -> ProviderJobStatus:
        record = self._pending.pop(job_id, None)
        if not record:
            return ProviderJobStatus(
                job_id=job_id,
                status="failed",
                assets={},
                error="Result not found",
                data={},
                status_code=404,
                duration_ms=0,
            )
        asset_map = record.get("asset_map") or {}
        inline_assets = record.get("inline_assets") or []
        assets_meta = record.get("assets") or []
        payload = record.get("payload") or {}
        status = "completed" if asset_map or inline_assets else "failed"
        error: Optional[str] = None
        if status != "completed":
            error = "OpenAI Images API did not return an image"
        data = {
            "payload": payload,
            "inline_assets": inline_assets,
            "assets": assets_meta,
        }
        return ProviderJobStatus(
            job_id=job_id,
            status=status,
            assets=asset_map,
            error=error,
            data=data,
            status_code=int(record.get("status_code", 200)),
            duration_ms=int(record.get("duration_ms", 0)),
        )


__all__ = ["OpenAIImageClient"]
