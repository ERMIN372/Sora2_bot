"""Kling AI Motion Control video generation provider via fal.ai queue API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp

from config import Config
from providers.base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
    extract_url,
    iter_nodes,
)

log = logging.getLogger(__name__)

FAL_QUEUE_BASE_URL = os.getenv("FAL_QUEUE_BASE_URL", "https://queue.fal.run")
FAL_KLING_MODEL = "fal-ai/kling-video/v2.6/standard/motion-control"

DEFAULT_MODE = "std"  # std = 720p, pro = 1080p

# fal queue status mapping → normalised provider statuses
_STATUS_MAP: Dict[str, str] = {
    "in_queue": "running",
    "in_progress": "running",
    "completed": "completed",
    "failed": "failed",
}


def _mask(value: str, visible: int = 4) -> str:
    if len(value) <= visible:
        return "***"
    return value[:visible] + "***"


class KlingVideoClient(BaseProviderClient):
    """Client for Kling AI Motion Control generation routed through fal.ai."""

    def __init__(self, *, config: Config) -> None:
        self._fal_key = config.fal_key
        if not self._fal_key:
            raise RuntimeError("fal.ai key is not configured; set FAL_KEY for Kling Motion Control")
        super().__init__(
            config=config,
            base_url=FAL_QUEUE_BASE_URL,
            api_key=self._fal_key,
            provider_name="kling",
        )
        self._cache: Dict[str, Dict[str, Any]] = {}
        log.info(
            "KlingVideoClient configured via fal.ai key=%s",
            _mask(self._fal_key),
        )

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Key {self._fal_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    async def enqueue_job(
        self,
        *,
        prompt: str = "",
        settings: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        **_: Any,
    ) -> ProviderJobSubmission:
        settings = dict(settings or {})

        # Build the character image data URI from Telegram inline data
        image_url = self._resolve_image_url(settings)
        if not image_url:
            raise ProviderAPIError(
                provider="kling",
                status_code=400,
                message="Motion Control requires a character image (photo mode)",
                error_type="validation",
            )

        # Motion reference URL is required by Motion Control API
        video_url = self._resolve_motion_video_url(settings)
        if not video_url:
            raise ProviderAPIError(
                provider="kling",
                status_code=400,
                message="Motion Control requires an accessible video_url",
                error_type="validation",
            )

        mode = settings.get("kling_mode") or DEFAULT_MODE

        body: Dict[str, Any] = {
            "mode": mode,
            "image_url": image_url,
            "video_url": video_url,
            "character_orientation": settings.get("character_orientation") or "video",
            "keep_original_sound": settings.get("keep_original_sound") or "yes",
        }

        if prompt and prompt.strip():
            body["prompt"] = prompt.strip()[:2500]

        log.info(
            "kling.enqueue mode=%s has_video_url=%s prompt_len=%s",
            mode,
            bool(video_url),
            len(prompt or ""),
        )

        data, status_code, duration_ms = await self._request(
            "POST",
            f"/{FAL_KLING_MODEL}",
            json={"input": body},
        )

        task_id = data.get("request_id")
        if not task_id:
            raise ProviderAPIError(
                provider="kling",
                status_code=status_code,
                message="fal.ai queue did not return request_id",
                error_type="protocol",
                provider_message=json.dumps(data, ensure_ascii=False),
            )

        self._cache[task_id] = data
        log.info(
            "kling.enqueue success task_id=%s status_code=%s duration_ms=%s",
            task_id,
            status_code,
            duration_ms,
        )
        return ProviderJobSubmission(
            job_id=str(task_id),
            status_code=status_code,
            duration_ms=duration_ms,
            data=data,
        )

    async def get_job_status(self, job_id: str) -> ProviderJobStatus:
        cached = self._cache.get(job_id)
        if cached:
            cached_status = self._extract_status(cached)
            if cached_status in ("completed", "failed"):
                return ProviderJobStatus(
                    job_id=job_id,
                    status=cached_status,
                    error=self._extract_error(cached),
                    assets=self._extract_assets(cached),
                    data=cached,
                    status_code=200,
                    duration_ms=0,
                )

        data, status_code, duration_ms = await self._request(
            "GET",
            f"/{FAL_KLING_MODEL}/requests/{job_id}/status",
        )
        status = self._extract_status(data)

        if status == "completed":
            result_data, _, _ = await self._request(
                "GET",
                f"/{FAL_KLING_MODEL}/requests/{job_id}",
            )
            self._cache[job_id] = result_data
            data = result_data
        else:
            self._cache[job_id] = data

        error = self._extract_error(data)
        assets = self._extract_assets(data)

        log.info(
            "kling.status task_id=%s status=%s assets=%s duration_ms=%s",
            job_id,
            status,
            list(assets.keys()) if assets else "none",
            duration_ms,
        )
        return ProviderJobStatus(
            job_id=job_id,
            status=status,
            error=error,
            assets=assets,
            data=data,
            status_code=status_code,
            duration_ms=duration_ms,
        )

    async def download_content(self, video_id: str, format: str = "mp4") -> Path:
        """Download video from the asset URL stored in cache."""
        cached = self._cache.get(video_id, {})
        assets = self._extract_assets(cached)
        video_url = assets.get("video") or assets.get("url")
        if not video_url:
            raise ProviderAPIError(
                provider="kling",
                status_code=404,
                message=f"No download URL for task {video_id}",
                error_type="not_found",
            )

        session = await self._ensure_session()
        timeout = aiohttp.ClientTimeout(
            total=self._config.request_read_timeout or None,
            sock_read=self._config.request_read_timeout or None,
        )
        started = time.monotonic()
        try:
            async with session.get(video_url, allow_redirects=True, timeout=timeout) as response:
                if response.status != 200:
                    body = await response.text()
                    raise ProviderAPIError(
                        provider="kling",
                        status_code=response.status,
                        message=f"Failed to download Kling video: {response.status}",
                        provider_message=body[:500],
                    )
                target_dir = Path(tempfile.mkdtemp(prefix="kling-video-"))
                file_path = target_dir / f"{video_id}.mp4"
                with open(file_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(64 * 1024):
                        f.write(chunk)
                size = file_path.stat().st_size
                duration_ms = int((time.monotonic() - started) * 1000)
                log.info(
                    "kling.download success task_id=%s bytes=%s duration_ms=%s",
                    video_id,
                    size,
                    duration_ms,
                )
                return file_path
        except asyncio.TimeoutError as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            raise ProviderAPIError(
                provider="kling",
                status_code=0,
                message="Kling video download timed out",
                error_type="timeout",
                duration_ms=duration_ms,
            ) from exc

    # ------------------------------------------------------------------
    # Hooks (extract from fal queue response format)
    # ------------------------------------------------------------------

    def _extract_status(self, payload: Dict[str, Any]) -> str:
        raw_status = str(payload.get("status") or "").strip().lower()
        return _STATUS_MAP.get(raw_status, "running")

    def _extract_error(self, payload: Dict[str, Any]) -> Optional[str]:
        msg = payload.get("error")
        if isinstance(msg, dict):
            msg = msg.get("message")
        if msg and isinstance(msg, str) and msg.strip():
            return msg.strip()
        if self._extract_status(payload) == "failed":
            return payload.get("message") or "Generation failed"
        return None

    def _extract_assets(self, payload: Dict[str, Any]) -> Dict[str, str]:
        assets: Dict[str, str] = {}
        output = payload.get("output")
        if isinstance(output, dict):
            candidates = [output]
        else:
            candidates = []

        candidates.append(payload)
        seen: set[str] = set()
        index = 0
        for candidate in candidates:
            for node in iter_nodes(candidate):
                url = extract_url(node)
                if not url or url in seen:
                    continue
                seen.add(url)
                key = "video" if index == 0 else f"video_{index}"
                assets[key] = url
                index += 1
        return assets

    @staticmethod
    def _resolve_motion_video_url(settings: Dict[str, Any]) -> Optional[str]:
        explicit_url = settings.get("video_url") or settings.get("motion_video_url")
        if isinstance(explicit_url, str) and explicit_url.strip():
            return explicit_url.strip()

        inline = settings.get("motion_video_inline_data")
        if isinstance(inline, dict):
            # Backward-compatible fallback: older code paths may still pass inline payload.
            # API expects URL, but keep this to avoid hard break for legacy callers.
            mime = inline.get("mime_type") or "video/mp4"
            data = inline.get("data")
            if data:
                return f"data:{mime};base64,{data}"
        return None

    # ------------------------------------------------------------------
    # Image resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_image_url(settings: Dict[str, Any]) -> Optional[str]:
        """Build an image URL / data URI from the settings dict.

        The handler layer downloads the Telegram photo and puts it in
        ``settings["reference_inline_data"]`` as ``{"mime_type": ..., "data": ...}``
        (base64-encoded).  We pass base64 directly as required by the Kling Motion model on fal.ai.
        """
        # Direct URL (for testing / future use)
        explicit_url = settings.get("image_url") or settings.get("kling_image_url")
        if explicit_url:
            return explicit_url

        # Inline data from Telegram download (base64)
        inline = settings.get("reference_inline_data")
        if isinstance(inline, dict):
            data = inline.get("data")
            if data:
                # Motion Control docs require raw base64 without data: prefix.
                return str(data)

        return None
