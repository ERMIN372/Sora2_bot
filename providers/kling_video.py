"""Kling AI Motion Control video generation provider.

Uses the official Kling API (https://api.klingai.com) with JWT (HS256)
authentication to generate Motion Control videos from character images
and preset motions.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from config import Config
from providers.base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
)

log = logging.getLogger(__name__)

KLING_BASE_URL = os.getenv("KLING_BASE_URL", "https://api-singapore.klingai.com")
# Backward-compatible alias: some stale deployments may still reference this name.
FAL_QUEUE_BASE_URL = KLING_BASE_URL

DEFAULT_MODE = "std"  # std = 720p, pro = 1080p

# Kling task status mapping → normalised provider statuses
_STATUS_MAP: Dict[str, str] = {
    "submitted": "running",
    "processing": "running",
    "succeed": "completed",
    "failed": "failed",
}


def _mask(value: str, visible: int = 4) -> str:
    if len(value) <= visible:
        return "***"
    return value[:visible] + "***"


# ------------------------------------------------------------------
# JWT generation (HS256) without PyJWT dependency
# ------------------------------------------------------------------


def _b64url_encode(data: bytes) -> bytes:
    """Base64url-encode without padding."""
    # Local import guards against mixed/stale runtime states where module globals
    # may not include `base64` yet.
    import base64 as _base64

    return _base64.urlsafe_b64encode(data).rstrip(b"=")


def _generate_jwt(access_key: str, secret_key: str, expire_seconds: int = 1800) -> str:
    """Generate a JWT token for Kling API authentication."""
    # Local imports guard against mixed/stale runtime states where module globals
    # may not include stdlib symbols yet.
    import hashlib as _hashlib
    import hmac as _hmac
    import json as _json
    import time as _time

    header = _b64url_encode(
        _json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode()
    )
    now = int(_time.time())
    payload = _b64url_encode(
        _json.dumps(
            {
                "iss": access_key,
                "exp": now + expire_seconds,
                "nbf": now - 5,
            },
            separators=(",", ":"),
        ).encode()
    )
    signing_input = header + b"." + payload
    signature = _b64url_encode(
        _hmac.new(secret_key.encode(), signing_input, _hashlib.sha256).digest()
    )
    return (signing_input + b"." + signature).decode()


class KlingVideoClient(BaseProviderClient):
    """Client for Kling AI Motion Control video generation."""

    @staticmethod
    def _resolve_credentials(config: Config) -> tuple[str, str]:
        access_key = str(
            getattr(config, "kling_access_key", "") or os.getenv("KLING_ACCESS_KEY", "")
        ).strip()
        secret_key = str(
            getattr(config, "kling_secret_key", "") or os.getenv("KLING_SECRET_KEY", "")
        ).strip()
        return access_key, secret_key

    @staticmethod
    def _resolve_legacy_fal_key(config: Config) -> str:
        """Return a legacy FAL key for compatibility with stale deployments.

        Some older runtime bundles referenced ``self._fal_key`` while constructing
        provider clients. We keep this field initialised to avoid AttributeError
        under mixed-version rollouts.
        """

        return str(getattr(config, "fal_key", "") or os.getenv("FAL_KEY", "") or "").strip()

    def __init__(self, *, config: Config) -> None:
        self._fal_key = self._resolve_legacy_fal_key(config)
        self._access_key, self._secret_key = self._resolve_credentials(config)
        if not self._access_key or not self._secret_key:
            raise RuntimeError(
                "Kling API credentials not configured; " "set KLING_ACCESS_KEY and KLING_SECRET_KEY"
            )
        super().__init__(
            config=config,
            base_url=KLING_BASE_URL,
            api_key="",  # We use JWT, not a static key
            provider_name="kling",
        )
        self._token: Optional[str] = None
        self._token_expiry: int = 0
        self._cache: Dict[str, Dict[str, Any]] = {}
        log.info(
            "KlingVideoClient configured access_key=%s",
            _mask(self._access_key),
        )

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        """Return a cached JWT or generate a fresh one."""
        now = int(time.time())
        if self._token and now < self._token_expiry - 60:
            return self._token
        self._token = _generate_jwt(self._access_key, self._secret_key)
        self._token_expiry = now + 1800
        return self._token

    def _build_headers(self) -> Dict[str, str]:
        token_getter = getattr(self, "_get_token", None)
        raw_token = ""
        if callable(token_getter):
            try:
                raw_token = token_getter()
            except Exception:
                raw_token = ""
        if not raw_token:
            # Backward-compatible fallback for mixed/stale deployments where
            # instance/class shape may miss _get_token.
            raw_token = _generate_jwt(
                getattr(self, "_access_key", "") or "",
                getattr(self, "_secret_key", "") or "",
            )
        token = (raw_token or "").strip()
        if token.lower().startswith("bearer "):
            auth_value = token
        else:
            auth_value = f"Bearer {token}"
        return {
            "Authorization": auth_value,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _classify_api_error(message: Optional[str]) -> tuple[int, str, str]:
        text = str(message or "").strip()
        lowered = text.lower()
        if "account balance not enough" in lowered or "balance not enough" in lowered:
            return 402, "provider_insufficient_balance", "ACCOUNT_BALANCE_NOT_ENOUGH"
        return 502, "api_error", "KLING_API_ERROR"

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
            "/v1/videos/motion-control",
            json=body,
        )

        # Kling wraps the response in {"code": 0, "data": {"task_id": ...}}
        api_code = data.get("code")
        if api_code != 0:
            error_msg = data.get("message") or f"Kling API error code={api_code}"
            mapped_status, mapped_type, mapped_code = self._classify_api_error(error_msg)
            raise ProviderAPIError(
                provider="kling",
                status_code=mapped_status if status_code == 200 else status_code,
                message=error_msg,
                error_type=mapped_type,
                error_code=mapped_code if mapped_type != "api_error" else str(api_code),
                provider_message=json.dumps(data, ensure_ascii=False),
            )

        task_data = data.get("data") or {}
        task_id = task_data.get("task_id")
        if not task_id:
            raise ProviderAPIError(
                provider="kling",
                status_code=status_code,
                message="Kling API did not return a task_id",
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
            f"/v1/videos/motion-control/{job_id}",
        )

        api_code = data.get("code")
        if api_code != 0:
            error_msg = data.get("message") or f"Kling API error code={api_code}"
            log.warning(
                "kling.status error task_id=%s code=%s message=%s",
                job_id,
                api_code,
                error_msg,
            )
            return ProviderJobStatus(
                job_id=job_id,
                status="failed",
                error=error_msg,
                assets={},
                data=data,
                status_code=status_code,
                duration_ms=duration_ms,
            )

        self._cache[job_id] = data
        status = self._extract_status(data)
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
    # Hooks (extract from Kling response format)
    # ------------------------------------------------------------------

    def _extract_status(self, payload: Dict[str, Any]) -> str:
        task_data = payload.get("data") or {}
        raw_status = (task_data.get("task_status") or "").lower()
        return _STATUS_MAP.get(raw_status, "running")

    def _extract_error(self, payload: Dict[str, Any]) -> Optional[str]:
        task_data = payload.get("data") or {}
        msg = task_data.get("task_status_msg")
        if msg and isinstance(msg, str) and msg.strip():
            return msg.strip()
        if self._extract_status(payload) == "failed":
            return payload.get("message") or "Generation failed"
        return None

    def _extract_assets(self, payload: Dict[str, Any]) -> Dict[str, str]:
        task_data = payload.get("data") or {}
        result = task_data.get("task_result") or {}
        videos = result.get("videos") or []
        assets: Dict[str, str] = {}
        for i, video in enumerate(videos):
            url = video.get("url")
            if url:
                key = "video" if i == 0 else f"video_{i}"
                assets[key] = url
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
        (base64-encoded).  We convert that to a data URI accepted by
        the Kling API.
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
