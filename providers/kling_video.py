"""Kling AI Motion Control video generation provider via fal.ai queue API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import aiohttp

from config import Config
from providers.base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
    extract_url,
    iter_nodes,
    validate_download_url,
)

log = logging.getLogger(__name__)

FAL_QUEUE_BASE_URL = os.getenv("FAL_QUEUE_BASE_URL", "https://queue.fal.run")
FAL_KLING_MODEL_STANDARD = "fal-ai/kling-video/v2.6/standard/motion-control"
FAL_KLING_MODEL_PRO = "fal-ai/kling-video/v2.6/pro/motion-control"
FAL_KLING_MODEL_BASE = "fal-ai/kling-video"
FAL_KLING_MODEL = FAL_KLING_MODEL_STANDARD  # back-compat alias for tests/imports
KLING_FAL_IMPL_REV = "kling-fal-queue-splitpath-v6"

DEFAULT_MODE = "std"  # std = 720p, pro = 1080p
_ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}
_ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".m4v", ".gif"}
_ALLOWED_ORIENTATION = {"image", "video"}

# fal queue status mapping → normalised provider statuses
_STATUS_MAP: Dict[str, str] = {
    "in_queue": "running",
    "in_progress": "running",
    "running": "running",
    "completed": "completed",
    "failed": "failed",
}

_RUNNING_STATUSES = {"in_queue", "in_progress", "running"}
_COMPLETED_STATUSES = {"completed"}
_FAILED_STATUSES = {"failed", "error", "errored", "cancelled", "canceled"}
_EARLY_RESULT_MISSING_FIELDS = {"image_url", "video_url", "character_orientation"}


def _mask(value: str, visible: int = 4) -> str:
    if len(value) <= visible:
        return "***"
    return value[:visible] + "***"


def _mask_url(value: Optional[str], visible: int = 18) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    candidate = value.strip()
    if len(candidate) <= visible:
        return "***"
    return candidate[:visible] + "***"


def _is_balance_error(text: Optional[str]) -> bool:
    candidate = (text or "").strip().lower()
    return "balance" in candidate and any(
        token in candidate for token in ("not enough", "insufficient")
    )


def _normalise_fal_key(raw: Optional[str]) -> str:
    candidate = (raw or "").strip().strip('"').strip("'")
    if candidate.lower().startswith("key "):
        candidate = candidate[4:].strip()
    return candidate


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


class KlingVideoClient(BaseProviderClient):
    """Client for Kling AI Motion Control generation routed through fal.ai."""

    def __init__(self, *, config: Config) -> None:
        self._fal_key = _normalise_fal_key(config.fal_key or os.getenv("FAL_API_KEY", ""))
        if not self._fal_key:
            raise RuntimeError("fal.ai key is not configured; set FAL_KEY for Kling Motion Control")
        super().__init__(
            config=config,
            base_url=FAL_QUEUE_BASE_URL,
            api_key=self._fal_key,
            provider_name="kling",
        )
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._submit_models: Dict[str, str] = {}  # job_id → submit model path
        log.info(
            "KlingVideoClient configured via fal.ai key=%s impl_rev=%s",
            _mask(self._fal_key),
            KLING_FAL_IMPL_REV,
        )

    def _queue_target(self, url_or_path: Optional[str], *, fallback: str) -> str:
        candidate = str(url_or_path or "").strip()
        if not candidate:
            return fallback
        if candidate.startswith("http://") or candidate.startswith("https://"):
            parsed = urlparse(candidate)
            base = urlparse(self._base_url)
            if parsed.scheme != base.scheme or parsed.netloc != base.netloc:
                log.warning(
                    "kling.queue_url_host_mismatch requested=%s expected_base=%s using_fallback=%s",
                    candidate,
                    self._base_url,
                    fallback,
                )
                return fallback
            path = parsed.path or "/"
            if parsed.query:
                path = f"{path}?{parsed.query}"
            return path
        return candidate

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Key {self._fal_key}",
        }

    @staticmethod
    def _pick_submit_model(mode: str) -> str:
        return (
            FAL_KLING_MODEL_PRO if str(mode).strip().lower() == "pro" else FAL_KLING_MODEL_STANDARD
        )

    @staticmethod
    def _extension_is_allowed(url: str, allowed: set[str]) -> bool:
        path = (urlparse(url).path or "").lower()
        return any(path.endswith(ext) for ext in allowed)

    async def _validate_public_asset_url(self, *, url: str, kind: str) -> None:
        validate_download_url(url)
        session = await self._ensure_session()
        timeout = aiohttp.ClientTimeout(total=20)
        statuses: list[str] = []
        for method in ("HEAD", "GET"):
            request_kwargs: Dict[str, Any] = {"allow_redirects": True, "timeout": timeout}
            if method == "GET":
                request_kwargs["headers"] = {"Range": "bytes=0-0"}
            response = await session.request(method, url, **request_kwargs)
            try:
                statuses.append(f"{method}:{response.status}")
                if response.status >= 400:
                    continue
                content_type = str(response.headers.get("Content-Type") or "").lower()
                if kind == "image":
                    valid = self._extension_is_allowed(
                        url, _ALLOWED_IMAGE_EXTENSIONS
                    ) or content_type.startswith("image/")
                else:
                    valid = self._extension_is_allowed(
                        url, _ALLOWED_VIDEO_EXTENSIONS
                    ) or content_type.startswith("video/")
                if not valid:
                    raise ProviderAPIError(
                        provider="kling",
                        status_code=400,
                        message=f"Invalid {kind}_url format for Kling Motion Control",
                        error_type="validation",
                        provider_message=f"content_type={content_type or 'n/a'} url={url}",
                    )
                return
            finally:
                response.release()
        raise ProviderAPIError(
            provider="kling",
            status_code=400,
            message=f"{kind}_url is not publicly accessible",
            error_type="validation",
            provider_message="; ".join(statuses),
        )

    async def _probe_video_duration_seconds(self, url: str) -> float:
        session = await self._ensure_session()
        with tempfile.NamedTemporaryFile(prefix="kling-probe-", suffix=".mp4", delete=False) as tmp:
            file_path = Path(tmp.name)
        try:
            async with session.get(
                url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status >= 400:
                    raise ProviderAPIError(
                        provider="kling",
                        status_code=400,
                        message="video_url is not downloadable",
                        error_type="validation",
                        provider_message=f"status={response.status}",
                    )
                with open(file_path, "wb") as out:
                    async for chunk in response.content.iter_chunked(128 * 1024):
                        out.write(chunk)

            proc = await asyncio.create_subprocess_exec(
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(file_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise ProviderAPIError(
                    provider="kling",
                    status_code=400,
                    message="Unable to detect motion video duration",
                    error_type="validation",
                    provider_message=(stderr.decode("utf-8", errors="ignore") or "ffprobe failed")[
                        :500
                    ],
                )
            return float((stdout or b"0").decode("utf-8", errors="ignore").strip() or "0")
        except FileNotFoundError as exc:
            raise ProviderAPIError(
                provider="kling",
                status_code=500,
                message="В контейнере отсутствует ffprobe, установите ffmpeg/ffprobe",
                error_type="validation",
            ) from exc
        finally:
            file_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # HTTP helpers (legacy JWT fallback + direct fal HTTP)
    # ------------------------------------------------------------------

    async def _request_via_fal_http(
        self,
        *,
        method: str,
        path: str,
        json_payload: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], int, int]:
        """Perform a direct fal.ai HTTP request bypassing BaseProviderClient helpers.

        Used as a defensive fallback for rare stale runtime states where legacy
        JWT helper references may still be invoked from older in-memory code.
        """

        session = await self._ensure_session()
        started = time.monotonic()
        url = f"{self._base_url}{path}"
        headers = {"Authorization": f"Key {self._fal_key}"}
        if json_payload is not None or method.upper() in {"POST", "PUT", "PATCH"}:
            headers["Content-Type"] = "application/json"
        request_kwargs: Dict[str, Any] = {}
        if json_payload is not None:
            request_kwargs["json"] = json_payload
        response = await session.request(method, url, headers=headers, **request_kwargs)
        try:
            text = await response.text()
            duration_ms = int((time.monotonic() - started) * 1000)
            if response.status >= 400:
                raise ProviderAPIError(
                    provider="kling",
                    status_code=response.status,
                    message="fal.ai request failed",
                    error_type="api_error",
                    provider_message=text,
                    duration_ms=duration_ms,
                )
            if not text:
                return {}, response.status, duration_ms
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ProviderAPIError(
                    provider="kling",
                    status_code=response.status,
                    message="Unable to decode fal.ai response",
                    error_type="decode",
                    provider_message=text,
                    duration_ms=duration_ms,
                ) from exc
            return payload, response.status, duration_ms
        finally:
            response.release()

    async def _request_with_legacy_fallback(
        self,
        method: str,
        path: str,
        *,
        json_payload: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], int, int]:
        try:
            kwargs: Dict[str, Any] = {}
            if json_payload is not None or method.upper() in {"POST", "PUT", "PATCH"}:
                kwargs["headers"] = {"Content-Type": "application/json"}
            if json_payload is not None:
                kwargs["json"] = json_payload
            return await self._request(method, path, **kwargs)
        except NameError as exc:
            if "_generate_jwt" not in str(exc):
                raise
            log.warning(
                "kling.request detected legacy JWT NameError; using direct fal HTTP fallback method=%s path=%s",
                method,
                path,
            )
            return await self._request_via_fal_http(
                method=method,
                path=path,
                json_payload=json_payload,
            )

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
        submit_model = self._pick_submit_model(str(mode))
        character_orientation = (
            str(settings.get("character_orientation") or "video").strip().lower()
        )
        if character_orientation not in _ALLOWED_ORIENTATION:
            raise ProviderAPIError(
                provider="kling",
                status_code=400,
                message="character_orientation must be one of: image, video",
                error_type="validation",
            )

        try:
            await self._validate_public_asset_url(url=image_url, kind="image")
            await self._validate_public_asset_url(url=video_url, kind="video")
            duration_seconds = await self._probe_video_duration_seconds(video_url)
            max_duration = 10 if character_orientation == "image" else 30
            if duration_seconds > max_duration:
                raise ProviderAPIError(
                    provider="kling",
                    status_code=400,
                    message="motion reference video duration exceeds character_orientation limit",
                    error_type="validation",
                    provider_message=(
                        f"character_orientation={character_orientation} duration={duration_seconds:.2f}s "
                        f"max={max_duration}s"
                    ),
                )

            input_payload: Dict[str, Any] = {
                "image_url": image_url,
                "video_url": video_url,
                "character_orientation": character_orientation,
                "keep_original_sound": _as_bool(settings.get("keep_original_sound"), default=True),
            }

            if prompt and prompt.strip():
                input_payload["prompt"] = prompt.strip()[:2500]

            request_idempotency_key = str(idempotency_key or uuid.uuid4())
            input_payload["idempotency_key"] = request_idempotency_key
            submit_payload = {"input": input_payload}

            input_block = submit_payload.get("input")
            input_keys = sorted(input_block.keys()) if isinstance(input_block, dict) else []
            log.debug(
                "kling.submit_payload has_input=%s input_keys=%s has_image_url=%s has_video_url=%s has_character_orientation=%s image_url=%s video_url=%s",
                isinstance(input_block, dict),
                input_keys,
                bool(input_block.get("image_url")) if isinstance(input_block, dict) else False,
                bool(input_block.get("video_url")) if isinstance(input_block, dict) else False,
                bool(input_block.get("character_orientation")) if isinstance(input_block, dict) else False,
                _mask_url(input_block.get("image_url")) if isinstance(input_block, dict) else "",
                _mask_url(input_block.get("video_url")) if isinstance(input_block, dict) else "",
            )

            log.info(
                "kling.enqueue model=%s mode=%s has_image_url=%s has_video_url=%s keep_original_sound=%s prompt_len=%s duration_seconds=%.2f",
                submit_model,
                mode,
                bool(image_url),
                bool(video_url),
                input_payload.get("keep_original_sound"),
                len(prompt or ""),
                duration_seconds,
            )

            data, status_code, duration_ms = await self._request_with_legacy_fallback(
                "POST",
                f"/{submit_model}",
                json_payload=submit_payload,
            )
        except ProviderAPIError as exc:
            ffprobe_hint = f"{exc.message or ''} {exc.provider_message or ''}".lower()
            if "ffprobe" in ffprobe_hint and "required" in ffprobe_hint:
                raise ProviderAPIError(
                    provider="kling",
                    status_code=exc.status_code or 503,
                    message="В контейнере отсутствует ffprobe, установите ffmpeg/ffprobe",
                    error_type="provider_unavailable",
                    error_code="ffprobe_missing",
                    provider_message=exc.provider_message or str(exc),
                    retryable=False,
                    duration_ms=exc.duration_ms,
                ) from exc
            if _is_balance_error(exc.provider_message) or _is_balance_error(str(exc)):
                raise ProviderAPIError(
                    provider="kling",
                    status_code=exc.status_code or 402,
                    message="Account balance not enough",
                    error_type="billing",
                    error_code=exc.error_code or "insufficient_balance",
                    provider_message=exc.provider_message or str(exc),
                    retryable=False,
                    duration_ms=exc.duration_ms,
                ) from exc
            raise
        task_id = data.get("request_id")
        if not task_id:
            raise ProviderAPIError(
                provider="kling",
                status_code=status_code,
                message="fal.ai queue did not return request_id",
                error_type="protocol",
                provider_message=json.dumps(data, ensure_ascii=False),
            )

        self._cache[task_id] = {
            **data,
            "request_id": str(task_id),
            "status_url": data.get("status_url"),
            "response_url": data.get("response_url"),
        }
        self._submit_models[task_id] = submit_model
        log.info(
            "kling.enqueue success task_id=%s status_code=%s duration_ms=%s submit_model=%s status_url=%s response_url=%s cached_keys=%s",
            task_id,
            status_code,
            duration_ms,
            submit_model,
            data.get("status_url") or "",
            data.get("response_url") or "",
            list(data.keys()),
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

        poll_model = FAL_KLING_MODEL_BASE
        status_path = self._queue_target(
            cached.get("status_url") if isinstance(cached, dict) else None,
            fallback=f"/{poll_model}/requests/{job_id}/status",
        )

        log.info(
            "kling.poll job_id=%s status_path=%s poll_model=%s has_cached=%s",
            job_id,
            status_path,
            poll_model,
            bool(cached),
        )

        data, status_code, duration_ms = await self.poll_status(job_id=job_id, status_path=status_path)
        if isinstance(cached, dict):
            data.setdefault("status_url", cached.get("status_url"))
            data.setdefault("response_url", cached.get("response_url"))
            data.setdefault("request_id", cached.get("request_id") or job_id)
        status = self._extract_status(data)
        error = self._extract_error(data)
        assets = self._extract_assets(data)
        polled_status = status
        state, should_fetch_result, _retry_after_seconds = self._poll_status(
            status_code=status_code,
            status=status,
            error=error,
            assets=assets,
        )

        if should_fetch_result:
            result_path = self._queue_target(
                data.get("response_url") if isinstance(data, dict) else None,
                fallback=f"/{poll_model}/requests/{job_id}",
            )
            try:
                result_data, _, _ = await self.fetch_result(job_id=job_id, result_path=result_path)
            except ProviderAPIError as exc:
                if self._is_early_result_error(exc):
                    log.info(
                        "kling.result_not_ready job_id=%s status_code=%s message=%s",
                        job_id,
                        exc.status_code,
                        (exc.message or "")[:300],
                    )
                    self._cache[job_id] = data
                    status = "running"
                    error = None
                    assets = {}
                else:
                    raise
            else:
                if isinstance(data, dict):
                    result_data.setdefault("status_url", data.get("status_url"))
                    result_data.setdefault("response_url", data.get("response_url"))
                    result_data.setdefault("request_id", data.get("request_id") or job_id)
                self._cache[job_id] = result_data
                data = result_data
                status = self._extract_status(data)
                error = self._extract_error(data)
                assets = self._extract_assets(data)
                if polled_status == "completed" and status == "running" and not error:
                    status = "completed"
                if status == "completed" and not assets and not error:
                    log.info(
                        "kling.result_pending_media job_id=%s result_path=%s",
                        job_id,
                        result_path,
                    )
                    status = "running"
        else:
            self._cache[job_id] = data
            if state == "running":
                status = "running"

        log.info(
            "kling.status task_id=%s request_id=%s status=%s queue_position=%s assets=%s duration_ms=%s status_path=%s error=%s",
            job_id,
            data.get("request_id") or job_id,
            status,
            data.get("queue_position"),
            list(assets.keys()) if assets else "none",
            duration_ms,
            status_path,
            error,
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

    async def poll_status(self, *, job_id: str, status_path: str) -> tuple[Dict[str, Any], int, int]:
        """Single status polling request for queue state."""

        data, status_code, duration_ms = await self._request_with_legacy_fallback("GET", status_path)
        status_value = data.get("status") if isinstance(data, dict) else None
        log.info(
            "kling.status_parsed status_code=%s status=%s job_id=%s",
            status_code,
            status_value,
            job_id,
        )
        return data, status_code, duration_ms

    async def fetch_result(self, *, job_id: str, result_path: str) -> tuple[Dict[str, Any], int, int]:
        """Fetch queue response payload only after completed state."""

        log.info("kling.result_fetch job_id=%s result_path=%s", job_id, result_path)
        return await self._request_with_legacy_fallback("GET", result_path)

    def _is_early_result_error(self, exc: ProviderAPIError) -> bool:
        if exc.status_code == 400:
            return True
        return False

    def _poll_status(
        self,
        *,
        status_code: int,
        status: str,
        error: Optional[str],
        assets: Dict[str, str],
    ) -> tuple[str, bool, float]:
        """Return (state, should_fetch_result, retry_after_seconds) for queue polling."""

        normalised = (status or "").strip().lower()
        if error:
            return "failed", False, 0.0
        if status_code == 202 or normalised in _RUNNING_STATUSES:
            return "running", False, 2.0
        if status_code == 200 and normalised in _COMPLETED_STATUSES:
            if assets:
                return "completed", False, 0.0
            return "completed", True, 0.0
        if normalised in _FAILED_STATUSES:
            return "failed", False, 0.0
        return "running", False, 2.0

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
