"""Client wrapper for Gemini Veo video generation."""
from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import time
from fractions import Fraction
from typing import Any, Dict, Optional
from uuid import uuid4

from google.genai import errors as _genai_errors, types as _genai_types

from config import Config
from services.gemini_client import get_gemini_client

from .base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
)

log = logging.getLogger(__name__)


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _infer_aspect_ratio(size: Optional[str]) -> Optional[str]:
    if not size or "x" not in size:
        return None
    try:
        width_str, height_str = size.lower().split("x", 1)
        width = int(width_str)
        height = int(height_str)
        if width <= 0 or height <= 0:
            return None
    except (ValueError, ZeroDivisionError):  # pragma: no cover - defensive
        return None
    ratio = Fraction(width, height).limit_denominator(48)
    return f"{ratio.numerator}:{ratio.denominator}"


def _infer_resolution(size: Optional[str]) -> Optional[str]:
    if not size or "x" not in size:
        return None
    try:
        width_str, height_str = size.lower().split("x", 1)
        width = int(width_str)
        height = int(height_str)
    except ValueError:  # pragma: no cover - defensive
        return None
    longest = max(width, height)
    if longest >= 1080:
        return "1080p"
    if longest >= 720:
        return "720p"
    return None


def _decode_reference_image(data: Dict[str, Any]) -> Optional[_genai_types.Image]:
    raw = data.get("data") or data.get("base64")
    if not isinstance(raw, str):
        return None
    try:
        binary = base64.b64decode(raw, validate=False)
    except (ValueError, binascii.Error):  # pragma: no cover - defensive
        try:
            binary = base64.b64decode(raw.encode("utf-8"), validate=False)
        except Exception:  # pragma: no cover - defensive
            return None
    mime = data.get("mime") or data.get("mime_type") or "image/jpeg"
    return _genai_types.Image(image_bytes=binary, mime_type=mime)


class VeoVideoClient(BaseProviderClient):
    """Asynchronous wrapper around the Gemini video generation API."""

    def __init__(self, *, config: Config) -> None:
        super().__init__(
            config=config,
            base_url="https://generativelanguage.googleapis.com",
            api_key=config.gemini_api_key,
            provider_name="veo",
        )
        self._client = get_gemini_client(config)
        self._operations: Dict[str, _genai_types.GenerateVideosOperation] = {}
        self._requests: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def enqueue_job(
        self,
        *,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        **_: Any,
    ) -> ProviderJobSubmission:
        if not self._config.gemini_video_enabled:
            raise RuntimeError("Gemini video generation is not configured")

        request_settings = dict(settings or {})
        model_name = request_settings.get("model") or self._config.gemini_model_video
        source = self._build_source(prompt, request_settings)
        config_obj, config_meta = self._build_config(request_settings)

        start = time.perf_counter()
        try:
            operation = await asyncio.to_thread(
                self._generate_videos_operation,
                model_name,
                source,
                config_obj,
            )
        except _genai_errors.APIError as exc:  # pragma: no cover - network guard
            raise ProviderAPIError(
                provider=self.provider_name,
                status_code=getattr(exc, "code", 500) or 500,
                message=str(exc),
                error_type=getattr(exc, "status", None),
                provider_message=str(getattr(exc, "response", exc)),
            ) from exc
        duration_ms = int((time.perf_counter() - start) * 1000)
        job_id = operation.name or str(uuid4())
        data = operation.model_dump(exclude_none=True)
        async with self._lock:
            self._operations[job_id] = operation
            self._requests[job_id] = {
                "model": model_name,
                "config": config_meta,
            }
        return ProviderJobSubmission(
            job_id=job_id,
            status_code=200,
            duration_ms=duration_ms,
            data={"operation": data},
        )

    async def get_job_status(self, job_id: str) -> ProviderJobStatus:
        operation = await self._get_operation(job_id)
        start = time.perf_counter()
        try:
            refreshed = await asyncio.to_thread(self._client.operations.get, operation)
        except _genai_errors.APIError as exc:  # pragma: no cover - network guard
            raise ProviderAPIError(
                provider=self.provider_name,
                status_code=getattr(exc, "code", 500) or 500,
                message=str(exc),
                error_type=getattr(exc, "status", None),
                provider_message=str(getattr(exc, "response", exc)),
            ) from exc
        duration_ms = int((time.perf_counter() - start) * 1000)
        payload: Dict[str, Any] = {"operation": refreshed.model_dump(exclude_none=True)}
        async with self._lock:
            self._operations[job_id] = refreshed
            request_meta = self._requests.get(job_id, {}).copy()
        assets: Dict[str, str] = {}
        inline_assets: list[Dict[str, Any]] = []
        assets_meta: list[Dict[str, Any]] = []
        error_text: Optional[str] = None

        if not getattr(refreshed, "done", False):
            return ProviderJobStatus(
                job_id=job_id,
                status="running",
                status_code=200,
                duration_ms=duration_ms,
                assets=assets,
                error=None,
                data=payload,
            )

        if refreshed.error:
            error_text = refreshed.error.get("message") if isinstance(refreshed.error, dict) else str(refreshed.error)
            payload["error"] = refreshed.error
        else:
            result = refreshed.result or refreshed.response
            if result and result.generated_videos:
                primary = next((item.video for item in result.generated_videos if item.video), None)
                if primary:
                    mime = primary.mime_type or "video/mp4"
                    asset_payload: Dict[str, Any] = {
                        "type": "video",
                        "mime": mime,
                        "model": request_meta.get("model"),
                    }
                    video_bytes = getattr(primary, "video_bytes", None)
                    if video_bytes:
                        base64_data = base64.b64encode(video_bytes).decode("ascii")
                        data_uri = f"data:{mime};base64,{base64_data}"
                        inline_entry = {
                            "kind": "video",
                            "mime": mime,
                            "bytes": len(video_bytes),
                            "data_uri": data_uri,
                            "filename": "veo_result.mp4",
                        }
                        inline_assets.append(inline_entry)
                        assets_meta.append(
                            {
                                "key": f"inline_{len(inline_assets) - 1}",
                                "inline": True,
                                "mime": mime,
                                "bytes": len(video_bytes),
                            }
                        )
                        asset_payload.update(
                            {
                                "data_uri": data_uri,
                                "base64": base64_data,
                                "bytes": len(video_bytes),
                            }
                        )
                    if primary.uri:
                        asset_key = "video"
                        assets[asset_key] = primary.uri
                        assets_meta.append(
                            {
                                "key": asset_key,
                                "inline": False,
                                "url": primary.uri,
                                "mime": mime,
                                "bytes": len(video_bytes) if video_bytes else None,
                            }
                        )
                        asset_payload["uri"] = primary.uri
                    config_meta = request_meta.get("config") or {}
                    if config_meta:
                        asset_payload.setdefault("config", config_meta)
                    payload["payload"] = asset_payload
        if inline_assets:
            payload.setdefault("inline_assets", inline_assets)
        if assets_meta:
            payload.setdefault("assets_meta", assets_meta)

        status = "completed" if not error_text else "failed"
        result_error = error_text
        if status == "completed":
            async with self._lock:
                self._operations.pop(job_id, None)
                self._requests.pop(job_id, None)
        else:
            payload.setdefault("error", error_text)
        return ProviderJobStatus(
            job_id=job_id,
            status=status,
            status_code=200,
            duration_ms=duration_ms,
            assets=assets,
            error=result_error,
            data=payload,
        )

    def _generate_videos_operation(
        self,
        model: str,
        source: Optional[_genai_types.GenerateVideosSource],
        config: Optional[_genai_types.GenerateVideosConfig],
    ) -> _genai_types.GenerateVideosOperation:
        models = self._client.models
        if hasattr(models, "generate_videos"):
            return models.generate_videos(model=model, source=source, config=config)
        return models._generate_videos(model=model, source=source, config=config)

    async def _get_operation(self, job_id: str) -> _genai_types.GenerateVideosOperation:
        async with self._lock:
            cached = self._operations.get(job_id)
        if cached is not None:
            return cached
        return _genai_types.GenerateVideosOperation(name=job_id)

    def _build_source(
        self, prompt: str, settings: Dict[str, Any]
    ) -> Optional[_genai_types.GenerateVideosSource]:
        inline = settings.get("reference_inline_data")
        image: Optional[_genai_types.Image] = None
        if isinstance(inline, dict):
            image = _decode_reference_image(inline)
        if not prompt and not image:
            raise RuntimeError("Prompt or reference image is required for video generation")
        return _genai_types.GenerateVideosSource(prompt=prompt or None, image=image)

    def _build_config(
        self, settings: Dict[str, Any]
    ) -> tuple[Optional[_genai_types.GenerateVideosConfig], Dict[str, Any]]:
        config_data: Dict[str, Any] = {}
        summary: Dict[str, Any] = {}

        size = settings.get("size")
        if isinstance(size, str):
            summary["size"] = size
            inferred = _infer_aspect_ratio(size)
            if inferred:
                config_data.setdefault("aspect_ratio", inferred)
                summary.setdefault("aspect_ratio", inferred)
            resolution = _infer_resolution(size)
            if resolution:
                config_data.setdefault("resolution", resolution)
                summary.setdefault("resolution", resolution)

        duration = _coerce_int(settings.get("duration_seconds") or settings.get("duration"))
        if duration:
            config_data["duration_seconds"] = duration
            summary["duration_seconds"] = duration

        fps = _coerce_int(settings.get("fps"))
        if fps:
            config_data["fps"] = fps
            summary["fps"] = fps

        seed = _coerce_int(settings.get("seed"))
        if seed is not None:
            config_data["seed"] = seed
            summary["seed"] = seed

        num_videos = _coerce_int(settings.get("number_of_videos"))
        if num_videos:
            config_data["number_of_videos"] = num_videos
            summary["number_of_videos"] = num_videos

        for key in (
            "aspect_ratio",
            "resolution",
            "negative_prompt",
            "person_generation",
            "output_gcs_uri",
            "pubsub_topic",
            "compression_quality",
        ):
            if key in settings and settings[key] not in (None, ""):
                config_data[key] = settings[key]
                summary[key] = settings[key]

        for key in ("enhance_prompt", "generate_audio"):
            coerced = _coerce_bool(settings.get(key))
            if coerced is not None:
                config_data[key] = coerced
                summary[key] = coerced

        if not config_data:
            return None, summary
        try:
            config_obj = _genai_types.GenerateVideosConfig.model_validate(config_data)
        except Exception:  # pragma: no cover - defensive parsing
            log.warning("Failed to validate Veo config; using raw data", exc_info=True)
            config_obj = _genai_types.GenerateVideosConfig(**config_data)
        return config_obj, summary


__all__ = ["VeoVideoClient"]
