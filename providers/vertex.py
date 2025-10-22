"""Clients for Google Vertex AI video and image generation."""
from __future__ import annotations

import base64
import binascii
import logging
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from config import Config

from .base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
)
from .google_auth import ServiceAccountTokenProvider

log = logging.getLogger(__name__)


class VertexGenerativeClient(BaseProviderClient):
    """Base client for synchronous Vertex AI generateContent calls."""

    def __init__(
        self,
        *,
        config: Config,
        model: str,
        method: str,
        provider_name: str,
    ) -> None:
        endpoint = (
            f"https://{config.vertex_location}-aiplatform.googleapis.com/v1/projects/"
            f"{config.gcp_project_id}/locations/{config.vertex_location}/publishers/google/models"
        )
        super().__init__(
            config=config,
            base_url=endpoint,
            api_key="",
            provider_name=provider_name,
        )
        self._model = model
        self._method = method
        self._pending: Dict[str, Tuple[Dict[str, Any], int, int, Dict[str, Any]]] = {}
        self._token_provider = ServiceAccountTokenProvider(
            config.google_service_account,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

    def _build_headers(self) -> Dict[str, str]:
        headers = dict(self._default_headers)
        headers.setdefault("Content-Type", "application/json")
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        idempotency_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], int, int]:
        try:
            token = await self._token_provider.get_token()
        except Exception as exc:  # pragma: no cover - auth failure
            raise ProviderAPIError(
                provider=self.provider_name,
                status_code=0,
                message="Failed to obtain Vertex access token",
                error_type="auth",
                provider_message=str(exc),
            ) from exc
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {token}"
        return await super()._request(
            method,
            path,
            idempotency_key=idempotency_key,
            headers=headers,
            **kwargs,
        )

    def _jobs_path(self) -> str:
        return f"/{self._model}:{self._method}"

    def _build_error(self, status: int, text: str, duration_ms: int) -> ProviderAPIError:
        error = super()._build_error(status, text, duration_ms)
        if status == 429:
            error.retryable = True
        return error

    async def enqueue_job(
        self,
        *,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        **_: Any,
    ) -> ProviderJobSubmission:
        body = self._build_enqueue_payload(
            prompt=prompt, settings=settings, payload=payload
        )
        data, status_code, duration_ms = await self._request(
            "POST",
            self._jobs_path(),
            json=body,
            idempotency_key=idempotency_key,
        )
        job_id = idempotency_key or str(uuid4())
        meta = {
            "prompt": prompt,
            "settings": settings or {},
            "payload": body,
            "assets_meta": {},
        }
        self._pending[job_id] = (data, status_code, duration_ms, meta)
        size = (settings or {}).get("size")
        duration = (settings or {}).get("duration")
        log.info(
            "vertex.call corr_id=%s provider=%s model=%s size=%s duration=%s status=%s latency_ms=%s",
            idempotency_key or job_id,
            self.provider_name,
            self._model,
            size or "",
            duration or "",
            status_code,
            duration_ms,
        )
        return ProviderJobSubmission(
            job_id=job_id,
            status_code=status_code,
            duration_ms=duration_ms,
            data=data,
        )

    async def get_job_status(self, job_id: str) -> ProviderJobStatus:
        record = self._pending.pop(job_id, None)
        if record is None:
            return ProviderJobStatus(
                job_id=job_id,
                status="failed",
                error="Result not found",
                assets={},
                data={},
                status_code=404,
                duration_ms=0,
            )
        data, status_code, duration_ms, meta = record
        status, error, assets, assets_meta = self._extract_result(data)
        if isinstance(meta, dict):
            meta["assets_meta"] = assets_meta
        if status != "completed" and not error:
            error = "Не удалось создать. Ошибка неизвестна"
        return ProviderJobStatus(
            job_id=job_id,
            status=status,
            error=error,
            assets=assets,
            data={"response": data, "meta": meta, "assets_meta": assets_meta},
            status_code=status_code,
            duration_ms=duration_ms,
        )

    def _extract_result(
        self, payload: Dict[str, Any]
    ) -> Tuple[str, Optional[str], Dict[str, str], Dict[str, Any]]:
        if not isinstance(payload, dict):
            return "failed", "Пустой ответ провайдера", {}, {}
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("status")
            return "failed", f"{message}" if message else "Неизвестная ошибка", {}, {}
        candidates = payload.get("candidates")
        if not candidates:
            return "failed", "Пустой ответ модели", {}, {}
        first = candidates[0] or {}
        content = first.get("content") if isinstance(first, dict) else {}
        parts = []
        if isinstance(content, dict):
            raw_parts = content.get("parts")
            if isinstance(raw_parts, list):
                parts = [part for part in raw_parts if isinstance(part, dict)]
        assets: Dict[str, str] = {}
        assets_meta: Dict[str, Any] = {}
        for part in parts:
            self._collect_assets(part, assets)
        if assets:
            return "completed", None, assets, assets_meta
        finish_reason = first.get("finishReason") if isinstance(first, dict) else None
        if isinstance(finish_reason, str) and finish_reason:
            return "failed", finish_reason, {}, {}
        return "failed", "Модель не вернула результат", {}, {}

    def _collect_assets(self, part: Dict[str, Any], assets: Dict[str, str]) -> None:
        media = part.get("media")
        if isinstance(media, dict):
            uri = media.get("uri") or media.get("downloadUri")
            if uri:
                assets[f"media_{len(assets)}"] = str(uri)
        file_data = part.get("fileData")
        if isinstance(file_data, dict):
            uri = file_data.get("fileUri") or file_data.get("uri")
            if uri:
                assets[f"media_{len(assets)}"] = str(uri)
        inline = part.get("inlineData") or part.get("inline_data")
        if isinstance(inline, dict):
            data = inline.get("data")
            if data:
                mime = inline.get("mimeType") or inline.get("mime_type") or "application/octet-stream"
                assets[f"inline_{len(assets)}"] = f"data:{mime};base64,{data}"

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
        raise NotImplementedError


class VertexVideoClient(VertexGenerativeClient):
    """Generate video clips using Veo models on Vertex AI."""

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
        dimension = str(config.get("size") or "1280x720")
        if dimension not in {"1280x720", "720x1280"}:
            dimension = "1280x720"
        duration = int(config.get("duration") or 6)
        if duration not in {4, 6, 8}:
            duration = 6
        reference = config.get("reference_inline_data")
        parts = [{"text": prompt}]
        if isinstance(reference, dict) and reference.get("data"):
            inline = {
                "mime_type": reference.get("mime_type", "image/jpeg"),
                "data": reference.get("data"),
            }
            parts.append({"inline_data": inline})
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": {
                "video": {
                    "dimension": dimension,
                    "duration_seconds": duration,
                    "fps": 24,
                }
            },
        }
        config.setdefault("size", dimension)
        config.setdefault("duration", duration)
        return body


class VertexVeoPreviewClient(VertexGenerativeClient):
    """Generate Veo 3.1 inline preview videos using the predict endpoint."""

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
        duration = int(config.get("duration_seconds") or config.get("duration") or 6)
        if duration <= 0:
            duration = 6
        fps = int(config.get("fps") or 24)
        if fps <= 0:
            fps = 24
        aspect_ratio = str(config.get("aspect_ratio") or "16:9")
        generation_config = {
            "duration_seconds": duration,
            "fps": fps,
            "aspect_ratio": aspect_ratio,
        }
        body = {
            "instances": [
                {
                    "prompt": prompt,
                    "generation_config": generation_config,
                }
            ]
        }
        config.setdefault("duration_seconds", duration)
        config.setdefault("fps", fps)
        config.setdefault("aspect_ratio", aspect_ratio)
        return body

    def _extract_result(
        self, payload: Dict[str, Any]
    ) -> Tuple[str, Optional[str], Dict[str, str], Dict[str, Any]]:
        if not isinstance(payload, dict):
            return "failed", "Пустой ответ провайдера", {}, {}
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("status")
            return "failed", message or "Неизвестная ошибка", {}, {}
        predictions = payload.get("predictions")
        if not isinstance(predictions, list) or not predictions:
            return "failed", "Пустой ответ модели", {}, {}
        assets: Dict[str, str] = {}
        assets_meta: Dict[str, Any] = {}
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            encoded_raw = prediction.get("bytesBase64Encoded") or prediction.get("data")
            if not encoded_raw:
                continue
            encoded = "".join(str(encoded_raw).split())
            if not encoded:
                continue
            mime = (
                prediction.get("mimeType")
                or prediction.get("mime_type")
                or "video/mp4"
            )
            key = f"inline_video_{len(assets)}"
            assets[key] = f"data:{mime};base64,{encoded}"
            meta: Dict[str, Any] = {"mime_type": mime, "source": "inline"}
            duration_value = prediction.get("durationSeconds") or prediction.get(
                "duration_seconds"
            )
            if isinstance(duration_value, (int, float)) and duration_value > 0:
                meta["duration_seconds"] = float(duration_value)
            fps_value = prediction.get("fps")
            if isinstance(fps_value, (int, float)) and fps_value > 0:
                meta["fps"] = float(fps_value)
            try:
                decoded = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError):
                decoded = None
            if decoded is not None:
                meta["size_bytes"] = len(decoded)
            assets_meta[key] = meta
        metadata = payload.get("metadata")
        if isinstance(metadata, dict) and assets_meta:
            for meta in assets_meta.values():
                meta.setdefault("response_metadata", metadata)
        if assets:
            return "completed", None, assets, assets_meta
        return "failed", "Модель не вернула результат", {}, {}


class VertexImageClient(VertexGenerativeClient):
    """Generate images using Gemini models on Vertex AI."""

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
        reference = (settings or {}).get("reference_inline_data")
        parts = [{"text": prompt}]
        if isinstance(reference, dict) and reference.get("data"):
            inline = {
                "mime_type": reference.get("mime_type", "image/jpeg"),
                "data": reference.get("data"),
            }
            parts.append({"inline_data": inline})
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ]
        }
