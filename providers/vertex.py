"""Clients for Google Vertex AI video and image generation."""
from __future__ import annotations

import base64
import binascii
import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from google.auth.transport.requests import AuthorizedSession
from google.oauth2.service_account import Credentials

from config import Config

from .base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
)
from .google_auth import ServiceAccountTokenProvider

log = logging.getLogger(__name__)


def list_models(config: Config) -> Tuple[bool, str]:
    """Fetch the available Vertex AI models for the configured project."""

    if not config.google_service_account:
        return False, "missing credentials"

    url = (
        "https://"
        f"{config.vertex_location}-aiplatform.googleapis.com/v1/projects/"
        f"{config.gcp_project_id}/locations/{config.vertex_location}/publishers/google/models"
    )

    try:
        credentials = Credentials.from_service_account_info(
            config.google_service_account,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    except Exception as exc:  # pragma: no cover - auth failure
        return False, f"auth: {exc}"

    try:
        with AuthorizedSession(credentials) as session:
            response = session.get(url, timeout=config.request_timeout or None)
    except Exception as exc:  # pragma: no cover - network guard
        log.warning("Vertex models request failed", exc_info=True)
        return False, str(exc)

    if response.status_code >= 400:
        text = (response.text or "").replace("\n", " ")
        detail = f"{response.status_code}: {text}" if text else str(response.status_code)
        return False, detail

    try:
        payload = response.json() if response.content else {}
    except ValueError:
        return False, "invalid JSON"

    models = payload.get("models") if isinstance(payload, dict) else None
    if isinstance(models, list):
        return True, f"count={len(models)}"
    return True, "empty"


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
        status, error, assets, inline_assets = self._extract_result(data)
        if status != "completed" and not error:
            error = "Не удалось создать. Ошибка неизвестна"
        return ProviderJobStatus(
            job_id=job_id,
            status=status,
            error=error,
            assets=assets,
            data={
                "response": data,
                "meta": meta,
                "inline_assets": inline_assets,
            },
            status_code=status_code,
            duration_ms=duration_ms,
        )

    def _extract_result(
        self, payload: Dict[str, Any]
    ) -> Tuple[str, Optional[str], Dict[str, str], List[Dict[str, Any]]]:
        if not isinstance(payload, dict):
            return "failed", "Пустой ответ провайдера", {}, []
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("status")
            return "failed", f"{message}" if message else "Неизвестная ошибка", {}, []
        candidates = payload.get("candidates")
        if not candidates:
            return "failed", "Пустой ответ модели", {}, []
        first = candidates[0] or {}
        content = first.get("content") if isinstance(first, dict) else {}
        parts = []
        if isinstance(content, dict):
            raw_parts = content.get("parts")
            if isinstance(raw_parts, list):
                parts = [part for part in raw_parts if isinstance(part, dict)]
        assets: Dict[str, str] = {}
        inline_assets: List[Dict[str, Any]] = []
        for part in parts:
            self._collect_assets(part, assets, inline_assets)
        if assets:
            return "completed", None, assets, inline_assets
        finish_reason = first.get("finishReason") if isinstance(first, dict) else None
        if isinstance(finish_reason, str) and finish_reason:
            return "failed", finish_reason, {}, []
        return "failed", "Модель не вернула результат", {}, []

    def _collect_assets(
        self,
        part: Dict[str, Any],
        assets: Dict[str, str],
        inline_assets: List[Dict[str, Any]],
    ) -> None:
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
                mime = (
                    inline.get("mimeType")
                    or inline.get("mime_type")
                    or "application/octet-stream"
                )
                key = f"inline_{len(inline_assets)}"
                data_str = str(data)
                inline_assets.append(
                    {
                        "kind": "inlineData",
                        "mime": mime,
                        "bytes": self._measure_inline_bytes(data_str),
                        "data_uri": f"data:{mime};base64,{data_str}",
                    }
                )
                assets[key] = key

    @staticmethod
    def _measure_inline_bytes(data: str) -> int:
        payload = "".join(str(data).split())
        try:
            decoded = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            try:
                decoded = base64.b64decode(payload)
            except (binascii.Error, ValueError):
                return 0
        return len(decoded)

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
        if dimension == "720x1280":
            width, height, aspect_ratio = 720, 1280, "9:16"
        else:
            width, height, aspect_ratio = 1280, 720, "16:9"

        generation_config: Dict[str, Any] = {
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": duration,
            "fps": 24,
        }

        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": generation_config,
        }
        config.setdefault("size", dimension)
        config.setdefault("duration", duration)
        config.setdefault("width", width)
        config.setdefault("height", height)
        config.setdefault("aspect_ratio", aspect_ratio)
        return body


class VertexVeoPreviewClient(VertexGenerativeClient):
    """Generate Veo 3.1 inline preview videos using the predict endpoint."""

    _GENERATION_CONFIG_ALLOWLIST = {
        "width",
        "height",
        "aspect_ratio",
        "duration_seconds",
        "fps",
        "seed",
        "safety_settings",
    }

    _SIZE_PRESETS: Dict[str, Tuple[int, int, str]] = {
        "1280x720": (1280, 720, "16:9"),
        "720x1280": (720, 1280, "9:16"),
    }

    _SIZE_ALIASES: Dict[str, str] = {
        "horizontal": "1280x720",
        "горизонталь": "1280x720",
        "vertical": "720x1280",
        "вертикаль": "720x1280",
    }

    _ALLOWED_ROOT_KEYS = {"instances"}

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
        invalid_keys: List[str] = []
        validation_errors: List[str] = []

        raw_generation_config = (
            config.get("generation_config")
            if isinstance(config.get("generation_config"), dict)
            else {}
        )
        for key in raw_generation_config:
            if key not in self._GENERATION_CONFIG_ALLOWLIST:
                invalid_keys.append(key)

        size_value = str(config.get("size") or "").strip()
        normalized_size = size_value.lower()
        normalized_size = self._SIZE_ALIASES.get(normalized_size, normalized_size)
        if not normalized_size:
            normalized_size = "1280x720"
        if normalized_size in self._SIZE_PRESETS:
            width_default, height_default, aspect_default = self._SIZE_PRESETS[
                normalized_size
            ]
            size_key = normalized_size
        else:
            width_default = height_default = 0
            aspect_default = ""
            if size_value:
                validation_errors.append(f"size={size_value}")
            size_key = "1280x720"

        def _coerce_positive_int(value: Any, field: str) -> Optional[int]:
            if value is None:
                return None
            try:
                number = int(value)
            except (TypeError, ValueError):
                validation_errors.append(f"{field}=not_int")
                return None
            if number <= 0:
                validation_errors.append(f"{field}=non_positive")
                return None
            return number

        def _reduce_aspect_ratio(width: int, height: int) -> str:
            divisor = math.gcd(width, height) or 1
            return f"{width // divisor}:{height // divisor}"

        width = _coerce_positive_int(
            raw_generation_config.get("width", config.get("width", width_default)),
            "width",
        )
        height = _coerce_positive_int(
            raw_generation_config.get("height", config.get("height", height_default)),
            "height",
        )

        duration_seconds = _coerce_positive_int(
            raw_generation_config.get(
                "duration_seconds",
                config.get("duration_seconds", config.get("duration")),
            ),
            "duration",
        )
        if duration_seconds is None:
            duration_seconds = 6

        fps = _coerce_positive_int(
            raw_generation_config.get(
                "fps", config.get("fps", config.get("frame_rate"))
            ),
            "fps",
        )
        if fps is None:
            fps = 24

        aspect_ratio = raw_generation_config.get(
            "aspect_ratio", config.get("aspect_ratio", aspect_default)
        )
        if not isinstance(aspect_ratio, str) or not aspect_ratio:
            aspect_ratio = ""
        if width and height and not aspect_ratio:
            aspect_ratio = _reduce_aspect_ratio(width, height)

        seed = raw_generation_config.get("seed", config.get("seed"))
        if seed is not None:
            seed = _coerce_positive_int(seed, "seed")

        safety_settings = raw_generation_config.get(
            "safety_settings", config.get("safety_settings")
        )
        if safety_settings is not None and not isinstance(safety_settings, (list, dict)):
            validation_errors.append("safety_settings=invalid_type")
            safety_settings = None

        generation_config: Dict[str, Any] = {}
        if width:
            generation_config["width"] = width
        if height:
            generation_config["height"] = height
        if aspect_ratio:
            generation_config["aspect_ratio"] = aspect_ratio
        if duration_seconds:
            generation_config["duration_seconds"] = duration_seconds
        if fps:
            generation_config["fps"] = fps
        if seed:
            generation_config["seed"] = seed
        if safety_settings is not None:
            generation_config["safety_settings"] = safety_settings

        body = {
            "instances": [
                {
                    "prompt": prompt,
                    "generation_config": generation_config,
                }
            ]
        }

        payload_preview = {
            "instances": [
                {
                    "prompt_preview": (prompt or "")[:120],
                    "generation_config": generation_config,
                }
            ]
        }

        invalid_keys = sorted(set(invalid_keys))
        validation_errors = sorted(set(validation_errors))

        if invalid_keys or validation_errors:
            log.warning(
                "vertex.veo_preview.invalid_payload invalid_keys=%s validation_errors=%s payload_preview=%s",
                invalid_keys,
                validation_errors,
                payload_preview,
            )
            raise ProviderAPIError(
                provider=self.provider_name,
                status_code=0,
                message="Неверные параметры запроса. Проверь размер, длительность и модель.",
                error_type="invalid_request",
            )

        extra_root_keys = sorted(set(body.keys()) - self._ALLOWED_ROOT_KEYS)
        if extra_root_keys:
            log.warning(
                "vertex.veo_preview.invalid_payload invalid_keys=%s payload_preview=%s",
                extra_root_keys,
                payload_preview,
            )
            raise ProviderAPIError(
                provider=self.provider_name,
                status_code=0,
                message="Неверные параметры запроса. Проверь размер, длительность и модель.",
                error_type="invalid_request",
            )

        log.info(
            "vertex.veo_preview.payload_preview invalid_keys=%s validation_errors=%s payload_preview=%s",
            invalid_keys,
            validation_errors,
            payload_preview,
        )

        config.setdefault("size", size_key)
        config.setdefault("duration_seconds", duration_seconds)
        config.setdefault("fps", fps)
        if aspect_ratio:
            config.setdefault("aspect_ratio", aspect_ratio)
        if width:
            config.setdefault("width", width)
        if height:
            config.setdefault("height", height)
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
