"""Clients for Google Vertex AI video and image generation."""
from __future__ import annotations

import base64
import binascii
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from dataclasses import dataclass

from google.auth.transport.requests import AuthorizedSession
from google.oauth2.service_account import Credentials

try:  # Optional google-genai client
    from google.genai import Client as _GenAIClient
    from google.genai import models as _genai_models
    from google.genai import types as _genai_types
except ImportError:  # pragma: no cover - optional dependency
    _GenAIClient = None
    _genai_models = None
    _genai_types = None

try:  # Optional legacy Vertex SDK
    from vertexai.preview.generative_models import GenerativeModel as _VertexGenerativeModel
except ImportError:  # pragma: no cover - optional dependency
    _VertexGenerativeModel = None

from config import Config

from .base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
)
from .google_auth import ServiceAccountTokenProvider

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _VeoSanitisedConfig:
    config_fields: Dict[str, Any]
    accepted_args: List[str]
    rejected_args: List[str]
    validation_errors: List[str]
    user_messages: List[str]
    payload_preview: Dict[str, Any]
    reference_inline: Optional[Dict[str, Any]]


@dataclass(slots=True)
class _VeoRequestPlan:
    client: str
    body: Dict[str, Any]
    accepted_args: List[str]
    rejected_args: List[str]
    payload_preview: Dict[str, Any]


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


_VEO_ALLOWED_DURATIONS = {4, 6, 8}
_VEO_ALLOWED_ASPECT_RATIOS = {"16:9", "9:16"}
_VEO_ALLOWED_RESOLUTIONS = {"720p", "1080p"}

_VEO_ASPECT_ALIASES = {
    "horizontal": "16:9",
    "горизонталь": "16:9",
    "vertical": "9:16",
    "вертикаль": "9:16",
    "16x9": "16:9",
    "9x16": "9:16",
}

_VEO_SIZE_TO_ASPECT = {
    "1280x720": "16:9",
    "720x1280": "9:16",
}

_VEO_RESOLUTION_ALIASES = {
    "hd": "720p",
    "720": "720p",
    "1280x720": "720p",
    "720p": "720p",
    "full hd": "1080p",
    "full_hd": "1080p",
    "fhd": "1080p",
    "1080": "1080p",
    "1080p": "1080p",
}

_VEO_TRUE_VALUES = {"1", "true", "yes", "on", "да"}
_VEO_FALSE_VALUES = {"0", "false", "no", "off", "нет"}

_GOOGLE_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

_VEO_MODEL_ALIASES = {
    "veo-3.0-generate-001": "veo-3.0",
    "veo-3.0": "veo-3.0",
    "veo-3": "veo-3.0",
    "veo3": "veo-3.0",
    "veo-3.1-generate-preview": "veo-3.1",
    "veo-3.1": "veo-3.1",
    "veo3.1": "veo-3.1",
    "veo31": "veo-3.1",
}


def _coerce_int(value: Any, *, field: str, positive: bool = True, errors: List[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        errors.append(f"{field}=not_int")
        return None
    if positive and number <= 0:
        errors.append(f"{field}=non_positive")
        return None
    return number


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _VEO_TRUE_VALUES:
            return True
        if normalized in _VEO_FALSE_VALUES:
            return False
    return None


def _normalise_aspect_ratio(config: Dict[str, Any]) -> Optional[str]:
    raw_aspect = config.get("aspectRatio") or config.get("aspect_ratio")
    if isinstance(raw_aspect, str) and raw_aspect.strip():
        normalized = raw_aspect.strip().lower()
        return _VEO_ASPECT_ALIASES.get(normalized, raw_aspect.strip())
    orientation = str(config.get("orientation") or "").strip().lower()
    if orientation:
        mapped = _VEO_ASPECT_ALIASES.get(orientation)
        if mapped:
            return mapped
    size_value = str(config.get("size") or "").strip().lower()
    if size_value:
        size_mapped = _VEO_ASPECT_ALIASES.get(size_value)
        if size_mapped:
            return size_mapped
        size_mapped = _VEO_SIZE_TO_ASPECT.get(size_value)
        if size_mapped:
            return size_mapped
    return None


def _normalise_resolution(config: Dict[str, Any]) -> Optional[str]:
    raw_resolution = config.get("resolution") or config.get("quality")
    if isinstance(raw_resolution, str) and raw_resolution.strip():
        normalized = raw_resolution.strip().lower()
        return _VEO_RESOLUTION_ALIASES.get(normalized, raw_resolution.strip())
    size_value = str(config.get("size") or "").strip().lower()
    if size_value:
        mapped = _VEO_RESOLUTION_ALIASES.get(size_value)
        if mapped:
            return mapped
    return None


def _detect_veo_client() -> str:
    if _GenAIClient and _genai_models and _genai_types:
        return "genai"
    if _VertexGenerativeModel:
        return "vertexai"
    return "rest"


def _normalise_veo_model(model: str) -> str:
    if not isinstance(model, str):
        return model
    key = model.strip()
    return _VEO_MODEL_ALIASES.get(key, key)


def _prepare_veo_config(config: Dict[str, Any]) -> _VeoSanitisedConfig:
    provided_keys = {
        key for key in config.keys() if isinstance(key, str)
    }
    consumed_keys: set[str] = set()
    config_fields: Dict[str, Any] = {}
    accepted_args: List[str] = []
    validation_errors: List[str] = []
    user_messages: List[str] = []
    payload_preview: Dict[str, Any] = {}

    unsupported_map = {
        "fps": "fps недоступен для Veo",
        "width": "width/height не поддерживаются, выбери 16:9 или 9:16 и 720p/1080p",
        "height": "width/height не поддерживаются, выбери 16:9 или 9:16 и 720p/1080p",
        "parameters": "Поле parameters недоступно для Veo",
        "generation_config": "Поле generation_config недоступно для Veo",
        "generationConfig": "Поле generation_config недоступно для Veo",
    }
    for key, message in unsupported_map.items():
        if key in config:
            consumed_keys.add(key)
            if message not in user_messages:
                user_messages.append(message)

    duration_value: Optional[int] = None
    for key in ("durationSeconds", "duration_seconds", "duration"):
        if key in config:
            consumed_keys.add(key)
            duration_value = _coerce_int(
                config.get(key), field="duration", positive=True, errors=validation_errors
            )
            break
    if duration_value is None:
        duration_value = 6
    if duration_value is not None:
        if duration_value not in _VEO_ALLOWED_DURATIONS:
            validation_errors.append(f"duration={duration_value}")
        else:
            config_fields["duration_seconds"] = duration_value
            payload_preview["durationSeconds"] = duration_value
            accepted_args.append("duration_seconds")

    for key in ("aspectRatio", "aspect_ratio", "orientation", "size"):
        if key in config:
            consumed_keys.add(key)
    aspect_ratio_value = _normalise_aspect_ratio(config)
    if aspect_ratio_value:
        if aspect_ratio_value not in _VEO_ALLOWED_ASPECT_RATIOS:
            validation_errors.append(f"aspectRatio={aspect_ratio_value}")
        else:
            config_fields["aspect_ratio"] = aspect_ratio_value
            payload_preview["aspectRatio"] = aspect_ratio_value
            accepted_args.append("aspect_ratio")
    else:
        config_fields.setdefault("aspect_ratio", "16:9")
        payload_preview.setdefault("aspectRatio", "16:9")
        if "aspect_ratio" not in accepted_args:
            accepted_args.append("aspect_ratio")

    for key in ("resolution", "quality"):
        if key in config:
            consumed_keys.add(key)
    resolution_value = _normalise_resolution(config)
    if resolution_value:
        if resolution_value not in _VEO_ALLOWED_RESOLUTIONS:
            validation_errors.append(f"resolution={resolution_value}")
        else:
            config_fields["resolution"] = resolution_value
            payload_preview["resolution"] = resolution_value
            accepted_args.append("resolution")
    else:
        config_fields.setdefault("resolution", "720p")
        payload_preview.setdefault("resolution", "720p")
        if "resolution" not in accepted_args:
            accepted_args.append("resolution")

    generate_audio_value: Optional[bool] = None
    for key in ("generateAudio", "generate_audio", "audio", "sound"):
        if key in config:
            consumed_keys.add(key)
            generate_audio_value = _coerce_bool(config.get(key))
            if generate_audio_value is None:
                validation_errors.append("generateAudio=invalid")
            break
    if generate_audio_value is not None:
        config_fields["generate_audio"] = generate_audio_value
        payload_preview["generateAudio"] = generate_audio_value
        accepted_args.append("generate_audio")

    if "seed" in config:
        consumed_keys.add("seed")
        seed_value = _coerce_int(
            config.get("seed"), field="seed", positive=False, errors=validation_errors
        )
        if seed_value is not None:
            config_fields["seed"] = seed_value
            payload_preview["seed"] = seed_value
            accepted_args.append("seed")

    sample_value: Optional[int] = None
    for key in ("sampleCount", "sample_count", "samples"):
        if key in config:
            consumed_keys.add(key)
            sample_value = _coerce_int(
                config.get(key), field="sampleCount", positive=True, errors=validation_errors
            )
            break
    if sample_value is not None and sample_value > 0:
        config_fields["number_of_videos"] = sample_value
        payload_preview["sampleCount"] = sample_value
        accepted_args.append("number_of_videos")

    reference_inline = config.get("reference_inline_data")
    if isinstance(reference_inline, dict):
        consumed_keys.add("reference_inline_data")
        if reference_inline.get("data"):
            payload_preview["hasReference"] = True
        else:
            reference_inline = None
    else:
        reference_inline = None

    rejected_args = sorted(provided_keys - consumed_keys)

    return _VeoSanitisedConfig(
        config_fields=config_fields,
        accepted_args=sorted(set(accepted_args)),
        rejected_args=rejected_args,
        validation_errors=sorted(set(validation_errors)),
        user_messages=user_messages,
        payload_preview=payload_preview,
        reference_inline=reference_inline,
    )

def _build_veo_parameters(
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str], List[str], bool]:
    invalid_keys: List[str] = []
    validation_errors: List[str] = []

    forbidden_root = {
        "generation_config",
        "generationConfig",
        "width",
        "height",
    }
    for forbidden in forbidden_root:
        if forbidden in config:
            invalid_keys.append(forbidden)

    raw_parameters = config.get("parameters") if isinstance(config.get("parameters"), dict) else {}
    allowed_parameter_keys = {
        "durationSeconds",
        "aspectRatio",
        "resolution",
        "generateAudio",
        "seed",
        "sampleCount",
    }
    extra_raw_parameter_keys = sorted(
        set(raw_parameters.keys()) - allowed_parameter_keys
    )
    if extra_raw_parameter_keys:
        invalid_keys.extend(extra_raw_parameter_keys)

    parameters: Dict[str, Any] = {
        key: raw_parameters[key]
        for key in allowed_parameter_keys
        if key in raw_parameters
    }

    duration_seconds = config.get("durationSeconds")
    if duration_seconds is None:
        duration_seconds = config.get("duration_seconds")
    if duration_seconds is None:
        duration_seconds = config.get("duration")
    duration_value = _coerce_int(
        duration_seconds, field="duration", positive=True, errors=validation_errors
    )
    if duration_value is not None:
        if duration_value not in _VEO_ALLOWED_DURATIONS:
            validation_errors.append(f"duration={duration_value}")
        else:
            parameters["durationSeconds"] = duration_value

    aspect_ratio_value = _normalise_aspect_ratio(config)
    if aspect_ratio_value:
        if aspect_ratio_value not in _VEO_ALLOWED_ASPECT_RATIOS:
            validation_errors.append(f"aspectRatio={aspect_ratio_value}")
        else:
            parameters["aspectRatio"] = aspect_ratio_value

    resolution_value = _normalise_resolution(config)
    if resolution_value:
        if resolution_value not in _VEO_ALLOWED_RESOLUTIONS:
            validation_errors.append(f"resolution={resolution_value}")
        else:
            parameters["resolution"] = resolution_value

    generate_audio_source = None
    for key in ("generateAudio", "generate_audio", "audio", "sound"):
        if key in config:
            generate_audio_source = config[key]
            break
    if generate_audio_source is None and "generateAudio" in raw_parameters:
        generate_audio_source = raw_parameters["generateAudio"]
    generate_audio_value = _coerce_bool(generate_audio_source)
    if generate_audio_value is None:
        generate_audio_value = False
    parameters["generateAudio"] = generate_audio_value

    seed_value = config.get("seed")
    if seed_value is None and "seed" in raw_parameters:
        seed_value = raw_parameters["seed"]
    seed_int = _coerce_int(seed_value, field="seed", positive=False, errors=validation_errors)
    if seed_int is not None:
        parameters["seed"] = seed_int

    sample_count_value = config.get("sampleCount")
    if sample_count_value is None:
        sample_count_value = config.get("sample_count")
    if sample_count_value is None and "sampleCount" in raw_parameters:
        sample_count_value = raw_parameters["sampleCount"]
    sample_count_int = _coerce_int(
        sample_count_value, field="sampleCount", positive=True, errors=validation_errors
    )
    if sample_count_int is not None:
        parameters["sampleCount"] = sample_count_int

    if "durationSeconds" not in parameters:
        parameters["durationSeconds"] = 6
    if parameters["durationSeconds"] not in _VEO_ALLOWED_DURATIONS:
        validation_errors.append(f"duration={parameters['durationSeconds']}")

    if "aspectRatio" not in parameters:
        parameters["aspectRatio"] = "16:9"
    if parameters["aspectRatio"] not in _VEO_ALLOWED_ASPECT_RATIOS:
        validation_errors.append(f"aspectRatio={parameters['aspectRatio']}")

    if "resolution" not in parameters:
        parameters["resolution"] = "720p"
    if parameters["resolution"] not in _VEO_ALLOWED_RESOLUTIONS:
        validation_errors.append(f"resolution={parameters['resolution']}")

    reference = config.get("reference_inline_data")
    has_reference_inline = bool(
        isinstance(reference, dict) and reference.get("data")
    )

    return parameters, sorted(set(invalid_keys)), sorted(set(validation_errors)), has_reference_inline


def veo_request(
    *,
    prompt: str,
    settings: Optional[Dict[str, Any]],
    model: str,
    config: Config,
    provider_name: str,
) -> _VeoRequestPlan:
    raw_settings = settings or {}
    sanitised = _prepare_veo_config(raw_settings)
    model_name = _normalise_veo_model(model)

    if sanitised.user_messages:
        raise ProviderAPIError(
            provider=provider_name,
            status_code=0,
            message=sanitised.user_messages[0],
            error_type="invalid_request",
        )

    if sanitised.validation_errors:
        raise ProviderAPIError(
            provider=provider_name,
            status_code=0,
            message=(
                "Неверные параметры для Veo. Длительность поддерживается 4, 6 или 8 секунд; "
                "соотношение сторон 16:9 или 9:16; разрешение 720p или 1080p."
            ),
            error_type="invalid_request",
        )

    selected_client = _detect_veo_client()

    if selected_client == "genai" and _GenAIClient and _genai_types and _genai_models:
        if not config.google_service_account:
            raise ProviderAPIError(
                provider=provider_name,
                status_code=0,
                message="Не настроен сервисный аккаунт Google для Veo",
                error_type="config",
            )
        try:
            credentials = Credentials.from_service_account_info(
                config.google_service_account,
                scopes=_GOOGLE_SCOPES,
            )
        except Exception as exc:  # pragma: no cover - auth failure
            raise ProviderAPIError(
                provider=provider_name,
                status_code=0,
                message="Не удалось создать учетные данные для Veo",
                error_type="config",
                provider_message=str(exc),
            ) from exc

        client = _GenAIClient(
            vertexai=True,
            project=config.gcp_project_id,
            location=config.vertex_location,
            credentials=credentials,
        )

        config_kwargs = dict(sanitised.config_fields)

        reference_inline = sanitised.reference_inline
        if reference_inline and _genai_types:
            data = reference_inline.get("data")
            if isinstance(data, str):
                payload = "".join(data.split())
                try:
                    binary = base64.b64decode(payload, validate=True)
                except (binascii.Error, ValueError):
                    binary = None
                if binary:
                    image_kwargs = {"image_bytes": binary}
                    mime = reference_inline.get("mime_type") or reference_inline.get("mimeType")
                    if isinstance(mime, str) and mime:
                        image_kwargs["mime_type"] = mime
                    reference_image = _genai_types.VideoGenerationReferenceImage(
                        image=_genai_types.Image(**image_kwargs)
                    )
                    config_kwargs.setdefault("reference_images", []).append(reference_image)

        if config_kwargs:
            config_obj = _genai_types.GenerateVideosConfig(**config_kwargs)
        else:
            config_obj = None

        parameters = _genai_types._GenerateVideosParameters(
            model=model_name,
            prompt=prompt,
            config=config_obj,
        )
        request_dict = _genai_models._GenerateVideosParameters_to_vertex(
            client._api_client, parameters
        )
        request_dict.pop("_url", None)
        body = request_dict

    elif selected_client == "vertexai" and _VertexGenerativeModel:
        instance = {
            "prompt": prompt,
        }
        if sanitised.config_fields.get("duration_seconds") is not None:
            instance["durationSeconds"] = sanitised.config_fields["duration_seconds"]
        if sanitised.config_fields.get("aspect_ratio"):
            instance["aspectRatio"] = sanitised.config_fields["aspect_ratio"]
        if sanitised.config_fields.get("resolution"):
            instance["resolution"] = sanitised.config_fields["resolution"]
        if sanitised.config_fields.get("generate_audio") is not None:
            instance["generateAudio"] = sanitised.config_fields["generate_audio"]
        if sanitised.config_fields.get("seed") is not None:
            instance["seed"] = sanitised.config_fields["seed"]
        if sanitised.config_fields.get("number_of_videos"):
            instance["sampleCount"] = sanitised.config_fields["number_of_videos"]
        body = {"instances": [instance]}
    else:
        instance = {"prompt": prompt}
        if sanitised.config_fields.get("duration_seconds") is not None:
            instance["durationSeconds"] = sanitised.config_fields["duration_seconds"]
        if sanitised.config_fields.get("aspect_ratio"):
            instance["aspectRatio"] = sanitised.config_fields["aspect_ratio"]
        if sanitised.config_fields.get("resolution"):
            instance["resolution"] = sanitised.config_fields["resolution"]
        if sanitised.config_fields.get("generate_audio") is not None:
            instance["generateAudio"] = sanitised.config_fields["generate_audio"]
        if sanitised.config_fields.get("seed") is not None:
            instance["seed"] = sanitised.config_fields["seed"]
        if sanitised.config_fields.get("number_of_videos"):
            instance["sampleCount"] = sanitised.config_fields["number_of_videos"]
        body = {"instances": [instance]}

    log.info(
        "vertex.veo.request selected_client=%s accepted_args=%s rejected_args=%s payload_preview=%s",
        selected_client,
        sanitised.accepted_args,
        sanitised.rejected_args,
        sanitised.payload_preview,
    )

    return _VeoRequestPlan(
        client=selected_client,
        body=body,
        accepted_args=sanitised.accepted_args,
        rejected_args=sanitised.rejected_args,
        payload_preview=sanitised.payload_preview,
    )


class VertexVideoClient(VertexGenerativeClient):
    """Generate video clips using Veo models on Vertex AI."""

    def _jobs_path(self) -> str:
        return f"/{self._model}:predictLongRunning"

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
        plan = veo_request(
            prompt=prompt,
            settings=settings,
            model=self._model,
            config=self._config,
            provider_name=self.provider_name,
        )
        return plan.body

    async def enqueue_job(
        self,
        *,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderJobSubmission:
        if payload is not None:
            return await super().enqueue_job(
                prompt=prompt,
                settings=settings,
                payload=payload,
                idempotency_key=idempotency_key,
                **kwargs,
            )

        plan = veo_request(
            prompt=prompt,
            settings=settings,
            model=self._model,
            config=self._config,
            provider_name=self.provider_name,
        )
        data, status_code, duration_ms = await self._request(
            "POST",
            self._jobs_path(),
            json=plan.body,
            idempotency_key=idempotency_key,
        )
        job_id = idempotency_key or str(uuid4())
        meta = {
            "prompt": prompt,
            "settings": settings or {},
            "payload": plan.body,
            "assets_meta": {},
            "veo": {
                "selected_client": plan.client,
                "accepted_args": plan.accepted_args,
                "rejected_args": plan.rejected_args,
                "payload_preview": plan.payload_preview,
            },
        }
        self._pending[job_id] = (data, status_code, duration_ms, meta)
        size = (settings or {}).get("size")
        duration = (settings or {}).get("duration")
        log.info(
            "vertex.call corr_id=%s provider=%s model=%s size=%s duration=%s status=%s latency_ms=%s client=%s",
            idempotency_key or job_id,
            self.provider_name,
            self._model,
            size or "",
            duration or "",
            status_code,
            duration_ms,
            plan.client,
        )
        return ProviderJobSubmission(
            job_id=job_id,
            status_code=status_code,
            duration_ms=duration_ms,
            data=data,
        )


class VertexVeoPreviewClient(VertexGenerativeClient):
    """Generate Veo 3.1 inline preview videos using the predict endpoint."""

    _ALLOWED_ROOT_KEYS = {"instances"}
    _ALLOWED_INSTANCE_KEYS = {"prompt", "parameters"}

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

        parameters, invalid_keys, validation_errors, _ = _build_veo_parameters(config)

        if invalid_keys or validation_errors:
            log.warning(
                "vertex.veo_preview.invalid_params invalid_keys=%s validation_errors=%s",
                invalid_keys,
                validation_errors,
            )
            raise ProviderAPIError(
                provider=self.provider_name,
                status_code=0,
                message=(
                    "Неверные параметры для Veo. Длительность поддерживается 4, 6 или 8 секунд; "
                    "соотношение сторон 16:9 или 9:16; разрешение 720p или 1080p."
                ),
                error_type="invalid_request",
            )

        instance = {
            "prompt": prompt,
            "parameters": parameters,
        }

        extra_instance_keys = sorted(set(instance.keys()) - self._ALLOWED_INSTANCE_KEYS)
        if extra_instance_keys:
            log.warning(
                "vertex.veo_preview.invalid_instance_keys keys=%s", extra_instance_keys
            )
            raise ProviderAPIError(
                provider=self.provider_name,
                status_code=0,
                message=(
                    "Неверные параметры для Veo. Длительность поддерживается 4, 6 или 8 секунд; "
                    "соотношение сторон 16:9 или 9:16; разрешение 720p или 1080p."
                ),
                error_type="invalid_request",
            )

        body = {"instances": [instance]}

        extra_root_keys = sorted(set(body.keys()) - self._ALLOWED_ROOT_KEYS)
        if extra_root_keys:
            log.warning(
                "vertex.veo_preview.invalid_payload root_keys=%s", extra_root_keys
            )
            raise ProviderAPIError(
                provider=self.provider_name,
                status_code=0,
                message=(
                    "Неверные параметры для Veo. Длительность поддерживается 4, 6 или 8 секунд; "
                    "соотношение сторон 16:9 или 9:16; разрешение 720p или 1080p."
                ),
                error_type="invalid_request",
            )

        log.info(
            "vertex.veo_preview.payload_preview parameters=%s",
            {
                "durationSeconds": parameters.get("durationSeconds"),
                "aspectRatio": parameters.get("aspectRatio"),
                "resolution": parameters.get("resolution"),
                "generateAudio": parameters.get("generateAudio"),
                "seed": parameters.get("seed"),
                "sampleCount": parameters.get("sampleCount"),
            },
        )

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
