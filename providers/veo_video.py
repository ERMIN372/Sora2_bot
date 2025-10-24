"""Client wrapper for Gemini Veo video generation."""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import time
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from uuid import uuid4

from google.genai import Client as _GenAIClient, errors as _genai_errors, types as _genai_types

from config import Config
from services.gemini_client import get_media_client
from services.gemini_key import ensure_gemini_key_logged
from services.gemini_router import GeminiRoutingError, get_gemini_router

from .base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
)

log = logging.getLogger(__name__)

_ASSET_ID_KEYS: Sequence[str] = (
    "file_id",
    "fileId",
    "file",
    "name",
    "id",
)
_ASSET_URL_KEYS: Sequence[str] = (
    "download_uri",
    "downloadUri",
    "signed_uri",
    "signedUri",
    "uri",
    "url",
)
_MIME_KEYS: Sequence[str] = (
    "mime",
    "mime_type",
    "mimeType",
    "content_type",
    "contentType",
)
_DURATION_KEYS: Sequence[str] = (
    "duration_seconds",
    "durationSeconds",
    "duration",
    "durationSec",
    "durationMillis",
    "duration_millis",
    "durationMs",
)
_SIZE_KEYS: Sequence[str] = (
    "size_bytes",
    "sizeBytes",
    "size",
    "bytes",
)
_FILENAME_KEYS: Sequence[str] = (
    "filename",
    "file_name",
    "display_name",
    "displayName",
)
_QUALITY_KEYS: Sequence[str] = (
    "quality",
    "quality_label",
    "qualityLabel",
    "resolution",
    "variant",
    "name",
)


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


def _to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(exclude_none=True)
        except Exception:  # pragma: no cover - defensive parsing
            log.debug("Failed to dump Veo operation object", exc_info=True)
    return {}


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _first_str(data: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        raw = data.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _first_number(data: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        raw = data.get(key)
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                continue
            try:
                return float(text)
            except ValueError:
                continue
    return None


def _normalise_quality_label(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if "720" in lowered:
        return "720p"
    if "540" in lowered:
        return "540p"
    if "480" in lowered:
        return "480p"
    if "preview" in lowered:
        return "preview"
    return cleaned


def _coerce_duration_seconds(value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    if value <= 0:
        return None
    return int(round(value))


def _collect_candidate_file_ids(payload: Dict[str, Any]) -> List[str]:
    candidates: Set[str] = set()
    stack: List[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for value in node.values():
                if isinstance(value, str):
                    text = value.strip()
                    if not text:
                        continue
                    if text.startswith("files/"):
                        candidates.add(text.split("?", 1)[0])
                    elif text.startswith("file-"):
                        base = text.split(":", 1)[0]
                        candidates.add(f"files/{base}")
                elif isinstance(value, (dict, list, tuple)):
                    stack.append(value)
        elif isinstance(node, (list, tuple)):
            stack.extend(list(node))
    return sorted(candidates)


def _extract_operation_assets(operation: Dict[str, Any]) -> List[Dict[str, Any]]:
    containers: List[Dict[str, Any]] = []
    for key in ("result", "response", "metadata"):
        value = operation.get(key)
        if isinstance(value, dict):
            containers.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, dict):
                    containers.append(item)

    assets: List[Dict[str, Any]] = []
    seen: Set[Tuple[Optional[str], Optional[str]]] = set()

    def _register(descriptor: Dict[str, Any]) -> None:
        file_id = descriptor.get("file_id")
        url = descriptor.get("url")
        marker = (file_id, url)
        if marker in seen:
            return
        seen.add(marker)
        assets.append(descriptor)

    queue: List[Tuple[Any, Dict[str, Any]]] = []
    for container in containers:
        queue.append((container, {}))

    while queue:
        node, context = queue.pop()
        if isinstance(node, dict):
            updated_context = dict(context)
            quality = _first_str(node, _QUALITY_KEYS)
            if quality and "quality" not in updated_context:
                updated_context["quality"] = quality
            mime = _first_str(node, _MIME_KEYS)
            if mime and "mime" not in updated_context:
                updated_context["mime"] = mime
            duration = _first_number(node, _DURATION_KEYS)
            if duration is not None and "duration" not in updated_context:
                updated_context["duration"] = duration
            size = _first_number(node, _SIZE_KEYS)
            if size is not None and "size" not in updated_context:
                updated_context["size"] = size
            filename = _first_str(node, _FILENAME_KEYS)
            if filename and "filename" not in updated_context:
                updated_context["filename"] = filename

            file_id = _first_str(node, _ASSET_ID_KEYS)
            url = _first_str(node, _ASSET_URL_KEYS)
            if url and url.startswith("files/") and not file_id:
                file_id = url.split("?", 1)[0]
            if file_id and not file_id.startswith("files/"):
                file_id = f"files/{file_id.split(':', 1)[0]}"
            if file_id or url:
                descriptor: Dict[str, Any] = {"kind": "video"}
                if file_id:
                    descriptor["file_id"] = file_id
                    descriptor.setdefault("url", f"{file_id}:download")
                if url:
                    descriptor["url"] = url
                mime_value = updated_context.get("mime") or _first_str(node, _MIME_KEYS)
                if mime_value:
                    descriptor["mime"] = mime_value
                duration_value = updated_context.get("duration")
                if duration_value is None:
                    duration_value = _first_number(node, _DURATION_KEYS)
                duration_seconds = _coerce_duration_seconds(duration_value)
                if duration_seconds is not None:
                    descriptor["duration_seconds"] = duration_seconds
                size_value = updated_context.get("size")
                if size_value is None:
                    size_value = _first_number(node, _SIZE_KEYS)
                if size_value is not None and size_value > 0:
                    descriptor["bytes"] = int(size_value)
                    descriptor["size_bytes"] = int(size_value)
                filename_value = updated_context.get("filename") or _first_str(
                    node, _FILENAME_KEYS
                )
                if filename_value:
                    descriptor["filename"] = filename_value
                quality_value = _normalise_quality_label(
                    updated_context.get("quality") or _first_str(node, _QUALITY_KEYS)
                )
                if quality_value:
                    descriptor["quality"] = quality_value
                _register(descriptor)

            for child in node.values():
                if isinstance(child, (dict, list, tuple)):
                    queue.append((child, updated_context))
        elif isinstance(node, (list, tuple)):
            for item in node:
                queue.append((item, context))

    return assets


def _asset_priority(asset: Dict[str, Any]) -> float:
    quality = str(asset.get("quality") or "").lower()
    score = 0.0
    if "720" in quality:
        score += 300.0
    elif "540" in quality:
        score += 220.0
    elif "preview" in quality:
        score += 250.0
    elif "480" in quality:
        score += 180.0
    mime = str(asset.get("mime") or "").lower()
    if mime == "video/mp4":
        score += 80.0
    elif mime.startswith("video/"):
        score += 40.0
    size = asset.get("size_bytes") or asset.get("bytes")
    try:
        size_value = float(size or 0)
    except (TypeError, ValueError):
        size_value = 0.0
    score += min(size_value / 1_000_000.0, 50.0)
    if asset.get("file_id"):
        score += 10.0
    if asset.get("url") and str(asset.get("url")).startswith("https://"):
        score += 5.0
    return score


def _select_primary_asset(assets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not assets:
        return None
    ranked = sorted(assets, key=_asset_priority, reverse=True)
    return dict(ranked[0])


def _extract_inline_video_assets(
    result: Any, request_meta: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    inline_assets: List[Dict[str, Any]] = []
    assets_meta: List[Dict[str, Any]] = []
    payload: Optional[Dict[str, Any]] = None

    generated = getattr(result, "generated_videos", None)
    if not generated:
        return inline_assets, assets_meta, payload

    for item in _ensure_list(generated):
        video = getattr(item, "video", None)
        if video is None:
            continue
        mime = getattr(video, "mime_type", None) or "video/mp4"
        entry_payload: Dict[str, Any] = {
            "type": "video",
            "mime": mime,
            "model": request_meta.get("model"),
        }
        video_bytes = getattr(video, "video_bytes", None)
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
                    "type": "video",
                }
            )
            entry_payload.update(
                {
                    "data_uri": data_uri,
                    "base64": base64_data,
                    "bytes": len(video_bytes),
                }
            )
        duration = getattr(video, "duration_seconds", None) or getattr(video, "duration", None)
        if duration:
            try:
                entry_payload.setdefault("duration_seconds", int(round(float(duration))))
            except (TypeError, ValueError):  # pragma: no cover - defensive coercion
                pass
        uri = getattr(video, "uri", None)
        if isinstance(uri, str) and uri:
            entry_payload["uri"] = uri
        payload = entry_payload
        break

    return inline_assets, assets_meta, payload

class VeoVideoClient(BaseProviderClient):
    """Asynchronous wrapper around the Gemini video generation API."""

    def __init__(self, *, config: Config) -> None:
        super().__init__(
            config=config,
            base_url="https://generativelanguage.googleapis.com",
            api_key=config.gemini_api_key,
            provider_name="veo",
        )
        self._client = get_media_client(config)
        self._router = get_gemini_router(config)
        self._operations: Dict[str, _genai_types.GenerateVideosOperation] = {}
        self._requests: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._environment = (config.environment or "dev").lower()
        self._key_mask = ensure_gemini_key_logged(
            config,
            context="veo_video_client",
            process_id=f"pid={os.getpid()}",  # pragma: no cover - runtime value
            logger=log,
        )
        self._schema_logged = False
        self._asset_recovery_attempted: Set[str] = set()

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
        try:
            decision = self._router.route(
                task="video",
                model=model_name,
                prompt=prompt,
                assets=request_settings,
            )
        except GeminiRoutingError as exc:
            raise ProviderAPIError(
                provider=self.provider_name,
                status_code=400,
                message=str(exc),
                error_type="routing",
                retryable=False,
            ) from exc

        source = self._build_source(decision.prompt, request_settings)
        config_obj, config_meta = self._build_config(request_settings)

        start = time.perf_counter()
        corr_label = idempotency_key or ""
        log.info(
            "gemini.veo.submit start corr_id=%s env=%s key_mask=%s model=%s version=%s",
            corr_label or "n/a",
            self._environment,
            self._key_mask or "",
            decision.model,
            decision.api_version,
        )
        try:
            operation = await asyncio.to_thread(
                self._generate_videos_operation,
                decision.client,
                decision.model,
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
        operation_name = operation.name or str(uuid4())
        job_id = operation_name
        data = operation.model_dump(exclude_none=True)
        async with self._lock:
            self._operations[job_id] = operation
            self._requests[job_id] = {
                "model": decision.model,
                "config": config_meta,
                "key_mask": self._key_mask,
                "api_version": decision.api_version,
                "corr_id": corr_label or job_id,
                "operation_name": operation_name,
            }
            if decision.rewrite_notes:
                self._requests[job_id]["prompt_notes"] = decision.rewrite_notes
        log.info(
            "gemini.veo.submit done corr_id=%s job_id=%s env=%s key_mask=%s model=%s version=%s status=%s latency_ms=%s",
            corr_label or job_id,
            job_id,
            self._environment,
            self._key_mask or "",
            decision.model,
            decision.api_version,
            200,
            duration_ms,
        )
        return ProviderJobSubmission(
            job_id=job_id,
            status_code=200,
            duration_ms=duration_ms,
            data={"operation": data, "operation_name": operation_name},
        )

    async def get_job_status(self, job_id: str) -> ProviderJobStatus:
        operation = await self._get_operation(job_id)
        start = time.perf_counter()
        async with self._lock:
            request_info = self._requests.get(job_id, {}).copy()
        corr_id = request_info.get("corr_id") or job_id
        model_name = request_info.get("model") or ""
        api_version = request_info.get("api_version") or ""
        log.info(
            "gemini.veo.poll start corr_id=%s job_id=%s env=%s key_mask=%s model=%s version=%s",
            corr_id,
            job_id,
            self._environment,
            self._key_mask or "",
            model_name,
            api_version,
        )
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
        log.info(
            "gemini.veo.poll done corr_id=%s job_id=%s env=%s key_mask=%s model=%s version=%s status=%s status_code=%s latency_ms=%s",
            corr_id,
            job_id,
            self._environment,
            self._key_mask or "",
            model_name,
            api_version,
            getattr(refreshed, "done", False),
            200,
            duration_ms,
        )

        operation_dict = refreshed.model_dump(exclude_none=True)
        payload: Dict[str, Any] = {"operation": operation_dict}
        async with self._lock:
            self._operations[job_id] = refreshed
            request_meta = self._requests.get(job_id, {}).copy()

        if not getattr(refreshed, "done", False):
            return ProviderJobStatus(
                job_id=job_id,
                status="running",
                status_code=200,
                duration_ms=duration_ms,
                assets={},
                error=None,
                data=payload,
            )

        assets_map: Dict[str, str] = {}
        inline_assets: List[Dict[str, Any]] = []
        assets_meta: List[Dict[str, Any]] = []
        error_text: Optional[str] = None

        if refreshed.error:
            error_text = (
                refreshed.error.get("message") if isinstance(refreshed.error, dict) else str(refreshed.error)
            )
            payload["error"] = refreshed.error
            self._asset_recovery_attempted.discard(job_id)
        else:
            if not self._schema_logged:
                try:
                    serialised = json.dumps(operation_dict, ensure_ascii=False)
                except Exception:  # pragma: no cover - defensive logging
                    serialised = str(operation_dict)
                log.debug(
                    "gemini.veo.operation.completed job_id=%s payload=%s",
                    job_id,
                    serialised,
                )
                self._schema_logged = True

            result_obj = refreshed.result or refreshed.response
            inline_assets, inline_meta, inline_payload = _extract_inline_video_assets(
                result_obj, request_meta
            )
            if inline_assets:
                assets_meta.extend(inline_meta)
                payload.setdefault("inline_assets", inline_assets)
                if inline_payload:
                    payload["payload"] = inline_payload

            assets_info = _extract_operation_assets(operation_dict)
            recovery_attempted = False
            no_media_confirmed = False

            if not assets_info:
                if job_id not in self._asset_recovery_attempted:
                    self._asset_recovery_attempted.add(job_id)
                    recovery_attempted = True
                    try:
                        refreshed_name = getattr(refreshed, "name", None) or job_id
                        operation_arg = (
                            _genai_types.GenerateVideosOperation(name=refreshed_name)
                            if refreshed_name
                            else refreshed
                        )
                        secondary = await asyncio.to_thread(
                            self._client.operations.get,
                            operation_arg,
                        )
                    except Exception:  # pragma: no cover - defensive logging
                        log.warning(
                            "gemini.veo.poll recovery operations.get failed job_id=%s key_mask=%s",
                            job_id,
                            self._key_mask or "",
                            exc_info=True,
                        )
                    else:
                        secondary_dict = secondary.model_dump(exclude_none=True)
                        payload["operation_refresh"] = secondary_dict
                        assets_info = _extract_operation_assets(secondary_dict)
                        if assets_info:
                            payload["operation"] = secondary_dict
                            operation_dict = secondary_dict
                            refreshed = secondary
                else:
                    recovery_attempted = True

                if not assets_info:
                    file_ids = _collect_candidate_file_ids(operation_dict)
                    if file_ids:
                        recovery_attempted = True
                        recovered = await self._recover_files_assets(job_id, file_ids)
                        if recovered:
                            assets_info.extend(recovered)

                if not assets_info and job_id in self._asset_recovery_attempted:
                    no_media_confirmed = True

            if assets_info:
                self._asset_recovery_attempted.discard(job_id)
                primary_asset = _select_primary_asset(assets_info)
                for index, descriptor in enumerate(assets_info):
                    entry = dict(descriptor)
                    key = entry.get("key") or f"asset_{index}"
                    entry["key"] = key
                    entry.setdefault("inline", False)
                    url_value = entry.get("url")
                    file_id = entry.get("file_id")
                    if file_id and not url_value:
                        entry["url"] = f"{file_id}:download"
                        url_value = entry["url"]
                    if url_value:
                        assets_map[key] = str(url_value)
                    elif file_id:
                        assets_map[key] = str(file_id)
                    if entry.get("bytes") and not entry.get("size_bytes"):
                        entry["size_bytes"] = entry["bytes"]
                    if primary_asset and (
                        entry.get("file_id") == primary_asset.get("file_id")
                        and entry.get("url") == primary_asset.get("url")
                    ):
                        entry["preferred"] = True
                    assets_meta.append(entry)
                payload["assets"] = assets_meta
                if primary_asset:
                    preferred = dict(primary_asset)
                    preferred.setdefault("preferred", True)
                    payload["primary_asset"] = preferred
            else:
                payload["assets"] = assets_meta

            payload["assets_recovery_attempted"] = recovery_attempted
            if no_media_confirmed:
                payload["no_media_confirmed"] = True

        if inline_assets and not payload.get("inline_assets"):
            payload["inline_assets"] = inline_assets
        if assets_meta and not payload.get("assets_meta"):
            payload["assets_meta"] = assets_meta

        status = "completed" if not error_text else "failed"
        if status == "completed":
            async with self._lock:
                self._operations.pop(job_id, None)
                self._requests.pop(job_id, None)
            self._asset_recovery_attempted.discard(job_id)
        else:
            payload.setdefault("error", error_text)
            self._asset_recovery_attempted.discard(job_id)

        return ProviderJobStatus(
            job_id=job_id,
            status=status,
            status_code=200,
            duration_ms=duration_ms,
            assets=assets_map,
            error=error_text,
            data=payload,
        )

    def _generate_videos_operation(
        self,
        client: _GenAIClient,
        model: str,
        source: Optional[_genai_types.GenerateVideosSource],
        config: Optional[_genai_types.GenerateVideosConfig],
    ) -> _genai_types.GenerateVideosOperation:
        models = client.models
        prompt_text = None
        if source is not None:
            prompt_text = getattr(source, "prompt", None) or None
        if prompt_text is None:
            raise RuntimeError("Prompt is required for video generation")
        kwargs = {"model": model, "prompt": prompt_text, "config": config}
        if hasattr(models, "generate_videos"):
            return models.generate_videos(**{k: v for k, v in kwargs.items() if v is not None})
        return models._generate_videos(**{k: v for k, v in kwargs.items() if v is not None})

    async def _get_operation(self, job_id: str) -> _genai_types.GenerateVideosOperation:
        async with self._lock:
            cached = self._operations.get(job_id)
        if cached is not None:
            return cached
        return _genai_types.GenerateVideosOperation(name=job_id)

    async def _recover_files_assets(
        self, job_id: str, file_ids: Sequence[str]
    ) -> List[Dict[str, Any]]:
        recovered: List[Dict[str, Any]] = []
        for raw_id in file_ids:
            if not isinstance(raw_id, str) or not raw_id.strip():
                continue
            name = raw_id.strip()
            if not name.startswith("files/"):
                name = f"files/{name.split(':', 1)[0]}"
            try:
                file_meta = await asyncio.to_thread(self._client.files.get, name=name)
            except _genai_errors.APIError as exc:  # pragma: no cover - network guard
                log.warning(
                    "gemini.veo.files.get failed job_id=%s file=%s status=%s key_mask=%s",
                    job_id,
                    name,
                    getattr(exc, "status", None),
                    self._key_mask or "",
                )
                continue
            except Exception:  # pragma: no cover - defensive
                log.warning(
                    "gemini.veo.files.get error job_id=%s file=%s key_mask=%s",
                    job_id,
                    name,
                    self._key_mask or "",
                    exc_info=True,
                )
                continue
            meta_dict = _to_dict(file_meta)
            file_name = getattr(file_meta, "name", None) or meta_dict.get("name") or name
            mime = (
                getattr(file_meta, "mime_type", None)
                or meta_dict.get("mime_type")
                or "video/mp4"
            )
            size_value = (
                getattr(file_meta, "size_bytes", None)
                or meta_dict.get("size_bytes")
                or 0
            )
            try:
                size_int = int(size_value) if size_value else 0
            except (TypeError, ValueError):  # pragma: no cover - defensive
                size_int = 0
            display_name = (
                getattr(file_meta, "display_name", None)
                or meta_dict.get("display_name")
                or None
            )
            descriptor: Dict[str, Any] = {
                "file_id": file_name,
                "url": f"{file_name}:download",
                "mime": mime,
                "kind": "video",
                "recovered": True,
            }
            if size_int > 0:
                descriptor["bytes"] = size_int
                descriptor["size_bytes"] = size_int
            if display_name:
                descriptor["filename"] = display_name
            recovered.append(descriptor)
        return recovered

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
