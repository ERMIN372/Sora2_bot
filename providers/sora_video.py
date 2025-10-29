"""OpenAI Sora video generation client."""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, Iterable, Optional

from config import Config

from .base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
)

log = logging.getLogger(__name__)


_SIZE_PATTERN = re.compile(r"^(\d+)[xX](\d+)$")
_TERMINAL_STATUSES = {"completed", "succeeded", "failed", "errored", "cancelled", "canceled"}
_VIDEO_MIME_PREFIXES = ("video/", "application/octet-stream")


def _infer_aspect_ratio(size: Optional[str]) -> Optional[str]:
    if not size:
        return None
    match = _SIZE_PATTERN.match(size.strip())
    if not match:
        return None
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    divisor = math.gcd(width, height) or 1
    return f"{width // divisor}:{height // divisor}"


def _iter_nodes(value: Any) -> Iterable[Dict[str, Any]]:
    stack: list[Any] = [value]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            identifier = id(current)
            if identifier in seen:
                continue
            seen.add(identifier)
            yield current
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)


def _extract_url(entry: Dict[str, Any]) -> Optional[str]:
    for key in ("download_url", "url", "video_url", "asset_url"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    for nested_key in ("file", "video", "asset"):
        nested = entry.get(nested_key)
        if isinstance(nested, dict):
            for key in ("download_url", "url"):
                value = nested.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _extract_base64(entry: Dict[str, Any]) -> Optional[str]:
    for key in ("b64_json", "base64", "data", "inline_data"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            payload = value.get("data") or value.get("base64")
            if isinstance(payload, str) and payload:
                return payload
    return None


def _guess_mime(entry: Dict[str, Any]) -> Optional[str]:
    for key in ("mime_type", "content_type", "mime"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    nested = entry.get("file") or entry.get("video")
    if isinstance(nested, dict):
        for key in ("mime_type", "content_type", "mime"):
            value = nested.get(key)
            if isinstance(value, str) and value:
                return value
    return None


class SoraVideoClient(BaseProviderClient):
    """Thin wrapper around OpenAI's Responses API for Sora generation."""

    def __init__(self, *, config: Config) -> None:
        api_key = config.openai_key
        if not api_key:
            raise RuntimeError("Sora API key is not configured; set SORA_API_KEY or OPENAI_API_KEY")
        base_url = (config.openai_api_base or "https://api.openai.com/v1").rstrip("/")
        default_headers = {"OpenAI-Beta": "assistants=v2"}
        super().__init__(
            config=config,
            base_url=base_url,
            api_key=api_key,
            provider_name="sora",
            default_headers=default_headers,
        )
        self._default_model = (config.sora_model_video or "sora-2").strip() or "sora-2"
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # High level operations
    # ------------------------------------------------------------------

    async def enqueue_job(
        self,
        *,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        **_: Any,
    ) -> ProviderJobSubmission:
        request_payload = self._build_request_payload(
            prompt=prompt,
            settings=settings or {},
            payload=payload or {},
        )
        data, status_code, duration_ms = await self._request(
            "POST",
            "/responses",
            json=request_payload,
            idempotency_key=idempotency_key,
        )
        job_id = self._resolve_job_id(data)
        if not job_id:
            raise ProviderAPIError(
                provider="sora",
                status_code=status_code,
                message="Sora API did not return a response id",
                error_type="protocol",
                provider_message=str(data),
            )
        self._cache[job_id] = data
        return ProviderJobSubmission(
            job_id=job_id,
            status_code=status_code,
            duration_ms=duration_ms,
            data=data,
        )

    async def get_job_status(self, job_id: str) -> ProviderJobStatus:
        cached = self._cache.get(job_id)
        if cached and self._is_terminal(cached):
            data = cached
            status_code = 200
            duration_ms = 0
        else:
            data, status_code, duration_ms = await self._request("GET", f"/responses/{job_id}")
            self._cache[job_id] = data
        status = self._extract_status(data)
        error = self._extract_error(data)
        assets = self._extract_assets(data)
        return ProviderJobStatus(
            job_id=job_id,
            status=status,
            assets=assets,
            error=error,
            data=data,
            status_code=status_code,
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_request_payload(
        self,
        *,
        prompt: str,
        settings: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not prompt:
            raise ValueError("prompt must not be empty")
        model_name = (settings.get("model") or self._default_model or "sora-2").strip()
        content: list[Dict[str, Any]] = [
            {"type": "input_text", "text": prompt},
        ]
        reference_url = settings.get("reference_image_url") or settings.get("reference_url")
        if isinstance(reference_url, str) and reference_url.strip():
            content.append(
                {
                    "type": "input_image",
                    "image_url": {"url": reference_url.strip()},
                }
            )
        reference_file = settings.get("reference_file_id") or settings.get("image_file_id")
        if isinstance(reference_file, str) and reference_file.strip():
            content.append(
                {
                    "type": "input_image",
                    "image_file": {"file_id": reference_file.strip()},
                }
            )
        inline_data = settings.get("reference_inline_data")
        if isinstance(inline_data, dict):
            encoded = inline_data.get("data") or inline_data.get("base64")
            if isinstance(encoded, str) and encoded.strip():
                entry: Dict[str, Any] = {"type": "input_image", "image_base64": encoded.strip()}
                mime = inline_data.get("mime_type") or inline_data.get("mime")
                if isinstance(mime, str) and mime:
                    entry["mime_type"] = mime
                content.append(entry)
        if payload.get("content") and isinstance(payload["content"], list):
            for item in payload["content"]:
                if isinstance(item, dict):
                    content.append(item)
        request: Dict[str, Any] = {
            "model": model_name,
            "modalities": ["video"],
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
        response_format = settings.get("response_format") or payload.get("response_format")
        if response_format:
            request["response_format"] = response_format
        metadata: Dict[str, Any] = {}
        aspect_ratio = settings.get("aspect_ratio") or _infer_aspect_ratio(str(settings.get("size") or ""))
        if aspect_ratio:
            metadata["aspect_ratio"] = aspect_ratio
        duration = (
            settings.get("duration")
            or settings.get("duration_sec")
            or settings.get("duration_seconds")
        )
        if isinstance(duration, (int, float)) and duration > 0:
            metadata["duration_sec"] = int(duration)
        elif isinstance(duration, str) and duration.isdigit():
            metadata["duration_sec"] = int(duration)
        if payload.get("metadata") and isinstance(payload["metadata"], dict):
            metadata.update({k: v for k, v in payload["metadata"].items() if v is not None})
        if metadata:
            request["metadata"] = metadata
        tools = payload.get("tools")
        if isinstance(tools, list):
            request["tools"] = tools
        tool_choice = payload.get("tool_choice")
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        return request

    def _resolve_job_id(self, payload: Dict[str, Any]) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        for key in ("id", "response_id"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        response = payload.get("response")
        if isinstance(response, dict):
            value = response.get("id")
            if isinstance(value, str) and value:
                return value
        return None

    def _extract_status(self, payload: Dict[str, Any]) -> str:
        raw: Optional[str] = None
        if isinstance(payload, dict):
            candidate = payload.get("status")
            if isinstance(candidate, str) and candidate:
                raw = candidate
            elif isinstance(payload.get("response"), dict):
                response_status = payload["response"].get("status")
                if isinstance(response_status, str) and response_status:
                    raw = response_status
        if not raw:
            return "unknown"
        normalised = raw.strip().lower()
        if normalised in {"succeeded", "success"}:
            return "completed"
        if normalised in {"queued", "pending", "in_progress", "running"}:
            return "running"
        if normalised in {"failed", "errored", "error", "cancelled", "canceled"}:
            return "failed"
        if normalised == "completed":
            return "completed"
        return normalised or "unknown"

    def _extract_error(self, payload: Dict[str, Any]) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        error = payload.get("error")
        if isinstance(error, dict):
            detail = error.get("message") or error.get("detail") or error.get("code")
            if isinstance(detail, str) and detail:
                return detail
            return str(error)
        if isinstance(error, str) and error:
            return error
        last_error = payload.get("last_error")
        if isinstance(last_error, dict):
            message = last_error.get("message") or last_error.get("code")
            if isinstance(message, str) and message:
                return message
        return None

    def _extract_assets(self, payload: Dict[str, Any]) -> Dict[str, str]:
        assets: Dict[str, str] = {}
        if not isinstance(payload, dict):
            return assets
        for entry in _iter_nodes(payload):
            if not isinstance(entry, dict):
                continue
            entry_type = str(entry.get("type") or entry.get("kind") or "").lower()
            mime = _guess_mime(entry) or ""
            is_video_type = entry_type in {"output_video", "video", "video_asset"}
            is_video_mime = any(mime.startswith(prefix) for prefix in _VIDEO_MIME_PREFIXES)
            if not is_video_type and not is_video_mime:
                continue
            key = entry.get("file_id") or entry.get("id")
            if isinstance(key, str) and not key.strip():
                key = None
            url = _extract_url(entry)
            if not url:
                base64_payload = _extract_base64(entry)
                if base64_payload:
                    mime_type = mime or "video/mp4"
                    url = f"data:{mime_type};base64,{base64_payload}"
            if not isinstance(url, str) or not url:
                continue
            asset_key = str(key or f"asset_{len(assets)}")
            if asset_key not in assets:
                assets[asset_key] = url
        return assets

    def _is_terminal(self, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        status = self._extract_status(payload)
        return status in _TERMINAL_STATUSES
