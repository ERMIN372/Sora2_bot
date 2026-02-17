"""Background job queue for video generation."""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import mimetypes
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

from config import Config
from db import DatabaseInterface, ErrorLogRecord, GenerationJobRecord
from moderation import classify_safety_response, policy_message, render_error_json
from monitoring.metrics import job_failed, job_finished, job_started, record_job_enqueued
from observability import increment_metric, log_event
from services.gemini_key import current_key_mask, ensure_gemini_key_logged
from services.error_reporter import ErrorReporter
from generation_gate import GenerationRequestGate, compute_generation_idempotency_key
from providers import BaseProviderClient, ProviderAPIError, ProviderJobStatus, ProviderJobSubmission

log = logging.getLogger(__name__)


_REASON_MESSAGES: Dict[str, str] = {
    "api_access_disabled": "Доступ к API отключён/ограничен. Исправим и вернёмся.",
    "service_busy": "Сервис занят. Квота на исходе. Попробуйте позже.",
    "invalid_request": "Запрос оформлен неподдерживаемо для выбранной модели.",
    "provider_error": "Сервис ответил с ошибкой. Мы уже всё перезапустили.",
    "provider_timeout": "Провайдер не закончил вовремя. Мы вернули кредиты.",
}

_PROGRESS_KEYS: Tuple[str, ...] = (
    "progress_percent",
    "progressPercent",
    "progress_pct",
    "progressPct",
    "progress",
)

_UPDATE_KEYS: Tuple[str, ...] = (
    "update_time",
    "updateTime",
    "latest_update_time",
    "latestUpdateTime",
    "update_time_seconds",
    "last_update_time",
    "lastUpdateTime",
)


_RETRYABLE_IDEMPOTENCY_STATUSES: Tuple[str, ...] = (
    "failed",
    "timeout",
    "no_media",
    "refunded",
)


VIDEO_TASK_CREATE = "video_create"
VIDEO_TASK_REMIX = "video_remix"
VIDEO_TASK_RETRIEVE = "video_retrieve"
VIDEO_TASK_DOWNLOAD = "video_download"
ASSET_TASK_RETRIEVE = "asset_retrieve"
IMAGE_TASK_RETRIEVE = "image_retrieve"
IMAGE_TASK_GENERATE = "image_generate"
ASSET_KIND_VIDEO = "video"
ASSET_KIND_IMAGE = "image"

_DOWNLOAD_MAX_ATTEMPTS = 3
_POLL_MAX_ATTEMPTS = 5
_POLL_MAX_ATTEMPTS_KLING = 24
_POLL_BACKOFF_BASE = 1.5

_IMAGE_ARTIFACTS_DIR = Path("attached_assets") / "images"


@dataclass(slots=True)
class _FailureReason:
    reason: str
    user_message: str
    provider_error_code: Optional[str]
    provider_error_message: Optional[str]


@dataclass(slots=True)
class _OperationTracker:
    job_id: str
    corr_id: str
    provider: str
    started_at: float
    deadline_at: float
    last_progress_at: float
    last_progress: Optional[float] = None
    last_update_token: Optional[str] = None

    def touch(
        self,
        *,
        progress: Optional[float],
        update_token: Optional[str],
        now: float,
    ) -> None:
        changed = False
        if progress is not None and progress != self.last_progress:
            self.last_progress = progress
            changed = True
        if update_token:
            token = str(update_token)
            if token and token != self.last_update_token:
                self.last_update_token = token
                changed = True
        if changed:
            self.last_progress_at = now

    def deadline_exceeded(self, now: float) -> bool:
        return now >= self.deadline_at

    def idle_exceeded(self, now: float, timeout: float) -> bool:
        if timeout <= 0:
            return False
        return (now - self.last_progress_at) >= timeout

    def snapshot(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "deadline_at": self.deadline_at,
            "last_progress_at": self.last_progress_at,
            "last_progress": self.last_progress,
            "last_update_token": self.last_update_token,
        }


def _poll_max_attempts_for_provider(*, provider: Optional[str], default_attempts: int) -> int:
    provider_key = (provider or "").strip().lower()
    if provider_key.startswith("kling"):
        return max(default_attempts, _POLL_MAX_ATTEMPTS_KLING)
    return default_attempts


@dataclass(slots=True)
class _FallbackSubmissionResult:
    submission: ProviderJobSubmission
    client: BaseProviderClient
    provider_key: str
    model: str
    idempotency_key: str
    settings: Dict[str, Any]


def _artifact_directory(job_id: str) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", job_id or "job") or "job"
    return _IMAGE_ARTIFACTS_DIR / safe_id


def _decode_data_uri(data_uri: Optional[str]) -> Optional[tuple[str, bytes]]:
    if not data_uri:
        return None
    text = data_uri.strip()
    if not text.lower().startswith("data:"):
        return None
    try:
        header, encoded = text.split(",", 1)
    except ValueError:
        return None
    if ";base64" not in header.lower():
        return None
    mime = header[5:].split(";", 1)[0] or "application/octet-stream"
    normalised = "".join(encoded.split())
    try:
        payload = base64.b64decode(normalised)
    except (binascii.Error, ValueError):
        return None
    return mime, payload


def _store_image_artifacts(job_id: str, assets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not assets:
        return []
    directory = _artifact_directory(job_id)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception:
        log.warning("Failed to create artifact directory path=%s", directory, exc_info=True)
        return []
    stored: List[Dict[str, Any]] = []
    for index, asset in enumerate(assets):
        data_uri = asset.get("data_uri") or asset.get("data_url")
        decoded = _decode_data_uri(data_uri if isinstance(data_uri, str) else None)
        if not decoded:
            continue
        mime, payload = decoded
        extension = mimetypes.guess_extension(mime) or ".bin"
        filename = f"image_{index}{extension}"
        path = directory / filename
        try:
            with path.open("wb") as handle:
                handle.write(payload)
        except Exception:
            log.warning("Failed to persist image artifact job_id=%s path=%s", job_id, path, exc_info=True)
            continue
        stored.append({
            "path": str(path),
            "mime": mime,
            "bytes": len(payload),
        })
    return stored


def hydrate_image_artifacts(job_id: str) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    if not job_id:
        return artifacts
    directory = _artifact_directory(job_id)
    if not directory.exists():
        return artifacts
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        try:
            payload = path.read_bytes()
        except Exception:
            log.warning("Failed to read image artifact job_id=%s path=%s", job_id, path, exc_info=True)
            continue
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        encoded = base64.b64encode(payload).decode("ascii")
        data_uri = f"data:{mime};base64,{encoded}"
        artifacts.append({
            "kind": "image",
            "mime": mime,
            "bytes": len(payload),
            "data_uri": data_uri,
            "path": str(path),
        })
    return artifacts


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_progress_markers(data: Dict[str, Any]) -> Tuple[Optional[float], Optional[str]]:
    if not isinstance(data, dict):
        return None, None
    progress: Optional[float] = None
    update_token: Optional[str] = None
    stack: list[Any] = [data]
    while stack and (progress is None or update_token is None):
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if progress is None and key in _PROGRESS_KEYS:
                    progress = _coerce_float(value)
                if update_token is None and key in _UPDATE_KEYS:
                    if isinstance(value, (str, int, float)):
                        text = str(value).strip()
                        if text:
                            update_token = text
                if progress is not None and update_token is not None:
                    break
            if progress is not None and update_token is not None:
                break
            for value in node.values():
                if isinstance(value, (dict, list, tuple)):
                    stack.append(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                if isinstance(item, (dict, list, tuple)):
                    stack.append(item)
    return progress, update_token


def _should_retry_idempotency(status: Optional[str]) -> bool:
    if not status:
        return False
    status_lower = status.lower()
    if status_lower in _RETRYABLE_IDEMPOTENCY_STATUSES:
        return True
    if "failed" in status_lower:
        return True
    return False


def _extract_video_id(data: Any, *, client: Optional[BaseProviderClient] = None) -> Optional[str]:
    def _pick(mapping: Mapping[str, Any]) -> Optional[str]:
        for key in ("video_id", "videoId", "id", "job_id"):
            value = mapping.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    if isinstance(data, Mapping):
        direct = _pick(data)
        if direct:
            return direct
        response = data.get("response")
        if isinstance(response, Mapping):
            candidate = _pick(response)
            if candidate:
                return candidate
        video_payload = data.get("video") or data.get("asset")
        if isinstance(video_payload, Mapping):
            candidate = _pick(video_payload)
            if candidate:
                return candidate
        inner_data = data.get("data")
        if isinstance(inner_data, list):
            for item in inner_data:
                if isinstance(item, Mapping):
                    candidate = _pick(item)
                    if candidate:
                        return candidate
    elif isinstance(data, list):
        for item in data:
            candidate = _extract_video_id(item, client=client)
            if candidate:
                return candidate

    if client is not None:
        getter = getattr(client, "get_video_object", None)
        if callable(getter):
            try:
                video_obj = getter(data)
            except Exception:  # pragma: no cover - defensive
                video_obj = None
            if isinstance(video_obj, Mapping):
                candidate = _pick(video_obj)
                if candidate:
                    return candidate
    return None


def _find_error_payload(data: Any) -> Optional[Dict[str, Any]]:
    if data is None:
        return None
    stack: list[Any] = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            candidate = node.get("error")
            if isinstance(candidate, dict):
                return candidate
            for value in node.values():
                if isinstance(value, (dict, list, tuple)):
                    stack.append(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                if isinstance(item, (dict, list, tuple)):
                    stack.append(item)
    return None


def _normalise_error_code(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return str(int(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return str(value)
    text = str(value).strip()
    if not text:
        return None
    return text.upper()


def _map_error_reason(
    status_code: Optional[int],
    codes: Tuple[str, ...],
    *,
    provider_timeout: bool = False,
) -> str:
    if provider_timeout:
        return "provider_timeout"
    upper_codes = {code.upper() for code in codes if code}
    numeric_codes = set()
    for code in upper_codes:
        if code.isdigit():
            try:
                numeric_codes.add(int(code))
            except ValueError:  # pragma: no cover - defensive
                continue
    http_status = status_code
    if http_status is None and numeric_codes:
        http_status = next(iter(numeric_codes))
    if upper_codes.intersection({"PERMISSION_DENIED", "SERVICE_DISABLED"}) or http_status == 403:
        return "api_access_disabled"
    if any("RESOURCE_EXHAUSTED" in code or "QUOTA" in code for code in upper_codes) or http_status == 429:
        return "service_busy"
    if upper_codes.intersection({"INVALID_ARGUMENT", "FAILED_PRECONDITION"}) or http_status in {400, 422}:
        return "invalid_request"
    if upper_codes.intersection({"INTERNAL", "UNAVAILABLE"}) or (
        http_status is not None and http_status >= 500
    ):
        return "provider_error"
    return "provider_error"


def _build_failure_reason(
    *,
    status_code: Optional[int],
    error_payload: Optional[Dict[str, Any]],
    fallback_message: str,
    provider_timeout: bool = False,
) -> _FailureReason:
    codes: list[str] = []
    provider_error_code: Optional[str] = None
    provider_error_message: Optional[str] = None
    if isinstance(error_payload, dict):
        for key in ("status", "code", "reason", "error_code"):
            normalised = _normalise_error_code(error_payload.get(key))
            if normalised:
                codes.append(normalised)
                if provider_error_code is None:
                    provider_error_code = normalised
        raw_message = error_payload.get("message")
        if isinstance(raw_message, (dict, list)):
            try:
                provider_error_message = json.dumps(raw_message, ensure_ascii=False)
            except Exception:  # pragma: no cover - defensive serialisation
                provider_error_message = str(raw_message)
        elif raw_message:
            provider_error_message = str(raw_message)
    if provider_error_code is None and status_code is not None:
        provider_error_code = str(status_code)
    if provider_timeout and not provider_error_code:
        provider_error_code = "provider_timeout"
    if provider_error_message is None:
        provider_error_message = fallback_message or ""
    if provider_timeout and not provider_error_message:
        provider_error_message = "operation timed out"
    reason_code = _map_error_reason(status_code, tuple(codes), provider_timeout=provider_timeout)
    fallback_text = (fallback_message or "").strip()
    user_message = fallback_text or _REASON_MESSAGES.get(reason_code) or _REASON_MESSAGES[
        "provider_error"
    ]
    return _FailureReason(
        reason=reason_code,
        user_message=user_message,
        provider_error_code=provider_error_code,
        provider_error_message=provider_error_message or None,
    )


def _prompt_preview(prompt: str, limit: int = 120) -> str:
    snippet = (prompt or "").strip().replace("\n", " ")
    if len(snippet) > limit:
        return f"{snippet[:limit]}…(+{len(snippet) - limit} chars)"
    return snippet


def _extract_duration_seconds(extra: Mapping[str, Any]) -> Optional[int]:
    for key in ("duration_seconds", "seconds", "duration"):
        value = extra.get(key)
        if value in (None, ""):
            continue
        text_value: Optional[str]
        if isinstance(value, str):
            text_value = value.strip()
            if not text_value:
                continue
        else:
            text_value = None
        try:
            numeric = float(text_value or value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            return int(numeric)
    return None


@dataclass
class PendingJob:
    job_id: str
    user_id: int
    prompt: str
    corr_id: str
    size: str
    model: str
    provider: str
    username: Optional[str]
    original_prompt: Optional[str] = None
    sanitized_prompt: Optional[str] = None
    auto_sanitized: bool = False
    preflight_reason: Optional[str] = None
    preflight_scope: Optional[str] = None
    task_type: str = ASSET_TASK_RETRIEVE
    asset_kind: str = ASSET_KIND_VIDEO
    provider_job_id: Optional[str] = None
    video_id: Optional[str] = None
    asset_id: Optional[str] = None
    attempt: int = 0
    max_attempts: int = 0


class JobQueue:
    """Manage generation jobs with concurrency limits."""

    def __init__(
        self,
        *,
        db: DatabaseInterface,
        providers: Dict[str, BaseProviderClient],
        default_provider: str,
        config: Config,
        gate: Optional[GenerationRequestGate] = None,
        error_reporter: Optional[ErrorReporter] = None,
    ) -> None:
        self._db = db
        self._providers = providers
        self._default_provider = default_provider
        self._config = config
        self._gate = gate
        self._queue: asyncio.Queue[PendingJob] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._stopped = asyncio.Event()
        self._notification_callbacks: Dict[str, Callable[[GenerationJobRecord], Awaitable[None]]] = {}
        self._environment = (config.environment or "dev").lower()
        self._debug_enabled = bool(getattr(config, "debug_gemini", False))
        if self._debug_enabled and log.getEffectiveLevel() > logging.DEBUG:
            log.setLevel(logging.DEBUG)
        self._error_reporter = error_reporter
        self._db_log_jobs = bool(getattr(config, "db_log_jobs", True))
        self._gemini_key_mask = ensure_gemini_key_logged(
            config,
            context="job_queue",
            process_id=f"pid={os.getpid()}",  # pragma: no cover - runtime value
            logger=log,
        )
        self._operation_trackers: Dict[str, _OperationTracker] = {}

    def register_notification_callback(
        self, *, name: str, callback: Callable[[GenerationJobRecord], Awaitable[None]]
    ) -> None:
        self._notification_callbacks[name] = callback

    def _resolve_provider(self, provider_key: Optional[str]) -> tuple[BaseProviderClient, str]:
        if not self._providers:
            raise RuntimeError(
                "No generation providers are configured; ensure at least one provider is enabled"
            )

        key = provider_key or self._default_provider
        if key and (client := self._providers.get(key)) is not None:
            return client, key

        if self._default_provider and self._default_provider in self._providers:
            log.warning(
                "Unknown provider %s, falling back to %s",
                provider_key,
                self._default_provider,
            )
            return self._providers[self._default_provider], self._default_provider

        raise RuntimeError(
            "Requested generation provider is not configured and no default provider is available"
        )

    def _debug_event(self, name: str, **details: Any) -> None:
        try:
            payload = {
                key: value
                for key, value in details.items()
                if value not in (None, "", [], {}, ())
            }
            message = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except Exception:
            message = str(details)
        logger = log.info if self._debug_enabled else log.debug
        logger("%s %s", name, message)

    async def _safe_log_error_record(self, record: ErrorLogRecord, *, context: str) -> None:
        try:
            await self._db.log_error_record(record)
        except Exception:
            log.warning("Failed to persist error log context=%s", context, exc_info=True)

    async def _persist_job_record(self, record: GenerationJobRecord, *, stage: str) -> None:
        if not self._db_log_jobs:
            return
        try:
            await self._db.create_job(record)
        except Exception:
            log.exception("Failed to persist job record job_id=%s stage=%s", record.id, stage)
            error_record = ErrorLogRecord(
                ts=datetime.now(timezone.utc),
                user_id=record.user_id,
                username=record.username,
                corr_id=record.corr_id,
                job_id=record.id,
                model=record.model,
                size=record.size,
                status_code=None,
                error_type="db_log_failed",
                error_msg_short=f"job_create_failed:{stage}",
                refunded=False,
                preflight_blocked=False,
                preflight_reason="",
                auto_sanitized=False,
                sanitized_prompt=record.prompt,
                error_scope="db",
                error_json="",
                job_status=record.status,
                reason="db_log_failed",
                provider_error_code=None,
                provider_error_message=None,
                stage=stage,
            )
            await self._safe_log_error_record(error_record, context="persist_job_record")

    def get_provider(self, key: Optional[str]) -> Optional[BaseProviderClient]:
        if not key:
            return None
        return self._providers.get(key)

    async def _attempt_gemini_fallback(
        self,
        *,
        error: ProviderAPIError,
        provider_key: str,
        prompt: str,
        size: str,
        settings: Dict[str, Any],
        preset: Optional[str],
        payload: Optional[Dict[str, Any]],
        content_type: str,
        user_id: int,
        corr_id: str,
    ) -> Optional[_FallbackSubmissionResult]:
        if error.error_type not in {"safety", "provider_unavailable"}:
            return None
        if not provider_key.startswith("gemini"):
            return None
        if (content_type or "").strip().lower() not in {"image", "photo"}:
            return None
        fallback_models = [
            model_name.strip()
            for model_name in self._config.fallback_image_models
            if isinstance(model_name, str) and model_name.strip()
        ]
        requested_model = str(settings.get("model") or "").strip().lower()
        primary_model = (self._config.gemini_model_image or "").strip().lower()
        if requested_model and primary_model and requested_model == primary_model:
            log.info(
                "Gemini fallback skipped corr_id=%s reason=user_selected_model",
                corr_id,
            )
            return None
        if not fallback_models:
            log.info(
                "Gemini fallback skipped corr_id=%s reason=no_fallback_models",
                corr_id,
            )
            return None
        log.info(
            "Gemini fallback initiated corr_id=%s provider=%s error_type=%s error_status=%s",
            corr_id,
            provider_key,
            error.error_type,
            error.status_code,
        )
        for fallback_model in fallback_models:
            candidate_keys = [fallback_model, "openai-image", "openai", "sora"]
            fallback_client: Optional[BaseProviderClient] = None
            fallback_provider_key = ""
            for candidate in candidate_keys:
                if not candidate:
                    continue
                client = self.get_provider(candidate)
                if client is not None:
                    fallback_client = client
                    fallback_provider_key = candidate
                    break
            if fallback_client is None:
                log.debug(
                    "Fallback provider not found corr_id=%s model=%s", corr_id, fallback_model
                )
                continue
            fallback_settings = dict(settings or {})
            fallback_settings["model"] = fallback_model
            fallback_idempotency_key = compute_generation_idempotency_key(
                user_id=user_id,
                prompt=prompt,
                size=size,
                model=fallback_model,
                content_type=content_type,
            )
            try:
                submission = await fallback_client.enqueue_job(
                    prompt=prompt,
                    settings=fallback_settings or None,
                    preset=preset,
                    payload=payload,
                    idempotency_key=fallback_idempotency_key,
                )
            except ProviderAPIError as fallback_error:
                log.warning(
                    "Fallback provider error corr_id=%s provider=%s model=%s status=%s type=%s",
                    corr_id,
                    fallback_provider_key,
                    fallback_model,
                    fallback_error.status_code,
                    fallback_error.error_type,
                )
                continue
            except Exception:
                log.exception(
                    "Fallback provider unexpected error corr_id=%s provider=%s model=%s",
                    corr_id,
                    fallback_provider_key,
                    fallback_model,
                )
                continue
            log.info(
                "Gemini fallback succeeded corr_id=%s fallback_provider=%s model=%s",
                corr_id,
                fallback_provider_key,
                fallback_model,
            )
            return _FallbackSubmissionResult(
                submission=submission,
                client=fallback_client,
                provider_key=fallback_provider_key,
                model=fallback_model,
                idempotency_key=fallback_idempotency_key,
                settings=fallback_settings,
            )
        return None

    def _poll_interval_seconds(self) -> float:
        try:
            low_value = float(self._config.veo_poll_interval_min_seconds)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            low_value = 2.0
        try:
            high_value = float(self._config.veo_poll_interval_max_seconds)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            high_value = 5.0
        low = max(low_value, 0.5)
        high = max(high_value, low)
        return random.uniform(low, high)

    def _ensure_operation_tracker(
        self,
        *,
        pending: PendingJob,
        provider_key: str,
        now: float,
    ) -> _OperationTracker:
        tracker = self._operation_trackers.get(pending.job_id)
        try:
            timeout_value = float(self._config.veo_operation_timeout_seconds)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            timeout_value = 12 * 60.0
        if timeout_value <= 0:
            timeout_value = 12 * 60.0
        if tracker is None:
            tracker = _OperationTracker(
                job_id=pending.job_id,
                corr_id=pending.corr_id,
                provider=provider_key,
                started_at=now,
                deadline_at=now + timeout_value,
                last_progress_at=now,
            )
            self._operation_trackers[pending.job_id] = tracker
        else:
            tracker.corr_id = pending.corr_id
            tracker.provider = provider_key
        return tracker

    def _reset_operation_tracker(self, job_id: str) -> None:
        self._operation_trackers.pop(job_id, None)

    async def _report_error(
        self,
        record: ErrorLogRecord,
        *,
        context: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._error_reporter:
            return
        try:
            await self._error_reporter.report(record, context=context, extra=extra)
        except Exception:  # pragma: no cover - defensive
            log.exception("Failed to publish error notification context=%s", context)

    async def _handle_operation_timeout(
        self,
        *,
        pending: PendingJob,
        provider_key: str,
        key_mask: str,
        result: ProviderJobStatus,
        provider_data: Dict[str, Any],
        tracker: _OperationTracker,
        job_extra: Dict[str, Any],
        timeout_kind: str,
    ) -> None:
        self._reset_operation_tracker(pending.job_id)
        raw_error = result.error or ""
        error_payload = _find_error_payload(provider_data)
        failure_reason = _build_failure_reason(
            status_code=result.status_code,
            error_payload=error_payload,
            fallback_message=raw_error,
            provider_timeout=True,
        )
        reason_code = failure_reason.reason
        reason_message = failure_reason.user_message
        provider_error_code = failure_reason.provider_error_code
        provider_error_message = failure_reason.provider_error_message
        job_extra.setdefault("reason", reason_code)
        job_extra.setdefault("reason_message", reason_message)
        if provider_error_code:
            job_extra.setdefault("provider_error_code", provider_error_code)
        if provider_error_message:
            job_extra.setdefault("provider_error_message", provider_error_message)
        job_extra.setdefault("timeout_kind", timeout_kind)
        job_extra.setdefault("operation_tracker", tracker.snapshot())
        if isinstance(error_payload, dict):
            job_extra.setdefault("provider_error", error_payload)
        if provider_data and "provider_data" not in job_extra:
            job_extra["provider_data"] = provider_data
        try:
            await self._db.update_job(pending.job_id, "timeout", error=reason_message)
        except Exception:
            log.exception("Failed to update job %s as timeout", pending.job_id)
        refunded = False
        try:
            await self._db.add_credits(pending.user_id, self._config.generation_cost_credits)
            refunded = True
            increment_metric("refunds_total")
            increment_metric("refund_total")
        except Exception:
            log.exception("Refunding credits failed for timeout job %s", pending.job_id)
        error_record = ErrorLogRecord(
            ts=datetime.now(timezone.utc),
            user_id=pending.user_id,
            username=pending.username,
            corr_id=pending.corr_id,
            job_id=pending.job_id,
            model=pending.model,
            size=pending.size,
            status_code=result.status_code,
            error_type="job_timeout",
            error_msg_short=reason_message[:240],
            refunded=refunded,
            preflight_blocked=False,
            preflight_reason=pending.preflight_reason or "",
            auto_sanitized=pending.auto_sanitized,
            sanitized_prompt=pending.sanitized_prompt or pending.prompt,
            error_scope="unknown",
            error_json="",
            job_status="timeout",
            reason=reason_code,
            provider_error_code=provider_error_code,
            provider_error_message=provider_error_message,
            stage="poll",
        )
        sheet_ok = await self._db.log_error_record(error_record)
        await self._report_error(
            error_record,
            context="job_timeout",
            extra={
                "gsheets_ok": sheet_ok,
                "provider": provider_key,
                "timeout_kind": timeout_kind,
                "reason_message": reason_message,
                "operation_tracker": tracker.snapshot(),
            },
        )
        log_event(
            level="ERROR",
            event="error",
            corr_id=pending.corr_id,
            job_id=pending.job_id,
            user_id=pending.user_id,
            username=pending.username,
            model=pending.model,
            provider=provider_key,
            size=pending.size,
            status_code=result.status_code,
            duration_ms=result.duration_ms,
            error_type="job_timeout",
            error_msg_short=reason_message,
            gsheets_ok=sheet_ok,
            refund_done=refunded,
            extra={
                "provider_message": raw_error,
                "reason": reason_code,
                "reason_message": reason_message,
                "provider_error_code": provider_error_code,
                "provider_error_message": provider_error_message,
                "timeout_kind": timeout_kind,
                "operation_tracker": tracker.snapshot(),
                "gemini_key_mask": key_mask or None,
            },
        )
        log_event(
            level="INFO",
            event="refund",
            corr_id=pending.corr_id,
            job_id=pending.job_id,
            user_id=pending.user_id,
            username=pending.username,
            model=pending.model,
            provider=provider_key,
            size=pending.size,
            credits_cost=self._config.generation_cost_credits,
            refund_done=refunded,
            extra={"gemini_key_mask": key_mask or None, "reason": reason_code},
        )
        increment_metric("veo_running_timeout_total")
        increment_metric("veo_failed_total", tags={"reason": reason_code})
        await self._release_gate(pending, reason="provider_timeout", status="timeout")


    async def start(self) -> None:
        if self._workers:
            return
        self._stopped.clear()
        if not self._providers:
            log.warning("No generation providers configured; job queue workers will not start")
            return
        for _ in range(self._config.jobs_concurrency):
            task = asyncio.create_task(self._worker())
            self._workers.append(task)
            ensure_gemini_key_logged(
                self._config,
                context="job_worker",
                process_id=f"pid={os.getpid()} worker={len(self._workers)}",  # pragma: no cover - runtime value
                logger=log,
            )
        await self._recover_pending_jobs()

    async def stop(self) -> None:
        self._stopped.set()
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        await asyncio.gather(
            *[client.close() for client in self._providers.values()],
            return_exceptions=True,
        )

    async def submit(
        self,
        *,
        user_id: int,
        prompt: str,
        size: str,
        model: Optional[str],
        corr_id: str,
        image_file_id: Optional[str] = None,
        username: Optional[str] = None,
        provider: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        preset: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        original_prompt: Optional[str] = None,
        sanitized_prompt: Optional[str] = None,
        auto_sanitized: bool = False,
        preflight_reason: Optional[str] = None,
        preflight_scope: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        content_type: str = "video",
        credits_cost: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> GenerationJobRecord:
        model_key = model or self._config.default_video_model
        client, provider_key = self._resolve_provider(provider or model_key)
        corr_id = corr_id or str(uuid.uuid4())
        request_settings: Dict[str, Any] = dict(settings or {})
        if size:
            request_settings.setdefault("size", size)
        request_settings.setdefault("model", model_key)
        if image_file_id:
            request_settings.setdefault("image_file_id", image_file_id)
        effective_prompt = sanitized_prompt or prompt
        content_type_value = content_type or "video"
        stable_idempotency_key = idempotency_key or compute_generation_idempotency_key(
            user_id=user_id,
            prompt=effective_prompt,
            size=size,
            model=model_key,
            content_type=content_type_value,
        )
        existing_job = await self._db.find_job_by_idempotency_key(stable_idempotency_key)
        if existing_job is not None:
            if _should_retry_idempotency(existing_job.status):
                log.info(
                    "Retrying submission after previous failure corr_id=%s user_id=%s existing_job_id=%s status=%s",
                    corr_id,
                    user_id,
                    existing_job.id,
                    existing_job.status,
                )
            else:
                log.info(
                    "Duplicate submission suppressed corr_id=%s user_id=%s existing_job_id=%s status=%s",
                    corr_id,
                    user_id,
                    existing_job.id,
                    existing_job.status,
                )
                return existing_job
        prompt_hash = hashlib.sha256(effective_prompt.encode("utf-8")).hexdigest()
        log.debug(
            "Submitting provider job corr_id=%s user_id=%s provider=%s model=%s settings=%s prompt_hash=%s",
            corr_id,
            user_id,
            provider_key,
            model_key,
            request_settings,
            prompt_hash,
        )
        try:
            submission: ProviderJobSubmission = await client.enqueue_job(
                prompt=effective_prompt,
                settings=request_settings or None,
                preset=preset,
                payload=payload,
                idempotency_key=stable_idempotency_key,
            )
        except ProviderAPIError as exc:
            fallback = await self._attempt_gemini_fallback(
                error=exc,
                provider_key=provider_key,
                prompt=effective_prompt,
                size=size,
                settings=request_settings,
                preset=preset,
                payload=payload,
                content_type=content_type_value,
                user_id=user_id,
                corr_id=corr_id,
            )
            if fallback is None:
                raise
            client = fallback.client
            provider_key = fallback.provider_key
            submission = fallback.submission
            model_key = fallback.model
            request_settings = fallback.settings
            stable_idempotency_key = fallback.idempotency_key
        submission_payload = submission.data if isinstance(submission.data, dict) else {}
        extras_payload = dict(extra or {})
        inline_assets_payload = []
        if isinstance(submission_payload.get("inline_assets"), list):
            inline_assets_payload = submission_payload.get("inline_assets")  # type: ignore[assignment]
        now = datetime.now(timezone.utc)
        effective_cost = credits_cost or self._config.generation_cost_credits
        is_flash_inline = (
            provider_key == "gemini-image"
            and (model_key or "").strip().lower() == "gemini-2.5-flash-image"
            and not submission.job_id
            and bool(inline_assets_payload)
        )
        if is_flash_inline:
            inline_job_id = str(uuid.uuid4())
            record = GenerationJobRecord(
                id=inline_job_id,
                user_id=user_id,
                prompt=effective_prompt,
                status="completed",
                video_url=None,
                video_id=None,
                error=None,
                created_at=now,
                updated_at=now,
                image_file_id=image_file_id,
                size=size,
                model=model_key,
                cost_credits=effective_cost,
                username=username,
                corr_id=corr_id,
                content_type="image",
                idempotency_key=stable_idempotency_key,
                operation_name=None,
            )
            if isinstance(record.extra, dict):
                record.extra.update(
                    {
                        "inline_result": submission_payload,
                        "gemini_key_mask": self._gemini_key_mask or None,
                        "provider": provider_key,
                        "duration_ms": submission.duration_ms,
                        "status_code": submission.status_code,
                    }
                )
                if extras_payload:
                    record.extra.update(extras_payload)
            await self._persist_job_record(record, stage="submit_inline")
            log_event(
                level="INFO",
                event="request",
                corr_id=corr_id,
                job_id=inline_job_id,
                user_id=user_id,
                username=username,
                model=model_key,
                provider=provider_key,
                size=size,
                credits_cost=effective_cost,
                duration_ms=submission.duration_ms,
                status_code=submission.status_code,
                prompt=None,
                gsheets_ok=True,
                extra={
                    "gemini_key_mask": self._gemini_key_mask or None,
                    "preflight_blocked": False,
                    "preflight_reason": preflight_reason,
                    "preflight_scope": preflight_scope,
                    "preflight_auto_sanitized": auto_sanitized,
                    "video_id": None,
                    "inline_result": True,
                },
            )
            return record

        job_id = submission.job_id
        video_id = _extract_video_id(submission.data, client=client)
        operation_name = submission_payload.get("operation_name") if isinstance(submission.data, dict) else None
        record = GenerationJobRecord(
            id=job_id,
            user_id=user_id,
            prompt=effective_prompt,
            status="queued",
            video_url=None,
            video_id=video_id,
            error=None,
            created_at=now,
            updated_at=now,
            image_file_id=image_file_id,
            size=size,
            model=model_key,
            cost_credits=effective_cost,
            username=username,
            corr_id=corr_id,
            content_type=content_type_value,
            idempotency_key=stable_idempotency_key,
            operation_name=operation_name,
        )
        if extras_payload:
            record.extra.update(extras_payload)
        await self._persist_job_record(record, stage="submit")
        asset_kind = ASSET_KIND_IMAGE if content_type_value == "image" else ASSET_KIND_VIDEO
        task_type = IMAGE_TASK_GENERATE if asset_kind == ASSET_KIND_IMAGE else ASSET_TASK_RETRIEVE
        await self.enqueue(
            job_id=job_id,
            user_id=user_id,
            prompt=effective_prompt,
            corr_id=corr_id,
            size=size,
            model=model_key,
            provider=provider_key,
            username=username,
            original_prompt=original_prompt or prompt,
            sanitized_prompt=effective_prompt,
            auto_sanitized=auto_sanitized,
            preflight_reason=preflight_reason,
            preflight_scope=preflight_scope,
            task_type=task_type,
            asset_kind=asset_kind,
            provider_job_id=submission.job_id,
            video_id=video_id,
        )
        log.debug(
            "Job queued corr_id=%s job_id=%s provider=%s user_id=%s queue_size=%s",
            corr_id,
            job_id,
            provider_key,
            user_id,
            self._queue.qsize(),
        )
        gemini_mask = self._gemini_key_mask if provider_key.startswith("veo") or provider_key.startswith("gemini") else ""
        log_event(
            level="INFO",
            event="request",
            corr_id=corr_id,
            job_id=job_id,
            user_id=user_id,
            username=username,
            model=model_key,
            provider=provider_key,
            size=size,
            credits_cost=effective_cost,
            duration_ms=submission.duration_ms,
            status_code=submission.status_code,
            prompt=None,
            gsheets_ok=True,
            extra={
                "gemini_key_mask": gemini_mask,
                "preflight_blocked": False,
                "preflight_reason": preflight_reason,
                "preflight_scope": preflight_scope,
                "preflight_auto_sanitized": auto_sanitized,
                "video_id": video_id,
            },
        )
        return record

    async def register_external_job(
        self,
        *,
        job_id: str,
        user_id: int,
        prompt: str,
        size: str,
        model: str,
        corr_id: str,
        provider: Optional[str],
        username: Optional[str],
        video_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        content_type: str = "video",
        task_type: str = ASSET_TASK_RETRIEVE,
        sanitized_prompt: Optional[str] = None,
        original_prompt: Optional[str] = None,
        auto_sanitized: bool = False,
        preflight_reason: Optional[str] = None,
        preflight_scope: Optional[str] = None,
        operation_name: Optional[str] = None,
        status_code: Optional[int] = None,
        duration_ms: Optional[int] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> GenerationJobRecord:
        provider_key = provider or model
        sanitized = sanitized_prompt or prompt
        original = original_prompt or prompt
        extras_payload = dict(extra or {})
        existing = await self._db.get_job(job_id)
        asset_kind = (
            ASSET_KIND_IMAGE if (content_type or "video").strip().lower() == "image" else ASSET_KIND_VIDEO
        )
        if existing is not None:
            if extras_payload:
                existing.extra.update(extras_payload)
                seconds_override = _extract_duration_seconds(extras_payload)
                if seconds_override is not None:
                    existing.seconds = seconds_override
            if video_id or operation_name:
                try:
                    await self._db.update_job(
                        job_id,
                        existing.status or "queued",
                        video_id=video_id or existing.video_id,
                        operation_name=operation_name or existing.operation_name,
                    )
                except Exception:  # pragma: no cover - defensive
                    log.exception("Failed to update existing job metadata job_id=%s", job_id)
            await self.enqueue(
                job_id=existing.id,
                user_id=existing.user_id,
                prompt=existing.prompt or sanitized,
                corr_id=existing.corr_id or corr_id,
                size=existing.size or size,
                model=existing.model or model,
                provider=provider_key,
                username=existing.username,
                original_prompt=existing.prompt or original,
                sanitized_prompt=existing.prompt or sanitized,
                auto_sanitized=auto_sanitized,
                preflight_reason=preflight_reason,
                preflight_scope=preflight_scope,
                task_type=task_type,
                asset_kind=asset_kind,
                provider_job_id=existing.id,
                video_id=video_id or existing.video_id,
            )
            return existing

        now = datetime.now(timezone.utc)
        record = GenerationJobRecord(
            id=job_id,
            user_id=user_id,
            prompt=sanitized,
            status="queued",
            video_url=None,
            video_id=video_id,
            error=None,
            created_at=now,
            updated_at=now,
            image_file_id=None,
            size=size,
            model=model,
            cost_credits=self._config.generation_cost_credits,
            username=username,
            corr_id=corr_id,
            content_type=content_type,
            idempotency_key=idempotency_key,
            operation_name=operation_name,
        )
        if extras_payload:
            record.extra.update(extras_payload)
            seconds_override = _extract_duration_seconds(extras_payload)
            if seconds_override is not None:
                record.seconds = seconds_override
        await self._persist_job_record(record, stage="external_submit")
        await self.enqueue(
            job_id=job_id,
            user_id=user_id,
            prompt=sanitized,
            corr_id=corr_id,
            size=size,
            model=model,
            provider=provider_key,
            username=username,
            original_prompt=original,
            sanitized_prompt=sanitized,
            auto_sanitized=auto_sanitized,
            preflight_reason=preflight_reason,
            preflight_scope=preflight_scope,
            task_type=task_type,
            asset_kind=asset_kind,
            provider_job_id=job_id,
            video_id=video_id,
        )
        log_event(
            level="INFO",
            event="request",
            corr_id=corr_id,
            job_id=job_id,
            user_id=user_id,
            username=username,
            model=model,
            provider=provider_key,
            size=size,
            credits_cost=self._config.generation_cost_credits,
            duration_ms=duration_ms,
            status_code=status_code,
            prompt=None,
            gsheets_ok=True,
            extra={
                "preflight_blocked": False,
                "preflight_reason": preflight_reason,
                "preflight_scope": preflight_scope,
                "preflight_auto_sanitized": auto_sanitized,
                "video_id": video_id,
                **({} if not extras_payload else {
                    key: value for key, value in extras_payload.items() if key not in {
                        "preflight_blocked",
                        "preflight_reason",
                        "preflight_scope",
                        "preflight_auto_sanitized",
                        "video_id",
                    }
                }),
            },
        )
        return record

    async def enqueue(
        self,
        *,
        job_id: str,
        user_id: int,
        prompt: str,
        corr_id: str,
        size: str,
        model: str,
        provider: Optional[str],
        username: Optional[str],
        original_prompt: Optional[str] = None,
        sanitized_prompt: Optional[str] = None,
        auto_sanitized: bool = False,
        preflight_reason: Optional[str] = None,
        preflight_scope: Optional[str] = None,
        task_type: str = ASSET_TASK_RETRIEVE,
        asset_kind: str = ASSET_KIND_VIDEO,
        provider_job_id: Optional[str] = None,
        video_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        attempt: int = 0,
        max_attempts: int = 0,
    ) -> None:
        provider_key = provider or model
        log.debug(
            "Enqueueing pending job job_id=%s provider=%s corr_id=%s user_id=%s",
            job_id,
            provider_key,
            corr_id,
            user_id,
        )
        poll_tasks = {ASSET_TASK_RETRIEVE, IMAGE_TASK_RETRIEVE, VIDEO_TASK_RETRIEVE}
        max_attempts_value = max_attempts
        if max_attempts_value <= 0 and task_type in poll_tasks:
            max_attempts_value = _POLL_MAX_ATTEMPTS
        await self._queue.put(
            PendingJob(
                job_id=job_id,
                user_id=user_id,
                prompt=prompt,
                corr_id=corr_id,
                size=size,
                model=model,
                provider=provider_key,
                username=username,
                original_prompt=original_prompt or prompt,
                sanitized_prompt=sanitized_prompt or prompt,
                auto_sanitized=auto_sanitized,
                preflight_reason=preflight_reason,
                preflight_scope=preflight_scope,
                task_type=task_type,
                asset_kind=asset_kind,
                provider_job_id=provider_job_id,
                video_id=video_id,
                asset_id=asset_id,
                attempt=attempt,
                max_attempts=max_attempts_value,
            )
        )
        record_job_enqueued(
            task_type=task_type,
            provider=provider_key,
            queue_depth=self._queue.qsize(),
            concurrency=self._config.jobs_concurrency,
        )

    async def _recover_pending_jobs(self) -> None:
        pending = await self._db.list_pending_jobs(limit=100)
        stale_cutoff = datetime.now(timezone.utc) - timedelta(hours=12)
        stale_skipped = 0
        epoch = datetime.fromtimestamp(0, tz=timezone.utc)
        for job in pending:
            asset_kind = (
                ASSET_KIND_IMAGE
                if (job.content_type or "").strip().lower() == "image"
                else ASSET_KIND_VIDEO
            )
            model_lower = (job.model or "").strip().lower()
            has_timestamp = bool(job.created_at and job.created_at > epoch)
            if (
                model_lower.startswith("veo")
                and has_timestamp
                and job.created_at < stale_cutoff
            ):
                message = "Veo operation not accessible; marked failed on startup"
                log.warning(
                    "Marking stale Veo pending job as failed job_id=%s created_at=%s",
                    job.id,
                    job.created_at,
                )
                await self._db.update_job(
                    job.id,
                    "failed",
                    video_id=job.video_id,
                    error=message,
                )
                stale_skipped += 1
                continue
            await self.enqueue(
                job_id=job.id,
                user_id=job.user_id,
                prompt=job.prompt,
                corr_id=job.corr_id or job.id,
                size=job.size or "",
                model=job.model or self._config.default_video_model,
                provider=job.model or self._default_provider,
                username=job.username,
                original_prompt=job.prompt,
                sanitized_prompt=job.prompt,
                asset_kind=asset_kind,
                provider_job_id=job.id,
                video_id=job.video_id,
            )
        if pending:
            log.info("Recovered %s pending jobs", len(pending))
        if stale_skipped:
            log.info("Skipped %s stale Veo pending jobs", stale_skipped)

    async def _notify(self, job: GenerationJobRecord) -> None:
        for callback in self._notification_callbacks.values():
            try:
                await callback(job)
            except Exception:  # pragma: no cover - defensive
                log.exception("Notification callback %s failed", callback)

    async def _worker(self) -> None:
        while not self._stopped.is_set():
            try:
                pending = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                await self._process_job(pending)
            except Exception:  # pragma: no cover - background guard
                log.exception("Unexpected error while processing job %s", pending.job_id)
            finally:
                self._queue.task_done()

    async def _handle_asset_retrieve(self, pending: PendingJob) -> None:
        provider_hint = pending.provider or ""
        provider_job_id = pending.provider_job_id or pending.job_id
        task_type = pending.task_type or ASSET_TASK_RETRIEVE
        asset_kind = (pending.asset_kind or ASSET_KIND_VIDEO).strip().lower()
        attempt = max(0, pending.attempt or 0)
        max_attempts = pending.max_attempts or _POLL_MAX_ATTEMPTS
        max_attempts = _poll_max_attempts_for_provider(
            provider=pending.provider,
            default_attempts=max_attempts,
        )
        if asset_kind not in {ASSET_KIND_VIDEO, ASSET_KIND_IMAGE}:
            asset_kind = ASSET_KIND_VIDEO
        key_mask = self._gemini_key_mask if provider_hint.startswith(("veo", "gemini")) else ""
        self._debug_event(
            "jobs.poll.start",
            job_id=pending.job_id,
            corr_id=pending.corr_id,
            provider=provider_hint,
            provider_job=provider_job_id,
            task=task_type,
            asset_kind=asset_kind,
        )
        log.info(
            "Processing job %s corr_id=%s provider=%s key_mask=%s env=%s task=%s asset_kind=%s provider_job=%s",
            pending.job_id,
            pending.corr_id,
            provider_hint,
            key_mask or "",
            self._environment,
            task_type,
            asset_kind,
            provider_job_id,
        )
        await self._db.update_job(pending.job_id, "running", video_id=pending.video_id)
        result: Optional[ProviderJobStatus] = None
        inline_assets: List[Dict[str, Any]] = []
        assets_meta: List[Dict[str, Any]] = []
        job_extra: Dict[str, Any] = {}
        if pending.video_id:
            job_extra["video_id"] = pending.video_id
        job_extra["asset_kind"] = asset_kind
        asset_label: Optional[str] = None
        tracker: Optional[_OperationTracker] = None
        try:
            client, provider_key = self._resolve_provider(pending.provider)
            provider_name = getattr(client, "provider_name", provider_key or "")
            is_veo_provider = provider_name.startswith("veo")
            if provider_key.startswith(("veo", "gemini")) or provider_name.startswith(
                ("veo", "gemini")
            ):
                key_mask = self._gemini_key_mask
            else:
                key_mask = ""
            poll_target = pending.video_id or provider_job_id

            async def _schedule_retry(provider_key: str) -> None:
                if attempt + 1 >= max_attempts:
                    final_message = "poll retries exhausted"
                    log.warning(
                        "Polling attempts exhausted job_id=%s corr_id=%s provider=%s attempts=%s",
                        pending.job_id,
                        pending.corr_id,
                        provider_key,
                        max_attempts,
                    )
                    try:
                        await self._db.update_job(
                            pending.job_id,
                            "failed",
                            video_id=pending.video_id,
                            error=final_message,
                        )
                    except Exception:
                        log.exception("Failed to mark job %s as failed", pending.job_id)
                    await self._release_gate(
                        pending,
                        reason="poll_retry_exhausted",
                        status="failed",
                    )
                    return

                delay = self._poll_interval_seconds() * (_POLL_BACKOFF_BASE ** attempt)
                delay = min(delay, 60.0)
                await asyncio.sleep(delay)
                await self.enqueue(
                    job_id=pending.job_id,
                    user_id=pending.user_id,
                    prompt=pending.prompt,
                    corr_id=pending.corr_id,
                    size=pending.size,
                    model=pending.model,
                    provider=provider_key,
                    username=pending.username,
                    original_prompt=pending.original_prompt,
                    sanitized_prompt=pending.sanitized_prompt,
                    auto_sanitized=pending.auto_sanitized,
                    preflight_reason=pending.preflight_reason,
                    preflight_scope=pending.preflight_scope,
                    task_type=pending.task_type or ASSET_TASK_RETRIEVE,
                    asset_kind=asset_kind,
                    provider_job_id=provider_job_id,
                    video_id=pending.video_id,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                )
            try:
                log.debug(
                    "Polling provider job job_id=%s provider_job=%s provider=%s corr_id=%s",
                    pending.job_id,
                    poll_target,
                    provider_key,
                    pending.corr_id,
                )
                result = await client.get_job_status(poll_target)
                self._debug_event(
                    "jobs.poll.result",
                    job_id=pending.job_id,
                    corr_id=pending.corr_id,
                    provider=provider_key,
                    status=result.status,
                    status_code=result.status_code,
                    inline_assets=len(
                        result.data.get("inline_assets", []) if isinstance(result.data, dict) else []
                    ),
                    assets=len(result.assets),
                    error=result.error,
                )
                provider_job_id = poll_target
            except ProviderAPIError as exc:
                error_code_upper = (exc.error_code or exc.error_type or "").upper()
                non_retryable = {403}
                is_permission_denied = exc.status_code in non_retryable or error_code_upper == "PERMISSION_DENIED"
                is_veo_poll = is_veo_provider and task_type == ASSET_TASK_RETRIEVE
                if is_permission_denied and is_veo_poll:
                    terminal_reason = "provider_forbidden"
                    terminal_message = "operation not accessible; stop retry"
                    log.warning(
                        "poll_terminal job_id=%s corr_id=%s provider=%s operation_id=%s status_code=%s message=%s",
                        pending.job_id,
                        pending.corr_id,
                        provider_key,
                        poll_target,
                        exc.status_code,
                        terminal_message,
                    )
                    self._debug_event(
                        "jobs.poll.error",
                        job_id=pending.job_id,
                        corr_id=pending.corr_id,
                        provider=provider_key,
                        status_code=exc.status_code,
                        error_type=exc.error_type,
                        error_code=exc.error_code,
                    )
                    log_event(
                        level="WARNING",
                        event="poll_terminal",
                        corr_id=pending.corr_id,
                        job_id=pending.job_id,
                        user_id=pending.user_id,
                        username=pending.username,
                        model=pending.model,
                        provider=provider_key,
                        size=pending.size,
                        status_code=exc.status_code,
                        error_type=exc.error_type,
                        error_code=exc.error_code,
                        error_msg_short=terminal_message,
                        duration_ms=exc.duration_ms,
                        extra={
                            "operation_id": poll_target,
                            "provider_message": exc.provider_message,
                            "gemini_key_mask": key_mask or None,
                            "asset_kind": asset_kind,
                        },
                    )
                    try:
                        await self._db.update_job(
                            pending.job_id,
                            "failed",
                            video_id=pending.video_id,
                            error=terminal_message,
                        )
                    except Exception:
                        log.exception("Failed to mark job %s as failed", pending.job_id)
                    refunded = False
                    try:
                        await self._db.add_credits(
                            pending.user_id, self._config.generation_cost_credits
                        )
                        refunded = True
                        increment_metric("refunds_total")
                        increment_metric("refund_total")
                    except Exception:
                        log.exception("Refunding credits failed for job %s", pending.job_id)
                    job_extra.setdefault("reason", terminal_reason)
                    job_extra["reason_message"] = terminal_message
                    job_extra["provider_message"] = exc.provider_message
                    increment_metric("veo_failed_total", tags={"reason": terminal_reason})
                    await self._release_gate(
                        pending,
                        reason=terminal_reason,
                        status="failed",
                    )
                    return
                # 4xx client errors (400, 401, 422) are never retryable – the
                # request is structurally wrong and will never succeed.
                is_client_error = 400 <= exc.status_code < 500 and exc.status_code not in {
                    408,  # Request Timeout – may succeed on retry
                    429,  # Too Many Requests – may succeed after backoff
                }
                if is_client_error:
                    terminal_reason = "invalid_request"
                    terminal_message = f"Provider rejected request ({exc.status_code}); not retryable"
                    log.warning(
                        "poll_terminal_client_error job_id=%s corr_id=%s provider=%s status_code=%s message=%s provider_message=%s",
                        pending.job_id,
                        pending.corr_id,
                        provider_key,
                        exc.status_code,
                        terminal_message,
                        (exc.provider_message or "")[:300],
                    )
                    log_event(
                        level="WARNING",
                        event="poll_terminal",
                        corr_id=pending.corr_id,
                        job_id=pending.job_id,
                        user_id=pending.user_id,
                        username=pending.username,
                        model=pending.model,
                        provider=provider_key,
                        size=pending.size,
                        status_code=exc.status_code,
                        error_type=exc.error_type,
                        error_code=exc.error_code,
                        error_msg_short=terminal_message,
                        duration_ms=exc.duration_ms,
                        extra={
                            "provider_message": exc.provider_message,
                            "asset_kind": asset_kind,
                        },
                    )
                    try:
                        await self._db.update_job(
                            pending.job_id,
                            "failed",
                            video_id=pending.video_id,
                            error=terminal_message,
                        )
                    except Exception:
                        log.exception("Failed to mark job %s as failed", pending.job_id)
                    try:
                        await self._db.add_credits(
                            pending.user_id, self._config.generation_cost_credits
                        )
                        increment_metric("refunds_total")
                        increment_metric("refund_total")
                    except Exception:
                        log.exception("Refunding credits failed for job %s", pending.job_id)
                    job_extra.setdefault("reason", terminal_reason)
                    job_extra["reason_message"] = terminal_message
                    job_extra["provider_message"] = exc.provider_message
                    await self._release_gate(
                        pending,
                        reason=terminal_reason,
                        status="failed",
                    )
                    return
                log.warning(
                    "Provider polling failed job_id=%s corr_id=%s provider=%s status=%s error_type=%s error_code=%s message=%s",
                    pending.job_id,
                    pending.corr_id,
                    provider_key,
                    exc.status_code,
                    exc.error_type,
                    exc.error_code,
                    exc,
                )
                error_extra = {
                    "provider_message": exc.provider_message,
                    "gemini_key_mask": key_mask or None,
                    "asset_kind": asset_kind,
                }
                self._debug_event(
                    "jobs.poll.error",
                    job_id=pending.job_id,
                    corr_id=pending.corr_id,
                    provider=provider_key,
                    status_code=exc.status_code,
                    error_type=exc.error_type,
                    error_code=exc.error_code,
                )
                log_event(
                    level="ERROR",
                    event="poll",
                    corr_id=pending.corr_id,
                    job_id=pending.job_id,
                    user_id=pending.user_id,
                    username=pending.username,
                    model=pending.model,
                    provider=provider_key,
                    size=pending.size,
                    status_code=exc.status_code,
                    error_type=exc.error_type,
                    error_code=exc.error_code,
                    error_msg_short=str(exc),
                    duration_ms=exc.duration_ms,
                    extra=error_extra,
                )
                await _schedule_retry(provider_key)
                return
            except Exception as exc:  # pragma: no cover - defensive network guard
                log.exception("Unexpected error while polling job %s", pending.job_id)
                self._debug_event(
                    "jobs.poll.error",
                    job_id=pending.job_id,
                    corr_id=pending.corr_id,
                    provider=provider_key,
                    error_type="exception",
                    error_message=str(exc),
                )
                log_event(
                    level="ERROR",
                    event="poll",
                    corr_id=pending.corr_id,
                    job_id=pending.job_id,
                    user_id=pending.user_id,
                    username=pending.username,
                    model=pending.model,
                    provider=provider_key,
                    size=pending.size,
                    error_type="unknown",
                    error_msg_short=str(exc),
                    extra={"gemini_key_mask": key_mask or None},
                )
                await _schedule_retry(provider_key)
                return
            provider_data: Dict[str, Any] = {}
            if isinstance(result.data, dict):
                provider_data = dict(result.data)

            raw_inline = provider_data.get("inline_assets") if provider_data else []
            inline_assets = (
                [dict(asset) for asset in raw_inline if isinstance(asset, dict)]
                if isinstance(raw_inline, list)
                else []
            )

            assets_meta: List[Dict[str, Any]] = []
            raw_assets_meta = provider_data.get("assets") if provider_data else []
            if isinstance(raw_assets_meta, list):
                for item in raw_assets_meta:
                    if isinstance(item, dict):
                        entry = dict(item)
                        entry["key"] = str(entry.get("key") or f"asset_{len(assets_meta)}")
                        assets_meta.append(entry)
            if not assets_meta:
                for key, value in result.assets.items():
                    assets_meta.append({"key": key, "inline": False, "url": value})

            provider_payload: Optional[Dict[str, Any]] = None
            raw_payload = provider_data.get("payload") if provider_data else None
            if isinstance(raw_payload, dict):
                provider_payload = raw_payload

            extracted_video_id = pending.video_id
            if not extracted_video_id:
                extracted_video_id = _extract_video_id(provider_data, client=client)
            if not extracted_video_id and result is not None:
                extracted_video_id = _extract_video_id(result.data, client=client)
            if extracted_video_id:
                if extracted_video_id != pending.video_id:
                    pending.video_id = extracted_video_id
                job_extra.setdefault("video_id", extracted_video_id)
                job_extra["payload"] = provider_payload
            if provider_payload:
                data_uri = provider_payload.get("data_uri")
                if isinstance(data_uri, str):
                    already_inline = any(
                        isinstance(asset.get("data_uri"), str)
                        and asset.get("data_uri") == data_uri
                        for asset in inline_assets
                    )
                    if not already_inline:
                        mime = provider_payload.get("mime") or "application/octet-stream"
                        inline_entry = {
                            "kind": provider_payload.get("type") or "video",
                            "mime": mime,
                            "bytes": provider_payload.get("bytes"),
                            "data_uri": data_uri,
                            "filename": provider_payload.get("filename") or "result.bin",
                        }
                        inline_assets.append(inline_entry)
                        key = f"inline_{len(inline_assets) - 1}"
                        assets_meta.append(
                            {
                                "key": key,
                                "inline": True,
                                "mime": mime,
                                "bytes": provider_payload.get("bytes"),
                            }
                        )
                uri = provider_payload.get("uri")
                if isinstance(uri, str):
                    matched = False
                    for entry in assets_meta:
                        if entry.get("url") == uri:
                            matched = True
                            if provider_payload.get("mime"):
                                entry.setdefault("mime", provider_payload.get("mime"))
                            if provider_payload.get("bytes"):
                                entry.setdefault("bytes", provider_payload.get("bytes"))
                            entry.setdefault("type", provider_payload.get("type"))
                            break
                    if not matched:
                        assets_meta.append(
                            {
                                "key": "payload",
                                "inline": False,
                                "url": uri,
                                "mime": provider_payload.get("mime"),
                                "bytes": provider_payload.get("bytes"),
                                "type": provider_payload.get("type"),
                            }
                        )
            if inline_assets:
                job_extra["inline_assets"] = inline_assets
                if asset_kind == ASSET_KIND_IMAGE:
                    stored_meta = _store_image_artifacts(pending.job_id, inline_assets)
                    if stored_meta:
                        job_extra.setdefault("image_artifacts", stored_meta)
            if assets_meta:
                job_extra["assets_meta"] = assets_meta
            if key_mask:
                job_extra.setdefault("gemini_key_mask", key_mask)
            if provider_data:
                job_extra.setdefault("provider_data", provider_data)

            if asset_kind == ASSET_KIND_IMAGE:
                payload_source = provider_payload if isinstance(provider_payload, dict) else {}
                if not payload_source and isinstance(provider_data.get("payload"), dict):
                    payload_source = provider_data.get("payload")  # type: ignore[assignment]
                candidates_payload = []
                if isinstance(payload_source, dict):
                    raw_candidates = payload_source.get("candidates")
                    if isinstance(raw_candidates, list):
                        candidates_payload = [candidate for candidate in raw_candidates if isinstance(candidate, dict)]
                finish_reason = ""
                if candidates_payload:
                    primary = candidates_payload[0]
                    finish_reason = str(
                        primary.get("finish_reason")
                        or primary.get("finishReason")
                        or primary.get("finish_reason_code")
                        or ""
                    )
                total_bytes = 0
                for asset in inline_assets:
                    try:
                        total_bytes += int(asset.get("bytes") or 0)
                    except Exception:
                        continue
                log_event(
                    level="INFO",
                    event="image_result",
                    corr_id=pending.corr_id,
                    job_id=pending.job_id,
                    user_id=pending.user_id,
                    username=pending.username,
                    model=pending.model,
                    provider=provider_key,
                    size=pending.size,
                    duration_ms=result.duration_ms,
                    extra={
                        "task": IMAGE_TASK_GENERATE,
                        "candidates": len(candidates_payload),
                        "image_parts_count": len(inline_assets),
                        "bytes": total_bytes,
                        "finish_reason": finish_reason or "",
                    },
                )

            recovery_attempted = bool(provider_data.get("assets_recovery_attempted"))
            no_media_confirmed = bool(provider_data.get("no_media_confirmed"))

            if recovery_attempted:
                job_extra.setdefault("assets_recovery_attempted", True)
            if no_media_confirmed:
                job_extra.setdefault("no_media_confirmed", True)

            primary_asset = (
                provider_data.get("primary_asset")
                if isinstance(provider_data.get("primary_asset"), dict)
                else None
            )
            asset_value = None
            if isinstance(primary_asset, dict):
                asset_value = primary_asset.get("url") or primary_asset.get("file_id")
            if asset_value is None and not inline_assets:
                asset_value = next(iter(result.assets.values()), None)

            if inline_assets:
                first_inline = inline_assets[0]
                mime = first_inline.get("mime") or "application/octet-stream"
                size_bytes = int(first_inline.get("bytes") or 0)
                asset_label_display = f"[inline {mime}, {size_bytes} bytes]"
            else:
                descriptor = None
                if isinstance(primary_asset, dict):
                    descriptor = primary_asset.get("mime") or primary_asset.get("quality")
                if descriptor:
                    asset_label_display = f"[remote {descriptor}]"
                elif asset_value:
                    asset_label_display = "[remote asset]"
                else:
                    asset_label_display = None

            if provider_data and "provider_data" not in job_extra:
                job_extra["provider_data"] = provider_data
            progress_value: Optional[float] = None
            progress_token: Optional[str] = None
            if is_veo_provider:
                now_monotonic = time.monotonic()
                tracker = self._ensure_operation_tracker(
                    pending=pending, provider_key=provider_key, now=now_monotonic
                )
                progress_value, progress_token = _extract_progress_markers(provider_data)
                if progress_value is not None:
                    job_extra["progress_percent"] = progress_value
                if progress_token:
                    job_extra["progress_update_time"] = progress_token
                tracker.touch(
                    progress=progress_value,
                    update_token=progress_token,
                    now=now_monotonic,
                )
                try:
                    idle_timeout_value = float(self._config.veo_operation_idle_timeout_seconds)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    idle_timeout_value = 120.0
                if idle_timeout_value < 0:
                    idle_timeout_value = 0.0
                if tracker.deadline_exceeded(now_monotonic):
                    await self._handle_operation_timeout(
                        pending=pending,
                        provider_key=provider_key,
                        key_mask=key_mask,
                        result=result,
                        provider_data=provider_data,
                        tracker=tracker,
                        job_extra=job_extra,
                        timeout_kind="deadline",
                    )
                    return
                if idle_timeout_value and tracker.idle_exceeded(now_monotonic, idle_timeout_value):
                    await self._handle_operation_timeout(
                        pending=pending,
                        provider_key=provider_key,
                        key_mask=key_mask,
                        result=result,
                        provider_data=provider_data,
                        tracker=tracker,
                        job_extra=job_extra,
                        timeout_kind="idle",
                    )
                    return
            poll_extra: Dict[str, Any] = {
                "status": result.status,
                "gemini_key_mask": key_mask or None,
                "asset_kind": asset_kind,
            }
            if pending.video_id:
                poll_extra["video_id"] = pending.video_id
            if progress_value is not None:
                poll_extra["progress_percent"] = progress_value
                job_extra["progress_percent"] = progress_value
            if progress_token:
                poll_extra["progress_update_time"] = progress_token
                job_extra["progress_update_time"] = progress_token
            if assets_meta:
                poll_extra["assets_meta"] = assets_meta
            if recovery_attempted:
                poll_extra["assets_recovery_attempted"] = True
            if no_media_confirmed:
                poll_extra["no_media_confirmed"] = True
            log_event(
                level="INFO",
                event="poll",
                corr_id=pending.corr_id,
                job_id=pending.job_id,
                user_id=pending.user_id,
                username=pending.username,
                model=pending.model,
                provider=provider_key,
                size=pending.size,
                status_code=result.status_code,
                duration_ms=result.duration_ms,
                extra=poll_extra,
            )
            if result.status == "completed":
                self._reset_operation_tracker(pending.job_id)
                
                video_url_value = asset_value or asset_label_display or ""
                display_label = asset_label_display or video_url_value or ""
                log.debug(
                    "Job completed job_id=%s corr_id=%s provider=%s assets=%s has_url=%s",
                    pending.job_id,
                    pending.corr_id,
                    provider_key,
                    list(result.assets.keys()),
                    bool(inline_assets or asset_value),
                )
                if primary_asset:
                    job_extra.setdefault("primary_asset", primary_asset)
                done_extra: Dict[str, Any] = {
                    "assets": list(result.assets.keys()),
                    "asset_kind": asset_kind,
                }
                if assets_meta:
                    done_extra["assets_meta"] = assets_meta
                if primary_asset:
                    done_extra["primary_asset"] = primary_asset
                if recovery_attempted:
                    done_extra["assets_recovery_attempted"] = True
                if key_mask:
                    done_extra["gemini_key_mask"] = key_mask
                if pending.video_id:
                    done_extra["video_id"] = pending.video_id
                done_extra["video_label"] = display_label
                
                download_handler = getattr(client, "download_content", None)
                if callable(download_handler) and pending.video_id:
                    handled = await self._handle_inline_download(
                        pending,
                        client=client,
                        provider_key=provider_key,
                        result=result,
                        video_url_value=video_url_value,
                        display_label=display_label,
                        done_extra=done_extra,
                        job_extra=job_extra,
                    )
                    if handled:
                        await self._release_gate(pending, reason="success", status="completed")
                    else:
                        await self._release_gate(
                            pending,
                            reason="download_failed",
                            status="failed_to_download",
                        )
                    return
                gsheets_ok = True
                try:
                    await self._db.update_job(
                        pending.job_id,
                        "completed",
                        video_url=video_url_value,
                        file_url=video_url_value,
                        video_id=pending.video_id,
                        error=None,
                    )
                except Exception:
                    log.exception("Failed to update job %s as completed", pending.job_id)
                    gsheets_ok = False
                log_event(
                    level="INFO",
                    event="done",
                    corr_id=pending.corr_id,
                    job_id=pending.job_id,
                    user_id=pending.user_id,
                    username=pending.username,
                    model=pending.model,
                    provider=provider_key,
                    size=pending.size,
                    status_code=result.status_code,
                    duration_ms=result.duration_ms,
                    gsheets_ok=gsheets_ok,
                    extra=done_extra,
                )
                if asset_kind == ASSET_KIND_VIDEO:
                    await self._schedule_video_download(
                        pending,
                        provider_key=provider_key,
                        client=client,
                        primary_asset=primary_asset,
                        assets_meta=assets_meta,
                        result=result,
                    )
                await self._release_gate(pending, reason="success", status="completed")
            elif result.status in {"failed", "errored", "stopped"}:
                self._reset_operation_tracker(pending.job_id)
                log.warning(
                    "Job failed job_id=%s corr_id=%s provider=%s status=%s error=%s",
                    pending.job_id,
                    pending.corr_id,
                    provider_key,
                    result.status,
                    result.error,
                )
                raw_error_text = (result.error or "").strip()
                if not raw_error_text:
                    raw_error_text = "Unknown error"
                error_payload = _find_error_payload(provider_data)
                failure_reason = _build_failure_reason(
                    status_code=result.status_code,
                    error_payload=error_payload,
                    fallback_message=raw_error_text,
                )
                reason_code = failure_reason.reason
                provider_error_code = failure_reason.provider_error_code
                provider_error_message = failure_reason.provider_error_message
                response_payload: Optional[Dict[str, Any]] = None
                if isinstance(provider_data, dict):
                    response_payload = provider_data.get("response")
                scope, categories, summary = classify_safety_response(response_payload)
                if scope == "unknown" and pending.preflight_scope:
                    scope = pending.preflight_scope
                error_json = render_error_json(summary)
                policy_text = policy_message(scope)
                error_message = failure_reason.user_message or raw_error_text
                if scope != "unknown":
                    error_message = policy_text
                elif raw_error_text.strip().upper() in {"STOP", "SAFETY"}:
                    error_message = policy_message("unknown")
                if scope and scope != "unknown":
                    job_extra["error_scope"] = scope
                if summary:
                    job_extra["error_summary"] = summary
                if error_json:
                    job_extra["error_json"] = error_json
                if pending.preflight_reason:
                    job_extra.setdefault("preflight_reason", pending.preflight_reason)
                if pending.auto_sanitized:
                    job_extra.setdefault("preflight_auto_sanitized", pending.auto_sanitized)
                if pending.sanitized_prompt:
                    job_extra.setdefault("sanitized_prompt", pending.sanitized_prompt)
                if categories:
                    job_extra.setdefault("safety_categories", categories)
                job_extra.setdefault("reason", reason_code)
                job_extra["reason_message"] = error_message
                if provider_error_code:
                    job_extra.setdefault("provider_error_code", provider_error_code)
                if provider_error_message:
                    job_extra.setdefault("provider_error_message", provider_error_message)
                if isinstance(error_payload, dict):
                    job_extra.setdefault("provider_error", error_payload)
                gsheets_ok = True
                try:
                    await self._db.update_job(
                        pending.job_id,
                        "failed",
                        file_url="",
                        video_id=pending.video_id,
                        error=error_message,
                    )
                except Exception:
                    log.exception("Failed to mark job %s as failed", pending.job_id)
                    gsheets_ok = False
                refunded = False
                try:
                    await self._db.add_credits(
                        pending.user_id, self._config.generation_cost_credits
                    )
                    refunded = True
                    increment_metric("refunds_total")
                    increment_metric("refund_total")
                except Exception:
                    log.exception("Refunding credits failed for job %s", pending.job_id)
                error_record = ErrorLogRecord(
                    ts=datetime.now(timezone.utc),
                    user_id=pending.user_id,
                    username=pending.username,
                    corr_id=pending.corr_id,
                    job_id=pending.job_id,
                    model=pending.model,
                    size=pending.size,
                    status_code=result.status_code,
                    error_type="job_failed",
                    error_msg_short=error_message[:240],
                    refunded=refunded,
                    preflight_blocked=False,
                    preflight_reason=pending.preflight_reason or "",
                    auto_sanitized=pending.auto_sanitized,
                    sanitized_prompt=pending.sanitized_prompt or pending.prompt,
                    error_scope=scope,
                    error_json=error_json,
                    job_status="failed",
                    reason=reason_code,
                    provider_error_code=provider_error_code,
                    provider_error_message=provider_error_message,
                    stage="poll",
                )
                sheet_ok = await self._db.log_error_record(error_record)
                await self._report_error(
                    error_record,
                    context="job_failed",
                    extra={
                        "gsheets_ok": sheet_ok,
                        "provider": provider_key,
                        "reason_message": error_message,
                        "provider_message": raw_error_text,
                        "safety_categories": categories,
                    },
                )
                log_event(
                    level="ERROR",
                    event="error",
                    corr_id=pending.corr_id,
                    job_id=pending.job_id,
                    user_id=pending.user_id,
                    username=pending.username,
                    model=pending.model,
                    provider=provider_key,
                    size=pending.size,
                    status_code=result.status_code,
                    duration_ms=result.duration_ms,
                    error_type="job_failed",
                    error_msg_short=error_message,
                    gsheets_ok=gsheets_ok and sheet_ok,
                    refund_done=refunded,
                    extra={
                        "provider_message": raw_error_text,
                        "error_scope": scope,
                        "error_json": error_json,
                        "safety_categories": categories,
                        "preflight_blocked": False,
                        "preflight_reason": pending.preflight_reason,
                        "preflight_scope": scope,
                        "preflight_auto_sanitized": pending.auto_sanitized,
                        "gemini_key_mask": key_mask or None,
                        "reason": reason_code,
                        "reason_message": error_message,
                        "provider_error_code": provider_error_code,
                        "provider_error_message": provider_error_message,
                    },
                )
                log_event(
                    level="INFO",
                    event="refund",
                    corr_id=pending.corr_id,
                    job_id=pending.job_id,
                    user_id=pending.user_id,
                    username=pending.username,
                    model=pending.model,
                    provider=provider_key,
                    size=pending.size,
                    credits_cost=self._config.generation_cost_credits,
                    refund_done=refunded,
                    extra={"gemini_key_mask": key_mask or None, "reason": reason_code},
                )
                increment_metric("veo_failed_total", tags={"reason": reason_code})
                await self._release_gate(
                    pending,
                    reason="job_failed",
                    status="failed",
                )
            else:
                # Job still running - requeue for later polling
                log.debug(
                    "Job still running job_id=%s corr_id=%s provider=%s status=%s requeueing", 
                    pending.job_id,
                    pending.corr_id,
                    provider_key,
                    result.status,
                )
                await self._db.update_job(
                    pending.job_id,
                    result.status,
                    video_id=pending.video_id,
                )
                await asyncio.sleep(self._poll_interval_seconds())
                await self.enqueue(
                    job_id=pending.job_id,
                    user_id=pending.user_id,
                    prompt=pending.prompt,
                    corr_id=pending.corr_id,
                    size=pending.size,
                    model=pending.model,
                    provider=provider_key,
                    username=pending.username,
                    original_prompt=pending.original_prompt,
                    sanitized_prompt=pending.sanitized_prompt,
                    auto_sanitized=pending.auto_sanitized,
                    preflight_reason=pending.preflight_reason,
                    preflight_scope=pending.preflight_scope,
                    task_type=ASSET_TASK_RETRIEVE,
                    asset_kind=asset_kind,
                    provider_job_id=provider_job_id,
                    video_id=pending.video_id,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                )
                return
        finally:
            job = await self._db.get_job(pending.job_id)
            if job:
                if "payload" not in job_extra and isinstance(
                    job_extra.get("provider_data"), dict
                ) and "payload" in job_extra["provider_data"]:
                    job_extra["payload"] = job_extra["provider_data"]["payload"]
                if job_extra:
                    job.extra.update(job_extra)
                await self._notify(job)

    async def _release_gate(
        self, pending: PendingJob, *, reason: str, status: Optional[str]
    ) -> None:
        if not self._gate:
            return
        status_value = status or reason
        setattr(pending, "metrics_status", status_value)
        try:
            await self._gate.release(
                user_id=pending.user_id,
                corr_id=pending.corr_id,
                reason=reason,
                status=status,
            )
        except Exception:  # pragma: no cover - defensive
            log.exception(
                "Failed to release gate for user %s corr_id=%s",
                pending.user_id,
                pending.corr_id,
            )

    async def _schedule_video_download(
        self,
        pending: PendingJob,
        *,
        provider_key: str,
        client: BaseProviderClient,
        primary_asset: Optional[Dict[str, Any]],
        assets_meta: List[Dict[str, Any]],
        result: ProviderJobStatus,
    ) -> None:
        video_id = pending.video_id
        if not video_id:
            return
        content_handler = getattr(client, "content", None)
        if not callable(content_handler):
            log.debug(
                "Provider %s does not expose content endpoint; skipping download scheduling job_id=%s",
                provider_key,
                pending.job_id,
            )
            return
        asset_id: Optional[str] = None
        if isinstance(primary_asset, dict):
            for key in ("asset_id", "id", "key", "name"):
                value = primary_asset.get(key)
                if isinstance(value, str) and value.strip():
                    asset_id = value.strip()
                    break
        if not asset_id:
            for entry in assets_meta:
                candidate = entry.get("asset_id") or entry.get("key")
                if isinstance(candidate, str) and candidate.strip():
                    asset_id = candidate.strip()
                    break
        if not asset_id and result.assets:
            first_key = next(iter(result.assets.keys()), None)
            if isinstance(first_key, str) and first_key.strip():
                asset_id = first_key.strip()
        await self.enqueue(
            job_id=pending.job_id,
            user_id=pending.user_id,
            prompt=pending.prompt,
            corr_id=pending.corr_id,
            size=pending.size,
            model=pending.model,
            provider=provider_key,
            username=pending.username,
            original_prompt=pending.original_prompt,
            sanitized_prompt=pending.sanitized_prompt,
            auto_sanitized=pending.auto_sanitized,
            preflight_reason=pending.preflight_reason,
            preflight_scope=pending.preflight_scope,
            task_type=VIDEO_TASK_DOWNLOAD,
            provider_job_id=pending.provider_job_id or pending.job_id,
            video_id=video_id,
            asset_id=asset_id,
            attempt=0,
            max_attempts=_DOWNLOAD_MAX_ATTEMPTS,
        )

    async def _handle_video_download(self, pending: PendingJob) -> None:
        provider_hint = pending.provider or ""
        provider_job_id = pending.provider_job_id or pending.job_id
        video_id = pending.video_id or ""
        asset_id = pending.asset_id or ""
        attempt = max(0, pending.attempt or 0)
        max_attempts = pending.max_attempts or _DOWNLOAD_MAX_ATTEMPTS
        job_extra: Dict[str, Any] = {}
        if pending.video_id:
            job_extra["video_id"] = pending.video_id
        download_meta: Dict[str, Any] = {"asset_id": asset_id, "attempt": attempt + 1}
        try:
            client, provider_key = self._resolve_provider(provider_hint or pending.model)
        except Exception:  # pragma: no cover - defensive
            log.exception("Failed to resolve provider for download job %s", pending.job_id)
            return
        content_handler = getattr(client, "content", None)
        if not callable(content_handler):
            log.debug(
                "Provider %s has no content handler; skipping download job job_id=%s",
                provider_key,
                pending.job_id,
            )
            return
        if not video_id:
            log.debug(
                "Skipping download for job_id=%s provider=%s - missing video_id",
                pending.job_id,
                provider_key,
            )
            return
        download_handler = getattr(client, "download_content", None)
        if callable(download_handler):
            await self._handle_inline_download(
                pending,
                client=client,
                provider_key=provider_key,
                result=ProviderJobStatus(
                    job_id=pending.job_id,
                    status="completed",
                    assets={},
                    error=None,
                    data={},
                    status_code=0,
                    duration_ms=0,
                ),
                video_url_value="",
                display_label="",
                done_extra={},
                job_extra=job_extra,
            )
            return
        log.debug(
            "Downloading content job_id=%s provider_job=%s provider=%s video_id=%s asset_id=%s attempt=%s/%s",
            pending.job_id,
            provider_job_id,
            provider_key,
            video_id,
            asset_id or "",
            attempt + 1,
            max_attempts,
        )
        try:
            body, headers = await content_handler(video_id, asset_id=asset_id or None)
        except ProviderAPIError as exc:
            key_mask = self._gemini_key_mask if provider_key.startswith(("veo", "gemini")) else None
            log.warning(
                "Download failed job_id=%s corr_id=%s provider=%s video_id=%s asset_id=%s status=%s error=%s attempt=%s/%s",
                pending.job_id,
                pending.corr_id,
                provider_key,
                video_id,
                asset_id or "",
                exc.status_code,
                exc,
                attempt + 1,
                max_attempts,
            )
            log_event(
                level="ERROR",
                event="content",
                corr_id=pending.corr_id,
                job_id=pending.job_id,
                user_id=pending.user_id,
                username=pending.username,
                model=pending.model,
                provider=provider_key,
                size=pending.size,
                status_code=exc.status_code,
                duration_ms=exc.duration_ms,
                extra={
                    "video_id": video_id,
                    "asset_id": asset_id or "",
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                    "error_code": exc.error_code,
                    "error_type": exc.error_type,
                    "gemini_key_mask": key_mask,
                },
            )
            if attempt + 1 < max_attempts:
                await asyncio.sleep(self._poll_interval_seconds())
                await self.enqueue(
                    job_id=pending.job_id,
                    user_id=pending.user_id,
                    prompt=pending.prompt,
                    corr_id=pending.corr_id,
                    size=pending.size,
                    model=pending.model,
                    provider=provider_key,
                    username=pending.username,
                    original_prompt=pending.original_prompt,
                    sanitized_prompt=pending.sanitized_prompt,
                    auto_sanitized=pending.auto_sanitized,
                    preflight_reason=pending.preflight_reason,
                    preflight_scope=pending.preflight_scope,
                    task_type=VIDEO_TASK_DOWNLOAD,
                    provider_job_id=provider_job_id,
                    video_id=video_id,
                    asset_id=asset_id,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                )
            return
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("Unexpected error while downloading content for job %s", pending.job_id)
            if attempt + 1 < max_attempts:
                await asyncio.sleep(self._poll_interval_seconds())
                await self.enqueue(
                    job_id=pending.job_id,
                    user_id=pending.user_id,
                    prompt=pending.prompt,
                    corr_id=pending.corr_id,
                    size=pending.size,
                    model=pending.model,
                    provider=provider_key,
                    username=pending.username,
                    original_prompt=pending.original_prompt,
                    sanitized_prompt=pending.sanitized_prompt,
                    auto_sanitized=pending.auto_sanitized,
                    preflight_reason=pending.preflight_reason,
                    preflight_scope=pending.preflight_scope,
                    task_type=VIDEO_TASK_DOWNLOAD,
                    provider_job_id=provider_job_id,
                    video_id=video_id,
                    asset_id=asset_id,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                )
            return
        size_bytes = len(body)
        content_type = ""
        if isinstance(headers, dict):
            for key, value in headers.items():
                if isinstance(key, str) and key.lower() == "content-type" and isinstance(value, str):
                    content_type = value
                    break
        download_meta["bytes"] = size_bytes
        if content_type:
            download_meta["content_type"] = content_type
        del body
        job_extra["download"] = download_meta
        log_event(
            level="INFO",
            event="content",
            corr_id=pending.corr_id,
            job_id=pending.job_id,
            user_id=pending.user_id,
            username=pending.username,
            model=pending.model,
            provider=provider_key,
            size=pending.size,
            extra={
                "video_id": video_id,
                "asset_id": asset_id or "",
                "bytes": size_bytes,
                "content_type": content_type,
                "attempt": attempt + 1,
            },
        )
        job = await self._db.get_job(pending.job_id)
        if job:
            job.extra.update(job_extra)
            await self._notify(job)

    async def _handle_inline_download(
        self,
        pending: PendingJob,
        *,
        client: BaseProviderClient,
        provider_key: str,
        result: ProviderJobStatus,
        video_url_value: str,
        display_label: str,
        done_extra: Dict[str, Any],
        job_extra: Dict[str, Any],
    ) -> bool:
        download_handler = getattr(client, "download_content", None)
        if not callable(download_handler) or not pending.video_id:
            return False
        try:
            file_path = await download_handler(pending.video_id)
        except ProviderAPIError as exc:
            await self._handle_download_failure_inline(
                pending,
                provider_key=provider_key,
                error=exc,
                job_extra=job_extra,
            )
            return False
        except Exception as exc:  # pragma: no cover - defensive
            log.exception(
                "Unexpected error during download job_id=%s video_id=%s provider=%s",
                pending.job_id,
                pending.video_id,
                provider_key,
            )
            fallback_error = ProviderAPIError(
                provider=provider_key,
                status_code=0,
                message="Unexpected download error",
                error_type="unknown",
                error_code="unknown",
                provider_message=str(exc),
            )
            await self._handle_download_failure_inline(
                pending,
                provider_key=provider_key,
                error=fallback_error,
                job_extra=job_extra,
            )
            return False
        meta = getattr(client, "_last_download_meta", {}) or {}
        request_id = str(meta.get("request_id") or "")
        status_code = int(meta.get("status_code") or result.status_code)
        duration_ms = int(meta.get("duration_ms") or result.duration_ms)
        attempt = int(meta.get("attempt") or 1)
        size_bytes = 0
        try:
            size_bytes = file_path.stat().st_size
        except Exception:
            size_bytes = 0
        download_meta: Dict[str, Any] = {
            "status": "success",
            "file_path": str(file_path),
            "bytes": size_bytes,
            "format": file_path.suffix.lstrip("."),
            "request_id": request_id,
            "status_code": status_code,
            "attempt": attempt,
            "url": meta.get("url", ""),
        }
        job_extra.setdefault("download", {}).update(download_meta)
        billing_extra: Dict[str, Any] = {}
        credits_cost = self._config.generation_cost_credits
        charged = False
        if credits_cost > 0:
            try:
                charged = await self._db.deduct_credit(pending.user_id, credits_cost)
            except Exception:  # pragma: no cover - external dependency
                log.exception(
                    "Failed to deduct credits after download job_id=%s user=%s",
                    pending.job_id,
                    pending.user_id,
                )
                billing_extra.update({"status": "error", "credits": credits_cost})
            else:
                if charged:
                    billing_extra.update({"status": "charged", "credits": credits_cost})
                else:
                    billing_extra.update({"status": "insufficient", "credits": credits_cost})
                    log.warning(
                        "Credits charge declined user=%s job_id=%s",  # pragma: no cover - accounting guard
                        pending.user_id,
                        pending.job_id,
                    )
        else:
            billing_extra.update({"status": "free", "credits": 0})
        if billing_extra:
            job_extra.setdefault("billing", {}).update(billing_extra)
        log_event(
            level="INFO",
            event="content",
            corr_id=pending.corr_id,
            job_id=pending.job_id,
            user_id=pending.user_id,
            username=pending.username,
            model=pending.model,
            provider=provider_key,
            size=pending.size,
            status_code=status_code,
            duration_ms=duration_ms,
            extra={
                "video_id": pending.video_id or "",
                "bytes": size_bytes,
                "request_id": request_id,
                "attempt": attempt,
                "file_path": str(file_path),
                "billing": billing_extra or None,
            },
        )
        gsheets_ok = True
        try:
            await self._db.update_job(
                pending.job_id,
                "completed",
                video_url=video_url_value,
                file_url=video_url_value or str(file_path),
                video_id=pending.video_id,
                error=None,
            )
        except Exception:
            log.exception("Failed to update job %s as completed", pending.job_id)
            gsheets_ok = False
        done_extra = dict(done_extra)
        download_meta_shallow = {
            "bytes": size_bytes,
            "request_id": request_id,
            "attempt": attempt,
        }
        done_extra["download"] = download_meta_shallow
        if billing_extra:
            done_extra["billing"] = dict(billing_extra)
        log_event(
            level="INFO",
            event="done",
            corr_id=pending.corr_id,
            job_id=pending.job_id,
            user_id=pending.user_id,
            username=pending.username,
            model=pending.model,
            provider=provider_key,
            size=pending.size,
            status_code=result.status_code,
            duration_ms=result.duration_ms,
            gsheets_ok=gsheets_ok,
            extra=done_extra,
        )
        job = await self._db.get_job(pending.job_id)
        if job:
            if job_extra:
                job.extra.update(job_extra)
            await self._notify(job)
        return True

    async def _handle_download_failure_inline(
        self,
        pending: PendingJob,
        *,
        provider_key: str,
        error: ProviderAPIError,
        job_extra: Dict[str, Any],
    ) -> None:
        message = "Видео сгенерировано, но скачать его не удалось; попробуйте позже"
        log.warning(
            "Download failed job_id=%s video_id=%s provider=%s status=%s error_code=%s",
            pending.job_id,
            pending.video_id,
            provider_key,
            error.status_code,
            error.error_code or error.code,
        )
        download_info = job_extra.setdefault("download", {})
        download_info.update(
            {
                "status": "failed",
                "error_code": error.error_code or error.code,
                "status_code": error.status_code,
                "provider_message": error.provider_message,
                "request_id": getattr(error, "request_id", ""),
                "message": message,
            }
        )
        gsheets_ok = True
        try:
            await self._db.update_job(
                pending.job_id,
                "failed_to_download",
                video_id=pending.video_id,
                error=message,
            )
        except Exception:
            log.exception("Failed to mark job %s as failed_to_download", pending.job_id)
            gsheets_ok = False
        billing_info = job_extra.setdefault("billing", {})
        refund_amount = self._config.generation_cost_credits
        job_record = None
        try:
            job_record = await self._db.get_job(pending.job_id)
        except Exception:
            log.exception("Failed to load job %s for refund handling", pending.job_id)
        if job_record and job_record.cost_credits:
            refund_amount = job_record.cost_credits
        if refund_amount > 0:
            billing_info.setdefault("credits", refund_amount)
        if "status" not in billing_info:
            billing_info["status"] = "charged" if refund_amount > 0 else "skipped"
        refunded = False
        if billing_info.get("status") == "charged" and refund_amount > 0:
            try:
                await self._db.add_credits(pending.user_id, refund_amount)
                refunded = True
                increment_metric("refunds_total")
                increment_metric("refund_total")
            except Exception:
                log.exception(
                    "Refunding credits failed for download failure job %s", pending.job_id
                )
        log_event(
            level="ERROR",
            event="download_failed",
            corr_id=pending.corr_id,
            job_id=pending.job_id,
            user_id=pending.user_id,
            username=pending.username,
            model=pending.model,
            provider=provider_key,
            size=pending.size,
            status_code=error.status_code,
            duration_ms=error.duration_ms,
            error_type=error.error_type or "download_failed",
            error_msg_short=message,
            gsheets_ok=gsheets_ok,
            refund_done=refunded,
            extra={
                "video_id": pending.video_id or "",
                "provider_message": error.provider_message,
                "error_code": error.error_code or error.code,
                "request_id": getattr(error, "request_id", ""),
            },
        )

    async def _handle_no_media_assets(
        self,
        pending: PendingJob,
        *,
        provider_key: str,
        key_mask: str,
        provider_data: Dict[str, Any],
        result: ProviderJobStatus,
    ) -> None:
        self._reset_operation_tracker(pending.job_id)
        message = "Провайдер завершил операцию без доступных видеофайлов"
        increment_metric("veo_completed_without_assets")
        log.warning(
            "No media assets after completion job_id=%s corr_id=%s provider=%s",
            pending.job_id,
            pending.corr_id,
            provider_key,
        )
        try:
            await self._db.update_job(
                pending.job_id,
                "failed",
                error=message,
                video_id=pending.video_id,
            )
        except Exception:
            log.exception("Failed to update job %s after empty assets", pending.job_id)
        refunded = False
        try:
            await self._db.add_credits(pending.user_id, self._config.generation_cost_credits)
            refunded = True
            increment_metric("refunds_total")
            increment_metric("refund_total")
        except Exception:
            log.exception("Refunding credits failed for no-media job %s", pending.job_id)

        snippet_source: Any = provider_data.get("operation") if isinstance(provider_data, dict) else None
        if not isinstance(snippet_source, dict):
            snippet_source = provider_data
        try:
            operation_snippet = json.dumps(snippet_source, ensure_ascii=False)[:512]
        except Exception:  # pragma: no cover - defensive serialisation
            operation_snippet = str(snippet_source)[:512]

        error_record = ErrorLogRecord(
            ts=datetime.now(timezone.utc),
            user_id=pending.user_id,
            username=pending.username,
            corr_id=pending.corr_id,
            job_id=pending.job_id,
            model=pending.model,
            size=pending.size,
            status_code=result.status_code,
            error_type="no_media",
            error_msg_short=message,
            refunded=refunded,
            preflight_blocked=False,
            preflight_reason=pending.preflight_reason or "",
            auto_sanitized=pending.auto_sanitized,
            sanitized_prompt=pending.sanitized_prompt or pending.prompt,
            error_scope="unknown",
            error_json="",
            job_status="failed",
            reason="no_media_assets",
            provider_error_code=None,
            provider_error_message=None,
            stage="poll",
        )
        sheet_ok = await self._db.log_error_record(error_record)
        await self._report_error(
            error_record,
            context="no_media_assets",
            extra={
                "gsheets_ok": sheet_ok,
                "provider": provider_key,
                "operation_snippet": operation_snippet,
                "assets_recovery_attempted": provider_data.get(
                    "assets_recovery_attempted"
                )
                if isinstance(provider_data, dict)
                else None,
                "no_media_confirmed": provider_data.get("no_media_confirmed")
                if isinstance(provider_data, dict)
                else None,
            },
        )

        log_event(
            level="ERROR",
            event="error",
            corr_id=pending.corr_id,
            job_id=pending.job_id,
            user_id=pending.user_id,
            username=pending.username,
            model=pending.model,
            provider=provider_key,
            size=pending.size,
            status_code=result.status_code,
            duration_ms=result.duration_ms,
            error_type="no_media_assets",
            error_msg_short=message,
            gsheets_ok=sheet_ok,
            refund_done=refunded,
            extra={
                "reason": "no_media_assets_after_completed",
                "failure_reason": "no_media_assets",
                "gemini_key_mask": key_mask or None,
                "operation_snippet": operation_snippet,
                "assets_recovery_attempted": provider_data.get("assets_recovery_attempted"),
                "no_media_confirmed": provider_data.get("no_media_confirmed"),
            },
        )
        log_event(
            level="INFO",
            event="refund",
            corr_id=pending.corr_id,
            job_id=pending.job_id,
            user_id=pending.user_id,
            username=pending.username,
            model=pending.model,
            provider=provider_key,
            size=pending.size,
            credits_cost=self._config.generation_cost_credits,
            refund_done=refunded,
            extra={"gemini_key_mask": key_mask or None, "reason": "no_media_assets"},
        )
        await self._release_gate(pending, reason="no_media", status="failed")

    async def _handle_image_generate(self, pending: PendingJob) -> None:
        pending.asset_kind = ASSET_KIND_IMAGE
        log_event(
            level="INFO",
            event="image_generate",
            corr_id=pending.corr_id,
            job_id=pending.job_id,
            user_id=pending.user_id,
            username=pending.username,
            model=pending.model,
            provider=pending.provider,
            size=pending.size,
            extra={"task": IMAGE_TASK_GENERATE},
        )
        await self._handle_asset_retrieve(pending)

    async def _process_job(self, pending: PendingJob) -> None:
        task_type = pending.task_type or ASSET_TASK_RETRIEVE
        provider_label = pending.provider or pending.model
        job_started(provider=provider_label)
        status_label = "success"
        try:
            if task_type == VIDEO_TASK_DOWNLOAD:
                await self._handle_video_download(pending)
                return
            if task_type in {VIDEO_TASK_CREATE, VIDEO_TASK_REMIX}:
                stage = "create" if task_type == VIDEO_TASK_CREATE else "remix"
                provider_key = pending.provider or pending.model
                log_event(
                    level="INFO",
                    event=stage,
                    corr_id=pending.corr_id,
                    job_id=pending.job_id,
                    user_id=pending.user_id,
                    username=pending.username,
                    model=pending.model,
                    provider=provider_key,
                    size=pending.size,
                    extra={
                        "task_type": task_type,
                        "video_id": pending.video_id,
                    },
                )
            if task_type == IMAGE_TASK_GENERATE:
                await self._handle_image_generate(pending)
                return
            await self._handle_asset_retrieve(pending)
        except Exception:
            status_label = "error"
            job_failed(task_type=task_type, provider=provider_label)
            raise
        finally:
            status_value = getattr(pending, "metrics_status", status_label)
            job_finished(task_type=task_type, provider=provider_label, status=status_value)


__all__ = ["JobQueue", "hydrate_image_artifacts", "IMAGE_TASK_GENERATE", "IMAGE_TASK_RETRIEVE"]
