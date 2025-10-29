"""Background job queue for video generation."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from config import Config
from db import Database, ErrorLogRecord, GenerationJobRecord
from moderation import classify_safety_response, policy_message, render_error_json
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
    user_message = _REASON_MESSAGES.get(reason_code) or fallback_message or _REASON_MESSAGES[
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


class JobQueue:
    """Manage generation jobs with concurrency limits."""

    def __init__(
        self,
        *,
        db: Database,
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
        self._error_reporter = error_reporter
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
            ts=datetime.utcnow(),
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
    ) -> GenerationJobRecord:
        model_key = model or self._config.default_video_model
        client, provider_key = self._resolve_provider(provider or model_key)
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
        submission: ProviderJobSubmission = await client.enqueue_job(
            prompt=effective_prompt,
            settings=request_settings or None,
            preset=preset,
            payload=payload,
            idempotency_key=stable_idempotency_key,
        )
        job_id = submission.job_id
        operation_name = submission.data.get("operation_name") if isinstance(submission.data, dict) else None
        now = datetime.utcnow()
        record = GenerationJobRecord(
            id=job_id,
            user_id=user_id,
            prompt=effective_prompt,
            status="queued",
            video_url=None,
            error=None,
            created_at=now,
            updated_at=now,
            image_file_id=image_file_id,
            size=size,
            model=model_key,
            cost_credits=self._config.generation_cost_credits,
            username=username,
            corr_id=corr_id,
            content_type=content_type_value,
            idempotency_key=stable_idempotency_key,
            operation_name=operation_name,
        )
        await self._db.create_job(record)
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
            credits_cost=self._config.generation_cost_credits,
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
    ) -> None:
        provider_key = provider or model
        log.debug(
            "Enqueueing pending job job_id=%s provider=%s corr_id=%s user_id=%s", 
            job_id,
            provider_key,
            corr_id,
            user_id,
        )
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
            )
        )

    async def _recover_pending_jobs(self) -> None:
        pending = await self._db.list_pending_jobs(limit=100)
        for job in pending:
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
            )
        if pending:
            log.info("Recovered %s pending jobs", len(pending))

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

    async def _process_job(self, pending: PendingJob) -> None:
        provider_hint = pending.provider or ""
        key_mask = self._gemini_key_mask if provider_hint.startswith(("veo", "gemini")) else ""
        log.info(
            "Processing job %s corr_id=%s provider=%s key_mask=%s env=%s",
            pending.job_id,
            pending.corr_id,
            provider_hint,
            key_mask or "",
            self._environment,
        )
        await self._db.update_job(pending.job_id, "running")
        result: Optional[ProviderJobStatus] = None
        inline_assets: List[Dict[str, Any]] = []
        assets_meta: List[Dict[str, Any]] = []
        job_extra: Dict[str, Any] = {}
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
            try:
                log.debug(
                    "Polling provider job job_id=%s provider=%s corr_id=%s",
                    pending.job_id,
                    provider_key,
                    pending.corr_id,
                )
                result = await client.get_job_status(pending.job_id)
            except ProviderAPIError as exc:
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
                error_extra = {"provider_message": exc.provider_message, "gemini_key_mask": key_mask or None}
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
                )
                return
            except Exception as exc:  # pragma: no cover - defensive network guard
                log.exception("Unexpected error while polling job %s", pending.job_id)
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
                )
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
            if assets_meta:
                job_extra["assets_meta"] = assets_meta
            if key_mask:
                job_extra.setdefault("gemini_key_mask", key_mask)
            if provider_data:
                job_extra.setdefault("provider_data", provider_data)

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
            poll_extra: Dict[str, Any] = {"status": result.status, "gemini_key_mask": key_mask or None}
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
                if not inline_assets and not asset_value:
                    if no_media_confirmed:
                        await self._handle_no_media_assets(
                            pending,
                            provider_key=provider_key,
                            key_mask=key_mask,
                            provider_data=provider_data,
                            result=result,
                        )
                        return
                    log.warning(
                        "Job completed without assets job_id=%s corr_id=%s provider=%s retrying",
                        pending.job_id,
                        pending.corr_id,
                        provider_key,
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
                    )
                    return

                video_url_value = asset_value or asset_label_display or ""
                display_label = asset_label_display or video_url_value or ""
                log.debug(
                    "Job completed job_id=%s corr_id=%s provider=%s assets=%s",
                    pending.job_id,
                    pending.corr_id,
                    provider_key,
                    list(result.assets.keys()),
                )
                gsheets_ok = True
                try:
                    await self._db.update_job(
                        pending.job_id,
                        "completed",
                        video_url=video_url_value,
                        file_url=video_url_value,
                        error=None,
                    )
                except Exception:
                    log.exception("Failed to update job %s as completed", pending.job_id)
                    gsheets_ok = False
                if primary_asset:
                    job_extra.setdefault("primary_asset", primary_asset)
                done_extra: Dict[str, Any] = {"assets": list(result.assets.keys())}
                if assets_meta:
                    done_extra["assets_meta"] = assets_meta
                if primary_asset:
                    done_extra["primary_asset"] = primary_asset
                if recovery_attempted:
                    done_extra["assets_recovery_attempted"] = True
                if key_mask:
                    done_extra["gemini_key_mask"] = key_mask
                done_extra["video_label"] = display_label
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
                    ts=datetime.utcnow(),
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
                await self._db.update_job(pending.job_id, result.status)
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
                )
                return
        finally:
            job = await self._db.get_job(pending.job_id)
            if job:
                if job_extra:
                    job.extra.update(job_extra)
                await self._notify(job)

    async def _release_gate(
        self, pending: PendingJob, *, reason: str, status: Optional[str]
    ) -> None:
        if not self._gate:
            return
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
            await self._db.update_job(pending.job_id, "failed", error=message)
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
            ts=datetime.utcnow(),
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


__all__ = ["JobQueue"]
