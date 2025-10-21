"""Background job queue for Sora video generation."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

from config import Config
from db import Database, ErrorLogRecord, GenerationJobRecord
from observability import log_event
from providers import BaseProviderClient, ProviderAPIError, ProviderJobStatus, ProviderJobSubmission

log = logging.getLogger(__name__)


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


class JobQueue:
    """Manage generation jobs with concurrency limits."""

    def __init__(
        self,
        *,
        db: Database,
        providers: Dict[str, BaseProviderClient],
        default_provider: str,
        config: Config,
    ) -> None:
        self._db = db
        self._providers = providers
        self._default_provider = default_provider
        self._config = config
        self._queue: asyncio.Queue[PendingJob] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._stopped = asyncio.Event()
        self._notification_callbacks: Dict[str, Callable[[GenerationJobRecord], Awaitable[None]]] = {}

    def register_notification_callback(
        self, *, name: str, callback: Callable[[GenerationJobRecord], Awaitable[None]]
    ) -> None:
        self._notification_callbacks[name] = callback

    def _resolve_provider(self, provider_key: Optional[str]) -> tuple[BaseProviderClient, str]:
        key = provider_key or self._default_provider
        client = self._providers.get(key)
        if client is None:
            log.warning("Unknown provider %s, falling back to %s", provider_key, self._default_provider)
            key = self._default_provider
            client = self._providers[key]
        return client, key

    async def start(self) -> None:
        if self._workers:
            return
        self._stopped.clear()
        for _ in range(self._config.jobs_concurrency):
            task = asyncio.create_task(self._worker())
            self._workers.append(task)
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
    ) -> GenerationJobRecord:
        model_key = model or self._config.sora_model
        client, provider_key = self._resolve_provider(provider or model_key)
        request_settings: Dict[str, Any] = dict(settings or {})
        if size:
            request_settings.setdefault("size", size)
        request_settings.setdefault("model", model_key)
        if image_file_id:
            request_settings.setdefault("image_file_id", image_file_id)
        submission: ProviderJobSubmission = await client.enqueue_job(
            prompt=prompt,
            settings=request_settings or None,
            preset=preset,
            payload=payload,
            idempotency_key=corr_id,
        )
        job_id = submission.job_id
        now = datetime.utcnow()
        record = GenerationJobRecord(
            id=job_id,
            user_id=user_id,
            prompt=prompt,
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
        )
        await self._db.create_job(record)
        await self.enqueue(
            job_id=job_id,
            user_id=user_id,
            prompt=prompt,
            corr_id=corr_id,
            size=size,
            model=model_key,
            provider=provider_key,
            username=username,
        )
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
            prompt=prompt,
            gsheets_ok=True,
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
    ) -> None:
        provider_key = provider or model
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
                model=job.model or self._config.sora_model,
                provider=job.model or self._default_provider,
                username=job.username,
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
        log.info("Processing job %s", pending.job_id)
        await self._db.update_job(pending.job_id, "running")
        try:
            client, provider_key = self._resolve_provider(pending.provider)
            try:
                result: ProviderJobStatus = await client.get_job_status(pending.job_id)
            except ProviderAPIError as exc:
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
                    extra={"provider_message": exc.provider_message},
                )
                await asyncio.sleep(3)
                await self.enqueue(
                    job_id=pending.job_id,
                    user_id=pending.user_id,
                    prompt=pending.prompt,
                    corr_id=pending.corr_id,
                    size=pending.size,
                    model=pending.model,
                    provider=provider_key,
                    username=pending.username,
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
                )
                await asyncio.sleep(3)
                await self.enqueue(
                    job_id=pending.job_id,
                    user_id=pending.user_id,
                    prompt=pending.prompt,
                    corr_id=pending.corr_id,
                    size=pending.size,
                    model=pending.model,
                    provider=provider_key,
                    username=pending.username,
                )
                return
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
                extra={"status": result.status},
            )
            asset_url = next(iter(result.assets.values()), None)
            if result.status == "completed" and asset_url:
                gsheets_ok = True
                try:
                    await self._db.update_job(
                        pending.job_id,
                        "completed",
                        video_url=asset_url,
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
                    extra={"assets": list(result.assets.keys())},
                )
            elif result.status in {"failed", "errored"}:
                error_message = result.error or "Unknown error"
                gsheets_ok = True
                try:
                    await self._db.update_job(
                        pending.job_id,
                        "failed",
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
                except Exception:
                    log.exception("Refunding credits failed for job %s", pending.job_id)
                sheet_ok = await self._db.log_error_record(
                    ErrorLogRecord(
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
                    )
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
                    extra={"provider_message": error_message},
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
                )
            else:
                # Job still running - requeue for later polling
                await self._db.update_job(pending.job_id, result.status)
                await asyncio.sleep(3)
                await self.enqueue(
                    job_id=pending.job_id,
                    user_id=pending.user_id,
                    prompt=pending.prompt,
                    corr_id=pending.corr_id,
                    size=pending.size,
                    model=pending.model,
                    provider=provider_key,
                    username=pending.username,
                )
                return
        finally:
            job = await self._db.get_job(pending.job_id)
            if job:
                await self._notify(job)


__all__ = ["JobQueue"]
