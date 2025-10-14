"""Background job queue for Sora video generation."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable, Dict, Optional

from config import Config
from db import Database, GenerationJobRecord
from sora_client import SoraClient

log = logging.getLogger(__name__)


@dataclass
class PendingJob:
    job_id: str
    user_id: int
    prompt: str


class JobQueue:
    """Manage Sora generation jobs with concurrency limits."""

    def __init__(
        self,
        *,
        db: Database,
        sora_client: SoraClient,
        config: Config,
    ) -> None:
        self._db = db
        self._sora_client = sora_client
        self._config = config
        self._queue: asyncio.Queue[PendingJob] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._stopped = asyncio.Event()
        self._notification_callbacks: Dict[str, Callable[[GenerationJobRecord], Awaitable[None]]] = {}

    def register_notification_callback(
        self, *, name: str, callback: Callable[[GenerationJobRecord], Awaitable[None]]
    ) -> None:
        self._notification_callbacks[name] = callback

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

    async def submit(
        self,
        *,
        user_id: int,
        prompt: str,
        size: str,
        model: str,
        image_file_id: Optional[str] = None,
    ) -> GenerationJobRecord:
        settings = {"size": size, "model": model}
        if image_file_id:
            settings["image_file_id"] = image_file_id
        job_id = await self._sora_client.submit_job(prompt=prompt, settings=settings)
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
            model=model,
        )
        await self._db.create_job(record)
        await self.enqueue(job_id=job_id, user_id=user_id, prompt=prompt)
        return record

    async def enqueue(self, *, job_id: str, user_id: int, prompt: str) -> None:
        await self._queue.put(PendingJob(job_id=job_id, user_id=user_id, prompt=prompt))

    async def _recover_pending_jobs(self) -> None:
        pending = await self._db.list_pending_jobs(limit=100)
        for job in pending:
            await self.enqueue(job_id=job.id, user_id=job.user_id, prompt=job.prompt)
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
            try:
                result = await self._sora_client.fetch_job(pending.job_id)
            except Exception as exc:
                log.warning("Failed to poll job %s: %s", pending.job_id, exc, exc_info=True)
                await asyncio.sleep(3)
                await self.enqueue(job_id=pending.job_id, user_id=pending.user_id, prompt=pending.prompt)
                return
            if result.status == "completed" and result.video_url:
                await self._db.update_job(
                    pending.job_id,
                    "completed",
                    video_url=result.video_url,
                    error=None,
                )
            elif result.status in {"failed", "errored"}:
                await self._db.update_job(
                    pending.job_id,
                    "failed",
                    error=result.error or "Unknown error",
                )
                await self._db.add_credits(pending.user_id, self._config.credits_per_generation)
            else:
                # Job still running - requeue for later polling
                await self._db.update_job(pending.job_id, result.status)
                await asyncio.sleep(3)
                await self.enqueue(job_id=pending.job_id, user_id=pending.user_id, prompt=pending.prompt)
                return
        finally:
            job = await self._db.get_job(pending.job_id)
            if job:
                await self._notify(job)


__all__ = ["JobQueue"]
