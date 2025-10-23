"""Background job queue for video generation."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from config import Config
from db import Database, ErrorLogRecord, GenerationJobRecord
from moderation import classify_safety_response, policy_message, render_error_json
from observability import log_event
from services.gemini_key import current_key_mask, ensure_gemini_key_logged
from generation_gate import GenerationRequestGate
from providers import BaseProviderClient, ProviderAPIError, ProviderJobStatus, ProviderJobSubmission

log = logging.getLogger(__name__)


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
        self._gemini_key_mask = ensure_gemini_key_logged(
            config,
            context="job_queue",
            process_id=f"pid={os.getpid()}",  # pragma: no cover - runtime value
            logger=log,
        )

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
            idempotency_key=idempotency_key or corr_id,
        )
        job_id = submission.job_id
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
        try:
            client, provider_key = self._resolve_provider(pending.provider)
            if provider_key.startswith(("veo", "gemini")):
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
                    original_prompt=pending.original_prompt,
                    sanitized_prompt=pending.sanitized_prompt,
                    auto_sanitized=pending.auto_sanitized,
                    preflight_reason=pending.preflight_reason,
                    preflight_scope=pending.preflight_scope,
                )
                return
            raw_inline = []
            if isinstance(result.data, dict):
                raw_inline = result.data.get("inline_assets") or []
            if isinstance(raw_inline, list):
                inline_assets = [dict(asset) for asset in raw_inline if isinstance(asset, dict)]
            inline_asset_map = {
                f"inline_{idx}": asset for idx, asset in enumerate(inline_assets)
            }
            assets_meta = []
            for key, value in result.assets.items():
                entry: Dict[str, Any] = {"key": key}
                inline_meta = inline_asset_map.get(key)
                if inline_meta:
                    entry.update(
                        {
                            "inline": True,
                            "kind": inline_meta.get("kind"),
                            "mime": inline_meta.get("mime"),
                            "bytes": inline_meta.get("bytes"),
                        }
                    )
                else:
                    entry.update({"inline": False, "url": value})
                assets_meta.append(entry)
            provider_payload: Optional[Dict[str, Any]] = None
            provider_data: Optional[Dict[str, Any]] = None
            if isinstance(result.data, dict):
                provider_data = result.data
                raw_payload = provider_data.get("payload")
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
                        inline_asset_map[key] = inline_entry
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
                first_inline = inline_assets[0]
                mime = first_inline.get("mime") or "application/octet-stream"
                size_bytes = first_inline.get("bytes") or 0
                asset_label = f"[inline {mime}, {size_bytes} bytes]"
                job_extra["inline_assets"] = inline_assets
            else:
                asset_label = next(iter(result.assets.values()), None)
                if asset_label:
                    mime_hint = None
                    for entry in assets_meta:
                        if entry.get("url") == asset_label and entry.get("mime"):
                            mime_hint = entry.get("mime")
                            break
                    descriptor = mime_hint or "asset"
                    asset_label = f"[remote {descriptor}]"
            if assets_meta:
                job_extra["assets_meta"] = assets_meta
            if key_mask:
                job_extra.setdefault("gemini_key_mask", key_mask)
            if provider_data:
                job_extra.setdefault("provider_data", provider_data)
            poll_extra: Dict[str, Any]
            if assets_meta:
                poll_extra = {"status": result.status, "assets_meta": assets_meta, "gemini_key_mask": key_mask or None}
            else:
                poll_extra = {"status": result.status, "gemini_key_mask": key_mask or None}
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
            if result.status == "completed" and asset_label:
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
                        video_url=asset_label,
                        error=None,
                    )
                except Exception:
                    log.exception("Failed to update job %s as completed", pending.job_id)
                    gsheets_ok = False
                done_extra: Dict[str, Any] = {"assets": list(result.assets.keys())}
                if assets_meta:
                    done_extra["assets_meta"] = assets_meta
                if key_mask:
                    done_extra["gemini_key_mask"] = key_mask
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
                log.warning(
                    "Job failed job_id=%s corr_id=%s provider=%s status=%s error=%s",
                    pending.job_id,
                    pending.corr_id,
                    provider_key,
                    result.status,
                    result.error,
                )
                raw_error = result.error or "Unknown error"
                response_payload: Optional[Dict[str, Any]] = None
                if isinstance(result.data, dict):
                    response_payload = result.data.get("response")
                scope, categories, summary = classify_safety_response(response_payload)
                if scope == "unknown" and pending.preflight_scope:
                    scope = pending.preflight_scope
                error_json = render_error_json(summary)
                policy_text = policy_message(scope)
                if scope != "unknown":
                    error_message = policy_text
                elif raw_error.strip().upper() in {"STOP", "SAFETY"}:
                    error_message = policy_message("unknown")
                else:
                    error_message = raw_error
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
                        preflight_blocked=False,
                        preflight_reason=pending.preflight_reason or "",
                        auto_sanitized=pending.auto_sanitized,
                        sanitized_prompt=pending.sanitized_prompt or pending.prompt,
                        error_scope=scope,
                        error_json=error_json,
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
                    extra={
                        "provider_message": raw_error,
                        "error_scope": scope,
                        "error_json": error_json,
                        "safety_categories": categories,
                        "preflight_blocked": False,
                        "preflight_reason": pending.preflight_reason,
                        "preflight_scope": scope,
                        "preflight_auto_sanitized": pending.auto_sanitized,
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
                    extra={"gemini_key_mask": key_mask or None},
                )
                await self._release_gate(
                    pending,
                    reason="job_failed",
                    status=result.status,
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


__all__ = ["JobQueue"]
