"""Background job queue for video generation."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from config import Config
from db import Database, ErrorLogRecord, GenerationJobRecord
from moderation import classify_safety_response, policy_message, render_error_json
from observability import increment_metric, log_event
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
            content_type=content_type or "video",
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

            poll_extra: Dict[str, Any] = {"status": result.status, "gemini_key_mask": key_mask or None}
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
                    increment_metric("refunds_total")
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

    async def _handle_no_media_assets(
        self,
        pending: PendingJob,
        *,
        provider_key: str,
        key_mask: str,
        provider_data: Dict[str, Any],
        result: ProviderJobStatus,
    ) -> None:
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
        )
        sheet_ok = await self._db.log_error_record(error_record)

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
            extra={"gemini_key_mask": key_mask or None},
        )
        await self._release_gate(pending, reason="no_media", status="failed")


__all__ = ["JobQueue"]
