from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

import gsheets_db

from .interface import DatabaseInterface
from .models import (
    ArchiveLogRecord,
    ErrorLogRecord,
    GenerationJobRecord,
    User,
    job_from_mapping,
    normalise_job_status,
    parse_metadata,
    user_from_mapping,
)

log = logging.getLogger(__name__)


class SheetsDatabase(DatabaseInterface):
    """Google Sheets-backed implementation of :class:`DatabaseInterface`."""

    async def init(self) -> None:
        await gsheets_db.init()

    async def close(self) -> None:  # pragma: no cover - nothing to close
        return None

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    async def ensure_user(
        self,
        telegram_id: int,
        username: Optional[str],
        *,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> User:
        if username is None and first_name is None and last_name is None:
            data = await gsheets_db.get_or_create_user(telegram_id)
        else:
            data = await gsheets_db.upsert_user_profile(
                telegram_id,
                username=username,
                first_name=first_name,
                last_name=last_name,
            )
        return user_from_mapping(data)

    async def sync_user_profile(
        self,
        telegram_id: int,
        *,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> None:
        if username is None and first_name is None and last_name is None:
            return
        try:
            await gsheets_db.upsert_user_profile(
                telegram_id,
                username=username,
                first_name=first_name,
                last_name=last_name,
            )
        except Exception:  # pragma: no cover - external dependency
            log.warning("Failed to sync user profile for %s", telegram_id, exc_info=True)

    async def get_user_credits(self, telegram_id: int) -> int:
        data = await gsheets_db.get_or_create_user(telegram_id)
        return int(data.get("credits", 0))

    async def add_credits(self, telegram_id: int, amount: int) -> int:
        return await gsheets_db.add_credits(telegram_id, amount)

    async def deduct_credit(self, telegram_id: int, amount: int = 1) -> bool:
        try:
            await gsheets_db.add_credits(telegram_id, -amount)
            return True
        except ValueError:
            return False

    async def is_bonus_granted(self, telegram_id: int) -> bool:
        data = await gsheets_db.get_or_create_user(telegram_id)
        return bool(data.get("bonus_granted"))

    async def mark_bonus_granted(self, telegram_id: int) -> None:
        await gsheets_db.mark_bonus_granted(telegram_id)

    async def mark_economy_v2(self, telegram_id: int) -> None:
        await gsheets_db.mark_economy_v2(telegram_id)

    async def count_active_jobs(self, user_id: int) -> int:
        return await gsheets_db.count_jobs_by_status(user_id, ("queued", "running"))

    async def migrate_credit_balances(self, multiplier: int) -> int:
        users = await gsheets_db.list_users()
        migrated = 0
        for record in users:
            user_id = int(record.get("user_id", 0))
            if not user_id:
                continue
            if record.get("economy_v2"):
                continue
            current = int(record.get("credits", 0))
            if current > 0:
                new_balance = current * multiplier
                await gsheets_db.set_user_credits(user_id, new_balance)
                migrated += 1
                log.info(
                    "Migrated user %s credits=%s cost_multiplier=%s", user_id, current, multiplier
                )
            await gsheets_db.mark_economy_v2(user_id)
        if migrated:
            log.info("Credit economy migration applied to %s users", migrated)
        return migrated

    async def backfill_user_profiles(
        self, fetcher: Callable[[int], Awaitable[Optional[Dict[str, Optional[str]]]]]
    ) -> None:
        try:
            await gsheets_db.backfill_user_profiles(fetcher)
        except Exception:  # pragma: no cover - background guard
            log.warning("Failed to backfill user profiles", exc_info=True)

    # ------------------------------------------------------------------
    # Payments
    # ------------------------------------------------------------------

    async def record_payment(
        self,
        *,
        user_id: int,
        provider_payment_charge_id: str,
        telegram_payment_charge_id: str,
        amount: int,
        credits_added: int,
    ) -> None:
        metadata = {
            "telegram_payment_charge_id": telegram_payment_charge_id,
            "credits_added": credits_added,
        }
        await gsheets_db.create_payment_record(
            "stars",
            provider_payment_charge_id,
            user_id,
            amount,
            credits_added,
            "succeeded",
            payload=telegram_payment_charge_id,
            metadata=metadata,
        )

    async def create_payment_record(
        self,
        provider: str,
        ext_id: str,
        user_id: int,
        amount_cp: int,
        items: int,
        status: str,
        *,
        payload: str,
        metadata: Optional[Dict[str, Any]] = None,
        idempotency_key: str = "",
        username: Optional[str] = None,
        package_id: Optional[str] = None,
        purchased_credits: Optional[int] = None,
        processed_at: Optional[str] = None,
        record_type: str = "payment",
        ref_payment_id: Optional[str] = None,
    ) -> None:
        metadata_dict = metadata if metadata is not None else parse_metadata(payload)
        username_value = username
        if username_value is None:
            try:
                record = await gsheets_db.get_or_create_user(user_id)
            except Exception:  # pragma: no cover - external dependency
                log.warning(
                    "Failed to resolve username for payment record user=%s", user_id, exc_info=True
                )
            else:
                username_value = record.get("username") or None
        await gsheets_db.create_payment_record(
            provider,
            ext_id,
            user_id,
            amount_cp,
            items,
            status,
            payload=payload,
            metadata=metadata_dict,
            idempotency_key=idempotency_key,
            username=username_value,
            package_id=package_id,
            purchased_credits=purchased_credits,
            processed_at=processed_at,
            record_type=record_type,
            ref_payment_id=ref_payment_id,
        )

    async def update_payment_status_by_ext(
        self,
        provider: str,
        ext_id: str,
        status: str,
        *,
        credits_added: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        processed_at: Optional[str] = None,
        package_id: Optional[str] = None,
        purchased_credits: Optional[int] = None,
        record_type: Optional[str] = None,
        ref_payment_id: Optional[str] = None,
    ) -> None:
        record = await gsheets_db.get_payment_by_ext(provider, ext_id)
        existing_metadata = parse_metadata(record.get("metadata") if record else None)
        if metadata is not None:
            incoming = metadata
            existing_metadata.update(incoming)
        if credits_added is not None:
            existing_metadata["credits_added"] = credits_added
        metadata_str = json.dumps(existing_metadata) if existing_metadata else ""
        await gsheets_db.update_payment_status_by_ext(
            provider,
            ext_id,
            status,
            metadata=metadata_str,
            idempotency_key=idempotency_key,
            processed_at=processed_at,
            package_id=package_id,
            purchased_credits=purchased_credits,
            record_type=record_type,
            ref_payment_id=ref_payment_id,
        )

    async def get_payment_by_ext(self, provider: str, ext_id: str) -> Optional[Dict[str, Any]]:
        return await gsheets_db.get_payment_by_ext(provider, ext_id)

    async def list_payments_by_status(
        self,
        status: str,
        *,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await gsheets_db.list_payments_by_status(
            status,
            created_after=created_after,
            created_before=created_before,
        )

    async def get_payment_by_order_id(self, provider: str, order_id: str) -> Optional[Dict[str, Any]]:
        return await gsheets_db.get_payment_by_order_id(provider, order_id)

    async def list_payments(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        return await gsheets_db.list_payments(provider)

    async def set_user_credits(self, telegram_id: int, value: int) -> int:
        return await gsheets_db.set_user_credits(telegram_id, value)

    async def list_users(self) -> List[Dict[str, Any]]:
        return await gsheets_db.list_users()

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    async def create_job(self, job: GenerationJobRecord) -> None:
        await gsheets_db.create_job(
            job.id,
            job.user_id,
            job.prompt,
            image_file_id=job.image_file_id,
            sora_req_id=job.sora_req_id,
            size=job.size,
            seconds=job.seconds,
            model=job.model,
            cost_credits=job.cost_credits,
            username=job.username,
            corr_id=job.corr_id,
            status=normalise_job_status(job.status),
            idempotency_key=job.idempotency_key,
            content_type=job.content_type,
            metadata=job.extra,
        )

    async def find_job_by_idempotency_key(
        self, idempotency_key: str
    ) -> Optional[GenerationJobRecord]:
        data = await gsheets_db.get_job_by_idempotency_key(idempotency_key)
        if not data:
            return None
        return job_from_mapping(data)

    async def update_job(
        self,
        job_id: str,
        status: str,
        *,
        video_url: Optional[str] = None,
        video_id: Optional[str] = None,
        file_url: Optional[str] = None,
        operation_name: Optional[str] = None,
        error: Optional[str] = None,
        image_file_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        status_message_id: Optional[int] = None,
        status_message_index: Optional[int] = None,
        status_message_updated_at: Optional[datetime] = None,
    ) -> None:
        normalised_status = normalise_job_status(status)
        updates: Dict[str, Any] = {}
        if video_url is not None:
            updates["video_url"] = video_url
        if file_url is not None:
            updates["file_url"] = file_url
        if operation_name is not None:
            updates["operation_name"] = operation_name
        if video_id is not None:
            updates["video_id"] = video_id
        if error is not None:
            updates["error"] = error
        if image_file_id is not None:
            updates["image_file_id"] = image_file_id
        if metadata is not None:
            updates["metadata"] = metadata
        if status_message_id is not None:
            updates["status_message_id"] = status_message_id
        if status_message_index is not None:
            updates["status_message_index"] = status_message_index
        if status_message_updated_at is not None:
            updates["status_message_updated_at"] = status_message_updated_at.isoformat()
        await gsheets_db.update_job_status(job_id, normalised_status, **updates)

    async def update_job_fields(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        status_message_id: Optional[int] = None,
        status_message_index: Optional[int] = None,
        status_message_updated_at: Optional[datetime] = None,
    ) -> None:
        updates: Dict[str, Any] = {}
        if status_message_id is not None:
            updates["status_message_id"] = status_message_id
        if status_message_index is not None:
            updates["status_message_index"] = status_message_index
        if status_message_updated_at is not None:
            updates["status_message_updated_at"] = status_message_updated_at.isoformat()
        normalised_status = normalise_job_status(status) if status is not None else ""
        await gsheets_db.update_job_status(job_id, normalised_status, **updates)

    async def list_pending_jobs(self, *, limit: int) -> List[GenerationJobRecord]:
        jobs = await gsheets_db.list_jobs_by_status(["queued", "running"])
        pending = [job_from_mapping(job) for job in jobs]
        return pending[:limit]

    async def get_job(self, job_id: str) -> Optional[GenerationJobRecord]:
        data = await gsheets_db.get_job(job_id)
        if not data:
            return None
        return job_from_mapping(data)

    async def list_jobs_by_status(self, statuses: Iterable[str]) -> List[Dict[str, Any]]:
        return await gsheets_db.list_jobs_by_status(statuses)

    # ------------------------------------------------------------------
    # Error log
    # ------------------------------------------------------------------

    async def log_error_record(self, record: ErrorLogRecord) -> bool:
        try:
            await gsheets_db.append_error_record(
                ts=record.ts.replace(microsecond=0).isoformat(),
                user_id=record.user_id,
                username=record.username or "",
                corr_id=record.corr_id or "",
                job_id=record.job_id or "",
                model=record.model or "",
                size=record.size or "",
                status_code=record.status_code or "",
                error_type=record.error_type or "",
                error_msg_short=record.error_msg_short,
                refunded=record.refunded,
                preflight_blocked=record.preflight_blocked,
                preflight_reason=record.preflight_reason,
                auto_sanitized=record.auto_sanitized,
                sanitized_prompt=record.sanitized_prompt or "",
                error_scope=record.error_scope or "",
                error_json=record.error_json or "",
                job_status=record.job_status or "",
                reason=record.reason or "",
                provider_error_code=record.provider_error_code or "",
                provider_error_message=record.provider_error_message or "",
                stage=record.stage or "",
            )
        except Exception:  # pragma: no cover - external dependency
            log.warning("Failed to append error record", exc_info=True)
            return False
        return True

    # ------------------------------------------------------------------
    # Archive log
    # ------------------------------------------------------------------

    async def archive_was_sent(self, corr_id: str) -> bool:
        if not corr_id:
            return False
        try:
            return await gsheets_db.archive_was_sent(corr_id)
        except Exception:  # pragma: no cover - external dependency
            log.warning("Failed to fetch archive status corr_id=%s", corr_id, exc_info=True)
            return False

    async def log_archive_record(self, record: ArchiveLogRecord) -> None:
        try:
            await gsheets_db.append_archive_log(
                ts=record.ts.replace(microsecond=0).isoformat(),
                corr_id=record.corr_id,
                archive_status=record.archive_status,
                channel_id=record.channel_id or "",
                message_id=record.message_id or "",
                content_type=record.content_type,
                model_name=record.model_name or "",
                username=record.username or "",
                user_id=record.user_id or "",
                caption_len=record.caption_len,
                file_size=record.file_size,
                duration_seconds=record.duration_seconds or "",
                attempts=record.attempts,
                error_short=record.error_short or "",
            )
        except Exception:  # pragma: no cover - external dependency
            log.warning("Failed to append archive log corr_id=%s", record.corr_id, exc_info=True)


__all__ = ["SheetsDatabase"]
