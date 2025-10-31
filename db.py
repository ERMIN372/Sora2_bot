"""Database helpers implemented on top of Google Sheets."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

import gsheets_db


log = logging.getLogger(__name__)


@dataclass
class User:
    telegram_id: int
    credits: int
    bonus_granted: bool
    created_at: datetime
    updated_at: datetime
    notes: str = ""
    economy_v2: bool = False


@dataclass
class GenerationJobRecord:
    id: str
    user_id: int
    prompt: str
    status: str
    video_url: Optional[str]
    video_id: Optional[str]
    error: Optional[str]
    created_at: datetime
    updated_at: datetime
    image_file_id: Optional[str] = None
    sora_req_id: Optional[str] = None
    size: Optional[str] = None
    seconds: Optional[int] = None
    model: Optional[str] = None
    cost_credits: Optional[int] = None
    username: Optional[str] = None
    corr_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    content_type: str = "video"
    status_message_id: Optional[int] = None
    status_message_index: int = 0
    status_message_updated_at: Optional[datetime] = None
    operation_name: Optional[str] = None
    file_url: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorLogRecord:
    ts: datetime
    user_id: int
    username: Optional[str]
    corr_id: Optional[str]
    job_id: Optional[str]
    model: Optional[str]
    size: Optional[str]
    status_code: Optional[int]
    error_type: Optional[str]
    error_msg_short: str
    refunded: bool
    preflight_blocked: bool = False
    preflight_reason: str = ""
    auto_sanitized: bool = False
    sanitized_prompt: Optional[str] = None
    error_scope: Optional[str] = None
    error_json: Optional[str] = None
    job_status: Optional[str] = None
    reason: Optional[str] = None
    provider_error_code: Optional[str] = None
    provider_error_message: Optional[str] = None
    stage: str = "submit"


@dataclass
class ArchiveLogRecord:
    ts: datetime
    corr_id: str
    archive_status: str
    channel_id: Optional[int]
    message_id: Optional[int]
    content_type: str
    model_name: Optional[str]
    username: Optional[str]
    caption_len: int
    file_size: int
    attempts: int
    error_short: Optional[str]
    user_id: Optional[int] = None
    duration_seconds: Optional[int] = None


def _parse_datetime(value: str) -> datetime:
    if not value:
        return datetime.utcfromtimestamp(0)
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.utcfromtimestamp(0)


def _user_from_dict(data: Dict[str, Any]) -> User:
    return User(
        telegram_id=int(data.get("user_id", 0)),
        credits=int(data.get("credits", 0)),
        bonus_granted=bool(data.get("bonus_granted")),
        created_at=_parse_datetime(data.get("created_at", "")),
        updated_at=_parse_datetime(data.get("updated_at", "")),
        notes=str(data.get("notes", "")) if data.get("notes") is not None else "",
        economy_v2=bool(int(data.get("economy_v2", 0) or 0)),
    )


def _job_from_dict(data: Dict[str, Any]) -> GenerationJobRecord:
    video_url = data.get("video_url") or None
    file_url = data.get("file_url") or None
    error = data.get("error") or None
    image_file_id = data.get("image_file_id") or None
    sora_req_id = data.get("sora_req_id") or None
    size = data.get("size") or None
    seconds_value = data.get("seconds")
    try:
        seconds = int(seconds_value) if seconds_value not in (None, "") else None
    except (TypeError, ValueError):
        seconds = None
    model = data.get("model") or None
    cost_raw = data.get("cost_credits")
    try:
        cost_credits = int(cost_raw) if cost_raw not in (None, "") else None
    except (TypeError, ValueError):
        cost_credits = None
    idempotency_key = data.get("idempotency_key") or None
    status_message_id_raw = data.get("status_message_id")
    try:
        status_message_id = (
            int(status_message_id_raw)
            if status_message_id_raw not in (None, "")
            else None
        )
    except (TypeError, ValueError):
        status_message_id = None
    status_message_index_raw = data.get("status_message_index")
    try:
        status_message_index = (
            int(status_message_index_raw)
            if status_message_index_raw not in (None, "")
            else 0
        )
    except (TypeError, ValueError):
        status_message_index = 0
    status_message_updated_raw = data.get("status_message_updated_at") or ""
    status_message_updated_at = _parse_datetime(status_message_updated_raw)
    content_type = str(data.get("content_type") or "video")
    operation_name = data.get("operation_name") or None
    return GenerationJobRecord(
        id=str(data.get("job_id", "")),
        user_id=int(data.get("user_id", 0)),
        prompt=str(data.get("prompt", "")),
        status=str(data.get("status", "")),
        video_url=video_url,
        video_id=data.get("video_id") or None,
        error=error,
        created_at=_parse_datetime(data.get("created_at", "")),
        updated_at=_parse_datetime(data.get("updated_at", "")),
        image_file_id=image_file_id,
        sora_req_id=sora_req_id,
        size=size,
        seconds=seconds,
        model=model,
        cost_credits=cost_credits,
        username=str(data.get("username", "")) or None,
        corr_id=str(data.get("corr_id", "")) or None,
        idempotency_key=idempotency_key,
        content_type=content_type or "video",
        status_message_id=status_message_id,
        status_message_index=status_message_index,
        status_message_updated_at=status_message_updated_at,
        operation_name=operation_name,
        file_url=file_url,
        extra={},
    )


def _normalise_job_status(status: str) -> str:
    status_lower = (status or "").lower()
    if status_lower in {"queued", "running", "completed", "failed"}:
        return status_lower
    if status_lower in {"processing", "pending", "in_progress"}:
        return "running"
    if status_lower in {"errored", "error"}:
        return "failed"
    return status_lower or "queued"


def _parse_metadata(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {"raw": value}


class Database:
    """Thin proxy to the Google Sheets data access layer."""

    def __init__(self, *_: Any, **__: Any) -> None:
        """Signature kept for compatibility; configuration is read from env."""

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
        return _user_from_dict(data)

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
        self,
        fetcher: Callable[[int], Awaitable[Optional[Dict[str, Optional[str]]]]],
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
        metadata_dict = metadata if metadata is not None else _parse_metadata(payload)
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
        existing_metadata = _parse_metadata(record.get("metadata") if record else None)
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
        *,
        provider: str,
        statuses: Iterable[str],
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        return await gsheets_db.list_payments_by_status(provider=provider, statuses=statuses, limit=limit)

    async def get_payment_by_order_id(self, provider: str, order_id: str) -> Optional[Dict[str, Any]]:
        return await gsheets_db.get_payment_by_order_id(provider, order_id)

    async def list_payments(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        return await gsheets_db.list_payments(provider=provider)

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
            job.image_file_id,
            job.size or "",
            job.seconds or 0,
            job.model or "",
            job.cost_credits or 0,
            job.corr_id,
            username=job.username,
            content_type=job.content_type,
            idempotency_key=job.idempotency_key or "",
            operation_name=job.operation_name,
            file_url=job.file_url,
            video_id=job.video_id,
        )

    async def find_job_by_idempotency_key(
        self, idempotency_key: str
    ) -> Optional[GenerationJobRecord]:
        if not idempotency_key:
            return None
        record = await gsheets_db.get_job_by_idempotency_key(idempotency_key)
        if not record:
            return None
        return _job_from_dict(record)

    async def update_job(
        self,
        job_id: str,
        status: str,
        *,
        video_url: Optional[str] = None,
        file_url: Optional[str] = None,
        operation_name: Optional[str] = None,
        video_id: Optional[str] = None,
        error: Optional[str] = None,
        status_message_id: Optional[int] = None,
        status_message_index: Optional[int] = None,
        status_message_updated_at: Optional[datetime] = None,
    ) -> None:
        normalised_status = _normalise_job_status(status)
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
        normalised_status = _normalise_job_status(status) if status is not None else ""
        await gsheets_db.update_job_status(job_id, normalised_status, **updates)

    async def list_pending_jobs(self, *, limit: int) -> List[GenerationJobRecord]:
        jobs = await gsheets_db.list_jobs_by_status(["queued", "running"])
        pending = [_job_from_dict(job) for job in jobs]
        return pending[:limit]

    async def get_job(self, job_id: str) -> Optional[GenerationJobRecord]:
        data = await gsheets_db.get_job(job_id)
        if not data:
            return None
        return _job_from_dict(data)

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


__all__ = [
    "Database",
    "User",
    "GenerationJobRecord",
    "ErrorLogRecord",
    "ArchiveLogRecord",
]
