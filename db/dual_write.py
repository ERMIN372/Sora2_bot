from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .interface import DatabaseInterface
from .models import ArchiveLogRecord, ErrorLogRecord, GenerationJobRecord, User

log = logging.getLogger(__name__)


class DualWriteDatabase(DatabaseInterface):
    """Fan-out database that reads from primary and mirrors writes to secondary."""

    def __init__(self, primary: DatabaseInterface, secondary: DatabaseInterface) -> None:
        self._primary = primary
        self._secondary = secondary

    async def init(self) -> None:
        await self._primary.init()
        await self._secondary.init()

    async def close(self) -> None:
        await self._primary.close()
        await self._secondary.close()

    # Users
    async def ensure_user(
        self,
        telegram_id: int,
        username: Optional[str],
        *,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> User:
        result = await self._primary.ensure_user(
            telegram_id,
            username,
            first_name=first_name,
            last_name=last_name,
        )
        await self._mirror(
            self._secondary.ensure_user,
            telegram_id,
            username,
            first_name=first_name,
            last_name=last_name,
        )
        return result

    async def sync_user_profile(
        self,
        telegram_id: int,
        *,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> None:
        await self._primary.sync_user_profile(
            telegram_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
        )
        await self._mirror(
            self._secondary.sync_user_profile,
            telegram_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
        )

    async def get_user_credits(self, telegram_id: int) -> int:
        return await self._primary.get_user_credits(telegram_id)

    async def add_credits(self, telegram_id: int, amount: int) -> int:
        value = await self._primary.add_credits(telegram_id, amount)
        await self._mirror(self._secondary.add_credits, telegram_id, amount)
        return value

    async def deduct_credit(self, telegram_id: int, amount: int = 1) -> bool:
        ok = await self._primary.deduct_credit(telegram_id, amount)
        if ok:
            await self._mirror(self._secondary.deduct_credit, telegram_id, amount)
        return ok

    async def is_bonus_granted(self, telegram_id: int) -> bool:
        return await self._primary.is_bonus_granted(telegram_id)

    async def mark_bonus_granted(self, telegram_id: int) -> None:
        await self._primary.mark_bonus_granted(telegram_id)
        await self._mirror(self._secondary.mark_bonus_granted, telegram_id)

    async def mark_economy_v2(self, telegram_id: int) -> None:
        await self._primary.mark_economy_v2(telegram_id)
        await self._mirror(self._secondary.mark_economy_v2, telegram_id)

    async def count_active_jobs(self, user_id: int) -> int:
        return await self._primary.count_active_jobs(user_id)

    async def migrate_credit_balances(self, multiplier: int) -> int:
        migrated = await self._primary.migrate_credit_balances(multiplier)
        await self._mirror(self._secondary.migrate_credit_balances, multiplier)
        return migrated

    async def backfill_user_profiles(
        self, fetcher: Callable[[int], Awaitable[Optional[Dict[str, Optional[str]]]]]
    ) -> None:
        await self._primary.backfill_user_profiles(fetcher)
        await self._mirror(self._secondary.backfill_user_profiles, fetcher)

    # Payments
    async def record_payment(
        self,
        *,
        user_id: int,
        provider_payment_charge_id: str,
        telegram_payment_charge_id: str,
        amount: int,
        credits_added: int,
    ) -> None:
        await self._primary.record_payment(
            user_id=user_id,
            provider_payment_charge_id=provider_payment_charge_id,
            telegram_payment_charge_id=telegram_payment_charge_id,
            amount=amount,
            credits_added=credits_added,
        )
        await self._mirror(
            self._secondary.record_payment,
            user_id=user_id,
            provider_payment_charge_id=provider_payment_charge_id,
            telegram_payment_charge_id=telegram_payment_charge_id,
            amount=amount,
            credits_added=credits_added,
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
        await self._primary.create_payment_record(
            provider,
            ext_id,
            user_id,
            amount_cp,
            items,
            status,
            payload=payload,
            metadata=metadata,
            idempotency_key=idempotency_key,
            username=username,
            package_id=package_id,
            purchased_credits=purchased_credits,
            processed_at=processed_at,
            record_type=record_type,
            ref_payment_id=ref_payment_id,
        )
        await self._mirror(
            self._secondary.create_payment_record,
            provider,
            ext_id,
            user_id,
            amount_cp,
            items,
            status,
            payload=payload,
            metadata=metadata,
            idempotency_key=idempotency_key,
            username=username,
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
        await self._primary.update_payment_status_by_ext(
            provider,
            ext_id,
            status,
            credits_added=credits_added,
            metadata=metadata,
            idempotency_key=idempotency_key,
            processed_at=processed_at,
            package_id=package_id,
            purchased_credits=purchased_credits,
            record_type=record_type,
            ref_payment_id=ref_payment_id,
        )
        await self._mirror(
            self._secondary.update_payment_status_by_ext,
            provider,
            ext_id,
            status,
            credits_added=credits_added,
            metadata=metadata,
            idempotency_key=idempotency_key,
            processed_at=processed_at,
            package_id=package_id,
            purchased_credits=purchased_credits,
            record_type=record_type,
            ref_payment_id=ref_payment_id,
        )

    async def get_payment_by_ext(self, provider: str, ext_id: str) -> Optional[Dict[str, Any]]:
        return await self._primary.get_payment_by_ext(provider, ext_id)

    async def list_payments_by_status(
        self,
        status: str,
        *,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await self._primary.list_payments_by_status(
            status, created_after=created_after, created_before=created_before
        )

    async def get_payment_by_order_id(
        self, provider: str, order_id: str
    ) -> Optional[Dict[str, Any]]:
        return await self._primary.get_payment_by_order_id(provider, order_id)

    async def list_payments(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        return await self._primary.list_payments(provider)

    async def set_user_credits(self, telegram_id: int, value: int) -> int:
        credits = await self._primary.set_user_credits(telegram_id, value)
        await self._mirror(self._secondary.set_user_credits, telegram_id, value)
        return credits

    async def list_users(self) -> List[Dict[str, Any]]:
        return await self._primary.list_users()

    # Jobs
    async def create_job(self, job: GenerationJobRecord) -> None:
        await self._primary.create_job(job)
        await self._mirror(self._secondary.create_job, job)

    async def find_job_by_idempotency_key(
        self, idempotency_key: str
    ) -> Optional[GenerationJobRecord]:
        return await self._primary.find_job_by_idempotency_key(idempotency_key)

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
        status_message_id: Optional[int] = None,
        status_message_index: Optional[int] = None,
        status_message_updated_at: Optional[datetime] = None,
    ) -> None:
        await self._primary.update_job(
            job_id,
            status,
            video_url=video_url,
            video_id=video_id,
            file_url=file_url,
            operation_name=operation_name,
            error=error,
            status_message_id=status_message_id,
            status_message_index=status_message_index,
            status_message_updated_at=status_message_updated_at,
        )
        await self._mirror(
            self._secondary.update_job,
            job_id,
            status,
            video_url=video_url,
            video_id=video_id,
            file_url=file_url,
            operation_name=operation_name,
            error=error,
            status_message_id=status_message_id,
            status_message_index=status_message_index,
            status_message_updated_at=status_message_updated_at,
        )

    async def update_job_fields(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        status_message_id: Optional[int] = None,
        status_message_index: Optional[int] = None,
        status_message_updated_at: Optional[datetime] = None,
    ) -> None:
        await self._primary.update_job_fields(
            job_id,
            status=status,
            status_message_id=status_message_id,
            status_message_index=status_message_index,
            status_message_updated_at=status_message_updated_at,
        )
        await self._mirror(
            self._secondary.update_job_fields,
            job_id,
            status=status,
            status_message_id=status_message_id,
            status_message_index=status_message_index,
            status_message_updated_at=status_message_updated_at,
        )

    async def list_pending_jobs(self, *, limit: int) -> List[GenerationJobRecord]:
        return await self._primary.list_pending_jobs(limit=limit)

    async def get_job(self, job_id: str) -> Optional[GenerationJobRecord]:
        return await self._primary.get_job(job_id)

    # Error log
    async def log_error_record(self, record: ErrorLogRecord) -> bool:
        primary_ok = await self._primary.log_error_record(record)
        await self._mirror(self._secondary.log_error_record, record)
        return primary_ok

    # Archive log
    async def archive_was_sent(self, corr_id: str) -> bool:
        return await self._primary.archive_was_sent(corr_id)

    async def log_archive_record(self, record: ArchiveLogRecord) -> None:
        await self._primary.log_archive_record(record)
        await self._mirror(self._secondary.log_archive_record, record)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _mirror(self, func: Any, *args: Any, **kwargs: Any) -> None:
        try:
            await func(*args, **kwargs)
        except Exception:
            log.warning(
                "Dual-write secondary call failed for %s", getattr(func, "__name__", func), exc_info=True
            )


__all__ = ["DualWriteDatabase"]
