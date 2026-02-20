from __future__ import annotations

import abc
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .models import ArchiveLogRecord, ErrorLogRecord, GenerationJobRecord, User


class DatabaseInterface(abc.ABC):
    """Async interface for database backends."""

    async def init(self) -> None:  # pragma: no cover - interface
        return None

    async def close(self) -> None:  # pragma: no cover - interface
        return None

    async def healthcheck(self) -> bool:  # pragma: no cover - interface
        """Return True if the database is reachable."""
        return True

    # Users
    @abc.abstractmethod
    async def ensure_user(
        self,
        telegram_id: int,
        username: Optional[str],
        *,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> User:
        raise NotImplementedError

    @abc.abstractmethod
    async def sync_user_profile(
        self,
        telegram_id: int,
        *,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_user_credits(self, telegram_id: int) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    async def add_credits(self, telegram_id: int, amount: int) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    async def grant_bonus_if_needed(self, telegram_id: int, bonus: int) -> Optional[int]:
        """Atomically grant *bonus* credits once per user, returning the new balance if applied."""
        raise NotImplementedError

    @abc.abstractmethod
    async def deduct_credit(self, telegram_id: int, amount: int = 1) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    async def is_bonus_granted(self, telegram_id: int) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    async def mark_bonus_granted(self, telegram_id: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def mark_economy_v2(self, telegram_id: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def count_active_jobs(self, user_id: int) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    async def migrate_credit_balances(self, multiplier: int) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    async def backfill_user_profiles(
        self, fetcher: Callable[[int], Awaitable[Optional[Dict[str, Optional[str]]]]]
    ) -> None:
        raise NotImplementedError

    # Payments
    @abc.abstractmethod
    async def record_payment(
        self,
        *,
        user_id: int,
        provider_payment_charge_id: str,
        telegram_payment_charge_id: str,
        amount: int,
        credits_added: int,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    async def get_payment_by_ext(self, provider: str, ext_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    async def list_payments_by_status(
        self,
        statuses: str | Iterable[str],
        *,
        provider: Optional[str] = None,
        created_after: Optional[str | datetime] = None,
        created_before: Optional[str | datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_payment_by_order_id(
        self, provider: str, order_id: str
    ) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    async def list_payments(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    async def set_user_credits(self, telegram_id: int, value: int) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    async def list_users(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    async def describe(self) -> Optional[Dict[str, object]]:
        """Return optional connection diagnostics (e.g., db name, host, schema)."""
        raise NotImplementedError

    # Jobs
    @abc.abstractmethod
    async def create_job(self, job: GenerationJobRecord) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def find_job_by_idempotency_key(
        self, idempotency_key: str
    ) -> Optional[GenerationJobRecord]:
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    async def update_job_fields(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        status_message_id: Optional[int] = None,
        status_message_index: Optional[int] = None,
        status_message_updated_at: Optional[datetime] = None,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def list_pending_jobs(self, *, limit: int) -> List[GenerationJobRecord]:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_job(self, job_id: str) -> Optional[GenerationJobRecord]:
        raise NotImplementedError

    # Error log
    @abc.abstractmethod
    async def log_error_record(self, record: ErrorLogRecord) -> bool:
        raise NotImplementedError

    # Archive log
    @abc.abstractmethod
    async def archive_was_sent(self, corr_id: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    async def log_archive_record(self, record: ArchiveLogRecord) -> None:
        raise NotImplementedError

    # Referrals (RevShare)
    @abc.abstractmethod
    async def create_referral(self, referrer_id: int, referred_id: int) -> bool:
        """Link referred_id to referrer_id permanently. Return True if created."""
        raise NotImplementedError

    @abc.abstractmethod
    async def get_referrer_id(self, referred_id: int) -> Optional[int]:
        """Return the referrer for *referred_id*, or None."""
        raise NotImplementedError

    @abc.abstractmethod
    async def record_referral_payout(
        self,
        *,
        referrer_id: int,
        payer_id: int,
        payment_ext_id: str,
        topup_amount_cp: int,
        payout_credits: int,
        payout_pct: float,
    ) -> None:
        """Log a RevShare payout in the audit table."""
        raise NotImplementedError

    @abc.abstractmethod
    async def get_referral_stats(self, referrer_id: int) -> Dict[str, Any]:
        """Return aggregate referral stats for the referrer."""
        raise NotImplementedError


__all__ = ["DatabaseInterface"]
