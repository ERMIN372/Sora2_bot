from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

import asyncpg

from .interface import DatabaseInterface
from .models import (
    ArchiveLogRecord,
    ErrorLogRecord,
    GenerationJobRecord,
    User,
    derive_payment_breakdown,
    job_from_mapping,
    normalise_job_status,
    parse_metadata,
    user_from_mapping,
)

log = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PostgresDatabase(DatabaseInterface):
    """Asyncpg-based PostgreSQL implementation."""

    def __init__(
        self,
        *,
        dsn: str,
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size

    async def init(self) -> None:
        if self._pool:
            return
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._min_pool_size,
            max_size=self._max_pool_size,
        )
        async with self._pool.acquire() as conn:
            await conn.execute("SELECT 1")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _require_pool(self) -> asyncpg.Pool:
        if not self._pool:
            raise RuntimeError("Postgres pool is not initialised; call init() first")
        return self._pool

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
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO users (user_id, username, first_name, last_name, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $5)
                ON CONFLICT (user_id) DO UPDATE
                SET username = COALESCE(EXCLUDED.username, users.username),
                    first_name = COALESCE(EXCLUDED.first_name, users.first_name),
                    last_name = COALESCE(EXCLUDED.last_name, users.last_name),
                    updated_at = NOW()
                RETURNING *
                """,
                telegram_id,
                username,
                first_name,
                last_name,
                _now(),
            )
        return user_from_mapping(dict(row))

    async def sync_user_profile(
        self,
        telegram_id: int,
        *,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> None:
        pool = self._require_pool()
        updates = {
            "username": username,
            "first_name": first_name,
            "last_name": last_name,
        }
        filtered = {key: value for key, value in updates.items() if value is not None}
        if not filtered:
            return
        assignments = ", ".join(f"{key} = ${idx}" for idx, key in enumerate(filtered, start=2))
        values = list(filtered.values())
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"UPDATE users SET {assignments}, updated_at = NOW() WHERE user_id = $1",
                telegram_id,
                *values,
            )

    async def get_user_credits(self, telegram_id: int) -> int:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            value = await conn.fetchval(
                "SELECT credits FROM users WHERE user_id = $1", telegram_id
            )
        return int(value or 0)

    async def add_credits(self, telegram_id: int, amount: int) -> int:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            value = await conn.fetchval(
                """
                INSERT INTO users (user_id, credits, created_at, updated_at)
                VALUES ($1, $2, $3, $3)
                ON CONFLICT (user_id) DO UPDATE
                SET credits = users.credits + EXCLUDED.credits,
                    updated_at = NOW()
                RETURNING credits
                """,
                telegram_id,
                amount,
                _now(),
            )
        return int(value or 0)

    async def deduct_credit(self, telegram_id: int, amount: int = 1) -> bool:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            value = await conn.fetchval(
                """
                UPDATE users
                SET credits = credits - $2, updated_at = NOW()
                WHERE user_id = $1 AND credits >= $2
                RETURNING credits
                """,
                telegram_id,
                amount,
            )
        return value is not None

    async def is_bonus_granted(self, telegram_id: int) -> bool:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            value = await conn.fetchval(
                "SELECT bonus_granted FROM users WHERE user_id = $1", telegram_id
            )
        return bool(value)

    async def mark_bonus_granted(self, telegram_id: int) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE users
                SET bonus_granted = TRUE, updated_at = NOW()
                WHERE user_id = $1
                """,
                telegram_id,
            )

    async def mark_economy_v2(self, telegram_id: int) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE users
                SET economy_v2 = TRUE, updated_at = NOW()
                WHERE user_id = $1
                """,
                telegram_id,
            )

    async def count_active_jobs(self, user_id: int) -> int:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            value = await conn.fetchval(
                """
                SELECT COUNT(*) FROM jobs
                WHERE user_id = $1 AND status IN ('queued', 'running')
                """,
                user_id,
            )
        return int(value or 0)

    async def migrate_credit_balances(self, multiplier: int) -> int:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM users WHERE credits > 0 AND economy_v2 = FALSE"
            )
            await conn.execute(
                """
                UPDATE users
                SET credits = credits * $1,
                    economy_v2 = TRUE,
                    updated_at = NOW()
                WHERE credits > 0 AND economy_v2 = FALSE
                """,
                multiplier,
            )
        migrated = int(count or 0)
        if migrated:
            log.info("Credit economy migration applied to %s users", migrated)
        return migrated

    async def backfill_user_profiles(
        self, fetcher: Callable[[int], Awaitable[Optional[Dict[str, Optional[str]]]]]
    ) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT user_id FROM users
                WHERE (username IS NULL OR username = '')
                   OR (first_name IS NULL OR first_name = '')
                   OR (last_name IS NULL OR last_name = '')
                LIMIT 5000
                """
            )
        user_ids = [int(row[0]) for row in rows]
        for user_id in user_ids:
            try:
                profile = await fetcher(user_id)
            except Exception:  # pragma: no cover - Telegram API interaction
                log.debug("Failed to fetch profile for user_id=%s", user_id, exc_info=True)
                continue
            if not profile:
                continue
            await self.sync_user_profile(
                user_id,
                username=profile.get("username"),
                first_name=profile.get("first_name"),
                last_name=profile.get("last_name"),
            )

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
        await self.create_payment_record(
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
        pool = self._require_pool()
        metadata_payload = metadata if metadata is not None else parse_metadata(payload)
        price_rub, credits_bought, bonus_credits, bonus_pct = derive_payment_breakdown(
            amount_cp, purchased_credits
        )
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO payments (
                    provider, ext_id, user_id, amount_cp, items, price_rub,
                    credits_bought, bonus_credits, bonus_pct, status,
                    payload, metadata, created_at, updated_at, idempotency_key,
                    username, package_id, purchased_credits, processed_at, type, ref_payment_id
                )
                VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7, $8, $9, $10,
                    $11, $12, $13, $13, $14,
                    $15, $16, $17, $18, $19, $20
                )
                ON CONFLICT (provider, ext_id) DO NOTHING
                """,
                provider,
                ext_id,
                user_id,
                amount_cp,
                items,
                price_rub,
                credits_bought,
                bonus_credits,
                float(bonus_pct),
                status,
                payload,
                json.dumps(metadata_payload) if metadata_payload else None,
                _now(),
                idempotency_key or None,
                username,
                package_id,
                purchased_credits,
                processed_at,
                record_type,
                ref_payment_id,
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
        pool = self._require_pool()
        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT metadata FROM payments WHERE provider = $1 AND ext_id = $2",
                provider,
                ext_id,
            )
            existing_metadata = parse_metadata(existing[0] if existing else None)
            if metadata is not None:
                existing_metadata.update(metadata)
            if credits_added is not None:
                existing_metadata["credits_added"] = credits_added
            await conn.execute(
                """
                UPDATE payments
                SET status = $3,
                    metadata = $4,
                    updated_at = NOW(),
                    idempotency_key = COALESCE($5, idempotency_key),
                    processed_at = COALESCE($6, processed_at),
                    package_id = COALESCE($7, package_id),
                    purchased_credits = COALESCE($8, purchased_credits),
                    type = COALESCE($9, type),
                    ref_payment_id = COALESCE($10, ref_payment_id)
                WHERE provider = $1 AND ext_id = $2
                """,
                provider,
                ext_id,
                status,
                json.dumps(existing_metadata) if existing_metadata else None,
                idempotency_key,
                processed_at,
                package_id,
                purchased_credits,
                record_type,
                ref_payment_id,
            )

    async def get_payment_by_ext(self, provider: str, ext_id: str) -> Optional[Dict[str, Any]]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM payments WHERE provider = $1 AND ext_id = $2",
                provider,
                ext_id,
            )
        return dict(row) if row else None

    async def list_payments_by_status(
        self,
        statuses: str | Iterable[str],
        *,
        provider: Optional[str] = None,
        created_after: Optional[str | datetime] = None,
        created_before: Optional[str | datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        pool = self._require_pool()
        if isinstance(statuses, str):
            status_values = [statuses]
        else:
            status_values = list(statuses)
        if not status_values:
            return []
        params: List[Any] = []
        conditions: List[str] = []

        params.append(status_values)
        conditions.append("status = ANY($%d)" % len(params))
        if provider:
            params.append(provider)
            conditions.append("provider = $%d" % len(params))
        if created_after:
            params.append(created_after)
            conditions.append("created_at >= $%d" % len(params))
        if created_before:
            params.append(created_before)
            conditions.append("created_at <= $%d" % len(params))

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        query = f"SELECT * FROM payments WHERE {where_clause} ORDER BY created_at DESC"
        if limit:
            params.append(limit)
            query += " LIMIT $%d" % len(params)
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return [dict(row) for row in rows]

    async def get_payment_by_order_id(
        self, provider: str, order_id: str
    ) -> Optional[Dict[str, Any]]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM payments WHERE provider = $1 AND ext_id = $2",
                provider,
                order_id,
            )
        return dict(row) if row else None

    async def list_payments(self, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        pool = self._require_pool()
        if provider:
            query = "SELECT * FROM payments WHERE provider = $1 ORDER BY created_at DESC"
            params = [provider]
        else:
            query = "SELECT * FROM payments ORDER BY created_at DESC"
            params = []
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
        return [dict(row) for row in rows]

    async def set_user_credits(self, telegram_id: int, value: int) -> int:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            credits = await conn.fetchval(
                """
                UPDATE users SET credits = $2, updated_at = NOW()
                WHERE user_id = $1 RETURNING credits
                """,
                telegram_id,
                value,
            )
        return int(credits or 0)

    async def list_users(self) -> List[Dict[str, Any]]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM users ORDER BY created_at ASC")
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    async def create_job(self, job: GenerationJobRecord) -> None:
        pool = self._require_pool()
        normalised_status = normalise_job_status(job.status)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO jobs (
                    job_id, user_id, prompt, status, video_url, video_id, error,
                    created_at, updated_at, image_file_id, sora_req_id, size,
                    seconds, model, cost_credits, username, corr_id,
                    idempotency_key, content_type, metadata,
                    status_message_id, status_message_index, status_message_updated_at,
                    operation_name, file_url
                )
                VALUES (
                    $1, $2, $3, $4, $5, $6, $7,
                    $8, $8, $9, $10, $11,
                    $12, $13, $14, $15, $16,
                    $17, $18, $19,
                    $20, $21, $22,
                    $23, $24
                )
                ON CONFLICT (job_id) DO NOTHING
                """,
                job.id,
                job.user_id,
                job.prompt,
                normalised_status,
                job.video_url,
                job.video_id,
                job.error,
                _now(),
                job.image_file_id,
                job.sora_req_id,
                job.size,
                job.seconds,
                job.model,
                job.cost_credits,
                job.username,
                job.corr_id,
                job.idempotency_key,
                job.content_type,
                json.dumps(job.extra) if job.extra else None,
                job.status_message_id,
                job.status_message_index,
                job.status_message_updated_at,
                job.operation_name,
                job.file_url,
            )

    async def find_job_by_idempotency_key(
        self, idempotency_key: str
    ) -> Optional[GenerationJobRecord]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM jobs WHERE idempotency_key = $1",
                idempotency_key,
            )
        return job_from_mapping(dict(row)) if row else None

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
        pool = self._require_pool()
        normalised_status = normalise_job_status(status)
        updates: Dict[str, Any] = {
            "status": normalised_status,
            "video_url": video_url,
            "video_id": video_id,
            "file_url": file_url,
            "operation_name": operation_name,
            "error": error,
            "image_file_id": image_file_id,
            "metadata": json.dumps(metadata) if metadata is not None else None,
            "status_message_id": status_message_id,
            "status_message_index": status_message_index,
            "status_message_updated_at": status_message_updated_at,
        }
        assignments = []
        values: List[Any] = []
        for key, value in updates.items():
            if value is None and key not in {"status_message_index", "status_message_updated_at"}:
                continue
            assignments.append(f"{key} = ${len(values) + 2}")
            values.append(value)
        assignments.append(f"updated_at = ${len(values) + 2}")
        values.append(_now())
        set_clause = ", ".join(assignments)
        async with pool.acquire() as conn:
            await conn.execute(
                f"UPDATE jobs SET {set_clause} WHERE job_id = $1",
                job_id,
                *values,
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
        pool = self._require_pool()
        updates: Dict[str, Any] = {}
        if status is not None:
            updates["status"] = normalise_job_status(status)
        if status_message_id is not None:
            updates["status_message_id"] = status_message_id
        if status_message_index is not None:
            updates["status_message_index"] = status_message_index
        if status_message_updated_at is not None:
            updates["status_message_updated_at"] = status_message_updated_at
        if not updates:
            return
        assignments = [f"{key} = ${idx}" for idx, key in enumerate(updates, start=2)]
        values = list(updates.values())
        assignments.append(f"updated_at = ${len(values) + 2}")
        values.append(_now())
        set_clause = ", ".join(assignments)
        async with pool.acquire() as conn:
            await conn.execute(
                f"UPDATE jobs SET {set_clause} WHERE job_id = $1",
                job_id,
                *values,
            )

    async def list_pending_jobs(self, *, limit: int) -> List[GenerationJobRecord]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM jobs
                WHERE status IN ('queued', 'running')
                ORDER BY created_at ASC
                LIMIT $1
                """,
                limit,
            )
        return [job_from_mapping(dict(row)) for row in rows]

    async def get_job(self, job_id: str) -> Optional[GenerationJobRecord]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM jobs WHERE job_id = $1", job_id)
        return job_from_mapping(dict(row)) if row else None

    # ------------------------------------------------------------------
    # Error log
    # ------------------------------------------------------------------

    async def log_error_record(self, record: ErrorLogRecord) -> bool:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO error_logs (
                        ts, user_id, username, corr_id, job_id, model, size,
                        status_code, error_type, error_msg_short, refunded,
                        preflight_blocked, preflight_reason, auto_sanitized,
                        sanitized_prompt, error_scope, error_json, job_status,
                        reason, provider_error_code, provider_error_message, stage
                    )
                    VALUES (
                        $1, $2, $3, $4, $5, $6, $7,
                        $8, $9, $10, $11,
                        $12, $13, $14,
                        $15, $16, $17, $18,
                        $19, $20, $21, $22
                    )
                    """,
                    record.ts.replace(microsecond=0),
                    record.user_id,
                    record.username,
                    record.corr_id,
                    record.job_id,
                    record.model,
                    record.size,
                    record.status_code,
                    record.error_type,
                    record.error_msg_short,
                    record.refunded,
                    record.preflight_blocked,
                    record.preflight_reason,
                    record.auto_sanitized,
                    record.sanitized_prompt,
                    record.error_scope,
                    record.error_json,
                    record.job_status,
                    record.reason,
                    record.provider_error_code,
                    record.provider_error_message,
                    record.stage,
                )
            except Exception:
                log.warning("Failed to persist error log", exc_info=True)
                return False
        return True

    # ------------------------------------------------------------------
    # Archive log
    # ------------------------------------------------------------------

    async def archive_was_sent(self, corr_id: str) -> bool:
        if not corr_id:
            return False
        pool = self._require_pool()
        async with pool.acquire() as conn:
            status = await conn.fetchval(
                """
                SELECT archive_status FROM archive_logs
                WHERE corr_id = $1
                ORDER BY ts DESC
                LIMIT 1
                """,
                corr_id,
            )
        return bool(status and status.lower() == "sent")

    async def log_archive_record(self, record: ArchiveLogRecord) -> None:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO archive_logs (
                    ts, corr_id, archive_status, channel_id, message_id,
                    content_type, model_name, username, user_id, caption_len,
                    file_size, duration_seconds, attempts, error_short
                )
                VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, $9, $10,
                    $11, $12, $13, $14
                )
                """,
                record.ts.replace(microsecond=0),
                record.corr_id,
                record.archive_status,
                record.channel_id,
                record.message_id,
                record.content_type,
                record.model_name,
                record.username,
                record.user_id,
                record.caption_len,
                record.file_size,
                record.duration_seconds,
                record.attempts,
                record.error_short,
            )


__all__ = ["PostgresDatabase"]
