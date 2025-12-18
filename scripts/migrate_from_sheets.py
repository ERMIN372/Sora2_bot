"""Migrate data from the legacy Google Sheets backend into PostgreSQL."""

import asyncio
import logging
from typing import Dict, Iterable, Optional

from config import load_config
from db import PostgresDatabase, SheetsDatabase
from db.models import GenerationJobRecord, job_from_mapping, parse_metadata, user_from_mapping

log = logging.getLogger(__name__)


def _parse_int(value: object) -> Optional[int]:
    try:
        if value in (None, "", " "):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _statuses_for_migration() -> Iterable[str]:
    return {"queued", "running", "completed", "failed", "pending", "processing", "in_progress"}


async def _migrate_users(source: SheetsDatabase, target: PostgresDatabase) -> None:
    users = await source.list_users()
    migrated = 0
    for record in users:
        user = user_from_mapping(record)
        await target.ensure_user(
            user.telegram_id,
            user.username,
            first_name=user.first_name,
            last_name=user.last_name,
        )
        await target.set_user_credits(user.telegram_id, user.credits)
        if user.bonus_granted:
            await target.mark_bonus_granted(user.telegram_id)
        if user.economy_v2:
            await target.mark_economy_v2(user.telegram_id)
        migrated += 1
    log.info("Migrated %s users", migrated)


async def _migrate_payments(source: SheetsDatabase, target: PostgresDatabase) -> None:
    payments = await source.list_payments()
    migrated = 0
    for record in payments:
        provider = (record.get("provider") or "").strip() or "unknown"
        ext_id = (record.get("ext_id") or "").strip() or f"legacy-{migrated}"
        user_id = _parse_int(record.get("user_id")) or 0
        amount_cp = _parse_int(record.get("amount_cp")) or 0
        items = _parse_int(record.get("items")) or 0
        status = (record.get("status") or "").strip() or "pending"
        purchased_credits = _parse_int(record.get("purchased_credits"))
        metadata = parse_metadata(record.get("metadata"))
        await target.create_payment_record(
            provider,
            ext_id,
            user_id,
            amount_cp,
            items,
            status,
            payload=record.get("payload") or "",
            metadata=metadata,
            idempotency_key=record.get("idempotency_key") or "",
            username=record.get("username") or None,
            package_id=record.get("package_id") or None,
            purchased_credits=purchased_credits,
            processed_at=record.get("processed_at") or None,
            record_type=record.get("type") or "payment",
            ref_payment_id=record.get("ref_payment_id") or None,
        )
        migrated += 1
    log.info("Migrated %s payments", migrated)


async def _migrate_jobs(source: SheetsDatabase, target: PostgresDatabase) -> None:
    jobs = await source.list_jobs_by_status(_statuses_for_migration())
    migrated = 0
    for record in jobs:
        job: GenerationJobRecord = job_from_mapping(record)
        await target.create_job(job)
        # Preserve terminal fields for completed/failed jobs
        await target.update_job(
            job.id,
            job.status,
            video_url=job.video_url,
            video_id=job.video_id,
            file_url=job.file_url,
            operation_name=job.operation_name,
            error=job.error,
            status_message_id=job.status_message_id,
            status_message_index=job.status_message_index,
            status_message_updated_at=job.status_message_updated_at,
        )
        migrated += 1
    log.info("Migrated %s jobs", migrated)


async def migrate() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    if not config.database_url:
        raise RuntimeError("DATABASE_URL is required to run the migration")

    source = SheetsDatabase()
    target = PostgresDatabase(
        dsn=config.database_url,
        min_pool_size=config.postgres_pool_min_size,
        max_pool_size=config.postgres_pool_max_size,
    )
    await source.init()
    await target.init()
    try:
        await _migrate_users(source, target)
        await _migrate_payments(source, target)
        await _migrate_jobs(source, target)
    finally:
        await source.close()
        await target.close()


if __name__ == "__main__":
    asyncio.run(migrate())
