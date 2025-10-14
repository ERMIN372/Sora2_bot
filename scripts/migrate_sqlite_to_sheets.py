"""Migrate existing SQLite data into Google Sheets storage."""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from typing import Any, Dict

import gsheets_db


async def _migrate_users(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        "SELECT telegram_id, credits, bonus_granted FROM users"
    )
    migrated = 0
    for row in cursor.fetchall():
        telegram_id = int(row["telegram_id"])
        current = await gsheets_db.get_or_create_user(telegram_id)
        delta = int(row["credits"] or 0) - int(current.get("credits", 0))
        if delta:
            await gsheets_db.add_credits(telegram_id, delta)
        if row["bonus_granted"]:
            await gsheets_db.mark_bonus_granted(telegram_id)
        migrated += 1
    return migrated


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {"raw": raw}


async def _migrate_payments(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        """
        SELECT provider, ext_id, user_id, amount, items, status, metadata,
               provider_payment_charge_id, telegram_payment_charge_id
          FROM payments
        """
    )
    migrated = 0
    for row in cursor.fetchall():
        provider = row["provider"] or "stars"
        ext_id = row["ext_id"] or row["provider_payment_charge_id"]
        user_id = int(row["user_id"] or 0)
        amount_cp = int(row["amount"] or 0)
        items = int(row["items"] or 0)
        status = row["status"] or "pending"
        metadata = _parse_metadata(row["metadata"])
        payload = row["telegram_payment_charge_id"] or row["provider_payment_charge_id"] or ""
        await gsheets_db.create_payment_record(
            provider,
            ext_id,
            user_id,
            amount_cp,
            items,
            status,
            payload,
            metadata,
        )
        migrated += 1
    return migrated


async def _migrate_jobs(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        """
        SELECT id, user_id, prompt, status, video_url, error
          FROM jobs
        """
    )
    migrated = 0
    for row in cursor.fetchall():
        job_id = row["id"]
        user_id = int(row["user_id"] or 0)
        prompt = row["prompt"] or ""
        status = row["status"] or "queued"
        video_url = row["video_url"] or None
        error = row["error"] or None
        await gsheets_db.create_job(job_id, user_id, prompt, None, "", 0, "")
        await gsheets_db.update_job_status(
            job_id,
            status,
            video_url=video_url,
            error=error,
        )
        migrated += 1
    return migrated


async def migrate(sqlite_path: str) -> None:
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        await gsheets_db.init()
        users_count = await _migrate_users(conn)
        payments_count = await _migrate_payments(conn)
        jobs_count = await _migrate_jobs(conn)
    finally:
        conn.close()
    print(
        f"Migrated {users_count} users, {payments_count} payments, {jobs_count} jobs to Google Sheets."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite_path", help="Path to the legacy SQLite database")
    args = parser.parse_args()
    asyncio.run(migrate(args.sqlite_path))


if __name__ == "__main__":
    main()
