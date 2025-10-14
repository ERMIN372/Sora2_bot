"""Database helpers and migrations for the Sora Telegram bot."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import aiosqlite


@dataclass
class User:
    telegram_id: int
    username: Optional[str]
    credits: int
    bonus_granted: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class GenerationJobRecord:
    id: str
    user_id: int
    prompt: str
    status: str
    video_url: Optional[str]
    error: Optional[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class Payment:
    id: int
    user_id: int
    provider_payment_charge_id: str
    telegram_payment_charge_id: str
    amount: int
    credits_added: int
    provider: str
    ext_id: str
    items: int
    status: str
    metadata: Optional[str]
    created_at: datetime


MIGRATIONS: List[Tuple[str, str]] = [
    (
        "0001_users",
        """
    CREATE TABLE IF NOT EXISTS users (
        telegram_id INTEGER PRIMARY KEY,
        username TEXT,
        credits INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """,
    ),
    (
        "0002_jobs",
        """
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        user_id INTEGER NOT NULL,
        prompt TEXT NOT NULL,
        status TEXT NOT NULL,
        video_url TEXT,
        error TEXT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(telegram_id)
    );
    """,
    ),
    (
        "0003_payments",
        """
    CREATE TABLE IF NOT EXISTS payments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        provider_payment_charge_id TEXT NOT NULL,
        telegram_payment_charge_id TEXT NOT NULL,
        amount INTEGER NOT NULL,
        credits_added INTEGER NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(telegram_id)
    );
    """,
    ),
    (
        "0004_users_bonus",
        """
    ALTER TABLE users ADD COLUMN bonus_granted INTEGER NOT NULL DEFAULT 0;
    """,
    ),
    (
        "0005_payments_yookassa",
        """
    ALTER TABLE payments ADD COLUMN provider TEXT NOT NULL DEFAULT 'stars';
    ALTER TABLE payments ADD COLUMN ext_id TEXT NOT NULL DEFAULT '';
    ALTER TABLE payments ADD COLUMN items INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE payments ADD COLUMN status TEXT NOT NULL DEFAULT 'succeeded';
    ALTER TABLE payments ADD COLUMN metadata TEXT;
    UPDATE payments SET ext_id = provider_payment_charge_id WHERE provider = 'stars' AND ext_id = '';
    UPDATE payments SET items = credits_added WHERE provider = 'stars' AND items = 0;
    UPDATE payments SET status = 'succeeded' WHERE provider = 'stars' AND status = 'succeeded';
    CREATE INDEX IF NOT EXISTS idx_payments_provider_ext ON payments(provider, ext_id);
    """,
    ),
]


class Database:
    """Lightweight async wrapper around :mod:`aiosqlite` with migrations."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._connection: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._connection is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self._path.as_posix())
        self._connection.row_factory = aiosqlite.Row
        await self._connection.execute("PRAGMA foreign_keys = ON;")
        await self._connection.commit()

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def execute(self, query: str, params: Iterable[Any] = ()) -> None:
        if self._connection is None:
            raise RuntimeError("Database connection has not been initialised")
        async with self._lock:
            await self._connection.execute(query, tuple(params))
            await self._connection.commit()

    async def executescript(self, script: str) -> None:
        if self._connection is None:
            raise RuntimeError("Database connection has not been initialised")
        async with self._lock:
            await self._connection.executescript(script)
            await self._connection.commit()

    async def fetchone(self, query: str, params: Iterable[Any] = ()) -> Optional[aiosqlite.Row]:
        if self._connection is None:
            raise RuntimeError("Database connection has not been initialised")
        async with self._lock:
            cursor = await self._connection.execute(query, tuple(params))
            row = await cursor.fetchone()
            await cursor.close()
            return row

    async def fetchall(self, query: str, params: Iterable[Any] = ()) -> List[aiosqlite.Row]:
        if self._connection is None:
            raise RuntimeError("Database connection has not been initialised")
        async with self._lock:
            cursor = await self._connection.execute(query, tuple(params))
            rows = await cursor.fetchall()
            await cursor.close()
            return rows

    async def run_migrations(self) -> None:
        await self.connect()
        await self.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        for name, migration in MIGRATIONS:
            row = await self.fetchone(
                "SELECT 1 FROM schema_migrations WHERE name = ?", (name,)
            )
            if row:
                continue
            await self.executescript(migration)
            await self.execute(
                "INSERT INTO schema_migrations (name) VALUES (?)",
                (name,),
            )

    # -- DAO helpers -----------------------------------------------------

    async def ensure_user(self, telegram_id: int, username: Optional[str]) -> User:
        await self.execute(
            "INSERT INTO users (telegram_id, username) VALUES (?, ?)"
            " ON CONFLICT(telegram_id) DO UPDATE SET"
            " username = excluded.username,"
            " updated_at = CURRENT_TIMESTAMP",
            (telegram_id, username),
        )
        row = await self.fetchone(
            """
            SELECT telegram_id, username, credits, bonus_granted, created_at, updated_at
              FROM users
             WHERE telegram_id = ?
            """,
            (telegram_id,),
        )
        return User(
            telegram_id=row["telegram_id"],
            username=row["username"],
            credits=row["credits"],
            bonus_granted=bool(row["bonus_granted"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def get_user_credits(self, telegram_id: int) -> int:
        row = await self.fetchone(
            "SELECT credits FROM users WHERE telegram_id = ?",
            (telegram_id,),
        )
        return int(row["credits"]) if row else 0

    async def add_credits(self, telegram_id: int, amount: int) -> int:
        await self.execute(
            "UPDATE users SET credits = credits + ?, updated_at = CURRENT_TIMESTAMP WHERE telegram_id = ?",
            (amount, telegram_id),
        )
        return await self.get_user_credits(telegram_id)

    async def deduct_credit(self, telegram_id: int, amount: int = 1) -> bool:
        row = await self.fetchone(
            "SELECT credits FROM users WHERE telegram_id = ?",
            (telegram_id,),
        )
        if not row or row["credits"] < amount:
            return False
        await self.execute(
            "UPDATE users SET credits = credits - ?, updated_at = CURRENT_TIMESTAMP WHERE telegram_id = ?",
            (amount, telegram_id),
        )
        return True

    async def create_job(self, job: GenerationJobRecord) -> None:
        await self.execute(
            """
            INSERT INTO jobs (id, user_id, prompt, status, video_url, error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job.id, job.user_id, job.prompt, job.status, job.video_url, job.error),
        )

    async def update_job(self, job_id: str, status: str, *, video_url: Optional[str] = None, error: Optional[str] = None) -> None:
        await self.execute(
            """
            UPDATE jobs
               SET status = ?,
                   video_url = COALESCE(?, video_url),
                   error = ?,
                   updated_at = CURRENT_TIMESTAMP
             WHERE id = ?
            """,
            (status, video_url, error, job_id),
        )

    async def list_pending_jobs(self, *, limit: int) -> List[GenerationJobRecord]:
        rows = await self.fetchall(
            """
            SELECT id, user_id, prompt, status, video_url, error, created_at, updated_at
              FROM jobs
             WHERE status IN ('queued', 'processing')
             ORDER BY created_at ASC
             LIMIT ?
            """,
            (limit,),
        )
        return [
            GenerationJobRecord(
                id=row["id"],
                user_id=row["user_id"],
                prompt=row["prompt"],
                status=row["status"],
                video_url=row["video_url"],
                error=row["error"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    async def record_payment(
        self,
        *,
        user_id: int,
        provider_payment_charge_id: str,
        telegram_payment_charge_id: str,
        amount: int,
        credits_added: int,
    ) -> None:
        await self.execute(
            """
            INSERT INTO payments (
                user_id,
                provider_payment_charge_id,
                telegram_payment_charge_id,
                amount,
                credits_added,
                provider,
                ext_id,
                items,
                status,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                provider_payment_charge_id,
                telegram_payment_charge_id,
                amount,
                credits_added,
                "stars",
                provider_payment_charge_id,
                credits_added,
                "succeeded",
                "{}",
            ),
        )

    async def create_payment_record(
        self,
        provider: str,
        ext_id: str,
        user_id: int,
        amount_cp: int,
        items: int,
        status: str,
        payload: str,
    ) -> None:
        payload_str = payload or "{}"
        await self.execute(
            """
            INSERT INTO payments (
                user_id,
                provider_payment_charge_id,
                telegram_payment_charge_id,
                amount,
                credits_added,
                provider,
                ext_id,
                items,
                status,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                "",
                "",
                amount_cp,
                0,
                provider,
                ext_id,
                items,
                status,
                payload_str,
            ),
        )

    async def update_payment_status_by_ext(
        self,
        provider: str,
        ext_id: str,
        status: str,
        *,
        credits_added: Optional[int] = None,
        metadata: Optional[str] = None,
    ) -> None:
        assignments = ["status = ?"]
        params: List[Any] = [status]
        if credits_added is not None:
            assignments.append("credits_added = ?")
            params.append(credits_added)
        if metadata is not None:
            assignments.append("metadata = ?")
            params.append(metadata or "{}")
        params.extend([provider, ext_id])
        await self.execute(
            f"UPDATE payments SET {', '.join(assignments)} WHERE provider = ? AND ext_id = ?",
            params,
        )

    async def get_payment_by_ext(self, provider: str, ext_id: str) -> Optional[Dict[str, Any]]:
        row = await self.fetchone(
            """
            SELECT id,
                   user_id,
                   provider_payment_charge_id,
                   telegram_payment_charge_id,
                   amount,
                   credits_added,
                   provider,
                   ext_id,
                   items,
                   status,
                   metadata,
                   created_at
              FROM payments
             WHERE provider = ? AND ext_id = ?
            """,
            (provider, ext_id),
        )
        if not row:
            return None
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "provider_payment_charge_id": row["provider_payment_charge_id"],
            "telegram_payment_charge_id": row["telegram_payment_charge_id"],
            "amount": row["amount"],
            "credits_added": row["credits_added"],
            "provider": row["provider"],
            "ext_id": row["ext_id"],
            "items": row["items"],
            "status": row["status"],
            "metadata": row["metadata"],
            "created_at": row["created_at"],
        }

    async def list_payments_by_status(
        self,
        *,
        provider: str,
        statuses: Iterable[str],
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        status_list = list(statuses)
        if not status_list:
            return []
        placeholders = ",".join("?" for _ in status_list)
        params: List[Any] = [provider, *status_list, limit]
        rows = await self.fetchall(
            f"""
            SELECT id,
                   user_id,
                   provider_payment_charge_id,
                   telegram_payment_charge_id,
                   amount,
                   credits_added,
                   provider,
                   ext_id,
                   items,
                   status,
                   metadata,
                   created_at
              FROM payments
             WHERE provider = ? AND status IN ({placeholders})
             ORDER BY created_at ASC
             LIMIT ?
            """,
            params,
        )
        return [
            {
                "id": row["id"],
                "user_id": row["user_id"],
                "provider_payment_charge_id": row["provider_payment_charge_id"],
                "telegram_payment_charge_id": row["telegram_payment_charge_id"],
                "amount": row["amount"],
                "credits_added": row["credits_added"],
                "provider": row["provider"],
                "ext_id": row["ext_id"],
                "items": row["items"],
                "status": row["status"],
                "metadata": row["metadata"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    async def get_payment_by_order_id(self, provider: str, order_id: str) -> Optional[Dict[str, Any]]:
        try:
            row = await self.fetchone(
                """
                SELECT id,
                       user_id,
                       provider_payment_charge_id,
                       telegram_payment_charge_id,
                       amount,
                       credits_added,
                       provider,
                       ext_id,
                       items,
                       status,
                       metadata,
                       created_at
                  FROM payments
                 WHERE provider = ? AND json_extract(metadata, '$.idemp') = ?
                """,
                (provider, order_id),
            )
        except aiosqlite.OperationalError:
            # Fallback when JSON1 extension is unavailable: scan in Python.
            rows = await self.fetchall(
                """
                SELECT id,
                       user_id,
                       provider_payment_charge_id,
                       telegram_payment_charge_id,
                       amount,
                       credits_added,
                       provider,
                       ext_id,
                       items,
                       status,
                       metadata,
                       created_at
                  FROM payments
                 WHERE provider = ?
                """,
                (provider,),
            )
            for candidate in rows:
                metadata = candidate["metadata"]
                if not metadata:
                    continue
                try:
                    payload = json.loads(metadata)
                except json.JSONDecodeError:
                    continue
                if payload.get("idemp") == order_id or payload.get("order_id") == order_id:
                    row = candidate
                    break
            else:
                row = None
        if not row:
            return None
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "provider_payment_charge_id": row["provider_payment_charge_id"],
            "telegram_payment_charge_id": row["telegram_payment_charge_id"],
            "amount": row["amount"],
            "credits_added": row["credits_added"],
            "provider": row["provider"],
            "ext_id": row["ext_id"],
            "items": row["items"],
            "status": row["status"],
            "metadata": row["metadata"],
            "created_at": row["created_at"],
        }

    async def is_bonus_granted(self, telegram_id: int) -> bool:
        row = await self.fetchone(
            "SELECT bonus_granted FROM users WHERE telegram_id = ?",
            (telegram_id,),
        )
        return bool(row and row["bonus_granted"])

    async def mark_bonus_granted(self, telegram_id: int) -> None:
        await self.execute(
            "UPDATE users SET bonus_granted = 1, updated_at = CURRENT_TIMESTAMP WHERE telegram_id = ?",
            (telegram_id,),
        )

    async def count_active_jobs(self, user_id: int) -> int:
        row = await self.fetchone(
            "SELECT COUNT(*) AS count FROM jobs WHERE user_id = ? AND status IN ('queued', 'processing')",
            (user_id,),
        )
        return int(row["count"]) if row else 0

    async def get_job(self, job_id: str) -> Optional[GenerationJobRecord]:
        row = await self.fetchone(
            "SELECT id, user_id, prompt, status, video_url, error, created_at, updated_at FROM jobs WHERE id = ?",
            (job_id,),
        )
        if not row:
            return None
        return GenerationJobRecord(
            id=row["id"],
            user_id=row["user_id"],
            prompt=row["prompt"],
            status=row["status"],
            video_url=row["video_url"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )


__all__ = [
    "Database",
    "User",
    "GenerationJobRecord",
    "Payment",
]
