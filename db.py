"""Database helpers and migrations for the Sora Telegram bot."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import aiosqlite


@dataclass
class User:
    telegram_id: int
    username: Optional[str]
    credits: int
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
            await self.execute(migration)
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
            "SELECT telegram_id, username, credits, created_at, updated_at FROM users WHERE telegram_id = ?",
            (telegram_id,),
        )
        return User(
            telegram_id=row["telegram_id"],
            username=row["username"],
            credits=row["credits"],
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
                user_id, provider_payment_charge_id, telegram_payment_charge_id, amount, credits_added
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                provider_payment_charge_id,
                telegram_payment_charge_id,
                amount,
                credits_added,
            ),
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
