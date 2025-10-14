"""Database helpers implemented on top of Google Sheets."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import gsheets_db


@dataclass
class User:
    telegram_id: int
    credits: int
    bonus_granted: bool
    created_at: datetime
    updated_at: datetime
    notes: str = ""


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
    image_file_id: Optional[str] = None
    sora_req_id: Optional[str] = None
    size: Optional[str] = None
    seconds: Optional[int] = None
    model: Optional[str] = None


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
    )


def _job_from_dict(data: Dict[str, Any]) -> GenerationJobRecord:
    video_url = data.get("video_url") or None
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
    return GenerationJobRecord(
        id=str(data.get("job_id", "")),
        user_id=int(data.get("user_id", 0)),
        prompt=str(data.get("prompt", "")),
        status=str(data.get("status", "")),
        video_url=video_url,
        error=error,
        created_at=_parse_datetime(data.get("created_at", "")),
        updated_at=_parse_datetime(data.get("updated_at", "")),
        image_file_id=image_file_id,
        sora_req_id=sora_req_id,
        size=size,
        seconds=seconds,
        model=model,
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

    async def ensure_user(self, telegram_id: int, username: Optional[str]) -> User:
        data = await gsheets_db.get_or_create_user(telegram_id)
        return _user_from_dict(data)

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
        payload: str,
    ) -> None:
        metadata = _parse_metadata(payload)
        await gsheets_db.create_payment_record(
            provider,
            ext_id,
            user_id,
            amount_cp,
            items,
            status,
            payload=payload,
            metadata=metadata,
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
        record = await gsheets_db.get_payment_by_ext(provider, ext_id)
        existing_metadata = _parse_metadata(record.get("metadata") if record else None)
        if metadata is not None:
            incoming = _parse_metadata(metadata)
            existing_metadata.update(incoming)
        if credits_added is not None:
            existing_metadata["credits_added"] = credits_added
        metadata_str = json.dumps(existing_metadata) if existing_metadata else ""
        await gsheets_db.update_payment_status_by_ext(
            provider,
            ext_id,
            status,
            metadata=metadata_str,
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
        )

    async def update_job(
        self,
        job_id: str,
        status: str,
        *,
        video_url: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        normalised_status = _normalise_job_status(status)
        updates: Dict[str, Any] = {}
        if video_url is not None:
            updates["video_url"] = video_url
        if error is not None:
            updates["error"] = error
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


__all__ = [
    "Database",
    "User",
    "GenerationJobRecord",
]
