from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Mapping, Optional


@dataclass
class User:
    telegram_id: int
    credits: int
    bonus_granted: bool
    created_at: datetime
    updated_at: datetime
    notes: str = ""
    economy_v2: bool = False
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None
    tg_link: Optional[str] = None


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


_MAX_TS_DEFAULT = datetime.utcfromtimestamp(0)


def parse_datetime(value: Any) -> datetime:
    if not value:
        return _MAX_TS_DEFAULT
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return _MAX_TS_DEFAULT


def parse_metadata(value: Optional[Any]) -> Dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, json.JSONDecodeError):
        pass
    return {"raw": value}


def normalise_job_status(status: str) -> str:
    status_lower = (status or "").lower()
    if status_lower in {"queued", "running", "completed", "failed"}:
        return status_lower
    if status_lower in {"processing", "pending", "in_progress"}:
        return "running"
    if status_lower in {"errored", "error"}:
        return "failed"
    return status_lower or "queued"


def derive_payment_breakdown(amount_cp: int, purchased_credits: Optional[int]) -> tuple[Decimal, int, int, Decimal]:
    amount_dec = Decimal(int(amount_cp)) / Decimal(100)
    price_rub = amount_dec.quantize(Decimal("0.01"))
    credits_bought = int(price_rub.to_integral_value(rounding=ROUND_HALF_UP))
    total_credits: Optional[int] = None
    if purchased_credits is not None:
        try:
            total_credits = int(purchased_credits)
        except (TypeError, ValueError):
            total_credits = None
    if total_credits is None:
        total_credits = credits_bought
    bonus_credits = max(0, total_credits - credits_bought)
    bonus_pct = Decimal("0")
    if credits_bought > 0 and bonus_credits > 0:
        bonus_pct = (
            Decimal(bonus_credits) / Decimal(credits_bought)
        ).quantize(Decimal("0.0001"))
    return price_rub, credits_bought, bonus_credits, bonus_pct


def user_from_mapping(data: Mapping[str, Any]) -> User:
    return User(
        telegram_id=int(data.get("user_id", 0)),
        credits=int(data.get("credits", 0)),
        bonus_granted=bool(data.get("bonus_granted")),
        created_at=parse_datetime(data.get("created_at", "")),
        updated_at=parse_datetime(data.get("updated_at", "")),
        notes=str(data.get("notes", "")) if data.get("notes") is not None else "",
        economy_v2=bool(int(data.get("economy_v2", 0) or 0)),
        username=str(data.get("username", "")) or None,
        first_name=str(data.get("first_name", "")) or None,
        last_name=str(data.get("last_name", "")) or None,
        display_name=str(data.get("display_name", "")) or None,
        tg_link=str(data.get("tg_link", "")) or None,
    )


def job_from_mapping(data: Mapping[str, Any]) -> GenerationJobRecord:
    video_url = data.get("video_url") or None
    if video_url == "None":
        video_url = None
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
    content_type = str(data.get("content_type") or "video")
    operation_name = data.get("operation_name") or None
    metadata = parse_metadata(data.get("metadata"))
    return GenerationJobRecord(
        id=str(data.get("job_id", "")),
        user_id=int(data.get("user_id", 0)),
        prompt=str(data.get("prompt", "")),
        status=str(data.get("status", "")),
        video_url=video_url,
        video_id=data.get("video_id") or None,
        error=error,
        created_at=parse_datetime(data.get("created_at", "")),
        updated_at=parse_datetime(data.get("updated_at", "")),
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
        status_message_updated_at=parse_datetime(status_message_updated_raw),
        operation_name=operation_name,
        file_url=file_url,
        extra=dict(metadata),
    )


__all__ = [
    "ArchiveLogRecord",
    "ErrorLogRecord",
    "GenerationJobRecord",
    "User",
    "derive_payment_breakdown",
    "job_from_mapping",
    "normalise_job_status",
    "parse_datetime",
    "parse_metadata",
    "user_from_mapping",
]
