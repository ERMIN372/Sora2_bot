"""Utilities for structured logging and diagnostic tracking."""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from hashlib import sha256
from html import escape
from typing import Any, Deque, Dict, List, Optional, Tuple

log = logging.getLogger("bot.observability")


@dataclass(slots=True)
class EventRecord:
    """Structured event emitted during the generation pipeline."""

    ts: float
    payload: Dict[str, Any]


@dataclass(slots=True)
class ErrorSnapshot:
    """Details about the last critical failure."""

    ts: float
    corr_id: Optional[str]
    job_id: Optional[str]
    status_code: Optional[int]
    error_type: Optional[str]
    error_code: Optional[str]
    error_msg_short: Optional[str]
    provider_message: Optional[str]
    attempts: int = 1


@dataclass(slots=True)
class HealthCheckResult:
    """Aggregate outcome of the startup health check."""

    ts: float
    mode: str
    ok: bool
    version: str
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class _EventStore:
    def __init__(self, max_events: int = 500) -> None:
        self._events: Deque[EventRecord] = deque(maxlen=max_events)

    def add(self, payload: Dict[str, Any]) -> None:
        self._events.append(EventRecord(ts=time.time(), payload=payload))

    def history(self, corr_id: str) -> List[Dict[str, Any]]:
        return [event.payload for event in self._events if event.payload.get("corr_id") == corr_id]


_EVENTS = _EventStore()
_LAST_ERROR: Optional[ErrorSnapshot] = None
_LAST_HEALTH: Optional[HealthCheckResult] = None
_SUPPORT_NOTIFICATIONS: Dict[str, float] = {}
_METRICS: Dict[str, float] = {}
_METRIC_TIMINGS: Dict[str, Tuple[float, int]] = {}


def _prompt_details(prompt: Optional[str]) -> Dict[str, Any]:
    if not prompt:
        return {}
    digest = sha256(prompt.encode("utf-8")).hexdigest()
    return {"prompt_hash": digest}


def _tagged_metric_name(name: str, tags: Optional[Dict[str, Any]]) -> str:
    if not tags:
        return name
    parts = []
    for key in sorted(tags):
        value = tags[key]
        if value is None:
            continue
        parts.append(f"{key}={value}")
    if not parts:
        return name
    return f"{name}{{{','.join(parts)}}}"


def increment_metric(
    name: str,
    value: float = 1.0,
    *,
    tags: Optional[Dict[str, Any]] = None,
) -> None:
    if not name:
        return
    metric_key = _tagged_metric_name(name, tags)
    current = _METRICS.get(metric_key, 0.0)
    try:
        increment = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive conversion
        increment = 0.0
    _METRICS[metric_key] = current + increment


def record_timing_metric(
    name: str,
    value: float,
    *,
    tags: Optional[Dict[str, Any]] = None,
) -> None:
    if not name:
        return
    try:
        timing = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive conversion
        return
    metric_key = _tagged_metric_name(name, tags)
    total, count = _METRIC_TIMINGS.get(metric_key, (0.0, 0))
    _METRIC_TIMINGS[metric_key] = (total + timing, count + 1)


def metrics_snapshot() -> Dict[str, float]:
    snapshot = dict(_METRICS)
    for metric, (total, count) in _METRIC_TIMINGS.items():
        if count:
            snapshot[f"{metric}_avg"] = total / count
    return snapshot


def log_event(
    *,
    level: str,
    event: str,
    corr_id: Optional[str] = None,
    job_id: Optional[str] = None,
    user_id: Optional[int] = None,
    username: Optional[str] = None,
    provider: str = "veo",
    model: Optional[str] = None,
    size: Optional[str] = None,
    credits_cost: Optional[int] = None,
    duration_ms: Optional[int] = None,
    status_code: Optional[int] = None,
    error_type: Optional[str] = None,
    error_code: Optional[str] = None,
    error_msg_short: Optional[str] = None,
    prompt: Optional[str] = None,
    gsheets_ok: Optional[bool] = None,
    refund_done: Optional[bool] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Log a structured event to the standard logger and diagnostics store."""

    payload: Dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": level,
        "event": event,
        "corr_id": corr_id,
        "job_id": job_id,
        "user_id": user_id,
        "username": username,
        "provider": provider,
        "model": model,
        "size": size,
        "credits_cost": credits_cost,
        "duration_ms": duration_ms,
        "status_code": status_code,
        "error_type": error_type,
        "error_code": error_code,
        "error_msg_short": error_msg_short,
        "gsheets_ok": gsheets_ok,
        "refund_done": refund_done,
    }
    payload.update(_prompt_details(prompt))
    if extra:
        payload.update(extra)
    serialised = json.dumps({k: v for k, v in payload.items() if v is not None}, ensure_ascii=False)
    log.log(getattr(logging, level.upper(), logging.INFO), serialised)
    _EVENTS.add(payload)
    if event == "error":
        _record_error(payload)
    return payload


def _record_error(payload: Dict[str, Any]) -> None:
    global _LAST_ERROR
    snapshot = ErrorSnapshot(
        ts=time.time(),
        corr_id=payload.get("corr_id"),
        job_id=payload.get("job_id"),
        status_code=payload.get("status_code"),
        error_type=payload.get("error_type"),
        error_code=payload.get("error_code"),
        error_msg_short=payload.get("error_msg_short"),
        provider_message=(payload.get("provider_message") if isinstance(payload.get("provider_message"), str) else None),
        attempts=int(payload.get("attempt", 1) or 1),
    )
    _LAST_ERROR = snapshot


def get_last_error() -> Optional[ErrorSnapshot]:
    return _LAST_ERROR


def get_job_history(corr_id: str) -> List[Dict[str, Any]]:
    return _EVENTS.history(corr_id)


def record_healthcheck(result: HealthCheckResult) -> None:
    global _LAST_HEALTH
    _LAST_HEALTH = result


def get_last_healthcheck() -> Optional[HealthCheckResult]:
    return _LAST_HEALTH


def should_notify_support(key: str, interval_seconds: int) -> bool:
    now = time.time()
    last = _SUPPORT_NOTIFICATIONS.get(key)
    if last is not None and now - last < interval_seconds:
        return False
    _SUPPORT_NOTIFICATIONS[key] = now
    return True


__all__ = [
    "ErrorSnapshot",
    "HealthCheckResult",
    "increment_metric",
    "get_job_history",
    "get_last_error",
    "get_last_healthcheck",
    "metrics_snapshot",
    "log_event",
    "record_timing_metric",
    "record_healthcheck",
    "should_notify_support",
]
