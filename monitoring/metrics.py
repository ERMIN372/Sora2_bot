from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest

log = logging.getLogger(__name__)

router = APIRouter()

# Core metrics
_HANDLER_CALLS = Counter(
    "bot_handler_calls_total",
    "Total number of Telegram handler invocations",
    labelnames=("handler", "status"),
)
_JOBS_ENQUEUED = Counter(
    "jobs_enqueued_total",
    "Total jobs queued for processing",
    labelnames=("task_type", "provider"),
)
_JOBS_COMPLETED = Counter(
    "jobs_completed_total",
    "Total jobs completed with status label",
    labelnames=("task_type", "status", "provider"),
)
_JOB_ERRORS = Counter(
    "job_errors_total",
    "Total jobs that failed with an exception",
    labelnames=("task_type", "provider"),
)
_ACTIVE_JOBS = Gauge(
    "active_jobs_count",
    "Number of jobs currently being processed",
    labelnames=("provider",),
)
_JOB_QUEUE_SIZE = Gauge(
    "jobs_queue_size",
    "Current number of jobs waiting in the in-memory queue",
)
_POOL_EXHAUSTED = Counter(
    "job_pool_exhausted_total",
    "Number of times the worker pool was saturated",
)


def _label_value(value: Optional[str], fallback: str = "unknown") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def observe_handler(handler: str, *, success: bool) -> None:
    """Increment handler counter with success/error label."""

    status = "success" if success else "error"
    _HANDLER_CALLS.labels(handler=handler, status=status).inc()


async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/metrics", include_in_schema=False)
async def metrics_endpoint() -> Response:
    return await metrics()


def record_job_enqueued(
    *, task_type: str, provider: Optional[str], queue_depth: Optional[int] = None, concurrency: Optional[int] = None
) -> None:
    provider_value = _label_value(provider)
    task_value = _label_value(task_type, "unknown")
    _JOBS_ENQUEUED.labels(task_type=task_value, provider=provider_value).inc()
    if queue_depth is not None:
        try:
            _JOB_QUEUE_SIZE.set(float(queue_depth))
        except Exception:
            log.debug("Failed to set queue depth metric", exc_info=True)
        if concurrency and queue_depth >= concurrency:
            _POOL_EXHAUSTED.inc()


def job_started(*, provider: Optional[str]) -> None:
    provider_value = _label_value(provider)
    _ACTIVE_JOBS.labels(provider=provider_value).inc()


def job_finished(*, task_type: str, provider: Optional[str], status: str) -> None:
    provider_value = _label_value(provider)
    task_value = _label_value(task_type, "unknown")
    status_value = _label_value(status, "unknown")
    _ACTIVE_JOBS.labels(provider=provider_value).dec()
    _JOBS_COMPLETED.labels(task_type=task_value, status=status_value, provider=provider_value).inc()


def job_failed(*, task_type: str, provider: Optional[str]) -> None:
    provider_value = _label_value(provider)
    task_value = _label_value(task_type, "unknown")
    _JOB_ERRORS.labels(task_type=task_value, provider=provider_value).inc()

