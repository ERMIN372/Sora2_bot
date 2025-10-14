"""Utility helpers for the Sora Telegram bot."""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, TypeVar

from aiogram import Bot

from config import Config


T = TypeVar("T")
log = logging.getLogger(__name__)


async def async_retry(
    func: Callable[[], Awaitable[T]],
    *,
    retries: int,
    backoff: float,
) -> T:
    """Retry *func* with exponential backoff.

    The callable is awaited immediately.  On failure the exception is logged and
    retried up to ``retries`` times with exponential backoff delays.
    """

    attempt = 0
    delay = backoff
    last_exc: Exception | None = None
    while attempt <= retries:
        try:
            return await func()
        except Exception as exc:  # pragma: no cover - network interactions
            last_exc = exc
            if attempt == retries:
                raise
            log.warning("Retrying after error: %s", exc, exc_info=True)
            await asyncio.sleep(delay)
            delay *= backoff
            attempt += 1
    assert last_exc is not None  # pragma: no cover - defensive
    raise last_exc


async def run_cancellable(task: Awaitable[T]) -> T:
    """Await *task* and suppress cancellation caused by shutdown."""

    try:
        return await task
    except asyncio.CancelledError:  # pragma: no cover - runtime guard
        log.debug("Task was cancelled")
        raise


__all__ = ["async_retry", "run_cancellable"]


async def check_subscription(bot: Bot, user_id: int, config: Config) -> bool:
    """Check whether *user_id* is a member of the configured subscription chat."""

    chat_id = config.subscription_chat_id
    if not chat_id:
        return False
    try:
        member = await bot.get_chat_member(chat_id, user_id)
    except Exception:  # pragma: no cover - Telegram API interaction
        log.warning("Failed to verify subscription for user %s", user_id, exc_info=True)
        return False
    return member.status in {"creator", "administrator", "member"}


__all__.append("check_subscription")
