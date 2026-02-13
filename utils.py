"""Utility helpers for the video generation bot."""

from __future__ import annotations

import asyncio
import base64
import imghdr
import io
import logging
import mimetypes
from typing import Awaitable, Callable, Optional, TypeVar

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


__all__ = [
    "async_retry",
    "run_cancellable",
    "build_inline_data_from_telegram_file",
    "build_telegram_file_url",
]


async def build_inline_data_from_telegram_file(bot: Bot, file_id: str) -> Optional[dict[str, str]]:
    """Download *file_id* and return Gemini inline_data payload."""

    try:
        telegram_file = await bot.get_file(file_id)
    except Exception:  # pragma: no cover - Telegram API interaction
        log.exception("Failed to fetch file metadata file_id=%s", file_id)
        return None
    buffer = io.BytesIO()
    try:
        await bot.download_file(telegram_file.file_path, buffer)
    except Exception:  # pragma: no cover - Telegram API interaction
        log.exception("Failed to download reference file file_id=%s", file_id)
        return None
    data = buffer.getvalue()
    if not data:
        return None
    mime = None
    if telegram_file.file_path:
        mime = mimetypes.guess_type(telegram_file.file_path)[0]
    if not mime:
        kind = imghdr.what(None, data)
        if kind:
            mime = f"image/{kind}"
    mime = mime or "image/jpeg"
    encoded = base64.b64encode(data).decode("ascii")
    return {"mime_type": mime, "data": encoded}


async def build_telegram_file_url(bot: Bot, file_id: str) -> Optional[str]:
    """Return Telegram file download URL for *file_id*.

    The resulting URL is publicly downloadable and includes bot token.
    """

    try:
        telegram_file = await bot.get_file(file_id)
    except Exception:  # pragma: no cover - Telegram API interaction
        log.exception("Failed to fetch file metadata for URL file_id=%s", file_id)
        return None
    file_path = (telegram_file.file_path or "").strip()
    if not file_path:
        return None
    token = (
        getattr(bot, "token", "")
        or getattr(bot, "_token", "")
        or getattr(bot, "_Bot__token", "")
        or ""
    ).strip()
    if not token:
        return None
    return f"https://api.telegram.org/file/bot{token}/{file_path}"


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
