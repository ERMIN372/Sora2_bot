import asyncio
import io
import logging
import math
import mimetypes
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

import aiohttp
from aiogram import Bot
from aiogram.types import InputFile
from aiogram.utils import exceptions as aiogram_exceptions

BadRequest = aiogram_exceptions.BadRequest
ChatNotFound = aiogram_exceptions.ChatNotFound
NetworkError = aiogram_exceptions.NetworkError
RetryAfter = aiogram_exceptions.RetryAfter
TelegramAPIError = aiogram_exceptions.TelegramAPIError
TelegramServerError = getattr(
    aiogram_exceptions, "TelegramServerError", TelegramAPIError
)
Unauthorized = aiogram_exceptions.Unauthorized
CantTalkWithBot = getattr(aiogram_exceptions, "CantTalkWithBot", TelegramAPIError)

from config import Config
from db import ArchiveLogRecord, Database

log = logging.getLogger(__name__)


_MD_V2_SPECIAL_CHARS = "_*[]()~`>#+-=|{}.!"
_MAX_TELEGRAM_FILE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_CAPTION_LENGTH = 1024
_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 2.0


@dataclass(slots=True)
class ArchivePayload:
    corr_id: str
    prompt: Optional[str]
    model_name: Optional[str]
    username: Optional[str]
    user_id: Optional[int]
    content_type: Literal["image", "video"]
    mime_type: Optional[str]
    file_bytes: Optional[bytes] = None
    file_path: Optional[str] = None
    file_url: Optional[str] = None
    file_size: Optional[int] = None
    filename: Optional[str] = None
    duration_seconds: Optional[int] = None

    def resolved_file_size(self) -> Optional[int]:
        if self.file_size is not None:
            return self.file_size
        if self.file_bytes is not None:
            self.file_size = len(self.file_bytes)
        elif self.file_path:
            try:
                self.file_size = os.path.getsize(self.file_path)
            except OSError:
                self.file_size = None
        return self.file_size


class ArchivePublisher:
    """Best-effort publisher that mirrors generation results to an archive channel."""

    def __init__(self, *, bot: Bot, config: Config, db: Database) -> None:
        self._bot = bot
        self._config = config
        self._db = db
        self._channel_id = config.archive_channel_id
        self._lock = asyncio.Lock()
        self._inflight: set[str] = set()
        self._sent_cache: set[str] = set()
        self._admin_checked = False
        self._can_publish = False
        self._bot_id: Optional[int] = None

    def schedule(self, payload: ArchivePayload) -> None:
        if not payload.corr_id:
            log.debug("Skipping archive publish without corr_id")
            return
        log.debug(
            "Scheduling archive publish corr_id=%s type=%s source=%s size=%s",
            payload.corr_id,
            payload.content_type,
            "bytes"
            if payload.file_bytes is not None
            else "path"
            if payload.file_path
            else "url"
            if payload.file_url
            else "unknown",
            payload.file_size,
        )
        asyncio.create_task(self._publish_safe(payload))

    async def _publish_safe(self, payload: ArchivePayload) -> None:
        try:
            await self._publish(payload)
        except Exception:  # pragma: no cover - defensive guard
            log.exception("Archive publication crashed corr_id=%s", payload.corr_id)

    async def _publish(self, payload: ArchivePayload) -> None:
        corr_id = payload.corr_id
        if not self._channel_id:
            log.debug("Archive channel is not configured; skipping corr_id=%s", corr_id)
            await self._log_attempt(payload, archive_status="skipped", error_short="no_channel")
            return

        async with self._lock:
            if corr_id in self._inflight:
                log.debug("Archive publication already in progress corr_id=%s", corr_id)
                return
            self._inflight.add(corr_id)

        message = None
        attempts = 0
        final_status = "pending"
        try:
            log.debug("Archive publish started corr_id=%s channel_id=%s", corr_id, self._channel_id)
            if corr_id in self._sent_cache or await self._db.archive_was_sent(corr_id):
                self._sent_cache.add(corr_id)
                log.debug("Archive already sent corr_id=%s", corr_id)
                final_status = "skipped_already_sent"
                await self._log_attempt(
                    payload,
                    archive_status="skipped",
                    error_short="already_sent",
                    attempts=0,
                )
                return

            size = payload.resolved_file_size()
            log.debug(
                "Archive payload resolved size corr_id=%s size=%s has_bytes=%s has_path=%s has_url=%s",
                corr_id,
                size,
                payload.file_bytes is not None,
                bool(payload.file_path),
                bool(payload.file_url),
            )
            if size is not None and size > _MAX_TELEGRAM_FILE_BYTES:
                log.warning(
                    "Archive payload too large corr_id=%s size=%s", corr_id, size
                )
                final_status = "failed_too_large"
                await self._log_attempt(
                    payload,
                    archive_status="failed",
                    error_short="too_large",
                    attempts=0,
                )
                return

            can_publish = await self._ensure_admin_access()
            log.debug("Archive admin access corr_id=%s can_publish=%s", corr_id, can_publish)
            if not can_publish:
                final_status = "failed_no_admin"
                await self._log_attempt(
                    payload,
                    archive_status="failed",
                    error_short="no_admin",
                    attempts=0,
                )
                return

            caption, caption_len = self._build_caption(payload)
            error_short: Optional[str] = None

            log.debug(
                "Archive publish sending corr_id=%s caption_len=%s attempts=%s",
                corr_id,
                caption_len,
                _RETRY_ATTEMPTS,
            )

            for attempt in range(1, _RETRY_ATTEMPTS + 1):
                attempts = attempt
                try:
                    log.debug("Archive send attempt corr_id=%s attempt=%s", corr_id, attempt)
                    message = await self._send_media(payload, caption)
                except RetryAfter as exc:
                    error_short = "429"
                    delay = exc.timeout or (_RETRY_BASE_DELAY * math.pow(2, attempt - 1))
                    log.warning(
                        "Archive publish rate limited corr_id=%s attempt=%s delay=%s error=%s",
                        corr_id,
                        attempt,
                        delay,
                        getattr(exc, "message", str(exc)),
                    )
                    if attempt >= _RETRY_ATTEMPTS:
                        log.warning(
                            "Archive publish giving up after rate limit corr_id=%s attempts=%s",
                            corr_id,
                            attempt,
                        )
                        message = None
                        break
                    await asyncio.sleep(delay)
                    continue
                except (
                    NetworkError,
                    TelegramServerError,
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                ) as exc:
                    error_short = self._network_error_short(exc)
                    delay = _RETRY_BASE_DELAY * math.pow(2, attempt - 1)
                    log.warning(
                        "Archive publish network error corr_id=%s attempt=%s retry_in=%s error=%s",
                        corr_id,
                        attempt,
                        delay,
                        str(exc),
                        exc_info=True,
                    )
                    if attempt >= _RETRY_ATTEMPTS:
                        log.warning(
                            "Archive publish giving up after network errors corr_id=%s attempts=%s",
                            corr_id,
                            attempt,
                        )
                        message = None
                        break
                    await asyncio.sleep(delay)
                    continue
                except (BadRequest, Unauthorized, ChatNotFound, CantTalkWithBot) as exc:
                    error_short = self._map_error_short(exc)
                    log.warning(
                        "Archive publish failed corr_id=%s error=%s", corr_id, error_short
                    )
                    message = None
                    break
                except TelegramAPIError as exc:  # pragma: no cover - defensive mapping
                    error_short = self._map_error_short(exc)
                    log.warning(
                        "Archive publish failed corr_id=%s api_error=%s", corr_id, error_short
                    )
                    message = None
                    break
                except Exception as exc:  # pragma: no cover - defensive
                    error_short = exc.__class__.__name__.lower()
                    log.exception("Archive publish crashed corr_id=%s", corr_id)
                    message = None
                    break
                else:
                    error_short = None
                    break

            if message:
                self._sent_cache.add(corr_id)
                final_status = "sent"
                await self._log_attempt(
                    payload,
                    archive_status="sent",
                    channel_message_id=message.message_id,
                    caption_len=caption_len,
                    attempts=attempts,
                    error_short=error_short,
                )
            else:
                final_status = "failed"
                await self._log_attempt(
                    payload,
                    archive_status="failed",
                    caption_len=caption_len,
                    attempts=attempts,
                    error_short=error_short or "retry_exhausted",
                )
        finally:
            log.debug(
                "Archive publish finished corr_id=%s status=%s attempts=%s", corr_id, final_status, attempts
            )
            async with self._lock:
                self._inflight.discard(corr_id)

    async def _ensure_admin_access(self) -> bool:
        if self._admin_checked:
            return self._can_publish
        try:
            if self._bot_id is None:
                me = await self._bot.get_me()
                self._bot_id = me.id
                log.debug("Fetched bot identity for archive publishing bot_id=%s", self._bot_id)
            member = await self._bot.get_chat_member(self._channel_id, self._bot_id)
            status = getattr(member, "status", "")
            self._can_publish = status in {"administrator", "creator"}
            if not self._can_publish:
                log.error(
                    "Bot lacks admin rights in archive channel=%s status=%s", self._channel_id, status
                )
        except (BadRequest, Unauthorized, ChatNotFound) as exc:
            log.error(
                "Failed to verify admin rights in archive channel=%s: %s",
                self._channel_id,
                exc,
            )
            self._can_publish = False
        except Exception:  # pragma: no cover - defensive guard
            log.exception("Unexpected error while verifying archive admin rights")
            self._can_publish = False
        finally:
            self._admin_checked = True
        return self._can_publish

    async def _send_media(self, payload: ArchivePayload, caption: str):
        data, filename = await self._prepare_input(payload)
        input_file = InputFile(data, filename=filename)
        if payload.content_type == "video":
            log.debug(
                "Sending archive video corr_id=%s filename=%s duration=%s", payload.corr_id, filename, payload.duration_seconds
            )
            return await self._bot.send_video(
                self._channel_id,
                input_file,
                caption=caption,
                parse_mode="MarkdownV2",
                duration=payload.duration_seconds,
            )
        log.debug("Sending archive document corr_id=%s filename=%s", payload.corr_id, filename)
        return await self._bot.send_document(
            self._channel_id,
            input_file,
            caption=caption,
            parse_mode="MarkdownV2",
        )

    async def _prepare_input(self, payload: ArchivePayload) -> tuple[io.BytesIO, str]:
        filename = payload.filename or self._build_filename(payload)
        if payload.file_bytes is not None:
            log.debug("Preparing archive payload from bytes corr_id=%s filename=%s", payload.corr_id, filename)
            buffer = io.BytesIO(payload.file_bytes)
            buffer.seek(0)
            return buffer, filename
        if payload.file_path:
            log.debug(
                "Preparing archive payload from path corr_id=%s path=%s filename=%s",
                payload.corr_id,
                payload.file_path,
                filename,
            )
            data = await asyncio.to_thread(self._read_file, payload.file_path)
            return io.BytesIO(data), filename
        if payload.file_url:
            log.debug(
                "Preparing archive payload from url corr_id=%s url=%s filename=%s",
                payload.corr_id,
                payload.file_url,
                filename,
            )
            data = await self._download(payload.file_url)
            return io.BytesIO(data), filename
        raise RuntimeError("Archive payload has no source data")

    async def _download(self, url: str) -> bytes:
        timeout = aiohttp.ClientTimeout(total=60)
        log.debug("Downloading archive payload url=%s", url)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.read()

    @staticmethod
    def _read_file(path: str) -> bytes:
        with open(path, "rb") as handle:
            return handle.read()

    def _build_filename(self, payload: ArchivePayload) -> str:
        if payload.filename:
            return payload.filename
        if payload.content_type == "video":
            return "video.mp4"
        mime = payload.mime_type or "image/png"
        suffix = mimetypes.guess_extension(mime) or ".png"
        if suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            suffix = ".png"
        return f"photo{suffix}"

    def _build_caption(self, payload: ArchivePayload) -> tuple[str, int]:
        model = payload.model_name or "-"
        prompt = (payload.prompt or "").strip()
        user_part = f"@{payload.username}" if payload.username else "аноним"
        if payload.user_id is not None:
            user_part = f"{user_part} (id {payload.user_id})"
        base_model = f"Модель: {self._escape_markdown(model)}"
        base_user = f"Пользователь: {self._escape_markdown(user_part)}"

        truncated = False
        current_prompt = prompt
        while True:
            suffix = "…" if truncated and current_prompt else ""
            prompt_value = current_prompt + suffix if current_prompt else "-"
            line_prompt = f"Промпт: {self._escape_markdown(prompt_value)}"
            caption = "\n".join([line_prompt, base_model, base_user])
            if len(caption) <= _MAX_CAPTION_LENGTH or not current_prompt:
                if len(caption) > _MAX_CAPTION_LENGTH:
                    trimmed = caption[: _MAX_CAPTION_LENGTH - 1]
                    while trimmed.endswith("\\") and not trimmed.endswith("\\\\"):
                        trimmed = trimmed[:-1]
                    caption = trimmed + "…"
                return caption, len(caption)
            overflow = len(caption) - _MAX_CAPTION_LENGTH
            cut = max(1, overflow)
            current_prompt = current_prompt[:-cut].rstrip()
            truncated = True

    @staticmethod
    def _escape_markdown(value: str) -> str:
        if not value:
            value = "-"
        text = value.replace("\\", "\\\\")
        pattern = re.compile(rf"([{re.escape(_MD_V2_SPECIAL_CHARS)}])")
        text = pattern.sub(r"\\\\\1", text)
        text = text.replace("\n", "\\n").replace("\r", "")
        return text

    def _map_error_short(self, exc: TelegramAPIError) -> str:
        message = getattr(exc, "message", "") or exc.__class__.__name__
        lowered = message.lower()
        if isinstance(exc, RetryAfter):
            return "429"
        if isinstance(exc, ChatNotFound):
            return "404"
        if isinstance(exc, Unauthorized):
            return "403"
        if isinstance(exc, CantTalkWithBot):
            return "403"
        if isinstance(exc, BadRequest):
            if "file is too big" in lowered:
                return "too_large"
            if "message is not modified" in lowered:
                return "bad_request"
            if "wrong file identifier" in lowered:
                return "bad_file_id"
            if "not enough rights" in lowered:
                return "no_admin"
            return "400"
        return exc.__class__.__name__.lower()

    @staticmethod
    def _network_error_short(exc: BaseException) -> str:
        if isinstance(exc, asyncio.TimeoutError):
            return "timeout"
        if isinstance(exc, aiohttp.ClientResponseError):
            return f"http_{exc.status}"
        if isinstance(exc, aiohttp.ClientConnectorError):
            return "connect"
        if isinstance(exc, TelegramServerError):
            return "telegram_server"
        if isinstance(exc, NetworkError):
            return "network"
        return exc.__class__.__name__.lower()

    async def _log_attempt(
        self,
        payload: ArchivePayload,
        *,
        archive_status: str,
        channel_message_id: Optional[int] = None,
        caption_len: Optional[int] = None,
        attempts: int = 0,
        error_short: Optional[str] = None,
    ) -> None:
        record = ArchiveLogRecord(
            ts=datetime.utcnow(),
            corr_id=payload.corr_id,
            archive_status=archive_status,
            channel_id=self._channel_id,
            message_id=channel_message_id,
            content_type=payload.content_type,
            model_name=payload.model_name or "",
            username=payload.username or "",
            caption_len=caption_len or 0,
            file_size=payload.resolved_file_size() or 0,
            attempts=attempts,
            error_short=error_short or "",
            user_id=payload.user_id,
            duration_seconds=payload.duration_seconds,
        )
        try:
            await self._db.log_archive_record(record)
        except Exception:  # pragma: no cover - best effort logging
            log.exception("Failed to persist archive log corr_id=%s", payload.corr_id)


__all__ = ["ArchivePayload", "ArchivePublisher"]
