"""Utilities for broadcasting user-facing errors to the archive channel."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from html import escape as html_escape
from typing import Mapping, Optional

from aiogram import Bot

from config import Config
from db import ErrorLogRecord

log = logging.getLogger(__name__)


def _shorten(value: str, limit: int = 400) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _format_bool(value: bool) -> str:
    return "Да" if value else "Нет"


@dataclass(slots=True)
class _MessageContext:
    """Input data required to compose an error message."""

    record: ErrorLogRecord
    context: str
    environment: str
    extra: Mapping[str, object]


class ErrorReporter:
    """Send detailed error reports to the configured archive channel."""

    def __init__(self, *, bot: Bot, config: Config) -> None:
        self._bot = bot
        self._channel_id = config.archive_channel_id
        self._environment = (config.environment or "dev").lower()
        self._lock = asyncio.Lock()

    async def report(
        self,
        record: ErrorLogRecord,
        *,
        context: str,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        """Schedule sending of the error details to the archive channel."""

        if not self._channel_id:
            return
        message_context = _MessageContext(
            record=record,
            context=context or "unknown",
            environment=self._environment,
            extra=dict(extra or {}),
        )
        async with self._lock:
            await self._send(message_context)

    async def _send(self, ctx: _MessageContext) -> None:
        try:
            text = self._build_message(ctx)
            await self._bot.send_message(self._channel_id, text)
        except Exception:  # pragma: no cover - Telegram API interaction
            log.exception("Failed to send error notification to archive channel")

    def _build_message(self, ctx: _MessageContext) -> str:
        record = ctx.record
        ts_value = record.ts if isinstance(record.ts, datetime) else None
        timestamp = ts_value.strftime("%Y-%m-%d %H:%M:%S") if ts_value else "-"
        username = record.username or "-"
        user_label = f"{html_escape(username)} (ID: {record.user_id})"
        stage = record.stage or "-"
        scope = record.error_scope or "-"
        base_lines = [
            "<b>⚠️ Ошибка обработки запроса</b>",
            f"<b>Источник:</b> {html_escape(ctx.context)}",
            f"<b>Среда:</b> {html_escape(ctx.environment)}",
            f"<b>Время:</b> {html_escape(timestamp)}",
            f"<b>Пользователь:</b> {user_label}",
        ]
        if record.corr_id:
            base_lines.append(f"<b>Corr ID:</b> {html_escape(record.corr_id)}")
        if record.job_id:
            base_lines.append(f"<b>Job ID:</b> {html_escape(record.job_id)}")
        if record.model:
            base_lines.append(f"<b>Модель:</b> {html_escape(record.model)}")
        if record.size:
            base_lines.append(f"<b>Размер:</b> {html_escape(record.size)}")
        error_type = record.error_type or "-"
        base_lines.append(f"<b>Тип ошибки:</b> {html_escape(str(error_type))}")
        if record.status_code is not None:
            base_lines.append(f"<b>Status code:</b> {html_escape(str(record.status_code))}")
        base_lines.append(f"<b>Этап:</b> {html_escape(stage)} / {html_escape(scope)}")
        short_message = record.error_msg_short or "-"
        base_lines.append(f"<b>Сообщение:</b> {html_escape(_shorten(short_message, 600))}")
        refund_status = _format_bool(record.refunded)
        base_lines.append(f"<b>Возврат кредитов:</b> {html_escape(refund_status)}")
        if record.reason:
            base_lines.append(f"<b>Причина:</b> {html_escape(record.reason)}")
        if record.provider_error_code:
            base_lines.append(
                f"<b>Код провайдера:</b> {html_escape(str(record.provider_error_code))}"
            )
        if record.provider_error_message:
            base_lines.append(
                "<b>Сообщение провайдера:</b> "
                f"{html_escape(_shorten(str(record.provider_error_message), 600))}"
            )
        if record.preflight_blocked or record.preflight_reason:
            reason = record.preflight_reason or "-"
            base_lines.append(f"<b>Preflight:</b> блокировано — {html_escape(reason)}")
        if record.auto_sanitized and record.sanitized_prompt:
            base_lines.append(
                "<b>Санитарный промпт:</b> "
                f"{html_escape(_shorten(record.sanitized_prompt, 400))}"
            )
        elif record.sanitized_prompt:
            base_lines.append(
                "<b>Промпт:</b> "
                f"{html_escape(_shorten(record.sanitized_prompt, 400))}"
            )

        extras = self._format_extra(ctx.extra)
        if extras:
            base_lines.append("<b>Дополнительно:</b>")
            base_lines.extend(extras)

        return "\n".join(base_lines)

    def _format_extra(self, extra: Mapping[str, object]) -> list[str]:
        if not extra:
            return []
        lines: list[str] = []
        for key, value in extra.items():
            if value in (None, ""):
                continue
            try:
                if isinstance(value, (str, int, float, bool)):
                    text = str(value)
                else:
                    text = json.dumps(value, ensure_ascii=False)
            except Exception:
                text = str(value)
            lines.append(f"• {html_escape(str(key))}: {html_escape(_shorten(text, 400))}")
        return lines


__all__ = ["ErrorReporter"]
