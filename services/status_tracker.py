from __future__ import annotations

import asyncio
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Deque, Dict, Optional, Sequence

from aiogram import Bot
from aiogram.utils import exceptions as tg_exceptions

from observability import log_event

if TYPE_CHECKING:  # pragma: no cover
    from db import Database, GenerationJobRecord


_DEFAULT_PHRASES: Sequence[str] = (
    "⚙️ Генерирую…",
    "⏳ Ещё немного…",
    "🎞️ Собираю кадры…",
    "🧩 Склеиваю сцену…",
    "🔍 Проверяю качество…",
)

_ACTION_BY_CONTENT = {
    "image": "upload_photo",
    "photo": "upload_photo",
    "video": "upload_video",
}


@dataclass
class _StatusState:
    job_id: str
    user_id: int
    chat_id: int
    message_id: int
    content_type: str
    phrase_index: int
    last_text: str
    active: bool = True
    next_edit_at: float = 0.0
    next_action_at: float = 0.0
    delete_task: Optional[asyncio.Task[None]] = None


class StatusMessageManager:
    """Manage a single live status message per job."""

    def __init__(
        self,
        *,
        phrases: Sequence[str] = _DEFAULT_PHRASES,
        edit_interval: tuple[float, float] = (6.0, 10.0),
        action_interval: tuple[float, float] = (5.0, 10.0),
        delete_interval: tuple[float, float] = (60.0, 300.0),
    ) -> None:
        self._phrases: Sequence[str] = tuple(phrases) or _DEFAULT_PHRASES
        self._edit_interval = edit_interval
        self._action_interval = action_interval
        self._delete_interval = delete_interval
        self._states: Dict[str, _StatusState] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._active_by_user: Dict[int, set[str]] = defaultdict(set)
        self._edit_history: Deque[float] = deque()

    async def ensure_started(
        self,
        *,
        bot: Bot,
        db: "Database",
        job: "GenerationJobRecord",
    ) -> _StatusState:
        state = self._states.get(job.id)
        if state is not None:
            return state
        lock = self._locks.setdefault(job.id, asyncio.Lock())
        async with lock:
            state = self._states.get(job.id)
            if state is not None:
                return state
            message_id = job.status_message_id
            phrase_index = job.status_message_index or 0
            last_text = ""
            now = time.monotonic()
            content_type = (job.content_type or "video").lower()
            if message_id is None:
                initial_phrase = self._phrases[0]
                try:
                    message = await bot.send_message(job.user_id, initial_phrase)
                except Exception:  # pragma: no cover - external dependency
                    raise
                message_id = message.message_id
                phrase_index = 1 % len(self._phrases)
                last_text = initial_phrase
                self._active_by_user[job.user_id].add(job.id)
                await db.update_job_fields(
                    job.id,
                    status=job.status,
                    status_message_id=message_id,
                    status_message_index=phrase_index,
                    status_message_updated_at=datetime.now(timezone.utc),
                )
                job.status_message_id = message_id
                job.status_message_index = phrase_index
                job.status_message_updated_at = datetime.now(timezone.utc)
            else:
                last_text = ""
                self._active_by_user[job.user_id].add(job.id)
            state = _StatusState(
                job_id=job.id,
                user_id=job.user_id,
                chat_id=job.user_id,
                message_id=message_id,
                content_type=content_type,
                phrase_index=phrase_index,
                last_text=last_text,
                next_edit_at=now + self._jitter(self._edit_interval),
                next_action_at=now + self._jitter(self._action_interval),
            )
            self._states[job.id] = state
            return state

    async def tick(
        self,
        *,
        bot: Bot,
        db: "Database",
        job: "GenerationJobRecord",
    ) -> None:
        state = await self.ensure_started(bot=bot, db=db, job=job)
        if not state.active:
            return
        await self._maybe_send_action(bot, state)
        await self._maybe_rotate(bot, db, job, state)

    async def mark_sending(
        self,
        *,
        bot: Bot,
        db: "Database",
        job: "GenerationJobRecord",
        text: str,
    ) -> None:
        state = await self.ensure_started(bot=bot, db=db, job=job)
        state.active = False
        await self._edit_text(bot, db, job, state, text, force=True)

    async def mark_completed(
        self,
        *,
        bot: Bot,
        db: "Database",
        job: "GenerationJobRecord",
        text: str,
        schedule_deletion: bool = True,
    ) -> None:
        state = await self.ensure_started(bot=bot, db=db, job=job)
        state.active = False
        self._active_by_user[state.user_id].discard(job.id)
        await self._edit_text(bot, db, job, state, text, force=True)
        if schedule_deletion:
            self._schedule_deletion(bot, state)

    async def mark_failed(
        self,
        *,
        bot: Bot,
        db: "Database",
        job: "GenerationJobRecord",
        text: str,
    ) -> None:
        state = await self.ensure_started(bot=bot, db=db, job=job)
        state.active = False
        self._active_by_user[state.user_id].discard(job.id)
        await self._edit_text(bot, db, job, state, text, force=True)

    def cancel(self, job_id: str) -> None:
        state = self._states.pop(job_id, None)
        if state and state.delete_task:
            state.delete_task.cancel()
        lock = self._locks.pop(job_id, None)
        if lock and lock.locked():  # pragma: no cover - safety guard
            lock.release()

    def active_statuses_for_user(self, user_id: int) -> int:
        return len(self._active_by_user.get(user_id, set()))

    async def _maybe_rotate(
        self,
        bot: Bot,
        db: "Database",
        job: "GenerationJobRecord",
        state: _StatusState,
    ) -> None:
        now = time.monotonic()
        if now < state.next_edit_at:
            return
        phrase = self._phrases[state.phrase_index % len(self._phrases)]
        state.phrase_index = (state.phrase_index + 1) % len(self._phrases)
        await self._edit_text(bot, db, job, state, phrase)

    async def _maybe_send_action(self, bot: Bot, state: _StatusState) -> None:
        now = time.monotonic()
        if now < state.next_action_at:
            return
        action = _ACTION_BY_CONTENT.get(state.content_type, "upload_video")
        try:
            await bot.send_chat_action(state.chat_id, action)
        except tg_exceptions.TelegramAPIError:  # pragma: no cover - network call
            pass
        state.next_action_at = now + self._jitter(self._action_interval)

    async def _edit_text(
        self,
        bot: Bot,
        db: "Database",
        job: "GenerationJobRecord",
        state: _StatusState,
        text: str,
        *,
        force: bool = False,
    ) -> None:
        if not force and text == state.last_text:
            return
        try:
            await bot.edit_message_text(text, state.chat_id, state.message_id)
        except tg_exceptions.MessageNotModified:
            state.last_text = text
        except (tg_exceptions.MessageToEditNotFound, tg_exceptions.MessageCantBeEdited):
            state.active = False
        except tg_exceptions.TelegramAPIError:  # pragma: no cover - network call
            pass
        else:
            state.last_text = text
            self._record_edit(job, state)
            await db.update_job_fields(
                job.id,
                status=job.status,
                status_message_index=state.phrase_index,
                status_message_updated_at=datetime.now(timezone.utc),
            )
            job.status_message_index = state.phrase_index
            job.status_message_updated_at = datetime.now(timezone.utc)
        finally:
            now = time.monotonic()
            state.next_edit_at = now + self._jitter(self._edit_interval)

    def _schedule_deletion(self, bot: Bot, state: _StatusState) -> None:
        if state.delete_task is not None:
            return
        delay = self._jitter(self._delete_interval)
        state.delete_task = asyncio.create_task(self._delete_later(bot, state, delay))

    async def _delete_later(self, bot: Bot, state: _StatusState, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            await bot.delete_message(state.chat_id, state.message_id)
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancel
            raise
        except tg_exceptions.TelegramAPIError:  # pragma: no cover - network call
            pass
        finally:
            self._states.pop(state.job_id, None)
            self._locks.pop(state.job_id, None)

    def _record_edit(self, job: "GenerationJobRecord", state: _StatusState) -> None:
        now = time.time()
        self._edit_history.append(now)
        while self._edit_history and now - self._edit_history[0] > 60:
            self._edit_history.popleft()
        edits_per_minute = len(self._edit_history)
        active_statuses = self.active_statuses_for_user(state.user_id)
        log_event(
            level="INFO",
            event="status_edit",
            corr_id=job.corr_id,
            job_id=job.id,
            user_id=job.user_id,
            username=job.username,
            model=job.model,
            provider="status",
            size=job.size,
            extra={
                "edits_per_minute": edits_per_minute,
                "active_statuses_user": active_statuses,
            },
        )

    @staticmethod
    def _jitter(window: tuple[float, float]) -> float:
        start, end = window
        return random.uniform(start, end)


__all__ = ["StatusMessageManager"]
