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

from observability import increment_metric, log_event

if TYPE_CHECKING:  # pragma: no cover
    from db import Database, GenerationJobRecord


_DEFAULT_PHRASES: Sequence[str] = (
    "Генерация идёт…",
    "Ещё немного…",
    "Подчищаю кадры…",
    "Почти готово…",
)

_MAX_EDITS = 12
_TERMINAL_STATUSES = {"completed", "failed", "no_media", "refunded", "timeout"}

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
    terminal_state: Optional[str] = None
    closed: bool = False
    edit_count: int = 0


class StatusMessageManager:
    """Manage a single live status message per job."""

    def __init__(
        self,
        *,
        phrases: Sequence[str] = _DEFAULT_PHRASES,
        edit_interval: tuple[float, float] = (5.0, 8.0),
        action_interval: tuple[float, float] = (6.0, 12.0),
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
                initial_phrase, next_index = self._select_next_phrase(
                    job.status_message_index or -1
                )
                try:
                    message = await bot.send_message(job.user_id, initial_phrase)
                except Exception:  # pragma: no cover - external dependency
                    raise
                message_id = message.message_id
                phrase_index = next_index
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
        status_value = (job.status or "").lower()
        if status_value in _TERMINAL_STATUSES and not state.closed:
            state.active = False
            state.next_edit_at = float("inf")
            state.next_action_at = float("inf")
            return
        if state.edit_count >= _MAX_EDITS and not state.closed:
            state.active = False
            state.next_edit_at = float("inf")
            state.next_action_at = float("inf")
            return
        if not state.active or state.closed:
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
        await self._finalise_state(
            bot=bot,
            db=db,
            job=job,
            state=state,
            text=text,
            terminal_state="delivered",
            schedule_deletion=schedule_deletion,
        )

    async def mark_failed(
        self,
        *,
        bot: Bot,
        db: "Database",
        job: "GenerationJobRecord",
        text: str,
        terminal_state: str = "failed",
        schedule_deletion: bool = False,
    ) -> None:
        state = await self.ensure_started(bot=bot, db=db, job=job)
        await self._finalise_state(
            bot=bot,
            db=db,
            job=job,
            state=state,
            text=text,
            terminal_state=terminal_state,
            schedule_deletion=schedule_deletion,
        )

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
        if state.closed:
            return
        if state.edit_count >= _MAX_EDITS:
            state.active = False
            state.next_edit_at = float("inf")
            return
        phrase, next_index = self._select_next_phrase(state.phrase_index)
        state.phrase_index = next_index
        await self._edit_text(bot, db, job, state, phrase)

    async def _maybe_send_action(self, bot: Bot, state: _StatusState) -> None:
        now = time.monotonic()
        if now < state.next_action_at:
            return
        if state.closed:
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
        if state.closed and not force:
            increment_metric("status_edits_after_terminal")
            return
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
            if not force:
                state.edit_count += 1
        finally:
            now = time.monotonic()
            if state.closed:
                state.next_edit_at = float("inf")
                state.next_action_at = float("inf")
            elif state.edit_count >= _MAX_EDITS and not force:
                state.active = False
                state.next_edit_at = float("inf")
                state.next_action_at = float("inf")
            else:
                state.next_edit_at = now + self._jitter(self._edit_interval)

    async def _finalise_state(
        self,
        *,
        bot: Bot,
        db: "Database",
        job: "GenerationJobRecord",
        state: _StatusState,
        text: str,
        terminal_state: str,
        schedule_deletion: bool,
    ) -> None:
        state.active = False
        state.closed = True
        state.terminal_state = terminal_state
        self._active_by_user[state.user_id].discard(job.id)
        await self._edit_text(bot, db, job, state, text, force=True)
        state.next_edit_at = float("inf")
        state.next_action_at = float("inf")
        if schedule_deletion:
            self._schedule_deletion(bot, state)
        elif state.delete_task:
            state.delete_task.cancel()
            state.delete_task = None

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

    def _select_next_phrase(self, previous_index: int) -> tuple[str, int]:
        if not self._phrases:
            return "…", previous_index
        indices = list(range(len(self._phrases)))
        if len(indices) > 1 and 0 <= previous_index < len(self._phrases):
            try:
                indices.remove(previous_index)
            except ValueError:  # pragma: no cover - defensive
                pass
        next_index = random.choice(indices) if indices else previous_index
        return self._phrases[next_index], next_index


__all__ = ["StatusMessageManager"]
