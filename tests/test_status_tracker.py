import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from aiogram.utils import exceptions as tg_exceptions


if not hasattr(tg_exceptions, "MessageCantBeEdited"):
    class MessageCantBeEdited(tg_exceptions.TelegramAPIError):
        pass

    tg_exceptions.MessageCantBeEdited = MessageCantBeEdited  # type: ignore[attr-defined]

if not hasattr(tg_exceptions, "MessageToEditNotFound"):
    class MessageToEditNotFound(tg_exceptions.TelegramAPIError):
        pass

    tg_exceptions.MessageToEditNotFound = MessageToEditNotFound  # type: ignore[attr-defined]

if not hasattr(tg_exceptions, "MessageNotModified"):
    class MessageNotModified(tg_exceptions.TelegramAPIError):
        pass

    tg_exceptions.MessageNotModified = MessageNotModified  # type: ignore[attr-defined]

from db import GenerationJobRecord
from services.status_tracker import StatusMessageManager


def run(coro):
    return asyncio.run(coro)


class DummyBot:
    def __init__(self) -> None:
        self.sent_messages = []
        self.edited_messages = []
        self.deleted_messages = []

    async def send_message(self, chat_id: int, text: str):
        message_id = len(self.sent_messages) + 1
        record = {"chat_id": chat_id, "text": text, "message_id": message_id}
        self.sent_messages.append(record)
        return SimpleNamespace(message_id=message_id)

    async def edit_message_text(self, text: str, chat_id: int, message_id: int):
        self.edited_messages.append(
            {"chat_id": chat_id, "message_id": message_id, "text": text}
        )
        raise tg_exceptions.MessageCantBeEdited("can't edit")

    async def send_chat_action(self, chat_id: int, action: str):  # pragma: no cover - noop
        return None

    async def delete_message(self, chat_id: int, message_id: int):
        self.deleted_messages.append({"chat_id": chat_id, "message_id": message_id})


class DummyDB:
    def __init__(self) -> None:
        self.updates = []

    async def update_job_fields(
        self,
        job_id: str,
        *,
        status: str | None = None,
        status_message_id: int | None = None,
        status_message_index: int | None = None,
        status_message_updated_at=None,
    ) -> None:
        self.updates.append(
            {
                "job_id": job_id,
                "status": status,
                "status_message_id": status_message_id,
                "status_message_index": status_message_index,
                "status_message_updated_at": status_message_updated_at,
            }
        )


def _make_job(status: str = "failed") -> GenerationJobRecord:
    now = datetime.now(timezone.utc)
    return GenerationJobRecord(
        id="job-1",
        user_id=123,
        prompt="",
        status=status,
        video_url=None,
        video_id=None,
        error="Ошибка",
        created_at=now,
        updated_at=now,
    )


def test_mark_failed_sends_fallback_message_when_edit_fails():
    manager = StatusMessageManager()
    bot = DummyBot()
    db = DummyDB()
    job = _make_job()

    async def scenario():
        await manager.mark_failed(
            bot=bot,
            db=db,
            job=job,
            text="Генерация не удалась",
            terminal_state="failed",
        )

    run(scenario())

    assert len(bot.sent_messages) == 2
    initial_message, fallback_message = bot.sent_messages
    assert initial_message["text"] != fallback_message["text"]
    assert fallback_message["text"] == "Генерация не удалась"
    assert job.status_message_id == fallback_message["message_id"]
    assert any(
        update["status_message_id"] == fallback_message["message_id"]
        for update in db.updates
    )
    # The original status message should be deleted when possible
    assert any(
        entry["message_id"] == initial_message["message_id"] for entry in bot.deleted_messages
    )
