"""Test helper to provide minimal aiogram/yookassa stubs for handler imports."""
from __future__ import annotations

import sys
import types
from typing import Any


def ensure_aiogram_stub() -> None:
    """Install lightweight aiogram and yookassa stubs for tests."""

    if "aiogram" in sys.modules:
        types_module = sys.modules.get("aiogram.types")
        if types_module and hasattr(types_module, "ContentType"):
            return

    aiogram = types.ModuleType("aiogram")
    aiogram.__path__ = []  # mark as package
    aiogram.Bot = type("Bot", (), {})
    aiogram.Dispatcher = type("Dispatcher", (), {})

    dispatcher = types.ModuleType("aiogram.dispatcher")
    dispatcher.Dispatcher = aiogram.Dispatcher
    dispatcher.FSMContext = type("FSMContext", (), {})

    filters = types.ModuleType("aiogram.dispatcher.filters")
    filters.Command = type("Command", (), {})
    filters.CommandStart = type("CommandStart", (), {})

    state = types.ModuleType("aiogram.dispatcher.filters.state")

    class _State:
        def __init__(self, name: str | None = None) -> None:
            self.state = name or "state"

        async def set(self) -> None:  # pragma: no cover - simple stub
            return None

    class _StatesGroup:
        pass

    state.State = _State
    state.StatesGroup = _StatesGroup

    types_module = types.ModuleType("aiogram.types")

    class _Base:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
            for key, value in kwargs.items():
                setattr(self, key, value)

    class CallbackQuery(_Base):
        pass

    class InlineKeyboardButton(_Base):
        def __init__(self, *, text: str, callback_data: str | None = None, url: str | None = None) -> None:
            super().__init__(text=text, callback_data=callback_data, url=url)

    class InlineKeyboardMarkup(_Base):
        def __init__(self, *, inline_keyboard: list[list[Any]]) -> None:  # type: ignore[override]
            super().__init__(inline_keyboard=inline_keyboard)

    class InputFile(_Base):
        pass

    class InputMediaPhoto(_Base):
        pass

    class KeyboardButton(_Base):
        def __init__(self, *, text: str) -> None:
            super().__init__(text=text)

    class Message(_Base):
        pass

    class ReplyKeyboardMarkup(_Base):
        def __init__(self, *, keyboard: list[list[Any]], resize_keyboard: bool = False) -> None:  # type: ignore[override]
            super().__init__(keyboard=keyboard, resize_keyboard=resize_keyboard)

    class ReplyKeyboardRemove(_Base):
        def __init__(self, *, selective: bool | None = None) -> None:  # type: ignore[override]
            super().__init__(selective=selective)

    types_module.CallbackQuery = CallbackQuery
    types_module.InlineKeyboardButton = InlineKeyboardButton
    types_module.InlineKeyboardMarkup = InlineKeyboardMarkup
    types_module.InputFile = InputFile
    types_module.InputMediaPhoto = InputMediaPhoto
    types_module.KeyboardButton = KeyboardButton
    types_module.Message = Message
    types_module.ReplyKeyboardMarkup = ReplyKeyboardMarkup
    types_module.ReplyKeyboardRemove = ReplyKeyboardRemove

    class _ContentType:
        ANY = "*"
        TEXT = "text"
        SUCCESSFUL_PAYMENT = "successful_payment"
        INVOICE = "invoice"
        MESSAGE_AUTO_DELETE_TIMER_CHANGED = "message_auto_delete_timer_changed"
        MIGRATE_FROM_CHAT_ID = "migrate_from_chat_id"
        MIGRATE_TO_CHAT_ID = "migrate_to_chat_id"
        NEW_CHAT_MEMBERS = "new_chat_members"
        LEFT_CHAT_MEMBER = "left_chat_member"
        NEW_CHAT_TITLE = "new_chat_title"
        NEW_CHAT_PHOTO = "new_chat_photo"
        DELETE_CHAT_PHOTO = "delete_chat_photo"
        GROUP_CHAT_CREATED = "group_chat_created"
        SUPERGROUP_CHAT_CREATED = "supergroup_chat_created"
        CHANNEL_CHAT_CREATED = "channel_chat_created"
        VENUE = "venue"

    types_module.ContentType = _ContentType()

    utils_module = types.ModuleType("aiogram.utils")
    exceptions_module = types.ModuleType("aiogram.utils.exceptions")

    class _RetryAfter(Exception):
        def __init__(self, timeout: float | None = None, *args, **kwargs) -> None:
            super().__init__(*args)
            self.timeout = timeout

    class _BadRequest(Exception):
        def __init__(self, message: str | None = None, *args, **kwargs) -> None:
            super().__init__(message or "", *args)

    class _ChatNotFound(Exception):
        pass

    class _NetworkError(Exception):
        pass

    class _TelegramAPIError(Exception):
        def __init__(self, message: str | None = None, *args, **kwargs) -> None:
            super().__init__(message or "", *args)

    class _Unauthorized(Exception):
        pass

    class _CantTalkWithBot(Exception):
        pass

    class _BotBlocked(Exception):
        pass

    class _UserDeactivated(Exception):
        pass

    exceptions_module.BadRequest = _BadRequest
    exceptions_module.ChatNotFound = _ChatNotFound
    exceptions_module.NetworkError = _NetworkError
    exceptions_module.RetryAfter = _RetryAfter
    exceptions_module.TelegramAPIError = _TelegramAPIError
    exceptions_module.Unauthorized = _Unauthorized
    exceptions_module.CantTalkWithBot = _CantTalkWithBot
    exceptions_module.BotBlocked = _BotBlocked
    exceptions_module.UserDeactivated = _UserDeactivated

    sys.modules["aiogram"] = aiogram
    sys.modules["aiogram.dispatcher"] = dispatcher
    sys.modules["aiogram.dispatcher.filters"] = filters
    sys.modules["aiogram.dispatcher.filters.state"] = state
    sys.modules["aiogram.types"] = types_module
    sys.modules["aiogram.utils"] = utils_module
    sys.modules["aiogram.utils.exceptions"] = exceptions_module

    yookassa_module = types.ModuleType("yookassa")
    yookassa_module.Configuration = type("Configuration", (), {})
    yookassa_module.Payment = type("Payment", (), {})
    sys.modules.setdefault("yookassa", yookassa_module)

    yookassa_client_module = types.ModuleType("yookassa_client")
    yookassa_client_module.create_payment = lambda *args, **kwargs: None
    sys.modules.setdefault("yookassa_client", yookassa_client_module)

