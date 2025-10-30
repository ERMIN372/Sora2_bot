"""Test helper to provide minimal aiogram/yookassa stubs for handler imports."""
from __future__ import annotations

import sys
import types


def ensure_aiogram_stub() -> None:
    """Install lightweight aiogram and yookassa stubs for tests."""

    if "aiogram" in sys.modules:
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
    state.State = type("State", (), {})
    state.StatesGroup = type("StatesGroup", (), {})

    types_module = types.ModuleType("aiogram.types")
    placeholder = type("_Type", (), {})
    types_module.CallbackQuery = placeholder
    types_module.InlineKeyboardButton = placeholder
    types_module.InlineKeyboardMarkup = placeholder
    types_module.InputFile = placeholder
    types_module.KeyboardButton = placeholder
    types_module.Message = placeholder
    types_module.ReplyKeyboardMarkup = placeholder

    utils_module = types.ModuleType("aiogram.utils")
    exceptions_module = types.ModuleType("aiogram.utils.exceptions")
    for name in (
        "BadRequest",
        "ChatNotFound",
        "NetworkError",
        "RetryAfter",
        "TelegramAPIError",
        "Unauthorized",
        "CantTalkWithBot",
    ):
        setattr(exceptions_module, name, type(name, (), {}))

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

