from datetime import datetime
import sys
import types


def _ensure_aiogram_stub() -> None:
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


_ensure_aiogram_stub()

from db import GenerationJobRecord
from handlers import _select_remote_video_asset


def _make_job(*, video_url=None, extra=None):
    return GenerationJobRecord(
        id="job-1",
        user_id=1,
        prompt="",
        status="completed",
        video_url=video_url,
        error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        extra=extra or {},
    )


def test_select_remote_asset_accepts_kind_video():
    job = _make_job(
        extra={
            "assets_meta": [
                {
                    "key": "asset_0",
                    "kind": "video",
                    "url": "https://example.com/video.mp4",
                }
            ]
        }
    )

    asset = _select_remote_video_asset(job)

    assert asset is not None
    assert asset.get("url") == "https://example.com/video.mp4"


def test_select_remote_asset_fallback_to_video_url():
    job = _make_job(video_url="https://example.com/direct.mp4", extra={})

    asset = _select_remote_video_asset(job)

    assert asset is not None
    assert asset.get("url") == "https://example.com/direct.mp4"
    assert asset.get("fallback") is True
