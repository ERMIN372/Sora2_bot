import pathlib
import sys
import types
from typing import TYPE_CHECKING, Any, Callable, Dict

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from config import Config

try:  # pragma: no cover - prefer real dependency
    import dotenv  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - test environment shim
    dotenv_stub = types.ModuleType("dotenv")

    def load_dotenv(*_: Any, **__: Any) -> None:
        return None

    dotenv_stub.load_dotenv = load_dotenv
    sys.modules["dotenv"] = dotenv_stub

try:  # pragma: no cover - prefer real dependency
    from aiogram import Bot as _AiogramBot  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - test environment shim
    aiogram_stub = types.ModuleType("aiogram")

    class Bot:  # pragma: no cover - minimal stub for tests
        def __init__(self, *_: Any, **__: Any) -> None:
            return None

        async def send_message(self, *_: Any, **__: Any) -> None:
            return None

    class Dispatcher:  # pragma: no cover - minimal stub for tests
        def __init__(self, *_: Any, **__: Any) -> None:
            return None

    aiogram_stub.Bot = Bot
    aiogram_stub.Dispatcher = Dispatcher
    sys.modules["aiogram"] = aiogram_stub

    dispatcher_stub = types.ModuleType("aiogram.dispatcher")

    class FSMContext:  # pragma: no cover - minimal stub
        def __init__(self, *_: Any, **__: Any) -> None:
            return None

    dispatcher_stub.FSMContext = FSMContext
    sys.modules["aiogram.dispatcher"] = dispatcher_stub

    filters_stub = types.ModuleType("aiogram.dispatcher.filters")

    class Command:  # pragma: no cover - minimal stub
        def __init__(self, *_: Any, **__: Any) -> None:
            return None

    class CommandStart(Command):  # pragma: no cover - minimal stub
        pass

    filters_stub.Command = Command
    filters_stub.CommandStart = CommandStart
    sys.modules["aiogram.dispatcher.filters"] = filters_stub

    state_stub = types.ModuleType("aiogram.dispatcher.filters.state")

    class State:  # pragma: no cover - minimal stub
        def __init__(self, *_: Any, **__: Any) -> None:
            return None

    class StatesGroup:  # pragma: no cover - minimal stub
        pass

    state_stub.State = State
    state_stub.StatesGroup = StatesGroup
    sys.modules["aiogram.dispatcher.filters.state"] = state_stub

    types_stub = types.ModuleType("aiogram.types")

    class _BaseType:  # pragma: no cover - minimal stub
        def __init__(self, *_: Any, **__: Any) -> None:
            return None

    class Message(_BaseType):
        pass

    class CallbackQuery(_BaseType):
        pass

    class InlineKeyboardButton(_BaseType):
        pass

    class InlineKeyboardMarkup(_BaseType):
        pass

    class KeyboardButton(_BaseType):
        pass

    class ReplyKeyboardMarkup(_BaseType):
        pass

    class ReplyKeyboardRemove(_BaseType):
        pass

    class InputFile(_BaseType):
        pass

    types_stub.Message = Message
    types_stub.CallbackQuery = CallbackQuery
    types_stub.InlineKeyboardButton = InlineKeyboardButton
    types_stub.InlineKeyboardMarkup = InlineKeyboardMarkup
    types_stub.KeyboardButton = KeyboardButton
    types_stub.ReplyKeyboardMarkup = ReplyKeyboardMarkup
    types_stub.ReplyKeyboardRemove = ReplyKeyboardRemove
    types_stub.InputFile = InputFile
    sys.modules["aiogram.types"] = types_stub

    utils_stub = types.ModuleType("aiogram.utils")
    exceptions_stub = types.ModuleType("aiogram.utils.exceptions")

    class AiogramError(Exception):  # pragma: no cover - minimal stub
        pass

    class TelegramAPIError(AiogramError):
        pass

    class BadRequest(TelegramAPIError):
        pass

    class ChatNotFound(TelegramAPIError):
        pass

    class NetworkError(TelegramAPIError):
        pass

    class RetryAfter(TelegramAPIError):
        def __init__(self, timeout: float = 0, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args)
            self.timeout = timeout

    class Unauthorized(TelegramAPIError):
        pass

    class CantTalkWithBot(TelegramAPIError):
        pass

    exceptions_stub.AiogramError = AiogramError
    exceptions_stub.TelegramAPIError = TelegramAPIError
    exceptions_stub.BadRequest = BadRequest
    exceptions_stub.ChatNotFound = ChatNotFound
    exceptions_stub.NetworkError = NetworkError
    exceptions_stub.RetryAfter = RetryAfter
    exceptions_stub.Unauthorized = Unauthorized
    exceptions_stub.CantTalkWithBot = CantTalkWithBot
    utils_stub.exceptions = exceptions_stub
    sys.modules["aiogram.utils"] = utils_stub
    sys.modules["aiogram.utils.exceptions"] = exceptions_stub

try:  # pragma: no cover - prefer real dependency
    import aiohttp  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - test environment shim
    aiohttp_stub = types.ModuleType("aiohttp")

    class ClientTimeout:
        def __init__(self, total=None, sock_connect=None, sock_read=None):
            self.total = total
            self.sock_connect = sock_connect
            self.sock_read = sock_read

    class ClientSession:
        def __init__(self, *_, timeout=None, **__):
            self.timeout = timeout
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    class ClientError(Exception):
        pass

    aiohttp_stub.ClientTimeout = ClientTimeout
    aiohttp_stub.ClientSession = ClientSession
    aiohttp_stub.ClientError = ClientError
    sys.modules["aiohttp"] = aiohttp_stub

try:  # pragma: no cover - prefer real dependency
    import google.genai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - test environment shim
    google_mod = types.ModuleType("google")
    sys.modules.setdefault("google", google_mod)

    genai_mod = types.ModuleType("google.genai")
    sys.modules["google.genai"] = genai_mod

    class HttpOptions:
        def __init__(self, timeout=None):
            self.timeout = timeout

    api_client_mod = types.ModuleType("google.genai._api_client")
    api_client_mod.HttpOptions = HttpOptions
    sys.modules["google.genai._api_client"] = api_client_mod

    class APIError(Exception):
        def __init__(self, message="", code=500, status="ERROR", response=None):
            super().__init__(message)
            self.code = code
            self.status = status
            self.response = response

    errors_mod = types.ModuleType("google.genai.errors")
    errors_mod.APIError = APIError
    sys.modules["google.genai.errors"] = errors_mod

    class _BaseModel:
        def __init__(self, **fields: Any) -> None:
            for key, value in fields.items():
                setattr(self, key, value)

        @classmethod
        def model_validate(cls, data: Dict[str, Any]):
            return cls(**data)

        def model_dump(self, exclude_none: bool = False) -> Dict[str, Any]:
            items = self.__dict__.copy()
            if exclude_none:
                items = {k: v for k, v in items.items() if v is not None}
            return items

    class Image(_BaseModel):
        def __init__(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> None:
            super().__init__(image_bytes=image_bytes, mime_type=mime_type)

    class Video(_BaseModel):
        def __init__(self, uri: str | None = None, mime_type: str | None = None, video_bytes: bytes | None = None) -> None:
            super().__init__(uri=uri, mime_type=mime_type, video_bytes=video_bytes)

    class GeneratedVideo(_BaseModel):
        def __init__(self, video: Video | None = None) -> None:
            super().__init__(video=video)

    class GenerateVideosResult(_BaseModel):
        def __init__(self, generated_videos=None):
            super().__init__(generated_videos=generated_videos or [])

    class GenerateVideosSource(_BaseModel):
        def __init__(self, prompt: str | None = None, image: Image | None = None) -> None:
            super().__init__(prompt=prompt, image=image)

    class GenerateVideosConfig(_BaseModel):
        pass

    class GenerateVideosOperation(_BaseModel):
        def __init__(self, name: str = "", done: bool = False, error=None, result=None, response=None):
            super().__init__(name=name, done=done, error=error, result=result, response=response)

    class HarmCategory:
        HARM_CATEGORY_HARASSMENT = "HARASSMENT"
        HARM_CATEGORY_HATE_SPEECH = "HATE_SPEECH"
        HARM_CATEGORY_SEXUALLY_EXPLICIT = "SEXUALLY_EXPLICIT"
        HARM_CATEGORY_DANGEROUS_CONTENT = "DANGEROUS_CONTENT"
        HARM_CATEGORY_CIVIC_INTEGRITY = "CIVIC_INTEGRITY"

    class HarmBlockThreshold:
        BLOCK_NONE = "BLOCK_NONE"
        BLOCK_ONLY_HIGH = "BLOCK_ONLY_HIGH"

    class SafetySetting(_BaseModel):
        def __init__(self, category=None, threshold=None):
            super().__init__(category=category, threshold=threshold)

    types_mod = types.ModuleType("google.genai.types")
    types_mod.Image = Image
    types_mod.Video = Video
    types_mod.GeneratedVideo = GeneratedVideo
    types_mod.GenerateVideosResult = GenerateVideosResult
    types_mod.GenerateVideosSource = GenerateVideosSource
    class GenerateImagesConfig(_BaseModel):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)

    types_mod.GenerateVideosConfig = GenerateVideosConfig
    types_mod.GenerateVideosOperation = GenerateVideosOperation
    types_mod.GenerateImagesConfig = GenerateImagesConfig
    types_mod.HarmCategory = HarmCategory
    types_mod.HarmBlockThreshold = HarmBlockThreshold
    types_mod.SafetySetting = SafetySetting
    types_mod.HTTPOptions = HttpOptions
    types_mod.HttpOptions = HttpOptions
    sys.modules["google.genai.types"] = types_mod

    oauth2_mod = types.ModuleType("google.oauth2")
    service_account_mod = types.ModuleType("google.oauth2.service_account")

    class Credentials:  # pragma: no cover - stub
        @classmethod
        def from_service_account_info(cls, *_args: Any, **_kwargs: Any) -> "Credentials":
            return cls()

    service_account_mod.Credentials = Credentials
    oauth2_mod.service_account = service_account_mod
    sys.modules["google.oauth2"] = oauth2_mod
    sys.modules["google.oauth2.service_account"] = service_account_mod

try:  # pragma: no cover - prefer real dependency
    import tenacity  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - test environment shim
    tenacity_stub = types.ModuleType("tenacity")

    def _identity_decorator(fn=None, **_kwargs):
        def _wrap(func):
            return func

        if fn is not None:
            return fn
        return _wrap

    tenacity_stub.retry = _identity_decorator
    tenacity_stub.retry_if_exception_type = lambda *_, **__: None
    tenacity_stub.stop_after_attempt = lambda *_args, **_kwargs: None
    tenacity_stub.wait_exponential = lambda *_args, **_kwargs: None
    sys.modules["tenacity"] = tenacity_stub

    class _Models:
        def generate_videos(self, *args: Any, **kwargs: Any):  # pragma: no cover - not used in tests
            raise NotImplementedError

    class _Operations:
        def get(self, operation):  # pragma: no cover - tests stub behaviour
            return operation

    class Client:
        def __init__(self, api_key: str, http_options: HttpOptions | None = None) -> None:
            self.api_key = api_key
            self.http_options = http_options
            self.models = _Models()
            self.operations = _Operations()

    genai_mod.Client = Client
    genai_mod.types = types_mod
    genai_mod.errors = errors_mod

try:  # pragma: no cover - prefer real dependency
    import gspread  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - test environment shim
    gspread_mod = types.ModuleType("gspread")

    class _GSpreadBase:
        pass

    class Client(_GSpreadBase):
        pass

    class Worksheet(_GSpreadBase):
        pass

    class Spreadsheet(_GSpreadBase):
        pass

    class _Exceptions:
        APIError = type("APIError", (Exception,), {})
        WorksheetNotFound = type("WorksheetNotFound", (Exception,), {})

    def authorize(credentials):  # pragma: no cover - tests do not use
        return Client()

    gspread_mod.Client = Client
    gspread_mod.Worksheet = Worksheet
    gspread_mod.Spreadsheet = Spreadsheet
    gspread_mod.exceptions = _Exceptions()
    gspread_mod.authorize = authorize
    sys.modules["gspread"] = gspread_mod


@pytest.fixture
def make_openai_video_config() -> Callable[..., "Config"]:
    """Factory fixture for constructing ``Config`` objects in tests."""

    def _factory(**overrides: Any) -> "Config":
        from config import Config  # Lazy import to allow dependency shims above

        base: Dict[str, Any] = {
            "bot_token": "test-token",
            "gemini_api_key": "",
            "openai_api_key": "sk-test",
            "sora_api_key": "sk-test",
            "openai_api_base": "https://api.openai.com",
        }
        base.update(overrides)
        return Config(**base)  # type: ignore[arg-type]

    return _factory
