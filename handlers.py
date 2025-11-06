"""Aiogram handlers for the video generation bot."""
from __future__ import annotations

import asyncio
import base64
import binascii
import fnmatch
import json
import logging
import mimetypes
import time
from dataclasses import dataclass, field
from decimal import Decimal
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Set, Tuple

from aiogram import Bot, Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command, CommandStart
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from aiogram.utils import exceptions as aiogram_exceptions

BadRequest = aiogram_exceptions.BadRequest
ChatNotFound = aiogram_exceptions.ChatNotFound
RetryAfter = aiogram_exceptions.RetryAfter
TelegramAPIError = aiogram_exceptions.TelegramAPIError
Unauthorized = aiogram_exceptions.Unauthorized
InvalidQueryID = getattr(
    aiogram_exceptions,
    "InvalidQueryID",
    TelegramAPIError,
)
CantTalkWithBot = getattr(
    aiogram_exceptions,
    "CantTalkWithBot",
    getattr(
        aiogram_exceptions,
        "CantInitiateConversation",
        TelegramAPIError,
    ),
)

from archive import ArchivePayload, ArchivePublisher
from config import (
    Config,
    PRODUCT_PRICING_UI,
    SORA_SUPPORTED_MODELS,
    TOPUP_UI_PACKS,
    UI_MODEL_LABELS,
)
from db import Database, ErrorLogRecord, GenerationJobRecord
from generation_gate import GateDecision, GenerationRequestGate, normalize_prompt
from i18n import SafeText, escape_html, format_credits, format_prompt, i18n
from jobs import (
    JobQueue,
    VIDEO_TASK_CREATE,
    VIDEO_TASK_DOWNLOAD,
    VIDEO_TASK_REMIX,
    VIDEO_TASK_RETRIEVE,
)
from observability import (
    get_job_history,
    get_last_error,
    get_last_healthcheck,
    increment_metric,
    log_event,
    record_timing_metric,
    should_notify_support,
)
from moderation import policy_message, run_preflight
from providers import ProviderAPIError
from providers.base import BaseProviderClient
from providers.openai_chat import OpenAIChatClient
from providers.openai_video import OpenAIVideoClient
from services.gemini_catalog import list_veo_video_models
from services.gemini_client import get_media_client, get_text_client
from services.gemini_downloader import (
    GeminiConfigurationError,
    GeminiDownloadError,
    GeminiKeyMismatchError,
    download_asset,
)
from services.sora_downloader import SoraDownloadError, download_sora_asset
from services.error_reporter import ErrorReporter
from services.status_tracker import StatusMessageManager
from telegram_files import BufferedInputFile
from utils import build_inline_data_from_telegram_file
import yookassa_client


# Поддерживаемые соотношения сторон
ASPECT_RATIO_OPTIONS: Dict[str, str] = {
    "horizontal": "16:9",
    "vertical": "9:16",
}
DEFAULT_ASPECT_RATIO = ASPECT_RATIO_OPTIONS["horizontal"]
DEFAULT_HD_ENABLED = False

DEFAULT_VIDEO_DURATION = 10
VIDEO_DURATION_OPTIONS: Tuple[int, int, int] = (10, 15, 25)
VIDEO_PRICE_RUB: Dict[int, Decimal] = {
    10: Decimal("119"),
    15: Decimal("169"),
    25: Decimal("269"),
}


log = logging.getLogger(__name__)


SIZE_OPTIONS: Dict[str, str] = {
    "vertical": "720x1280",
    "horizontal": "1280x720",
}


def _resolve_video_size(aspect_ratio: str, hd: bool) -> str:
    if aspect_ratio == ASPECT_RATIO_OPTIONS["vertical"]:
        return "1080x1920" if hd else "720x1280"
    return "1920x1080" if hd else "1280x720"


async def safe_callback_answer(callback: CallbackQuery, *args: Any, **kwargs: Any) -> None:
    """Answer a callback query without raising for stale queries.

    Telegram occasionally retries webhook updates, which means the callback query
    might already be answered by the time we handle it. In that case Telegram
    returns ``InvalidQueryID``. We swallow that specific error to avoid crashing
    the webhook handler while keeping the default behaviour for other
    exceptions.
    """

    try:
        await callback.answer(*args, **kwargs)
    except InvalidQueryID:
        log.debug("Callback query %s already expired", callback.id)

_INLINE_ASSET_MAX_BYTES = 9 * 1024 * 1024
_TELEGRAM_VIDEO_MAX_BYTES = 48 * 1024 * 1024
_TELEGRAM_PHOTO_MAX_BYTES = 10 * 1024 * 1024
_TELEGRAM_DOCUMENT_MAX_BYTES = 2 * 1024 * 1024 * 1024


def _append_checkmark(label: str, selected: bool) -> str:
    return f"{label} ✅" if selected else label

MAX_INLINE_VIDEO_BYTES = 9 * 1024 * 1024

MAIN_MENU_BUTTONS = {
    i18n.t("buttons.generate_text"),
    i18n.t("buttons.generate_photo"),
    i18n.t("buttons.balance"),
    i18n.t("buttons.top_up"),
    i18n.t("buttons.help"),
    i18n.t("buttons.chatgpt"),
}

_PRO_REQUEST_PATTERN = re.compile(
    r"\b(?:veo(?:[-\s]?2)?[-\s]*pro|veo2pro|model\s*[:=]?\s*pro|pro-?версия|pro version)\b",
    re.IGNORECASE,
)

_SUPPORTED_VEO_PREFIXES: Tuple[str, ...] = ("veo-3.0-", "veo-3.1-")
_SUPPORTED_SORA_PREFIXES: Tuple[str, ...] = (
    "sora-",
    "sora2",
    "gpt-4.1-sora",
    "gpt-4.2-sora",
)

_VIDEO_PROVIDER_PRICE_KEYS: Dict[str, str] = {
    "veo": "veo3",
    "sora": "sora",
}

SUPPORTED_SORA_MODELS: Set[str] = set(SORA_SUPPORTED_MODELS)


class GenerationStates(StatesGroup):
    """Conversation states for collecting generation inputs."""

    video_model = State()
    video_mode = State()
    image_mode = State()
    text_prompt = State()
    photo_prompt = State()


class ChatGPTState(StatesGroup):
    """Conversation states for the GPT-4.1 assistant."""

    awaiting_input = State()


class AdminStates(StatesGroup):
    """FSM states for the administrator panel."""

    menu = State()
    broadcast_message = State()
    broadcast_confirm = State()
    direct_target = State()
    direct_message = State()
    direct_confirm = State()


_ADMIN_BROADCAST_THROTTLE_SECONDS = 0.05


def _admin_menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=i18n.t("admin.menu.broadcast_all"))],
            [KeyboardButton(text=i18n.t("admin.menu.direct_message"))],
            [KeyboardButton(text=i18n.t("admin.menu.close"))],
        ],
        resize_keyboard=True,
    )


def _admin_cancel_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=i18n.t("admin.buttons.cancel"))]],
        resize_keyboard=True,
    )


def _admin_confirm_keyboard(action: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=i18n.t("admin.buttons.send"),
                    callback_data=f"admin:{action}:confirm",
                ),
                InlineKeyboardButton(
                    text=i18n.t("admin.buttons.abort"),
                    callback_data=f"admin:{action}:cancel",
                ),
            ]
        ]
    )


def _escape_format_value(value: str) -> str:
    return value.replace("{", "{{").replace("}", "}}")


def _is_cancel_action(text: Optional[str]) -> bool:
    if text is None:
        return False
    normalized = text.strip().lower()
    if not normalized:
        return False
    cancel_tokens = {
        "/cancel",
        "cancel",
        "отмена",
        i18n.t("admin.buttons.cancel").strip().lower(),
    }
    return normalized in cancel_tokens


async def _collect_broadcast_recipients(db: Database) -> List[int]:
    try:
        raw_users = await db.list_users()
    except Exception:  # pragma: no cover - defensive guard around external IO
        log.warning("Failed to list users for broadcast", exc_info=True)
        return []
    recipients: List[int] = []
    seen: Set[int] = set()
    for entry in raw_users or []:
        if not isinstance(entry, Mapping):
            continue
        value = entry.get("user_id")
        try:
            user_id = int(value)
        except (TypeError, ValueError):
            continue
        if user_id <= 0 or user_id in seen:
            continue
        seen.add(user_id)
        recipients.append(user_id)
    return recipients


def _normalise_username(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.startswith("@"):
        text = text[1:]
    return text.lower()


def _find_user_record(
    users: Iterable[Mapping[str, Any]], identifier: str
) -> Optional[Mapping[str, Any]]:
    candidate = (identifier or "").strip()
    if not candidate:
        return None
    if candidate.startswith("@"):
        username = candidate[1:].strip().lower()
        for entry in users:
            if not isinstance(entry, Mapping):
                continue
            if _normalise_username(entry.get("username")) == username:
                return entry
        return None
    if candidate.startswith("+"):
        candidate = candidate[1:]
    try:
        user_id = int(candidate)
    except ValueError:
        return None
    for entry in users:
        if not isinstance(entry, Mapping):
            continue
        value = entry.get("user_id")
        try:
            entry_id = int(value)
        except (TypeError, ValueError):
            continue
        if entry_id == user_id:
            return entry
    return None


def _format_user_label(record: Mapping[str, Any]) -> str:
    username = str(record.get("username") or "").strip()
    user_id_raw = record.get("user_id")
    try:
        user_id = int(user_id_raw)
    except (TypeError, ValueError):
        user_id = None
    if username:
        handle = username if username.startswith("@") else f"@{username}"
        if user_id:
            return f"{handle} ({user_id})"
        return handle
    if user_id:
        return str(user_id)
    return "—"


async def _send_text_safely(bot: Bot, user_id: int, text: str) -> bool:
    try:
        await bot.send_message(user_id, text)
        return True
    except RetryAfter as exc:  # pragma: no cover - timing dependent
        timeout = getattr(exc, "timeout", None)
        try:
            delay = float(timeout) if timeout is not None else 1.0
        except (TypeError, ValueError):
            delay = 1.0
        if delay < 0:
            delay = 1.0
        await asyncio.sleep(delay)
        try:
            await bot.send_message(user_id, text)
            return True
        except (RetryAfter, TelegramAPIError) as err:  # pragma: no cover - defensive
            log.warning(
                "Retry failed for broadcast delivery user=%s error=%s", user_id, err, exc_info=True
            )
            return False
    except (ChatNotFound, Unauthorized, CantTalkWithBot, BadRequest) as exc:
        log.info("Skipping delivery to user=%s error=%s", user_id, exc)
        return False
    except TelegramAPIError as exc:  # pragma: no cover - network guard
        log.warning(
            "Failed to deliver broadcast to user=%s error=%s", user_id, exc, exc_info=True
        )
        return False


async def _broadcast_to_users(bot: Bot, recipients: Iterable[int], text: str) -> Tuple[int, int]:
    sent = 0
    failed = 0
    for user_id in recipients:
        if await _send_text_safely(bot, user_id, text):
            sent += 1
        else:
            failed += 1
        if _ADMIN_BROADCAST_THROTTLE_SECONDS > 0:
            await asyncio.sleep(_ADMIN_BROADCAST_THROTTLE_SECONDS)
    return sent, failed


async def _enter_admin_menu(state: FSMContext, message: Message) -> None:
    await AdminStates.menu.set()
    await message.answer(i18n.t("admin.menu.title"), reply_markup=_admin_menu_keyboard())


async def _return_to_admin_menu(state: FSMContext, message: Message) -> None:
    await state.reset_data()
    await _enter_admin_menu(state, message)

@dataclass
class OrderContext:
    """Details of a generation order pending confirmation."""

    category: Literal["video", "image"]
    flow: Literal["text", "photo"]
    prompt: str
    size: str
    model: str
    provider: Optional[str] = None
    product: str = "sora"
    credits_cost: int = 0
    model_label: str = ""
    image_file_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    corr_id: Optional[str] = None
    aspect_ratio: Optional[str] = None
    duration_seconds: Optional[int] = None
    hd: bool = False
    price_rub: Decimal = Decimal("0")

    def key(self) -> str:
        parts = [
            self.category,
            self.flow,
            self.prompt,
            self.image_file_id or "",
            self.size,
            self.model,
            self.aspect_ratio or "",
            str(self.duration_seconds or 0),
            "hd" if self.hd else "sd",
            str(self.price_rub),
        ]
        return "|".join(parts)


def _shorten(text: str, limit: int = 240) -> str:
    snippet = (text or "").strip()
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 3] + "..."


@dataclass(frozen=True)
class VideoModelOption:
    key: str
    model: str
    provider: str
    label: str


def _normalise_video_model_name(name: str) -> str:
    value = (name or "").strip()
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    return value


def _is_veo_video_model_name(name: str) -> bool:
    normalised = _normalise_video_model_name(name)
    lowered = normalised.lower()
    return any(lowered.startswith(prefix) for prefix in _SUPPORTED_VEO_PREFIXES)


def _is_sora_video_model_name(name: str) -> bool:
    normalised = _normalise_video_model_name(name)
    lowered = normalised.lower()
    if lowered in {"sora", "sora-2", "sora2"}:
        return True
    return any(lowered.startswith(prefix) for prefix in _SUPPORTED_SORA_PREFIXES)


def _make_model_option_key(name: str, used: Set[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    slug = slug.strip("_") or "veo"
    key = slug
    counter = 2
    while key in used:
        key = f"{slug}_{counter}"
        counter += 1
    used.add(key)
    return key


def _video_model_options(config: Config) -> list[VideoModelOption]:
    used_keys: Set[str] = set()
    seen_models: Set[str] = set()
    options: list[VideoModelOption] = []

    def _ui_model_label(model_name: str, provider: str) -> str:
        base_key = _normalise_video_model_name(model_name)
        provider_key = (provider or "").strip().lower()
        base_label = UI_MODEL_LABELS.get(base_key) or UI_MODEL_LABELS.get(provider_key)
        if not base_label:
            if provider_key == "veo":
                base_label = i18n.t("video.models.veo")
            elif provider_key == "sora":
                base_label = i18n.t("video.models.sora")
            else:
                base_label = base_key or provider
        price_key = _VIDEO_PROVIDER_PRICE_KEYS.get(provider_key)
        if price_key:
            price = PRODUCT_PRICING_UI.get(price_key)
            if price is not None:
                base_label = f"{base_label} — {price} ₽"
        return base_label

    def _register(model_name: str, provider: str) -> None:
        normalised = _normalise_video_model_name(model_name)
        if not normalised or normalised in seen_models:
            return
        seen_models.add(normalised)
        key = _make_model_option_key(normalised, used_keys)
        options.append(
            VideoModelOption(
                key=key,
                model=normalised,
                provider=provider,
                label=_ui_model_label(normalised, provider),
            )
        )

    if config.gemini_video_enabled:
        default_name = _normalise_video_model_name(config.gemini_model_video)
        if default_name:
            _register(default_name, "veo")
        for candidate in list_veo_video_models(config):
            normalised = _normalise_video_model_name(candidate)
            if _is_veo_video_model_name(normalised):
                _register(normalised, "veo")

    if config.sora_video_enabled:
        default_sora = _normalise_video_model_name(config.sora_model_video)
        if default_sora:
            _register(default_sora, "sora")

    return options


def _find_video_model_option(config: Config, key: str) -> Optional[VideoModelOption]:
    for option in _video_model_options(config):
        if option.key == key:
            return option
    return None


def _provider_for_model(model: Optional[str], config: Config) -> str:
    name = _normalise_video_model_name(model or "")
    if name and _is_veo_video_model_name(name):
        return "veo"
    if name and _is_sora_video_model_name(name):
        return "sora"
    if name and config.sora_model_video and name == _normalise_video_model_name(config.sora_model_video):
        return "sora"
    if name and config.gemini_model_video and name == _normalise_video_model_name(config.gemini_model_video):
        return "veo"
    lowered = (model or "").strip().lower()
    if lowered in {"sora", "sora-video", "openai-video"}:
        return "sora"
    if lowered in {"veo", "veo2", "gemini-video"}:
        return "veo"
    if config.sora_video_enabled and not config.gemini_video_enabled:
        return "sora"
    if config.gemini_video_enabled:
        return "veo"
    return "video"


def _resolve_product_key(
    category: str,
    provider: Optional[str],
    model: Optional[str],
    product_hint: Optional[str] = None,
) -> str:
    if product_hint:
        return product_hint
    if category == "image":
        return "image"
    provider_key = (provider or "").lower()
    if provider_key in {"gemini-image", "image", "gemini"}:
        return "image"
    if provider_key.startswith("veo") or provider_key in {"gemini-video", "veo"}:
        return "veo3"
    model_key = _normalise_video_model_name(model or "")
    if _is_veo_video_model_name(model_key):
        return "veo3"
    if _is_sora_video_model_name(model_key):
        return "sora"
    return "sora"


def _resolve_model_label(model: str, config: Config) -> str:
    for option in _video_model_options(config):
        if option.model == model or option.label == model:
            return option.label
        if _normalise_video_model_name(option.model) == _normalise_video_model_name(model):
            return option.label
    if model in {config.gemini_model_video, "veo", "veo2", "gemini-video"}:
        return i18n.t("video.models.veo")
    if model in {config.sora_model_video, "sora", "sora-video", "openai-video"}:
        return i18n.t("video.models.sora")
    if model in {config.gemini_model_image, "gemini-image", "gemini"}:
        return i18n.t("image.model.gemini")
    return model


async def _ensure_supported_video_model(
    message: Message,
    *,
    category: str,
    model: str,
    provider: str,
    model_label: str,
    config: Config,
) -> Tuple[str, str, str]:
    if category != "video":
        return model, provider, model_label
    normalised = _normalise_video_model_name(model)
    if provider == "sora" or _is_sora_video_model_name(model):
        if normalised not in SUPPORTED_SORA_MODELS:
            await message.answer("Выбранная модель не поддерживается, используется sora")
            model = "sora"
            provider = "sora"
            model_label = _resolve_model_label("sora", config)
    return model, provider, model_label


def _video_models_description(config: Config) -> str:
    labels = [option.label for option in _video_model_options(config)]
    if not labels:
        return i18n.t("video.models.unavailable")
    return ", ".join(dict.fromkeys(labels))


def _resolve_openai_video_client(
    job_queue: JobQueue,
    config: Config,
    client_hint: Optional[OpenAIVideoClient] = None,
) -> Optional[OpenAIVideoClient]:
    if client_hint is not None:
        return client_hint
    candidates: Iterable[Optional[str]] = (
        "openai-video-api",
        config.sora_model_video,
        "openai-video",
        "sora",
        "sora-2",
        "sora-2-fast",
        "sora-2-pro",
    )
    for key in candidates:
        if not key:
            continue
        provider = job_queue.get_provider(key)
        if isinstance(provider, OpenAIVideoClient):
            return provider
    try:
        return OpenAIVideoClient(config=config)
    except RuntimeError:
        return None


def _parse_video_command_args(
    args: str,
    *,
    default_prompt: str = "",
) -> tuple[str, Dict[str, Any], Dict[str, Any], Optional[str], Optional[str], Optional[str]]:
    prompt = default_prompt
    settings: Dict[str, Any] = {}
    payload: Dict[str, Any] = {}
    size: Optional[str] = None
    model: Optional[str] = None
    idempotency_key: Optional[str] = None
    text = (args or "").strip()
    if not text:
        return prompt, settings, payload, size, model, idempotency_key
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text, settings, payload, size, model, idempotency_key
        if isinstance(data, dict):
            prompt = str(data.get("prompt") or data.get("text") or default_prompt or "").strip()
            raw_settings = data.get("settings")
            if isinstance(raw_settings, dict):
                settings.update(raw_settings)
            raw_payload = data.get("payload")
            if isinstance(raw_payload, dict):
                payload.update(raw_payload)
            model_value = data.get("model") or settings.get("model")
            if isinstance(model_value, str) and model_value.strip():
                model = model_value.strip()
                settings.setdefault("model", model)
            size_value = data.get("size") or settings.get("size")
            if isinstance(size_value, str) and size_value.strip():
                size = size_value.strip()
                settings.setdefault("size", size)
            idempotency = data.get("idempotency_key")
            if isinstance(idempotency, str) and idempotency.strip():
                idempotency_key = idempotency.strip()
            for key in ("duration", "duration_seconds", "aspect_ratio"):
                value = data.get(key)
                if value is not None and key not in settings:
                    settings[key] = value
            preset = data.get("preset")
            if preset is not None:
                settings.setdefault("preset", preset)
        elif isinstance(data, str):
            prompt = data.strip()
    else:
        prompt = text
    return prompt, settings, payload, size, model, idempotency_key


def _search_mapping(payload: Any, keys: Iterable[str]) -> Optional[str]:
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            candidate = _search_mapping(value, keys)
            if candidate:
                return candidate
    elif isinstance(payload, list):
        for item in payload:
            candidate = _search_mapping(item, keys)
            if candidate:
                return candidate
    return None


def _extract_video_identifiers(
    payload: Any,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    job_id = _search_mapping(payload, ("job_id", "id"))
    video_id = _search_mapping(payload, ("video_id", "id"))
    operation_name = _search_mapping(payload, ("operation_name", "operationName"))
    return job_id, video_id, operation_name


def _image_model_description(config: Config) -> str:
    if config.gemini_enabled:
        return i18n.t("image.model.gemini")
    return i18n.t("image.model.unavailable")


def _map_provider_error(error: ProviderAPIError) -> tuple[str, str, bool, str]:
    status = error.status_code or 0
    raw_provider_message = (error.provider_message or "").strip()
    default_message = (str(error) or "").strip()
    provider_message = raw_provider_message or default_message
    error_type = (error.error_type or "").lower()
    error_code = (error.error_code or "").lower()
    short = _shorten(provider_message or (error.args[0] if error.args else "Ошибка"))
    notify_support = False
    hint = error_type or error_code or "unknown"

    available_models_text = ", ".join(sorted(SUPPORTED_SORA_MODELS))

    policy_trigger = (
        "policy" in error_type
        or "policy" in error_code
        or "safety" in error_type
        or "safety" in error_code
    )

    if error_code == "invalid_model" or error_type == "invalid_model":
        message = (
            "Выбранная модель Sora не поддерживается. "
            f"Доступные модели: {escape_html(available_models_text)}"
        )
        short = f"Недоступная модель Sora ({available_models_text})"
        hint = "invalid_model"
        return message, short, notify_support, hint

    if policy_trigger:
        detail_source = raw_provider_message
        message = "Запрос нарушает правила контента."
        if detail_source:
            detail = escape_html(detail_source)
            if detail:
                message = f"Запрос нарушает правила контента: {detail}"
        short = _shorten(detail_source or default_message or (error.args[0] if error.args else ""))
        if not short:
            short = "Запрос нарушает правила контента"
        hint = "policy"
    elif status in {401, 403}:
        notify_support = True
        hint = "auth"
        message = (
            "API-ключ недействителен или нет доступа к провайдеру. "
            "Проверьте ключ/организацию. Сообщили оператору, скоро поправим."
        )
        short = "Недействительный API-ключ"
    elif status == 429:
        notify_support = True
        hint = "rate_limit"
        message = "Превышен лимит. Подождите и повторите."
        short = "Лимит запросов"
    elif status == 400 or "validation" in error_type:
        detail = escape_html(provider_message or (error.args[0] if error.args else ""))
        message = f"Неверные параметры запроса: {detail or 'проверьте входные данные.'}"
        short = _shorten(provider_message or "Ошибка параметров")
        hint = "validation"
    elif status >= 500 or error_type in {"timeout", "network"} or status == 0:
        notify_support = True
        provider_detail = escape_html(provider_message or (error.args[0] if error.args else ""))
        provider_code = error_code or error_type or ("timeout" if status == 0 else str(status))
        hint = provider_code or "provider_unavailable"
        detail_text = provider_detail or "попробуйте позже."
        message = f"Ошибка провайдера ({provider_code or 'unknown'}): {detail_text}"
        short = _shorten(provider_message or message)
    else:
        detail = escape_html(provider_message or (error.args[0] if error.args else ""))
        message = f"Не удалось выполнить запрос: {detail or 'попробуйте позже.'}"
        short = _shorten(provider_message or "Ошибка провайдера")
    return message, short, notify_support, hint


async def _notify_support(
    bot: Bot,
    config: Config,
    *,
    event: str,
    corr_id: str,
    user_id: int,
    status_code: Optional[int],
    error_type: Optional[str],
    hint: str,
) -> None:
    chat_id = config.support_chat_id
    if not chat_id:
        return
    key = f"{event}:{status_code}:{hint}"
    if not should_notify_support(key, config.support_notify_interval):
        return
    lines = [
        "⚠️ Критическая ошибка генерации",
        f"event: {event}",
        f"corr_id: {corr_id}",
        f"user_id: {user_id}",
        f"status: {status_code or 'n/a'}",
        f"type: {error_type or '-'}",
        f"hint: {hint}",
    ]
    try:
        await bot.send_message(chat_id, "\n".join(lines))
    except Exception:  # pragma: no cover - external dependency
        log.warning("Failed to notify support chat", exc_info=True)


@dataclass
class UserSession:
    """Transient per-user settings and cached context."""

    last_size: str = field(
        default_factory=lambda: _resolve_video_size(
            DEFAULT_ASPECT_RATIO, DEFAULT_HD_ENABLED
        )
    )
    last_aspect_ratio: str = DEFAULT_ASPECT_RATIO
    video_duration: int = DEFAULT_VIDEO_DURATION
    hd_enabled: bool = DEFAULT_HD_ENABLED
    pending_order: Optional[OrderContext] = None
    awaiting_payment: bool = False
    last_launch_key: Optional[str] = None
    last_launch_ts: float = 0.0

    def update_video_size(self) -> None:
        self.last_size = _resolve_video_size(self.last_aspect_ratio, self.hd_enabled)


class SessionManager:
    """In-memory storage for user sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[int, UserSession] = {}

    def get(self, user_id: int) -> UserSession:
        return self._sessions.setdefault(user_id, UserSession())


SESSION_MANAGER = SessionManager()
STATUS_MESSAGES = StatusMessageManager()


def _main_keyboard(config: Config) -> ReplyKeyboardMarkup:
    keyboard: List[List[KeyboardButton]] = [
        [KeyboardButton(text=i18n.t("buttons.generate_text"))],
        [KeyboardButton(text=i18n.t("buttons.generate_photo"))],
        [KeyboardButton(text=i18n.t("buttons.balance"))],
        [KeyboardButton(text=i18n.t("buttons.top_up"))],
        [KeyboardButton(text=i18n.t("buttons.help"))],
    ]
    keyboard.append([KeyboardButton(text=i18n.t("buttons.chatgpt"))])
    return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)


def _back_button(target: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text=i18n.t("buttons.back"), callback_data=f"flow:back:{target}"
    )


def _build_back_keyboard(target: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[_back_button(target)]])


def _build_video_settings_keyboard(
    session: UserSession, context: str, *, include_back: bool = True
) -> InlineKeyboardMarkup:
    prompt_row = [
        InlineKeyboardButton(
            text=i18n.t("video.prompt.current_prompt"), callback_data="noop"
        )
    ]
    image_row = [
        InlineKeyboardButton(
            text=i18n.t("video.prompt.current_image"), callback_data="noop"
        )
    ]
    duration_row: list[InlineKeyboardButton] = []
    for option in VIDEO_DURATION_OPTIONS:
        label = _append_checkmark(f"{option} сек.", session.video_duration == option)
        duration_row.append(
            InlineKeyboardButton(
                text=label, callback_data=f"opt:duration:{context}:{option}"
            )
        )
    aspect_row: list[InlineKeyboardButton] = []
    for token, ratio in ASPECT_RATIO_OPTIONS.items():
        label = _append_checkmark(ratio, session.last_aspect_ratio == ratio)
        aspect_row.append(
            InlineKeyboardButton(
                text=label, callback_data=f"opt:aspect:{context}:{token}"
            )
        )
    hd_button = InlineKeyboardButton(
        text=_append_checkmark("HD", session.hd_enabled),
        callback_data=f"opt:hd:{context}:toggle",
    )
    start_button = InlineKeyboardButton(
        text=i18n.t("video.prompt.start_button"), callback_data="start_generation"
    )
    rows: list[list[InlineKeyboardButton]] = [
        prompt_row,
        image_row,
        duration_row,
        aspect_row,
        [hd_button],
        [start_button],
    ]
    if include_back:
        rows.append([_back_button("video_mode")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _video_price_rub(duration: int) -> Decimal:
    return VIDEO_PRICE_RUB.get(duration, VIDEO_PRICE_RUB[DEFAULT_VIDEO_DURATION])


def _format_video_price(duration: int, config: Config) -> str:
    return config.format_rubles(_video_price_rub(duration))


def _video_quality_label(hd: bool) -> str:
    return i18n.t("video.prompt.quality_hd") if hd else i18n.t("video.prompt.quality_sd")


def _build_confirmation_keyboard(*, can_launch: bool, include_back: bool = True) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    if can_launch:
        rows.append(
            [InlineKeyboardButton(text=i18n.t("buttons.launch"), callback_data="order:launch")]
        )
    rows.append([InlineKeyboardButton(text=i18n.t("buttons.edit"), callback_data="order:edit")])
    if include_back:
        rows.append([InlineKeyboardButton(text=i18n.t("buttons.back"), callback_data="order:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _build_generation_in_progress_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="⏳ Генерация идёт",
                    callback_data="order:busy",
                )
            ]
        ]
    )


def _build_balance_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=i18n.t("buttons.top_up_short"), callback_data="menu:topup")],
        ]
    )


def _build_video_models_keyboard(options: list[VideoModelOption]) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for option in options:
        rows.append(
            [
                InlineKeyboardButton(
                    text=option.label,
                    callback_data=f"video:model:{option.key}",
                )
            ]
        )
    rows.append([_back_button("main")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _build_mode_keyboard(category: Literal["video", "image"]) -> InlineKeyboardMarkup:
    if category == "video":
        text_label = i18n.t("video.mode.text")
        photo_label = i18n.t("video.mode.photo")
        prefix = "video:mode"
        back_target = "video_model"
    else:
        text_label = i18n.t("image.mode.text")
        photo_label = i18n.t("image.mode.photo")
        prefix = "image:mode"
        back_target = "main"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=text_label, callback_data=f"{prefix}:text")],
            [InlineKeyboardButton(text=photo_label, callback_data=f"{prefix}:photo")],
            [_back_button(back_target)],
        ]
    )


async def _send_balance_info(message: Message, db: Database) -> None:
    user_id = await _ensure_user(message, db)
    credits = await db.get_user_credits(user_id)
    balance_text = i18n.t("balance.info", credits=credits)
    await message.answer(balance_text, reply_markup=_build_balance_keyboard())


def _build_help_keyboard(config: Config) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text="👤 Написать оператору",
                url="https://t.me/hr_assistt",
            )
        ]
    ]
    terms_url = (config.terms_url or "").strip()
    if terms_url:
        rows.append(
            [
                InlineKeyboardButton(
                    text=i18n.t("buttons.offer"),
                    url=terms_url,
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _send_main_menu(message: Message, config: Config) -> None:
    await message.answer(
        i18n.t(
            "main.welcome",
            video_models=_video_models_description(config),
            image_model=_image_model_description(config),
        ),
        reply_markup=_main_keyboard(config),
    )


async def _ensure_user(message: Message, db: Database) -> int:
    user = message.from_user
    if user is None:  # pragma: no cover - defensive
        raise RuntimeError("Message without user")
    await db.ensure_user(
        user.id,
        user.username,
        first_name=user.first_name,
        last_name=user.last_name,
    )
    return user.id


def _create_order(
    *,
    category: Literal["video", "image"],
    flow: Literal["text", "photo"],
    prompt: str,
    config: Config,
    session: UserSession,
    model: str,
    provider: Optional[str],
    product: str,
    model_label: str,
    include_size: bool,
    image_file_id: Optional[str] = None,
) -> OrderContext:
    size = session.last_size if include_size else ""
    aspect_ratio = session.last_aspect_ratio if include_size else None
    duration_seconds = session.video_duration if category == "video" else None
    hd_enabled = session.hd_enabled if category == "video" else False
    resolved_product = _resolve_product_key(category, provider, model, product)
    price_rub = Decimal("0")
    credits_cost = config.get_product_credits(resolved_product)
    if category == "video" and duration_seconds:
        price_rub = _video_price_rub(duration_seconds)
        credits_cost = config.credits_for_rubles(price_rub)
    if credits_cost <= 0:
        credits_cost = config.generation_cost_credits
    return OrderContext(
        category=category,
        flow=flow,
        prompt=prompt,
        size=size,
        model=model,
        provider=provider,
        product=resolved_product,
        credits_cost=credits_cost,
        model_label=model_label or _resolve_model_label(model, config),
        image_file_id=image_file_id,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
        hd=hd_enabled,
        price_rub=price_rub,
    )


def _detect_pro_request(text: Optional[str]) -> bool:
    if not text:
        return False
    normalised = text.strip().lower()
    if normalised in {"pro", "pro-версия", "pro версия", "pro version"}:
        return True
    return bool(_PRO_REQUEST_PATTERN.search(text))


_MODEL_DIRECTIVE_PATTERN = re.compile(
    r"model\s*(?:[:=]\s*)?(?:veo(?:[-\s]?2)?[-\s]*pro|veo2pro|pro)",
    re.IGNORECASE,
)
_MODEL_NAME_PATTERN = re.compile(r"veo(?:[-\s]?2)?[-\s]*pro", re.IGNORECASE)


def _strip_pro_directives(text: str) -> str:
    cleaned = _MODEL_DIRECTIVE_PATTERN.sub("", text)
    cleaned = _MODEL_NAME_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


async def _notify_pro_unavailable(message: Message) -> None:
    await message.answer(i18n.t("errors.pro_unavailable"))


def _video_prompt_body(
    *, session: UserSession, mode: Literal["text", "photo"], model_label: str, config: Config
) -> str:
    template = "video.prompt.photo" if mode == "photo" else "video.prompt.text"
    return i18n.t(
        template,
        model=model_label,
        duration=session.video_duration,
        aspect=session.last_aspect_ratio,
        quality=_video_quality_label(session.hd_enabled),
        price=_format_video_price(session.video_duration, config),
    )


async def _send_generation_prompt(
    message: Message,
    *,
    session: UserSession,
    category: Literal["video", "image"],
    mode: Literal["text", "photo"],
    model_label: str,
    include_size: bool,
    config: Config,
) -> None:
    if category == "video":
        body = _video_prompt_body(
            session=session, mode=mode, model_label=model_label, config=config
        )
        if include_size:
            markup = _build_video_settings_keyboard(session, mode)
        else:
            markup = _build_back_keyboard("video_mode")
    else:
        if mode == "photo":
            body = i18n.t("image.prompt.photo", model=model_label)
        else:
            body = i18n.t("image.prompt.text", model=model_label)
        markup = _build_back_keyboard("image_mode")
    await message.answer(body, reply_markup=markup)


def _order_price_text(order: OrderContext, config: Config) -> str:
    credits_cost = order.credits_cost or config.generation_cost_credits
    if order.price_rub and order.price_rub > 0:
        rub_value = order.price_rub
    else:
        rub_value = config.rubles_for_credits(credits_cost)
    rub_text = config.format_rubles(rub_value)
    credits_text = format_credits(credits_cost)
    return f"{rub_text} · {credits_text}"


async def _send_order_confirmation(
    *,
    bot: Bot,
    chat_id: int,
    order: OrderContext,
    config: Config,
    can_launch: bool,
    include_back: bool = True,
) -> Message:
    price_text = _order_price_text(order, config)
    prompt_text: SafeText = format_prompt(order.prompt)
    if order.category == "image":
        if order.flow == "photo":
            body = i18n.t(
                "image.confirm.photo",
                prompt=prompt_text,
                model=order.model_label,
                price=price_text,
            )
        else:
            body = i18n.t(
                "image.confirm.text",
                prompt=prompt_text,
                model=order.model_label,
                price=price_text,
            )
    else:
        size_value = order.size or "—"
        duration_value = order.duration_seconds or DEFAULT_VIDEO_DURATION
        aspect_value = order.aspect_ratio or DEFAULT_ASPECT_RATIO
        quality_value = _video_quality_label(order.hd)
        if order.flow == "photo":
            body = i18n.t(
                "video.confirm.photo",
                prompt=prompt_text,
                size=size_value,
                duration=duration_value,
                aspect=aspect_value,
                quality=quality_value,
                model=order.model_label,
                price=price_text,
            )
        else:
            body = i18n.t(
                "video.confirm.text",
                prompt=prompt_text,
                size=size_value,
                duration=duration_value,
                aspect=aspect_value,
                quality=quality_value,
                model=order.model_label,
                price=price_text,
            )
    keyboard = _build_confirmation_keyboard(can_launch=can_launch, include_back=include_back)
    return await bot.send_message(chat_id, body, reply_markup=keyboard)


async def flow_back_callback_handler(
    callback: CallbackQuery, state: FSMContext, config: Config
) -> None:
    await safe_callback_answer(callback)
    parts = (callback.data or "").split(":", maxsplit=2)
    if len(parts) != 3:
        return
    _, _, target = parts
    if target == "main":
        await state.finish()
        await _send_main_menu(callback.message, config)
        return
    data = await state.get_data()
    if target == "video_model":
        current_model = data.get("model")
        current_provider = data.get("provider")
        product = _resolve_product_key("video", current_provider, current_model, data.get("product"))
        await state.update_data(
            flow_type="video",
            model=None,
            provider=None,
            model_label=None,
            include_size=True,
            product=product,
            mode=None,
        )
        await GenerationStates.video_model.set()
        keyboard = _build_video_models_keyboard(_video_model_options(config))
        try:
            await callback.message.edit_text(
                i18n.t("video.models.prompt"), reply_markup=keyboard
            )
        except Exception:  # pragma: no cover - Telegram edits may fail
            await callback.message.answer(
                i18n.t("video.models.prompt"), reply_markup=keyboard
            )
        return
    if target == "video_mode":
        model = data.get("model") or config.default_video_model
        provider = data.get("provider") or model
        model_label = data.get("model_label") or _resolve_model_label(model, config)
        product = _resolve_product_key("video", provider, model, data.get("product"))
        await state.update_data(
            flow_type="video",
            model=model,
            provider=provider,
            model_label=model_label,
            include_size=True,
            product=product,
            mode=None,
        )
        await GenerationStates.video_mode.set()
        keyboard = _build_mode_keyboard("video")
        try:
            await callback.message.edit_text(
                i18n.t("video.mode.prompt"), reply_markup=keyboard
            )
        except Exception:  # pragma: no cover - Telegram edits may fail
            await callback.message.answer(
                i18n.t("video.mode.prompt"), reply_markup=keyboard
            )
        return
    if target == "image_mode":
        model = data.get("model") or config.gemini_model_image
        provider = data.get("provider") or "gemini-image"
        model_label = data.get("model_label") or _resolve_model_label(model, config)
        product = _resolve_product_key("image", provider, model, data.get("product"))
        await state.update_data(
            flow_type="image",
            model=model,
            provider=provider,
            model_label=model_label,
            include_size=False,
            product=product,
            mode=None,
        )
        await GenerationStates.image_mode.set()
        keyboard = _build_mode_keyboard("image")
        try:
            await callback.message.edit_text(
                i18n.t("image.mode.prompt"), reply_markup=keyboard
            )
        except Exception:  # pragma: no cover - Telegram edits may fail
            await callback.message.answer(
                i18n.t("image.mode.prompt"), reply_markup=keyboard
            )
        return


def _format_packages_list(config: Config) -> str:
    terms_url = (config.terms_url or "").strip()
    terms_suffix: SafeText | str = ""
    if terms_url:
        escaped_url = escape_html(terms_url)
        link = SafeText(f'<a href="{escaped_url}">{escaped_url}</a>')
        terms_suffix = SafeText(f": {link}")
    return i18n.t("payment.packages.title", terms=terms_suffix).strip()


def _build_packages_keyboard(config: Config) -> Optional[InlineKeyboardMarkup]:
    if not config.yookassa_ready:
        return None
    rows: list[list[InlineKeyboardButton]] = []
    packages = {package.package_id: package for package in config.credit_packages}
    for pack in TOPUP_UI_PACKS:
        amount_value = int(pack.get("amount", 0) or 0)
        if amount_value <= 0:
            continue
        package = packages.get(str(amount_value))
        if package is None:
            continue
        bonus_ratio = pack.get("bonus_pct", 0) or 0
        bonus_percent_value = int(round(bonus_ratio * 100))
        if bonus_percent_value > 0:
            text = f"{amount_value}₽ +{bonus_percent_value}%"
        else:
            text = f"{amount_value}₽ +0%"
        rows.append(
            [
                InlineKeyboardButton(
                    text=text,
                    callback_data=f"pay:{package.package_id}",
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _send_payment_showcase(bot: Bot, chat_id: int, config: Config) -> None:
    text = _format_packages_list(config)
    keyboard = _build_packages_keyboard(config)
    if keyboard is not None:
        await bot.send_message(chat_id, text, reply_markup=keyboard)
    else:
        body = f"{text}\n\n{i18n.t('payment.unavailable')}"
        await bot.send_message(chat_id, body.strip())


async def _handle_not_enough_credits(message: Message, config: Config, session: UserSession) -> None:
    await message.answer(i18n.t("flow.not_enough"))
    session.awaiting_payment = True
    await _send_payment_showcase(message.bot, message.chat.id, config)


async def _show_confirmation_and_balance(
    *,
    message: Message,
    order: OrderContext,
    session: UserSession,
    db: Database,
    config: Config,
) -> None:
    user_id = message.from_user.id
    credits = await db.get_user_credits(user_id)
    required = order.credits_cost or config.generation_cost_credits
    can_launch = credits >= required
    session.pending_order = order
    session.awaiting_payment = not can_launch
    await _send_order_confirmation(
        bot=message.bot,
        chat_id=message.chat.id,
        order=order,
        config=config,
        can_launch=can_launch,
    )
    if not can_launch:
        await _handle_not_enough_credits(message, config, session)


async def _launch_order(
    *,
    callback: CallbackQuery,
    order: OrderContext,
    session: UserSession,
    db: Database,
    job_queue: JobQueue,
    config: Config,
    gate: GenerationRequestGate,
    error_reporter: ErrorReporter,
) -> None:
    user = callback.from_user
    if user is None:  # pragma: no cover - defensive
        return
    user_id = user.id
    now = time.time()

    original_prompt = order.prompt
    preflight = run_preflight(original_prompt)
    corr_id = str(uuid.uuid4())
    order.corr_id = corr_id

    async def _release_lock(reason: str, status: str) -> None:
        try:
            await gate.release(
                user_id=user_id,
                corr_id=corr_id,
                reason=reason,
                status=status,
            )
        except Exception:  # pragma: no cover - defensive
            log.exception("Failed to release generation gate lock", exc_info=True)

    if preflight.blocked:
        explanation = preflight.user_message or policy_message(preflight.scope or "unknown")
        log_event(
            level="WARNING",
            event="preflight_block",
            corr_id=corr_id,
            user_id=user_id,
            username=user.username,
            model=order.model,
            provider=order.provider or order.model,
            size=order.size,
            error_type="preflight_blocked",
            error_msg_short=_shorten(explanation),
            prompt=original_prompt,
            extra={
                "preflight_blocked": True,
                "preflight_reason": preflight.reason,
                "preflight_scope": preflight.scope,
            },
        )
        error_record = ErrorLogRecord(
            ts=datetime.utcnow(),
            user_id=user_id,
            username=user.username,
            corr_id=corr_id,
            job_id="",
            model=order.model,
            size=order.size,
            status_code=0,
            error_type="preflight_blocked",
            error_msg_short=_shorten(explanation),
            refunded=False,
            preflight_blocked=True,
            preflight_reason=preflight.reason or "",
            auto_sanitized=False,
            sanitized_prompt=preflight.sanitized_prompt,
            error_scope=preflight.scope or "unknown",
            error_json="",
            stage="submit",
        )
        sheet_ok = await db.log_error_record(error_record)
        await error_reporter.report(
            error_record,
            context="preflight_block",
            extra={
                "gsheets_ok": sheet_ok,
                "provider": order.provider or order.model,
                "preflight_scope": preflight.scope,
            },
        )
        session.pending_order = None
        await callback.message.answer(explanation)
        return

    if preflight.auto_sanitized:
        order.prompt = preflight.sanitized_prompt
        session.pending_order = order
        await callback.message.answer(
            preflight.user_message or policy_message(preflight.scope or "policy.brand")
        )
    else:
        lower_prompt = original_prompt.lower()
        if any(word in lower_prompt for word in ("логотип", "logo", "бренд")):
            await callback.message.answer("Создание оригинальных логотипов разрешено — запускаю как есть.")

    key = order.key()
    await db.sync_user_profile(
        user_id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
    )
    provider_key = order.provider or order.model
    credits_cost = order.credits_cost or config.generation_cost_credits

    normalized = normalize_prompt(order.prompt)
    gate_options: Dict[str, Any] = {
        "category": order.category,
        "flow": order.flow,
        "size": order.size,
        "model": order.model,
        "provider": provider_key,
        "image_file_id": order.image_file_id,
        "aspect_ratio": order.aspect_ratio,
        "product": order.product,
        "duration_seconds": order.duration_seconds,
        "hd": order.hd,
    }

    gate_result = await gate.evaluate(
        user_id=user_id,
        corr_id=corr_id,
        model=order.model,
        normalized_prompt=normalized,
        options=gate_options,
    )
    if gate_result.timeout_released:
        await callback.message.answer(
            "Похоже очередь подвисла. Я остановил задачу. Можешь запустить снова."
        )
    if gate_result.decision is GateDecision.BLOCK_BUSY:
        try:
            await callback.message.edit_reply_markup(
                _build_generation_in_progress_keyboard()
            )
        except Exception:  # pragma: no cover - Telegram may block edits on old messages
            log.debug("Failed to switch keyboard to busy state", exc_info=True)
        await callback.message.answer(
            "Я уже генерирую по текущему запросу. Дай мне пару минут, пришлю результат сюда."
        )
        return
    if gate_result.decision is GateDecision.BLOCK_DUPLICATE:
        await callback.message.answer(
            "Вижу повтор того же запроса. Использую уже запущенную задачу."
        )
        return

    idempotency_key = gate_result.idempotency_key or corr_id
    try:
        await callback.message.edit_reply_markup(
            _build_generation_in_progress_keyboard()
        )
    except Exception:  # pragma: no cover - Telegram may block edits on old messages
        log.debug("Failed to switch keyboard to busy state", exc_info=True)

    log_event(
        level="INFO",
        event="enqueue",
        corr_id=corr_id,
        user_id=user_id,
        username=user.username,
        model=order.model,
        provider=provider_key,
        size=order.size,
        credits_cost=credits_cost,
        prompt=order.prompt,
        extra={
            "preflight_blocked": False,
            "preflight_reason": preflight.reason,
            "preflight_scope": preflight.scope,
            "preflight_auto_sanitized": preflight.auto_sanitized,
            "duration_seconds": order.duration_seconds,
            "hd": order.hd,
        },
    )

    if not await db.deduct_credit(user_id, credits_cost):
        log_event(
            level="ERROR",
            event="error",
            corr_id=corr_id,
            user_id=user_id,
            username=user.username,
            model=order.model,
            provider=provider_key,
            size=order.size,
            error_type="insufficient_credits",
            error_msg_short="Недостаточно кредитов",
            prompt=order.prompt,
        )
        await _release_lock("submit_failed", "insufficient_credits")
        await callback.message.answer(i18n.t("flow.not_enough"))
        session.awaiting_payment = True
        await _send_payment_showcase(callback.message.bot, callback.message.chat.id, config)
        return

    log_event(
        level="INFO",
        event="charge",
        corr_id=corr_id,
        user_id=user_id,
        username=user.username,
        model=order.model,
        provider=provider_key,
        size=order.size,
        credits_cost=credits_cost,
    )

    balance_after = await db.get_user_credits(user_id)
    log.info(
        "Launching job user=%s cost_credits=%s balance_after=%s",
        user_id,
        credits_cost,
        balance_after,
    )

    provider_settings: Dict[str, Any] = {}
    if order.category == "video":
        if order.duration_seconds:
            provider_settings["duration"] = order.duration_seconds
            provider_settings.setdefault("duration_seconds", order.duration_seconds)
        if order.aspect_ratio:
            provider_settings.setdefault("aspect_ratio", order.aspect_ratio)
    if order.image_file_id:
        inline_data = await build_inline_data_from_telegram_file(
            callback.message.bot, order.image_file_id
        )
        if inline_data:
            provider_settings["reference_inline_data"] = inline_data

    try:
        job_record = await job_queue.submit(
            user_id=user_id,
            prompt=order.prompt,
            size=order.size,
            model=order.model,
            corr_id=corr_id,
            image_file_id=order.image_file_id,
            username=user.username,
            provider=provider_key,
            settings=provider_settings or None,
            credits_cost=credits_cost,
            original_prompt=original_prompt,
            sanitized_prompt=order.prompt,
            auto_sanitized=preflight.auto_sanitized,
            preflight_reason=preflight.reason,
            preflight_scope=preflight.scope,
            idempotency_key=idempotency_key,
            content_type=order.category,
        )
    except RuntimeError as exc:
        log.warning(
            "Job submission failed corr_id=%s provider=%s error=%s",
            corr_id,
            provider_key,
            exc,
        )
        refunded = False
        try:
            await db.add_credits(user_id, credits_cost)
            refunded = True
        except Exception:  # pragma: no cover - external dependency
            log.exception("Failed to refund credits after configuration error")
        await _release_lock("submit_failed", "provider_unavailable")
        error_record = ErrorLogRecord(
            ts=datetime.utcnow(),
            user_id=user_id,
            username=user.username,
            corr_id=corr_id,
            job_id="",
            model=order.model,
            size=order.size,
            status_code=0,
            error_type="provider_unavailable",
            error_msg_short=_shorten(str(exc)),
            refunded=refunded,
            preflight_blocked=False,
            preflight_reason=preflight.reason or "",
            auto_sanitized=preflight.auto_sanitized,
            sanitized_prompt=order.prompt,
            error_scope=preflight.scope or "provider_unavailable",
            error_json="",
            stage="submit",
        )
        sheet_ok = await db.log_error_record(error_record)
        await error_reporter.report(
            error_record,
            context="provider_unavailable",
            extra={
                "gsheets_ok": sheet_ok,
                "provider": provider_key,
                "refunded": refunded,
            },
        )
        await callback.message.answer(i18n.t("errors.video_models_unavailable"))
        log_event(
            level="ERROR",
            event="error",
            corr_id=corr_id,
            user_id=user_id,
            username=user.username,
            model=order.model,
            provider=provider_key,
            size=order.size,
            credits_cost=credits_cost,
            error_type="provider_unavailable",
            error_msg_short=_shorten(str(exc)),
            prompt=order.prompt,
            gsheets_ok=sheet_ok,
            refund_done=refunded,
        )
        log_event(
            level="INFO",
            event="refund",
            corr_id=corr_id,
            user_id=user_id,
            username=user.username,
            model=order.model,
            provider=provider_key,
            size=order.size,
            credits_cost=credits_cost,
            refund_done=refunded,
        )
        return
    except ProviderAPIError as exc:
        log.warning(
            "Provider submission failed corr_id=%s provider=%s status=%s error_type=%s error_code=%s retryable=%s message=%s",
            corr_id,
            provider_key,
            exc.status_code,
            exc.error_type,
            exc.error_code,
            exc.retryable,
            exc.provider_message or str(exc),
        )
        message, short, notify_support, hint = _map_provider_error(exc)
        refunded = False
        try:
            await db.add_credits(user_id, credits_cost)
            refunded = True
        except Exception:  # pragma: no cover - external dependency
            log.exception("Failed to refund credits after submit error")
        await _release_lock("submit_failed", exc.error_type or "submit_failed")
        error_record = ErrorLogRecord(
            ts=datetime.utcnow(),
            user_id=user_id,
            username=user.username,
            corr_id=corr_id,
            job_id="",
            model=order.model,
            size=order.size,
            status_code=exc.status_code,
            error_type=exc.error_type or "submit_failed",
            error_msg_short=short,
            refunded=refunded,
            preflight_blocked=False,
            preflight_reason=preflight.reason or "",
            auto_sanitized=preflight.auto_sanitized,
            sanitized_prompt=order.prompt,
            error_scope=preflight.scope or hint,
            error_json="",
            stage="submit",
        )
        sheet_ok = await db.log_error_record(error_record)
        await error_reporter.report(
            error_record,
            context=exc.error_type or "submit_failed",
            extra={
                "gsheets_ok": sheet_ok,
                "provider": provider_key,
                "refunded": refunded,
                "notify_support": notify_support,
            },
        )
        log_event(
            level="ERROR",
            event="error",
            corr_id=corr_id,
            user_id=user_id,
            username=user.username,
            model=order.model,
            provider=provider_key,
            size=order.size,
            status_code=exc.status_code,
            duration_ms=exc.duration_ms,
            error_type=exc.error_type,
            error_code=exc.error_code,
            error_msg_short=short,
            prompt=order.prompt,
            gsheets_ok=sheet_ok,
            refund_done=refunded,
            extra={
                "provider_message": exc.provider_message,
                "preflight_blocked": False,
                "preflight_reason": preflight.reason,
                "preflight_scope": preflight.scope,
                "preflight_auto_sanitized": preflight.auto_sanitized,
            },
        )
        log_event(
            level="INFO",
            event="refund",
            corr_id=corr_id,
            user_id=user_id,
            username=user.username,
            model=order.model,
            provider=provider_key,
            size=order.size,
            credits_cost=credits_cost,
            refund_done=refunded,
        )
        if notify_support:
            await _notify_support(
                callback.message.bot,
                config,
                event="request",
                corr_id=corr_id,
                user_id=user_id,
                status_code=exc.status_code,
                error_type=exc.error_type,
                hint=hint,
            )
        await callback.message.answer(message)
        return
    except Exception as exc:  # pragma: no cover - API interaction
        log.exception("Failed to submit job")
        refunded = False
        try:
            await db.add_credits(user_id, credits_cost)
            refunded = True
        except Exception:
            log.exception("Failed to refund credits after unexpected error")
        await _release_lock("submit_failed", "unexpected_error")
        short = _shorten(str(exc) or "Неизвестная ошибка")
        error_record = ErrorLogRecord(
            ts=datetime.utcnow(),
            user_id=user_id,
            username=user.username,
            corr_id=corr_id,
            job_id="",
            model=order.model,
            size=order.size,
            status_code=None,
            error_type="internal",
            error_msg_short=short,
            refunded=refunded,
            preflight_blocked=False,
            preflight_reason=preflight.reason or "",
            auto_sanitized=preflight.auto_sanitized,
            sanitized_prompt=order.prompt,
            error_scope=preflight.scope or "unknown",
            error_json="",
            stage="submit",
        )
        sheet_ok = await db.log_error_record(error_record)
        await error_reporter.report(
            error_record,
            context="internal_error",
            extra={
                "gsheets_ok": sheet_ok,
                "provider": provider_key,
                "refunded": refunded,
            },
        )
        log_event(
            level="ERROR",
            event="error",
            corr_id=corr_id,
            user_id=user_id,
            username=user.username,
            model=order.model,
            provider=provider_key,
            size=order.size,
            error_type="internal",
            error_msg_short=short,
            prompt=order.prompt,
            gsheets_ok=sheet_ok,
            refund_done=refunded,
            extra={
                "preflight_blocked": False,
                "preflight_reason": preflight.reason,
                "preflight_scope": preflight.scope,
                "preflight_auto_sanitized": preflight.auto_sanitized,
            },
        )
        log_event(
            level="INFO",
            event="refund",
            corr_id=corr_id,
            user_id=user_id,
            username=user.username,
            model=order.model,
            provider=provider_key,
            size=order.size,
            credits_cost=credits_cost,
            refund_done=refunded,
        )
        await callback.message.answer(
            i18n.t("status.failed", error=escape_html(short) or i18n.t("errors.unknown"))
        )
        return

    session.last_launch_key = key
    session.last_launch_ts = now
    session.awaiting_payment = False
    session.pending_order = None

    queue_pos = await db.count_active_jobs(user_id)
    await callback.message.answer(i18n.t("flow.order_submitted", queue_pos=queue_pos))
    await STATUS_MESSAGES.ensure_started(
        bot=callback.message.bot,
        db=db,
        job=job_record,
    )


@dataclass
class SentMediaInfo:
    message: Message
    method: Literal["video", "document", "photo"]
    file_id: str
    file_size: int
    duration_seconds: Optional[int] = None
    mime_type: Optional[str] = None
    file_bytes: Optional[bytes] = None
    filename: Optional[str] = None


async def _send_job_update(
    dp: Dispatcher,
    job: GenerationJobRecord,
    archive: Optional[ArchivePublisher],
    config: Config,
    db: Database,
    error_reporter: ErrorReporter,
) -> None:
    try:
        await STATUS_MESSAGES.ensure_started(bot=dp.bot, db=db, job=job)
        status = (job.status or "").lower()
        if status in {"queued", "running"}:
            await STATUS_MESSAGES.tick(bot=dp.bot, db=db, job=job)
            return
        if status == "no_media":
            await STATUS_MESSAGES.mark_failed(
                bot=dp.bot,
                db=db,
                job=job,
                text=i18n.t("status.delivery_generic"),
                terminal_state="no_media",
            )
            return
        if status == "failed":
            error_text = i18n.t("status.failed", error=job.error or i18n.t("errors.unknown"))
            await STATUS_MESSAGES.mark_failed(
                bot=dp.bot, db=db, job=job, text=error_text, terminal_state="failed"
            )
            return
        if status == "timeout":
            timeout_text = i18n.t(
                "status.failed", error=job.error or i18n.t("errors.unknown")
            )
            await STATUS_MESSAGES.mark_failed(
                bot=dp.bot,
                db=db,
                job=job,
                text=timeout_text,
                terminal_state="timeout",
            )
            return
        if status == "refunded":
            refunded_text = job.error or i18n.t("status.delivery_generic")
            await STATUS_MESSAGES.mark_failed(
                bot=dp.bot,
                db=db,
                job=job,
                text=refunded_text,
                terminal_state="refunded",
            )
            return
        if status != "completed":
            generic_text = i18n.t(
                "status.generic", status=job.status or i18n.t("errors.unknown")
            )
            await STATUS_MESSAGES.mark_failed(bot=dp.bot, db=db, job=job, text=generic_text)
            return

        sending_text = (
            i18n.t("status.completed_photo")
            if (job.content_type or "video") == "image"
            else i18n.t("status.completed_file")
        )
        await STATUS_MESSAGES.mark_sending(
            bot=dp.bot,
            db=db,
            job=job,
            text=sending_text,
        )

        sent_info: Optional[SentMediaInfo] = None
        inline_assets = _collect_inline_assets(job)
        if inline_assets:
            sent_info = await _send_inline_assets(
                dp, job, inline_assets, config, db, error_reporter
            )
        if sent_info is None and job.video_url:
            inline_image = _parse_inline_image(job.video_url)
            if inline_image:
                sent_info = await _send_inline_image(
                    dp, job, inline_image, config, db, error_reporter
                )
        if sent_info is None:
            remote_asset = _select_remote_media_asset(job)
            if remote_asset and remote_asset.get("url"):
                sent_info = await _deliver_remote_media(
                    dp=dp,
                    job=job,
                    asset=remote_asset,
                    config=config,
                    db=db,
                    error_reporter=error_reporter,
                )

        if sent_info is None:
            await _handle_delivery_failure(
                dp,
                job,
                config,
                db,
                error_reporter=error_reporter,
                message=i18n.t("status.delivery_generic"),
                reason="no_media",
                extra_log={"stage": "no_media_assets"},
            )
            return

        summary_text = _format_media_summary(
            job,
            file_size=sent_info.file_size,
            duration_seconds=sent_info.duration_seconds,
        )
        await STATUS_MESSAGES.mark_completed(
            bot=dp.bot,
            db=db,
            job=job,
            text=summary_text,
        )
        increment_metric("deliver_success_total")
        if archive:
            payload = _build_archive_payload_from_sent(job, sent_info)
            if payload:
                archive.schedule(payload)
    except Exception:  # pragma: no cover - external dependency
        log.exception("Failed to process job update for user_id=%s", job.user_id)


def _parse_inline_image(data_uri: Optional[str]) -> Optional[tuple[str, bytes]]:
    if not data_uri:
        return None
    decoded = _decode_data_uri(data_uri)
    if not decoded:
        return None
    mime, payload = decoded
    if not mime.startswith("image/"):
        return None
    return mime, payload


def _decode_data_uri(data_uri: str) -> Optional[tuple[str, bytes]]:
    uri = (data_uri or "").strip()
    if not uri.lower().startswith("data:"):
        return None
    try:
        header, encoded = uri.split(",", 1)
    except ValueError:
        return None
    if ";base64" not in header.lower():
        return None
    mime = header[5:].split(";", 1)[0] or "application/octet-stream"
    normalized = "".join(encoded.split())
    try:
        payload = base64.b64decode(normalized)
    except (binascii.Error, ValueError):
        return None
    return mime, payload


def _collect_inline_assets(job: GenerationJobRecord) -> List[Dict[str, Any]]:
    extra = getattr(job, "extra", {}) or {}
    raw_assets = extra.get("inline_assets") if isinstance(extra, dict) else None
    if not isinstance(raw_assets, list):
        return []
    assets: List[Dict[str, Any]] = []
    for asset in raw_assets:
        if isinstance(asset, dict) and isinstance(asset.get("data_uri"), str):
            assets.append(asset)
    return assets


def _collect_assets_meta(job: GenerationJobRecord) -> List[Dict[str, Any]]:
    extra = getattr(job, "extra", {}) or {}
    raw_meta = extra.get("assets_meta") if isinstance(extra, dict) else None
    if not isinstance(raw_meta, list):
        return []
    entries: List[Dict[str, Any]] = []
    for item in raw_meta:
        if isinstance(item, dict):
            entries.append(dict(item))
    return entries


def _expected_key_mask(job: GenerationJobRecord) -> Optional[str]:
    extra = getattr(job, "extra", {}) or {}
    mask = extra.get("gemini_key_mask") if isinstance(extra, dict) else None
    if isinstance(mask, str) and mask:
        return mask
    return None


def _select_remote_media_asset(job: GenerationJobRecord) -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    target_media = (job.content_type or "video").lower()
    for asset in _collect_assets_meta(job):
        if asset.get("inline"):
            continue
        url = asset.get("url") or asset.get("file_id")
        if not isinstance(url, str) or not url:
            continue
        mime = str(asset.get("mime") or "")
        asset_type = str(asset.get("type") or "")
        asset_kind = str(asset.get("kind") or "")
        if target_media == "image":
            if not (
                mime.startswith("image/")
                or asset_type.lower() == "image"
                or asset_kind.lower() == "image"
            ):
                continue
        else:
            if not (
                mime.startswith("video/")
                or asset_type.lower() == "video"
                or asset_kind.lower() == "video"
            ):
                continue
        score = 1
        if asset.get("preferred"):
            score += 10
        candidates.append((score, asset))
    if not candidates:
        primary_fallback = (
            job.file_url if isinstance(job.file_url, str) and job.file_url else job.video_url
        )
        fallback = primary_fallback if isinstance(primary_fallback, str) else None
        if fallback:
            candidate = fallback.strip()
            if candidate and not candidate.startswith("["):
                if candidate.startswith(("http://", "https://", "files/", "file-", "/tmp/")):
                    return {
                        "key": "media_url",
                        "url": candidate,
                        "kind": target_media or "video",
                        "fallback": True,
                    }
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _format_video_summary(duration_seconds: Optional[int], size_bytes: int) -> str:
    size_mb = max(size_bytes / (1024 * 1024), 0)
    if duration_seconds and duration_seconds > 0:
        return i18n.t(
            "status.completed_video_summary",
            duration=duration_seconds,
            size_mb=f"{size_mb:.1f}",
        )
    return i18n.t(
        "status.completed_video_summary_short",
        size_mb=f"{size_mb:.1f}",
    )


def _format_media_summary(
    job: GenerationJobRecord,
    *,
    file_size: int,
    duration_seconds: Optional[int],
) -> str:
    if (job.content_type or "video") == "image":
        size_mb = max(file_size / (1024 * 1024), 0)
        return i18n.t("status.completed_image_summary", size_mb=f"{size_mb:.1f}")
    return _format_video_summary(duration_seconds, file_size)


def _extract_video_duration(job: GenerationJobRecord) -> Optional[int]:
    if isinstance(job.seconds, int) and job.seconds > 0:
        return job.seconds
    extra = getattr(job, "extra", {}) or {}
    primary = extra.get("primary_asset") if isinstance(extra, dict) else None
    if isinstance(primary, dict):
        try:
            duration = int(primary.get("duration_seconds") or 0)
        except (TypeError, ValueError):
            duration = 0
        if duration > 0:
            return duration
    payload = extra.get("payload") if isinstance(extra, dict) else None
    if isinstance(payload, dict):
        config = payload.get("config")
        if isinstance(config, dict):
            value = config.get("duration_seconds")
            try:
                duration_int = int(value)
            except (TypeError, ValueError):
                duration_int = None
            if duration_int and duration_int > 0:
                return duration_int
    return None


async def _send_inline_image(
    dp: Dispatcher,
    job: GenerationJobRecord,
    inline_image: tuple[str, bytes],
    config: Config,
    db: Database,
    error_reporter: ErrorReporter,
) -> Optional[SentMediaInfo]:
    mime, payload = inline_image
    filename = "image.png"
    input_file = BufferedInputFile(payload, filename=filename, mime_type=mime)
    try:
        message = await dp.bot.send_photo(job.user_id, input_file)
    except Exception:
        log.exception("Failed to send inline image to user_id=%s", job.user_id)
        await _handle_delivery_failure(
            dp,
            job,
            config,
            db,
            error_reporter=error_reporter,
            message=i18n.t("status.delivery_generic"),
            reason="telegram_error",
            extra_log={"stage": "inline_image"},
        )
        return None
    photo = message.photo[-1] if message.photo else None
    if photo is None:
        size = len(payload)
        file_id = ""
    else:
        size = photo.file_size or len(payload)
        file_id = photo.file_id
    return SentMediaInfo(
        message=message,
        method="photo",
        file_id=file_id,
        file_size=size,
        duration_seconds=None,
        mime_type=mime,
        file_bytes=payload,
        filename=filename,
    )


async def _send_inline_assets(
    dp: Dispatcher,
    job: GenerationJobRecord,
    assets: List[Dict[str, Any]],
    config: Config,
    db: Database,
    error_reporter: ErrorReporter,
) -> Optional[SentMediaInfo]:
    send_as_photo = (job.content_type or "video") == "image"
    for index, asset in enumerate(assets):
        data_uri = asset.get("data_uri")
        if not isinstance(data_uri, str):
            continue
        decoded = _decode_data_uri(data_uri)
        if not decoded:
            await _handle_delivery_failure(
                dp,
                job,
                config,
                db,
                error_reporter=error_reporter,
                message=i18n.t("status.inline_decode_failed"),
                reason="inline_decode",
                extra_log={"asset_index": index},
            )
            return None
        mime, payload = decoded
        size = len(payload)
        if size > _INLINE_ASSET_MAX_BYTES:
            await _handle_delivery_failure(
                dp,
                job,
                config,
                db,
                error_reporter=error_reporter,
                message=i18n.t(
                    "status.inline_too_large",
                    mime=mime,
                    size=size,
                    limit_mb=_INLINE_ASSET_MAX_BYTES // (1024 * 1024),
                ),
                reason="inline_too_large",
                extra_log={"asset_index": index, "asset_size": size},
            )
            return None
        suffix = mimetypes.guess_extension(mime) or ".bin"
        base_name = re.sub(r"[^a-zA-Z0-9_-]", "", str(asset.get("kind") or "asset"))
        if not base_name:
            base_name = "asset"
        filename = f"{base_name}_{index}{suffix}"
        if send_as_photo and mime.startswith("image/"):
            filename = "image.png"
        try:
            if send_as_photo and mime.startswith("image/") and size <= _INLINE_ASSET_MAX_BYTES:
                message = await dp.bot.send_photo(
                    job.user_id,
                    BufferedInputFile(payload, filename=filename, mime_type=mime),
                )
                method: Literal["photo", "document"] = "photo"
            else:
                input_file = BufferedInputFile(payload, filename=filename, mime_type=mime)
                message = await dp.bot.send_document(job.user_id, input_file)
                method = "document"
        except Exception:
            log.exception("Failed to send inline asset to user_id=%s", job.user_id)
            await _handle_delivery_failure(
                dp,
                job,
                config,
                db,
                error_reporter=error_reporter,
                message=i18n.t("status.delivery_generic"),
                reason="telegram_error",
                extra_log={"stage": "inline_asset", "asset_index": index},
            )
            return None
        document = message.document
        if method == "photo":
            photo = message.photo[-1] if message.photo else None
            file_size = photo.file_size if photo else size
            file_id = photo.file_id if photo else ""
        else:
            file_size = document.file_size if document else size
            file_id = document.file_id if document else ""
        return SentMediaInfo(
            message=message,
            method=method,
            file_id=file_id,
            file_size=file_size or size,
            duration_seconds=None,
            mime_type=mime,
            file_bytes=payload,
            filename=filename,
        )
    return None


async def _deliver_remote_media(
    *,
    dp: Dispatcher,
    job: GenerationJobRecord,
    asset: Dict[str, Any],
    config: Config,
    db: Database,
    error_reporter: ErrorReporter,
) -> Optional[SentMediaInfo]:
    corr_id = job.corr_id or job.id
    asset_name = str(asset.get("key") or "video")
    mime_hint = asset.get("mime") if isinstance(asset.get("mime"), str) else None
    filename_hint = asset.get("filename") if isinstance(asset.get("filename"), str) else None
    asset_reference = asset.get("url") or asset.get("file_id")
    if not isinstance(asset_reference, str) or not asset_reference:
        log.error("Missing asset reference for job_id=%s asset=%s", job.id, asset)
        await _handle_delivery_failure(
            dp,
            job,
            config,
            db,
            error_reporter=error_reporter,
            message=i18n.t("status.delivery_generic"),
            reason="missing_asset",
            extra_log={"asset": asset},
            key_mask=_expected_key_mask(job),
        )
        return None
    expected_mask = _expected_key_mask(job)
    provider_key = _provider_for_model(job.model, config)
    target_media = (job.content_type or "video").lower()

    if asset_reference.startswith("/tmp/"):
        from pathlib import Path
        try:
            local_path = Path(asset_reference)
            if not local_path.exists():
                log.error("Local file not found: %s job_id=%s", asset_reference, job.id)
                await _handle_delivery_failure(
                    dp,
                    job,
                    config,
                    db,
                    error_reporter=error_reporter,
                    message=i18n.t("status.delivery_generic"),
                    reason="file_not_found",
                    extra_log={"local_path": asset_reference},
                    key_mask=expected_mask,
                )
                return None
            content = local_path.read_bytes()
            size = len(content)
            downloaded = type('DownloadedAsset', (), {
                'content': content,
                'size': size,
                'mime': mime_hint or 'video/mp4',
                'filename': filename_hint or local_path.name,
                'key_mask': expected_mask
            })()
        except Exception as exc:
            log.exception("Failed to read local file %s job_id=%s", asset_reference, job.id)
            await _handle_delivery_failure(
                dp,
                job,
                config,
                db,
                error_reporter=error_reporter,
                message=i18n.t("status.delivery_generic"),
                reason="file_read_error",
                extra_log={"local_path": asset_reference, "error": str(exc)},
                key_mask=expected_mask,
            )
            return None
    else:
        try:
            if provider_key == "sora":
                downloaded = await download_sora_asset(
                    asset_url=str(asset_reference),
                    config=config,
                    corr_id=corr_id or "",
                    asset_name=asset_name,
                    expected_mask=expected_mask,
                    mime_hint=mime_hint,
                    filename_hint=filename_hint,
                )
            else:
                downloaded = await download_asset(
                    asset_url=str(asset_reference),
                    config=config,
                    corr_id=corr_id or "",
                    asset_name=asset_name,
                    expected_mask=expected_mask,
                    mime_hint=mime_hint,
                    filename_hint=filename_hint,
                )
        except GeminiKeyMismatchError as exc:
            log.error(
                "Gemini key mismatch corr_id=%s expected=%s actual=%s",
                corr_id,
                exc.expected,
                exc.actual,
            )
            await _handle_delivery_failure(
                dp,
                job,
                config,
                db,
                error_reporter=error_reporter,
                message=i18n.t("status.delivery_key_mismatch"),
                reason="key_mismatch",
                extra_log={"expected_mask": exc.expected, "actual_mask": exc.actual},
                key_mask=exc.actual,
            )
            return None
        except GeminiConfigurationError as exc:
            log.error(
                "Gemini configuration error corr_id=%s status=%s",
                corr_id,
                exc.error_status,
            )
            await _handle_delivery_failure(
                dp,
                job,
                config,
                db,
                error_reporter=error_reporter,
                message=i18n.t("status.delivery_config_error"),
                reason="config_error",
                extra_log={"error_status": exc.error_status},
                key_mask=expected_mask,
            )
            return None
        except SoraDownloadError as exc:
            log.warning(
                "Sora download failed corr_id=%s status=%s", corr_id, getattr(exc, "status_code", 0)
            )
            await _handle_delivery_failure(
                dp,
                job,
                config,
                db,
                error_reporter=error_reporter,
                message=i18n.t("status.delivery_generic"),
                reason="download_error",
                extra_log={"status_code": getattr(exc, "status_code", 0)},
                key_mask=None,
            )
            return None
        except GeminiDownloadError as exc:
            log.warning(
                "Gemini download failed corr_id=%s status=%s error=%s",
                corr_id,
                exc.status_code,
                exc.error_status,
            )
            await _handle_delivery_failure(
                dp,
                job,
                config,
                db,
                error_reporter=error_reporter,
                message=i18n.t("status.delivery_generic"),
                reason="download_error",
                extra_log={
                    "status_code": exc.status_code,
                    "error_status": exc.error_status,
                },
                key_mask=expected_mask,
            )
            return None

    method: Literal["video", "document", "photo"] = "video"
    message: Optional[Message] = None
    duration_seconds = _extract_video_duration(job)
    download_mime = downloaded.mime or mime_hint or ""
    send_as_image = target_media == "image" and download_mime.startswith("image/")
    filename = downloaded.filename or (
        "image.png" if send_as_image else "video.mp4"
    )
    if send_as_image:
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            filename = "image.png"
    elif not filename.lower().endswith(".mp4"):
        filename = "video.mp4"

    try:
        if send_as_image:
            if downloaded.size <= _TELEGRAM_PHOTO_MAX_BYTES:
                message = await dp.bot.send_photo(
                    job.user_id,
                    BufferedInputFile(
                        downloaded.content,
                        filename=filename,
                        mime_type=downloaded.mime,
                    ),
                )
                method = "photo"
            elif downloaded.size <= _TELEGRAM_DOCUMENT_MAX_BYTES:
                message = await dp.bot.send_document(
                    job.user_id,
                    BufferedInputFile(
                        downloaded.content,
                        filename=filename,
                        mime_type=downloaded.mime,
                    ),
                )
                method = "document"
            else:
                await _handle_delivery_failure(
                    dp,
                    job,
                    config,
                    db,
                    error_reporter=error_reporter,
                    message=i18n.t("status.delivery_too_large"),
                    reason="too_large",
                    extra_log={"asset_bytes": downloaded.size},
                    key_mask=downloaded.key_mask or expected_mask,
                )
                return None
        else:
            if downloaded.size <= _TELEGRAM_VIDEO_MAX_BYTES:
                message = await dp.bot.send_video(
                    job.user_id,
                    BufferedInputFile(
                        downloaded.content,
                        filename=filename,
                        mime_type=downloaded.mime,
                    ),
                    supports_streaming=True,
                    duration=duration_seconds or None,
                )
            elif downloaded.size <= _TELEGRAM_DOCUMENT_MAX_BYTES:
                message = await dp.bot.send_document(
                    job.user_id,
                    BufferedInputFile(
                        downloaded.content,
                        filename=filename,
                        mime_type=downloaded.mime,
                    ),
                )
                method = "document"
            else:
                await _handle_delivery_failure(
                    dp,
                    job,
                    config,
                    db,
                    error_reporter=error_reporter,
                    message=i18n.t("status.delivery_too_large"),
                    reason="too_large",
                    extra_log={"asset_bytes": downloaded.size},
                    key_mask=downloaded.key_mask or expected_mask,
                )
                return None
    except Exception:
        log.exception("Failed to send Gemini video to user_id=%s", job.user_id)
        await _handle_delivery_failure(
            dp,
            job,
            config,
            db,
            error_reporter=error_reporter,
            message=i18n.t("status.delivery_generic"),
            reason="telegram_error",
            extra_log={"asset_bytes": downloaded.size},
            key_mask=downloaded.key_mask or expected_mask,
        )
        return None

    if message is None:
        return None

    file_id: str
    file_size: int
    duration_seconds = duration_seconds or _extract_video_duration(job)
    mime_type = downloaded.mime
    if method == "video" and message.video:
        file_id = message.video.file_id
        file_size = message.video.file_size or downloaded.size
        duration_seconds = message.video.duration or duration_seconds
    elif method == "document" and message.document:
        file_id = message.document.file_id
        file_size = message.document.file_size or downloaded.size
    elif method == "photo" and message.photo:
        photo = message.photo[-1]
        file_id = photo.file_id
        file_size = photo.file_size or downloaded.size
    else:
        file_id = ""
        file_size = downloaded.size

    if isinstance(job.created_at, datetime):
        elapsed = (datetime.utcnow() - job.created_at).total_seconds()
        if elapsed >= 0:
            record_timing_metric("first_frame_seconds", elapsed)

    increment_metric("delivery_success")
    log_event(
        level="INFO",
        event="delivery",
        corr_id=job.corr_id,
        job_id=job.id,
        user_id=job.user_id,
        username=job.username,
        model=job.model,
        provider=provider_key,
        size=job.size,
        extra={
            "delivery_method": method,
            "asset_bytes": downloaded.size,
            "gemini_key_mask": downloaded.key_mask or expected_mask,
            "archive_status": "pending",
            "duration_seconds": duration_seconds,
        },
    )
    return SentMediaInfo(
        message=message,
        method=method,
        file_id=file_id,
        file_size=file_size,
        duration_seconds=duration_seconds,
        mime_type=mime_type,
        file_bytes=downloaded.content,
        filename=downloaded.filename,
    )


async def _handle_delivery_failure(
    dp: Dispatcher,
    job: GenerationJobRecord,
    config: Config,
    db: Database,
    *,
    error_reporter: ErrorReporter,
    message: str,
    reason: str,
    extra_log: Optional[Dict[str, Any]] = None,
    key_mask: Optional[str] = None,
) -> None:
    credits = job.cost_credits or config.generation_cost_credits
    refunded = False
    if credits > 0:
        try:
            await db.add_credits(job.user_id, credits)
            refunded = True
        except Exception:
            log.exception("Failed to refund credits after delivery failure job_id=%s", job.id)
    if refunded:
        increment_metric("refunds_total")
    try:
        await db.update_job(job.id, "failed", file_url="", error=message)
    except Exception:
        log.exception("Failed to update job status after delivery failure job_id=%s", job.id)
    job.status = "failed"
    job.error = message
    job.file_url = ""
    await STATUS_MESSAGES.mark_failed(
        bot=dp.bot,
        db=db,
        job=job,
        text=message,
        terminal_state="refunded" if refunded else "failed",
    )
    extra_payload = dict(extra_log or {})
    if key_mask:
        extra_payload.setdefault("gemini_key_mask", key_mask)
    extra_payload.setdefault("delivery_reason", reason)
    stage = "download" if reason in {"download_error", "key_mismatch", "config_error"} else "deliver"
    if reason == "no_media":
        stage = "poll"
    status_code_value = None
    provider_error_code = ""
    provider_key = _provider_for_model(job.model, config)
    if isinstance(extra_log, dict):
        raw_status = extra_log.get("status_code")
        if isinstance(raw_status, int):
            status_code_value = raw_status
        provider_error_code = str(extra_log.get("error_status") or raw_status or extra_log.get("status") or "")
        provider_error_message = str(extra_log.get("error_status") or extra_log.get("reason_message") or "")
    else:
        provider_error_message = ""
    error_record = ErrorLogRecord(
        ts=datetime.utcnow(),
        user_id=job.user_id,
        username=job.username,
        corr_id=job.corr_id,
        job_id=job.id,
        model=job.model,
        size=job.size,
        status_code=status_code_value,
        error_type=reason,
        error_msg_short=message[:240],
        refunded=refunded,
        preflight_blocked=False,
        preflight_reason="",
        auto_sanitized=False,
        sanitized_prompt=job.prompt,
        error_scope="delivery",
        error_json="",
        job_status="failed",
        reason=reason,
        provider_error_code=provider_error_code[:120],
        provider_error_message=provider_error_message[:240],
        stage=stage,
    )
    sheet_ok = await db.log_error_record(error_record)
    await error_reporter.report(
        error_record,
        context=f"delivery_{reason}",
        extra={
            **extra_payload,
            "gsheets_ok": sheet_ok,
            "provider": provider_key,
            "refunded": refunded,
        },
    )
    log_event(
        level="ERROR",
        event="delivery_failed",
        corr_id=job.corr_id,
        job_id=job.id,
        user_id=job.user_id,
        username=job.username,
        model=job.model,
        provider=provider_key,
        size=job.size,
        extra=extra_payload,
    )
    log_event(
        level="INFO",
        event="refund",
        corr_id=job.corr_id,
        job_id=job.id,
        user_id=job.user_id,
        username=job.username,
        model=job.model,
        provider=provider_key,
        size=job.size,
        credits_cost=config.generation_cost_credits,
        refund_done=refunded,
        extra={"gemini_key_mask": key_mask},
    )


def _build_archive_payload_from_sent(
    job: GenerationJobRecord, sent: SentMediaInfo
) -> Optional[ArchivePayload]:
    file_bytes = sent.file_bytes if sent.file_bytes is not None else None
    if file_bytes is None and not sent.file_id:
        return None
    content_type = "image" if (job.content_type or "video") == "image" else "video"
    duration = sent.duration_seconds if content_type == "video" else None
    payload = ArchivePayload(
        corr_id=job.corr_id or job.id,
        prompt=job.prompt,
        model_name=job.model,
        username=job.username,
        user_id=job.user_id,
        content_type=content_type,
        mime_type=sent.mime_type,
        file_bytes=bytes(file_bytes) if file_bytes is not None else None,
        file_id=None if file_bytes is not None else sent.file_id,
        file_size=len(file_bytes) if file_bytes is not None else sent.file_size,
        filename=sent.filename,
        duration_seconds=duration,
        delivery_method=sent.method,
        sent_at=datetime.utcnow(),
    )
    return payload


async def start_command(
    message: Message, db: Database, state: FSMContext, config: Config
) -> None:
    await state.finish()
    await _ensure_user(message, db)
    await _send_main_menu(message, config)


async def help_command(
    message: Message, db: Database, state: FSMContext, config: Config
) -> None:
    await _ensure_user(message, db)
    await message.answer(
        i18n.t(
            "help.main",
            video_models=_video_models_description(config),
            image_model=_image_model_description(config),
        ),
        reply_markup=_build_help_keyboard(config),
    )


async def chatgpt_menu(
    message: Message,
    state: FSMContext,
    config: Config,
    chat_client: Optional[OpenAIChatClient],
) -> None:
    await state.finish()
    if chat_client is None:
        await message.answer(i18n.t("errors.chatgpt_unavailable"))
        await _send_main_menu(message, config)
        return
    await ChatGPTState.awaiting_input.set()
    await message.answer(i18n.t("help.chatgpt"), reply_markup=ReplyKeyboardRemove())


async def chatgpt_respond(
    message: Message,
    state: FSMContext,
    config: Config,
    chat_client: Optional[OpenAIChatClient],
) -> None:
    if chat_client is None:
        await message.answer(i18n.t("errors.chatgpt_unavailable"))
        await state.finish()
        await _send_main_menu(message, config)
        return
    user_text = (message.text or "").strip()
    if not user_text:
        await message.answer(i18n.t("errors.chatgpt_empty"))
        return
    try:
        response = await chat_client.generate_reply(user_text)
    except Exception:
        log.exception("ChatGPT response failed")
        await message.answer(i18n.t("errors.chatgpt_failed"))
        return
    await message.answer(response)


async def balance_command(message: Message, db: Database, state: FSMContext) -> None:
    await _send_balance_info(message, db)


async def generate_video_menu(
    message: Message, db: Database, state: FSMContext, config: Config
) -> None:
    await state.finish()
    user_id = await _ensure_user(message, db)
    SESSION_MANAGER.get(user_id)
    options = _video_model_options(config)
    if not options:
        await message.answer(i18n.t("errors.video_models_unavailable"))
        await _send_main_menu(message, config)
        return
    keyboard = _build_video_models_keyboard(options)
    default_model = config.default_video_model
    default_provider = _provider_for_model(default_model, config)
    default_product = _resolve_product_key("video", default_provider, default_model)
    await state.update_data(
        flow_type="video",
        model=None,
        provider=None,
        model_label=None,
        include_size=True,
        product=default_product,
    )
    await GenerationStates.video_model.set()
    await message.answer(i18n.t("video.models.prompt"), reply_markup=keyboard)


async def generate_image_menu(
    message: Message, db: Database, state: FSMContext, config: Config
) -> None:
    await state.finish()
    user_id = await _ensure_user(message, db)
    SESSION_MANAGER.get(user_id)
    if not config.gemini_enabled:
        await message.answer(i18n.t("errors.image_generation_disabled"))
        await _send_main_menu(message, config)
        return
    model_label = i18n.t("image.model.gemini")
    await state.update_data(
        flow_type="image",
        model=config.gemini_model_image,
        provider="gemini-image",
        model_label=model_label,
        include_size=False,
        product=_resolve_product_key("image", "gemini-image", config.gemini_model_image),
    )
    await GenerationStates.image_mode.set()
    await message.answer(
        i18n.t("image.mode.prompt"),
        reply_markup=_build_mode_keyboard("image"),
    )


async def help_button(
    message: Message, db: Database, state: FSMContext, config: Config
) -> None:
    await help_command(message, db, state, config)


async def top_up_menu(message: Message, db: Database, state: FSMContext, config: Config) -> None:
    await state.finish()
    await _ensure_user(message, db)
    await _send_payment_showcase(message.bot, message.chat.id, config)


async def handle_text_input(
    message: Message,
    state: FSMContext,
    db: Database,
    config: Config,
) -> None:
    user_id = await _ensure_user(message, db)
    session = SESSION_MANAGER.get(user_id)
    state_data = await state.get_data()
    category = state_data.get("flow_type", "video")
    model = state_data.get("model") or config.default_video_model
    provider = state_data.get("provider") or model
    model_label = state_data.get("model_label") or _resolve_model_label(model, config)
    include_size = bool(state_data.get("include_size", category == "video"))
    model, provider, model_label = await _ensure_supported_video_model(
        message,
        category=category,
        model=model,
        provider=provider,
        model_label=model_label,
        config=config,
    )
    product = _resolve_product_key(category, provider, model, state_data.get("product"))
    raw_prompt = (message.text or "").strip()
    prompt = raw_prompt
    if _detect_pro_request(raw_prompt):
        await _notify_pro_unavailable(message)
        prompt = _strip_pro_directives(raw_prompt)
    if not prompt:
        await message.answer(i18n.t("flow.no_prompt"))
        return
    await state.finish()
    order = _create_order(
        category=category,
        flow="text",
        prompt=prompt,
        config=config,
        session=session,
        model=model,
        provider=provider,
        product=product,
        model_label=model_label,
        include_size=include_size,
    )
    await _show_confirmation_and_balance(
        message=message,
        order=order,
        session=session,
        db=db,
        config=config,
    )


async def handle_photo_input(
    message: Message,
    state: FSMContext,
    db: Database,
    config: Config,
) -> None:
    user_id = await _ensure_user(message, db)
    session = SESSION_MANAGER.get(user_id)
    state_data = await state.get_data()
    category = state_data.get("flow_type", "video")
    model = state_data.get("model") or config.default_video_model
    provider = state_data.get("provider") or model
    model_label = state_data.get("model_label") or _resolve_model_label(model, config)
    include_size = bool(state_data.get("include_size", category == "video"))
    model, provider, model_label = await _ensure_supported_video_model(
        message,
        category=category,
        model=model,
        provider=provider,
        model_label=model_label,
        config=config,
    )
    product = _resolve_product_key(category, provider, model, state_data.get("product"))
    if not message.photo:
        await _send_generation_prompt(
            message,
            session=session,
            category=category,
            mode="photo",
            model_label=model_label,
            include_size=include_size,
            config=config,
        )
        return
    largest = max(message.photo, key=lambda item: item.file_size or 0)
    raw_caption = (message.caption or "").strip()
    caption = raw_caption
    if _detect_pro_request(raw_caption):
        await _notify_pro_unavailable(message)
        caption = _strip_pro_directives(raw_caption)
    order = _create_order(
        category=category,
        flow="photo",
        prompt=caption,
        config=config,
        session=session,
        model=model,
        provider=provider,
        product=product,
        model_label=model_label,
        include_size=include_size,
        image_file_id=largest.file_id,
    )
    await state.finish()
    await _show_confirmation_and_balance(
        message=message,
        order=order,
        session=session,
        db=db,
        config=config,
    )


async def video_model_callback_handler(
    callback: CallbackQuery, state: FSMContext, config: Config
) -> None:
    await safe_callback_answer(callback)
    parts = (callback.data or "").split(":")
    if len(parts) != 3:
        return
    _, _, key = parts
    option = _find_video_model_option(config, key)
    if option is None:
        await safe_callback_answer(callback, "Модель недоступна", show_alert=True)
        return
    await state.update_data(
        flow_type="video",
        model=option.model,
        provider=option.provider,
        model_label=option.label,
        include_size=True,
        product=_resolve_product_key("video", option.provider, option.model),
    )
    await GenerationStates.video_mode.set()
    try:
        await callback.message.edit_text(
            i18n.t("video.mode.prompt"),
            reply_markup=_build_mode_keyboard("video"),
        )
    except Exception:  # pragma: no cover - Telegram edits may fail
        await callback.message.answer(
            i18n.t("video.mode.prompt"), reply_markup=_build_mode_keyboard("video")
        )


async def video_mode_callback_handler(
    callback: CallbackQuery, state: FSMContext, config: Config
) -> None:
    await safe_callback_answer(callback)
    parts = (callback.data or "").split(":")
    if len(parts) != 3:
        return
    _, _, mode = parts
    if mode not in {"text", "photo"}:
        return
    user = callback.from_user
    if user is None:  # pragma: no cover - defensive
        return
    session = SESSION_MANAGER.get(user.id)
    data = await state.get_data()
    model = data.get("model") or config.default_video_model
    provider = data.get("provider") or model
    model_label = data.get("model_label") or _resolve_model_label(model, config)
    include_size = True
    product = _resolve_product_key("video", provider, model, data.get("product"))
    await state.update_data(
        flow_type="video",
        mode=mode,
        model=model,
        provider=provider,
        model_label=model_label,
        include_size=include_size,
        product=product,
    )
    target_state = GenerationStates.text_prompt if mode == "text" else GenerationStates.photo_prompt
    await target_state.set()
    try:
        await callback.message.edit_reply_markup()
    except Exception:  # pragma: no cover - Telegram edits may fail
        log.debug("Failed to clear mode keyboard", exc_info=True)
    await _send_generation_prompt(
        callback.message,
        session=session,
        category="video",
        mode=mode,
        model_label=model_label,
        include_size=include_size,
        config=config,
    )


async def image_mode_callback_handler(
    callback: CallbackQuery, state: FSMContext, config: Config
) -> None:
    await safe_callback_answer(callback)
    parts = (callback.data or "").split(":")
    if len(parts) != 3:
        return
    _, _, mode = parts
    if mode not in {"text", "photo"}:
        return
    user = callback.from_user
    if user is None:  # pragma: no cover - defensive
        return
    session = SESSION_MANAGER.get(user.id)
    data = await state.get_data()
    model = data.get("model") or config.gemini_model_image
    provider = data.get("provider") or "gemini-image"
    model_label = data.get("model_label") or _resolve_model_label(model, config)
    product = _resolve_product_key("image", provider, model, data.get("product"))
    await state.update_data(
        flow_type="image",
        mode=mode,
        model=model,
        provider=provider,
        model_label=model_label,
        include_size=False,
        product=product,
    )
    target_state = GenerationStates.text_prompt if mode == "text" else GenerationStates.photo_prompt
    await target_state.set()
    try:
        await callback.message.edit_reply_markup()
    except Exception:  # pragma: no cover - Telegram edits may fail
        log.debug("Failed to clear mode keyboard", exc_info=True)
    await _send_generation_prompt(
        callback.message,
        session=session,
        category="image",
        mode=mode,
        model_label=model_label,
        include_size=False,
        config=config,
    )


async def option_callback_handler(
    callback: CallbackQuery, state: FSMContext, config: Config
) -> None:
    data = (callback.data or "").split(":")
    if len(data) < 3:
        await safe_callback_answer(callback)
        return
    _, option_type, context, *rest = data
    user = callback.from_user
    if user is None:  # pragma: no cover - defensive
        await safe_callback_answer(callback)
        return
    session = SESSION_MANAGER.get(user.id)
    token = rest[-1] if rest else ""
    if option_type == "size":
        value = SIZE_OPTIONS.get(token)
        if value:
            session.last_size = value
            aspect = ASPECT_RATIO_OPTIONS.get(token)
            if aspect:
                session.last_aspect_ratio = aspect
                session.update_video_size()
    elif option_type == "duration":
        try:
            duration_value = int(token)
        except ValueError:
            duration_value = None
        if duration_value in VIDEO_DURATION_OPTIONS:
            session.video_duration = duration_value
    elif option_type == "aspect":
        ratio = ASPECT_RATIO_OPTIONS.get(token)
        if ratio:
            session.last_aspect_ratio = ratio
            session.update_video_size()
    elif option_type == "hd":
        session.hd_enabled = not session.hd_enabled
        session.update_video_size()
    await safe_callback_answer(callback)
    state_data = await state.get_data()
    if state_data.get("flow_type", "video") != "video":
        return
    model = state_data.get("model") or config.default_video_model
    model_label = state_data.get("model_label") or _resolve_model_label(model, config)
    mode = context if context in {"text", "photo"} else "text"
    try:
        body = _video_prompt_body(
            session=session, mode=mode, model_label=model_label, config=config
        )
        await callback.message.edit_text(
            body,
            reply_markup=_build_video_settings_keyboard(session, mode),
        )
    except Exception:  # pragma: no cover - Telegram may block edits on old messages
        log.debug("Failed to edit options message", exc_info=True)


async def noop_callback_handler(callback: CallbackQuery) -> None:
    await safe_callback_answer(callback)


async def start_generation_callback_handler(callback: CallbackQuery) -> None:
    await safe_callback_answer(callback, i18n.t("video.prompt.start_hint"))


async def order_callback_handler(
    callback: CallbackQuery,
    state: FSMContext,
    db: Database,
    job_queue: JobQueue,
    config: Config,
    gate: GenerationRequestGate,
    error_reporter: ErrorReporter,
) -> None:
    await safe_callback_answer(callback)
    action = (callback.data or "").split(":", maxsplit=1)[-1]
    user = callback.from_user
    if user is None:  # pragma: no cover - defensive
        return
    session = SESSION_MANAGER.get(user.id)
    order = session.pending_order
    if action == "busy":
        return
    if action == "back":
        session.pending_order = None
        session.awaiting_payment = False
        await state.finish()
        await callback.message.answer(
            i18n.t("help.main"), reply_markup=_main_keyboard(config)
        )
        return
    if not order:
        await callback.message.answer(i18n.t("flow.no_prompt"))
        return
    if action == "edit":
        await state.finish()
        if order.flow == "photo":
            await GenerationStates.photo_prompt.set()
        else:
            await GenerationStates.text_prompt.set()
        model_label = order.model_label or _resolve_model_label(order.model, config)
        include_size = order.category == "video"
        await state.update_data(
            flow_type=order.category,
            mode=order.flow,
            model=order.model,
            provider=order.provider,
            model_label=model_label,
            include_size=include_size,
            product=_resolve_product_key(
                order.category, order.provider, order.model, order.product
            ),
        )
        await _send_generation_prompt(
            callback.message,
            session=session,
            category=order.category,
            mode=order.flow,
            model_label=model_label,
            include_size=include_size,
            config=config,
        )
        return
    if action == "launch":
        await _launch_order(
            callback=callback,
            order=order,
            session=session,
            db=db,
            job_queue=job_queue,
            config=config,
            gate=gate,
            error_reporter=error_reporter,
        )


async def payment_callback_handler(
    callback: CallbackQuery,
    db: Database,
    config: Config,
) -> None:
    await safe_callback_answer(callback)
    if not config.yookassa_ready:
        await callback.message.answer(i18n.t("payment.unavailable"))
        return
    data = callback.data or ""
    try:
        _, package_id = data.split(":", maxsplit=1)
    except (ValueError, AttributeError):
        return
    package = config.get_credit_package(package_id)
    if package is None:
        await callback.message.answer(i18n.t("payment.unknown_package"))
        return

    user = callback.from_user
    if user is None:  # pragma: no cover - defensive
        return
    await db.ensure_user(
        user.id,
        user.username,
        first_name=user.first_name,
        last_name=user.last_name,
    )

    description = f"Video generation {package.credits_int} credits"
    try:
        payment = await asyncio.to_thread(
            yookassa_client.create_payment,
            package,
            user.id,
            description,
        )
    except Exception:  # pragma: no cover - external API
        log.exception("Failed to create YooKassa payment")
        await callback.message.answer(i18n.t("payment.creation_failed"))
        return

    confirmation_url = payment.get("confirmation_url")
    payment_id = payment.get("payment_id")
    metadata_raw = payment.get("metadata", "{}")
    metadata = metadata_raw if isinstance(metadata_raw, str) else str(metadata_raw)
    if not confirmation_url or not payment_id:
        await callback.message.answer(i18n.t("payment.link_failed"))
        return

    status = payment.get("status", "pending") or "pending"
    idempotency_key = payment.get("idempotency_key", "")
    metadata = payment.get("metadata", {})
    amount_cp = int(payment.get("amount_cp", package.price_kopeks))
    await db.create_payment_record(
        "yookassa",
        payment_id,
        user.id,
        amount_cp,
        package.credits_int,
        status,
        payload=str(description),
        metadata=metadata,
        idempotency_key=idempotency_key,
        username=user.username,
        package_id=package.package_id,
        purchased_credits=package.credits_int,
    )
    log.info(
        "Created YooKassa payment %s credits=%s amount_cp=%s net_cp=%s",
        payment_id,
        package.credits_int,
        amount_cp,
        amount_cp,
    )

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=i18n.t("buttons.payment_open"),
                    url=confirmation_url,
                )
            ]
        ]
    )
    await callback.message.answer(i18n.t("payment.store.instructions"), reply_markup=keyboard)


async def menu_callback_handler(
    callback: CallbackQuery,
    db: Database,
    config: Config,
    state: FSMContext,
) -> None:
    await safe_callback_answer(callback)
    data = callback.data or ""
    if data.endswith("topup"):
        await _send_payment_showcase(callback.message.bot, callback.message.chat.id, config)
        return


async def resend_pending_order(
    *,
    bot: Bot,
    db: Database,
    config: Config,
    user_id: int,
) -> bool:
    session = SESSION_MANAGER.get(user_id)
    order = session.pending_order
    if not order:
        return False
    credits = await db.get_user_credits(user_id)
    required = order.credits_cost or config.generation_cost_credits
    can_launch = credits >= required
    session.awaiting_payment = not can_launch
    await _send_order_confirmation(
        bot=bot,
        chat_id=user_id,
        order=order,
        config=config,
        can_launch=can_launch,
    )
    if not can_launch:
        await bot.send_message(user_id, i18n.t("flow.not_enough"))
        await _send_payment_showcase(bot, user_id, config)
    return True


def _is_admin(user_id: int, config: Config) -> bool:
    return user_id in config.admin_ids


async def admin_command(message: Message, state: FSMContext, config: Config) -> None:
    user_id = message.from_user.id if message.from_user else 0
    if not _is_admin(user_id, config):
        return
    await state.finish()
    await _enter_admin_menu(state, message)


async def admin_menu_handler(
    message: Message, state: FSMContext, db: Database, config: Config
) -> None:
    user_id = message.from_user.id if message.from_user else 0
    if not _is_admin(user_id, config):
        return
    choice = (message.text or "").strip()
    broadcast_label = i18n.t("admin.menu.broadcast_all")
    direct_label = i18n.t("admin.menu.direct_message")
    close_label = i18n.t("admin.menu.close")
    if choice == broadcast_label:
        await AdminStates.broadcast_message.set()
        await message.answer(i18n.t("admin.broadcast.prompt"), reply_markup=_admin_cancel_keyboard())
        return
    if choice == direct_label:
        await AdminStates.direct_target.set()
        await message.answer(i18n.t("admin.direct.prompt_id"), reply_markup=_admin_cancel_keyboard())
        return
    if choice == close_label:
        await state.finish()
        await message.answer(i18n.t("admin.menu.closed"), reply_markup=_main_keyboard(config))
        return
    await message.answer(i18n.t("admin.menu.invalid"))


async def admin_broadcast_message_handler(
    message: Message, state: FSMContext, db: Database, config: Config
) -> None:
    user_id = message.from_user.id if message.from_user else 0
    if not _is_admin(user_id, config):
        return
    if _is_cancel_action(message.text):
        await message.answer(i18n.t("admin.status.cancelled"))
        await _return_to_admin_menu(state, message)
        return
    text = (message.text or "").strip()
    if not text:
        await message.answer(i18n.t("admin.broadcast.empty_message"), reply_markup=_admin_cancel_keyboard())
        return
    recipients = await _collect_broadcast_recipients(db)
    if not recipients:
        await message.answer(i18n.t("admin.broadcast.empty_audience"))
        await _return_to_admin_menu(state, message)
        return
    await state.update_data(broadcast_message=text)
    await AdminStates.broadcast_confirm.set()
    preview = i18n.t(
        "admin.broadcast.preview",
        count=len(recipients),
        message=_escape_format_value(text),
    )
    await message.answer(preview, reply_markup=_admin_confirm_keyboard("broadcast"))


async def admin_direct_target_handler(
    message: Message, state: FSMContext, db: Database, config: Config
) -> None:
    user_id = message.from_user.id if message.from_user else 0
    if not _is_admin(user_id, config):
        return
    if _is_cancel_action(message.text):
        await message.answer(i18n.t("admin.status.cancelled"))
        await _return_to_admin_menu(state, message)
        return
    identifier = (message.text or "").strip()
    if not identifier:
        await message.answer(i18n.t("admin.direct.prompt_id"), reply_markup=_admin_cancel_keyboard())
        return
    try:
        users = await db.list_users()
    except Exception:  # pragma: no cover - external dependency guard
        log.warning("Failed to load users for direct message", exc_info=True)
        await message.answer(i18n.t("admin.direct.lookup_error"))
        await _return_to_admin_menu(state, message)
        return
    record = _find_user_record(users, identifier)
    if not record:
        await message.answer(i18n.t("admin.direct.not_found"), reply_markup=_admin_cancel_keyboard())
        return
    try:
        target_id = int(record.get("user_id", 0))
    except (TypeError, ValueError):
        target_id = 0
    if target_id <= 0:
        await message.answer(i18n.t("admin.direct.not_found"), reply_markup=_admin_cancel_keyboard())
        return
    label = _format_user_label(record)
    await state.update_data(
        direct_target_id=target_id,
        direct_target_label=label,
    )
    await AdminStates.direct_message.set()
    await message.answer(
        i18n.t("admin.direct.prompt_message", user=_escape_format_value(label)),
        reply_markup=_admin_cancel_keyboard(),
    )


async def admin_direct_message_handler(
    message: Message, state: FSMContext, db: Database, config: Config
) -> None:
    user_id = message.from_user.id if message.from_user else 0
    if not _is_admin(user_id, config):
        return
    if _is_cancel_action(message.text):
        await message.answer(i18n.t("admin.status.cancelled"))
        await _return_to_admin_menu(state, message)
        return
    text = (message.text or "").strip()
    if not text:
        await message.answer(i18n.t("admin.direct.empty_message"), reply_markup=_admin_cancel_keyboard())
        return
    data = await state.get_data()
    label = data.get("direct_target_label")
    if not label:
        await _return_to_admin_menu(state, message)
        return
    await state.update_data(direct_message=text)
    await AdminStates.direct_confirm.set()
    preview = i18n.t(
        "admin.direct.preview",
        user=_escape_format_value(str(label)),
        message=_escape_format_value(text),
    )
    await message.answer(preview, reply_markup=_admin_confirm_keyboard("direct"))


async def admin_panel_callback_handler(
    callback: CallbackQuery, state: FSMContext, db: Database, config: Config
) -> None:
    await safe_callback_answer(callback)
    user_id = callback.from_user.id if callback.from_user else 0
    if not _is_admin(user_id, config):
        return
    payload = callback.data or ""
    parts = payload.split(":", maxsplit=2)
    if len(parts) != 3:
        return
    _, action, decision = parts
    message = callback.message
    if message is None:
        return
    if decision == "cancel":
        try:
            await message.edit_reply_markup()
        except Exception:  # pragma: no cover - Telegram edit failures are non-fatal
            pass
        await message.answer(i18n.t("admin.status.cancelled"))
        await _return_to_admin_menu(state, message)
        return
    bot = getattr(message, "bot", None) or getattr(callback, "bot", None)
    if bot is None:
        log.warning("Admin callback without bot instance action=%s", action)
        await message.answer(i18n.t("admin.status.internal_error"))
        await _return_to_admin_menu(state, message)
        return
    if action == "broadcast" and decision == "confirm":
        data = await state.get_data()
        broadcast_text = data.get("broadcast_message")
        if not broadcast_text:
            await _return_to_admin_menu(state, message)
            return
        recipients = await _collect_broadcast_recipients(db)
        if not recipients:
            await message.answer(i18n.t("admin.broadcast.empty_audience"))
            await _return_to_admin_menu(state, message)
            return
        try:
            await message.edit_reply_markup()
        except Exception:  # pragma: no cover - Telegram edit failures are non-fatal
            pass
        progress = await message.answer(i18n.t("admin.broadcast.started", count=len(recipients)))
        sent, failed = await _broadcast_to_users(bot, recipients, broadcast_text)
        summary = i18n.t("admin.broadcast.completed", sent=sent, failed=failed)
        try:
            await progress.edit_text(summary)
        except Exception:  # pragma: no cover - message edits may fail
            await message.answer(summary)
        log.info(
            "Admin broadcast completed by %s: audience=%s sent=%s failed=%s",
            user_id,
            len(recipients),
            sent,
            failed,
        )
        await _return_to_admin_menu(state, message)
        return
    if action == "direct" and decision == "confirm":
        data = await state.get_data()
        direct_text = data.get("direct_message")
        target_id = data.get("direct_target_id")
        label = data.get("direct_target_label")
        if not direct_text or not target_id:
            await _return_to_admin_menu(state, message)
            return
        try:
            await message.edit_reply_markup()
        except Exception:  # pragma: no cover - Telegram edit failures are non-fatal
            pass
        success = await _send_text_safely(bot, int(target_id), direct_text)
        result_key = "admin.direct.sent" if success else "admin.direct.failed"
        await message.answer(
            i18n.t(result_key, user=_escape_format_value(str(label or target_id)))
        )
        await _return_to_admin_menu(state, message)

def _format_health_text(result) -> str:
    mode = escape_html(result.details.get("mode", "-")) if isinstance(result.details, dict) else "-"
    lines = [
        f"Версия: {escape_html(result.version)}",
        f"Режим: {mode}",
        "",
    ]
    checks = result.details.get("checks", []) if isinstance(result.details, dict) else []
    for item in checks:
        if not isinstance(item, dict):
            continue
        label = escape_html(str(item.get("label", "")))
        ok = bool(item.get("ok"))
        detail_raw = item.get("detail")
        detail = escape_html(str(detail_raw)) if detail_raw not in (None, "") else ""
        icon = "✅" if ok else "❌"
        if detail:
            lines.append(f"{icon} {label}: {detail}")
        else:
            lines.append(f"{icon} {label}")
    if result.errors:
        lines.append("")
        lines.append("Ошибки:")
        for entry in result.errors:
            lines.append(f"• {escape_html(str(entry))}")
    lines.append("")
    lines.append(f"startup_ok={'true' if result.ok else 'false'}")
    return "\n".join(line for line in lines if line).strip()


async def status_admin_command(message: Message, config: Config) -> None:
    if not _is_admin(message.from_user.id if message.from_user else 0, config):
        return
    result = get_last_healthcheck()
    if not result:
        await message.answer("Данные проверки недоступны.")
        return
    await message.answer(_format_health_text(result))


async def diag_admin_command(message: Message, config: Config) -> None:
    if not _is_admin(message.from_user.id if message.from_user else 0, config):
        return
    snapshot = get_last_error()
    if not snapshot:
        await message.answer("Ошибок не зафиксировано.")
        return
    ts = datetime.fromtimestamp(snapshot.ts).isoformat(timespec="seconds")
    lines = [
        "Последняя ошибка:",
        f"время: {escape_html(ts)}",
        f"corr_id: {escape_html(snapshot.corr_id or '-')}",
        f"job_id: {escape_html(snapshot.job_id or '-')}",
        f"status: {escape_html(str(snapshot.status_code) if snapshot.status_code is not None else '-')}",
        f"type: {escape_html(snapshot.error_type or '-')} / {escape_html(snapshot.error_code or '-')}",
        f"msg: {escape_html(snapshot.error_msg_short or '-')}",
    ]
    if snapshot.provider_message:
        lines.append("")
        lines.append(f"provider: {escape_html(_shorten(snapshot.provider_message, 400))}")
    await message.answer("\n".join(lines))


async def job_admin_command(message: Message, config: Config) -> None:
    if not _is_admin(message.from_user.id if message.from_user else 0, config):
        return
    args = (message.get_args() or "").strip()
    if not args:
        await message.answer("Укажите corr_id: /job <corr_id>")
        return
    history = get_job_history(args)
    if not history:
        await message.answer("События не найдены.")
        return
    lines = [f"История для {escape_html(args)}:"]
    for event in history:
        ts = escape_html(str(event.get("ts", "-")))
        name = escape_html(str(event.get("event", "-")))
        status_code = event.get("status_code")
        status_label = f" status={status_code}" if status_code not in (None, "") else ""
        error_type = event.get("error_type") or ""
        error_label = f" error={escape_html(str(error_type))}" if error_type else ""
        msg = event.get("error_msg_short") or ""
        msg_label = f" msg={escape_html(_shorten(str(msg), 80))}" if msg else ""
        lines.append(f"• {ts} · {name}{status_label}{error_label}{msg_label}")
    await message.answer("\n".join(lines))


_VIDEO_MODEL_PATTERNS: Tuple[str, ...] = ("sora-2*", "gpt-*-sora*")


def _filter_video_model_ids(models: Iterable[str]) -> List[str]:
    filtered: List[str] = []
    for model in models:
        if not isinstance(model, str):
            continue
        candidate = model.strip()
        if not candidate:
            continue
        lowered = candidate.lower()
        if any(fnmatch.fnmatch(lowered, pattern) for pattern in _VIDEO_MODEL_PATTERNS):
            filtered.append(candidate)
    return filtered


async def video_models_command(
    message: Message,
    job_queue: JobQueue,
    config: Config,
    *,
    openai_video_client: Optional[OpenAIVideoClient] = None,
) -> None:
    if not config.sora_video_enabled:
        await message.answer(i18n.t("video.models.disabled"))
        return
    user = message.from_user
    user_id = user.id if user else 0
    client = _resolve_openai_video_client(job_queue, config, openai_video_client)
    if client is None:
        await message.answer(i18n.t("video.common.init_failed"))
        return
    try:
        available_models = await client.list_models()
    except ProviderAPIError as exc:
        log.warning(
            "OpenAI video list_models failed status=%s code=%s message=%s",
            exc.status_code,
            exc.error_code or exc.code,
            exc,
        )
        if _is_admin(user_id, config):
            await message.answer(
                f"API error: {escape_html(exc.error_code or exc.code or 'unknown')} · {escape_html(str(exc))}"
            )
        else:
            await message.answer(i18n.t("video.models.fetch_failed"))
        return
    except Exception as exc:  # pragma: no cover - network/config guard
        log.warning("Failed to fetch OpenAI models error=%s", exc, exc_info=True)
        await message.answer(i18n.t("video.models.fetch_failed"))
        return
    if not available_models:
        await message.answer(i18n.t("errors.video_models_unavailable"))
        return
    models = list(available_models)
    visible_models = _filter_video_model_ids(models)
    if not models:
        visible_models = _filter_video_model_ids(client.supported_models)
    elif not visible_models:
        fallback = _filter_video_model_ids(client.supported_models)
        if fallback:
            visible_models = fallback
    if not visible_models:
        await message.answer(i18n.t("video.models.empty"))
        return
    lines = [i18n.t("video.models.header")]
    for model_name in sorted(dict.fromkeys(visible_models)):
        lines.append(f"• {escape_html(model_name)}")
    await message.answer("\n".join(lines))


async def video_create_command(
    message: Message,
    db: Database,
    job_queue: JobQueue,
    config: Config,
    *,
    openai_video_client: Optional[OpenAIVideoClient] = None,
) -> None:
    if not config.sora_video_enabled:
        await message.answer(i18n.t("video.create.disabled"))
        return
    user = message.from_user
    if user is None:  # pragma: no cover - defensive guard
        return
    await _ensure_user(message, db)
    args = message.get_args() or ""
    prompt, settings, payload, size, model, idempotency_key = _parse_video_command_args(args)
    prompt = prompt.strip()
    if not prompt:
        await message.answer(i18n.t("video.create.prompt_hint"))
        return
    client = _resolve_openai_video_client(job_queue, config, openai_video_client)
    if client is None:
        await message.answer(i18n.t("video.common.init_failed"))
        return
    try:
        available_models = await client.list_models()
    except ProviderAPIError as exc:
        log.warning(
            "OpenAI video list_models failed status=%s code=%s message=%s",
            exc.status_code,
            exc.error_code or exc.code,
            exc,
        )
        if _is_admin(user.id, config):
            await message.answer(
                f"API error: {escape_html(exc.error_code or exc.code or 'unknown')} · {escape_html(str(exc))}"
            )
        else:
            await message.answer(i18n.t("video.models.fetch_failed"))
        return
    except Exception as exc:  # pragma: no cover - network/config guard
        log.warning("Failed to fetch OpenAI models error=%s", exc, exc_info=True)
        await message.answer(i18n.t("video.models.fetch_failed"))
        return
    if not available_models:
        await message.answer(i18n.t("errors.video_models_unavailable"))
        return
    available_lookup = {model.lower(): model for model in available_models}
    model_name_raw = (
        (model or str(settings.get("model") or "").strip())
        or config.resolved_sora_video_model
    )
    model_name_lower = model_name_raw.lower() if model_name_raw else ""
    if available_lookup:
        if model_name_lower not in available_lookup:
            fallback_model = available_models[0]
            if _is_admin(user.id, config):
                await message.answer(
                    i18n.t(
                        "video.create.model_unavailable",
                        requested=model_name_raw or "-",
                        fallback=fallback_model,
                    )
                )
            model_name = fallback_model
        else:
            model_name = available_lookup[model_name_lower]
    else:
        model_name = model_name_raw or config.resolved_sora_video_model
    settings = dict(settings)
    settings.setdefault("model", model_name)
    size_value = size or str(settings.get("size") or "").strip()
    if not size_value:
        size_value = SIZE_OPTIONS.get("horizontal", "")
    if size_value:
        settings.setdefault("size", size_value)
    prepared_request = client.prepare_create_request(
        prompt=prompt,
        settings=settings,
        payload=payload,
    )
    local_extras = dict(prepared_request.extras)
    credits_cost = config.generation_cost_credits
    if credits_cost > 0:
        balance = await db.get_user_credits(user.id)
        if balance < credits_cost:
            await message.answer(i18n.t("flow.not_enough"))
            return
    corr_id = str(uuid.uuid4())
    try:
        data, status_code, duration_ms = await client.create(
            prompt=prompt,
            settings=None,
            payload=None,
            idempotency_key=idempotency_key,
            return_meta=True,
            prepared=prepared_request,
        )
    except Exception as exc:  # pragma: no cover - network/config guard
        log.exception("OpenAI video create failed")
        await message.answer(i18n.t("video.common.send_failed", error=str(exc)))
        return
    job_id, video_id, operation_name = _extract_video_identifiers(data)
    if not job_id:
        snippet = ""
        try:
            snippet = json.dumps(data, ensure_ascii=False)[:400] if data else ""
        except Exception:  # pragma: no cover - defensive
            snippet = str(data)[:400]
        message_text = i18n.t("video.create.missing_job_id")
        if snippet:
            message_text += "\n" + i18n.t("video.common.response_snippet", snippet=snippet)
        await message.answer(message_text)
        return
    last_request = client.last_request
    dropped_fields_snapshot = tuple(last_request.get("dropped_fields", ())) if isinstance(last_request, dict) else ()
    try:
        record = await job_queue.register_external_job(
            job_id=job_id,
            user_id=user.id,
            prompt=prompt,
            size=size_value or "",
            model=model_name,
            corr_id=corr_id,
            provider="openai-video-api",
            username=user.username,
            video_id=video_id,
            idempotency_key=idempotency_key,
            content_type="video",
            task_type=VIDEO_TASK_CREATE,
            sanitized_prompt=prompt,
            original_prompt=prompt,
            auto_sanitized=False,
            operation_name=operation_name,
            status_code=status_code,
            duration_ms=duration_ms,
            extra=local_extras,
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("Failed to register OpenAI video job")
        await message.answer(i18n.t("video.create.registration_failed", error=str(exc)))
        return
    lines = [
        i18n.t("video.create.enqueued"),
        i18n.t("video.common.job_id", job_id=record.id),
        i18n.t(
            "video.common.video_id",
            video_id=video_id or record.video_id or "-",
        ),
        i18n.t("video.common.corr_id", corr_id=corr_id),
        i18n.t("video.common.model", model=model_name),
    ]
    await message.answer("\n".join(lines))
    if dropped_fields_snapshot and _is_admin(user.id, config):
        filtered_fields = sorted({str(field) for field in dropped_fields_snapshot if field})
        if filtered_fields:
            await message.answer("Отброшенные параметры: " + ", ".join(filtered_fields))


async def video_remix_command(
    message: Message,
    db: Database,
    job_queue: JobQueue,
    config: Config,
    *,
    openai_video_client: Optional[OpenAIVideoClient] = None,
) -> None:
    if not config.sora_video_enabled:
        await message.answer(i18n.t("video.create.disabled"))
        return
    user = message.from_user
    if user is None:  # pragma: no cover - defensive guard
        return
    await _ensure_user(message, db)
    args = (message.get_args() or "").strip()
    if not args:
        await message.answer(i18n.t("video.remix.usage"))
        return
    video_id: Optional[str] = None
    payload_text = ""
    if args.startswith("{"):
        try:
            data = json.loads(args)
        except json.JSONDecodeError:
            await message.answer(i18n.t("video.common.invalid_json"))
            return
        if not isinstance(data, dict):
            await message.answer(i18n.t("video.common.expected_object"))
            return
        video_id = str(data.get("video_id") or data.get("id") or "").strip() or None
        payload_text = json.dumps({k: v for k, v in data.items() if k not in {"video_id", "id"}}, ensure_ascii=False)
    else:
        parts = args.split(maxsplit=1)
        video_id = parts[0]
        if len(parts) > 1:
            payload_text = parts[1]
    if not video_id:
        await message.answer(i18n.t("video.remix.missing_video_id"))
        return
    prompt, settings, payload, size, model, idempotency_key = _parse_video_command_args(
        payload_text,
        default_prompt="",
    )
    client = _resolve_openai_video_client(job_queue, config, openai_video_client)
    if client is None:
        await message.answer(i18n.t("video.common.init_failed"))
        return
    model_name = (
        (model or str(settings.get("model") or "").strip())
        or (config.sora_model_video or "").strip()
    )
    if not model_name:
        supported = client.supported_models
        model_name = supported[0] if supported else "sora-2"
    settings = dict(settings)
    settings.setdefault("model", model_name)
    size_value = size or str(settings.get("size") or "").strip()
    if not size_value:
        size_value = SIZE_OPTIONS.get("horizontal", "")
    if size_value:
        settings.setdefault("size", size_value)
    credits_cost = config.generation_cost_credits
    deducted = False
    if credits_cost > 0:
        deducted = await db.deduct_credit(user.id, credits_cost)
        if not deducted:
            await message.answer(i18n.t("flow.not_enough"))
            return
    prompt_text = prompt.strip() or f"Remix {video_id}"
    corr_id = str(uuid.uuid4())
    try:
        data, status_code, duration_ms = await client.remix(
            video_id=video_id,
            prompt=prompt or None,
            settings=settings or None,
            payload=payload or None,
            idempotency_key=idempotency_key,
            return_meta=True,
        )
    except Exception as exc:  # pragma: no cover - network/config guard
        log.exception("OpenAI video remix failed")
        if deducted:
            try:
                await db.add_credits(user.id, credits_cost)
            except Exception:
                log.exception("Failed to refund credits after remix error")
        await message.answer(i18n.t("video.common.send_failed", error=str(exc)))
        return
    job_id, new_video_id, operation_name = _extract_video_identifiers(data)
    if not job_id:
        if deducted:
            try:
                await db.add_credits(user.id, credits_cost)
            except Exception:
                log.exception("Failed to refund credits after remix missing job id")
        await message.answer(i18n.t("video.remix.missing_job_id"))
        return
    effective_video_id = new_video_id or video_id
    try:
        record = await job_queue.register_external_job(
            job_id=job_id,
            user_id=user.id,
            prompt=prompt_text,
            size=size_value or "",
            model=model_name,
            corr_id=corr_id,
            provider="openai-video-api",
            username=user.username,
            video_id=effective_video_id,
            idempotency_key=idempotency_key,
            content_type="video",
            task_type=VIDEO_TASK_REMIX,
            sanitized_prompt=prompt_text,
            original_prompt=prompt or prompt_text,
            auto_sanitized=False,
            operation_name=operation_name,
            status_code=status_code,
            duration_ms=duration_ms,
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("Failed to register OpenAI video remix")
        if deducted:
            try:
                await db.add_credits(user.id, credits_cost)
            except Exception:
                log.exception("Failed to refund credits after remix registration error")
        await message.answer(i18n.t("video.remix.registration_failed", error=str(exc)))
        return
    lines = [
        i18n.t("video.remix.started"),
        i18n.t("video.common.job_id", job_id=record.id),
        i18n.t("video.common.video_id", video_id=effective_video_id or "-"),
        i18n.t("video.common.corr_id", corr_id=corr_id),
        i18n.t("video.common.model", model=model_name),
    ]
    if deducted and credits_cost > 0:
        lines.append(i18n.t("video.common.credits_deducted", credits=credits_cost))
    await message.answer("\n".join(lines))


async def video_list_command(
    message: Message,
    job_queue: JobQueue,
    config: Config,
    *,
    openai_video_client: Optional[OpenAIVideoClient] = None,
) -> None:
    if not config.sora_video_enabled:
        await message.answer(i18n.t("video.list.disabled"))
        return
    client = _resolve_openai_video_client(job_queue, config, openai_video_client)
    if client is None:
        await message.answer(i18n.t("video.common.init_failed"))
        return
    args = (message.get_args() or "").strip()
    params: Dict[str, Any] = {}
    if args:
        if args.startswith("{"):
            try:
                data = json.loads(args)
            except json.JSONDecodeError:
                await message.answer(i18n.t("video.common.invalid_json"))
                return
            if isinstance(data, dict):
                for key in ("limit", "order", "after", "before"):
                    if key in data:
                        params[key] = data[key]
        else:
            for token in args.split():
                if "=" in token:
                    key, value = token.split("=", maxsplit=1)
                    params[key] = value
                elif token.isdigit():
                    params["limit"] = int(token)
    try:
        data, status_code, duration_ms = await client.list(
            limit=params.get("limit"),
            order=params.get("order"),
            after=params.get("after"),
            before=params.get("before"),
            return_meta=True,
        )
    except Exception as exc:  # pragma: no cover - network/config guard
        log.exception("OpenAI video list failed")
        await message.answer(i18n.t("video.list.fetch_failed", error=str(exc)))
        return
    entries: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        raw_entries = data.get("data")
        if isinstance(raw_entries, list):
            entries = [entry for entry in raw_entries if isinstance(entry, dict)]
    lines = [
        i18n.t("video.list.status", status=status_code, duration=duration_ms),
        i18n.t("video.list.header" if entries else "video.list.empty"),
    ]
    for entry in entries[:10]:
        vid = str(entry.get("id") or entry.get("video_id") or "-")
        status = str(entry.get("status") or entry.get("state") or "-")
        created = entry.get("created_at") or entry.get("createdAt") or entry.get("created")
        created_label = (
            SafeText(i18n.t("video.list.entry_created", created=created)) if created else SafeText("")
        )
        lines.append(
            i18n.t(
                "video.list.entry",
                video_id=vid,
                status=status,
                created_label=created_label,
            )
        )
    if len(entries) > 10:
        lines.append(i18n.t("video.list.more", count=len(entries) - 10))
    await message.answer("\n".join(lines))


async def video_info_command(
    message: Message,
    job_queue: JobQueue,
    config: Config,
    *,
    openai_video_client: Optional[OpenAIVideoClient] = None,
) -> None:
    if not config.sora_video_enabled:
        await message.answer(i18n.t("video.list.disabled"))
        return
    video_id = (message.get_args() or "").strip()
    if not video_id:
        await message.answer(i18n.t("video.info.usage"))
        return
    client = _resolve_openai_video_client(job_queue, config, openai_video_client)
    if client is None:
        await message.answer(i18n.t("video.common.init_failed"))
        return
    try:
        data, status_code, duration_ms = await client.retrieve(video_id, return_meta=True)
    except Exception as exc:  # pragma: no cover - network/config guard
        log.exception("OpenAI video retrieve failed video_id=%s", video_id)
        await message.answer(i18n.t("video.info.fetch_failed", error=str(exc)))
        return
    video_obj = client.get_video_object(data) if isinstance(client, OpenAIVideoClient) else None
    info_lines = [
        i18n.t("video.info.header", video_id=video_id),
        i18n.t("video.info.status", status=status_code, duration=duration_ms),
    ]
    if isinstance(video_obj, dict):
        status = str(video_obj.get("status") or video_obj.get("state") or "unknown")
        info_lines.append(i18n.t("video.info.state", status=status))
        created = video_obj.get("created_at") or video_obj.get("createdAt")
        if created:
            info_lines.append(i18n.t("video.info.created", created=created))
    try:
        pretty = json.dumps(video_obj or data, ensure_ascii=False, indent=2)
    except Exception:  # pragma: no cover - defensive
        pretty = str(video_obj or data)
    if pretty:
        snippet = pretty[:1500]
        info_lines.append("")
        info_lines.append(f"<pre>{escape_html(snippet)}</pre>")
    await message.answer("\n".join(info_lines))


async def video_get_command(
    message: Message,
    db: Database,
    job_queue: JobQueue,
    config: Config,
) -> None:
    user = message.from_user
    if not user or not _is_admin(user.id, config):
        await message.answer(i18n.t("video.get.admin_only"))
        return
    args = (message.get_args() or "").strip()
    parts = [part for part in args.split() if part]
    if not parts:
        await message.answer(i18n.t("video.get.usage_admin"))
        return
    video_id = parts[0]
    if not video_id:
        await message.answer(i18n.t("video.get.usage_admin"))
        return
    fmt = parts[1] if len(parts) > 1 else "mp4"
    provider: Optional[BaseProviderClient] = None
    for key in ("openai-video-api", "openai-video", config.sora_model_video, "sora"):
        if not key:
            continue
        candidate = job_queue.get_provider(key)
        if candidate and callable(getattr(candidate, "download_content", None)):
            provider = candidate
            break
    if provider is None:
        await message.answer(i18n.t("video.get.provider_missing"))
        return
    download_handler = getattr(provider, "download_content", None)
    if not callable(download_handler):
        await message.answer(i18n.t("video.get.provider_missing"))
        return
    try:
        file_path: Path = await download_handler(video_id, format=fmt)
    except ProviderAPIError as exc:
        await message.answer(
            i18n.t(
                "video.get.error",
                status=exc.status_code,
                code=exc.error_code or exc.code or "unknown",
                message=str(exc),
            )
        )
        return
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("Manual download failed for video %s", video_id)
        await message.answer(
            i18n.t("video.get.error", status=0, code="exception", message=str(exc))
        )
        return
    try:
        payload = file_path.read_bytes()
    except Exception as exc:  # pragma: no cover - filesystem guard
        log.exception("Failed to read downloaded file %s", file_path)
        await message.answer(i18n.t("video.get.read_error", error=str(exc)))
        return
    mime, _ = mimetypes.guess_type(file_path.name)
    input_file = BufferedInputFile(payload, filename=file_path.name, mime_type=mime)
    caption = i18n.t(
        "video.get.sent",
        video_id=video_id,
        size=len(payload),
        provider=getattr(provider, "provider_name", "sora"),
    )
    await message.answer_document(input_file, caption=caption)
    try:
        file_path.unlink()
    except FileNotFoundError:  # pragma: no cover - filesystem cleanup
        pass
    except Exception:  # pragma: no cover - filesystem cleanup
        log.debug("Temporary video file %s already removed", file_path)
    try:
        file_path.parent.rmdir()
    except Exception:  # pragma: no cover - filesystem cleanup
        pass


async def diag_openai_video_command(
    message: Message,
    job_queue: JobQueue,
    config: Config,
    *,
    openai_video_client: Optional[OpenAIVideoClient] = None,
) -> None:
    if not config.sora_video_enabled:
        await message.answer("OpenAI Sora 2 не настроена.")
        return
    client = _resolve_openai_video_client(job_queue, config, openai_video_client)
    if client is None:
        await message.answer("Не удалось инициализировать OpenAI Videos API.")
        return
    supported = ", ".join(client.supported_models) or "—"
    diagnostics = client.get_diagnostics()
    headers = diagnostics.get("headers") or {}
    headers_text = ", ".join(
        f"{escape_html(str(key))}={escape_html(str(value))}" for key, value in sorted(headers.items())
    ) or "—"
    content_endpoint = diagnostics.get("content_endpoint") or "—"
    available_models = diagnostics.get("available_models") or []
    available_models_text = ", ".join(map(str, available_models)) or "—"
    base_url_value = diagnostics.get("base_url") or config.openai_api_base or "—"
    last_status = diagnostics.get("last_content_status")
    last_request_id = diagnostics.get("last_content_request_id") or "—"
    last_video_id = diagnostics.get("last_content_video_id") or "—"
    last_timestamp = diagnostics.get("last_content_timestamp")
    last_download_url = diagnostics.get("last_content_url") or "—"
    last_download_note = diagnostics.get("last_content_note") or "—"
    last_download_duration = diagnostics.get("last_content_duration_ms")
    last_download_asset = diagnostics.get("last_content_asset_id") or "—"
    last_download_variant = diagnostics.get("last_content_variant") or "—"
    last_request_url = diagnostics.get("last_request_url") or "—"
    if last_status is None:
        last_status_text = "—"
    else:
        last_status_text = str(last_status)
    last_timestamp_text = str(last_timestamp) if last_timestamp is not None else "—"
    lines = [
        "Диагностика OpenAI Videos API:",
        f"configured_model: {escape_html(config.sora_model_video or '-')}",
        f"supported_models: {escape_html(supported)}",
        f"api_version: {escape_html(diagnostics.get('api_version') or '—')}",
        f"beta_header: {escape_html(diagnostics.get('beta_header') or '—')}",
        f"organization_id: {escape_html(diagnostics.get('organization_id') or '—')}",
        f"base_url: {escape_html(str(base_url_value))}",
        f"headers: {headers_text}",
        f"content_endpoint: {escape_html(content_endpoint)}",
        f"last_request_url: {escape_html(str(last_request_url))}",
        (
            "download_status: "
            f"status={escape_html(last_status_text)} "
            f"request_id={escape_html(str(last_request_id))} "
            f"video_id={escape_html(str(last_video_id))} "
            f"url={escape_html(str(last_download_url))} "
            f"note={escape_html(str(last_download_note))} "
            f"asset={escape_html(str(last_download_asset))} "
            f"variant={escape_html(str(last_download_variant))} "
            f"ts={escape_html(last_timestamp_text)} "
            f"duration_ms={escape_html(str(last_download_duration) if last_download_duration is not None else '—')}"
        ),
        f"available_models: {escape_html(available_models_text)}",
    ]
    queue_obj = getattr(job_queue, "_queue", None)
    if queue_obj is not None:
        try:
            size = queue_obj.qsize()
        except Exception:  # pragma: no cover - defensive
            size = "?"
        lines.append(f"queue_size: {size}")
    try:
        data, status_code, duration_ms = await client.list(limit=1, return_meta=True)
        has_more_value: Optional[Any] = None
        if isinstance(data, Mapping):
            has_more_value = data.get("has_more") if "has_more" in data else None
        has_more_text = "—"
        if has_more_value is not None:
            if isinstance(has_more_value, bool):
                has_more_text = "true" if has_more_value else "false"
            else:
                has_more_text = str(has_more_value)
        lines.append(
            f"list: status={status_code} has_more={escape_html(has_more_text)} time={duration_ms} мс"
        )
    except ProviderAPIError as exc:
        lines.append(
            f"list: error status={exc.status_code} code={escape_html(exc.error_code or '-')}"
        )
    except Exception as exc:  # pragma: no cover - network guard
        lines.append(f"list: error={escape_html(str(exc))}")
    models: List[str] = []
    try:
        models = await client.list_models()
    except ProviderAPIError as exc:
        lines.append(
            f"models.list: error status={exc.status_code} code={escape_html(exc.error_code or '-')}"
        )
    except Exception as exc:  # pragma: no cover - network guard
        lines.append(f"models.list: error={escape_html(str(exc))}")
    else:
        models_text = ", ".join(models) if models else "—"
        lines.append(f"models.list (/models/list): {escape_html(models_text)}")
        filtered_models = _filter_video_model_ids(models)
        if filtered_models:
            preview = ", ".join(filtered_models[:5])
            lines.append(f"list_models: {escape_html(preview)}")
            if len(filtered_models) > 5:
                lines.append(f"… ещё {len(filtered_models) - 5}")
        else:
            lines.append("list_models: пусто")
    sample_model = (config.sora_model_video or "").strip() or (client.supported_models[0] if client.supported_models else "")
    try:
        prepared = client.prepare_create_request(
            prompt="Диагностика", payload={"model": sample_model} if sample_model else None
        )
        request_preview = json.dumps(prepared.request, ensure_ascii=False)
        extras_preview = json.dumps(prepared.extras, ensure_ascii=False)
        if len(request_preview) > 160:
            request_preview = f"{request_preview[:157]}…"
        if len(extras_preview) > 160:
            extras_preview = f"{extras_preview[:157]}…"
        dropped_text = ", ".join(prepared.dropped_fields) if prepared.dropped_fields else "—"
        lines.append(
            f"create.prepare: model={escape_html(str(prepared.request.get('model', '-')))} "
            f"dropped={escape_html(dropped_text)} corr={escape_html(prepared.correlation_id or '—')}"
        )
        lines.append(f"create.request: {escape_html(request_preview)}")
        if prepared.extras:
            lines.append(f"create.extras: {escape_html(extras_preview)}")
    except Exception as exc:  # pragma: no cover - validation guard
        lines.append(f"create.prepare: error={escape_html(str(exc))}")
    probe_result = "—"
    try:
        await client.content("diag-probe")
    except ProviderAPIError as exc:
        code_value = exc.error_code or exc.code or "-"
        probe_result = f"status={exc.status_code} code={code_value}"
    except Exception as exc:  # pragma: no cover - network guard
        probe_result = f"error={str(exc)}"
    else:
        probe_result = "ok"
    lines.append(f"content.test: {escape_html(probe_result)}")
    history = getattr(client, "diagnostic_history", [])
    if history:
        lines.append("")
        lines.append("Последние запросы OpenAI Videos:")
        for entry in list(history)[-3:][::-1]:
            method = str(entry.get("method") or "?")
            path = str(entry.get("path") or "?")
            url = str(entry.get("url") or "")
            dropped = entry.get("dropped_fields") or ()
            dropped_text = ", ".join(dropped) if dropped else "—"
            correlation = str(entry.get("correlation_id") or "")
            status_code = entry.get("status_code")
            duration = entry.get("duration_ms")
            meta_parts: List[str] = []
            if correlation:
                meta_parts.append(f"corr={correlation}")
            meta_parts.append(f"dropped={dropped_text}")
            if status_code is not None:
                meta_parts.append(f"status={status_code}")
            if duration is not None:
                meta_parts.append(f"{duration} мс")
            error_info = entry.get("error")
            if isinstance(error_info, dict):
                err_parts: List[str] = []
                err_code = error_info.get("error_code")
                err_type = error_info.get("error_type")
                err_msg = error_info.get("message")
                if err_code:
                    err_parts.append(str(err_code))
                if err_type and err_type != err_code:
                    err_parts.append(str(err_type))
                if err_msg:
                    text = str(err_msg)
                    if len(text) > 160:
                        text = f"{text[:157]}…"
                    err_parts.append(text)
                if err_parts:
                    meta_parts.append("error: " + " — ".join(err_parts))
            base_text = escape_html(f"{method} {path}")
            if url and url != path:
                meta_parts.append(f"url={url}")
            details = "; ".join(escape_html(part) for part in meta_parts if part)
            if details:
                lines.append(f"• {base_text} ({details})")
            else:
                lines.append(f"• {base_text}")
    error_snapshots = diagnostics.get("error_snapshots") or []
    if error_snapshots:
        lines.append("")
        lines.append("Ошибки OpenAI (последние):")
        for snapshot in list(error_snapshots)[-3:][::-1]:
            if isinstance(snapshot, Mapping):
                status_code = snapshot.get("status_code")
                error_code = snapshot.get("error_code")
                message = snapshot.get("message")
                provider_message = snapshot.get("provider_message")
                parts: List[str] = []
                if status_code is not None:
                    parts.append(f"status={status_code}")
                if error_code:
                    parts.append(f"code={error_code}")
                if message:
                    parts.append(_shorten(str(message), 180))
                if provider_message:
                    parts.append(_shorten(str(provider_message), 180))
                lines.append("• " + escape_html("; ".join(parts) or json.dumps(snapshot, ensure_ascii=False)))
            else:
                lines.append("• " + escape_html(str(snapshot)))
    await message.answer("\n".join(lines))


async def models_command(
    message: Message,
    job_queue: JobQueue,
    config: Config,
    *,
    openai_video_client: Optional[OpenAIVideoClient] = None,
) -> None:
    sections: List[str] = []
    failures: List[str] = []
    found_any = False

    if config.gemini_enabled:
        gemini_sections: List[str] = ["Доступные модели Gemini:"]
        scopes = (("Text (v1)", get_text_client), ("Media (v1beta)", get_media_client))
        gemini_found = False
        for label, getter in scopes:
            try:
                client = getter(config)
            except Exception as exc:  # pragma: no cover - network/config guard
                log.warning(
                    "Failed to build Gemini client scope=%s error=%s", label, exc, exc_info=True
                )
                failures.append(label)
                continue
            try:
                catalog = list(client.models.list())
            except Exception as exc:  # pragma: no cover - network guard
                log.warning(
                    "Failed to list Gemini models scope=%s error=%s", label, exc, exc_info=True
                )
                failures.append(label)
                continue
            entries: List[str] = []
            seen: Set[str] = set()
            for item in catalog:
                name = getattr(item, "name", None) or getattr(item, "model", None)
                if not isinstance(name, str) or not name:
                    continue
                short_name = name.rsplit("/", 1)[-1]
                if short_name in seen:
                    continue
                seen.add(short_name)
                raw_methods = getattr(item, "supported_generation_methods", None) or []
                method_labels = sorted({str(method) for method in raw_methods if method})
                methods_text = ", ".join(method_labels) if method_labels else "—"
                entries.append(f"• {escape_html(short_name)} — {escape_html(methods_text)}")
            if not entries:
                continue
            gemini_found = True
            gemini_sections.append(label + ":")
            gemini_sections.extend(entries)
        if gemini_found:
            sections.extend(gemini_sections)
            found_any = True

    if config.sora_video_enabled:
        client = _resolve_openai_video_client(job_queue, config, openai_video_client)
        if client is None:
            failures.append("Sora 2")
        else:
            try:
                models = await client.list_models()
            except Exception as exc:  # pragma: no cover - network/config guard
                log.warning("Failed to list OpenAI video models error=%s", exc, exc_info=True)
                failures.append("Sora 2")
            else:
                visible_models = models or list(client.supported_models)
                if visible_models:
                    if found_any:
                        sections.append("")
                    sections.append("Модели OpenAI Sora 2:")
                    for model_name in sorted(dict.fromkeys(visible_models)):
                        sections.append(f"• {escape_html(model_name)}")
                    found_any = True
                else:
                    failures.append("Sora 2")

    if not found_any:
        if failures:
            await message.answer("Список моделей недоступен: " + ", ".join(sorted(set(failures))))
        else:
            await message.answer("Список моделей пока недоступен. Попробуйте позже.")
        return
    if failures:
        sections.append("")
        sections.append("Не удалось загрузить: " + ", ".join(sorted(set(failures))))
    await message.answer("\n".join(sections))


async def successful_text_handler(
    message: Message,
    state: FSMContext,
    db: Database,
    config: Config,
) -> None:
    if message.text in MAIN_MENU_BUTTONS or message.is_command():
        return
    await handle_text_input(message, state, db, config)


async def photo_message_handler(
    message: Message,
    state: FSMContext,
    db: Database,
    config: Config,
) -> None:
    await handle_photo_input(message, state, db, config)


def register_handlers(
    dp: Dispatcher,
    *,
    db: Database,
    config: Config,
    job_queue: JobQueue,
    gate: GenerationRequestGate,
    archive_publisher: ArchivePublisher,
    error_reporter: ErrorReporter,
    chatgpt_client: Optional[OpenAIChatClient] = None,
    openai_video_client: Optional[OpenAIVideoClient] = None,
) -> None:
    job_queue.register_notification_callback(
        name="telegram",
        callback=lambda job: _send_job_update(
            dp,
            job,
            archive_publisher,
            config,
            db,
            error_reporter,
        ),
    )

    dp.register_message_handler(
        lambda message, state: start_command(message, db, state, config),
        CommandStart(),
        state="*",
    )
    dp.register_message_handler(
        lambda message, state: help_command(message, db, state, config),
        Command("help"),
        state="*",
    )
    dp.register_message_handler(
        lambda message, state: admin_command(message, state, config),
        Command("admin"),
        state="*",
    )
    dp.register_message_handler(
        lambda message: status_admin_command(message, config),
        Command("status"),
        state="*",
    )
    dp.register_message_handler(
        lambda message: diag_admin_command(message, config),
        Command("diag"),
        state="*",
    )
    dp.register_message_handler(
        lambda message: job_admin_command(message, config),
        Command("job"),
        state="*",
    )
    dp.register_message_handler(
        lambda message: models_command(
            message,
            job_queue,
            config,
            openai_video_client=openai_video_client,
        ),
        Command("models"),
        state="*",
    )
    dp.register_message_handler(
        lambda message: video_models_command(
            message,
            job_queue,
            config,
            openai_video_client=openai_video_client,
        ),
        Command("video_models"),
        state="*",
    )
    dp.register_message_handler(
        lambda message: video_list_command(
            message,
            job_queue,
            config,
            openai_video_client=openai_video_client,
        ),
        Command("video_list"),
        state="*",
    )
    dp.register_message_handler(
        lambda message: video_info_command(
            message,
            job_queue,
            config,
            openai_video_client=openai_video_client,
        ),
        Command("video_info"),
        state="*",
    )
    dp.register_message_handler(
        lambda message: video_create_command(
            message,
            db,
            job_queue,
            config,
            openai_video_client=openai_video_client,
        ),
        Command("video_create"),
        state="*",
    )
    dp.register_message_handler(
        lambda message: video_remix_command(
            message,
            db,
            job_queue,
            config,
            openai_video_client=openai_video_client,
        ),
        Command("video_remix"),
        state="*",
    )
    dp.register_message_handler(
        lambda message: video_get_command(message, db, job_queue, config),
        Command("video_get"),
        state="*",
    )
    dp.register_message_handler(
        lambda message: diag_openai_video_command(
            message,
            job_queue,
            config,
            openai_video_client=openai_video_client,
        ),
        Command("diag_openai_video"),
        state="*",
    )
    dp.register_message_handler(
        lambda message, state: balance_command(message, db, state),
        Command("balance"),
        state="*",
    )
    dp.register_message_handler(
        lambda message, state: _notify_pro_unavailable(message),
        Command("pro"),
        state="*",
    )
    dp.register_message_handler(
        lambda message, state: top_up_menu(message, db, state, config),
        Command("buy_card"),
        state="*",
    )
    dp.register_message_handler(
        lambda message, state: generate_video_menu(message, db, state, config),
        lambda message: message.text == i18n.t("buttons.generate_text"),
        state="*",
    )
    dp.register_message_handler(
        lambda message, state: generate_image_menu(message, db, state, config),
        lambda message: message.text == i18n.t("buttons.generate_photo"),
        state="*",
    )
    dp.register_message_handler(
        lambda message, state: balance_command(message, db, state),
        lambda message: message.text == i18n.t("buttons.balance"),
        state="*",
    )
    dp.register_message_handler(
        lambda message, state: top_up_menu(message, db, state, config),
        lambda message: message.text == i18n.t("buttons.top_up"),
        state="*",
    )
    dp.register_message_handler(
        lambda message, state: help_button(message, db, state, config),
        lambda message: message.text == i18n.t("buttons.help"),
        state="*",
    )
    dp.register_message_handler(
        lambda message, state: chatgpt_menu(
            message, state, config, chatgpt_client
        ),
        lambda message: message.text == i18n.t("buttons.chatgpt"),
        state="*",
    )
    dp.register_message_handler(
        lambda message, state: admin_menu_handler(message, state, db, config),
        state=AdminStates.menu,
        content_types=["text"],
    )
    dp.register_message_handler(
        lambda message, state: admin_broadcast_message_handler(message, state, db, config),
        state=AdminStates.broadcast_message,
        content_types=["text"],
    )
    dp.register_message_handler(
        lambda message, state: admin_direct_target_handler(message, state, db, config),
        state=AdminStates.direct_target,
        content_types=["text"],
    )
    dp.register_message_handler(
        lambda message, state: admin_direct_message_handler(message, state, db, config),
        state=AdminStates.direct_message,
        content_types=["text"],
    )
    dp.register_message_handler(
        lambda message, state: handle_text_input(message, state, db, config),
        state=GenerationStates.text_prompt,
    )
    dp.register_message_handler(
        lambda message, state: handle_photo_input(message, state, db, config),
        state=GenerationStates.photo_prompt,
        content_types=["photo"],
    )
    dp.register_message_handler(
        lambda message, state: photo_message_handler(message, state, db, config),
        content_types=["photo"],
        state=None,
    )
    dp.register_message_handler(
        lambda message, state: successful_text_handler(message, state, db, config),
        content_types=["text"],
        state=None,
    )
    dp.register_message_handler(
        lambda message, state: chatgpt_respond(
            message, state, config, chatgpt_client
        ),
        state=ChatGPTState.awaiting_input,
        content_types=["text"],
    )

    dp.register_callback_query_handler(
        lambda call, state: video_model_callback_handler(call, state, config),
        lambda call: call.data and call.data.startswith("video:model:"),
        state="*",
    )
    dp.register_callback_query_handler(
        lambda call, state: video_mode_callback_handler(call, state, config),
        lambda call: call.data and call.data.startswith("video:mode:"),
        state="*",
    )
    dp.register_callback_query_handler(
        lambda call, state: image_mode_callback_handler(call, state, config),
        lambda call: call.data and call.data.startswith("image:mode:"),
        state="*",
    )
    dp.register_callback_query_handler(
        lambda call: noop_callback_handler(call),
        lambda call: call.data == "noop",
        state="*",
    )
    dp.register_callback_query_handler(
        lambda call: start_generation_callback_handler(call),
        lambda call: call.data == "start_generation",
        state="*",
    )
    dp.register_callback_query_handler(
        lambda call, state: option_callback_handler(call, state, config),
        lambda call: call.data and call.data.startswith("opt:"),
        state="*",
    )
    dp.register_callback_query_handler(
        lambda call, state: flow_back_callback_handler(call, state, config),
        lambda call: call.data and call.data.startswith("flow:back:"),
        state="*",
    )
    dp.register_callback_query_handler(
        lambda call, state: order_callback_handler(
            call, state, db, job_queue, config, gate, error_reporter
        ),
        lambda call: call.data and call.data.startswith("order:"),
        state="*",
    )
    dp.register_callback_query_handler(
        lambda call, state: admin_panel_callback_handler(call, state, db, config),
        lambda call: call.data and call.data.startswith("admin:"),
        state="*",
    )
    dp.register_callback_query_handler(
        lambda call: payment_callback_handler(call, db, config),
        lambda call: call.data and call.data.startswith("pay:"),
        state="*",
    )
    dp.register_callback_query_handler(
        lambda call, state: menu_callback_handler(call, db, config, state),
        lambda call: call.data and call.data.startswith("menu:"),
        state="*",
    )


__all__ = [
    "register_handlers",
    "start_command",
    "help_command",
    "chatgpt_menu",
    "chatgpt_respond",
    "balance_command",
    "top_up_menu",
    "resend_pending_order",
]

