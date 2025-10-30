"""Aiogram handlers for the video generation bot."""
from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import mimetypes
import time
from dataclasses import dataclass, field
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

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
)

from archive import ArchivePayload, ArchivePublisher
from config import Config
from db import Database, ErrorLogRecord, GenerationJobRecord
from generation_gate import GateDecision, GenerationRequestGate, normalize_prompt
from i18n import SafeText, escape_html, format_credits, format_prompt, i18n
from jobs import JobQueue
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
from providers import BaseProviderClient, ProviderAPIError
from services.gemini_catalog import list_veo_video_models
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
ASPECT_RATIO_OPTIONS = {
    "horizontal": "16:9",
    "vertical": "9:16",
}
DEFAULT_ASPECT_RATIO = ASPECT_RATIO_OPTIONS["horizontal"]


log = logging.getLogger(__name__)


SIZE_OPTIONS: Dict[str, str] = {
    "vertical": "720x1280",
    "horizontal": "1280x720",
}

_INLINE_ASSET_MAX_BYTES = 9 * 1024 * 1024
_TELEGRAM_VIDEO_MAX_BYTES = 48 * 1024 * 1024
_TELEGRAM_DOCUMENT_MAX_BYTES = 2 * 1024 * 1024 * 1024


SIZE_LABEL_KEYS: Dict[str, str] = {
    "vertical": "chips.vertical",
    "horizontal": "chips.horizontal",
}

MAX_INLINE_VIDEO_BYTES = 9 * 1024 * 1024

MAIN_MENU_BUTTONS = {
    i18n.t("buttons.generate_text"),
    i18n.t("buttons.generate_photo"),
    i18n.t("buttons.balance"),
    i18n.t("buttons.top_up"),
    i18n.t("buttons.help"),
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


class GenerationStates(StatesGroup):
    """Conversation states for collecting generation inputs."""

    video_model = State()
    video_mode = State()
    image_mode = State()
    text_prompt = State()
    photo_prompt = State()


@dataclass
class OrderContext:
    """Details of a generation order pending confirmation."""

    category: Literal["video", "image"]
    flow: Literal["text", "photo"]
    prompt: str
    size: str
    model: str
    provider: Optional[str] = None
    product: str = "sora_video"
    credits_cost: int = 0
    model_label: str = ""
    image_file_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    corr_id: Optional[str] = None
    aspect_ratio: Optional[str] = None

    def key(self) -> str:
        parts = [
            self.category,
            self.flow,
            self.prompt,
            self.image_file_id or "",
            self.size,
            self.model,
            self.aspect_ratio or "",
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

    def _register(model_name: str, provider: str, label: str) -> None:
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
                label=label,
            )
        )

    if config.gemini_video_enabled:
        fallback_label = i18n.t("video.models.veo")
        default_name = _normalise_video_model_name(config.gemini_model_video)
        if default_name:
            label = default_name if _is_veo_video_model_name(default_name) else fallback_label
            _register(default_name, "veo", label)
        for candidate in list_veo_video_models(config):
            normalised = _normalise_video_model_name(candidate)
            if _is_veo_video_model_name(normalised):
                _register(normalised, "veo", normalised)

    if config.sora_video_enabled:
        fallback_label = i18n.t("video.models.sora")
        default_sora = _normalise_video_model_name(config.sora_model_video)
        if default_sora:
            label = fallback_label or default_sora
            _register(default_sora, "sora", label)

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


def _video_models_description(config: Config) -> str:
    labels = [option.label for option in _video_model_options(config)]
    if not labels:
        return i18n.t("video.models.unavailable")
    return ", ".join(dict.fromkeys(labels))


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

    policy_trigger = (
        "policy" in error_type
        or "policy" in error_code
        or "safety" in error_type
        or "safety" in error_code
    )

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
        hint = "provider_unavailable"
        message = "Провайдер недоступен. Попробуем ещё раз позже."
        short = _shorten(provider_message or "Провайдер недоступен")
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

    last_size: str = SIZE_OPTIONS["horizontal"]
    last_aspect_ratio: str = DEFAULT_ASPECT_RATIO
    pending_order: Optional[OrderContext] = None
    awaiting_payment: bool = False
    last_launch_key: Optional[str] = None
    last_launch_ts: float = 0.0


class SessionManager:
    """In-memory storage for user sessions."""

    def __init__(self) -> None:
        self._sessions: Dict[int, UserSession] = {}

    def get(self, user_id: int) -> UserSession:
        return self._sessions.setdefault(user_id, UserSession())


SESSION_MANAGER = SessionManager()
STATUS_MESSAGES = StatusMessageManager()


def _main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=i18n.t("buttons.generate_text"))],
            [KeyboardButton(text=i18n.t("buttons.generate_photo"))],
            [KeyboardButton(text=i18n.t("buttons.balance"))],
            [KeyboardButton(text=i18n.t("buttons.top_up"))],
            [KeyboardButton(text=i18n.t("buttons.help"))],
        ],
        resize_keyboard=True,
    )


def _mark_selected(label: str, selected: bool) -> str:
    return f"✅ {label}" if selected else label


def _back_button(target: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(
        text=i18n.t("buttons.back"), callback_data=f"flow:back:{target}"
    )


def _build_back_keyboard(target: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[_back_button(target)]])


def _build_options_keyboard(
    session: UserSession, context: str, back_target: Optional[str] = None
) -> InlineKeyboardMarkup:
    size_row = []
    for token, value in SIZE_OPTIONS.items():
        label_key = SIZE_LABEL_KEYS[token]
        label = i18n.t(label_key)
        size_row.append(
            InlineKeyboardButton(
                text=_mark_selected(label, session.last_size == value),
                callback_data=f"opt:size:{context}:{token}",
            )
        )
    rows: list[list[InlineKeyboardButton]] = [size_row]
    if back_target:
        rows.append([_back_button(back_target)])
    return InlineKeyboardMarkup(inline_keyboard=rows)


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
        reply_markup=_main_keyboard(),
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
    credits_cost = config.get_product_credits(product)
    if credits_cost <= 0:
        credits_cost = config.generation_cost_credits
    return OrderContext(
        category=category,
        flow=flow,
        prompt=prompt,
        size=size,
        model=model,
        provider=provider,
        product=product,
        credits_cost=credits_cost,
        model_label=model_label or _resolve_model_label(model, config),
        image_file_id=image_file_id,
        aspect_ratio=aspect_ratio,
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


async def _send_generation_prompt(
    message: Message,
    *,
    session: UserSession,
    category: Literal["video", "image"],
    mode: Literal["text", "photo"],
    model_label: str,
    include_size: bool,
) -> None:
    if category == "video":
        size = session.last_size if include_size else ""
        if mode == "photo":
            body = i18n.t("video.prompt.photo", size=size, model=model_label)
        else:
            body = i18n.t("video.prompt.text", size=size, model=model_label)
        if include_size:
            markup = _build_options_keyboard(session, mode, back_target="video_mode")
        else:
            markup = _build_back_keyboard("video_mode")
    else:
        if mode == "photo":
            body = i18n.t("image.prompt.photo", model=model_label)
        else:
            body = i18n.t("image.prompt.text", model=model_label)
        markup = _build_back_keyboard("image_mode")
    await message.answer(body, reply_markup=markup)


async def _send_order_confirmation(
    *,
    bot: Bot,
    chat_id: int,
    order: OrderContext,
    config: Config,
    can_launch: bool,
    include_back: bool = True,
) -> Message:
    price_text = config.format_price_tag(order.credits_cost or config.generation_cost_credits)
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
        if order.flow == "photo":
            body = i18n.t(
                "video.confirm.photo",
                prompt=prompt_text,
                size=size_value,
                model=order.model_label,
                price=price_text,
            )
        else:
            body = i18n.t(
                "video.confirm.text",
                prompt=prompt_text,
                size=size_value,
                model=order.model_label,
                price=price_text,
            )
    keyboard = _build_confirmation_keyboard(can_launch=can_launch, include_back=include_back)
    return await bot.send_message(chat_id, body, reply_markup=keyboard)


async def flow_back_callback_handler(
    callback: CallbackQuery, state: FSMContext, config: Config
) -> None:
    await callback.answer()
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
        product = data.get("product", "sora_video")
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
        product = data.get("product", "sora_video")
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
        product = data.get("product", "sora_video")
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
    lines: list[str] = [i18n.t("payment.packages.title")]
    for package in config.credit_packages:
        credits_label = format_credits(package.credits_int)
        price_label = config.format_rubles(package.price_rub)
        lines.append(i18n.t("payment.packages.line", credits=credits_label, price=price_label))
    lines.append("")
    lines.append(i18n.t("payment.store.instructions"))
    terms_url = (config.terms_url or "").strip()
    if terms_url:
        escaped_url = escape_html(terms_url)
        link = SafeText(f'<a href="{escaped_url}">{escaped_url}</a>')
        lines.append("")
        lines.append(i18n.t("payment.terms_notice", terms_url=link))
    return "\n".join(line for line in lines if line).strip()


def _build_packages_keyboard(config: Config) -> Optional[InlineKeyboardMarkup]:
    if not config.yookassa_ready:
        return None
    rows: list[list[InlineKeyboardButton]] = []
    for package in config.credit_packages:
        text = format_credits(package.credits_int)
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
    corr_id = order.corr_id or str(uuid.uuid4())
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
        provider_settings["duration"] = 6
        provider_settings.setdefault("duration_seconds", 6)
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
            remote_asset = _select_remote_video_asset(job)
            if remote_asset and remote_asset.get("url"):
                sent_info = await _deliver_remote_video(
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


def _select_remote_video_asset(job: GenerationJobRecord) -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for asset in _collect_assets_meta(job):
        if asset.get("inline"):
            continue
        url = asset.get("url") or asset.get("file_id")
        if not isinstance(url, str) or not url:
            continue
        mime = str(asset.get("mime") or "")
        asset_type = str(asset.get("type") or "")
        asset_kind = str(asset.get("kind") or "")
        if (
            mime.startswith("video/")
            or asset_type.lower() == "video"
            or asset_kind.lower() == "video"
        ):
            score = 1
            if asset.get("preferred"):
                score += 10
            candidates.append((score, asset))
    if not candidates:
        fallback_source = job.file_url if isinstance(job.file_url, str) and job.file_url else job.video_url
        fallback = fallback_source if isinstance(fallback_source, str) else None
        if fallback:
            candidate = fallback.strip()
            if candidate and not candidate.startswith("["):
                if candidate.startswith(("http://", "https://", "files/", "file-")):
                    return {
                        "key": "video_url",
                        "url": candidate,
                        "kind": "video",
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


async def _deliver_remote_video(
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

    method: Literal["video", "document"] = "video"
    message: Optional[Message] = None
    duration_seconds = _extract_video_duration(job)
    video_filename = downloaded.filename or "video.mp4"
    if not video_filename.lower().endswith(".mp4"):
        video_filename = "video.mp4"
    try:
        if downloaded.size <= _TELEGRAM_VIDEO_MAX_BYTES:
            message = await dp.bot.send_video(
                job.user_id,
                BufferedInputFile(
                    downloaded.content,
                    filename=video_filename,
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
                    filename=video_filename,
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
    await state.update_data(
        flow_type="video",
        model=None,
        provider=None,
        model_label=None,
        include_size=True,
        product="sora_video",
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
        product="sora_video",
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
    product = state_data.get("product", "sora_video")
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
    product = state_data.get("product", "sora_video")
    if not message.photo:
        await _send_generation_prompt(
            message,
            session=session,
            category=category,
            mode="photo",
            model_label=model_label,
            include_size=include_size,
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
    await callback.answer()
    parts = (callback.data or "").split(":")
    if len(parts) != 3:
        return
    _, _, key = parts
    option = _find_video_model_option(config, key)
    if option is None:
        await callback.answer("Модель недоступна", show_alert=True)
        return
    await state.update_data(
        flow_type="video",
        model=option.model,
        provider=option.provider,
        model_label=option.label,
        include_size=True,
        product="sora_video",
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
    await callback.answer()
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
    product = data.get("product", "sora_video")
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
    )


async def image_mode_callback_handler(
    callback: CallbackQuery, state: FSMContext, config: Config
) -> None:
    await callback.answer()
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
    product = data.get("product", "sora_video")
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
    )


async def option_callback_handler(
    callback: CallbackQuery, state: FSMContext, config: Config
) -> None:
    data = (callback.data or "").split(":")
    if len(data) != 4:
        await callback.answer()
        return
    _, option_type, context, token = data
    user = callback.from_user
    if user is None:  # pragma: no cover - defensive
        await callback.answer()
        return
    session = SESSION_MANAGER.get(user.id)
    if option_type == "size":
        value = SIZE_OPTIONS.get(token)
        if value:
            session.last_size = value
            aspect = ASPECT_RATIO_OPTIONS.get(token)
            if aspect:
                session.last_aspect_ratio = aspect
    await callback.answer()
    state_data = await state.get_data()
    if state_data.get("flow_type", "video") != "video":
        return
    model = state_data.get("model") or config.default_video_model
    model_label = state_data.get("model_label") or _resolve_model_label(model, config)
    try:
        if context == "photo":
            body = i18n.t("video.prompt.photo", size=session.last_size, model=model_label)
        else:
            body = i18n.t("video.prompt.text", size=session.last_size, model=model_label)
        await callback.message.edit_text(
            body,
            reply_markup=_build_options_keyboard(
                session, context, back_target="video_mode"
            ),
        )
    except Exception:  # pragma: no cover - Telegram may block edits on old messages
        log.debug("Failed to edit options message", exc_info=True)


async def order_callback_handler(
    callback: CallbackQuery,
    state: FSMContext,
    db: Database,
    job_queue: JobQueue,
    config: Config,
    gate: GenerationRequestGate,
    error_reporter: ErrorReporter,
) -> None:
    await callback.answer()
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
        await callback.message.answer(i18n.t("help.main"), reply_markup=_main_keyboard())
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
            product=order.product or "sora_video",
        )
        await _send_generation_prompt(
            callback.message,
            session=session,
            category=order.category,
            mode=order.flow,
            model_label=model_label,
            include_size=include_size,
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
    await callback.answer()
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
    await callback.answer()
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


async def models_command(message: Message, job_queue: JobQueue, config: Config) -> None:
    client: Optional[BaseProviderClient] = None
    for key in (
        config.gemini_model_image,
        "gemini-image",
        "gemini",
    ):
        if not key:
            continue
        candidate = job_queue.get_provider(key)
        if candidate is not None:
            client = candidate
            break
    if client is None or not hasattr(client, "_available_models"):
        await message.answer("Gemini недоступен или не инициализирован.")
        return
    models = getattr(client, "_available_models", []) or []
    if not models:
        await message.answer("Список моделей Gemini пока недоступен. Попробуйте позже.")
        return
    lines = ["Доступные модели Gemini:"]
    for model_name in models:
        if not isinstance(model_name, str):
            continue
        lines.append(f"• {escape_html(model_name)}")
    await message.answer("\n".join(lines))


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
        lambda message: models_command(message, job_queue, config),
        Command("models"),
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
    "balance_command",
    "top_up_menu",
    "resend_pending_order",
]

