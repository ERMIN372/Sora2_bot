"""Aiogram handlers for the Sora Telegram bot."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
import re
import uuid
from datetime import datetime
from typing import Dict, Literal, Optional

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

from config import Config
from db import Database, ErrorLogRecord, GenerationJobRecord
from i18n import SafeText, escape_html, format_credits, format_prompt, i18n
from jobs import JobQueue
from observability import (
    get_job_history,
    get_last_error,
    get_last_healthcheck,
    log_event,
    should_notify_support,
)
from sora_client import SoraAPIError
import yookassa_client

log = logging.getLogger(__name__)


SIZE_OPTIONS: Dict[str, str] = {
    "vertical": "720x1280",
    "horizontal": "1280x720",
}

SIZE_LABEL_KEYS: Dict[str, str] = {
    "vertical": "chips.vertical",
    "horizontal": "chips.horizontal",
}

MAIN_MENU_BUTTONS = {
    i18n.t("buttons.generate_text"),
    i18n.t("buttons.generate_photo"),
    i18n.t("buttons.balance"),
    i18n.t("buttons.top_up"),
    i18n.t("buttons.help"),
}

_PRO_REQUEST_PATTERN = re.compile(
    r"\b(?:sora(?:[-\s]?2)?[-\s]*pro|sora2pro|model\s*[:=]?\s*pro|pro-?версия|pro version)\b",
    re.IGNORECASE,
)


class GenerationStates(StatesGroup):
    """Conversation states for collecting generation inputs."""

    text_prompt = State()
    photo_prompt = State()


@dataclass
class OrderContext:
    """Details of a generation order pending confirmation."""

    flow: Literal["text", "photo"]
    prompt: str
    size: str
    model: str
    image_file_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    corr_id: Optional[str] = None

    def key(self) -> str:
        parts = [self.flow, self.prompt, self.image_file_id or "", self.size, self.model]
        return "|".join(parts)


def _shorten(text: str, limit: int = 240) -> str:
    snippet = (text or "").strip()
    if len(snippet) <= limit:
        return snippet
    return snippet[: limit - 3] + "..."


def _map_sora_error(error: SoraAPIError) -> tuple[str, str, bool, str]:
    status = error.status_code or 0
    provider_message = (error.provider_message or str(error) or "").strip()
    error_type = (error.error_type or "").lower()
    error_code = (error.error_code or "").lower()
    short = _shorten(provider_message or (error.args[0] if error.args else "Ошибка"))
    notify_support = False
    hint = error_type or error_code or "unknown"

    if status in {401, 403}:
        notify_support = True
        hint = "auth"
        message = (
            "API-ключ недействителен или нет доступа к Sora. "
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
    elif "policy" in error_type or "policy" in error_code:
        hint = "policy"
        message = "Запрос нарушает правила контента."
        short = _shorten(provider_message or "Нарушение политики")
    else:
        detail = escape_html(provider_message or (error.args[0] if error.args else ""))
        message = f"Не удалось выполнить запрос: {detail or 'попробуйте позже.'}"
        short = _shorten(provider_message or "Ошибка Sora")
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
        "⚠️ Критическая ошибка Sora",
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


def _build_options_keyboard(session: UserSession, context: str) -> InlineKeyboardMarkup:
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
    return InlineKeyboardMarkup(inline_keyboard=[size_row])


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


def _build_balance_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=i18n.t("buttons.top_up_short"), callback_data="menu:topup")],
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


async def _send_main_menu(message: Message) -> None:
    await message.answer(i18n.t("main.welcome"), reply_markup=_main_keyboard())


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
    flow: Literal["text", "photo"],
    prompt: str,
    config: Config,
    session: UserSession,
    image_file_id: Optional[str] = None,
) -> OrderContext:
    return OrderContext(
        flow=flow,
        prompt=prompt,
        size=session.last_size,
        model=config.sora_model,
        image_file_id=image_file_id,
    )


def _detect_pro_request(text: Optional[str]) -> bool:
    if not text:
        return False
    normalised = text.strip().lower()
    if normalised in {"pro", "pro-версия", "pro версия", "pro version"}:
        return True
    return bool(_PRO_REQUEST_PATTERN.search(text))


_MODEL_DIRECTIVE_PATTERN = re.compile(
    r"model\s*(?:[:=]\s*)?(?:sora(?:[-\s]?2)?[-\s]*pro|sora2pro|pro)",
    re.IGNORECASE,
)
_MODEL_NAME_PATTERN = re.compile(r"sora(?:[-\s]?2)?[-\s]*pro", re.IGNORECASE)


def _strip_pro_directives(text: str) -> str:
    cleaned = _MODEL_DIRECTIVE_PATTERN.sub("", text)
    cleaned = _MODEL_NAME_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


async def _notify_pro_unavailable(message: Message) -> None:
    await message.answer(i18n.t("errors.pro_unavailable"))


async def _send_options_prompt(message: Message, *, session: UserSession, context: str) -> None:
    if context == "photo":
        body = i18n.t("flow.photo.prompt", size=session.last_size)
    else:
        body = i18n.t("flow.text.prompt", size=session.last_size)
    await message.answer(body, reply_markup=_build_options_keyboard(session, context))


async def _send_order_confirmation(
    *,
    bot: Bot,
    chat_id: int,
    order: OrderContext,
    config: Config,
    can_launch: bool,
    include_back: bool = True,
) -> Message:
    approx_rub = config.generation_cost_approx_rubles()
    price_text = i18n.t(
        "flow.price_tag",
        credits=format_credits(config.generation_cost_credits),
        approx=f"{approx_rub} ₽",
    )
    prompt_text: SafeText = format_prompt(order.prompt)
    if order.flow == "photo":
        body = i18n.t(
            "flow.confirm.photo",
            prompt=prompt_text,
            size=order.size,
            price=price_text,
        )
    else:
        body = i18n.t(
            "flow.confirm.text",
            prompt=prompt_text,
            size=order.size,
            price=price_text,
        )
    keyboard = _build_confirmation_keyboard(can_launch=can_launch, include_back=include_back)
    return await bot.send_message(chat_id, body, reply_markup=keyboard)


def _format_packages_list(config: Config) -> str:
    lines: list[str] = [i18n.t("payment.packages.title")]
    for package in config.credit_packages:
        credits_label = format_credits(package.credits_int)
        price_label = f"{package.price_rubles} ₽"
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
        text = i18n.t(
            "payment.packages.line",
            credits=format_credits(package.credits_int),
            price=f"{package.price_rubles} ₽",
        )
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
    can_launch = credits >= config.generation_cost_credits
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
) -> None:
    user = callback.from_user
    if user is None:  # pragma: no cover - defensive
        return
    user_id = user.id
    now = time.time()
    key = order.key()
    await db.sync_user_profile(
        user_id,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
    )
    if session.last_launch_key == key and now - session.last_launch_ts < 5:
        await callback.message.answer(i18n.t("flow.duplicate"))
        return

    corr_id = order.corr_id or str(uuid.uuid4())
    order.corr_id = corr_id
    log_event(
        level="INFO",
        event="enqueue",
        corr_id=corr_id,
        user_id=user_id,
        username=user.username,
        model=config.sora_model,
        size=order.size,
        credits_cost=config.generation_cost_credits,
        prompt=order.prompt,
    )

    if not await db.deduct_credit(user_id, config.generation_cost_credits):
        log_event(
            level="ERROR",
            event="error",
            corr_id=corr_id,
            user_id=user_id,
            username=user.username,
            model=config.sora_model,
            size=order.size,
            error_type="insufficient_credits",
            error_msg_short="Недостаточно кредитов",
            prompt=order.prompt,
        )
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
        model=config.sora_model,
        size=order.size,
        credits_cost=config.generation_cost_credits,
    )

    balance_after = await db.get_user_credits(user_id)
    log.info(
        "Launching job user=%s cost_credits=%s balance_after=%s",
        user_id,
        config.generation_cost_credits,
        balance_after,
    )

    try:
        await job_queue.submit(
            user_id=user_id,
            prompt=order.prompt,
            size=order.size,
            model=order.model,
            corr_id=corr_id,
            image_file_id=order.image_file_id,
            username=user.username,
        )
    except SoraAPIError as exc:
        message, short, notify_support, hint = _map_sora_error(exc)
        refunded = False
        try:
            await db.add_credits(user_id, config.generation_cost_credits)
            refunded = True
        except Exception:  # pragma: no cover - external dependency
            log.exception("Failed to refund credits after submit error")
        sheet_ok = await db.log_error_record(
            ErrorLogRecord(
                ts=datetime.utcnow(),
                user_id=user_id,
                username=user.username,
                corr_id=corr_id,
                job_id="",
                model=config.sora_model,
                size=order.size,
                status_code=exc.status_code,
                error_type=exc.error_type or "submit_failed",
                error_msg_short=short,
                refunded=refunded,
            )
        )
        log_event(
            level="ERROR",
            event="error",
            corr_id=corr_id,
            user_id=user_id,
            username=user.username,
            model=config.sora_model,
            size=order.size,
            status_code=exc.status_code,
            duration_ms=exc.duration_ms,
            error_type=exc.error_type,
            error_code=exc.error_code,
            error_msg_short=short,
            prompt=order.prompt,
            gsheets_ok=sheet_ok,
            refund_done=refunded,
            extra={"provider_message": exc.provider_message},
        )
        log_event(
            level="INFO",
            event="refund",
            corr_id=corr_id,
            user_id=user_id,
            username=user.username,
            model=config.sora_model,
            size=order.size,
            credits_cost=config.generation_cost_credits,
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
            await db.add_credits(user_id, config.generation_cost_credits)
            refunded = True
        except Exception:
            log.exception("Failed to refund credits after unexpected error")
        short = _shorten(str(exc) or "Неизвестная ошибка")
        sheet_ok = await db.log_error_record(
            ErrorLogRecord(
                ts=datetime.utcnow(),
                user_id=user_id,
                username=user.username,
                corr_id=corr_id,
                job_id="",
                model=config.sora_model,
                size=order.size,
                status_code=None,
                error_type="internal",
                error_msg_short=short,
                refunded=refunded,
            )
        )
        log_event(
            level="ERROR",
            event="error",
            corr_id=corr_id,
            user_id=user_id,
            username=user.username,
            model=config.sora_model,
            size=order.size,
            error_type="internal",
            error_msg_short=short,
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
            model=config.sora_model,
            size=order.size,
            credits_cost=config.generation_cost_credits,
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


def _format_job_message(job: GenerationJobRecord) -> str:
    if job.status == "completed":
        if job.video_url:
            return i18n.t("status.completed_url", url=job.video_url)
        return i18n.t("status.completed_file")
    if job.status == "failed":
        return i18n.t("status.failed", error=job.error or i18n.t("errors.unknown"))
    if job.status == "running":
        return i18n.t("status.running")
    return i18n.t("status.generic", status=job.status or i18n.t("errors.unknown"))


async def _send_job_update(dp: Dispatcher, job: GenerationJobRecord) -> None:
    try:
        await dp.bot.send_message(job.user_id, _format_job_message(job))
    except Exception:  # pragma: no cover - external dependency
        log.exception("Failed to send job update to %s", job.user_id)


async def start_command(message: Message, db: Database, state: FSMContext) -> None:
    await state.finish()
    await _ensure_user(message, db)
    await _send_main_menu(message)


async def help_command(
    message: Message, db: Database, state: FSMContext, config: Config
) -> None:
    await _ensure_user(message, db)
    await message.answer(
        i18n.t("help.main"),
        reply_markup=_build_help_keyboard(config),
    )


async def balance_command(message: Message, db: Database, state: FSMContext) -> None:
    await _send_balance_info(message, db)


async def generate_text_menu(message: Message, db: Database, state: FSMContext) -> None:
    await state.finish()
    user_id = await _ensure_user(message, db)
    session = SESSION_MANAGER.get(user_id)
    await GenerationStates.text_prompt.set()
    await _send_options_prompt(message, session=session, context="text")


async def generate_photo_menu(message: Message, db: Database, state: FSMContext) -> None:
    await state.finish()
    user_id = await _ensure_user(message, db)
    session = SESSION_MANAGER.get(user_id)
    await GenerationStates.photo_prompt.set()
    await _send_options_prompt(message, session=session, context="photo")


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
    raw_prompt = (message.text or "").strip()
    prompt = raw_prompt
    if _detect_pro_request(raw_prompt):
        await _notify_pro_unavailable(message)
        prompt = _strip_pro_directives(raw_prompt)
    if not prompt:
        await message.answer(i18n.t("flow.no_prompt"))
        return
    await state.finish()
    order = _create_order(flow="text", prompt=prompt, config=config, session=session)
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
    if not message.photo:
        await _send_options_prompt(message, session=session, context="photo")
        return
    largest = max(message.photo, key=lambda item: item.file_size or 0)
    raw_caption = (message.caption or "").strip()
    caption = raw_caption
    if _detect_pro_request(raw_caption):
        await _notify_pro_unavailable(message)
        caption = _strip_pro_directives(raw_caption)
    order = _create_order(
        flow="photo",
        prompt=caption,
        config=config,
        session=session,
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


async def option_callback_handler(callback: CallbackQuery, state: FSMContext) -> None:
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
    await callback.answer()
    try:
        if context == "photo":
            body = i18n.t("flow.photo.prompt", size=session.last_size)
        else:
            body = i18n.t("flow.text.prompt", size=session.last_size)
        await callback.message.edit_text(
            body,
            reply_markup=_build_options_keyboard(session, context),
        )
    except Exception:  # pragma: no cover - Telegram may block edits on old messages
        log.debug("Failed to edit options message", exc_info=True)


async def order_callback_handler(
    callback: CallbackQuery,
    state: FSMContext,
    db: Database,
    job_queue: JobQueue,
    config: Config,
) -> None:
    await callback.answer()
    action = (callback.data or "").split(":", maxsplit=1)[-1]
    user = callback.from_user
    if user is None:  # pragma: no cover - defensive
        return
    session = SESSION_MANAGER.get(user.id)
    order = session.pending_order
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
        context = "photo" if order.flow == "photo" else "text"
        if order.flow == "photo":
            await GenerationStates.photo_prompt.set()
        else:
            await GenerationStates.text_prompt.set()
        await _send_options_prompt(callback.message, session=session, context=context)
        return
    if action == "launch":
        await _launch_order(
            callback=callback,
            order=order,
            session=session,
            db=db,
            job_queue=job_queue,
            config=config,
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

    description = f"Sora2 {package.credits_int} credits"
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
    can_launch = credits >= config.generation_cost_credits
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
) -> None:
    job_queue.register_notification_callback(
        name="telegram",
        callback=lambda job: _send_job_update(dp, job),
    )

    dp.register_message_handler(
        lambda message, state: start_command(message, db, state),
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
        lambda message, state: generate_text_menu(message, db, state),
        lambda message: message.text == i18n.t("buttons.generate_text"),
        state="*",
    )
    dp.register_message_handler(
        lambda message, state: generate_photo_menu(message, db, state),
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
        option_callback_handler,
        lambda call: call.data and call.data.startswith("opt:"),
        state="*",
    )
    dp.register_callback_query_handler(
        lambda call, state: order_callback_handler(call, state, db, job_queue, config),
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

