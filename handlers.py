"""Aiogram handlers for the Sora Telegram bot."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
import re
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
from db import Database, GenerationJobRecord
from i18n import SafeText, format_credits, format_prompt, i18n
from jobs import JobQueue
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

    def key(self) -> str:
        parts = [self.flow, self.prompt, self.image_file_id or "", self.size, self.model]
        return "|".join(parts)


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
            [InlineKeyboardButton(text=i18n.t("buttons.back"), callback_data="menu:back")],
        ]
    )


def _build_help_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="👤 Написать оператору",
                    url="https://t.me/hr_assistt",
                )
            ]
        ]
    )


async def _send_main_menu(message: Message) -> None:
    await message.answer(i18n.t("main.welcome"), reply_markup=_main_keyboard())


async def _ensure_user(message: Message, db: Database) -> int:
    user = message.from_user
    if user is None:  # pragma: no cover - defensive
        raise RuntimeError("Message without user")
    await db.ensure_user(user.id, user.username)
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
        rows.append([InlineKeyboardButton(text=text, callback_data=f"pay:{package.credits_int}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _send_payment_showcase(bot: Bot, chat_id: int, config: Config) -> None:
    text = _format_packages_list(config)
    keyboard = _build_packages_keyboard(config)
    if keyboard is not None:
        await bot.send_message(chat_id, text, reply_markup=keyboard)
    else:
        body = f"{text}\n\n{i18n.t('payment.unavailable')}"
        await bot.send_message(chat_id, body.strip())


def _price_for_items(config: Config, items: int) -> Optional[int]:
    for package in config.credit_packages:
        if package.credits_int == items:
            return package.price_kopeks
    return None


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
    if session.last_launch_key == key and now - session.last_launch_ts < 5:
        await callback.message.answer(i18n.t("flow.duplicate"))
        return

    if not await db.deduct_credit(user_id, config.generation_cost_credits):
        await callback.message.answer(i18n.t("flow.not_enough"))
        session.awaiting_payment = True
        await _send_payment_showcase(callback.message.bot, callback.message.chat.id, config)
        return

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
            image_file_id=order.image_file_id,
        )
    except Exception as exc:  # pragma: no cover - API interaction
        log.exception("Failed to submit job")
        refund_balance = await db.add_credits(user_id, config.generation_cost_credits)
        log.info(
            "Order launch failed, refund issued user=%s cost_credits=%s balance_after=%s",
            user_id,
            config.generation_cost_credits,
            refund_balance,
        )
        await callback.message.answer(
            i18n.t("status.failed", error=str(exc) or i18n.t("errors.unknown"))
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


async def help_command(message: Message, db: Database, state: FSMContext) -> None:
    await _ensure_user(message, db)
    await message.answer(i18n.t("help.main"), reply_markup=_build_help_keyboard())


async def balance_command(message: Message, db: Database, state: FSMContext) -> None:
    await state.finish()
    user_id = await _ensure_user(message, db)
    credits = await db.get_user_credits(user_id)
    balance_text = i18n.t("balance.info", credits=format_credits(credits))
    await message.answer(
        f"{balance_text}\n{i18n.t('balance.actions')}",
        reply_markup=_build_balance_keyboard(),
    )


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


async def help_button(message: Message, db: Database, state: FSMContext) -> None:
    await help_command(message, db, state)


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
        _, items_raw = data.split(":", maxsplit=1)
        items = int(items_raw)
    except (ValueError, AttributeError):
        return
    package_price_cp = _price_for_items(config, items)
    if package_price_cp is None:
        await callback.message.answer(i18n.t("payment.unknown_package"))
        return

    user = callback.from_user
    if user is None:  # pragma: no cover - defensive
        return
    await db.ensure_user(user.id, user.username)

    description = f"Sora2 {items} credits"
    try:
        payment = await asyncio.to_thread(
            yookassa_client.create_payment,
            package_price_cp,
            user.id,
            items,
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
    await db.create_payment_record(
        "yookassa",
        payment_id,
        user.id,
        package_price_cp,
        items,
        status,
        payload=str(description),
        metadata=metadata,
        idempotency_key=idempotency_key,
    )
    log.info(
        "Created YooKassa payment %s credits=%s amount_cp=%s net_cp=%s",
        payment_id,
        items,
        package_price_cp,
        package_price_cp,
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
    if data.endswith("back"):
        await state.finish()
        await callback.message.answer(i18n.t("help.main"), reply_markup=_main_keyboard())


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
        lambda message, state: help_command(message, db, state),
        Command("help"),
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
        lambda message, state: top_up_menu(message, db, state, config),
        lambda message: message.text == i18n.t("buttons.top_up"),
        state="*",
    )
    dp.register_message_handler(
        lambda message, state: help_button(message, db, state),
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

