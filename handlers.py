"""Aiogram handlers for the Sora Telegram bot."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

from aiogram import Dispatcher
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command, CommandStart
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import (
    Bot,
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

MODEL_OPTIONS: Dict[str, str] = {
    "sora2": "Sora-2",
    "sora2pro": "Sora-2 Pro",
}

SIZE_LABEL_KEYS: Dict[str, str] = {
    "vertical": "chips.vertical",
    "horizontal": "chips.horizontal",
}

MODEL_LABEL_KEYS: Dict[str, str] = {
    "sora2": "chips.model_basic",
    "sora2pro": "chips.model_pro",
}

PAYMENT_PACKAGES: Tuple[Tuple[int, str], ...] = (
    (1, "price_one_rub"),
    (5, "price_five_rub"),
    (10, "price_ten_rub"),
    (30, "price_thirty_rub"),
)

MAIN_MENU_BUTTONS = {
    i18n.t("buttons.generate_text"),
    i18n.t("buttons.generate_photo"),
    i18n.t("buttons.choose_model"),
    i18n.t("buttons.top_up"),
    i18n.t("buttons.help"),
}


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
    last_model: str = MODEL_OPTIONS["sora2"]
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
            [
                KeyboardButton(text=i18n.t("buttons.choose_model")),
                KeyboardButton(text=i18n.t("buttons.top_up")),
            ],
            [KeyboardButton(text=i18n.t("buttons.help"))],
        ],
        resize_keyboard=True,
    )


def _mark_selected(label: str, selected: bool) -> str:
    return f"✅ {label}" if selected else label


def _build_options_keyboard(session: UserSession, context: str) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    include_size = context in {"text", "photo"}
    if include_size:
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
        rows.append(size_row)
    model_row = []
    for token, value in MODEL_OPTIONS.items():
        label_key = MODEL_LABEL_KEYS[token]
        label = i18n.t(label_key)
        model_row.append(
            InlineKeyboardButton(
                text=_mark_selected(label, session.last_model == value),
                callback_data=f"opt:model:{context}:{token}",
            )
        )
    rows.append(model_row)
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


def _build_balance_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=i18n.t("buttons.top_up_short"), callback_data="menu:topup")],
            [InlineKeyboardButton(text=i18n.t("buttons.back"), callback_data="menu:back")],
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
    session: UserSession,
    image_file_id: Optional[str] = None,
) -> OrderContext:
    return OrderContext(
        flow=flow,
        prompt=prompt,
        size=session.last_size,
        model=session.last_model,
        image_file_id=image_file_id,
    )


async def _send_options_prompt(message: Message, *, session: UserSession, context: str) -> None:
    if context == "photo":
        body = i18n.t("flow.photo.prompt", size=session.last_size, model=session.last_model)
    elif context == "model":
        body = (
            f"{i18n.t('model.prompt')}\n"
            f"{i18n.t('model.current', model=session.last_model)}"
        )
    else:
        body = i18n.t("flow.text.prompt", size=session.last_size, model=session.last_model)
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
    price_text = format_credits(config.credits_per_generation)
    prompt_text: SafeText = format_prompt(order.prompt)
    if order.flow == "photo":
        body = i18n.t(
            "flow.confirm.photo",
            prompt=prompt_text,
            size=order.size,
            model=order.model,
            price=price_text,
        )
    else:
        body = i18n.t(
            "flow.confirm.text",
            prompt=prompt_text,
            size=order.size,
            model=order.model,
            price=price_text,
        )
    keyboard = _build_confirmation_keyboard(can_launch=can_launch, include_back=include_back)
    return await bot.send_message(chat_id, body, reply_markup=keyboard)


def _format_packages_list(config: Config) -> str:
    lines: list[str] = [i18n.t("payment.packages.title")]
    for items, attr in PAYMENT_PACKAGES:
        price = getattr(config, attr)
        lines.append(i18n.t(f"payment.package.label.{items}", price=price))
        lines.append(i18n.t("payment.package.subtitle"))
        lines.append("")
    lines.append(i18n.t("payment.store.instructions"))
    return "\n".join(line for line in lines if line).strip()


def _build_packages_keyboard(config: Config) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    for items, attr in PAYMENT_PACKAGES:
        price = getattr(config, attr)
        text = i18n.t(f"payment.package.label.{items}", price=price)
        rows.append([InlineKeyboardButton(text=text, callback_data=f"pay:{items}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def _send_payment_showcase(bot: Bot, chat_id: int, config: Config) -> None:
    text = _format_packages_list(config)
    keyboard = _build_packages_keyboard(config)
    await bot.send_message(chat_id, text, reply_markup=keyboard)


def _price_for_items(config: Config, items: int) -> Optional[int]:
    for count, attr in PAYMENT_PACKAGES:
        if count == items:
            return getattr(config, attr)
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
    can_launch = credits >= config.credits_per_generation
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

    if not await db.deduct_credit(user_id, config.credits_per_generation):
        await callback.message.answer(i18n.t("flow.not_enough"))
        session.awaiting_payment = True
        await _send_payment_showcase(callback.message.bot, callback.message.chat.id, config)
        return

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
        await db.add_credits(user_id, config.credits_per_generation)
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
    await state.finish()
    await _ensure_user(message, db)
    await message.answer(i18n.t("help.main"), reply_markup=_main_keyboard())


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


async def choose_model_menu(message: Message, db: Database, state: FSMContext) -> None:
    await state.finish()
    user_id = await _ensure_user(message, db)
    session = SESSION_MANAGER.get(user_id)
    await _send_options_prompt(message, session=session, context="model")


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
    prompt = (message.text or "").strip()
    if not prompt:
        await message.answer(i18n.t("flow.no_prompt"))
        return
    await state.finish()
    order = _create_order(flow="text", prompt=prompt, session=session)
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
    caption = (message.caption or "").strip()
    order = _create_order(
        flow="photo",
        prompt=caption,
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
    elif option_type == "model":
        value = MODEL_OPTIONS.get(token)
        if value:
            session.last_model = value
    await callback.answer()
    try:
        await callback.message.edit_text(
            (
                i18n.t("flow.text.prompt", size=session.last_size, model=session.last_model)
                if context == "text"
                else i18n.t("flow.photo.prompt", size=session.last_size, model=session.last_model)
                if context == "photo"
                else (
                    f"{i18n.t('model.prompt')}\n"
                    f"{i18n.t('model.current', model=session.last_model)}"
                )
            ),
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
    if not config.yookassa_enabled or not config.public_base_url:
        await callback.message.answer(i18n.t("payment.unavailable"))
        return
    data = callback.data or ""
    try:
        _, items_raw = data.split(":", maxsplit=1)
        items = int(items_raw)
    except (ValueError, AttributeError):
        return
    amount = _price_for_items(config, items)
    if amount is None:
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
            amount,
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

    await db.create_payment_record(
        "yookassa",
        payment_id,
        user.id,
        amount * 100,
        items,
        payment.get("status", "pending") or "pending",
        metadata,
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
    can_launch = credits >= config.credits_per_generation
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
        lambda message, state: choose_model_menu(message, db, state),
        lambda message: message.text == i18n.t("buttons.choose_model"),
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

