"""Aiogram handlers for the Sora Telegram bot."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from aiogram import Dispatcher
from aiogram.dispatcher.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from config import Config
from db import Database, GenerationJobRecord
from jobs import JobQueue
from payments_stars import TelegramStarPaymentProcessor
import yookassa_client

from i18n import format_prompt, i18n

log = logging.getLogger(__name__)


async def _send_job_update(dp: Dispatcher, job: GenerationJobRecord) -> None:
    try:
        await dp.bot.send_message(job.user_id, _format_job_message(job))
    except Exception:  # pragma: no cover - external dependency
        log.exception("Failed to send job update to %s", job.user_id)


def _format_job_message(job: GenerationJobRecord) -> str:
    if job.status == "completed":
        prompt = format_prompt(job.prompt)
        if job.video_url:
            return i18n.t("job.completed_with_url", url=job.video_url, prompt=prompt)
        return i18n.t("job.completed_no_url", prompt=prompt)
    if job.status == "failed":
        return i18n.t("job.failed", error=job.error or i18n.t("errors.unknown"))
    if job.status == "running":
        return i18n.t("job.status.running")
    if job.status == "queued":
        return i18n.t("job.status.queued")
    return i18n.t(
        "job.status.generic",
        job_id=job.id,
        status=job.status or i18n.t("errors.unknown"),
    )


async def start_command(message: Message, db: Database) -> None:
    await db.ensure_user(message.from_user.id, message.from_user.username)
    await message.answer(i18n.t("start.welcome"))


async def balance_command(message: Message, db: Database) -> None:
    credits = await db.get_user_credits(message.from_user.id)
    bonus_granted = await db.is_bonus_granted(message.from_user.id)
    bonus_status = (
        i18n.t("bonus.status.received") if bonus_granted else i18n.t("bonus.status.not_received")
    )
    await message.answer(i18n.t("balance.info", credits=credits, bonus_status=bonus_status))


async def generate_command(message: Message, *, db: Database, config: Config, job_queue: JobQueue) -> None:
    prompt = _extract_prompt(message.text)
    if not prompt:
        await message.answer(i18n.t("generate.prompt_required"))
        return

    user_id = message.from_user.id
    await db.ensure_user(user_id, message.from_user.username)

    active_jobs = await db.count_active_jobs(user_id)
    if active_jobs >= config.max_jobs_per_user:
        await message.answer(
            i18n.t("generate.too_many_jobs", limit=config.max_jobs_per_user)
        )
        return

    if not await db.deduct_credit(user_id, config.credits_per_generation):
        await message.answer(i18n.t("generate.not_enough_credits"))
        return

    await message.answer(i18n.t("generate.queued", prompt=format_prompt(prompt)))

    try:
        await job_queue.submit(user_id=user_id, prompt=prompt)
    except Exception as exc:  # pragma: no cover - API interaction
        log.exception("Failed to submit job")
        await db.add_credits(user_id, config.credits_per_generation)
        await message.answer(i18n.t("generate.submission_failed", error=str(exc)))
        return


async def successful_payment_handler(
    message: Message,
    *,
    payments: TelegramStarPaymentProcessor,
) -> None:
    result = await payments.handle_successful_payment(message)
    if not result:
        return
    await message.answer(
        i18n.t("payment.stars.received", credits=result.credits_added)
    )
    if result.bonus_status == "granted":
        await message.answer(i18n.t("bonus.granted"))
    elif result.bonus_status == "already":
        await message.answer(i18n.t("bonus.already"))


def _build_buy_card_keyboard(config: Config) -> InlineKeyboardMarkup:
    buttons = [
        [
            InlineKeyboardButton(
                text=i18n.t("buy_card.button", items=1, price=config.price_one_rub),
                callback_data="buy_card:1",
            )
        ],
        [
            InlineKeyboardButton(
                text=i18n.t("buy_card.button", items=5, price=config.price_five_rub),
                callback_data="buy_card:5",
            )
        ],
        [
            InlineKeyboardButton(
                text=i18n.t("buy_card.button", items=10, price=config.price_ten_rub),
                callback_data="buy_card:10",
            )
        ],
        [
            InlineKeyboardButton(
                text=i18n.t("buy_card.button", items=30, price=config.price_thirty_rub),
                callback_data="buy_card:30",
            )
        ],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


async def buy_card_command(message: Message, db: Database, config: Config) -> None:
    if not config.yookassa_enabled or not config.public_base_url:
        await message.answer(i18n.t("buy_card.unavailable"))
        return
    await db.ensure_user(message.from_user.id, message.from_user.username)
    await message.answer(
        i18n.t("buy_card.choose_package"),
        reply_markup=_build_buy_card_keyboard(config),
    )


def _price_for_items(config: Config, items: int) -> Optional[int]:
    mapping = {
        1: config.price_one_rub,
        5: config.price_five_rub,
        10: config.price_ten_rub,
        30: config.price_thirty_rub,
    }
    return mapping.get(items)


async def buy_card_callback(
    callback: CallbackQuery,
    *,
    db: Database,
    config: Config,
) -> None:
    await callback.answer()
    if not config.yookassa_enabled or not config.public_base_url:
        await callback.message.answer(i18n.t("buy_card.unavailable"))
        return
    data = callback.data or ""
    try:
        _, raw_items = data.split(":", maxsplit=1)
        items = int(raw_items)
    except (ValueError, AttributeError):
        return
    amount_rub = _price_for_items(config, items)
    if amount_rub is None:
        await callback.message.answer(i18n.t("buy_card.unknown_package"))
        return

    user = callback.from_user
    await db.ensure_user(user.id, user.username)

    description = i18n.t("buy_card.invoice_description", items=items)
    try:
        payment = await asyncio.to_thread(
            yookassa_client.create_payment,
            amount_rub,
            user.id,
            items,
            description,
        )
    except Exception:  # pragma: no cover - external API
        log.exception("Failed to create YooKassa payment")
        await callback.message.answer(i18n.t("buy_card.payment_creation_failed"))
        return

    confirmation_url = payment.get("confirmation_url")
    payment_id = payment.get("payment_id")
    metadata_raw = payment.get("metadata", "{}")
    try:
        metadata = json.loads(metadata_raw)
    except json.JSONDecodeError:
        metadata = {}
    metadata.update({"amount_rub": amount_rub, "order_id": payment.get("order_id")})

    if not payment_id or not confirmation_url:
        await callback.message.answer(i18n.t("buy_card.payment_link_failed"))
        return

    await db.create_payment_record(
        "yookassa",
        payment_id,
        user.id,
        amount_rub * 100,
        items,
        payment.get("status", "pending") or "pending",
        json.dumps(metadata),
    )

    pay_keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text=i18n.t("buy_card.pay_button"), url=confirmation_url)]]
    )

    await callback.message.answer(
        i18n.t("buy_card.payment_link"),
        reply_markup=pay_keyboard,
    )


def _extract_prompt(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        return None
    return parts[1].strip() or None


def register_handlers(
    dp: Dispatcher,
    *,
    db: Database,
    config: Config,
    job_queue: JobQueue,
    payments: TelegramStarPaymentProcessor,
) -> None:
    job_queue.register_notification_callback(
        name="telegram",
        callback=lambda job: _send_job_update(dp, job),
    )

    dp.register_message_handler(lambda message: start_command(message, db), Command("start"))
    dp.register_message_handler(lambda message: balance_command(message, db), Command("balance"))
    dp.register_message_handler(
        lambda message: generate_command(message, db=db, config=config, job_queue=job_queue),
        Command("generate"),
    )
    dp.register_message_handler(
        lambda message: buy_card_command(message, db, config),
        Command("buy_card"),
    )
    dp.register_message_handler(
        lambda message: successful_payment_handler(message, payments=payments),
        content_types=["successful_payment"],
    )
    dp.register_callback_query_handler(
        lambda call: buy_card_callback(call, db=db, config=config),
        lambda call: call.data and call.data.startswith("buy_card:"),
    )


__all__ = [
    "register_handlers",
    "start_command",
    "balance_command",
    "generate_command",
    "buy_card_command",
]
