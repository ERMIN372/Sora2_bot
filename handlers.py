"""Aiogram handlers for the Sora Telegram bot."""
from __future__ import annotations

import logging
from typing import Optional

from aiogram import Dispatcher
from aiogram.dispatcher.filters import Command
from aiogram.types import Message

from config import Config
from db import Database, GenerationJobRecord
from jobs import JobQueue
from payments_stars import TelegramStarPaymentProcessor

log = logging.getLogger(__name__)


async def _send_job_update(dp: Dispatcher, job: GenerationJobRecord) -> None:
    try:
        await dp.bot.send_message(
            job.user_id,
            _format_job_message(job),
        )
    except Exception:  # pragma: no cover - external dependency
        log.exception("Failed to send job update to %s", job.user_id)


def _format_job_message(job: GenerationJobRecord) -> str:
    if job.status == "completed" and job.video_url:
        return (
            "🎉 Your video is ready!\n"
            f"Prompt: {job.prompt}\n"
            f"Download: {job.video_url}"
        )
    if job.status == "failed":
        return (
            "⚠️ Video generation failed. Your credit has been refunded.\n"
            f"Reason: {job.error or 'Unknown'}"
        )
    return f"⏳ Job {job.id} is {job.status}."


async def start_command(message: Message, db: Database) -> None:
    await db.ensure_user(message.from_user.id, message.from_user.username)
    await message.answer(
        "👋 Welcome to the Sora bot!\n"
        "Use /generate <prompt> to create a video once you have credits."
    )


async def balance_command(message: Message, db: Database) -> None:
    credits = await db.get_user_credits(message.from_user.id)
    await message.answer(f"You have {credits} credits available.")


async def generate_command(message: Message, *, db: Database, config: Config, job_queue: JobQueue) -> None:
    prompt = _extract_prompt(message.text)
    if not prompt:
        await message.answer("Please provide a prompt: /generate <prompt>")
        return

    user_id = message.from_user.id
    await db.ensure_user(user_id, message.from_user.username)

    active_jobs = await db.count_active_jobs(user_id)
    if active_jobs >= config.max_jobs_per_user:
        await message.answer("You have too many pending jobs. Please wait for them to finish.")
        return

    if not await db.deduct_credit(user_id, config.credits_per_generation):
        await message.answer("You do not have enough credits. Purchase more using Telegram Stars.")
        return

    await message.answer("Your prompt has been queued. I'll let you know when it's done!")

    try:
        await job_queue.submit(user_id=user_id, prompt=prompt)
    except Exception as exc:  # pragma: no cover - API interaction
        log.exception("Failed to submit job")
        await db.add_credits(user_id, config.credits_per_generation)
        await message.answer(f"Failed to submit job: {exc}")
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
        "✨ Payment received! "
        f"You have been credited with {result.credits_added} credits."
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
        lambda message: successful_payment_handler(message, payments=payments),
        content_types=["successful_payment"],
    )


__all__ = [
    "register_handlers",
    "start_command",
    "balance_command",
    "generate_command",
]
