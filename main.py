"""Entrypoint for the Sora Telegram bot."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiogram import Bot, Dispatcher, executor

from config import Config, load_config
from db import Database
from handlers import register_handlers
from jobs import JobQueue
from payments_stars import TelegramStarPaymentProcessor
from sora_client import SoraClient

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


async def create_app(config: Config) -> dict[str, Any]:
    bot = Bot(token=config.bot_token, parse_mode="HTML")
    dp = Dispatcher(bot)
    db = Database(config.database_path)
    sora_client = SoraClient(config=config)
    job_queue = JobQueue(db=db, sora_client=sora_client, config=config)
    payments = TelegramStarPaymentProcessor(db=db, config=config)

    register_handlers(dp, db=db, config=config, job_queue=job_queue, payments=payments)

    async def on_startup(dispatcher: Dispatcher) -> None:
        await db.connect()
        await db.run_migrations()
        await job_queue.start()
        log.info("Bot started")

    async def on_shutdown(dispatcher: Dispatcher) -> None:
        await job_queue.stop()
        await sora_client.close()
        await db.close()
        log.info("Bot stopped")

    executor_params = {
        "skip_updates": True,
        "on_startup": on_startup,
        "on_shutdown": on_shutdown,
    }

    return {
        "bot": bot,
        "dispatcher": dp,
        "db": db,
        "sora_client": sora_client,
        "job_queue": job_queue,
        "payments": payments,
        "executor_params": executor_params,
    }


def main() -> None:
    config = load_config()
    loop = asyncio.get_event_loop()
    app = loop.run_until_complete(create_app(config))
    executor.start_polling(app["dispatcher"], **app["executor_params"])


if __name__ == "__main__":
    main()
