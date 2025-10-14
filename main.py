"""Entrypoint for the Sora Telegram bot."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

from aiogram import Bot, Dispatcher, executor
from fastapi import FastAPI
import uvicorn

from config import Config, load_config
from db import Database
from handlers import register_handlers
from jobs import JobQueue
from payments_stars import TelegramStarPaymentProcessor
from sora_client import SoraClient
from yookassa_server import YooKassaProcessor, create_router, poll_pending_payments
import yookassa_client

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


async def create_app(config: Config) -> dict[str, Any]:
    bot = Bot(token=config.bot_token, parse_mode="HTML")
    dp = Dispatcher(bot)
    db = Database(config.database_path)
    sora_client = SoraClient(config=config)
    job_queue = JobQueue(db=db, sora_client=sora_client, config=config)
    payments = TelegramStarPaymentProcessor(db=db, config=config)
    fastapi_app: Optional[FastAPI] = None
    yookassa_processor: Optional[YooKassaProcessor] = None
    background_tasks: List[asyncio.Task[Any]] = []

    async def _serve_fastapi(app: FastAPI) -> None:
        server_config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=config.yookassa_server_port,
            log_level="info",
            loop="asyncio",
        )
        server = uvicorn.Server(server_config)
        try:
            await server.serve()
        except asyncio.CancelledError:  # pragma: no cover - shutdown guard
            server.should_exit = True
            raise

    if config.yookassa_enabled and config.public_base_url:
        fastapi_app = FastAPI()
        yookassa_processor = YooKassaProcessor(db=db, config=config, bot=bot)
        fastapi_app.include_router(create_router(yookassa_processor, config))

    register_handlers(dp, db=db, config=config, job_queue=job_queue, payments=payments)

    async def on_startup(dispatcher: Dispatcher) -> None:
        await db.connect()
        await db.run_migrations()
        await job_queue.start()
        if config.yookassa_enabled and config.public_base_url and fastapi_app and yookassa_processor:
            try:
                yookassa_client.init(config)
            except RuntimeError:
                log.warning("YooKassa is not fully configured; skipping card payments")
            else:
                background_tasks.append(
                    asyncio.create_task(_serve_fastapi(fastapi_app))
                )
                background_tasks.append(
                    asyncio.create_task(
                        poll_pending_payments(yookassa_processor, config.yookassa_poll_interval)
                    )
                )
        log.info("Bot started")

    async def on_shutdown(dispatcher: Dispatcher) -> None:
        await job_queue.stop()
        await sora_client.close()
        for task in background_tasks:
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
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
        "fastapi_app": fastapi_app,
        "background_tasks": background_tasks,
        "yookassa_processor": yookassa_processor,
        "executor_params": executor_params,
    }


def main() -> None:
    config = load_config()
    loop = asyncio.get_event_loop()
    app = loop.run_until_complete(create_app(config))
    executor.start_polling(app["dispatcher"], **app["executor_params"])


if __name__ == "__main__":
    main()
