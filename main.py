"""Entrypoint for the Sora Telegram bot."""
from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aiogram import Bot, Dispatcher, executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from fastapi import FastAPI
import uvicorn

from app_server import YooKassaProcessor, create_app, poll_pending_payments
from config import CFG, Config, load_config
from db import Database
from handlers import register_handlers
from jobs import JobQueue
from healthcheck import run_startup_healthcheck
from providers import VertexImageClient, VertexVideoClient
import yookassa_client

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ALLOWED_UPDATES: List[str] = ["message", "callback_query"]


@dataclass
class ApplicationState:
    config: Config
    bot: Bot
    dp: Dispatcher
    db: Database
    job_queue: JobQueue
    app: FastAPI
    yookassa_processor: Optional[YooKassaProcessor]
    background_tasks: List[asyncio.Task[Any]] = field(default_factory=list)
    uvicorn_task: Optional[asyncio.Task[Any]] = None
    uvicorn_server: Optional[uvicorn.Server] = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sora Telegram bot runner")
    parser.add_argument(
        "--mode",
        choices=("polling", "webhook"),
        help="Override BOT_MODE from the environment",
    )
    return parser.parse_args()


def _init_application(config: Config) -> ApplicationState:
    bot = Bot(token=config.bot_token, parse_mode="HTML")
    dp = Dispatcher(bot, storage=MemoryStorage())
    db = Database()
    veo3_client = VertexVideoClient(
        config=config,
        model="veo-3.0-generate-001",
        method="generateContent",
        provider_name="veo3",
    )
    veo31_client = VertexVideoClient(
        config=config,
        model="veo-3.1-generate-preview",
        method="generateContent",
        provider_name="veo3.1",
    )
    gemini_client = VertexImageClient(
        config=config,
        model="gemini-2.5-flash-image",
        method="generateContent",
        provider_name="gemini-image",
    )
    providers = {
        "veo3": veo3_client,
        "sora2": veo3_client,
        "veo3.1": veo31_client,
        "veo31": veo31_client,
        "gemini-image": gemini_client,
    }
    if config.sora_model not in providers:
        providers[config.sora_model] = veo3_client
    job_queue = JobQueue(
        db=db,
        providers=providers,
        default_provider=config.sora_model,
        config=config,
    )
    register_handlers(dp, db=db, config=config, job_queue=job_queue)

    app, yookassa_processor = create_app(config=config, dp=dp, bot=bot, db=db)

    return ApplicationState(
        config=config,
        bot=bot,
        dp=dp,
        db=db,
        job_queue=job_queue,
        app=app,
        yookassa_processor=yookassa_processor,
    )


async def _startup(state: ApplicationState, *, mode: str) -> None:
    log.info("Starting services mode=%s", mode)

    if mode == "polling":
        # [DELETE_WEBHOOK_ON_START]
        await state.bot.delete_webhook(drop_pending_updates=True)
        log.info("Webhook removed before polling starts")
    elif mode == "webhook":
        # [SET_WEBHOOK_ON_START]
        await state.bot.set_webhook(
            CFG.WEBHOOK_URL,
            secret_token=CFG.TELEGRAM_SECRET_TOKEN,
            drop_pending_updates=True,
            allowed_updates=ALLOWED_UPDATES,
        )
        log.info("Webhook configured at %s", CFG.WEBHOOK_URL)

    await state.db.init()
    await run_startup_healthcheck(config=state.config, mode=mode)

    async def _fetch_profile(user_id: int) -> Optional[Dict[str, Optional[str]]]:
        try:
            chat = await state.bot.get_chat(user_id)
        except Exception:  # pragma: no cover - Telegram API interaction
            log.debug("Failed to fetch profile for backfill user_id=%s", user_id, exc_info=True)
            return None
        return {
            "username": chat.username,
            "first_name": chat.first_name,
            "last_name": chat.last_name,
        }

    state.background_tasks.append(
        asyncio.create_task(state.db.backfill_user_profiles(_fetch_profile))
    )
    try:
        migrated = await state.db.migrate_credit_balances(state.config.generation_cost_credits)
        if migrated:
            log.info("Migrated %s legacy user balances", migrated)
    except Exception:  # pragma: no cover - defensive migration guard
        log.exception("Failed to migrate legacy balances to credit economy")
    await state.job_queue.start()

    if state.yookassa_processor:
        try:
            yookassa_client.init(state.config)
        except RuntimeError:
            log.warning("YooKassa is not fully configured; skipping card payments")
        else:
            state.background_tasks.append(
                asyncio.create_task(
                    poll_pending_payments(
                        state.yookassa_processor, state.config.yookassa_poll_interval
                    )
                )
            )

    log.info("Startup sequence completed mode=%s", mode)


async def _shutdown(state: ApplicationState, *, mode: str) -> None:
    log.info("Shutting down services mode=%s", mode)

    for task in state.background_tasks:
        task.cancel()
    if state.background_tasks:
        await asyncio.gather(*state.background_tasks, return_exceptions=True)
    state.background_tasks.clear()

    await state.job_queue.stop()

    try:
        await state.bot.delete_webhook(drop_pending_updates=False)
    except Exception:  # pragma: no cover - defensive cleanup
        log.exception("Failed to delete webhook on shutdown")

    await state.bot.session.close()

    await state.db.close()

    if state.uvicorn_server:
        state.uvicorn_server.should_exit = True
    if state.uvicorn_task:
        await state.uvicorn_task
        state.uvicorn_task = None

    log.info("Shutdown sequence completed mode=%s", mode)


# [RUN_POLLING]
def _run_polling(state: ApplicationState, loop: asyncio.AbstractEventLoop) -> None:
    log.info("Launching polling mode")

    # [UVICORN_RUN]
    uvicorn_config = uvicorn.Config(
        state.app, host=CFG.HOST, port=CFG.PORT, log_level="info"
    )
    server = uvicorn.Server(uvicorn_config)
    state.uvicorn_server = server
    state.uvicorn_task = loop.create_task(server.serve())

    async def on_startup(_: Dispatcher) -> None:
        await _startup(state, mode="polling")

    async def on_shutdown(_: Dispatcher) -> None:
        await _shutdown(state, mode="polling")

    executor.start_polling(
        state.dp,
        skip_updates=True,
        allowed_updates=ALLOWED_UPDATES,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
    )


# [RUN_WEBHOOK]
async def _run_webhook(state: ApplicationState) -> None:
    log.info("Launching webhook mode")

    # [UVICORN_RUN]
    uvicorn_config = uvicorn.Config(
        state.app, host=CFG.HOST, port=CFG.PORT, log_level="info"
    )
    server = uvicorn.Server(uvicorn_config)
    state.uvicorn_server = server

    try:
        await _startup(state, mode="webhook")
        await server.serve()
    finally:
        await _shutdown(state, mode="webhook")


# [ENTRYPOINT]
def main() -> None:
    args = _parse_args()
    config = load_config()
    state = _init_application(config)

    mode = (args.mode or CFG.BOT_MODE).lower()
    log.info("Selected run mode=%s", mode)

    loop = asyncio.get_event_loop()
    if mode == "polling":
        _run_polling(state, loop)
    else:
        loop.run_until_complete(_run_webhook(state))


if __name__ == "__main__":
    main()
