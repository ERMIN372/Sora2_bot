"""Minimal ASGI entrypoint for Uvicorn/Railway."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from aiogram import Bot, Dispatcher, executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from fastapi import FastAPI
import uvicorn
import redis.asyncio as redis_asyncio

from app_server import YooKassaProcessor, create_app, poll_pending_payments
from archive import ArchivePublisher
from config import CFG, Config, load_config
from db import DatabaseInterface, create_database
from generation_gate import GenerationRequestGate
from handlers import register_handlers
from jobs import JobQueue
from healthcheck import run_startup_healthcheck
from middleware.admin_check import AdminCheckMiddleware
from middleware.rate_limit import RateLimitMiddleware
from providers import (
    BaseProviderClient,
    GeminiImageClient,
    KlingVideoClient,
    OpenAIChatClient,
    OpenAIImageClient,
    OpenAIVideoClient,
    SoraVideoClient,
    VeoVideoClient,
)
from services.gemini_key import ensure_gemini_key_logged
from services.error_reporter import ErrorReporter
import yookassa_client

_LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, None)
_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s"
_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"

if not isinstance(_LOG_LEVEL, int):
    _LOG_LEVEL = logging.INFO


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.funcName:
            payload["function"] = record.funcName
        return json.dumps(payload, ensure_ascii=False)


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    if os.getenv("LOG_JSON", "true").lower() in {"1", "true", "yes"}:
        formatter: logging.Formatter = JsonFormatter(datefmt=_LOG_DATEFMT)
    else:
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(_LOG_LEVEL)
    logging.captureWarnings(True)
    logging.getLogger("aiogram").setLevel(logging.DEBUG)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)


_configure_logging()
log = logging.getLogger(__name__)

ALLOWED_UPDATES: List[str] = ["message", "callback_query"]

if not isinstance(getattr(logging, _LOG_LEVEL_NAME, None), int):
    log.warning("Unknown LOG_LEVEL %s, defaulting to INFO", _LOG_LEVEL_NAME)
else:
    log.debug("Logging configured level=%s", logging.getLevelName(_LOG_LEVEL))

app: FastAPI | None = None
_ASGI_STATE: "ApplicationState | None" = None
_SENTRY_INITIALIZED = False


def _init_sentry(config: Config) -> None:
    global _SENTRY_INITIALIZED
    if _SENTRY_INITIALIZED:
        return
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        log.info("Sentry disabled: no SENTRY_DSN provided")
        return
    try:
        import sentry_sdk
    except Exception:
        log.warning("SENTRY_DSN is set but sentry-sdk is not available")
        return

    traces_sample_rate = 0.0
    try:
        traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0"))
    except ValueError:
        traces_sample_rate = 0.0
    sentry_sdk.init(
        dsn=dsn,
        environment=config.environment or "dev",
        traces_sample_rate=traces_sample_rate,
    )
    _SENTRY_INITIALIZED = True
    log.info("Sentry initialised")


@dataclass
class ApplicationState:
    config: Config
    bot: Bot
    dp: Dispatcher
    db: DatabaseInterface
    job_queue: JobQueue
    gate: GenerationRequestGate
    app: FastAPI
    yookassa_processor: Optional[YooKassaProcessor]
    background_tasks: List[asyncio.Task[Any]] = field(default_factory=list)
    uvicorn_task: Optional[asyncio.Task[Any]] = None
    uvicorn_server: Optional[uvicorn.Server] = None
    archive_publisher: Optional[ArchivePublisher] = None
    error_reporter: Optional[ErrorReporter] = None
    openai_chat_client: Optional[OpenAIChatClient] = None
    redis: Optional[redis_asyncio.Redis] = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video generation bot runner")
    parser.add_argument(
        "--mode",
        choices=("polling", "webhook"),
        help="Override BOT_MODE from the environment",
    )
    return parser.parse_args()


def _init_application(config: Config) -> ApplicationState:
    bot = Bot(token=config.bot_token, parse_mode="HTML")
    dp = Dispatcher(bot, storage=MemoryStorage())
    redis_client: Optional[redis_asyncio.Redis] = None
    db = create_database(config)
    gate = GenerationRequestGate()
    error_reporter = ErrorReporter(bot=bot, config=config)
    providers: Dict[str, BaseProviderClient] = {}
    chat_client: Optional[OpenAIChatClient] = None
    openai_video_client: Optional[OpenAIVideoClient] = None

    if config.gemini_enabled:
        gemini_client = GeminiImageClient(config=config)
        providers.update(
            {
                "gemini-image": gemini_client,
                "gemini": gemini_client,
                config.gemini_model_image: gemini_client,
            }
        )
    if config.gemini_video_enabled:
        veo_client = VeoVideoClient(config=config)
        providers.update(
            {
                "veo": veo_client,
                "veo31": veo_client,
                "gemini-video": veo_client,
                config.gemini_model_video: veo_client,
            }
        )
    if config.sora_video_enabled:
        sora_client = SoraVideoClient(config=config)
        providers.update(
            {
                "sora": sora_client,
                "openai-video": sora_client,
            }
        )
        if config.sora_model_video:
            providers.setdefault(config.sora_model_video, sora_client)
        try:
            openai_video_client = OpenAIVideoClient(config=config)
        except RuntimeError:
            log.warning(
                "OpenAI Videos API client is not configured; continuing without it",
                exc_info=True,
            )
        else:
            providers["openai-video-api"] = openai_video_client
            for model_name in openai_video_client.supported_models:
                providers[model_name] = openai_video_client
            configured_model = (config.sora_model_video or "").strip()
            if configured_model and configured_model not in providers:
                providers[configured_model] = openai_video_client
    if config.kling_video_enabled:
        try:
            kling_client = KlingVideoClient(config=config)
        except RuntimeError:
            log.warning(
                "Kling AI client is not configured; skipping",
                exc_info=True,
            )
        else:
            providers.update({
                "kling": kling_client,
                "kling_mc": kling_client,
                "kling-video": kling_client,
                "kling-v2-6-motion": kling_client,
            })
            log.info("Kling AI Motion Control provider registered")
    if config.openai_image_enabled:
        try:
            openai_image_client = OpenAIImageClient(config=config)
        except RuntimeError:
            log.warning(
                "OpenAI image fallback is misconfigured; skipping",
                exc_info=True,
            )
        else:
            entries: Dict[str, BaseProviderClient] = {
                "openai-image": openai_image_client
            }
            for model_name in config.fallback_image_models:
                if not isinstance(model_name, str):
                    continue
                key = model_name.strip()
                if not key or key in entries:
                    continue
                entries[key] = openai_image_client
            for key, client in entries.items():
                if key in providers:
                    continue
                providers[key] = client
            log.info("OpenAI image fallback enabled models=%s", ",".join(entries.keys()))
    if config.sora_enabled:
        try:
            chat_client = OpenAIChatClient(config=config)
        except RuntimeError:
            log.warning(
                "OpenAI chat client is not configured; skipping ChatGPT menu",
                exc_info=True,
            )
        else:
            log.info("OpenAI ChatGPT client initialised model=gpt-4o")

    default_provider = config.default_video_model
    if default_provider not in providers:
        if providers:
            default_provider = next(iter(providers))
        else:
            default_provider = ""
    archive_publisher = ArchivePublisher(bot=bot, config=config, db=db)
    job_queue = JobQueue(
        db=db,
        providers=providers,
        default_provider=default_provider,
        config=config,
        gate=gate,
        error_reporter=error_reporter,
    )
    register_handlers(
        dp,
        db=db,
        config=config,
        job_queue=job_queue,
        gate=gate,
        archive_publisher=archive_publisher,
        error_reporter=error_reporter,
        chatgpt_client=chat_client,
        openai_video_client=openai_video_client,
    )

    dp.middleware.setup(AdminCheckMiddleware(config.admin_ids))
    redis_url = config.aiogram_redis_url
    if redis_url:
        redis_client = redis_asyncio.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        dp.middleware.setup(
            RateLimitMiddleware(
                redis_client,
                limits=config.command_rate_limits,
            )
        )

    local_app, yookassa_processor = create_app(config=config, dp=dp, bot=bot, db=db)

    return ApplicationState(
        config=config,
        bot=bot,
        dp=dp,
        db=db,
        job_queue=job_queue,
        gate=gate,
        app=local_app,
        yookassa_processor=yookassa_processor,
        archive_publisher=archive_publisher,
        error_reporter=error_reporter,
        openai_chat_client=chat_client,
        redis=redis_client,
    )


async def _startup(state: ApplicationState, *, mode: str) -> None:
    log.info("Starting services mode=%s", mode)

    if mode == "polling":
        await state.bot.delete_webhook(drop_pending_updates=True)
        log.info("Webhook removed before polling starts")

    await state.db.init()
    try:
        await state.db.describe()
    except Exception:  # pragma: no cover - defensive
        log.warning("Failed to describe database connection", exc_info=True)
    await run_startup_healthcheck(config=state.config, mode=mode)

    async def _fetch_profile(user_id: int) -> Optional[Dict[str, Optional[str]]]:
        try:
            chat = await state.bot.get_chat(user_id)
        except Exception:  # pragma: no cover - Telegram API interaction
            log.debug(
                "Failed to fetch profile for backfill user_id=%s",
                user_id,
                exc_info=True,
            )
            return None
        return {
            "username": chat.username,
            "first_name": chat.first_name,
            "last_name": chat.last_name,
        }

    state.background_tasks.append(asyncio.create_task(state.db.backfill_user_profiles(_fetch_profile)))
    if state.config.credit_migration_enabled:
        try:
            migrated = await state.db.migrate_credit_balances(state.config.generation_cost_credits)
            if migrated:
                log.info("Migrated %s legacy user balances", migrated)
        except Exception:  # pragma: no cover - defensive migration guard
            log.exception("Failed to migrate legacy balances to credit economy")
    else:
        log.info("Credit migration disabled on startup")
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
                        state.yookassa_processor,
                        state.config.yookassa_poll_interval,
                    )
                )
            )

    log.info("Startup sequence completed mode=%s", mode)


def _is_valid_webhook_host(host: str) -> bool:
    if not host:
        return False
    parsed = urlparse(host if "://" in host else f"https://{host}")
    hostname = parsed.hostname or ""
    return bool(hostname and "." in hostname)


async def _configure_webhook(state: ApplicationState) -> bool:
    webhook_task = asyncio.create_task(
        state.bot.set_webhook(
            CFG.WEBHOOK_URL,
            secret_token=CFG.TG_WEBHOOK_SECRET or None,
            drop_pending_updates=True,
            allowed_updates=ALLOWED_UPDATES,
        )
    )
    try:
        await webhook_task
    except Exception:
        log.exception("Failed to configure Telegram webhook at %s", CFG.WEBHOOK_URL)
        return False

    log.info("webhook_set url=%s", CFG.WEBHOOK_URL)
    return True


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

    if state.redis:
        try:
            close_fn = getattr(state.redis, "aclose", None)
            if callable(close_fn):
                await close_fn()
            else:
                await state.redis.close()
        except Exception:  # pragma: no cover - defensive cleanup
            log.warning("Failed to close Redis connection", exc_info=True)

    if state.uvicorn_server:
        state.uvicorn_server.should_exit = True
    if state.uvicorn_task:
        await state.uvicorn_task
        state.uvicorn_task = None

    log.info("Shutdown sequence completed mode=%s", mode)


def _run_polling(state: ApplicationState, loop: asyncio.AbstractEventLoop) -> None:
    log.info("Launching polling mode")

    host = os.getenv("HOST", CFG.HOST or "0.0.0.0") or "0.0.0.0"
    port = int(os.getenv("PORT", str(CFG.PORT)))
    uvicorn_config = uvicorn.Config(
        state.app,
        host=host,
        port=port,
        log_level="info",
        lifespan="on",
        loop="uvloop",
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


async def _run_webhook(state: ApplicationState) -> bool:
    log.info("Launching webhook mode")

    host = os.getenv("HOST", CFG.HOST or "0.0.0.0") or "0.0.0.0"
    port = int(os.getenv("PORT", str(CFG.PORT)))
    uvicorn_config = uvicorn.Config(
        state.app,
        host=host,
        port=port,
        log_level="info",
        lifespan="on",
        loop="uvloop",
    )
    server = uvicorn.Server(uvicorn_config)
    state.uvicorn_server = server

    try:
        if not _is_valid_webhook_host(CFG.WEBHOOK_HOST):
            log.error("WEBHOOK_HOST не задан или задан неверно; переключаюсь на polling")
            return False

        loop = asyncio.get_running_loop()
        state.uvicorn_task = loop.create_task(server.serve())

        await _startup(state, mode="webhook")

        started_event = getattr(server, "started", None)
        if isinstance(started_event, asyncio.Event):
            try:
                await asyncio.wait_for(started_event.wait(), timeout=30)
            except asyncio.TimeoutError:
                log.error("Application failed to open port %s in time", port)
                return False
        else:
            await asyncio.sleep(1.0)

        if not await _configure_webhook(state):
            return False

        await state.uvicorn_task
        return True
    finally:
        await _shutdown(state, mode="webhook")


def _build_asgi_state() -> ApplicationState:
    config = load_config()
    log.info(
        "Runtime startup environment=%s mode=%s db_log_jobs=%s archive_enabled=%s "
        "archive_channel_configured=%s postgres_enabled=%s",
        config.environment or "dev",
        CFG.BOT_MODE,
        getattr(config, "db_log_jobs", True),
        getattr(config, "archive_enabled", True),
        bool(config.archive_channel_id),
        config.postgres_enabled,
    )
    _init_sentry(config)
    ensure_gemini_key_logged(
        config,
        context="bot_main",
        process_id=f"pid={os.getpid()}",
        logger=log,
    )
    return _init_application(config)


if __name__ != "__main__":
    _ASGI_STATE = _build_asgi_state()
    app = _ASGI_STATE.app

    @app.on_event("startup")
    async def _asgi_startup() -> None:  # pragma: no cover - uvicorn lifecycle
        if _ASGI_STATE is None:
            return
        mode = (CFG.BOT_MODE or "webhook").lower()
        if mode == "polling":
            log.warning(
                "ASGI startup: BOT_MODE=polling is not supported; keeping webhook handlers active",
            )
            mode = "webhook"
        await _startup(_ASGI_STATE, mode=mode)

        if mode != "webhook":
            log.info("Webhook setup skipped because BOT_MODE=%s", mode)
            return

        if not _is_valid_webhook_host(CFG.WEBHOOK_HOST):
            log.warning(
                "WEBHOOK_HOST is empty or invalid; webhook setup skipped and bot will not receive updates"
            )
            return

        configured = await _configure_webhook(_ASGI_STATE)
        if not configured:
            log.warning("Webhook setup failed; bot may be unreachable via webhook")

    @app.on_event("shutdown")
    async def _asgi_shutdown() -> None:  # pragma: no cover - uvicorn lifecycle
        if _ASGI_STATE is None:
            return
        await _shutdown(_ASGI_STATE, mode=(CFG.BOT_MODE or "webhook").lower())


if __name__ == "__main__":
    def main() -> None:
        args = _parse_args()
        config = load_config()
        _init_sentry(config)
        ensure_gemini_key_logged(
            config,
            context="bot_main",
            process_id=f"pid={os.getpid()}",
            logger=log,
        )
        state = _init_application(config)
        global app
        app = state.app

        mode = (args.mode or CFG.BOT_MODE).lower()
        log.info("Selected run mode=%s", mode)

        if mode == "polling":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                _run_polling(state, loop)
            finally:
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()
        else:
            success = asyncio.run(_run_webhook(state))
            if not success:
                log.warning("Falling back to polling mode after webhook failure")
                state = _init_application(config)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    _run_polling(state, loop)
                finally:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()

    main()
