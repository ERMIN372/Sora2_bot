"""Application configuration utilities for the Sora Telegram bot."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    """Configuration values loaded from the environment.

    The bot relies on a single Telegram bot token and an API key for the Sora
    video generation service.  A SQLite database is used by default for local
    development, but the location can be customised via ``DATABASE_PATH``.
    """

    bot_token: str
    sora_api_key: str
    sora_api_url: str = "https://api.sora.ai/v1"
    database_path: str = "./bot.db"
    jobs_concurrency: int = 2
    max_jobs_per_user: int = 3
    credits_per_payment: int = 10
    credits_per_generation: int = 1
    request_timeout: float = 30.0
    request_retries: int = 3
    retry_backoff: float = 2.0
    yookassa_shop_id: Optional[str] = None
    yookassa_secret_key: Optional[str] = None
    yookassa_test_mode: bool = True
    public_base_url: str = ""
    yookassa_return_path: str = "/pay/return"
    yookassa_webhook_path: str = "/pay/webhook"
    yookassa_send_receipts: bool = False
    price_one_rub: int = 119
    price_five_rub: int = 595
    price_ten_rub: int = 1190
    price_thirty_rub: int = 3570
    subscription_chat_id: Optional[str] = None
    yookassa_poll_interval: int = 60
    yookassa_server_port: int = 8080

    @property
    def aiogram_redis_url(self) -> Optional[str]:
        """Optional Redis URL for rate limiting or FSM storage."""

        return os.getenv("AIROGRAM_REDIS_URL")

    @property
    def yookassa_enabled(self) -> bool:
        """Return ``True`` if YooKassa credentials are configured."""

        return bool(self.yookassa_shop_id and self.yookassa_secret_key)


def _get_env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Environment variable {key!r} must be an integer") from exc


def _get_env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Environment variable {key!r} must be a float") from exc


def _get_env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> Config:
    """Load configuration from the process environment."""

    bot_token = os.getenv("BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("BOT_TOKEN environment variable is required")

    sora_api_key = os.getenv("SORA_API_KEY")
    if not sora_api_key:
        raise RuntimeError("SORA_API_KEY environment variable is required")

    sora_api_url = os.getenv("SORA_API_URL", Config.sora_api_url)
    database_path = os.getenv("DATABASE_PATH", Config.database_path)

    return Config(
        bot_token=bot_token,
        sora_api_key=sora_api_key,
        sora_api_url=sora_api_url,
        database_path=database_path,
        jobs_concurrency=_get_env_int("JOBS_CONCURRENCY", Config.jobs_concurrency),
        max_jobs_per_user=_get_env_int("MAX_JOBS_PER_USER", Config.max_jobs_per_user),
        credits_per_payment=_get_env_int("CREDITS_PER_PAYMENT", Config.credits_per_payment),
        credits_per_generation=_get_env_int("CREDITS_PER_GENERATION", Config.credits_per_generation),
        request_timeout=_get_env_float("REQUEST_TIMEOUT", Config.request_timeout),
        request_retries=_get_env_int("REQUEST_RETRIES", Config.request_retries),
        retry_backoff=_get_env_float("RETRY_BACKOFF", Config.retry_backoff),
        yookassa_shop_id=os.getenv("YOOKASSA_SHOP_ID"),
        yookassa_secret_key=os.getenv("YOOKASSA_SECRET_KEY"),
        yookassa_test_mode=_get_env_bool("YOOKASSA_TEST_MODE", Config.yookassa_test_mode),
        public_base_url=os.getenv("PUBLIC_BASE_URL", Config.public_base_url),
        yookassa_return_path=os.getenv("YOOKASSA_RETURN_PATH", Config.yookassa_return_path),
        yookassa_webhook_path=os.getenv("YOOKASSA_WEBHOOK_PATH", Config.yookassa_webhook_path),
        yookassa_send_receipts=_get_env_bool("YOOKASSA_SEND_RECEIPTS", Config.yookassa_send_receipts),
        price_one_rub=_get_env_int("PRICE_ONE_RUB", Config.price_one_rub),
        price_five_rub=_get_env_int("PRICE_FIVE_RUB", Config.price_five_rub),
        price_ten_rub=_get_env_int("PRICE_TEN_RUB", Config.price_ten_rub),
        price_thirty_rub=_get_env_int("PRICE_THIRTY_RUB", Config.price_thirty_rub),
        subscription_chat_id=os.getenv("SUBSCRIPTION_CHAT_ID"),
        yookassa_poll_interval=_get_env_int("YOOKASSA_POLL_INTERVAL", Config.yookassa_poll_interval),
        yookassa_server_port=_get_env_int("YOOKASSA_SERVER_PORT", Config.yookassa_server_port),
    )


__all__ = ["Config", "load_config"]
