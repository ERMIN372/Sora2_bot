"""Application configuration utilities for the Sora Telegram bot."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


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

    @property
    def aiogram_redis_url(self) -> Optional[str]:
        """Optional Redis URL for rate limiting or FSM storage."""

        return os.getenv("AIROGRAM_REDIS_URL")


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
    )


__all__ = ["Config", "load_config"]
