"""Application configuration utilities for the Sora Telegram bot."""
from __future__ import annotations

import base64
import binascii
import json
import logging
import os
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, Mapping, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

CREDIT_PRICE_KOPEKS = 2580
SORA_VIDEO_CREDITS = Decimal("5")


@dataclass(frozen=True)
class CreditPackage:
    """A bundle of credits sold for a fixed price in kopeks."""

    credits: Decimal
    price_kopeks: int

    @property
    def credits_int(self) -> int:
        """Return the credit amount as an integer for current economy version."""

        return int(self.credits)

    @property
    def price_rubles(self) -> int:
        """Return the rounded package price in rubles."""

        return self.price_kopeks // 100


DEFAULT_PRODUCTS: Dict[str, Decimal] = {
    "sora_video": SORA_VIDEO_CREDITS,
}

DEFAULT_CREDIT_PACKAGES: Tuple[CreditPackage, ...] = (
    CreditPackage(Decimal("5"), 12900),
    CreditPackage(Decimal("10"), 25000),
    CreditPackage(Decimal("25"), 61300),
    CreditPackage(Decimal("50"), 120000),
    CreditPackage(Decimal("150"), 356000),
)


@dataclass(frozen=True)
class Config:
    """Configuration values loaded from the environment."""

    bot_token: str
    sora_api_key: str
    sora_api_url: str = "https://api.sora.ai/v1"
    sora_model: str = "sora-2"
    database_path: str = "./bot.db"
    jobs_concurrency: int = 2
    max_jobs_per_user: int = 3
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
    subscription_chat_id: Optional[str] = None
    yookassa_poll_interval: int = 60
    google_sheet_id: str = ""
    google_service_account: Dict[str, object] = field(default_factory=dict)
    gs_users_sheet: str = "users"
    gs_payments_sheet: str = "payments"
    gs_jobs_sheet: str = "jobs"
    credit_price_kopeks: int = CREDIT_PRICE_KOPEKS
    products: Mapping[str, Decimal] = field(default_factory=lambda: DEFAULT_PRODUCTS.copy())
    credit_packages: Tuple[CreditPackage, ...] = DEFAULT_CREDIT_PACKAGES

    @property
    def yookassa_enabled(self) -> bool:
        """Return ``True`` if YooKassa credentials are configured."""

        return bool(self.yookassa_shop_id and self.yookassa_secret_key)

    @property
    def yookassa_ready(self) -> bool:
        """Return ``True`` if YooKassa payments can be offered to users."""

        return self.yookassa_enabled and bool(self.public_base_url)

    @property
    def generation_cost_credits(self) -> int:
        """Return the configured credit price for a single Sora video."""

        return int(self.products.get("sora_video", SORA_VIDEO_CREDITS))

    def credits_to_kopeks(self, credits: Decimal | int) -> int:
        """Convert *credits* to kopeks using the base credit price."""

        value = Decimal(credits)
        total = (value * Decimal(self.credit_price_kopeks)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return int(total)

    def generation_cost_approx_rubles(self) -> int:
        """Return the approximate ruble cost of one Sora video without discounts."""

        return self.credits_to_kopeks(self.products.get("sora_video", SORA_VIDEO_CREDITS)) // 100

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


def _load_service_account_info() -> Dict[str, object]:
    raw = os.getenv("GOOGLE_SA_JSON_BASE64")
    if not raw:
        raise RuntimeError("GOOGLE_SA_JSON_BASE64 environment variable is required")
    raw = raw.strip()
    try:
        decoded = base64.b64decode(raw, validate=True)
        text = decoded.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        text = raw
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "GOOGLE_SA_JSON_BASE64 must contain JSON or base64-encoded JSON"
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError("Service account JSON must decode to an object")
    return data


def load_config() -> Config:
    """Load configuration from the process environment."""

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
    if not bot_token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN environment variable is required (BOT_TOKEN is accepted for backwards compatibility)"
        )

    sora_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("SORA_API_KEY")
    if not sora_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is required (SORA_API_KEY is accepted for backwards compatibility)"
        )

    sora_api_url = os.getenv("SORA_API_URL", Config.sora_api_url)
    raw_model = os.getenv("SORA_MODEL", Config.sora_model)
    if (raw_model or "").strip().lower() != "sora-2":
        log.warning("Unsupported SORA_MODEL value %r; forcing 'sora-2'", raw_model)
    sora_model = Config.sora_model
    database_path = os.getenv("DATABASE_PATH", Config.database_path)
    google_sheet_id = os.getenv("GOOGLE_SHEET_ID")
    if not google_sheet_id:
        raise RuntimeError("GOOGLE_SHEET_ID environment variable is required")
    service_account = _load_service_account_info()
    users_sheet = os.getenv("GS_USERS_SHEET", Config.gs_users_sheet)
    payments_sheet = os.getenv("GS_PAYMENTS_SHEET", Config.gs_payments_sheet)
    jobs_sheet = os.getenv("GS_JOBS_SHEET", Config.gs_jobs_sheet)

    return Config(
        bot_token=bot_token,
        sora_api_key=sora_api_key,
        sora_api_url=sora_api_url,
        database_path=database_path,
        google_sheet_id=google_sheet_id,
        google_service_account=service_account,
        gs_users_sheet=users_sheet,
        gs_payments_sheet=payments_sheet,
        gs_jobs_sheet=jobs_sheet,
        sora_model=sora_model,
        jobs_concurrency=_get_env_int("JOBS_CONCURRENCY", Config.jobs_concurrency),
        max_jobs_per_user=_get_env_int("MAX_JOBS_PER_USER", Config.max_jobs_per_user),
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
        subscription_chat_id=os.getenv("SUBSCRIPTION_CHAT_ID"),
        yookassa_poll_interval=_get_env_int("YOOKASSA_POLL_INTERVAL", Config.yookassa_poll_interval),
    )


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration that controls the bot launch mode."""

    BOT_MODE: str
    WEBHOOK_HOST: str
    WEBHOOK_PATH: str
    WEBHOOK_URL: str
    TELEGRAM_SECRET_TOKEN: str
    HOST: str
    PORT: int


def _normalise_path(path: str) -> str:
    if not path.startswith("/"):
        path = f"/{path}"
    return path.rstrip("/") or "/"


def _load_runtime_config() -> RuntimeConfig:
    mode = (os.getenv("BOT_MODE", "polling") or "polling").strip().lower()
    if mode not in {"polling", "webhook"}:
        raise RuntimeError(
            "BOT_MODE must be one of 'polling' or 'webhook'"
        )

    webhook_host_raw = os.getenv("WEBHOOK_HOST", "").strip()
    webhook_host = webhook_host_raw.rstrip("/")
    webhook_path = _normalise_path(os.getenv("WEBHOOK_PATH", "/tg/webhook").strip() or "/tg/webhook")
    webhook_url = f"{webhook_host}{webhook_path}" if webhook_host else webhook_path
    secret_token = os.getenv("TELEGRAM_SECRET_TOKEN", "").strip()
    host = os.getenv("HOST", "0.0.0.0") or "0.0.0.0"
    port = _get_env_int("PORT", 8080)

    if mode == "webhook":
        if not webhook_host.startswith("https://"):
            raise RuntimeError("WEBHOOK_HOST must start with https:// when BOT_MODE=webhook")
        if not secret_token:
            raise RuntimeError("TELEGRAM_SECRET_TOKEN must be set when BOT_MODE=webhook")

    return RuntimeConfig(
        BOT_MODE=mode,
        WEBHOOK_HOST=webhook_host,
        WEBHOOK_PATH=webhook_path,
        WEBHOOK_URL=webhook_url,
        TELEGRAM_SECRET_TOKEN=secret_token,
        HOST=host,
        PORT=port,
    )


CFG = _load_runtime_config()


__all__ = [
    "CREDIT_PRICE_KOPEKS",
    "SORA_VIDEO_CREDITS",
    "CreditPackage",
    "Config",
    "RuntimeConfig",
    "CFG",
    "load_config",
]
