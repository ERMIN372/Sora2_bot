"""Application configuration utilities for the Sora Telegram bot."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, Decimal
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing constants
# ---------------------------------------------------------------------------

DEFAULT_CREDIT_PRICE_RUB = Decimal("25.8")
DEFAULT_MARKUP_PCT = Decimal("30")
DEFAULT_FIX_FEE_RUB = Decimal("0")
_DEFAULT_PROVIDER_COSTS: Mapping[str, Decimal] = {
    "gemini_image": Decimal("18.0"),
}

DEFAULT_PRODUCTS: Dict[str, Decimal] = {
    "sora_video": Decimal("5"),
}

@dataclass(frozen=True)
class CreditPackage:
    """A bundle of credits sold for a fixed price in rubles."""

    package_id: str
    credits: int
    price_rub: Decimal
    discount_pct: Decimal = Decimal("0")

    def __post_init__(self) -> None:  # pragma: no cover - dataclass validation
        if self.credits <= 0:
            raise ValueError("credits must be positive")
        if not self.package_id:
            raise ValueError("package_id must be set")

    @property
    def credits_int(self) -> int:
        """Return the credit amount as an integer."""

        return int(self.credits)

    @property
    def price_kopeks(self) -> int:
        """Return the rounded package price in kopeks."""

        value = (self.price_rub * Decimal(100)).quantize(Decimal("1"))
        return int(value)

    @property
    def price_rubles(self) -> str:
        value = self.price_rub.quantize(Decimal("0.01"))
        return f"{value:.2f}".replace(".", ",")


_DEFAULT_PACKAGE_LAYOUT: Tuple[Tuple[str, int], ...] = (
    ("c5", 5),
    ("c10", 10),
    ("c25", 25),
    ("c50", 50),
    ("c150", 150),
)


def _build_packages(
    credit_price_rub: Decimal, discounts: Optional[Mapping[str, Decimal]] = None
) -> Tuple[CreditPackage, ...]:
    discounts = discounts or {}
    packages: list[CreditPackage] = []
    for package_id, credits in _DEFAULT_PACKAGE_LAYOUT:
        discount = discounts.get(package_id, Decimal("0"))
        base_price = credit_price_rub * Decimal(credits)
        price = base_price * (Decimal("100") - discount) / Decimal("100")
        packages.append(
            CreditPackage(
                package_id,
                credits,
                price.quantize(Decimal("0.01")),
                discount_pct=discount,
            )
        )
    return tuple(packages)


DEFAULT_CREDIT_PACKAGES: Tuple[CreditPackage, ...] = _build_packages(
    DEFAULT_CREDIT_PRICE_RUB
)


@dataclass(frozen=True)
class PricingConfig:
    """Runtime pricing parameters loaded from the environment."""

    credit_price_rub: Decimal
    markup_pct: Decimal
    fix_fee_rub: Decimal
    provider_costs: Mapping[str, Decimal]

    def cost_for(self, key: str, default: Optional[Decimal] = None) -> Optional[Decimal]:
        value = self.provider_costs.get(key)
        if value is not None:
            return value
        if default is not None:
            return default
        return _DEFAULT_PROVIDER_COSTS.get(key)


def _load_provider_costs(prefixes: Iterable[str]) -> Dict[str, Decimal]:
    costs: Dict[str, Decimal] = {}
    for env_key, value in os.environ.items():
        for prefix in prefixes:
            if env_key.startswith(prefix):
                key = env_key[len(prefix) :].lower()
                try:
                    costs[key] = Decimal(value)
                except Exception as exc:  # pragma: no cover - config guard
                    raise RuntimeError(
                        f"Environment variable {env_key!r} must be a decimal number"
                    ) from exc
                break
    return costs


@dataclass(frozen=True)
class Config:
    """Configuration values loaded from the environment."""

    bot_token: str
    sora_enabled: bool = False
    sora_api_key: str = ""
    sora_api_base: str = "https://api.sora.ai/v1"
    sora_model: str = "sora-2"
    gemini_api_key: str = ""
    gemini_model_text: str = "gemini-2.0-flash"
    gemini_model_image: str = "gemini-2.5-flash-image"
    default_video_model: str = "sora-2"
    database_path: str = "./bot.db"
    jobs_concurrency: int = 2
    max_jobs_per_user: int = 3
    request_timeout: float = 20.0
    request_connect_timeout: float = 10.0
    request_read_timeout: float = 20.0
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
    support_chat_id: Optional[int] = None
    archive_channel_id: Optional[int] = None
    yookassa_poll_interval: int = 60
    google_sheet_id: str = ""
    gs_users_sheet: str = "users"
    gs_payments_sheet: str = "payments"
    gs_jobs_sheet: str = "jobs"
    gs_errors_sheet: str = "errors"
    pricing: PricingConfig = field(
        default_factory=lambda: PricingConfig(
            credit_price_rub=DEFAULT_CREDIT_PRICE_RUB,
            markup_pct=DEFAULT_MARKUP_PCT,
            fix_fee_rub=DEFAULT_FIX_FEE_RUB,
            provider_costs=dict(_DEFAULT_PROVIDER_COSTS),
        )
    )
    products: Mapping[str, Decimal] = field(default_factory=lambda: DEFAULT_PRODUCTS.copy())
    credit_packages: Tuple[CreditPackage, ...] = DEFAULT_CREDIT_PACKAGES
    package_discounts: Mapping[str, Decimal] = field(default_factory=dict)
    payments_read_only: bool = False
    terms_url: str = "https://telegra.ph/Oferta-10-15-3"
    admin_ids: Tuple[int, ...] = ()
    bot_version: str = "dev"
    support_notify_interval: int = 600

    @property
    def yookassa_enabled(self) -> bool:
        """Return ``True`` if YooKassa credentials are configured."""

        return bool(self.yookassa_shop_id and self.yookassa_secret_key)

    @property
    def yookassa_ready(self) -> bool:
        """Return ``True`` if YooKassa payments can be offered to users."""

        return self.yookassa_enabled and bool(self.public_base_url)

    @property
    def gemini_enabled(self) -> bool:
        """Return ``True`` if Gemini API key is configured."""

        return bool(self.gemini_api_key)

    @property
    def allowed_package_ids(self) -> Tuple[str, ...]:
        """Return the tuple of permitted package identifiers."""

        return tuple(package.package_id for package in self.credit_packages)

    def get_credit_package(self, package_id: str) -> Optional[CreditPackage]:
        """Return the package matching *package_id*, if any."""

        for package in self.credit_packages:
            if package.package_id == package_id:
                return package
        return None

    @property
    def credit_price_rub(self) -> Decimal:
        return self.pricing.credit_price_rub

    @property
    def credit_price_kopeks(self) -> int:
        return int((self.pricing.credit_price_rub * Decimal(100)).quantize(Decimal("1")))

    def credits_for_rubles(self, value_rub: Decimal) -> int:
        """Convert *value_rub* to credits using ceil rounding (min 1)."""

        if value_rub <= 0:
            return 1
        credits = (value_rub / self.pricing.credit_price_rub).quantize(
            Decimal("1"), rounding=ROUND_CEILING
        )
        return max(1, int(credits))

    def rubles_for_credits(self, credits: int) -> Decimal:
        return (self.pricing.credit_price_rub * Decimal(credits)).quantize(Decimal("0.01"))

    def format_rubles(self, value: Decimal) -> str:
        amount = Decimal(value).quantize(Decimal("0.01"))
        formatted = f"{amount:,.2f}".replace(",", " ").replace(".", ",")
        return f"{formatted}\u00A0₽"

    def format_price_tag(self, credits: int) -> str:
        rub_str = self.format_rubles(self.rubles_for_credits(credits))
        return f"{credits} кредитов (~{rub_str})"

    def get_product_credits(self, key: str = "sora_video") -> int:
        value = self.products.get(key)
        if value is None:
            return 0
        return int(Decimal(value))

    @property
    def generation_cost_credits(self) -> int:
        return self.get_product_credits("sora_video")

    def generation_cost_approx_rubles(self, key: str = "sora_video") -> Decimal:
        credits = self.get_product_credits(key)
        if credits <= 0:
            return Decimal("0")
        return self.rubles_for_credits(credits)

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


def _get_env_decimal(key: str, default: Decimal) -> Decimal:
    value = os.getenv(key)
    if value is None or not value.strip():
        return Decimal(default)
    try:
        return Decimal(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Environment variable {key!r} must be a decimal number") from exc


def _get_env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - configuration guard
        raise RuntimeError("Expected integer value") from exc


def _get_env_int_list(key: str) -> Tuple[int, ...]:
    raw = os.getenv(key, "")
    if not raw:
        return ()
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    result: list[int] = []
    for item in parts:
        try:
            result.append(int(item))
        except ValueError as exc:  # pragma: no cover - configuration guard
            raise RuntimeError(f"Environment variable {key!r} must contain integers") from exc
    return tuple(result)


def _parse_package_discounts(raw: Optional[str]) -> Dict[str, Decimal]:
    if not raw:
        return {}
    items = {}
    for part in raw.split(","):
        if not part:
            continue
        key, _, value = part.partition(":")
        key = key.strip()
        if not key or not value:
            continue
        try:
            items[key] = Decimal(value)
        except Exception as exc:  # pragma: no cover - config guard
            raise RuntimeError(
                f"Invalid discount value for package {key!r}: {value!r}"
            ) from exc
    return items


def load_config() -> Config:
    """Load configuration from the process environment."""

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
    if not bot_token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN environment variable is required (BOT_TOKEN is accepted for backwards compatibility)"
        )

    sora_enabled = _get_env_bool("SORA_ENABLED", Config.sora_enabled)
    gemini_api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    gemini_model_text = (
        os.getenv("GEMINI_MODEL_TEXT", Config.gemini_model_text).strip()
        or Config.gemini_model_text
    )
    gemini_model_image = (
        os.getenv("GEMINI_MODEL_IMAGE", Config.gemini_model_image).strip()
        or Config.gemini_model_image
    )
    gemini_enabled = bool(gemini_api_key)

    if not gemini_enabled and not sora_enabled:
        log.warning(
            "No generation providers are configured; video and image generation will be disabled"
        )

    sora_api_key = (os.getenv("SORA_API_KEY") or "").strip()
    sora_api_base = os.getenv("SORA_API_BASE", Config.sora_api_base).strip() or Config.sora_api_base
    raw_sora_model = os.getenv("SORA_MODEL", Config.sora_model)
    sora_model = (raw_sora_model or Config.sora_model).strip() or Config.sora_model
    normalized_sora = sora_model.replace("_", "-").lower()
    if normalized_sora in {"sora", "sora-2", "sora2"}:
        sora_model = "sora-2"

    if sora_enabled and not sora_api_key:
        raise RuntimeError("SORA_API_KEY environment variable is required when SORA_ENABLED=true")

    def _normalize_default_model(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        normalized = value.replace("_", "-").strip().lower()
        if normalized in {"sora", "sora2", "sora-2"}:
            return sora_model
        return normalized or None

    default_video_model = _normalize_default_model(os.getenv("DEFAULT_VIDEO_MODEL"))
    if not default_video_model:
        default_video_model = sora_model if sora_enabled else ""

    available_defaults: List[str] = []
    if sora_enabled:
        available_defaults.append(sora_model)

    if default_video_model not in available_defaults and available_defaults:
        default_video_model = available_defaults[0]
    elif not available_defaults and not default_video_model:
        default_video_model = ""
    database_path = os.getenv("DATABASE_PATH", Config.database_path)
    google_sheet_id = os.getenv("GOOGLE_SHEET_ID")
    if not google_sheet_id:
        raise RuntimeError("GOOGLE_SHEET_ID environment variable is required")
    if not os.getenv("GOOGLE_SA_JSON_BASE64"):
        raise RuntimeError("GOOGLE_SA_JSON_BASE64 environment variable is required")
    users_sheet = os.getenv("GS_USERS_SHEET", Config.gs_users_sheet)
    payments_sheet = os.getenv("GS_PAYMENTS_SHEET", Config.gs_payments_sheet)
    jobs_sheet = os.getenv("GS_JOBS_SHEET", Config.gs_jobs_sheet)
    errors_sheet = os.getenv("GS_ERRORS_SHEET", Config.gs_errors_sheet)

    prefixes = [
        "GEMINI_COST_RUB_",
        "SORA2_COST_RUB_",
        "SORA_COST_RUB_",
    ]
    pricing = PricingConfig(
        credit_price_rub=_get_env_decimal("CREDIT_PRICE_RUB", DEFAULT_CREDIT_PRICE_RUB),
        markup_pct=_get_env_decimal("MARKUP_PCT", DEFAULT_MARKUP_PCT),
        fix_fee_rub=_get_env_decimal("FIX_FEE_RUB", DEFAULT_FIX_FEE_RUB),
        provider_costs=_load_provider_costs(prefixes),
    )
    package_discounts = _parse_package_discounts(os.getenv("CREDIT_PACKAGE_DISCOUNTS"))

    return Config(
        bot_token=bot_token,
        sora_enabled=sora_enabled,
        sora_api_key=sora_api_key,
        sora_api_base=sora_api_base,
        sora_model=sora_model,
        gemini_api_key=gemini_api_key,
        gemini_model_text=gemini_model_text,
        gemini_model_image=gemini_model_image,
        default_video_model=default_video_model,
        database_path=database_path,
        google_sheet_id=google_sheet_id,
        gs_users_sheet=users_sheet,
        gs_payments_sheet=payments_sheet,
        gs_jobs_sheet=jobs_sheet,
        gs_errors_sheet=errors_sheet,
        pricing=pricing,
        credit_packages=_build_packages(pricing.credit_price_rub, package_discounts),
        package_discounts=package_discounts,
        jobs_concurrency=_get_env_int("JOBS_CONCURRENCY", Config.jobs_concurrency),
        max_jobs_per_user=_get_env_int("MAX_JOBS_PER_USER", Config.max_jobs_per_user),
        request_timeout=_get_env_float("REQUEST_TIMEOUT", Config.request_timeout),
        request_connect_timeout=_get_env_float(
            "REQUEST_CONNECT_TIMEOUT", Config.request_connect_timeout
        ),
        request_read_timeout=_get_env_float(
            "REQUEST_READ_TIMEOUT", Config.request_read_timeout
        ),
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
        support_chat_id=_get_optional_int(os.getenv("SUPPORT_CHAT_ID")),
        archive_channel_id=_get_optional_int(os.getenv("ARCHIVE_CHANNEL_ID")),
        yookassa_poll_interval=_get_env_int("YOOKASSA_POLL_INTERVAL", Config.yookassa_poll_interval),
        payments_read_only=_get_env_bool("PAYMENTS_READ_ONLY", Config.payments_read_only),
        terms_url=os.getenv("TERMS_URL", Config.terms_url),
        admin_ids=_get_env_int_list("ADMIN_IDS"),
        bot_version=os.getenv("BOT_VERSION", Config.bot_version),
        support_notify_interval=_get_env_int(
            "SUPPORT_NOTIFY_INTERVAL", Config.support_notify_interval
        ),
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
    "CreditPackage",
    "PricingConfig",
    "Config",
    "RuntimeConfig",
    "CFG",
    "load_config",
]
