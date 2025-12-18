"""Application configuration utilities for the video generation bot."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Debug / tracing flags
# ---------------------------------------------------------------------------

DEBUG_GEMINI = bool(os.getenv("DEBUG_GEMINI", "").strip())
DEBUG_SORA = bool(os.getenv("DEBUG_SORA", "").strip())


def _normalise_openai_version(raw: str, default: str = "v1") -> str:
    candidate = (raw or default).strip().strip("/") or default
    lowered = candidate.lower()
    if lowered not in {"v1", "v1beta"}:
        log.warning(
            "Unsupported OpenAI API version %s; falling back to %s",
            candidate,
            default,
        )
        return default
    return lowered


OPENAI_API_VERSION = _normalise_openai_version(os.getenv("OPENAI_API_VERSION", "v1"))


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


GEMINI_TRACE = _env_flag("GEMINI_TRACE")
GEMINI_TRACE_SAVE_JSON = _env_flag("GEMINI_TRACE_SAVE_JSON")
GEMINI_TRACE_SAVE_B64 = _env_flag("GEMINI_TRACE_SAVE_B64")
GEMINI_TRACE_CURL = _env_flag("GEMINI_TRACE_CURL")

_TRACE_HEADERS_DEFAULT = (
    "x-generative-ai-finish-reason,x-generative-ai-output-status,x-goog-rai-filtered-reason,"
    "x-goog-image-response-status,x-goog-ai-response-code,x-request-id,date,server,content-type"
)
_trace_headers_raw = os.getenv("GEMINI_TRACE_HEADERS", _TRACE_HEADERS_DEFAULT)
if _trace_headers_raw.strip().lower() in {"1", "true", "all", "*"}:
    GEMINI_TRACE_HEADERS = ["*"]
else:
    GEMINI_TRACE_HEADERS = [
        header.strip()
        for header in _trace_headers_raw.split(",")
        if header.strip()
    ]

GEMINI_PREDICT_DISABLE_TTL_SEC = int(os.getenv("GEMINI_PREDICT_DISABLE_TTL_SEC", "1800"))
GEMINI_IMAGE_API_VERSION = os.getenv("GEMINI_IMAGE_API_VERSION", "v1beta").strip() or "v1beta"
GEMINI_IMAGE_DISABLE_PREDICT = _env_flag("GEMINI_IMAGE_DISABLE_PREDICT")
GEMINI_IMAGE_MAX_RETRIES = max(1, int(os.getenv("GEMINI_IMAGE_MAX_RETRIES", "4") or 4))
GEMINI_IMAGE_BACKOFF_BASE_MS = max(1, int(os.getenv("GEMINI_IMAGE_BACKOFF_BASE_MS", "600") or 600))
GEMINI_IMAGE_STRICT_INLINE_ONLY = _env_flag("GEMINI_IMAGE_STRICT_INLINE_ONLY")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SORA_SUPPORTED_MODELS = ("sora-2", "sora-2-pro")
SORA_DEFAULT_MODEL = SORA_SUPPORTED_MODELS[0]
OPENAI_API_BASE = "https://api.openai.com"
DEFAULT_OPENAI_BETA_HEADER = "video=1"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_secret(value: Optional[str], visible: int = 4) -> str:
    """Return a masked representation of *value* for safe logging."""

    if not value:
        return ""
    text = str(value)
    if len(text) <= visible:
        return "*" * len(text)
    return f"{text[:visible]}{'*' * (len(text) - visible)}"


def env_str(key: str, default: str = "", *, strip: bool = True) -> str:
    """Fetch a string environment variable with optional whitespace stripping."""

    value = os.getenv(key)
    if value is None:
        return default
    return value.strip() if strip else value


def env_float(key: str, default: float) -> float:
    """Fetch a floating-point environment variable with validation."""

    value = os.getenv(key)
    if value is None or not value.strip():
        return float(default)
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Environment variable {key!r} must be a float") from exc


# ---------------------------------------------------------------------------
# Pricing constants
# ---------------------------------------------------------------------------

CREDIT_COST = Decimal("1")
TAROT_READING_PRICE_CREDITS = 29

PRODUCT_PRICING: Dict[str, Decimal] = {
    "sora": Decimal("99"),
    "veo3": Decimal("89"),
    "image": Decimal("5"),
    "tarot": Decimal(TAROT_READING_PRICE_CREDITS),
}

PRODUCT_PRICE_ALIASES: Dict[str, str] = {
    "sora_video": "sora",
    "veo_video": "veo3",
    "gemini_video": "veo3",
    "veo": "veo3",
    "gemini-image": "image",
    "image_generation": "image",
}

# ---------------------------------------------------------------------------
# UI-only pricing labels
# ---------------------------------------------------------------------------

PRODUCT_PRICING_UI = {
    "sora": 99,
    "veo3": 89,
    "image": 5,
    "tarot": TAROT_READING_PRICE_CREDITS,
}

TOPUP_UI_PACKS = [
    {"amount": 100, "bonus_pct": 0.00},
    {"amount": 300, "bonus_pct": 0.05},
    {"amount": 700, "bonus_pct": 0.08},
    {"amount": 1500, "bonus_pct": 0.12},
    {"amount": 3000, "bonus_pct": 0.15},
]

UI_MODEL_LABELS = {
    # UI labels only, the actual API model identifiers remain unchanged
    "veo": "💚Gemini veo3",
    "veo-3.0-generate-001": "💚Gemini veo3",
    "sora": "☁️OpenAI Sora 2",
    "sora-2": "☁️OpenAI Sora 2",
    "sora-2-pro": "☁️OpenAI Sora 2",
}

TOP_UP_PACKAGES: Dict[str, Dict[str, Decimal]] = {
    "100": {"amount": Decimal("100"), "bonus": Decimal("0.00")},
    "300": {"amount": Decimal("300"), "bonus": Decimal("0.05")},
    "700": {"amount": Decimal("700"), "bonus": Decimal("0.08")},
    "1500": {"amount": Decimal("1500"), "bonus": Decimal("0.12")},
    "3000": {"amount": Decimal("3000"), "bonus": Decimal("0.15")},
}

DEFAULT_CREDIT_PRICE_RUB = CREDIT_COST
DEFAULT_MARKUP_PCT = Decimal("30")
DEFAULT_FIX_FEE_RUB = Decimal("0")
_DEFAULT_PROVIDER_COSTS: Mapping[str, Decimal] = {
    "gemini_image": Decimal("18.0"),
}

DEFAULT_PRODUCTS: Dict[str, Decimal] = {
    name: Decimal(value) for name, value in PRODUCT_PRICING.items()
}
for alias, target in PRODUCT_PRICE_ALIASES.items():
    DEFAULT_PRODUCTS[alias] = DEFAULT_PRODUCTS[target]


@dataclass(frozen=True)
class RateLimitRule:
    """Represents a rate limit rule for a command."""

    limit: int
    period: int


DEFAULT_COMMAND_RATE_LIMITS: Mapping[str, RateLimitRule] = {
    "video_create": RateLimitRule(limit=5, period=60),
    "video_remix": RateLimitRule(limit=5, period=60),
    "video_models": RateLimitRule(limit=10, period=60),
    "video_get": RateLimitRule(limit=3, period=60),
    "models": RateLimitRule(limit=10, period=60),
}

@dataclass(frozen=True)
class CreditPackage:
    """A bundle of credits sold for a fixed price in rubles with bonus support."""

    package_id: str
    credits: int
    price_rub: Decimal
    bonus_pct: Decimal = Decimal("0")

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

    @property
    def base_credits(self) -> int:
        """Return credits purchased excluding the bonus."""

        return int((self.price_rub / CREDIT_COST).to_integral_value(rounding=ROUND_HALF_UP))

    @property
    def bonus_credits(self) -> int:
        """Return the number of bonus credits included in the package."""

        return max(0, self.credits_int - self.base_credits)

    @property
    def bonus_percent(self) -> Decimal:
        """Return the bonus percentage scaled to 0-100."""

        return (self.bonus_pct * Decimal("100")).quantize(Decimal("0.01"))


def _build_credit_packages(
    packages: Mapping[str, Mapping[str, Decimal]]
) -> Tuple[CreditPackage, ...]:
    result: list[CreditPackage] = []
    for package_id, data in packages.items():
        amount = Decimal(data.get("amount") or package_id)
        bonus_pct = Decimal(data.get("bonus", Decimal("0")))
        base_credits = int(
            (amount / CREDIT_COST).to_integral_value(rounding=ROUND_HALF_UP)
        )
        bonus_credits = int(
            (amount * bonus_pct).to_integral_value(rounding=ROUND_HALF_UP)
        )
        total_credits = max(1, base_credits + bonus_credits)
        result.append(
            CreditPackage(
                package_id=package_id,
                credits=total_credits,
                price_rub=amount.quantize(Decimal("0.01")),
                bonus_pct=bonus_pct,
            )
        )
    return tuple(result)


DEFAULT_CREDIT_PACKAGES: Tuple[CreditPackage, ...] = _build_credit_packages(
    TOP_UP_PACKAGES
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
    environment: str = "dev"
    gemini_api_key: str = ""
    gemini_model_text: str = "gemini-2.0-flash"
    gemini_model_text_fallback: str = "gemini-1.5-flash"
    gemini_model_image: str = "gemini-2.5-flash-image"
    gemini_model_video: str = "veo-3.0-generate-001"
    gemini_api_mode: str = "developer"
    vertex_project_id: Optional[str] = None
    vertex_location: Optional[str] = None
    gemini_api_endpoint: Optional[str] = None
    gemini_safety_threshold: str = "BLOCK_NONE"
    fallback_image_models: List[str] = field(
        default_factory=lambda: ["dall-e-3", "sdxl"]
    )
    openai_api_key: str = ""
    sora_api_key: str = ""
    openai_org_id: Optional[str] = None
    # Значение "sora" больше не поддерживается и приведёт к ошибке "Model not found" в Sora API.
    sora_model_video: str = SORA_DEFAULT_MODEL
    openai_api_base: str = OPENAI_API_BASE
    openai_api_version_video: str = OPENAI_API_VERSION
    openai_beta_header: str = DEFAULT_OPENAI_BETA_HEADER
    sora_requests_per_minute: int = 60
    gemini_requests_per_minute: int = 120
    default_video_model: str = "veo-3.0-generate-001"
    database_path: str = "./bot.db"
    jobs_concurrency: int = 2
    max_jobs_per_user: int = 3
    request_timeout: float = 20.0
    request_connect_timeout: float = 10.0
    request_read_timeout: float = 20.0
    provider_timeout_s: float = 60.0
    request_retries: int = 3
    retry_backoff: float = 2.0
    command_rate_limits: Mapping[str, "RateLimitRule"] = field(
        default_factory=lambda: dict(DEFAULT_COMMAND_RATE_LIMITS)
    )
    yookassa_shop_id: Optional[str] = None
    yookassa_secret_key: Optional[str] = None
    yookassa_test_mode: bool = True
    public_base_url: str = ""
    yookassa_return_path: str = "/pay/return"
    yookassa_webhook_path: str = "/pay/webhook"
    yookassa_send_receipts: bool = False
    subscription_chat_id: Optional[str] = None
    support_chat_id: Optional[int] = None
    archive_channel_id: Optional[int | str] = None
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
    payments_read_only: bool = False
    terms_url: str = "https://telegra.ph/Oferta-10-15-3"
    admin_ids: Tuple[int, ...] = ()
    bot_version: str = "dev"
    support_notify_interval: int = 600
    veo_poll_interval_min_seconds: float = 4.0
    veo_poll_interval_max_seconds: float = 6.0
    veo_operation_timeout_seconds: float = 12 * 60.0
    veo_operation_idle_timeout_seconds: float = 120.0
    debug_gemini: bool = False
    debug_sora: bool = False

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
    def gemini_video_enabled(self) -> bool:
        """Return ``True`` if Gemini video generation can be used."""

        return self.gemini_enabled and bool(self.gemini_model_video)

    @property
    def openai_key(self) -> str:
        return (self.sora_api_key or self.openai_api_key or "").strip()

    @property
    def sora_enabled(self) -> bool:
        """Return ``True`` if Sora (OpenAI) API key is configured."""

        return bool(self.openai_key)

    @property
    def sora_video_enabled(self) -> bool:
        """Return ``True`` if Sora video generation can be used."""

        return self.sora_enabled and bool((self.sora_model_video or "").strip())

    @property
    def openai_image_enabled(self) -> bool:
        """Return ``True`` if OpenAI image generation can be used as fallback."""

        return self.sora_enabled and bool((self.openai_api_base or "").strip())

    @property
    def resolved_sora_video_model(self) -> str:
        """Return the default Sora model ensuring it is supported."""

        candidate_raw = (self.sora_model_video or "").strip()
        if not candidate_raw:
            return SORA_DEFAULT_MODEL
        lowered = candidate_raw.lower()
        if lowered in SORA_SUPPORTED_MODELS or lowered.startswith("sora-2"):
            return lowered
        return SORA_DEFAULT_MODEL

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
        return f"{rub_str} (спишем {credits} кредитов)"

    def get_product_credits(self, key: str = "sora") -> int:
        value = self.products.get(key)
        if value is None:
            alias = PRODUCT_PRICE_ALIASES.get(key)
            if alias:
                value = self.products.get(alias) or DEFAULT_PRODUCTS.get(alias)
            else:
                value = DEFAULT_PRODUCTS.get(key)
        if value is None:
            return 0
        return int(Decimal(value))

    @property
    def generation_cost_credits(self) -> int:
        return self.get_product_credits("sora")

    def generation_cost_approx_rubles(self, key: str = "sora") -> Decimal:
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


def _parse_rate_limits(raw: Optional[str]) -> Dict[str, RateLimitRule]:
    """Parse comma-separated rate limits from the environment."""

    if raw is None:
        return dict(DEFAULT_COMMAND_RATE_LIMITS)
    result: Dict[str, RateLimitRule] = {}
    for chunk in raw.split(","):
        if not chunk.strip():
            continue
        parts = [item.strip() for item in chunk.split(":")]
        if len(parts) != 3:
            log.warning(
                "Invalid RATE_LIMITS entry %s; expected command:limit:period", chunk
            )
            continue
        name, limit_raw, period_raw = parts
        try:
            limit = int(limit_raw)
            period = int(period_raw)
        except ValueError:
            log.warning("Invalid RATE_LIMITS numbers for command=%s", name)
            continue
        if limit <= 0 or period <= 0:
            log.warning("RATE_LIMITS must be positive values command=%s", name)
            continue
        result[name.lower()] = RateLimitRule(limit=limit, period=period)
    if not result:
        return dict(DEFAULT_COMMAND_RATE_LIMITS)
    return result


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


def _parse_chat_id(value: Optional[str], *, env_name: str) -> Optional[int | str]:
    if value is None:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.startswith("@"):
        return candidate
    try:
        return int(candidate)
    except ValueError:
        log.warning(
            "%s should be a numeric ID or @username, got %s", env_name, candidate
        )
        return None


def _parse_int_list(raw: str, *, key: str) -> Tuple[int, ...]:
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    result: list[int] = []
    for item in parts:
        try:
            result.append(int(item))
        except ValueError as exc:  # pragma: no cover - configuration guard
            raise RuntimeError(f"Environment variable {key!r} must contain integers") from exc
    return tuple(result)


def _get_env_int_list(key: str) -> Tuple[int, ...]:
    raw = os.getenv(key, "")
    if not raw:
        return ()
    return _parse_int_list(raw, key=key)


def _get_admin_ids_from_env() -> Tuple[int, ...]:
    admin_ids = _get_env_int_list("ADMIN_IDS")
    if admin_ids:
        return admin_ids
    admins_raw = os.getenv("ADMINS", "")
    if not admins_raw:
        return ()
    return _parse_int_list(admins_raw, key="ADMINS")


def load_config() -> Config:
    """Load configuration from the process environment."""

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
    if not bot_token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN environment variable is required (BOT_TOKEN is accepted for backwards compatibility)"
        )

    raw_gemini_api_key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or ""
    )
    if os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
        log.info(
            "GOOGLE_API_KEY detected; preferring it over GEMINI_API_KEY for Gemini API access"
        )
    gemini_api_key = raw_gemini_api_key.strip()
    gemini_text_override = os.getenv("GEMINI_TEXT_MODEL")
    if gemini_text_override is not None and gemini_text_override.strip():
        gemini_model_text = gemini_text_override.strip()
    else:
        gemini_model_text = (
            os.getenv("GEMINI_MODEL_TEXT", Config.gemini_model_text).strip()
            or Config.gemini_model_text
        )
    gemini_model_text_fallback = (
        os.getenv("GEMINI_TEXT_MODEL_FALLBACK", Config.gemini_model_text_fallback).strip()
        or Config.gemini_model_text_fallback
    )
    gemini_model_image = (
        os.getenv("GEMINI_MODEL_IMAGE", Config.gemini_model_image).strip()
        or Config.gemini_model_image
    )
    gemini_model_video = (
        os.getenv("GEMINI_MODEL_VIDEO", Config.gemini_model_video).strip()
        or Config.gemini_model_video
    )
    gemini_api_mode = (
        os.getenv("GEMINI_API_MODE", Config.gemini_api_mode).strip().lower()
        or Config.gemini_api_mode
    )
    vertex_project_id = (
        os.getenv("VERTEX_PROJECT_ID", Config.vertex_project_id or "").strip() or None
    )
    vertex_location = (
        os.getenv("VERTEX_LOCATION", Config.vertex_location or "").strip() or None
    )
    gemini_api_endpoint = (
        os.getenv("GEMINI_API_ENDPOINT", Config.gemini_api_endpoint or "").strip() or None
    )
    gemini_safety_threshold = (
        os.getenv("GEMINI_SAFETY_THRESHOLD", Config.gemini_safety_threshold).strip()
        or Config.gemini_safety_threshold
    )
    fallback_image_models_env = os.getenv("FALLBACK_IMAGE_MODELS")
    if fallback_image_models_env is None:
        fallback_image_models = Config.__dataclass_fields__["fallback_image_models"].default_factory()  # type: ignore[index]
    else:
        fallback_image_models = [
            item.strip()
            for item in fallback_image_models_env.split(",")
            if item.strip()
        ]
        if not fallback_image_models:
            fallback_image_models = Config.__dataclass_fields__["fallback_image_models"].default_factory()  # type: ignore[index]
    raw_sora_api_key = env_str("SORA_API_KEY", "")
    raw_openai_api_key = env_str("OPENAI_API_KEY", "")
    if raw_sora_api_key and raw_openai_api_key and raw_sora_api_key.strip() != raw_openai_api_key.strip():
        log.info("SORA_API_KEY detected; preferring it over OPENAI_API_KEY for Sora access")
    openai_api_key = raw_openai_api_key.strip()
    sora_api_key = (raw_sora_api_key or raw_openai_api_key or "").strip()
    if openai_api_key:
        log.debug("OPENAI_API_KEY detected=%s", _mask_secret(openai_api_key))
    if sora_api_key and sora_api_key != openai_api_key:
        log.debug("SORA_API_KEY detected=%s", _mask_secret(sora_api_key))
    openai_api_base = (
        env_str("OPENAI_API_BASE", Config.openai_api_base).strip()
        or Config.openai_api_base
    )
    openai_org_id = env_str("OPENAI_ORG_ID", Config.openai_org_id or "").strip() or None
    if openai_org_id:
        log.debug("OPENAI_ORG_ID configured=%s", _mask_secret(openai_org_id))
    openai_api_version_video_raw = env_str(
        "OPENAI_API_VERSION_VIDEO", Config.openai_api_version_video
    ).strip()
    openai_api_version_video = _normalise_openai_version(
        openai_api_version_video_raw or Config.openai_api_version_video,
        Config.openai_api_version_video,
    )
    openai_beta_header = env_str("OPENAI_BETA_HEADER", Config.openai_beta_header).strip()
    provider_timeout_s = env_float("PROVIDER_TIMEOUT_S", Config.provider_timeout_s)
    log.debug(
        "OpenAI API config base_url=%s version=%s beta=%s timeout_s=%s",
        openai_api_base,
        openai_api_version_video or "default",
        openai_beta_header or "default",
        provider_timeout_s,
    )
    debug_gemini = DEBUG_GEMINI
    debug_sora = DEBUG_SORA
    raw_sora_model = os.getenv("SORA_MODEL_VIDEO", Config.sora_model_video).strip()
    sora_model_video = raw_sora_model or Config.sora_model_video
    if sora_model_video not in SORA_SUPPORTED_MODELS:
        log.warning(
            "Unsupported Sora video model %s; falling back to %s",
            sora_model_video,
            Config.sora_model_video,
        )
        sora_model_video = Config.sora_model_video
    gemini_enabled = bool(gemini_api_key)
    if not gemini_enabled:
        log.warning("Gemini API key is not configured; generation features will be disabled")

    environment = (
        os.getenv("APP_ENV")
        or os.getenv("ENVIRONMENT")
        or Config.environment
    ).strip() or Config.environment

    default_video_model = (os.getenv("DEFAULT_VIDEO_MODEL") or "").strip()
    if not default_video_model:
        if gemini_enabled:
            default_video_model = gemini_model_video
        elif sora_api_key:
            default_video_model = sora_model_video
        else:
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
        "VEO_COST_RUB_",
        "SORA_COST_RUB_",
    ]
    pricing = PricingConfig(
        credit_price_rub=_get_env_decimal("CREDIT_PRICE_RUB", DEFAULT_CREDIT_PRICE_RUB),
        markup_pct=_get_env_decimal("MARKUP_PCT", DEFAULT_MARKUP_PCT),
        fix_fee_rub=_get_env_decimal("FIX_FEE_RUB", DEFAULT_FIX_FEE_RUB),
        provider_costs=_load_provider_costs(prefixes),
    )
    return Config(
        bot_token=bot_token,
        gemini_api_key=gemini_api_key,
        environment=environment,
        gemini_model_text=gemini_model_text,
        gemini_model_text_fallback=gemini_model_text_fallback,
        gemini_model_image=gemini_model_image,
        gemini_model_video=gemini_model_video,
        gemini_api_mode=gemini_api_mode,
        vertex_project_id=vertex_project_id,
        vertex_location=vertex_location,
        gemini_api_endpoint=gemini_api_endpoint,
        gemini_safety_threshold=gemini_safety_threshold,
        fallback_image_models=fallback_image_models,
        openai_api_key=openai_api_key,
        sora_api_key=sora_api_key,
        openai_org_id=openai_org_id,
        sora_model_video=sora_model_video,
        openai_api_base=openai_api_base,
        openai_api_version_video=openai_api_version_video,
        openai_beta_header=openai_beta_header,
        sora_requests_per_minute=_get_env_int(
            "SORA_REQUESTS_PER_MINUTE", Config.sora_requests_per_minute
        ),
        gemini_requests_per_minute=_get_env_int(
            "GEMINI_REQUESTS_PER_MINUTE", Config.gemini_requests_per_minute
        ),
        default_video_model=default_video_model,
        database_path=database_path,
        google_sheet_id=google_sheet_id,
        gs_users_sheet=users_sheet,
        gs_payments_sheet=payments_sheet,
        gs_jobs_sheet=jobs_sheet,
        gs_errors_sheet=errors_sheet,
        pricing=pricing,
        credit_packages=_build_credit_packages(TOP_UP_PACKAGES),
        jobs_concurrency=_get_env_int("JOBS_CONCURRENCY", Config.jobs_concurrency),
        max_jobs_per_user=_get_env_int("MAX_JOBS_PER_USER", Config.max_jobs_per_user),
        request_timeout=_get_env_float("REQUEST_TIMEOUT", Config.request_timeout),
        request_connect_timeout=_get_env_float(
            "REQUEST_CONNECT_TIMEOUT", Config.request_connect_timeout
        ),
        request_read_timeout=_get_env_float(
            "REQUEST_READ_TIMEOUT", Config.request_read_timeout
        ),
        provider_timeout_s=provider_timeout_s,
        request_retries=_get_env_int("REQUEST_RETRIES", Config.request_retries),
        retry_backoff=_get_env_float("RETRY_BACKOFF", Config.retry_backoff),
        command_rate_limits=_parse_rate_limits(os.getenv("RATE_LIMITS")),
        yookassa_shop_id=os.getenv("YOOKASSA_SHOP_ID"),
        yookassa_secret_key=os.getenv("YOOKASSA_SECRET_KEY"),
        yookassa_test_mode=_get_env_bool("YOOKASSA_TEST_MODE", Config.yookassa_test_mode),
        public_base_url=os.getenv("PUBLIC_BASE_URL", Config.public_base_url),
        yookassa_return_path=os.getenv("YOOKASSA_RETURN_PATH", Config.yookassa_return_path),
        yookassa_webhook_path=os.getenv("YOOKASSA_WEBHOOK_PATH", Config.yookassa_webhook_path),
        yookassa_send_receipts=_get_env_bool("YOOKASSA_SEND_RECEIPTS", Config.yookassa_send_receipts),
        subscription_chat_id=os.getenv("SUBSCRIPTION_CHAT_ID"),
        support_chat_id=_get_optional_int(os.getenv("SUPPORT_CHAT_ID")),
        archive_channel_id=_parse_chat_id(
            os.getenv("ARCHIVE_CHANNEL_ID"), env_name="ARCHIVE_CHANNEL_ID"
        ),
        yookassa_poll_interval=_get_env_int("YOOKASSA_POLL_INTERVAL", Config.yookassa_poll_interval),
        payments_read_only=_get_env_bool("PAYMENTS_READ_ONLY", Config.payments_read_only),
        terms_url=os.getenv("TERMS_URL", Config.terms_url),
        admin_ids=_get_admin_ids_from_env(),
        bot_version=os.getenv("BOT_VERSION", Config.bot_version),
        support_notify_interval=_get_env_int(
            "SUPPORT_NOTIFY_INTERVAL", Config.support_notify_interval
        ),
        debug_gemini=debug_gemini,
        debug_sora=debug_sora,
    )


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime configuration that controls the bot launch mode."""

    BOT_MODE: str
    WEBHOOK_HOST: str
    WEBHOOK_PATH: str
    WEBHOOK_URL: str
    TG_WEBHOOK_SECRET: str
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
    raw_secret = os.getenv("TG_WEBHOOK_SECRET", "").strip()
    if not raw_secret:
        raw_secret = os.getenv("TELEGRAM_SECRET_TOKEN", "").strip()
    webhook_secret = ""
    if raw_secret:
        if re.fullmatch(r"^[A-Za-z0-9_-]{1,256}$", raw_secret):
            webhook_secret = raw_secret
        else:
            log.warning(
                "TG_WEBHOOK_SECRET does not match ^[A-Za-z0-9_-]{1,256}$; secret token will be ignored",
            )
    host = os.getenv("HOST", "0.0.0.0") or "0.0.0.0"
    port = _get_env_int("PORT", 8080)

    if mode == "webhook":
        if webhook_host and not webhook_host.startswith("https://"):
            log.warning(
                "WEBHOOK_HOST does not start with https://; Telegram may reject it",
            )
        if not webhook_host:
            log.warning(
                "WEBHOOK_HOST is empty while BOT_MODE=webhook; falling back to polling is expected",
            )
        if raw_secret and not webhook_secret:
            log.warning(
                "TG_WEBHOOK_SECRET is invalid while BOT_MODE=webhook; webhook will be set without secret token",
            )
        if not raw_secret:
            log.warning(
                "TG_WEBHOOK_SECRET is empty while BOT_MODE=webhook; webhook will be set without secret token",
            )

    return RuntimeConfig(
        BOT_MODE=mode,
        WEBHOOK_HOST=webhook_host,
        WEBHOOK_PATH=webhook_path,
        WEBHOOK_URL=webhook_url,
        TG_WEBHOOK_SECRET=webhook_secret,
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
    "RateLimitRule",
    "load_config",
    "env_str",
    "env_float",
]


class SafetyCfg:
    ENABLE_AUTO_REPHRASE: bool = True
    ENABLE_NEGATIVE_PROMPT: bool = True
    MAX_RETRIES: int = 2
    ENABLE_PRE_CLEAN: bool = True
    NEGATIVE_PROMPT_BASE: str = (
        "no nudity; no violence; no gore; no hate; family-friendly; safe; neutral"
    )


class RoutingCfg:
    TEXT_API_VERSION = "v1"
    MEDIA_API_VERSION = "v1beta"


class Models:
    TEXT_MODEL = "gemini-2.5-flash"
    IMAGE_MODEL = "gemini-2.5-flash-image"
    VIDEO_MODEL = "veo-3.0-generate-001"


# Apply runtime overrides derived from environment variables.
if GEMINI_IMAGE_API_VERSION:
    RoutingCfg.MEDIA_API_VERSION = GEMINI_IMAGE_API_VERSION

