from decimal import Decimal

import pytest

from config import (
    Config,
    DEFAULT_CREDIT_PRICE_RUB,
    DEFAULT_FIX_FEE_RUB,
    DEFAULT_MARKUP_PCT,
    PricingConfig,
)
from handlers import (
    ADMIN_SORA_DURATION_OPTION,
    DEFAULT_VIDEO_DURATION,
    UserSession,
    VEO_FIXED_DURATION,
    VEO_FIXED_PRICE_RUB,
    VIDEO_DURATION_OPTIONS,
    _available_duration_options,
    _create_order,
)


@pytest.fixture()
def config() -> Config:
    pricing = PricingConfig(
        credit_price_rub=DEFAULT_CREDIT_PRICE_RUB,
        markup_pct=DEFAULT_MARKUP_PCT,
        fix_fee_rub=DEFAULT_FIX_FEE_RUB,
        provider_costs={},
    )
    return Config(
        bot_token="test-token",
        gemini_api_key="test-key",
        pricing=pricing,
    )


def test_create_order_for_veo_uses_fixed_duration_and_price(config: Config) -> None:
    session = UserSession()
    session.video_duration = 25
    session.hd_enabled = True

    order = _create_order(
        category="video",
        flow="text",
        prompt="demo",
        config=config,
        session=session,
        model="veo-3.0-generate-001",
        provider="veo",
        product="veo3",
        model_label="veo",
        include_size=True,
    )

    assert order.duration_seconds == VEO_FIXED_DURATION
    assert order.hd is False
    assert order.price_rub == VEO_FIXED_PRICE_RUB
    assert order.credits_cost == config.credits_for_rubles(VEO_FIXED_PRICE_RUB)


def test_create_order_for_sora_preserves_duration_and_hd(config: Config) -> None:
    session = UserSession()
    session.video_duration = 15
    session.hd_enabled = True

    order = _create_order(
        category="video",
        flow="text",
        prompt="demo",
        config=config,
        session=session,
        model="sora-2",
        provider="sora",
        product="sora",
        model_label="Sora",
        include_size=True,
    )

    assert order.duration_seconds == 15
    assert order.hd is True
    assert order.price_rub == Decimal("169")
    assert order.credits_cost == config.credits_for_rubles(order.price_rub)

    session.video_duration = DEFAULT_VIDEO_DURATION
    order_default = _create_order(
        category="video",
        flow="text",
        prompt="demo",
        config=config,
        session=session,
        model="sora-2",
        provider="sora",
        product="sora",
        model_label="Sora",
        include_size=True,
    )

    assert order_default.duration_seconds == DEFAULT_VIDEO_DURATION


def test_available_duration_options_for_admin_sora() -> None:
    options = _available_duration_options(
        provider="sora",
        product="sora",
        model="sora-2",
        is_admin=True,
    )

    assert options == (ADMIN_SORA_DURATION_OPTION,) + VIDEO_DURATION_OPTIONS


def test_available_duration_options_for_regular_user() -> None:
    options = _available_duration_options(
        provider="sora",
        product="sora",
        model="sora-2",
        is_admin=False,
    )

    assert options == VIDEO_DURATION_OPTIONS


def test_available_duration_options_for_veo_returns_empty() -> None:
    options = _available_duration_options(
        provider="veo",
        product="veo3",
        model="veo-3.0-generate-001",
        is_admin=True,
    )

    assert options == ()
