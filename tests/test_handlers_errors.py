import pytest
from tests.aiogram_stub import ensure_aiogram_stub

ensure_aiogram_stub()

from handlers import _map_provider_error
from providers import ProviderAPIError


def test_map_provider_error_policy_from_safety_message():
    error = ProviderAPIError(
        provider="gemini",
        status_code=400,
        message="Запрос отклонён политиками Gemini",
        error_type="safety",
        provider_message="Запрос отклонён политиками Gemini",
    )

    message, short, notify_support, hint = _map_provider_error(error)

    assert "правила контента" in message
    assert "Gemini" in message
    assert short == "Запрос отклонён политиками Gemini"
    assert notify_support is False
    assert hint == "policy"


def test_map_provider_error_policy_without_details():
    error = ProviderAPIError(
        provider="gemini",
        status_code=400,
        message="Запрос отклонён политиками Gemini",
        error_type="policy",
        provider_message="",
    )

    message, short, notify_support, hint = _map_provider_error(error)

    assert message == "Запрос нарушает правила контента."
    assert short == "Запрос отклонён политиками Gemini"
    assert notify_support is False
    assert hint == "policy"
