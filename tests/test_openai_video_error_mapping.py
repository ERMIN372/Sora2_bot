"""Tests for OpenAI video API error normalisation."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

import pytest

from providers.base import BaseProviderClient, ProviderAPIError
from providers.openai_video import OpenAIVideoClient


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize(
    "status_code,error_payload,expected_type,expected_message,expected_retryable",
    [
        (
            401,
            {"error": {"message": "Incorrect API key", "code": "invalid_api_key"}},
            "auth",
            "OpenAI отклонил запрос: проверьте корректность API‑ключа и регион доступа.",
            False,
        ),
        (
            429,
            {"error": {"message": "quota_exceeded"}},
            "rate_limit",
            "Превышен лимит запросов. Пополните баланс в кабинете OpenAI",
            True,
        ),
    ],
)
def test_openai_video_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
    make_openai_video_config,
    status_code: int,
    error_payload: Dict[str, Any],
    expected_type: str,
    expected_message: str,
    expected_retryable: bool,
) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)

    raw_error = ProviderAPIError(
        provider="openai-video",
        status_code=status_code,
        message="error",
        provider_message=json.dumps(error_payload),
        error_code="base_error",
        retryable=False,
    )

    async def fake_request(self, method: str, path: str, **kwargs: Any):
        raise raw_error

    monkeypatch.setattr(BaseProviderClient, "_request", fake_request)

    async def exercise() -> None:
        await client._request_json("GET", "/v1/videos")

    with pytest.raises(ProviderAPIError) as exc_info:
        _run(exercise())

    error = exc_info.value
    assert error.status_code == status_code
    assert error.error_type == expected_type
    assert str(error) == expected_message
    assert error.retryable is expected_retryable
