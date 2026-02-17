from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from providers.base import BaseProviderClient


def _run(coro):
    return asyncio.run(coro)


class _FakeResponse:
    def __init__(
        self, status: int = 200, *, text: str = "{}", headers: Optional[Dict[str, str]] = None
    ) -> None:
        self.status = status
        self._text = text
        self.headers = headers or {}

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def text(self) -> str:
        return self._text


class _FakeRequest:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return await self._response.__aenter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return await self._response.__aexit__(exc_type, exc, tb)


class _FakeSession:
    def __init__(self, response: Optional[_FakeResponse] = None) -> None:
        self.response = response or _FakeResponse()
        self.last_headers: Dict[str, str] = {}

    def request(self, method: str, url: str, **kwargs: Any) -> _FakeRequest:
        self.last_headers = dict(kwargs.get("headers") or {})
        return _FakeRequest(self.response)


def test_get_without_body_strips_content_type(make_openai_video_config) -> None:
    client = BaseProviderClient(
        config=make_openai_video_config(),
        base_url="https://example.invalid",
        api_key="sk-test",
        provider_name="test",
    )
    fake_session = _FakeSession()

    async def fake_ensure_session():
        return fake_session

    client._ensure_session = fake_ensure_session  # type: ignore[method-assign]

    _run(
        client._perform_request(
            "GET",
            "https://example.invalid/v1/jobs",
            headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"},
        )
    )

    assert "Content-Type" not in fake_session.last_headers


def test_get_with_json_body_keeps_content_type(make_openai_video_config) -> None:
    client = BaseProviderClient(
        config=make_openai_video_config(),
        base_url="https://example.invalid",
        api_key="sk-test",
        provider_name="test",
    )
    fake_session = _FakeSession()

    async def fake_ensure_session():
        return fake_session

    client._ensure_session = fake_ensure_session  # type: ignore[method-assign]

    _run(
        client._perform_request(
            "GET",
            "https://example.invalid/v1/jobs",
            headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"},
            json={"probe": True},
        )
    )

    assert fake_session.last_headers.get("Content-Type") == "application/json"
