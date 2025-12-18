import asyncio
import time
from types import SimpleNamespace

import pytest
from aiogram.dispatcher.handler import CancelHandler

from config import RateLimitRule
from middleware.admin_check import AdminCheckMiddleware
from middleware.rate_limit import RateLimitMiddleware


class FakeRedis:
    def __init__(self) -> None:
        self._values: dict[str, int] = {}
        self._expires: dict[str, float] = {}

    async def incr(self, key: str) -> int:
        self._cleanup()
        self._values[key] = self._values.get(key, 0) + 1
        return self._values[key]

    async def expire(self, key: str, seconds: int) -> None:  # pragma: no cover - trivial setter
        self._expires[key] = time.monotonic() + seconds

    def _cleanup(self) -> None:
        now = time.monotonic()
        expired = [key for key, ts in self._expires.items() if ts <= now]
        for key in expired:
            self._values.pop(key, None)
            self._expires.pop(key, None)


class DummyMessage:
    def __init__(self, user_id: int, text: str) -> None:
        self.from_user = SimpleNamespace(id=user_id)
        self.text = text
        self.responses: list[str] = []

    def get_command(self, pure: bool = True) -> str:
        return self.text.split()[0].lstrip("/") if self.text.startswith("/") else ""

    async def answer(self, text: str) -> None:
        self.responses.append(text)


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def test_rate_limit_allows_within_limit() -> None:
    middleware = RateLimitMiddleware(
        FakeRedis(),
        {"video_create": RateLimitRule(limit=2, period=60)},
    )
    message = DummyMessage(user_id=1, text="/video_create")

    _run(middleware.on_pre_process_message(message, {}))
    _run(middleware.on_pre_process_message(message, {}))

    assert message.responses == []


def test_rate_limit_blocks_when_exceeded() -> None:
    middleware = RateLimitMiddleware(
        FakeRedis(),
        {"video_create": RateLimitRule(limit=2, period=60)},
    )
    message = DummyMessage(user_id=1, text="/video_create")

    _run(middleware.on_pre_process_message(message, {}))
    _run(middleware.on_pre_process_message(message, {}))

    with pytest.raises(CancelHandler):
        _run(middleware.on_pre_process_message(message, {}))

    assert message.responses == ["Превышен лимит запросов, попробуйте позже."]


def test_admin_check_blocks_non_admin() -> None:
    middleware = AdminCheckMiddleware((123,))
    message = DummyMessage(user_id=42, text="/admin_status")

    with pytest.raises(CancelHandler):
        _run(middleware.on_pre_process_message(message, {}))

    assert message.responses == ["Эта команда доступна только администраторам"]


def test_admin_check_allows_admin() -> None:
    middleware = AdminCheckMiddleware((123,))
    message = DummyMessage(user_id=123, text="/admin_status")

    _run(middleware.on_pre_process_message(message, {}))

    assert message.responses == []
