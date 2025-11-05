import asyncio
import sys
import types

import pytest

from tests.aiogram_stub import ensure_aiogram_stub

ensure_aiogram_stub()

if "yookassa" not in sys.modules:
    yookassa_module = types.ModuleType("yookassa")
    yookassa_module.Configuration = type("Configuration", (), {})
    yookassa_module.Payment = type("Payment", (), {})
    sys.modules["yookassa"] = yookassa_module

if "yookassa_client" not in sys.modules:
    yookassa_client_module = types.ModuleType("yookassa_client")
    yookassa_client_module.create_payment = lambda *args, **kwargs: None
    sys.modules["yookassa_client"] = yookassa_client_module

from aiogram.utils.exceptions import ChatNotFound, RetryAfter  # type: ignore

import handlers


def _run(coro):
    return asyncio.run(coro)


def test_collect_broadcast_recipients_filters_duplicates():
    class _DB:
        async def list_users(self):
            return [
                {"user_id": "101"},
                {"user_id": "202"},
                {"user_id": None},
                {"user_id": "not-int"},
                {"user_id": 101},
                {"user_id": 303},
            ]

    recipients = _run(handlers._collect_broadcast_recipients(_DB()))

    assert recipients == [101, 202, 303]


def test_find_user_record_matches_by_username_and_id():
    users = [
        {"user_id": "101", "username": "Admin"},
        {"user_id": "202", "username": "operator"},
    ]

    record_username = handlers._find_user_record(users, "@admin")
    record_id = handlers._find_user_record(users, "202")
    missing = handlers._find_user_record(users, "@missing")

    assert record_username == users[0]
    assert record_id == users[1]
    assert missing is None


def test_send_text_safely_retries_on_retry_after(monkeypatch: pytest.MonkeyPatch):
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("handlers.asyncio.sleep", fake_sleep)

    class _Bot:
        def __init__(self) -> None:
            self.attempts = 0

        async def send_message(self, user_id: int, text: str) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise RetryAfter(timeout=0)

    bot = _Bot()

    ok = _run(handlers._send_text_safely(bot, 1, "hi"))

    assert ok is True
    assert bot.attempts == 2
    assert sleep_calls == [0]


def test_broadcast_to_users_counts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("handlers._ADMIN_BROADCAST_THROTTLE_SECONDS", 0)

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("handlers.asyncio.sleep", fake_sleep)

    class _Bot:
        def __init__(self) -> None:
            self.behaviour = {
                1: ["ok"],
                2: ["retry", "ok"],
                3: ["fail"],
            }

        async def send_message(self, user_id: int, text: str) -> None:
            sequence = self.behaviour[user_id]
            outcome = sequence.pop(0)
            if outcome == "ok":
                return None
            if outcome == "retry":
                raise RetryAfter(timeout=0)
            if outcome == "fail":
                raise ChatNotFound()
            raise AssertionError(f"unexpected outcome {outcome}")

    bot = _Bot()

    sent, failed = _run(handlers._broadcast_to_users(bot, [1, 2, 3], "hello"))

    assert sent == 2
    assert failed == 1
    # RetryAfter sleeps are recorded even when the timeout is zero
    assert sleep_calls == [0]
