from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import app_server


class _StubDispatcher:
    def __init__(self) -> None:
        self.updates = []

    async def process_update(self, update) -> None:
        self.updates.append(update)


class _StubUpdate:
    @staticmethod
    def to_object(data):
        return {"update_id": data.get("update_id", 0)}


class _StubBotClass:
    @staticmethod
    def set_current(_value) -> None:
        return None


class _StubDispatcherClass:
    @staticmethod
    def set_current(_value) -> None:
        return None


def test_telegram_webhook_uses_configured_path_and_legacy_alias(monkeypatch):
    monkeypatch.setattr(
        app_server,
        "CFG",
        SimpleNamespace(WEBHOOK_PATH="/custom/webhook", TG_WEBHOOK_SECRET=""),
    )
    monkeypatch.setattr(app_server.types, "Update", _StubUpdate)
    monkeypatch.setattr(app_server, "Bot", _StubBotClass)
    monkeypatch.setattr(app_server, "Dispatcher", _StubDispatcherClass)

    dp = _StubDispatcher()
    app = FastAPI()
    app.include_router(app_server._telegram_router(dp=dp, bot=object()))
    client = TestClient(app)

    payload = {"update_id": 101, "message": {"text": "/start"}}

    custom_response = client.post("/custom/webhook", json=payload)
    legacy_response = client.post("/tg/webhook", json=payload)

    assert custom_response.status_code == 200
    assert legacy_response.status_code == 200
    assert len(dp.updates) == 2


def test_telegram_webhook_secret_validation(monkeypatch):
    monkeypatch.setattr(
        app_server,
        "CFG",
        SimpleNamespace(WEBHOOK_PATH="/tg/webhook", TG_WEBHOOK_SECRET="secret"),
    )
    monkeypatch.setattr(app_server.types, "Update", _StubUpdate)
    monkeypatch.setattr(app_server, "Bot", _StubBotClass)
    monkeypatch.setattr(app_server, "Dispatcher", _StubDispatcherClass)

    dp = _StubDispatcher()
    app = FastAPI()
    app.include_router(app_server._telegram_router(dp=dp, bot=object()))
    client = TestClient(app)

    payload = {"update_id": 202, "message": {"text": "/start"}}

    rejected = client.post("/tg/webhook", json=payload)
    accepted = client.post(
        "/tg/webhook",
        json=payload,
        headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
    )

    assert rejected.status_code == 200
    assert accepted.status_code == 200
    assert len(dp.updates) == 1
