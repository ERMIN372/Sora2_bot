import asyncio
import logging
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import app_server


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


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


def test_telegram_webhook_debug_logging(monkeypatch, caplog):
    monkeypatch.setenv("TG_DEBUG_UPDATES", "1")
    monkeypatch.setattr(
        app_server,
        "CFG",
        SimpleNamespace(WEBHOOK_PATH="/tg/webhook", TG_WEBHOOK_SECRET=""),
    )
    monkeypatch.setattr(app_server.types, "Update", _StubUpdate)
    monkeypatch.setattr(app_server, "Bot", _StubBotClass)
    monkeypatch.setattr(app_server, "Dispatcher", _StubDispatcherClass)

    dp = _StubDispatcher()
    app = FastAPI()
    app.include_router(app_server._telegram_router(dp=dp, bot=object()))
    client = TestClient(app)

    caplog.set_level(logging.INFO)
    payload = {"update_id": 303, "message": {"text": "/start", "chat": {"id": 77}}}
    response = client.post("/tg/webhook", json=payload)

    assert response.status_code == 200
    assert "tg.webhook.dispatch.debug" in caplog.text
    assert "tg.webhook.processed update_id=303" in caplog.text


def test_telegram_webhook_strict_secret_mode_blocks_invalid_token(monkeypatch):
    monkeypatch.setenv("TG_WEBHOOK_SECRET_MODE", "strict")
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

    response = client.post("/tg/webhook", json={"update_id": 404, "message": {"text": "/start"}})

    assert response.status_code == 200
    assert len(dp.updates) == 0


def test_telegram_webhook_warn_secret_mode_allows_invalid_token(monkeypatch):
    monkeypatch.setenv("TG_WEBHOOK_SECRET_MODE", "warn")
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

    response = client.post("/tg/webhook", json={"update_id": 405, "message": {"text": "/start"}})

    assert response.status_code == 200
    assert len(dp.updates) == 1


class _StubWebhookInfo:
    url = "https://example.org/tg/webhook"
    pending_update_count = 7
    last_error_date = None
    last_error_message = None
    max_connections = 40
    ip_address = "1.2.3.4"


class _DiagBot:
    async def get_webhook_info(self):
        return _StubWebhookInfo()


def test_diag_returns_live_telegram_webhook_info(monkeypatch):
    monkeypatch.setattr(
        app_server,
        "CFG",
        SimpleNamespace(
            BOT_MODE="webhook",
            WEBHOOK_URL="https://example.org/tg/webhook",
            WEBHOOK_PATH="/tg/webhook",
            TG_WEBHOOK_SECRET="",
        ),
    )
    app = app_server._create_base_app()
    app.state.bot = _DiagBot()
    client = TestClient(app)

    response = client.get("/diag")

    assert response.status_code == 200
    payload = response.json()
    assert payload["webhook_url"] == "https://example.org/tg/webhook"
    assert payload["telegram_webhook_info"]["pending_update_count"] == 7


class _GuardBot:
    def __init__(self, current_url: str) -> None:
        self._info = SimpleNamespace(
            url=current_url,
            pending_update_count=3,
            last_error_date=None,
            last_error_message=None,
            max_connections=40,
            ip_address="1.1.1.1",
        )
        self.calls = []

    async def get_webhook_info(self):
        return self._info

    async def set_webhook(
        self, url, secret_token=None, drop_pending_updates=True, allowed_updates=None
    ):
        self.calls.append(
            {
                "url": url,
                "secret_token": secret_token,
                "drop_pending_updates": drop_pending_updates,
                "allowed_updates": allowed_updates,
            }
        )


def test_ensure_webhook_configured_repairs_mismatch(monkeypatch):
    monkeypatch.setattr(
        app_server,
        "CFG",
        SimpleNamespace(
            WEBHOOK_URL="https://example.org/tg/webhook",
            TG_WEBHOOK_SECRET="secret",
            BOT_MODE="webhook",
        ),
    )
    bot = _GuardBot(current_url="")

    _run(app_server._ensure_webhook_configured(bot))

    assert len(bot.calls) == 1
    assert bot.calls[0]["url"] == "https://example.org/tg/webhook"
    assert bot.calls[0]["drop_pending_updates"] is False
