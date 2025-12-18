import hashlib
import hmac
import json
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_server import _yookassa_router
from config import Config
import yookassa_client


class _StubProcessor:
    def __init__(self) -> None:
        self.called = False
        self.source = None
        self.payment = None

    async def process_payment(self, payment, *, source: str) -> None:  # pragma: no cover - awaited in tests
        self.called = True
        self.source = source
        self.payment = payment


def _build_client(secret: str, monkeypatch) -> tuple[TestClient, _StubProcessor, object]:
    config = Config(bot_token="dummy-token", yookassa_secret_key=secret, yookassa_shop_id="shop")
    processor = _StubProcessor()
    app = FastAPI()
    app.include_router(_yookassa_router(config=config, processor=processor))
    payment = object()
    monkeypatch.setattr(yookassa_client, "get_payment", lambda *_args, **_kwargs: payment)
    return TestClient(app), processor, payment


def _signature(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _payload_body(payload: dict) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def test_webhook_accepts_valid_signature(monkeypatch):
    secret = "topsecret"
    payload = {"type": "notification", "event": "payment.succeeded", "object": {"id": "pay_123"}}
    body = _payload_body(payload)
    client, processor, payment = _build_client(secret, monkeypatch)

    response = client.post(
        "/pay/webhook",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-YooKassa-Signature": f"sha256={_signature(secret, body)}",
        },
    )

    assert response.status_code == 200
    assert processor.called
    assert processor.source == "webhook"
    assert processor.payment is payment


def test_webhook_rejects_invalid_signature(monkeypatch, caplog):
    secret = "topsecret"
    payload = {"type": "notification", "event": "payment.succeeded", "object": {"id": "pay_123"}}
    body = _payload_body(payload)
    client, processor, _payment = _build_client(secret, monkeypatch)
    caplog.set_level(logging.WARNING)

    response = client.post(
        "/pay/webhook",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-YooKassa-Signature": "sha256=not-valid",
        },
    )

    assert response.status_code == 401
    assert not processor.called
    assert "invalid signature" in caplog.text
