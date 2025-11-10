import asyncio
import json

import httpx

from config import Config
from providers.openai_image import OpenAIImageClient


class _StubResponse:
    def __init__(self, status_code: int, payload, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = dict(headers or {})
        self.encoding = "utf-8"

    async def aread(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    async def aclose(self) -> None:
        return None


def test_openai_image_enqueue_default_and_dalle(tmp_path, monkeypatch):
    config = Config(bot_token="token", gemini_api_key="key", environment="test", openai_api_key="sk-test")
    client = OpenAIImageClient(config=config)
    client._pending_dir = tmp_path / "openai"
    client._pending_dir.mkdir(parents=True, exist_ok=True)

    captured = {}

    async def _fake_request(*args, **kwargs):
        captured["json"] = kwargs.get("json")
        return {"data": []}, 200, 120

    monkeypatch.setattr(client, "_request", _fake_request)

    asyncio.run(
        client.enqueue_job(
            prompt="draw a cat",
            settings={},
            payload={},
            idempotency_key="idem-default",
        )
    )
    assert captured["json"]["model"] == "gpt-image-1"
    assert "n" not in captured["json"]

    async def _fake_request_dalle(*args, **kwargs):
        captured["json"] = kwargs.get("json")
        return {"data": []}, 200, 80

    monkeypatch.setattr(client, "_request", _fake_request_dalle)
    asyncio.run(
        client.enqueue_job(
            prompt="draw a dog",
            settings={"model": "dall-e-3"},
            payload={},
            idempotency_key="idem-dalle",
        )
    )
    assert captured["json"]["model"] == "dall-e-3"
    assert captured["json"]["n"] == 1


def test_openai_image_retries_on_timeout(monkeypatch):
    config = Config(bot_token="token", gemini_api_key="key", environment="test", openai_api_key="sk-test")
    client = OpenAIImageClient(config=config)

    attempts = {"count": 0}

    class _StubClient:
        async def request(self, method, url, headers=None, **kwargs):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise httpx.ReadTimeout("timeout", request=httpx.Request(method, url))
            return _StubResponse(200, {"data": []}, headers={"x-request-id": "req-1"})

    async def _fake_ensure_http_client():
        return _StubClient()

    async def _no_sleep(*_args, **_kwargs):
        return None

    monkeypatch.setattr(client, "_ensure_http_client", _fake_ensure_http_client)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    async def _run_request():
        return await client._request("POST", "/images/generations", json={"prompt": "hello"})

    data, status, _ = asyncio.run(_run_request())

    assert attempts["count"] == 3
    assert status == 200
    assert data == {"data": []}
