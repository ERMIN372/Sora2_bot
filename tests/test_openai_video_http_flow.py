"""Tests covering the OpenAI video client's HTTP-facing helpers."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest

from providers.base import BaseProviderClient, ProviderAPIError
from providers.openai_video import OpenAIVideoClient


def _run(coro):
    return asyncio.run(coro)


class _BinaryFakeResponse:
    def __init__(
        self,
        status: int,
        *,
        body: bytes = b"",
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.status = status
        self._body = body
        self.headers = dict(headers or {})
        self.history: List[Any] = []

    async def __aenter__(self) -> "_BinaryFakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def read(self) -> bytes:
        return self._body


class _BinaryFakeRequest:
    def __init__(self, response: _BinaryFakeResponse) -> None:
        self._response = response

    def __await__(self):
        async def _inner() -> _BinaryFakeResponse:
            return self._response

        return _inner().__await__()

    async def __aenter__(self) -> _BinaryFakeResponse:
        return await self._response.__aenter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return await self._response.__aexit__(exc_type, exc, tb)


class _BinaryFakeSession:
    def __init__(self, responses: List[_BinaryFakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: List[tuple[str, str, Dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> _BinaryFakeRequest:
        if not self._responses:
            raise AssertionError("No more fake responses queued")
        self.calls.append((method, url, kwargs))
        return _BinaryFakeRequest(self._responses.pop(0))


def test_openai_video_binary_request_follows_redirect(
    monkeypatch: pytest.MonkeyPatch, make_openai_video_config
) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)

    final_response = _BinaryFakeResponse(
        200,
        body=b"binary-data",
        headers={"x-request-id": "req-bin", "content-type": "video/mp4"},
    )
    final_response.history = [
        _BinaryFakeResponse(302, headers={"Location": "https://cdn.example/video.mp4"})
    ]
    fake_session = _BinaryFakeSession([final_response])

    async def fake_ensure(self) -> _BinaryFakeSession:
        return fake_session

    monkeypatch.setattr(OpenAIVideoClient, "_ensure_session", fake_ensure)

    body, headers = _run(
        client.content("vid-bin", asset_id="primary", params={"variant": "full"})
    )

    assert body == b"binary-data"
    assert headers["content-type"].startswith("video/")
    assert fake_session.calls
    method, url, kwargs = fake_session.calls[0]
    assert method == "GET"
    assert url.endswith("/v1/videos/vid-bin/content/primary")
    assert kwargs.get("params") == {"variant": "full"}
    meta = client._last_download_meta
    assert meta.get("status_code") == 200
    assert meta.get("request_id") == "req-bin"


def test_openai_video_request_retries_conflict(
    monkeypatch: pytest.MonkeyPatch, make_openai_video_config
) -> None:
    config = make_openai_video_config(request_retries=5, retry_backoff=0.1)
    client = OpenAIVideoClient(config=config)

    attempts: List[tuple[str, str]] = []

    async def fake_perform(
        self, method: str, url: str, **kwargs: Any
    ) -> tuple[Dict[str, Any], int, int]:
        attempts.append((method, url))
        if len(attempts) < 3:
            raise ProviderAPIError(
                provider=self.provider_name,
                status_code=409,
                message="conflict",
                error_type="conflict",
                error_code="conflict",
                retryable=True,
            )
        return {"ok": True}, 200, 42

    sleeps: List[float] = []

    async def fast_sleep(delay: float) -> None:
        sleeps.append(delay)
        return None

    monkeypatch.setattr(BaseProviderClient, "_perform_request", fake_perform)
    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    data, status_code, duration_ms = _run(
        client._request_json("GET", "/v1/videos/test", diagnostic={})
    )

    assert data == {"ok": True}
    assert status_code == 200
    assert len(attempts) == 3
    assert len(sleeps) == 2
    assert all(delay >= 0 for delay in sleeps)


@pytest.mark.parametrize(
    "status,payload,expected_type,expected_code,expected_snippet",
    [
        (
            404,
            {"error": {"code": "not_found", "message": "Video not found", "type": "invalid_request_error"}},
            "invalid_request_error",
            "not_found",
            "Video not found",
        ),
        (
            401,
            {"error": {"code": "unauthorized", "message": "missing auth"}},
            "auth",
            "auth",
            "корректность",
        ),
        (
            403,
            {"error": {"code": "forbidden", "message": "region blocked"}},
            "region_blocked",
            "region_blocked",
            "регион",
        ),
    ],
)
def test_openai_video_error_mapping_from_http(
    monkeypatch: pytest.MonkeyPatch,
    make_openai_video_config,
    status: int,
    payload: Dict[str, Any],
    expected_type: str,
    expected_code: str,
    expected_snippet: str,
) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)
    body = json.dumps(payload)

    async def fake_perform(
        self, method: str, url: str, **kwargs: Any
    ) -> tuple[Dict[str, Any], int, int]:
        raise ProviderAPIError(
            provider=self.provider_name,
            status_code=status,
            message="boom",
            error_type="http",
            error_code=None,
            provider_message=body,
            retryable=False,
        )

    monkeypatch.setattr(BaseProviderClient, "_perform_request", fake_perform)

    with pytest.raises(ProviderAPIError) as excinfo:
        _run(client._request_json("GET", "/v1/videos/problem", diagnostic={}))

    error = excinfo.value
    assert error.status_code == status
    assert error.error_type == expected_type
    assert error.error_code == expected_code
    assert expected_snippet in str(error)


def test_openai_video_full_cycle(monkeypatch: pytest.MonkeyPatch, make_openai_video_config) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)

    calls: list[tuple[str, str, Dict[str, Any]]] = []

    async def fake_request(self, method: str, path: str, **kwargs: Any):
        calls.append((method, path, kwargs))
        if method == "POST" and path == "/v1/videos":
            payload = kwargs.get("json") or {}
            assert payload["prompt"] == "Render a serene landscape"
            assert payload["model"] == "sora-2"
            assert payload.get("duration") == 8
            assert payload.get("n") == 1
            assert payload.get("response_format") == "url"
            assert "metadata" not in payload
            return {"id": "vid_123", "status": "queued"}, 201, 120
        if method == "GET" and path == "/v1/videos/vid_123":
            return (
                {
                    "id": "vid_123",
                    "status": "ready",
                    "assets": {"primary": {"download_url": "https://cdn.local/video.mp4"}},
                },
                200,
                45,
            )
        raise AssertionError(f"Unexpected request {method} {path}")

    async def fake_binary(self, method: str, path: str, params=None, **_: Any):
        assert method == "GET"
        assert path == "/v1/videos/vid_123/content/primary"
        assert params is None
        self._record_response_history(
            method=method,
            path=path,
            status_code=200,
            duration_ms=25,
            request_id="diag-bin",
            diagnostic={"dropped_fields": self.last_request.get("dropped_fields", ())},
        )
        return b"video-bytes", {"content-type": "video/mp4"}

    monkeypatch.setattr(BaseProviderClient, "_request", fake_request)
    monkeypatch.setattr(OpenAIVideoClient, "_request_binary", fake_binary)

    async def exercise() -> None:
        created = await client.create(
            prompt="Render a serene landscape",
            settings={"model": "sora-2", "duration": 8},
            payload={"metadata": {"user": "test"}},
        )
        assert created["id"] == "vid_123"
        create_request = client.last_request
        assert create_request["method"] == "POST"
        assert create_request["path"] == "/v1/videos"
        assert "payload" in create_request
        assert create_request["payload"].get("duration") == 8
        assert "metadata" not in create_request["payload"]
        assert "payload.metadata" in create_request["dropped_fields"]

        retrieved = await client.retrieve("vid_123")
        assert retrieved["status"] == "ready"

        body, headers = await client.content(
            "vid_123", asset_id="primary", params={"quality": "original"}
        )
        assert body == b"video-bytes"
        assert headers["content-type"] == "video/mp4"
        content_request = client.last_request
        assert content_request["method"] == "GET"
        assert content_request["path"] == "/v1/videos/vid_123/content/primary"
        assert "params.quality" in content_request["dropped_fields"]

    _run(exercise())

    assert len(calls) == 2
    method, path, kwargs = calls[0]
    assert method == "POST" and path == "/v1/videos"
    assert "json" in kwargs and kwargs["json"]["prompt"] == "Render a serene landscape"
    assert "metadata" not in kwargs["json"]

    method, path, kwargs = calls[1]
    assert method == "GET" and path == "/v1/videos/vid_123"
    assert kwargs.get("params") is None

    history = client.diagnostic_history
    assert len(history) >= 3
    first_entry = history[0]
    assert first_entry["method"] == "POST"
    assert first_entry["path"] == "/v1/videos"
    assert first_entry["endpoint"] == "/v1/videos"
    assert "payload.metadata" in first_entry["dropped_fields"]
    assert "status" in first_entry and first_entry["status"] >= 0
    assert "duration_ms" in first_entry
    assert "request_id" in first_entry
    last_entry = history[-1]
    assert last_entry["method"] == "GET"
    assert last_entry["path"] == "/v1/videos/vid_123/content/primary"
    assert last_entry["endpoint"] == "/v1/videos/vid_123/content/primary"
    assert "params.quality" in last_entry["dropped_fields"]
    assert "status" in last_entry and last_entry["status"] >= 0
    assert "duration_ms" in last_entry
    assert "request_id" in last_entry


def test_openai_video_list_pagination(monkeypatch: pytest.MonkeyPatch, make_openai_video_config) -> None:
    config = make_openai_video_config(openai_api_version_video="2024-05-01-preview")
    client = OpenAIVideoClient(config=config)

    captured: Dict[str, Any] = {}

    async def fake_request(self, method: str, path: str, **kwargs: Any):
        captured["method"] = method
        captured["path"] = path
        captured["params"] = kwargs.get("params")
        return {"data": []}, 200, 30

    monkeypatch.setattr(BaseProviderClient, "_request", fake_request)

    async def exercise() -> None:
        await client.list(limit="5", order="desc", after="cursor-a", before="cursor-b")

    _run(exercise())

    assert captured["method"] == "GET"
    assert captured["path"] == "/2024-05-01-preview/videos"
    params = captured["params"]
    assert params["limit"] == 5
    assert params["order"] == "desc"
    assert params["after"] == "cursor-a"
    assert "before" not in params
    assert "api-version" not in params
    last_request = client.last_request
    assert last_request["method"] == "GET"
    assert last_request["path"] == "/2024-05-01-preview/videos"
    assert "params" in last_request
    assert "params.before" in last_request["dropped_fields"]


def test_openai_video_models_filtering(monkeypatch: pytest.MonkeyPatch, make_openai_video_config) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)

    async def fake_request(self, method: str, path: str, **kwargs: Any):
        assert method == "GET"
        assert path == "/v1/models"
        return (
            {
                "data": {
                    "models": [
                        {"id": "sora-2"},
                        {"id": "gpt-4.1"},
                        {"id": ""},
                        {"identifier": "missing"},
                    ],
                    "available": ["gpt-4.1"],
                }
            },
            200,
            20,
        )

    monkeypatch.setattr(BaseProviderClient, "_request", fake_request)

    async def exercise() -> list[str]:
        return await client.list_models()

    models = _run(exercise())

    assert models == ["sora-2"]
    supported = client.supported_models
    assert "sora-2" in supported


def test_openai_video_unknown_parameter_error(make_openai_video_config) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)
    client._record_last_request(
        method="POST",
        path="/v1/videos",
        payload={"prompt": "test", "extra": "value"},
        dropped_fields=["payload.extra"],
        correlation_id="diag-1",
    )
    error_payload = {
        "error": {
            "message": "Unknown parameter 'extra' provided",
            "code": "invalid_parameter",
            "param": "extra",
        }
    }
    error = ProviderAPIError(
        provider="openai-video",
        status_code=400,
        message="Bad request",
        error_type="invalid_request_error",
        error_code="invalid_parameter",
        provider_message=json.dumps(error_payload),
        retryable=False,
    )
    normalised = client._normalise_error(error)
    assert isinstance(normalised, ProviderAPIError)
    assert normalised.error_type == "invalid_request"
    assert normalised.error_code == "invalid_request"
    message = str(normalised)
    assert "Unknown parameter" in message
    assert "payload.extra" in message
    history = client.diagnostic_history
    assert history
    diagnostic = history[-1]
    assert diagnostic.get("status_code") == 400
    assert "payload.extra" in diagnostic.get("dropped_fields", ())
    error_info = diagnostic.get("error") or {}
    assert error_info.get("error_code") == "invalid_request"
    assert "payload.extra" in error_info.get("message", "")
    offending = error_info.get("offending_parameters") or ()
    assert "extra" in offending


def test_prepare_create_request_returns_extras(make_openai_video_config) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)

    prepared = client.prepare_create_request(
        prompt="Create a video",
        settings={"duration_seconds": "12", "aspect_ratio": "4:3", "size": "1440x1080"},
        payload={"metadata": {"notes": "test"}},
    )

    assert prepared.request["prompt"] == "Create a video"
    assert prepared.request["duration"] == 12
    assert prepared.request["size"] == "1440x1080"
    assert "metadata" not in prepared.request
    assert prepared.extras["duration_seconds"] == 12
    assert prepared.extras["aspect_ratio"] == "4:3"
    assert prepared.extras["size"] == "1440x1080"
