"""Tests covering the OpenAI video client's HTTP-facing helpers."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

import pytest

from providers.base import BaseProviderClient, ProviderAPIError
from providers.openai_video import OpenAIVideoClient


def _run(coro):
    return asyncio.run(coro)


def test_openai_video_full_cycle(monkeypatch: pytest.MonkeyPatch, make_openai_video_config) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)

    calls: list[tuple[str, str, Dict[str, Any]]] = []

    async def fake_request(self, method: str, path: str, **kwargs: Any):
        calls.append((method, path, kwargs))
        if method == "POST" and path == "/videos":
            payload = kwargs.get("json") or {}
            assert payload["prompt"] == "Render a serene landscape"
            assert payload["model"] == "sora-2"
            assert payload.get("seconds") == "8"
            assert "metadata" not in payload
            return {"id": "vid_123", "status": "queued"}, 201, 120
        if method == "GET" and path == "/videos/vid_123":
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
        assert path == "/videos/vid_123/content/primary"
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
        assert create_request["path"] == "/videos"
        assert "payload" in create_request
        assert create_request["payload"].get("seconds") == "8"
        assert "metadata" not in create_request["payload"]
        assert "payload.metadata" in create_request["dropped_fields"]
        assert "settings.duration" in create_request["dropped_fields"]

        retrieved = await client.retrieve("vid_123")
        assert retrieved["status"] == "ready"

        body, headers = await client.content(
            "vid_123", asset_id="primary", params={"quality": "original"}
        )
        assert body == b"video-bytes"
        assert headers["content-type"] == "video/mp4"
        content_request = client.last_request
        assert content_request["method"] == "GET"
        assert content_request["path"] == "/videos/vid_123/content/primary"
        assert "params.quality" in content_request["dropped_fields"]

    _run(exercise())

    assert len(calls) == 2
    method, path, kwargs = calls[0]
    assert method == "POST" and path == "/videos"
    assert "json" in kwargs and kwargs["json"]["prompt"] == "Render a serene landscape"
    assert "metadata" not in kwargs["json"]

    method, path, kwargs = calls[1]
    assert method == "GET" and path == "/videos/vid_123"
    assert kwargs.get("params") is None

    history = client.diagnostic_history
    assert len(history) >= 3
    first_entry = history[0]
    assert first_entry["method"] == "POST"
    assert first_entry["path"] == "/videos"
    assert first_entry["endpoint"] == "/videos"
    assert "payload.metadata" in first_entry["dropped_fields"]
    assert "status" in first_entry and first_entry["status"] >= 0
    assert "duration_ms" in first_entry
    assert "request_id" in first_entry
    last_entry = history[-1]
    assert last_entry["method"] == "GET"
    assert last_entry["path"] == "/videos/vid_123/content/primary"
    assert last_entry["endpoint"] == "/videos/vid_123/content/primary"
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
    assert captured["path"] == "/videos"
    params = captured["params"]
    assert params["limit"] == 5
    assert params["order"] == "desc"
    assert params["after"] == "cursor-a"
    assert "before" not in params
    assert params["api-version"] == "2024-05-01-preview"
    last_request = client.last_request
    assert last_request["method"] == "GET"
    assert last_request["path"] == "/videos"
    assert "params" in last_request
    assert "params.before" in last_request["dropped_fields"]


def test_openai_video_models_filtering(monkeypatch: pytest.MonkeyPatch, make_openai_video_config) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)

    async def fake_request(self, method: str, path: str, **kwargs: Any):
        assert method == "GET"
        assert path == "/models"
        return (
            {
                "data": [
                    {"id": "sora-2"},
                    {"id": "gpt-4.1"},
                    {"id": ""},
                    {"identifier": "missing"},
                    "not-a-dict",
                ]
            },
            200,
            20,
        )

    monkeypatch.setattr(BaseProviderClient, "_request", fake_request)

    async def exercise() -> list[str]:
        return await client.list_models()

    models = _run(exercise())

    assert models == ["sora-2", "gpt-4.1"]
    supported = client.supported_models
    assert "sora-2" in supported
    assert "gpt-4.1" in supported


def test_openai_video_unknown_parameter_error(make_openai_video_config) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)
    client._record_last_request(
        method="POST",
        path="/videos",
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
    assert prepared.request["seconds"] == "12"
    assert prepared.request["size"] == "1440x1080"
    assert "metadata" not in prepared.request
    assert prepared.extras["duration_seconds"] == 12
    assert prepared.extras["aspect_ratio"] == "4:3"
    assert prepared.extras["size"] == "1440x1080"
