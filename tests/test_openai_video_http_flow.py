"""Tests covering the OpenAI video client's HTTP-facing helpers."""
from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from providers.base import BaseProviderClient
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
            assert payload.get("metadata", {}).get("duration_sec") == "8"
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

    async def fake_binary(self, method: str, path: str, params=None):
        assert method == "GET"
        assert path == "/videos/vid_123/content/primary"
        assert params == {"quality": "original"}
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

        retrieved = await client.retrieve("vid_123")
        assert retrieved["status"] == "ready"

        body, headers = await client.content(
            "vid_123", asset_id="primary", params={"quality": "original"}
        )
        assert body == b"video-bytes"
        assert headers["content-type"] == "video/mp4"

    _run(exercise())

    assert len(calls) == 2
    method, path, kwargs = calls[0]
    assert method == "POST" and path == "/videos"
    assert "json" in kwargs and kwargs["json"]["prompt"] == "Render a serene landscape"

    method, path, kwargs = calls[1]
    assert method == "GET" and path == "/videos/vid_123"
    assert kwargs.get("params") is None


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
    assert params["before"] == "cursor-b"
    assert params["api-version"] == "2024-05-01-preview"


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
