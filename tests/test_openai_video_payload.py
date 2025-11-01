"""Payload sanitisation tests for the OpenAI video client."""
from __future__ import annotations

import asyncio
import json

import pytest

from providers.base import BaseProviderClient, ProviderAPIError
from providers.openai_video import OpenAIVideoClient


def _run(coro):
    return asyncio.run(coro)


def test_create_filters_extraneous_fields(monkeypatch: pytest.MonkeyPatch, make_openai_video_config) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)

    calls = []

    async def fake_request(self, method: str, path: str, **kwargs):
        calls.append((method, path, kwargs))
        assert method == "POST"
        assert path == "/v1beta/videos"
        payload = kwargs.get("json") or {}
        assert payload["prompt"] == "Clean this"
        assert payload["model"] == "sora-2"
        assert payload["duration"] == 5
        assert payload["n"] == 1
        assert payload["response_format"] == "url"
        assert "metadata" not in payload
        assert "foo" not in payload
        assert "bar" not in payload
        return {"id": "vid_filtered", "status": "queued"}, 201, 32

    monkeypatch.setattr(BaseProviderClient, "_request", fake_request)

    async def exercise() -> None:
        created = await client.create(
            prompt="Clean this",
            settings={"model": "sora-2", "foo": "ignored"},
            payload={"seconds": 5, "metadata": {"user": "abc"}, "bar": "baz"},
        )
        assert created["id"] == "vid_filtered"

    _run(exercise())

    assert len(calls) == 1
    request = client.last_request
    assert request["method"] == "POST"
    assert request["path"] == "/v1beta/videos"
    assert request["payload"]["duration"] == 5
    assert request["payload"]["model"] == "sora-2"
    assert request["payload"]["n"] == 1
    assert request["payload"]["response_format"] == "url"
    assert "metadata" not in request["payload"]
    assert "foo" not in request["payload"]
    assert "bar" not in request["payload"]
    assert set(request["dropped_fields"]) == {
        "payload.bar",
        "payload.metadata",
        "payload.seconds",
        "settings.foo",
    }

    history = client.diagnostic_history
    assert history
    diagnostic = history[-1]
    assert diagnostic["status"] == 201
    assert set(diagnostic["dropped_fields"]) == {
        "payload.bar",
        "payload.metadata",
        "payload.seconds",
        "settings.foo",
    }


def test_create_unknown_parameter_error(monkeypatch: pytest.MonkeyPatch, make_openai_video_config) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)

    async def fake_request(self, method: str, path: str, **kwargs):
        assert method == "POST"
        assert path == "/v1beta/videos"
        raise ProviderAPIError(
            provider="openai-video",
            status_code=400,
            message="Bad request",
            error_type="invalid_request_error",
            error_code="invalid_parameter",
            provider_message=json.dumps(
                {
                    "error": {
                        "message": "Unknown parameter 'unexpected' provided",
                        "code": "invalid_parameter",
                        "param": "unexpected",
                    }
                }
            ),
            retryable=False,
        )

    monkeypatch.setattr(BaseProviderClient, "_request", fake_request)

    async def exercise() -> None:
        with pytest.raises(ProviderAPIError) as excinfo:
            await client.create(
                prompt="hello", settings={"model": "sora-2"}, payload={"unexpected": "value"}
            )
        error = excinfo.value
        assert error.error_type == "invalid_request"
        assert error.error_code == "invalid_request"
        assert "Unknown parameter" in str(error)

    _run(exercise())

    request = client.last_request
    assert set(request["dropped_fields"]) >= {"payload.unexpected"}
    diagnostic = client.diagnostic_history[-1]
    assert diagnostic["status_code"] == 400
    assert "payload.unexpected" in diagnostic["dropped_fields"]
    error_info = diagnostic.get("error") or {}
    assert error_info.get("error_type") == "invalid_request"
    offending = error_info.get("offending_parameters") or ()
    assert "unexpected" in offending
