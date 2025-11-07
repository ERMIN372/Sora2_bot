import asyncio
import logging
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import Config
from providers import gemini


class _StubAPIError(Exception):
    def __init__(self, *, code: int, status: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.message = message
        self.details = None
        self.response = SimpleNamespace(text="", status=status)


class _DummyLimiter:
    async def acquire(self) -> None:
        return None


class _StubHarmBlockThreshold(Enum):
    BLOCK_NONE = 0
    LOW = 1


async def _fake_to_thread(func, *args, **kwargs):
    return func(*args, **kwargs)


async def _fake_sleep(*args, **kwargs) -> None:
    return None


@pytest.fixture(autouse=True)
def _patch_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    gemini.DEBUG_GEMINI = True
    monkeypatch.setattr(gemini.genai_errors, "APIError", _StubAPIError)
    monkeypatch.setattr(gemini.genai, "Client", lambda *args, **kwargs: SimpleNamespace(models=SimpleNamespace()))
    monkeypatch.setattr(gemini.types, "HttpOptions", lambda *args, **kwargs: SimpleNamespace(api_version=kwargs.get("api_version")))
    monkeypatch.setattr(gemini.types, "HarmBlockThreshold", _StubHarmBlockThreshold)
    monkeypatch.setattr(gemini.types, "SafetySetting", lambda *args, **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(gemini, "_SAFETY_FALLBACK_THRESHOLD", _StubHarmBlockThreshold.LOW)
    monkeypatch.setattr(gemini, "AsyncRateLimiter", lambda *args, **kwargs: _DummyLimiter())
    monkeypatch.setattr(gemini.asyncio, "to_thread", _fake_to_thread)
    monkeypatch.setattr(gemini.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(gemini, "ensure_gemini_key_logged", lambda *args, **kwargs: "mask")
    monkeypatch.setattr(gemini.GeminiGenerativeClient, "_load_available_models", lambda self: None)
    default_router = SimpleNamespace(
        route=lambda **kwargs: SimpleNamespace(
            task="image",
            model=kwargs.get("model"),
            prompt=kwargs.get("prompt"),
            method="generate_images",
            api_version="v1",
            client=SimpleNamespace(models=SimpleNamespace()),
            supported_methods=("generate_images", "generate_content"),
            rewrite_notes=None,
        )
    )
    monkeypatch.setattr(gemini, "get_gemini_router", lambda _cfg: default_router)


def _make_client(tmp_path: Path) -> gemini.GeminiImageClient:
    config = Config(bot_token="token", gemini_api_key="key", environment="test", debug_gemini=True)
    client = gemini.GeminiImageClient(config=config)
    client._pending_dir = tmp_path
    return client


class _StubResponse:
    def __init__(self, *, with_images: bool = True) -> None:
        if with_images:
            self.images = [SimpleNamespace(data=b"img-bytes", mime_type="image/png")]
        else:
            self.images = []
        self.response_id = "resp-test"
        self.candidates = [SimpleNamespace(finish_reason="STOP" if with_images else "NO_IMAGE")]
        self.sdk_http_response = SimpleNamespace(headers={"x-generative-ai-finish-reason": "STOP" if with_images else "NO_IMAGE"})

    def model_dump(self, mode: str = "json"):
        finish = self.candidates[0].finish_reason if self.candidates else "STOP"
        return {"candidates": [{"finishReason": finish}]}


def _set_router(client: gemini.GeminiImageClient, factory):
    client._router = SimpleNamespace(route=factory)


def _patch_prepare(client: gemini.GeminiImageClient, generator_factory):
    def _prepare_generate_call(*, decision, prompt, settings, payload, force_mode=None):
        mode = force_mode or decision.method
        generator = generator_factory(decision, mode)
        return generator, {}, {"mode": mode, "api_version": decision.api_version}

    client._prepare_generate_call = _prepare_generate_call  # type: ignore[assignment]


def test_empty_analyse_logs(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    caplog.set_level(logging.DEBUG, logger="providers.gemini")
    client = _make_client(tmp_path)

    def _router(**kwargs):
        version = kwargs.get("forced_api_version") or "v1"
        return SimpleNamespace(
            task="image",
            model=client._model,
            prompt=kwargs.get("prompt"),
            method="generate_images",
            api_version=version,
            client=SimpleNamespace(models=SimpleNamespace()),
            supported_methods=("generate_images",),
            rewrite_notes=None,
        )

    _set_router(client, _router)

    def _generator_factory(decision, mode):
        return lambda **_: _StubResponse(with_images=False)

    _patch_prepare(client, _generator_factory)

    with pytest.raises(gemini.ProviderAPIError):
        asyncio.run(client._submit_single_attempt(prompt="ping"))

    log_text = "\n".join(record.message for record in caplog.records)
    assert "gemini.sync.result.empty" in log_text
    assert "gemini.empty.analyse" in log_text


def test_version_switch_logs(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    caplog.set_level(logging.WARNING, logger="providers.gemini")
    client = _make_client(tmp_path)
    state = {"count": 0}

    def _router(**kwargs):
        version = kwargs.get("forced_api_version") or "v1"
        return SimpleNamespace(
            task="image",
            model=client._model,
            prompt=kwargs.get("prompt"),
            method="generate_images",
            api_version=version,
            client=SimpleNamespace(models=SimpleNamespace()),
            supported_methods=("generate_images",),
            rewrite_notes=None,
        )

    _set_router(client, _router)

    def _generator_factory(decision, mode):
        def _call(**kwargs):
            state["count"] += 1
            if state["count"] == 1:
                raise gemini.genai_errors.APIError(code=404, status="NOT_FOUND", message="missing")
            return _StubResponse(with_images=True)

        return _call

    _patch_prepare(client, _generator_factory)

    asyncio.run(client._submit_single_attempt(prompt="ping"))

    assert any("gemini.api.version_switch" in record.message for record in caplog.records)


def test_assets_summary_logs(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    caplog.set_level(logging.INFO, logger="providers.gemini")
    client = _make_client(tmp_path)

    def _router(**kwargs):
        version = kwargs.get("forced_api_version") or "v1"
        return SimpleNamespace(
            task="image",
            model=client._model,
            prompt=kwargs.get("prompt"),
            method="generate_images",
            api_version=version,
            client=SimpleNamespace(models=SimpleNamespace()),
            supported_methods=("generate_images",),
            rewrite_notes=None,
        )

    _set_router(client, _router)

    def _generator_factory(decision, mode):
        return lambda **_: _StubResponse(with_images=True)

    _patch_prepare(client, _generator_factory)

    asyncio.run(client._submit_single_attempt(prompt="ping"))

    assert any("gemini.image.assets" in record.message for record in caplog.records)
