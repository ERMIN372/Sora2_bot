import logging
from enum import Enum
from types import SimpleNamespace

import pytest

from config import Config
from providers import gemini


class _StubGenAIClient:
    def __init__(self, *args, **kwargs) -> None:
        self.models = SimpleNamespace(list=lambda: [])
        self._http_options = SimpleNamespace(api_version=kwargs.get("http_options", SimpleNamespace()).api_version)


class _StubHttpOptions:
    def __init__(self, *args, **kwargs) -> None:
        self.api_version = kwargs.get("api_version")


class _StubHarmBlockThreshold(Enum):
    BLOCK_NONE = 0
    LOW = 1
    MEDIUM = 2


@pytest.fixture(autouse=True)
def _patch_genai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemini.genai, "Client", _StubGenAIClient)
    monkeypatch.setattr(gemini.types, "HttpOptions", _StubHttpOptions)
    monkeypatch.setattr(gemini.types, "HarmBlockThreshold", _StubHarmBlockThreshold)
    monkeypatch.setattr(gemini.types, "SafetySetting", lambda *args, **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(gemini.GeminiGenerativeClient, "_load_available_models", lambda self: None)
    monkeypatch.setattr(gemini, "_SAFETY_FALLBACK_THRESHOLD", _StubHarmBlockThreshold.LOW)
    monkeypatch.setattr(gemini, "get_gemini_router", lambda _cfg: SimpleNamespace(route=lambda **_: SimpleNamespace(
        task="image",
        model=_cfg.gemini_model_image,
        prompt="ping",
        method="generate_images",
        api_version="v1",
        client=SimpleNamespace(models=SimpleNamespace(generate_images=lambda **_: SimpleNamespace(images=[]))),
        supported_methods=("generate_images",),
        rewrite_notes=None,
    )))


def test_gemini_debug_config_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="providers.gemini")
    config = Config(bot_token="token", gemini_api_key="key", environment="test", debug_gemini=True)
    gemini.GeminiImageClient(config=config)
    assert any("gemini.config" in record.message for record in caplog.records)


def test_empty_image_analysis_debug_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="providers.gemini")
    error = gemini.GeminiImageEmptyError(
        provider="gemini-image",
        finish_reason="NO_IMAGE",
        response_id="resp-123",
        headers={"x-generative-ai-finish-reason": "NO_IMAGE"},
    )
    payload = {"generated_images": [{"rai_filtered_reason": "POLICY"}]}
    gemini._analyse_empty_image_response(payload, response=SimpleNamespace(), error=error)
    assert any("gemini.empty_image.analysis" in record.message for record in caplog.records)
