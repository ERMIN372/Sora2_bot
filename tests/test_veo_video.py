import base64
from types import SimpleNamespace

import pytest

from config import (
    Config,
    DEFAULT_CREDIT_PRICE_RUB,
    DEFAULT_FIX_FEE_RUB,
    DEFAULT_MARKUP_PCT,
    PricingConfig,
)
from providers.veo_video import VeoVideoClient


@pytest.fixture()
def config() -> Config:
    pricing = PricingConfig(
        credit_price_rub=DEFAULT_CREDIT_PRICE_RUB,
        markup_pct=DEFAULT_MARKUP_PCT,
        fix_fee_rub=DEFAULT_FIX_FEE_RUB,
        provider_costs={},
    )
    return Config(
        bot_token="test-token",
        gemini_api_key="test-key",
        pricing=pricing,
    )


@pytest.fixture(autouse=True)
def stub_gemini_client(monkeypatch):
    class DummyModels:
        def generate_videos(self, *args, **kwargs):  # pragma: no cover - not used
            raise AssertionError("generate_videos should not be called in tests")

    class DummyClient:
        def __init__(self):
            self.models = DummyModels()
            self.operations = SimpleNamespace(get=lambda operation: operation)

    monkeypatch.setattr(
        "providers.veo_video.get_gemini_client", lambda config: DummyClient()
    )
    return None


def test_build_config_infers_and_validates(config: Config) -> None:
    client = VeoVideoClient(config=config)
    config_obj, summary = client._build_config(
        {
            "size": "1280x720",
            "duration_seconds": "8",
            "fps": "30",
            "seed": "42",
            "enhance_prompt": "true",
            "generate_audio": "false",
            "negative_prompt": "no watermarks",
        }
    )

    assert config_obj is not None
    assert config_obj.aspect_ratio == "16:9"
    assert config_obj.resolution == "1080p"
    assert config_obj.duration_seconds == 8
    assert config_obj.fps == 30
    assert config_obj.seed == 42
    assert config_obj.enhance_prompt is True
    assert config_obj.generate_audio is False
    assert summary["size"] == "1280x720"
    assert summary["duration_seconds"] == 8
    assert summary["negative_prompt"] == "no watermarks"


def test_build_config_handles_empty_input(config: Config) -> None:
    client = VeoVideoClient(config=config)
    config_obj, summary = client._build_config({})

    assert config_obj is None
    assert summary == {}


def test_build_source_supports_inline_reference(config: Config) -> None:
    client = VeoVideoClient(config=config)
    inline = {
        "mime": "image/png",
        "data": base64.b64encode(b"image-bytes").decode("ascii"),
    }

    source = client._build_source("", {"reference_inline_data": inline})
    assert source.prompt is None
    assert source.image is not None


def test_build_source_requires_prompt_or_image(config: Config) -> None:
    client = VeoVideoClient(config=config)
    with pytest.raises(RuntimeError):
        client._build_source("", {})
