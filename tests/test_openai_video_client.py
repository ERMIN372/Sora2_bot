from config import Config
from providers.openai_video import OpenAIVideoClient


def _make_config(**overrides: object) -> Config:
    base = {
        "bot_token": "test-token",
        "gemini_api_key": "",
        "openai_api_key": "sk-test",
        "sora_api_key": "sk-test",
        "openai_api_base": "https://api.openai.com/v1",
    }
    base.update(overrides)
    return Config(**base)  # type: ignore[arg-type]


def test_openai_video_client_applies_custom_headers() -> None:
    config = _make_config(
        openai_beta_header="video=custom",
        openai_org_id="org-custom",
        openai_api_version_video="2024-05-01-preview",
    )
    client = OpenAIVideoClient(config=config)

    headers = client._build_headers()
    assert headers["OpenAI-Beta"] == "video=custom"
    assert headers["OpenAI-Organization"] == "org-custom"

    merged = client._merge_params({"foo": "bar"})
    assert merged is not None
    assert merged["api-version"] == "2024-05-01-preview"
    assert merged["foo"] == "bar"


def test_openai_video_client_uses_default_beta_header() -> None:
    config = _make_config(openai_beta_header="")
    client = OpenAIVideoClient(config=config)

    headers = client._build_headers()
    assert headers["OpenAI-Beta"] == "video=1"
