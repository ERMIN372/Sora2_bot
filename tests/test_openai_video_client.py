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


def test_openai_video_client_get_diagnostics_includes_error_snapshots() -> None:
    config = _make_config(openai_beta_header="video=2", openai_org_id="org-diag")
    client = OpenAIVideoClient(config=config)

    client._record_last_request(method="GET", path="/videos", dropped_fields=[], correlation_id=None)
    client._update_last_diagnostic(
        status_code=429, error={"error_code": "rate_limit", "message": "Too many requests"}
    )
    diagnostics = client.get_diagnostics()

    assert diagnostics["beta_header"] == "video=2"
    assert diagnostics["organization_id"] == "org-diag"
    assert diagnostics["api_version"] == ""

    headers = diagnostics["headers"]
    assert headers["OpenAI-Beta"] == "video=2"
    assert "Authorization" in headers and headers["Authorization"] != "Bearer sk-test"

    snapshots = diagnostics["error_snapshots"]
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["status_code"] == 429
    assert snapshot["error_code"] == "rate_limit"
    assert snapshot["message"] == "Too many requests"
