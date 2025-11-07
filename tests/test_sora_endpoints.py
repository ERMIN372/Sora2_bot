import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from config import Config
from providers.base import _join_url
from providers.base import BaseProviderClient
from providers.sora_video import SoraVideoClient


@pytest.mark.parametrize(
    "base, parts, expected",
    [
        ("https://api.openai.com", ["v1", "videos"], "https://api.openai.com/v1/videos"),
        (
            "https://api.openai.com/v1",
            ["v1", "videos"],
            "https://api.openai.com/v1/videos",
        ),
        ("https://api.openai.com", ["v1beta", "videos"], "https://api.openai.com/v1beta/videos"),
    ],
)
def test_join_url_normalises_versions(base: str, parts: list[str], expected: str) -> None:
    assert _join_url(base, *parts) == expected


def test_sora_request_uses_normalised_url() -> None:
    config = Config(
        bot_token="test-token",
        gemini_api_key="",
        openai_api_key="sk-test",
        sora_api_key="sk-test",
        sora_model_video="sora-2",
        openai_api_base="https://api.openai.com/v1",
        openai_api_version_video="v1beta",
    )
    client = SoraVideoClient(config=config)

    async def scenario() -> str:
        client._rate_limiter.acquire = AsyncMock(return_value=None)  # type: ignore[assignment]
        mock_perform = AsyncMock(return_value=({}, 200, 10))
        with patch.object(BaseProviderClient, "_perform_request", new=mock_perform):
            await client._request("POST", ("v1", "videos"), json={})
        await client.close()
        return mock_perform.await_args.args[1]

    called_url = asyncio.run(scenario())
    assert called_url == "https://api.openai.com/v1/videos"
