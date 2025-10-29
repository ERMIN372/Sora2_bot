import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from config import Config
from providers.sora_video import SoraVideoClient


@pytest.fixture()
def sora_config() -> Config:
    return Config(
        bot_token="test-token",
        gemini_api_key="",
        openai_api_key="sk-test",
        sora_api_key="sk-test",
        sora_model_video="sora",
    )


def test_enqueue_job_builds_payload(sora_config: Config) -> None:
    client = SoraVideoClient(config=sora_config)
    async def scenario() -> None:
        with patch.object(
            client,
            "_request",
            new=AsyncMock(return_value=({"id": "resp-1", "status": "in_progress"}, 200, 42)),
        ) as mock_request:
            submission = await client.enqueue_job(
                prompt="Make a cool video",
                settings={"size": "1280x720", "duration_sec": 6},
                idempotency_key="idem-1",
            )

            assert submission.job_id == "resp-1"
            mock_request.assert_awaited_once()
            call_args = mock_request.await_args
            assert call_args.args[0] == "POST"
            assert call_args.args[1] == "/responses"
            payload = call_args.kwargs["json"]
            assert payload["model"] == "sora"
            assert payload["input"][0]["content"][0]["text"] == "Make a cool video"
            assert "modalities" not in payload
            assert "tools" not in payload
            assert "tool_choice" not in payload
            assert "response_format" not in payload
            assert payload["metadata"]["aspect_ratio"] == "16:9"
            assert payload["metadata"]["duration_sec"] == "6"

    asyncio.run(scenario())


def test_get_job_status_uses_cache(sora_config: Config) -> None:
    client = SoraVideoClient(config=sora_config)
    client._cache["resp-2"] = {
        "id": "resp-2",
        "status": "completed",
        "output": [
            {
                "content": [
                    {
                        "type": "output_file",
                        "file_id": "file-1",
                        "url": "https://example.com/video.mp4",
                        "mime_type": "video/mp4",
                    }
                ]
            }
        ],
    }
    async def scenario() -> None:
        with patch.object(client, "_request", new=AsyncMock()) as mock_request:
            status = await client.get_job_status("resp-2")

            assert status.status == "completed"
            assert status.assets["file-1"] == "https://example.com/video.mp4"
            mock_request.assert_not_awaited()

    asyncio.run(scenario())


def test_get_job_status_fetches_remote(sora_config: Config) -> None:
    client = SoraVideoClient(config=sora_config)
    async def scenario() -> None:
        with patch.object(
            client,
            "_request",
            new=AsyncMock(
                return_value=(
                    {"id": "resp-3", "status": "failed", "error": {"message": "bad"}},
                    500,
                    12,
                )
            ),
        ) as mock_request:
            status = await client.get_job_status("resp-3")

            assert status.status == "failed"
            assert status.error == "bad"
            assert status.status_code == 500
            mock_request.assert_awaited_once_with("GET", "/responses/resp-3")

    asyncio.run(scenario())
