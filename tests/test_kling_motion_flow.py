import asyncio
from types import SimpleNamespace

from handlers._core import UserSession, _build_video_settings_keyboard
from providers.kling_video import KlingVideoClient
from utils import build_telegram_file_url


class _FakeBot:
    _token = "12345:abc"

    async def get_file(self, file_id: str):
        assert file_id == "file123"
        return SimpleNamespace(file_path="videos/ref.mp4")


def test_kling_video_settings_keyboard_back_to_model() -> None:
    keyboard = _build_video_settings_keyboard(
        UserSession(),
        "photo",
        provider="kling",
        product="kling_mc",
        model="kling-v2-6-motion",
    )

    assert keyboard.inline_keyboard[-1][0].callback_data == "flow:back:video_model"


def test_non_kling_video_settings_keyboard_back_to_mode() -> None:
    keyboard = _build_video_settings_keyboard(
        UserSession(),
        "text",
        provider="sora",
        product="sora",
        model="sora-2",
    )

    assert keyboard.inline_keyboard[-1][0].callback_data == "flow:back:video_mode"


def test_kling_resolve_motion_video_url_prefers_explicit_url() -> None:
    url = KlingVideoClient._resolve_motion_video_url(
        {
            "motion_video_url": "https://example.com/ref.mp4",
            "motion_video_inline_data": {
                "mime_type": "video/mp4",
                "data": "ZmFrZV9kYXRh",
            },
        }
    )

    assert url == "https://example.com/ref.mp4"


def test_kling_resolve_motion_video_url_from_inline_data() -> None:
    url = KlingVideoClient._resolve_motion_video_url(
        {
            "motion_video_inline_data": {
                "mime_type": "video/mp4",
                "data": "ZmFrZV9kYXRh",
            }
        }
    )

    assert url == "data:video/mp4;base64,ZmFrZV9kYXRh"


def test_kling_resolve_image_url_from_inline_data_returns_raw_base64() -> None:
    image_value = KlingVideoClient._resolve_image_url(
        {
            "reference_inline_data": {
                "mime_type": "image/jpeg",
                "data": "aW1hZ2VfYmFzZTY0",
            }
        }
    )

    assert image_value == "aW1hZ2VfYmFzZTY0"


def test_build_telegram_file_url() -> None:
    bot = _FakeBot()
    url = asyncio.run(build_telegram_file_url(bot, "file123"))

    assert url == "https://api.telegram.org/file/bot12345:abc/videos/ref.mp4"


def test_build_telegram_file_url_with_public_token_attr() -> None:
    class _FakeBotPublicToken(_FakeBot):
        token = "99999:xyz"

    bot = _FakeBotPublicToken()
    url = asyncio.run(build_telegram_file_url(bot, "file123"))

    assert url == "https://api.telegram.org/file/bot99999:xyz/videos/ref.mp4"


def test_kling_extract_status_for_fal_queue() -> None:
    client = KlingVideoClient.__new__(KlingVideoClient)

    assert client._extract_status({"status": "IN_PROGRESS"}) == "running"
    assert client._extract_status({"status": "COMPLETED"}) == "completed"


def test_kling_extract_assets_for_fal_output() -> None:
    client = KlingVideoClient.__new__(KlingVideoClient)

    assets = client._extract_assets(
        {
            "output": {
                "video": {
                    "url": "https://cdn.example/video.mp4",
                }
            }
        }
    )

    assert assets["video"] == "https://cdn.example/video.mp4"
