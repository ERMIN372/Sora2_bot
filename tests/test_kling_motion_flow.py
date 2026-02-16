import asyncio
from types import SimpleNamespace

import pytest

from handlers._core import UserSession, _build_video_settings_keyboard
from providers.base import ProviderAPIError
from providers.kling_video import (
    FAL_KLING_MODEL,
    KlingVideoClient,
    _as_bool,
    _normalise_fal_key,
)
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


def test_kling_resolve_image_url_prefers_explicit_url() -> None:
    image_value = KlingVideoClient._resolve_image_url(
        {
            "image_url": "https://example.com/ref.png",
            "reference_inline_data": {
                "mime_type": "image/jpeg",
                "data": "aW1hZ2VfYmFzZTY0",
            },
        }
    )

    assert image_value == "https://example.com/ref.png"


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


def test_kling_enqueue_maps_balance_error_to_billing() -> None:
    class _FailingKlingClient(KlingVideoClient):
        async def _request(self, *_args, **_kwargs):  # type: ignore[override]
            raise ProviderAPIError(
                provider="kling",
                status_code=0,
                message="Account balance not enough",
                error_type="provider_unavailable",
                provider_message="Account balance not enough",
            )

    client = _FailingKlingClient.__new__(_FailingKlingClient)

    with pytest.raises(ProviderAPIError) as exc_info:
        asyncio.run(
            client.enqueue_job(
                prompt="test",
                settings={
                    "motion_video_url": "https://example.com/ref.mp4",
                    "reference_inline_data": {"data": "abc"},
                },
            )
        )

    error = exc_info.value
    assert error.error_type == "billing"
    assert error.error_code == "insufficient_balance"
    assert error.status_code == 402


def test_normalise_fal_key_accepts_prefixed_value() -> None:
    assert _normalise_fal_key('"Key fal_test_123"') == "fal_test_123"


def test_kling_as_bool_handles_string_values() -> None:
    assert _as_bool("false", default=True) is False
    assert _as_bool("yes", default=False) is True


def test_kling_enqueue_recovers_from_legacy_jwt_name_error() -> None:
    class _NameErrorKlingClient(KlingVideoClient):
        async def _request(self, *_args, **_kwargs):  # type: ignore[override]
            raise NameError("name '_generate_jwt' is not defined")

        async def _request_via_fal_http(  # type: ignore[override]
            self, *, method: str, path: str, json_payload=None
        ):
            assert method == "POST"
            assert path == f"/{FAL_KLING_MODEL}"
            assert isinstance(json_payload, dict)
            return {"request_id": "req_fallback"}, 200, 123

    client = _NameErrorKlingClient.__new__(_NameErrorKlingClient)
    client._cache = {}

    submission = asyncio.run(
        client.enqueue_job(
            prompt="test",
            settings={
                "motion_video_url": "https://example.com/ref.mp4",
                "reference_inline_data": {"data": "abc"},
            },
        )
    )

    assert submission.job_id == "req_fallback"
    assert submission.status_code == 200
    assert submission.duration_ms == 123


def test_kling_status_recovers_from_legacy_jwt_name_error() -> None:
    class _NameErrorStatusClient(KlingVideoClient):
        async def _request(self, *_args, **_kwargs):  # type: ignore[override]
            raise NameError("name '_generate_jwt' is not defined")

        async def _request_via_fal_http(  # type: ignore[override]
            self, *, method: str, path: str, json_payload=None
        ):
            assert method == "GET"
            assert json_payload is None
            if path.endswith("/status"):
                return {"status": "COMPLETED"}, 200, 50
            return (
                {"status": "COMPLETED", "output": {"video": {"url": "https://cdn.example/v.mp4"}}},
                200,
                70,
            )

    client = _NameErrorStatusClient.__new__(_NameErrorStatusClient)
    client._cache = {}

    status = asyncio.run(client.get_job_status("req_status"))

    assert status.status == "completed"
    assert status.assets["video"] == "https://cdn.example/v.mp4"


def test_kling_direct_fallback_retries_without_status_on_405() -> None:
    class _FakeResponse:
        def __init__(self, status: int, text: str):
            self.status = status
            self._text = text

        async def text(self) -> str:
            return self._text

    class _ResponseCtx:
        def __init__(self, response: _FakeResponse):
            self._response = response

        async def __aenter__(self):
            return self._response

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakeSession:
        def __init__(self):
            self.calls = []

        def request(self, method, url, headers=None, **kwargs):
            self.calls.append((method, url, kwargs))
            if len(self.calls) == 1:
                return _ResponseCtx(_FakeResponse(405, "405: Method Not Allowed"))
            return _ResponseCtx(_FakeResponse(200, '{"status":"COMPLETED"}'))

    class _DirectClient(KlingVideoClient):
        async def _ensure_session(self):  # type: ignore[override]
            return self._session

    client = _DirectClient.__new__(_DirectClient)
    client._base_url = "https://queue.fal.run"
    client._fal_key = "fal_test"
    client._session = _FakeSession()

    data, status_code, _ = asyncio.run(
        client._request_via_fal_http(
            method="GET",
            path=f"/{FAL_KLING_MODEL}/requests/abc/status",
            json_payload=None,
        )
    )

    assert status_code == 200
    assert data["status"] == "COMPLETED"
    assert client._session.calls[0][1].endswith("/status")
    assert client._session.calls[1][1].endswith("/requests/abc")
    assert "json" not in client._session.calls[0][2]
    assert "json" not in client._session.calls[1][2]
