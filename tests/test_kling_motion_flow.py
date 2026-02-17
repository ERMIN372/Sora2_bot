import asyncio
from types import SimpleNamespace

import pytest

from handlers._core import UserSession, _build_video_settings_keyboard
from providers.base import ProviderAPIError
from providers.kling_video import (
    FAL_KLING_MODEL,
    FAL_KLING_MODEL_BASE,
    FAL_KLING_MODEL_PRO,
    FAL_KLING_MODEL_STANDARD,
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

        async def _validate_public_asset_url(self, *, url: str, kind: str) -> None:  # type: ignore[override]
            return None

        async def _probe_video_duration_seconds(self, url: str) -> float:  # type: ignore[override]
            return 8.0

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


def test_kling_enqueue_maps_missing_ffprobe_to_provider_unavailable() -> None:
    class _FailingKlingClient(KlingVideoClient):
        async def _request(self, *_args, **_kwargs):  # type: ignore[override]
            raise ProviderAPIError(
                provider="kling",
                status_code=500,
                message="ffprobe is required for Kling Motion Control duration validation",
                error_type="validation",
            )

        async def _validate_public_asset_url(self, *, url: str, kind: str) -> None:  # type: ignore[override]
            return None

        async def _probe_video_duration_seconds(self, url: str) -> float:  # type: ignore[override]
            return 8.0

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
    assert error.error_type == "provider_unavailable"
    assert error.error_code == "ffprobe_missing"
    assert "ffprobe" in error.message.lower()


def test_kling_enqueue_maps_missing_ffprobe_from_probe_stage() -> None:
    class _ProbeFailClient(KlingVideoClient):
        async def _validate_public_asset_url(self, *, url: str, kind: str) -> None:  # type: ignore[override]
            return None

        async def _probe_video_duration_seconds(self, url: str) -> float:  # type: ignore[override]
            raise ProviderAPIError(
                provider="kling",
                status_code=500,
                message="ffprobe is required for Kling Motion Control duration validation",
                error_type="validation",
            )

    client = _ProbeFailClient.__new__(_ProbeFailClient)

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
    assert error.error_type == "provider_unavailable"
    assert error.error_code == "ffprobe_missing"
    assert "ffprobe" in error.message.lower()


def test_normalise_fal_key_accepts_prefixed_value() -> None:
    assert _normalise_fal_key('"Key fal_test_123"') == "fal_test_123"


def test_kling_as_bool_handles_string_values() -> None:
    assert _as_bool("false", default=True) is False
    assert _as_bool("yes", default=False) is True


def test_kling_enqueue_sends_bool_keep_original_sound() -> None:
    class _CaptureClient(KlingVideoClient):
        async def _request_with_legacy_fallback(self, _method, _path, *, json_payload=None):  # type: ignore[override]
            self.last_json = json_payload
            return {"request_id": "req_bool"}, 200, 10

        async def _validate_public_asset_url(self, *, url: str, kind: str) -> None:  # type: ignore[override]
            return None

        async def _probe_video_duration_seconds(self, url: str) -> float:  # type: ignore[override]
            return 8.0

    client = _CaptureClient.__new__(_CaptureClient)
    client._cache = {}
    client._submit_models = {}

    submission = asyncio.run(
        client.enqueue_job(
            prompt="test",
            settings={
                "image_url": "https://example.com/ref.png",
                "motion_video_url": "https://example.com/ref.mp4",
                "keep_original_sound": "false",
            },
        )
    )

    assert submission.job_id == "req_bool"
    payload = client.last_json["input"]
    assert payload["keep_original_sound"] is False
    assert payload["idempotency_key"]


def test_kling_enqueue_default_keep_original_sound_true() -> None:
    class _CaptureClient(KlingVideoClient):
        async def _request_with_legacy_fallback(self, _method, _path, *, json_payload=None):  # type: ignore[override]
            self.last_json = json_payload
            return {"request_id": "req_default_sound"}, 200, 12

        async def _validate_public_asset_url(self, *, url: str, kind: str) -> None:  # type: ignore[override]
            return None

        async def _probe_video_duration_seconds(self, url: str) -> float:  # type: ignore[override]
            return 7.0

    client = _CaptureClient.__new__(_CaptureClient)
    client._cache = {}
    client._submit_models = {}

    asyncio.run(
        client.enqueue_job(
            prompt="test",
            settings={
                "image_url": "https://example.com/ref.png",
                "motion_video_url": "https://example.com/ref.mp4",
            },
        )
    )

    payload = client.last_json["input"]
    assert payload["keep_original_sound"] is True
    assert payload["idempotency_key"]


def test_kling_enqueue_uses_provided_idempotency_key() -> None:
    class _CaptureClient(KlingVideoClient):
        async def _request_with_legacy_fallback(self, _method, _path, *, json_payload=None):  # type: ignore[override]
            self.last_json = json_payload
            return {"request_id": "req_idempotent"}, 200, 10

        async def _validate_public_asset_url(self, *, url: str, kind: str) -> None:  # type: ignore[override]
            return None

        async def _probe_video_duration_seconds(self, url: str) -> float:  # type: ignore[override]
            return 8.0

    client = _CaptureClient.__new__(_CaptureClient)
    client._cache = {}
    client._submit_models = {}

    asyncio.run(
        client.enqueue_job(
            prompt="test",
            idempotency_key="job-123",
            settings={
                "image_url": "https://example.com/ref.png",
                "motion_video_url": "https://example.com/ref.mp4",
            },
        )
    )

    payload = client.last_json["input"]
    assert payload["idempotency_key"] == "job-123"


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

        async def _validate_public_asset_url(self, *, url: str, kind: str) -> None:  # type: ignore[override]
            return None

        async def _probe_video_duration_seconds(self, url: str) -> float:  # type: ignore[override]
            return 8.0

    client = _NameErrorKlingClient.__new__(_NameErrorKlingClient)
    client._cache = {}
    client._submit_models = {}

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
            if path.endswith("/status"):
                assert method == "GET"
            else:
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
    client._submit_models = {}
    client._base_url = "https://queue.fal.run"

    status = asyncio.run(client.get_job_status("req_status"))

    assert status.status == "completed"
    assert status.assets["video"] == "https://cdn.example/v.mp4"


def test_kling_status_uses_get_with_submit_model_path() -> None:
    """Status polling must use GET and the full submit model path."""

    class _StatusClient(KlingVideoClient):
        async def _request_with_legacy_fallback(  # type: ignore[override]
            self, method: str, path: str, *, json_payload=None
        ):
            self.calls.append((method, path, json_payload))
            return {"status": "IN_PROGRESS"}, 200, 52

    client = _StatusClient.__new__(_StatusClient)
    client._cache = {}
    client._submit_models = {}
    client._base_url = "https://queue.fal.run"
    client.calls = []

    result = asyncio.run(client.get_job_status("req-poll"))

    assert result.status == "running"
    assert result.assets == {}
    # Falls back to FAL_KLING_MODEL_STANDARD when submit model is unknown
    assert client.calls[0][:2] == ("GET", f"/{FAL_KLING_MODEL_STANDARD}/requests/req-poll/status")
    assert len(client.calls) == 1


def test_kling_status_ignores_cached_status_url_uses_submit_model_path() -> None:
    """Cached status_url from enqueue is ignored; submit model path is always used.

    fal.ai normalises status_url to the base "fal-ai/kling-video" which resolves
    to a different endpoint that validates motion-control body params → 422.
    """

    class _CachedUrlClient(KlingVideoClient):
        async def _request_with_legacy_fallback(  # type: ignore[override]
            self, method: str, path: str, *, json_payload=None
        ):
            self.calls.append((method, path, json_payload))
            return {"status": "IN_PROGRESS"}, 200, 40

    client = _CachedUrlClient.__new__(_CachedUrlClient)
    client._base_url = "https://queue.fal.run"
    client._cache = {
        "req-cached": {
            "status": "IN_QUEUE",
            "status_url": "https://queue.fal.run/fal-ai/kling-video/requests/req-cached/status",
            "response_url": "https://queue.fal.run/fal-ai/kling-video/requests/req-cached",
        }
    }
    client._submit_models = {"req-cached": FAL_KLING_MODEL_PRO}
    client.calls = []

    result = asyncio.run(client.get_job_status("req-cached"))

    assert result.status == "running"
    # Must use submit model path, NOT any cached status_url value or base path.
    assert client.calls[0][:2] == ("GET", f"/{FAL_KLING_MODEL_PRO}/requests/req-cached/status")
    assert len(client.calls) == 1


def test_kling_submit_model_selection() -> None:
    assert KlingVideoClient._pick_submit_model("std") == FAL_KLING_MODEL_STANDARD
    assert KlingVideoClient._pick_submit_model("pro") == FAL_KLING_MODEL_PRO


def test_kling_request_via_fal_http_omits_content_type_for_get_without_body() -> None:
    class _Response:
        status = 200

        async def text(self) -> str:
            return "{}"

        def release(self) -> None:
            return None

    class _Session:
        def __init__(self) -> None:
            self.last_headers = None

        async def request(self, _method, _url, *, headers=None, **_kwargs):
            self.last_headers = dict(headers or {})
            return _Response()

    client = KlingVideoClient.__new__(KlingVideoClient)
    client._fal_key = "fal_test"
    client._base_url = "https://queue.fal.run"
    session = _Session()

    async def _ensure_session():
        return session

    client._ensure_session = _ensure_session  # type: ignore[method-assign]

    asyncio.run(
        client._request_via_fal_http(
            method="GET",
            path="/fal-ai/kling-video/requests/req-1/status",
            json_payload=None,
        )
    )

    assert session.last_headers == {"Authorization": "Key fal_test"}


def test_kling_request_via_fal_http_sets_content_type_for_post() -> None:
    class _Response:
        status = 200

        async def text(self) -> str:
            return "{}"

        def release(self) -> None:
            return None

    class _Session:
        def __init__(self) -> None:
            self.last_headers = None

        async def request(self, _method, _url, *, headers=None, **_kwargs):
            self.last_headers = dict(headers or {})
            return _Response()

    client = KlingVideoClient.__new__(KlingVideoClient)
    client._fal_key = "fal_test"
    client._base_url = "https://queue.fal.run"
    session = _Session()

    async def _ensure_session():
        return session

    client._ensure_session = _ensure_session  # type: ignore[method-assign]

    asyncio.run(
        client._request_via_fal_http(
            method="POST",
            path="/fal-ai/kling-video/v2.6/standard/motion-control",
            json_payload={"input": {"foo": "bar"}},
        )
    )

    assert session.last_headers == {
        "Authorization": "Key fal_test",
        "Content-Type": "application/json",
    }
