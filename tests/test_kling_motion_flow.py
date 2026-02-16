import asyncio
from types import SimpleNamespace

import pytest

from handlers._core import UserSession, _build_video_settings_keyboard
from providers.kling_video import FAL_QUEUE_BASE_URL, KLING_BASE_URL, KlingVideoClient
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


def test_kling_resolve_credentials_falls_back_to_env(monkeypatch) -> None:
    cfg = SimpleNamespace()
    monkeypatch.setenv("KLING_ACCESS_KEY", "env-ak")
    monkeypatch.setenv("KLING_SECRET_KEY", "env-sk")

    access_key, secret_key = KlingVideoClient._resolve_credentials(cfg)  # type: ignore[arg-type]

    assert access_key == "env-ak"
    assert secret_key == "env-sk"


def test_kling_base_url_alias_is_defined() -> None:
    assert FAL_QUEUE_BASE_URL == KLING_BASE_URL


def test_kling_resolve_legacy_fal_key_falls_back_to_env(monkeypatch) -> None:
    cfg = SimpleNamespace()
    monkeypatch.setenv("FAL_KEY", "legacy-fal")

    value = KlingVideoClient._resolve_legacy_fal_key(cfg)  # type: ignore[arg-type]

    assert value == "legacy-fal"


def test_kling_build_headers_adds_bearer_prefix() -> None:
    client = object.__new__(KlingVideoClient)
    client._get_token = lambda: "abc.jwt.token"  # type: ignore[method-assign]

    headers = KlingVideoClient._build_headers(client)

    assert headers["Authorization"] == "Bearer abc.jwt.token"


def test_kling_build_headers_does_not_double_bearer_prefix() -> None:
    client = object.__new__(KlingVideoClient)
    client._get_token = lambda: "Bearer abc.jwt.token"  # type: ignore[method-assign]

    headers = KlingVideoClient._build_headers(client)

    assert headers["Authorization"] == "Bearer abc.jwt.token"


def test_kling_build_headers_falls_back_when_get_token_missing() -> None:
    client = object.__new__(KlingVideoClient)
    client._access_key = "ak"
    client._secret_key = "sk"

    headers = KlingVideoClient._build_headers(client)

    assert headers["Authorization"].startswith("Bearer ")
    assert len(headers["Authorization"].split()) == 2


def test_b64url_encode_works_without_module_base64_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    import providers.kling_video as kling_module

    monkeypatch.delattr(kling_module, "base64", raising=False)

    encoded = kling_module._b64url_encode(b"abc")

    assert encoded == b"YWJj"


def test_generate_jwt_works_without_module_stdlib_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    import providers.kling_video as kling_module

    monkeypatch.delattr(kling_module, "hmac", raising=False)
    monkeypatch.delattr(kling_module, "hashlib", raising=False)
    monkeypatch.delattr(kling_module, "json", raising=False)
    monkeypatch.delattr(kling_module, "time", raising=False)

    token = kling_module._generate_jwt("ak", "sk")

    assert token.count(".") == 2


def test_kling_classify_api_error_detects_provider_balance() -> None:
    status, err_type, err_code = KlingVideoClient._classify_api_error("Account balance not enough")

    assert status == 402
    assert err_type == "provider_insufficient_balance"
    assert err_code == "ACCOUNT_BALANCE_NOT_ENOUGH"


def test_kling_classify_api_error_defaults_to_api_error() -> None:
    status, err_type, err_code = KlingVideoClient._classify_api_error(
        "Some transient upstream issue"
    )

    assert status == 502
    assert err_type == "api_error"
    assert err_code == "KLING_API_ERROR"
