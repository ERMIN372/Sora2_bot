import base64
from typing import List, Optional

import pytest

from providers.gemini import (
    _prepare_flash_image_assets,
    _prepare_images_assets,
    GeminiImageEmptyError,
)


class _DummyContent:
    def __init__(self) -> None:
        self.parts = []


class _DummyCandidate:
    def __init__(self, finish_reason: Optional[str]) -> None:
        self.content = _DummyContent()
        self.finish_reason = finish_reason


class _DummySdkResponse:
    def __init__(self) -> None:
        self.headers = {"x-request-id": "req-1"}


class _DummyResponse:
    def __init__(self, finish_reason: Optional[str]) -> None:
        self.candidates = [_DummyCandidate(finish_reason)]
        self.response_id = "resp-123"
        self.sdk_http_response = _DummySdkResponse()


class _DummyImage:
    def __init__(self, payload: bytes, mime: str = "image/png") -> None:
        self.data = base64.b64encode(payload).decode("ascii")
        self.mime_type = mime


class _DummyImageResponse:
    def __init__(self, images: List[_DummyImage]) -> None:
        self.images = images
        self.response_id = "resp-img"
        self.sdk_http_response = _DummySdkResponse()


def test_prepare_flash_image_assets_raises_for_empty_inline_data() -> None:
    response = _DummyResponse("NO_IMAGE")
    payload = {"prompt_feedback": {"safety_ratings": ["BLOCK_NONE"]}}

    with pytest.raises(GeminiImageEmptyError) as exc_info:
        _prepare_flash_image_assets(
            response,
            payload,
            provider_name="gemini-image",
            duration_ms=150,
        )

    error = exc_info.value
    assert error.error_type == "gemini_empty_image"
    assert error.finish_reason == "NO_IMAGE"
    assert error.response_id == "resp-123"
    assert error.headers.get("x-request-id") == "req-1"
    assert "не вернул изображение" in error.message.lower()


def test_prepare_images_assets_returns_inline_entries() -> None:
    image_bytes = b"test-bytes"
    response = _DummyImageResponse([_DummyImage(image_bytes, "image/jpeg")])

    inline_assets, asset_map, asset_meta = _prepare_images_assets(
        response,
        provider_name="gemini-image",
        duration_ms=42,
    )

    assert len(inline_assets) == 1
    asset = inline_assets[0]
    assert asset["mime"] == "image/jpeg"
    assert asset["bytes"] == len(image_bytes)
    key = asset["key"]
    assert key in asset_map
    assert key in asset_meta
    encoded = asset_map[key].split(",", 1)[1]
    assert base64.b64decode(encoded) == image_bytes
