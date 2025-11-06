from typing import Optional

import pytest

from providers.gemini import _prepare_flash_image_assets, GeminiImageEmptyError


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
