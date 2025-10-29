from services.payload_sanitize import sanitize_payload
from services.payload_whitelists import (
    GEMINI_IMAGE_ALLOWED,
    SORA_VIDEO_ALLOWED,
)


def test_sora_sanitizer_removes_gemini_keys() -> None:
    payload = {
        "prompt": "video",
        "aspect_ratio": "16:9",
        "duration_sec": 5,
        "modalities": ["IMAGE"],
        "tools": [],
    }
    cleaned = sanitize_payload(payload, SORA_VIDEO_ALLOWED)
    assert "modalities" not in cleaned
    assert "tools" not in cleaned
    assert cleaned["prompt"] == "video"


def test_gemini_image_keeps_responseModalities() -> None:
    payload = {
        "contents": "inline",
        "config": {"responseModalities": ["TEXT", "IMAGE"]},
        "modalities": ["X"],
    }
    cleaned = sanitize_payload(payload, GEMINI_IMAGE_ALLOWED)
    assert "contents" in cleaned
    assert "config" in cleaned
    assert "modalities" not in cleaned
