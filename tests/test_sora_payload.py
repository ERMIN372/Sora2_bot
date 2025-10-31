"""Tests for Sora payload normalization."""
from services.payload_sanitize import normalize_sora_payload


def test_duration_sec_top_level_to_string() -> None:
    payload = {"prompt": "x", "duration_sec": 5}
    cleaned = normalize_sora_payload(payload)
    assert cleaned["duration_sec"] == "5"
    assert "metadata" not in cleaned


def test_duration_sec_in_metadata_to_string_and_sync() -> None:
    payload = {"prompt": "x", "metadata": {"duration_sec": 7}}
    cleaned = normalize_sora_payload(payload)
    assert cleaned["duration_sec"] == "7"
    assert "metadata" not in cleaned


def test_aspect_ratio_and_format_to_string() -> None:
    payload = {"aspect_ratio": 169, "format": 264}
    cleaned = normalize_sora_payload(payload)
    assert cleaned["aspect_ratio"] == "169"
    assert cleaned["format"] == "264"


def test_metadata_seed_to_string_if_weird_type() -> None:
    class Weird:  # pragma: no cover - structure only
        pass

    payload = {"metadata": {"seed": Weird()}}
    cleaned = normalize_sora_payload(payload)
    assert cleaned["seed"] == str(payload["metadata"]["seed"])
    assert "metadata" not in cleaned
