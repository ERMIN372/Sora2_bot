"""Tests for Sora payload normalization."""
from services.payload_sanitize import normalize_sora_payload


def test_duration_sec_top_level_to_string() -> None:
    payload = {"prompt": "x", "duration_sec": 5}
    cleaned = normalize_sora_payload(payload)
    assert isinstance(cleaned["duration_sec"], str)
    assert cleaned["metadata"]["duration_sec"] == cleaned["duration_sec"]


def test_duration_sec_in_metadata_to_string_and_sync() -> None:
    payload = {"prompt": "x", "metadata": {"duration_sec": 7}}
    cleaned = normalize_sora_payload(payload)
    assert isinstance(cleaned["metadata"]["duration_sec"], str)
    assert cleaned["duration_sec"] == cleaned["metadata"]["duration_sec"]


def test_aspect_ratio_and_format_to_string() -> None:
    payload = {"aspect_ratio": 169, "format": 264}
    cleaned = normalize_sora_payload(payload)
    assert isinstance(cleaned["aspect_ratio"], str)
    assert isinstance(cleaned["format"], str)


def test_metadata_seed_to_string_if_weird_type() -> None:
    class Weird:  # pragma: no cover - structure only
        pass

    payload = {"metadata": {"seed": Weird()}}
    cleaned = normalize_sora_payload(payload)
    assert isinstance(cleaned["metadata"]["seed"], str)
