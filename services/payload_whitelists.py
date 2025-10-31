"""Allowed payload keys for provider-specific sanitization."""
from __future__ import annotations

from typing import Final, Set

GEMINI_IMAGE_ALLOWED: Final[Set[str]] = {
    "contents",
    "config",
    "tools",
    "system_instruction",
}

GEMINI_TEXT_ALLOWED: Final[Set[str]] = {
    "contents",
    "config",
    "tools",
    "system_instruction",
}

SORA_VIDEO_ALLOWED: Final[Set[str]] = {
    "prompt",
    "aspect_ratio",
    "duration_sec",
    "format",
    "seed",
}

__all__ = [
    "GEMINI_IMAGE_ALLOWED",
    "GEMINI_TEXT_ALLOWED",
    "SORA_VIDEO_ALLOWED",
]
