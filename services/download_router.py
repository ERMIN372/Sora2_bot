from __future__ import annotations

from enum import Enum
from urllib.parse import urlparse


class AssetUrlRoute(str, Enum):
    GEMINI_FILE_API = "GEMINI_FILE_API"
    FAL_MEDIA_HTTP = "FAL_MEDIA_HTTP"
    TELEGRAM_HTTP = "TELEGRAM_HTTP"
    GENERIC_HTTP = "GENERIC_HTTP"


def classify_asset_url(url: str) -> AssetUrlRoute:
    cleaned = (url or "").strip()
    if cleaned.startswith("https://generativelanguage.googleapis.com/v1beta/files/"):
        return AssetUrlRoute.GEMINI_FILE_API

    parsed = urlparse(cleaned)
    host = (parsed.hostname or "").lower()
    if host == "generativelanguage.googleapis.com":
        return AssetUrlRoute.GEMINI_FILE_API
    if host.endswith("fal.media") or host == "queue.fal.run":
        return AssetUrlRoute.FAL_MEDIA_HTTP
    if "api.telegram.org" in host:
        return AssetUrlRoute.TELEGRAM_HTTP
    return AssetUrlRoute.GENERIC_HTTP


__all__ = ["AssetUrlRoute", "classify_asset_url"]
