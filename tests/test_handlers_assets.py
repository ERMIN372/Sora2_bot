from datetime import datetime
import sys
import types
from datetime import datetime

from tests.aiogram_stub import ensure_aiogram_stub

ensure_aiogram_stub()

from db import GenerationJobRecord
from handlers import _select_remote_media_asset


def _make_job(*, video_url=None, file_url=None, extra=None, content_type="video"):
    return GenerationJobRecord(
        id="job-1",
        user_id=1,
        prompt="",
        status="completed",
        video_url=video_url,
        video_id=None,
        error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        file_url=file_url,
        content_type=content_type,
        extra=extra or {},
    )


def test_select_remote_asset_accepts_kind_video():
    job = _make_job(
        extra={
            "assets_meta": [
                {
                    "key": "asset_0",
                    "kind": "video",
                    "url": "https://example.com/video.mp4",
                }
            ]
        }
    )

    asset = _select_remote_media_asset(job)

    assert asset is not None
    assert asset.get("url") == "https://example.com/video.mp4"


def test_select_remote_asset_fallback_to_video_url():
    job = _make_job(video_url="https://example.com/direct.mp4", extra={})

    asset = _select_remote_media_asset(job)

    assert asset is not None
    assert asset.get("url") == "https://example.com/direct.mp4"
    assert asset.get("fallback") is True


def test_select_remote_asset_accepts_images():
    job = _make_job(
        content_type="image",
        extra={
            "assets_meta": [
                {
                    "key": "asset_1",
                    "kind": "image",
                    "url": "https://example.com/image.png",
                    "mime": "image/png",
                }
            ]
        },
    )

    asset = _select_remote_media_asset(job)

    assert asset is not None
    assert asset.get("url") == "https://example.com/image.png"


def test_select_remote_asset_fallback_to_file_url_for_images():
    job = _make_job(
        content_type="image",
        file_url="https://example.com/backup.png",
        extra={},
    )

    asset = _select_remote_media_asset(job)

    assert asset is not None
    assert asset.get("url") == "https://example.com/backup.png"
    assert asset.get("fallback") is True
