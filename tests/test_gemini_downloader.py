from __future__ import annotations

import asyncio

import pytest

from config import Config
from services import gemini_downloader as downloader


class _DummyFile:
    def __init__(self, name: str, *, display_name: str = "clip.mp4", mime_type: str = "video/mp4", size_bytes: int = 321) -> None:
        self.name = name
        self.display_name = display_name
        self.mime_type = mime_type
        self.size_bytes = size_bytes


class _SuccessfulFiles:
    def __init__(self) -> None:
        self.get_calls: list[str] = []
        self.download_calls: list[_DummyFile] = []

    def get(self, *, name: str):
        self.get_calls.append(name)
        return _DummyFile(name)

    def download(self, *, file: _DummyFile) -> bytes:
        self.download_calls.append(file)
        return b"video-bytes"


class _DownloadOnlyFiles(_SuccessfulFiles):
    def get(self, *, name: str):
        self.get_calls.append(name)
        return _DummyFile(name, display_name="", mime_type="", size_bytes=0)


class _ForbiddenFiles:
    def __init__(self, *, status: str = "PERMISSION_DENIED") -> None:
        from google.genai import errors as genai_errors

        payload = {"error": {"status": status, "message": "Forbidden"}}
        self._error = genai_errors.APIError(403, payload)
        self.attempts = 0

    def get(self, *, name: str):
        self.attempts += 1
        raise self._error

    def download(self, *, file):  # pragma: no cover - should never be called
        raise AssertionError("download() must not be called on forbidden files")


class _DummyClient:
    def __init__(self, files) -> None:
        self.files = files


def _make_config() -> Config:
    return Config(bot_token="token", gemini_api_key="ABCD1234567890", environment="test")


def test_download_asset_success(monkeypatch: pytest.MonkeyPatch) -> None:
    files = _SuccessfulFiles()
    monkeypatch.setattr(downloader, "get_gemini_client", lambda _cfg: _DummyClient(files))
    config = _make_config()

    result = asyncio.run(
        downloader.download_asset(
            asset_url="https://generativelanguage.googleapis.com/download/v1beta/files/file-123:download?alt=media",
            config=config,
            corr_id="corr-1",
            asset_name="video",
            expected_mask=None,
        )
    )

    assert result.filename == "clip.mp4"
    assert result.mime == "video/mp4"
    assert result.size == 321
    assert files.get_calls == ["files/file-123"]
    assert files.download_calls and files.download_calls[0].name == "files/file-123"


def test_download_asset_uses_hints_when_metadata_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    files = _DownloadOnlyFiles()
    monkeypatch.setattr(downloader, "get_gemini_client", lambda _cfg: _DummyClient(files))
    config = _make_config()

    result = asyncio.run(
        downloader.download_asset(
            asset_url="files/file-456:download",
            config=config,
            corr_id="corr-2",
            asset_name="video",
            expected_mask=None,
            mime_hint="video/mp4",
            filename_hint="custom.mp4",
        )
    )

    assert result.filename == "custom.mp4"
    assert result.mime == "video/mp4"
    assert result.size == len(b"video-bytes")
    assert files.get_calls == ["files/file-456"]


def test_download_asset_permission_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    files = _ForbiddenFiles()
    monkeypatch.setattr(downloader, "get_gemini_client", lambda _cfg: _DummyClient(files))

    async def _instant_sleep(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(downloader.asyncio, "sleep", _instant_sleep)
    config = _make_config()

    with pytest.raises(downloader.GeminiConfigurationError) as excinfo:
        asyncio.run(
            downloader.download_asset(
                asset_url="https://generativelanguage.googleapis.com/download/v1beta/files/file-789:download?alt=media",
                config=config,
                corr_id="corr-3",
                asset_name="video",
                expected_mask=None,
            )
        )

    assert files.attempts == 2
    assert excinfo.value.status_code == 403
    assert excinfo.value.error_status == "PERMISSION_DENIED"

