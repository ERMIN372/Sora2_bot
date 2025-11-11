from __future__ import annotations

import asyncio
from typing import Optional

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

    def get(self, *, name: str):
        self.get_calls.append(name)
        return _DummyFile(name)


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


class _DummyClient:
    def __init__(self, files) -> None:
        self.files = files


def _make_config() -> Config:
    return Config(bot_token="token", gemini_api_key="ABCD1234567890", environment="test")


class _Headers(dict):
    def __init__(self, data: Optional[dict[str, str]] = None) -> None:
        super().__init__()
        for key, value in (data or {}).items():
            self[key.lower()] = value

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return super().get(key.lower(), default)


class _HttpResponse:
    def __init__(self, *, status_code: int = 200, body: bytes = b"video-bytes", headers: Optional[dict[str, str]] = None) -> None:
        self.status_code = status_code
        self._body = body
        base_headers = {"content-type": "video/mp4", "content-length": str(len(body))}
        if headers:
            base_headers.update(headers)
        try:
            desired_length = int(base_headers.get("content-length", len(body)))
        except ValueError:
            desired_length = len(body)
        if desired_length != len(self._body):
            seed = body[:1] or b"x"
            self._body = seed * desired_length
        self.headers = _Headers(base_headers)

    async def aread(self) -> bytes:
        return self._body


class _HttpClient:
    responses: list[_HttpResponse] = []
    calls: list[tuple[str, dict[str, str]]] = []

    def __init__(self, *args, **kwargs) -> None:
        self._responses = list(self.__class__.responses)

    async def __aenter__(self) -> "_HttpClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str, headers: Optional[dict[str, str]] = None) -> _HttpResponse:
        self.__class__.calls.append((url, headers or {}))
        if not self._responses:
            raise AssertionError("No HTTP responses configured")
        return self._responses.pop(0)


class _Timeout:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class _HttpxStub:
    AsyncClient = _HttpClient
    Timeout = _Timeout
    Headers = _Headers
    TimeoutException = Exception
    RequestError = Exception


def test_download_asset_success(monkeypatch: pytest.MonkeyPatch) -> None:
    files = _SuccessfulFiles()
    monkeypatch.setattr(
        downloader, "get_media_client", lambda _cfg: _DummyClient(files)
    )
    _HttpClient.responses = [_HttpResponse(headers={"content-length": "321"})]
    _HttpClient.calls.clear()
    monkeypatch.setattr(downloader, "httpx", _HttpxStub)
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
    assert result.path and result.path.exists()
    assert files.get_calls == ["files/file-123"]
    result.path.unlink(missing_ok=True)


def test_download_asset_uses_hints_when_metadata_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    files = _DownloadOnlyFiles()
    monkeypatch.setattr(
        downloader, "get_media_client", lambda _cfg: _DummyClient(files)
    )
    _HttpClient.responses = [_HttpResponse()]
    _HttpClient.calls.clear()
    monkeypatch.setattr(downloader, "httpx", _HttpxStub)
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
    assert result.path and result.path.exists()
    assert files.get_calls == ["files/file-456"]
    result.path.unlink(missing_ok=True)


def test_download_asset_permission_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    files = _ForbiddenFiles()
    monkeypatch.setattr(
        downloader, "get_media_client", lambda _cfg: _DummyClient(files)
    )

    async def _instant_sleep(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(downloader.asyncio, "sleep", _instant_sleep)
    _HttpClient.responses = []
    _HttpClient.calls.clear()
    monkeypatch.setattr(downloader, "httpx", _HttpxStub)
    config = _make_config()

    with pytest.raises(downloader.GeminiDownloadError) as excinfo:
        asyncio.run(
            downloader.download_asset(
                asset_url="https://generativelanguage.googleapis.com/download/v1beta/files/file-789:download?alt=media",
                config=config,
                corr_id="corr-3",
                asset_name="video",
                expected_mask=None,
            )
        )

    assert files.attempts == 4
    assert excinfo.value.status_code == 403
    assert excinfo.value.error_status == "PERMISSION_DENIED"


def test_download_asset_direct_url(monkeypatch: pytest.MonkeyPatch) -> None:
    files = _SuccessfulFiles()
    monkeypatch.setattr(
        downloader, "get_media_client", lambda _cfg: _DummyClient(files)
    )
    _HttpClient.responses = [_HttpResponse()]
    _HttpClient.calls.clear()
    monkeypatch.setattr(downloader, "httpx", _HttpxStub)
    config = _make_config()

    expected_mask = downloader.current_key_mask(config)

    result = asyncio.run(
        downloader.download_asset(
            asset_url="https://example.com/direct.mp4",
            config=config,
            corr_id="corr-4",
            asset_name="video",
            expected_mask=expected_mask,
            filename_hint="direct.mp4",
        )
    )

    assert result.mime == "video/mp4"
    assert result.size == len(b"video-bytes")
    assert _HttpClient.calls
    assert _HttpClient.calls[0][1].get("x-goog-api-key") == config.gemini_api_key
    if result.path:
        result.path.unlink(missing_ok=True)

