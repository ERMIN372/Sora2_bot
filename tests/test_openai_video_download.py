import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

import pytest

from providers.base import ProviderAPIError
from providers.openai_video import OpenAIVideoClient


class _Stream:
    def __init__(self, chunks: Iterable[bytes]) -> None:
        self._chunks = list(chunks)

    async def iter_chunked(self, _size: int) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            await asyncio.sleep(0)
            yield chunk


class FakeResponse:
    def __init__(
        self,
        status: int,
        *,
        chunks: Optional[Iterable[bytes]] = None,
        headers: Optional[Dict[str, str]] = None,
        text: str = "",
    ) -> None:
        self.status = status
        self._chunks = list(chunks or [])
        self.headers = dict(headers or {})
        self._text = text
        self.history: List[Any] = []
        self.content = _Stream(self._chunks)

    async def __aenter__(self) -> "FakeResponse":
        self.content = _Stream(self._chunks)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            await asyncio.sleep(0)
            yield chunk

    async def text(self) -> str:
        return self._text


class FakeRequest:
    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    def __await__(self):
        async def _inner() -> FakeResponse:
            return self._response

        return _inner().__await__()

    async def __aenter__(self) -> FakeResponse:
        return await self._response.__aenter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return await self._response.__aexit__(exc_type, exc, tb)


class FakeSession:
    def __init__(self, responses: Iterable[FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: List[Any] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        if not self._responses:
            raise AssertionError("No more fake responses queued")
        self.calls.append((url, kwargs))
        return FakeRequest(self._responses.pop(0))


def _run(coro):
    return asyncio.run(coro)


def test_download_follows_redirect(monkeypatch: pytest.MonkeyPatch, make_openai_video_config) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)

    response = FakeResponse(
        200,
        chunks=[b"hello", b"world"],
        headers={"x-request-id": "req-302", "content-type": "video/mp4"},
    )
    response.history = [FakeResponse(302, headers={"Location": "https://cdn.test/video.mp4"})]
    fake_session = FakeSession([response])

    async def fake_ensure(self) -> FakeSession:
        return fake_session

    monkeypatch.setattr(OpenAIVideoClient, "_ensure_session", fake_ensure)

    path = _run(client.download_content("vid302"))
    assert path.exists()
    assert path.read_bytes() == b"helloworld"
    assert path.suffix == ".mp4"
    assert fake_session.calls
    _url, kwargs = fake_session.calls[0]
    assert kwargs.get("allow_redirects") is True
    assert kwargs.get("timeout") is not None
    Path(path).unlink()
    path.parent.rmdir()


def test_download_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch, make_openai_video_config) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)

    responses = [
        FakeResponse(409, headers={"x-request-id": "req-1"}, text="pending"),
        FakeResponse(409, headers={"x-request-id": "req-2"}, text="pending"),
        FakeResponse(200, chunks=[b"data"], headers={"x-request-id": "req-3"}),
    ]
    fake_session = FakeSession(responses)

    async def fake_ensure(self) -> FakeSession:
        return fake_session

    async def fast_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(OpenAIVideoClient, "_ensure_session", fake_ensure)
    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    path = _run(client.download_content("vid-retry"))
    assert path.exists()
    assert path.read_bytes() == b"data"
    assert len(fake_session.calls) == 3
    Path(path).unlink()
    path.parent.rmdir()


def test_download_content_not_ready(monkeypatch: pytest.MonkeyPatch, make_openai_video_config) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)

    responses = [
        FakeResponse(404, headers={"x-request-id": "req-a"}, text="missing"),
        FakeResponse(404, headers={"x-request-id": "req-b"}, text="missing"),
        FakeResponse(404, headers={"x-request-id": "req-c"}, text="missing"),
        FakeResponse(404, headers={"x-request-id": "req-d"}, text="missing"),
    ]
    fake_session = FakeSession(responses)

    async def fake_ensure(self) -> FakeSession:
        return fake_session

    async def fast_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(OpenAIVideoClient, "_ensure_session", fake_ensure)
    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    with pytest.raises(ProviderAPIError) as excinfo:
        _run(client.download_content("vid-missing"))
    error = excinfo.value
    assert error.error_code == "content_not_ready"
    assert error.code == "content_not_ready"
    assert len(fake_session.calls) == 4


def test_download_auth_error(monkeypatch: pytest.MonkeyPatch, make_openai_video_config) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)

    fake_session = FakeSession([FakeResponse(401, headers={"x-request-id": "req-auth"}, text="unauthorized")])

    async def fake_ensure(self) -> FakeSession:
        return fake_session

    monkeypatch.setattr(OpenAIVideoClient, "_ensure_session", fake_ensure)

    with pytest.raises(ProviderAPIError) as excinfo:
        _run(client.download_content("vid-auth"))
    error = excinfo.value
    assert error.error_code == "auth"
    assert error.code == "auth"
    assert "HTTP 401" in str(error)


def test_download_forbidden_error(monkeypatch: pytest.MonkeyPatch, make_openai_video_config) -> None:
    config = make_openai_video_config()
    client = OpenAIVideoClient(config=config)

    fake_session = FakeSession([FakeResponse(403, headers={"x-request-id": "req-block"}, text="forbidden")])

    async def fake_ensure(self) -> FakeSession:
        return fake_session

    monkeypatch.setattr(OpenAIVideoClient, "_ensure_session", fake_ensure)

    with pytest.raises(ProviderAPIError) as excinfo:
        _run(client.download_content("vid-block"))
    error = excinfo.value
    assert error.error_code == "region_blocked"
    assert error.code == "region_blocked"
    assert "HTTP 403" in str(error)
