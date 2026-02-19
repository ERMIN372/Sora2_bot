from __future__ import annotations

import asyncio
import base64
import io
import sys
import types
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from PIL import Image

from tests.aiogram_stub import ensure_aiogram_stub

ensure_aiogram_stub()

if "yookassa" not in sys.modules:
    yookassa_module = types.ModuleType("yookassa")
    yookassa_module.Configuration = type("Configuration", (), {})
    yookassa_module.Payment = type("Payment", (), {})
    sys.modules["yookassa"] = yookassa_module

if "yookassa_client" not in sys.modules:
    yookassa_client_module = types.ModuleType("yookassa_client")
    yookassa_client_module.create_payment = lambda *args, **kwargs: None
    sys.modules["yookassa_client"] = yookassa_client_module

from db import GenerationJobRecord
from handlers import _send_job_update
import handlers._core as handlers_core
from services.gemini_downloader import DownloadedAsset, GeminiDownloadError


def _run(coro):
    return asyncio.run(coro)


class _StubStatusMessages:
    def __init__(self) -> None:
        self.started: List[str] = []
        self.sending: List[str] = []
        self.completed: List[str] = []
        self.failed: List[Dict[str, Any]] = []

    async def ensure_started(self, *, bot, db, job) -> None:
        self.started.append(job.id)

    async def mark_sending(self, *, bot, db, job, text: str) -> None:
        self.sending.append(text)

    async def mark_completed(
        self, *, bot, db, job, text: str, schedule_deletion: bool = True
    ) -> None:
        self.completed.append(text)

    async def mark_failed(
        self,
        *,
        bot,
        db,
        job,
        text: str,
        terminal_state: str = "failed",
        schedule_deletion: bool = False,
    ) -> None:
        self.failed.append({"text": text, "terminal_state": terminal_state})

    async def tick(self, *, bot, db, job) -> None:  # pragma: no cover - not used
        return None


class _FakeBot:
    def __init__(self, *, file_size: int, fail_photo: bool = False) -> None:
        self.file_size = file_size
        self.fail_photo = fail_photo
        self.sent: List[Dict[str, Any]] = []

    async def send_video(
        self,
        user_id: int,
        file: Any,
        *,
        supports_streaming: bool = True,
        duration: Optional[int] = None,
    ) -> Any:
        self.sent.append(
            {
                "method": "video",
                "user_id": user_id,
                "supports_streaming": supports_streaming,
                "duration": duration,
                "file": file,
            }
        )
        video = SimpleNamespace(
            file_id="file-video",
            file_size=self.file_size,
            duration=duration or 0,
        )
        return SimpleNamespace(video=video, document=None)

    async def send_document(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        self.sent.append({"method": "document", "args": args, "kwargs": kwargs})
        document = SimpleNamespace(file_id="file-doc", file_size=self.file_size)
        return SimpleNamespace(video=None, document=document, photo=None)

    async def send_photo(self, user_id: int, file: Any, *args: Any, **kwargs: Any) -> Any:
        self.sent.append({"method": "photo", "user_id": user_id, "file": file})
        if self.fail_photo:
            raise handlers_core.TelegramAPIError("photo failed")
        photo = SimpleNamespace(file_id="file-photo", file_size=self.file_size)
        return SimpleNamespace(photo=[photo], document=None, video=None)


class _FakeDispatcher:
    def __init__(self, bot: _FakeBot) -> None:
        self.bot = bot


class _FakeDB:
    def __init__(self) -> None:
        self.update_calls: List[Dict[str, Any]] = []
        self.refunds: List[tuple[int, int]] = []
        self.errors: List[Any] = []

    async def update_job(self, job_id: str, status: str, **fields: Any) -> None:
        payload = {"job_id": job_id, "status": status, **fields}
        self.update_calls.append(payload)

    async def add_credits(self, user_id: int, amount: int) -> None:
        self.refunds.append((user_id, amount))

    async def log_error_record(self, record: Any) -> bool:
        self.errors.append(record)
        return True


class _FakeErrorReporter:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def report(self, record: Any, *, context: str, extra: Dict[str, Any]) -> None:
        self.calls.append({"record": record, "context": context, "extra": extra})


def _make_job(*, status: str = "completed", cost: int = 7) -> GenerationJobRecord:
    now = datetime.now(timezone.utc)
    job = GenerationJobRecord(
        id="job-1",
        user_id=42,
        prompt="render scene",
        status=status,
        video_url=None,
        video_id=None,
        error=None,
        created_at=now,
        updated_at=now,
        size="720p",
        model="veo-3.0-pro",
        cost_credits=cost,
        username="tester",
        corr_id="corr-1",
    )
    job.extra = {
        "assets_meta": [
            {
                "key": "primary",
                "url": "https://generativelanguage.googleapis.com/v1beta/files/file-123:download?alt=media",
                "mime": "video/mp4",
            }
        ]
    }
    return job


def test_send_job_update_remote_success(
    monkeypatch: pytest.MonkeyPatch, make_openai_video_config
) -> None:
    config = make_openai_video_config()
    job = _make_job(cost=9)
    status_stub = _StubStatusMessages()
    db = _FakeDB()
    reporter = _FakeErrorReporter()
    downloaded = DownloadedAsset(
        content=b"video-bytes",
        mime="video/mp4",
        filename="result.mp4",
        size=4096,
        key_mask="MASK",
    )

    async def fake_download_asset(*_args: Any, **_kwargs: Any) -> DownloadedAsset:
        return downloaded

    bot = _FakeBot(file_size=downloaded.size)
    dp = _FakeDispatcher(bot)

    monkeypatch.setattr(handlers_core, "STATUS_MESSAGES", status_stub)
    monkeypatch.setattr(handlers_core, "download_asset", fake_download_asset)
    monkeypatch.setattr(handlers_core, "increment_metric", lambda *a, **k: None)
    monkeypatch.setattr(handlers_core, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(handlers_core, "record_timing_metric", lambda *a, **k: None)

    _run(
        _send_job_update(
            dp=dp,
            job=job,
            archive=None,
            config=config,
            db=db,
            error_reporter=reporter,
        )
    )

    assert bot.sent and bot.sent[0]["method"] == "video"
    assert status_stub.completed
    assert not status_stub.failed
    assert db.refunds == []
    assert all(call["status"] != "failed" for call in db.update_calls)
    assert job.status == "completed"


def test_send_job_update_remote_failure(
    monkeypatch: pytest.MonkeyPatch, make_openai_video_config
) -> None:
    config = make_openai_video_config()
    job = _make_job(cost=11)
    status_stub = _StubStatusMessages()
    db = _FakeDB()
    reporter = _FakeErrorReporter()

    async def failing_download_asset(*_args: Any, **_kwargs: Any) -> DownloadedAsset:
        raise GeminiDownloadError("download failed", status_code=502)

    bot = _FakeBot(file_size=2048)
    dp = _FakeDispatcher(bot)

    monkeypatch.setattr(handlers_core, "STATUS_MESSAGES", status_stub)
    monkeypatch.setattr(handlers_core, "download_asset", failing_download_asset)
    monkeypatch.setattr(handlers_core, "increment_metric", lambda *a, **k: None)
    monkeypatch.setattr(handlers_core, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(handlers_core, "record_timing_metric", lambda *a, **k: None)

    _run(
        _send_job_update(
            dp=dp,
            job=job,
            archive=None,
            config=config,
            db=db,
            error_reporter=reporter,
        )
    )

    assert not status_stub.completed
    assert status_stub.failed and status_stub.failed[0]["terminal_state"] == "refunded"
    expected_refund = (job.user_id, job.cost_credits or config.generation_cost_credits)
    assert db.refunds and all(entry == expected_refund for entry in db.refunds)
    assert any(call["status"] == "failed" for call in db.update_calls)
    assert job.status == "failed"


def test_delivery_routes_fal_media_to_http_not_gemini(
    monkeypatch: pytest.MonkeyPatch, make_openai_video_config
) -> None:
    config = make_openai_video_config()
    job = _make_job(cost=5)
    job.model = "kling-v2.6"
    job.extra["provider"] = "kling"
    job.extra["assets_meta"][0]["url"] = "https://v3b.fal.media/files/b/abc/output.mp4"

    status_stub = _StubStatusMessages()
    db = _FakeDB()
    reporter = _FakeErrorReporter()
    bot = _FakeBot(file_size=1024)
    dp = _FakeDispatcher(bot)

    async def fail_gemini(*_args: Any, **_kwargs: Any) -> DownloadedAsset:
        raise AssertionError("gemini downloader must not be called for fal.media")

    async def fail_http(*_args: Any, **_kwargs: Any) -> DownloadedAsset:
        raise AssertionError("HTTP fallback should not be called when Telegram accepts URL")

    monkeypatch.setattr(handlers_core, "STATUS_MESSAGES", status_stub)
    monkeypatch.setattr(handlers_core, "download_asset", fail_gemini)
    monkeypatch.setattr(handlers_core, "_download_remote_asset_http", fail_http)
    monkeypatch.setattr(handlers_core, "increment_metric", lambda *a, **k: None)
    monkeypatch.setattr(handlers_core, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(handlers_core, "record_timing_metric", lambda *a, **k: None)

    _run(
        _send_job_update(
            dp=dp,
            job=job,
            archive=None,
            config=config,
            db=db,
            error_reporter=reporter,
        )
    )

    assert bot.sent
    assert bot.sent[0]["method"] == "video"
    assert bot.sent[0].get("file") == "https://v3b.fal.media/files/b/abc/output.mp4"


def test_delivery_accepts_gemini_files_only_for_generativelanguage_host(
    monkeypatch: pytest.MonkeyPatch, make_openai_video_config
) -> None:
    config = make_openai_video_config()
    job = _make_job(cost=5)
    job.model = "veo-3.0-generate-001"
    job.extra["provider"] = "veo"
    job.extra["assets_meta"][0][
        "url"
    ] = "https://generativelanguage.googleapis.com/v1beta/files/abc123:download?alt=media"

    status_stub = _StubStatusMessages()
    db = _FakeDB()
    reporter = _FakeErrorReporter()
    bot = _FakeBot(file_size=4096)
    dp = _FakeDispatcher(bot)

    calls: List[str] = []

    async def fake_gemini(*_args: Any, **_kwargs: Any) -> DownloadedAsset:
        calls.append("gemini")
        return DownloadedAsset(
            content=b"video-bytes",
            mime="video/mp4",
            filename="gemini.mp4",
            size=4096,
            key_mask="MASK",
        )

    monkeypatch.setattr(handlers_core, "STATUS_MESSAGES", status_stub)
    monkeypatch.setattr(handlers_core, "download_asset", fake_gemini)
    monkeypatch.setattr(handlers_core, "increment_metric", lambda *a, **k: None)
    monkeypatch.setattr(handlers_core, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(handlers_core, "record_timing_metric", lambda *a, **k: None)

    _run(
        _send_job_update(
            dp=dp,
            job=job,
            archive=None,
            config=config,
            db=db,
            error_reporter=reporter,
        )
    )

    assert calls == ["gemini"]


def test_deliver_remote_media_sends_image_mime_as_photo_even_for_veo(
    monkeypatch: pytest.MonkeyPatch, make_openai_video_config
) -> None:
    config = make_openai_video_config()
    job = _make_job(cost=5)
    job.model = "gemini-3-pro-image-preview"
    job.content_type = "video"
    job.extra["provider"] = "veo"
    asset = {
        "key": "primary",
        "url": "https://example.com/generated.jpg",
        "mime": "image/jpeg",
        "filename": "generated.jpg",
    }
    status_stub = _StubStatusMessages()
    db = _FakeDB()
    reporter = _FakeErrorReporter()
    bot = _FakeBot(file_size=1024)
    dp = _FakeDispatcher(bot)

    async def fake_http(*_args: Any, **_kwargs: Any) -> DownloadedAsset:
        return DownloadedAsset(
            content=b"\xff\xd8\xffvalid-jpeg",
            mime="image/jpeg",
            filename="generated.jpg",
            size=1024,
            key_mask="MASK",
        )

    monkeypatch.setattr(handlers_core, "STATUS_MESSAGES", status_stub)
    monkeypatch.setattr(handlers_core, "_download_remote_asset_http", fake_http)

    sent = _run(
        handlers_core._deliver_remote_media(
            dp=dp,
            job=job,
            asset=asset,
            config=config,
            db=db,
            error_reporter=reporter,
        )
    )

    assert sent is not None
    assert sent.method == "photo"
    assert bot.sent[0]["method"] == "photo"


def test_deliver_remote_media_falls_back_to_document_when_photo_fails(
    monkeypatch: pytest.MonkeyPatch, make_openai_video_config
) -> None:
    config = make_openai_video_config()
    job = _make_job(cost=5)
    job.model = "gemini-3-pro-image-preview"
    job.content_type = "image"
    job.extra["provider"] = "veo"
    asset = {
        "key": "primary",
        "url": "https://example.com/generated.jpg",
        "mime": "image/jpeg",
        "filename": "generated.jpg",
    }
    status_stub = _StubStatusMessages()
    db = _FakeDB()
    reporter = _FakeErrorReporter()
    bot = _FakeBot(file_size=1024, fail_photo=True)
    dp = _FakeDispatcher(bot)

    async def fake_http(*_args: Any, **_kwargs: Any) -> DownloadedAsset:
        return DownloadedAsset(
            content=b"\xff\xd8\xffvalid-jpeg",
            mime="image/jpeg",
            filename="generated.jpg",
            size=1024,
            key_mask="MASK",
        )

    monkeypatch.setattr(handlers_core, "STATUS_MESSAGES", status_stub)
    monkeypatch.setattr(handlers_core, "_download_remote_asset_http", fake_http)

    sent = _run(
        handlers_core._deliver_remote_media(
            dp=dp,
            job=job,
            asset=asset,
            config=config,
            db=db,
            error_reporter=reporter,
        )
    )

    assert sent is not None
    assert sent.method == "document"
    assert [entry["method"] for entry in bot.sent[:2]] == ["photo", "document"]


def test_deliver_remote_media_rejects_html_payload_for_image(
    monkeypatch: pytest.MonkeyPatch, make_openai_video_config
) -> None:
    config = make_openai_video_config()
    job = _make_job(cost=5)
    job.model = "gemini-3-pro-image-preview"
    job.content_type = "image"
    job.extra["provider"] = "gemini-image-pro"
    asset = {
        "key": "primary",
        "url": "https://example.com/generated.jpg",
        "mime": "image/jpeg",
        "filename": "generated.jpg",
    }
    status_stub = _StubStatusMessages()
    db = _FakeDB()
    reporter = _FakeErrorReporter()
    bot = _FakeBot(file_size=1024)
    dp = _FakeDispatcher(bot)

    async def fake_http(*_args: Any, **_kwargs: Any) -> DownloadedAsset:
        return DownloadedAsset(
            content=b"<!DOCTYPE html><html>error</html>",
            mime="image/jpeg",
            filename="generated.jpg",
            size=33,
            key_mask="MASK",
        )

    monkeypatch.setattr(handlers_core, "STATUS_MESSAGES", status_stub)
    monkeypatch.setattr(handlers_core, "_download_remote_asset_http", fake_http)

    sent = _run(
        handlers_core._deliver_remote_media(
            dp=dp,
            job=job,
            asset=asset,
            config=config,
            db=db,
            error_reporter=reporter,
        )
    )

    assert sent is None
    assert bot.sent == []
    assert db.refunds


def test_decode_image_base64_payload_supports_plain_base64() -> None:
    payload = base64.b64encode(b"\xff\xd8\xffvalid-jpeg").decode("ascii")
    decoded = handlers_core._decode_image_base64_payload(payload, mime_hint="image/jpeg")

    assert decoded is not None
    mime, content, source_kind = decoded
    assert mime == "image/jpeg"
    assert content.startswith(b"\xff\xd8\xff")
    assert source_kind == "base64"


def _valid_png_data_uri() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _make_inline_image_job(
    *, model: str = "gemini-3-pro-image-preview", provider: str = "veo"
) -> GenerationJobRecord:
    job = _make_job(cost=5)
    job.content_type = "image"
    job.model = model
    job.extra = {
        "provider": provider,
        "inline_assets": [
            {"data_uri": _valid_png_data_uri(), "mime": "image/png", "kind": "image"}
        ],
    }
    return job


def test_decode_data_uri_tolerates_spaces_as_plus() -> None:
    original = b"abc+def"
    encoded = base64.b64encode(original).decode("ascii").replace("+", " ")

    decoded = handlers_core.decode_data_uri(f"data:image/jpeg;base64,{encoded}")

    assert decoded is not None
    mime, payload = decoded
    assert mime == "image/jpeg"
    assert payload == original


def test_decode_data_uri_supports_urlsafe_base64() -> None:
    original = b"\xfb\xef\xffbinary"
    encoded = base64.urlsafe_b64encode(original).decode("ascii")

    decoded = handlers_core.decode_data_uri(f"data:image/webp;base64,{encoded}")

    assert decoded is not None
    mime, payload = decoded
    assert mime == "image/webp"
    assert payload == original


def test_decode_data_uri_supports_percent_encoded_base64() -> None:
    original = b"\xff\xd8\xffjpeg-bytes"
    encoded = base64.b64encode(original).decode("ascii")
    percent_encoded = encoded.replace("+", "%2B").replace("/", "%2F")

    decoded = handlers_core.decode_data_uri(f"data:image/jpeg;base64,{percent_encoded}")

    assert decoded is not None
    mime, payload = decoded
    assert mime == "image/jpeg"
    assert payload == original


def test_decode_data_uri_supports_double_wrapped_data_uri() -> None:
    inner_payload = b"\xff\xd8\xffdouble-wrapped-jpeg"
    inner_b64 = base64.b64encode(inner_payload).decode("ascii")
    inner_uri = f"data:image/jpeg;base64,{inner_b64}"
    outer_b64 = base64.b64encode(inner_uri.encode("utf-8")).decode("ascii")

    decoded = handlers_core.decode_data_uri(f"data:text/plain;base64,{outer_b64}")

    assert decoded is not None
    mime, payload = decoded
    assert mime == "image/jpeg"
    assert payload == inner_payload


def test_decode_data_uri_supports_double_base64_wrapped() -> None:
    jpeg_bytes = b"\xFF\xD8\xFF\xE0" + b"JFIF\x00" + b"\x00" * 16
    inner = base64.urlsafe_b64encode(jpeg_bytes).decode("ascii")
    outer = base64.b64encode(inner.encode("ascii")).decode("ascii")

    decoded = handlers_core.decode_data_uri(f"data:image/jpeg;base64,{outer}")

    assert decoded is not None
    mime, payload = decoded
    assert mime == "image/jpeg"
    assert payload == jpeg_bytes


def test_send_inline_assets_falls_back_to_document_when_photo_fails(
    monkeypatch: pytest.MonkeyPatch, make_openai_video_config
) -> None:
    config = make_openai_video_config()
    job = _make_inline_image_job()
    status_stub = _StubStatusMessages()
    db = _FakeDB()
    reporter = _FakeErrorReporter()
    bot = _FakeBot(file_size=1024, fail_photo=True)
    dp = _FakeDispatcher(bot)

    monkeypatch.setattr(handlers_core, "STATUS_MESSAGES", status_stub)

    sent = _run(
        handlers_core._send_inline_assets(
            dp=dp,
            job=job,
            assets=job.extra["inline_assets"],
            config=config,
            db=db,
            error_reporter=reporter,
        )
    )

    assert sent is not None
    assert sent.method == "document"
    assert [entry["method"] for entry in bot.sent[:2]] == ["photo", "document"]


def test_send_job_update_video_path_unchanged(
    monkeypatch: pytest.MonkeyPatch, make_openai_video_config
) -> None:
    config = make_openai_video_config()
    job = _make_job(cost=8)
    status_stub = _StubStatusMessages()
    db = _FakeDB()
    reporter = _FakeErrorReporter()
    downloaded = DownloadedAsset(
        content=b"video-bytes",
        mime="video/mp4",
        filename="result.mp4",
        size=4096,
        key_mask="MASK",
    )

    async def fake_download_asset(*_args: Any, **_kwargs: Any) -> DownloadedAsset:
        return downloaded

    bot = _FakeBot(file_size=downloaded.size)
    dp = _FakeDispatcher(bot)

    monkeypatch.setattr(handlers_core, "STATUS_MESSAGES", status_stub)
    monkeypatch.setattr(handlers_core, "download_asset", fake_download_asset)
    monkeypatch.setattr(handlers_core, "increment_metric", lambda *a, **k: None)
    monkeypatch.setattr(handlers_core, "log_event", lambda *a, **k: None)
    monkeypatch.setattr(handlers_core, "record_timing_metric", lambda *a, **k: None)

    _run(
        _send_job_update(
            dp=dp,
            job=job,
            archive=None,
            config=config,
            db=db,
            error_reporter=reporter,
        )
    )

    assert bot.sent and bot.sent[0]["method"] == "video"
