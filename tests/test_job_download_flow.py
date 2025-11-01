import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from db import GenerationJobRecord
from jobs import JobQueue, PendingJob
from providers.base import ProviderAPIError, ProviderJobStatus


def _run(coro):
    return asyncio.run(coro)


class InlineFakeDB:
    def __init__(self) -> None:
        self.jobs: Dict[str, GenerationJobRecord] = {}
        self.update_calls: List[tuple] = []
        self.added_credits: List[int] = []

    async def create_job(self, job: GenerationJobRecord) -> None:
        self.jobs[job.id] = job

    async def update_job(
        self,
        job_id: str,
        status: str,
        video_url: Optional[str] = None,
        file_url: Optional[str] = None,
        video_id: Optional[str] = None,
        operation_name: Optional[str] = None,
        error: Optional[str] = None,
        status_message_id: Optional[int] = None,
        status_message_index: Optional[int] = None,
        status_message_updated_at: Optional[datetime] = None,
    ) -> None:
        self.update_calls.append(
            (
                job_id,
                status,
                video_url,
                file_url,
                video_id,
                operation_name,
                error,
                status_message_id,
                status_message_index,
                status_message_updated_at,
            )
        )
        job = self.jobs.get(job_id)
        if job:
            job.status = status
            if video_url is not None:
                job.video_url = video_url
            if file_url is not None:
                job.file_url = file_url
            if video_id is not None:
                job.video_id = video_id
            if error is not None:
                job.error = error
            job.updated_at = datetime.now(timezone.utc)

    async def add_credits(self, user_id: int, amount: int) -> None:
        self.added_credits.append(amount)

    async def log_error_record(self, record: Any) -> bool:
        return True

    async def get_job(self, job_id: str) -> Optional[GenerationJobRecord]:
        return self.jobs.get(job_id)


class FakeProvider:
    provider_name = "openai-video"

    def __init__(
        self,
        *,
        status: ProviderJobStatus,
        download_path: Optional[Path] = None,
        download_error: Optional[ProviderAPIError] = None,
    ) -> None:
        self.status = status
        self.download_path = download_path
        self.download_error = download_error
        self.download_calls: List[str] = []
        self._last_download_meta: Dict[str, Any] = {}

    async def get_job_status(self, job_id: str) -> ProviderJobStatus:
        return self.status

    async def download_content(self, video_id: str) -> Path:
        self.download_calls.append(video_id)
        if self.download_error is not None:
            raise self.download_error
        assert self.download_path is not None
        self._last_download_meta = {
            "status_code": 200,
            "duration_ms": 25,
            "request_id": "req-inline",
            "attempt": len(self.download_calls),
            "url": "https://api.openai.com/v1beta/videos/test/content",
        }
        return self.download_path

    async def content(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover - fallback path
        raise AssertionError("content handler should not be used in inline download tests")


def test_inline_download_marks_job_completed(make_openai_video_config) -> None:
    config = make_openai_video_config()
    db = InlineFakeDB()
    tmp_dir = Path(tempfile.mkdtemp())
    download_path = tmp_dir / "vid-success.mp4"
    download_path.write_bytes(b"video-bytes")

    status = ProviderJobStatus(
        job_id="job-1",
        status="completed",
        assets={"primary": "https://cdn.local/video.mp4"},
        error=None,
        data={},
        status_code=200,
        duration_ms=120,
    )
    provider = FakeProvider(status=status, download_path=download_path)
    job_queue = JobQueue(
        db=db,
        providers={"openai": provider},
        default_provider="openai",
        config=config,
    )
    record = GenerationJobRecord(
        id="job-1",
        user_id=123,
        prompt="make video",
        status="queued",
        video_url=None,
        video_id="vid-1",
        error=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        size="720p",
        model="openai",
        cost_credits=config.generation_cost_credits,
        username="tester",
        corr_id="corr-1",
        content_type="video",
    )
    _run(db.create_job(record))
    pending = PendingJob(
        job_id="job-1",
        user_id=123,
        prompt="make video",
        corr_id="corr-1",
        size="720p",
        model="openai",
        provider="openai",
        username="tester",
        original_prompt="make video",
        sanitized_prompt="make video",
        task_type="video_retrieve",
        provider_job_id="job-1",
        video_id="vid-1",
    )

    _run(job_queue._handle_video_retrieve(pending))

    assert provider.download_calls == ["vid-1"]
    assert db.update_calls[-1][1] == "completed"
    assert db.jobs["job-1"].status == "completed"
    assert db.added_credits == []


def test_inline_download_failure_refunds(make_openai_video_config) -> None:
    config = make_openai_video_config()
    db = InlineFakeDB()
    failure = ProviderAPIError(
        provider="openai-video",
        status_code=500,
        message="download failed",
        error_type="server_error",
        error_code="server_error",
        provider_message="failure",
    )
    status = ProviderJobStatus(
        job_id="job-2",
        status="completed",
        assets={"primary": "https://cdn.local/video.mp4"},
        error=None,
        data={},
        status_code=200,
        duration_ms=120,
    )
    provider = FakeProvider(status=status, download_error=failure)
    job_queue = JobQueue(
        db=db,
        providers={"openai": provider},
        default_provider="openai",
        config=config,
    )
    record = GenerationJobRecord(
        id="job-2",
        user_id=321,
        prompt="make video",
        status="queued",
        video_url=None,
        video_id="vid-2",
        error=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        size="720p",
        model="openai",
        cost_credits=config.generation_cost_credits,
        username="tester",
        corr_id="corr-2",
        content_type="video",
    )
    _run(db.create_job(record))
    pending = PendingJob(
        job_id="job-2",
        user_id=321,
        prompt="make video",
        corr_id="corr-2",
        size="720p",
        model="openai",
        provider="openai",
        username="tester",
        original_prompt="make video",
        sanitized_prompt="make video",
        task_type="video_retrieve",
        provider_job_id="job-2",
        video_id="vid-2",
    )

    _run(job_queue._handle_video_retrieve(pending))

    assert provider.download_calls == ["vid-2"]
    assert any(call[1] == "failed_to_download" for call in db.update_calls)
    assert db.added_credits == [config.generation_cost_credits]
    assert db.jobs["job-2"].status == "failed_to_download"
