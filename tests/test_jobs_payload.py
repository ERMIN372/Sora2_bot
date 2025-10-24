import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import pytest

from config import (
    Config,
    DEFAULT_CREDIT_PRICE_RUB,
    DEFAULT_FIX_FEE_RUB,
    DEFAULT_MARKUP_PCT,
    PricingConfig,
)
from jobs import JobQueue, PendingJob
from providers.base import ProviderJobStatus, ProviderJobSubmission
from services.gemini_key import mask_gemini_key


@dataclass
class GenerationJobRecord:
    id: str
    user_id: int
    prompt: str
    status: str
    video_url: Optional[str]
    error: Optional[str]
    created_at: datetime
    updated_at: datetime
    image_file_id: Optional[str] = None
    sora_req_id: Optional[str] = None
    size: Optional[str] = None
    seconds: Optional[int] = None
    model: Optional[str] = None
    cost_credits: Optional[int] = None
    username: Optional[str] = None
    corr_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    content_type: str = "video"
    status_message_id: Optional[int] = None
    status_message_index: int = 0
    status_message_updated_at: Optional[datetime] = None
    operation_name: Optional[str] = None
    file_url: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@pytest.fixture()
def config() -> Config:
    pricing = PricingConfig(
        credit_price_rub=DEFAULT_CREDIT_PRICE_RUB,
        markup_pct=DEFAULT_MARKUP_PCT,
        fix_fee_rub=DEFAULT_FIX_FEE_RUB,
        provider_costs={},
    )
    return Config(
        bot_token="test-token",
        gemini_api_key="test-key",
        pricing=pricing,
    )


class FakeDB:
    def __init__(self) -> None:
        self.jobs: Dict[str, GenerationJobRecord] = {}
        self.update_calls: list[tuple] = []
        self.added_credits: list[int] = []
        self.logged_errors: list[Any] = []
        self.idempotency_index: Dict[str, str] = {}

    async def init(self) -> None:  # pragma: no cover - not used
        return None

    async def create_job(self, job: GenerationJobRecord) -> None:  # pragma: no cover
        self.jobs[job.id] = job
        if job.idempotency_key:
            self.idempotency_index[job.idempotency_key] = job.id

    async def find_job_by_idempotency_key(
        self, key: str
    ) -> Optional[GenerationJobRecord]:  # pragma: no cover - tests override
        job_id = self.idempotency_index.get(key)
        if job_id is None:
            return None
        return self.jobs.get(job_id)

    async def update_job(
        self,
        job_id: str,
        status: str,
        video_url: Optional[str] = None,
        file_url: Optional[str] = None,
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
            if operation_name is not None:
                job.operation_name = operation_name
            if error is not None:
                job.error = error
            job.updated_at = datetime.utcnow()
            if status_message_id is not None:
                job.status_message_id = status_message_id
            if status_message_index is not None:
                job.status_message_index = status_message_index
            if status_message_updated_at is not None:
                job.status_message_updated_at = status_message_updated_at

    async def update_job_fields(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        status_message_id: Optional[int] = None,
        status_message_index: Optional[int] = None,
        status_message_updated_at: Optional[datetime] = None,
    ) -> None:
        await self.update_job(
            job_id,
            status or self.jobs[job_id].status,
            status_message_id=status_message_id,
            status_message_index=status_message_index,
            status_message_updated_at=status_message_updated_at,
        )

    async def add_credits(self, user_id: int, credits: int) -> None:
        self.added_credits.append(credits)

    async def log_error_record(self, record) -> bool:
        self.logged_errors.append(record)
        return True

    async def get_job(self, job_id: str) -> Optional[GenerationJobRecord]:
        return self.jobs.get(job_id)

    async def list_pending_jobs(self, limit: int = 100):  # pragma: no cover - unused
        return []


class StubProvider:
    def __init__(self, status: ProviderJobStatus):
        self.status = status
        self.submissions: list[ProviderJobSubmission] = []

    async def enqueue_job(self, **kwargs: Any) -> ProviderJobSubmission:  # pragma: no cover
        submission = ProviderJobSubmission(
            job_id=self.status.job_id,
            status_code=200,
            duration_ms=10,
            data={},
        )
        self.submissions.append(submission)
        return submission

    async def get_job_status(self, job_id: str) -> ProviderJobStatus:
        return self.status

    async def close(self) -> None:  # pragma: no cover - not used
        return None


@pytest.fixture()
def fake_db(config: Config) -> FakeDB:
    db = FakeDB()
    job = GenerationJobRecord(
        id="job-1",
        user_id=42,
        prompt="make a video",
        status="queued",
        video_url=None,
        error=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        model=config.gemini_model_video,
        size="1280x720",
        cost_credits=config.generation_cost_credits,
        username="tester",
        corr_id="corr-1",
    )
    job.idempotency_key = "stub-idemp"
    db.jobs[job.id] = job
    db.idempotency_index[job.idempotency_key] = job.id
    return db


_ORIGINAL_ASYNCIO_SLEEP = asyncio.sleep


def _immediate_sleep(_: float):
    return _ORIGINAL_ASYNCIO_SLEEP(0)


def test_process_job_records_inline_payload(monkeypatch, config: Config, fake_db: FakeDB) -> None:
    async def _run() -> None:
        base64_data = base64.b64encode(b"video").decode("ascii")
        data_uri = f"data:video/mp4;base64,{base64_data}"
        provider_status = ProviderJobStatus(
            job_id="job-1",
            status="completed",
            status_code=200,
            duration_ms=1200,
            assets={"video": "https://storage.example/video.mp4"},
            error=None,
            data={
                "payload": {
                    "type": "video",
                    "mime": "video/mp4",
                    "bytes": 5,
                    "data_uri": data_uri,
                    "uri": "https://storage.example/video.mp4",
                }
            },
        )

        provider = StubProvider(provider_status)
        queue = JobQueue(
            db=fake_db,
            providers={"veo": provider},
            default_provider="veo",
            config=config,
            gate=None,
        )

        pending = PendingJob(
            job_id="job-1",
            user_id=42,
            prompt="make a video",
            corr_id="corr-1",
            size="1280x720",
            model=config.gemini_model_video,
            provider="veo",
            username="tester",
        )

        events: list[dict] = []
        monkeypatch.setattr(
            "jobs.log_event", lambda **kwargs: events.append(kwargs)
        )
        monkeypatch.setattr("jobs.asyncio.sleep", _immediate_sleep)

        await queue._process_job(pending)

        assert fake_db.jobs["job-1"].status == "completed"
        assert fake_db.jobs["job-1"].video_url == "[inline video/mp4, 5 bytes]"
        assert "payload" in fake_db.jobs["job-1"].extra
        assert fake_db.jobs["job-1"].extra["payload"]["mime"] == "video/mp4"
        assert fake_db.jobs["job-1"].extra.get("gemini_key_mask") == mask_gemini_key(config.gemini_api_key)
        assert any(
            event.get("event") == "done" and event.get("extra", {}).get("gemini_key_mask")
            for event in events
        )

    asyncio.run(_run())


def test_process_job_requeues_running_status(monkeypatch, config: Config, fake_db: FakeDB) -> None:
    async def _run() -> None:
        provider_status = ProviderJobStatus(
            job_id="job-1",
            status="running",
            status_code=200,
            duration_ms=500,
            assets={},
            error=None,
            data={},
        )

        provider = StubProvider(provider_status)
        queue = JobQueue(
            db=fake_db,
            providers={"veo": provider},
            default_provider="veo",
            config=config,
            gate=None,
        )

        pending = PendingJob(
            job_id="job-1",
            user_id=42,
            prompt="make a video",
            corr_id="corr-1",
            size="1280x720",
            model=config.gemini_model_video,
            provider="veo",
            username="tester",
        )

        monkeypatch.setattr("jobs.asyncio.sleep", _immediate_sleep)
        monkeypatch.setattr("jobs.log_event", lambda **kwargs: None)

        await queue._process_job(pending)

        assert fake_db.update_calls[-1][1] == "running"
        assert queue._queue.qsize() == 1
        requeued = await queue._queue.get()
        assert requeued.job_id == "job-1"
        queue._queue.task_done()

    asyncio.run(_run())
