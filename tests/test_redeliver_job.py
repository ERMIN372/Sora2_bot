from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from db import GenerationJobRecord
import importlib.util
from pathlib import Path


def _load_script_module():
    path = Path("scripts/mre_redeliver_job.py")
    spec = importlib.util.spec_from_file_location("mre_redeliver_job", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mre_redeliver_job = _load_script_module()


class _FakeDB:
    def __init__(self, job: GenerationJobRecord) -> None:
        self._job = job
        self.update_calls: List[Dict[str, Any]] = []

    async def init(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def get_job(self, job_id: str) -> GenerationJobRecord | None:
        if self._job.id == job_id:
            return self._job
        return None

    async def update_job(self, job_id: str, status: str, **fields: Any) -> None:
        self.update_calls.append({"job_id": job_id, "status": status, **fields})


class _FakeSession:
    async def close(self) -> None:
        return None


class _FakeBot:
    def __init__(self, token: str) -> None:
        self.token = token
        self.session = _FakeSession()


def test_redeliver_uses_existing_url_no_enqueue(monkeypatch) -> None:
    now = datetime.utcnow()
    job = GenerationJobRecord(
        id="job-redeliver-1",
        user_id=77,
        prompt="demo",
        status="completed",
        video_url="https://v3b.fal.media/files/b/foo/output.mp4",
        video_id=None,
        error=None,
        created_at=now,
        updated_at=now,
    )
    job.extra = {"provider": "kling"}

    fake_db = _FakeDB(job)
    sent: List[str] = []

    async def fake_send_job_update(**kwargs: Any) -> None:
        sent.append(kwargs["job"].id)

    def fake_load_config() -> Any:
        class _Cfg:
            bot_token = "TEST_TOKEN"
            archive_channel_id = None
            environment = "test"

        return _Cfg()

    monkeypatch.setattr(mre_redeliver_job, "load_config", fake_load_config)
    monkeypatch.setattr(mre_redeliver_job, "create_database", lambda _cfg: fake_db)
    monkeypatch.setattr(mre_redeliver_job, "Bot", _FakeBot)
    monkeypatch.setattr(mre_redeliver_job, "_send_job_update", fake_send_job_update)

    import asyncio

    result = asyncio.run(mre_redeliver_job._run(job.id, force_completed=False))

    assert result == 0
    assert sent == [job.id]
    assert fake_db.update_calls == []
