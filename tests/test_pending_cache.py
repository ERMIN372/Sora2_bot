import asyncio
from enum import Enum
from types import SimpleNamespace

from config import Config
from providers import gemini
from providers.openai_image import OpenAIImageClient


class _StubModelList:
    def list(self):  # pragma: no cover - simple stub
        return []


class _StubGenAIClient:
    def __init__(self, *args, **kwargs) -> None:
        self.models = _StubModelList()


class _StubHttpOptions:
    def __init__(self, *args, **kwargs) -> None:
        self.api_version = kwargs.get("api_version")


class _StubHarmBlockThreshold(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


def test_gemini_get_job_status_recovers_from_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(gemini.genai, "Client", _StubGenAIClient)
    monkeypatch.setattr(gemini.types, "HttpOptions", _StubHttpOptions)
    monkeypatch.setattr(gemini.types, "HarmBlockThreshold", _StubHarmBlockThreshold)
    monkeypatch.setattr(gemini.GeminiGenerativeClient, "_load_available_models", lambda self: None)
    monkeypatch.setattr(gemini, "get_gemini_router", lambda _cfg: SimpleNamespace(_clients={}))
    monkeypatch.setattr(gemini, "_SAFETY_FALLBACK_THRESHOLD", _StubHarmBlockThreshold.LOW)

    config = Config(bot_token="token", gemini_api_key="key", environment="test")
    client = gemini.GeminiImageClient(config=config)
    client._pending_dir = tmp_path / "gemini"
    client._pending_dir.mkdir(parents=True, exist_ok=True)

    job_id = "job-1"
    payload = {
        "inline_assets": [
            {
                "key": "inline_0",
                "data_uri": "data:image/png;base64,AAAA",
                "mime": "image/png",
                "bytes": 4,
            }
        ]
    }
    meta = {"prompt": "hello"}
    client._persist_pending_record(job_id, payload=payload, status_code=200, duration_ms=120, meta=meta)

    status = asyncio.run(client.get_job_status(job_id))

    assert status.status == "completed"
    assert status.assets.get("inline_0") == "data:image/png;base64,AAAA"
    assert not (client._pending_dir / "job-1.json").exists()


def test_openai_get_job_status_recovers_from_disk(tmp_path, monkeypatch):
    config = Config(
        bot_token="token",
        gemini_api_key="key",
        environment="test",
        openai_api_key="sk-test",
    )
    client = OpenAIImageClient(config=config)
    client._pending_dir = tmp_path / "openai"
    client._pending_dir.mkdir(parents=True, exist_ok=True)

    job_id = "img-1"
    record = {
        "payload": {},
        "inline_assets": [
            {
                "key": "inline_0",
                "data_uri": "data:image/png;base64,BBBB",
                "mime": "image/png",
                "bytes": 4,
            }
        ],
        "assets": [{"key": "inline_0", "inline": True, "mime": "image/png", "bytes": 4}],
        "asset_map": {"inline_0": "data:image/png;base64,BBBB"},
        "status_code": 200,
        "duration_ms": 50,
    }
    client._persist_pending_record(job_id, record)

    status = asyncio.run(client.get_job_status(job_id))

    assert status.status == "completed"
    assert status.assets.get("inline_0") == "data:image/png;base64,BBBB"
    assert not (client._pending_dir / "img-1.json").exists()
