import asyncio
import base64
from enum import Enum
from types import SimpleNamespace

from config import Config
from providers import gemini
from providers.openai_image import OpenAIImageClient
from services.gemini_router import RouteDecision


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


def test_gemini_inline_image_submission_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(gemini.genai, "Client", lambda *args, **kwargs: SimpleNamespace(models=SimpleNamespace()))
    monkeypatch.setattr(gemini.types, "HttpOptions", _StubHttpOptions)
    monkeypatch.setattr(gemini.types, "HarmBlockThreshold", _StubHarmBlockThreshold)
    monkeypatch.setattr(gemini.types, "SafetySetting", lambda *args, **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(gemini.GeminiGenerativeClient, "_load_available_models", lambda self: None)
    monkeypatch.setattr(gemini, "_SAFETY_FALLBACK_THRESHOLD", _StubHarmBlockThreshold.LOW)

    class _StubConfigObject:
        def __init__(self, **data):
            self._data = data

        def model_dump(self, mode="json"):
            return dict(self._data)

    monkeypatch.setattr(
        gemini.GeminiGenerativeClient,
        "_build_generation_config",
        lambda self, settings, method: _StubConfigObject(),
    )
    monkeypatch.setattr(
        gemini.GeminiGenerativeClient,
        "_build_contents",
        lambda self, prompt, settings=None: [{"role": "user", "parts": [{"text": prompt}]}],
    )

    response_bytes = b"PNG"
    encoded = base64.b64encode(response_bytes).decode("ascii")

    class _Image:
        def __init__(self, data: str) -> None:
            self.data = data
            self.mime_type = "image/png"

    class _StubResponse:
        def __init__(self) -> None:
            self.images = [_Image(encoded)]
            self.response_id = "resp-1"

        def model_dump(self, mode="json"):
            return {
                "images": [{"data": encoded, "mime_type": "image/png"}],
                "response_id": self.response_id,
            }

    stub_response = _StubResponse()

    def _generate_images(**kwargs):
        return stub_response

    stub_models = SimpleNamespace(generate_images=_generate_images)
    stub_client = SimpleNamespace(models=stub_models)
    decision = RouteDecision(
        task="image",
        model="gemini-2.5-flash-image",
        prompt="draw a cat",
        method="generate_images",
        api_version="v1",
        client=stub_client,
        supported_methods=("generate_images",),
    )

    class _StaticRouter:
        def __init__(self, decision):
            self._decision = decision

        def route(self, *args, **kwargs):
            return self._decision

    router = _StaticRouter(decision)
    monkeypatch.setattr(gemini, "get_gemini_router", lambda _cfg: router)

    async def _fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _fake_to_thread)

    config = Config(bot_token="token", gemini_api_key="key", environment="test")
    client = gemini.GeminiImageClient(config=config)
    client._pending_dir = tmp_path / "gemini-inline"
    client._pending_dir.mkdir(parents=True, exist_ok=True)

    class _RateLimiter:
        async def acquire(self):
            return None

        async def penalize(self, *args, **kwargs):
            return None

    client._rate_limiter = _RateLimiter()

    client._router = router
    client._client = stub_client

    async def _run_flow():
        submission = await client.enqueue_job(
            prompt="draw a cat",
            settings={},
            payload={},
            idempotency_key="idem-1",
        )
        status = await client.get_job_status(submission.job_id)
        return submission, status

    submission, status = asyncio.run(_run_flow())

    assert submission.job_id
    assert status.status == "completed"
    assert status.status_code == 200
    assert status.assets
    assert not (client._pending_dir / f"{submission.job_id}.json").exists()


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
