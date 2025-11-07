import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from providers import gemini
from providers.base import ProviderJobSubmission
def test_stop_is_not_safety_block(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(gemini, "DEBUG_GEMINI", True)
    monkeypatch.setattr(gemini.SafetyCfg, "ENABLE_NEGATIVE_PROMPT", False)
    monkeypatch.setattr(gemini.SafetyCfg, "ENABLE_AUTO_REPHRASE", False)
    client = object.__new__(gemini.GeminiGenerativeClient)
    client.__dict__["provider_name"] = "gemini-image"
    client._task = "image"
    client._pending = {}
    client._pending_dir = tmp_path
    client._log_debug_event = lambda *args, **kwargs: None
    client._log_generation_feedback = lambda *args, **kwargs: None
    client._persist_pending_record = lambda *args, **kwargs: None
    client._key_mask = "mask"
    client._environment = "test"
    client._model = "gemini-2.5-flash-image"
    client._api_version = "v1"
    client._api_key = "secret"
    client._sanitize_debug_value = gemini.GeminiGenerativeClient._sanitize_debug_value.__get__(
        client, gemini.GeminiGenerativeClient
    )

    payload = {
        "inline_assets": [
            {
                "key": "inline_0",
                "mime": "image/png",
                "bytes": 64,
                "data_uri": "data:image/png;base64,AAAA",
            }
        ],
        "assets_meta": {"inline_0": {"mime": "image/png", "bytes": 64}},
        "finish_reason": "STOP",
        "prompt_feedback": {"safetyRatings": []},
    }
    submission = ProviderJobSubmission(
        status_code=200,
        duration_ms=120,
        job_id="job-1",
        data=payload,
    )

    async def fake_submit(**_kwargs):
        return submission

    client._submit_single_attempt = fake_submit

    async def _run() -> ProviderJobSubmission:
        return await client.enqueue_job(
            prompt="sunset",
            settings={},
            payload={},
            idempotency_key="corr",
        )

    with caplog.at_level("INFO"):
        result = asyncio.run(_run())
    assert isinstance(result, ProviderJobSubmission)
    assert "gemini.safety block" not in caplog.text


def test_predict_404_blocks_further_predict(monkeypatch, caplog):
    monkeypatch.setattr(gemini, "DEBUG_GEMINI", True)
    fake_content_cls = type(
        "FakeContentConfig",
        (),
        {"__init__": lambda self, *args, **kwargs: None, "model_dump": lambda self, mode="json": {}},
    )
    fake_image_cls = type(
        "FakeImagesConfig",
        (),
        {"__init__": lambda self, *args, **kwargs: None, "model_dump": lambda self, mode="json": {}},
    )
    monkeypatch.setattr(gemini.types, "GenerateContentConfig", fake_content_cls, raising=False)
    monkeypatch.setattr(gemini.types, "GenerateImagesConfig", fake_image_cls, raising=False)
    client = object.__new__(gemini.GeminiGenerativeClient)
    client._predict_block_until = None
    client._task = "image"
    client._build_contents = lambda prompt, settings: []
    client._build_generation_config = (
        lambda settings, method: fake_content_cls() if method == "generate_content" else fake_image_cls()
    )
    client._log_debug_event = lambda *args, **kwargs: None

    models = SimpleNamespace(
        generate_images=lambda **_kwargs: None,
        generate_content=lambda **_kwargs: None,
    )
    decision = SimpleNamespace(
        task="image",
        model="gemini-2.5-flash-image",
        method="generate_images",
        api_version="v1beta",
        client=SimpleNamespace(models=models),
        prompt="ping",
        rewrite_notes=None,
    )

    with caplog.at_level("WARNING"):
        client._block_predict_now(120)
    assert "gemini.predict.disabled" in caplog.text
    caplog.clear()
    client._predict_block_until = time.time() + 3600
    with caplog.at_level("WARNING"):
        generator, kwargs, meta = client._prepare_generate_call(
            decision=decision,
            prompt="ping",
            settings={},
            payload={},
            force_mode=None,
        )
    assert meta["mode"] == "generate_content"
    assert generator == models.generate_content
    assert "gemini.predict.skip" in caplog.text


def test_no_image_saves_diag_snapshot(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(gemini, "DEBUG_GEMINI", True)
    monkeypatch.setattr(gemini, "Path", lambda value="": tmp_path / value if value else tmp_path)
    client = object.__new__(gemini.GeminiGenerativeClient)
    client._sanitize_debug_value = gemini.GeminiGenerativeClient._sanitize_debug_value.__get__(
        client, gemini.GeminiGenerativeClient
    )

    decision = SimpleNamespace(model="gemini-2.5-flash-image", prompt="diag", api_version="v1beta")
    payload = {"candidates": [{"finishReason": "NO_IMAGE"}]}
    headers = {"x-generative-ai-finish-reason": "NO_IMAGE"}
    response = SimpleNamespace(response_id="resp-1")
    error = SimpleNamespace(safety_ratings=[{"blocked": True}])

    with caplog.at_level("INFO"):
        path = client._save_no_image_diag(
            decision=decision,
            method="generate_content",
            payload_json=payload,
            headers=headers,
            response=response,
            finish="NO_IMAGE",
            safety_feedback={"safety_ratings": []},
            error=error,
            duration_ms=123,
            corr_id="corr",
        )
    assert path is not None
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["finish"] == "NO_IMAGE"
    assert data["counts"]["candidates"] == 1
    assert "gemini.diag.saved" in caplog.text
