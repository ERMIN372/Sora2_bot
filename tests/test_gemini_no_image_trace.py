import base64
import json
import zipfile
from types import SimpleNamespace

from providers import gemini as g


def test_no_image_trace_exports_case(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "_TRACE_DIR", tmp_path)
    monkeypatch.setattr(g, "_TRACE_CASES_DIR", tmp_path / "cases")
    monkeypatch.setattr(g, "_TRACE_INLINE_DIR", tmp_path / "inline")
    g._TRACE_HISTORY.clear()
    monkeypatch.setattr(g, "GEMINI_TRACE", True)
    monkeypatch.setattr(g, "GEMINI_TRACE_SAVE_JSON", True)
    monkeypatch.setattr(g, "GEMINI_TRACE_SAVE_B64", True)

    corr_id = "corr-test"
    decision = SimpleNamespace(
        model="gemini-2.5-flash-image",
        api_version="v1beta",
        task="image",
        method="generate_images",
        prompt="draw",
        rewrite_notes=None,
        client=SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **_: None,
                generate_images=lambda **_: None,
            )
        ),
    )
    dummy_client = SimpleNamespace(_base_url="https://example.com", _api_version="v1beta")
    request_payload = {"model": decision.model, "contents": []}
    request_event = g._trace_request(
        client=dummy_client,
        decision=decision,
        method="generate_content",
        api_version="v1beta",
        attempt=1,
        corr_id=corr_id,
        request_kwargs=request_payload,
        payload={},
        force_mode=None,
        payload_path=None,
    )
    image_b64 = base64.b64encode(b"png-data").decode("ascii")
    payload_json = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"inline_data": {"data": image_b64, "mime_type": "image/png"}}
                    ]
                }
            }
        ]
    }
    g._trace_response(
        client=dummy_client,
        decision=decision,
        method="generate_content",
        api_version="v1beta",
        corr_id=corr_id,
        status=200,
        headers={"x-request-id": "req-1"},
        duration_ms=123,
        payload_json=payload_json,
        request_event=request_event,
        response_id="resp-1",
    )
    archive_path = g.dump_gemini_case(corr_id)
    assert archive_path is not None
    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert "summary.json" in names
        summary = json.loads(archive.read("summary.json"))
        assert summary["corr_id"] == corr_id
        assert len(summary["events"]) >= 2


def test_analyse_no_image_context(monkeypatch):
    error = g.GeminiImageEmptyError(
        provider="gemini-image",
        finish_reason="NO_IMAGE",
        response_id="resp-ctx",
        headers={"x-generative-ai-finish-reason": "NO_IMAGE"},
    )
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inline_data": {
                                "data": base64.b64encode(b"").decode("ascii"),
                                "mime_type": "image/png",
                            }
                        }
                    ]
                }
            }
        ]
    }
    dummy_response = SimpleNamespace(
        response_id="resp-ctx", sdk_http_response=SimpleNamespace(headers={})
    )
    finish, _, safety = g._analyse_empty_image_response(
        payload,
        response=dummy_response,
        error=error,
    )
    assert finish == "NO_IMAGE"
    assert safety is False
    context = getattr(error, "analysis_context", {})
    assert context.get("response_id") == "resp-ctx"
