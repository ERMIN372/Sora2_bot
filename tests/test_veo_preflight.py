from pathlib import Path

from config import (
    Config,
    DEFAULT_CREDIT_PRICE_RUB,
    DEFAULT_FIX_FEE_RUB,
    DEFAULT_MARKUP_PCT,
    PricingConfig,
)
from handlers._core import _video_model_options
from providers.veo_video import VeoVideoClient
from services.gemini_catalog import preflight_check_veo_model


def _config(**overrides):
    pricing = PricingConfig(
        credit_price_rub=DEFAULT_CREDIT_PRICE_RUB,
        markup_pct=DEFAULT_MARKUP_PCT,
        fix_fee_rub=DEFAULT_FIX_FEE_RUB,
        provider_costs={},
    )
    base = dict(
        bot_token="test-token",
        gemini_api_key="test-key",
        pricing=pricing,
    )
    base.update(overrides)
    return Config(**base)


def test_veo_preflight_blocks_charge_when_model_missing(monkeypatch) -> None:
    config = _config()
    monkeypatch.setattr(
        "services.gemini_catalog._list_veo_model_methods",
        lambda *_args, **_kwargs: {"veo-3.1-fast-generate-preview": {"predictLongRunning"}},
    )

    charge_called = False
    preflight = preflight_check_veo_model(config, "veo-3.1-generate-preview")
    if preflight.available:
        charge_called = True

    assert preflight.available is False
    assert preflight.reason == "model_not_found"
    assert charge_called is False


def test_veo_uses_vertex_api_not_generative_language(monkeypatch) -> None:
    class DummyModels:
        pass

    class DummyClient:
        def __init__(self):
            self.models = DummyModels()
            self.operations = type("Ops", (), {"get": staticmethod(lambda operation: operation)})()

    monkeypatch.setattr("providers.veo_video.get_media_client", lambda _config: DummyClient())

    config = _config(
        gemini_api_mode="vertex",
        gemini_api_key="",
        vertex_project_id="demo-project",
        vertex_location="us-central1",
    )
    client = VeoVideoClient(config=config)

    assert "aiplatform.googleapis.com" in client._base_url
    assert "generativelanguage.googleapis.com" not in client._base_url


def test_veo_model_id_mapping_valid(monkeypatch) -> None:
    monkeypatch.setattr("handlers._core.list_veo_video_models", lambda _config: [])
    options = _video_model_options(_config())
    models = {entry.model for entry in options}

    assert "veo-3.1-generate-preview" in models
    assert "veo-3.1-generate-001" not in models

    source_roots = [Path("config.py"), Path("handlers/_core.py"), Path("services/gemini_catalog.py")]
    for path in source_roots:
        text = path.read_text(encoding="utf-8")
        assert "veo-3.1-generate-001" not in text


def test_veo_option_routing_keeps_veo3_and_veo31_separate(monkeypatch) -> None:
    monkeypatch.setattr(
        "handlers._core.list_veo_video_models",
        lambda _config: ["veo-3.1-generate-preview"],
    )
    options = _video_model_options(_config())

    veo3 = next((item for item in options if item.model == "veo-3.1-fast-generate-preview"), None)
    veo31 = next((item for item in options if item.model == "veo-3.1-generate-preview"), None)

    assert veo3 is not None
    assert veo3.provider == "veo"
    assert "fast" in veo3.model

    assert veo31 is not None
    assert veo31.provider == "veo31"
    assert "veo-3.1" in veo31.model


def test_veo_preflight_allows_veo_model_without_advertised_predict_method(monkeypatch) -> None:
    config = _config()
    monkeypatch.setattr(
        "services.gemini_catalog._list_veo_model_methods",
        lambda _config: {"veo-3.1-fast-generate-preview": {"generate_content"}},
    )

    preflight = preflight_check_veo_model(config, "veo-3.1-fast-generate-preview")

    assert preflight.reason == "ok"
    assert preflight.available is True


def test_veo_preflight_empty_catalog_returns_no_models(monkeypatch) -> None:
    """When models.list returns nothing, reason must be 'no_models'."""
    config = _config()
    monkeypatch.setattr(
        "services.gemini_catalog._list_veo_model_methods",
        lambda *_args, **_kwargs: {},
    )

    preflight = preflight_check_veo_model(config, "veo-3.1-fast-generate-preview")

    assert preflight.available is False
    assert preflight.reason == "no_models"


def test_veo_preflight_model_not_found_vs_method_not_supported(monkeypatch) -> None:
    """Preflight must distinguish 'model_not_found' from 'method_not_supported'."""
    config = _config()

    # Model present but no video method — veo-* override should still pass
    monkeypatch.setattr(
        "services.gemini_catalog._list_veo_model_methods",
        lambda *_args, **_kwargs: {"veo-3.1-fast-generate-preview": {"generate_content"}},
    )
    result = preflight_check_veo_model(config, "veo-3.1-fast-generate-preview")
    assert result.available is True
    assert result.reason == "ok"

    # Model absent from catalog
    result2 = preflight_check_veo_model(config, "veo-3.1-generate-preview")
    assert result2.available is False
    assert result2.reason == "model_not_found"


def test_trend_veo_choice_does_not_resolve_to_veo31_model(monkeypatch) -> None:
    from handlers._core import _resolve_trend_model

    config = _config(gemini_model_video="veo-3.1-generate-preview")
    monkeypatch.setattr(
        "handlers._core.list_veo_video_models",
        lambda _config: ["veo-3.1-generate-preview"],
    )

    provider, model = _resolve_trend_model("veo", config)

    assert provider == "veo"
    assert model == "veo-3.1-fast-generate-preview"
    assert "fast" in model
