from config import Config
from handlers._core import (
    _image_model_options,
    _resolve_product_key,
    _resolve_task_label,
    _veo_series_label,
    _video_model_options,
)


def test_image_model_options_include_gemini_pro_preview() -> None:
    config = Config(
        bot_token="test-token",
        gemini_api_key="test-key",
        gemini_model_image="gemini-2.5-flash-image",
    )

    options = _image_model_options(config)
    models = {option.model: option for option in options}

    assert "gemini-2.5-flash-image" in models
    assert "gemini-3-pro-image-preview" in models
    assert models["gemini-3-pro-image-preview"].provider == "gemini-image-pro"


def test_resolve_product_key_for_gemini_pro_preview_image() -> None:
    assert (
        _resolve_product_key(
            "image",
            provider="gemini-image-pro",
            model="gemini-3-pro-image-preview",
        )
        == "image_pro"
    )


def test_nano_banana_pro_routes_to_image_generate_task() -> None:
    assert _resolve_task_label("image", "gemini-image-pro") == "image_generate"


def test_image_category_always_routes_to_image_generate() -> None:
    """Any image category job should route to image_generate, regardless of provider."""
    assert _resolve_task_label("image", "gemini-image") == "image_generate"
    assert _resolve_task_label("image", "gemini-image-pro") == "image_generate"
    assert _resolve_task_label("image", "dall-e-3") == "image_generate"
    assert _resolve_task_label("image", "unknown-provider") == "image_generate"
    assert _resolve_task_label("image", "") == "image_generate"
    assert _resolve_task_label("image", None) == "image_generate"


def test_veo_menu_model_mapping() -> None:
    config = Config(
        bot_token="test-token",
        gemini_api_key="test-key",
    )

    options = _video_model_options(config)
    models = {option.label: option.model for option in options}

    assert any(model == "veo-3.1-fast-generate-preview" for model in models.values())
    assert any(model == "veo-3.1-generate-preview" for model in models.values())


def test_veo3_no_duplicate_labels_in_menu(monkeypatch) -> None:
    """Video model keyboard must not contain two entries with the same label."""
    monkeypatch.setattr("handlers._core.list_veo_video_models", lambda _cfg: [])
    config = Config(
        bot_token="test-token",
        gemini_api_key="test-key",
        # Use an older model id that differs from _DEFAULT_VEO3_MODEL but
        # resolves to the same UI label.
        gemini_model_video="veo-3.0-generate-001",
    )
    options = _video_model_options(config)
    labels = [opt.label for opt in options]
    assert len(labels) == len(set(labels)), f"Duplicate labels in video menu: {labels}"


def test_veo_series_label_fast_is_veo3() -> None:
    """veo-3.1-fast-* must be labelled Veo 3, not Veo 3.1."""
    assert _veo_series_label("veo-3.1-fast-generate-preview", "veo") == "Veo 3"
    assert _veo_series_label("veo-3.1-generate-preview", "veo31") == "Veo 3.1"
    assert _veo_series_label("veo-3.0-generate-001", "veo") == "Veo 3"
    assert _veo_series_label(None, "veo") == "Veo 3"
    assert _veo_series_label(None, "veo31") == "Veo 3.1"
