from config import Config
from handlers._core import _image_model_options, _resolve_product_key


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
