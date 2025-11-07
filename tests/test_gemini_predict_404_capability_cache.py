from types import SimpleNamespace

from providers import gemini as g


def test_predict_404_disables_generate_images(monkeypatch):
    g._MODEL_CAPABILITIES.clear()
    model_name = "gemini-2.5-flash-image"
    g.mark_predict_capability(model_name, False)
    assert g.get_predict_capability(model_name) is False

    DummyConfig = type("DummyConfig", (), {"model_dump": lambda self, mode=None: {}})
    monkeypatch.setattr(g.types, "GenerateImagesConfig", DummyConfig)
    monkeypatch.setattr(g.types, "GenerateContentConfig", DummyConfig)

    client = object.__new__(g.GeminiGenerativeClient)
    client._task = "image"
    client._predict_block_until = 0
    client._build_contents = lambda **kwargs: [
        {"parts": [{"text": kwargs.get("prompt", "")}]}]
    client._build_generation_config = lambda settings, method: DummyConfig()

    decision = SimpleNamespace(
        task="image",
        model=model_name,
        method="generate_images",
        api_version="v1",
        prompt="draw",
        rewrite_notes=None,
        client=SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **_: None,
                generate_images=lambda **_: None,
            )
        ),
    )
    settings = {}
    payload = {}
    generator, request_kwargs, meta = client._prepare_generate_call(
        decision=decision,
        prompt="draw",
        settings=settings,
        payload=payload,
    )
    assert meta["mode"] == "generate_content"
    assert request_kwargs["model"] == model_name
    assert callable(generator)
    g._MODEL_CAPABILITIES.clear()
