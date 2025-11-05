from providers.gemini import _has_generate_content_flag


def test_has_generate_content_flag_handles_camel_case_key() -> None:
    meta = {"supportedGenerationMethods": ["generateContent", "embedContent"]}
    assert _has_generate_content_flag(meta)


def test_has_generate_content_flag_handles_capabilities_dict() -> None:
    meta = {"capabilities": {"generateContent": True, "embedContent": True}}
    assert _has_generate_content_flag(meta)
