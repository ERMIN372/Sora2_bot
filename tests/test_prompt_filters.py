from services.prompt_filters import (
    attach_negative_prompt,
    build_rephrase_prompt,
    scrub_brands_and_persons,
)


def test_scrub_brands_and_persons() -> None:
    original = "Dog drives C3 Corvette to Walmart. Portrait of Napoleon."
    result = scrub_brands_and_persons(original)
    assert "Walmart" not in result
    assert "Napoleon" not in result
    assert "generic subject" in result


def test_attach_negative_prompt_once() -> None:
    prompt = "two pugs running around the room"
    result = attach_negative_prompt(prompt, "family-friendly")
    assert "negative:" in result.lower()


def test_build_rephrase_prompt() -> None:
    prompt = build_rephrase_prompt("girl in bikini on the beach")
    assert "Перефразируй" in prompt
    assert "Оригинал:" in prompt
