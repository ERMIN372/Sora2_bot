from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RetryDecision:
    do_retry: bool
    next_prompt: str


def next_attempt(
    original_prompt: str,
    last_prompt: str,
    last_response_json: dict,
    cfg: Any,  # SafetyCfg-like object
    rephraser: Callable[[str], str],
    scrubber: Callable[[str], str],
    attach_neg: Callable[[str, str], str],
) -> RetryDecision:
    if not getattr(cfg, "ENABLE_AUTO_REPHRASE", False) and not getattr(
        cfg, "ENABLE_PRE_CLEAN", False
    ):
        return RetryDecision(False, last_prompt)

    cleaned = scrubber(last_prompt)
    rewritten = rephraser(cleaned) if getattr(cfg, "ENABLE_AUTO_REPHRASE", False) else cleaned
    final = (
        attach_neg(rewritten, getattr(cfg, "NEGATIVE_PROMPT_BASE", ""))
        if getattr(cfg, "ENABLE_NEGATIVE_PROMPT", False)
        else rewritten
    )

    if final == last_prompt:
        return RetryDecision(False, last_prompt)
    return RetryDecision(True, final)


__all__ = ["RetryDecision", "next_attempt"]
