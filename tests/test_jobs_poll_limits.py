from jobs import _poll_max_attempts_for_provider


def test_poll_max_attempts_for_kling_is_extended() -> None:
    assert _poll_max_attempts_for_provider(provider="kling_mc", default_attempts=5) >= 24


def test_poll_max_attempts_for_non_kling_unchanged() -> None:
    assert _poll_max_attempts_for_provider(provider="veo", default_attempts=5) == 5
