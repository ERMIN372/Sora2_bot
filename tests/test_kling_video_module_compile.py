from __future__ import annotations

import py_compile
from pathlib import Path


def test_kling_video_module_compiles() -> None:
    target = Path(__file__).resolve().parents[1] / "providers" / "kling_video.py"
    py_compile.compile(str(target), doraise=True)


def test_kling_video_has_no_async_with_session_request_pattern() -> None:
    """Guard against reintroducing the parse-error-prone request pattern.

    Deployment incidents showed stale artifacts with
    ``async with session.request(...)`` in this module; keep a textual guard
    to detect accidental reintroduction in future edits.
    """

    target = Path(__file__).resolve().parents[1] / "providers" / "kling_video.py"
    source = target.read_text(encoding="utf-8")
    assert "async with session.request(" not in source
