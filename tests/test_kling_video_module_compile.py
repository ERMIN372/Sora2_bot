from __future__ import annotations

import py_compile
from pathlib import Path


def test_kling_video_module_compiles() -> None:
    target = Path(__file__).resolve().parents[1] / "providers" / "kling_video.py"
    py_compile.compile(str(target), doraise=True)
