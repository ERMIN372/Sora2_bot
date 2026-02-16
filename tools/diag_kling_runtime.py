"""Runtime diagnostics for Kling provider deployment issues.

Usage:
    python tools/diag_kling_runtime.py
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys


TARGET = Path("providers/kling_video.py")
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    print("cwd=", Path.cwd())
    print("python=", os.sys.version.replace("\n", " "))
    print("RAILWAY_GIT_COMMIT_SHA=", os.getenv("RAILWAY_GIT_COMMIT_SHA", ""))

    if not TARGET.exists():
        print("kling_file_exists=false path=", TARGET)
        return 2

    source = TARGET.read_text(encoding="utf-8")
    lines = source.splitlines()
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    forbidden = "async with session.request("
    bad_lines = [idx + 1 for idx, line in enumerate(lines) if forbidden in line]

    print("kling_file_exists=true path=", TARGET)
    print("kling_sha256=", digest)
    print("contains_forbidden_async_with=", bool(bad_lines))
    print("forbidden_line_numbers=", bad_lines)
    if len(lines) >= 125:
        print("line_125=", lines[124])
    else:
        print("line_125=<missing>")

    try:
        import providers.kling_video as module

        print("import_ok=true")
        print("impl_rev=", getattr(module, "KLING_FAL_IMPL_REV", ""))
    except Exception as exc:  # pragma: no cover - runtime diagnostic
        print("import_ok=false")
        print("import_error=", repr(exc))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
