"""Command-line diagnostic for Gemini image routing."""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Dict, Optional

from config import load_config
from services import gemini_router


async def _run(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config()
    report = await gemini_router.run_diag(
        config,
        prompt=args.prompt,
        model=args.model,
        attempts=args.n,
    )
    return report


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Gemini diagnostic snapshot")
    parser.add_argument("--prompt", default="Gemini diagnostic ping", help="Prompt text for test calls")
    parser.add_argument("--model", default=None, help="Override image model identifier")
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level for the JSON output (default: 2)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of diagnostic attempts per method",
    )
    args = parser.parse_args(argv)
    report = asyncio.run(_run(args))
    print(json.dumps(report, ensure_ascii=False, indent=args.indent))


if __name__ == "__main__":
    main()
