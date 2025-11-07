#!/usr/bin/env python3
"""CLI helper to inspect the configured Sora Videos endpoint."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Tuple

from config import load_config
from providers.sora_video import SoraVideoClient, sora_self_test


async def _compute_endpoint(config) -> Tuple[str, str]:
    client = SoraVideoClient(config=config)
    try:
        url = client.responses_url()
        path = client.responses_path()
    finally:
        await client.close()
    return url, path


async def _run(args: argparse.Namespace) -> int:
    config = load_config()
    url, path = await _compute_endpoint(config)
    print(f"Videos path: {path}")
    print(f"Videos URL:  {url}")
    if "/v1/v1beta/" in url:
        print("\033[91mWARNING:\033[0m suspect misconfigured endpoint (contains /v1/v1beta/)", file=sys.stderr)
        return 2
    if args.self_test:
        status, details = await sora_self_test(config)
        print(json.dumps({"status": status, "details": details}, ensure_ascii=False, indent=2))
        return 0 if status == "ok" else 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the Sora Videos endpoint configuration.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Perform a minimal POST request to validate the endpoint.",
    )
    args = parser.parse_args()
    exit_code = asyncio.run(_run(args))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
