#!/usr/bin/env python3
"""MRE for fal.ai Kling 2.6 Motion Control queue flow.

Usage:
  FAL_KEY=... python scripts/mre_kling_fal_queue.py \
    --model pro \
    --image-url https://.../image.png \
    --video-url https://.../motion.mp4 \
    --character-orientation video \
    --prompt "cinematic lighting"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from urllib.request import Request, urlopen

QUEUE_BASE = "https://queue.fal.run"
SUBMIT_MODELS = {
    "standard": "fal-ai/kling-video/v2.6/standard/motion-control",
    "pro": "fal-ai/kling-video/v2.6/pro/motion-control",
}


def _request(method: str, url: str, api_key: str, payload: dict | None = None) -> tuple[int, dict]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(url=url, data=body, method=method)
    req.add_header("Authorization", f"Key {api_key}")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=60) as resp:
        status_code = int(resp.status)
        raw = resp.read().decode("utf-8")
    return status_code, (json.loads(raw) if raw else {})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["standard", "pro"], default="standard")
    parser.add_argument("--image-url", required=True)
    parser.add_argument("--video-url", required=True)
    parser.add_argument("--character-orientation", choices=["image", "video"], required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--keep-original-sound", action="store_true", default=True)
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--max-polls", type=int, default=80)
    args = parser.parse_args()

    api_key = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
    if not api_key:
        raise SystemExit("Set FAL_KEY or FAL_API_KEY")

    submit_model = SUBMIT_MODELS[args.model]
    submit_url = f"{QUEUE_BASE}/{submit_model}"
    payload = {
        "input": {
            "image_url": args.image_url,
            "video_url": args.video_url,
            "character_orientation": args.character_orientation,
            "keep_original_sound": bool(args.keep_original_sound),
        }
    }
    if args.prompt.strip():
        payload["input"]["prompt"] = args.prompt.strip()

    print("[submit]", submit_url)
    submit_status, submit = _request("POST", submit_url, api_key, payload)
    request_id = submit.get("request_id")
    print(f"submit status={submit_status}:", json.dumps(submit, ensure_ascii=False))
    if not request_id:
        raise SystemExit("No request_id in submit response")

    status_url = submit.get("status_url") or f"{QUEUE_BASE}/fal-ai/kling-video/requests/{request_id}/status"
    response_url = submit.get("response_url") or f"{QUEUE_BASE}/fal-ai/kling-video/requests/{request_id}"
    print(f"queue urls: status_url={status_url} response_url={response_url}")

    start = time.monotonic()
    for i in range(1, args.max_polls + 1):
        time.sleep(args.poll_interval)
        status_code, status_data = _request("GET", status_url, api_key)
        status = str(status_data.get("status") or "").lower()
        elapsed = time.monotonic() - start
        if status_data.get("response_url"):
            response_url = status_data["response_url"]
        print(
            f"poll#{i} t={elapsed:.1f}s code={status_code} status={status} "
            f"queue={status_data.get('queue_position')} request_id={request_id}"
        )
        if status_code == 202 or status in {"in_queue", "in_progress", "running"}:
            continue
        if status_code == 200 and status == "completed":
            result_code, result = _request("GET", response_url, api_key)
            print(f"result status={result_code}:", json.dumps(result, ensure_ascii=False))
            return 0
        if status == "failed":
            print("failed:", json.dumps(status_data, ensure_ascii=False))
            return 2

    print("poll retries exhausted")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
