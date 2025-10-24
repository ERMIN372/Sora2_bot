#!/usr/bin/env python3
"""Minimal smoke test for Gemini Veo video generation."""
from __future__ import annotations

import os
import sys
import time

from google import genai
from google.genai import types as t


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is required", file=sys.stderr)
        return 1
    model = os.environ.get("GEMINI_MODEL_VIDEO", "veo-3.0-generate-001")
    prompt = os.environ.get("SMOKE_PROMPT", "A quick landscape flyover, 1 second")

    client = genai.Client(
        api_key=api_key,
        http_options=t.HTTPOptions(api_version="v1beta"),
    )
    operation = client.models.generate_videos(
        model=model,
        prompt=prompt,
        config=t.GenerateVideosConfig(duration_seconds=1),
    )
    attempts = 0
    while not getattr(operation, "done", False) and attempts < 10:
        attempts += 1
        time.sleep(3)
        operation = client.operations.get(operation)
    if not getattr(operation, "done", False):
        raise RuntimeError("operation not done")
    videos = getattr(operation.result, "videos", None)
    if not videos:
        response = getattr(operation.result, "response", {}) if hasattr(operation, "result") else {}
        videos = response.get("videos", []) if isinstance(response, dict) else []
    if not videos:
        raise RuntimeError("no media assets in result")
    video_file = getattr(videos[0], "video", None)
    if video_file is None:
        raise RuntimeError("missing video file reference")
    blob = client.files.download(video_file)
    payload = bytes(blob)
    if not payload:
        raise RuntimeError("empty video download")
    print(f"OK video op: done {getattr(operation, 'done', False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
