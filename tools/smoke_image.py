#!/usr/bin/env python3
"""Minimal smoke test for Gemini image generation."""
from __future__ import annotations

import os
import sys

from google import genai
from google.genai import types as t


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is required", file=sys.stderr)
        return 1
    model = os.environ.get("GEMINI_MODEL_IMAGE", "gemini-2.5-flash-image")
    prompt = os.environ.get("SMOKE_PROMPT", "test image, monochrome icon")
    debug_enabled = bool(os.environ.get("DEBUG_GEMINI"))
    if debug_enabled:
        print(
            "[debug] gemini.smoke config model=%s prompt_len=%s" % (model, len(prompt or ""))
        )

    client = genai.Client(
        api_key=api_key,
        http_options=t.HTTPOptions(api_version="v1beta"),
    )
    if debug_enabled:
        version = getattr(client._http_options, "api_version", "v1beta")  # type: ignore[attr-defined]
        print("[debug] gemini.smoke http_options api_version=%s" % version)
    response = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=t.GenerateImagesConfig(mime_type="image/png"),
    )
    images = getattr(response, "images", None) or []
    if not images:
        raise RuntimeError("no images returned")
    if debug_enabled:
        print("[debug] gemini.smoke images=%s" % len(images))
    data = getattr(images[0], "data", None)
    if isinstance(data, str):
        import base64

        data = base64.b64decode(data)
    if not data:
        raise RuntimeError("empty image payload")
    size = len(bytes(data))
    if debug_enabled:
        print("[debug] gemini.smoke image_bytes=%s" % size)
    print(f"OK image bytes: {size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
