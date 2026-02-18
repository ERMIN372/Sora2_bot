#!/usr/bin/env python3
"""Read-only preflight for Veo Gemini API models (no generation/charges)."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
TARGET_MODELS = (
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview",
)


def _api_key() -> str:
    key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
    return key


def _http_get_json(path: str, *, api_key: str) -> dict[str, Any]:
    url = f"{API_BASE}{path}"
    separator = "&" if "?" in url else "?"
    request_url = f"{url}{separator}key={urllib.parse.quote(api_key)}"
    req = urllib.request.Request(request_url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected response type for {path}: {type(data)!r}")
    return data


def _strip_model_name(name: str) -> str:
    if "/" in name:
        return name.rsplit("/", 1)[-1]
    return name


def main() -> int:
    api_key = _api_key()
    try:
        payload = _http_get_json("/models", api_key=api_key)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"GET /models failed: HTTP {exc.code}\n{body}", file=sys.stderr)
        return 2

    models = payload.get("models")
    if not isinstance(models, list):
        print("GET /models returned no models list", file=sys.stderr)
        return 2

    index: dict[str, dict[str, Any]] = {}
    for entry in models:
        if not isinstance(entry, dict):
            continue
        raw_name = str(entry.get("name") or "")
        short = _strip_model_name(raw_name)
        if short:
            index[short] = entry

    print("Veo preflight via GET /v1beta/models")
    for model_id in TARGET_MODELS:
        entry = index.get(model_id)
        if entry is None:
            print(f"- {model_id}: found=False supportedGenerationMethods=[]")
            continue
        methods = entry.get("supportedGenerationMethods")
        if not isinstance(methods, list):
            methods = []
        methods_text = ", ".join(str(item) for item in methods)
        print(f"- {model_id}: found=True supportedGenerationMethods=[{methods_text}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
