import logging
import os
import re

MASK = re.compile(r'([A-Za-z0-9_\-]{8,})')


def mask_secret(s: str) -> str:
    if not s:
        return s
    return s[:4] + "…" + s[-4:] if len(s) > 12 else "****"


def kv(**kwargs):
    out = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, (bytes, bytearray)):
            value = f"<bytes:{len(value)}>"
        text = str(value)
        if len(text) > 800:
            text = text[:800] + "…<truncated>"
        out[key] = text
    return out


def get_logger(name: str = "providers.gemini"):
    return logging.getLogger(name)
