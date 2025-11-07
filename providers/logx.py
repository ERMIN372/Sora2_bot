import logging
import re


def get_logger(name: str = "providers.gemini"):
    return logging.getLogger(name)


def kv(**kwargs):
    out = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, (bytes, bytearray)):
            v = f"<bytes:{len(v)}>"
        s = str(v)
        if len(s) > 800:
            s = s[:800] + "…<truncated>"
        out[k] = s
    return out


def mask(s):
    if not s:
        return s
    return s[:4] + "…" + s[-4:] if len(s) > 12 else "****"
