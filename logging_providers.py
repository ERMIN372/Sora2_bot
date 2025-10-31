"""Lightweight helper for provider-specific logging hooks.

This module registers a virtual ``logging.providers`` package that exposes
:func:`log_error`.  The shim avoids shadowing the standard :mod:`logging`
module while still allowing call sites to import
``from logging.providers import log_error`` as instructed.
"""
from __future__ import annotations

import json
import logging as _logging
import sys
from types import ModuleType
from typing import Mapping, MutableMapping, Optional

from observability import log_event


_LOGGER = _logging.getLogger("providers")


def _coerce_detail(detail: Optional[Mapping[str, object]]) -> MutableMapping[str, object]:
    if not detail:
        return {}
    if isinstance(detail, dict):
        return dict(detail)
    try:
        return dict(detail.items())  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - defensive fallback
        return {"detail": str(detail)}


def log_error(*, provider: str, message: str, detail: Optional[Mapping[str, object]] = None) -> None:
    """Log a structured provider error to the standard logger and observability."""

    provider_key = (provider or "unknown").lower()
    payload: MutableMapping[str, object] = _coerce_detail(detail)
    payload.setdefault("provider", provider_key)
    payload.setdefault("message", message)

    try:
        serialised = json.dumps(payload, ensure_ascii=False)
    except Exception:  # pragma: no cover - defensive serialisation
        serialised = str(payload)

    _LOGGER.error("provider_error %s", serialised)
    log_event(
        level="ERROR",
        event="provider_error",
        provider=provider_key,
        error_msg_short=message,
        extra=payload,
    )


_module = ModuleType("logging.providers")
_module.log_error = log_error  # type: ignore[attr-defined]
_module.__doc__ = "Namespace module exposing provider logging helpers."

sys.modules.setdefault("logging.providers", _module)
setattr(_logging, "providers", _module)


__all__ = ["log_error"]
