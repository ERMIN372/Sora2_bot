"""Shared helpers for Gemini safety configuration and observability."""
from __future__ import annotations

import logging
import re
import threading
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from google.genai import types as _genai_types

_SAFETY_SETTING_CLS = getattr(_genai_types, "SafetySetting", None)

_FORCE_CATEGORIES: Tuple[str, ...] = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
    "HARM_CATEGORY_VIOLENCE",
)
_DEFAULT_CATEGORIES: Tuple[str, ...] = (
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
)

FORCE_THRESHOLD = "BLOCK_NONE"
DEFAULT_THRESHOLD = "BLOCK_MEDIUM_AND_ABOVE"
FALLBACK_THRESHOLD = "BLOCK_ONLY_HIGH"

_STATE_LOCK = threading.Lock()
_STATE: Dict[str, object] = {
    "configured": False,
    "force_none": False,
    "thresholds": {},
    "order": (),
    "last_fallback": None,
    "key_mask": "",
}

_THRESHOLD_PATTERN = re.compile(r"(HARM_CATEGORY_[A-Z_]+)")


def _normalise_category(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return str(value)


def _extract_threshold(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or default
    return str(value)


def _settings_to_pairs(
    settings: Sequence[_genai_types.SafetySetting],
    *,
    default_threshold: str,
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for setting in settings:
        category = _normalise_category(getattr(setting, "category", None))
        if not category:
            continue
        threshold = _extract_threshold(getattr(setting, "threshold", None), default_threshold)
        pairs.append((category, threshold))
    return pairs


def _ensure_thresholds(
    categories: Sequence[str],
    default_threshold: str,
) -> Tuple[List[str], Dict[str, str]]:
    order = [category for category in categories if category]
    thresholds = {category: default_threshold for category in order}
    return order, thresholds


def initialise(
    *,
    force_none: bool,
    settings: Sequence[_genai_types.SafetySetting],
    key_mask: str,
    logger: logging.Logger,
) -> Tuple[List[str], Dict[str, str]]:
    """Prepare thresholds and update shared state."""

    if force_none:
        order, thresholds = _ensure_thresholds(_FORCE_CATEGORIES, FORCE_THRESHOLD)
    else:
        pairs = _settings_to_pairs(settings, default_threshold=DEFAULT_THRESHOLD)
        if not pairs:
            order, thresholds = _ensure_thresholds(_DEFAULT_CATEGORIES, DEFAULT_THRESHOLD)
        else:
            order = [category for category, _ in pairs]
            thresholds = {category: threshold for category, threshold in pairs}

    _configure_state(force_none, order, thresholds, key_mask, logger)
    return order, thresholds


def _configure_state(
    force_none: bool,
    order: Sequence[str],
    thresholds: Mapping[str, str],
    key_mask: str,
    logger: logging.Logger,
) -> None:
    with _STATE_LOCK:
        previous_force = bool(_STATE.get("force_none"))
        previous_order = tuple(_STATE.get("order", ()))
        previous_thresholds: Dict[str, str] = dict(
            _STATE.get("thresholds", {})  # type: ignore[arg-type]
        )
    with _STATE_LOCK:
        _STATE["configured"] = True
        _STATE["force_none"] = bool(force_none)
        _STATE["order"] = tuple(order)
        _STATE["thresholds"] = {category: thresholds.get(category, "n/a") for category in order}
        _STATE["key_mask"] = key_mask or ""
        changed = (
            not previous_order
            or previous_force != bool(force_none)
            or tuple(order) != previous_order
            or any(
                previous_thresholds.get(category) != thresholds.get(category, FORCE_THRESHOLD)
                for category in order
            )
        )
    if force_none and changed:
        thresholds_label = ", ".join(
            f"{category}={thresholds.get(category, FORCE_THRESHOLD)}" for category in order
        )
        logger.info("force-none: ON key=%s thresholds=%s", key_mask or "", thresholds_label)


def build_safety_settings(
    order: Sequence[str],
    thresholds: Mapping[str, str],
) -> List[_genai_types.SafetySetting]:
    if _SAFETY_SETTING_CLS is None:
        return [
            {"category": category, "threshold": thresholds.get(category, FORCE_THRESHOLD)}
            for category in order
        ]
    return [
        _SAFETY_SETTING_CLS(category=category, threshold=thresholds.get(category, FORCE_THRESHOLD))
        for category in order
    ]


def build_override(
    base_thresholds: Mapping[str, str],
    category: str,
    *,
    threshold: str,
) -> Dict[str, str]:
    updated = {key: value for key, value in base_thresholds.items()}
    if category in updated:
        updated[category] = threshold
    return updated


def record_fallback(
    category: Optional[str],
    *,
    key_mask: str,
    provider: str,
    logger: logging.Logger,
) -> None:
    label = category or "n/a"
    logger.warning(
        "fallback: %s -> %s provider=%s key=%s",
        label,
        FALLBACK_THRESHOLD,
        provider,
        key_mask or "",
    )
    with _STATE_LOCK:
        _STATE["last_fallback"] = label


def extract_conflict_category(exc: Exception) -> Optional[str]:
    parts: List[str] = []
    for attr in ("message", "details", "status", "response"):
        value = getattr(exc, attr, None)
        if not value:
            continue
        if isinstance(value, (list, tuple, set)):
            parts.extend(str(item) for item in value)
        else:
            parts.append(str(value))
    parts.append(str(exc))
    payload = " ".join(parts)
    lowered = payload.lower()
    if "threshold" not in lowered or "support" not in lowered:
        return None
    match = _THRESHOLD_PATTERN.search(payload)
    if match:
        return match.group(1)
    return None


def health_snapshot() -> Dict[str, object]:
    with _STATE_LOCK:
        order = tuple(_STATE.get("order", ()))
        thresholds: Mapping[str, str] = _STATE.get("thresholds", {})  # type: ignore[assignment]
        last_fallback = _STATE.get("last_fallback")
    return {
        "thresholds": [(category, thresholds.get(category, "n/a")) for category in order],
        "last_fallback": last_fallback,
    }


def current_configuration() -> Tuple[List[str], Dict[str, str], bool]:
    with _STATE_LOCK:
        order = list(_STATE.get("order", ()))
        thresholds: Dict[str, str] = dict(_STATE.get("thresholds", {}))  # type: ignore[arg-type]
        configured = bool(_STATE.get("configured"))
    return order, thresholds, configured


__all__ = [
    "DEFAULT_THRESHOLD",
    "FALLBACK_THRESHOLD",
    "FORCE_THRESHOLD",
    "build_override",
    "build_safety_settings",
    "current_configuration",
    "extract_conflict_category",
    "health_snapshot",
    "initialise",
    "record_fallback",
]
