"""Shared helpers for Gemini safety configuration and observability."""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from google.genai import types as _genai_types

_SAFETY_SETTING_CLS = getattr(_genai_types, "SafetySetting", None)

_PROFILE_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "text": (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_CIVIC_INTEGRITY",
    ),
    "image": (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ),
    "video": (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ),
}

FORCE_THRESHOLD = "BLOCK_NONE"
DEFAULT_THRESHOLD = FORCE_THRESHOLD
FALLBACK_THRESHOLD = "BLOCK_ONLY_HIGH"

_STATE_LOCK = threading.Lock()
_STATE: Dict[str, object] = {
    "profiles": {},
    "last_fallback": {},
}

_THRESHOLD_PATTERN = re.compile(r"(HARM_CATEGORY_[A-Z_]+)")


def _profile_allowed(profile: str) -> Tuple[str, ...]:
    return _PROFILE_CATEGORIES.get(profile, ())


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
    categories: Iterable[str],
    default_threshold: str,
) -> Dict[str, str]:
    thresholds: Dict[str, str] = {}
    for category in categories:
        if not category:
            continue
        thresholds[category] = default_threshold
    return thresholds


def initialise(
    *,
    profile: str,
    force_none: bool,
    settings: Sequence[_genai_types.SafetySetting],
    key_mask: str,
    logger: logging.Logger,
) -> Tuple[List[str], Dict[str, str]]:
    """Prepare thresholds and update shared state for a profile."""

    allowed = list(_profile_allowed(profile))
    thresholds = _ensure_thresholds(allowed, FORCE_THRESHOLD)
    dropped: List[str] = []
    if not force_none and settings:
        pairs = _settings_to_pairs(settings, default_threshold=DEFAULT_THRESHOLD)
        for category, threshold in pairs:
            if category not in thresholds:
                dropped.append(category)
                continue
            thresholds[category] = threshold

    _configure_state(
        profile=profile,
        force_none=force_none,
        order=allowed,
        thresholds=thresholds,
        key_mask=key_mask,
        dropped=dropped,
        logger=logger,
    )
    return allowed, thresholds


def _configure_state(
    *,
    profile: str,
    force_none: bool,
    order: Sequence[str],
    thresholds: Mapping[str, str],
    key_mask: str,
    dropped: Sequence[str],
    logger: logging.Logger,
) -> None:
    with _STATE_LOCK:
        profiles: Dict[str, Dict[str, object]] = dict(
            _STATE.get("profiles", {})  # type: ignore[arg-type]
        )
        previous = profiles.get(profile, {})
        previous_force = bool(previous.get("force_none"))
        previous_order = tuple(previous.get("order", ()))
        previous_thresholds: Dict[str, str] = dict(
            previous.get("thresholds", {})  # type: ignore[arg-type]
        )

    new_state = {
        "configured": True,
        "force_none": bool(force_none),
        "order": tuple(order),
        "thresholds": {category: thresholds.get(category, FORCE_THRESHOLD) for category in order},
        "key_mask": key_mask or "",
        "dropped": list(dict.fromkeys(filter(None, dropped))),
    }

    with _STATE_LOCK:
        profiles = dict(_STATE.get("profiles", {}))  # type: ignore[arg-type]
        profiles[profile] = new_state
        _STATE["profiles"] = profiles

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
        logger.info(
            "force-none: ON profile=%s key=%s thresholds=%s",
            profile,
            key_mask or "",
            thresholds_label,
        )
    if dropped:
        logger.info(
            "safety-config dropped profile=%s key=%s categories=%s",
            profile,
            key_mask or "",
            ", ".join(dict.fromkeys(filter(None, dropped))),
        )


def _coerce_setting(category: str, threshold: str) -> _genai_types.SafetySetting:
    if _SAFETY_SETTING_CLS is None:  # pragma: no cover - fallback for legacy clients
        return {"category": category, "threshold": threshold}  # type: ignore[return-value]
    return _SAFETY_SETTING_CLS(category=category, threshold=threshold)


def build_profile_settings(
    profile: str,
    *,
    thresholds: Optional[Mapping[str, str]] = None,
) -> List[_genai_types.SafetySetting]:
    """Build :class:`SafetySetting` objects for a specific profile."""

    allowed = list(_profile_allowed(profile))
    base_thresholds = _ensure_thresholds(allowed, FORCE_THRESHOLD)
    if thresholds:
        for category, value in thresholds.items():
            if category in base_thresholds:
                base_thresholds[category] = value
    return [_coerce_setting(category, base_thresholds[category]) for category in allowed]


def build_text_safety_settings(
    *, thresholds: Optional[Mapping[str, str]] = None
) -> List[_genai_types.SafetySetting]:
    return build_profile_settings("text", thresholds=thresholds)


def build_image_safety_settings(
    *, thresholds: Optional[Mapping[str, str]] = None
) -> List[_genai_types.SafetySetting]:
    return build_profile_settings("image", thresholds=thresholds)


def build_video_safety_settings(
    *, thresholds: Optional[Mapping[str, str]] = None
) -> List[_genai_types.SafetySetting]:
    return build_profile_settings("video", thresholds=thresholds)


def build_override(
    profile: str,
    base_thresholds: Mapping[str, str],
    category: str,
    *,
    threshold: str,
) -> Dict[str, str]:
    allowed = _profile_allowed(profile)
    updated = {key: base_thresholds.get(key, FORCE_THRESHOLD) for key in allowed}
    if category in allowed:
        updated[category] = threshold
    return updated


def filter_profile_settings(
    profile: str,
    safety_settings: Sequence[_genai_types.SafetySetting],
) -> Tuple[List[_genai_types.SafetySetting], List[str]]:
    allowed = list(_profile_allowed(profile))
    if not allowed:
        return list(safety_settings), []
    allowed_set = set(allowed)
    thresholds = _ensure_thresholds(allowed, FORCE_THRESHOLD)
    dropped: List[str] = []
    pairs = _settings_to_pairs(safety_settings, default_threshold=DEFAULT_THRESHOLD)
    for category, threshold in pairs:
        if category not in allowed_set:
            dropped.append(category)
            continue
        thresholds[category] = threshold
    filtered = [_coerce_setting(category, thresholds[category]) for category in allowed]
    return filtered, list(dict.fromkeys(filter(None, dropped)))


def describe_settings(
    safety_settings: Sequence[_genai_types.SafetySetting],
) -> Tuple[List[str], Dict[str, str]]:
    pairs = _settings_to_pairs(safety_settings, default_threshold=DEFAULT_THRESHOLD)
    order: List[str] = []
    thresholds: Dict[str, str] = {}
    for category, threshold in pairs:
        if category not in order:
            order.append(category)
        thresholds[category] = threshold
    return order, thresholds


def record_fallback(
    profile: str,
    category: Optional[str],
    *,
    key_mask: str,
    provider: str,
    logger: logging.Logger,
) -> None:
    label = category or "n/a"
    logger.warning(
        "fallback: %s -> %s provider=%s profile=%s key=%s",
        label,
        FALLBACK_THRESHOLD,
        provider,
        profile,
        key_mask or "",
    )
    timestamp = time.time()
    with _STATE_LOCK:
        current: Dict[str, Dict[str, float]] = dict(
            _STATE.get("last_fallback", {})  # type: ignore[arg-type]
        )
        profile_map = dict(current.get(profile, {}))
        profile_map[label] = timestamp
        current[profile] = profile_map
        _STATE["last_fallback"] = current


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
        profiles: Dict[str, Dict[str, object]] = dict(
            _STATE.get("profiles", {})  # type: ignore[arg-type]
        )
        last_fallback = dict(_STATE.get("last_fallback", {}))  # type: ignore[arg-type]
    profile_snapshot: Dict[str, object] = {}
    for profile, info in profiles.items():
        allowed = _profile_allowed(profile)
        thresholds = info.get("thresholds", {})
        profile_snapshot[profile] = {
            "allowed": list(allowed),
            "thresholds": [
                (category, str(thresholds.get(category, FORCE_THRESHOLD)))
                for category in allowed
            ],
            "configured": bool(info.get("configured")),
            "force_none": bool(info.get("force_none")),
            "dropped": list(info.get("dropped", ())),
        }
    return {"profiles": profile_snapshot, "last_fallback": last_fallback}


def current_configuration(profile: str) -> Tuple[List[str], Dict[str, str], bool, bool]:
    with _STATE_LOCK:
        profiles: Mapping[str, Mapping[str, object]] = _STATE.get("profiles", {})  # type: ignore[assignment]
        info = profiles.get(profile, {})
    if not info:
        allowed = list(_profile_allowed(profile))
        thresholds = _ensure_thresholds(allowed, FORCE_THRESHOLD)
        return allowed, thresholds, False, False
    order = list(info.get("order", ()))  # type: ignore[list-item]
    thresholds = dict(info.get("thresholds", {}))  # type: ignore[arg-type]
    configured = bool(info.get("configured"))
    force_none = bool(info.get("force_none"))
    return order, thresholds, configured, force_none


__all__ = [
    "DEFAULT_THRESHOLD",
    "FALLBACK_THRESHOLD",
    "FORCE_THRESHOLD",
    "build_image_safety_settings",
    "build_override",
    "build_profile_settings",
    "build_text_safety_settings",
    "build_video_safety_settings",
    "current_configuration",
    "describe_settings",
    "extract_conflict_category",
    "filter_profile_settings",
    "health_snapshot",
    "initialise",
    "record_fallback",
]
