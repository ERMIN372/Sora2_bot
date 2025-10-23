"""Prompt moderation helpers for the video generation bot."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

PolicyScope = str


_POLICY_MESSAGES: dict[str, str] = {
    "unknown": (
        "Запрос отклонён по соображениям безопасности. Попробуйте переформулировать "
        "описание."
    ),
    "policy.sexual": "Мы не можем создавать откровенный сексуальный контент.",
    "policy.violence": "Запрос содержит сцены насилия, которые запрещены политикой.",
    "policy.hate": "Мы не можем генерировать контент с призывами к ненависти.",
    "policy.self_harm": "Мы не поддерживаем запросы, связанные с саморазрушением.",
    "policy.weapons": "Мы не можем создавать инструкции или рекламу оружия.",
    "policy.drugs": "Мы не можем распространять информацию о запрещённых веществах.",
    "policy.terrorism": "Контент с поддержкой терроризма запрещён.",
    "policy.brand": (
        "Мы не можем использовать зарегистрированные товарные знаки. Опишите идею без "
        "упоминания брендов."
    ),
}


@dataclass(slots=True)
class PreflightResult:
    """Outcome of a lightweight prompt moderation pass."""

    blocked: bool
    sanitized_prompt: str
    auto_sanitized: bool = False
    reason: Optional[str] = None
    scope: Optional[PolicyScope] = None
    user_message: Optional[str] = None


def _stringify_summary(value: Any) -> Optional[str]:
    """Convert arbitrary provider summaries into a readable text snippet."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        parts = []
        for item in value:
            item_text = _stringify_summary(item)
            if item_text:
                parts.append(item_text)
        if not parts:
            return None
        return "; ".join(dict.fromkeys(parts))
    if isinstance(value, Mapping):
        candidates = []
        for key in ("summary", "message", "text", "detail", "reason", "explanation"):
            if key in value:
                text = _stringify_summary(value[key])
                if text:
                    candidates.append(text)
        if candidates:
            return "; ".join(dict.fromkeys(candidates))
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value).strip() or None


def _normalise_categories(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        categories: list[str] = []
        for item in value:
            categories.extend(_normalise_categories(item))
        seen: set[str] = set()
        result: list[str] = []
        for category in categories:
            if category not in seen:
                seen.add(category)
                result.append(category)
        return result
    return []


def _extract_scope(payload: Mapping[str, Any]) -> Optional[str]:
    for key in ("scope", "policy_scope", "category", "code", "label"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def policy_message(scope: PolicyScope) -> str:
    """Return a short, user-facing description for a policy scope."""

    key = (scope or "unknown").strip().lower()
    return _POLICY_MESSAGES.get(key, _POLICY_MESSAGES["unknown"])


def classify_safety_response(
    response: Optional[Mapping[str, Any]]
) -> tuple[PolicyScope, list[str], Optional[str]]:
    """Extract policy scope, categories and summary from provider payloads."""

    scope: PolicyScope = "unknown"
    categories: list[str] = []
    summary: Optional[str] = None

    if not isinstance(response, Mapping):
        return scope, categories, summary

    payload: Mapping[str, Any] = response
    safety = response.get("safety")
    if isinstance(safety, Mapping):
        payload = safety

    extracted_scope = _extract_scope(payload)
    if extracted_scope:
        scope = extracted_scope

    cat_value = payload.get("categories") or payload.get("category")
    categories = _normalise_categories(cat_value)

    summary = _stringify_summary(payload.get("summary") or payload.get("message"))

    violations = payload.get("violations")
    if isinstance(violations, Sequence) and not isinstance(violations, (bytes, bytearray, str)):
        for violation in violations:
            if not isinstance(violation, Mapping):
                continue
            if scope == "unknown":
                violation_scope = _extract_scope(violation)
                if violation_scope:
                    scope = violation_scope
            if not categories:
                categories = _normalise_categories(
                    violation.get("categories") or violation.get("category")
                )
            if summary is None:
                summary = _stringify_summary(
                    violation.get("summary")
                    or violation.get("message")
                    or violation.get("detail")
                    or violation.get("reason")
                )

    if scope == "unknown":
        extra_scope = _extract_scope(response)
        if extra_scope:
            scope = extra_scope

    if not categories:
        categories = _normalise_categories(response.get("categories"))

    if summary is None:
        summary = _stringify_summary(response.get("summary") or response.get("message"))

    return scope, categories, summary


def render_error_json(summary: Optional[Any]) -> Optional[str]:
    """Render a JSON string with provider error details for storage/logging."""

    if summary is None:
        return None

    payload: Any
    if isinstance(summary, str):
        text = summary.strip()
        if not text:
            return None
        payload = {"error": text}
    elif isinstance(summary, Mapping):
        payload = summary
    elif isinstance(summary, Sequence) and not isinstance(summary, (bytes, bytearray, str)):
        items = [_stringify_summary(item) for item in summary]
        filtered = [item for item in items if item]
        if not filtered:
            return None
        payload = {"errors": filtered}
    else:
        text = str(summary).strip()
        if not text:
            return None
        payload = {"error": text}

    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def run_preflight(prompt: str) -> PreflightResult:
    """Allow every prompt without additional moderation."""

    text = (prompt or "").strip()
    return PreflightResult(blocked=False, sanitized_prompt=text)
