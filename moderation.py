"""Prompt moderation helpers for Sora bot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

PolicyScope = str


@dataclass(slots=True)
class PreflightResult:
    """Outcome of a lightweight prompt moderation pass."""

    blocked: bool
    sanitized_prompt: str
    auto_sanitized: bool = False
    reason: Optional[str] = None
    scope: Optional[PolicyScope] = None
    user_message: Optional[str] = None


def policy_message(scope: PolicyScope) -> str:
    """Return a short, user-facing description for a policy scope."""

    return ""


def run_preflight(prompt: str) -> PreflightResult:
    """Allow every prompt without additional moderation."""

    text = (prompt or "").strip()
    return PreflightResult(blocked=False, sanitized_prompt=text)
