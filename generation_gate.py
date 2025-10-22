"""Request gating and locking for generation workflows."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from typing import Any, Dict, Optional

from observability import log_event

log = logging.getLogger(__name__)


_WHITESPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


class GateDecision(str, Enum):
    """Possible outcomes of :func:`GenerationRequestGate.evaluate`."""

    ALLOW = "ALLOW"
    BLOCK_BUSY = "BLOCK_BUSY"
    BLOCK_DUPLICATE = "BLOCK_DUPLICATE"


@dataclass(slots=True)
class GateResult:
    """Result returned by :class:`GenerationRequestGate`."""

    decision: GateDecision
    idempotency_key: Optional[str] = None
    timeout_released: bool = False


def normalize_prompt(prompt: str) -> str:
    """Return a normalised representation of *prompt*.

    The normalisation removes HTML tags, collapses repeated whitespace, strips
    leading/trailing spaces and forces lower case while keeping the semantic
    content intact.
    """

    cleaned = _HTML_TAG_RE.sub(" ", prompt or "")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip().lower()


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, frozenset, list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return value
    return str(value)


def _stable_hash(payload: Any) -> str:
    serialised = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return sha256(serialised.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class _StatusLock:
    idempotency_key: str
    corr_id: str
    model: str
    options_hash: str
    normalized_prompt_hash: str
    set_at: float
    ttl: float

    def is_expired(self, now: float) -> bool:
        return now - self.set_at >= self.ttl


@dataclass(slots=True)
class _UserState:
    debounce_until: float = 0.0
    status_lock: Optional[_StatusLock] = None
    recent_keys: Dict[str, float] = field(default_factory=dict)

    def describe(self, now: float) -> str:
        if self.status_lock and not self.status_lock.is_expired(now):
            return "busy"
        if self.debounce_until > now:
            return "debounce"
        if self.status_lock and self.status_lock.is_expired(now):
            return "expired"
        return "free"

    def cleanup_recent(self, now: float, ttl: float) -> None:
        for key, ts in list(self.recent_keys.items()):
            if now - ts >= ttl:
                self.recent_keys.pop(key, None)

    def is_empty(self) -> bool:
        return not self.status_lock and not self.recent_keys and self.debounce_until <= 0


class GenerationRequestGate:
    """Coordinate access to generation providers with per-user locking."""

    def __init__(
        self,
        *,
        debounce_ttl: float = 2.5,
        duplicate_ttl: float = 60.0,
        video_timeout: float = 15 * 60.0,
        image_timeout: float = 2 * 60.0,
    ) -> None:
        self._states: Dict[int, _UserState] = {}
        self._locks: Dict[int, asyncio.Lock] = {}
        self._debounce_ttl = debounce_ttl
        self._duplicate_ttl = duplicate_ttl
        self._video_timeout = video_timeout
        self._image_timeout = image_timeout

    async def evaluate(
        self,
        *,
        user_id: int,
        corr_id: str,
        model: str,
        normalized_prompt: str,
        options: Dict[str, Any],
    ) -> GateResult:
        """Check whether a new generation request may proceed."""

        mutex = self._locks.setdefault(user_id, asyncio.Lock())
        async with mutex:
            state = self._states.setdefault(user_id, _UserState())
            now = time.time()
            state.cleanup_recent(now, self._duplicate_ttl)
            normalized_prompt_hash = sha256(normalized_prompt.encode("utf-8")).hexdigest()
            options_hash = _stable_hash(options)
            idempotency_payload = {
                "model": model,
                "prompt": normalized_prompt,
                "options": options,
                "user_id": user_id,
            }
            idempotency_key = _stable_hash(idempotency_payload)

            decision = GateDecision.ALLOW
            dedup_result = "new"
            timeout_released = False
            previous_state = state.describe(now)
            busy_lock = state.status_lock
            if busy_lock and busy_lock.is_expired(now):
                timeout_released = True
                self._log_release(
                    user_id=user_id,
                    corr_id=busy_lock.corr_id,
                    lock=busy_lock,
                    reason="timeout",
                )
                state.status_lock = None
                busy_lock = None
            busy = busy_lock is not None
            debounce_active = state.debounce_until > now

            if busy_lock:
                if busy_lock.idempotency_key == idempotency_key:
                    decision = GateDecision.BLOCK_DUPLICATE
                    dedup_result = "active_duplicate"
                else:
                    decision = GateDecision.BLOCK_BUSY
            elif idempotency_key in state.recent_keys:
                decision = GateDecision.BLOCK_DUPLICATE
                dedup_result = "recent_duplicate"

            lock_set_at: Optional[float] = None
            lock_ttl = None

            if decision is GateDecision.ALLOW:
                state.debounce_until = now + self._debounce_ttl
                ttl = self._resolve_ttl(options)
                lock = _StatusLock(
                    idempotency_key=idempotency_key,
                    corr_id=corr_id,
                    model=model,
                    options_hash=options_hash,
                    normalized_prompt_hash=normalized_prompt_hash,
                    set_at=now,
                    ttl=ttl,
                )
                state.status_lock = lock
                state.recent_keys[idempotency_key] = now
                lock_set_at = lock.set_at
                lock_ttl = lock.ttl
            else:
                state.recent_keys.setdefault(idempotency_key, now)

            self._log_gate_event(
                user_id=user_id,
                corr_id=corr_id,
                model=model,
                normalized_prompt_hash=normalized_prompt_hash,
                idempotency_key=idempotency_key,
                dedup_result=dedup_result,
                previous_state=previous_state,
                debounce_active=debounce_active,
                busy=busy,
                decision=decision,
                lock_set_at=lock_set_at,
                lock_ttl=lock_ttl,
            )

            if state.is_empty():
                self._states.pop(user_id, None)
            else:
                self._states[user_id] = state

            return GateResult(
                decision=decision,
                idempotency_key=idempotency_key if decision is GateDecision.ALLOW else None,
                timeout_released=timeout_released,
            )

    async def release(
        self,
        *,
        user_id: int,
        corr_id: str,
        reason: str,
        status: Optional[str] = None,
    ) -> None:
        mutex = self._locks.setdefault(user_id, asyncio.Lock())
        async with mutex:
            state = self._states.get(user_id)
            if not state or not state.status_lock:
                return
            if state.status_lock.corr_id != corr_id:
                return
            lock = state.status_lock
            state.status_lock = None
            self._log_release(
                user_id=user_id,
                corr_id=corr_id,
                lock=lock,
                reason=reason,
                status=status,
            )
            if state.is_empty():
                self._states.pop(user_id, None)

    def _resolve_ttl(self, options: Dict[str, Any]) -> float:
        category = str(options.get("category") or "video").lower()
        if category == "image":
            return self._image_timeout
        return self._video_timeout

    def _log_gate_event(
        self,
        *,
        user_id: int,
        corr_id: str,
        model: str,
        normalized_prompt_hash: str,
        idempotency_key: str,
        dedup_result: str,
        previous_state: str,
        debounce_active: bool,
        busy: bool,
        decision: GateDecision,
        lock_set_at: Optional[float],
        lock_ttl: Optional[float],
    ) -> None:
        extra = {
            "lock_state_before": previous_state,
            "lock_free": previous_state == "free",
            "lock_debounce": debounce_active,
            "lock_busy": busy,
            "idempotency_key": idempotency_key,
            "dedup_result": dedup_result,
            "allow": decision is GateDecision.ALLOW,
            "block_busy": decision is GateDecision.BLOCK_BUSY,
            "block_duplicate": decision is GateDecision.BLOCK_DUPLICATE,
            "lock_set_at": lock_set_at,
            "lock_cleared_at": None,
            "lock_ttl_sec": lock_ttl,
            "model": model,
            "normalized_prompt_hash": normalized_prompt_hash,
        }
        log_event(
            level="INFO",
            event="gate",  # structured event for request gating
            corr_id=corr_id,
            user_id=user_id,
            model=model,
            extra=extra,
        )

    def _log_release(
        self,
        *,
        user_id: int,
        corr_id: str,
        lock: _StatusLock,
        reason: str,
        status: Optional[str] = None,
    ) -> None:
        now = time.time()
        extra = {
            "lock_state_before": "busy",
            "lock_free": True,
            "lock_debounce": False,
            "lock_busy": False,
            "idempotency_key": lock.idempotency_key,
            "dedup_result": reason,
            "allow": False,
            "block_busy": False,
            "block_duplicate": False,
            "lock_set_at": lock.set_at,
            "lock_cleared_at": now,
            "lock_ttl_sec": lock.ttl,
            "model": lock.model,
            "normalized_prompt_hash": lock.normalized_prompt_hash,
            "gate_release_reason": reason,
            "gate_release_status": status,
        }
        log_event(
            level="INFO",
            event="gate_release",
            corr_id=corr_id,
            user_id=user_id,
            model=lock.model,
            extra=extra,
        )


__all__ = ["GateDecision", "GateResult", "GenerationRequestGate", "normalize_prompt"]

