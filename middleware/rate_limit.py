"""Aiogram middleware to enforce per-command rate limits using Redis."""
from __future__ import annotations

import logging
from typing import Mapping, Tuple

from aiogram import types
from aiogram.dispatcher.handler import CancelHandler
from aiogram.dispatcher.middlewares import BaseMiddleware

from config import RateLimitRule

log = logging.getLogger(__name__)


class RateLimitMiddleware(BaseMiddleware):
    """Limit command usage per user with Redis counters."""

    def __init__(
        self,
        redis_client,
        limits: Mapping[str, RateLimitRule],
        *,
        key_prefix: str = "rate_limit",
    ) -> None:
        super().__init__()
        self._redis = redis_client
        self._limits = {name.lower(): self._normalise_rule(rule) for name, rule in limits.items()}
        self._key_prefix = key_prefix

    async def on_pre_process_message(self, message: types.Message, data: dict) -> None:  # type: ignore[override]
        command = self._extract_command(message)
        if not command:
            return
        rule = self._limits.get(command)
        if not rule:
            return
        user = message.from_user
        if user is None:
            log.debug("Skipping rate limit for message without user command=%s", command)
            return
        allowed = await self._is_allowed(user.id, command, rule)
        if allowed:
            return
        await message.answer("Превышен лимит запросов, попробуйте позже.")
        raise CancelHandler()

    def _extract_command(self, message: types.Message) -> str:
        if not getattr(message, "text", ""):
            return ""
        raw_command = None
        if hasattr(message, "get_command"):
            raw_command = message.get_command(pure=True)
        if not raw_command and message.text.startswith("/"):
            raw_command = message.text.split()[0].lstrip("/").split("@")[0]
        return (raw_command or "").lower()

    async def _is_allowed(
        self, user_id: int, command: str, rule: Tuple[int, int]
    ) -> bool:
        limit, period = rule
        key = f"{self._key_prefix}:{user_id}:{command}"
        current = await self._redis.incr(key)
        if current == 1:
            await self._redis.expire(key, period)
        if current > limit:
            log.info(
                "Rate limit exceeded user=%s command=%s count=%s limit=%s period=%s",
                user_id,
                command,
                current,
                limit,
                period,
            )
            return False
        return True

    @staticmethod
    def _normalise_rule(rule: RateLimitRule) -> Tuple[int, int]:
        limit = max(1, int(rule.limit))
        period = max(1, int(rule.period))
        return (limit, period)


__all__ = ["RateLimitMiddleware"]
