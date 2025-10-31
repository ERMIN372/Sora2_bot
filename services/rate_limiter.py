"""Asynchronous rate limiting helpers for provider clients."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Deque


class AsyncRateLimiter:
    """Simple sliding-window rate limiter for asyncio workflows."""

    def __init__(self, max_calls: int, period: float) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if period <= 0:
            raise ValueError("period must be positive")
        self._max_calls = max_calls
        self._period = float(period)
        self._timestamps: Deque[float] = deque()
        self._lock = asyncio.Lock()
        self._penalty_until: float = 0.0

    async def acquire(self) -> None:
        """Wait until the next request is allowed."""

        while True:
            async with self._lock:
                now = time.monotonic()
                if self._penalty_until > now:
                    delay = self._penalty_until - now
                else:
                    delay = 0.0
                    while self._timestamps and now - self._timestamps[0] >= self._period:
                        self._timestamps.popleft()
                    if len(self._timestamps) < self._max_calls:
                        self._timestamps.append(now)
                        return
                    oldest = self._timestamps[0]
                    delay = max(0.0, self._period - (now - oldest))
            await asyncio.sleep(delay if delay > 0 else 0.01)

    async def penalize(self, seconds: float) -> None:
        """Introduce an additional cooldown period before the next acquire."""

        if seconds <= 0:
            return
        async with self._lock:
            target = time.monotonic() + float(seconds)
            if target > self._penalty_until:
                self._penalty_until = target


__all__ = ["AsyncRateLimiter"]
