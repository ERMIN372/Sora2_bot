"""Helpers for authenticating with Google APIs using service accounts."""

from __future__ import annotations

import asyncio
from typing import Iterable, Mapping

from google.auth.transport.requests import Request
from google.oauth2 import service_account


class ServiceAccountTokenProvider:
    """Fetch and cache OAuth tokens for a Google service account."""

    def __init__(self, info: Mapping[str, object], scopes: Iterable[str]) -> None:
        if not info:
            raise ValueError("Service account info must be provided")
        self._credentials = service_account.Credentials.from_service_account_info(
            info, scopes=list(scopes)
        )
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        """Return a valid OAuth access token for the configured scopes."""

        async with self._lock:
            if self._credentials.valid and self._credentials.token:
                return str(self._credentials.token)
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, self._refresh_credentials)
            except Exception as exc:  # pragma: no cover - network/auth error
                raise RuntimeError(f"Failed to refresh service account token: {exc}") from exc
            if not self._credentials.valid or not self._credentials.token:
                raise RuntimeError("Service account credentials did not return an access token")
            return str(self._credentials.token)

    def _refresh_credentials(self) -> None:
        request = Request()
        self._credentials.refresh(request)


__all__ = ["ServiceAccountTokenProvider"]

