"""Middleware that restricts /admin_ commands to configured admins."""
from __future__ import annotations

from aiogram import types
from aiogram.dispatcher.handler import CancelHandler
from aiogram.dispatcher.middlewares import BaseMiddleware


class AdminCheckMiddleware(BaseMiddleware):
    """Prevent non-admin users from executing admin-only commands."""

    def __init__(self, admin_ids: tuple[int, ...]):
        super().__init__()
        self._admin_ids = set(admin_ids)

    async def on_pre_process_message(self, message: types.Message, data: dict) -> None:  # type: ignore[override]
        if not getattr(message, "text", ""):
            return
        if not message.text.startswith("/admin_"):
            return
        user_id = getattr(getattr(message, "from_user", None), "id", None)
        if user_id in self._admin_ids:
            return
        await message.answer("Эта команда доступна только администраторам")
        raise CancelHandler()


__all__ = ["AdminCheckMiddleware"]
