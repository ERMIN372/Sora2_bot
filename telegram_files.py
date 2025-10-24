"""Compatibility helpers for Telegram file uploads."""
from __future__ import annotations

import io

from aiogram.types import InputFile

try:  # pragma: no cover - depends on aiogram version
    from aiogram.types import BufferedInputFile as _BufferedInputFile
except ImportError:  # pragma: no cover - aiogram<3 provides only InputFile

    class BufferedInputFile(InputFile):
        """Fallback BufferedInputFile implementation for aiogram 2.x."""

        def __init__(self, data: bytes | bytearray | memoryview, filename: str, *, mime_type: str | None = None) -> None:
            payload = bytes(data)
            buffer = io.BytesIO(payload)
            buffer.seek(0)
            super().__init__(buffer, filename=filename)
            self.mime_type = mime_type

        @property
        def data(self) -> bytes:
            buffer: io.BytesIO = self.file  # type: ignore[assignment]
            position = buffer.tell()
            buffer.seek(0)
            payload = buffer.read()
            buffer.seek(position)
            return payload

else:  # pragma: no cover - aiogram>=3 ships BufferedInputFile

    class BufferedInputFile(_BufferedInputFile):
        """Proxy subclass to provide a uniform import location."""

        def __init__(self, data: bytes | bytearray | memoryview, filename: str, *, mime_type: str | None = None) -> None:
            super().__init__(data, filename=filename)
            if mime_type is not None:
                setattr(self, "mime_type", mime_type)


__all__ = ["BufferedInputFile", "InputFile"]
