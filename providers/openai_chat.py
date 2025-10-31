"""Async client wrapper for OpenAI ChatGPT interactions."""
from __future__ import annotations

import logging
from typing import List, Mapping, Optional, Sequence

try:  # pragma: no cover - optional dependency during tests
    from openai import AsyncOpenAI
except ModuleNotFoundError:  # pragma: no cover - optional dependency during tests
    AsyncOpenAI = None  # type: ignore[assignment]

from config import Config

log = logging.getLogger(__name__)


class OpenAIChatClient:
    """Thin wrapper around the OpenAI Chat Completions API."""

    def __init__(self, *, config: Config, model: str = "gpt-4.1-turbo") -> None:
        if AsyncOpenAI is None:
            raise RuntimeError("openai package is required for ChatGPT integration")
        api_key = (config.openai_key or "").strip()
        if not api_key:
            raise RuntimeError("OpenAI API key is required for ChatGPT integration")
        base_url = (config.openai_api_base or "https://api.openai.com/v1").rstrip("/")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def generate_reply(
        self,
        prompt: str,
        *,
        conversation: Optional[Sequence[Mapping[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a reply using the configured chat model."""

        messages: List[Mapping[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        if conversation:
            for message in conversation:
                role = str(message.get("role", ""))
                content = str(message.get("content", ""))
                if not role or not content:
                    continue
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
            )
        except Exception as exc:  # pragma: no cover - network failure
            log.exception("OpenAI chat completion failed")
            raise RuntimeError("Failed to call OpenAI chat completion") from exc
        if not response.choices:
            raise RuntimeError("OpenAI chat completion returned no choices")
        choice = response.choices[0]
        if not getattr(choice, "message", None):
            raise RuntimeError("OpenAI chat completion returned an empty message")
        content = choice.message.content or ""
        if isinstance(content, list):
            # The SDK may return structured content; concatenate text segments.
            text_parts = []
            for item in content:
                segment = getattr(item, "text", None)
                if segment:
                    text_parts.append(str(segment))
            content = "\n".join(text_parts)
        if not isinstance(content, str):
            content = str(content or "")
        return content.strip()


__all__ = ["OpenAIChatClient"]
