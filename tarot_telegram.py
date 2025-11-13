"""Telegram helpers for tarot card deliveries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from aiogram.types import InputFile, Message

from tarot_data import draw_cards, get_image_path

__all__ = [
    "FILE_ID_CACHE",
    "format_card_caption",
    "send_card",
    "send_simple_spread",
]


_BASE_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _BASE_DIR / "data" / "tarot_cache.json"


def _load_cache() -> dict[str, str]:
    try:
        with _CACHE_PATH.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:  # pragma: no cover - corrupted cache
        return {}
    if not isinstance(payload, dict):  # pragma: no cover - defensive
        return {}
    return {str(key): str(value) for key, value in payload.items() if isinstance(value, str)}


FILE_ID_CACHE: dict[str, str] = _load_cache()


def _save_cache() -> None:
    if not FILE_ID_CACHE:
        if _CACHE_PATH.exists():
            _CACHE_PATH.unlink()
        return

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("w", encoding="utf-8") as fp:
        json.dump(FILE_ID_CACHE, fp, ensure_ascii=False, indent=2, sort_keys=True)


def format_card_caption(card: Mapping[str, object], *, prefix: str | None = None) -> str:
    parts: list[str] = []

    if prefix:
        parts.append(prefix)

    name = str(card.get("name", "Unknown card"))
    arcana = card.get("arcana")
    if arcana:
        parts.append(f"{name} — {arcana}")
    else:
        parts.append(name)

    keywords = [str(item) for item in card.get("keywords", []) if item]
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}")

    fortunes = [str(item) for item in card.get("fortune_telling", []) if item]
    if fortunes:
        parts.append(f"Fortune: {fortunes[0]}")

    return "\n".join(parts)


async def send_card(
    message: Message,
    card: Mapping[str, object],
    *,
    disable_notification: bool = False,
    spread_prefix: str | None = None,
) -> Message:
    image_path = get_image_path(card)
    cache_key = str(image_path)
    caption = format_card_caption(card, prefix=spread_prefix)

    file_id = FILE_ID_CACHE.get(cache_key)
    if file_id:
        sent_message = await message.answer_photo(
            file_id,
            caption=caption,
            disable_notification=disable_notification,
        )
        return sent_message

    telegram_file = InputFile(image_path)
    sent_message = await message.answer_photo(
        telegram_file,
        caption=caption,
        disable_notification=disable_notification,
    )

    photo_sizes = getattr(sent_message, "photo", None)
    if photo_sizes:
        try:
            new_file_id = photo_sizes[-1].file_id
        except (IndexError, AttributeError):  # pragma: no cover - aiogram stubs variations
            new_file_id = None
        if new_file_id:
            FILE_ID_CACHE[cache_key] = new_file_id
            _save_cache()

    return sent_message


async def send_simple_spread(
    message: Message,
    *,
    number_of_cards: int = 3,
    disable_notification: bool = False,
) -> list[Message]:
    cards = draw_cards(number_of_cards)
    results: list[Message] = []

    for idx, card in enumerate(cards, start=1):
        prefix = f"Card {idx}"
        sent_message = await send_card(
            message,
            card,
            disable_notification=disable_notification,
            spread_prefix=prefix,
        )
        results.append(sent_message)

    return results
