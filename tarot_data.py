"""Tarot cards data helpers."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping

__all__ = ["CARDS", "draw_cards", "get_image_path"]

_BASE_DIR = Path(__file__).resolve().parent
_DATA_PATH = _BASE_DIR / "data" / "tarot-images.json"
_CARDS_DIR = _BASE_DIR / "data" / "tarot_cards"


def _load_cards() -> list[dict]:
    with _DATA_PATH.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    cards = payload.get("cards", [])
    if not isinstance(cards, list):  # pragma: no cover - defensive programming
        raise ValueError("Unexpected tarot data format: 'cards' must be a list")
    return cards


CARDS: list[dict] = _load_cards()


def draw_cards(number: int, *, rng: random.Random | None = None) -> list[dict]:
    """Return a random selection of tarot cards."""

    if number <= 0:
        raise ValueError("Number of cards to draw must be greater than zero")

    if not CARDS:
        return []

    rng = rng or random
    deck_size = min(number, len(CARDS))
    return rng.sample(CARDS, deck_size)


def get_image_path(card: Mapping[str, object]) -> Path:
    """Resolve the local path to a card image."""

    img_name = card.get("img") if isinstance(card, Mapping) else None
    if not img_name:
        raise KeyError("Card does not define an 'img' attribute")
    return _CARDS_DIR / str(img_name)
