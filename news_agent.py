"""In-memory storage for news drafts used by the admin panel."""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional


NEWS_DRAFTS: Dict[str, dict] = {}
"""Mapping draft_id -> draft payload."""

ADMIN_IMAGE_WAIT: Dict[int, str] = {}
"""Mapping user_id -> draft_id while waiting for an admin to send a photo."""


def create_fake_drafts() -> List[dict]:
    """Create several fake drafts for demo purposes."""
    fake_texts = [
        "Новость дня: Добавили новый генератор видео!",
        "Обновление бота: теперь поддерживаем быструю отправку в архив.",
        "Напоминание: подписывайтесь на канал, чтобы не пропустить новинки.",
    ]
    drafts: List[dict] = []
    for text in fake_texts:
        draft_id = str(uuid.uuid4())
        draft = {
            "id": draft_id,
            "text": text,
            "status": "pending",
            "image_file_id": None,
        }
        NEWS_DRAFTS[draft_id] = draft
        drafts.append(draft)
    return drafts


def get_draft(draft_id: str) -> Optional[dict]:
    """Return draft by id if it exists."""
    return NEWS_DRAFTS.get(draft_id)


def set_draft_image(draft_id: str, file_id: str) -> None:
    """Attach image file_id to a draft if present."""
    draft = NEWS_DRAFTS.get(draft_id)
    if draft is None:
        return
    draft["image_file_id"] = file_id


def set_draft_status(draft_id: str, status: str) -> None:
    """Update draft status if the draft exists."""
    draft = NEWS_DRAFTS.get(draft_id)
    if draft is None:
        return
    draft["status"] = status
