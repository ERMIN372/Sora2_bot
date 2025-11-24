"""Admin panel handlers for managing news drafts."""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

from aiogram import Dispatcher, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from news_agent import (
    ADMIN_IMAGE_WAIT,
    create_fake_drafts,
    get_draft,
    set_draft_image,
    set_draft_status,
)

log = logging.getLogger(__name__)


def _parse_chat_id(value: str, *, env_name: str) -> Optional[int]:
    try:
        return int(value)
    except ValueError:
        log.warning("%s should be an integer, got %s", env_name, value)
    return None


def get_news_environment() -> Tuple[Optional[int], Optional[int]]:
    """Return configured channel and admin chat IDs from the environment."""
    channel_raw = os.getenv("NEWS_CHANNEL_ID")
    admin_raw = os.getenv("NEWS_ADMIN_CHAT_ID")

    channel_id = _parse_chat_id(channel_raw, env_name="NEWS_CHANNEL_ID") if channel_raw else None
    admin_chat_id = _parse_chat_id(admin_raw, env_name="NEWS_ADMIN_CHAT_ID") if admin_raw else None

    if channel_raw is None:
        log.warning("NEWS_CHANNEL_ID is not set; approved posts will not be sent")
    if admin_raw is None:
        log.warning("NEWS_ADMIN_CHAT_ID is not set; news admin panel will be disabled")

    return channel_id, admin_chat_id


NEWS_CHANNEL_ID, NEWS_ADMIN_CHAT_ID = get_news_environment()


def build_news_admin_keyboard(draft_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup().add(
        InlineKeyboardButton("✅ Одобрить", callback_data=f"news:approve:{draft_id}"),
        InlineKeyboardButton("🗑 Отклонить", callback_data=f"news:reject:{draft_id}"),
        InlineKeyboardButton("📷 Картинка", callback_data=f"news:image:{draft_id}"),
    )


async def news_demo_handler(message: types.Message) -> None:
    if NEWS_ADMIN_CHAT_ID is None or message.chat.id != NEWS_ADMIN_CHAT_ID:
        await message.answer("Недоступно")
        return

    drafts = create_fake_drafts()
    for draft in drafts:
        await message.bot.send_message(
            chat_id=NEWS_ADMIN_CHAT_ID,
            text=draft["text"],
            reply_markup=build_news_admin_keyboard(draft["id"]),
        )


def _extract_draft_id(callback_data: Optional[str]) -> Optional[str]:
    if not callback_data:
        return None
    parts = callback_data.split(":", 2)
    if len(parts) != 3:
        return None
    return parts[-1]


async def approve_draft_callback(callback_query: types.CallbackQuery) -> None:
    chat_id = callback_query.message.chat.id if callback_query.message else None
    if NEWS_ADMIN_CHAT_ID and chat_id != NEWS_ADMIN_CHAT_ID:
        await callback_query.answer("Недоступно")
        return

    draft_id = _extract_draft_id(callback_query.data)
    if not draft_id:
        await callback_query.answer("Некорректный запрос")
        return

    draft = get_draft(draft_id)
    if draft is None:
        await callback_query.answer("Черновик не найден", show_alert=True)
        return

    if draft.get("status") == "approved":
        await callback_query.answer("Уже опубликовано", show_alert=True)
        if callback_query.message and callback_query.message.reply_markup:
            try:
                await callback_query.message.edit_reply_markup(reply_markup=None)
            except Exception:
                log.debug(
                    "Failed to remove inline keyboard for already approved draft %s",
                    draft_id,
                    exc_info=True,
                )
        return

    if NEWS_CHANNEL_ID is None:
        log.warning("Cannot publish draft %s: NEWS_CHANNEL_ID is not configured", draft_id)
        await callback_query.answer("Канал не настроен", show_alert=True)
        return

    if draft.get("image_file_id"):
        await callback_query.bot.send_photo(
            chat_id=NEWS_CHANNEL_ID,
            photo=draft["image_file_id"],
            caption=draft["text"],
        )
    else:
        await callback_query.bot.send_message(
            chat_id=NEWS_CHANNEL_ID,
            text=draft["text"],
        )

    set_draft_status(draft_id, "approved")
    await callback_query.answer("Пост опубликован")

    if callback_query.message:
        try:
            await callback_query.message.edit_text(
                f"{callback_query.message.text}\n\n✅ Опубликовано",
                reply_markup=None,
            )
        except Exception:
            log.debug("Failed to edit admin message for draft %s", draft_id, exc_info=True)


async def reject_draft_callback(callback_query: types.CallbackQuery) -> None:
    chat_id = callback_query.message.chat.id if callback_query.message else None
    if NEWS_ADMIN_CHAT_ID and chat_id != NEWS_ADMIN_CHAT_ID:
        await callback_query.answer("Недоступно")
        return

    draft_id = _extract_draft_id(callback_query.data)
    if not draft_id:
        await callback_query.answer("Некорректный запрос")
        return

    set_draft_status(draft_id, "rejected")
    await callback_query.answer("Черновик отклонен")

    if callback_query.message:
        try:
            await callback_query.message.edit_text(
                f"{callback_query.message.text}\n\n🗑 Отклонено",
                reply_markup=callback_query.message.reply_markup,
            )
        except Exception:
            log.debug("Failed to edit admin message for draft %s", draft_id, exc_info=True)


async def image_request_callback(callback_query: types.CallbackQuery) -> None:
    chat_id = callback_query.message.chat.id if callback_query.message else None
    if NEWS_ADMIN_CHAT_ID and chat_id != NEWS_ADMIN_CHAT_ID:
        await callback_query.answer("Недоступно")
        return

    draft_id = _extract_draft_id(callback_query.data)
    if not draft_id:
        await callback_query.answer("Некорректный запрос")
        return

    user_id = callback_query.from_user.id if callback_query.from_user else None
    if user_id is None:
        await callback_query.answer("Неизвестный пользователь")
        return

    ADMIN_IMAGE_WAIT[user_id] = draft_id
    await callback_query.answer("Пришли картинку одним сообщением для этого поста")


async def admin_photo_handler(message: types.Message) -> None:
    user = message.from_user
    if user is None:
        return

    draft_id = ADMIN_IMAGE_WAIT.pop(user.id, None)
    if draft_id is None:
        return

    draft = get_draft(draft_id)
    if not draft:
        await message.answer("Черновик не найден")
        return

    photo = message.photo[-1]
    set_draft_image(draft_id, photo.file_id)
    await message.answer("Картинка привязана к черновику")

    await message.bot.send_photo(
        chat_id=message.chat.id,
        photo=photo.file_id,
        caption=draft["text"],
        reply_markup=build_news_admin_keyboard(draft_id),
    )


def register_news_admin_handlers(dp: Dispatcher) -> None:
    dp.register_message_handler(
        news_demo_handler,
        commands=["news_demo"],
        state="*",
    )
    dp.register_callback_query_handler(
        approve_draft_callback,
        lambda callback_query: callback_query.data
        and callback_query.data.startswith("news:approve:"),
    )
    dp.register_callback_query_handler(
        reject_draft_callback,
        lambda callback_query: callback_query.data
        and callback_query.data.startswith("news:reject:"),
    )
    dp.register_callback_query_handler(
        image_request_callback,
        lambda callback_query: callback_query.data
        and callback_query.data.startswith("news:image:"),
    )
    dp.register_message_handler(
        admin_photo_handler,
        lambda message: (
            NEWS_ADMIN_CHAT_ID is not None
            and message.chat.id == NEWS_ADMIN_CHAT_ID
            and message.from_user
            and message.from_user.id in ADMIN_IMAGE_WAIT
        ),
        content_types=["photo"],
        state="*",
    )
