"""FSM state groups for all conversation flows."""

from __future__ import annotations

from aiogram.dispatcher.filters.state import State, StatesGroup


class GenerationStates(StatesGroup):
    """Conversation states for collecting generation inputs."""

    video_model = State()
    video_mode = State()
    image_model = State()
    image_mode = State()
    text_prompt = State()
    photo_prompt = State()
    kling_motion_video = State()


class AutoStyleStates(StatesGroup):
    """States for navigating auto-style flows (trends and Nano Banana)."""

    trends_menu = State()
    choosing_styles = State()
    choosing_prompt_mode = State()


class ChatGPTState(StatesGroup):
    """Conversation states for the GPT-4.1 assistant."""

    awaiting_input = State()


class TarotStates(StatesGroup):
    """Conversation states for tarot spreads."""

    choosing_type = State()
    waiting_for_question = State()


class TrendStates(StatesGroup):
    """Conversation states for preset trend videos."""

    choosing_trend = State()
    choosing_model = State()
    answering_params = State()


class AdminStates(StatesGroup):
    """FSM states for the administrator panel."""

    menu = State()
    broadcast_message = State()
    broadcast_confirm = State()
    direct_target = State()
    direct_message = State()
    direct_confirm = State()
