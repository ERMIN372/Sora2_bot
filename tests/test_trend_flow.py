from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from tests.aiogram_stub import ensure_aiogram_stub

ensure_aiogram_stub()

from config import (  # noqa: E402  # isort:skip
    Config,
    DEFAULT_CREDIT_PRICE_RUB,
    DEFAULT_FIX_FEE_RUB,
    DEFAULT_MARKUP_PCT,
    PricingConfig,
)
from handlers import (  # noqa: E402  # isort:skip
    trend_answers_handler,
    trend_menu,
    trend_model_callback_handler,
    trend_select_callback_handler,
)
from trend_presets import TREND_PRESETS, get_trend_by_id  # noqa: E402  # isort:skip


@pytest.fixture()
def config() -> Config:
    pricing = PricingConfig(
        credit_price_rub=DEFAULT_CREDIT_PRICE_RUB,
        markup_pct=DEFAULT_MARKUP_PCT,
        fix_fee_rub=DEFAULT_FIX_FEE_RUB,
        provider_costs={},
    )
    return Config(bot_token="test", gemini_api_key="key", pricing=pricing)


class StubFSMContext:
    def __init__(self) -> None:
        self.data: Dict[str, Any] = {}
        self.finished: bool = False

    async def finish(self) -> None:
        self.finished = True
        self.data.clear()

    async def update_data(self, **kwargs: Any) -> None:
        self.data.update(kwargs)

    async def get_data(self) -> Dict[str, Any]:
        return dict(self.data)


class StubDB:
    def __init__(self, *, credits: int) -> None:
        self.credits = credits
        self.users: Dict[int, Dict[str, Any]] = {}
        self.deducted: List[tuple[int, int]] = []
        self.added: List[tuple[int, int]] = []

    async def ensure_user(
        self,
        user_id: int,
        username: Optional[str],
        *,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> None:
        self.users[user_id] = {
            "username": username,
            "first_name": first_name,
            "last_name": last_name,
        }

    async def get_user_credits(self, user_id: int) -> int:
        return self.credits

    async def deduct_credit(self, user_id: int, amount: int) -> bool:
        if self.credits < amount:
            return False
        self.credits -= amount
        self.deducted.append((user_id, amount))
        return True

    async def add_credits(self, user_id: int, amount: int) -> None:
        self.credits += amount
        self.added.append((user_id, amount))


class StubBot:
    def __init__(self) -> None:
        self.sent_messages: List[Dict[str, Any]] = []

    async def send_message(
        self,
        chat_id: int,
        text: str,
        reply_markup: Any = None,
        parse_mode: Optional[str] = None,
    ) -> Any:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "reply_markup": reply_markup,
            "parse_mode": parse_mode,
        }
        self.sent_messages.append(payload)
        return SimpleNamespace(message_id=len(self.sent_messages), chat=SimpleNamespace(id=chat_id))


class StubMessage:
    def __init__(self, text: str, bot: StubBot, *, user_id: int = 123) -> None:
        self.text = text
        self.bot = bot
        self.chat = SimpleNamespace(id=user_id)
        self.from_user = SimpleNamespace(id=user_id, username="tester", first_name="Test", last_name=None)
        self.replies: List[Dict[str, Any]] = []

    async def answer(
        self,
        text: str,
        reply_markup: Any = None,
        parse_mode: Optional[str] = None,
    ) -> Any:
        payload = {
            "text": text,
            "reply_markup": reply_markup,
            "parse_mode": parse_mode,
            "type": "answer",
        }
        self.replies.append(payload)
        return SimpleNamespace(message_id=len(self.replies))

    async def edit_text(self, text: str, reply_markup: Any = None) -> None:
        self.replies.append({"text": text, "reply_markup": reply_markup, "type": "edit"})

    async def edit_reply_markup(self) -> None:
        self.replies.append({"type": "edit_markup"})


class StubCallback:
    def __init__(self, data: str, message: StubMessage) -> None:
        self.data = data
        self.message = message
        self.from_user = message.from_user
        self.answered = False

    async def answer(self, *args: Any, **kwargs: Any) -> None:
        self.answered = True


class StubJobQueue:
    def __init__(self) -> None:
        self.submissions: List[Dict[str, Any]] = []

    async def submit(self, **kwargs: Any) -> Any:
        self.submissions.append(kwargs)
        return SimpleNamespace(extra=kwargs.get("extra", {}))


class StubChatClient:
    def __init__(self) -> None:
        self.prompts: List[Dict[str, str]] = []

    async def generate_reply(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.prompts.append({"prompt": prompt, "system": system_prompt or ""})
        return "🔥 Готово! Переходи по ссылке."


def test_get_trend_by_id() -> None:
    first = TREND_PRESETS[0]
    assert get_trend_by_id(first.id) is first
    assert get_trend_by_id("missing") is None


def test_trend_flow_success(config: Config) -> None:
    async def scenario() -> None:
        bot = StubBot()
        state = StubFSMContext()
        db = StubDB(credits=200)
        job_queue = StubJobQueue()
        chat_client = StubChatClient()
        message = StubMessage("📈 Тренды", bot)

        await trend_menu(message, state, db, config)
        assert any("📈" in entry["text"] for entry in message.replies if entry["type"] == "answer")
        state.finished = False

        select_callback = StubCallback(f"trend_select:{TREND_PRESETS[0].id}", message)
        await trend_select_callback_handler(select_callback, state, config)
        assert state.data.get("trend_id") == TREND_PRESETS[0].id

        model_callback = StubCallback("trend_model:sora", message)
        await trend_model_callback_handler(model_callback, state, config)
        assert state.data.get("model_choice") == "sora"

        answers = ["курс по дизайну", "основатели", "3 ошибки", "жми на ссылку"]
        for answer in answers:
            answer_message = StubMessage(answer, bot)
            await trend_answers_handler(answer_message, state, db, job_queue, config, chat_client)

        assert db.deducted == [(123, 99)]
        assert job_queue.submissions, "job queue should receive submission"
        submission = job_queue.submissions[0]
        assert "курс по дизайну" in submission["prompt"]
        assert submission["provider"] == "sora"
        assert submission["extra"]["trend"]["caption"]
        assert chat_client.prompts, "caption prompt should be requested"
        assert any("основатели" in entry["prompt"] for entry in chat_client.prompts)
        assert any(
            "🚀" in reply["text"] for reply in answer_message.replies if reply["type"] == "answer"
        )
        assert state.finished is True

    asyncio.run(scenario())


def test_trend_flow_not_enough_credits(config: Config) -> None:
    async def scenario() -> None:
        bot = StubBot()
        state = StubFSMContext()
        db = StubDB(credits=10)
        job_queue = StubJobQueue()
        chat_client = StubChatClient()
        message = StubMessage("📈 Тренды", bot)

        await trend_menu(message, state, db, config)
        state.finished = False
        select_callback = StubCallback(f"trend_select:{TREND_PRESETS[0].id}", message)
        await trend_select_callback_handler(select_callback, state, config)
        model_callback = StubCallback("trend_model:veo", message)
        await trend_model_callback_handler(model_callback, state, config)

        answers = ["курс", "маркетологи", "хук", "действие"]
        for answer in answers:
            answer_message = StubMessage(answer, bot)
            await trend_answers_handler(answer_message, state, db, job_queue, config, chat_client)

        assert not job_queue.submissions
        assert state.finished is True
        assert bot.sent_messages, "payment showcase should be triggered"
        assert chat_client.prompts == []

    asyncio.run(scenario())
