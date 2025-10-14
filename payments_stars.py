"""Telegram Stars payment processing helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from aiogram.types import Message

from config import Config
from db import Database
from utils import check_subscription


@dataclass
class PaymentResult:
    user_id: int
    amount: int
    credits_added: int


class TelegramStarPaymentProcessor:
    """Process successful Telegram Star payments and award credits."""

    def __init__(self, *, db: Database, config: Config) -> None:
        self._db = db
        self._config = config

    async def handle_successful_payment(self, message: Message) -> Optional[PaymentResult]:
        """Persist a successful payment and top up the user's credits.

        Returns the :class:`PaymentResult` if the message contains a successful
        payment, otherwise ``None``.
        """

        if not message.successful_payment:
            return None

        successful_payment = message.successful_payment
        user = message.from_user
        if user is None:  # pragma: no cover - defensive
            return None

        await self._db.ensure_user(user.id, user.username)
        credits_added = self._config.credits_per_payment
        await self._db.add_credits(user.id, credits_added)

        await self._db.record_payment(
            user_id=user.id,
            provider_payment_charge_id=successful_payment.provider_payment_charge_id,
            telegram_payment_charge_id=successful_payment.telegram_payment_charge_id,
            amount=successful_payment.total_amount,
            credits_added=credits_added,
        )

        if not await self._db.is_bonus_granted(user.id):
            if await check_subscription(message.bot, user.id, self._config):
                await self._db.add_credits(user.id, 2)
                await self._db.mark_bonus_granted(user.id)

        return PaymentResult(
            user_id=user.id,
            amount=successful_payment.total_amount,
            credits_added=credits_added,
        )


__all__ = ["TelegramStarPaymentProcessor", "PaymentResult"]
