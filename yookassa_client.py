"""Helpers to interact with the YooKassa payments API."""
from __future__ import annotations

import inspect
import uuid
from decimal import Decimal
from typing import Any, Dict

from yookassa import Configuration, Payment

from config import Config, CreditPackage

cfg: Config | None = None


def init(config: Config) -> None:
    """Initialise the YooKassa SDK with credentials from *config*."""

    if not (config.yookassa_shop_id and config.yookassa_secret_key):
        raise RuntimeError("YooKassa credentials are not configured")
    if not config.public_base_url:
        raise RuntimeError("PUBLIC_BASE_URL is required for YooKassa payments")
    Configuration.account_id = config.yookassa_shop_id
    Configuration.secret_key = config.yookassa_secret_key
    Configuration.configure(config.yookassa_shop_id, config.yookassa_secret_key)
    global cfg
    cfg = config


def _require_config() -> Config:
    if cfg is None:  # pragma: no cover - startup guard
        raise RuntimeError("YooKassa client has not been initialised")
    return cfg


def build_return_url(order_id: str) -> str:
    config = _require_config()
    return f"{config.public_base_url}{config.yookassa_return_path}?order_id={order_id}"


def _build_metadata(package: CreditPackage, user_id: int, idemp: str) -> Dict[str, Any]:
    return {
        "kind": "credits",
        "package_id": package.package_id,
        "purchased_credits": package.credits_int,
        "expected_amount_cp": package.price_kopeks,
        "user_id": str(user_id),
        "idempotency_key": idemp,
        "idemp": idemp,
        "order_id": idemp,
    }


def create_payment(package: CreditPackage, user_id: int, description: str) -> Dict[str, Any]:
    """Create a YooKassa payment and return identifiers and confirmation URL."""

    config = _require_config()
    idemp = str(uuid.uuid4())
    metadata = _build_metadata(package, user_id, idemp)
    amount_value = (Decimal(package.price_kopeks) / Decimal(100)).quantize(Decimal("0.01"))
    body: Dict[str, Any] = {
        "amount": {"value": str(amount_value), "currency": "RUB"},
        "capture": True,
        "confirmation": {"type": "redirect", "return_url": build_return_url(idemp)},
        "description": description[:128],
        "metadata": metadata,
    }
    if config.yookassa_send_receipts:
        body["receipt"] = {
            "customer": {"email": "user@example.com"},
            "items": [
                {
                    "description": "Video generation credits",
                    "quantity": "1.0",
                    "amount": {"value": str(amount_value), "currency": "RUB"},
                    "vat_code": 1,
                }
            ],
        }
    payment = _create_payment(body, idemp)
    return {
        "payment_id": payment.id,
        "confirmation_url": payment.confirmation.confirmation_url,
        "status": payment.status,
        "order_id": idemp,
        "metadata": metadata,
        "idempotency_key": idemp,
        "amount_cp": package.price_kopeks,
    }


def get_payment(payment_id: str):
    """Fetch a payment from YooKassa by its *payment_id*."""

    return Payment.find_one(payment_id)


def _create_payment(body: Dict[str, Any], idemp: str):
    """Compatibility wrapper around :func:`Payment.create`."""

    parameters = inspect.signature(Payment.create).parameters
    if "idempotence_key" in parameters:
        return Payment.create(body, idempotence_key=idemp)
    if "idempotency_key" in parameters:
        return Payment.create(body, idempotency_key=idemp)
    return Payment.create(body, idemp)


__all__ = ["init", "create_payment", "get_payment", "build_return_url"]
