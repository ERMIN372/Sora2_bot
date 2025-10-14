"""Unified FastAPI application exposing health, payments and Telegram webhooks."""
from __future__ import annotations

import asyncio
import json
import logging
from decimal import Decimal, InvalidOperation
from ipaddress import ip_address, ip_network
from typing import Dict, Optional

from aiogram import Bot, Dispatcher, types
from fastapi import APIRouter, FastAPI, Header, Request, Response
from fastapi.responses import HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from config import CFG, Config
from db import Database
from handlers import resend_pending_order
from i18n import format_credits, i18n
import yookassa_client

log = logging.getLogger(__name__)

YOOKASSA_IP_RANGES = [
    ip_network("185.71.76.0/27"),
    ip_network("185.71.77.0/27"),
    ip_network("77.75.153.0/25"),
    ip_network("77.75.154.128/25"),
    ip_network("2a02:5180::/32"),
]

MAX_REQUEST_SIZE = 2 * 1024 * 1024  # 2 MiB


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests that exceed a predefined size."""

    def __init__(self, app: ASGIApp, *, max_body_size: int) -> None:  # type: ignore[override]
        super().__init__(app)
        self._max_body_size = max_body_size

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
            except ValueError:
                return Response(status_code=400)
            if length > self._max_body_size:
                log.warning("Request body too large: %s bytes", length)
                return Response(status_code=413)
        return await call_next(request)


class YooKassaProcessor:
    """Reconcile YooKassa payments with the local database."""

    def __init__(self, *, db: Database, config: Config, bot: Bot) -> None:
        self._db = db
        self._config = config
        self._bot = bot

    async def process_payment(self, payment, *, source: str) -> None:
        metadata = self._extract_metadata(payment)
        payment_id = getattr(payment, "id", None)
        status = getattr(payment, "status", "")
        amount_cp = self._amount_to_cop(getattr(payment, "amount", None))
        user_id = metadata.get("user_id")
        items = metadata.get("items", 0)
        if payment_id is None or user_id is None:
            log.warning(
                "Incomplete YooKassa payment payload from %s: id=%s user_id=%s",
                source,
                payment_id,
                user_id,
            )
            return
        try:
            user_id_int = int(user_id)
        except ValueError:
            log.warning("Invalid user_id %s in YooKassa metadata", user_id)
            return
        items_int = int(items) if items else 0
        metadata_str = json.dumps(metadata)

        existing = await self._db.get_payment_by_ext("yookassa", payment_id)
        if not existing:
            await self._db.create_payment_record(
                "yookassa",
                payment_id,
                user_id_int,
                amount_cp,
                items_int,
                status or "pending",
                metadata_str,
            )
        elif metadata_str and metadata_str != (existing.get("metadata") or "{}"):
            await self._db.update_payment_status_by_ext(
                "yookassa", payment_id, existing["status"], metadata=metadata_str
            )

        if status == "succeeded":
            if existing and existing.get("status") == "succeeded":
                log.info("Duplicate success webhook for payment %s", payment_id)
                return
            await self._db.ensure_user(user_id_int, None)
            if items_int:
                await self._db.add_credits(user_id_int, items_int)
            await self._db.update_payment_status_by_ext(
                "yookassa",
                payment_id,
                "succeeded",
                credits_added=items_int,
                metadata=metadata_str,
            )
            await self._notify_success(user_id_int, items_int)
        elif status in {"pending", "waiting_for_capture", "canceled"}:
            await self._db.update_payment_status_by_ext(
                "yookassa",
                payment_id,
                status,
                metadata=metadata_str,
            )
        else:
            await self._db.update_payment_status_by_ext(
                "yookassa",
                payment_id,
                status or "pending",
                metadata=metadata_str,
            )

    async def poll_pending_once(self) -> None:
        pending = await self._db.list_payments_by_status(
            provider="yookassa", statuses=("pending", "waiting_for_capture")
        )
        if not pending:
            return
        log.debug("Polling %s pending YooKassa payments", len(pending))
        for record in pending:
            payment_id = record.get("ext_id")
            if not payment_id:
                continue
            try:
                payment = await asyncio.to_thread(yookassa_client.get_payment, payment_id)
            except Exception:  # pragma: no cover - network interaction
                log.exception("Failed to refresh YooKassa payment %s", payment_id)
                continue
            await self.process_payment(payment, source="poller")

    async def _notify_success(self, user_id: int, items: int) -> None:
        try:
            if items > 0:
                await self._bot.send_message(
                    user_id,
                    i18n.t("payment.received", items=format_credits(items)),
                )
            await resend_pending_order(bot=self._bot, db=self._db, config=self._config, user_id=user_id)
        except Exception:  # pragma: no cover - Telegram API interaction
            log.exception("Failed to notify user %s about YooKassa success", user_id)

    def _extract_metadata(self, payment) -> dict:
        metadata = getattr(payment, "metadata", None) or {}
        if isinstance(metadata, dict):
            return metadata
        try:
            return dict(metadata)
        except Exception:  # pragma: no cover - defensive
            return {}

    def _amount_to_cop(self, amount_obj) -> int:
        if not amount_obj:
            return 0
        value = getattr(amount_obj, "value", None)
        try:
            quantised = Decimal(str(value)) * 100
        except (InvalidOperation, TypeError):
            return 0
        return int(quantised)

    async def get_payment_by_order_id(self, order_id: str) -> Optional[Dict[str, object]]:
        return await self._db.get_payment_by_order_id("yookassa", order_id)


def _is_ip_allowed(ip: Optional[str]) -> bool:
    if not ip:
        return False
    try:
        ip_obj = ip_address(ip)
    except ValueError:
        return False
    return any(ip_obj in network for network in YOOKASSA_IP_RANGES)


async def poll_pending_payments(processor: YooKassaProcessor, interval: int) -> None:
    """Background task to poll pending YooKassa payments."""

    while True:
        try:
            await processor.poll_pending_once()
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            raise
        except Exception:  # pragma: no cover - defensive
            log.exception("YooKassa pending poller error")
        await asyncio.sleep(interval)


def _health_router() -> APIRouter:
    router = APIRouter()

    @router.get("/healthz")
    async def healthz() -> Dict[str, object]:
        return {"ok": True, "mode": CFG.BOT_MODE}

    return router


def _yookassa_router(
    *, config: Config, processor: Optional[YooKassaProcessor]
) -> APIRouter:
    router = APIRouter()

    @router.post(config.yookassa_webhook_path)
    async def webhook(request: Request) -> Response:
        if processor is None:
            return Response(status_code=503)
        remote = request.client.host if request.client else "unknown"
        log.info("Received YooKassa webhook from %s", remote)
        if not config.yookassa_test_mode and not _is_ip_allowed(remote):
            log.warning("Rejected YooKassa webhook from unauthorized IP %s", remote)
            return Response(status_code=403)
        try:
            payload = await request.json()
        except Exception:
            log.exception("Failed to parse YooKassa webhook body")
            return Response(status_code=400)
        if payload.get("type") != "notification":
            log.warning("Unexpected YooKassa webhook type: %s", payload.get("type"))
            return Response(status_code=400)
        event = payload.get("event")
        obj = payload.get("object") or {}
        payment_id = obj.get("id")
        if not payment_id:
            log.warning("Webhook without payment id")
            return Response(status_code=400)
        if event == "payment.succeeded":
            log.info("Payment %s succeeded via webhook", payment_id)
        try:
            payment = await asyncio.to_thread(yookassa_client.get_payment, payment_id)
        except Exception:  # pragma: no cover - network interaction
            log.exception("Failed to re-fetch YooKassa payment %s", payment_id)
            return Response(status_code=200)
        await processor.process_payment(payment, source="webhook")
        return Response(status_code=200)

    @router.get(config.yookassa_return_path)
    async def return_page(order_id: Optional[str] = None) -> HTMLResponse:
        if processor is None:
            return HTMLResponse(i18n.t("web.return.unavailable"), status_code=503)
        if not order_id:
            return HTMLResponse(i18n.t("web.return.not_found"), status_code=400)
        record = await processor.get_payment_by_order_id(order_id)
        if not record:
            body = i18n.t("web.return.not_found")
        else:
            status = record.get("status", "pending")
            if status == "succeeded":
                body = i18n.t("web.return.success")
            elif status == "canceled":
                body = i18n.t("web.return.canceled")
            else:
                body = i18n.t("web.return.pending")
        return HTMLResponse(body)

    return router


def _telegram_router(dp: Dispatcher) -> APIRouter:
    router = APIRouter()

    # [TG_WEBHOOK_ROUTE]
    @router.post(CFG.WEBHOOK_PATH)
    async def tg_webhook(
        request: Request,
        x_telegram_bot_api_secret_token: str | None = Header(default=None),
    ) -> Response:
        # Проверка секрета
        if x_telegram_bot_api_secret_token != CFG.TELEGRAM_SECRET_TOKEN:
            return Response(status_code=403)

        try:
            data = await request.json()
        except Exception:
            return Response(status_code=400)

        update = types.Update(**data)
        # ВАЖНО: process_update, а не polling
        await dp.process_update(update)
        return Response(status_code=200)

    return router


# [FASTAPI_APP_FACTORY]
def create_app(
    *, config: Config, dp: Dispatcher, bot: Bot, db: Database
) -> tuple[FastAPI, Optional[YooKassaProcessor]]:
    processor: Optional[YooKassaProcessor] = None
    if config.yookassa_enabled and config.public_base_url:
        processor = YooKassaProcessor(db=db, config=config, bot=bot)
    else:
        log.info("YooKassa integration disabled or misconfigured")

    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware, max_body_size=MAX_REQUEST_SIZE)

    app.include_router(_health_router())
    app.include_router(_telegram_router(dp))
    app.include_router(_yookassa_router(config=config, processor=processor))

    return app, processor


__all__ = [
    "BodySizeLimitMiddleware",
    "YooKassaProcessor",
    "create_app",
    "poll_pending_payments",
]
