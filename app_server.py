"""Unified FastAPI application exposing health, payments and Telegram webhooks."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from decimal import Decimal, InvalidOperation
from ipaddress import ip_address, ip_network
from typing import Any, Dict, Optional

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

from services.healthcheck import (
    PROBE_REFRESH_INTERVAL_SECONDS,
    run_all_probes,
    schedule_probe_refresh,
)

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
        self.credits_credited_total = 0
        self.credits_refused_total = 0
        self.duplicate_webhooks_total = 0

    async def process_payment(self, payment, *, source: str) -> None:
        metadata = self._extract_metadata(payment)
        payment_id = getattr(payment, "id", None)
        if payment_id is None:
            log.warning("Incomplete YooKassa payment payload from %s without id", source)
            return
        status = getattr(payment, "status", "") or ""
        paid = bool(getattr(payment, "paid", False))
        amount_cp = self._amount_to_cop(getattr(payment, "amount", None))
        idempotency_key = str(
            metadata.get("idempotency_key")
            or metadata.get("idempotency_key".upper())
            or metadata.get("idemp")
            or ""
        )
        existing = await self._db.get_payment_by_ext("yookassa", payment_id)
        user_id_int = self._resolve_user_id(metadata, existing)
        if not user_id_int:
            log.warning(
                "YooKassa payment %s missing valid user_id (source=%s)",
                payment_id,
                source,
            )
            return

        package_id = str(
            metadata.get("package_id")
            or (existing.get("package_id") if existing else "")
            or ""
        )
        purchased_credits = self._coerce_optional_int(metadata.get("purchased_credits"))
        if purchased_credits is None and existing is not None:
            purchased_credits = self._coerce_optional_int(existing.get("purchased_credits"))
        if purchased_credits is None:
            purchased_credits = self._coerce_optional_int(metadata.get("items"))
        expected_amount_cp = self._coerce_optional_int(metadata.get("expected_amount_cp"))
        if expected_amount_cp is None and existing is not None:
            expected_amount_cp = self._coerce_optional_int(existing.get("amount_cp"))

        package = self._config.get_credit_package(package_id) if package_id else None
        purchased_for_record = purchased_credits if purchased_credits is not None else 0
        metadata_payload = json.dumps(metadata, ensure_ascii=False)
        initial_status = existing.get("status") if existing else "pending"

        if not existing:
            await self._db.create_payment_record(
                "yookassa",
                payment_id,
                user_id_int,
                amount_cp or (expected_amount_cp or 0),
                purchased_for_record,
                initial_status,
                payload=metadata_payload,
                metadata=metadata,
                idempotency_key=idempotency_key,
                package_id=package.package_id if package else package_id or "",
                purchased_credits=purchased_credits,
            )
        else:
            await self._db.update_payment_status_by_ext(
                "yookassa",
                payment_id,
                initial_status,
                metadata=metadata,
                idempotency_key=idempotency_key,
                package_id=package.package_id if package else package_id or None,
                purchased_credits=purchased_credits,
            )

        if status != "succeeded":
            target_status = status or initial_status
            await self._db.update_payment_status_by_ext(
                "yookassa",
                payment_id,
                target_status,
                metadata=metadata,
                idempotency_key=idempotency_key,
                package_id=package.package_id if package else package_id or None,
                purchased_credits=purchased_credits,
            )
            return

        if existing and existing.get("status") == "succeeded":
            self.duplicate_webhooks_total += 1
            log.info("Duplicate success webhook for payment %s", payment_id)
            await self._db.update_payment_status_by_ext(
                "yookassa",
                payment_id,
                "succeeded",
                metadata=metadata,
                idempotency_key=idempotency_key,
                package_id=package.package_id if package else package_id or None,
                purchased_credits=purchased_credits,
            )
            return

        validation_error = self._validate_payment(
            payment_id=payment_id,
            paid=paid,
            package=package,
            purchased_credits=purchased_credits,
            expected_amount_cp=expected_amount_cp,
            amount_cp=amount_cp,
            metadata=metadata,
        )
        if validation_error is not None:
            reason, refused_credits = validation_error
            self.credits_refused_total += refused_credits
            log.warning("Refused YooKassa payment %s: %s", payment_id, reason)
            await self._db.update_payment_status_by_ext(
                "yookassa",
                payment_id,
                "refused",
                metadata={"refused_reason": reason, **metadata},
                idempotency_key=idempotency_key,
                package_id=package.package_id if package else package_id or None,
                purchased_credits=purchased_credits,
            )
            return

        assert package is not None  # for type checkers
        assert purchased_credits is not None

        await self._db.ensure_user(user_id_int, None)
        balance_before = await self._db.get_user_credits(user_id_int)
        processed_at = self._now_iso()
        metadata_update: Dict[str, Any] = {
            "validated_amount_cp": amount_cp,
            "validated_at": processed_at,
            "validation_source": source,
        }

        if self._config.payments_read_only:
            log.warning(
                "PAYMENTS_READ_ONLY enabled – recorded YooKassa payment %s without crediting",
                payment_id,
            )
            await self._db.update_payment_status_by_ext(
                "yookassa",
                payment_id,
                "succeeded",
                credits_added=0,
                metadata={**metadata, **metadata_update, "read_only": True},
                idempotency_key=idempotency_key,
                processed_at=processed_at,
                package_id=package.package_id,
                purchased_credits=purchased_credits,
            )
            return

        balance_after = await self._db.add_credits(user_id_int, purchased_credits)
        self.credits_credited_total += purchased_credits
        await self._db.update_payment_status_by_ext(
            "yookassa",
            payment_id,
            "succeeded",
            credits_added=purchased_credits,
            metadata={**metadata, **metadata_update},
            idempotency_key=idempotency_key,
            processed_at=processed_at,
            package_id=package.package_id,
            purchased_credits=purchased_credits,
        )
        log.info(
            "YooKassa payment succeeded id=%s user=%s credits=%s balance=%s→%s amount_cp=%s",
            payment_id,
            user_id_int,
            purchased_credits,
            balance_before,
            balance_after,
            amount_cp,
        )
        await self._notify_success(user_id_int, purchased_credits)

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

    @staticmethod
    def _coerce_optional_int(value: Any) -> Optional[int]:
        if value in (None, "", " "):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat()

    def _resolve_user_id(self, metadata: Dict[str, Any], existing: Optional[Dict[str, Any]]) -> Optional[int]:
        candidate = metadata.get("user_id")
        value = self._coerce_optional_int(candidate)
        if value:
            return value
        if existing:
            return self._coerce_optional_int(existing.get("user_id"))
        return None

    def _validate_payment(
        self,
        *,
        payment_id: str,
        paid: bool,
        package,
        purchased_credits: Optional[int],
        expected_amount_cp: Optional[int],
        amount_cp: int,
        metadata: Dict[str, Any],
    ) -> Optional[tuple[str, int]]:
        if not paid:
            return ("payment not marked as paid", purchased_credits or 0)
        if metadata.get("kind") != "credits":
            return ("unexpected payment kind", purchased_credits or 0)
        if package is None:
            return ("unknown package_id", purchased_credits or 0)
        if purchased_credits is None:
            return ("missing purchased_credits", 0)
        if not (1 <= purchased_credits <= 1000):
            return (f"invalid purchased_credits {purchased_credits}", purchased_credits)
        if purchased_credits != package.credits_int:
            return (
                f"credits mismatch metadata={purchased_credits} expected_package={package.credits_int}",
                purchased_credits,
            )
        if expected_amount_cp is None:
            return ("missing expected_amount_cp", purchased_credits)
        if abs(amount_cp - expected_amount_cp) > 100:
            return (
                f"amount mismatch actual={amount_cp} expected={expected_amount_cp}",
                purchased_credits,
            )
        return None

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

    app.state.probe_task: Optional[asyncio.Task[None]] = None
    app.state.probe_interval: float = PROBE_REFRESH_INTERVAL_SECONDS

    @app.on_event("startup")
    async def _startup_probes() -> None:
        await run_all_probes(config=config, mode=CFG.BOT_MODE)
        interval = getattr(app.state, "probe_interval", PROBE_REFRESH_INTERVAL_SECONDS)
        if interval and interval > 0:
            app.state.probe_task = asyncio.create_task(
                schedule_probe_refresh(
                    config=config,
                    interval_seconds=interval,
                    mode=CFG.BOT_MODE,
                )
            )

    @app.on_event("shutdown")
    async def _shutdown_probes() -> None:
        task = getattr(app.state, "probe_task", None)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            app.state.probe_task = None

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
