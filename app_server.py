"""Unified FastAPI application exposing health, payments and Telegram webhooks."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from ipaddress import ip_address, ip_network
from typing import Any, Dict, Optional

from aiogram import Bot, Dispatcher, types
from fastapi import APIRouter, FastAPI, Header, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from config import CFG, Config
from db import DatabaseInterface
from handlers import resend_pending_order
from i18n import format_credits, i18n
from monitoring.metrics import router as metrics_router
from services import gsheets_ref
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


def _debug_updates_enabled() -> bool:
    return os.getenv("TG_DEBUG_UPDATES", "").strip().lower() in {"1", "true", "yes", "on"}


def _extract_update_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    message = data.get("message") if isinstance(data.get("message"), dict) else {}
    callback_query = (
        data.get("callback_query") if isinstance(data.get("callback_query"), dict) else {}
    )
    callback_message = (
        callback_query.get("message") if isinstance(callback_query.get("message"), dict) else {}
    )
    user = message.get("from") if isinstance(message.get("from"), dict) else {}
    callback_user = (
        callback_query.get("from") if isinstance(callback_query.get("from"), dict) else {}
    )
    chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
    callback_chat = (
        callback_message.get("chat") if isinstance(callback_message.get("chat"), dict) else {}
    )
    return {
        "update_id": data.get("update_id"),
        "has_message": bool(message),
        "has_callback": bool(callback_query),
        "text": message.get("text") if isinstance(message.get("text"), str) else None,
        "callback_data": (
            callback_query.get("data") if isinstance(callback_query.get("data"), str) else None
        ),
        "user_id": user.get("id") or callback_user.get("id"),
        "chat_id": chat.get("id") or callback_chat.get("id"),
    }


async def _load_available_models_async() -> None:
    """Placeholder for asynchronous model loading during startup."""

    await asyncio.sleep(0)


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

    def __init__(self, *, db: DatabaseInterface, config: Config, bot: Bot) -> None:
        self._db = db
        self._config = config
        self._bot = bot
        self.credits_credited_total = 0
        self.credits_refused_total = 0
        self.duplicate_webhooks_total = 0
        self._payment_locks: dict[str, asyncio.Lock] = {}

    async def process_payment(self, payment, *, source: str) -> None:
        metadata = self._extract_metadata(payment)
        payment_id = getattr(payment, "id", None)
        if payment_id is None:
            log.warning("Incomplete YooKassa payment payload from %s without id", source)
            return
        lock = self._payment_locks.setdefault(payment_id, asyncio.Lock())
        async with lock:
            await self._process_payment_locked(
                payment=payment,
                payment_id=payment_id,
                metadata=metadata,
                source=source,
            )

    async def _process_payment_locked(
        self,
        *,
        payment,
        payment_id: str,
        metadata: Dict[str, Any],
        source: str,
    ) -> None:
        status = getattr(payment, "status", "") or ""
        paid = bool(getattr(payment, "paid", False))
        amount_obj = getattr(payment, "amount", None)
        amount_cp = self._amount_to_cop(amount_obj)
        amount_value_str = ""
        amount_currency = ""
        if amount_obj is not None:
            value_raw = getattr(amount_obj, "value", "")
            currency_raw = getattr(amount_obj, "currency", "")
            if value_raw is not None:
                amount_value_str = str(value_raw)
            if currency_raw is not None:
                amount_currency = str(currency_raw)
        if not amount_value_str and amount_cp:
            try:
                amount_value_str = str(
                    (Decimal(amount_cp) / Decimal(100)).quantize(Decimal("0.01"))
                )
            except Exception:
                amount_value_str = str(amount_cp)
        idempotency_key = str(
            metadata.get("idempotency_key")
            or metadata.get("idempotency_key".upper())
            or metadata.get("idemp")
            or ""
        )
        existing = await self._db.get_payment_by_ext("yookassa", payment_id)
        existing_metadata = self._parse_metadata_field(
            existing.get("metadata") if existing else None
        )
        metadata = {**existing_metadata, **metadata}
        user_id_int = self._resolve_user_id(metadata, existing)
        if not user_id_int:
            log.warning(
                "YooKassa payment %s missing valid user_id (source=%s)",
                payment_id,
                source,
            )
            return

        package_id = str(
            metadata.get("package_id") or (existing.get("package_id") if existing else "") or ""
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

        already_processed = bool(
            existing
            and (
                existing.get("status") == "succeeded"
                or existing.get("processed_at")
                or existing_metadata.get("credits_added")
            )
        )
        if already_processed:
            self.duplicate_webhooks_total += 1
            log.info("Duplicate success webhook for payment %s (source=%s)", payment_id, source)
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

        # Mark payment as processed BEFORE crediting to prevent double-processing
        await self._db.update_payment_status_by_ext(
            "yookassa",
            payment_id,
            "succeeded",
            credits_added=0,
            metadata={**metadata, **metadata_update},
            idempotency_key=idempotency_key,
            processed_at=processed_at,
            package_id=package.package_id,
            purchased_credits=purchased_credits,
        )

        # Now credit the user (if this fails, payment is already marked as processed)
        balance_after = await self._db.add_credits(user_id_int, purchased_credits)
        self.credits_credited_total += purchased_credits

        # Update metadata with actual credits added
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
        await self._apply_referral_bonus(
            payer_user_id=user_id_int,
            payment_id=payment_id,
            amount_value=amount_value_str,
            currency=amount_currency,
        )
        await self._notify_success(user_id_int, purchased_credits)

    async def poll_pending_once(self) -> None:
        pending = await self._db.list_payments_by_status(
            ("pending", "waiting_for_capture"),
            provider="yookassa",
            limit=200,
        )
        if not pending:
            return
        log.info("YooKassa poller found %s pending payments", len(pending))
        for record in pending:
            payment_id = record.get("ext_id")
            if not payment_id:
                continue
            log.info(
                "YooKassa poller refresh ext_id=%s user=%s status=%s created_at=%s",
                payment_id,
                record.get("user_id"),
                record.get("status"),
                record.get("created_at"),
            )
            try:
                payment = await asyncio.to_thread(yookassa_client.get_payment, payment_id)
            except Exception as exc:  # pragma: no cover - network interaction
                error_details = yookassa_client.describe_error(exc)
                suffix = f" ({error_details})" if error_details else ""
                log.exception("Failed to refresh YooKassa payment %s%s", payment_id, suffix)
                continue
            status = getattr(payment, "status", None) or "unknown"
            paid_flag = getattr(payment, "paid", None)
            log.info(
                "YooKassa poller status id=%s status=%s paid=%s",
                payment_id,
                status,
                paid_flag,
            )
            await self.process_payment(payment, source="poller")

    async def _notify_success(self, user_id: int, items: int) -> None:
        try:
            if items > 0:
                await self._bot.send_message(
                    user_id,
                    i18n.t("payment.received", items=format_credits(items)),
                )
            await resend_pending_order(
                bot=self._bot, db=self._db, config=self._config, user_id=user_id
            )
        except Exception:  # pragma: no cover - Telegram API interaction
            log.exception("Failed to notify user %s about YooKassa success", user_id)

    async def _apply_referral_bonus(
        self,
        *,
        payer_user_id: int,
        payment_id: str,
        amount_value: str,
        currency: str,
    ) -> None:
        try:
            referral_row = await gsheets_ref.find_ref_by_new_user_id(
                payer_user_id, only_pending=True
            )
        except Exception:
            log.exception(
                "Failed to lookup referral for payer=%s payment=%s",
                payer_user_id,
                payment_id,
            )
            return
        if not referral_row:
            return
        referrer_raw = referral_row.get("referrer_user_id")
        try:
            referrer_id = int(str(referrer_raw))
        except (TypeError, ValueError):
            log.warning(
                "Invalid referrer id %r for referral payer=%s payment=%s",
                referrer_raw,
                payer_user_id,
                payment_id,
            )
            return
        if referrer_id == payer_user_id:
            log.warning(
                "Skipping referral bonus for self-referral referrer=%s payment=%s",
                referrer_id,
                payment_id,
            )
            return
        bonus_credits = 100
        try:
            await self._db.ensure_user(referrer_id, None)
            await self._db.add_credits(referrer_id, bonus_credits)
        except Exception:
            log.exception(
                "Failed to credit referral bonus referrer=%s payer=%s payment=%s",
                referrer_id,
                payer_user_id,
                payment_id,
            )
            return
        try:
            marked = await gsheets_ref.mark_credited(
                referral_row,
                amount=amount_value,
                currency=currency,
                payment_id=payment_id,
            )
        except Exception:
            log.exception(
                "Failed to mark referral credited referrer=%s payer=%s payment=%s",
                referrer_id,
                payer_user_id,
                payment_id,
            )
            marked = False
        if not marked:
            log.warning(
                "Referral row not marked credited referrer=%s payer=%s payment=%s",
                referrer_id,
                payer_user_id,
                payment_id,
            )
        try:
            await self._bot.send_message(
                referrer_id,
                "🎉 Твой друг пополнил баланс. +100 кредитов за рефералку начислены.",
            )
        except Exception:
            log.exception("Failed to notify referrer %s about referral bonus", referrer_id)
        try:
            await self._bot.send_message(
                payer_user_id,
                "Спасибо за пополнение! Твоя покупка засчитана по реферал-ссылке друга.",
            )
        except Exception:
            log.exception(
                "Failed to notify payer %s about referral attribution",
                payer_user_id,
            )
        log.info(
            "Referral bonus granted referrer=%s payer=%s payment=%s",
            referrer_id,
            payer_user_id,
            payment_id,
        )

    def _extract_metadata(self, payment) -> dict:
        metadata = getattr(payment, "metadata", None) or {}
        if isinstance(metadata, dict):
            return metadata
        try:
            return dict(metadata)
        except Exception:  # pragma: no cover - defensive
            return {}

    @staticmethod
    def _parse_metadata_field(metadata: Any) -> Dict[str, Any]:
        if not metadata:
            return {}
        if isinstance(metadata, dict):
            return dict(metadata)
        if isinstance(metadata, str):
            try:
                parsed = json.loads(metadata)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return parsed
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
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _resolve_user_id(
        self, metadata: Dict[str, Any], existing: Optional[Dict[str, Any]]
    ) -> Optional[int]:
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


def _validate_yookassa_signature(
    body: bytes, signature_header: str | None, secret_key: Optional[str]
) -> bool:
    if not secret_key:
        return False
    candidate = (signature_header or "").strip()
    if not candidate:
        return False
    if "=" in candidate:
        prefix, provided = candidate.split("=", 1)
        if prefix.lower() != "sha256":
            provided = candidate
    else:
        provided = candidate
    if not provided:
        return False
    digest = hmac.new(secret_key.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, provided)


async def poll_pending_payments(processor: YooKassaProcessor, interval: int) -> None:
    """Background task to poll pending YooKassa payments."""

    backoff = interval
    max_backoff = max(interval * 5, interval + 60)
    while True:
        try:
            await processor.poll_pending_once()
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            raise
        except Exception:  # pragma: no cover - defensive
            log.exception("YooKassa pending poller error")
            backoff = min(max_backoff, int(backoff * 2))
        else:
            if backoff != interval:
                log.info("YooKassa poller recovered; backoff reset")
            backoff = interval
            log.info("YooKassa poller tick ok; sleeping %ss", backoff)
        await asyncio.sleep(backoff)


def _health_router() -> APIRouter:
    router = APIRouter()

    @router.get("/healthz")
    async def healthz() -> Dict[str, object]:
        return {"ok": True, "mode": CFG.BOT_MODE}

    return router


def _yookassa_router(*, config: Config, processor: Optional[YooKassaProcessor]) -> APIRouter:
    router = APIRouter()

    @router.post(config.yookassa_webhook_path)
    async def webhook(
        request: Request, x_yookassa_signature: str | None = Header(default=None)
    ) -> Response:
        if processor is None:
            return Response(status_code=503)
        remote = request.client.host if request.client else "unknown"
        log.info("Received YooKassa webhook from %s", remote)
        if not config.yookassa_test_mode and not _is_ip_allowed(remote):
            log.warning("Rejected YooKassa webhook from unauthorized IP %s", remote)
            return Response(status_code=403)
        body = await request.body()
        if not _validate_yookassa_signature(body, x_yookassa_signature, config.yookassa_secret_key):
            log.warning("Rejected YooKassa webhook with invalid signature from %s", remote)
            return Response(status_code=401)
        try:
            payload = json.loads(body.decode("utf-8"))
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


def _telegram_router(dp: Dispatcher, bot: Bot) -> APIRouter:
    router = APIRouter()

    webhook_paths = [CFG.WEBHOOK_PATH]
    if CFG.WEBHOOK_PATH != "/tg/webhook":
        webhook_paths.append("/tg/webhook")

    async def tg_webhook(
        request: Request,
        x_telegram_bot_api_secret_token: str | None = Header(default=None),
    ) -> Response:
        webhook_secret = CFG.TG_WEBHOOK_SECRET
        if webhook_secret:
            if x_telegram_bot_api_secret_token != webhook_secret:
                log.warning(
                    "Webhook: bad secret token from %s",
                    request.client.host if request.client else "unknown",
                )
                return Response(
                    status_code=200
                )  # Telegram requires 2xx; update is silently dropped

        remote = request.client.host if request.client else "unknown"
        raw = await request.body()
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            log.exception("Webhook: invalid JSON body", extra={"raw_length": len(raw)})
            return Response(status_code=200)

        log.info(
            "tg.webhook.hit",
            extra={
                "remote": remote,
                "update_id": data.get("update_id"),
                "keys": list(data.keys()),
            },
        )
        if _debug_updates_enabled():
            log.info(
                "tg.webhook.dispatch.debug %s",
                json.dumps(_extract_update_summary(data), ensure_ascii=False),
            )

        try:
            if hasattr(types.Update, "to_object"):
                update = types.Update.to_object(data)
            elif hasattr(types.Update, "model_validate"):
                update = types.Update.model_validate(
                    data,
                    context={"bot": bot, "dispatcher": dp},
                )
            else:
                update = types.Update(**data)
            if hasattr(update, "as_"):
                update = update.as_(bot)
        except Exception:
            log.exception(
                "Webhook: cannot convert to aiogram Update",
                extra={"update_id": data.get("update_id"), "keys": list(data.keys())},
            )
            return Response(status_code=200)

        try:
            set_bot_ctx = getattr(Bot, "set_current", None)
            if callable(set_bot_ctx):
                set_bot_ctx(bot)
            set_dispatcher_ctx = getattr(Dispatcher, "set_current", None)
            if callable(set_dispatcher_ctx):
                set_dispatcher_ctx(dp)

            await dp.process_update(update)
            if _debug_updates_enabled():
                log.info("tg.webhook.processed update_id=%s", data.get("update_id"))
        except Exception:
            log.exception(
                "Webhook: handler crashed",
                extra={"update_id": data.get("update_id")},
            )
            return Response(status_code=200)

        return Response(status_code=200)

    for webhook_path in webhook_paths:
        router.add_api_route(
            webhook_path,
            tg_webhook,
            methods=["POST"],
            include_in_schema=False,
        )

    return router


def _create_base_app() -> FastAPI:
    app = FastAPI(title="Sora2 Bot API")

    @app.get("/health", include_in_schema=False)
    async def health() -> Dict[str, bool]:
        return {"ok": True}

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> Dict[str, bool]:
        return {"ok": True}

    @app.get("/ready", include_in_schema=False)
    async def readiness() -> JSONResponse:
        """Readiness probe: checks that DB is reachable."""
        checks: Dict[str, bool] = {}
        try:
            if hasattr(app.state, "db") and app.state.db is not None:
                await app.state.db.healthcheck()
                checks["db"] = True
            else:
                checks["db"] = False
        except Exception:
            checks["db"] = False
        ok = all(checks.values())
        return JSONResponse(
            content={"ok": ok, **checks},
            status_code=200 if ok else 503,
        )

    @app.get("/", include_in_schema=False)
    async def root() -> Dict[str, str | bool]:
        return {"ok": True, "service": "sora2-bot", "mode": "webhook"}

    app.add_middleware(BodySizeLimitMiddleware, max_body_size=MAX_REQUEST_SIZE)
    app.include_router(metrics_router)

    @app.on_event("startup")
    async def _startup_bg() -> None:
        if os.getenv("SKIP_GEMINI_MODEL_LOADING", "1") == "1":
            return
        loader_task = asyncio.create_task(_load_available_models_async())
        app.state.model_loader_task = loader_task

    return app


# [FASTAPI_APP_FACTORY]
def create_app(
    *, config: Config, dp: Dispatcher, bot: Bot, db: DatabaseInterface
) -> tuple[FastAPI, Optional[YooKassaProcessor]]:
    processor: Optional[YooKassaProcessor] = None
    if config.yookassa_enabled and config.public_base_url:
        processor = YooKassaProcessor(db=db, config=config, bot=bot)
    else:
        log.info("YooKassa integration disabled or misconfigured")

    app = _create_base_app()
    app.state.db = db

    app.include_router(_telegram_router(dp, bot))
    app.include_router(_yookassa_router(config=config, processor=processor))

    return app, processor


app = _create_base_app()


@app.exception_handler(Exception)
async def _unhandled_exc(request: Request, exc: Exception):
    log.exception("Unhandled exception", extra={"path": str(request.url)})
    return JSONResponse({"error": "internal"}, status_code=500)


__all__ = [
    "BodySizeLimitMiddleware",
    "YooKassaProcessor",
    "app",
    "create_app",
    "poll_pending_payments",
]
