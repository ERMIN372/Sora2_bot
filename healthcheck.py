"""Startup health check utilities."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Tuple

import aiohttp
import gsheets_db
from google.genai import Client as _GenAIClient, errors as _genai_errors

from config import Config
from observability import HealthCheckResult, record_healthcheck

log = logging.getLogger(__name__)


async def _probe_gemini_models(config: Config) -> Tuple[bool, str]:
    if not config.gemini_enabled:
        return True, "disabled"
    if not config.gemini_api_key:
        return False, "missing api key"
    client = _GenAIClient(api_key=config.gemini_api_key)
    model_name = config.gemini_model_image or config.gemini_model_text
    try:
        response = await asyncio.to_thread(client.models.get, name=model_name)
    except _genai_errors.APIError as exc:  # pragma: no cover - external API
        detail = f"{getattr(exc, 'code', 0)}:{getattr(exc, 'status', '')}".strip(":")
        log.warning("Gemini models probe failed: %s", exc, exc_info=True)
        return False, detail or str(exc)
    except Exception as exc:  # pragma: no cover - network guard
        log.warning("Gemini models probe failed", exc_info=True)
        return False, str(exc)
    display = getattr(response, "name", None) or model_name
    log.info("Gemini models: ok %s", display)
    return True, display


async def _probe_sora_models(config: Config) -> Tuple[bool, str]:
    if not config.sora_enabled:
        return True, "disabled"
    if not config.sora_api_key:
        return False, "missing credentials"
    url = f"{config.sora_api_base.rstrip('/')}/models"
    headers = {
        "Authorization": f"Bearer {config.sora_api_key}",
        "Content-Type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(
        total=None,
        sock_connect=config.request_connect_timeout,
        sock_read=config.request_read_timeout,
    )
    payload: Dict[str, Any] = {}
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                text = await response.text()
                if response.status >= 400:
                    short = text[:200].replace("\n", " ")
                    return False, f"{response.status}: {short}"
                try:
                    payload = json.loads(text) if text else {}
                except json.JSONDecodeError:
                    return False, "invalid JSON"
    except Exception as exc:  # pragma: no cover - network guard
        log.warning("Sora models probe failed", exc_info=True)
        return False, str(exc)
    if isinstance(payload, dict):
        models = payload.get("data") or payload.get("models")
        if isinstance(models, list):
            return True, f"{len(models)} models"
    return True, "ok"


async def run_startup_healthcheck(*, config: Config, mode: str) -> HealthCheckResult:
    checks: List[Dict[str, Any]] = []
    errors: List[str] = []

    def add_check(label: str, ok: bool, detail: Any = "") -> None:
        entry: Dict[str, Any] = {"label": label, "ok": bool(ok)}
        if detail not in (None, ""):
            entry["detail"] = detail
        if not ok:
            errors.append(f"{label}: {detail if detail else 'failed'}")
        checks.append(entry)

    add_check("GEMINI_ENABLED", config.gemini_enabled, str(config.gemini_enabled).lower())
    if config.gemini_enabled:
        add_check("GEMINI_MODEL_TEXT", bool(config.gemini_model_text), config.gemini_model_text)
        add_check("GEMINI_MODEL_IMAGE", bool(config.gemini_model_image), config.gemini_model_image)
    else:
        add_check("GEMINI_MODEL_TEXT", True, "disabled")
        add_check("GEMINI_MODEL_IMAGE", True, "disabled")
    add_check("SORA_ENABLED", config.sora_enabled, str(config.sora_enabled).lower())
    if config.sora_enabled:
        add_check("SORA_API_KEY", bool(config.sora_api_key))
        add_check("SORA_API_BASE", bool(config.sora_api_base), config.sora_api_base)
    else:
        add_check("SORA_API_KEY", True, "disabled")
        add_check("SORA_API_BASE", True, "disabled")
    add_check("CREDIT_PRICE_KOPEKS", config.credit_price_kopeks > 0, config.credit_price_kopeks)
    add_check("SORA_VIDEO_CREDITS", config.generation_cost_credits > 0, config.generation_cost_credits)

    sheets_ok = True
    sheet_titles: List[str] = []
    try:
        sheet_titles = await gsheets_db.list_worksheet_titles()
    except Exception as exc:  # pragma: no cover - external dependency
        sheets_ok = False
        errors.append(f"gsheets: {exc}")
        log.warning("Google Sheets healthcheck failed", exc_info=True)
    add_check("Google Sheets", sheets_ok, ", ".join(sheet_titles) if sheet_titles else "unavailable")

    if config.yookassa_enabled:
        add_check("YooKassa", config.yookassa_ready, "готово" if config.yookassa_ready else "нет base URL")
    else:
        add_check("YooKassa", True, "отключено")

    gemini_ok, gemini_detail = await _probe_gemini_models(config)
    add_check("Gemini models", gemini_ok, gemini_detail)

    sora_ok, sora_detail = await _probe_sora_models(config)
    add_check("Sora models", sora_ok, sora_detail)

    details: Dict[str, Any] = {"mode": mode, "checks": checks}
    ok = all(item.get("ok") for item in checks)
    result = HealthCheckResult(
        ts=time.time(),
        mode=mode,
        ok=ok,
        version=config.bot_version,
        details=details,
        errors=errors,
    )

    payload = {
        "event": "startup_healthcheck",
        "mode": mode,
        "version": config.bot_version,
        "checks": checks,
        "errors": errors,
        "startup_ok": ok,
    }
    log.info(json.dumps(payload, ensure_ascii=False))
    if not ok and errors:
        log.info("startup_ok=false details=%s", "; ".join(errors))
    else:
        log.info("startup_ok=%s", str(ok).lower())
    record_healthcheck(result)
    return result


__all__ = ["run_startup_healthcheck"]
