"""Startup health check utilities."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Tuple

import aiohttp
import gsheets_db

from config import Config
from providers.google_auth import ServiceAccountTokenProvider
from observability import HealthCheckResult, record_healthcheck

log = logging.getLogger(__name__)


async def _probe_vertex_models(config: Config) -> Tuple[bool, str]:
    if not config.vertex_enabled:
        return True, "disabled"
    if not config.google_service_account or not config.gcp_project_id:
        return False, "missing credentials"
    base = (
        f"https://{config.vertex_location}-aiplatform.googleapis.com/v1/projects/{config.gcp_project_id}"
        f"/locations/{config.vertex_location}/publishers/google/models"
    )
    token_provider = ServiceAccountTokenProvider(
        config.google_service_account,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    try:
        token = await token_provider.get_token()
    except Exception as exc:  # pragma: no cover - auth failure
        return False, f"auth: {exc}"
    headers = {
        "Authorization": f"Bearer {token}",
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
            async with session.get(base, headers=headers) as response:
                text = await response.text()
                if response.status >= 400:
                    short = text[:200].replace("\n", " ")
                    return False, f"{response.status}: {short}"
                try:
                    payload = json.loads(text) if text else {}
                except json.JSONDecodeError:
                    return False, "invalid JSON"
    except Exception as exc:  # pragma: no cover - network guard
        log.warning("Vertex models probe failed", exc_info=True)
        return False, str(exc)
    models = payload.get("models") if isinstance(payload, dict) else None
    if isinstance(models, list):
        return True, f"{len(models)} models"
    return True, "ok"


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

    add_check("VERTEX_ENABLED", config.vertex_enabled, str(config.vertex_enabled).lower())
    if config.vertex_enabled:
        add_check("GOOGLE_SERVICE_ACCOUNT", bool(config.google_service_account))
        add_check("GCP_PROJECT_ID", bool(config.gcp_project_id))
        add_check("VERTEX_LOCATION", bool(config.vertex_location), config.vertex_location)
    else:
        add_check("GOOGLE_SERVICE_ACCOUNT", True, "disabled")
        add_check("GCP_PROJECT_ID", True, "disabled")
        add_check("VERTEX_LOCATION", True, "disabled")
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

    vertex_ok, vertex_detail = await _probe_vertex_models(config)
    add_check("Vertex models", vertex_ok, vertex_detail)

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
