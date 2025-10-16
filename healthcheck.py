"""Startup health check utilities."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

import gsheets_db

from config import Config
from observability import HealthCheckResult, record_healthcheck
from sora_client import SoraAPIError, SoraClient

log = logging.getLogger(__name__)


async def run_startup_healthcheck(*, config: Config, mode: str, sora_client: SoraClient) -> HealthCheckResult:
    checks: List[Dict[str, Any]] = []
    errors: List[str] = []

    def add_check(label: str, ok: bool, detail: Any = "") -> None:
        entry: Dict[str, Any] = {"label": label, "ok": bool(ok)}
        if detail not in (None, ""):
            entry["detail"] = detail
        if not ok:
            errors.append(f"{label}: {detail if detail else 'failed'}")
        checks.append(entry)

    add_check("OPENAI_API_KEY", bool(config.sora_api_key))
    add_check("SORA_MODEL", config.sora_model == "sora-2", config.sora_model)
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

    sora_ok = True
    latency_ms: float | None = None
    try:
        start = time.monotonic()
        await sora_client.list_models()
        latency_ms = (time.monotonic() - start) * 1000
    except SoraAPIError as exc:
        sora_ok = False
        latency_ms = exc.duration_ms
        errors.append(f"sora_api: {exc.status_code} {exc.error_type or exc.error_code or exc}")
        log.warning("Sora healthcheck failed", exc_info=True)
    except Exception as exc:  # pragma: no cover - defensive
        sora_ok = False
        errors.append(f"sora_api: {exc}")
        log.warning("Sora healthcheck failed", exc_info=True)
    add_check("Sora ping", sora_ok, f"{latency_ms:.0f}ms" if latency_ms is not None else "n/a")

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
