"""Startup health check utilities."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Tuple

import gsheets_db
from google.genai import errors as _genai_errors

from config import Config
from observability import HealthCheckResult, record_healthcheck
from providers.gemini import _SAFETY_CATEGORIES, _SAFETY_DEFAULT_THRESHOLD
from services.gemini_router import GeminiRoutingError, get_gemini_router

log = logging.getLogger(__name__)


def _enum_name(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


async def _probe_gemini_models(config: Config) -> Tuple[bool, str]:
    if not config.gemini_enabled:
        return True, "disabled"
    if not config.gemini_api_key:
        return False, "missing api key"
    router = get_gemini_router(config)
    summary: Dict[str, Dict[str, Any]] = {}
    tasks: List[Tuple[str, str]] = [
        ("text", config.gemini_model_text),
        ("image", config.gemini_model_image),
        ("video", config.gemini_model_video),
    ]
    for task, model_name in tasks:
        if not model_name:
            continue
        try:
            decision = router.route(
                task=task,
                model=model_name,
                prompt="healthcheck ping",
            )
        except GeminiRoutingError as exc:
            log.warning("Gemini routing failed task=%s", task, exc_info=True)
            return False, f"{task}: {exc}"
        client = decision.client
        try:
            await asyncio.to_thread(client.models.get, model=decision.model)
        except _genai_errors.APIError as exc:  # pragma: no cover - external API
            detail = f"{getattr(exc, 'code', 0)}:{getattr(exc, 'status', '')}".strip(":")
            log.warning(
                "Gemini model probe failed task=%s model=%s error=%s",
                task,
                decision.model,
                exc,
                exc_info=True,
            )
            return False, detail or str(exc)
        except Exception as exc:  # pragma: no cover - network guard
            log.warning(
                "Gemini model probe failed task=%s model=%s",
                task,
                decision.model,
                exc_info=True,
            )
            return False, str(exc)

        entry: Dict[str, Any] = {
            "model": decision.model,
            "api_version": decision.api_version,
            "methods": list(decision.supported_methods),
        }
        if task == "text":
            entry["safety_settings"] = [
                {
                    "category": _enum_name(category),
                    "threshold": _enum_name(_SAFETY_DEFAULT_THRESHOLD),
                }
                for category in _SAFETY_CATEGORIES
            ]
        if task == "image":
            try:
                image_response = await asyncio.to_thread(
                    client.models.generate_images,
                    model=decision.model,
                    prompt="ping",
                )
            except _genai_errors.APIError as exc:  # pragma: no cover - external API
                detail = f"{getattr(exc, 'code', 0)}:{getattr(exc, 'status', '')}".strip(":")
                log.warning("Gemini image ping failed", exc_info=True)
                return False, detail or str(exc)
            except Exception as exc:  # pragma: no cover - network guard
                log.warning("Gemini image ping failed", exc_info=True)
                return False, str(exc)
            else:
                binary_ok = False
                generated = getattr(image_response, "generated_images", None)
                if not generated:
                    generated = getattr(image_response, "generatedImages", None)
                if generated:
                    candidates = generated if isinstance(generated, (list, tuple)) else [generated]
                    for item in candidates:
                        if not isinstance(item, dict):
                            continue
                        image_obj = item.get("image") or {}
                        meta = image_obj
                        if not isinstance(meta, dict):
                            meta = getattr(image_obj, "model_dump", lambda **_: {})()
                        data = None
                        if isinstance(meta, dict):
                            data = meta.get("image_bytes") or meta.get("data")
                        if isinstance(data, (bytes, bytearray)) and data:
                            binary_ok = True
                            break
                if not binary_ok:
                    log.warning(
                        "Gemini image ping produced invalid payload type=%s",
                        type(image_response),
                    )
                    return False, "invalid image payload"
                entry["image_ping"] = "ok"
        summary[task] = entry

    detail_payload = {
        "routes": summary,
        "catalog": router.supported_models,
    }
    detail_json = json.dumps(detail_payload, ensure_ascii=False)
    log.info("Gemini models: ok %s", detail_json)
    return True, detail_json


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
        add_check("GEMINI_MODEL_VIDEO", bool(config.gemini_model_video), config.gemini_model_video)
    else:
        add_check("GEMINI_MODEL_TEXT", True, "disabled")
        add_check("GEMINI_MODEL_IMAGE", True, "disabled")
        add_check("GEMINI_MODEL_VIDEO", True, "disabled")
    add_check("CREDIT_PRICE_KOPEKS", config.credit_price_kopeks > 0, config.credit_price_kopeks)
    add_check("VIDEO_CREDITS", config.generation_cost_credits > 0, config.generation_cost_credits)

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
