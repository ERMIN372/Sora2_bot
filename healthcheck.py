"""Startup health check utilities."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Mapping, Tuple

import gsheets_db
from google.genai import Client as _GenAIClient, errors as _genai_errors

from config import Config
from observability import HealthCheckResult, record_healthcheck
from services import gemini_safety

log = logging.getLogger(__name__)


async def _probe_gemini_models(config: Config) -> Tuple[bool, str]:
    if not config.gemini_enabled:
        return True, "disabled"
    if not config.gemini_api_key:
        return False, "missing api key"
    client = _GenAIClient(api_key=config.gemini_api_key)
    model_names = [
        name
        for name in (
            config.gemini_model_text,
            config.gemini_model_image,
            config.gemini_model_video,
        )
        if name
    ]
    checked: List[str] = []
    for model_name in model_names:
        try:
            response = await asyncio.to_thread(
                client.models.get, model=model_name
            )
        except _genai_errors.APIError as exc:  # pragma: no cover - external API
            detail = f"{getattr(exc, 'code', 0)}:{getattr(exc, 'status', '')}".strip(":")
            log.warning(
                "Gemini model probe failed name=%s error=%s", model_name, exc, exc_info=True
            )
            return False, detail or str(exc)
        except Exception as exc:  # pragma: no cover - network guard
            log.warning("Gemini model probe failed name=%s", model_name, exc_info=True)
            return False, str(exc)
        else:
            display = getattr(response, "name", None) or model_name
            checked.append(display)

    try:
        image_response = await asyncio.to_thread(
            client.models.generate_images,
            model=config.gemini_model_image,
            prompt="ping",
        )
    except _genai_errors.APIError as exc:  # pragma: no cover - external API
        detail = f"{getattr(exc, 'code', 0)}:{getattr(exc, 'status', '')}".strip(":")
        log.warning("Gemini image ping failed error=%s", exc, exc_info=True)
        return False, detail or str(exc)
    except Exception as exc:  # pragma: no cover - network guard
        log.warning("Gemini image ping failed", exc_info=True)
        return False, str(exc)
    else:
        binary_ok = False
        if isinstance(image_response, (bytes, bytearray)):
            binary_ok = True
        else:
            generated = getattr(image_response, "generated_images", None)
            if not generated:
                generated = getattr(image_response, "generatedImages", None)
            if generated:
                candidates = generated if isinstance(generated, (list, tuple)) else [generated]
                for item in candidates:
                    image_obj = None
                    if hasattr(item, "image"):
                        image_obj = getattr(item, "image")
                    elif isinstance(item, dict):
                        image_obj = item.get("image")
                    if image_obj is None and isinstance(item, dict):
                        image_obj = item.get("inlineData") or item.get("inline_data")
                    if image_obj is None:
                        continue
                    data = None
                    if hasattr(image_obj, "image_bytes"):
                        data = getattr(image_obj, "image_bytes")
                    elif isinstance(image_obj, dict):
                        data = image_obj.get("image_bytes") or image_obj.get("data")
                    elif isinstance(image_obj, (bytes, bytearray)):
                        data = image_obj
                    if isinstance(data, (bytes, bytearray)) and data:
                        binary_ok = True
                        break
        if not binary_ok:
            log.warning(
                "Gemini image ping produced invalid payload type=%s", type(image_response)
            )
            return False, "invalid image payload"
        checked.append("generate_images")
    detail = ", ".join(dict.fromkeys(checked)) if checked else "ok"
    log.info("Gemini models: ok %s", detail)
    return True, detail


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
    safety_state = gemini_safety.health_snapshot()
    profiles_info = safety_state.get("profiles", {})
    thresholds_detail: List[str] = []
    if isinstance(profiles_info, Mapping):
        for profile, info in profiles_info.items():
            if not isinstance(info, Mapping):
                continue
            thresholds = info.get("thresholds") or []
            dropped = info.get("dropped") or []
            label_parts = [
                f"{category}={threshold}"
                for category, threshold in thresholds
                if category
            ]
            if dropped:
                label_parts.append(f"dropped({', '.join(str(item) for item in dropped)})")
            thresholds_detail.append(
                f"{profile}: {'; '.join(label_parts) if label_parts else 'n/a'}"
            )
    last_fallback = safety_state.get("last_fallback") or {}
    if isinstance(last_fallback, Mapping):
        for profile, items in last_fallback.items():
            if isinstance(items, Mapping):
                for category, ts in items.items():
                    label = str(category or "n/a")
                    if isinstance(ts, (int, float)) and ts > 0:
                        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
                        thresholds_detail.append(f"fallback {profile}:{label}@{timestamp}")
                    else:
                        thresholds_detail.append(f"fallback {profile}:{label}")
            elif items:
                thresholds_detail.append(f"fallback {profile}:{items}")
    elif last_fallback:
        thresholds_detail.append(f"fallback seen: {last_fallback}")
    add_check("Gemini safety", True, ", ".join(thresholds_detail) if thresholds_detail else "n/a")

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
