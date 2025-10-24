"""Startup health check utilities."""
from __future__ import annotations
import asyncio
import base64
import binascii
import json
import logging
import random
import time
from typing import Any, Dict, List, Tuple

import gsheets_db
from google.genai import errors as _genai_errors, types as _genai_types

from config import Config
from observability import HealthCheckResult, record_healthcheck
from services.gemini_client import get_gemini_client

log = logging.getLogger(__name__)


def _collect_files(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    stack: List[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "files" and isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            files.append(item)
                if isinstance(value, (dict, list, tuple)):
                    stack.append(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                if isinstance(item, (dict, list, tuple)):
                    stack.append(item)
    return files


def _supports_method(catalog: Dict[str, List[str]], model: str, method: str) -> bool:
    methods = catalog.get(model, [])
    target = method.replace("-", "_").lower()
    for value in methods:
        cleaned = str(value).replace("-", "_").lower()
        if cleaned == target:
            return True
    return False


async def _probe_gemini_models(config: Config) -> Tuple[bool, str]:
    if not config.gemini_enabled:
        return True, "disabled"
    if not config.gemini_api_key:
        return False, "missing api key"
    clients: Dict[str, Any] = {}
    for version in ("v1", "v1beta"):
        try:
            clients[version] = get_gemini_client(config, api_version=version)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Gemini client init failed version=%s", version, exc_info=True)

    if not clients:
        return False, "models.list:no_client"

    supported: Dict[str, List[str]] = {}
    model_versions: Dict[str, str] = {}

    for version, client in clients.items():
        try:
            catalog = await asyncio.to_thread(lambda cl=client: list(cl.models.list()))
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Gemini models.list failed version=%s", version, exc_info=True)
            continue
        for item in catalog:
            name = getattr(item, "name", None) or getattr(item, "model", None) or ""
            short = name.split("/")[-1] if name else ""
            methods = getattr(item, "supported_generation_methods", None) or []
            method_list = [str(method) for method in methods]
            if short:
                supported[short] = method_list
                model_versions.setdefault(short, version)
            if name:
                supported[name] = method_list
                model_versions.setdefault(name, version)

    if not supported:
        return False, "models.list:empty"

    detail: Dict[str, Any] = {"catalog": supported}

    def _resolve_version(model_name: str, default_version: str) -> str:
        if model_name in model_versions:
            return model_versions[model_name]
        short_name = model_name.split("/")[-1]
        return model_versions.get(short_name, default_version)

    text_model = (config.gemini_model_text or "").strip()
    if text_model:
        if text_model not in supported:
            return False, f"text model {text_model} missing"
        if not _supports_method(supported, text_model, "generate_content"):
            return False, f"text model {text_model} lacks generate_content"

    image_model = (config.gemini_model_image or "").strip()
    if image_model:
        if image_model not in supported:
            return False, f"image model {image_model} missing"
        if not _supports_method(supported, image_model, "generate_images"):
            return False, f"image model {image_model} lacks generate_images"
        image_version = _resolve_version(image_model, "v1")
        image_client = clients.get(image_version) or clients.get("v1")
        if image_client is None:
            return False, f"image client {image_version} unavailable"
        start = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    image_client.models.generate_images,
                    model=image_model,
                    prompt="ping",
                ),
                timeout=60.0,
            )
        except _genai_errors.APIError as exc:
            detail_msg = f"{getattr(exc, 'code', 0)}:{getattr(exc, 'status', '')}".strip(":")
            log.warning("Gemini image ping failed model=%s", image_model, exc_info=True)
            return False, detail_msg or str(exc)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Gemini image ping crashed model=%s", image_model, exc_info=True)
            return False, str(exc)
        duration_ms = int((time.perf_counter() - start) * 1000)
        image_candidates: List[Any] = []
        for attr in ("images", "generated_images", "generatedImages"):
            value = getattr(response, attr, None)
            if value is None and isinstance(response, dict):
                value = response.get(attr)
            if value:
                if isinstance(value, (list, tuple)):
                    image_candidates = list(value)
                else:
                    image_candidates = [value]
                break

        image_found = False
        for candidate in image_candidates:
            data_field: Any = None
            if hasattr(candidate, "data"):
                data_field = getattr(candidate, "data")
            elif hasattr(candidate, "image_bytes"):
                data_field = getattr(candidate, "image_bytes")
            elif isinstance(candidate, dict):
                data_field = candidate.get("data") or candidate.get("image_bytes")
            if data_field is None:
                inline_candidate = getattr(candidate, "inline_data", None) or getattr(
                    candidate, "inlineData", None
                )
                if hasattr(inline_candidate, "data"):
                    data_field = getattr(inline_candidate, "data")
                elif isinstance(inline_candidate, dict):
                    data_field = (
                        inline_candidate.get("data")
                        or inline_candidate.get("data_base64")
                        or inline_candidate.get("dataBase64")
                    )
            if isinstance(data_field, str):
                try:
                    decoded = base64.b64decode(data_field)
                except (binascii.Error, ValueError):
                    decoded = b""
                if decoded:
                    image_found = True
                    break
            elif isinstance(data_field, (bytes, bytearray, memoryview)):
                if data_field:
                    image_found = True
                    break
            if isinstance(candidate, dict):
                uri = candidate.get("uri") or candidate.get("url")
            else:
                uri = getattr(candidate, "uri", None) or getattr(candidate, "url", None)
            if uri:
                image_found = True
                break
        if not image_found:
            log.warning("Gemini image ping returned no binary data model=%s", image_model)
            return False, "image:no_binary"
        detail.setdefault("image", {})
        detail["image"].update(
            {"model": image_model, "latency_ms": duration_ms, "version": image_version}
        )

    video_model = (config.gemini_model_video or "").strip()
    if video_model:
        if video_model not in supported:
            return False, f"video model {video_model} missing"
        if not _supports_method(supported, video_model, "generate_videos"):
            return False, f"video model {video_model} lacks generate_videos"
        video_version = _resolve_version(video_model, "v1beta")
        video_client = clients.get(video_version) or clients.get("v1beta")
        if video_client is None:
            return False, f"video client {video_version} unavailable"
        video_config = _genai_types.GenerateVideosConfig(duration_seconds=2)
        try:
            operation = await asyncio.wait_for(
                asyncio.to_thread(
                    video_client.models.generate_videos,
                    model=video_model,
                    prompt="ping",
                    config=video_config,
                ),
                timeout=60.0,
            )
        except _genai_errors.APIError as exc:
            detail_msg = f"{getattr(exc, 'code', 0)}:{getattr(exc, 'status', '')}".strip(":")
            log.warning("Gemini video submit failed model=%s", video_model, exc_info=True)
            return False, detail_msg or str(exc)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Gemini video submit crashed model=%s", video_model, exc_info=True)
            return False, str(exc)

        operation_name = getattr(operation, "name", None) or ""
        if not operation_name:
            return False, "video:missing_operation_name"

        poll_start = time.perf_counter()
        poll_deadline = poll_start + 45.0
        refreshed = operation
        polls = 0
        while time.perf_counter() < poll_deadline:
            polls += 1
            refreshed = await asyncio.to_thread(video_client.operations.get, operation_name)
            if refreshed.error:
                error_message = (
                    refreshed.error.get("message")
                    if isinstance(refreshed.error, dict)
                    else str(refreshed.error)
                )
                try:
                    await asyncio.to_thread(video_client.operations.cancel, operation_name)
                except Exception:
                    log.debug("Gemini video cancel failed name=%s", operation_name, exc_info=True)
                return False, f"video:error:{error_message}"
            if getattr(refreshed, "done", False):
                break
            await asyncio.sleep(random.uniform(2.0, 3.5))

        cancelled = False
        try:
            await asyncio.to_thread(video_client.operations.cancel, operation_name)
            cancelled = True
        except Exception:
            log.debug("Gemini video cancel failed name=%s", operation_name, exc_info=True)

        detail.setdefault("video", {})
        detail_video = detail["video"]
        detail_video.update(
            {
                "model": video_model,
                "version": video_version,
                "polls": polls,
                "operation": operation_name,
                "cancelled": cancelled,
                "latency_ms": int((time.perf_counter() - poll_start) * 1000),
                "done": bool(getattr(refreshed, "done", False)),
            }
        )

        if getattr(refreshed, "done", False):
            payload = refreshed.model_dump(exclude_none=True)
            files = _collect_files(payload)
            if not files:
                return False, "video:no_media"
            detail_video["files"] = len(files)

    detail_json = json.dumps(detail, ensure_ascii=False)
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
