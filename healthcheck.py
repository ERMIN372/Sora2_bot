"""Startup health check utilities."""
from __future__ import annotations
import asyncio
import base64
import binascii
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import gsheets_db
from google.genai import errors as _genai_errors, types as _genai_types

from config import (
    Config,
    DEBUG_GEMINI,
    GEMINI_IMAGE_API_VERSION,
    GEMINI_IMAGE_STRICT_INLINE_ONLY,
)
from observability import HealthCheckResult, record_healthcheck
from services.gemini_client import get_media_client, get_text_client
from providers.gemini import (
    _has_generate_content_flag,
    _extract_response_headers,
)
from providers.logx import kv

log = logging.getLogger(__name__)

def _supports_method(catalog: Dict[str, List[str]], model: str, method: str) -> bool:
    methods = catalog.get(model, [])
    target = method.replace("-", "").replace("_", "").lower()
    for value in methods:
        cleaned = str(value).replace("-", "").replace("_", "").lower()
        if cleaned == target:
            return True
    return False


async def _probe_gemini_models(config: Config) -> Tuple[bool, str]:
    if not config.gemini_enabled:
        return True, "disabled"
    if not config.gemini_api_key:
        return False, "missing api key"

    debug_enabled = bool(getattr(config, "debug_gemini", False))
    if debug_enabled and log.getEffectiveLevel() > logging.DEBUG:
        log.setLevel(logging.DEBUG)
    log_method = log.info if debug_enabled else log.debug
    log_method(
        "gemini.health.config env=%s text_model=%s image_model=%s video_model=%s api_mode=%s",
        config.environment,
        config.gemini_model_text,
        config.gemini_model_image,
        config.gemini_model_video,
        config.gemini_api_mode,
    )

    clients: Dict[str, Any] = {}
    for scope, factory in (("text", get_text_client), ("media", get_media_client)):
        try:
            clients[scope] = factory(config)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Gemini client init failed scope=%s", scope, exc_info=True)

    if not clients:
        return False, "models.list:no_client"

    supported: Dict[str, List[str]] = {}
    model_scope: Dict[str, str] = {}
    metadata: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []

    async def _load_catalog(scope: str, client: Any) -> None:
        try:
            catalog = await asyncio.to_thread(lambda cl=client: list(cl.models.list()))
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Gemini models.list failed scope=%s", scope, exc_info=True)
            return
        for item in catalog:
            name = getattr(item, "name", None) or getattr(item, "model", None) or ""
            short = name.split("/")[-1] if name else ""
            raw_methods = (
                getattr(item, "supported_generation_methods", None)
                or getattr(item, "generation_methods", None)
                or getattr(item, "capabilities", None)
                or []
            )
            if isinstance(raw_methods, (list, tuple, set, frozenset)):
                iterable_methods = list(raw_methods)
            elif raw_methods:
                iterable_methods = [raw_methods]
            else:
                iterable_methods = []
            method_list = [str(method) for method in iterable_methods]
            meta: Dict[str, Any] = {}
            dump = getattr(item, "model_dump", None)
            if callable(dump):
                try:
                    dumped = dump(mode="json")  # type: ignore[misc]
                except TypeError:
                    dumped = dump()  # type: ignore[call-arg]
                except Exception:  # pragma: no cover - defensive serialisation
                    dumped = None
                if isinstance(dumped, dict):
                    meta.update(dumped)
            if isinstance(item, dict):
                meta.update(item)
            for attr_name in (
                "supported_generation_methods",
                "generation_methods",
                "capabilities",
            ):
                if attr_name not in meta:
                    value = getattr(item, attr_name, None)
                    if value is not None:
                        meta[attr_name] = value
            if method_list and "supported_generation_methods" not in meta:
                meta["supported_generation_methods"] = method_list
            if name:
                meta.setdefault("name", name)
            meta.setdefault("scope", scope)
            variants = []
            if short:
                variants.append(short)
            if name:
                variants.append(name)
            for variant in variants:
                supported[variant] = list(method_list)
                model_scope.setdefault(variant, scope)
                metadata[variant] = dict(meta)

    await asyncio.gather(*[_load_catalog(scope, client) for scope, client in clients.items()])

    def _get_metadata(model_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not model_name:
            return None, None
        if model_name in metadata:
            return metadata[model_name], model_name
        short_name = model_name.split("/")[-1]
        if short_name in metadata:
            return metadata[short_name], short_name
        return None, None

    if not supported:
        return False, "models.list:empty"

    detail: Dict[str, Any] = {"catalog": supported}

    def _resolve_scope(model_name: str, default_scope: str) -> str:
        if model_name in model_scope:
            return model_scope[model_name]
        short = model_name.split("/")[-1]
        return model_scope.get(short, default_scope)

    text_model = (config.gemini_model_text or "").strip()
    fallback_text_model = getattr(config, "gemini_model_text_fallback", "")
    fallback_text_model = (fallback_text_model or "").strip()
    if text_model:
        text_meta, resolved_key = _get_metadata(text_model)
        fallback_used = False
        if text_meta is None and fallback_text_model and fallback_text_model != text_model:
            fallback_meta, _ = _get_metadata(fallback_text_model)
            if fallback_meta is not None:
                text_meta = fallback_meta
                fallback_used = True
        model_known = resolved_key is not None or text_model in supported
        if not model_known:
            return False, f"text model {text_model} missing"
        text_detail = detail.setdefault("text", {"model": text_model})
        if fallback_used:
            text_detail["checked_with"] = fallback_text_model
        else:
            text_detail["checked_with"] = text_model
        if text_meta is not None and not _has_generate_content_flag(text_meta):
            warning_msg = (
                f"text model {text_model} does not advertise generateContent; continuing (warning)"
            )
            warnings.append(warning_msg)
            text_detail["warning"] = "missing generateContent"
            log.warning(
                "Gemini text model %s does not advertise generateContent; healthcheck continues",
                text_model,
            )

    image_model = (config.gemini_model_image or "").strip()
    if image_model:
        if image_model not in supported:
            return False, f"image model {image_model} missing"
        image_scope = _resolve_scope(image_model, "media")
        image_client = clients.get(image_scope) or clients.get("media")
        if image_client is None:
            return False, f"image client {image_scope} unavailable"
        http_options = getattr(getattr(image_client, "_client", None), "_http_options", None)
        api_version = getattr(http_options, "api_version", None)
        image_detail = detail.setdefault(
            "image", {"model": image_model, "scope": image_scope}
        )
        image_detail["api_version"] = (
            getattr(http_options, "api_version", None) or GEMINI_IMAGE_API_VERSION
        )
        image_detail["ping_mode"] = "generate_content"
        log.info(
            "health.gemini.start %s",
            kv(model=image_model, version=api_version, method="generate_content"),
        )
        start = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    image_client.models.generate_content,
                    model=image_model,
                    contents="ping",
                    config=_genai_types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=_genai_types.ImageConfig(aspect_ratio="1:1"),
                    ),
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
        payload: Dict[str, Any]
        if hasattr(response, "model_dump"):
            try:
                payload = response.model_dump(mode="json")
            except Exception:  # pragma: no cover - defensive
                payload = {}
        elif isinstance(response, dict):
            payload = dict(response)
        else:
            payload = {}

        image_found = False
        inline_count = 0
        response_id = getattr(response, "response_id", None)
        candidates = payload.get("candidates") or []
        log_method(
            "gemini.health.image payload candidates=%s latency_ms=%s",
            len(candidates) if isinstance(candidates, list) else 0,
            duration_ms,
        )
        headers = _extract_response_headers(response)
        finish_reason = None
        if isinstance(candidates, list) and candidates:
            first_candidate = candidates[0]
            if isinstance(first_candidate, dict):
                finish_reason = first_candidate.get("finishReason") or first_candidate.get(
                    "finish_reason"
                )
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content") or {}
            parts = content.get("parts") if isinstance(content, dict) else []
            if not parts and isinstance(content, list):
                parts = content
            if not parts:
                parts = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                inline_data = part.get("inline_data") or part.get("inlineData")
                if not inline_data and isinstance(part.get("video"), dict):  # pragma: no cover - guard
                    inline_data = part.get("video")
                if isinstance(inline_data, dict):
                    data_field = inline_data.get("data") or inline_data.get("data_base64")
                    if isinstance(data_field, str):
                        try:
                            decoded = base64.b64decode(data_field)
                        except (binascii.Error, ValueError):
                            decoded = b""
                        if decoded:
                            image_found = True
                            inline_count = max(inline_count, 1)
                            break
                    elif isinstance(inline_data.get("data"), (bytes, bytearray, memoryview)):
                        if inline_data.get("data"):
                            image_found = True
                            inline_count = max(inline_count, 1)
                            break
                uri = part.get("file_data") or part.get("fileData")
                if isinstance(uri, dict):
                    file_uri = uri.get("file_uri") or uri.get("fileUri")
                    if file_uri:
                        image_found = True
                        inline_count = max(inline_count, 1)
                        break
                uri_value = part.get("inline_data", {}).get("uri") if isinstance(part.get("inline_data"), dict) else None
                if uri_value:
                    image_found = True
                    inline_count = max(inline_count, 1)
                    break
            if image_found:
                break
        if not image_found:
            log.warning(
                "Gemini image ping NO_IMAGE_NON_CRITICAL model=%s response_id=%s",
                image_model,
                response_id,
            )
            log.warning(
                "health.gemini.no_image %s",
                kv(method="generate_content", finish=finish_reason or ""),
            )
            status_text = "no_image_non_critical"
        else:
            status_text = "ok"
        log.info(
            "health.gemini.ok %s",
            kv(
                model=image_model,
                version=api_version,
                method="generate_content",
                status=status_text,
                finish=finish_reason or "",
                headers=headers,
                latency_ms=duration_ms,
            ),
        )
        log_method(
            "gemini.health.image result model=%s scope=%s found=%s latency_ms=%s api_version=%s",
            image_model,
            image_scope,
            image_found,
            duration_ms,
            api_version,
        )
        image_detail.update(
            {
                "latency_ms": duration_ms,
                "finish_reason": finish_reason or "",
                "inline_found": image_found,
                "inline_count": inline_count,
                "response_id": response_id or "",
                "status": status_text,
            }
        )
        if not image_found:
            image_detail["warning"] = "NO_IMAGE_NON_CRITICAL"

    video_model = (config.gemini_model_video or "").strip()
    if video_model:
        if video_model not in supported:
            return False, f"video model {video_model} missing"
        video_detail = detail.setdefault("video", {"model": video_model})
        if not _supports_method(supported, video_model, "generate_videos"):
            warning_msg = f"video model {video_model} lacks generate_videos"
            warnings.append(warning_msg)
            video_detail["warning"] = "missing generate_videos"
        else:
            video_client = clients.get("media")
            if video_client is None:
                return False, "video client media unavailable"
            video_config = _genai_types.GenerateVideosConfig(duration_seconds=1)
            try:
                start = time.perf_counter()
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

            polls = 0
            refreshed = operation
            if not getattr(operation, "done", False):
                for _ in range(2):
                    polls += 1
                    await asyncio.sleep(random.uniform(2.0, 3.0))
                    try:
                        refreshed = await asyncio.to_thread(video_client.operations.get, operation)
                    except _genai_errors.APIError as exc:
                        detail_msg = f"{getattr(exc, 'code', 0)}:{getattr(exc, 'status', '')}".strip(":")
                        log.warning(
                            "Gemini video poll failed model=%s op=%s", video_model, operation_name, exc_info=True
                        )
                        return False, detail_msg or str(exc)
                    if getattr(refreshed, "done", False):
                        break

            video_detail.update(
                {
                    "operation": getattr(refreshed, "name", None) or operation_name,
                    "polls": polls,
                    "done": bool(getattr(refreshed, "done", False)),
                    "latency_ms": int((time.perf_counter() - start) * 1000),
                }
            )

    if warnings:
        detail["warnings"] = warnings

    detail_json = json.dumps(detail, ensure_ascii=False)
    log.info("Gemini models: ok %s", detail_json)
    if warnings:
        return True, "; ".join(warnings)
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
    add_check("SORA_ENABLED", config.sora_enabled, str(config.sora_enabled).lower())
    if config.sora_enabled:
        add_check("SORA_MODEL_VIDEO", bool(config.sora_model_video), config.sora_model_video)
    else:
        add_check("SORA_MODEL_VIDEO", True, "disabled")
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
