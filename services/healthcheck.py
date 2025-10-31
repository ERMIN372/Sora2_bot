"""Provider availability probes executed during application startup."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import aiohttp

from config import Config, CFG
from observability import HealthCheckResult, record_healthcheck
from services.gemini_client import get_media_client

import logging_providers  # noqa: F401  # Ensure logging.providers shim is registered.
from logging.providers import log_error

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ProbeResult:
    """Outcome of a single provider availability probe."""

    name: str
    ok: bool
    enabled: bool
    detail: str = ""
    error: Optional[str] = None
    latency_ms: Optional[int] = None


SORA_ENABLED: bool = False
VEO_ENABLED: bool = False
GEMINI_IMAGE_ENABLED: bool = False
OPENAI_CHAT_ENABLED: bool = False
OPENAI_IMAGE_ENABLED: bool = False
PROBE_RESULTS: Dict[str, ProbeResult] = {}
PROBE_REFRESH_INTERVAL_SECONDS: float = 15 * 60


def _client_timeout(config: Config) -> aiohttp.ClientTimeout:
    total = float(config.request_timeout or 20.0)
    connect = float(config.request_connect_timeout or 10.0)
    read = float(config.request_read_timeout or 20.0)
    return aiohttp.ClientTimeout(total=total, connect=connect, sock_read=read)


def _normalise_models(raw: Sequence[object]) -> List[str]:
    models: List[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            candidate = item.get("id") or item.get("name")
            if isinstance(candidate, str) and candidate:
                models.append(candidate.strip())
    return models


async def _probe_openai_models(
    *,
    config: Config,
    probe_name: str,
    enabled: bool,
    required_models: Sequence[str],
    headers: Optional[Mapping[str, str]] = None,
) -> ProbeResult:
    if not enabled:
        return ProbeResult(name=probe_name, ok=True, enabled=False, detail="disabled")

    api_key = (config.openai_key or "").strip()
    if not api_key:
        return ProbeResult(name=probe_name, ok=False, enabled=False, detail="missing api key")

    base_url = (config.openai_api_base or "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/models"
    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if headers:
        request_headers.update(headers)
    timeout = _client_timeout(config)
    start = time.perf_counter()
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=request_headers, params={"limit": 200}) as response:
                body_text = await response.text()
                latency = int((time.perf_counter() - start) * 1000)
                if response.status >= 400:
                    error_msg = f"http {response.status}"
                    detail = body_text.strip()[:200]
                    return ProbeResult(
                        name=probe_name,
                        ok=False,
                        enabled=True,
                        error=error_msg,
                        detail=detail,
                        latency_ms=latency,
                    )
                try:
                    payload = json.loads(body_text) if body_text else {}
                except json.JSONDecodeError:
                    payload = {}
                entries = payload.get("data") if isinstance(payload, Mapping) else None
                models = _normalise_models(entries) if isinstance(entries, Sequence) else []
                missing = [model for model in required_models if model and model not in models]
                ok = not missing
                detail_items: List[str] = []
                if models:
                    detail_items.append(f"models={','.join(models[:5])}")
                if missing:
                    detail_items.append(f"missing={','.join(missing)}")
                detail = " ".join(detail_items)
                return ProbeResult(
                    name=probe_name,
                    ok=ok,
                    enabled=True,
                    detail=detail,
                    error=(f"missing:{','.join(missing)}" if missing else None),
                    latency_ms=latency,
                )
    except Exception as exc:  # pragma: no cover - network failure
        log.warning("OpenAI probe failed provider=%s", probe_name, exc_info=True)
        return ProbeResult(
            name=probe_name,
            ok=False,
            enabled=True,
            error=str(exc),
        )


async def probe_sora(config: Config) -> ProbeResult:
    required = []
    model = (config.sora_model_video or "").strip()
    if model:
        required.append(model)
    required.append("sora")
    return await _probe_openai_models(
        config=config,
        probe_name="SORA",
        enabled=config.sora_video_enabled,
        required_models=required,
        headers={"OpenAI-Beta": "assistants=v2"},
    )


async def probe_openai_chat(config: Config) -> ProbeResult:
    return await _probe_openai_models(
        config=config,
        probe_name="OPENAI_CHAT",
        enabled=config.sora_enabled,
        required_models=(),
    )


async def probe_sdxl_or_dalle(config: Config) -> ProbeResult:
    fallback_models = [
        model.strip()
        for model in config.fallback_image_models
        if isinstance(model, str) and model.strip()
    ]
    openai_models = [
        model for model in fallback_models if "dall" in model.lower() or "gpt" in model.lower()
    ]
    enabled = config.openai_image_enabled and bool(openai_models)
    return await _probe_openai_models(
        config=config,
        probe_name="OPENAI_IMAGE",
        enabled=enabled,
        required_models=openai_models,
        headers={"OpenAI-Beta": "assistants=v2"},
    )


async def probe_veo(config: Config) -> ProbeResult:
    enabled = config.gemini_video_enabled
    if not enabled:
        return ProbeResult(name="VEO", ok=True, enabled=False, detail="disabled")
    model = (config.gemini_model_video or "").strip()
    if not model:
        return ProbeResult(name="VEO", ok=False, enabled=True, error="missing model")
    client = get_media_client(config)
    start = time.perf_counter()
    try:
        result = await asyncio.to_thread(client.models.get, model=model)
    except Exception as exc:  # pragma: no cover - network failure
        log.warning("Gemini video probe failed model=%s", model, exc_info=True)
        return ProbeResult(name="VEO", ok=False, enabled=True, error=str(exc))
    latency = int((time.perf_counter() - start) * 1000)
    detail = getattr(result, "name", None) or getattr(result, "model", None) or model
    return ProbeResult(name="VEO", ok=True, enabled=True, detail=str(detail), latency_ms=latency)


async def probe_gemini_image(config: Config) -> ProbeResult:
    enabled = config.gemini_enabled
    if not enabled:
        return ProbeResult(name="GEMINI_IMAGE", ok=True, enabled=False, detail="disabled")
    model = (config.gemini_model_image or "").strip()
    if not model:
        return ProbeResult(
            name="GEMINI_IMAGE",
            ok=False,
            enabled=True,
            error="missing model",
        )
    client = get_media_client(config)
    start = time.perf_counter()
    try:
        result = await asyncio.to_thread(client.models.get, model=model)
    except Exception as exc:  # pragma: no cover - network failure
        log.warning("Gemini image probe failed model=%s", model, exc_info=True)
        return ProbeResult(name="GEMINI_IMAGE", ok=False, enabled=True, error=str(exc))
    latency = int((time.perf_counter() - start) * 1000)
    detail = getattr(result, "name", None) or getattr(result, "model", None) or model
    return ProbeResult(
        name="GEMINI_IMAGE",
        ok=True,
        enabled=True,
        detail=str(detail),
        latency_ms=latency,
    )


def _update_flags(results: Iterable[ProbeResult]) -> None:
    global SORA_ENABLED, VEO_ENABLED, GEMINI_IMAGE_ENABLED, OPENAI_CHAT_ENABLED, OPENAI_IMAGE_ENABLED
    PROBE_RESULTS.clear()
    for result in results:
        PROBE_RESULTS[result.name] = result
    SORA_ENABLED = bool(PROBE_RESULTS.get("SORA") and PROBE_RESULTS["SORA"].enabled and PROBE_RESULTS["SORA"].ok)
    VEO_ENABLED = bool(PROBE_RESULTS.get("VEO") and PROBE_RESULTS["VEO"].enabled and PROBE_RESULTS["VEO"].ok)
    GEMINI_IMAGE_ENABLED = bool(
        PROBE_RESULTS.get("GEMINI_IMAGE")
        and PROBE_RESULTS["GEMINI_IMAGE"].enabled
        and PROBE_RESULTS["GEMINI_IMAGE"].ok
    )
    OPENAI_CHAT_ENABLED = bool(
        PROBE_RESULTS.get("OPENAI_CHAT")
        and PROBE_RESULTS["OPENAI_CHAT"].enabled
        and PROBE_RESULTS["OPENAI_CHAT"].ok
    )
    OPENAI_IMAGE_ENABLED = bool(
        PROBE_RESULTS.get("OPENAI_IMAGE")
        and PROBE_RESULTS["OPENAI_IMAGE"].enabled
        and PROBE_RESULTS["OPENAI_IMAGE"].ok
    )


def provider_available(name: str) -> bool:
    result = PROBE_RESULTS.get(name.upper())
    return bool(result and result.enabled and result.ok)


def _probe_label(func) -> str:
    name = getattr(func, "__name__", "").lower()
    if name.startswith("probe_"):
        name = name[6:]
    return name.upper() or "PROBE"


async def run_all_probes(*, config: Config, mode: Optional[str] = None) -> List[ProbeResult]:
    probes = [
        probe_sora,
        probe_veo,
        probe_gemini_image,
        probe_openai_chat,
        probe_sdxl_or_dalle,
    ]
    results: List[ProbeResult] = []
    for probe in probes:
        label = _probe_label(probe)
        try:
            result = await probe(config)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - defensive catch
            log.exception("Probe crashed name=%s", label)
            result = ProbeResult(name=label, ok=False, enabled=True, error=str(exc))
        results.append(result)
        if result.enabled and not result.ok:
            log_error(
                provider=result.name.lower(),
                message=result.error or "probe failed",
                detail={"detail": result.detail or "", "enabled": result.enabled},
            )
    _update_flags(results)

    checks: List[Dict[str, object]] = []
    errors: List[str] = []
    for result in results:
        ok = result.ok if result.enabled else True
        entry: Dict[str, object] = {
            "label": result.name,
            "ok": ok,
        }
        if result.detail:
            entry["detail"] = result.detail
        if result.latency_ms is not None:
            entry["latency_ms"] = result.latency_ms
        if result.enabled and not result.ok:
            errors.append(f"{result.name}: {result.error or result.detail or 'unavailable'}")
        checks.append(entry)

    health = HealthCheckResult(
        ts=time.time(),
        mode=mode or CFG.BOT_MODE,
        ok=all(entry["ok"] for entry in checks),
        version=config.bot_version,
        details={"mode": mode or CFG.BOT_MODE, "checks": checks},
        errors=errors,
    )
    record_healthcheck(health)
    log.info("startup_probes %s", json.dumps(health.details, ensure_ascii=False))
    if errors:
        log.info("probe_errors=%s", "; ".join(errors))
    return results


async def schedule_probe_refresh(
    *, config: Config, interval_seconds: float = PROBE_REFRESH_INTERVAL_SECONDS, mode: Optional[str] = None
) -> None:
    if interval_seconds <= 0:
        return
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await run_all_probes(config=config, mode=mode)
        except Exception:  # pragma: no cover - defensive
            log.exception("Scheduled probes failed")


__all__ = [
    "ProbeResult",
    "SORA_ENABLED",
    "VEO_ENABLED",
    "GEMINI_IMAGE_ENABLED",
    "OPENAI_CHAT_ENABLED",
    "OPENAI_IMAGE_ENABLED",
    "PROBE_RESULTS",
    "PROBE_REFRESH_INTERVAL_SECONDS",
    "probe_sora",
    "probe_veo",
    "probe_gemini_image",
    "probe_openai_chat",
    "probe_sdxl_or_dalle",
    "run_all_probes",
    "provider_available",
    "schedule_probe_refresh",
]
