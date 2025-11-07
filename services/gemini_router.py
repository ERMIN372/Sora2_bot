"""High-level routing helpers for Gemini multimodal generation."""
from __future__ import annotations

import logging
import asyncio
import re
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from google.genai import Client as _GenAIClient
from google.genai import errors as genai_errors

from config import Config, DEBUG_GEMINI, GEMINI_TRACE_HEADERS, RoutingCfg, load_config
from services.gemini_client import get_gemini_client
from providers.logx import kv

log = logging.getLogger(__name__)

_TASK_METHOD: Mapping[str, str] = {
    "text": "generate_content",
    "image": "generate_content",
    "video": "generate_videos",
}

_METHOD_HUMAN: Mapping[str, str] = {
    "generate_content": "content",
    "generate_images": "image",
    "generate_videos": "video",
}

_BRAND_DESCRIPTIONS: Mapping[str, str] = {
    "apple": "sleek, minimalist high-end consumer electronics aesthetic (no logos)",
    "nike": "dynamic athletic sportswear styling with bold swooping shapes (no brand marks)",
    "adidas": "modern athletic wear with triple-line inspired motifs (no branding)",
    "tesla": "futuristic electric car design language with aerodynamic curves (no badges)",
    "disney": "whimsical fairytale-inspired family-friendly atmosphere (no franchise names)",
    "marvel": "cinematic superhero action aesthetic with dramatic lighting (no franchise names)",
    "pixar": "family-friendly 3D animation style with expressive characters (no studio references)",
    "star wars": "epic sci-fi space opera vibe with glowing energy weapons (no franchise titles)",
    "harry potter": "fantastical wizarding world mood with gothic castles (no series titles)",
    "barbie": "playful pink-inspired fashion aesthetic with glossy materials (no trademark text)",
    "gucci": "luxury high-fashion styling with opulent fabrics and gold accents (no logos)",
    "prada": "minimalist high-fashion editorial look with sharp tailoring (no labels)",
    "ferrari": "high-performance sports car silhouette in racing red (no emblems)",
}


class GeminiRoutingError(RuntimeError):
    """Raised when a prompt cannot be routed to a valid Gemini API method."""


@dataclass(frozen=True)
class RouteDecision:
    """Resolved information about how to execute a Gemini request."""

    task: str
    model: str
    prompt: str
    method: str
    api_version: str
    client: _GenAIClient
    supported_methods: Tuple[str, ...]
    rewritten: bool = False
    rewrite_notes: Optional[str] = None


def _normalise_task(task: str) -> str:
    return (task or "text").strip().lower()


def _strip_model_prefix(model: str) -> str:
    if not model:
        return model
    if "/" in model:
        return model.split("/")[-1]
    return model


def _normalise_method(method: str) -> str:
    if not method:
        return ""
    text = str(method)
    if text.startswith("generate") and text[8:9].isupper():
        # camelCase from the public API
        text = re.sub(r"([A-Z])", r"_\1", text).lower()
    return text.strip().lower()


def _normalise_supported(methods: Iterable[str]) -> Tuple[str, ...]:
    seen: List[str] = []
    for method in methods:
        normalised = _normalise_method(method)
        if not normalised:
            continue
        if normalised not in seen:
            seen.append(normalised)
    return tuple(seen)


def _preferred_api_version(task: str, model: str) -> str:
    if task == "text":
        return RoutingCfg.TEXT_API_VERSION
    default = RoutingCfg.MEDIA_API_VERSION
    if task != "image":
        return default
    model_name = (model or "").strip().lower()
    if model_name.endswith("-image"):
        return "v1"
    return default


def _rewrite_prompt(prompt: str) -> Tuple[str, bool, Optional[str]]:
    if not prompt:
        return prompt, False, None
    rewritten = prompt
    hits: List[str] = []
    for raw_term, description in _BRAND_DESCRIPTIONS.items():
        pattern = re.compile(re.escape(raw_term), re.IGNORECASE)
        if pattern.search(rewritten):
            rewritten = pattern.sub(description, rewritten)
            hits.append(raw_term)
    if not hits:
        return prompt, False, None
    note = (
        "Брендовые или франшизные упоминания заменены описанием: "
        + ", ".join(sorted(set(hits)))
        + ". Логотип можно добавить постфактум на сервере."
    )
    return rewritten, True, note


class GeminiRouter:
    """Resolve which Gemini API variant should be used for a task."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._catalog_lock = threading.Lock()
        self._loaded_versions: Set[str] = set()
        self._supported_methods: Dict[str, Tuple[str, ...]] = {}

    @property
    def supported_models(self) -> Mapping[str, Tuple[str, ...]]:
        return dict(self._supported_methods)

    def _load_catalog(self, client: _GenAIClient, cache_key: str) -> None:
        if cache_key in self._loaded_versions:
            return
        with self._catalog_lock:
            if cache_key in self._loaded_versions:
                return
            try:
                models = list(client.models.list())
            except Exception as exc:  # pragma: no cover - network guard
                log.warning(
                    "Failed to fetch Gemini models list scope=%s error=%s",
                    cache_key,
                    exc,
                    exc_info=True,
                )
                self._loaded_versions.add(cache_key)
                return
            for model in models:
                name = getattr(model, "name", None) or getattr(model, "model", None)
                if not isinstance(name, str):
                    continue
                methods = getattr(model, "supported_generation_methods", None) or []
                normalised = _normalise_supported(methods)
                if not normalised:
                    continue
                short_name = _strip_model_prefix(name)
                self._supported_methods[short_name] = normalised
                self._supported_methods[name] = normalised
            self._loaded_versions.add(cache_key)

    def _default_model_for_task(self, task: str) -> str:
        if task == "image":
            return self._config.gemini_model_image
        if task == "video":
            return self._config.gemini_model_video
        return self._config.gemini_model_text

    def _validate_task_model(self, task: str, model: str) -> None:
        short_name = _strip_model_prefix(model)
        if not short_name:
            raise GeminiRoutingError("Не указана модель Gemini для запроса")
        lower = short_name.lower()
        if lower.startswith("veo-"):
            if task != "video":
                raise GeminiRoutingError(
                    "Эта модель предназначена для видео. Выбрана video-модель, но задача не video."
                )
            return
        if lower.endswith("-image"):
            if task != "image":
                raise GeminiRoutingError(
                    "Эта модель не поддерживает данный метод. Выбрана image-модель, но вы вызвали text. Поменяй модель или метод."
                )
            return
        if lower.startswith("gemini-2") and task != "text":
            raise GeminiRoutingError(
                "Эта модель работает через generate_content. Используй её только для text-задач."
            )
        if task == "image":
            raise GeminiRoutingError(
                "Для изображений нужна модель семейства gemini-*-image. Поменяй модель на image-вариант."
            )
        if task == "video":
            raise GeminiRoutingError(
                "Для видео используй модель семейства veo-3.x-*."
            )

    def _resolve_supported_methods(self, model: str) -> Tuple[str, ...]:
        if model in self._supported_methods:
            return self._supported_methods[model]
        short = _strip_model_prefix(model)
        return self._supported_methods.get(short, ())

    def _ensure_supported(self, model: str, method: str, task: str) -> None:
        supported = self._resolve_supported_methods(model)
        if not supported:
            return
        if method not in supported:
            if method == "generate_content" and task == "image":
                if "generate_images" in supported:
                    return
            human_methods = [
                _METHOD_HUMAN.get(item, item.replace("generate_", ""))
                for item in supported
            ]
            raise GeminiRoutingError(
                "Эта модель не поддерживает данный метод. Она умеет: "
                + ", ".join(human_methods)
                + "."
            )

    def route(
        self,
        *,
        task: str,
        model: Optional[str],
        prompt: str,
        assets: Optional[Mapping[str, object]] = None,
        forced_api_version: Optional[str] = None,
    ) -> RouteDecision:
        normalised_task = _normalise_task(task)
        if normalised_task not in _TASK_METHOD:
            raise GeminiRoutingError(f"Неизвестная задача {task!r}")
        selected_model = (model or self._default_model_for_task(normalised_task)).strip()
        self._validate_task_model(normalised_task, selected_model)
        method = _TASK_METHOD[normalised_task]
        supported_methods = self._resolve_supported_methods(selected_model)
        if normalised_task == "image" and method not in supported_methods:
            if "generate_content" in supported_methods:
                method = "generate_content"
            elif "generate_images" in supported_methods:
                method = "generate_images"
        if forced_api_version:
            api_version = forced_api_version.strip() or _preferred_api_version(
                normalised_task, selected_model
            )
        else:
            api_version = _preferred_api_version(normalised_task, selected_model)
        client = get_gemini_client(self._config, api_version=api_version)
        cache_key = f"{api_version}:{normalised_task}"
        self._load_catalog(client, cache_key)
        self._ensure_supported(selected_model, method, normalised_task)

        processed_prompt = prompt
        rewritten = False
        notes: Optional[str] = None
        if normalised_task in {"image", "video"}:
            processed_prompt, rewritten, notes = _rewrite_prompt(prompt)
            if rewritten:
                log.info(
                    "gemini.router.prompt_rewrite task=%s model=%s note=%s",
                    normalised_task,
                    selected_model,
                    notes,
                )

        return RouteDecision(
            task=normalised_task,
            model=selected_model,
            prompt=processed_prompt,
            method=method,
            api_version=api_version,
            client=client,
            supported_methods=supported_methods,
            rewritten=rewritten,
            rewrite_notes=notes,
        )


_ROUTER_CACHE: Dict[int, GeminiRouter] = {}
_CACHE_LOCK = threading.Lock()
_LAST_CONFIG: Optional[Config] = None


def get_gemini_router(config: Config) -> GeminiRouter:
    """Return a cached :class:`GeminiRouter` bound to *config*."""

    key = id(config)
    with _CACHE_LOCK:
        router = _ROUTER_CACHE.get(key)
        if router is None:
            router = GeminiRouter(config)
            _ROUTER_CACHE[key] = router
        global _LAST_CONFIG
        _LAST_CONFIG = config
    return router


def _diag_extract_headers(response: Any) -> Dict[str, str]:
    sdk_response = getattr(response, "sdk_http_response", None)
    headers = getattr(sdk_response, "headers", None)
    if headers is None:
        return {}
    if isinstance(headers, Mapping):
        raw = {str(key): headers[key] for key in headers}
    elif hasattr(headers, "items"):
        try:
            raw = {str(key): value for key, value in headers.items()}
        except Exception:  # pragma: no cover - defensive
            raw = {}
    else:
        raw = {}
    normalised = {str(key).lower(): str(value) for key, value in raw.items()}
    keep = {
        *(h.strip().lower() for h in GEMINI_TRACE_HEADERS if h.strip()),
        "x-request-id",
        "date",
        "server",
        "content-type",
    }
    return {key: normalised[key] for key in normalised if key in keep}


def _count_inline_parts(payload: Mapping[str, Any]) -> int:
    count = 0
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            content = candidate.get("content")
            parts: List[Any] = []
            if isinstance(content, Mapping):
                raw_parts = content.get("parts")
                if isinstance(raw_parts, list):
                    parts = raw_parts
            elif isinstance(content, list):
                parts = content
            for part in parts:
                if not isinstance(part, Mapping):
                    continue
                inline_data = part.get("inline_data") or part.get("inlineData")
                file_data = part.get("file_data") or part.get("fileData")
                if isinstance(inline_data, Mapping):
                    if any(
                        inline_data.get(key)
                        for key in ("data", "data_base64", "dataBase64", "uri", "download_uri")
                    ):
                        count += 1
                        continue
                elif inline_data:
                    count += 1
                    continue
                if isinstance(file_data, Mapping) and (
                    file_data.get("file_uri") or file_data.get("fileUri")
                ):
                    count += 1
    return count


def _extract_finish_reason(payload: Mapping[str, Any], response: Any) -> Optional[str]:
    finish_reason = payload.get("finish_reason") or payload.get("finishReason")
    if finish_reason:
        return str(finish_reason)
    candidates = payload.get("candidates")
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, Mapping):
            finish = first.get("finishReason") or first.get("finish_reason")
            if finish:
                return str(finish)
    if getattr(response, "candidates", None):
        primary = response.candidates[0]
        finish = getattr(primary, "finish_reason", None)
        if finish:
            return str(finish)
    return None


def _extract_safety_summary(payload: Mapping[str, Any]) -> Optional[str]:
    feedback = payload.get("prompt_feedback") or payload.get("promptFeedback")
    if not isinstance(feedback, Mapping):
        return None
    ratings = feedback.get("safety_ratings") or feedback.get("safetyRatings")
    if isinstance(ratings, list):
        categories: List[str] = []
        for rating in ratings:
            if not isinstance(rating, Mapping):
                continue
            category = rating.get("category") or rating.get("category_name")
            if isinstance(category, Mapping):
                category = category.get("name") or category.get("value")
            if category:
                categories.append(str(category))
        if categories:
            return ",".join(sorted({entry for entry in categories if entry}))
        return f"len={len(ratings)}"
    if ratings is not None:
        return str(ratings)
    return None


def _count_images(method: str, response: Any, payload: Mapping[str, Any]) -> int:
    if method == "generate_images":
        images = getattr(response, "images", None)
        if isinstance(images, list):
            return len(images)
        generated = payload.get("generated_images") or payload.get("generatedImages")
        if isinstance(generated, list):
            return len(generated)
        payload_images = payload.get("images")
        if isinstance(payload_images, list):
            return len(payload_images)
        return 0
    return _count_inline_parts(payload)


def _extract_response_id(response: Any, payload: Mapping[str, Any]) -> Optional[str]:
    response_id = getattr(response, "response_id", None)
    if response_id:
        return str(response_id)
    rid = payload.get("response_id") or payload.get("responseId")
    return str(rid) if rid else None


async def _diagnose_call(
    *,
    router: GeminiRouter,
    model: str,
    prompt: str,
    initial_method: str,
) -> Dict[str, Any]:
    attempt_method = initial_method
    forced_version: Optional[str] = None
    version_switch = False
    fallback_used = False
    attempted: Set[Tuple[str, str]] = set()
    max_attempts = 5
    while max_attempts > 0:
        max_attempts -= 1
        decision = router.route(
            task="image",
            model=model,
            prompt=prompt,
            assets={},
            forced_api_version=forced_version,
        )
        version_label = (decision.api_version or "").lower()
        key = (attempt_method, version_label)
        if key in attempted and not fallback_used:
            break
        attempted.add(key)
        generator = getattr(decision.client.models, attempt_method, None)
        if generator is None:
            return {
                "status": "error",
                "method_requested": initial_method,
                "method_used": attempt_method,
                "api_version": decision.api_version,
                "error": f"method {attempt_method} unavailable",
                "version_switch": version_switch,
                "fallback_used": fallback_used,
            }
        if attempt_method == "generate_images":
            request_kwargs: Dict[str, Any] = {
                "model": decision.model,
                "prompt": prompt,
            }
        else:
            request_kwargs = {
                "model": decision.model,
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt,
                            }
                        ],
                    }
                ],
            }
        start = time.perf_counter()
        try:
            response = await asyncio.to_thread(generator, **request_kwargs)
        except genai_errors.APIError as exc:
            status_code = int(getattr(exc, "code", 0) or getattr(exc, "status", 0) or 0)
            message = getattr(exc, "message", "") or str(exc)
            alt_version = None
            if status_code in {400, 404}:
                alt_version = "v1beta" if version_label == "v1" else "v1"
            if alt_version and not version_switch and (attempt_method, alt_version) not in attempted:
                version_switch = True
                forced_version = alt_version
                continue
            if (
                attempt_method == "generate_images"
                and not fallback_used
                and status_code in {400, 404}
            ):
                fallback_used = True
                attempt_method = "generate_content"
                forced_version = None
                attempted.clear()
                continue
            duration_ms = int((time.perf_counter() - start) * 1000)
            return {
                "status": "error",
                "method_requested": initial_method,
                "method_used": attempt_method,
                "api_version": decision.api_version,
                "status_code": status_code,
                "error": message,
                "duration_ms": duration_ms,
                "version_switch": version_switch,
                "fallback_used": fallback_used,
            }
        except Exception as exc:  # pragma: no cover - defensive
            duration_ms = int((time.perf_counter() - start) * 1000)
            return {
                "status": "error",
                "method_requested": initial_method,
                "method_used": attempt_method,
                "api_version": decision.api_version,
                "error": str(exc),
                "duration_ms": duration_ms,
                "version_switch": version_switch,
                "fallback_used": fallback_used,
            }
        duration_ms = int((time.perf_counter() - start) * 1000)
        if hasattr(response, "model_dump"):
            try:
                payload: Mapping[str, Any] = response.model_dump(mode="json")  # type: ignore[assignment]
            except Exception:  # pragma: no cover - defensive serialisation
                payload = {}
        elif isinstance(response, Mapping):
            payload = response
        else:
            payload = {}
        headers = _diag_extract_headers(response)
        finish_reason = _extract_finish_reason(payload, response)
        images_count = _count_images(attempt_method, response, payload)
        inline_parts = _count_inline_parts(payload)
        safety_summary = _extract_safety_summary(payload)
        response_id = _extract_response_id(response, payload)
        candidates = payload.get("candidates")
        candidates_count = len(candidates) if isinstance(candidates, list) else 0
        if attempt_method == "generate_images" and images_count == 0 and not fallback_used:
            fallback_used = True
            attempt_method = "generate_content"
            forced_version = None
            attempted.clear()
            continue
        status = "ok" if images_count or attempt_method == "generate_content" else "empty"
        summary = {
            "status": status,
            "method_requested": initial_method,
            "method_used": attempt_method,
            "api_version": decision.api_version,
            "duration_ms": duration_ms,
            "finish_reason": finish_reason,
            "images_count": images_count,
            "inline_parts": inline_parts,
            "candidates": candidates_count,
            "safety": safety_summary,
            "version_switch": version_switch,
            "fallback_used": fallback_used,
            "response_id": response_id,
            "headers": headers,
        }
        return summary
    return {
        "status": "error",
        "method_requested": initial_method,
        "method_used": attempt_method,
        "api_version": forced_version,
        "error": "diagnostic attempts exhausted",
        "version_switch": version_switch,
        "fallback_used": fallback_used,
    }


async def run_diag(
    config: Optional[Config] = None,
    *,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    attempts: int = 1,
) -> Dict[str, Any]:
    active_config = config or _LAST_CONFIG or load_config()
    if not active_config.gemini_enabled or not active_config.gemini_api_key:
        return {
            "debug": bool(getattr(active_config, "debug_gemini", False)),
            "enabled": False,
            "reason": "gemini_disabled",
        }
    router = get_gemini_router(active_config)
    target_model = (model or active_config.gemini_model_image or "").strip()
    diag_prompt = prompt or "Gemini diagnostic ping"
    try:
        base_decision = router.route(
            task="image",
            model=target_model,
            prompt=diag_prompt,
            assets={},
        )
    except GeminiRoutingError as exc:
        log.warning("gemini.diag.route_failed %s", kv(error=str(exc), model=target_model))
        return {
            "debug": bool(getattr(active_config, "debug_gemini", False)),
            "enabled": True,
            "error": str(exc),
        }
    attempts = max(1, int(attempts or 1))
    methods = ("generate_images", "generate_content")
    calls: Dict[str, List[Dict[str, Any]]] = {method: [] for method in methods}
    for attempt in range(attempts):
        for method in methods:
            result = await _diagnose_call(
                router=router,
                model=target_model,
                prompt=diag_prompt,
                initial_method=method,
            )
            result["attempt"] = attempt + 1
            calls[method].append(result)

    summary: Dict[str, Any] = {}

    def _summarise(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not entries:
            return {"attempts": 0}
        latencies = [
            entry.get("duration_ms")
            for entry in entries
            if isinstance(entry.get("duration_ms"), (int, float))
        ]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        images_ok = sum(
            1
            for entry in entries
            if entry.get("images_count", 0)
            and entry.get("status") == "ok"
        )
        no_image = sum(1 for entry in entries if entry.get("images_count", 0) == 0)
        errors = [entry for entry in entries if entry.get("status") == "error"]
        versions = sorted(
            {
                str(entry.get("api_version"))
                for entry in entries
                if entry.get("api_version")
            }
        )
        headers = entries[-1].get("headers") if entries else {}
        finishes = [
            entry.get("finish_reason")
            for entry in entries
            if entry.get("finish_reason")
        ]
        safety_flags = [
            entry.get("safety")
            for entry in entries
            if entry.get("safety")
        ]
        return {
            "attempts": len(entries),
            "ok": images_ok,
            "no_image": no_image,
            "errors": errors,
            "avg_latency_ms": round(avg_latency, 2),
            "latencies_ms": latencies,
            "api_versions": versions,
            "headers": headers,
            "finish_reasons": finishes,
            "safety": safety_flags,
        }

    for method, entries in calls.items():
        summary[method] = _summarise(entries)

    latest_diag_path: Optional[str] = None
    diag_dir = Path("tmp") / "gemini_diag"
    try:
        if diag_dir.exists():
            latest = max(
                (path for path in diag_dir.glob("*.json") if path.is_file()),
                key=lambda item: item.stat().st_mtime,
            )
            latest_diag_path = str(latest)
    except ValueError:
        latest_diag_path = None
    except Exception as exc:  # pragma: no cover - defensive filesystem guard
        log.debug("gemini.diag.latest_failed %s", kv(error=str(exc)))
    report = {
        "debug": bool(getattr(active_config, "debug_gemini", False)),
        "enabled": True,
        "model": target_model,
        "prompt": diag_prompt,
        "default_method": base_decision.method,
        "default_version": base_decision.api_version,
        "calls": calls,
        "summary": summary,
        "attempts": attempts,
        "latest_diag": latest_diag_path,
    }
    if DEBUG_GEMINI:
        log.info("gemini.diag.report %s", kv(model=target_model, prompt=diag_prompt))
    return report


__all__ = [
    "GeminiRouter",
    "GeminiRoutingError",
    "RouteDecision",
    "get_gemini_router",
    "run_diag",
]
