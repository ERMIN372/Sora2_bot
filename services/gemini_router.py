"""High-level routing helpers for Gemini multimodal generation."""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

from google.genai import Client as _GenAIClient

from config import Config
from services.gemini_client import get_media_client, get_text_client

log = logging.getLogger(__name__)

_TASK_METHOD: Mapping[str, str] = {
    "text": "generate_content",
    "image": "generate_images",
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
    ) -> RouteDecision:
        normalised_task = _normalise_task(task)
        if normalised_task not in _TASK_METHOD:
            raise GeminiRoutingError(f"Неизвестная задача {task!r}")
        selected_model = (model or self._default_model_for_task(normalised_task)).strip()
        self._validate_task_model(normalised_task, selected_model)
        method = _TASK_METHOD[normalised_task]
        supported_methods = self._resolve_supported_methods(selected_model)
        if normalised_task == "image" and method not in supported_methods:
            if "generate_images" in supported_methods:
                method = "generate_images"
            elif "generate_content" in supported_methods:
                method = "generate_content"
        if normalised_task == "text":
            api_version = "v1"
            client = get_text_client(self._config)
            cache_key = "text"
        else:
            api_version = "v1beta"
            client = get_media_client(self._config)
            cache_key = "media"
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


def get_gemini_router(config: Config) -> GeminiRouter:
    """Return a cached :class:`GeminiRouter` bound to *config*."""

    key = id(config)
    with _CACHE_LOCK:
        router = _ROUTER_CACHE.get(key)
        if router is None:
            router = GeminiRouter(config)
            _ROUTER_CACHE[key] = router
    return router


__all__ = [
    "GeminiRouter",
    "GeminiRoutingError",
    "RouteDecision",
    "get_gemini_router",
]
