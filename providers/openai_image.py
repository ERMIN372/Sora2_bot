"""OpenAI Images API client for fallback generation."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import httpx

from config import Config, DEFAULT_OPENAI_BETA_HEADER
from providers.base import (
    BaseProviderClient,
    ProviderAPIError,
    ProviderJobStatus,
    ProviderJobSubmission,
    _extract_request_id,
    _mask_secret,
    _sanitize_headers,
    _serialise_for_log,
    _truncate,
)

log = logging.getLogger(__name__)


PROVIDER_IMAGE_TIMEOUT_CONNECT = 10.0
PROVIDER_IMAGE_TIMEOUT_READ = 120.0
PROVIDER_IMAGE_TIMEOUT_WRITE = 120.0
PROVIDER_IMAGE_TIMEOUT_POOL = 60.0
PROVIDER_IMAGE_MAX_CONCURRENCY = 3
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504, 524}


class OpenAIImageClient(BaseProviderClient):
    """Minimal client for OpenAI image generation used as a fallback."""

    def __init__(self, *, config: Config) -> None:
        api_key = (config.openai_key or "").strip()
        if not api_key:
            raise RuntimeError("OpenAI API key is required for image generation")
        base_url = (config.openai_api_base or "https://api.openai.com").rstrip("/")
        beta_header = (config.openai_beta_header or "assistants=v2").strip()
        if beta_header == DEFAULT_OPENAI_BETA_HEADER:
            beta_header = "assistants=v2"
        default_headers: Dict[str, str] = {}
        if beta_header:
            default_headers["OpenAI-Beta"] = beta_header
        if config.openai_org_id:
            default_headers["OpenAI-Organization"] = config.openai_org_id
        super().__init__(
            config=config,
            base_url=base_url,
            api_key=api_key,
            provider_name="openai-image",
            default_headers=default_headers,
        )
        log.debug(
            "OpenAIImageClient configured base_url=%s beta=%s org_id=%s api_key=%s",
            base_url,
            beta_header or "default",
            _mask_secret(config.openai_org_id),
            _mask_secret(api_key),
        )
        fallback_models = [
            model.strip().lower()
            for model in config.fallback_image_models
            if isinstance(model, str) and model.strip()
        ]
        self._default_model = fallback_models[0] if fallback_models else "gpt-image-1"
        self._image_read_timeout = max(
            1.0, float(config.image_read_timeout_sec or PROVIDER_IMAGE_TIMEOUT_READ)
        )
        self._image_max_retries = max(1, int(config.image_max_retries or 1))
        self._http_client_lock = asyncio.Lock()
        self._http_client: Optional[httpx.AsyncClient] = None
        self._request_gate = asyncio.Semaphore(PROVIDER_IMAGE_MAX_CONCURRENCY)
        log.debug(
            "OpenAIImageClient timeouts connect=%s read=%s write=%s pool=%s max_retries=%s concurrency=%s",
            PROVIDER_IMAGE_TIMEOUT_CONNECT,
            self._image_read_timeout,
            PROVIDER_IMAGE_TIMEOUT_WRITE,
            PROVIDER_IMAGE_TIMEOUT_POOL,
            self._image_max_retries,
            PROVIDER_IMAGE_MAX_CONCURRENCY,
        )
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._pending_dir = Path("attached_assets") / "pending" / self.provider_name
        try:
            self._pending_dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - filesystem guard
            log.warning(
                "Failed to initialise pending cache directory dir=%s", self._pending_dir,
                exc_info=True,
            )

    async def _ensure_http_client(self) -> httpx.AsyncClient:
        async with self._http_client_lock:
            if self._http_client is None or self._http_client.is_closed:
                timeout = httpx.Timeout(
                    connect=PROVIDER_IMAGE_TIMEOUT_CONNECT,
                    read=self._image_read_timeout,
                    write=PROVIDER_IMAGE_TIMEOUT_WRITE,
                    pool=PROVIDER_IMAGE_TIMEOUT_POOL,
                )
                self._http_client = httpx.AsyncClient(timeout=timeout)
        return self._http_client

    async def close(self) -> None:
        await super().close()
        async with self._http_client_lock:
            if self._http_client and not self._http_client.is_closed:
                await self._http_client.aclose()
            self._http_client = None

    def _jobs_path(self) -> str:  # pragma: no cover - unused by enqueue
        return "/images/generations"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        idempotency_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], int, int]:
        url = f"{self._base_url}{path}"
        headers = self._build_headers()
        if idempotency_key:
            headers.setdefault("Idempotency-Key", idempotency_key)
        headers.update(kwargs.pop("headers", {}))
        params_payload = kwargs.get("params")
        json_payload = kwargs.get("json")
        data_payload = kwargs.get("data")
        context = dict(self._pending_log_context or {})
        context_text = _serialise_for_log(context) if context else ""
        max_attempts = max(1, self._image_max_retries)
        attempt = 0
        delay = 1.0
        last_error: Optional[ProviderAPIError] = None
        safe_headers = _sanitize_headers(headers)
        try:
            while attempt < max_attempts:
                attempt += 1
                log.debug(
                    "Provider request attempt=%s provider=%s method=%s url=%s headers=%s params=%s json=%s data=%s context=%s",
                    attempt,
                    self._provider_name,
                    method,
                    url,
                    safe_headers,
                    _serialise_for_log(params_payload),
                    _serialise_for_log(json_payload),
                    _serialise_for_log(data_payload),
                    context_text,
                )
                client = await self._ensure_http_client()
                start = time.monotonic()
                try:
                    async with self._request_gate:
                        response = await client.request(
                            method,
                            url,
                            headers=headers,
                            **kwargs,
                        )
                        content = await response.aread()
                except (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException) as exc:
                    duration_ms = int((time.monotonic() - start) * 1000)
                    log.warning(
                        "Provider request timed out locally provider=%s method=%s url=%s duration_ms=%s attempt=%s/%s",
                        self._provider_name,
                        method,
                        url,
                        duration_ms,
                        attempt,
                        max_attempts,
                    )
                    last_error = ProviderAPIError(
                        provider=self._provider_name,
                        status_code=0,
                        message="Image generation request timed out",
                        error_type="timeout_local",
                        provider_message="Local timeout waiting for OpenAI Images response",
                        duration_ms=duration_ms,
                    )
                    if attempt >= max_attempts:
                        raise last_error from exc
                    jitter = random.uniform(0, 0.3)
                    await asyncio.sleep(delay + jitter)
                    delay *= 2
                    continue
                except httpx.HTTPError as exc:
                    duration_ms = int((time.monotonic() - start) * 1000)
                    log.warning(
                        "Provider request transport error provider=%s method=%s url=%s duration_ms=%s error=%s",
                        self._provider_name,
                        method,
                        url,
                        duration_ms,
                        exc,
                    )
                    last_error = ProviderAPIError(
                        provider=self._provider_name,
                        status_code=0,
                        message="Image generation request failed",
                        error_type="network",
                        provider_message=str(exc),
                        duration_ms=duration_ms,
                    )
                    raise last_error from exc
                duration_ms = int((time.monotonic() - start) * 1000)
                response_headers = dict(response.headers)
                request_id = _extract_request_id(response_headers) or ""
                safe_response_headers = _sanitize_headers(response_headers)
                encoding = response.encoding or "utf-8"
                text = content.decode(encoding, errors="replace") if content else ""
                await response.aclose()
                self._record_response_meta(
                    method=method,
                    url=url,
                    status=response.status_code,
                    duration_ms=duration_ms,
                    headers=response_headers,
                    context=context,
                )
                log.debug(
                    "Provider response provider=%s method=%s url=%s status=%s duration_ms=%s request_id=%s headers=%s context=%s body=%s",
                    self._provider_name,
                    method,
                    url,
                    response.status_code,
                    duration_ms,
                    request_id,
                    safe_response_headers,
                    context_text,
                    _truncate(text, limit=3072),
                )
                status = response.status_code
                if status >= 400:
                    error = self._build_error(status, text, duration_ms)
                    if status in {408, 504, 524}:
                        error.error_type = "timeout_edge"
                    else:
                        error.error_type = f"http_error_{status}"
                    error.retryable = status in _RETRYABLE_STATUS_CODES
                    last_error = error
                    log.warning(
                        "Provider responded with error provider=%s method=%s url=%s status=%s duration_ms=%s request_id=%s error_type=%s retryable=%s",
                        self._provider_name,
                        method,
                        url,
                        status,
                        duration_ms,
                        request_id,
                        error.error_type,
                        error.retryable,
                    )
                    if error.retryable and attempt < max_attempts:
                        jitter = random.uniform(0, 0.3)
                        log.warning(
                            "Retrying provider request provider=%s method=%s url=%s attempt=%s/%s reason=%s",
                            self._provider_name,
                            method,
                            url,
                            attempt + 1,
                            max_attempts,
                            error.error_type,
                        )
                        await asyncio.sleep(delay + jitter)
                        delay *= 2
                        continue
                    raise error
                if not text:
                    data: Dict[str, Any] = {}
                else:
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError as exc:
                        log.error(
                            "Failed to decode provider response provider=%s method=%s url=%s status=%s request_id=%s headers=%s context=%s body=%s",
                            self._provider_name,
                            method,
                            url,
                            status,
                            request_id,
                            safe_response_headers,
                            context_text,
                            _truncate(text),
                        )
                        raise ProviderAPIError(
                            provider=self._provider_name,
                            status_code=status,
                            message="Unable to decode provider response",
                            error_type="decode",
                            provider_message=text,
                            retryable=False,
                            duration_ms=duration_ms,
                        ) from exc
                return data, status, duration_ms
        finally:
            self._pending_log_context = None
        assert last_error is not None  # pragma: no cover - defensive guard
        raise last_error

    async def enqueue_job(
        self,
        *,
        prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
        **_: Any,
    ) -> ProviderJobSubmission:
        if not self._api_key:
            raise ProviderAPIError(
                provider=self.provider_name,
                status_code=401,
                message="OpenAI API key is not configured",
                error_type="auth",
            )
        options = dict(settings or {})
        model_candidate = options.get("model") or self._default_model or "gpt-image-1"
        model = str(model_candidate).strip().lower() or "gpt-image-1"
        size = options.get("size")
        quality = options.get("quality")
        response_format = options.get("response_format") or "b64_json"
        user = options.get("user")
        request: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "response_format": response_format,
        }
        if model == "dall-e-3":
            request["n"] = 1
        if size:
            request["size"] = str(size)
        if quality:
            request["quality"] = quality
        if user:
            request["user"] = str(user)
        if payload and isinstance(payload, dict):
            request.setdefault("prompt", prompt)
        data, status_code, duration_ms = await self._request(
            "POST",
            "/images/generations",
            json=request,
            idempotency_key=idempotency_key,
        )
        job_id = str(uuid4())
        inline_assets: List[Dict[str, Any]] = []
        assets_meta: List[Dict[str, Any]] = []
        asset_map: Dict[str, str] = {}
        images = data.get("data") if isinstance(data, dict) else None
        if isinstance(images, list):
            for index, item in enumerate(images):
                if not isinstance(item, dict):
                    continue
                key = f"image_{index}"
                if item.get("b64_json"):
                    raw = str(item["b64_json"]) or ""
                    try:
                        binary = base64.b64decode(raw)
                    except Exception:
                        binary = b""
                    data_uri = f"data:image/png;base64,{raw}" if raw else ""
                    inline_assets.append(
                        {
                            "key": key,
                            "kind": "image",
                            "mime": "image/png",
                            "bytes": len(binary),
                            "data_uri": data_uri,
                            "data_url": data_uri,
                        }
                    )
                    if data_uri:
                        asset_map[key] = data_uri
                        assets_meta.append(
                            {
                                "key": key,
                                "inline": True,
                                "mime": "image/png",
                                "bytes": len(binary),
                            }
                        )
                elif item.get("url"):
                    url = str(item.get("url"))
                    if url:
                        asset_map[key] = url
                        assets_meta.append({"key": key, "inline": False, "url": url})
        record = {
            "payload": data,
            "inline_assets": inline_assets,
            "assets": assets_meta,
            "asset_map": asset_map,
            "status_code": status_code,
            "duration_ms": duration_ms,
        }
        self._pending[job_id] = record
        self._persist_pending_record(job_id, record)
        result_payload = {
            "payload": data,
            "inline_assets": inline_assets,
            "assets": assets_meta,
        }
        return ProviderJobSubmission(
            job_id=job_id,
            status_code=status_code,
            duration_ms=duration_ms,
            data=result_payload,
        )

    async def get_job_status(self, job_id: str) -> ProviderJobStatus:
        record = self._pending.pop(job_id, None)
        if not record:
            record = self._load_pending_record(job_id)
        if not record:
            return ProviderJobStatus(
                job_id=job_id,
                status="failed",
                assets={},
                error="Result not found",
                data={},
                status_code=404,
                duration_ms=0,
            )
        self._delete_pending_record(job_id)
        asset_map = record.get("asset_map") or {}
        inline_assets = record.get("inline_assets") or []
        assets_meta = record.get("assets") or []
        payload = record.get("payload") or {}
        status = "completed" if asset_map or inline_assets else "failed"
        error: Optional[str] = None
        if status != "completed":
            error = "OpenAI Images API did not return an image"
        data = {
            "payload": payload,
            "inline_assets": inline_assets,
            "assets": assets_meta,
        }
        return ProviderJobStatus(
            job_id=job_id,
            status=status,
            assets=asset_map,
            error=error,
            data=data,
            status_code=int(record.get("status_code", 200)),
            duration_ms=int(record.get("duration_ms", 0)),
        )

    def _pending_path(self, job_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id or "job") or "job"
        return self._pending_dir / f"{safe_id}.json"

    def _persist_pending_record(self, job_id: str, record: Dict[str, Any]) -> None:
        path = self._pending_path(job_id)
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False)
        except Exception:  # pragma: no cover - filesystem guard
            log.warning("Failed to persist OpenAI image result job_id=%s", job_id, exc_info=True)

    def _load_pending_record(self, job_id: str) -> Optional[Dict[str, Any]]:
        path = self._pending_path(job_id)
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return None
        except Exception:  # pragma: no cover - filesystem guard
            log.warning("Failed to load OpenAI image result job_id=%s", job_id, exc_info=True)
            return None
        if isinstance(data, dict):
            return data
        return None

    def _delete_pending_record(self, job_id: str) -> None:
        path = self._pending_path(job_id)
        try:
            path.unlink(missing_ok=True)
        except Exception:  # pragma: no cover - filesystem guard
            log.debug("Failed to remove OpenAI image cache job_id=%s", job_id, exc_info=True)


__all__ = ["OpenAIImageClient"]
