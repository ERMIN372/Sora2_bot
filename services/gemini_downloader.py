from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import aiohttp

from config import Config
from services.gemini_key import current_key_mask, ensure_gemini_key_logged

log = logging.getLogger(__name__)


@dataclass(slots=True)
class DownloadedAsset:
    """Container for downloaded Gemini assets."""

    content: bytes
    mime: str
    filename: str
    size: int
    key_mask: str


class GeminiDownloadError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 0, error_status: Optional[str] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_status = error_status


class GeminiConfigurationError(GeminiDownloadError):
    pass


class GeminiKeyMismatchError(GeminiDownloadError):
    def __init__(self, expected: str, actual: str) -> None:
        super().__init__(
            "Используются разные ключи Gemini для генерации и скачивания",
            status_code=0,
            error_status="KEY_MISMATCH",
        )
        self.expected = expected
        self.actual = actual


async def download_asset(
    *,
    asset_url: str,
    config: Config,
    corr_id: str,
    asset_name: str,
    expected_mask: Optional[str],
    mime_hint: Optional[str] = None,
    filename_hint: Optional[str] = None,
) -> DownloadedAsset:
    """Download *asset_url* using the configured Gemini API key."""

    ensure_gemini_key_logged(
        config,
        context="gemini_downloader",
        process_id=f"pid={os.getpid()}",
        logger=log,
    )
    api_key = (config.gemini_api_key or "").strip()
    if not api_key:
        raise GeminiConfigurationError("Gemini API key is not configured", status_code=401)

    mask = current_key_mask(config)
    if expected_mask and mask and expected_mask != mask:
        raise GeminiKeyMismatchError(expected_mask, mask)

    timeout = aiohttp.ClientTimeout(
        total=max(config.request_timeout or 60.0, 30.0),
        sock_connect=config.request_connect_timeout or 10.0,
        sock_read=config.request_read_timeout or 30.0,
    )

    attempt = 0
    max_attempts = 3
    delay = max(config.retry_backoff or 2.0, 1.5)
    permission_retry = False

    headers = {"x-goog-api-key": api_key}

    while attempt < max_attempts:
        attempt += 1
        log.info(
            "gemini.download start corr_id=%s asset=%s attempt=%s/%s env=%s key_mask=%s",
            corr_id,
            asset_name,
            attempt,
            max_attempts,
            (config.environment or "dev").lower(),
            mask or "",
        )
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(asset_url, headers=headers) as response:
                    body = await response.read()
                    status = response.status
                    content_type = response.headers.get("content-type") or mime_hint or "application/octet-stream"
                    if status >= 400:
                        error_status = None
                        message = f"Gemini download failed ({status})"
                        if body:
                            try:
                                payload = json.loads(body.decode("utf-8"))
                                if isinstance(payload, dict):
                                    error_obj = payload.get("error")
                                    if isinstance(error_obj, dict):
                                        error_status = str(error_obj.get("status") or "")
                                        message = str(error_obj.get("message") or message)
                            except (UnicodeDecodeError, json.JSONDecodeError):
                                message = message
                        log.warning(
                            "gemini.download error corr_id=%s asset=%s status=%s status_label=%s key_mask=%s",
                            corr_id,
                            asset_name,
                            status,
                            error_status,
                            mask or "",
                        )
                        if error_status in {"PERMISSION_DENIED", "SERVICE_DISABLED"}:
                            if permission_retry:
                                raise GeminiConfigurationError(message, status_code=status, error_status=error_status)
                            permission_retry = True
                            await asyncio.sleep(2.0)
                            continue
                        if status >= 500 or status in {408, 429}:
                            await asyncio.sleep(delay)
                            delay *= config.retry_backoff or 2.0
                            continue
                        raise GeminiDownloadError(message, status_code=status, error_status=error_status)
                    if not body:
                        raise GeminiDownloadError("Gemini вернул пустой файл", status_code=status)
                    if filename_hint:
                        filename = filename_hint
                    elif asset_name and "." in asset_name:
                        filename = asset_name
                    else:
                        base_name = asset_name or "result"
                        extension = ".mp4" if content_type.startswith("video/") else ".bin"
                        filename = f"{base_name}{extension}"
                    log.info(
                        "gemini.download success corr_id=%s asset=%s bytes=%s key_mask=%s",
                        corr_id,
                        asset_name,
                        len(body),
                        mask or "",
                    )
                    return DownloadedAsset(
                        content=body,
                        mime=content_type,
                        filename=filename,
                        size=len(body),
                        key_mask=mask,
                    )
        except GeminiDownloadError:
            raise
        except GeminiConfigurationError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            log.warning(
                "gemini.download network error corr_id=%s asset=%s attempt=%s/%s key_mask=%s error=%s",
                corr_id,
                asset_name,
                attempt,
                max_attempts,
                mask or "",
                exc,
            )
            if attempt >= max_attempts:
                raise GeminiDownloadError(str(exc)) from exc
            await asyncio.sleep(delay)
            delay *= config.retry_backoff or 2.0

    raise GeminiDownloadError("Не удалось скачать файл Gemini", status_code=0)


__all__ = [
    "DownloadedAsset",
    "GeminiConfigurationError",
    "GeminiDownloadError",
    "GeminiKeyMismatchError",
    "download_asset",
]
