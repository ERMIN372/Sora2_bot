from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import aiohttp

from config import Config
from services.gemini_client import get_gemini_client
from services.gemini_key import current_key_mask, ensure_gemini_key_logged

try:  # pragma: no cover - import guard for optional dependency
    from google.genai import errors as _genai_errors, types as _genai_types
except ModuleNotFoundError:  # pragma: no cover - tests provide a stub
    from google.genai import errors as _genai_errors  # type: ignore
    from google.genai import types as _genai_types  # type: ignore

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


def _extract_file_name(asset_url: str) -> str:
    """Return the Files API resource name extracted from *asset_url*."""

    cleaned = (asset_url or "").strip()
    if not cleaned:
        raise ValueError("Пустая ссылка на файл Gemini")
    if cleaned.startswith("files/"):
        base = cleaned.split("?", 1)[0]
        return base.split(":", 1)[0]

    parsed = urlparse(cleaned)
    path = parsed.path or ""
    marker = "/files/"
    marker_index = path.find(marker)
    if marker_index == -1:
        raise ValueError("Не удалось определить идентификатор файла Gemini")
    remainder = path[marker_index + len(marker) :]
    if not remainder:
        raise ValueError("Не удалось определить идентификатор файла Gemini")
    remainder = remainder.split(":", 1)[0].strip("/")
    if not remainder:
        raise ValueError("Не удалось определить идентификатор файла Gemini")
    return f"files/{remainder}"


def _parse_disposition_filename(disposition: Optional[str]) -> Optional[str]:
    if not disposition:
        return None
    parts = disposition.split(";")
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key.strip().lower() not in {"filename", "filename*"}:
            continue
        cleaned = value.strip().strip('"')
        if cleaned:
            return cleaned
    return None


def _build_filename(
    *,
    asset_name: str,
    file_meta: Optional[_genai_types.File],
    filename_hint: Optional[str],
    mime: str,
) -> str:
    """Determine the best filename for the downloaded asset."""

    if file_meta and isinstance(getattr(file_meta, "display_name", None), str):
        candidate = file_meta.display_name.strip()
        if candidate:
            return candidate
    if filename_hint:
        return filename_hint
    if asset_name and "." in asset_name:
        return asset_name
    extension = None
    if mime:
        extension = mimetypes.guess_extension(mime.split(";")[0].strip())
    if not extension:
        extension = ".mp4" if mime.startswith("video/") else ".bin"
    base = asset_name or "result"
    return f"{base}{extension}"


def _extract_mime(
    *,
    file_meta: Optional[_genai_types.File],
    mime_hint: Optional[str],
) -> str:
    if file_meta and isinstance(getattr(file_meta, "mime_type", None), str):
        candidate = file_meta.mime_type.strip()
        if candidate:
            return candidate
    if mime_hint:
        return mime_hint
    return "application/octet-stream"


async def _download_direct_asset(
    *,
    url: str,
    api_key: str,
    config: Config,
    corr_id: str,
    asset_name: str,
    mask: str,
    mime_hint: Optional[str],
    filename_hint: Optional[str],
) -> DownloadedAsset:
    if not api_key:
        raise GeminiConfigurationError("Gemini API key is not configured", status_code=401)

    timeout = aiohttp.ClientTimeout(
        total=None,
        sock_connect=config.request_connect_timeout,
        sock_read=config.request_read_timeout,
    )
    headers = {"x-goog-api-key": api_key}
    max_attempts = 3
    delay = max(config.retry_backoff or 2.0, 1.5)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for attempt in range(1, max_attempts + 1):
            log.info(
                "gemini.download direct start corr_id=%s asset=%s attempt=%s/%s key_mask=%s",
                corr_id,
                asset_name,
                attempt,
                max_attempts,
                mask or "",
            )
            try:
                async with session.get(url, headers=headers) as response:
                    payload = await response.read()
                    status = response.status
                    if status == 200:
                        mime = response.headers.get("Content-Type") or mime_hint or "video/mp4"
                        disposition_name = _parse_disposition_filename(
                            response.headers.get("Content-Disposition")
                        )
                        effective_hint = filename_hint or disposition_name
                        filename = _build_filename(
                            asset_name=asset_name,
                            file_meta=None,
                            filename_hint=effective_hint,
                            mime=mime,
                        )
                        try:
                            size_bytes = int(response.headers.get("Content-Length") or 0)
                        except (TypeError, ValueError):
                            size_bytes = 0
                        if size_bytes <= 0:
                            size_bytes = len(payload)
                        log.info(
                            "gemini.download direct success corr_id=%s asset=%s bytes=%s key_mask=%s",
                            corr_id,
                            asset_name,
                            len(payload),
                            mask or "",
                        )
                        return DownloadedAsset(
                            content=payload,
                            mime=mime,
                            filename=filename,
                            size=size_bytes,
                            key_mask=mask,
                        )
                    elif status in {401, 403}:
                        raise GeminiConfigurationError(
                            "Сервер отказал в доступе к прямой ссылке Gemini",
                            status_code=status,
                            error_status="PERMISSION_DENIED",
                        )
                    elif status >= 500 or status in {408, 429}:
                        if attempt >= max_attempts:
                            raise GeminiDownloadError(
                                "Gemini вернул ошибку при скачивании",
                                status_code=status,
                                error_status=str(status),
                            )
                        await asyncio.sleep(delay)
                        delay *= config.retry_backoff or 2.0
                        continue
                    else:
                        raise GeminiDownloadError(
                            "Gemini вернул ошибку при скачивании",
                            status_code=status,
                            error_status=str(status),
                        )
            except (GeminiDownloadError, GeminiConfigurationError):
                raise
            except Exception as exc:
                log.warning(
                    "gemini.download direct error corr_id=%s asset=%s attempt=%s/%s key_mask=%s error=%s",
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
                continue

    raise GeminiDownloadError("Не удалось скачать файл Gemini", status_code=0)


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
    """Download the Gemini asset referenced by *asset_url* via the Files API."""

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

    cleaned_url = (asset_url or "").strip()
    if not cleaned_url:
        raise GeminiDownloadError("Пустая ссылка на файл Gemini", status_code=400)

    if cleaned_url.startswith("http://") or cleaned_url.startswith("https://"):
        if "/files/" not in cleaned_url:
            return await _download_direct_asset(
                url=cleaned_url,
                api_key=api_key,
                config=config,
                corr_id=corr_id,
                asset_name=asset_name,
                mask=mask or "",
                mime_hint=mime_hint,
                filename_hint=filename_hint,
            )

    try:
        file_name = _extract_file_name(cleaned_url)
    except ValueError as exc:
        raise GeminiDownloadError(str(exc), status_code=400) from exc

    client = get_gemini_client(config)
    attempt = 0
    max_attempts = 3
    delay = max(config.retry_backoff or 2.0, 1.5)
    permission_retry = False
    file_meta: Optional[_genai_types.File] = None

    while attempt < max_attempts:
        attempt += 1
        log.info(
            "gemini.download start corr_id=%s asset=%s file=%s attempt=%s/%s env=%s key_mask=%s",
            corr_id,
            asset_name,
            file_name,
            attempt,
            max_attempts,
            (config.environment or "dev").lower(),
            mask or "",
        )
        try:
            if file_meta is None:
                file_meta = await asyncio.to_thread(client.files.get, name=file_name)
            payload = await asyncio.to_thread(client.files.download, file=file_meta)
        except _genai_errors.APIError as exc:
            raw_code = getattr(exc, "code", 0)
            payload_error = None
            if isinstance(raw_code, dict):
                payload_error = raw_code.get("error") if isinstance(raw_code.get("error"), dict) else None
                raw_code = (
                    (payload_error or {}).get("code")
                    or (payload_error or {}).get("status")
                    or 0
                )
            try:
                status_code = int(raw_code or 0)
            except (TypeError, ValueError):
                status_code = int(exc.args[0]) if exc.args else 0
            status_label = str(getattr(exc, "status", "") or "").upper() or None
            if payload_error:
                status_label = (
                    str(payload_error.get("status") or status_label or "").upper() or None
                )
            message = str(exc)
            log.warning(
                "gemini.download api_error corr_id=%s asset=%s file=%s status=%s status_label=%s key_mask=%s",
                corr_id,
                asset_name,
                file_name,
                status_code or "",
                status_label or "",
                mask or "",
            )
            file_meta = None
            if status_code == 403 or status_label in {"PERMISSION_DENIED", "SERVICE_DISABLED"}:
                if permission_retry:
                    raise GeminiConfigurationError(
                        message,
                        status_code=status_code or 403,
                        error_status=status_label or "PERMISSION_DENIED",
                    ) from exc
                permission_retry = True
                await asyncio.sleep(min(delay, 3.0))
                continue
            if status_code >= 500 or status_code in {408, 429}:
                if attempt >= max_attempts:
                    raise GeminiDownloadError(
                        message, status_code=status_code, error_status=status_label
                    ) from exc
                await asyncio.sleep(delay)
                delay *= config.retry_backoff or 2.0
                continue
            raise GeminiDownloadError(
                message, status_code=status_code, error_status=status_label
            ) from exc
        except Exception as exc:
            log.warning(
                "gemini.download error corr_id=%s asset=%s file=%s attempt=%s/%s key_mask=%s error=%s",
                corr_id,
                asset_name,
                file_name,
                attempt,
                max_attempts,
                mask or "",
                exc,
            )
            file_meta = None
            if attempt >= max_attempts:
                raise GeminiDownloadError(str(exc)) from exc
            await asyncio.sleep(delay)
            delay *= config.retry_backoff or 2.0
            continue
        else:
            if not payload:
                raise GeminiDownloadError("Gemini вернул пустой файл", status_code=0)
            mime = _extract_mime(file_meta=file_meta, mime_hint=mime_hint)
            filename = _build_filename(
                asset_name=asset_name,
                file_meta=file_meta,
                filename_hint=filename_hint,
                mime=mime,
            )
            size_bytes = (
                int(getattr(file_meta, "size_bytes", 0) or 0)
                if file_meta is not None
                else 0
            )
            if size_bytes <= 0:
                size_bytes = len(payload)
            log.info(
                "gemini.download success corr_id=%s asset=%s file=%s bytes=%s key_mask=%s",
                corr_id,
                asset_name,
                file_name,
                len(payload),
                mask or "",
            )
            return DownloadedAsset(
                content=payload,
                mime=mime,
                filename=filename,
                size=size_bytes,
                key_mask=mask,
            )

    raise GeminiDownloadError("Не удалось скачать файл Gemini", status_code=0)


__all__ = [
    "DownloadedAsset",
    "GeminiConfigurationError",
    "GeminiDownloadError",
    "GeminiKeyMismatchError",
    "download_asset",
]
