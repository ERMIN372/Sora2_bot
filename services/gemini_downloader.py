from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

try:  # pragma: no cover - optional dependency during tests
    import httpx
except ModuleNotFoundError:  # pragma: no cover - tests may stub the client
    class _HttpxMissing:
        class Timeout:  # type: ignore[return-type]
            def __init__(self, *args, **kwargs) -> None:
                raise ModuleNotFoundError("httpx is required")

        class AsyncClient:  # type: ignore[return-type]
            def __init__(self, *args, **kwargs) -> None:
                raise ModuleNotFoundError("httpx is required")

    httpx = _HttpxMissing()  # type: ignore[assignment]

from config import Config
from providers.base import validate_download_url
from services.gemini_client import get_media_client
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
    path: Optional[Path] = None


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


_DOWNLOAD_DIR = Path("attached_assets") / "downloads" / "gemini"


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


def _sanitise_segment(segment: str) -> str:
    cleaned = segment.strip().replace(os.sep, "_")
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", cleaned) or "asset"


def _ensure_download_path(asset_name: str, filename: str) -> Path:
    segments = [seg for seg in asset_name.split("/") if seg.strip()]
    if not segments:
        segments = ["asset"]
    safe_segments = [_sanitise_segment(seg) for seg in segments]
    directory = _DOWNLOAD_DIR.joinpath(*safe_segments[:-1]) if len(safe_segments) > 1 else _DOWNLOAD_DIR
    directory.mkdir(parents=True, exist_ok=True)
    safe_filename = _sanitise_segment(filename)
    return directory / safe_filename


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

    should_fetch_meta = True
    if cleaned_url.startswith("http://") or cleaned_url.startswith("https://"):
        direct_url = cleaned_url
        try:
            file_name = _extract_file_name(cleaned_url)
        except ValueError:
            file_name = asset_name
            should_fetch_meta = False
    else:
        try:
            file_name = _extract_file_name(cleaned_url)
        except ValueError as exc:
            raise GeminiDownloadError(str(exc), status_code=400) from exc
        base = cleaned_url.split("?", 1)[0]
        if not base:
            base = file_name
        if ":download" not in base:
            base = f"{base}:download"
        query = "alt=media"
        if "?" in cleaned_url:
            _, existing_query = cleaned_url.split("?", 1)
            if existing_query:
                parts = [part for part in existing_query.split("&") if part]
                if not any(part.startswith("alt=") for part in parts):
                    parts.append("alt=media")
                query = "&".join(parts)
        direct_url = f"https://generativelanguage.googleapis.com/v1beta/{base}?{query}"

    client = get_media_client(config)
    retry_schedule = (0.5, 1.0, 2.0)
    max_attempts = len(retry_schedule) + 1
    attempt = 0
    file_meta: Optional[_genai_types.File] = None
    disposition_hint: Optional[str] = None

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
        if file_meta is None and should_fetch_meta:
            try:
                file_meta = await asyncio.to_thread(client.files.get, name=file_name)
            except _genai_errors.APIError as exc:
                status_code, status_label = _extract_status(exc)
                log.warning(
                    "gemini.download file.meta error corr_id=%s asset=%s file=%s status=%s status_label=%s key_mask=%s",
                    corr_id,
                    asset_name,
                    file_name,
                    status_code or "",
                    status_label or "",
                    mask or "",
                )
                file_meta = None
                if status_code in {401, 429, 500, 502, 503, 504, 408} or status_label in {
                    "UNAVAILABLE",
                    "RESOURCE_EXHAUSTED",
                }:
                    if attempt >= max_attempts:
                        raise GeminiDownloadError(
                            str(exc), status_code=status_code, error_status=status_label
                        ) from exc
                    await asyncio.sleep(retry_schedule[min(attempt - 1, len(retry_schedule) - 1)])
                    continue
                if status_code in {403} or status_label in {
                    "PERMISSION_DENIED",
                    "SERVICE_DISABLED",
                    "UNAUTHENTICATED",
                }:
                    if attempt >= max_attempts:
                        raise GeminiDownloadError(
                            str(exc) or "Gemini вернул ошибку доступа при скачивании",
                            status_code=status_code or 403,
                            error_status=status_label or "PERMISSION_DENIED",
                        ) from exc
                    await asyncio.sleep(retry_schedule[min(attempt - 1, len(retry_schedule) - 1)])
                    continue
                raise GeminiDownloadError(
                    str(exc), status_code=status_code, error_status=status_label
                ) from exc
            except Exception as exc:
                log.warning(
                    "gemini.download file.meta exception corr_id=%s asset=%s file=%s attempt=%s/%s key_mask=%s error=%s",
                    corr_id,
                    asset_name,
                    file_name,
                    attempt,
                    max_attempts,
                    mask or "",
                    exc,
                )
                if attempt >= max_attempts:
                    raise GeminiDownloadError(str(exc)) from exc
                await asyncio.sleep(retry_schedule[min(attempt - 1, len(retry_schedule) - 1)])
                continue

        try:
            payload, headers = await _download_via_http(
                url=direct_url,
                api_key=api_key,
                corr_id=corr_id,
                asset_name=asset_name,
                mask=mask or "",
                attempt=attempt,
                max_attempts=max_attempts,
                config=config,
            )
        except GeminiDownloadError as exc:
            if attempt >= max_attempts:
                raise
            await asyncio.sleep(retry_schedule[min(attempt - 1, len(retry_schedule) - 1)])
            continue

        disposition_hint = _parse_disposition_filename(headers.get("content-disposition"))
        mime = headers.get("content-type") or _extract_mime(file_meta=file_meta, mime_hint=mime_hint)
        if not mime:
            mime = _extract_mime(file_meta=file_meta, mime_hint=mime_hint)
        if not mime.lower().startswith("video/"):
            raise GeminiDownloadError(
                "Gemini вернул файл неожиданного типа",
                status_code=200,
                error_status=mime or "invalid_mime",
            )
        filename = _build_filename(
            asset_name=asset_name,
            file_meta=file_meta,
            filename_hint=filename_hint or disposition_hint,
            mime=mime,
        )
        path = _ensure_download_path(asset_name, filename)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            log.warning("Failed to create download directory path=%s", path.parent, exc_info=True)
        try:
            path.write_bytes(payload)
        except Exception as exc:
            raise GeminiDownloadError(str(exc)) from exc
        size_bytes = path.stat().st_size if path.exists() else len(payload)
        if size_bytes <= 0:
            raise GeminiDownloadError("Gemini вернул пустой файл", status_code=0)
        log.info(
            "gemini.download success corr_id=%s asset=%s file=%s bytes=%s key_mask=%s",
            corr_id,
            asset_name,
            file_name,
            size_bytes,
            mask or "",
        )
        return DownloadedAsset(
            content=payload,
            mime=mime,
            filename=filename,
            size=size_bytes,
            key_mask=mask or "",
            path=path,
        )

    raise GeminiDownloadError("Не удалось скачать файл Gemini", status_code=0)


def _extract_status(exc: _genai_errors.APIError) -> Tuple[int, Optional[str]]:
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
        status_label = str(payload_error.get("status") or status_label or "").upper() or None
    return status_code, status_label


async def _download_via_http(
    *,
    url: str,
    api_key: str,
    corr_id: str,
    asset_name: str,
    mask: str,
    attempt: int,
    max_attempts: int,
    config: Config,
) -> Tuple[bytes, httpx.Headers]:
    timeout = httpx.Timeout(
        connect=config.request_connect_timeout,
        read=config.request_read_timeout,
        write=config.request_connect_timeout,
        pool=None,
    )
    headers = {"x-goog-api-key": api_key}
    retryable_status = {408, 429, 500, 502, 503, 504}

    log.info(
        "gemini.download http start corr_id=%s asset=%s attempt=%s/%s key_mask=%s",
        corr_id,
        asset_name,
        attempt,
        max_attempts,
        mask or "",
    )
    validate_download_url(url)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
    except httpx.TimeoutException as exc:
        raise GeminiDownloadError("Gemini не ответил вовремя", status_code=0) from exc
    except httpx.RequestError as exc:
        raise GeminiDownloadError(str(exc), status_code=0) from exc

    status = response.status_code
    if status == 200:
        payload = await response.aread()
        declared_length = response.headers.get("content-length")
        if declared_length is not None:
            try:
                if int(declared_length) <= 0:
                    raise GeminiDownloadError("Gemini вернул пустой файл", status_code=0)
            except ValueError:
                pass
        if not payload:
            raise GeminiDownloadError("Gemini вернул пустой файл", status_code=0)
        return payload, response.headers

    if status in retryable_status:
        raise GeminiDownloadError(
            "Gemini вернул ошибку при скачивании",
            status_code=status,
            error_status=str(status),
        )
    if status in {401}:
        raise GeminiConfigurationError(
            "Gemini вернул ошибку доступа при скачивании",
            status_code=status,
            error_status=str(status),
        )
    if status == 403:
        raise GeminiConfigurationError(
            "Сервер отказал в доступе к прямой ссылке Gemini",
            status_code=status,
            error_status="PERMISSION_DENIED",
        )
    raise GeminiDownloadError(
        "Gemini вернул ошибку при скачивании",
        status_code=status,
        error_status=str(status),
    )


__all__ = [
    "DownloadedAsset",
    "GeminiConfigurationError",
    "GeminiDownloadError",
    "GeminiKeyMismatchError",
    "download_asset",
]
