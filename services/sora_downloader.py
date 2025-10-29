"""Utility helpers for downloading Sora video assets."""
from __future__ import annotations

import logging
import mimetypes
from typing import Optional
from urllib.parse import urlparse

import aiohttp

from config import Config
from services.gemini_downloader import DownloadedAsset

log = logging.getLogger(__name__)


class SoraDownloadError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = status_code


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


def _guess_filename(asset_url: str, mime: str, filename_hint: Optional[str]) -> str:
    if filename_hint:
        return filename_hint
    parsed = urlparse(asset_url)
    candidate = (parsed.path or "").rstrip("/").rsplit("/", 1)[-1]
    if candidate and "." in candidate:
        return candidate
    extension = mimetypes.guess_extension(mime.split(";")[0].strip()) if mime else None
    if not extension:
        extension = ".mp4"
    return f"sora_video{extension}"


async def download_sora_asset(
    *,
    asset_url: str,
    config: Config,
    corr_id: str,
    asset_name: str,
    expected_mask: Optional[str],
    mime_hint: Optional[str] = None,
    filename_hint: Optional[str] = None,
) -> DownloadedAsset:
    """Download the asset referenced by *asset_url* using the OpenAI API."""

    api_key = config.openai_key
    if not api_key:
        raise SoraDownloadError("Sora API key is not configured", status_code=401)

    cleaned = (asset_url or "").strip()
    if not cleaned:
        raise SoraDownloadError("Пустая ссылка на файл Sora", status_code=400)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "sora-video-bot/1.0",
    }

    timeout = aiohttp.ClientTimeout(total=config.request_timeout or 60.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(cleaned, headers=headers) as response:
                body = await response.read()
                if response.status >= 400:
                    log.warning(
                        "sora.download failed corr_id=%s status=%s asset=%s", corr_id, response.status, asset_name
                    )
                    raise SoraDownloadError(
                        f"Не удалось скачать файл Sora: {response.status}",
                        status_code=response.status,
                    )
                mime = mime_hint or response.headers.get("Content-Type", "video/mp4")
                disposition = response.headers.get("Content-Disposition")
                filename = (
                    filename_hint
                    or _parse_disposition_filename(disposition)
                    or _guess_filename(cleaned, mime, filename_hint)
                )
                log.debug(
                    "sora.download ok corr_id=%s asset=%s bytes=%s", corr_id, asset_name, len(body)
                )
                return DownloadedAsset(
                    content=body,
                    mime=mime,
                    filename=filename,
                    size=len(body),
                    key_mask="",
                )
        except aiohttp.ClientError as exc:  # pragma: no cover - network failures
            log.warning(
                "sora.download error corr_id=%s asset=%s err=%s", corr_id, asset_name, exc,
            )
            raise SoraDownloadError("Сетевая ошибка при скачивании файла Sora") from exc

