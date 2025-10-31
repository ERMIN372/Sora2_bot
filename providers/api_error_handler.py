"""Centralised mapping of provider API errors to user-facing messages."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional, Sequence

from .base import ProviderAPIError


@dataclass(slots=True)
class ApiErrorResolution:
    """Result of matching an API error to a friendly explanation."""

    message: str
    error_type: str
    error_code: Optional[str] = None
    retryable: bool = False
    notify_admin: bool = False
    cooldown_seconds: float = 0.0


class ApiErrorHandler:
    """Normalises API errors based on the official OpenAI error table."""

    SERVER_ERROR_STATUSES = {500, 503}
    SERVER_RETRY_DELAYS = (1.0, 5.0, 15.0)
    RATE_LIMIT_PENALTY_SECONDS = 10.0

    def __init__(
        self,
        *,
        provider: str,
        logger: logging.Logger,
        supported_models: Optional[Sequence[str]] = None,
    ) -> None:
        self._provider = provider
        self._logger = logger
        self._supported_models = list(supported_models or ())

    def server_retry_delay(self, index: int) -> Optional[float]:
        if 0 <= index < len(self.SERVER_RETRY_DELAYS):
            return self.SERVER_RETRY_DELAYS[index]
        return None

    def resolve(
        self,
        *,
        status_code: int,
        error_code: Optional[str],
        error_message: Optional[str],
    ) -> Optional[ApiErrorResolution]:
        code = (error_code or "").strip().lower()
        message = (error_message or "").strip()

        if code == "model_not_found":
            available = ", ".join(self._supported_models)
            hint = (
                f" Доступные модели: {available}" if available else ""
            )
            return ApiErrorResolution(
                message=f"Указанная модель недоступна.{hint}",
                error_type="model_not_found",
                error_code=error_code or "model_not_found",
                retryable=False,
            )

        if status_code == 401:
            return ApiErrorResolution(
                message="Проверьте API‑ключ и организацию",
                error_type="auth",
                error_code=error_code or "unauthorized",
                retryable=False,
                notify_admin=True,
            )

        if status_code == 403:
            return ApiErrorResolution(
                message="Данная страна/регион не поддерживается",
                error_type="forbidden",
                error_code=error_code or "forbidden",
                retryable=False,
            )

        if status_code == 429:
            quota_hint = "quota_exceeded" in {code, message.lower()}
            text = "Превышен лимит запросов"
            if quota_hint:
                text += ". Пополните баланс в кабинете OpenAI"
            return ApiErrorResolution(
                message=text,
                error_type="rate_limit",
                error_code=error_code or ("quota_exceeded" if quota_hint else "rate_limit"),
                retryable=True,
                cooldown_seconds=self.RATE_LIMIT_PENALTY_SECONDS,
            )

        if status_code in self.SERVER_ERROR_STATUSES:
            return ApiErrorResolution(
                message="Сервис временно недоступен, попробуйте позже",
                error_type="unavailable",
                error_code=error_code or "server_error",
                retryable=False,
            )

        return None

    def apply(
        self,
        *,
        base_error: ProviderAPIError,
        resolution: ApiErrorResolution,
        raw_body: Optional[str] = None,
        fallback_code: Optional[str] = None,
    ) -> ProviderAPIError:
        code = resolution.error_code or fallback_code or base_error.error_code
        return ProviderAPIError(
            provider=base_error.provider,
            status_code=base_error.status_code,
            message=resolution.message,
            error_type=resolution.error_type,
            error_code=code,
            provider_message=raw_body or base_error.provider_message,
            retryable=resolution.retryable,
            duration_ms=base_error.duration_ms,
        )

    def log_admin_hint(self, resolution: ApiErrorResolution) -> None:
        if resolution.notify_admin:
            self._logger.error(
                "%s API reported an authentication issue; administrator attention required",
                self._provider,
            )


__all__ = ["ApiErrorHandler", "ApiErrorResolution"]
