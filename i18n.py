"""Simple localisation and HTML escaping helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "escape_html",
    "SafeText",
    "format_prompt",
    "i18n",
    "format_credits",
]


_AMP_PATTERN = re.compile(r"&(?![#a-zA-Z0-9]+;)")


def escape_html(value: Any) -> str:
    """Escape Telegram HTML special characters in *value* without double escaping."""

    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    text = _AMP_PATTERN.sub("&amp;", text)
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    return text


class SafeText(str):
    """Marker type for HTML-safe snippets that should not be escaped again."""

    __slots__ = ()


def _safe(value: Any) -> str:
    if isinstance(value, SafeText):
        return str(value)
    return escape_html(value)


def format_prompt(prompt: Optional[str]) -> SafeText:
    """Return an HTML-safe representation of *prompt* wrapped for Telegram."""

    if not prompt:
        return SafeText("<code>—</code>")
    escaped = escape_html(prompt)
    if "\n" in prompt:
        return SafeText(f"<pre>{escaped}</pre>")
    return SafeText(f"<code>{escaped}</code>")


def format_credits(amount: Any) -> str:
    """Return *amount* followed by a grammatically correct credit word."""

    try:
        value = int(amount)
    except (TypeError, ValueError):
        return escape_html(amount)
    mod10 = value % 10
    mod100 = value % 100
    if mod10 == 1 and mod100 != 11:
        word = "кредит"
    elif 2 <= mod10 <= 4 and not 12 <= mod100 <= 14:
        word = "кредита"
    else:
        word = "кредитов"
    return f"{value} {word}"


@dataclass(slots=True)
class _Translator:
    translations: Mapping[str, Mapping[str, str]]
    default_locale: str = "ru"

    def t(self, key: str, *, locale: Optional[str] = None, **kwargs: Any) -> str:
        lang = locale or self.default_locale
        table = self.translations.get(lang)
        if table is None:
            raise KeyError(f"Unknown locale: {lang}")
        if key not in table:
            raise KeyError(f"Missing translation key: {key}")
        template = table[key]
        if not kwargs:
            return template
        safe_kwargs = {name: _safe(value) for name, value in kwargs.items()}
        return template.format(**safe_kwargs)


TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "ru": {
        "start.welcome": (
            "👋 Привет! Это бот Sora.\n"
            "Используйте команду <code>/generate</code>, чтобы создать видео.\n"
            "Пример: <code>/generate &lt;prompt size=&quot;720p&quot; seconds=&quot;10&quot;&gt;котята играют&lt;/prompt&gt;</code>\n"
            "Баланс и бонус: /balance. Пополнение: Telegram Stars или /buy_card."
        ),
        "generate.prompt_required": (
            "Пожалуйста, укажите описание: <code>/generate &lt;prompt&gt;текст&lt;/prompt&gt;</code>."
        ),
        "generate.too_many_jobs": (
            "Слишком много активных задач (максимум {limit}). Дождитесь завершения текущих генераций."
        ),
        "generate.not_enough_credits": (
            "Недостаточно кредитов. Пополните баланс через Telegram Stars или команду /buy_card."
        ),
        "generate.queued": "Задача поставлена в очередь.\nЗапрос: {prompt}",
        "generate.submission_failed": "Не удалось отправить задачу: {error}. Кредит возвращён.",
        "balance.info": "Ваш баланс: {credits} кредит(ов). Бонус за подписку: {bonus_status}.",
        "bonus.status.received": "получен",
        "bonus.status.not_received": "не получен",
        "bonus.granted": "Начислен бонус +2",
        "bonus.already": "Бонус уже начислялся",
        "payment.stars.received": "✨ Платёж получен! Зачислено {credits} кредит(ов).",
        "buy_card.unavailable": "Оплата картой временно недоступна. Попробуйте позже.",
        "buy_card.choose_package": "Выберите пакет кредитов для оплаты картой:",
        "buy_card.button": "{items} кредит(ов) — {price}₽",
        "buy_card.unknown_package": "Неизвестный пакет оплаты.",
        "buy_card.invoice_description": "Покупка {items} кредит(ов) Sora2",
        "buy_card.pay_button": "Оплатить картой",
        "buy_card.payment_creation_failed": "Не удалось создать платёж. Попробуйте позже.",
        "buy_card.payment_link_failed": "Не удалось получить ссылку оплаты. Попробуйте позже.",
        "buy_card.payment_link": "Оплатите по ссылке, после оплаты вернитесь — баланс обновится автоматически.",
        "job.completed_with_url": "🎉 Готово: {url}\nЗапрос: {prompt}",
        "job.completed_no_url": "🎉 Готово, отправляю ролик.\nЗапрос: {prompt}",
        "job.failed": "⚠️ Не удалось создать видео: {error}. Кредит возвращён.",
        "job.status.queued": "⏳ Задача поставлена в очередь.",
        "job.status.running": "⏳ Генерирую…",
        "job.status.generic": "⏳ Задача {job_id}: {status}.",
        "errors.unknown": "неизвестная ошибка",
        "payment.yookassa.received": "Зачислено {credits} кредит(ов). Спасибо!",
        "web.return.unavailable": "<h1>Сервис временно недоступен</h1>",
        "web.return.not_found": "<h1>Платёж не найден</h1>",
        "web.return.success": "<h1>Оплата принята</h1><p>Спасибо за покупку!</p>",
        "web.return.canceled": "<h1>Оплата отменена</h1><p>Средства не списаны.</p>",
        "web.return.pending": "<h1>Платёж обрабатывается</h1><p>Проверьте баланс позже.</p>",
    }
}


i18n = _Translator(TRANSLATIONS)
