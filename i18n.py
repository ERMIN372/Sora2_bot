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
        "main.welcome": (
            "👋 Привет! Я генерирую видео через: {video_models}.\n"
            "Фото — {image_model}.\n"
            "Выберите действие ниже или пришлите запрос текстом.\n\n"
            "Пример: <code>кот идёт по неону и что-то рассказывает, стиль: киберпанк</code>\n"
        ),
        "help.main": (
            "ℹ️ Как пользоваться ботом\n\n"
            "🎬 Видео: {video_models}\n"
            "🖼 Изображения: {image_model}\n\n"
            "1) Сгенерировать по тексту  \n"
            "Опишите идею 1–2 фразами. Пример: <code>кот идёт по неону и что-то рассказывает, стиль: киберпанк</code>  \n"
            "2) Сгенерировать по фото  \n"
            "Пришлите фото и, при желании, подпись-промпт.\n\n"
            "💳 Оплата и кредиты  \n"
            "• 1 видео = 5 кредитов (~129 ₽)  \n"
            "• Пополнение — по ссылке оплаты (ЮKassa), кредиты начисляются автоматически.\n\n"
            "🔁 Статусы  \n"
            "После запуска увидите: «заказ принят», затем «генерирую…», по готовности пришлём ролик.  \n"
            "Если генерация не удалась, 5 кредитов возвращаются.\n\n"
            "📘 Команды  \n"
            "/start — меню  \n"
            "/balance — баланс и пополнение"
        ),
        "buttons.generate_text": "🎬 Сгенерировать видео",
        "buttons.generate_photo": "🖼️ Сгенерировать фото",
        "buttons.balance": "💼 Баланс",
        "buttons.top_up": "💳 Пополнить баланс",
        "buttons.help": "ℹ️ Помощь",
        "buttons.offer": "📄 Оферта",
        "buttons.launch": "🚀 Запустить",
        "buttons.edit": "✏️ Изменить",
        "buttons.back": "↩️ Назад",
        "buttons.top_up_short": "➕ Пополнить",
        "buttons.payment_open": "💳 Перейти к оплате",
        "chips.vertical": "Вертикаль",
        "chips.horizontal": "Горизонталь",
        "video.models.prompt": "Выберите модель видеогенерации:",
        "video.models.sora": "sora 2",
        "video.models.veo3": "Veo 3",
        "video.models.veo31": "Veo 3.1 (Vertex)",
        "video.models.unavailable": "недоступно",
        "video.mode.prompt": "Как будем генерировать видео?",
        "video.mode.text": "По тексту",
        "video.mode.photo": "По фото",
        "image.mode.prompt": "Как будем генерировать изображение?",
        "image.mode.text": "По тексту",
        "image.mode.photo": "По фото",
        "image.model.gemini": "Gemini Image",
        "image.model.unavailable": "недоступно",
        "video.prompt.text": (
            "Опишите, что нужно создать. Текущий кадр: {size}. Модель: {model}.\n"
            "Используйте подсказки ниже или пришлите описание."
        ),
        "video.prompt.photo": (
            "Пришлите фото и, при желании, подпись-промпт.\n"
            "Текущий кадр: {size}. Модель: {model}."
        ),
        "image.prompt.text": (
            "Опишите, что нужно создать. Модель: {model}.\n"
            "Пришлите описание сообщением."
        ),
        "image.prompt.photo": (
            "Пришлите фото и, при желании, подпись-промпт для модели {model}."
        ),
        "video.confirm.text": (
            "✍️ Запрос: {prompt}\n"
            "🎛 Кадр: {size} · Модель: {model}\n"
            "Стоимость: {price}"
        ),
        "video.confirm.photo": (
            "📎 Референс принят.\n"
            "✍️ Запрос: {prompt}\n"
            "🎛 Кадр: {size} · Модель: {model}\n"
            "Стоимость: {price}"
        ),
        "image.confirm.text": (
            "✍️ Запрос: {prompt}\n"
            "🖼 Модель: {model}\n"
            "Стоимость: {price}"
        ),
        "image.confirm.photo": (
            "📎 Референс принят.\n"
            "✍️ Запрос: {prompt}\n"
            "🖼 Модель: {model}\n"
            "Стоимость: {price}"
        ),
        "flow.price_tag": "{credits} (~{approx})",
        "flow.no_prompt": "Добавьте описание. Пример: <code>кот идёт по неону, киберпанк</code>",
        "flow.order_submitted": "🧾 Заказ принят. Место в очереди: {queue_pos}. Пришлём видео сюда.",
        "flow.duplicate": "Задача уже в очереди.",
        "flow.not_enough": "Недостаточно кредитов. Выберите пакет ниже.",
        "status.running": "⚙️ Генерирую…",
        "status.completed_file": "✅ Готово. Отправляю ролик.",
        "status.completed_photo": "✅ Готово. Отправляю фото.",
        "status.completed_url": "Готово: {url}",
        "status.inline_too_large": (
            "Файл от модели ({mime}, {size} байт) превышает лимит Telegram ({limit_mb} МБ)."
        ),
        "status.inline_decode_failed": "Не удалось обработать файл от модели.",
        "status.failed": "Не удалось создать. Ошибка {error}.",
        "status.generic": "Статус задачи: {status}.",
        "errors.video_models_unavailable": "Нет доступных моделей видеогенерации. Обратитесь к оператору или попробуйте позже.",
        "errors.image_generation_disabled": "Генерация изображений недоступна. Обратитесь к оператору или попробуйте позже.",
        "errors.unknown": "неизвестная ошибка",
        "balance.info": "Ваш баланс: {credits} кредит(ов).",
        "payment.unavailable": "Оплата картой временно недоступна.",
        "payment.packages.title": "Доступные пакеты:",
        "payment.packages.line": "{credits} — {price}",
        "payment.store.instructions": "Перейдите по ссылке для оплаты. После оплаты кредиты начислятся автоматически.",
        "payment.terms_notice": "«Оплачивая, вы принимаете условия Оферты: {terms_url}»",
        "payment.unknown_package": "Неизвестный пакет оплаты.",
        "payment.creation_failed": "Не удалось создать платёж. Попробуйте позже.",
        "payment.link_failed": "Не удалось получить ссылку оплаты. Попробуйте позже.",
        "payment.received": "Оплата получена. Зачислено: {items}. Спасибо!",
        "web.return.unavailable": "<h1>Сервис временно недоступен</h1>",
        "web.return.not_found": "<h1>Платёж не найден</h1>",
        "web.return.success": "<h1>Оплата принята</h1><p>Спасибо за покупку!</p>",
        "web.return.canceled": "<h1>Оплата отменена</h1><p>Средства не списаны.</p>",
        "web.return.pending": "<h1>Платёж обрабатывается</h1><p>Проверьте баланс позже.</p>",
        "errors.pro_unavailable": "Pro-версия недоступна. Используется Sora 2.",
    }
}


i18n = _Translator(TRANSLATIONS)
