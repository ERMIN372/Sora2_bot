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
            "👋 Привет! Я генерирую:\n"
            "🎬 Видео — veo3 — 89 ₽, OpenAI Sora 2 — от 99 ₽.\n"
            "🖼️ Фото — Nano Banana — 5 ₽ за 1 изображение, DALL·E 3 — 5 ₽ за 1 изображение.\n"
            "💬 Со мной также можно просто поболтать или спросить что-нибудь интересное через ChatGPT\n\n"
            "🎯 Выберите действие ниже и пришлите запрос.\n\n"
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
            "3) ChatGPT  \n"
            "Нажмите «ChatGPT», чтобы задать вопрос модели GPT-4.1 и получить ответ.\n\n"
            "💳 Оплата и кредиты  \n"
            "• OpenAI Sora 2 — от 99 ₽ за видео  \n"
            "• veo3 — 89 ₽ за видео  \n"
            "• Фото: Nano Banana или DALL·E 3 — 5 ₽ за 1 изображение  \n"
            "• Пополнение: 100₽ (0%), 300₽ (+5%), 700₽ (+8%), 1500₽ (+12%), 3000₽ (+15%) — YooKassa, бонус начисляется автоматически.\n\n"
            "🔁 Статусы  \n"
            "После запуска увидите: «заказ принят», затем «генерирую…», по готовности пришлём ролик.  \n"
            "Если генерация не удалась, стоимость возвращается.\n\n"
            "📘 Команды  \n"
            "/start — меню  \n"
            "/balance — баланс и пополнение"
        ),
        "buttons.generate_text": "🎬 Сгенерировать видео",
        "buttons.generate_photo": "🖼️ Сгенерировать фото",
        "buttons.balance": "💼 Баланс",
        "buttons.top_up": "💳 Пополнить баланс",
        "buttons.help": "ℹ️ Помощь",
        "buttons.chatgpt": "ChatGPT",
        "buttons.video_create": "/video_create",
        "buttons.video_remix": "/video_remix",
        "buttons.video_list": "/video_list",
        "buttons.video_info": "/video_info",
        "buttons.video_get": "/video_get",
        "buttons.video_models": "/video_models",
        "buttons.offer": "📄 Оферта",
        "buttons.launch": "🚀 Запустить",
        "buttons.edit": "✏️ Изменить",
        "buttons.back": "↩️ Назад",
        "buttons.top_up_short": "➕ Пополнить",
        "buttons.payment_open": "💳 Перейти к оплате",
        "chips.vertical": "Вертикаль",
        "chips.horizontal": "Горизонталь",
        "admin.menu.title": "Админ-панель. Выберите действие:",
        "admin.menu.broadcast_all": "📢 Рассылка всем",
        "admin.menu.direct_message": "🎯 Сообщение пользователю",
        "admin.menu.close": "⬅️ Закрыть админку",
        "admin.menu.invalid": "Пожалуйста, воспользуйтесь кнопками меню.",
        "admin.menu.closed": "Админ-панель закрыта.",
        "admin.broadcast.prompt": (
            "Отправьте ТЕКСТ или ОДНО сообщение с вложением (фото/видео/документ/голосовое/стикер)"
            " и подписью. Мы разошлём ровно это сообщение всем пользователям."
        ),
        "admin.broadcast.empty_message": "Сообщение не должно быть пустым.",
        "admin.broadcast.empty_audience": "Нет пользователей для рассылки.",
        "admin.broadcast.accepted": "Ок, разошлю сообщение {count} пользователям. Начинаю рассылку…",
        "admin.broadcast.media_group_unsupported": "Альбомы пока не поддерживаются. Отправьте одно сообщение с медиа и подписью.",
        "admin.broadcast.unsupported_service": "Этот тип сообщения не поддерживается для рассылки. Отправьте обычный текст или одно медиа.",
        "admin.broadcast.unsupported_type": "Telegram не разрешил переслать это сообщение. Попробуйте другой тип контента.",
        "admin.broadcast.started": "Рассылка запущена. Получателей: {count}.",
        "admin.broadcast.completed": "Рассылка завершена. Успешно: {sent}, заблокировано: {blocked}, ошибок: {failed}.",
        "admin.buttons.cancel": "❌ Отмена",
        "admin.buttons.send": "✅ Отправить",
        "admin.buttons.abort": "↩️ Назад",
        "admin.status.cancelled": "Действие отменено.",
        "admin.status.internal_error": "Не удалось выполнить действие. Попробуйте позже.",
        "admin.direct.prompt_id": "Укажите ID или @username получателя.",
        "admin.direct.lookup_error": "Не удалось загрузить список пользователей. Попробуйте позже.",
        "admin.direct.not_found": "Пользователь не найден.",
        "admin.direct.prompt_message": "Отправьте текст сообщения для {user}.",
        "admin.direct.empty_message": "Сообщение не должно быть пустым.",
        "admin.direct.preview": "Отправить сообщение для {user}:\n\n{message}",
        "admin.direct.sent": "Сообщение отправлено для {user}.",
        "admin.direct.failed": "Не удалось отправить сообщение для {user}.",
        "video.models.prompt": "Выберите модель видеогенерации:",
        "video.models.veo": "veo3",
        "video.models.sora": "OpenAI Sora 2",
        "video.models.unavailable": "недоступно",
        "video.models.disabled": "OpenAI Sora 2 недоступна в конфигурации.",
        "video.models.fetch_failed": "Список моделей Sora 2 временно недоступен.",
        "video.models.empty": "API не вернул список моделей Sora 2.",
        "video.models.header": "Доступные модели Sora 2:",
        "video.common.init_failed": "Не удалось инициализировать OpenAI Videos API.",
        "video.create.disabled": "Видео через OpenAI Sora недоступно.",
        "video.create.prompt_hint": "Укажите описание. Пример: /video_create {\"prompt\": \"кот идёт по неону\"}",
        "video.common.send_failed": "Не удалось отправить запрос: {error}",
        "video.create.missing_job_id": "API не вернул идентификатор задачи. Повторите попытку позднее.",
        "video.common.response_snippet": "Ответ: {snippet}",
        "video.create.registration_failed": "Не удалось зарегистрировать задачу: {error}",
        "video.create.enqueued": "Задача отправлена в очередь.",
        "video.create.model_unavailable": "Модель {requested} недоступна. Используем {fallback}.",
        "video.common.job_id": "job_id: {job_id}",
        "video.common.video_id": "video_id: {video_id}",
        "video.common.corr_id": "corr_id: {corr_id}",
        "video.common.model": "модель: {model}",
        "video.common.credits_deducted": "Списано: {credits} ₽",
        "video.common.invalid_json": "Не удалось разобрать параметры. Передайте корректный JSON.",
        "video.common.expected_object": "Ожидался JSON-объект с параметрами.",
        "video.remix.usage": "Использование: /video_remix <video_id> {\"prompt\": \"описание\"}",
        "video.remix.missing_video_id": "Укажите video_id исходного ролика.",
        "video.remix.missing_job_id": "API не вернул идентификатор ремикса.",
        "video.remix.registration_failed": "Не удалось зарегистрировать ремикс: {error}",
        "video.remix.started": "Ремикс запущен.",
        "video.list.disabled": "OpenAI Sora 2 недоступна.",
        "video.list.fetch_failed": "Не удалось получить список: {error}",
        "video.list.status": "Статус: {status} · время: {duration} мс",
        "video.list.header": "Видео:",
        "video.list.empty": "Видео не найдены.",
        "video.list.entry_created": " · {created}",
        "video.list.entry": "• {video_id} · {status}{created_label}",
        "video.list.more": "… и ещё {count}",
        "video.info.usage": "Использование: /video_info <video_id>",
        "video.info.fetch_failed": "Не удалось получить информацию: {error}",
        "video.info.header": "Видео: {video_id}",
        "video.info.status": "Статус запроса: {status} · {duration} мс",
        "video.info.state": "Статус: {status}",
        "video.info.created": "Создано: {created}",
        "video.get.usage_admin": "Использование: /video_get <video_id> [format]",
        "video.get.admin_only": "Команда доступна только администраторам.",
        "video.get.provider_missing": "Клиент OpenAI Videos не настроен.",
        "video.get.error": "Не удалось скачать видео (status={status}, code={code}): {message}",
        "video.get.read_error": "Файл скачан, но не удалось прочитать: {error}",
        "video.get.sent": "Видео {video_id} отправлено. Размер: {size} байт. Провайдер: {provider}.",
        "video.mode.prompt": "Как будем генерировать видео?",
        "video.mode.text": "По тексту",
        "video.mode.photo": "По фото",
        "image.mode.prompt": "Как будем генерировать изображение?",
        "image.mode.text": "По тексту",
        "image.mode.photo": "По фото",
        "image.models.prompt": "Выберите модель генерации фото:",
        "image.model.gemini": "Nano Banana — 5 ₽ за 1 изображение",
        "image.model.dalle3": "DALL·E 3 — 5 ₽ за 1 изображение",
        "image.model.unavailable": "недоступно",
        "video.prompt.text": (
            "В этом разделе нужно задать настройки для видео с помощью {model}.\n"
            "1. Опишите ролик в сообщении.\n"
            "2. Прикрепите изображение, если нужно.\n"
            "3. Выберите длительность и соотношение сторон ниже.\n\n"
            "Текущие настройки: {duration} сек · {aspect}\n"
            "Стоимость: {price}"
        ),
        "video.prompt.text_veo": (
            "В этом разделе нужно задать настройки для видео с помощью {model}.\n"
            "1. Опишите ролик в сообщении.\n"
            "2. Прикрепите изображение, если нужно.\n"
            "3. Выберите соотношение сторон ниже.\n\n"
            "Длительность: {duration} сек\n"
            "Стоимость: {price}"
        ),
        "video.prompt.photo": (
            "Пришлите фото и, при желании, подпись-промпт для модели {model}.\n"
            "Задайте длительность и формат на клавиатуре ниже.\n\n"
            "Текущие настройки: {duration} сек · {aspect} · {quality}\n"
            "Стоимость: {price}"
        ),
        "video.prompt.photo_veo": (
            "Пришлите фото и, при желании, подпись-промпт для модели {model}.\n"
            "Выберите соотношение сторон на клавиатуре ниже.\n\n"
            "Длительность: {duration} сек\n"
            "Стоимость: {price}"
        ),
        "video.prompt.quality_hd": "Качество: HD",
        "video.prompt.quality_sd": "Качество: SD",
        "video.prompt.current_prompt": "Текущий промпт: отправьте текст",
        "video.prompt.current_image": "Изображение: не добавлено",
        "video.prompt.start_hint": "Пришлите описание или фото, чтобы продолжить.",
        "image.prompt.text": (
            "Опишите, что нужно создать. Модель: {model}.\n"
            "Пришлите описание сообщением."
        ),
        "image.prompt.photo": (
            "Пришлите фото и, при желании, подпись-промпт для модели {model}."
        ),
        "video.confirm.text": (
            "✍️ Запрос: {prompt}\n"
            "🎛 Формат: {duration} сек · {aspect} · {quality}\n"
            "🎞 Кадр: {size} · Модель: {model}\n"
            "Стоимость: {price}"
        ),
        "video.confirm.photo": (
            "📎 Референс принят.\n"
            "✍️ Запрос: {prompt}\n"
            "🎛 Формат: {duration} сек · {aspect} · {quality}\n"
            "🎞 Кадр: {size} · Модель: {model}\n"
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
        "flow.order_submitted_image": "🧾 Заказ принят. Место в очереди: {queue_pos}. Пришлём фото сюда.",
        "flow.nano_image_progress": "Генерация изображения…",
        "flow.nano_image_done": "Готово!",
        "flow.duplicate": "Задача уже в очереди.",
        "flow.not_enough": "Недостаточно кредитов. Выберите пакет ниже.",
        "status.running": "⚙️ Генерирую…",
        "status.completed_file": "✅ Готово. Отправляю ролик.",
        "status.completed_photo": "✅ Готово. Отправляю фото.",
        "status.completed_url": "Готово: {url}",
        "status.completed_video_summary": "🎬 Готово. Длительность: {duration} с. Размер: ~{size_mb} МБ.",
        "status.completed_video_summary_short": "🎬 Готово. Размер: ~{size_mb} МБ.",
        "status.completed_image_summary": "🖼 Готово. Размер: ~{size_mb} МБ.",
        "status.inline_too_large": (
            "Файл от модели ({mime}, {size} байт) превышает лимит Telegram ({limit_mb} МБ)."
        ),
        "status.inline_decode_failed": "Не удалось обработать файл от модели.",
        "status.failed": "Не удалось создать. Кредиты вернули. {error}",
        "status.generic": "Статус задачи: {status}.",
        "status.delivery_key_mismatch": "Остановили задачу: используется другой ключ Gemini. Кредиты вернули.",
        "status.delivery_config_error": "Gemini временно недоступен. Мы вернули кредиты — попробуйте позже.",
        "status.delivery_generic": "Не удалось скачать видео. Мы вернули кредиты — попробуйте позже.",
        "status.delivery_telegram_error": "Не удалось отправить видео в Telegram: {error}. Кредиты вернули.",
        "status.delivery_too_large": "Видео слишком большое для Telegram. Кредиты вернули.",
        "errors.video_models_unavailable": "Нет доступных моделей видеогенерации. Обратитесь к оператору или попробуйте позже.",
        "errors.image_generation_disabled": "Генерация изображений недоступна. Обратитесь к оператору или попробуйте позже.",
        "errors.unknown": "неизвестная ошибка",
        "errors.chatgpt_unavailable": "ChatGPT временно недоступен. Попробуйте позже.",
        "errors.chatgpt_failed": "Не удалось получить ответ от GPT-4.1. Попробуйте ещё раз.",
        "errors.chatgpt_empty": "Отправьте текстовый запрос для GPT-4.1.",
        "balance.info": "Ваш баланс: {credits} кредит(ов).",
        "payment.unavailable": "Оплата картой временно недоступна.",
        "payment.packages.title": (
            "Пополните баланс (1 кредит = 1 ₽)\n\n"
            "• 100 ₽ → 100 кредитов (без бонуса)\n"
            "• 300 ₽ → 315 кредитов (+5%)\n"
            "• 700 ₽ → 756 кредитов (+8%)\n"
            "• 1 500 ₽ → 1 680 кредитов (+12%)\n"
            "• 3 000 ₽ → 3 450 кредитов (+15%)\n\n"
            "После оплаты кредиты начисляются автоматически.\n"
            "Оплачивая, вы принимаете условия Оферты{terms}."
        ),
        "payment.packages.line": "{price} → {total} (бонус {bonus} · +{bonus_pct}%)",
        "payment.store.instructions": "Перейдите по ссылке для оплаты. После оплаты кредиты начислятся автоматически.",
        "payment.terms_notice": "«Оплачивая, вы принимаете условия Оферты: {terms_url}»",
        "payment.unknown_package": "Неизвестный пакет оплаты.",
        "payment.creation_failed": "Не удалось создать платёж. Попробуйте позже.",
        "payment.link_failed": "Не удалось получить ссылку оплаты. Попробуйте позже.",
        "payment.received": "Оплата получена. Зачислено: {items}. Спасибо!",
        "help.chatgpt": "Спросите GPT-4.1 любой вопрос, и я отвечу.",
        "web.return.unavailable": "<h1>Сервис временно недоступен</h1>",
        "web.return.not_found": "<h1>Платёж не найден</h1>",
        "web.return.success": "<h1>Оплата принята</h1><p>Спасибо за покупку!</p>",
        "web.return.canceled": "<h1>Оплата отменена</h1><p>Средства не списаны.</p>",
        "web.return.pending": "<h1>Платёж обрабатывается</h1><p>Проверьте баланс позже.</p>",
        "errors.pro_unavailable": "Pro-версия недоступна. Используется Veo 2.",
    }
}


i18n = _Translator(TRANSLATIONS)
