# Telegram Bot - Neirosetolog

## Обзор проекта
Телеграм-бот для генерации видео с использованием Gemini API, Sora и других AI провайдеров.

## Последние изменения (01.11.2025)

### Настройка окружения
- **Python версия**: 3.11
- **Режим работы**: polling (по умолчанию)
- **Workflow**: `telegram-bot` - запускает `python main.py --mode polling`

### Исправление проблемы с git reset (31.10.2025)
**Проблема**: После выполнения `git fetch origin && git reset --hard origin/main` удалялся файл `.replit` и все настройки, включая workflow.

**Решение**:
1. Удалили `.replit` и `requirements.txt` из `.gitignore`
2. Теперь эти файлы сохраняются в репозитории и не удаляются при `git reset --hard`
3. Workflow автоматически восстанавливается после перезагрузки проекта

### Исправление совместимости с новым OpenAI Video API (01.11.2025)
**Проблема**: OpenAI изменил API для генерации видео (Sora), убрав поддержку параметров `"n"` и `"response_format"`.

**Решение**:
1. Удалили неподдерживаемый параметр `"n": 1` из `providers/openai_video.py` (строка 988)
2. Удалили неподдерживаемый параметр `"response_format": "url"` из `providers/openai_video.py` (строка 991)
3. Теперь запросы к OpenAI Video API содержат только поддерживаемые параметры: `model`, `prompt`, `size`, `duration`

## Архитектура проекта

### Основные компоненты
- **main.py** - точка входа, управляет запуском бота в режиме polling или webhook
- **config.py** - конфигурация из переменных окружения
- **handlers.py** - обработчики команд Telegram
- **jobs.py** - очередь задач для генерации
- **providers/** - клиенты для разных AI провайдеров (Gemini, OpenAI, Sora, Veo)

### База данных
- **SQLite** (локально): `./bot.db`
- **Google Sheets**: используется для постоянного хранения (users, payments, jobs, errors)

### Режимы работы
1. **polling** (текущий) - бот сам опрашивает Telegram API
2. **webhook** - Telegram отправляет обновления на webhook URL

## Переменные окружения (ключевые)
- `TELEGRAM_BOT_TOKEN` - токен бота
- `BOT_MODE` - режим работы (`polling` или `webhook`)
- `GEMINI_API_KEY` - ключ для Gemini API
- `GOOGLE_SHEET_ID` - ID таблицы Google Sheets для хранения данных
- `GOOGLE_SA_JSON_BASE64` - закодированные credentials для Google Service Account

## Пользовательские предпочтения
- Предпочтительный режим работы: **polling**
- При проблемах с запуском всегда проверять наличие Python модуля
- Не удалять `.replit` из репозитория

## Запуск проекта
После клонирования или git reset проект запускается автоматически через workflow `telegram-bot`.

Для ручного запуска:
```bash
python main.py --mode polling
```

## Важные заметки
- После `git reset --hard origin/main` может потребоваться переустановка Python модуля, если его нет в `.replit` файле
- Workflow автоматически перезапускается при изменении конфигурации
- Логи доступны через панель Replit
