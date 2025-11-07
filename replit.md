# Telegram Bot - Neirosetolog

## Обзор проекта
Телеграм-бот для генерации видео с использованием Gemini API, Sora и других AI провайдеров.

## Последние изменения

### Критическое исправление загрузки моделей Gemini (07.11.2025)
**Проблема**: Бот не запускался - модель `gemini-2.5-flash-image` не обнаруживалась. Загружалось только 20 моделей из `/v1/models` вместо 132 из `/v1beta/models`.

**Корневая причина**: В `providers/gemini.py` метод `_resolve_api_version()` содержал логику:
```python
if task == "image" and model_name.endswith("-image"):
    return "v1"  # ❌ НЕПРАВИЛЬНО! Модели -image находятся в v1beta
```

**Решение**:
1. Удалена ошибочная проверка `model_name.endswith("-image")`
2. Для `task="image"` теперь всегда используется `v1beta`
3. Исправлены LSP ошибки (type: ignore комментарии)

**Результат**:
- ✅ 132 модели загружено (было 20)
- ✅ `gemini-2.5-flash-image` доступна
- ✅ Бот успешно запускается
- ✅ Webhook работает

**Местоположение**: `providers/gemini.py` строки 654-675

### Расширенное детальное логирование Gemini API (06.11.2025)
**Цель**: Отладка 404 "Result not found" и 200 "Модель не вернула результат"

**Добавлено логирование Submit (создание задачи)**:
1. `gemini.submit.response` - тип response, ключи payload, размер, latency
2. `gemini.submit.payload` - полный JSON payload от Gemini API (до 5000 символов, DEBUG level)
3. `gemini.submit.saved_record` - подтверждение сохранения pending record в памяти

**Добавлено логирование Poll (получение результата)**:
4. `gemini.poll.start` - начало poll, проверка наличия record в памяти
5. `gemini.poll.memory_miss` - если record не найден в памяти, загрузка из файла
6. `gemini.poll.not_found` - когда pending record не найден (404 ошибка)
7. `gemini.poll.record_found` - успешное получение record с status_code и duration

**Как использовать**: 
При следующей ошибке проверьте логи на наличие:
- `gemini.submit.response` - что Gemini вернул при создании
- `gemini.submit.saved_record` - был ли record сохранён
- `gemini.poll.start` - был ли record в памяти при poll
- `gemini.poll.not_found` - почему record не найден

**Местоположение**: `providers/gemini.py` строки 1019-1073, 1427-1464

### Настройка окружения (01.11.2025)
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
