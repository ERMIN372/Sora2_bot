# Pull Request: Fix bot deployment and media generation issues

## Описание

Этот PR исправляет критические проблемы, блокировавшие деплой бота на Railway и генерацию медиа (видео/изображений).

## Проблемы, которые были исправлены

### 🔴 Критическая ошибка: TypeError при запуске на Railway
**Симптомы:**
```
TypeError: can't compare offset-naive and offset-aware datetimes
at jobs.py:1446
```

**Причина:** PostgreSQL возвращает timezone-aware datetime объекты, но код использовал timezone-naive datetime (через `datetime.utcnow()` и `datetime.utcfromtimestamp()`). При сравнении этих объектов Python выбрасывал TypeError.

**Решение:**
- Заменены все `datetime.utcnow()` → `datetime.now(timezone.utc)` (22 места в 6 файлах)
- Заменены `datetime.utcfromtimestamp()` → `datetime.fromtimestamp(tz=timezone.utc)` (2 места)
- Улучшена функция `parse_datetime()` для автоматического добавления UTC timezone к naive datetime объектам из БД

### 📝 Отсутствие документации по деплою
**Проблема:** Не было инструкций по настройке и запуску бота, что приводило к ошибкам конфигурации.

**Решение:**
- Создан `DEPLOYMENT.md` с полным руководством по развертыванию
- Обновлен `README.md` с секцией быстрого старта и критическими предупреждениями
- Добавлены инструкции по устранению типовых проблем

### 🎨 Устаревшие метки UI
**Проблема:** В интерфейсе отображалось "Veo 2" вместо актуальной версии "Veo 3".

**Решение:** Обновлены строки локализации в `i18n.py`.

## Изменения по файлам

### Коммит 1: `fbf017c` - Документация и UI
- ✅ **Добавлено:** `DEPLOYMENT.md` - полное руководство по развертыванию
- ✅ **Обновлено:** `README.md` - быстрый старт с важными предупреждениями
- ✅ **Исправлено:** `i18n.py` - метки "Veo 2" → "Veo 3"

### Коммит 2: `83a5bc4` - Timezone-aware datetime
Исправлено 22 места в 6 файлах:
- ✅ **jobs.py** (7 мест) - восстановление pending jobs, логирование ошибок
- ✅ **handlers.py** (9 мест) - обработчики команд бота
- ✅ **app_server.py** (1 место) - генерация ISO timestamp
- ✅ **archive.py** (2 места) - архивирование медиа
- ✅ **gsheets_db.py** (1 место) - timestamp для Google Sheets
- ✅ **providers/gemini.py** (2 места) - debug информация

### Коммит 3: `49046ca` - Исправление parse_datetime()
- ✅ **db/models.py** - улучшена функция `parse_datetime()`:
  - Автоматически добавляет UTC timezone к naive datetime
  - Обрабатывает datetime объекты напрямую из PostgreSQL
  - Фиксирует `_MAX_TS_DEFAULT` как timezone-aware

## Технические детали

**До исправления:**
```python
# ❌ Timezone-naive datetime
stale_cutoff = datetime.utcnow() - timedelta(hours=12)
epoch = datetime.utcfromtimestamp(0)
# PostgreSQL возвращает timezone-aware → TypeError при сравнении
```

**После исправления:**
```python
# ✅ Timezone-aware datetime
stale_cutoff = datetime.now(timezone.utc) - timedelta(hours=12)
epoch = datetime.fromtimestamp(0, tz=timezone.utc)
# Все datetime объекты теперь timezone-aware (UTC)
```

## Тестирование

### Локальное тестирование
```bash
# 1. Создать .env файл
cp .env.example .env
# Установить TELEGRAM_BOT_TOKEN и GEMINI_API_KEY

# 2. Запустить через Docker Compose
docker-compose up -d

# 3. Проверить healthcheck
curl http://localhost:8080/healthz
```

### Railway deployment
- ✅ Бот успешно стартует без TypeError
- ✅ Healthcheck проходит
- ✅ Генерация медиа работает

## Обратная совместимость

✅ Изменения полностью обратно совместимы:
- Существующие данные в БД не требуют миграции
- Функция `parse_datetime()` корректно обрабатывает как timezone-aware, так и timezone-naive datetime
- Все API эндпоинты работают без изменений

## Чеклист

- [x] Код протестирован локально
- [x] Исправлены все timezone comparison errors
- [x] Добавлена документация по развертыванию
- [x] Обновлены UI метки
- [x] Все изменения запушены в ветку
- [x] Коммиты содержат подробные описания

## Связанные issues

Fixes: Railway deployment crash - "TypeError: can't compare offset-naive and offset-aware datetimes"
Resolves: Отсутствие документации по деплою
Resolves: Устаревшие метки "Veo 2" в UI

## Дополнительные материалы

- `DEPLOYMENT.md` - полное руководство по настройке и устранению проблем
- Все datetime объекты теперь явно используют UTC timezone
- Совместимость с PostgreSQL TIMESTAMP WITH TIME ZONE

---

**Статистика:**
- 📝 8 файлов изменено
- ➕ 273 строки добавлено (включая документацию)
- ➖ 30 строк удалено
- 🔧 3 коммита
- ⏱️ 22 места с timezone fixes
