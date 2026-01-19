# Руководство по развертыванию Sora2_bot

## Быстрый старт

### 1. Настройка переменных окружения

Создайте файл `.env` на основе `.env.example`:

```bash
cp .env.example .env
```

### 2. Обязательные настройки

Отредактируйте `.env` и установите следующие **критически важные** параметры:

#### Telegram Bot Token (ОБЯЗАТЕЛЬНО)
```env
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz123456789
```
Получите токен у [@BotFather](https://t.me/BotFather) в Telegram.

#### Gemini API Key (ОБЯЗАТЕЛЬНО для генерации видео/изображений)
```env
GEMINI_API_KEY=AIza...your-api-key
```
Получите ключ на [Google AI Studio](https://aistudio.google.com/app/apikey).

#### Модели Gemini (уже настроены правильно)
```env
GEMINI_MODEL_TEXT=gemini-2.0-flash
GEMINI_MODEL_IMAGE=gemini-2.5-flash-image
GEMINI_MODEL_VIDEO=veo-3.0-generate-001
```
⚠️ **Важно**: Используйте `veo-3.0-generate-001` или новее, НЕ `veo-2.0-generate-001`!

#### База данных (необязательно для локального запуска)
Для production настройте PostgreSQL или Google Sheets:

**Вариант 1: Google Sheets (проще для старта)**
```env
GOOGLE_SHEET_ID=your-spreadsheet-id
GOOGLE_SA_JSON_BASE64=base64-encoded-service-account-json
```

**Вариант 2: PostgreSQL (рекомендуется для production)**
```env
POSTGRES_ENABLED=true
DATABASE_URL=postgresql://user:password@localhost:5432/sora
```

#### YooKassa (для приема платежей)
```env
YOOKASSA_SHOP_ID=your-shop-id
YOOKASSA_SECRET_KEY=your-secret-key
YOOKASSA_TEST_MODE=true
PUBLIC_BASE_URL=https://your-domain.com
```

### 3. Запуск через Docker Compose (рекомендуется)

```bash
# Билд и запуск всех сервисов
docker-compose up -d

# Проверка логов
docker-compose logs -f bot

# Проверка healthcheck
curl http://localhost:8080/healthz
```

Сервисы:
- `bot` - основной бот (порт 8080)
- `postgres` - база данных (порт 5432)
- `redis` - хранилище состояний (порт 6379)
- `prometheus` - метрики (порт 9090)

### 4. Запуск в режиме разработки

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск миграций (если используется PostgreSQL)
psql -f migrations/001_initial_schema.sql

# Запуск бота
python main.py
```

## Решение проблем

### Проблема: Бот не запускается

**Симптомы:**
```
KeyError: 'TELEGRAM_BOT_TOKEN'
```

**Решение:**
1. Убедитесь, что файл `.env` создан и содержит `TELEGRAM_BOT_TOKEN`
2. Проверьте, что токен валиден через [@BotFather](https://t.me/BotFather)

### Проблема: Healthcheck падает с ошибкой "model not found"

**Симптомы:**
```
models/gemini-2.5-flash-image not found
models/veo-2.0-generate-001 not found
```

**Решение:**
Обновите `.env`:
```env
GEMINI_MODEL_VIDEO=veo-3.0-generate-001  # НЕ veo-2.0!
```

### Проблема: Видео не генерируются (400 INVALID_ARGUMENT)

**Симптомы:**
```
400 INVALID_ARGUMENT: "resolution" isn't supported by this model
```

**Решение:**
Эта проблема уже исправлена в коде. Убедитесь, что используете последнюю версию из ветки `claude/fix-bot-deployment-media-90o70`.

### Проблема: "Нет доступных моделей видеогенерации"

**Причины:**
1. `GEMINI_API_KEY` не установлен или невалиден
2. `GEMINI_MODEL_VIDEO` пустой или содержит несуществующую модель

**Решение:**
```env
GEMINI_API_KEY=your-valid-api-key
GEMINI_MODEL_VIDEO=veo-3.0-generate-001
```

### Проблема: Docker контейнер постоянно перезапускается

**Проверка:**
```bash
docker-compose logs bot
```

**Частые причины:**
1. Отсутствует `.env` файл → создайте из `.env.example`
2. Неверные credentials в БД → проверьте `DATABASE_URL`
3. Healthcheck падает → проверьте `/healthz` endpoint

## Режимы работы

### Polling (для разработки)
```env
BOT_MODE=polling
```
Бот сам опрашивает Telegram API. Не требует публичного URL.

### Webhook (для production)
```env
BOT_MODE=webhook
WEBHOOK_HOST=https://your-domain.com
TG_WEBHOOK_SECRET=generate-random-32-chars
PUBLIC_BASE_URL=https://your-domain.com
```
Telegram отправляет события на ваш сервер. Требует HTTPS и публичный URL.

## Мониторинг

### Healthcheck
```bash
curl http://localhost:8080/healthz
```

Возвращает статус всех компонентов:
- Gemini API connectivity
- Database connectivity
- Redis connectivity
- Model availability

### Prometheus Metrics
```bash
curl http://localhost:8080/metrics
```

Доступные метрики:
- `bot_requests_total` - количество запросов
- `bot_request_duration_seconds` - время обработки
- `bot_errors_total` - количество ошибок
- `generation_jobs_total` - количество генераций

Grafana dashboard доступен на `http://localhost:9090` (Prometheus UI).

## Производительность

### Рекомендуемые настройки для production

```env
# Параллельная обработка заданий
JOBS_CONCURRENCY=5
MAX_JOBS_PER_USER=3

# Таймауты
REQUEST_TIMEOUT=60.0
PROVIDER_TIMEOUT_S=120.0
REQUEST_RETRIES=3

# Rate limiting
RATE_LIMITS=video_create:10:60,image:20:60
```

### Масштабирование

Для высоких нагрузок:
1. Увеличьте `JOBS_CONCURRENCY` (но следите за лимитами API)
2. Используйте PostgreSQL вместо Google Sheets
3. Настройте connection pooling для БД
4. Добавьте несколько инстансов бота за load balancer

## Безопасность

### Обязательно:
- ✅ Используйте сильные пароли для БД
- ✅ Храните `.env` в `.gitignore` (уже настроено)
- ✅ Используйте HTTPS для webhook режима
- ✅ Регулярно ротируйте API ключи
- ✅ Включите rate limiting

### Не делайте:
- ❌ Не коммитьте `.env` в git
- ❌ Не используйте `YOOKASSA_TEST_MODE=false` без проверки
- ❌ Не выставляйте порты БД наружу в production
- ❌ Не используйте слабые `TG_WEBHOOK_SECRET`

## Дополнительная информация

- **Документация API:** https://ai.google.dev/gemini-api/docs
- **Veo 3 Models:** https://ai.google.dev/api/generate-media/video
- **Telegram Bot API:** https://core.telegram.org/bots/api
- **Диагностика:** см. файл `DIAGNOSIS.md`

## Поддержка

При возникновении проблем:
1. Проверьте логи: `docker-compose logs -f bot`
2. Проверьте healthcheck: `curl localhost:8080/healthz`
3. Прочитайте `DIAGNOSIS.md` с подробным анализом ошибок
4. Создайте issue в репозитории с логами и описанием проблемы
