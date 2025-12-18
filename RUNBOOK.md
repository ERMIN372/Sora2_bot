# Runbook: эксплуатация Sora2_bot

## Предстартовый чек-лист
- Заданы ключевые ENV: `TELEGRAM_BOT_TOKEN`, `GOOGLE_API_KEY`/`GEMINI_API_KEY`, `GOOGLE_SA_JSON_BASE64`, `GOOGLE_SHEET_ID`, `DATABASE_URL` (если Postgres), `PUBLIC_BASE_URL`/`TG_WEBHOOK_SECRET` для вебхуков.
- Применены миграции PostgreSQL (`psql -f migrations/001_initial_schema.sql`).
- Redis доступен (если используется) — `redis-cli -u "$REDIS_URL" ping`.
- В `.env` указан `BOT_MODE` (`polling` для локали, `webhook` для продакшена).

## Запуск и остановка
- **Локально**:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python main.py --mode polling
  ```
- **Docker Compose**:
  ```bash
  docker compose up -d
  docker compose logs -f bot
  ```
- Остановка: `Ctrl+C` в интерактивном режиме или `docker compose down` для compose.

## Миграции и dual-write
1. Примените SQL-файлы из `migrations/` в порядке версий.
2. Включите двойную запись на период миграции:
   ```bash
   export DUAL_WRITE_ENABLED=true
   export DUAL_WRITE_PRIMARY=postgres
   ```
3. Выполните перенос из таблиц:
   ```bash
   python scripts/migrate_from_sheets.py
   ```
4. Сверьте суммы кредитов/платежей между Sheets и Postgres; при совпадении можно отключать dual-write (`DUAL_WRITE_ENABLED=false`).
5. Для отката чтения на Google Sheets установите `DUAL_WRITE_PRIMARY=sheets`.

## Проверки здоровья
- `curl http://localhost:8080/healthz` — базовый healthcheck.
- `curl http://localhost:8080/metrics` — убедитесь, что Prometheus-метрики доступны.
- Команда `/diag_openai_video` в Telegram — проверка подключения к OpenAI Video API.

## Наблюдаемость и алерты
- Prometheus собирает `/metrics` (см. `prometheus.yml`); правила алертов — `alerts.yml`.
- Рекомендуется подключить Grafana к Prometheus (порт 9090) и добавить дашборд по очереди задач и ошибкам генерации.
- Логи выводятся в stdout; при необходимости направьте их в систему логирования контейнерной платформы.

## Инциденты и восстановление
- **Накопление очереди**: проверьте метрики `active_jobs_count` и `jobs_completed_total`. Можно временно увеличить `JOBS_CONCURRENCY` или удалить застрявшие задачи, вернув кредиты.
- **Проблемы с платежами**: повторите вебхук YooKassa или Telegram Stars, убедитесь в корректности `TG_WEBHOOK_SECRET` и ключей. Проверяйте таблицы `payments`/лист `payments`.
- **Недоступность провайдеров**: временно блокируйте новые запросы, повышая `CREDITS_PER_GENERATION` или отключая команду генерации; следите за логами повторов.
- **Postgres упал**: при включённом dual-write переключите `DUAL_WRITE_PRIMARY=sheets`, чтобы чтение шло из таблиц, и восстановите БД из бэкапа (`pg_restore`).

## Бэкапы
- **PostgreSQL**: `pg_dump "$DATABASE_URL" > backup.sql` и `pg_restore` для восстановления.
- **Google Sheets**: делайте копию таблицы или экспортируйте в CSV перед массовыми изменениями.

## Обновления
- Перед деплоем новой версии прогоните `pytest` и линтеры (`black --check .`, `ruff check .`, `mypy .`).
- После деплоя убедитесь, что `/healthz` и ключевые команды (`/start`, `/balance`) работают.
