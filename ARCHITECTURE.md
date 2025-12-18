# Архитектура Sora2_bot

## Компоненты

- **Telegram/Aiogram** — приём команд пользователей, inline-кнопки, FSM.
- **FastAPI** — HTTP-вебхук Telegram, вебхук YooKassa, `/healthz` и `/metrics`.
- **Очередь заданий (`jobs.py`)** — планирование генерации, трекинг статусов, возврат кредитов.
- **Провайдеры генерации** — Gemini (текст/картинки/видео Veo), OpenAI Video API (Sora 2), OpenAI Chat/Image.
- **Платёжные интеграции** — Telegram Stars и YooKassa с валидацией подписи.
- **Хранилища** — Google Sheets (основной источник), PostgreSQL (миграционная/боевые записи), Redis (FSM/лимиты).
- **Наблюдаемость** — Prometheus endpoint `/metrics`, healthchecks, структурные логи.

## Диаграмма компонентов

```mermaid
graph TD
  TG[Telegram] --> Bot[Aiogram bot]
  Bot -->|HTTP| FastAPI[FastAPI app]
  FastAPI --> YooKassa[YooKassa webhook]
  Bot --> Queue[JobQueue]
  Queue --> Providers[Gemini/Veo/Sora/OpenAI]
  Queue --> Storage{Dual-write layer}
  Storage --> Sheets[Google Sheets]
  Storage --> Postgres[PostgreSQL]
  Bot --> Redis[Redis FSM/rate limit]
  FastAPI --> Metrics[/metrics]
  Metrics --> Prometheus
```

## Потоки данных

1. Пользователь отправляет команду → Aiogram формирует задачу и кладёт в очередь.
2. Очередь вызывает нужного провайдера (Gemini, Veo, Sora/OpenAI) и отслеживает статус.
3. Результат записывается через слой dual-write: чтение из `DUAL_WRITE_PRIMARY`, запись в Sheets и/или Postgres.
4. FastAPI отвечает за вебхуки Telegram/YooKassa и отдаёт `/metrics`/`/healthz` для мониторинга.
5. Redis хранит состояния FSM и ключи rate-limit, не содержит бизнес-данных.

## Стратегия dual-write и переключение на PostgreSQL

- При `DUAL_WRITE_ENABLED=true` запись дублируется в Google Sheets и PostgreSQL.
- `DUAL_WRITE_PRIMARY=postgres|sheets` определяет, откуда читаем и куда откладываем ошибки вторичного бэкенда.
- Миграция: примените SQL из `migrations/`, включите dual-write, прогоните `scripts/migrate_from_sheets.py`, сверяйте суммы кредитов/платежей в обеих сторонах.
- После валидации отключите dual-write и оставьте `POSTGRES_ENABLED=true` (или только `DATABASE_URL`). Для отката временно верните `DUAL_WRITE_PRIMARY=sheets`.

## Метрики и здоровье

- `/healthz` проверяет доступность провайдеров и ключевых зависимостей.
- `/metrics` отдаёт Prometheus-метрики (`active_jobs_count`, `job_errors_total`, `jobs_completed_total`, задержки провайдеров).
- `prometheus.yml` и `alerts.yml` содержат базовые правила; Grafana может подключаться напрямую к Prometheus.

## Хранение и миграции

- Схема Postgres задаётся SQL-файлами в `migrations/` (начиная с `001_initial_schema.sql`).
- Применяйте миграции транзакционно (`psql -f file.sql`); проверяйте `\dt` и `\d <table>`.
- Для резервного копирования используйте `pg_dump`/`pg_restore` для Postgres и экспорт/копию таблиц в Google Sheets для вторичного хранилища.
