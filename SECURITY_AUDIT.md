# Аудит безопасности и анализ слабых мест Sora2_bot

## Содержание

1. [Критические уязвимости безопасности](#1-критические-уязвимости-безопасности)
2. [Архитектурные проблемы](#2-архитектурные-проблемы)
3. [Утечки памяти и конкурентность](#3-утечки-памяти-и-конкурентность)
4. [Обработка ошибок](#4-обработка-ошибок)
5. [Конфигурация и деплой](#5-конфигурация-и-деплой)
6. [CI/CD и pipeline](#6-cicd-и-pipeline)
7. [Мониторинг и наблюдаемость](#7-мониторинг-и-наблюдаемость)
8. [Рекомендации по приоритету](#8-рекомендации-по-приоритету)

---

## 1. Критические уязвимости безопасности

### 1.1. Захардкоженные пароли БД (CRITICAL)

**Файл:** `docker-compose.yml:9,34`

```yaml
DATABASE_URL: postgresql://sora:sora_password@postgres:5432/sora
POSTGRES_PASSWORD: sora_password
```

Пароль БД находится в version control. Любой с доступом к репозиторию получает доступ к базе.

**Исправление:** Использовать Docker Secrets или `.env` файл (который уже в `.gitignore`).

---

### 1.2. Command injection в CI/CD (CRITICAL)

**Файл:** `.github/workflows/test.yml:18-19`

```bash
TASK=$(echo "${{ github.event.issue.body }}" | sed -n '/```task/,/```/p' | sed '1d;$d')
```

Тело issue передаётся напрямую в shell без экранирования. Злоумышленник может создать issue с shell-метасимволами и получить выполнение произвольного кода.

**Файл:** `.github/scripts/execute_task.py:51-52`

```python
os.makedirs(os.path.dirname(filepath), exist_ok=True)
with open(filepath, 'w') as f:
    f.write(content)
```

Путь файла не валидируется — возможна path traversal атака (например, `../../../../etc/passwd`).

**Исправление:** Валидировать и экранировать все входные данные. Добавить whitelist разрешённых путей.

---

### 1.3. Логирование пользовательских данных (HIGH)

**Файл:** `app_server.py:709,735,751`

```python
log.exception("Webhook: invalid JSON body", extra={"raw": raw[:1000]})
log.exception(..., extra={"update_json": data})
```

Полные Telegram-обновления (содержащие личные сообщения, имена, ID пользователей) записываются в логи.

**Исправление:** Логировать только `update_id` и метаданные, не содержимое сообщений.

---

### 1.4. Неправильная обработка ошибки webhook-аутентификации (MEDIUM)

**Файл:** `app_server.py:699-702`

```python
if WEBHOOK_SECRET:
    if x_telegram_bot_api_secret_token != WEBHOOK_SECRET:
        log.warning("Webhook: bad secret token")
        return Response(status_code=200)  # Возвращает 200 вместо 401/403
```

При неверном секрете возвращается HTTP 200 (успех). Если `WEBHOOK_SECRET` пуст — проверка отключена.

**Исправление:** Возвращать 401/403 при ошибке аутентификации. Требовать `WEBHOOK_SECRET` в production.

---

### 1.5. SSRF через follow_redirects (MEDIUM)

**Файлы:**
- `providers/openai_video.py:571` — `allow_redirects=True`
- `providers/sora_video.py:934` — `allow_redirects=True`
- `services/gemini_downloader.py:422` — `follow_redirects=True`

HTTP-запросы следуют за редиректами без валидации целевого URL. Если пользовательский ввод влияет на URL, возможна SSRF-атака.

**Исправление:** Валидировать URL после редиректа, вести whitelist разрешённых хостов.

---

### 1.6. YooKassa test mode по умолчанию (MEDIUM)

**Файл:** `config.py:374`

```python
yookassa_test_mode: bool = True
```

Тестовый режим включён по умолчанию. В тестовом режиме пропускается проверка IP (`app_server.py:635`).

**Исправление:** Явно требовать `YOOKASSA_TEST_MODE=false` в production.

---

## 2. Архитектурные проблемы

### 2.1. God-объекты (CRITICAL для поддержки)

| Файл | Строк | Функций | Проблема |
|------|-------|---------|----------|
| `handlers.py` | **7 962** | 209 | Монолитный файл: UI, платежи, генерация, админка, тарот, ChatGPT |
| `providers/gemini.py` | **4 292** | 75 | Изображения + видео + текст + retry в одном файле |
| `jobs.py` | **3 039** | 50 | Очередь + скачивание + повторы + все провайдеры |

`handlers.py` — главный антипаттерн. 209 функций в одном файле делают его практически неподдерживаемым.

**Исправление:** Разбить на модули по доменам:
- `handlers/payments.py`
- `handlers/generation.py`
- `handlers/admin.py`
- `handlers/tarot.py`
- `handlers/chatgpt.py`

---

### 2.2. Дублирование кода в провайдерах (HIGH)

**Идентичные функции:**
- `_iter_nodes()` — `sora_video.py:74-87` и `openai_video.py:55-68` (побайтовое совпадение)
- `_extract_url()` — `sora_video.py:90` и `openai_video.py:71`
- `download_content()` — дублирован в 3 файлах (`sora_video.py:904-1020`, `openai_video.py:515-650`, `veo_video.py:821-856`), ~80% совпадение

**Исправление:** Вынести общую логику в `providers/base.py` или отдельный `providers/download.py`.

---

### 2.3. Устаревший фреймворк aiogram 2.x (HIGH)

**Файл:** `requirements.txt`

```
aiogram==2.25.1
```

aiogram 2.x **deprecated** — текущая стабильная версия 3.x. Проблемы:
- Не получает обновления безопасности
- Несовместим с Python 3.13+
- Устаревший API (State, Middleware, Filters)

**Исправление:** Планировать миграцию на aiogram 3.x. Это масштабный рефакторинг, затрагивающий `handlers.py`, `main.py`, все middleware.

---

### 2.4. Тесная связанность (HIGH)

**Файл:** `handlers.py:123-127`

```python
from providers.gemini import dump_gemini_case
from providers.openai_chat import OpenAIChatClient
from providers.openai_video import OpenAIVideoClient
```

Обработчики напрямую импортируют конкретные реализации провайдеров. Невозможно заменить провайдер без изменения обработчиков.

**Исправление:** Использовать фабрику или Dependency Injection.

---

### 2.5. Низкое покрытие тестами (MEDIUM)

| Метрика | Значение |
|---------|----------|
| Файлов тестов | 31 |
| Строк тестов | ~4 400 |
| Строк исходного кода | ~34 000 |
| Оценочное покрытие | **~13%** |

**Не покрыты тестами:**
- Управление сессиями (`SESSION_MANAGER`) — утечки памяти не обнаруживаются
- Платёжная обработка (`_PENDING_PAYMENT_LOCKS`) — race conditions не тестируются
- 209 функций в handlers.py — только 4 файла тестов

---

## 3. Утечки памяти и конкурентность

### 3.1. Утечка памяти: словарь сессий (CRITICAL)

**Файл:** `handlers.py:1360-1366`

```python
class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[int, UserSession] = {}

    def get(self, user_id: int) -> UserSession:
        return self._sessions.setdefault(user_id, UserSession())

SESSION_MANAGER = SessionManager()
```

Словарь `_sessions` растёт бесконечно. Каждый новый пользователь добавляет запись, которая **никогда не удаляется**. При 1М пользователей — значительный расход памяти.

**Исправление:** Добавить TTL-based eviction (LRU-кеш или периодическую очистку).

---

### 3.2. Утечка памяти: платёжные блокировки (CRITICAL)

**Файл:** `handlers.py:2856-2865`

```python
_PENDING_PAYMENT_LOCKS: dict[int, asyncio.Lock] = {}
_PENDING_PAYMENT_LAST_ATTEMPT: dict[int, float] = {}

def _get_payment_lock(user_id: int) -> asyncio.Lock:
    lock = _PENDING_PAYMENT_LOCKS.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        _PENDING_PAYMENT_LOCKS[user_id] = lock
    return lock
```

Два словаря растут бесконечно. Каждый пользователь, сделавший платёж, добавляет Lock + float, которые **никогда не очищаются**.

---

### 3.3. Утечка памяти: _TRACE_HISTORY (HIGH)

**Файл:** `providers/gemini.py:90`

```python
_TRACE_HISTORY: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
```

Список trace-данных растёт бесконечно без ограничения.

---

### 3.4. Race conditions (HIGH)

- `_TRACE_HISTORY` (gemini.py:90) — `defaultdict(list)` без блокировки, запись из множества async-задач
- `_MODEL_CAPABILITIES` (gemini.py:87) — общий `dict` без синхронизации
- `_PENDING_PAYMENT_LOCKS` (handlers.py:2860-2865) — `dict.get()` + присвоение не атомарны

---

### 3.5. Fire-and-forget задачи (MEDIUM)

**Файлы:**
- `app_server.py:782` — `asyncio.create_task()` без обработки исключений
- `archive.py:131` — `asyncio.create_task()` без отслеживания
- `main.py:334,353` — фоновые задачи без explicit error handling

Если задача падает с исключением — оно будет потеряно (в лучшем случае — warning от asyncio).

**Исправление:** Добавить `task.add_done_callback()` для логирования ошибок.

---

## 4. Обработка ошибок

### 4.1. Слишком широкие except-блоки (15+ случаев)

**Примеры:**
- `jobs.py:184,200,223,330,426,574` — `except Exception:` с `pass` или минимальным логированием
- `providers/gemini.py:582,690,715,1021,4010` — глушение ошибок файловых операций
- `providers/openai_image.py:225,235` — проглатывание ошибок FS

**Исправление:** Ловить конкретные исключения (`IOError`, `json.JSONDecodeError`, и т.д.).

---

### 4.2. Незащищённые файловые операции

**Файлы:**
- `archive.py:526-527` — `_read_file()` без try/except
- `tarot_data.py:17-18` — чтение JSON при загрузке модуля без обработки
- `tarot_telegram.py:47-48` — запись кеша без обработки ошибок
- `.github/scripts/execute_task.py:14,52` — file I/O без защиты

---

### 4.3. Утечки ресурсов при ошибках скачивания

**Файлы:**
- `providers/sora_video.py:952-965`
- `providers/openai_video.py:592-605`
- `providers/veo_video.py:833`

```python
target_dir = Path(tempfile.mkdtemp(prefix="sora-video-"))
# ... если ошибка при скачивании, директория НЕ удаляется
```

Временные директории создаются, но **не очищаются** при ошибках. Постепенное заполнение диска.

---

## 5. Конфигурация и деплой

### 5.1. Dockerfile: контейнер под root (HIGH)

**Файл:** `Dockerfile`

Нет директивы `USER` — процесс запускается от root. При компрометации контейнера атакующий получает root-доступ.

**Исправление:**
```dockerfile
RUN useradd -m appuser
USER appuser
```

---

### 5.2. Docker-compose: открытые порты БД (HIGH)

**Файл:** `docker-compose.yml:44,57`

```yaml
ports:
  - "5432:5432"   # PostgreSQL открыт на 0.0.0.0
  - "6379:6379"   # Redis открыт на 0.0.0.0
```

БД и Redis доступны из интернета. Redis без пароля.

**Исправление:** Убрать `ports` или привязать к `127.0.0.1:5432:5432`.

---

### 5.3. Незакреплённые версии зависимостей (HIGH)

**Файл:** `requirements.txt`

| Пакет | Версия | Проблема |
|-------|--------|----------|
| python-dotenv | не указана | Могут сломать API |
| yookassa | не указана | Неизвестная совместимость |
| fastapi | не указана | Major-версия может измениться |
| gspread | не указана | Deprecated API возможен |
| google-auth | не указана | Breaking changes |
| tenacity | не указана | API может измениться |
| asyncpg | не указана | Breaking changes |

**Исправление:** Закрепить все версии: `pip freeze > requirements.txt`.

---

### 5.4. Отсутствие миграций БД (HIGH)

- Нет системы версионирования миграций (нет Alembic)
- Нет таблицы `schema_version` для отслеживания
- Нет автоматического запуска миграций при старте
- Нет возможности rollback

---

### 5.5. Отсутствие бекапов (HIGH)

- Нет автоматизированного бекапа PostgreSQL
- Нет политики retention
- Нет проверенной процедуры восстановления
- Нет off-site хранилища бекапов

---

## 6. CI/CD и pipeline

### 6.1. Отсутствие code review в workflow (HIGH)

`.github/workflows/test.yml` автоматически создаёт PR и пушит код без:
- Проверки тестов
- Code review
- Валидации сгенерированных файлов
- Проверки максимального размера файлов

---

### 6.2. Секреты передаются ненадёжному коду (HIGH)

**Файл:** `.github/workflows/test.yml:22-24`

```yaml
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

API-ключи передаются в скрипт, который выполняет сгенерированный код.

---

## 7. Мониторинг и наблюдаемость

### 7.1. Недостающие алерты

**Файл:** `alerts.yml`

| Алерт | Статус |
|-------|--------|
| ActiveJobsHigh | Есть |
| JobErrorRateHigh | Есть |
| WorkerPoolExhausted | Есть |
| **Postgres недоступен** | **НЕТ** |
| **Redis недоступен** | **НЕТ** |
| **Высокая латентность API** | **НЕТ** |
| **Нехватка памяти** | **НЕТ** |
| **Заполнение диска** | **НЕТ** |
| **Ошибки аутентификации** | **НЕТ** |

---

### 7.2. Нет readiness probe

`/healthz` не проверяет зависимости (БД, Redis). Нет `/ready` эндпоинта для Kubernetes/оркестратора.

---

### 7.3. Нет мониторинга БД и Redis

**Файл:** `prometheus.yml`

Prometheus мониторит только сам бот. Нет:
- postgres_exporter
- redis_exporter
- node_exporter (системные метрики)

---

## 8. Рекомендации по приоритету

### CRITICAL (исправить немедленно)

1. **Убрать пароль из docker-compose.yml** — перенести в `.env` или Docker Secrets
2. **Исправить command injection в CI/CD** — валидировать и экранировать входные данные
3. **Добавить eviction в SESSION_MANAGER** — предотвратить утечку памяти
4. **Убрать логирование пользовательских данных** — GDPR/privacy compliance

### HIGH (исправить в ближайшем спринте)

5. Закрыть порты БД/Redis в docker-compose
6. Добавить `USER` в Dockerfile
7. Закрепить все версии зависимостей
8. Вынести дублированную логику скачивания в общий модуль
9. Добавить алерты на недоступность БД и Redis
10. Исправить webhook-аутентификацию (возвращать 401 вместо 200)

### MEDIUM (планировать в roadmap)

11. Разбить `handlers.py` на модули
12. Внедрить Alembic для миграций
13. Настроить автоматические бекапы
14. Добавить readiness/liveness probes
15. Планировать миграцию на aiogram 3.x
16. Увеличить покрытие тестами до 50%+
17. Добавить mutex для `_TRACE_HISTORY` и `_MODEL_CAPABILITIES`
18. Очищать временные директории при ошибках скачивания

### LOW (улучшения)

19. Добавить CORS middleware
20. Добавить `pip-audit` в CI
21. Настроить ротацию секретов
22. Документировать RTO/RPO для disaster recovery
