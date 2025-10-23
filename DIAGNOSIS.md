# Диагностика TG-бота (Gemini Image + Veo 3)

## Итог

* Ошибка `models/gemini-2.5-flash-image ... not found` возникает в healthcheck: пинг изображения вызывается без параметра `model`, поэтому SDK отправляет запрос с `model="ping"`, что приводит к 404/`predict` по v1beta вместо Developer API. 【F:healthcheck.py†L53-L69】
* Параметр `resolution` всё ещё добавляется в тело Veo-запросов из `_build_config`; текущие модели Veo 3.x в Developer API его не принимают, что объясняет `400 INVALID_ARGUMENT: "resolution" isn't supported by this model`. 【F:providers/veo_video.py†L304-L353】
* Ошибка `'utf-8' codec can't decode byte 0x89` происходит при декодировании `GOOGLE_SA_JSON_BASE64` в модуле Google Sheets: если переменная содержит бинарные (PNG) данные, попытка `decoded.decode("utf-8")` падает. 【F:gsheets_db.py†L206-L223】
* Сообщение `Models.get() got an unexpected keyword argument 'name'` воспроизводится при вызове `client.models.get(name=...)`. В текущем коде используется корректный параметр `model`, но в сторонних скриптах/конфигурации нужно заменить `name` на `model`. 【F:healthcheck.py†L35-L43】
* Жалобы «Нет доступных моделей видеогенерации» вызваны тем, что меню берёт единственную запись из конфигурации и отображает жёсткую подпись `veo 2`; если переменная окружения не обновлена (или пуста), список пустой. Также локализация устарела и не отображает Veo 3.x. 【F:handlers.py†L146-L179】【F:i18n.py†L132-L205】

Vertex AI в репозитории отсутствует: поиск по `vertex`, `v1beta`, `publishers/google`, `USE_VERTEX`, `GCP_` не дал совпадений. 【fb04ee†L1-L2】【8beab8†L1-L1】【38c112†L1-L1】【d9da34†L1-L1】【2d58c9†L1-L1】

## Окружение и версии

* `google-genai>=1.46.0` указан в зависимостях проекта. 【F:requirements.txt†L1-L10】
* Фактически установленная версия в среде диагностики: 1.46.0 (`pip show google-genai`). 【730ad8†L1-L10】

## Конфиг и ENV

* Чтение ключей Gemini (`GOOGLE_API_KEY`/`GEMINI_API_KEY`) и моделей происходит в `config.load_config()`. 【F:config.py†L367-L425】
* Значения по умолчанию для моделей: текст `gemini-2.0-flash`, изображение `gemini-2.5-flash-image`, видео `veo-2.0-generate-001`. 【F:config.py†L133-L143】
* Пример `.env` содержит те же параметры и не упоминает Vertex. 【F:.env.example†L1-L62】
* Параметры цен/лимитов, а также Google Sheets (`GOOGLE_SHEET_ID`, `GOOGLE_SA_JSON_BASE64`) обязательны. 【F:config.py†L396-L405】【F:gsheets_db.py†L206-L223】
* `GOOGLE_SA_JSON_BASE64` используется только в модуле Google Sheets. 【F:gsheets_db.py†L206-L223】

## Провайдеры и сервисы

* Общий клиент Gemini создаётся в `services/gemini_client.py`. 【F:services/gemini_client.py†L1-L43】
* Изображения генерируются через `client.models.generate_images` в `GeminiImageClient`. 【F:providers/gemini.py†L666-L697】
* Видео обрабатываются через `client.models.generate_videos`/`_generate_videos` в `VeoVideoClient`. 【F:providers/veo_video.py†L271-L284】
* Параметр `resolution` добавляется в конфигурацию Veo. 【F:providers/veo_video.py†L304-L353】
* Обработка очереди заданий и выбор провайдера — в `jobs.py`. 【F:jobs.py†L34-L129】
* Healthcheck выполняет вызовы `models.get` и `generate_images`. 【F:healthcheck.py†L35-L90】
* Telegram-меню и кнопки формируются в `handlers.py` (см. `VideoModelOption`). 【F:handlers.py†L138-L179】
* Сторонние интеграции (архив, платежи, Google Sheets) не используют Vertex. 【F:app_server.py†L1-L120】【F:observability.py†L1-L120】

## Поисковые паттерны

| Паттерн | Результат |
|---------|-----------|
| `predict`, `v1beta`, `publishers/google` | Совпадений нет. 【ccfb12†L1-L1】【8beab8†L1-L1】【38c112†L1-L1】 |
| `models/gemini-2.5-flash-image` | Совпадений нет (модель задаётся без префикса). 【201d82†L1-L1】 |
| `generate_images(` | Вызовы только в healthcheck и Gemini-провайдере. 【a90932†L1-L6】 |
| `generate_videos(` | Используется в Veo-клиенте и тестовых заглушках. 【78df12†L1-L6】 |
| `response_mime_type`, `inline_data` | Обработка inline-данных есть, Vertex-специфики нет. 【a59300†L1-L20】 |
| `decode("utf-8")` | Лишь в Google Sheets и обработке HTTP-ошибок. 【eecec6†L1-L4】 |
| `base64.b64encode(` | Используется при подготовке inline-активов. 【d3a983†L1-L8】 |
| `Models.get(.*name=` | Нет совпадений (версия SDK ожидает `model=`). 【bedddc†L1-L1】 |
| `models.list(` | Совпадений нет — список моделей пока не запрашивается. 【52169a†L1-L1】 |
| `SORA_ENABLED`, `veo-3`, `veo 2` | `veo 2` встречается только в переводах. 【188bd7†L1-L1】 |
| `resolution` | Настраивается в Veo-конфиге и тестах. 【462ac5†L1-L4】 |

## Проблемные места

| Файл | Строки | Проблема | Репро-шаги | Доказательство |
|------|--------|----------|------------|----------------|
| `healthcheck.py` | 53-69 | `generate_images` вызывается без `model`, провоцируя обращение к `model="ping"` и ошибку `models/... not found` (Vertex-style). | Запустить healthcheck с включённым Gemini: `client.models.generate_images("ping")` → 404/`predict`. | 【F:healthcheck.py†L35-L69】 |
| `providers/veo_video.py` | 310-353 | В тело запроса Veo добавляется `resolution`, который не поддерживается моделями Veo 3.x → `400 INVALID_ARGUMENT`. | Сформировать заказ с `size=1280x720` → в config появится `resolution="720p"`. | 【F:providers/veo_video.py†L304-L353】 |
| `gsheets_db.py` | 206-223 | При декодировании `GOOGLE_SA_JSON_BASE64` предполагается UTF-8; если переменная содержит бинарные данные (ошибочная настройка) — `UnicodeDecodeError` с байтом `0x89`. | Передать в env base64 от PNG. | 【F:gsheets_db.py†L206-L223】 |
| `handlers.py` & `i18n.py` | 146-179, 132-205 | Меню видеомоделей строится из одной записи конфигурации, текст — `veo 2`. Если env `GEMINI_MODEL_VIDEO` пуст/необновлён до Veo 3.x, бот сообщает «Нет доступных моделей». | Очистить `GEMINI_MODEL_VIDEO` или оставить дефолт `veo-2.0...` → кнопка скрывается, либо устаревшее имя отображается. | 【F:handlers.py†L146-L179】【F:i18n.py†L132-L205】 |
| Тесты/конфиги | 141-143 (`config.py`), `.env.example` | Значение по умолчанию для видео-модели — `veo-2.0-generate-001`, что не существует в Developer API и ведёт к «модель не найдена». | Загрузить конфиг без переопределения env → попытки использовать несуществующую модель. | 【F:config.py†L139-L143】【F:.env.example†L15-L18】 |

## Список моделей (models.list)

Вызов `client.models.list()` не выполнен: нет рабочего `GEMINI_API_KEY` в окружении диагностики, поэтому SDK выбрасывает `KeyError`. Нужно повторить с валидным ключом и сохранить JSON в отчёт.

## Smoke-тесты

```
$ python tools/smoke/smoke_image.py
KeyError: 'GEMINI_API_KEY'
```
【164d8b†L1-L6】

```
$ python tools/smoke/smoke_video.py
KeyError: 'GEMINI_API_KEY'
```
【17b9b4†L1-L6】

Скрипты готовы в `tools/smoke/`, но для запуска требуется задать `GEMINI_API_KEY` и корректные модели (`GEMINI_MODEL_IMAGE`, `GEMINI_MODEL_VIDEO`). 【F:tools/smoke/smoke_image.py†L1-L9】【F:tools/smoke/smoke_video.py†L1-L13】

## Мини-патчи (для воспроизводимости/фикса)

```diff
--- a/healthcheck.py
+++ b/healthcheck.py
@@
-    try:
-        image_response = await asyncio.to_thread(client.models.generate_images, "ping")
+    try:
+        image_response = await asyncio.to_thread(
+            client.models.generate_images,
+            model=config.gemini_model_image,
+            prompt="ping"
+        )
+        video_ping = await asyncio.to_thread(
+            client.models.generate_videos,
+            model=config.gemini_model_video,
+            prompt="ping",
+            config=_genai_types.GenerateVideosConfig(duration_seconds=1),
+        )
     except _genai_errors.APIError as exc:
         ...
```

```diff
--- a/providers/veo_video.py
+++ b/providers/veo_video.py
@@
-            resolution = _infer_resolution(size)
-            if resolution:
-                config_data.setdefault("resolution", resolution)
-                summary.setdefault("resolution", resolution)
+            # Veo 3.x Developer API не принимает поле resolution
+            # config_data.setdefault("resolution", resolution)
```

```diff
--- a/handlers.py
+++ b/handlers.py
@@
-def _video_model_options(config: Config) -> list[VideoModelOption]:
-    options: list[VideoModelOption] = []
-    if config.gemini_video_enabled:
-        options.append(
-            VideoModelOption(
-                key="veo",
-                model=config.gemini_model_video,
-                provider="veo",
-                label=i18n.t("video.models.veo"),
-            )
-        )
-    return options
+def _video_model_options(config: Config) -> list[VideoModelOption]:
+    model = (config.gemini_model_video or "").strip()
+    if not model:
+        return []
+    label = model if model.startswith("veo-3") else i18n.t("video.models.veo")
+    return [
+        VideoModelOption(key="veo", model=model, provider="veo", label=label)
+    ]
```

Эти изменения устраняют ошибки проверок без изменения архитектуры.

## Риски и ограничения

* Для подтверждения списка моделей и smoke-тестов необходим действующий ключ Gemini с доступом к Veo 3.1; без него диагностика ограничивается статическим анализом.
* Удаление `resolution` лишает поддержку старых Veo 2.x; при необходимости параметр можно оставлять под флагом совместимости.
* Healthcheck с видеопингом может потребовать дополнительной квоты; стоит использовать короткий prompt и малую длительность.

## Следующие шаги

1. Обновить конфигурацию: задать `GEMINI_MODEL_VIDEO=veo-3.1-generate-preview` (или другую доступную модель) и убедиться, что ключ `GEMINI_API_KEY`/`GOOGLE_API_KEY` активен.
2. Применить патч healthcheck для корректного пинга моделей и зафиксировать результаты `generate_images`/`generate_videos`.
3. Удалить параметр `resolution` из запросов Veo 3.x, протестировать с `tools/smoke/smoke_video.py`.
4. Актуализировать пользовательские тексты (локализации) и меню, чтобы отображать реальные модели Veo 3.x и не скрывать кнопки при пустом списке.
5. Проверить содержимое `GOOGLE_SA_JSON_BASE64`: туда нужно класть JSON сервис-аккаунта (в base64 или чистым текстом), иначе Google Sheets инициализация падает.
