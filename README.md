# News Risk Agent

Production-ready агент для мониторинга корпоративных рисков в RSS/Atom-лентах.
Он загружает и дедуплицирует новости, выполняет дешёвую фильтрацию по ключевым
словам и отправляет кандидатов в OpenAI пакетами со строгой схемой ответа.

## Возможности

- import-safe Python-пакет и CLI;
- конфигурация через переменные окружения и `.env`;
- таймауты, HTTP-ретраи, редиректы и понятные ошибки;
- дедупликация и ограничение объёма ленты;
- типизированный structured output через OpenAI Responses API;
- JSON-вывод, пригодный для cron, Airflow и контейнеров;
- unit-тесты без реальных HTTP/LLM-вызовов;
- Docker и GitHub Actions.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
# добавьте API-ключ в .env
news-risk-agent --pretty
```

Можно передать другую ленту:

```bash
news-risk-agent --url https://example.com/feed.xml --pretty
```

Программа пишет служебные сообщения в stderr, а результат — JSON в stdout. Поле
`analyzed_news` содержит исходные заголовки и ссылки; оценки ссылаются на них через `news_id`.
Код завершения `0` означает успех, `1` — ошибку конфигурации, RSS или OpenAI.

## Конфигурация

Все параметры имеют префикс `NEWS_AGENT_`:

| Переменная | По умолчанию | Назначение |
|---|---:|---|
| `OPENAI_API_KEY` | — | Ключ OpenAI (обязателен) |
| `OPENAI_MODEL` | `gpt-5.6-terra` | Модель анализа |
| `RSS_URL` | лента «Ведомостей» | RSS/Atom URL |
| `REQUEST_TIMEOUT_SECONDS` | `15` | HTTP-таймаут |
| `MAX_FEED_ITEMS` | `100` | Максимум записей из ленты |
| `MAX_CANDIDATES` | `30` | Максимум новостей для LLM |
| `BATCH_SIZE` | `10` | Размер LLM-пакета |
| `RETRIES` | `3` | Число повторов |
| `MIN_CONFIDENCE` | `0.55` | Порог уверенности |
| `KEYWORDS` | встроенный список | Ключевые основы через запятую |

При пустом `NEWS_AGENT_KEYWORDS` анализируются все записи до лимита кандидатов.

## Проверки

```bash
ruff check .
mypy llm_agent_news
pytest
```

## Docker

```bash
docker build -t news-risk-agent .
docker run --rm --env-file .env news-risk-agent --pretty
```

Для регулярного запуска используйте cron/Kubernetes CronJob и сохраняйте stdout
в вашу БД или очередь. API-ключ передавайте через secret manager, не добавляйте
`.env` в Git.
