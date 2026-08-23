FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --system app && useradd --system --gid app --create-home app
WORKDIR /app

COPY pyproject.toml README.md ./
COPY llm_agent_news ./llm_agent_news
RUN pip install --upgrade pip && pip install .

USER app
ENTRYPOINT ["news-risk-agent"]

