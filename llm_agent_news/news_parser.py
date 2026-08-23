"""Production-ready RSS news risk analysis agent.

The module is import-safe: network and API calls happen only through explicit
method calls or the CLI entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol
from urllib.parse import urlparse

import feedparser  # type: ignore[import-untyped]
import httpx
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

LOGGER = logging.getLogger("news_risk_agent")

DEFAULT_KEYWORDS = (
    "банкрот",
    "санкц",
    "уголов",
    "обыск",
    "арест",
    "дефолт",
    "ликвидац",
    "мошеннич",
    "расследован",
    "репутацион",
    "иск",
)

SYSTEM_PROMPT = """Ты — аналитик корпоративных рисков. Оценивай только новости,
переданные пользователем. Не делай выводов, которых нет в тексте. Для каждой новости
верни ровно одну оценку. Риск есть, если текст содержит обоснованные признаки
банкротства, дефолта, санкций, уголовного или регуляторного преследования, мошенничества,
существенного судебного спора либо репутационного ущерба. Причину формулируй кратко
на русском языке. При недостатке данных снижай уверенность и явно указывай это.
"""


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="NEWS_AGENT_", extra="ignore")

    rss_url: HttpUrl = HttpUrl("https://www.vedomosti.ru/rss/news.xml")
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.6-terra"
    request_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    max_feed_items: int = Field(default=100, ge=1, le=1000)
    max_candidates: int = Field(default=30, ge=1, le=200)
    batch_size: int = Field(default=10, ge=1, le=50)
    retries: int = Field(default=3, ge=1, le=8)
    min_confidence: float = Field(default=0.55, ge=0, le=1)
    keywords: str = ",".join(DEFAULT_KEYWORDS)

    @property
    def keyword_list(self) -> tuple[str, ...]:
        return tuple(part.strip().lower() for part in self.keywords.split(",") if part.strip())


class RiskCategory(StrEnum):
    bankruptcy = "bankruptcy"
    sanctions = "sanctions"
    criminal = "criminal"
    regulatory = "regulatory"
    fraud = "fraud"
    litigation = "litigation"
    reputation = "reputation"
    none = "none"


class NewsItem(BaseModel):
    """Normalized item from an RSS/Atom feed."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str = Field(min_length=1, max_length=1000)
    summary: str = Field(default="", max_length=10_000)
    link: str = ""
    published_at: datetime | None = None

    @property
    def text(self) -> str:
        return f"{self.title}. {self.summary}".strip()


class RiskAssessment(BaseModel):
    """Risk classification returned for a single news item."""

    news_id: str
    has_risk: bool
    category: RiskCategory
    confidence: float = Field(ge=0, le=1)
    companies: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1, max_length=1000)


class AnalysisBatch(BaseModel):
    assessments: list[RiskAssessment]


class AnalysisResult(BaseModel):
    generated_at: datetime
    source: str
    fetched: int
    candidates: int
    analyzed_news: list[NewsItem]
    assessments: list[RiskAssessment]


class Analyzer(Protocol):
    def analyze(self, items: Sequence[NewsItem]) -> list[RiskAssessment]: ...


class FeedError(RuntimeError):
    """Raised when a feed cannot be downloaded or parsed."""


class AnalysisError(RuntimeError):
    """Raised when model output is unavailable or violates the contract."""


def _struct_time_to_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parts = [int(value[index]) for index in range(6)]
        return datetime(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], tzinfo=UTC)
    except (IndexError, TypeError, ValueError):
        return None


def parse_feed(content: bytes, *, limit: int = 100) -> list[NewsItem]:
    """Parse RSS/Atom bytes, normalize records and remove duplicates."""

    parsed = feedparser.parse(content)
    if parsed.bozo and not parsed.entries:
        raise FeedError(f"Invalid feed: {parsed.get('bozo_exception', 'unknown parse error')}")

    result: list[NewsItem] = []
    seen: set[str] = set()
    for entry in parsed.entries[:limit]:
        title = str(entry.get("title", "")).strip()
        if not title:
            continue
        summary = str(entry.get("summary") or entry.get("description") or "").strip()
        link = str(entry.get("link", "")).strip()
        raw_id = str(entry.get("id") or entry.get("guid") or link or title)
        item_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
        dedupe_key = link or title.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        result.append(
            NewsItem(
                id=item_id,
                title=title,
                summary=summary,
                link=link,
                published_at=_struct_time_to_datetime(
                    entry.get("published_parsed") or entry.get("updated_parsed")
                ),
            )
        )
    return result


class RssClient:
    """Small resilient HTTP client for RSS/Atom feeds."""

    def __init__(self, *, timeout: float = 15, retries: int = 3) -> None:
        self.timeout = timeout
        self.retries = retries

    def fetch(self, url: str, *, limit: int = 100) -> list[NewsItem]:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise FeedError("RSS URL must use http or https")

        headers = {"User-Agent": "news-risk-agent/1.0 (+RSS risk monitoring)"}
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                    response = client.get(url, headers=headers)
                    response.raise_for_status()
                return parse_feed(response.content, limit=limit)
            except (httpx.HTTPError, FeedError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
        raise FeedError(f"Failed to fetch RSS feed after {self.retries} attempts: {last_error}")


class OpenAIRiskAnalyzer:
    """Batch classifier using OpenAI Responses structured outputs."""

    def __init__(self, *, api_key: str, model: str, retries: int = 3) -> None:
        self.client = OpenAI(api_key=api_key, max_retries=retries)
        self.model = model

    def analyze(self, items: Sequence[NewsItem]) -> list[RiskAssessment]:
        if not items:
            return []
        payload = [
            {"news_id": item.id, "title": item.title, "summary": item.summary}
            for item in items
        ]
        try:
            response = self.client.responses.parse(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=json.dumps(payload, ensure_ascii=False),
                text_format=AnalysisBatch,
            )
        except Exception as exc:
            raise AnalysisError(f"OpenAI analysis failed: {exc}") from exc

        parsed = response.output_parsed
        if parsed is None:
            raise AnalysisError("OpenAI returned no structured result")

        expected_ids = {item.id for item in items}
        actual_ids = {assessment.news_id for assessment in parsed.assessments}
        if actual_ids != expected_ids or len(parsed.assessments) != len(items):
            raise AnalysisError(
                "OpenAI result does not contain exactly one assessment per news item"
            )
        return parsed.assessments


def _chunks(items: Sequence[NewsItem], size: int) -> Iterable[Sequence[NewsItem]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class NewsRiskAgent:
    """Orchestrates collection, filtering, model analysis and result validation."""

    def __init__(
        self,
        settings: Settings,
        *,
        feed_client: RssClient | None = None,
        analyzer: Analyzer | None = None,
    ) -> None:
        self.settings = settings
        self.feed_client = feed_client or RssClient(
            timeout=settings.request_timeout_seconds, retries=settings.retries
        )
        if analyzer is None:
            if not settings.openai_api_key:
                raise ValueError("Set NEWS_AGENT_OPENAI_API_KEY before running analysis")
            analyzer = OpenAIRiskAnalyzer(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                retries=settings.retries,
            )
        self.analyzer = analyzer

    def prefilter(self, items: Sequence[NewsItem]) -> list[NewsItem]:
        keywords = self.settings.keyword_list
        if not keywords:
            return list(items[: self.settings.max_candidates])
        return [
            item
            for item in items
            if any(keyword in item.text.casefold() for keyword in keywords)
        ][: self.settings.max_candidates]

    def run(self, url: str | None = None) -> AnalysisResult:
        source = url or str(self.settings.rss_url)
        LOGGER.info("Fetching feed", extra={"source": source})
        items = self.feed_client.fetch(source, limit=self.settings.max_feed_items)
        candidates = self.prefilter(items)
        LOGGER.info("Feed prepared", extra={"fetched": len(items), "candidates": len(candidates)})

        assessments: list[RiskAssessment] = []
        for batch in _chunks(candidates, self.settings.batch_size):
            assessments.extend(self.analyzer.analyze(batch))
        assessments = [
            item
            for item in assessments
            if item.has_risk and item.confidence >= self.settings.min_confidence
        ]
        assessments.sort(key=lambda item: item.confidence, reverse=True)
        return AnalysisResult(
            generated_at=datetime.now(UTC),
            source=source,
            fetched=len(items),
            candidates=len(candidates),
            analyzed_news=candidates,
            assessments=assessments,
        )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze an RSS feed for corporate risks")
    parser.add_argument("--url", help="RSS/Atom URL; defaults to NEWS_AGENT_RSS_URL")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        result = NewsRiskAgent(Settings()).run(args.url)
    except (ValueError, FeedError, AnalysisError) as exc:
        LOGGER.error("Agent failed: %s", exc)
        return 1
    print(result.model_dump_json(indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
