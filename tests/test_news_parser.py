from __future__ import annotations

from collections.abc import Sequence

import pytest

from llm_agent_news.news_parser import (
    FeedError,
    NewsItem,
    NewsRiskAgent,
    RiskAssessment,
    RiskCategory,
    Settings,
    parse_feed,
)

RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Test</title>
<item><guid>1</guid><title>Company faces bankruptcy</title><description>Risk rises</description><link>https://example.com/1</link></item>
<item><guid>1-copy</guid><title>Duplicate</title><link>https://example.com/1</link></item>
<item><guid>2</guid><title>New product launched</title><link>https://example.com/2</link></item>
</channel></rss>"""


class StubFeedClient:
    def __init__(self, items: list[NewsItem]) -> None:
        self.items = items

    def fetch(self, url: str, *, limit: int = 100) -> list[NewsItem]:
        return self.items[:limit]


class StubAnalyzer:
    def analyze(self, items: Sequence[NewsItem]) -> list[RiskAssessment]:
        return [
            RiskAssessment(
                news_id=item.id,
                has_risk=True,
                category=RiskCategory.bankruptcy,
                confidence=0.9,
                companies=["Company"],
                reason="Bankruptcy is explicitly mentioned",
            )
            for item in items
        ]


def test_parse_feed_normalizes_and_deduplicates() -> None:
    items = parse_feed(RSS)
    assert len(items) == 2
    assert items[0].title == "Company faces bankruptcy"
    assert items[0].link == "https://example.com/1"


def test_invalid_feed_raises() -> None:
    with pytest.raises(FeedError):
        parse_feed(b"not xml")


def test_agent_filters_and_analyzes_without_network() -> None:
    items = parse_feed(RSS)
    settings = Settings(
        openai_api_key=None,
        keywords="bankrupt",
        batch_size=1,
        min_confidence=0.5,
    )
    agent = NewsRiskAgent(
        settings,
        feed_client=StubFeedClient(items),  # type: ignore[arg-type]
        analyzer=StubAnalyzer(),
    )
    result = agent.run("https://example.com/rss")
    assert result.fetched == 2
    assert result.candidates == 1
    assert result.analyzed_news[0].link == "https://example.com/1"
    assert len(result.assessments) == 1
    assert result.assessments[0].category == RiskCategory.bankruptcy


def test_agent_requires_api_key_without_injected_analyzer() -> None:
    with pytest.raises(ValueError, match="NEWS_AGENT_OPENAI_API_KEY"):
        NewsRiskAgent(Settings(openai_api_key=None))
