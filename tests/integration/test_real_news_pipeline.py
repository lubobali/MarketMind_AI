"""Integration tests — Real news fetch → NLP pipeline.

No mocks. Hits real Yahoo Finance news API, cleans text,
scores sentiment, aggregates per symbol, computes market mood.
Validates the full Phase 4 data flow end-to-end.
"""

import json

import pytest

from notebooks.news_nlp_pipeline import (
    compute_market_mood,
    compute_symbol_sentiment,
    score_article,
)
from notebooks.news_producer import fetch_news_for_symbols, write_news_to_json

pytestmark = pytest.mark.integration


# ══════════════════════════════════════════════════════════
# Real news fetch
# ══════════════════════════════════════════════════════════
class TestRealNewsFetch:
    """Fetch real news from yfinance and validate structure."""

    @pytest.fixture(autouse=True)
    def _fetch_news(self):
        """Fetch NVDA news once, reuse across tests."""
        self.articles = fetch_news_for_symbols(["NVDA"])

    def test_returns_at_least_one_article(self):
        assert len(self.articles) >= 1, "No news articles returned for NVDA"

    def test_article_has_required_keys(self):
        required = {"headline", "summary", "source", "url", "symbols", "published_at"}
        for article in self.articles:
            assert required.issubset(set(article.keys())), f"Missing keys in {article}"

    def test_headline_is_nonempty_string(self):
        for article in self.articles:
            assert isinstance(article["headline"], str)
            assert len(article["headline"]) > 0

    def test_symbols_contains_nvda(self):
        for article in self.articles:
            assert "NVDA" in article["symbols"]

    def test_published_at_is_iso_format(self):
        for article in self.articles:
            pub = article["published_at"]
            if pub:  # some may be empty
                assert "T" in pub, f"Not ISO format: {pub}"


# ══════════════════════════════════════════════════════════
# Real news → sentiment scoring
# ══════════════════════════════════════════════════════════
class TestRealNewsSentiment:
    """Score real articles and validate sentiment output."""

    @pytest.fixture(autouse=True)
    def _fetch_and_score(self):
        """Fetch → score pipeline."""
        articles = fetch_news_for_symbols(["NVDA"])
        assert len(articles) >= 1
        self.scored = [score_article(a) for a in articles]

    def test_all_articles_have_sentiment(self):
        for article in self.scored:
            assert "sentiment_score" in article
            assert "sentiment_label" in article
            assert "confidence" in article

    def test_scores_are_in_valid_range(self):
        for article in self.scored:
            assert -1.0 <= article["sentiment_score"] <= 1.0
            assert 0.0 <= article["confidence"] <= 1.0
            assert article["sentiment_label"] in ("positive", "negative", "neutral")

    def test_original_fields_preserved(self):
        for article in self.scored:
            assert "headline" in article
            assert "symbols" in article
            assert "source" in article


# ══════════════════════════════════════════════════════════
# Real news → symbol aggregation
# ══════════════════════════════════════════════════════════
class TestRealSymbolAggregation:
    """Aggregate real scored articles per symbol."""

    @pytest.fixture(autouse=True)
    def _aggregate(self):
        """Fetch → score → aggregate pipeline."""
        articles = fetch_news_for_symbols(["NVDA", "AAPL"])
        assert len(articles) >= 1
        scored = [score_article(a) for a in articles]
        self.agg = compute_symbol_sentiment(scored)

    def test_at_least_one_symbol(self):
        assert len(self.agg) >= 1

    def test_aggregation_has_required_keys(self):
        for symbol, data in self.agg.items():
            assert "avg_sentiment" in data
            assert "article_count" in data
            assert "most_positive" in data
            assert "most_negative" in data

    def test_avg_sentiment_in_range(self):
        for symbol, data in self.agg.items():
            assert -1.0 <= data["avg_sentiment"] <= 1.0

    def test_article_count_positive(self):
        for symbol, data in self.agg.items():
            assert data["article_count"] >= 1


# ══════════════════════════════════════════════════════════
# Real news → market mood
# ══════════════════════════════════════════════════════════
class TestRealMarketMood:
    """Compute market mood from real scored articles."""

    @pytest.fixture(autouse=True)
    def _compute_mood(self):
        """Full pipeline: fetch → score → mood."""
        articles = fetch_news_for_symbols(["NVDA", "AAPL", "TSLA"])
        assert len(articles) >= 1
        scored = [score_article(a) for a in articles]
        self.mood = compute_market_mood(scored)

    def test_mood_is_valid(self):
        assert self.mood["mood"] in ("bullish", "bearish", "neutral")

    def test_avg_score_in_range(self):
        assert -1.0 <= self.mood["avg_score"] <= 1.0

    def test_counts_add_up(self):
        total = self.mood["positive_count"] + self.mood["negative_count"] + self.mood["neutral_count"]
        assert total == self.mood["article_count"]

    def test_article_count_matches(self):
        assert self.mood["article_count"] >= 1


# ══════════════════════════════════════════════════════════
# Real news → JSON file write
# ══════════════════════════════════════════════════════════
class TestRealNewsJsonWrite:
    """Write real scored articles to JSON and read back."""

    def test_write_and_read_back(self, tmp_path):
        articles = fetch_news_for_symbols(["NVDA"])
        assert len(articles) >= 1

        scored = [score_article(a) for a in articles]
        filepath = write_news_to_json(scored, str(tmp_path))

        with open(filepath) as f:
            lines = f.readlines()

        assert len(lines) == len(scored)
        for line in lines:
            parsed = json.loads(line)
            assert "headline" in parsed
            assert "sentiment_score" in parsed
