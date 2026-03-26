"""MarketMind AI — News producer unit tests.

Tests the news fetcher and JSON writer for financial news data.
Mocks yfinance (external API boundary) — tests internal logic only.
"""

import json
import os

import pytest

from notebooks.news_producer import (
    fetch_news_for_symbols,
    normalize_yfinance_article,
    write_news_to_json,
)


# ── Raw yfinance article fixture ─────────────────────────
@pytest.fixture
def raw_yfinance_article():
    """Realistic yfinance news article (nested structure)."""
    return {
        "id": "abc-123",
        "content": {
            "id": "abc-123",
            "contentType": "STORY",
            "title": "NVIDIA beats Q4 earnings expectations",
            "summary": "Record revenue driven by AI chip demand.",
            "pubDate": "2026-03-24T13:00:00Z",
            "provider": {
                "displayName": "Reuters",
                "sourceId": "reuters.com",
            },
            "canonicalUrl": {
                "url": "https://finance.yahoo.com/news/nvidia-earnings.html",
            },
        },
    }


@pytest.fixture
def raw_yfinance_article_minimal():
    """Article with missing optional fields."""
    return {
        "id": "def-456",
        "content": {
            "id": "def-456",
            "contentType": "STORY",
            "title": "Market update for today",
            "summary": "",
            "pubDate": "2026-03-24T14:00:00Z",
            "provider": None,
            "canonicalUrl": None,
        },
    }


@pytest.fixture
def raw_yfinance_video():
    """Video content type — should still be normalized."""
    return {
        "id": "vid-789",
        "content": {
            "id": "vid-789",
            "contentType": "VIDEO",
            "title": "Wall Street reacts to Fed decision",
            "summary": "Analysts discuss rate implications.",
            "pubDate": "2026-03-24T15:00:00Z",
            "provider": {
                "displayName": "Yahoo Finance Video",
                "sourceId": "video.yahoofinance.com",
            },
            "canonicalUrl": {
                "url": "https://finance.yahoo.com/video/fed-decision.html",
            },
        },
    }


# ══════════════════════════════════════════════════════════
# normalize_yfinance_article
# ══════════════════════════════════════════════════════════
class TestNormalizeArticle:
    """Flatten nested yfinance structure into our clean schema."""

    def test_extracts_all_fields(self, raw_yfinance_article):
        result = normalize_yfinance_article(raw_yfinance_article, "NVDA")

        assert result["headline"] == "NVIDIA beats Q4 earnings expectations"
        assert result["summary"] == "Record revenue driven by AI chip demand."
        assert result["source"] == "Reuters"
        assert result["url"] == "https://finance.yahoo.com/news/nvidia-earnings.html"
        assert result["published_at"] == "2026-03-24T13:00:00Z"
        assert result["symbols"] == ["NVDA"]

    def test_handles_missing_provider(self, raw_yfinance_article_minimal):
        result = normalize_yfinance_article(raw_yfinance_article_minimal, "AAPL")

        assert result["headline"] == "Market update for today"
        assert result["source"] == "Unknown"
        assert result["url"] == ""
        assert result["symbols"] == ["AAPL"]

    def test_handles_missing_canonical_url(self, raw_yfinance_article_minimal):
        result = normalize_yfinance_article(raw_yfinance_article_minimal, "AAPL")
        assert result["url"] == ""

    def test_handles_video_content_type(self, raw_yfinance_video):
        result = normalize_yfinance_article(raw_yfinance_video, "SPY")

        assert result["headline"] == "Wall Street reacts to Fed decision"
        assert result["source"] == "Yahoo Finance Video"
        assert result["symbols"] == ["SPY"]

    def test_empty_summary_stays_empty(self, raw_yfinance_article_minimal):
        result = normalize_yfinance_article(raw_yfinance_article_minimal, "AAPL")
        assert result["summary"] == ""

    def test_returns_all_required_keys(self, raw_yfinance_article):
        result = normalize_yfinance_article(raw_yfinance_article, "NVDA")
        required_keys = {"headline", "summary", "source", "url", "symbols", "published_at"}
        assert set(result.keys()) == required_keys

    def test_missing_content_key_returns_none(self):
        """Malformed article with no 'content' key."""
        bad_article = {"id": "bad-1"}
        result = normalize_yfinance_article(bad_article, "AAPL")
        assert result is None

    def test_missing_title_returns_none(self):
        """Article with content but no title — skip it."""
        bad_article = {
            "id": "bad-2",
            "content": {
                "id": "bad-2",
                "summary": "Some text",
                "pubDate": "2026-03-24T13:00:00Z",
            },
        }
        result = normalize_yfinance_article(bad_article, "AAPL")
        assert result is None


# ══════════════════════════════════════════════════════════
# fetch_news_for_symbols
# ══════════════════════════════════════════════════════════
class TestFetchNewsForSymbols:
    """Fetch and normalize news across multiple symbols."""

    def test_returns_list_of_normalized_articles(self, monkeypatch, raw_yfinance_article):
        """Mock yfinance, verify we get normalized output."""

        class MockTicker:
            def __init__(self, symbol):
                self.symbol = symbol

            @property
            def news(self):
                return [raw_yfinance_article]

        monkeypatch.setattr("notebooks.news_producer.yf.Ticker", MockTicker)

        results = fetch_news_for_symbols(["NVDA"])
        assert len(results) == 1
        assert results[0]["headline"] == "NVIDIA beats Q4 earnings expectations"
        assert results[0]["symbols"] == ["NVDA"]

    def test_multiple_symbols_tagged_correctly(self, monkeypatch):
        """Each article gets the symbol it was fetched for."""

        class MockTicker:
            def __init__(self, symbol):
                self.symbol = symbol

            @property
            def news(self):
                # Each symbol returns a unique article
                return [
                    {
                        "id": f"art-{self.symbol}",
                        "content": {
                            "id": f"art-{self.symbol}",
                            "title": f"Breaking news about {self.symbol}",
                            "summary": "Details here.",
                            "pubDate": "2026-03-24T13:00:00Z",
                            "provider": {"displayName": "Reuters", "sourceId": "reuters"},
                            "canonicalUrl": {"url": "https://example.com"},
                        },
                    }
                ]

        monkeypatch.setattr("notebooks.news_producer.yf.Ticker", MockTicker)

        results = fetch_news_for_symbols(["AAPL", "MSFT"])
        assert len(results) == 2
        symbols = [r["symbols"][0] for r in results]
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    def test_deduplicates_by_headline(self, monkeypatch, raw_yfinance_article):
        """Same headline from multiple symbols should appear once."""

        class MockTicker:
            def __init__(self, symbol):
                self.symbol = symbol

            @property
            def news(self):
                return [raw_yfinance_article]

        monkeypatch.setattr("notebooks.news_producer.yf.Ticker", MockTicker)

        results = fetch_news_for_symbols(["NVDA", "AAPL"])
        # Same article for both tickers — should deduplicate
        headlines = [r["headline"] for r in results]
        assert len(set(headlines)) == len(headlines)

    def test_skips_symbol_on_api_error(self, monkeypatch):
        """If yfinance throws for one symbol, others still work."""

        call_count = 0

        class MockTicker:
            def __init__(self, symbol):
                nonlocal call_count
                call_count += 1
                self.symbol = symbol
                if symbol == "BAD":
                    raise Exception("API error")

            @property
            def news(self):
                return [
                    {
                        "id": f"art-{self.symbol}",
                        "content": {
                            "id": f"art-{self.symbol}",
                            "title": f"News about {self.symbol}",
                            "summary": "Some summary.",
                            "pubDate": "2026-03-24T13:00:00Z",
                            "provider": {"displayName": "Test", "sourceId": "test"},
                            "canonicalUrl": {"url": "https://example.com"},
                        },
                    }
                ]

        monkeypatch.setattr("notebooks.news_producer.yf.Ticker", MockTicker)

        results = fetch_news_for_symbols(["AAPL", "BAD", "MSFT"])
        assert len(results) == 2  # BAD skipped

    def test_empty_symbol_list_returns_empty(self):
        results = fetch_news_for_symbols([])
        assert results == []

    def test_symbol_with_no_news_returns_empty(self, monkeypatch):
        """Some symbols have no news articles."""

        class MockTicker:
            def __init__(self, symbol):
                self.symbol = symbol

            @property
            def news(self):
                return []

        monkeypatch.setattr("notebooks.news_producer.yf.Ticker", MockTicker)

        results = fetch_news_for_symbols(["AAPL"])
        assert results == []


# ══════════════════════════════════════════════════════════
# write_news_to_json
# ══════════════════════════════════════════════════════════
class TestWriteNewsToJson:
    """Write normalized news records as newline-delimited JSON."""

    def test_writes_valid_json_lines(self, tmp_path, sample_news_batch):
        filepath = write_news_to_json(sample_news_batch, str(tmp_path))

        assert os.path.exists(filepath)
        assert filepath.endswith(".json")

        with open(filepath) as f:
            lines = f.readlines()

        assert len(lines) == 3
        for line in lines:
            parsed = json.loads(line)
            assert "headline" in parsed

    def test_filename_contains_news_prefix(self, tmp_path, sample_news_batch):
        filepath = write_news_to_json(sample_news_batch, str(tmp_path))
        filename = os.path.basename(filepath)
        assert filename.startswith("news_")

    def test_creates_output_directory(self, tmp_path, sample_news_batch):
        new_dir = str(tmp_path / "subdir" / "deep")
        filepath = write_news_to_json(sample_news_batch, new_dir)
        assert os.path.exists(filepath)

    def test_empty_batch_writes_empty_file(self, tmp_path):
        filepath = write_news_to_json([], str(tmp_path))
        with open(filepath) as f:
            content = f.read()
        assert content == ""

    def test_preserves_unicode(self, tmp_path):
        records = [
            {
                "headline": "Märkte reagieren auf EZB-Entscheidung",
                "summary": "Die Europäische Zentralbank hält Zinsen stabil.",
                "source": "Reuters",
                "url": "https://example.com",
                "symbols": ["EWG"],
                "published_at": "2026-03-24T13:00:00Z",
            }
        ]
        filepath = write_news_to_json(records, str(tmp_path))
        with open(filepath, encoding="utf-8") as f:
            parsed = json.loads(f.readline())
        assert "Märkte" in parsed["headline"]
        assert "Europäische" in parsed["summary"]
