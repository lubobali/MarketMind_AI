"""MarketMind AI — News NLP pipeline unit tests.

Tests text cleaning, per-article sentiment scoring, per-symbol
aggregation, and market mood computation.
"""

import pytest

from notebooks.news_nlp_pipeline import (
    clean_text,
    compute_market_mood,
    compute_symbol_sentiment,
    score_article,
)


# ══════════════════════════════════════════════════════════
# clean_text
# ══════════════════════════════════════════════════════════
class TestCleanText:
    """Strip HTML, normalize whitespace, handle edge cases."""

    def test_strips_html_tags(self):
        raw = "<p>NVIDIA <b>beats</b> earnings</p>"
        assert clean_text(raw) == "NVIDIA beats earnings"

    def test_strips_nested_html(self):
        raw = "<div><span>Apple <a href='x'>stock</a> rises</span></div>"
        assert clean_text(raw) == "Apple stock rises"

    def test_normalizes_whitespace(self):
        raw = "  Multiple   spaces   and\n\nnewlines\there  "
        result = clean_text(raw)
        assert "  " not in result
        assert "\n" not in result
        assert "\t" not in result
        assert result == result.strip()

    def test_decodes_html_entities(self):
        raw = "AT&amp;T rises 5% &mdash; analysts bullish"
        result = clean_text(raw)
        assert "&amp;" not in result
        assert "AT&T" in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_returns_empty(self):
        assert clean_text(None) == ""

    def test_plain_text_unchanged(self):
        text = "Tesla reports Q4 earnings"
        assert clean_text(text) == text

    def test_preserves_financial_symbols(self):
        text = "Stock up +5.2% to $178.50"
        assert clean_text(text) == text


# ══════════════════════════════════════════════════════════
# score_article
# ══════════════════════════════════════════════════════════
class TestScoreArticle:
    """Add sentiment fields to a news article."""

    def test_adds_sentiment_fields(self, sample_news_record):
        scored = score_article(sample_news_record)

        assert "sentiment_score" in scored
        assert "sentiment_label" in scored
        assert "confidence" in scored

    def test_preserves_original_fields(self, sample_news_record):
        scored = score_article(sample_news_record)

        assert scored["headline"] == sample_news_record["headline"]
        assert scored["source"] == sample_news_record["source"]
        assert scored["symbols"] == sample_news_record["symbols"]

    def test_does_not_mutate_input(self, sample_news_record):
        original_keys = set(sample_news_record.keys())
        score_article(sample_news_record)
        assert set(sample_news_record.keys()) == original_keys

    def test_positive_article_scores_positive(self):
        article = {
            "headline": "NVIDIA beats earnings, stock surges to all-time high",
            "summary": "Record revenue and strong guidance.",
            "source": "Reuters",
            "url": "https://example.com",
            "symbols": ["NVDA"],
            "published_at": "2026-03-24T13:00:00Z",
        }
        scored = score_article(article)
        assert scored["sentiment_score"] > 0
        assert scored["sentiment_label"] == "positive"

    def test_negative_article_scores_negative(self):
        article = {
            "headline": "Tesla recalls 500,000 vehicles over safety defects",
            "summary": "Major safety investigation launched by regulators.",
            "source": "Bloomberg",
            "url": "https://example.com",
            "symbols": ["TSLA"],
            "published_at": "2026-03-24T12:30:00Z",
        }
        scored = score_article(article)
        assert scored["sentiment_score"] < 0
        assert scored["sentiment_label"] == "negative"

    def test_uses_headline_and_summary_combined(self):
        """Sentiment should consider both headline and summary, not just one."""
        article_headline_only = {
            "headline": "Company reports results",
            "summary": "",
            "source": "Reuters",
            "url": "",
            "symbols": ["AAPL"],
            "published_at": "2026-03-24T13:00:00Z",
        }
        article_with_summary = {
            "headline": "Company reports results",
            "summary": "Revenue surged 200%, record profits, strong buy rating issued.",
            "source": "Reuters",
            "url": "",
            "symbols": ["AAPL"],
            "published_at": "2026-03-24T13:00:00Z",
        }
        score_no_summary = score_article(article_headline_only)["sentiment_score"]
        score_with_summary = score_article(article_with_summary)["sentiment_score"]

        # Adding a very positive summary should push score higher
        assert score_with_summary > score_no_summary

    def test_html_in_headline_gets_cleaned(self):
        article = {
            "headline": "<b>NVIDIA</b> beats earnings",
            "summary": "Strong results.",
            "source": "Reuters",
            "url": "",
            "symbols": ["NVDA"],
            "published_at": "2026-03-24T13:00:00Z",
        }
        scored = score_article(article)
        # Should not crash and should still detect positive sentiment
        assert scored["sentiment_score"] > 0


# ══════════════════════════════════════════════════════════
# compute_symbol_sentiment
# ══════════════════════════════════════════════════════════
class TestComputeSymbolSentiment:
    """Aggregate sentiment across articles for each symbol."""

    def test_single_symbol_single_article(self):
        scored_articles = [
            {
                "headline": "NVDA beats earnings",
                "symbols": ["NVDA"],
                "sentiment_score": 0.8,
                "sentiment_label": "positive",
                "confidence": 0.9,
            }
        ]
        result = compute_symbol_sentiment(scored_articles)

        assert "NVDA" in result
        assert result["NVDA"]["avg_sentiment"] == 0.8
        assert result["NVDA"]["article_count"] == 1

    def test_single_symbol_multiple_articles(self):
        scored_articles = [
            {
                "headline": "NVDA up",
                "symbols": ["NVDA"],
                "sentiment_score": 0.6,
                "sentiment_label": "positive",
                "confidence": 0.9,
            },
            {
                "headline": "NVDA mixed signals",
                "symbols": ["NVDA"],
                "sentiment_score": -0.2,
                "sentiment_label": "negative",
                "confidence": 0.7,
            },
        ]
        result = compute_symbol_sentiment(scored_articles)

        assert result["NVDA"]["article_count"] == 2
        assert result["NVDA"]["avg_sentiment"] == pytest.approx(0.2, abs=0.01)

    def test_multiple_symbols(self):
        scored_articles = [
            {
                "headline": "NVDA up",
                "symbols": ["NVDA"],
                "sentiment_score": 0.8,
                "sentiment_label": "positive",
                "confidence": 0.9,
            },
            {
                "headline": "TSLA down",
                "symbols": ["TSLA"],
                "sentiment_score": -0.5,
                "sentiment_label": "negative",
                "confidence": 0.8,
            },
        ]
        result = compute_symbol_sentiment(scored_articles)

        assert len(result) == 2
        assert result["NVDA"]["avg_sentiment"] > 0
        assert result["TSLA"]["avg_sentiment"] < 0

    def test_tracks_most_positive_and_negative(self):
        scored_articles = [
            {
                "headline": "AAPL great quarter",
                "symbols": ["AAPL"],
                "sentiment_score": 0.9,
                "sentiment_label": "positive",
                "confidence": 0.9,
            },
            {
                "headline": "AAPL minor concern",
                "symbols": ["AAPL"],
                "sentiment_score": -0.1,
                "sentiment_label": "negative",
                "confidence": 0.7,
            },
            {
                "headline": "AAPL decent day",
                "symbols": ["AAPL"],
                "sentiment_score": 0.3,
                "sentiment_label": "positive",
                "confidence": 0.8,
            },
        ]
        result = compute_symbol_sentiment(scored_articles)

        assert result["AAPL"]["most_positive"] == "AAPL great quarter"
        assert result["AAPL"]["most_negative"] == "AAPL minor concern"

    def test_empty_input_returns_empty(self):
        result = compute_symbol_sentiment([])
        assert result == {}

    def test_output_has_required_keys(self):
        scored_articles = [
            {
                "headline": "Test",
                "symbols": ["AAPL"],
                "sentiment_score": 0.5,
                "sentiment_label": "positive",
                "confidence": 0.8,
            }
        ]
        result = compute_symbol_sentiment(scored_articles)
        entry = result["AAPL"]

        required = {"avg_sentiment", "article_count", "most_positive", "most_negative"}
        assert required.issubset(set(entry.keys()))


# ══════════════════════════════════════════════════════════
# compute_market_mood
# ══════════════════════════════════════════════════════════
class TestComputeMarketMood:
    """Overall market sentiment from all scored articles."""

    def test_bullish_market(self):
        scored_articles = [
            {"sentiment_score": 0.8, "sentiment_label": "positive", "symbols": ["NVDA"]},
            {"sentiment_score": 0.6, "sentiment_label": "positive", "symbols": ["AAPL"]},
            {"sentiment_score": 0.4, "sentiment_label": "positive", "symbols": ["MSFT"]},
        ]
        mood = compute_market_mood(scored_articles)

        assert mood["mood"] == "bullish"
        assert mood["avg_score"] > 0.3
        assert mood["article_count"] == 3

    def test_bearish_market(self):
        scored_articles = [
            {"sentiment_score": -0.7, "sentiment_label": "negative", "symbols": ["TSLA"]},
            {"sentiment_score": -0.5, "sentiment_label": "negative", "symbols": ["XOM"]},
            {"sentiment_score": -0.3, "sentiment_label": "negative", "symbols": ["JPM"]},
        ]
        mood = compute_market_mood(scored_articles)

        assert mood["mood"] == "bearish"
        assert mood["avg_score"] < -0.3

    def test_neutral_market(self):
        scored_articles = [
            {"sentiment_score": 0.1, "sentiment_label": "positive", "symbols": ["AAPL"]},
            {"sentiment_score": -0.1, "sentiment_label": "negative", "symbols": ["MSFT"]},
        ]
        mood = compute_market_mood(scored_articles)

        assert mood["mood"] == "neutral"

    def test_includes_positive_and_negative_counts(self):
        scored_articles = [
            {"sentiment_score": 0.8, "sentiment_label": "positive", "symbols": ["NVDA"]},
            {"sentiment_score": 0.5, "sentiment_label": "positive", "symbols": ["AAPL"]},
            {"sentiment_score": -0.3, "sentiment_label": "negative", "symbols": ["TSLA"]},
        ]
        mood = compute_market_mood(scored_articles)

        assert mood["positive_count"] == 2
        assert mood["negative_count"] == 1
        assert mood["neutral_count"] == 0

    def test_empty_input(self):
        mood = compute_market_mood([])

        assert mood["mood"] == "neutral"
        assert mood["avg_score"] == 0.0
        assert mood["article_count"] == 0

    def test_output_has_required_keys(self):
        scored_articles = [
            {"sentiment_score": 0.5, "sentiment_label": "positive", "symbols": ["AAPL"]},
        ]
        mood = compute_market_mood(scored_articles)

        required = {"mood", "avg_score", "article_count", "positive_count", "negative_count", "neutral_count"}
        assert required.issubset(set(mood.keys()))
