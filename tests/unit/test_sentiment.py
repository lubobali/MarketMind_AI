"""MarketMind AI — Sentiment analysis unit tests.

Tests the dual-model sentiment engine (VADER + TextBlob) with
finance-specific keyword boosting.
"""

import pytest

from utils.sentiment import analyze_sentiment


# ── Output structure ─────────────────────────────────────
class TestSentimentOutputStructure:
    """Every call must return score, label, and confidence."""

    def test_returns_dict_with_required_keys(self):
        result = analyze_sentiment("NVIDIA beats earnings expectations")
        assert isinstance(result, dict)
        assert "sentiment_score" in result
        assert "sentiment_label" in result
        assert "confidence" in result

    def test_score_is_float_in_range(self):
        result = analyze_sentiment("Stock price surged today")
        assert isinstance(result["sentiment_score"], float)
        assert -1.0 <= result["sentiment_score"] <= 1.0

    def test_confidence_is_float_in_range(self):
        result = analyze_sentiment("Market closed flat")
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_label_is_valid_string(self):
        result = analyze_sentiment("Revenue increased sharply")
        assert result["sentiment_label"] in ("positive", "negative", "neutral")


# ── Positive sentiment detection ─────────────────────────
class TestPositiveSentiment:
    """Clearly positive financial text should score positive."""

    @pytest.mark.parametrize(
        "text",
        [
            "NVIDIA beats Q4 earnings expectations, revenue surges 200%",
            "Apple stock hits all-time high after strong iPhone sales",
            "Company reports record profits and raises dividend",
            "Strong buy rating issued by Goldman Sachs analysts",
        ],
    )
    def test_positive_headlines(self, text):
        result = analyze_sentiment(text)
        assert result["sentiment_score"] > 0
        assert result["sentiment_label"] == "positive"


# ── Negative sentiment detection ─────────────────────────
class TestNegativeSentiment:
    """Clearly negative financial text should score negative."""

    @pytest.mark.parametrize(
        "text",
        [
            "Tesla recalls 500,000 vehicles over safety defects",
            "Company reports massive quarterly loss, stock plunges",
            "SEC investigation launched into accounting fraud",
            "Analysts downgrade stock to sell after earnings miss",
        ],
    )
    def test_negative_headlines(self, text):
        result = analyze_sentiment(text)
        assert result["sentiment_score"] < 0
        assert result["sentiment_label"] == "negative"


# ── Neutral sentiment detection ──────────────────────────
class TestNeutralSentiment:
    """Factual / neutral text should score near zero."""

    @pytest.mark.parametrize(
        "text",
        [
            "Federal Reserve holds interest rates steady",
            "Company announces quarterly earnings date",
            "Stock market closed for holiday",
        ],
    )
    def test_neutral_headlines(self, text):
        result = analyze_sentiment(text)
        assert -0.3 <= result["sentiment_score"] <= 0.3


# ── Edge cases ───────────────────────────────────────────
class TestEdgeCases:
    """Handle bad input gracefully."""

    def test_empty_string_returns_neutral(self):
        result = analyze_sentiment("")
        assert result["sentiment_label"] == "neutral"
        assert result["sentiment_score"] == 0.0

    def test_none_returns_neutral(self):
        result = analyze_sentiment(None)
        assert result["sentiment_label"] == "neutral"
        assert result["sentiment_score"] == 0.0

    def test_whitespace_only_returns_neutral(self):
        result = analyze_sentiment("   \n\t  ")
        assert result["sentiment_label"] == "neutral"
        assert result["sentiment_score"] == 0.0

    def test_very_long_text(self):
        """Should not crash on long input."""
        long_text = "Revenue increased. " * 500
        result = analyze_sentiment(long_text)
        assert -1.0 <= result["sentiment_score"] <= 1.0
