"""Tests for utils/agent_tools.py — 6 agent tools for querying market data."""

import pytest

from utils.agent_tools import (
    compare_stocks,
    get_market_sentiment,
    get_market_summary,
    get_sector_performance,
    get_stock_price,
    get_technical_signals,
)

# ── Shared test data fixtures ────────────────────────────────


@pytest.fixture
def daily_records():
    """Multiple days of daily summary records for several symbols."""
    return [
        {
            "symbol": "NVDA",
            "date": "2026-03-26",
            "sector": "Technology",
            "day_open": 173.50,
            "day_high": 176.80,
            "day_low": 170.10,
            "day_close": 171.24,
            "total_volume": 182_162_282,
            "avg_change_pct": -2.05,
        },
        {
            "symbol": "NVDA",
            "date": "2026-03-25",
            "sector": "Technology",
            "day_open": 176.00,
            "day_high": 179.50,
            "day_low": 175.20,
            "day_close": 178.68,
            "total_volume": 162_602_100,
            "avg_change_pct": 1.98,
        },
        {
            "symbol": "AAPL",
            "date": "2026-03-26",
            "sector": "Technology",
            "day_open": 251.00,
            "day_high": 254.30,
            "day_low": 250.50,
            "day_close": 252.89,
            "total_volume": 45_230_100,
            "avg_change_pct": 0.06,
        },
        {
            "symbol": "JPM",
            "date": "2026-03-26",
            "sector": "Finance",
            "day_open": 290.00,
            "day_high": 293.10,
            "day_low": 289.50,
            "day_close": 291.66,
            "total_volume": 9_800_000,
            "avg_change_pct": 0.86,
        },
    ]


# ════════════════════════════════════════════════════════════════
#  get_stock_price Tests
# ════════════════════════════════════════════════════════════════


class TestGetStockPrice:
    """Tests for get_stock_price()."""

    def test_returns_dict(self, daily_records):
        """Should return a dict."""
        result = get_stock_price(daily_records, "NVDA")
        assert isinstance(result, dict)

    def test_returns_correct_symbol(self, daily_records):
        """Should return data for the requested symbol."""
        result = get_stock_price(daily_records, "NVDA")
        assert result["symbol"] == "NVDA"

    def test_returns_latest_date(self, daily_records):
        """Should return the most recent record for the symbol."""
        result = get_stock_price(daily_records, "NVDA")
        assert result["date"] == "2026-03-26"

    def test_returns_all_price_fields(self, daily_records):
        """Should include price, open, high, low, volume, change_pct, sector."""
        result = get_stock_price(daily_records, "NVDA")
        assert result["price"] == 171.24
        assert result["open"] == 173.50
        assert result["high"] == 176.80
        assert result["low"] == 170.10
        assert result["volume"] == 182_162_282
        assert result["change_pct"] == -2.05
        assert result["sector"] == "Technology"

    def test_case_insensitive_symbol(self, daily_records):
        """Symbol lookup should be case-insensitive."""
        result = get_stock_price(daily_records, "nvda")
        assert result["symbol"] == "NVDA"

    def test_strips_whitespace(self, daily_records):
        """Should strip whitespace from symbol."""
        result = get_stock_price(daily_records, "  NVDA  ")
        assert result["symbol"] == "NVDA"

    def test_different_symbol(self, daily_records):
        """Should work for any tracked symbol."""
        result = get_stock_price(daily_records, "JPM")
        assert result["symbol"] == "JPM"
        assert result["price"] == 291.66
        assert result["sector"] == "Finance"

    def test_unknown_symbol_returns_error(self, daily_records):
        """Unknown symbol should return error dict."""
        result = get_stock_price(daily_records, "FAKE")
        assert "error" in result

    def test_empty_records_returns_error(self):
        """Empty records list should return error."""
        result = get_stock_price([], "NVDA")
        assert "error" in result

    def test_values_are_rounded(self, daily_records):
        """Numeric values should be rounded to 2 decimal places."""
        result = get_stock_price(daily_records, "AAPL")
        assert result["price"] == 252.89
        assert result["change_pct"] == 0.06


# ════════════════════════════════════════════════════════════════
#  get_sector_performance Tests
# ════════════════════════════════════════════════════════════════


@pytest.fixture
def sector_daily_records():
    """Daily records for a single sector on the same date."""
    return [
        {
            "symbol": "AAPL",
            "date": "2026-03-26",
            "sector": "Technology",
            "day_close": 252.89,
            "total_volume": 45_230_100,
            "avg_change_pct": 0.06,
        },
        {
            "symbol": "MSFT",
            "date": "2026-03-26",
            "sector": "Technology",
            "day_close": 365.97,
            "total_volume": 28_500_000,
            "avg_change_pct": -2.73,
        },
        {
            "symbol": "NVDA",
            "date": "2026-03-26",
            "sector": "Technology",
            "day_close": 171.24,
            "total_volume": 182_162_282,
            "avg_change_pct": -0.27,
        },
        {
            "symbol": "GOOGL",
            "date": "2026-03-26",
            "sector": "Technology",
            "day_close": 280.92,
            "total_volume": 33_000_000,
            "avg_change_pct": -3.89,
        },
        {
            "symbol": "AMZN",
            "date": "2026-03-26",
            "sector": "Technology",
            "day_close": 207.54,
            "total_volume": 44_500_000,
            "avg_change_pct": -1.43,
        },
    ]


@pytest.fixture
def sector_rankings():
    """Rankings for the same sector and date."""
    return [
        {"symbol": "AAPL", "date": "2026-03-26", "sector": "Technology", "sector_rank": 1, "percentile": 100.0},
        {"symbol": "NVDA", "date": "2026-03-26", "sector": "Technology", "sector_rank": 2, "percentile": 80.0},
        {"symbol": "AMZN", "date": "2026-03-26", "sector": "Technology", "sector_rank": 3, "percentile": 60.0},
        {"symbol": "MSFT", "date": "2026-03-26", "sector": "Technology", "sector_rank": 4, "percentile": 40.0},
        {"symbol": "GOOGL", "date": "2026-03-26", "sector": "Technology", "sector_rank": 5, "percentile": 20.0},
    ]


class TestGetSectorPerformance:
    """Tests for get_sector_performance()."""

    def test_returns_dict(self, sector_daily_records, sector_rankings):
        result = get_sector_performance(sector_daily_records, sector_rankings, "Technology")
        assert isinstance(result, dict)

    def test_returns_correct_sector(self, sector_daily_records, sector_rankings):
        result = get_sector_performance(sector_daily_records, sector_rankings, "Technology")
        assert result["sector"] == "Technology"

    def test_stock_count(self, sector_daily_records, sector_rankings):
        result = get_sector_performance(sector_daily_records, sector_rankings, "Technology")
        assert result["stock_count"] == 5

    def test_top_gainer_is_best_rank(self, sector_daily_records, sector_rankings):
        result = get_sector_performance(sector_daily_records, sector_rankings, "Technology")
        assert result["top_gainer"] == "AAPL"

    def test_top_loser_is_worst_rank(self, sector_daily_records, sector_rankings):
        result = get_sector_performance(sector_daily_records, sector_rankings, "Technology")
        assert result["top_loser"] == "GOOGL"

    def test_stocks_list_has_all_fields(self, sector_daily_records, sector_rankings):
        result = get_sector_performance(sector_daily_records, sector_rankings, "Technology")
        for stock in result["stocks"]:
            assert "symbol" in stock
            assert "change_pct" in stock
            assert "rank" in stock
            assert "percentile" in stock

    def test_case_insensitive_sector(self, sector_daily_records, sector_rankings):
        result = get_sector_performance(sector_daily_records, sector_rankings, "technology")
        assert result["sector"] == "Technology"

    def test_unknown_sector_returns_error(self, sector_daily_records, sector_rankings):
        result = get_sector_performance(sector_daily_records, sector_rankings, "Aerospace")
        assert "error" in result

    def test_empty_records_returns_error(self, sector_rankings):
        result = get_sector_performance([], sector_rankings, "Technology")
        assert "error" in result


# ════════════════════════════════════════════════════════════════
#  get_market_sentiment Tests
# ════════════════════════════════════════════════════════════════


@pytest.fixture
def symbol_sentiment_records():
    """Per-symbol sentiment aggregation records."""
    return [
        {
            "symbol": "NVDA",
            "avg_sentiment": 0.42,
            "article_count": 25,
            "most_positive": "NVIDIA beats Q4 earnings expectations",
            "most_negative": "NVIDIA faces export restrictions to China",
        },
        {
            "symbol": "TSLA",
            "avg_sentiment": -0.18,
            "article_count": 15,
            "most_positive": "Tesla expands Supercharger network",
            "most_negative": "Tesla recalls 500,000 vehicles",
        },
        {
            "symbol": "AAPL",
            "avg_sentiment": 0.02,
            "article_count": 10,
            "most_positive": "Apple announces new product",
            "most_negative": "Apple faces antitrust scrutiny",
        },
    ]


@pytest.fixture
def market_mood_record():
    """Market mood record."""
    return {
        "mood": "cautiously_optimistic",
        "avg_score": 0.12,
        "article_count": 131,
        "positive_count": 52,
        "negative_count": 38,
        "neutral_count": 41,
    }


class TestGetMarketSentiment:
    """Tests for get_market_sentiment()."""

    def test_symbol_returns_dict(self, symbol_sentiment_records, market_mood_record):
        result = get_market_sentiment(symbol_sentiment_records, market_mood_record, symbol="NVDA")
        assert isinstance(result, dict)

    def test_symbol_returns_correct_symbol(self, symbol_sentiment_records, market_mood_record):
        result = get_market_sentiment(symbol_sentiment_records, market_mood_record, symbol="NVDA")
        assert result["symbol"] == "NVDA"

    def test_symbol_bullish_label(self, symbol_sentiment_records, market_mood_record):
        """Positive avg_sentiment > 0.05 → bullish."""
        result = get_market_sentiment(symbol_sentiment_records, market_mood_record, symbol="NVDA")
        assert result["label"] == "bullish"

    def test_symbol_bearish_label(self, symbol_sentiment_records, market_mood_record):
        """Negative avg_sentiment < -0.05 → bearish."""
        result = get_market_sentiment(symbol_sentiment_records, market_mood_record, symbol="TSLA")
        assert result["label"] == "bearish"

    def test_symbol_neutral_label(self, symbol_sentiment_records, market_mood_record):
        """avg_sentiment near zero → neutral."""
        result = get_market_sentiment(symbol_sentiment_records, market_mood_record, symbol="AAPL")
        assert result["label"] == "neutral"

    def test_symbol_includes_article_count(self, symbol_sentiment_records, market_mood_record):
        result = get_market_sentiment(symbol_sentiment_records, market_mood_record, symbol="NVDA")
        assert result["article_count"] == 25

    def test_symbol_includes_headlines(self, symbol_sentiment_records, market_mood_record):
        result = get_market_sentiment(symbol_sentiment_records, market_mood_record, symbol="NVDA")
        assert "most_positive" in result
        assert "most_negative" in result

    def test_unknown_symbol_returns_error(self, symbol_sentiment_records, market_mood_record):
        result = get_market_sentiment(symbol_sentiment_records, market_mood_record, symbol="FAKE")
        assert "error" in result

    def test_no_symbol_returns_market_mood(self, symbol_sentiment_records, market_mood_record):
        """No symbol → return overall market mood."""
        result = get_market_sentiment(symbol_sentiment_records, market_mood_record)
        assert "mood" in result
        assert result["article_count"] == 131

    def test_market_mood_includes_counts(self, symbol_sentiment_records, market_mood_record):
        result = get_market_sentiment(symbol_sentiment_records, market_mood_record)
        assert "positive_count" in result
        assert "negative_count" in result
        assert "neutral_count" in result

    def test_none_mood_returns_error(self, symbol_sentiment_records):
        result = get_market_sentiment(symbol_sentiment_records, None)
        assert "error" in result


# ════════════════════════════════════════════════════════════════
#  get_technical_signals Tests
# ════════════════════════════════════════════════════════════════


@pytest.fixture
def signal_records():
    """Market signal records (one per symbol)."""
    return [
        {
            "symbol": "NVDA",
            "sector": "Technology",
            "latest_date": "2026-03-26",
            "latest_close": 171.24,
            "rsi": 38.76,
            "macd_line": -2.8289,
            "macd_signal": -2.0379,
            "macd_histogram": -0.791,
            "bb_upper": 187.72,
            "bb_middle": 179.91,
            "bb_lower": 172.10,
            "signal": "buy",
        },
        {
            "symbol": "XOM",
            "sector": "Energy",
            "latest_date": "2026-03-26",
            "latest_close": 165.43,
            "rsi": 71.20,
            "macd_line": 4.91,
            "macd_signal": 4.36,
            "macd_histogram": 0.55,
            "bb_upper": 166.09,
            "bb_middle": 155.84,
            "bb_lower": 145.59,
            "signal": "sell",
        },
    ]


class TestGetTechnicalSignals:
    """Tests for get_technical_signals()."""

    def test_returns_dict(self, signal_records):
        result = get_technical_signals(signal_records, "NVDA")
        assert isinstance(result, dict)

    def test_returns_correct_symbol(self, signal_records):
        result = get_technical_signals(signal_records, "NVDA")
        assert result["symbol"] == "NVDA"

    def test_includes_rsi(self, signal_records):
        result = get_technical_signals(signal_records, "NVDA")
        assert result["rsi"] == 38.76

    def test_includes_macd_nested(self, signal_records):
        result = get_technical_signals(signal_records, "NVDA")
        assert "macd" in result
        assert result["macd"]["line"] == -2.8289
        assert result["macd"]["signal"] == -2.0379
        assert result["macd"]["histogram"] == -0.791

    def test_includes_bollinger_nested(self, signal_records):
        result = get_technical_signals(signal_records, "NVDA")
        assert "bollinger" in result
        assert result["bollinger"]["upper"] == 187.72
        assert result["bollinger"]["middle"] == 179.91
        assert result["bollinger"]["lower"] == 172.10

    def test_includes_signal(self, signal_records):
        result = get_technical_signals(signal_records, "NVDA")
        assert result["signal"] == "buy"

    def test_different_symbol(self, signal_records):
        result = get_technical_signals(signal_records, "XOM")
        assert result["signal"] == "sell"
        assert result["rsi"] == 71.20

    def test_case_insensitive(self, signal_records):
        result = get_technical_signals(signal_records, "nvda")
        assert result["symbol"] == "NVDA"

    def test_unknown_symbol_returns_error(self, signal_records):
        result = get_technical_signals(signal_records, "FAKE")
        assert "error" in result

    def test_empty_records_returns_error(self):
        result = get_technical_signals([], "NVDA")
        assert "error" in result


# ════════════════════════════════════════════════════════════════
#  compare_stocks Tests
# ════════════════════════════════════════════════════════════════


@pytest.fixture
def comparison_signals():
    """Signal records for comparison."""
    return [
        {
            "symbol": "AAPL",
            "sector": "Technology",
            "latest_close": 252.89,
            "rsi": 42.71,
            "macd_histogram": -0.19,
            "signal": "hold",
        },
        {
            "symbol": "GOOGL",
            "sector": "Technology",
            "latest_close": 280.92,
            "rsi": 28.40,
            "macd_histogram": -1.76,
            "signal": "buy",
        },
        {
            "symbol": "JPM",
            "sector": "Finance",
            "latest_close": 291.66,
            "rsi": 46.11,
            "macd_histogram": 1.41,
            "signal": "buy",
        },
    ]


@pytest.fixture
def comparison_averages():
    """Moving average records for comparison."""
    return [
        {"symbol": "AAPL", "sma_5": 250.10, "sma_20": 256.15, "vwap_20": 255.50},
        {"symbol": "GOOGL", "sma_5": 285.30, "sma_20": 302.38, "vwap_20": 300.10},
        {"symbol": "JPM", "sma_5": 289.50, "sma_20": 290.89, "vwap_20": 290.00},
    ]


class TestCompareStocks:
    """Tests for compare_stocks()."""

    def test_returns_dict_with_comparison(self, comparison_signals, comparison_averages):
        result = compare_stocks(comparison_signals, comparison_averages, "AAPL", "GOOGL")
        assert isinstance(result, dict)
        assert "comparison" in result

    def test_both_symbols_present(self, comparison_signals, comparison_averages):
        result = compare_stocks(comparison_signals, comparison_averages, "AAPL", "GOOGL")
        assert "AAPL" in result["comparison"]
        assert "GOOGL" in result["comparison"]

    def test_includes_price_and_signal(self, comparison_signals, comparison_averages):
        result = compare_stocks(comparison_signals, comparison_averages, "AAPL", "GOOGL")
        aapl = result["comparison"]["AAPL"]
        assert aapl["price"] == 252.89
        assert aapl["signal"] == "hold"
        assert aapl["rsi"] == 42.71

    def test_includes_moving_averages(self, comparison_signals, comparison_averages):
        result = compare_stocks(comparison_signals, comparison_averages, "AAPL", "GOOGL")
        aapl = result["comparison"]["AAPL"]
        assert aapl["sma_5"] == 250.10
        assert aapl["sma_20"] == 256.15
        assert aapl["vwap_20"] == 255.50

    def test_cross_sector_comparison(self, comparison_signals, comparison_averages):
        result = compare_stocks(comparison_signals, comparison_averages, "AAPL", "JPM")
        assert result["comparison"]["AAPL"]["sector"] == "Technology"
        assert result["comparison"]["JPM"]["sector"] == "Finance"

    def test_case_insensitive(self, comparison_signals, comparison_averages):
        result = compare_stocks(comparison_signals, comparison_averages, "aapl", "googl")
        assert "AAPL" in result["comparison"]

    def test_unknown_symbol_returns_error(self, comparison_signals, comparison_averages):
        result = compare_stocks(comparison_signals, comparison_averages, "AAPL", "FAKE")
        assert "error" in result

    def test_both_unknown_returns_error(self, comparison_signals, comparison_averages):
        result = compare_stocks(comparison_signals, comparison_averages, "FAKE1", "FAKE2")
        assert "error" in result


# ════════════════════════════════════════════════════════════════
#  get_market_summary Tests
# ════════════════════════════════════════════════════════════════


@pytest.fixture
def summary_daily():
    """Daily records for market summary."""
    return [
        {"symbol": "XOM", "sector": "Energy", "date": "2026-03-26", "avg_change_pct": 2.64, "total_volume": 21_700_000},
        {
            "symbol": "BAC",
            "sector": "Finance",
            "date": "2026-03-26",
            "avg_change_pct": 1.30,
            "total_volume": 31_700_000,
        },
        {
            "symbol": "AAPL",
            "sector": "Technology",
            "date": "2026-03-26",
            "avg_change_pct": 0.06,
            "total_volume": 45_230_100,
        },
        {
            "symbol": "NVDA",
            "sector": "Technology",
            "date": "2026-03-26",
            "avg_change_pct": -0.27,
            "total_volume": 182_162_282,
        },
        {
            "symbol": "GOOGL",
            "sector": "Technology",
            "date": "2026-03-26",
            "avg_change_pct": -3.89,
            "total_volume": 33_000_000,
        },
    ]


@pytest.fixture
def summary_signals():
    """Signal distribution records."""
    return [
        {"symbol": "XOM", "signal": "sell"},
        {"symbol": "BAC", "signal": "buy"},
        {"symbol": "AAPL", "signal": "hold"},
        {"symbol": "NVDA", "signal": "buy"},
        {"symbol": "GOOGL", "signal": "buy"},
    ]


@pytest.fixture
def summary_spikes():
    """Volume spike records."""
    return [
        {"symbol": "NVDA", "date": "2026-03-26", "volume_ratio": 2.5, "is_spike": True},
    ]


class TestGetMarketSummary:
    """Tests for get_market_summary()."""

    def test_returns_dict(self, summary_daily, summary_signals, summary_spikes):
        result = get_market_summary(summary_daily, summary_signals, summary_spikes)
        assert isinstance(result, dict)

    def test_total_stocks(self, summary_daily, summary_signals, summary_spikes):
        result = get_market_summary(summary_daily, summary_signals, summary_spikes)
        assert result["total_stocks"] == 5

    def test_top_gainers_sorted(self, summary_daily, summary_signals, summary_spikes):
        result = get_market_summary(summary_daily, summary_signals, summary_spikes)
        assert result["top_gainers"][0]["symbol"] == "XOM"
        assert result["top_gainers"][0]["change_pct"] == 2.64

    def test_top_losers_sorted(self, summary_daily, summary_signals, summary_spikes):
        result = get_market_summary(summary_daily, summary_signals, summary_spikes)
        assert result["top_losers"][0]["symbol"] == "GOOGL"

    def test_most_active_by_volume(self, summary_daily, summary_signals, summary_spikes):
        result = get_market_summary(summary_daily, summary_signals, summary_spikes)
        assert result["most_active"][0]["symbol"] == "NVDA"

    def test_signal_distribution(self, summary_daily, summary_signals, summary_spikes):
        result = get_market_summary(summary_daily, summary_signals, summary_spikes)
        assert result["signal_distribution"]["buy"] == 3
        assert result["signal_distribution"]["sell"] == 1
        assert result["signal_distribution"]["hold"] == 1

    def test_volume_spikes(self, summary_daily, summary_signals, summary_spikes):
        result = get_market_summary(summary_daily, summary_signals, summary_spikes)
        assert len(result["volume_spikes_today"]) == 1
        assert result["volume_spikes_today"][0]["symbol"] == "NVDA"

    def test_empty_daily_returns_error(self, summary_signals, summary_spikes):
        result = get_market_summary([], summary_signals, summary_spikes)
        assert "error" in result

    def test_no_spikes(self, summary_daily, summary_signals):
        result = get_market_summary(summary_daily, summary_signals, [])
        assert result["volume_spikes_today"] == []
