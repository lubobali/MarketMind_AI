"""Tests for Gold layer — analytics-ready aggregations.

Requirements:
- Daily summary per symbol (open, high, low, close, volume, trade_count)
- Sector performance aggregation (avg change, total volume, top gainer/loser)
- Market signals (combine price + indicators for buy/sell labels)
- DLT table config with partitioning and Z-ordering
"""

import pytest


# ── Test: Daily summary aggregation ─────────────────────
class TestDailySummary:
    """Gold must aggregate intraday records into daily summaries."""

    def test_daily_summary_fields(self):
        """Daily summary must have OHLCV + trade_count."""
        from notebooks.gold_aggregation import compute_daily_summary

        records = [
            {
                "symbol": "NVDA",
                "price": 870.0,
                "open": 868.0,
                "high": 875.0,
                "low": 865.0,
                "volume": 10_000_000,
                "timestamp": "2026-03-24T10:00:00Z",
            },
            {
                "symbol": "NVDA",
                "price": 880.0,
                "open": 870.0,
                "high": 882.0,
                "low": 869.0,
                "volume": 12_000_000,
                "timestamp": "2026-03-24T11:00:00Z",
            },
            {
                "symbol": "NVDA",
                "price": 875.0,
                "open": 880.0,
                "high": 881.0,
                "low": 873.0,
                "volume": 8_000_000,
                "timestamp": "2026-03-24T12:00:00Z",
            },
        ]

        result = compute_daily_summary(records, "NVDA", "2026-03-24")

        assert result["symbol"] == "NVDA"
        assert result["date"] == "2026-03-24"
        assert "day_open" in result
        assert "day_high" in result
        assert "day_low" in result
        assert "day_close" in result
        assert "total_volume" in result
        assert "trade_count" in result

    def test_high_is_max_of_all_highs(self):
        """Day high = max of all intraday highs."""
        from notebooks.gold_aggregation import compute_daily_summary

        records = [
            {
                "symbol": "NVDA",
                "price": 870.0,
                "open": 868.0,
                "high": 875.0,
                "low": 865.0,
                "volume": 10_000_000,
                "timestamp": "2026-03-24T10:00:00Z",
            },
            {
                "symbol": "NVDA",
                "price": 880.0,
                "open": 870.0,
                "high": 882.0,
                "low": 869.0,
                "volume": 12_000_000,
                "timestamp": "2026-03-24T11:00:00Z",
            },
        ]

        result = compute_daily_summary(records, "NVDA", "2026-03-24")

        assert result["day_high"] == 882.0

    def test_low_is_min_of_all_lows(self):
        """Day low = min of all intraday lows."""
        from notebooks.gold_aggregation import compute_daily_summary

        records = [
            {
                "symbol": "NVDA",
                "price": 870.0,
                "open": 868.0,
                "high": 875.0,
                "low": 865.0,
                "volume": 10_000_000,
                "timestamp": "2026-03-24T10:00:00Z",
            },
            {
                "symbol": "NVDA",
                "price": 880.0,
                "open": 870.0,
                "high": 882.0,
                "low": 869.0,
                "volume": 12_000_000,
                "timestamp": "2026-03-24T11:00:00Z",
            },
        ]

        result = compute_daily_summary(records, "NVDA", "2026-03-24")

        assert result["day_low"] == 865.0

    def test_open_is_first_record(self):
        """Day open = open price of the first record (by timestamp)."""
        from notebooks.gold_aggregation import compute_daily_summary

        records = [
            {
                "symbol": "NVDA",
                "price": 870.0,
                "open": 868.0,
                "high": 875.0,
                "low": 865.0,
                "volume": 10_000_000,
                "timestamp": "2026-03-24T10:00:00Z",
            },
            {
                "symbol": "NVDA",
                "price": 880.0,
                "open": 870.0,
                "high": 882.0,
                "low": 869.0,
                "volume": 12_000_000,
                "timestamp": "2026-03-24T11:00:00Z",
            },
        ]

        result = compute_daily_summary(records, "NVDA", "2026-03-24")

        assert result["day_open"] == 868.0

    def test_close_is_last_record(self):
        """Day close = price of the last record (by timestamp)."""
        from notebooks.gold_aggregation import compute_daily_summary

        records = [
            {
                "symbol": "NVDA",
                "price": 870.0,
                "open": 868.0,
                "high": 875.0,
                "low": 865.0,
                "volume": 10_000_000,
                "timestamp": "2026-03-24T10:00:00Z",
            },
            {
                "symbol": "NVDA",
                "price": 875.50,
                "open": 870.0,
                "high": 882.0,
                "low": 869.0,
                "volume": 12_000_000,
                "timestamp": "2026-03-24T11:00:00Z",
            },
        ]

        result = compute_daily_summary(records, "NVDA", "2026-03-24")

        assert result["day_close"] == 875.50

    def test_total_volume(self):
        """Total volume = sum of all record volumes."""
        from notebooks.gold_aggregation import compute_daily_summary

        records = [
            {
                "symbol": "NVDA",
                "price": 870.0,
                "open": 868.0,
                "high": 875.0,
                "low": 865.0,
                "volume": 10_000_000,
                "timestamp": "2026-03-24T10:00:00Z",
            },
            {
                "symbol": "NVDA",
                "price": 880.0,
                "open": 870.0,
                "high": 882.0,
                "low": 869.0,
                "volume": 12_000_000,
                "timestamp": "2026-03-24T11:00:00Z",
            },
        ]

        result = compute_daily_summary(records, "NVDA", "2026-03-24")

        assert result["total_volume"] == 22_000_000
        assert result["trade_count"] == 2


# ── Test: Sector performance ───────────────────────────
class TestSectorPerformance:
    """Aggregate performance by sector."""

    def test_sector_avg_change(self):
        """Average price change % across sector stocks."""
        from notebooks.gold_aggregation import compute_sector_performance

        records = [
            {"symbol": "AAPL", "price_change_pct": 1.5, "volume": 50_000_000, "sector": "Technology"},
            {"symbol": "NVDA", "price_change_pct": 3.0, "volume": 40_000_000, "sector": "Technology"},
            {"symbol": "MSFT", "price_change_pct": -0.5, "volume": 30_000_000, "sector": "Technology"},
        ]

        result = compute_sector_performance(records, "Technology")

        assert result["sector"] == "Technology"
        assert result["avg_change_pct"] == pytest.approx(4.0 / 3, abs=0.01)
        assert result["stock_count"] == 3

    def test_top_gainer_and_loser(self):
        """Identify the best and worst performers in a sector."""
        from notebooks.gold_aggregation import compute_sector_performance

        records = [
            {"symbol": "AAPL", "price_change_pct": 1.5, "volume": 50_000_000, "sector": "Technology"},
            {"symbol": "NVDA", "price_change_pct": 3.0, "volume": 40_000_000, "sector": "Technology"},
            {"symbol": "MSFT", "price_change_pct": -0.5, "volume": 30_000_000, "sector": "Technology"},
        ]

        result = compute_sector_performance(records, "Technology")

        assert result["top_gainer"] == "NVDA"
        assert result["top_loser"] == "MSFT"

    def test_total_sector_volume(self):
        """Total volume across all stocks in sector."""
        from notebooks.gold_aggregation import compute_sector_performance

        records = [
            {"symbol": "AAPL", "price_change_pct": 1.5, "volume": 50_000_000, "sector": "Technology"},
            {"symbol": "NVDA", "price_change_pct": 3.0, "volume": 40_000_000, "sector": "Technology"},
        ]

        result = compute_sector_performance(records, "Technology")

        assert result["total_volume"] == 90_000_000


# ── Test: DLT configuration ────────────────────────────
class TestGoldDLTConfig:
    def test_daily_summary_table_name(self):
        from notebooks.gold_aggregation import DAILY_SUMMARY_TABLE

        assert DAILY_SUMMARY_TABLE == "stock_daily_summary"

    def test_sector_performance_table_name(self):
        from notebooks.gold_aggregation import SECTOR_PERFORMANCE_TABLE

        assert SECTOR_PERFORMANCE_TABLE == "sector_performance"
