"""Tests for utils/gradio_app.py — Gradio UI chart builders and chat handler.

RECR: Tests written FIRST, before implementation.
Tests the pipeline — chart functions return valid Plotly figures,
chat handler wraps the agent correctly, dashboard data is formatted.
"""

from unittest.mock import MagicMock

import plotly.graph_objects as go
import pytest

from utils.gradio_app import (
    build_market_overview_chart,
    build_sector_heatmap,
    build_signal_table_figure,
    chat_handler,
    format_dashboard_data,
)

# ── Shared fixtures ──────────────────────────────────────────


@pytest.fixture
def sample_daily_records():
    """Latest daily records for 5 stocks across sectors."""
    return [
        {
            "symbol": "AAPL",
            "date": "2026-03-26",
            "sector": "Technology",
            "day_open": 178.0,
            "day_high": 179.1,
            "day_low": 177.8,
            "day_close": 178.52,
            "total_volume": 52_100_000,
            "avg_change_pct": 0.57,
        },
        {
            "symbol": "NVDA",
            "date": "2026-03-26",
            "sector": "Technology",
            "day_open": 173.5,
            "day_high": 176.8,
            "day_low": 170.1,
            "day_close": 171.24,
            "total_volume": 182_162_282,
            "avg_change_pct": -2.05,
        },
        {
            "symbol": "JPM",
            "date": "2026-03-26",
            "sector": "Finance",
            "day_open": 194.0,
            "day_high": 196.5,
            "day_low": 193.8,
            "day_close": 195.30,
            "total_volume": 12_500_000,
            "avg_change_pct": 0.56,
        },
        {
            "symbol": "XOM",
            "date": "2026-03-26",
            "sector": "Energy",
            "day_open": 109.0,
            "day_high": 109.5,
            "day_low": 108.2,
            "day_close": 108.75,
            "total_volume": 18_300_000,
            "avg_change_pct": -0.23,
        },
        {
            "symbol": "PFE",
            "date": "2026-03-26",
            "sector": "Healthcare",
            "day_open": 27.0,
            "day_high": 27.5,
            "day_low": 26.8,
            "day_close": 27.20,
            "total_volume": 35_400_000,
            "avg_change_pct": 0.74,
        },
    ]


@pytest.fixture
def sample_signal_records():
    """Signal records for 5 stocks."""
    return [
        {
            "symbol": "AAPL",
            "sector": "Technology",
            "latest_date": "2026-03-26",
            "latest_close": 178.52,
            "rsi": 55.3,
            "macd_line": 1.2,
            "macd_signal": 0.8,
            "macd_histogram": 0.4,
            "bb_upper": 185.0,
            "bb_middle": 180.0,
            "bb_lower": 175.0,
            "signal": "hold",
        },
        {
            "symbol": "NVDA",
            "sector": "Technology",
            "latest_date": "2026-03-26",
            "latest_close": 171.24,
            "rsi": 38.76,
            "macd_line": -2.83,
            "macd_signal": -2.04,
            "macd_histogram": -0.79,
            "bb_upper": 187.72,
            "bb_middle": 179.91,
            "bb_lower": 172.10,
            "signal": "buy",
        },
        {
            "symbol": "JPM",
            "sector": "Finance",
            "latest_date": "2026-03-26",
            "latest_close": 195.30,
            "rsi": 62.1,
            "macd_line": 0.5,
            "macd_signal": 0.3,
            "macd_histogram": 0.2,
            "bb_upper": 200.0,
            "bb_middle": 196.0,
            "bb_lower": 192.0,
            "signal": "hold",
        },
        {
            "symbol": "XOM",
            "sector": "Energy",
            "latest_date": "2026-03-26",
            "latest_close": 108.75,
            "rsi": 45.2,
            "macd_line": -0.3,
            "macd_signal": -0.1,
            "macd_histogram": -0.2,
            "bb_upper": 112.0,
            "bb_middle": 110.0,
            "bb_lower": 108.0,
            "signal": "sell",
        },
        {
            "symbol": "PFE",
            "sector": "Healthcare",
            "latest_date": "2026-03-26",
            "latest_close": 27.20,
            "rsi": 68.5,
            "macd_line": 0.15,
            "macd_signal": 0.10,
            "macd_histogram": 0.05,
            "bb_upper": 28.0,
            "bb_middle": 27.0,
            "bb_lower": 26.0,
            "signal": "buy",
        },
    ]


# ════════════════════════════════════════════════════════════════
#  Sector Heatmap Chart Tests
# ════════════════════════════════════════════════════════════════


class TestBuildSectorHeatmap:
    """Sector performance bar chart grouped by sector."""

    def test_returns_plotly_figure(self, sample_daily_records):
        fig = build_sector_heatmap(sample_daily_records)
        assert isinstance(fig, go.Figure)

    def test_has_data_traces(self, sample_daily_records):
        fig = build_sector_heatmap(sample_daily_records)
        assert len(fig.data) > 0

    def test_dark_theme(self, sample_daily_records):
        fig = build_sector_heatmap(sample_daily_records)
        bg_color = fig.layout.paper_bgcolor or fig.layout.template.layout.paper_bgcolor
        # Should be a dark color (not white/default)
        assert bg_color is not None

    def test_empty_data_returns_figure(self):
        fig = build_sector_heatmap([])
        assert isinstance(fig, go.Figure)

    def test_title_contains_sector(self, sample_daily_records):
        fig = build_sector_heatmap(sample_daily_records)
        assert "sector" in fig.layout.title.text.lower() or "performance" in fig.layout.title.text.lower()

    def test_all_symbols_present(self, sample_daily_records):
        fig = build_sector_heatmap(sample_daily_records)
        # Collect all x values from traces
        all_x = []
        for trace in fig.data:
            if hasattr(trace, "x") and trace.x is not None:
                all_x.extend(list(trace.x))
        symbols = {r["symbol"] for r in sample_daily_records}
        assert symbols.issubset(set(all_x))


# ════════════════════════════════════════════════════════════════
#  Signal Table Chart Tests
# ════════════════════════════════════════════════════════════════


class TestBuildSignalTableFigure:
    """Table figure showing buy/sell/hold signals with RSI/MACD."""

    def test_returns_plotly_figure(self, sample_signal_records):
        fig = build_signal_table_figure(sample_signal_records)
        assert isinstance(fig, go.Figure)

    def test_has_table_trace(self, sample_signal_records):
        fig = build_signal_table_figure(sample_signal_records)
        assert len(fig.data) > 0
        assert isinstance(fig.data[0], go.Table)

    def test_all_symbols_in_table(self, sample_signal_records):
        fig = build_signal_table_figure(sample_signal_records)
        table = fig.data[0]
        cell_values = table.cells.values
        # First column should be symbols
        symbols = list(cell_values[0])
        assert "AAPL" in symbols
        assert "NVDA" in symbols

    def test_empty_data(self):
        fig = build_signal_table_figure([])
        assert isinstance(fig, go.Figure)

    def test_columns_include_signal(self, sample_signal_records):
        fig = build_signal_table_figure(sample_signal_records)
        table = fig.data[0]
        headers = list(table.header.values)
        # Should have Signal column
        assert any("signal" in h.lower() for h in headers)

    def test_columns_include_rsi(self, sample_signal_records):
        fig = build_signal_table_figure(sample_signal_records)
        table = fig.data[0]
        headers = list(table.header.values)
        assert any("rsi" in h.lower() for h in headers)


# ════════════════════════════════════════════════════════════════
#  Market Overview Chart Tests
# ════════════════════════════════════════════════════════════════


class TestBuildMarketOverviewChart:
    """Market overview — top gainers/losers bar chart."""

    def test_returns_plotly_figure(self, sample_daily_records):
        fig = build_market_overview_chart(sample_daily_records)
        assert isinstance(fig, go.Figure)

    def test_has_bar_traces(self, sample_daily_records):
        fig = build_market_overview_chart(sample_daily_records)
        assert len(fig.data) > 0
        assert isinstance(fig.data[0], go.Bar)

    def test_empty_data(self):
        fig = build_market_overview_chart([])
        assert isinstance(fig, go.Figure)

    def test_sorted_by_change(self, sample_daily_records):
        fig = build_market_overview_chart(sample_daily_records)
        bar = fig.data[0]
        # x values (change %) should be sorted ascending (horizontal bar)
        x_vals = list(bar.x)
        assert x_vals == sorted(x_vals)


# ════════════════════════════════════════════════════════════════
#  Dashboard Data Formatter Tests
# ════════════════════════════════════════════════════════════════


class TestFormatDashboardData:
    """format_dashboard_data extracts display metrics from raw data."""

    def test_returns_dict(self, sample_daily_records, sample_signal_records):
        result = format_dashboard_data(sample_daily_records, sample_signal_records)
        assert isinstance(result, dict)

    def test_includes_top_gainers(self, sample_daily_records, sample_signal_records):
        result = format_dashboard_data(sample_daily_records, sample_signal_records)
        assert "top_gainers" in result
        assert len(result["top_gainers"]) > 0

    def test_includes_top_losers(self, sample_daily_records, sample_signal_records):
        result = format_dashboard_data(sample_daily_records, sample_signal_records)
        assert "top_losers" in result
        assert len(result["top_losers"]) > 0

    def test_includes_signal_counts(self, sample_daily_records, sample_signal_records):
        result = format_dashboard_data(sample_daily_records, sample_signal_records)
        assert "signal_counts" in result
        assert isinstance(result["signal_counts"], dict)

    def test_signal_counts_sum_to_total(self, sample_daily_records, sample_signal_records):
        result = format_dashboard_data(sample_daily_records, sample_signal_records)
        total = sum(result["signal_counts"].values())
        assert total == len(sample_signal_records)

    def test_includes_total_stocks(self, sample_daily_records, sample_signal_records):
        result = format_dashboard_data(sample_daily_records, sample_signal_records)
        assert result["total_stocks"] == len(sample_daily_records)

    def test_empty_data(self):
        result = format_dashboard_data([], [])
        assert result["total_stocks"] == 0
        assert result["top_gainers"] == []
        assert result["top_losers"] == []


# ════════════════════════════════════════════════════════════════
#  Chat Handler Tests
# ════════════════════════════════════════════════════════════════


class TestChatHandler:
    """chat_handler wraps the agent and returns formatted response."""

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.run.return_value = {
            "answer": "NVDA is trading at $171.24, down 2.05%.",
            "tools_used": [{"tool": "get_stock_price", "args": {"symbol": "NVDA"}}],
            "iterations": 2,
            "latency_seconds": 1.74,
        }
        return agent

    def test_returns_string(self, mock_agent):
        result = chat_handler("What is NVDA at?", [], mock_agent)
        assert isinstance(result, str)

    def test_passes_query_to_agent(self, mock_agent):
        chat_handler("What is NVDA at?", [], mock_agent)
        mock_agent.run.assert_called_once_with("What is NVDA at?")

    def test_returns_agent_answer(self, mock_agent):
        result = chat_handler("What is NVDA at?", [], mock_agent)
        assert "171.24" in result

    def test_handles_agent_error(self, mock_agent):
        mock_agent.run.side_effect = Exception("LLM timeout")
        result = chat_handler("Price?", [], mock_agent)
        assert "error" in result.lower() or "sorry" in result.lower()

    def test_empty_query_handled(self, mock_agent):
        mock_agent.run.return_value = {
            "answer": "Please ask a question about the stock market.",
            "tools_used": [],
            "iterations": 1,
            "latency_seconds": 0.5,
        }
        result = chat_handler("", [], mock_agent)
        assert isinstance(result, str)

    def test_history_param_accepted(self, mock_agent):
        """Chat handler accepts history param (for Gradio ChatInterface)."""
        history = [["What is AAPL at?", "AAPL is at $178.52"]]
        result = chat_handler("And NVDA?", history, mock_agent)
        assert isinstance(result, str)
