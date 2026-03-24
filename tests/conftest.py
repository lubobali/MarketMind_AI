"""MarketMind AI — Shared test fixtures."""

import pytest


# ── Sample stock data for tests ─────────────────────────
@pytest.fixture
def sample_stock_record():
    """Single raw stock price record."""
    return {
        "symbol": "NVDA",
        "price": 875.50,
        "open": 870.00,
        "high": 880.25,
        "low": 868.10,
        "prev_close": 872.30,
        "volume": 45_230_100,
        "market_cap": 2_150_000_000_000,
        "timestamp": "2026-03-24T14:30:00Z",
    }


@pytest.fixture
def sample_stock_batch():
    """Batch of stock records across sectors."""
    return [
        {
            "symbol": "AAPL",
            "price": 178.52,
            "open": 178.00,
            "high": 179.10,
            "low": 177.80,
            "prev_close": 177.50,
            "volume": 52_100_000,
            "market_cap": 2_800_000_000_000,
            "timestamp": "2026-03-24T14:30:00Z",
        },
        {
            "symbol": "JPM",
            "price": 195.30,
            "open": 194.00,
            "high": 196.50,
            "low": 193.80,
            "prev_close": 194.20,
            "volume": 12_500_000,
            "market_cap": 560_000_000_000,
            "timestamp": "2026-03-24T14:30:00Z",
        },
        {
            "symbol": "XOM",
            "price": 108.75,
            "open": 109.00,
            "high": 109.50,
            "low": 108.20,
            "prev_close": 109.10,
            "volume": 18_300_000,
            "market_cap": 460_000_000_000,
            "timestamp": "2026-03-24T14:30:00Z",
        },
    ]


@pytest.fixture
def sample_news_record():
    """Single raw news record."""
    return {
        "headline": "NVIDIA beats Q4 earnings expectations, data center revenue surges",
        "summary": "NVIDIA reported record quarterly revenue driven by AI chip demand.",
        "source": "Reuters",
        "url": "https://example.com/nvda-earnings",
        "symbols": ["NVDA"],
        "published_at": "2026-03-24T13:00:00Z",
    }


@pytest.fixture
def sample_news_batch():
    """Batch of news records with mixed sentiment."""
    return [
        {
            "headline": "NVIDIA beats Q4 earnings expectations",
            "summary": "Record revenue driven by AI demand.",
            "source": "Reuters",
            "url": "https://example.com/1",
            "symbols": ["NVDA"],
            "published_at": "2026-03-24T13:00:00Z",
        },
        {
            "headline": "Tesla recalls 500,000 vehicles over safety concerns",
            "summary": "NHTSA investigation prompts major recall.",
            "source": "Bloomberg",
            "url": "https://example.com/2",
            "symbols": ["TSLA"],
            "published_at": "2026-03-24T12:30:00Z",
        },
        {
            "headline": "Federal Reserve holds interest rates steady",
            "summary": "Fed signals patience as inflation cools.",
            "source": "CNBC",
            "url": "https://example.com/3",
            "symbols": [],
            "published_at": "2026-03-24T12:00:00Z",
        },
    ]
