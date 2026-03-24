"""Tests for MarketMind AI schema definitions and fixtures."""


def test_sample_stock_record_has_required_fields(sample_stock_record):
    """Stock record must have all required fields."""
    required = {"symbol", "price", "open", "high", "low", "prev_close", "volume", "market_cap", "timestamp"}
    assert required.issubset(sample_stock_record.keys())


def test_sample_stock_record_values(sample_stock_record):
    """Stock record values must be valid."""
    assert sample_stock_record["symbol"] == "NVDA"
    assert sample_stock_record["price"] > 0
    assert sample_stock_record["volume"] > 0


def test_sample_stock_batch_has_multiple_sectors(sample_stock_batch):
    """Batch should contain stocks from different sectors."""
    symbols = {r["symbol"] for r in sample_stock_batch}
    assert len(symbols) >= 3
    assert "AAPL" in symbols  # Tech
    assert "JPM" in symbols  # Finance
    assert "XOM" in symbols  # Energy


def test_sample_news_record_has_required_fields(sample_news_record):
    """News record must have all required fields."""
    required = {"headline", "summary", "source", "url", "symbols", "published_at"}
    assert required.issubset(sample_news_record.keys())


def test_sample_news_batch_has_mixed_content(sample_news_batch):
    """News batch should have varied content for testing."""
    assert len(sample_news_batch) >= 3
    # At least one should have symbols, one should be market-wide (empty symbols)
    has_symbols = [n for n in sample_news_batch if n["symbols"]]
    market_wide = [n for n in sample_news_batch if not n["symbols"]]
    assert len(has_symbols) >= 1
    assert len(market_wide) >= 1
