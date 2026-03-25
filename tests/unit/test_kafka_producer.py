"""Tests for stock price Kafka producer.

Requirements:
- Fetch stock prices from yfinance for a list of symbols
- Format each record with required fields
- Handle API failures gracefully (no crash, log and skip)
- Support configurable symbol list and poll interval
"""

from unittest.mock import MagicMock, patch

import pytest


# ── Test: fetch_stock_prices returns correct structure ──────
class TestFetchStockPrices:
    """The producer must fetch prices and return structured records."""

    def test_returns_list_of_records(self):
        """fetch_stock_prices should return a list of dicts."""
        from notebooks.kafka_producer import fetch_stock_prices

        with patch("notebooks.kafka_producer.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = {
                "regularMarketPrice": 178.52,
                "regularMarketOpen": 178.00,
                "regularMarketDayHigh": 179.10,
                "regularMarketDayLow": 177.80,
                "regularMarketPreviousClose": 177.50,
                "regularMarketVolume": 52_100_000,
                "marketCap": 2_800_000_000_000,
            }
            mock_yf.Ticker.return_value = mock_ticker

            results = fetch_stock_prices(["AAPL"])

            assert isinstance(results, list)
            assert len(results) == 1

    def test_record_has_all_required_fields(self):
        """Each record must have symbol, price, open, high, low, prev_close, volume, market_cap, timestamp."""
        from notebooks.kafka_producer import fetch_stock_prices

        required_fields = {"symbol", "price", "open", "high", "low", "prev_close", "volume", "market_cap", "timestamp"}

        with patch("notebooks.kafka_producer.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = {
                "regularMarketPrice": 875.50,
                "regularMarketOpen": 870.00,
                "regularMarketDayHigh": 880.25,
                "regularMarketDayLow": 868.10,
                "regularMarketPreviousClose": 872.30,
                "regularMarketVolume": 45_230_100,
                "marketCap": 2_150_000_000_000,
            }
            mock_yf.Ticker.return_value = mock_ticker

            results = fetch_stock_prices(["NVDA"])
            record = results[0]

            assert required_fields.issubset(record.keys()), f"Missing fields: {required_fields - record.keys()}"

    def test_record_values_are_correct(self):
        """Record values must match what yfinance returns."""
        from notebooks.kafka_producer import fetch_stock_prices

        with patch("notebooks.kafka_producer.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = {
                "regularMarketPrice": 875.50,
                "regularMarketOpen": 870.00,
                "regularMarketDayHigh": 880.25,
                "regularMarketDayLow": 868.10,
                "regularMarketPreviousClose": 872.30,
                "regularMarketVolume": 45_230_100,
                "marketCap": 2_150_000_000_000,
            }
            mock_yf.Ticker.return_value = mock_ticker

            results = fetch_stock_prices(["NVDA"])
            record = results[0]

            assert record["symbol"] == "NVDA"
            assert record["price"] == 875.50
            assert record["open"] == 870.00
            assert record["high"] == 880.25
            assert record["low"] == 868.10
            assert record["volume"] == 45_230_100

    @pytest.mark.parametrize("symbols", [["AAPL", "MSFT", "NVDA"], ["JPM", "GS"], ["XOM"]])
    def test_fetches_multiple_symbols(self, symbols):
        """Must handle any number of symbols."""
        from notebooks.kafka_producer import fetch_stock_prices

        with patch("notebooks.kafka_producer.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = {
                "regularMarketPrice": 100.0,
                "regularMarketOpen": 99.0,
                "regularMarketDayHigh": 101.0,
                "regularMarketDayLow": 98.0,
                "regularMarketPreviousClose": 99.5,
                "regularMarketVolume": 1_000_000,
                "marketCap": 500_000_000_000,
            }
            mock_yf.Ticker.return_value = mock_ticker

            results = fetch_stock_prices(symbols)

            assert len(results) == len(symbols)
            returned_symbols = {r["symbol"] for r in results}
            assert returned_symbols == set(symbols)

    def test_handles_api_failure_gracefully(self):
        """If yfinance fails for one symbol, skip it — don't crash."""
        from notebooks.kafka_producer import fetch_stock_prices

        with patch("notebooks.kafka_producer.yf") as mock_yf:
            mock_yf.Ticker.side_effect = Exception("API timeout")

            results = fetch_stock_prices(["AAPL"])

            assert isinstance(results, list)
            assert len(results) == 0  # Failed symbol is skipped

    def test_timestamp_is_iso_format(self):
        """Timestamp must be ISO 8601 format."""
        from notebooks.kafka_producer import fetch_stock_prices

        with patch("notebooks.kafka_producer.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = {
                "regularMarketPrice": 100.0,
                "regularMarketOpen": 99.0,
                "regularMarketDayHigh": 101.0,
                "regularMarketDayLow": 98.0,
                "regularMarketPreviousClose": 99.5,
                "regularMarketVolume": 1_000_000,
                "marketCap": 500_000_000_000,
            }
            mock_yf.Ticker.return_value = mock_ticker

            results = fetch_stock_prices(["AAPL"])
            ts = results[0]["timestamp"]

            # Must parse as ISO format
            from datetime import datetime

            datetime.fromisoformat(ts.replace("Z", "+00:00"))


# ── Test: write_to_json_files for Spark readStream ──────
class TestWriteToJsonFiles:
    """Producer must write records as JSON files for Spark Structured Streaming."""

    def test_writes_json_file(self, tmp_path):
        """Must write a .json file to the output directory."""
        from notebooks.kafka_producer import write_to_json_files

        records = [
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
            }
        ]

        write_to_json_files(records, str(tmp_path))

        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1

    def test_json_file_contains_valid_records(self, tmp_path):
        """JSON file must contain parseable records."""
        import json

        from notebooks.kafka_producer import write_to_json_files

        records = [
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
                "symbol": "NVDA",
                "price": 875.50,
                "open": 870.00,
                "high": 880.25,
                "low": 868.10,
                "prev_close": 872.30,
                "volume": 45_230_100,
                "market_cap": 2_150_000_000_000,
                "timestamp": "2026-03-24T14:30:00Z",
            },
        ]

        write_to_json_files(records, str(tmp_path))

        json_files = list(tmp_path.glob("*.json"))
        with open(json_files[0]) as f:
            lines = f.readlines()

        assert len(lines) == 2
        parsed = json.loads(lines[0])
        assert parsed["symbol"] == "AAPL"
        assert parsed["price"] == 178.52
