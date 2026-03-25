"""Integration tests — Real yfinance API calls.

No mocks. Hits real Yahoo Finance API.
Validates the actual data our pipeline will consume.
"""

import json
from datetime import datetime

import pytest

from notebooks.kafka_producer import fetch_stock_prices, write_to_json_files

pytestmark = pytest.mark.integration


# ── Real API: Single stock ──────────────────────────────
class TestRealYFinanceSingleStock:
    """Fetch real NVDA data and validate everything."""

    @pytest.fixture(autouse=True)
    def _fetch_nvda(self):
        """Fetch NVDA once, reuse across all tests in this class."""
        results = fetch_stock_prices(["NVDA"])
        assert len(results) == 1, "Failed to fetch NVDA from yfinance"
        self.record = results[0]

    def test_symbol_is_correct(self):
        assert self.record["symbol"] == "NVDA"

    def test_price_is_positive(self):
        assert self.record["price"] is not None
        assert self.record["price"] > 0

    def test_volume_is_positive(self):
        assert self.record["volume"] is not None
        assert self.record["volume"] > 0

    def test_market_cap_is_reasonable(self):
        """NVDA market cap should be at least $500B."""
        assert self.record["market_cap"] is not None
        assert self.record["market_cap"] > 500_000_000_000

    def test_open_high_low_are_populated(self):
        assert self.record["open"] is not None
        assert self.record["high"] is not None
        assert self.record["low"] is not None

    def test_high_gte_low(self):
        """Day high must be >= day low."""
        assert self.record["high"] >= self.record["low"]

    def test_price_within_high_low_range(self):
        """Current price should be between day low and day high (or very close)."""
        # Allow 1% tolerance for after-hours movement
        tolerance = self.record["low"] * 0.01
        assert self.record["price"] >= self.record["low"] - tolerance
        assert self.record["price"] <= self.record["high"] + tolerance

    def test_prev_close_is_populated(self):
        assert self.record["prev_close"] is not None
        assert self.record["prev_close"] > 0

    def test_timestamp_is_valid_iso(self):
        ts = self.record["timestamp"]
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        assert parsed.year >= 2026


# ── Real API: Multiple stocks across sectors ────────────
class TestRealYFinanceMultipleStocks:
    """Fetch a batch and validate data quality across sectors."""

    SYMBOLS = ["AAPL", "NVDA", "JPM", "XOM", "PFE"]

    @pytest.fixture(autouse=True)
    def _fetch_batch(self):
        self.results = fetch_stock_prices(self.SYMBOLS)

    def test_all_symbols_returned(self):
        returned = {r["symbol"] for r in self.results}
        assert returned == set(self.SYMBOLS)

    def test_all_prices_positive(self):
        for record in self.results:
            assert record["price"] > 0, f"{record['symbol']} has invalid price: {record['price']}"

    def test_all_volumes_positive(self):
        for record in self.results:
            assert record["volume"] > 0, f"{record['symbol']} has invalid volume: {record['volume']}"

    def test_no_null_required_fields(self):
        required = ["symbol", "price", "open", "high", "low", "prev_close", "volume", "timestamp"]
        for record in self.results:
            for field in required:
                assert record[field] is not None, f"{record['symbol']} has null {field}"


# ── Real API: Invalid symbol handling ───────────────────
class TestRealYFinanceEdgeCases:
    """Producer must handle bad symbols without crashing."""

    def test_invalid_symbol_returns_empty(self):
        results = fetch_stock_prices(["ZZZZZZZZZ"])
        assert isinstance(results, list)
        assert len(results) == 0

    def test_mixed_valid_invalid_returns_valid_only(self):
        results = fetch_stock_prices(["NVDA", "ZZZZZZZZZ", "AAPL"])
        symbols = {r["symbol"] for r in results}
        assert "NVDA" in symbols
        assert "AAPL" in symbols
        assert len(results) == 2


# ── Real API → JSON file → validate format ──────────────
class TestRealEndToEnd:
    """Full pipeline: real API → JSON file → validate Spark can read it."""

    def test_real_data_to_json_file(self, tmp_path):
        """Fetch real data, write to JSON, read back and validate."""
        records = fetch_stock_prices(["NVDA", "AAPL"])
        assert len(records) >= 2

        filepath = write_to_json_files(records, str(tmp_path))

        # Read back and validate
        with open(filepath) as f:
            lines = f.readlines()

        assert len(lines) == len(records)

        for line in lines:
            parsed = json.loads(line)
            assert parsed["symbol"] in ("NVDA", "AAPL")
            assert parsed["price"] > 0
            assert parsed["volume"] > 0
            assert "timestamp" in parsed

    def test_json_file_matches_spark_schema(self, tmp_path):
        """Every field in the JSON must match what our Spark schema expects."""
        from config.schemas import STOCK_PRICE_RAW_SCHEMA

        expected_fields = {f.name for f in STOCK_PRICE_RAW_SCHEMA.fields}

        records = fetch_stock_prices(["NVDA"])
        write_to_json_files(records, str(tmp_path))

        json_files = list(tmp_path.glob("*.json"))
        with open(json_files[0]) as f:
            parsed = json.loads(f.readline())

        record_fields = set(parsed.keys())
        assert record_fields == expected_fields, (
            f"Schema mismatch: extra={record_fields - expected_fields}, missing={expected_fields - record_fields}"
        )
